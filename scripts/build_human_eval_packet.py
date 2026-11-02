"""Build the blinded review packet for Deaf ASL reviewers (plan item 3b; docs/HUMAN_EVALUATION_PROTOCOL.md).

Utterances come from a plan manifest (``build_asllrp_plan_manifest.py``: English, plan, recorded motion). Every
utterance is produced by every ``--system``:

- ``LABEL=reference``: the recorded motion (video estimate) - the human upper reference through the same renderer;
- ``LABEL=plan:KEY@SHA``: a plan-conditioned generator checkpoint ``KEY`` (trained on manifest ``SHA``), conditioned on
  the utterance's plan line - the planner's gold plan, so this scores plan -> motion alone;
- ``LABEL=english:KEY@SHA``: a generator conditioned on the English sentence (the pre-plan baseline);
- ``LABEL=translate:KEY@SHA``: the full chain - the gloss planner (``--planner-run``) plans the English, the generator
  signs the predicted plan;
- ``LABEL=stitch:LEXICON_KEY@SHA``: the citation-stitching baseline (``generation.citation_stitch``) on the gold plan.

Every system signs for the recorded utterance's duration at ``--fps`` (24, the generators' training rate; the reference
is resampled to it), so length is no cue. Only utterances whose recorded hands are well tracked
(``--min-hand-coverage``) are used, so the reference is a fair upper bound (ASLLRP val/test hands are often lost). All videos are
drawn the same way: the four-panel skeleton preview plus the face panel (``face_preview``), with invalid frames of any
channel held from the nearest valid frame (for display only), so tracking dropouts do not reveal the recorded
reference.

The design (``eval_framework.human_study``) is a Latin square over reviewer groups; outputs, under ``--output-dir``
and copied to ``--output-prefix`` in the store:

- ``packet/group-<g>/sheet.csv`` + ``<item>.mp4`` + the reviewer guide (+ ``group-<g>.zip``): what group ``g``'s
  reviewers receive - no system names, no correct answers;
- ``key/key.json``: the private key (systems, correct options, grammatical truth, seed). Never sent to reviewers.

The ``test`` partition is the final study's and requires ``--final-study``; pilots use ``val``.

**How2Sign mode** (first Deaf-rater read; ``reports/rater_package_plan.md``, docs/NEXT_PHASE_PLAN.md "Rater read terms
defined"), two subcommands, D20 only:

1. ``how2sign-draw``: the eligible D20 sentences (a selected word under the official arm's selection, at least ``--min-seconds``,
   the better hand tracked on at least ``--min-hand-coverage`` of the recording, then distinct by ``round20_lib.text_key``; the
   sentences with no selected word are left out and listed), the English negation candidates
   (``human_study.negation_words``), and the draw: every eligible candidate among the ``--main`` sentences, the rest
   and the ``--pilot`` sentences at random. ``--seed`` and ``--name`` are required: the seed is declared before the draw. Writes
   ``draw.json`` (its ``name`` and ``signs`` are the sentence list of ``round15_compose_eval.py --export-sentences-file``, and it
   records the code commit), ``negation_check.csv``, the Deaf checker's sheet, which the owner fills in the session
   (``reviewer_package/CHECKER_GUIDE.md``), and ``checker_clips.csv``, the owner's private list of which recording to play. All
   three are also copied write-once to the store under ``--name``, and the seed is printed. ``how2sign-checker-sheet`` writes the
   sheet and the list for a draw that already exists, without drawing again.
2. ``how2sign-packet``: one ``--set`` (``main`` or ``pilot``) signed by every ``--system``: ``LABEL=reference`` (the
   recording) or ``LABEL=composition:INDEX_KEY@SHA`` (a ``round15_compose_eval.py --export-prefix`` index, checked for its
   frames and form; the key's timing label comes from them). The filled negation sheet fixes the truth first: only
   confirmed sentences count as negated, and the checker cannot be a rater. Distractors come from D20's other sentences
   and nothing else (``d20_distractor_pool`` refuses more; the whole-partition pool would read C20 and fresh text), and
   the key is checked for it. The sentence type and topic have no truth here and are not scored. Every system is drawn
   by the same renderer on a rebuilt clock (t0 + n/fps); item ids come from a secret source; zip times are fixed; the
   summary gives MP4 bytes per frame by system. The packet holds stage 1 only: no options. An index written by
   ``round15_compose_eval.py --export-word-format`` is a word-format arm (``check_word_format``): pilot only, report only, and
   the key says so, and they need ``@predicted`` and a corpus named in the owner's note. A pilot's options hold no negation word,
   since its sentences hold none and an option that does could be crossed out. ``--renderer`` is required. Under ``--renderer avatar``
   each item keeps the render's ``acceptance``, the key gets ``avatar_acceptance`` by arm, every ``verification.json`` stays under
   ``key/avatar-verification/``, and a mirror failure stops the build. Once a pilot packet of a draw exists, its main packet refuses
   to build without ``--excluded-raters-roster``; before that it warns.
3. ``how2sign-stage2``: when a rater returns stage 1, hash it, record the receipt and write that rater's stage 2 sheet
   (the options and the forced choice). Stage 1 is final once released. A sheet with rows unanswered needs ``--partial``.

Neither subcommand reads the generators.
"""
from __future__ import annotations

import argparse
import collections
import csv
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import random
import secrets
import shutil
import sys
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.corpus_manifest import load_manifest  # noqa: E402
from signtranslator.eval_framework.human_study import (SheetError, check_roster, decode_sheet, design_study, stage1_answers_sha256,  # noqa: E402
                                                       draw_study_sentences, key_sha256, load_roster, negation_read_first, negation_stratum,
                                                       negation_truth, negation_words, parse_negation_sheet, parse_stage1,
                                                       strata_shares, word_format_pairing, write_negation_sheet, write_sheet,
                                                       write_stage1_sheet, write_stage2_sheet)

GUIDE = ROOT / "docs" / "HUMAN_EVALUATION_REVIEWER_GUIDE.md"   # what reviewers read (the protocol stays with the team)
TWO_STAGE_GUIDE = ROOT / "docs" / "HUMAN_EVALUATION_REVIEWER_GUIDE_TWO_STAGE.md"
EVAL_MIX = "acquisition/2026-10-17-v1/youtube-asl-v1/round10-mixed-v1/manifest.json"   # the mix D20 was drawn from
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"
D20_SHA256 = "d311703b748821822f65f8e65c5e6bd81422d5e16231171f0ab1ea54e185dabe"   # the stored set (s2_lib.D20_SHA)
DRAWS = "human-eval/draws"                                                         # draw.json and its sheets, write-once by --name
H2S_PUBLIC = "acquisition/2026-10-17-v1/how2sign-public"                           # How2Sign's published CSVs and clip zips
CHECKER_CLIP_COLUMNS = ("row", "sign", "found", "clip_zip", "clip_file", "seconds", "session_video", "start_seconds", "end_seconds",
                        "realigned_start_seconds", "realigned_end_seconds")


def parse_system(spec: str) -> dict:
    label, _, rest = spec.partition("=")
    if not label or not rest:
        raise ValueError(f"--system {spec!r}: expected LABEL=reference or LABEL=<plan|english|translate>:KEY@SHA")
    if rest == "reference":
        return dict(label=label, kind="reference")
    kind, _, target = rest.partition(":")
    key, _, sha = target.rpartition("@")
    if kind not in ("plan", "english", "translate", "stitch", "composition") or not key or len(sha) != 64:
        raise ValueError(f"--system {spec!r}: expected "
                         "LABEL=<plan|english|translate|stitch|composition>:KEY@<64-hex SHA>")
    if kind == "composition":
        return dict(label=label, kind=kind, key=key, index_sha256=sha)
    return dict(label=label, kind=kind, key=key, manifest_sha256=sha)


def held_for_display(state):
    """Every channel's invalid frames filled from the nearest valid frame (forward, then backward); a channel with no
    valid frame keeps its values. For drawing only - the result is never a training or scoring input."""
    channels = {}
    for name, c in state.channels.items():
        values, valid = c.values.copy(), c.valid.copy()
        for j in range(values.shape[1]):
            ok = valid[:, j].astype(bool)
            if ok.all() or not ok.any():
                continue
            idx = np.where(ok, np.arange(len(ok)), -1)
            idx = np.maximum.accumulate(idx)                       # last valid frame at or before t
            first = int(np.argmax(ok))
            idx[idx < 0] = first
            values[:, j] = values[idx, j]
            valid[:, j] = True
        channels[name] = replace(c, values=values, valid=valid)
    return replace(state, channels=channels)


UNIFORM_VIDEO_BYTES = 8 << 20
ARM_JOINTS = ("LeftUpperArm", "LeftForearm", "RightUpperArm", "RightForearm")


def pad_video(path: Path) -> int:
    """Pad an MP4 with a top-level ``free`` box to the next multiple of 8 MiB, so a file's size says nothing about how much moved. Players skip the box; frames, timing and
    stream are untouched. Returns the size as delivered."""
    size = path.stat().st_size
    target = (size + 8 + UNIFORM_VIDEO_BYTES - 1) // UNIFORM_VIDEO_BYTES * UNIFORM_VIDEO_BYTES
    with path.open("ab") as stream:
        stream.write((target - size).to_bytes(4, "big") + b"free" + bytes(target - size - 8))
    return target


def arm_invalid_share(state) -> float:
    """The share of a state's frames with any arm joint not valid as stored: there the avatar's arm is interpolated, not the signer's."""
    body = state.channels["body"]
    joints = [body.labels.index(j) for j in ARM_JOINTS if j in body.labels]
    return float(1. - body.valid[:, joints].all(axis=1).mean()) if joints else 1.


def render_avatar_video(state, path: Path, *, resolution: int = 768, verification: Path | None = None, archive=None) -> dict:
    """The MakeHuman avatar render (``render_avatar.py``) of ``state`` to ``path``, one argument list for every system: ``dict(frames,
    acceptance, settings, commit)``, the render's own record kept at ``verification`` if named. A render that fails its mirror check would show a
    mirrored signer, so it stops here; a direction error is kept and summarised (``avatar_acceptance``), not refused.

    The renderer fills gaps and filters by itself, so nothing is held first: a held value and then a step would mark the recording. Only the
    padded video and, under the private key, its record leave here for the packet. ``archive``, (store, prefix, fields): the whole render, frames and bundle,
    goes to the store with ``INDEX.json`` before the temporary directory is deleted. A render whose settings hash is not the one for this
    resolution is refused.
    """
    import render_avatar
    with tempfile.TemporaryDirectory() as tmp:
        npz = Path(tmp) / "state.npz"
        state.save(npz)
        out = Path(tmp) / "render"
        code = render_avatar.main(["--state", str(npz), "--out-dir", str(out), "--allow-inferred",
                                   "--resolution", str(resolution)])
        if code != 0 or not (out / "avatar.mp4").exists():
            raise RuntimeError(f"avatar render failed ({code}); see {out}")
        record = json.loads((out / "verification.json").read_text())
        if record["render_settings_sha256"] != render_avatar.settings_sha256(resolution):
            raise RuntimeError("the avatar render used other settings")
        if not record["acceptance"]["mirror"]:
            raise SystemExit(f"{path.name}: the render failed its mirror check, so the signer would be shown mirrored: nothing is built on it")
        encoded_bytes = (out / "avatar.mp4").stat().st_size
        stored = render_avatar.archive_render(archive[0], archive[1], out, archive[2]) if archive else None
        shutil.copy(out / "avatar.mp4", path)
        pad_video(path)
        if verification is not None:
            verification.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(out / "verification.json", verification)
    return dict(frames=len(state.timestamps), acceptance=record["acceptance"], settings=record["render_settings_sha256"], code=record["code"],
                encoded_bytes=encoded_bytes, archive=stored)


def keep_render_identity(item: dict, rendered: dict) -> None:
    """What identifies a render on its item in the key, which is written once."""
    item.update(frames=rendered["frames"], acceptance=rendered["acceptance"], render_settings_sha256=rendered["settings"], code_commit=rendered["code"]["commit"],
                code_dirty=rendered["code"]["dirty"], encoded_bytes=rendered["encoded_bytes"], render_archive=rendered["archive"])


def refuse_mixed_renders(items) -> dict:
    """One packet, one render: every item's settings hash and code commit must agree, or the packet stops before its key is written."""
    rendered = [i for i in items if "render_settings_sha256" in i]
    settings, commits = {i["render_settings_sha256"] for i in rendered}, {(i["code_commit"], i["code_dirty"]) for i in rendered}
    if len(settings) > 1 or len(commits) > 1:
        raise SystemExit(f"this packet mixes renders: {len(settings)} settings hashes and {len(commits)} code commits. Nothing is written")
    return dict(settings_sha256=next(iter(settings)), code_commit=next(iter(commits))[0], code_dirty=next(iter(commits))[1]) if rendered else {}


def avatar_acceptance(items) -> dict:
    """The renders' acceptance by arm: clips, how many passed, and for each group of bones the clips under its limit and the worst
    median angle. Empty for a packet with no avatar render."""
    out = {}
    for item in items:
        if "acceptance" not in item:
            continue
        accepted = item["acceptance"]
        row = out.setdefault(item["system"], dict(clips=0, passed=0, mirror_passed=0, groups={}))
        row["clips"] += 1
        row["passed"] += bool(accepted["passed"])
        row["mirror_passed"] += bool(accepted["mirror"])
        for name, group in accepted["groups"].items():
            mine = row["groups"].setdefault(name, dict(passed=0, worst_median_deg=None))
            mine["passed"] += bool(group["passed"])
            if group.get("median_deg") is not None:
                mine["worst_median_deg"] = max(mine["worst_median_deg"] or 0., round(float(group["median_deg"]), 3))
    return dict(sorted(out.items()))


def render(state, path: Path, *, panel: int) -> int:
    from signtranslator.generation.canonical_skeleton import skeleton_positions
    from signtranslator.generation.face_preview import with_face_panel
    from signtranslator.generation.skeleton_preview import encode_preview, render_preview_frames
    shown = held_for_display(state)
    joints, positions, bones = skeleton_positions(shown)
    frames = with_face_panel(render_preview_frames(positions, joints, bones, panel=panel, bounds_quantile=.02), shown)
    clock = np.asarray(state.timestamps[:len(frames)]) - state.timestamps[0]
    encode_preview(frames, clock, path)
    return len(frames)


def hand_coverage(state) -> float:
    """Fraction of frames on which the better-tracked hand is valid (any joint)."""
    return max(float(state.channels[h].valid.any(axis=1).mean()) for h in ("left_hand", "right_hand"))


def choose_utterances(records, partition: str, count: int | None, min_seconds: float, rng: random.Random) -> list[dict]:
    """``count`` (all, when None: in random order) records of ``partition`` with an English sentence, a plan and a
    state, distinct sentences."""
    pool, seen = [], set()
    for r in sorted(records, key=lambda r: r["sign"]):
        english = (r.get("english") or "").strip()
        if (r.get("partition") != partition or not english or not r.get("plan") or not r.get("text")
                or (r.get("continuous") or {}).get("seconds", 0) < min_seconds):
            continue
        if english.lower() in seen:
            continue
        seen.add(english.lower()); pool.append(r)
    rng.shuffle(pool)
    if count is None:
        return pool
    if len(pool) < count:
        raise SystemExit(f"only {len(pool)} usable utterances in {partition}")
    return sorted(pool[:count], key=lambda r: r["sign"])


def write_zip(path: Path, folder: str, files) -> None:
    """A stored zip with fixed times and modes: when a clip was rendered is not in the packet."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(files):
            info = zipfile.ZipInfo(f"{folder}/{f.name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            with f.open("rb") as source, z.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target)


def write_group_files(key: dict, out: Path, store, output_prefix: str | None, *, sheet=write_sheet, guide=GUIDE) -> None:
    """Each group's sheet, guide and zip, the hashed private key, and the copy to the store, write-once."""
    for g in range(key["groups"]):
        group_dir = out / "packet" / f"group-{g}"
        (group_dir / "sheet.csv").write_text(sheet(key, g))
        shutil.copy(guide, group_dir / guide.name)
        write_zip(out / "packet" / f"group-{g}.zip", f"group-{g}", list(group_dir.iterdir()))
    key["key_sha256"] = key_sha256(key)
    (out / "key" / "key.json").write_text(json.dumps(key, indent=1, sort_keys=True) + "\n")
    if output_prefix:
        prefix = output_prefix.rstrip("/")
        files = {f"{prefix}/{f.relative_to(out).as_posix()}": f for f in sorted(p for p in out.rglob("*") if p.is_file())}
        taken = [k for k in files if store.exists(k)]
        if taken:
            raise SystemExit(f"{len(taken)} keys exist under {prefix}: written once, first {taken[0]}")
        for k, f in files.items():
            store.put(k, f.read_bytes())


# ---- How2Sign mode (module docstring) --------------------------------------------------------------------------------

def load_mix(store, mix_key: str, mix_sha256: str) -> dict:
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(mix_key) + b"\n")
        return load_manifest(path, expected_sha256=mix_sha256)


def load_d20(store, mix_key: str, mix_sha256: str, sentences_key: str, sentences_sha256: str) -> tuple[list, dict]:
    """(D20's manifest records in the set's order, the stored set), after pinning the set and refusing any other set."""
    from round20_lib import load_sentence_set
    payload = store.get(sentences_key)
    if hashlib.sha256(payload).hexdigest() != sentences_sha256:
        raise SystemExit(f"{sentences_key} does not match --sentences-sha256")
    if sentences_sha256 != D20_SHA256:
        raise SystemExit(f"the rater read draws from D20 only: {sentences_sha256} is not D20's {D20_SHA256}")
    stored = json.loads(payload)
    if stored.get("set") != "D20":
        raise SystemExit(f"the rater read draws from D20 only, not {stored.get('set')!r}")
    return load_sentence_set(stored, load_mix(store, mix_key, mix_sha256), mix_key, mix_sha256), stored


def selected_words(manifest, records) -> dict:
    """{sign: its selected words} under the official arm's selection: stage 1's, on the evaluation mix's vocabulary, from the English
    alone. A sentence with none has no signing in any composed form. No generator or state is read."""
    from round15_compose_eval import stage1_selection
    from signtranslator.generation.compositional import isolated_corpora, training_vocabulary
    vocab = training_vocabulary(manifest, isolated_corpora(manifest))
    _, words = stage1_selection([r["text"] for r in records], vocab)
    return {r["sign"]: w for r, w in zip(records, words)}


def eligible_sentences(records, state_of, *, min_seconds: float, min_hand_coverage: float, words_of=None) -> tuple[list[dict], dict]:
    """(eligible rows in the records' order, counts): a selected word (``words_of``: {sign: words}), at least ``min_seconds``, hands
    tracked, then distinct texts (the first eligible record of a text stays). ``state_of(record)`` is called only for records long
    enough."""
    from round20_lib import text_key
    rows, seen = [], set()
    counts = dict(sentences=len(records), too_short=0, hands_lost=0, repeated_text=0, **({"no_selected_word": 0} if words_of is not None else {}))
    for r in records:
        if words_of is not None and not words_of[r["sign"]]:
            counts["no_selected_word"] += 1; continue
        if (r.get("continuous") or {}).get("seconds", 0) < min_seconds:
            counts["too_short"] += 1; continue
        coverage = hand_coverage(state_of(r))
        if coverage < min_hand_coverage:
            counts["hands_lost"] += 1; continue
        text = text_key(r["text"])
        if text in seen:
            counts["repeated_text"] += 1; continue
        seen.add(text)
        rows.append(dict(sign=r["sign"], english=r["text"].strip(), seconds=r["continuous"]["seconds"],
                         hand_coverage=round(coverage, 4), negation_words=negation_words(r["text"])))
    counts.update(eligible=len(rows), negation_candidates=sum(bool(x["negation_words"]) for x in rows))
    return rows, counts


def d20_distractor_pool(d20_texts, pool=None) -> list[str]:
    """The distractor pool of a How2Sign study: D20's sentences. A pool with any other text is refused: it would read
    C20 or text outside the development set."""
    allowed = list(dict.fromkeys(d20_texts))
    if pool is None:
        return allowed
    known = set(allowed)
    beyond = sorted({t for t in pool if t not in known})
    if beyond:
        raise SystemExit(f"{len(beyond)} distractor sentences are not in D20: {beyond[:3]}")
    return list(dict.fromkeys(pool))


def without_negation(texts) -> list[str]:
    """The sentences with no negation word. A pilot holds no negation candidate, so a rater who noticed could cross out any option with
    one: 29.9% rather than 25% by elimination alone, on the plan's 16.5% of distractors. Its distractors come from these alone."""
    return [t for t in texts if not negation_words(t)]


def check_key_within_d20(key: dict, d20_texts) -> None:
    """Every option the design offers is a D20 sentence, and the study's own sentences are D20's."""
    known = set(d20_texts)
    beyond = sorted({t for i in key["items"] for t in i["options"] if t not in known})
    if beyond:
        raise SystemExit(f"{len(beyond)} offered sentences are not in D20: {beyond[:3]}")


def how2sign_key(chosen: list[dict], systems: list[str], *, seed: int, d20_texts, confirmed: dict, eligible: int,
                 pool=None) -> dict:
    """The study key of the chosen rows (``sign``, ``english``): negation truth and stratum fixed from the checker's
    answers, comprehension's weights from D20's eligible sentences, the sentence type and topic without truth,
    distractors within D20, item ids from a secret source independent of the design seed (128 bits), two stages."""
    from round20_lib import text_key
    utterances = [dict(sign=r["sign"], english=r["english"],
                       truth=dict(negated=negation_truth(r["sign"], confirmed),
                                  stratum=negation_stratum(r["sign"], confirmed))) for r in chosen]
    key = design_study(utterances, systems, seed=seed, distractor_pool=d20_distractor_pool(d20_texts, pool),
                       text_key=text_key, new_item_id=lambda: secrets.token_hex(16))
    check_key_within_d20(key, d20_texts)
    key["stages"] = 2
    key["bar"] = dict(strata=strata_shares(confirmed, eligible), eligible=eligible,
                      confirmed=sum(v == "yes" for v in confirmed.values()),
                      rejected=sum(v == "no" for v in confirmed.values()),
                      population="D20's eligible sentences (every candidate among them was checked)")
    return key


def uniform_clock(state, fps: float):
    """The state on a clock rebuilt as t0 + n/fps: the recorded clock's jitter would mark the recording."""
    t = state.timestamps
    return replace(state, timestamps=float(t[0]) + np.arange(len(t), dtype=t.dtype) / fps)


def composition_timing(row: dict, label: str) -> str:
    """The timing a composed state's own numbers show; a main form that does not fill its reference is refused."""
    if row["form"].endswith("@predicted"):
        return "predicted"
    if row["frames"] != row["reference_frames"]:
        raise SystemExit(f"{label}: {row['sign']} has {row['frames']} frames in form {row['form']}, its reference "
                         f"{row['reference_frames']}: duration would be a cue")
    return "reference-length"


def packet_timing(compositions: dict) -> tuple[str, str]:
    """(the timing label, the diagnostic) from the arms' own numbers; arms of different timings are refused."""
    timing = {k: v["timing"] for k, v in compositions.items()}
    if len(set(timing.values())) > 1:
        raise SystemExit(f"the arms have different timings: {timing}")
    label = next(iter(timing.values()), "reference-length")
    return label, "cannot clear the release bar: val signers" + (", reference-length timing" if label != "predicted" else "")


def load_composition(store, spec: dict, *, signs, list_sha256: str, sentences_sha256: str) -> dict:
    """{sign: its row} of a composed-state index, after pinning it: one arm, the draw's sentence list, D20, one form per
    sentence, and every sentence composed. A sentence with no selected word has no signing: refused, not rendered."""
    payload = store.get(spec["key"])
    if hashlib.sha256(payload).hexdigest() != spec["index_sha256"]:
        raise SystemExit(f"{spec['key']} does not match its hash")
    index = json.loads(payload)
    if index.get("kind") != "round15-composed-states":
        raise SystemExit(f"{spec['key']} is not a composed-state index")
    if (index["sentences"]["sha256"] != list_sha256
            or (index.get("sentence_set") or {}).get("sha256") != sentences_sha256):
        raise SystemExit(f"{spec['key']}: another sentence list or set")
    rows = collections.defaultdict(list)
    for row in index["states"]:
        rows[row["sign"]].append(row)
    if any(len(v) > 1 for v in rows.values()):
        raise SystemExit(f"{spec['key']} holds several forms of a sentence: export one form per arm")
    missing = sorted(s for s in signs if s not in rows)
    word_format = index.get("word_format")
    if missing and word_format:
        raise SystemExit(f"{spec['label']} is a word-format arm, report-only and pilot only: it holds the draw's pilot sentences alone")
    if missing:
        raise SystemExit(f"{spec['key']} lacks {len(missing)} sentences: {missing[:3]}")
    empty = sorted(s for s in signs if rows[s][0]["empty"])
    if empty and (word_format or {}).get("role") != "oracle":      # the oracle's: check_word_format drops them from the pilot
        raise SystemExit(f"{spec['label']}: {len(empty)} sentences have no selected word, so no signing to show: "
                         f"{empty}. The declared terms do not say how to count them: decide before building")
    chosen = {s: rows[s][0] for s in signs}
    if any(chosen[s].get(f) is None for s in signs for f in ("frames", "reference_frames", "form")):
        raise SystemExit(f"{spec['key']}: a row has no frames, reference_frames or form")
    forms = {r["form"] for r in chosen.values()}
    if len(forms) != 1:
        raise SystemExit(f"{spec['label']}: several forms in one arm: {sorted(forms)}")
    timings = {composition_timing(r, spec["label"]) for r in chosen.values()}
    return dict(arm=index["arm"], word_cache=index.get("word_cache"), rows=chosen, form=next(iter(forms)),
                timing=next(iter(timings)), word_format=word_format)


def check_word_format(systems: list[dict], compositions: dict, *, which: str, signs: list[str]) -> dict | None:
    """The pilot's word-format arms, checked before anything is rendered; None when the packet has none. Pilot only; one
    plain composition, one shuffled and one oracle of that same composition, and the reference; every sentence at the
    composition's frame count; the shuffled form holding exactly the composition's words, the oracle a part of them, from a
    corpus with the owner's recorded note. A sentence the oracle has no word for leaves the pilot, listed in the key, before
    the design is drawn."""
    from collections import Counter
    from rater_word_format import check_oracle_corpora, check_predicted_form, ORACLE, ORACLE_DEFAULT, SHUFFLED
    arms = {label: c for label, c in compositions.items() if c.get("word_format")}
    if not arms:
        return None
    if which != "pilot":
        raise SystemExit("the word-format arms are report-only and pilot only: not in the main set")
    by_role = {c["word_format"]["role"]: label for label, c in arms.items()}
    plain = [label for label in compositions if label not in arms]
    if len(arms) != 2 or set(by_role) != {SHUFFLED, ORACLE} or len(plain) != 1:
        raise SystemExit("the word-format pilot is one composition, its composition_shuffled and its oracle, and the reference")
    base = compositions[plain[0]]
    check_predicted_form(base["form"])
    reference = next(s["label"] for s in systems if s["kind"] == "reference")
    roles = dict(composition=plain[0], composition_shuffled=by_role[SHUFFLED], oracle=by_role[ORACLE], reference=reference)
    try:
        pairing = word_format_pairing([s["label"] for s in systems], roles)
    except ValueError as error:
        raise SystemExit(str(error))
    note = arms[by_role[ORACLE]]["word_format"].get("licence_note")
    corpora = sorted({w["corpus"] for r in arms[by_role[ORACLE]]["rows"].values() for w in r["words"]})
    check_oracle_corpora(corpora or ORACLE_DEFAULT, note)
    for label, arm in arms.items():
        if arm["word_format"]["composition"]["arm"] != base["arm"] or arm["word_format"]["form"] != base["form"]:
            raise SystemExit(f"{label} controls another composition than {plain[0]}")
    for sign in signs:
        row = base["rows"][sign]
        for label, arm in arms.items():
            mine = arm["rows"][sign]
            if mine["frames"] != row["frames"] or mine["composition_state_sha256"] != row["state_sha256"]:
                raise SystemExit(f"{label}: {sign} is not at its composition's frames, or from another composition")
        shuffled = Counter(w["word"] for w in arms[by_role[SHUFFLED]]["rows"][sign]["words"])
        if shuffled != Counter(w["word"] for w in row["words"]):
            raise SystemExit(f"{by_role[SHUFFLED]}: {sign} does not hold exactly the composition's words")
        if Counter(w["word"] for w in arms[by_role[ORACLE]]["rows"][sign]["words"]) - shuffled:
            raise SystemExit(f"{by_role[ORACLE]}: {sign} holds a word the composition does not")
    dropped = sorted(s for s in signs if arms[by_role[ORACLE]]["rows"][s]["empty"])
    return dict(roles=roles, report_only=True, pilot_only=True, pairing=pairing, dropped_no_real_clip=dropped,
                signs=[s for s in signs if s not in dropped], oracle_corpora=corpora, licence_note=note,
                seed=arms[by_role[SHUFFLED]]["word_format"].get("seed"))


def bytes_per_frame(items) -> dict:
    """Mean and range of MP4 bytes per frame by system: a size that differs by system could give it away."""
    by_system = collections.defaultdict(list)
    for i in items:
        by_system[i["system"]].append(i["video_bytes"] / max(i["frames"], 1))
    return {s: dict(clips=len(v), mean=round(float(np.mean(v)), 1), low=round(float(min(v)), 1),
                    high=round(float(max(v)), 1)) for s, v in sorted(by_system.items())}


def negation_checkers(text: str) -> list[str]:
    """The checkers' keys on a filled negation sheet (stored in the key: a checker cannot be a rater)."""
    rows = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    return sorted({(r["checker_sha256"] or "").strip() for r in rows} - {""})


def code_commit() -> dict:
    """The commit this code ran from and whether the tree held other changes: what a stored draw needs to be read later."""
    import subprocess
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()
    try:
        return dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")))
    except (OSError, subprocess.CalledProcessError):
        raise SystemExit("the draw records its code commit, and this is not a git checkout: run it from one")


def draw_prefix(name: str) -> str:
    if not name or not all(c.isalnum() or c in "._-" for c in name) or name.startswith("."):
        raise SystemExit(f"--name {name!r}: letters, digits, dot, dash and underscore only")
    return f"{DRAWS}/{name}"


def _metadata_rows(store, name: str) -> list[dict]:
    """A How2Sign CSV's rows (the published files are tab-separated), or nothing if it is not stored."""
    key = f"{H2S_PUBLIC}/{name}"
    if not store.exists(key):
        return []
    text = store.get(key).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t" if "\t" in text.partition("\n")[0] else ","))


def checker_clip_table(store, candidates, seconds_of) -> tuple[list[dict], int]:
    """(one row per candidate, how many were not found): which recording the owner plays the checker. The manifest names only the clip
    and its length, so the session recording and the time range come from How2Sign's own CSVs, by ``SENTENCE_NAME``; train and val only,
    since the held-out CSV has no business here. ``clip_file`` is a member of ``clip_zip``. A row not found says so, and the guide says
    how to find it by hand."""
    wanted = {c["sign"] for c in candidates}
    found = {}
    for split in ("train", "val"):
        for r in _metadata_rows(store, f"how2sign_{split}.csv"):
            if r["SENTENCE_NAME"] in wanted:
                found.setdefault(r["SENTENCE_NAME"], {}).update(split=split, session_video=r["VIDEO_NAME"], start=r["START"], end=r["END"])
        for r in _metadata_rows(store, f"how2sign_realigned_{split}.csv"):
            if r["SENTENCE_NAME"] in wanted:
                found.setdefault(r["SENTENCE_NAME"], {}).update(split=split, session_video=r["VIDEO_NAME"], realigned_start=r["START_REALIGNED"], realigned_end=r["END_REALIGNED"])
    rows = []
    for n, c in enumerate(candidates, start=1):
        hit = found.get(c["sign"], {})
        rows.append(dict(row=n, sign=c["sign"], found="yes" if hit else "no", clip_zip=f"{H2S_PUBLIC}/{hit['split']}_rgb_front_clips.zip" if hit else "",
                         clip_file=f"raw_videos/{c['sign']}.mp4", seconds=seconds_of(c["sign"]), session_video=hit.get("session_video", ""),
                         start_seconds=hit.get("start", ""), end_seconds=hit.get("end", ""),
                         realigned_start_seconds=hit.get("realigned_start", ""), realigned_end_seconds=hit.get("realigned_end", "")))
    return rows, sum(not hit for hit in (found.get(c["sign"]) for c in candidates))


def write_checker_clips(rows) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CHECKER_CLIP_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def store_draw_files(store, name: str, files: dict, *, same_bytes_ok: bool) -> str:
    """The draw's files under its name, written once. ``same_bytes_ok``: an equal file already there is fine (a draw being re-copied);
    anything else there is refused, and nothing is written."""
    prefix = draw_prefix(name)
    for file, data in files.items():
        key = f"{prefix}/{file}"
        if store.exists(key) and not (same_bytes_ok and store.get(key) == data):
            raise SystemExit(f"{key} exists with other content: a draw's name is written once")
    for file, data in files.items():
        key = f"{prefix}/{file}"
        if not store.exists(key):
            store.put(key, data)
    return prefix


def how2sign_checker_sheet(argv) -> int:
    parser = argparse.ArgumentParser(prog="how2sign-checker-sheet", description="the checker's sheet and the owner's list of recordings, "
                                     "for a draw that exists; it draws nothing")
    parser.add_argument("--store", default="r2")
    parser.add_argument("--draw", type=Path, required=True, help="draw.json from how2sign-draw")
    parser.add_argument("--draw-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty")
    payload = args.draw.read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.draw_sha256:
        raise SystemExit(f"{args.draw} does not match --draw-sha256")
    draw = json.loads(payload)
    if draw.get("kind") != "human-eval-how2sign-draw":
        raise SystemExit(f"{args.draw} is not a draw")
    from train_text_to_motion import make_store
    store = make_store(args.store)
    rows, missing = checker_clip_table(store, draw["candidates"], lambda sign: draw["rows"][sign]["seconds"])
    sheet, clips = write_negation_sheet(draw["candidates"]), write_checker_clips(rows)
    prefix = store_draw_files(store, draw["name"], {"draw.json": payload, "negation_check.csv": sheet.encode(), "checker_clips.csv": clips.encode()}, same_bytes_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "negation_check.csv").write_text(sheet)
    (args.output_dir / "checker_clips.csv").write_text(clips)
    print(json.dumps(dict(name=draw["name"], seed=draw["seed"], rows=len(rows), recordings_not_found=missing, store_prefix=prefix)))
    return 0


def how2sign_draw(argv) -> int:
    parser = argparse.ArgumentParser(prog="how2sign-draw", description="the D20 draw and the negation sheet")
    parser.add_argument("--store", default="r2")
    parser.add_argument("--mix-key", default=EVAL_MIX)
    parser.add_argument("--mix-sha256", default=EVAL_MIX_SHA)
    parser.add_argument("--sentences-key", required=True, help="D20's stored set")
    parser.add_argument("--sentences-sha256", required=True)
    parser.add_argument("--main", type=int, default=200)
    parser.add_argument("--pilot", type=int, default=60, help="the pilot's sentences: 60, with 8 raters (protocol section 3)")
    parser.add_argument("--min-seconds", type=float, default=1.5)
    parser.add_argument("--min-hand-coverage", type=float, default=.7)
    parser.add_argument("--seed", type=int, required=True, help="the seed declared before the draw")
    parser.add_argument("--name", required=True, help="the draw's name: its files go to the store under it, written once")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty")
    prefix = draw_prefix(args.name)
    code = code_commit()
    from train_text_to_motion import make_store
    from evaluate_text_to_motion import load_state
    store = make_store(args.store)
    if store.exists(f"{prefix}/draw.json"):
        raise SystemExit(f"{prefix}/draw.json exists: a draw's name is written once")
    records, stored = load_d20(store, args.mix_key, args.mix_sha256, args.sentences_key, args.sentences_sha256)      # the set's pin comes first
    words_of = selected_words(load_mix(store, args.mix_key, args.mix_sha256), records)
    rows, counts = eligible_sentences(
        records, lambda r: load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]),
        min_seconds=args.min_seconds, min_hand_coverage=args.min_hand_coverage, words_of=words_of)
    seed = args.seed
    try:
        draw = draw_study_sentences(rows, seed=seed, main=args.main, pilot=args.pilot)
    except ValueError as e:
        raise SystemExit(str(e))
    by_sign = {r["sign"]: r for r in rows}
    candidates = [dict(sign=r["sign"], english=r["english"], negation_words=r["negation_words"]) for r in rows
                  if r["negation_words"]]
    out = dict(kind="human-eval-how2sign-draw", name=args.name, seed=seed, code=code, mix_key=args.mix_key,
               mix_sha256=args.mix_sha256, sentences_key=args.sentences_key, sentences_sha256=args.sentences_sha256,
               rule=dict(min_seconds=args.min_seconds, min_hand_coverage=args.min_hand_coverage, main=args.main,
                         pilot=args.pilot, distinct="round20_lib.text_key", selected_word="stage 1 selection on the mix's vocabulary"),
               excluded_no_selected_word=[r["sign"] for r in records if not words_of[r["sign"]]],
               counts=counts, signs=draw["main"] + draw["pilot"], main=draw["main"], pilot=draw["pilot"],
               rows={s: by_sign[s] for s in draw["main"] + draw["pilot"]}, candidates=candidates)
    payload = json.dumps(out, indent=1, sort_keys=True).encode()
    sheet = write_negation_sheet(candidates)
    clips, missing = checker_clip_table(store, candidates, lambda sign: by_sign[sign]["seconds"])
    clip_list = write_checker_clips(clips)
    store_draw_files(store, args.name, {"draw.json": payload, "negation_check.csv": sheet.encode(), "checker_clips.csv": clip_list.encode()}, same_bytes_ok=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "draw.json").write_bytes(payload)
    (args.output_dir / "negation_check.csv").write_text(sheet)
    (args.output_dir / "checker_clips.csv").write_text(clip_list)
    print(json.dumps(dict(name=args.name, seed=seed, code=code, store_prefix=prefix, counts=counts, draw_sha256=hashlib.sha256(payload).hexdigest(),
                          main=len(draw["main"]), pilot=len(draw["pilot"]), candidates=len(candidates), recordings_not_found=missing)))
    return 0


def how2sign_packet(argv) -> int:
    parser = argparse.ArgumentParser(prog="how2sign-packet", description="the blinded packet of one drawn set")
    parser.add_argument("--store", default="r2")
    parser.add_argument("--draw", type=Path, required=True, help="draw.json from how2sign-draw")
    parser.add_argument("--draw-sha256", required=True)
    parser.add_argument("--negation-sheet", type=Path, required=True, help="the Deaf checker's filled sheet")
    parser.add_argument("--negation-sheet-sha256", required=True)
    parser.add_argument("--set", dest="which", choices=("main", "pilot"), required=True)
    parser.add_argument("--system", action="append", required=True, help="LABEL=reference | LABEL=composition:KEY@SHA")
    parser.add_argument("--seed", type=int, help="private design seed; drawn at random and kept in the key if omitted")
    parser.add_argument("--fps", type=float, default=24.)
    parser.add_argument("--panel", type=int, default=240)
    parser.add_argument("--renderer", choices=("skeleton", "avatar"), required=True, help="the study's is avatar; skeleton is for smoke runs")
    parser.add_argument("--avatar-resolution", type=int, default=768, help="one value for the whole packet, every system")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", help="store prefix (private) to copy the packet and key to")
    parser.add_argument("--excluded-raters-roster", type=Path, help="private JSON {reviewer key: group}: raters who cannot rate this "
                        "read, the pilot's roster for the main set; kept in the key and checked as the negation checker is")
    args = parser.parse_args(argv)
    systems = [parse_system(s) for s in args.system]
    if len({s["label"] for s in systems}) != len(systems):
        raise SystemExit("system labels must be distinct")
    if sum(s["kind"] == "reference" for s in systems) != 1 or any(s["kind"] not in ("reference", "composition")
                                                                 for s in systems):
        raise SystemExit("a How2Sign packet is one reference and composition systems")
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    draw_payload = args.draw.read_bytes()
    if hashlib.sha256(draw_payload).hexdigest() != args.draw_sha256:
        raise SystemExit(f"{args.draw} does not match --draw-sha256")
    draw = json.loads(draw_payload)
    sheet = args.negation_sheet.read_bytes()
    if hashlib.sha256(sheet).hexdigest() != args.negation_sheet_sha256:
        raise SystemExit(f"{args.negation_sheet} does not match --negation-sheet-sha256")
    sheet_text = decode_sheet(sheet)
    confirmed, problems = parse_negation_sheet(sheet_text, draw["candidates"])
    if problems:
        raise SystemExit(f"the negation truth is not fixed: {len(problems)} problems, first {problems[:3]}")
    checkers = negation_checkers(sheet_text)
    excluded = load_roster(args.excluded_raters_roster) if args.excluded_raters_roster else None
    from train_text_to_motion import make_store
    from evaluate_text_to_motion import at_rate, load_state
    store = make_store(args.store)
    if args.output_prefix and store.exists(f"{args.output_prefix.rstrip('/')}/key/key.json"):
        raise SystemExit(f"{args.output_prefix} already holds a key: written once")
    pilot_marker = f"{draw_prefix(draw['name'])}/pilot-packet.json"
    if args.which == "main" and not excluded:
        if store.exists(pilot_marker):
            raise SystemExit("a pilot packet of this draw exists, so pass its roster as --excluded-raters-roster: pilot raters have seen "
                             "scrambled forms and do not rate the main read")
        print("WARNING: a main packet without --excluded-raters-roster. If any pilot has run, on any store, stop and pass its roster: "
              "pilot raters do not rate the main read", file=sys.stderr, flush=True)
    records, _ = load_d20(store, draw["mix_key"], draw["mix_sha256"], draw["sentences_key"], draw["sentences_sha256"])
    d20 = {r["sign"]: r for r in records}
    signs = draw[args.which]
    if any(s not in d20 for s in signs):
        raise SystemExit("the draw names a sentence outside D20")
    compositions = {s["label"]: load_composition(store, s, signs=signs, list_sha256=args.draw_sha256,
                                                 sentences_sha256=draw["sentences_sha256"])
                    for s in systems if s["kind"] == "composition"}
    word_format = check_word_format(systems, compositions, which=args.which, signs=signs)
    if word_format:
        signs = word_format.pop("signs")
    seed = args.seed if args.seed is not None else secrets.randbits(31)
    d20_texts = [r["text"].strip() for r in records]
    key = how2sign_key([draw["rows"][s] for s in sorted(signs)], [s["label"] for s in systems], seed=seed,
                       d20_texts=d20_texts, confirmed=confirmed, eligible=draw["counts"]["eligible"],
                       pool=without_negation(d20_texts) if args.which == "pilot" else None)
    if args.which == "pilot" and any(negation_words(t) for i in key["items"] for t in i["options"]):
        raise SystemExit("a pilot sentence holds a negation word: the pilot holds no negation candidate")
    timing_label, diagnostic = packet_timing(compositions)
    key.update(study="how2sign-rater-read", set=args.which, partition="val", manifest_key=draw["mix_key"],
               manifest_sha256=draw["mix_sha256"], sentences=dict(key=draw["sentences_key"],
                                                                  sha256=draw["sentences_sha256"]),
               draw_sha256=args.draw_sha256, negation_sheet_sha256=args.negation_sheet_sha256,
               negation_confirmed=confirmed, negation_checkers=checkers, negation_read_first=negation_read_first(sheet_text),
               system_specs=systems, fps=args.fps,
               excluded_raters=sorted(excluded or ()), excluded_no_selected_word=draw.get("excluded_no_selected_word", []),
               distractors=dict(pool="D20's other sentences", without_negation_words=args.which == "pilot"),
               timing=timing_label, diagnostic=diagnostic,
               arms={k: dict(arm=v["arm"], word_cache=v["word_cache"], form=v["form"], timing=v["timing"])
                     for k, v in compositions.items()})
    if word_format:
        key.update(word_format=word_format, report_only_systems=[word_format["roles"]["composition_shuffled"],
                                                                 word_format["roles"]["oracle"]])
    spec = {s["label"]: s for s in systems}
    (out / "key").mkdir(parents=True, exist_ok=True)
    archive_prefix = f"renders/rater-packets/{draw['name']}-{args.which}"
    for n, item in enumerate(key["items"]):
        record, system = d20[item["sign"]], spec[item["system"]]
        reference = uniform_clock(at_rate(load_state(store, record["continuous"]["state_key"],
                                                     record["continuous"]["state_sha256"]), args.fps), args.fps)
        item["reference_arm_invalid_share"] = arm_invalid_share(reference)      # read by the scorer, report only: never the draw, never the bar
        if system["kind"] == "reference":
            state = reference
            item["state"] = dict(key=record["continuous"]["state_key"], sha256=record["continuous"]["state_sha256"])
        else:
            row = compositions[system["label"]]["rows"][item["sign"]]
            state = uniform_clock(at_rate(load_state(store, row["state_key"], row["state_sha256"]), args.fps), args.fps)
            if len(state.timestamps) != row["frames"] or len(reference.timestamps) != row["reference_frames"]:
                raise SystemExit(f"{item['sign']}: the index says {row['frames']} frames against "
                                 f"{row['reference_frames']}; the states have {len(state.timestamps)} and "
                                 f"{len(reference.timestamps)}")
            item["state"] = dict(key=row["state_key"], sha256=row["state_sha256"], form=row["form"], words=row["words"],
                                 overlong=row["overlong"],
                                 **{k: row[k] for k in ("order", "moved", "dropped", "same_words") if k in row})
        group_dir = out / "packet" / f"group-{item['group']}"
        group_dir.mkdir(parents=True, exist_ok=True)
        video = group_dir / f"{item['item_id']}.mp4"
        if args.renderer == "avatar":
            rendered = render_avatar_video(state, video, resolution=args.avatar_resolution,
                                           verification=out / "key" / "avatar-verification" / f"{item['item_id']}.json",
                                           archive=(store, f"{archive_prefix}/{item['item_id']}", dict(item_id=item["item_id"], state=item["state"])))
            keep_render_identity(item, rendered)
        else:
            item["frames"] = render(state, video, panel=args.panel)
        item["renderer"] = args.renderer
        item["seconds"] = round(len(state.timestamps) / args.fps, 3)
        item["reference_seconds"] = round(len(reference.timestamps) / args.fps, 3)
        item["video_bytes"] = item.get("encoded_bytes", video.stat().st_size)     # before padding: what the encoder made of the motion
        item["delivered_bytes"] = video.stat().st_size
        print(json.dumps(dict(n=n + 1, of=len(key["items"]), group=item["group"])), flush=True)
    key["video_bytes_per_frame"] = bytes_per_frame(key["items"])
    key["avatar_acceptance"] = avatar_acceptance(key["items"])
    key["render"] = refuse_mixed_renders(key["items"])
    write_group_files(key, out, store, args.output_prefix, sheet=write_stage1_sheet, guide=TWO_STAGE_GUIDE)
    if args.which == "pilot" and not store.exists(pilot_marker):
        store.put(pilot_marker, json.dumps(dict(key_sha256=key["key_sha256"], output_prefix=args.output_prefix)).encode())
    length_cue = sum(abs(i["seconds"] - i["reference_seconds"]) > 1.5 / args.fps for i in key["items"])
    print(json.dumps(dict(utterances=key["utterances"], systems=key["systems"], groups=key["groups"],
                          items=len(key["items"]), strata=key["bar"]["strata"], confirmed=key["bar"]["confirmed"],
                          timing=key["timing"], video_bytes_per_frame=key["video_bytes_per_frame"],
                          avatar_acceptance=key["avatar_acceptance"],
                          **({"word_format": key["word_format"]} if word_format else {}),
                          items_longer_or_shorter_than_their_reference=length_cue, key=str(out / "key" / "key.json"))))
    return 0


def how2sign_stage2(argv) -> int:
    parser = argparse.ArgumentParser(prog="how2sign-stage2", description="release one rater's stage 2 sheet")
    parser.add_argument("--key", type=Path, required=True, help="key/key.json")
    parser.add_argument("--roster", type=Path, required=True, help="private JSON: {reviewer key: group}")
    parser.add_argument("--stage1-sheet", type=Path, required=True, help="the rater's returned stage 1 sheet")
    parser.add_argument("--receipts", type=Path, required=True, help="private JSON: {reviewer key: stage 1 hash}; updated. A copy of each released stage 1 sits in released-stage1/ beside it")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partial", action="store_true", help="the rater stopped early: release stage 2 on the rows answered")
    args = parser.parse_args(argv)
    key = json.loads(args.key.read_text())
    recorded = key.pop("key_sha256", None)
    if recorded != key_sha256(key):
        raise SystemExit("the key does not match its recorded hash")
    if key.get("stages") != 2:
        raise SystemExit("this key is not a two-stage study")
    roster = load_roster(args.roster)
    payload = args.stage1_sheet.read_bytes()
    answers, problems = parse_stage1(decode_sheet(payload), key)
    if problems or not answers:
        raise SystemExit(f"stage 1 is not ready: {len(answers)} answers, problems {problems[:3]}")
    reviewer = answers[0]["reviewer_sha256"]
    if reviewer in set(key.get("negation_checkers") or ()):
        raise SystemExit("the Deaf negation checker cannot be a rater")
    if reviewer in set(key.get("excluded_raters") or ()):
        raise SystemExit("this key is on the study's excluded list: the pilot's raters do not rate the main read")
    groups = {i["item_id"]: i["group"] for i in key["items"]}
    if reviewer not in roster or any(groups[a["item_id"]] != roster[reviewer] for a in answers):
        raise SystemExit("this key is not on the roster for this group")
    issues = check_roster(roster, key)
    if issues:
        raise SystemExit(f"{args.roster}: {'; '.join(issues)}")
    designed = [i["item_id"] for i in key["items"] if i["group"] == roster[reviewer]]
    answered = {a["item_id"] for a in answers}
    unanswered = [i for i in designed if i not in answered]
    if unanswered and not args.partial:
        raise SystemExit(f"stage 1 has {len(unanswered)} of its {len(designed)} rows unanswered, first {unanswered[:3]}. Nothing was released, "
                         "and the sheet is untouched. Enter the missing answers and run this again, or, if the rater stopped there, pass "
                         "--partial: the answered rows then go to stage 2 and stage 1 is final")
    sha = hashlib.sha256(payload).hexdigest()
    receipts = json.loads(args.receipts.read_text()) if args.receipts.exists() else {}
    released = receipts.setdefault("answers_sha256", {})
    same_answers = stage1_answers_sha256(answers)
    kept_dir = args.receipts.parent / "released-stage1"
    if receipts.get(reviewer, sha) != sha:
        if released.get(reviewer) != same_answers:
            raise SystemExit(f"stage 2 was already released on another stage 1: stage 1 is final. The released sheet is kept in {kept_dir}")
        sha = receipts[reviewer]                         # re-saved with the same answers: the released sheet's hash stands
    else:                                                # a copy of what was released, so a later re-save can't lose the rater's rows
        kept_dir.mkdir(parents=True, exist_ok=True)
        copy = kept_dir / f"{reviewer[:16]}-{sha[:12]}.csv"
        if not copy.exists():
            copy.write_bytes(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"stage2-{reviewer[:12]}.csv"
    text = write_stage2_sheet(key, answers, sha)
    if not path.exists():
        path.write_bytes(b"\xef\xbb\xbf" + text.encode())    # the mark tells Excel it's UTF-8, for an offered sentence's non-ASCII letters
    elif path.read_bytes().decode("utf-8-sig") != text:
        raise SystemExit(f"{path} exists with other content")
    receipts[reviewer] = sha
    released[reviewer] = same_answers
    args.receipts.write_text(json.dumps(receipts, indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(stage1_sha256=sha, rows=sum(a["viewable"] == "yes" for a in answers), unanswered=len(unanswered), sheet=str(path))))
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    commands = {"how2sign-draw": how2sign_draw, "how2sign-checker-sheet": how2sign_checker_sheet, "how2sign-packet": how2sign_packet,
                "how2sign-stage2": how2sign_stage2}
    if argv and argv[0] in commands:
        try:
            return commands[argv[0]](argv[1:])
        except SheetError as error:
            raise SystemExit(str(error))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True, help="plan manifest")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--system", action="append", required=True, help="LABEL=reference | LABEL=<kind>:KEY@SHA")
    parser.add_argument("--planner-run", help="gloss planner run, for translate: systems")
    parser.add_argument("--partition", default="val")
    parser.add_argument("--final-study", action="store_true", help="required to read the test partition")
    parser.add_argument("--count", type=int, default=60, help="utterances (each signed by every system)")
    parser.add_argument("--min-seconds", type=float, default=1.5)
    parser.add_argument("--seed", type=int, help="private design seed; drawn at random and kept in the key if omitted")
    parser.add_argument("--fps", type=float, default=24., help="every system signs, and the reference is shown, at this "
                                                                 "rate (the generators' training rate)")
    parser.add_argument("--min-hand-coverage", type=float, default=.7,
                        help="keep utterances whose better-tracked hand is valid on at least this fraction of the "
                             "recording, so the reference is a fair upper bound")
    parser.add_argument("--guidance", type=float, default=2.)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--panel", type=int, default=240)
    parser.add_argument("--renderer", choices=("skeleton", "avatar"), default="skeleton",
                        help="skeleton + face panel, or the MakeHuman avatar (Blender; slower)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", help="store prefix (private) to copy the packet and key to")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.partition == "test" and not args.final_study:
        raise SystemExit("the test partition is the final study's: pass --final-study")
    systems = [parse_system(s) for s in args.system]
    if len({s["label"] for s in systems}) != len(systems):
        raise SystemExit("system labels must be distinct")
    if any(s["kind"] == "composition" for s in systems):
        raise SystemExit("composition systems belong to how2sign-packet")
    if any(s["kind"] == "translate" for s in systems) and not args.planner_run:
        raise SystemExit("translate: systems need --planner-run")
    from train_text_to_motion import make_store
    from evaluate_text_to_motion import at_rate, load_generator, load_state
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(p, expected_sha256=args.manifest_sha256)
    seed = args.seed if args.seed is not None else secrets.randbits(31)
    candidates = choose_utterances(manifest["records"], args.partition, None, args.min_seconds, random.Random(seed))
    chosen, references = [], {}
    for r in candidates:                                       # in the seeded random order, until enough are kept
        state = load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"])
        if hand_coverage(state) >= args.min_hand_coverage:
            chosen.append(r); references[r["sign"]] = at_rate(state, args.fps)
        if len(chosen) == args.count:
            break
    if len(chosen) < args.count:
        raise SystemExit(f"only {len(chosen)} utterances reach hand coverage {args.min_hand_coverage}")
    chosen.sort(key=lambda r: r["sign"])
    key = design_study([dict(sign=r["sign"], english=r["english"], markers=r["plan"]["markers"]) for r in chosen],
                       [s["label"] for s in systems], seed=seed,
                       distractor_pool=sorted({(r.get("english") or "").strip() for r in manifest["records"]
                                               if r.get("partition") == args.partition and (r.get("english") or "").strip()}))
    key.update(manifest_key=args.manifest_key, manifest_sha256=args.manifest_sha256, partition=args.partition,
               system_specs=systems, planner_run=args.planner_run, guidance=args.guidance, steps=args.steps,
               fps=args.fps, min_hand_coverage=args.min_hand_coverage)
    from signtranslator.generation.citation_stitch import load_stitcher
    samplers = {s["label"]: (load_stitcher(store, s["key"], s["manifest_sha256"]).sample if s["kind"] == "stitch" else
                             load_generator(store, s["key"], s["manifest_sha256"], args.device)[2])
                for s in systems if s["kind"] != "reference"}
    planned = {}
    if args.planner_run:
        from translate_english import load_planner, plan_sentences
        model, words, glosses, best = load_planner(store, args.planner_run, args.device)
        lines = plan_sentences(model, words, glosses, [r["english"] for r in chosen], device=args.device)
        planned = {r["sign"]: p["line"] for r, p in zip(chosen, lines)}
        key["planner_checkpoint"] = best["checkpoint_key"]
    spec = {s["label"]: s for s in systems}
    by_sign = {r["sign"]: r for r in chosen}
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    (out / "key").mkdir(parents=True, exist_ok=True)
    for n, item in enumerate(key["items"]):
        record, system = by_sign[item["sign"]], spec[item["system"]]
        reference = references[item["sign"]]
        fps = args.fps
        seconds = len(reference.timestamps) / fps
        condition = dict(plan=record["text"], english=record["english"], translate=planned.get(item["sign"]),
                         stitch=record["text"])
        if system["kind"] == "reference":
            state = reference
        else:
            text = condition[system["kind"]]
            item["condition"] = text
            state = samplers[system["label"]](text, fps, seconds, args.guidance, args.steps, seed + n)
        group_dir = out / "packet" / f"group-{item['group']}"
        group_dir.mkdir(parents=True, exist_ok=True)
        video = group_dir / f"{item['item_id']}.mp4"
        if args.renderer == "avatar":
            rendered = render_avatar_video(state, video, verification=out / "key" / "avatar-verification" / f"{item['item_id']}.json",
                                           archive=(store, f"renders/human-eval-packets/{args.manifest_sha256[:16]}/{item['item_id']}", dict(item_id=item["item_id"])))
            keep_render_identity(item, rendered)
        else:
            item["frames"] = render(state, video, panel=args.panel)
        item["renderer"] = args.renderer
        item["seconds"] = round(seconds, 3)
        print(json.dumps(dict(n=n + 1, of=len(key["items"]), group=item["group"])), flush=True)
    key["avatar_acceptance"] = avatar_acceptance(key["items"])
    key["render"] = refuse_mixed_renders(key["items"])
    write_group_files(key, out, store, args.output_prefix)
    print(json.dumps(dict(utterances=len(chosen), systems=[s["label"] for s in systems], groups=key["groups"],
                          items=len(key["items"]), key=str(out / "key" / "key.json"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
