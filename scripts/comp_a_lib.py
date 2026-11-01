"""Post-S2 workstream A (reports/post_s2_plan.md section 3: A1 to A4): the guards and the shared pieces. Nothing here trains and
nothing here declares anything.

**Sets.** Every option of these tests reads D20 only (``require_d20``): its key and SHA-256 are compared before the key is
fetched, so C20's list and the 1,000 are never read; the records then pass ``s2_lib.check_read_records`` (How2Sign val of c01,
c03, c05: never test, held-out or c00).

**Sources.** ``check_source_record`` keeps vocabulary clips to the train partition of ``ALLOWED_CORPORA``. ASL Citizen and STEM
Wiki are refused by name, and any c00, test or held-out record is refused, whichever list named it.

**Token list.** ``reports/comp_a3_tokens.json`` (``scripts/comp_a3_tokens.py``): ranked on How2Sign train texts only, committed
before any read, pinned by its hash. ``arm_of`` turns it into one arm: the first ``top`` ranks. A rank with no source stays
unsigned and is reported: nothing slides up into its place, so "top 10" means the same ten ranks however the sources fall. Every ``n't``
token maps to ``not`` in every arm, whatever its rank: negation is meaning, and ``not`` is already an r10 entry, so this adds no clip.

**Selection.** ``apply_token_map`` runs after the unchanged round-15 selection on the unchanged vocabulary: only tokens the
selection left unmatched can change, so every r10 word, its position and its kind are exactly today's.

**Reads.** ``is_loop`` (a whitespace 3-gram repeated at least 3 times) and ``echo_rows`` (of the content entries a clip signs that
are also in its reference, the share found in the hypothesis, on BLEU's own tokens) are the definitions behind
reports/post_s2_plan.md section 1: on the stored reads they give loop share real 28.2, blank 22.8, a1 54.4, a4 46.6 and echo real
13.8, a4 13.7, a1 12.0, blank 3.2. The echo's stop list holds the pronouns, so ``new_entry_echo`` reads an arm's new entries apart.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

O = "acquisition/2026-10-17-v1/youtube-asl-v1"
D20_KEY = f"{O}/round20/d20.json"
D20_SHA = "d311703b748821822f65f8e65c5e6bd81422d5e16231171f0ab1ea54e185dabe"
D20_NAME = "D20"
SEGMENT_CORPUS = "asllrp-continuous-segment"
ALLOWED_CORPORA = ("asllvd-text-v1", "rit-text-r6", "wlasl-text-r6", "semlex-text-v1", "semlex-text-v2", SEGMENT_CORPUS)
FORBIDDEN_SOURCE = ("citizen", "stem")           # never a vocabulary source (task of 2026-11-01)
TOPS = (10, 30)                                  # the arms' list lengths
REAL_SCALED = "real@armscale"                    # the control: the real clip with the arm's own scales
# the stop list of the plan's echo read (scratch analysis behind reports/post_s2_plan.md, section 1.2)
ECHO_STOP = frozenset("the a an to of is are was were be been have has had do does did and or but if so that this it its in on "
                      "at for with as by from i you we they he she them us my your our their me not no just can will would"
                      .split())


# ---- guards ------------------------------------------------------------------------------------------------------------

def require_d20(key, sha) -> None:
    """Stop unless this is D20's own key and hash. Compared before anything is fetched."""
    if key != D20_KEY or sha != D20_SHA:
        raise SystemExit(f"{key}: these tests read D20 only ({D20_KEY}, SHA-256 {D20_SHA}); C20, the 1,000 and fresh clips are "
                         "refused")


def check_d20_payload(payload: dict) -> None:
    if payload.get("set") != D20_NAME:
        raise SystemExit(f"this list says it is {payload.get('set')!r}, not {D20_NAME}")


def check_source_record(record: dict) -> None:
    """A vocabulary clip: the train partition of an allowed corpus; never ASL Citizen, STEM Wiki, c00, test or held-out."""
    corpus = str(record.get("corpus"))
    if any(word in corpus.lower() for word in FORBIDDEN_SOURCE) or corpus not in ALLOWED_CORPORA:
        raise SystemExit(f"{record.get('sign')}: corpus {corpus} is not a vocabulary source here")
    if record.get("partition") != "train":
        raise SystemExit(f"{record.get('sign')}: partition {record.get('partition')}; vocabulary clips are train only")
    if record.get("signer_component") == "c00":
        raise SystemExit(f"{record.get('sign')}: How2Sign signer 9 (c00) is never read")


def parse_arm_scale(text: str) -> dict:
    """``left=0.5,right=1`` -> {"left": 0.5, "right": 1.0}: both sides named, each in (0, 1], not both 1."""
    out = {}
    for part in (p for p in text.split(",") if p):
        name, _, value = part.partition("=")
        if name not in ("left", "right") or name in out or not value:
            raise ValueError(f"--arm-scale: {part!r}")
        out[name] = float(value)
    if set(out) != {"left", "right"}:
        raise ValueError("--arm-scale: give left=F,right=F")
    if not all(0. < v <= 1. for v in out.values()):
        raise ValueError("--arm-scale: each scale in (0, 1]")
    if out["left"] == 1. and out["right"] == 1.:
        raise ValueError("--arm-scale: 1 and 1 is no scale; leave the option out")
    return out


# ---- the token list ------------------------------------------------------------------------------------------------------

def load_token_list(path, sha: str) -> dict:
    """The committed list, checked against its hash, with every listed clip through ``check_source_record``."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha:
        raise SystemExit(f"{path}: SHA-256 {digest}, expected {sha}")
    payload = json.loads(raw)
    if payload.get("kind") != "comp-a3-tokens":
        raise SystemExit(f"{path}: not a token list")
    for row in payload["tokens"]:
        for source in row.get("sources") or ():
            check_source_record(source)
    return payload


def arm_of(token_list: dict, top: int) -> dict:
    """One arm: the first ``top`` ranks. ``token_map`` {token: entry}, ``entries`` {new entry: its row (sources, typical
    seconds)}, ``dropped`` {token: why it stays unsigned}. A contraction joins only when the entry it points at is signed in this
    arm (an r10 entry, or a rank within ``top``)."""
    if top not in TOPS:
        raise SystemExit(f"--vocabulary-extra-top: one of {TOPS}")
    rows = sorted((r for r in token_list["tokens"] if r["rank"] <= top), key=lambda r: r["rank"])
    if [r["rank"] for r in rows] != list(range(1, top + 1)):
        raise SystemExit(f"the token list does not hold ranks 1 to {top}")
    entries = {r["entry"]: r for r in rows if r["status"] == "clips"}
    token_map, dropped = {r["token"]: r["entry"] for r in rows if r["status"] == "clips"}, {}
    for token in token_list.get("negation_tokens") or ():
        token_map[token] = "not"
    for r in rows:
        if r["status"] == "clips":
            continue
        if r["status"] == "alias_r10" or (r["status"] == "alias_list" and r["entry"] in entries):
            token_map[r["token"]] = r["entry"]
        elif r["token"] not in token_map:
            dropped[r["token"]] = r["status"] if r["status"] != "alias_list" else f"its entry {r['entry']} is not in this arm"
    return dict(top=top, token_map=token_map, entries=entries, dropped=dropped)


def apply_token_map(details, token_map: dict) -> tuple[list, list]:
    """(details, words) with each single token the base selection left ``unmatched`` and the map names turned into a kept word
    (kind ``exact``, ``extra`` true). Every other row is the base selection's own object, untouched."""
    from signtranslator.generation.compositional import KEPT
    out = []
    for rows in details:
        new = []
        for row in rows:
            if row["kind"] == "unmatched" and len(row["tokens"]) == 1 and row["tokens"][0] in token_map:
                new.append(dict(row, entry=token_map[row["tokens"][0]], kind="exact", extra=True))
            else:
                new.append(row)
        out.append(new)
    return out, [[r["entry"] for r in rows if r["kind"] in KEPT] for rows in out]


def segment_widening(token_list: dict) -> float | None:
    """The floor of a segment's piece the list declares (part of the arm: disclosed in its settings)."""
    return token_list.get("min_segment_seconds")


def extra_vocabulary(arm: dict) -> dict:
    """{new entry: typical seconds} of an arm."""
    return {e: float(r["typical_seconds"]) for e, r in arm["entries"].items()}


def extra_pools(arm: dict, manifest_records, gloss_records) -> tuple[dict, dict]:
    """(isolated pools, segment pools) of an arm's new entries. Isolated clips are the mix's own records, looked up by sign and
    sorted as ``word_pools`` sorts. Segments carry their utterance's gloss-manifest record, and every record passes the source
    check again here."""
    isolated, segments = {}, {}
    for entry, row in arm["entries"].items():
        if row["source_kind"] == "isolated":
            found = []
            for source in row["sources"]:
                record = manifest_records.get(source["sign"])
                if record is None:
                    raise SystemExit(f"{entry}: {source['sign']} is not in the source mix")
                check_source_record(record)
                found.append(record)
            isolated[entry] = sorted(found, key=lambda r: (r["corpus"], r["sign"], r["continuous"]["state_sha256"]))
        elif row["source_kind"] == "segment":
            found = []
            for source in row["sources"]:
                record = gloss_records.get(source["record_sign"])
                if record is None or record.get("partition") != "train":
                    raise SystemExit(f"{entry}: {source['record_sign']} is not a train utterance of the gloss manifest")
                found.append(dict(source, entry=entry, record=record))
            segments[entry] = sorted(found, key=lambda s: s["sign"])
        else:
            raise SystemExit(f"{entry}: unknown source kind {row['source_kind']}")
    return isolated, segments


def median_seconds(values) -> float:
    return float(statistics.median(values))


# ---- annotation and motion, per side ------------------------------------------------------------------------------------

WRISTS = dict(left=4, right=5)        # the wrists in the keypoint view (keypoints2d)
CONTACT_WIDTHS = .5                   # both wrists within half a shoulder width: the keypoint view's own unit is one shoulder width


def annotated_two_handed(record) -> bool | None:
    """ASLLVD's and RIT's annotation: a non-dominant handshape means two hands. None where a record has no annotation."""
    shapes = ((record or {}).get("gloss") or {}).get("handshapes")
    if shapes is None:
        return None
    return any(k.lower().startswith("non-dominant") and v for k, v in shapes.items())


def annotated_label(record) -> bool | None:
    """True for an annotated one-handed record, False for two-handed, None without an annotation."""
    two = annotated_two_handed(record)
    return None if two is None else not two


def entry_annotations(pools) -> dict:
    """{entry: True / False}: the majority annotation of an entry's records, for words with no record of their own (generated). A tie is
    not a majority of one-handed records, so it is two-handed; an entry with no annotation has no label."""
    out = {}
    for entry, records in pools.items():
        labels = [x for x in map(annotated_label, records) if x is not None]
        if labels:
            out[entry] = sum(labels) * 2 > len(labels)
    return out


def bridge_kind(a, b) -> str:
    """The one set of joins every table uses: both sides annotated one-handed; one-handed against annotated two-handed; one-handed against
    an unannotated word; anything else."""
    if a is True and b is True:
        return "one_to_one"
    if (a is True and b is False) or (a is False and b is True):
        return "one_to_two"
    if a is True or b is True:
        return "one_to_unannotated"
    return "other"


def keypoint_view(state):
    """(points, valid) in the report's own view (``compositional_baseline.motion_rows``'s path)."""
    from signtranslator.data.keypoints2d import from_state, normalise
    from evaluate_text_to_motion import at_rate
    points, valid = from_state(at_rate(state, 24.))
    return normalise(points, valid, y_down=False), valid


def side_read(points, valid, fps: float = 24.) -> dict:
    """Each wrist's speed and spread, exactly as ``keypoint_statistics`` reads both together."""
    import numpy as np
    out = {}
    for side, w in WRISTS.items():
        ok = valid[1:, w] & valid[:-1, w]
        out[f"{side}_speed"] = float((np.linalg.norm(points[1:, w] - points[:-1, w], axis=-1)[ok] * fps).mean()) if ok.any() else None
        seen = valid[:, w]
        out[f"{side}_spread"] = float(points[seen, w].std(0).mean()) if seen.sum() >= 2 else None
    return out


def contact_counts(points, valid, labels, one_handed) -> list[int]:
    """[frames with both wrists within ``CONTACT_WIDTHS``, frames with both wrists seen], on the cores of words annotated two-handed."""
    import numpy as np
    labels = np.asarray(labels)
    two = np.array([k >= 0 and one_handed[k] is False for k in labels], dtype=bool)
    seen = two & valid[:, WRISTS["left"]] & valid[:, WRISTS["right"]]
    near = np.linalg.norm(points[:, WRISTS["left"]] - points[:, WRISTS["right"]], axis=-1) <= CONTACT_WIDTHS
    return [int((near & seen).sum()), int(seen.sum())]


def motion_row(state, plan=None, one_handed=None) -> dict:
    """One rendered form's per-side read, and its contact counts when given its plan and labels."""
    points, valid = keypoint_view(state)
    row = side_read(points, valid)
    if plan is not None:
        from signtranslator.generation.composition_v2 import frame_labels
        row["contact"] = contact_counts(points, valid, frame_labels(plan), one_handed)
    return row


def motion_report(rows) -> dict:
    """Per form over the clips that read it, against ``real`` on the same clips: each side's mean speed and spread as a ratio to real's,
    and the contact share of frames."""
    import numpy as np
    forms = sorted({f for r in rows for f in r.get("comp_a_motion", {})})
    out = {}
    for form in forms:
        have = [r["comp_a_motion"] for r in rows if form in r.get("comp_a_motion", {})]
        row = {}
        for key in ("left_speed", "right_speed", "left_spread", "right_spread"):
            pairs = [(x[form][key], x["real"][key]) for x in have if x[form].get(key) is not None and x["real"].get(key) is not None]
            row[f"{key}_over_real"] = round(float(np.mean([a for a, _ in pairs]) / np.mean([b for _, b in pairs])), 4) if pairs else None
        counts = [x[form]["contact"] for x in have if x[form].get("contact")]
        total = sum(b for _, b in counts)
        row["contact_share"] = round(sum(a for a, _ in counts) / total, 4) if total else None
        row["contact_frames"] = total
        row["clips"] = len(have)
        out[form] = row
    return out


def load_hold_stats(path, sha: str) -> dict:
    """The committed How2Sign train numbers of the hold: {"ready": {joint: 6D}, "cap": m/s}."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha:
        raise SystemExit(f"{path}: SHA-256 {digest}, expected {sha}")
    payload = json.loads(raw)
    if payload.get("kind") != "comp-a-hold-stats":
        raise SystemExit(f"{path}: not the hold's statistics")
    train = payload["how2sign_train"]
    return dict(ready=train["ready_pose"], cap=float(train["left_speed_p99"]), sha256=sha)


# ---- reads on hypotheses -------------------------------------------------------------------------------------------------

def is_loop(hypothesis: str) -> bool:
    """A whitespace 3-gram repeated at least 3 times."""
    t = hypothesis.split()
    counts = collections.Counter(tuple(t[i:i + 3]) for i in range(len(t) - 2))
    return bool(counts) and max(counts.values()) >= 3


def loop_flags(hypotheses) -> list[float]:
    return [float(is_loop(h)) for h in hypotheses]


def echo_rows(hypotheses, word_lists, references) -> list[tuple[int, int]]:
    """Per clip (content entries found in the hypothesis, content entries the clip signs that its reference holds), on BLEU's
    tokens (``motion_to_text.words``)."""
    from signtranslator.generation.motion_to_text import words as tokens_of
    rows = []
    for h, words, r in zip(hypotheses, word_lists, references):
        tokens, ref = set(tokens_of(h)), set(tokens_of(r))
        content = [w for w in {x.lower() for x in words} if w not in ECHO_STOP and w in ref]
        rows.append((sum(w in tokens for w in content), len(content)))
    return rows


def new_entry_echo(hypotheses, word_lists, references, entries) -> float | None:
    """The echo of an arm's new entries alone, stop list or not (they are mostly pronouns)."""
    from signtranslator.generation.motion_to_text import words as tokens_of
    entries, found, total = set(entries), 0, 0
    for h, words, r in zip(hypotheses, word_lists, references):
        tokens, ref = set(tokens_of(h)), set(tokens_of(r))
        content = [w for w in {x.lower() for x in words} if w in entries and w in ref]
        found += sum(w in tokens for w in content)
        total += len(content)
    return found / total if total else None


def echo_share(rows) -> float | None:
    total = sum(b for _, b in rows)
    return sum(a for a, _ in rows) / total if total else None


__all__ = ["annotated_label", "annotated_two_handed", "entry_annotations", "bridge_kind", "keypoint_view", "side_read", "contact_counts", "motion_row",
           "motion_report", "load_hold_stats", "segment_widening", "D20_KEY", "D20_SHA", "ALLOWED_CORPORA", "FORBIDDEN_SOURCE", "TOPS", "REAL_SCALED", "require_d20", "check_d20_payload",
           "check_source_record", "parse_arm_scale", "load_token_list", "arm_of", "apply_token_map", "extra_vocabulary",
           "extra_pools", "median_seconds", "is_loop", "loop_flags", "echo_rows", "new_entry_echo", "echo_share"]
