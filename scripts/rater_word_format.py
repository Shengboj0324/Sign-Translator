"""The rater pilot's two report-only arms (docs/HUMAN_EVALUATION_PROTOCOL.md section 2; docs/NEXT_PHASE_PLAN.md, "Rater pilot:
word-format arms"). Called by ``round15_compose_eval.py run --export-word-format`` in the same run as the composition it
controls, so both forms come from the very clips, cuts and options of that composition.

- ``composition_shuffled``: the composition's own clips in a seeded random order, every word kept. Each word keeps its
  planned core frames and the gap lengths keep their places, so the frame count and every core are the composition's; only
  the joins are bridged anew (``round15_stage2.render_form``'s ``keep_frames``, as ``parent_controls.moved_plan``).
- ``oracle``: the composition's word list, each word a real clip chosen by chance (``parent_controls.pick_index``, no
  instrument), through the same tokenizer round trip as the generated clips (the word cache's prior's own tokenizer, the same
  gains setting: ``build_recon_states.decode_state``), then the same cut and stitching, at the same frame count. A word with
  no permitted real clip is dropped and the rest share the frames, so oracle against composition mixes real clips with
  fewer words: ``same_words`` marks the sentences where it does not.

**Licence.** The oracle takes train records of ``asllvd-text-v1`` only, under the DAI terms (reviewer_package/OPERATOR_GUIDE.md
section 0). RIT and WLASL need the owner's recorded confirmation (``--word-format-licence-note``). Everything else, ASL Citizen,
ASL STEM Wiki, YouTube-ASL and Sem-Lex among them, is refused. No record of How2Sign signer c00 or of a test partition is ever
fetched here, and only D20's pilot sentences are composed.

These forms are report only and pilot only: they are not bar forms, and no instrument reads them here.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

SHUFFLED, ORACLE = "composition_shuffled", "oracle"
ORACLE_DEFAULT = ("asllvd-text-v1",)
NEEDS_NOTE = ("rit-text-r6", "wlasl-text-r6")        # OPERATOR_GUIDE section 0: "Not unless you confirm its terms allow it"
SHUFFLE_VARIANT = 1000                               # shuffle_order's variant index: apart from /shuffled's 0, and parent_controls's 100 x draw + seed


def parse_corpora(text: str) -> tuple:
    return tuple(c for c in (text or "").split(",") if c)


def check_oracle_corpora(corpora, note) -> None:
    """Refuse a corpus the operator guide does not allow: only ASLLVD by default, RIT and WLASL with the owner's note."""
    if not corpora:
        raise SystemExit("the oracle needs a corpus of real clips")
    refused = [c for c in corpora if c not in ORACLE_DEFAULT + NEEDS_NOTE]
    if refused:
        raise SystemExit(f"the oracle may not show {refused}: raters see ASLLVD, or RIT and WLASL once the owner confirms their terms")
    restricted = [c for c in corpora if c in NEEDS_NOTE]
    if restricted and not (note or "").strip():
        raise SystemExit(f"{restricted} need the owner's confirmation of their terms: --word-format-licence-note")


def check_oracle_record(record: dict, corpora) -> dict:
    """The record, or SystemExit before anything is fetched: a permitted corpus, exactly ``train``, never c00."""
    if record.get("signer_component") == "c00":
        raise SystemExit("refused: How2Sign signer 9 (c00) is never read")
    if record.get("partition") != "train":
        raise SystemExit(f"refused: a word record whose partition is not exactly train ({record.get('sign')})")
    if record.get("corpus") not in corpora:
        raise SystemExit(f"refused: a word record outside {tuple(corpora)} ({record.get('sign')})")
    return record


def oracle_pools(manifest, corpora, added) -> dict:
    """{entry: its train records of ``corpora``}, in ``word_pools``'s fixed order. An added entry has none, whatever is spelled
    the same: composition's own rule."""
    from round15_oracle import word_pools
    pools = word_pools(manifest, tuple(corpora))
    for entry in added:
        pools.pop(entry, None)
    for records in pools.values():
        for record in records:
            check_oracle_record(record, corpora)
    return pools


def check_template_record(manifest) -> None:
    """``layout_template`` takes the first How2Sign train record: never c00."""
    from round15_oracle import TEMPLATE_CORPUS
    record = next((r for r in manifest["records"] if r.get("corpus") == TEMPLATE_CORPUS and r.get("partition") == "train"), None)
    if record is None or record.get("signer_component") == "c00":
        raise SystemExit("the layout template would come from no record, or from c00")


def shuffled_order(n: int, clip: int, seed: int) -> list[int]:
    """The order of a clip's ``n`` words: seeded by clip and seed, never the identity from two words up."""
    from compositional_baseline import shuffle_order
    return shuffle_order(n, clip, SHUFFLE_VARIANT + seed)


def moved(words, order) -> bool:
    """Whether the word sequence changed: a one-word sentence and a repeated word may not."""
    return [words[k] for k in order] != list(words)


def form_name(form: str, role: str) -> str:
    """``core_crossfade/<role>``, with ``@predicted`` kept last, which is how the packet builder reads the timing."""
    base, _, mark = form.partition("@")
    return f"{base}/{role}" + (f"@{mark}" if mark else "")


def oracle_job(words, plan_model, vocab) -> dict:
    """The job of the words the oracle kept: planned exactly as composition plans a word list."""
    import round15_stage2 as r2
    plan = r2.plan_words(words, plan_model, vocab)
    return dict(mode="oracle_stage2", words=list(plan["entries"]), generate=list(plan["generate"]), targets=list(plan["targets"]),
                weights=list(plan["weights"]), seeds=[None] * len(words), block="matched")


def _put_state(store, key: str, state) -> str:
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "s.npz"
        sha = state.save(path)
        data = path.read_bytes()
    if store.exists(key):
        raise SystemExit(f"{key} exists: written once")
    store.put(key, data)
    return sha


def _index(store, args, exported: dict, role: str, rows: list, form: str, composition: dict, **extra) -> tuple[str, str]:
    index = dict(kind="round15-composed-states", arm=f"{args.export_arm}/{role}", prefix=f"{args.export_prefix}/word-format/{role}",
                 sentences=exported["sentences"], sentence_set=exported["sentence_set"], word_cache=exported["word_cache"],
                 generator_mix=exported["generator_mix"], model_sha256=exported["model_sha256"], corpora=exported["corpora"],
                 settings=exported["settings"], indexing=exported["indexing"], states=rows,
                 word_format=dict(role=role, report_only=True, pilot_only=True, form=form, composition=composition, **extra))
    key = f"{index['prefix']}/index.json"
    if store.exists(key):
        raise SystemExit(f"{key} exists: written once")
    data = json.dumps(index, indent=1, default=float).encode()
    store.put(key, data)
    print(f"{key} {hashlib.sha256(data).hexdigest()}", flush=True)
    return key, hashlib.sha256(data).hexdigest()


def check_export_inputs(export: dict, val, sentence_set) -> None:
    """Pilot sentences named, D20 alone, and each of them How2Sign val of the validation signers: never test, held-out or c00."""
    from s2_lib import check_read_records
    if not export["pilot"]:
        raise SystemExit("--export-word-format: the list names no pilot sentences; these arms are pilot only")
    if (sentence_set or {}).get("set") != "D20":
        raise SystemExit("--export-word-format reads D20 only")
    pilot = set(export["pilot"])
    check_read_records([r for r in val if r["sign"] in pilot])


def export_word_format(store, args, exported: dict, *, export, val, jobs, material, plan_model, vocab, model, manifest, pool_manifest,
                       fractions, margin, rule, added, seconds, duration_model, word_cache, render) -> dict:
    """Both arms' states and indexes for the list's pilot sentences. ``render(material, job, seconds, order, keep_frames,
    sample_id)``: the composition's own renderer, every option of the run applied. Returns {role: (index key, hash)}."""
    from build_recon_states import decode_state
    from evaluate_text_to_motion import load_generator
    from exposure_diagnostic import make_post
    from parent_controls import RealClips
    from round15_oracle import TEMPLATE_CORPUS, word_clip
    from signtranslator.data_engineering.corpus_manifest import ContentCache
    from signtranslator.generation.composition_v2 import predicted_seconds
    from train_motion_tokens import layout_template
    import round15_stage2 as r2
    corpora = parse_corpora(args.word_format_corpora)
    check_oracle_corpora(corpora, args.word_format_licence_note)
    check_export_inputs(export, val, exported.get("sentence_set"))
    forms = {r["form"] for r in exported["states"]}
    if len(forms) != 1:
        raise SystemExit(f"--export-word-format moves one form, not {sorted(forms)}")
    form = next(iter(forms))
    predicted = form.endswith("@predicted")
    if predicted and duration_model is None:
        raise SystemExit("no fitted duration model for the predicted form")
    position = {r["sign"]: i for i, r in enumerate(val)}
    composed = {r["sign"]: r for r in exported["states"]}
    settings = json.loads(store.get(args.word_cache_key))["settings"]
    _, _, sample = load_generator(store, word_cache["checkpoint_key"], settings.get("generator_mix_sha256") or args.mix_sha256, args.device,
                                  decoder_gains=not word_cache.get("no_decoder_gains"))
    tokenizer, post = sample.tokenizer, make_post(sample.decoder_gains)
    cache = ContentCache(Path(args.cache_dir) / "word-format", 2_000_000_000)
    check_template_record(manifest)
    template = layout_template(manifest, TEMPLATE_CORPUS, cache, store.get)

    def load(record: dict, entry: str):
        check_oracle_record(record, corpora)
        return decode_state(word_clip(record, cache=cache, fetch=store.get, template=template), tokenizer, tokenizer.config.motion, post, text=entry, device=args.device)
    pools = oracle_pools(pool_manifest, corpora, added)
    real = RealClips(pools, args.word_format_seed, load)
    shuffled_rows, oracle_rows = [], []
    counts = collections.Counter()
    dropped_reasons, by_corpus = collections.Counter(), collections.Counter()
    prefix = f"{args.export_prefix}/word-format"
    for sign in sorted(export["pilot"], key=position.get):
        i, row = position[sign], composed[sign]
        tag = hashlib.sha256(sign.encode()).hexdigest()[:20]
        job = jobs[("core_crossfade", "", i)]
        base = {k: row[k] for k in ("clip", "sign", "text", "reference_frames", "reference_sha256")}
        base.update(frames=row["frames"], composition_state_sha256=row["state_sha256"])
        if job is None:
            for rows, role in ((shuffled_rows, SHUFFLED), (oracle_rows, ORACLE)):
                rows.append(dict(base, form=form_name(form, role), empty=True, state_key=None, state_sha256=None, words=[], fit=None, overlong=None))
            continue
        secs = predicted_seconds(duration_model, val[i]["text"]) if predicted else seconds[i]
        source = material[("core_crossfade", "", i)]
        entries = job["words"]
        order = shuffled_order(len(entries), i, args.word_format_seed)
        counts["moved"] += moved(entries, order)
        several = len(entries) > 1           # a one-word sentence has no other order: it is the composition again, and not moved
        state, plan = render(source, job, secs, order if several else None, several, f"round15-eval-core_crossfade/shuffled:{i}")
        if len(state.timestamps) != row["frames"]:
            raise SystemExit(f"clip {i}: shuffled has {len(state.timestamps)} frames, the composition {row['frames']}")
        key = f"{prefix}/{SHUFFLED}/states/{tag}.npz"
        shuffled_rows.append(dict(base, form=form_name(form, SHUFFLED), empty=False, state_key=key, state_sha256=_put_state(store, key, state),
                                  sample_id=state.sample_id, fit=plan.get("fit"), overlong=plan.get("overlong"), order=order, moved=moved(entries, order),
                                  words=[row["words"][k] for k in order]))
        picked = [real.of(e) for e in entries]
        kept = [p for p, x in enumerate(picked) if x is not None]
        dropped = [dict(word=entries[p], reason="added_entry" if entries[p] in added else "not_served" if pools.get(entries[p]) else "no_permitted_clip")
                   for p, x in enumerate(picked) if x is None]
        dropped_reasons.update(d["reason"] for d in dropped)
        counts["words_selected"] += len(entries)
        counts["words_kept"] += len(kept)
        counts["same_words"] += not dropped
        record = dict(base, form=form_name(form, ORACLE), dropped=dropped, same_words=not dropped)
        if not kept:
            counts["empty"] += 1
            oracle_rows.append(dict(record, empty=True, state_key=None, state_sha256=None, words=[], fit=None, overlong=None))
            continue
        kept_job = oracle_job([entries[p] for p in kept], plan_model, vocab)
        made = r2.oracle_material(kept_job, i, None, loaded={picked[p][0]["sign"]: picked[p][1] for p in kept}, load=None, rule=rule, timing=args.timing,
                                  fractions=fractions, margin=margin, tighten=args.tighten_cores, records=[picked[p][0] for p in kept])
        state, plan = render(made, kept_job, secs, None, False, f"round15-eval-core_crossfade/oracle:{i}")
        if len(state.timestamps) != row["frames"]:
            raise SystemExit(f"clip {i}: the oracle has {len(state.timestamps)} frames, the composition {row['frames']}: duration would be a cue")
        key = f"{prefix}/{ORACLE}/states/{tag}.npz"
        words = [{k: s.get(k) for k in ("word", "sign", "corpus", "lo", "hi", "reason")} for s in made["segments"]]
        by_corpus.update(w["corpus"] for w in words)
        oracle_rows.append(dict(record, empty=False, state_key=key, state_sha256=_put_state(store, key, state), sample_id=state.sample_id,
                                fit=plan.get("fit"), overlong=plan.get("overlong"), words=words))
    index_key = f"{args.export_prefix}/index.json"
    composition = dict(key=index_key, sha256=hashlib.sha256(store.get(index_key)).hexdigest(), arm=exported["arm"], form=form)
    out = {}
    out[SHUFFLED] = _index(store, args, exported, SHUFFLED, shuffled_rows, form, composition, seed=args.word_format_seed, shuffle_variant=SHUFFLE_VARIANT, sentences_moved=counts["moved"])
    out[ORACLE] = _index(store, args, exported, ORACLE, oracle_rows, form, composition, seed=args.word_format_seed, oracle_corpora=list(corpora), licence_note=args.word_format_licence_note, pick="chance, by a hash of seed and entry: no instrument",
                         round_trip=dict(checkpoint_key=word_cache["checkpoint_key"], decoder_gains=not word_cache.get("no_decoder_gains")),
                         coverage=dict(sentences=len(oracle_rows), words_selected=counts["words_selected"], words_kept=counts["words_kept"], same_words=counts["same_words"], empty=counts["empty"], dropped=dict(sorted(dropped_reasons.items())), corpora=dict(sorted(by_corpus.items())), entries_not_served=real.entries_not_served(), picks_not_served=real.failed))
    return out
