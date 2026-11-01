"""Post-S2 A4, the clip-choice read (reports/post_s2_plan.md section 3 called it a "ceiling"; it is not one, so it was renamed): real clips
of D20's r10 entries, one clip per entry, chosen by recogniser v3d the way a4 chooses samples (``round20_word_cache.py``). Nothing trains, and
no tokenizer or prior is in the path. No back-translator v1 or v2 reranks anything: the standing rule bars them from best-of-N.

**Why this is no ceiling.** v3d is one word instrument with its own biases, it trained on real clips of these corpora, and it scores a clip
alone rather than in a sentence. If the read does not rise, that is *not informative* about word source: it never shows there is no room.

**Candidates.** Each entry's pool, as the oracle's (``round15_oracle.word_pools``: train records of r10's corpora; Sem-Lex-only entries
restricted by ``round20_lib.tracked_pools`` as a1's ``--semlex-oracle-tracked``), at most ``--samples`` of them in a fixed draw seeded by
``round20_lib.entry_seed``. The draw's order is the sample order, so best-of-N for N in ``--choose`` come from one run, as a4's do.

**Score.** v3d's unrounded log-probability of ``round20_lib.entry_label(entry)`` on the whole word clip, ties to the lower sample index
(``rerank.choose``), exactly a4's rule.

**Picks.** ``chosen`` holds, per entry: each best-of-N; ``sample0``, the first candidate with no scorer, so best-of-16 minus sample0 is the
choice alone, and sample0 minus a1 is repetition alone (one clip per entry against a1's draw per occurrence); and ``draw1`` to ``draw4``,
independent fixed draws, for how much the read moves by which clip an entry happens to get.

**What differs from a4, on purpose.** An entry without a v3d label has no score. a4 fell back to the prior's PMI, which needs the tokenizer and
the prior; this read keeps both out of the path, so that entry's best-of-N is its sample 0 (``scorer`` ``none``). Its share is in ``summary``.

**Saturation.** ``summary`` reports how many entries tie at the top or are top-1 on every clip: a choice among clips the recogniser has
memorised is weaker evidence than a4's choice among samples. The index goes to ``round15_compose_eval.py --oracle-choice-key``; recogniser v2
spotting stays the guard. D20 only; write-once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from comp_a_lib import check_d20_payload, check_source_record, provenance_row, require_d20  # noqa: E402

FPS = 24.
EVAL_MIX = "acquisition/2026-10-17-v1/youtube-asl-v1/round10-mixed-v1/manifest.json"
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"
TIE = 1e-9                                   # log-probabilities this close count as a tie in the saturation read only
DRAWS = 4


def candidates(pool: list, entry: str, samples: int) -> list:
    """At most ``samples`` of the pool in a fixed order by the entry; the first N of the order are a best-of-N's candidates."""
    from round20_lib import entry_seed
    return random.Random(entry_seed(entry)).sample(pool, min(samples, len(pool)))


def independent_draw(pool: list, entry: str, k: int) -> dict:
    """Draw ``k`` of an entry's pool: one record, by a seed of its own, whatever the other draws took."""
    from round20_lib import entry_seed
    return random.Random(f"{entry_seed(entry)}|clip-draw|{k}").choice(pool)


def summarise(rows, ns) -> dict:
    labelled = [r for r in rows if r["label"]]
    out = dict(entries=len(rows), labelled=len(labelled), unlabelled=len(rows) - len(labelled),
               unlabelled_note="no score: sample 0, a fixed draw", clips_per_entry_median=float(np.median([len(r["signs"]) for r in rows]))
               if rows else None, single_clip_entries=int(sum(len(r["signs"]) == 1 for r in rows)))
    if not labelled:
        return out

    def read(pick):
        return dict(p_label=round(float(np.mean([r["probabilities"][pick(r)] for r in labelled])), 4),
                    top1=round(float(np.mean([r["ranks"][pick(r)] == 1 for r in labelled])), 4),
                    top5=round(float(np.mean([r["ranks"][pick(r)] <= 5 for r in labelled])), 4))
    out["sample0"] = read(lambda r: 0)
    for n in ns:
        out[f"best_of_{n}"] = read(lambda r, n=n: r["chosen"][str(n)]["index"])
    many = [r for r in labelled if len(r["signs"]) > 1]
    out["saturation"] = dict(entries_with_several_clips=len(many), tied_at_the_top=int(sum(
        sum(abs(s - max(r["scores"])) < TIE for s in r["scores"]) > 1 for r in many)),
        top1_every_clip=int(sum(all(k == 1 for k in r["ranks"]) for r in many)))
    return out


def run(args) -> dict:
    import torch
    from signtranslator.data_engineering.corpus_manifest import ContentCache
    from signtranslator.generation.compositional import isolated_corpora, training_vocabulary
    from signtranslator.generation.rerank import choose
    from evaluate_text_to_motion import at_rate
    from round15_compose_eval import TRACKED_FLOOR, stage1_selection
    from round15_oracle import TEMPLATE_CORPUS, word_clip, word_pools
    from round20_lib import entry_label, load_sentence_set, semlex_only_entries, tracked_pools
    from s2_lib import check_read_records, load_manifest_at
    from train_motion_tokens import layout_template
    from train_text_to_motion import make_store
    import validate_sign_recognizer as vsr
    store = make_store(args.store)
    require_d20(args.sentences_key, args.sentences_sha256)
    manifest = load_manifest_at(store, args.mix_key, args.mix_sha256)
    raw = store.get(args.sentences_key)
    if hashlib.sha256(raw).hexdigest() != args.sentences_sha256:
        raise SystemExit(f"{args.sentences_key} does not match --sentences-sha256")
    payload = json.loads(raw)
    check_d20_payload(payload)
    val = load_sentence_set(payload, manifest, args.mix_key, args.mix_sha256)[:args.clips]
    check_read_records(val)
    corpora = isolated_corpora(manifest)
    vocab = training_vocabulary(manifest, corpora)
    _, words = stage1_selection([r["text"] for r in val], vocab)
    entries = sorted({w for ws in words for w in ws})
    cache = ContentCache(Path(args.cache_dir), 8 * 2 ** 30)
    template = layout_template(manifest, TEMPLATE_CORPUS, cache, store.get)
    load = lambda r: word_clip(r, cache=cache, fetch=store.get, template=template)    # noqa: E731
    pools = word_pools(manifest, corpora)
    pools, tracked = tracked_pools(pools, set(entries) & set(semlex_only_entries(pools)), load=load, loaded={}, floor=TRACKED_FLOOR)
    recogniser, labels, best = vsr.load_model(store, args.recognizer_run, args.device)
    index = {g: k for k, g in enumerate(labels)}
    ns = sorted({int(x) for x in args.choose.split(",") if x})
    start, rows = time.time(), []
    for j, e in enumerate(entries):
        chosen_pool = candidates(pools[e], e, args.samples)
        for r in chosen_pool:
            check_source_record(r)
        label = entry_label(e, labels)
        logps, probs, ranks = [], [], []
        if label:
            for r in chosen_pool:
                logits = vsr.logits_of(recogniser, [at_rate(load(r), FPS)], args.device)[0].double()
                lp = torch.log_softmax(logits, 0)
                logps.append(float(lp[index[label]]))
                probs.append(round(float(lp[index[label]].exp()), 6))
                ranks.append(int((lp > lp[index[label]]).sum()) + 1)
        chosen = {}
        for n in ns:
            k = choose(logps[:n]) if label else 0
            chosen[str(n)] = dict(index=k, sign=chosen_pool[k]["sign"], corpus=chosen_pool[k]["corpus"])
        chosen["sample0"] = dict(index=0, sign=chosen_pool[0]["sign"], corpus=chosen_pool[0]["corpus"])
        for k in range(1, DRAWS + 1):
            r = independent_draw(pools[e], e, k)
            check_source_record(r)
            chosen[f"draw{k}"] = dict(sign=r["sign"], corpus=r["corpus"])
        rows.append(dict(entry=e, label=label, scorer="recogniser" if label else "none", signs=[r["sign"] for r in chosen_pool],
                         corpora=[r["corpus"] for r in chosen_pool], scores=logps, probabilities=probs, ranks=ranks, chosen=chosen))
        if (j + 1) % 50 == 0 or j + 1 == len(entries):
            print(f"{j + 1}/{len(entries)} entries ({time.time() - start:.0f} s)", flush=True)
    settings = dict(mix_key=args.mix_key, mix_sha256=args.mix_sha256, corpora=list(corpora),
                    sentences=dict(key=args.sentences_key, sha256=args.sentences_sha256, count=len(val)), samples=args.samples,
                    choose=ns, picks=[str(n) for n in ns] + ["sample0"] + [f"draw{k}" for k in range(1, DRAWS + 1)],
                    recognizer_run=args.recognizer_run, labels=len(labels), semlex_tracked=tracked, provenance=provenance_row(),
                    recognizer_checkpoint=best.get("checkpoint_key") if isinstance(best, dict) else None)
    return dict(kind="comp-a4-clip-choice", report_only=True, settings=settings, entries=rows, summary=summarise(rows, ns),
                seconds=round(time.time() - start, 1), circularity=f"chosen by {args.recognizer_run}: its spotting of them is circular; "
                "sign-recognizer-v2 is the guard, and it shares corpora with v3d")


def load_choice(store, args, pools, corpora) -> dict:
    """{"records": {entry: its picked record}, "info"} of the composition's ``--oracle-choice-key``, after checking hash, set and vocabulary.
    Every word of the run must have a pick and every record passes the source check."""
    raw = store.get(args.oracle_choice_key)
    if hashlib.sha256(raw).hexdigest() != args.oracle_choice_sha256:
        raise SystemExit(f"{args.oracle_choice_key} does not match --oracle-choice-sha256")
    index = json.loads(raw)
    s = index["settings"]
    if index.get("kind") != "comp-a4-clip-choice" or s["sentences"]["sha256"] != args.sentences_sha256:
        raise SystemExit("--oracle-choice-key: not a choice on this sentence set")
    if s["mix_sha256"] != args.mix_sha256 or list(s["corpora"]) != list(corpora):
        raise SystemExit("--oracle-choice-key: chosen on another vocabulary")
    pick = str(args.oracle_choice_pick)
    if pick not in s["picks"]:
        raise SystemExit(f"--oracle-choice-key has no pick {pick} (it has {s['picks']})")
    records = {}
    for row in index["entries"]:
        sign = row["chosen"][pick]["sign"]
        found = next((r for r in pools.get(row["entry"], ()) if r["sign"] == sign), None)
        if found is None:
            raise SystemExit(f"{row['entry']}: the picked clip {sign} is not in its pool")
        check_source_record(found)
        records[row["entry"]] = found
    return dict(records=records, info=dict(key=args.oracle_choice_key, sha256=args.oracle_choice_sha256, pick=pick,
                                           scorer=s["recognizer_run"], samples=s["samples"], summary=index.get("summary")))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mix-key", default=EVAL_MIX)
    p.add_argument("--mix-sha256", default=EVAL_MIX_SHA)
    p.add_argument("--sentences-key", required=True)
    p.add_argument("--sentences-sha256", required=True)
    p.add_argument("--clips", type=int, default=500)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--choose", default="4,8,16")
    p.add_argument("--recognizer-run", default="sign-recognizer-v3d")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-dir", default=str(Path(tempfile.gettempdir()) / "sign-translator-cache"))
    p.add_argument("--store", default="r2", help="r2, or dir:/path (tests)")
    p.add_argument("--output-key")
    p.add_argument("--output-file")
    args = p.parse_args(argv)
    if not (args.output_key or args.output_file):
        p.error("give --output-key or --output-file")
    if args.recognizer_run != "sign-recognizer-v3d":
        p.error("--recognizer-run: only v3d chooses; the v1 and v2 instruments never rerank (standing rule)")
    ns = [int(x) for x in args.choose.split(",") if x]
    if not ns or min(ns) < 1 or max(ns) > args.samples:
        p.error("--choose: sample counts in 1..--samples")
    try:
        require_d20(args.sentences_key, args.sentences_sha256)
    except SystemExit as stop:
        p.error(str(stop))
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
    if args.output_file and Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    out = run(args)
    print(json.dumps(out["summary"]), flush=True)
    data = json.dumps(out, indent=1).encode()
    if args.output_file:
        with open(args.output_file, "xb") as stream:
            stream.write(data)
    if args.output_key:
        if store.exists(args.output_key):
            raise SystemExit(f"{args.output_key} appeared during the run: refusing to overwrite")
        store.put(args.output_key, data)
    print(args.output_key or args.output_file, hashlib.sha256(data).hexdigest(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
