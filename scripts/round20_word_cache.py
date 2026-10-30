"""Round 20 A4/A5: a best-of-N cache of generated word signs, and word-level recogniser reads of a prior
(docs/NEXT_PHASE_PLAN.md, "Round 20: composition v2", declared 2026-10-30 19:09 UTC). Generation and scoring only;
nothing trains.

**Entries:** every vocabulary entry the stage-1 selection keeps for the sentences (``--sentences-key``: D20 / C20;
else ``validation_sentences(mix, --clips)``), on the vocabulary of ``--vocabulary-mix-key`` (default the evaluation
mix) and its corpora, exactly as ``round15_compose_eval.py`` selects them (the matched and the mismatched words of a
set are the same set's entries); with a set, ``--clips`` keeps its first N sentences (as the composition does).
``--limit`` keeps the first K entries (smoke runs).

**Samples:** entry e is generated ``--samples`` times exactly as the composition generates a word
(``compositional.generate_word``: ``generate_rvq_motion`` at the entry's typical isolated seconds, guidance and part
temperatures as the composition's, resampled to 24 fps, the generator's post-processing), sample k with seed
``rerank.rerank_seeds(round20_lib.entry_seed(e), N)[k]`` (fixed per entry, independent of any sentence; sample 0's seed
is the entry's own).

**Scores:** recogniser ``--recognizer-run`` (v3d, the instrument) reads each whole sample; the score is the softmax
probability of the entry's label (``round20_lib.entry_label``: the entry upper-cased, multi-word entries joined); each
sample's rank of the label is kept for top-1 / top-5 reads. An entry without a label falls back to the prior's own
teacher-forced PMI (``rerank.word_pmi`` on the sampled codes). **Choice** for each N of ``--choose`` (each <=
``--samples``): the best of samples 0..N-1 (``rerank.choose``: the first on a tie), so N = 4, 8, 16 come from one run.
The chosen samples are stored (``<prefix>/states/<entry hash>-s<k>.npz``) with every score.

**Circularity:** a composition built from this cache is chosen by the cache's recogniser, so its word spotting by that
recogniser is circular; ``round15_compose_eval.py`` marks it, and recogniser v2 spotting is the independent guard.

**A5 (word-level reads):** the merged index summarises, over labelled entries, the probability of the label, top-1 and
top-5 on sample 0 (one plain sample per entry), on all samples, and on each choice; ``scripts/round20_compare.py words``
pairs two priors' indexes entry by entry.

Subcommands: ``run`` (one shard of the entries: entry k belongs to shard k mod ``--shards``; writes
``<prefix>/shard-<s>-of-<S>.json``) and ``merge`` (all shards, complete and disjoint, into ``<prefix>/index.json``).
Every key is written once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

FPS = 24.
EVAL_MIX = "acquisition/2026-10-17-v1/youtube-asl-v1/round10-mixed-v1/manifest.json"
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"


def entry_hash(entry: str) -> str:
    return hashlib.sha256(entry.encode()).hexdigest()[:20]


def choices(scores, ns) -> dict:
    """{N: the index of the best of the first N scores}."""
    from signtranslator.generation.rerank import choose
    return {str(n): choose(list(scores[:n])) for n in ns}


def summarise(rows, ns) -> dict:
    """Word-level reads over labelled entries (module docstring) and the unlabelled count."""
    lab = [r for r in rows if r["label"]]
    out = dict(entries=len(rows), labelled=len(lab), unlabelled=len(rows) - len(lab))
    if not lab:
        return out

    def read(pick):
        p = [pick(r)[0] for r in lab]
        rank = [pick(r)[1] for r in lab]
        return dict(p_label=round(float(np.mean(p)), 4), top1=round(float(np.mean([k == 1 for k in rank])), 4),
                    top5=round(float(np.mean([k <= 5 for k in rank])), 4))
    out["sample0"] = read(lambda r: (r["probabilities"][0], r["ranks"][0]))
    out["all_samples"] = dict(
        p_label=round(float(np.mean([p for r in lab for p in r["probabilities"]])), 4),
        top1=round(float(np.mean([k == 1 for r in lab for k in r["ranks"]])), 4),
        top5=round(float(np.mean([k <= 5 for r in lab for k in r["ranks"]])), 4))
    for n in ns:
        out[f"best_of_{n}"] = read(lambda r, n=n: (r["probabilities"][r["chosen"][str(n)]["index"]],
                                                   r["ranks"][r["chosen"][str(n)]["index"]]))
    return out


def select_entries(args, store):
    """(entries in order, vocabulary, vocabulary mix, corpora, sentence info) as the composition selects them."""
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from signtranslator.generation.compositional import isolated_corpora, training_vocabulary
    from pilot_content_check import validation_sentences
    from round15_compose_eval import stage1_selection
    from round20_lib import load_sentence_set
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    vmanifest, vmix = manifest, dict(key=args.mix_key, sha256=args.mix_sha256)
    if args.vocabulary_mix_key:
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "v.json"; path.write_bytes(store.get(args.vocabulary_mix_key) + b"\n")
            vmanifest = load_manifest(path, expected_sha256=args.vocabulary_mix_sha256)
        vmix = dict(key=args.vocabulary_mix_key, sha256=args.vocabulary_mix_sha256)
    corpora = tuple(c for c in args.corpora.split(",") if c) if args.corpora else isolated_corpora(vmanifest)
    vocab = training_vocabulary(vmanifest, corpora)
    if args.sentences_key:
        payload = store.get(args.sentences_key)
        if hashlib.sha256(payload).hexdigest() != args.sentences_sha256:
            raise SystemExit(f"{args.sentences_key} does not match --sentences-sha256")
        val = load_sentence_set(json.loads(payload), manifest, args.mix_key, args.mix_sha256)[:args.clips]
        sentences = dict(key=args.sentences_key, sha256=args.sentences_sha256, count=len(val))
    else:
        val = validation_sentences(manifest, args.clips)
        sentences = dict(validation_sentences=args.clips, count=len(val))
    _, words = stage1_selection([r["text"] for r in val], vocab)
    entries = sorted({w for ws in words for w in ws})
    return entries, vocab, vmix, corpora, sentences


def run(args) -> dict:
    import torch
    from signtranslator.generation.motion_rvq import generate_rvq_motion
    from signtranslator.generation.rerank import rerank_seeds, word_pmi
    from signtranslator.pose.resample import resample_motion
    from evaluate_text_to_motion import at_rate, load_generator
    from exposure_diagnostic import make_post
    from round20_lib import entry_label, entry_seed
    from train_text_to_motion import make_store
    import validate_sign_recognizer as vsr
    store = make_store(args.store)
    entries, vocab, vmix, corpora, sentences = select_entries(args, store)
    total = len(entries)
    if args.limit:
        entries = entries[:args.limit]
    mine = [e for k, e in enumerate(entries) if k % args.shards == args.shard]
    temperature = {k: float(v) for k, v in (x.split("=") for x in args.part_temperature.split(","))}
    gen_sha = args.generator_mix_sha256 or args.mix_sha256
    _, _, sample = load_generator(store, args.checkpoint_key, gen_sha, args.device, temperature=temperature)
    post = make_post(sample.decoder_gains)
    recogniser, labels, best = vsr.load_model(store, args.recognizer_run, args.device)
    index = {g: k for k, g in enumerate(labels)}
    ns = sorted({int(x) for x in args.choose.split(",") if x})
    start = time.time()
    rows = []
    for j, e in enumerate(mine):
        label = entry_label(e, labels)
        seconds = float(vocab[e])
        seeds = rerank_seeds(entry_seed(e), args.samples)
        states, probs, ranks, pmis = [], [], [], []
        for s in seeds:
            raw, codes = generate_rvq_motion(sample.prior, sample.tokenizer, [e], fps=FPS, durations=[seconds],
                                             guidance=args.guidance, temperature=temperature, seed=s,
                                             return_codes=True)
            state = post(resample_motion(raw[0], FPS))
            states.append(state)
            if label:
                logits = vsr.logits_of(recogniser, [at_rate(state, FPS)], args.device)[0]
                p = torch.softmax(logits.float(), 0)
                probs.append(round(float(p[index[label]]), 6))
                ranks.append(int((p > p[index[label]]).sum()) + 1)
            else:
                pmis.append(word_pmi(sample.prior, e, codes[0], seconds)["pmi_per_code"])
        scores = probs if label else pmis
        chosen = {}
        for n, k in choices(scores, ns).items():
            key = f"{args.prefix}/states/{entry_hash(e)}-s{k}.npz"
            with tempfile.TemporaryDirectory() as t:
                path = Path(t) / "s.npz"
                sha = states[k].save(path)
                payload = path.read_bytes()
            if store.exists(key):
                if hashlib.sha256(store.get(key)).hexdigest() != sha:
                    raise SystemExit(f"{key} exists with other content: written once")
            else:
                store.put(key, payload)
            chosen[n] = dict(index=k, seed=seeds[k], score=scores[k], state_key=key, state_sha256=sha)
        rows.append(dict(entry=e, label=label, scorer="recogniser" if label else "prior_pmi", seconds=seconds,
                         seeds=seeds, scores=scores, probabilities=probs, ranks=ranks, chosen=chosen))
        if (j + 1) % 25 == 0 or j + 1 == len(mine):
            print(f"{j + 1}/{len(mine)} entries ({time.time() - start:.0f} s)", flush=True)
    settings = dict(mix_key=args.mix_key, mix_sha256=args.mix_sha256, vocabulary_mix=vmix, corpora=list(corpora),
                    sentences=sentences, entries_total=total, limit=args.limit, checkpoint_key=args.checkpoint_key,
                    generator_mix_sha256=gen_sha, samples=args.samples, choose=ns, guidance=args.guidance,
                    part_temperature=temperature, recognizer_run=args.recognizer_run,
                    recognizer_checkpoint=best.get("checkpoint_key") if isinstance(best, dict) else None,
                    shards=args.shards, labels=len(labels))
    return dict(kind="round20-word-cache-shard", shard=args.shard, settings=settings, entries=rows,
                seconds=round(time.time() - start, 1))


def merge(args, store) -> dict:
    parts = [json.loads(store.get(f"{args.prefix}/shard-{s}-of-{args.shards}.json")) for s in range(args.shards)]
    first = parts[0]["settings"]
    for p in parts[1:]:
        if p["settings"] != first:
            raise SystemExit(f"shard {p['shard']} differs in its settings")
    rows = [r for p in parts for r in p["entries"]]
    names = [r["entry"] for r in rows]
    if len(set(names)) != len(names):
        raise SystemExit("an entry is in two shards")
    expected = first["entries_total"] if not first.get("limit") else min(first["limit"], first["entries_total"])
    if len(rows) != expected:
        raise SystemExit(f"the shards hold {len(rows)} of {expected} entries")
    rows.sort(key=lambda r: r["entry"])
    return dict(kind="round20-word-cache", settings=first, entries=rows, summary=summarise(rows, first["choose"]),
                circularity=f"compositions from this cache are chosen by {first['recognizer_run']}: its spotting of "
                            "them is circular (use sign-recognizer-v2 as the guard)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mix-key", default=EVAL_MIX)
    r.add_argument("--mix-sha256", default=EVAL_MIX_SHA)
    r.add_argument("--vocabulary-mix-key")
    r.add_argument("--vocabulary-mix-sha256")
    r.add_argument("--corpora")
    r.add_argument("--sentences-key")
    r.add_argument("--sentences-sha256")
    r.add_argument("--clips", type=int, default=1000)
    r.add_argument("--checkpoint-key", required=True)
    r.add_argument("--generator-mix-sha256", help="the manifest the prior was trained on (default: the eval mix)")
    r.add_argument("--samples", type=int, default=16)
    r.add_argument("--choose", default="4,8,16")
    r.add_argument("--recognizer-run", default="sign-recognizer-v3d")
    r.add_argument("--guidance", type=float, default=1.5)
    r.add_argument("--part-temperature", default="body=1.1,left_hand=1.1,right_hand=1.1,face=1.0")
    r.add_argument("--device", default="cuda")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--shards", type=int, default=1)
    r.add_argument("--limit", type=int, help="the first K entries only (smoke)")
    m = sub.add_parser("merge")
    m.add_argument("--shards", type=int, default=1)
    for q in (r, m):
        q.add_argument("--store", default="r2", help="r2, or dir:/path (tests)")
        q.add_argument("--prefix", required=True, help="the cache's write-once prefix")
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("ignore")
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if args.command == "run":
        ns = [int(x) for x in args.choose.split(",") if x]
        if not ns or min(ns) < 1 or max(ns) > args.samples:
            p.error("--choose: sample counts in 1..--samples")
        if bool(args.sentences_key) != bool(args.sentences_sha256):
            p.error("--sentences-key and --sentences-sha256 go together")
        if bool(args.vocabulary_mix_key) != bool(args.vocabulary_mix_sha256):
            p.error("--vocabulary-mix-key and --vocabulary-mix-sha256 go together")
        if not 0 <= args.shard < args.shards:
            p.error("--shard must be in 0..--shards-1")
        key = f"{args.prefix}/shard-{args.shard}-of-{args.shards}.json"
        if store.exists(key) or store.exists(f"{args.prefix}/index.json"):
            raise SystemExit(f"{key} (or the merged index) exists: refusing to overwrite")
        payload = run(args)
    else:
        key = f"{args.prefix}/index.json"
        if store.exists(key):
            raise SystemExit(f"{key} exists: refusing to overwrite")
        payload = merge(args, store)
        print(json.dumps(payload["summary"]), flush=True)
    data = json.dumps(payload, indent=1).encode()
    if store.exists(key):
        raise SystemExit(f"{key} appeared during the run: refusing to overwrite")
    store.put(key, data)
    print(f"{key} {hashlib.sha256(data).hexdigest()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
