"""Teacher-forced word retrieval of a prior against a comparison prior (round 14b pilot add-on; the round-14
diagnosis's measure, docs/NEXT_PHASE_PLAN.md "Round 14 diagnosis", productionised). Report-only.

The pilot's free-run lexicon reads a generated word with recogniser v2, which cannot read Sem-Lex-style signing (real
Sem-Lex val clips decoded in the pilot's form read 0.009). This asks the prior directly whether it has learned a word:
does its teacher-forced likelihood of a real clip's codes pick the clip's own word among other words?

**Clips** (all from ``--mix-key``, served exactly as the trainer serves them: ``TextMotionDataset`` with the
``how2sign-two-view-text-v1`` layout template, 24 fps, ``--max-seconds``; the **val** partition only):
- ``uncovered``: every val clip of ``--semlex-corpus`` (default ``semlex-text-v2``) whose word is one of the split's 42
  uncovered R5 words (also reported by whether the word has a train record of that corpus in the mix: newly covered,
  or still uncovered);
- ``covered``: val clips of ``--covered-corpora`` (ASLLVD, RIT, WLASL) whose word is one of the 108 covered R5 words,
  ``--covered-per-word`` per word (default 1, the diagnosis's), drawn per word with ``random.Random(f"{word}-{seed}")``.
A clip's word is its text upper-cased with spaces removed (``exposure_diagnostic``'s rule). ``--per-word`` caps the
uncovered clips per word the same way (default 0: all).

**Measure.** Each clip's frozen-tokenizer codes are scored under its own word and the first ``--candidates`` - 1 other
R5 words of a fixed seeded order (``sha256(f"{order_seed}-{word}")``; default 49 candidates, the diagnosis's), in the
pilot's prompt form ``word.lower()``, and under the empty text, with the clip's own loci and the EMA weights. The score
is **part-valid**: the summed teacher-forced log-probability of the codes whose part was observed
(``motion_rvq.part_code_weights``, the ``--part-valid-loss`` weighting the priors trained with); the unweighted sum
(``rvq_sequence_log_likelihood``, checked equal on the first two clips of every set) is reported beside it. The rank of
the own word counts ties against it. Chance top-5 is 5/49 = 0.102.

**Reported** per set and checkpoint: top-1, top-5, mean rank (word-clustered 95% bootstrap intervals), the own word's
gain over the empty text and its NLL per part-valid code; **paired** checkpoint minus comparison per clip (top-1, top-5,
mean rank, NLL per part-valid code), with word-clustered and clip bootstrap intervals; per word; every clip's ranks and
part-valid candidate scores.

**Hygiene.** The two checkpoints must share their tokenizer (its state's hash; refused otherwise: other codes make
the per-clip pairing meaningless). Only val records of the named corpora are read, plus the trainer's layout template (the first How2Sign
*train* record). No test or held-out partition is read, and no How2Sign clip other than the template (How2Sign signer
9, the generator's final-test signer, is in no partition this reads). The output is written once (``--output-key``,
refused if it exists, checked before any scoring; and/or ``--output-file``). GPU (``--device cuda``) or CPU.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

ACQ = "acquisition/2026-10-17-v1/"
WORD_SPLIT = ACQ + "youtube-asl-v1/round14/r5-word-split.json"
WORD_SPLIT_SHA256 = "676f56cbbaa9bac08f0bc305027807827f982cd4c242de03f7891323e723102a"
SEMLEX_CORPUS = "semlex-text-v2"
COVERED_CORPORA = ("asllvd-text-v1", "rit-text-r6", "wlasl-text-r6")
TEMPLATE_CORPUS = "how2sign-two-view-text-v1"
READ_PARTITION = "val"
FORBIDDEN_PARTITIONS = ("test", "heldout_official")
SETS = ("uncovered", "covered")


# ---- pure parts ----------------------------------------------------------------------------------------------------

def word_of(text: str) -> str:
    """``exposure_diagnostic``'s label of a record's text: upper case, spaces removed."""
    return text.upper().replace(" ", "")


def read_split(payload: bytes) -> tuple[list[str], list[str]]:
    """(covered, uncovered) R5 words of the split JSON: two disjoint lists of distinct words."""
    split = json.loads(payload)
    lists = [split.get(k) for k in ("covered", "uncovered")]
    for k, v in zip(("covered", "uncovered"), lists):
        if not isinstance(v, list) or not v or not all(isinstance(w, str) and w for w in v) or len(set(v)) != len(v):
            raise SystemExit(f"word split: {k!r} must be a nonempty list of distinct words")
    if set(lists[0]) & set(lists[1]):
        raise SystemExit("word split: a word is both covered and uncovered")
    return lists[0], lists[1]


def candidate_order(words, seed: str) -> list[str]:
    """The fixed seeded order of the candidate words (the diagnosis's: sha256 of ``f"{seed}-{word}"``)."""
    return sorted(words, key=lambda w: hashlib.sha256(f"{seed}-{w}".encode()).hexdigest())


def candidates(word: str, order: list[str], n: int) -> list[str]:
    """The clip's own word first, then the first ``n - 1`` other words of ``order``."""
    if word not in order:
        raise ValueError(f"{word} is not a candidate word")
    return [word] + [w for w in order if w != word][:n - 1]


def rank_own(scores) -> int:
    """1-based rank of ``scores[0]`` (the own word) among ``scores``; ties count against it."""
    s = np.asarray(scores, dtype=float)
    return int(1 + (s > s[0]).sum() + ((s == s[0]).sum() - 1))


def select(records, corpora, words, per_word: int, seed: int) -> list[tuple[int, str]]:
    """(index, word) of the records of ``corpora`` whose word is in ``words``, words in sorted order; ``per_word`` > 0
    draws that many per word with ``random.Random(f"{word}-{seed}")`` (kept in record order), 0 keeps all."""
    by = collections.defaultdict(list)
    words = set(words)
    for k, r in enumerate(records):
        w = word_of(r.get("text") or "")
        if r.get("corpus") in corpora and w in words:
            by[w].append(k)
    out = []
    for w in sorted(by):
        ks = by[w] if per_word <= 0 else sorted(random.Random(f"{w}-{seed}").sample(by[w], min(per_word, len(by[w]))))
        out += [(k, w) for k in ks]
    return out


def bootstrap(values, groups=None, *, n: int, seed: int, name: str) -> dict | None:
    """Mean and 95% percentile bootstrap interval of ``values``: over clips, or over groups (words) of clips when
    ``groups`` is given (each draw takes whole groups with replacement; the mean is over the drawn clips). The draws
    are seeded by ``seed`` and ``name``, so a statistic does not depend on the order statistics are computed in."""
    x = np.asarray(values, dtype=float)
    if not len(x):
        return None
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    if groups is None:
        means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    else:
        keys = sorted(set(groups))
        index = {g: i for i, g in enumerate(keys)}
        g = np.array([index[v] for v in groups])
        sums, counts = np.bincount(g, weights=x, minlength=len(keys)), np.bincount(g, minlength=len(keys))
        pick = rng.integers(0, len(keys), (n, len(keys)))
        means = sums[pick].sum(1) / counts[pick].sum(1)
    return dict(mean=round(float(x.mean()), 4), ci95=[round(float(np.quantile(means, .025)), 4),
                                                      round(float(np.quantile(means, .975)), 4)], n=int(len(x)),
                **({"groups": len(set(groups))} if groups is not None else {}))


def summarise(rows, model: str, *, n: int, seed: int, name: str) -> dict:
    """One checkpoint's retrieval on a set's rows: top-1, top-5, mean rank (word-clustered), gain over empty, NLL per
    part-valid code; and the unweighted ranks' top-5."""
    words = [r["word"] for r in rows]
    rank = np.array([r[model]["rank"] for r in rows])
    unweighted = np.array([r[model]["rank_unweighted"] for r in rows])
    b = lambda v, key: bootstrap(v, words, n=n, seed=seed, name=f"{name}/{model}/{key}")
    return dict(top1=b(rank <= 1, "top1"), top5=b(rank <= 5, "top5"), mean_rank=b(rank, "mean_rank"),
                gain_over_empty=b([r[model]["gain_over_empty"] for r in rows], "gain"),
                nll_per_part_valid_code=b([r[model]["nll_per_part_valid_code"] for r in rows], "nll"),
                unweighted_top5=round(float((unweighted <= 5).mean()), 4) if len(rows) else None)


def paired(rows, a: str, b: str, *, n: int, seed: int, name: str) -> dict:
    """Per clip ``a`` minus ``b``: top-1 and top-5 hits, rank, NLL per part-valid code; word-clustered and clip
    intervals."""
    words = [r["word"] for r in rows]
    diffs = dict(top1=[int(r[a]["rank"] <= 1) - int(r[b]["rank"] <= 1) for r in rows],
                 top5=[int(r[a]["rank"] <= 5) - int(r[b]["rank"] <= 5) for r in rows],
                 mean_rank=[r[a]["rank"] - r[b]["rank"] for r in rows],
                 nll_per_part_valid_code=[r[a]["nll_per_part_valid_code"] - r[b]["nll_per_part_valid_code"]
                                          for r in rows])
    return {k: dict(word_cluster=bootstrap(d, words, n=n, seed=seed, name=f"{name}/paired/{k}/w"),
                    clips=bootstrap(d, n=n, seed=seed, name=f"{name}/paired/{k}/c")) for k, d in diffs.items()}


def set_report(rows, *, n: int, seed: int, name: str, candidates_per_clip: int) -> dict:
    words = sorted({r["word"] for r in rows})
    per_word = {w: dict(clips=sum(r["word"] == w for r in rows),
                        checkpoint_ranks=[r["checkpoint"]["rank"] for r in rows if r["word"] == w],
                        comparison_ranks=[r["comparison"]["rank"] for r in rows if r["word"] == w]) for w in words}
    return dict(clips=len(rows), words=len(words), signers=len({r["signer"] for r in rows}),
                chance=dict(top1=round(1 / candidates_per_clip, 4), top5=round(min(5, candidates_per_clip) /
                                                                               candidates_per_clip, 4),
                            mean_rank=(candidates_per_clip + 1) / 2),
                checkpoint=summarise(rows, "checkpoint", n=n, seed=seed, name=name),
                comparison=summarise(rows, "comparison", n=n, seed=seed, name=name),
                paired_checkpoint_minus_comparison=paired(rows, "checkpoint", "comparison", n=n, seed=seed, name=name),
                per_word=per_word)


# ---- the model's scores -------------------------------------------------------------------------------------------

def part_valid_scores(prior, tokenizer, tokens, mask, motion, loci):
    """(unweighted (B,), part-valid (B,), part-valid codes, token steps, codes per step) of each row's teacher-forced
    log-probability of the clip's valid codes: ``rvq_sequence_log_likelihood``'s path, and the same with every code
    weighted by ``part_code_weights`` (1 where its part was observed). Every row holds the same motion."""
    import torch
    import torch.nn.functional as F
    from signtranslator.generation.motion_rvq import part_code_weights
    from signtranslator.generation.text_to_motion import batch_seconds, diffusion_targets
    config = prior.config
    with torch.no_grad():
        unit = diffusion_targets(config.tokenizer.motion, motion)
        _, codes, _ = tokenizer.quantize(tokenizer.encode(unit))
        valid = tokenizer.token_valid(motion.frame_valid)
        hidden, _, text_mask = prior.encode(tokens, mask, loci)
        start = torch.full_like(codes[:, :1], prior.K)
        previous = torch.cat((start, codes[:, :-1]), 1)
        seconds = batch_seconds(motion).float()
        step_seconds = seconds[:, ::config.tokenizer.step][:, :codes.shape[1]]
        step_seconds = F.pad(step_seconds, (0, codes.shape[1] - step_seconds.shape[1]))
        duration = (seconds * motion.frame_valid).amax(1)
        h = prior.temporal_states(hidden, text_mask, previous, step_seconds, duration)
        target = codes[valid]
        lp = prior.depth_logits(h[valid], target[:, :-1]).log_softmax(-1).gather(-1, target[..., None])[..., 0]
        weight = part_code_weights(tokenizer, config, motion, valid).to(lp.dtype)                  # (N, J)
        row = torch.arange(len(codes), device=codes.device)[:, None].expand_as(valid)[valid]
        zeros = lambda: torch.zeros(len(codes), dtype=lp.dtype, device=lp.device)
        total = zeros().index_add_(0, row, lp.sum(1))
        weighted = zeros().index_add_(0, row, (lp * weight).sum(1))
        n_valid = float(weight[row == 0].sum())
    return (total.double().cpu().numpy(), weighted.double().cpu().numpy(), n_valid, int(valid[0].sum()),
            int(codes.shape[2]))


def load_prior(store, key: str, expected_sha256: str | None, device):
    """(prior, tokenizer, provenance) of a stored prior checkpoint (EMA weights when present), checked against its
    SHA-256 when given."""
    import torch
    from signtranslator.generation.motion_rvq import MotionRVQ, RVQPrior, RVQPriorConfig
    payload = store.get(key)
    sha = hashlib.sha256(payload).hexdigest()
    if expected_sha256 and sha != expected_sha256:
        raise SystemExit(f"{key}: SHA-256 {sha}, expected {expected_sha256}")
    ck = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    prior = RVQPrior(RVQPriorConfig.from_dict(ck["config"]))
    prior.load_state_dict(ck.get("ema") or ck["model"])
    tokenizer = MotionRVQ(prior.config.tokenizer)
    tokenizer.load_state_dict(ck["tokenizer_state"])
    prior.to(device).eval(); tokenizer.to(device).eval()
    state_sha = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for _, v in sorted(ck["tokenizer_state"].items())))
    return prior, tokenizer, dict(key=key, sha256=sha, step=ck.get("step"), tokenizer_key=ck.get("tokenizer_key"),
                                  tokenizer_state_sha256=state_sha.hexdigest(), weights="ema" if ck.get("ema") else
                                  "model", locus_conditioning=bool(prior.config.locus_conditioning))


def score_clip(prior, tokenizer, state, cands, device) -> dict:
    """One clip under its candidates (own word first) and the empty text: ranks, the own word's gain over the empty
    text, NLL per part-valid code and the part-valid candidate scores."""
    import torch
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.generation.motion_tokens import _empty_like
    from signtranslator.generation.text_to_motion import PAD, encode_texts
    from signtranslator.grammar_motion.loci_conditioning import state_loci
    from train_text_to_motion import to_device
    cand, _ = encode_texts([w.lower() for w in cands])
    tokens = torch.cat((cand, _empty_like(cand[:1])), 0)
    mask = tokens != PAD
    n = len(cands)
    motion = collate_multichannel([state] * (n + 1))
    loci = torch.as_tensor(np.stack([state_loci(state)] * (n + 1)), dtype=torch.float32).to(device) \
        if prior.config.locus_conditioning else None
    tk, mk, mo = to_device((tokens, mask, motion, None), device)[:3]
    total, weighted, n_valid, steps, J = part_valid_scores(prior, tokenizer, tk, mk, mo, loci)
    return dict(rank=rank_own(weighted[:n]), rank_unweighted=rank_own(total[:n]),
                gain_over_empty=round(float(weighted[0] - weighted[n]), 4),
                nll_per_part_valid_code=round(float(-weighted[0] / max(n_valid, 1.)), 5),
                part_valid_codes=n_valid, steps=steps, codes_per_step=J,
                part_valid_scores=np.round(weighted[:n], 3).tolist(), empty_part_valid=round(float(weighted[n]), 3),
                _unweighted=total, _inputs=(tk, mk, mo, loci))


def check_against_repository(prior, tokenizer, scored: dict) -> None:
    """The unweighted scores equal ``rvq_sequence_log_likelihood`` (the repository's word-retrieval measure)."""
    from signtranslator.generation.motion_rvq import rvq_sequence_log_likelihood
    ref = rvq_sequence_log_likelihood(prior, tokenizer, *scored["_inputs"]).double().cpu().numpy()
    if not np.allclose(ref, scored["_unweighted"], atol=1e-3, rtol=1e-6):
        raise AssertionError(f"part_valid_scores disagrees with rvq_sequence_log_likelihood: {ref[:3]} vs "
                             f"{scored['_unweighted'][:3]}")


# ---- the run ------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint-key", required=True, help="the prior under test (e.g. W14b's pilot checkpoint)")
    p.add_argument("--checkpoint-sha256")
    p.add_argument("--compare-checkpoint-key", required=True, help="the comparison prior (e.g. F12 step 60000)")
    p.add_argument("--compare-checkpoint-sha256")
    p.add_argument("--mix-key", required=True, help="the manifest the clips come from (round 14b: round14b-mixed-v1)")
    p.add_argument("--mix-sha256", required=True)
    p.add_argument("--word-split-key", default=WORD_SPLIT, help="the R5 covered/uncovered split (round 14's)")
    p.add_argument("--word-split-sha256", default=WORD_SPLIT_SHA256)
    p.add_argument("--semlex-corpus", default=SEMLEX_CORPUS, help="the Sem-Lex corpus of the uncovered words' clips")
    p.add_argument("--covered-corpora", default=",".join(COVERED_CORPORA))
    p.add_argument("--covered-per-word", type=int, default=1)
    p.add_argument("--per-word", type=int, default=0, help="uncovered clips per word (0: all)")
    p.add_argument("--candidates", type=int, default=49)
    p.add_argument("--order-seed", default="r14diag", help="the candidate order's seed string (the diagnosis's)")
    p.add_argument("--seed", type=int, default=14, help="per-word clip draws and the bootstrap")
    p.add_argument("--bootstrap", type=int, default=4000)
    p.add_argument("--max-seconds", type=float, default=20., help="the trainer's clip-length limit")
    p.add_argument("--limit", type=int, default=0, help="smoke: at most this many clips per set")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--store", default="r2", help="'r2' or 'dir:/path' (tests)")
    p.add_argument("--cache-dir", help="state cache (default: a temporary directory)")
    p.add_argument("--output-key", help="the JSON report in the store (written once)")
    p.add_argument("--output-file", help="the JSON report on disk (written once)")
    args = p.parse_args(argv)
    if not (args.output_key or args.output_file):
        p.error("give --output-key or --output-file")
    if args.candidates < 2 or args.covered_per_word < 1 or args.per_word < 0 or args.bootstrap < 10:
        p.error("--candidates >= 2, --covered-per-word >= 1, --per-word >= 0, --bootstrap >= 10")
    covered_corpora = tuple(c for c in args.covered_corpora.split(",") if c)
    if args.semlex_corpus in covered_corpora or TEMPLATE_CORPUS in covered_corpora + (args.semlex_corpus,):
        p.error("--semlex-corpus and --covered-corpora must be distinct isolated-sign corpora")
    if args.output_file and Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    import warnings
    warnings.simplefilter("ignore")
    import torch
    from signtranslator.data.text_motion_dataset import TextMotionDataset
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest
    from train_motion_tokens import layout_template
    from train_text_to_motion import make_store
    clock = time.time()
    store = make_store(args.store)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
    device = torch.device(args.device)
    split_payload = store.get(args.word_split_key)
    split_sha = hashlib.sha256(split_payload).hexdigest()
    if args.word_split_sha256 and split_sha != args.word_split_sha256:
        raise SystemExit(f"{args.word_split_key}: SHA-256 {split_sha}, expected {args.word_split_sha256}")
    covered, uncovered = read_split(split_payload)
    words = sorted(covered + uncovered)
    if args.candidates > len(words):
        raise SystemExit(f"--candidates {args.candidates}: the split has {len(words)} words")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    present = {r.get("corpus") for r in manifest["records"]}
    if args.semlex_corpus not in present or not set(covered_corpora) <= present:
        raise SystemExit(f"the mix lacks {sorted(({args.semlex_corpus} | set(covered_corpora)) - present)}")
    semlex_train = collections.Counter(word_of(r["text"]) for r in manifest["records"] if r.get("text") and
                                       r.get("corpus") == args.semlex_corpus and r.get("partition") == "train")
    checkpoint = load_prior(store, args.checkpoint_key, args.checkpoint_sha256, device)
    comparison = load_prior(store, args.compare_checkpoint_key, args.compare_checkpoint_sha256, device)
    if checkpoint[2]["tokenizer_state_sha256"] != comparison[2]["tokenizer_state_sha256"]:
        raise SystemExit(f"the checkpoints' tokenizers differ ({checkpoint[2]['tokenizer_state_sha256'][:12]} vs "
                         f"{comparison[2]['tokenizer_state_sha256'][:12]}): their codes, and so their retrieval "
                         "scores, are not comparable clip for clip")
    models = dict(checkpoint=checkpoint, comparison=comparison)
    with tempfile.TemporaryDirectory() as t:
        cache = ContentCache(Path(args.cache_dir or t), 8 * 2 ** 30)
        template_record = next((r for r in manifest["records"] if r.get("corpus") == TEMPLATE_CORPUS and
                                r.get("partition") == "train"), None)
        if template_record is None or template_record.get("partition") in FORBIDDEN_PARTITIONS:
            raise SystemExit(f"no {TEMPLATE_CORPUS} train record for the layout template")
        template = layout_template(manifest, TEMPLATE_CORPUS, cache, store.get)
        val = TextMotionDataset(manifest, partition=READ_PARTITION, cache=cache, fetch=store.get,
                                max_seconds=args.max_seconds, fps=24., template=template)
        chosen = dict(uncovered=select(val.records, (args.semlex_corpus,), uncovered, args.per_word, args.seed),
                      covered=select(val.records, covered_corpora, covered, args.covered_per_word, args.seed))
        order = candidate_order(words, args.order_seed)
        sets = {}
        for name in SETS:
            items = chosen[name][:args.limit] if args.limit else chosen[name]
            rows, start = [], time.time()
            for i, (k, w) in enumerate(items):
                record = val.records[k]
                if record.get("partition") != READ_PARTITION or record.get("corpus") == TEMPLATE_CORPUS:
                    raise AssertionError(f"{record.get('sign')}: not an isolated-sign val record")
                text, state = val[k]
                cands = candidates(w, order, args.candidates)
                row = dict(word=w, text=text, corpus=record["corpus"], sign=record.get("sign"),
                           state_sha256=record["continuous"]["state_sha256"],
                           signer=(record.get("source") or {}).get("signer_id") or record.get("signer_component") or
                           (record.get("performer_sha256") or "")[:12],
                           semlex_train_records=semlex_train.get(w, 0), candidates=cands)
                for role, (prior, tokenizer, _) in models.items():
                    scored = score_clip(prior, tokenizer, state, cands, device)
                    if i < 2:
                        check_against_repository(prior, tokenizer, scored)
                        row[f"{role}_checked_vs_rvq_sequence_log_likelihood"] = True
                    row[role] = {k_: v for k_, v in scored.items() if not k_.startswith("_")}
                rows.append(row)
                if i % 20 == 0:
                    print(name, i, len(items), round(time.time() - start, 1), flush=True)
            report = set_report(rows, n=args.bootstrap, seed=args.seed, name=name, candidates_per_clip=args.candidates)
            if name == "uncovered":
                for sub, keep in (("newly_covered", lambda r: r["semlex_train_records"] > 0),
                                  ("still_uncovered", lambda r: r["semlex_train_records"] == 0)):
                    part = [r for r in rows if keep(r)]
                    report[sub] = {k_: v for k_, v in set_report(
                        part, n=args.bootstrap, seed=args.seed, name=f"{name}/{sub}",
                        candidates_per_clip=args.candidates).items() if k_ != "per_word"} if part else None
                report["words_without_val_clips"] = sorted(set(uncovered) - {r["word"] for r in rows})
            report["seconds"] = round(time.time() - start, 1)
            report["rows"] = rows
            sets[name] = report
            print(name, json.dumps({k_: report[k_]["top5"] for k_ in ("checkpoint", "comparison")}),
                  json.dumps(report["paired_checkpoint_minus_comparison"]["top5"]), flush=True)
    out = dict(
        purpose="teacher-forced part-valid word retrieval of a prior against a comparison prior (round 14b pilot "
                "add-on; report-only)",
        measure=__doc__.split("\n\n")[3].replace("\n", " "),
        inputs=dict(mix_key=args.mix_key, mix_sha256=args.mix_sha256, word_split_key=args.word_split_key,
                    word_split_sha256=split_sha, semlex_corpus=args.semlex_corpus,
                    covered_corpora=list(covered_corpora), template=dict(corpus=TEMPLATE_CORPUS,
                                                                         sign=template_record.get("sign"),
                                                                         partition=template_record.get("partition"))),
        checkpoints={role: m[2] for role, m in models.items()},
        same_tokenizer=True,                           # refused otherwise (above)
        parameters=dict(candidates=args.candidates, order_seed=args.order_seed, seed=args.seed,
                        covered_per_word=args.covered_per_word, per_word=args.per_word, bootstrap=args.bootstrap,
                        max_seconds=args.max_seconds, limit=args.limit or None, device=str(device), fps=24.,
                        prompt_form="word.lower()"),
        candidate_order=order, words=dict(covered=len(covered), uncovered=len(uncovered)),
        hygiene=f"only {READ_PARTITION} records of {[args.semlex_corpus, *covered_corpora]} and the layout template "
                f"(first {TEMPLATE_CORPUS} train record) are read; no test or held-out partition, no other How2Sign "
                "clip (signer 9 is in none of them)",
        sets=sets, seconds=round(time.time() - clock, 1))
    payload = json.dumps(out, indent=1, sort_keys=True, default=float).encode()
    if args.output_file:
        with open(args.output_file, "xb") as stream:
            stream.write(payload)
    if args.output_key:
        if store.exists(args.output_key):
            raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
        store.put(args.output_key, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
