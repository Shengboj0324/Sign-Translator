"""Training pairs of the sentence refiner (docs/NEXT_PHASE_PLAN.md, "Next lever after the oracle finding", option 1;
``signtranslator.generation.refiner``). Built and tested only; it runs once declared.

**Sentences.** How2Sign *train* sentences of the mix (``--mix-key``, round10-mixed-v1 by default), and only those
(``sentence_jobs``): corpus ``how2sign-two-view-text-v1``, ``partition`` and any ``source_partition`` both ``train``,
How2Sign signer 9 and records of unknown component excluded (``recognizer_corpora.is_how2sign_test_signer``) and every
kept record passed through ``recognizer_corpora.check_loadable`` (never a test or held-out record, never signer 9)
before it is read; a train sentence whose normalised English is also a How2Sign val, test or held-out sentence is
excluded too, so no evaluation reference is ever a target. Sentences without usable continuous channels, without a
stage-1 word or longer than ``--max-seconds`` are skipped and counted.

**Selection signers.** The train components named by ``--selection-components``, or by default the smallest train
components (fewest sentences, ties by name) taken until they hold ``--selection-share`` of the sentences (never every
component): their pairs are the refiner's checkpoint-selection set (``split: selection``), the rest ``split: train``.
The split is part of the configuration (``config.json``).

**Inputs** (K = ``--k`` per sentence, default 4): the round-15 real-word composition exactly as the
``oracle_stage2`` form of ``round15_compose_eval.py`` makes it at ``--selection stage1 --timing sign-fitted
--core-margin 0.1``, with this sentence as the reference:
- stage-1 word selection from the sentence's English (``compositional.selection_details`` on the training vocabulary
  of ``--corpora``: by default ``compositional.isolated_corpora`` of the mix, round 15's ASLLVD, RIT, WLASL (and Sem-Lex
  where the mix has it); any ASL Citizen corpus is refused);
- one real isolated *train* clip per word from those corpora (``round15_oracle.word_pools``: train partition only),
  drawn uniformly by a generator seeded per (sentence, k) (``draw_seed``: SHA-256 of the sentence identity and k), word
  by word in order; every drawn record is checked again (``check_word_record``);
- cores, plan and rendering by ``round15_stage2.oracle_material`` and ``render_form`` with the sign-fitted timing:
  annotated sign cores plus the margin, the reference's frames shared in proportion to the signs' isolated annotated
  durations and the stage-2 model's gap ratio (the model pinned by ``--expect-model-sha256``, fitted on ASLLRP train
  only, or ``--model-file``); joins interpolated;
- the clips and the sentence are served as the prior's training dataset serves them (``build_recon_states.serve``:
  usable channels, trim window, conformed to the template, 24 fps).

**Codes.** Inputs and target are tokenized with ``--tokenizer-key`` (F12's tokenizer, ``tokens-rvq-d0b-r3d-d2`` step
5000, SHA-256 pinned; checked tensor by tensor against ``--prior-checkpoint-key``'s ``tokenizer_state``), all stages,
on the CPU with one thread per worker (``refiner.tokenize_state``). A pair whose input and target lengths differ is
refused.

**Store layout, write-once, hash-indexed:** ``<prefix>/config.json`` (a later run with another configuration is
refused), ``<prefix>/pairs/<identity>.bin`` (one sentence: ``refiner.pack_pair``, deterministic bytes; identity the
SHA-256 of corpus, sign and state hash), ``<prefix>/index/shard-III-of-NNN.json`` (every sentence of the shard with its
status, split and the pair file's SHA-256) and, with ``--merge``, ``<prefix>/index.json``. **Resumable:** a shard with
an index is skipped; an existing pair file is re-hashed and unpacked, never rewritten. **Deterministic:** draws,
plans and codes depend only on the sentence and the configuration, not on the sharding (tested).
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

SCHEMA = "refiner-pairs-config-v1"
FPS = 24.
SENTENCE_CORPUS = "how2sign-two-view-text-v1"
EVAL_MIX = "acquisition/2026-10-17-v1/youtube-asl-v1/round10-mixed-v1/manifest.json"
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"
TOKENIZER_KEY = "training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt"
TOKENIZER_SHA256 = "021ee00a777261f915c2738efb671115c71eaf790f953ffc791931761dda08de"
PRIOR_KEY = "training-runs/tokens-rvq-d0b-prior-r12-f12/checkpoints/step-00060000.pt"
MODEL_SHA256 = "42389ca48d48ac9923843557d66cd3e98c0faf0d5df54a3d5752ddb41f4e6f21"
DONE = ("converted", "already_present")
_WORKER = {}


# ---- pure logic --------------------------------------------------------------------------------------------------

def normalised_text(text) -> str:
    from signtranslator.generation.compositional import normalise
    return " ".join(normalise(text))


def sentence_jobs(manifest, corpus: str = SENTENCE_CORPUS) -> tuple[list[dict], collections.Counter]:
    """(the train sentences that may become pairs, sorted by (sign, state hash); counts of every exclusion). See the
    module docstring; every kept record has passed ``check_loadable``."""
    from signtranslator.data_engineering.corpus_manifest import usable_channels
    from signtranslator.generation.recognizer_corpora import check_loadable, is_how2sign_test_signer
    held = {normalised_text(r.get("text")) for r in manifest["records"] if r.get("corpus") == corpus
            and (r.get("partition") != "train" or r.get("source_partition", "train") != "train")} - {""}
    jobs, counts, seen = [], collections.Counter(), set()
    for r in manifest["records"]:
        if r.get("corpus") != corpus:
            continue
        partition, source = r.get("partition"), r.get("source_partition", r.get("partition"))
        if partition != "train" or source != "train":
            counts[f"not_train:{partition}/{source}"] += 1
            continue
        if is_how2sign_test_signer(r):
            counts["signer9_or_unknown_component"] += 1
            continue
        if not r.get("text") or not normalised_text(r["text"]):
            counts["no_text"] += 1
            continue
        if normalised_text(r["text"]) in held:
            counts["text_in_val_or_test"] += 1
            continue
        if not usable_channels(r, "continuous"):
            counts["no_usable_channels"] += 1
            continue
        check_loadable(r)                                   # a hard guard, also under python -O
        if r["sign"] in seen:
            raise SystemExit(f"{r['sign']}: listed twice")
        seen.add(r["sign"])
        jobs.append(r)
        counts["train_sentences"] += 1
    jobs.sort(key=lambda r: (r["sign"], r["continuous"]["state_sha256"]))
    return jobs, counts


def selection_components(jobs, share: float = .05, given=None) -> list[str]:
    """The components whose sentences form the selection set: ``given`` (each must hold train sentences, and not
    all of them), else the smallest ones (fewest sentences, ties by name) until they hold ``share`` of the
    sentences, at least one and never every component."""
    sizes = collections.Counter(r["signer_component"] for r in jobs)
    if len(sizes) < 2:
        raise SystemExit(f"the train sentences come from {len(sizes)} component(s): no signer can be held out")
    if given:
        unknown = sorted(set(given) - set(sizes))
        if unknown or set(given) >= set(sizes):
            raise SystemExit(f"--selection-components {sorted(given)}: unknown {unknown} or every train component")
        return sorted(set(given))
    total, chosen = sum(sizes.values()), []
    for c, n in sorted(sizes.items(), key=lambda kv: (kv[1], kv[0])):
        if chosen and sum(sizes[x] for x in chosen) >= share * total:
            break
        if len(chosen) == len(sizes) - 1:
            break
        chosen.append(c)
    return sorted(chosen)


def identity(record) -> str:
    return hashlib.sha256(f"{record['corpus']}|{record['sign']}|{record['continuous']['state_sha256']}"
                          .encode()).hexdigest()


def pair_key(prefix: str, record) -> str:
    return f"{prefix}/pairs/{identity(record)}.bin"


def shard_key(prefix: str, shard: int, shards: int) -> str:
    return f"{prefix}/index/shard-{shard:03d}-of-{shards:03d}.json"


def draw_seed(record, k: int) -> int:
    """The seed of input k's clip draws: the first 8 bytes of a SHA-256 of the sentence identity and k."""
    digest = hashlib.sha256(f"refiner-pairs|{identity(record)}|{int(k)}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def draw_records(pools, record, k: int, words) -> list[dict]:
    """One train clip per word, in order, uniformly from the word's pool, by one generator seeded ``draw_seed``."""
    rng = np.random.default_rng(draw_seed(record, k))
    out = []
    for w in words:
        pool = pools.get(w)
        if not pool:
            raise KeyError(f"{w!r} has no training record")
        out.append(pool[int(rng.integers(len(pool)))])
    return out


def check_corpora(corpora) -> tuple[str, ...]:
    """The word corpora, refusing ASL Citizen (recognition-only) and the sentence corpus."""
    bad = [c for c in corpora if "citizen" in c.lower() or c == SENTENCE_CORPUS]
    if bad or not corpora:
        raise SystemExit(f"--corpora: {bad or 'none'} may not provide words (ASL Citizen is recognition-only)")
    return tuple(corpora)


class HygieneError(RuntimeError):
    """A record that may never enter a pair reached one: the run stops at once (never retried, never skipped)."""


def check_readable(record) -> None:
    """``recognizer_corpora.check_loadable`` as a ``HygieneError``."""
    from signtranslator.generation.recognizer_corpora import check_loadable
    try:
        check_loadable(record)
    except RuntimeError as error:
        raise HygieneError(str(error)) from error


def check_word_record(record, corpora) -> None:
    """Raise ``HygieneError`` unless a drawn word clip is a train record of an allowed corpus that may be read."""
    if record.get("corpus") not in corpora or "citizen" in str(record.get("corpus")).lower():
        raise HygieneError(f"{record.get('sign')}: corpus {record.get('corpus')!r} may not provide words")
    if record.get("partition") != "train" or record.get("source_partition", "train") != "train":
        raise HygieneError(f"{record.get('sign')}: a {record.get('partition')} clip may not provide words")
    check_readable(record)


def split_of(record, selection) -> str:
    return "selection" if record["signer_component"] in set(selection) else "train"


# ---- one sentence ------------------------------------------------------------------------------------------------

class LRU(collections.OrderedDict):
    """A bounded mapping (the loaded word clips): the least recently used entry goes first."""

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.size:
            self.popitem(last=False)


def entry_of(job, **fields) -> dict:
    return dict(corpus=job["corpus"], sign=job["sign"], state_sha256=job["continuous"]["state_sha256"],
                signer_component=job["signer_component"], **fields)


def build_pair(job, *, context, fetch, load) -> tuple[bytes | None, dict]:
    """(pair bytes or None, entry fields) of one sentence: stage-1 words, K sign-fitted real-word compositions fitted
    to the served sentence, all tokenized. None with a status when the sentence is skipped."""
    import round15_stage2 as r2
    from signtranslator.generation.compositional import CORE_RULE
    from signtranslator.generation.refiner import pack_pair, pair_arrays, tokenize_state
    from build_recon_states import serve
    from round15_compose_eval import stage1_selection
    c = context
    details, words = stage1_selection([job["text"]], c["vocab"])
    details, words = details[0], words[0]
    if not words:
        return None, dict(status="no_words")
    target = serve(job, c["template"], cache=c["cache"], fetch=fetch)
    frames = len(target.timestamps)
    seconds = float(target.timestamps[-1] - target.timestamps[0]) + 1 / FPS
    if seconds > c["max_seconds"]:
        return None, dict(status="too_long", frames=frames)
    plan = r2.plan_words(words, c["model"], c["vocab"], rows=details)
    composition = dict(mode="oracle_stage2", words=list(plan["entries"]), generate=list(plan["generate"]),
                       targets=list(plan["targets"]), weights=list(plan["weights"]), seeds=[], block="matched",
                       overlap=4, durations=[None] * len(plan["entries"]))
    rule = dict(CORE_RULE, margin=c["margin"])
    sources, inputs = [], []
    for k in range(c["k"]):
        records = draw_records(c["pools"], job, k, composition["words"])
        for r in records:
            check_word_record(r, c["corpora"])
        material = r2.oracle_material(composition, 0, c["pools"], loaded=c["loaded"], load=load, rule=rule,
                                      timing="sign-fitted", fractions=c["fractions"], margin=c["margin"],
                                      records=records)
        state, fp = r2.render_form(material, composition, seconds, c["model"], timing="sign-fitted",
                                   sample_id=f"refiner-pair:{job['sign']}:{k}")
        if len(state.timestamps) != frames:
            raise ValueError(f"composition {k} has {len(state.timestamps)} frames, the sentence {frames}")
        sources.append(tokenize_state(c["tokenizer"], state))
        inputs.append(dict(k=k, seed=draw_seed(job, k),
                           records=[[r["corpus"], r["sign"], r["continuous"]["state_sha256"]] for r in records],
                           cores=[[s["lo"], s["hi"], s["reason"]] for s in material["segments"]],
                           plan=dict(cores=list(fp["cores"]), gaps=list(fp["gaps"]), lead=fp.get("lead"),
                                     trail=fp.get("trail"))))
    tokens = tokenize_state(c["tokenizer"], target)
    meta = dict(corpus=job["corpus"], sign=job["sign"], state_sha256=job["continuous"]["state_sha256"],
                text=job["text"], signer_component=job["signer_component"], split=split_of(job, c["selection"]),
                words=composition["words"], frames=frames, config_sha256=c["config_sha256"], inputs=inputs)
    payload = pack_pair(pair_arrays(tokens, sources), meta)
    return payload, dict(status="converted", frames=frames, words=len(composition["words"]))


def process_once(job, prefix, context) -> dict:
    from signtranslator.generation.refiner import unpack_pair
    from round15_oracle import word_clip
    store = context["store"]
    check_readable(job)
    if job.get("partition") != "train" or job.get("source_partition", "train") != "train":
        raise HygieneError(f"{job.get('sign')}: a {job.get('partition')} sentence may not become a pair")
    key = pair_key(prefix, job)
    split = split_of(job, context["selection"])
    if store.exists(key):                                  # write-once: re-hash and unpack, never rewrite
        payload = store.get(key)
        _, meta = unpack_pair(payload)
        if meta["sign"] != job["sign"] or meta["config_sha256"] != context["config_sha256"]:
            raise SystemExit(f"{key} holds another sentence or configuration")
        return entry_of(job, status="already_present", split=split, pair_key=key,
                        pair_sha256=hashlib.sha256(payload).hexdigest(), frames=meta["frames"],
                        words=len(meta["words"]))
    part = job["continuous"]
    data = store.get(part["state_key"])                    # transient failures raise here and are retried
    if hashlib.sha256(data).hexdigest() != part["state_sha256"]:
        raise IOError(f"{part['state_key']}: fetched bytes differ from the manifest's SHA-256")

    def fetched(k):
        if k != part["state_key"]:
            raise IOError(f"{k}: unexpected fetch")
        return data
    load = lambda r: word_clip(r, cache=context["cache"], fetch=store.get, template=context["template"])  # noqa: E731
    try:
        payload, fields = build_pair(job, context=context, fetch=fetched, load=load)
    except (ValueError, KeyError) as error:                # a clip or the sentence refused by load, conform or plan
        return entry_of(job, status="refused", split=split, reason=f"{type(error).__name__}: {error}"[:300])
    if payload is None:
        return entry_of(job, split=split, **fields)
    if store.exists(key):                                  # another pass wrote it meanwhile: keep what is there
        return process_once(job, prefix, context)
    store.put(key, payload)
    return entry_of(job, split=split, pair_key=key, pair_sha256=hashlib.sha256(payload).hexdigest(), **fields)


def _init(context_path: str):
    import torch
    from signtranslator.data_engineering.corpus_manifest import ContentCache
    from signtranslator.generation.refiner import state_dict_digest
    from signtranslator.generation.motion_rvq import MotionRVQ, MotionRVQConfig
    from signtranslator.pose.multichannel import MultichannelMotion
    from train_text_to_motion import make_store
    torch.set_num_threads(1)
    with open(context_path, "rb") as stream:
        c = pickle.load(stream)
    tokenizer = MotionRVQ(MotionRVQConfig.from_dict(c.pop("tokenizer_config")))
    tokenizer.load_state_dict(c.pop("tokenizer_weights"))
    if state_dict_digest(tokenizer.state_dict()) != c["tokenizer_digest"]:
        raise SystemExit("the worker's tokenizer differs from the configuration's")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "template.npz"
        path.write_bytes(c.pop("template_bytes"))
        template = MultichannelMotion.load(path, expected_sha256=c["template_state_digest"])
    _WORKER.clear()
    _WORKER.update(c, store=make_store(c["store_spec"]), tokenizer=tokenizer.eval().requires_grad_(False),
                   template=template, cache=ContentCache(Path(c["cache_dir"]) / "content", 4 * 2 ** 30),
                   loaded=LRU(c["clip_cache"]))


def process(job, prefix: str, attempts: int = 4) -> dict:
    """One sentence, transient failures retried; a sentence that still fails is ``error`` (rerun the shard)."""
    import random
    for attempt in range(attempts):
        try:
            return process_once(job, prefix, _WORKER)
        except (SystemExit, HygieneError):
            raise
        except Exception as error:  # noqa: BLE001
            if attempt == attempts - 1:
                return entry_of(job, status="error", reason=f"{type(error).__name__}: {error}"[:300])
            time.sleep(min(60., 2. * 2 ** attempt) * (.5 + random.random()))


# ---- main --------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2", help="r2, or dir:/path (tests)")
    p.add_argument("--prefix", required=True, help="write-once store prefix of the pairs and their index")
    p.add_argument("--mix-key", default=EVAL_MIX)
    p.add_argument("--mix-sha256", default=EVAL_MIX_SHA)
    p.add_argument("--corpora", help="the isolated word corpora (default: compositional.isolated_corpora of the mix)")
    p.add_argument("--tokenizer-key", default=TOKENIZER_KEY)
    p.add_argument("--tokenizer-sha256", default=TOKENIZER_SHA256)
    p.add_argument("--prior-checkpoint-key", default=PRIOR_KEY, help="the prior the refiner warm-starts from: its "
                   "tokenizer_state must equal the tokenizer, tensor by tensor")
    p.add_argument("--model-file", help="a fitted stage-2 model (default: fit it on ASLLRP train now)")
    p.add_argument("--expect-model-sha256", default=MODEL_SHA256, help="the declared stage-2 model ('' : any)")
    p.add_argument("--k", type=int, default=4, help="composed inputs per sentence")
    p.add_argument("--core-margin", type=float, default=.1)
    p.add_argument("--max-seconds", type=float, default=20.)
    p.add_argument("--selection-components", help="comma-separated train components held out for checkpoint "
                   "selection (default: the smallest components holding --selection-share of the sentences)")
    p.add_argument("--selection-share", type=float, default=.05)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--clip-cache", type=int, default=4000, help="word clips kept loaded per worker")
    p.add_argument("--limit", type=int, help="smoke runs: only the first LIMIT sentences of the shard (no index)")
    p.add_argument("--cache-dir", default=str(Path(tempfile.gettempdir()) / "refiner-pairs"))
    p.add_argument("--merge", action="store_true", help="write <prefix>/index.json from every shard's index")
    args = p.parse_args(argv)
    if not 0 <= args.shard < args.shards:
        p.error("--shard must be in [0, --shards)")
    if args.k < 1 or not 0 < args.selection_share < 1:
        p.error("--k >= 1 and 0 < --selection-share < 1")
    import warnings
    warnings.simplefilter("ignore")
    import torch
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest, load_sample
    from signtranslator.generation.compositional import isolated_corpora, training_vocabulary
    from signtranslator.generation.recognizer_corpora import check_loadable
    from signtranslator.generation.refiner import state_dict_digest
    import round15_stage2_fit as fit
    from build_recon_states import load_tokenizer, state_bytes
    from round15_oracle import TEMPLATE_CORPUS, word_pools
    from train_text_to_motion import make_store
    prefix = args.prefix.rstrip("/")
    store = make_store(args.store)
    if args.merge:
        return merge(store, prefix, args.shards)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"
        path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    corpora = check_corpora(tuple(c for c in args.corpora.split(",") if c) if args.corpora
                            else isolated_corpora(manifest))
    # the stage-2 model (ASLLRP train only) before any How2Sign sentence is read
    if args.model_file:
        model = json.loads(Path(args.model_file).read_text())
        fit.check_model(model)
        if model.get("mix_key") != args.mix_key or model["inputs"].get(args.mix_key) != args.mix_sha256 \
                or list(model.get("corpora") or ()) != list(corpora):
            raise SystemExit("the stage-2 model was fitted on another vocabulary")
    else:
        model, _ = fit.fit_model(fit.agc._store(args.store), args.mix_key, args.mix_sha256, corpora)
    if args.expect_model_sha256 and model["sha256"] != args.expect_model_sha256:
        raise SystemExit(f"the stage-2 model is {model['sha256']}, not the declared {args.expect_model_sha256}")
    if not model["timing"].get("default_seconds"):
        raise SystemExit("the stage-2 model has no default_seconds: the sign-fitted gap ratio is undefined")
    vocab = training_vocabulary(manifest, corpora)
    pools = word_pools(manifest, corpora)
    fractions = fit.sign_fractions(manifest, vocab)
    # tokenizer, checked against the prior's
    payload = store.get(args.tokenizer_key)
    if hashlib.sha256(payload).hexdigest() != args.tokenizer_sha256:
        raise SystemExit(f"{args.tokenizer_key}: SHA-256 differs from --tokenizer-sha256")
    tokenizer, _ = load_tokenizer(payload, "cpu")
    weights = tokenizer.state_dict()
    digest = state_dict_digest(weights)
    prior = torch.load(io.BytesIO(store.get(args.prior_checkpoint_key)), map_location="cpu", weights_only=False)
    theirs = prior["tokenizer_state"]
    if set(weights) != set(theirs) or not all(torch.equal(weights[k], theirs[k].cpu()) for k in weights):
        raise SystemExit(f"the tokenizer differs from {args.prior_checkpoint_key}'s tokenizer_state")
    del prior, theirs, payload
    # template (the prior's --conform-to-corpus layout)
    t_record = next((r for r in manifest["records"] if r.get("corpus") == TEMPLATE_CORPUS
                     and r.get("partition") == "train"), None)
    if t_record is None:
        raise SystemExit(f"no training record of {TEMPLATE_CORPUS!r} for the layout template")
    check_loadable(t_record)
    template_bytes, template_digest = state_bytes(load_sample(t_record, "continuous",
                                                              cache=ContentCache(cache_dir / "content", 4 * 2 ** 30),
                                                              fetch=store.get))
    jobs, counts = sentence_jobs(manifest)
    given = [c for c in (args.selection_components or "").split(",") if c]
    selection = selection_components(jobs, args.selection_share, given or None)
    del manifest
    config = dict(schema=SCHEMA, mix_key=args.mix_key, mix_sha256=args.mix_sha256, sentence_corpus=SENTENCE_CORPUS,
                  corpora=list(corpora), vocabulary_sha256=hashlib.sha256(json.dumps(vocab, sort_keys=True).encode())
                  .hexdigest(), tokenizer_key=args.tokenizer_key, tokenizer_sha256=args.tokenizer_sha256,
                  tokenizer_digest=digest, prior_checkpoint_key=args.prior_checkpoint_key, model_sha256=model["sha256"],
                  selection="stage1", timing="sign-fitted", core_margin=args.core_margin, k=args.k,
                  max_seconds=args.max_seconds, fps=FPS, selection_components=selection,
                  selection_share=None if given else args.selection_share,
                  template=dict(corpus=TEMPLATE_CORPUS, sign=t_record["sign"],
                                state_sha256=t_record["continuous"]["state_sha256"], digest=template_digest),
                  sentence_counts=dict(sorted(counts.items())))
    config_payload = json.dumps(config, indent=1, sort_keys=True).encode()
    config_sha = hashlib.sha256(config_payload).hexdigest()
    config_key = f"{prefix}/config.json"
    if store.exists(config_key):
        if store.get(config_key) != config_payload:
            raise SystemExit(f"{config_key} holds another configuration: this prefix is written once")
    else:
        store.put(config_key, config_payload)
    out_key = shard_key(prefix, args.shard, args.shards)
    if args.limit is None and store.exists(out_key):
        print(f"{out_key} exists: shard done", flush=True)
        return 0
    shard = [j for i, j in enumerate(jobs) if i % args.shards == args.shard][:args.limit]
    print(f"{len(jobs)} train sentences ({json.dumps(dict(counts))}); selection components {selection}; shard "
          f"{args.shard}/{args.shards}: {len(shard)}", flush=True)
    context = dict(store_spec=args.store, cache_dir=str(cache_dir), clip_cache=args.clip_cache,
                   tokenizer_config=tokenizer.config.to_dict(), tokenizer_weights=weights, tokenizer_digest=digest,
                   template_bytes=template_bytes, template_state_digest=template_digest, vocab=vocab, pools=pools,
                   corpora=corpora, model=model, fractions=fractions, k=args.k, margin=args.core_margin,
                   max_seconds=args.max_seconds, selection=selection, config_sha256=config_sha)
    context_path = cache_dir / f"context-{config_sha}.pkl"
    with open(context_path, "wb") as stream:
        pickle.dump(context, stream)
    del context, pools
    results, started = [], time.monotonic()
    if args.workers <= 1:
        _init(str(context_path))
        iterator, pool = (process(j, prefix) for j in shard), None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(str(context_path),))
        iterator = pool.map(process, shard, [prefix] * len(shard), chunksize=4)
    try:
        for i, result in enumerate(iterator):
            results.append(result)
            if i % 200 == 0:
                print(f"{i}/{len(shard)} {result['status']} {time.monotonic() - started:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    status = dict(collections.Counter(r["status"] for r in results))
    print(json.dumps(dict(status=status, seconds=round(time.monotonic() - started))), flush=True)
    if status.get("error"):
        print("transient errors: no shard index written; run the shard again (it resumes)", flush=True)
        return 2
    if args.limit is not None:
        print("smoke run (--limit): no shard index written", flush=True)
        return 0
    index = dict(schema=SCHEMA, config_sha256=config_sha, shard=args.shard, shards=args.shards,
                 status_counts=status, entries=results)
    if store.exists(out_key):
        raise SystemExit(f"{out_key} appeared while this shard ran: not overwritten")
    store.put(out_key, json.dumps(index, sort_keys=True).encode())
    print(f"wrote {out_key}", flush=True)
    return 0


def merge(store, prefix: str, shards: int) -> int:
    """``<prefix>/index.json`` from every shard index (one configuration, every shard present), written once."""
    config_payload = store.get(f"{prefix}/config.json")
    config_sha = hashlib.sha256(config_payload).hexdigest()
    entries, parts = [], []
    for shard in range(shards):
        key = shard_key(prefix, shard, shards)
        if not store.exists(key):
            raise SystemExit(f"{key} is missing: run that shard first")
        payload = store.get(key)
        part = json.loads(payload)
        if part["config_sha256"] != config_sha or part["shards"] != shards or part["shard"] != shard:
            raise SystemExit(f"{key} was made with another configuration or sharding")
        parts.append(dict(key=key, sha256=hashlib.sha256(payload).hexdigest(), status_counts=part["status_counts"]))
        entries += part["entries"]
    signs = [e["sign"] for e in entries]
    if len(set(signs)) != len(signs):
        raise SystemExit("a sentence appears in two shards")
    entries.sort(key=lambda e: (e["sign"], e["state_sha256"]))
    done = [e for e in entries if e["status"] in DONE]
    index = dict(schema=SCHEMA, config=json.loads(config_payload), config_sha256=config_sha, shards=parts,
                 status_counts=dict(collections.Counter(e["status"] for e in entries)),
                 splits=dict(collections.Counter(e["split"] for e in done)), entries=entries)
    payload = json.dumps(index, sort_keys=True).encode()
    key = f"{prefix}/index.json"
    if store.exists(key):
        if store.get(key) != payload:
            raise SystemExit(f"{key} exists with other contents: written once")
    else:
        store.put(key, payload)
    print(json.dumps(dict(index=key, sha256=hashlib.sha256(payload).hexdigest(), entries=len(entries),
                          status_counts=index["status_counts"], splits=index["splits"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
