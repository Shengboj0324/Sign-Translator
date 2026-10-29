"""Tokenizer-decoded copies of recogniser training and validation clips, in the pilot's generated form (recogniser v3;
docs/NEXT_PHASE_PLAN.md, "Instrument finding and the recogniser v3 plan").

For every ``train`` / ``val`` record of the given manifests (``--manifest KEY@SHA``, repeated; the ``test`` partition
and How2Sign signer 9 are never read, ``recognizer_corpora.check_loadable``), the clip is served as the prior's
dataset serves it and round-tripped through F12's tokenizer exactly as ``tokenizer_domain_read.py``'s ``recon_full``
form, the form the pilot's generated words take:
1. ``corpus_manifest.load_sample`` (usable channels, trim window, mirror flag), ``layout_adapter.conform`` to the
   template (``--template``: round10-mixed-v1's first How2Sign training state, F12's ``--conform-to-corpus``), then
   24 fps when the rate differs by more than 1% (``evaluate_text_to_motion.at_rate``, the dataset's rule);
2. ``diffusion_targets`` -> ``tokenizer.encode`` -> ``tokenizer.quantize`` (all stages) -> codes;
3. ``compositional.make_decode`` with ``exposure_diagnostic.make_post(gains)``: the codes decoded and assembled as a
   generation (every trained channel valid), the tokenizer's accepted decoder gains, then head rotation and head
   translation smoothed over 0.1 s (``pilot_content_check.postprocess``, the round-3e post-processing).

**Tokenizer:** ``--tokenizer-key`` (F12's: ``tokens-rvq-d0b-r3d-d2`` step 5000, SHA-256 pinned), its EMA weights,
and the accepted gains of that key (``evaluate_text_to_motion.accepted_decoder_gains``). ``--prior-checkpoint-key``
optionally checks it tensor by tensor against the tokenizer a prior checkpoint carries (F12 step 60000).

**Store layout, write-once:** ``<prefix>/config.json`` (tokenizer, gains, template, form; a later shard with another
configuration is refused), ``<prefix>/states/<load digest>.npz`` (``recognizer_corpora.load_digest``: the record's
state, usable channels, window and mirror flag, so a copy is used only for the record it was made from; an existing
state is never rewritten, only re-hashed), ``<prefix>/index/shard-III-of-NNN.json`` (every record of the shard with
its status and hashes, written when the shard has no transient error) and, with ``--merge``, ``<prefix>/index.json``
(all shards, written once). **Resumable:** a shard with an index is skipped; a crashed shard is rerun and re-hashes
the states it already wrote. Runs on CPU (``--device cpu``, one torch thread per worker) or GPU.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

SCHEMA = "recon-states-v1"
TOKENIZER_KEY = "training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt"
TOKENIZER_SHA256 = "021ee00a777261f915c2738efb671115c71eaf790f953ffc791931761dda08de"
TEMPLATE = ("acquisition/2026-10-17-v1/youtube-asl-v1/round10-mixed-v1/manifest.json"
            "@ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a:how2sign-two-view-text-v1")
FORM = ("recon_full: tokenizer codes decoded as a generation (every trained channel valid), accepted decoder gains, "
        "head rotation and head translation smoothed 0.1 s (tokenizer_domain_read.read_forms / "
        "pilot_content_check.postprocess)")
PARTITIONS = ("train", "val")
STATUSES_DONE = ("converted", "already_present", "refused", "skipped")
_WORKER = {}


# ------------------------------------------------------------------------------------------------- pure helpers

def parse_manifest_spec(text: str) -> tuple[str, str]:
    if text.count("@") != 1:
        raise SystemExit(f"--manifest {text!r}: expected KEY@SHA")
    key, sha = text.split("@")
    if len(sha) != 64:
        raise SystemExit(f"--manifest {text!r}: {sha!r} is not a SHA-256")
    return key, sha


def shard_key(prefix: str, shard: int, shards: int) -> str:
    return f"{prefix}/index/shard-{shard:03d}-of-{shards:03d}.json"


def jobs_of(manifests, *, partitions=PARTITIONS, whole_clip=()) -> tuple[list[dict], collections.Counter]:
    """Every record to decode, sorted by (corpus, sign): (manifest, corpus) pairs in; each job a record copy with its
    ``corpus``. Test and How2Sign signer-9 records raise (``check_loadable``) only if they would be read; the test
    partition is dropped first, unread."""
    from signtranslator.data_engineering.corpus_manifest import usable_channels
    from signtranslator.generation.recognizer_corpora import FORBIDDEN_PARTITIONS, check_loadable, corpus_name
    if set(partitions) & set(FORBIDDEN_PARTITIONS):
        raise SystemExit(f"partitions {sorted(set(partitions) & set(FORBIDDEN_PARTITIONS))} are never read")
    jobs, counts, seen = [], collections.Counter(), set()
    for manifest in manifests:
        for r in manifest["records"]:
            name = corpus_name(manifest, r)
            if r.get("partition") not in partitions:
                counts[f"{name}:not_read:{r.get('partition')}"] += 1
                continue
            record = dict(r, corpus=name)
            if name in whole_clip:
                record["continuous"] = {k: v for k, v in r["continuous"].items() if k != "window_seconds"}
            check_loadable(record)
            if not usable_channels(record, "continuous"):
                counts[f"{name}:no_usable_channels"] += 1
                continue
            if (name, r["sign"]) in seen:
                raise SystemExit(f"{name} {r['sign']}: listed twice")
            seen.add((name, r["sign"]))
            jobs.append(record)
            counts[f"{name}:{r['partition']}"] += 1
    jobs.sort(key=lambda r: (r["corpus"], r["sign"]))
    return jobs, counts


def decode_state(state, tokenizer, motion_config, post, *, text: str, device="cpu"):
    """``state`` (served: conformed, 24 fps) -> its decoded copy, exactly ``tokenizer_domain_read.read_forms``'s
    ``recon_full`` (same calls, in the same order)."""
    import torch
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.generation.compositional import make_decode
    from signtranslator.generation.text_to_motion import diffusion_targets
    decode = make_decode(tokenizer, motion_config, post)
    with torch.no_grad():
        motion = collate_multichannel([state])
        unit = diffusion_targets(motion_config, motion).to(device)
        _, codes, _ = tokenizer.quantize(tokenizer.encode(unit))
        frames = int(motion.frame_valid.shape[1])
        if frames != len(state.timestamps):
            raise RuntimeError("collated frames differ from the state's")
        return decode(codes, frames, text)


def serve(record, template, *, cache, fetch):
    """The record as the prior's dataset serves it: load and mask, conform to the template, 24 fps."""
    from signtranslator.data.layout_adapter import conform
    from signtranslator.data_engineering.corpus_manifest import load_sample
    from evaluate_text_to_motion import at_rate
    return at_rate(conform(load_sample(record, "continuous", cache=cache, fetch=fetch), template), 24.)


def state_bytes(state) -> tuple[bytes, str]:
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "s.npz"
        digest = state.save(path)
        return path.read_bytes(), digest


def load_tokenizer(payload: bytes, device):
    import torch
    from signtranslator.generation.motion_rvq import MotionRVQ, MotionRVQConfig
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if saved.get("kind") != "motion_rvq_tokenizer":
        raise SystemExit(f"not a motion_rvq_tokenizer checkpoint: {saved.get('kind')}")
    tokenizer = MotionRVQ(MotionRVQConfig.from_dict(saved["config"]))
    tokenizer.load_state_dict(saved.get("ema") or saved["model"])
    return tokenizer.to(device).eval().requires_grad_(False), saved


# ------------------------------------------------------------------------------------------------------ workers

def _init(store_spec, tokenizer_path, template_path, template_sha, gains, device, cache_dir):
    import torch
    from signtranslator.data_engineering.corpus_manifest import ContentCache
    from signtranslator.pose.multichannel import MultichannelMotion
    from exposure_diagnostic import make_post
    from train_text_to_motion import make_store
    torch.set_num_threads(1)
    tokenizer, _ = load_tokenizer(Path(tokenizer_path).read_bytes(), device)
    _WORKER.update(store=make_store(store_spec), tokenizer=tokenizer, motion=tokenizer.config.motion,
                   post=make_post(gains), device=device,
                   template=MultichannelMotion.load(Path(template_path), expected_sha256=template_sha),
                   cache=ContentCache(Path(cache_dir) / "content", 4 * 2 ** 30))


def process(job, prefix: str, attempts: int = 4) -> dict:
    """One record, transient failures retried; a record that still fails is ``error`` (rerun the shard)."""
    import random
    for attempt in range(attempts):
        try:                                            # refusals are returned by process_once itself; anything
            return process_once(job, prefix, **_WORKER)  # raised here is the store's or the network's
        except Exception as error:  # noqa: BLE001
            if attempt == attempts - 1:
                return entry_of(job, status="error", reason=f"{type(error).__name__}: {error}"[:300])
            time.sleep(min(60., 2. * 2 ** attempt) * (.5 + random.random()))


def entry_of(job, **fields) -> dict:
    from signtranslator.generation.recognizer_corpora import load_digest
    return dict(corpus=job["corpus"], sign=job["sign"], partition=job["partition"],
                source_state_sha256=job["continuous"]["state_sha256"], load_digest=load_digest(job), **fields)


def process_once(job, prefix, *, store, tokenizer, motion, post, device, template, cache) -> dict:
    from signtranslator.generation.recognizer_corpora import check_loadable, load_digest
    from signtranslator.pose.multichannel import MultichannelMotion
    check_loadable(job)
    digest = load_digest(job)
    key = f"{prefix}/states/{digest}.npz"
    if store.exists(key):                               # write-once: re-hash, never rewrite
        payload = store.get(key)
        sha = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "s.npz"
            path.write_bytes(payload)
            frames = len(MultichannelMotion.load(path, expected_sha256=sha).timestamps)
        return entry_of(job, status="already_present", recon_key=key, recon_sha256=sha, frames=frames)
    part = job["continuous"]
    data = store.get(part["state_key"])                 # transient failures raise here and are retried
    if hashlib.sha256(data).hexdigest() != part["state_sha256"]:
        raise IOError(f"{part['state_key']}: fetched bytes differ from the manifest's SHA-256")

    def fetched(key):
        if key != part["state_key"]:
            raise IOError(f"{key}: unexpected fetch")
        return data
    try:                                                # from the verified bytes on: the record's own outcome
        served = serve(job, template, cache=cache, fetch=fetched)
        text = job.get("text") or (job.get("gloss") or {}).get("label") or job["sign"]
        recon = decode_state(served, tokenizer, motion, post, text=str(text), device=device)
    except (ValueError, KeyError) as error:             # conform, load or decode refused the clip
        return entry_of(job, status="refused", reason=f"{type(error).__name__}: {error}"[:300])
    recon = replace(recon, sample_id=f"recon:{job['corpus']}:{job['sign']}")
    payload, sha = state_bytes(recon)
    if store.exists(key):                               # another pass wrote it meanwhile: keep what is there
        return process_once(job, prefix, store=store, tokenizer=tokenizer, motion=motion, post=post, device=device,
                            template=template, cache=cache)
    store.put(key, payload)
    return entry_of(job, status="converted", recon_key=key, recon_sha256=sha, frames=len(recon.timestamps))


# --------------------------------------------------------------------------------------------------------- main

def config_of(args, store, tokenizer_sha, template_info) -> dict:
    from evaluate_text_to_motion import accepted_decoder_gains
    from fit_tokenizer_gains import decoder_gains_key
    gains = accepted_decoder_gains(store, args.tokenizer_key)
    gains_key = decoder_gains_key(args.tokenizer_key)
    return dict(schema=SCHEMA, tokenizer_key=args.tokenizer_key, tokenizer_sha256=tokenizer_sha, weights="ema",
                decoder_gains=gains, decoder_gains_key=gains_key if gains else None,
                decoder_gains_sha256=hashlib.sha256(store.get(gains_key)).hexdigest() if gains else None,
                template=template_info, fps=24., form=FORM, stages="all")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", default="r2")
    p.add_argument("--manifest", action="append", help="KEY@SHA of a gloss manifest, repeated (not with --merge)")
    p.add_argument("--whole-clip", action="append", default=[], help="CORPUS whose trim windows are removed (as the "
                   "recogniser's --whole-clip; the copies follow exactly the records the recogniser loads)")
    p.add_argument("--prefix", required=True, help="write-once store prefix of the decoded states and index")
    p.add_argument("--tokenizer-key", default=TOKENIZER_KEY)
    p.add_argument("--tokenizer-sha256", default=TOKENIZER_SHA256)
    p.add_argument("--template", default=TEMPLATE, help="KEY@SHA:CORPUS: conform to that corpus's first training state")
    p.add_argument("--prior-checkpoint-key", help="check the tokenizer against this prior checkpoint's tokenizer_state")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, help="smoke runs: only the first LIMIT jobs of the shard (no index is written)")
    p.add_argument("--cache-dir", default=str(Path(tempfile.gettempdir()) / "recon-states"))
    p.add_argument("--merge", action="store_true", help="write <prefix>/index.json from every shard's index")
    args = p.parse_args(argv)
    import torch
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest, load_sample
    from signtranslator.generation.recognizer_corpora import check_loadable
    from train_text_to_motion import make_store
    if not 0 <= args.shard < args.shards:
        raise SystemExit("shard must be in [0, shards)")
    prefix = args.prefix.rstrip("/")
    store = make_store(args.store)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    content = ContentCache(cache_dir / "content", 4 * 2 ** 30)

    def manifest_of(key, sha):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "m.json"
            path.write_bytes(store.get(key) + b"\n")
            return load_manifest(path, expected_sha256=sha)

    if args.merge:
        return merge(store, prefix, args.shards)
    if not args.manifest:
        p.error("the following arguments are required: --manifest")
    # tokenizer, template and configuration
    payload = store.get(args.tokenizer_key)
    tokenizer_sha = hashlib.sha256(payload).hexdigest()
    if tokenizer_sha != args.tokenizer_sha256:
        raise SystemExit(f"{args.tokenizer_key}: SHA-256 {tokenizer_sha}, expected {args.tokenizer_sha256}")
    tokenizer, saved = load_tokenizer(payload, "cpu")
    slim = io.BytesIO()                                 # workers load the weights only (the checkpoint holds optimizer state)
    torch.save(dict(kind=saved["kind"], config=saved["config"], model=tokenizer.state_dict()), slim)
    tokenizer_path = cache_dir / f"tokenizer-{tokenizer_sha}-weights.pt"
    tokenizer_path.write_bytes(slim.getvalue())
    del saved, slim
    if args.prior_checkpoint_key:
        prior = torch.load(io.BytesIO(store.get(args.prior_checkpoint_key)), map_location="cpu", weights_only=False)
        if prior.get("tokenizer_key") != args.tokenizer_key:
            raise SystemExit(f"{args.prior_checkpoint_key} names tokenizer {prior.get('tokenizer_key')}")
        mine = tokenizer.state_dict()
        theirs = prior["tokenizer_state"]
        if set(mine) != set(theirs) or not all(torch.equal(mine[k], theirs[k].cpu()) for k in mine):
            raise SystemExit(f"the tokenizer differs from {args.prior_checkpoint_key}'s tokenizer_state")
        print(f"tokenizer identical to {args.prior_checkpoint_key}'s", flush=True)
        del prior
    del payload, tokenizer
    t_key, rest = args.template.split("@")
    t_sha, t_corpus = rest.split(":")
    t_manifest = manifest_of(t_key, t_sha)
    t_record = next((r for r in t_manifest["records"] if r.get("corpus") == t_corpus and r.get("partition") == "train"),
                    None)
    if t_record is None:
        raise SystemExit(f"--template: no training record of {t_corpus!r}")
    check_loadable(t_record)
    template = load_sample(t_record, "continuous", cache=content, fetch=store.get)
    template_payload, template_sha = state_bytes(template)
    template_path = cache_dir / f"template-{template_sha}.npz"
    if not template_path.is_file():
        template_path.write_bytes(template_payload)
    del t_manifest
    template_info = dict(key=t_key, sha256=t_sha, corpus=t_corpus, sign=t_record.get("sign"),
                         state_sha256=t_record["continuous"]["state_sha256"])
    config = config_of(args, store, tokenizer_sha, template_info)
    config_payload = json.dumps(config, indent=1, sort_keys=True).encode()
    config_key = f"{prefix}/config.json"
    if store.exists(config_key):
        if store.get(config_key) != config_payload:
            raise SystemExit(f"{config_key} holds another configuration: this prefix is written once")
    else:                                               # also for --limit: its states belong to this configuration
        store.put(config_key, config_payload)
    # jobs
    out_key = shard_key(prefix, args.shard, args.shards)
    if args.limit is None and store.exists(out_key):
        print(f"{out_key} exists: shard done", flush=True)
        return 0
    manifests = [manifest_of(*parse_manifest_spec(m)) for m in args.manifest]
    jobs, counts = jobs_of(manifests, whole_clip=set(args.whole_clip))
    shard = [j for i, j in enumerate(jobs) if i % args.shards == args.shard][:args.limit]
    print(f"{len(jobs)} records, shard {args.shard}/{args.shards}: {len(shard)}; {json.dumps(dict(counts))}", flush=True)
    results, started = [], time.monotonic()
    initargs = (args.store, str(tokenizer_path), str(template_path), template_sha, config["decoder_gains"],
                args.device, str(cache_dir))
    if args.workers <= 1:
        _init(*initargs)
        iterator = (process(j, prefix) for j in shard)
        pool = None
    else:
        import multiprocessing                         # CUDA cannot be used in forked children
        context = multiprocessing.get_context("spawn" if str(args.device).startswith("cuda") else None)
        pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=initargs, mp_context=context)
        iterator = pool.map(process, shard, [prefix] * len(shard), chunksize=8)
    try:
        for i, result in enumerate(iterator):
            results.append(result)
            if i % 500 == 0:
                print(f"{i}/{len(shard)} {result['status']} {time.monotonic() - started:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    status = dict(collections.Counter(r["status"] for r in results))
    print(json.dumps(dict(status=status)), flush=True)
    if status.get("error"):
        print("transient errors: no shard index written; run the shard again (it resumes)", flush=True)
        return 2
    if args.limit is not None:
        print("smoke run (--limit): no shard index written", flush=True)
        return 0
    index = dict(schema=SCHEMA, config_sha256=hashlib.sha256(config_payload).hexdigest(), shard=args.shard,
                 shards=args.shards, manifests=args.manifest, whole_clip=sorted(args.whole_clip),
                 job_counts=dict(counts), status_counts=status, entries=results)
    if store.exists(out_key):
        raise SystemExit(f"{out_key} appeared while this shard ran: not overwritten")
    store.put(out_key, json.dumps(index, sort_keys=True).encode())
    print(f"wrote {out_key}", flush=True)
    return 0


def merge(store, prefix: str, shards: int) -> int:
    """``<prefix>/index.json`` from every shard index (one configuration, every shard present), written once."""
    config_payload = store.get(f"{prefix}/config.json")
    config_sha = hashlib.sha256(config_payload).hexdigest()
    entries, parts, manifests = [], [], None
    for shard in range(shards):
        key = shard_key(prefix, shard, shards)
        if not store.exists(key):
            raise SystemExit(f"{key} is missing: run that shard first")
        payload = store.get(key)
        part = json.loads(payload)
        if part["config_sha256"] != config_sha or part["shards"] != shards or part["shard"] != shard:
            raise SystemExit(f"{key} was made with another configuration or sharding")
        if manifests is not None and (part["manifests"], part["whole_clip"]) != manifests:
            raise SystemExit(f"{key} decoded other manifests")
        manifests = (part["manifests"], part["whole_clip"])
        parts.append(dict(key=key, sha256=hashlib.sha256(payload).hexdigest(), status_counts=part["status_counts"]))
        entries += part["entries"]
    keys = [(e["corpus"], e["sign"]) for e in entries]
    if len(set(keys)) != len(keys):
        raise SystemExit("a record appears in two shards")
    entries.sort(key=lambda e: (e["corpus"], e["sign"]))
    index = dict(schema=SCHEMA, config=json.loads(config_payload), config_sha256=config_sha, shards=parts,
                 manifests=manifests[0], whole_clip=manifests[1],
                 status_counts=dict(collections.Counter(e["status"] for e in entries)), entries=entries)
    payload = json.dumps(index, sort_keys=True).encode()
    key = f"{prefix}/index.json"
    if store.exists(key):
        if store.get(key) != payload:
            raise SystemExit(f"{key} exists with other contents: written once")
    else:
        store.put(key, payload)
    print(json.dumps(dict(index=key, sha256=hashlib.sha256(payload).hexdigest(), entries=len(entries),
                          status_counts=index["status_counts"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
