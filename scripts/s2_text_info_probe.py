"""Stage S2's probe P1, text without history (bars declared 2026-10-31 18:10 UTC, docs/NEXT_PHASE_PLAN.md, "Road to 3.8: S2
design received"). The one training the owner's rule allows here: a small head (``signtranslator.eval.s2_text_info``)
on cached stage-1 codes. It answers: does removing the history path let text reach the codes of unseen sentences?

**Stages, each write-once in ``--work``:**

- ``cache``: d2's stage-1 codes of the three lists.
  * **fit:** How2Sign c02 train clips, no others. Clips sharing a normalised text with the read list or the stop clips,
    or near one (``build_refiner_pairs.NearDuplicates``, as instrument v2's text hygiene), are dropped and counted.
    At most ``--fit-clips``, in sha256("s2-p1-fit|sign") order.
  * **stop (the early-stopping split, fixed here, before any read):** How2Sign c04 train clips, the first ``--stop-clips``
    in sha256("s2-p1-stop|sign") order, without a text equal or near to the read list's. c04 is train data, another
    signer than c02, and the fit drops its texts, so the split measures unseen text and an unseen signer. No val clip
    and no read-list result enters stopping or selection. Selection: lowest text-branch chain NLL per second.
  * **read:** D20, or ``--smoke-train N`` c02 clips kept out of the fit (a smoke).
- ``embed``: frozen flan-T5-base tokens of every text and of the empty text (max 128 tokens, as the AR prior's path).
- ``fit``: the head, on the fit list alone, with the declared stop clips.
- ``read``: per read clip, chain-rule gain over empty text in nats, with the clip's own text and with a deranged text;
  by part and scale.
- ``rank``: on the short clips (≤ 8 tokens; D20's 86), the own text among 100 (99 other list texts), by chain likelihood.
- ``sample``: one chain sample per read clip from text alone (guidance 1, temperature 1), decoded through d2 from
  stage 1: time-averaged ρ_faith per clip and the DTW latent distance to the real clip, the axis of P2's curve.
- ``control``: the probe's own validity check, on a separate head: word clips of the generator's mix (train partition
  of Sem-Lex, RIT, ASLLVD, WLASL; one token), performers held out by hash; 64 words, one held-out clip each; own word
  among the 64 by chain likelihood.
- ``verdict``: joins these with ``s1_text_kl.py --sentence-set d20``'s report, which reads the flan-T5 and W14b AR priors
  on the same clips, and computes every declared bar.

**The bars:** gain ≥ 2.84 nats/s and ≥ 2× flan-T5 AR on the same clips; shuffled ≤ 0.2 nats/s; each signer group > 0;
own text among 100 top-5 ≥ 15% on the 86 short clips; word-clip 64-way top-1 ≥ 0.3 on held-out signers, else the probe is
void. Rates are ``s2_lib.pooled_rate``, the unit of the stored 1.42 and −0.51 nats/s; ``verdict`` checks that the S1 report's
per-code gain × 96 equals its rate on the clips it read, and pairs the arms by sign.

A failed read on the Mac (fewer clips, ``--scale mac``) is reported as such: the plan confirms a fail once on a pod.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import s2_lib as L  # noqa: E402
from signtranslator.eval import s0_metrics as M  # noqa: E402

TEXT_MODEL = "google/flan-t5-base"
WORD_CORPORA = ("semlex-text-v2", "rit-text-r6", "asllvd-text-v1", "wlasl-text-r6")
CONTROL_WORDS = 64
FIT_MAX_SECONDS = 20.                 # clips longer than this are not fitted on (they are still read)
PARTS = ("body", "left_hand", "right_hand", "face")   # d2's order (MotionRVQ.parts)
GAIN_BAR, RATIO_BAR, SHUFFLED_BAR, SHORT_TOP5_BAR, CONTROL_BAR = 2.84, 2., .2, .15, .3


def keyed(prefix: str, sign: str) -> str:
    return hashlib.sha256(f"{prefix}|{sign}".encode()).hexdigest()


# ---- lists -----------------------------------------------------------------------------------------------------------

def fit_lists(manifest, read_records, *, fit_clips: int, stop_clips: int) -> tuple[list, list, dict]:
    """(fit records, stop records, counts): see the module's docstring. Raises if a read clip or a non-train clip could
    enter either list."""
    from build_refiner_pairs import NearDuplicates
    from round20_lib import text_key
    read_signs = {r["sign"] for r in read_records}
    read_texts = {text_key(r["text"]) for r in read_records}
    near_read = NearDuplicates(sorted({r["text"] for r in read_records}))
    counts = dict(c02=0, c04=0, stop_dropped_text=0, fit_dropped_read_text=0, fit_dropped_stop_text=0, fit_dropped_read_clip=0)
    stop = []
    c04 = L.train_records(manifest, ("c04",))
    counts["c04"] = len(c04)
    for r in sorted(c04, key=lambda r: keyed("s2-p1-stop", r["sign"])):
        if text_key(r["text"]) in read_texts or near_read.reason(r["text"]):
            counts["stop_dropped_text"] += 1
            continue
        if len(stop) < stop_clips:
            stop.append(r)
    stop_texts = {text_key(r["text"]) for r in c04}
    near_stop = NearDuplicates(sorted({r["text"] for r in c04}))
    fit = []
    c02 = L.train_records(manifest, ("c02",))
    counts["c02"] = len(c02)
    for r in sorted(c02, key=lambda r: keyed("s2-p1-fit", r["sign"])):
        if r["sign"] in read_signs:
            counts["fit_dropped_read_clip"] += 1
        elif text_key(r["text"]) in read_texts or near_read.reason(r["text"]):
            counts["fit_dropped_read_text"] += 1
        elif text_key(r["text"]) in stop_texts or near_stop.reason(r["text"]):
            counts["fit_dropped_stop_text"] += 1
        elif len(fit) < fit_clips:
            fit.append(r)
    L.check_train_records(fit, ("c02",))
    L.check_train_records(stop, ("c04",))
    if read_signs & {r["sign"] for r in fit + stop}:
        raise SystemExit("a read clip entered the fit or stop list")
    counts.update(fit=len(fit), stop=len(stop))
    return fit, stop, counts


def derangement(texts, seed: int = 7) -> list[int]:
    """Each clip's index of another clip with a different normalised text (the shuffled control)."""
    from round20_lib import text_key
    keys = [text_key(t) for t in texts]
    if len(set(keys)) < 2:
        raise ValueError("the shuffled control needs at least two different texts")
    rng = random.Random(seed)
    n = len(texts)
    for _ in range(10000):
        p = list(range(n))
        rng.shuffle(p)
        if all(keys[i] != keys[p[i]] for i in range(n)):
            return p
    raise ValueError("no derangement found")


def control_lists(manifest, *, fit_clips: int, words: int = CONTROL_WORDS, seed: int = 0) -> tuple[list, list, dict]:
    """(fit records, read records, counts) of the word control: train-partition one-token clips of ``WORD_CORPORA``. A
    tenth of the performers (by hash) are held out; ``words`` words that the fit also has, one held-out clip each."""
    from signtranslator.generation.compositional import normalise
    fit_pool, held = [], {}
    for r in manifest["records"]:
        if r.get("corpus") not in WORD_CORPORA or r.get("partition") != "train" or not r.get("text"):
            continue
        if r.get("signer_component") == "c00" or not (r.get("continuous") or {}).get("state_key"):
            continue
        if len(normalise(r["text"])) != 1:
            continue
        performer = r.get("performer_sha256") or ""
        if not performer:
            continue
        if int(keyed("s2-p1-control", performer)[:8], 16) % 10 == 0:
            held.setdefault(r["text"], []).append(r)
        else:
            fit_pool.append(r)
    fit_words = {r["text"] for r in fit_pool}
    choices = sorted(w for w in held if w in fit_words)
    rng = random.Random(seed)
    chosen = rng.sample(choices, min(words, len(choices)))
    read = [rng.choice(sorted(held[w], key=lambda r: r["sign"])) for w in chosen]
    fit = sorted(fit_pool, key=lambda r: keyed("s2-p1-control-fit", r["sign"]))[:fit_clips]
    if {r.get("performer_sha256") for r in fit} & {r.get("performer_sha256") for r in read}:
        raise SystemExit("a held-out performer entered the control's fit")
    return fit, read, dict(fit_pool=len(fit_pool), held_words=len(held), eligible_words=len(choices), words=len(read), fit=len(fit))


# ---- cache and embeddings --------------------------------------------------------------------------------------------

def stage1_codes(tokenizer, state) -> np.ndarray:
    """(steps, parts) d2's stage-1 codes of a state (j = q·P + p, so the first P of each step)."""
    import torch
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.generation.text_to_motion import diffusion_targets
    with torch.no_grad():
        unit = diffusion_targets(tokenizer.config.motion, collate_multichannel([state]))
        _, codes, _ = tokenizer.quantize(tokenizer.encode(unit))
    return codes[0, :, :len(tokenizer.parts)].cpu().numpy().astype(np.uint16)


def write_code_list(path: Path, name: str, records, tokenizer, fetch_states, *, chunk: int = 64) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists (write-once)")
    rows, signs = [], []
    for start in range(0, len(records), chunk):
        batch = records[start:start + chunk]
        for r, state in zip(batch, fetch_states(batch)):
            rows.append(stage1_codes(tokenizer, state))
            signs.append(r["sign"])
    meta = dict(name=name, signs=signs, texts=[r["text"] for r in records], groups=[r.get("signer_component") for r in records],
                partitions=[r.get("partition") for r in records])
    offsets = np.cumsum([0] + [len(x) for x in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, codes=np.concatenate(rows, 0), offsets=offsets, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))


def read_code_list(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(z["meta"].tobytes().decode())
        offsets, codes = z["offsets"], z["codes"].astype(np.int64)
    meta["rows"] = [codes[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    return meta


def embed_all(work: Path, texts, *, text_model: str = TEXT_MODEL, revision: str | None = None, batch: int = 32,
              tokens_of=None) -> None:
    """``tokens.npy`` (float32, concatenated) and ``tokens.json`` (text -> [start, length], sha, model). The empty text is
    the AR prior's empty branch (``_pretrained_inputs`` of ""): the encoder's end token alone."""
    out, index = work / "tokens.npy", work / "tokens.json"
    if out.exists() or index.exists():
        raise SystemExit(f"{out} exists (write-once)")
    texts = sorted(set(texts) | {""})
    commit = None
    if tokens_of is None:
        import torch
        from signtranslator.generation.motion_rvq import load_text_model
        lm, tok = load_text_model(text_model, revision)
        lm = lm.eval()
        commit = getattr(lm.config, "_commit_hash", None)

        def tokens_of(group):
            batch_in = tok(group, padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                states = lm(input_ids=batch_in["input_ids"], attention_mask=batch_in["attention_mask"]).last_hidden_state
            return [s[m.bool()].numpy().astype(np.float32) for s, m in zip(states, batch_in["attention_mask"])]
    table, rows, at = {}, [], 0
    for start in range(0, len(texts), batch):
        group = texts[start:start + batch]
        for text, row in zip(group, tokens_of(group)):
            table[text] = [at, len(row)]
            rows.append(row)
            at += len(row)
    np.save(out, np.concatenate(rows, 0))
    index.write_text(json.dumps(dict(model=text_model, revision=revision, commit=commit, tokens=table), sort_keys=True))


class Tokens:
    def __init__(self, work: Path) -> None:
        self.array = np.load(work / "tokens.npy", mmap_mode="r")
        self.info = json.loads((work / "tokens.json").read_text())
        self.table = self.info["tokens"]

    def __call__(self, text: str) -> np.ndarray:
        start, size = self.table[text]
        return np.array(self.array[start:start + size], np.float32)


# ---- reads -----------------------------------------------------------------------------------------------------------

def gain_arrays(head, rows, texts, other_texts, tokens, *, batch: int = 16, reveal_fn=None, device="cpu") -> dict:
    """Per clip (in ``rows`` order): chain log-probabilities with the text, with ``other_texts`` and with the empty text,
    and gains by group. Clips are batched by length; a clip's read does not depend on its batch."""
    from signtranslator.eval.s2_text_info import chain_log_probs, padded_codes, padded_text, reveal, chain_groups
    n = len(rows)
    own, other, empty = np.zeros(n), np.zeros(n), np.zeros(n)
    steps = max(len(r) for r in rows)
    groups = chain_groups(steps, rows[0].shape[1])
    per = np.zeros((n, len(groups)))
    reveal_fn = reveal_fn or reveal
    empty_tokens = tokens("")
    order = sorted(range(n), key=lambda i: len(rows[i]))
    for start in range(0, n, batch):
        idx = order[start:start + batch]
        codes, lengths = padded_codes([rows[i] for i in idx], device)
        own_t, own_m = padded_text([tokens(texts[i]) for i in idx], device)
        lp, by, gs = chain_log_probs(head, codes, lengths, own_t, own_m, reveal_fn=reveal_fn)
        lp0, by0, _ = chain_log_probs(head, codes, lengths, *padded_text([empty_tokens] * len(idx), device), reveal_fn=reveal_fn)
        lpx = chain_log_probs(head, codes, lengths, *padded_text([tokens(other_texts[i]) for i in idx], device), reveal_fn=reveal_fn)[0]
        own[idx], empty[idx], other[idx] = lp.cpu().numpy(), lp0.cpu().numpy(), lpx.cpu().numpy()
        slots = [groups.index(g) for g in gs]
        for row, i in enumerate(idx):
            per[i, slots] = (by[row] - by0[row]).cpu().numpy()
    return dict(own=own, empty=empty, other=other, per_group=per, groups=groups)


def read_report(head, listing: dict, tokens, parts, *, batch: int = 16, device="cpu") -> dict:
    from signtranslator.eval.s2_text_info import group_totals
    texts, rows = listing["texts"], listing["rows"]
    deranged = derangement(texts)
    arrays = gain_arrays(head, rows, texts, [texts[j] for j in deranged], tokens, batch=batch, device=device)
    seconds = np.array([len(r) / L.FPS for r in rows])
    gain, shuffled = arrays["own"] - arrays["empty"], arrays["other"] - arrays["empty"]
    groups = np.array(listing["groups"])
    return dict(signs=listing["signs"], groups=listing["groups"], seconds=np.round(seconds, 5).tolist(),
                gain_nats=np.round(gain, 4).tolist(), shuffled_gain_nats=np.round(shuffled, 4).tolist(),
                gain=L.pooled_rate(gain, seconds), shuffled_gain=L.pooled_rate(shuffled, seconds),
                by_signer_group={g: L.pooled_rate(gain[groups == g], seconds[groups == g]) for g in sorted(set(listing["groups"]))},
                breakdown_nats_per_clip=group_totals(arrays["per_group"], arrays["groups"], parts),
                derangement_seed=7)


def rank_report(head, listing: dict, tokens, *, indices, candidates: int = 100, batch: int = 50, device="cpu") -> dict:
    """Own text among ``candidates`` (own + others drawn from the list's other texts) on the clips ``indices``."""
    import torch
    from signtranslator.eval.s2_text_info import chain_log_probs, padded_text
    texts, rows = listing["texts"], listing["rows"]
    table = M.retrieval_candidates(len(texts), candidates, seed=0)
    ranks = []
    for i in indices:
        row = table[i]
        scores = np.zeros(len(row))
        for start in range(0, len(row), batch):
            part = row[start:start + batch]
            codes = torch.as_tensor(rows[i], dtype=torch.long, device=device)[None].expand(len(part), -1, -1)
            lengths = torch.full((len(part),), len(rows[i]), device=device)
            text, mask = padded_text([tokens(texts[j]) for j in part], device)
            scores[start:start + batch] = chain_log_probs(head, codes, lengths, text, mask)[0].cpu().numpy()
        ranks.append(float(M.mid_ranks(scores[None], lower_is_better=False)[0]))
    return dict(clips=len(indices), signs=[listing["signs"][i] for i in indices], ranks=ranks,
                summary=M.retrieval_summary(np.array(ranks), table.shape[1]))


def word_control_report(head, listing: dict, tokens, *, device="cpu") -> dict:
    """Own word among all the list's words, one clip each."""
    n = len(listing["rows"])
    indices = list(range(n))
    out = rank_report(head, listing, tokens, indices=indices, candidates=n, batch=32, device=device)
    out["words"] = n
    return out


def sample_report(head, tokenizer, listing, tokens, refs, *, seed: int = 0, batch: int = 16, device="cpu") -> dict:
    """One sample per clip, decoded from stage 1 (no gains): ρ_faith (time-averaged, K = 10, as S0) and the DTW latent
    distance to the real clip."""
    import torch
    import s0_ceilings as s0
    from signtranslator.eval.s2_text_info import padded_text, sample_chain
    from signtranslator.generation.text_to_motion import assemble_state, native_values, unit_to_raw
    rows, texts = listing["rows"], listing["texts"]
    cfg = tokenizer.config.motion
    samples: list = [None] * len(rows)
    match = np.zeros(len(rows))
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
    for start in range(0, len(rows), batch):
        idx = order[start:start + batch]
        lengths = torch.tensor([len(rows[i]) for i in idx], device=device)
        text, mask = padded_text([tokens(texts[i]) for i in idx], device)
        codes = sample_chain(head, lengths, text, mask, seed=seed + start)
        for k, i in enumerate(idx):
            samples[i] = codes[k, :len(rows[i])].cpu().numpy()
            match[i] = float((samples[i] == rows[i]).mean())
    states = []
    with torch.no_grad():
        for i, grid in enumerate(samples):
            q = {part: tokenizer._codebook_buffers("codes")[p, 0][torch.as_tensor(grid[:, p])][None]
                 for p, (part, _) in enumerate(tokenizer.parts)}
            unit = tokenizer.decode_latents(q, len(grid))
            raw = unit_to_raw(cfg, unit.double())
            states.append(assemble_state(cfg, "x", native_values(cfg, {k: v[0] for k, v in raw.items()}), len(grid), L.FPS,
                                         f"s2p1:{i}", "s2p1-clock"))
    real = s0.d2_latents(tokenizer, refs, "cpu")
    mean, std = M.standardiser(real)
    made = s0.d2_latents(tokenizer, states, "cpu")
    norm = lambda zs: [(z - mean) / std for z in zs]        # noqa: E731
    rho = M.rho_faith(norm(made), norm(real), K=10, distance="mean")
    distance = M.dtw_distances(norm(made), norm(real))
    return dict(signs=listing["signs"], rho_mean=rho["mean"], rho_mean_ci95=rho["ci95"], per_clip_rho_mean=rho["per_clip"],
                latent_distance=M.bootstrap_mean(distance), per_clip_latent_distance=np.round(distance, 5).tolist(),
                code_match=round(float(match.mean()), 5), seed=seed)


# ---- the verdict -----------------------------------------------------------------------------------------------------

def ar_rates(s1_report: dict, labels=("flant5", "w14b")) -> dict:
    """Each prior's per-clip gain nats and seconds by sign, with the unit check: per-code gain × 96 is its rate."""
    out = {}
    for label in labels:
        prior = s1_report["priors"][label]
        rate = prior["gain_nats_per_second"]["rate"]
        if abs(prior["gain"] * L.STAGE1_CODES_PER_SECOND - rate) > .005:
            raise SystemExit(f"{label}: gain {prior['gain']} × 96 is not its rate {rate}: the units differ")
        out[label] = dict(rate=rate, by_sign=dict(zip(s1_report["signs"], prior["per_clip"]["gain_nats"])),
                          seconds=dict(zip(s1_report["signs"], prior["per_clip"]["seconds"])))
    return out


def verdict(read: dict, rank: dict, control: dict, s1_report: dict, *, declared_read: bool, scale: str) -> dict:
    """Every declared bar. ``outcome`` is void (the control failed), pass, or fail; a fail at ``scale`` mac needs its
    one confirmation on a pod."""
    if declared_read:
        if s1_report.get("sentence_set", {}).get("sha256") != L.D20_SHA:
            raise SystemExit("the S1 report is not a D20 read")
        if len(read["signs"]) != L.D20_CLIPS or rank["clips"] != L.D20_SHORT:
            raise SystemExit(f"a declared read has {L.D20_CLIPS} clips and {L.D20_SHORT} short clips, not "
                             f"{len(read['signs'])} and {rank['clips']}")
    ar = ar_rates(s1_report)
    keep = [i for i, s in enumerate(read["signs"]) if ar["flant5"]["by_sign"].get(s) is not None]
    if not keep:
        raise SystemExit("the S1 report shares no clip with this read")
    signs = [read["signs"][i] for i in keep]
    gain = np.asarray(read["gain_nats"])[keep]
    seconds = np.asarray(read["seconds"])[keep]
    flant5 = np.array([ar["flant5"]["by_sign"][s] for s in signs])
    w14b = np.array([ar["w14b"]["by_sign"][s] for s in signs])
    ar_seconds = np.array([ar["flant5"]["seconds"][s] for s in signs])
    if np.abs(ar_seconds - seconds).max() > 2 / L.FPS:
        raise SystemExit("the S1 report's clip durations are not this read's")
    rate = L.pooled_rate(gain, seconds)
    flant5_rate = L.pooled_rate(flant5, seconds)
    bars = [
        L.bar("gain_nats_per_second", read["gain"]["rate"], GAIN_BAR, read["gain"]["rate"] >= GAIN_BAR, ci95=read["gain"]["ci95"]),
        L.bar("gain_over_flant5_ar_same_clips", round(rate["rate"] / flant5_rate["rate"], 4) if flant5_rate["rate"] > 0 else None,
              RATIO_BAR, flant5_rate["rate"] > 0 and rate["rate"] >= RATIO_BAR * flant5_rate["rate"],
              flant5_ar=flant5_rate, head_on_those_clips=rate, clips=len(keep)),
        L.bar("shuffled_gain_nats_per_second", read["shuffled_gain"]["rate"], SHUFFLED_BAR,
              read["shuffled_gain"]["rate"] <= SHUFFLED_BAR, ci95=read["shuffled_gain"]["ci95"]),
        L.bar("each_signer_group_above_zero", {g: v["rate"] for g, v in read["by_signer_group"].items()}, 0.,
              all(v["rate"] > 0 for v in read["by_signer_group"].values())),
        L.bar("short_clips_own_text_top5_of_100", rank["summary"]["top5"], SHORT_TOP5_BAR, rank["summary"]["top5"] >= SHORT_TOP5_BAR,
              clips=rank["clips"]),
    ]
    control_ok = control["summary"]["top1"] >= CONTROL_BAR
    outcome = "void" if not control_ok else ("pass" if all(b["passed"] for b in bars) else "fail")
    return dict(kind="s2-p1-verdict", declared_read=declared_read, scale=scale, outcome=outcome,
                confirm_on_pod=outcome == "fail" and scale == "mac", bars=bars,
                control=L.bar("word_control_top1_of_64", control["summary"]["top1"], CONTROL_BAR, control_ok,
                              words=control["words"]),
                ar_on_same_clips=dict(flant5=flant5_rate, w14b=L.pooled_rate(w14b, seconds)),
                paired_against_flant5=L.paired_rate_difference(gain, flant5, seconds),
                paired_against_w14b=L.paired_rate_difference(gain, w14b, seconds),
                stored_ar_nats_per_second={k: round(v * L.STAGE1_CODES_PER_SECOND, 2) for k, v in L.STORED_AR_GAIN_PER_CODE.items()},
                breakdown_nats_per_clip=read["breakdown_nats_per_clip"],
                declared="docs/NEXT_PHASE_PLAN.md, 'Road to 3.8: S2 design received'",
                note=None if declared_read else "SMOKE on train clips: no bar here is a declared result")


# ---- stages ----------------------------------------------------------------------------------------------------------

def make_head(config_overrides=None):
    from signtranslator.eval.s2_text_info import HeadConfig, TextInfoHead
    return TextInfoHead(HeadConfig(**(config_overrides or {})))


def stop_reader(head, listing, tokens, *, batch: int = 16, device="cpu"):
    """The declared stopping read: text-branch chain NLL per second on the stop clips (no gain, no read list)."""
    from signtranslator.eval.s2_text_info import chain_log_probs, padded_codes, padded_text

    def read() -> float:
        rows, texts = listing["rows"], listing["texts"]
        total, seconds = 0., 0.
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
        for start in range(0, len(rows), batch):
            idx = order[start:start + batch]
            codes, lengths = padded_codes([rows[i] for i in idx], device)
            text, mask = padded_text([tokens(texts[i]) for i in idx], device)
            total -= float(chain_log_probs(head, codes, lengths, text, mask)[0].sum())
            seconds += sum(len(rows[i]) / L.FPS for i in idx)
        return total / seconds
    return read


def fit_rows(listing, tokens) -> list:
    keep = [(r, tokens(t)) for r, t in zip(listing["rows"], listing["texts"]) if len(r) / L.FPS <= FIT_MAX_SECONDS]
    if not keep:
        raise SystemExit("no clip to fit on")
    return keep


def run_fit(work: Path, name: str, fit_listing, stop_listing, tokens, args) -> None:
    import torch
    from signtranslator.eval.s2_text_info import fit_head
    if (work / f"{name}.pt").exists():
        raise SystemExit(f"{work / f'{name}.pt'} exists (write-once)")
    for r in fit_listing["partitions"]:
        if r != "train":
            raise SystemExit("the fit list holds a clip that is not train")
    torch.manual_seed(args.seed)
    device = getattr(args, "device", "cpu")
    head = make_head(dict(codebook=args.codebook) if args.codebook != 512 else None).to(device)
    started = time.time()
    report = fit_head(head, fit_rows(fit_listing, tokens), empty_text=torch.as_tensor(tokens(""))[None], steps=args.steps,
                      batch=args.batch, seed=args.seed, every=args.every, patience=args.patience,
                      stop_read=stop_reader(head, stop_listing, tokens, device=device),
                      log=lambda row: print(json.dumps({name: row}), flush=True))
    torch.save(head.state_dict(), work / f"{name}.pt")
    report.update(seconds=round(time.time() - started, 1), fit_clips=len(fit_listing["rows"]), stop_clips=len(stop_listing["rows"]),
                  fit_list_sha256=hashlib.sha256("\n".join(fit_listing["signs"]).encode()).hexdigest(),
                  max_seconds=FIT_MAX_SECONDS, stop_rule="lowest text-branch chain NLL per second on the stop clips")
    L.write_file_once(work / f"{name}.json", report)


def load_head(work: Path, name: str, codebook: int, device="cpu"):
    import torch
    head = make_head(dict(codebook=codebook) if codebook != 512 else None)
    head.load_state_dict(torch.load(work / f"{name}.pt", map_location="cpu", weights_only=True))
    return head.to(device).eval()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=("cache", "embed", "fit", "read", "rank", "sample", "control-cache", "control-fit",
                                     "control-read", "verdict"))
    p.add_argument("--work", required=True)
    p.add_argument("--store", default="r2")
    p.add_argument("--smoke-train", type=int, default=0, help="read this many c02 clips instead of D20 (a smoke)")
    p.add_argument("--fit-clips", type=int, default=8000)
    p.add_argument("--stop-clips", type=int, default=64)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--every", type=int, default=500)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--codebook", type=int, default=512)
    p.add_argument("--control-fit-clips", type=int, default=3000)
    p.add_argument("--text-model-revision")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1, help="1 on the Mac; a pod may use more")
    p.add_argument("--s1-report", help="s1_text_kl.py --sentence-set d20's report (file) for the verdict")
    p.add_argument("--scale", choices=("mac", "pod"), default="mac")
    p.add_argument("--output-key")
    p.add_argument("--output-file")
    args = p.parse_args(argv)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if args.stage == "cache":
        import torch
        torch.set_num_threads(args.threads)
        records, declared, described = L.open_read_list(store, args.smoke_train)
        manifest = L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA)
        fit, stop, counts = fit_lists(manifest, records, fit_clips=args.fit_clips, stop_clips=args.stop_clips)
        tokenizer = L.load_tokenizer(store)
        started = time.time()
        fetch = lambda batch: L.load_states(store, batch)            # noqa: E731
        for name, recs in (("read", records), ("stop", stop), ("fit", fit)):
            write_code_list(work / f"codes-{name}.npz", name, recs, tokenizer, fetch)
        L.write_file_once(work / "cache.json", dict(declared_read=declared, list=described, counts=counts,
                                                    seconds=round(time.time() - started, 1)))
        print(json.dumps(counts), flush=True)
        return 0
    if args.stage == "embed":
        import torch
        torch.set_num_threads(args.threads)
        lists = [read_code_list(path) for path in sorted(work.glob("codes-*.npz"))]
        started = time.time()
        embed_all(work, [t for x in lists for t in x["texts"]], revision=args.text_model_revision)
        print(json.dumps(dict(texts=sum(len(x["texts"]) for x in lists), seconds=round(time.time() - started, 1))), flush=True)
        return 0
    cache = json.loads((work / "cache.json").read_text()) if (work / "cache.json").exists() else {}
    declared = bool(cache.get("declared_read"))
    if args.stage != "verdict":
        import torch
        torch.set_num_threads(args.threads)
    tokens = Tokens(work) if args.stage in ("fit", "read", "rank", "sample") else None
    parts = PARTS
    if args.stage == "fit":
        run_fit(work, "head", read_code_list(work / "codes-fit.npz"), read_code_list(work / "codes-stop.npz"), tokens, args)
    elif args.stage == "read":
        report = read_report(load_head(work, "head", args.codebook, args.device), read_code_list(work / "codes-read.npz"), tokens, parts,
                             batch=args.batch, device=args.device)
        L.write_file_once(work / "read.json", dict(report, declared_read=declared))
        print(json.dumps(dict(gain=report["gain"], shuffled=report["shuffled_gain"])), flush=True)
    elif args.stage == "rank":
        listing = read_code_list(work / "codes-read.npz")
        from signtranslator.generation.compositional import normalise
        short = [i for i, t in enumerate(listing["texts"]) if len(normalise(t)) <= L.SHORT_TOKENS]
        report = rank_report(load_head(work, "head", args.codebook, args.device), listing, tokens, indices=short, device=args.device)
        L.write_file_once(work / "rank.json", dict(report, declared_read=declared))
        print(json.dumps(report["summary"]), flush=True)
    elif args.stage == "sample":
        listing = read_code_list(work / "codes-read.npz")
        records, _, _ = L.open_read_list(store, args.smoke_train)
        if [r["sign"] for r in records] != listing["signs"]:
            raise SystemExit("the cached read list is not this list")
        report = sample_report(load_head(work, "head", args.codebook, args.device), L.load_tokenizer(store), listing, tokens,
                               L.load_states(store, records), batch=args.batch, device=args.device)
        L.write_file_once(work / "sample.json", dict(report, declared_read=declared))
        print(json.dumps(dict(rho_mean=report["rho_mean"], distance=report["latent_distance"])), flush=True)
    elif args.stage in ("control-cache", "control-fit", "control-read"):
        run_control(args, store, work)
    else:
        L.refuse_existing(store, args.output_key, args.output_file)
        if not args.s1_report:
            raise SystemExit("verdict needs --s1-report")
        load = lambda name: json.loads((work / name).read_text())    # noqa: E731
        report = verdict(load("read.json"), load("rank.json"), load("control-read.json"),
                         json.loads(Path(args.s1_report).read_text()), declared_read=declared, scale=args.scale)
        report["sample"] = load("sample.json") if (work / "sample.json").exists() else None
        report["fit"] = {k: v for k, v in load("head.json").items() if k != "history"}
        report["cache"] = cache
        L.write_once(store, args.output_key, args.output_file, report)
        print(json.dumps(dict(outcome=report["outcome"], bars=[(b["name"], b["value"], b["passed"]) for b in report["bars"]])), flush=True)
    return 0


def run_control(args, store, work: Path) -> None:
    """The word control's own lists, head and read, kept apart from the sentence head."""
    control = work / "control"
    control.mkdir(exist_ok=True)
    if args.stage == "control-cache":
        from signtranslator.data.layout_adapter import conform
        from evaluate_text_to_motion import at_rate, load_state
        manifest = L.load_manifest_at(store, L.TRAIN_MIX, L.TRAIN_MIX_SHA)
        fit, read, counts = control_lists(manifest, fit_clips=args.control_fit_clips)
        first = L.train_records(manifest, ("c02",))[0]
        template = load_state(store, first["continuous"]["state_key"], first["continuous"]["state_sha256"])
        tokenizer = L.load_tokenizer(store)

        def fetch(batch):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(4) as pool:
                return list(pool.map(lambda r: at_rate(conform(load_state(store, r["continuous"]["state_key"],
                                                                          r["continuous"]["state_sha256"]), template), L.FPS), batch))
        started = time.time()
        write_code_list(control / "codes-read.npz", "control-read", read, tokenizer, fetch)
        write_code_list(control / "codes-fit.npz", "control-fit", fit, tokenizer, fetch)
        L.write_file_once(control / "cache.json", dict(counts=counts, seconds=round(time.time() - started, 1)))
        lists = [read_code_list(control / f"codes-{n}.npz") for n in ("read", "fit")]
        embed_all(control, [t for x in lists for t in x["texts"]], revision=args.text_model_revision)
        print(json.dumps(counts), flush=True)
        return
    tokens = Tokens(control)
    if args.stage == "control-fit":
        fit = read_code_list(control / "codes-fit.npz")
        held = read_code_list(control / "codes-read.npz")
        run_fit(control, "head", fit, held_for_stop(fit, held), tokens, args)
        return
    report = word_control_report(load_head(control, "head", args.codebook, args.device), read_code_list(control / "codes-read.npz"), tokens,
                                 device=args.device)
    L.write_file_once(work / "control-read.json", report)
    print(json.dumps(report["summary"]), flush=True)


def held_for_stop(fit: dict, held: dict, clips: int = 32) -> dict:
    """The control head's stopping split: the last ``clips`` fit clips, by the fit list's hash order. Train-only and
    disjoint from the 64 words' held-out clips, which are read once afterwards."""
    tail = dict(fit)
    tail.update({k: fit[k][-clips:] for k in ("signs", "texts", "groups", "partitions", "rows")})
    for k in ("signs", "texts", "groups", "partitions", "rows"):
        fit[k] = fit[k][:-clips]
    return tail


if __name__ == "__main__":
    raise SystemExit(main())
