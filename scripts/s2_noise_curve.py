"""Stage S2's probe P2, code noise (bars declared 2026-10-31 18:10 UTC, docs/NEXT_PHASE_PLAN.md, "Road to 3.8: S2 design
received"): how does the instrument read stages 1-4 of real D20 clips when their codes are damaged? Nothing trains.

**Forms** (every clip: d2's stage 1-4 codes of the real clip, changed as below, decoded with no gains; seeds depend on
the clip and the form only, so a form read alone in a process is the same as read with the others):
- ``real``, ``blank``, ``clean`` (stages 1-4 unchanged: ``s0_ceilings.truncated_round_trip(model, state, 4)``);
- ``iid10``, ``iid25``, ``iid50``: that share of codes, chosen per code, replaced by draws from the list's own marginal of
  that stage and part;
- ``block25``, ``block50``: per part, that share of the clip's 0.25 s blocks (6 steps) replaced by the same blocks' worth of
  contiguous stage 1-4 codes from another clip of the list, as a wrong-content generator would;
- ``jitter``: the same codes on a monotone time warp with offsets of up to ±0.25 s, one offset about every second, end
  points fixed. Content is intact and per-step code accuracy is not, so this asks how much exact timing the read needs;
- ``key12``, ``key6``: stage 1-4 latents kept at 12 or 6 Hz, linearly interpolated between, requantised to 4 stages.

**Bars** (``verdict``): ``block25`` BLEU ≥ 2.328 (D20 composition, ``a4-w14b-n16-r10``), paired lower bound above blank;
``jitter`` within 0.3 of ``clean``; ``key12`` ≥ 0.9× ``clean``'s share above blank, or option 2 of the design drops (not a
stop). ``key6`` is reported. ``curve`` places every form on one axis, its DTW latent distance to the real clip
(``s2_lib.latent_distances``, ρ_faith's own distance).

**Evidence test** (``verdict --p1-report --b0-report``, computed only when both are given): at the latent distance of
P1-head samples, the curve (linear between forms, held flat beyond its ends and said so) must predict BLEU at least 0.2
above B0's W14b no-gains read and ΔNLL at least 0.10 above it; those samples' time-averaged ρ must be below W14b's,
paired lower bound above 0.

    python scripts/s2_noise_curve.py read --forms real,blank,clean --out-dir W      (up to 5 processes side by side)
    python scripts/s2_noise_curve.py verdict --out-dir W --output-file W/verdict.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import s2_lib as L  # noqa: E402
from signtranslator.eval import s0_metrics as M  # noqa: E402

STAGES = 4
BLOCK_STEPS = 6                       # 0.25 s at 24 steps/s
JITTER_STEPS = 6                      # ±0.25 s
JITTER_EVERY = 24                     # one offset about every second
FORMS = ("real", "blank", "clean", "iid10", "iid25", "iid50", "block25", "block50", "jitter", "key12", "key6")
FORM_ID = {name: i for i, name in enumerate(FORMS)}
NOISE_SEED = 20261031
COMPOSITION_D20_BLEU = 2.328          # a4-w14b-n16-r10 on D20 (plan, Round 20 A4)
JITTER_BAR, KEY_BAR = .3, .9
EVIDENCE_BLEU, EVIDENCE_DNLL = .2, .1


# ---- damage (pure, on (steps, stages, parts) code grids) ------------------------------------------------------------

def rng_for(clip: int, form: str) -> np.random.Generator:
    return np.random.default_rng([NOISE_SEED, clip, FORM_ID[form]])


def iid_noise(grid: np.ndarray, share: float, marginals: np.ndarray, rng) -> np.ndarray:
    """``grid`` with each code replaced with probability ``share`` by a draw from ``marginals`` (stages, parts, K)."""
    out = grid.copy()
    S, Q, P = grid.shape
    hit = rng.random((S, Q, P)) < share
    for q in range(Q):
        for p in range(P):
            k = int(hit[:, q, p].sum())
            if k:
                out[hit[:, q, p], q, p] = rng.choice(marginals.shape[2], k, p=marginals[q, p])
    return out


def block_noise(grid: np.ndarray, share: float, clip: int, grids, rng, *, block: int = BLOCK_STEPS) -> np.ndarray:
    """``grid`` with, per part, ``share`` of its blocks replaced by contiguous codes from another clip of the list."""
    out = grid.copy()
    S, Q, P = grid.shape
    blocks = max(1, S // block)
    donors = [i for i in range(len(grids)) if i != clip and len(grids[i]) >= 1]
    if not donors:
        raise ValueError("block noise needs another clip")
    for p in range(P):
        count = int(round(share * blocks))
        for b in rng.choice(blocks, count, replace=False):
            a = int(b) * block
            end = S if int(b) == blocks - 1 else min(S, a + block)
            donor = grids[donors[int(rng.integers(len(donors)))]]
            size = end - a
            start = int(rng.integers(0, max(1, len(donor) - size + 1)))
            piece = donor[start:start + size, :, p]
            if len(piece) < size:                                   # a donor shorter than the block: repeat its last step
                piece = np.concatenate((piece, np.repeat(piece[-1:], size - len(piece), 0)), 0)
            out[a:end, :, p] = piece
    return out


def jitter_index(steps: int, rng, *, offset: int = JITTER_STEPS, every: int = JITTER_EVERY) -> np.ndarray:
    """Monotone (steps,) index into the clip: knots about every second moved by up to ±``offset`` steps, ends fixed."""
    if steps < 3:
        return np.arange(steps)
    knots = np.arange(0, steps, every)
    if knots[-1] != steps - 1:
        knots = np.r_[knots, steps - 1]
    moved = knots.astype(float)
    moved[1:-1] += rng.uniform(-offset, offset, len(knots) - 2)
    moved = np.maximum.accumulate(np.clip(moved, 0, steps - 1))
    return np.clip(np.round(np.interp(np.arange(steps), knots, moved)).astype(np.int64), 0, steps - 1)


def key_latents(latents: dict, stride: int) -> dict:
    """Each part's (1, S, D) latents kept every ``stride`` steps (and at the last) and linearly interpolated between."""
    import torch
    out = {}
    for name, z in latents.items():
        S = z.shape[1]
        keys = np.arange(0, S, stride)
        if keys[-1] != S - 1:
            keys = np.r_[keys, S - 1]
        at = np.clip(np.searchsorted(keys, np.arange(S), side="right") - 1, 0, max(len(keys) - 2, 0))
        if len(keys) < 2:
            out[name] = z
            continue
        w = torch.as_tensor((np.arange(S) - keys[at]) / (keys[at + 1] - keys[at]), dtype=z.dtype, device=z.device)[:, None]
        out[name] = (z[0, keys[at]] * (1 - w) + z[0, keys[at + 1]] * w)[None]
    return out


# ---- forms -----------------------------------------------------------------------------------------------------------

class _Fixed:
    """``round_trip``'s plan hook: the decoder reads these quantised latents, so the masks and neutral fills are those of
    every other round trip."""

    def __init__(self, quantised=None, compute=None) -> None:
        self.quantised, self.compute = quantised, compute

    def eval(self):
        return self

    def reconstruct(self, latents, valid):
        return self.quantised if self.compute is None else self.compute(latents)


def code_grid(tokenizer, state) -> np.ndarray:
    """(steps, 4, parts) d2's first four stages of ``state``."""
    import torch
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.generation.text_to_motion import diffusion_targets
    with torch.no_grad():
        unit = diffusion_targets(tokenizer.config.motion, collate_multichannel([state]))
        _, codes, _ = tokenizer.quantize(tokenizer.encode(unit))
    Q, P = tokenizer.config.stages, len(tokenizer.parts)
    return codes[0].reshape(-1, Q, P)[:, :STAGES].cpu().numpy()


def decode_grid(tokenizer, state, grid: np.ndarray):
    """``state``'s round trip with the decoder reading these codes (no gains)."""
    import torch
    from tokenizer_recognisability_check import round_trip
    books = tokenizer._codebook_buffers("codes")
    quantised = {}
    for p, (part, _) in enumerate(tokenizer.parts):
        quantised[part] = sum(books[p, q][torch.as_tensor(grid[:, q, p])] for q in range(grid.shape[1]))[None]
    with torch.no_grad():
        return round_trip(tokenizer, tokenizer.config.motion, state, L.FPS, plan=_Fixed(quantised))


def marginals_of(grids, K: int) -> np.ndarray:
    Q, P = grids[0].shape[1:]
    out = np.zeros((Q, P, K))
    for g in grids:
        for q in range(Q):
            for p in range(P):
                out[q, p] += np.bincount(g[:, q, p], minlength=K)
    return out / out.sum(-1, keepdims=True)


def build_form(name: str, context: dict):
    import s0_ceilings as s0
    refs, tokenizer = context["refs"], context["tokenizer"]
    if name not in FORM_ID:
        raise SystemExit(f"unknown form {name}")
    if name in ("real", "blank"):
        return s0.build_form(name, refs, tokenizer, None)
    if name == "clean":
        return s0.build_form("stages4", refs, tokenizer, None)
    if name.startswith("key"):
        import torch
        from tokenizer_recognisability_check import round_trip
        stride = {"key12": 2, "key6": 4}[name]
        hook = _Fixed(compute=lambda latents: tokenizer.quantize(key_latents(tokenizer.quantize(latents, STAGES)[0], stride), STAGES)[0])
        with torch.no_grad():
            return [round_trip(tokenizer, tokenizer.config.motion, s, L.FPS, plan=hook) for s in refs]
    if "grids" not in context:
        context["grids"] = [code_grid(tokenizer, s) for s in refs]
        context["marginals"] = marginals_of(context["grids"], tokenizer.config.codebook)
    grids = context["grids"]
    out = []
    for i, (state, grid) in enumerate(zip(refs, grids)):
        rng = rng_for(i, name)
        if name.startswith("iid"):
            damaged = iid_noise(grid, int(name[3:]) / 100, context["marginals"], rng)
        elif name.startswith("block"):
            damaged = block_noise(grid, int(name[5:]) / 100, i, grids, rng)
        else:
            damaged = grid[jitter_index(len(grid), rng)]
        out.append(decode_grid(tokenizer, state, damaged))
    return out


# ---- verdict ---------------------------------------------------------------------------------------------------------

def curve_at(points: list[dict], distance: float, key: str) -> dict:
    """``key`` at ``distance`` on the curve: linear between neighbouring forms by latent distance, flat beyond its ends."""
    rows = sorted(points, key=lambda r: r["latent_distance"])
    x = np.array([r["latent_distance"] for r in rows])
    y = np.array([r[key] for r in rows])
    return dict(value=round(float(np.interp(distance, x, y)), 4), beyond_curve=bool(distance < x[0] or distance > x[-1]),
                distance_range=[round(float(x[0]), 5), round(float(x[-1]), 5)])


def evidence(curve: list[dict], p1: dict, b0: dict) -> dict:
    """The owner's rule, from P2's curve, P1-head samples and B0's W14b no-gains read (reference lengths, as P2's forms)."""
    arm = b0["forms"]["w14b_nogains_ref"]
    if p1["signs"] != b0["signs"]:
        raise SystemExit("P1's samples and B0 did not read the same clips")
    d = p1["latent_distance"]["mean"]
    bleu, dnll = curve_at(curve, d, "bleu"), curve_at(curve, d, "delta_nll")
    gain_bleu = bleu["value"] - arm["bleu"]["bleu"]
    gain_dnll = dnll["value"] - arm["delta_nll"]["mean"]
    rho = M.paired_mean_difference(arm["per_clip"]["rho_mean"], p1["per_clip_rho_mean"])
    bars = [L.bar("predicted_bleu_over_w14b_nogains", round(gain_bleu, 4), EVIDENCE_BLEU, gain_bleu >= EVIDENCE_BLEU, prediction=bleu),
            L.bar("predicted_delta_nll_over_w14b_nogains", round(gain_dnll, 4), EVIDENCE_DNLL, gain_dnll >= EVIDENCE_DNLL, prediction=dnll),
            L.bar("rho_mean_w14b_minus_samples_lower_bound", rho["ci95"][0], 0., rho["ci95"][0] > 0, difference=rho["difference"],
                  samples=p1["rho_mean"], w14b=arm["rho_faith"]["mean"]["mean"])]
    return dict(p1_latent_distance=d, bars=bars, passed=all(b["passed"] for b in bars),
                caveat="P1 samples stage 1 only and a valid sample with other timing still has a high distance, so this "
                       "reading is a proxy, not a generator's read.")


def make_verdict(reads: dict, *, p1: dict | None = None, b0: dict | None = None) -> dict:
    report = L.verdict_shell("p2", reads)
    panels = {name: L.form_panel(reads, name) for name in reads}
    block, clean, jitter = panels["block25"], panels["clean"], panels["jitter"]
    blank, key12, key6 = panels["blank"], panels["key12"], panels["key6"]
    above = clean["bleu"] - blank["bleu"]
    share = lambda x: round((x["bleu"] - blank["bleu"]) / above, 4) if above else None   # noqa: E731
    bars = [L.bar("block25_bleu", block["bleu"], COMPOSITION_D20_BLEU, block["bleu"] >= COMPOSITION_D20_BLEU, ci95=block["ci95"]),
            L.bar("block25_minus_blank_lower_bound", block["paired"]["blank"]["bleu"]["ci95"][0], 0.,
                  block["paired"]["blank"]["bleu"]["ci95"][0] > 0, difference=block["paired"]["blank"]["bleu"]["difference"]),
            L.bar("jitter_within_clean", round(abs(jitter["bleu"] - clean["bleu"]), 4), JITTER_BAR,
                  abs(jitter["bleu"] - clean["bleu"]) <= JITTER_BAR)]
    keep = share(key12) is not None and share(key12) >= KEY_BAR
    report.update(forms=panels, bars=bars, passed=all(b["passed"] for b in bars),
                  option2=L.bar("key12_share_of_clean", share(key12), KEY_BAR, keep, key6_share=share(key6)),
                  option2_kept=bool(keep),
                  curve=[dict(form=n, latent_distance=panels[n]["latent_distance"]["mean"], bleu=panels[n]["bleu"],
                              delta_nll=panels[n]["delta_nll"]["mean"], wrist_speed_over_real=panels[n]["wrist_speed_over_real"])
                         for n in ("clean", "iid10", "iid25", "iid50", "block25", "block50", "jitter", "key12", "key6")])
    if p1 is not None and b0 is not None:
        report["evidence"] = evidence(report["curve"], p1, b0)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("read")
    L.add_read_arguments(r, ",".join(FORMS))
    v = sub.add_parser("verdict")
    v.add_argument("--store", default="r2")
    v.add_argument("--out-dir", required=True)
    v.add_argument("--p1-report", help="P1's sample.json")
    v.add_argument("--b0-report", help="B0's s0 report on D20")
    v.add_argument("--output-key")
    v.add_argument("--output-file")
    args = p.parse_args(argv)
    if args.command == "read":
        unknown = [f for f in args.forms.split(",") if f and f not in FORM_ID]
        if unknown:
            raise SystemExit(f"unknown forms {unknown}")
        import torch
        torch.set_num_threads(1)
        return L.read_forms(args, probe="p2", build=build_form)
    from train_text_to_motion import make_store
    store = make_store(args.store) if args.output_key else None
    L.refuse_existing(store, args.output_key, args.output_file)
    load = lambda path: json.loads(Path(path).read_text()) if path else None  # noqa: E731
    report = make_verdict(L.load_form_reads(args.out_dir, FORMS), p1=load(args.p1_report), b0=load(args.b0_report))
    L.write_once(store, args.output_key, args.output_file, report)
    print(json.dumps(dict(passed=report["passed"], bars=[(b["name"], b["value"], b["passed"]) for b in report["bars"]],
                          option2=report["option2_kept"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
