"""Post-S2 A6, before anything is declared: does A1's likelihood and spotting loss sit in the sentences holding the uses hold-x excludes? Read from A1's stored per-clip rows on D20 (forms already read), on CPU, report only; no new form is read and nothing is rendered.

hold-x keeps the left arm on entries with at least 25% two-handed records. If A1's loss came from erasing those entries' left arm, it should concentrate in the sentences that hold them. Three estimates of what excluding them removes, largest first: the whole loss of those sentences (a ceiling); that loss scaled by the
sentence's share of held uses that are excluded; and a regression of each clip's loss on its excluded and other held uses. None of them is a read of hold-x: A1's stretches differ in frames, and the share of frames hold-x stops replacing is reported beside them.

``power``: at declaration, the chance of passing each bound on eight draws for a range of true losses, from A5's paired SD (noncentral t). The table's slot in the report stays empty until then.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from comp_a_lib import annotated_label, entry_annotations, hold_flags  # noqa: E402

O = "acquisition/2026-10-17-v1/youtube-asl-v1"
EVAL_MIX = f"{O}/round10-mixed-v1/manifest.json"
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"
OUT = f"{O}/round20/comp-a"
MARGIN0, HOLD = f"{OUT}/a1-margin0.json", f"{OUT}/a1-hold.json"
FORM = "core_crossfade@predicted"
BLANK = "blank@predicted"
EXCLUDE = .25
BOOT = 2000
QUOTED_NEEDED = dict(likelihood=.71, spotting=.54)     # the review's figures, against the earlier floors


def excluded_entries(pools) -> set:
    """Entries A1 held and hold-x keeps: the plain hold's majority one-handed, not under the 25% cut."""
    a1, x = entry_annotations(pools), hold_flags(pools, two_handed_share=EXCLUDE)
    return {e for e, flag in a1.items() if flag is True and x.get(e) is not True}


def clip_counts(words_per_clip, a1_flags: dict, excluded: set) -> np.ndarray:
    """(n, 2): each clip's held uses A1 made that hold-x keeps, and its other held uses."""
    out = np.zeros((len(words_per_clip), 2))
    for i, words in enumerate(words_per_clip):
        held = [w for w in words if a1_flags.get(w) is True]
        out[i] = sum(w in excluded for w in held), sum(w not in excluded for w in held)
    return out


def concentration(delta, counts, *, boot: int = BOOT, seed: int = 0) -> dict:
    """Where one metric's paired change sits, with a bootstrap over clips. ``delta``: hold minus margin0 per clip, NaN where absent. A loss is negative, and a removal share is of the total change."""
    ok = np.isfinite(delta)
    d, c = np.asarray(delta, float)[ok], np.asarray(counts, float)[ok]
    has = c[:, 0] > 0
    held = c.sum(1)
    scaled = np.where(held > 0, c[:, 0] / np.clip(held, 1, None), 0.)

    def estimates(dd, cc, hh, ss):
        total = dd.sum()
        slope = np.linalg.lstsq(np.column_stack([np.ones(len(dd)), cc]), dd, rcond=None)[0]
        share = lambda x: x / total if total else np.nan     # noqa: E731   (a resample with no change at all)
        return dict(mean=dd.mean(), mean_with=dd[hh].mean() if hh.any() else np.nan, mean_without=dd[~hh].mean() if (~hh).any() else np.nan, ceiling=share(dd[hh].sum()), scaled=share((dd * ss).sum()),
                    regression=share(slope[1] * cc[:, 0].sum()), slope_excluded=slope[1], slope_other=slope[2])
    point = estimates(d, c, has, scaled)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point}
    for _ in range(boot):
        pick = rng.integers(0, len(d), len(d))
        row = estimates(d[pick], c[pick], has[pick], scaled[pick])
        for k, v in row.items():
            draws[k].append(v)
    ci = {k: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)] for k, v in draws.items()}
    return dict(clips=int(len(d)), clips_with_excluded=int(has.sum()), share_of_clips_with_excluded=round(float(has.mean()), 4), share_of_held_uses_excluded=round(float(c[:, 0].sum() / max(c.sum(), 1)), 4),
                **{k: round(float(v), 4) for k, v in point.items()}, ci95=ci)


def needed_share(loss: float, floor: float) -> float:
    """The share of a loss to remove for its point estimate to reach ``floor``."""
    return round(float((abs(loss) - abs(floor)) / abs(loss)), 4) if loss < 0 and abs(loss) > abs(floor) else 0.


def power_table(sd: float, floor: float, losses, n: int = 8) -> list[dict]:
    """P(the t interval's 95% lower bound over ``n`` draws is above ``floor``), for true mean changes ``losses``, at paired SD ``sd`` (noncentral t)."""
    from scipy import stats
    crit = stats.t.ppf(.975, n - 1)
    return [dict(true_change=float(x), passes=round(float(stats.nct.sf(crit, n - 1, (x - floor) / (sd / np.sqrt(n)))), 3)) for x in losses]


def read(store, *, words_key: str = "stage1_words") -> dict:
    from round15_oracle import word_pools
    from round20_compare import report_shards, shard_recalls
    from s2_lib import load_manifest_at
    from signtranslator.generation.compositional import isolated_corpora
    manifest = load_manifest_at(store, EVAL_MIX, EVAL_MIX_SHA)
    pools = word_pools(manifest, isolated_corpora(manifest))
    a1_flags, excluded = entry_annotations(pools), excluded_entries(pools)
    two = {e: sum(not x for x in map(annotated_label, r) if x is not None) for e, r in pools.items()}
    raw_m, raw_h = store.get(MARGIN0), store.get(HOLD)
    margin0, hold = json.loads(raw_m), json.loads(raw_h)
    if margin0["signs"] != hold["signs"]:
        raise SystemExit("A1's two reports read other clips")
    shards_m, shards_h = report_shards(store, MARGIN0, margin0, None), report_shards(store, HOLD, hold, None)
    words = {}
    for shard in shards_h:
        for row in shard["clips"]:
            words[row["sign"]] = row[words_key]
    words_per_clip = [words[s] for s in hold["signs"]]
    counts = clip_counts(words_per_clip, a1_flags, excluded)
    number = lambda v: np.array([np.nan if x is None else float(x) for x in v])            # noqa: E731
    ll = lambda r: number(r["likelihood"]["per_form"][FORM]) - number(r["likelihood"]["per_form"][BLANK])    # noqa: E731
    spot = number(shard_recalls(hold, shards_h, FORM)) - number(shard_recalls(margin0, shards_m, FORM))
    rows = dict(likelihood=concentration(ll(hold) - ll(margin0), counts), spotting=concentration(spot, counts))
    floors = dict(likelihood=-.0077, spotting=-.0089)
    earlier = dict(likelihood=-.01, spotting=-.02)
    for name, row in rows.items():
        row["needed_share_at_current_floor"] = needed_share(row["mean"], floors[name])
        row["needed_share_at_earlier_floor"] = needed_share(row["mean"], earlier[name])
        row["needed_share_quoted_by_review"] = QUOTED_NEEDED[name]
        row["ceiling_reaches_needed"] = bool(row["ceiling"] >= row["needed_share_at_current_floor"])
        row["regression_reaches_needed"] = bool(row["regression"] >= row["needed_share_at_current_floor"])
    return dict(kind="comp-a6-hold-concentration", report_only=True, note="A1's stored forms only, already read; A1's words (a9's list), not top10's; none of these is a read of hold-x",
                inputs=dict(margin0=dict(key=MARGIN0, sha256=hashlib.sha256(raw_m).hexdigest()), hold=dict(key=HOLD, sha256=hashlib.sha256(raw_h).hexdigest())), exclude=EXCLUDE,
                excluded_entries=len(excluded), excluded_uses=int(counts[:, 0].sum()), other_held_uses=int(counts[:, 1].sum()), excluded_with_two_handed_records=sum(1 for e in excluded if two.get(e)),
                metrics=rows, power_table_at_declaration=None, power_note="slot: filled at declaration from A5's paired SD (comp_a6_hold_concentration.py power)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("read")
    r.add_argument("--store", default="r2")
    r.add_argument("--output-file", required=True)
    w = sub.add_parser("power")
    w.add_argument("--a5-summary-file", required=True)
    w.add_argument("--report-file", required=True)
    args = p.parse_args(argv)
    if args.command == "read":
        if Path(args.output_file).exists():
            raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
        from train_text_to_motion import make_store
        out = read(make_store(args.store))
        data = json.dumps(out, indent=1, sort_keys=True).encode()
        with open(args.output_file, "xb") as stream:
            stream.write(data)
        print(args.output_file, hashlib.sha256(data).hexdigest(), flush=True)
        return 0
    from comp_a6_read import floors, slot_from_summary         # only here: this module has no other tie to the reader
    raw = Path(args.a5_summary_file).read_bytes()
    slot = slot_from_summary(json.loads(raw), hashlib.sha256(raw).hexdigest())
    bars = floors(slot)
    report = json.loads(Path(args.report_file).read_text())
    report["power_table_at_declaration"] = {m: dict(sd=slot["paired_sd"][m], floor=bars[m], rows=power_table(slot["paired_sd"][m], bars[m], [report["metrics"][m]["mean"] * (1 - s) for s in (0., .25, .5, .75, 1.)])) for m in ("likelihood", "spotting")}
    Path(args.report_file).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report["power_table_at_declaration"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
