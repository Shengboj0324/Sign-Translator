import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("round17_pt_verdict", ROOT / "scripts" / "round17_pt_verdict.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

REF = {"reference_stage1_perplexity": {f"{k}_{p}": x for k in ("every_step", "every_4th_step")
                                       for p, x in zip(v.PARTS, (174.5, 155.2, 64.6, 114.6) if k == "every_step"
                                                       else (162.7, 147.0, 63.9, 110.4))}}


def best(**over):
    val = {f"val/stage1_perplexity_{p}": x for p, x in zip(v.PARTS, (175., 200., 60., 95.))}
    val.update({f"stability/reseed_fraction_per_1k_{p}": 0. for p in v.PARTS})
    val.update(over)
    return dict(checkpoint_key="k", step=10000, validation=val)


FID = dict(r2b=dict(real=dict(bleu=4.136, ci95=[3.6, 4.7]), round_trip=dict(bleu=2.80, ci95=[2.4, 3.2]),
                    blank=dict(bleu=1.859, ci95=[1.537, 2.175]), tokenizer_round_trip=dict(bleu=3.365, ci95=[2.9, 3.8])),
           ratios={"left_hand_speed_ratio": .6})
REC = dict(ratio=.93)


def test_pass_and_arm_reference():
    assert v.verdict(best(), REF, FID, REC, "a", [{"step": 1, "loss": 1.}])["overall"] == "PASS"
    # arm b's bar uses d2 every step: right hand 0.8 x 114.6 = 91.68 > 0.8 x 110.4 = 88.32
    assert v.verdict(best(**{"val/stage1_perplexity_right_hand": 90.}), REF, FID, REC, "a")["overall"] == "PASS"
    assert v.verdict(best(**{"val/stage1_perplexity_right_hand": 90.}), REF, FID, REC, "b")["overall"] == "FAIL"


def test_content_ratio_exact_and_missing():
    fid = dict(FID, r2b=dict(FID["r2b"], round_trip=dict(bleu=0.8 * 3.365 - 1e-9, ci95=[2.4, 3.0])))
    assert v.verdict(best(), REF, fid, REC, "a")["overall"] == "FAIL"
    assert v.verdict(best(), REF, None, REC, "a")["overall"] == "NOT-EVALUABLE"
    assert v.verdict(best(), REF, FID, None, "a")["overall"] == "NOT-EVALUABLE"
    assert v.verdict(best(), REF, FID, dict(ratio=.89), "a")["overall"] == "FAIL"


def test_floor_and_divergence():
    ref = {"reference_stage1_perplexity": {k: 10. for k in REF["reference_stage1_perplexity"]}}
    assert v.verdict(best(**{"val/stage1_perplexity_left_hand": 31.}), ref, FID, REC, "a")["overall"] == "FAIL"
    assert v.verdict(best(), REF, FID, REC, "a", [{"step": 5, "loss": float("nan")}])["overall"] == "FAIL"


def test_above_blank_uses_declared_bounds():
    fid = dict(FID, r2b=dict(FID["r2b"], round_trip=dict(bleu=2.8, ci95=[2.1, 3.2])))
    assert v.verdict(best(), REF, fid, REC, "a")["overall"] == "FAIL"           # 2.1 is not above blank's 2.175
