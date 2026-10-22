"""Citation stitching baseline: gloss lookup, exact duration, holds for uncovered signs, generated-state contract."""
import numpy as np
import pytest

from signtranslator.generation.citation_stitch import INFERENCE, lookup_names, resolve, stitch, trim_to_hands
from tests.test_pose_resample import state


def test_lookup_strips_decoration_splits_compounds_and_maps_indices():
    lex = {"WHY", "NICE/CLEAN", "MOTHER", "PAST", "NIGHT", "citizen:YOU", "citizen:HE", "WHEN", "CLEAN"}
    assert resolve("(25)WHY", lex) == ("WHY",)
    assert resolve("NICE/CLEAN+", lex) == ("NICE/CLEAN",)
    assert resolve("MOTHERwg", lex) == ("MOTHER",)
    assert resolve("PAST__+NIGHT", lex) == ("PAST", "NIGHT")
    assert resolve("IX-2p", lex) == ("citizen:YOU",)
    assert resolve("IX-3p-pl-2:i/j", lex) == ("citizen:HE",)
    assert resolve("#WHEN", lex) == ("WHEN",)
    assert resolve("WHY_2", lex) == ("WHY",) and resolve("MOTHER-indef", lex) == ("MOTHER",)
    assert resolve("PAST___+NIGHT", lex) == ("PAST", "NIGHT")
    assert resolve("fs-BEACH", lex | {"fs-BEACH"}) is None             # fingerspelling has no citation form
    assert resolve("UNKNOWN", lex) is None
    assert lookup_names("WHY")[0] == ("WHY",)


def test_stitched_state_fills_the_duration_and_follows_the_generated_contract():
    a, b = state(np.arange(20) / 60.), state(np.arange(30) / 30.)
    out = stitch([a, None, b], fps=24., seconds=2., source_sha256="b" * 64)
    assert len(out.timestamps) == 48 and np.allclose(np.diff(out.timestamps), 1 / 24.)
    for c in out.channels.values():
        assert not c.observed.any() and (c.confidence == 0).all() and c.inference_method == INFERENCE
    # the second sign has no citation form: the first clip's last pose is held through its share
    body = out.channels["body"].values[:, 0]
    held = [t for t in range(1, 47) if np.allclose(body[t], body[t - 1])]
    assert len(held) >= 5
    with pytest.raises(ValueError):
        stitch([None, None], fps=24., seconds=1., source_sha256="b" * 64)


def test_trim_keeps_the_span_where_a_hand_is_tracked():
    s = state(np.arange(10) / 30.)
    from dataclasses import replace
    for side in ("left_hand", "right_hand"):
        c = s.channels[side]
        valid = c.valid.copy(); valid[:3] = False; valid[8:] = False
        s = replace(s, channels=dict(s.channels, **{side: replace(c, valid=valid, observed=valid.copy(),
                                                                  confidence=np.where(valid, 1., 0.),
                                                                  values=np.where(valid[..., None], c.values, 0.))}))
    t = trim_to_hands(s, margin=1)
    assert len(t.timestamps) == 7 and t.timestamps[0] == s.timestamps[2]
