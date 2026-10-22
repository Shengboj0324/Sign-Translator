"""End-to-end avatar render of a synthetic canonical state (needs the local Blender; skipped without it)."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from signtranslator.avatar_render.makehuman import load_makehuman_rig
from signtranslator.avatar_render.retarget_check import image_mirror_check, raised_arm_points, state_from_points

ROOT = Path(__file__).resolve().parents[1]
BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
sys.path.insert(0, str(ROOT / "scripts"))


def test_image_mirror_check_reads_the_viewers_side():
    image = np.full((20, 20, 3), 100, dtype=np.uint8)
    image[2:6, 14:16] = 200                       # arm above the head on the viewer's right
    assert image_mirror_check(image, image[0, 0], head_top_row=8)["passed"]
    flipped = image[:, ::-1].copy()
    result = image_mirror_check(flipped, flipped[0, 0], head_top_row=8)
    assert not result["passed"] and result["fraction_on_viewer_right"] == 0.


@pytest.mark.skipif(not BLENDER.exists(), reason="local Blender is not installed")
def test_render_avatar_writes_a_verified_timestamped_video(tmp_path):
    import render_avatar
    rig = load_makehuman_rig(ROOT / "Sign Translator Stage Documentation/evidence/w1-w2-2026-09-26/source-intake"
                             / "makehuman")
    world, visible, hands = raised_arm_points(rig, "Left", frames=3)
    # vary the raised arm a little so frames differ; a nonuniform clock
    world[1:, 13:16:2] += [.0, -.5, .5]
    clock = np.array([0., .04, .1])
    state = state_from_points(clock, world, visible, hands, sample_id="render-avatar-test")
    path = tmp_path / "state.npz"
    digest = state.save(path)
    out = tmp_path / "render"
    assert render_avatar.main(["--state", str(path), "--sha256", digest, "--out-dir", str(out),
                               "--resolution", "96", "--allow-inferred", "--blender", str(BLENDER)]) == 0
    record = json.loads((out / "verification.json").read_text())
    assert record["frames"] == 3 and record["source"]["sha256"] == digest
    assert record["acceptance"]["passed"] and record["mirror"]["rendered_probe"]["passed"]
    assert record["deterministic_rerender"]
    assert record["verification"]["max_clock_error_seconds"] <= record["verification"]["half_tick_bound_seconds"]
    assert record["encoding"]["pixel_format"] == "yuv420p"
    assert record["posed_inferred_values"] is True and record["accepted_provenance"] is False
    for group in ("upper_arm", "forearm", "finger", "thumb"):
        assert record["direction_errors"]["groups"][group]["max_deg"] < 1e-5
    assert (out / "avatar.mp4").stat().st_size > 0
    with pytest.raises(FileExistsError):
        render_avatar.main(["--state", str(path), "--out-dir", str(out), "--allow-inferred",
                            "--blender", str(BLENDER)])


def test_inferred_states_need_the_explicit_flag(tmp_path):
    import render_avatar
    rig = load_makehuman_rig(ROOT / "Sign Translator Stage Documentation/evidence/w1-w2-2026-09-26/source-intake"
                             / "makehuman")
    state = state_from_points(np.array([0.]), *raised_arm_points(rig, "Left"))
    path = tmp_path / "state.npz"
    state.save(path)
    with pytest.raises(ValueError, match="allow_inferred"):
        render_avatar.main(["--state", str(path), "--out-dir", str(tmp_path / "r"), "--blender", str(BLENDER)])
