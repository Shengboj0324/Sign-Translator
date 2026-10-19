"""Joint intrinsics over the pilot val sessions, then each session's gate with them (and alone)."""
import io, json, os, sys
import numpy as np
sys.path.insert(0, "/Users/jiangshengbo/Desktop/Sign-Translator"); sys.path.insert(0, "/Users/jiangshengbo/Desktop/Sign-Translator/scripts")
from signtranslator.data.text_motion_dataset import R2Store
from signtranslator.data_engineering import two_view_states as tv
from triangulate_how2sign_sessions import discover_sessions, synchronised
s = R2Store(os.environ)
D = "acquisition/2026-10-17-v1/derived/pilot-val-"
names = sys.argv[1:]
sessions = {}
for n in names:
    found = discover_sessions(s, D + n + "/sapiens-front", D + n + "/sapiens-side")
    (key, sentences), = found.items()
    sync = synchronised(s, sentences)
    print(n, "lag", sync.get("lag"), "clips", len(sync["aligned"]), "of", len(sync["syncs"]), sync["reason"] or "",
          "sharp", (sync["epipolar"] or {}).get("sharp"))
    if sync["aligned"]:
        sessions[n] = [a[4][1:] for a in sync["aligned"]]
def mono_cv(n):
    cvs = []
    for o in s.s3.list_objects_v2(Bucket=s.bucket, Prefix=D + n + "/video-estimate/landmarks/")["Contents"]:
        with np.load(io.BytesIO(s.get(o["Key"]))) as m:
            cvs.append(tv.monocular_bone_cv(m["pose_world"], m["pose_visibility"]))
    return float(np.nanmedian(cvs))
def evaluate(n, cal):
    pts = np.concatenate([tv.triangulated_points(cal, *p)[0] for p in sessions[n]])
    return dict(focal=(round(cal.focal1), round(cal.focal2)), reproj=tuple(np.round(cal.median_reprojection_px, 2)),
                tri_cv=round(tv.triangulated_bone_cv(pts), 4), sym=round(tv.symmetry_error(pts), 3))
monos = {n: round(mono_cv(n), 4) for n in sessions}
print("monocular CV", monos)
for n in sessions:
    print("alone ", n, evaluate(n, tv.calibrate_session(sessions[n])))
order = list(sessions)
joint = tv.calibrate_cameras([sessions[n] for n in order], max_frames=1500)
print("joint intrinsics", round(joint[0].focal1), round(joint[0].focal2))
for n, c in zip(order, joint):
    print("joint ", n, evaluate(n, c))
for held in order:                                   # leave-one-session-out: intrinsics from the others
    rest = [n for n in order if n != held]
    if len(rest) < 2: continue
    shared = tv.calibrate_cameras([sessions[n] for n in rest], max_frames=1500)[0]
    c = tv.calibrate_session(sessions[held], intrinsics=(shared.focal1, shared.focal2, 0., 0.))
    print("held-out", held, "intrinsics from", rest, evaluate(held, c))
