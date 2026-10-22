"""Render a canonical state on the audited MakeHuman rig, with a verification record.

Run with the project Python::

    render_avatar.py --state STATE.npz [--sha256 HEX] --out-dir NEW_DIR --allow-inferred
    render_avatar.py --key R2_KEY --sha256 HEX [--store r2|dir:/path] --out-dir NEW_DIR --allow-inferred

The state is retargeted by :mod:`signtranslator.avatar_render.retarget` (body, both
hands with all 15 finger joints, head, face pose units, eyes), skinned in the
project Python (body, high-poly eyes, teeth, tongue), and the posed meshes are
written to a bundle. The same file then runs inside a separate
``--background --factory-startup`` Blender process that reads only the bundle and
renders PNG frames with the Workbench engine (fixed settings, a fixed orthographic
front camera: the viewer sees the signer from the front, signer's left on the
viewer's right). The frames are encoded to H.264 with presentation timestamps from
the state's own clock and decoded back to verify every timestamp.

``verification.json`` records the retarget's angular errors against
``canonical_skeleton.skeleton_positions`` (median/p95/max per bone group, stored-valid
frames only), the head error against ``head_rotation``, three left/right checks
(structural, per-frame lateral agreement, and a rendered raised-left-arm probe),
which face coefficients were rendered, every filled frame, a re-render determinism
check and the timestamp verification. Nothing is uploaded anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
INTAKE = ROOT / "Sign Translator Stage Documentation" / "evidence" / "w1-w2-2026-09-26" / "source-intake"
DEFAULT_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
OBJECTS = ("body", "eyes", "teeth", "tongue")
SKIN = (0.80, 0.64, 0.54)
TOP = (0.16, 0.19, 0.26)        # dark, plain long-sleeved top: hands contrast with it, as interpreters dress
TROUSERS = (0.20, 0.20, 0.22)
SKIN_BONES = ("head", "neck", "wrist", "metacarpal", "finger", "jaw", "eye", "tongue", "levator", "oris",
              "special", "temporalis", "risorius", "oculi", "orbicularis")
TROUSER_BONES = ("root", "pelvis", "upperleg", "lowerleg", "foot", "toe")
BROW = (0.23, 0.16, 0.12)
BACKGROUND = 0.18
ACCEPTANCE = {"upper_arm": 5., "forearm": 5., "finger": 10., "thumb": 10.}


# --------------------------------------------------------------------------- inside Blender

def render_inside_blender(bundle: Path, out_dir: Path, resolution: int, extra_first: bool) -> None:
    import bpy
    import numpy as np

    archive = np.load(bundle, allow_pickle=False)
    camera_centre, span = archive["camera.centre"].astype(float), float(archive["camera.span"])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "VERTEX"
    scene.display.shading.show_cavity = False
    scene.display.shading.show_shadows = False
    scene.display.shading.show_specular_highlight = False
    scene.display.render_aa = "8"
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("plain")
    scene.world.color = (BACKGROUND, BACKGROUND, BACKGROUND)

    def converted(vertices):            # declared proper rotation (x, y, z) -> (x, -z, y)
        return np.stack((vertices[..., 0], -vertices[..., 2], vertices[..., 1]), axis=-1)

    meshes = []
    for name in OBJECTS:
        if f"{name}.vertices" not in archive.files:
            continue
        vertices = converted(archive[f"{name}.vertices"].astype(np.float64))
        faces = archive[f"{name}.faces"]
        colours = archive[f"{name}.corner_colours"]
        polygons = [tuple(int(v) for v in face if v >= 0) for face in faces]
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices[0].tolist(), [], polygons)
        mesh.update()
        try:
            mesh.shade_smooth()
        except AttributeError:
            mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
        attribute = mesh.color_attributes.new("colour", "FLOAT_COLOR", "CORNER")
        rgba = np.concatenate([np.concatenate((colours[i, :len(p)], np.ones((len(p), 1))), axis=1)
                               for i, p in enumerate(polygons)])
        attribute.data.foreach_set("color", rgba.astype(np.float64).reshape(-1))
        mesh.color_attributes.active_color = attribute
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)
        meshes.append((mesh, vertices))

    camera_data = bpy.data.cameras.new("front")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = span
    camera_data.clip_start, camera_data.clip_end = 0.01, 1000.
    camera = bpy.data.objects.new("front", camera_data)
    x, y, z = camera_centre
    camera.location = (float(x), float(-z) - 4 * span, float(y))   # in front of the signer (+z source)
    camera.rotation_euler = (1.5707963267948966, 0.0, 0.0)          # look along +Y (Blender) = -z source
    scene.collection.objects.link(camera)
    scene.camera = camera

    frames = meshes[0][1].shape[0]
    order = list(range(frames)) + ([0] if extra_first else [])
    for k, index in enumerate(order):
        for mesh, vertices in meshes:
            mesh.vertices.foreach_set("co", vertices[index].reshape(-1))
            mesh.update()
        name = f"frame-{index:06d}.png" if k < frames else "rerender-000000.png"
        scene.render.filepath = str(out_dir / name)
        bpy.ops.render.render(write_still=True)


# --------------------------------------------------------------------------- project Python

def _srgb_to_linear(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _texture_corner_colours(attachment):
    """Per-corner colours sampled from the attachment's pinned texture at each corner's UV.

    Faces whose texture is transparent (the eyes' cornea shell) are dropped, since the
    Workbench render is opaque; returns (kept face indices, (F, C, 3) linear colours).
    """
    import numpy as np
    from PIL import Image
    with Image.open(attachment.texture_path) as image:
        pixels = np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.
    height, width = pixels.shape[:2]
    uv = np.asarray(attachment.texcoords, dtype=np.float64)
    width_max = max(len(face) for face in attachment.faces)
    colours = np.zeros((len(attachment.faces), width_max, 4))
    for i, corners in enumerate(attachment.face_uvs):
        u, v = uv[list(corners)].T
        cols = np.clip(np.round(u % 1. * (width - 1)).astype(int), 0, width - 1)
        rows = np.clip(np.round((1. - v % 1.) * (height - 1)).astype(int), 0, height - 1)
        colours[i, :len(corners)] = pixels[rows, cols]
    counts = np.array([len(face) for face in attachment.faces])
    alpha = np.array([colours[i, :c, 3].mean() for i, c in enumerate(counts)])
    kept = np.flatnonzero(alpha >= .5)
    return kept, _srgb_to_linear(colours[kept, :, :3])


def _body_corner_colours(rig, faces):
    """Skin on head, neck and hands; a plain dark top and trousers elsewhere.

    Each vertex mixes the region colours by its skinning weights, so garment edges
    follow the skin's own blend zones instead of a jagged dominant-bone boundary.
    """
    import numpy as np
    weights = rig.weights.detach().cpu().numpy()
    palette = []
    for name in rig.bone_names:
        if name.startswith(SKIN_BONES):
            palette.append(SKIN)
        elif name.startswith(TROUSER_BONES):
            palette.append(TROUSERS)
        else:
            palette.append(TOP)
    palette = np.asarray(palette)
    skin = weights @ np.array([p == SKIN for p in map(tuple, palette)], dtype=np.float64)
    edge = np.clip((skin - .35) / .3, 0., 1.)
    edge = edge * edge * (3 - 2 * edge)                                   # smoothstep across the blend zone
    cloth = weights @ np.where(np.array([p == SKIN for p in map(tuple, palette)])[:, None], 0., palette)
    cloth = cloth / np.clip(1. - skin, 1e-9, None)[:, None]
    cloth = np.where((skin >= 1. - 1e-9)[:, None], np.asarray(TOP), cloth)
    colour = edge[:, None] * np.asarray(SKIN) + (1. - edge[:, None]) * cloth
    # Painted eyebrows: the skin the brow bones (oculi01) carry, in a band above each eye, so
    # brow raise and furrow (skinned by those bones) read at review resolution.
    index = {name: i for i, name in enumerate(rig.bone_names)}
    rest = rig.rest_vertices.detach().cpu().numpy()
    heads = rig.rest_globals.detach().cpu().numpy()[:, :3, 3]
    brow = np.zeros(len(rest))
    for s in ("L", "R"):
        eye = heads[index[f"eye.{s}"]]
        offset = rest[:, 0] - eye[0] * 1.08                      # centred slightly outward of the pupil
        lateral = np.abs(offset)
        centre = eye[1] + 0.22 - 0.9 * offset ** 2 + (0.03 * offset if s == "L" else -0.03 * offset)
        band = (np.exp(-((rest[:, 1] - centre) / 0.035) ** 2) * (lateral < 0.27)
                * (np.abs(rest[:, 0]) > 0.10) * (rest[:, 2] > eye[2]))
        brow = np.maximum(brow, band * np.clip(weights[:, index[f"oculi01.{s}"]] / 0.25, 0., 1.))
    brow = np.clip((brow - 0.25) / 0.35, 0., 1.)[:, None]
    colour = brow * np.asarray(BROW) + (1. - brow) * colour
    vertex = _srgb_to_linear(colour)
    width = max(len(face) for face in faces)
    colours = np.zeros((len(faces), width, 3))
    for i, face in enumerate(faces):
        colours[i, :len(face)] = vertex[list(face)]
    return colours


def _padded(faces):
    import numpy as np
    out = np.full((len(faces), max(len(face) for face in faces)), -1, dtype=np.int64)
    for i, face in enumerate(faces):
        out[i, :len(face)] = face
    return out


def framing(rig, body_vertices, *, margin: float = 1.06):
    """Fixed front camera over the clip: waist up, every posed arm/hand vertex included."""
    import numpy as np
    names = rig.bone_names
    index = {name: i for i, name in enumerate(names)}
    rest = rig.rest_vertices.detach().cpu().numpy()
    waist = float(rig.rest_globals[index["spine04"], 1, 3])
    dominant = rig.weights.detach().cpu().numpy().argmax(1)
    arm_bones = {i for i, n in enumerate(names)
                 if any(n.startswith(p) for p in ("upperarm", "lowerarm", "wrist", "finger", "metacarpal"))}
    keep = (rest[:, 1] > waist) | np.isin(dominant, sorted(arm_bones))
    selected = body_vertices[:, keep]
    low, high = selected.min(axis=(0, 1)), selected.max(axis=(0, 1))
    low[1] = max(low[1], waist - 1.)          # never below a hand-width under the waist
    centre = (low + high) / 2
    span = float(max(high[0] - low[0], high[1] - low[1])) * margin
    return centre, span


def project(points, centre, span, resolution):
    """(column, row) pixels of source points in the fixed orthographic front camera."""
    import numpy as np
    points = np.asarray(points, dtype=np.float64)
    column = (points[..., 0] - (centre[0] - span / 2)) / span * resolution
    row = ((centre[1] + span / 2) - points[..., 1]) / span * resolution
    return np.stack((column, row), axis=-1)


def pose_and_bundle(rig, retargeted, attachments, path: Path, *, extra: dict | None = None):
    """Skin every frame; write the bundle; return bone globals, body vertices and framing."""
    import numpy as np
    import torch
    from signtranslator.avatar_render.rigged import apply_lbs
    frames = retargeted.local_rotations.shape[0]
    bone_globals = np.empty((frames, len(rig.bone_names), 4, 4))
    vertices = {"body": np.empty((frames, rig.rest_vertices.shape[0], 3), dtype=np.float32)}
    for name, attachment in attachments.items():
        vertices[name] = np.empty((frames, attachment.rest_vertices.shape[0], 3), dtype=np.float32)
    for t in range(frames):
        posed = rig.pose(torch.from_numpy(np.ascontiguousarray(retargeted.local_rotations[t])))
        bone_globals[t] = posed.bone_globals.numpy()
        vertices["body"][t] = posed.vertices.numpy()
        for name, attachment in attachments.items():
            vertices[name][t] = apply_lbs(attachment.rest_vertices, attachment.weights,
                                          posed.skin_transforms).numpy()
    if not all(np.isfinite(v).all() for v in vertices.values()):
        raise ValueError("posed vertices are not finite")
    keep = [i for i, groups in enumerate(rig.face_groups) if "body" in groups]
    body_faces = [rig.faces[i] for i in keep]
    arrays = {"body.vertices": vertices["body"], "body.faces": _padded(body_faces),
              "body.corner_colours": _body_corner_colours(rig, body_faces), "clock": retargeted.clock}
    dropped = {}
    for name, attachment in attachments.items():
        kept, colours = _texture_corner_colours(attachment)
        arrays[f"{name}.vertices"] = vertices[name]
        arrays[f"{name}.faces"] = _padded([attachment.faces[i] for i in kept])
        arrays[f"{name}.corner_colours"] = colours
        dropped[name] = len(attachment.faces) - len(kept)
    centre, span = framing(rig, vertices["body"])
    arrays["camera.centre"], arrays["camera.span"] = centre, np.array(span)
    provenance = dict(retargeted.provenance(), objects=sorted(["body", *attachments]),
                      coordinate_conversion_to_blender="(x, y, z) -> (x, -z, y), proper rotation, no scale",
                      transparent_attachment_faces_dropped=dropped, **(extra or {}))
    arrays["provenance"] = np.array(json.dumps(provenance, sort_keys=True, default=float))
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
    return bone_globals, vertices["body"], (centre, span), hashlib.sha256(path.read_bytes()).hexdigest()


def run_blender(blender: Path, bundle: Path, frames_dir: Path, resolution: int, log: Path, *,
                extra_first: bool) -> None:
    frames_dir.mkdir()
    completed = subprocess.run(
        [str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
         "--python", str(Path(__file__).resolve()), "--", "--inside-blender",
         str(bundle), str(frames_dir), str(resolution), "1" if extra_first else "0"],
        capture_output=True, text=True)
    log.write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(f"Blender failed with {completed.returncode}; see {log}")


def read_png(path: Path):
    import av
    with av.open(str(path)) as container:
        return next(container.decode(video=0)).to_ndarray(format="rgb24")


def slice_state(state, start_seconds: float | None, max_seconds: float | None):
    import numpy as np
    from dataclasses import replace
    from signtranslator.pose.multichannel import MultichannelMotion
    t = state.timestamps
    lo = t[0] + (start_seconds or 0.)
    keep = (t >= lo) & ((t < lo + max_seconds) if max_seconds else True)
    if keep.all():
        return state
    if not keep.any():
        raise SystemExit("the requested window contains no frames")
    channels = {name: replace(c, values=c.values[keep], valid=c.valid[keep], observed=c.observed[keep],
                              confidence=c.confidence[keep]) for name, c in state.channels.items()}
    return MultichannelMotion(np.ascontiguousarray(t[keep]), state.clock_id, state.sample_id, channels)


def load_input_state(args):
    import tempfile
    from signtranslator.pose.multichannel import MultichannelMotion
    if args.state is not None:
        payload = args.state.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if args.sha256 is not None and args.sha256 != digest:
            raise SystemExit("state bytes do not match --sha256")
        return MultichannelMotion.load(args.state, expected_sha256=digest), dict(path=str(args.state), sha256=digest)
    if args.sha256 is None:
        raise SystemExit("--key requires --sha256")
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_text_to_motion import make_store
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "state.npz"
        path.write_bytes(store.get(args.key))
        state = MultichannelMotion.load(path, expected_sha256=args.sha256)
    return state, dict(key=args.key, sha256=args.sha256, store=args.store.split(":")[0])


def mirror_probe(rig, binding, out_dir: Path, blender: Path) -> dict:
    """Render a synthetic raised-left-arm state and check where the arm lands in the image."""
    import numpy as np
    from signtranslator.avatar_render.retarget import retarget_state
    from signtranslator.avatar_render.retarget_check import image_mirror_check, raised_arm_points, state_from_points
    world, visible, hands = raised_arm_points(rig, "Left")
    state = state_from_points(np.array([0.]), world, visible, hands, sample_id="mirror-probe-raised-left-arm")
    retargeted = retarget_state(rig, binding, state, allow_inferred=True)
    probe = out_dir / "mirror-probe"
    probe.mkdir()
    bone_globals, body, (centre, span), _ = pose_and_bundle(rig, retargeted, {}, probe / "bundle.npz")
    resolution = 128
    run_blender(blender, probe / "bundle.npz", probe / "frames", resolution, probe / "blender.log", extra_first=False)
    image = read_png(probe / "frames" / "frame-000000.png")
    index = {name: i for i, name in enumerate(rig.bone_names)}
    head_bones = [i for i, n in enumerate(rig.bone_names) if n == "head"]
    dominant = rig.weights.detach().cpu().numpy().argmax(1)
    head_top = float(body[0][np.isin(dominant, head_bones), 1].max())
    row = int(np.floor(project([[0., head_top, 0.]], centre, span, resolution)[0, 1])) - 2
    check = image_mirror_check(image, image[0, 0], head_top_row=row)
    wrist = bone_globals[0, index["wrist.L"], :3, 3]
    check.update(left_wrist_source_xyz=wrist.tolist(),
                 left_wrist_pixel=project(wrist, centre, span, resolution).tolist(),
                 left_wrist_above_head=bool(wrist[1] > head_top), resolution=resolution,
                 rule="the signer's left is drawn on the viewer's right (front view of a signer facing the camera)")
    check["passed"] = bool(check["passed"] and check["left_wrist_above_head"] and wrist[0] > 0)
    return check


def main(argv=None) -> int:
    if "--inside-blender" in sys.argv:
        args = sys.argv[sys.argv.index("--inside-blender") + 1:]
        render_inside_blender(Path(args[0]), Path(args[1]), int(args[2]), args[3] == "1")
        return 0
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--state", type=Path, help="local canonical-state .npz")
    source.add_argument("--key", help="store key of a canonical-state .npz (read only)")
    parser.add_argument("--sha256", help="expected state SHA-256 (required with --key)")
    parser.add_argument("--store", default="r2", help="'r2' (uses R2_* environment) or 'dir:/path'")
    parser.add_argument("--out-dir", type=Path, required=True, help="new directory")
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--start-seconds", type=float)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--face-baseline", choices=("clip_median", "none"), default="clip_median")
    parser.add_argument("--no-hinge-twist", action="store_true", help="disable the elbow-hinge twist choice")
    parser.add_argument("--drive-legs", action="store_true", help="pose thighs/shins from the state (default: rest)")
    parser.add_argument("--lossless", action="store_true", help="H.264 4:4:4 CRF 0 instead of review 4:2:0")
    parser.add_argument("--allow-inferred", action="store_true",
                        help="required to pose estimated or generated (observed=False) values")
    args = parser.parse_args(argv)
    if args.resolution % 2:
        parser.error("--resolution must be even")
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from signtranslator.avatar_render.makehuman import load_makehuman_rig
    from signtranslator.avatar_render.makehuman_eyes import load_makehuman_eyes
    from signtranslator.avatar_render.makehuman_face import load_makehuman_face_units
    from signtranslator.avatar_render.makehuman_mouth import load_makehuman_mouth
    from signtranslator.avatar_render.retarget import (
        FACE_UNIT_SOURCES, UNREPRESENTED_BLENDSHAPES, build_retarget_binding, retarget_state,
    )
    from signtranslator.avatar_render.retarget_check import (
        direction_errors, lateral_agreement, structural_handedness,
    )
    from signtranslator.generation.stream_render import encode_timestamped_video, verify_timestamped_video

    state, source_record = load_input_state(args)
    full_frames = len(state.timestamps)
    state = slice_state(state, args.start_seconds, args.max_seconds)
    args.out_dir.mkdir(parents=False, exist_ok=False)
    rig = load_makehuman_rig(INTAKE / "makehuman")
    face_units = load_makehuman_face_units(INTAKE / "makehuman-face", rig)
    attachments = {"eyes": load_makehuman_eyes(INTAKE / "makehuman-eyes", rig),
                   **load_makehuman_mouth(INTAKE / "makehuman-mouth", rig)}
    binding = build_retarget_binding(rig)
    for attachment in attachments.values():
        if attachment.rig_identity != binding.rig_identity:
            raise SystemExit("attachment fitted to a different rig")
    retargeted = retarget_state(rig, binding, state, allow_inferred=args.allow_inferred, face_units=face_units,
                                face_baseline=args.face_baseline, hinge_twist=not args.no_hinge_twist,
                                drive_legs=args.drive_legs)
    bundle_path = args.out_dir / "bundle.npz"
    bone_globals, body, (centre, span), bundle_sha = pose_and_bundle(
        rig, retargeted, attachments, bundle_path, extra=dict(source=source_record))
    errors = direction_errors(rig, state, retargeted, bone_globals)
    lateral = lateral_agreement(rig, state, retargeted, bone_globals)
    probe = mirror_probe(rig, binding, args.out_dir, args.blender)

    frames_dir = args.out_dir / "frames"
    run_blender(args.blender, bundle_path, frames_dir, args.resolution, args.out_dir / "blender.log",
                extra_first=True)
    images = [read_png(frames_dir / f"frame-{i:06d}.png") for i in range(len(state.timestamps))]
    rerender = read_png(frames_dir / "rerender-000000.png")
    video = args.out_dir / "avatar.mp4"
    encoded = encode_timestamped_video(images, state.timestamps, video,
                                       quality="lossless" if args.lossless else "review")
    verified = verify_timestamped_video(video, state.timestamps)

    groups = errors["groups"]
    acceptance = {group: dict(threshold_median_deg=limit, median_deg=groups[group]["median_deg"],
                              n=groups[group]["n"],
                              passed=groups[group]["n"] > 0 and groups[group]["median_deg"] < limit)
                  for group, limit in ACCEPTANCE.items()}
    handedness = structural_handedness(rig)
    lateral_ok = all(v["agreement"] is None or v["agreement"] == 1. for v in lateral.values())
    mirror_ok = bool(probe["passed"] and lateral_ok and handedness["left_bones_at_positive_x"]
                     and handedness["right_bones_at_negative_x"])
    record = dict(
        schema_version=1, renderer="scripts/render_avatar.py", source=source_record,
        sample_id=state.sample_id, clock_id=state.clock_id, frames=len(state.timestamps),
        source_frames=full_frames, window=dict(start_seconds=args.start_seconds, max_seconds=args.max_seconds),
        retarget_binding=dict(sha256=binding.sha256(), version=binding.version, rig_axes=binding.rig_axes,
                              hinge_twist=not args.no_hinge_twist),
        bundle_sha256=bundle_sha, video=video.name, video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        resolution=args.resolution, camera=dict(kind="orthographic front", centre_source=centre.tolist(),
                                                span_source_units=span),
        direction_errors=errors, head_error_reference="stored head_rotation (the canonical skeleton has no head "
                                                     "orientation)",
        acceptance=dict(groups=acceptance, mirror=mirror_ok,
                        passed=bool(all(a["passed"] for a in acceptance.values()) and mirror_ok)),
        mirror=dict(structural=handedness, lateral_agreement=lateral, rendered_probe=probe),
        fill=retargeted.canonical.fill, posed_inferred_values=retargeted.canonical.posed_inferred_values,
        options=retargeted.options,
        face=dict(retargeted.face, units=sorted(FACE_UNIT_SOURCES),
                  unrepresented=UNREPRESENTED_BLENDSHAPES),
        deterministic_rerender=bool(np.array_equal(images[0], rerender)),
        distinct_frame_images=len({hashlib.sha256(image.tobytes()).hexdigest() for image in images}),
        encoding={k: v for k, v in encoded.items() if k != "ticks"}, verification=verified,
        blender=str(args.blender),
        limitations=[
            "Bone directions are copied; lengths are the MakeHuman default body's, so contacts "
            "(hand-to-face, hand-to-hand) are not preserved and no collision handling is applied.",
            "Twist about each limb axis is not observed by the swing-IK state; it is chosen "
            "(elbow hinge alignment, half the forearm-to-hand twist on lowerarm02).",
            "No eyebrow, hair or lip geometry: brow and mouth motion is visible only as skin deformation.",
            "Face coefficients are MediaPipe estimates rescaled against the clip median; "
            "left/right blendshape names are assumed to name the signer's own side.",
            "Workbench inspection render; not a qualified avatar appearance or a signing-fidelity result.",
        ],
        accepted_provenance=False, phase_exit_approved=False)
    (args.out_dir / "verification.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    summary = {g: (round(v["median_deg"], 4) if v["median_deg"] is not None else None,
                   round(v["p95_deg"], 4) if v["p95_deg"] is not None else None, v["n"])
               for g, v in groups.items()}
    print(json.dumps(dict(frames=record["frames"], acceptance=record["acceptance"]["passed"],
                          mirror=mirror_ok, errors_median_p95_n=summary,
                          deterministic=record["deterministic_rerender"],
                          max_clock_error_seconds=verified["max_clock_error_seconds"],
                          video=str(video)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
