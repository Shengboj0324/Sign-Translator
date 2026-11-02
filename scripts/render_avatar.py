"""Render a canonical state on the audited MakeHuman rig, with a verification record.

Run with the project Python::

    render_avatar.py --state STATE.npz [--sha256 HEX] --out-dir NEW_DIR --allow-inferred
    render_avatar.py --key R2_KEY --sha256 HEX [--store r2|dir:/path] --out-dir NEW_DIR --allow-inferred

The state's gaps are filled and its channels filtered (:mod:`signtranslator.avatar_render.motion_filter`), it is retargeted by
:mod:`signtranslator.avatar_render.retarget` (body, both hands with all 15 finger joints, head, face pose units, eyes), hands
and forearms inside the body are moved out along the camera axis (:mod:`signtranslator.avatar_render.collision`), it is skinned
in the project Python (body, eyes, teeth, tongue, top, hair, brows, lashes), and the posed meshes are written to a bundle. The same
file then runs inside a separate ``--background --factory-startup`` Blender process that reads only the bundle and renders
PNG frames with EEVEE (:data:`RENDER_SETTINGS`, a fixed orthographic front camera, the same frame for every clip: the viewer
sees the signer from the front, signer's left on the viewer's right). The frames are encoded to H.264 with presentation
timestamps from the state's own clock and decoded back to verify every timestamp.

**Every clip gets the same render.** No option here depends on what a clip is, nothing reads a clip's name or clock id, and
``render_settings_sha256`` in ``verification.json`` states what was used, so a packet can check it. The camera is not
fitted to the clip: a frame fitted to each clip would draw hands at a size that depends on the motion.

``verification.json`` records the retarget's angular errors against
``canonical_skeleton.skeleton_positions`` (median/p95/max per bone group, stored-valid
frames only), the head error against ``head_rotation``, three left/right checks
(structural, per-frame lateral agreement, and a rendered raised-left-arm probe),
which face coefficients were rendered, every filled frame, shake and articulation before and after the filter, hand and forearm
penetration and covered hands before and after the correction, a re-render determinism check and the timestamp verification. The
direction errors, acceptance and left/right checks read the pose as drawn, after the correction; the errors before it, and what the correction did to the arm in
the image, are kept beside them. Nothing is uploaded from here; ``archive_render`` is for callers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
INTAKE = ROOT / "Sign Translator Stage Documentation" / "evidence" / "w1-w2-2026-09-26" / "source-intake"
DEFAULT_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
OBJECTS = ("body", "eyes", "teeth", "tongue", "top", "hair", "brows", "lashes")
OBJECTS_WITH_TEXTURE = ("body", "hair", "brows", "lashes")
TOP = (0.13, 0.16, 0.23)        # one plain dark colour, as interpreters dress: hands contrast with it
#: Everything Blender is told. One table for every clip; its hash goes in verification.json and in a packet's key.
RENDER_SETTINGS = dict(
    engine="BLENDER_EEVEE", samples=32, shadow_rays=3, shadow_steps=6, view_transform="Standard",
    backdrop=(0.07, 0.08, 0.10),         # linear, what the camera sees: darker than the skin and lighter than the top by about the same ratio
    ambient=(0.20, 0.22, 0.26),          # linear, what lights the figure
    frame_centre=(0., 4.7), frame_span=9.,   # source units: hair to hip, hands out to 45 cm either side of the midline
    key=dict(energy=2200., location=(-6., -18., 14.), size=10., colour=(1., .97, .92)),
    fill=dict(energy=1000., location=(9., -16., 6.), size=12., colour=(.92, .96, 1.)),
    rim=dict(energy=1100., location=(3., 10., 16.), size=8., colour=(1., 1., 1.)),
    skin=dict(roughness=.6, specular=.2), cloth=dict(roughness=.95, specular=.05, subdivision=1),
    hair=dict(roughness=.7, specular=.2, alpha_threshold=.5),
    brows=dict(roughness=.8, specular=.1, alpha_threshold=.35, colour=(.07, .045, .03)),   # a mid brown: the texture's near-black read as a stern line
    lashes=dict(roughness=.8, specular=.1, alpha_threshold=.5), eye=dict(roughness=.3, specular=.25),
)
DEFAULT_OPTIONS = dict(face_baseline="clip_median", hinge_twist=True, drive_legs=False, lossless=False)
ACCEPTANCE = {"upper_arm": 5., "forearm": 5., "finger": 10., "thumb": 10.}


# --------------------------------------------------------------------------- inside Blender

def render_inside_blender(bundle: Path, out_dir: Path, resolution: int, first_only: bool) -> None:
    import bpy
    import mathutils
    import numpy as np

    archive = np.load(bundle, allow_pickle=False)
    settings = RENDER_SETTINGS
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = settings["engine"]
    scene.eevee.taa_render_samples = settings["samples"]
    scene.eevee.use_shadows = True
    scene.eevee.shadow_ray_count = settings["shadow_rays"]
    scene.eevee.shadow_step_count = settings["shadow_steps"]
    scene.view_settings.view_transform = settings["view_transform"]
    scene.view_settings.look = "None"
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("plain")
    scene.world.use_nodes = True
    nodes, links = scene.world.node_tree.nodes, scene.world.node_tree.links
    ambient, backdrop = nodes["Background"], nodes.new("ShaderNodeBackground")
    ambient.inputs[0].default_value = (*settings["ambient"], 1.)
    backdrop.inputs[0].default_value = (*settings["backdrop"], 1.)
    camera_ray, mix = nodes.new("ShaderNodeLightPath"), nodes.new("ShaderNodeMixShader")
    links.new(camera_ray.outputs["Is Camera Ray"], mix.inputs[0])
    links.new(ambient.outputs[0], mix.inputs[1])
    links.new(backdrop.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], nodes["World Output"].inputs[0])

    def converted(vertices):            # declared proper rotation (x, y, z) -> (x, -z, y)
        return np.stack((vertices[..., 0], -vertices[..., 2], vertices[..., 1]), axis=-1)

    def colour_material(name, values):
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        shader = nodes["Principled BSDF"]
        colour = nodes.new("ShaderNodeVertexColor")
        colour.layer_name = "colour"
        material.node_tree.links.new(colour.outputs[0], shader.inputs["Base Color"])
        shader.inputs["Roughness"].default_value = values["roughness"]
        shader.inputs["Specular IOR Level"].default_value = values["specular"]
        return material

    def texture_material(name, path, values, threshold=None):
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        shader = nodes["Principled BSDF"]
        image = nodes.new("ShaderNodeTexImage")
        image.image = bpy.data.images.load(path)
        image.interpolation = "Cubic"
        if "colour" in values:                         # the texture's alpha gives the shape, and this one colour replaces its near-black
            shader.inputs["Base Color"].default_value = (*values["colour"], 1.)
        else:
            links.new(image.outputs["Color"], shader.inputs["Base Color"])
        shader.inputs["Roughness"].default_value = values["roughness"]
        shader.inputs["Specular IOR Level"].default_value = values["specular"]
        if threshold is not None:                      # a hard cut, so no frame's noise decides a strand
            cut = nodes.new("ShaderNodeMath")
            cut.operation = "GREATER_THAN"
            cut.inputs[1].default_value = threshold
            links.new(image.outputs["Alpha"], cut.inputs[0])
            links.new(cut.outputs[0], shader.inputs["Alpha"])
        return material

    meshes = []
    for name in OBJECTS:
        if f"{name}.vertices" not in archive.files:
            continue
        vertices = converted(archive[f"{name}.vertices"].astype(np.float64))
        faces = archive[f"{name}.faces"]
        polygons = [tuple(int(v) for v in face if v >= 0) for face in faces]
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices[0].tolist(), [], polygons)
        mesh.update()
        mesh.shade_smooth()
        if f"{name}.corner_colours" in archive.files:
            colours = archive[f"{name}.corner_colours"]
            attribute = mesh.color_attributes.new("colour", "FLOAT_COLOR", "CORNER")
            rgba = np.concatenate([np.concatenate((colours[i, :len(p)], np.ones((len(p), 1))), axis=1)
                                   for i, p in enumerate(polygons)])
            attribute.data.foreach_set("color", rgba.astype(np.float64).reshape(-1))
            mesh.color_attributes.active_color = attribute
            mesh.materials.append(colour_material(name, settings["cloth" if name == "top" else "eye"]))
        else:
            uvs = archive[f"{name}.corner_uvs"]
            layer = mesh.uv_layers.new(name="uv")
            layer.data.foreach_set("uv", np.concatenate([uvs[i, :len(p)] for i, p in enumerate(polygons)]).reshape(-1))
            texture = str(ROOT / str(archive[f"{name}.texture"]))
            mesh.materials.append(texture_material(name, texture, settings["skin"]) if name == "body" else
                                  texture_material(name, texture, settings[name], settings[name]["alpha_threshold"]))
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)
        if name == "top":
            modifier = obj.modifiers.new("smooth", "SUBSURF")
            modifier.levels = modifier.render_levels = settings["cloth"]["subdivision"]
        meshes.append((mesh, vertices))

    x, y = settings["frame_centre"]
    camera_data = bpy.data.cameras.new("front")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = settings["frame_span"]
    camera_data.clip_start, camera_data.clip_end = 0.01, 1000.
    camera = bpy.data.objects.new("front", camera_data)
    camera.location = (float(x), -40., float(y))                    # in front of the signer (+z source)
    camera.rotation_euler = (1.5707963267948966, 0.0, 0.0)          # look along +Y (Blender) = -z source
    scene.collection.objects.link(camera)
    scene.camera = camera
    for name in ("key", "fill", "rim"):
        spec = settings[name]
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.size, light.color = spec["energy"], spec["size"], spec["colour"]
        light.use_shadow = True
        lamp = bpy.data.objects.new(name, light)
        lamp.location = spec["location"]
        aim = mathutils.Vector((0., 0., 5.)) - mathutils.Vector(spec["location"])
        lamp.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(lamp)

    frames = 1 if first_only else meshes[0][1].shape[0]
    for index in range(frames):
        for mesh, vertices in meshes:
            mesh.vertices.foreach_set("co", vertices[index].reshape(-1))
            mesh.update()
        scene.render.filepath = str(out_dir / f"frame-{index:06d}.png")
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


def _padded(faces):
    import numpy as np
    out = np.full((len(faces), max(len(face) for face in faces)), -1, dtype=np.int64)
    for i, face in enumerate(faces):
        out[i, :len(face)] = face
    return out


def frame() -> tuple["np.ndarray", float]:
    """The fixed front camera's centre and span in source units; the same for every clip, whatever its motion."""
    import numpy as np
    x, y = RENDER_SETTINGS["frame_centre"]
    return np.array([x, y, 0.]), float(RENDER_SETTINGS["frame_span"])


def project(points, centre, span, resolution):
    """(column, row) pixels of source points in the fixed orthographic front camera."""
    import numpy as np
    points = np.asarray(points, dtype=np.float64)
    column = (points[..., 0] - (centre[0] - span / 2)) / span * resolution
    row = ((centre[1] + span / 2) - points[..., 1]) / span * resolution
    return np.stack((column, row), axis=-1)


def pose_and_bundle(rig, retargeted, attachments, path: Path, *, deleted_vertices=None, skin_uvs=None, skin_texture=None,
                    extra: dict | None = None):
    """Skin every frame; write the bundle; return bone globals, body vertices and the frame."""
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
    arrays = {"body.vertices": vertices["body"], "clock": retargeted.clock}
    covered = set() if deleted_vertices is None else set(int(v) for v in deleted_vertices)
    keep = [i for i, groups in enumerate(rig.face_groups) if "body" in groups and not all(v in covered for v in rig.faces[i])]
    arrays["body.faces"] = _padded([rig.faces[i] for i in keep])
    if skin_uvs is None:
        arrays["body.corner_colours"] = np.full((len(keep), 4, 3), _srgb_to_linear(np.array((.8, .64, .54))))
    else:
        arrays["body.corner_uvs"] = skin_uvs(keep)
        arrays["body.texture"] = np.array(str(Path(skin_texture).resolve().relative_to(ROOT)))
    dropped = {}
    for name, attachment in attachments.items():
        arrays[f"{name}.vertices"] = vertices[name]
        if name == "top":
            arrays["top.faces"] = _padded(attachment.faces)
            arrays["top.corner_colours"] = np.full((len(attachment.faces), max(map(len, attachment.faces)), 3),
                                                   _srgb_to_linear(np.array(TOP)))
        elif name in OBJECTS_WITH_TEXTURE:
            arrays[f"{name}.faces"] = _padded(attachment.faces)
            uv = np.asarray(attachment.texcoords, dtype=np.float64)
            corners = np.zeros((len(attachment.faces), max(map(len, attachment.faces)), 2))
            for i, face_uv in enumerate(attachment.face_uvs):
                corners[i, :len(face_uv)] = uv[list(face_uv)]
            arrays[f"{name}.corner_uvs"] = corners
            arrays[f"{name}.texture"] = np.array(str(Path(attachment.texture_path).resolve().relative_to(ROOT)))
        else:
            kept, colours = _texture_corner_colours(attachment)
            arrays[f"{name}.faces"] = _padded([attachment.faces[i] for i in kept])
            arrays[f"{name}.corner_colours"] = colours
            dropped[name] = len(attachment.faces) - len(kept)
    centre, span = frame()
    arrays["camera.centre"], arrays["camera.span"] = centre, np.array(span)
    provenance = dict(retargeted.provenance(), objects=sorted(["body", *attachments]),
                      coordinate_conversion_to_blender="(x, y, z) -> (x, -z, y), proper rotation, no scale",
                      transparent_attachment_faces_dropped=dropped, **(extra or {}))
    arrays["provenance"] = np.array(json.dumps(provenance, sort_keys=True, default=float))
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
    return bone_globals, vertices["body"], (centre, span), hashlib.sha256(path.read_bytes()).hexdigest()


def run_blender(blender: Path, bundle: Path, frames_dir: Path, resolution: int, log: Path, *,
                first_only: bool = False) -> None:
    frames_dir.mkdir()
    completed = subprocess.run(
        [str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
         "--python", str(Path(__file__).resolve()), "--", "--inside-blender",
         str(bundle), str(frames_dir), str(resolution), "1" if first_only else "0"],
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


def mirror_probe(rig, binding, out_dir: Path, blender: Path, top) -> dict:
    """Render a synthetic raised-left-arm state and check where the arm lands in the image (top on, no hair: hair stands above the head)."""
    import numpy as np
    from signtranslator.avatar_render.retarget import retarget_state
    from signtranslator.avatar_render.retarget_check import image_mirror_check, raised_arm_points, state_from_points
    world, visible, hands = raised_arm_points(rig, "Left")
    state = state_from_points(np.array([0.]), world, visible, hands, sample_id="mirror-probe-raised-left-arm")
    retargeted = retarget_state(rig, binding, state, allow_inferred=True)
    probe = out_dir / "mirror-probe"
    probe.mkdir()
    bone_globals, body, (centre, span), _ = pose_and_bundle(rig, retargeted, {"top": top}, probe / "bundle.npz")
    resolution = 128
    run_blender(blender, probe / "bundle.npz", probe / "frames", resolution, probe / "blender.log")
    image = read_png(probe / "frames" / "frame-000000.png")
    index = {name: i for i, name in enumerate(rig.bone_names)}
    head_bones = [i for i, n in enumerate(rig.bone_names) if n == "head"]
    dominant = rig.weights.detach().cpu().numpy().argmax(1)
    head_top = float(body[0][np.isin(dominant, head_bones), 1].max())
    row = int(np.floor(project([[0., head_top, 0.]], centre, span, resolution)[0, 1])) - 2
    check = image_mirror_check(image, image[0, 0], head_top_row=max(row, 1))
    wrist = bone_globals[0, index["wrist.L"], :3, 3]
    check.update(left_wrist_source_xyz=wrist.tolist(),
                 left_wrist_pixel=project(wrist, centre, span, resolution).tolist(),
                 left_wrist_above_head=bool(wrist[1] > head_top), resolution=resolution,
                 rule="the signer's left is drawn on the viewer's right (front view of a signer facing the camera)")
    check["passed"] = bool(check["passed"] and check["left_wrist_above_head"] and wrist[0] > 0)
    return check


#: Every file that shapes a video, by path from the repo root; the assets are hashed by folder.
RENDER_SOURCES = ("scripts/render_avatar.py", "signtranslator/avatar_render/collision.py", "signtranslator/avatar_render/motion_filter.py",
                  "signtranslator/avatar_render/motion_quality.py", "signtranslator/avatar_render/retarget.py", "signtranslator/avatar_render/retarget_check.py",
                  "signtranslator/avatar_render/makehuman.py", "signtranslator/avatar_render/makehuman_attachment.py", "signtranslator/avatar_render/makehuman_eyes.py",
                  "signtranslator/avatar_render/makehuman_face.py", "signtranslator/avatar_render/makehuman_mouth.py", "signtranslator/avatar_render/makehuman_wardrobe.py",
                  "signtranslator/avatar_render/rigged.py", "signtranslator/generation/stream_render.py", "signtranslator/generation/canonical_skeleton.py",
                  "signtranslator/data_engineering/video_landmarks.py", "signtranslator/pose/multichannel.py")
RENDER_ASSETS = ("makehuman", "makehuman-eyes", "makehuman-face", "makehuman-mouth", "makehuman-system-assets")
ENCODER = dict(codec="libx264", pixel_format="yuv420p", crf=16, preset="slow")          # stream_render's "review" quality, stated here for the hash
_file_hashes: dict = {}


def _hash_of_file(path: Path) -> str:
    key = str(path)
    if key not in _file_hashes:
        _file_hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _file_hashes[key]


def source_hashes() -> dict:
    """The hash of every source file and asset behind a render."""
    out = {name: _hash_of_file(ROOT / name) for name in RENDER_SOURCES}
    for folder in RENDER_ASSETS:
        for path in sorted(p for p in (INTAKE / folder).rglob("*") if p.is_file()):
            out[str(path.relative_to(INTAKE))] = _hash_of_file(path)
    return out


def blender_version(blender: Path = DEFAULT_BLENDER) -> str:
    """The first line ``blender --version`` prints; ``absent`` if it isn't installed."""
    key = f"version:{blender}"
    if key not in _file_hashes:
        try:
            _file_hashes[key] = subprocess.run([str(blender), "--version"], capture_output=True, text=True, timeout=60).stdout.splitlines()[0].strip()
        except (OSError, IndexError, subprocess.TimeoutExpired):
            _file_hashes[key] = "absent"
    return _file_hashes[key]


def settings_payload(resolution: int, *, options: dict | None = None, blender: Path = DEFAULT_BLENDER) -> dict:
    """Everything that shapes a video apart from its state."""
    import av
    from signtranslator.avatar_render import collision
    from signtranslator.avatar_render.motion_filter import FILTER_V1
    return dict(render=RENDER_SETTINGS, filter=FILTER_V1.describe(), resolution=resolution, options=options or DEFAULT_OPTIONS, top=TOP,
                collision=dict(settings=collision.CORRECTION_V2.__dict__, clearance=collision.CLEARANCE, rounds=collision.ROUNDS, dilate_frames=collision.DILATE_FRAMES,
                               smooth_sigma_frames=collision.SMOOTH_SIGMA_FRAMES, reach_margin=collision.REACH_MARGIN, front_cell=collision.FRONT_CELL,
                               front_samples=collision.FRONT_SAMPLES, swivel_step_deg=collision.SWIVEL_STEP_DEG),
                encoder=dict(ENCODER, pyav=av.__version__, libraries={k: list(v) for k, v in av.library_versions.items()}),
                blender=blender_version(blender), sources=source_hashes())


def settings_sha256(resolution: int, *, options: dict | None = None, blender: Path = DEFAULT_BLENDER) -> str:
    """The hash of everything that shapes a video apart from its state. Two videos with one hash had one render."""
    payload = settings_payload(resolution, options=options, blender=blender)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def code_commit() -> dict:
    """The commit this renderer runs from, and whether any file that shapes a video differs from it."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()     # noqa: E731
    return dict(commit=git("rev-parse", "HEAD") or "unknown", dirty=bool(git("status", "--porcelain", "--", *RENDER_SOURCES)))


def archive_render(store, prefix: str, directory: Path, fields: dict | None = None) -> dict:
    """Every file of a render under ``prefix`` with ``INDEX.json`` (bytes, SHA-256), as the owner's rule asks. The store re-reads what it writes."""
    from concurrent.futures import ThreadPoolExecutor
    prefix = prefix.rstrip("/")
    paths = sorted(p for p in directory.rglob("*") if p.is_file())

    def put(path):
        payload = path.read_bytes()
        digest = store.put(f"{prefix}/{path.relative_to(directory).as_posix()}", payload)
        return path.relative_to(directory).as_posix(), dict(bytes=len(payload), sha256=digest)

    with ThreadPoolExecutor(8) as pool:
        files = dict(pool.map(put, paths))
    index = dict(fields or {}, files=files)
    store.put(f"{prefix}/INDEX.json", json.dumps(index, indent=1, sort_keys=True).encode())
    return dict(prefix=prefix, files=len(files), bytes=sum(f["bytes"] for f in files.values()))


def motion_quality_record(rig, before, after, fps, settings, gaps_filled, source_units_per_pixel, shake, articulation, preservation, strip) -> dict:
    """Shake and articulation of this clip, retargeted without and with the filter (fills on both sides)."""
    pixels_per_unit = 1. / source_units_per_pixel
    a0, a1 = articulation(rig, before.local_rotations, fps), articulation(rig, after.local_rotations, fps)
    kept = preservation(a0, a1, fps)
    return dict(settings=settings.describe(), settings_sha256=settings.sha256(), gaps_filled=gaps_filled,
                shake_before=shake(rig, before.local_rotations, fps, screen_pixels_per_unit=pixels_per_unit),
                shake_after=shake(rig, after.local_rotations, fps, screen_pixels_per_unit=pixels_per_unit),
                articulation_before=strip(a0), articulation_after=strip(a1), preserved=kept)


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
    parser.add_argument("--resolution", type=int, default=768)
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
    from signtranslator.avatar_render.makehuman_wardrobe import body_corner_uvs, load_wardrobe
    from signtranslator.avatar_render.collision import apply_correction, arm_image_changes, build_field
    from signtranslator.avatar_render.motion_filter import FILTER_V1, fill_gaps, filter_state
    from signtranslator.avatar_render.motion_quality import articulation, forward_kinematics, preservation, shake, strip_arrays
    from signtranslator.avatar_render.retarget import (
        FACE_UNIT_SOURCES, UNREPRESENTED_BLENDSHAPES, build_retarget_binding, canonical_motion, retarget_state,
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
    wardrobe = load_wardrobe(INTAKE / "makehuman-system-assets", rig)
    attachments = {"eyes": load_makehuman_eyes(INTAKE / "makehuman-eyes", rig),
                   **load_makehuman_mouth(INTAKE / "makehuman-mouth", rig), **wardrobe.attachments}
    binding = build_retarget_binding(rig)
    for attachment in attachments.values():
        if attachment.rig_identity != binding.rig_identity:
            raise SystemExit("attachment fitted to a different rig")
    retarget_options = dict(allow_inferred=args.allow_inferred, face_units=face_units, face_baseline=args.face_baseline,
                            hinge_twist=not args.no_hinge_twist, drive_legs=args.drive_legs)
    stored = canonical_motion(state, allow_inferred=args.allow_inferred, rest_locals=binding.rest_locals())
    filled, gaps_filled = fill_gaps(state)
    filtered = filter_state(filled, FILTER_V1)
    fps = 1. / float(np.median(np.diff(state.timestamps))) if len(state.timestamps) > 1 else 24.
    before_filter = retarget_state(rig, binding, filled, **retarget_options)
    uncorrected = retarget_state(rig, binding, filtered, **retarget_options)
    retargeted, collision = apply_correction(rig, build_field(rig, drawn=wardrobe.attachments), uncorrected)
    # Checks read the retarget of the filtered state on the frames valid as stored; fills are never scored.
    checked = replace(uncorrected, canonical=replace(uncorrected.canonical, raw_valid=stored.raw_valid, chain_valid=stored.chain_valid))
    motion_record = None if len(state.timestamps) < 12 else motion_quality_record(rig, before_filter, uncorrected, fps, FILTER_V1, gaps_filled, frame()[1] / args.resolution,
                                          shake, articulation, preservation, strip_arrays)
    bundle_path = args.out_dir / "bundle.npz"
    _, _, (centre, span), bundle_sha = pose_and_bundle(
        rig, retargeted, attachments, bundle_path, deleted_vertices=wardrobe.body_deleted_vertices,
        skin_uvs=lambda keep: body_corner_uvs(INTAKE / "makehuman", rig, keep), skin_texture=wardrobe.skin_texture,
        extra=dict(source=source_record))
    # Acceptance reads the pose raters see, after the correction; the retarget before it is kept beside.
    before_correction = forward_kinematics(rig, uncorrected.local_rotations)
    drawn = forward_kinematics(rig, retargeted.local_rotations)
    errors = direction_errors(rig, filtered, checked, drawn)
    errors_before_correction = direction_errors(rig, filtered, checked, before_correction)
    lateral = lateral_agreement(rig, filtered, checked, drawn)
    arm_in_image = arm_image_changes(rig, before_correction, drawn, args.resolution / frame()[1])
    probe = mirror_probe(rig, binding, args.out_dir, args.blender, wardrobe.attachments["top"])

    frames_dir = args.out_dir / "frames"
    render_started = time.perf_counter()
    run_blender(args.blender, bundle_path, frames_dir, args.resolution, args.out_dir / "blender.log")
    blender_seconds = time.perf_counter() - render_started
    images = [read_png(frames_dir / f"frame-{i:06d}.png") for i in range(len(state.timestamps))]
    # A second process renders the first frame: EEVEE's shadow cache makes a re-render inside one session differ by a few levels.
    run_blender(args.blender, bundle_path, args.out_dir / "rerender", args.resolution, args.out_dir / "rerender.log", first_only=True)
    rerender = read_png(args.out_dir / "rerender" / "frame-000000.png")
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
        resolution=args.resolution, camera=dict(kind="orthographic front, one frame for every clip", centre_source=centre.tolist(),
                                                span_source_units=span),
        render_settings=RENDER_SETTINGS, render_settings_sha256=settings_sha256(args.resolution, blender=args.blender, options=dict(
            face_baseline=args.face_baseline, hinge_twist=not args.no_hinge_twist, drive_legs=args.drive_legs, lossless=args.lossless)),
        code=code_commit(),
        motion=motion_record, collision=collision,
        assets=dict(wardrobe_licence=wardrobe.licence, wardrobe_source=wardrobe.source),
        direction_errors=errors, direction_errors_before_correction=errors_before_correction, arm_in_image_after_correction=arm_in_image,
        head_error_reference="stored head_rotation (the canonical skeleton has no head "
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
        blender=str(args.blender), blender_seconds=blender_seconds,
        limitations=[
            "Bone directions are copied; lengths are the MakeHuman default body's, so contacts "
            "(hand-to-face, hand-to-hand) are not preserved. A hand covered by the drawn body is moved in front of it along the camera axis, "
            "with the elbow kept near where it was drawn; a forearm may stay inside the torso; hand against hand or against the other arm is not handled.",
            "Twist about each limb axis is not observed by the swing-IK state; it is chosen "
            "(elbow hinge alignment, half the forearm-to-hand twist on lowerarm02).",
            "No lip geometry: mouth motion is visible only as skin deformation. Hair does not move on its own.",
            "Face coefficients are MediaPipe estimates rescaled against the clip median; "
            "left/right blendshape names are assumed to name the signer's own side.",
            "EEVEE render; not a qualified avatar appearance or a signing-fidelity result.",
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
