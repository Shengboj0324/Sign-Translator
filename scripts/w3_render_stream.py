"""Render a posed canonical-state bundle to a timestamp-exact video.

Run with the project Python. The same file runs inside a separate factory-startup,
background Blender process that reads only the bundle and writes PNG frames; it
never touches an interactive session or downloads anything. The project Python
then encodes the frames with presentation timestamps taken from the canonical
clock and decodes the result to verify every timestamp.

Coordinates follow the existing probe's declared proper rotation
(x, y, z) -> (x, -z, y) with no unit scale. The Workbench engine with fixed
settings is used because it is deterministic; this is an inspection render, not
a qualified avatar appearance. ``--synthetic-demo`` renders a declared,
explicitly non-linguistic arm motion so the path can be exercised before any
real source exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def render_inside_blender(bundle: Path, out_dir: Path, resolution: int) -> None:
    import bpy
    import numpy as np

    with np.load(bundle, allow_pickle=False) as archive:
        vertices = archive["vertices"].astype(np.float64)
        faces = archive["faces"]
        renderable = archive["renderable"]
    converted = np.stack((vertices[..., 0], -vertices[..., 2], vertices[..., 1]), axis=-1)
    polygons = [tuple(int(v) for v in face if v >= 0)
                for face, keep in zip(faces, renderable) if keep]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (0.82, 0.70, 0.62)
    scene.display.render_aa = "OFF"
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("plain")
    scene.world.color = (0.18, 0.18, 0.18)

    mesh = bpy.data.meshes.new("posed")
    mesh.from_pydata(converted[0].tolist(), [], polygons)
    mesh.update()
    body = bpy.data.objects.new("posed", mesh)
    scene.collection.objects.link(body)

    used = sorted({index for polygon in polygons for index in polygon})
    extent = converted[:, used, :]
    low, high = extent.min(axis=(0, 1)), extent.max(axis=(0, 1))
    centre = (low + high) / 2
    span = float(max(high[0] - low[0], high[2] - low[2])) * 1.08
    camera_data = bpy.data.cameras.new("front")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = span
    camera = bpy.data.objects.new("front", camera_data)
    camera.location = (float(centre[0]), float(low[1]) - span, float(centre[2]))
    camera.rotation_euler = (1.5707963267948966, 0.0, 0.0)  # look along +Y
    scene.collection.objects.link(camera)
    scene.camera = camera

    flat = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    for index, frame in enumerate(converted):
        flat[:] = frame.reshape(-1)
        mesh.vertices.foreach_set("co", flat)
        mesh.update()
        scene.render.filepath = str(out_dir / f"frame-{index:06d}.png")
        bpy.ops.render.render(write_still=True)


def synthetic_demo_state(frames: int):
    """Declared, non-linguistic left-arm raise on a deliberately nonuniform clock."""
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from signtranslator.pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
    from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d
    clock = np.cumsum(np.where(np.arange(frames) % 3 == 0, 1 / 24, 1 / 36))
    clock = clock - clock[0]
    channels = {}
    for name, (kind, width, units) in CHANNELS.items():
        values = np.zeros((frames, 1, width), dtype=np.float64)
        if kind == "rotation_6d_columns":
            values[:] = [1, 0, 0, 0, 1, 0]
        elif kind == "unit_vector":
            values[:] = [1, 0, 0]
        label = name
        if name == "body":
            angles = torch.from_numpy(np.sin(np.pi * clock / clock[-1]) * 1.1)
            axis = torch.tensor([0., 0., 1.], dtype=torch.float64)
            values = matrix_to_rotation_6d(axis_angle_to_matrix(axis * angles[:, None])
                                           ).numpy().reshape(frames, 1, 6)
            label = "upperarm01.L"
        if name == "root_translation":
            label = "root"
        channels[name] = MotionChannel(
            values=values, valid=np.ones((frames, 1), dtype=bool),
            observed=np.zeros((frames, 1), dtype=bool),
            confidence=np.zeros((frames, 1), dtype=np.float64), labels=(label,),
            coordinate_frame="makehuman-bone-local", units=units,
            source_id="synthetic-demo", source_sha256="d" * 64,
            convention="synthetic-demo-v1",
            inference_method="declared-synthetic-demo-not-observed")
    return MultichannelMotion(clock.astype(np.float64), "synthetic-demo-clock",
                              "synthetic-demo", channels)


def main(argv=None) -> int:
    if "--inside-blender" in sys.argv:
        args = sys.argv[sys.argv.index("--inside-blender") + 1:]
        render_inside_blender(Path(args[0]), Path(args[1]), int(args[2]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="new directory")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--synthetic-demo", type=int, metavar="FRAMES")
    args = parser.parse_args(argv)
    if not args.synthetic_demo:
        parser.error("supply --synthetic-demo FRAMES until a real canonical state exists")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import numpy as np
    from signtranslator.avatar_render.makehuman import load_makehuman_rig
    from signtranslator.generation.state_rig import (
        BONE_LOCAL_AFTER_REST, ChannelBoneMapping, StateRigBinding,
    )
    from signtranslator.generation.stream_render import (
        encode_timestamped_video, verify_timestamped_video, write_frame_bundle,
    )
    from signtranslator.pose.multichannel import CHANNELS

    args.out_dir.mkdir(parents=False, exist_ok=False)
    rig = load_makehuman_rig(root / "Sign Translator Stage Documentation" / "evidence"
                             / "w1-w2-2026-09-26/source-intake/makehuman")
    state = synthetic_demo_state(args.synthetic_demo)
    binding = StateRigBinding(
        rotation_mappings=(ChannelBoneMapping("body", "upperarm01.L", "upperarm01.L"),),
        translation_mappings=(ChannelBoneMapping("root_translation", "root", "root"),),
        rest_bones=tuple(name for name in rig.bone_names if name != "upperarm01.L"),
        unmapped_channels=tuple(n for n in CHANNELS if n not in ("body", "root_translation")),
        metres_per_source_unit=.1, rotation_basis=BONE_LOCAL_AFTER_REST,
        asset_commit=rig.asset_commit)
    bundle = write_frame_bundle(rig, state, binding, args.out_dir / "bundle.npz",
                                allow_inferred=True)
    frames_dir = args.out_dir / "frames"
    frames_dir.mkdir()
    completed = subprocess.run(
        [str(args.blender), "--background", "--factory-startup", "--python-exit-code", "1",
         "--python", str(Path(__file__).resolve()), "--", "--inside-blender",
         str(bundle.path), str(frames_dir), str(args.resolution)],
        capture_output=True, text=True)
    (args.out_dir / "blender.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(f"Blender failed with {completed.returncode}; see blender.log")
    import av
    images = []
    for index in range(bundle.frames):
        with av.open(str(frames_dir / f"frame-{index:06d}.png")) as container:
            images.append(next(container.decode(video=0)).to_ndarray(format="rgb24"))
    video = args.out_dir / "stream.mp4"
    encoded = encode_timestamped_video(images, state.timestamps, video)
    verified = verify_timestamped_video(video, state.timestamps)
    distinct = len({hashlib.sha256(image.tobytes()).hexdigest() for image in images})
    record = dict(
        schema_version=1, bundle_sha256=bundle.sha256, bundle_provenance=bundle.provenance,
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        frames=bundle.frames, distinct_frame_images=distinct, resolution=args.resolution,
        encoding={k: v for k, v in encoded.items() if k != "ticks"},
        verification=verified, blender=str(args.blender),
        clock_seconds=[float(t) for t in state.timestamps],
        limitations=["Synthetic, declared, non-linguistic arm motion; not signing.",
                     "Workbench inspection render; not a qualified avatar appearance.",
                     "Only body faces are rendered; eyes, teeth and tongue are not attached here."],
        accepted_provenance=False, phase_exit_approved=False)
    (args.out_dir / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(frames=record["frames"], distinct=distinct,
                          max_clock_error_seconds=verified["max_clock_error_seconds"],
                          video_sha256=record["video_sha256"]), indent=2))
    del np
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
