"""Create and render an explicitly non-linguistic probe of the pinned rig.

Run with the project Python and --asset-root, --out-dir (new), --blender.
The same file runs inside a separate factory-startup Blender process. It never
changes an interactive Blender session or downloads assets. Source coordinates
are converted by (x,y,z)->(x,-z,y), a declared proper rotation, with no unit scale.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_bundle(bundle_path: Path) -> None:
    import bpy
    from mathutils import Vector

    payload = json.loads(bundle_path.read_text())
    out = bundle_path.parent
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 24
    scene.cycles.seed = 0
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = payload["source_meters_per_unit"]
    scene.world.color = (0.18, 0.18, 0.18)
    scene.view_settings.view_transform = "Standard"

    def point(value):
        x, y, z = value
        return (x, -z, y)

    mesh = bpy.data.meshes.new("PinnedBaseBody")
    mesh.from_pydata([point(v) for v in payload["rest_vertices"]], [], payload["body_faces"])
    mesh.update()
    obj = bpy.data.objects.new("Body_only_helpers_excluded", mesh)
    scene.collection.objects.link(obj)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    material = bpy.data.materials.new("Neutral diagnostic clay")
    material.diffuse_color = (0.32, 0.53, 0.64, 1)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = material.diffuse_color
    bsdf.inputs["Roughness"].default_value = 0.65
    obj.data.materials.append(material)
    obj.shape_key_add(name="Basis")
    key = obj.shape_key_add(name="Nonlinguistic_articulation_probe")
    for vertex, value in zip(key.data, payload["posed_vertices"], strict=True):
        vertex.co = point(value)
    key.value = 0
    key.keyframe_insert(data_path="value", frame=1)
    key.value = 1
    key.keyframe_insert(data_path="value", frame=2)
    scene.frame_start, scene.frame_end = 1, 2
    obj["asset_commit"] = payload["asset_commit"]
    obj["qualification"] = "Geometry probe only; no observed motion or linguistic acceptance"
    obj["coordinate_conversion"] = "(x,y,z)->(x,-z,y); source units unchanged"
    obj["rest_geometry_kind"] = payload.get("rest_geometry", {}).get("kind", "pinned_raw_base")
    obj["rest_geometry_sha256"] = payload.get("rest_geometry", {}).get("sha256", "")

    attachments = dict(payload.get("mouth", {}))
    if "eyes" in payload:
        attachments["eyes"] = payload["eyes"]
    for kind, eye in attachments.items():
        eye_mesh = bpy.data.meshes.new("PinnedFitted_"+kind)
        eye_mesh.from_pydata([point(v) for v in eye["rest_vertices"]], [], eye["faces"])
        eye_mesh.update()
        eye_obj = bpy.data.objects.new("FittedEyes_not_gaze_qualified" if kind == "eyes" else "Fitted_"+kind, eye_mesh)
        scene.collection.objects.link(eye_obj)
        layer = eye_mesh.uv_layers.new(name="SourceUV")
        for polygon, face_uvs in zip(eye_mesh.polygons, eye["face_uvs"], strict=True):
            polygon.use_smooth = True
            for loop_index, uv_index in zip(polygon.loop_indices, face_uvs, strict=True):
                layer.data[loop_index].uv = eye["texcoords"][uv_index]
        eye_obj.shape_key_add(name="Basis")
        eye_key = eye_obj.shape_key_add(name="Nonlinguistic_attachment_probe")
        for vertex, value in zip(eye_key.data, eye["posed_vertices"], strict=True):
            vertex.co = point(value)
        eye_key.value = 0
        eye_key.keyframe_insert(data_path="value", frame=1)
        eye_key.value = 1
        eye_key.keyframe_insert(data_path="value", frame=2)
        eye_material = bpy.data.materials.new("Pinned texture_"+kind)
        eye_material.use_nodes = True
        eye_bsdf = eye_material.node_tree.nodes.get("Principled BSDF")
        texture = eye_material.node_tree.nodes.new("ShaderNodeTexImage")
        if digest(Path(eye["texture_path"])) != eye["texture_sha256"]:
            raise ValueError("Eye texture identity changed after export")
        texture.image = bpy.data.images.load(eye["texture_path"], check_existing=False)
        texture.image.pack()
        eye_material.node_tree.links.new(texture.outputs["Color"], eye_bsdf.inputs["Base Color"])
        eye_material.node_tree.links.new(texture.outputs["Alpha"], eye_bsdf.inputs["Alpha"])
        eye_bsdf.inputs["Roughness"].default_value = 0.3
        eye_obj.data.materials.append(eye_material)

    def light(name, position, energy, size):
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.shape, data.size = energy, "DISK", size
        item = bpy.data.objects.new(name, data)
        scene.collection.objects.link(item)
        item.location = position
        direction = Vector((0, 0, 3))-item.location
        item.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    light("Key", (7, -12, 14), 2600, 8)
    light("Fill", (-8, -6, 5), 1500, 7)
    light("Rim", (0, 5, 12), 2200, 6)
    camera_data = bpy.data.cameras.new("DiagnosticCamera")
    camera = bpy.data.objects.new("DiagnosticCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "ORTHO"
    camera_data.clip_start, camera_data.clip_end = 0.01, 200
    hand_target, hand_scale = (4.45, 2.1, 2.4), 2.7
    if payload.get("inspection_hand_vertex_indices"):
        indices = payload["inspection_hand_vertex_indices"]
        points = [Vector(payload[field][i]) for field in ("rest_vertices", "posed_vertices") for i in indices]
        lower = Vector(tuple(min(p[k] for p in points) for k in range(3)))
        upper = Vector(tuple(max(p[k] for p in points) for k in range(3)))
        center = (lower + upper) * 0.5
        # A bounding sphere fits the square view for any fixed camera direction.
        hand_scale = 2.2 * max((p-center).length for p in points)
        if hand_scale <= 0:
            raise ValueError("Hand inspection geometry has zero extent")
        hand_target = tuple(center)
    views = [
        ("body-rest", 1, (0, 0, 0), (0, 0, 30), 19, 640, 800),
        ("hand-rest", 1, hand_target, (1.5, 0.4, 10), hand_scale, 800, 800),
        ("hand-articulated", 2, hand_target, (1.5, 0.4, 10), hand_scale, 800, 800),
        ("face-rest", 1, (0, 7.2, 1.0), (0, 0, 10), 3.1, 800, 800),
        ("face-articulated", 2, (0, 7.2, 1.0), (0, 0, 10), 3.1, 800, 800),
    ]
    if "mouth" in payload:
        views.append(("mouth-articulated", 2, (0, 6.55, 1.25), (0, 0, 10), 1.6, 800, 600))
    records = []
    for name, frame, target, offset, scale, width, height in views:
        scene.frame_set(frame)
        focus = Vector(point(target))
        camera.location = focus + Vector(point(offset))
        camera.rotation_euler = (focus-camera.location).to_track_quat("-Z", "Y").to_euler()
        camera_data.ortho_scale = scale
        scene.render.resolution_x, scene.render.resolution_y = width, height
        scene.render.filepath = str(out / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        records.append({"name": name, "frame": frame, "camera_location": list(camera.location),
                        "camera_rotation_euler": list(camera.rotation_euler), "ortho_scale": scale,
                        "width": width, "height": height, "sha256": digest(out/f"{name}.png")})
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/"rig-diagnostic.blend"))
    (out/"render-record.json").write_text(json.dumps({
        "blender_version": bpy.app.version_string, "render_engine": scene.render.engine,
        "samples": scene.cycles.samples, "seed": scene.cycles.seed,
        "bundle_sha256": digest(bundle_path), "views": records,
        "blend_sha256": digest(out/"rig-diagnostic.blend"),
        "observed_motion": False, "linguistic_acceptance": False,
        "phase_exit_approved": False,
    }, indent=2)+"\n")


def export_and_render() -> None:
    import torch
    from signtranslator.avatar_render.makehuman import SOURCE_METERS_PER_UNIT, load_makehuman_rig
    from signtranslator.pose.rotations import axis_angle_to_matrix

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--eyes-root", type=Path)
    parser.add_argument("--mouth-root", type=Path)
    parser.add_argument("--jaw-angle-radians", type=float, default=0.10)
    parser.add_argument("--face-units-root", type=Path)
    parser.add_argument("--face-unit")
    parser.add_argument("--rest-mesh-npy", type=Path)
    parser.add_argument("--rest-mesh-sha256")
    args = parser.parse_args()
    if (args.rest_mesh_npy is None) != (args.rest_mesh_sha256 is None):
        raise ValueError("--rest-mesh-npy and --rest-mesh-sha256 must be supplied together")
    rest = None
    rest_geometry = {"kind": "pinned_raw_base"}
    if args.rest_mesh_npy is not None:
        import numpy as np
        expected = args.rest_mesh_sha256
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("Rest mesh SHA-256 must be 64 lowercase hexadecimal characters")
        raw = args.rest_mesh_npy.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Rest mesh identity mismatch")
        values = np.load(io.BytesIO(raw), allow_pickle=False)
        if not isinstance(values, np.ndarray) or values.dtype != np.dtype("float64"):
            raise ValueError("Rest mesh must be a float64 NPY array")
        rest = torch.from_numpy(values.copy())
        rest_geometry = {"kind": "explicit_rebuilt_rest", "path": str(args.rest_mesh_npy.resolve()),
                         "sha256": expected, "provenance_accepted": False}
    rig = load_makehuman_rig(args.asset_root, rest_vertices=rest)
    if not args.blender.is_file():
        raise ValueError("Blender executable not found")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    angles = {name: [0.35, 0, 0] for name in rig.bone_names if name.startswith("finger") and name.endswith(".L")}
    angles.update({"jaw": [args.jaw_angle_radians, 0, 0], "eye.L": [0, 0, 0.15], "eye.R": [0, 0, 0.15]})
    local = torch.zeros(len(rig.bone_names), 3, dtype=torch.float64)
    for name, value in angles.items():
        local[rig.bone_names.index(name)] = torch.tensor(value, dtype=torch.float64)
    rotations = axis_angle_to_matrix(local)
    if (args.face_unit is None) != (args.face_units_root is None):
        raise ValueError("--face-unit and --face-units-root must be supplied together")
    if args.face_unit is not None:
        from signtranslator.avatar_render.makehuman_face import load_makehuman_face_units
        rotations = load_makehuman_face_units(args.face_units_root, rig).rotations(args.face_unit)
        angles = {}  # Source catalogue angles were converted into bone-rest bases.
    posed = rig.pose(rotations)
    selected = [face for face, groups in zip(rig.faces, rig.face_groups, strict=True) if "body" in groups]
    hand_bones = [i for i, name in enumerate(rig.bone_names)
                  if name.endswith(".L") and (name.startswith("finger") or name in ("hand.L", "wrist.L"))]
    body_vertices = {v for face in selected for v in face}
    hand_support = rig.weights[:, hand_bones].sum(1) > 0.05
    inspection_hand_vertices = [i for i in sorted(body_vertices) if bool(hand_support[i])]
    if not inspection_hand_vertices:
        raise ValueError("No supported visible hand region for inspection")
    payload = {
        "asset_commit": rig.asset_commit, "coordinate_units": rig.coordinate_units,
        "rest_geometry": rest_geometry,
        "source_meters_per_unit": SOURCE_METERS_PER_UNIT,
        "probe_type": "authored static face pose; not observed signing" if args.face_unit else "invented articulation for geometry debugging; not an ASL sign",
        "authored_face_unit": args.face_unit,
        "baseline": "target bind pose; not the named catalogue Rest row",
        "local_rotation_matrices": rotations.tolist(),
        "local_axis_angles_radians": angles, "bone_names": rig.bone_names,
        "rest_vertices": rig.rest_vertices.tolist(), "posed_vertices": posed.vertices.tolist(),
        "body_faces": selected, "all_face_group_counts": dict(Counter(g for gs in rig.face_groups for g in gs)),
        "inspection_hand_vertex_indices": inspection_hand_vertices,
        "inspection_hand_support_threshold": 0.05,
        "excluded_face_count": len(rig.faces)-len(selected),
        "manual_axis_angle_count": len(angles),
        "pose_rotation_count": int(((rotations-torch.eye(3, dtype=rotations.dtype)).abs().amax(dim=(-1, -2)) > 1e-12).sum()),
        "pose_rotation_count_matrix_threshold": 1e-12, "observed_motion": False,
        "adapter_sha256": digest(Path(__file__).resolve().parents[1]/"signtranslator/avatar_render/makehuman.py"),
        "script_sha256": digest(Path(__file__).resolve()),
    }
    if args.eyes_root is not None:
        from signtranslator.avatar_render.makehuman_eyes import load_makehuman_eyes
        eyes = load_makehuman_eyes(args.eyes_root, rig)
        eye_pose = eyes.pose(rotations)
        payload["eyes"] = {
            "rest_vertices": eyes.rest_vertices.tolist(), "posed_vertices": eye_pose.tolist(),
            "faces": eyes.faces, "texcoords": eyes.texcoords, "face_uvs": eyes.face_uvs,
            "texture_path": str(eyes.texture_path), "texture_sha256": digest(eyes.texture_path),
            "negative_geometry_coefficients_preserved": eyes.negative_geometry_coefficients,
            "adapter_sha256": digest(Path(__file__).resolve().parents[1]/"signtranslator/avatar_render/makehuman_eyes.py"),
        }
    if args.mouth_root is not None:
        from signtranslator.avatar_render.makehuman_mouth import load_makehuman_mouth
        payload["mouth"] = {}
        for kind, attachment in load_makehuman_mouth(args.mouth_root, rig).items():
            payload["mouth"][kind] = {
                "rest_vertices": attachment.rest_vertices.tolist(),
                "posed_vertices": attachment.pose(rotations).tolist(),
                "faces": attachment.faces, "texcoords": attachment.texcoords,
                "face_uvs": attachment.face_uvs, "texture_path": str(attachment.texture_path),
                "texture_sha256": digest(attachment.texture_path),
                "negative_geometry_coefficients_preserved": attachment.negative_geometry_coefficients,
            }
    bundle = out/"mesh-probe.json"
    bundle.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False)+"\n")
    command = [str(args.blender.resolve()), "--background", "--factory-startup", "--python-exit-code", "1",
               "--python", str(Path(__file__).resolve()), "--", "--render-bundle", str(bundle)]
    (out/"command.json").write_text(json.dumps(command, indent=2)+"\n")
    with (out/"blender.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(out/"render-record.json")


if __name__ == "__main__":
    if "--" in sys.argv and sys.argv[sys.argv.index("--")+1:][:1] == ["--render-bundle"]:
        custom = sys.argv[sys.argv.index("--")+1:]
        if len(custom) != 2:
            raise ValueError("Expected exactly --render-bundle PATH")
        render_bundle(Path(custom[1]))
    else:
        export_and_render()
