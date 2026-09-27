"""Run in Blender after loading rig-diagnostic.blend; inspect and replay saved state."""
import hashlib
import json
from pathlib import Path

import bpy
import numpy as np

out = Path(bpy.data.filepath).parent
payload = json.loads((out/'mesh-probe.json').read_text())
record = json.loads((out/'render-record.json').read_text())
obj = bpy.data.objects['Body_only_helpers_excluded']

def converted(values):
    values = np.array(values, dtype=np.float64)
    return values[:, [0, 2, 1]]*np.array([1, -1, 1])

rest = np.array([v.co[:] for v in obj.data.vertices], dtype=np.float64)
pose = np.array([v.co[:] for v in obj.data.shape_keys.key_blocks['Nonlinguistic_articulation_probe'].data], dtype=np.float64)
rest_error = float(np.abs(rest-converted(payload['rest_vertices'])).max())
pose_error = float(np.abs(pose-converted(payload['posed_vertices'])).max())
assert max(rest_error, pose_error) < 1e-6
assert [list(p.vertices) for p in obj.data.polygons] == payload['body_faces']
assert obj['asset_commit'] == payload['asset_commit']
assert not any(m.type == 'ARMATURE' for m in obj.modifiers)  # Baked shape-key diagnostic, not a native armature claim.
attachments = dict(payload.get('mouth', {}))
attachments['eyes'] = payload['eyes']
for kind, expected in attachments.items():
    name = 'FittedEyes_not_gaze_qualified' if kind == 'eyes' else 'Fitted_'+kind
    item = bpy.data.objects[name]
    assert [list(p.vertices) for p in item.data.polygons] == expected['faces']
    for key, field in [('Basis', 'rest_vertices'), ('Nonlinguistic_attachment_probe', 'posed_vertices')]:
        actual = np.array([v.co[:] for v in item.data.shape_keys.key_blocks[key].data], dtype=np.float64)
        assert np.abs(actual-converted(expected[field])).max() < 1e-6
    layer = item.data.uv_layers['SourceUV']
    for polygon, uv_indices in zip(item.data.polygons, expected['face_uvs'], strict=True):
        for loop, index in zip(polygon.loop_indices, uv_indices, strict=True):
            assert np.max(np.abs(np.array(layer.data[loop].uv[:])-expected['texcoords'][index])) < 1e-6
    textures = [n.image for n in item.active_material.node_tree.nodes if n.type == 'TEX_IMAGE']
    assert len(textures) == 1 and textures[0].packed_file is not None
    assert hashlib.sha256(bytes(textures[0].packed_file.data)).hexdigest() == expected['texture_sha256']
assert bpy.context.scene.unit_settings.system == 'METRIC'
assert abs(bpy.context.scene.unit_settings.scale_length-0.1) < 1e-8
scene = bpy.context.scene
results = []
for view in record['views']:
    scene.frame_set(view['frame'])
    scene.camera.location = view['camera_location']
    scene.camera.rotation_euler = view['camera_rotation_euler']
    scene.camera.data.ortho_scale = view['ortho_scale']
    scene.render.resolution_x, scene.render.resolution_y = view['width'], view['height']
    path = out/(view['name']+'-reload.png')
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    original_image = bpy.data.images.load(str(out/(view['name']+'.png')), check_existing=False)
    replay_image = bpy.data.images.load(str(path), check_existing=False)
    original_pixels = np.asarray(original_image.pixels[:], dtype=np.float32)
    replay_pixels = np.asarray(replay_image.pixels[:], dtype=np.float32)
    assert original_image.size[:] == replay_image.size[:]
    pixel_equal = bool(np.array_equal(original_pixels, replay_pixels))
    results.append({'view': view['name'], 'sha256': digest,
                    'byte_identical': digest == view['sha256'], 'decoded_pixels_identical': pixel_equal})
    bpy.data.images.remove(original_image)
    bpy.data.images.remove(replay_image)
report = {'blender_version': bpy.app.version_string,
          'max_rest_float32_storage_error_source_units': rest_error,
          'max_posed_float32_storage_error_source_units': pose_error,
          'body_faces_preserved': len(obj.data.polygons), 'views': results,
          'all_byte_identical': all(x['byte_identical'] for x in results),
          'all_decoded_pixels_identical': all(x['decoded_pixels_identical'] for x in results),
          'png_metadata_note': 'File path, timestamps and render timings differ; file hashes are not pixel hashes',
          'scope': 'Same-machine saved Blender reload; not cross-platform determinism or native rig runtime parity',
          'phase_exit_approved': False}
(out/'reload-verification.json').write_text(json.dumps(report,indent=2)+'\n')
assert report['all_decoded_pixels_identical'], 'Saved-state pixels differ; see report'
