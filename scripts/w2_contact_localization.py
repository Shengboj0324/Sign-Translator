"""Blender-only localization of static crossing candidates; no geometry repair."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def main(root, out):
    spec = importlib.util.spec_from_file_location('crossing_audit', Path(__file__).with_name('w2_face_contact_audit.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = json.loads((root/'catalogue.json').read_text())
    archive = root/'posed-meshes.npz'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != meta['mesh_archive_sha256']:
        raise ValueError('Mesh archive changed')
    out.mkdir(parents=True, exist_ok=False)
    results = []
    selected = ['TargetBind', 'LeftUpperLidClosed', 'JawDropStretched', 'TongueOut', 'lowerLipUp']
    pairs = [('body','eyes'), ('body','teeth'), ('body','tongue'), ('teeth','tongue')]
    with np.load(archive, allow_pickle=False) as data:
        for name in selected:
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)
            geometry = {}
            frame = meta['pose_names'].index(name)
            for part in ['body','eyes','teeth','tongue']:
                mesh = bpy.data.meshes.new(part)
                mesh.from_pydata(data[part][frame].tolist(), [], meta['faces'][part])
                mesh.update(); mesh.calc_loop_triangles()
                vertices = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
                triangles = np.array([t.vertices[:] for t in mesh.loop_triangles], dtype=np.int64)
                polygon = np.array([t.polygon_index for t in mesh.loop_triangles])
                tree = BVHTree.FromPolygons(vertices.tolist(), triangles.tolist(), all_triangles=True, epsilon=0.0)
                obj = bpy.data.objects.new(part, mesh)
                bpy.context.collection.objects.link(obj)
                geometry[part] = (vertices, triangles, polygon, tree, obj)
            for left,right in pairs:
                a, ta, pa, tree_a, _ = geometry[left]
                b, tb, pb, tree_b, _ = geometry[right]
                candidates = np.array(tree_a.overlap(tree_b), dtype=np.int64).reshape(-1,2)
                hits, _ = module.classify_pairs(a[ta[candidates[:,0]]], b[tb[candidates[:,1]]])
                crossing = candidates[hits]
                entry = {'pose':name, 'pair':left+'__'+right, 'crossing_pairs':len(crossing),
                         'triangle_pairs':crossing.tolist()}
                for side, coords, triangles, polygons, indices in [
                    ('left',a,ta,pa,crossing[:,0]),('right',b,tb,pb,crossing[:,1])]:
                    unique = np.unique(indices)
                    entry[side+'_triangle_indices'] = unique.tolist()
                    entry[side+'_polygon_indices'] = np.unique(polygons[unique]).tolist()
                    points = coords[triangles[unique]].reshape(-1,3)
                    entry[side+'_involved_vertex_bounds_source_units'] = [points.min(0).tolist(),points.max(0).tolist()] if len(points) else None
                results.append(entry)
            # Mouth-only diagnostic: removing the body deliberately exposes hidden geometry.
            # These views establish geometric location, never visibility in the composed avatar.
            geometry['body'][4].hide_render = True
            geometry['eyes'][4].hide_render = True
            for part, color in [('teeth',(0.85,0.82,0.65,1)),('tongue',(0.75,0.12,0.18,1))]:
                obj=geometry[part][4]
                obj.color=color
                for face in obj.data.polygons: face.use_smooth=False
            scene=bpy.context.scene
            scene.render.engine='BLENDER_WORKBENCH'
            scene.display.shading.light='STUDIO'
            scene.display.shading.color_type='OBJECT'
            scene.display.shading.show_shadows=True
            scene.display.shading.show_cavity=True
            scene.display.shading.background_type='WORLD'
            scene.world.color=(0.08,0.08,0.08)
            scene.render.resolution_x=900; scene.render.resolution_y=900
            scene.render.resolution_percentage=100
            scene.render.image_settings.file_format='PNG'
            camera_data=bpy.data.cameras.new('Inspection')
            camera=bpy.data.objects.new('Inspection',camera_data)
            bpy.context.collection.objects.link(camera)
            scene.camera=camera
            camera_data.type='ORTHO'; camera_data.ortho_scale=1.45
            target=Vector((0,6.60,1.08))
            for view,position in [('front',(0,6.6,5)),('side',(4,6.6,1.08)),('top',(0,10.6,1.08))]:
                camera.location=position
                camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
                scene.render.filepath=str(out/(name+'-'+view+'.png'))
                bpy.ops.render.render(write_still=True)
            bpy.ops.wm.save_as_mainfile(filepath=str(out/(name+'.blend')))
    (out/'localization.json').write_text(json.dumps({'source_archive_sha256':meta['mesh_archive_sha256'],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'coordinate_frame':'Native X/Y/Z; Y up; decimeters',
        'scope':'All strict crossing pairs in five selected static cases; no candidate recall guarantee',
        'render_scope':'Teeth and tongue only, body and eyes deliberately hidden; not composed-avatar visibility',
        'bounds_scope':'Bounds of involved triangle vertices, not intersection points or penetration depth',
        'rows':results,'phase_exit_approved':False},indent=2)+'\n')


if __name__=='__main__':
    args=sys.argv[sys.argv.index('--')+1:]
    if len(args)!=2: raise ValueError('Expected audit root and new output directory')
    main(Path(args[0]).resolve(),Path(args[1]).resolve())
