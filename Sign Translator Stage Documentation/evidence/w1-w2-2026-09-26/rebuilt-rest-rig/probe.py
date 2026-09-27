from pathlib import Path
import hashlib,json
import numpy as np
import torch
from signtranslator.avatar_render.makehuman import load_makehuman_rig
from signtranslator.avatar_render.makehuman_eyes import load_makehuman_eyes
from signtranslator.avatar_render.makehuman_mouth import load_makehuman_mouth
from signtranslator.avatar_render.makehuman_face import load_makehuman_face_units
from signtranslator.avatar_render.rigged import apply_lbs
out=Path(__file__).resolve().parent; base=out.parent; intake=base/'source-intake'
source=base/'native-rest-probe/posed-meshes.npz'
expected=json.loads((source.parent/'catalogue.json').read_text())['mesh_archive_sha256']
if hashlib.sha256(source.read_bytes()).hexdigest()!=expected: raise ValueError('Rest source changed')
with np.load(source,allow_pickle=False) as d: coords=torch.from_numpy(d['body'][0].copy())
rig=load_makehuman_rig(intake/'makehuman',rest_vertices=coords)
raw=load_makehuman_rig(intake/'makehuman')
attachments={'eyes':load_makehuman_eyes(intake/'makehuman-eyes',rig),**load_makehuman_mouth(intake/'makehuman-mouth',rig)}
face=load_makehuman_face_units(intake/'makehuman-face',rig)
poses=[('TargetBind',torch.eye(3,dtype=torch.float64).repeat(len(rig.bone_names),1,1))]+[(n,face.rotations(n)) for n in face.names]
arrays={k:[] for k in ['body',*attachments]}
for name,rotation in poses:
 posed=rig.pose(rotation);arrays['body'].append(posed.vertices.numpy())
 for k,a in attachments.items():arrays[k].append(apply_lbs(a.rest_vertices,a.weights,posed.skin_transforms).numpy())
for k,v in arrays.items():
 if not np.isfinite(v).all():raise ValueError('Nonfinite posed geometry')
error=float(np.max(np.abs(arrays['body'][0]-coords.numpy())))
if error>1e-12:raise ValueError('Bind identity failed')
archive=out/'posed-meshes.npz';np.savez_compressed(archive,**{k:np.stack(v) for k,v in arrays.items()})
faces={'body':[f for f,g in zip(rig.faces,rig.face_groups,strict=True) if 'body' in g]};faces.update({k:a.faces for k,a in attachments.items()})
(out/'catalogue.json').write_text(json.dumps({'pose_names':[n for n,_ in poses],'faces':faces,'mesh_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'observed_motion':False}))
(out/'rig-measurements.json').write_text(json.dumps({'input_rest_archive_sha256':expected,'bones':len(rig.bone_names),'static_cases':len(poses),'bind_max_coordinate_error':error,'maximum_bone_head_shift_source_units':float(torch.linalg.vector_norm(rig.rest_globals[:,:3,3]-raw.rest_globals[:,:3,3],dim=1).max()),'weights_unchanged':torch.equal(rig.weights,raw.weights),'native_application_parity':False,'phase_exit_approved':False},indent=2)+'\n')
print('Rebuilt 163 bone frames and exported 61 static cases')
