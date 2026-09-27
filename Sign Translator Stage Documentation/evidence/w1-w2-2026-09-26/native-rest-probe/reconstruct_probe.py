from pathlib import Path
from dataclasses import replace
import hashlib,json,urllib.request
import numpy as np
import torch
from signtranslator.avatar_render.makehuman import load_makehuman_rig
from signtranslator.avatar_render.makehuman_eyes import load_makehuman_eyes
from signtranslator.avatar_render.makehuman_mouth import load_makehuman_mouth
base=Path('Sign Translator Stage Documentation/evidence/w1-w2-2026-09-26')
out=base/'native-rest-probe';out.mkdir(exist_ok=False)
intake=base/'source-intake'
entries=json.loads(Path('/tmp/signtranslator-makehuman-tree.json').read_text())['tree']
selected=[e for e in entries if '/macrodetails/' in e['path'] and (any(e['path'].endswith(f'{eth}-{sex}-young.target') for eth in ['caucasian','asian','african'] for sex in ['male','female']) or e['path'].endswith('young-averagemuscle-averageweight.target'))]
rig=load_makehuman_rig(intake/'makehuman'); original=rig.rest_vertices.numpy(); coords=original.copy();records=[]
for e in selected:
 payload=urllib.request.urlopen('https://raw.githubusercontent.com/makehumancommunity/makehuman/a8bc2d54ff0ac92e78ff71431b1023eda42bf482/'+e['path'],timeout=30).read()
 if hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()!=e['sha']:raise ValueError('Blob mismatch')
 (out/Path(e['path']).name).write_bytes(payload)
 weight=.5 if '/universal-' in e['path'] else 1/6
 ids=[];deltas=[]
 for line in payload.decode().splitlines():
  f=line.split()
  if not f or f[0].startswith('#'):continue
  if len(f)!=4:raise ValueError('Bad target row')
  ids.append(int(f[0]));deltas.append([float(x) for x in f[1:]])
 if len(ids)!=len(set(ids)):raise ValueError('Duplicate target vertex')
 if ids:
  indices=np.array(ids);delta=np.array(deltas)
  if indices.min()<0 or indices.max()>=len(coords) or not np.isfinite(delta).all():raise ValueError('Bad delta')
  coords[indices]+=weight*delta
 records.append({'path':e['path'],'git_blob':e['sha'],'sha256':hashlib.sha256(payload).hexdigest(),'weight':weight,'changed_vertices':len(ids)})
# Rest-fitting probe only: no posing or use of the old skeleton after replacing geometry.
probe=replace(rig,rest_vertices=torch.from_numpy(coords))
attachments={'eyes':load_makehuman_eyes(intake/'makehuman-eyes',probe),**load_makehuman_mouth(intake/'makehuman-mouth',probe)}
archive=out/'posed-meshes.npz'
np.savez_compressed(archive,body=coords[None],**{k:v.rest_vertices.numpy()[None] for k,v in attachments.items()})
faces={'body':[f for f,g in zip(rig.faces,rig.face_groups,strict=True) if 'body' in g]}
faces.update({k:v.faces for k,v in attachments.items()})
(out/'catalogue.json').write_text(json.dumps({'pose_names':['DefaultFactorRestProbe'],'faces':faces,'mesh_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'observed_motion':False}))
(out/'provenance.json').write_text(json.dumps({'targets':records,'interpretation':'Diagnostic weighted rest-target reconstruction from inspected default macro factors; not full native application parity or adopted rig','old_skeleton_used_for_posing':False,'maximum_vertex_shift_source_units':float(np.linalg.norm(coords-original,axis=1).max()),'phase_exit_approved':False},indent=2)+'\n')
print('Target probe exported',len(records))
