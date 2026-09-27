"""Pinned native-method rest/FK parity on two meshes; not application qualification."""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from signtranslator.avatar_render.makehuman import load_makehuman_rig
from signtranslator.avatar_render.makehuman_face import load_makehuman_face_units


def extract(path, expected, names, namespace):
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected: raise ValueError('Native source changed')
    tree=ast.parse(payload)
    for name in names:
        candidates=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
        if len(candidates)!=1: raise ValueError('Ambiguous method')
        exec(compile(ast.Module(body=candidates,type_ignores=[]),str(path),'exec'),namespace)


def main():
    root=Path(__file__).resolve().parents[1]
    base=root/'Sign Translator Stage Documentation/evidence/w1-w2-2026-09-26'
    out=base/'native-skeleton-parity-attempt2';out.mkdir(exist_ok=False)
    ns={'np':np,'math':math,'la':np.linalg}
    extract(Path('/tmp/mh-native-matrix.py'),'7a31251b2f3873d4c33a3e939d00fcfb01d15edcab5ef3f49c2016707c5fa8ba',['magnitude','normalize'],ns)
    ns['matrix']=SimpleNamespace(normalize=ns['normalize'])
    extract(Path('/tmp/signtranslator-makehuman-semantics/skeleton.py'),'ef3a8564e6c55f85d045b45a5307c6cf8af03a7b8a50a9211b95edc3f5137a07',['getMatrix','get_normal'],ns)
    tree=ast.parse(Path('/tmp/signtranslator-makehuman-semantics/skeleton.py').read_text())
    boneclass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Bone')
    update=next(n for n in boneclass.body if isinstance(n,ast.FunctionDef) and n.name=='update')
    exec(compile(ast.Module(body=[update],type_ignores=[]),'pinned_Bone_update','exec'),ns)
    spec=json.loads((base/'source-intake/makehuman/makehuman/data/rigs/default.mhskel').read_text())
    archive=base/'native-rest-probe/posed-meshes.npz'
    expected=json.loads((archive.parent/'catalogue.json').read_text())['mesh_archive_sha256']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:raise ValueError('Shape changed')
    with np.load(archive,allow_pickle=False) as data:changed=torch.from_numpy(data['body'][0].copy())
    report=[]
    for label,shape in [('raw_base',None),('default_factor_probe',changed)]:
        rig=load_makehuman_rig(base/'source-intake/makehuman',rest_vertices=shape)
        coords=rig.rest_vertices.numpy().astype(np.float32)
        joints={k:coords[v].mean(0) for k,v in spec['joints'].items()}
        skeleton=SimpleNamespace(scale=1.,getJointPosition=lambda name,human:joints[name])
        rest=[]
        for name in rig.bone_names:
            definition=spec['bones'][name]
            normal=ns['get_normal'](skeleton,definition['rotation_plane'],spec['planes'],object())
            rest.append(ns['getMatrix'](joints[definition['head']],joints[definition['tail']],normal))
        rest=np.stack(rest)
        inv=np.linalg.inv(rest)
        face=load_makehuman_face_units(base/'source-intake/makehuman-face',rig)
        cases=[('TargetBind',torch.eye(3,dtype=torch.float64).repeat(len(rest),1,1))]+[(n,face.rotations(n)) for n in face.names]
        rows=[]
        for name,rotation in cases:
            bones=[]
            for i,parent in enumerate(rig.parents):
                local=np.eye(4,dtype=np.float32);local[:3,:3]=rotation[i].numpy().astype(np.float32)
                bone=SimpleNamespace(parent=None if parent==-1 else bones[parent],matPose=local,
                    matRestGlobal=rest[i],matRestRelative=rest[i] if parent==-1 else inv[parent]@rest[i])
                ns['update'](bone);bones.append(bone)
            skin=np.stack([b.matPoseVerts for b in bones]).astype(np.float64)
            # Independent numpy contraction on the same declared normalized weights.
            vertices=np.einsum('vb,bij,vj->vi',rig.weights.numpy(),skin[:,:3,:3],rig.rest_vertices.numpy())+rig.weights.numpy()@skin[:,:3,3]
            result=rig.pose(rotation)
            err=float(np.max(np.abs(vertices-result.vertices.numpy())))
            if not np.isfinite(err):raise ValueError('Nonfinite native comparison')
            rows.append({'case':name,'max_coordinate_difference_source_units':err})
        error=float(np.max(np.abs(rest-rig.rest_globals.numpy())))
        report.append({'mesh':label,'bone_count':len(rest),'maximum_rest_matrix_entry_difference':error,'cases':rows})
    (out/'verification.json').write_text(json.dumps({'scope':'Extracted upstream float32 rest and Bone.update methods; same local rotations and adapter-normalized weights; independent numpy skin contraction','native_full_application':False,'native_skin_weight_mapping':False,'reference_rest_sha256':expected,'results':report,'phase_exit_approved':False},indent=2)+'\n')
    print([(r['mesh'],r['maximum_rest_matrix_entry_difference'],max(x['max_coordinate_difference_source_units'] for x in r['cases'])) for r in report])


if __name__=='__main__':main()
