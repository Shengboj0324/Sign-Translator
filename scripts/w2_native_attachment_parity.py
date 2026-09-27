"""Run two hash-pinned upstream geometry methods on the same unmorphed rest input.

This is method-level fitting parity, not full MakeHuman application parity.
The upstream file is supplied externally, never imported as a complete module.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace, MethodType

import numpy as np
from signtranslator.avatar_render.makehuman import load_makehuman_rig
from signtranslator.avatar_render.makehuman_eyes import load_makehuman_eyes
from signtranslator.avatar_render.makehuman_mouth import load_makehuman_mouth

PIN='a83091c0677eb714b04f32e0be5e870241f135ecaf27ab935629319a6204a5ff'


def run(source, intake, out):
    payload=source.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=PIN:
        raise ValueError('Upstream geometry source identity mismatch')
    parsed=ast.parse(payload)
    methods={}
    namespace={'np':np}
    for cls,method in [('Proxy','getCoords'),('TMatrix','getMatrix')]:
        container=next(n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name==cls)
        node=next(n for n in container.body if isinstance(n,ast.FunctionDef) and n.name==method)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
        methods[method]=namespace[method]
    rig=load_makehuman_rig(intake/'makehuman')
    attachments={'eyes':load_makehuman_eyes(intake/'makehuman-eyes',rig),**load_makehuman_mouth(intake/'makehuman-mouth',rig)}
    proxies={
        'eyes':next((intake/'makehuman-eyes').rglob('*.mhclo')),
        'teeth':intake/'makehuman-mouth/teeth/teeth_base/teeth_base.mhclo',
        'tongue':intake/'makehuman-mouth/tongue/tongue01/tongue01.mhclo'}
    records=[]
    for part, attachment in attachments.items():
        scale=[None]*3; refs=[]; weights=[]; offsets=[]; reading=False
        for line in proxies[part].read_text().splitlines():
            f=line.split()
            if not f or f[0].startswith('#'): continue
            if f[0] in ('x_scale','y_scale','z_scale'):
                scale['xyz'.index(f[0][0])]=(int(f[1]),int(f[2]),float(f[3]))
            elif f[0]=='verts': reading=True
            elif reading:
                if len(f)==1:
                    refs.append([int(f[0])]*3); weights.append([1.,0.,0.]); offsets.append([0.,0.,0.])
                elif len(f)==9:
                    refs.append([int(x) for x in f[:3]])
                    weights.append([float(x) for x in f[3:6]])
                    offsets.append([float(x) for x in f[6:]])
                else: raise ValueError('Unsupported pinned proxy mapping')
        for dtype in [np.float64,np.float32]:
            coordinates=rig.rest_vertices.numpy().astype(dtype)
            matrix=SimpleNamespace(scaleData=scale)
            matrix.getMatrix=MethodType(methods['getMatrix'],matrix)
            instance=SimpleNamespace(human=SimpleNamespace(getRestposeCoordinates=lambda:coordinates),
                tmatrix=matrix,ref_vIdxs=np.array(refs,dtype=np.uint32),
                weights=np.array(weights,dtype=dtype),offsets=np.array(offsets,dtype=dtype))
            native=methods['getCoords'](instance)
            delta=native-attachment.rest_vertices.numpy()
            error=float(np.max(np.abs(delta)))
            # Float64 is an arithmetic oracle; float32 measures the source input/weight precision path.
            limit=1e-12 if dtype==np.float64 else 5e-6
            if not np.isfinite(native).all() or error>limit:
                raise ValueError(f'{part} {dtype} fitting discrepancy {error} exceeds {limit}')
            records.append({'attachment':part,'input_precision':np.dtype(dtype).name,
                'vertices':len(native),'max_coordinate_error_source_units':error,
                'max_vertex_distance_meters':float(np.linalg.norm(delta,axis=1).max())*.1,
                'diagnostic_absolute_limit_source_units':limit})
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps({'upstream_sha256':PIN,'executed_methods':['Proxy.getCoords','TMatrix.getMatrix'],
        'whole_upstream_module_imported':False,'whole_native_application_executed':False,
        'input':'Same pinned unmorphed base mesh; no character modifiers or native scene state',
        'float32_scope':'Coordinates, proxy coefficients and offsets cast to native-style float32; output matrix arithmetic follows inspected upstream methods',
        'results':records,'inference':'Large rest-fit intersections are not explained by disagreement with these upstream geometry methods on the same input',
        'limitations':['No native skin-weight or pose parity','No native default/morphed character qualification','No anatomical or linguistic acceptance'],
        'phase_exit_approved':False},indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--intake',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); run(a.source,a.intake,a.out)
