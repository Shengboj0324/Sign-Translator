"""Independent exact-rational plane/barycentric certificates for selected mesh pairs."""
from fractions import Fraction as F
import json
from pathlib import Path


def sub(a, b):
    return [x-y for x,y in zip(a,b,strict=True)]


def dot(a, b):
    return sum(x*y for x,y in zip(a,b,strict=True))


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def certificate(first, second):
    for direction, (source, target) in enumerate(((first,second),(second,first))):
        e, f = sub(target[1],target[0]), sub(target[2],target[0])
        normal = cross(e,f)
        gram = dot(e,e)*dot(f,f)-dot(e,f)**2
        if gram == 0:
            continue
        for edge in range(3):
            start, end = source[edge], source[(edge+1)%3]
            d0, d1 = dot(sub(start,target[0]),normal), dot(sub(end,target[0]),normal)
            if d0*d1 >= 0:
                continue
            t = d0/(d0-d1)
            point = [x+t*(y-x) for x,y in zip(start,end,strict=True)]
            relative = sub(point,target[0])
            u = (dot(relative,e)*dot(f,f)-dot(relative,f)*dot(e,f))/gram
            v = (dot(relative,f)*dot(e,e)-dot(relative,e)*dot(e,f))/gram
            if 0 < t < 1 and u > 0 and v > 0 and u+v < 1:
                return {'direction':direction,'edge':edge,'segment_fraction':str(t),
                        'barycentric_weights':[str(1-u-v),str(u),str(v)],
                        'intersection_source_units':[str(x) for x in point]}
    raise ValueError('No strict rational crossing witness')


if __name__ == '__main__':
    root=Path(__file__).resolve().parent
    rows=json.loads((root/'witness-triangles.json').read_text())
    results=[]
    for row in rows:
        first=[[F(v) for v in point] for point in row['a']]
        second=[[F(v) for v in point] for point in row['b']]
        results.append({k:row[k] for k in ['pose','pair','triangle_indices']} | {'certificate':certificate(first,second)})
    (root/'exact-witness-verification.json').write_text(json.dumps({
        'method':'Exact rational segment-plane intersection and Gram barycentric coordinates; no tolerance or BVH used',
        'scope':'Selected positive crossings only; not exhaustive coverage, clearance, anatomical or linguistic acceptance',
        'verified_crossings':len(results),'certificates':results},indent=2)+'\n')
    print(f'{len(results)} selected crossings independently certified with exact rational arithmetic')
