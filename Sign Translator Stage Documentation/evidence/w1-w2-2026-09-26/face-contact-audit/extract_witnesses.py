import bpy
import json
import numpy as np
from pathlib import Path
root=Path('/Users/jiangshengbo/Desktop/Sign-Translator/Sign Translator Stage Documentation/evidence/w1-w2-2026-09-26/face-contact-audit')
meta=json.loads((root/'catalogue.json').read_text())
report=json.loads((root/'contact-audit.json').read_text())
selected={'TargetBind','LeftUpperLidClosed','JawDropStretched','TongueOut','lowerLipUp'}
witnesses=[]
with np.load(root/'posed-meshes.npz',allow_pickle=False) as data:
 for row in report['rows']:
  if row['pose'] not in selected: continue
  geometry={}
  for part in ['body','eyes','teeth','tongue']:
   mesh=bpy.data.meshes.new(part)
   mesh.from_pydata(data[part][meta['pose_names'].index(row['pose'])].tolist(),[],meta['faces'][part])
   mesh.update(); mesh.calc_loop_triangles()
   vertices=np.array([v.co[:] for v in mesh.vertices])
   triangles=np.array([t.vertices[:] for t in mesh.loop_triangles])
   geometry[part]=(vertices,triangles)
   bpy.data.meshes.remove(mesh)
  for pair,entry in row['pairs'].items():
   left,right=pair.split('__')
   a,ta=geometry[left]; b,tb=geometry[right]
   for i,j in entry['first_crossing_triangle_pairs'][:3]:
    witnesses.append({'pose':row['pose'],'pair':pair,'triangle_indices':[i,j],'a':a[ta[i]].tolist(),'b':b[tb[j]].tolist()})
(root/'witness-triangles.json').write_text(json.dumps(witnesses,indent=2)+'\n')
print('Exported',len(witnesses),'selected reported crossings')
