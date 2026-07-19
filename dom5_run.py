import torch
from signtranslator.planning.factorized import EvidenceEncoder, ContentHead, representation_probe_accuracy
import torch.nn.functional as F
K=6
# Clean separable: src = [content] only, content directly the token.
def data(n,seed):
    g=torch.Generator().manual_seed(seed)
    y=torch.randint(0,K,(n,),generator=g); src=y.unsqueeze(1)+0  # length-1 sequence
    return src, y
src_tr,y_tr=data(120,0); src_ev,y_ev=data(60,999)
torch.manual_seed(0)
enc=EvidenceEncoder(K, d_model=32, num_layers=1); head=ContentHead(32,K)
opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()),lr=3e-3)
enc.train();head.train()
for _ in range(200):
    _,pooled=enc(src_tr); loss=F.cross_entropy(head(pooled),y_tr)
    opt.zero_grad();loss.backward();opt.step()
# content-head train acc
enc.eval();head.eval()
with torch.no_grad():
    _,p=enc(src_tr); acc=(head(p).argmax(-1)==y_tr).double().mean()
print('stage-A content head train acc:', round(float(acc),3))
pf=representation_probe_accuracy(enc,src_tr,y_tr,src_ev,y_ev,epochs=200)
print('probe on trained encoder:', round(pf,3), '| chance', round(1/K,3))
# random encoder baseline
enc_r=EvidenceEncoder(K,d_model=32,num_layers=1)
pr=representation_probe_accuracy(enc_r,src_tr,y_tr,src_ev,y_ev,epochs=200)
print('probe on RANDOM encoder:', round(pr,3))
