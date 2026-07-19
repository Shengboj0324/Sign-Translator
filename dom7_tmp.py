import torch, torch.nn.functional as F
from signtranslator.planning.factorized import EvidenceEncoder, ContentHead, representation_probe_accuracy
M=4  # token alphabet
# content = 2*(src0==src1) + (src2==src3)  -> 4 classes, NONLINEAR (equality)
def data(n,seed):
    g=torch.Generator().manual_seed(seed)
    s=torch.randint(0,M,(n,4),generator=g)
    y=2*(s[:,0]==s[:,1]).long()+(s[:,2]==s[:,3]).long()
    return s,y
src_tr,y_tr=data(400,0); src_ev,y_ev=data(200,999)
torch.manual_seed(0)
enc=EvidenceEncoder(M,d_model=32,num_layers=2); head=ContentHead(32,4)
opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()),lr=3e-3)
enc.train();head.train()
for _ in range(400):
    _,p=enc(src_tr);loss=F.cross_entropy(head(p),y_tr);opt.zero_grad();loss.backward();opt.step()
enc.eval();head.eval()
with torch.no_grad(): _,p=enc(src_ev); ha=(head(p).argmax(-1)==y_ev).double().mean()
pf=representation_probe_accuracy(enc,src_tr,y_tr,src_ev,y_ev,epochs=250)
enc_r=EvidenceEncoder(M,d_model=32,num_layers=2)
pr=representation_probe_accuracy(enc_r,src_tr,y_tr,src_ev,y_ev,epochs=250)
print('content-head eval', round(float(ha),3),'| probe trained', round(pf,3),'| probe random', round(pr,3),'| chance 0.25')
