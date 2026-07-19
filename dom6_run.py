import torch, torch.nn.functional as F
from signtranslator.planning.factorized import EvidenceEncoder, ContentHead, representation_probe_accuracy
K=6; VOCABSZ=2*K
def data(n,seed):  # content at pos0; positions 1..3 are distractors from a DISJOINT range
    g=torch.Generator().manual_seed(seed)
    y=torch.randint(0,K,(n,),generator=g)
    distr=torch.randint(K,VOCABSZ,(n,3),generator=g)
    src=torch.cat([y.unsqueeze(1), distr],dim=1)
    return src,y
src_tr,y_tr=data(160,0); src_ev,y_ev=data(80,999)
torch.manual_seed(0)
enc=EvidenceEncoder(VOCABSZ,d_model=32,num_layers=2); head=ContentHead(32,K)
opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()),lr=3e-3)
enc.train();head.train()
for _ in range(300):
    _,p=enc(src_tr);loss=F.cross_entropy(head(p),y_tr);opt.zero_grad();loss.backward();opt.step()
enc.eval();head.eval()
with torch.no_grad(): _,p=enc(src_ev); ha=(head(p).argmax(-1)==y_ev).double().mean()
print('content-head EVAL acc:', round(float(ha),3))
pf=representation_probe_accuracy(enc,src_tr,y_tr,src_ev,y_ev,epochs=200)
enc_r=EvidenceEncoder(VOCABSZ,d_model=32,num_layers=2)
pr=representation_probe_accuracy(enc_r,src_tr,y_tr,src_ev,y_ev,epochs=200)
print('probe trained:', round(pf,3), '| probe random:', round(pr,3), '| chance', round(1/K,3))
