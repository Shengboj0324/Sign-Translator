import torch, torch.nn.functional as F, statistics
from signtranslator.planning.factorized import EvidenceEncoder, ContentHead, representation_probe_accuracy
M=5
def data(n,seed):  # binary: are the first two tokens equal?
    g=torch.Generator().manual_seed(seed)
    s=torch.randint(0,M,(n,4),generator=g)
    y=(s[:,0]==s[:,1]).long()
    return s,y
src_tr,y_tr=data(400,0); src_ev,y_ev=data(200,999)
tg=[];rg=[]
for trial in range(3):
    torch.manual_seed(trial)
    enc=EvidenceEncoder(M,d_model=48,num_layers=2,nhead=4); head=ContentHead(48,2)
    opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()),lr=3e-3)
    enc.train();head.train()
    for _ in range(500):
        _,p=enc(src_tr);loss=F.cross_entropy(head(p),y_tr);opt.zero_grad();loss.backward();opt.step()
    pf=representation_probe_accuracy(enc,src_tr,y_tr,src_ev,y_ev,epochs=250)
    torch.manual_seed(trial+100)
    enc_r=EvidenceEncoder(M,d_model=48,num_layers=2,nhead=4)
    pr=representation_probe_accuracy(enc_r,src_tr,y_tr,src_ev,y_ev,epochs=250)
    tg.append(pf); rg.append(pr)
print('trained', round(statistics.mean(tg),3), [round(x,2) for x in tg])
print('random ', round(statistics.mean(rg),3), [round(x,2) for x in rg])
