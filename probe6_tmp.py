import itertools, math, torch, torch.nn.functional as F
from signtranslator.speech import LogMelSpectrogram, N_MELS, ctc_forced_alignment, expected_calibration_error
from signtranslator.speech.objective import (SpeechTrainingObjective, ObjectiveWeights, BoundaryHead,
                                             boundary_targets_from_alignment)
from signtranslator.models import SpeechRecognizer
SR=16000; VOCAB={120.0:1,210.0:2,320.0:3}
def utt(f0s, noise=0.0, seed=0):
    g=torch.Generator().manual_seed(seed); parts=[]
    for f0 in f0s:
        n=int(0.20*SR); t=torch.arange(n,dtype=torch.float32)/SR
        x=sum(torch.sin(2*math.pi*f0*h*t)/h for h in (1,2,3))
        parts.append(x*torch.hann_window(n,periodic=False)*0.5); parts.append(torch.zeros(int(0.08*SR)))
    w=torch.cat(parts)
    return w + noise*torch.randn(w.numel(),generator=g) if noise>0 else w
def batch(fe, noise=0.0, seed=0):
    seqs=[list(p) for p in itertools.permutations([120.0,210.0,320.0])]
    f=[];t=[]
    for i,s in enumerate(seqs):
        f.append(fe(utt(s,noise,seed+i)).t().unsqueeze(0)); t.append([VOCAB[x] for x in s])
    n=min(a.shape[1] for a in f)
    return torch.cat([a[:,:n] for a in f],0), torch.tensor(t)
def train(bw, brw, seed=0, iters=280):
    torch.manual_seed(seed)
    fe=LogMelSpectrogram(); feats,tg=batch(fe)
    rec=SpeechRecognizer(input_dim=N_MELS,num_tokens=3,hidden_dim=96,num_layers=2,num_heads=4,subsample=2)
    head=BoundaryHead(96)
    obj=SpeechTrainingObjective(rec,ObjectiveWeights(contrastive=0.0,boundary=bw,brier=brw),boundary_head=head)
    L=torch.full((feats.shape[0],),3,dtype=torch.long)
    opt=torch.optim.Adam(obj.parameters(),lr=3e-3)
    for _ in range(iters):
        out=obj(feats,tg,L); opt.zero_grad(); out.total.backward(); opt.step()
    return rec,head,obj,fe
def evaluate(rec,head,fe,noise=0.0,seed=0):
    feats,tg=batch(fe,noise=noise,seed=seed)
    rec.eval();head.eval()
    with torch.no_grad():
        hid=rec.encode(feats); lp=F.log_softmax(rec.classifier(hid),dim=-1)
        bl=head(hid)
    tp=fp=fn=0; confs=[];corr=[]
    for i in range(feats.shape[0]):
        al=ctc_forced_alignment(lp[i],tg[i].tolist())
        true=boundary_targets_from_alignment(al,lp.shape[1]).bool()
        pred=torch.sigmoid(bl[i])>=0.5
        tp+=int((pred&true).sum()); fp+=int((pred&~true).sum()); fn+=int((~pred&true).sum())
        p=lp[i].exp(); c,pr=p.max(-1); lab=torch.tensor(al.state_tokens())
        confs+= c.tolist(); corr += (pr==lab).tolist()
    f1 = 2*tp/max(2*tp+fp+fn,1)
    return f1, expected_calibration_error(confs,corr,n_bins=12)
for bw,name in [(0.0,'no boundary term'),(0.5,'with boundary term')]:
    res=[]
    for seed in (0,1,2):
        rec,head,obj,fe=train(bw,0.0,seed=seed,iters=280)
        res.append(evaluate(rec,head,fe)[0])
    print(f'{name:20s} train-fit boundary F1 per seed: {[round(x,3) for x in res]}')

