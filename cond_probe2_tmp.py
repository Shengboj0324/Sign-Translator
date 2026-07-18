import itertools, math, torch, torch.nn.functional as F
from signtranslator.speech import LogMelSpectrogram, N_MELS, ctc_greedy_path, collapse
from signtranslator.speech.evaluation import characterise_condition, Condition
from signtranslator.models import SpeechRecognizer
SR=16000
PRIMARY={120.0:1,210.0:2,320.0:3}
SECONDARY={160.0:4,260.0:5,400.0:6}   # a second "language" for code-switching
def utt(f0s, noise=0.0, pitch=1.0, seed=0):
    g=torch.Generator().manual_seed(seed); parts=[]
    for f0 in f0s:
        n=int(0.20*SR); t=torch.arange(n,dtype=torch.float32)/SR
        x=sum(torch.sin(2*math.pi*(f0*pitch)*h*t)/h for h in (1,2,3))
        parts.append(x*torch.hann_window(n,periodic=False)*0.5); parts.append(torch.zeros(int(0.08*SR)))
    w=torch.cat(parts)
    return w+noise*torch.randn(w.numel(),generator=g) if noise else w
def make(cond, fe, seed=0, n_utt=6):
    allmap={**PRIMARY,**SECONDARY}
    prim=list(PRIMARY); sec=list(SECONDARY)
    g=torch.Generator().manual_seed(seed)
    feats=[];tg=[]
    for i in range(n_utt):
        if cond.vocabulary=="primary": pool=prim
        elif cond.vocabulary=="secondary": pool=sec
        else: pool=None
        seq=[]
        for j in range(cond.words):
            if pool is None:
                src = prim if (j%2==0) else sec
            else: src = pool
            seq.append(src[int(torch.randint(len(src),(1,),generator=g))])
        feats.append(fe(utt(seq,cond.noise,cond.pitch_scale,seed*100+i)).t().unsqueeze(0))
        tg.append([allmap[f] for f in seq])
    n=min(f.shape[1] for f in feats)
    return torch.cat([f[:,:n] for f in feats],0), tg
torch.manual_seed(0)
fe=LogMelSpectrogram()
# train on clean PRIMARY only, 3 words
train_c=Condition("clean")
feats,tg=make(train_c,fe,seed=0,n_utt=6)
tg_t=torch.tensor(tg)
rec=SpeechRecognizer(input_dim=N_MELS,num_tokens=6,hidden_dim=96,num_layers=2,num_heads=4,subsample=2)
L=torch.full((feats.shape[0],),3,dtype=torch.long)
opt=torch.optim.Adam(rec.parameters(),lr=3e-3)
for _ in range(320):
    loss=rec.loss(feats,tg_t,L); opt.zero_grad(); loss.backward(); opt.step()
rec.eval()
conds=[Condition(f"noise{n}",noise=n) for n in (0.005,0.010,0.015,0.020,0.030)] + \
      [Condition(f"pitch{p}",pitch_scale=p) for p in (1.25,1.4,1.6,1.8)]
for cond in conds:
    f,t=make(cond,fe,seed=7,n_utt=6)
    with torch.no_grad(): lp=rec(f)
    hyps=[collapse(ctc_greedy_path(lp[i])) for i in range(f.shape[0])]
    p=characterise_condition(cond.name,hyps,t)
    print(p.summary())

