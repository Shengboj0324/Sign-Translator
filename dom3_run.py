import torch, statistics
from signtranslator.planning.schema import PlanVocabulary, SignPlan, SemanticFrame, serialize_plan
from signtranslator.planning.planner import pad_plan_batch
from signtranslator.planning.factorized import (EvidenceEncoder, ContentHead, HeavyDecoder,
    factorized_train, joint_train, representation_probe_accuracy)
V = PlanVocabulary(num_predicates=10, num_roles=2, num_referents=2, num_tam=2,
                   num_loci=3, num_lexemes=4, num_nonmanual=2, max_units=3, num_conf_buckets=2)
K=V.num_predicates
def plan_for(pred):  # constant except predicate -> strong LM prior
    return SignPlan(frame=SemanticFrame(predicate=pred), referents=[0], loci={0:0},
                    manual_units=[0,1], tam=0, conf_bucket=1)
def dataset(n, seed):
    g=torch.Generator().manual_seed(seed); src=[];y=[];pl=[]
    for _ in range(n):
        a=int(torch.randint(0,K,(1,),generator=g)); b=int(torch.randint(0,K,(1,),generator=g))
        pred=(a+b)%K
        src.append([a+2,b+2]+[int(torch.randint(2,K+2,(1,),generator=g)) for _ in range(4)])
        y.append(pred); pl.append(serialize_plan(plan_for(pred),V))
    return torch.tensor(src),torch.tensor(y),pad_plan_batch(pl,V.size+1)
src_tr,y_tr,plans_tr=dataset(96,0); src_ev,y_ev,plans_ev=dataset(48,999)
fs=[];js=[]
for trial in range(1):
    torch.manual_seed(trial)
    enc=EvidenceEncoder(K+2,d_model=32,num_layers=1); head=ContentHead(32,K); dec=HeavyDecoder(V.size,d_model=32,num_layers=3)
    factorized_train(enc,head,dec,src_tr,y_tr,plans_tr,stage_a_steps=100,stage_b_steps=100)
    fs.append(representation_probe_accuracy(enc,src_tr,y_tr,src_ev,y_ev,epochs=120))
    torch.manual_seed(trial)
    enc2=EvidenceEncoder(K+2,d_model=32,num_layers=1); dec2=HeavyDecoder(V.size,d_model=32,num_layers=3)
    joint_train(enc2,dec2,src_tr,plans_tr,steps=200)
    js.append(representation_probe_accuracy(enc2,src_tr,y_tr,src_ev,y_ev,epochs=120))
print('chance =', round(1/K,3))
print('factorized', round(statistics.mean(fs),3), [round(x,2) for x in fs])
print('joint     ', round(statistics.mean(js),3), [round(x,2) for x in js])
