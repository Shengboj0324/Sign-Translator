import torch
from signtranslator.planning.schema import PlanVocabulary, SignPlan, SemanticFrame, serialize_plan
from signtranslator.planning.planner import pad_plan_batch
from signtranslator.planning.factorized import (EvidenceEncoder, ContentHead, HeavyDecoder,
    factorized_train, joint_train, representation_probe_accuracy, DominanceReport)

V = PlanVocabulary(num_predicates=12, num_roles=2, num_referents=2, num_tam=2,
                   num_loci=3, num_lexemes=4, num_nonmanual=2, max_units=3, num_conf_buckets=2)
K = V.num_predicates
# Plans are CONSTANT except the predicate (the only content-bearing token) -> strong LM prior.
def plan_for(pred):
    return SignPlan(frame=SemanticFrame(predicate=pred), referents=[0], loci={0:0},
                    manual_units=[0, 1], tam=0, conf_bucket=1)
# content = (src[1]+src[2]) mod K  -> requires the encoder to AGGREGATE, with distractors
def dataset(reps, seed):
    g=torch.Generator().manual_seed(seed); src=[];content=[];plans=[]
    for _ in range(reps*K):
        a=int(torch.randint(0,K,(1,),generator=g)); b=int(torch.randint(0,K,(1,),generator=g))
        pred=(a+b)%K
        row=[a+2, b+2] + [int(torch.randint(2, K+2,(1,),generator=g)) for _ in range(4)]  # distractors
        src.append(row); content.append(pred); plans.append(serialize_plan(plan_for(pred),V))
    return torch.tensor(src), torch.tensor(content), pad_plan_batch(plans,V.size+1)
src_tr,y_tr,plans_tr=dataset(40,0); src_ev,y_ev,plans_ev=dataset(15,999)
src_vocab=K+2
import statistics
for dcap,ecap,sa,sb,jt in [(6,1,150,300,450),(6,2,150,300,450)]:
    fs=[];js=[]
    for trial in range(3):
        torch.manual_seed(trial)
        enc=EvidenceEncoder(src_vocab, d_model=32, num_layers=ecap); head=ContentHead(32,K); dec=HeavyDecoder(V.size,d_model=32,num_layers=dcap)
        factorized_train(enc,head,dec,src_tr,y_tr,plans_tr,stage_a_steps=sa,stage_b_steps=sb)
        fs.append(representation_probe_accuracy(enc,src_tr,y_tr,src_ev,y_ev))
        torch.manual_seed(trial)
        enc2=EvidenceEncoder(src_vocab,d_model=32,num_layers=ecap); dec2=HeavyDecoder(V.size,d_model=32,num_layers=dcap)
        joint_train(enc2,dec2,src_tr,plans_tr,steps=jt)
        js.append(representation_probe_accuracy(enc2,src_tr,y_tr,src_ev,y_ev))
    print(f'dec_layers={dcap} enc_layers={ecap}: factorized {statistics.mean(fs):.3f} {[round(x,2) for x in fs]} | joint {statistics.mean(js):.3f} {[round(x,2) for x in js]}')
