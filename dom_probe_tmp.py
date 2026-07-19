import torch
from signtranslator.planning.schema import PlanVocabulary, SignPlan, SemanticFrame, serialize_plan
from signtranslator.planning.planner import pad_plan_batch
from signtranslator.planning.factorized import (EvidenceEncoder, ContentHead, HeavyDecoder,
    factorized_train, joint_train, representation_probe_accuracy, DominanceReport)

V = PlanVocabulary(num_predicates=6, num_roles=2, num_referents=3, num_tam=3,
                   num_loci=4, num_lexemes=8, num_nonmanual=2, max_units=4, num_conf_buckets=3)
# Content = predicate. Each src code maps to a distinct predicate -> distinct plan.
n_codes = 6
def plan_for(code):
    return SignPlan(frame=SemanticFrame(predicate=code % V.num_predicates),
                    referents=[0], loci={0:0},
                    manual_units=[code % V.num_lexemes, (code+1)%V.num_lexemes],
                    tam=code % V.num_tam, conf_bucket=1)
def dataset(reps, seed):
    g=torch.Generator().manual_seed(seed)
    src=[]; content=[]; plans=[]
    for _ in range(reps):
        for c in range(n_codes):
            # src = the code plus some noise tokens (evidence with distractors)
            row=[c] + [int(torch.randint(n_codes, 2*n_codes, (1,), generator=g)) for _ in range(3)]
            src.append(row); content.append(c % V.num_predicates); plans.append(serialize_plan(plan_for(c), V))
    return (torch.tensor(src), torch.tensor(content), pad_plan_batch(plans, V.size+1))

src_tr, y_tr, plans_tr = dataset(8, 0)
src_ev, y_ev, plans_ev = dataset(4, 999)
src_vocab = 2*n_codes
for trial in range(3):
    torch.manual_seed(trial)
    # factorized
    enc_f=EvidenceEncoder(src_vocab); head=ContentHead(enc_f.d_model, V.num_predicates); dec_f=HeavyDecoder(V.size)
    rf=factorized_train(enc_f, head, dec_f, src_tr, y_tr, plans_tr, stage_a_steps=200, stage_b_steps=200)
    pf=representation_probe_accuracy(enc_f, src_tr, y_tr, src_ev, y_ev)
    # joint
    torch.manual_seed(trial)
    enc_j=EvidenceEncoder(src_vocab); dec_j=HeavyDecoder(V.size)
    rj=joint_train(enc_j, dec_j, src_tr, plans_tr, steps=400)
    pj=representation_probe_accuracy(enc_j, src_tr, y_tr, src_ev, y_ev)
    rep=DominanceReport(pf,pj,rf.final_plan_nll,rj.final_plan_nll)
    print(f'trial {trial}: {rep.summary()} | nll fact {rf.final_plan_nll:.3f} joint {rj.final_plan_nll:.3f}')
