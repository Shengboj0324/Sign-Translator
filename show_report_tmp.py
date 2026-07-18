import torch, tests.test_speech_stage5_harness as H
from signtranslator.speech.evaluation import (STANDARD_CONDITIONS, characterise_condition,
    word_error_rate, character_error_rate, ArmResult, EvaluationReport, Condition)
from signtranslator.speech import LatencyModel, measure_emission_latency, LogMelSpectrogram, N_MELS
from signtranslator.models import SpeechRecognizer
torch.manual_seed(0)
fe=LogMelSpectrogram()
feats,targets,_=H._make(Condition("clean",is_baseline=True),fe,seed=0)
rec=SpeechRecognizer(input_dim=N_MELS,num_tokens=6,hidden_dim=96,num_layers=2,num_heads=4,subsample=2)
tgt=torch.tensor(targets); L=torch.full((feats.shape[0],),tgt.shape[1],dtype=torch.long)
opt=torch.optim.Adam(rec.parameters(),lr=3e-3)
for _ in range(320):
    l=rec.loss(feats,tgt,L); opt.zero_grad(); l.backward(); opt.step()
rec.eval()
lat=LatencyModel(chunk_frames=8,right_context=4); meas=measure_emission_latency(16000,lat)
rep=EvaluationReport(streaming_config=lat.describe(),latency_median_s=meas.median_s,latency_p95_s=meas.p95_s)
for c in STANDARD_CONDITIONS:
    f,t,_=H._make(c,fe,seed=7); _,h=H._decode(rec,f)
    rep.profiles.append(characterise_condition(c.name,h,t,is_baseline=c.is_baseline))
    rep.add(ArmResult(arm="fused",condition=c.name,wer=word_error_rate(h,t),
                      cer=character_error_rate(h,t,H.SPELLING)))
print(rep.summary())
