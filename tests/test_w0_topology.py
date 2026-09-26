"""Topology and joint semantics must survive construction and checkpoints."""
import json
import pytest
import torch
from signtranslator.data.corpus import CorpusSpec, generate_corpus
from signtranslator.run import build_model, run_pipeline
from signtranslator.skeleton.graph import SkeletonGraph
from signtranslator.training import Trainer
from test_trainer import _tiny_setup
from signtranslator import TrainerConfig


def test_137_joint_construction_uses_declared_graph():
    spec=CorpusSpec.build(4,2,137,3,8)
    with pytest.raises(ValueError,match='explicit skeleton'):
        build_model(spec,diff_timesteps=10)
    graph=SkeletonGraph(137,[(i,i+1) for i in range(136)],0,[f'sensor_{i}' for i in range(137)])
    model=build_model(spec,diff_timesteps=10,graph=graph)
    assert model.graph.to_dict()==graph.to_dict()
    assert model.embed_motion(torch.randn(2,3,8,137)).shape[0]==2


@pytest.mark.parametrize('change',[{'edges':[[0,1],[1,0]]},{'edges':[[0,True]]},{'joint_names':['a','a']},{'center':True},{'num_nodes':2.5},{'extra':1}])
def test_malformed_topology_rejected(change):
    payload=SkeletonGraph(2,[(0,1)],0,['a','b']).to_dict();payload.update(change)
    with pytest.raises((ValueError,TypeError)): SkeletonGraph.from_dict(payload)


def test_joint_order_mismatch_rejected_before_checkpoint_weights_loaded(tmp_path):
    model,train,val=_tiny_setup(tmp_path)
    trainer=Trainer(model,TrainerConfig(epochs=1),train,val)
    path=tmp_path/'graph.pt';trainer.save(path)
    other,train2,val2=_tiny_setup(tmp_path)
    other.graph=SkeletonGraph(joint_names=list(reversed(other.graph.joint_names)))
    snapshot={name:t.clone() for name,t in other.state_dict().items()}
    receiver=Trainer(other,TrainerConfig(epochs=1),train2,val2)
    with pytest.raises(ValueError,match='model contract'):
        receiver.load(path,mode='weights')
    assert all(torch.equal(t,snapshot[name]) for name,t in other.state_dict().items())


def test_alternate_manifest_reloads_through_runtime(tmp_path):
    corpus=tmp_path/'corpus';spec=CorpusSpec.build(4,2,5,3,8)
    generate_corpus(str(corpus),spec=spec,counts={'train':32,'val':16})
    graph=SkeletonGraph(5,[(0,1),(1,2),(1,3),(3,4)],0,['a','b','c','d','e'])
    path=corpus/'manifest.json';manifest=json.loads(path.read_text());manifest['skeleton']=graph.to_dict()
    path.write_text(json.dumps(manifest))
    result=run_pipeline(str(corpus),epochs=1,batch_size=8,diff_timesteps=10,
                        do_train=False,do_analyze=False,require_ready=False,verbose=False)
    assert result['model'].graph.to_dict()==graph.to_dict()


def test_exported_topology_roundtrip_and_wrong_order_rejection(tmp_path):
    from test_de_integration import _extracted_records, LANDMARK_PARTS
    from signtranslator.data_engineering.exporter import export_corpus
    from signtranslator.data.corpus import validate_corpus, load_manifest
    names = [f'joint-{index}' for index in range(5)]
    graph = SkeletonGraph(5, [(i,i+1) for i in range(4)], 0, names)
    result = export_corpus(
        _extracted_records(), tmp_path / 'export', joint_names=names,
        landmark_parts=LANDMARK_PARTS, split_ratios=(.5,.25,.25), skeleton=graph)
    validate_corpus(result.corpus_dir)
    assert load_manifest(result.corpus_dir)['skeleton'] == graph.to_dict()
    with pytest.raises(ValueError, match='exact joint order'):
        export_corpus(_extracted_records(), tmp_path / 'wrong',
                      joint_names=list(reversed(names)), landmark_parts=LANDMARK_PARTS,
                      skeleton=graph)
