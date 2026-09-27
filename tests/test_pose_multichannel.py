"""Multichannel payload integrity and provenance adversaries, using synthetic data."""
from dataclasses import replace
import numpy as np
import pytest
from signtranslator.pose.multichannel import CHANNELS,MotionChannel,MultichannelMotion


def fixture():
    channels={}
    for name,(kind,width,units) in CHANNELS.items():
        values=np.zeros((3,1,width),dtype=np.float64)
        if kind=='rotation_6d_columns':values[:]=[1,0,0,0,1,0]
        elif kind=='unit_vector':values[:]=[1,0,0]
        channels[name]=MotionChannel(values,np.ones((3,1),dtype=bool),np.ones((3,1),dtype=bool),
            np.ones((3,1),dtype=np.float32),(name,), 'fixture-right-handed',units,
            'synthetic-fixture','a'*64,'fixture-convention-v1')
    return MultichannelMotion(np.array([0.,.1,.3]),'synthetic-clock','fixture',channels)


def test_exact_roundtrip_preserves_every_channel_and_mask(tmp_path):
    state=fixture();path=tmp_path/'motion.npz';digest=state.save(path)
    loaded=MultichannelMotion.load(path,expected_sha256=digest)
    assert np.array_equal(loaded.timestamps,state.timestamps)
    for name in CHANNELS:
        a,b=state.channels[name],loaded.channels[name]
        for key in ('values','valid','observed','confidence'):
            assert np.array_equal(getattr(a,key),getattr(b,key))
            assert getattr(a,key).dtype==getattr(b,key).dtype
        assert a.labels==b.labels and a.source_sha256==b.source_sha256
    with pytest.raises(FileExistsError):state.save(path)
    with pytest.raises(ValueError,match='hash mismatch'):
        MultichannelMotion.load(path,expected_sha256='b'*64)


def test_absent_channel_is_explicit_and_cannot_gain_confidence():
    state=fixture();channel=state.channels['left_gaze']
    absent=replace(channel,values=np.zeros_like(channel.values),valid=np.zeros_like(channel.valid),
                   observed=np.zeros_like(channel.observed),confidence=np.zeros_like(channel.confidence))
    replace(state,channels={**state.channels,'left_gaze':absent})
    with pytest.raises(ValueError,match='confidence'):
        replace(state,channels={**state.channels,'left_gaze':replace(absent,confidence=np.ones((3,1)))})
    with pytest.raises(ValueError,match='explicitly present'):
        replace(state,channels={k:v for k,v in state.channels.items() if k!='left_gaze'})


def test_inferred_state_requires_named_method_and_stays_unobserved(tmp_path):
    state=fixture();channel=replace(state.channels['body'],observed=np.zeros((3,1),dtype=bool))
    with pytest.raises(ValueError,match='inference method'):
        replace(state,channels={**state.channels,'body':channel})
    state=replace(state,channels={**state.channels,'body':replace(channel,inference_method='fixture-fit-v1')})
    path=tmp_path/'inferred.npz';digest=state.save(path)
    recovered=MultichannelMotion.load(path,expected_sha256=digest)
    assert not recovered.channels['body'].observed.any()
    assert recovered.channels['body'].inference_method=='fixture-fit-v1'


@pytest.mark.parametrize('name,updates',[
    ('left_gaze',{'values':np.full((3,1,3),2.)}),
    ('body',{'values':np.zeros((3,1,6))}),
    ('blink',{'values':np.full((3,1,1),1.1)}),
    ('contact',{'values':np.full((3,1,1),-.1)}),
    ('root_translation',{'units':'cm'}),
    ('face',{'source_sha256':'unknown'}),
    ('face',{'values':np.full((3,1,1),float('nan'))}),
])
def test_geometry_units_and_provenance_domains_rejected(name,updates):
    state=fixture()
    with pytest.raises(ValueError):
        replace(state,channels={**state.channels,name:replace(state.channels[name],**updates)})


def test_mutated_payload_cannot_bypass_save_validation(tmp_path):
    state=fixture();state.channels['left_gaze'].values[:]=0
    with pytest.raises(ValueError,match='unit norm'):state.save(tmp_path/'bad.npz')
    assert not (tmp_path/'bad.npz').exists()


@pytest.mark.parametrize('clock',[np.array([0.,0.,1.]),np.array([0.,np.nan,1.]),np.array([0.,1.,2.],dtype=np.float32)])
def test_clock_contract(clock):
    with pytest.raises(ValueError):replace(fixture(),timestamps=clock)


def test_scoped_ingestion_requires_real_bytes_and_never_approves_phase(tmp_path):
    from test_de_phase2_policy import fixture as policy_fixture
    from signtranslator.data_engineering.state_ingestion import export_phase2_state
    from signtranslator.data_engineering.phase2_policy import Phase2Scope
    import hashlib
    state=fixture();source,auth=policy_fixture(tmp_path)
    media=tmp_path/'synthetic-source';media.write_bytes(b'fictional motion acquisition fixture')
    digest=hashlib.sha256(media.read_bytes()).hexdigest()
    state=replace(state,channels={name:replace(ch,source_id=source.source_id,source_sha256=digest)
                                  for name,ch in state.channels.items()})
    options=dict(sources=(source,),scope=Phase2Scope.RESEARCH,
                 authorizations={'fixture':auth},source_files={'fixture':media})
    with pytest.raises(PermissionError):
        export_phase2_state(state,tmp_path/'unauthorized.npz',**{**options,'authorizations':{}})
    assert not (tmp_path/'unauthorized.npz').exists()
    media.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='hash'):
        export_phase2_state(state,tmp_path/'tampered.npz',**options)
    assert not (tmp_path/'tampered.npz').exists()
    media.write_bytes(b'fictional motion acquisition fixture')
    result=export_phase2_state(state,tmp_path/'eligible.npz',**options)
    assert result['eligibility']['portfolio_eligible']
    assert not result['phase_exit_approved']
    recovered=MultichannelMotion.load(tmp_path/'eligible.npz',expected_sha256=result['motion_sha256'])
    assert recovered.sample_id==state.sample_id


@pytest.mark.parametrize('mutation',['duplicate_metadata','extra_array','wrong_dtype'])
def test_archive_rejects_structural_tampering_even_with_new_hash(tmp_path,mutation):
    import hashlib,json
    state=fixture();original=tmp_path/'original.npz';state.save(original)
    with np.load(original,allow_pickle=False) as archive:
        arrays={key:archive[key].copy() for key in archive.files}
    if mutation=='duplicate_metadata':
        raw=str(arrays['metadata'])
        arrays['metadata']=np.array('{"schema_version":1,'+raw[1:])
    elif mutation=='extra_array':arrays['unexpected']=np.zeros(1)
    else:arrays['body.valid']=arrays['body.valid'].astype(np.int64)
    modified=tmp_path/'modified.npz';np.savez_compressed(modified,**arrays)
    with pytest.raises(ValueError):
        MultichannelMotion.load(modified,expected_sha256=hashlib.sha256(modified.read_bytes()).hexdigest())


def test_extreme_finite_domains_fail_without_runtime_warning():
    state=fixture()
    with pytest.raises(ValueError,match='finite intervals'):
        replace(state,timestamps=np.array([-1e308,1e308,1.5e308]))
    gaze=replace(state.channels['left_gaze'],values=np.full((3,1,3),1e308))
    with pytest.raises(ValueError,match='unit norm'):
        replace(state,channels={**state.channels,'left_gaze':gaze})


def test_observation_origin_is_distinct_from_validity_and_inference(tmp_path):
    state=fixture();original=state.channels['face']
    # Frame 0 was measured but rejected, frame 1 is valid inference, frame 2 observed/valid.
    channel=replace(original,valid=np.array([[False],[True],[True]]),
                    observed=np.array([[True],[False],[True]]),
                    confidence=np.array([[0.],[.7],[.8]]),inference_method='fixture-fit-v1')
    state=replace(state,channels={**state.channels,'face':channel})
    assert np.array_equal(channel.supervision_weights(),[[0.],[0.],[.8]])
    assert np.array_equal(channel.supervision_weights(allow_inferred=True),[[0.],[.7],[.8]])
    path=tmp_path/'origins.npz';digest=state.save(path)
    restored=MultichannelMotion.load(path,expected_sha256=digest).channels['face']
    assert restored.observed[0,0] and not restored.valid[0,0]
