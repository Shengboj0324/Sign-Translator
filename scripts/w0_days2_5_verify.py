"""Verify W0 Days 2–5 repairs and isolated-wheel behavior; never authorize ASL use."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from w0_day1_baseline import file_record, run_command, snapshot


WHEEL_SMOKE = r'''
import json, sys, torch
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import signtranslator
assert Path(signtranslator.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
from signtranslator.run import run_pipeline
from signtranslator.training import checkpoint_paths
from signtranslator.reproducibility import package_implementation_identity
from signtranslator.data.corpus import CorpusSpec, generate_corpus
from signtranslator.skeleton.graph import SkeletonGraph
root = Path(sys.argv[2])
corpus = root / 'corpus'
spec = CorpusSpec.build(4, 2, 5, 3, 8)
generate_corpus(str(corpus), spec=spec, counts={'train':32, 'val':16})
graph = SkeletonGraph(5, [(0,1),(1,2),(1,3),(3,4)], 0, ['a','b','c','d','e'])
manifest_path = corpus / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest['skeleton'] = graph.to_dict()
manifest_path.write_text(json.dumps(manifest))
prefix = root / 'model.pt'
# Readiness override is for this deliberately tiny SYNTHETIC numerical fixture.
# It does not demonstrate real-corpus readiness.
result = run_pipeline(str(corpus), batch_size=8, diff_timesteps=10,
    require_ready=False, do_analyze=False, ckpt_path=str(prefix), verbose=False)
paths = checkpoint_paths(prefix)
assert paths['best'] != paths['last'] and all(path.is_file() for path in paths.values())
loaded = run_pipeline(str(corpus), batch_size=8, diff_timesteps=10,
    require_ready=False, do_train=False, do_analyze=False,
    ckpt_path=str(paths['last']), verbose=False)
result['model'].eval(); loaded['model'].eval()
tokens = torch.tensor([[3,4]])
torch.manual_seed(91)
a = result['model'].generate_from_gloss(tokens, ddim_steps=3)
torch.manual_seed(91)
b = loaded['model'].generate_from_gloss(tokens, ddim_steps=3)
assert torch.equal(a,b) and torch.isfinite(b).all()
assert loaded['model'].graph.to_dict() == graph.to_dict()
identity = package_implementation_identity()
assert identity['identity_kind'] == 'content-only' and identity['git_revision'] is None
print(json.dumps({'wheel_import_verified':True,'synthetic_custom_topology':graph.to_dict(),
 'optimizer_steps':result['trainer'].global_step,'reload_equal':True,'output_finite':True,
 'output_shape':list(b.shape),'best_last_distinct':True,'identity_kind':identity['identity_kind'],
 'empirical_readiness_approved':False}))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(repo))
    from signtranslator.reproducibility import runtime_environment
    from signtranslator.eval_framework.statistics import sign_test_pvalue, paired_permutation_pvalue
    from signtranslator.planning.phase3c import assess_phase3c_readiness
    from signtranslator.data_engineering.source_portfolio import CURRENT_PRE_PHASE_2_DECISION

    before = snapshot(repo)
    before['scripts/w0_days2_5_verify.py'] = file_record(Path(__file__))
    report = {'schema_version':1,'started_at_utc':datetime.now(timezone.utc).isoformat(),
              'runtime':runtime_environment(),'source_test_config_inventory':before,
              'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
              'worktree_status':subprocess.check_output(['git','status','--short'],cwd=repo,text=True)}
    commands = {}
    for name, argv in (
        ('dependencies',[sys.executable,'-m','pip','check']),
        ('compile',[sys.executable,'-m','compileall','-q','signtranslator','tests','scripts']),
        ('pytest',[sys.executable,'-m','pytest','-o','addopts=','-q','-W','error',
                   '-p','no:cacheprovider',f'--junitxml={output / "pytest.xml"}']),
    ):
        print(f'Running {name}',flush=True)
        commands[name] = run_command(argv,repo,output/f'{name}.log')
    suites = list(ET.parse(output/'pytest.xml').getroot().iter('testsuite'))
    report['tests'] = {key:sum(int(suite.attrib.get(key,'0')) for suite in suites)
                       for key in ('tests','failures','errors','skipped')}
    report['statistical_oracle'] = {'sixty_same_sign_pairs':sign_test_pvalue([1.]*60,[0.]*60),
                                   'exact_reference':2.**-59}
    try:
        paired_permutation_pvalue([float('nan'),1.],[0.,0.])
        report['nan_scores_rejected'] = False
    except ValueError:
        report['nan_scores_rejected'] = True
    with tempfile.TemporaryDirectory(prefix='signtranslator-w0-wheel-') as temporary:
        base = Path(temporary)
        source = base/'source'; source.mkdir()
        for relative in before:
            destination = source/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(repo/relative,destination)
        shutil.copyfile(repo/'README.md',source/'README.md')
        wheels = base/'wheels'
        commands['wheel_build'] = run_command(
            [sys.executable,'-m','pip','wheel','--no-build-isolation','--no-deps',
             '--wheel-dir',str(wheels),str(source)],base,output/'wheel-build.log')
        wheel_files = list(wheels.glob('signtranslator-*.whl'))
        if len(wheel_files) != 1:
            raise RuntimeError('expected exactly one built wheel; inspect wheel-build.log')
        report['wheel'] = file_record(wheel_files[0])
        target = base/'installed'
        commands['wheel_install'] = run_command(
            [sys.executable,'-m','pip','install','--no-deps','--no-compile',
             '--target',str(target),str(wheel_files[0])],base,output/'wheel-install.log')
        commands['wheel_smoke'] = run_command(
            [sys.executable,'-c',WHEEL_SMOKE,str(target),str(base)],base,output/'wheel-smoke.log')
        if commands['wheel_smoke']['returncode'] == 0:
            report['wheel_smoke'] = json.loads((output/'wheel-smoke.log').read_text())
    after = snapshot(repo)
    after['scripts/w0_days2_5_verify.py'] = file_record(Path(__file__))
    report['source_unchanged_during_verification'] = before == after
    report['commands'] = commands
    report['phase3c_without_external_evidence'] = asdict(assess_phase3c_readiness())
    report['source_portfolio'] = CURRENT_PRE_PHASE_2_DECISION.to_dict()
    report['empirical_readiness_approved'] = False
    report['verification_passed'] = (
        before == after and all(c['returncode']==0 for c in commands.values())
        and report['tests']['tests'] > 0
        and all(report['tests'][key]==0 for key in ('failures','errors','skipped'))
        and report['nan_scores_rejected']
        and report['statistical_oracle']['sixty_same_sign_pairs']==2.**-59)
    report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['evidence_files'] = {p.name:file_record(p) for p in sorted(output.iterdir())}
    with (output/'verification.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({'verification_passed':report['verification_passed'],'tests':report['tests']}),flush=True)
    return 0 if report['verification_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
