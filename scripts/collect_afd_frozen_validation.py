"""No-hook native validation of a frozen partial calibration, then route capture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from scripts.collect_afd_component_validation import guard


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validation_batches(bundle):
    if bundle.get('schema_version')==4:
        if (bundle.get('training_batches')!=[1,8] or bundle.get('regression_batches')!=[4]
            or bundle.get('heldout_batches')!=[7] or bundle.get('batch7_used_for_fit') is not False):
            raise ValueError('v4 requires training1/8, regression4 and unseen heldout7')
        for name in ('host_profile','operation_profile'):
            if bundle.get(name,{}).get('evidence',{}).get('heldout_batches')!=[7]:
                raise ValueError('v4 profiles must reserve batch7')
        if not bundle.get('device_service_profile') or not bundle.get('transport_profile'):
            raise ValueError('v4 requires separate device and transport services')
        return (1,8,4,7),(7,)
    if bundle.get('schema_version') == 3:
        if (bundle.get('training_batches') != [1,8] or bundle.get('regression_batches') != [4,2,3]
                or bundle.get('heldout_batches') != [5] or bundle.get('batch5_used_for_fit') is not False):
            raise ValueError('v3 requires training1/8, regression4/2/3 and unseen heldout5')
        for name in ('host_profile', 'operation_profile'):
            if bundle.get(name, {}).get('evidence', {}).get('heldout_batches') != [5]:
                raise ValueError('v3 profiles must reserve batch5')
        return (1,8,4,2,3,5),(5,)
    if bundle.get('schema_version')==1:
        return (1,8,4),(4,)
    if (bundle.get('schema_version')!=2 or bundle.get('training_batches')!=[1,8]
            or bundle.get('regression_batches')!=[4] or bundle.get('heldout_batches')!=[2]):
        raise ValueError('v2 requires training1/8, regression4 and new heldout2')
    for name in ('host_profile','operation_profile'):
        profile=bundle.get(name,{})
        if profile.get('evidence',{}).get('heldout_batches')!=[2]:
            raise ValueError('profile must reserve batch2 before native validation')
    return (1,8,4,2),(2,)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--supplemental-batch',type=int,choices=[3],
                   help='Additional unseen workload; preserves original batch2 acceptance')
    a=p.parse_args()
    base,root=a.base.resolve(),a.output.resolve()
    bundle=json.loads(a.bundle.read_text())
    if root.exists() or base not in root.parents or bundle.get('scope')!='partial_calibration_step_validation':
        raise ValueError('new experiment and explicit frozen calibration scope required')
    for name in ('host_profile','operation_profile'):
        if not bundle.get(name): raise ValueError('both profiles must be frozen')
    batches,capture_batches=validation_batches(bundle)
    if a.supplemental_batch is not None:
        if bundle.get('supplemental_validation_batches')!=[a.supplemental_batch]:
            raise ValueError('supplemental workload must be declared before execution')
        batches=capture_batches=(a.supplemental_batch,)
    source=base/'profiling/afd-native-collection-20261007-v1'
    driver=(source/'timing-b1-w0/attention.py').read_text()
    driver='\n'.join(line for line in driver.splitlines() if 'profile incomplete' not in line)+'\n'
    runner=(source/'timing-b1-w0/run_curve.py').read_text().replace('dict(ENABLE=1,WAIT=32','dict(ENABLE=0,WAIT=32')
    runner='\n'.join(line for line in runner.splitlines() if "assert len(list(directory.glob('*/*.pt.trace.json')))==2" not in line)+'\n'
    if 'dict(ENABLE=0,WAIT=32' not in runner: raise ValueError('profiler disable point drift')
    guard()
    root.mkdir()
    (root/'frozen-bundle.json').write_text(json.dumps(bundle,indent=2))
    frozen_sha=sha(root/'frozen-bundle.json')
    results=[]
    # All training windows precede held-out batch4; no parameter fitting here.
    for batch in batches:
        for window in range(3):
            guard()
            directory=root/f'timing-b{batch}-w{window}'
            directory.mkdir()
            (directory/'attention.py').write_text(driver)
            (directory/'run_curve.py').write_text(runner)
            env={k:v for k,v in os.environ.items() if not k.startswith(('AFD_CAPTURE_','AFD_PREPARE_','AFD_CPU_','AFD_ENGINE_'))}
            env.update(AF_WORKSPACE=str(base),AFD_ISOLATED_PLUGIN=str(base/'src/afd-plugin-dbo-cache-20261007'))
            with (directory/'collection.log').open('x') as log:
                subprocess.run([str(base/'envs/afd026/bin/python'),str(directory/'run_curve.py'),
                                '--concurrency',str(batch)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((directory/f'c{batch}/result.json').read_text())
            calls=json.loads((directory/f'c{batch}/calls.json').read_text())
            values=[step*1000 for call in calls for step in call['step_ms'][1:]]
            outputs=json.loads((directory/f'c{batch}/outputs.json').read_text())
            assert result['repeat_token_min_match']==1 and len(values)>=310
            assert all(o==outputs[0] for o in outputs)
            assert sha(root/'frozen-bundle.json')==frozen_sha
            row=dict(batch=batch,window=window,samples=len(values),mean_us=statistics.mean(values),
                     p50_us=statistics.median(values),p95_us=sorted(values)[int(.95*len(values))-1],
                     output_sha256=hashlib.sha256(json.dumps(outputs[0]).encode()).hexdigest(),
                     profiler_enabled=False,hooks_enabled=False)
            results.append(row)
            print(json.dumps(row),flush=True)
    (root/'timing-summary.json').write_text(json.dumps(dict(frozen_bundle_sha256=frozen_sha,
        scope=bundle['scope'],windows=results),indent=2))
    # Separate diagnostic capture; never use its duration for calibration.
    batch=capture_batches[0]
    guard()
    directory=root/f'capture-b{batch}'
    directory.mkdir()
    capture_driver=(source/'capture-b4-w0/attention.py').read_text()
    capture_driver='\n'.join(line for line in capture_driver.splitlines() if 'profile incomplete' not in line)+'\n'
    (directory/'attention.py').write_text(capture_driver)
    (directory/'run_curve.py').write_text(runner)
    env={k:v for k,v in os.environ.items() if not k.startswith(('AFD_CAPTURE_','AFD_PREPARE_'))}
    env.update(AF_WORKSPACE=str(base),AFD_ISOLATED_PLUGIN=str(source/'plugin'),
               AFD_CAPTURE_DIR=str(directory),AFD_CAPTURE_BATCH=str(batch),AFD_CAPTURE_STEPS='31')
    with (directory/'collection.log').open('x') as log:
        subprocess.run([str(base/'envs/afd026/bin/python'),str(directory/'run_curve.py'),
                        '--concurrency',str(batch)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert sha(root/'frozen-bundle.json')==frozen_sha
    print('validation and separate held-out route capture complete',flush=True)


if __name__=='__main__': main()
