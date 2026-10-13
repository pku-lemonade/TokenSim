"""Native GPU-only component windows; profiler intervals never fit CPU or step."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from scripts.collect_afd_component_validation import guard


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--heldout-bundle',type=Path)
    a=p.parse_args();base=a.base.resolve();root=a.output.resolve()
    if root.exists() or base not in root.parents:raise ValueError('new workspace experiment required')
    batches=(7,) if a.heldout_bundle else (1,8)
    frozen=hashlib.sha256(a.heldout_bundle.read_bytes()).hexdigest() if a.heldout_bundle else None
    source=base/'profiling/afd-native-collection-20261007-v1/timing-b1-w0'
    driver=(source/'attention.py').read_text();runner=(source/'run_curve.py').read_text()
    if 'dict(ENABLE=1,WAIT=32' not in runner:raise ValueError('native profiler driver drift')
    # Import only definitions: the historical script has a command-line footer.
    code=Path(__file__).with_name('summarize_layer_compute.py').read_text()
    ns={};exec(code.split('root = Path(sys.argv[1])')[0],ns)
    guard();root.mkdir();rows=[]
    (root/'contract.json').write_text(json.dumps(dict(batches=batches,windows=3,
        timing_scope='GPU kernel service excluding NCCL and CPU',heldout_bundle_sha256=frozen,
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.encode()).hexdigest(),
        runner_sha256=hashlib.sha256(runner.encode()).hexdigest()),indent=2))
    for w in range(3):
        for b in batches:
            guard()
            if frozen and hashlib.sha256(a.heldout_bundle.read_bytes()).hexdigest()!=frozen:
                raise ValueError('heldout bundle changed')
            case=root/f'b{b}-w{w}';case.mkdir()
            (case/'attention.py').write_text(driver);(case/'run_curve.py').write_text(runner)
            env={k:v for k,v in os.environ.items() if not k.startswith(('AFD_CPU_','AFD_CAPTURE_','AFD_PREPARE_'))}
            env.update(AF_WORKSPACE=str(base),AFD_ISOLATED_PLUGIN=str(base/'src/afd-plugin-dbo-cache-20261007'))
            with (case/'collection.log').open('x') as f:
                subprocess.run([str(base/'envs/afd026/bin/python'),str(case/'run_curve.py'),
                    '--concurrency',str(b)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            row=dict(batch=b,window=w)
            for role in ('attention','ffn'):
                path=next((case/f'c{b}'/role).glob('*.pt.trace.json'))
                row[role]=ns['summarize'](path,role,b)
                row[role]['trace_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                if row[role]['unclassified']:raise ValueError('unclassified device kernel')
            rows.append(row)
            (root/'device-summary.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(dict(batch=b,window=w,device_components_complete=True)),flush=True)


if __name__=='__main__':main()
