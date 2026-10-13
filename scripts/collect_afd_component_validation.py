"""Independent batch6 checks against unchanged frozen costs, never fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from scripts.measure_afd_control_staging import UUIDS, check_idle


def guard():
    for attempt in range(6):
        try:
            check_idle()
            break
        except RuntimeError:
            if attempt==5:
                raise
            time.sleep(1)
    for port in (18039,18040,19093,18479):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',port))==0:
                raise RuntimeError(f'experiment port occupied: {port}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args()
    base=a.base.resolve(); root=a.output.resolve()
    if base not in root.parents or root.exists():
        raise ValueError('new output directory within authorized workspace required')
    bundle=json.loads(a.bundle.read_text())
    if 6 in bundle['host_profile']['evidence']['training_batches']:
        raise ValueError('batch6 already used for training')
    digest=hashlib.sha256(a.bundle.read_bytes()).hexdigest()
    guard();root.mkdir()
    manifest=dict(validation_batch=6,used_for_fit=False,bundle_sha256=digest,
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in Path(__file__).parent.glob('*.py')},completed=[])
    with (root/'frozen-validation-contract.json').open('x') as f:json.dump(manifest,f,indent=2)
    env={k:v for k,v in os.environ.items() if not k.startswith(('AFD_PREPARE_','AFD_CAPTURE_','AFD_CPU_'))}
    env.update(AFD_COMPONENT_VALIDATION='6',
        AFD_CPU_COUNTER_LIB=str(base/'profiling/afd_cpu_api_counter_20261009.so'))
    python=str(base/'envs/afd026/bin/python')
    def run(name,module,args):
        guard()
        if hashlib.sha256(a.bundle.read_bytes()).hexdigest()!=digest:
            raise ValueError('frozen bundle changed')
        with (root/f'{name}.log').open('x') as f:
            subprocess.run([python,'-m','scripts.'+module,*map(str,args)],env=env,
                           stdout=f,stderr=subprocess.STDOUT,check=True)
        manifest['completed'].append(name)
        (root/'progress.json').write_text(json.dumps(manifest,indent=2))
        print('completed '+name,flush=True)
    for mode in ('inputs','metadata','sampled_tokens','bookkeeping','compute_cpu'):
        run(mode,'collect_afd_prepare',['--base',base,'--output',root/mode,
            '--mode',mode,'--validation-batch',6])
    run('engine','collect_afd_engine_cpu',['--base',base,'--output',root/'engine'])
    run('staging','measure_afd_control_staging',[root/'staging','--validation-batch',6])
    for backend in ('torch','pynccl'):
        for w in range(3):
            run(f'{backend}-w{w}','measure_afd_submit',['--output',root/f'{backend}-w{w}',
                '--backend',backend,'--window',w,'--validation-batch',6])
    source=base/'src/afd-plugin-dbo-cache-20261007/afd_plugin/connectors/metadata.py'
    for w in range(3):
        run(f'codec-w{w}','measure_afd_cpu_codecs',[source,root/f'codec-w{w}.json','--batch',6,'--window',w])
    if hashlib.sha256(a.bundle.read_bytes()).hexdigest()!=digest:
        raise ValueError('frozen bundle changed')
    print('independent component collection complete',flush=True)


if __name__=='__main__':main()
