"""Validate an already frozen transport candidate on unseen batch6 payloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from scripts.collect_afd_component_validation import guard


def main():
    p=argparse.ArgumentParser()
    for name in ('base','output','profile'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.output.resolve();base=a.base.resolve()
    if root.exists() or base not in root.parents:raise ValueError('new workspace experiment required')
    profile=json.loads(a.profile.read_text())
    if profile['evidence']['heldout_batches']!=[6] or 6 in profile['evidence']['training_batches']:
        raise ValueError('transport candidate must reserve batch6')
    digest=hashlib.sha256(a.profile.read_bytes()).hexdigest()
    guard();root.mkdir()
    contract=dict(profile_sha256=digest,batch=6,used_for_fit=False,
        collectors_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    (root/'contract.json').write_text(json.dumps(contract,indent=2))
    python=str(base/'envs/afd026/bin/python')
    env={k:v for k,v in os.environ.items() if not k.startswith(('AFD_CPU_','AFD_PREPARE_','AFD_CAPTURE_'))}
    for w in range(3):
        for backend in ('pynccl','torch'):
            for profiled in (False,True):
                guard()
                if hashlib.sha256(a.profile.read_bytes()).hexdigest()!=digest:raise ValueError('profile changed')
                name=f'{backend}-w{w}-'+('profiled' if profiled else 'unprofiled');directory=root/name
                command=[python,'-m','scripts.measure_afd_preposted_service',str(directory),
                    '--backend',backend,'--window',str(w),'--validation-batch','6']
                if profiled:command.append('--profile')
                with (root/f'{name}.log').open('x') as log:
                    subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                if profiled:
                    subprocess.run([python,'-m','scripts.summarize_afd_preposted_kernels',str(directory),
                        '--backend',backend,'--window',str(w),'--validation-batch','6',
                        '--output',str(root/f'{backend}-w{w}-kernels.json')],env=env,stdout=subprocess.DEVNULL,check=True)
                print('completed '+name,flush=True)
    (root/'completed.json').write_text(json.dumps(contract,indent=2))


if __name__=='__main__':main()
