"""Same-route three-group ablation; synthetic zero readback preserves dependency."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False)
    full=load('docs/plans/afd-compute-cpu-profile-20261009.json')
    operation=load('docs/plans/afd-readback-profile-20261009.json')
    codec=load('docs/plans/afd-measured-codec-profile-20261007.json')
    preparation=load('docs/plans/afd-preparation-profile-20261007.json')
    result=[]
    for name in ('baseline','readback','compute','control_periphery','all'):
        host=copy.deepcopy(full)
        included={k for k in full['costs'] if k.startswith('control_') or name=='all'
                  or (name=='compute' and k.startswith('compute_'))
                  or (name=='control_periphery' and k.startswith(('engine_','send_','receive_')))}
        host['costs']={k:v for k,v in full['costs'].items() if k in included}
        host['evidence']['quantiles_us']={k:v for k,v in full['evidence']['quantiles_us'].items() if k in included}
        host['profile_id']+=':ablation:'+name
        op=copy.deepcopy(operation if name in ('readback','all','control_periphery') else preparation)
        if name=='control_periphery':
            # Preserve CPU-token-ready ordering without charging the readback group.
            # This document is explicitly a counterfactual, never a measured zero.
            op['profile_id']+=':synthetic-zero-readback-ablation'
            op['operations']['sampled_token_readback']['fixed_us']=0
            op['evidence']['kind']='synthetic_test'
            op['evidence']['source']+=';counterfactual-zero-readback-not-measured'
        for label,doc in (('host',host),('operation',op)):
            (a.output/f'{name}-{label}.json').write_text(json.dumps(doc,indent=2),encoding='utf-8')
        for batch in (1,8):
            output=a.output/f'{name}-b{batch}'
            subprocess.run([sys.executable,'-m','scripts.run_tiny_afd_alignment','--output',str(output),
                '--batches',str(batch),'--host-profile',str(a.output/f'{name}-host.json'),
                '--operation-profile',str(a.output/f'{name}-operation.json'),
                '--routing-replay',f'outputs/afd-live-capture-20261007/replay-b{batch}.json',
                '--calibration-environment','outputs/afd-live-capture-20261007/environment.json'],check=True)
            replay=load(output/f'tokensim-b{batch}.json')
            result.append(dict(case=name,batch=batch,mean_step_us=replay['decode']['mean_step_us'],
                               critical_chain_us=replay['decode']['mean_critical_chain_us']))
    for row in result:
        row['delta_from_baseline_us']=row['mean_step_us']-next(
            r['mean_step_us'] for r in result if r['case']=='baseline' and r['batch']==row['batch'])
    (a.output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
