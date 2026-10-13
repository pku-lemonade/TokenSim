"""Fit CPU bookkeeping exclusive of the separately calibrated token readback."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
from TokenSim.af.calibration import HostOverheadProfile,EXTENDED_HOST_COVERAGE,content_fingerprint


def fit(documents,base):
    groups={};sources=set()
    for doc in documents:
        key=(doc['batch'],doc['window'])
        if any(type(v) is not int for v in key) or key not in {(b,w) for b in (1,8) for w in range(3)} or key in groups:
            raise ValueError('unique training window required')
        expected=dict(operation='runner_bookkeeping_cpu',timing_scope='host_service',
            external_wait_excluded=True,prior_work_synchronize_outside_timer=True,
            profiler_enabled=False,readback_excluded=True,async_scheduling=False,routing_d2h=False,logprobs=False,
            gpu_uuid=base['environment']['gpu_pair'].split(',')[0])
        if any(doc.get(k)!=v or (isinstance(v,bool) and doc.get(k) is not v) for k,v in expected.items()):
            raise ValueError('invalid CPU bookkeeping coverage/boundary')
        if not doc.get('source_sha256'):raise ValueError('missing source')
        sources.add(doc['source_sha256'])
        if len(doc['samples'])!=200:raise ValueError('200 samples required')
        values=[]
        for row in doc['samples']:
            if (type(row['context']) is not int or not 129<=row['context']<=159 or row['readback_calls']!=1
                    or any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) or row[k]<0
                           for k in ('service_us','wall_us','excluded_readback_us'))
                    or not math.isclose(row['wall_us'],row['service_us']+row['excluded_readback_us'],abs_tol=1e-7)):
                raise ValueError('invalid exclusive interval')
            values.append(row['service_us'])
        groups[key]=values
    if len(groups)!=6 or len(sources)!=1:raise ValueError('six windows and stable source required')
    result=copy.deepcopy(base)
    result['profile_id']='pkua100-submit-output-complete-20261009-v1'
    name='engine_runner_bookkeeping_us'
    constant=statistics.mean(statistics.median(v) for v in groups.values())
    values=sorted(v for group in groups.values() for v in group)
    result['costs'][name]=dict(fixed_us=constant,per_token_us=0,coverage=[EXTENDED_HOST_COVERAGE[name]])
    evidence=result['evidence']
    evidence.update(validation_status='pending')
    evidence['quantiles_us'][name]=dict(p50=statistics.median(values),p95=values[math.ceil(.95*len(values))-1])
    evidence['sample_count']+=len(values)
    evidence['source']+=';bookkeeping:'+content_fingerprint(documents)
    evidence['residual_us']=max(evidence['residual_us'],max(abs(statistics.median(v)-constant) for v in groups.values()))
    HostOverheadProfile(result,expected_runtime=base['runtime_fingerprint'],expected_environment=base['environment'],explicit_costs=())
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('samples',type=Path)
    p.add_argument('--base-profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=fit([json.loads(f.read_text()) for f in sorted(a.samples.glob('b*-w*.json'))],
        json.loads(a.base_profile.read_text(encoding='utf-8')))
    with a.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,ensure_ascii=False)
    print(result['costs']['engine_runner_bookkeeping_us'])


if __name__=='__main__':main()
