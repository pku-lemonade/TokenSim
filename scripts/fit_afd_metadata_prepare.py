"""Fit isolated A/F metadata operations; no communication waits are included."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
from TokenSim.af.calibration import content_fingerprint
from TokenSim.af.operation_profile import OperationServiceProfile


def fit(documents,base):
    groups = {}
    sources = {}
    for d in documents:
        name = d['operation']
        if name not in ('a_attention_metadata','f_dp_metadata'):
            raise ValueError('unexpected metadata operation')
        key = (name,d['batch'],d['window'])
        if key in groups or d['batch'] not in (1,8) or d['window'] not in range(3):
            raise ValueError('duplicate or untrained metadata window')
        a_side = name == 'a_attention_metadata'
        completion = 'device_synchronize_after_operation' if a_side else 'cpu_api_return_data_ready'
        if (d.get('timing_scope')!='operation_service' or d.get('external_wait_excluded') is not True
            or d.get('profiler_enabled') is not False or d.get('completion')!=completion
            or d.get('prior_work_synchronize_outside_timer') is not a_side
            or d.get('gpu_uuid')!=base['environment']['gpu_pair'].split(',')[0 if a_side else 1]):
            raise ValueError('metadata boundary/environment mismatch')
        if len(d['samples'])!=200:
            raise ValueError('200 samples required')
        values = []
        for r in d['samples']:
            if any(type(r[k]) not in (int,float) or not math.isfinite(r[k]) or r[k]<0
                   for k in ('service_us','api_us','completion_tail_us')):
                raise ValueError('invalid metadata duration')
            if not math.isclose(r['service_us'],r['api_us']+r['completion_tail_us'],abs_tol=1e-8):
                raise ValueError('metadata interval mismatch')
            values.append(r['service_us'])
        groups[key] = values
        sources.setdefault(name,set()).add(d['source_sha256'])
    if len(groups)!=12 or any(len(s)!=1 for s in sources.values()):
        raise ValueError('three windows per operation/batch and stable source required')
    result = copy.deepcopy(base)
    result['profile_id']='pkua100-preparation-and-control-20261007-v3'
    for name in ('a_attention_metadata','f_dp_metadata'):
        windows = [v for (op,_,_),v in groups.items() if op==name]
        values = sorted(v for group in windows for v in group)
        result['operations'][name]=dict(fixed_us=statistics.mean(statistics.median(v) for v in windows),
            coverage=[name],input_ready='prior_device_work_complete' if name.startswith('a_') else 'control_decoded_cpu_metadata_ready',
            completion='device_synchronize_after_operation' if name.startswith('a_') else 'cpu_api_return_data_ready',
            resource_model='calling_thread_single_inflight')
        result['evidence']['statistics'][name]=dict(p50_us=statistics.median(values),
            p95_us=values[math.ceil(len(values)*.95)-1],samples=len(values))
    result['evidence']['source']+=';metadata:'+content_fingerprint(documents)
    result['evidence']['audit']+=' A metadata constructs attention descriptors after input preparation and before control send; F DP metadata transforms already decoded CPU counts before activation recv. No network I/O in either operation. A includes explicit own completion synchronize; F is local CPU work. Source hashes='+str({k:next(iter(v)) for k,v in sources.items()})
    OperationServiceProfile(result,expected_runtime=base['runtime_fingerprint'],expected_environment=base['environment'],explicit_costs=())
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('samples',type=Path)
    p.add_argument('--base-profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=fit([json.loads(f.read_text()) for f in sorted(a.samples.glob('*.json'))],json.loads(a.base_profile.read_text()))
    with a.output.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps({k:v['fixed_us'] for k,v in result['operations'].items()}))


if __name__=='__main__': main()
