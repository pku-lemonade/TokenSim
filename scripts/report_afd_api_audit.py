"""Classify excluded CUDA API intervals without relabeling them as fixed fees."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


QUERY_APIS={'cudaGetDevice','cudaEventQuery','cuDevicePrimaryCtxGetState','cuKernelGetName',
            'cudaGetLastError','cudaDeviceGetAttribute','cuCtxGetCurrent','cuPointerGetAttribute',
            'cuKernelGetFunction','cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags'}
LAUNCH_APIS={'cudaLaunchKernel','cuLaunchKernel','cuLaunchKernelEx','cudaLaunchKernelExC'}


def summarize(doc):
    if doc.get('timing_scope')!='diagnostic_api_audit_not_fit' or not doc.get('prior_sync_outside_timer'):
        raise ValueError('API audit boundaries required')
    result={}
    for group,rows in doc['groups'].items():
        if len(rows)!=200:raise ValueError('complete 200-call window required')
        components=defaultdict(list);names=defaultdict(int)
        for row in rows:
            sums=dict(query=0.,launch=0.,async_copy=0.,unclassified=0.)
            for api in row['outer_cuda_apis']:
                v=api['duration_us'];name=api['name']
                if type(v) not in (int,float) or not math.isfinite(v) or v<0 or type(api['calls']) is not int or api['calls']<1:
                    raise ValueError('invalid API sample')
                category=('query' if name in QUERY_APIS else 'launch' if name in LAUNCH_APIS
                          else 'async_copy' if name=='cudaMemcpyAsync' else 'unclassified')
                sums[category]+=v;names[name]+=api['calls']
            if not math.isclose(sum(sums.values()),row['excluded_cuda_api_us'],abs_tol=1e-7):
                raise ValueError('nested API union mismatch')
            for category,value in sums.items():components[category].append(value)
        result[group]=dict(api_calls=dict(names),median_us={k:statistics.median(v) for k,v in components.items()},
            total_api_median_us=statistics.median(r['excluded_cuda_api_us'] for r in rows),
            unknown_api_present=any(components['unclassified']),admissible_for_fixed_cost=False)
    return dict(role=doc['role'],batch=doc['batch'],window=doc['window'],groups=result,
        scope='diagnostic outer API interval union; not calibrated service',
        limitations=['query/event return can depend on device state',
                    'async copy may block on pageable staging or stream dependencies',
                    'launch CPU intervals must not duplicate analytical kernel startup coverage',
                    'instrumented callback audit does not establish uninstrumented costs'])


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();paths=sorted(a.root.rglob('*-compute-cpu-samples.json'))
    if not paths:raise ValueError('no API samples')
    report=dict(results=[summarize(json.loads(f.read_text(encoding='utf-8'))) for f in paths],
        source_sha256={str(f.relative_to(a.root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
        fixed_cost_profile_generated=False)
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)


if __name__=='__main__':main()
