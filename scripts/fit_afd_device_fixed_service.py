"""Fit one GPU-only service offset to kernel sums, never to step residuals."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def device_totals(row):
    a=row['attention'];f=row['ffn']
    if a['unclassified'] or f['unclassified'] or a['decode_steps']!=31 or f['decode_steps']!=31:
        raise ValueError('complete classified decode device evidence required')
    # Slot mapping belongs to the already measured input preparation operation.
    # All other non-NCCL kernels are the model's GPU computation, not CPU APIs.
    return {'a-pool:0':a['mean_us']['total']-a['mean_us'].get('outside/kv_slot_mapping',0),
            'f-pool:0':f['mean_us']['total']}


def fit(rows,bases):
    if len(rows)!=6 or {(r['batch'],r['window']) for r in rows}!={(b,w) for b in (1,8) for w in range(3)}:
        raise ValueError('batch1/8 three independent GPU windows required')
    points=[]
    for row in rows:
        for role,native in device_totals(row).items():
            b=row['batch'];intercept=bases[0,b][role];slope=bases[1,b][role]-intercept
            if not all(math.isfinite(x) for x in (native,intercept,slope)) or native<=0 or intercept<0 or slope<=0:
                raise ValueError('invalid device calibration basis')
            points.append(dict(batch=b,window=row['window'],role=role,native_us=native,intercept_us=intercept,slope=slope))
    numerator=sum(p['slope']/p['native_us']*(1-p['intercept_us']/p['native_us']) for p in points)
    denominator=sum((p['slope']/p['native_us'])**2 for p in points)
    constant=max(0.,numerator/denominator)
    for p in points:
        p['predicted_us']=p['intercept_us']+constant*p['slope']
        p['residual_us']=p['predicted_us']-p['native_us']
    return constant,points


def main():
    p=argparse.ArgumentParser()
    for name in ('native','basis-root','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();raw=a.native.read_bytes();rows=json.loads(raw);bases={};hashes={}
    for floor in (0,1):
        for b in (1,8):
            path=a.basis_root/f'floor{floor}'/f'b{b}'/f'tokensim-b{b}.json'
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            bases[floor,b]=json.loads(path.read_text(encoding='utf-8'))['decode']['compute_service_by_endpoint_us']
    constant,points=fit(rows,bases)
    d=dict(schema_version=1,model_id='qwen3-moe-tiny-random',unit='us',timing_scope='device_service',
        contains_cpu_submit=False,contains_wait=False,kernel_fixed_us=constant,training_batches=[1,8],heldout_batches=[7],
        source_sha256=hashlib.sha256(raw).hexdigest(),basis_sha256=hashes,fit_points=points,
        method='nonnegative relative least squares of classified A/F GPU kernel service sums',
        excluded_coverage=['NCCL occupancy','CPU/API time','input preparation slot mapping','step residual'],
        limitations=['single device-service coefficient is an aggregate approximation, not measured CPU launch latency',
                     'profiling perturbation and device clocks remain uncertainty; independent batch7 validation required'])
    with a.output.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2)
    print(json.dumps(dict(kernel_fixed_us=constant,training_points=len(points))))


if __name__=='__main__':main()
