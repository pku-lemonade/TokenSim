"""Fit only explicitly attributable host service samples; no whole-step residuals."""
import argparse
import json
import math
import statistics
from pathlib import Path
from TokenSim.af.calibration import HOST_COVERAGE, HostOverheadProfile, content_fingerprint


def fit(document, *, profile_id, method='constant'):
    if document.get('schema_version') != 1 or method not in ('constant','nonnegative_linear'):
        raise ValueError('unsupported samples version or fitting rule')
    rows=document['samples']
    if not rows:
        raise ValueError('no host service samples')
    by_cost={}
    for row in rows:
        name=row['cost_field']
        if (name not in HOST_COVERAGE or row.get('timing_scope')!='host_service'
            or row.get('contains_wait') is not False or row.get('excludes_device_launch') is not True
            or row.get('coverage')!=[HOST_COVERAGE[name]] or row.get('phase')!='decode'
            or row.get('profiler_enabled') is not False or not row.get('source')):
            raise ValueError('only sourced, non-profiled, uniquely covered host service excluding wait/launch is admissible')
        for field in ('batch','context'):
            if type(row[field]) is not int or row[field]<1:
                raise ValueError('positive integer batch/context required')
        if row['batch']==4:
            raise ValueError('batch4 is held out and cannot enter fitting')
        value=row['duration_us']
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError('invalid host duration')
        if not isinstance(row.get('window_id'),str) or not row['window_id']:
            raise ValueError('sample requires independent window identity')
        by_cost.setdefault(name,[]).append(row)
    costs,quantiles,residuals={},{},[]
    for name, samples in by_cost.items():
        batches={r['batch'] for r in samples}
        if not {1,8}.issubset(batches):
            raise ValueError('each cost needs batch1 and batch8 training')
        for batch in batches:
            if len({r['window_id'] for r in samples if r['batch']==batch})<3:
                raise ValueError('each cost/batch needs three independent windows')
        # Equal weight per batch/window; do not let one long window dominate.
        groups={}
        for row in samples:
            groups.setdefault((row['batch'],row['window_id']),[]).append(row['duration_us'])
        points=[(batch,statistics.median(values)) for (batch,_),values in groups.items()]
        ys=[value for _,value in points]
        if method=='constant':
            intercept,slope=statistics.mean(ys),0.0
        else:
            if len(batches)<3:
                raise ValueError('scale fit needs three distinct training batches')
            xs=[batch for batch,_ in points]
            mean_x,mean_y=statistics.mean(xs),statistics.mean(ys)
            slope=sum((x-mean_x)*(y-mean_y) for x,y in points)/sum((x-mean_x)**2 for x in xs)
            intercept=mean_y-slope*mean_x
            candidates=[(max(0,mean_y),0), (0,max(0,sum(x*y for x,y in points)/sum(x*x for x in xs)))]
            if intercept>=0 and slope>=0:
                candidates.append((intercept,slope))
            intercept,slope=min(candidates,key=lambda pair:sum((y-pair[0]-pair[1]*x)**2 for x,y in points))
        costs[name]=dict(fixed_us=intercept,per_token_us=slope,coverage=[HOST_COVERAGE[name]])
        values=sorted(r['duration_us'] for r in samples)
        quantiles[name]=dict(p50=statistics.median(values),p95=values[math.ceil(.95*len(values))-1])
        residuals.extend((y-intercept-slope*x)**2 for x,y in points)
    batches=sorted({r['batch'] for r in rows})
    contexts=[r['context'] for r in rows]
    profile=dict(schema_version=1, profile_id=profile_id,unit='us',timing_scope='host_service',
        contains_wait=False,excludes_device_launch=True,protocol='vllm_afd',
        runtime_fingerprint=document['runtime_fingerprint'],environment=document['environment'],
        batch_range=[min(batches),max(batches)], context_range=[min(contexts),max(contexts)],costs=costs,
        evidence=dict(source=document['source']+':'+content_fingerprint(document),
            kind=document['kind'],sample_count=len(rows),
            window_count=min(len({r['window_id'] for r in samples if r['batch']==batch})
                for samples in by_cost.values() for batch in {r['batch'] for r in samples}),
            training_batches=batches,heldout_batches=[4],method=method,
            residual_us=math.sqrt(statistics.mean(residuals)),quantiles_us=quantiles,validation_status='pending'))
    HostOverheadProfile(profile,expected_runtime=document['runtime_fingerprint'],
        expected_environment=document['environment'],explicit_costs=())
    return profile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('samples',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile-id',required=True)
    parser.add_argument('--method',choices=['constant','nonnegative_linear'],default='constant')
    args=parser.parse_args()
    profile=fit(json.loads(args.samples.read_text()),profile_id=args.profile_id,method=args.method)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(profile,stream,indent=2)


if __name__=='__main__': main()
