"""Fit only audited scheduler/frontend CPU functions; never a step residual."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics

from TokenSim.af.calibration import HostOverheadProfile, EXTENDED_HOST_COVERAGE, content_fingerprint


LABELS = ('schedule','update_from_output','process_outputs')
FUNCTIONS = ['Scheduler.schedule','Scheduler.update_from_output','OutputProcessor.process_outputs']


def fit(documents, base, gpu_pair):
    groups, sources = {}, set()
    for doc in documents:
        key = (doc['batch'],doc['window'])
        if (any(type(v) is not int for v in key) or key in groups
                or key not in {(b,w) for b in (1,8) for w in range(3)}):
            raise ValueError('unique training batch1/8 window0/1/2 required')
        if (doc.get('timing_scope')!='host_service' or doc.get('contains_wait') is not False
                or doc.get('profiler_enabled') is not False or doc.get('execution_mode')!='eager'
                or doc.get('async_scheduling') is not False or doc.get('detokenize') is not False
                or doc.get('kv_transfer_enabled') is not False or doc.get('gpu_uuids')!=gpu_pair
                or doc.get('coverage')!=FUNCTIONS):
            raise ValueError('unsupported CPU measurement scope or configuration')
        if set(doc.get('sources',{}))!=set(LABELS):
            raise ValueError('missing source evidence')
        for value in doc['sources'].values():
            if not value.get('file') or not value.get('sha256') or type(value.get('line')) is not int:
                raise ValueError('invalid source evidence')
        sources.add(content_fingerprint(doc['sources']))
        rows = doc['samples']
        if len(rows)!=200:
            raise ValueError('200 steps per window required')
        for row in rows:
            if set(row)!=set(LABELS) or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in row.values()):
                raise ValueError('invalid CPU interval')
        groups[key] = dict(engine_step_schedule_us=[r['schedule'] for r in rows],
            engine_output_process_us=[r['update_from_output']+r['process_outputs'] for r in rows])
    if len(groups)!=6 or len(sources)!=1:
        raise ValueError('six independent windows and stable source required')
    result = copy.deepcopy(base)
    result.update(schema_version=2,profile_id='pkua100-codec-and-engine-cpu-20261009-v1')
    evidence = result['evidence']
    evidence.update(heldout_batches=[2],validation_status='pending',sample_count=1200,window_count=3)
    evidence['source'] += ';inherited-profile:'+content_fingerprint(base)+';engine-cpu:'+content_fingerprint(documents)
    report = dict(coverage=FUNCTIONS,excludes=['sample_tokens','_bookkeeping_sync','GPU and message submission'],costs={})
    for name in ('engine_step_schedule_us','engine_output_process_us'):
        windows = [group[name] for group in groups.values()]
        constant = statistics.mean(statistics.median(v) for v in windows)
        values = sorted(v for window in windows for v in window)
        quantiles = dict(p50=statistics.median(values),p95=values[math.ceil(len(values)*.95)-1])
        result['costs'][name] = dict(fixed_us=constant,per_token_us=0,coverage=[EXTENDED_HOST_COVERAGE[name]])
        evidence['quantiles_us'][name] = quantiles
        per_batch = {str(b):statistics.mean(statistics.median(groups[b,w][name]) for w in range(3)) for b in (1,8)}
        report['costs'][name] = dict(fixed_us=constant,quantiles_us=quantiles,batch_medians_us=per_batch,
            batch_residual_us={b:v-constant for b,v in per_batch.items()})
    evidence['residual_us'] = max(abs(v) for cost in report['costs'].values() for v in cost['batch_residual_us'].values())
    HostOverheadProfile(result,expected_runtime=base['runtime_fingerprint'],
        expected_environment=base['environment'],explicit_costs=())
    return result,report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('samples',type=Path)
    p.add_argument('--base-profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    base=json.loads(args.base_profile.read_text(encoding='utf-8'))
    result,report=fit([json.loads(f.read_text(encoding='utf-8')) for f in sorted(args.samples.glob('b*-w*.json'))],
        base,base['environment']['gpu_pair'].split(','))
    for path,data in ((args.output,result),(args.report,report)):
        with path.open('x',encoding='utf-8') as stream: json.dump(data,stream,indent=2,ensure_ascii=False)
    print(json.dumps(report))


if __name__=='__main__': main()
