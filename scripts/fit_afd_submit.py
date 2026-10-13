"""Fit nonblocking CPU submit intervals only after delayed-peer return audit."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics

from TokenSim.af.calibration import HostOverheadProfile, EXTENDED_HOST_COVERAGE, content_fingerprint


def fit(documents,base):
    groups={}
    sources={}
    pair=base['environment']['gpu_pair'].split(',')
    for doc in documents:
        if (doc.get('gpu_uuids')!=pair or doc.get('timing_scope')!='cpu_api_entry_to_return'
                or doc.get('completion_sync_outside_timer') is not True
                or doc.get('prior_device_work_synchronized') is not True
                or doc.get('blocking_wait') is not False or doc.get('profiler_enabled') is not False):
            raise ValueError('unqualified submission boundary/environment')
        for row in doc['rows']:
            backend=row['backend']
            if backend not in ('torch','pynccl') or row['action'] not in ('send','receive'):
                raise ValueError('unexpected API')
            source=doc['sources'][row['action']]
            if not source.get('file') or not source.get('sha256'):raise ValueError('missing API source')
            sources.setdefault((backend,row['action']),set()).add(content_fingerprint(source))
            batch,window=row['batch'],row['window']
            if type(batch) is not int or type(window) is not int or batch not in (1,8) or window not in range(3):
                raise ValueError('non-training window')
            shapes={(8,'int64'),(179,'uint8')} if backend=='torch' else {(batch*512,'bfloat16')}
            if (row['bytes'],row['dtype']) not in shapes:raise ValueError('wire shape mismatch')
            if row['direction'] not in (('a_to_f',) if backend=='torch' else ('a_to_f','f_to_a')):
                raise ValueError('invalid direction')
            treatment=(row['delayed_role'],row['delay_ms'])
            if treatment not in (('none',0),('sender',5),('sender',20),('receiver',5),('receiver',20)):
                raise ValueError('missing supported delay treatment')
            key=(backend,batch,window,row['bytes'],row['direction'],row['action'],*treatment)
            if key in groups:raise ValueError('duplicate submission window')
            samples=row['samples']
            if len(samples)!=50:raise ValueError('50 post-warmup samples required')
            values=[]
            early=(row['action']=='send' and treatment[0]=='receiver') or (row['action']=='receive' and treatment[0]=='sender')
            for sample in samples:
                value=sample['service_us']
                if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                    raise ValueError('invalid CPU duration')
                if early and sample['returned_before_peer_entered'] is not True:
                    raise ValueError('API did not return before delayed peer; external waiting not excluded')
                values.append(value)
            groups[key]=values
    if len(groups)!=240 or any(len(s)!=1 for s in sources.values()):
        raise ValueError('all backend/batch/direction/delay/window cases and stable source required')
    result=copy.deepcopy(base)
    result.update(schema_version=2,profile_id='pkua100-codec-engine-submit-20261009-v1')
    evidence=result['evidence']
    evidence.update(heldout_batches=[2],validation_status='pending')
    evidence['source']+=';parent-profile:'+content_fingerprint(base)+';submit:'+content_fingerprint(documents)
    report={}
    for backend,directions in (('torch',('a_to_f',)),('pynccl',('a_to_f','f_to_a'))):
        for direction in directions:
            for action in ('send','receive'):
                matched={k:v for k,v in groups.items() if (k[0],k[4],k[5])==(backend,direction,action)}
                clean={k:v for k,v in matched.items() if k[-2:]==('none',0)}
                per_window=[statistics.mean(statistics.median(v) for k,v in clean.items() if (k[1],k[2])==(b,w))
                            for b in (1,8) for w in range(3)]
                constant=statistics.mean(per_window)
                early_role='receiver' if action=='send' else 'sender'
                delayed=[statistics.median(v) for k,v in matched.items() if k[-2]==early_role]
                # A returned API plus a bounded delay-control median is evidence
                # for enqueue service; delayed samples are never training data.
                if max(delayed)>max(3*constant,constant+100):
                    raise ValueError('delayed-peer CPU service is unstable; do not fit')
                name=f'{action}_{"torch_distributed_nccl" if backend=="torch" else "pynccl"}_{direction}_us'
                values=sorted(v for rows in clean.values() for v in rows)
                quantiles=dict(p50=statistics.median(values),p95=values[math.ceil(len(values)*.95)-1])
                result['costs'][name]=dict(fixed_us=constant,per_token_us=0,coverage=[EXTENDED_HOST_COVERAGE[name]])
                evidence['quantiles_us'][name]=quantiles
                report[name]=dict(fixed_us=constant,quantiles_us=quantiles,window_medians_us=per_window,
                    max_delayed_peer_median_us=max(delayed),all_early_calls_returned_before_peer=True)
    evidence['sample_count']+=sum(len(v) for k,v in groups.items() if k[-2:]==('none',0))
    evidence['residual_us']=max(evidence['residual_us'],max(abs(v-r['fixed_us']) for r in report.values() for v in r['window_medians_us']))
    HostOverheadProfile(result,expected_runtime=base['runtime_fingerprint'],expected_environment=base['environment'],explicit_costs=())
    return result,report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('samples',type=Path)
    p.add_argument('--base-profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    result,report=fit([json.loads(f.read_text()) for f in sorted(a.samples.glob('*-rank*.json'))],
        json.loads(a.base_profile.read_text(encoding='utf-8')))
    for path,data in ((a.output,result),(a.report,report)):
        with path.open('x',encoding='utf-8') as stream:json.dump(data,stream,indent=2,ensure_ascii=False)
    print(json.dumps(report))


if __name__=='__main__':main()
