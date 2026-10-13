"""Partition one native A-step wall clock; nested calls are never summed twice."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


PRIORITY={'execute_model':1,'_model_forward':2,'_prepare_inputs':3,
          '_build_attention_metadata':3,'control_payload':3,'sample_tokens':3,
          'decoder_layer_0':3,'decoder_layer_1':3}


def partition(step):
    lo,hi=step['start_ns'],step['end_ns']
    rows=[]
    layer=0
    for event in sorted(step['events'],key=lambda e:e['start_ns']):
        name=event['name']
        if name=='decoder_layer':
            name=f'decoder_layer_{layer}';layer+=1
        assert lo<=event['start_ns']<=event['end_ns']<=hi
        if name in PRIORITY:
            rows.append((event['start_ns'],event['end_ns'],name))
    assert layer==2
    boundaries=sorted({lo,hi,*[x for a,b,_ in rows for x in (a,b)]})
    parts=defaultdict(float)
    for a,b in zip(boundaries,boundaries[1:]):
        active=[name for left,right,name in rows if left<=a and right>=b]
        name=max(active,key=lambda n:PRIORITY[n]) if active else 'other_engine'
        parts[name]+=(b-a)/1000
    assert abs(sum(parts.values())-(hi-lo)/1000)<1e-6
    return dict(parts)


def summarize(path):
    data=json.loads(path.read_text())
    partitions=[partition(s) for s in data['steps']]
    names=set().union(*(p.keys() for p in partitions))
    calls=defaultdict(list)
    counts=defaultdict(list)
    for step in data['steps']:
        durations=defaultdict(float);nums=defaultdict(int)
        for e in step['events']:
            durations[e['name']]+=(e['end_ns']-e['start_ns'])/1000
            nums[e['name']]+=1
        for name,value in durations.items():calls[name].append(value);counts[name].append(nums[name])
    values=[(s['end_ns']-s['start_ns'])/1000 for s in data['steps']]
    return dict(samples=len(values),mean_step_us=statistics.mean(values),p50_step_us=statistics.median(values),
        partition_mean_us={n:statistics.mean(p.get(n,0) for p in partitions) for n in sorted(names)},
        inclusive_calls={n:dict(mean_us=statistics.mean(v),p50_us=statistics.median(v),
            p95_us=sorted(v)[int(.95*len(v))-1],calls_per_step=statistics.mean(counts[n])) for n,v in calls.items()},
        sources=data['sources'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope='native wall partition; includes CPU execution, device/API blocking and waiting, not fixed-cost calibration')


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    a=p.parse_args()
    results=[]
    for batch in (1,8):
        path=a.root/f'b{batch}-phases/c{batch}/phase-samples.json'
        result=dict(batch=batch,**summarize(path));results.append(result)
        print(json.dumps(result))
    with (a.root/'phase-summary.json').open('x') as f:json.dump(results,f,indent=2)


if __name__=='__main__':main()
