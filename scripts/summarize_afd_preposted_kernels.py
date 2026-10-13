"""Pair native NCCL device intervals by explicit exchange identity, excluding peer startup wait."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def paired_interval(sender,receiver):
    s0,s1=sender
    r0,r1=receiver
    started=max(s0,r0)
    if r1<started or s1<started:
        raise ValueError('cross-rank device clocks violate send/receive causality')
    return dict(device_service_us=max(s1,r1)-started,
                receiver_data_ready_us=r1-started,sender_peer_start_wait_us=started-s0,
                receiver_peer_start_wait_us=started-r0)


def extract(document,reference_base,expected_exchanges=840):
    offset=(document['baseTimeNanoseconds']-reference_base)/1000
    events=document['traceEvents']
    labels=sorted((e for e in events if e.get('cat')=='user_annotation'
                   and e.get('ph')=='X' and e.get('name','').startswith('exchange:')),key=lambda e:e['ts'])
    starts=[e['ts'] for e in labels]
    found=defaultdict(list)
    for e in events:
        if e.get('cat')!='kernel' or 'nccl' not in e.get('name','').lower(): continue
        idx=bisect_right(starts,e['ts'])-1
        if idx<0: continue
        label=labels[idx]
        if e['ts']+e['dur']>label['ts']+label['dur']+1:
            continue  # Unlabelled communicator warmup between workload sizes.
        found[label['name']].append((e['ts']+offset,e['ts']+offset+e['dur']))
    if len(labels)!=expected_exchanges or len(found)!=expected_exchanges or any(len(v)!=1 for v in found.values()):
        raise ValueError('every exchange must contain exactly one NCCL kernel')
    return {name:value[0] for name,value in found.items()}


def summarize(root,backend,window,validation_batch=None):
    paths=[root/f'{backend}-rank{rank}.trace.json' for rank in (0,1)]
    documents=[json.loads(p.read_text()) for p in paths]
    base=documents[0]['baseTimeNanoseconds']
    ranks=[extract(d,base,420 if validation_batch and backend=='pynccl' else 840) for d in documents]
    if set(ranks[0])!=set(ranks[1]): raise ValueError('rank exchange identities mismatch')
    groups=defaultdict(list)
    for identity in sorted(ranks[0]):
        _,size,sender,delay,iteration=identity.split(':')
        size,sender,delay,iteration=map(int,(size,sender,delay,iteration))
        if iteration<20: continue
        groups[(size,sender,delay)].append(paired_interval(ranks[sender][identity],ranks[1-sender][identity]))
    results=[]
    for (size,sender,delay),values in sorted(groups.items()):
        if len(values)!=50: raise ValueError('incomplete exchange samples')
        durations=sorted(v['device_service_us'] for v in values)
        results.append(dict(backend=backend,bytes=size,sender=sender,receiver_delay_ms=delay,window=window,
            samples=len(values),p50_device_service_us=statistics.median(durations),p95_device_service_us=durations[47],
            min_device_service_us=min(durations),max_device_service_us=max(durations),
            median_receiver_peer_start_wait_us=statistics.median(v['receiver_peer_start_wait_us'] for v in values)))
    return dict(scope='paired_NCCL_device_service_after_both_kernels_started_until_both_complete',
        profiler_enabled=True,full_alignment_passed=False,
        limitations=['excludes CPU launch and device queue before kernel start',
                     'CUPTI timestamp calibration across devices; not a physical wire-only measurement',
                     'profiler diagnostic must be compared with unprofiled exchange bounds; not auto-imported'],
        trace_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],results=results)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--backend',choices=['torch','pynccl'],required=True)
    p.add_argument('--window',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation-batch',type=int,choices=[6])
    a=p.parse_args()
    result=summarize(a.root,a.backend,a.window,a.validation_batch)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result['results']))


if __name__=='__main__': main()
