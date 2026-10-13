"""Derive a scoped candidate from audited paired GPU service, never outer wall time."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
from TokenSim.af.transport_profile import TransportServiceProfile
from TokenSim.hardware.domains import CommunicationDomainConfig,MODEL_VERSION


def fit(root,bundle):
    by_case=defaultdict(dict);sources={}
    for backend in ('pynccl','torch'):
        for w in range(3):
            path=root/f'{backend}-w{w}-kernels.json';d=json.loads(path.read_text(encoding='utf-8'))
            sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            if d['scope']!='paired_NCCL_device_service_after_both_kernels_started_until_both_complete' or not d['profiler_enabled']:
                raise ValueError('paired device evidence required; CPU inclusive bounds forbidden')
            if len(d['trace_sha256'])!=2:raise ValueError('both rank traces required')
            for row in d['results']:
                key=(backend,row['bytes'],row['sender'],row['receiver_delay_ms'])
                if row['window']!=w or row['samples']!=50 or w in by_case[key]:raise ValueError('window mismatch')
                by_case[key][w]=row['p50_device_service_us']
    points=defaultdict(list)
    for backend,sizes in [('pynccl',(512,4096)),('torch',(8,179))]:
        for sender in (0,1) if backend=='pynccl' else (0,):
            for size in sizes:
                medians=[]
                for delay in (0,5,20):
                    rows=by_case[backend,size,sender,delay]
                    if set(rows)!={0,1,2}:raise ValueError('three windows required')
                    medians.append(statistics.mean(rows.values()))
                if min(medians)<=0 or max(medians)/min(medians)>1.5:
                    raise ValueError('peer delay changes measured device service; refusing fit')
                dtype='bf16' if backend=='pynccl' else 'int64' if size==8 else 'uint8'
                points[backend,sender,dtype].append(dict(bytes=size,service_us=medians[0]))
    domains=asdict(CommunicationDomainConfig(**bundle['communication_domains']))
    document=dict(schema_version=1,profile_id='pkua100-paired-device-transport-20261012-candidate',
        timing_scope='transport_service',unit='us',contains_wait=False,contains_cpu_submit=False,
        model_version=MODEL_VERSION,environment=bundle['host_profile']['environment'],
        domains=domains,device_indices={'a-pool:0':2,'f-pool:0':3},
        records=[dict(source_device=2+sender,target_device=3-sender,
            backend='torch_distributed_nccl' if backend=='torch' else backend,dtype=dtype,
            points=values,interpolation='linear_within_training_range') for (backend,sender,dtype),values in points.items()],
        evidence=dict(kind='measured',source=sources,window_count=3,sample_count=900,
            training_batches=[1,8],heldout_batches=[6],validation_status='pending',
            measurement_boundary='after_both_device_kernels_started_until_both_complete',
            peer_delay_audit_passed=True,method='mean of three window medians; zero-delay only; bounded linear bytes interpolation',
            limitations=['CUPTI device timestamps, not physical wire-only latency',
                'cross-device clock calibration and profiler perturbation remain measurement uncertainty',
                'excludes CPU launch and device scheduling before both kernels start',
                'historical raw diagnostic labels preserved; independent validation pending']))
    TransportServiceProfile(document,environment=document['environment'],domains=domains,device_indices=document['device_indices'])
    return document


def main():
    p=argparse.ArgumentParser()
    for name in ('root','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();d=fit(a.root,json.loads(a.bundle.read_text(encoding='utf-8')))
    with a.output.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2)


if __name__=='__main__':main()
