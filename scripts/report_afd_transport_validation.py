"""Independent transport service comparison; inclusive exchange spans stay diagnostic."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from TokenSim.af.transport_profile import TransportServiceProfile


def main():
    p=argparse.ArgumentParser()
    for name in ('root','profile','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();d=json.loads(a.profile.read_text(encoding='utf-8'))
    digest=hashlib.sha256(a.profile.read_bytes()).hexdigest()
    for name in ('contract.json','completed.json'):
        c=json.loads((a.root/name).read_text())
        if c['profile_sha256']!=digest or c['used_for_fit'] is not False or c['batch']!=6:
            raise ValueError('independent transport validation contract mismatch')
    profile=TransportServiceProfile(d,environment=d['environment'],domains=d['domains'],device_indices=d['device_indices'])
    results=[];sources={};stability=defaultdict(list);outer=[]
    for backend in ('torch','pynccl'):
        for w in range(3):
            path=a.root/f'{backend}-w{w}-kernels.json';doc=json.loads(path.read_text())
            sources[str(path.relative_to(a.root))]=hashlib.sha256(path.read_bytes()).hexdigest()
            for row in doc['results']:
                if row['samples']!=50 or row['window']!=w:raise ValueError('incomplete device window')
                if backend=='torch' and row['sender']==1:continue
                size=row['bytes'];sender=row['sender']
                model=profile.estimate_us(source=2+sender,target=3-sender,
                    backend='torch_distributed_nccl' if backend=='torch' else backend,
                    dtype='bf16' if backend=='pynccl' else 'int64' if size==8 else 'uint8',size=size)
                native=row['p50_device_service_us']
                if native<=0:raise ValueError('invalid native device service')
                ratio=max(model,native)/min(model,native)
                results.append(dict(backend=backend,window=w,bytes=size,sender=sender,delay_ms=row['receiver_delay_ms'],
                    model_us=model,native_us=native,ratio=ratio,order_pass=ratio<10))
                stability[backend,size,sender,w].append(native)
            for mode in ('profiled','unprofiled'):
                for rank in (0,1):
                    path=a.root/f'{backend}-w{w}-{mode}'/f'{backend}-w{w}-rank{rank}.json'
                    sources[str(path.relative_to(a.root))]=hashlib.sha256(path.read_bytes()).hexdigest()
                    for row in json.loads(path.read_text()):
                        if row['admissible_for_pure_transport_profile'] is not False or row['profiler_enabled']!=(mode=='profiled'):
                            raise ValueError('inclusive timing must stay diagnostic')
                        outer.append(dict(backend=backend,mode=mode,window=w,bytes=row['bytes'],sender=row['sender'],
                            delay_ms=row['receiver_delay_ms'],median_lower_us=row['median_lower_us'],median_upper_us=row['median_upper_us']))
    if len(results)!=36:raise ValueError('all four native cases x three delays x three windows required')
    delay_ok=all(max(v)/min(v)<1.5 for v in stability.values())
    report=dict(profile_sha256=digest,results=results,outer_diagnostics=outer,source_sha256=sources,
        device_service_order_pass=all(r['order_pass'] for r in results),peer_delay_stable=delay_ok,
        scope='device transport after both kernels started; not physical wire-only or CPU-inclusive exchange',
        full_alignment_passed=False)
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(max_ratio=max(r['ratio'] for r in results),delay_stable=delay_ok)))


if __name__=='__main__':main()
