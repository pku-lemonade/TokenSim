"""Ready receiver exchange service bounds; CPU/GPU clock uncertainty is explicit.

Measures sender API entry to receiver device completion, including sender launch.
It is NOT pure wire time and must not be loaded as a transport-only profile.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import socket
import statistics
import time
from datetime import timedelta

from scripts.measure_afd_control_staging import UUIDS, check_idle


def worker(rank,root,backend,port,window,profile_enabled=False,validation_batch=None):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2,
                            timeout=timedelta(seconds=90))
    group=None
    comm=None
    if backend=='torch':
        group=dist.new_group([0,1],backend='nccl')
    else:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        comm=PyNcclCommunicator(dist.group.WORLD,device=rank)
        if comm.disabled: raise RuntimeError('PyNccl disabled')
    configs=[(8,'int64'),(179,'uint8')] if backend=='torch' else [(512,'bfloat16'),(4096,'bfloat16')]
    if validation_batch and backend=='pynccl':configs=[(validation_batch*512,'bfloat16')]
    stream=torch.cuda.current_stream()
    anchors=[torch.cuda.Event(enable_timing=True) for _ in range(5)]
    end=torch.cuda.Event(enable_timing=True)
    for event in [*anchors,end]:
        event.record(stream)
        event.synchronize()
    results=[]
    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) if profile_enabled else None
    if prof is not None: prof.start()
    try:
        for size,dtype_name in configs:
            dtype=getattr(torch,dtype_name)
            tensor=torch.full((size//torch.empty((),dtype=dtype).element_size(),),7,device=rank,dtype=dtype)
            # Initialize each NCCL peer before the ready handshake. Lazy communicator
            # creation can otherwise block recv enqueue while the sender waits for it.
            for warm_sender in (0,1):
                if comm is not None:
                    if rank==warm_sender: comm.send(tensor,1-rank,stream=torch.cuda.current_stream())
                    else: comm.recv(tensor,1-rank,stream=torch.cuda.current_stream())
                elif rank==warm_sender: dist.send(tensor,1-rank,group=group)
                else: dist.recv(tensor,1-rank,group=group)
                torch.cuda.synchronize()
                dist.barrier()
            for sender in (0,1):
                for delay_ms in (0,5,20):
                    rows=[]
                    for iteration in range(70):
                        if rank!=sender: tensor.zero_()
                        torch.cuda.synchronize()
                        dist.barrier()
                        # Anchor the local GPU clock to the shared host monotonic clock.
                        brackets=[]
                        for event in anchors:
                            lo=time.perf_counter_ns()
                            event.record(stream)
                            event.synchronize()
                            hi=time.perf_counter_ns()
                            brackets.append((hi-lo,lo,hi,event))
                        _,before,after,anchor=min(brackets,key=lambda b:b[0])
                        signal=torch.zeros(1,dtype=torch.int64)
                        entry=None
                        post_us=None
                        label=f'exchange:{size}:{sender}:{delay_ms}:{iteration}'
                        with torch.profiler.record_function(label) if profile_enabled else nullcontext():
                            if rank!=sender:
                                if delay_ms: time.sleep(delay_ms/1000)
                                posted=time.perf_counter_ns()
                                if comm is not None:
                                    comm.recv(tensor,sender,stream=torch.cuda.current_stream())
                                else:
                                    work=dist.irecv(tensor,sender,group=group)
                                    work.wait()  # Enqueue current-stream dependency, not data readiness.
                                end.record()
                                post_us=(time.perf_counter_ns()-posted)/1000
                                dist.send(signal,sender)  # CPU sidecar: recv is already enqueued.
                                end.synchronize()
                                assert bool(torch.all(tensor==7).item()), 'received data mismatch'
                            else:
                                ready=time.perf_counter_ns()
                                dist.recv(signal,1-rank)
                                entry=time.perf_counter_ns()
                                if comm is not None:
                                    comm.send(tensor,1-rank,stream=torch.cuda.current_stream())
                                else:
                                    dist.send(tensor,1-rank,group=group)
                                end.record()
                                end.synchronize()
                        local=dict(rank=rank,anchor_lo_ns=before,anchor_hi_ns=after,
                                   completion_delta_ns=anchor.elapsed_time(end)*1e6,
                                   sender_entry_ns=entry,post_api_us=post_us,
                                   ready_wait_us=(entry-ready)/1000 if rank==sender else None)
                        pair=[None,None]
                        dist.all_gather_object(pair,local)
                        if rank==sender and iteration>=20:
                            received=pair[1-sender]
                            lo=(received['anchor_lo_ns']+received['completion_delta_ns']-entry)/1000
                            hi=(received['anchor_hi_ns']+received['completion_delta_ns']-entry)/1000
                            rows.append(dict(service_lower_us=lo,service_upper_us=hi,
                                             clock_bound_us=hi-lo,ready_wait_us=local['ready_wait_us'],
                                             receiver_post_api_us=received['post_api_us']))
                    if rank==sender:
                        results.append(dict(backend=backend,bytes=size,dtype=dtype_name,sender=sender,
                            receiver_delay_ms=delay_ms,window=window,samples=rows,
                            scope='sender_api_entry_to_preposted_receiver_device_completion',
                            contains_sender_launch=True,profiler_enabled=profile_enabled,admissible_for_pure_transport_profile=False,
                            timing_method='GPU event mapped to shared host clock interval',
                            median_lower_us=statistics.median(r['service_lower_us'] for r in rows),
                            median_upper_us=statistics.median(r['service_upper_us'] for r in rows)))
        if prof is not None:
            prof.stop()
            prof.export_chrome_trace(str(Path(root)/f'{backend}-rank{rank}.trace.json'))
        with (Path(root)/f'{backend}-w{window}-rank{rank}.json').open('x') as stream:
            json.dump(results,stream,indent=2)
    finally:
        dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    p.add_argument('--backend',choices=['torch','pynccl'],required=True)
    p.add_argument('--window',type=int,required=True)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--port',type=int,default=18473)
    p.add_argument('--validation-batch',type=int,choices=[6])
    a=p.parse_args()
    check_idle()
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',a.port))==0: raise RuntimeError('experiment port occupied')
    if os.environ.get('TORCH_NCCL_BLOCKING_WAIT','0')!='0':
        raise RuntimeError('preposting requires nonblocking NCCL CPU wait semantics')
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(UUIDS)
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    import torch.multiprocessing as mp
    mp.spawn(worker,args=(str(a.output),a.backend,a.port,a.window,a.profile,a.validation_batch),nprocs=2,join=True)
    print(f'complete {a.backend} window {a.window}',flush=True)


if __name__=='__main__': main()
