"""Native NCCL CPU API return, separate from device completion and peer delay."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import time
from datetime import timedelta

from scripts.measure_afd_control_staging import UUIDS, check_idle


def worker(rank, root, backend, port, window, batches=(1,8)):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2,
        timeout=timedelta(seconds=90))
    group = dist.new_group([0,1],backend='nccl') if backend=='torch' else None
    comm = None
    if backend=='pynccl':
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        comm = PyNcclCommunicator(dist.group.WORLD,device=rank)
        if comm.disabled: raise RuntimeError('PyNccl disabled')
    stream = torch.cuda.current_stream()
    funcs = (comm.send,comm.recv) if comm else (dist.send,dist.recv)
    sources = {name:dict(file=inspect.getsourcefile(inspect.unwrap(fn)),
        sha256=hashlib.sha256(inspect.getsource(inspect.unwrap(fn)).encode()).hexdigest())
        for name,fn in zip(('send','receive'),funcs)}

    def call(tensor,sender):
        if comm:
            if rank==sender: comm.send(tensor,1-rank,stream=stream)
            else: comm.recv(tensor,1-rank,stream=stream)
        elif rank==sender: dist.send(tensor,1-rank,group=group)
        else: dist.recv(tensor,1-rank,group=group)

    reports=[]
    try:
        for batch in batches:
            specs = [(8,'int64'),(179,'uint8')] if backend=='torch' else [(batch*256*2,'bfloat16')]
            for size,dtype_name in specs:
                dtype=getattr(torch,dtype_name)
                tensor=torch.full((size//torch.empty((),dtype=dtype).element_size(),),7,dtype=dtype,device=rank)
                for sender in ((0,) if backend=='torch' else (0,1)):
                    for _ in range(10):
                        call(tensor,sender)
                        torch.cuda.synchronize();dist.barrier()
                    for delayed_role,delay_ms in [('none',0),('sender',5),('sender',20),('receiver',5),('receiver',20)]:
                        samples=[]
                        for i in range(60):
                            if rank!=sender:tensor.zero_()
                            torch.cuda.synchronize()
                            dist.barrier()
                            role='sender' if rank==sender else 'receiver'
                            if role==delayed_role:time.sleep(delay_ms/1000)
                            start=time.perf_counter_ns()
                            call(tensor,sender)
                            end=time.perf_counter_ns()
                            # Device/peer completion never enters CPU API interval.
                            torch.cuda.synchronize()
                            if rank!=sender:assert bool(torch.all(tensor==7).item())
                            pair=[None,None]
                            dist.all_gather_object(pair,dict(rank=rank,start_ns=start,end_ns=end))
                            if i>=10:
                                peer=pair[1-rank]
                                samples.append(dict(service_us=(end-start)/1000,
                                    returned_before_peer_entered=end<peer['start_ns']))
                        reports.append(dict(backend=backend,batch=batch,bytes=size,dtype=dtype_name,
                            direction='a_to_f' if sender==0 else 'f_to_a',action='send' if rank==sender else 'receive',
                            delayed_role=delayed_role,delay_ms=delay_ms,window=window,samples=samples))
        with (Path(root)/f'rank{rank}.json').open('x') as f:
            json.dump(dict(rows=reports,sources=sources,gpu_uuids=UUIDS,
                timing_scope='cpu_api_entry_to_return',completion_sync_outside_timer=True,
                prior_device_work_synchronized=True,blocking_wait=False,profiler_enabled=False),f)
    finally:
        dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['torch','pynccl'],required=True)
    p.add_argument('--window',type=int,required=True)
    p.add_argument('--port',type=int,default=18479)
    p.add_argument('--validation-batch',type=int,choices=[6])
    a=p.parse_args()
    check_idle()
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',a.port))==0:raise RuntimeError('port occupied')
    if os.environ.get('TORCH_NCCL_BLOCKING_WAIT','0')!='0':raise RuntimeError('blocking wait unsupported')
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(UUIDS)
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    import torch.multiprocessing as mp
    batches=(a.validation_batch,) if a.validation_batch else (1,8)
    mp.spawn(worker,args=(str(a.output),a.backend,a.port,a.window,batches),nprocs=2,join=True)
    print(f'complete {a.backend} w{a.window}',flush=True)


if __name__=='__main__':main()
