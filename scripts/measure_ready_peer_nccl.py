"""Ready-peer NCCL microbenchmark; this interval still includes launch overhead."""
import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path


def message_specs(backend):
    return [(8, "int64"), (179, "uint8"), (181, "uint8")] if backend == "torch" else [(size, "bfloat16") for size in (512, 1024, 4096, 8192)]


def worker(rank, root, backend, port, delays):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo' if backend == 'pynccl' else 'nccl',
                            init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=2, timeout=timedelta(seconds=60))
    communicator = None
    if backend == 'pynccl':
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        communicator = PyNcclCommunicator(dist.group.WORLD, device=rank)
        if communicator.disabled:
            raise RuntimeError('PyNcclCommunicator disabled')
    rows = []
    try:
        for size, dtype_name in message_specs(backend):
            dtype = getattr(torch, dtype_name)
            element_size = torch.empty((), dtype=dtype).element_size()
            tensor = torch.ones(size // element_size, device=rank, dtype=dtype)
            for delay_ms in delays:
                for sender in (0, 1):
                    measurements = []
                    for iteration in range(120):
                        dist.barrier()
                        torch.cuda.synchronize()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        if rank != sender and delay_ms:
                            time.sleep(delay_ms / 1000)
                        start.record()
                        if communicator is not None:
                            if rank == sender:
                                communicator.send(tensor, 1-rank, stream=torch.cuda.current_stream())
                            else:
                                communicator.recv(tensor, 1-rank, stream=torch.cuda.current_stream())
                        elif rank == sender:
                            dist.send(tensor, 1-rank)
                        else:
                            dist.recv(tensor, 1-rank)
                        end.record()
                        end.synchronize()
                        if iteration >= 20:
                            measurements.append(start.elapsed_time(end) * 1000)
                    rows.append(dict(bytes=size, dtype=dtype_name, backend=backend, sender=sender, rank=rank,
                                     receiver_delay_ms=delay_ms, samples=len(measurements),
                                     timing_scope="local_cuda_event_interval_including_launch_and_possible_peer_wait",
                                     admissible_for_transport_profile=False,
                                     median_us=statistics.median(measurements),
                                     min_us=min(measurements), max_us=max(measurements)))
        (Path(root) / f'comm-{backend}-rank{rank}.json').write_text(json.dumps(rows, indent=2))
    finally:
        dist.destroy_process_group()


def main():
    import subprocess
    import torch.multiprocessing as mp
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--backend', choices=['torch', 'pynccl'], default='pynccl')
    parser.add_argument('--port', type=int, default=17649)
    parser.add_argument('--receiver-delay-ms', type=float, nargs='+', default=[0,5,20])
    args = parser.parse_args()
    import math
    if not 1024 <= args.port <= 65535 or any(not math.isfinite(d) or d < 0 for d in args.receiver_delay_ms):
        parser.error('invalid port or receiver delay')
    uuids = ['GPU-8df5e331-8f9a-3cdc-af30-42b94f908018',
             'GPU-76ca30e6-17c9-9a38-cb06-57aebeaa6973']
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    if any(u in apps for u in uuids):
        raise RuntimeError('Target GPU occupied; refusing to interfere')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    selected = {row.split(',')[0].strip():row.split(',')[1:] for row in inventory.splitlines()}
    for uuid in uuids:
        memory, utilization = map(lambda value: float(value.strip()), selected[uuid])
        if memory > 100 or utilization > 5:
            raise RuntimeError('Target GPU not idle')
    import socket
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', args.port)) == 0:
            raise RuntimeError('Experiment port is occupied')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(uuids)
    mp.spawn(worker, args=(str(args.output), args.backend, args.port, args.receiver_delay_ms), nprocs=2, join=True)
    print('Ready-peer NCCL complete; includes host launch skew, not pure wire time.')


if __name__ == '__main__':
    main()
