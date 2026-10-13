"""Guarded GPU2/3 PyNccl service and delayed-receiver measurements.

Ready mode preposts recv; a sender stream gate lets Python queue both CUDA
events and send before timing begins. Receiver occupancy is NOT service time.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from datetime import timedelta
from pathlib import Path

UUIDS = ['GPU-8df5e331-8f9a-3cdc-af30-42b94f908018', 'GPU-76ca30e6-17c9-9a38-cb06-57aebeaa6973']


def worker(rank, output, port, ack_mode):
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    comm = PyNcclCommunicator(dist.group.WORLD, device=rank)
    assert not comm.disabled
    stream = torch.cuda.current_stream()
    rows = []
    ack = torch.ones(16, dtype=torch.bfloat16, device=rank)
    def send_data(tensor, peer):
        comm.send(tensor, peer, stream=stream)
        if ack_mode:
            comm.recv(ack, peer, stream=stream)
    def recv_data(tensor, peer):
        comm.recv(tensor, peer, stream=stream)
        if ack_mode:
            comm.send(ack, peer, stream=stream)
    # Warm up both directions before using recv-preposted CPU handshakes.
    warm = torch.ones(256, dtype=torch.bfloat16, device=rank)
    for sender in (0, 1):
        (comm.send if rank == sender else comm.recv)(warm, 1-rank, stream=stream)
        torch.cuda.synchronize()
        dist.barrier()
    sizes = (32, 256, 512, 2048, 4096, 8192, 65536, 1048576, 16777216)
    cases = [('ready', size, 0, 5_000_000) for size in sizes]
    cases += [('gate_check', 4096, 0, 20_000_000)]
    cases += [('delayed', 4096, delay, 0) for delay in (0, .005, .020)]
    for mode, size, delay, gate in cases:
        for sender in (0, 1):
            tensor = torch.full((size//2,), 7 if rank == sender else 0, dtype=torch.bfloat16, device=rank)
            for window in range(3):
                samples, host_calls = [], []
                for iteration in range(40):
                    torch.cuda.synchronize()
                    dist.barrier()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    if mode != 'delayed':
                        if rank != sender:
                            start.record(stream)
                            t = time.perf_counter_ns()
                            recv_data(tensor, sender)
                            host_us = (time.perf_counter_ns()-t)/1000
                            end.record(stream)
                        dist.barrier()  # recv has been submitted before sender proceeds
                        if rank == sender:
                            torch.cuda._sleep(gate)
                            start.record(stream)
                            t = time.perf_counter_ns()
                            send_data(tensor, 1-rank)
                            host_us = (time.perf_counter_ns()-t)/1000
                            end.record(stream)
                    else:
                        if rank == sender:
                            start.record(stream)
                            t = time.perf_counter_ns()
                            send_data(tensor, 1-rank)
                            host_us = (time.perf_counter_ns()-t)/1000
                            end.record(stream)
                        dist.barrier()  # send submitted, then delay receiver
                        if rank != sender:
                            time.sleep(delay)
                            start.record(stream)
                            t = time.perf_counter_ns()
                            recv_data(tensor, sender)
                            host_us = (time.perf_counter_ns()-t)/1000
                            end.record(stream)
                    end.synchronize()
                    if iteration >= 10:
                        samples.append(start.elapsed_time(end)*1000)
                        host_calls.append(host_us)
                assert bool(torch.all(tensor == 7).item())
                assert bool(torch.all(ack == 1).item())
                row = dict(mode=mode, bytes=size, sender=sender, rank=rank, window=window,
                           receiver_delay_ms=delay*1000, gate_cycles=gate, n=len(samples),
                           median_us=statistics.median(samples), min_us=min(samples), max_us=max(samples),
                           host_api_median_us=statistics.median(host_calls), samples_us=samples)
                rows.append(row)
                if rank == sender:
                    print(json.dumps({k:v for k,v in row.items() if k != 'samples_us'}), flush=True)
                (Path(output)/f'rank{rank}.json').write_text(json.dumps(rows, indent=2))
    dist.barrier()
    comm.destroy()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    p.add_argument('--port', type=int, default=17849)
    p.add_argument('--ack', action='store_true', help='Time send through receive of a 32-byte acknowledgement')
    args = p.parse_args()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if any(u in apps for u in UUIDS):
        raise RuntimeError('GPU2/3 occupied: refusing to start')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(UUIDS)
    import torch
    import torch.multiprocessing as mp
    manifest = dict(uuids=UUIDS, torch=torch.__version__, scope='PyNccl BF16 P2P; no plugin metadata protocol',
                    ack_bytes=32 if args.ack else 0,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    topology=subprocess.check_output(['nvidia-smi','topo','-m'], text=True),
                    gate='sender CUDA sleep before timing events; check gate duration sensitivity',
                    ready_scope='sender device event interval after recv prepost; receiver includes wait',
                    delayed_scope='sender includes CPU handshake and receiver delay; never fit as service')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    mp.spawn(worker, args=(str(args.output), args.port, args.ack), nprocs=2, join=True)
    (args.output/'COMPLETE').write_text('all windows and payload checks passed\n')


if __name__ == '__main__':
    main()
