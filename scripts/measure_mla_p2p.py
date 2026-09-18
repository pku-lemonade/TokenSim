"""Two-GPU P2P microbenchmarks for payloads extracted from TokenSim's DAG.

This is communication calibration, not a native transformer-layer driver.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', nargs=2, required=True)
    parser.add_argument('--window-seconds', type=float, default=2.)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpus)
    os.environ['OMP_NUM_THREADS'] = '1'
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    (root/'traces').mkdir(exist_ok=True)
    import fcntl
    lock = (root/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def exclusive():
        for uuid in args.gpus:
            raw = subprocess.check_output(['nvidia-smi', '-i', uuid,
                '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
            others = [s for s in raw.splitlines() if s.strip() and int(s.split(',')[0]) != os.getpid()]
            if others:
                raise RuntimeError('Another process on selected GPU: ' + str(others))

    exclusive()
    import torch
    torch.set_num_threads(1)
    assert torch.cuda.device_count() == 2
    assert torch.cuda.can_device_access_peer(0, 1) and torch.cuda.can_device_access_peer(1, 0)
    sizes = sorted({s['payload_bytes'] for s in json.loads((root.parent/'communication-shapes.json').read_text())})
    contract = dict(gpu_uuids=args.gpus, torch=torch.__version__, cuda=torch.version.cuda,
        implementation='torch.Tensor.copy_ with non_blocking=True, independent source/destination streams per direction',
        timing='Same-device CUDA events per direction over 100-copy chunks; includes eager host gaps. Host wall includes both-device completion.',
        duplex='Disjoint source and destination tensors; both directions enqueued before synchronization',
        warmup=20, windows=3, window_seconds=args.window_seconds,
        boundary='Preallocated device buffers; no NCCL, no model, no CPU tensor staging requested',
        topology=subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True))
    dump(root/'contract.json', contract)
    results = []
    for size in sizes:
        for mode in ('a_to_f', 'f_to_a', 'bidirectional'):
            exclusive()
            pairs = [(0, 1)] if mode == 'a_to_f' else [(1, 0)] if mode == 'f_to_a' else [(0, 1), (1, 0)]
            legs = []
            for source, dest in pairs:
                source_tensor = torch.full((size,), 37 + source, device=f'cuda:{source}', dtype=torch.uint8)
                dest_tensor = torch.zeros((size,), device=f'cuda:{dest}', dtype=torch.uint8)
                legs.append((source_tensor, dest_tensor, torch.cuda.Stream(source), torch.cuda.Stream(dest)))
            for device in (0, 1):
                torch.cuda.synchronize(device)

            def copy():
                for source_tensor, dest_tensor, source_stream, dest_stream in legs:
                    with torch.cuda.stream(source_stream), torch.cuda.stream(dest_stream):
                        dest_tensor.copy_(source_tensor, non_blocking=True)

            copy()
            for device in (0, 1):
                torch.cuda.synchronize(device)
            for source_tensor, dest_tensor, _, _ in legs:
                assert torch.equal(source_tensor.cpu(), dest_tensor.cpu())
            for _ in range(20):
                copy()
            for device in (0, 1):
                torch.cuda.synchronize(device)
            blocks, windows = [], []
            for window in range(3):
                exclusive()
                begun, calls = time.perf_counter(), 0
                while calls < 100 or time.perf_counter() - begun < args.window_seconds:
                    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in legs]
                    wall = time.perf_counter()
                    for leg, (start, _) in zip(legs, events):
                        start.record(leg[2])
                    for _ in range(100):
                        copy()
                    for leg, (_, end) in zip(legs, events):
                        end.record(leg[2])
                    for _, end in events:
                        end.synchronize()
                    elapsed = (time.perf_counter()-wall)*1e6/100
                    times = [start.elapsed_time(end)*1000/100 for start, end in events]
                    blocks.append(dict(window=window, calls=100, device_intervals_us=times,
                        device_interval_us=max(times), eager_wall_us=elapsed))
                    calls += 100
                current = [r for r in blocks if r['window'] == window]
                windows.append(dict(calls=calls, wall_seconds=time.perf_counter()-begun,
                    device_interval_us=statistics.mean(r['device_interval_us'] for r in current),
                    eager_wall_us=statistics.mean(r['eager_wall_us'] for r in current)))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function(f'{mode}:{size}'):
                    for _ in range(3):
                        copy()
                    for device in (0, 1):
                        torch.cuda.synchronize(device)
            trace = root/'traces'/f'{mode}-{size}.trace.json'
            prof.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())['traceEvents']
            copies = [e for e in events if e.get('cat') == 'gpu_memcpy']
            if not copies:
                raise RuntimeError('No GPU memcpy activities in profiling trace')
            if any('DtoH' in e.get('name', '') or 'HtoD' in e.get('name', '') for e in copies):
                raise RuntimeError('Unexpected host staging in measured transfer')
            vals = sorted(r['device_interval_us'] for r in blocks)
            results.append(dict(mode=mode, payload_bytes=size, status='ok', correctness=True,
                device_interval_us=statistics.median(r['device_interval_us'] for r in windows),
                eager_wall_us=statistics.median(r['eager_wall_us'] for r in windows),
                chunk_mean_p50_us=statistics.median(vals),
                chunk_mean_p95_us=vals[min(len(vals)-1, math.ceil(.95*len(vals))-1)],
                windows=windows, blocks=blocks, trace=str(trace.relative_to(root)),
                gpu_memcpy_events=len(copies), memcpy_names=sorted({e['name'] for e in copies})))
            dump(root/'measurements.json', results)
            dump(root/'progress.json', dict(completed=len(results), planned=len(sizes)*3,
                mode=mode, payload_bytes=size, updated=time.strftime('%Y-%m-%d %H:%M:%S %z')))
            print(mode, size, results[-1]['device_interval_us'], flush=True)
            del legs, source_tensor, dest_tensor
            for device in (0, 1):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
    dump(root/'done.json', dict(completed=len(results), status='ok'))


if __name__ == '__main__':
    main()
