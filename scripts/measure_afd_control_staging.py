"""Isolate ready-input native control staging/readback; never fit mixed spans."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

UUIDS = ('GPU-8df5e331-8f9a-3cdc-af30-42b94f908018',
         'GPU-76ca30e6-17c9-9a38-cb06-57aebeaa6973')


def check_idle():
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu',
                                         '--format=csv,noheader,nounits'], text=True)
    values = {p[0].strip(): (float(p[1]), float(p[2]))
              for p in (line.split(',') for line in inventory.splitlines())}
    if any(u in apps or values[u][0] > 100 or values[u][1] > 5 for u in UUIDS):
        raise RuntimeError('Target GPUs occupied; refusing to start')


def window(output, index, batches=(1,8)):
    import torch
    rows = []
    for rank, role in ((0, 'a'), (1, 'f')):
        torch.cuda.set_device(rank)
        for batch in batches:
            body = json.dumps(dict(dp_metadata_list={'0': dict(num_tokens_across_dp_cpu=[batch],
                max_tokens_across_dp_cpu=batch)}, is_graph_capturing=False, is_warmup=False,
                is_graph_replaying=False, is_profile=False), separators=(',', ':'), sort_keys=True).encode()
            assert len(body) == 179
            length = torch.tensor([len(body)], dtype=torch.long, device=rank)
            payload = torch.tensor(list(body), dtype=torch.uint8, device=rank)

            def stage():
                cpu = torch.frombuffer(bytearray(body), dtype=torch.uint8)
                gpu = cpu.to(rank)
                size = torch.tensor([gpu.numel()], dtype=torch.long, device=rank)
                return gpu, size

            operations = [('h2d', stage)] if role == 'a' else [
                ('length_readback', lambda: int(length.item())),
                ('payload_readback', lambda: payload.cpu().numpy().tobytes())]
            for name, operation in operations:
                wall, cpu_times = [], []
                for iteration in range(220):
                    # Exclude all previously submitted GPU work from this span.
                    torch.cuda.synchronize()
                    cpu_start = time.thread_time_ns()
                    start = time.perf_counter_ns()
                    result = operation()
                    stop = time.perf_counter_ns()
                    cpu_end = time.thread_time_ns()
                    if name == 'length_readback':
                        assert result == 179
                    elif name == 'payload_readback':
                        assert result == body
                    if iteration >= 20:
                        wall.append((stop-start)/1000)
                        cpu_times.append((cpu_end-cpu_start)/1000)
                rows.append(dict(window=index, batch=batch, role=role, gpu_uuid=UUIDS[rank],
                    operation=name, samples=len(wall), wall_us=wall, thread_cpu_us=cpu_times,
                    median_wall_us=statistics.median(wall), median_thread_cpu_us=statistics.median(cpu_times),
                    timing_scope='ready_input_cpu_api_interval_including_own_device_work_and_runtime',
                    prior_gpu_work_synchronized=True, profiler_enabled=False,
                    admissible_for_host_profile=False, admissible_for_device_profile=False))
    with output.open('x') as stream:
        json.dump(rows, stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--window', type=int)
    parser.add_argument('--validation-batch',type=int,choices=[6])
    args = parser.parse_args()
    if args.window is not None:
        window(args.output, args.window, (args.validation_batch,) if args.validation_batch else (1,8))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    for index in range(3):
        # Utilization is sampled over a recent interval and can outlive our
        # previous process. Retry the same full guard; never bypass it.
        for attempt in range(6):
            try:
                check_idle()
                break
            except RuntimeError:
                if attempt == 5:
                    raise
                time.sleep(1)
        subprocess.run([sys.executable, __file__, str(args.output/f'window-{index}.json'),
                        '--window', str(index),
                        *(['--validation-batch',str(args.validation_batch)] if args.validation_batch else [])], check=True,
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(UUIDS)})
        print(f'Control staging window {index} complete', flush=True)


if __name__ == '__main__':
    main()
