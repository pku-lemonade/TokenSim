"""Measure unmodified native CPU codecs in independent processes, without GPUs.

This isolates codec service; it does not measure the full engine's host overhead.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


def measure(source, batch, window):
    # GPU execution is neither necessary nor allowed in this microbenchmark.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    native = runpy.run_path(str(source))
    dp = native['AFDDPMetadata'](torch.tensor([batch], dtype=torch.int32, device='cpu'))
    payload = native['AFDControlPayload']({0: dp}, False, False, False, False)
    assert dp.num_tokens_across_dp_cpu.device.type == 'cpu'
    assert dp.max_tokens_across_dp_cpu.device.type == 'cpu'
    encode, decode = native['encode_control_payload'], native['decode_control_payload']
    wire = encode(payload)
    assert len(wire) == 179 and encode(decode(wire)) == wire
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    functions = [('control_encode_us', 'control_encode', encode, payload),
                 ('control_decode_us', 'control_decode', decode, wire)]
    rows = []
    for name, coverage, function, argument in functions:
        for _ in range(200):
            function(argument)
        # Codecs do not consume context, but validate the supported decode range.
        for context in range(129, 160):
            for _ in range(4):
                start = time.perf_counter_ns()
                result = function(argument)
                end = time.perf_counter_ns()
                if name == 'control_encode_us':
                    assert result == wire
                else:
                    assert encode(result) == wire
                rows.append(dict(cost_field=name, coverage=[coverage], batch=batch,
                    context=context, window_id=f'cpu-codec-b{batch}-w{window}',
                    duration_us=(end-start)/1000, phase='decode',
                    timing_scope='host_service', contains_wait=False,
                    excludes_device_launch=True, profiler_enabled=False,
                    source=f'{source}:sha256={digest}:isolated_cpu_codec'))
    assert not torch.cuda.is_initialized()
    return dict(samples=rows, source_sha256=digest, wire_bytes=len(wire),
                measurement_scope='isolated native CPU codec; excludes caller wrapping and device readback',
                cuda_initialized=False, pid=os.getpid())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--batch', type=int, choices=[1, 6, 8])
    parser.add_argument('--window', type=int)
    args = parser.parse_args()
    if args.batch is not None:
        result = measure(args.source, args.batch, args.window)
        with args.output.open('x') as stream:
            json.dump(result, stream)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    for batch in (1, 8):
        for window in range(3):
            subprocess.run([sys.executable, __file__, str(args.source),
                str(args.output/f'b{batch}-w{window}.json'), '--batch', str(batch),
                '--window', str(window)], check=True,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
            print(f'CPU codec batch={batch} window={window} complete', flush=True)


if __name__ == '__main__':
    main()
