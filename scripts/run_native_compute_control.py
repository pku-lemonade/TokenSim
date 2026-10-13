"""Run real, single-GPU Qwen vLLM as a no-AF-communication compute control.

Run only on an explicitly selected idle GPU. Profiles remain on the server.
This measures kernels, not Python overhead or end-to-end latency calibration.
"""
import argparse
import functools
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    if args.gpu in apps:
        raise RuntimeError('Selected GPU is occupied; refusing to start')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    import torch
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models import qwen3_moe as native

    def instrument(cls, label):
        original = cls.forward
        @functools.wraps(original)
        def annotated(self, *a, **kw):
            with torch.profiler.record_function('compute_control::' + label):
                return original(self, *a, **kw)
        cls.forward = annotated

    instrument(native.Qwen3MoeAttention, 'attention')
    instrument(native.Qwen3MoeSparseMoeBlock, 'moe')
    llm = LLM(model=args.model, dtype='bfloat16', tensor_parallel_size=1,
              data_parallel_size=1, enable_expert_parallel=True, enforce_eager=True,
              max_model_len=256, max_num_seqs=32, max_num_batched_tokens=4096,
              gpu_memory_utilization=.10, seed=42, enable_prefix_caching=False,
              async_scheduling=False, skip_tokenizer_init=True)
    original_step = llm.llm_engine.step
    counter = [0]
    def step():
        label = 'prefill' if counter[0] == 0 else 'decode'
        with torch.profiler.record_function(f'compute_step::{label}::{counter[0]}'):
            result = original_step()
        counter[0] += 1
        return result
    llm.llm_engine.step = step
    params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True, seed=42, detokenize=False)
    outputs = {}
    try:
        for batch in (1, 8, 4):
            prompts = [{'prompt_token_ids': [4+(j+i)%100 for j in range(128)]} for i in range(batch)]
            for _ in range(5):
                counter[0] = 0
                llm.generate(prompts, params, use_tqdm=False)
            for repeat in range(3):
                counter[0] = 0
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                     torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                    generated = llm.generate(prompts, params, use_tqdm=False)
                    torch.cuda.synchronize()
                assert counter[0] == 32, counter
                prof.export_chrome_trace(str(args.output / f'b{batch}-r{repeat}.json'))
                ids = [list(r.outputs[0].token_ids) for r in generated]
                outputs[f'b{batch}-r{repeat}'] = ids
                print(f'PROFILE batch={batch} repeat={repeat}', flush=True)
        (args.output / 'outputs.json').write_text(json.dumps(outputs))
        (args.output / 'manifest.json').write_text(json.dumps(dict(gpu=args.gpu, model=args.model,
            batches=[1,8,4], repeats=3, prefill='excluded', timing='profiled GPU kernel sums',
            mode='real native vLLM TP=DP=EP=1, no AFD connector; no parameter fitting'), indent=2))
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
