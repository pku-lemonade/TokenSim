"""Collect a short annotated vLLM forward trace using PyTorch Profiler."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time


def module_annotation(name):
    if name in ("model.norm", "lm_head"):
        return "tokensim.model." + ("final_norm" if name == "model.norm" else "lm_head")
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.(.*))?$", name)
    if not match:
        return "tokensim.forward" if name == "" else None
    layer, suffix = match.groups()
    operations = {
        "self_attn.qkv_proj": "qkv",
        "self_attn.q_proj": "q_proj",
        "self_attn.k_proj": "k_proj",
        "self_attn.v_proj": "v_proj",
        "self_attn.o_proj": "output",
        "self_attn.q_norm": "q_norm",
        "self_attn.k_norm": "k_norm",
        "self_attn.attn": "attention",
        "input_layernorm": "attention_norm",
        "post_attention_layernorm": "ffn_norm",
        "mlp.gate_up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
        "mlp.act_fn": "activation",
        "mlp.gate": "router",
        "mlp.experts": "moe_module",
    }
    if suffix is None:
        return f"tokensim.layer.{layer}"
    operation = operations.get(suffix)
    return f"tokensim.layer.{layer}.{operation}" if operation else None


def annotate_model(model):
    """Install observation-only ranges in the worker's loaded model."""
    import torch

    annotated = []
    for name, module in model.named_modules():
        annotation = module_annotation(name)
        if annotation is None:
            continue
        stack = []

        def before(module, args, *, label=annotation, contexts=stack):
            context = torch.profiler.record_function(label)
            context.__enter__()
            contexts.append(context)

        def after(module, args, result, *, contexts=stack):
            if contexts:
                contexts.pop().__exit__(None, None, None)

        module.register_forward_pre_hook(before)
        module.register_forward_hook(after, always_call=True)
        annotated.append(annotation)
    return annotated


class ProfilerWorkerExtension:
    def tokensim_install_annotations(self):
        return annotate_model(self.get_model())


def collect(args):
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(vllm.__file__).resolve().parent.parent
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    manifest = dict(
        schema_version=1,
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        vllm=importlib.metadata.version("vllm"),
        vllm_revision=revision,
        vllm_file=str(Path(vllm.__file__).resolve()),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        devices=[
            dict(
                name=torch.cuda.get_device_name(index),
                total_memory=torch.cuda.get_device_properties(index).total_memory,
            )
            for index in range(torch.cuda.device_count())
        ],
        capture_policy="cuda_graph" if args.cuda_graph else "eager",
        annotations="module forward hooks; CPU range overhead is included in this profile",
        status="initializing",
    )
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            max_model_len=max(128, args.input_tokens + args.output_tokens + 16),
            max_num_seqs=args.batch_size,
            max_num_batched_tokens=max(128, args.batch_size * args.input_tokens),
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=not args.cuda_graph,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            skip_tokenizer_init=True,
            seed=0,
            profiler_config={
                "profiler": "torch",
                "torch_profiler_dir": str(output / "raw"),
                "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": True,
                "torch_profiler_use_gzip": True,
            },
            worker_extension_cls=f"{Path(__file__).stem}.ProfilerWorkerExtension",
        )
        manifest["annotated_modules_per_rank"] = llm.collective_rpc(
            "tokensim_install_annotations"
        )
        config = llm.llm_engine.vllm_config.model_config.hf_config
        manifest["model_config"] = config.to_dict()
        manifest["model_revision"] = getattr(config, "_commit_hash", None)
        prompts = [
            {"prompt_token_ids": [42] * args.input_tokens}
            for _ in range(args.batch_size)
        ]
        sampling = SamplingParams(
            temperature=0,
            max_tokens=args.output_tokens,
            min_tokens=args.output_tokens,
            ignore_eos=True,
        )
        for _ in range(args.warmup):
            llm.generate(prompts, sampling, use_tqdm=False)
        llm.start_profile("tokensim")
        start = time.perf_counter()
        results = llm.generate(prompts, sampling, use_tqdm=False)
        llm.stop_profile()
        manifest["profile_wall_seconds"] = time.perf_counter() - start
        manifest["generated_tokens"] = [
            len(result.outputs[0].token_ids) for result in results
        ]
        manifest["trace_files"] = [
            str(p.relative_to(output)) for p in (output / "raw").rglob("*.json*")
        ]
        if not manifest["trace_files"]:
            raise RuntimeError("profiler produced no Chrome trace files")
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "tensor_parallel_size",
        "batch_size",
        "input_tokens",
        "output_tokens",
        "warmup",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")
    collect(args)


if __name__ == "__main__":
    main()
