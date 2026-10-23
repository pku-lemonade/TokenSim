"""Collect separate semantic and GPU timing passes from a warmed vLLM engine."""

import argparse
import importlib.metadata
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

from scripts.collect.operator_trace_worker import ProfilerWorkerExtension
from TokenSim.profiler.collection import bind_captures, timing_quality


def run_generation(llm, prompts, sampling):
    if len(prompts) == 1:
        return llm.generate(prompts, sampling, use_tqdm=False)
    client = llm.llm_engine.engine_core
    if not hasattr(client, "call_utility") or not hasattr(llm, "enqueue"):
        raise RuntimeError(
            "batched paired capture requires vLLM scheduler pause/resume and enqueue support"
        )
    client.call_utility("pause_scheduler", "keep", False)
    try:
        llm.enqueue(prompts, sampling, use_tqdm=False)
    finally:
        client.call_utility("resume_scheduler")
    return llm.wait_for_completion(use_tqdm=False)


def collect(args):
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(vllm.__file__).resolve().parent.parent
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    manifest = dict(
        schema_version=2,
        measurement_protocol="paired_structure_gpu_timing_v1",
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
                name=torch.cuda.get_device_name(i),
                total_memory=torch.cuda.get_device_properties(i).total_memory,
            )
            for i in range(torch.cuda.device_count())
        ],
        capture_policy="cuda_graph" if args.cuda_graph else "eager",
        prompt_pattern=args.prompt_pattern,
        annotations="structure pass only; timing passes have no module hooks or shape recording",
        status="initializing",
        paired_captures=[],
        admission_policy="gated_burst" if args.batch_size > 1 else "single_request",
        collector_sources={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(inspect.getfile(ProfilerWorkerExtension)))
        },
    )
    path = output / "manifest.json"

    def save():
        path.write_text(json.dumps(manifest, indent=2))

    save()
    llm = None
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
            seed=args.seed,
            profiler_config={
                "profiler": "torch",
                "torch_profiler_dir": str(output / "structure"),
                "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": False,
                "torch_profiler_use_gzip": True,
                "torch_profiler_dump_cuda_time_total": False,
            },
            worker_extension_cls=f"{ProfilerWorkerExtension.__module__}.ProfilerWorkerExtension",
        )
        config = llm.llm_engine.vllm_config.model_config.hf_config
        manifest.update(
            model_config=config.to_dict(),
            model_revision=getattr(config, "_commit_hash", None),
        )
        generator = random.Random(args.seed)
        prompts = [
            {
                "prompt_token_ids": [
                    (
                        42
                        if args.prompt_pattern == "repeated"
                        else generator.randrange(100, config.vocab_size)
                    )
                    for _ in range(args.input_tokens)
                ]
            }
            for _ in range(args.batch_size)
        ]
        manifest["prompt_token_ids"] = [
            prompt["prompt_token_ids"] for prompt in prompts
        ]
        sampling = SamplingParams(
            temperature=0,
            max_tokens=args.output_tokens,
            min_tokens=args.output_tokens,
            ignore_eos=True,
        )
        expected = None

        def generate():
            nonlocal expected
            start = time.perf_counter()
            results = run_generation(llm, prompts, sampling)
            elapsed = time.perf_counter() - start
            tokens = [result.outputs[0].token_ids for result in results]
            if expected is None:
                expected = tokens
            elif tokens != expected:
                manifest["workload_mismatch"] = dict(
                    expected_token_ids=expected, observed_token_ids=tokens
                )
                raise RuntimeError("paired capture changed generated token IDs")
            return elapsed

        for _ in range(args.warmup):
            generate()
        manifest["worker_identity"] = llm.collective_rpc("tokensim_identity")
        before = [generate() for _ in range(args.repeats)]
        manifest["baseline_before_seconds"] = before

        def capture(name, structure):
            directory = output / name
            directory.mkdir()
            identities = llm.collective_rpc(
                "tokensim_configure_profiler",
                args=(
                    str(directory),
                    structure,
                    args.record_shapes if structure else False,
                ),
            )
            if not structure and any(
                identity["annotations_active"] for identity in identities
            ):
                raise RuntimeError("timing capture still has semantic annotations")
            llm.start_profile(name)
            try:
                elapsed = generate()
            finally:
                llm.stop_profile()
            files = sorted(directory.glob("*.json*"))
            if len(files) != args.tensor_parallel_size:
                raise RuntimeError(
                    "profiler did not produce one trace per tensor-parallel rank"
                )
            return files, elapsed

        manifest["annotated_modules_per_rank"] = llm.collective_rpc(
            "tokensim_annotations", args=(True,)
        )
        try:
            structure, elapsed = capture("structure", True)
            manifest["structure_generate_seconds"] = elapsed
        finally:
            llm.collective_rpc("tokensim_annotations", args=(False,))
        manifest["structure_trace_files"] = [
            str(p.relative_to(output)) for p in structure
        ]
        samples = []
        for index in range(args.repeats):
            files, elapsed = capture(f"timing{index}", False)
            manifest["paired_captures"].extend(
                bind_captures(
                    structure, files, output, expected_forwards=args.output_tokens
                )
            )
            samples.append(
                dict(
                    index=index,
                    generate_seconds=elapsed,
                    trace_files=[str(p.relative_to(output)) for p in files],
                )
            )
            manifest["timing_samples"] = samples
            save()
        after = [generate() for _ in range(args.repeats)]
        selected = sorted(samples, key=lambda sample: sample["generate_seconds"])[
            len(samples) // 2
        ]
        manifest.update(
            baseline_after_seconds=after,
            selected_timing_sample=selected["index"],
            trace_files=selected["trace_files"],
            generated_tokens=[len(tokens) for tokens in expected],
            generated_token_ids=expected,
            timing_quality=timing_quality(before + after, selected["generate_seconds"]),
            status="completed",
        )
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if llm is not None:
                llm.collective_rpc("tokensim_annotations", args=(False,))
        except Exception as error:
            manifest["cleanup_error"] = f"{type(error).__name__}: {error}"
            if manifest["status"] != "failed":
                manifest["status"] = "failed"
                raise
        finally:
            save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt-pattern", choices=("varied", "repeated"), default="varied"
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="record shapes in the structure pass only",
    )
    args = parser.parse_args(argv)
    for name in (
        "tensor_parallel_size",
        "batch_size",
        "input_tokens",
        "output_tokens",
        "warmup",
        "repeats",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.seed < 0 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error(
            "seed must be nonnegative and gpu-memory-utilization must be in (0, 1]"
        )
    collect(args)


if __name__ == "__main__":
    main()
