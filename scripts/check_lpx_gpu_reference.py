"""Check a GPU study prompt with native Transformers, without storing the prompt.

Use --prepare against an idle vLLM server to verify the exact template tokens.
Stop that server before running the independent SDPA generation on the same GPU.
This is a separately labeled diagnostic, not a replacement for the study runs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.errors import WorkloadValidationError
from scripts.lpx.gpu_metrics import output_quality


def token_fingerprint(tokens):
    encoded = json.dumps(tokens, separators=(",", ":")).encode()
    return dict(tokens=len(tokens), sha256=hashlib.sha256(encoded).hexdigest())


def finish_reason(tokens, eos_tokens, maximum):
    if isinstance(eos_tokens, int):
        eos_tokens = [eos_tokens]
    if tokens and tokens[-1] in (eos_tokens or []):
        return "stop"
    return "length" if len(tokens) == maximum else "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--prompt-check", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    import torch
    import transformers
    from transformers import (
        AutoModelForImageTextToText,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    requests = [json.loads(line) for line in args.requests.read_text().splitlines()]
    selected = [request for request in requests if request["id"] == args.request_id]
    if len(selected) != 1 or len(selected[0]["turns"]) != 1:
        parser.error("select one existing single-turn request")
    request = selected[0]
    messages = [dict(role="user", content=request["turns"][0])]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=request["reasoning"],
    )
    fingerprint = token_fingerprint(prompt_ids)
    request_hash = hashlib.sha256(args.requests.read_bytes()).hexdigest()
    if args.prepare:
        body = dict(
            model="google/gemma-4-31B-it",
            messages=messages,
            add_generation_prompt=True,
            chat_template_kwargs=dict(enable_thinking=request["reasoning"]),
        )
        http_request = urllib.request.Request(
            args.api_base.rstrip("/") + "/tokenize",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(http_request, timeout=60) as response:
            served = json.load(response)
        served_fingerprint = token_fingerprint(served["tokens"])
        check = dict(
            id=request["id"],
            requests_sha256=request_hash,
            native=fingerprint,
            vllm=served_fingerprint,
            passed=fingerprint == served_fingerprint,
        )
        args.prompt_check.write_text(json.dumps(check, indent=2) + "\n")
        if not check["passed"]:
            raise WorkloadValidationError("Native and vLLM prompt tokens differ")
        print("Prompt token fingerprints match", flush=True)
        return 0
    check = json.loads(args.prompt_check.read_text())
    if not (
        check["passed"]
        and check["id"] == request["id"]
        and check["requests_sha256"] == request_hash
        and check["native"] == fingerprint
    ):
        raise WorkloadValidationError("Prompt check does not match this diagnostic")
    result = dict(
        id=request["id"],
        requests_sha256=request_hash,
        checkpoint=str(args.checkpoint),
        config=dict(
            implementation="native Transformers",
            transformers_version=transformers.__version__,
            torch_version=torch.__version__,
            attention="sdpa",
            dtype="bfloat16",
            do_sample=False,
            seed=1,
            max_new_tokens=args.max_tokens,
            thinking=request["reasoning"],
        ),
        prompt_fingerprint=fingerprint,
        status="running",
    )

    class ProgressObserver(StoppingCriteria):
        def __init__(self):
            self.next_tokens = 1024

        def __call__(self, input_ids, scores, **kwargs):
            generated = input_ids.shape[1] - len(prompt_ids)
            if generated >= self.next_tokens:
                tokens = input_ids[0, len(prompt_ids) :].tolist()
                diagnostic = output_quality(dict(token_ids=tokens))
                print(
                    f"Native progress tokens={generated} "
                    f"periodic_span={diagnostic['longest_periodic_token_span']} "
                    f"period={diagnostic['periodic_token_period']}",
                    flush=True,
                )
                self.next_tokens = (generated // 1024 + 1) * 1024
            return False

    try:
        torch.manual_seed(1)
        start = time.perf_counter()
        model = (
            AutoModelForImageTextToText.from_pretrained(
                args.checkpoint,
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .eval()
            .to("cuda")
        )
        torch.cuda.synchronize()
        result["load_s"] = time.perf_counter() - start
        print("Native model loaded", flush=True)
        input_ids = torch.tensor([prompt_ids], device="cuda")
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=args.max_tokens,
                do_sample=False,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([ProgressObserver()]),
            )
        torch.cuda.synchronize()
        result["generation_s"] = time.perf_counter() - start
        tokens = output[0, len(prompt_ids) :].tolist()
        stopped = finish_reason(
            tokens, model.generation_config.eos_token_id, args.max_tokens
        )
        result.update(
            status="complete",
            completion_tokens=len(tokens),
            token_ids=tokens,
            generated_text=tokenizer.decode(tokens, skip_special_tokens=False),
            finish_reason=stopped,
            quality=output_quality(dict(token_ids=tokens, finish_reason=stopped)),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
    except Exception as error:
        result.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["quality"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
