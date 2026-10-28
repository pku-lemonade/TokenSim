"""Measure sequential local vLLM requests; export with the existing LPX rows format.

Each output JSON line contains one completed request, including every turn.
Use a separate output file per server configuration. Prompts remain in the
external requests file; --export writes prompt-containing replay inputs and
must also point outside the repository. No automatic retries hide failed runs.
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
from scripts.generate_lpx_gemma_outputs import export, request_turns, SEED, TIMEOUT_S
from scripts.lpx.gpu_metrics import counters, counter_delta, decode_timing


def stream_events(response):
    """Read SSE data events, including multiline data, and require the terminator."""
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif not line and data:
            payload = "\n".join(data)
            data = []
            if payload == "[DONE]":
                return
            event = json.loads(payload)
            if "error" in event:
                raise WorkloadValidationError(f"Server stream error: {event['error']}")
            yield event
    raise WorkloadValidationError("Stream ended without [DONE]")


def metrics(args):
    with urllib.request.urlopen(args.metrics_url, timeout=30) as response:
        return counters(response.read().decode(), args.model)


def complete(args, messages, request):
    before = metrics(args)
    body = dict(
        model=args.model,
        messages=messages,
        temperature=0,
        seed=SEED,
        max_tokens=request["max_tokens"],
        chat_template_kwargs=dict(enable_thinking=request["reasoning"]),
        stream=True,
        stream_options=dict(include_usage=True),
        return_token_ids=True,
    )
    http_request = urllib.request.Request(
        args.api_base.rstrip("/") + "/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    arrivals, text_arrivals, content, reasoning = [], [], [], []
    token_arrivals, token_ids = [], []
    usage, finish_reason = None, None
    with urllib.request.urlopen(http_request, timeout=TIMEOUT_S) as response:
        for event in stream_events(response):
            arrived = time.perf_counter() - start
            arrivals.append(arrived)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                generated_ids = choice.get("token_ids") or []
                if generated_ids:
                    token_arrivals.append(arrived)
                    token_ids.extend(generated_ids)
                delta = choice.get("delta", {})
                answer = delta.get("content") or ""
                thought = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if answer or thought:
                    text_arrivals.append(arrived)
                    content.append(answer)
                    reasoning.append(thought)
                finish_reason = choice.get("finish_reason") or finish_reason
    elapsed = time.perf_counter() - start
    if usage is None or finish_reason is None:
        raise WorkloadValidationError("Stream lacks usage or finish reason")
    if len(token_ids) != usage["completion_tokens"]:
        raise WorkloadValidationError(
            "Streamed generated token IDs do not match completion usage"
        )
    time.sleep(args.metrics_settle_seconds)
    after = metrics(args)
    return dict(
        content="".join(content),
        reasoning="".join(reasoning),
        usage=usage,
        finish_reason=finish_reason,
        provider="local-vllm",
        model=args.model,
        elapsed_s=elapsed,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        chunk_arrivals_s=arrivals,
        text_chunk_arrivals_s=text_arrivals,
        token_chunk_arrivals_s=token_arrivals,
        token_ids=token_ids,
        metrics_before=before,
        metrics_after=after,
        **decode_timing(usage["completion_tokens"], token_arrivals),
        **counter_delta(before, after, args.method != "plain"),
    )


def generate(args):
    requests = [json.loads(line) for line in args.requests.read_text().splitlines()]
    digest = hashlib.sha256(args.requests.read_bytes()).hexdigest()
    config = dict(
        method=args.method,
        K=args.draft_tokens,
        weight_precision=args.weight_precision,
        kv_precision=args.kv_precision,
        drafter_precision=args.drafter_precision,
        model=args.model,
        run_label=args.run_label,
        max_tokens_override=args.max_tokens,
        metrics_settle_seconds=args.metrics_settle_seconds,
        requests_sha256=digest,
        return_token_ids=True,
        environment_sha256=(
            hashlib.sha256(args.environment.read_bytes()).hexdigest()
            if args.environment
            else None
        ),
        seed=SEED,
    )
    done = set()
    if args.outputs.exists():
        for line in args.outputs.read_text().splitlines():
            output = json.loads(line)
            if output["config"] != config:
                raise WorkloadValidationError(
                    "Output file contains a different configuration"
                )
            if output["id"] in done:
                raise WorkloadValidationError("Duplicate completed request")
            done.add(output["id"])
    selected = [
        request
        for request in requests
        if (not args.set or request["set"] == args.set)
        and (args.thinking == "both" or request["reasoning"] == (args.thinking == "on"))
        and (not args.context or request["meta"].get("context") == args.context)
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    args.outputs.parent.mkdir(parents=True, exist_ok=True)
    for original in selected:
        if original["id"] in done:
            continue
        request = dict(original)
        if args.max_tokens is not None:
            request["max_tokens"] = args.max_tokens
        turns = []
        for index, turn in enumerate(
            request_turns(
                request, lambda messages, request: complete(args, messages, request)
            )
        ):
            turn["turn"] = index
            turns.append(turn)
            print(
                f"{request['id']} turn={index} tokens={turn['completion_tokens']} "
                f"tps={turn['output_tps']} drafts={turn['drafts']}",
                flush=True,
            )
        output = dict(
            id=request["id"],
            set=request["set"],
            meta=request["meta"],
            thinking=request["reasoning"],
            max_tokens=request["max_tokens"],
            config=config,
            turns=turns,
        )
        with args.outputs.open("a") as handle:
            handle.write(json.dumps(output) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument("--method", choices=("plain", "mtp", "ngram"), default="plain")
    parser.add_argument("--draft-tokens", type=int, default=0)
    parser.add_argument("--weight-precision", default="bf16")
    parser.add_argument("--kv-precision", default="bf16")
    parser.add_argument("--drafter-precision", default="bf16")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--metrics-settle-seconds", type=float, default=6)
    parser.add_argument("--set", choices=("aa-docs", "speed-coding", "speed-cc16k"))
    parser.add_argument("--thinking", choices=("on", "off", "both"), default="both")
    parser.add_argument("--context", type=int, choices=(10000, 100000))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args(argv)
    if args.export:
        if args.export.resolve().is_relative_to(ROOT):
            parser.error("--export contains prompts and must be outside the repository")
        export(args)
        return 0
    if not args.environment:
        parser.error("--environment must identify the recorded server environment")
    if (args.method == "plain" and args.draft_tokens != 0) or (
        args.method != "plain" and args.draft_tokens not in (1, 3, 5, 7)
    ):
        parser.error("plain requires K=0; speculative runs require K in 1,3,5,7")
    if args.metrics_settle_seconds < 0:
        parser.error("metrics settle time must be nonnegative")
    if any(value is not None and value < 1 for value in (args.limit, args.max_tokens)):
        parser.error("limit and max-tokens must be positive")
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
