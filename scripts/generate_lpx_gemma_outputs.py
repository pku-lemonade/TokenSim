"""Generate Gemma 4 31B outputs for the LPX workload study through OpenRouter.

Reads the requests of build_lpx_api_prompts.py, sends each request's turns in
order at temperature 0 to one pinned provider, and appends one JSON line per
finished request to ``--outputs``, so an interrupted run resumes where it
stopped.  The API key comes from the ``OPENROUTER_API_KEY`` environment
variable or ``--key-file`` and is never written.  The run stops starting new
requests once the key's spending, read at start and summed from each
response's reported cost, reaches ``--budget`` US dollars.

``--export DIRECTORY`` instead writes one dataset-server-style rows file per
set, thinking mode and, for document questions, context and question style,
with one row per turn, for measure_lpx_lookup_acceptance.py
(``#combined`` and the other parts) and measure_lpx_token_ratio.py
(``#thinking`` and ``#answer``).  See docs/lpx-api-outputs.md.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

API = "https://openrouter.ai/api/v1"
MODEL = "google/gemma-4-31b-it"
SEED = 1
RETRIES = 6
TIMEOUT_S = 1800


def call(key, path, body=None):
    request = urllib.request.Request(
        API + path,
        json.dumps(body).encode() if body is not None else None,
        {"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.load(response)


def complete(key, provider, messages, request):
    body = dict(
        model=MODEL,
        messages=messages,
        temperature=0,
        seed=SEED,
        max_tokens=request["max_tokens"],
        reasoning=dict(enabled=request["reasoning"]),
        provider=dict(order=[provider], allow_fallbacks=False),
        usage=dict(include=True),
    )
    for attempt in range(RETRIES):
        start = time.time()
        try:
            result = call(key, "/chat/completions", body)
            if "choices" in result:
                result["elapsed_s"] = time.time() - start
                return result
            error = result.get("error")
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            json.JSONDecodeError,
            TimeoutError,
            ConnectionError,
        ) as exc:
            error = repr(exc)
        print(f"retry {attempt + 1} {request['id']}: {str(error)[:200]}", flush=True)
        time.sleep(min(300, 10 * 2**attempt))
    raise RuntimeError(f"{request['id']} failed after {RETRIES} attempts")


def run_request(key, provider, request):
    messages, turns = [], []
    for text in request["turns"]:
        messages.append(dict(role="user", content=text))
        result = complete(key, provider, messages, request)
        message = result["choices"][0]["message"]
        turns.append(
            dict(
                content=message.get("content") or "",
                reasoning=message.get("reasoning") or "",
                finish_reason=result["choices"][0].get("finish_reason"),
                provider=result.get("provider"),
                model=result.get("model"),
                usage=result.get("usage"),
                elapsed_s=result["elapsed_s"],
            )
        )
        messages.append(dict(role="assistant", content=turns[-1]["content"]))
    return dict(id=request["id"], set=request["set"], turns=turns)


def generate(args, key):
    requests = [json.loads(line) for line in args.requests.open()]
    done = set()
    if args.outputs.exists():
        done = {json.loads(line)["id"] for line in args.outputs.open()}
    pending = [r for r in requests if r["id"] not in done]
    spent = [call(key, "/key")["data"]["usage"]]
    lock = threading.Lock()
    print(f"{len(pending)} of {len(requests)} pending, spent ${spent[0]:.4f}")

    def work(request):
        with lock:
            if spent[0] >= args.budget:
                return
        output = run_request(key, args.provider, request)
        cost = sum((t["usage"] or {}).get("cost", 0) for t in output["turns"])
        with lock:
            spent[0] += cost
            with args.outputs.open("a") as handle:
                handle.write(json.dumps(output) + "\n")
            tokens = sum(
                (t["usage"] or {}).get("completion_tokens", 0) for t in output["turns"]
            )
            print(
                f"done {request['id']}: {tokens} output tokens, ${cost:.4f},"
                f" total ${spent[0]:.4f}",
                flush=True,
            )

    with ThreadPoolExecutor(args.concurrency) as pool:
        for future in [pool.submit(work, r) for r in pending]:
            future.result()
    print(f"finished, spent ${spent[0]:.4f}")


def file_name(request):
    """Rows file of a request: set, thinking mode, then context and style."""
    meta = request["meta"]
    parts = [request["set"], "think" if request["reasoning"] else "plain"]
    if "context" in meta:
        parts += [f"{meta['context'] // 1000}k", meta["style"]]
    return "-".join(parts)


def export(args):
    requests = {r["id"]: r for r in (json.loads(line) for line in args.requests.open())}
    files = {}
    for line in args.outputs.open():
        output = json.loads(line)
        request = requests[output["id"]]
        history = []
        for index, (text, turn) in enumerate(zip(request["turns"], output["turns"])):
            history.append(text)
            files.setdefault(file_name(request), []).append(
                dict(
                    row=dict(
                        id=output["id"],
                        turn=index,
                        meta=request["meta"],
                        messages=[
                            dict(role="user", content="\n\n".join(history)),
                            dict(
                                role="assistant",
                                content=turn["content"],
                                reasoning=turn["reasoning"],
                            ),
                        ],
                        thinking=turn["reasoning"],
                        answer=turn["content"],
                        finish_reason=turn["finish_reason"],
                        provider=turn["provider"],
                        usage=turn["usage"],
                        elapsed_s=turn["elapsed_s"],
                    )
                )
            )
            history.append(turn["content"])
    args.export.mkdir(parents=True, exist_ok=True)
    for name, rows in sorted(files.items()):
        rows.sort(key=lambda r: (r["row"]["id"], r["row"]["turn"]))
        (args.export / f"{name}.json").write_text(json.dumps(dict(rows=rows)) + "\n")
        print(f"{name}: {len(rows)} rows")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--provider", default="parasail/fp8")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--budget", type=float, default=2.5)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args(argv)
    if args.export:
        export(args)
        return 0
    key = os.environ.get("OPENROUTER_API_KEY")
    if args.key_file:
        key = args.key_file.read_text().strip()
    if not key:
        parser.error("set OPENROUTER_API_KEY or pass --key-file")
    generate(args, key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
