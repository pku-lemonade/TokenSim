# Gemma 4 31B GPU measurements for the LPX study

Record date: 2026-10-28.
Status: both checkpoints and all inputs are downloaded; the plain server is starting for T0.
No model inference result is available yet.
The expectations in the [GPU TODO](lpx-gpu-gemma-todo.md#expectations) remain unchanged.

## Environment

The machine has one NVIDIA H20 with 97,871 MiB reported by `nvidia-smi`.
The driver version is 535.261.03.
The isolated server environment uses vLLM 0.32.0, PyTorch 2.13.0+cu130 and CUDA 13.0.
A CUDA tensor allocation and transfer succeeded in that environment.
The [environment record](../experiments/lpx/gpu-environment.json) contains the complete server package versions, GPU settings and checkpoint revisions.
The repository environment uses Python 3.12 with the development dependencies, `tokenizers==0.21.1`, `pyarrow==17.0.0` and `tiktoken==0.9.0`.
The system Python and PyTorch installations were not changed.

| Checkpoint | Revision |
| --- | --- |
| `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| `google/gemma-4-31B-it-assistant` | `627c5ec1458b9086b841a91e0512fd31fd2fbbf1` |

The model cache, inputs, virtual environments and logs are under `/tmp/lpx-gpu`, outside the repository.
Direct Project Gutenberg transfers failed or timed out during input preparation.
The retry uses compressed transfers with four concurrent downloads.
The initial Hugging Face Xet transfer reduced its concurrency to one or two streams.
The replacement transfers set `HF_XET_HIGH_PERFORMANCE=1` and use minimum, initial and maximum download concurrencies of 16/16/32 and then 64/64/128.
Download settings have no effect on the inference settings.

The active [plain-server command](../experiments/lpx/gpu-server-plain-active-command.json) specifies one GPU, BF16 weights and key/value cache, a 120,000-token sequence limit, one maximum sequence and disabled prefix caching.
The offline model cache changes vLLM's default served model name to a local path, so the command sets `--served-model-name` explicitly.
An earlier startup without that alias was stopped before readiness.
Its immediate replacement overlapped GPU cleanup and rejected the remaining free memory; no inference ran on that attempt.
The active startup began after the GPU was free and loaded both target weight shards successfully.

## Inputs

All 264 requests were rebuilt with the existing prompt builder.
Every source-file hash and the request-file hash matches the TODO's recorded values.
The [input-hash record](../experiments/lpx/gpu-input-hashes.json) lists each comparison.
The request-file SHA-256 is `7abb2f93587b01649c27e262e041afe96dd68cc055908d80f35610b5de9506ae`.
Two initial book downloads were truncated despite a successful HTTP transfer status.
Their hashes and missing endings exposed the truncation; retrying through the `/ebooks/<id>.txt.utf-8` URLs recovered the exact expected files before the final prompt rebuild.
No inference request used the truncated files.

## Collection

[`generate_lpx_gpu_outputs.py`](../scripts/generate_lpx_gpu_outputs.py) sends streaming requests to a local vLLM server at concurrency one.
It shares the multi-turn conversation loop and export function with the OpenRouter client.
It omits the OpenRouter-specific request fields and maps thinking to `chat_template_kwargs.enable_thinking`.
Temperature is zero and the seed is one.

Each JSON Lines record contains one completed request and all its turns, without the prompts.
The record includes the input-file hash, environment-file hash, precision settings, draft length, run label and output-length override.
Resuming a file with different settings raises an error.
Smoke runs with shorter outputs must use separate files from the full measurements.
A failed stream is not automatically retried or recorded as a completed request.
If a multi-turn request fails, its completed earlier turns are rerun when the request resumes.

Every server-sent JSON event has an arrival timestamp relative to the request start.
Time to first token uses the earliest event with generated token IDs, including reasoning delimiters that the text parser may hide.
Decode time ends at the last such event.
Events without generated tokens remain in `chunk_arrivals_s` but do not extend the measured decode interval.
This avoids including usage-report delivery time in output speed.
The client reports `(completion_tokens - 1) / decode_s` and leaves speed null when all tokens arrive in one event.
It retains generated token IDs and checks their count against completion usage.
Prompt token IDs returned by the server are discarded and never written.
The timestamps measure client delivery, including transport and buffering; they do not identify individual GPU steps.

The client reads `/metrics` before each turn and after a configurable settling delay, which defaults to six seconds.
The vLLM 0.32.0 implementation exports the four requested counters with the `_total` suffix and zero-based `position` labels.
The parser selects the served model and sums its engine series.
Raw snapshots and their differences are saved together.
Missing speculative counters, counter resets and incomplete streams raise errors.
Only this client may send requests to the server while it measures a turn.
The counter identity still needs verification on model outputs.

Example collection command after server startup:

```bash
/tmp/lpx-gpu/repo-venv/bin/python scripts/generate_lpx_gpu_outputs.py \
  --requests /tmp/lpx-gpu/inputs/requests.jsonl \
  --outputs experiments/lpx/gpu-plain-smoke.jsonl \
  --environment experiments/lpx/gpu-environment.json \
  --api-base http://127.0.0.1:8000/v1 \
  --metrics-url http://127.0.0.1:8000/metrics \
  --run-label t0-plain \
  --method plain \
  --set aa-docs \
  --thinking on \
  --context 10000 \
  --limit 1
```

`--export` uses the existing dataset-server rows format for the lookup replay and token-ratio scripts.
Exports include prompts and the client requires an export directory outside the repository.

## Acceptance and LP30 pricing

[`analyze_lpx_gpu_acceptance.py`](../scripts/analyze_lpx_gpu_acceptance.py) groups turns by configuration, set, thinking mode, context and question style.
It reports the arithmetic mean of per-turn committed tokens per step, `tau`, and also reports the step-weighted aggregate as `pooled_tau`.
Equivalent acceptance is derived from the mean `tau` with the existing helper.
Per-position acceptance divides the summed accepted-token counters by the summed draft-step counters.
The analysis records the counter-identity residual and any shortfall from `K` proposed tokens per step.
It rejects impossible token counts instead of clipping them to an acceptance of zero or one.

MTP turns have the requested `[decode_tokens, counts]` representation with only the `K + 1` verification-row count populated.
N-gram turns do not receive that representation because their proposed lengths can vary.
The 10K and 100K analysis joins document turns by book identity and reports unmatched books.
Turn-level counters cannot split thinking acceptance from answer acceptance.

Pricing reads the existing speculative results and uses the `tp32-pp8-kvp8` layout at attention matrix efficiency 0.50.
This is TP32/PP8 with decode context parallelism (DCP) of eight.
The bare `tp32-pp8` rows named in the TODO's pricing paragraph have a different DCP setting and are not used.
The two drafter costs are `mtp-0.05` and `mtp-0.10`.
AA prices use each context's measured `tau`, the existing `priced` and `window` helpers and a default token ratio of 1.006.
`--ratio` supplies a replacement ratio when measurements on the new outputs require it.
Coding prices use the existing `context_model`, add the interpolated drafter cost and evaluate each turn at its observed prompt length plus half its decode length.
GPU and LP30 speed tables both report the median, 80th percentile and maximum, including their ratios to the median.

[`analyze_lpx_gpu_smoke.py`](../scripts/analyze_lpx_gpu_smoke.py) compares plain and MTP outputs by request and turn identity.
It records the earliest differing native token position and also compares the re-tokenized generated text.
It also estimates generation time from the measured plain 10K rate, using the previous API output lengths and a separate calculation in which every output reaches its configured cap.
These estimates exclude prefill, server startup, metrics settling and profiling overhead, and assume no speculative speedup.

Example analysis command:

```bash
/tmp/lpx-gpu/repo-venv/bin/python scripts/analyze_lpx_gpu_acceptance.py \
  --outputs experiments/lpx/gpu-mtp-k3-outputs.jsonl \
  --summary experiments/lpx/gpu-mtp-acceptance.json \
  --ratio 1.006
```

## Task status

| Task | Status |
| --- | --- |
| T0 | Checkpoints downloaded; plain-server startup and subsequent smoke requests in progress |
| T1 | Collection and acceptance analysis implemented; no measured rows |
| T2 | Identity-based pairing implemented; no measured pairs |
| T3 | Not run |
| T4 | N-gram counter collection and compatible export implemented; comparison not run |
| T5 | GPU and LP30 speed summaries implemented; no measured speed spread |
| T6 | Nsight Systems is installed; profiling not run |
| T7 | Skipped because `data/operator_data/h20` does not exist |

The device specification `data/devices/h20.yaml` exists, but it does not provide the operator measurements required by T7.
No substituted GPU operator table was used.

## Validation

Black's repository-wide formatting check passes.
The complete unit suite passes all 678 tests, including the new collection, acceptance, pricing and smoke-comparison tests.
The initial full-suite attempt lacked `matplotlib`, which the existing report tests import; installing it in the repository environment resolved that failure.
`git diff --check` passes.

## Expectations

| Expectation | Outcome |
| --- | --- |
| 1: Plain and MTP output equality | Not tested |
| 2: AA-document acceptance | Not tested |
| 3: Coding acceptance advantage | Not tested |
| 4: Paired context-length difference | Not tested |
| 5: LP30 pricing at measured acceptance | Not tested |
| 6: FP8-weight acceptance difference | Not tested |
| 7: N-gram and replay agreement | Not tested |
| 8: GPU coding speed spread | Not tested |

## Differences from the published runs

| Item | Published runs | GPU measurements |
| --- | --- | --- |
| Model build | AA: an updated private, pre-release build; SPEED-Bench: not stated | Public checkpoint at the revision recorded above |
| Hardware | LP30 chips in an LPX rack | One NVIDIA H20; LP30 prices come from TokenSim |
| Precision | FP8, according to press reports | BF16 planned, with FP8 weights in T3 |
| Drafter and `K` | Not stated | Released MTP assistant and vLLM n-gram drafter |
| Prompts | AA's prompts unpublished; SPEED-Bench split not stated | Study book questions and public SPEED-Bench prompts |
| Thinking | Not stated | Explicit per-request setting |
| Sampling | Not stated | Temperature zero, seed one |
| Load | AA: concurrency one; SPEED-Bench: not stated | Concurrency one and maximum batch size one |

The implementation and environment checks do not change the study's conclusions about AA or NVIDIA's deployment.
