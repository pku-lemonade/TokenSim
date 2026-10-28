# Gemma 4 31B GPU measurements for the LPX study

Record date: 2026-10-28.
Status: T0 is complete; the priority-1 sweep is queued after the full-cap document control and a plain engine-counter check.
The plain 10K document request entered a repetition loop, which needs investigation before a representative document-acceptance claim.
The expectations in the [GPU TODO](lpx-gpu-gemma-todo.md#expectations) remain unchanged.

## Environment

The machine has one NVIDIA H20 with 97,871 MiB reported by `nvidia-smi`.
The driver version is 535.261.03.
The isolated server environment uses vLLM 0.32.0, PyTorch 2.13.0+cu130 and CUDA 13.0.
A CUDA tensor allocation and transfer succeeded in that environment.
The [environment record](../experiments/lpx/gpu-environment.json) contains the complete server package versions, GPU settings and checkpoint revisions.
The repository environment uses Python 3.12 with the development dependencies, `tokenizers==0.21.1`, `pyarrow==17.0.0` and `tiktoken==0.9.0`.
The system Python and PyTorch installations were not changed.
The [host-runtime record](../experiments/lpx/gpu-host-runtime.json) records the CPU model, affinity and inherited `OMP_NUM_THREADS=18` setting.

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
The client logs token counts and the longest identical-token run at intervals of approximately 1,024 generated tokens, without logging prompts or generated text.

Metrics schema 2 additionally records `iteration_tokens_total_count` as `emitted_steps` and `num_preemptions_total` as `preemptions` when the server exposes them.
The raw before/after snapshots and differences preserve both counters; missing counters in older records are not filled with zeros.
This schema version is part of the resume fingerprint, so it cannot silently mix with earlier collection files.
In vLLM 0.32.0, the iteration histogram counts nonempty `EngineCoreOutputs` batches in the asynchronous output handler, including the prefill output and zero-proposal n-gram decode outputs.
It does not count every GPU forward, because prefill chunks without output are omitted.
Using it as a decode-step denominator requires concurrency one, no preemptions, and plain/MTP identity checks on the running server before T4.
The analysis preserves these auxiliary counters without yet deriving n-gram acceptance from them.

The client reads `/metrics` before each turn and after a configurable settling delay, which defaults to six seconds.
The vLLM 0.32.0 implementation exports the four requested counters with the `_total` suffix and zero-based `position` labels.
The parser selects the served model and sums its engine series.
Raw snapshots and their differences are saved together.
Missing speculative counters, counter resets and incomplete streams raise errors.
Only this client may send requests to the server while it measures a turn.
The T0 counter identity includes the prefill token and a bounded terminal-token truncation, as recorded below.

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
The vLLM 0.32.0 scheduler also omits zero-proposal forwards from the speculative draft counter.
The analysis therefore retains n-gram counters without converting them into an overall committed-tokens-per-step estimate.
T4 needs a total-step measurement or a separately validated correction for these omitted forwards.
The [vLLM 0.32.0 n-gram proposer](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/spec_decode/ngram_proposer.py) chooses the earliest occurrence among equally long matches.
The existing lookup replay overwrites its index with each later occurrence, so it chooses the most recent one.
T4 must report this matching-rule difference when comparing the replay with vLLM.
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

[`run_lpx_gpu_study.py`](../scripts/run_lpx_gpu_study.py) runs T0 MTP and the priority-1 sweep in separate stages.
It reuses the recorded plain-server command, pins the assistant revision, checks the MTP startup method and counters, and waits for GPU memory release between servers.
An exclusive file lock prevents duplicate sweep runners.
Progress is written to `/tmp/lpx-gpu/runner-status.json` and completed requests resume from their JSON Lines records.
The T0 MTP coding turns are kept separate from the priority-1 `K = 3` sweep because the initial request triggered just-in-time kernel compilation.
Their file was renamed to `gpu-mtp-k3-coding-smoke.jsonl` after T0 completed, without altering any records.
Its SHA-256 is `ebe0ee9006c1cd555fc46eb4b0b5b6b5bbb23ac7f953201b77b047d33230fb12`.
Those historical records retain the original `t1-mtp-k3` run label, but they are T0 observations and are excluded from the full-run files.
The runner now gives smoke runs distinct labels and records a separate 128-token warmup before each priority-1 configuration.
Warmups do not guarantee that every later prompt shape has already compiled.

```bash
/tmp/lpx-gpu/repo-venv/bin/python scripts/run_lpx_gpu_study.py \
  --work-dir /tmp/lpx-gpu \
  --stage t0-mtp
/tmp/lpx-gpu/repo-venv/bin/python scripts/run_lpx_gpu_study.py \
  --work-dir /tmp/lpx-gpu \
  --stage priority1
```

| Task | Status |
| --- | --- |
| T0 | Complete; 5/10 coding pairs identical, so expectation 1 missed |
| T1 | Collection and acceptance analysis implemented; no measured rows |
| T2 | Identity-based pairing implemented; no measured pairs |
| T3 | Not run |
| T4 | N-gram counter collection and compatible export implemented; comparison not run |
| T5 | Ten-request smoke spread available; full coding runs pending |
| T6 | Nsight Systems is installed; profiling not run |
| T7 | Skipped because `data/operator_data/h20` does not exist |

The detached continuation records its status in `/tmp/lpx-gpu/continuation-status.json` and its output in `/tmp/lpx-gpu/continuation.log`.
It waits for the full mixed-attention control, checks the emitted-step counter on a separate 128-token plain request, then runs the original BF16/FlashAttention 4 priority-1 configurations at K=3 and K=7.
The full sweep retains every prescribed request even if it produces a failed or repetitive answer, with quality diagnostics included in the summaries.
Those failures will qualify the acceptance interpretation and will not be omitted to improve a reported result.
The request files are split by K as `gpu-mtp-k3-outputs.jsonl` and `gpu-mtp-k7-outputs.jsonl`; the combined analysis is `gpu-mtp-acceptance.json`.
The lower-priority T1 rows and T3 through T6 remain pending.

The device specification `data/devices/h20.yaml` exists, but it does not provide the operator measurements required by T7.
No substituted GPU operator table was used.

## T0 plain results

The [raw document turn](../experiments/lpx/gpu-plain-smoke.jsonl), [raw coding turns](../experiments/lpx/gpu-plain-coding-smoke.jsonl) and [summary](../experiments/lpx/gpu-plain-smoke-summary.json) retain every completed turn.
The document turn has 9,998 prompt tokens and 16,384 completion tokens, stops at the output limit and contains no final answer.
Time to first token is 5.280 seconds and decode time is 828.970 seconds, giving 19.763 output tokens per second.
Token 514 appears 16,135 times, with a longest identical run of 16,132 tokens.
Its rendered reasoning repeats `al` for most of the output.
This is a failed answer and cannot establish ordinary document-writing acceptance.
The raw result remains in the record.

The ten coding requests produce 12 turns, all with final answers and stop reasons of `stop`.
Their median speed is 42.834 tokens per second, with P80/median of 1.041 and maximum/median of 1.077.
These are smoke-test statistics on ten requests, not the full T5 comparison.
Repository unit tests ran on the CPU during parts of T0, so these timings do not establish an isolated serving-stack performance result.

The analysis now reports identical-token runs and flags runs of 128 tokens or more.
That diagnostic threshold was chosen after observing the plain T0 loop, and no flagged turn is excluded from acceptance or speed summaries.
The prescribed temperature remains zero.
The cause of the loop is not yet established; the result alone does not distinguish greedy model behavior from a serving-path problem.

The [runtime estimates](../experiments/lpx/gpu-runtime-estimates.json) use the observed 19.763-token/s rate and the previous API output lengths.
They give 7.972 generation hours for the priority-1 rows and 36.106 generation hours for the listed GPU tasks, excluding prefill, server startup, metrics settling and profiling overhead.
The rate comes from the looping document output, the length profile comes from different outputs, and speculative speedups are not assumed, so these are provisional planning estimates.

## T0 MTP results

Both MTP startups report `method='mtp'` and expose all four speculative-counter families.
The [server commands](../experiments/lpx/gpu-mtp-server-commands.jsonl) pin the assistant revision and preserve the plain BF16 settings.
The [K=1 document diagnostic](../experiments/lpx/gpu-mtp-k1-smoke.jsonl) uses the same 10K prompt with a separately recorded 128-token cap.
It produces 128 tokens from 74 drafts and 53 accepted draft tokens, satisfying `128 = 1 + 74 + 53`.
Its committed tokens per step are 1.716216, with equivalent acceptance 0.716216.
The short diagnostic does not test expectation 2, which requires the full K=3 document set.

The [K=3 coding smoke outputs](../experiments/lpx/gpu-mtp-k3-coding-smoke.jsonl) contain ten requests and 12 turns.
The [comparison](../experiments/lpx/gpu-smoke-comparison.json) finds five identical request pairs out of ten, below the preregistered minimum of nine.
Expectation 1 is missed.
The seven nonidentical turns have zero-based earliest differing native-token positions of 516, 43, 166, 175, 858, 613 and 40, listed by request identity in the comparison file.
All coding turns finish with `stop`, include an answer and have no identical-token repetition flag.
Output differences at greedy decoding are observed facts; this comparison does not isolate their numerical or implementation cause.
Later turns of a multi-turn request also inherit any differences in the preceding generated answer.

For the 12 coding turns, `completion_tokens - 1 - drafts - accepted_tokens` is zero in five turns, minus one in two turns and minus two in five turns.
The [vLLM scheduler](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/sched/scheduler.py) counts sampled and accepted tokens before `_update_request_with_output` truncates the generated list at a stop condition.
The observed identity is therefore `completion_tokens = 1 + drafts + accepted_tokens - terminal_trim`, with `terminal_trim` between zero and two in this sample.
Acceptance pricing uses the delivered completion-token count, so terminally discarded tokens do not inflate committed tokens per step.
Per-position acceptance remains the server's pre-truncation verification statistic.

The [smoke acceptance analysis](../experiments/lpx/gpu-mtp-k3-coding-smoke-acceptance.json) gives mean committed tokens per step of 3.569670 and equivalent acceptance 0.924555.
The pooled committed tokens per step are 3.515133.
MTP's smoke median speed is 142.027 tokens per second, with P80/median of 1.091656 and maximum/median of 1.195024.
These ten-request timings and their LP30 prices are smoke results; expectation 8 remains untested until the full coding runs.

## Document-quality controls

The [diagnostic summary](../experiments/lpx/gpu-document-diagnostics.json) compares native token prefixes and records repetition without excluding any output.
`analyze_lpx_gpu_smoke.py --diagnostics` accepts separately capped single-turn runs of the same request and compares only their shared prefix length.
The control settings were selected after observing the plain document loop and are not additional preregistered expectations.
All controls retain temperature zero, seed one, thinking on and the same request hash.

The [eager command](../experiments/lpx/gpu-server-eager-diagnostic-command.json) adds only `--enforce-eager` to the recorded plain command.
In vLLM 0.32.0 this disables both torch compilation and CUDA graphs.
The [eager output](../experiments/lpx/gpu-eager-diagnostic.jsonl) is capped at 512 tokens and still loops, with a longest identical-token run of 258 and no final answer.
Its native prefix differs from the original plain output at position 20.
The observed 23.693-token/s rate is a diagnostic timing, not a controlled CUDA-graph performance comparison.
The loop persisting in eager mode does not establish its cause or exclude faults shared by both serving paths.

The [FlashAttention control command](../experiments/lpx/gpu-server-fa3-diagnostic-command.json) adds `--attention-config` requesting `FLASH_ATTN` version 3 to the original compiled plain command.
The [startup selection record](../experiments/lpx/gpu-fa3-backend-selection.json) selects version 3 for supported layers but upgrades attention with head size 512 to version 4 on SM90.
This is a mixed FA3/FA4 control, not a test without FlashAttention 4.
It keeps the 8,192-token prefill batch limit and uses the same 512-token output cap as the eager control.
The [capped output](../experiments/lpx/gpu-fa3-diagnostic.jsonl) has no repetition flag, a longest identical-token run of three and an earliest prefix difference from the original plain run at position 75.
The cap ends during reasoning, so the absence of an early loop does not establish successful completion.
A separately labeled full-cap run of this control is in progress to check answer completion.
The primary run remains on the original FlashAttention 4 settings while this control is assessed.
CPU unit tests overlap portions of the control runs; their timings are not isolated throughput measurements.

## Validation

Black's repository-wide formatting check passes.
The completed unit-suite run passes all 687 tests, including diagnostic-prefix comparison, smoke/warmup separation and auxiliary engine counters.
The initial full-suite attempt lacked `matplotlib`, which the existing report tests import; installing it in the repository environment resolved that failure.
`git diff --check` passes.

The tooling and verified inputs are saved in local commit `fcf13ac`.
Pushing that commit to `origin/lpx-context-parallel` returned HTTP 403 because the configured GitHub account does not have repository write permission.

## Expectations

| Expectation | Outcome |
| --- | --- |
| 1: Plain and MTP output equality | Missed: 5/10 pairs identical, below the expected 9/10 |
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
