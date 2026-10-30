# Gemma 4 31B GPU measurements for the LPX study

Record date: 2026-10-30.
Status: T0–T6 measurements and exploratory analyses are complete under the separately recorded Triton profiles, including all MTP priorities, FP8, native n-gram replay, context pairing, LP30 pricing and graph/eager decode profiling.
The native reference and full-cap Triton plain/MTP controls complete without repetition.
The original document failures remain recorded; T6 reports unclassified activity and instrumentation limits explicitly.
The [completed follow-up results](lpx-gpu-followup-results.md) report all remaining measurements, including 23 capped thinking turns without final answers.
The [profiling report](lpx-gpu-profile-results.md) separates GPU costs from overlapping scheduler/streaming scopes and does not infer an uninstrumented LP30 overhead.
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
The client logs token counts, identical-token runs and periodic spans at intervals of approximately 1,024 generated tokens, without logging prompts or generated text.
The interrupted K=3 document client retained its earlier single-token-only progress logging; the analysis applies the extended periodic diagnostic to its completed records.

Metrics schema 2 additionally records `iteration_tokens_total_count` as `emitted_steps` and `num_preemptions_total` as `preemptions` when the server exposes them.
The raw before/after snapshots and differences preserve both counters; missing counters in older records are not filled with zeros.
This schema version is part of the resume fingerprint, so it cannot silently mix with earlier collection files.
In vLLM 0.32.0, the iteration histogram counts nonempty `EngineCoreOutputs` batches in the asynchronous output handler, including the prefill output and zero-proposal n-gram decode outputs.
It does not count every GPU forward, because prefill chunks without output are omitted.
Using it as a decode-step denominator requires concurrency one, no preemptions, and plain/MTP identity checks on the running server before T4.
The [counter crosscheck](../experiments/lpx/gpu-engine-counter-crosscheck.json) observes 128 emitted steps for 128 plain tokens and 44 emitted steps for 43 MTP draft steps, with no preemptions.
The n-gram acceptance calculation requires the explicit `--validated-ngram-engine-steps` flag after checking the serving configuration and these identities.
It uses `emitted_steps - 1` as the total decode-step count, includes zero-proposal steps, rejects preemptions and checks the terminal-trimming residual.
N-gram per-position acceptance then uses all decode steps as its denominator, including zero-proposal steps.
The default analysis still withholds n-gram acceptance when that validation is not asserted.

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
N-gram turns do not receive that representation because the analysis does not assume fixed verification-row counts across native implementations.
The vLLM 0.32.0 scheduler also omits zero-proposal forwards from the speculative draft counter.
The default analysis therefore retains n-gram counters without converting them into an overall committed-tokens-per-step estimate.
The guarded engine-step calculation described above provides a denominator for T4 after its serving-path checks.
The [vLLM 0.32.0 V1 CPU n-gram proposer](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/spec_decode/ngram_proposer.py) chooses the earliest occurrence among equally long matches.
The existing lookup replay overwrites its index with each later occurrence, so it chooses the most recent one.
The V2 runner dispatches the same `method=ngram` setting to `NgramGPUSpeculator`, whose Triton scan selects the longest n-gram and then the most recent occurrence.
V2 fills unavailable proposal slots with the last sampled token and verifies them as ordinary drafts, while the replay uses shorter proposals or no proposal.
The [source audit](../experiments/lpx/gpu-triton-followup-ngram-source-audit.json) records the installed dispatch and kernel source hashes and the replay implementation.
T4 must identify the runner implementation before reporting matching-rule differences.
The replay also tokenizes exported text without the server's chat-template tokens, inserts two newlines between thinking and answer, and flattens multi-turn history into user text.
Its numerator includes all re-tokenized output text, while native decode acceptance excludes the prefill token and includes generated control tokens hidden by the text parser.
These differences prevent attributing a measured discrepancy solely to the matching implementation.
[`analyze_lpx_gpu_ngram.py`](../scripts/analyze_lpx_gpu_ngram.py) reuses `texts` and `replay` from the existing lookup script at minimum n-gram length one, maximum length three and K=7.
It joins exports and measured turns by request/turn identity, rejects duplicate or mismatched text, and checks the recorded server command's lookup bounds and maximum sequence count.
It reports per-turn differences and set-level equivalent acceptances derived from mean committed tokens per step, without inventing fixed verification-row counts.
It records `native_implementation` from the command's explicit `VLLM_USE_V2_MODEL_RUNNER` environment setting and reports implementation-specific matching rules.
An absent or automatic runner setting is reported as `unknown`, without assuming V1 behavior.
Its `--validated-engine-steps` flag is mandatory; the [completed coding comparison](#completed-n-gram-coding-comparison) uses it with the recorded V2 server command.
After collecting T4, export its rows outside the repository and use the recorded n-gram server command with the comparison:

```bash
/tmp/lpx-gpu/repo-venv/bin/python scripts/analyze_lpx_gpu_ngram.py \
  --outputs experiments/lpx/gpu-triton-followup-ngram-k7-outputs.jsonl \
  --rows /tmp/lpx-gpu/rows-gpu-triton-followup-ngram-k7-outputs \
  --gemma-tokenizer /tmp/lpx-gpu/inputs/tokenizer.json \
  --server-command experiments/lpx/gpu-triton-followup-ngram-k7-server-command.json \
  --validated-engine-steps \
  --summary experiments/lpx/gpu-triton-followup-ngram-comparison.json
```

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
Warmup filenames carry a numeric run tag, so restarting a server cannot silently reuse a previously completed warmup instead of executing one.
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
| T1 | Original failed outputs preserved; Triton K=3/K=7 priority-1 sweep complete; remaining rows queued |
| T2 | Triton priority-1 context pairing complete |
| T3 | Complete; all four FP8 document acceptance differences are below 0.03 |
| T4 | Complete coding set replayed; document collection and its replay comparison pending |
| T5 | Full plain and priority-1 MTP coding spreads complete; remaining T1 configurations pending |
| T6 | Graph/eager collection queued after the full follow-up sweep; attribution pending |
| T7 | Skipped because `data/operator_data/h20` does not exist |

The detached continuation records its status in `/tmp/lpx-gpu/continuation-status.json` and its output in `/tmp/lpx-gpu/continuation.log`.
It completed the full mixed-attention control and plain counter check, then began the original BF16/FlashAttention 4 K=3 document sweep.
Its recorded failed subprocess status now corresponds to the intentional interruption described below; it is not an unreported server crash.
Every completed turn remains in the results, with failures qualifying the acceptance interpretation.
The uncompleted prescribed requests remain pending and are not treated as measured turns.
On resumption, the runner collects K=3 and K=7 coding requests before the document sweeps, prioritizing the set that passed the T0 answer checks.
This execution-order change was selected after observing the document failures and leaves the preregistered expectations unchanged.
The request files are split by K as `gpu-mtp-k3-outputs.jsonl` and `gpu-mtp-k7-outputs.jsonl`; the combined analysis is `gpu-mtp-acceptance.json`.
The lower-priority T1 rows, T4 comparison, remaining T5 configurations and T6 attribution remain pending.

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

The analysis reports identical-token runs and periodic spans of 128 tokens or more, searching native-token periods from one through eight.
The 128-token threshold was chosen after observing the plain T0 loop; the periodic search was added after observing a two-token loop in the T1 K=3 document output.
No flagged turn is excluded from acceptance or speed summaries.
This bounded diagnostic does not detect every possible repeated phrase or establish semantic answer quality.
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
The [full-cap control](../experiments/lpx/gpu-fa3-full-diagnostic.jsonl) also fails: it reaches 16,384 tokens with a longest identical-token run of 14,910.
It starts an answer but then repeats `al`, so a nonempty answer field does not establish successful completion.
Its decode time is 774.451 seconds, giving 21.154 tokens per second.
The [cap comparison](../experiments/lpx/gpu-fa3-cap-comparison.json) confirms identical native prefixes over all 512 tokens shared with the short control.
The short control therefore missed a loop that began later in the same trajectory.
The primary run retains the original FlashAttention 4 settings; the mixed-attention control did not eliminate the failure.
CPU unit tests overlap portions of the control runs; their timings are not isolated throughput measurements.

## T1 progress

The K=3 warmup completes with 128 tokens, 43 draft steps and 44 emitted engine steps.
The [progress snapshot](../experiments/lpx/gpu-mtp-k3-progress-acceptance.json) covers two completed 10K requests and is separate from the eventual full-set analysis.
Both requests reach 16,384 tokens without a final answer.
The summary request repeats a two-token pattern rendered as `alized`, with a periodic span of 15,346 tokens.
Its longest identical-token run is only three, which motivated the periodic diagnostic described above.
The turn has 4,210 draft steps and 4,211 emitted steps, with equivalent acceptance 0.981686 and a counter-identity residual of minus one.
That high acceptance measures the failed repetitive output and is not evidence of ordinary prose acceptance.
The quote request has an identical-token run of 15,594 tokens and is also retained.
The full-set expectation outcomes remain pending until the prescribed requests are collected and their failure counts are reported.
Validation tests overlap portions of the T1 document collection; document GPU timings are not isolated serving-performance measurements.

## Native reference check

The 100K summary request also entered an identical-token loop.
The [interruption record](../experiments/lpx/gpu-interrupted-attempts.jsonl) captures the active configuration, cumulative counter snapshot and latest logged progress: 8,193 generated tokens with an identical-token run of 8,028.
The client was deliberately terminated to investigate the model path before spending further GPU time on repeated failed answers.
Its two completed 10K records are preserved, and the interrupted 100K attempt is not a complete acceptance measurement.
The server was stopped and GPU memory released before starting the independent reference.

[`check_lpx_gpu_reference.py`](../scripts/check_lpx_gpu_reference.py) uses native Transformers 5.19.0, the same checkpoint, BF16 weights, scaled dot-product attention (SDPA), greedy decoding and seed one.
The [prompt check](../experiments/lpx/gpu-reference-prompt-check.json) compares exact token-sequence fingerprints from native template application and the idle vLLM `/tokenize` endpoint.
Both produce 9,998 tokens with SHA-256 `d1f307b82e800e781c23b28fa6d12542eb9d6115ee2bdd8ecb684e7c8e665124`.
No prompt token IDs are saved.
The [native command](../experiments/lpx/gpu-native-reference-command.json) keeps the full 16,384-token cap because the short mixed-attention control failed to expose its later loop.
The script saves generated token IDs and text, completion diagnostics, package versions and elapsed generation time; that elapsed time includes prefill and is not an AA output-speed measurement.
The [native output](../experiments/lpx/gpu-transformers-diagnostic.json) finishes with `stop` after 3,933 tokens and has no periodic repetition flag.
The [parsed comparison](../experiments/lpx/gpu-native-comparison.json) counts 2,002 whitespace-separated words in the answer and 5,863 reasoning characters.
Model loading takes 13.161 seconds and generation including prefill takes 314.743 seconds.
The native prefix differs from the original, eager and mixed-FA3/FA4 vLLM plain paths at token position 19.
This completed answer shows that the same checkpoint, prompt tokens and greedy choice can finish without the observed loop on an independent implementation.
It does not isolate the responsible numerical operation, attention backend or model-runner behavior in vLLM.

The [V1-runner command](../experiments/lpx/gpu-server-v1-plain-command.json) sets `VLLM_USE_V2_MODEL_RUNNER=0` and otherwise retains the original plain-server arguments.
It ran the same full-cap document request under a separate label and did not replace the earlier failed results.
The V1 client was deliberately interrupted after logging 3,072 generated tokens and an identical-token run of 2,943; the interruption record retains its configuration and counters.
A separate [512-token V1 control](../experiments/lpx/gpu-v1-plain-capped.jsonl) confirms repetition with an identical-token run of 317 and no answer.
Its native prefix differs from Transformers at position four.

## Triton-attention control

The [Triton command](../experiments/lpx/gpu-server-triton-command.json) retains the original plain arguments and V2 runner, adding `--attention-config '{"backend":"TRITON_ATTN"}'`.
The startup log confirms `Using AttentionBackendEnum.TRITON_ATTN backend.`.
The initial collection wrapper expected a different log spelling and failed after successful server startup; manual verification allowed collection without a server restart, as recorded in the [collection command](../experiments/lpx/gpu-triton-collection-command.json).
No installed model or kernel code was changed.

The [2,048-token diagnostic](../experiments/lpx/gpu-triton-diagnostic.jsonl) begins an answer without repetition but reaches its diagnostic cap.
The [full-cap control](../experiments/lpx/gpu-triton-full-diagnostic.jsonl) retains the original 16,384-token cap and finishes with `stop` after 3,736 tokens.
It produces 1,849 whitespace-separated answer words and 5,727 reasoning characters, with a longest periodic span of three tokens and no repetition flag.
Its TTFT is 5.987244 seconds, decode time is 89.758367 seconds and output speed is 41.611719 native tokens per second.
The [native comparison](../experiments/lpx/gpu-triton-native-comparison.json) places the earliest differing token at position 125 for both Triton controls.
This is a successful completion check for one document request, not evidence that every document or MTP configuration will complete normally.
The [recomputed runtime estimate](../experiments/lpx/gpu-triton-runtime-estimates.json) uses this plain decode rate and the existing API output lengths; startup, prefill and profiling costs remain excluded.
It estimates 3.786 generation hours for the priority-1 rows and 17.148 hours for the full task list without assuming speculative speedup.

Triton is selected after observing these controls and is reported as a separate serving profile.
The runner's `--profile triton` option uses the recorded Triton base command, writes `gpu-triton-*` artifacts and uses distinct run labels; the default `--profile original` preserves the earlier paths.
Server reuse requires matching arguments, recorded environment and the corresponding live environment values, so matching draft length alone cannot reuse another attention configuration.
The [ten plain coding requests](../experiments/lpx/gpu-triton-plain-coding-smoke.jsonl) complete all 12 turns with `stop`, nonempty answers and no repetition flags.
Their [analysis](../experiments/lpx/gpu-triton-plain-coding-analysis.json) gives median 48.599113 TPS, P80/median 1.006888 and maximum/median 1.011650.
Repository validation overlaps part of that collection, so these timings are not isolated serving measurements.
The profile-specific checks run under the [recorded smoke command](../experiments/lpx/gpu-triton-t0-command.json), with server starts retained in the [profile command log](../experiments/lpx/gpu-triton-mtp-server-commands.jsonl).
The [K=1 diagnostic](../experiments/lpx/gpu-triton-mtp-k1-smoke.jsonl) completes its 128-token cap with 70 drafts, 58 accepted tokens, 71 emitted steps and no preemptions.
Its [counter analysis](../experiments/lpx/gpu-triton-mtp-k1-smoke-analysis.json) gives `128 = 1 + 70 + 58 - 1`, tau 1.814286 and equivalent acceptance 0.814286; the subtracted token is terminal trimming.
The [K=3 coding comparison](../experiments/lpx/gpu-triton-smoke-comparison.json) matches six of ten request pairs, which also falls below the nine-pair threshold but does not replace the original five-pair result.
All 12 turns finish with `stop`, nonempty answers and no repetition flags.
Their [analysis](../experiments/lpx/gpu-triton-mtp-k3-coding-analysis.json) gives mean tau 3.601586, equivalent acceptance 0.930427, median 150.349133 TPS, P80/median 1.036637 and maximum/median 1.141568.
Counter residuals are zero on two turns, minus one on five, minus two on three and minus three on two, consistent with terminal trimming of up to K tokens.
These are smoke timings, with validation overlapping early turns; they are separate from the warmed full-set measurements.
The [full-cap K=3 document control](../experiments/lpx/gpu-triton-mtp-k3-smoke.jsonl) stops after 3,765 tokens and produces 1,939 whitespace-separated answer words without repetition.
Its [analysis](../experiments/lpx/gpu-triton-mtp-k3-document-analysis.json) gives 1,339 drafts, 2,426 accepted tokens, 1,340 emitted steps, no preemptions, tau 2.811053 and equivalent acceptance 0.768057.
The counter identity is `3765 = 1 + 1339 + 2426 - 1`.
Output speed is 74.566560 TPS, with TTFT 6.094277 seconds and decode time 50.478391 seconds.
The [plain/MTP document comparison](../experiments/lpx/gpu-triton-document-comparison.json) differs at native token position 443.
This single successful document control does not grade the full-set acceptance expectations.
An attempted coding analysis with the single-request diagnostic helper rejected the multi-request input, and an attempted helper import failed; the existing acceptance-analysis CLI produced the linked coding analysis without changing the outputs.
Original failed outputs and the missed preregistered T0 equality expectation remain unchanged.
Every restart gets a distinct warmup artifact, and the priority sweep collects both coding configurations before documents; these ordering changes were chosen after observing the document failures.
Run the profile-specific smoke stage with:

```bash
python scripts/run_lpx_gpu_study.py \
  --work-dir /tmp/lpx-gpu \
  --profile triton \
  --stage t0-mtp
```

The smoke stage leaves the K=3 server available for inspection and requires a separate decision to launch `--stage priority1`.
The Triton priority stage rejects a missing, capped, empty, length-limited or repetition-flagged K=3 document control.
After the [smoke stage completed](../experiments/lpx/gpu-triton-t0-status.json), the [recorded priority command](../experiments/lpx/gpu-triton-priority1-command.json) resumed collection with K=3 coding, K=7 coding, K=3 documents and K=7 documents, followed by token-ratio measurement and LP30 pricing.
This is a post-observation decision based on successful full-cap plain and MTP controls; all original failures and diagnostic caps remain separate artifacts.

## Completed Triton priority-1 results

The priority-1 runner completed at 2026-10-29 00:14:09 Asia/Shanghai after 6.796 hours.
The [K=3 outputs](../experiments/lpx/gpu-triton-mtp-k3-outputs.jsonl) and [K=7 outputs](../experiments/lpx/gpu-triton-mtp-k7-outputs.jsonl) each contain all 64 document requests and 80 coding prompts from the recorded input file.
Nine coding prompts have a follow-up turn, giving 153 turns per configuration and 306 in total.
Input identities, metadata and turn coverage match exactly, with no duplicate requests or output-cap overrides.
Every turn has finish reason `stop`, a nonempty answer, no repetition flag and valid speculative counters.
The [validation record](../experiments/lpx/gpu-triton-priority1-validation.json) stores output hashes, coverage and counter checks.
Every turn proposes K tokens per draft step, has `emitted_steps = drafts + 1` and has no preemptions.
The residual `completion_tokens - 1 - drafts - accepted_tokens` lies between -3 and 0 at both K values, within the terminal-trimming bound of -K to 0.
No turn is excluded.
Recomputing the [combined acceptance and pricing artifact](../experiments/lpx/gpu-triton-mtp-acceptance.json) from the raw outputs gives byte-identical JSON.

### Acceptance and context pairing

Equivalent acceptance is derived from mean per-turn committed tokens per decode step, `tau`.
Each document cell has 16 turns and each coding cell has 89 turns.

| K | 10K quotes | 100K quotes | 10K summary | 100K summary | Coding, thinking off |
| --- | --- | --- | --- | --- | --- |
| 3 | 0.784691 | 0.773551 | 0.753419 | 0.745948 | 0.913127 |
| 7 | 0.760633 | 0.744803 | 0.728016 | 0.720491 | 0.896233 |

K=3 document acceptance is within the expected 0.45 to 0.80 range in all four groups.
Coding acceptance exceeds every document group by at least 0.128 at K=3 and 0.136 at K=7, above the expected 0.05 advantage.
The paired difference is each book's 100K equivalent acceptance minus its 10K equivalent acceptance.
All pairs use the same book and question style, without missing contexts.

| K | Style | Paired books | Mean difference | Books higher at 100K |
| --- | --- | --- | --- | --- |
| 3 | Quotes | 16 | -0.011240 | 5 |
| 3 | Summary | 16 | -0.007673 | 5 |
| 7 | Quotes | 16 | -0.016092 | 1 |
| 7 | Summary | 16 | -0.008081 | 6 |

The K=3 paired differences meet the expected threshold of 0.03 for both styles.
The counters combine thinking and answer tokens, so separate acceptance for those parts is unavailable.
K=1 and K=5 are unmeasured.

### LP30 pricing

The new document outputs give Gemma-to-`o200k_base` ratios of 1.004662 at K=3 and 1.004687 at K=7.
Both are within 0.01 of 1.006, so the preregistered pricing rule retains 1.006.
The following rows use TP32/PP8 with DCP8 and attention matrix efficiency 0.50, with both specified drafter costs.
The interval is the common extra time per step that places both AA context speeds within 5%; an empty intersection is shown as `none`.

| K | Style | Drafter fraction | AA 10K TPS | AA 100K TPS | Common extra time, µs |
| --- | --- | --- | --- | --- | --- |
| 3 | Quotes | 0.05 | 8,375.8 | 7,603.8 | 465.0–493.9 |
| 3 | Quotes | 0.10 | 7,497.5 | 6,861.6 | 424.9–453.9 |
| 3 | Summary | 0.05 | 7,984.0 | 7,289.1 | 427.2–458.2 |
| 3 | Summary | 0.10 | 7,146.9 | 6,577.5 | 387.1–418.1 |
| 7 | Quotes | 0.05 | 8,249.5 | 6,981.5 | none |
| 7 | Quotes | 0.10 | 6,821.9 | 5,890.6 | none |
| 7 | Summary | 0.05 | 7,531.1 | 6,529.7 | 501.0–506.8 |
| 7 | Summary | 0.10 | 6,227.9 | 5,509.4 | 407.4–413.2 |

All eight measured configurations exceed AA's 10K speed by 84.1% to 147.7% without extra time.
None admits a common extra time of 25 µs or less.
This supports expectation 5 for the measured K=3 and K=7 rows; its claim over every specified K awaits K=1 and K=5.
These prices use simulated LP30 steps and measured acceptance, not measured LP30 execution times.

### Coding speed spread

The H20 timings include GPU step time and streaming behavior.
The LP30 timings use the measured acceptance with the existing context-dependent verification and drafter step model.
Each row covers the same 89 coding turns, with thinking off.
P80 denotes the 80th percentile.

| Execution | K | Drafter fraction | Median TPS | P80 TPS | Maximum TPS | P80 / median | Maximum / median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H20 | 0 | Plain | 47.58 | 47.86 | 48.14 | 1.005874 | 1.011811 |
| H20 | 3 | Released assistant | 149.04 | 158.84 | 176.40 | 1.065741 | 1.183562 |
| H20 | 7 | Released assistant | 220.54 | 251.41 | 300.98 | 1.139963 | 1.364720 |
| LP30 model | 0 | Plain | 3,764.52 | 3,764.87 | 3,765.18 | 1.000093 | 1.000177 |
| LP30 model | 3 | 0.05 | 10,400.24 | 10,981.52 | 11,611.70 | 1.055891 | 1.116483 |
| LP30 model | 3 | 0.10 | 9,300.47 | 9,820.15 | 10,383.51 | 1.055878 | 1.116451 |
| LP30 model | 7 | 0.05 | 12,832.60 | 14,639.12 | 16,764.91 | 1.140776 | 1.306431 |
| LP30 model | 7 | 0.10 | 10,583.99 | 12,074.34 | 13,827.13 | 1.140811 | 1.306419 |

All measured spread ratios are below NVIDIA's reported P80/median of 1.158 and maximum/median of 1.464.
All four MTP LP30 coding medians exceed NVIDIA's 4,767 TPS; the plain median is 21.03% below it.
The H20 K=3 P80/median lies within the expected 1.03 to 1.20 range.
The full plain control covers all 80 prompts and 89 turns, with normal completion, no empty answers, no repetition flags and zero preemptions.
Its P80/median is 1.005874, below 1.02, so both parts of expectation 8 hold for the measured configurations.
The [coverage validation](../experiments/lpx/gpu-triton-followup-plain-k0-outputs-validation.json) and [combined analysis](../experiments/lpx/gpu-triton-followup-plain-acceptance.json) retain the complete plain and MTP comparisons.
T5 results for the remaining T1 configurations are pending.

The differences from the published runs remain those in the table below: a public checkpoint, H20 hardware, BF16 weights, a released drafter, study prompts, explicit thinking settings and concurrency one.
The Triton profile was selected after observing failures in other serving paths, and original failed outputs remain available.
The measurements constrain these specified configurations without identifying NVIDIA's or AA's deployed configuration.

## Completed FP8 comparison

[`run_lpx_gpu_fp8.py`](../scripts/run_lpx_gpu_fp8.py) starts a separate K=3 Triton run with `--quantization fp8` for the target and the separately loaded, unquantized BF16 assistant.
The key/value cache remains BF16.
The startup log confirms FP8 target quantization and Triton attention.
In the installed vLLM source, `Gemma4MultiTokenPredictor` obtains the assistant's independent quantization configuration through `get_draft_quant_config`; the separately supplied assistant has no quantization setting and inherits the target's BF16 activation dtype.
The runner shares the existing exclusive GPU lock and process-ownership checks, writes a fresh capped warmup, and requires a normally completed full-cap FP8 document control before collecting the 64 document requests.
A failed startup or document control preserves its error and prevents the full sweep; failure alone does not establish that vLLM rejects FP8 MTP.
It releases its owned server on completion or error.
Outputs and precision metadata use the separate `gpu-triton-fp8-` prefix and do not overwrite BF16 artifacts.
The [launch command](../experiments/lpx/gpu-triton-fp8-command.json) and [status record](../experiments/lpx/gpu-triton-fp8-status.json) identify this attempt.
The log is `/tmp/lpx-gpu/triton-fp8.log`.
Repository validation overlaps FP8 startup, its document control and initial collection turns, so those GPU timings are not isolated performance measurements.
The [full-cap FP8 control](../experiments/lpx/gpu-triton-fp8-mtp-k3-smoke.jsonl) completes with 3,391 tokens, a nonempty answer, no repetition flag and equivalent acceptance 0.750934.
The control passes the quality gate and the 64-request document sweep completes with 64 normal stops, no empty answers, no repetition flags and zero preemptions.
The [coverage validation](../experiments/lpx/gpu-triton-fp8-mtp-k3-outputs-validation.json) confirms every prescribed document request and the [BF16 comparison](../experiments/lpx/gpu-triton-fp8-comparison.json) records source hashes and all four groups.

| Context | Style | Turns | BF16 acceptance | FP8 acceptance | FP8 minus BF16 |
| --- | --- | --- | --- | --- | --- |
| 10K | Quotes | 16 | 0.784691 | 0.777301 | -0.007390 |
| 10K | Summary | 16 | 0.753419 | 0.746485 | -0.006934 |
| 100K | Quotes | 16 | 0.773551 | 0.759189 | -0.014362 |
| 100K | Summary | 16 | 0.745948 | 0.741671 | -0.004277 |

Combined equivalent acceptance is 0.764586 for BF16 and 0.756309 for FP8, a difference of -0.008278.
Each combined value is derived from the mean committed tokens per step across the 64 turns, rather than an average of group acceptance values.
All four absolute group differences and the combined difference are below 0.03, supporting expectation 6 for this FP8 target with a BF16 assistant and KV cache.
This result does not identify the precision or drafter of the published deployment.

```bash
/tmp/lpx-gpu/repo-venv/bin/python scripts/run_lpx_gpu_fp8.py \
  --work-dir /tmp/lpx-gpu
```

## Queued follow-up measurements

This section records the original queue configuration.
All stages are complete; see the [follow-up results](lpx-gpu-followup-results.md) and [resumed validation](../experiments/lpx/gpu-triton-followup-resumed-validation.json).

[`run_lpx_gpu_followup.py`](../scripts/run_lpx_gpu_followup.py) starts after successful FP8 completion.
The [command](../experiments/lpx/gpu-triton-followup-command.json), [plan](../experiments/lpx/gpu-triton-followup-plan.json) and [status](../experiments/lpx/gpu-triton-followup-status.json) record its process and requested stages.
The log is `/tmp/lpx-gpu/triton-followup.log`.
The queue waits for the shared GPU lock and requires the FP8 status to be `complete` before starting a server.
A separate queue lock rejects duplicate follow-up runners.
The plain stage is complete and the native n-gram stage is running at the time of this update.
Remaining MTP rows and T6 collection are still queued; partial n-gram outputs do not establish full-set results.
The complete coding portion has a separate [n-gram replay comparison](#completed-n-gram-coding-comparison), while document results remain pending.
CPU-side replay checks and repository validation overlap parts of n-gram collection, so its GPU timings are not isolated performance measurements.

| Stage | Configurations | Measurement requests | Measurement turns |
| --- | --- | --- | --- |
| Plain coding control, T5 | BF16, thinking off, K=0 | 80 | 89 |
| Native n-gram and replay, T4 | BF16, K=7, document thinking on and coding thinking off | 144 | 153 |
| Remaining MTP, T1 priorities 2 and 3 | K=1/5 documents and coding in both thinking modes; K=3/7 coding with thinking and `speed-cc16k` without thinking | 688 | 742 |
| Total | Six serving configurations | 912 | 984 |

Counts exclude the fresh capped warmup and full-cap 10K document control for each configuration.
The plain run precedes T4 to validate output-bearing engine-step counters on the same Triton serving profile.
Lower-priority MTP rows follow T4.
This sequence was chosen after the completed priority-1 results and FP8 control; the preregistered expectations remain unchanged.

The runner checks exact request identities, turn counts, input hashes, precision metadata and counter identities before analysis.
Failed document controls stop the stage and release its owned server.
Full-set outputs retain all completed turns, including quality flags, and each collection gets a separate validation artifact.
Warmups, document controls and full measurements use distinct filenames and configuration labels under `gpu-triton-followup-`.
The original K=3/K=7 priority-1 outputs remain unchanged.

Successful FP8 completion triggers a 64-request coverage check and a raw-output comparison with the BF16 baseline in `gpu-triton-fp8-comparison.json`.
The comparison reports per-group differences, combined acceptance derived from mean committed tokens per step, source hashes and quality diagnostics.
The plain stage produces `gpu-triton-followup-plain-acceptance.json` with the measured plain and existing MTP coding speed spreads and both LP30 drafter prices.
The n-gram stage pins `VLLM_USE_V2_MODEL_RUNNER=1`, checks startup confirmation and validates plain/MTP engine counters before comparing against the lookup replay.
It uses the V2 matching rules described above, including repeated-token padding.
Remaining MTP stages collect K=1, K=5, K=3 and K=7 in that order and produce `gpu-triton-followup-remaining-mtp-acceptance.json` with all measured MTP groups, paired contexts and LP30 prices.
Prompt-containing exports remain outside the repository.

```bash
/tmp/lpx-gpu/repo-venv/bin/python scripts/run_lpx_gpu_followup.py \
  --work-dir /tmp/lpx-gpu \
  --stage all \
  --after-fp8
```

After inspecting a failed stage, `--stage plain`, `--stage ngram` or `--stage remaining-mtp` can resume completed request records without repeating them.
Restarted servers receive a new warmup.
The runner does not proceed to later stages after an error.
T6 graph/eager decode profiling uses a separate worker queued after successful completion of all these stages.

## Completed n-gram coding comparison

The active T4 runner collects coding before documents, so all 80 coding requests and 89 turns are available while document collection continues.
The [coding validation](../experiments/lpx/gpu-triton-followup-ngram-coding-validation.json) checks exact input coverage, BF16 target and KV precision, uncapped output, zero preemptions and native n-gram counter identities.
All 89 turns finish normally with nonempty answers and no repetition flags.
A frozen coding-only snapshot and its prompt-containing exports remain outside the repository; the validation record identifies the source file, selection and snapshot hash.
The live output file and running GPU worker are unchanged.

The [coding comparison](../experiments/lpx/gpu-triton-followup-ngram-coding-comparison.json) replays every completed coding turn with K=7 and n-gram lengths one through three against the recorded V2 GPU implementation.

| Coding metric | Native n-gram | Lookup replay | Native minus replay |
| --- | --- | --- | --- |
| Mean committed tokens per step | 1.562622 | 1.561947 | 0.000675 |
| Equivalent acceptance from mean committed tokens | 0.360232 | 0.359954 | 0.000278 |

The mean absolute per-turn acceptance difference is 0.002235 and the maximum is 0.010451.
The coding-set difference is below 0.03, supporting the coding portion of expectation 7.
The subsequent [full T4 comparison](lpx-gpu-followup-results.md#native-n-gram-versus-replay) completes the document portion; this earlier coding snapshot remains unchanged.
V2 and replay share longest-then-most-recent matching, but retain the padding, chat-template, control-token and multi-turn export differences described in [Acceptance and LP30 pricing](#acceptance-and-lp30-pricing).
Small observed differences do not establish identical native and replay algorithms or identical per-turn step counts.
The completed single-document control also passes the export/replay pipeline check, without serving as a full document-set result.

## Queued T6 profile collection

[`run_lpx_gpu_profile.py`](../scripts/run_lpx_gpu_profile.py) waits for the full follow-up queue to complete before acquiring `runner.lock`.
The separate `profile.lock` rejects duplicate profiling workers without occupying the GPU lock while earlier measurements run.
The [launch command](../experiments/lpx/gpu-triton-t6-command.json) and [status](../experiments/lpx/gpu-triton-t6-status.json) record the queued collection; its log is `/tmp/lpx-gpu/triton-t6.log`.
The initial collection completed both graph cases, then timed out during the eager 10K profiler stop request.
Both eager 10K gzip JSON traces parse completely and contain GPU events; the recovered case manifest records the original failure and source hashes.
The [resume command](../experiments/lpx/gpu-triton-t6-resume-command.json) reuses all three verified cases and completes the missing eager 100K case.
New servers disable `torch_profiler_dump_cuda_time_total`, avoiding the optional post-export aggregation table while retaining CPU/CUDA activity, Python stacks and shapes.
The initial worker exported its trace but did not produce its table before the 900-second timeout; this localizes the failure to post-export processing without proving which internal operation stalled.

The runner starts a plain BF16 Triton server with CPU and CUDA PyTorch profiling enabled, then repeats with `--enforce-eager`.
Each mode receives one fresh 128-token capped warmup and an unprofiled control paired with a profiled request at each of 10K and 100K context.
The four control/profile pairs use a maximum of 256 generated tokens per request, with thinking enabled and prefix caching disabled.
Traces include frontend events, stack information and tensor shapes and remain in unique directories under `/tmp/lpx-gpu/t6-traces-*`.
Repository manifests retain output and trace hashes, server configuration, trace event counts and streamed timing diagnostics.
Capture requires newly generated traces with GPU kernel events and stops the profiler even if request generation fails.
Resuming verifies completed-case output, trace, input and environment hashes before skipping those cases; interrupted attempts retain separate filenames.

```bash
/tmp/lpx-gpu/repo-venv/bin/python scripts/run_lpx_gpu_profile.py \
  --work-dir /tmp/lpx-gpu \
  --after-followup
```

The worker's `complete` status means collection has finished; `analysis_status` remains `pending`.
Subsequent analysis must separate prefill from decode and attribute transformer layers, vocabulary head, logits and sampling, CPU scheduling, kernel gaps and output streaming, leaving unclassified time explicit.
The [trace-analysis workflow](lpx-gpu-trace-analysis.md) provides hash-verified, explicitly selected decode windows and a tested GPU wall-time partition that preserves concurrent-category overlap and unclassified activity.
The [measured T6 report](lpx-gpu-profile-results.md) adds generation-boundary and CUDA-correlation audits, GPU partitions, aligned host/streaming scopes and an explicit instrumentation-overhead comparison.
Unclassified GPU activity and gaps remain visible; inclusive host scopes do not establish causal scheduling costs.
Stream-step timing is normalized by the actual decode-token count and reports control/profile token equality and overhead ratios.
It includes client buffering and profiler overhead and does not establish a GPU category breakdown.
GPU host costs serve as a size reference and do not transfer directly to LP30.

## Automatic continuation monitor

The original monitor delivered the T6 failure event, then exited.
The target session failed with an application error, so the user resumed work in another session on 2026-10-30.
The following commands document the previous monitor; no automatic continuation is currently armed by this recovery.

[`monitor_lpx_gpu_study.py`](../scripts/monitor_lpx_gpu_study.py) watches the FP8, follow-up and T6 collection status files every 60 seconds.
The [current monitor command](../experiments/lpx/gpu-codex-monitor-t6-command.json) records the existing Codex thread, process and executable paths.
The [original command](../experiments/lpx/gpu-codex-monitor-command.json) and archived state preserve the earlier two-worker watch.
It does not send inference requests, acquire the GPU runner lock or poll a language model while waiting.
Two matching observations of completion of all watched jobs trigger `codex queue --thread ... --message ...` for the existing session.
FP8 or follow-up completion alone does not trigger the monitor while T6 collection remains queued or active.
Failed or interrupted jobs, lost worker processes and persistently unreadable status files also trigger a continuation for inspection.
The continuation asks the agent to verify outputs, update results and checklists, recover failures safely and continue remaining work, including T6 graph/eager profiling.
It does not request a different model or a new session.

Monitor state is saved atomically to `/tmp/lpx-gpu/codex-monitor-state.json`; its current log is `/tmp/lpx-gpu/codex-monitor-t6.log`.
A separate lock prevents duplicate monitors, and a successful CLI submission is recorded as `delivered` so restarting the monitor does not submit it again.
That state confirms queue-command success, not completion of the subsequent agent turn or of the study.
Unsuccessful queue commands retry on the polling interval.
A timeout or interrupted submission is marked uncertain and requires inspecting the thread before retrying, to avoid duplicate continuations.
The detached monitor and local Codex daemon must remain available on this host; the monitor is not a machine-reboot service.

Read the current worker statuses without sending a message:

```bash
/tmp/lpx-gpu/repo-venv/bin/python scripts/monitor_lpx_gpu_study.py \
  --watch fp8=experiments/lpx/gpu-triton-fp8-status.json \
  --watch followup=experiments/lpx/gpu-triton-followup-status.json \
  --watch t6=experiments/lpx/gpu-triton-t6-status.json \
  --once
```

After launching a new long-running task, re-arm against its new status file, not the already completed pipeline:

```bash
/tmp/lpx-gpu/repo-venv/bin/python scripts/monitor_lpx_gpu_study.py \
  --thread EXISTING_SESSION_UUID \
  --work-dir /tmp/lpx-gpu \
  --codex /home/tiger/.local/bin/codex \
  --watch t6=/tmp/lpx-gpu/t6-status.json \
  --poll-seconds 60 \
  --rearm
```

Active watched statuses require a positive worker PID; terminal statuses use `complete`, `failed` or `interrupted`.
Monitor tests mock the queue command and cover stable completion, transient events, failure recovery, duplicate prevention, persistent read errors and uncertain delivery.

## RMSNorm numerical audit

A separate [CPU numerical audit](../experiments/lpx/gpu-rmsnorm-numerics.json) compares Transformers Gemma4 RMSNorm with vLLM's native IR RMSNorm on seeded BF16 inputs.
Transformers multiplies by the weight in FP32 before casting back; vLLM's native IR path casts the normalized value to the weight dtype before multiplication.
At widths 256, 512 and 5,376, stock BF16 weights produce 4,254, 8,203 and 82,612 differing elements, respectively, with maximum absolute difference 0.0625.
Promoting the same weights to FP32 gives exact agreement in those synthetic cases.
This audit does not establish that RMSNorm caused the document loops, and no such modification is applied to the serving runs.

## Validation

Black's repository-wide formatting check passes.
The completed unit-suite run passes all 759 tests, including native-reference helpers, serving-profile isolation, document gates, fresh warmups, FP8 comparison, follow-up queue locks, predecessor failures, exact coverage checks, stage cleanup, V1/V2 n-gram descriptions and completion-monitor delivery safeguards.
The resumed suite also passes all 759 tests after disabling the optional CUDA profiler table, and 100 seeded interval checks validate the host-audit union and overlap arithmetic.
All six follow-up output validations reproduce exactly, and recomputed MTP and n-gram analyses are byte-identical to the saved artifacts.
T6 tests cover profiler configuration, normalized timing, trace hashes, GPU-event requirements, verified resume, profiler cleanup and predecessor ordering before GPU lock acquisition.
Trace-analysis tests cover explicit decode-window counts, prefill exclusion, single-device activity, overlapping kernels, unclassified time, hash-bound selections and protection against overwriting source artifacts.
An additional deterministic check compares 100 synthetic interval partitions with independent integer-time occupancy counts; these checks do not supply measured profiling results.
The n-gram changes preserve the earlier MTP smoke acceptance and pricing values; the extended quality diagnostic adds periodic-span fields.
The [FP8/plain verification](../experiments/lpx/gpu-triton-fp8-plain-validation.json) recomputes exact coverage and FP8 comparison records without modifying them, reproduces the plain/MTP analysis byte-identically and confirms unchanged priority-1 raw hashes.
The initial full-suite attempt lacked `matplotlib`, which the existing report tests import; installing it in the repository environment resolved that failure.
`git diff --check` passes.

The tooling and verified inputs are saved in local commit `fcf13ac`.
The T0 results and sweep runner are saved in local commit `edab20d`.
The periodic diagnostics, n-gram comparison and completed T1 rows are saved in local commit `bca4d8a`.
Pushing those commits to `origin/lpx-context-parallel` returned HTTP 403 because the configured GitHub account does not have repository write permission.

## Expectations

| Expectation | Outcome |
| --- | --- |
| 1: Plain and MTP output equality | Missed: 5/10 pairs identical, below the expected 9/10 |
| 2: AA-document acceptance | Held on the Triton profile: K=3 document groups give 0.746 to 0.785 |
| 3: Coding acceptance advantage | Held across K=1/3/5/7: coding without thinking exceeds every same-K document group by at least 0.117012 |
| 4: Paired context-length difference | Held: K=3 means are -0.011240 for quotes and -0.007673 for summary |
| 5: LP30 pricing at measured acceptance | Held across K=1/3/5/7: 10K prices are at least 72.2616% too fast and no common extra-time interval includes 25 µs or less |
| 6: FP8-weight acceptance difference | Held: combined FP8 minus BF16 is -0.008278; all four group differences have magnitude below 0.03 |
| 7: N-gram and replay agreement | Held for both complete sets: native minus replay is +0.000278 for coding and -0.000205 for documents |
| 8: GPU coding speed spread | Held: K=3 MTP P80/median is 1.065741 and full plain is 1.005874 |

## Differences from the published runs

| Item | Published runs | GPU measurements |
| --- | --- | --- |
| Model build | AA: an updated private, pre-release build; SPEED-Bench: not stated | Public checkpoint at the revision recorded above |
| Hardware | LP30 chips in an LPX rack | One NVIDIA H20; LP30 prices come from TokenSim |
| Precision | FP8, according to press reports | BF16 and FP8 target weights measured; assistant and KV cache remain BF16 |
| Drafter and `K` | Not stated | Released MTP assistant and vLLM n-gram drafter |
| Prompts | AA's prompts unpublished; SPEED-Bench split not stated | Study book questions and public SPEED-Bench prompts |
| Thinking | Not stated | Explicit per-request setting |
| Sampling | Not stated | Temperature zero, seed one |
| Load | AA: concurrency one; SPEED-Bench: not stated | Concurrency one and maximum batch size one |

The implementation and environment checks do not change the study's conclusions about AA or NVIDIA's deployment.
