# T6 decode profiling results

Record date: 2026-10-30.
Status: the four-case observed analysis is complete; the [analysis manifest](../experiments/lpx/gpu-triton-t6-analysis.json) records verification and attribution limits.
This exploratory task measures a plain BF16 Gemma 4 31B server on one H20 with Triton attention, batch size one, thinking enabled and 256 generated tokens per request.
Each context and execution mode has an unprofiled control and a CPU/CUDA-profiled request after a separate warmup.
The [measurement record](lpx-gpu-measurements.md) gives the environment and differences from the published LP30 deployments.
Raw traces remain outside the repository; case manifests record their hashes, generated-output hashes and complete server commands.

## Collection recovery

The original worker completed the graph cases at 10K and 100K, then timed out waiting for `stop_profile` after the eager 10K request.
Its frontend and worker gzip JSON traces are complete, and the recovered worker trace contains 12,368,627 events, including 344,951 kernels.
The [recovered manifest](../experiments/lpx/gpu-triton-t6-eager-10000-result.json) retains the failure state and log hash.
Recovery verifies all trace and output hashes and the full 255-step decode sequence; it does not repeat the completed request.
The missing eager 100K case uses `torch_profiler_dump_cuda_time_total=false`, disabling vLLM's optional post-export table.
This change retains CPU and CUDA activity, Python stacks and input shapes.
The original trace export completed without the worker table appearing before the timeout; the evidence localizes the problem to post-export processing without identifying the stalled internal operation.
The resumed eager 100K request and profiler stop both complete, producing a valid worker trace with 13,084,181 events and 357,117 kernels.
The [collection manifest](../experiments/lpx/gpu-triton-t6-collection.json) now contains all four cases.
The GPU is released after collection, and the verified orphan frontend from the failed attempt is terminated separately.

## Boundaries and semantic assignments

The [trace-analysis workflow](lpx-gpu-trace-analysis.md) describes the selection format and the tested overlap-aware analyzer.
Each case has 255 CPU `execute_1_context_0` generation annotations and 255 matching GPU annotations, paired in order by their `External id`.
Chunked prefill annotations are excluded.
Decode windows run from one GPU generation scope's start to the next, retaining every intervening gap.
The final boundary includes the last GPU scope and every GPU activity correlated to a CUDA launch inside the final CPU generation scope.
This is a GPU-anchored decode interval; client delivery has separate boundaries and is reported separately.

The GPU partition uses kernel, memory-copy and memory-set intervals on the recorded device.
Its categories are assigned from the inspected trace and installed serving source:

- Model forward: CUDA correlations point to launches inside `gemma4_mm.forward`, or to the 255 full model-forward graph replays.
  This includes transformer layers, embedding, final normalization and model-forward memory activity.
- Vocabulary head: CUDA launches fall inside `aten::mm` with input shapes `[1, 5376]` and `[5376, 262144]`.
  The model runner calls this projection separately from model forward.
- Logits and sampling: CUDA launches fall inside `logits_processor.forward` or the GPU sampler's `__call__`, excluding the separately assigned vocabulary projection.
- Unclassified GPU: remaining observed GPU activity, including transfers and preparation operations outside those scopes.
- No GPU activity: wall intervals without a kernel, copy or memset in the recorded trace.
  Their cause is not inferred solely from the lack of GPU work.

The selection-audit artifacts record source hashes, generation-annotation counts, assigned event counts and unclassified activity names.
The hash-bound selection files and their generators remain beside the raw traces under `/tmp/lpx-gpu`.
The analyzer unions same-category overlap and reports cross-category overlap separately, so concurrent work is not charged twice.

## GPU partition

Cells show mean microseconds per decode step and the percentage of the selected wall interval.
The per-case `gpu-triton-t6-*-gpu-partition.json` artifacts retain every step and non-additive activity-name diagnostics.

| Mode | Context | Model forward | Vocabulary head | Logits and sampling | Unclassified GPU | No GPU activity | Mean step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Graphs | 10K | 22698.340 (92.049%) | 760.998 (3.086%) | 32.202 (0.131%) | 26.596 (0.108%) | 1140.880 (4.627%) | 24659.016 |
| Graphs | 100K | 31608.433 (94.168%) | 760.822 (2.267%) | 32.246 (0.096%) | 29.653 (0.088%) | 1134.744 (3.381%) | 33565.897 |
| Eager | 10K | 23036.234 (25.917%) | 760.238 (0.855%) | 31.851 (0.036%) | 27.434 (0.031%) | 65028.516 (73.161%) | 88884.272 |
| Eager | 100K | 31959.010 (36.401%) | 760.355 (0.866%) | 33.170 (0.038%) | 30.597 (0.035%) | 55015.039 (62.661%) | 87798.171 |

No distinct-category GPU overlap is observed in these cases.
Graph-mode vocabulary projection costs about 761 microseconds per step at both contexts, while model-forward work increases with context.
These absolute H20 costs do not directly transfer to LP30.

## Instrumentation and client timing

Client step time is streamed decode duration divided by 255; it includes buffering and delivery effects.
It is not interchangeable with the GPU-anchored interval.

| Mode | Context | Control step, microseconds | Profiled step, microseconds | Profile/control | Identical token IDs |
| --- | --- | --- | --- | --- | --- |
| Graphs | 10K | 24101.306 | 24603.899 | 1.020853 | Yes |
| Graphs | 100K | 32653.600 | 33146.177 | 1.015085 | Yes |
| Eager | 10K | 45183.876 | 88738.819 | 1.963949 | Yes |
| Eager | 100K | 44870.625 | 87278.845 | 1.945122 | Yes |

Both eager profiles nearly double client step time, while graph profiles change it by about 1.5–2.1%.
The 65-millisecond eager GPU gaps therefore cannot be treated as uninstrumented scheduling overhead.
The unprofiled graph/eager comparison also includes the execution-mode changes to compilation and kernel fusion; it does not isolate graph launch overhead alone.
These are single paired requests per case, with no repeat-run confidence intervals.

## CPU and output streaming

Separate host-audit artifacts align worker and frontend traces on the same host using their recorded `baseTimeNanoseconds` values.
Frontend timestamps are translated by the difference between the two trace bases; no remote-host clock is aligned.
The audits clip inclusive Python scope intervals to the GPU-anchored decode window, union same-group overlaps and record overlap with GPU-busy and no-GPU-activity intervals.
They cover scheduler `schedule` and `update_from_output`, worker `execute_model`, frontend output processing, streaming-response execution and HTTP send.
These are scope wall occupancies, which can contain waiting, not CPU self-times or proven critical-path costs.
HTTP send is a subset of streaming work, and frontend work overlaps GPU execution; those quantities cannot be added to the GPU partition.
The final streaming-response scope extends beyond the GPU window and is reported separately in each host audit.

Cells show inclusive scope microseconds per decode step and the fraction of the GPU-anchored wall interval.
The scheduler-gap column measures only the intersection of scheduler scopes with no-GPU-activity intervals.

| Mode | Context | Worker scheduler | Worker execute | Frontend output processing | Frontend streaming response | HTTP send | Scheduler-gap intersection, microseconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Graphs | 10K | 475.903 (1.930%) | 3213.180 (13.030%) | 164.070 (0.665%) | 333.205 (1.351%) | 70.555 (0.286%) | 24.052 |
| Graphs | 100K | 617.139 (1.839%) | 3434.335 (10.232%) | 461.719 (1.376%) | 368.788 (1.099%) | 66.938 (0.199%) | 25.213 |
| Eager | 10K | 441.516 (0.497%) | 86435.123 (97.245%) | 182.201 (0.205%) | 393.784 (0.443%) | 80.898 (0.091%) | 441.516 |
| Eager | 100K | 501.178 (0.571%) | 85291.150 (97.145%) | 514.358 (0.586%) | 494.097 (0.563%) | 78.579 (0.089%) | 473.772 |

In the eager 10K profile, worker execution scopes overlap 63,383.759 microseconds per step of the 65,028.516 microseconds without GPU activity.
Scheduler scopes overlap only 441.516 microseconds of those gaps.
The trace therefore does not support calling the entire gap scheduler time.
The final streaming-response scope ends 3,644.544 microseconds after the selected GPU endpoint for graphs at 10K, 11,418.269 for graphs at 100K, 5,344.507 for eager at 10K, and 11,413.255 for eager at 100K.
These are per-request tail differences and are not additional per-token GPU costs.

## Interpretation limits

The observed partition separates model-forward work, the large vocabulary projection, logits/sampling, GPU gaps and unclassified GPU activity.
The host audits identify where scheduler and streaming scopes overlap that timeline without assigning a causal explanation to every gap.
The PyTorch profiler reports a Python replay-stack warning during eager 100K stop; stack-derived host labels require this caveat, while CUDA correlations, recorded matrix shapes and GPU activity remain separate evidence.
Full attribution of uninstrumented CPU scheduling and client delivery costs would require less intrusive tracing and repeated controls.
The measurements serve as a size reference for a per-step cost audit and do not calibrate or validate an LP30 fixed-overhead constant.

## Validation

All four GPU partitions recompute byte-identically through the tested trace-analysis command.
All 1,020 decode-step partitions balance, and each host-scope union equals its GPU-busy plus no-GPU-activity overlap.
One hundred seeded synthetic interval checks verify the host union and overlap arithmetic.
All 759 repository tests pass, including profiler configuration and trace partition tests, and Black and `git diff --check` pass.
The collector's `analysis_status: pending` describes its collection-only responsibility; the separate analysis manifest records completed observed analysis.
