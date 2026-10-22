# LPX and AF context-length sensitivity

## Question and evidence status

This experiment tests numerical explanations for the published Gemma 4 31B generation speeds of 3,382 tokens/s at 10K input and 3,431 tokens/s at 100K input.
The resulting 100K/10K ratio is 1.014488, or a 1.449% increase.
The values come from the [NVIDIA benchmark article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/).
The architecture comparison is restricted to homogeneous LPX and attention/feed-forward network (AF) disaggregation.
Conventional AF moves every attention block to Rubin, while global-only AF moves the ten global attention blocks and output head to Rubin.

The results are deterministic analytical sensitivity calculations using the existing Gemma adapter, TokenSim rooflines, and communication models.
They contain no new physical hardware measurements, generated answers, or quality evaluation.
Time to first token (TTFT), prefill, and initial key/value (KV) handoff remain outside the output-speed metric.
Parameters fitted to the two published speeds demonstrate numerical possibilities and do not independently validate a mechanism or inventory.
Raw request distributions and streaming timestamps are unavailable.

## Fixed-parameter scan

The scan evaluates 55 parameter profiles and 9,761 feasible profile/layout combinations.
Each combination keeps its precision, allocated chips, tensor parallelism (TP), pipeline parallelism (PP), numerical parameters, and native/AA token conversion ratios fixed across context lengths.
The two workloads use their existing 1,500-token and 2,000-token answer-length assumptions.

| Parameter family | Values tested |
| --- | --- |
| Operator fixed overhead | 0 to 1 microseconds |
| Cross-rack startup | 0 to 5 microseconds |
| All link startup multiplier | 0 to 5 |
| All link bandwidth multiplier | 0.25 to 16 |
| Memory bandwidth multiplier | 0.25 to 16 |
| Compute multiplier | 0.25 to 16 |
| Memory efficiency | 0.1 to 1 |
| Attention compute efficiency | 0.1 to 1 |
| Global-attention-only memory efficiency | 0.01 to 1 |
| Native/AA input and output ratios | 0.8, 1, 1.2 |
| Weight and KV precision | BF16, FP8 |
| Collectives | Scheduled direct, native ring |
| Output timing window | Full generation, last 80% |

Separate profiles combine idealized fast assumptions and slow constant assumptions.
Zero startup and above-catalog bandwidth or compute are stress controls without a hardware feasibility claim.
FP8 profiles test sensitivity and do not establish preserved output quality.

There are zero fixed-setting reversals in the scan.
For the current cost composition, each decode interval is a sum of context-independent costs and nondecreasing attention costs.
A longer context therefore cannot reduce latency under the fixed assumptions above.
The scan also finds no pair with identical chip counts and changed searched TP/PP layouts that produces the requested reversal in the baseline profile.
It does not exclude other schedules or layouts outside this search space.

## Length-dependent numerical hypotheses

The following examples hold the physical inventory fixed and deliberately change one effective parameter between the 10K and 100K workloads.
Every endpoint fit targets the published ratio of 1.014488.
The operator-overhead hypotheses are applied to both devices in mixed deployments.

| Inventory | Effective operator overhead at 10K / 100K | Predicted 10K / 100K tokens/s |
| --- | --- | --- |
| 248 LP30 | 0.200000 / 0.175218 microseconds | 3,723 / 3,777 |
| 352 LP30 | 0.200000 / 0.182209 microseconds | 4,149 / 4,209 |
| Conventional AF, 8 Rubin + 108 LP30 | 0.200000 / 0.068236 microseconds | 1,047 / 1,063 |
| Global-only AF, 8 Rubin + 160 LP30 | 0.200000 / 0.076404 microseconds | 2,476 / 2,512 |

Linear interpolation of these effective overheads gives increasing speeds at the sampled context lengths from 10K to 100K.
The experiment repeats the curves with 2,000 answer tokens at every context length, and the operator-overhead reversals remain.
These values represent a hypothesis about a changing execution schedule or amortized operator cost.
The current operator shapes and measured data do not establish that increasing context causes these overhead reductions.

Changing the native/AA output ratio also produces endpoint reversals.
The required 100K ratios are 0.9174 for 248 LP30, 0.9328 for 352 LP30, 0.8813 for conventional AF, and 0.7703 for global-only AF, with the 10K ratio fixed at one.
This changes the output counting rate and native answer length.
It does not demonstrate faster native token generation.
The interpolated global-only AF token-ratio example is nonmonotonic between its endpoints.

Increasing only global-attention memory efficiency from short to long context requires large changes.
The endpoint fits require approximately 0.0552 to 0.9 for 248 LP30, 0.0772 to 0.9 for 352 LP30, and 0.0855 to 0.85 for global-only AF.
These factors approach tenfold to sixteenfold and require hardware evidence before use as an explanation.
The 248-chip efficiency interpolation also fails the intermediate monotonicity check.

Reducing cross-rack startup only at 100K from 2 to 0.9426 microseconds can reverse conventional AF at its low absolute speed.
Global-only AF cannot attain the requested ratio by changing only that startup within zero to two microseconds.
The remaining context-dependent attention penalty exceeds the removable startup cost in that layout.

## Matching both absolute published speeds

The following fits target 3,382 and 3,431 tokens/s separately.
They retain BF16 weights and KV and the listed fixed inventories.

| Inventory | Shared additional setting | Effective overhead at 10K / 100K |
| --- | --- | --- |
| 248 LP30 | Baseline communication | 0.228108 / 0.202925 microseconds |
| 352 LP30 | Baseline communication | 0.256778 / 0.238176 microseconds |
| Global-only AF, 8 Rubin + 160 LP30 | Cross-rack startup 0.25 microseconds | 0.127643 / 0.005652 microseconds |

The 248-chip fit requires an 11.0% reduction in effective operator overhead between workloads.
The global-only AF example requires an eightfold reduction of the baseline cross-rack startup and a 95.6% reduction in effective overhead between workloads.
Both examples produce gradually increasing speeds at the sampled intermediate contexts.
With baseline communication, global-only AF reaches at most approximately 3,082 tokens/s at 100K even with zero operator fixed overhead in the selected layout.
Conventional AF cannot reach either absolute published speed through these overhead fits, including the 0.25-microsecond startup profile.
These failures concern the selected layouts and assumptions and do not prove architecture-wide impossibility.

The fitted values are not evidence that LPX or AF uses the corresponding schedules.
Two aggregate measurements do not identify device count, operator overhead, communication startup, and token conversion simultaneously.

## Identical post-initial-token stalls and unequal answer lengths

The [Artificial Analysis methodology](https://artificialanalysis.ai/methodology/performance-benchmarking) specifies at least 1,500 answer tokens for 10K inputs and at least 2,000 for 100K inputs.
The original benchmark's actual output lengths are unknown.
An identical fixed stall inside each measured generation window is amortized differently by those assumed answer lengths.
The following model keeps every parameter identical across both contexts:

```text
measured_time = sum(native_decode_intervals) + fixed_post_first_token_gap
output_TPS = (native_output_tokens - 1) / rho_out / measured_time
```

| Inventory | Shared operator overhead | Shared post-initial-token gap | Predicted 10K / 100K tokens/s |
| --- | --- | --- | --- |
| 248 LP30 | 0.127427 microseconds | 145.337 milliseconds | 3,382 / 3,431 |
| 352 LP30 | 0.182406 microseconds | 107.359 milliseconds | 3,382 / 3,431 |

The stall must occur after the initial output token to enter the metric.
An equivalent stall before that token changes TTFT and cannot explain this result.
This extension adds a per-request generation-window cost that the original fixed-parameter scan does not contain.
Setting both answer lengths to 2,000 tokens removes the reversal and gives 100K/10K ratios of 0.9315 and 0.9532 for the two inventories.
The example demonstrates a measurement-amortization possibility while native decoding remains slower at 100K.
It does not establish the presence of an API stall, buffering delay, or client pause in the measured endpoint.
Within the selected AF layouts and baseline communication, matching both absolute speeds with one shared nonnegative operator overhead and one shared nonnegative stall is infeasible.

## Reproduction and validation

The experiment is isolated in [the entry point](../scripts/explore_lpx_context.py) and [the numerical helpers](../scripts/lpx/context_sensitivity.py).
It does not modify the native simulator or hardware catalog.

```bash
python scripts/explore_lpx_context.py \
  --output outputs/lpx-context-sensitivity-20261019
```

Optional arguments are `--config`, `--output`, and `--no-resume`.
The runner fingerprints source and inputs and resumes completed matching profiles.
The output `summary.json` includes every fixed scan point, fitted parameters, intermediate curves, equal-output controls, and unreachable fits.
The raw outputs are local artifacts under `outputs/lpx-context-sensitivity-20261019/`.
The [compact evidence](../experiments/lpx/context-sensitivity-results.json) preserves parameter ranges, scan totals, representative fits, input/source digests, and the detailed result digest.

All 513 repository unit tests pass, including eight context-sensitivity tests for constant-parameter ordering, hardware-copy isolation, efficiency scope, bounded fitting, changing overhead, output-window amortization, TTFT exclusion, and invalid inputs.
The Black check and whitespace checks pass.
These tests validate the numerical implementation and metric controls and do not calibrate real hardware.
