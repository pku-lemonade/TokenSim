# Conventional AF context-length experiment

## Scope and interpretation

This experiment places all attention on Rubin and all feed-forward networks (FFNs) on LPX during generation.
It uses the Gemma model and 1K, 10K, and 100K workloads from the [architecture study](lpx-aa-architecture-study.md).
The task is to test whether communication can explain 100K generation exceeding 10K generation while approaching the reference speeds of 3,382 and 3,431 tokens/s.
All results are deterministic analytical predictions without new hardware measurements or generated answers.
The baseline uses the existing scheduled-direct collective hypothesis, BF16 precision, and a 0.2-microsecond operator overhead.
Time to first token (TTFT) and prefill remain outside the generation-speed metric.

## Communication and KV ownership

Prefill runs on the Rubin pool that subsequently owns every attention layer.
All key/value (KV) caches stay on Rubin, and LPX owns only FFN weights.
The conventional layout therefore needs no initial KV transfer to LPX.
This differs from global-only attention disaggregation, where local attention moves to LPX and its caches require a handoff.

The model charges an activation round trip and the corresponding tensor-parallel broadcasts at each of the 60 layers for every generated token.
The baseline AF boundary cost is 490.397 microseconds per token at all three input lengths.
At 100K, this boundary cost is 45.9% of the 1,067.910-microsecond mean generation interval.
Adding the separately counted within-device collectives gives a total communication share of 52.3%.
Repeated AF communication cannot be amortized by generating more output tokens because its total cost grows with the number of generated tokens.

## Fixed-parameter scan

The scan evaluates 40 profiles and 6,000 feasible conventional AF profile/layout combinations.
The grid combines operator overheads of 0, 0.02, 0.05, 0.1, and 0.2 microseconds with all-link startup multipliers of 0, 0.025, 0.05, 0.1, 0.25, 0.5, and 1.
Four additional profiles change only cross-rack startup to 0, 0.25, 0.5, or 1 microseconds.
One additional profile sets cross-rack startup and operator overhead to zero while retaining catalog within-device startup.
Every profile evaluates 150 layouts under the existing whole-layer capacity, tensor parallelism, and pipeline parallelism constraints.
Parameters, precision, token conversion ratios, and allocated inventory remain fixed across lengths within each layout.

No fixed-setting layout produces higher output speed at 100K than at 10K.
The following rows show selected results with their inventories stated explicitly.
Inventories may differ between rows because the search selects layouts separately under each profile.

| Shared settings at both lengths | Rubin / LP30 chips | 10K tokens/s | 100K tokens/s |
| --- | --- | --- | --- |
| Baseline | 8 / 108 | 1,047.5 | 936.4 |
| Cross-rack startup 0.25 microseconds | 8 / 108 | 1,342.9 | 1,165.6 |
| Cross-rack startup zero | 8 / 108 | 1,399.3 | 1,207.9 |
| Cross-rack startup and operator overhead zero | 8 / 108 | 1,915.5 | 1,574.0 |
| All-link startup at 5%, operator overhead 0.02 microseconds | 16 / 96 | 5,782.3 | 3,494.1 |
| All-link startup and operator overhead zero | 32 / 256 | 12,262.2 | 5,133.3 |

The 16-Rubin/96-LP30 row is the smallest searched inventory that exceeds the user-requested 3,400 tokens/s at all three lengths under its aggressive shared assumptions.
Its 1K prediction is 6,198.4 tokens/s.
Under the same assumptions, retaining the baseline 8-Rubin/108-LP30 inventory gives only 2,684.2 tokens/s at 100K.
The zero-startup row is an idealized stress control without a hardware feasibility claim.
These results show that the existing baseline does not meet the target, while sufficiently reduced startup costs can change the conclusion within this analytical search.
They do not establish that real hardware can attain those startup values.

## A fixed stall inside generation

The baseline conventional layout can reproduce the reference 100K/10K ratio of 1.014488 by adding an identical 806.494-millisecond stall after the initial output token in every request.
The assumed 1,500-token and 2,000-token answer lengths amortize that stall differently.
The resulting speeds are only 669.94 and 679.64 tokens/s.
Setting both answer lengths to 2,000 removes the reversal.

No KV migration in this conventional layout provides a source for that hypothetical stall.
A two-endpoint fit of one shared operator overhead and one shared stall requires a negative operator overhead and is infeasible under baseline communication.
A stall before the initial output token changes TTFT and cannot explain the output-speed reversal.

## A length-dependent communication hypothesis

A separate constructed fit keeps 32 Rubin and 256 LP30 chips, the same placements, BF16 precision, all bandwidths, and a 0.02-microsecond operator overhead at both lengths.
It allows the effective startup multiplier of every link class to change with input length.

| Input length | All-link startup multiplier | Effective cross-rack startup | Predicted tokens/s |
| --- | --- | --- | --- |
| 10K | 0.1462453 | 0.292491 microseconds | 3,382 |
| 100K | 0.0580819 | 0.116164 microseconds | 3,431 |

The 100K effective startup is approximately 60.3% lower than the 10K value.
Linear interpolation of the multipliers produces increasing output speeds at every sampled 10K increment through 100K.
The increase also remains when every workload produces exactly 2,000 output tokens.
The fit therefore depends on a changing effective execution or communication cost and does not rely on answer-length amortization.
No measured scheduling, transfer overlap, or communication traces establish that this change occurs on hardware.
The model continues to add its communication and compute costs serially and does not implement a new overlap schedule.

## Reproduction and validation

The [entry point](../scripts/explore_regular_af.py) reuses the existing cost, search, fit, replay, and experiment-runner implementations.
It changes no simulator defaults or hardware catalog entries.

```bash
python scripts/explore_regular_af.py \
  --output outputs/lpx-regular-af-20261019
```

Optional arguments are `--config`, `--output`, and `--no-resume`.
The input configuration must contain the `10k` and `100k` workloads.
The runner fingerprints code and input data and resumes matching completed profiles.
The local output contains every fixed-profile layout, each profile's selected and unchanged baseline inventories, hypothetical fits, and replay samples.
The [compact evidence](../experiments/lpx/conventional-af-results.json) preserves scan totals, selected points, fitted curves, source digests, and the detailed output digest.

Each baseline and fitted endpoint was replayed with 50 serial measured requests after five warm-up requests.
The replay medians agree with the analytical speeds within numerical tolerance.
These deterministic repetitions validate event timing and do not represent independent prompt samples or measurement confidence intervals.
All 514 repository unit tests pass, including a conventional AF ownership test that verifies zero KV migration, repeated context-independent boundary costs, and decreasing speed with context under fixed settings.
The Black and whitespace checks pass.
