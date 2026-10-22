# Native LPX minimum-chip search

## Question and model boundary

This page records the balanced-stage baseline.
The [execution and placement follow-up](lpx-execution-boundaries.md) adds optional nonuniform contiguous stages and reduces the conditional common-target minimum to 112 chips under the same catalog timing assumptions.
That follow-up separately evaluates compiled execution hypotheses, precision sensitivity, allocator reserve, and complete prompt ingestion.

The search asks how many LP30 chips can meet the requested generation speed under the current TokenSim model.
It uses Gemma 4 31B text with FP8 weights and key/value (KV) cache, BF16 activations, one serial client, and token conversion ratios of one.
Time to first token (TTFT) is excluded.
The three input/output lengths are 1,000/1,000, 10,000/1,500, and 100,000/2,000 tokens.
The common target is 3,400 tokens per second (TPS) in every workload.
The additional published-target check uses 3,382 TPS at 10K and 3,431 TPS at 100K, while retaining the user target of 3,400 TPS at 1K.
Those two published measurements are described in the [NVIDIA endpoint article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/).

The older `calibrate_lpx_aa.py` script uses isolated operator composition.
The new search obtains its completion times from the native attention/feed-forward (AF) directed acyclic graph (DAG), with [colocated dense reuse](lpx-colocated-dense-reuse.md) and the existing [output-vector overlap refinement](lpx-compute-communication-overlap.md).
It does not introduce a separate operator latency formula or change hardware catalog defaults.

The search covers pure LPX within one 256-chip rack.
Tensor parallelism (TP) is enumerated over 1, 2, 4, 8, 16, and 32.
For each TP, pipeline parallelism (PP) covers every nonempty balanced contiguous layer partition that fits the inventory limit.
Arbitrary layer placement, separate embedding stages, external memory, multi-rack communication, speculative decoding, and mixed Rubin/LPX placement are outside this search.
Groups above TP16 retain their full catalog collective costs because multi-tray vector overlap is unsupported.
A minimum in this search is conditional on these layout and cost assumptions.
It does not identify the private endpoint's chip count or imply that a partial rack can be purchased independently.

## Capacity and numerical checks

The production runtime compiles each layout before speed estimation.
`ResidentKV` admission reserves one request at the largest context, 102,000 tokens, and rejects layouts that exceed per-rank KV budgets or physical memory.
The memory result includes native weight components, protocol buffers, the retained full residual, bounded local-attention KV, and full global-attention KV.
Each feasible point records its busiest physical rank and minimum remaining headroom.
Prefill workspace is not certified by this decode-only admission check.

Each workload samples native decode DAGs at its generation-context endpoints and midpoint.
Attention-window boundaries are explicit interpolation knots.
Midpoint disagreement above 0.0001 microseconds triggers recursive subdivision of the integer context interval.
The generation-time sum uses every subsequent-token context from `input_tokens + 1` through `input_tokens + output_tokens - 1`.
Selected candidates receive additional withheld quarter-point probes.
Probe agreement checks numerical interpolation error at sampled positions; it does not prove accuracy at every unsampled context or validate the hardware assumptions.

## Serial request verification

Selected candidates run five warm-up requests and 50 measured requests through the ordinary `AFService` admission, forward, KV commit, and release lifecycle.
Every subsequent token advances the SimPy clock using the sampled native-DAG latency curve at its own context.
Each request begins with resident prompt KV and the prefill-produced first output available.
Prefill time, request ingress, and TTFT are not simulated in this verification.
Generation TPS is `(output_tokens - 1) / (last_output_at - first_output_at)`.
The result checks each request against the integrated curve and verifies that no KV leases remain after completion.

The serial verification uses sampled completion times and does not execute the full resource DAG for every generated token.
This reduction applies to one deterministic dense client without inter-request contention.
The driver rejects concurrent requests and multi-token forwards.
Equal-length repeated requests do not provide independent observations or a statistical confidence interval.
The raw Artificial Analysis texts, native tokenizer ratios, serving precision, and actual deployment remain unavailable.

## Results on 2026-10-20

Each complete startup scenario checks 236 layouts and admits 29 layouts for all three workloads.
The baseline uses the unchanged 0.2-microsecond operator startup and 0.5-microsecond collective startup.

| LP30 chips | TP / PP | 1K generation TPS | 10K generation TPS | 100K generation TPS | All workloads at 3,400 TPS |
| ---: | --- | ---: | ---: | ---: | --- |
| 128 | 8 / 16 | 3,416.71 | 3,392.91 | 3,177.02 | Fail |
| 144 | 16 / 9 | 3,543.04 | 3,530.20 | 3,409.66 | Pass |
| 256 | 16 / 16 | 3,537.89 | 3,525.09 | 3,404.90 | Pass |
| 160 | 32 / 5 | 3,413.36 | 3,405.21 | 3,327.64 | Fail |

The conditional minimum for the common target is 144 chips with TP16/PP9.
The busiest rank reserves 220,549,632 bytes of weights, 1,059,328 bytes of protocol buffers, and 211,517,440 bytes of KV, totaling 433,126,400 bytes against 500,000,000 bytes of SRAM.
TP1, TP2, and TP4 have no capacity-feasible balanced layout within this search.
TP8 requires at least 128 chips but misses the generation target.
TP16/PP8 fails resident-KV admission, and TP16/PP9 is the smallest feasible TP16 layout.
TP32 requires at least 160 chips and retains catalog pricing without multi-tray vector overlap.
Adding PP stages increases serial transfer costs without reducing the amount of model work along one request's path.

No baseline candidate meets the additional published-target check.
The fastest 100K estimate, 3,409.66 TPS, is 21.34 TPS below the published 3,431 TPS, while exceeding the common target by only 0.284%.
Every admitted layout estimates 100K slower than 10K.
These results do not reproduce the observed context-length ordering.

### Startup sensitivity

The complete sensitivity searches retain the baseline efficiencies, bandwidths, precisions, shapes, capacity admission, and layout enumeration.

| Operator startup, microseconds | Collective startup, microseconds | Minimum for common target | Minimum for published-target check | Fastest 100K TPS |
| ---: | ---: | --- | --- | ---: |
| 0.1 | 0.1 | 128 chips, TP8/PP16 | 128 chips, TP8/PP16 | 6,752.90 |
| 0.2 | 0.5 | 144 chips, TP16/PP9 | No passing layout | 3,409.66 |
| 0.3 | 1.0 | No passing layout | No passing layout | 2,218.85 |

The lower-startup minimum uses TP8/PP16 and reaches 5,897.60 TPS at 100K.
Its fastest layout is TP16/PP9, which has more chips than that scenario's minimum.
The following additional runs vary one startup parameter at a time with TP16/PP9 fixed.

| Operator startup, microseconds | Collective startup, microseconds | 100K TPS |
| ---: | ---: | ---: |
| 0.1 | 0.5 | 5,079.12 |
| 0.2 | 0.1 | 4,090.24 |
| 0.2 | 0.5 | 3,409.66 |
| 0.2 | 1.0 | 2,822.59 |
| 0.3 | 0.5 | 2,566.18 |

These hypotheses are not confidence intervals and are not evidence that the hardware achieves the higher speeds.
The minimum count is highly dependent on unmeasured startup costs.
Measuring or constraining operator and compiled-region startup is the highest-priority evidence task before interpreting 144 chips as a robust provisioning result.

### Saved evidence and validation

The [portable results](../experiments/lpx/native-minimum-results.json) contain all admitted layouts, all seven scenarios, memory headroom, target decisions, and source fingerprints.
The full [baseline](../outputs/lpx-native-minimum-20261020/baseline/results.json), [lower-startup search](../outputs/lpx-native-minimum-20261020/lower-startup/results.json), and [higher-startup search](../outputs/lpx-native-minimum-20261020/higher-startup/results.json) retain context probes, request timestamps, and checkpoint receipts.
The [result audit](../outputs/lpx-native-minimum-20261020/audit.json) checks fingerprints, complete workload coverage, target decisions, every serialized token count, chronological request boundaries, and agreement with the earlier 256-chip result.

Baseline serial verification covers TP16/PP9, TP8/PP16, and TP16/PP16.
Each full sensitivity scenario verifies its minimum or fastest candidate.
Together these five configuration/scenario pairs cover 15 workloads, 825 requests including warm-up, 750 measured requests, and 1,236,675 subsequent-token forwards.
All 40 withheld native-DAG probes agree with the interpolation within 0.0001 microseconds.
The largest observed discrepancy is approximately 4.88e-19 seconds.
All 570 repository tests and the full Black check passed.
The baseline search and serial verification completed in approximately 375 seconds on the local host while the independent sensitivity runs were also active.
An identical invocation reused all 239 completed screening and verification checkpoints in approximately 2.14 seconds.
The [resume audit](../outputs/lpx-native-minimum-20261020/resume-audit.json) confirms that screening rows, serial rows, and the selected minimum are identical.

## Run and resume

```bash
.venv/bin/python scripts/search_lpx_native.py \
  --output outputs/lpx-native-minimum-20261020/baseline
```

The defaults enable colocated dense reuse and 320-byte output-vector overlap for supported groups.
Operator startup remains 0.2 microseconds, and collective startup remains 0.5 microseconds.
The default verification limit is four distinct candidates, chosen from both target minima, the strongest smaller failure, the fastest layout, and the earlier TP16/PP16 reference.
`--verify-limit 0` performs screening without request verification.
`--requests` and `--warmup-requests` configure the measured and warm-up request counts.
`--tp`, `--pp`, and `--max-chips` restrict the search space.
`--weight-dtype` and `--kv-dtype` accept `fp8` or `bf16`.
`--no-colocated-dense-reuse` retains the legacy local tensor protocol.
`--kernel-launch-us` and `--collective-launch-us` declare startup hypotheses without editing device files.

`ExperimentRunner` saves complete points under `points/` and resumes only matching source, input, environment, and parameter fingerprints.
`--no-resume` recomputes points.
`progress.json` records completed screening points, and `results.json` records the complete search, target checks, verification runs, and checkpoint receipts.
An interrupted point is retried on the next invocation.

The streaming collective estimate cache belongs to each sampled DAG builder.
Completed samples can be collected without retaining their native events or vector nodes in a class-level cache.
A weak-reference regression test checks this lifecycle as part of the search validation.

The following sensitivity runs change startup assumptions and retain all other model parameters.
Their bounds are exploratory hypotheses and are not measured uncertainty intervals.

```bash
.venv/bin/python scripts/search_lpx_native.py \
  --kernel-launch-us 0.1 \
  --collective-launch-us 0.1 \
  --verify-limit 1 \
  --output outputs/lpx-native-minimum-20261020/lower-startup

.venv/bin/python scripts/search_lpx_native.py \
  --kernel-launch-us 0.3 \
  --collective-launch-us 1.0 \
  --verify-limit 1 \
  --output outputs/lpx-native-minimum-20261020/higher-startup
```
