# LPX execution boundaries and placement sensitivity

## Evidence and experiment question

This study continues the [native minimum-chip search](lpx-native-minimum-search.md) without fitting unknown costs to the published endpoint speed.
The [NVIDIA long-context article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) describes compiler-scheduled execution, transfers that begin while matrix outputs are being produced, and 320-byte scheduling vectors.
The [NVIDIA deterministic-execution article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-deterministic-execution-drives-power-efficient-high-interactivity-inference-on-nvidia-vera-rubin/) describes a cycle-level schedule spanning compute, memory, and communication across the rack.
These mechanisms support distinguishing external invocation from internal instruction and pipeline latency.
Neither article supplies the numerical invocation or per-instruction latency required to calibrate those terms.

The catalog charges 0.2 microseconds for every logical operator.
That field combines an unmeasured fixed execution cost with a launch interpretation that requires revision for a compiled program.
Removing the field entirely also removes any real pipeline latency it may have approximated.
The experiment therefore retains separate controls for logical launches, additive instruction costs, one external invocation per decode pass, and communication propagation.
None of the new numerical settings is a measured LP30 parameter.

## Implemented execution hypotheses

`--execution-model logical` retains the existing analytical operator cost and its `--kernel-launch-us` charge.
`--execution-model compiled` removes that logical launch and adds `--instruction-floor-us` to each analytical LP30 operator.
The instruction setting is an additive fixed service term, despite its short command-line name.
It is not a measured instruction count or a cycle-accurate model.
Both modes retain each operator's compute and static random-access memory (SRAM) roofline service, native graph dependencies, and chip resource occupancy.
The existing terminal-output streaming refinement remains active for supported tensor-parallel groups.

`--invocation-us` adds one external invocation gate shared by every root of the whole captured decode graph.
It is charged once per subsequent token, independent of tensor-parallel and pipeline-parallel rank counts.
The gate does not occupy LP30 compute or wire resources.
All per-chip work starts after the gate while retaining its original delays and dependencies.
This option currently requires pure LP30 decode.
A heterogeneous program requires separately identified invocation domains and handoff boundaries.

`--c2c-hop-us` overrides the experiment's LP30 chip-to-chip (C2C) link latency without modifying the catalog files.
The same override applies to the aggregate-injection link and the individual C2C link class.
The [parameter evidence audit](lpx-parameter-evidence-audit.md) records a published 0.35-microsecond SRAM-to-SRAM endpoint reference at 8 GB accessible capacity.
Mapping this endpoint reference to each modeled direct-transfer phase remains a hypothesis.
It does not establish the latency of an arbitrary collective or every route across a rack.
Collective startup remains controlled separately by `--collective-launch-us`.
Interpreting both as physical measurements would require proving that their timing boundaries do not overlap.

Experiment overrides clone the device and link catalogs.
Graphics processing unit (GPU) launch costs, heterogeneous transport parameters, matrix rates, memory bandwidth, and SRAM capacity retain their existing values.
Results record all effective execution settings, source fingerprints, interpolation probes, and serial verification receipts.
The experiment result schema is version 2.
Existing default invocations retain their numerical model.

## Contiguous pipeline placement

`--pipeline-partition balanced` retains the original equal-layer allocation.
Tensor parallelism (TP) divides tensors across ranks, and pipeline parallelism (PP) assigns consecutive model layers to stages.
`--pipeline-partition capacity` searches a capacity-minimizing contiguous allocation for each TP and PP pair.
The dynamic program uses native per-layer weights, bounded local key/value (KV) cache, full global KV, tied endpoint embeddings, communication buffers, whole KV-block reservations, and the runtime's 1% free-block watermark.
For each stage it inverts the compiler's `available_blocks = raw_blocks - int(watermark * raw_blocks)` rule.
Physical KV bytes fitting SRAM alone do not guarantee that this allocator budget admits a request.
It minimizes the maximum fraction of any rank's SRAM occupied by the allocation.
Every selected allocation is compiled again and admitted through `ResidentKV` before its native decode graph is sampled.
The memory optimizer does not enumerate every speed-optimal partition.

Production attention/feed-forward (AF) configuration accepts optional `parallel.pipeline_layer_counts` in each execution domain.
The positive integer counts must match the domain's PP size and sum to the complete model layer count.
Omitting this field preserves balanced placement.
Layer ownership, component weights, KV partitions, collectives, return routes, cross-partition KV handoff, and complete-model prefill use the declared boundaries.
The experiment applies the same boundaries to its colocated attention and feed-forward network (FFN) domains.
An example for TP8 and PP11 is `[4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 3]`.
Each stage contains consecutive layers, and the ten global-attention layers retain their original layer positions.

The complete-prefill backend accepts these counts explicitly and groups its primitive operators by the stage's actual attention geometry.
Its nonuniform placement support covers dense ordinary models.
Nonuniform compressed or mixture-of-experts (MoE) execution through that aggregate backend is rejected.
The capacity solver does not supply an internal kernel workspace model or a measured SRAM safety margin.

## Results

The complete searches enumerate 236 TP/PP pairs in each execution scenario.
The balanced searches admit 29 pairs with eight-bit floating-point (FP8) weights.
The capacity searches admit 36 pairs with FP8 weights and seven with bfloat16 (BF16) weights.
All rows retain FP8 KV, BF16 activations, the existing efficiency assumptions, and a 0.5-microsecond collective startup.
Compiled scenarios charge one 1-microsecond invocation per generated-token pass.
The table reports the smallest configuration meeting 3,400 tokens per second (TPS) for every workload, or the fastest configuration when no layout passes.

| Placement | Weight dtype | Operator fixed term, microseconds | C2C assumption, microseconds | LP30 chips, TP/PP | 10K TPS | 100K TPS | Common 3,400 target | Published-target check |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| Balanced, logical execution | FP8 | 0.2 | 0.35 | 144, 16/9 | 2,788.85 | 2,713.08 | Fail | Fail |
| Balanced, compiled sensitivity | FP8 | 0 | 0.35 | 128, 8/16 | 5,545.33 | 4,991.01 | Pass | Pass |
| Balanced, compiled sensitivity | FP8 | 0.05 | 0.35 | 128, 8/16 | 4,375.75 | 4,023.17 | Pass | Pass |
| Capacity, logical catalog execution | FP8 | 0.2 | 0.05 | 112, 16/7 | 3,531.66 | 3,411.02 | Pass | Fail |
| Capacity, compiled sensitivity | FP8 | 0.05 | 0.35 | 88, 8/11 | 4,410.36 | 4,052.40 | Pass | Pass |
| Capacity, compiled sensitivity | FP8 | 0.1 | 0.35 | 112, 16/7 | 3,810.20 | 3,670.17 | Pass | Pass |
| Capacity, compiled sensitivity | BF16 | 0.05 | 0.35 | 176, 16/11 | 4,433.45 | 4,244.99 | Pass | Pass |

The published-target check requires 3,382 TPS at 10K and 3,431 TPS at 100K, alongside the user's 3,400 TPS target at 1K.
The 1K rates for the 88-chip and logical-catalog 112-chip candidates are 4,450.65 and 3,544.51 TPS respectively.
The 88-chip candidate uses counts `[4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 3]`.
The 112-chip candidate uses `[9, 9, 9, 9, 9, 9, 6]`.
The fastest layout in the compiled 0.05-microsecond scenario is 112 chips, reaching 4,458.97 TPS at 100K.
The smallest passing layout and the fastest layout are distinct objectives.

Under unchanged catalog timing assumptions, capacity placement reduces the earlier balanced minimum from 144 to 112 chips, a 22.2% inventory reduction.
Under the same compiled 0.05-microsecond hypothesis, placement reduces the balanced minimum from 128 to 88 chips, a 31.25% reduction.
These comparisons hold execution assumptions constant within each pair.
The 88-chip case falls to 3,390.21 TPS at 100K when its additive instruction term rises to 0.1 microseconds.
The result therefore does not establish 88 chips as a robust hardware requirement.
Changing weights from FP8 to BF16 also changes the conditional minimum substantially.
The actual endpoint precision remains unknown.

Each minimum is limited to the stated TP grid, contiguous positive-size stages, endpoint embedding placement, one rack, and one memory-minimizing allocation per PP count.
The capacity solver examines all contiguous cuts for memory feasibility, but the speed search does not examine every feasible partition.
Separate embedding stages, heterogeneous per-layer ownership, different head sharding, and other architectures remain outside this minimum claim.

## Prefill and complete-request validation

The configured communication and residual buffers scale with the chosen prefill chunk.
The capacity matrix below reserves the peak 102,000-token KV state and the native allocator watermark.
Every reported minimum is recompiled and admitted through the native runtime.

| Prefill chunk, tokens | Minimum chips at TP8 | Minimum chips at TP16 | Minimum chips at TP32 |
| ---: | ---: | ---: | ---: |
| 1 | 88 | 112 | 160 |
| 4 | 88 | 112 | 160 |
| 8 | 88 | 128 | 160 |
| 16 | 88 | 128 | 160 |
| 32 | 104 | 128 | 160 |
| 64 | 104 | 160 | 192 |
| 128 | 128 | 176 | 224 |
| 256 | No placement | No placement | No placement |

This matrix uses the existing conservative pair of full-vocabulary boundary buffers on every rank, plus a retained residual buffer.
It does not establish a hardware minimum for prefill workspace.
The 88-chip, 16-token configuration leaves 5,895,936 bytes on its busiest chip after persistent weights, maximum resident KV, and configured buffers.
Unmodeled internal workspace or a larger physical safety reserve could invalidate that allocation.

The 88-chip candidate executes a complete 100,000-token prompt in 6,250 native aggregate-prefill chunks of 16 tokens.
It then executes all 1,999 subsequent-token steps through the serial driver using timings sampled from the native directed acyclic graph (DAG) and releases its KV lease.
The generated-token interval reaches 4,052.40 TPS, agreeing with the decode-only search.
This full request completes in approximately 75.1 host seconds in the saved validation run.
The same configuration also executes a complete 1K prompt and 999 subsequent-token steps.
The 112-chip candidate separately checks the final four-token prefill chunk at 100K before all 1,999 generation steps.
That 112-chip probe does not execute its full prompt.

The native prefill path aggregates primitive costs and catalog collectives at each stage.
It does not use the decode vector-overlap refinement, and its elapsed simulated time is not a calibrated time-to-first-token (TTFT) prediction.
This validation establishes consistent stage ownership, request progression, configured capacity admission, and cleanup within the implemented model.

## Commands

The following commands perform independent complete searches over the existing balanced pipeline partitions.
The chosen 1-microsecond invocation and 0.05-microsecond instruction term are sensitivity values.
They are not calibrated replacements for the catalog startup.

```bash
.venv/bin/python scripts/search_lpx_native.py \
  --c2c-hop-us 0.35 \
  --verify-limit 2 \
  --output outputs/lpx-execution-20261020/logical-endpoint

.venv/bin/python scripts/search_lpx_native.py \
  --execution-model compiled \
  --c2c-hop-us 0.35 \
  --invocation-us 1 \
  --verify-limit 2 \
  --output outputs/lpx-execution-20261020/compiled-zero

.venv/bin/python scripts/search_lpx_native.py \
  --execution-model compiled \
  --instruction-floor-us 0.05 \
  --c2c-hop-us 0.35 \
  --invocation-us 1 \
  --verify-limit 2 \
  --output outputs/lpx-execution-20261020/compiled-floor-005
```

The zero-instruction-cost scenario omits unmodeled instruction and pipeline latency.
It is an optimistic execution-cost scenario and cannot establish hardware attainability.
The positive-instruction-cost scenario tests the sensitivity to that omission.

The following commands reproduce the four capacity-placement scenarios and the complete-request validation.
An identical invocation resumes completed points only when source, input, environment, and configuration fingerprints match.

```bash
.venv/bin/python scripts/search_lpx_native.py \
  --pipeline-partition capacity \
  --verify-limit 3 \
  --output outputs/lpx-execution-20261020/capacity-catalog

.venv/bin/python scripts/search_lpx_native.py \
  --pipeline-partition capacity \
  --execution-model compiled \
  --instruction-floor-us 0.05 \
  --c2c-hop-us 0.35 \
  --invocation-us 1 \
  --verify-limit 3 \
  --output outputs/lpx-execution-20261020/capacity-floor-005

.venv/bin/python scripts/search_lpx_native.py \
  --pipeline-partition capacity \
  --execution-model compiled \
  --instruction-floor-us 0.1 \
  --c2c-hop-us 0.35 \
  --invocation-us 1 \
  --verify-limit 3 \
  --output outputs/lpx-execution-20261020/capacity-floor-010

.venv/bin/python scripts/search_lpx_native.py \
  --pipeline-partition capacity \
  --execution-model compiled \
  --instruction-floor-us 0.05 \
  --c2c-hop-us 0.35 \
  --invocation-us 1 \
  --weight-dtype bf16 \
  --verify-limit 2 \
  --output outputs/lpx-execution-20261020/capacity-bf16-floor-005

.venv/bin/python scripts/validate_lpx_placement.py \
  --output outputs/lpx-execution-20261020

.venv/bin/python scripts/audit_lpx_placement.py \
  --output outputs/lpx-execution-20261020 \
  --report experiments/lpx/execution-placement-results.json
```

The [portable result bundle](../experiments/lpx/execution-placement-results.json) records every admitted point, each selected minimum, effective assumptions, source identities, workspace checks, and complete-prefill validation.
Full context probes and serial request timestamps remain in the [study output directory](../outputs/lpx-execution-20261020/).
The audit independently integrates the saved context curves, checks the entire TP/PP grid, verifies checkpoint content hashes and parameters, and validates the recorded request clocks.
The exact code and data snapshots used by the balanced and final capacity searches are saved beside their metadata in that directory.
The snapshots have different source identities because explicit stage placement and prefill support were added between those search groups.
The two validation entry points are retained under `scripts/`.

The saved audit checks 1,652 layout scenarios, 1,669 checkpoint artifacts, 136 withheld context probes, and 51 serial workload verifications.
Those serial verifications execute 2,805 requests and 4,204,695 subsequent-token steps.
An identical capacity-search invocation reuses all 239 completed points in 2.19 host seconds, compared with 504.48 seconds for the original complete search.
Its settings, source identity, rows, selected results, and serial verification results remain identical.
A later compatibility correction preserves empty default homogeneous pipeline stages in the ordinary operator backend.
The correction leaves every screening field unchanged for the 88-chip and 112-chip candidates across all three workloads, apart from source fingerprints.
The complete repository suite passes 587 tests.

## SRAM and fixed-cost robustness

The [robustness result bundle](../experiments/lpx/robustness-results.json) extends the compiled 0.05-microsecond scenario with additional per-chip SRAM reservations.
These reservations represent unmeasured runtime storage or workspace and do not change compute or communication rates.
The sweep checks TP8, TP16, and TP32, retaining the same capacity-minimizing contiguous placement and native 1% KV-block watermark.
Each selected minimum is recompiled and admitted with its smaller native SRAM budget.
Decode rates reuse the exact layouts in the saved native search.
This sweep does not establish a measured workspace requirement.

| Additional reserved SRAM per chip, decimal MB | Minimum chips, prefill chunk 1 | Minimum chips, prefill chunk 4 | Minimum chips, prefill chunk 16 |
| ---: | ---: | ---: | ---: |
| 0 | 88 | 88 | 88 |
| 5 | 88 | 88 | 104 |
| 10 | 88 | 88 | 104 |
| 20 | 88 | 104 | 104 |
| 30 | 104 | 104 | 104 |
| 50 | 104 | 104 | 104 |
| 75 | 104 | 104 | 128 |
| 100 | 128 | 128 | 128 |

Every minimum in this table meets all three published-or-user targets under the stated execution hypothesis.
The 88-chip, 16-token case loses admission after reserving 5 MB despite its 5.90 MB physical headroom in the earlier report.
The native allocator also requires free KV-block slack, so physical headroom and the available extra reservation differ.
Reducing the prefill chunk trades additional prompt processing for a smaller configured buffer requirement when TTFT is unconstrained.
Additional SRAM can also hold kernels' internal workspace, whose size remains unknown.

The 104-chip candidate receives a complete-request check with a 50 MB additional reservation on every chip and a prefill chunk of 16.
It uses TP8/PP13 with counts `[3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2]`.
The busiest rank holds 307,596,288 bytes of weights, 16,949,248 bytes of configured buffers, and 108,642,304 bytes of KV.
Its total is 433,187,840 bytes against a reduced 450,000,000-byte native budget, leaving 16,812,160 bytes after the additional reservation.
The request ingests all 100,000 prompt tokens in 6,250 prefill chunks, executes all 1,999 subsequent-token steps, and releases its KV lease.
Its generation interval reaches 4,040.66 TPS, agreeing with the saved decode-only estimate.
The complete request takes approximately 65.9 host seconds in the saved run.
This validates the modeled placement with a stated reserve allowance and does not measure a physical kernel's workspace or the LP30 instruction term.

The same analysis derives a fixed-cost boundary for the two main candidates and checks both sides with fresh native graph samples and serial requests.
Each check retains a 0.35-microsecond C2C assumption, 0.5-microsecond collective startup, and one 1-microsecond invocation per token.

| Candidate | Estimated maximum additive operator term, microseconds | Native passing probe | Native failing probe | Maximum invocation at the 0.05-microsecond operator term, microseconds |
| --- | ---: | ---: | ---: | ---: |
| 88 chips, TP8/PP11 | 0.096362 | 0.096262 | 0.096462 | 45.69 |
| 112 chips, TP16/PP7 | 0.119702 | 0.119602 | 0.119802 | 68.19 |

The 100K workload determines both boundaries at its 3,431 TPS target.
The instruction-term estimate interpolates or extrapolates the native per-step durations from the 0.05- and 0.1-microsecond searches.
Fresh native probes on either side agree with that affine estimate and verify the target transition.
All four boundary points execute five warmup requests and fifty measured serial requests for each workload.
The additional boundary checks execute 660 requests and 989,340 subsequent-token steps across twelve serial workload verifications.
The capacity sweep separately verifies 72 native admissions across its 24 chunk/reserve combinations.
The invocation budget follows from the single shared invocation gate's additive duration while holding the operator term at 0.05 microseconds.
Neither column estimates a physical instruction's duration or calibrates LP30 startup.
These boundaries identify which timing measurements can validate or rule out each conditional chip count.

```bash
.venv/bin/python scripts/analyze_lpx_robustness.py \
  --baseline outputs/lpx-execution-20261020 \
  --output outputs/lpx-execution-20261020/robustness \
  --report experiments/lpx/robustness-results.json

.venv/bin/python scripts/audit_lpx_placement.py \
  --output outputs/lpx-execution-20261020 \
  --report experiments/lpx/execution-placement-results.json
```

## Remaining boundaries

All measured-speed comparisons retain the native search's fixed token ratios of one and deterministic equal-length serial requests.
The actual endpoint token ratios, deployment layout, serving precision, instruction schedule, and invocation overhead remain unknown.
Internal prefill kernel workspace remains unverified after accounting for the configured boundary and residual buffers.
The published 100K median being slightly faster than the 10K median remains a separate modeling question.
Increasing a context-independent invocation cost can reduce their relative difference but cannot reverse their ordering when all context-dependent work is monotonic.

The existing [global-attention-only AF prototype](lpx-native-global-attention-af.md) retains its earlier operator-composition execution model.
Its results cannot be relabeled as this compiled native-DAG study.
A comparable heterogeneous result requires explicit Rubin attention ownership, LPX local-attention and FFN ownership, resource-scoped invocation boundaries, and the associated activation/KV transfers.
The present study establishes conditional pure-LPX candidates and does not infer an updated Rubin-to-LPX ratio.
