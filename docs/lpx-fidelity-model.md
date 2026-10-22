# LP30 execution, fabric, workspace and token-unit study

Study date: 2026-10-21.
This implementation follows the six modeling tasks in the [public-evidence audit](lpx-public-model-validity-audit.md).
The experiment remains a conditional pure-LPX Gemma 4 31B study.
Tensor parallelism (TP) splits work within a layer, and pipeline parallelism (PP) assigns consecutive layers to stages.
The study models attention and feed-forward networks (FFN), general matrix multiplication (GEMM), key-value (KV) state and static random-access memory (SRAM).
Throughput is reported in output tokens per second (TPS), with time to first token (TTFT) excluded.
Model floating-point operation utilization (MFU) denotes the selected fraction of the matching matrix arithmetic peak.
The study uses 8-bit floating-point (FP8) weights and KV, brain floating-point 16-bit (BF16) activations and 32-bit floating-point (FP32) scratch where specified.
Vector throughput settings use trillions of operations per second (TOPS).
Public information supports several architectural mechanisms, while operator rates, compiled instruction schedules, physical routes and the endpoint deployment remain uncalibrated.
The bundled [configuration](../experiments/lpx/fidelity-study.json) supplies illustrative assumptions and exposes each one for replacement or sensitivity analysis.
Its values were selected before the new comparison and card-count search.

## Implemented behavior

| Audit task | Implementation | Remaining evidence requirement |
| --- | --- | --- |
| Native and Artificial Analysis (AA) tokens | Separate input/output conversion ratios, native lengths, native throughput and AA-equivalent throughput; a same-text tokenizer measurement tool. | Representative endpoint prompt/output text and a matching native tokenizer revision. |
| Local and cross-tray communication | Scope-dependent propagation, explicit optional intermediate routes, directed port/injection occupancy and vector-unit reductions. | Actual LPX wiring, placement, routing, collective algorithms and achieved transfer timings. |
| Matrix, vector, switch and SRAM work | Independent services, softmax exponentials/reductions, cache and GEMM conversions, BF16 external activation traffic, Gemma scaling and logit softcap. | LP30 vector/switch throughput and matching operator microbenchmarks. |
| Shape-dependent execution costs | A tiled pipeline service model plus a serialized sensitivity; explicit fill/drain residual. | Instruction-level schedules, pipeline depths, padding, bubbles and compiler acceptance. |
| Live-set working-memory checks | Live attention/FFN/logit scratch, auxiliary weights, configurable reserve and runtime prefill/context bounds. | Compiler liveness, alignment, quantization metadata, instruction storage, physical SRAM banks and off-chip movement. |
| Conditional reproduction search | Capacity-selected integer partitions, all configured context points, sensitivity ablations, native probe checks and serial request replay. | Real endpoint configuration, text distribution, internal batching, numerical quality and device measurements. |

The generic operator-table schema and device-catalog defaults are preserved.
The new behavior requires the experiment configuration or the corresponding LP30 analytical fields.
Existing GPU configurations retain their prior behavior.

## Evidence and numerical confidence

[NVIDIA's architecture description](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) identifies distinct matrix execution modules (MXM), vector execution modules (VXM), switch execution modules (SXM), 320-byte vectors, 500 MB SRAM and 96 links at 112 Gbps per chip.
Those specifications support the accounting categories without establishing their sustained service rates.
[NVIDIA's long-context article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) describes scheduled communication, intermediate language processing unit (LPU) routing and early output.
It does not publish a matching Gemma operator schedule or endpoint chip allocation.

| Configuration field | Study value | Evidence status |
| --- | ---: | --- |
| `compute.gemm_mfu` | 0.30 | User-selected sensitivity assumption. |
| `compute.attention_mfu` | 0.10 | User-selected matrix-core sensitivity; softmax is charged separately. |
| `compute.memory_efficiency` | 0.90 | Unmeasured sustained-SRAM fraction. |
| `operator.activation_dtype` | `bf16` | Declared experiment format; weights and KV are FP8. |
| `operator.vector_bytes` | 320 bytes | Public architecture specification; its use as an analytical production tile remains an assumption. |
| `operator.vector_ops_per_s` | 20 TOPS | Unmeasured illustrative LP30 VXM rate. |
| `operator.exp_ops_per_s` | 1 trillion/s | Unmeasured illustrative nonlinear-function rate. |
| `operator.shuffle_bytes_per_s` | 10 TB/s | Unmeasured illustrative SXM rate. |
| `operator.vector_efficiency` | 0.50 | Unmeasured fraction applied to scalar, conversion and exponential services. |
| `operator.shuffle_efficiency` | 0.80 | Unmeasured fraction. |
| `operator.attention_tile_keys` | 256 | Assumed tiled attention algorithm. |
| `operator.fill_drain_us` | 0.05 microseconds | Unmeasured residual; it is not a measured instruction latency. |
| `fabric.local_us`, `middle_us`, `rack_us` | 0.35, 0.75, 1.15 microseconds | Scope hypotheses based on the references in the [parameter audit](lpx-parameter-evidence-audit.md); placement and routes require validation. |
| `fabric.local_chips`, `middle_chips` | 16, 128 | Tray size is public; contiguous index scopes are an assumed placement policy. |
| `fabric.ports_per_chip` | 96 | Public link count; the directed resource representation is an assumption. |
| `fabric.lanes_per_peer`, `p2p_lanes` | 6, 6 | Synthetic striping assumptions. |
| `fabric.lane_bytes_per_s` | 14 GB/s | Conversion from the stated 112 Gbps link rate. |
| `fabric.injection_bytes_per_s` | 1.25 TB/s | Per-direction interpretation of 2.5 TB/s aggregate bidirectional bandwidth. |
| `fabric.efficiency` | 1.0 | Ideal wire efficiency; protocol/framing losses are unmeasured. |
| `fabric.reduction_ops_per_s` | 10 TOPS | Illustrative effective reduction rate, equal to the study's VXM rate times its efficiency. |
| `memory.prefill_chunk` | 16 | Selected serial prefill chunk; larger chunks require a new memory reservation and search. |
| `memory.maximum_context` | 102,000 | Covers 100K input plus 2K reserved output. |
| `memory.tile_buffers` | 2 | Assumed reusable double buffering. |
| `memory.reserve_bytes` | 5,000,000 | Unmeasured per-chip allowance; compiler feasibility is not certified. |
| `token_units.native_per_aa_input`, `native_per_aa_output` | 1.0, 1.0 | Explicit unmeasured identity assumption. |

No reviewed public source validates the 20 TOPS, 1 trillion/s, 10 TB/s or 0.05-microsecond values for LP30.
The ablation suite includes slower and faster vector-service points to expose this uncertainty.
Rates from earlier Groq generations cannot establish these LP30 values.
The separate fabric reduction rate remains independently configurable when changing the operator rate.
The inherited BF16 matrix peak is also an assumption recorded in the parameter audit.
FP8 precision does not establish quality equivalence to the published endpoint.

## Operator timing

The [LP30 estimator](../TokenSim/operator_data/analytical/lp30.py) computes four independent services for each analytical primitive.
The MXM service uses matrix work divided by the selected dtype peak and matrix efficiency.
The VXM service sums scalar/conversion work divided by the vector rate and exponential work divided by the nonlinear rate, then divides by vector efficiency.
The SXM service uses explicit movement bytes and its independent bandwidth and efficiency.
The SRAM service uses weight, KV and external activation bytes divided by effective SRAM bandwidth.

For service values `s_i` and assumed tile count `N`, the tiled hypothesis uses `max(s_i) + (sum(s_i) - max(s_i)) / N`.
This describes a leading tile traversing the services followed by ideally pipelined tiles.
The serialized hypothesis uses `sum(s_i)`.
Both add the configured fill/drain residual and any independently configured operator startup.
The provided compiled experiment sets the older instruction floor to zero, avoiding an additional 0.05-microsecond charge on top of the new residual.
The inherited 4-microsecond native graph submission cost remains present.

FP8 GEMMs read BF16 external activations and write BF16 outputs while charging conversions into and out of their matrix format.
Attention separately counts QK/AV matrix work, five scalar softmax operations and one exponential per logit, online-state updates and KV conversion work.
The implementation also counts Gemma embedding scaling, layer scaling and the final logit softcap in decode.
These counts specify an algorithmic hypothesis; hardware fusion can change their cost.
The estimator reports every unit's service and the conversion/exponential counts for inspection.
It does not simulate cycle-level instruction issue, SRAM bank conflicts, exact producer output order or cross-layer operator fusion.

## Fabric and overlap

The [fabric refinement](../scripts/lpx/fabric.py) maps physical worker indices into local, intermediate and rack scopes.
An optional route has the form `{"source": 0, "target": 2, "via": [16]}`.
Each hop occupies source transmit ports, destination receive ports and endpoint injection resources during wire service.
Propagation releases these resources and adds delay before consumption.
Shared ports serialize concurrent rounds, and intermediate LPUs contribute their own port and injection usage.
Deterministic port assignment and default direct paths describe a synthetic topology with explicit budgets.
They do not establish that each generated path exists in the production wiring.
Multi-hop wire service assumes cut-through pipelining, with the largest link load and the sum of hop propagation delays.
Routes using intermediate chips outside the allocated inventory raise a configuration error, preventing uncounted routing chips from improving the reported result.

The collective hypothesis uses two direct phases with reduction work on VXM between them.
Early-output refinement retains dependencies on the producer and launches successive output-vector transfers.
The model conservatively holds VXM for a logical attention/FFN producer, so independent unit-level overlap inside that node is restricted.
Port-resource bundles also conservatively serialize partially shared groups of lanes.
TP32 retains whole-buffer communication because multi-tray early-output decomposition is unsupported.
These restrictions can increase latency relative to a feasible optimized compiler schedule.
The inherited uniform-output assumption can increase overlap relative to an uneven production schedule.

## SRAM and full-prefill scope

The [memory policy](../TokenSim/af/runtime/lpx_memory.py) reserves attention query/key/value activations, FP32 score tiles, online softmax state and double-buffered cache tiles.
It also reserves the maximum FFN intermediate live set or final-logit/selection live set.
Sequential layers and attention/FFN reuse the same scratch arena, so these mutually exclusive buffers use a maximum.
Additional Gemma norm/scaling weights and a configurable unused allowance are charged separately.
FP8 quantization scales and packed-layout metadata have no independently measured allocation in this study.
Their storage and any compiler-specific overhead require a justified allowance or a more detailed memory policy when those formats become known.
The capacity partitioner includes per-layer auxiliary weights and validates the selected allocation through native KV admission.
The partition search conservatively applies the largest seed-stage scratch allowance across stages of the same TP rank.
It does not optimize scratch separately for every proposed layer partition.
The boundary allowance permits every query in the configured chunk to produce logits, covering multi-request chunks as well as a single request.
An implementation that produces only one final prefill logit row can use a smaller boundary live set.
The service rejects requests exceeding the reserved total prefill query count or maximum context.

The full-prefill validation executes all 6,250 chunks of a 100K prompt at chunk size 16, with 2K output capacity reserved, then releases the KV lease.
This is a complete aggregate native simulation of prefill feasibility and state progression.
It does not execute model numerics or prove compiler acceptance.
Prefill uses the native aggregate communication backend; the detailed routed streaming refinement applies to decode.
Its simulated TTFT is outside the comparison target and should not be interpreted as a calibrated LPX TTFT.

## Token units and request replay

The [token-unit helper](../scripts/lpx/token_units.py) converts AA lengths to native lengths using separate corpus ratios and upward integer rounding.
If the output ratio is `rho_out`, a native sequence with `O` output tokens contributes `(O - 1) / rho_out` AA-equivalent intervals after excluding its initial native token.
The reported AA-equivalent TPS is native TPS divided by `rho_out`.
Input conversion also changes native attention work and KV capacity.
[AA's methodology](https://artificialanalysis.ai/methodology/performance-benchmarking) uses `o200k_base` for performance and excludes the initial received token from output speed.
The exact tokenization of the removed prefix can differ from this average-ratio approximation.

The conversion tool counts identical text with both encoders, separates input/output totals, and records a corpus SHA-256, tokenizer name and revision.
It disables special tokens and does not reconstruct an endpoint chat template.
No representative AA text corpus or native tokenizer is bundled, so this study does not claim a measured conversion.
Tokenizer tests verify counting and provenance using offline fixtures.

The serial replay performs five warm-up requests followed by 50 measured requests with one active client.
Each request advances every subsequent token, checks native context progression and releases its KV lease before the following request.
Native directed acyclic graph (DAG) context samples supply the costs through adaptive interpolation, with extra probes around attention-tile and sliding-window boundaries.
Every native curve checks quarter points against a 1e-10-second tolerance and inserts a new knot when a check fails.
The resulting segments are checked again until their guards pass.
This guard runs with verification enabled or disabled, so the verification flag does not change the predicted curve.
`guard_refinements` records the failed checks and their original errors.
The verification pass rechecks guard points outside the fitting samples and validates serial clocks.
These adaptive numerical checks do not prove a bound at every unsampled context.
The complete resource DAG is not rerun for every generated token.
Equal-length deterministic clients do not reproduce real prompt diversity, server variance, speculative acceptance, network output buffering or endpoint internal batching.

## Reproduce and configure

Run incremental ablations and sensitivity cases:

```bash
.venv/bin/python scripts/audit_lpx_fidelity.py \
  --config experiments/lpx/fidelity-study.json \
  --output outputs/lpx-fidelity-final-20261021/audit
```

Search TP1/2/4/8/16/32 and every positive PP stage count within 128 chips:
The integrated 128-chip point already passes the selected targets, so larger allocations cannot improve the minimum chip count.
The command-line interface supports extending the bound to one 256-chip rack.

```bash
.venv/bin/python scripts/search_lpx_native.py \
  --fidelity-config experiments/lpx/fidelity-study.json \
  --execution-model compiled \
  --instruction-floor-us 0 \
  --c2c-hop-us 0.35 \
  --collective-launch-us 0 \
  --pipeline-partition capacity \
  --max-chips 128 \
  --verify-limit 3 \
  --output outputs/lpx-fidelity-final-20261021/search
```

Run the complete prompt and serial decode validation:

```bash
.venv/bin/python scripts/validate_lpx_fidelity.py \
  --config experiments/lpx/fidelity-study.json \
  --tp 16 \
  --pp 8 \
  --output outputs/lpx-fidelity-final-20261021/validation
```

Measure a supplied JSON Lines (JSONL) corpus whose records each contain `input` and `output` text:

```bash
.venv/bin/python scripts/measure_lpx_tokens.py \
  --samples /path/to/representative-text.jsonl \
  --tokenizer google/gemma-4-31B-it \
  --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --output outputs/lpx-token-conversion.json
```

The measurement command requires optional `transformers` and `tiktoken` packages plus access to the pinned tokenizer files.
Copy its `token_units` object into the experiment configuration to use the measured corpus ratios.
Change memory limits when the converted native context exceeds the existing reservation.
The native-search command accepts partial configurations, and omitted sections retain their previous behavior.
The ablation command requires the `compute`, `operator`, `memory`, `fabric` and `token_units` sections, so its experiments should start from the bundled configuration.
Unknown fields, invalid rates, mismatched activation dtypes and inconsistent attention tile sizes raise configuration errors.
`ExperimentRunner` checkpoints fingerprint parameters, input data and source files so changed assumptions trigger recomputation.

## Results and interpretation

The configured search screens 180 TP/PP layouts within 128 chips and finds seven memory-feasible layouts.
Only TP16 PP8 at 128 chips passes all three selected rate thresholds under the bundled configuration.
The thresholds are the user-selected 3,400 TPS at 1K and the published 3,382 TPS at 10K and 3,431 TPS at 100K.
It is the minimum within this declared search, using a 5 MB allowance and one capacity-selected partition per layout.
The zero-reserve sensitivity and the instruction-residual sensitivities demonstrate why this conditional minimum does not establish the endpoint hardware allocation.

The [complete-prefill and serial validation](../outputs/lpx-fidelity-final-20261021/validation/results.json) uses TP16 PP8 with layer counts `[6, 8, 9, 7, 7, 10, 7, 6]`.
The 100K prompt executes all 6,250 chunks and the three decode workloads execute 54,945, 82,445 and 109,945 subsequent-token steps across their 55 total requests each.
The 128-chip peak SRAM occupancy is 440,169,038 bytes, including 5,000,000 bytes of extra allowance.
The minimum per-chip headroom is 59,830,962 bytes.

All rates in this table use the study's unmeasured identity token conversion.
TPS values describe modeled output generation, and host runtimes are recorded separately.

| Case | TP / PP | Chips | 1K TPS | 10K TPS | 100K TPS | Capacity status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Historical accounting | 16 / 7 | 112 | 6,368.50 | 6,128.15 | 4,478.95 | Decode-only admission. |
| Operator changes alone | 16 / 7 | 112 | 6,051.67 | 5,705.59 | 3,664.04 | Complete scratch remains excluded. |
| Fabric changes alone | 16 / 7 | 112 | 6,244.51 | 6,013.25 | 4,417.26 | Complete scratch remains excluded. |
| All study mechanisms | 16 / 7 | 112 | — | — | — | Rejected with the selected live scratch and 5 MB allowance. |
| All study mechanisms | 16 / 8 | 128 | 5,908.78 | 5,578.41 | 3,611.17 | Prefill and serial decode validation passed. |
| All study mechanisms | 8 / 17 | 136 | 5,174.61 | 4,864.01 | 3,069.69 | Capacity passes; 100K target fails. |

Operator accounting alone reduces the fixed 112-chip 100K rate by 18.19% relative to historical accounting.
Fabric accounting alone reduces it by 1.38% under the default direct routes.
This comparison does not imply that routing uncertainty is small for every placement or workload.
The combined 128-chip rate is 19.37% below the historical 112-chip rate, with a different allocation and capacity scope.

The zero-extra-reserve sensitivity retains all modeled scratch and auxiliary weights.
Its 112-chip TP16 PP7 point reaches 5,939.60, 5,605.86 and 3,622.65 TPS at 1K, 10K and 100K.
Its minimum headroom is only 3,827,182 bytes per chip.
The selected 5 MB allowance therefore changes the conditional capacity result; public information does not determine which allowance a compiled program needs.
The zero-extra-reserve configuration also passes the complete 100K-prefill and serial-decode validation.
This is a feasible candidate under that configuration; the timing search below uses the 5 MB allowance.

The [memory-only sensitivity](../outputs/lpx-fidelity-final-20261021/memory-sensitivity.json) screens every PP count within 256 chips at fixed TP16.
Chunk sizes 1 and 16 admit 112 chips with zero extra allowance and require 128 with a 5 MB allowance.
Chunk size 128 requires 128 chips even with zero extra allowance.
Chunk size 4096 has no feasible TP16 allocation within this SRAM-only policy.
This result does not exclude an off-chip prefill policy, a different tensor partition or another live-memory schedule.

| Sensitivity at 136 chips, TP8 PP17 | 1K TPS | 10K TPS | 100K TPS |
| --- | ---: | ---: | ---: |
| Study VXM/nonlinear rates: 20 TOPS / 1 trillion per second | 5,174.61 | 4,864.01 | 3,069.69 |
| Faster VXM/nonlinear rates: 50 TOPS / 2.5 trillion per second | 5,259.97 | 4,939.67 | 3,099.67 |
| Slower VXM/nonlinear rates: 1 TOPS / 0.1 trillion per second | 2,865.86 | 1,809.94 | 393.99 |
| Serialized unit service at the study rates | 4,553.88 | 4,106.27 | 2,096.23 |
| One lane per peer and pipeline edge, study unit rates | 4,741.46 | 4,479.36 | 2,911.89 |

These vector rates are stress assumptions without probability assignments.
The slower case shows that matrix MFU alone cannot determine the card count or output speed.
Cache conversion and softmax require execution-unit evidence as well.
The serialized case shows the timing contribution of the assumed intra-operator overlap.
The one-lane case reduces the 100K rate by 5.14% relative to six lanes under the default synthetic routes.
It does not measure the impact of a different physical topology or collective algorithm.

| Fill/drain residual at 128 chips, TP16 PP8 | 1K TPS | 10K TPS | 100K TPS |
| --- | ---: | ---: | ---: |
| 0 microseconds | 8,478.92 | 7,814.78 | 4,432.26 |
| 0.05 microseconds, study configuration | 5,908.78 | 5,578.41 | 3,611.17 |
| 0.10 microseconds | 4,534.33 | 4,337.22 | 3,046.75 |

These residuals are sensitivity settings, not confidence bounds or measured LP30 instruction costs.
The 0.10-microsecond case fails the published 100K target at the same allocation.
Selecting a residual to match an endpoint would fit an unknown parameter without independently validating it.

| Native/AA token ratio at 128 chips, applied to both input and output | 1K AA-equivalent TPS | 10K AA-equivalent TPS | 100K AA-equivalent TPS |
| --- | ---: | ---: | ---: |
| 0.8 | 7,404.81 | 7,064.71 | 4,900.78 |
| 1.0, study configuration | 5,908.78 | 5,578.41 | 3,611.17 |
| 1.2 | 4,914.46 | 4,589.11 | 2,789.16 |

These cases change native input lengths, native output lengths and the output-speed conversion together.
The 1.2 case increases the configured maximum context to 122,400 native tokens.
They illustrate the consequence of an unknown ratio and do not estimate its likely distribution.
The corpus measurement separates input and output because their real ratios can differ.

The [representative unit ledger](../outputs/lpx-fidelity-final-20261021/unit-ledger.json) inspects a TP16 shard using the study configuration.
At context 101,000, a global-attention core has 6.724 microseconds of MXM service and 10.929 microseconds of VXM service under these assumptions.
The corresponding SRAM service is 0.766 microseconds.
At context 10,750, MXM and VXM services are 0.716 and 1.163 microseconds.
These are independent unit services whose overlap follows the configured pipeline formula; summing them does not give a decode critical-path breakdown.
FP8 KV conversion is a substantial part of the modeled VXM work, explaining why an unmeasured vector rate affects long-context predictions.

The integrated 128-chip prediction remains faster at 10K than at 100K, while the published pair has the opposite ordering.
Passing a lower throughput threshold does not reproduce that context dependence or establish a match to the measured endpoint.
Token conversion can depend on prompt/output content, and the supplied corpus tool should be run separately for distinct workload distributions.

Any minimum reported by this search is conditional on these assumptions and one capacity-selected partition per TP/PP pair.
The search does not optimize every partition, physical placement, collective algorithm, precision, speculative scheme or off-chip memory policy.
Independent hardware timings and representative tokenizer measurements are required to turn this implementation into a calibrated endpoint prediction.

## Experiment provenance and validation

The [portable results](../experiments/lpx/fidelity-results.json) contain the 14 main ablations, two extra fill/drain cases, the zero-reserve case, memory sensitivities, representative unit services, all 180 search outcomes and both complete-prefill validations.
The [complete search results](../outputs/lpx-fidelity-final-20261021/search/results.json) retain native probes, adaptive guard records and request clocks.
The 128-chip passing candidate and the 88-chip smaller candidate both pass serial replay checks.
The two complete-prefill validations cover 128 chips with the study allowance and 112 chips with zero extra allowance.
Across these repeated validation configurations, 12 workload replays execute 989,340 subsequent-token steps with one client, five warm-up requests and 50 measured requests each.
Repeating the same configuration through different entry points checks consistency without providing independent hardware observations.

Independent packaging checks verify 201 artifact hashes, 78 stored workload curves, all native/AA conversions, request clocks and direct sums over every interpolated subsequent-token interval.
The saved curves contain 2,870 fitting probes, 151 guard refinements and 174 additional verification probe records.
Fitting-probe counts exclude successful guards retained only in each process's probe cache, so they do not count every native DAG execution.
The 78 curves include repeated configurations from search, ablation and validation entry points.
Every checked probe uses a 1e-10-second tolerance.
These checks validate numerical consistency of the simulation and do not establish hardware accuracy.

The source fingerprint is `2686611479adfc9af8ce401de376b995ee9d9fb2b21d57bd9c2824b412236119`.
The [source snapshot metadata](../outputs/lpx-fidelity-final-20261021/source-snapshot.json) records the archive hash, base commit, branch, Python version and inclusion of preexisting working-tree changes.
The archive contains the working sources, scripts, tests, input data and study configuration.
The complete test suite passes 602 tests, Black checks 459 Python files, and source/test/script files satisfy the 500-line limit.

The search takes 1,812.70 host seconds on this machine.
The 14 ablation point wall times sum to 4,704.51 seconds while independent groups run concurrently.
That sum includes contention and does not represent elapsed study time.
The 0.45-second ablation aggregation reuses all completed checkpoints and is not a fresh simulation benchmark.
The [verification helper](../outputs/lpx-fidelity-final-20261021/package_results.py) checks the complete artifacts and regenerates the portable result file.
