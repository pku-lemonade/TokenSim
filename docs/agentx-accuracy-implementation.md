# AgentX accuracy and simulation performance implementation plan

This document specifies the implementation work accepted after the AISimulate comparison and the additional operator-accounting review.
The implementation must improve replay fidelity, timing estimates, and central processing unit (CPU) execution cost while preserving reproducible experiments.
The implementation record below identifies completed work; the remaining sections specify planned behavior and validation.
CPU validation does not establish hardware accuracy.
Configuration names introduced here are proposed API names and are unavailable until their implementation lands.

Imported snapshot selections, live trajectory warmup, learned forward models, and the standalone offload least recently used (LRU) implementation described below include changes after v0.13.0.
AISimulate's AgentX qualification is functional; its finite 722-request memory result does not establish accuracy or memory requirements for TokenSim's continuous DeepSeek-V4-Pro workload.

## Implementation record

| Item | Completed behavior | Validation |
| --- | --- | --- |
| W1 | Default branch-bound analytical coordinates, empirical-fit sample binding, and ordinary/AF lookup provenance | Eight focused tests and all 571 tests pass; baseline contained 563 passing tests |
| W2 decode | Shared ordinary/AF histogram queries, one-launch ragged decode, strict measured-domain handling, and complete generic verification query widths | Six additional focused tests; all 577 tests pass |
| W3 | Versioned per-round mean/conditional/empirical acceptance, legacy replay, committed execution counters, and CLI configuration | Eleven focused tests, including real paged-cache execution; all 588 tests pass |
| W8 running queue | Constant-time deque removal at both ends with stable arrival ordering and the same preemption victims | Complete AgentX result and replay-file equality against the list queue; all 589 tests pass |

W1's synthetic roofline fixture uses the test GPU, BF16 attention/KV, 32 query heads, eight KV heads, head dimension 128, and a batch-1/context-256 anchor.
At batch 16/context 1024, the corrected lookup agrees with the analytical target of 83.33711058823529 us; the reviewed default returned 168.9648785036551 us.
Mixed-direction and three-axis transfers, ragged-grid interpolation, empirical fitting, policy failures, and ordinary/AF consumers have regression coverage in `tests/test_lookup_growth.py`.
W2's sixteen-context analytical step removes fifteen redundant launches per layer while retaining identical FLOPs and KV bytes.
Measured homogeneous rows retain their exact values; ragged measured rows require an exact histogram.
Generic cached prefill and mixed physical invocation fusion remain planned work requiring the backend invocation evidence specified below.
W3 uses a stateless draw keyed by worker seed, request ID, committed round, and draft position; scheduler retries do not consume random state.
The mean-2.49 test checks 20,000 draws and adjacent-round changes; conditional and empirical tests check full acceptance distributions.
Hardware acceptance probabilities and serving-error improvements remain unmeasured.
The queue-only microbenchmark alternated list/deque order across five pairs of 500 queue constructions and drains.
Median speedups were 1.44x at length 64, 1.87x at length 512, and 6.26x at length 4096.
Those figures cover queue construction and removal only; full simulator speedup has not been established.
The equivalence test compares every result field except simulator wall time and every replay file byte-for-byte at the same output path.

A cold two-lane AgentX smoke replay used the committed fixture, DeepSeek-V4-Pro, B300 DEP8, K=3, and a two-second issue window.
It completed nine requests with no unfinished requests and retained the per-worker `blake2b_round_v1` acceptance records.
The trace is a functional fixture with short outputs; it cannot establish hardware latency error.
Logs retain measured-table clamp, analytical-fallback, and missing-shape diagnostics for the remaining timing work.

## Scope, priorities, and evidence

AgentX uses the ordinary simulation engine, so shared lookup and attention fixes must cover both AgentX and other workloads.
Attention/feed-forward disaggregation (AF) shares the operator resolver and must participate in lookup regression checks.
The target hardware experiment remains DeepSeek-V4-Pro with eight B200 or B300 graphics processing units (GPUs), data parallelism (DP) and expert parallelism (EP) across eight ranks (DEP8), multi-token prediction (MTP), and CPU key/value (KV) offload.
Small dense and multi-head latent attention (MLA) cases provide independent checks of shared code.

| Work item | Priority | Required outcome | Evidence needed for activation |
| --- | --- | --- | --- |
| W0: Frozen inputs and diagnostic baseline | P0 | Reproducible workload, timing, cache, and host-cost comparisons | Local fixtures and retained experiment manifests |
| W1: Multi-axis analytical extrapolation | P0 | Correct anchor coordinates in every recursive path | CPU consistency and consumer integration tests |
| W2: Ragged attention invocation accounting | P0 | Charge work and fixed cost at the physical invocation boundary | CPU accounting tests; framework evidence for each enabled batching contract |
| W3: MTP acceptance sampling | P0 | Reproducible per-round acceptance with an explicit distribution | CPU distribution tests; hardware acceptance telemetry for fitted probabilities |
| W4: AgentX dependencies and initial state | P0 | Typed dependency triggers and reproducible snapshot import | Trace fixtures and comparison with the benchmark's pinned harness |
| W5: Whole-forward timing and conditional calibration | P1 | Deployment-bound measured timing and bounded correction modes | Validated forward observations and held-out evaluation |
| W6: Coverage and execution-regime rules | P1 | Visible domain limits, capture boundaries, and timing provenance | Synthetic coverage tests and profile metadata |
| W7: Runtime KV capacity and transfer calibration | P1 | Measured capacity import and separately validated transfer service | Runtime allocation records and directional transfer measurements |
| W8: CPU and memory optimization | P1 | Measured speed or memory improvement with equivalent simulated results | Profiles, paired benchmarks, and output comparisons |
| W9: Accuracy and performance regression suite | P1 | Repeatable release evidence across supported configurations | Independent measurement cohorts and all earlier correctness gates |
| Later: Tree models and online adaptation | Deferred | Additional estimator only if simpler models leave structured error | Sufficient deployment-matched training data and a separate evaluation campaign |

The following observations establish the investigation baseline.
The user-reported probes and independently repeated probes use different fixtures and must retain separate identities.

| Observation | Evidence status | Interpretation |
| --- | --- | --- |
| Multi-axis probe: analytical target 43.67 us; default lookup 108.23 us; bound lookup 43.67 us | Reported review probe; mechanism independently reproduced | Internal model consistency failure; no silicon error estimate follows |
| Sixteen attention buckets: 74.51 us versus 14.51 us for a single ragged call | Reported review probe; one-launch accounting independently reproduced | 60 us of additional fixed cost at a 4 us launch assumption |
| MTP mean 2.49, K=3: six sampled requests retained one acceptance value throughout 200 rounds each | Direct execution of the existing function | Strong within-request correlation and restricted acceptance support |
| 934 of 9,450 parsed child streams start before their assigned parent's recorded completion | Scan of all 393 local trace trees | Candidate dispatch/completion differences; individual harness behavior still requires comparison |
| Parent executes during seconds 0–10; child recorded at second 2 dispatches at second 10 | Minimal TokenSim replay with fixed service times | Completion-gated spawn behavior is reproducible |
| Existing B300 report: throughput -16.3%, inter-token latency p90 +36.6%, GPU prefix hits 0.947 versus 0.899 | Historical [AgentX comparison](agentx.md#comparison-with-the-inferencex-reference) | Useful baseline; it has not been remeasured for this document |

The complete trace scan used the local dataset described in the [AgentX guide](agentx.md#install-and-obtain-traces).
Future retained experiments must verify the documented dataset checksum and save the parse/protocol version.
The 934 streams comprise 600 explicit subagent streams and 334 inferred flat streams.
These counts describe parser output, not a count of proven hardware-performance errors.

## W0: Establish comparable experiments

Extend [ExperimentRunner](../TokenSim/experiments/runner.py) and its fingerprints to run the cases below.
Reuse the existing result export, replay logs, and [profiler comparison](../TokenSim/profiler/compare.py) infrastructure.
Avoid adding a separate experiment cache or a competing result format.

Every experiment manifest must record the code revision and dirty-diff hash, dataset checksum, model and hardware configuration hashes, operator/profile hashes, backend revision, scheduler settings, both seeds, and protocol version.
It must also record each lane's selected tree and snapshot cut, the shared corpus cursor, warmup policy, idle caps, issue window, grace period, cancellation timeout, and metric definitions.
A measurement identity includes attention, mixture-of-experts (MoE), and communication kernel choices, precision, graph configuration, and the complete parallel topology.
Saving an input JSON without its referenced artifact hashes is insufficient for resuming a comparison.

Maintain three independent test layers:

| Layer | Controlled input | Compared output | Purpose |
| --- | --- | --- | --- |
| Arithmetic and accounting | Synthetic operator work, tables, and fixed service times | Latencies, work totals, dependency events, allocation invariants | Identify implementation errors without hardware uncertainty |
| Forward prediction | Recorded scheduled batches from hardware | Device-scope prediction versus the matching measured scope | Evaluate timing accuracy without changing scheduling |
| Serving replay | Matched workload selections and serving configuration | Request latency, throughput, batch composition, cache behavior, failures | Evaluate interactions among replay, scheduling, and timing |

Retain the original baseline and produce a cumulative result after each behavior change.
Run isolated ablations for W1, W2, W3, and W4 as well as their combined result.
Closed-loop request populations can change when service times change; compare source identities and completion/cancellation cohorts alongside aggregate metrics.
Only the fixed observed-batch evaluation permits a direct per-step error comparison across changed timing models.

Add a compact iteration record before implementing W5.
Capture actual scheduled request work, phase, rank, graph descriptor, padding, predicted duration, and cache/load state at the existing scheduling boundary.
Detailed records must be optional and streamed with bounded buffering.
Benchmark diagnostic collection separately and verify that enabling it leaves simulated outputs equivalent.
Full hardware traces must remain outside version control; commit small attributed fixtures and artifact manifests.

## W1: Correct multi-axis extrapolation

The affected implementation is [OperatorLookup](../TokenSim/operator_data/lookup.py).
Its current default leaves outer-axis coordinates at the target while descending into an anchor branch.
For a nonseparable analytical model, inner growth and outer growth can then overlap.
The ordinary [operator backend](../TokenSim/latency/operator_table/backend.py) and [AF resolver](../TokenSim/latency/operator_table/resolver.py) currently inherit this behavior.
Compressed attention already requests `bind_analytical_axes=True`.

For a measured anchor `a`, target `q`, analytical estimate `A`, and measured latency `M`, the single-anchor contract is:

```text
L(q) = M(a) * A(q) / A(a)
```

Each recursive child must receive the full coordinates of the branch it represents.
Each analytical growth ratio must connect that resolved branch to its parent query.
For pure analytical extrapolation from one anchor, the ratios must telescope to the equation above, including when axes move in different directions.
For multiple anchors, preserve the declared interpolation order and blending rules; general ragged interpolation is not required to be axis-order invariant.

Implementation requirements:

- Make bound analytical coordinates the default after auditing all `OperatorLookup` construction sites.
- Reuse the existing option and recursion; avoid a second lookup implementation.
- Bind lower, upper, and boundary branch coordinates consistently.
- Audit `_fit_scaling_exponent()` as well: its recursive evaluation currently passes the unmodified key for each sampled outer-axis value.
- Preserve exact measured values, source identities, policy limits, and discrete-key validation.
- Preserve existing `none`, `hold`, and plain `scale` semantics where analytical coordinates are unused.
- Keep unbound behavior available only for explicit legacy reproduction if compatibility requires it; ordinary callers must use corrected behavior.
- Include lookup semantics in saved provenance and experiment fingerprints so an old cached result cannot be reused as a corrected result.

Audit callers in `latency/attention`, `latency/moe.py`, `latency/residual.py`, `latency/operator_table`, and `operator_data/coverage.py`.
Communication lookups may supply a scaler per query; include them even when construction has no analytical scaler.
Resolve any correction effect before fitting W5 calibration, so a fitted multiplier cannot conceal this arithmetic error.

Required tests cover two-axis and three-axis extrapolation, mixed interpolation/extrapolation, both bounds, ragged grids, empirical fallback, invalid analytical estimates, and policy-limit failures.
Use a nonseparable analytical function with fixed cost and coupled work; a separable power law alone would miss the defect.
For single-anchor fixtures, assert `M(a) * A(q) / A(a)` at floating-point tolerance and verify that the input key is unchanged.
Add consumer tests proving ordinary and AF queries reach the corrected lookup.
Existing compressed-attention fixtures must retain their already-bound behavior.

Optimization during W1: retain shared package indexes and existing caches, and avoid evaluating the same analytical coordinate repeatedly within one query.
Measure allocation cost before introducing any additional cross-query cache.
Any cache key must include the analytical identity, lookup policy, and all coordinates that affect the estimate.

## W2: Price ragged attention at invocation boundaries

The ordinary backend groups requests by query/context lengths and sums independently priced attention groups.
Each analytical group includes its own launch cost.
When the target backend executes those groups in one variable-length attention invocation, this composition repeats fixed costs.
Grouping work for estimation does not establish a physical kernel boundary.

Reuse the existing [histogram schema](../TokenSim/operator_data/schema.py), [ragged work calculation](../TokenSim/operator_data/ragged_attention.py), and [AF attention provider](../TokenSim/af/providers/attention.py).
The AF provider already selects `ragged_generation_attention` for heterogeneous ordinary decode contexts.
Compressed attention has its own measured-module composition and remains a separate consumer of shared shape utilities.

Introduce one shared attention invocation descriptor containing the phase, per-request query length, past KV length, count, geometry, precision, window, and execution mode.
Canonicalize ordering once and retain exact lengths before any optional approximation.
Represent past KV and total attended KV separately to prevent an off-by-one conversion between helpers.
An execution descriptor records whether prefill and decode share an invocation or execute separately.

Implementation sequence:

1. Route heterogeneous ordinary decode through `ragged_generation_attention` using the existing analytical work helper and exact measured rows where present.
2. Preserve the homogeneous measured-table path and its exact values.
3. Extend shared work calculation to cached ragged prefill using `(query_length, past_kv_length, count)` tuples.
4. Enable mixed and speculative-query invocation composition only for a verified backend execution contract.
5. Use the shared descriptor in AF where the same physical boundary applies, including its currently grouped prefill path.

The initial analytical contract sums compute and memory work over requests, applies the bottleneck calculation once per invocation, and charges that invocation's fixed cost once.
It must preserve attention masks, sliding windows, grouped-query geometry, and KV precision.
A speculative verification query with width greater than one cannot be relabeled as independent ordinary decode requests without a supported work model.

Measured homogeneous rows already contain an opaque execution scope.
Subtracting a guessed launch cost from each measured row does not create a measured ragged kernel.
Use exact ragged measurements where available; classify any transfer from homogeneous measurements as an explicit estimate with separate provenance and held-out validation.
Under `table_only`, an absent exact ragged row must remain a structured coverage failure.
Under `table_first`, the analytical fallback remains visible.

Required tests:

- Splitting one analytical invocation into more histogram bins leaves its fixed-cost count unchanged.
- Histogram ordering and duplicate-bin merging preserve identical work and estimates.
- Work totals match independently calculated floating-point operations (FLOPs) and KV bytes for short, long, and windowed contexts.
- A one-bin invocation agrees with the corresponding homogeneous analytical path.
- Empty work, malformed counts, query lengths exceeding total contexts, and incompatible geometry fail at the boundary.
- Separate physical invocations retain separate fixed costs.
- Cached prefill and MTP tests preserve exact query widths and selected-logits ownership.

Optimization during W2: build and canonicalize the histogram once per physical invocation and share immutable descriptors across layer-equivalent queries.
A histogram-based path must not perform one whole-module lookup per distinct context length.
Retain bounds on descriptor and estimate caches.
Do not increase context bucketing merely to obtain more cache hits without measuring the induced error.

## W3: Model MTP acceptance per committed round

The current [acceptance function](../TokenSim/llm/paged_tokens.py) samples between the floor and ceiling of a configured mean using a linear expression in request ID and generated position.
This restricts the distribution and produces strong correlation across successive rounds.
The reference [conditional sampler][ais-speculative] provides a suitable probability contract.

Support an explicit acceptance model with mutually exclusive inputs:

| Proposed mode | Input | Behavior |
| --- | --- | --- |
| `conditional` | K probabilities in [0, 1] | Accept draft positions sequentially until rejection |
| `empirical` | Distribution of accepted lengths from 1 through K+1 | Sample a retained deployment/workload observation distribution |
| `mean` | Expected accepted length in [1, K+1] | Retain the documented two-point approximation with corrected per-round random sampling |
| `legacy` | Existing mean and seed | Reproduce the reviewed implementation during migration only |

For conditional probabilities `p1 ... pK`, the untruncated expected output length is `1 + p1 + p1*p2 + ... + p1*...*pK`.
Record proposed tokens, verified tokens, accepted tokens before output clipping, and emitted tokens separately.
Output truncation and model-context limits must not change the measured acceptance distribution retrospectively.
Verification and draft cost continue to follow scheduled work, independently of accepted output progress.

Use a local reproducible random stream keyed by seed, stable request-instance identity, and committed verification-round ordinal.
A counter-based generator or a retained per-request generator can implement this contract; avoid Python's process-randomized `hash()`.
Specify the generator and sampling version in provenance.
Scheduling probes, failed allocation, repeated estimation, and canceled uncommitted work must not advance the stream.
Preserve the pending sample when a schedule decision is retried.
Resumption after KV recomputation must retain the committed-round ordinal.

Required tests include all-reject/all-accept boundaries, conditional mean and distribution, nondegenerate adjacent-round behavior, output truncation, token-budget limits, retries, cancellation, recomputation, and seed repeatability.
Use deterministic large-sample CPU tests with statistical tolerances derived from sample count.
Do not assert hardware ITL values in unit tests.
Collect hardware acceptance statistics by phase, workload class, and context/load regime before fitting deployment-specific probabilities.
Treat the existing value 2.49 as a mean observation, with no implied conditional distribution.

Optimization during W3: draw only for a committed round, avoid reseeding a generator per token, and retain constant-sized state per live request.
Benchmark the sampling cost separately from any change in the number of simulated rounds.

## W4: Preserve dependency triggers, snapshots, and phase boundaries

Extend [Branch and graph models](../TokenSim/workload/agentx_support/models.py) to express request-level dependency edges with `source`, `target`, `trigger`, `delay`, `relation`, and source evidence.
Supported triggers are dispatch and completion; relations include sequence, spawn, join, and inferred frontier.
A target with several dependencies is eligible only when all required edges are satisfied and their deadlines have elapsed.
Avoid encoding this state solely as a root turn index.

The [AISimulate Weka importer][ais-weka] chooses dispatch-triggered spawn when the child's recorded start overlaps the parent request.
Implement that interpretation as a versioned protocol candidate and compare it with the benchmark's actual AIPerf revision.
Keep the current protocol selectable for historical reproduction.
A difference from AISimulate alone does not authorize relabeling either protocol as a hardware-matched default.
Promote the matched protocol only after retained harness evidence establishes the relevant dispatch, join, idle-cap, and cancellation semantics.

Build dispatch and completion adjacency indexes once per immutable source graph.
Maintain unresolved edge counts and deadline events for each live play.
Release only the dependents of the changed request.
Use stable lane, stream, and event ordinals for simultaneous events; preserve documented tie ordering during optimization.
Include background children, rootless retained frontiers, nested/inferred streams, and children surviving root completion.
Preserve original source request identities and isolate cache identities across recycled plays.

Add snapshot export/import with dataset and graph hashes, lane order, source tree ID, absolute source cut, retained frontier, completed boundaries, primer selections, and shared corpus cursor.
Specify the source-clock origin and time unit; AISimulate's imported Weka cuts are play-relative and require conversion before comparison.
Matching seeds is insufficient because sampling algorithms and rejected selections can differ.
Validate imported dependencies and identities before mutating live replay state.
Use [AISimulate's later snapshot interface][ais-snapshot] as a reference for explicit selections and cursor state.

Retain TokenSim's live trajectory warmup and document it as a protocol property.
The v0.13.0 repeated-prefix policy is a separate workload preparation procedure.
No profiling request may issue until all required preparation requests and server work settle, including pending KV transfers and reservations.
A failed preparation request must invalidate preparation according to the selected harness contract; silent advancement into profiling is unacceptable.
Reset measured counters while retaining cache and request identities.
Keep source-clock offsets, global phase anchoring, idle caps, and quota accounting observable.

Required tests include the 0/2/10-second overlap example, completion-triggered spawn, simultaneous events, multi-child joins, warmup barriers, recycled lanes, and timer preservation.
Add differential fixtures against a pinned harness with identical injected service times so timing estimation cannot explain event-order differences.
Retain failures and censored requests in the accounting ledger while excluding them from successful-request latency summaries.
Measure request latency, time to first token (TTFT), per-request inter-token latency (ITL), and trajectory makespan with explicit populations and denominators.

Optimization during W4: replace repeated full-play scans only after the event sequence is specified and tested.
Reclaim terminal mutable play state when no callback or log writer references it.
Keep the immutable corpus shared and retain compact lifecycle evidence instead of completed request object graphs.

## W5: Add whole-forward timing and conditional correction

### Scheduled-work contract

Implement a `ForwardLatencyBackend` behind the existing [LatencyBackend interface](../TokenSim/latency/base.py) and [factory](../TokenSim/latency/__init__.py).
Use the existing worker and [DP coordinator](../TokenSim/llm/dp_coordinator.py) boundaries to construct one immutable scheduled-step snapshot.
Extend the existing [ForwardPass/RankShape](../TokenSim/llm/forward.py) representation where appropriate; share the shape extraction used by operator and MTP estimators.
Do not build a second scheduler inside the backend.
Add an optional `set_step_context(snapshot)` hook with a default no-op implementation so existing latency backends retain their public estimate interface.
Every participating worker must receive the same immutable group snapshot before any synchronized estimate is evaluated.
Adapt the existing `set_moe_step()` integration to consume the same snapshot without charging or constructing group work twice.
Distinguish device-work duration from rank waiting and group elapsed duration in the returned diagnostics.

| Snapshot field | Required semantics |
| --- | --- |
| Request identity | Stable instance identity for audit; excluded from predictive numeric features |
| Phase | Prefill, decode, recompute, locally mixed, or mixed across ranks |
| Query length | Actual target positions executed, including verification and declared placeholders |
| Past KV length | Materialized context before this invocation, per request |
| Output progress | Accepted/emitted lengths for accounting; unavailable future acceptance cannot be a prediction input |
| Logits rows | Explicit rows owning sampling/output work |
| Padding | Valid rows, executed rows, capture descriptor, and dummy-rank work |
| Parallel work | All participating DP ranks, attention geometry, expert topology, and worker role |
| Execution identity | Eager/graph mode, attention/MoE kernels, target/draft configuration, and measured scope |

Waiting requests are excluded from device-work features.
Measured wall time is a target, never an input feature.
Preserve per-request query/KV lengths even when a model also uses sums, maxima, and variances.
For DEP8, retain the full rank vector; an independent local batch estimate cannot infer the synchronized expert workload.
Distinguish a locally mixed rank from prefill-only and decode-only work on different ranks.

### Profile identity and storage

Store observations in a versioned Parquet artifact with a metadata sidecar and content hashes.
Use explicit units and validate them at import.
Retain original observations, repetition identity, sample selection, instrumentation status, and exclusion reasons.
Summaries must reference the raw samples that produced them.

The deployment identity must include model/checkpoint revision, architecture/config hash, GPU SKU and relevant clock/power configuration, framework revision, kernel revisions, and weight/activation/KV precision.
It must include tensor parallelism (TP), pipeline parallelism (PP), DP, EP, context parallelism, topology, worker role, scheduler limits, chunking, graph policy/capture sizes, and draft model/depth.
Transfer and offload settings belong to the environment identity when they influence the measured scope.
Keep donor hardware/framework provenance separate from the requested target identity.
Reject conflicting duplicates, unknown schema versions, nonfinite timings, invalid dimensions, and missing scope declarations.

### Timing ownership

Use the [whole-forward performance model (FPM) contract][ais-fpm] as a reference for measurement identity and supported domains.
Represent scope explicitly; a latency number without a boundary description cannot enter the runtime.

| Scope | Included work | Runtime ownership |
| --- | --- | --- |
| `target_device_forward` | Declared target model operations and internal communication | Replace those operator terms; retain separately modeled draft/output work only when excluded by the measurement |
| `speculative_device_round` | Target verification, sampling, and draft work as declared | Replace the complete matching device round; suppress duplicate target/draft composition |
| `worker_iteration_wall` | Device work plus a declared host execution boundary | Admit only after host overlap and wait ownership are implemented and tested |
| `dp_iteration_device` | One synchronized device iteration over the recorded DP/EP group | Coordinator consumes one group duration and records its cost once |

Request queueing, authored agent delays, and external KV transfers remain separately modeled unless a supported scope explicitly owns them.
Cross-rank collective waits must follow one defined timing scope and cannot also be added as payload service.
Expose base prediction, correction, selected scope, and separately charged work in diagnostics.
GPU-device measurements must never be corrected using profiler-perturbed host wall times without a qualified boundary conversion.
Existing `step_overhead_us` and kernel launch floors require an ownership audit when a measured scope replaces their operations.

### Estimation modes

Provide two initial modes in the same backend configuration.

- `measured`: exact forward rows and supported interpolation within one deployment identity, phase, and execution regime.
- `corrected_operator`: the existing operator prediction multiplied by a frozen, condition-specific factor trained on measurements of the identical scope.

Measured mixed rounds require mixed measurements or an explicitly validated model.
Adding independent measured prefill and decode times does not establish a measured mixed-round estimate.
Query-length lists and graph padding must remain available when selecting or evaluating a profile.
An autoregressive measurement cannot supply a speculative round merely by changing its label or multiplying by query width.

For conditional correction, begin with a positive robust ratio per phase, executed-token band, KV-work band, and graph regime.
Fit ratios from `observed_scope_time / base_scope_prediction`, preserving the base model/profile hash.
Store sample count, support bounds, fit dispersion, and validation error for every cell.
Define the initial factor as the median positive ratio within each supported cell and store its interquartile range.
Define bucket boundaries from the training set and freeze them before evaluation.
Require a configured minimum sample count; sparse or unseen conditions follow the declared coverage policy.
Apply one correction at the declared scope, without altering measured operator rows or stacking hidden corrections.
An additional host offset requires separately measured host evidence and an explicit timing owner.

Split training and validation by collection run or held-out workload/configuration, keeping repeated samples of the same execution together.
Report error on unseen batch shapes and concurrency points, including long-context and mixed cases.
Use chronological prediction-before-update evaluation if online adaptation is added later.
Tree models are deferred until the measured and corrected-operator modes leave reproducible residual structure on held-out data.

### Integration and support stages

Add validated forward-profile configuration in a dedicated configuration module, referenced by `SimulationConfig` and the CLI adapters.
Expose one proposed backend selector, `latency_backend_type: forward`, with profile path, profile hash, mode, and coverage policy.
Use `fallback: deny` by default for measured validation runs.
An explicitly selected operator fallback may handle a coverage hole; a corrupt profile, deployment mismatch, or incompatible timing scope must fail.
Analytical hardware exploration continues through the existing analytical/operator backend or the explicit coverage fallback, with its source retained.

| Stage | Required support | Exit condition |
| --- | --- | --- |
| Schema and offline evaluator | Pure, mixed, ragged, padded, and multi-rank snapshots | Lossless import and scope/identity validation |
| Runtime S1 | One worker, PP1, pure prefill/decode, no speculative round | Device-scope replacement and deterministic results |
| Runtime S2 | Matched mixed batches, MTP, DEP8, and dummy ranks | Full target AgentX configuration supported with measured group ownership |
| Later extensions | Additional PP, prefill/decode disaggregation (PD), AF, and host-overlap modes | Separate scope and scheduling qualification |

Runtime S1 is an intermediate checkpoint; the AgentX timing objective requires Runtime S2.
Reject unsupported combinations before a simulation starts.
The existing AISimulate speculative-round profile restrictions do not imply support for TokenSim DEP8.
AF receives W1/W2 shared fixes independently; whole-forward AF timing needs its own stage/resource contract.
PD also requires an explicit resource handoff contract before admission to whole-forward replay.

Keep implementation responsibilities in the following modules or small extensions of their existing equivalents.
New paths in this table are proposed file locations.

| Responsibility | Existing integration point | Proposed addition |
| --- | --- | --- |
| Validate profile configuration | `TokenSim/config/simulation.py`, `TokenSim/cli/configuration.py` | `TokenSim/config/forward_timing.py` |
| Extract immutable scheduled work | `TokenSim/llm/forward.py`, worker and coordinator boundaries | Shared request/rank snapshot types and the optional context hook |
| Load profiles and verify identity | Operator package validation conventions | `TokenSim/latency/forward/profile.py` and `schema.py` |
| Resolve measured rows and support | Existing lookup evidence conventions | `TokenSim/latency/forward/lookup.py` |
| Apply conditional correction | Existing operator latency backend | `TokenSim/latency/forward/correction.py` |
| Select and invoke the backend | `TokenSim/latency/__init__.py` | `TokenSim/latency/forward/backend.py` |
| Capture or import observations | Existing collection manifests and profiler boundary checks | Small collector/importer modules under `scripts/collect` |
| Export scope and coverage diagnostics | `TokenSim/reporting/export.py`, `TokenSim/results.py` | Validated result fields and documented schema additions |
| Evaluate held-out observations | `TokenSim/experiments/runner.py` | A forward-evaluation runner using the existing experiment store |

Profile loading, fitting, and collection must remain outside simulation event callbacks.
Report any profile load or validation failure before warmup starts.

Optimization during W5: prepare indexes and condition features at profile load, share immutable profile data across workers, and evaluate one synchronized group once.
Avoid Python object expansion proportional to context-token count; retain lengths and histograms.
Bound lookup caches and include profile hash, scope, all predictive features, and execution regime in cache keys.
Do not attach fabricated per-operator timings to a whole-forward measurement for visualization.

## W6: Make coverage and execution regimes explicit

Reuse `LookupResult`, `OperatorStats`, and `MissingShapeReport` while extending the evidence needed for forward and ragged estimates.
Distinguish exact measurements, supported interpolation, analytical transfer, clamp/hold, analytical fallback, and conditional correction.
Record donor deployment, complete supporting coordinates, interpolation weights, extrapolation ratios, and the selected execution regime when applicable.
Normal summaries must remain compact; detailed supporting rows can be emitted on demand.

Add a strict validation policy allowing only exact measurements and supported interpolation for measured scopes.
Do not change `table_only` into an undocumented stronger policy; expose and record the additional restriction explicitly.
Compressed attention's existing 64x extrapolation bound remains an exploration assumption until a measured domain justifies it.
Avoid silently lowering that bound for historical runs; validation mode must reject or expose such use.
Use [capture-regime-aware direct interpolation][ais-fpm-op] as a reference.
Queries crossing eager/graph transitions or incompatible kernel dispatch regions must not be blended as one smooth surface.

Report both query-count coverage and estimated-time coverage.
For estimated-time coverage, use exclusive timing ownership so fused modules, overlap regions, target/draft scopes, and parent scopes are not counted twice.
When exclusive attribution is unavailable, report the scope's coverage separately and leave the aggregate unavailable.
Coverage is a measure of support, not a confidence interval on hardware error.

Optimization during W6: accumulate fixed-size counters and bounded shape summaries during the run.
Rank collection candidates by their frequency, estimated time contribution, and distance from measured support.
Retain a representative sample of request/step identities to connect coverage gaps to tail latency.

## W7: Import runtime memory and validate transfer service

Add an observed-capacity mode alongside the existing [geometry-based memory budget](../TokenSim/block/hybrid/memory_budget.py).
Follow the identity and reservation accounting in [AISimulate runtime memory][ais-resources].
Preserve the geometry-based mode for unmeasured hardware exploration.

An observed record must contain the exact deployment identity, memory utilization setting, scheduler limits, graph and MTP settings, each KV group's page size/count/layout, and allocation-log provenance.
Reject a profile when the selected worker layout cannot interpret its groups.
Do not spread an observed aggregate byte count among heterogeneous groups by assumption.
The imported usable KV pool already includes measured graph/workspace deductions; do not subtract those deductions again.
Keep physical layout validation, speculative lookahead, allocator rounding, and reference invariants active.

Compare observed and modeled capacity before changing eviction or routing behavior.
Track per-rank/per-group residency, GPU and CPU admission hits, evictions, reloads, recomputation, transfer bytes, and queue wait.
Analyze cache hit rates only with aligned request populations, warmup, routing, and snapshot cuts.
Do not fit capacity to reproduce a desired hit ratio.

Replace the example transfer-utilization assumption with measured transfer profiles where available.
Collect host-to-device and device-to-host service versus payload size, including simultaneous transfers, page batching, and shared host-memory/link contention.
Separate transfer service from queue wait and explicitly identify which ranks share each resource.
The existing asynchronous SimpleCPU connector and directional queues remain the implementation base.
Changing overlap or publication timing requires independent tests for source holds, destination reservations, cancellation, and cache visibility.

Required tests cover capacity identity mismatch, group geometry, exact-fit and insufficient capacity, double-deduction prevention, offload enabled/disabled, eviction under pressure, and canceled transfers.
Hardware validation must retain runtime block counts and directional bandwidth evidence alongside latency metrics.

## W8: Optimize CPU execution without changing model semantics

Profile initialization, trace parsing, graph preparation, scheduling, lookup, cache operations, replay bookkeeping, and reporting separately.
Retain CPU time, elapsed time, peak resident memory, request count, verification rounds, scheduled tokens, event count, and estimate-cache statistics.
An accuracy change can alter the amount of simulated work; report that change separately from throughput of the simulator implementation.

| Candidate | Current cost | Implementation direction | Equivalence requirement |
| --- | --- | --- | --- |
| Running queue | Sorting and list `pop(0)` each step | Stable ordered queue with efficient front/back operations | Preserve arrival ties, preemption victim choice, and loaded-request precedence |
| AgentX readiness | Full lane scans and repeated active-stream sorting | W4 dependency indexes and stable deadline/ready queues | Preserve protocol events and simultaneous-event ordering |
| Recycled trace hashes | Repeated regrouping and SHA-256 in `engine_hashes()` | Bounded cache of immutable conversions keyed by source identity and both block sizes | Preserve every hash and keep play cache salt outside shared conversions |
| Offload eviction | Scanning pinned pages from the LRU head | Maintain eviction eligibility or a resumable victim scan | Preserve recency semantics across pin/unpin and the exact victim sequence |
| Lookup preparation | Repeated shape normalization and analytical references | Reuse package indexes and immutable invocation descriptors | Preserve rounding, source flags, and returned values |
| Completed replay state | Retained Python objects during long profiles | Stream terminal evidence and release unreferenced live state | Preserve result cohorts, logs, failure diagnostics, and cache references |

The reference [queue statistics][ais-queue] and [later offload LRU][ais-lru] provide implementation ideas.
A standalone native engine rewrite is outside this plan.
Consider a compiled hotspot only after profiling shows a material remaining cost and Python-level structural changes have been measured.
Any compiled kernel must have a Python reference and deterministic differential tests.

Run five paired measurements in fresh processes pinned to the same CPU with fixed thread limits and local input files.
Alternate implementation order and retain both initialization-inclusive and replay-only timing boundaries.
Use a fixed workload count for implementation benchmarks; include a separate fixed simulated-duration case for user-visible cost.
Follow the [AISimulate performance gate][ais-perf] method while retaining TokenSim's stricter result-equivalence checks.
Flag a slowdown exceeding both 10% and 100 ms in at least four of five paired rounds for investigation.
Treat this threshold as a proposed regression policy, not a promised speedup.
Retain peak memory and duration scaling even when the wall-time gate passes.

Pure performance changes must preserve result JSON and `replay/*.jsonl` after excluding only documented host execution-time fields.
When a required metadata addition changes serialization, verify existing fields and events exactly and review new fields separately.
Do not use broad numeric tolerance or field deletion to hide altered scheduling, caching, or arithmetic order.

## W9: Validation matrix and implementation delivery

| Test family | Required cases | Main evidence |
| --- | --- | --- |
| Lookup | Multi-axis, ragged grid, empirical fallback, strict coverage | Analytical invariants, exact-value preservation, source coordinates |
| Attention | Uniform/ragged decode, cached prefill, mixed execution, MTP widths, windowed layers | FLOPs/bytes, invocation count, physical boundary evidence |
| Speculation | K=0 and K=3, deterministic extremes, stochastic acceptance, retries | Acceptance distributions, committed rounds, emitted-token accounting |
| Replay | Finite trees, continuous recycling, warm/cold, spawn/join overlap, cancellation | Harness-aligned event sequences and complete outcome accounting |
| Scheduling | Low/high load, memory pressure, DP lockstep, dummy ranks | Scheduled-batch snapshots and queue/preemption decisions |
| Forward model | Pure/mixed, graph/eager, interpolation boundaries, identity mismatch | Held-out per-step signed error, absolute percentage error, and coverage |
| Memory/offload | GPU-only, per-rank CPU, shared store where supported, constrained capacity | Runtime pages, hits, evictions, transfer service/wait, reference integrity |
| Deployment | Small dense reference plus B200/B300 DEP8 target; several concurrency points | TTFT/ITL/end-to-end distributions, throughput, failures, and observation duration |
| Host performance | Short fixed work and long continuous profile | Paired timing, peak memory, cache growth, and output equivalence |

Use low load, the reported target load, and a higher-load point only where the deployment can execute it.
Include held-out concurrency and context-length ranges; unsupported points must be recorded explicitly.
Do not project the full long-context AgentX dataset onto a small model that cannot admit it.
Pin backend-specific scheduling behavior before claiming SGLang equivalence; selecting SGLang operator tables alone does not replace the current vLLM-derived scheduler.

For positive observed latency `y`, report signed error `(prediction-y)/y` and absolute percentage error `abs(prediction-y)/y`.
Report per-step median and p95 absolute percentage error, signed bias, and sample count separately for each phase and regime.
For serving, report TTFT, per-request mean ITL, and end-to-end latency at p50/p90/p99 where sample size supports them.
Specify that a percentile of per-request mean ITL differs from a percentile of individual token intervals.
Retain successful, failed, canceled, and unfinished request counts and output-token totals.
Pin throughput's denominator, including treatment of grace completions, and report issue-window and observation duration separately.
Never convert unsupported, missing, or zero-completion predictions into zero error.

Provisional engineering targets for the matched B200/B300 target configurations are absolute throughput error at most 10% and TTFT/ITL p50/p90 errors at most 15%.
These are proposed acceptance targets, not measured outcomes or guarantees for other hardware.
Evaluate them on held-out runs and report repeated-run variation; retain any failed operating point in the report.
Single-step improvements without serving improvement must trigger a scheduling/cache/protocol investigation.
A change that improves throughput while worsening failures or hiding incomplete work cannot pass.

Deliver the implementation in independently reviewable changes:

| Change | Contents | Dependencies | Completion evidence |
| --- | --- | --- | --- |
| A | W0 baseline, small fixtures, and iteration evidence | None | Retained manifests; diagnostic-on/off equivalence |
| B | W1 coordinate correction and caller audit | A | CPU consistency matrix and ordinary/AF regression results |
| C | W2 ordinary ragged decode, then qualified prefill/mixed support | A; B for any table transfer | Invocation accounting and supported framework evidence |
| D | W3 acceptance models and deterministic sampling | A | Sampling/retry tests and isolated serving ablation |
| E | W4 typed dependencies, imported snapshots, phase validation | A | Pinned-harness differential fixtures and corpus audit |
| F | W6 strict coverage and scope provenance | B, C | Coverage failures and time-weighted accounting tests |
| G | W5 profile schema, collector/importer, offline evaluation, Runtime S1 | A, B, C, F | Scope-qualified observations and held-out timing report |
| H | W7 observed capacity and transfer evidence | A, E | Runtime allocation comparison and transfer invariants |
| I | W5 Runtime S2 for mixed DEP8/MTP AgentX | D, E, G, H | Full target configuration, group timing ownership, serving validation |
| J | W8 hotspot optimizations in separate patches | Stable behavior baseline for each affected component | Paired benchmarks and exact output comparisons |
| K | W9 release matrix and regression automation | B through I; J for claimed speedups | Complete accuracy, coverage, failure, and host-cost report |

Raw hardware collection can proceed while CPU correctness fixes are implemented.
Calibration fitting must use the corrected arithmetic, attention composition, and declared execution scope.
Performance-only changes must remain separate from semantic changes so their equivalence requirements can be evaluated.
Each source, test, and script file must remain within the project's 500-line limit.

Before submitting an implementation change, run the checks required by [CONTRIBUTING.md](../CONTRIBUTING.md):

```bash
python -m black TokenSim tests benchmark.py scripts examples/storage/hbf
python -m black --check TokenSim tests benchmark.py scripts examples/storage/hbf
python -m unittest discover \
  -s tests \
  -t .
git diff --check
```

Add focused tests in the existing test families where practical; create small dedicated lookup, forward-profile, and replay modules when those responsibilities have no suitable home.
Document all new CLI fields, backend modes, profile schemas, result fields, and changed metric semantics in their existing guides.
Use the baseline and ablation artifacts to describe expected and measured result changes in each pull request.

## Reproducible CPU probes and source references

An independent documentation-time probe used the bundled `a100_sxm_80g` device and `analytical_model_for(device.family)` at the reviewed TokenSim revision.
It used bfloat16 (BF16) attention and KV, 32 query heads, eight KV heads, head dimension 128, and window size zero.
Create one `generation_attention` row at batch 1/context 256 with latency equal to the analytical estimate and a synthetic grade-D source.
Query batch 16/context 1024 using the same analytical scaler.
The anchor is 3.655380088278568 us, the analytical target is 44.341677292790585 us, default lookup returns 95.44141723339801 us, and bound lookup returns 44.341677292790585 us.
This reproduces the consistency defect with fully specified inputs independently of the review's 43.67/108.23 us fixture.

For the ragged probe, use the same device and geometry with sixteen requests of context lengths `32, 64, ..., 512`.
Sum sixteen `generation_attention` analytical estimates and compare with one `ragged_generation_attention` estimate using exact integer histogram counts.
The results are 59.128906326630705 us and 14.128906326630702 us.
FLOPs and KV bytes agree; the 45 us difference equals fifteen additional launches at this device's 3 us assumption.
The review's 60 us difference corresponds to a 4 us launch assumption and remains a separate fixture.
Neither probe establishes the physical launch count of a production backend.

The MTP probe uses mean 2.49, K=3, seed 0, request IDs `0, 1, 2, 3, 17, 101`, and initial generation index 1.
Call `_decode_output_tokens()` 200 times per request with a large remaining-output budget, advancing generation index by each accepted count.
IDs 0, 1, and 17 return 3 throughout; IDs 2, 3, and 101 return 2 throughout.
Retain this as a historical diagnostic and test the corrected distribution contract separately.

The references below identify designs and validation practices.
Any future copied or adapted source requires immutable upstream attribution and applicable license notices in the implementation change.

[ais-speculative]: https://github.com/ai-dynamo/aisimulate/blob/289f8fb22599bfece5619c84753a3f8f65d33e38/crates/core/src/engine/common/speculative.rs
[ais-weka]: https://github.com/ai-dynamo/aisimulate/blob/289f8fb22599bfece5619c84753a3f8f65d33e38/crates/core/src/replay/loadgen/weka.rs
[ais-snapshot]: https://github.com/ai-dynamo/aisimulate/blob/dcadbc5087bc039bbe2672e7888440b792813720/crates/core/src/replay/loadgen/snapshot.rs
[ais-fpm]: https://github.com/ai-dynamo/aisimulate/blob/dcadbc5087bc039bbe2672e7888440b792813720/docs/perf-model/methods/whole-forward.md
[ais-fpm-op]: https://github.com/ai-dynamo/aisimulate/blob/289f8fb22599bfece5619c84753a3f8f65d33e38/crates/core/src/perfmodel/operators/fpm_forward.rs
[ais-resources]: https://github.com/ai-dynamo/aisimulate/blob/289f8fb22599bfece5619c84753a3f8f65d33e38/crates/core/src/perfmodel/fpm/resources.rs
[ais-queue]: https://github.com/ai-dynamo/aisimulate/blob/289f8fb22599bfece5619c84753a3f8f65d33e38/crates/core/src/engine/scheduler/queue_metrics.rs
[ais-lru]: https://github.com/ai-dynamo/aisimulate/blob/dcadbc5087bc039bbe2672e7888440b792813720/crates/core/src/engine/offload_lru.rs
[ais-perf]: https://github.com/ai-dynamo/aisimulate/blob/dcadbc5087bc039bbe2672e7888440b792813720/python/aisimulate/tools/simulation_perf_gate/README.md
