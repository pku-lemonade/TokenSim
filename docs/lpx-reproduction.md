# Groq 3 LPX figure reproduction

The target is Figure 11 of [NVIDIA's long-context LPX blog](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/).
The workload has 400,000 cached input tokens, 4,000 new input tokens and 400 output tokens.
The four deployments are Rubin, external drafting, attention–feed-forward network (AF) separation and prefill–decode (PD) separation.
NVIDIA's scaled model, placement, power budget, acceptance settings and curve samples are unpublished.
The simulator exports independent architecture and execution assumptions; matching NVIDIA's numerical gains requires those inputs and measured runtime profiles.

## Native execution

`scripts/reproduce_lpx.py` version 2 executes decode through production `ModelAFExecutor` DAGs in `AFService`.
Independent request cohorts share physical compute resources, pipeline stages and transmit/receive endpoint budgets.
Pipeline parallelism (PP) owns its stage's weights and KV layers and charges activation handoffs.
Expert parallelism (EP) supports tensor-sharded experts with resolved physical collective groups.
Colocated A/F domains share one physical compute and memory inventory.
The native factory defaults to the reference planner because independently sampled routing rarely reuses an exact structural template.
Factory callers can enable structural caching for repeated routes, with eligibility limited to at most 32 query tokens.
Differential coverage compares complete native results and traces between reference and cached planning.
All LPX deployments use complete-model prefill on colocated ranks through the ordinary operator-table latency backend.
AF separation applies to decode.
Prefill never submits an AF iteration or sends A/F activation transfers.
The complete-model prefill path reserves each participating pipeline stage and its expert ranks, and charges ordinary TP/EP collectives and PP handoffs against shared endpoint resources.
Its trace identifies aggregate prefill compute as `complete_model_prefill`.
Stage compute releases its GPU reservations before the stage's ordinary model collectives acquire endpoint budgets.
`services[*].prefill_operator_stats` reports the ordinary prefill backend separately from decode operator statistics.
Split A/F pools reject prefill; AF and PD scenarios use their separate complete-model prefill pool.
Reported prefill, TTFT and factory-throughput estimates change with this execution model, so archived results require a new run for comparison.
Intermediate prefill chunks extend KV without final normalization or unused vocabulary logits.
The final suffix chunk normalizes and projects the output row for each request.
Decode traces undergo AF invariant validation before optional retention as compact timing records.
The compact history produces the same serving metrics as the full history.
Worker and host resource names include their service namespace when metrics are aggregated.
KV handoff and draft token-control transfers participate in endpoint utilization and queue accounting.

Each cohort warms the decode path for `--warmup-steps` passes, waits for the other cohorts, restores its declared 400K prefix and starts a fresh measured turn.
The default `--warmup-mode resident_decode` supplies a cached-plus-new-context snapshot only for unmeasured warm-up.
The measured turn computes every new suffix token and performs the configured handoff.
`--warmup-mode prefill_and_decode` also executes suffix prefill and handoff during warm-up.
The analytical service has no modeled cold-compilation penalty; equivalence coverage confirms the warm-up snapshot preserves measured throughput.
The measurement lasts at least `--measurement-seconds` and includes at least `--measurement-turns` complete turns per cohort.
The default requires one complete turn so a short decode-only interval cannot omit suffix prefill and handoff from factory throughput.
Increasing the number of measured turns reduces finite-window boundary effects.
Setting the turn minimum to zero is available for fixed-window diagnostics and can produce a decode-dominated sample.
In-flight work drains after measurement ends.
Admission closes at the final required turn or timeout before another turn is submitted.
In-flight prefill chunks drain, and subsequent chunks are not submitted after closure.
Warm-up and drain tokens are excluded from steady-state output throughput.
The closing timestamp is immutable after admission stops, including when an in-flight turn finishes during drain.
Resident warm-up restores cached-plus-new context between short warm-up turns.
Native measurements explicitly tag warm-up tokens and include output completions at the closing timestamp.
This boundary convention retains the final speculative token group without admitting the warm-up group that triggered measurement.

## Hardware and precision

`groq_lp30` is independent of the first-generation `groqchip_v1` model.
The [LPX architecture article](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) supplies 500 MB decimal SRAM and 150 TB/s SRAM bandwidth per chip, 256 chips per rack and 315 PFLOPS FP8 per rack.
The catalog derives 1.23046875 PFLOPS FP8 per chip and 1.25 TB/s injection bandwidth per direction from the bidirectional 2.5 TB/s total.
The separate 112 Gb/s physical-port entry has 14 GB/s bandwidth.
Aggregate injection is counted once per endpoint and aliases share the same budget.
The native experiment reserves endpoint bandwidth for point-to-point serialization and permits hop latency to overlap independent messages.
Consumers wait for complete message arrival; hop latency is retained.
`--endpoint-transfer-occupancy service` provides the conservative complete-service reservation comparison.
Ring collective rounds retain their complete service reservations.
This aggregate model books transmit and receive injection together and has no packet-level switch simulation.
LPX external DRAM is excluded from resident experiments.

The `rubin` catalog follows the [product specifications](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) retrieved on 2026-10-14.
The catalog uses 288 GB HBM4, 19.2 TB/s memory bandwidth, 1.5 TB/s NVLink per direction and dense compute specifications.
Compute efficiency, launch cost, transport startup and processor power remain evidence-grade-D assumptions.
Processor defaults are 1,800 W for Rubin and 450 W for LP30.
`--facility-watts` adds a declared host, network and cooling allowance.
These defaults do not establish measured rack efficiency.

Expert storage uses MXFP4 payload with one scale byte per 32 weights and padded rows.
Other weights, activations and KV use BF16 in the corrected candidates.
LP30 executes packed weights using its FP8 compute capability with an explicit analytical unpack cost.
Four-bit storage does not double its FP8 execution rate.
GPU graph mode charges a host submission per cohort pass and an explicit device-kernel overhead.
The defaults are 4 microseconds for submission and 0.2 microseconds per device kernel.
LP30 timing is unaffected by the GPU graph switch.
`--execution-mode eager` and `--graph-kernel-us` support sensitivity checks without changing hardware specifications.
Measured GPU graph or LP30 compiler profiles are not supplied.
Ready endpoint messages are batched with byte conservation and explicit producer dependencies.
The simulator does not claim NVIDIA's 320-byte compiler overlap or a calibrated fused implementation.

## Architecture candidates

The [public GPT-OSS-120B config](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json) alternates 128-token local attention and full attention and uses top-4 routed experts.
The default candidate retains that pattern and mixed precision while assuming a larger width and layer count.
Each exported summary separates total parameters, activated routed parameters and linear FLOPs per token.
Attention work dependent on KV length is priced separately by the native provider.

| Candidate | Layers / width / experts | Total parameters | Activated routed parameters per token | Expert precision |
| --- | --- | ---: | ---: | --- |
| `gpt-oss-2t-wide-proxy` | 80 / 8192 / 128 | 2.077T | 64.42B | MXFP4 |
| `gpt-oss-2t-expert-proxy` | 36 / 2880 / 2240 | 2.009T | 3.58B | MXFP4 |
| Historical `gpt-oss-2t-proxy` | 36 / 2880 / 2240 | 2.009T | 3.58B | FP8 |

The historical candidate retains uniform full attention for existing result reproducibility.
Both corrected candidates have an assumed 1M context limit.
Neither corrected architecture is identified as NVIDIA's scaled GPT-OSS deployment.
The independent four-layer `gpt-oss-draft-proxy` retains public GPT-OSS width and precision instead of inheriting the target's widened geometry.
Draft quality and acceptance require external measurements.
Corrected candidates use seeded, stateless uniform top-k sampling without replacement.
The generic routing default remains cyclic for historical reproducibility.
Consecutive cyclic experts can share a contiguous EP owner and impose correlated single-token load.
`--routing-selection cyclic` provides that placement-sensitive comparison.
Real router correlations and expert hotness are unmeasured.

## Run and inspect

The default sweep declares eight topologies and three client counts in `experiments/lpx/sweep.json`.
`--batches` means concurrent clients; `--micro-batch` controls requests in each independently progressing cohort.
`--a-replicas` adds attention serving replicas and charges their physical inventory.
FFN stages remain shared across those request streams.
Capacity failures are exported and excluded from Pareto frontiers.
`experiments/lpx/pipeline-sweep.json` also sweeps attention PP8, a matching PP8 prefill pool and enough cohorts to populate those stages.
This separates pipeline throughput from the latency of one request on an idle deployment.
LP30-only decode can reject the larger cohorts because full-attention KV and the output-head shard share SRAM.
The whole-model CPU simulation can take substantial time because routing, physical dependencies and trace invariants are evaluated explicitly.
`--jobs` runs independent points in isolated processes and does not change their simulated resource sharing or power.
Sequential and parallel point results have deterministic-equivalence coverage.

```bash
python scripts/reproduce_lpx.py \
  --skip-audit \
  --cases rubin-tp8 af-tp8 pd-tp16 draft-tp8 \
  --batches 1 2 \
  --measurement-turns 1 \
  --jobs 4 \
  --plot \
  --output outputs/lpx-native-reproduction
```

Use the matching topology file for expert-count scaling:

```bash
python scripts/reproduce_lpx.py \
  --model gpt-oss-2t-expert-proxy \
  --config experiments/lpx/expert-sweep.json \
  --skip-audit \
  --batches 1 \
  --plot \
  --output outputs/lpx-native-expert
```

A short script smoke test preserves long-context capacity checks while reducing timed work:

```bash
python scripts/reproduce_lpx.py \
  --skip-audit \
  --cases rubin-tp8 af-tp8 pd-tp16 draft-tp8 \
  --batches 1 \
  --new-tokens 8 \
  --output-tokens 8 \
  --measurement-seconds 0.003 \
  --plot \
  --output outputs/lpx-native-smoke
```

`--cached-tokens`, `--new-tokens`, `--output-tokens`, `--prefill-chunk` and `--acceptance` override workload inputs.
`--config`, `--model`, `--draft-model` and `--cases` select explicit configurations.
`--routing-selection` overrides the uniform selection rule on both target and drafter.
`--trace` retains complete native events for client-count-one points.
`--audit-only` checks one resident full-attention layer and a rejected whole-model placement on 8 Rubin GPUs and 256 LP30 chips.
`--rubin-watts` and `--lp30-watts` override processor power.
`--context-samples` is retained for historical configuration compatibility and has no sampling effect in the native executor.
Invalid CLI settings fail before output creation.

`summary.json` records revision and dirty-state snapshots at the start and end of evaluation, model/draft inputs, architecture arithmetic, hardware, run settings, physical inventories, capacity failures and frontiers.
New summaries record the requested client counts in `run_settings.client_counts` and reject missing or unexpected case/client combinations during export.
`partial-results.json` checkpoints finished points while a sweep is running.
`power_model` preserves effective processor watts, source identifiers and evidence grades independently of the hardware performance snapshot.
Each newly exported summary runs arithmetic, physical-inventory, complete-turn, memory-reservation, transfer-byte and Pareto-frontier validation.
`points.csv` contains interactivity, raw output throughput, power and normalized efficiency.
Successful points export bounded/resident KV bytes, service counts, complete/partial turns, TTFT, utilization, queue delays and transfer events.
New results retain token completion events and decode intervals with cohort and turn identifiers.
The independent audit reconstructs output counts, per-user elapsed time, complete-turn token counts and the earliest valid closing timestamp from that evidence.
`--original-workload` requires a positive complete-turn minimum and rejects fixed-window decode diagnostics even if their declared token lengths match the original figure.

Complete-turn CLI runs accumulate resource metrics online and release each iteration's events after native invariant validation.
Live event positions remain stable while other iterations are executing.
This bounds retained event objects by active iterations instead of total experiment length.
`--retain-timing-history` preserves completed timing records for differential diagnostics.
`--trace` retains full events for client-count-one points.
Retained histories also receive a global capacity audit across iterations, which can detect short resource overlaps even when average utilization is below one.
New exports require their token-event evidence; older summaries report `event_accounting: legacy_turn_totals` to identify the narrower audit scope.

Interactivity divides committed decode tokens by per-user decode-pass elapsed time, including queueing.
Factory throughput counts completed output tokens over the measured wall-clock interval, including suffix prefill, cache handoff and draft service.
TPS/MW divides factory throughput by all allocated processor watts plus the facility allowance.
All simulated modes share the maximum feasible Rubin TPS/MW denominator from that sweep.
Each mode's frontier contains only nondominated feasible points.

PD supports `--handoff-mode full` for a cold destination and `incremental` for a verified retained warm prefix.
Handoffs copy missing positions, retain bounded layer tails and replicate shared KV heads through destination-local transfers.
Every speculative cycle executes all `gamma + 1` verifier positions, even on complete rejection.
Complete acceptance charges an additional draft forward when its final accepted token still lacks draft KV.
Seeded accepted prefixes commit a correction/bonus token and roll back uncommitted KV on both models.
The default micro-batch of one gives independently sampled request acceptance.
Target and independent drafter suffix prefill execute concurrently on their own physical pools.
Larger fixed cohorts share one accepted-prefix sample to retain synchronized request lengths.

## Concurrency coverage

One or two resident clients provide low-load latency checks and cannot establish a multi-stage deployment's throughput frontier.
The retained eight-point run used an explicit `--batches 1 2` override to limit host simulation time.
That choice omitted the larger client counts already declared by the default and pipeline sweeps.
Its efficiency ratios describe only the measured operating points.

Keep physical inventory fixed within each load sweep.
Vary client counts and micro-batch sizes independently, with enough progressing cohorts to exercise all pipeline stages.
Continue until throughput stops increasing across successive load increments or resident memory rejects another point.
Increase the measurement length and check stability at candidate peaks.
Compare feasible frontiers across the declared placements and use a common Rubin denominator from the expanded experiment.

`scripts/profile_lpx_concurrency.py` checks resident capacity through the production physical rank ledger.
It includes every service pool declared by each case and reserves the original workload's peak context, including speculative lookahead where applicable.
The limit accounts for weights, protocol buffers, full/windowed KV layers, pipeline placement and request distribution across attention replicas.
Partial final cohorts use their actual request counts.
Run capacity checks without executing decoder passes:

```bash
python scripts/profile_lpx_concurrency.py \
  --config experiments/lpx/pipeline-sweep.json \
  --capacity-only \
  --output outputs/lpx-pipeline-capacity
```

Rubin and AF can also run resident decode concurrency probes:

```bash
python scripts/profile_lpx_concurrency.py \
  --config experiments/lpx/pipeline-sweep.json \
  --cases rubin-tp8-pp8 af-tp8-pp8 \
  --batches 1 2 4 8 16 32 64 128 256 \
  --micro-batches 1 4 16 \
  --steps 8 \
  --warmup-steps 2 \
  --output outputs/lpx-decode-concurrency
```

`--steps` is the minimum number of measured decode passes per cohort.
Micro-batch sizes greater than a requested client count are skipped to avoid duplicate single-cohort probes.
Each cohort continues submitting work until all cohorts meet that minimum, so early finishers do not reduce the active population during measurement.
Warm-up and drain are excluded from the measured decode rates.
The result records actual committed tokens, completed passes per cohort, micro-batch size and mean/maximum compute utilization.
Mean utilization includes every allocated compute rank in the target runtime, including idle ranks.
`profile.json` checkpoints capacity and probe outcomes after each completed point.
Probe timing uses `ModelAFExecutor` and the same runtime configurations as complete-turn experiments.

These probes exclude suffix prefill, KV handoff and speculative verification.
Their `resident_decode_only` scope cannot supply complete-turn factory throughput or a normalization denominator for the reference comparison.
PD and draft cases support capacity checks and require their complete serving protocols for throughput measurements.
Repeat candidate complete turns with the original 400K/4K/400 workload using `reproduce_lpx.py` before reporting deployment efficiency.
The PP8 configuration is a distinct physical inventory and requires its own matching Rubin baseline.

## Optional sampled native-DAG execution

The [expanded validation study](lpx-full-validation.md) found numerical errors up to 45.8% in concurrent replay controls.
This backend is experimental and its high-load results are exploratory.
An accounting audit does not certify its timing predictions.

`--execution-backend sampled-replay` captures native whole-model operator directed acyclic graphs (DAGs) and schedules their dependencies and physical resource reservations with a compiled C++17 event engine.
The native backend remains the default.
The optional backend requires `g++` or `clang++`; its shared library is built locally under `.codex/lpx-replay`.
Every captured profile must reconstruct its own isolated native pass duration before it can be used.
Cached JSON profiles include source, model, effective hardware, link, runtime-version and configuration fingerprints, plus a digest of their node data.
The profile cache uses file locks and atomic replacement and does not deserialize executable Python objects.

The sampled backend still executes all requested output tokens, complete turns, suffix-prefill chunks, cache handoffs and speculative verification/rollback operations.
It replaces per-pass graph construction and Python operator scheduling with a finite corpus of native profiles.
`--replay-samples` controls decode profiles per signature and defaults to four.
`--replay-prefill-samples` defaults to one.
`--replay-cache` chooses the profile cache directory.
Sample selection uses a separate deterministic seed and does not consume speculative acceptance draws.
Request cohorts must have synchronized contexts.
For inputs below 16K tokens, profiles use the exact request context.
Long-context decode uses the midpoint of the output range; nonterminal prefill uses the suffix midpoint, and terminal prefill uses the complete input context.
Routing samples, representative contexts and cross-pass resource tie ordering introduce numerical approximation.
No fitted throughput multiplier or vendor-curve calibration is applied.

Sampled summaries use version 3, `executor: sampled_native_dag_replay` and per-point `native_executor: SampledNativeReplay`.
Their resource/token/inventory audit reports `status: accounting_passed` and `numerical_accuracy: requires_separate_native_calibration`.
Passing that audit establishes accounting consistency; native controls across the intended modes, client counts and batch sizes must establish numerical accuracy separately.
Self-reconstruction of isolated profiles alone does not validate contention under concurrent passes.
The `steady_state` result key is retained for compatibility and describes the measured finite window.
Window length, complete-turn count, micro-batch size and routing samples require separate convergence checks.
The native 64-client Rubin control changed by 70.2% when measurement increased from eight to 32 decode passes per cohort.
Warm-up exclusion and positive complete-turn counts alone do not establish saturation.
The concurrency profiler accepts the same optional backend for direct comparisons with resident decode controls.
Sampled execution does not support full native trace export.

`--lp30-reserve-bytes` and `--rubin-reserve-bytes` add an explicit per-device allowance to the catalog's existing reserved memory.
The allowance reduces usable memory before production admission and capacity checks and is saved with the effective hardware snapshot.
It is a declared assumption, not a measured workspace requirement.
Flat-link LPX configurations account for resident weights and KV but do not derive all activation, compiler or allocator workspace requirements.
Capacity without an additional allowance is therefore a resident-memory bound.

## Figure artifacts and limits

Install plotting dependencies with `python -m pip install -e '.[dev,plot]'`.
`--plot` exports `reference`, `simulation` and `comparison` in PNG and editable SVG.
`reference` uses approximate manual samples in `experiments/lpx/reference.json` and never contributes to simulation timing or calibration.
The comparison preserves independent reference and simulation normalizations and expands shared axes if a result exceeds the supplied figure's range.
Reference 3x/5x annotations are not applied to simulated points.

Native context parallelism is unimplemented and declared CP greater than one fails configuration.
Higher attention TP can replicate KV heads, so capacity and reads stop shrinking beyond the head-sharding limit.
Local attention executes on the configured attention pool with bounded KV.
A separate LP30-local-attention placement at full-attention exchange boundaries is an unvalidated deployment hypothesis and is not implemented.
These limits restrict feasible PD concurrency and high-interactivity candidates.
The defaults use flat aggregate links and do not resolve detailed rack routing.
Resident cohorts have no paged admission, preemption or cache eviction.

The original serialized sweep is preserved as `experiments/lpx/legacy-serialized-sweep.json` with historical outputs under `outputs/lpx-reproduction`.
The [diagnosis](lpx-diagnosis.md) describes that historical executor and its probes.
The native output directory is separate to preserve the original evidence.
Audit a saved native summary without rerunning its requests:

```bash
python scripts/check_lpx_results.py \
  --summary outputs/lpx-native-reproduction/summary.json \
  --config experiments/lpx/sweep.json \
  --cases rubin-tp8 af-tp8 pd-tp16 draft-tp8 \
  --batches 1 2 \
  --original-workload \
  --output outputs/lpx-native-reproduction/validation.json \
  --export outputs/lpx-native-reproduction/audited
```

`--config` compares the saved case layouts with the selected configurations and checks the complete case/client grid.
`--cases` selects names from that configuration and requires `--config`.
`--batches` overrides the configuration's client counts or supplies the expected counts for an older summary.
Without `--batches`, a supplied configuration uses its own client counts.
Without external counts, the audit uses `run_settings.client_counts` when available.
An older summary with neither source reports `grid_coverage: not_recorded`; its point arithmetic can pass without establishing experiment completeness.
Draft verification commits must equal measured decode tokens, with warm-up and drain cycles excluded and partial cohorts weighted by their request count.
`--export` writes a compact `evidence.json` and `points.csv` after successful validation.
The evidence preserves effective power assumptions, inputs, frontiers, complete turns and measured verification totals, and identifies the unmodified source summary by its SHA-256 digest.
Audit output paths must differ from the source summary and each other.

For version-2 summaries produced without a power snapshot, this command verifies the saved hardware snapshot and reconstructs processor power from the explicitly loaded local catalog.
The audit checks every saved power total against that catalog before accepting the result.
Passing software tests and repairing the known execution defects do not establish NVIDIA's projected speed or efficiency gains.
The [completed native experiment report](lpx-native-results.md) records the original-workload measurements and their remaining limits.
The [expanded validation report](lpx-full-validation.md) records the load matrix, failed numerical calibration and window, memory and architecture sensitivities.
