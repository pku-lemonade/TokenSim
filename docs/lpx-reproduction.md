# Groq 3 LPX figure reproduction

The target is Figure 11 of [NVIDIA's long-context LPX blog](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/).
The workload has 400,000 cached input tokens, 4,000 new input tokens and 400 output tokens.
The four deployments are Rubin, external drafting, attention–feed-forward network (AF) separation and prefill–decode (PD) separation.
NVIDIA's scaled model, placement, power budget, acceptance settings and curve samples are unpublished.
The simulator exports independent architecture and execution assumptions; matching NVIDIA's numerical gains requires those inputs and measured runtime profiles.

## Native execution

`scripts/reproduce_lpx.py` version 2 executes whole-model production `ModelAFExecutor` DAGs through `AFService`.
Independent request cohorts share physical compute resources, pipeline stages and transmit/receive endpoint budgets.
Pipeline parallelism (PP) owns its stage's weights and KV layers and charges activation handoffs.
Expert parallelism (EP) supports tensor-sharded experts with resolved physical collective groups.
Colocated A/F domains share one physical compute and memory inventory.
Large prefill graphs bypass the structural template cache above 32 query tokens to bound compilation cost and retained routing records.
Intermediate prefill chunks extend KV without producing unused vocabulary logits; the final suffix chunk computes the prefill output token.
Full traces undergo invariant validation before optional retention as compact timing records.
The compact history produces the same serving metrics as the full history.
Worker and host resource names include their service namespace when metrics are aggregated.
KV handoff and draft token-control transfers participate in endpoint utilization and queue accounting.

Each cohort warms the prefill/decode path for `--warmup-steps` decode passes, waits for the other cohorts, restores its declared warm prefix and starts a fresh measured turn.
The measurement lasts at least `--measurement-seconds` and includes at least `--measurement-turns` complete turns per cohort.
The default requires one complete turn so a short decode-only interval cannot omit suffix prefill and handoff from factory throughput.
Increasing the number of measured turns reduces finite-window boundary effects.
Setting the turn minimum to zero is available for fixed-window diagnostics and can produce a decode-dominated sample.
In-flight work drains after measurement ends.
Warm-up and drain tokens are excluded from steady-state output throughput.
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
  --batches 1 \
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

`summary.json` records revision, dirty state, model/draft inputs, architecture arithmetic, hardware, run settings, physical inventories, capacity failures and frontiers.
`partial-results.json` checkpoints finished points while a sweep is running.
`points.csv` contains interactivity, power and normalized efficiency.
Successful points export bounded/resident KV bytes, service counts, complete/partial turns, TTFT, utilization, queue delays and transfer events.
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
Larger fixed cohorts share one accepted-prefix sample to retain synchronized request lengths.

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
Passing software tests and repairing the known execution defects do not establish NVIDIA's projected speed or efficiency gains.
