# Attention–FFN separation

Attention–feed-forward network (AF) separation places attention and FFN work on independent execution domains.
Attention workers retain request scheduling and key/value (KV) cache ownership.
FFN workers execute dense, routed and shared experts on their assigned physical ranks.
The engine compiles a model-specific runtime plan before starting workers.

## Run a simulation

The examples under `data/clusters/af` cover both phases, decode-only execution, full/half duplex links and independent A/F parallelism.
Use the standard CLI with a catalog model:

```bash
python benchmark.py \
  --cluster data/clusters/af/prefill_decode_full_duplex.json \
  --model llama-3-8b \
  --latency_backend operator_table \
  --latency_fallback analytical_only \
  --batching paged-attn \
  --block_size 16 \
  --distribution burst \
  --request_count 2 \
  --prefill_mean_len 16 \
  --prefill_range_len 0 \
  --decode_mean_len 4 \
  --decode_range_len 0 \
  --verbose none \
  --results_path outputs/af-example
```

Analytical estimates use the device catalog's compute and memory parameters.
`table_first` prefers compatible measurements; `table_only` fails when a required operator shape is missing.
Use [DeepSeek V3 MLA](deepseek-v3-mla.md) for the full-model MLA example.

## Configure domains and communication

| Field | Meaning |
| --- | --- |
| `af.enabled` | Enable AF execution |
| `af.phases` | Select `prefill`, `decode`, or both |
| `af.af_block_size` | Maximum requests per AF micro-batch |
| `af.execution_domains` | Domain capabilities, parallel sizes and physical rank mappings |
| `af.links` | Phase-specific A/F endpoints and communication resources |
| `af.router_location` | Execute routing on the A or F leader |
| `af.execution_protocol` | `partial_return` or restricted native `vllm_afd` execution |
| `af.trace_enabled` | Collect detailed simulated events |

`--block_size` specifies tokens per KV block independently of `af_block_size`.
An unselected phase requires a worker with `prefill` or `decode_full` capability.
Omitted execution domains are generated from the endpoint groups, with one single-rank replica per worker.
Explicit domains support independent tensor parallelism (TP) and data parallelism (DP).
Pipeline parallelism must be one, and generic `dp_lockstep` cannot be combined with AF.

Expert parallelism (EP) reuses the F domain's `TP*DP` physical ranks.
`EP=1` gives each replica a TP-sharded expert set.
`EP=TP*DP` distributes experts across that world with one rank per expert shard.
Intermediate EP sizes fail validation.
The configurations `decode_tp2_dp2_ep1.json` and `decode_tp2_dp2_ep4_rank_reuse.json` demonstrate these layouts.

Physical communication uses explicit eight-slot nodes, placement and node bandwidth budgets.
Declare `af.communication_domains.node_count` and choose `manual`, `mixed` or `dedicated` placement.
See [AF communication](af-communication.md) for contention, collective volumes and measurement scopes.
`flat_link` supports explicitly configured communication fabrics and lanes.
Full duplex provides separate directional resources; half duplex shares one resource.
Collectives share a fabric lane only when their configuration names that lane.

## Execution and memory ownership

Every compiled layer contains attention work followed by the model's dense or MoE FFN graph.
The graph includes required TP collectives, normalization, routing/top-k, precision conversions, weighted combination and residual addition.
Cross-domain activation and result legs carry the declared tensor ownership and routing assignments.
Routed dispatch/return traffic accounts for EP movement without duplicate internal dispatch charges.

With `partial_return`, F ranks return expert outputs and A ranks perform weighted combination and residual addition.
An F-side router sends selected expert IDs and weights to all consumers that need them.
Native `vllm_afd` combines expert results on F and returns one hidden activation per original token.
It supports all-MoE Qwen3 models without shared experts, BF16 precision and one A/F rank.
Its eager control protocol permits one in-flight transaction.

The A scheduler retains an admitted request's owner until an explicit KV transfer occurs.
Each logical KV block maps to local shards on all required A TP ranks.
Allocation, swapping, forking and release update the shard set atomically.
The replica's usable capacity is the minimum capacity across those ranks.
Equivalent prefill/decode domains sharing physical ranks reuse one scheduler and cache.

Memory accounting includes model components, declared protocol buffers and KV blocks.
`gpu_memory_utilization` limits each physical worker's budget.
An explicit KV limit caps the available memory calculation.
A total KV limit is divided by A-domain DP replicas.
Weight, activation, KV-cache and directional wire dtypes have independent roles.
See [AF execution contracts](af-correctness.md) for precision, prefill, mixed batches and recomputation details.

## Results and calibration

The engine includes embedding lookup, final normalization, language-model head and token selection in whole-model latency.
Incomplete prefill chunks skip logits and sampling work.
`af_runtime_state` reports topology, component bytes, KV partitions and capacities.
`af_trace` reports dependency readiness, resource queueing, service time and provenance.
`af_phase_fidelity` identifies AF and aggregate phases.

`communication_timing` separates host submission, transport service and completion work.
Optional host, operation and transport profiles validate their timing scope, runtime identity, environment and workload range before use.
Synthetic test profiles provide regression fixtures only.
Changing runtime configuration requires a new provider/runtime; service and DAG caches retain static estimates without retaining resource availability or completion times.

## Code layout

`TokenSim/af` groups planning, execution, runtime state, latency providers, communication resources, calibration, caching and traces into separate packages.
AF configuration and topology validation live in `TokenSim/config/af`.
Worker groups and shared cluster construction extend the existing `TokenSim/config/cluster_config.py` entrypoint.
`tests/af` mirrors these responsibilities and keeps shared fixtures under `tests/af/support`.
MLA operator and model tests live in `tests/mla`.
Internal imports refer to the defining modules; the old flat AF forwarding modules have been removed.
