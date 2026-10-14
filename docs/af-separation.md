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
Pipeline parallelism partitions consecutive layers onto distinct physical ranks within each A/F domain.
The domains may have different stage counts.
Embedding and output-head storage belongs to the boundary A stages, and each A stage owns its local KV shards.
Activation handoffs between A stages use the declared A communication fabric.
Independent micro-batches can occupy different stages concurrently.
Generic `dp_lockstep` cannot be combined with AF.

Expert parallelism (EP) reuses the F domain's `TP*DP` physical ranks.
`EP=1` gives each replica a TP-sharded expert set.
`EP=TP*DP` distributes experts across that world with one rank per expert shard.
Intermediate EP sizes divide the `TP*DP` stage world and retain `Expert TP=(TP*DP)/EP` contiguous tensor shards per expert.
Expert TP must divide base TP so one expert's shards fit within a base replica.
The runtime resolves exact expert owners and scopes routed-expert collectives to those shards.
The configurations `decode_tp2_dp2_ep1.json` and `decode_tp2_dp2_ep4_rank_reuse.json` demonstrate these layouts.

Physical communication uses explicit eight-slot nodes, placement and node bandwidth budgets.
Declare `af.communication_domains.node_count` and choose `manual`, `mixed` or `dedicated` placement.
The communication model below describes contention, collective volumes and measurement scopes.
`flat_link` supports explicitly configured communication fabrics and lanes.
Full duplex provides separate directional resources; half duplex shares one resource.
Collectives share a fabric lane only when their configuration names that lane.

## Execution and memory ownership

Every compiled layer contains attention work followed by the model's dense or MoE FFN graph.
The graph includes required TP collectives, normalization, routing/top-k, precision conversions, weighted combination and residual addition.
Cross-domain activation and result legs carry the declared tensor ownership and routing assignments.
Routed dispatch/return traffic accounts for EP movement without duplicate internal dispatch charges.
The graph owns post-attention normalization and the A-side residual addition.
Router service includes projection and top-k selection; routed expert service excludes normalization, combination and residual work.
Analytical routed-expert service uses the exact local assignment histogram and active-expert count.
Measured aggregate MoE rows retain their balanced-equivalent batch interpretation.
FFN weight accounting uses the model's projection count for gated and ungated activations.
Combine and residual queries use the A rank's owned hidden interval.

With `partial_return`, F ranks return expert outputs and A ranks perform weighted combination and residual addition.
An F-side router sends selected expert IDs and weights to all consumers that need them, including one-A/one-F deployments.
Each token transmits one 32-bit expert ID and one 32-bit weight per selected expert.
Native `vllm_afd` combines expert results on F and returns one hidden activation per original token.
It supports all-MoE Qwen3 models without shared experts, BF16 precision and one A/F rank.
Its eager control protocol permits one in-flight transaction.

The A scheduler retains an admitted request's owner until an explicit KV transfer occurs.
Each logical KV block maps to local shards on all required A TP ranks.
Allocation, swapping, forking and release update the shard set atomically.
The replica's usable capacity is the minimum capacity across those ranks.
Equivalent prefill/decode domains sharing physical ranks reuse one scheduler and cache.
Prefix lookup occurs once on the A scheduler leader, and every logical hit or miss maps to local shards on each A TP rank.
Reference counts, eviction, free, preemption and swap update the complete shard set.
F ranks own no prefix-cache blocks.

Memory accounting includes model components, declared protocol buffers and KV blocks.
`gpu_memory_utilization` limits each physical worker's budget.
An explicit KV limit caps the available memory calculation.
A total KV limit is divided by A-domain DP replicas.
Weight, activation, KV-cache and directional wire dtypes have independent roles.
GEMM/expert queries use weight precision; normalization, rotary embeddings and residuals use activation precision; attention uses KV-cache precision.
`ModelSpec.attention_windows` can declare a different key window for each model layer.
Attention service-cache keys include that geometry so a local-attention estimate cannot satisfy a full-attention query.
`operator_dtypes` selects component weight storage independently of activation and KV formats.
The `routed_experts` entry supports weight-only MXFP4 storage with row padding and one scale byte per 32 weights.
Packed formats cannot be selected for activations, KV blocks or communication tensors.
LP30 packed-weight estimates use its supported compute-pipe fallback and charge an explicit analytical vector-dequantization term.
These vector efficiency and instruction-count inputs remain unmeasured assumptions.

## Request geometry and execution contracts

Routing snapshots retain each request's absolute token identity and decode step.
External assignments are validated against that request's position, including requests at different decode positions in one block.
Prefill records query lengths and available KV contexts separately for every request.
Cached keys contribute attention FLOPs and KV reads when only a suffix is computed.
Whole-prompt measurements used for cached prefill are scaled by an analytical growth ratio and labeled extrapolated.
Uneven requests retain separate query shapes.

Wire precision changes create encode/decode nodes on the producer and consumer compute resources.
Conversion names record both formats, such as `cast_bf16_to_fp8`, and the table dtype identifies the output format.
Analytical conversion estimates include source reads, target writes, conversion work and kernel overhead.
Strict lookup requires a matching measurement.

Mixed serving batches are partitioned by logical phase and whether they build context.
AF groups run concurrently on shared physical resources and join at one serving-step completion barrier.
Recomputation retains decode ownership while rebuilding context with prefill attention kernels.
MLA prefill/recomputation require equal-length, whole contexts without cached prefixes or partial chunks.
Unsupported MLA passes fail before ownership or iteration state changes.
Failure or cancellation interrupts sibling groups and releases their resources.

## Results and calibration

The engine includes embedding lookup, final normalization, language-model head and token selection in whole-model latency.
Incomplete prefill chunks skip logits, logits collectives and sampling work.
The chunk completing a prompt produces one logits row per completed request.
`af_runtime_state` reports topology, component bytes, KV partitions and capacities.
`af_trace` reports dependency readiness, resource queueing, service time and provenance.
`af_phase_fidelity` identifies AF and aggregate phases.
The fixed-batch `simulate_decode` helper checks the selected A replica's resident KV capacity and model context limit before submitting work.

`communication_timing` separates host submission, transport service and completion work.
Optional host, operation and transport profiles validate their timing scope, runtime identity, environment and workload range before use.
Synthetic test profiles provide regression fixtures only.
Changing runtime configuration requires a new provider/runtime; service and DAG caches retain static estimates without retaining resource availability or completion times.

## Physical communication

Physical AF communication uses the `node-domain-fifo-v1` model.
Each node has eight GPU slots, one shared local communication resource, and independent inter-node transmit and receive resources.
Empty slots remain part of the declared inventory.

```json
{
  "transfer_topology_mode": "physical",
  "communication_domains": {
    "node_size": 8,
    "node_count": 2,
    "placement_strategy": "manual",
    "device_indices": {
      "a-pool:0": 0,
      "f-pool:0": 8
    },
    "local_bandwidth_bytes_per_s": 300000000000,
    "inter_tx_bandwidth_bytes_per_s": 50000000000,
    "inter_rx_bandwidth_bytes_per_s": 50000000000,
    "local_latency_us": 1,
    "inter_latency_us": 5
  }
}
```

Place these fields inside the cluster's `af` object.
Bandwidth is unidirectional bytes/s, with decimal GB units.
The defaults above are simulator assumptions: 300 GB/s is a shared node budget.
The physical mode reads these parameters independently of the catalog link name.
`manual` supports explicit device indices; `dedicated` separates A/F nodes; `mixed` reserves the configured A slots within each node.
Placement rejects duplicate slots, insufficient inventory and conflicting overrides before execution.

A local transfer holds its node's local resource.
A remote transfer atomically acquires the source transmit and target receive resources and uses their lower bandwidth.
Requests sharing either resource queue in FIFO order; disjoint resources can run concurrently.
Waiting transfers do not hold a subset of their resources.
Cancellation releases reservations and queued requests.

### Collective volumes

Collective input `M` is the logical input bytes per rank, and `N` is the actual participant count.

| Operation | Logical output per rank | Ring bytes sent per rank |
| --- | --- | --- |
| AllGather | `N*M` | `(N-1)*M` |
| ReduceScatter | `M/N` | `(N-1)*M/N` |
| AllReduce | `M` | `2*(N-1)*M/N` |

Reduction shards must contain whole dtype elements.
Hierarchical collectives require equally sized participating subgroups on each node.
AllReduce executes local ReduceScatter, inter-node shard AllReduce and local AllGather.
AllGather and ReduceScatter add local layout conversion when their input/output ordering requires it.
These conversions occupy compute resources and send no network bytes.
Each subgroup waits for its own producers; consumers wait for all required collective phases.

All-to-All may use declared locality fractions and load imbalance in physical mode.
Expected output bytes retain fractional statistical traffic, and total traffic is conserved.
Flat All-to-All supports the uniform distribution with default imbalance and no destination redistribution.
Flat-link experiments can select `communication_resource_scope: endpoint` to share a transmit and receive injection budget per physical worker and link class.
Alias network names referencing the same link class share those budgets.
Disjoint endpoint sets can transfer concurrently.
Atomic admission prevents a waiting collective from holding a subset of its endpoints.
Unsupported statistical settings fail before no-op handling or cache lookup.

### Service measurements and traces

Analytical phase service is fixed latency plus wire bytes divided by effective bandwidth.
Measured phase records require `timing_scope=transport_service` and matching model version, algorithm, operation, dtype, participants, node layout, wire schedule and domain parameters.
Hardware/backend applicability and transport backend also participate in matching.
A whole-collective measurement cannot be split into invented phase measurements.
`table_only` rejects missing phase measurements; policies allowing fallback label analytical service explicitly.

Caches retain static traffic plans and service estimates.
Every invocation creates its own readiness events, resource reservations and queueing delays.
Trace records keep dependency waiting, resource queueing and service duration separate.
The node FIFO and coarse phase resource reservations are modeling approximations.
The cohort-serving API under `TokenSim.af.serving` executes independent streams through `ModelAFExecutor` and exports steady-state token counts, queue delays and resource utilization.
Its resident KV leases account for each stage's bounded or full-context layer windows and reserve the declared peak context before execution.
This fixed-cohort mode does not page or preempt requests.
KV handoffs verify model/cache identity and source coverage before scheduling copies.
Incremental mode sends only positions absent from the declared warm destination.
Local layers send their retained window tail, and replicated KV heads receive a cross-pool canonical copy followed by destination-local replication.
Cancellation releases every endpoint reservation and leaves destination validity unchanged.

## Code layout

`TokenSim/af` groups planning, execution, runtime state, latency providers, communication resources, calibration, caching and traces into separate packages.
AF configuration and topology validation live in `TokenSim/config/af`.
Worker groups and shared cluster construction extend the existing `TokenSim/config/cluster_config.py` entrypoint.
`tests/af` mirrors these responsibilities and keeps shared fixtures under `tests/af/support`.
MLA operator and model tests live in `tests/mla`.
Internal imports refer to the defining modules; the old flat AF forwarding modules have been removed.
