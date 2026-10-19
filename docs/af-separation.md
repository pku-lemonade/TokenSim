# Attention–FFN separation

Attention–feed-forward network (AF) separation places decode attention and FFN work on independent execution domains.
Prefill executes the complete model on ordinary workers.
Attention workers retain request scheduling and key/value (KV) cache ownership.
FFN workers execute dense, routed and shared experts on their assigned physical ranks.
The engine compiles a model-specific runtime plan before starting workers.

## Run a simulation

The examples under `data/clusters/af` cover decode AF execution, full/half duplex links and independent A/F parallelism.
Use the standard CLI with a catalog model:

```bash
python benchmark.py \
  --cluster data/clusters/af/decode_half_duplex.json \
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
| `af.enabled` | Enable decode AF execution |
| `af.af_block_size` | Maximum query tokens per AF micro-batch |
| `af.ffn_batching` | Combine ready F work across A owners under `partial_return`; defaults to `true` |
| `af.ffn_batch_wait_us` | `partial_return` F collection window in microseconds; defaults to the uncalibrated assumption `5.0`, with no window for one A replica |
| `af.ffn_batch_max_tokens` | Maximum original tokens combined on one F rank; `0` uses `af_block_size * A_DP` |
| `af.execution_domains` | Domain capabilities, parallel sizes and physical rank mappings |
| `af.links` | Decode A/F endpoints and communication resources |
| `af.router_location` | Execute routing on the A or F leader |
| `af.execution_protocol` | `partial_return` or restricted native `vllm_afd` execution |
| `af.trace_enabled` | Collect detailed simulated events |

`--block_size` specifies tokens per KV block independently of `af_block_size`.
Recomputation and multi-token verification count every scheduled query token toward the AF micro-batch budget.
Requests are packed into blocks by their scheduled query-token counts.
A request exceeding `af_block_size` is rejected before AF ownership assignment or execution.
Increase `af_block_size` to reserve sufficient workspace, or schedule a smaller query chunk.
Serving requires a complete-model worker with `prefill` capability.
AF configuration, links and execution domains have no `phases` field.
Each enabled runtime has one decode link and its two A/F execution domains.
Configurations containing the removed field are rejected.
The executor accepts decode requests and rejects prefill requests before changing ownership or creating an iteration.
Decode-only runtime helpers can use a cluster without a prefill worker.
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
The configuration `decode_tp2_dp2_ep4_rank_reuse.json` uses `TP=2`, `DP=2` and `EP=4`.
An `EP=1` variant uses the same inventory with F `expert_parallel_size` set to one and every F `ep_rank` set to zero.

Physical communication uses explicit node slots, placement and node bandwidth budgets.
`communication_domains.node_size` defaults to eight and accepts any positive integer, including four for GB200 compute nodes.
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
F batching combines compatible dense, shared-expert or grouped routed-expert queries across A owners under `partial_return`.
The batch key retains the layer, physical F rank, TP/EP groups, precision and operator geometry.
Each batch contains at most one block from each A owner.
Collection starts when a compatible input becomes ready, waits for `ffn_batch_wait_us`, and freezes the available members before joining the device queue.
A zero window collects inputs ready at the same simulated timestamp.
Late inputs form another batch, and an atomic block exceeding the configured combination limit executes alone.
Batch waiting and device queueing remain visible, and outputs retain their original request, A owner and layer dependencies.
Setting `ffn_batching: false` retains independent F invocations under `partial_return` for sensitivity comparisons.
The physical F workspace reservation covers the inferred or configured aggregate token bound.
This reservation covers one active F batch and does not model every queued input buffer or persistent-kernel lane.
FFN weight accounting uses the model's projection count for gated and ungated activations.
Combine and residual queries use the A rank's owned hidden interval.

With `partial_return`, F ranks return expert outputs and A ranks perform weighted combination and residual addition.
An F-side router sends selected expert IDs and weights to all consumers that need them, including one-A/one-F deployments.
Each token transmits one 32-bit expert ID and one 32-bit weight per selected expert.
Native `vllm_afd` combines expert results on F and returns one hidden activation per original token.
It supports all-MoE Qwen3 models without shared experts, BF16 precision, single-rank A replicas and one F rank.
Multiple A replicas can share that F service when native control is disabled.
Each A owner's submission sequence identifies a shared transaction round, and block indices identify matching microbatches within that round.
Peers must submit matching microbatch counts and forward phases, with potentially different token counts.
F waits for every mapped A peer before routing, then combines their expert and weighted-combine queries.
Rounds and F layer/microbatch invocations retain transaction order.
Native peer synchronization is mandatory even when `ffn_batching` is disabled; `ffn_batch_wait_us` does not release a partial native transaction.
A native peer batch exceeding `ffn_batch_max_tokens` fails without splitting its transaction.
Missing peers leave the transaction pending; idle-peer placeholders are not synthesized.
Peer cancellation or failure aborts in-flight native transactions and requires a new executor for retry.
This path models peer synchronization without claiming calibrated multi-A connector timing.
Its eager control protocol permits one in-flight transaction.
Native control, call-order replay and native calibration profiles require 1A1F.

The A scheduler retains an admitted request's owner until an explicit KV transfer occurs.
Each logical KV block maps to local shards on all required A TP ranks.
Allocation, swapping, forking and release update the shard set atomically.
The replica's usable capacity is the minimum capacity across those ranks.
Each decode A replica owns one scheduler and cache.
Prefix lookup occurs once on the A scheduler leader, and every logical hit or miss maps to local shards on each A TP rank.
Reference counts, eviction, free, preemption and swap update the complete shard set.
F ranks own no prefix-cache blocks.

Memory accounting includes model components, declared protocol buffers and KV blocks.
`gpu_memory_utilization` limits each physical worker's budget.
An explicit KV limit caps the available memory calculation.
`AFService` admission enforces the compiled per-rank KV budgets and shared physical capacity before adding a lease.
Sliding-window cohorts charge their bounded resident KV bytes against those budgets.
Failed admission leaves existing leases unchanged, and release returns the reserved budget.
A total KV limit is divided by A-domain DP replicas.
Weight, activation, KV-cache and directional wire dtypes have independent roles.
GEMM/expert queries use weight precision; normalization, rotary embeddings and residuals use activation precision; attention uses KV-cache precision.
`ModelSpec.attention_windows` can declare a different key window for each model layer.
Attention service-cache keys include that geometry so a local-attention estimate cannot satisfy a full-attention query.
`operator_dtypes` selects component weight storage independently of activation and KV formats.
The `routed_experts` entry supports weight-only MXFP4 storage with row padding and one scale byte per 32 weights.
Packed formats cannot be selected for activations, KV blocks or communication tensors.
Groq packed-weight estimates use the device's supported compute-pipe fallback and charge an explicit analytical vector-dequantization term.
These vector efficiency and instruction-count inputs remain unmeasured assumptions.

## Request geometry and execution contracts

Routing snapshots retain each request's absolute token identity and decode step.
External assignments are validated against that request's position, including requests at different decode positions in one block.
Decode recomputation records query lengths and available KV contexts separately for every request.
Cached keys contribute attention FLOPs and KV reads when a context suffix is rebuilt.
Uneven requests retain separate query shapes.

Wire precision changes create batched encode/decode nodes on the producer and consumer compute resources.
Conversions sharing a producer, receive boundary, rank, dtype and element width execute as one invocation while retaining every logical tensor slice.
Source encoding is shared when the same token is sent to several expert destinations.
The planner also batches messages produced by that shared encoding without changing their payload bytes.
Conversion names record both formats, such as `cast_bf16_to_fp8`, and the table dtype identifies the output format.
Analytical conversion estimates include source reads, target writes, conversion work and kernel overhead.
Strict lookup requires a matching measurement.
Variable-length decode uses one `ragged_generation_attention` query with the exact context-length histogram.
Its analytical work sums per-request FLOPs and KV traffic, applies each request's attention window, and charges one kernel invocation.
Uniform decode continues to use `generation_attention`.
`table_only` requires a matching ragged measurement; `table_first` labels the missing-table fallback as analytical.
Uniform measurements are not silently substituted for a ragged batch.
The model field `qk_norm: true` declares per-head query/key root mean square (RMS) normalization independently of transport protocol.
The Qwen3 catalog enables this field, and AF and ordinary operator-table execution both include its service and parameter storage.

Mixed serving batches are partitioned by logical phase and whether they build context.
Complete prefill and AF decode groups share physical compute resources and join at one serving-step completion barrier.
Recomputation retains decode ownership while rebuilding context with prefill attention kernels.
AF MLA recomputation requires equal-length, whole contexts without cached prefixes or partial chunks.
Unsupported MLA passes fail before ownership or iteration state changes.
Failure or cancellation interrupts sibling groups and releases their resources.

## Results and calibration

The engine includes embedding lookup, final normalization, language-model head and token selection in whole-model latency.
Complete-model prefill chunks omit output-head work when they do not emit logits.
`af_runtime_state` reports topology, component bytes, KV partitions and capacities.
Runtime-state schema version 3.0 removes phase fields from decode request owners and partial destinations.
`af_trace` reports dependency readiness, resource queueing, service time and provenance.
Members of one shared F invocation retain separate dependency spans with the same `execution_group_id` and device-service interval.
`input_ready_at` preserves each member's readiness before shared collection and host submission.
`resource_queue_intervals` records the shared collection wait and device-resource queue separately from host service and host queueing.
Physical queue totals use these intervals once per execution, independently of member order, including compact timing records and clipped measurement windows.
Their local routing edges describe each member, while `ffn_batch_*` details describe the combined provider query.
Resource utilization, operation counts and additive timeline totals count a shared execution once.
Trace consumers aggregating physical service should use `unique_executions` from `TokenSim.af.trace.occupancy`.
Directly summing every member's service duration counts the shared interval repeatedly.
`af_phase_fidelity` reports `prefill_aggregate` for prefill and `decode_af_rank_local` for decode.
The fixed-batch `simulate_decode` helper checks the selected A replica's resident KV capacity and model context limit before submitting work.
The helper requires one A replica for `vllm_afd`; multi-A native execution must submit every peer through the service or executor interfaces.

`communication_timing` separates host submission, transport service and completion work.
Optional host, operation and transport profiles validate their timing scope, runtime identity, environment and workload range before use.
Synthetic test profiles provide regression fixtures only.
Changing runtime configuration requires a new provider/runtime; service and DAG caches retain static estimates without retaining resource availability or completion times.

## Physical communication

Physical AF communication uses the `node-domain-fifo-v1` model.
Each node has the configured number of GPU slots, one shared local communication resource, and independent inter-node transmit and receive resources.
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
Rack-wide NVLink deployments require explicit bandwidth and resource-sharing inputs; changing the node slot count does not calibrate those parameters.
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
`endpoint_transfer_occupancy: wire` reserves endpoint injection during analytical point-to-point serialization and releases it while the message completes its hop latency.
Message consumers still wait for serialization and hop completion.
The default `service` mode holds the endpoints for the complete service duration.
Measured records with inseparable timing and ring collective rounds retain complete service occupancy.
Trace occupancy intervals describe atomic transmit/receive injection reservations and do not resolve packet arrival timing inside switches.
Capacity validation, serving utilization and resource blockers use reservation completion separately from message completion.
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
The cohort-serving API under `TokenSim.af.serving` executes decode streams through `ModelAFExecutor` and exports steady-state token counts, queue delays and resource utilization.
Its complete-model prefill path uses the ordinary operator-table latency backend on colocated ranks and never submits an AF iteration.
Prefill reserves the participating pipeline stage, including its expert ranks, and uses the shared endpoint budgets for model collectives.
Prefill traces contain aggregate model compute and ordinary TP/EP/PP communication without A/F activation transfers.
Its resident KV leases account for each stage's bounded or full-context layer windows and reserve the declared peak context before execution.
This fixed-cohort mode does not page or preempt requests.
KV handoffs verify model/cache identity and source coverage before scheduling copies.
Incremental mode sends only positions absent from the declared warm destination.
Local layers send their retained window tail, and replicated KV heads receive a cross-pool canonical copy followed by destination-local replication.
Cancellation releases every endpoint reservation and leaves destination validity unchanged.
`compute_execution_mode: cuda_graph` selects an analytical GPU device-kernel overhead and charges one host submission per cohort forward in `AFService`.
`graph_submit_us` defaults to 4 microseconds and `graph_device_kernel_us` defaults to 0.2 microseconds.
These costs have no measured calibration and are explicit sensitivity inputs.
CUDA graph launch costs do not represent persistent MegaMoE fusion or GPU communication SM occupancy.
Graph overrides require analytical-only operator fallback to avoid rewriting measured timing scopes.
Groq device service remains unchanged by the GPU graph setting.
`coalesce_ready_transfers: true` batches matching endpoint messages and collectives while retaining payload bytes, tensor slices and producer dependencies.
Endpoint batches wait for all their declared producers.
The default disables batching.
`dag_template_cache_max_tokens` bounds the query-token count eligible for structural template caching.
Zero retains unlimited shape eligibility; larger graphs always use the validated reference planner.
`AFService(retain_trace=False)` validates full native events at forward completion and retains their timing records for serving metrics.
Full exported traces defer immutable tensor and routing serialization until the export boundary.
Explicit A/F domains can share physical workers when their link declares `colocated: true`.
Those workers retain one compute resource and one memory capacity across both capabilities.

## Code layout

`TokenSim/af` groups planning, execution, runtime state, latency providers, communication resources, calibration, caching and traces into separate packages.
AF configuration and topology validation live in `TokenSim/config/af`.
Worker groups and shared cluster construction extend the existing `TokenSim/config/cluster_config.py` entrypoint.
`tests/af` mirrors these responsibilities and keeps shared fixtures under `tests/af/support`.
MLA operator and model tests live in `tests/mla`.
Internal imports refer to the defining modules; the old flat AF forwarding modules have been removed.
