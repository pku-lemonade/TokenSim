# Parallelism and Worker Layouts

TokenSim models two related dimensions:

- Worker roles define where prefill and decode requests are scheduled.
- `parallel_config` defines tensor, pipeline, data, and expert ranks used for
  compute, memory, placement, and communication latency.

## Worker Roles

A cluster is built from `worker_groups` with one of three roles:

| Role | Behavior |
| --- | --- |
| `hybrid` | Accepts both prefill and decode |
| `prefill` | Accepts prefill and produces KV for transfer |
| `decode` | Accepts decode and consumes transferred KV |

`data/clusters/1_a100/h1.json` is a single hybrid worker.
`data/clusters/8_a100/p2d5.json` separates two prefill workers from six decode
workers. After prefill, a request's KV blocks move to a decode worker over the
point-to-point link of the lowest topology level the two workers share; a
`MooncakeConnector` replaces this with Mooncake transfer timing (see
[Mooncake](mooncake.md)).

The prefill and decode pools select workers with `round_robin`,
`least_gpu_memory` or `balanced_load`, set by `--prefill_worker_pool_type`
(default `round_robin`) and `--decode_worker_pool_type` (default
`least_gpu_memory`).

## TP, PP, and DP

For explicit model-parallel topologies, add `parallel_config` to a cluster:

```json
{
  "tensor_parallel_size": 2,
  "pipeline_parallel_size": 2,
  "data_parallel_size": 2,
  "data_parallel_size_local": 2
}
```

When `parallel_config` describes more than one rank, the sum of
`worker_groups[*].num_workers` must equal:

```text
world_size = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
```

Without `parallel_config` (one rank), every worker is an independent replica.

`data/clusters/8_a100/tp2_pp2_dp2.json` is a complete eight-rank example:

```bash
./benchmark.py \
  --batching paged-attn \
  --request_count 8 \
  --cluster ./data/clusters/8_a100/tp2_pp2_dp2.json \
  --model ./data/models/llama-7b.yaml \
  --qps 10 \
  --verbose none
```

Rank mapping uses TP as the fastest-changing dimension, then PP, then DP. For
TP=2, PP=2, DP=2, global ranks 0-3 form DP group 0 and ranks 4-7 form DP group
1. Within each DP group, ranks 0/1 are PP stage 0 and ranks 2/3 are PP stage 1.

`parallel_config` fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `tensor_parallel_size` | 1 | TP ranks per replica |
| `pipeline_parallel_size` | 1 | PP stages per replica |
| `data_parallel_size` | 1 | Replicas |
| `data_parallel_rank` | 0 | Recorded DP rank of this deployment (validated metadata) |
| `data_parallel_size_local` | 1 | DP ranks per node (validated metadata; cannot exceed `data_parallel_size`) |
| `tensor_parallel_collective_latency_scale` | 1.0 | Multiplier on TP collective time |
| `pipeline_parallel_activation_latency_scale` | 1.0 | Multiplier on PP activation-transfer time |
| `enable_expert_parallel` | `false` | Shard MoE experts across ranks |
| `expert_parallel_scope` | `global` | `global` or `per_dp` (see [MoE](moe.md)) |
| `expert_parallel_size` | whole scope | Ranks per expert-parallel group |
| `expert_placement_strategy` | `linear` | `linear` or `round_robin` |
| `all2all_backend` | `allgather_reducescatter` | `allgather_reducescatter`, `naive`, `deepep_high_throughput`, `deepep_low_latency` |

### Tensor Parallelism

TP shards dense projection/attention compute and model/KV memory across TP
ranks. The latency backend queries every GEMM and attention operator with its
rank-local shape (`ceil(heads / TP)` heads, `ceil(inter / TP)` FFN columns) and
adds one all-reduce after the attention output projection and one after the
FFN per layer (expert-parallel MoE layers use dispatch/combine instead of the
FFN all-reduce). KV heads must be divisible by TP, or TP must be a multiple of
the KV head count (heads are then replicated).

### Pipeline Parallelism

PP splits layers, model memory, and KV memory across stages. Remainder layers
are assigned to the earliest stages. The latency backend composes only the
layers owned by a stage (the last stage adds the LM head) and adds one
adjacent-stage activation transfer per step.

### Data Parallelism

DP creates replica groups. New requests are placed round-robin across groups
and keep their assigned `dp_rank` when moving from prefill to decode; the
selected worker policy is then applied within that DP group.
`--sticky_sessions` keeps requests of one conversation (`chat_id`) on the same
replica, and `--router_policy consistent_hash` (session hashing as in
vllm-router) or `fixed_mapping` (`--router_mapping_path`) replace the
round-robin placement. `data_parallel_size_local` and `data_parallel_rank` are
validated and exported as topology metadata; the single-process simulator
instantiates every rank in `world_size`.

Only the TP=0, PP=0 worker in each DP group is directly schedulable. Other
ranks represent the corresponding model shards and participate through
per-rank memory and communication modeling.

By default the DP replicas step independently. `--dp_lockstep` runs them in
synchronized steps, with dummy batches on idle replicas and a finished-state
check every `--dp_sync_interval` steps (default 16); it requires hybrid
workers and PP1. The fused MoE module needs it whenever DP > 1.

## Expert Parallelism

EP is enabled with `enable_expert_parallel`. `expert_parallel_scope` chooses
the ranks that share one copy of the experts: `global` (default) spreads them
over all `tensor_parallel_size * data_parallel_size` ranks of a pipeline stage,
`per_dp` gives every data-parallel replica its own expert-parallel group over
the replica's TP ranks. `expert_parallel_size` states the group size and is
validated against that layout. MoE layers follow their PP stages. EP adds
dispatch/combine and load-imbalance latency and requires a MoE model
configuration. See [MoE](moe.md).

## Configuration Precedence

The effective parallel configuration is selected in this order:

1. CLI values such as `--tensor_parallel_size` override individual fields
   (TP, PP, DP sizes, DP rank and local size, and the expert-parallel
   settings).
2. The cluster's `parallel_config` supplies the base configuration.
3. Missing values default to one rank with expert parallelism disabled.

Changing TP/PP/DP on the CLI does not add workers. The cluster must already
contain exactly the resulting `world_size` workers.

## Communication Topology and Metrics

Workers are placed on a hierarchical topology (`data/topologies/*.yaml`). A
cluster names one with `"topology": "hgx_h100_8x_ndr"` and may pin workers with
`device_indices`; otherwise a two-level topology is synthesized from `networks`
(workers sharing a network form a node joined by the device's scale-up link,
nodes are joined by the network's link class).

Network names must reference catalog link classes. Unknown links are rejected;
they are not replaced by a default bandwidth. Mixed device or network link
types require an explicit topology instead of selecting the first type.

Collectives are priced by `TokenSim/comm/collectives.py`: the TP/EP group is
mapped onto the topology levels it spans and a hierarchical alpha-beta model
(ring / tree / direct per level, NCCL-style) adds latency and bandwidth terms
level by level. When the device's operator data contains a measured `collective`
table for the same operation, dtype, group size and node count, the measured
curve is used instead; a query without a measured row follows
`--latency_fallback` (formula plus a recorded miss under `table_first`, an
error under `table_only`). Point-to-point transfers (PP activations,
prefill-to-decode KV) use the link of the lowest topology level shared by the
two workers. See
[operator-latency-model.md](operator-latency-model.md#6-communication-model).

Results export the effective `parallel_config`, expected and actual rank counts,
per-rank TP/PP/DP IDs and utilization, DP placement counts, the TP and EP group
of rank 0 with the topology level each spans (`parallel_groups`), TP collective
latency, PP transfer latency, EP all-to-all latency, total synchronization
latency, synchronization event count, per-topology-level event counts
(`parallel_link_type_counts`) and how many collective estimates came from
measurements versus the analytical model (`parallel_comm_match_type_counts`).
