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
workers. Add `P2PConnector` or `MooncakeConnector` to model KV movement between
the two pools; see [Mooncake](mooncake.md).

The prefill and decode pools support `round_robin`, `least_gpu_memory`, and
`balanced_load` selection through `--prefill_worker_pool_type` and
`--decode_worker_pool_type`.

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

The sum of `worker_groups[*].num_workers` must equal:

```text
world_size = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
```

`data/clusters/8_a100/tp2_pp2_dp2.json` is a complete eight-rank example:

```bash
./benchmark.py \
  --batching paged-attn \
  --request_count 8 \
  --cluster ./data/clusters/8_a100/tp2_pp2_dp2.json \
  --model ./data/psla/llama-7b.json \
  --qps 10 \
  --verbose none
```

Rank mapping uses TP as the fastest-changing dimension, then PP, then DP. For
TP=2, PP=2, DP=2, global ranks 0-3 form DP group 0 and ranks 4-7 form DP group
1. Within each DP group, ranks 0/1 are PP stage 0 and ranks 2/3 are PP stage 1.

### Tensor Parallelism

TP shards dense projection/attention compute and model/KV memory across TP
ranks. The latency backend queries every GEMM and attention operator with its
rank-local shape (`ceil(heads / TP)` heads, `ceil(inter / TP)` FFN columns) and
adds one all-reduce after the attention output projection and one after the FFN
per layer. KV heads must be divisible by TP, or TP must be a multiple of the KV
head count (heads are then replicated).

### Pipeline Parallelism

PP splits layers, model memory, and KV memory across stages. Remainder layers are
assigned to the earliest stages. The latency backend composes only the layers
owned by a stage (the last stage adds the LM head) and adds one adjacent-stage
activation transfer per step.

### Data Parallelism

DP creates replica groups. New requests are placed round-robin across groups and
keep their assigned `dp_rank` when moving from prefill to decode. The selected
worker policy is then applied within that DP group. `data_parallel_size_local`
and `data_parallel_rank` are validated and exported as topology metadata; the
current single-process simulator still instantiates every rank in `world_size`.

Only the TP=0, PP=0 worker in each DP group is directly schedulable. Other ranks
represent the corresponding model shards and participate through per-rank
memory and communication modeling.

## Expert Parallelism

EP is enabled with `enable_expert_parallel`. Experts are placed across
`tensor_parallel_size * data_parallel_size` ranks, while MoE layers follow their
PP stages. EP adds all-to-all and load-imbalance latency. It requires a MoE model
configuration. See [MoE](moe.md).

### AF execution domains

AF uses a stricter topology than the non-AF path. One logical instance is one
DP replica: an A replica contains `TP` ranks, while an F replica contains
`EP * TP` ranks. F expert placement is fully replicated inside every DP
replica. The AF planner executes rank-local work and TP collectives; these sizes
are not configuration-only metadata. The A replica leader runs Router/Top-K
after A TP compute/collective. A-to-F directly scatters routed activations from
that leader to the resident Expert ranks, and F-to-A gathers the partial Expert
results back for combination. These two cross-domain nodes are the EP
dispatch/combine; the same payload is not moved again through synthetic
F-internal nodes.

An Expert histogram counts routed token-expert assignments, so its total is
`source_tokens * top_k`. Expert compute consumes that count directly and never
multiplies by `top_k` again. F-domain TP shards the Expert or dense FFN
intermediate dimension on every rank; the corresponding TP collective remains
a separate communication cost.

AF currently requires `PP=1` in both domains because pipeline stages,
activation send/receive and bubble scheduling are not implemented for the
Attention-FFN split. See [AF separation](af-separation.md) and the executable
`data/clusters/af/tp2_dp2_ep2.json` example.

When AF is enabled, domain-local parallel values override the unrelated global
`parallel_config` for AF ranks. `AFRuntimePlan` is compiled once and shared by
the Engine, Planner, memory accounting, and validation. Each A DP replica has
one scheduler leader and one logical request queue, while every A TP rank owns
its own model and KV shard state. F ranks own only the Expert/FFN components
selected by their EP placement. Each A DP replica's `tp_rank=0, ep_rank=0`
leader additionally owns Router; F ranks do not own Router weights.

One logical A KV block maps atomically to one local shard on every required TP
rank. The replica capacity is the minimum rank-local capacity, not the leader
capacity and not the sum across TP ranks. KV-head partitioning also uses the A
domain TP size; small MQA/GQA head counts are explicitly replicated when the
replication groups divide evenly.

Separate Prefill and Decode A domains may share one physical TP group when
their TP/DP sizes and rank-to-worker mappings are identical. The aliases then
share one scheduler, request queue, and KV state. Non-equivalent aliases fail
at startup instead of creating conflicting logical state on the same devices.

The result field `af_runtime_state` reports the compiled topology, component
bytes, local KV bytes/capacity/usage, and replica logical capacity/usage. It is
the memory counterpart to the compute/communication events in `af_trace`.

## Configuration Precedence

The effective parallel configuration is selected in this order:

1. CLI values such as `--tensor_parallel_size` override individual fields.
2. The cluster's `parallel_config` supplies the base configuration.
3. If the cluster has none, the model PSLA's `parallel_config` is used.
4. Missing values default to one rank with expert parallelism disabled.

Changing TP/PP/DP on the CLI does not add workers. The cluster must already
contain exactly the resulting `world_size` workers.

## Communication Topology and Metrics

Workers are placed on a hierarchical topology (`data/topologies/*.yaml`). A
cluster names one with `"topology": "hgx_h100_8x_ndr"` and may pin workers with
`device_indices`; otherwise a two-level topology is synthesized from `networks`
(workers sharing a network form a node joined by the device's scale-up link,
nodes are joined by the network's link class).

Collectives are priced by `TokenSim/comm/collectives.py`: the TP/EP group is
mapped onto the topology levels it spans and a hierarchical alpha-beta model
(ring / tree / direct per level, NCCL-style) adds latency and bandwidth terms
level by level. When the device's operator data contains a measured `collective`
table for the same operation, dtype, group size and node count, the measured
curve is used instead. Point-to-point KV transfers use the link of the lowest
topology level shared by the two workers. See
[operator-latency-model.md](operator-latency-model.md#5-通信模型).

Results export the effective `parallel_config`, expected and actual rank counts,
per-rank TP/PP/DP IDs and utilization, DP placement counts, TP collective
latency, PP transfer latency, EP all-to-all latency, total synchronization
latency, synchronization event count, per-topology-level event counts
(`parallel_link_type_counts`) and how many collective estimates came from
measurements versus the analytical model (`parallel_comm_match_type_counts`).
