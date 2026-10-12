# Parallelism and Worker Layouts

TokenSim describes a cluster through worker roles and its parallel configuration.

Prompt processing is called prefill, and output token generation is called decode.
A rank identifies a worker's position in the parallel configuration.
The attention key/value (KV) cache stores the context needed to generate later tokens.

- Worker roles define where prefill and decode requests are scheduled.
- `parallel_config` defines tensor, pipeline, data, and expert ranks used for compute, memory, placement, and communication latency.

## Worker Roles

A cluster's `worker_groups` use the following roles.

| Role | Behavior |
| --- | --- |
| `hybrid` | Accepts both prefill and decode |
| `prefill` | Accepts prefill and produces KV for transfer |
| `decode` | Accepts decode and consumes transferred KV |

`data/clusters/1_a100/h1.json` is a single hybrid worker.
`data/clusters/8_a100/p2d5.json` separates two prefill workers from six decode workers.
After prefill, a request's KV blocks move to a decode worker over the link at the smallest topology level containing both workers.
For example, workers on the same node use that node's local link, while workers on different nodes use their shared network level.
With `MooncakeConnector`, transfer times follow the rules in [Mooncake](mooncake.md).

The prefill and decode pools select workers with `round_robin`, `least_gpu_memory` or `balanced_load`.
Set these policies with `--prefill_worker_pool_type` and `--decode_worker_pool_type`, whose defaults are `round_robin` and `least_gpu_memory` respectively.

## TP, PP, and DP

Tensor parallelism (TP) divides operations within a model layer, pipeline parallelism (PP) divides layers into stages, and data parallelism (DP) runs replicas that handle separate request batches.
Add `parallel_config` to the cluster configuration to specify these sizes.

```json
{
  "tensor_parallel_size": 2,
  "pipeline_parallel_size": 2,
  "data_parallel_size": 2,
  "data_parallel_size_local": 2
}
```

When `parallel_config` describes more than one rank, the sum of `worker_groups[*].num_workers` must equal the following `world_size`.

```text
world_size = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
```

Without `parallel_config` (one rank), every worker is an independent replica.

The following command uses the eight-rank configuration in `data/clusters/8_a100/tp2_pp2_dp2.json`.

```bash
./benchmark.py \
  --batching paged-attn \
  --request_count 8 \
  --cluster ./data/clusters/8_a100/tp2_pp2_dp2.json \
  --model ./data/models/llama-7b.yaml \
  --qps 10 \
  --verbose none
```

Rank mapping uses TP as the fastest-changing dimension, then PP, then DP.
For TP=2, PP=2, DP=2, global ranks 0-3 form DP group 0 and ranks 4-7 form DP group 1.
Within each DP group, ranks 0/1 are PP stage 0 and ranks 2/3 are PP stage 1.

The following table describes the `parallel_config` fields.

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

TP divides projection and attention work, model weights and KV cache memory among its ranks.
The latency backend looks up matrix multiplication (GEMM) and attention times using each rank's local dimensions: `ceil(heads / TP)` heads and `ceil(inter / TP)` feed-forward network (FFN) columns.
Each layer adds an all-reduce, which combines results across ranks, after the attention output projection and after the FFN.
Mixture-of-experts (MoE) layers with expert parallelism distribute tokens to experts and collect the results in place of the FFN all-reduce.
The KV head count must be divisible by TP, or TP must be a multiple of the KV head count so that heads can be replicated across ranks.

### Pipeline Parallelism

PP splits layers, model memory, and KV memory across stages.
Remainder layers are assigned to the earliest stages.
The latency backend estimates only the layers assigned to each stage and adds one activation transfer between adjacent stages per step.
The last stage also includes the output projection that produces vocabulary logits, called the LM head.

### Data Parallelism

DP creates replica groups.
New requests are assigned to replica groups in round-robin order and keep their `dp_rank` when moving from prefill to decode.
The selected worker policy chooses a worker within that DP group.
`--sticky_sessions` keeps requests of one conversation (`chat_id`) on the same replica, and `--router_policy consistent_hash` (session hashing as in vllm-router) or `fixed_mapping` (`--router_mapping_path`) replace the round-robin placement.
The simulator validates `data_parallel_size_local` and `data_parallel_rank` and records them in the result's topology metadata.
It represents every rank in `world_size` within one simulation process.

Only the TP=0, PP=0 worker in each DP group is directly schedulable.
Other ranks represent the corresponding model shards and participate through per-rank memory and communication modeling.

By default the DP replicas step independently.
`--dp_lockstep` advances replicas in synchronized steps and supplies dummy batches for idle replicas.
It checks whether execution has finished every `--dp_sync_interval` steps, with a default interval of 16.
This mode requires hybrid workers and one pipeline stage (PP1).
The fused MoE module needs it whenever DP > 1.

## Expert Parallelism

Expert parallelism (EP) distributes MoE experts across ranks and is enabled with `enable_expert_parallel`.
`expert_parallel_scope` chooses which ranks share a copy of the experts.
The default, `global`, distributes experts across all `tensor_parallel_size * data_parallel_size` ranks of a pipeline stage.
`per_dp` gives each data-parallel replica its own group of experts distributed across that replica's TP ranks.
`expert_parallel_size` states the group size and is validated against that layout.
MoE layers follow their PP stages.
EP adds time for distributing tokens to experts (dispatch), collecting expert results (combine) and waiting for ranks with more work.
It requires a MoE model configuration.
See [MoE](moe.md).

## Configuration Precedence

The effective parallel configuration follows this precedence order.

1. CLI values such as `--tensor_parallel_size` override individual fields (TP, PP, DP sizes, DP rank and local size, and the expert-parallel settings).
2. The cluster's `parallel_config` supplies the base configuration.
3. Missing values default to one rank with expert parallelism disabled.

Changing TP/PP/DP on the CLI does not add workers.
The cluster must already contain exactly the resulting `world_size` workers.

## Communication Topology and Metrics

Workers are placed on a hierarchical topology (`data/topologies/*.yaml`).
A cluster selects a topology with a field such as `"topology": "hgx_h100_8x_ndr"` and can assign workers to specific devices with `device_indices`.
When no explicit topology is supplied, the simulator builds two levels from `networks`.
Workers sharing a network form a node connected by the device's local interconnect, and the network's link class connects nodes.

Network names must reference catalog link classes.
An unknown link name causes a configuration error.
Clusters with mixed device or network link types require an explicit topology.

`TokenSim/comm/collectives.py` estimates communication operations involving a group of ranks, called collectives.
For each topology level crossed by the TP or EP group, the alpha-beta model adds fixed transfer latency and a term based on message size and bandwidth.
The topology selects a ring, tree or direct communication algorithm at each level.
If the device's `collective` table contains matching measurements for the operation, data type, group size and node count, the simulator uses that curve.
Missing measurements follow `--latency_fallback`: `table_first` uses a formula and records the missing query, while `table_only` returns an error.
Point-to-point transfers (PP activations, prefill-to-decode KV) use the link of the lowest topology level shared by the two workers.
See [operator-latency-model.md](operator-latency-model.md#6-communication-model).

Results record the effective `parallel_config`, expected and actual rank counts, each rank's TP/PP/DP identifiers and utilization, and request placement counts for DP groups.
`parallel_groups` describes rank 0's TP and EP groups and the topology levels they span.
Timing fields include TP collectives, PP transfers, EP all-to-all communication and total synchronization time.
Results also record synchronization event counts, events at each topology level (`parallel_link_type_counts`) and estimates from measurements or formulas (`parallel_comm_match_type_counts`).
