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
ranks. The roofline adapter adds two hidden-state collective synchronizations
per simulated step. The model's KV head count must be divisible by TP size.

### Pipeline Parallelism

PP splits layers, model memory, and KV memory across stages. Remainder layers are
assigned to the earliest stages. The latency model passes `Pipeline_Stage` to
TransformerRoofline and adds adjacent-stage activation transfer latency.

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

## Configuration Precedence

The effective parallel configuration is selected in this order:

1. CLI values such as `--tensor_parallel_size` override individual fields.
2. The cluster's `parallel_config` supplies the base configuration.
3. If the cluster has none, the model PSLA's `parallel_config` is used.
4. Missing values default to one rank with expert parallelism disabled.

Changing TP/PP/DP on the CLI does not add workers. The cluster must already
contain exactly the resulting `world_size` workers.

## Communication Topology and Metrics

For peers on the same named network, TokenSim prefers the accelerator NVLink
model when available. Otherwise it uses the configured network link; if neither
is available, the communication event is treated as local with zero latency.
TP uses the slowest peer link in the TP group, PP uses the adjacent stage, and EP
uses the slowest participating expert peer.

Results export the effective `parallel_config`, expected and actual rank counts,
per-rank TP/PP/DP IDs and utilization, DP placement counts, TP collective
latency, PP transfer latency, EP all-to-all latency, total synchronization
latency, synchronization event count, and roofline conversion count.
