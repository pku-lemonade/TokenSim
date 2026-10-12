# Mixture-of-Experts Simulation

In a mixture-of-experts (MoE) layer, a router selects expert feed-forward networks to process each token.
TokenSim models which layers use MoE, the selected experts' compute time and the extra time caused by uneven work across experts.
Expert parallelism (EP) places experts on different GPU workers, called ranks.
Dispatch sends tokens to their selected experts, and combine collects the expert results.
A MoE run needs a model entry in `data/models/*.yaml` with MoE fields under `moe:`.
The model configuration defines layer counts, expert counts and dimensions.
Workload and latency-target settings do not change those dimensions.

Expert compute comes from the `moe` operator table (or the analytical model), queried with the layer's token count, `top_k`, expert count and the effective `tp_size`/`ep_size`.
The routing histogram records how many token-expert assignments each expert receives during a step.
These counts determine the extra time spent waiting for the busiest rank, and the communication model estimates dispatch and combine time.
A fused MoE module combines expert computation and token transfers in one timing estimate, as described in [Fused MoE module](#fused-moe-module).

## Quick Start

The included toy model has four experts and sends each token to two of them, called top-2 routing.
Tensor parallelism (TP=2) divides each replica across two ranks, and data parallelism (DP=4) runs four replicas.
Expert parallelism spans all eight ranks in this example.

```bash
./benchmark.py \
  --batching paged-attn \
  --block_size 16 \
  --request_count 8 \
  --prefill_mean_len 32 \
  --prefill_range_len 0 \
  --decode_mean_len 4 \
  --decode_range_len 0 \
  --cluster ./data/clusters/8_a100/moe_tp2_dp4.json \
  --model ./data/models/moe-toy.yaml \
  --qps 10 \
  --verbose none
```

The cluster uses A100 40 GB devices, which have no operator tables in the bundled data.
Every operator time is therefore estimated by the analytical model, and the run summary records this.

## Model Configuration

Add the following fields under `moe:` in the model YAML.

| Field | Default | Meaning |
| --- | --- | --- |
| `num_experts` | required | Number of routed experts |
| `num_experts_per_tok` | 1 | Experts selected per token (`top_k`) |
| `moe_intermediate_size` | `intermediate_size` | Hidden width of each routed expert |
| `num_shared_experts` | 0 | Shared experts; a dense MLP applied to every token |
| `num_moe_layers` | `num_layers` | Number of MoE layers |
| `first_k_dense_replace` | 0 | Index of the first MoE layer |
| `moe_layer_freq` | 1 | Spacing between MoE layers |
| `interleave_moe_layer_step` | 1 | Validated model metadata; not a separate latency control |
| `routing` | `uniform` | Synthetic routing: a distribution name, or `{distribution, hot_experts, hot_expert_fraction}` |

Shared dimensions such as `hidden_size`, `intermediate_size` and attention head counts belong to the model's top level.
Routing CLI arguments control how tokens are assigned to experts while keeping the model dimensions unchanged.

For an enabled MoE model, `num_experts`, `num_experts_per_tok`, `num_moe_layers`, `moe_layer_freq` and `interleave_moe_layer_step` must be positive.
`num_experts_per_tok` cannot exceed `num_experts`.
The last MoE layer, at index `first_k_dense_replace + (num_moe_layers - 1) * moe_layer_freq`, must lie within the model's layer count.
Unknown fields are rejected.

Shared experts process every token through a feed-forward network of width `num_shared_experts * moe_intermediate_size`.
Their time is estimated with `gemm` queries and, for gated activations, `swiglu` queries.
Their weights are TP-sharded like a dense FFN, or replicated when expert parallelism shards MoE tokens over TP ranks (TP > 1 with several DP replicas or the fused MoE module).
Routed experts are TP-sharded without expert parallelism and kept whole on their EP rank with it.

## Expert Parallelism

Configure expert parallelism inside the cluster's `parallel_config`, as shown below.

```json
{
  "tensor_parallel_size": 2,
  "pipeline_parallel_size": 1,
  "data_parallel_size": 4,
  "enable_expert_parallel": true,
  "expert_parallel_scope": "global",
  "expert_parallel_size": 8,
  "expert_placement_strategy": "linear",
  "all2all_backend": "allgather_reducescatter"
}
```

`expert_parallel_scope` selects the ranks that share a copy of the experts, as described in the following table.

| Scope | Expert-parallel group | Example cluster |
| --- | --- | --- |
| `global` (default) | every `TP * DP` rank of a pipeline stage; one wide-EP group fed by all DP replicas | `data/clusters/8_b200/dep8.json` (TP1, DP8, EP8) |
| `per_dp` | the `TP` ranks of one data-parallel replica; `DP` independent groups, each holding all experts | `data/clusters/64_b200/tp8_dp8_ep.json` (eight TP8 replicas, each its own EP8 group) |

`expert_parallel_size` is the number of ranks in one group.
When omitted, the group covers the entire configured scope.
A `per_dp` group must have exactly `tensor_parallel_size` ranks.
A `global` group must be a multiple of `tensor_parallel_size` that divides `TP * DP`, so each group contains complete replicas.
Any other size, or a size supplied without `enable_expert_parallel`, causes a configuration error.
A group receives the tokens from its `expert_parallel_size / tensor_parallel_size` replicas, which are assumed to advance in synchronized steps.
With `per_dp`, the group receives only its own replica's tokens.

Pipeline parallelism (PP) assigns each MoE layer to its corresponding pipeline stage.
The following strategies control expert placement within a group.

- `linear`, the default, assigns a contiguous range of expert IDs to each rank.
- `round_robin` places expert `i` on rank `i % ep_rank_count`.

Supported all-to-all models are `allgather_reducescatter` (default), `naive`, `deepep_high_throughput` and `deepep_low_latency`.
For DeepEP modes, the simulator uses the selected package's measured `ep_all2all` table when available.
Queries specify tokens per rank, hidden size, `top_k`, expert count, EP group size and node count.
If the requested DeepEP mode has no rows, the simulator tries the other DeepEP mode.
When measurements are unavailable, the communication model estimates two all-to-all transfers from the topology and scales them by `EP_ALL2ALL_MODE_SCALE`.
The factors are 1.5 for `naive`, 1.0 for `allgather_reducescatter`, 0.7 for `deepep_high_throughput` and 0.5 for `deepep_low_latency`, all recorded as grade D assumptions.
Grade D identifies formulas or unmeasured engineering assumptions.
Serving deployments often choose high-throughput kernels for prompt processing (prefill) and low-latency kernels for token generation (decode).
TokenSim applies the configured mode to every step.

CLI flags override cluster values.
The following example gives each DP replica of the toy cluster its own EP group and places experts in round-robin order.

```bash
./benchmark.py \
  --batching paged-attn \
  --block_size 16 \
  --request_count 8 \
  --prefill_mean_len 32 \
  --prefill_range_len 0 \
  --decode_mean_len 4 \
  --decode_range_len 0 \
  --cluster ./data/clusters/8_a100/moe_tp2_dp4.json \
  --model ./data/models/moe-toy.yaml \
  --qps 10 \
  --enable_expert_parallel \
  --expert_parallel_scope per_dp \
  --expert_parallel_size 2 \
  --expert_placement_strategy round_robin \
  --all2all_backend deepep_low_latency \
  --verbose none
```

Dispatch and combine use the configured expert-parallel group.
For a `per_dp` group spanning all eight GPUs of one node, the DeepEP query is `ep_size=8, nodes=1` over NVLink.
Results record the group size, group count and shared topology level in `parallel_groups`, using `ep_group_size`, `ep_group_count` and `ep_group_link`.
The Hopper and Blackwell packages contain DeepEP rows for single-node groups (8 ranks; 4 on GB200/GB300), and the SGLang H100/H200 packages also for groups of 16, 32 and 64 ranks over 2, 4 and 8 nodes.
The following example runs Mixtral-8x7B on eight H100 GPUs with one global EP8 group and uses measured DeepEP tables.

```bash
./benchmark.py \
  --batching paged-attn \
  --model mixtral-8x7b \
  --cluster ./data/clusters/8_h100/moe_ep8_deepep.json \
  --request_count 16 \
  --prefill_mean_len 512 \
  --prefill_range_len 0 \
  --decode_mean_len 32 \
  --decode_range_len 0 \
  --qps 4 \
  --verbose none
```

Using `--enable_expert_parallel` with a dense model is a configuration error.
See [Parallelism](parallelism.md) for the rank count and topology rules.

## Routing Workloads

When requests provide no routing data, TokenSim generates repeatable token counts per expert.
The default is `uniform`.
`skew`, `hot` and `burst` use the same distribution, with a selected set of experts receiving more work.
`hot_expert_fraction`, default 0.5, is the share of token-expert assignments sent to `hot_experts`, default expert 0.
The remaining assignments are spread across the other experts.

```bash
./benchmark.py \
  --batching paged-attn \
  --block_size 16 \
  --request_count 8 \
  --prefill_mean_len 32 \
  --prefill_range_len 0 \
  --decode_mean_len 4 \
  --decode_range_len 0 \
  --cluster ./data/clusters/8_a100/moe_tp2_dp4.json \
  --model ./data/models/moe-toy.yaml \
  --qps 10 \
  --moe_routing_distribution hot \
  --moe_hot_experts 0,1 \
  --moe_hot_expert_fraction 0.8 \
  --verbose none
```

With a skewed routing the `moe` lookup first tries rows measured with the `power_law_1.2` distribution.
Those measurements already include the effect of uneven work, so the simulator adds no extra waiting time for the busiest rank.
The histogram still determines the reported expert load statistics.

For measured traces, supply one of the following fields in the optional metadata of a `json_pairs` record or directly in a `qwen_jsonl` record.

| Field | Scope |
| --- | --- |
| `expert_histogram` | Fallback used for both phases |
| `prefill_expert_histogram` | Used during prefill |
| `decode_expert_histogram` | Used for each decode step |

A histogram can map expert IDs to assignment counts, list expert IDs or list `[expert_id, count]` pairs, as in the following example.

```json
{
  "input_length": 32,
  "output_length": 4,
  "prefill_expert_histogram": {
    "0": 40,
    "1": 20,
    "2": 4
  },
  "decode_expert_histogram": [
    [0, 2],
    [1, 1]
  ]
}
```

Expert IDs must be in `[0, num_experts)` and counts must be non-negative.
A prefill chunk routes its share of the prompt's histogram.
Requests without a histogram use the synthetic distribution.

## Fused MoE Module

`--moe_backend deep_gemm_amxf4_mega_moe` estimates dispatch, routed expert compute and combine together from the measured module selected by `operator_modules.moe`.
For example, `data/models/deepseek-v4-pro.yaml` selects `dsv4_megamoe`.
This mode requires expert parallelism, one pipeline stage (PP1) and a complete EP group on one node.
With DP > 1, it also requires `--dp_lockstep`.
Shared expert costs are accounted for separately, and `--disable_moe_shared_fusion` adds their execution time to the routed expert time.
See [Operator-module tables](operator-latency-model.md#23-operator-module-tables).

## Metrics

The result JSON includes the following metrics.

| Field | Meaning |
| --- | --- |
| `effective_moe_config` | MoE geometry and routing used by the run |
| `moe_expert_placement`, `moe_ep_rank_count` | Strategy, scope, EP rank and group counts, rank-to-expert map, MoE layers per PP stage |
| `moe_per_expert_load`, `moe_per_rank_expert_load` | Routed token-expert pairs per expert and per EP rank |
| `moe_routed_token_count`, `moe_max_expert_load`, `moe_mean_expert_load` | Load totals |
| `moe_expert_load_imbalance_ratio` | Max over mean expert load |
| `moe_compute_latency` | Expert compute time at balanced load, including shared experts |
| `moe_straggler_latency` | Additional time waiting for the busiest rank |
| `moe_all2all_event_count` | Steps with EP dispatch/combine, summed over ranks |
| `parallel_ep_all2all_latency` | Dispatch/combine time (also part of `parallel_latency_total`) |

`parallel_comm_match_type_counts` tells whether communication estimates came from measured tables or the analytical model.
