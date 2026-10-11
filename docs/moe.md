# Mixture-of-Experts Simulation

TokenSim models MoE layer placement, routed expert compute, expert load
imbalance, and expert-parallel (EP) dispatch/combine communication. A MoE run
needs a model entry in `data/models/*.yaml` with MoE fields under `moe:`. The
selected model is the sole source of its layer and expert geometry; workload
and SLO configuration contain no model metadata.

Expert compute comes from the `moe` operator table (or the analytical model),
queried with the layer's token count, `top_k`, expert count and the effective
`tp_size`/`ep_size`. The step's routing histogram adds straggler time for the
busiest expert rank, and the dispatch/combine pair is priced by the
communication model. A fused expert-parallel MoE module can replace both; see
[Fused MoE module](#fused-moe-module).

## Quick Start

The included toy model uses four experts, top-2 routing, TP=2, DP=4, and expert
parallelism across all eight ranks:

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

The A100 40 GB part in this cluster has no operator tables, so every operator
is priced by the analytical model; the run summary says so.

## Model Configuration

Add these fields under `moe:` in the model YAML:

| Field | Default | Meaning |
| --- | --- | --- |
| `num_experts` | required | Number of routed experts |
| `num_experts_per_tok` | 1 | Experts selected per token (`top_k`) |
| `moe_intermediate_size` | `intermediate_size` | Hidden width of each routed expert |
| `num_shared_experts` | 0 | Shared experts; counted in parameter memory |
| `num_moe_layers` | `num_layers` | Number of MoE layers |
| `first_k_dense_replace` | 0 | Index of the first MoE layer |
| `moe_layer_freq` | 1 | Spacing between MoE layers |
| `interleave_moe_layer_step` | 1 | Validated model metadata; not a separate latency control |
| `routing` | `uniform` | Synthetic routing: a distribution name, or `{distribution, hot_experts, hot_expert_fraction}` |

Shared dimensions such as `hidden_size`, `intermediate_size` and attention head
counts belong to the model's top level. Routing CLI arguments override only
the routing distribution; they do not change the model's geometry.

For an enabled MoE model, `num_experts`, `num_experts_per_tok`,
`num_moe_layers`, `moe_layer_freq` and `interleave_moe_layer_step` must be
positive, `num_experts_per_tok` cannot exceed `num_experts`, and the last MoE
layer (`first_k_dense_replace + (num_moe_layers - 1) * moe_layer_freq`) must
lie inside the model's layers. Unknown fields are rejected.

Shared experts are counted in weight memory. The generic MoE path does not
price their compute; the fused MoE module path does.

## Expert Parallelism

Expert parallelism is configured inside the cluster's `parallel_config`:

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

`expert_parallel_scope` says which ranks share one copy of the experts:

| Scope | Expert-parallel group | Example cluster |
| --- | --- | --- |
| `global` (default) | every `TP * DP` rank of a pipeline stage; one wide-EP group fed by all DP replicas | `data/clusters/8_b200/dep8.json` (TP1, DP8, EP8) |
| `per_dp` | the `TP` ranks of one data-parallel replica; `DP` independent groups, each holding all experts | `data/clusters/64_b200/tp8_dp8_ep.json` (eight TP8 replicas, each its own EP8 group) |

`expert_parallel_size` is the number of ranks in one group. It may be omitted
(the whole scope), but stating it makes the configuration self-checking: a
`per_dp` group must have exactly `tensor_parallel_size` ranks, and a `global`
group must be a multiple of `tensor_parallel_size` that divides `TP * DP`
(groups of whole replicas). Any other value, or a size without
`enable_expert_parallel`, is a configuration error, so a cluster file cannot
silently turn EP8 into EP64. The group size also decides how many tokens a MoE
layer sees: a group receives the tokens of the
`expert_parallel_size / tensor_parallel_size` replicas it contains, which are
assumed to step in lockstep; with `per_dp` that is the replica's own tokens.

Pipeline parallelism assigns MoE layers to their owning PP stage. Supported
placement strategies are:

- `linear` (default): contiguous expert ID ranges per expert rank.
- `round_robin`: expert `i` is placed on rank `i % ep_rank_count`.

Supported all-to-all models are `allgather_reducescatter` (default), `naive`,
`deepep_high_throughput` and `deepep_low_latency`. The two DeepEP modes are
priced from the package's measured `ep_all2all` table (AIConfigurator's DeepEP
dispatch/combine curves, keyed by tokens per rank, hidden size, `top_k`,
expert count, EP size and node count) when the selected operator backend ships
one; the other DeepEP mode is tried when the requested one has no rows.
Otherwise every mode is priced as two topology-derived all-to-all transfers
scaled by `EP_ALL2ALL_MODE_SCALE` (naive 1.5, allgather_reducescatter 1.0,
deepep_high_throughput 0.7, deepep_low_latency 0.5; grade D assumptions).
Deployments often use the high-throughput kernels for prefill and the
low-latency kernels for decode; TokenSim applies one mode to every step.

CLI flags override cluster values. For example, one EP group per DP replica of
the toy cluster, with round-robin placement:

```bash
./benchmark.py \
  --batching paged-attn --block_size 16 --request_count 8 \
  --prefill_mean_len 32 --prefill_range_len 0 \
  --decode_mean_len 4 --decode_range_len 0 \
  --cluster ./data/clusters/8_a100/moe_tp2_dp4.json \
  --model ./data/models/moe-toy.yaml --qps 10 \
  --enable_expert_parallel \
  --expert_parallel_scope per_dp \
  --expert_parallel_size 2 \
  --expert_placement_strategy round_robin \
  --all2all_backend deepep_low_latency \
  --verbose none
```

The dispatch/combine group is exactly the expert-parallel group, so with
`per_dp` on an 8-GPU node the DeepEP query is `ep_size=8, nodes=1` over
NVLink; results export the group and the topology level it spans under
`parallel_groups` (`ep_group_size`, `ep_group_count`, `ep_group_link`). The
Hopper and Blackwell packages contain DeepEP rows for single-node groups (8
ranks; 4 on GB200/GB300), and the SGLang H100/H200 packages also for groups of
16, 32 and 64 ranks over 2, 4 and 8 nodes. Mixtral-8x7B on eight H100 GPUs
with one global EP8 group priced from the measured DeepEP tables:

```bash
./benchmark.py \
  --batching paged-attn --model mixtral-8x7b \
  --cluster ./data/clusters/8_h100/moe_ep8_deepep.json \
  --request_count 16 --prefill_mean_len 512 --prefill_range_len 0 \
  --decode_mean_len 32 --decode_range_len 0 --qps 4 \
  --verbose none
```

Using `--enable_expert_parallel` with a dense model is a configuration error.
See [Parallelism](parallelism.md) for the rank count and topology rules.

## Routing Workloads

Without per-request routing data, TokenSim generates deterministic histograms.
The default is `uniform`. `skew`, `hot` and `burst` currently share one
hot/cold split: `hot_expert_fraction` of the routed token-expert pairs
(default 0.5) go to the `hot_experts` (default expert 0) and the rest are
spread over the other experts:

```bash
./benchmark.py \
  --batching paged-attn --block_size 16 --request_count 8 \
  --prefill_mean_len 32 --prefill_range_len 0 \
  --decode_mean_len 4 --decode_range_len 0 \
  --cluster ./data/clusters/8_a100/moe_tp2_dp4.json \
  --model ./data/models/moe-toy.yaml --qps 10 \
  --moe_routing_distribution hot \
  --moe_hot_experts 0,1 \
  --moe_hot_expert_fraction 0.8 \
  --verbose none
```

With a skewed routing the `moe` lookup first tries rows measured with the
`power_law_1.2` distribution. Such rows already contain the skew, so no
straggler time is added on top of them; the histogram then only changes the
reported load statistics.

For measured traces, provide one of these fields in the optional metadata of a
`json_pairs` record or directly in a `qwen_jsonl` record:

| Field | Scope |
| --- | --- |
| `expert_histogram` | Fallback used for both phases |
| `prefill_expert_histogram` | Used during prefill |
| `decode_expert_histogram` | Used for each decode step |

A histogram may be an object mapping expert IDs to route counts, a list of
expert IDs, or a list of `[expert_id, count]` pairs:

```json
{
  "input_length": 32,
  "output_length": 4,
  "prefill_expert_histogram": {"0": 40, "1": 20, "2": 4},
  "decode_expert_histogram": [[0, 2], [1, 1]]
}
```

Expert IDs must be in `[0, num_experts)` and counts must be non-negative. A
prefill chunk routes its share of the prompt's histogram. Requests without a
histogram use the synthetic distribution.

## Fused MoE Module

`--moe_backend deep_gemm_amxf4_mega_moe` prices routed dispatch, expert
compute and combine together from a measured fused-MoE module table bound in
the model's `operator_modules.moe` (`data/models/deepseek-v4-pro.yaml` binds
`dsv4_megamoe`). It requires expert parallelism, PP1 and a complete
single-node EP group, and with DP > 1 also `--dp_lockstep`. Shared experts are
then priced separately (`--disable_moe_shared_fusion` prices them serially).
See [Operator-module tables](operator-latency-model.md#23-operator-module-tables).

## Metrics

The result JSON includes:

| Field | Meaning |
| --- | --- |
| `effective_moe_config` | MoE geometry and routing used by the run |
| `moe_expert_placement`, `moe_ep_rank_count` | Strategy, scope, EP rank and group counts, rank-to-expert map, MoE layers per PP stage |
| `moe_per_expert_load`, `moe_per_rank_expert_load` | Routed token-expert pairs per expert and per EP rank |
| `moe_routed_token_count`, `moe_max_expert_load`, `moe_mean_expert_load` | Load totals |
| `moe_expert_load_imbalance_ratio` | Max over mean expert load |
| `moe_compute_latency` | Expert compute time at balanced load |
| `moe_straggler_latency` | Additional time waiting for the busiest rank |
| `moe_all2all_event_count` | Steps with EP dispatch/combine, summed over ranks |
| `parallel_ep_all2all_latency` | Dispatch/combine time (also part of `parallel_latency_total`) |

`parallel_comm_match_type_counts` tells whether communication estimates came
from measured tables or the analytical model.
