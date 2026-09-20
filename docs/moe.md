# Mixture-of-Experts Simulation

TokenSim models MoE layer placement, routed expert compute, expert load
imbalance, and expert-parallel all-to-all communication. A MoE run needs a
model entry in `data/models/*.yaml` (MoE fields under `moe:`) or MoE metadata in
the PSLA file under `data/psla/`; PSLA metadata overrides the catalog entry.
Expert compute comes from the `moe` operator table (or the analytical model)
queried with the layer's token count, `top_k`, expert count and the effective
`tp_size`/`ep_size`; the routing histogram scales it by the per-rank load
imbalance and the dispatch/combine all-to-all is priced by the communication
model.

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
  --model ./data/psla/moe-toy.json \
  --qps 10 \
  --verbose none
```

## Model Configuration

Add these fields to a PSLA model configuration:

| Field | Meaning |
| --- | --- |
| `is_moe_model` | Enables MoE validation and latency modeling |
| `num_experts` | Number of routed experts |
| `num_experts_per_tok` | Experts selected per token |
| `moe_intermediate_size` | Hidden width of each routed expert |
| `num_shared_experts` | Shared experts included in parameter memory |
| `num_moe_layers` | Number of MoE layers |
| `first_k_dense_replace` | Index of the first MoE layer |
| `moe_layer_freq` | Spacing between MoE layers |
| `interleave_moe_layer_step` | Validated model metadata; not a separate latency control today |
| `hidden_size` | Model hidden width used for compute and communication |
| `intermediate_size` | Dense FFN width used outside MoE layers |
| `num_attention_heads` | Attention head count metadata |
| `num_key_value_heads` | KV head count metadata |

`num_experts`, `num_experts_per_tok`, `num_moe_layers`, `moe_layer_freq`, and
`interleave_moe_layer_step` must be positive for an enabled MoE model, and
`num_experts_per_tok` cannot exceed `num_experts`.

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

| Scope | Expert-parallel group | Typical deployment |
| --- | --- | --- |
| `global` (default) | every `TP * DP` rank of a pipeline stage; one wide-EP group fed by all DP replicas | vLLM / SGLang wide EP |
| `per_dp` | the `TP` ranks of one data-parallel replica; `DP` independent groups, each holding all experts | InferenceX AgentX points: 64 GPUs = 8 x (TP8, EP8) with DP attention |

`expert_parallel_size` is the number of ranks in one group. It may be omitted
(the whole scope), but stating it makes the configuration self-checking: a
`per_dp` group must have exactly `tensor_parallel_size` ranks, and a `global`
group must be a multiple of `tensor_parallel_size` that divides `TP * DP`
(groups of whole replicas). Any other value is a configuration error, so a
cluster file cannot silently turn EP8 into EP64. The scope also decides how
many tokens a MoE layer sees: with `global` the group processes every
replica's tokens in lockstep, with `per_dp` only the replica's own tokens.

Pipeline parallelism assigns MoE layers to their owning PP stage. Supported
placement strategies are:

- `linear`: contiguous expert ID ranges per expert rank.
- `round_robin`: expert `i` is placed on rank `i % ep_rank_count`.

Supported all-to-all models are `naive`, `allgather_reducescatter`,
`deepep_high_throughput`, and `deepep_low_latency`. The two DeepEP modes are
priced from the device package's measured `ep_all2all` table (AIConfigurator's
DeepEP dispatch/combine curves, keyed by tokens per rank, hidden size, `top_k`,
expert count, EP size and node count) when the selected operator backend ships
one; otherwise every mode falls back to two topology-derived all-to-all
transfers scaled by `EP_ALL2ALL_MODE_SCALE`. Real deployments use the
high-throughput kernels for prefill and the low-latency kernels for decode;
TokenSim currently applies one mode to both phases. CLI flags override cluster
values:

```bash
--enable_expert_parallel \
--expert_parallel_scope per_dp \
--expert_parallel_size 8 \
--expert_placement_strategy round_robin \
--all2all_backend deepep_low_latency
```

The dispatch/combine group is exactly the expert-parallel group, so with
`per_dp` on an 8-GPU-per-node cluster the DeepEP query is `ep_size=8, nodes=1`
over NVLink; results export the group and the topology level it spans under
`parallel_groups` (`ep_group_size`, `ep_group_count`, `ep_group_link`).

Using `--enable_expert_parallel` with a dense model is a configuration error.
See [Parallelism](parallelism.md) for the rank count and topology rules.

## Routing Workloads

Without per-request routing data, TokenSim generates deterministic histograms.
The default is `uniform`. `skew`, `hot`, and `burst` use the configured hot/cold
split in the current model:

```bash
--moe_routing_distribution hot \
--moe_hot_experts 0,1 \
--moe_hot_expert_fraction 0.8
```

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

Expert IDs must be in `[0, num_experts)` and counts must be non-negative.

## Metrics

The result JSON includes the effective MoE configuration and placement, routed
load per expert/rank, imbalance, MoE compute latency, all-to-all latency,
straggler latency, and all-to-all event counts. Parallel communication totals
also include `parallel_ep_all2all_latency`.
