# DeepSeek-V3 multi-head latent attention (MLA)

Multi-head latent attention (MLA) compresses the key/value (KV) state retained for each token and changes the attention projections used during decode.
TokenSim represents DeepSeek-V3 with the [catalog model](../data/models/deepseek-v3-671b-mla.yaml), whose architecture metadata references the [official configuration](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) and a [checked-in snapshot](../data/models/deepseek-v3-hf-config.json).
The model supplies attention geometry, KV storage and operator shapes for latency estimation without loading checkpoint weights.

## Model geometry

The model separates the attention head's content dimensions from its rotary position embedding (RoPE) dimensions.
Its mixture-of-experts (MoE) structure follows three dense layers with 58 expert layers.

| Parameter | Catalog value |
| --- | --- |
| Hidden size | 7168 |
| Transformer layers | 61 |
| Query heads | 128 |
| Query compression rank | 1536 |
| KV compression rank | 512 |
| Content dimensions per query/key head | 128 |
| RoPE dimensions per query/key head | 64 |
| Value dimensions per head | 128 |
| Dense layers / MoE layers | 3 / 58 |
| Routed experts / selected experts per token | 256 / 8 |
| Shared experts | 1 |
| Vocabulary size | 129280 |
| Catalog context limit | 163840 tokens |

A query/key head has `128 + 64 = 192` elements before decode absorption.
The query compression rank controls projection work and weight storage, while the KV compression rank controls both the retained cache and compressed attention work.
These ranks have separate roles in the model.

## Compressed KV cache

Each token retains one 512-element KV latent vector and one 64-element RoPE key vector per layer.
The cache width is therefore

$$
d_{\mathrm{cache}} = r_{\mathrm{KV}} + d_{\mathrm{RoPE}} = 512 + 64 = 576.
$$

At bfloat16 (BF16) precision, this is `576 * 2 = 1152` bytes per token per layer.
The cached latent vector is shared by the query heads; the cache does not store a separate expanded key and value vector for every head.

Tensor parallelism (TP) shards query heads and the corresponding head-specific projections.
The current cache layout replicates the compressed KV state on each TP rank that owns the layer.
Increasing TP reduces the local head count while retaining the 576-element cache width on each rank.

For a rank covering `L` layers and requests with resident context lengths `s_i`, the logical cache allocation is

$$
M_{\mathrm{KV}} = 1152 L \sum_i s_i \quad \text{bytes at BF16}.
$$

A single 8192-token context covering all 61 layers therefore requires 549 mebibytes (MiB) of logical KV storage per rank.
This calculation excludes block padding, model weights and execution workspaces.
Configured KV block sizes determine the padded allocation used by capacity checks.

## Attention computation

The latency model decomposes MLA into projections, normalization, positional work and an attention core.
Its decode path uses absorbed attention, with these stages for each new token:

- Compress the query from hidden size 7168 to rank 1536, normalize it and project it into the local query heads.
- Project the hidden state into the 576-element KV-plus-RoPE representation and normalize the 512-element latent component.
- Apply RoPE to the positional query and key components.
- Absorb the key expansion into each head's content query, mapping 128 content dimensions into the 512-dimensional latent space.
- Compute attention scores and the weighted aggregation over the compressed KV cache.
- Expand each head's aggregated latent value from 512 to 128 dimensions and apply the output projection.

The compressed attention core includes softmax and the cache read/write work.
Query absorption, value expansion and output projection have separate service estimates.
Decode accepts one context length per query token, so batches with different request lengths retain their actual cache-read workload.

Full-sequence prefill uses an expanded attention path.
It expands the latent KV representation into the local key and value heads before causal attention, then retains the compressed cache for later decode.
This path has a KV-expansion estimate and omits the separate decode query-absorption and post-attention value-expansion operations.
The [standalone MLA operator adapter](../TokenSim/operator_data/mla.py) requires equal-length full sequences for prefill and rejects cached prefixes or partial chunks.

Dense feed-forward networks (FFNs), routed experts, shared experts, model boundaries and required communication contribute separately to whole-model latency.
An MLA-core measurement alone covers the attention core defined here.

## Operator data and estimation

| Operator table | Modeled MLA work |
| --- | --- |
| `gemm` | Query/KV projections, prefill KV expansion and output projection |
| `batched_gemm` | Per-head query absorption and value expansion during decode |
| `elementwise` | Normalization, RoPE and configured elementwise operations |
| `mla_attention` | Phase-specific attention core, including its cache accesses |

MLA-core lookup identifies the phase, execution mode, batch/query lengths, context-length distribution, local query-head count, latent dimensions and activation/cache precision.
The query compression rank affects projection estimates and does not change the compressed attention core's cache width.
Measurements must match the intended operator boundary to avoid charging a fused kernel together with its separately modeled components.

`table_first` uses compatible measurements before analytical fallback.
`table_only` rejects missing shapes, and `analytical_only` derives work from the supplied geometry and device parameters.
Reported provenance distinguishes measurements, interpolation, extrapolation and analytical estimates.
See [operator latency data](operator-latency-model.md) for package generation and lookup policies.

## Inspect the catalog model

Run this example from the repository root:

```python
from TokenSim.hardware import HardwareContext

model = HardwareContext.load("data").model("deepseek-v3-671b-mla")
shape = model.mla_dimensions.local_shape(model.num_attention_heads, tp=8)

print(model.num_layers, model.num_attention_heads)  # 61 128
print(shape["num_heads"], shape["kv_width"])  # 16 576
```

The catalog defaults to BF16 architectural estimates for weights, activations and KV storage.
The official checkpoint declares 8-bit floating-point (FP8) quantization; its complete quantized execution behavior is outside these default estimates.
Expert assignments are seeded synthetic inputs and do not reproduce the checkpoint's `noaux_tc` grouped routing policy.
Multi-token prediction is outside the current model.
