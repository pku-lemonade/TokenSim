# AF execution contracts

Attention–feed-forward network (AF) execution uses its execution-domain graph to synchronize tensor-parallel ranks and contend for shared FFN resources.
The generic `dp_lockstep` coordinator does not execute this graph.
Combining `dp_lockstep=true` with an AF cluster raises `ConfigurationError` before workers start.
Use `dp_lockstep=false` for AF simulations.

## Memory budgets

`gpu_memory_utilization` limits the usable memory of every AF physical worker before model weights, protocol buffers, and KV blocks are allocated.
An explicit per-replica KV capacity caps every attention tensor-parallel shard at the same logical block count.
The override cannot increase the capacity calculated from physical memory.

## F-side routing metadata

The partial-return protocol performs weighted combination on the attention worker.
When routing executes on an FFN worker, every attention consumer receives the selected expert IDs and weights before combining expert outputs, including a single-attention/single-FFN deployment.
Each token transmits one 32-bit expert ID and one 32-bit weight per selected expert.

## Operator ownership

The runtime DAG owns post-attention normalization and the attention-side residual addition.
Its router service includes the routing projection and top-k selection.
Routed expert service excludes normalization, combination, and residual addition.
Combination and residual queries use the hidden interval owned by the attention rank.

## MLA prefill

The MLA adapter accepts equal-length, whole-prompt prefill without cached prefixes.
Scheduled query tokens must cover the entire prompt, and the request must have no previously computed context.
Unsupported partial passes fail before request ownership or iteration state changes.

## Precision

GEMM and expert-table queries use the effective weight dtype.
Normalization, rotary embeddings, residuals, and communication use their activation or transfer dtypes.
Attention queries use the effective KV-cache dtype, including overrides from AF configuration.
