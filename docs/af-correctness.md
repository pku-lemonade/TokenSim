# AF execution contracts

Attention–feed-forward network (AF) execution uses its execution-domain graph to synchronize tensor-parallel ranks and contend for shared FFN resources.
The generic `dp_lockstep` coordinator does not execute this graph.
Combining `dp_lockstep=true` with an AF cluster raises `ConfigurationError` before workers start.
Use `dp_lockstep=false` for AF simulations.

## Memory budgets

`gpu_memory_utilization` limits the usable memory of every AF physical worker before model weights, protocol buffers, and KV blocks are allocated.
An explicit per-replica KV capacity caps every attention tensor-parallel shard at the same logical block count.
The override cannot increase the capacity calculated from physical memory.
`kv_cache_capacity_tokens_total` is divided by the attention execution domain's DP replica count, independently of the generic serving parallel configuration.

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

## Continuous decode routing

Requests in one AF block may have different generation positions.
Routing snapshots retain each request's decode step and absolute token identity.
External assignments are selected and validated against that request's step.

## Transfer precision conversion

Changing wire precision creates explicit encode and decode nodes on the producing and consuming compute resources.
The elementwise operator name records both dtypes, such as `cast_bf16_to_fp8`, and the table dtype identifies the output format.
The analytical fallback accounts for reads in the source format, writes in the target format, element conversion, and kernel submission overhead.
Strict table lookup requires a matching conversion measurement.

## Prefill attention contexts

An attention query records the query length and available KV context of each request separately.
Cached keys contribute attention FLOPs and KV reads even when only a short suffix is computed.
Measured whole-prompt rows used for cached prefill are scaled by the analytical growth ratio and reported as extrapolated estimates.
Uneven query lengths retain separate shapes instead of being replaced by a mean.

## Model layer structure

Every compiled runtime uses its model's dense and MoE layer metadata for prefill and decode, including automatically generated execution domains.
Dense FFNs, routed experts, and shared experts are separate services with their declared dependencies and weight ownership.
AF submission requires a compiled runtime plan; all layers use its model-specific graph.

## Model boundaries

Serving engines use whole-model AF execution for MHA, GQA, and MLA architectures.
Embedding lookup, required tensor-parallel boundary collectives, final normalization, the language-model head, and token selection contribute to request latency.
An unfinished prefill chunk performs no language-model head, logits gather, or sampling work.
The chunk that completes a prompt produces one logits row per completed request.

## Mixed serving and recomputation

A serving batch is partitioned by its logical prefill/decode phase and whether it builds context.
Selected AF phases execute their graphs concurrently on the shared physical resources, with one serving-step completion barrier.
Native eager control retains its single-in-flight execution rule.
Recomputation retains decode ownership while using prefill attention kernels over the restored context.
MLA recomputation requires a whole, equal-length context, matching its unchunked prefill restriction.
Failure or cancellation of one group interrupts the remaining groups and releases their resources.
