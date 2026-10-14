# AF execution contracts

Attention–feed-forward network (AF) execution uses its execution-domain graph to synchronize tensor-parallel ranks and contend for shared FFN resources.
The generic `dp_lockstep` coordinator does not execute this graph.
Combining `dp_lockstep=true` with an AF cluster raises `ConfigurationError` before workers start.
Use `dp_lockstep=false` for AF simulations.

## Memory budgets

`gpu_memory_utilization` limits the usable memory of every AF physical worker before model weights, protocol buffers, and KV blocks are allocated.
An explicit per-replica KV capacity caps every attention tensor-parallel shard at the same logical block count.
The override cannot increase the capacity calculated from physical memory.
