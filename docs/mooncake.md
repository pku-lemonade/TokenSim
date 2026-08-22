# Mooncake KV Transfer and Storage

TokenSim provides three Mooncake-compatible connector configurations:

| Connector | Use case |
| --- | --- |
| `MooncakeConnector` | Direct KV transfer from a prefill worker to a decode worker |
| `MooncakeStoreConnector` | Prefix-addressed shared store with memory and optional SSD offload |
| `MultiConnector` | Composition of multiple connectors, such as direct transfer plus a store |

These are simulator models. They do not start or communicate with an external
Mooncake deployment.

## Selecting a Connector

Pass a standalone connector file with `--kv_transfer_config`:

```bash
./benchmark.py \
  --batching paged-attn \
  --request_count 8 \
  --cluster ./data/clusters/8_a100/p2d5.json \
  --kv_transfer_config ./data/kv_transfer/mooncake_p2p.json \
  --model ./data/psla/llama-7b.json \
  --qps 10 \
  --verbose none
```

Alternatively, put the same object under `kv_transfer` in a cluster JSON. A
command-line file overrides the cluster's embedded connector. Worker roles infer
their KV roles automatically: `prefill` becomes `kv_producer`, `decode` becomes
`kv_consumer`, and `hybrid` becomes `kv_both`.

## Direct Prefill-to-Decode Transfer

Use `MooncakeConnector` with a disaggregated cluster containing `prefill` and
`decode` worker groups. The connector computes bytes from the actual transferred
KV blocks and models a VRAM-to-VRAM transfer. Transfers within one worker are
local; transfers between workers use cross-node timing.

The included `data/kv_transfer/mooncake_p2p.json` selects RDMA. This connector
does not require `hash_ids` because it transfers the live request's blocks
between the selected workers.

## Shared Store and Prefix Reuse

Use `MooncakeStoreConnector` with a workload containing `hash_ids`:

```bash
./benchmark.py \
  --batching paged-attn \
  --block_size 16 \
  --request_count 4 \
  --cluster ./data/clusters/1_a100/h1.json \
  --kv_transfer_config ./data/kv_transfer/mooncake_store_standalone_ssd.json \
  --dataset_path ./dataset/mooncake_reuse.json \
  --workload_type json_pairs \
  --model ./data/psla/llama-7b.json \
  --qps 50 \
  --verbose none
```

Store keys include the model, TP/PP/DP rank context, engine ID, cache group,
`group_id`, `pcp_rank`, `dcp_rank`, `cache_salt`, `reuse_group`, and the chained
block hashes. A lookup returns only the longest contiguous prefix. Local GPU
hits are considered before the store; only additional external blocks are
loaded. The default `save_policy: "mooncake"` saves missing full prompt blocks
once during prefill, skips saving on a load hit, and does not save decode steps.

See [Prefix Cache](prefix-cache.md) for workload formats and exact hash matching
rules.

## Memory and SSD Tiers

The memory tier uses `memory_capacity_blocks` when set; otherwise it derives
capacity from `global_segment_size`. With `eviction_policy: "lru"`, pressure
evicts the least recently used object. If SSD offload is enabled, an evicted
memory object is written to SSD; otherwise it is removed.

Enable the SSD tier with:

```json
{
  "enable_offload": true,
  "offload_tier": "ssd",
  "ssd_capacity_blocks": 64,
  "ssd_read_bw_gbps": 7.0,
  "ssd_write_bw_gbps": 3.0,
  "ssd_read_latency_us": 100,
  "ssd_write_latency_us": 200
}
```

`ssd_capacity_blocks` takes precedence over `ssd_capacity_gb`. SSD read latency
is fixed latency plus bytes divided by bandwidth. SSD writes use the same model
and are blocking persistence operations, even when connector network overlap is
enabled. The memory tier uses its configured eviction policy; SSD uses LRU at
object/block-key granularity.

## Configuration Reference

Core fields in `kv_connector_extra_config` are:

| Field | Supported values or behavior |
| --- | --- |
| `mode` | `embedded` or `standalone-store` |
| `protocol` | `tcp`, `rdma`, `nvlink`, or `nvmeof` |
| `global_segment_size` / `_gb` | Default memory-store capacity |
| `local_buffer_size` / `_gb` | Local buffer size metadata |
| `memory_capacity_blocks` | Explicit memory capacity override |
| `admission_policy` | `always` admits, `never` rejects writes |
| `eviction_policy` | `lru` or `none` |
| `save_policy` | `mooncake`, `once`, or legacy `every_step` |
| `replica_num` | Replica count recorded on stored objects |
| `load_async` | Marks current load and save transfer plans asynchronous |
| `transfer_overlap` | Removes connector blocking wait when combined with `load_async` |
| `num_nics`, `parallel_paths` | Effective path count is their maximum |
| `fixed_latency_us`, `bandwidth_gbps` | Global protocol timing overrides |
| `protocol_latency_us`, `protocol_bandwidth_gbps` | Per-protocol timing overrides |
| `preferred_segment` | Segment label stored with admitted objects |
| `group_id`, `pcp_rank`, `dcp_rank` | Additional pool-key namespaces |

Default protocol timing is defined in
`TokenSim/mooncake/transfer_engine.py`. Cross-node transfers apply a 1.5x fixed
latency multiplier. `local_buffer_size` and `staging_buffer_size` are retained as
configuration metadata; the current model does not simulate separate contention
queues for them. `mode` is likewise a validated deployment label; both supported
values currently use the same in-process simulation service.

## MultiConnector

`data/kv_transfer/mooncake_multi_connector.json` combines direct transfer and a
shared store. Each child contributes transfer plans and metrics. External prefix
matching uses the largest hit reported by the children, while load and save
latencies are accumulated.

## Metrics

Results include connector-wide transfer counts/bytes/waits and Mooncake-specific
store hit/miss counts, memory/SSD hit counts, SSD reads/writes, transfer latency,
effective bandwidth, admission/rejection, eviction, pending jobs, hit tokens,
sampled pool keys, and the effective SSD profile. Fields use the
`mooncake_...` prefix.
