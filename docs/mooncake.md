# Mooncake KV Transfer and Storage

TokenSim provides these Mooncake-related simulator connectors:

| Connector | Use case |
| --- | --- |
| `MooncakeConnector` | Direct KV transfer from a prefill worker to a decode worker |
| `MooncakeStoreConnector` | Prefix-addressed shared store with uniform or grouped KV pages |
| `MultiConnector` | Composition of several connectors, such as direct transfer plus a store |

These are simulator models. They do not start or communicate with an external
Mooncake deployment. The other registered connectors are `NoopConnector` (the
default), `P2PConnector` (topology-priced direct transfer) and
`SimpleCPUOffloadConnector` (per-rank CPU offload, see the
[AgentX guide](agentx.md)).

## Selecting a Connector

Pass a connector JSON file with `--kv_transfer_config`:

```bash
./benchmark.py \
  --batching paged-attn \
  --request_count 8 \
  --cluster ./data/clusters/8_a100/p2d5.json \
  --kv_transfer_config ./data/kv_transfer/mooncake_p2p.json \
  --model ./data/models/llama-7b.yaml \
  --qps 10 \
  --verbose none
```

Alternatively, put the same object under `kv_transfer` in a cluster JSON. A
command-line file replaces the cluster's connector. The top-level fields are:

| Field | Default | Meaning |
| --- | --- | --- |
| `kv_connector` | `NoopConnector` | Connector class name |
| `kv_role` | inferred | `kv_producer`, `kv_consumer` or `kv_both`; when omitted, `prefill` workers become `kv_producer`, `decode` workers `kv_consumer` and `hybrid` workers `kv_both` |
| `kv_connector_extra_config` | `{}` | Connector options (configuration reference below) |
| `kv_rank` | worker id | Rank recorded in transfer plans; each worker uses its own id |
| `engine_id` | random UUID | Engine identity, shared by the engine's workers |

Unknown top-level fields and unknown Mooncake connector options are rejected.

## Direct Prefill-to-Decode Transfer

Use `MooncakeConnector` with a disaggregated cluster containing `prefill` and
`decode` worker groups. The connector computes bytes from the actual
transferred KV blocks and models a VRAM-to-VRAM transfer with the protocol
timing described under [Transfer Timing](#transfer-timing). Transfers within
one worker are local; transfers between workers use cross-node timing.

The included `data/kv_transfer/mooncake_p2p.json` selects RDMA. This connector
does not require `hash_ids` because it transfers the live request's blocks
between the selected workers. Without any connector, a disaggregated cluster
moves the blocks over the point-to-point link of the lowest topology level the
two workers share (see [Parallelism](parallelism.md)).

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
  --model ./data/models/llama-7b.yaml \
  --qps 50 \
  --verbose none
```

The standalone example keeps one block in memory and up to 64 on SSD;
`data/kv_transfer/mooncake_store_embedded.json` is an embedded-mode store with
128 memory blocks and no SSD tier. On this single-worker cluster the repeated
prefixes are served by the worker's own GPU prefix cache; the store answers
requests whose blocks are no longer on the GPU, or that run on another worker
(see the MultiConnector example below).

Store keys include the model and compatible payload layout, TP/PP shard
context, `tenant_id`, `cache_prefix`, `group_id`, `pcp_rank`, `dcp_rank`,
`cache_salt`, `reuse_group`, and the chained block hashes. A lookup returns
only the longest contiguous prefix. Local GPU hits are considered before the
store; only additional external blocks are loaded. A full-prompt hit still
computes at least the last prompt token; with `save_policy: mooncake` the last
block is not loaded from the store.

Saving depends on `save_policy`:

- `mooncake` (default): once the prompt has finished computing, save the full
  prompt blocks that are not already in the store; skip the save in a step
  that loaded from the store; never save decode steps.
- `once`: like `mooncake`, but without the existence check, so all full prompt
  blocks are transferred.
- `every_step`: save the computed full prompt blocks on every scheduled step,
  decode steps included.

Grouped storage (`cache_layout: grouped`) supports only `mooncake`; it
publishes newly computed retained pages through its block-manager callbacks.

### Store Sharing

Stores belong to the SimPy simulation environment. Engines connected to the
same `master_server_address` share the store, including engines with
different `engine_id` or DP identity. Embedded engines contribute their
rank-local memory segments once; contributions and staging are summed from
each client's own byte settings. Standalone capacity is shared once, with
additional staging for each client. Admission, save and async policies,
namespaces and staging sizes may differ between clients without creating
separate pools; so may the transfer settings, except for standalone grouped
stores, whose shared network queue they define. The store's mode and layout,
eviction policy, `memory_capacity_blocks`, standalone pool size, replica, SSD
and host-budget settings must agree. Separate environments start with fresh
stores.

Store membership does not imply a cache hit: model/layout, shard,
tenant/prefix and hash identities must match. A uniform store requires one
page-byte format; incompatible formats or settings are rejected rather than
creating hidden independent pools. A uniform object becomes visible only after
its transfer and blocking SSD persistence finish. In-flight writes and reads
protect their objects from eviction until their modeled completion time.
Model geometry and precision are part of both layouts' identities; changing
operator timing tables does not change the cached content identity. Unknown
generated tokens and partial prompts use an opaque request identity, so equal
public request IDs in different engines cannot grant synthetic KV reuse.

See [Prefix Cache](prefix-cache.md) for workload formats and exact hash
matching rules.

## Cache Layouts and Capacity

`cache_layout: uniform` (default) stores one object per KV block of a uniform
cache. Set `cache_layout: grouped` for a model with several cache-page
geometries (for example compressed attention with sliding-window and
compressed caches); the setting is checked against the engine's actual KV
layout. `data/kv_transfer/mooncake_store_dram_agentx.json` is a grouped
example; the [AgentX guide](agentx.md) explains its budget. Grouped storage
currently supports:

- one compute node and PP1,
- a single DRAM replica (`replica_num: 1`, `enable_offload: false`) with byte
  budgets (no `memory_capacity_blocks`),
- `eviction_policy: lru` with `load_async: true` and `transfer_overlap: true`,
- `protocol` `rdma` or `tcp`, no `preferred_segment`,
- a direct connector only (not a `MultiConnector` child).

In embedded mode, `global_segment_size` is each rank's contribution to the
shared KV pool; in standalone mode it is the external owner's total pool. A
uniform store uses `memory_capacity_blocks` instead when it is set. Staging
space is additional: `local_buffer_size` per rank. When `host_available_bytes`
is supplied, the pool plus staging of all clients must fit in it. Integer size
fields are bytes; the `global_segment_size_gb` and `local_buffer_size_gb`
aliases are in GiB.

## Memory and SSD Tiers

With `eviction_policy: "lru"`, memory pressure evicts the least recently used
object that no transfer is using. With the SSD tier enabled, every admitted
object is also written to SSD as a blocking persistence operation, and an
object evicted from memory stays readable from SSD (an object not yet on SSD
is written there on eviction); without the tier, an evicted object is
removed. With `eviction_policy: "none"`, a write that does not fit in memory
is rejected, or goes directly to SSD when the tier is enabled.
`enable_offload: false` disables only the SSD tier; DRAM storage and prefix
reuse remain.

Enable the SSD tier (uniform storage only) with:

```json
{
  "enable_offload": true,
  "offload_tier": "ssd",
  "ssd_capacity_blocks": 64,
  "ssd_read_bandwidth_bytes_per_s": 7516192768,
  "ssd_write_bandwidth_bytes_per_s": 3221225472,
  "ssd_read_latency_us": 100,
  "ssd_write_latency_us": 200
}
```

`ssd_capacity_blocks` takes precedence over `ssd_capacity_bytes`; an enabled
tier needs a positive capacity. An SSD read takes the fixed latency plus bytes
divided by the read bandwidth; writes use the same model and are blocking
persistence operations. SSD eviction is LRU at object (block-key)
granularity.

## Transfer Timing

| Path | Timing |
| --- | --- |
| `MooncakeConnector` | `fixed_latency + bytes / (bandwidth * transfer_utilization * paths)`, where `paths = max(num_nics, parallel_paths)`; cross-node transfers multiply the fixed latency by 1.5 |
| Uniform store, embedded | The device's host link (`interconnect.host_link`, typically PCIe): link latency plus `bytes / (bandwidth * transfer_utilization)` |
| Uniform store, standalone | The host-link time plus a network hop at the protocol timing, its bandwidth capped by the worker's network link |
| Grouped store, embedded | Queued per-worker host-link transfers (separate read and write queues) |
| Grouped store, standalone | Serial host-link and shared network queues; the network queue is capped by the node's network link and shared by all workers |

Unless `fixed_latency_us` / `bandwidth_bytes_per_s` or the per-protocol
`protocol_latency_us` / `protocol_bandwidth_bytes_per_s` override them, the
protocol defaults in `TokenSim/mooncake/transfer_engine.py` are unmeasured
assumptions:

| Protocol | Fixed latency | Bandwidth |
| --- | --- | --- |
| `tcp` | 40 us | 12.5 GiB/s |
| `rdma` | 8 us | 50 GiB/s |
| `nvlink` | 2 us | 300 GiB/s |
| `nvmeof` | 80 us | 7 GiB/s |

Grouped storage models contention on its queues and overlaps transfers with
compute. Uniform and direct paths add blocking transfer latencies and do not
model shared-link contention or overlap. SSD timing is a separate storage
operation and does not change the configured network protocol. Staging
capacity is accounted for but does not impose a separate buffering queue. The
included small uniform and SSD configurations demonstrate caching and
eviction; their block budgets are not host-memory measurements.

## Configuration Reference

Fields of `kv_connector_extra_config`:

| Field | Default | Supported values or behavior |
| --- | --- | --- |
| `cache_layout` | `uniform` | `uniform` or `grouped`, matched against the engine's KV layout |
| `master_server_address` | `default` | Identity of the simulated shared store; no real connection is made |
| `tenant_id`, `cache_prefix` | `default`, `""` | Logical cache namespaces, independent of engine/DP identity |
| `mode` | `embedded` | `embedded` or `standalone-store` |
| `protocol` | `rdma` | `tcp`, `rdma`, `nvlink` or `nvmeof` |
| `global_segment_size` / `_gb` | 8 GiB | Memory-store capacity (see [Cache Layouts and Capacity](#cache-layouts-and-capacity)) |
| `local_buffer_size` / `_gb` | 1 GiB | Per-rank staging memory, additional to the KV pool |
| `memory_capacity_blocks` | unset | Uniform store: explicit capacity in blocks |
| `host_available_bytes` | unset | Upper bound for pool plus staging |
| `admission_policy` | `always` | `always` admits, `never` rejects writes |
| `eviction_policy` | `lru` | `lru` or `none` |
| `save_policy` | `mooncake` | `mooncake`, `once` or `every_step` |
| `replica_num` | 1 | Storage models one replica; other counts are rejected |
| `load_async` | `false` | Marks load and save plans asynchronous; a store releases a request's blocks after its asynchronous save completes. Uniform and direct paths still charge the transfer time to the step |
| `transfer_overlap` | `false` | Requires `load_async`; grouped storage overlaps transfers with compute; uniform storage and `MooncakeConnector` reject it |
| `num_nics`, `parallel_paths` | 1, 1 | Effective path count is their maximum |
| `fixed_latency_us`, `bandwidth_bytes_per_s` | unset | Protocol timing overrides for every protocol |
| `protocol_latency_us`, `protocol_bandwidth_bytes_per_s` | `{}` | Per-protocol timing overrides, keyed by protocol name |
| `transfer_utilization` | 1.0 | Fraction of peak transfer bandwidth, in `(0, 1]`; the bundled grouped example uses 0.75 |
| `preferred_segment` | unset | Segment label stored with admitted objects (uniform only) |
| `group_id`, `pcp_rank`, `dcp_rank` | 0 | Additional pool-key namespaces |
| `enable_offload`, `offload_tier` | `false`, `ssd` | SSD tier switch; `ssd` is the only tier |
| `ssd_capacity_blocks`, `ssd_capacity_bytes` | unset, 0 | SSD capacity; blocks take precedence |
| `ssd_read_bandwidth_bytes_per_s`, `ssd_write_bandwidth_bytes_per_s` | 7 GiB/s, 3 GiB/s | SSD bandwidth |
| `ssd_read_latency_us`, `ssd_write_latency_us` | 100, 200 | SSD fixed latency |

Capacity fields ending in `_bytes` are byte counts; bandwidth fields ending in
`_bytes_per_s` are bytes per second. Conflicting size aliases, fractional byte
budgets and invalid rates are rejected. Parallelism comes from the cluster and
serving configuration, not from connector options.

## MultiConnector

`data/kv_transfer/mooncake_multi_connector.json` combines direct transfer and
an embedded store on a disaggregated cluster:

```bash
./benchmark.py \
  --batching paged-attn \
  --block_size 16 \
  --request_count 4 \
  --cluster ./data/clusters/8_a100/p2d5.json \
  --kv_transfer_config ./data/kv_transfer/mooncake_multi_connector.json \
  --dataset_path ./dataset/mooncake_reuse.json \
  --workload_type json_pairs \
  --model ./data/models/llama-7b.yaml \
  --qps 50 \
  --verbose none
```

Each child contributes transfer plans and metrics; the only option of the
`MultiConnector` itself is its `connectors` list. External prefix matching
uses the largest hit reported by the children, while load and save latencies
are accumulated. Only the child with that largest hit loads the request and
updates its cached-token count. Each transfer plan is routed to its
originating child, including same-type or nested children. Children inherit
the parent's worker role, rank and engine identity unless explicitly
overridden. Grouped storage cannot be a child.

## Metrics

Results include connector-wide transfer counts, blocks, bytes and waits
(`connector_...`), and Mooncake-specific fields with the `mooncake_...`
prefix: store gets, puts, hits and misses, memory/SSD tier hit counts, SSD
reads/writes, transfer and store latency, effective bandwidth,
admission/rejection, eviction, pending asynchronous jobs, local GPU, memory
and disk hit tokens, sampled pool keys, and the effective SSD profile.

Operation counters belong to each connector, even when its store is shared.
Exporting or resetting one engine's profile does not include or clear another
engine's operations. Cache contents and outstanding object lifetimes survive
counter resets.
