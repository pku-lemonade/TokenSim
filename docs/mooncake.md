# Mooncake KV Transfer and Storage

TokenSim provides these Mooncake-related simulator connectors:

| Connector | Use case |
| --- | --- |
| `MooncakeConnector` | Direct KV transfer from a prefill worker to a decode worker |
| `MooncakeStoreConnector` | Prefix-addressed shared store with configurable uniform or grouped KV pages |
| `MultiConnector` | Composition of multiple connectors, such as direct transfer plus a store |

These are simulator models. They do not start or communicate with an external
Mooncake deployment.

`MooncakeStoreConnector` uses `cache_layout: uniform` by default. Set
`cache_layout: grouped` for a model with multiple cache-page geometries; the
setting is checked against the engine's actual KV layout. Grouped storage
currently supports one compute node, PP1, a single DRAM replica, LRU and
asynchronous transfers. In embedded mode, `global_segment_size` is each
rank's contribution to the shared KV pool. In standalone mode it represents
the external owner's total KV pool. Staging space is additional:
`local_buffer_size * world_size`. The sum is checked against
`host_available_bytes` when supplied. Integer size fields are bytes; the
legacy `_gb` aliases use GiB.

Embedded transfers use each GPU's catalog PCIe link. Standalone grouped
transfers add a shared network queue to the PCIe path. `enable_offload`
controls the SSD tier; false preserves GPU-to-DRAM storage and prefix reuse.
The grouped path currently supports DRAM only. See [AgentX](agentx.md) for the
budget derivation and the remaining modeling assumptions.

## Selecting a Connector

Pass a standalone connector file with `--kv_transfer_config`:

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
  --model ./data/models/llama-7b.yaml \
  --qps 50 \
  --verbose none
```

Store keys include the model and compatible payload layout, TP/PP shard context,
`tenant_id`, `cache_prefix`, `group_id`, `pcp_rank`, `dcp_rank`, `cache_salt`,
`reuse_group`, and the chained
block hashes. A lookup returns only the longest contiguous prefix. Local GPU
hits are considered before the store; only additional external blocks are
loaded. Uniform `save_policy: "mooncake"` saves missing full prompt blocks once
the prompt has finished computing, skips saving on a load hit, and does not
save decode steps. `once` likewise waits for a complete prompt; `every_step`
can save each step's computed full prompt blocks. Grouped storage publishes
newly computed retained pages through its block-manager callbacks.

Services belong to the SimPy simulation environment. Engines connected to the
same `master_server_address` share the store, including engines with different
`engine_id` or DP identity. Embedded engines contribute their rank-local memory
segments once; standalone capacity is shared once, with additional staging for
each client. Separate environments start with fresh stores. Store membership
does not imply a cache hit: model/layout, shard, tenant/prefix and hash identities
must match. The uniform simulator currently requires equal page-byte formats
within one store; incompatible formats/settings are rejected rather than
creating hidden independent pools. A uniform object becomes visible only after
its transfer and blocking SSD persistence finish. In-flight writes and reads
protect their objects from eviction until their modeled completion time.
Model geometry and precision are part of both layouts' identities; changing
operator timing tables does not change the cached content identity. Unknown
generated tokens and partial prompts use an opaque request identity, so equal
public request IDs in different engines cannot grant synthetic KV reuse.

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
  "ssd_read_bandwidth_bytes_per_s": 7516192768,
  "ssd_write_bandwidth_bytes_per_s": 3221225472,
  "ssd_read_latency_us": 100,
  "ssd_write_latency_us": 200
}
```

`ssd_capacity_blocks` takes precedence over `ssd_capacity_bytes`. SSD read latency
is fixed latency plus bytes divided by bandwidth. SSD writes use the same model
and are blocking persistence operations, even when connector network overlap is
enabled. The memory tier uses its configured eviction policy; SSD uses LRU at
object/block-key granularity.

## Configuration Reference

Core fields in `kv_connector_extra_config` are:

| Field | Supported values or behavior |
| --- | --- |
| `cache_layout` | `uniform` or `grouped`, matched against the model |
| `master_server_address` | Identity of the simulated shared store; default `default`, no real connection is made |
| `tenant_id`, `cache_prefix` | Logical cache namespaces, independent of engine/DP identity |
| `mode` | `embedded` or `standalone-store` |
| `protocol` | `tcp`, `rdma`, `nvlink`, or `nvmeof` |
| `global_segment_size` / `_gb` | Default memory-store capacity |
| `local_buffer_size` / `_gb` | Per-rank staging memory, additional to the KV pool |
| `memory_capacity_blocks` | Explicit memory capacity override |
| `admission_policy` | `always` admits, `never` rejects writes |
| `eviction_policy` | `lru` or `none` |
| `save_policy` | `mooncake`, `once`, or legacy `every_step` |
| `replica_num` | Replica count recorded on stored objects |
| `load_async` | Marks current load and save transfer plans asynchronous |
| `transfer_overlap` | Queued grouped storage overlaps transfers; uniform/P2P metadata timing rejects this option |
| `num_nics`, `parallel_paths` | Effective path count is their maximum |
| `fixed_latency_us`, `bandwidth_bytes_per_s` | Global protocol timing overrides |
| `protocol_latency_us`, `protocol_bandwidth_bytes_per_s` | Per-protocol timing overrides |
| `transfer_utilization` | Fraction of peak transfer bandwidth, in `(0, 1]`; generic default 1, AgentX example 0.75 |
| `preferred_segment` | Segment label stored with admitted objects |
| `group_id`, `pcp_rank`, `dcp_rank` | Additional pool-key namespaces |

Capacity fields ending in `_bytes` are byte counts; bandwidth fields ending
in `_bytes_per_s` are bytes per second. The SSD example uses 7 and 3 GiB/s.
Conflicting size aliases, fractional byte budgets, invalid rates and unknown
options are rejected. Removed fields `kv_buffer_device`, `kv_parallel_size`
and `staging_buffer_size` had no runtime effect; parallelism comes from the
cluster/serving configuration.

Embedded store transfers use the device's PCIe link. Standalone transfers add
a network hop capped by the cluster's injection bandwidth; SSD timing is a
separate storage operation and does not change the configured network protocol.
Grouped storage queues contention. Uniform/P2P paths aggregate blocking
metadata latencies and do not model shared-link contention or overlap.
Fallback protocol timing in `TokenSim/mooncake/transfer_engine.py` is an
unmeasured analytical assumption; cross-node fixed latency uses a 1.5 factor.
Staging capacity is accounted for but does not impose a separate buffering
queue. The included small uniform/SSD configurations demonstrate caching and
eviction; their block budgets are not host-memory measurements.

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

Operation counters belong to each connector, even when its store is shared.
Exporting or resetting one engine's profile does not include or clear another
engine's operations. Cache contents and outstanding object lifetimes survive
counter resets.
