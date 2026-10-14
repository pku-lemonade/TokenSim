# AF communication

Physical AF communication uses the `node-domain-fifo-v1` model.
Each node has eight GPU slots, one shared local communication resource, and independent inter-node transmit and receive resources.
Empty slots remain part of the declared inventory.

```json
{
  "transfer_topology_mode": "physical",
  "communication_domains": {
    "node_size": 8,
    "node_count": 2,
    "placement_strategy": "manual",
    "device_indices": {
      "a-pool:0": 0,
      "f-pool:0": 8
    },
    "local_bandwidth_bytes_per_s": 300000000000,
    "inter_tx_bandwidth_bytes_per_s": 50000000000,
    "inter_rx_bandwidth_bytes_per_s": 50000000000,
    "local_latency_us": 1,
    "inter_latency_us": 5
  }
}
```

Place these fields inside the cluster's `af` object.
Bandwidth is unidirectional bytes/s, with decimal GB units.
The defaults above are simulator assumptions: 300 GB/s is a shared node budget.
The physical mode reads these parameters independently of the catalog link name.
`manual` supports explicit device indices; `dedicated` separates A/F nodes; `mixed` reserves the configured A slots within each node.
Placement rejects duplicate slots, insufficient inventory and conflicting overrides before execution.

A local transfer holds its node's local resource.
A remote transfer atomically acquires the source transmit and target receive resources and uses their lower bandwidth.
Requests sharing either resource queue in FIFO order; disjoint resources can run concurrently.
Waiting transfers do not hold a subset of their resources.
Cancellation releases reservations and queued requests.

## Collective volumes

Collective input `M` is the logical input bytes per rank, and `N` is the actual participant count.

| Operation | Logical output per rank | Ring bytes sent per rank |
| --- | --- | --- |
| AllGather | `N*M` | `(N-1)*M` |
| ReduceScatter | `M/N` | `(N-1)*M/N` |
| AllReduce | `M` | `2*(N-1)*M/N` |

Reduction shards must contain whole dtype elements.
Hierarchical collectives require equally sized participating subgroups on each node.
AllReduce executes local ReduceScatter, inter-node shard AllReduce and local AllGather.
AllGather and ReduceScatter add local layout conversion when their input/output ordering requires it.
These conversions occupy compute resources and send no network bytes.
Each subgroup waits for its own producers; consumers wait for all required collective phases.

All-to-All may use declared locality fractions and load imbalance in physical mode.
Expected output bytes retain fractional statistical traffic, and total traffic is conserved.
Flat All-to-All supports the uniform distribution with default imbalance and no destination redistribution.
Unsupported statistical settings fail before no-op handling or cache lookup.

## Service measurements and traces

Analytical phase service is fixed latency plus wire bytes divided by effective bandwidth.
Measured phase records require `timing_scope=transport_service` and matching model version, algorithm, operation, dtype, participants, node layout, wire schedule and domain parameters.
Hardware/backend applicability and transport backend also participate in matching.
A whole-collective measurement cannot be split into invented phase measurements.
`table_only` rejects missing phase measurements; policies allowing fallback label analytical service explicitly.

Caches retain static traffic plans and service estimates.
Every invocation creates its own readiness events, resource reservations and queueing delays.
Trace records keep dependency waiting, resource queueing and service duration separate.
The node FIFO and coarse phase resource reservations are modeling approximations.
