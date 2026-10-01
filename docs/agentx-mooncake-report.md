# Mooncake backend status

The unified AgentX runner now accepts `--backend mooncake` and writes a per-run Mooncake configuration. The hybrid scheduler supports synchronous external load/save latency in DP lockstep, and the Mooncake store has a bulk in-memory admission path for long prompt page chains.

The native DSV4 hybrid manager now has an asynchronous shared-page adapter: group slot keys are namespaced into shared PoolKeys, pages are acquired and pinned before import, writes stay invisible until transfer completion, and DP lockstep wakes when `loaded_waiting` becomes runnable. Read and write transfers use per-worker PCIe queues plus a node-shared network queue. Unit and small-engine tests cover cross-rank hits, key isolation, cancellation, shared budget, queue contention, and reset behavior.

The production SimpleCPU full windows and all native operator stages remain valid. A new full Mooncake main-point run is required after this adapter commit; no pre-adapter Mooncake control number is included in the accuracy summary.
