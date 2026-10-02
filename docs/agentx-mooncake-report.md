# Mooncake backend status

The unified AgentX runner now accepts `--backend mooncake` and writes a per-run Mooncake configuration. The hybrid scheduler supports synchronous external load/save latency in DP lockstep, and the Mooncake store has a bulk in-memory admission path for long prompt page chains.

The native DSV4 hybrid manager now has an asynchronous shared-page adapter: group slot keys are namespaced into shared PoolKeys, pages are acquired and pinned before import, writes stay invisible until transfer completion, and DP lockstep wakes when `loaded_waiting` becomes runnable. Read and write transfers use per-worker PCIe queues plus a node-shared network queue. Unit and small-engine tests cover cross-rank hits, key isolation, cancellation, shared budget, queue contention, and reset behavior.

The production SimpleCPU full windows and all native operator stages remain valid. Two real native Mooncake windows also completed with exit code 0 and the full 3,600-second profile:

| 指标 | B200 | B300 |
|---|---:|---:|
| 成功请求 | 3,010 | 5,803 |
| 输出吞吐 | 583.269 token/s | 1,045.469 token/s |
| p90 TTFT | 197.329 s | 101.596 s |
| p90 ITL | 395.629 ms | 420.136 ms |
| GPU / CPU 首次命中率 | 26.667% / 68.290% | 94.980% / 0.760% |
| 共享 DRAM used at final window | 481.5 GB | 752.9 GB |
| connector transfer bytes | 1.314 TB | 328.5 GB |

Validation used B200 `20261002-mooncake-native3` and B300 `20261002-mooncake-native4`; B300 was rerun after the attention-axis indexing optimization. The exported Mooncake counters from the earlier process image did not include the later stats hook, so the validator uses authoritative connector transfer bytes and first-admission CPU tokens as fallbacks while preserving the source manifest and final reference-release checks.
