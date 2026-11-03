# LP30 SRAM KV exchange and attention on storage chips

Progress date: 2026-11-04, Asia/Shanghai.
The selected finite-bank key-value (KV) exchange controllers do not improve the current 3.827772117% experimental decode model floating-point operation utilization (MFU) record.
Their complete native-simulator graphs expose a buffer deadlock and substantial cache-slot waiting that capacity and bulk bandwidth calculations omit.
A separate preliminary design keeps four additional KV caches on existing local and head chips and computes their attention partitions there.
Its memory and operator screen passes, while numerical and integrated timing validation remain open.
The objective remains pure LP30 hardware, MFU priority and no minimum tokens per second (TPS).
The objective-wide maximum remains unproven.

## Workload and physical accounting

The workload remains Gemma 4 31B decode at a representative 101,000-token context, with storage reserved through 102,000 tokens.
The deployment retains 440 transformer chips and five complete embedding and language-model head chips.
Every allocated LP30 contributes to the peak denominator, including the 125 local and head chips used for additional KV storage.
Weights and KV remain eight-bit floating point, and activations and attention remain bfloat16.
The experimental cache layout remains `value_rope_fp8` with the `split_dot` kernel.
Useful credit remains 129,261,109,248 floating-point operations per committed output token.
Additional copies, controller operations, repeated projections and partial-result processing receive no additional useful credit.

All reported times use the existing analytical LP30 operator rates and routed lane, injection and propagation model.
The simulations preserve native worker and vector execution module (VXM) contention for data reorders.
They do not assume a separate background copy engine.
Physical compiler execution, SRAM banks, matching shared-cache model quality and multi-rack timing remain unverified.
Complete-request serving retains its [separate workload and records](lpx-complete-serving-mfu.md).

## Finite exchange buffers

The [earlier capacity study](lpx-boundary-pool-width-search.md#proposed-sram-kv-exchange) holds 58 local request caches, 54 global physical slots and four additional complete global KV cells distributed over the other 125 chips.
At each global stage, 53 clients remain permanent and five pageable clients share one execution cell and four home cells.
A full independently watermarked global-rank cell occupies 8,346,240 bytes, including scales and unused reserved capacity.
The exchange conservatively transfers the complete cell in both directions.
An updated execution cell moves to a home cell while the incoming home cell becomes the execution cell.

The initial packet program uses at most eight home peers and 8,000 bytes per peer in one packet.
Seven separate send/receive pairs occupy 896,000 bytes within the previously charged 960,000-byte global packet budget.
Each home chip reserves four paired batches, each containing 32 chunks per direction, for a total of 2,048,000 bytes.
The native communication frames, scratch frames, vocabulary guard charge, 5 MB safety reserve and one-percent KV watermark remain.
These packet buffers are logical allocations; physical SRAM bank placement remains unverified.

Every exchange explicitly executes source copies, routed bidirectional transmission, propagation and destination stores.
All copies use the existing analytical `data_reorder` operator and its worker/VXM reservations.
Each participant also receives a 128-byte descriptor copy and an explicit 64-operation VXM controller budget.
A routed 16-byte descriptor broadcast distributes the exchange choice.
Bank leases retain outgoing data and incoming storage until their corresponding consumers finish.

| Isolated program | Native nodes | Simulated completion |
| --- | ---: | ---: |
| Original packet plan, global stage zero | 12,802 | 40.030570 microseconds |
| Original packet plan, global stage eight | 12,802 | 39.630410 microseconds |
| Original plan, all ten stages sharing homes | 128,011 | 139.606115 microseconds |

These isolated completions include one four-microsecond host submission.
They include finite staging and copies that the earlier approximately 6.75-microsecond per-stage bulk wire estimate omits.
The all-stage program includes simultaneous demand on shared home buffers and physical resources.

## Coalesced copies within the same budget

A further implementation combines one, two or four waves into each home-side copy.
Global send/receive banks and home send/receive batches have independent release targets.
A global pair can be reused after its own received data is stored, while the home batch remains protected until all of its received pieces are stored.
This removes an unnecessary dependency between the two destination stores.
The global packet pool remains 960,000 bytes and the home pool remains 2,048,000 bytes.

| Peers per packet | Home waves per copy | Packet groups | Stage-zero completion |
| ---: | ---: | ---: | ---: |
| 8 | 1 | 150 | 37.665478 microseconds |
| 8 | 2 | 151 | 37.983236 microseconds |
| 8 | 4 | 153 | 38.548991 microseconds |
| 16 | 1 | 77 | 44.153346 microseconds |
| 16 | 2 | 77 | 44.023024 microseconds |
| 16 | 4 | 77 | 44.269346 microseconds |
| 10 | 4 | 127 | 37.391933 microseconds |
| 12 | 4 | 106 | 37.706784 microseconds |
| 15 | 4 | 81 | 36.114933 microseconds |

Fifteen peers use four 120,000-byte send/receive pairs, exactly filling the global packet pool.
Four home waves use one 1,024,000-byte buffer per direction.
The design preserves full vector coverage, opposite-direction storage and all native copy costs.
The isolated measurements do not establish pipeline throughput.

## Closed-loop feedback and buffer ordering

The integrated programs retain every native target operator's duration, occupancy, physical resource set and dependency.
Readiness anchors express supplied native schedules without adding useful work.
The pageable client occupies row one of a two-request batch.
This preserves the existing row-zero attention gate and early row-zero handoff.

A per-cohort feedback lease protects all useful forward work until the previous token completes.
With depth two, a bounded future KV preparation job can exist before that completion, while its token forward remains blocked by the feedback lease.
Depth one permits one outstanding job per cohort.
The driver counts committed outputs, preserves request identities, reports measurement drift and drains all submitted jobs.
No additional logical request or useful-work credit is assigned to the preparation job.

The unpaired readiness variant deadlocks after one completion.
The native scheduler snapshot contains the following waiting cycle:

```text
pageable cohort A waits for global row-one active frame
common cohort B holds that frame and waits for next local input
pageable cohort A holds that local input while waiting for its row one
```

A read-only native-engine inspection extension records the cycle without changing scheduler code.
The retained witness checks both occupied resource identities and the release-node dependency paths.
A small two-cohort control reproduces the same deadlock and removes it when both rows wait for KV readiness.
The revised integrated variant gates both active rows on KV readiness and uses the five already charged incoming rows separately for the five pageable clients.

| Supplied controller | Depth | Measured outputs | Drained cohort jobs | Aggregate TPS | Screen MFU | Relative half-window drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed clock, 25-microsecond advance, 45-microsecond lead | 2 | 232 | 208 | 48,561.213 | 1.146375268% | 0.001306 |
| Upstream-triggered rolling clock, zero advance, 45-microsecond lead | 1 | 232 | 203 | 51,887.928 | 1.224908388% | -0.002973 |

Both screens discard two warm-up rounds and measure four rounds.
Each of the 29 cohorts contributes four measured completions.
The short windows remain finite screens and are not promoted to the decode record.
The fixed-clock run occupies every pageable execution cell throughout the measurement, while the greatest physical VXM utilization is approximately 23%.
The observed controller is limited by cache-cell reservations and progress dependencies.
This observation does not exclude other prefetch schedules, buffer policies or remote-attention placements.

## Controls and ideal schedule targets

The new readiness-anchor construction reproduces the existing 54-request native replay at approximately 162,146.954 TPS.
For a hypothetical fully resident 58-request body, the canonical driver and the new bounded driver both reproduce approximately 164,239.867 TPS at the same supplied schedule.
The latter control omits KV I/O and has no eligible 58-slot global memory admission.
It establishes the timing transformation and metric convention for that body.

Seventy supplied physical resource orders are re-solved with 29-cohort token feedback.
The greatest numerical target is 3.915951300% before I/O costs and the replay timing margin.
The selected integrated screens stretch that target's period by 1.01.
The numerical target remains ineligible as an attained serving result.
The artifact retains its selected primal/dual certificate and the original physical constraint reference.

A separate driver control proves that two queued jobs on independent frame resources still obey token feedback when the lease is present.
Removing that lease produces an invalid earlier next-token completion, which the negative control detects.
The packet byte control executes 47 requests with 38 exchanges, nine hits and 47 dirty appends under randomized valid dependency orders.
It uses the complete recorded vector intervals, global/home addresses and finite bank reuse.
Two independent bytes per opaque vector validate the copy permutation, while complete interval coverage establishes its extension to every byte of each 320-byte vector.
Bytes belonging to other global stages remain unchanged.
A hit writes each endpoint's own outgoing bank back while charging the same transfer and copy costs.
An incorrect incoming-bank store on a hit fails the reference-state check.

## Attention computed on home chips

The primary [Infinite-LLM paper, Section 4](https://arxiv.org/html/2401.02669v2#S4), describes sequence-partitioned attention computed with local KV, followed by aggregation of partial results.
Applying that algebra to the spare SRAM and compute capacity of the existing LP30 deployment is a new design inference here.
The paper's GPU throughput measurements do not calibrate LP30 execution.

This preliminary placement keeps 54 permanent global clients and four complete home-resident clients.
It removes the shared execution-cell requirement for those four clients.
Each storage block contains sixteen positions of all four KV heads, including their scales, for 41,472 bytes.
Each global layer has 6,375 reserved blocks at 102,000 tokens.
The placement distributes all ten layers over the same 125 local and head chips, with a separate one-percent watermark for each home client on each chip.
The four caches occupy 10,677,381,120 bytes after the stated block and watermark allocation.
Fresh native local admission confirms all 58 local request caches after the additional allocations are charged.
The smallest full home-chip headroom is 14,904 bytes with the existing safety reserve retained.

The home operator screen serializes four single-KV-head attention kernels, each with eight query heads and a 128-key tile.
It preserves the baseline native unit rates, kernel costs and shared-cache conversion charges.
The target's existing operators retain their 256-key policy.
The largest four-kernel sum is approximately 2.344423 microseconds using the reserved positions conservatively.

| Home allocation within the previous 2,048,000-byte budget | Bytes |
| --- | ---: |
| Conservative native scratch for one KV head | 1,092,160 |
| Ten global layers of native auxiliary-weight allowance | 440,340 |
| Complete query and receive storage | 65,536 |
| Partial output and send storage | 131,328 |
| New-KV message allowance | 5,120 |
| Additional controller allowance | 4,096 |
| Total | 1,738,580 |

One full query contains 32,768 bytes per home chip.
Each home returns 65,664 bytes of full-precision partial output and log-sum-exp statistics.
Each global output rank would receive 256,500 bytes from 125 homes.
Query routing, newly appended KV ownership, complete numerical attention and output projection, return-buffer lifetimes and integrated contention remain unverified.
No TPS or MFU is assigned to this preliminary design.

## Evidence and remaining work

The [portable experiment artifact](../experiments/lpx/sram-kv-exchange-and-home-attention-screen.json.gz) retains the packet graphs, integrated graphs, finite traces, failed-controller cycle, byte and feedback controls, ideal-order receipts, home-attention ledger and reproduction sources.
It references the [previous boundary-width artifact](../experiments/lpx/boundary-width-and-wide-pipeline-search.json.gz) by file hash.
Native bodies use checked hashes and lossless relative-node deltas where those reduce storage.
The production simulation source identity remains `9dfe67e50e9ab76edafd96f37033622e8ae62b2abb3a03f0d233a81920e2407e`.
Production simulation code and the displayed decode and complete-request records remain unchanged.

The next attention-on-home experiment must preserve target Q/K/V projections, normalization and output shards while replacing only the distributed core and its partial-result merge.
It must validate newly appended KV placement and equivalent attention values before assigning serving performance.
Other request counts, head widths, block placements, cache controllers, resource orders and inventories remain open.
The subsequent [native attention-on-home study](lpx-home-attention-native-search.md) completes numerical controls, routed native components and selected 56/58-request pipeline measurements.
Its supplied whole-buffer and grouped schedules remain below the attained decode record.
