# Heterogeneous tensor widths for LP30 decode

An analytical replay of a research pipeline assembled from native operator graphs reaches 3.699683848029% model floating-point operation utilization (MFU) on 423 LP30 chips with 54 resident requests.
Its long closed-loop replay measures 148,973.043666 aggregate tokens per second (TPS) and drains every submitted forward.
The result exceeds the earlier [163-chip periodic record](lpx-cyclic-order-optimality.md) by about 3.98% relative.
The additional inventory uses the simulator's uncalibrated propagation fallback beyond 256 chips.
Heterogeneous tensor-parallel compiler execution, physical static random-access memory (SRAM) banks and matching shared-cache model quality remain unverified.
The objective-wide maximum remains unproven.

## Metric and workload

The comparison uses pure LP30 hardware, counts every allocated transformer and head chip, and imposes no minimum throughput.
The workload is Gemma 4 31B decode at a representative context of 101,000 tokens, with key-value (KV) storage reserved through 102,000 tokens.
Its sixty layers contain fifty local-window layers and ten global-attention layers.
Each consecutive block contains five local layers followed by one global layer.
Weights and KV storage use eight-bit floating point (FP8), with the experimental `value_rope_fp8` global representation and `split_dot` kernel.
Activations and attention compute use bfloat16 (BF16).
Useful work remains 129,261,109,248 floating-point operations per committed decode output.
Each allocated chip contributes `315e15 / 256` FP8 operations per second to the denominator.
Prompt-inclusive complete-request serving retains its [separate metric](lpx-complete-serving-mfu.md).

## Homogeneous screening

An optimistic screen evaluates 1,665 supplied homogeneous tensor-parallel (TP), decode-context-parallel (DCP), ownership and pipeline-stage combinations at their archived maximum resident counts.
The screen independently minimizes local and global layer times over three native kernels.
It gives additional frames and boundary execution zero cost, removes the vocabulary backcharge from transformer memory, and charges each adjacent stage cut only 0.35 microseconds of propagation.
Three complete head chips remain in the allocated inventory.
Seventeen points retain upper bounds above 3.558002959671% MFU.
The largest surviving upper bound is 4.611593861125%.
This finite screen excludes no heterogeneous layouts, other kernels or unrepresented memory policies.
Its archived native timing references use source snapshot `60deeb309b8f50165a6782c73305f6c969f022d730e92848a6e590b370eaff40`.
The eight current segment captures reproduce their corresponding archived per-layer times.
The screen remains conditional on the supplied timing references.

## Native segments and memory

The experiment captures eight native segments with their actual local or global attention geometry.
The three-frame policy reserves two additional copies of the complete native scratch and dense residual, plus three incoming hidden rows.
Every transformer chip retains its 5,000,000-byte safety reserve and the 1,027,072-byte vocabulary backcharge.
Native admission retains sixteen-token KV block rounding and the one-percent KV watermark.
Every reported maximum is admitted, and one additional request is rejected.

| Segment | TP | DCP | Layers | Three-frame capacity | Two-frame capacity | Native body, microseconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local | 8 | 1 | 5 | 35 | 36 | 14.332728 |
| Local | 16 | 1 | 5 | 129 | 129 | 12.883808 |
| Global | 16 | 4 | 1 | 26 | 26 | 7.764021 |
| Local | 8 | 1 | 4 | 58 | 59 | 11.466182 |
| Local | 2 | 1 | 1 | 55 | 57 | 4.604152 |
| Global | 32 | 8 | 1 | 53 | 54 | 7.033211 |
| Local | 4 | 1 | 2 | 57 | 58 | 6.891370 |
| Local | 8 | 1 | 3 | 97 | 97 | 8.599637 |

Fresh two-frame captures reproduce every corresponding three-frame native node body exactly.
Their capacity differences come from the changed memory reservation.
The two-frame global TP32 segment admits 54 requests with 400,494 watermark-adjusted bytes remaining on its limiting chip.
The three-frame version admits 53 with 6,545,518 bytes remaining.
The selected two-frame schedule additionally duplicates the complete native communication workspace on every transformer chip.
This conservative control reserves another 21,504 bytes per chip and leaves 378,990 watermark-adjusted bytes on the limiting global chip.
Fresh native captures preserve every node body and retain the same maximum resident counts.

The head pool retains the complete tied embedding matrix, final normalization, vocabulary projection, softcap and greedy selection.
Embedding and logits use separate frame arenas while sharing all physical worker and vector resources.
The three-frame head ledger charges six complete query workspaces per head chip, and the two-frame ledger charges four.
These memory rows cover concurrent role workspaces and do not increase the compute batch size.

## Composition and physical resources

Native dense-reuse validation requires a complete replicated hidden row on every segment rank after the final reduction.
Handoffs therefore copy one full BF16 hidden row to each destination rank from an owning source replica.
The source assignment balances contiguous destination groups across source replicas.
Every handoff uses the existing fabric's aggregate wire pricing, endpoint bandwidth and transmit, receive and port resources.
The composition retains all native non-host services, predecessor edges, readiness lags and release descendants.
One complete-forward four-microsecond submission replaces per-component submissions.
Additional barriers wait for each complete segment before its outgoing handoff.

Global groups occupy contiguous aligned regions, followed by aligned local groups and the complete head pool.
The physical inventory contains no unused holes.
Translating a native component changes chip and port-group identities together.
An independent oracle compares translated resources, wire service and propagation for every directed intra-group rank pair at two payload sizes and each relevant lane width.
Inter-stage messages are priced on their actual physical endpoints.
This procedure validates the declared synthetic topology and supplies no hardware calibration.

Five patterns are tested with three and four head chips.
Each listed block is repeated ten times in the original layer order.

| Local segment widths and layer counts | Global width | Transformer chips | Total chips | Three-frame residents |
| --- | ---: | ---: | ---: | ---: |
| TP8, five layers | TP16/DCP4 | 240 | 243 or 244 | 26 |
| TP16, five layers | TP16/DCP4 | 320 | 323 or 324 | 26 |
| TP8, four layers; TP2, one layer | TP32/DCP8 | 420 | 423 or 424 | 53 |
| TP4, two layers; TP4, two layers; TP2, one layer | TP32/DCP8 | 420 | 423 or 424 | 53 |
| TP4, two layers; TP8, three layers | TP32/DCP8 | 440 | 443 or 444 | 53 |

## Periodic schedules and closed-loop replay

Circular calendars reserve every positive physical occupancy interval at the output period P.
Each frame-specific arena repeats every three periods for the three-frame policy or every two periods for the two-frame policy.
Arena occupancy extends to its actual release descendant.
The complete forward must finish within N times P for N resident requests.
A first-in, first-out (FIFO) queue and a critical-tail queue each test ten initial periods per composition.
Greedy placement failures concern those orders and do not establish general infeasibility.

The cyclic-order linear program retains one start variable per native node, the period, all dependency lags and all resource successor constraints.
It also imposes complete-forward feedback at the admitted resident count.
Two heuristic orders are optimized for each of the eight non-control compositions under both frame policies.
The two policies contribute four hundred initial calendar trials and thirty-two cyclic-order linear programs.
The strongest three-frame result uses forty transformer stages, three head chips and 53 requests at 3.567032240594% MFU.
Its numerical period is 6.962253697255 microseconds.

An additional search retains physical calendars whose initial latency misses the resident-feedback limit and reoptimizes their cyclic orders with full feedback imposed.
The three-frame search tests twenty calendars and obtains sixteen valid physical-and-arena references for numerical optimization.
It finds no improvement over the three-frame record.
The two-frame search tests twenty-six calendars and obtains twenty valid references.
Its strongest result uses 54 requests at a numerical period of 6.712623139012 microseconds.
A relative pacing margin of `1e-7` gives the independently checked replay period of 6.712623810274 microseconds.
The complete forward spans 358.765853520244 microseconds.
The stronger witness adds readiness delays to 9,904 of its 37,295 nodes while preserving every original service and resource.
These finite cyclic-order searches do not bound alternative schedules.

Independent interval sweeps inspect three neighboring copies of every physical resource and frame arena.
They check all native readiness constraints, exact service durations, release reachability and complete-forward feedback.
The replay changes only readiness delays and frame names.
It rotates concrete frame programs by forward submission and permits one outstanding forward per client.

| Frames | Requests | Warm-up rounds | Measured rounds | Measured outputs | Drained forwards | Aggregate TPS | MFU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 53 | 4 | 16 | 848 | 1,113 | 143,631.637611 | 3.567032240594% |
| 3 | 53 | 8 | 32 | 1,696 | 2,173 | 143,631.637611 | 3.567032240594% |
| 2 | 54 | 4 | 16 | 864 | 1,134 | 148,973.043666 | 3.699683848029% |
| 2 | 54 | 8 | 32 | 1,728 | 2,214 | 148,973.043666 | 3.699683848029% |

Every client completes the same measured output count.
The stronger long replay's measurement halves differ by approximately `-1.98e-13` relative.
The replay drains all submissions and leaves no outstanding jobs.
The stronger two-frame layout's limiting global segment retains 378,990 watermark-adjusted bytes per chip after the additional communication-workspace reservation.
Recorded physical utilization remains within one for every resource.

## Evidence and open questions

The [compressed experiment record](../experiments/lpx/heterogeneous-tp-search.json.gz) retains the homogeneous screen, native component bodies, composition inputs, memory controls, calendar trials, selected cyclic-order constraints, replay witnesses and reproduction sources.
Independent portable checks validate component hashes, memory reservations, placements, hidden-row handoffs, resource intervals, numerical primal/dual constraints and drained per-client traces.
The simulation source identity remains `ae29c716bbf5693a8c9d70df1ee5c6435c3e93af93781bcc5aff57c2e8074986`.
The existing-format decode record and prompt-inclusive serving record remain unchanged.
Different resource orders, batch widths, stage cuts, cache representations and complete-request schedules remain open.
The native-segment composition requires compiler and functional validation before it can establish a deployable heterogeneous-TP serving result.
