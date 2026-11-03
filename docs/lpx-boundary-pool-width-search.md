# Boundary-pool width and wider native LP30 pipelines

The current experimental fixed-context decode record reaches 3.827772117% model floating-point operation utilization (MFU) on 445 LP30 chips with 54 resident requests.
Aggregate throughput is 162,146.954 tokens per second (TPS).
The deployment uses 440 transformer chips and five chips for the complete tied embedding and language-model head.
The objective remains pure LP30 hardware, MFU priority and no minimum TPS.
Every allocated chip contributes its peak to the denominator.
The objective-wide maximum remains unproven.

## Workload and selected deployment

The target is Gemma 4 31B decode at a fixed 101,000-token context, with key-value (KV) storage reserved through 102,000 tokens.
Weights and KV use eight-bit floating point, while activations and attention use bfloat16.
The experimental global cache remains `value_rope_fp8` with the `split_dot` kernel.
Useful-work credit remains 129,261,109,248 floating-point operations per committed output token.
Matching model quality remains unverified.
Prompt-inclusive complete-request serving retains its [separate metric and records](lpx-complete-serving-mfu.md).

Each six-layer block contains one local layer on four tensor-parallel (TP) chips, four local layers on eight TP chips and one global layer on 32 TP chips with decode context parallelism (DCP) eight.
Ten repeated blocks use 440 transformer chips and thirty sequential stages.
Local stages execute two independent requests together.
Each global stage executes two separate native batch-one graphs, retaining the dependency gate that limits overlap of their long attention kernels.
The deployment holds 27 two-request cohorts.

Local stages and the boundary pool retain two complete frames that alternate by actual cohort submission.
Global stages retain one fixed frame for each of the two request rows.
Their arena reuse period is one cohort issue interval, while each alternating local or boundary frame repeats after two intervals.
The full native scratch, dense residual, communication and incoming-row reservations remain.
The batch-scaled vocabulary backcharge, 5,000,000-byte safety reserve, sixteen-token block rounding and one-percent KV watermark remain enabled.
Minimum watermark-adjusted headroom remains 378,990 bytes on the limiting global rank.

| Native replay | Chips | Requests | Cohort issue interval | Aggregate TPS | MFU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous four-chip boundary | 444 | 54 | 12.378273 microseconds | 161,573.433 | 3.822823730% |
| Five-chip boundary | 445 | 54 | 12.334490 microseconds | 162,146.954 | 3.827772117% |

The fifth boundary chip reduces the head work enough to improve useful throughput per allocated peak.
The resulting improvement is 0.004948387 percentage points of MFU.
This change preserves the transformer node costs, physical resources and request work.
The four-chip recapture reproduces the previous complete native node body bit for bit.

## Integer boundary-width envelope

The boundary study fixes the 440-chip transformer layout, physical placement, 54 requests, local batch two, global batch one, memory policy and four-microsecond host submission cost.
Removing boundary work and boundary transfers leaves a transformer critical-path lower bound of 310.898267 microseconds.
Including the mandatory host submission gives 314.898267 microseconds and at most 171,483.954 TPS, even with a free boundary.
Charging every additional head chip in the denominator excludes all integer head widths of 32 or more from exceeding the previous 3.822823730% record.
One and two boundary chips cannot store the full tied matrix and required workspaces.

For widths three through 31, the experiment builds the complete embedding, normalization, vocabulary projection and greedy selection graph on the actual chip coordinates.
It includes the configured input and output wire service and propagation.
Widths 22 through 31 have necessary bounds below the previous record.
The remaining widths are three through 21.
The greatest conditional necessary bound is 3.954763710%, at width five.
This bound does not establish that a matching schedule exists.

The study captures 21 complete native graphs, covering every remaining width and controls at widths 24 and 32.
Every capture has the same normalized transformer work graph and reproduces its boundary-envelope critical path.
Nineteen widths receive 72 calendar trials each, using two explicit priority orders and 36 supplied issue intervals.
The trials retain seeds with valid physical calendars even when their original buffer leases or token feedback fail.
The linear programs then enforce those leases and feedback directly.
An infeasible supplied resource order excludes only that order.

| Boundary chips | Total chips | Native-graph MFU bound | Highest supplied numerical MFU |
| ---: | ---: | ---: | ---: |
| 3 | 443 | 3.944598799% | 3.819584270% |
| 4 | 444 | 3.953141116% | 3.822824112% |
| 5 | 445 | 3.954763710% | 3.827772500% |
| 6 | 446 | 3.952915959% | 3.823649860% |
| 7 | 447 | 3.949090340% | 3.812181169% |
| 8 | 448 | 3.944039498% | 3.807307729% |
| 9 | 449 | 3.918636727% | 3.792426338% |
| 10 | 450 | 3.912244032% | 3.787185967% |
| 11 | 451 | 3.905461417% | 3.777648736% |
| 12 | 452 | 3.898395847% | 3.770731216% |
| 13 | 453 | 3.891121040% | 3.763845336% |
| 14 | 454 | 3.883688630% | 3.756272770% |
| 15 | 455 | 3.876137912% | 3.748733764% |
| 16 | 456 | 3.868498642% | 3.741943612% |
| 17 | 457 | 3.860428865% | 3.733755551% |
| 18 | 458 | 3.852965968% | 3.719201531% |
| 19 | 459 | 3.845175407% | 3.711807363% |
| 20 | 460 | 3.836991756% | 3.703738217% |
| 21 | 461 | 3.829437992% | 3.696409941% |
| 24 | 464 | 3.805907733% | Bound control |
| 32 | 472 | 3.743423320% | Bound control |

The head study solves 1,334 supplied fixed-order models across 1,368 calendars.
The numerical linear-program values precede the replay's `1.0000001` timing margin.
Primal and dual checks certify the selected supplied resource order for each retained numerical case.
They do not certify optimality over other orders or layouts.

## Wider global pipelines

The [output-channel fragment catalog](lpx-output-channel-fragments.md) enables native TP64, TP128 and TP256 global attention while retaining complete query/key heads and nonzero output-projection shards.
The completed extension constructs all 88 supplied segment orders in twenty selected multiset classes from its bound survivors.
Eighty-seven compositions use independent batch two and one uses batch four.
The complete graphs include physical placement, full boundary work, entry copies, transfers, native costs and buffer leases.
Seven selected compositions receive 182 calendar trials and 42 numerical fixed-order models.

| Selected composition | Total chips | Requests | Highest supplied numerical MFU |
| --- | ---: | ---: | ---: |
| `b2-wide000-head4` | 884 | 108 | 3.357712369% |
| `b2-wide001-head4` | 884 | 108 | 2.970742194% |
| `b4-wide008-head4` | 804 | 108 | 3.277339363% |
| `b2-wide009-head4` | 964 | 108 | 3.341148437% |
| `b2-wide010-head4` | 964 | 108 | 3.237164671% |
| `b2-wide020-head4` | 804 | 96 | 2.793186106% |
| `b2-wide021-head4` | 804 | 96 | 2.795611944% |

The strongest tested wider schedule uses local segments of two layers at TP8 and three layers at TP16, followed by one global layer at TP64/DCP16, repeated ten times.
It uses 880 transformer chips and four boundary chips, with 108 requests in 54 batch-two cohorts.
The longer native replay reaches 282,552.024 TPS and 3.357712033% MFU.
Its greater aggregate throughput does not offset its greater allocated peak in the MFU denominator.
The tested wider schedules provide no new record.
Other resource orders and untested multiset classes remain open.

Wide collectives require reconstructing the physical port bundles at their actual chip coordinates.
The translation verifies the native participant set, lanes per peer, unit wire cost and propagation before rebuilding the placed resource set.
The complete selected wider graph is independently recaptured with an explicit inclusion ledger for all thirty component invocations in each frame.
Both recaptured node bodies match their originals exactly.

## Native replay and portable validation

The 445-chip parent graph contains 98,087 nodes.
Readiness delays encode the selected cyclic schedule without changing native node service, occupied physical resources, dependencies or release targets.
Independent checks rebuild linear constraints from every original physical resource key, check primal feasibility and dual stationarity, and inspect cyclic resource and arena intervals.
Equivalent resource incidences accelerate calendar construction without merging arena leases or dropping physical ownership.

The short replay uses four warm-up rounds and sixteen measured rounds.
The long replay uses eight warm-up rounds and 32 measured rounds, producing 1,728 measured tokens from 864 paired completions.
It drains all 1,107 submitted cohort jobs, corresponding to 2,214 logical request forwards.
Every request receives 32 measured outputs and respects its previous token completion.
The two halves differ by approximately `7.0e-13` relative throughput.
The scheduled forward duration is 333.031232 microseconds.

The longer 884-chip replay measures 3,456 tokens and drains all 2,214 submitted cohort jobs.
Its two measurement halves differ by approximately `-2.3e-14` relative throughput.
Both native witnesses retain complete drained traces and physical-resource metrics.

The [portable artifact](../experiments/lpx/boundary-width-and-wide-pipeline-search.json.gz) retains all 109 composition records, the integer-width envelope, all 21 head graphs, seven calibrated wider graphs, selected numerical certificates, both replay witnesses, memory ledgers and reproduction sources.
Head graphs use lossless relative-node deltas against the four-chip boundary graph, with complete reconstructed-body hash checks.
The other 81 wider bodies retain capture receipts and require recapture from the recorded native catalog and scripts.
The artifact references the [native fragment catalog](../experiments/lpx/output-channel-fragment-catalog.json.gz) by its file hash.
The production simulation source identity remains `9dfe67e50e9ab76edafd96f37033622e8ae62b2abb3a03f0d233a81920e2407e`.
This checkpoint changes experiment records and documentation without changing production simulation code.

The previous full project suite ran 898 tests with 896 passing and two recorded host-environment failures.
Those failures concern unavailable `nvidia-smi` and macOS resolving `/tmp` to `/private/tmp`.
The current checks focus on the new graph, numerical, memory and replay evidence.
Physical compilation, static random-access memory (SRAM) bank allocation, shared-cache quality and the uncalibrated cross-rack transport fallback remain unresolved.

## Proposed SRAM KV exchange

A separate capacity experiment investigates placing additional global KV in unused SRAM on local-stage and boundary chips.
The 445-chip layout admits 58 local-layer request caches through fresh native admission after charging additional home storage and exchange buffers.
The global-stage plan still admits 54 physical execution slots.
At each global stage, the proposal keeps 53 permanent clients and lets five pageable clients share one execution slot and four home slots.
An opaque byte exchange swaps a resident client and a home client in place, including the former resident's updated KV.
All four home slots use the same physical distribution and distinct addresses.
Home-slot ownership can differ between global stages as requests traverse the pipeline.

A global rank stores 8,263,296 raw bytes per reserved request prefix.
The global cache retains 648 bytes per owned KV-head entry, including scale metadata.
An independently watermarked home cell reserves 8,346,240 bytes per global rank, or 2,670,796,800 bytes over all 320 global chips.
Four home cells require 10,683,187,200 bytes.
An exact 320-byte-vector packing distributes those cells across 125 local and head chips.
Each home chip also reserves 2,048,000 bytes of packet staging and 4,096 bytes of control state.
Fresh local native admission charges the maximum additional allocation of each component uniformly across its ranks.
The smallest home-chip headroom is 31,288 bytes after the stated reserves.

This proposal assigns 1,017,856 bytes of the already charged 1,027,072-byte transformer vocabulary guard to global exchange staging.
The assignment changes the guard's use and preserves its full memory charge.
The global transformer graph contains no vocabulary-head operators.
Both native communication frames, both scratch frames, the separate 5 MB safety reserve and the KV watermark remain charged.
The proposed global staging contains fifteen 64,000-byte packet banks, five 10,752-byte incoming rows and 4,096 bytes of control state.
Separate send and receive banks permit seven paired exchanges and leave one spare packet bank.
The capacity ledger does not assume fifteen simultaneous bidirectional exchanges.

A deterministic byte control exercises 215 pageable requests, including 169 exchanges and 46 hits, with dirty KV appends and partial chunks across three synthetic ranks in one stage.
It checks client ownership, one canonical copy per client and byte equality after every update.
A destructive receive-before-save control detects lost outgoing data.
These checks establish capacity and small functional controls under the proposed policy.
An optimistic bulk wire calculation uses every recorded global-rank/home-chip byte count and the existing physical lane and injection policy.
It charges both directions and yields approximately 6.75 microseconds of wire service for one global stage, or 44.31 microseconds for all ten stages together.
This calculation assumes simultaneously available payloads and unlimited staging, and excludes switch execution module (SXM) work, buffer reuse, control and target-operator competition.
It does not establish attainable prefetch latency or throughput.
The subsequent [finite-bank exchange study](lpx-sram-kv-exchange-search.md) implements native copies, routed transfers, bank leases and closed-loop token feedback.
Its selected controllers do not improve the decode record and leave other ordering policies open.
That study also develops a separate whole-entry placement for computing attention on the chips storing KV.
