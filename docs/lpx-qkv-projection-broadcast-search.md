# LP30 QKV projection sharing and nonuniform KV ownership

Progress date: 2026-11-04, Asia/Shanghai.
The selected 56-request analytical decode replay reaches 3.832974589% model floating-point operation utilization (MFU) and 162,367.334 tokens per second (TPS) on 445 LP30 chips.
This is approximately 0.135914% greater throughput than the preceding 54-request, 3.827772117% experimental decode replay with the same chip count.
The new result uses a custom key-value (KV) ownership table that the default AFService does not compile.
It is recorded separately from the existing native-ownership records.
Hardware compilation, physical SRAM banks, fixed physical routes, multi-rack timing and matching model quality remain unverified.
The objective-wide maximum remains unproven.

## Workload and placement

The objective remains pure LP30 hardware, MFU priority and no minimum TPS.
The workload remains Gemma 4 31B decode at 101,000 positions, with KV storage reserved through 102,000 positions.
Eight-bit floating-point weights and KV, bfloat16 attention and activations, the experimental value/rotary shared-cache layout and the existing split-dot attention kernel remain.
Useful credit remains 129,261,109,248 floating-point operations per committed output token.
Communication, control and repeated computations receive no extra useful credit.
All 445 allocated chips remain in the peak denominator.
Prefill and complete-request serving retain their [separate workload and records](lpx-complete-serving-mfu.md).

The transformer placement remains ten repetitions of a one-layer local stage on four chips, a four-layer local stage on eight chips and a one-layer global stage on 32 chips.
Local stages execute paired requests with batch size two, and each global stage executes the two request rows separately with batch size one.
Global tensor parallelism is 32, with eight-way context parallelism (CP) inside each of four KV-head groups.
The tied embedding and language-model head retain their complete five-chip pool.
Fifty-six independent requests form 28 paired cohorts, with one outstanding forward per cohort.

## Weight provenance and capacity

The original eight CP ranks in a KV-head group each store the same query/key/value (QKV) projection matrix.
The experiment keeps that matrix on the group's lowest-index rank and removes the seven duplicates.
Each retained matrix has shape `[4608, 5376]` and occupies 24,772,608 bytes.
Removing 28 matrices in each of ten global layers releases 6,936,330,240 bytes across the transformer.
Output projection weights, normalization parameters, feed-forward network (FFN) weights, auxiliary weights and every original buffer charge remain.

The ledger recompiles the original native physical states and reproduces their original footprint exactly.
An explicit experimental physical-state modification removes only the identified matrix bytes and passes native plan consistency validation and footprint reconstruction.
The unmodified local stages separately pass fresh native 56-request admission checks.
The global owner map uses an independent per-rank byte ledger because the default allocator assumes uniform CP ownership.
This evidence does not establish default AFService admission for the custom 56-request layout.

Each request reserves 6,375 whole sixteen-position blocks per KV-head group.
Every rank and request receives its own one-percent watermark allowance.
The selected policy assigns 769 reserved blocks to the projection root and 800 or 801 blocks to each other rank.
It places the 101,000 visible positions across those ranges before assigning the remaining reserved positions.
The largest visible partition contains 12,672 positions.
The minimum watermark-adjusted global headroom is 551,358 bytes per chip.
The preceding [capacity-only screen](../experiments/lpx/qkv-projection-broadcast-capacity.json) rejects 57 and 58 requests under the same declared rank/block policy.
This rejection does not exclude other owner maps, precisions or placements.

## Broadcast, scratch and numerical controls

Each projection root retains its original native normalization, QKV projection and rotary prefix.
The other seven ranks receive its normalized query shard and newest-position KV data.
Each peer receives 8,192 query bytes, 1,280 KV bytes and a sixteen-byte request/position/head descriptor.
Only the mapped owner appends the new persistent KV entry.
All 32 attention cores use their actual visible partition lengths and the original analytical kernel rates, including the conservative append and query-folding charges.
The original CP result combination, output projections, FFN operations and their communication remain.

Source packing and receiver copies have explicit native costs.
The source copies 66,544 bytes and each receiver copies 9,616 bytes, including the associated local control data.
All copies reserve the native worker and vector execution module resources.
The broadcast uses the configured routed injection and port resources, followed by the configured propagation delay.
The selected schedule uses twelve lanes per peer, requiring 84 outgoing lanes for the seven-peer fanout.
Six- and thirteen-lane variants are retained as finite comparisons.

Two data-message frames and 4,096 control bytes occupy 155,648 bytes inside the already charged 1,027,072-byte guard.
Message tags and local descriptors use that control region; they are included in the charged copy lengths and do not require additional data-frame allocations.
The independent control audit also allows a conservative extra 768 descriptor bytes and remains within 4,096 bytes.
Both complete native active frames, the native input frames and the separate 5 MB safety reserve remain charged.
Every new positive-service operation starts after its original active-frame acquisition and precedes its invocation completion.
The row-zero gate is rewired to wait for all 32 new attention-core completions, preserving the original conservative constraint on row one.

Four deterministic full-shape synthetic projection fixtures verify bit-identical outputs from the eight duplicate producers and the broadcast recipients.
Thirty-six cached-attention controls cover three owner policies, four KV heads and three context boundaries, including 101,000 positions.
The controls check compact owner tables, all request/rank address bounds, unique persistent writes, unchanged non-owner partitions and partitioned log-sum-exp merging.
The maximum absolute error against the independent full-context floating-point reference is approximately `1.05985e-7`, below the `1e-5` tolerance.
These checks validate the experimental representation and communication algebra using synthetic weights.
They do not establish accuracy equivalence to the trained model or an LP30 compiler implementation.

## Finite schedule search and attained replays

The isolated component screen compares three owner policies at three lane widths.
The selected owner map has an isolated twelve-lane component duration of approximately 11.566036 microseconds, including the standalone component's four-microsecond host charge.
Complete pipelines retain one original host charge per paired forward and suppress the component-local host charges as in the parent pipeline.
Four complete pipeline captures retain both request rows across all ten global stages.
Each unanchored capture has 100,047 native nodes, comprising 98,087 parent nodes and 1,960 added nodes.
Exactly 730 original nodes change: 720 component nodes and ten external attention-core gates.
Every other parent node remains identical.

The search evaluates 80 specified fixed-order linear programs across lifted parent orders and fresh periodic calendars.
Fifty models have successful primal and dual certificates, and thirty are reported infeasible by the solver without retained Farkas certificates.
The unsuccessful models do not exclude other resource orders.
The selected twelve-lane order has a numerical period of 12.3177473265 microseconds.
Adding the retained numerical margin gives a native issue period of 12.3177485583 microseconds.
Independent audits verify dependencies, all declared physical resources, 82 arena leases, neighboring periodic copies and cohort token feedback.

| Complete native replay | Requests | MFU | Aggregate TPS |
| --- | ---: | ---: | ---: |
| Six-lane broadcast with inherited timing anchors, short screen | 56 | 3.790199504% | 160,555.354 |
| Twelve-lane broadcast without timing anchors, short screen | 56 | 3.796179649% | 160,808.677 |
| Selected periodic schedule, four warmup and sixteen measured rounds | 56 | 3.832974589% | 162,367.334 |
| Selected periodic schedule, eight warmup and 32 measured rounds | 56 | 3.832974589% | 162,367.334 |

The long replay measures 1,792 committed tokens and 896 paired completions.
It drains all 1,148 submitted paired jobs, corresponding to 2,296 logical forwards.
The measured cohort cycle is approximately 344.896960 microseconds.
Every request's later forward waits for its preceding token completion.
Every declared physical-resource utilization remains at or below one.
The relative throughput difference between the measurement's two halves is approximately `6.4e-12`.
The thirteen-lane fixed-order target reaches the same period as the selected twelve-lane target, with no additional attained-performance claim.

## Evidence and remaining work

The [portable artifact](../experiments/lpx/qkv-projection-broadcast-search.json.gz) includes the fresh weight ledger, owner maps, numerical controls, isolated components, complete pipeline bodies, every tested order's lossless seed vector, successful primal/dual certificates and drained replay traces.
It also retains the independent validation and reproduction sources.
The production simulator source is unchanged at SHA-256 `9dfe67e50e9ab76edafd96f37033622e8ae62b2abb3a03f0d233a81920e2407e`.
The existing native-ownership, homogeneous, existing-format and complete-serving record dictionaries remain unchanged.
The custom-ownership result has a separate record and explicit compilation limitations.

The primary [NVIDIA LPX technical article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) describes 96 point-to-point chip links, compiler-scheduled transfers and overlap at 320-byte vector granularity.
Applying those mechanisms to this projection-sharing design is an inference.
The article does not provide the fixed neighbor map or establish that this schedule can be compiled.
The current multi-rack propagation assumption remains uncalibrated.

Other resource orders, guarded early projection preparation, custom-owner compiler support, SRAM bank allocation, different chip inventories and model-quality validation remain open.
The experiment provides a larger analytical decode result within its stated custom-ownership assumptions.
It does not certify the maximum MFU for the user's full serving objective.

The subsequent [boundary-pool projection study](lpx-qkv-boundary-projection-pool-search.md) admits 57 requests under an explicit matrix-relocation policy.
Its longer 448-chip decode replay reaches approximately 2.95940% MFU, and the tested schedules leave this page's 3.83297% record unchanged.
