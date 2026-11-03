# Periodic resource calendars for pure-LP30 decode

A verified closed-loop periodic replay reaches 55,207.215756 aggregate tokens per second (TPS), corresponding to 3.557996616% model floating-point operation utilization (MFU) on 163 LP30 chips.
The result improves the earlier 3.419600500% experimental shared-cache decode reference by 4.04714% relative.
The later [cyclic-order study](lpx-cyclic-order-optimality.md) closes the small remaining timing gap and derives an exact conditional ceiling for this resource order.
The schedule preserves every native operator dependency, compute service time, communication cost, physical resource binding and buffer release index.
It adds readiness delays so independent requests occupy compatible phases of a repeating resource calendar.
The useful model work and allocated hardware denominator remain unchanged.
Shared-cache model quality, static random-access memory (SRAM) bank placement and physical compiler execution remain unverified.
The global maximum remains open.

## Workload and memory

The workload uses Gemma 4 31B text decode at 101,000 tokens, with key-value (KV) capacity reserved through 102,000 tokens.
Weights and the experimental shared global cache use eight-bit floating point (FP8), while activations use bfloat16 (BF16).
The transformer uses tensor parallelism (TP) 16, decode context parallelism (DCP) 4 and pipeline parallelism (PP) 10, with six layers per stage.
Its 160 chips are separate from the three-chip tied embedding and language-model head pool.
Every allocated chip contributes to MFU.
The hardware constraint requires pure LP30 and imposes no throughput floor.
Prompt work and request replacement remain outside these resident-prefix decode measurements.
The [complete-request serving study](lpx-complete-serving-mfu.md) retains its separate prompt-plus-output metric.

The verified periodic programs use fourteen resident requests and two active/input slots per stage.
The native transformer memory policy charges an additional 2,190,464 bytes per chip, covering a complete extra native scratch allocation and dense residual.
This charge exceeds the [persistent-frame allocation](lpx-frame-workspace-search.md) for two active frames and two incoming rows by 1,436,800 bytes.
The complete original scratch allocation, 5,000,000-byte safety reserve, 1,027,072-byte vocabulary backcharge, native KV block rounding and one-percent watermark remain enabled.
The resulting transformer reserve field is 8,217,536 bytes.
The minimum headroom after watermark inversion is 16,861,300 bytes per chip.
The head pool retains the complete target matrix, logits, distributed greedy selection, communication buffers and safety reserve.
Its exclusive boundary arena prevents overlapping use of those shared tensors.
The candidate therefore does not need the vocabulary-charge removal used by the [fifteen-request protocol projection](lpx-decode-protocol-buffer-search.md).

## Periodic construction

NVIDIA describes compiler scheduling of individual operations, data movement and SRAM conflicts in its [deterministic-execution article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-deterministic-execution-drives-power-efficient-high-interactivity-inference-on-nvidia-vera-rubin/).
That description motivates this scheduling experiment and does not validate its physical bank allocation or analytical unit rates.

The construction places a complete native forward on a circular calendar of period P.
Each ordinary exclusive resource reserves its original occupancy interval modulo P.
The scheduler visits nodes in a dependency order and delays a node until its required resources have a common free interval.
A node that needs more contiguous time than any available interval causes that trial to fail.
Failure identifies a limitation of the supplied list-scheduling trial and does not prove that every schedule at that period is infeasible.

The input and active arenas alternate between two concrete profiles, so each frame slot repeats every 2P.
Their leases extend through the original release descendants.
The shared head arena repeats every P and includes both embedding and output-head leases.
All declared data dependencies remain intact.
N denotes the number of resident requests.
The final forward completion must fit within N times P to allow the same client to submit its subsequent token after feedback.

Two ordering policies and eleven periods produce 22 coarse trials.
A twelve-period grid and nine local bisection trials refine the promising range, giving 43 calendar trials overall.
The bisection is a local search heuristic; schedule feasibility is not assumed monotone across arbitrary periods or ordering choices.
The best tested calendar uses P equal to 18.11357421875 microseconds.
Its complete forward spans 237.917502487 microseconds, which fits the fourteen-client feedback cycle of 253.590039063 microseconds.
The construction adds delays to 1,160 of the 96,774 nodes and shortens no service time.

## Independent verification and replay

An independent verifier expands each occupied interval into three neighboring periodic copies and checks them with a sorted interval sweep.
It also checks every dependency, service duration, arena hold and token-feedback limit.
The verifier checks 5,778 physical resource keys and 21 arena definitions in the template.
Alternating profiles instantiate twenty active and twenty incoming arenas plus the shared head arena.
The greatest interval overlap attributable to floating-point roundoff is below `1e-18` seconds.
The delayed profiles are also reconstructed and verified with the repository replay engine.

The selected short replay uses four warm-up and sixteen measured outputs per client.
The long replay uses eight warm-up and 32 measured outputs per client.
Both use the original closed-loop cohort driver with the selected period as initial spacing and global issue interval.
The long control drains all remaining forwards and records resource occupancy over the measurement window.

| Period, microseconds | Warm-up outputs | Measured outputs | Aggregate TPS | MFU |
| ---: | ---: | ---: | ---: | ---: |
| 18.40000000000 | 56 | 224 | 54,347.826087 | 3.502610640% |
| 18.40000000000 | 112 | 448 | 54,347.826087 | 3.502610640% |
| 18.11357421875 | 56 | 224 | 55,207.215756 | 3.557996616% |
| 18.11357421875 | 112 | 448 | 55,207.215756 | 3.557996616% |

The selected long replay measures 448 committed output tokens over 8.114881250 milliseconds.
Each of fourteen clients completes 32 measured outputs.
All 574 submitted forwards complete after draining, and the measurement halves differ by approximately `2.66e-15` relative.
Measured occupancy peaks at 65.7196% on a vector execution resource, while the exclusive boundary arena occupies 42.3332% of the window.

The 18.4-microsecond control independently reproduces 3.502610640% MFU in both its short and long replays.
The more closely spaced periodic candidate improves on that control without reducing the charged memory policy.
Its resource statistics remain specific to this schedule and do not establish a universal throughput bound.

## Additional exclusions and controls

Six optimistic screens increase active/input slots to three or four while treating their additional storage as free.
They preserve all original costs, dependencies and physical resources and test issue factors zero, 1.001 and 1.16 with rotating submission slots.
Every screen remains at approximately 3.410066841% MFU.
These screens do not establish SRAM admission for the extra frames and do not exclude other scheduling choices.

The protocol-buffer change also requires revisiting earlier independent-batch exclusions.
The revised envelope subtracts the 1,027,072-byte transformer vocabulary backcharge, retains the native safety reserve and KV rounding/watermark, and treats additional frame storage as free.
It covers 36 TP/DCP/ownership families and 108 workspace-policy cases, including three TP-one cases rejected by their 533,988,352-byte global-layer weights alone.
The revised batch/PP envelope has 28,545 points, adding 43 to the [earlier study](lpx-pure-lp30-independent-batches.md).
Native layer-path bounds leave only TP16/DCP4, PP10, batch two and fourteen residents above the earlier 3.419600500% comparison reference.
An exact count finds one admissible partition in each of the three relaxed policies: ten six-layer stages.
The retained native PP-handoff path bounds that point at 3.303839087% MFU even with free boundary execution.
The greatest bound across the revised independent-batch class is therefore 3.406617750%.
These supplied batch choices cannot improve either the earlier comparison reference or the verified periodic candidate.

## Evidence and remaining scope

The [compressed experiment record](../experiments/lpx/periodic-resource-calendar-search.json.gz) retains the 43 calendar trials, six extra-frame screens, two independently verified certificates, four full-model replays, drained trace and resource statistics, revised batch exclusions, prior record values and reproduction sources.
Native program bodies and additional trial certificates remain at the local paths identified by their hashes.
The source identity remains `ae29c716bbf5693a8c9d70df1ee5c6435c3e93af93781bcc5aff57c2e8074986`.
No repository simulation code changes in this study.
The existing code baseline has 43 passing focused tests and 888 passing tests out of 890, with two documented environment issues.
Pinned Black formatting and checking leave all 582 Python files unchanged.

The calendar trial does not optimize every legal cyclic schedule or prove a minimum feasible period.
Different placement rules, batch shapes, tensor lifetimes, compute-unit overlap, cache formats and offload policies remain open.
Physical compiler execution, matching model-quality evidence and an objective-wide maximum are unverified.
