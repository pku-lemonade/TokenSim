# Independent request cohorts with different local and global batch widths

An analytical native-graph replay reaches approximately 3.822824% model floating-point operation utilization (MFU) on 444 LP30 chips with 54 resident requests.
Aggregate throughput is approximately 161,573.433 tokens per second (TPS).
Local-window stages execute two independent requests together, while each global stage executes the requests separately.
The previous [batch-one record](lpx-heterogeneous-batch-search.md) was approximately 3.820105% MFU on the same chip inventory and request count.
The improvement is approximately 0.002719 percentage points.
The objective-wide maximum remains unproven.
Physical compiler execution, SRAM bank allocation, shared-cache model quality and multi-rack timing remain unverified.

## Objective and useful work

The objective uses pure LP30 hardware, prioritizes MFU and imposes no minimum TPS.
All allocated transformer and boundary chips contribute to the denominator.
The workload remains Gemma 4 31B with fifty local-window layers, ten global-attention layers and a representative decode context of 101,000 tokens.
Key-value (KV) storage remains reserved through 102,000 tokens.
Weights and the experimental `value_rope_fp8` global cache use eight-bit floating point (FP8).
Activations and attention use bfloat16 (BF16), and the attention kernel remains `split_dot`.
Useful work remains 129,261,109,248 floating-point operations per committed output token.
Each allocated chip contributes `315e15 / 256` FP8 operations per second to the peak denominator.
Prompt-inclusive [complete-request serving](lpx-complete-serving-mfu.md) retains its separate metric and record.

## Cohort semantics and row ownership

A macro job contains two distinct requests and produces one output for each request.
The 54 resident requests form 27 fixed feedback cohorts.
Every local invocation executes the complete native batch-two graph.
Every global stage executes two complete native batch-one graphs on its assigned chips.
The embedding and vocabulary head each execute a complete batch-two graph.
Every cohort row passes through each of the sixty logical layers exactly once.
No additional output receives credit for repeated computation or auxiliary execution.

For macro-job issue period P, useful throughput is `2/P`.
Each macro job must finish within `27 * P` before its cohort can issue another token.
Both requests retain their own KV allocation and feedback history.
The replay uses the canonical cohort driver with `resident_requests=54` and `batch=2`.
Its token counts and throughput come directly from that driver.
Batch-one and batch-two toy controls reproduce the canonical fixed-assignment metrics bit for bit.
Additional controls check rotating frames with an odd number of cohorts, one outstanding forward per request and complete drain.

Native segments retain complete replicated hidden rows after their output reduction and residual reuse.
Local-to-global handoffs transfer two rows, totaling 21,504 BF16 bytes per destination.
Global-to-local handoffs transfer each 10,752-byte row separately and record its target row offset.
The next local stage waits for both incoming rows.
The second global invocation acquires active storage after the first invocation's longest compute has completed on every participating rank.
That additional scheduling gate preserves all original dependencies and service costs.

## SRAM ledgers and different frame periods

Every transformer chip retains two complete native scratch, dense-residual and communication workspaces.
Two complete incoming hidden-row buffers remain reserved at the native batch width of that stage.
Each chip retains its 5,000,000-byte safety reserve, batch-scaled vocabulary backcharge, sixteen-token KV block rounding and one-percent KV watermark.
The boundary pool retains the complete tied embedding matrix, final normalization, vocabulary projection, softcap and greedy selection.
Its memory ledger reserves eight complete query workspaces for the paired-cohort cases.

Local and boundary frames alternate by macro-job submission.
Each concrete local frame therefore repeats every `2 * P`.
The global stage uses one fixed batch-one frame for each of the two rows.
Each concrete global frame repeats every P.
Those two global frames are the two reserved native workspaces.
The search does not assume four global workspaces or multiply global residency by the pipeline depth.

Input-buffer acquisition waits until the upstream output exists.
The source active frame remains leased through the outgoing transfer, and the target input frame remains leased through its entry copy.
Physical resource intervals repeat every P.
The linear program assigns the recorded period multiplier to each concrete arena lease.
Independent seven-time-unit toy leases require `P >= 7` for a frame reused every P and `P >= 3.5` for a frame reused every `2 * P`.
Separate calendar controls distinguish those lease constraints from full-cohort feedback.

Fresh native memory admission succeeds for all six searched compositions.
The limiting global rank retains 378,990 watermark-adjusted bytes, and request 55 exceeds the global capacity.
The selected batch-two local segments retain 252,229,054 bytes for the one-layer TP4 role and 21,197,368 bytes for the four-layer TP8 role.
This memory evidence covers the declared byte ledger and does not establish a physical SRAM bank layout.

## Finite search coverage

Twelve complete compositions compare different local segment widths, local cut orders, head widths and batch-one controls.
Six have necessary native-graph bounds above the previous 3.820105% record and receive calendar searches.
The remaining six are excluded only for their supplied native graphs and memory policy.
Calendar construction uses exact physical resource-incidence equivalence to reduce repeated exclusion constraints.
The final graphs, interval audits and native replays retain every original physical key.
The experiment tests 156 initial calendars and 100 additional calendars near the two strongest paired layouts.
Those calendars supply 177 numerical fixed-order linear programs, of which 127 succeed and 50 report infeasible supplied orders.
A failed supplied order does not exclude another schedule for the same graph.

| Local pattern within each six-layer block | Head chips | Total chips | Native bound, MFU | Best fixed-order numerical MFU |
| --- | ---: | ---: | ---: | ---: |
| Batch two, five layers at TP16 | 3 | 483 | 4.206424% | 3.665772% |
| Batch two, five layers at TP16 | 4 | 484 | 4.219359% | 3.669139% |
| Batch two, four layers at TP8 then one at TP4 | 3 | 443 | 3.939744% | 3.793954% |
| Batch two, four layers at TP8 then one at TP4 | 4 | 444 | 3.948255% | 3.812291% |
| Batch two, one layer at TP4 then four at TP8 | 4 | 444 | 3.953141% | 3.822824% |
| Batch one, TP4 cuts of one, two and two layers | 4 | 444 | 4.374209% | 3.820106% |

Every row in this table uses a batch-one global layer at TP32 with decode context parallelism (DCP) 8.
The numerical column precedes the small pacing margin and independent replay checks.
The batch-one control with delayed input acquisition reproduces the previous record after applying that margin.
It does not demonstrate a separate throughput gain from delayed acquisition alone.

The other paired layouts have upper bounds of 3.529444% for TP4 cuts of one, two and two layers, and 3.814646% for TP8 cuts of two and three layers.
Paired single-rack layouts using a five-layer TP8 local segment and a TP16/DCP4 global layer have bounds of 3.408886% on 243 chips and 3.409714% on 244 chips.
Batch-one TP16 five-layer controls have bounds of 3.662408% on 483 chips and 3.654841% on 484 chips.
These bounds leave other native geometries, memory policies, frame counts and graph compositions open.

## Selected replay

The selected block contains one local layer at TP4, four local layers at TP8 and one global layer at TP32/DCP8.
Ten blocks allocate 120 local-stage chips and 320 global-stage chips across thirty transformer stages.
Four additional boundary chips bring the allocation to 444 chips.
The optimized paired issue period receives a relative pacing margin of `1e-7`, producing approximately 12.378273 microseconds per macro job.
The complete forward spans approximately 334.213360 microseconds.
The schedule retains 98,081 original nodes and adds readiness delays to 17,151 nodes.
Independent interval sweeps cover 13,731 original physical resources and 82 arena leases.
They check three neighboring copies, native dependencies, release reachability and complete-cohort feedback.
Concrete delayed profiles preserve every original non-delay field.

The short replay uses four warm-up rounds and sixteen measured rounds.
It measures 864 output tokens from 432 paired completions and drains all 567 submitted macro jobs.
The long replay uses eight warm-up rounds and thirty-two measured rounds.
It measures 1,728 output tokens from 864 paired completions and drains all 1,107 submitted macro jobs.
Those drained macro jobs represent 2,214 logical request forwards.
Each of the 54 requests contributes 32 measured outputs.
Both measurement windows produce approximately 161,573.433 TPS and 3.822824% MFU.
No submitted job remains outstanding after drain.

## Remaining coverage and evidence

The simulator retains its declared 1.15-microsecond transport fallback beyond 256 chips.
Multi-rack timing is uncalibrated, so the reported result remains conditional on that assumption.
Other cyclic resource orders and nonuniform local/global block patterns remain open.
Global-only TP64 and wider groups require explicit output-channel fragments within attention heads before the native planner can represent them.
The current whole-head restriction does not establish hardware infeasibility.
Relocating immutable past KV onto spare local-stage LP30 SRAM also requires explicit capacity, transfer and prefetch-lifetime validation.

The [compressed experiment record](../experiments/lpx/mixed-stage-cohort-search.json.gz) retains the twelve complete native compositions, required component graphs, all calendar and numerical receipts, six native memory controls, best numerical primal/dual certificates, semantic controls and both complete replay witnesses.
The portable audit reconstructs constraints from original physical keys and checks useful-output counts, row coverage, native costs, copied frame names, memory blocks and drained request traces.
The production simulation source identity remains `ae29c716bbf5693a8c9d70df1ee5c6435c3e93af93781bcc5aff57c2e8074986`.
