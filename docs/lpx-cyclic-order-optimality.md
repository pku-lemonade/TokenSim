# Conditional cycle bounds for LP30 decode schedules

An exact rational certificate establishes a minimum period of 18.113541923102 microseconds for the supplied cyclic resource order and iteration assignment.
On 163 LP30 chips, this gives a conditional ceiling of 3.558002959680344% model floating-point operation utilization (MFU).
The earlier [periodic replay](lpx-periodic-resource-calendar-search.md) lies within two parts per million of this ceiling.
The certificate does not bound different resource orders, operator graphs, placements or hardware inventories.
The objective-wide maximum remains unproven.

## Workload and scheduling scope

The workload remains Gemma 4 31B decode at 101,000 tokens, with key-value (KV) storage reserved through 102,000 tokens.
Weights and the experimental shared global cache use eight-bit floating point (FP8), while activations use bfloat16 (BF16).
The transformer uses tensor parallelism (TP) 16, decode context parallelism (DCP) 4 and pipeline parallelism (PP) 10, with six layers per stage.
Its 160 chips are separate from the three-chip tied embedding and language-model head pool.
All allocated chips contribute to MFU, and no minimum throughput is imposed.
Prompt-inclusive serving retains its [separate metric](lpx-complete-serving-mfu.md).

The current replay reserves two active and two incoming slots per stage.
The fourteen-client policy retains the full extra native scratch charge, vocabulary backcharge, 5,000,000-byte safety reserve, native KV block rounding and one-percent watermark.
Its reserve field is 8,217,536 bytes, and its minimum watermark-adjusted headroom is 16,861,300 bytes per chip.
Physical compiler execution, static random-access memory (SRAM) bank placement and matching shared-cache model quality remain unverified.

## Additional calendar trials

The anchored search reserves the longest native compute interval on each of 160 transformer chips before placing the remaining operations.
Eight phase policies at five periods give forty trials.
Four trials cannot place a later operation in a common free interval.
The other thirty-six place physical operations but violate arena lifetimes or the fourteen/fifteen-client feedback limits.
Their longer end-to-end paths require more resident requests than the tested policies admit.
These trials supply no eligible throughput improvement.

A backward construction schedules operations from completion toward submission while preserving the readiness lag on every reversed edge.
Its twelve tested periods all fail to place a later operation in the reverse calendar.
Controlled examples check wraparound, occupied-gap rejection and reversed dependency timing.
Calendar failure concerns the specific heuristic and does not prove general schedule infeasibility.

## Fixed-order linear program

The linear program retains the cyclic operation order and iteration labels from the independently checked periodic certificate.
Its variables are one start time per node and the output period P.
Every constraint has the following form:

```text
start[v] >= start[u] + lag[u,v] - distance[u,v] * P
```

Native dependency rows have zero iteration distance and retain the predecessor duration plus the consumer's original readiness delay.
An exclusive-resource row places the subsequent operation after the current occupancy interval, with the supplied number of period crossings.
Arena rows use the actual release descendant as their interval endpoint.
Alternating frame arenas repeat every 2P, while the shared head arena repeats every P.
Forward completion must fit within N times P, where N is the resident request count.
All original compute durations, occupancy intervals, resource identities and dependency edges remain present.

The input graph contains 96,774 nodes.
Its 370,379 dependency rows and 4,384,136 resource-order rows reduce to 388,978 distinct inequalities before feedback constraints.
The complete program has 96,775 variables and 388,989 constraints.
[SciPy's linear-programming interface](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.optimize.linprog.html) invokes the HiGHS solver.
The experiment records SciPy 1.13.1 and NumPy 2.0.2 from the existing system environment.

Fourteen and fifteen residents produce the same minimum period within numerical precision.
The numerical primal violations remain below `1e-9` nanoseconds.
The independent rational certificate supplies the period bound without relying on the solver's success status or floating-point dual residual alone.

## Exact critical-cycle certificate

The certificate contains 432 inequalities involving 432 node variables.
It uses 430 native dependency rows and two resource-order rows.
Each row is assigned unit weight.
The sum cancels every start-time variable exactly, and the total iteration distance is one.
The cycle traverses the initial entry copy, local-layer work and a global-attention compute operation.
Its resource-order successors are checked independently against repeated intervals from the original valid schedule.

```text
sum(start[v] - start[u]) = 0
sum(distance[u,v]) = 1
P >= sum(lag[u,v])
P >= 18.11354192310192 microseconds
MFU_percent <= 100 * 129261109248 / (163 * (315e15 / 256) * P_seconds)
```

The verifier sums the original binary floating-point durations and readiness lags as exact rational numbers.
It preserves the rational numerator and denominator with every supporting inequality.
No request-feedback row participates in this cycle.
Increasing only the resident count cannot lower its period bound while the supplied operator graph and cyclic resource order remain fixed.
Changing that order or changing the operators requires a different proof.

## Reordered models and native replay

Two proposed resource orders move the entry copy, or the entry copy plus the initial attention prefix, before the longer global-attention interval.
Each proposal changes 320 worker/vector resource lists and retains the native data dependencies.
The screen tests the original iteration labels and one alternate label assignment at fourteen and fifteen residents.
HiGHS reports all eight specified models infeasible.
These numerical outcomes do not exclude untested cyclic orders or iteration assignments.

The fourteen-client long replay reaches 55,207.314188 aggregate tokens per second and 3.558002959671% MFU.
It measures 448 committed outputs and drains all 574 submitted forwards.
Each client completes 32 measured outputs, and the measurement halves differ by approximately `1.03e-13` relative.
The forward spans 237.917179530296 microseconds at a period of 18.113541923102 microseconds.
The observed MFU differs from the exact conditional ceiling by less than `1e-10` percentage points.
The construction adds readiness delays to 1,748 nodes and preserves every native service time.

The replay uses the original compute and communication services and adds only readiness delays.
Independent interval sweeps check the original and delayed graphs across three neighboring periods.
The two concrete programs retain the original dependencies, physical resources and release indices.
The long control records actual resource occupancy and drains every submitted forward.

## Evidence and remaining work

The [compressed experiment record](../experiments/lpx/cyclic-order-optimization-search.json.gz) retains all 52 calendar trials, ten LP receipts, complete base constraints, changed-order constraint deltas, primal/dual data, the exact rational certificate, both native replays, drained traces and reproduction sources.
Local native program bodies are identified by their file hashes.
Independent checks validate both 96,774-node programs, the portable rational certificate and the base primal/dual constraints.
Pinned Black formatting and checking leave all 582 Python files unchanged.
The simulation source identity remains `ae29c716bbf5693a8c9d70df1ee5c6435c3e93af93781bcc5aff57c2e8074986`.
The existing-format decode and prompt-inclusive serving records remain unchanged.

Groq's [ISCA 2022 paper](https://www.groq.com/wp-content/uploads/2023/05/GroqISCAPaper2022_ASoftwareDefinedTensorStreamingMultiprocessorForLargeScaleMachineLearning-1.pdf) describes compiler-controlled communication and synchronization on an earlier processor generation.
That paper supports investigating explicit scheduling and supplies no LP30 calibration for this experiment.
Different cyclic orders, cuts, tensor widths, cache policies, compute-unit overlap and complete-request schedules remain open.
The fixed-order certificate does not establish the maximum across those choices.
