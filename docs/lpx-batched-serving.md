# LPX batched-serving control for the AA Gemma pair

Record date: 2026-10-23.
This control tests reason 4 of the [placement plausibility record](lpx-placement-plausibility.md).
Reason 4 proposes that the Artificial Analysis (AA) request shared each decode forward with other traffic, which would lower its speed and could flatten its dependence on context length.

## Question

Can background requests in the same decode forward, possibly combined with a higher attention matrix efficiency, bring the simulated AA request to the published 3,382 tokens per second (TPS) at 10K input and 3,431 TPS at 100K input?
A point matches when both simulated speeds lie within 5% of the published values, which means 3,213 to 3,551 TPS at 10K and 3,259 to 3,603 TPS at 100K.

## Method

[`scripts/run_lpx_batched_serving.py`](../scripts/run_lpx_batched_serving.py) builds each deployment with the bundled [fidelity configuration](../experiments/lpx/fidelity-study.json), compiled execution, a 0.35-microsecond chip-to-chip hop and zero collective launch cost.
These are the settings of the [context-parallel results](lpx-context-parallel.md#results).

Each point captures one native decode forward with the LPX streaming and fabric refinement.
The forward contains the AA request and `n` background requests.
The AA request sits at its mean generation context, 10,750 tokens for the 10K workload and 101,000 tokens for the 100K workload.
Its resident KV reserve covers 11,500 and 102,000 tokens.
Each background request decodes one token at a fixed context and reserves exactly that context.
The AA request's speed is one token per forward, so its TPS equals the reciprocal of the forward time.
Every request in the forward commits one token, and the background requests are not reported.

The mean-context approximation reproduces the curve-integrated serial results to 0.1 TPS at `n = 0`: 5,578.4 and 3,611.2 TPS on TP16/PP8.
Resident KV admission rejects a point when the AA request at its maximum context and all background requests do not fit in the static random-access memory (SRAM) budget of every rank.

| Dimension | Values |
| --- | --- |
| Layout | TP16/PP8 on 128 LP30 chips with layer counts 6, 8, 9, 7, 7, 10, 7, 6; the same with `KVP = 4`; TP16/PP16 on a full 256-chip rack with balanced layer counts; the same with `KVP = 4` |
| Background requests `n` | 0, 1, 3, 7, 15 |
| Background context | 1,000, 4,000, 16,000 tokens |
| Attention matrix efficiency `compute.attention_mfu` | 0.10 (study value), 0.30, 0.50 |

TP denotes tensor parallelism, PP pipeline parallelism and KVP the Helix-style KV-parallel size.
The attention matrix efficiency is the fraction of the matrix-unit peak that attention matrix work achieves.
The study value of 0.10 is an uncalibrated assumption.

### Batched LPX refinement

The LPX streaming refinement splits each all-reduce into 320-byte output vectors that the final matrix multiplication produces uniformly.
It previously accepted only one token per forward.
A batch of `B` single-token decode requests produces a final matrix multiplication with `B` output rows, and the all-reduce payload is the complete `B`-row output.
The refinement now accepts such batches and still rejects decode requests with more than one query token.
Its uniform-production assumption is unchanged, and single-token results are identical.
Without the refinement, the catalog collective backend gives 730.9 and 682.2 TPS for the TP16/PP8 serial case, so the refinement is required for comparable results.

## Expectations recorded before the sweep

A smoke run on TP16/PP8 at efficiency 0.10 gives 5,086.0 and 3,398.2 TPS with one 4,000-token background request.
One such request adds approximately 17 microseconds per forward at 10K and 15 microseconds at 100K.
The AA request's own global attention still adds the same amount of time between 10K and 100K inside a batched forward.
That increment is approximately 98 microseconds with `KVP = 1`, where vector-unit KV conversion bounds global attention and the matrix efficiency has little effect.
With `KVP = 4` it is approximately 60 microseconds at efficiency 0.10 and approximately 29 microseconds at efficiency 0.50, where the matrix unit no longer bounds the core.

Background requests that slow the 10K forward to the published 295.7 microseconds therefore leave the 100K forward near 295.7 microseconds plus that increment:

| Layout and efficiency | Expected 100K TPS when 10K matches 3,382 TPS |
| --- | ---: |
| `KVP = 1`, any efficiency | approximately 2,550 |
| `KVP = 4`, efficiency 0.10 | approximately 2,810 |
| `KVP = 4`, efficiency 0.50 | approximately 3,050 |

The expectation is that no point matches both published values within 5%.
Batching is expected to explain the 10K level but not the flat context dependence.
On 128 chips, SRAM limits the background traffic, and 15 background requests at 4,000 tokens already exceed the budget by approximately 2 MB on one rank.

## Results

The sweep completed all 156 points on 2026-10-23 with source fingerprint `d3fdcca7c42fa0350691c55e86875999a994277d5000939e770f8649d7150244`.
[`experiments/lpx/batched-serving-results.json`](../experiments/lpx/batched-serving-results.json) holds every point, its step time, its error against the published value and every rejection reason.
Each cell gives the AA request's speed in TPS at 10K and at 100K input, and the background column gives the number of background requests times their context.
An infeasible cell is one that resident KV admission rejected.

### TP16/PP8

| Background | Efficiency 0.10 | Efficiency 0.30 | Efficiency 0.50 |
| --- | ---: | ---: | ---: |
| none | 5,578 / 3,611 | 5,591 / 3,616 | 5,593 / 3,617 |
| 1 × 1,000 | 5,174 / 3,437 | 5,195 / 3,446 | 5,199 / 3,448 |
| 1 × 4,000 | 5,086 / 3,398 | 5,107 / 3,407 | 5,111 / 3,409 |
| 1 × 16,000 | 4,771 / 3,255 | 4,789 / 3,263 | 4,793 / 3,265 |
| 3 × 1,000 | 4,363 / 3,059 | 4,393 / 3,074 | 4,399 / 3,077 |
| 3 × 4,000 | 4,181 / 2,969 | 4,208 / 2,983 | 4,214 / 2,985 |
| 3 × 16,000 | infeasible | infeasible | infeasible |
| 7 × 1,000 | 3,317 / 2,505 | 3,352 / 2,525 | 3,359 / 2,529 |
| 7 × 4,000 | infeasible | infeasible | infeasible |
| 7 × 16,000 | infeasible | infeasible | infeasible |
| 15 × 1,000 | infeasible | infeasible | infeasible |
| 15 × 4,000 | infeasible | infeasible | infeasible |
| 15 × 16,000 | infeasible | infeasible | infeasible |

### TP16/PP8 with `KVP = 4`

| Background | Efficiency 0.10 | Efficiency 0.30 | Efficiency 0.50 |
| --- | ---: | ---: | ---: |
| none | 5,534 / 4,152 | 5,664 / 4,880 | 5,669 / 4,884 |
| 1 × 1,000 | 5,113 / 3,910 | 5,244 / 4,565 | 5,252 / 4,571 |
| 1 × 4,000 | 5,060 / 3,879 | 5,216 / 4,544 | 5,224 / 4,550 |
| 1 × 16,000 | 4,863 / 3,763 | 5,116 / 4,467 | 5,123 / 4,473 |
| 3 × 1,000 | 4,278 / 3,401 | 4,397 / 3,907 | 4,407 / 3,916 |
| 3 × 4,000 | 4,165 / 3,330 | 4,337 / 3,861 | 4,347 / 3,870 |
| 3 × 16,000 | 3,787 / 3,084 | 4,134 / 3,700 | 4,144 / 3,708 |
| 7 × 1,000 | 3,223 / 2,695 | 3,320 / 3,030 | 3,331 / 3,041 |
| 7 × 4,000 | 3,072 / 2,593 | 3,239 / 2,966 | 3,251 / 2,976 |
| 7 × 16,000 | 2,622 / 2,265 | 2,984 / 2,751 | 2,994 / 2,760 |
| 15 × 1,000 | 2,153 / 1,905 | 2,225 / 2,091 | 2,235 / 2,100 |
| 15 × 4,000 | 2,016 / 1,796 | 2,150 / 2,025 | 2,160 / 2,034 |
| 15 × 16,000 | infeasible | infeasible | infeasible |

### TP16/PP16

| Background | Efficiency 0.10 | Efficiency 0.30 | Efficiency 0.50 |
| --- | ---: | ---: | ---: |
| none | 5,357 / 3,517 | 5,368 / 3,522 | 5,370 / 3,523 |
| 1 × 1,000 | 4,957 / 3,340 | 4,976 / 3,349 | 4,980 / 3,351 |
| 1 × 4,000 | 4,877 / 3,303 | 4,895 / 3,312 | 4,899 / 3,314 |
| 1 × 16,000 | 4,586 / 3,168 | 4,603 / 3,176 | 4,606 / 3,177 |
| 3 × 1,000 | 4,172 / 2,964 | 4,199 / 2,978 | 4,204 / 2,981 |
| 3 × 4,000 | 4,005 / 2,879 | 4,030 / 2,892 | 4,036 / 2,895 |
| 3 × 16,000 | 3,464 / 2,589 | 3,483 / 2,599 | 3,487 / 2,601 |
| 7 × 1,000 | 3,164 / 2,417 | 3,196 / 2,435 | 3,202 / 2,439 |
| 7 × 4,000 | 2,947 / 2,288 | 2,975 / 2,305 | 2,980 / 2,308 |
| 7 × 16,000 | 2,324 / 1,894 | 2,342 / 1,906 | 2,345 / 1,908 |
| 15 × 1,000 | 2,133 / 1,764 | 2,161 / 1,784 | 2,167 / 1,788 |
| 15 × 4,000 | 1,927 / 1,622 | 1,951 / 1,638 | 1,955 / 1,642 |
| 15 × 16,000 | infeasible | infeasible | infeasible |

### TP16/PP16 with `KVP = 4`

| Background | Efficiency 0.10 | Efficiency 0.30 | Efficiency 0.50 |
| --- | ---: | ---: | ---: |
| none | 5,315 / 4,028 | 5,436 / 4,710 | 5,440 / 4,713 |
| 1 × 1,000 | 4,901 / 3,785 | 5,021 / 4,395 | 5,029 / 4,401 |
| 1 × 4,000 | 4,852 / 3,756 | 4,996 / 4,376 | 5,003 / 4,381 |
| 1 × 16,000 | 4,671 / 3,647 | 4,904 / 4,305 | 4,911 / 4,310 |
| 3 × 1,000 | 4,094 / 3,284 | 4,203 / 3,753 | 4,212 / 3,761 |
| 3 × 4,000 | 3,990 / 3,217 | 4,148 / 3,711 | 4,158 / 3,719 |
| 3 × 16,000 | 3,642 / 2,988 | 3,962 / 3,562 | 3,971 / 3,569 |
| 7 × 1,000 | 3,078 / 2,593 | 3,167 / 2,902 | 3,177 / 2,912 |
| 7 × 4,000 | 2,941 / 2,498 | 3,093 / 2,843 | 3,104 / 2,852 |
| 7 × 16,000 | 2,525 / 2,193 | 2,859 / 2,645 | 2,869 / 2,653 |
| 15 × 1,000 | 2,053 / 1,825 | 2,117 / 1,996 | 2,127 / 2,004 |
| 15 × 4,000 | 1,927 / 1,725 | 2,050 / 1,935 | 2,059 / 1,944 |
| 15 × 16,000 | 1,565 / 1,430 | 1,836 / 1,745 | 1,844 / 1,752 |

### Match outcome

No point matches both published values within 5%.
The nearest point by the larger of its two errors is TP16/PP8 with `KVP = 4` at efficiency 0.50 and seven 1,000-token background requests.
It gives 3,330.9 TPS at 10K, which is 1.5% below the published value, and 3,040.8 TPS at 100K, which is 11.4% below.
The same background at efficiency 0.30 gives 3,320.1 and 3,030.4 TPS.

The largest 100K-to-10K speed ratio in the sweep is 0.950, from TP16/PP16 with `KVP = 4` at efficiency 0.30 and fifteen 16,000-token background requests.
That point runs at 1,836 and 1,745 TPS, so it reaches the flattest shape only by slowing both workloads far below the published level.
The published ratio is 1.015.

### Comparison with the recorded expectations

The AA request's own 10K-to-100K increment stays constant across background traffic, as expected.
The table below takes the point whose 10K step time is nearest the published 295.7 microseconds and adds that point's increment to 295.7 microseconds.

| Layout and efficiency | Measured increment | Implied 100K TPS when 10K is 3,382 TPS | Expected |
| --- | ---: | ---: | ---: |
| `KVP = 1`, any efficiency, either layout | 97.7 microseconds | 2,542 | approximately 2,550 |
| `KVP = 4`, efficiency 0.10 | 60.1 to 60.7 microseconds | 2,806 to 2,810 | approximately 2,810 |
| `KVP = 4`, efficiency 0.30 | 28.8 microseconds | 3,082 | not recorded |
| `KVP = 4`, efficiency 0.50 | 28.6 microseconds | 3,083 | approximately 3,050 |

Every implied value lies within 1.1% of its expectation.
The published increment is a decrease of 4.2 microseconds, because the published 100K speed is higher than the published 10K speed.
Raising the efficiency from 0.30 to 0.50 changes the increment by only 0.2 microseconds, so a cost other than the matrix-unit attention work bounds the remaining 28.6 microseconds.

SRAM admission limits the background traffic as expected, with a stricter limit on 128 chips than recorded.
TP16/PP8 with `KVP = 1` rejects three 16,000-token, seven 4,000-token and all fifteen-request backgrounds.
Fifteen 4,000-token requests exceed the budget by 1.80 MiB on the first rejecting rank, close to the expected 2 MB, and seven 4,000-token requests are also rejected by 4.09 MiB.
`KVP = 4` spreads the AA request's KV over four ranks, so TP16/PP8 with `KVP = 4` admits everything except fifteen 16,000-token requests.
The full rack admits everything except fifteen 16,000-token requests with `KVP = 1` and admits every point with `KVP = 4`.

### Interpretation

Batching with other traffic explains the 10K level.
About seven short background requests in each forward slow the AA request to the published 10K speed on 128 or 256 chips.
Batching does not explain the context dependence.
Background requests add the same time at 10K and at 100K, so they cannot remove the AA request's own attention increment of 28.6 to 97.7 microseconds.
The simulated 100K speed therefore stays 10% to 26% below the published value whenever the 10K speed matches.

Reason 4 of the [placement plausibility record](lpx-placement-plausibility.md) is therefore only partly supported.
Server batching alone, with these cost parameters, cannot produce the published pair.
A matching deployment would also need a 100K attention cost close to the 10K cost.
Speculative decoding that commits several tokens per forward, or attention costs far below the current vector-unit KV conversion estimate, could provide that, and neither has been tested here.
