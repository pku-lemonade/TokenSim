# Published evidence on Groq fixed costs per collective and per kernel

Record date: 2026-10-28.
The [step-cost audit](lpx-step-cost-audit.md) found that the closest AA setup, TP32/PP8 with decode context parallelism (DCP) of 8 on 256 LP30 chips, runs with three fixed costs at zero: the collective launch, the per-kernel instruction floor and the per-step invocation.
Both AA values fall within 5% only if these and any other unmodeled costs add 12.7 to 23.5 µs per step, for example a collective launch of 0.096 to 0.178 µs over 132 collectives or an instruction floor of 0.012 to 0.023 µs over 1,036 kernel components.
This record searches published Groq and NVIDIA material for values or bounds of these costs.

## Sources

The search covers the sources in the [Groq hardware evidence record](groq-hardware-evidence.md) that describe execution or communication timing:

| Source | Hardware | Content searched |
| --- | --- | --- |
| SHIP, MLSys 2026, paper and supplement | First-generation LPU | All-reduce bandwidth against message size (Figure 6), hop latency, algorithms, decode loop and sampling |
| Think Fast, ISCA 2020 | First-generation TSP | Instruction issue, functional-unit latencies and static scheduling |
| Software-defined tensor streaming multiprocessor, ISCA 2022 | First-generation TSP systems | Chip-to-chip latency, synchronization and collective timing |
| Answer Fast, ASAP 2022 | First-generation TSP | Per-operator timing of BERT-base |
| NVIDIA LPX blog posts and Hot Chips 2026 slides | LP30 | Chip-to-chip latency, synchronization and execution model |

Values from first-generation hardware do not transfer to LP30 directly.
The record reports them as references for the order of magnitude.

## Expectations recorded before the search

1. SHIP's Figure 6 or its text allows a fixed latency per all-reduce to be derived for the first-generation LPU, and the part not explained by its 300 ns hop latency is below 0.5 µs per all-reduce.
2. None of the sources states a runtime launch cost per kernel or per collective; they describe compile-time scheduling without runtime dispatch.
3. SHIP describes a production decode loop that samples tokens without a host round trip per token.
4. No public LP30 source gives a collective launch or a per-kernel cost.
5. The search bounds at most one of the three fixed costs, the collective launch, and leaves the per-kernel instruction floor without a published bound.

## Findings

Pending.
