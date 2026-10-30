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

The search ran on 2026-10-28.
It read the SHIP paper and supplement, the ISCA 2022 paper, which Groq now serves at [groq.com/isca-2022-paper](https://groq.com/isca-2022-paper), and NVIDIA's [LPX architecture](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) and [deterministic-execution](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-deterministic-execution-drives-power-efficient-high-interactivity-inference-on-nvidia-vera-rubin/) posts.
The ISCA 2020 paper was not available: its Groq download link returns a web page instead of the PDF.
The [Answer Fast BERT record](groqchip-v1-bert-validation.md) already notes that operators lasting microseconds hide fixed costs of a few hundredths of a microsecond, so it gives no bound.

### Collective cost on the first-generation LPU

[SHIP](https://proceedings.mlsys.org/paper_files/paper/2026/file/9c20f16b05f5e5e70fa07e2a4364b80e-Paper-Conference.pdf) page 6 states that an all-reduce across 8 LPUs reaches 50% of saturated bandwidth at 32 KiB and 90% at 80 KiB, and its Figure 6a marks the theoretical 8-LPU all-reduce peak at about 60 GB/s, read from the figure.
At those points one all-reduce takes about 32,768 B / 30 GB/s = 1.09 µs and 81,920 B / 54 GB/s = 1.52 µs.
Subtracting the wire time at 60 GB/s leaves a fixed time of 0.55 µs and 0.15 µs.
Page 5 gives 300 ns per hop.
Within one fully connected 8-LPU node, the supplement's RedBcast algorithm needs one hop and TiledRedBcast two, so an all-reduce there pays 0.3 or 0.6 µs of hop latency.
The fixed time of a small all-reduce on the first-generation LPU exceeds one hop by at most about 0.25 µs and stays within two hops, so the data leave no measurable launch cost beyond the hop latencies.

The [ISCA 2022 paper](https://groq.com/isca-2022-paper) measures seven intra-node links at a mean of 216 to 217 cycles (Table 2), about 0.24 µs at the 900 MHz of the first-generation product brief.
It gives the worst-case all-reduce latency of a 256-TSP system as 722 ns per hop over 3 hops, about 2.1 µs.
It states that the compiler tracks the order and time of every memory reference, so a consumer needs no flag or semaphore.

For comparison, the 132 collectives on the critical path of the closest LP30 setup take about 190 µs, about 1.4 µs each including their hop latencies, following the 188.5 µs of TP-group collectives in the [attention-group breakdown](lpx-attention-group.md).
A separate launch cost of 0.10 to 0.18 µs per collective is not visible in the first-generation data, and it is also not excluded, because the figure is read approximately and LP30 is a different chip.

### Instruction issue

SHIP page 2 states that each functional unit is issued one instruction per cycle and that the compiler statically schedules all instruction dispatch.
Page 6 states that pipelined small-tensor operations leave "no room for kernel-launch overhead, host intervention, or non-deterministic operation execution" and that the compiler pipelines on a cycle-by-cycle basis.
NVIDIA's posts state that the LPX compiler explicitly schedules computation, data movement and synchronization instead of relying on runtime schedulers.
No source gives a number for a per-kernel cost.
The model already charges 0.05 µs of fill and drain per operator, so these statements argue against an additional per-kernel floor without bounding it.

### Synchronization and invocation

ISCA 2022 section 3 describes the synchronization of a multi-TSP program.
Before a distributed inference starts, each level of a spanning tree of chips aligns on a 252-cycle epoch, costing `(floor(L / 252) + 1) * h` epochs for a link latency `L` and tree height `h`, which is one epoch per level for the measured 216-cycle links.
During execution, a lighter runtime resynchronization takes about one epoch.
At 900 MHz an epoch is 0.28 µs, so if every decode step started a new distributed inference, synchronization would add about 0.3 µs per tree level, or about 1 to 2 µs for a tree of three to five levels.
NVIDIA's architecture post mentions "periodic software synchronization" on LPX without a cost.

### Host involvement per token

SHIP page 13 states that production systems use the host "only for the initial pipeline invocation and final output collection", while the LPUs invoke the middle pipeline partitions themselves.
The supplement adds that the runtime sends a mask encoding of the request's KV pages to the LPUs at every timestep.
This description implies that at batch size 1 each token leaves the last pipeline stage for the host and that the host invokes the first stage with the next step's inputs.
No source states where the token is sampled or how long the host's collection and invocation take.
The closest setup charges a 4 µs host submission and omits the output collection.

### Expectations

1. Held: SHIP's Figure 6 gives a fixed all-reduce time of about 0.15 to 0.55 µs on 8 LPUs, at most about 0.25 µs above one 300 ns hop.
2. Held: no source states a runtime launch cost per kernel or per collective, and SHIP excludes kernel-launch overhead explicitly.
3. Missed: SHIP describes a host that collects each pipeline's output and invokes the pipeline, so a batch-one decode loop passes through the host at every token; the sampling location is not stated.
4. Held: NVIDIA's LPX posts give no collective launch or per-kernel cost.
5. Partly held: the search bounds the collective launch on first-generation hardware and gives an order of magnitude for synchronization per invocation, and it leaves the per-kernel floor and the host's per-token cost without a number.

## Interpretation

On first-generation hardware, published data show no measurable launch cost per collective beyond hop latency, state that instruction dispatch has no runtime launch, and give synchronization at the start of each invocation as about one 0.28 µs epoch per tree level.
If LP30 behaves the same way, per-step synchronization of about 1 to 2 µs is the only one of the three zeroed costs with a published size, and it supplies a small part of the 13 to 24 µs.
The per-token costs that the sources describe without a size are the host's collection of each output and its invocation of the next step; a host loop that costs about 10 to 24 µs more than the modeled 4 µs submission would supply the gap.
The [tray-latency record](lpx-tray-latency.md) gives the other candidate, a cross-tray latency of 0.80 to 0.85 µs instead of 0.75 µs.
None of these sources measures LP30, so the record narrows the candidates without identifying the cost.
