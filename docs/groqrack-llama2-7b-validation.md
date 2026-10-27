# GroqRack Llama-2-7B batch-one decode check

Record date: 2026-10-26.
The [Llama-2 70B check](groqchip-v1-llama2-validation.md) found simulated batch-one decode on first-generation Groq chips 3.5 to 5.2 times faster than GroqCloud's published speeds, whose configuration and load are unknown.
This record compares the simulator with a batch-one measurement on a known first-generation configuration.

## Measurement

Usami, Vishwanath and Bethel, [Prefill/Decode-Aware Evaluation of LLM Inference on Emerging AI Accelerators](https://arxiv.org/abs/2606.17104) (HPAI4S'26, co-located with IEEE IPDPS 2026), run Llama-2-7B in FP16 on a GroqRack.

| Item | Value |
| --- | --- |
| System | 72 GroqCards in nine nodes, 230 MB of SRAM and 80 TB/s per card, 15.42 GiB of SRAM in total |
| Software | GroqFlow SDK with a compiled Llama-2-7B binary provided by Groq |
| Execution | One request; the binary processes one token per invocation |
| Sampling | Each output token is sampled on the rank-0 host CPU and broadcast with MPI before the next invocation |
| Workload | Input and output lengths of 100, 200, 400, 800 and 1,600 tokens, equal to each other |
| Time per output token (TPOT) | 2.98 to 3.05 ms, nearly constant across lengths |
| Decode throughput | 328 to 336 tokens per second |

The TPOT is the decode loop's wall-clock time divided by the output length, so it includes the host sampling, the broadcast and the invocation of each forward.
The paper does not state the tensor-parallel and pipeline layout.

## Method

[`scripts/validate_groqrack_llama2_7b.py`](../scripts/validate_groqrack_llama2_7b.py) prices one batch-one decode forward with the method of the 70B check: the native attention/feed-forward runtime on the `groqchip_v1` device, its `groq_c2c_25g` links and TokenSim's generic Groq operator model.
It uses two layouts that fit FP16 weights in 230 MB per card, TP8/PP9 and TP4/PP18, both all-reduce algorithms (`ring` and `direct`), hop latencies of 0.5 and 0.3 µs and mean decode contexts of 150 and 2,400 tokens.

## Expectations recorded before the run

1. With direct all-reduce, the simulated step is 0.2 to 0.4 ms on both layouts.
2. With ring all-reduce, the simulated step is 0.5 to 0.9 ms.
3. Context changes the step by less than 5%, as in the measurement.
4. The measured TPOT therefore exceeds every simulated step by 2.1 to 2.8 ms, which this record attributes to per-token work outside the device forward: host sampling, broadcast and invocation.
5. Adding the direct-case difference to the direct 70B steps of 0.80 to 0.97 ms gives 3.4 to 3.8 ms, or 263 to 294 tokens per second, inside or near GroqCloud's published 241 to 300 tokens per second.

## Results

The run took 5 seconds.
[`experiments/lpx/groqrack-llama2-7b-results.json`](../experiments/lpx/groqrack-llama2-7b-results.json) holds every step and critical-path breakdown.

```bash
.venv/bin/python scripts/validate_groqrack_llama2_7b.py \
  --summary experiments/lpx/groqrack-llama2-7b-results.json
```

| All-reduce | Hop | TP8/PP9 | TP4/PP18 |
| --- | ---: | ---: | ---: |
| Ring | 0.5 µs | 645.1 to 647.2 µs | 405.3 to 409.4 µs |
| Ring | 0.3 µs | 460.1 to 462.2 µs | 323.3 to 327.4 µs |
| Direct | 0.5 µs | 252.1 to 254.2 µs | 274.3 to 278.4 µs |
| Direct | 0.3 µs | 224.3 to 226.4 µs | 244.7 to 248.8 µs |

Each range covers the mean decode contexts of 150 and 2,400 tokens.
On TP8/PP9 with direct all-reduce and 0.5 µs hops, the critical path holds 139.5 µs of all-reduce, 46.0 µs of attention, 34.2 µs of feed-forward compute and 9.2 µs of pipeline handoffs.

### Comparison with the expectations

1. Held: direct all-reduce gave 0.224 to 0.278 ms.
2. Missed: ring all-reduce gave 0.323 to 0.647 ms, below the expected 0.5 ms on TP4/PP18 and on TP8/PP9 with 0.3 µs hops, because TP4 rings have 6 rounds instead of 14.
3. Held: the 2,400-token context lengthened the step by 0.8% to 1.7%.
4. Held within 0.03 ms: the measured TPOT exceeds the direct steps by 2.70 to 2.83 ms and the ring steps by 2.33 to 2.73 ms, against an expected 2.1 to 2.8 ms.
5. Held: adding 2.70 to 2.83 ms to the direct 70B steps of 0.796 to 0.965 ms gives 3.50 to 3.79 ms, or 264 to 286 tokens per second, inside GroqCloud's published 241 to 300.

The derived values in this record are reproduced from the committed results by [`scripts/analyze_lpx_published_conditions.py`](../scripts/analyze_lpx_published_conditions.py), which writes [`experiments/lpx/published-conditions-analysis.json`](../experiments/lpx/published-conditions-analysis.json).

## Interpretation

The simulated device forward of Llama-2-7B takes about a tenth of the measured time per token, and the rest is nearly the same at every context and output length.
That pattern fits a fixed cost per token outside the forward, which the paper's execution model supplies: each token returns to the host CPU for sampling, is broadcast with MPI and starts a new invocation of the compiled binary.
The simulator does not model that loop.

The same fixed cost reconciles the [Llama-2 70B check](groqchip-v1-llama2-validation.md).
With 2.70 to 2.83 ms per token added, the simulated 70B decode with direct all-reduce gives 264 to 286 tokens per second, inside the 241 to 300 published for GroqCloud.
The earlier 3.5 to 5.2 times gap is therefore consistent with an accurate device model plus a host loop per token, provided that GroqCloud's 2023 and 2024 deployment used a loop like the GroqFlow binary's.
That proviso is untested: the 7B measurement uses a research SDK path, and GroqCloud's serving path is not described.

For LPX the result sets a bound.
AA's 3,382 and 3,431 TPS leave about 290 µs per token in total, so the LPX endpoint cannot run a host round trip of milliseconds per token.
Any per-token cost outside the forward on LPX must fit within the 0 to 25 µs that the [article-condition sweep](lpx-article-sweep.md), the [token-ratio record](lpx-token-ratio.md) and the [tray-latency record](lpx-tray-latency.md) leave.
The check supports the device-side multi-chip decode model on first-generation hardware, including direct all-reduce, and it does not test LP30 rates.
