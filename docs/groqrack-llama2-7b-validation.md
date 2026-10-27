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

Pending.
