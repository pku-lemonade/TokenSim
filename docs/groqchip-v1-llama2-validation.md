# GroqChip v1 Llama-2 70B decode check

Record date: 2026-10-26.
The [BERT validation](groqchip-v1-bert-validation.md) supported the LP30 method's matrix efficiency on one chip but could not test multi-chip decode.
This record compares simulated Llama-2 70B decode on first-generation Groq chips with published GroqCloud speeds.

## Published speeds

| Source | Speed | Conditions |
| --- | ---: | --- |
| [Groq press release](https://www.prnewswire.com/news-releases/groq-sets-new-large-language-model-performance-record-of-300-tokens-per-second-per-user-on-meta-ai-foundational-llm-llama-2-70b-301980280.html), November 2023 | more than 300 tokens per second per user | Groq's own measurement |
| Artificial Analysis, February 2024, as quoted by [Groq](https://groq.com/newsroom/groq-lpu-inference-engine-leads-in-first-independent-llm-benchmark) | 241 tokens per second median | Public GroqCloud API, prompts of about 100 tokens and about 200 output tokens |
| Anyscale LLMPerf, as quoted by [Groq](https://groq.com/groq-lpu-inference-engine-crushes-first-public-llm-benchmark/) | 270 or more tokens per second | Groq states that LLMPerf includes input processing time |

300 and 241 tokens per second correspond to 3.33 and 4.15 milliseconds per output token.
No source states the chip count, weight precision, parallel layout, batching or load of these deployments.
A 576-chip system appears only in an unverified [forum comment](https://news.ycombinator.com/item?id=38739951).
Llama-2 70B in FP16 needs about 140 GB, which exceeds the 132.9 GB of SRAM on 576 chips, so a 576-chip deployment would need weights of about 8 bits or fewer.

## Method

[`scripts/validate_groqchip_llama2.py`](../scripts/validate_groqchip_llama2.py) builds the native attention/feed-forward runtime on the `groqchip_v1` device, whose collectives and pipeline handoffs use the device's `groq_c2c_25g` links of 4 lanes at 25 Gb/s.
The runtime's nominal placement topology is `rubin_nvl72`, and it does not enter the step: on 2026-10-26, setting that topology's NVLink latency to 50 µs left the TP8/PP41 direct step at 908.68 µs.
An earlier version of this record named the `groq_node8_rack72` topology instead.
It prices one batch-one decode forward at a context of 200 tokens and attributes the critical path to operations.
The first and last pipeline stages hold one layer each beside the embedding and output head, and the other stages share the remaining 78 layers evenly.
Operators use TokenSim's generic Groq model with matrix efficiency 0.8, attention efficiency 0.6 and a 0.2-microsecond launch per operator.

| Dimension | Values |
| --- | --- |
| Layout and weight precision | TP8/PP41 (328 chips) INT8, TP8/PP72 (576 chips) INT8, TP16/PP36 (576 chips) INT8, TP16/PP41 (656 chips) FP16 |
| All-reduce algorithm | `ring`, the layout default with 14 rounds on 8 chips; `direct`, with 2 rounds |
| Chip-to-chip hop latency | 0.5 microseconds (catalog value), 0.3 microseconds (the LPUv1 hop latency reported by SHIP) |

SHIP reports that the production LPUv1 system uses direct all-reduce algorithms (RedBcast, TiledRedBcast and PAARD) over fully connected 8-chip nodes, so the `direct` cases represent the production collectives.
The LP30 study also prices collectives as direct exchanges.
No expectation was recorded before the first run, so this record is exploratory.

## Results

[`experiments/lpx/groqchip-v1-llama2-results.json`](../experiments/lpx/groqchip-v1-llama2-results.json) holds every step time and critical-path breakdown.

| All-reduce | Hop | TP8/PP41 INT8 | TP8/PP72 INT8 | TP16/PP36 INT8 | TP16/PP41 FP16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ring | 0.5 µs | 533 | 517 | 320 | 313 |
| Ring | 0.3 µs | 705 | 681 | 465 | 450 |
| Direct | 0.5 µs | 1,100 | 1,036 | 1,153 | 1,067 |
| Direct | 0.3 µs | 1,196 | 1,129 | 1,256 | 1,156 |

Values are simulated tokens per second for one user.

On TP8/PP72 with ring all-reduce and 0.5-microsecond hops, the 1,933.8-microsecond step contains 1,496.3 microseconds of all-reduce, 145.9 of feed-forward compute, 117.7 of attention compute and 128.6 of pipeline handoffs.
Each ring all-reduce of a 16 KiB activation takes 9.29 microseconds.
With direct all-reduce the same layout spends 530.3 microseconds in all-reduce, about 3.3 microseconds per all-reduce, and the step falls to 964.8 microseconds.

## Interpretation

The ring cases overlap the published 241 to 300 tokens per second, but they rely on a 14-round ring that the production system does not use.
With direct all-reduce, the simulated speeds are 1,036 to 1,256 tokens per second, 3.5 to 5.2 times the published values.
The published speeds therefore leave 2.4 to 3.3 milliseconds per token, or about 30 to 41 microseconds per layer, that the simulation does not produce.

Several causes could contribute:

- **Load and configuration.**
  The published speeds come from a shared production service whose batching, chip count and layout are unknown.
  Pipeline parallelism alone does not slow a user's token when each stage serves one micro-batch at a time, but larger micro-batches, queueing or time-sharing would.
- **Weight streaming.**
  If matrix weights stream at the BERT paper's 20 TB/s operand bandwidth instead of 90% of 80 TB/s, the 264 microseconds of decode compute on TP8 grow to roughly 950 microseconds.
  That accounts for about a quarter of the missing time.
- **Per-token costs outside the model**, such as host interaction for sampling or synchronization between stages.

The check cannot separate these causes, because the published speeds describe an unknown production configuration.
It agrees in direction with the [acceptance evidence](lpx-acceptance-evidence.md): both suggest that the simulated decode step is shorter than real decode on Groq hardware.
A measured batch-one decode latency on a known Groq configuration would be needed to calibrate the size of the difference.
The [GroqRack Llama-2-7B check](groqrack-llama2-7b-validation.md) supplies one: a fixed per-token cost of 2.70 to 2.83 ms outside the forward, measured on a host-driven decode loop, brings these direct-all-reduce steps to 264 to 286 tokens per second.
