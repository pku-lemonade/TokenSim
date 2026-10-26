# GroqChip v1 BERT-base validation of the LP30 cost method

Record date: 2026-10-26.
The [acceptance evidence record](lpx-acceptance-evidence.md) concluded that the simulated LP30 decode step may be 1.3 to 3.2 times too short.
This record tests the LP30 cost method against the only paper-grade single-chip measurement on the first-generation Groq chip.

## Measurement

Ahmed et al., [Answer Fast: Accelerating BERT on the Tensor Streaming Processor](https://arxiv.org/abs/2206.11062) (ASAP 2022), run quantized BERT-base on one first-generation tensor streaming processor (TSP) at 900 MHz.
The workload is batch 1 at sequence length 128, with INT8 matrix multiplication and FP32 nonlinear operations.
Over 4,000 inferences the mean latency is 128.9 microseconds, and the 1st and 99th percentiles are 128.5 and 129.5 microseconds.
The latency includes host-to-chip and chip-to-host transfers over PCIe and excludes tokenization.
The encoder blocks account for 83% of the latency, divided equally between the 12 layers, which gives about 107 microseconds for the encoder and 8.9 microseconds per layer.
The paper states that its implementation hides the latency of the non-matrix operations behind the matrix multiplications.
It also reports that the GroqAPI latency estimate was within 1% of the measurement.

One encoder layer performs approximately 1.86 GFLOP of matrix work, so 8.9 microseconds per layer corresponds to about 209 TOPS, or 28% of the 750 TOPS INT8 peak.

## Method

[`scripts/validate_groqchip_bert.py`](../scripts/validate_groqchip_bert.py) prices the eight operators of one encoder layer on the `groqchip_v1` device: the Q/K/V, output and two feed-forward matrix multiplications, the attention core, two layer normalizations and GELU.
It reports a serialized sum of all operators and a matrix-only sum that represents the paper's hiding of non-matrix work.
The attention query uses TokenSim's causal prefill formula, which halves the attention work of the bidirectional encoder; attention is about 3% of the layer's matrix work.

| Case | Estimator | Settings |
| --- | --- | --- |
| Generic Groq model | `GroqTSPModel` with the device defaults | Matrix efficiency 0.8, attention efficiency 0.6, 0.2-microsecond launch per operator |
| LP30 method, study efficiencies | LP30 unit estimator with GroqChip v1 rates | Matrix efficiency 0.3 and attention efficiency 0.1, as in the LP30 fidelity study |
| LP30 method, attention efficiency 0.5 | Same | Matrix efficiency 0.3, attention efficiency 0.5 |
| LP30 method, device-default efficiencies | Same | Matrix efficiency 0.8, attention efficiency 0.6 |

The LP30 unit estimator uses the LP30 study's vector efficiency of 0.5, shuffle efficiency of 0.8, 320-byte vectors and 0.05-microsecond fill and drain, with no launch cost.
The GroqChip v1 vector rate is 320 lanes of 16 ALUs at 900 MHz, or 4.608 trillion operations per second.
The shuffle rate is the 20 TB/s operand bandwidth into the MXM and VXM that the BERT paper states.
Exponentials run at 1/20 of the vector rate, the ratio that the LP30 study assumes.
No expectation was recorded before the first run, so this record is exploratory.

```bash
.venv/bin/python scripts/validate_groqchip_bert.py \
  --summary experiments/lpx/groqchip-v1-bert-results.json
```

## Results

[`experiments/lpx/groqchip-v1-bert-results.json`](../experiments/lpx/groqchip-v1-bert-results.json) holds every operator time.

| Case | Layer, matrix-only | Layer, serialized | Encoder error, matrix-only | Encoder error, serialized |
| --- | ---: | ---: | ---: | ---: |
| Generic Groq model | 4.08 µs | 4.71 µs | −54.3% | −47.2% |
| LP30 method, study efficiencies | 9.76 µs | 14.78 µs | +9.5% | +65.7% |
| LP30 method, attention efficiency 0.5 | 9.49 µs | 14.51 µs | +6.5% | +62.7% |
| LP30 method, device-default efficiencies | 4.45 µs | 9.46 µs | −50.1% | +6.2% |

The measured layer takes 8.9 microseconds.
Under the LP30 method with the study efficiencies, the matrix multiplications take 2.06, 0.72, 2.73 and 2.74 microseconds and are bound by the matrix unit, and GELU takes 4.49 microseconds on the vector unit.

## Interpretation

The generic Groq model with its default matrix efficiency of 0.8 predicts about half of the measured latency.
The LP30 method with the LP30 study's matrix efficiency of 0.3 predicts the measured layer within 6.5% to 9.5% when the non-matrix work is hidden, as the paper describes.
The 28% matrix utilization implied by the measurement agrees with that efficiency.
The serialized device-default case also lands within 6.2%, but it requires 80% matrix utilization at 128 tokens and serial non-matrix work, which contradicts the paper's description of its schedule.

The check supports the LP30 study's matrix efficiency for compute-bound matrix work on a single chip.
It does not test the costs that dominate the simulated LP30 decode step:

- Decode matrix multiplications process one token per request, so they are bound by weight streaming into the matrix unit and not by matrix arithmetic.
  The BERT paper's 20 TB/s operand bandwidth is one quarter of the 80 TB/s on-die memory bandwidth in the GroqChip v1 product brief.
  If LP30's operand bandwidth were similarly below the 150 TB/s SRAM bandwidth that the LP30 study uses, decode weight streaming would be slower than simulated.
- Collectives and chip-to-chip hops, which take about half of the simulated LP30 step, do not occur in a single-chip run.
- Fixed per-operator costs are invisible in operators that each take microseconds.
- The GELU estimate depends on the assumed exponential rate, so the vector-unit rates are not validated.

The BERT check therefore leaves the possible decode-step underestimate unresolved and narrows its likely location to weight streaming, communication and fixed per-operator costs.
A multi-chip decode measurement, such as the GroqCloud Llama-2 70B throughput, is needed to test those costs.
