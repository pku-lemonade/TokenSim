# Collecting operator latency data

These scripts produce the measured (`grade A`) tables that `operator_table`
prefers over the analytical model. Run them **on the target device** inside a
PyTorch environment matching the serving stack you want to model (the kernel
choice, not just the GPU, decides the latency).

| Script | Table | Needs |
| --- | --- | --- |
| `profile_gemm.py` | `gemm_perf` | torch (cuBLAS), optional `vllm`/`tensorrt_llm` quant kernels |
| `profile_attention.py` | `context_attention_perf`, `generation_attention_perf` | torch + `flash_attn` (prefill) and `flashinfer` or vLLM paged attention (decode) |
| `profile_elementwise.py` | `elementwise_perf` | torch |
| `profile_moe.py` | `moe_perf` | vLLM fused MoE kernels |
| `nccl-tests` | `collective_perf` | `nccl-tests` binaries; import with `operator_data.cli import-nccl` |

Every script:

1. reads the shapes it must measure from a shape manifest
   (`python -m TokenSim.operator_data.cli manifest --models ... --tp ...`),
2. warms up, then times each shape with CUDA events (median of N repeats),
3. writes a CSV with the table's key columns plus `latency_us`, `kernel`,
   `source_id`, `device_name`, `driver`, `framework_versions`.

Convert CSVs into a package with `operator_data.cli import-csv` (or drop them
next to a `generation_meta.yaml`; `OperatorDataPackage.load` reads `*_perf.csv`
when no parquet exists).

Measurement rules that keep tables comparable with the AIConfigurator data:

* Report the time of **one** kernel invocation (no batching of launches),
  median over at least 20 iterations after 5 warm-ups.
* Lock clocks (`nvidia-smi -lgc`) or record that clocks were unlocked.
* Use the weight layout the serving framework uses (e.g. column-major
  `[k, n]` for cuBLAS, weight-only quant kernels for `int8_wo`/`int4_wo`).
* For attention, `context_len` is the KV length seen by each query; batches
  are padded to a uniform length.
* For MoE, `num_tokens` is the number of tokens entering the layer *before*
  top-k expansion; `distribution` names the routing pattern used by the
  micro-benchmark (`uniform` = evenly spread).
