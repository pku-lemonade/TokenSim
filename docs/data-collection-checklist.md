# Operator Latency Data Collection Checklist

This checklist enumerates the measured data required to run the `operator_table`
backend without falling back to analytical estimation. It is organized by device,
operator table, shape grid, and collection method.

After any simulation run, the file `results/.../missing_shapes_<qps>.json`
reports the exact primary keys that were absent during that experiment; this file
can be fed directly as input to the next collection round. Before running a
simulation, use the coverage preview command:

```bash
python -m TokenSim.operator_data.cli coverage --device <id> --models <m> --tp <n>
```

---

## 1. Current Data Status

All measured GPU tables come from NVIDIA AIConfigurator (commit `f254959`),
imported for the vLLM (default), TensorRT-LLM and SGLang stacks; per-device
details and versions are in [`data/devices/README.md`](../data/devices/README.md).
No package has an `elementwise` table, and multi-node collectives exist only in
the SGLang DeepEP tables for H100/H200: those are what our own scripts must add.

| Device | Existing A-Grade Data | Missing | Collection Priority |
| --- | --- | --- | --- |
| `a100_sxm_80g` | vLLM 0.14.0 (bf16 only), TRT-LLM 1.0.0 (adds int8_wo/int4_wo/int8_sq GEMM), SGLang 0.5.10 (adds int4_wo MoE); NCCL 2.27.3 + custom all-reduce (2/4/8 GPUs) | elementwise; fp8-free by hardware; head_dim 64/256; multi-node collectives; DeepEP | **High**: our lab has A100s. Run `scripts/collect/run_aiconfigurator.sh` with a current vLLM (0.14.0 is old), `collect_elementwise.py`, and `run_nccl_tests.sh` across two nodes |
| `a100_sxm_40g` | none | all | **High** if the lab machines are the 40 GB part: same commands with `DEVICE=a100_sxm_40g` |
| `rtx_4090` | none | all | **High**: our lab has 4090s. `run_aiconfigurator.sh` (bf16/fp8/fp8_block GEMM, attention, MoE) plus PCIe NCCL curves; use `BACKEND=vllm` if TRT-LLM modules refuse the consumer part |
| `h100_sxm`, `h200_sxm` | vLLM 0.24.0, TRT-LLM 1.3.0rc20 (+rc23/rc10), SGLang 0.5.14: gemm (bf16/fp8/fp8_block), attention (bf16, fp8 KV), moe (bf16/fp8/fp8_block/mxfp4_wo), NCCL 2.29.2 + custom, DeepEP dispatch/combine (SGLang tables reach 8 nodes) | elementwise; multi-node NCCL | Medium: no local machines |
| `l40s`, `rtx_pro_6000_server` | vLLM 0.24.0, TRT-LLM 1.3.0rc20, SGLang 0.5.14: gemm, attention, moe, PCIe NCCL | elementwise; multi-node; DeepEP | Low: no local machines |
| `b200_sxm`, `b300_sxm`, `gb200`, `gb300` | vLLM 0.24.0, TRT-LLM 1.3.0rc20 (+rc23), SGLang 0.5.14: gemm (incl. nvfp4), attention, moe (incl. mxfp4), NCCL 2.29.2 (GB200/GB300: 2/4 GPUs only), DeepEP dispatch/combine | elementwise; 8+ GPU NVLink-domain collectives; peak specs partly derived (grade C) | Low: no local machines |
| `h20` | none | all | Medium: specs are third-party (B); analytical parameters borrowed from H100 |
| `a100_pcie_80g`, `h100_pcie`, `l4`, `a30` | none (upstream data directories are empty) | all | Low |
| `groqchip_v1` | none; `analytical` package is grade D | all | **High (paper-grade)**: see Section 4 |
| `intel_arc_pro_b60` | vLLM 0.26.0 (bf16/fp8 GEMM, bf16 attention, small MoE grid), oneCCL 2021.17.2 collectives | elementwise; multi-node | Low |
| `v100_sxm2` | legacy nccl-tests all-reduce (fp32, 2/4/8 GPUs) | compute tables | Low |

Every simulation writes the shapes it could not answer to
`missing_shapes_<qps>.json`; feed that list back into the next collection run.

---

## 2. Per-Table Collection Requirements

Shape grids are based on the defaults from `operator_data.cli manifest`; values in
parentheses indicate the minimum set of points to cover.

### 2.1 `gemm`

**Schema key fields:** `dtype`, `m`, `n`, `k`
**Interpolation axes:** `k` (log), `n` (log), `m` (log)

* **Precision:** One full sweep for each weight precision the device supports.
  fp16/bf16 are mandatory for every device. A100 adds int8_wo and int4_wo.
  RTX 4090 adds fp8, int8_wo, and int4_wo.
* **`m` (token count):** 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384,
  512, 768, 1024, 2048, 4096, 8192, 16384.
  The low-m region must be densely sampled -- all decode latencies fall here.
* **`(n, k)` pairs:** Determined by target models and TP degree. The `manifest`
  command expands six projection roles: QKV, O, gate-up, down, router, and
  lm_head.
  For the LLaMA family (h = 4096/8192, inter = 11008/14336/28672, vocab =
  32000/128256) at TP 1/2/4/8, this yields approximately 60 distinct `(n, k)`
  pairs.
* **Collection method:** `scripts/collect/run_aiconfigurator.sh` with
  `OPS=gemm`. The AIConfigurator collector benchmarks the serving framework's
  own GEMM kernels (TRT-LLM plugins, or vLLM `gptq_marlin`/`awq_marlin`/FP8
  paths with `BACKEND=vllm`) and records the selected kernel in the
  `kernel_source` column, which the importer keeps as `kernel`. Its token grid
  is denser than ours (1-17 one by one, then 32/33, 48/49 ... to catch tensor
  core tile boundaries).

### 2.2 `context_attention` (prefill)

**Schema key fields:** `attn_dtype`, `kv_cache_dtype`, `batch_size`,
`input_seq_len`, `num_heads`, `num_kv_heads`, `head_dim`, `window_size`
**Interpolation axes:** `num_heads` (linear), `batch_size` (log),
`input_seq_len` (log)

* **`input_seq_len`:** 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
  16384.
  **`batch_size`:** 1, 2, 4, 8 (total tokens must not exceed 32K).
* **Head configuration:** `(num_heads, num_kv_heads, head_dim)` set to the
  per-rank local values for target models at TP 1/2/4/8. For example,
  LLaMA-3-70B at TP 8 yields `(8, 1, 128)`.
* **`window_size`:** 0 (causal, no sliding window). Add 4096 when profiling
  models with sliding-window attention.
* **Collection method:** `scripts/collect/run_aiconfigurator.sh` with
  `OPS=attention` (prefill and decode share one collector op). The collector
  uses the framework's attention kernels (TRT-LLM fused MHA / XQA, or vLLM
  FlashAttention / FlashInfer with `BACKEND=vllm`) and covers bf16 and fp8 KV
  caches.

### 2.3 `generation_attention` (decode)

**Schema key fields:** `attn_dtype`, `kv_cache_dtype`, `batch_size`,
`context_len`, `num_heads`, `num_kv_heads`, `head_dim`, `window_size`
**Interpolation axes:** `num_heads` (linear), `batch_size` (log),
`context_len` (log)

* **`context_len`:** 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768.
  For long-context experiments, add 65536 and 131072.
* **`batch_size`:** 1, 2, 4, 8, 16, 32, 64, 128, 256 (up to 512 when KV cache
  capacity allows).
* **Head configuration:** Same as context_attention. **`kv_cache_dtype`:** In
  addition to the weight precision, collect fp8 and int8 KV-cache variants
  when the framework supports them.
* **Collection method:** same collector op as prefill (`OPS=attention`); the
  decode grid spans batch 1-2048 and context 2-131072. Paged-attention kernels
  are what the frameworks run, so no dense-KV baseline is needed.

### 2.4 `moe`

**Schema key fields:** `dtype`, `distribution`, `num_tokens`, `hidden_size`,
`inter_size`, `top_k`, `num_experts`, `tp_size`, `ep_size`
**Interpolation axes:** `num_tokens` (log)

* **Target models:**
  - Mixtral-8x7B: E = 8, k = 2, inter = 14336
  - Qwen3-235B-A22B: E = 128, k = 8, inter = 1536
  - DeepSeek-V3 class: E = 256, k = 8, inter = 2048
* **`num_tokens`:** 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
  8192, 16384.
* **`(tp_size, ep_size)` configurations:** (1,1), (2,1), (4,1), (8,1), (1,8),
  (1,16), (1,32).
* **`distribution`:** `uniform`. To study hot-expert effects, add
  `power_law_1.2` (aligned with AIConfigurator convention).
* **Collection method:** `scripts/collect/run_aiconfigurator.sh` with
  `OPS=moe` (TRT-LLM fused MoE, or vLLM `fused_moe` with `BACKEND=vllm`).
  Upstream labels the routing pattern `power_law_1.01` / `power_law_1.2`; the
  importer keeps the label in the `distribution` key and the backend tries
  `uniform`, `balanced`, `power_law_1.01` in turn. Use `MODEL_PATH=<hf id>` to
  restrict the grid to one model's expert shape.

### 2.5 `elementwise`

**Schema key fields:** `op_name`, `dtype`, `num_tokens`, `hidden_size`
**Interpolation axes:** `hidden_size` (log), `num_tokens` (log)
**Recognized ops:** rmsnorm, layernorm, rope, residual_add, swiglu, gelu,
data_reorder, other_residual

* **`op_name`:** rmsnorm, rope, residual_add, swiglu (for dense FFN layers).
* **`num_tokens`:** 1 through 16384, using the same grid as the GEMM `m` axis.
  **`hidden_size`:** Model hidden dimension `h`; for rope, use
  `q_local + kv_local`; for swiglu, use `inter_local`.
* **Collection method:** `scripts/collect/collect_elementwise.py` (AIConfigurator
  has no elementwise table). `--kernel eager` times plain PyTorch ops;
  `--kernel vllm` times vLLM's fused `rms_norm` / `fused_add_rms_norm` /
  `rotary_embedding` / `silu_and_mul`, which is the production figure. The
  script merges its rows straight into the device package.

### 2.6 `collective`

**Schema key fields:** `dtype`, `operation`, `group_size`, `nodes`,
`message_bytes`
**Interpolation axes:** `message_bytes` (log)
**Recognized operations:** all_reduce, all_gather, reduce_scatter, all_to_all,
send_recv

* **Operations:** all_reduce (tensor parallelism), all_to_all (expert
  parallelism dispatch/combine), all_gather and reduce_scatter (sequence
  parallelism, reserved).
* **`group_size` x `nodes`:** Single-node: 2, 4, 8. Multi-node: 16 (2 nodes),
  32, 64. When available, add 128.
* **`message_bytes`:** 1 KiB to 1 GiB in power-of-2 steps. Precision: half and
  int8.
* **Collection method:** single node: `scripts/collect/run_aiconfigurator.sh`
  (its `collect_comm.sh` step runs nccl-tests for 2/4/8 GPUs plus the
  framework's custom all-reduce). Multi-node: `scripts/collect/run_nccl_tests.sh`
  with `LAUNCHER="mpirun ..."`; record the NIC model, all NCCL environment
  variables, and the `NCCL_DEBUG=INFO` output showing the selected
  algorithm/protocol, which calibrates the topology's `collective_algorithm`.

### 2.7 Step-Level Fixed Overhead (`step_overhead_us` in device YAML)

* Run a real inference framework with batch = 1, seq = 1 for several decode
  steps. Subtract the sum of per-table operator latencies from the total
  step time to obtain the fixed scheduling + kernel-launch overhead.
* On A100 with vLLM, the typical value is on the order of 1--3 ms. This
  replaces the legacy model's `PREFILL_OFFSET_SECONDS = 9 ms` constant.

---

## 3. Recording Requirements

Every CSV or Parquet row must include the following fields beyond the primary
key and `latency_us`:

| Field | Description |
| --- | --- |
| `kernel` | The CUDA kernel name or API call name (e.g., `sm80_xmma_gemm_bf16bf16_bf16f32_f32...`, `flash_fwd_splitkv_kernel`) |
| `source_id` | A structured identifier linking to a `generation_meta.yaml` source entry (e.g., `aiconfigurator:a100_sxm:gemm:trtllm:1.0.0`) |

### `generation_meta.yaml` Required Fields

Each data package directory must contain a `generation_meta.yaml` file with at
minimum the following structure:

```yaml
schema_version: 1
latency_unit: us
dataset_version: <device>-<backend>-<tag>
device_id: <device_id>
backend: <backend>
framework_version: <version>

sources:
  - source_id: <unique_source_id>
    grade: A          # A = measured, B = third-party benchmark, C = derived, D = analytical
    method: measured   # measured | imported | analytical
    reference: <URL or path>
    notes: <free text>
    device: <device_id>
    backend: <framework:version>

calibration: {}

tables:
  gemm:
    rows: <count>
  # ... one entry per table present in the package
```

The `sources` block must record:
- **Device model** (as reported by `nvidia-smi`)
- **Driver and CUDA version**
- **Framework and version** (e.g., vLLM 0.6.x, TRT-LLM 1.0.0)
- **Clock lock state** (whether GPU clocks were locked during profiling)
- **Measurement method** (CUDA event median, iteration count)
- **Date and operator** (collection date and the person who performed it)

---

## 4. Groq Special Case

Direct kernel-level measurement is not possible on the Groq TSP. The following
public data sources are available:

| Data Source | Usage | Grade |
| --- | --- | --- |
| Product Brief v1.5: 750 TOPS INT8, 188 TFLOPS FP16, 230 MB SRAM, 80 TB/s memory bandwidth, 16 chip-to-chip links, PCIe Gen4 | Device catalog specifications | A |
| ISCA 2020 (TSP architecture): 320 lanes, 88 MEM slices, deterministic execution model | Structural assumptions for the analytical model | A |
| ISCA 2022 (RealScale): 4 lanes x 25 Gbps/link, 8 TSPs/node, 9 nodes/rack, 11-link topology | Link class `groq_c2c_25g`, topology `groq_node8_rack72` | A |
| "Accelerating BERT on TSP": batch 1, seq 128, approximately 130 us end-to-end | Layer-level validation: run `bert-base` on a single chip using the `analytical` backend and compare the 12-layer total time | A (paper-grade) |
| GroqCloud public throughput: LLaMA-2-70B approximately 300 tok/s at 576 chips; Mixtral approximately 500--800 tok/s | End-to-end order-of-magnitude check; does not decompose into per-operator times | B |

### Items Requiring Confirmation

- **480 GB/s bandwidth figure:** The current device catalog interprets this as
  16 links x 4 lanes x 30 Gbps x 2 directions.
- **Chip-to-chip hop latency:** The catalog currently assumes 0.5 us per hop.
- **Host-side step overhead:** Whether a host scheduling overhead exists in the
  Groq execution model (the TSP's deterministic execution may eliminate this).

### With GroqRack Access

If GroqRack hardware access becomes available, the priority is to use the
GroqFlow profiler to collect single-operator cycle counts for GEMM and
attention. Conversion to microseconds: `cycles x (1 / 900 MHz)`.

---

## 5. Validation Data (B-Grade, for End-to-End Calibration)

B-grade data is not used to populate operator tables directly, but serves as
ground truth for validating that the simulator's end-to-end predictions are in
the right order of magnitude.

* **vLLM / TRT-LLM official benchmarks:** LLaMA-3-8B and LLaMA-3-70B on
  A100/H100. Metrics: TTFT (time to first token), TPOT (time per output token),
  and throughput, broken down by batch size and input/output sequence length.
* **Internal A100 and RTX 4090 measurements:** Run `vllm bench latency` on our
  local cluster, then simulate the same configuration with TokenSim and compare
  prefill/decode median latencies.
* **Legacy repository `measure_vs_roofline.csv`:** Contains M6 model V100
  measured breakdowns. These can be converted into validation checkpoints for
  the V100 compute tables.
