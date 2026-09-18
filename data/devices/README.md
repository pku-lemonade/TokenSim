# Device catalog (`data/devices/`)

One YAML per accelerator. Every number carries a `source_id` (declared in the
file's `sources:` block) and an evidence grade: **A** official spec or
first-party measurement, **B** public third-party benchmark, **C** calibrated
or derived estimate, **D** project assumption. See
[`TokenSim/hardware/README.md`](../../TokenSim/hardware/README.md) for the
field definitions and
[`docs/operator-latency-model.md`](../../docs/operator-latency-model.md) for how
the values are used.

## Devices and available operator data

Operator tables come from NVIDIA AIConfigurator's measured kernel database
(commit `f254959eb89e2f206b8f9a77051644d7c1cbdb89`, Apache-2.0, see
[`../operator_data/LICENSES.md`](../operator_data/LICENSES.md)). Every system
that has upstream data is imported for **all three serving stacks**: `vllm`
(project default when a cluster names no `operator_backend`), `trtllm` and
`sglang`. Select one per worker group with `"operator_backend": "<name>"` or
globally with `--operator_backend`.

| device_id | Device | Family | Spec grade | AIConfigurator system | Imported backends (framework versions) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `a100_sxm_80g` | NVIDIA A100-SXM4-80GB | nvidia_gpu | A | `a100_sxm` | vllm 0.14.0, trtllm 1.0.0, sglang 0.5.10; NCCL 2.27.3 | Ampere: no fp8; trtllm adds int8_wo/int4_wo/int8_sq GEMM, sglang int4_wo MoE |
| `a100_sxm_40g` | NVIDIA A100-SXM4-40GB | nvidia_gpu | A | — | none | bandwidth 1.555 TB/s differs from the 80 GB part |
| `a100_pcie_80g` | NVIDIA A100-PCIE-80GB | nvidia_gpu | A | `a100_pcie` | none — upstream data directory is empty | PCIe peers, NVLink bridge pairs only |
| `a30` | NVIDIA A30 | nvidia_gpu | A | `a30` | none — upstream data directory is empty | 24 GB HBM2 |
| `v100_sxm2` | NVIDIA V100-SXM2-32GB | nvidia_gpu | A | — | `nccl` collective only (legacy nccl-tests fp32, 2/4/8 GPUs) | Volta, not in AIConfigurator |
| `h100_sxm` | NVIDIA H100 80GB HBM3 | nvidia_gpu | A | `h100_sxm` | vllm 0.24.0, trtllm 1.3.0rc20 (+rc23/rc10 backfill), sglang 0.5.14 (+0.5.6.post2); NCCL 2.29.2; DeepEP `ep_all2all` from all three | vllm attention FA3/FA4; sglang adds int8_wo GEMM; sglang DeepEP tables cover 1-8 nodes |
| `h100_pcie` | NVIDIA H100 PCIe 80GB | nvidia_gpu | A | `h100_pcie` | none — upstream data directory is empty | 756 TFLOPS FP16 dense, PCIe peers |
| `h200_sxm` | NVIDIA H200 141GB HBM3e | nvidia_gpu | A | `h200_sxm` | same as H100 | |
| `h20` | NVIDIA H20 96GB | nvidia_gpu | **B** (third-party specs) | — | none | analytical parameters applied from `h100_sxm` |
| `l40s` | NVIDIA L40S | nvidia_gpu | A | `l40s` | vllm 0.24.0, trtllm 1.3.0rc20 (+1.0.0), sglang 0.5.14; PCIe NCCL | Ada, fp8 supported, no NVLink |
| `l4` | NVIDIA L4 | nvidia_gpu | A | `l4` | none — upstream data directory is empty | 72 W inference card |
| `rtx_4090` | NVIDIA GeForce RTX 4090 | nvidia_gpu | A (Ada whitepaper) | — | none — collect locally with `scripts/collect/` | analytical parameters applied from `l40s` (same AD102) |
| `rtx_pro_6000_server` | NVIDIA RTX PRO 6000 Blackwell Server Edition | nvidia_gpu | **C** (tensor peaks scaled to observed clock) | `rtx_pro_6000_server` | vllm 0.24.0, trtllm 1.3.0rc20, sglang 0.5.14; PCIe NCCL | NVIDIA page lists 1.6 TB/s memory bandwidth vs 1.792 upstream; verify |
| `b200_sxm` | NVIDIA B200 SXM (HGX B200) | nvidia_gpu | A compute / C memory | `b200_sxm` | vllm 0.24.0, trtllm 1.3.0rc20 (+rc23), sglang 0.5.14; NCCL 2.29.2; DeepEP `ep_all2all` | nvfp4 and mxfp4 GEMM/MoE |
| `b300_sxm` | NVIDIA B300 SXM (HGX B300) | nvidia_gpu | A compute / C memory+fp4 | `b300_sxm` | same as B200 | INT8 tensor throughput reduced on Blackwell Ultra |
| `gb200` | NVIDIA GB200 (NVL72) | nvidia_gpu | **C** (per-GPU derived) | `gb200` | same as B200; collectives 2/4 GPUs only (4 GPUs per tray, `--gpus-per-node 4`) | topology `gb200_nvl72` |
| `gb300` | NVIDIA GB300 (NVL72, Blackwell Ultra) | nvidia_gpu | **C** (per-GPU derived) | `gb300` | same as B200; collectives 2/4 GPUs only | topology `gb300_nvl72` |
| `groqchip_v1` | Groq GroqChip v1 (TSP / LPU) | groq_tsp | A (product brief, ISCA papers) | — | `analytical` only (grade D formulas) | no measured kernels; see docs/data-collection-checklist.md §4 |
| `intel_arc_pro_b60` | Intel Arc Pro B60 (XPU) | generic | A memory / C compute | `b60` | vllm 0.26.0 (+0.20.0); oneCCL 2021.17.2 collectives | only vLLM XPU exists upstream; analytical family `generic` |

"none" means the `operator_table` backend falls back to the analytical model
for that device (reported as `match_type = analytical` in results). Priority for
local measurement is documented in
[`docs/data-collection-checklist.md`](../../docs/data-collection-checklist.md).

### Why versions differ between systems

Each `(system, framework)` pair upstream is pinned to the framework version that
was current when NVIDIA last ran a full collection campaign on that hardware
(`aic-core/src/aiconfigurator_core/systems/query_versions.yaml`). Hopper and
Blackwell systems are re-collected regularly (TensorRT-LLM 1.3.0rc20, vLLM
0.24.0, SGLang 0.5.14); the A100 was frozen at TensorRT-LLM 1.0.0 / vLLM 0.14.0
/ SGLang 0.5.10 because newer releases stopped adding Ampere kernels. Newer
directories such as `1.3.0rc23` hold only the shapes that changed (MXFP4 MoE)
and rely on the previous full collection for everything else; the importer
merges them the same way (newest version wins per key, older versions backfill).

### Why the same shape differs between backends

The three stacks pick different kernels for the same GEMM or attention shape
(cuBLAS/CUTLASS heuristics vs FlashInfer vs Triton, FlashAttention-2/3/4 vs
TRT-LLM fused MHA/XQA), support different quantization formats on the same GPU,
and fuse operators differently. The `kernel` column of every imported row keeps
the upstream `kernel_source` so the difference can be traced.

## Analytical defaults shared with AIConfigurator (grade C)

Every NVIDIA GPU file overrides the family defaults with the empirical
corrections NVIDIA recorded in its `systems/<system>.yaml` files while
validating AIConfigurator against TensorRT-LLM deployments:

| Field | Value | Upstream key |
| --- | --- | --- |
| `analytical.memory_efficiency` | 0.8 | `mem_bw_empirical_scaling_factor` |
| `analytical.kernel_launch_us` | 3.0 | `mem_empirical_constant_latency` (3 µs) |
| `analytical.collective_launch_us` | 10.0 | `node.p2p_latency` (10 µs) |
| `memory.reserved_bytes` | 4,169,138,176 (392 MB NCCL buffers + 3.5 GB runtime reserve) | `misc.nccl_mem[8] + misc.other_mem` |

The Intel B60 file uses the upstream B60 values instead (memory efficiency 0.5,
2 µs p2p latency, 500 MiB reserve). Devices without an upstream system file
(H20, RTX 4090, V100) borrow the values of the closest architecture and say so
in their `sources` notes. `CacheConfig` subtracts `reserved_bytes` before
computing KV-cache blocks; `operator_data.cli calibrate` can replace the
analytical values with device-specific fits.

## Adding a device

1. Copy the closest YAML, change `device_id`, `display_name`, `aliases`.
2. Fill `peak_compute` per dtype (dense figures; halve NVIDIA sparsity numbers),
   `memory`, optional `on_chip_memory`, `interconnect` (link ids from
   `data/topologies/links.yaml`), `power`.
3. Declare every `source_id` in `sources:` with a grade and a reference.
4. Import or collect operator data into `data/operator_data/<device_id>/<backend>/`
   (see `scripts/collect/README.md`), or leave it to the analytical fallback.
5. Run `python -m unittest tests.test_hardware_catalog` — the catalog test loads
   every file and checks aliases and link references.
