# Operator Latency Collection (`scripts/collect/`)

The scripts in this directory measure grade-A operator latencies on a
**target GPU node** and write them as operator packages under
`data/operator_data/<device_id>/<backend>/`, the format the `operator_table`
latency backend reads. GEMM, attention, MoE and single-node collectives are
collected with NVIDIA AIConfigurator's collector, which times the kernels of
TensorRT-LLM, vLLM or SGLang themselves rather than PyTorch eager operators.
The TokenSim scripts cover what that collector does not: elementwise kernels
and multi-node collectives.

| Script | Tables produced | Requirements |
| --- | --- | --- |
| `run_aiconfigurator.sh` | `gemm`, `context_attention`, `generation_attention`, `moe`, single-node `collective` (NCCL plus the framework's custom all-reduce), and any other table of the requested collector ops | target framework, AIConfigurator source (cloned by the script), nccl-tests, mpirun |
| `collect_elementwise.py` | `elementwise` (rmsnorm / rope / residual_add / swiglu) | PyTorch with CUDA; vLLM for `--kernel vllm` |
| `run_nccl_tests.sh` | multi-node or fabric-specific `collective` | nccl-tests; mpirun or srun for several nodes |
| `common.py` | shared timing and environment helpers | PyTorch |

Run the scripts from the repository root, with TokenSim's Python requirements
(`requirements.txt`) installed next to the framework.

How the scripts write packages:

- `run_aiconfigurator.sh` imports with `import-aiconfigurator`, which
  **replaces** the tables of the target package (`OperatorDataPackage.write()`
  removes table files the new package does not contain).
- `collect_elementwise.py` and `run_nccl_tests.sh` **merge** their rows into
  the existing package (`merge_packages`; new rows win on key clashes).

Run `run_aiconfigurator.sh` first and the other two afterwards. Do not import
a collector run directly into a bundled package you want to keep; import it
into its own directory and merge it (section 4).

---

## 0. Before Collecting

1. **Use an exclusive node, lock the clocks and enable persistence mode.**
   Unlocked clocks drift between boost and throttled states by 10% or more.

   ```bash
   sudo nvidia-smi -pm 1
   sm=$(nvidia-smi -q -i 0 | grep -A4 "Max Clocks" | grep "SM " | grep -o "[0-9]\+" | head -1)
   mem=$(nvidia-smi -q -i 0 | grep -A4 "Max Clocks" | grep "Memory " | grep -o "[0-9]\+" | head -1)
   sudo nvidia-smi -ac "$mem,$sm"
   ```

   Without root access, record the output of
   `nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv` in the
   source notes.

2. **Make sure the device is in the catalog.** `data/devices/<device_id>.yaml`
   must exist, for example `rtx_4090` or `a100_sxm_40g` (both have no bundled
   operator data). For a new device, add a YAML file as described in
   `data/devices/README.md`.

3. **Prepare the framework environment.** Collector modules declare the
   framework versions they support (`__compat__`), and the upstream
   `collector/framework_manifest.yaml` pins digest-verified container images.
   The bundled data was collected with:

   | Backend | Version (AIConfigurator `f254959`) | Image |
   | --- | --- | --- |
   | trtllm | 1.3.0rc20 (the A100 data uses 1.0.0) | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20` |
   | vllm | 0.24.0 | `vllm/vllm-openai:v0.24.0` |
   | sglang | 0.5.14 | `lmsysorg/sglang:v0.5.14` |

   The simulator uses the `vllm` package when a cluster names no
   `operator_backend` (then `trtllm`, then `sglang`), while the collection
   scripts default to `BACKEND=trtllm`; set `BACKEND` to the serving stack you
   want to simulate.

   The container also needs this repository, TokenSim's requirements, and
   compiled [nccl-tests](https://github.com/NVIDIA/nccl-tests) (`make MPI=1`,
   with the binary directory on `PATH`).

4. **Write a shape manifest** (input for `collect_elementwise.py`, and for
   checking coverage afterwards):

   ```bash
   python -m TokenSim.operator_data.cli manifest \
       --models llama-3-8b,llama-3-70b,mixtral-8x7b \
       --tp 1,2,4,8 --out tmp/manifests/llama3-mixtral.yaml
   ```

---

## 1. `run_aiconfigurator.sh`: Compute Operators and Single-Node Collectives

```bash
DEVICE=rtx_4090 BACKEND=vllm ./scripts/collect/run_aiconfigurator.sh
```

What the script does:

1. Clones AIConfigurator into `AIC_DIR` (default `./tmp/aiconfigurator`) and
   checks out `AIC_COMMIT` (default `f254959eb89e2f206b8f9a77051644d7c1cbdb89`,
   the commit recorded in `data/operator_data/LICENSES.md`, so shape grids and
   column names match the bundled data).
2. Checks that the target framework is importable and prints the clock state
   with `nvidia-smi` (also saved as `<RUN_DIR>/env.txt`).
3. Runs `collector/collect.py --backend <BACKEND> --resume --ops <OPS>` inside
   `RUN_DIR` (default `./tmp/collect/<DEVICE>/<BACKEND>`). The collector writes
   `gemm_perf.parquet`, `context_attention_perf.parquet`,
   `generation_attention_perf.parquet` and `moe_perf.parquet` into that
   directory; `--resume` makes a rerun after an interruption collect only the
   missing cases.
4. Unless `SKIP_COMM=1`, runs `collector/network/collect_comm.sh
   --all_reduce_backend <BACKEND>`: nccl-tests all_gather / alltoall /
   reduce_scatter / all_reduce on 2, 4 and 8 local GPUs, as far as the node
   has them (half and int8, 512 B to 512 MB), plus the framework's custom
   all-reduce, written to
   `nccl_perf.txt` and `custom_allreduce_perf.txt`. A failure here is reported
   and the compute tables are still imported.
5. Runs `python -m TokenSim.operator_data.cli import-aiconfigurator
   --system-dir <RUN_DIR> --device <DEVICE> --backend <BACKEND>
   --upstream-commit <AIC_COMMIT> --out <PACKAGE>`. The importer recognizes
   this flat directory (and reads `*_perf.txt` CSV staging files directly),
   converts milliseconds to microseconds, normalizes dtype names, keeps the
   fastest kernel per shape, and records sources as
   `aiconfigurator-collector:<device>:<family>:<backend>:<version>` with grade
   A and method `measured`. It then validates the package.

Variables:

| Variable | Default | Effect |
| --- | --- | --- |
| `DEVICE` | required | TokenSim `device_id` |
| `BACKEND` | `trtllm` | `trtllm`, `vllm` or `sglang`; the package directory is `<device>/<backend>` |
| `OPS` | `gemm attention_context attention_generation moe` | Collector ops, space-separated (`gemm`, `attention_context`, `attention_generation`, `moe`, module ops such as `mhc_module`; see the collector registries). The MoE grid is the largest |
| `MODEL_PATH` | unset | Hugging Face model id: collect only the shapes of that model (AIConfigurator's model-centric "healing" run) |
| `GPU_TYPE` | local SM | AIConfigurator system name (e.g. `a100_sxm`) for hardware capability floors; leave it empty for GPUs without an upstream system, such as the RTX 4090 |
| `SKIP_COMM` | 0 | 1 skips `collect_comm.sh` (no nccl-tests or mpirun) |
| `AIC_DIR`, `AIC_COMMIT` | see above | AIConfigurator checkout and commit |
| `RUN_DIR` | `./tmp/collect/<DEVICE>/<BACKEND>` | Collector output directory |
| `PACKAGE` | `./data/operator_data/<DEVICE>/<BACKEND>` | Import target (replaced) |

Pass `--import-only` as the first argument to skip measuring and only import
an existing `RUN_DIR`.

The base grids of the pinned collector (`collector/cases/base_ops/*.yaml`):
GEMM token counts 1-17 one by one, then pairs such as 32/33 and 48/49 up to
4352/4353 to catch tile boundaries, then 8192, 16384 and 32768, with 22
feature sizes from 32 to 65536 for `n` and `k`; decode attention batches
1-2048 and contexts 2-131072; MoE token counts 1-16384. The collector times
most kernels inside a CUDA graph and reports the mean per call.

**RTX 4090 and other consumer GPUs**: if TensorRT-LLM modules report the GPU
as unsupported, use `BACKEND=vllm`. Multi-GPU 4090 servers have no NVLink, so
`collect_comm.sh` measures PCIe curves, which is what the `pcie_8x_4090`
topology needs.

---

## 2. `collect_elementwise.py`: Elementwise Operators

```bash
python scripts/collect/collect_elementwise.py \
    --manifest tmp/manifests/llama3-mixtral.yaml --device rtx_4090 \
    --package data/operator_data/rtx_4090/vllm --backend vllm \
    --kernel vllm --operator "$USER"
```

* Takes the shapes of the manifest's `elementwise` table (`op_name, dtype,
  num_tokens, hidden_size`). The simulator queries elementwise rows with the
  model's activation dtype; the manifest uses the model dtype, so check that
  the two agree for quantized checkpoints.
* `--kernel eager` times PyTorch implementations (an upper-bound reference);
  `--kernel vllm` times vLLM's `rms_norm` / `rotary_embedding` /
  `silu_and_mul` kernels. With `--kernel vllm`, `residual_add` is skipped:
  vLLM's `fused_add_rms_norm` includes an RMSNorm that is priced separately, so
  it cannot stand in for a primitive residual-add row.
* Times each kernel with CUDA events: 5 warm-up calls, then the median of
  `--iters` calls (default 20).
* Merges into the `--package` directory (created if missing; existing rows
  with the same key are replaced) and adds the source
  `measured:<device>:elementwise:<torch-eager|vllm-custom-ops>`. Its notes
  record the device name, driver and clocks, torch and CUDA versions, the
  iteration count, the date and `--operator`. `--backend` (default `trtllm`)
  labels a newly created package and must match the directory name.

---

## 3. `run_nccl_tests.sh`: Multi-Node Collectives

```bash
# One node, 8 GPUs (overlaps with collect_comm.sh; useful as a cross-check)
DEVICE=rtx_4090 BACKEND=vllm DTYPE=bfloat16 NCCL_TESTS=/opt/nccl-tests/build \
    ./scripts/collect/run_nccl_tests.sh 8 1
# Two nodes, 16 GPUs, one process per GPU
DEVICE=h100_sxm BACKEND=vllm DTYPE=bfloat16 NCCL_TESTS=/opt/nccl-tests/build \
    LAUNCHER="mpirun -np 16 -H nodeA:8,nodeB:8 -x NCCL_DEBUG=INFO" \
    ./scripts/collect/run_nccl_tests.sh 8 2
```

Arguments: `<gpus per node> <nodes>`. The script sweeps all_reduce,
all_gather, reduce_scatter and alltoall from 1 KiB to 1 GiB (factor 2, 20
warm-up and 50 timed iterations). A single node runs the binaries with `-g
<gpus>`; with `LAUNCHER` set, every process drives one GPU. Each log is saved
under `OUT` (default `./tmp/nccl/<DEVICE>`) and imported with
`operator_data.cli import-nccl` into `PACKAGE` (default
`./data/operator_data/<DEVICE>/<BACKEND>`, `BACKEND` default `trtllm`), keyed
by `(dtype, operation, group_size, nodes, message_bytes)`.

TokenSim queries collectives with the model's activation dtype, so set `DTYPE`
accordingly (`bfloat16` for bf16 models; the default `half` produces fp16
rows). For multi-node runs, keep the algorithm and protocol that
`NCCL_DEBUG=INFO` reports (ring/tree, LL/Simple); they calibrate the
`collective_algorithm` of the topology YAML.

---

## 4. After Collecting

```bash
python -m TokenSim.operator_data.cli validate data/operator_data/a100_sxm_80g/trtllm
python -m TokenSim.operator_data.cli coverage --device a100_sxm_80g --backend trtllm \
    --models llama-3-8b --tp 1,2,4,8 --out tmp/coverage-a100.yaml
python -m TokenSim.operator_data.cli calibrate --device a100_sxm_80g --backend trtllm \
    --out tmp/calib-a100.yaml
```

* `coverage` reports for every manifest key whether it is answered exactly,
  by interpolation or extrapolation, or missing; the missing keys are the
  shapes to collect next.
* `calibrate` fits the efficiency parameters of the device YAML's
  `analytical:` block against the measured tables and reports holdout errors.
  When copying the values into the YAML, mark them `grade: C`.
* `step_overhead_us` (fixed scheduling and launch cost) is not part of any
  table: run decode steps at batch 1 in the real framework, subtract the
  operator times composed from the tables, and enter the result in the device
  YAML.

To add a collector run to a package that already holds data, import the run
into its own directory (`PACKAGE=tmp/operator_data/<DEVICE>/<BACKEND>` with
`run_aiconfigurator.sh`) and merge it:

```bash
python - <<'EOF'
from TokenSim.operator_data.generator import merge_packages
from TokenSim.operator_data.package import OperatorDataPackage

target = "data/operator_data/h100_sxm/vllm"
local = OperatorDataPackage.load("tmp/operator_data/h100_sxm/vllm")
merge_packages(OperatorDataPackage.load(target), local, prefer="extra").write(target)
EOF
python -m TokenSim.operator_data.cli validate data/operator_data/h100_sxm/vllm
```

`prefer="extra"` lets the new rows win on key clashes; `prefer="base"` keeps
the existing ones.

---

## 5. Recording Requirements

Every source in a package's `generation_meta.yaml` should answer: device model
(`nvidia-smi` name), driver and CUDA version, framework and version, whether
clocks were locked, measurement method and iteration count, date, and who
collected the data. The scripts record part of this automatically:

| Script | Recorded in the source entry |
| --- | --- |
| `run_aiconfigurator.sh` | Run directory, AIConfigurator commit, framework and collection version per family |
| `collect_elementwise.py` | Device name, driver and clocks, torch and CUDA versions, iterations, date, operator |
| `run_nccl_tests.sh` | nccl-tests command, group and node count, NCCL version, `NCCL_DEBUG`, launcher |

Complete the missing items in the `notes` of the sources (for collector runs:
driver, clock state and operator), then run `validate` again.

The upstream collector tables also carry `framework`, `version`, `device` and
`kernel_source` columns; the importer keeps `kernel_source` in the `kernel`
column, which distinguishes results of different kernels for the same shape.
