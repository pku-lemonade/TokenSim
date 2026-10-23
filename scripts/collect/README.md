# Operator Latency Collection (`scripts/collect/`)

These scripts measure operator latencies on the target GPU node and save data packages under `data/operator_data/<device_id>/<backend>/` for the `operator_table` backend.
Grade A identifies measurements on the target hardware and software stack.
NVIDIA AIConfigurator measures the inference framework's kernels for matrix multiplication (GEMM), attention, mixture-of-experts (MoE) computation and communication within one node.
TokenSim's additional scripts measure elementwise kernels and communication between nodes.
A collective is a communication operation involving a group of GPU workers.

## Operator timeline collection

`collect_operator_trace.py` records a short vLLM prefill/decode workload with PyTorch Profiler and writes Chrome Trace JSON files plus a collection manifest.
It uses a separate annotated structure pass and GPU-only timing passes through a vLLM worker extension.
The structure pass identifies layers, embedding, the LM head and token selection.
All added hooks and method wrappers are removed before timing.
The extension preserves model return values and adds no kernel replacements.
The target vLLM installation must support `profiler_config`, `worker_extension_cls` and `collective_rpc`.

```bash
CUDA_VISIBLE_DEVICES=0 \
  python -m scripts.collect.collect_operator_trace \
    --model /path/to/local/Qwen3-8B \
    --output outputs/profiler/qwen3-8b \
    --tensor-parallel-size 1 \
    --batch-size 1 \
    --input-tokens 32 \
    --output-tokens 3
```

Use the authorized inference environment's Python executable for collection.
For the agreed `PKUa100` campaign, the source installation is `/home/youwei/bzh/project/TokenCake/RFC/vllm-rfc` and the independent working copy is `/home/youwei/bzh/project/TokenSim/vllm-rfc`.
Keep the source installation unchanged, including editable-install mappings and generated caches.
Store remote data under `/home/youwei/bzh/project/TokenSim/profiler`.
The [development specification](../../docs/operator-tracing-design.md#10-remote-environment-and-collection) describes the copy and verification procedure.

The collector warms up the same fixed workload before starting capture.
It uses seeded token identifiers, greedy sampling and fixed output lengths.
`--prompt-pattern varied` is the default; `--prompt-pattern repeated` reproduces the repeated-token workload.
`--seed` controls the token identifiers and framework seed.
For batches larger than one request, the collector pauses the scheduler, enqueues the complete burst and resumes it before waiting for results.
This path requires vLLM's synchronous scheduler utility calls, `enqueue` and `wait_for_completion` APIs.
The observed forward count must match the fixed-batch contract.
Output-token changes between passes fail collection and are recorded in the manifest.
Eager execution is the default; `--cuda-graph` selects graph capture when supported by the target installation.
`--record-shapes` adds shape recording to the structure pass only.
The timing passes collect CUDA activities without CPU operator tracing, shape recording or added module annotations.
Kernel category, name, available launch dimensions, device inventory and per-stream order must agree with the structure pass before its ownership can be reused.
Changed inventories fail explicitly; operation ownership is never chosen by duration similarity.
The structure trace and timing traces are pinned by SHA-256.
The importer requires retained structure files and rejects changed artifacts.

`--repeats` defaults to three timing captures and three unprofiled baseline invocations on each side of capture.
The manifest selects the actual upper-median timing sample and retains every other sample.
Profiler startup, shutdown and export are excluded from `generate` wall-time measurements.
`timing_quality` compares the selected sample with the unprofiled baseline and marks differences outside 10% as `perturbed`.
This check reports collection quality; it does not correct measured durations.
Manifests also record software versions, source hashes, model configuration, worker clock identities, annotation coverage, capture policy and completion status.
Inspect the run log and manifest before using a trace as validation evidence.
GPU-only PyTorch profiling can still perturb execution, and the original timing traces retain that effect.
Local `outputs/` data is already ignored by Git; preserve raw traces and logs there and commit only small relevant fixtures.

Use the [offline comparison tool](../../docs/operator-latency-model.md#pytorch-profiler-comparison) to generate a combined timeline and difference report.
Pass `--collection outputs/profiler/qwen3-8b/manifest.json` to select the manifest's timing sample automatically.
Do not combine the structure pass or independent timing replicates into one comparison input.

| Script | Tables produced | Requirements |
| --- | --- | --- |
| `run_aiconfigurator.sh` | `gemm`, `context_attention`, `generation_attention`, `moe`, single-node `collective` (NCCL plus the framework's custom all-reduce), and any other table of the requested collector ops | target framework, AIConfigurator source (cloned by the script), nccl-tests, mpirun |
| `collect_elementwise.py` | `elementwise` (rmsnorm / rope / residual_add / swiglu) | PyTorch with CUDA; vLLM for `--kernel vllm` |
| `run_nccl_tests.sh` | multi-node or fabric-specific `collective` | nccl-tests; mpirun or srun for several nodes |
| `common.py` | shared timing and environment helpers | PyTorch |

Install TokenSim in the inference framework's environment with `python -m pip install -e .` from the repository root.
Run the collection scripts from that directory.

The scripts write data packages in the following ways.

- `run_aiconfigurator.sh` imports with `import-aiconfigurator` and replaces the target package's tables.
  `OperatorDataPackage.write()` removes table files absent from the new package.
- `collect_elementwise.py` and `run_nccl_tests.sh` merge rows into the existing package with `merge_packages`.
  Newly collected rows replace existing rows with the same key.

Run `run_aiconfigurator.sh` first and the other two afterwards.
To preserve an existing package, import the profiling run into a separate directory and merge it as described in section 4.

---

## 0. Before Collecting

1. Use a node reserved for this profiling run, lock GPU clocks and enable persistence mode.
   Unlocked clocks can vary by 10% or more between boost and throttled states.

   ```bash
   sudo nvidia-smi -pm 1
   sm=$(nvidia-smi -q -i 0 | grep -A4 "Max Clocks" | grep "SM " | grep -o "[0-9]\+" | head -1)
   mem=$(nvidia-smi -q -i 0 | grep -A4 "Max Clocks" | grep "Memory " | grep -o "[0-9]\+" | head -1)
   sudo nvidia-smi -ac "$mem,$sm"
   ```

   Without root access, record the current and maximum GPU compute clocks in the source notes using this command.

   ```bash
   nvidia-smi \
       --query-gpu=clocks.sm,clocks.max.sm \
       --format=csv
   ```

2. Check that `data/devices/<device_id>.yaml` exists for the target device.
   For example, `rtx_4090` and `a100_sxm_40g` have catalog entries and no bundled operator measurements.
   For a new device, add a YAML file as described in `data/devices/README.md`.

3. Prepare an environment with a supported inference framework.
   Collector modules declare supported framework versions in `__compat__`, and upstream `collector/framework_manifest.yaml` specifies container images verified by their digests.
   The bundled data uses the framework versions below.

   | Backend | Version (AIConfigurator `f254959`) | Image |
   | --- | --- | --- |
   | trtllm | 1.3.0rc20 (the A100 data uses 1.0.0) | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20` |
   | vllm | 0.24.0 | `vllm/vllm-openai:v0.24.0` |
   | sglang | 0.5.14 | `lmsysorg/sglang:v0.5.14` |

   When the cluster does not specify `operator_backend`, the simulator prefers `vllm`, followed by `trtllm` and `sglang`.
   Collection scripts default to `BACKEND=trtllm`, so set `BACKEND` to the framework you intend to simulate.

   The container also needs this repository, TokenSim's requirements, and compiled [nccl-tests](https://github.com/NVIDIA/nccl-tests) (`make MPI=1`, with the binary directory on `PATH`).

4. Generate a manifest listing the operator dimensions required by your models and parallel configurations.
   Use it as input to `collect_elementwise.py` and to check measurement coverage after collection.

   ```bash
   python -m TokenSim.operator_data.cli manifest \
       --models llama-3-8b,llama-3-70b,mixtral-8x7b \
       --tp 1,2,4,8 \
       --out tmp/manifests/llama3-mixtral.yaml
   ```

---

## 1. `run_aiconfigurator.sh`: Compute Operators and Single-Node Collectives

```bash
DEVICE=rtx_4090 \
BACKEND=vllm \
    ./scripts/collect/run_aiconfigurator.sh
```

The script performs the following operations.

1. The script clones AIConfigurator into `AIC_DIR`, default `./tmp/aiconfigurator`, and checks out `AIC_COMMIT`, default `f254959eb89e2f206b8f9a77051644d7c1cbdb89`.
   This is the revision recorded in `data/operator_data/LICENSES.md`, so collection dimensions and column names match the bundled data.
2. It checks that the target framework can be imported and records GPU clocks with `nvidia-smi` in `<RUN_DIR>/env.txt`.
3. It runs `collector/collect.py` with `--backend` set by `BACKEND`, `--ops` set by `OPS` and `--resume` enabled.
   Measurements are written inside `RUN_DIR`, which defaults to `./tmp/collect/<DEVICE>/<BACKEND>`.
   The files are `gemm_perf.parquet`, `context_attention_perf.parquet`, `generation_attention_perf.parquet` and `moe_perf.parquet`.
   `--resume` allows an interrupted run to collect only the missing cases.
4. Unless `SKIP_COMM=1`, it runs `collector/network/collect_comm.sh` with `--all_reduce_backend` set by `BACKEND`.
   This measures nccl-tests all_gather, alltoall, reduce_scatter and all_reduce on 2, 4 and 8 local GPUs, limited by the available device count.
   It uses half and int8 data types with message sizes from 512 B to 512 MB and also measures the framework's custom all-reduce.
   Results are saved in `nccl_perf.txt` and `custom_allreduce_perf.txt`.
   A failure here is reported and the compute tables are still imported.
5. It imports `RUN_DIR` into `PACKAGE` with `python -m TokenSim.operator_data.cli import-aiconfigurator`.
   Import arguments map `RUN_DIR` to `--system-dir`, `DEVICE` to `--device`, `BACKEND` to `--backend`, `AIC_COMMIT` to `--upstream-commit` and `PACKAGE` to `--out`.
   The importer reads the run directory, including `*_perf.txt` CSV files, converts milliseconds to microseconds and converts data type names to TokenSim's standard names.
   It keeps the fastest kernel for each set of dimensions and records individual operator sources as `aiconfigurator-collector:<device>:<family>:<backend>:<version>` with grade A and method `measured`.
   It validates the imported package.

The following environment variables configure the script.

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

Pass `--import-only` as the first argument to skip measuring and only import an existing `RUN_DIR`.

The recorded collector defines its base measurement values in `collector/cases/base_ops/*.yaml`.
GEMM measurements cover every token count from 1-17, pairs such as 32/33 and 48/49 through 4352/4353, and 8192, 16384 and 32768.
Closely spaced token counts capture latency changes when a kernel needs another tile.
The `n` and `k` dimensions use 22 feature sizes from 32 to 65536.
Decode attention covers batch sizes 1-2048 and context lengths 2-131072, and MoE covers token counts 1-16384.
The collector times most kernels inside a CUDA graph and reports the mean per call.

If TensorRT-LLM reports an RTX 4090 or another consumer GPU as unsupported, use `BACKEND=vllm`.
Multi-GPU 4090 servers have no NVLink, so `collect_comm.sh` measures PCIe curves, which is what the `pcie_8x_4090` topology needs.

---

## 2. `collect_elementwise.py`: Elementwise Operators

```bash
python scripts/collect/collect_elementwise.py \
    --manifest tmp/manifests/llama3-mixtral.yaml \
    --device rtx_4090 \
    --package data/operator_data/rtx_4090/vllm \
    --backend vllm \
    --kernel vllm \
    --operator "$USER"
```

* The script reads `op_name, dtype, num_tokens, hidden_size` from the manifest's `elementwise` entries.
  The simulator uses activation precision for these queries, and the manifest uses model precision, so check that they agree for quantized checkpoints.
* `--kernel eager` times PyTorch implementations as an upper-bound reference.
  `--kernel vllm` times vLLM's `rms_norm`, `rotary_embedding` and `silu_and_mul` kernels.
  This mode skips `residual_add` because vLLM's `fused_add_rms_norm` also includes normalization, whose latency is estimated separately.
* It warms each kernel with 5 calls and reports the median CUDA event time over `--iters` calls, default 20.
* It merges rows into the `--package` directory, creates the directory if absent and replaces existing rows with matching keys.
  It adds the source `measured:<device>:elementwise:<torch-eager|vllm-custom-ops>`.
  Its notes record the device name, driver and clocks, torch and CUDA versions, the iteration count, the date and `--operator`.
  `--backend` (default `trtllm`) labels a newly created package and must match the directory name.

---

## 3. `run_nccl_tests.sh`: Multi-Node Collectives

```bash
# One node, 8 GPUs (overlaps with collect_comm.sh; useful as a cross-check)
DEVICE=rtx_4090 \
BACKEND=vllm \
DTYPE=bfloat16 \
NCCL_TESTS=/opt/nccl-tests/build \
    ./scripts/collect/run_nccl_tests.sh 8 1
# Two nodes, 16 GPUs, one process per GPU
DEVICE=h100_sxm \
BACKEND=vllm \
DTYPE=bfloat16 \
NCCL_TESTS=/opt/nccl-tests/build \
    LAUNCHER="mpirun -np 16 -H nodeA:8,nodeB:8 -x NCCL_DEBUG=INFO" \
    ./scripts/collect/run_nccl_tests.sh 8 2
```

The positional arguments are `<gpus per node> <nodes>`.
The script measures all_reduce, all_gather, reduce_scatter and alltoall for message sizes from 1 KiB to 1 GiB, doubling the size at each point.
Each point uses 20 warmup iterations and 50 timed iterations.
A single-node run passes `-g <gpus>` to the binaries, and a run using `LAUNCHER` assigns one GPU per process.
Logs are saved under `OUT`, default `./tmp/nccl/<DEVICE>`, and imported into `PACKAGE` with `operator_data.cli import-nccl`.
`PACKAGE` defaults to `./data/operator_data/<DEVICE>/<BACKEND>`, and `BACKEND` defaults to `trtllm`.
Rows are keyed by `(dtype, operation, group_size, nodes, message_bytes)`.

TokenSim queries collectives with the model's activation dtype, so set `DTYPE` accordingly (`bfloat16` for bf16 models; the default `half` produces fp16 rows).
For runs across nodes, record the algorithm and protocol reported by `NCCL_DEBUG=INFO`, such as ring/tree and LL/Simple.
Use these records to calibrate `collective_algorithm` in the topology YAML.

---

## 4. After Collecting

```bash
python -m TokenSim.operator_data.cli validate data/operator_data/a100_sxm_80g/trtllm
python -m TokenSim.operator_data.cli coverage \
    --device a100_sxm_80g \
    --backend trtllm \
    --models llama-3-8b \
    --tp 1,2,4,8 \
    --out tmp/coverage-a100.yaml
python -m TokenSim.operator_data.cli calibrate \
    --device a100_sxm_80g \
    --backend trtllm \
    --out tmp/calib-a100.yaml
```

* `coverage` reports whether each manifest key has an exact match, can be interpolated or extrapolated, or is missing.
  Missing keys identify the operator dimensions to measure in another run.
* `calibrate` fits the efficiency parameters of the device YAML's `analytical:` block against the measured tables and reports holdout errors.
  When copying the values into the YAML, mark them `grade: C`.
* `step_overhead_us` (fixed scheduling and launch cost) is not part of any table: run decode steps at batch 1 in the real framework, subtract the operator times composed from the tables, and enter the result in the device YAML.

To add measurements to an existing package, use `PACKAGE=tmp/operator_data/<DEVICE>/<BACKEND>` with `run_aiconfigurator.sh` to import into a separate directory.
Merge the packages with the following commands.

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

`prefer="extra"` lets the new rows win on key clashes; `prefer="base"` keeps the existing ones.

---

## 5. Recording Requirements

Every source in `generation_meta.yaml` should record the device model reported by `nvidia-smi`, driver and CUDA versions, framework version, clock lock state, measurement method, iteration count, date and the person who collected the data.
The scripts automatically record the information listed below.

| Script | Recorded in the source entry |
| --- | --- |
| `run_aiconfigurator.sh` | Run directory, AIConfigurator commit, framework and collection version per family |
| `collect_elementwise.py` | Device name, driver and clocks, torch and CUDA versions, iterations, date, operator |
| `run_nccl_tests.sh` | nccl-tests command, group and node count, NCCL version, `NCCL_DEBUG`, launcher |

Add missing details to each source's `notes`, including the driver, clock state and person collecting the data for AIConfigurator runs.
Run `validate` after completing these records.

Upstream tables include `framework`, `version`, `device` and `kernel_source` columns.
The importer preserves `kernel_source` in `kernel` to identify which implementation was measured for each set of dimensions.
