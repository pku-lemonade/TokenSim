# AIConfigurator data in TokenSim

TokenSim imports measured kernel and communication latencies from [NVIDIA AIConfigurator](https://github.com/ai-dynamo/aiconfigurator/tree/f254959eb89e2f206b8f9a77051644d7c1cbdb89).
The bundled data uses commit `f254959eb89e2f206b8f9a77051644d7c1cbdb89`.
TokenSim runs its own event simulator; AIConfigurator, PyTorch and a GPU are not runtime requirements for consuming the bundled tables.

Each data package in `data/operator_data/<device>/<backend>` contains `generation_meta.yaml` and Parquet tables converted to TokenSim's schema.
Every row has a `source_id` that identifies its source record in the metadata.
That record lists the upstream path, collection version, device, framework, units and operations included in the measurement.
Its evidence grade distinguishes direct measurements, measurements from another device or software stack, derived values and formula-based estimates, as defined in the operator latency model's section 7.
Original millisecond measurements become microseconds.
Source data that already uses microseconds keeps its units.

## Rebuild an imported package

Check out the recorded commit to reproduce the bundled data.
The documented directory format belongs to that revision.

```bash
git clone https://github.com/ai-dynamo/aiconfigurator.git /tmp/aiconfigurator
git -C /tmp/aiconfigurator checkout f254959eb89e2f206b8f9a77051644d7c1cbdb89
python -m TokenSim.operator_data.cli import-aiconfigurator \
  --system-dir /tmp/aiconfigurator/aic-core/src/aiconfigurator_core/systems/data/b200_sxm \
  --device b200_sxm \
  --backend vllm \
  --upstream-commit f254959eb89e2f206b8f9a77051644d7c1cbdb89 \
  --module-source dsv4_megamoe=/tmp/aiconfigurator/aic-core/src/aiconfigurator_core/systems/data/gb200 \
  --out /tmp/b200-vllm
```

For B300 use `b300_sxm` and explicitly select the `gb300` MegaMoE source.
B200's own MegaMoE rows use hidden size 3584, 896 experts and 16 experts per token, and B300 has no MegaMoE measurements of its own.
The example packages use GB200 and GB300 rows because their model dimensions match DeepSeek-V4-Pro.
The explicit `--module-source` option records that the measurements come from another device.
Omit `--module-source` to import only modules available in the selected system.
`--version` limits the collection version for the requested framework.
Communication tables and explicitly selected module sources keep their own collection versions.

Every bundled package except `v100_sxm2/nccl` and `groqchip_v1/analytical` is the output of this command at the recorded commit.
The system directory has the device's name, except `a100_sxm` for `a100_sxm_80g` and `b60` for `intel_arc_pro_b60`.
The B200 and B300 vLLM MegaMoE sources above are the only non-default options.
SGLang's DeepEP measurements for expert-parallel groups spanning multiple nodes record the node count.
To obtain the expert-parallel group size, the importer multiplies that count by `num_gpus_per_node` from the upstream `systems/<system>.yaml`, using 4 for GB200 and GB300 and 8 elsewhere.
Pass `--gpus-per-node` for a collector run directory, which has no such file.

The importer handles matrix multiplication (GEMM), attention, mixture-of-experts (MoE) operations, group communication (collectives) and DeepEP token transfers.
It also imports tables that time complete sparse-attention, residual-mixing and fused expert modules.
Imported rows retain the model identity and dimensions.
All imported module source IDs are registered before the package is validated.

```bash
python -m TokenSim.operator_data.cli validate /tmp/b200-vllm
```

Keep the original `generation_meta.yaml` beside copied tables so that each row's source remains identifiable.
TokenSim checks loaded tables against the row counts declared in the metadata.
Missing or truncated tables raise a validation error.
Device and backend metadata must match the selected package directory.
Registering an additional in-memory package preserves the other packages available on disk.
A directory from your own profiling run can also be imported with `--system-dir`.
Its source records identify that local collection.

Writing a package replaces its recognized tables and removes stale table files.
Use `merge_packages` before writing when updating only a subset.
Loading rejects undeclared performance tables as well as missing or truncated ones.

## Consume data and select a cost scope

```bash
tokensim --operator_backend vllm \
  --latency_backend operator_table \
  --latency_fallback table_first \
  --results_path results/example
```

`table_first` uses measured operator dimensions where available and reports queries estimated by formulas.
`table_only` returns an error when measurements cannot answer a query, and `analytical_only` uses formulas for every query.
For queries outside the measured range, the default extrapolation scales the nearest boundary measurement by the analytical model's predicted change in work.
You can explicitly select a fitted power-law extrapolation policy.
If its fitted growth is faster than linear, the lookup uses analytical extrapolation.

A complete module table can include projections or communication whose latencies are estimated separately with individual operator tables.
The model's `operator_modules` configuration selects the source table and fields that must match exactly, such as kernel name and precision.
The latency calculation skips individual operations already included in the selected module's measured time.
TokenSim checks that the model's data layout and dimensions satisfy the measured kernel's requirements.

Full-model simulations combine these imported operator and module tables with configurable hardware parameters, cache layouts and analytical estimators as described in [the AgentX guide](agentx.md).
Result metadata records the exact provenance and match type of every operator and communication estimate.

AIConfigurator data is Apache-2.0 licensed.
Attribution and retained license text are in [the data notices](../data/operator_data/LICENSES.md), [NOTICE](../NOTICE) and `third_party/licenses/AIConfigurator-Apache-2.0.txt`.
