# AIConfigurator data in TokenSim

TokenSim imports measured kernel and communication latencies from
[NVIDIA AIConfigurator](https://github.com/ai-dynamo/aiconfigurator/tree/f254959eb89e2f206b8f9a77051644d7c1cbdb89).
The bundled data uses commit `f254959eb89e2f206b8f9a77051644d7c1cbdb89`.
TokenSim runs its own event simulator; AIConfigurator, PyTorch and a GPU are
not runtime requirements for consuming the bundled tables.

Each package in `data/operator_data/<device>/<backend>` contains
`generation_meta.yaml` and normalized parquet tables. Every row names a
registered `source_id`; the metadata records the upstream path, collection
version, device, framework, units, timing scope and source grade. Original
millisecond measurements become microseconds. Source data that already uses
microseconds keeps its units.

## Rebuild an imported package

Check out the recorded commit, rather than whatever is currently on upstream
main. The documented directory format belongs to that revision.

```bash
git clone https://github.com/ai-dynamo/aiconfigurator.git /tmp/aiconfigurator
git -C /tmp/aiconfigurator checkout f254959eb89e2f206b8f9a77051644d7c1cbdb89
python -m TokenSim.operator_data.cli import-aiconfigurator \
  --system-dir /tmp/aiconfigurator/aic-core/src/aiconfigurator_core/systems/data/b200_sxm \
  --device b200_sxm --backend vllm \
  --upstream-commit f254959eb89e2f206b8f9a77051644d7c1cbdb89 \
  --module-source dsv4_megamoe=/tmp/aiconfigurator/aic-core/src/aiconfigurator_core/systems/data/gb200 \
  --out /tmp/b200-vllm
```

For B300 use `b300_sxm` and explicitly select the `gb300` MegaMoE source.
Cross-device use is a recorded input, not an automatic hardware substitution.
Omit `--module-source` to import only modules available in the selected system.
`--version` limits the requested framework's collection version; communication
and explicitly selected module sources retain their own collection versions.

The importer handles ordinary GEMM, attention, MoE, collectives and DeepEP
communication, plus complete sparse-attention, residual-mixing and fused-expert
modules. Model identity and geometry stay in the normalized rows. All imported
module source IDs are registered before the package is validated.

```bash
python -m TokenSim.operator_data.cli validate /tmp/b200-vllm
```

Keep the original `generation_meta.yaml` beside copied tables; parquet files
alone do not preserve source declarations. Loaded TokenSim packages are checked
against their declared table row counts; missing or truncated tables raise a
validation error. Device and backend metadata must match the selected
package directory. Registering an additional in-memory package preserves the
other packages available on disk. A custom collector run directory
can also be imported with `--system-dir`; its source records identify the
local collection rather than an upstream database directory.

## Consume data and select a cost scope

```bash
tokensim --operator_backend vllm --latency_backend operator_table \
  --latency_fallback table_first --results_path results/example
```

`table_first` uses measured shapes where available and reports analytical
fallbacks. `table_only` rejects missing shapes; `analytical_only` bypasses the
measurements. Extrapolation defaults to measured-boundary efficiency multiplied
by analytical work growth. Empirical power-law extrapolation is an explicit
lookup policy; superlinear fitted growth falls back to the analytical model.

A complete module table can include projections or communication that ordinary
operator tables price separately. Model `operator_modules` bindings select the
source table and its discrete selectors. The compositor respects that scope
and avoids counting included primitives again. Supported source layouts and
kernel restrictions are checked explicitly.

Measured data does not remove approximation error. The borrowed fused-expert
modules, FlashMLA/FlashInfer attention difference, synthetic expert spectra
and analytical draft/output work are explained in [the AgentX guide](agentx.md)
and in result provenance. Runtime hardware efficiencies and storage geometry
remain separate from measured kernel timing.

AIConfigurator data is Apache-2.0 licensed. Attribution and retained license
text are in [the data notices](../data/operator_data/LICENSES.md),
[NOTICE](../NOTICE) and `third_party/licenses/AIConfigurator-Apache-2.0.txt`.
