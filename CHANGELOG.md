# Changelog

## 0.1.0 (unreleased)

First packaged release. Changes are relative to the previous source-only
version of TokenSim.

### Added

- `pip install` support with the `tokensim` and `tokensim-operator-data`
  commands, an `agentx` extra for trace downloads and fast trace parsing, and a
  `dev` extra for tests and formatting. Bundled model, hardware and operator
  data resolve from the installed package; `--data_root` or
  `TOKENSIM_DATA_ROOT` select another data bundle.
- AgentX workload adapter (`--workload_type agentx_weka`): closed-loop replay
  of SemiAnalysis WEKA coding-agent traces with conversation dependencies,
  subagent spawn/join, recorded prefix hashes, warmup, grace period and
  cancellation. See `docs/agentx.md`.
- Model features selected through model YAML geometry and operator bindings:
  compressed sparse attention, grouped KV caches with sliding-window groups,
  hyper-connection residual mixing, fused expert-parallel MoE modules and
  multi-token-prediction speculative decoding with an average acceptance
  length. DeepSeek-V4-Pro is included as a complete model description.
- Complete operator-module tables (`dsv4_attention`, `dsv4_megamoe`, `mhc`)
  imported from AIConfigurator for every system that has them. Module tables
  replace the primitive operators they contain, so their cost is not counted
  twice.
- Data-parallel lockstep execution with dummy batches and finished-state
  consensus, session-aware consistent-hash routing, fixed routing maps and
  optional sticky sessions.
- Asynchronous CPU offload (`SimpleCPUOffloadConnector`) and a grouped-page
  Mooncake store with queued PCIe transfers and page reference ownership.
- Memory budgets derived from model weights, persistent buffers, activations,
  CUDA-graph capture and communication buffers.
- `--sim_time` cutoffs with partial results, and failure diagnostics
  (`failure.json`, `failed.txt`) in the result directory.
- `docs/aiconfigurator.md`: importing and validating AIConfigurator data.
- Apache-2.0 license, third-party notices and retained upstream licenses.

### Changed

- `--model` takes a model catalog name or a model YAML path; request
  generation and SLO targets move to CLI options or `--workload_config`.
- `--max_parallem_sum` is now `--max_num_seqs` (default 64): the active
  sequence limit for every batching mode.
- Extrapolation beyond measured operator data keeps the boundary row's
  efficiency and applies analytical work growth by default.
- The paged-attention scheduler follows vLLM V1: running requests in arrival
  order, no admissions in a step that preempted, and a failed first-chunk
  allocation stops admission instead of preempting the new request.
- Results reject unknown fields. Operator-module statistics are grouped under
  `module_stats`; `output_token_ps` counts generated tokens.
- Latency targets have no built-in defaults. When the workload sets them,
  results report each target's attainment and the goodput under `slo`.
- Each engine owns its timing recorder; repeated runs in one process are
  independent.
- Python 3.11 or newer is required; 3.12 is recommended.

### Removed

- PSLA workload files (`data/psla`) and proxy model configurations.
- Research artifacts, stage reports and accuracy-alignment scripts.
- The interpreter wrapper `scripts/run_python.sh`.
- The `moe_all2all_latency` result field, which was always zero; expert
  dispatch and combine time is `parallel_ep_all2all_latency`.
