# Changelog

## 0.1.0 (unreleased)

This is the first packaged release of TokenSim.
The listed changes are relative to its source-only release.

### Added

- The package supports `pip install` and provides the `tokensim` and `tokensim-operator-data` commands.
  The `agentx` extra adds trace downloading and fast parsing, and the `dev` extra adds testing and formatting tools.
  Model, hardware and operator data load from the installed package, with `--data_root` or `TOKENSIM_DATA_ROOT` selecting another data bundle.
- `--workload_type agentx_weka` replays SemiAnalysis WEKA coding-agent traces while preserving request dependencies and subagent start and join relationships.
  Replay uses recorded prefix hashes and includes cache warmup, time for outstanding requests to finish and cancellation of remaining requests.
  See `docs/agentx.md`.
- Model YAML files define the dimensions and select measured modules for compressed sparse attention, mixing residual streams through hyper-connections and fused mixture-of-experts (MoE) operations.
  Models also support grouped key/value (KV) caches with sliding-window groups and multi-token prediction, which proposes draft tokens with a configured average acceptance length.
  DeepSeek-V4-Pro is included as a complete model description.
- AIConfigurator imports include complete module tables (`dsv4_attention`, `dsv4_megamoe`, `mhc`) for each system with available measurements.
  Module times replace the individual operations included in those measurements.
- Data-parallel workers can advance in synchronized steps, use dummy batches on idle workers and agree on when execution has finished.
  Routing supports consistent conversation hashing, fixed routing maps and optional sticky sessions that keep requests on the same replica.
- `SimpleCPUOffloadConnector` supports asynchronous transfers to CPU memory.
  The grouped Mooncake store queues PCIe transfers and tracks references that protect pages while transfers use them.
- Memory budgets account for model weights, GPU buffers retained between kernel calls, activations, CUDA-graph capture and communication buffers.
- `--sim_time` stops execution at a simulated-time limit and saves partial results, while failures produce `failure.json` and `failed.txt` in the result directory.
- `docs/aiconfigurator.md` documents importing and validating AIConfigurator data.
- The repository includes the Apache-2.0 license, third-party notices and retained upstream licenses.

### Changed

- `--model` takes a model catalog name or a model YAML path.
  Request generation and latency targets, also called service-level objectives (SLOs), are configured through CLI options or `--workload_config`.
- `--max_parallem_sum` has been renamed to `--max_num_seqs`, which limits active sequences in every batching mode and defaults to 64.
- Extrapolation beyond measured operator data keeps the boundary row's efficiency and applies analytical work growth by default.
- The paged-attention scheduler follows vLLM V1 and serves running requests in arrival order.
  A step that pauses a request to reclaim memory admits no new requests.
  If memory allocation for a waiting request's first chunk fails, admission stops and the request remains waiting.
- Results reject unknown fields.
  Operator-module statistics are grouped under `module_stats`; `output_token_ps` counts generated tokens.
- Latency targets have no built-in defaults.
  When the workload sets targets, `slo` reports which were met and goodput for requests meeting all per-request limits.
- Each engine owns its timing recorder; repeated runs in one process are independent.
- Python 3.11 or newer is required; 3.12 is recommended.

### Removed

- PSLA workload files (`data/psla`) and proxy model configurations have been removed.
- Research artifacts, stage reports and scripts that fit results to reference measurements have been removed.
- The interpreter wrapper `scripts/run_python.sh` has been removed.
- The unused `moe_all2all_latency` result field, which always reported zero, has been removed.
  `parallel_ep_all2all_latency` records the time for sending tokens to experts and collecting their results.
