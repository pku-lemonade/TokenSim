# AIC same-shape offline diagnostic

An isolated AIC 0.12.0 runtime was built from source commit
`f254959eb89e2f206b8f9a77051644d7c1cbdb89` using a user-local Rust toolchain.
The diagnostic shape is `ISL=1024`, `OSL=128`, batch 8, TP8, EP8, attention
DP1. AIC uses its local DeepSeek-V4-Pro NVFP4 preview config; this is recorded
as a model/dtype difference from the formal native FP8/MXFP4 TokenSim model.

For B200, AIC reports TTFT 177.408 ms and TPOT 29.881 ms, with per-operation
breakdown and source provenance. The same-shape TokenSim native attention
lookup is 82.912 ms for a 1024-token prefill query and 4.492 ms for decode at
context 1024. The context discrepancy is retained as evidence of different
module scope/backend assumptions; no scalar calibration is applied. B300 AIC
estimation fails explicitly because its vLLM database has no matching NVFP4
MegaMoE row, so no cross-SKU estimate is synthesized.

Machine-readable evidence:

- [AIC runtime and per-op result](/home/zhuohang/TokenSim/TokenSim-Open/data/reference/agentx-aic-diagnostic.json)
- [AIC/TokenSim shape comparison](/home/zhuohang/TokenSim/TokenSim-Open/data/reference/agentx-aic-shape-comparison.json)
- [Diagnostic entrypoint](/home/zhuohang/TokenSim/TokenSim-Open/scripts/run_agentx_aic_diagnostic.py)
