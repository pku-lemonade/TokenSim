# DSV4 CSA/HCA 原生 attention 阶段

本阶段已导入并接入 B200/B300 vLLM 0.24.0 的 DeepSeek-V4-Pro CSA/HCA attention module 表。每一行是完整 `DeepseekV4Attention.forward` 的模块测量，包含 fused input projections、compressor/indexer、稀疏 attention 与 output projection；运行时因此跳过原先独立的 DSV4 projection GEMM，避免重复计时。主干 61 层按 30 个 CSA（ratio 4）和 31 个 HCA（ratio 128）组合，MTP/SWA 保留为下一项成本。

导入器保留 FlashMLA 采集后端、FP8 source KV、collector 默认 indexer layout、版本和原始 Parquet。参考 serving 使用 FlashInfer 与 MXFP4 indexer，因此 backend/layout 差异在 provenance 中明确列出，不能把借用表宣称为完全同条件实测。B300 CSA generation 没有 batch<64 的 Pro 行，native lookup 对该下界执行可追踪 clamp；表外长度使用测量边界加 DSV4 压缩 work analytical growth，最大外推 64 倍。

`DeepSeekV4AttentionEstimator` 的 context 查询使用 `(batch, query_tokens, prefix_tokens)`，generation 查询把 MTP q=4 展开为 `batch × query_width`，并保留 CSA/HCA 各自的 measured/interpolated/extrapolated 计数。module 内含 rope/projection，外层只保留模型层 norm/residual。`native_dsv4` 是显式后端，generic attention 不受影响。

已接入代码与回归测试，当前全仓 **463 项通过**。两组完整窗口使用同一 393-trace 池、warmup、3,600 秒 profiling、MTP 平均接受长度 2.49 和 q=4 target verify，最终验收通过。结果（8 GPU 合计）：

| 指标 | B200 | B300 |
|---|---:|---:|
| 成功请求 | 3,072 | 6,306 |
| 输出吞吐 | 592.617 token/s | 1,140.733 token/s |
| p90 TTFT | 260.352 s | 88.404 s |
| p90 ITL | 370.474 ms | 384.778 ms |
| GPU / CPU 首次命中率 | 26.887% / 67.348% | 95.015% / 0.404% |
| decode accepted/query tokens | 2,490,361 / 4,137,531 | 4,823,036 / 8,072,309 |
| query/accepted ratio | 1.661 | 1.674 |
| attention table matches | 29,348 exact; 800,418 interpolated; 1,266,072 extrapolated | 26,976 exact; 723,791 interpolated; 3,321,097 extrapolated |

两组 native attention 时间与 DP local compute 守恒；MegaMoE 仍不另计 EP all-to-all，mHC 保持 generic。B200/B300 输出吞吐相对 MegaMoE 阶段下降，主要反映完整 CSA/HCA 模块和 MTP q=4 校验成本；该结果清楚暴露出原先 generic attention 与单 token decode 对成本的低估。FlashInfer/FlashMLA、plain-FP8/MXFP4 layout 和长上下文外推敏感性仍需在最终报告中单独说明，不能把借用表结果表述为同条件 kernel 实测。

导入复现：

```bash
python scripts/import_agentx_attention.py --upstream-root /tmp/aiconfigurator
```

结果目录：`results/agentx-dsv4-full/native-attention/20261002-csa-hca-mtp-fast/`。

完整验证：

```bash
python scripts/validate_agentx_attention.py \
  results/agentx-dsv4-full/native-attention/20261002-csa-hca-mtp-fast \
  --output data/reference/agentx-native-attention/pipeline-validation.json
```
