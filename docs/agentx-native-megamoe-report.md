# DSV4 MegaMoE 与 shared expert 阶段

本阶段把 DeepSeek-V4-Pro 的 routed expert 从独立 MXFP4 expert 表和独立 EP all-to-all，替换为与参考实现语义一致的 MegaMoE 模块：借用 AIConfigurator 的 DeepGEMM `deepgemm_megamoe` 表，保留 pre-dispatch、dispatch、expert compute、combine 的完整模块范围；gate/top-k 不在表内，shared expert 按 FP8 fused incremental cost 单独估算。数据导入保留原始 Parquet、采集版本、容量字段、硬件来源和 Apache license，跨 GB200/GB300 到 B200/B300 的初始比例固定为 1.0，并在结果中标注为 B 级借用数据。

`MegaMoEEstimator` 使用 DP lockstep 的整步 rank snapshot。每个 EP rank 采用自己的 rank-local token 数，图模式按同步批次的最大形状补齐到 q=4，混合 context/decode 或超出 graph 容量时保留 eager 的实际形状；TP sequence-parallel 会把输入行数分到每个 TP rank。组耗时取 EP 组中最大的测量估计，避免把 DP 或 EP 规模重复乘入。native 路径不再另计 EP all-to-all，shared FP8 fused cost 不调用独立 shared GEMM launch；关闭融合时才使用两次 FP8 GEMM 加 SwiGLU 的显式回退。

新增 `dsv4_megamoe` operator-data schema、导入器、CLI、`MoEExecutionConfig`、DP 形状同步、native 统计和结果 provenance。完整逻辑回归包括 TP/SP 行数、DP 不均匀形状、图/eager 选择、shared fusion scope、跨 buffer regime 插值、拓扑检查、数据 scope 校验和 profile reset。

本阶段逻辑验证：**459 项全仓测试通过**，其中 10 项 MegaMoE 回归测试。两套 operator package 均能加载 `dsv4_megamoe` 表；B200 128 行、B300 152 行，均为 Pro、EP4/8、TP1、power-law 1.01、gate/top-k 和 shared expert 排除的同一 scope。

B200/B300 完整主点均已结束，使用全部 393 条 trace、42/0 workload seed、warmup、3,600 秒 profiling、30 秒 grace 和取消收尾。结果（8 GPU 合计）：

| 指标 | B200 native MegaMoE | B300 native MegaMoE |
|---|---:|---:|
| 成功请求 | 7,606 | 15,041 |
| 输出吞吐 | 1,733.192 token/s | 3,440.686 token/s |
| p90 TTFT | 53.947 s | 15.023 s |
| p90 ITL | 111.667 ms | 106.754 ms |
| GPU / CPU 首次命中率 | 27.697% / 67.116% | 94.553% / 0.750% |
| 抢占次数 | 147 | 0 |
| 专家最大/平均负载 | 1.000004 | 1.000001 |
| 显式 straggler | 0 s | 0 s |
| EP all-to-all 单独耗时 | 0 s | 0 s |
| MegaMoE matched / interpolated 查询 | 131,312 / 1,964,784 | 27,840 / 1,997,880 |

组件时间与 DP local compute 守恒；B200/B300 分别完成 303,616/291,744 个同步 step，每步对应一个 native MegaMoE 查询。最终 worker、加载、预留、DMA 引用和取消状态均归零。输出吞吐相对上一阶段提高约 19.9%/11.1%；B300 已接近参考吞吐，但 B200 仍有容量边缘引起的抢占和缓存误差。该阶段仍未达到整体精度门槛，shared fusion 为 D 级解析增量，Attention、mHC 和 MTP 尚未计入本阶段结果。

复现导入：

```bash
python scripts/import_agentx_megamoe.py --upstream-root /tmp/aiconfigurator
```

复现完整窗口验收（B300 目录生成后同时指定两个 SKU）：

```bash
python scripts/validate_agentx_megamoe.py \
  results/agentx-dsv4-full/native-megamoe/20261002-megamoe \
  --output data/reference/agentx-native-megamoe/pipeline-validation.json
```

当前阶段的证据包括 `data/reference/agentx-native-megamoe/import-manifest.json`、原始表和 license，以及 `data/reference/agentx-native-megamoe/pipeline-validation.json`。下一阶段使用已导入的 DSV4 CSA/HCA attention module 表，显式建模完整投影、compressor/indexer、稀疏 attention 和 output projection，随后再接入 mHC 与 MTP。
