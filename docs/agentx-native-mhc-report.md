# DSV4 mHC 阶段

已导入 B200/B300 vLLM 0.24.0 `mhc_module_perf`，筛选 DeepSeek-V4-Pro hidden size 7168、`hc_mult=4`、`num_sites=2`、Sinkhorn 20 的 pre/post TileLang 记录。原始表、采集 metadata、版本、硬件和 Apache license 保存在 `data/reference/agentx-native-mhc/`；`num_sites=2` 是 collector 的测量范围，估算器不会再次乘以 site 数。记录同时保留 BF16 residual 和 FP32 mHC 参数的 scope，避免将模块笼统标成单一 dtype。

`DeepSeekV4MHC` 以实测 pre/post 行为一层查询，按 `num_tokens` 插值和测量边界 analytical 外推，并将 mHC 时间纳入 elementwise component；查询 provenance 与 profile reset 均有回归测试。完整主点使用 native CSA/HCA + MTP 配置，结果如下：

| 指标 | B200 | B300 |
|---|---:|---:|
| 成功请求 | 2,969 | 6,137 |
| 输出吞吐 | 567.198 token/s | 1,114.907 token/s |
| p90 TTFT | 239.850 s | 90.613 s |
| p90 ITL | 366.039 ms | 396.488 ms |
| GPU / CPU 首次命中率 | 27.532% / 66.629% | 95.029% / 0.394% |
| mHC exact / interpolated 查询 | 20,438 / 85,146 | 11,304 / 84,248 |

两组完整窗口的 component 与 DP local compute 守恒，mHC pre/post scope、FP32 参数/BF16 residual 和 `num_sites=2` 均在 backend provenance 中保留。相对 attention 阶段，mHC 增加 elementwise 时间并进一步降低吞吐；该结果仍受 FlashMLA 借表、MTP 近似和缓存压力影响，不能单独作为整体精度结论。

导入复现：

```bash
python scripts/import_agentx_mhc.py --upstream-root /tmp/aiconfigurator
```

完整验证：

```bash
python scripts/validate_agentx_mhc.py \
  results/agentx-dsv4-full/native-mhc/20261002-mhc \
  --output data/reference/agentx-native-mhc/pipeline-validation.json
```
