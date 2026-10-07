# Native mHC 的残差范围与 SP 行数

接续 `5f9c1a8`，本部分修正 main native mHC 外层的重复工作与 native
summary 导出。完整 draft 成本沿用上一阶段，不合并其他算子或容量修改。

只读 vLLM `30f5c01d4ebb` 的 `DeepseekV4DecoderLayer.forward` 使用
mhc post/pre 重建 weighted residual；collector 的 `mhc_post_tilelang`
也已包含这个工作。此前 backend 在 native pre/post 之外又对 attention
与 FFN 各增加 generic residual_add，重复支付了残差费用。新实现移除
这两处；未启用 native mHC 的分支仍保留普通残差。

collector pre 明确使用 `norm_weight=None`，因此其模块表不含 attention /
FFN 输出 RMSNorm；当前 serving 把该 norm 融入 post/pre。仍保留单独
norm 作为借表范围的明确估算，不移除工作，也不使用 H20 的 fusion
比值缩放 B200/B300 模块。

源码的 SP predicate 为 PP1、EP、TP>1 且 MegaMoE 或 DP>1。main mHC
与 norm 在 attention all-gather 前运行，应查询 ceil(T/TP) 行；backend
现在在本轮 PP1 配置下按该 predicate 处理。目标 lm_head 的 logits
行数独立于 SP 分片；主干最终 norm 的全 forward 行范围另列后续
修正。TP1 DEP8 主点不改变 layer mHC/norm 行数，仅验证残差去重。

结果 schema 同时保留 `native_megamoe_routed_seconds`、shared seconds、
max_rank_rows 和 counts。producer 上一阶段已合并主干及三个 draft
pass；本阶段修复的是过滤后丢失字段。可直接以 routed + shared 核对
全局 MoE，并继续用独立 main MoEStats + explicit draft 账本交叉验证。
旧结果不改写，上一阶段的字段缺口留在其报告中。

## 验证

新增回归核对无 generic residual_add、norm 继续保留、66 target 行在
TP4 下的 main mHC/norm 为 17 行，以及 summary 进入 result schema。
全仓 **509 passed**、14 subtests passed、32 个已有 warning，142.95 秒，
日志 `/tmp/agentx-mhc-scope-tests.log`。完整窗口使用全部 393 trace 池、
完整预热、seed0、3,600 秒窗口及 grace/取消。B200/B300 均已结束
且通过严格验收。测试日志已归档到
`data/reference/agentx-mhc-scope/full-tests.txt`。本部分不代表整体
25% 精度或最终三 seed matrix 通过。

## 成对完整检查点（2026-10-08 结束）

8 GPU 合计吞吐如下；比较基线为 `5f9c1a8`。

| 指标 | B200 c196 | B300 c384 |
|---|---:|---:|
| 输出 token/s | 1,887.763 | 3,785.819 |
| 相对上一阶段增幅 % | 4.697 | 0.233 |
| 参考吞吐误差 % | -47.551 | -40.477 |
| p90 TTFT 秒 | 59.294 | 28.909 |
| p90 ITL 秒 | 0.124958 | 0.102969 |
| E2E 均值秒 | 96.371 | 84.811 |
| GPU 命中率 % | 30.8646 | 94.7746 |
| CPU 命中率 % | 64.2463 | 0.7624 |
| 抢占 | 72 | 0 |
| 成功请求 | 7,905 | 16,046 |
| 到期取消 | 86 | 285 |

两组均完成原始负载池上的预热和整小时窗口，全部取消返回；最终
各 worker 仅保留 null slot，DMA pin、外部缓存 pin、读写任务及在途
prefill 预留已释放。时间分项与 DP local compute、compute + sync wait
均守恒。实际 benchmark wall time 为 B200 1,635.879 秒、B300
3,254.326 秒。源码仅有两个 runtime 文件与基线不同，配置、数据
校验值及每个 worker 的计算容量预算一致。

原生 MoE 的独立账本（GPU 秒）：

| 分项 | B200 | B300 |
|---|---:|---:|
| routed | 8,603.655 | 7,733.565 |
| shared | 215.266 | 345.920 |
| routed + shared = 全局 MoE | 8,818.922 | 8,079.486 |

独立 main MoEStats + draft MoE 也分别等于上述总量。native rank
estimate 数为 626,624 / 615,296，均等于 4 × operator step；
完整 MTP 的三次调用计数保留。累计 missing-shape 清单均未截断，
没有 residual_add，保留 RMSNorm 解析估算。

平均 elementwise 为 B200 5,292.557 → 4,496.244 μs/step（−15.046%），
B300 7,569.994 → 6,485.909 μs/step（−14.321%）。闭环负载随执行
速度变化，其余分项与请求群体也变化；累计时间差不是固定工作量
的消融效果。B200 抢占 84 → 72，B300 仍为 0，完整缓存预算未调整。

相对参考的诊断误差分别列出；GPU/CPU 各以成功请求的首次准入输入
token 为模拟分母，保留与历史 telemetry 分母的差异。

| 指标相对误差 % | B200 | B300 |
|---|---:|---:|
| 输入吞吐 | -48.103 | -36.043 |
| 输出吞吐 | -47.551 | -40.477 |
| p90 TTFT | +265.865 | +52.829 |
| p90 ITL | +148.694 | +108.078 |
| E2E 均值 | +122.232 | +66.707 |
| GPU 命中率 | -54.759 | +5.422 |
| CPU 命中率 | +170.785 | -53.487 |

GPU 命中率绝对差为 -37.3584 / +4.8746 个百分点。

CPU 命中率绝对差为 +40.5203 / -0.8766 个百分点。

整体 25% 精度未通过。容量、其他配置和三 seed 全矩阵仍沿研究计划
逐项完成，本次只验收残差范围、SP layer 行数与 native MoE 导出。

验收命令：

```bash
python scripts/validate_agentx_mhc_scope.py \
  results/agentx-dsv4-full/native-mhc-scope/20261007-residual-scope \
  --output data/reference/agentx-mhc-scope/pipeline-validation.json
```

## 下一部分的只读核对

- `cached-prefill-followup.json`：waiting 的单个真实输入行可补齐为
  四个 verify query，invalid placeholder draft 会被拒绝，仍输出一个
  token。request progress、attention metadata 和 graph eligibility
  要分别建模；现有 input-only 日志不足以判定所有恢复尾部。
- `target-head-followup.json`：主干 forward 在所有 T 行上执行 hc_head
  和最终 norm，再选择 lm_head 行；当前主干漏 hc_head，且 norm 用
  selected rows。dedicated MTP 则把 hc_head 延迟到 compute_logits，
  其 selected-row 估算已符合源码，不扩大其 initial-pass head。
- mHC 后端已有本地统计，但 producer 未汇总主干/draft mHC，结果
  schema 也未导出；本阶段补齐的 summary 明确只包含 native MoE。
- `matched-aic-capability.json`：同模型 FP8/MXFP4 mixed 形状调用
  发现 AIC model builder 只支持 SGLang MegaMoE。更早把 config JSON
  文件当作 model directory 的 404 调用错误已独立保留，不能解释为
  数据缺失。借表 SGLang/GB200 的后续调用也记录实际失败与配置差异，
  未生成 formal vLLM 对照或采用历史 NVFP4 的比值。

这些后续核对均不改变本次运行的计时、容量、路由或负载输入。
源码与测试校验值、实际 PID/命令及每个 checkpoint 的严格验收分别
保存在 `data/reference/agentx-mhc-scope/`，历史运行不改写。
