# Native attention sensitivity

An explicit `native_attention_context_scale=0.17` was tested using the B200
AIC shape comparison as a diagnostic input. Formal runs keep the default scale
of 1.0; this experiment does not alter reference results.

| SKU | formal output token/s | scale 0.17 output token/s | change | reference error after scale |
|---|---:|---:|---:|---:|
| B200 c196 | 522.135 | 676.538 | +29.6% | −81.2% |
| B300 c384 | 1,027.989 | 1,260.572 | +22.7% | −80.2% |

The scale improves throughput but leaves the result far outside the 25%
threshold. It is therefore retained as an explicit sensitivity knob and not
promoted into the formal matrix. The result rules out a single context
attention scalar as a sufficient fix; remaining error includes the complete
module/backend scope, prefill scheduling, cache pressure and other operator
provenance differences. Evidence is in
`results/agentx-dsv4-full/sensitivity-aic-attn/`.

An upper-bound run with context attention set to zero still produced only
670.881 token/s on B200 and 1,294.908 token/s on B300. This confirms that the
remaining gap is not explained by context-attention latency alone; removing a
real operator would be an invalid production model, so the formal scale remains
1.0.

2026-10-07 中断恢复核查：全 attention 归零的两个进程最终均自然完成，
原始目录已保留；逐请求计数、取消、最终 GPU 页和 DP 时间校验通过。
B200/B300 输出吞吐为 5,490.768/10,254.271 token/s，attention 分项为 0。
这仍是删除真实成本的诊断，不能作为正式模型结果。后续代码审计发现
decode 按 KV 长度组重复累加完整模块成本；不能把上述差距直接归因于
FlashMLA/FlashInfer 的后端性能差异。修复见
[整批 decode 报告](agentx-decode-batch-report.md)。
