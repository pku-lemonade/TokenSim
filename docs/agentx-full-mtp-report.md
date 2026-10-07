# DeepSeek V4 完整 MTP draft 调用成本

接续 `235c820`；本阶段补齐 dedicated MTP 层的实际调用、输入形状与成本。
之前仅对三个 draft output 行估算两次 H×H 投影和一次 head，未包含
decoder SWA/MoE/mHC，也将独立 pass 合并成一组 GEMM。

只读 vLLM `30f5c01d4ebb` 的 autoregressive speculator 首个 `_prefill`
复用全部 target 隐藏行和 attention metadata，再执行 K−1 次单 query
pass。新模型使用一个独立 estimator，K=3 时分别计三次真实调用；
partial prefill、warmup OSL=1 和最后一个输出的 pass 也保留。不执行
真实张量或 token，2/3 token 输出推进与 source dataset 保持原逻辑。

| 调用 | forward 行数 | e_proj 行 | h_proj 行 | head 行 |
|---|---|---|---|---|
| 初始 draft | target 全部 T（FULL 时 capture padding） | ceil(T/TP)，SP 时 | hc_mult × e_proj 行 | 一行/request，FULL capture 包含 padding |
| 后续两次 | 一行/request，各自 q=1 capture | 同上，SP 时 | hc_mult × e_proj 行 | 每次同一行/request |

16 请求、target 64 行时，三个 pass 是 `[64,16,16]`，e_proj 共 96 行、
h_proj 共 384 行、head 共 48 行；不是旧估算的 96 个输入投影行。
DP snapshot 现在显式携带每 rank 的 request 数，不能由混合 token 数
盲猜。首个 pass 的 graph/eager 形状按 target capture，后续使用独立
q=1 capture，TP/SP 行数和通信分开处理。

每个 pass 包括共享 embedding 读取、fused input norm 的工作量、FP8
e/h projections 与 residual broadcast-add、四个 DSV4 低秩 attention
投影、SWA-only core 与 RoPE、两处 mHC pre/post 及 layer norm、BF16
router、一个 routed/shared MoE 层、hc_head、shared RMSNorm 和 BF16
lm_head。完整 target attention table 不用于 MTP SWA，也不叠加 generic
residual_add；main mHC 外层的重复残差仍是独立修正任务。

原生 MoE estimator 新增显式层数，不将一个 draft 层乘成主干 61 层。
其已包含 EP dispatch/combine，所以不另计同一通信；SP 的 attention
AG/RS 和最终 hyper-residual AG 则按实际消息大小单独计时。主干与 draft
native MoE 的 routed/shared seconds 合并导出，组件仍严格守恒。

套餐没有匹配的完整 Pro SWA 或 hc_head 表。SWA core 按 128 窗口、
单份 plain-FP8 latent 的 FLOP/byte 与设备既有 efficiency/launch 参数
解析估算；hc_head、embedding 和 input-add 同样明确标为 D 级。低秩
GEMM、mHC 和 native MoE 仍优先使用匹配或已确认借用的模块表，缺失
形状沿用现有插值/解析路径，source/module/row counters 分开记录。
这些不是 q=4 或完整 draft 的配对实测，不添加吞吐拟合系数。

初始 draft 也写 prefill KV，scheduler 对 prefill forward 同样预留 K，
异步 DMA 不执行 forward，因此加载阶段仍不分配它。整个 393-trace
池 input + output + 7 都在模型上限内。cached-prefill 单 token padding
与其他 max-model-len 边界行为仍单独核对，不据本阶段宣称都已完成。

## 验证

新增形状、TP/SP/capture/head、单层 MoE、SWA 窗口饱和、target/draft
attention 分项、全部 routed/shared 时间守恒、profile reset 与禁用 MTP
回归。全仓 **507 passed**、14 subtests passed、32 个已有 warning，
139.38 秒；日志 `/tmp/agentx-full-mtp-tests.log`。两个完整一小时主点
验收完成后补结果；拟使用 `native-full-mtp/20261007-three-passes/`，
全部 393 trace 池、完整预热、seed0、3,600 秒测量及 grace/取消。
最终随机三 seed 曲线与 25% 精度仍未完成。

两个完整主点已在测试通过后于独立进程会话启动，记录见
`data/reference/agentx-full-mtp/process.json`，源码保持冻结。后续由
`scripts/validate_agentx_full_mtp.py` 检查每步三次 pass、单层 MOE 范围、
e/h/head/capture 行数以及完整 target/draft 组件与 DP/缓存生命周期。

## 完整窗口验收

两个完整主点均退出 0，完整 dataset/code SHA、配置与静态容量、请求
输入来源/输出、取消确认、null slot、在途预留、CPU/DMA pin，以及
target/draft/component/DP 时间守恒通过。证据在
`data/reference/agentx-full-mtp/pipeline-validation.json`；393 trace 为
负载池，完整预热后测量 3,600 秒并保留 grace/取消，不要求全部原始
请求完成。本轮 seed0 不替代后续最终三 seed 曲线。

| 指标（8 GPU 合计） | B200 | B300 |
|---|---:|---:|
| profiling 成功 / 取消 | 7,813 / 122 | 15,997 / 272 |
| 输出吞吐 (token/s) | 1,803.074 | 3,777.004 |
| 较 target-budget 基线 | −2.30% | −2.30% |
| 相对参考吞吐误差 | −49.90% | −40.62% |
| p90 TTFT (s) | 57.039 | 29.229 |
| p90 ITL (s) | 0.130583 | 0.104981 |
| GPU 首次命中率 | 30.2026% | 94.8549% |
| CPU 首次命中率 | 64.7923% | 0.6933% |
| 抢占次数 | 84 | 0 |

累计 draft GPU 秒为 884.152 / 952.911，包括：

| draft 分项 | B200 | B300 |
|---|---:|---:|
| GEMM | 480.013 | 534.139 |
| SWA core | 3.394 | 5.506 |
| MoE | 228.426 | 221.839 |
| elementwise / mHC | 35.025 | 47.611 |
| hc_head / norm / lm_head | 137.293 | 143.817 |
| comm（DEP8 TP1） | 0 | 0 |

target steps 为 149,648 / 152,736；draft passes 为
448,944 / 458,208，均为每 target step 初始一次 + 后续两次。forward
行数为 64,555,090 / 130,873,128，h_proj 为
258,220,360 / 523,492,512，恰为四倍；head 行数为
11,123,571 / 21,799,852。graph/eager 与行数、profile reset 逐项核对。

本次验收发现旧 `LLMResult` schema 未保存 util 已生成的 native
routed/shared 汇总字段。总 MoE 仍由独立账本严格证明：主干
`moe_compute_latency + moe_straggler_latency` 加 explicit draft MoE，
等于全局 component；B200 为 8,506.526 + 228.426 = 8,734.952 秒，
B300 为 7,788.637 + 221.839 = 8,010.476 秒。unit 同时核对过滤前
routed/shared 汇总，验收没有略过 MoE 时间检查。缺失字段明确标为
schema export follow-up，最终矩阵前必须保留，不改写旧结果或宣称
字段已保存。

**整体 25% 精度仍未通过**；完整 draft 增加此前漏计的费用，不能为
提高吞吐而移除。下一独立部分消除 main native mHC 外层重复残差，并
保留上述 schema 汇总字段；cached-prefill padding、混合 target 投影、
不均匀 DP MoE、buffer 生命周期与最终 matrix 继续推进。
