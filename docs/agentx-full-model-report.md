# AgentX 完整模型结构与权重显存准备

日期：2026-09-30；分支：`agentx-support`。本次独立修改完成完整模型与并行配置任务 1、2、3、5。完整模型的一小时管线基线仍待运行入口、请求回放和混合缓存接入；没有用配置检查或单元测试替代模拟验收。

## 模型与张量核对

新增 `data/models/deepseek-v4-pro.yaml`、`data/psla/deepseek-v4-pro.json`，保存原始 HF config 与来源校验值。模型为 hidden=7168、61 层、384 专家、top-6、inter=3072；61 层全部为 MoE，前 3 层为 hash 路由。attention 为 30 层 CSA、31 层 HCA，另有一个 SWA-only MTP 模块。保留 q_lora=1536、o_lora=1024、o_groups=16、head_dim=512、RoPE=64、indexer=64×128/top-1024、SWA=128 和 mHC mult=4。

`ModelSpec` 支持逐层结构与逐算子 dtype map，明确区分 FP8 block、MXFP4、BF16、FP32，以及 mHC 的 FP32 参数和 BF16 residual。缺少结构或 dtype、把 hash 层改成 dense、MTP ratio 不为 1、PSLA 恢复旧 proxy 几何等情况会被拒绝。普通 transformer 的原有参数计算保留。

权重由 `TokenSim/config/weight_memory.py` 按真实张量形状计算，`CacheConfig` 使用分项字节量，而非将全部参数统一乘一个 dtype。对照只读 vLLM `30f5c01d4ebb5c493ed86195db24d0ec5b3ce3d7`：

- attention 使用低秩投影和 grouped BMM；`wo_a` 形状为 `[16384, 4096]`，不能当成全宽 dense 输出投影。
- FP8 scale 为 UE8M0 block scale，MXFP4 每 32 个权重另存一字节 scale；这些字节计入显存。
- hash 路由表、FP32 mHC 参数、compressor APE、BF16 router/norm/vocabulary 分别计数。
- EP8 每 rank 持有 48 个完整专家；不再对 EP 已分配的专家额外除以 TP。未启用 EP 时，专家按 intermediate 维度进行 TP 分片。
- MegaMoE 和 fused 布局区分 hash 表的 int64/int32 与 shared expert 的 TP 复制。主检查点使用 MegaMoE 布局。
- MTP 循环复用一个模块，target embedding 与 lm_head 被实际 proposer 代码共享，只有一次存储。

仅计算持久权重与量化元数据。激活、KV、通信缓冲区、CUDA graph 和 allocator 占用在缓存预算阶段分别处理，不把历史 profiler 的增量直接当作权重或补偿常数。当前确认的配置均为 PP1；未对齐 PP>1 的 MTP 所有权时明确拒绝该配置。

## 与日志对照

两个主点使用同一模型公式，无 SKU 专用系数，无日志容量输入。每个点的全部 8 个 rank 均完成权重核对。

| 项目 | 计算值（GiB/rank） |
|---|---:|
| routed experts，含一个 MTP 模块 | 97.268555 |
| attention 投影与 sink | 17.317493 |
| shared expert | 3.814686 |
| embedding + lm_head | 3.452148 |
| compressor | 1.252225 |
| indexer | 0.582418 |
| mHC | 0.318738 |
| router | 0.317955 |
| MTP 输入投影 | 0.095709 |
| hash 路由表 | 0.017338 |
| norm | 0.001945 |
| 合计，包含 scale 元数据 | **124.439211** |

历史 B200 / B300 日志为 125.88 / 126.08 GiB，误差分别为 **−1.145% / −1.301%**，均满足 ±2% 目标。保留未分解的约 1.44 / 1.64 GiB 差额；当前源码与历史镜像、打包以及加载时内存统计的差异尚未逐项消除，未为贴合日志添加修正系数。

新增 `data/clusters/8_b200/dep8.json` 和 `data/clusters/8_b300/dep8.json`：单节点 TP1×DP8×EP8、global EP、NVLink 5、allgather/reducescatter，无 KV 容量覆盖。矩阵已引用完整模型和这两份 cluster，49/96 sequences、8192/16384 tokens、256-token block 等参考参数仍保留；统一入口后续将这些参数传入实际调度器。完整 DSV4 配置禁止从参考日志覆盖 KV token 容量。

## 完整负载池分析与当前运行限制

使用全部原始文件，重算得到 393 条 trace、98,827 条请求；最大输入 **989,824**，最大原始请求上下文 **996,579**，小于模型上限 1,048,576。数据 SHA-256 与前一部分一致：`29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`。未修改、截断或替换原始 trace。

检查发现两项基础问题直接妨碍正式管线基线：

1. 旧回放要求 trace block 与 engine block 相同；原始 trace 为 64、参考引擎为 256，需要按前缀链聚合完整块。
2. 旧缓存将 latent KV 算成每层两份未压缩状态，得到 **62,464 B/token**。按 B200 0.90 / B300 0.92 的预算，旧方法分别给出 **585,472 / 2,046,976** token；B200 无法准入负载池的长请求。真实 DSV4 的压缩、SWA、indexer 与共享物理池尚未接入，不能用日志等效 token 数填平这个错误。

`CacheConfig` 导出时明确标记当前 KV 路径为 legacy uniform；模型审核输出 `formal_simulation_ready=false`，列出尚未接入的回放、混合缓存与耗时模块。这些容量仅用于定位当前实现问题，不能作为正式 DSV4 容量结果。预热、交接、3,600 秒窗口、30 秒宽限和取消收尾继续作为一小时验收要求。

旧 proxy 文件及旧脚本迁移尚未完成：已列出 `data/models/deepseek-v4-proxy.yaml`、`data/psla/deepseek-v4-proxy.json`、`scripts/run_agentx_comparison.py`、`scripts/run_agentx_sweep.sh`、`docs/agentx.md`、`docs/agentx-operator-data-gap-analysis.md`。研究计划和上一部分报告保留历史引用；`tests/test_agentx_reference.py` 的 proxy 字符串用于拒绝不兼容结果的回归测试。旧结果目录不动。运行入口迁移与 proxy 删除会作为下一次完整修改提交。

## 验证和复现

新增 18 个模型测试，覆盖 HF config 全字段、逐层结构、dtype、量化 scale、MTP 共享、TP/EP 守恒、错误配置和两个 cluster。完整执行：

```bash
python -W error::ResourceWarning -m unittest discover -s tests -v
python -m compileall -q TokenSim benchmark.py util scripts/validate_agentx_model.py
python scripts/validate_agentx_model.py \
  --dataset dataset/agentx/traces.jsonl \
  --output results/agentx-dsv4-full/model-preparation/your-batch
```

完整测试 **269 项通过**。模型审核读取完整负载池、对照两组全部 rank 的权重，输出每 rank 的张量清单与预算问题；它属于数据和模型核对，不属于 smoke 或一小时模拟。提交的证据在 `data/reference/agentx-model/validation.json`、`manifest.json`；完整张量清单在 `results/agentx-dsv4-full/model-preparation/20260930-model-and-weights-verified/`。

目前没有新模拟吞吐、延迟或算子覆盖比例可报告；原有 GQA 耗时路径不代表完整 DSV4 执行。全任务仍未完成，将继续实现统一入口和 AIPerf 等价回放、混合缓存，再运行两个主检查点的完整一小时流程，随后逐阶段修复调度与耗时。
