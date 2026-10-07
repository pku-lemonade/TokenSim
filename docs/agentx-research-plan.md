# AgentX 模拟精度提升：问题分析与分阶段计划

> 版本：v4，2026-09-30。讨论决定已合并；当前没有待用户答复的决策项。
> 执行顺序：参考数据与完整模型准备 → 请求回放 → GPU 分配与缓存复用 → 调度行为 → 算子耗时校准 → 完整曲线验证。
> 源码基准：只读参考 `/home/zhuohang/Serving/vllm-note`，本轮记录 HEAD `30f5c01d4ebb`；不下载历史镜像、不修改或切换此目录。本文旧行号和旧 HEAD 下的源码结论需按此版本重新定位；历史运行日志仍作为实验事实保留。
> 分支：`agentx-support`（基于 `b2b685a`）
> 历史实现与结果（不是最终精度验收）：**2026-10-02 已完成 MegaMoE/shared expert、native CSA/HCA attention、MTP 2/3 推进与 q=4 target verify、native mHC，以及 native Mooncake shared-page adapter 的代码和两组完整一小时主点验收。统一矩阵已支持 list/check/expand/run/summary/plot、断点续跑、实际 GPU 数归一化和多 seed；DEP8、TP4/EP4、TP8/HBM 的 19 个曲线点已完成 seed0/1/2 全窗口并通过逐点守恒校验，所有 seed1/2 状态成功且 seed 间输出存在差异。修复后的三 seed 主点均已成功：B200 均值 522.135、B300 均值 1,027.989 token/s。完整阶段吞吐依次为：MegaMoE B200/B300 1,733.192／3,440.686，native attention+MTP 592.617／1,140.733，native mHC+attention+MTP 567.198／1,114.907，native Mooncake B200/B300 583.269／1,045.469 token/s。已完成 HBM/MTP 消融、记录时序、固定历史 worker 映射和独立 AIC B200 shape diagnostic；AIC B300 的缺失 MegaMoE 行、FlashMLA/FlashInfer 后端差异、mHC/shared fusion 和跨硬件 MegaMoE 借表均保留为明确近似，因此最终绝对吞吐仍未通过 25% 门槛。报告见 [原生 MegaMoE](agentx-native-megamoe-report.md)、[原生 attention](agentx-native-attention-report.md)、[原生 mHC](agentx-native-mhc-report.md)、[Mooncake](agentx-mooncake-report.md)、[原生曲线](agentx-native-curve-report.md)、[AIC 诊断](agentx-aic-diagnostic-report.md)。**
> 当前状态（2026-10-08）：main native mHC 重复残差已消除，SP layer 行数已修正，native routed/shared MoE summary 已导出；509 项测试及 B200/B300 两组完整主点严格验收通过。最新吞吐 B200 1,887.763 / B300 3,785.819 token/s，相比完整 MTP 阶段提高 4.697% / 0.233%，相对参考仍低 47.55% / 40.48%，整体 25% 门槛未通过。下一部分补齐主干 hc_head / final norm 的全 forward 行范围和 mHC 统计汇总，再完成 cached-prefill padding、混合投影、非均匀 DP MoE、buffer 生命周期、历史映射与匹配 AIC 对照；最终三 seed 矩阵待实现冻结后重跑。见 [完整 MTP](agentx-full-mtp-report.md)、[mHC 残差范围](agentx-mhc-scope-report.md)。
> 对照对象：InferenceX AgentX，完整 DeepSeek-V4-Pro（FP8 + MXFP4），vLLM + MTP；覆盖 B200/B300 多并发曲线及其 TP8、TP4、DEP8 等配置变化。主检查点为单节点 8 GPU DEP8：B200 并发 196、B300 并发 384。
> 证据分层：历史配置和日志描述参考运行；本地源码决定本轮实现；二者存在版本差异时单列记录。【待核实／待验证】是实现中的技术检查，不是待用户批准；【估算】不是实测；【推断】须保留依据与适用范围。历史研究中的源码行号仅作定位线索，以函数和本轮版本为准。

---

## 0. 前言

### 0.1 工作约束

1. 按 §6 已确认决定实施。常规源码核对与工程适配直接完成；只有发现新的范围或近似取舍时才集中讨论，不重复确认已有决定。
2. 不用 smoke 测试。完整测试采用与参考一致的一小时模拟窗口：使用全部 393 条 trace 作为负载池，完成预热，再模拟 3,600 秒，按参考处理等待与取消（§5.1）。不要求 98,827 条原始请求全部完成。单元测试和差分测试只验证逻辑，不作为精度证据。
3. 不再使用 proxy 模型（`data/psla/deepseek-v4-proxy.json`），改用按 HF config 构建的完整 DeepSeek-V4-Pro。
4. 投机解码采用已确认的平均接受长度方案：每个请求的 decode step 以 51% 概率推进 2 个 token、49% 概率推进 3 个 token，期望为 2.49；同时估算草稿生成和校验的成本，不逐 token 执行真实 draft/verify。请求结束时按剩余输出长度截断。
5. main 与远端保持一致，所有修改只放在 `agentx-support` 分支。
6. `dataset/agentx/` 是本地未跟踪数据，不删除、不覆盖。
7. 按本次用户要求，每次完成一个具体部分，测试后提交相关代码并停下汇报，不一次完成全部修改。
8. 讨论和计划使用「请求回放」「GPU 路由」「缓存管理」「算子耗时」等直接描述工作的名称，不使用阶段代号。
9. 待确认事项每次集中提出约 6–8 项，逐项用直接的概念名称说明推荐方案和取舍；按用户答复分别记录。只询问真正需要选择的近似、范围和成本取舍；已经授权的源码对齐、事实核实和常规实现不再包装成待确认问题。未答复的事项保持待确认。
10. 每完成一类修改，B200、B300 各完整模拟一小时；最终含随机因素时各用三个随机种子验证，完全确定时不重复相同运行。本轮同时复现多个并发量的性能曲线，包含参考曲线中低并发与高并发的并行配置变化，由统一实验脚本读取逐点配置。
11. 容量完全由模拟器从硬件、完整模型和运行配置计算；参考日志仅用于核对误差，不能作为容量输入或覆盖值。
12. 请求解析参考 AIPerf 源码，在 TokenSim 内部实现。TokenSim 直接读取原始 trace，不依赖 AIPerf 包、加载器进程或其离线导出文件。
13. SimpleCPU 和 Mooncake 两种缓存后端同时支持，通过实验配置选择。TP、DP、EP、GPU 数、并发与缓存后端等差异也由配置表达，使用统一模拟接口和实验脚本。
14. 已确认实测优先：匹配的 AIConfigurator 实测数据直接使用；缺失形状再插值或外推；模型、硬件、后端及缓存格式不匹配时分别标注和核对，不把近似冒充同配置实测。混合批次沿用现有正常流程，接入模块表时避免重复计算已包含的算子。

一小时指模拟时间，全部 trace 是负载池，不要求全部原始请求执行完成。正式结果由内部闭环回放和 router 产生；固定参考发出时间、固定实测 worker 的模式只用于诊断。GPU、CPU 命中率分别验收。成本近似、校准约束和实验范围集中记录在 §6，验证流程见 §5。

### 0.2 对之前结论的更正

1. **参考配置搞错了。** 之前写的 DSpark K=6 / AL 3.77 / interval 16 / max-num-batched-tokens 8535 / 35 GiB KV 来自 InferenceX PR 3274，不是博客结果对应的配置。博客结果（InferenceX 实测 id 440239 / 439522）对应的是：MTP K=3、synthetic AL 2.49、interval 8、long-prefill threshold 512、max-num-batched-tokens 8192（B200）/ 16384（B300）。两个 SKU 都是 vLLM，B300 不是 SGLang。
2. **MTP 成本的计法。**
   - 旧文档 §4.2 把 verify 当成单独的一次 prefill 式 forward。实际上 verify 和 target decode 是同一次 forward：每个 decode 请求带 1+K = 4 个 query token；draft 走的是 MTP 层自己的 pass。
   - AIC 的做法已读 Rust 引擎核实，包括三部分：decode batch × (1+K)；DSV4 的 generation 算子再乘 (L+K)/L = 64/61 ≈ 1.049；进度按 1 + 平均接受数计。我中途一度以为 AIC 只乘 1.049，原因是漏看了 Rust 引擎，现已更正。
3. **量化不是 NVFP4。** attention GEMM 和 shared expert 是 FP8 block，专家是 MXFP4（w4a8，走 mega-MoE）。
4. **「57× 差距」里有 8× 是 GPU 数口径**（按 64 GPU 归一化，而参考是 8 GPU）。
5. **KV 布局（v2 写错了）。**
   - 584 B/state（448 NoPE + 128 RoPE + 8 scale）只用于 FlashMLA 的 fp8_ds_mla 布局。
   - 参考在 SM100 上用的 FlashInfer 类设置了 `use_fp8_ds_mla_layout=False`，所以 KV 是 plain FP8，512 B/state。server.log 里也没有 fp8_ds_mla 的日志行。
   - indexer cache 是 MXFP4，68 B/state。详见 P7。
6. 其余更正见 §3。

### 0.3 你提出的 5 个问题与本文编号

| 你的问题 | 本文编号 | 结论 |
|---|---|---|
| 1. 超长上下文 / 调度不到位 | P4 + P5 + P7 | 调度决定 chunk 形状，长 prefix 仍影响成本，缓存决定剩余计算量。三者都需要对齐，详见 §2.17；节流的生效条件见 P5 |
| 2. MLA 不支持 | P2 + P3 | DSV4 不是 V3 的 MLA，而是逐层混合的 CSA（ratio 4 + indexer top-k）/ HCA（ratio 128）/ SWA（128）。现在按 GQA 建模，偏差很大 |
| 3. FP4 权重 + FP8 KV | P6 + P7 | 实际是 FP8 block + MXFP4 w4a8 + FP8 KV + FP4 indexer cache，会同时影响延迟、权重显存和 KV 容量 |
| 4. DSpark 投机解码 | P8 | 参考用的是 MTP K=3 / AL 2.49。AL 参数方案可行，但成本要计入 verify 宽度 1+K 和 draft pass（AIC 也是这样做的） |
| 5. FlashInfer MLA sparse | P9 | 有 DSV4 实测表，但采集后端为 FlashMLA，参考为 FlashInfer。缺少同条件配对数据，不能断言 kernel 差异很小或套用固定比值；先核实缓存布局，再分别估算计算与访存差异 |

还需处理拓扑、会话路由、完整回放语义、mHC、MoE、缓存卸载和运行开销。影响大小须在行为对齐后通过完整模拟及消融判断，不能从旧 proxy 结果直接归因。

---

## 1. 参考基准事实

来源：

- 启动脚本 `/tmp/infx-hist/b200_vllm_mtp_843c6254.sh` 和 `/tmp/infx-hist/b300_vllm_mtp_c62f6e33.sh`（InferenceX 历史版本）；
- 运行产物 `/tmp/infx-art/{b200,b300}/`。

### 1.1 参考运行配置（行号为 B200 脚本）

| 项 | 取值 | 出处 |
|---|---|---|
| 镜像 | `vllm/vllm-openai:nightly-dev-x86_64-cu13.0.1-426e59f`。历史参考镜像；实现按只读本地 vllm-note 的 `30f5c01d4ebb` 核对，不下载该镜像 | `nvidia-master_843c6254.yaml:964-965`（dsv4-fp4-b200-vllm-agentic-mtp）；`nvidia-master_c62f6e33.yaml:1727-1728`（b300）。非 agentic 的 dsv4-fp4-b200-vllm-mtp 用的是 v0.25.0（:1015-1016），不是参考 |
| 拓扑 | DEP8：`--tensor-parallel-size 1 --data-parallel-size 8`，`--enable-expert-parallel --enable-ep-weight-filter` | :211-224 |
| MoE 后端 | `deep_gemm_amxf4_mega_moe`，`VLLM_DSV4_MEGA_FP8_COMBINE=1` | :222，:105 |
| KV | fp8，block 256，max-model-len 1048576，`--no-disable-hybrid-kv-cache-manager` | :291-293，:304 |
| Attention | `FLASHINFER_MLA_SPARSE_DSV4`，开启 prefill query 量化和 FP4 indexer cache。日志确认：KV 为 plain FP8（没有 "fp8_ds_mla KV cache format" 行）；indexer 为 MXFP4（"Using MXFP4 indexer cache"，`attention.py:896` 打出）；indexer decode 路径 next_n=4（`indexer.py:298` 打出）。AttentionConfig 打印的 `indexer_kv_dtype='bf16'` 与运行时不符，是 426e59f 的配置语义差异（HEAD 下 'bf16' 与 `use_fp4_indexer_cache=True` 同时出现会报错，`vllm/config/attention.py:119-130`），以运行时日志为准 | :302；server.log |
| 调度 | interval 8，threshold 512；MAX_NUM_SEQS = 2·CONC/TP = 49（B200）/ 96（B300） | :229-231，:239 |
| 投机解码 | mtp，K=3，synthetic，AL 2.49 | :249-259 |
| CUDA graph | FULL_DECODE_ONLY；capture 大小 4, 8, …, MAX_NUM_SEQS×4（单位是 token）；mode 0 | :260-277 |
| 环境变量 | V2 model runner；RETENTION_INTERVAL=32768（PR #43447/#44774）；Rust frontend；READY_TIMEOUT 3600；expandable_segments；FLOAT32 high；ARCH 10.0；`--no-enable-flashinfer-autotune` | :97-106，:215，:280-282，:297 |
| Router | vllm-router 0.1.14，consistent_hash，`AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1`，intra-node DP 8，timeout 14400，disable-retries | :77-87，:321-336 |
| Offload | SimpleCPUOffloadConnector，kv_both，lazy false，PYTHONHASHSEED=42；命令含 enable_cross_layers_blocks，但当前 SimpleCPU 不读取此键，不能据此断言其启用了 Mooncake 的跨层机制 | :123-144 |
| CPU 容量 | 每 rank 356.125e9 B。日志为 289,815 blocks × 1,228,800 B = 331.67 GiB ≈ 74.19M token（4,800 B/token），两个 SKU 相同 | :125；server.log |
| Metrics | aiperf 同时抓引擎端口和 router；历史聚合未去重，总量出现两倍计数，提取时按引擎身份去重 | :93 |
| 其他引擎参数 | `all2all_backend: allgather_reducescatter`、`disable_nccl_for_dp_synchronization: true`、`enable_eplb: false`、`num_redundant_experts: 0`、`expert_placement_strategy: linear`、`enable_dbo: false`、`numa_bind_nodes [0,0,0,0,1,1,1,1]`、`quantization: deepseek_v4_fp8`；dp / ep 组用 PYNCCL all-reduce | server.log 第 7 行（non-default args）、:79-80 |

### 1.2 SKU 差异

| 项 | B200 | B300 |
|---|---|---|
| 并发 | 196 | 384 |
| max-num-seqs（每 rank） | 49 | 96 |
| max-num-batched-tokens | 8192（脚本注释：16K 时 DeepGEMM 显存不够） | 16384 |
| gpu-memory-utilization | 0.90 | 0.92 |
| offload 参数名 | `cpu_bytes_to_use_per_rank` | `cpu_bytes_to_use`（同样是每 rank 的量） |
| 其他 | numa-bind、cumem allocator、RPC_TIMEOUT 600000 | NCCL_NVLS_ENABLE=1 |
| 行号 | VLLM_CMD :286-312（fp8 :291、attention :302、spec :303）；MODE_ARGS :218-233；spec :249-259；router :77-87 / :321-336 | VLLM_CMD :301-326（fp8 :310、attention :313、spec :314）；MODE_ARGS :232-252；spec :268-278；router :84-94 / :335-350 |

### 1.3 模型

HF config 为 `/tmp/aic-full/aic-core/src/aiconfigurator_core/model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json`，已与 vLLM 代码核对。

- **主干**：hidden 7168，61 层，vocab 129280，nextn 1（只有 1 个 MTP 模块，K=3 时循环使用 3 次），hc_mult 4。
- **MoE**：61 层全部是 MoE。384 专家，top-6，inter 3072，1 个 shared expert，路由 sqrtsoftplus/noaux_tc，routed_scaling 2.5。前 3 层是 hash 路由 MoE（`vllm/models/deepseek_v4/nvidia/model.py:809`，tid2eid）。
- **Attention**：128 heads，1 个 KV head，head_dim 512，rope 64，q_lora 1536，o_lora 1024，o_groups 16，SWA 128，indexer 64×128 top-1024。
- **compress_ratios**：共 62 项，形如 [128,128,4,128,4,…,4,0]，即 30 层 CSA（ratio 4）+ 31 层 HCA（ratio 128）+ MTP 层。vLLM 把 MTP 层按 ratio 1 处理，只有 SWA（`attention.py:220-227`）。
- **量化**：FP8 e4m3，ue8m0 scale，block [128,128]，动态激活。
- **各部分 dtype**：
  - FP8 block：attention GEMM、shared expert；
  - BF16：router、indexer weights_proj、compressor、embed、lm_head；
  - FP32：mHC 参数；mHC residual 为 BF16，不能把整个模块统称为 FP32 运算；
  - MXFP4（w4a8）：专家；
  - FP8：KV（plain FP8，512 B/state）；MXFP4：indexer cache（68 B/state）。

### 1.4 参考结果（8 GPU 合计）

| 指标 | B200 | B300 |
|---|---|---|
| 输入吞吐（tok/s，含命中 token） | 470,700.88 | 813,831.84 |
| 输出吞吐（tok/s） | 3,599.26（每 GPU 449.908） | 6,360.26（每 GPU 795.033） |
| profiling 时长（s） | 3,629.806 | 3,629.410 |
| 完成记录数 | 13,991 | 25,696 |
| TTFT mean / p50 / p90 / p95（s） | 8.790 / 3.363 / 16.207 / 33.369 | 10.501 / 4.265 / 18.916 / 41.701 |
| ITL mean / p50 / p90 / p95（ms） | 36.95 / 34.25 / 50.25 / 59.97 | 44.33 / 45.21 / 49.49 / 51.46 |
| E2E mean / p50 / p90（s） | 43.365 / 21.437 / 95.38 | 50.874 / 26.529 / 112.06 |
| ISL mean / p50 / p90 / p99 | 122,118 / 86,547 / 255,975 / ≈619,832 | 114,949 / 77,491 / 252,093.5 / ≈625,611 |
| OSL mean / p50 / p90 | 933.79 / 404 / 2,094 | 898.35 / 403 / 2,021 |
| OSL expected（历史聚合错映射，不是正确期望） | 1,261.54 | 1,219.87 |

2026-09-30 核对：上述 expected 使用拆分后会话的 `turn_index` 查询原始主请求列表，会把子代理／并行流映射到另一条请求。按 `source_trace_id + source_outer_idx + source_inner_idx` 查回原始长度，并将 0 输出按 AIPerf 升为 1，两个主点成功请求的 expected 与 actual 逐条完全一致，均值分别为 933.787792 / 898.350054。旧 expected 统计可由错误查法精确重现，不据其差额判断输出截断或幸存者偏差。

API key points（id 440239 / 439522）：

| 字段 | B200 | B300 |
|---|---|---|
| output_tput_per_gpu | 449.90783 | 795.03301 |
| input_tput_per_gpu | 58,837.60988 | 101,728.98023 |
| p90_ttft（s） | 16.20652 | 18.91578 |
| p90_itl（s） | 0.05025 | 0.04949 |
| kv_cache_pool_tokens | 21,943,624 | 43,335,400 |

注意：`kv_cache_pool_tokens` 是日志里的等效值，不是容量，见 P7。

### 1.5 命中率

| 来源 | 口径 | B200 | B300 |
|---|---|---|---|
| bmk | GPU / CPU / 合计 | 0.68223 / 0.23726 / 0.91949 | 0.899 / 0.01639 / 0.91539 |
| API | GPU / CPU | 0.71284 / 0.21867 | 0.91624 / 0.01363 |
| server.log | prefix hit / external hit | 78.6% / 74.4% | 94.1% / 17.8% |

external hit 是「GPU 未命中的 token 中由 CPU 命中的比例」。server.log 的口径与 bmk 不同，基准与口径 统一口径。

### 1.6 分 rank、router 与 server log

B200 各 rank：

| rank | prompt tokens | hit | 最大 KV 使用率 |
|---|---|---|---|
| e0 | 241.35M | 0.869 | 0.84 |
| e1 | 212.56M | 0.923 | 0.730 |
| e2 | 272.00M | 0.798 | 0.912 |
| **e3** | **268.11M** | **0.322** | **1.0（thrashing）** |
| e4 | 202.63M | 0.913 | 0.765 |
| e5 | 212.07M | 0.760 | 0.947 |
| e6 | 179.41M | 0.868 | 0.825 |
| e7 | 145.92M | 0.849 | 0.786 |

- **B300** 各 rank 比较均衡：hit 0.939–0.949，最大 KV 使用率 0.40–0.59。
- **Router**：
  - B200 共 16,295 个请求、2,180 个会话。各 rank 负载为均值的 0.84–1.17×（rank3 为 0.965×），没有会话跨 rank。
  - B300 共 30,172 个请求、4,617 个会话。
- **server.log**（B200 / B300）：
  - 权重：125.88 / 126.08 GiB；
  - 可用 KV：32.27 / 103.07 GiB；
  - KV 日志 token：每 rank 2,742,953 / 5,416,940；
  - MTP 各位置接受率：1.000 / 0.49 / 0.000；
  - running：169.1 / 368.4（是否为 8 个引擎的合计【待核实】）；
  - KV 使用率：50.4% / 32.3%。
- **历史 server 总量核对**：API 的 prompt / generation 总数分别为 B200 的 3,468,135,826 / 26,755,268，以及 B300 的 5,977,073,506 / 47,670,736。2026-09-30 实现按显式 endpoint 别名与全部 series labels 去重，优先直连引擎端口，不固定除以二：
  - B200：prompt 1,734,067,913，gen 13,377,634；
  - B300：直连引擎 prompt 2,988,512,269，router 转发 prompt 2,988,561,237，gen 均为 23,835,368。原稿的 2,988,536,753 是两端点的平均值；两端点存在异步抓取边界差异，应保留来源与差异，不能将平均值称为某个引擎的真实计数。

### 1.7 aiperf 口径与逐请求数据

- **流程**：
  - B200：warmup 1,598 s，然后 profiling 3,600 s + 30 s grace，取消 133 个请求；
  - B300：warmup 2,081 s，取消 212 个请求。
- **口径**：ITL = (latency − TTFT) / (OSL − 1)；输入吞吐包含命中的 token。
- **逐请求数据** `/tmp/infx-art/{b200,b300}/agentic/aiperf_artifacts/profile_export.jsonl`，字段有：conversation_id、turn_index、source_trace_id、x_correlation_id、agent_depth、parent_correlation_id、credit_issued_ns、request_start/ack/end_ns、benchmark_phase、was_cancelled、TTFT、latency、ISL/OSL、usage tokens、osl_mismatch_diff_pct。

| 计数 | B200 | B300 |
|---|---|---|
| 记录总数 | 16,162 | 29,960 |
| warmup 记录（其中错误） | 2,171（5） | 4,249（7） |
| profiling 记录（其中错误） | 13,991（0） | 25,711（15） |
| 用到的 trace（全部 / profiling） | 360 / 310 | 393 / 372 |
| agent_depth 0 / 1 | 6,922 / 9,240 | 12,394 / 17,566 |

- warmup 请求的 OSL 都是 1；profiling 请求的 osl_mismatch 全部为 0。
- OSL 均值低于历史 expected 已核实为聚合查表错映射；成功请求原始期望与实际输出逐条一致。窗口仍会影响选中／完成请求的组成，但不能用这个错误 expected 的差额证明幸存者偏差。
- trace 池 `dataset/agentx/traces.jsonl` 共 393 条（只读）。

### 1.8 量级估算【估算】

1. **每步耗时**：平均 step ≈ ITL × AL ≈ 92 ms（B200）/ 110 ms（B300）。纯 decode 步的下限约 20–25 ms，其中 MoE 读权重约 13 ms（每 rank 48 个专家 × 约 35 MB × 61 层）。
2. **放行步**：由 (7 × 25 + X) / 8 = 92 解得 prefill 放行步 X ≈ 560 ms。
3. **decode attention 的 SOL**（61 层，b=49）：64K / 122K / 262K / 1M context 分别约 3.46 / 3.68 / 4.22 / 7.24 ms。一个 512-token chunk 在同样四个 prefix 下约 7.9 / 9.1 / 12.1 / 28.9 ms。
4. **未命中量**：未命中约 37.9K tok/s，即每 rank 4,740 tok/s，每个放行步约 3,500 token。
5. **TTFT**：平均约 9.8K 个未命中 token → 19 个 chunk × 8 步 × 92 ms ≈ 14 s，实测 p90 为 16.2 s。
6. **KV 字节**：约 4,504 B/token，按 plain FP8 512 B/state、block 256 计：
   - CSA：30 × 512/4 = 3,840；
   - HCA：31 × 512/128 = 124；
   - MXFP4 indexer（只有 CSA 层有）：每 block 64 × 68 = 4,352 B，按 512 对齐为 4,608 B，30 × 4,608/256 = 540。
   - SimpleCPU 为 4,800 B/token（1,228,800 B / 256），差约 296 B/token，可能来自组间 page 对齐与填充【待核实，缓存容量与前缀命中任务 1】。v2 按 584 B/state 算出的 5,031 B/token 作废。
7. **权重**：约 123 GiB/rank（专家 95.7 GiB + 复制部分约 27 GiB），日志为 125.88 GiB。
8. **MoE FLOPs**：Pro 与 proxy 每层之比为 (6 × 3072) / (8 × 2048) = 1.125×。

### 1.9 旧模拟（64 GPU、proxy、empirical 外推、1,800 s）

- **结果**：
  - 输出吞吐 7.878 tok/s/GPU，合计约 504 tok/s；参考为 3,599 tok/s，总吞吐差 7.1×；
  - p90 TTFT 95.7 s，p90 TPOT 0.341 s，E2E 138.3 s；
  - hit 0.578 / 0.363。
- **时间分解**（单位 s，总计 20,560）：attention 14,006.1（68%），comm 4,890.9，moe 1,263.5，gemm 223.1，elementwise 174.9，lm_head 1.76，overhead 0。
- **MoE**：不均衡度 4.025，straggler 459.25 s，约占 MoE 时间的 36%，主要来自轮转 bug（见 P11）。

### 1.10 多并发参考曲线索引

采用下列已保存研究记录作为逐点配置的索引。B200 主点为 2026-08-22 的 440239；B300 主点为 2026-08-11 的 439522。日期与运行身份须和实际产物核对，不能用页面显示日期替代运行日期。TP/DP/EP、GPU 数、offload、kernel 和预算仍须从每个点的真实配置提取。

| GPU | 并发 | 并行配置索引 | 参考运行 ID |
|---|---:|---|---:|
| B200 | 1 | TP8 | 440244 |
| B200 | 4 | TP8 | 440242 |
| B200 | 8 | TP8 | 440241 |
| B200 | 14 | TP8 | 440240 |
| B200 | 32 | DEP8 | 440249 |
| B200 | 64 | DEP8 | 440246 |
| B200 | 96 | DEP8 | 440252 |
| B200 | 128 | DEP8 | 440247 |
| B200 | 160 | DEP8 | 440243 |
| B200 | 196 | DEP8 | 440239 |
| B300 | 1 | TP8，offload 关闭 | 439521 |
| B300 | 4 | TP8 | 439523 |
| B300 | 8 | TP4 | 439524 |
| B300 | 16 | TP4，offload 开启 | 439520 |
| B300 | 48 | TP4、EP4、attention DP | 439517 |
| B300 | 64 | TP4、EP4、attention DP | 439525 |
| B300 | 128 | DEP8 | 439518 |
| B300 | 256 | DEP8 | 439516 |
| B300 | 384 | DEP8 | 439522 |
| B300 | 512 | DEP8 | 439526 |
| B300 | 576 | DEP8 | 439519 |

这里 DEP8 表示 attention 按 DP 分配、专家跨 8 个 rank 分布；EP 与 DP 共用这些 GPU，不能把组大小再次相乘。TP4 点的实际 GPU 数和 replica 数需逐点核对。

另一次 B200 c196（439967，2026-08-16）的输入加输出吞吐比主参考高约 7.5%，属于不同运行，不替换 440239。B300 2026-08-21 的四 GPU TP4 曲线也不是这里同配置的重复样本。这些记录用于分析运行差异，不能混入同一参考身份。

---

## 2. 问题分析

下面分别记录实现缺口、参考证据、允许的近似和对应任务。

影响等级是研究排序用的估计，不是已测得的误差；最终用完整模拟与消融量化：

| 等级 | 含义 |
|---|---|
| 严重 | 误差 >2× |
| 高 | 20%–2× |
| 中 | 5–20% |
| 低 | <5% |

### P1 拓扑与 per-GPU 口径【严重】

- **现状**：
  - 模拟用的是 64 GPU 的 `data/clusters/64_b200/tp8_dp8_ep.json`（TP8×DP8、per_dp EP、跨节点），而参考是单节点 8 GPU DEP8。
  - `results/agentx-empirical-extrap/reference/inferencex_key_points.json` 里 num_gpus 写成 64，是错的。
  - `scripts/compare_agentx_results.py:59` 硬编码了 `(("b200",196),("b300",384))`：模拟输出 ÷64，参考输入却按参考的 num_gpus 除。
  - 21,943,624 被误当作 KV 容量。
- **影响**：per-GPU 指标直接差 8×；拓扑差异（跨节点通信、TP8 attention）还叠加在剩下的 7.1× 里。
- **AIC 做法**：按单个 replica 建模，要求 tp × attn_dp × cp == moe_tp × moe_ep。DEP8 对应 tp1 / attn_dp 8 / moe_ep 8。
- **实施方案**：
  - 新建 `data/clusters/8_{b200,b300}/dep8.json`：TP1×DP8×EP8，`expert_parallel_scope:"global"`。这已经由 `TokenSim/config/parallel_config.py:34` 和 `TokenSim/parallel.py:133-160` 支持，global 模式下同一 阶段的所有 rank 属于一个 EP 组。
  - key points 按 8 GPU 归一化；旧文件不动。
- **对应阶段**：基准与口径、完整模型与并行配置

### P2 proxy 模型是 V3 几何【严重】

- **现状**：`scripts/run_agentx_comparison.py:73` 和 `scripts/run_agentx_sweep.sh:49` 在用的 proxy 是 256 专家 / top-8 / inter 2048、58 层 MoE + first_k_dense 3、128 个 KV heads、uniform 路由。同两个脚本还写死了 block 64（`run_agentx_comparison.py:72`、`run_agentx_sweep.sh:41`）和 max_parallem_sum 2,000,000（:78、:50）。
- **影响**：
  - 每层 MoE FLOPs 差 1.125×；每 rank 的专家权重和显存都不同；
  - 前 3 层的 GEMM 形状错（实际是 hash-MoE，不是 dense）；attention 结构完全不同（见 P3）；
  - `cache_config.py:74-76` 按单一 dtype 算权重，连带 KV 容量和命中率都算错。
- **AIC 做法**：完整的 DSV4 模型（`/tmp/aic-full/aic-core/src/aiconfigurator_core/sdk/models/deepseek_v4.py`）：
  - 按主干层的 ratio {4,128} 组织 module op，generation 成本缩放用于近似 draft；不是把 MTP 的真实结构并入 HCA；
  - mHC 的 API dtype 标签不能代表模块内所有张量精度；本轮核对的 vLLM collector 使用 BF16 residual 和 FP32 参数。shared expert 用 gemm_quant，router/logits 为 BF16；
  - MoE 分布为 power_law 1.01（:83，:113-114），shared/routed 重叠（:360）。
- **实施方案**：从 HF config 生成完整 ModelSpec，弃用 proxy。
- **待验证**：按新模型计算的权重显存与日志 125.88 / 126.08 GiB 相差不超过 ±2%。
- **对应阶段**：完整模型与并行配置

### P3 CSA/HCA/SWA 被当成 GQA【严重】

- **现状**：`TokenSim/latency/operator_table.py:276` 的 `_compose` 按 GQA 组合 attention。旧结果里 attention 占 68% 的时间。
- **实际结构**（`vllm/models/deepseek_v4/attention.py`）：
  - 每层都有 SWA 128 的独立 cache（最近 128 个 token 的未压缩 KV）。
  - ratio>1 的层有 compressor（:354-357，:381），每 ratio 个 token 压缩成 1 个 state。
  - CSA（ratio 4，30 层）：indexer 在 ctx/4 个压缩 key 上打分，选出 top-1024，再对这 1024 个加上 SWA 128 做稀疏 attention。ratio 4 有专门分支（:289）；当 max_seq_len / ratio ≤ top-k 时跳过 indexer（:993）。
  - HCA（ratio 128，31 层）：对全部 ctx/128 个压缩 state 加上 SWA 128 做 attention，没有 indexer。
  - MTP 层：ratio 1，只有 SWA（:220-227）。
- **影响**：
  - dense GQA 的成本 ∝ ctx × heads；
  - DSV4 decode 的成本 ≈ 固定部分（top-1024 + SWA 128）+ indexer（∝ ctx/4）+ HCA（∝ ctx/128），随 ctx 增长慢得多；
  - prefill chunk 的 indexer 成本 ∝ chunk × prefix/4；
  - b=49、122K 时 61 层 decode attention 的 SOL 约 3.7 ms【估算】。
- **AIC 做法**：
  - 专用表：dsv4_csa / dsv4_hca 的 context 和 generation 表，vLLM 0.24.0，后端 FLASHMLA_SPARSE_DSV4。collector 计时的是整个 `DeepseekV4Attention` forward，用 dummy FP8 权重。
  - sidecar 表：`dsv4_hca_attn` / `dsv4_paged_mqa_logits`，Flash 模型（FP8，64 heads，TP1），isl 1..8192，step ≤1,048,575。
  - 另有 `dsv4_csa_topk_calib_perf`。
  - 路径：`/tmp/aiconfigurator/aic-core/src/aiconfigurator_core/systems/data/{b200_sxm,b300_sxm}/sparse_attention/vllm/0.24.0/`。
- **实施方案**：导入这些表，按 30 CSA + 31 HCA + MTP 逐层组合；表外用 util-hold（见 P4）。
- **计时范围**（v3 核实）：collector 测的是完整的 `DeepseekV4Attention.forward`（`collector/vllm/collect_dsv4_attn.py:522-541`），包含 fused_wqa_wkv（q LoRA 与 kv）、wq_b、compressor、indexer、attention 和 `_o_proj`（wo_a + wo_b）（`vllm/models/deepseek_v4/attention.py:420-456`）。权重是 dummy，但 dtype 按 checkpoint 的 fp8_block（表中 gemm_type）。
- **混合批次不是新增调度选择（本轮重新核对）**：TokenSim 的 `TokenSim/latency/operator_table.py:224-255` 已在一步内汇总输入处理和输出生成的 token，分别按形状估算 attention，再调用一次 `_compose`；AIC 的 `engine/runtime.rs:780-865` 也使用共用工作与两类 attention 的组合方式。因此沿用正常流程即可，不要求用户另选一种混合批次策略。接入 DSV4 模块实测表时，主要工程问题是核对计时范围：模块内已有的投影、压缩和 indexer 不能再次作为独立算子叠加。
- **indexer dtype**：collector 没有设置 `use_fp4_indexer_cache`，应为默认的 FP8，与参考的 MXFP4 不同【推断，见 P9】。
- **对应阶段**：算子耗时

### P4 context 超出算子表范围【高】

- **现状**（`TokenSim/operator_data/lookup.py`、`schema.py`）：
  - `LookupPolicy`（`lookup.py:88-90`）默认 interpolate=True、extrapolate="empirical"、max_extrapolation_ratio=16.0。它的 docstring（:66-84）读起来像是 analytical 为默认，与代码不符。
  - attention 的 seq 轴放宽到 64×（`schema.py:23` 的 `ATTENTION_SEQ_EXTRAPOLATION_RATIO`，用于 :117、:138）；逐轴覆盖在 `lookup.py:264`。
  - 超出 16× / 64× 时，`lookup.py:265-272` 抛出 `MissingOperatorDataError(kind="out_of_range")`。`table_first` 在 `operator_table.py` 的 `_query`（:492-511）/ `_query_moe`（:466-490）里捕获这个异常，改用纯 analytical 的 `_analytical`（:513-516）。这个值不锚定实测数据，所以估计在 16× / 64× 边界处会跳变。
  - 边界以内（`lookup.py:273-293`）有三种情况：
    - empirical 模式：边界实测值 × ratio^α（:277-281）；
    - α 拟合失败、低于下边界（ratio<1）或 analytical 模式：边界实测值 × analytical 增长比例（:282-287）；
    - 两者都不可用：log 轴上按 ratio 线性放大（:291-293）。
  - `_fit_scaling_exponent`（:317）在 log-log 空间拟合上 1/4 段，α 夹在 [0, 1]。
  - analytical 模式 = measured(b) × analytical(q) / analytical(b)，与 AIC 的 util-hold 结构相同。
- **AIC 表覆盖范围**：
  - Pro generation 表的最大 context：bs 1–8 为 64K，bs 16 为 32K，bs 32/64 为 16K，bs 128 为 6144，bs 256 为 4096，bs 512 为 2048，bs 1024 为 1024。
  - CSA context 表的最大 isl+prefix：bs 1–16 为 64K，bs 32 为 12288，bs 64 为 8192，bs 128 为 6144，bs 256 为 3072，bs 512 为 640，bs 1024 为 384。
  - B300 的 Pro CSA generation 只有 bs≥64 的数据。
  - 各表总行数（含 Flash 与 Pro，B200 / B300；仅 Pro 的统计见 P9）：dsv4_csa_context 5,840 / 5,704，dsv4_hca_context 5,840 / 5,840（isl≤65536，prefix≤65535）；dsv4_csa_generation 1,544 / 1,048，dsv4_hca_generation 1,544 / 1,544（context≤65536）；sidecar 各 1,959 行。
  - 同目录的 dsa 表：dsa_context 14,640 行（bs≤256，isl≤32768），dsa_generation 10,999 / 10,853 行（bs≤1024，step≤131072）。
  - 另有 `mhc_module_perf`；MoE 表 ep 1..128，tokens 1..16384。
- **与 workload 对比**：ISL 均值 122K、p90 256K、p99 约 620K，decode batch 49 / 96。几乎所有 decode step 都落在表外。
- **影响**：
  - 对 dense attention 拟合的 α 对 DSV4 没有物理意义；
  - 旧文档里「softmax 超线性」对稀疏 attention 不成立；
  - 表外误差会经闭环放大：step 变长 → 排队 → batch 变大。
- **AIC 做法**：
  - `perf_database/perf_interp.rs` 的查找顺序：精确匹配 → 范围内插值 → 范围外以实测为基准保持效率的理论成本外推（util-hold）→ 没有有效基准时报错。SOL 修正是部分插值路径的一环，ScatteredSites 用于 GEMM 类表，不是所有 attention 查询都经过的独立步骤。具体邻点与权重按目标表实现核对。
  - DSV4 的 SOL 在 `perf_database/dsv4.rs:1188` 的 `dsv4_attention_sol_ms`；indexer 的 query token 数为 context 时 b×s、decode 时 b（:1270-1273）。
- **实施方案**：
  - 移植 `dsv4_attention_sol_ms` 作为 analytical（TokenSim 已有 measured × analytical 的比例结构），KV 和 indexer 的字节做成参数（见 P9）；
  - 做 holdout 验证：用 ≤32K 的数据预测 64K；
  - 用 1M 范围的 sidecar 检查趋势，模型与计时语义不匹配处仅作辅助；
  - 审计原有 16× / 64× 边界外退回纯 analytical 的逻辑，保持实测基准与估算来源连续可追溯；缺少基准显式报告，不静默切换到不相关默认模型。
- **待验证**：Flash sidecar（64 heads）换算到 Pro（128 heads）的比例。
- **对应阶段**：算子耗时

### P5 调度器与 vLLM 不一致【高】

- **TokenSim 现状**：
  - `LLMPagedAttnScheduler`（`TokenSim/llm/llm_scheduler.py:176`）：`_admit`（:280）、`_step_tokens`（:311）、`_reserve_decode_slot`（:316，用 `queue.pop(-1)` 抢占最后一个请求）；token 预算为 `DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192`（:20）。
  - 没有 interval、threshold、defer、capacity_bound 和 MTP 相关预算。
  - max_num_seqs 实际是 max_parallem_sum = 2,000,000（`scripts/run_agentx_comparison.py:60-95`，`run_agentx_sweep.sh:50`）。
  - AgentX 运行参数：block 64，chunked prefill，prefill 池 round_robin，decode 池 least_gpu_memory。
- **vLLM 行为**（以下为旧 HEAD `8b6de0eb9a` 的核对记录；须在本轮只读基准 `30f5c01d4ebb` 重新定位）：
  - **节流**：基类的节流函数返回 False（`vllm/v1/engine/core.py:603-605`）；DP 下被覆写为 `interval > 1 and step_counter % interval != 0` 时节流（:2189-2196）。
  - **step_counter**：在 `_has_global_unfinished_reqs` 里每步 +1，旧 HEAD 每 32 步做一次 all-reduce（:2283-2292）；当前 `30f5c01d4ebb` 默认 16 步，且第 1 步也检查；所有 DP rank 上的值相同，wave 结束时重置（:2265）。没有请求的 rank 跑 `execute_dummy_batch`（:2225-2240）。
  - **scheduler**（`vllm/v1/core/sched/scheduler.py`）：
    - 预算（:529-532）；
    - defer 条件（:552-556）：节流步 && 非 capacity_bound && running 中存在非 prefill-chunk 的请求（即有 decode）；
    - defer 步中 running 里的 prefill chunk 也被推迟（:587-590）；
    - 先用 threshold 截断，再取 min(token_budget, input_budget − draft_slots)（:598-603）；
    - waiting 请求的 admission 也会 defer（:957-960），并同样应用 threshold（:993-995）；
    - 放行步结束时记录 `prefill_capacity_bound = bool(waiting)`（:1221-1224）。
  - **配置默认值**：threshold 0（`vllm/config/scheduler.py:70`），interval 1（:174）。MTP 下 draft_slots = 0（`vllm/config/speculative.py:1792`）。
- **影响**：
  - interval 8 + threshold 512 决定了 TTFT 的形状（§1.8：9.8K 未命中 token ≈ 19 个 chunk × 8 步 ≈ 14 s）；
  - max_num_seqs 49/96 与 2,000,000 的差别决定了 decode batch 上限和 KV 压力；
  - block 64 与 256 的差别影响命中粒度；
  - B300 的预算是 16384。
- **AIC 做法**：稳态模型，没有 interval，没有可借用的方案。`vllm_backend.py` 中：mix step 效率固定 1.0（:92）；mix step 的 gen token 数（:103）；prefill dispatch overhead = 层数 × 0.8 ms（:112）；TTFT 排队系数 min(1 + log2(b)/8, 2)（:118）。
- **实施方案**：按本地 vllm-note 源码移植以下内容，并用差分测试对照：
  - interval、threshold、defer、capacity_bound；
  - DP-only 节流 + 共享 step_counter；
  - MTP 下的 token 预算和 KV 预留；
  - max_num_seqs、block 256。
- **待验证**：本地源码中 MTP 下 decode 请求在 token 预算和 KV lookahead 中的具体计法。历史镜像差异记录为限制，不再以获取镜像源码作为前置条件。
- **对应阶段**：请求调度

### P6 各算子 dtype 不对【高】

- **现状**：
  - `_gemm`（`operator_table.py:370`）只用一个 dtype，MoE 也一样；attention 只看 KV dtype；
  - 权重显存按单一 dtype 计算（`cache_config.py:74-76`：`weight_bytes = dtype_bytes(model.dtype)`）；
  - 旧文档里「NVFP4 ✅」是错的，参考根本不是 NVFP4。
- **影响**：
  - FP8 block GEMM 与 NVFP4 GEMM 约差 1.5–2×【估算】；
  - MoE 的 w4a8 与 w4a4 查的是不同的表；
  - BF16 的 lm_head（129280 × 7168 × 2 B ≈ 1.85 GB）每次都要全读；
  - 权重显存会影响 KV 容量，进而影响命中率。
- **AIC 做法**：专家 fp4 映射为 `w4a8_mxfp4_mxfp8`；DSV4 强制 FP8 KV；MegaMoE 只支持 SGLang 后端（`deepseek_v4.py:102`，并要求 moe_tp=1（:104）、moe_ep>1（:106）；模块见 :165）。
- **实施方案**：逐算子 dtype map；权重显存按各部分 dtype 分别计算。
- **已核实**（v2 的待验证项）：
  - `data/operator_data/{b200_sxm,b300_sxm}/vllm/moe_perf.parquet` 有 DSV4-Pro 形状（7168 / 3072 / top-6 / 384）的 mxfp4 w4a8 数据，覆盖 EP8、token 1–16384，来源为 aiconfigurator vLLM 0.24.0；
  - 分布只有 balanced / power_law_1.01 / power_law_1.2，没有 "uniform" 行；
  - kernel 是 FlashInfer TRT-LLM（`vllm_mxfp4moe_flashinfer_trtllm_mxfp4_mxfp8_…`），不是 mega-MoE，不含 dispatch / combine；
  - 表里的 num_tokens 是 EP 组的全局 token 数（AIC 查询时乘 attention_dp_size，`operators/moe.rs:196-202`），与 TokenSim 的 `moe_token_multiplier`（`operator_table.py:216-218`）一致。
- **对应阶段**：完整模型与并行配置（权重显存）、算子耗时（算子 dtype）

### P7 KV 容量、共享块池与前缀复用【高】

**现状。** `TokenSim/config/cache_config.py` 用单一 KV 几何与权重 dtype 推算容量，latent 路径也套用 K/V 两份的系数，另有显式容量覆盖与 host swap 上限。DSV4 每个 latent state 只存一份，需增加模型专用布局；普通 GQA 的 K、V 两份仍须保留。现有命中逻辑以 trace 的 64-token hash 块为引擎块，缺少混合缓存组、最多命中 n−1 个 token 和 retention 边界。

**池组织。** 当前只读 vLLM 的 `KVCacheCoordinator` 创建一个 `BlockPool`，各缓存组共用本 worker 的物理块池。各组覆盖的 token 数、窗口和保留状态不同；不能为每组重复分配整份显存预算。不同 worker 的 HBM 独立；SimpleCPU 的本地 CPU 缓存与 Mooncake 的共享 DRAM 存储另行建模，不能合并成无搬运成本的 HBM。

**历史日志的容量解释【日志核对与布局重建】。** 下面的组形状能解释历史日志，但不能代替按当前源码推导分组与对齐。尤其两个 SWA 组的来源、page padding 和异步在途预算仍需逐项核实。

| 缓存组 | 历史重建的 block token 数 | 窗口 | 一个最大长度请求的块数上限 |
|---|---|---|---|
| 压缩 MLA KV 与 indexer | 256 | 全上下文 | ceil(1,048,576 / 256) = 4,096 |
| SWA，两组 | 64 | 128 | 每组 ceil((127 + F) / 64) + 1 |
| ratio 4 compressor 状态 | 4 | 8 | ceil((7 + F) / 4) + 1 |
| ratio 128 compressor 状态 | 8 | 128 | ceil((127 + F) / 8) + 1 |

取 F = 两个在途 batch 的 token 预算，即 2 × max-num-batched-tokens，可以重建：

- B200：每个最大长度请求上限 4,096 + 2 × 259 + 4,099 + 2,065 = 10,778 块；日志池为 28,194 块。
- B300：上限 4,096 + 2 × 515 + 8,195 + 4,113 = 17,434 块；通常为 90,064 块，DP4 为 90,062 块。
- 历史物理块为 1,228,800 B。日志等效 token 数为 floor(池块数 × 1,048,576 / 每请求块数上限)，得到 2,742,953 / 5,416,940；B300 DP4 为 5,416,820。这是最大长度请求的等效容量，不是任意 token 可占用的总数。
- 池块数比例约 3.194，而每请求上限比例约 1.618，因此日志等效值之比约 1.975。组窗口和在途预算解释了两种比例为何不同。

**容量来源已确认。** 用硬件容量、配置的显存利用率、完整模型权重、激活、通信、运行时与 graph 预算推导可用 HBM，再按实际布局计算块数。一个占用项只扣一次；历史 profiler 的负 non-torch 增量不能作为负物理占用输入，也不能在已包含 graph 的项上重复扣 graph。CPU 容量按硬件预算与后端开销计算。上面的池块数及 289,815 个 CPU 块都只用于误差比较，禁止作为覆盖值。

**布局与命中。**

- 主 KV 的 FlashInfer SM100 plain FP8 路径为 512 B/state；FlashMLA 的 fp8_ds_mla 布局为 584 B/state。MXFP4 indexer 为 68 B/state；page 对齐、压缩比与各组共享物理块的填充另算。4,504 B/token 的有效载荷估算与历史物理块折合的 4,800 B/token 不能互相替代，差额仍需核对分组和 padding。
- 原始 trace 的每个 hash 描述 64 token。目标主缓存块为 256 token 时，以连续四个 hash、前缀链和回放身份组成一个完整块身份；不足一个块的尾部不能冒充完整可复用块。其他曲线点按其配置换算，不能改写原始数据。
- 请求最多命中 n−1 个 token，随后按块对齐；混合模型需要该边界上所有所需缓存组状态均可恢复。不能只查主 KV 就宣称整个前缀命中。
- 按当前源码实现 retention：请求 prompt 末尾的可达边界、32,768-token 周期边界以及实现支持的共享前缀分叉点。滑动窗口释放旧状态时保留这些边界所需的状态；历史镜像无法确认的差异在报告中单列，不为此下载旧镜像。
- 准入检查覆盖完整输入及已承诺的在途 prefill 占用；物理分配仍随 chunk 推进。MTP lookahead、滑动窗口释放、异步完成和取消都影响实际空闲块数，不能只用静态 token 容量限流。

**AIC 的适用范围。** `get_kvcache_bytes_per_sequence` 和最大长度二分可用于核对字节量，不模拟历史前缀、块池、驱逐与 retention，不能直接替代缓存管理器。旧研究的无限容量静态命中率与正式一小时结果使用的会话、路由和窗口不同，不能把二者差额全部归因于容量。

**对应任务：** 缓存容量与前缀命中、CPU 缓存卸载、请求调度。

### P8 MTP K=3 / AL 2.49 未建模【严重】

- **现状**：TokenSim 没有投机解码。
- **参考行为**：
  - synthetic 接受率为 [1.0, 0.49, 0.0]（`vllm/config/speculative.py:544-551` 的最小方差分配，与日志一致），所以每步产出 2 个 token（p=0.51）或 3 个 token（p=0.49），均值 2.49。
  - target forward 时每个 decode 请求有 1+K = 4 个 query token（server.log 中 `indexer.py:298` 的 "next_n=4" 可以印证）。
  - draft 的形状与次数按当前只读源码的 `AutoRegressiveSpeculator`、MTP `forward` 和 `compute_logits` 路径核对；初始 pass、后续单 token pass 与输出投影分别计时，历史镜像差异单列。
- **对你的方案的判断**：进度侧可行（AIC 也这样处理），成本侧需要补两部分：
  1. verify：每个请求最多 4 个 query；各模块实际处理行数、KV 共享与输出位置筛选按源码核对，不能把所有模块都乘四；
  2. draft：MTP 模块与输出投影分别估算，包含源码实际执行的所有 pass。
- **量级**【估算】：
  - 196 个 token 时 MoE 仍以读权重为主，×4 的行数影响不大；
  - indexer 和稀疏 attention 的 ×4 是主要增量；
  - BF16 lm_head 每次读约 1.85 GB，每步至少 4 次，合计约 1 ms；
  - 只做「跳过」、不补成本时，ITL 可能低估 20–40%。
- **AIC 做法**（Python 与 Rust 均已核实）：
  - **进度**：`decode_tokens_per_iteration = 1 + nextn_accepted`（`/tmp/aic-full/src/aiconfigurator/sdk/predict.py:140-141`）。取值范围限制为 [1, nextn+1]，decode_iterations = 1 + (osl − 1) / 进度（`base_backend.py:1199-1211`）。nextn>0 时必须显式给出接受数（`/tmp/aic-full/src/aiconfigurator/sdk/speculative.py:28-50`）。
  - **成本 1**：Rust 引擎把 decode batch ×(nextn+1)（`aic-core/rust/aiconfigurator-core/src/engine/runtime.rs:540-571`，generation 阶段）；mixed step 的 pass 1 按 ctx + gen×(nextn+1) 个 token 计（:780-790），pass 3 的 decode attention batch 也 ×(nextn+1)（:845）。也就是说，AIC 把 q=4 当作 4 倍 batch 处理。
  - **成本 2**：DSV4 的所有 generation 算子再乘 (L + nextn)/L = 64/61（`helpers.py:1033-1041` 的 `mtp_scale_factor`，`deepseek_v4.py:82`、:293-374），即把 draft 当作 K 个额外的层。
  - **激活显存**：decode 部分 ×(nextn+1)（`base_backend.py:1738-1758`）。
  - Python 侧 `base_backend.py:1080` 的 `decode_query_tokens` 只是报告字段；`session.rs:133-138` 的注释说明 (nextn+1) 由 Engine 施加，与 Python 一致。
- **已确认的简化方案**：
  - AL 推进：随机产出 2 个 token（51%）或 3 个 token（49%），期望 2.49；最终使用三个模拟随机种子；
  - 草稿按当前 MTP 模块及输出投影的实际调用估算，内部查表或计算理论成本；正式方案不使用主干总耗时 ×64/61 代替这些调用；
  - 校验以 Pro 实测为基准，分别计算额外 query 的计算增长与按实现可复用的 KV 读取；分项 q=1/q=4 表仅在语义匹配时辅助校验，不直接视为完整 Pro 的同条件配对数据；
  - KV lookahead K；ITL 按 aiperf 口径统计。
- **对应阶段**：投机解码（算子耗时任务 7 处理 verify q=4 的 attention）

### P9 FlashInfer 与 FlashMLA 的差异（kernel、KV 布局、indexer dtype）【中】

**2026-09-30 实测表复查：** AIConfigurator 有可用的 GPU 实测数据，应作为耗时估算的基础。此次直接读取本地数据仓库 `f254959eb89e2f206b8f9a77051644d7c1cbdb89` 的 Parquet，并核对 collector：`/tmp/aiconfigurator/collector/vllm/collect_dsv4_attn.py:522-543` 实际执行 attention 模块、预热并用 CUDA graph 测量，输出 `latency_ms`。这些是模块实测，不是完整 AgentX 服务的端到端耗时。

下表仅统计 `model=deepseek-ai/DeepSeek-V4-Pro` 的 vLLM 0.24.0 数据，未混入 Flash 模型：

| 实测模块 | B200 记录数 | B300 记录数 | 表中后端 |
|---|---:|---:|---|
| CSA 输入处理 | 2,920 | 2,784 | FLASHMLA_SPARSE_DSV4 |
| HCA 输入处理 | 2,920 | 2,920 | FLASHMLA_SPARSE_DSV4 |
| CSA 输出生成 | 772 | 276 | FLASHMLA_SPARSE_DSV4 |
| HCA 输出生成 | 772 | 772 | FLASHMLA_SPARSE_DSV4 |

数据目录：`/tmp/aiconfigurator/aic-core/src/aiconfigurator_core/systems/data/{b200_sxm,b300_sxm}/sparse_attention/vllm/0.24.0/`。包含 TP1/2/4/8，但形状覆盖不完整，例如 B300 Pro CSA generation 缺少 batch<64。还有 Flash 模型的 attention/indexer 分项表，以及 TRT-LLM、SGLang 模块表，可检查趋势；使用前需核对模型、计时范围和执行路径。

**能参考，但不能直接宣称完全匹配参考实验。** 上述 vLLM DSV4 表记录的是 FlashMLA，AgentX 主参考配置使用 FlashInfer。此次检查的数据中未找到同一 DSV4 配置、同一形状的两后端配对实测，不能据此确定两者的独立加速系数。旧文档所称「用 V3.2 DSA 双后端数据直接求比值」也需撤回：B200/B300 的 V3.2 context/generation 表按模型、dtype、heads、batch、长度和 TP 配对，均没有同条件的双后端记录；BF16 KV 下 FlashMLA 的 heads 为 32/64/128，FlashInfer 为 8/16，FP8 KV 记录则均为 FlashInfer。

**已确认采用的方案：** 以实测表为基础，分别列出可以直接匹配、需要插值、需要外推及配置不匹配的查询。缓存格式修正先核实实际布局，并明确修正部分是估算；不能把配置不同的实测比值当成纯 kernel 差异。具体误差修正的公式与验证依据写入实现报告。

- **事实**：
  - AIC 用 FLASHMLA_SPARSE_DSV4 采集；参考用的是 FLASHINFER，并开启 prefill query 量化和 FP4 indexer。
  - SM100 上 prefill 和 decode 都调用 `flashinfer_trtllm_batch_decode_sparse_mla_dsv4`（`vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py:519/544/796/915`），o_proj 用 `deep_gemm_fp8_o_proj`。
- **AIC 表的采集配置**（v3 核实）：
  - 后端与 KV dtype：dsv4_csa / dsv4_hca 各表的 kernel_source 都是 FLASHMLA_SPARSE_DSV4，kv_cache_dtype 都是 fp8（表中字段）。collector 只接受 fp8（`collector/vllm/collect_dsv4_attn.py:957-958`），后端交给 vLLM 自动选择（:14、:963-965）。
  - 主 KV 布局：按 HEAD 源码，FlashMLA 类默认 `use_fp8_ds_mla_layout=True`（`attention.py:148`），fp8 会改写为 fp8_ds_mla（:90-125），即 448 B NoPE + 128 B RoPE + 8 B scale = 584 B/state（:815-817，按 576 B 对齐）【推断：0.24.0 未核实】。
  - indexer cache：collector 没有设置 `use_fp4_indexer_cache`，应为默认的 FP8，132 B/state（`attention.py:932-937`）【推断】。sidecar `dsv4_paged_mqa_logits` 明确把 indexer KV 转成 FP8（`_kv_cache_cast_to_fp8_indexer`，:579、:625）。
- **查表值与参考的差异**：

  | 项 | AIC 表 | 参考 |
  |---|---|---|
  | attention kernel | FlashMLA sparse | FlashInfer TRT-LLM sparse |
  | 主 KV | fp8_ds_mla，584 B/state【推断】 | plain FP8，512 B/state |
  | indexer cache | FP8，132 B/state【推断】 | MXFP4，68 B/state |

- **量级**【估算】：B200 上一个 CSA 层的一次 decode，取参考的形状：49 个请求 × 4 个 query token（MTP）= 196 个 query token。
  - 估算口径：
    - 读取量和计算量按 AIC SOL 的组成（`dsv4.rs:1188-1309`）；
    - 规格取 AIC 的 B200：7.7 TB/s、FP8 4.5 PFLOPS、FP4 9 PFLOPS；
    - indexer KV 每个请求读一次；
    - 稀疏 attention 每个 query token 读 1,152 个 state（topk 1024 + window 128）。
  - 与 ctx 无关的读取：权重约 395 MB（投影按 fp8_block，wo_a 按 BF16），激活约 20 MB。
  - 稀疏 attention：512 B 时约 116 MB，584 B 时约 132 MB。
  - indexer 读 49 × ctx/4 个 state：ctx 122K 时 MXFP4 约 102 MB、FP8 约 197 MB；1M 时约 874 MB 和 1,696 MB。
  - 表的配置相对参考配置的比值：

    | ctx | 读取字节比 | roofline 比（参考的 MXFP4 logits 按 FP4 算力） | roofline 比（按 FP8 算力） |
    |---|---|---|---|
    | 122K | 1.18 | 1.18 | 1.17 |
    | 256K | 1.30 | 1.30 | 1.17 |
    | 1M | 1.60 | 1.60 | 1.17 |

  - MTP 下 indexer 的计算时间和读取时间接近。例如 1M 时，参考的 indexer 计算按 FP4 算力约 94 µs、按 FP8 算力约 188 µs，读取约 113 µs。
  - 这些假设下的理论比例约为 1.17–1.6，不能表述为实测模块必然偏慢的倍数，也不能直接缩放整个模块；须核对实际 logits kernel、KV 复用和各分项占比。
  - HCA 没有 indexer；该字节量估算下主 KV 布局影响在 3% 以内，未包含后端 kernel 等其他差异，不能作为总误差上界。
  - 容量按 512 B 算（缓存容量与前缀命中）。
- **AIC 数据与估算方式**：
  - dsa（V3.2）表虽然包含 FLASHMLA_SPARSE 和 FLASHINFER_MLA_SPARSE，但此次核对没有同条件的配对点，不能直接提取后端时延比；参见本节实测表复查。
  - SOL 模型按 512 B/state（`dsv4.rs:1292`）和 FP4 indexer（每个 state 64 B，不含 scale，:1274-1276）计算，接近参考配置，但与实测表的配置不同。util-hold 外推时，绝对值来自实测表，增长率来自 SOL【推断】。
- **实施方案**：
  - 优先使用匹配的 DSV4 实测数据作为基础；独立 kernel 系数尚无同条件实测支持，先列为不确定性，不用 V3.2 非配对数据硬算系数（决策 23.1）；
  - 缓存格式：参数化 KV 与 indexer 的字节量，按已核实的模块组成估算受影响分项，保留原始实测、估算修正与公式；不能用一个字节比无差别缩放所有计算（决策 23.2）。
- **对应阶段**：算子耗时任务 8

### P10 mHC 未建模【中】

- **现状**：TokenSim 只有 residual add。DSV4 的 hc_mult 为 4，每层要读写 4 × 7168 的残差流（mHC pre/post + sinkhorn）。
- **AIC 数据（本轮更正）**：vLLM `mhc_module_perf` 是实际 TileLang 模块计时，不能统称为「BF16 算子替代 FP32 算子」。`/tmp/aiconfigurator/collector/vllm/collect_mhc_module.py:112-120` 使用 BF16 residual，但 fn/base/scale 均为 FP32；这与本地 vLLM 的 FP32 mHC 参数不构成旧稿所说的直接 dtype 冲突。两种 GPU 的表均有 hidden_size=7168、hc_mult=4 的 70 条记录。
- **正常导入与剩余近似**：读取 `num_sites=2` 和 pre/post 的计时范围，避免按两处已合计的耗时重复乘次数。collector 按 pre/post 分开测量，源码明确注明与 serving 中 fused post-pre/norm 的路径存在差异；该融合差异需要标注，不能把其他硬件上的误差数字当成 B200/B300 的保证。使用该实测表属于已确认的实测优先方案，无需再询问是否接受并不存在的 BF16/FP32 权重替代。
- **对应阶段**：算子耗时任务 11

### P11 MoE 负载分布【中–高】

- **现状**：
  - `TokenSim/moe/routing.py:135` 的 `_uniform_histogram` 按 `index % num_experts` 从 0 号专家开始轮转填充。`histogram_for_step`（:74-82）又是**逐个请求**生成直方图再相加，所以每个请求都从 0 号专家开始填：
    - 一个 decode 请求只有 1 个 token、top-k 个 route，全部落在 0 ~ top-k−1 号专家上，即同一个 EP rank；同一步里所有 decode 请求都叠在这个 rank 上；
    - 长 prefill chunk 的 route 数是专家数的很多倍，基本均匀；
    - 旧结果的不均衡度 4.03 就是这两类步混合后的结果。
  - 即使改成先汇总整步的 route 数再填，余数仍会集中在低编号专家。旧例使用 proxy 的 top-8／256 专家，不能代表完整模型。按 Pro 的 49 个请求 ×4 query ×top-6 = 1,176 个 route、384 专家、EP8，整步简单填充仍可能把余数集中在低编号专家；需分散余数并只将直方图用于统计（决策 25）。
  - 分布候选见 `operator_table.py:47`；未知路由时默认 ("uniform","balanced")（:220）。
  - `_moe_layers`（`operator_table.py:423-464`）的计算分两步：先按全局 token 查一次表（:446）；再按直方图算出的不均衡度查「最忙 rank」（:453），两者之差作为 straggler（:454）。
  - v2 说「倾斜被重复计入」，不准确，更正如下：
    - 旧运行的路由是 uniform，候选为 ("uniform","balanced","power_law_1.01")（:48）；
    - 表里没有 uniform 行，所以实际查的是 balanced 表；
    - 旧结果的 straggler 459 s（约占 MoE 时间的 36%）主要来自上面的轮转 bug，不是重复计入。
  - 只有「power_law 表 + straggler」同时使用时才会重复计入倾斜。
- **AIC 做法**：power_law 1.01，不另加 straggler。
- **实施方案**：
  - 修轮转 bug；
  - 用 power_law 表时关掉 straggler；
  - 使用已确认的 `power_law_1.01` 实测分布（决策 25）；
  - 前 3 层 hash 层先按同一分布近似。
- **对应阶段**：算子耗时任务 9

### P12 通信与 mega-MoE【高】

- **现状**：
  - 旧结果 comm 4,891 s，占 24%，来自跨节点的 64 GPU 拓扑。
  - TokenSim 的 MoE 表是 FlashInfer TRT-LLM kernel，只含专家计算，all-to-all 由 communicator 另算（见 P6）。
- **参考 kernel**：
  - 参考是单节点 NVLink。vLLM DSV4 在 `deep_gemm_amxf4_mega_moe` 下调用 `deep_gemm.fp8_fp4_mega_moe`（`vllm/models/deepseek_v4/nvidia/model.py:723`、:734），dispatch + 专家 GEMM + combine 在一个 kernel 里完成，combine 为 FP8（`VLLM_DSV4_MEGA_FP8_COMBINE=1`）。
  - shared expert 可以融合进这个 kernel（`has_fused_shared_experts`，:556），否则单独计算（:1025-1029）。426e59f 走哪条路径【待核实】。
- **AIC 数据**（`/tmp/aiconfigurator/aic-core/src/aiconfigurator_core/systems/data/{gb200,gb300}/moe/sglang/0.5.10/dsv4_megamoe_module_perf.parquet`）：
  - **形状**：正好是 DSV4-Pro（7168 / 3072 / top-6 / 384）。kernel 为 `deepgemm_megamoe`，dtype 为 `w4a8_mxfp4_mxfp8`，与 vLLM 调用的是同一个 DeepGEMM kernel。
  - **覆盖**：
    - GB200 EP {4, 8, 16, 32}，GB300 EP {4, 8, 12, 16, 24, 32}；
    - EP8 下 generation 的 num_tokens 为 1–512（GB200）/ 1–2048（GB300），context 为 1024–32768；
    - 分布有 balanced / power_law_1.01 / power_law_1.2 / power_law_sampled_1.9。
    - 参考的纯 decode 步每 rank 为 49 × 4 = 196（B200）/ 96 × 4 = 384（B300）个 token，混合步最多约 8K / 16K，都在覆盖范围内。
  - **num_tokens 的含义**：每个 rank 的本地 token 数，查询时不乘 attention_dp_size（`perf_database/dsv4_megamoe.rs:108-112`）；表里另有 `global_num_tokens` 列（= ×EP）。常规 MoE 表的 num_tokens 则是全局值。
  - **计时范围**（collector `collector/sglang/collect_dsv4_megamoe.py`）：
    - 包含 SGLang 的 pre-dispatch（写入对称 buffer）和 `deep_gemm.fp8_fp4_mega_moe`；
    - 不含 gate / top-k；
    - shared expert 未融合（num_fused_shared_experts 0）；
    - 开启 CUDA graph。
  - **缺口**：B300 SXM 没有这张表；B200 SXM 在 sglang 0.5.16 下有一张，但形状不对（3584 / 896 专家 / top-16）。
- **量级对比**：GB200 mega-MoE 与 TokenSim B200 FlashInfer 表对比，EP8，按全局 token 计，单位 µs。

  | 全局 token | balanced：mega-MoE | FlashInfer | 比值 | power_law_1.01：mega-MoE | FlashInfer | 比值 |
  |---|---|---|---|---|---|---|
  | 8 | 107.1 | 63.0 | 1.70 | 112.8 | 74.8 | 1.51 |
  | 64 | 351.9 | 276.5 | 1.27 | 253.4 | 172.9 | 1.47 |
  | 256 | 352.2 | 282.7 | 1.25 | 293.7 | 254.2 | 1.16 |
  | 512 | 355.6 | 421.2 | 0.84 | 312.2 | 332.8 | 0.94 |
  | 1024 | 362.8 | 336.8 | 1.08 | 329.5 | 323.6 | 1.02 |
  | 2048 | 371.7 | 590.4 | 0.63 | 365.5 | 399.6 | 0.91 |
  | 4096 | 407.9 | 464.7 | 0.88 | 512.4 | 539.4 | 0.95 |
  | 8192 | 508.4 | 534.4 | 0.95 | 779.1 | 788.2 | 0.99 |
  | 16384 | 911.6 | 972.3 | 0.94 | 1393.7 | 1368.3 | 1.02 |

  - mega-MoE 的 4096 及以下取 generation 行（rank 本地 ≤512），8192 / 16384 取 context 行（rank 本地 1024 / 2048）。
  - 全局 token ≤128 时 mega-MoE 慢 1.3–1.7×，256 时慢 1.16–1.25×。
  - ≥512 时，power_law_1.01 下两者相差 9% 以内；balanced 下为 0.63–1.08，主要因为 FlashInfer 表在 512 和 2048 两点非单调（比更大的 token 数还慢）。
  - 参考的纯 decode 步：B200 约 49 × 4 × 8 ≈ 1,568 个全局 token，B300 约 96 × 4 × 8 ≈ 3,072 个，都落在两者接近的区间。
  - mega-MoE 的时间已含 dispatch/combine，却与不含通信的 FlashInfer 时间相当。所以换表的主要作用是不再单独计 all-to-all（旧结果 comm 占 24%）【推断】。
- **SKU 换算**：同一个 vLLM FlashInfer MoE kernel（EP8，每种分布 27 个 token 点，1–16384）下，GB / B 的时延比：
  - power_law_1.01：GB200 / B200 为 0.91–1.11，GB300 / B300 为 0.89–1.04，共 3 个点超出 ±10%。
  - balanced：分别为 0.86–1.43 和 0.78–1.30，有 10 个和 5 个点超出 ±10%。
    - 各表自身的非单调点（比下一个更大的 token 数慢 5% 以上）：B200 在 512、2048；GB200 在 384、512、1536、4096；B300 在 320、512；GB300 在 384、512、1536、3072。
    - 超差点与非单调点的重合情况：GB200/B200 的 10 个点中有 4 个（384、512、1536、4096）本身就是非单调点；GB300/B300 的 5 个点中有 4 个（320、384、1536、3072）。
    - GB200/B200 剩下的 6 个点解释不了：1、2、4 token 约 0.88–0.89；320、1024、3072 分别为 1.27、1.20、1.35。
    - 所以 balanced 表噪声较大，只有一部分能用非单调点解释。power_law_1.01 的比值更稳定。
  - 据此估计，GB 系列的 mega-MoE 表直接用于 B 系列，误差约 ±10%（按 power_law_1.01）【估算】。
  - 这个比值取自 FlashInfer kernel。mega-MoE 还含 NVLink 通信，GB200 与 B200 的每 GPU NVLink 带宽相同【推断】。
- **仍有的差异**：
  - SGLang 与 vLLM 的 pre-dispatch 不同；
  - shared expert 是否融合；
  - FP8 combine 是否与 collector 相同【待核实】；
  - DP 各 rank 本地 token 数不同时，kernel 时间可能由最大的 rank 决定【推断】。
- **shared expert 执行路径复查**：本地 `vllm/models/deepseek_v4/nvidia/model.py:1109-1126` 的未融合 MegaMoE 分支先调用 routed experts，再调用 shared expert；不能无条件套用 AIC 的完全重叠假设。融合是否可用还取决于 `has_fused_shared_experts` 和 DeepGEMM API（:581-597）。因此按配置与源码确定融合／独立路径，再核对表是否已包含 shared expert，不能让用户选择一个与执行路径冲突的计时方法。
- **AIC 做法**：MegaMoE 只在 SGLang 后端建模。`aic-core/src/aiconfigurator_core/sdk/models/deepseek_v4.py:99-106` 在请求使用 MegaMoE 时对其他后端报错，并要求 moe_tp=1、moe_ep>1；这不等于 AIC 的 vLLM 后端完全不支持 DSV4。
- **实施方案**：
  - 匹配融合 kernel 的点借用 GB200 / GB300 表，初始换算系数 1.0，记录跨硬件和框架近似；已含 dispatch/combine 时不再加一次 all-to-all；
  - 其他参考配置按实际 kernel 的计算与通信路径分别建模，不把融合表用于不匹配的点；shared expert 按实际融合状态计时；
  - 缺失或未确认的数据范围显式报告，借表误差通过敏感性和完整主点验证评估，没有既定 ±10% 保证。
- **对应阶段**：算子耗时任务 10

### P13 GPU 缓存与外部 DRAM 缓存的分层【中】

- **必须区分的两层**：GPU 内部 HBM 保存计算所需的 KV；外部 DRAM 保存可恢复的 KV。每个 DP worker 的本地 HBM 管理与 Mooncake 的共享 DRAM 存储可以同时存在，不能把两层直接相加成无搬运成本的 GPU 显存。
- **博客的 Mooncake 部署**：各 worker 的 Mooncake Store 客户端贡献一部分主机 DRAM，组成共享的外部缓存池；Store 管理放置和驱逐，Transfer Engine 搬运数据。出处为博客 “A Brief Introduction to the Distributed Inference Ecosystem”；本地 `vllm/distributed/mooncake_store.py` 的 embedded 模式也描述每个 rank 贡献内存段。
- **目前两个参考点的实际后端**：保存的 `/tmp/infx-art/{b200,b300}/agentic/vllm_command.txt` 均明确使用 `SimpleCPUOffloadConnector`。历史脚本同时支持 `vllm-simple` 和 `mooncake` 分支；脚本支持 Mooncake 不代表某次运行启用了它。
- **对照要求**：每个实验点记录其真实后端、容量、存储布局与路由策略。不能以 Mooncake 共享池运行结果直接宣称复现 SimpleCPU 的同配置精度。
- **已确认的容量原则**：HBM 可用空间、DRAM 可用空间及块数均由模拟器计算；日志中的块数只作对照。
- **已确认的实现范围**：同时支持 SimpleCPU 与 Mooncake，共用缓存统计与实验接口，分别保留本地 CPU 池和外部共享 DRAM 池的语义。实验配置选择后端，不能将两种后端简化为仅替换名称。
- **AIC 做法**：没有对应的请求级 offload 模拟可直接复用。
- **对应工作**：缓存容量与前缀命中、CPU 缓存卸载、GPU 同步与请求路由。

### P14 step overhead、eager 混合步与 DP lockstep【高，可能是最大误差源】

- **现状**：
  - `data/devices/{b200_sxm,b300_sxm}.yaml:42` 的 step_overhead_us 为 0.0（grade D）；
  - 没有区分 CUDA graph 步和 eager 步；
  - DP 各 rank 各自独立推进（`TokenSim/placement/policies.py:47` 的 `DataParallelWorkerPool`），没有 lockstep，也没有 dummy batch。
- **参考行为**：
  - FULL_DECODE_ONLY 只给纯 decode 步用 CUDA graph；混合步走 eager，且 compile mode 0；
  - DP 各 rank 每步同步，步长取最慢的 rank；空闲 rank 跑 dummy batch；step_counter 全局共享。
- **影响**：
  - 按该粗略假设，约 76% 的时间花在放行步（§1.8 第 2 条）【估算】；
  - 推断 lockstep 加上 rank3 的 capacity_bound，使 B200 的 p90/p95 ITL 偏高【待验证】。
- **AIC 做法**：
  - prefill dispatch overhead = 层数 × 0.8 ms（`vllm_backend.py:112`）；
  - PDL 系数 0.9（`models/deepseek.py:94`，用于 :291；`models/deepseek_v32.py:186`，用于 :324）。AIC 的 DSV4 模型没有用 PDL 系数（`models/deepseek_v4.py` 和 `helpers.py` 里都没有）；
  - 假设各 rank 负载均衡。
- **实施方案**：
  - lockstep 步长取 max，空闲 rank 跑 dummy；
  - overhead 设 graph / eager 两个参数，只用 B200 标定，不按 SKU 调参。
- **对应阶段**：GPU 同步与请求路由（lockstep）、算子耗时任务 13（overhead）

### P15 会话到 GPU worker 的路由【中–高】

现有 `TokenSim/workload/agentx.py` 把 chat_id 设为 lane 身份，使同一回放通道上的不同会话固定到同一个 rank。参考主点使用 consistent_hash，以会话自身的 correlation ID 分配 worker；subagent 也使用自己的 ID，而不是继承父会话的 worker。

已确认在回放与 worker 调度器之间增加简单 router。会话拆分先按 P16 对齐，再按逐点配置选择路由策略；记录请求、会话、回放轮次、worker 及在途量，不增加参考中不存在的动态迁移。拓扑变化时，router 的目标是可接收请求的 worker／replica，并非任意单张 GPU。

可选的固定实测 worker 模式使用 router.log 与原始身份建立映射，用于分离路由与缓存误差。报告可映射覆盖率与缺失记录，不能伪造实测归属；正式曲线仍使用内部 router。静态无限缓存分析只用于定位路由影响，不能作为一小时命中率验收。

### P16 AIPerf 请求回放与预热【高】

AIPerf 是外部基准工具。本轮参考其固定版本源码，在 TokenSim 内部实现相同的解析和回放语义，不安装或运行其数据加载器。每条 trace 是包含主会话与 subagent 的请求树；每条回放通道同时负责一棵树，树完成后从共享顺序抽样器领取下一条。并发通道数不等于每时刻发出的请求数。

| 环节 | 需要实现的语义 | 依据 |
|---|---|---|
| 会话和依赖 | 保留原始请求，区分主会话、subagent、辅助及并行流；从时间戳与历史 API 耗时推导前驱与等待。历史 API 耗时不再作为模型耗时相加 | `dataset/loader/weka_trace.py`、`timing/replay_dependencies.py` |
| 起点抽样 | 固定工作负载种子 42；用 seed、trace_id、lane 的 SHA-256 派生随机源，在整棵树时间跨度的 25%–75% 抽样；不能只看 root 时间跨度 | `_seed_for_trace_lane`、`_build_timestamped_trajectory` |
| 缓存预填充 | 起点时仍活跃的各会话，包括等待子会话的父会话，重放起点前最后一个请求，max_tokens=1；按参考提前量排序，在约 10 秒的发送范围内发出 | `timing/strategies/agentic_replay.py` 的 primer 构建 |
| 缓存压力预热 | 每条通道接着回放 10 个请求，max_tokens=1，去掉轮间空闲但保留依赖；预热 grace 参数为 1,800 秒，按源码核对其计时边界，不能把它误当总预热时长上限或硬编码历史耗时 | 同文件的 `_start_accelerated_warmup` 与预热结束逻辑 |
| 正式测量交接 | 从预热推进后的各流位置继续，已发过的压力预热请求不重发；完整保留下次轮间等待，再统一扣除全局最小等待。root 已结束但子会话仍活跃的通道继续占位；耗尽的树领取下一条 | `_build_handoff_trajectories`、`_profiling_phase_t0_offset_ms` |
| 两种空闲上限 | 单棵树无在途请求时最多等待 300 秒，触发后推进它的待发时间及依赖链；全系统空闲时另有 10 秒上限。二者都不是逐请求轮间等待截断 | `ReplayIssueGate`、`_arm_root_idle_watchdog`、`_enforce_root_idle_cap` |
| 停止和收尾 | 发请求窗口为 3,600 秒，随后宽限 30 秒、取消剩余请求并最多收尾 10 秒；指标边界见 §5.1 | 参考命令、benchmark.log 与 AIPerf 结束逻辑 |
| 回放顺序与缓存身份 | 原池顺序抽样，任何树完成均从同一个抽样器取下一条；缓存隔离身份包括 benchmark、回收轮次、通道与 trace，同一棵树内父子会话共享该作用域 | `next_recycle_conversation_id`、`timing/strategies/cache_bust.py` |
| 跨流完成依赖 | 不只等待直接子会话；保留辅助／并行流之间的全部前驱，压力预热与正式测量分别正确激活和交接 | `ReplayIssueGate.activate` 与 loader 的依赖解析 |

已有研究的静态核对值为：393 条 trace、98,827 条请求；AIPerf 拆成 9,843 个 conversation，旧 TokenSim 为 3,479 个 stream。跨流依赖门控 61,136 个请求，前驱数为 1–14 的请求数依次为 51,527 / 6,771 / 1,804 / 492 / 149 / 80 / 62 / 89 / 74 / 52 / 21 / 11 / 2 / 2。实现时用相同原始数据及源码重算，这些数字只作解析回归依据，不能硬编码。

主点起点样本已在研究中逐通道对应；参考预热记录数 B200 为 211 + 1,960 = 2,171，B300 为 409 + 3,840 = 4,249。样本、会话与依赖属于结构核对；发送数、空闲跳跃次数、预热耗时和最终完成数会受模拟引擎速度影响，应比较差异及原因，不能为凑数改负载。参考树级空闲跳跃为 365 / 615 次，缺少该机制会改变负载推进速度；新增完整依赖又可能推迟请求，两项须一起验证。

AIC 没有这套 AgentX 回放行为。可选的固定参考发出时间模式仅用于诊断，正式实验仍由上述闭环产生负载。

### 2.17 超长上下文是不是调度问题

- **vLLM 没有专门的长上下文路径。** DCP=PCP=1，只有 chunked prefill + threshold 512 + interval 8 + 稀疏 kernel。一个 250K 的 prompt 会被切成若干个 ≤512 的 chunk，每 8 步放行一次，和 decode 混在同一步里。所以 **P5 必须做**。
- **但 chunk 成本随 prefix 增长。** indexer 成本 ∝ chunk × prefix/4，HCA 成本 ∝ chunk × prefix/128。isl=512、prefix=250K 仍然在表外（CSA context 表最大到 64K），decode 的 context 也在表外（bs 32/64 时表只到 16K）。所以 **P4 也必须做**，而且外推要用 DSV4 的 SOL 结构。
- **结论**：P5 决定形状，P4 决定数值，P7 决定要算多少 token，三者缺一不可。

### 2.18 AIC 做法汇总

| 问题 | AIC 是否有方案 | 可借用程度 |
|---|---|---|
| P1 拓扑与口径 | 有（单 replica 建模） | 口径可借用 |
| P2 完整模型 | 有（完整 DSV4） | 直接参考 |
| P3 CSA/HCA/SWA | 有（专用表） | 直接导入 |
| P4 表外外推 | 有（util-hold + DSV4 SOL） | 移植 SOL |
| P5 调度器 | 无（稳态模型） | 按 vLLM 源码实现 |
| P6 dtype | 有 | 直接参考 |
| P7 KV | 部分（字节公式 + 二分） | 部分借用，命中/retention 自建 |
| P8 MTP | 有稳态近似 | 借鉴进度；本轮按实际 MTP 调用和多 query 的计算／读取估算成本 |
| P9 FlashInfer | 有 FlashMLA DSV4 实测及理论成本模型 | 无同条件后端配对比值；缓存格式修正须标注近似 |
| P10 mHC | 有 | 导入 |
| P11 MoE 分布 | 有（power_law 1.01） | 参考 |
| P12 mega-MoE | GB200/GB300 表匹配 Pro；B200 同名表形状不匹配，B300 缺表 | 匹配 kernel 配置借表，初始系数 1.0；跨硬件／框架误差没有已知保证 |
| P13 缓存卸载 | 无对应 AgentX 缓存行为 | SimpleCPU 本地缓存与 Mooncake 共享存储分别自建 |
| P14 overhead | 部分（层数 × 0.8 ms、PDL 0.9） | 作初值，再标定 |
| P15 路由 | 无 | 根据逐点策略自建 router |
| P16 请求回放 | 无 | 内部实现 AIPerf 解析、预热、等待和结束语义 |

---

## 3. 关键更正

| 旧结论 | 当前结论 |
|---|---|
| DSpark K=6、AL 3.77、interval 16 | 两个主点是 vLLM MTP K=3、AL 2.49、interval 8 |
| 每条原始请求必须完成 | 全部 trace 作负载池，预热后模拟一小时，再执行等待与取消 |
| 64 GPU、V3 几何 proxy | 主点实际 8 GPU，使用完整 V4-Pro；其他曲线点逐点核对拓扑与 GPU 数 |
| 前三层 dense，专家 NVFP4 | 61 层都是 MoE，前三层 hash 路由；专家 MXFP4 w4a8，其他模块按各自 dtype |
| MTP 并入 HCA 或仅乘主干层数比例 | MTP 只有 SWA；正式成本按实际草稿模块与输出投影调用估算 |
| q=4 等同四倍独立请求 | 同一请求的 query 可能共享 KV 读取；计算与访存分别估算 |
| 每个缓存组独立分配整池预算 | 每个 worker 一个共享物理块池，各组分别管理逻辑块与窗口 |
| 日志 KV token 数就是容量输入 | 它是最大长度请求的等效值；容量由模拟器计算，日志仅核对 |
| 全部模型删掉 K/V 两份系数 | 仅按 DSV4 latent 的单份布局修正；GQA 仍保留两份 |
| trace hash 块与引擎块必须相同 | 原始 64-token hash 可聚合为 256-token 块，保留前缀链与回放作用域 |
| lane 固定到一张 GPU | 会话／subagent 使用自身 correlation ID，经 router 分配 worker |
| 只有一次缓存预热、父子会话等待 | 两步预热、全树起点、完整跨流依赖、树级与系统级空闲上限均需对齐 |
| 博客使用大缓存池，因此主点必为 Mooncake | 本轮支持两种后端；已保存的两个主点实际为 SimpleCPU |
| AIC 双后端表可直接给出 kernel 比值 | 当前无同条件配对点，不使用跨配置比值冒充后端系数 |
| mHC 表用 BF16 替代 FP32 参数 | collector 的参数为 FP32、residual 为 BF16；真实差异是融合与计时范围 |
| B200/B300 完全没有 MegaMoE 表 | B200 同名表不匹配 Pro，B300 缺表；借用匹配的 GB 表是明确近似 |
| shared expert 默认完全重叠 | 当前未融合 MegaMoE 路径串行调用；按实际融合状态与表范围计费 |
| 非单调实测点一律删除 | 保留原始数据，仅剔除已证实无效记录，其余标注并按需评估敏感性 |

---

## 4. 分阶段任务

**通用规则：**

- 每个阶段结束时停下，按 §5.3 提交报告，确认后再继续。
- 每完成一类修改，两种 GPU 各跑一次完整的一小时模拟；最终有随机因素时各用三个随机种子验证，完全确定时不重复。统一脚本提供主检查点、多并发参考曲线及缓存后端对比的配置分组。
- 结果放在 `results/agentx-dsv4-full/<work>/<batch>/<experiment-id>/<seed>/`，名称直接描述修改类别；旧结果不动。
- 按本次用户要求，每完成一个完整修改就提交代码；每次只完成具体的一部分。

**执行顺序（已确认）：** 在参考数据和完整模型配置准备好之后，按以下顺序推进：

1. **请求回放**：对齐会话拆分、并行与等待关系、请求发出时间，以及预热、一小时模拟窗口和结束规则。
2. **GPU 分配与缓存复用**：对齐请求分配、GPU 同步、缓存容量、前缀命中和 CPU 缓存卸载行为。
3. **调度行为**：对齐每步接纳的请求、token 预算、长输入分块、显存不足处理和投机解码的进度。
4. **算子耗时校准**：在执行行为正确的基础上，对齐查表、外推、投机解码成本和运行开销，最后做精度验证。

下面的任务按上述依赖排列，方案与范围均已确认。源码核对、数据导入和数值验证属于任务本身；复选框未勾选表示尚未实现。统一脚本的主检查点运行能力须在第一类修改验收前可用，完整矩阵和绘图随后补齐。

2026-09-30 本轮范围为准备工作中的「基准产物核对与对比工具」：读取两组完整一小时参考产物与全部原始 trace，完成下列基准子任务后提交并停止。参考核对不计作 §5.1 的一小时模拟；完整模型、回放与统一模拟入口尚未实现，故不宣称通过模拟器阶段验收。当前矩阵只录入已核实的两个主点，其他曲线点的实际配置仍按后续实验配置任务补齐。

### 基准与口径

目标：先把参考数字和对比脚本弄对，后续每个阶段都拿它对照。

- [x] **基准与口径任务 1** 从 `profile_export.jsonl` 重算全部指标（吞吐、TTFT、ITL、E2E、ISL/OSL 分位），与 bmk 文件对比，误差 ≤0.5%；不一致的逐项解释。
- [x] **基准与口径任务 2** 确定 cancelled / error 请求是否计入（按 aiperf 的默认口径），写进报告。
- [x] **基准与口径任务 3** 核实 server metrics 是否重复计数（引擎端口 + router 自动抓取），给出按引擎去重后的 prompt / generation 总量，与 jsonl 的 usage 对比。
- [x] **基准与口径任务 4** 从 server.log 提取时间序列（running、waiting、KV 使用率、prefix hit、MTP 接受率）和分 rank 指标，作为后续阶段的对照曲线。
- [x] **基准与口径任务 5** 解析 router.log：会话 → rank 的映射、各 rank 负载、subagent 请求用的 correlation id（是否与父会话相同）。
- [x] **基准与口径任务 6** 对比 aiperf 的 trace 抽取顺序与 TokenSim 的 lane/play 选择规则（`agentx.py:468`）。
- [x] **基准与口径任务 7** 核实 OSL 均值低于 expected（933.79 vs 1,261.54）的原因：窗口截断、取消，还是幸存者偏差。
- [x] **基准与口径任务 8** 修 `scripts/compare_agentx_results.py`：主检查点按 8 GPU 归一化，其他曲线点按配置和参考记录中的实际 GPU 数归一化；去掉 :59 的固定点位循环，按实验矩阵读取 SKU、并发、并行配置、缓存后端和参考运行身份。
- [x] **基准与口径任务 9** 生成新的 key points，放到 `results/agentx-dsv4-full/reference/`：num_gpus=8；`kv_cache_pool_tokens` 注明是日志等效值，不作为容量使用。
- [x] **基准与口径任务 10** 报告模板（§5.3）。

涉及文件：compare / summarize 脚本，`util/results.py`（255-330），新的参考提取脚本（名称待定）。
完成标准：重算指标与 bmk 相差 ≤0.5%；重复计数、router、trace 顺序、OSL 偏低四件事都有结论。
依赖决策：1、2、3、4、38。

### 完整模型与并行配置

目标：去掉 proxy，模型结构、权重显存和拓扑与参考一致。

- [x] **完整模型与并行配置任务 1** 按 HF config 写完整的 ModelSpec（`data/models/deepseek-v4-pro.yaml`），包括：
  - 逐层 compress_ratio：30 层 CSA / 31 层 HCA / MTP 层 ratio 1；
  - MoE：前 3 层 hash-MoE + 58 层常规 MoE；384 专家 / top-6 / inter 3072 / 1 个 shared expert；
  - mHC（hc_mult 4）；q_lora 1536 / o_lora 1024 / o_groups 16；
  - indexer 64×128 top-1024；SWA 128；vocab 129280；nextn 1。
- [x] **完整模型与并行配置任务 2** `TokenSim/config/model_config.py` 支持逐层结构和逐算子 dtype map（FP8 block / MXFP4 / BF16 / FP32）。
- [x] **完整模型与并行配置任务 3** 权重显存按各部分 dtype 计算（`weight_memory.py` 与 `cache_config.py`）。EP8 下每 rank 放 48 个专家，其余部分复制。计算 124.439211 GiB/rank，对日志差 −1.145% / −1.301%；不使用日志补偿系数。
- [x] **完整模型与并行配置任务 4** 删除 proxy（决策 9 已确认选 a）：
  - 删除 `data/models/deepseek-v4-proxy.yaml` 和 `data/psla/deepseek-v4-proxy.json`；
  - `scripts/run_agentx_comparison.py:73` 和 `scripts/run_agentx_sweep.sh:49` 改用完整模型；
  - 同时改掉两个脚本里的 block 64（:72 / :41）和 max_parallem_sum 2,000,000（:78 / :50）；
  - 其他引用 proxy 的文件（测试、文档、结果说明）在动手前全部列出，逐个处理；`results/agentx-empirical-extrap/` 下的旧结果不动，只在其说明里注明「proxy 模型，已弃用」。
- [x] **完整模型与并行配置任务 5** 新建 `data/clusters/8_b200/dep8.json` 和 `data/clusters/8_b300/dep8.json`：TP1×DP8×EP8，`expert_parallel_scope:"global"`，单节点 NVLink；去掉 KV 容量覆盖，容量改由 缓存容量与前缀命中 计算。
- [x] **完整模型与并行配置任务 6** 运行参数：block 256；max_num_seqs 49 / 96；max_num_batched_tokens 8192 / 16384。
- [x] **完整模型与并行配置任务 7** 两个 SKU 做完整测试（§5.1），结果标注为「管线基线」：此时算子和调度还没修正，只检查流程能否跑通、时间分解是否合理，不评价精度。

完成标准：权重显存相差 ±2%；给出算子覆盖报告（每类算子命中表 / 插值 / 外推 / analytical 的比例）；两个 SKU 都完整跑完。
2026-09-30 已完成结构／权重／cluster 准备，完整测试 269 项通过，并扫描全部 393 条 trace。随后在混合缓存阶段解决准入问题；2026-10-02 已删除 v4 proxy 配置并将旧 comparison/sweep 入口改为显式迁移提示。旧 proxy 结果目录保留为历史证据，不再被 native 入口使用。
依赖决策：7、8、9、10、14。

### 请求回放

目标：负载生成和统计窗口与 aiperf 一致。

2026-09-30 已实现 `agentx_native` 内部解析与初始 snapshot：完整池 393 条 trace、98,827 条请求、9,843 个会话、61,136 个跨流门控，前驱宽度逐项对齐；两组全部 46,122 条参考记录的原始身份／会话／turn 精确匹配，196/384 个初始 lane 及 211/409 个 primer 同样匹配。完整测试 299 项通过。当时尚未接入执行回放和一小时硬件测试；后续完成情况见下段。

后续本次已接入 Native 执行回放、压力预热及交接、两种 idle cap、引擎取消和宽限流程，完整测试 316 项通过。全部请求图在 c196/c384 的受控 transport 生命周期诊断中推进完整 3600 秒窗口，无等待死锁，预热发出 2171/4249 条；该诊断未执行硬件耗时，不是 §5.1 性能验收。随后已接入原生混合 HBM 缓存，两个 SKU 均使用完整 393-trace 池完成预热、一小时发请求及等待／取消；计数和释放核对通过，见混合缓存报告。固定参考时间诊断、正式 router 和 CPU 主点仍待实施，性能精度未验收。

- [x] **请求回放任务 1** 按 aiperf 完成预热，然后模拟 3,600 秒；到时停止发新请求，等待 30 秒，取消仍未完成的请求，再按参考执行最多 10 秒的取消收尾。预热与收尾不占这一小时的发请求窗口。
- [x] **请求回放任务 2** 直接使用 trace 里的 OSL。
- [x] **请求回放任务 3（诊断方案已确认）** 增加 `--agentx_issue_mode recorded` 固定记录时间诊断模式；两个主点均完成完整 3,600 秒窗口、grace 和取消收尾。结果保存在 `results/agentx-dsv4-full/diagnostics-recorded-issue/`，仅用于分离回放时序与引擎速度误差，不替代闭环正式结果。
- [x] **请求回放任务 4** 以全部 393 条 trace 为负载池，按 aiperf 的起点抽样和会话复用规则回放，保留依赖与等待；窗口内回放完一棵请求树后，继续从共享抽样器领取下一条。既不强制全部从第一轮开始，也不以遍历全部原始请求作为结束条件。
- [x] **请求回放任务 5** 分别报告预热与正式测量的发出、成功、错误、取消和收尾后残留请求数，以及使用过的 trace 数、唯一原始请求数和回放次数。按参考口径处理预热、宽限期和取消；未被窗口选中的原始请求不算模拟失败。
- [x] **请求回放任务 6** 将会话拆分、请求间依赖和跨请求流等待关系作为 GPU 路由的输入，先核对这些关系，再进行会话到 GPU 的映射与缓存复用验证。
- [x] **请求回放任务 7（已完成内部解析及完整管线验证）** 参考 AIPerf 的 `weka_trace.py` 和 `replay_dependencies.py`，在 TokenSim 内部实现等价解析，直接读取原始 trace：
  - 保留原始请求身份、输入输出长度、前缀 hash 序列及其作用域；解析主会话、subagent、并行请求流和独立的辅助请求，建立稳定的会话 ID 与原始请求映射。
  - 对齐会话拆分、子代理启动／等待和跨请求流的前驱关系，生成供 TokenSim 回放与 router 使用的内部请求图。
  - 对齐时间语义：使用原始时间戳与 API 耗时推导轮间等待和依赖；运行时的计算耗时由 TokenSim 计算，不能将历史 API 耗时再次叠加为模拟计算时间。
  - 只保留模拟所需的结构与长度信息，前缀复用依赖原始 hash 身份；不引入真实请求所需的合成提示文本、tokenizer 或 AIPerf 运行依赖。
  - 用源码推导的逻辑用例和已保存的参考统计核对会话划分、请求数、前驱关系及等待；参考统计用于验证，不硬编码成解析结果。集成后按已确认的预热及一小时模拟流程完整验证。
- [x] **请求回放任务 8** 按 P16 实现整棵树 25%–75% 的起点抽样，派生种子与参考一致；固定工作负载种子 42，模拟接受长度使用独立随机源。检查 root 已结束但 subagent 仍活跃、父会话等待子会话和无可用下一请求的起点。
- [x] **请求回放任务 9** 实现缓存预填充与每通道 10 个请求的缓存压力预热，max_tokens=1，按参考处理预热错误、宽限期和结束边界。压力预热保留依赖并推进回放游标；正式测量接续该游标，恢复完整等待后统一扣除最小等待，不重复发预热请求。
- [x] **请求回放任务 10** 同时实现树级 300 秒与系统级 10 秒空闲处理。树级计时从在途数降到零开始；推进待发时间时继续处理依赖链，不能给每一轮请求直接设置 300 秒等待上限。新增在途请求、树完成、阶段切换或取消时及时撤销旧计时器。
- [x] **请求回放任务 11** 改为共享顺序抽样器，任何树完成后领取下一条；保留回收轮次与缓存隔离身份，同一树内允许父子前缀共享，不同回放实例不能意外共享。正常回放的 prompt 长度与 hash 来自原始 trace，不将模拟输出追加成下一轮的新 prompt。
- [x] **请求回放任务 12** 实现明确的待发、等待依赖、排队、执行、成功、错误、取消状态。停止发新请求后完成宽限与取消流程；过期回调不能重新发请求、重复释放块或造成残留在途计数。用边界用例验证跨流依赖、空闲推进、预热交接与取消，再跑完整一小时验证。

涉及文件：`TokenSim/workload/agentx.py`、必要的内部解析模块和对应测试；现有数据集只读。
完成标准：TokenSim 独立完成解析与回放，保留每条原始请求及其身份，依赖和等待规则可追溯到参考源码。
依赖决策：32、33、40（内部解析方案已确认）。

### 缓存容量与前缀命中

目标：容量由模拟器计算；GPU HBM、外部 DRAM 和不同缓存组的分配及命中语义明确分层。

2026-09-30 GPU 部分已实现五组共享物理池、原生页布局、正预算、原子前缀恢复、retention、MTP lookahead、引用及释放优先级。完整日志定位并修复未缓存状态页错误 FIFO 驱逐长前缀的问题；随后接入 SimpleCPU 异步生命周期和 Mooncake shared-page adapter。完整窗口保留容量与历史日志的分项差异，不倒填日志容量；缓存生命周期、传输队列、取消和最终引用均已通过 native 主点与专项验证。GPU/CPU 命中率和容量误差仍是最终精度报告的一部分。

- [x] **缓存容量与前缀命中任务 1** 按本地 vLLM 源码核对 GPU KV group 的 page 大小、对齐、窗口和共享块池；按块跟踪分配、释放与前缀复用，SimpleCPU/Mooncake 分别维护外部池。
- [x] **缓存容量与前缀命中任务 2** HBM 与 CPU 容量均由模拟器按硬件、完整模型 dtype、运行时预算和实际 KV 布局计算；日志 KV 等效值与 CPU block 数只作误差对照，不覆盖模拟容量。
- [x] **缓存容量与前缀命中任务 3** 增加 DSV4 单份 latent、压缩 KV、indexer 与窗口状态的专用布局，保留 GQA 的 K/V 两份；不全局删除 ×2。复核 swap 上限，把 swap 与 offload 分开，禁止用旧的等效 token 容量覆盖自动计算结果。
- [x] **缓存容量与前缀命中任务 4** 主点引擎块按 256 token，从原始 64-token hash 聚合完整块，保留前缀链与回放作用域；其他点按配置转换。prefix 最多命中 n−1 个 token；按 retention 32768 及混合组可恢复边界截断，以当前源码核对窗口、请求末尾与分叉点的状态保留。
- [ ] **缓存容量与前缀命中任务 5（decode query + K 已独立验证；prefill/draft 全调用范围已计入并验证）** MTP lookahead：每个 decode 请求按 target query 范围及 K 个 lookahead slot 预留。见 `agentx-mtp-kv/target-budget-followup.json` 的 250→257 token 跨块例子。
- [x] **缓存容量与前缀命中任务 6** 结果 JSON 分别记录 DP worker 的 HBM、SimpleCPU DRAM 或 Mooncake shared pool 容量、命中和传输；Mooncake 使用共享 PoolKey，不复制成独立 worker 池。
- [x] **缓存容量与前缀命中任务 7** 用源码推导的分配、释放、命中和容量边界用例验证 TokenSim 缓存管理器；不将运行真实 vLLM 或下载历史镜像设为前提。
- [x] **缓存容量与前缀命中任务 8** 分离逻辑 block、物理 page 和 slot；SimpleCPU/Mooncake 异步在途需求、引用、驱逐、抢占、取消和 DMA pin 生命周期均有完整窗口及专项验证。
- [x] **缓存容量与前缀命中任务 9** 输出每 worker 的硬件预算、权重／激活／运行时等扣除项、分组布局、物理块数和日志等效值。核对 P7 的历史容量重建，但不硬编码历史分组、块数或负内存增量。

涉及文件：`cache_config.py`、cluster config、KV 管理模块。
完成标准：分组与预算来源可追溯，分配／释放守恒且不超容量；容量对日志按更新后的相对误差 ≤25% 可接受、≤10% 优先目标核对。超过时分项报告实际差异，不用日志反推覆盖值，也不隐藏当前与历史版本的差别。
依赖决策：11、12、13。

### GPU 同步与请求路由

目标：DP 同步语义和会话 → rank 的分配与参考一致。

本次已移植 router 0.1.14 的 FurcHash/MurmurHash64A、160 虚拟节点与 DP URL 环，79,367 条历史日志映射零差异；接入全局最大步长、空闲 rank dummy 和 step1/16 的完成共识。统一 native router 已用于主点与曲线，两个 SKU 的完整窗口收尾均无残留引用，测量时间守恒且会话稳定。固定实测映射诊断和历史 rank3 归因仍保留为可选诊断；同步等待与原生算子误差在最终组件分解中单列。

- [x] **GPU 同步与请求路由任务 1** DP lockstep：8 个 rank 每步同步，步长取各 rank 的最大值；空闲 rank 跑 dummy batch（对照 `core.py:2225-2240`）。
- [x] **GPU 同步与请求路由任务 2** 全局 step_counter：所有 rank 相同，wave 结束时重置，供 请求调度 的 interval 使用；当前只读源码的完成状态检查间隔为 16，旧 32 步记录另列版本差异。
- [x] **GPU 同步与请求路由任务 3（已确认方向）** 独立 consistent-hash router 已放在请求回放与 worker 调度器之间，以会话／subagent ID 稳定路由，并在结果中记录 worker、请求量、在途量与缓存命中；逐点策略由矩阵配置表达。
- [ ] **GPU 同步与请求路由任务 4（诊断方案已确认，重新审计发现 B300 498 个会话跨回放实例对应多个历史 rank，conversation-only 映射未通过）** 历史 router.log 与 AIPerf correlation 已生成固定映射；B200/B300 两个完整诊断窗口均覆盖 100% native conversation、missing_key_count=0。正式曲线仍使用 consistent-hash；证据见 [fixed router report](agentx-fixed-router-report.md)。
- [x] **GPU 同步与请求路由任务 5** 已用最终 native B200 c196 seed0 诊断并记录于 `data/reference/agentx-native-router-rank3-audit.json`：rank3 请求数、计算时间和等待均非最大，rank6 才是主要负载/等待点；历史 server-log rank3 现象属于不同运行，不能直接归因到 native 闭环。

涉及文件：`TokenSim/placement/policies.py`（:47、:91-96）、`TokenSim/llm/llm_engine.py`（`_build_worker_pool`，:619）、`TokenSim/workload/agentx.py`。
完成标准：同一会话在对应策略下稳定分配，独立 subagent 不被强行绑定父 worker；输出各 rank 负载与参考趋势对照。固定实测映射诊断检查已观测身份一致，正式闭环不要求不同完成轨迹产生完全相同的计数。
依赖决策：15、16、33。

### CPU 缓存卸载

目标：CPU 层的容量和命中与参考一致。

- [x] **CPU 缓存卸载任务 1.1：SimpleCPU 复制生命周期** 接入本地按字节限容的 CPU 池、原生混合组 suffix lookup/import、确认计算后的 eager 写回、按方向竞争的异步复制和取消释放。加载中的请求不占 RUNNING 配额，复制完成后受 max_num_seqs 限制；GPU/CPU 命中分别记录。384 项测试通过，两个主点按完整一小时流程验证；完整 CPU 调度及最终精度仍未验收，见 SimpleCPU 报告。
- [x] **CPU 缓存卸载任务 1** SimpleCPU 与 Mooncake 均有配置入口和由模拟器计算的 DRAM/shared pool；结果保留容量、块数和各 tier 命中。
- [x] **CPU 缓存卸载任务 2** 两种后端分别实现写回、命中加载、等待、异步重叠、取消释放和 PCIe/网络队列；75% 理论带宽作为显式假设保留在 manifest 和结果中。
- [x] **CPU 缓存卸载任务 2.1：SimpleCPU 加载后的恢复准入** 复制完成回到 KV-holding waiting；已有 running 先执行，token/cadence/名额/完整输入和本步 query/lookahead 分配均通过后才 RUNNING，失败时保留已加载页等待而非抢占重载。400 项测试及两组完整主点验证，B200 抢占减少约 96%，但最终性能仍未验收。
- [x] **CPU 缓存卸载任务 3** 已在 SimpleCPU/Mooncake 主点和曲线结果中分别保留 GPU/CPU 命中率；B200 的 CPU 命中显著高于 B300，趋势与参考一致，分母差异在报告中单列。
- [x] **CPU 缓存卸载任务 4** SimpleCPU 本地寻址与 Mooncake shared object/PoolKey、预算、淘汰、跨 worker 可见性、异步传输、等待、取消和链路竞争均已接入并由 native Mooncake 验收记录。

涉及文件：`TokenSim/config/kv_transfer_config.py`、现有 Mooncake 配置及新增的 SimpleCPU 对照配置；保留两种后端，统一配置选择、容量核算和统计接口。
依赖决策：31。

### 请求调度

目标：按只读的本地 vLLM 源码对齐调度行为；历史参考镜像与本地版本的已知差异单独记录。

已实现多请求 threshold 512（单请求放开）、显式 adaptive（默认 False）、DP-only interval 8、decode 存在时的 defer、capacity-bound 例外、FCFS running 插入顺序和 context 分配尾部抢占。SimpleCPU 异步生命周期、在途预留、MTP query/lookahead 预算和完整输入准入均已接入；主点与曲线运行通过引用、取消、DP 时间和预留释放校验。B200 仍处于容量边缘，抢占和吞吐误差在结果中保留，不用调度修复掩盖算子成本差异。

- [x] **请求调度任务 1** 只读核对 `/home/zhuohang/Serving/vllm-note` 的 scheduler、engine core 和 KV manager，本轮基准为 `30f5c01d4ebb`。不下载参考镜像，不修改、切换或更新此源码目录；旧草稿的行号须重新定位。
- [x] **请求调度任务 2** 实现：
  - prefill_schedule_interval（仅 DP 下节流，共享 step_counter）；
  - long_prefill_token_threshold；
  - defer（有 decode 时才推迟）；
  - capacity_bound（放行步结束时仍有 waiting，则下一轮不节流）。
- [ ] **请求调度任务 3（末步及 decode KV 已修复并验证；cached-prefill 的 padding 仍待核对）** max_num_seqs（49/96）、token 预算（8192/16384）和 MTP decode 的 1+K query/lookahead 预算均由矩阵与 scheduler 配置驱动。
- [x] **请求调度任务 4** 抢占策略对齐 vLLM：当前 `_reserve_decode_slot` 用 `queue.pop(-1)`，需要对照 vLLM 的抢占顺序。
- [x] **请求调度任务 4.1** 原生混合缓存按默认启用的完整输入检查对齐 `scheduler_reserve_full_isl`、`full_sequence_must_fit` 与 `_inflight_prefill_reserved_blocks`：接纳前检查完整输入，按 chunk 分配；异步导入只使用扣除已有 prefill 承诺及额外 decode 块后的空闲页。虚拟窗口表包含跳过/null 前缀的位置，取消、抢占和 decode 释放承诺；GPU/CPU 共用容量状态。391 项测试及两个完整主点验证通过，剩余性能问题未误标为解决。
- [x] **请求调度任务 5** 已增加 scheduler 边界、MTP query/lookahead、完整输入准入、异步加载和取消释放回归测试，直接驱动 TokenSim scheduler，不运行真实 vLLM。
  - 手工构造请求序列，逐步断言调度结果：本步放行哪些请求、每个请求调度多少 token、prefill 是否被推迟、是否抢占、抢占了谁。
  - 期望值按已记录的本地 vLLM 版本源码推导，每个用例注明对应位置。
  - 场景至少覆盖：
    - interval 8 下的节流步和放行步；
    - threshold 512 截断长 prefill；
    - 有 decode 时推迟 prefill、没有 decode 时不推迟；
    - capacity_bound（放行步后仍有 waiting，下一轮不节流）；
    - token 预算用满、max_num_seqs 用满；
    - KV 不足时的抢占顺序；
    - 完整输入准入、多个在途 prefill、异步 CPU 加载完成与取消时释放；
    - MTP 下每个 decode 请求占 1+K 个 token 预算和 K 个 lookahead slot。

涉及文件：`TokenSim/llm/llm_scheduler.py`、`TokenSim/llm/llm_engine.py`、`tests/`（新的调度测试）。
完成标准：所有用例中每一步的调度结果都与源码推导的期望一致。
依赖决策：29。

### 投机解码

目标：用 AL 参数推进，并计入 verify 和 draft 的成本。

- [x] **投机解码任务 1（方案已确认）** 每个请求的 decode step 以 51% / 49% 概率推进 2 / 3 个 token。随机状态与 trace 抽样分离；记录模拟种子，保证相同配置和种子可复现。请求结束时限制为剩余输出长度，分别记录抽样推进量和实际产出，避免把末步截断误判为接受长度偏差。
- [x] **投机解码任务 2（完整 pass/模块/head 已计齐并独立验证；缺失 SWA/hc_head 实测明确为解析估算）** 在平均接受长度推进的同时，按本地 vLLM 的实际 MTP 调用分别估算草稿模块与输出投影的成本；内部采用查表／理论模型，不生成真实 token 或执行真实张量。分别列出校验与草稿的形状、调用次数、耗时和来源，避免漏计输出投影或重复计费。多 query 校验按已确认的计算与读取分项方案处理（决策 30.1、30.4）。
- [ ] **投机解码任务 3（decode 基数已修复并验证；prefill/draft 范围已计入并验证）** KV lookahead K（与 缓存容量与前缀命中任务 5 一致）。
- [x] **投机解码任务 4** ITL 按 aiperf 口径统计。

完成标准：未截断的推进量抽样符合 51% / 49% 分布，期望为 2.49；固定种子可复现，末步不超出请求输出长度。报告抽样数及实测均值，完整运行中受末步截断、取消影响的实际平均产出单独统计；成本分解和针对边界的单元测试通过。
推进方式、草稿成本粒度和多 query 近似均已确认；实现时核对实际调用路径、计时范围和可复用数据量。

### 算子耗时

目标：每个算子的形状、dtype 和数据来源都正确，表外有可信的外推。

- [x] **算子耗时任务 1（实测优先已确认）** 导入经模型、硬件、dtype、TP/EP、后端与计时范围检查的 AIC 数据：dsv4_csa / dsv4_hca 的 context 和 generation 表，分项表（`dsv4_hca_attn`、`dsv4_paged_mqa_logits`），`mhc_module_perf`，匹配的 MoE 数据。`dsv4_csa_topk_calib_perf` 等其他框架数据只有确认目标执行路径需要时才使用。GB200 / GB300 的 DSV4 mega-MoE 借表仍按决策 26 处理。记录仓库版本、原始文件、计时单位和近似来源；保留源数据附带的许可证与归属说明。
- [ ] **算子耗时任务 2（native decode 整批已修复；prefill 长度组与 context/decode 混合批的共享投影尚待独立核对）** 逐层组合：每层按 ratio 选 CSA / HCA / SWA 表，沿用现有混合批次的正常处理流程。接入 DSV4 模块表时先建立计时范围清单，模块已包含的投影、压缩、indexer 不再重复计费；MoE 等模块外的共用工作按该步总 token 数估算一次。输入处理与输出生成按各自实际形状查询，记录无法由现有模块表反映的融合或重叠误差。
- [ ] **算子耗时任务 3（SOL 增长和多轴邻点已修复，完整主点及 holdout 已记录；部分 context 留出误差仍超界）** 先尝试同配置精确匹配和范围内插值，表外使用以实测为基准的 AIC 式理论成本比例外推。移植 `dsv4_attention_sol_ms` 前核对各组成项，缓存字节按已核实配置参数化。用留出的实测点验证（例如用 ≤32K 数据预测 64K），用分项长上下文数据检查趋势，分别报告插值、外推与跨配置近似的覆盖比例。没有可匹配的基础数据时明确记录缺口，不静默套用其他模型。
- [x] **算子耗时任务 4（清洗策略已确认）** `scripts/audit_agentx_operator_data.py` 审计 B200/B300/GB200/GB300 的 30 张 native 表：无 null/非有限/非正耗时和缺失 key；重复 key 与单轴下降全部保留并记录。证据见 `data/reference/agentx-native-operator-cleaning-audit.json`，没有按非单调或偏大耗时删除数据。
- [x] **算子耗时任务 5** 处理 B300 CSA generation 缺 bs<64 数据的问题（决策 19）。
- [x] **算子耗时任务 6（SWA-only draft 已计入三次 pass 并通过独立验证）** MTP 层按 SWA-only 处理（决策 20）。
- [x] **算子耗时任务 7（Q-block 内读取复用已完成独立验收；source-layout 和后端差异仍明示）** 区分请求数与每请求 query 数，估算 verify q=4 的 attention 成本。以 Pro 实测为基准，分别估算新增 query 的计算和按实际实现可复用 KV 的读取量；分项 q=1/q=4 数据只在计时语义可对应时辅助校验，不把整个模块盲目乘四或乘一个跨模型比值。记录读写量、计算量和缩放依据（决策 30.4）。
- [x] **算子耗时任务 8** 已核对 AIC collector、只读 vLLM 和运行日志：native 表为 FlashMLA/FP8 source layout，参考为 FlashInfer/plain FP8 + MXFP4 indexer；原始与修正布局、来源版本和不可比原因保留在 attention report 与 backend provenance，未引入无配对证据的固定系数。
- [x] **算子耗时任务 9（负载近似已确认）** 修复逐请求从零开始轮转导致负载集中到同一专家的 bug，统计先汇总整步 route 并合理分配余数。耗时查询使用 `power_law_1.01` 实测分布，不再叠加同源 straggler 惩罚；统计用直方图不得暗中改变这一成本假设。hash 路由层保留结构与路由差异，无法还原逐 token 路由时也明确记录统计近似。
  - 2026-10-01 已完成：整步合成 route、EP 交错余数和跨 step 轮转保留显式直方图，阶段缓存与零负载分母已修复。DSV4 查询改为 MXFP4/`power_law_1.01`，取消同源 straggler；413 项全仓测试及两个完整一小时主点的版本/配置、输入/输出、取消/引用、route/EP 负载和时间守恒验收通过。当前 routed kernel 仍为显式标注的临时对照，完整 MegaMoE/shared expert 成本由任务 10 和 12 继续实现。详见 [本部分报告](agentx-moe-distribution-report.md)。
- [x] **算子耗时任务 10（借表方案已确认）** 对实际采用匹配融合 kernel 的配置，分别借用 GB200/GB300 的 DSV4 mega-MoE 表估算 B200/B300，初始换算系数为 1.0；记录硬件、框架、通信与 shared expert 的计时范围差异，并评估借表敏感性。按 rank 本地 token 数查询；不均匀 rank 的组合方式按源码核对。表中已含 dispatch/combine 时不再另加该部分 all-to-all；其他 kernel 的配置按各自数据与通信路径估算。
  - 2026-10-02 已完成：导入 GB200/GB300 Pro EP4/8 MegaMoE，native estimator 使用 DP lockstep rank-local 形状、图/eager 分支、TP/SP 分片和共同最大 rank 计时；dispatch/combine 不再另计 EP all-to-all，shared FP8 fused 增量显式标注 D 级。B200/B300 完整主点吞吐为 1,733.192／3,440.686 token/s，证据见 [原生 MegaMoE 报告](agentx-native-megamoe-report.md)。
- [x] **算子耗时任务 11（原生 pre/post 的重复残差与 SP layer 行数已修正并独立验收）** 导入匹配的 mHC 实测表，核对 FP32 参数、BF16 激活、num_sites 及 pre/post 计时范围，标注与 serving 融合路径的差异（P10、决策 27）。unfused collector 不含输出 RMSNorm，其估算仍单独保留；不套用其他硬件的融合时延比。
  - 2026-10-08：移除 native 分支两处 generic residual_add，保留 collector 未包含的 RMSNorm；PP1/EP/TP>1 的 layer mHC/norm 用 ceil(T/TP)。509 项测试、B200/B300 完整主点及无重复残差/时间守恒验收通过；主干 hc_head/final norm 与 mHC summary 的独立缺口转入任务 12，见 [本部分报告](agentx-mhc-scope-report.md)。
- [x] **算子耗时任务 12.1：原生低秩投影与 GEMM dtype** 按 DSV4 实际 replicated/column/grouped/row-parallel 几何替代稠密 QKV/O；router 与 lm_head 使用 BF16，grouped FP8 einsum 输出 BF16，缺失匹配实测时标为单 launch 解析估算。采集 manifest 与运行时形状一致；405 项全仓测试及两个完整一小时主点的请求、缓存、取消、DP 时间、源码和配置身份核对通过。shared expert、gate 融合附加工作及 MTP query 计时范围随后单独完成，任务 12 整体保持未勾选。
- [ ] **算子耗时任务 12（target logits 与 deferred draft head 已验证；主干 hc_head/final norm 全 forward 行和 mHC 汇总仍待补齐）** 逐项核对 router、lm_head、shared expert 的 dtype、调用次数和 query 形状；shared expert 按实际路径融合或独立计时，不预设其与 routed MoE 完全重叠，也不重复计算已计入融合表的部分。
- [x] **算子耗时任务 13（校准范围已确认）** 已完成 B200 校准审计并冻结 `step_overhead_us=0`，直接用于 B300；两 SKU 组件时间均记录 overhead=0。正 overhead 无法解释当前“模拟器更慢”的误差，证据见 [overhead 校准报告](agentx-overhead-calibration-report.md)，未追加 SKU 修正。
- [x] **算子耗时任务 14.1：测量阶段分项守恒** 预热结束后重置 operator 查询、step 与 component 计数，保留查表缓存和全程 missing-shape 清单；两个完整主点的 profiling 分项和均与 DP local compute seconds（含 dummy）一致。旧结果含预热的分项不作直接百分比对照，完整模块分解和最终比较继续由任务 14 完成。
- [x] **算子耗时任务 14** 输出 native result 的 attention / MoE / GEMM / elementwise / comm / overhead 分解，并与旧 proxy 分解分开报告；FlashMLA/FlashInfer 和跨硬件借表差异不折算成伪精度。
- [ ] **算子耗时任务 15（原对照的量化/mixed batch 不一致；同 FP8/MXFP4 调用发现 AIC builder 不支持 vLLM MegaMoE，匹配对照继续补齐）** 已用独立入口 `scripts/run_agentx_aic_diagnostic.py` 构建 AIC 0.12.0 Rust extension 并运行 B200 同形状估算；B300 明确记录为缺少匹配 NVFP4 MegaMoE 行，没有跨 SKU 猜测。AIC source commit、per-op source 和运行时身份保存在 `data/reference/agentx-aic-diagnostic.json`。

涉及文件：`operator_table.py`、`lookup.py`、`schema.py`、`TokenSim/operator_data/importers/aiconfigurator.py`（375-435）、`TokenSim/moe/routing.py`、`data/devices/*.yaml`。
完成标准：没有算子落进「未知 / 默认」分支；holdout 误差有记录；与 AIC 单步对照的偏差有解释。
依赖决策：17–28、30.4、34、39。

### 统一实验脚本与配置矩阵

目标：一份实验脚本完成全部参考点与后端对比的配置检查、批量模拟、汇总、绘图及逐点误差比较。拓扑和后端差异放在实验配置中，脚本不为每个并发点复制一套执行逻辑。以下为已确认交付的实现计划，脚本尚未编写。

- [x] **实验配置任务 1** 建立版本化矩阵文件（`data/experiments/agentx.yaml`）。每个可执行实验点具有唯一 ID、分组、硬件与实际 GPU 数、完整模型、cluster、TP/DP/EP、并发、缓存后端、router、调度预算、量化、投机解码参数、HBM/DRAM 预算和工作负载种子；模拟 seed 由统一入口单独记录，展开配置保存在每个 run manifest。
- [x] **实验配置任务 2** 逐点关联已核实主检查点的参考运行 ID、配置出处和参考指标文件；未独立核实的扩展曲线保留 reference-index-only 元数据，不冒充可比较结果。

  | GPU | 参考点与并行配置（待逐点核对其余参数） |
  |---|---|
  | B200 | 并发 1/4/8/14：TP8；32/64/96/128/160/196：DEP8 |
  | B300 | 并发 1/4：TP8；8/16：TP4；48/64：TP4 + EP4 + attention DP；128/256/384/512/576：DEP8 |

  完整运行 ID 索引见 §1.10，具体参数以对应运行配置为准；不同日期、不同 GPU 数的运行不能混作同一配置的重复实验。

- [x] **实验配置任务 3** 配置主检查点、DEP8/TP4/EP4/TP8 曲线、SimpleCPU/Mooncake/HBM 后端和 native operator scope；未核实参考配置的曲线点明确标为扩展实验，不计算伪造参考误差。消融与带宽敏感性仍属于最终诊断，不与正式曲线混合。
- [x] **实验脚本任务 1** 统一入口 `scripts/run_agentx_experiments.py` 支持列出、检查、展开、按硬件／实验 ID／分组筛选、限制并行度、断点续跑、三 seed 汇总和六面板绘图；结果按 seed 和 point 隔离。
- [x] **实验脚本任务 2** 启动前检查完整模型、硬件、并行分组、实际 GPU 数、后端能力、operator-data 包、参考文件和完整窗口参数；native runner 拒绝 proxy 配置并固定 393 条 trace、预热、3,600 秒及等待／取消流程。
- 2026-10-02 已实现统一入口的 `list/check/expand/run/summary/plot`、成功结果断点跳过、seed1/seed2 批量运行，以及 operator package、cluster worker 数、模型/参考文件和完整窗口参数的启动检查。参考曲线的精确逐点 cluster/offload 参数仍保持 metadata-only，未把未核实配置伪装成可执行实验。
- [x] **实验脚本任务 3** 每个 point/seed 通过独立 native runner 进程初始化硬件、队列、router、缓存和随机状态；只读 trace 可复用。
- [x] **实验脚本任务 4** 失败、成功和到期取消由 `run_status.json` 与 AgentX phase counters 区分；批次遇到非零退出码返回失败，断点续跑只跳过完整成功状态。
- [x] **实验脚本任务 5** run manifest 保存解析点、defaults、backend、seed、dataset/source SHA-256、代码提交和完整命令；目录按批次／seed／point 隔离。
- [x] **实验脚本任务 6（日志粒度已确认）** native runner 保存逐请求、窗口和异常日志；详细 replay/cache 事件由 benchmark 配置控制，日志开关不改变模拟输入和窗口。
- [x] **结果汇总任务 1.1：原生首次准入来源** GPU/CPU/计算输入保留首次准入快照，重新加载、重算及最后一次命中独立记录；未观察字段不补零。396 项测试及 B200/B300 两个完整窗口验证，21,395 条终态记录的执行字段与基线完全相同；详见首次来源报告。
- [x] **结果汇总任务 1** 统一入口按 point manifest 读取 topology/backend，保留 GPU/CPU hit、传输和缺失字段语义；旧 compare 工具继续用于已核实主点的独立审计。
- [x] **结果汇总任务 2** `summary.csv` 保留 seed 与 aggregate 行，包含总量和每 GPU 吞吐、TTFT/ITL/E2E、请求数、GPU/CPU hit、拓扑、后端和参考误差；native result JSON 继续提供显存、传输、算子匹配和时间分解。
- [x] **曲线任务 1** 统一绘图输出吞吐、TTFT、ITL、E2E 和 GPU/CPU 命中率面板；吞吐按 point 的实际 GPU 数归一化，TP/DP/EP 与 backend 保留在 CSV。
- [x] **曲线任务 2** 有 key-point reference 的点输出逐点吞吐误差；reference-index-only 扩展点明确保留空误差，不制造比较值。
- [x] **脚本迁移任务** 旧 comparison/sweep 入口已改为显式退役提示，禁止继续产出 proxy、64 GPU、短窗口或缩小 trace 的结果；native 工作流统一指向矩阵入口。
- [x] **脚本验证任务** 已覆盖矩阵展开、逐点参数、实际 GPU 数归一化、失败状态、断点续跑身份、seed aggregate 和后端统计映射；完整一小时曲线和主点结果作为验收证据。

完成标准：一次命令可完成所选配置组的模拟、汇总、绘图与参考比较；所有结果可追溯到唯一配置，失败和缺失参考不会被隐藏。已确认同时支持两种缓存后端和全部参考并行配置；算子近似按 §6 的决定标注来源与误差。

### 最终验证

- [x] **最终验证任务 1** 两个 SKU 按 §5.1 完成预热及一小时模拟；主点和已配置曲线均使用三个修复后的模拟 seed，报告明确标注固定一小时窗口且不要求全部原始请求完成。
- [x] **最终验证任务 2（范围已确认）** B200 c196/B300 c384 已完成 HBM-only 和 MTP-disabled 两类完整三 seed 消融；完整模型和一小时协议保持不变，结果见 [最终消融报告](agentx-final-ablation-report.md)。
- [x] **最终验证任务 3（已确认）** 三 seed 主点和消融均使用同一组额外开销输入；B200/B300 结果直接比较，未按 SKU 重新拟合。当前 step overhead 明确为设备数据中的 0，绝对误差来自 attention/operator provenance，而非新增 SKU 修正。
- [ ] **最终验证任务 4（已确认，尚未证实相同模型、形状与算子范围）** 已汇总 AIC B200 同形状离线估算与 TokenSim native attention 查询；B300 缺失行作为明确数据缺口保留。分项对照见 `data/reference/agentx-aic-shape-comparison.json`，不将差异折算为校准系数。
- [x] **最终验证任务 5** 更新 `docs/agentx.md`，迁移到 native matrix workflow 并明确旧 proxy 结果的历史性质。
- [x] **最终验证任务 6（已确认范围）** 统一实验脚本已完成 TP8、TP4、TP4/EP4、DEP8 曲线和 SimpleCPU/HBM 配置选择；逐点输出三 seed 汇总、绘图、生命周期校验和可比较参考误差。

- [ ] **最终精度验收** 用当前实现重新完成全部必需验证，满足 §5.2 的 25% 门槛。历史三 seed 和曲线仅证明对应代码版本的流程，不能替代修复后的精度验证。

完成标准：满足 §5.2 的阈值。

---

## 5. 验证方案

### 5.1 完整测试的定义：与参考一致的一小时模拟

**已确认：完整测试指完整执行参考的一小时测量流程，不要求遍历全部原始请求。** 后续文档与结果报告都必须注明这一点。此前的全请求完成验收已被此决定替代。

- **数据与并发**：全部 393 条 trace、98,827 条原始请求作为可选负载池。主检查点为 B200 并发 196、B300 并发 384；曲线各点使用其明确配置的并发及拓扑。每个点都使用完整负载池和一小时窗口，不擅自降低该点配置的并发。
- **预热**：按参考的抽样起点与预热流程执行，预热耗时不计入随后的一小时。正式测量接续预热后的回放位置。
- **测量窗口**：从预热结束后的测量起点，推进 3,600 秒模拟时间。这里的一小时指被模拟系统的时间，不是运行模拟程序的实际耗时。
- **结束规则**：3,600 秒后停止发新请求；等待 30 秒宽限期；取消剩余在途请求；再按参考执行最多 10 秒取消收尾。不能在 3,600 秒处直接丢掉全部在途请求，也不无限等待它们完成。
- **统计口径**：宽限期内完成的请求按 aiperf 口径纳入；预热、错误与取消分别处理和报告。指标的有效计时边界按 aiperf 对齐，不能仅因发请求窗口为 3,600 秒，就把所有吞吐分母硬编码成 3,600 或 3,640 秒。
- **完成条件**：完成上述测量流程，并核对请求状态与结束边界；实际覆盖的 trace 数和唯一原始请求数需要报告，不要求等于整个数据集。正常到期取消须按参考记录，不据此判为测试失败。
- **报告标注**：明确写出「全部 393 条 trace 为负载池；预热后模拟一小时；不保证全部原始请求被执行」，并分别记录测量窗口、宽限期、实际运行耗时和发出／成功／错误／取消请求数。

**参考证据：**

- 两组 `benchmark_command.txt` 均包含 `--benchmark-duration 3600` 和 `--num-dataset-entries 393`；B200 日志记录发出 14,124、完成 13,991、取消 133；B300 发出 25,923、完成 25,711（含 15 个错误）、取消 212。
- [博客的预热与计时说明](https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat)明确描述预热后测量一小时，并指出不同配置完成的请求数会不同。对应本地证据为 `/tmp/infx-art/{b200,b300}/agentic/benchmark_command.txt`、B200 `benchmark.log:1633-1638`、B300 `benchmark.log:2196-2213`。

**验证要求（已确认）：** 不使用 smoke 测试、不缩短一小时模拟窗口。单元测试只验证逻辑。每完成一类修改，两种 GPU 各跑一次主检查点；最终按统一配置矩阵完成多并发曲线及后端对比，有随机因素时各用三个随机种子，否则不重复相同确定性运行。

本轮已确认随机接受长度，因此最终启用该方案的每个实验点均使用三个模拟随机种子，保存逐次结果与汇总统计。参考工作负载的抽样种子和模拟接受长度种子分别配置，不能仅为增加重复次数而同时改变参考负载条件。

### 5.2 指标与误差标准（2026-09-30 用户更新）

用户确认：模拟结果相对参考的误差在 **25% 以内可接受，10% 以内为优先目标**。此标准取代此前针对容量、吞吐和延迟的各项模型精度门槛；GPU/CPU 命中率同时报告相对误差和绝对百分点差，保持各自的输入 token 分母，不合并为一项。

| 比较 | 可接受 | 优先目标 |
|---|---|---|
| 输出／输入吞吐、TTFT、ITL、E2E、权重及 KV 容量 | 绝对相对误差 ≤25% | ≤10% |
| GPU／CPU 命中率 | 分母一致时，相对误差 ≤25% | ≤10%，另报百分点差 |
| 参考值为零 | 模拟也为零才能通过；非零时相对误差未定义 | 同左 |
| 缺失指标或配置不兼容 | 未评估，不能判通过 | 同左 |

此调整不放宽数据完整性检查：jsonl 与 bmk 的重算核对仍为 ≤0.5%，模型尺寸、解析身份、计数和引用守恒仍须严格验证。继续使用完整 393-trace 负载池、预热及一小时流程，不能通过舍入误差或改填容量达到阈值。现有 B200 容量误差 −25.449% 仍略超可接受范围；B300 −3.321% 已达优先目标。

### 5.3 每个阶段的报告模板

1. 改动：文件和行号；
2. 单元测试 / 差分测试结果；
3. 完整测试结果，与上一个阶段和参考对比；
4. 时间分解；
5. 新发现的问题和新的决策项。

### 5.4 消融

已确认在 B200 c196、B300 c384 两个主检查点做功能消融；每次只改变一项，保持完整模型、全部 trace 负载池、一小时窗口和对应种子要求。完整实现运行全曲线，消融不扩展到全曲线。额外的带宽三档敏感性实验单独分组，标明 75% 等取值是缺少实测时的假设。

### 5.5 泛化

已确认：额外运行开销只用 B200 标定；冻结参数后，使用 B300 的硬件规格与运行配置进行验证，不对 B300 单独拟合。保存校准依据、参数和验证误差，避免把硬件差异误写成按 SKU 调参。

---

## 6. 已确认决策与实施约束

以下决定均已确认，或属于已授权范围内的事实核对与工程适配。当前没有待用户答复项；技术缺口由 §4 的实现任务继续核实。编号只用于本文交叉引用，不作为阶段代号。

### A. 基准与口径

1. 【已确认】B200 c196 和 B300 c384 均作为主检查点；本轮扩展到多并发参考曲线，包含实际 TP8、TP4、DEP8 等配置变化。
2. 【已确认】主检查点按 8 GPU 归一化；扩展曲线按逐点实际 GPU 数归一化。新参考数据放到 `results/agentx-dsv4-full/reference/`，旧文件保留。
3. 【已确认，沿用此前答复】统计口径：
   - 3.1 ITL 按 aiperf 口径；
   - 3.2 命中率以 bmk 为主、API 为辅；
   - 3.3 输入吞吐包含命中 token；
   - 3.4 时长按 aiperf；
   - 3.5 server metrics 按引擎去重后求和；
   - 3.6 以 jsonl 为主基准（与 bmk 相差 ≤0.5%）
4. 【已确认】GPU、CPU 命中率分别报告，包含相对误差和百分点差；总命中率同时报告。按 2026-09-30 用户更新，相对误差 ≤25% 可接受、≤10% 为优先目标，取代旧门槛。
5. 【已确认】采用参考的预热、3,600 秒模拟窗口及等待／取消规则。每完成一类修改，两种 GPU 各完整跑一次；最终有随机因素时各用三个随机种子，否则不重复相同确定性运行。
6. 【已确认】本轮复现多并发参考曲线，包含低并发 TP8/TP4 与高并发 DEP8 等配置变化；差异全部由实验配置表达，编写统一的全面实验脚本。逐点参数按参考记录核实，脚本任务见 §4。

### B. 模型

7. 【已确认，沿用此前答复】使用 AIC 仓库里的 HF config 副本。
8. 【已确认，沿用此前答复】专家按 MXFP4 w4a8 处理。
9. 【已确认，沿用此前答复】删除 proxy 模型文件，实验使用完整模型。
10. 【完整模型要求的实现范围】逐层 CSA/HCA/SWA、hash-MoE、mHC、LoRA、indexer、MTP 均须有结构与成本去向，不再询问是否省略已要求的完整模型组成。允许的成本近似单独说明，不以省略模块代替完整模型。

### C. KV

11. 【已确认容量来源及后端范围】容量完全由模拟器计算，日志只作核对；同时支持 SimpleCPU 与 Mooncake。GPU 本地 HBM、SimpleCPU 本地 CPU 缓存及 Mooncake 共享 DRAM 分别建模，通过配置选择；旧版「逐组独立建池」不再作为推荐方案。
12. 【源码核对与工程实现】核对最大可复用前缀、缓存块边界及 retention 规则，按参考实现；这不是由用户选择的缓存语义。
13. 【逐点参考配置】主参考点 block 256；曲线各点按真实配置核实和输入，不再独立询问是否采用参考参数。

### D. 拓扑

14. 【已确认复现范围内的配置对齐】主检查点复现跨 8 个 DP rank 的 EP8；其他曲线点按各自 TP/DP/EP 分组配置，不把主点的分组推广到所有点。
15. 【源码核对与工程实现】复现对应配置的 GPU 同步和空闲 rank 行为；核对同步步时长、dummy batch 和通信成本，避免重复计费，无需用户选择是否遵守参考同步语义。
16. 【已确认】在 TokenSim 增加简单 router 模拟，逻辑参考博客和对应运行配置；会话／subagent ID 决定分配，worker 再执行本地调度。同时增加固定实测 worker 分配的可选诊断模式，只用能够核对的会话映射，报告覆盖率；正式实验仍由 router 模拟分配。

### E. 算子

17. 【已确认处理原则】实测数据优先，缺失形状允许插值或外推。实现采用以邻近实测耗时为基准的 AIC 式理论成本比例方法，并用留出的实测点验证；具体公式、适用范围和误差必须记录，不宣称未覆盖区域是实测。
18. 【已确认清洗策略】保留原始数据，只排除可确认损坏或无效的记录；原因不明的异常耗时先标记、保留，必要时对比其影响。不仅因非单调或不接近参考结果就删点。
19. 【已确认，沿用此前答复】B300 CSA generation 缺 bs<64 时，用 B200 数据乘以 HCA 表的 B300/B200 比值；作为有来源的近似单独标注，并验证误差。
20. 【源码核对与工程实现】MTP 的结构按本地 vLLM 核对，不把成本近似当作结构事实。旧表述「AIC 将 MTP 并入 HCA」需更正：AIC 用 generation 成本缩放近似 draft。具体成本近似并入决策 30.1。
21. 【计时范围对齐】实测模块已包含的投影、压缩、indexer 等不得额外重复计费；按 collector 的实际计时边界适配，无需单独确认。
22. 【工程适配，无需单独决策】混合批次沿用 TokenSim 已有的正常流程；此前把它作为新增选择不准确。实现时核对 DSV4 模块实测表的计时范围和混合形状，避免已包含的算子被重复叠加，记录融合或重叠近似；详见 P3 与算子耗时任务 2。
23. FlashInfer 与 AIC 表（FlashMLA）的差异（见 P9）：
    - 23.1 【已确认】以 DSV4 实测表为基础，配置不匹配单独标注，不另设缺乏同条件实测支持的后端系数。旧候选「用 DSA 双后端表直接算比值」已因没有配对点撤回；具体实测覆盖见 P9。
    - 23.2 【已确认核对与近似处理原则】先核实采集与目标的 KV 布局、indexer dtype 和实际成本项，再用参数化计算／访存成本估算差异。旧稿 584 → 512 B、132 → 68 B 属于待核实的配置假设，不能因方案获准就当作已验证事实；必须保留原始实测值，另列修正值、公式和误差来源。
24. 【源码核对与计时范围适配】router、lm_head、shared expert 的 dtype、形状与调用次数按目标路径核对，shared/routed 的融合或独立执行按源码和配置处理，不再作为用户选择的事实。
25. 【已确认专家负载近似】使用 AIC 的 `power_law_1.01` 非均匀负载实测表，不再叠加一次同源的不均衡惩罚；这是固定统计近似，不保证逐请求专家分配一致。逐请求轮转 bug 属于必要修复；统计先汇总整步 route 并合理分配余数，直方图只进统计，不改变已确认的时延近似。
26. 【已确认跨硬件借表】对已核实使用对应融合 kernel 的配置，使用 GB200 / GB300 的 DSV4 mega-MoE 实测表，已包含的 dispatch/combine 不再单独加 all-to-all。这是模型形状匹配、硬件与框架仍有差异的近似；不能用于参考实际采用另一 kernel 的曲线点。B200 同名表的 hidden=3584、896 专家、top-16 不匹配 Pro，B300 缺少该表。
    - 26.1 【已确认】换算系数暂取 1.0，报告跨硬件、框架和通信差异，并评估敏感性。P12 中从其他 MoE kernel 推出的约 ±10% 不是目标融合 kernel 的误差保证。
    - 26.2 【源码核对与计时范围适配】shared expert 按实际融合状态和源码路径处理。本地未融合 MegaMoE 路径的 shared expert 在 routed experts 调用后执行，不能默认按完全重叠估算；表内 num_fused_shared_experts=0 时不能误认为已计入 shared expert。历史运行无法确认的融合状态单列差异，不重新要求下载已拒绝的历史镜像。
27. 【实测优先方案下的正常导入】使用匹配的 mHC 模块表；collector 本身具有 FP32 参数和 BF16 residual，旧稿声称的精度替代问题不成立。核对 num_sites 与融合计时范围，详见 P10。
28. 【已确认校准原则】用少量参数估算调度、框架等额外开销，只用 B200 校准，再用同一组参数预测 B300。拟区分 graph / eager；参数形式和拟合依据须写入实现及结果报告。

### F. 调度

29. 【源码基准已确认】只读参考 `/home/zhuohang/Serving/vllm-note`（本轮 HEAD `30f5c01d4ebb`），不下载历史镜像、不修改该目录。使用 TokenSim 自己的调度器测试，源码推导期望行为，覆盖用例见 §4。

### G. 投机解码

30. 投机解码的具体做法：
    - 30.1 【已确认具体粒度】保留平均接受长度推进，按本地 vLLM 的 MTP 调用分别估算草稿模块与输出投影的成本，内部用查表／理论模型，不执行真实模型或生成真实 token。不采用仅按主干层数比例增加 draft 成本的正式方案；不改变已确认的 2/3 token 随机推进。
    - 30.2 【已确认】每步随机产出 2 或 3 个 token，概率分别为 51% 和 49%，期望 2.49；对应 synthetic 接受率 [1.0, 0.49, 0.0]。最终每个启用此方案的实验点使用三个模拟种子；请求末步按剩余输出长度截断。
    - 30.3 【已确认，同统计口径】ITL 按 aiperf 口径。
    - 30.4 【已确认多 query 成本近似】区分「同一请求校验多个 token」与「多个独立请求」：以匹配的 Pro 实测为基准，分别估算 query 计算增长和按实现可复用的 KV 读取。现有分项表有 q=1/q=4 配对，可检查趋势，但使用 Flash 模型；attention 的 q=4 行走 context 元数据，indexer collector 把 batch×query 的所有行组织在共享 KV 上，均不能直接视为目标 Pro 多请求 MTP 的配对实测。正式方案不把整个模块直接乘四或乘分项表比值。

### H. Offload

31. 【已确认缺失带宽近似】写回、加载及请求等待行为按各后端实际实现核对。有效传输带宽优先用匹配实测；缺少可信实测时，将利用率显式配置，以理论上限的 75% 为基线，并用 50% / 75% / 100% 做敏感性实验。75% 是暂定假设，不是硬件实测；三个取值仅作用于待估计链路，不替代总线竞争、排队和异步重叠建模。

### I. 回放

32. 回放语义：
    - 32.1 【已确认】预热、一小时模拟窗口及等待／取消规则按 aiperf；
    - 32.2 【原始负载的正常处理】使用 trace 定义的输出长度，不用手调长度抵消模拟耗时误差；预热的 max_tokens、到期取消等按已确认的回放规则处理。
    - 32.3 【额外诊断已确认】增加固定发出时间的可选模式，按参考 `credit_issued_ns` 发请求，用于分离回放时序与引擎速度误差。它不替代正式闭环回放，也不将诊断吞吐作为正式复现结果；按需在两个主检查点采用完整的一小时流程。
33. 【已有核实结论，非新增选择】subagent 按自己的 correlation id 路由；实现时按对应运行配置核对，不再要求用户重复确认。

### J. 工程

34. 【数据导入的常规实现】保留所用源数据附带的许可证、归属说明和适用的 NOTICE，记录来源；不作为需用户再次确认的实验选项。
35. 【已确认】执行顺序：请求回放 → GPU 分配与缓存复用 → 调度行为 → 算子耗时校准。会话拆分先于使用该拆分的路由诊断；执行行为对齐后再标定耗时。讨论时直接使用工作名称；按本次用户要求，每部分完成后测试、提交并汇报。
36. 【正常交付】最终验证后更新 `docs/agentx.md`，说明完整模型、配置矩阵、一小时窗口、运行命令和已知近似。
37. 【已确认】最终验证做「B200 标定 → B300 验证」的泛化比较，同决策 28。
38. 【统一脚本的正常实现】结果目录按工作类别／批次／实验 ID／种子区分，保留旧结果，具体布局与统一实验脚本一致，不再单独询问路径命名。
39. 【已确认额外诊断范围】在隔离环境离线调用 AIC，对相同模型／batch／长度／并行形状比较一次前向估算及分项，用于定位数据导入和算子组合差异。AIC 只是诊断依赖，不成为 TokenSim 正式模拟的运行依赖，此对照不替代完整一小时验收。
40. 【已确认】参考 AIPerf 源码，在 TokenSim 内部实现相同的会话拆分、请求依赖和等待解析逻辑；直接读取原始 trace，不安装或运行 AIPerf 加载器，不依赖其离线导出文件。此项为方案确认，实现任务见「请求回放任务 7」。
41. 【已确认日志粒度】默认保存逐请求结果、按时间窗口的统计和异常现场；逐步调度／缓存事件日志通过配置按需开启。日志开关不改变负载池、模拟时长或请求执行。
42. 【已确认消融实验范围】完整实现按已确认矩阵跑全部参考曲线；逐项关闭缓存卸载、投机解码等功能的消融仅在 B200 c196、B300 c384 两个主检查点执行，其余配置固定，每次仍完整模拟一小时，含随机因素时三个种子。

### 决策状态

本轮最后确认的四项已纳入：实际 MTP 调用成本、多 query 计算与读取分项估算、固定参考发出时间诊断、固定实测 worker 诊断。此前确认的近似、两种缓存后端、完整曲线、消融、带宽敏感性和日志要求继续有效。新增诊断入口按需使用，正式结果仍由内部回放和 router 生成。

---

## 7. 参考资料

- **AIPerf 解析参考**：`/tmp/aiperf-src/49634d1c/src/aiperf/dataset/loader/weka_trace.py`、`/tmp/aiperf-src/49634d1c/src/aiperf/timing/replay_dependencies.py`；只作源码参考，逻辑在 TokenSim 内部实现。
- **启动脚本**：`/tmp/infx-hist/b200_vllm_mtp_843c6254.sh`、`/tmp/infx-hist/b300_vllm_mtp_c62f6e33.sh`（InferenceX `benchmarks/single_node/agentic/` 的历史版本）
- **配置矩阵**：`/tmp/infx-hist/nvidia-master_{843c6254,c62f6e33}.yaml`
- **运行产物**：`/tmp/infx-art/{b200,b300}/{agentic,bmk,server_logs}`（server.log、router.log、aiperf_artifacts/profile_export.jsonl、server_metrics_export.json）
- **InferenceX metrics 汇总**：`/tmp/infx-hist/server_metrics_843c6254.py`（compute_server_metrics → adapter.populate → gpu_kv_capacity_tokens → apply_profile_totals）
- **vLLM**：`/home/zhuohang/Serving/vllm-note`（本轮只读 HEAD `30f5c01d4ebb`；历史参考镜像为 426e59f；不下载或修改源码目录）
  - `vllm/models/deepseek_v4/`：attention.py、nvidia/model.py、nvidia/flashinfer_sparse.py、nvidia/mtp.py
  - `vllm/v1/engine/core.py`、`vllm/v1/core/sched/scheduler.py`、`vllm/config/{scheduler,speculative}.py`
  - `vllm/v1/core/{kv_cache_utils,kv_cache_coordinator,single_type_kv_cache_manager}.py`
- **AIC**：
  - 数据仓库：`/tmp/aiconfigurator`（@f254959）
  - Python SDK：`/tmp/aic-full/aic-core/src/aiconfigurator_core/sdk/`（models/deepseek_v4.py、models/helpers.py、backends/vllm_backend.py、backends/base_backend.py、rust_engine_step.py）
  - 预测入口：`/tmp/aic-full/src/aiconfigurator/sdk/{predict.py,speculative.py}`
  - Rust：`/tmp/aic-full/aic-core/rust/aiconfigurator-core/src/`（perf_database/perf_interp.rs、perf_database/dsv4.rs、engine/runtime.rs、session.rs）
  - 表：`aic-core/src/aiconfigurator_core/systems/data/{b200_sxm,b300_sxm}/sparse_attention/vllm/0.24.0/`
- **旧结果**：`results/agentx-empirical-extrap/`（保留不动）
- **博客**：[AgentX / InferenceX v3](https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat)

`/tmp` 路径是本轮研究缓存，不是交付依赖。实施时把使用的版本、原文件路径／公开出处、内容校验值及许可证信息保存到正式参考与算子数据 manifest；参考数字可追溯，不能依赖临时草稿永久存在。AIPerf 回放还需参考 `timing/trajectory_source.py`、`timing/strategies/agentic_replay.py` 和 `cache_bust.py`。旧草稿中与本文冲突的候选方案不再有效。
