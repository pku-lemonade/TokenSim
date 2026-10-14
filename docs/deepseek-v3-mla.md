# DeepSeek-V3 MLA 理论模拟

The DeepSeek V3 runner decomposes the model into operators and schedules their service estimates on AF resources.
It simulates the complete model without loading checkpoints or executing GPU inference.
Use `scripts/run_deepseek_mla.py` with `--operator-fallback analytical_only` for analytical estimates, or provide compatible operator packages for table lookups.

## 结构与边界

结构来自 `data/models/deepseek-v3-hf-config.json`，来源为 [DeepSeek-V3 官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)。模拟配置为 `data/models/deepseek-v3-671b-mla.json`。

| 参数 | 数值 |
|---|---:|
| hidden size / 层数 | 7168 / 61 |
| query heads | 128 |
| Q / KV latent rank | 1536 / 512 |
| 每头 noPE / RoPE / V 维度 | 128 / 64 / 128 |
| Dense / MoE 层 | 前 3 / 后 58 |
| routed experts / top-k / shared experts | 256 / 8 / 1 |

Q/K head dimension 为 **192**，不能用 `7168/128=56` 代替。压缩 cache 宽度为 **576**，不乘 query head 数，也不乘普通 Attention 的 K/V 双份系数。BF16 每层每 token 为 **1152 字节**。A TP 时每个 rank 复制完整 latent cache，query heads 和对应 up/output 权重分片，down 投影复制。

只增加 MLA Attention 的特殊处理，Dense、Router、Expert、Combine 和传输沿用既有实现。路由是可复现的 synthetic uniform top-8，**不复现 V3 noaux_tc 分组选择**。模拟使用 BF16，区别于官方 FP8 checkpoint；不包含 MTP。显存账本包含 embedding 和未绑定的 lm_head 权重，但耗时范围为 transformer 层，不包含 embedding、lm_head、采样和 CPU 服务开销。

## 单层拆分与公式

设 T 为本批 token 数，H 为本卡 query heads，D=7168，rq=1536，r=512，dn=128，dr=64，dv=128。

1. 输入 RMSNorm。
2. Q down `[T,D]×[D,rq]` → latent RMSNorm → Q up `[T,rq]×[rq,H*(dn+dr)]`。
3. KV down `[T,D]×[D,r+dr]` → latent KV RMSNorm → Q 和共享 K 的 RoPE。
4. decode：H 组 Q absorption `[T,dn]×[dn,r]` → compressed Attention core → H 组 value expansion `[T,r]×[r,dv]`。
5. prefill：KV expansion `[T,r]×[r,H*(dn+dv)]` → expanded causal Attention；持久 cache 仍写 latent+RoPE。
6. 输出投影 `[T,H*dv]×[H*dv,D]`。Residual 等外围操作沿用 AF 既有边界。

逐头矩阵乘用 `batched_gemm`：工作量按 H 组累计，一次查表、一次启动开销。不构造跨 head 的无效稠密乘法，不按 head 重复启动。上述步骤按顺序计时，不自动假设 Q/KV 投影跨 stream 重叠或额外融合。

每个算子：

```text
variable_us = max(FLOPs / compute_peak, bytes / memory_bandwidth) * 1e6
latency_us  = variable_us * calibration_multiplier + launch_overhead_us
```

decode 核心中 S_i 为本步参与 Attention 的完整 context 长度：

```text
matmul_FLOPs = 2 * H * sum(S_i) * (2*r + dr)
softmax_FLOPs_approx = 5 * H * sum(S_i)
cache_read_bytes  = sum(S_i) * (r + dr) * cache_dtype_bytes
cache_write_bytes = B * (r + dr) * cache_dtype_bytes
activation_bytes = B * H * (2*r + dr) * activation_dtype_bytes
```

B=1、context=16384、H=128：cache 常驻量及一次逻辑读取均 **18 MiB**，本步写 **1152 B**，矩阵乘约 **4.563 GFLOPs**。Q latent rank 不进入核心公式，只影响 Q 投影。

prefill 只支持等长完整序列。P=sum(S_i*(S_i+1)/2) 为 causal 有效配对数：

```text
matmul_FLOPs = 2 * H * P * (dn + dr + dv)
softmax_FLOPs_approx = 5 * H * P
activation_bytes = 2 * T * H * (dn + dr + dv) * activation_dtype_bytes
cache_write_bytes = T * (r + dr) * cache_dtype_bytes
```

KV expansion 单独计时。核心假设 fused softmax、score 不落 HBM、cache 被 heads 共享读取一次。实际重复加载、tile padding、split-K、特殊单元吞吐、occupancy、YaRN/scale 可能产生偏差；softmax 暂用 activation peak 估计，需实测补足。

## 数据表与补测契约

新增可选 `mla_attention`、`batched_gemm` 表，旧 schema v2 表保持兼容。MLA 缺表/缺核心 shape 明确报错，**不在 runtime 回退公式或 GQA**。`match_type=exact` 只说明 shape 精确命中；是否实测看 `timing_method` 和 `source_id`。

核心 key 包含 device、backend、activation/cache dtype、phase、execution_mode、batch_size、query_len、完整 context_lengths 向量、heads、KV latent/RoPE/noPE/V 维度。核心暂不插值；GEMM/grouped GEMM 沿 m 在同离散 shape 内插值，不外推。

测量边界：

- `gemm`：独立投影，包含自身启动，不包含邻接 norm/RoPE。
- `batched_gemm`：本卡全部 head 的矩阵乘批量执行。
- `mla_attention/decode/absorbed`：压缩 Q/K dot、RoPE dot、softmax、压缩 value 聚合和本步 cache 写入；排除前置 absorption、后置 value expansion/O projection。
- `mla_attention/prefill/expanded`：已展开 Q/K/V 的 causal Attention 和压缩 cache 写入；排除 KV expansion。
- norm、RoPE、通信：使用输出 shape 清单中的维度、dtype 和端点。

若后端 cache 更新是独立 kernel，可将其顺序执行时间和核心共同记录为此表项。若后端融合跨越多个本实现组件，先对齐边界，不能将融合总时间重复写入多个组件。记录 GPU、kernel/库版本、CUDA graph 状态、预热、重复次数、分位数及包含的操作。

通过 `SourceRecord(method="measured", evidence_grade=A)` 和 `Observation` 重新生成 package；实测优先，不再添加理论 multiplier/启动时间。测试中的 synthetic observation 不是设备实测。后续首先补 MLA core、两种 grouped GEMM、小 batch 投影/启动时间和链路参数。`measurement-backlog.json` 提供全部待测 shape。

## 运行

```powershell
# 单层 V3 形状，虚拟 1A1F
python scripts/run_deepseek_mla.py --output outputs/deepseek-mla-single-layer
# 全 61 层，虚拟 1A32F，F TP=EP=32、expert TP=1
python scripts/run_deepseek_mla.py --full-model --output outputs/deepseek-v3-61layers
# 更换批量、context 和 AF 小批次
python scripts/run_deepseek_mla.py --a-tp 2 --batches 1 4 16 --context 4096 --af-batch-size 4
```

全部在 CPU 上模拟。BF16 V3 权重不能装入四张 80GB 卡，虚拟 33 卡不代表现有四卡可运行；保留容量检查。

decode 使用固定并发和预先填充的 KV cache；按 context 与 block padding 检查所有 A rank 的容量，不模拟请求到达、换出或动态 admission。容量账本包含权重和 KV，不精确估计 kernel workspace 峰值。表内 shape 扫描的占位耗时仅用于离线发现查询键，会丢弃；正式结果由生成后的 package 重新运行得到。

假设集中在 `data/operator_data/deepseek-v3-theoretical.json`：A100 级峰值计算/带宽、每算子启动 3 μs、传输固定 5 μs、每卡 250 W 都未按 PKUa100 校准。250 W 是可改功耗假设，不是测量结果，模型结构文件不混放卡数功耗。

输出 model manifest、cluster、容量/分片账本、shape 清单、带公式来源的 operator-data、Chrome/Perfetto 逐算子 trace、summary 和补测清单。trace 均为**模拟时间**。`component_service_us_sum` 是跨资源服务时间总和，包含并行工作；端到端时延看 `avg_step_us`。

decode 使用正式 AF runtime。prefill smoke 仅验证单层 MLA Attention、每个 A rank，不报告完整 TTFT。既有正式 layer graph 仅用于 decode。cached/chunked/ragged prefill 会明确报错。
