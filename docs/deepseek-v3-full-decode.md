# 完整 DeepSeek-V3 decode 仿真

`scripts/run_deepseek_v3_full.py` 在 CPU 上模拟一次或多次完整模型 decode，不加载 checkpoint，也不调用真实模型生成。与此前 V3 单个 MoE 层实验不同，执行路径为：

`embedding → 3 × (MLA + Dense FFN) → 58 × (MLA + MoE) → final RMSNorm → lm_head → greedy argmax`

层间依赖逐层执行，多个 AF 微批次竞争真实配置中的同一组虚拟资源。不是把单层时间乘以 61。输入、输出算子按微批次执行，和 Attention 共用 A 卡计算资源，输出投影不会免费并行。每步等待整批请求完成再开始下一步。

## 默认配置

- 模型结构：官方 V3 的 61 层、hidden 7168、vocab 129280、256 个 routed experts、top-8、1 个 shared expert；BF16 权重/激活/KV。
- 虚拟 1A32F，每卡按现有 A800 的 80 GiB 容量建模；F TP=EP=32，单个 routed expert TP=1，dense/shared FFN 使用 F TP。
- Router 在 F 主卡，Combine 在 A，固定该位置，不扫描 Router。
- context=4096，并发 1、8、32，AF 微批次上限 8，默认 1 个 decode step。
- 每卡固定假设 250 W，33 卡合计 8250 W；不含 CPU/网络功耗。

33 张是本次选定的虚拟拓扑，不是宣称所有精度与切分方式的最小卡数，也不要求物理服务器有 33 张卡。容量检查保留：两卡 BF16 完整模型应报错；超过 A 卡完整 61 层 KV 容量的 batch/context 同样报错。账本未精确估计临时 workspace。

### 1A1F 容量理想化诊断

完整模型的 F 侧权重为 1,315,517,235,200 bytes（1225.17 GiB），真实 80 GiB 的单 F 卡无法容纳。若仅研究一份 A/F 计算资源的调度曲线，可显式传入 `--f-gpus 1 --hypothetical-f-memory-gib 1536`：仅把 F 容量改为假想的 1536 GiB，A 容量仍用原配置，算力、内存带宽、通信和固定功耗假设保持原值。F 使用独立的容量标识，算子耗时仍查原 A800 表；容量检查没有关闭。该模式不模拟权重卸载，也不增加假想大容量存储的功耗，因此不能视为两张真实 A800 的可部署性能或真实能效。图标题、硬件假设和 summary 均标注此限制。

## Router 与通信

新增的多 F 卡路径先把完整微批次激活发给所有 F ranks，不能在 Router 执行前利用预先生成的路由结果省略目的卡。F 主卡收到完整输入后计算路由，向其他 F ranks 和每个 A Combine rank 发送每 token 的 top-k int32 expert IDs 与 float32 weights（top-8 时 64 bytes/token/目的卡）。远端专家等待本地激活和路由信息到达；A 侧 Combine 等待返回的路由权重和专家结果。

这一实现使用显式共享 fabric 假设：F→F 路由消息与 F→A 返回占用同一方向资源。完整模型入口支持 `--a-gpus 1/2/4/8`，表示同一请求的 A tensor parallel；F DP 仍为 1。所有跨卡通信使用声明的共享资源与带宽，不代表已标定的真实多机 NVLink 网络。

多 A 将 MLA query heads 及相关上投影、输出投影按头切分；Q/KV 降维投影与压缩 KV cache 保持每卡副本。输入/输出权重按词表切分，与权重账本一致。Embedding 每 rank 保守计一次完整行读取/输出，再 all-reduce 合并 masked lookup 的结果；final RMSNorm 每卡执行，lm_head 的输出宽度为 vocab/A_TP，然后 all-gather 完整 logits，各 A rank 执行 greedy argmax。两个额外 collective 的 host submit、传输与资源争用均计时。未采用仅交换最大值候选的输出优化。负 layer ID -1/-2 在 trace 中专指输入/输出通信，仍接受资源和依赖检查。

整模型实验使用 seed=42 的均匀随机无放回 top-8，种子由请求、token、层和 decode step 决定。改变 AF 微批次边界不会改变同一 token 的专家选择。原有单层入口仍保留旧路由规则；不能把旧规则每批从 expert 0 开始的负载集中误认为真实多卡瓶颈。该合成路由仍不代表 V3 的 noaux_tc。

通信沿用三部分建模：每消息源端 host submit 5 μs；链路按 payload/25 GB/s、额外链路固定服务 0；同步和排队由依赖及资源竞争产生。上述值是配置假设，不是新硬件标定结果。

## 时延数据与边界

严格匹配已有 A800 runtime shape 时复用测量值，holdout 数据不进入运行时。缺少的 shape 由离线 Roofline 加假设 3 μs 启动开销生成，并在 `measurement-backlog.json` 中列明。实测值不会再次添加该启动开销。由于专家分布和 TP 形状改变，不能把旧的 1A1F grouped expert 测量直接当作多 F 卡实测。

`shape-provenance.json` 对每个查询形状记录 measured/analytical 来源；完整模型尚未整体校准。旧 eager 测量包含 host gaps，重复暖权重测量也不等于完整 61 层的实际缓存行为。

这轮模拟涵盖完整主模型的 decode 算子链，但不是官方实现的逐项复刻：使用 synthetic uniform top-8，不实现 noaux_tc 分组路由；采用 BF16，非官方 FP8 kernel；选择 greedy argmax，不模拟其他采样策略；不含 MTP/投机解码。KV 预先填好，不包含 prefill、TTFT、动态到达或服务端 CPU/API 开销。

## 运行与产物

```powershell
python scripts/run_deepseek_v3_full.py
python scripts/run_deepseek_v3_full.py --batches 1 2 4 8 16 32 --contexts 4096 --af-batch-sizes 4 8 16 --steps 1 --output outputs/v3-full-sweep
```

- `summary.json`：整批 decode step 时延、tokens/s/user、总 tokens/s、tokens/s/W。
- `curves.png` / `curves.svg`：整模型 decode 吞吐–交互速度曲线。
- `traces/*.trace.json.gz`：压缩 Chrome/Perfetto profiling 文件，标明 SIMULATED；包括 61 层、计算组件、host submit、传输、同步、排队和输入/输出算子。
- `runtime-plan.json` / `cluster-af*.json`：显存账本和可复现的虚拟拓扑。
- `operator-data/`、`shape-provenance.json`、`measurement-backlog.json`：离线耗时表、依据与待补测项。

图的横轴与 NVIDIA 图 11 一致，标作 `Interactivity (TPS / User)`，数值是完整 decode step 耗时的倒数。纵轴标作 `Throughput (TPS / MW)`，计算总 TPS / 假设总 MW 后，按本图所有点中的最高 TPS/MW 统一归一化，显示 0.0x–1.0x 刻度。图脚与 `plot-axis-metadata.json` 给出 1.0x 的绝对 TPS/MW 和基准点；这一自有基准不等于 NVIDIA 未公布的绝对基准，不据此比较绝对性能高低。

`summary.json` 保留绝对 `tokens/s/W`，乘以 10^6 即为 `TPS/MW`。它们是整模型仿真预测，不再是单层等效数值。各层或各算子的服务时间相加可能包含并行部分，不能用作端到端时延。

执行器验证每一层的 DAG/资源约束；额外检查 embedding 在第 0 层之前、final norm 在第 60 层之后、输出投影及 argmax 顺序、输入输出与层计算不重叠占用同一个 GPU，以及 token 完成时间包含 argmax。
