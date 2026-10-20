# Groq 3 LPX 校准与 Figure 11 实验方案及运行说明

本方案包含 Artificial Analysis（AA）单并发校准和 NVIDIA Figure 11 的四种推理部署。
配置与资料核对日期为 2026 年 10 月 19 日。

## 已知条件与复现范围

博客披露的负载和硬件代际保持一致。
未披露的部署参数分别作为假设记录，避免将某个拟合成功的配置认作原始配置。

| 实验 | 公开条件 | 尚缺的条件 |
| --- | --- | --- |
| AA Gemma 4 31B | Reasoning 模型；10K、100K 输入；每组 50 次串行请求；客户端并发 1；输出速度中位数分别为 3,382、3,431 tokens/s；使用 `o200k_base` | 原生 token 数及原始文本、实际 LPU 数、并行布局、精度、服务端 batch、GPU 分工、投机解码和运行时版本 |
| Figure 11 | GPT-OSS 120B 扩展至 2T 参数；缓存输入 400K、新输入 4K、输出 400；Rubin 基线与三种 Rubin/LP30 组合；结果属于性能预测 | 模型扩展方法、激活参数量、精度、机柜数与配比、并发数、横轴统计量、功耗范围、draft 模型和接受率 |

AA 测试的是 NVIDIA 提供的私有预发布端点，经 Google Cloud PSC 访问。
请求条件来自 [100K 图及脚注](https://developer-blogs.nvidia.com/wp-content/uploads/2026/08/nvidia-groq-3-lpx-gemma-4-100k-artificial-analysis.webp)和 [10K 图及脚注](https://developer-blogs.nvidia.com/wp-content/uploads/2026/08/nvidia-groq-3-lpx-gemma-4-10k-artificial-analysis.webp)。
[AA 通用方法](https://artificialanalysis.ai/methodology/performance-benchmarking)补充了负载、token 和计时规则，但没有提供这个端点的部署清单。
[NVIDIA 正文](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/)说明 Figure 11 为预测，负载长度与坐标轴见[原图](https://developer-blogs.nvidia.com/wp-content/uploads/2026/08/gpt-oss-2t-interactivity-throughput.webp)。

部署清单齐备时，结果可标记为 `published-configuration-reproduction`。
存在重要假设时，结果标记为 `published-workload-hypothesis`。

## 硬件、结构与容量检查

| 单元 | 公开规格 | 配置要求 |
| --- | --- | --- |
| Rubin NVL72 | 72 GPU、36 Vera CPU、20.7 TB HBM4、NVLink 6 | 固定采用的规格版本，分别记录器件和整柜数量 |
| LPX | 256 LP30、128 GB SRAM、12 TB DDR5 | SRAM 与 DDR5 分开建模 |
| LP30 | 每颗 500 MB SRAM、150 TB/s SRAM 带宽 | 保留十进制容量与单位 |
| LPX 算力和互联 | 315 PFLOPS FP8；约 40 PB/s SRAM 聚合带宽；640 TB/s scale-up 聚合带宽 | 区分计算精度、单链路、端点聚合及双向口径 |

规格来源为 [NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)、[LPX](https://www.nvidia.com/en-us/data-center/lpx/)及[架构说明](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/)。
150 TB/s 乘 256 为 38.4 PB/s，与 40 PB/s 的公布近似值分别记录。
整柜聚合带宽不能填为每颗芯片或 GPU 到 LPX 的链路带宽。
原有 Groq v1 的 215 W、16 端口及 FP16 算力不作为 LP30 实测参数。
恢复配置中的 LP30 BF16 算力、450 W 功耗、Rubin 1,800 W 功耗和跨柜有效带宽均有假设标记。
当前恢复路径只模拟 SRAM 驻留，不将 12 TB DDR5 当成同速 SRAM。
压缩权重的解包成本单独计入，四位存储格式不能直接使 LP30 的 FP8 算力翻倍。
LP30 的 2.5 TB/s 双向聚合注入拆为每方向 1.25 TB/s，单条 112 Gb/s 链路另列为 14 GB/s。
聚合注入预算只收取一次，不能再乘 96 个端口。

物理配置表示为 `m * NVL72 + n * LPX`，另列 prefill、attention、FFN、target、draft 的分配。
历史扫描配置按分配的物理 rank 计资源，不能自动视为整柜满配。
整柜对比必须明确未分配芯片的容量、空闲功耗及网络资源。
256 芯片机柜也不能直接视为 TP256；上下文分片是否需要额外支持应单独检查。

Gemma 4 使用[官方文本配置](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json)并固定版本。
该模型有 60 层、隐藏宽度 5,376、FFN 宽度 21,504、32 个 query head、262,144 词表和共享 embedding。
每五个窗口 1,024 的局部层后有一个全局层，总计十组。
局部层使用 head dimension 256 和 16 个 KV head，全局层使用 dimension 512 和 4 个 KV head，并包含 K/V 共享语义。
门控 GELU、Q/K 归一化、投影形状、KV 存储和输出操作需对照[参考实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py)。
公开权重的 BF16 类型无法确定 AA 的服务精度。

GPT-OSS-2T 的真实结构仍需 NVIDIA 提供。
恢复的宽度/层数与专家数扩展模型是两组敏感性假设，分别保存总参数、激活参数、FFN 参数和 attention 结构。
两个模型保留局部/全局 attention 与低精度专家权重的区分。
总参数接近 2T 无法单独确定计算和存储开销。

每个点运行前检查逐设备容量，包括权重、量化元数据、工作区和峰值 KV。
AFD 的 F 侧保存 FFN/专家分片和工作区，A 侧承担 attention 与请求 KV。
专家并行改变放置方式，但单个 token 仅激活部分专家不会消除其余专家的存储需求。
全部 FFN 权重常驻 SRAM 时，可用 `ceil(FFN_weight_bytes / usable_SRAM_per_rack)` 进行容量筛选，再检查实际 TP 复制与 EP 分配。
需要 DDR5 权重流送但缺少相应模型的点，标记为不可用。

## AA 单并发校准

每个虚拟 user 同时最多一条请求，完成事件触发后才发送下一条。
10K、100K 各测量 50 条成功请求，预热请求单独记录。
初始可使用五条预热，这是项目设置。
不同 AA 请求使用不同前缀，避免意外缓存复用。

| 用例 | 标准化输入长度 | 输出长度初始假设 | 目标中位速度 |
| --- | ---: | ---: | ---: |
| `aa-10k` | 10,000 | 1,500 | 3,382 tokens/s |
| `aa-100k` | 100,000 | 2,000 | 3,431 tokens/s |

输出长度使用 AA 通用负载的最低要求，原始 50 条请求长度尚未取得。
取得原始记录后替换假设；缺少记录时报告长度敏感性。
确定性相同请求重复 50 次，不能作为 50 个独立的精度样本。
`--max_num_seqs 1` 只限制 worker 内接纳数量，仍可能存在入口排队，无法替代串行客户端。

定义 `rho_in = native_input_tokens / o200k_input_tokens` 与 `rho_out = native_output_tokens / o200k_output_tokens`。
模拟输入使用 `native_input_tokens = rho_in * AA_input_tokens`，预测 AA 速度使用 `native_output_speed / rho_out`。
转换比例优先来自相同文本和固定 tokenizer；文本缺失时作为敏感性参数，不默认为一。
保存发送、接纳、首输出、末输出和实际 token 数，按每请求生成阶段速度取中位数。
同时保存 TTFT 和完整请求速度，保留 streamed chunk 与原生逐 token 计量差异。
隐藏推理 token 时，AA 使用末尾回答区间的规则也需要匹配。

分别筛选 64、128、256 LPU 的可行部署假设，固定同一配置跨两个上下文测试。
精度、并行布局、GPU 参与和投机解码状态分别记录，不能依靠两个观测值推断全部设置。
关闭投机解码的纯 LPU 驻留方案可以作为一组假设，前提是结构与容量可表示。

先保存未拟合结果及算子、通信、固定开销分解。
在固定结构、硬件和 token 假设下，只用 10K 拟合至多一个共享修正参数，再以 100K 留出验证；反向划分用于诊断。
拟合必须满足物理算力与带宽约束，两个上下文不分别调整系数。
现有算子 `calibrate` 命令针对逐算子测量，不能将两条整体速度直接作为算子记录输入。
新增校准包装时复用实验检查点与来源记录。

项目建议验收条件为留出速度误差不超过 10%，100K/10K 速度比相对误差不超过 5%。
公开速度比约为 1.01449，原始方差缺失，无法证明这点差异具有统计显著性。
单用户 dense 校准不能验证 MoE、高并发合批、Rubin、跨柜通信或功耗。

## Figure 11 负载与部署

按缓存 400,000、新增 4,000、输出 400 设置固定负载，名义输入 404,000，峰值上下文约 404,400。
图中 tokenizer 未披露，暂用原生 token 时明确标为计数假设。
同一比较组中的目标模型、精度和 token 假设保持一致。
每个 user 有独立的常驻前缀；初始缓存建立在测量窗口之外，缓存容量始终计入。
每轮新增 4K 参与对已有上下文的 attention，并生成 400 输出。
请求边界恢复到固定前缀；会话持续增长另作敏感性实验。
需要跨池 KV 交接的部署计入相应开销，分别记录完整交接和增量交接。

| 曲线 | 执行分工 |
| --- | --- |
| Rubin | GPU 完整 target 执行 |
| Rubin ATTN + LP30 FFN | GPU attention/KV、LPU FFN，逐层双向激活传输 |
| Rubin Prefill + LP30 Decode | GPU 完整 prefill、KV 交接、LPU 完整 decode |
| Rubin Verifier + LP30 Draft | 独立小型 draft、GPU target 验证、接受/拒绝及两份 KV 状态 |

恢复脚本已覆盖四条原生执行路径。
外部 draft 计入其 prefill、逐步生成、完整验证宽度、拒绝位置、KV 回滚及必要的补算。
只统计提交的 target token，候选和拒绝 token 贡献成本，不增加输出吞吐。
验证工作区按微批请求数乘验证宽度设置。
draft 长度 1、2、4、8 及接受率可分别扫描，接受率仍是未实测假设。

## QPS 容量扫描

QPS 表示到达率，user 数表示未完成请求槽位数。
最大近零入口等待 QPS 仅给出一个配置的容量边界。
同时保留满足入口排队预算的前沿和所有稳定点的前沿。

每个可行配置测量单请求完整耗时 `T_iso`，从均匀到达率 `0.05 / T_iso` 开始逐次翻倍。
出现积压增长或接纳延迟超标后，将边界细化到约 2% QPS 分辨率。
仅在可行性单调的区间使用二分；合批产生非单调现象时加密扫描并保存全部可行区间。
可用最大通过 QPS 的 95% 作为带余量的初始工作点。

建议通过条件如下，阈值均为项目选择。

- 稳态完成率与到达率偏差不超过 1%，窗口边界请求另行对账。
- 积压没有持续正趋势，测量时长加倍后仍满足条件。
- 入口接纳延迟 P95 不超过 `max(50 微秒, 0.01 * T_iso)`。
- 不隐藏拒绝、内存失败、逐渐增长的未完成请求或驱逐重计算。

入口接纳等待与 prefill/decode 交接、FFN 合批等待、设备和网络资源等待分别统计。
并发流水线可以包含内部资源等待。
有希望的区间使用 Poisson 到达和种子 0～4 复测，采用明确的排队预算。
当前通用 CLI 的均匀/Poisson 源可供开放到达实验使用；LPX 入口尚未加入自动容量搜索。

## 固定 user 数、并行运行与前沿构造

一个 user 定义为最多维持一条未完成请求的逻辑客户端，完成后立即提交下一条。
主实验思考时间为零。
user 数、测量请求总数、A 副本数和 decode 微批大小分别记录。
`--users` 指总 user 数，旧参数名 `--batches` 是同一设置的别名。
`--micro-batch` 默认一；大于一时组内请求同步推进。
`--a-replicas` 控制 attention 服务副本数，资源和功耗另行计入。
固定 user 扫描 `1, 2, 4, 8, 16, 32, 64, 128, ...`，停止于容量或延迟限制。
实际 QPS 是测量结果，稳定时用 `N = achieved_QPS * mean_request_latency` 检查计数。
生成阶段速度无法替代这个公式中的完整请求耗时。

补充扫描各配置容量的 `0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0` 倍 QPS。
固定 user 且限制速率时，下一条发送时刻取前一请求完成与速率截止时刻的较晚值。
保存目标 QPS、实际 QPS、user 数和主动等待时间，反压后无法达到目标值时明确记录。

每组物理库存内筛选合法 TP/PP/DP/EP、batch 上限 `1, 2, 4, 8, 16, 32, 64`。
AFD 还扫描 A:F 分配、`af_block_size` 和 FFN 等待窗口 0、5、20 微秒。
初始整柜候选可取 NVL72 数 1、2、4，LPX 数 1、2、4、8、16、32，并在运行前剔除容量失败点。
只在已观测的非支配区间继续加密，结果称为已搜索范围内的可行前沿。

不同物理配置、user 数、种子和负载点可在独立进程中并行。
同一 QPS 边界搜索依赖前一点结果，保留顺序；不同配置的边界搜索可以并行。
`--jobs 4` 表示最多四个宿主进程，不改变任何一点的模拟 user、资源竞争和功耗。
每个点有独立 SimPy 环境和随机流，主进程按配置顺序汇总结果。
检查点按参数、代码和输入指纹复用；中断后重跑相同命令，仅重新执行未完成或失效的点。
默认 `--resume` 复用完整提交的点，`--no-resume` 强制重算所选点。
主进程更新 `partial-results.json`，子进程仅提交各自检查点；未知运行错误报错，配置或容量不满足记录为 `infeasible`。
内存占用较高时减少 `--jobs`，避免宿主交换影响运行耗时。

正式扫描建议预热至少 `max(1 秒, 10 * T_iso)`，测量至少 `max(10 秒, 100 * T_iso)`，并取得至少 500 个成功请求、每个 user 至少 20 条。
这是正式实验的目标；恢复脚本的短测默认值只用于初步验证，运行时需显式提高窗口与完整轮次。
稳态 token 按提交时间落入窗口计数，不将跨界请求的整个输出计入。
随机实验使用种子 0～4；确定性重复不作为独立置信区间样本。

## 运行命令与参数

以下命令使用已恢复的固定 user 数入口；AA 精确校准和自动 QPS 搜索仍需实现。

完整负载的运行示例：

```bash
python scripts/reproduce_lpx.py \
  --cases rubin-tp8 af-tp8 pd-tp16 draft-tp8 \
  --users 1 4 16 \
  --micro-batch 1 \
  --measurement-turns 20 \
  --measurement-seconds 10 \
  --jobs 4 \
  --plot \
  --output outputs/lpx-study
```

下面的短测缩短输入和输出，保留长上下文容量检查，不能作为原负载的性能结果：

```bash
python scripts/reproduce_lpx.py \
  --skip-audit \
  --cases rubin-tp8 af-tp8 pd-tp16 draft-tp8 \
  --users 1 \
  --new-tokens 8 \
  --output-tokens 8 \
  --measurement-seconds 0.003 \
  --jobs 4 \
  --output outputs/lpx-smoke
```

专家数扩展模型使用对应配置：

```bash
python scripts/reproduce_lpx.py \
  --model gpt-oss-2t-expert-proxy \
  --config experiments/lpx/expert-sweep.json \
  --users 1 4 \
  --jobs 2 \
  --output outputs/lpx-expert-study
```

### 参数说明

| 参数 | 含义 |
| --- | --- |
| `--model`、`--draft-model` | 模型目录名称或 YAML 路径 |
| `--config`、`--cases` | 部署矩阵文件及选中的部署名 |
| `--cached-tokens`、`--new-tokens`、`--output-tokens` | 负载长度覆盖 |
| `--prefill-chunk` | 每次完整模型 prefill 的新增 token 数 |
| `--warmup-steps` | 开始测量前每组至少执行的预热 decode 次数 |
| `--warmup-mode` | `resident_decode` 提供未计时快照；`prefill_and_decode` 也执行预热后缀处理 |
| `--measurement-seconds`、`--measurement-turns` | 最短测量时间和每组完成轮数，两项均满足后关闭测量 |
| `--acceptance` | 每个候选位置的假设接受概率 |
| `--execution-mode`、`--graph-kernel-us` | GPU eager/graph 执行和假设的设备开销 |
| `--endpoint-transfer-occupancy` | 端点仅保留 wire 时间，或保守保留完整 service 时间 |
| `--handoff-mode` | `incremental` 或 `full` KV 交接 |
| `--rubin-watts`、`--lp30-watts`、`--facility-watts` | 显式功耗覆盖和设施附加预算 |
| `--rubin-reserve-bytes`、`--lp30-reserve-bytes` | 额外设备内存保留 |
| `--trace` | 为 user 数为一的点保存原生完整轨迹 |
| `--retain-timing-history` | 保存测量资源历史，关闭在线资源统计路径 |
| `--audit-only`、`--skip-audit` | 仅运行 AF 层检查，或跳过该预检查 |

## 指标、功耗与结果检查

| 指标 | 定义与状态 |
| --- | --- |
| 生成速度 | 普通逐 token 情况使用 `(output_tokens - 1) / (last_output_at - first_output_at)`；投机提交保留时间戳 |
| 完整请求速度 | `output_tokens / (completed_at - submitted_at)` |
| 系统吞吐 | 稳态窗口内提交的 target 输出 token 总数除窗口长度 |
| 当前恢复脚本横轴 | 全部测量 decode 区间的提交 token 数除以 decode user-seconds，尚未实现每请求 P50 |
| Figure 11 主图计划 | 每请求生成速度 P50，另报完整请求速度、TTFT 和 TPOT 分布；该选择仍属假设 |
| 纵轴 | 明确功耗口径后的 TPS/MW；没有功耗输入时保留吞吐结果并标记缺失 |

恢复配置的功耗采用器件 TDP 假设与 `--facility-watts`，不能称为实测整柜功耗。
最终功耗表覆盖唯一物理 GPU/LPU、prefill、draft、CPU/host、内存和网络；共址 A/F 不重复收费。
有数据时分别报告额定预算与负载功耗。
先输出绝对 TPS 和 TPS/MW，再使用同一个 Rubin 基准归一化全部曲线。
原图绝对归一化基准未公开，不能据独立归一化结果宣称绝对坐标复现。

每种模式分别输出非支配点，并保留全部原始点和不可行原因。
另计算跨模式的联合前沿。
按相同横坐标比较图中的纵向 3×，按相同纵坐标比较横向 3×/5×，这些倍率不参与参数拟合。
恢复的参考坐标是明确标注的手工取样，比较时记录取样误差。

### 输出与独立检查

`summary.json` 记录代码版本、工作区状态、模型与 draft、配置假设、资源分配、不可行点和各模式前沿。
`points.csv` 保留 user 数、生成速度、绝对输出吞吐、功耗和归一化效率。
逐点结果包含 token 提交、decode 区间、轮次、KV 交接和验证事件。
预热与排空输出不计入测量窗口。

```bash
python scripts/check_lpx_results.py \
  --summary outputs/lpx-study/summary.json \
  --original-workload \
  --export outputs/lpx-study/evidence
```

`--original-workload` 检查 400K/4K/400 和完整测量轮次，不能验证未公开的 NVIDIA 部署设置。
检查器复核库存、功耗来源、单位、测量边界、输出数、KV 字节及前沿覆盖。
`--plot` 或 `scripts/plot_lpx.py` 可生成原图取样、独立模拟及并排图。
原图与模拟图的归一化来源分别说明，手工取样不参与校准。

## 执行顺序与交付

补充 Gemma 精确结构、AA token 对齐与串行用户驱动，再执行条件性校准。
Figure 11 复用恢复的原生路径，依次进行配置容量检查、QPS 边界搜索、固定 user 扫描与非支配区间加密。
QPS 搜索与新统计口径在实现并验证后接入同一实验入口，避免维护第二套模拟器。

交付包括来源清单、校准误差与敏感性、容量搜索记录、逐 user 计时、缓存占用、绝对/归一化曲线及功耗假设。
逐点检查点保存在输出目录的 `.experiments/<指纹>/` 下，包含 `manifest.json` 和 `result.json`。
新增实现遵循 `CONTRIBUTING.md`，使用小规模确定性测试验证行为。
