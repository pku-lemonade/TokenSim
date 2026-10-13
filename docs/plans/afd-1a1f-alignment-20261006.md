# PKUa100 1A1F 首轮数量级对照（2026-10-06）

## 结论

已在 PKUa100 完成 batch=1、8 的 vLLM+AFD 原生完整 tiny 模型推理和当前 TokenSim 完整模型 DAG 仿真。计算耗时初筛处于同一数量级；完整引擎 decode 单步相差约 15–19 倍，尚未达到所有环节数量级一致的目标。不得用一个总倍率校准 GPU 或网络公式。

当前仅以最大/最小耗时比小于 10 作为“同一数量级”的工作判据，不等于精度验证。两个 batch 点为首轮诊断，非统计充分的广泛验收。原生 profiling 有开销，kernel busy time 与未开启 profiler 的引擎 step 分开报告。

## 固定条件与证据

- 用户选择小模型完整流程，首轮复用已在服务器部署的 qwen3-moe-tiny-random：2 层、hidden=256、4 query heads、2 KV heads、4 experts、top-2、无 shared expert、vocab=128，BF16。不是 DSV3，不代表生产模型准确性或语义质量。
- GPU 2 为 A、GPU 3 为 F，均显示 A100 PCIe 80GB，2↔3 为 NV12；不能使用旧脚本说明中的 A800 标注。
- vLLM 0.26.0、torch 2.11.0、transformers 5.17.0；AFD 包版本 0.0.0。使用服务器已有源码快照，其中存在实验补丁且没有 .git；关键文件哈希已保存，不能称为最新官方 commit 的原样测试。官方参考仓库为 https://github.com/vllm-project/afd-plugin 。
- TP=DP=EP=1，P2pNcclAFDConnector，gate 在 F，eager，固定 seed=42，prefix cache 关闭，A 侧 async scheduling 关闭。输入 128 tokens、输出 32 tokens、ignore_eos，固定 cohort batch=1/8。
- 原生每点预热 5 个 cohort；profiling 在预热中采集 32 steps。性能计时使用后续无 profiler 窗口：batch=1 为 120 次、batch=8 为 116 次，均至少 15 秒，重复生成 token 一致率为 1.0。
- TokenSim 使用本地 e0c19e6 基础上的当前核心源码及独立测试驱动，在服务器 tokensim311 环境运行。两层及 embedding/lm_head 边界均经过 ModelAFExecutor。采用 analytical_only，未用本轮实测回填；每个 decode 为完整 batch、不拆 block，context=129..159，共 31 decode steps。
- 模拟器设备以 catalog A100 为基础，HBM 带宽独立设为 1935 GB/s 的未校准假设；NVLink 仍为现有 catalog 估值。模拟器采用 synthetic routing，尚未回放 native assignments；TokenSim full-model DAG 不包含 vLLM Python scheduler/IPC/采样实现的全部 CPU 成本。

## 初步结果

单位为微秒；原生数值为中位数，模拟为 31 decode steps 平均。GPU compute 为已匹配 kernel duration 的和，不含 NCCL，不表示关键路径长度。TokenSim service 包含其算子启动假设，因此也不是严格同义的 kernel-only 估值。

| 项目 | batch=1 TokenSim | batch=1 原生 | batch=8 TokenSim | batch=8 原生 |
| --- | ---: | ---: | ---: | ---: |
| A 侧计算（模拟含 A combine） | 66.67 | 98.24 | 67.49 | 136.74 |
| F 侧计算（原生含 F combine） | 37.03 | 53.03 | 38.14 | 59.23 |
| 两侧计算合计 | 103.70 | 151.27 | 105.63 | 195.97 |
| decode 整步 | 193.60 | 3706.14 | 255.52 | 3855.63 |
| 首个 prefill step | 151.70 | 3901.45 | 189.60 | 4249.29 |

两侧计算合计实测/模拟约 1.46、1.86 倍。A/F 单独表项因 combine 边界不同仅供诊断；合计能减少这种归属差异，但仍有融合方式、Q/K norm 和路由未对齐的限制。decode 整步比为 19.14、15.09；prefill 比为 25.72、22.41。端到端不通过数量级初筛。

## 分段发现

1. 原生完整 F MoE 返回已 combine 的 hidden tensor；当前 TokenSim 返回 expert partials 后在 A combine。模拟每完整两层 decode 的 A→F 字节为 1024/8192（batch=1/8），F→A 为 2048/16384。原生按 hidden 输出边界预期 F→A 数据量约为前者一半，另有协议 metadata；该判断来自源码边界，不是本轮实测逐消息 bytes 计数。必须先统一协议边界，再对通信速度做结论。
2. 模拟 A→F 两层合计约 30 微秒；F→A 为 60/120 微秒，发送粒度随专家分布变化。不能直接与原生两次返回比较，而忽略发送次数。
3. 原生 A 上 NCCL kernel 占用中位数为 309/64 微秒，F 上为 4022/4943 微秒；存在很长的跨 cohort/等对端尾部。该值包含等待和 profiler 扰动，禁止作为纯传输估值输入，A/F 两端也不能直接相加。
4. 通用 torch.distributed NCCL ready-peer 微基准已完成：512..8192 bytes、双方向、20 次预热后 100 次采样，sender CUDA event 中位数约 73–80 微秒。它包含 Python/ProcessGroup 提交与对端启动偏差，且插件使用 PyNcclCommunicator；因此该值只作诊断，不能判定 native 传输为 73–80 微秒。
5. 补测相同 PyNcclCommunicator 时，空闲保护发现 GPU 3 被新进程占用（PID 3426286，约 31GB），未启动测试，也未停止该进程。插件同款、双方就绪的通信时间仍待补测。
6. 3.7–3.9 毫秒的整步时长与百微秒级计算 busy time 之间存在明显缺口。CPU 调度/提交、同步、metadata 和 idle 都可能贡献，当前不能把两次独立测量的差值精确称为“CPU 时间”或“网络排队”。需要同一时间线的关键路径分析。

## Trace 方法与限制

服务器端原地解析 trace，不回传大数组。通过 GPU kernel 的 External id 关联 CPU op，再归属 ProfilerStep。A 的 decode 通过 split-KV kernel 标记，并排除 aten::mm 第一输入为完整 prefill 激活的 step；F 通过 topk_softmax 输入 token 数识别。两端每配置均识别出 31 decode steps，A 排除 1 个 prefill step。

A trace 每配置有 32 个未匹配 kernel，尚未确认其归属；F 为 0。报告明确保留这一计数，GPU 汇总尚不能宣称覆盖全部 kernel。不能把 profiler 中最大的 NCCL 时间用于拟合，也不能把不同进程的 ProfilerStep 编号默认当作严格对齐时钟。

## 下一轮顺序

1. GPU 2/3 空闲后，执行相同 PyNcclCommunicator 的 ready-peer 微基准；保留双方结果、发送/接收方向、metadata 与数据消息区别。不要改用另一对 PCIe 卡拼接 NV12 基线。
2. 明确 TokenSim 的 native 对照模式：F 侧 combine、返回完整 hidden、相同 grouped expert 调用粒度和实际路由；保留当前原始预测，不用本轮数据反向伪造基线。
3. 补齐未匹配 kernel、Q/K norm 与同一窗口 CPU/GPU 关键路径分析，单列引擎 overhead。对带等待的 NCCL kernel 继续拆分，不能归入纯 service。
4. 对齐后重跑 batch=1/8，并增加未参与任何拟合的 batch=4 holdout。达到单环节 0.1..10 倍再扩展负载；tiny 模型通过后仍需更大模型验证，不能外推为 DSV3 已验证。

## 复现与文件

远程根目录：`/home/youwei/autodl-tmp/af-workspace/profiling/afd-alignment-20261006/`。

- `manifest.json`：版本、模型配置、GPU UUID/拓扑、驱动和关键插件文件哈希。
- `c1/`、`c8/`：命令、输入、无 profiler 的逐次计时、输出 token、A/F trace；单 trace 约 5–9MB，留在服务器。
- `simulation/`：服务器运行的当前 TokenSim 预测与逐事件记录。
- `native-summary.json`：两端 kernel 与无 profiler 引擎汇总。
- `comm-rank0.json`、`comm-rank1.json`：首版通用 NCCL 微基准结果，必须按上述限制使用。
- 本地小型摘要：`outputs/afd-1a1f-alignment-20261006/`。本次模拟器代码/配置压缩传输仅约 0.26MB，未传模型。

本地新增 scripts/prepare_pkua100_alignment.py、run_tiny_afd_alignment.py、inspect_afd_trace.py、summarize_afd_alignment.py、measure_ready_peer_nccl.py。所有本次引擎进程均已由驱动清理；后续外部 GPU 占用未被修改。未更改 simulator 核心公式或插件源码，未提交新代码。
