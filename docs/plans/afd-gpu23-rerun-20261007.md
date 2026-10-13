# GPU2/3 NVLink真实AFD引擎复测，2026-10-07

用户明确确认后完成batch1/8/4复测。两张卡每次启动前检查无计算进程，使用隔离目录及端口，没有修改插件、权重或模拟器参数。

沿用tiny Qwen3 MoE、BF16、2层、1A1F、输入128/输出32、eager与固定cohort；native prefill仅用于建立KV。每点预热后进行至少15秒无profiler计时，并另采集一个32步profile窗口，排除其中prefill后分析31个decode步。该实验不是DSV3或prefill分离验收。

## 无profiler完整decode步

单位ms。不同GPU组合在不同时间采集，不是同时控制实验；百分比只是观察差异，不能全部归因于链路。模拟结果仍为旧的未校准参数。

| Batch | 原GPU0/3 SYS | 当前GPU2/3 NVLink | 相对变化 | TokenSim | 实测/模拟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.772 | 3.709 | -1.7% | 0.190 | 19.56× |
| 4 | 3.912 | 3.862 | -1.3% | 0.191 | 20.22× |
| 8 | 4.046 | 3.844 | -5.0% | 0.192 | 20.06× |

## 计算与NCCL占用

单位µs，逐decode步kernel时长和的中位数。NCCL占用包括对端等待及采样影响，禁止当成纯通信服务或与整步时长直接相减；跨进程可能重叠。

| Batch | A计算 | F计算 | A NCCL占用 | F NCCL占用 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 101.76 | 52.23 | 78.85 | 4481.32 |
| 4 | 134.75 | 58.43 | 97.34 | 3727.07 |
| 8 | 140.42 | 59.14 | 53.34 | 3736.16 |

## 采样扰动与一致性

| Batch | 无profiler重复次数 | A profile步中位数 µs | 无profiler步中位数 µs |
| --- | ---: | ---: | ---: |
| 1 | 124 | 6163.11 | 3709.44 |
| 8 | 115 | 4986.11 | 3844.31 |
| 4 | 115 | 5142.20 | 3862.35 |

全部无profiler重复的输出token一致，与此前GPU0/3输出也完全一致。每个角色恰好识别31个decode步，无未归属kernel。表中profile步跨度用于显示观测扰动，不能作为无profiler引擎延迟。真实路由尚未回放给模拟器。

## 结论与尚缺部分

- 在当前tiny模型工作负载下，换回NVLink后完整步仍为毫秒级。结合已完成的提前接收数据＋回执约22µs微基准，不能把完整步差额简单解释为数据传输带宽不足。
- 计算kernel、纯通信候选测量、NCCL含等待占用和完整引擎步分别保留，未用一个指标替代另一个。精确分解仍需CPU调度/提交/完成服务、metadata消息次数和接收就绪事件的关联。
- 没有将NCCL占用导入transport_service，没有把回执RTT直接除二或将send局部完成当成recv完成。没有发布校准profile，也没有宣称完整引擎已达到数量级一致。
- 各batch本轮只有一个独立无profiler计时窗口，窗口内包含多轮cohort；规范要求的每点三个独立无profiler窗口尚未满足。

## 证据

同名JSON保存原生摘要、case/result、源码哈希及对照计算；原始trace、calls和tokens保留服务器，仅拉取小摘要。

远端目录：`/home/youwei/autodl-tmp/af-workspace/profiling/afd-gpu23-rerun-20261007`。真实driver为该目录`run_curve.py --concurrency B`，按1/8/4运行；使用`envs/afd026/bin/python`及`AF_WORKSPACE=/home/youwei/autodl-tmp/af-workspace`。

远端`summarize_gpu23_rerun.py`只读汇总；本地`python scripts/report_gpu23_rerun.py`验证并生成报告。通信微基准单独见`nccl-gpu23-20261007.md`。
