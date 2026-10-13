# GPU2/3 原生复测、真实路由回放与双后端诊断

2026-10-07，PKUa100 的 GPU2/3 空闲后执行。真实运行使用既有隔离 vLLM+AFD 副本、tiny Qwen3 BF16、1A1F、输入128/输出32、F router；本地 TokenSim 仅 CPU 模拟。没有修改原插件，没有拟合新参数。

## 无 profiler 计时

每个 batch 三个独立窗口，计时阶段 profiler 关闭，输出重复一致性均100%。decode 排除首个 prefill step；每个请求含31个 decode step。

| batch | 三窗口平均 decode（µs） | 真实路由模拟（µs） | 实测/模拟 |
| --- | --- | --- | --- |
| 1 | 3794.61 / 3360.46 / 3385.85 | 229.60 | 14.64–16.53 |
| 8 | 3883.86 / 3575.00 / 3809.99 | 231.26 | 15.46–16.79 |

六窗口共23932个 decode 样本。完整摘要见 `afd-live-training-20261007.json`。尚未达到倍率小于10的整步目标。

## 真实路由及数据 buffer

两个独立诊断 case 均采集31个A请求身份记录、62个F逐层路由和62个buffer记录。严格转换通过：请求cohort、token位置、top-k、BF16 hidden shape/bytes完整。每个batch逐一对照62个模拟expert workload histogram，全部与原生路由一致；模拟31步共124条数据消息，每条batch1为512 bytes、batch8为4096 bytes，符合两层双向完整hidden契约。

batch1真实路由回放与合成路由的模拟结果相同；batch8从231.574变为231.263µs。因此这组实验的路由差异不能解释毫秒级差额。诊断采集有tensor CPU拷贝和文件写入，不能作为无扰动host service测量。

核对摘要见 `afd-live-routing-20261007.json`。原始小文件在 `outputs/afd-live-capture-20261007/`，回放输出在 `outputs/afd-live-replay-20261007-b1/` 和 `outputs/afd-live-replay-20261007-b8/`。metadata实际wire次数及精确调用边界仍需独立验证，任务2.4不据此整体勾选。

## 通信微基准

两方向各20次预热、100次测量，接收延迟0/5/20ms。PyNccl使用BF16消息512/1024/4096/8192 bytes；torch.distributed NCCL使用int64长度8 bytes及uint8 JSON 179/181 bytes。

| 后端 | 未注入接收延迟时，各消息/方向局部CUDA区间中位数范围 |
| --- | --- |
| PyNccl发送端 | 29.70–55.30µs |
| PyNccl接收端 | 37.89–56.32µs |
| torch.distributed发送端 | 72.70–223.23µs |
| torch.distributed接收端 | 71.68–234.50µs |

这些是局部CUDA事件区间，含host launch间隙且可能受peer状态影响，不是纯传输服务。接收侧start在注入sleep之后；发送侧局部完成也不能证明接收侧已可用。注入20ms后局部区间没有呈现20ms增长，不能因此判定端到端没有等待。此实验没有完整测到提交之前和跨端完成的时间线；所有记录明确 `admissible_for_transport_profile=false`，没有导入服务缓存或参数表。完整记录见 `afd-live-nccl-20261007.json`。

## 当前限制与下一步

GPU占用阻塞已解除，三窗口计时、真实路由回放、两后端诊断均已执行。仍缺无等待、无重复覆盖的host service样本，以及与两端消息身份对应的提交/API返回/设备可用时间线。这是测量证据缺口，不能用整步减GPU的差额或codec inclusive区间替代纯host成本。

下一步应补齐上述边界的低扰动测量，并先校验测量扰动；得到可辨识参数后冻结profile，再执行batch4三窗口留出验收。本轮未生成占位profile，也未消耗batch4留出数据；OpenSpec仍16/25。

所有原生测试均已退出。结束时只读检查GPU2/3均4MiB、0%利用率且无计算进程；此状态不构成资源预留。
