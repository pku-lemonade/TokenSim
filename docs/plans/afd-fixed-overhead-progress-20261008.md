# 固定开销续作状态（2026-10-08）

本轮实施 `align-vllm-afd-protocol-and-timing` 的 11.1–11.4；整体仍为 24/34。下列代码接口和合成 oracle 已验证，新增实测参数尚未正式接入，不代表完整校准完成。

## 实现

- operation profile v2 支持 `sampled_token_readback`：token selection 的设备完成后，在 A host 上执行完整回读；CPU token 可用后才允许输出处理和本步结束。标准 LLMEngine 使用同一路径，取消时释放资源。
- host profile v2 按原生计算调用组计费；grouped experts 不按专家数量重复提交，已覆盖的 combine 不再次提交。消息提交区分 PyNccl/torch.distributed NCCL、方向和发送/接收。
- 每步调度在 A 准备之前；每步输出处理在 token 回读之后。沿用原有 host 资源和依赖，不另建调度系统。缺失 readback 的输出 profile、旧显式通用成本与新调用组成本的覆盖冲突均拒绝。
- v1 保持原 batch4 留出约束；v2 使用 batch2 新留出，旧 batch4 改作新一轮回归。旧 profile 文件、冻结 bundle 和历史验收均未覆盖。
- 新拟合工具 `scripts/fit_afd_sampled_readback.py` 校验六个独立窗口、ready/完成边界、GPU UUID、源代码、token dtype/shape、样本范围及有限非负耗时；缺窗口或等待污染拒绝拟合。新组合 profile 保留旧 profile 内容指纹，验收重置 pending。

## 原生采样及阻塞

远端目录：`/home/youwei/autodl-tmp/af-workspace/profiling/afd-sampled-token-readback-20261008-v1`。
仅修改隔离插件副本。原生 `_to_list` 前先在计时外同步设备，测量原 API 至 CPU 数据可用。每窗口 200 样本；前序计算/通信等待不进入常数。

|窗口|样本数|回读中位数（µs）|重复输出一致率|
|---|---:|---:|---:|
|batch1/window0|200|58.9995|100%|
|batch8/window0|200|51.5010|100%|
|batch1/window1|200|59.8150|100%|

小型原始数据已保存 `outputs/afd-sampled-token-readback-20261008-v1/b*-w*.json`。这些是部分窗口的诊断统计，不是最终参数；没有用三份数据冒充 batch1/8 各三窗口。

batch8/window1 的 A worker 在 KV cache 初始化退出：`No available memory for the cache blocks`，日志报告可用 KV cache 为 -22.06 GiB。复查 GPU2 被其他用户 iflab 的 PID809831 占用约70928 MiB；GPU3空闲。没有停止该任务或重启测试，也未增加额外 GPU。空闲检查无法预留资源，启动阶段仍可能被其他任务同时占用。

## 验证与后续

- 新调用组、后端/方向计费、每步调度/输出顺序、重复覆盖拒绝、cache/trace等价和拟合证据拒绝共5项专项通过。
- AF 回归305项通过，日志 `outputs/afd-fixed-overhead-regression-20261008-elevated.log`。首次沙箱运行有3项临时目录权限错误，正常本地权限复跑通过。
- OpenSpec strict validate 通过。
- 后补标准engine回读及回读取消测试后，operation/native-call专项24项全部通过。

需要继续：空闲时在新目录补齐剩余回读窗口；按原生调用审计测量计算提交、分backend消息提交和引擎调度/输出CPU成本；参数实际加载后做三个独立消融与全部启用对照，再冻结并执行batch1/8、batch4回归、batch2留出各三窗口。

CPU提交与设备服务仍使用现有DAG资源近似，未宣称精确还原原生stream提交时间线。同步API中的外部等待不能作为host常数。不得使用整步差额补参数。新增三类成本未完成实测接入，11.1–11.4保持未勾选。
