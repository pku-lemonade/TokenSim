# 统一通信等待统计续作

继续实现 align-vllm-afd-protocol-and-timing。远端只读检查发现GPU2约21083MiB、36%利用率，GPU3空闲，未启动双卡任务。已有codec实测profile继续加载；本轮未新增或伪造固定成本参数。

## 代码与口径

`TokenSim/af/timeline.py` 新增单观察窗口分解及按iteration消息索引的汇总。每条消息分别保留sender/receiver观察端、physical endpoint、起止边界及message identity；从该端模拟提交ready到模拟数据完成。只扣除该观察端在窗口内的固定服务区间与transport区间，其余为总等待。同类服务区间取并集，固定与transport重叠时报错，不把重复区间重复收费。另一端CPU工作可以表现为本端等待，不再次累加到整步。

接收提交事件缺失时不输出虚构的零等待接收记录。当前接收仍受既有按需提交近似限制，本报告不证明原生提前recv已经复现。统计仅使用本iteration，不扫描历史trace，不改变事件调度/缓存或资源占用。

对照驱动输出 `decode.communication_windows_seconds`。每个观察窗口满足：总时间＝总等待＋本观察端固定服务＋provider传输服务。目前provider传输服务仍含未分开的设备启动部分，因此显式报告 `full_fixed_transfer_split_available=false`；不能把整个transport_service称作纯字节搬运，也不能把其设备启动虚构成已校准固定项。完整固定/搬运分解仍待服务参数与scope补齐。

## 验证

- 20µs开始等待、120µs开始5µs服务：等待100、总耗时105；链路早于对端可用时等待80，不累加重叠blocker。
- 固定3、等待至10、传输5：总15，等待7；重复固定区间仅计一次；固定/传输区间重叠显式拒绝。
- 真实executor消息观察窗口逐条重建，统计不改变makespan。
- batch1/8分别回放31步真实路由并加载已有实测codec参数，各372个观察窗口全部满足分解等式；模拟仍为235.758529/237.425598µs，与修改前逐值相同。
- AF相关277项回归通过，OpenSpec strict validation和diff检查通过。小摘要见 `afd-unified-wait-20261007.json`。

3.1/3.2整项仍未完成：还缺实际调用返回/目标数据可用边界及完整原生验证。本轮统一统计不是精度校准；整体保持16/25。GPU2被占用阻塞双卡同路径采集，剩余A/F准备、提交、回读等固定成本仍不能用旧inclusive差额填充。
