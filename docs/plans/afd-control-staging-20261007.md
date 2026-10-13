# 空闲触发后的控制staging补测

2026-10-07 11:09 UTC监控发现GPU2/3均4MiB、0%、无计算进程。检查实验进程及端口未发现重复任务后，启动控制staging/读回实验；监控已按用户要求暂停。

使用GPU2作为A、GPU3作为F，batch1/8、原生179-byte控制消息。三独立进程窗口，每项20次预热、200次测量。每次计时前synchronize清除前序GPU工作；计时区域分别复现原生CPU bytearray/frombuffer到GPU及长度tensor构建、length.item()、payload.cpu().numpy().tobytes()。记录wall和thread CPU时间，不启用profiler。输出逐次校验，无网络传输、无batch4、无插件修改。

三窗口、两batch的wall中位数范围：

| 操作 | µs |
| --- | --- |
| H2D及长度构建 | 37.524–38.959 |
| 长度回读 | 10.666–11.2795 |
| 内容回读与CPU字节转换 | 19.002–20.016 |

这是ready-input CPU API区间，包含CPU/runtime、自身设备执行和同步完成；排除了前序GPU工作，不等于纯host或纯device服务。thread CPU时间也不能直接代替纯服务：它包含runtime执行且与设备工作可能重叠。因此本轮未把这些值写入严格host/device profile。不能用整步差额或wall减thread CPU时间推断链路等待。结果不足以证明剩余毫秒差距的来源。

前两个窗口后空闲保护曾触发；随后只读复查两卡均空闲、无进程，单独补跑不存在的window-2.json，未覆盖前窗。无法由该次保护确定是否为利用率采样残留。脚本后续增加有界重试，每次完整执行同一空闲检查，不绕过保护。

原始数据在服务器profiling/afd-control-staging-20261007-v1及本地outputs同名目录，小摘要见afd-control-staging-20261007.json。复现入口scripts/measure_afd_control_staging.py，输出目录要求全新。

尚未完成A/F准备、提交和回读的可归因纯服务采集；现有两项codec profile仍为部分校准，整步数量级未通过。本次补测不勾选完整4.4/4.5，也不消耗batch4留出。后续需要在原生执行中记录提交/返回、设备完成及消息身份的边界，并评估采集扰动，才能将混合区间拆入对应服务资源。
