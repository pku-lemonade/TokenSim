# 完整 V3 A/F 配置扫描

用户已接受配置并授权在 AutoDL-A800 服务器上执行。CPU 模拟，已有 A800 算子微基准为校准输入；不运行完整 checkpoint，不占用 GPU，不中断其他任务。

- V3 671B BF16，61 层加 embedding、final norm、lm_head、argmax。
- A TP=1/2/4/8，F TP=EP=32；Router F，Combine A。
- 各虚拟卡 80 GiB，保留权重/KV 容量检查；不使用假想扩容。
- 上下文 4096，首轮每点一个同步 decode step，比较单位 TPS/user 与 TPS/MW。
- 并发 1/2/4/8/12/16/24/32/48/64/96/128。
- AF cap=1/2/4/8/16/32，另对每个并发运行 AF cap=batch 的整批对照。cap>=batch 的点执行路径等价，显式去重并保存 nominal cap 对应关系。
- 每卡固定假设平均功耗 250 W；分别为 8250/8500/9000/10000 W，不含 CPU/网络功耗。
- 通信沿用每消息提交 5 us、传输 25 GB/s、额外固定传输 0；排队由资源依赖产生。A TP collectives 与 A→F 使用同一假设 fabric lane。
- 先通过多 A shape/collective/metadata/资源/容量测试，并核对 1A 三个历史点，再启用完整扫描。
- 原始点全部保存。每种拓扑选择 Pareto 点展示性能边界，并另画相同并发下最大总吞吐点与整批对照。所有配置用同一 TPS/MW 归一化基准，保留绝对纵轴图。
- 每点保存完整模拟 profiling、shape 来源、未测 shape 清单、配置、资源忙碌率和排队时间。
- 结果只解释 A800/V3 模拟的配比和流水收益，不宣称复现 NV 的 Rubin/LP30、2T 模型或 400K+4K/400 工作负载。
