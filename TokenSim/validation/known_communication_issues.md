# 通信已知问题与修复状态

通信提交 `4f42e71` 及后续复查确认了以下实现问题。它们已在变更 `fix-af-communication-correctness` 中修复并通过相应回归；原三个问题与后追加节点归属修复的验证记录分开保存，此状态不表示其他通信代码或硬件近似已无缺陷。

1. **已修复：collective 的全局输入等待。** 原行为在两节点各四 rank、第二节点一张卡晚 100 µs 时，第一节点 local ReduceScatter 也等到 100 µs。现在 planner 保留 endpoint→producer 绑定，physical 入口阶段仅等待自身子组输入；后续 phase 与完整输出 join 不提前释放。`test_ready_subgroup_starts_at_zero`、子组/host/失败清理矩阵及模板重新绑定测试覆盖此修复。
2. **已修复：阶段测量的硬件适用范围缺失。** 原行为会将 A-only 7 µs 记录用于不同的 F 设备并标为 exact。现在保存来源设备、实际包 backend、同质或显式异构适用组合及独立 transport backend，参与匹配与缓存身份。`test_a_measurement_does_not_price_f_device` 和 `MeasurementApplicabilityTests` 覆盖设备别名、backend、异构、冲突及回退。
3. **已修复：flat All-to-All 的校验与 locality 语义不一致。** 原行为接受 `remote_fraction=-1` 和 FP16 的 3-byte payload。现在两模式均在 no-op/缓存返回前校验；flat 仅接受默认均匀参数，显式 locality、非默认 imbalance 或 redistribution 明确拒绝并提示使用 physical。两个最小 flat 反例及 `AllToAllContractTests` 覆盖公式、错误、流量守恒和零资源请求。
4. **已修复：旧拓扑入口与通信域节点归属不一致。** 旧入口按网络内 worker 数生成 level-0 边界，provider/资源管理器却按默认 8 卡解释物理索引，可能将两个节点误合并。AF physical 现在要求显式 8 卡库存，统一构建并校验 placement，两个消费者共用 `TopologyPlacement.node_id()`。slots 0/8 的 300 MB 跨节点传输为 6 ms，slots 0/7/8/15 一致映射到节点 0/0/1/1；旧 placement、缺失映射与非法卡位在执行前失败。此项属于后追加的第 6 组任务，独立证据见 [placement_correctness_review.md](placement_correctness_review.md) 与 [placement_correctness_validation.json](placement_correctness_validation.json)。

此前 575 项回归和 V1–V13 报告是当时工作区的验证记录，其中还包含未在本次提交中纳入的 AFD 校准与控制协议改动。独立提交快照的测试结果应以本次提交说明及 `communication_commit_regression.log` 为准；旧报告中任务全完成的状态不代表以上场景已经覆盖。

`4f42e71` 的原通信独立快照通过仓库回归 432 项和通信域专项 27 项，共 459 项，当时没有覆盖这三个问题。本次修复在原提交导出快照上记录四个失败断言，修复后的完整工作区通过 551+46=597 项，独立通信修复快照通过 432+46=478 项；最新专项包含原 27 项及新 19 项。

具体范围、场景映射与验证命令见 [communication_correctness_review.md](communication_correctness_review.md)，计数与源码哈希见 [communication_correctness_validation.json](communication_correctness_validation.json)。旧接口没有输入绑定时仍使用完整输入 barrier；只适用于正式绑定及其测试范围的修复，不能据此声称所有接口均获得新的子组执行粒度。

后追加的固定 8 卡节点修复通过新的完整工作区 551 项仓库测试与 56 项通信专项，共 607 项，包含新增 10 个 placement 测试方法；证据见 `results/communication_correctness_placement_apply.log` 和 `placement_correctness_validation.json`。该记录包含工作区原有 AFD 修改，不是新的独立通信快照，也不替代上文原 17 项的历史证据。

另一个已披露的模型限制是粗阶段持有资源并集直到结束，可能增加后到任务的等待。它属于当前模型近似，不能与上述实现缺陷混淆。

待修复：P2P 静态计划缓存键未包含 dtype。本轮固定 8 卡放置任务未修复这一独立问题；不能由本轮测试通过推断通信代码已无 bug。
