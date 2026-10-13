# fix-af-communication-correctness 验收与边界

基线为通信提交 `4f42e71`。`communication_correctness_before.log` 在该提交导出的独立快照上执行四个最小断言，结果为四项失败：节点 0 的本地阶段从 100 µs 才开始、A-only 测量被 F 标为 exact，以及 flat 接受负 fraction 和 FP16 3-byte payload。测试源为 `test_communication_correctness.py`。

## 场景映射

| 规范 Requirement | 主要回归入口 | 验证内容 |
| --- | --- | --- |
| Collective phases respect subgroup input readiness | `CommunicationBugReproductions.test_ready_subgroup_starts_at_zero`；`SubgroupReadinessTests.test_allgather_slot_and_reduce_scatter_reorder`；`test_binding_validation_and_cycle` | AllReduce 本地独立输入、AllGather slot 输入、ReduceScatter 重排；映射覆盖/顺序/循环验证 |
| Collective completion and failure remain explicit | `test_completion_and_phase_dependencies_with_host_and_cache`；`test_dependency_wait_is_not_resource_queue`；`test_failure_and_cancellation_release_waiters` | 最终 join、真实 trace 前驱、host 计费、依赖与资源排队分离、失败/取消清理 |
| Phase measurements have hardware and backend applicability | `MeasurementApplicabilityTests` | 设备及别名、实际包 backend、显式 transport backend、异构顺序、回退/table-only、重复与冲突 |
| All-to-All validates inputs before fallback or no-op | `AllToAllContractTests.test_invalid_fields_both_modes_and_noops`；最小 flat 反例 | payload/对齐、参与者/group_size、fraction/imbalance/redistribution，包含零数据与单 rank |
| Flat All-to-All supports only an explicit uniform contract | `test_uniform_formula_support_rejection_and_cache`；`test_physical_statistics_still_conserve_outputs` | 默认公式、明确拒绝 unsupported 参数、no-op 不请求资源、physical 统计能力及流量守恒 |
| Communication caches isolate applicability from runtime state | `test_invocations_and_template_binding_have_new_events`；`test_scope_and_measurement_content_cache_equivalence`；缓存开关矩阵 | 模板重新绑定本次 producer、计划缓存命中仍有新事件/队列、测量内容与适用范围隔离 |

## 代码与用户工作边界

本次生产修改位于 `af/planning.py`（输入绑定）、`af/executor.py`（阶段就绪及 trace）、`af/operator_provider.py`（scope、验证及 flat 契约）、`af/estimate_cache.py`（缓存返回前校验）、`comm/traffic.py`（共享参数校验及静态 transport backend）。工作区原有 AFD 控制、校准、native protocol、routing replay 等内容仍保留，不属于本次修复。

独立通信快照从 `4f42e71` 导出，应用本次上述修改及新增测试；对旧接口增加查询 `transport_backend="unspecified"` 字段以表达本次测量契约。工作区中该字段原本由 AFD 改动提供，此一字段是独立修复所需的集成条件，并不引入其他 native 行为。不会变更 Git 暂存区、提交或自动归档前置变更。

粗 phase 持有通信域资源并集、域级独占 FIFO、旧接口没有绑定时的全输入 barrier 仍是声明过的模型/兼容边界。本次测试证明指定反例与相关语义，不能据此保证绝对硬件精度或其他代码无 bug。

## 验证命令

从仓库根目录执行：

```powershell
python -X utf8 -B -m unittest TokenSim.validation.test_communication_correctness TokenSim.validation.test_communication_domains
python -X utf8 -B TokenSim/validation/run_correctness_regression.py
openspec validate fix-af-communication-correctness --strict
```

完整工作区通过 551 项仓库测试与 46 项专项，共 597 项，记录于 `results/communication_correctness_workspace.log`。独立快照通过 432 项仓库测试，见 `results/communication_correctness_isolated.log`；新增统计守恒场景及模板身份断言后的最新 46 项专项另见 `results/communication_correctness_isolated_specialists.log`，独立验收共 478 项，不将重复运行重复累计。

最终计数及源码校验信息见 `communication_correctness_validation.json`，独立快照补丁见 `communication_correctness_isolated.patch`。关键修复方法经 AST 比较与完整工作区实现一致，仅在旧调用接口处适配新增可选上下文；snapshot 不包含无关 native/校准实现。测试临时目录使用 validation 内专用目录，Windows 沙箱的 tempfile ACL 限制需要在获准的测试环境执行。
