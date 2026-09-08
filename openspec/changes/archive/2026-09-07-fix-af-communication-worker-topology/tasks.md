## 1. Execution Domain 配置与迁移

- [x] 1.1 在集群配置层加入 `PhysicalWorkerRef`、`ExecutionDomainConfig`、`LogicalInstanceConfig`、`DomainRankMapping` 和 domain-local parallel topology，并通过配置加载测试验证字段解析、稳定 identity 与序列化结果。
- [x] 1.2 将 P、Decode A、Decode F topology 改为逐 domain 校验，移除 AF 路径对全部 worker 总数执行统一 `TP * PP * DP` 校验的行为，并通过不同 P/A/F worker/rank 数的配置测试验证合法配置被接受。
- [x] 1.3 将 placement 校验收紧为每个 execution domain 内 logical rank identity 与 physical worker 严格一对一，拒绝 rank 共置、重复 physical reference、未映射 worker 和跨组引用，并通过失败用例验证错误包含 domain、instance、rank 与 worker。
- [x] 1.4 实现 `af.enabled=false` 与单设备 AF 旧配置的无歧义归一化，更新仓库内显式多 rank AF 示例，并运行 `tests/test_af_config.py` 验证兼容配置保持原行为且有歧义配置不会被猜测迁移。

## 2. Domain Placement 与通信计划

- [x] 2.1 更新 AF runtime 的 domain-local logical identity 到 physical compute resource 解析接口，移除 rank 共置支持，并验证每个合法 rank 精确解析到唯一 worker。
- [x] 2.2 更新 `AFExecutionPlanBuilder` 与 typed step 模型，按 phase、layer/model metadata、domain topology 和 expert placement 生成 A compute、AF transfer、Router compute、dispatch、Expert compute、combine、返回步骤，并通过计划测试验证 Router 在 dispatch 前、dispatch 在 Expert 前以及 participants 正确。
- [x] 2.3 将既有 TP/PP/EP group 语义适配为 domain communication plan，使 collective 只在 topology 要求时出现在指定位置，并通过有/无 collective 的完整计划对照验证次数和 operation 类型。
- [x] 2.4 实现稳定的 F logical target/expert placement 选择与 physical mapping，禁止绕过 placement 选择任意空闲 worker，并通过多 F worker 场景验证选择只发生在合法候选集合内。

## 3. Typed 通信查询与资源竞争

- [x] 3.1 扩展 `AFTransferQuery`，携带 operation、domains、logical participants、physical endpoints、network/resource identity、payload 和 topology details，并通过 provider 测试验证四类通信的完整规范化 key 与错误上下文。
- [x] 3.2 更新 operator-data transfer provider，使 `af_transfer`、`moe_dispatch`、`moe_combine` 和具体 collective 分别匹配模型并返回可追溯 service time；运行 `tests/test_operator_data_latency_provider.py` 验证单位、metadata、缺失记录报错且无旧后端 fallback。
- [x] 3.3 实现 `AFCommunicationResourceResolver`，让同一 physical fabric 上的 AF transfer、dispatch、combine、collective 共享规范化竞争域，同时保留 full/half duplex 语义，并通过资源映射测试验证相同/不同 resource group 的键与容量。
- [x] 3.4 为单个通信 step 实现稳定排序的多资源 acquire/release，确保 queue time 只由 SimPy 竞争产生，并用交叉 participants 的端到端场景验证无死锁、无超容量 service 重叠。
- [x] 3.5 将 `AFComputeQuery` 与正式/stub compute providers 从合并的 `stage="f"` 拆成 `stage="router"` 和 `stage="expert"`，实现互不重叠的 composition 与缺失 component 错误，并验证二者之和替代旧 F 纯计算总量且通信不进入 compute breakdown。

## 4. Plan-driven Executor 与 Trace

- [x] 4.1 更新 `AFExecutor._run_block()` 以逐层消费修订后的 execution plan，并通过单 block 端到端测试验证基础七段链 `A -> A-to-F -> Router -> dispatch -> Expert -> combine -> F-to-A` 的依赖和 9 ms oracle。
- [x] 4.2 接入计划中的 collective steps 和所有通信的共享资源申请，通过多 block 端到端测试验证 dispatch/combine/collective 的 service、queue 和竞争均进入 makespan。
- [x] 4.3 保持 A owner affinity、F placement 和 physical compute capacity 接入 Executor，删除 rank 复用 slot 的运行时验收路径，并验证一对一 placement 下正确资源申请与返回同一 A owner。
- [x] 4.4 扩展 `AFTraceEvent` 与汇总逻辑，分别记录 logical participants、physical endpoints/resources、operation、payload、provider metadata 和 queue/service 时间；通过 trace 重建测试验证 block completion、iteration makespan、barrier wait 与只计 service 的利用率。
- [x] 4.5 保持 iteration barrier、request 恰好推进一次和 KV Cache ownership 不变，并运行 AF/非 AF 对照端到端测试验证 F 侧不取得 request/KV 所有权且分配释放次数不变。
- [x] 4.6 将 Router/Expert provider metadata、composition breakdown 和 physical F resource 写入各自 trace 事件，并验证两段 service 可独立重建且没有重复算子。

## 5. Engine 集成与端到端验收

- [x] 5.1 在 `LLMEngine` 的 AF 构建路径注入归一化 domains、plan builder、resource resolver 与 providers，并从真实 YAML/config 启动一次 Decode-only PD+AF iteration 验证完整路径可运行。
- [x] 5.2 更新 `tests/test_af_executor.py` 的确定性多 block oracle，覆盖 Router/Expert 拆分、内部通信依赖、共享/独立资源、full/half duplex、F 容量和流水重叠，并逐事件手算断言最终 makespan。
- [x] 5.3 更新 operator-data 端到端 fixture，从 manifest 生成含 Router/Expert 独立 composition 的小型数据包后运行完整 AF iteration，验证 compute 与全部通信事件可追溯，缺失任一必要记录会明确失败。
- [x] 5.4 更新独立 P/A/F worker 数与 TP/PP/DP/EP mappings 的端到端配置矩阵，验证 A/F 两侧数量可不同、各 domain 内严格一对一、非法共置启动前失败，且不再出现统一 worker world-size 限制。
- [x] 5.5 在严格一对一 placement 完成后重新运行 AF-disabled、parallel、MoE expert parallelism、KV transfer 和 trace timestamp 回归测试，验证非 AF backend、既有 rank 行为、KV 与时间单位未发生回归。
- [x] 5.6 运行完整测试套件并执行 `openspec validate fix-af-communication-worker-topology --strict`，确认所有端到端验收通过且实现、主规格与 change artifacts 一致。
