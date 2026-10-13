## 1. 反例与实施边界

- [x] 1.1 将已知三项问题转为最小回归：两节点各四 rank 的晚输入、A-only 7 µs 测量用于不同 F 设备、flat 负 fraction/FP16 3-byte payload；以修复前失败及对应错误行为的记录验收。
- [x] 1.2 确认 `4f42e71` 基线和未提交 AFD 改动边界，建立本变更的场景到测试映射；以原六个规范 Requirement 均对应明确测试入口及受影响代码清单验收，不覆盖用户其他工作。

## 2. Collective 子组就绪与完成语义

- [x] 2.1 为正式 collective 建立 endpoint 到 producer 的静态输入绑定，并纳入 DAG 校验/模板身份；以参与顺序正确及缺失、重复 endpoint、歧义、循环引用在执行前失败验收。
- [x] 2.2 executor 为每次调用创建输入事件绑定，允许 physical 分层入口阶段仅等待自身输入；以 AllReduce 第一节点 t=0 启动、第二节点不早于 t=100 µs，AllGather 独立 slot 及时启动、ReduceScatter 输入重排仍等待本地输入验收。
- [x] 2.3 保留阶段前驱和逻辑 collective 全输出 join；以跨节点阶段不早于必要本地归约完成、输出重排结束前消费者不能启动，以及单阶段/flat/旧接口保持合法完整依赖验收。
- [x] 2.4 分离输入就绪、host/control 依赖与 FIFO 队列并更新 trace；以 ready=10/start=16/end=19 µs 的 queue=6/service=3 µs，host timing 开关下软件计费守恒及真实 trace 前驱验收。
- [x] 2.5 将等待 producer 的阶段纳入失败/取消清理；以晚 producer 失败、已运行子阶段失败、操作取消三类反例均无成功消费者、无资源/active-stage 泄漏且后续独立操作可执行验收。

## 3. 测量硬件与 backend 适用范围

- [x] 3.1 载入测量时保留实际设备、数据包 backend、来源/版本和适用组合，规范旧同质 scope 推导及 transport backend 未声明状态；以设备别名归一化、不同来源保留和相同来源重复去重的测试及元数据示例验收。
- [x] 3.2 在 phase 服务匹配中校验同质/异构 scope、包 backend 和显式 transport backend；以 A-only 7 µs 不用于不同 F 设备、同设备别名可匹配、backend 不符拒绝、异构需显式组合，以及解析回退/table-only/conflict 三类行为验收。
- [x] 3.3 将规范化适用上下文及测量内容纳入静态缓存身份和估值解释；以仅改变硬件/backend/scope/测量内容即不能命中不适用 exact，输出包含实际与记录来源信息验收。

## 4. All-to-All 校验与 flat 支持域

- [x] 4.1 抽出共享基础校验并确保在缓存/no-op/回退前生效；以两模式对无效 payload/dtype 对齐、participants/group_size、fraction 范围/互补、NaN/Inf/bool、imbalance 及 redistribution 类型的参数化测试验收。
- [x] 4.2 flat 明确拒绝显式 fractions、非默认 imbalance 和 redistribution，保留默认均匀 direct 公式并标注 per-rank wire bytes；以四 rank/400 bytes/50 GB/s/2 µs 得到 300 bytes 和 2.006 µs，以及超出支持域配置给出 physical 迁移提示验收。
- [x] 4.3 对合法零字节/单 rank 返回零网络服务且执行不申请网络资源；以默认配置 no-op 和非法参数仍报错、physical 合法 locality/热点路径保持原流量守恒验收。

## 5. 缓存、集成与交付

- [x] 5.1 覆盖连续调用与 cache on/off，包括 DAG 模板、静态计划和服务缓存；以相同流量不同 producer 时刻仍使用新事件/独立队列、硬件/backend/统计配置切换不绕过拒绝，估值/trace/完成结果一致验收。
- [x] 5.2 更新通信文档、测量元数据示例及 flat 迁移说明，逐项更新已知问题状态；以三个问题各有对应修复测试和新日志、FIFO/粗阶段近似仍披露验收。
- [x] 5.3 执行 `python -B -m unittest TokenSim.validation.test_communication_domains`、仓库 `python -B -m unittest discover -s tests -t .` 及新增反例；以通过记录、通信修复独立快照验证和完整工作区结果分列验收，不引用历史 459/575 项结果替代。
- [x] 5.4 执行 `openspec validate fix-af-communication-correctness --strict` 并核对 diff/场景覆盖；以严格校验通过、所有新增场景可定位验证证据、无无关 AFD 改动混入验收，不自动归档前置变更。

## 6. 固定 8 卡节点与统一 placement 归属（追加，待实施）

- [x] 6.1 将旧入口节点误合并转为失败回归，并确认本次工作区边界；以两个单 worker 网络生成 level-0=1、communication domain 错归 node0 的修复前记录，以及 slots 0/8 的合法 8 卡跨节点 300 MB=6 ms 对照验收，不覆盖其他 AFD 工作。
- [x] 6.2 为 AF-enabled physical 配置及独立构建入口增加显式 communication_domains、node_count 与固定 node_size=8 校验，统一 AF placement 分支进入 build_domain_placement；以缺配置、缺库存、非 8 卡在启动前失败，默认 node_size=8 可接受，以及旧 topology 不绕过校验验收。
- [x] 6.3 添加 TopologyPlacement.node_id(worker_id)，并让 provider 与 AFResourceRegistry 共用 placement 归属；以 slots 0/7/8/15 一致映射为 0/0/1/1，缺 worker 不回退为 worker_id，以及 local/inter 资源选择与排队正确验收。
- [x] 6.4 校验完整 worker 映射、非负整数卡位、重复占卡、库存和 level-0=8，并覆盖 dedicated/mixed/manual 与部分占用节点；以合法 A/F 空槽布局保持策略语义、非法 placement 在 runtime/provider/resource registry 三类入口均早失败验收。
- [x] 6.5 将原 node_size=4 的通信 fixture 迁移为 8 卡节点内四 rank 参与，并为依赖旧隐式放置的 AF physical 测试/配置补显式库存；以原迟到输入、阶段依赖、流量、scope/cache、失败清理及非 AF/flat 兼容断言仍通过验收。
- [x] 6.6 更新固定 8 卡迁移示例、已知问题状态和新验证记录，运行通信专项、受影响仓库回归与 openspec validate fix-af-communication-correctness --strict；以新反例转通过、placement/cache/trace 一致以及原 17 项与新增 6 项实施状态分别可查验收，不复用旧 597 项记录宣称新增修复已完成。
