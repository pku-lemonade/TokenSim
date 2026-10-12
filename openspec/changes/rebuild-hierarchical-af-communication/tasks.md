## 1. 基线与契约

- [x] 1.1 记录当前提交、配置及前置变更依赖，保存 flat/physical 的最小通信时间线和 trace 开关性能基线；以可复现实验 manifest 验收，不覆盖既有用户实验。
- [x] 1.2 定义每 rank 输入/输出、轮次、逐域 wire bytes、participants/fan 与模型版本契约，保留 ring_traffic_factor 辅助函数；以 AG/RS/AR N=1/2/4 手算、完整 buffer 转换及四 rank AG 节点总发送 1200 bytes 测试验收。
- [x] 1.3 校验整数 bytes、dtype/shard 整除、实际 participants 与 group_size，并明确协议冗余统计范围；以空/重复/缺失参与者失败及非零单 rank 输入仍无网络操作测试验收。

## 2. 纯服务公式修正

- [x] 2.1 统一 flat/physical collective payload/rounds，覆盖层内与 embedding/logits；以算法流量一致、聚合口径一致时估值一致、AllGather 不少算 N 倍及轮次不重复测试验收。
- [x] 2.2 为新基础解析通信使用 alpha+bytes/B，隔离 H/未校准 efficiency 的旧消费者并声明弃用范围；以仅改变 H 不影响新模型且 AF-disabled/未迁移消费者兼容测试验收。
- [x] 2.3 修正 oversubscription 只作用于数据项，防止显式瓶颈重复惩罚，统一 bit/byte 与方向单位；以 12→22 微秒手算、200/400 Gb/s 转换测试验收。
- [x] 2.4 保留 measured timing_scope/provenance，处理零字节、单 rank 与本地操作；以含 host 测量被明确拒绝、网络 no-op 零计费而必要本地工作仍存在的测试验收。
- [x] 2.5 在旧调度下运行公式回归并记录变化归因；以更新后的手算期望和独立公式差异报告验收，不将真实模型变化伪装为数值兼容。
- [x] 2.6 增加实测算法/拓扑/阶段范围匹配，禁止将整个 collective 时长随意拆分；以总时长记录在多阶段查询中按显式 fallback 解析或 measured-only 失败测试验收。

## 3. 拓扑与放置

- [x] 3.1 增加 node_size（默认 8）、每节点 local（默认聚合 300 GB/s）、独立 inter_tx/inter_rx 带宽及延迟的不可变配置；以独立参数加载、非法容量早失败和版本指纹测试验收，不加入并行链路分配字段。
- [x] 3.2 解析节点级 FIFO 域资源及方向单位，明确 300 GB/s 是节点聚合假设；以 400 Gb/s=50 GB/s、同节点所有 rank pair 共享 local、跨节点取源 tx/目标 rx 最小带宽测试验收。
- [x] 3.3 实现手工、dedicated/mixed 与手工覆盖的确定性放置，输出完整映射；以重复运行一致、尾节点空槽、占卡冲突和覆盖优先测试验收。
- [x] 3.4 接入 collective 相同节点参与数、shard 整除和 weights+KV+buffers 校验；以两节点各四 rank 合法、8/5 非均匀失败、singleton/P2P 不误拒绝、显存不足失败和 routing/ownership 不变测试验收。
- [x] 3.5 增加 8 卡节点 dedicated/mixed 样例及旧虚拟拓扑迁移说明；以样例解析并输出实际 placement、各域容量和建模假设验收。
- [x] 3.6 校验含节点/domain/方向的 canonical identity、别名冲突、缺失域及库存边界；以不同节点同名域独立、同节点不隐式占 inter 资源、策略覆盖冲突早失败测试验收。

## 4. 通信计划与粒度

- [x] 4.1 实现基础 ring 和单节点/单 rank 退化，输出各轮有向流量及域资源；以小规模数据 identity、每节点发送字节守恒及无网络操作不收费测试验收。
- [x] 4.2 实现层级 AllReduce 的 local RS、跨节点 shard AR、local AG；以阶段依赖、分片对应及结果等价 oracle 验收。
- [x] 4.3 实现分片跨节点 AllGather→节点内 AllGather，以及设计规定的 ReduceScatter，保留必要 layout 本地工作；以两节点各四 rank、每 rank 100 bytes 时跨节点总发送 800 bytes、每 rank 最终 800 bytes，以及 RS 数据重建/ownership/支持域测试验收。
- [x] 4.4 将 AF P2P/MoE legs 映射到节点 local 或源 tx/目标 rx，保留实际 ownership 和打包边界；以本地/远端、晚 ready、独立调用不合并、真实打包字节相加及不重复 EP 传输测试验收。
- [x] 4.5 实现 transmission 与 coarse subgroup phase 的域级纯服务估值；以 local alpha+M/B、跨节点 alpha+M/min(tx,rx)、逐节点聚合 wire bytes、串行相加及并行依赖 oracle 验收。
- [x] 4.6 生成零起点、输入已齐、无外部竞争的内部域时间表和去重资源并集；以同 local 两条各 5 微秒合计 10、不相交域并行、轮次依赖与启动成本不重复测试验收。

- [x] 4.7 实现统计分层 All-to-All 的源节点重排、远端交换和显式可选目标分发；以默认直接交付无第三阶段、可选路由有对应字节/依赖、同 rank 或已在正确中转位置不重复搬运测试验收。
- [x] 4.8 支持 local_fraction/remote_fraction、top_k payload 与热点分布的独立语义，记录 exact_routing/statistical_locality/uniform_fallback；以比例合法性、当前参与者均匀回退、偏本地/偏远端/固定 fraction 热点字节守恒及 flat/分层对照验收，保留已有精确 AF 路由。

## 5. 阶段与资源调度

- [x] 5.1 实现原子多资源预留和确定性等待仲裁，仅维护活跃等待索引；以无部分持有、相交请求不越过、不相交任务前进、有限前序不饿死的测试验收。
- [x] 5.2 接入逐阶段 ready/start/end 与域资源释放，physical 去掉重复全局 lane；以 t=14 串行示例、共享源 tx/目标 rx 排队、反向跨节点及不相交域并行测试验收。
- [x] 5.3 保留 host submit/transport/completion 调用边界，内部 phase 不重复 host 成本；以提交与网络重叠、完成等待、禁用分离计时兼容和边界 collective 集成测试验收。
- [x] 5.4 实现等待/运行中取消及失败清理、不可满足需求早失败；以多资源全部释放、后继不误推进且后续任务可继续测试验收。
- [x] 5.5 补齐 subgroup 输入到齐、同一时刻释放/就绪/仲裁顺序，并保留 per_worker/shared host scope；以晚到参与者不提前执行、不阻塞独立 subgroup、缓存/trace/容器顺序变化仍得到相同仲裁结果测试验收。

## 6. 缓存与 trace

- [x] 6.1 扩展既有 af/estimate_cache.py 和计划缓存，分别定义纯服务键与完整计划键，复用有界 LRU 和不可变配置指纹；以相同 payload 但不同目标节点/域映射/routing/locality/依赖不误命中，带宽/延迟/算法/数据版本变化建立新 runtime/provider/cache 验收。
- [x] 6.2 扩展 trace 的模型版本、phase/domain/resources、logical/wire bytes 和估值来源，分列 dependency/queue/service；以完整时间线重建 makespan、域流量和历史/延迟导出回归验收。

- [x] 6.3 缓存逐阶段服务和依赖，不用整体时长替代阶段申请/释放；以服务 3+8 微秒、第二阶段资源 t=10 空闲时完成 t=18，缓存开关结果一致验收。
- [x] 6.4 限定内部时间表为输入已齐且无外部竞争的零起点相对时间表；每次调用重新绑定 operation/phase 标识及事件，独立计算 ready/start/end 和排队；以不同队列状态下复用计划但完成时刻不同验收。
- [x] 6.5 检查 `routing_edges`/`tensor_slices` 等现有键排除项，规范化纳入会改变流量、资源和依赖的特征；以路由变化、缓存冷热和淘汰后时间线等价验收。

## 7. 专项验证与性能评估

- [x] 7.0 建立 design.md 第 9 节的 DSV3 测试 manifest 和 V1–V13 覆盖映射，固定模型/算子包、库存、dtype、assignments、seed 与输入长度；以紧凑 dedicated 基线、合法同库存 placement 配对通过 shape/显存校验，非法 mixed 布局明确拒绝验收。
- [x] 7.1 用测试专用带宽共享及细粒度域参考评估生产 FIFO 与粗 phase 资源并集持有；交付等长/异长/错峰、共享源/目标及独立域的完成时间、queue 与重叠差异报告，不新增生产共享模式或预设误差上界。
- [x] 7.2 评估去 H、固定延迟及域级服务的小消息敏感性，覆盖零字节、单 token、大小消息、多节点与单位；交付手算/可得实测对照及证据限制。
- [x] 7.3 固定硬件库存、模型、请求与路由比较合法 dedicated/mixed；交付 placement、显存、总/已用 GPU、节点/空槽和域容量、本地/远端 bytes、域 queue 与请求指标，不改变 experts 或隐式扩库存。
- [x] 7.4 对稀疏/稠密路由和不同规模运行性能基准；交付每层/step tasks、事件数、CPU/wall、峰值内存及 trace/export 成本，说明相对旧基线增加来自何种模型粒度，不截断任务。
- [x] 7.5 使用 DSV3 完整 61 层和固定输入/输出长度执行 V11 服务负载测试，各方案复用绝对到达率与请求序列；以 TTFT/TBT 口径、队列趋势、drain/未完成数及至少三 seed 报告验收，明确 decode-only 与完整 prefill→decode 的区别。
- [x] 7.6 汇总 V1–V13 的通过项、近似偏差与证据限制；以每项均有配置、命令、结果路径或明确阻塞原因验收，对影响 placement 排序的近似偏差提出设计复审，不虚构精度阈值。

- [x] 7.7 固定路由分别扫描节点内与节点间有效带宽；以两类参数独立、无 rank/链路数重复乘容量、无隐含集群总队列及 V6 报告验收。

## 8. 集成与交付

- [x] 8.1 运行完整回归，包括 AF-disabled、标准 LLMEngine V3 的受支持 prefill/decode 及 embedding/logits 边界、flat_link、physical 和 host timing 模式；以测试通过、所有入口采用一致网络契约及通信数值变更可归因验收。
- [x] 8.2 更新用户文档、配置迁移、公式/算法支持表与近似限制；以文档示例可运行、结果含模型版本且解释回退方式验收。
- [x] 8.3 核对全部规范场景与验证证据，执行 openspec validate rebuild-hierarchical-af-communication --strict；交付验证汇总，保留前置规范先同步的说明，不自动归档其他变更。
