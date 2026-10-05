## 1. 基线与契约

- [ ] 1.1 记录当前提交、配置及前置变更依赖，保存 flat/physical 的最小通信时间线和 trace 开关性能基线；以可复现实验 manifest 验收，不覆盖既有用户实验。
- [ ] 1.2 定义每 rank 输入、输出、wire bytes、子组/轮次与模型版本契约，并梳理测量表口径；以 AG/RS/AR 的 N=1/2/4 手算 oracle 和完整 buffer 转换测试验收。
- [ ] 1.3 校验整数 bytes、dtype/shard 整除、实际 participants 与 group_size，并明确协议冗余统计范围；以空/重复/缺失参与者失败及非零单 rank 输入仍无网络操作测试验收。

## 2. 纯服务公式修正

- [ ] 2.1 统一 flat 与 physical 的 collective payload/rounds 逻辑，覆盖层内及 embedding/logits 边界；以同算法同参数估值一致、AllGather 不少算 N 倍和轮次不重复测试验收。
- [ ] 2.2 为新基础解析通信使用 alpha+bytes/B，隔离 H/未校准 efficiency 的旧消费者并声明弃用范围；以仅改变 H 不影响新模型且 AF-disabled/未迁移消费者兼容测试验收。
- [ ] 2.3 修正 oversubscription 只作用于数据项，防止显式瓶颈重复惩罚，统一 bit/byte 与方向单位；以 12→22 微秒手算、200/400 Gb/s 转换测试验收。
- [ ] 2.4 保留 measured timing_scope/provenance，处理零字节、单 rank 与本地操作；以含 host 测量被明确拒绝、网络 no-op 零计费而必要本地工作仍存在的测试验收。
- [ ] 2.5 在旧调度下运行公式回归并记录变化归因；以更新后的手算期望和独立公式差异报告验收，不将真实模型变化伪装为数值兼容。
- [ ] 2.6 增加实测算法/拓扑/阶段范围匹配，禁止将整个 collective 时长随意拆分；以总时长记录在多阶段查询中按显式 fallback 解析或 measured-only 失败测试验收。

## 3. 拓扑与放置

- [ ] 3.1 增加 node_size、节点内交换结构、GPU-to-NIC、NIC、可选上联的配置与不可变快照；以合法加载、缺失路径/非法容量早失败、版本指纹测试验收。
- [ ] 3.2 实现方向接口与共享组解析，加入指定 SXM/NIC 带宽预设；以 fanout 聚合、全/半双工、共享 PCIe 与独立 NVLink/NIC 资源映射测试验收。
- [ ] 3.3 实现手工、dedicated/mixed 与手工覆盖的确定性放置，输出完整映射；以重复运行一致、尾节点空槽、占卡冲突和覆盖优先测试验收。
- [ ] 3.4 将 group/shard 支持域、NIC mapping 和 weights+KV+buffers 显存预算接入启动校验；以非法布局/显存不足失败和逻辑 routing/ownership 不变测试验收。
- [ ] 3.5 增加真实 8 卡节点 dedicated/mixed 样例与旧虚拟拓扑迁移说明；以样例成功解析并输出真实节点/NIC 表验收。
- [ ] 3.6 校验全局资源标识、别名、有向路径及 NIC rail 可达性，自动放置限定硬件库存；以不同节点 nic0 独立、同节点不占 NIC、不可达 rail/别名冲突/策略冲突覆盖失败测试验收。

## 4. 通信计划与粒度

- [ ] 4.1 实现基础 ring 和单节点/单 rank 退化计划，输出每轮流量、路径及资源需求；以小规模 AG/RS/AR 数据 identity 和字节守恒测试验收。
- [ ] 4.2 实现层级 AllReduce 的 local RS、跨节点 shard AR、local AG；以阶段依赖、分片对应及结果等价 oracle 验收。
- [ ] 4.3 实现分片跨节点 AllGather→节点内 AllGather，以及设计规定的 ReduceScatter，保留必要 layout 本地工作；以两节点各四 rank、每 rank 100 bytes 时跨节点总发送 800 bytes、每 rank 最终 800 bytes，以及 RS 数据重建/ownership/支持域测试验收。
- [ ] 4.4 将 AF P2P/MoE legs 映射到实际路径并约束打包；以混合节点内外、晚 ready、独立调用不合并，真实打包字节相加及不重复 EP 传输测试验收。
- [ ] 4.5 为 transmission 和 coarse subgroup phase 建立匹配资源流量的纯服务估值；以整路径公式、串行阶段相加、并行依赖、10/90 NIC 不池化及轮次固定成本 oracle 验收。
- [ ] 4.6 为 coarse phase 生成可缓存的内部资源时间表与去重资源并集，不展开为全部全局事件；以两条各 5 微秒的同 NIC 传输合计 10 微秒、不同接口并行、不重复申请同资源测试验收。

## 5. 阶段与资源调度

- [ ] 5.1 实现原子多资源预留和确定性等待仲裁，仅维护活跃等待索引；以无部分持有、相交请求不越过、不相交任务前进、有限前序不饿死的测试验收。
- [ ] 5.2 接入算法阶段的 ready/start/end 与资源释放，physical 去除重复全局 lane；以设计中的 t=14 串行链、独立路径重叠、共享 NIC/上联竞争测试验收。
- [ ] 5.3 保留 host submit/transport/completion 调用边界，内部 phase 不重复 host 成本；以提交与网络重叠、完成等待、禁用分离计时兼容和边界 collective 集成测试验收。
- [ ] 5.4 实现等待/运行中取消及失败清理、不可满足需求早失败；以多资源全部释放、后继不误推进且后续任务可继续测试验收。
- [ ] 5.5 补齐 subgroup 输入到齐、同一时刻释放/就绪/仲裁顺序，并保留 per_worker/shared host scope；以晚到参与者不提前执行、不阻塞独立 subgroup、缓存/trace/容器顺序变化仍得到相同仲裁结果测试验收。

## 6. 缓存与 trace

- [ ] 6.1 扩展既有计划/service 缓存键与不可变配置指纹，排除运行态对象，保持按需构造；以配置/NIC/路由变化失效、缓存命中仍独立排队、开关缓存时间线等价测试验收。
- [ ] 6.2 扩展 trace 的模型版本、phase/path/resources、logical/wire bytes 与估值来源，分离 dependency/queue/service；以完整时间线重建 makespan、历史查询与延迟导出回归验收。

## 7. 专项验证与性能评估

- [ ] 7.0 建立 design.md 第 9 节的 DSV3 测试 manifest 和 V1–V13 覆盖映射，固定模型/算子包、库存、dtype、assignments、seed 与输入长度；以紧凑 dedicated 基线、合法同库存 placement 配对通过 shape/显存校验，非法 mixed 布局明确拒绝验收。
- [ ] 7.1 用小规模细粒度参考模拟评估整路径提前占用接收端及 coarse phase 延迟释放；交付无竞争/共享接收端/共享上联/多路径案例差异报告，不预先宣称误差上界。
- [ ] 7.2 评估去 H 与流水近似的小消息偏差，覆盖零字节、单 token、大小消息、多节点/多 NIC 和单位敏感性；交付参考/可得实测对照报告并明确证据限制。
- [ ] 7.3 在相同硬件库存、模型、请求和路由下比较合法 dedicated/mixed 放置；交付物化 placement、显存校验、总/已用 GPU、节点/NIC 数及空槽、节点内外 bytes、queue 与请求指标对比，避免改变 experts 选择或隐式增加 NIC。
- [ ] 7.4 对稀疏/稠密路由和不同规模运行性能基准；交付每层/step tasks、事件数、CPU/wall、峰值内存及 trace/export 成本，说明相对旧基线增加来自何种模型粒度，不截断任务。
- [ ] 7.5 使用 DSV3 完整 61 层和固定输入/输出长度执行 V11 服务负载测试，各方案复用绝对到达率与请求序列；以 TTFT/TBT 口径、队列趋势、drain/未完成数及至少三 seed 报告验收，明确 decode-only 与完整 prefill→decode 的区别。
- [ ] 7.6 汇总 V1–V13 的通过项、近似偏差与证据限制；以每项均有配置、命令、结果路径或明确阻塞原因验收，对影响 placement 排序的近似偏差提出设计复审，不虚构精度阈值。

## 8. 集成与交付

- [ ] 8.1 运行完整回归，包括 AF-disabled、标准 LLMEngine V3 的受支持 prefill/decode 及 embedding/logits 边界、flat_link、physical 和 host timing 模式；以测试通过、所有入口采用一致网络契约及通信数值变更可归因验收。
- [ ] 8.2 更新用户文档、配置迁移、公式/算法支持表与近似限制；以文档示例可运行、结果含模型版本且解释回退方式验收。
- [ ] 8.3 核对全部规范场景与验证证据，执行 openspec validate rebuild-hierarchical-af-communication --strict；交付验证汇总，保留前置规范先同步的说明，不自动归档其他变更。
