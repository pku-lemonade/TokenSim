# AF 核心运行时验证

本变更采用完整等长、未分块 prefill 与完整 decode。前缀缓存命中后的部分
prefill、分块 prefill、checkpoint 和 MTP 不属于支持范围，不能作为完整 serving 结果。

通信资源的 `capacity=N` 表示 N 条独立物理服务 lane，每条 lane 有配置中声明的
带宽。它不表示 N 个流共享一条链路的总带宽。半双工共用 `shared` lane；全双工
分别使用 `a_to_f` 和 `f_to_a`。相同资源组的 network、duplex 和各 lane capacity
必须一致，冲突在配置归一化时失败。collective 的别名引用同一资源注册表。

差分观察器 `tests.af_runtime_oracle.observe_services` 在 trace 之外记录每次
resource service 的 ready/start/end、queue 和完整 estimate，不插入模拟事件。
trace 开关测试同时禁止构造 `AFTraceEvent` 和调用 `RankLocalWork.as_dict()`，
避免只检查 makespan 而漏过无用序列化。

复现小规模基线：

```powershell
python -m scripts.profile_af_core_runtime --output outputs/af-core-runtime/baseline.json
```

加 `--full-model` 可记录 61 层。结果使用 analytical-only 数据，不能称为硬件实测。
墙钟包含 cProfile 和服务观察器开销；profile 分类使用 self time，避免累加嵌套函数
的累计时间。planning、lookup、SimPy、trace 之外的代码归入 other。


## DAG 模板与参考路径

`af.dag_template_cache=false` 禁用模板；`af.service_estimate_cache=false` 禁用服务估值缓存。
模板只属于当前 planner/runtime plan，最多保留 128 个结构。安全键包含 phase、层几何、
placement、完整 AF 配置（含 dtype/拓扑/collective）、token/request 数量、token 的相对排序，
以及 token 到 expert 的有序分配、权重和 A owner。计数或路由结构变化会编译另一模板。
同构层可共享模板，实际 layer/iteration/block/request/token identity 在每轮绑定；context
继续从本轮 block 进入 provider query。缓存不保存 SimPy event、request、timeout 或排队结果。

模板用不可变 recipe 保存静态字段、整数依赖索引、endpoint、ownership、shape 和资源标识。
绑定只重建包含动态 identity 的字段，并在同轮共享重复 routing/tensor 对象。构图和完整
DAG 校验在首次编译执行，测试额外把每个绑定结果送入原 DAG 校验并逐字段对比参考构图。
executor 复用 resource registry 的映射，独立申请每次服务的资源；运行中的完成事件使用
整数索引列表。此优化未合并独立 ExecutionNode，provider 内原有不可抢占 service 仍保留
全部 component breakdown。

`tests/test_af_templates.py` 覆盖原构图与绑定的逐字段差分、连续 decode、batch/context、
A/F router、TP/EP、副本切换、路由 skew、KV dtype、缓存淘汰及静态对象共享。
`tests/test_af_estimate_cache.py` 另覆盖来源信息不可变、context miss、strict fallback 失败
以及排队状态独立于估值缓存。

## 遗留测试迁移

三个模块迁入当前 ModelSpec、work_for_table、OperatorResolver/OperatorLookup、package
write/load 与核心 executor。旧 ShapeDiscovery、RecordingLookup、ModelOperatorManifest
和 shape-scoped CalibrationRule 不恢复；当前 package 的 table/default affine calibration
只校准 analytical fallback，实测值直接采用。按表校准不提供旧 shape 规则选择能力。
分组 routed_experts 当前公式包含激活 FLOPs 与三次 kernel launch；校准 offset 只加一次。
标准容量测试使用 ReplicaKVBlockManager，显存容量扣除现有 workspace 预留。

旧 UniformRequestRouting 的分块独立性已迁入核心 synthetic routing。用户已确认修复范围，
保留原分块一致性断言并修复实现，不通过 skip 或放宽该断言消除失败。

## 后续范围

稳定周期检测和多-token 跳跃仍未实现，应另建变更证明 scheduler、KV、routing 与到达事件
的状态等价性。本实现始终逐轮推进 SimPy，不使用单步延迟乘 token 数代替仿真。


## 2026-09-29 路由修复前的完整 61 层差分与 profiling（历史记录）

固定 batch=2、初始 context=8、2 decode steps、AF block=1、seed=42、analytical_only。
标准入口另跑完整等长 prefill 和 2 decode steps；AF-disabled 使用非 MLA 兼容基线。
以下为单次带 cProfile 和服务观察器的测量，不能推导稳定的生产墙钟加速比。

| 用例 | 参考墙钟秒 | 优化墙钟秒 | 参考 planning 秒 | 优化 planning 秒 | 服务数 | 通信字节 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| script_trace_True | 35.228 | 116.789 | 3.115 | 3.236 | 19404 | 145539072 |
| script_trace_False | 66.563 | 49.114 | 5.435 | 2.738 | 19404 | 145539072 |
| af_disabled | 0.024 | 0.017 | 0.000 | 0.000 | 0 | 0 |
| standard_v3 | 73.847 | 84.474 | 3.972 | 2.266 | 32012 | 181923840 |

所有用例的逐条服务记录（含 ready/start/end、queue、来源和完整估值）SHA-256 一致，
请求完成结果、TTFT/decode time 或各轮 makespan 也完全一致；仅计数性质的
operator/optimization stats 不参与结果相等断言。
固定 decode 的模板命中 242/244（99.18%），transfer 估值命中 10083/10152（99.32%），
compute 估值命中 9098 次。trace-off 的 planning 成本与总墙钟下降；trace-on 和标准入口
本次总墙钟增加，因此本记录仅证明结果等价和部分路径收益，不宣称所有模式全面提速。
原始数据及分项 lookup/SimPy/trace/other 时间见 outputs/af-core-runtime/final-comparison.json。

复现命令：

```powershell
python -m scripts.profile_af_core_runtime --full-model --no-cache --output outputs/af-core-runtime/reference-final.json
python -m scripts.profile_af_core_runtime --full-model --output outputs/af-core-runtime/optimized-final.json
python -m scripts.compare_af_core_profiles outputs/af-core-runtime/reference-final.json outputs/af-core-runtime/optimized-final.json --output outputs/af-core-runtime/final-comparison.json
```


## 最终回归状态（路由决策前）

全量命令 `python -m unittest discover -s tests -t .` 执行 410 项：409 项通过，1 项失败，
无 import error、无 skip。失败为
`tests.test_deepseek_full_decode.FullDecodeTest.test_routing_does_not_change_when_af_blocks_change`：
旧脚本的分块独立路由与当前核心按 block 生成 synthetic assignments 的行为不同。
该历史失败已按用户确认纳入核心修复；修复后的回归结果见下文。
回归日志：outputs/af-core-runtime/regression-final.log。

标准 benchmark 小规模请求已完成并导出 result_inf.json，notdone=[]，
结果位于 outputs/af-core-runtime/standard-final/。OpenSpec strict validation 与
`git diff --check` 通过。

公式迁移使用现有公共 work_for_table 的口径：MLA 包含 softmax FLOPs，routed_experts
包含激活 FLOPs；embedding/argmax 的 elementwise 表只统计张量流量，不额外加旧脚本
每 token 4 字节的整数索引流量。本次不改写这些公共公式，避免把行为变化混入性能差分。


## 核心 synthetic routing 修复

旧实现用当前 block 的总路由数构造 expert 序列，每个 block 从头分配，导致改动
AF block 大小也改变 workload。新实现按 seed、phase、layer、request ID、token ID
确定 expert，不依赖 block、iteration、batch 成员、调用顺序或共享随机数状态。
执行标识与 A owner 仍从当前 snapshot 填入，不缓存上轮的执行状态。

Uniform 使用循环均匀分配，同位置连续 request ID 可覆盖整个 expert 集合；
hot/skew/burst 使用每个 token 独立、可复现的采样。每个 token 的 top-k expert
不重复、权重和为 1，热点集合不足时选择剩余 expert。外部显式 assignments
及旧统计型 histogram API 保持现有口径；histogram 不是新的 token 级路由来源。

此修复会改变旧 synthetic 工作负载的 expert、通信与延迟结果，因此旧扫描数据
不能作为修复后的性能基线。新的优化差分必须在相同修复后路由下比较缓存开关。


## 修复后的全量回归

`python -m unittest discover -s tests -t .`：412/412 通过，无失败、无跳过，耗时 124.037 秒。
日志：outputs/af-core-runtime/regression-routing-fixed.log。
新增检查覆盖 uniform/hot/skew/burst、热点比例 0/0.75/1、分块/调用顺序/iteration
变化、seed/layer/token/phase 区分、top-k 无重复、权重归一和热点耗尽后的补齐。
保留原端到端 AF block=8 与 32 的 request-to-expert 一致性断言。
模板测试仍逐字段比较参考构图与绑定结果，并用完全相同的结构验证实际缓存复用；
不再错误要求变化了路由结构的不同 token 轮次共享同一模板。


## 修复后最终差分与性能

参考路径与最终模板绑定优化后的 61 层服务记录 SHA-256 全部一致，
script trace-on、trace-off、标准 V3 serving 和 AF-disabled 的模拟输出完全相同。
当前统计见 outputs/af-core-runtime/routing-fixed-comparison.json。

| 带 profiler/服务记录器的用例 | 参考墙钟秒 | 优化墙钟秒 | 服务数 | 通信字节 |
| --- | ---: | ---: | ---: | ---: |
| script_trace_True | 24.853 | 32.708 | 19608 | 145539072 |
| script_trace_False | 22.353 | 29.789 | 19608 | 145539072 |
| af_disabled | 0.011 | 0.010 | 0 | 0 |
| standard_v3 | 31.124 | 41.893 | 32114 | 181923840 |

上述带观察器的运行未体现总墙钟收益，不能宣称全面提速。模板绑定现在批量复制静态
字段，仅递归绑定动态字段，避免每次遍历完整静态 dataclass 字段。为区分正常执行与
观察开销，另测量关闭 trace、无 profiler、无服务记录器的运行（batch=2、context=8）：

| decode steps | 缓存 | 墙钟秒 | CPU 秒 | 模板命中/请求 |
| ---: | --- | ---: | ---: | ---: |
| 2 | False | 6.821 | 6.531 | 0/0 |
| 2 | True | 6.993 | 6.766 | 169/244 |
| 8 | False | 28.800 | 27.594 | 0/0 |
| 8 | True | 25.069 | 23.938 | 853/976 |

无观察器的各轮 makespan 也逐项完全一致。8-step 墙钟减少约 13%，2-step 略慢；
短任务的编译成本尚未摊平。以上为单次测量，应按实际长序列工作负载复测，
不视为所有模式、所有机器的稳定加速保证。

复现：

```powershell
python -m scripts.profile_af_core_runtime --full-model --no-cache --output outputs/af-core-runtime/reference-routing-fixed.json
python -m scripts.profile_af_core_runtime --full-model --output outputs/af-core-runtime/optimized-routing-fixed.json
python -m scripts.compare_af_core_profiles outputs/af-core-runtime/reference-routing-fixed.json outputs/af-core-runtime/optimized-routing-fixed.json --output outputs/af-core-runtime/routing-fixed-comparison.json
python -m scripts.benchmark_af_decode --steps 2 8 --output outputs/af-core-runtime/routing-fixed-uninstrumented.json
```


最终代码（含静态字段批量绑定优化）再次全量回归：412/412 通过，无失败、无跳过，
耗时 136.052 秒；日志 outputs/af-core-runtime/regression-complete.log。
OpenSpec 严格校验和 git diff --check 均通过，20/20 任务完成。


## 已有缓存命中路径优化（2026-09-29，任务 6.1–6.6）

本节对照的是**已开启 DAG template 和 service estimate 缓存**的上一版实现，
不是缓存关闭路径。上文 20/20 和 412 项回归是本轮追加任务前的历史记录。

### 实现与配置生命周期

- `TemplateCache` 按实际使用的 layer/link 生成静态指纹，完整比较后将静态组合
  intern 为局部整数，热路径不再递归扫描或反复哈希整份 AF 配置。
- `FingerprintMemo` 仅复用深层不可变对象的规范化结果，容量为 4096；普通可变
  query 字典每次重新读取。标量直接处理，并区分 bool/int/float，避免非法输入
  因 `True == 1 == 1.0` 错误命中已验证条目。
- 内置 provider 使用预编译字段布局/defaults 的 `QuerySkeleton`、逐次动态值和
  `PreparedQuery`。命中时跳过完整 `AFComputeQuery`/`AFTransferQuery` 构造；miss
  才实例化完整 query。shape/topology 映射在 executor 内有界复用，effective dtype
  映射按 link 复用。完整 query 接口保留，第三方/subclass/被包装的 provider 和
  缓存关闭路径继续接收完整 query。`query_constructions` 统计 executor 路径的构造次数。
- executor/planner 的 cluster 配置使用脱离原输入的只读快照。已编译 runtime plan
  本身由 frozen records/tuples 组成，直接共享，避免复制整张 ownership 图。
  配置字段、fallback、lookup policy 等不支持运行中替换；显式修改会报错并要求
  新建 runtime/provider。调用方修改原始配置不会影响已创建实例。
- 设备和 operator package 在 provider 首次使用时取得只读快照，不扫描所有设备或
  所有 shape。已有 provider 固定已使用的数据包，新 provider 可读取注册的新版本；
  table rows、source 和 calibration 的嵌套容器也与原始包隔离。数据包首次加载前
  使用当时的数据源，固定后的实例不提供热更新入口。
- 保留每次执行的完整 runtime-node 校验、动态输入校验、timing-scope 检查、资源申请
  和 timeout。没有跳过独立 ExecutionNode，也没有改动 SimPy 事件调度顺序。

没有对全部计算配置进行预扫描。命中路径仍需收集动态字段和规范化动态 key，
并非零分配；DAG 首次编译与 SimPy 调度也仍然存在。

### 性能测量

Windows / Python 3.13.3，完整 61 层 V3，batch=2，context=8 起逐轮增长，
AF block=1，固定 seed=42，analytical_only，trace off。每组独立创建 runtime，
重复三次，测量包含 runtime/provider 初始化和逐轮执行，不包含配置文件加载。
性能计时不启用 cProfile 或完整服务记录器；函数级 profiling 和正确性观察另跑。

| decode 步数 | 版本 | wall 中位数（秒） | wall 最小–最大（秒） | CPU 中位数（秒） |
| ---: | --- | ---: | ---: | ---: |
| 2 | 既有缓存基线 | 6.700 | 6.332–6.824 | 6.391 |
| 2 | 本轮优化 | 5.825 | 5.800–6.012 | 5.609 |
| 8 | 既有缓存基线 | 21.937 | 21.360–22.054 | 21.062 |
| 8 | 本轮优化 | 20.183 | 19.673–20.202 | 19.516 |

wall 中位数分别减少 **13.1%** 和 **8.0%**。这是当前机器和样本的结果，
不代表所有模型、batch、context 或 measured package 的统一收益。

冷段按总耗时减去后续迭代耗时计算，包含初始化、首次迭代与轮间开销；
热段是首次迭代之后的平均耗时，仍可能遇到新的 routing/context，并非强制 100% 命中：

| decode 步数 | 冷段中位数：基线 → 优化（秒） | 热段每轮平均：基线 → 优化（秒） |
| ---: | ---: | ---: |
| 2 | 4.028 → 3.569 | 2.636 → 2.331 |
| 8 | 4.030 → 3.602 | 2.543 → 2.343 |

8-step compute 完整 query 构造 **37,824 → 2,120**，transfer **40,608 → 214**。
compute 命中率 35,704/37,824、transfer 40,394/40,608，模板命中 853/976，
前后完全相同。2-step 构造次数分别为 9,456 → 530 和 10,152 → 160。

函数级 profiling 的 2-step 缓存包装器累计成本从 11.862 秒降至 5.611 秒；
SimPy `step` 调用次数均为 176,494。profile 本身有观测开销，不将其时间当作实际 wall 提速。
首次实现虽减少 query 构造，但重复复制不可变 plan 抵消了收益；该复制已删除。
中间测量 `optimized.json` 和 `final.json` 是修订过程记录，最终结果以
`baseline.json` / `final-shared-plan.json` / `summary.json` 为准，均位于
`outputs/af-cache-fastpath/`。

### 复现

```powershell
python -m scripts.benchmark_af_cache_path --steps 2 8 --repeats 3 --output outputs/af-cache-fastpath/recheck.json --profile
python -m scripts.benchmark_af_cache_path --verify --output outputs/af-cache-fastpath/service-differential.json
python -m unittest discover -s tests -t .
openspec validate integrate-deepseek-v3-core-runtime --strict
```

`--verify` 单独比较关闭缓存、缓存开启但使用完整 query、缓存开启并使用紧凑 query
三条路径，流式 SHA-256 覆盖全部服务的 iteration/block/layer/node、ready/start/end、
queue、完整 estimate（含通信 payload、来源、dataset、timing scope 和 component details）。
小模型专项测试另比较 trace enabled/disabled 的完整事件与服务记录，覆盖配置只读、
数据包替换、可变 query 输入、非法类型、动态维度与兼容 provider。


本轮完整 61 层差分已通过：三条路径各产生 19,608 条服务记录，SHA-256 均为
`c8301af217cb7b96fc0b1200aae5685cc92d03103dbd2e82b6ada4a933b07cad`，
逐步 makespan 也完全一致。结果见 `outputs/af-cache-fastpath/service-differential.json`。

当前性能样本使用 analytical_only；实测包首次使用时会复制已加载的相关包表行以隔离
外部修改，可能增加初始化时间和内存，其成本随该包大小变化。它不枚举未出现的 shape，
但实测包 workload 的收益仍需单独测量。


本轮最终全量回归：**423/423 通过**，无失败、无跳过，耗时 134.801 秒。
日志：`outputs/af-cache-fastpath/regression-final.log`。Windows 沙箱内首次运行遇到
临时文件权限错误；最终回归在已获自动批准的沙箱外执行，未跳过相关测试。
两个旧测试按新配置生命周期调整：通信模式变化创建新 provider，dtype 边界测试
在构造时注入 runtime-plan 测试视图，保留原行为断言。
OpenSpec 严格校验与 `git diff --check` 通过，追加后的任务为 **26/26 完成**。


## Trace 索引与延迟导出验证（2026-09-29）

本轮只改 trace 管理：保留完整 `trace.events`，用 iteration 到全局位置的索引
检索本轮事件；invariant validator 仍检查本轮全部事件。模型边界通过原位置
替换补齐 predecessor，索引会读取最新对象。资源 busy time 按事件顺序单遍
累计，每个事件的资源去重，保持 capacity、零 makespan 和排序口径。
Python 3.12+ 的原 `sum` 使用补偿求和，因此增量累计也保留补偿项，避免
资源利用率末位变化；较旧 Python 保持普通浮点累加。

`LLMEngine.af_iteration_summaries` 只存聚合 metadata；`get_af_stats` 在导出时
调用 recorder 的 `export_summary`，复制摘要并按索引序列化完整事件。
对外 `af_trace` 的字段、顺序、数值保持不变，trace-off 仍为 `None`。
重复导出不消费历史、不把完整事件副本存回 engine，已返回的 trace 与后续
记录、原位置替换及摘要修改相互独立。

### 调用方审计与测试

- 生产事件统一由 `AFTraceRecorder.record` 追加；`ModelAFExecutor._run_model_dag`
  的原位置替换保留 iteration identity，无须更新位置索引。
- `run_v3_concurrency.py`、`run_v3_poisson.py`、`run_v3_user_qps.py` 已有写出并
  清空批次的行为，现在调用 `trace.clear()` 同时清理索引；标准 engine 不新增
  历史清理。直接 `events.append/extend/clear` 不属于 recorder 写入接口。
- 摘要生产方是 `LLMEngine.record_af_iteration`，生产消费方是 `get_af_stats`；
  测试中直接读取摘要完整事件的调用改为 recorder 检索或公开导出。
- 新增 `tests/test_af_trace_storage.py` 的 9 个测试，覆盖交错 iteration、
  位置替换、真实模型边界补依赖、单遍聚合、重复资源、capacity、浮点精度、
  空记录、trace-off、clear 后重新记录、重复导出与独立结果，以及执行期禁止
  `AFTraceEvent.as_dict`。既有 invariant 测试改用 `record` 填充数据，保留
  对非法依赖、payload、资源竞争等失败行为的断言。

固定每轮 4 个合法事件，每轮各检索、校验、汇总一次，计数底层事件列表的
迭代和下标访问；不把 validator 内部其他工作算作检索访问。

| 轮数 | 保留事件 | 修改前访问数 | 修改后访问数 |
| --- | ---: | ---: | ---: |
| 8 | 32 | 432 | 96 |
| 16 | 64 | 1,632 | 192 |
| 32 | 128 | 6,336 | 384 |

新的访问次数为 `3 × 总事件数`，不会随历史轮数增加而反复扫描旧事件。
资源统计另用遍历计数断言只遍历本轮事件一次。这不代表整个 validator 或
simulator 的复杂度都是线性。

### 复现与测量口径

```powershell
python -m scripts.benchmark_af_trace --output outputs/af-trace-index/after
python -m unittest tests.test_af_trace_storage tests.test_af_trace tests.test_af_engine_integration tests.test_af_executor tests.test_af_operator_provider tests.test_af_model_executor
python -m unittest discover -s tests -t .
openspec validate integrate-deepseek-v3-core-runtime --strict
```

基准采用标准 `LLMEngine`、目录中的完整 61 层 V3、`1a32f.json`、单请求、
prefill 长度 4、decode_len=2、analytical_only、DAG/service 缓存开启。
每种 trace 模式计时 3 次，每次创建全新 engine；执行期包含配置加载、初始化
及请求运行，导出期包含 `get_af_stats` 和完整 `json.dumps`，不包含磁盘写入。
计时不挂 service observer/cProfile/tracemalloc，内存另开一次运行使用
tracemalloc，报告 Python 跟踪的字节而非整个进程 RSS。内存测量中的 wall/CPU
受 instrumentation 影响，不用于计算提速。共享机器的测量包含调度波动。

修改前基线和修改后结果分别保存在 `outputs/af-trace-index/before/` 与
`after/`；每份目录包含 `measurements.json`、`trace-True.json`、
`trace-False.json`。基线进程在修改代码前启动，使用已载入的原实现。

### 本轮结果

完整 JSON 在两种 trace 模式下分别与修改前逐字节一致；trace-on 有 32,114 个
事件，trace-off 为零事件。两种模式的完成状态、TTFT、完成时间和 decode 总时长
也一致。每次重复计时及独立内存测量的导出哈希一致：

- trace=True: `787bd5d86edd11f49993767fc8be5cddd2d5649415bfff38b2823269a44ac75d`
- trace=False: `d95b2ca5cfc77ff715b8a3ad88fbf0eafee7086c9edcafb2dd90c62ef9e5dd54`

下表为三次独立运行的中位数；括号为 wall 最小–最大值，单位秒。

| Trace | 阶段 | 修改前 wall | 修改后 wall | 修改前 CPU | 修改后 CPU |
| --- | --- | ---: | ---: | ---: | ---: |
| True | execution | 15.366 (15.152–15.996) | 11.243 (10.967–11.262) | 14.781 | 10.812 |
| True | export | 1.375 (1.259–1.380) | 3.835 (3.681–4.069) | 1.328 | 3.750 |
| False | execution | 9.119 (8.571–9.261) | 9.105 (8.962–9.166) | 8.812 | 8.844 |
| False | export | 0.102 (0.092–0.111) | 0.098 (0.096–0.116) | 0.109 | 0.094 |

trace-on 执行加导出总 wall 中位数 **16.741 → 15.097 秒**，约减少 9.8%。
执行期加速包含延后序列化的作用，不能把执行期减少的全部成本当成总提速。
trace-off 基本不变，其小幅差异不作提速结论。

| Trace | 测量位置 | 修改前 MiB | 修改后 MiB |
| --- | --- | ---: | ---: |
| True | 执行结束常驻 | 307.11 | 197.84 |
| True | 执行期峰值 | 307.31 | 200.88 |
| True | 导出结束常驻 | 396.83 | 285.21 |
| True | 导出期峰值 | 411.52 | 412.58 |
| False | 执行结束常驻 | 99.36 | 99.36 |
| False | 执行期峰值 | 100.50 | 100.50 |
| False | 导出结束常驻 | 105.55 | 105.55 |
| False | 导出期峰值 | 115.29 | 115.29 |

导出期峰值以该阶段开始时重置 tracemalloc peak 后测得，包含仍存活的 engine、
事件与缓存，不是单独 DTO 的大小；导出结束值包含 JSON 字符串。trace-on 的
完整导出峰值基本不变且略增，不能宣称此优化降低了整份 JSON 的峰值内存。
标准入口仍保留全部历史，事件列表及其索引均随事件数线性增长；本轮不提供
有界内存或流式导出。既有三个脚本的显式批次清理不改变这一标准入口约定。

逐字节比对、请求结果、统计汇总见 `outputs/af-trace-index/comparison.json`。


本轮全量回归 **432/432 通过**，无失败、无跳过；其中新增 trace 存储契约测试
9 项。日志为 `outputs/af-trace-index/regression-final.log`。由于既有 Windows
沙箱临时文件权限问题，完整回归经自动批准在沙箱外执行，没有跳过相关测试。
该次回归日志计时为 7669.365 秒，工具等待期间出现了长时间间隔；它不用于
性能比较，上表性能数据来自此前独立完成的基准运行。
OpenSpec 严格校验和 `git diff --check` 通过；本变更任务 **32/32 完成**。
