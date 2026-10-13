# AFD DBO重复导入修复与建模缺项，2026-10-07

按用户授权和OpenSpec align-vllm-afd-protocol-and-timing实施。只修复已确认的参考插件DBO性能问题；原安装副本和TokenSim估值公式未改变。metadata/host模型及正式校准profile仍待完成。

## 修复

在新副本`/home/youwei/autodl-tmp/af-workspace/src/afd-plugin-dbo-cache-20261007`中增加缓存可选NPU helper解析结果的函数。成功解析和ImportError不可用结果均缓存；每次仍调用动态enabled检查，保持NPU优先、native fallback、实际yield与异常语义。不支持进程中途安装/卸载可选后端，需重启。

唯一变化的Python文件为`afd_plugin/v1/worker/dbo.py`。脚本`patch_afd_dbo_cache.py`拒绝源代码漂移、重复补丁和非全新兄弟目录。`dbo-cache.patch`、manifest和修复前后哈希已保存在副本；原文件哈希复核不变。原默认插件路径仍指向原副本；使用修复需显式选择新PYTHONPATH，不修改全局环境。

CPU-only guard中位数：164.73 → 0.17 µs/次。修复后100次guard的CPU profile不再出现重复导入。

## 验证

5项补丁行为回归：缺少NPU时只解析一次且CUDA状态动态变化、NPU状态动态变化及fallback、双方disabled、非ImportError传播、漂移/重复补丁拒绝。与7项已有native协议回归合计12项通过。NPU路径通过stub测试，未在NPU硬件验证。

GPU2/3实际加载路径已在A进程断言；两端driver均使用隔离PYTHONPATH。各batch无profiler至少15秒，所有重复输出与修复前及GPU0/3基线一致。该A/B按不同时间顺序采集，不是随机交错控制实验，以下差异不能全部归因于本补丁。

| Batch | 修复前ms/步 | 修复后ms/步 | 观察变化 | 重复cohort数 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3.709 | 3.680 | -0.8% | 130 |
| 4 | 3.862 | 3.210 | -16.9% | 142 |
| 8 | 3.844 | 3.253 | -15.4% | 139 |

每点仅一组独立无profiler窗口，不能替代原提案要求的三窗口完整精度验收。修复消除了明确的软件开销，未使约0.19ms的模拟整步变为已校准结果。

## 计算建模仍需修正/验证

| 项目 | 已有证据 | 后续实现约束 |
| --- | --- | --- |
| Attention core与KV写入 | batch8约3.39µs预测/11.85µs实测组合 | KV写入bytes已经计入；按真实kernel范围校准固定/规模成本，避免重复加流量 |
| 层内布局拷贝 | tiny batch4/8每层3个copy约10–11µs，无独立项 | 先按backend/shape/stride确定是否真实搬运，不能对所有模型一律加3次 |
| norm/residual融合 | 两个解析启动项约6µs，原生融合约2.9µs | fused覆盖唯一，保留数学residual依赖，不能既收融合又收拆分费用 |
| Router/专家/路由布局/combine | 局部估值偏低，真实路由未回放 | 同scope整组测量或准确拆分，保留histogram，不把moe_sum等同完整weighted combine |
| 层外GPU辅助计算 | 当前约15µs vs 51–55µs | embedding、采样、索引、slot mapping和拷贝分别核对，不摊入每层attention |

## 通信及host建模仍需修正/验证

| 项目 | 当前缺口 | 后续实现约束 |
| --- | --- | --- |
| 完成语义 | send局部完成与目标数据可用不同 | 消费者依赖recv/数据可用，RTT不直接除二导入单向服务 |
| metadata控制协议 | 每步长度/内容两条消息及CPU/GPU回读尚未表达完整 | 与每层activation单列，F等待匹配控制与data，不将cProfile inclusive成本当网络alpha |
| 主机输入准备和提交 | 仅四条data各5µs未覆盖完整eager host | 每步/每调用/每消息唯一计费，阻塞等待与纯CPU服务分开 |
| 双端ready与队列 | readiness接口已有，当前无原生ready回放 | 通过host/stream事件产生等待；缓存仅存service，不能缓存本次ready/queue |
| 纯传输标定 | 当前只有send局部区间和数据＋回执诊断 | 先确认交付计时边界与metadata协议，再拟合并进行延迟接收/竞争测试 |

这些均纳入既有2.3/2.4/3.x/4.x任务及本轮设计约束，没有标为已修复。多节点分层拓扑和collective仍属于rebuild-hierarchical-af-communication，单机1A1F复测不能验收它们。

## 文件和复现

本地脚本：patch_afd_dbo_cache.py、prepare_dbo_fixed_rerun.py、audit_dbo_fixed_rerun.py、report_dbo_fix.py；回归tests/test_afd_dbo_patch.py。紧凑证据见同名JSON；原始trace和tokens保留服务器profiling/afd-dbo-fixed-20261007。

未提交、未归档整个OpenSpec变更；本轮只完成新增7.1/7.2，其他未完成任务保留。
