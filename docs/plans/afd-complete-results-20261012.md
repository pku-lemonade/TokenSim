# 当前层级通信版本：固定成本校准完成报告

## 结论

OpenSpec剩余4.4、4.5、5.2已按证据完成，总计37/37。当前支持范围的host/operation、A/F GPU服务、双向设备transport及完整decode均通过既定max/min<10数量级目标；这不等同于高精度或生产模型泛化。

## 完整decode整步对照

单位ms。原生为三个独立无hook/无profiler窗口均值的等权平均；模拟为31步均值。未拟合和校准后使用同一batch真实路由assignments、相同协议/调用顺序/拓扑。

| batch | 原生均值 | 未拟合 | 校准后 | 校准偏差 | 原生窗口范围 |
|---|---:|---:|---:|---:|---:|
| 1 | 4.067 | 0.145 | 4.674 | +14.9% | 3.978–4.187 |
| 4 | 4.203 | 0.146 | 4.676 | +11.3% | 4.066–4.381 |
| 7（新留出） | 4.049 | 0.146 | 4.678 | +15.5% | 4.023–4.078 |
| 8 | 4.552 | 0.146 | 4.678 | +2.8% | 4.095–4.870 |

历史原始baseline（2026-10-06，协议及路由未对齐）另列：batch1模拟0.19360ms/原生3.70614ms，batch8模拟0.25552ms/原生3.85563ms；不能冒充本轮同配置结果。原始证据保留于`afd-1a1f-alignment-20261006.md`。

## 独立同scope分项

| 分项 | 原生/预测口径 | 最大误差倍率 |
|---|---|---:|
| 18项host＋7项operation | batch6各三独立窗口；新CPU提交口径为回溯独立对照，非新盲测 | 2.556 |
| 双向设备transport | batch6、两后端、0/5/20ms对端延迟，共36组 | 1.039 |
| A GPU计算 | 冻结后batch7三窗口，GPU kernel；原生135.506µs/模拟111.282µs | 1.218 |
| F GPU计算 | 冻结后batch7三窗口，GPU kernel；原生59.440µs/模拟56.789µs | 1.047 |

等待使用既有零竞争、peer延迟和FIFO竞争oracle；`tests/test_af_timeline.py`覆盖120/100µs启动边界对应100/80µs统一等待。等待不进入固定参数，重叠等待不重复求和。独立分项不可相加伪造真实step。

## 实际接入与验证

- 18项host与7项完整操作已列入`afd-complete-parameter-inventory-20261012.json`，包含数值/单位、覆盖、每步计数、原始来源和统计。GPU固定服务4.580234514µs来自batch1/8纯设备区间，不是CPU launch常数；完整CPU提交替换旧framework-only条目。
- 四类profile在通用executor实际加载。四个batch的31步均核对18项host、7项操作及每步6条消息；来源与计费值一致，不叠加旧默认提交成本。
- 12个无探针整步窗口、独立batch7路由采集与3个GPU留出窗口全部完成；每个batch三窗口输出hash一致，所有GPU计算kernel均归类。
- 376项AF回归、37项domain/placement回归通过；本轮新增报告字节哈希回归与冻结合同检查通过。OpenSpec strict validate通过。
- 774个冻结源码文件哈希保持不变，冻结源码重放复现参数与结果；原始profile与旧验收均未覆盖。远端采集结束，无计算进程或实验端口遗留，监控保持暂停。

## 残余偏差及审计说明

本轮整步均值偏高约2.8%–15.5%，A侧GPU服务低估约18%；最差host分项仍达2.556倍。历史原生窗口约2.9–3.1ms，本轮约4.0–4.6ms，因此不能把本轮较接近归因于模型高精度；调度/主机负载及测量扰动尚有不确定性。未用新留出或整步残差再次调参。

CPU提交与GPU计算仍是调用组聚合，未精确建模逐kernel提交交错；通信参数是配对设备服务而非物理线缆传输时间。当前通过只适用于tiny Qwen3 MoE、两层H256、BF16 eager、同节点GPU2/3、单inflight decode，不代表prefill、DSV3、跨节点或多inflight通过。

最终汇总脚本原先将JSON重新序列化后的LF字节与Windows原始CRLF文件hash比较，导致误报。修复只针对报告器：直接核对冻结文件原始字节hash，并核对解析内容等于bundle；冻结源码、参数、路由、原生结果均不改。新增测试同时验证CRLF有效、仅改换行也拒绝、内容改动拒绝。原始冻结报告器保留，复现最终汇总请使用工作区修复后的`scripts/report_afd_complete_acceptance.py`并传入`--source-root outputs/afd-complete-calibration-20261012/frozen-source`。

机器可读总验收：`afd-complete-acceptance-20261012.json`；整步：`afd-complete-step-validation-20261012.json`；复现：`afd-complete-reproduction-20261012.md`。
