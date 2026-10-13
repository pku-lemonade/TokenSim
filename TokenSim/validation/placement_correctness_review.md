# 固定 8 卡 placement 补充验收

本记录对应 `fix-af-communication-correctness` 后追加的 6.1–6.6。原 17 项通信纠错的 597/478 项验证属于历史记录，本轮单独验证固定 8 卡节点与共享节点归属。完整工作区包含用户原有 AFD 改动，本轮完整回归不能称为新的独立通信提交快照。

## 修复前证据

`results/placement_correctness_before.json` 保存实施前提交和源码 SHA-256；`results/placement_correctness_before.log` 在旧实现上运行新增 10 个测试方法，结果为 32 个失败子测试、4 个错误。旧网络分组可能为两个单 worker 节点生成 level-0=1，而通信模块按默认 8 卡重新解释索引，将两者误合并到 node0。

## 场景到测试映射

以下方法位于 `test_placement_correctness.py` 的 `PhysicalPlacementTests`：

| 场景 | 测试方法 | 验收行为 |
| --- | --- | --- |
| 缺通信域的旧网络入口 | `test_legacy_network_grouping_requires_explicit_migration` | builder/provider/registry/runtime 均在执行前拒绝 |
| 缺库存、非 8 卡 | `test_missing_inventory_and_non_eight_size_fail_at_all_entrances` | 不猜库存，不按参与人数调整物理节点 |
| 两节点各一张卡 | `test_partial_nodes_keep_cross_node_service_cache_and_queue` | slot 0/8 属于 node0/1；300 MB 服务时间 6 ms；缓存开关结果一致，两次调用独立排队 |
| 边界 slot 0/7/8/15 | `test_shared_node_identity_covers_slot_boundaries` | topology/provider/registry 均为 0/0/1/1，同节点选 local 资源 |
| 非法 supplied placement | `test_invalid_supplied_placement_is_rejected_by_both_consumers` | 缺映射、重复、负值、越界、bool/float、额外 worker、错误节点边界与 None 均被拒绝 |
| 缺 worker 的节点查询 | `test_node_interface_never_defaults_missing_mapping` | 不回退为 worker_id |
| 部分占用及角色策略 | `test_mixed_dedicated_manual_keep_role_and_override_semantics` | fixed8 空槽保持；mixed/dedicated/manual 行为保留 |
| supplied placement 的策略与覆盖 | `test_supplied_placement_obeys_role_policies_and_explicit_overrides` | 角色槽位违规与显式卡位冲突被拒绝 |
| legacy topology 名 | `test_legacy_topology_name_does_not_override_declared_domains` | 使用显式通信域，describe 输出来源、策略和卡位 |
| 非 AF 与 flat | `test_non_af_and_flat_keep_legacy_topology_path` | 原拓扑入口继续可用，flat 的缺映射索引回退与 registry 归属一致 |

两个重复调用的资源时间线为 `(ready,start,end)=(0,0,0.006)` 与 `(0,0.006,0.012)`，单位为秒。缓存复用静态服务，第二次仍排队 6 ms，不复用完成时刻。

原有 `test_communication_correctness.py` 将两节点各四 rank 的 fixture 改为节点大小 8、卡位 0–3/8–11；迟到输入、阶段依赖、流量、scope/cache 和失败清理断言保留。依赖旧隐式放置的仓库 AF 测试与四份 `data/clusters/af` 示例补充显式库存。自定义提供器的 executor 测试显式传入基于 runtime placement 的资源管理器；rack-hop 测试声明四节点库存以容纳 slot24，保留原 local/inter 时延关系断言。

## 实施边界

生产修改为 `hardware/domains.py` 的 AF physical 库存边界、`hardware/placement.py` 的唯一构建与校验、`hardware/topology.py` 的 `node_id()`，以及 `af/operator_provider.py`、`af/resources.py`、`af/runtime.py` 的入口校验与共享归属。通用通信域数学配置仍允许其他 node_size；固定 8 卡约束在 AF physical 边界执行。空槽不创建虚构 rank。

本轮没有调整 FIFO、带宽、热点定义、逐 Expert 路由或 AFD 协议。P2P 静态计划缓存未包含 dtype 的独立问题仍待修复。节点级独占 FIFO 和粗 phase 持有资源并集仍是模型近似，测试通过不代表绝对硬件精度或通信代码无缺陷。未自动提交或归档。

## 验证命令与记录

从仓库根目录执行：

```powershell
python -X utf8 -B -m unittest TokenSim.validation.test_communication_domains TokenSim.validation.test_communication_correctness TokenSim.validation.test_placement_correctness
python -X utf8 -B TokenSim/validation/run_correctness_regression.py --label placement_apply
openspec validate fix-af-communication-correctness --strict
```

本轮最终计数、日志及源码指纹见 [placement_correctness_validation.json](placement_correctness_validation.json)。新增 10 项与原 46 项专项共 56 项；仓库计数以最终日志为准，不重复累计多次运行。早期中断或失败的完整回归不作为通过证据：`results/communication_correctness_placement_fixture_migration.log` 记录 551 项中 rack-hop 测试库存声明缺失的一个错误，该 fixture 随后显式声明四节点库存。测试临时目录使用 validation 内专用目录，在获准的 Windows 测试环境执行。

最终完整工作区回归通过 551 项仓库测试与 56 项通信专项，共 607 项，见 `results/communication_correctness_placement_apply.log`；严格 OpenSpec 校验通过，见 `results/placement_openspec_validate.log`。新增 10 个 placement 测试方法全部转为通过，原 46 项通信专项保留。相较实施前指纹，生产源码仅上述六个文件发生变化；原有 AFD 工作保留，没有纳入新的提交或独立快照声明。
