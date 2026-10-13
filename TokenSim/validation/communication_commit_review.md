# 通信纠错提交与归档验收

本提交以 `4f42e71` 为基线，仅包含 `fix-af-communication-correctness` 的原 17 项通信纠错与追加 6 项固定 8 卡放置修复。原工作区的 AFD 校准、native protocol、控制协议与 routing replay 修改保留为未提交工作。

提交内容先由已验证的 `communication_correctness_isolated.patch` 构建，再加入 fixed8 的库存校验、统一 `node_id()`、provider/资源管理与 runtime 边界，以及相关配置和 fixture 迁移。混合文件仅暂存本次通信修改，不将当前文件整体作为提交版本；两个通信测试、服务缓存和相关生产函数均在独立快照中验证。

独立提交快照通过 432 项仓库测试与 56 项通信专项，共 **488 项**，见 `results/communication_correctness_commit_snapshot.log`。之前完整工作区的 551+56=607 项记录仍保留于 `placement_correctness_validation.json`，它包含未提交 AFD 工作，不能当作独立提交的测试计数。原 17 项更早的 597/478 项验收同样是历史记录。

主规格 `openspec/specs/af-communication-correctness/spec.md` 已同步全部 8 条要求，5 份主规格严格校验全部通过。变更以用户时区日期归档到 `openspec/changes/archive/2026-10-12-fix-af-communication-correctness/`，23 项任务均完成；前置 `rebuild-hierarchical-af-communication` 未在本次归档。

提交源码指纹、独立快照计数和日志摘要见 [communication_commit_validation.json](communication_commit_validation.json)。P2P 静态计划缓存缺少 dtype 的独立问题仍未修复；域级独占 FIFO 与粗 phase 的资源占用近似继续保留。
