# AF 本地实现收口与原生实测入口

最新进展：GPU2/3空闲后已完成batch1/8各三窗口真实计时、两组31步真实路由采集及回放、PyNccl/torch.distributed双向延迟接收诊断。整步仍差14.64–16.79倍，未生成校准profile。以下GPU占用描述为此前准备阶段的历史状态；当前证据与限制以 `afd-live-results-20261007.md` 为准。

本轮按剩余任务逐项推进，完成buffer预留、融合combine计费和原生采集/拟合/验收工具。OpenSpec当前16/25，剩余9项明确依赖原生采集和实测。此前“融合表一律拒绝”没有满足spec的融合combine场景，本轮已补齐，不以删改场景规避。

## 新完成的代码

1. `AFPhysicalRuntimeState.protocol_buffers` 记录单in-flight native control的逻辑协议buffer，按最大af_block_size预留一次并跨层复用；A保留residual/send/return，F保留input/output/router/top-k，双方保留控制staging。编译检查weights+buffer容量，KV块数扣除buffer。此项是协议预留，不声称涵盖所有CUDA workspace或碎片。
2. grouped expert按一个compute节点提交。实测表coverage可显式包括weighted_combine；此时只在expert计service，combine仍有语义完成事件但不重复计device/host费用。每iteration的覆盖标记结束或失败即释放。纯解析fallback保持独立combine。
3. `afd_capture_hooks.py` 在隔离插件读取A真实req_ids/token位置、F原生select_experts返回的weights/ids、逐层hidden shape/bytes和allocator快照。最多31个decode步；CPU codec区间标为可能含等待的诊断，不能冒充service。
4. `convert_afd_capture.py` 联合A/F记录，逐层逐token检查完整性、重复、实际行重排、top-k、hidden输入输出大小，生成严格routing_replay。缺记录即失败，不能按“通常的请求顺序”猜映射。
5. `fit_afd_host_profile.py` 只接收明确来源、无等待/无设备launch/无profiler、唯一覆盖的host样本；每项每batch至少三独立窗口。固定项或非负线性项由显式选定的模型拟合，窗口/batch等权；线性至少三个训练batch，batch4不得训练。保存分位数、残差、范围与待验证状态，不覆盖已有输出文件。
6. `run_afd_collection.py` 按训练/留出/诊断阶段顺序运行，batch4前必须冻结profile；检查源码/driver/hook漂移，有本实验独占执行锁。每case再检查GPU进程、显存、利用率和端口。这不构成服务器资源预留。
7. `measure_ready_peer_nccl.py` 区分PyNccl BF16 activation与torch.distributed NCCL int64长度/uint8 JSON，消息分别覆盖512/1024/4096/8192和8/179/181 bytes；提供0/5/20ms接收延迟。结果明确包含局部提交偏差/可能peer wait，不能自动导入transport profile。
8. `report_afd_acceptance.py` 仅在同batch三独立无profiler窗口、输出一致且数据有限正值时评估整步倍率；十倍及以上失败，缺窗口pending。即使step通过，也固定保留components和full alignment未验证，不能将整步对齐替代分项对齐。

## PKUa100 已准备内容

最新只读检查：GPU2 UUID `GPU-8df5e331-8f9a-3cdc-af30-42b94f908018` 被PID3337486占用，约37566MiB，GPU利用率49%；GPU3空闲。本轮没有启动远端GPU作业。

实验目录：`/home/youwei/autodl-tmp/af-workspace/profiling/afd-native-collection-20261007-v1`。

工具目录：`/home/youwei/autodl-tmp/af-workspace/profiling/afd-collection-tools-20261007-v1`。

已有12个准备case：batch1/8/4各三独立计时窗口、各一个独立诊断capture。安装中的插件未修改；隔离副本只加hook，源码哈希前后核对。远端prepared文件编译检查通过，train dry-run返回6个窗口。小型manifest和环境已回传为 `afd-prepared-collection-20261007.json`、`afd-prepared-environment-20261007.json`；版本为vLLM0.26.0、torch2.11.0，软件指纹包含隔离源码/hook/driver及包版本。

GPU可用后从远端执行（每次启动仍检查空闲）：

```bash
BASE=/home/youwei/autodl-tmp/af-workspace
PY=$BASE/envs/afd026/bin/python
TOOLS=$BASE/profiling/afd-collection-tools-20261007-v1
RUN=$BASE/profiling/afd-native-collection-20261007-v1
$PY $TOOLS/run_afd_collection.py $RUN/manifest.json --phase train
$PY $TOOLS/run_afd_collection.py $RUN/manifest.json --phase capture-train
```

依据无等待的可归因host样本，在本地运行 `python -m scripts.fit_afd_host_profile samples.json --profile-id <id> --method constant --output host-profile.json`。不能把capture中的codec inclusive区间直接改标签后喂给拟合工具。真实样本不可辨识时保持unknown；不得用整步减GPU总和填补。

真实profile冻结后才可执行 `--phase heldout --profile <远端profile路径>` 和 `--phase capture-heldout --profile <同一文件>`。随后用converter加载对应batch的capture及environment生成路由文件，再通过通用对照驱动加载profile/replay，最后出acceptance报告。详细字段见afd-host-profile-and-replay-20261007.md。

## 剩余9项及阻塞

| 任务 | 已有入口 | 缺少的证据/阻塞 |
| --- | --- | --- |
| 2.4 | bounded真实hook、行身份转换、通用replay | GPU2被占用，未拿到真实31步采集，不能验证采集路径和原生histogram一致 |
| 3.1 | 双端readiness、host/peer/fabric分离、控制消息因果oracle | 原生send/recv API返回、stream/预贴recv顺序尚无足够实际边界证据，当前保守按需接收近似不可冒充精确回放 |
| 3.2 | A/F独立准备、每次提交、completion资源，固定/规模项接口 | 与实际调用范围的host同步/异步边界仍需同窗口采集；不把整步差额塞进固定项 |
| 4.2 | 原生已有kernel归属报告，新增A/F身份及buffer诊断 | 同窗口CPU/GPU关联、跨进程时钟校验、hook/profiler扰动及实际allocator峰值未补测 |
| 4.3 | 两backend正确dtype/bytes及延迟接收微基准 | 目标双卡不可用；局部CUDA事件跨度即使测到也不能直接证明纯单向service |
| 4.4 | 非负拟合与污染样本拒绝工具 | 没有三个独立窗口的可归因纯host/设备服务样本 |
| 4.5 | profile通用engine加载与指纹/范围/冲突校验 | 尚无合格实测profile；禁止发布占位数值 |
| 5.1 | 三窗口准备、冻结profile、留出执行保护 | 上游实测参数未形成，无法运行正式留出验收 |
| 5.2 | 原始baseline与本地对照已有，step验收工具 | 校准后分项及整步数据缺失，不能声明数量级通过 |

上述不是另开一轮普通代码修改的清单；它们的共同前提是取得原生运行证据。取得证据后若揭示现有近似不成立，再按实际差异修正，不能在当前无数据时宣称“精确建模完成”。

## 验证

- 本轮buffer和工具纳入的全量回归487项通过（122.991秒）；随后融合combine、最终微基准类型和验收边界修改由完整AF相关273项复验通过（22.243秒）。
- 新工具7项测试覆盖行重排/缺项、真实返回保持、三窗口、污染样本、固定/线性拟合、留出冻结、源码漂移、正确wire dtype及NaN/十倍边界；不依赖GPU。
- 对照驱动将af_block_size固定为原生max_num_seqs=32；batch1/8/4导出的runtime指纹完全相同，避免容量预留随batch变化使同一profile失效。
- OpenSpec全部79个场景的证据/待测边界见 `afd-spec-evidence-20261007.md`。完整DSV3、分层通信重建和prefill-to-decode全流程未在本变更中验收。
