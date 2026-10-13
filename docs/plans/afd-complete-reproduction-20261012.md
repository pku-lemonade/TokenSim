# 完整固定成本校准复现

当前范围仅tiny Qwen3 MoE、H256、两层、BF16 eager、GPU2(A)/GPU3(F)、同节点单inflight decode。参数由batch1/8拟合；batch4仅回归；batch7为本轮冻结后的新留出。最终结果另见完成报告，本文不构成验收通过声明。

## 冻结与证据

- 参数及源码哈希：`afd-complete-frozen-bundle-20261012.json`。
- 源码快照：`outputs/afd-complete-calibration-20261012/frozen-source`；同级`frozen-source.zip`的哈希在bundle中。
- 参数统计、覆盖、计数：`afd-complete-parameter-inventory-20261012.json`。
- CPU API边界：`afd-api-boundary-audit-final-20261012.json`、`afd-complete-submission-fit-20261012.json`。原六窗口完整CPU调用区间替换旧framework-only值，不叠加两个host事件。新API诊断只审核来源/调用数/拷贝方向，不直接用于拟合。
- 25项host/operation独立batch6验证：`afd-complete-components-independent-20261012.json`。batch6在新提交规则形成前已被观测，明确属于独立回溯对照，不声称盲测。
- GPU纯服务参数：`afd-device-service-profile-20261012.json`。仅非NCCL kernel，排除输入准备slot mapping；4.580234514µs是解析模型的设备固定服务系数，不是CPU launch时延。
- 通信：`afd-transport-profile-20261012.json`与`afd-transport-independent-20261012.json`。双端kernel开始后的配对设备服务，非物理线缆时间。CPU提交另计，peer/FIFO等待由依赖生成。

## 模拟复现

从冻结源码目录执行，以下ROOT替换为仓库绝对路径；每次输出选未存在的新目录。batch可取1/8/4/7，路由文件必须使用相应batch。

```powershell
python -m scripts.run_tiny_afd_alignment --output ROOT/outputs/new-replay-b7 --batches 7 --native-call-order --pkua100-local-domains --device-service-profile ROOT/outputs/afd-complete-calibration-20261012/frozen-source/docs/plans/afd-device-service-profile-20261012.json --transport-profile ROOT/outputs/afd-complete-calibration-20261012/frozen-source/docs/plans/afd-transport-profile-20261012.json --host-profile ROOT/outputs/afd-complete-calibration-20261012/frozen-source/docs/plans/afd-complete-host-profile-20261012.json --operation-profile ROOT/outputs/afd-complete-calibration-20261012/frozen-source/docs/plans/afd-complete-operation-profile-20261012.json --routing-replay ROOT/outputs/afd-complete-calibration-20261012/replay-b7.json --calibration-environment ROOT/outputs/afd-live-capture-20261007/environment.json
```

未拟合对照保持相同协议、调用顺序、拓扑及路由assignments，去掉四类profile，并将路由runtime指纹绑定至对应未拟合runtime；不能给带profile的runtime伪造兼容指纹。此对照保留默认假设，不代表实测为零。

## 原生采集

远端使用`profiling/afd-complete-tools-20261012`下冻结工具；base为`/home/youwei/autodl-tmp/af-workspace`。先执行`collect_afd_frozen_validation --base BASE --output NEWDIR --bundle BUNDLE`，完成12个无探针窗口与独立batch7路由采集，再执行`collect_afd_device_service --base BASE --output ANOTHER_NEWDIR --heldout-bundle BUNDLE`完成batch7三个GPU分项窗口。每窗口启动前检查指定两张卡与实验端口，不占用其他卡，不停止外部进程。

## 判据与限制

分别审核同scope的host/operation、A/F GPU服务、双向设备transport及完整step。各项使用正值max/min<10的既定数量级目标；等待使用既有零竞争、peer延迟及FIFO注入oracle。不能将独立窗口分项相加构造原生整步，也不能用整步剩余差额调常数。

CPU/runtime与GPU服务采用调用组聚合，未精确重建逐kernel提交交错；CUPTI、同步测量扰动和原生运行波动保留。通过不代表低百分比误差，不外推至prefill、DSV3、跨节点或多inflight。冻结profile仍保存原始pending状态；最终验收结果在独立报告中，不改写历史profile以改变其哈希。
