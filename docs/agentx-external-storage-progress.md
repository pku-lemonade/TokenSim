# 原生外部存储基础与当前接入缺口

本次先实现供 SimpleCPU/Mooncake 使用的组感知、按字节限容的 LRU 存储及共享方向传输队列。它们尚未接入引擎，因此本次不是 CPU 后端的完整交付，也没有将既有 HBM 一小时结果作为 CPU 验收。

`NativePageStore` 写入时预留并 pin，传输结束前不能查到；读传输同样 pin，LRU 不得驱逐在读写中的页。SimpleCPU 的分配单位是 GPU packed backing stride；Mooncake 对象按缓存组实际 payload 字节计容。`QueuedKVLink` 的调用方共享同一个方向队列，计入等待与服务时间；利用率可选 50%/75%/100%，75% 明确为假设。

源码核对发现：当前 SimpleCPU 的 `cpu_bytes_to_use` 是服务器整体预算并除以 world size，`cpu_bytes_to_use_per_rank` 显式覆盖每 rank。历史 B300 命令使用旧键，不能在新版本下按原数字直接照抄。CPU slot 数应由预算除以 packed stride 计算；参考 289,815 仅作核对，不能作为输入。

当前 Mooncake 源码的对象键包含 model、TP/PP/PCP/DCP、group 和可选部署 namespace，不含 DP rank。现有模拟键含 DP rank，后续原生接入必须移除这一隔离；SimpleCPU 则保留本地 worker 池边界。写回只处理已确认计算的数据，保存完成前必须保护 GPU 源块；加载期间 GPU 目标块也不能提前释放或复用。

5 个基础用例覆盖写入可见性、读 pin 与驱逐、packed stride/对象字节差异、全池 pin 后拒绝超额写入、共享队列竞争及带宽三档。全仓 369 项测试通过，78.949 秒；现有 Mooncake 兼容检查也通过。这些逻辑结果不代替 CPU 模拟。

下一步接入原生混合组 lookup/import、加载中的请求状态、DP 下异步完成、取消和 eager 写回，再按完整 393-trace 池及一小时窗口验证两种后端。CPU DRAM 硬件与运行预算将分别记录；从历史 `runners.yaml` 已读取 available-cpu-dram-mib，用于后续预算核对，不从 CPU block 数反推容量。
