# PKUa100：TokenSim 与 vLLM+AFD 计算对照

2026-10-06。本轮已实际运行 GPU 测试。对于下面的小模型、batch 与 decode 范围，A/F 计算总量均达到 max/min < 10 的数量级目标；原始解析模型低估实测约 1.43–2.03 倍。没有拟合或修改模拟器参数，也没有发布校准 profile。

## 实验范围与口径

- 模型为本地 `qwen3-moe-tiny-random`：2 MoE 层，hidden=256，4 experts/top-2，BF16，Q/KV heads=4/2，head_dim=64，vocab=128。vLLM 0.26.0、PyTorch 2.11.0，eager，TP=DP=EP=1。
- 输入128 tokens，输出32 tokens；prefill建立KV并用于预热，仅统计后续31个decode步。batch1/8作为基线，batch4作为未参与拟合的检查点。模拟器三点在采集结束前已生成，没有根据结果调整参数。
- A使用GPU0、F使用GPU3，均为A100 PCIe 80GB。0↔3为SYS跨NUMA路径。本轮仅验证计算，不把该路径当作原GPU2↔3的NVLink替代。
- 原生指标为逐decode步非NCCL GPU kernel时长之和，再取中位数。模拟器指标为每步A/F计算纯服务之和的均值。两者都排除host_submit和通信，但kernel融合、设备启动成本及辅助操作覆盖仍有近似差异；不能解释成完全相同算子的一一拟合。
- A包含attention、norm/residual、模型首尾及其他非NCCL辅助GPU计算。F包含router、grouped专家与combine。NCCL kernel含peer/host等待，单独记录，不能用作纯传输服务。
- 每个AFD点采集1个32步profile窗口（1 prefill+31 decode），另外至少15秒无profiler重复运行。每张卡的单卡原生vLLM对照，每点预热5轮，再独立采样3个profile窗口。未完成规范要求的每点3个独立无profiler验收窗口；本轮不是完整校准验收。

## 计算结果

单位：微秒/完整两层decode步，倍率为实测/模拟。

| Batch | 计算范围 | TokenSim | vLLM+AFD | 倍率 |
| --- | --- | ---: | ---: | ---: |
| 1 | A侧 | 72.67 | 113.54 | 1.56× |
| 1 | F侧完整MoE | 37.03 | 53.09 | 1.43× |
| 8 | A侧 | 73.47 | 148.90 | 2.03× |
| 8 | F侧完整MoE | 38.16 | 59.17 | 1.55× |
| 4 | A侧 | 73.01 | 145.22 | 1.99× |
| 4 | F侧完整MoE | 38.10 | 58.85 | 1.54× |

AFD单窗口内A计算的min/max分别为112.99–115.20、148.16–149.73、144.54–146.14微秒；F分别为52.74–53.31、58.82–59.39、58.65–59.07微秒（顺序batch1/8/4）。这些不是跨独立窗口置信区间。

## 无通信对照

单卡运行相同原生vLLM模型，不接入AFD connector。按CPU forward标注将MoE归F，其余GPU kernel归A。以下是3个窗口中位数的中位数。

| Batch | GPU0 A / F，µs | GPU3 A / F，µs |
| --- | ---: | ---: |
| 1 | 116.55 / 53.50 | 116.80 / 53.63 |
| 8 | 150.12 / 59.20 | 150.34 / 59.33 |
| 4 | 148.25 / 58.94 | 148.51 / 59.01 |

两张卡的单卡计算差异小于0.3%；AFD计算与这些单卡结果相差小于3%。在本次小模型与运行条件下，未观察到SYS通信等待引起显著的计算kernel膨胀。该证据不能保证其他模型、负载或频率状态也相同。

全部profile均恰好识别31个decode步，未归属kernel为0。修复了原汇总遗漏的每步KV slot mapping kernel（约3.65微秒），方法为kernel correlation关联`cuda_driver/cuLaunchKernelEx`。CPU/GPU同名annotation仅选CPU `user_annotation`，避免步数翻倍。AFD三个点分别完成117/113/117轮无profiler重复；各点输出完全一致，且与GPU0、GPU3各3轮单卡采样的输出完全一致。实际expert路由尚未回放到模拟器，输出一致不等同于路由回放已完成。

## 差异与下一步

1. A侧增长趋势还不够准确：batch1→8原生从113.54升到148.90微秒，模拟仅从72.67升到73.47。单卡attention forward约从53.8升至82.7微秒，说明增长主要位于attention范围。精确到kernel的算法选择、launch数与有效效率仍需进一步定位。模拟attention stage还包含部分norm/residual，不能直接拿53.8/48.6当作同scope校准因子。
2. 原生A侧有KV映射等辅助kernel；解析模型对融合、模型边界操作和采样实现的覆盖仍近似。已确认的slot mapping约3.65微秒只能解释部分差异，不能把剩余误差都归因于固定启动开销。
3. F侧低估约1.4–1.6倍；当前只能确认完整MoE范围。router、expert与combine在原生可能融合，尚无严格分开的实测系数；不据此发布三个独立阶段的校准值。
4. 按用户数量级目标，本轮无需加入统一乘数。若后续要求更精确，应先匹配实际kernel/融合范围、路由与batch分段，再采集并拟合，另留新的验证点。

原生无profiler整步中位数仍为约3.77/4.05/3.91毫秒（batch1/8/4）；它包含CPU引擎、通信及等待，远大于GPU计算。本轮计算结果不能证明整步、通信或等待模型已对齐，也不验证DSV3、大模型、prefill分离或分层网络。

## 证据与复现

紧凑证据归档：`afd-compute-alignment-20261006.json`（本目录），包含模拟输出、原生摘要、单卡对照、token校验及源码/脚本哈希。

远端目录：`/home/youwei/autodl-tmp/af-workspace/profiling/afd-compute-gpu03-20261006`。原始trace和tokens保留远端；本地只下载小摘要。原生driver、case、命令及进程记录均保留实验目录。测试结束已核查GPU0/3无计算进程，未停止其他用户作业。

模拟命令：`python -m scripts.run_tiny_afd_alignment --output outputs/afd-compute-baseline-20261006 --batches 1 8 4`。

原生准备：`profiling/prepare_compute_alignment.py`，拒绝覆盖现有目录。实验目录的`run_curve.py --concurrency B`每次运行一个batch；本轮依次1、8、4，设定`AF_WORKSPACE=/home/youwei/autodl-tmp/af-workspace`，使用`envs/afd026/bin/python`。新运行需新目录和空闲GPU检查。

单卡控制：`profiling/run_native_compute_control.py --gpu UUID --output NEW_DIR --model /home/youwei/autodl-tmp/af-workspace/models/qwen3-moe-tiny-random`，GPU UUID见证据manifest。

远端分析：`profiling/summarize_compute_alignment.py EXPERIMENT_DIR`；单卡子目录加`--control`。`profiling/check_compute_outputs.py EXPERIMENT_DIR`执行跨模式token校验。本地`python scripts/report_compute_alignment.py`从紧凑摘要重建证据归档，验证窗口数及未归属kernel数。
