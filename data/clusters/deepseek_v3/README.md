# V3 标准入口实验配置

`1a32f.json` 将 1 个 Attention worker、32 个 FFN/Expert workers 放入同一
逻辑网络，复用既有 V3 脚本的虚拟 NVLink3 拓扑。它是 33 卡同域的仿真假设，
不是对现有 DGX A100 物理布线的描述。需要真实跨节点实验时，请显式设置
`topology` 和各 worker group 的 `device_indices`。

所有 worker 使用 catalog 的 A100 SXM 80GB 设备。F 域采用 TP=32、EP=32
的 rank-reuse 布局：Dense/Shared FFN 使用 TP，Routed Expert 驻留分布在 EP ranks。
模型结构来源保留在 `data/models/deepseek-v3-671b-mla.yaml`。

`communication_timing` 开启三阶段通信。每次 host submit 5 微秒、completion
0 微秒是独立的 D 级实验假设，不是设备实测；fabric capacity=1 表示每个方向
一条独立服务 lane。collective 的软件 launch 不再重复计入 transport。

默认 `analytical_only` 保证无完整 MLA 实测表时可复现；输出不可称为硬件实测。
改为 `table_only` 时缺少 shape 必须报错；分段通信只接受声明
`timing_scopes.collective: transport_service` 的测量数据，旧数据不会自动改贴标签。

```powershell
python benchmark.py --cluster data/clusters/deepseek_v3/1a32f.json --model data/psla/deepseek-v3.json --qps 1 --request_count 1 --batching paged-attn --distribution burst --block_size 16 --verbose none --results_path outputs/af-core-runtime/standard-smoke
```

支持完整等长、未分块 prefill 与完整 decode；没有 checkpoint/MTP 执行能力。
