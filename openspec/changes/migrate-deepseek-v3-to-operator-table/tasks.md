## 1. ModelSpec 与模型目录

- [x] 1.1 在模型配置层加入可选 MLA 结构对象及字段校验，覆盖 `attention_type`、`q_lora_rank`、`kv_lora_rank`、`rope_head_dim`、`nope_head_dim`、`v_head_dim`，并验证正数、RoPE 偶数和 TP/head 约束；用 MLA shape 单元测试验证非法配置在加载前失败。
- [x] 1.2 将 `data/models/deepseek-v3-671b-mla.json` 转换为 `data/models/deepseek-v3-671b-mla.yaml`，嵌入完整 Dense/MoE、routing、dtype 和来源元数据；运行 `HardwareContext.load("data")` 并通过 model id/alias 解析验证 catalog 可发现 V3。
- [x] 1.3 更新参数统计、local shape、KV cache ledger 和 AF runtime plan，使 MLA 使用 latent cache width 与显式维度；运行 runtime-plan 测试验证 61 层、前三层 Dense、其余 MoE 和每个 A rank 的 cache shape。

## 2. Operator-table MLA 估算

- [x] 2.1 为 MLA GEMM、elementwise 和 `mla_attention` query 定义规范化 key 及 package/analytical shape 入口，确保 phase、execution mode、context lengths、latent/head dimensions 和 dtype roles 可追溯；用 key normalization 测试验证 decode 查询稳定。
- [x] 2.2 在 AF compute provider 中加入 MLA composition，按 decode absorbed 路径组合 Q/KV 投影、norm、RoPE、Q absorb、attention core、value expand、output projection 及必要 residual；用单层 fixture 验证 component 顺序和 service time 总和。
- [x] 2.3 将 MLA provider 接入 `OperatorResolver` 的 exact/interpolation/fallback policy，补充 source id、dataset version、match type、missing key 和 operator stats；用 exact、合法插值和缺失 shape 测试验证 provenance 与显式失败。
- [x] 2.4 让 boundary operator 与 TP collective 继续通过当前 resolver/拓扑 provider 定价，并验证 embedding、LM head、collective 不进入 MLA compute breakdown；运行 boundary trace 测试检查资源和依赖顺序。

## 3. DeepSeek-V3 运行入口迁移

- [x] 3.1 新增共享 V3 runtime factory，按 `HardwareContext -> ModelSpec -> compile_af_runtime_plan -> providers -> AFExecutor` 构造运行时，并移除 metadata namespace 与旧 `OperatorData*` provider 依赖；运行 import smoke 验证两个脚本可导入。
- [x] 3.2 迁移 `scripts/run_deepseek_mla.py` 到共享 factory，保留单层/单卡参数、AF topology 和 profiling 输出；运行 MLA smoke 验证完成并输出 runtime plan、trace 和 operator provenance。
- [x] 3.3 迁移 `scripts/run_deepseek_v3_full.py` 的完整 decode，覆盖 boundary、61 层 Dense/MoE、routing、shared/routed expert、combine/residual 和 token selection；运行小 batch 单 step 验证每层 trace 完整且 makespan 可重建。
- [x] 3.4 在入口中显式限制当前阶段为 decode/受限 MLA smoke，对 prefill checkpoint/MTP 请求给出清晰错误；运行不支持阶段测试验证不会返回伪装的完整 serving latency。

## 4. 回归与交付验证

- [x] 4.1 增加 ModelSpec、MLA shape、MoE layer selection 和 AF placement 的自动化测试，并验证既有非 MLA 模型和 `af.enabled=false` 回归通过。
- [x] 4.2 增加 operator-table 端到端 fixture，检查 compute/communication service、queue、overlap、source provenance 和 missing-data failure；运行目标测试集并保存结果摘要。
- [x] 4.3 运行 V3 小规模 profiling，确认输出包含每个 event 的 service/queue duration、source id、match type 和 operator breakdown，并生成 coverage/backlog 报告。
- [ ] 4.4 运行完整项目测试和 OpenSpec 校验，记录仍缺少的目标硬件 MLA 实测 shape；确认所有提案要求均已实现或由明确的 operator-data backlog 覆盖。
