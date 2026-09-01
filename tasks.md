# TokenSim 重构任务说明

本文档用于指导后续 AI/开发者重构 TokenSim。当前目标不是一次性重写系统，而是在保持现有 benchmark 行为可复现的前提下，逐步补齐：

1. 现有输入请求的 KV cache reuse pattern。
2. vLLM v1 风格的 KV transfer connector / scheduler 逻辑，并删除当前旧 swap 逻辑。
3. 支持大模型模拟所需的 TP/PP/DP 基础并行语义。
4. 支持 MoE 模型和 EP。
5. Mooncake / Mooncake Store 模拟器，其中 Mooncake Store 作为 SSD offloading 后段。

参考代码位于 `ref/`：

- `ref/vllm/`：参考 vLLM 的 prefix caching、KV connector、disaggregated prefill、parallel config、MoE/EP、KV offload、plugin/registry 设计。
- `ref/Mooncake/`：参考 Mooncake 的 KVCache-centric 架构、Transfer Engine、Mooncake Store、SSD/offload 和 storage benchmark。用户提到的 `ref/mooncake` 在当前仓库中对应路径为 `ref/Mooncake/`。

## 总体原则

- 不要直接把新逻辑继续堆进 `LLMEngine.run`、`LLMWorker.run`、`LLMPagedAttnScheduler.schedule`。
- 每个方向都先定义接口、配置和指标，再迁移现有实现。
- 保持 `./scripts/benchmark.sh` 的默认行为不变，除非当前任务明确要求改变。
- `ref/vllm/` 和 `ref/Mooncake/` 是参考代码，不作为 TokenSim 运行依赖，不要修改。
- 运行产物不要提交到 `results/`、`img/`、`log/`、`tmp/`。
- 新命名以 serving 语义为准：`prefill_len` / `decode_len` / `request_count` / `--model`。长度分布参数同步使用 `prefill_mean_len` / `prefill_range_len` / `decode_mean_len` / `decode_range_len` / `decode_len_distribution`。旧命名 `prompt_len` / `generation_len` / `prompt_count` / `prompt_lens_mean` / `generation_lens_mean` / `--psla` 等只作为过渡兼容别名。
- reuse 第一版只针对输入 prefix；输出默认不写入 prefix cache，除非输入数据集显式提供可索引的后续 block/hash 信息。
- 不引入 `Sequence` 层；`Request` 仍是调度单位，request 与 KV cache block 的映射由类似 vLLM v1 的 block table / KV cache manager 维护。
- 内置 workload loader 不适配 Mooncake trace；后续如需使用 Mooncake trace，通过外部 adapter 转换成 TokenSim 统一 workload record。
- trace 到达时间规则：数据有 `timestamp` 时优先按 timestamp 重放；没有 timestamp 时继续按 `--qps` / `--distribution` 生成到达。
- 插件/connector 使用内部 registry 即可，接口命名尽量贴近 vLLM。
- SSD offloading 和 Mooncake 在当前任务中视为同一方向：使用 Mooncake / Mooncake Store 语义模拟 SSD offloading，不新增独立 SSD offload backend。后续如确实需要 generic `OffloadConnector`，再单独规划。
- Mooncake 相关命名与 vLLM 保持一致：`MooncakeConnector`、`MooncakeStoreConnector`、`MultiConnector`、`kv_connector_extra_config` 等。
- Mooncake 方向只做纯模拟器，不直接集成真实 Mooncake API；模拟器集成在 TokenSim 框架内，不另开项目。
- 当前旧 swap 逻辑是待删除的 legacy 路径。迁移到 vLLM-like KV transfer 后，不要求继续兼容 `Task.SWAP_*`、`SwapMetadata`、`swap_in/out` 语义。
- 大模型并行和 Mooncake 的实现顺序为：先 TP/PP/DP 基础，再 MoE/EP，再 Mooncake Store 完整模拟。
- 每完成一个阶段，都记录：
  - 改了哪些文件；
  - 新增/修改了哪些 CLI 参数和 JSON 配置；
  - 跑了哪些验证命令；
  - benchmark 指标是否变化，变化是否符合预期。

## 参考代码索引

### vLLM 参考点

- Plugin system：
  - `ref/vllm/docs/design/plugin_system.md`
  - 重点：通过 registry/entry point 加载外部组件；组件要可重复加载；平台、worker、backend、logger 等都通过接口扩展。

- Prefix caching：
  - `ref/vllm/docs/design/prefix_caching.md`
  - `ref/vllm/vllm/v1/core/kv_cache_manager.py`
  - `ref/vllm/vllm/v1/core/block_pool.py`
  - 重点：full block 才缓存；block hash 由 parent hash、block tokens、extra hash 组成；block 有 `block_id`、`block_hash`、`ref_cnt`；维护 block pool、free queue、hash->block 映射、request->block 映射；命中缓存时 touch 防止被驱逐；释放时按 LRU/free queue 驱逐。

- Disaggregated prefill / KV transfer：
  - `ref/vllm/docs/features/disagg_prefill.md`
  - `ref/vllm/vllm/config/kv_transfer.py`
  - `ref/vllm/vllm/distributed/kv_transfer/kv_connector/factory.py`
  - `ref/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`
  - `ref/vllm/vllm/v1/core/sched/scheduler.py`
  - `ref/vllm/vllm/v1/worker/gpu/kv_connector.py`
  - 重点：`KVTransferConfig` 包含 `kv_connector`、`kv_role`、`kv_buffer_device`、`kv_connector_extra_config`、`kv_connector_module_path`；connector 分 scheduler-side 和 worker-side；scheduler 负责匹配、分配后更新状态、构造 metadata；worker 负责加载/保存 KV、等待 layer load/save、返回 worker metadata。

- KV offload / tiering：
  - `ref/vllm/vllm/v1/kv_offload/base.py`
  - `ref/vllm/vllm/v1/kv_offload/tiering/base.py`
  - `ref/vllm/vllm/v1/kv_offload/tiering/spec.py`
  - `ref/vllm/vllm/v1/kv_offload/tiering/fs/manager.py`
  - 重点：这些文件只作为 future generic `OffloadConnector` / tiering 接口参考；当前 SSD offloading 任务通过 Mooncake Store 模拟，不单独实现 generic SSD backend。

- Mooncake connector：
  - `ref/vllm/docs/features/mooncake_connector_usage.md`
  - `ref/vllm/docs/features/mooncake_store_connector_usage.md`
  - `ref/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/`
  - `ref/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py`
  - `ref/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py`
  - `ref/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`
  - `ref/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py`
  - 重点：`MooncakeConnector` 是 P/D 之间的 P2P KV transfer；`MooncakeStoreConnector` 是共享 KV pool，可支持 CPU/disk offload、跨实例 prefix caching、single-node 和 multi-node；`MultiConnector` 可组合 P2P transfer 与 distributed store。

- TP/PP/DP 与 MoE/EP：
  - `ref/vllm/vllm/config/parallel.py`
  - `ref/vllm/vllm/v1/metrics/perf.py`
  - `ref/vllm/vllm/model_executor/layers/fused_moe/`
  - 重点：`ParallelConfig` 表达 `pipeline_parallel_size`、`tensor_parallel_size`、`data_parallel_size`、`enable_expert_parallel`、`expert_placement_strategy`、`all2all_backend`、EPLB 等；MoE 指标解析关注 `num_experts`、`num_experts_per_tok`、`moe_intermediate_size`、`num_shared_experts`、`num_moe_layers`。

### Mooncake 参考点

- 项目总览：
  - `ref/Mooncake/README.md`
  - 重点：Mooncake 是 KVCache-centric disaggregated architecture，分离 prefill/decode cluster，并利用 CPU/DRAM/SSD 构建 disaggregated KVCache pool。

- Transfer Engine：
  - `ref/Mooncake/docs/source/design/transfer-engine/index.md`
  - 重点：核心抽象是 `Segment` 和 `BatchTransfer`；支持 DRAM/VRAM/NVMeof；支持 TCP/RDMA/NVLink/NVMe-oF 等 transport；强调拓扑感知、多 NIC 聚合、异步 batch transfer。

- Mooncake Store：
  - `ref/Mooncake/docs/source/design/mooncake-store.md`
  - 重点：Master 管 metadata/空间分配，Client 同时可以是请求方和存储段提供方；支持 embedded、dummy-real client、standalone store；支持多副本、Put/Get/Remove、异步 copy/move task、SSD persistence/offload。

- SSD/storage benchmark：
  - `ref/Mooncake/benchmarks/storage_benchmark/storage_benchmark.py`
  - 重点：用单大文件和 offset allocator 模拟 SSD KV block 存储；`hash_id -> offset` 内存映射；block hit 读盘，miss 写盘；统计 read/write blocks、prefix hit blocks、latency、block hit rate、write ratio。

## 阶段 0：建立 baseline

目的：重构前先确认当前仓库能跑，后续行为变化才有比较对象。

任务：

1. 确认运行环境。
   - 使用 conda 环境 `tokensim11`。
   - 不要在任务过程中随意切换 Python 版本。

2. 跑基础验证。
   - `python -m compileall benchmark.py TokenSim util`
   - `./scripts/benchmark.sh`

3. 记录当前脆弱点。
   - `g_time` 是全局对象，多次运行可能串统计。
   - `benchmark.py` 默认 `--psla`、`--cluster` 路径是旧路径；Phase 1 后主入口应使用 `--model`，`--psla` 只作为兼容别名。
   - `max_parallem_sum` 拼写已经进入 CLI/API，不能直接改名破坏兼容。
   - `BlockAllocator` 当前只模拟数量，`PhysicalTokenBlock.block_id` 固定为 0。
   - `get_lens_from_file()` 固定跳过前 6789 条，小数据集会变空。

验收标准：

- 基础验证通过；若失败，记录完整错误和原因判断。
- 后续任务的 first commit 不包含无关大范围格式化。

## 阶段 1：代码基础改造

目的：先完成支撑 input-only reuse 的基础代码改造，在不改变默认 benchmark 行为的前提下，把 request 元数据、block table、dataset loader 和指标出口整理到清晰边界。

参考：

- `ref/vllm/benchmarks/multi_turn/`
- `ref/vllm/benchmarks/benchmark_prefix_caching.py`
- `ref/Mooncake/FAST25-release/traces/`
- `ref/Mooncake/benchmarks/storage_benchmark/storage_benchmark.py`

目标文件：

- `TokenSim/llm/llm_request.py`
- `TokenSim/utils.py`
- `util/request.py`
- 可新增 `TokenSim/workload/`
- trace 输入使用 `qwen-bailian-usagetraces-anon/` 子模块，不在主仓库新增 dataset fixture

任务：

1. 修正和隔离现有脆弱点。
   - 把 `get_lens_from_file()` 中“跳过前 6789 条”改成显式参数，例如 `--dataset_skip`，默认 0。
   - Qwen Bailian JSONL trace 必须能跑小规模 benchmark。
   - 保留旧 CLI 兼容；`max_parallem_sum` 拼写暂不破坏。
   - 避免 `g_time` 在多次进程内运行时串统计，可先封装 reset/recorder，不要求一步到位替换全部统计逻辑。

2. 统一现代化命名。
   - 数据结构主字段改为 `prefill_len` 和 `decode_len`，不再新增使用 `prompt_len` / `generation_len`。
   - CLI 主参数改为 `--request_count`，`--prompt_count` 作为兼容 alias，解析后统一写入 `args.request_count`。
   - synthetic 长度分布参数同步改名：
     - `--prefill_mean_len` 兼容旧 `--prompt_lens_mean`。
     - `--prefill_range_len` 兼容旧 `--prompt_lens_range`。
     - `--decode_mean_len` 兼容旧 `--generation_lens_mean`。
     - `--decode_range_len` 兼容旧 `--generation_lens_range`。
     - `--decode_len_distribution` 兼容旧 `--generation_lens_distribution`。
   - PSLA/model JSON 字段同步改名：`prefill_mean_len`、`prefill_range_len`、`decode_mean_len`、`decode_range_len`、`decode_len_distribution`；旧 JSON 字段读取时保留兼容。
   - CLI 主参数改为 `--model`，用于指定当前 `data/psla/*.json` 这类 model/SLO 配置文件；`--psla` 作为兼容 alias，解析后统一写入 `args.model_config_path` 或等价字段。
   - 结果 JSON 和内部变量优先使用 `request_count`、`prefill_lens`、`decode_lens`；旧字段如果需要保留，应明确标记为兼容输出。
   - 文档和示例脚本同步切换到新参数。

3. 定义 workload record。
   - 必须包含：`request_id`、`prefill_len`、`decode_len`。
   - 可选包含：`arrival_time` 或 `inter_arrival_time`、`chat_id`、`parent_chat_id`、`turn`。
   - 为 input reuse 预留：`hash_ids`、`cache_salt`、`reuse_group`。
   - 不新增 `Sequence`；这些字段最终映射到 `Request` 的输入元数据。

4. 拆分 request 生成逻辑。
   - `Request` 只负责运行时状态、timing 和输入 reuse 元数据。
   - synthetic 分布采样、JSON dataset、JSONL trace 读取放到 workload loader。
   - scheduler 不应该直接解析外部 trace 字段。

5. 增加 workload 类型。
   - `synthetic`：保持当前按均值/范围随机生成。
   - `json_pairs`：兼容当前 `[prefill_len, decode_len]` 二元组格式；旧文档里的 `[prompt_len, generation_len]` 只作为历史解释。
   - `qwen_jsonl`：支持 `timestamp`、`input_length`、`output_length`、`chat_id`、`parent_chat_id`、`turn`、`hash_ids`。
   - Mooncake trace 不做内置 workload 类型；需要时由外部 adapter 转换为 `json_pairs` 或 Qwen-like JSONL。
   - 到达时间：若 record 中有 `timestamp`，按 timestamp 重放；否则按 `--qps`/`--distribution` 生成 inter-arrival。

6. 新增基础 block table 抽象，但不启用 reuse 行为。
   - 建立 `request_id -> block_ids` 的映射层，参考 vLLM v1 `BlockTables` / KV cache manager 的职责。
   - 从 `Request` 私有字段直接保存 physical blocks 的模式中迁出；第一步可以保留兼容字段，但新逻辑应走 block table。
   - block table 只记录 request 与 KV cache block 的映射，不承担 workload 解析。
   - 暂时不改变 `BlockAllocator` 的 counting-only 行为，除非为了 block id/ref count 必须补齐。

7. 验证。
   - synthetic benchmark 不变。
   - 小 JSON dataset 不为空。
   - 设置 `--random_seed` 后 synthetic workload 可复现。
   - 新 block table 默认路径与旧 block 计数结果对齐。
   - 新旧 CLI 参数等价：`--request_count` 等价于旧 `--prompt_count`，`--model` 等价于旧 `--psla`。
   - 新旧长度参数等价：`--prefill_mean_len`/`--decode_mean_len` 等价于对应旧 `--*_lens_mean` 参数。

验收标准：

- `util.request.get_requests()` 不再直接承载所有 workload 解析细节。
- 现有 CLI 仍能跑。
- 新代码路径和文档使用 `prefill_len` / `decode_len` / `request_count` / `--model`，以及 `prefill_mean_len` / `decode_mean_len` 等新分布参数。
- workload record 能表达 `hash_ids`，为 Phase 2 的 input-only reuse 做准备。
- `Request` 不再作为 request->KV block 映射的唯一事实来源。

## 阶段 2：输入 prefix KV cache reuse pattern

目的：支持输入 prompt 的 block-level prefix reuse。对带 `hash_ids` 的输入做调度前 reuse 判断；对没有 `hash_ids` 的输入保持原行为。

参考：

- `ref/vllm/docs/design/prefix_caching.md`
- `ref/vllm/vllm/v1/core/kv_cache_manager.py`
- `ref/vllm/vllm/v1/core/block_pool.py`
- `ref/Mooncake/benchmarks/storage_benchmark/storage_benchmark.py`

目标文件：

- `TokenSim/llm/llm_request.py`
- `TokenSim/block/block.py`
- `TokenSim/block/block_manager.py`
- `TokenSim/llm/llm_scheduler.py`
- `util/results.py`
- workload loader 相关文件

设计要求：

1. 第一版只实现输入 block-level prefix reuse，不做完整 tokenizer。
   - 对 trace 请求，直接使用输入数据中的 `hash_ids`。
   - 如果数据集提供 `output_hash_ids`，这些字段可用于把本次输出产生的 full blocks 建成后续可复用索引。
   - 如果没有 `output_hash_ids`，decode 输出默认不写入 prefix cache。
   - 对没有 `hash_ids` 的请求，`cached_prompt_blocks=0`，完整 prefill，行为与 baseline 一致。
   - 对 synthetic 请求，可通过 crafted dataset 或手工构造 `hash_ids` 测试。
   - full block 才允许缓存，保持 vLLM 语义。

2. prefix key 设计参考 vLLM。
   - 不能只用单个 raw `hash_id` 判断命中；需要形成 prefix-aware key，等价于 parent hash + 当前 block hash/token signature + extra hash。
   - extra hash 预留：model、cache_salt、tenant/reuse_group、LoRA/multimodal 等。
   - 命中必须是连续前缀命中；中间 miss 后的后续 hit 不可复用。

3. 改造 block manager。
   - 当前 `BlockAllocator` 只计数，不足以表达 cache hit、ref count、eviction。
   - 新增或改造为 `BlockTable` + `BlockPool`/`KVCacheManager` 风格：
     - `block_id`
     - `block_hash`
     - `ref_count`
     - `is_full`
     - `device/tier`
     - free queue / LRU 信息
     - hash -> blocks 映射
     - request -> blocks 映射
   - 允许多个请求共享同一个 physical block，必须通过 `ref_count` 管理生命周期。
   - 如果一步到位风险太大，可先保留 counting-only backend，同时新增 hash-aware backend，但共享 physical block + ref_count 是 Phase 2 的目标语义。

4. scheduler 行为。
   - 新请求进入 running 前先做 input prefix reuse plan。
   - 命中 block 需要增加 ref count/touch，避免被驱逐。
   - miss 部分正常分配 block 并计算 prefill。
   - decode 输出默认不写入 prefix cache。
   - 只有当输入数据集显式提供 `output_hash_ids`，并且这些 full blocks 可作为后续输入的一部分时，才允许把这些 block 加入可复用索引。
   - 请求结束后 ref count 下降，未被引用的 cached blocks 进入 free queue 等待 LRU 驱逐。

5. latency 行为。
   - reuse 命中的 token/block 不再计入 full prefill compute。
   - 若命中 block 在 GPU 上，基本不加传输开销。
   - 若后续 Phase 6 引入 Mooncake Store，reuse 需要区分 local GPU hit、Mooncake memory hit、Mooncake disk hit。

6. 指标。
   - `reuse_hit_blocks`
   - `reuse_miss_blocks`
   - `reuse_hit_tokens`
   - `effective_prefill_tokens`
   - `prefix_cache_hit_rate`
   - cache eviction count

验收标准：

- 未启用 reuse 时，benchmark 行为与 baseline 对齐。
- crafted dataset 中两个请求共享前缀时，第二个请求出现 reuse hit。
- crafted dataset 中存在非连续 block 命中时，只复用连续前缀部分。
- 无 `hash_ids` 的 dataset 与旧逻辑一致。
- result JSON 中能看到 reuse 指标。

## 阶段 3：vLLM-like KV transfer connector，并删除旧 swap

目的：把当前 prompt/decode disaggregation 和内存搬运逻辑从 `LLMEngine` / `LLMPagedAttnScheduler` 的旧 swap 状态机中迁出，改为 vLLM v1 风格的 KV transfer connector。阶段完成后删除旧 `swap_in/out` 语义，不保留兼容路径。

参考：

- `ref/vllm/docs/features/disagg_prefill.md`
- `ref/vllm/vllm/config/kv_transfer.py`
- `ref/vllm/vllm/distributed/kv_transfer/kv_connector/factory.py`
- `ref/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`
- `ref/vllm/vllm/v1/core/sched/scheduler.py`
- `ref/vllm/vllm/v1/worker/gpu/kv_connector.py`
- `ref/vllm/docs/design/plugin_system.md`

目标文件：

- `TokenSim/llm/llm_engine.py`
- `TokenSim/llm/llm_scheduler.py`
- `TokenSim/llm/llm_comm.py`
- `TokenSim/block/block_manager.py`
- `TokenSim/config/config.py`
- `TokenSim/kv_transfer/`
- `TokenSim/placement/`
- `TokenSim/latency/`
- `data/`

任务：

1. 明确旧 swap 删除范围。
   - 当前 `Task.SWAP`、`Task.SWAP_LOCAL`、`Task.SWAP_IN_REMOTE`、`Task.SWAP_OUT_REMOTE`、`Task.SWAP_IN_REMOTE_DONE` 是 legacy。
   - 当前 `SwapMetadata`、`SwapInMetadata`、`RequestStatus.SWAPPED`、`RequestStatus.SWAPPED_REMOTE` 是 legacy。
   - 当前 `BlockManager.swap_in()`、`swap_out()`、`remote_swap_in_*()`、`remote_swap_out()` 是 legacy。
   - 阶段完成后不要求兼容 `--swap_policy`，CLI、脚本和数据配置应移除该路径。

2. 定义 `KVTransferConfig`，参考 vLLM 命名。
   - 字段：
     - `kv_connector`
     - `kv_role`: `kv_producer` / `kv_consumer` / `kv_both`
     - `kv_buffer_device`
     - `kv_connector_extra_config`
     - `kv_parallel_size`
     - `kv_rank`
     - `engine_id`
   - 保持内部 registry 即可，不需要 Python entry point 或外部 module path。
   - 配置可以单独放在 `data/kv_transfer/*.json`，也可以嵌入 cluster config；实现前需先选定一种并在文档中固定。

3. 定义 vLLM-like connector 接口。
   - Scheduler side：
     - `on_new_request(req)`
     - `get_num_new_matched_tokens(req, num_computed_tokens)`
     - `update_state_after_alloc(req, blocks, num_external_tokens)`
     - `build_connector_meta(scheduler_output)`
     - `update_connector_output(worker_output)`
     - `request_finished(req, block_ids)`
     - `take_events()`
     - `reset_cache()`
   - Worker side：
     - `bind_connector_metadata(meta)`
     - `handle_preemptions(meta)`
     - `start_load_kv(...)`
     - `wait_for_layer_load(layer)`
     - `save_kv_layer(layer, ...)`
     - `wait_for_save()`
     - `get_finished(finished_req_ids)`
     - `build_connector_worker_meta()`
   - TokenSim 不做真实 Tensor copy，但 metadata、async load/save、latency accounting 的状态机要保留。

4. 定义 connector factory。
   - 参考 vLLM `KVConnectorFactory`。
   - 内置：
     - `NoopConnector`：无外部 KV transfer。
     - `P2PConnector` 或等价内部名：用于迁移现有 P/D KV transfer 行为，但不再叫 swap。
     - `MooncakeConnector`：后续 Phase 6 实现，当前只注册占位。
     - `MooncakeStoreConnector`：后续 Phase 6 实现，当前只注册占位。
     - `MultiConnector`：组合 `MooncakeConnector` 与 `MooncakeStoreConnector`。
   - 不新增 `LocalSwapConnector` / `RemoteKVConnector` 作为长期命名；已有同名代码需要在本阶段迁移或删除。

5. 对齐 vLLM v1 scheduler 关键流程。
   - 新请求调度前：
     - 本地 prefix cache 查询。
     - connector `get_num_new_matched_tokens()` 查询外部 KV。
   - 分配 blocks 时：
     - 把 `num_external_computed_tokens` 传给 block/KV cache manager。
     - 对 async load 的请求进入等待远端 KV 的状态，不再用 `SWAPPED_REMOTE`。
   - 分配后：
     - connector `update_state_after_alloc()` 记录本次 load/save 计划。
   - scheduler output：
     - `build_connector_meta()` 生成 worker 可消费的 metadata。
   - worker step：
     - 根据 connector metadata 模拟 load/save KV，统计传输 latency。
   - 请求完成：
     - `request_finished()` 决定是否延迟释放 blocks。

6. 抽出 placement policy。
   - 当前 worker pool / prompt-decode 分发从 `llm_engine.py` 移到 `TokenSim/placement/`。
   - 接口示例：
     - `select_prefill_worker(request, workers)`
     - `select_decode_worker(request, workers)`
     - `select_transfer_target(requests, workers)`
   - 保留 round-robin、least-GPU-memory、balanced-load。

7. engine/worker 收口。
   - `LLMEngine.run` 只做 request 接收、worker 选择、scheduler/worker output 聚合、stop/broadcast。
   - 不再在 engine match 分支中判断 P/D swap 类型。
   - `LLMWorker.run` 不再处理 `Task.SWAP_*`；KV transfer 由 worker-side connector metadata 驱动。

8. 指标。
   - 每个 connector 的 transfer count、blocks、bytes、latency。
   - load/save/prefetch/store 的等待时间。
   - producer/consumer/both 角色统计。
   - placement policy 决策统计。
   - legacy swap counter 应删除或重命名为 connector transfer counter。

验收标准：

- `Task.SWAP_*`、`SwapMetadata`、旧 `swap_in/out` 逻辑从主路径删除。
- `scripts/benchmark.sh` 默认行为在数值上与 baseline 对齐，差异需解释。
- PD 多 worker benchmark 能通过 connector metadata 路径完成。
- 新增 connector 不需要修改 `LLMEngine.run` 的 match 分支。
- 默认 `NoopConnector` 和 P/D transfer connector 有单独测试。

## 阶段 4：TP/PP/DP 基础并行模拟

目的：为 671B 等大模型提供基础并行表达能力。该阶段先支持 dense model 的 TP/PP/DP，不引入 MoE/EP，也不实现 Mooncake Store。

参考：

- `ref/vllm/vllm/config/parallel.py`
- `ref/vllm/vllm/v1/core/sched/scheduler.py`
- `ref/vllm/vllm/v1/metrics/perf.py`
- TransformerRoofline 现有 model / hardware 配置

目标文件：

- `TokenSim/config/`
- `TokenSim/llm/llm_engine.py`
- `TokenSim/llm/llm_scheduler.py`
- `TokenSim/latency/`
- `TokenSim/block/`
- `data/psla/`
- `data/clusters/`
- `benchmark.py`

任务：

1. 定义 parallel config。
   - 字段参考 vLLM：
     - `tensor_parallel_size`
     - `pipeline_parallel_size`
     - `data_parallel_size`
     - `data_parallel_rank`
     - `data_parallel_size_local`
   - 暂不启用：
     - `enable_expert_parallel`
     - `expert_placement_strategy`
     - `all2all_backend`
   - CLI、cluster config、model config 的优先级需要明确。

2. 建立 worker group / rank 映射。
   - 为每个 worker 建立：
     - `tp_rank`
     - `pp_rank`
     - `dp_rank`
     - global rank
   - 支持按 DP group 复制完整服务实例。
   - placement policy 需要先选择 DP group，再选择该 group 内的 prefill/decode worker。

3. 修正模型参数和 KV memory 计算。
   - `CacheConfig.model_param_size` 不能继续只按 dense 公式固定估算。
   - TP 影响每张卡持有的 weight shard。
   - PP 影响每张卡持有的 layer subset 和 per-token KV bytes。
   - DP 复制完整 TP/PP 组，影响总吞吐和负载均衡，不降低单 group 内存。
   - KV block key 需要预留 `tp_rank`、`pp_rank`、`kv_cache_group_id`，以便后续 Mooncake Store key 与 vLLM 对齐。

4. latency backend 支持 TP/PP/DP。
   - TP：建模 attention/MLP shard 后的 compute 变化，以及必要 collective latency。
   - PP：建模 stage latency、pipeline bubble、跨 stage activation/KV 相关通信。
   - DP：建模多副本请求分发和聚合指标，不把 DP 当成单请求加速。
   - 第一版允许使用 calibration / measured constants，但配置字段要清楚。

5. 调度和结果。
   - scheduler 的 batch token budget 按 DP group / PP stage / TP group 计算。
   - result JSON 输出 parallel config、rank count、per-rank utilization。
   - 默认 `tp=pp=dp=1` 时 baseline 不变。

验收标准：

- `tp=pp=dp=1` 与 baseline 对齐。
- 设置 TP 后单卡 weight footprint 下降，KV block bytes 与 rank/group 语义正确。
- 设置 PP 后不同 stage 的 GPU occupancy 和 latency 可观测。
- 设置 DP 后请求能分发到不同 DP group，吞吐随副本数变化。
- 新配置能表达 671B dense model 的基础部署形态。

## 阶段 5：MoE 模型与 EP 模拟

目的：支持 MoE 模型，尤其是大模型场景中的 expert parallelism。该阶段在 Phase 4 的 TP/PP/DP 基础上增加 MoE/EP，不引入 Mooncake Store 完整行为。

参考：

- `ref/vllm/vllm/config/parallel.py`
- `ref/vllm/vllm/v1/metrics/perf.py`
- `ref/vllm/vllm/model_executor/layers/fused_moe/`
- `ref/vllm/csrc/moe/`
- Mooncake README 中 Elastic Expert Parallelism 相关说明

目标文件：

- `TokenSim/config/`
- `TokenSim/latency/`
- `TokenSim/placement/`
- `TokenSim/llm/llm_scheduler.py`
- `TokenSim/llm/llm_request.py`
- `data/psla/`
- `data/clusters/`

任务：

1. 扩展 model config，表达 MoE 参数。
   - 字段：
     - `is_moe_model`
     - `num_experts`
     - `num_experts_per_tok`
     - `moe_intermediate_size`
     - `num_shared_experts`
     - `num_moe_layers`
     - `first_k_dense_replace`
     - `moe_layer_freq`
     - `interleave_moe_layer_step`
   - dense model 保持兼容。

2. 扩展 parallel config，表达 EP。
   - 字段参考 vLLM：
     - `enable_expert_parallel`
     - `expert_placement_strategy`: `linear` / `round_robin`
     - `all2all_backend`
     - `enable_eplb`
     - `num_redundant_experts`
   - EP 与 TP/DP 的组合规则需要明确，尤其是 vLLM 中 MoE layer 可按 DP/TP 维度分片。

3. routing workload 模型。
   - 第一版可使用概率分布模拟 top-k expert routing，不需要真实 token router。
   - 支持从 Qwen trace 输入 per-token 或 per-request expert histogram。
   - 支持 skew、hot expert、burst pattern，用于验证 EPLB 和 Mooncake 压力。

4. MoE latency 模型。
   - dense attention 仍按原模型。
   - MoE FFN compute 按 active experts / top-k / token count 估算。
   - EP 需要建模 all-to-all latency、expert compute imbalance、straggler。
   - 支持 measured constants / calibration table。
   - 输出 prefill/decode 阶段的 MoE compute 与 all-to-all latency breakdown。

5. expert placement / load balancing。
   - 实现 `linear` 和 `round_robin` placement。
   - EPLB 第一版可只模拟周期性 rebalancing 的效果和迁移成本。
   - 支持记录 expert load、max/mean load、imbalance ratio。

6. memory model。
   - expert weights 只占所在 EP rank 的内存。
   - shared experts 和 dense layers 按 TP/PP/DP 规则计算。
   - KV cache 通常不按 expert 分片，但 block key 和 result 中要记录并行上下文。

验收标准：

- dense model 默认行为不变。
- MoE model config 能表达 DeepSeek-R1/V3 671B 这类规模的关键参数。
- EP 开启后专家 weight footprint、all-to-all latency、expert load 指标可观测。
- routing skew case 能看到 expert imbalance，并能通过 EPLB 配置改变模拟结果。
- TP/PP/DP/EP 组合下 result JSON 能解释每类并行配置。

## 阶段 6：Mooncake / Mooncake Store 模拟器

目的：在 vLLM-like connector 和 TP/PP/DP/MoE/EP 基础上，构建 TokenSim 内部 Mooncake 模拟器。Mooncake Store 是当前 SSD offloading 后段；不实现独立 generic SSD offload backend，不调用真实 Mooncake API。

参考：

- `ref/Mooncake/README.md`
- `ref/Mooncake/docs/source/design/transfer-engine/index.md`
- `ref/Mooncake/docs/source/design/mooncake-store.md`
- `ref/Mooncake/benchmarks/storage_benchmark/storage_benchmark.py`
- `ref/vllm/docs/features/mooncake_connector_usage.md`
- `ref/vllm/docs/features/mooncake_store_connector_usage.md`
- `ref/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/`
- `ref/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/`

目标文件：

- `TokenSim/kv_transfer/`
- `TokenSim/kv_transfer/mooncake/`
- 可新增 `TokenSim/mooncake/`
- `TokenSim/config/`
- `TokenSim/block/`
- `TokenSim/llm/llm_engine.py`
- `TokenSim/llm/llm_comm.py`
- `TokenSim/latency/`
- `data/`

任务：

1. 明确模拟范围和集成方式。
   - 模拟器集成在 TokenSim 内部，不另开项目。
   - `MooncakeConnector`：模拟 P/D 之间基于 Transfer Engine 的 P2P KV transfer。
   - `MooncakeStoreConnector`：模拟共享 KV pool，支持跨实例 prefix cache、memory tier、SSD persistence/offloading。
   - `MultiConnector`：组合 P2P transfer 与 shared store。
   - 所有命名与 vLLM 保持一致。
   - 不调用真实 Mooncake API；只模拟 metadata、调度行为、容量、命中、传输和 latency。

2. Mooncake Transfer Engine 模型。
   - 抽象为 `Segment` 和 `BatchTransfer`：
     - source segment / target segment
     - source tier / target tier: `vram` / `dram` / `ssd`
     - blocks / bytes
     - protocol: `tcp` / `rdma` / `nvlink` / `nvmeof`
     - fixed latency
     - bandwidth
     - num NICs / parallel paths
     - topology aware path selection
   - 支持 async transfer job 和完成轮询。
   - 支持 transfer overlap：compute 与 load/store 是否重叠由配置控制。

3. Mooncake Store metadata 模型。
   - Master 只管 metadata 和空间分配，不在数据路径上。
   - Client 可同时是 requester 和 segment owner。
   - 支持部署模式：
     - `embedded`
     - `standalone-store`
   - 支持 segment：
     - GPU/VRAM local segment
     - DRAM/global segment
     - SSD segment
   - 支持 object：
     - key = `PoolKey`
     - value size = KV block bytes
     - replicas
     - preferred segment
     - replica tier: memory / disk

4. Pool key 与 vLLM 对齐。
   - key metadata 至少包含：
     - `model_name`
     - `tp_rank`
     - `pp_rank`
     - `dp_rank` 或 engine/group id
     - `group_id`
     - `chunk_hash`
   - 参考 vLLM `PoolKey.to_string()`：
     - model
     - `tp_rank`
     - `pcp`
     - `dcp`
     - `pp_rank`
     - `group`
     - block hash
   - TokenSim 暂不需要实现 PCP/DCP，但字段要预留。

5. MooncakeStoreConnector scheduler-side 行为。
   - `get_num_new_matched_tokens()` 查询 store prefix hit。
   - local GPU hit 优先，其次 Mooncake memory hit，再 Mooncake disk hit。
   - 对命中的外部 KV，返回 `num_external_computed_tokens` 和是否 async load。
   - `update_state_after_alloc()` 记录本地 block ids 和外部命中 token。
   - `build_connector_meta()` 生成本步 load/save metadata。
   - `request_finished()` 决定是否延迟释放 blocks，直到 async put 完成。

6. MooncakeStoreConnector worker-side 行为。
   - 根据 metadata 模拟 get/put。
   - Store memory hit：按 network/RDMA/DRAM 路径计算 latency。
   - Store disk hit：按 SSD read + staging buffer + network transfer 计算 latency。
   - miss 后可按 admission policy 写入 Store。
   - persistence enabled 时，Put 可异步触发 SSD persistence。
   - 不做 layer-level tensor copy，但保留 load/save、wait、finished metadata。

7. SSD offloading 后段。
   - SSD 只通过 Mooncake Store 表达，不新增独立 `ssd` backend。
   - 参考 Mooncake storage benchmark 建模：
     - `block_hash -> offset`
     - fixed block size
     - capacity / max blocks
     - free offset / LRU
     - read/write latency
     - read/write bytes
     - staging buffer size
   - Store disk hit 和 persistence 写入都需要进入 Mooncake 指标。

8. 配置。
   - 示例：
     - `data/kv_transfer/mooncake_p2p.json`
     - `data/kv_transfer/mooncake_store_embedded.json`
     - `data/kv_transfer/mooncake_store_standalone_ssd.json`
     - `data/kv_transfer/mooncake_multi_connector.json`
   - 配置字段：
     - `kv_connector`: `MooncakeConnector` / `MooncakeStoreConnector` / `MultiConnector`
     - `mode`: `embedded` / `standalone-store`
     - `protocol`
     - `global_segment_size`
     - `local_buffer_size`
     - `enable_offload`
     - `ssd_capacity_gb`
     - `ssd_read_bw_gbps`
     - `ssd_write_bw_gbps`
     - `ssd_read_latency_us`
     - `ssd_write_latency_us`
     - `replica_num`
     - `admission_policy`
     - `eviction_policy`
     - `load_async`
     - `transfer_overlap`

9. 指标。
   - Mooncake get/put count。
   - Store hit/miss。
   - memory-tier hit / disk-tier hit。
   - SSD read/write blocks 和 bytes。
   - transferred bytes。
   - effective transfer bandwidth。
   - P2P vs Store latency。
   - store admission / eviction count。
   - async pending jobs。
   - prefix cache hit 按 tier 分解：
     - local GPU
     - Mooncake memory
     - Mooncake disk

验收标准：

- 当前阶段不要求真实 Mooncake API 可用。
- `MooncakeConnector`、`MooncakeStoreConnector`、`MultiConnector` 与 Phase 3 connector 接口兼容。
- 非 Mooncake 默认模式不受影响。
- 开启 `MooncakeStoreConnector` 后 crafted reuse case 能看到 store hit。
- 开启 SSD offloading 后能看到 Mooncake disk-tier hit 和 SSD read/write 指标。
- 671B/MoE/EP 配置下 PoolKey 和 block bytes 与并行上下文一致。

## 阶段 7：结果、测试和文档收口

目的：保证重构后的系统可用、可比较、可继续扩展。

任务：

1. 结果格式。
   - 扩展 `LLMResult`，加入 reuse/connector/Mooncake/parallel/MoE 指标。
   - 保持旧字段不变，避免破坏已有结果处理脚本。

2. 最小测试。
   - workload loader 单元测试。
   - prefix reuse crafted case。
   - vLLM-like connector factory 创建测试。
   - legacy swap 删除后的 PD 多 worker smoke test。
   - TP/PP/DP memory 和 latency smoke test。
   - MoE/EP routing skew crafted case。
   - Mooncake Store memory-tier crafted case。
   - Mooncake Store SSD-tier crafted case。

3. 示例脚本。
   - 保留 `scripts/benchmark.sh`。
   - 新增可选脚本：
     - `scripts/benchmark_reuse.sh`
     - `scripts/benchmark_parallel.sh`
     - `scripts/benchmark_moe_ep.sh`
     - `scripts/benchmark_mooncake.sh`

4. 文档。
   - 更新 `README.md` 的 quick start。
   - 在公共开发文档中记录新增模块边界。

验收标准：

- 默认 benchmark、reuse benchmark、PD connector benchmark、parallel benchmark、MoE/EP benchmark、Mooncake benchmark 都有明确命令。
- 新增指标在 result JSON 中可解释。
- 后续开发者能从 `tasks.md` 和 `README.md` 判断应该改哪个模块。

## 推荐实施顺序

1. Phase 0：baseline。
2. Phase 1：代码基础改造。
3. Phase 2：KV cache reuse。
4. Phase 3：vLLM-like KV transfer connector，并删除旧 swap。
5. Phase 4：TP/PP/DP 基础并行模拟。
6. Phase 5：MoE 模型与 EP 模拟。
7. Phase 6：Mooncake / Mooncake Store 模拟器。
8. Phase 7：测试、指标、文档收口。

说明：

- reuse 依赖 workload 表达和 block hash。
- Mooncake Store 依赖 reuse、vLLM-like connector、并行 rank/key 语义，所以不要先做 Mooncake 特判。
- SSD offloading 当前只通过 Mooncake Store 表达，不先做独立 offload backend。
- TP/PP/DP 先于 MoE/EP；MoE/EP 先于 Mooncake 完整 Store。
- 旧 swap 不做长期兼容，Phase 3 完成后应删除。

## 全局验证命令

基础验证：

```bash
python -m compileall benchmark.py TokenSim util
./scripts/benchmark.sh
```

Qwen Bailian trace 验证：

```bash
./benchmark.py \
  --batching paged-attn \
  --block_size 16 \
  --request_count 20 \
  --cluster ./data/clusters/1_a100/h1.json \
  --dataset_path ./qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl \
  --workload_type qwen_jsonl \
  --qps 5.9807 \
  --max_parallem_sum 100 \
  --verbose simple \
  --model ./data/psla/llama-7b.json
```

PD 多 worker 验证：

```bash
./benchmark.py \
  --batching paged-attn \
  --block_size 16 \
  --request_count 50 \
  --cluster ./data/clusters/8_a100/p2d5.json \
  --qps 50 \
  --max_parallem_sum 100 \
  --verbose none \
  --model ./data/psla/llama-7b.json
```
