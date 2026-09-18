# 算子延迟数据采集（`scripts/collect/`）

本目录的脚本在**目标 GPU 节点**上采集 `operator_table` 后端所需的 A 级实测数据，并直接写成
`data/operator_data/<device_id>/<backend>/` 下的算子包。它取代了原来的 `scripts/profiling/`：
GEMM、attention、MoE 与单机集合通信改由 NVIDIA AIConfigurator 的 collector 采集（它跑的是
TensorRT-LLM / vLLM / SGLang 真实 serving kernel，而不是 PyTorch eager 算子），TokenSim 只保留
collector 不覆盖的两块：逐元素算子和跨节点集合通信。

| 脚本 | 产出的表 | 依赖 |
| --- | --- | --- |
| `run_aiconfigurator.sh` | `gemm`、`context_attention`、`generation_attention`、`moe`，以及单机 `collective`（NCCL 四种操作 + 框架自定义 all-reduce） | 目标框架（默认 TensorRT-LLM）、AIConfigurator 源码、nccl-tests、mpirun |
| `collect_elementwise.py` | `elementwise`（rmsnorm / rope / residual_add / swiglu） | torch；`--kernel vllm` 时需要 vLLM |
| `run_nccl_tests.sh` | 多节点或指定 fabric 的 `collective` | nccl-tests，多机时需 mpirun/srun |
| `common.py` | 计时与环境记录的公共函数 | torch |

三者的输出都进入同一个包目录，`OperatorDataPackage.write()` 会按主键合并（新行覆盖同键旧行），
`generation_meta.yaml` 记录每个 `source_id` 的设备、驱动、框架版本、时钟状态、采集日期与采集人。

---

## 0. 采集前

1. **独占节点，锁频，开持久模式。** 未锁频的数据会在 boost 与降频之间漂移 10% 以上。

   ```bash
   sudo nvidia-smi -pm 1
   sm=$(nvidia-smi -q -i 0 | grep -A4 "Max Clocks" | grep "SM " | grep -o "[0-9]\+" | head -1)
   mem=$(nvidia-smi -q -i 0 | grep -A4 "Max Clocks" | grep "Memory " | grep -o "[0-9]\+" | head -1)
   sudo nvidia-smi -ac "$mem,$sm"
   ```

   若没有 root，把 `nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv` 的输出记入
   来源说明（`run_aiconfigurator.sh` 会自动保存到 `<RUN_DIR>.env.txt`）。

2. **确认设备在目录中。** `data/devices/<device_id>.yaml` 必须存在，例如 `a100_sxm_80g`、
   `rtx_4090`。新设备先照 `data/devices/README.md` 补一份 YAML。

3. **准备框架环境。** collector 的每个采集模块声明了兼容版本（`__compat__`）。推荐直接用上游
   `collector/framework_manifest.yaml` 里 digest 固定的容器镜像；本仓库默认后端是 TensorRT-LLM：

   | 后端 | 版本（AIConfigurator `f254959` 使用） | 镜像 |
   | --- | --- | --- |
   | trtllm | 1.3.0rc20（A100 数据用 1.0.0） | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20` |
   | vllm | 0.24.0 | `vllm/vllm-openai:v0.24.0` |
   | sglang | 0.5.14 | `lmsysorg/sglang:v0.5.14` |

   仿真时的默认后端是 `vllm`（集群未写 `operator_backend` 时按 vllm → trtllm → sglang 顺序选择），
   采集脚本的 `BACKEND` 默认仍为 `trtllm`，请按目标部署栈显式指定。

   容器里还要有：本仓库（挂载到 `/workspace/TokenSim`）、`pyarrow`、`PyYAML`，以及编译好的
   [nccl-tests](https://github.com/NVIDIA/nccl-tests)（`make MPI=1`，二进制目录加入 `PATH`）。

4. **生成形状清单**（给 `collect_elementwise.py` 用，也用来事后检查覆盖率）：

   ```bash
   python -m TokenSim.operator_data.cli manifest --models llama-3-8b,llama-3-70b,mixtral-8x7b \
       --tp 1,2,4,8 --out tmp/manifests/a100-llama3.yaml
   ```

---

## 1. `run_aiconfigurator.sh`：计算算子与单机集合通信

```bash
DEVICE=a100_sxm_80g ./scripts/collect/run_aiconfigurator.sh
```

脚本做的事：

1. 把 AIConfigurator 克隆到 `AIC_DIR`（默认 `./tmp/aiconfigurator`）并检出 `AIC_COMMIT`
   （默认与 `data/operator_data/LICENSES.md` 记录的导入提交一致，保证形状网格和列名与已有数据相同）。
2. 检查目标框架可导入，打印时钟状态。
3. 在 `RUN_DIR`（默认 `./tmp/collect/<DEVICE>/<BACKEND>`）里运行
   `collector/collect.py --backend <BACKEND> --ops <OPS> --resume`。collector 把
   `gemm_perf.parquet`、`context_attention_perf.parquet`、`generation_attention_perf.parquet`、
   `moe_perf.parquet` 写到当前目录；`--resume` 让中断后重跑只补缺失用例。
4. 运行 `collector/network/collect_comm.sh --all_reduce_backend <BACKEND>`：对本机 2/4/8 卡跑
   nccl-tests 的 all_gather / alltoall / reduce_scatter / all_reduce（half 与 int8，512 B 到 512 MB）
   和框架自定义 all-reduce，产出 `nccl_perf.txt`、`custom_allreduce_perf.txt`。
5. 调用 `python -m TokenSim.operator_data.cli import-aiconfigurator --system-dir <RUN_DIR>`。
   导入器识别这种"平铺"目录（也能直接读 `*_perf.txt` CSV），把毫秒换成微秒、精度名规范化、
   同一形状取最快 kernel，`source_id` 形如 `aiconfigurator-collector:<device>:<family>:<backend>:<version>`，
   等级 A、方法 `measured`。然后 `validate` 校验包。

常用变量：

| 变量 | 作用 |
| --- | --- |
| `OPS="gemm attention"` | 只采部分算子（可选 `gemm attention moe`，MoE 网格最大，A100 上约需数小时） |
| `MODEL_PATH=meta-llama/Llama-3.1-8B` | 模型中心模式：只采该模型结构需要的形状，对应上游的 "healing run" |
| `GPU_TYPE=a100_sxm` | 用上游系统名解析精度能力下限；4090 这类上游没有的卡留空，collector 会按本机 SM 版本判断（4090 是 sm_89：bf16 + fp8 + fp8_block） |
| `SKIP_COMM=1` | 跳过集合通信采集（没有 nccl-tests 或 mpirun 时） |
| `BACKEND=vllm` | 换框架；包目录随之变为 `<device>/vllm/`，仿真时用 `--operator_backend vllm` 选择 |
| `--import-only` | 不重新测量，只把已有 `RUN_DIR` 转成包 |

网格覆盖范围（上游 `collector/cases/base_ops/*.yaml`）：GEMM 的 token 数 1–17 逐个、之后
32/33、48/49 … 3328/3329 成对采样以捕捉 tile 边界，`n`、`k` 覆盖 32–65536 共 22 个值；
decode attention 的 batch 1–2048、上下文 2–131072、head 数 1–128；MoE 的 token 数 1–65536。
注意 collector 的每次测量是 CUDA graph 中多次 kernel 的均值，与 TokenSim 的"单次 kernel"口径一致。

**4090 的注意事项**：TensorRT-LLM 对消费级 Ada 支持有限，若模块报 `unsupported`，改用
`BACKEND=vllm`。多卡 4090 间没有 NVLink，`collect_comm.sh` 采到的就是 PCIe 曲线，正是
`pcie_8x_4090` 拓扑所需的数据。

---

## 2. `collect_elementwise.py`：逐元素算子

```bash
python scripts/collect/collect_elementwise.py \
    --manifest tmp/manifests/a100-llama3.yaml --device a100_sxm_80g \
    --package data/operator_data/a100_sxm_80g/trtllm --kernel eager --operator zhang
```

* 从清单的 `elementwise` 表取形状（`op_name, dtype, num_tokens, hidden_size`）。
* `--kernel eager`：PyTorch 逐算子实现（上界参考）；`--kernel vllm`：vLLM 的
  `rms_norm` / `fused_add_rms_norm` / `rotary_embedding` / `silu_and_mul` fused kernel（生产口径）。
* CUDA event 计时，5 次预热后取 20 次中位数。
* 直接合并进 `--package` 指定的包：已有行按主键被新行覆盖，`sources` 增加一条
  `measured:<device>:elementwise:<kernel>`，notes 里含设备名、驱动/时钟、torch/CUDA 版本、日期与采集人。

---

## 3. `run_nccl_tests.sh`：多节点集合通信

```bash
# 单机 8 卡（与 collector 重复，用于对照）
DEVICE=a100_sxm_80g NCCL_TESTS=/opt/nccl-tests/build ./scripts/collect/run_nccl_tests.sh 8 1
# 两节点 16 卡，每卡一个进程
DEVICE=a100_sxm_80g NODES=2 LAUNCHER="mpirun -np 16 -H nodeA:8,nodeB:8 -x NCCL_DEBUG=INFO" \
    ./scripts/collect/run_nccl_tests.sh 8 2
```

对 all_reduce / all_gather / reduce_scatter / all_to_all 各跑一次 1 KiB–1 GiB（2 倍步长）的扫描，
`operator_data.cli import-nccl` 把每条曲线写入 `collective` 表，主键为
`(dtype, operation, group_size, nodes, message_bytes)`。多机时请保留 `NCCL_DEBUG=INFO` 输出中
选中的算法/协议（ring/tree、LL/Simple），用于校准拓扑 YAML 的 `collective_algorithm`。

---

## 4. 采集后

```bash
python -m TokenSim.operator_data.cli validate data/operator_data/a100_sxm_80g/trtllm
python -m TokenSim.operator_data.cli coverage --device a100_sxm_80g --backend trtllm \
    --models llama-3-8b --tp 1,2,4,8 --out tmp/coverage-a100.yaml
python -m TokenSim.operator_data.cli calibrate --device a100_sxm_80g --backend trtllm --out tmp/calib-a100.yaml
```

* `coverage` 报告清单中每个查询是精确命中、插值、外推还是缺失，缺失项就是下一轮要补的形状。
* `calibrate` 用实测表拟合设备 YAML `analytical:` 段的效率参数并给出 holdout 误差；把结果
  写回 YAML 时把 `grade` 标为 `C`。
* `step_overhead_us`（调度与 launch 的固定项）不在任何表里：在真实框架跑 batch=1、seq=1 的
  decode，用实测 step 时间减去按表组合的算子时间得到，填入设备 YAML。

---

## 5. 记录规范

每个包的 `generation_meta.yaml` 中，每条 `sources` 必须能回答：设备型号（`nvidia-smi` 名称）、
驱动与 CUDA、框架及版本、是否锁频、测量方法与迭代次数、日期、采集人。三个脚本都会自动写入这些字段；
手工导入的数据请补全 `--notes`。

上游 collector 的 parquet 还带 `framework`、`version`、`device`、`kernel_source` 列，导入时保留在
`kernel` 列中，用于区分同一形状下不同 kernel 的结果。

---

## 6. 与旧 `scripts/profiling/` 的对应关系

| 旧脚本 | 现在 |
| --- | --- |
| `profile_gemm.py`（torch cuBLAS） | `run_aiconfigurator.sh --ops gemm`（框架真实 GEMM kernel，含量化） |
| `profile_attention.py`（FlashAttention / SDPA） | `run_aiconfigurator.sh --ops attention`（框架的 prefill/decode attention kernel，含 fp8 KV） |
| `profile_elementwise.py` | `collect_elementwise.py`（新增 vLLM fused kernel 选项，直接写包） |
| `run_nccl_tests.sh` | `run_nccl_tests.sh`（新增多机 launcher、直接写包） |
| README 中未实现的 `profile_moe.py` | `run_aiconfigurator.sh --ops moe` |
