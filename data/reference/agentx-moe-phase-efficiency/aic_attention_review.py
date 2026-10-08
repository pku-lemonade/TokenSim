"""Independent AIC Rust attention queries at matching Pro/TP/quant geometry.

The module interface uses the original FlashMLA source tables. This diagnostic
does not imply a complete target-vLLM mixed-step or serving accuracy result.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[2]
AIC = Path("/tmp/aiconfigurator")
SYSTEMS = AIC / "aic-core/src/aiconfigurator_core/systems"
sys.path.insert(0, str(ROOT))
from TokenSim.hardware.context import HardwareContext
from TokenSim.latency.deepseek_v4_attention_sol import DeepSeekV4AttentionSOL
from TokenSim.operator_data.lookup import OperatorLookup, LookupPolicy

sys.path.insert(0, str(AIC / "aic-core/src"))
from aiconfigurator_core.sdk.common import KVCacheQuantMode, FMHAQuantMode, GEMMQuantMode
from aiconfigurator_core.sdk.engine import EngineHandle, build_ops_json
from aiconfigurator_core.sdk.operations.dsv4 import (
    ContextDeepSeekV4AttentionModule, GenerationDeepSeekV4AttentionModule,
)
from aiconfigurator_core.sdk.perf_database import get_database


context = HardwareContext.load(ROOT / "data")
model = context.model("DeepSeek-V4-Pro")
spec = model.deepseek_v4
hardware = {}
for sku, system in (("b200", "b200_sxm"), ("b300", "b300_sxm")):
    database = get_database(system, "vllm", "0.24.0", systems_paths=str(SYSTEMS),
                            allow_unlisted_version=True, shared_layer=False)
    handle = EngineHandle.for_database(database, systems_path=str(SYSTEMS))
    sol = DeepSeekV4AttentionSOL(model, context.device(sku))
    package = context.operator_package(context.device(sku).device_id, "vllm")
    lookup = OperatorLookup(package, LookupPolicy(extrapolate="analytical", max_extrapolation_ratio=64),
                            analytical_scaler=lambda table, key: sol.estimate_us(key),
                            bind_analytical_axes=True)
    results = []
    for tp in (1, 4):
        for kind, ratio in (("csa", 4), ("hca", 128)):
            for phase, batch, queries, prefix in (
                ("context", 1, 512, 4096), ("context", 1, 512, 49152),
                ("context", 1, 4096, 61440), ("generation", 64, 1, 16384),
                ("generation", 64, 1, 131072),
            ):
                cls = ContextDeepSeekV4AttentionModule if phase == "context" else GenerationDeepSeekV4AttentionModule
                op = cls(kind, 1., model.num_attention_heads // tp, model.num_attention_heads, tp,
                         model.hidden_size, spec.q_lora_rank, spec.o_lora_rank, model.head_dim,
                         spec.qk_rope_head_dim, spec.index_n_heads, spec.index_head_dim, spec.index_topk,
                         model.sliding_window, ratio, spec.o_groups // tp,
                         KVCacheQuantMode.fp8, FMHAQuantMode.bfloat16, GEMMQuantMode.fp8_block,
                         architecture="DeepseekV4ForCausalLM")
                ops = build_ops_json([op])
                key = {"layer_kind": kind, "phase": phase, "mla_dtype": "bf16", "kv_cache_dtype": "fp8",
                       "kernel_source": "FLASHMLA_SPARSE_DSV4", "kernel_dtype": "fp8_block",
                       "model": "deepseek-ai/DeepSeek-V4-Pro", "architecture": "DeepseekV4ForCausalLM",
                       "num_heads": model.num_attention_heads // tp, "tp_size": tp, "batch_size": batch,
                       "input_seq_len": queries, "context_len": prefix, "compress_ratio": ratio}
                native = lookup.lookup("dsv4_attention", key)
                # Official generation ops consume total context through s;
                # the imported source key is the collector's past step, so
                # its total context is step+1. Prefix belongs to context ops.
                value = handle.evaluate_ops_json(ops, is_context=phase == "context", batch_size=batch,
                                                  s=queries if phase == "context" else prefix + 1,
                                                  prefix=prefix if phase == "context" else 0)[0]
                assert value[3] == "silicon"
                aic_us = value[1] * 1000
                results.append({"key": key, "op_spec": json.loads(ops)[0], "aic_us": aic_us,
                                "tokensim_us": native.latency_us, "tokensim_match": native.match_type,
                                "difference_percent": 100 * (native.latency_us / aic_us - 1),
                                "aic_source": value[3]})
    sources = []
    for phase in ("context", "generation"):
        for kind in ("csa", "hca"):
            path = SYSTEMS / f"data/{system}/sparse_attention/vllm/0.24.0/dsv4_{kind}_{phase}_module_perf.parquet"
            frozen = ROOT / f"data/reference/agentx-native-attention/originals/{system}-{path.name}"
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            assert source_hash == hashlib.sha256(frozen.read_bytes()).hexdigest()
            sources.append({"path": str(path), "sha256": source_hash,
                            "retained_original": str(frozen), "retained_original_sha256": source_hash})
    hardware[sku] = {"source_system": system, "source_framework": "vllm", "source_version": "0.24.0",
                     "queries": results, "raw_sources": sources,
                     "all_original_source_parquets_match": True,
                     "exact_context_anchors_equal": all(row["aic_us"] == row["tokensim_us"] for row in results
                                                        if row["key"]["phase"] == "context"
                                                        and row["tokensim_match"] == "exact")}

report = {
    "scope": "official independent AIC Rust attention module queries;Pro geometry/BF16 attention/FP8 KV/FP8 block projections,TP1/4 and long context",
    "aic_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AIC, text=True).strip(),
    "runtime_change": False, "formal_model_accuracy_accepted": False, "hardware": hardware,
    "input_semantics": "context:s=query,prefix=past;generation:s=source-past-step+1,prefix=0;native source key retains past-step",
    "probe_sources": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in (Path(__file__), AIC / "aic-core/src/aiconfigurator_core/_aiconfigurator_core.so",
                                   AIC / "aic-core/rust/aiconfigurator-core/src/perf_database/dsv4.rs",
                                   AIC / "aic-core/rust/aiconfigurator-core/src/perf_database/perf_interp.rs")],
    "limits": ["same source FlashMLA modules, not FlashInfer serving ground truth",
               "official AIC and native SOL/layout/interpolation paths may differ; discrepancies retained",
               "shared projection, mixed batches, q4/draft,mHC/MoE and full target-vLLM builder outside this diagnostic"],
}
destination = STAGE / "aic-attention-followup.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
for sku, row in hardware.items():
    print(sku, len(row["queries"]), "queries;max abs difference percent",
          max(abs(query["difference_percent"]) for query in row["queries"]))
