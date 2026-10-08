"""Independent AIC Rust routed-module queries with exact FP8/MXFP4 geometry.

This is an operator import/interpolation diagnostic, not a complete vLLM
serving step. The official model-less op-list interface is used; no AIC
builder support or shared/draft scope is manufactured.
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
from TokenSim.operator_data.lookup import OperatorLookup, LookupPolicy

sys.path.insert(0, str(AIC / "aic-core/src"))
from aiconfigurator_core.sdk.perf_database import get_database
from aiconfigurator_core.sdk.engine import EngineHandle, build_ops_json
from aiconfigurator_core.sdk.common import MoEQuantMode
from aiconfigurator_core.sdk.operations.dsv4 import DeepSeekV4MegaMoEModule


context = HardwareContext.load(ROOT / "data")
model = context.model("DeepSeek-V4-Pro")
selection = {"dtype": "mxfp4", "kernel_source": "deepgemm_megamoe", "kernel_dtype": "fp8_fp4",
             "distribution": "power_law_1.01", "pre_dispatch": "sglang_jit", "source_policy": "random",
             "hidden_size": model.hidden_size, "inter_size": model.moe_intermediate_size,
             "top_k": model.moe.num_experts_per_tok, "num_experts": model.moe.num_experts,
             "num_fused_shared_experts": 0, "tp_size": 1}
hardware = {}
for sku, source_hardware in (("b200", "gb200"), ("b300", "gb300")):
    package = context.operator_package(context.device(sku).device_id, "vllm")
    # Explicit legacy-coordinate mode is the SDK's supported way to inspect
    # frozen historical data, rather than silently using current/next slots.
    database = get_database(source_hardware, "sglang", "0.5.10", systems_paths=str(SYSTEMS),
                            allow_unlisted_version=True, shared_layer=False)
    handle = EngineHandle.for_database(database, systems_path=str(SYSTEMS))
    lookup = OperatorLookup(package, LookupPolicy(extrapolate="analytical"),
                            analytical_scaler=lambda table, key: float(key["num_tokens"]))
    results = []
    for ep in (4, 8):
        rows = [row for row in package.tables["dsv4_megamoe"]
                if row["ep_size"] == ep and all(row[key] == value for key, value in selection.items())]
        requests = [{"phase": row["phase"], "num_tokens": row["num_tokens"], "measured_us": row["latency_us"]}
                    for row in rows]
        for phase in ("generation", "context"):
            sizes = sorted(row["num_tokens"] for row in rows if row["phase"] == phase)
            requests.extend({"phase": phase, "num_tokens": (a + b) // 2, "measured_us": None}
                            for a, b in zip(sizes, sizes[1:]) if (a + b) // 2 not in sizes)
        for request in requests:
            phase, tokens = request["phase"], request["num_tokens"]
            op = DeepSeekV4MegaMoEModule("routed-only", 1., model.hidden_size, model.moe_intermediate_size,
                model.moe.num_experts_per_tok, model.moe.num_experts, 1, ep,
                MoEQuantMode.w4a8_mxfp4_mxfp8, "power_law_1.01", is_context=phase == "context")
            ops_json = build_ops_json([op])
            value = handle.evaluate_ops_json(ops_json, is_context=phase == "context",
                batch_size=1 if phase == "context" else tokens,
                s=tokens if phase == "context" else 1, x=tokens)[0]
            native = lookup.lookup("dsv4_megamoe", {**selection, "ep_size": ep, "phase": phase, "num_tokens": tokens})
            # AIC returns ms. Conversion is a unit conversion, not a latency
            # fit, SKU transfer factor or serving multiplier.
            aic_us = value[1] * 1000
            difference = aic_us - native.latency_us
            assert value[3] == "silicon"
            assert abs(difference) < 1e-8 * max(1, aic_us)
            results.append({**request, "ep": ep, "aic_latency_us": aic_us,
                            "tokensim_lookup_us": native.latency_us, "difference_us": difference,
                            "tokensim_match": native.match_type, "aic_source": value[3],
                            "op_spec": json.loads(ops_json)[0]})
    source = SYSTEMS / f"data/{source_hardware}/moe/sglang/0.5.10/dsv4_megamoe_module_perf.parquet"
    frozen = ROOT / f"data/reference/agentx-native-megamoe/originals/{source_hardware}-dsv4_megamoe_module_perf.parquet"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == hashlib.sha256(frozen.read_bytes()).hexdigest()
    hardware[sku] = {"source_hardware": source_hardware, "source_framework": "sglang",
                     "requested_legacy_version": "0.5.10", "raw_data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                     "case_count": len(results), "exact_measured_cases": sum(row["measured_us"] is not None for row in results),
                     "max_abs_difference_us": max(abs(row["difference_us"]) for row in results), "cases": results}

paths = [AIC / "aic-core/src/aiconfigurator_core/_aiconfigurator_core.so",
         AIC / "aic-core/src/aiconfigurator_core/sdk/engine.py",
         AIC / "aic-core/rust/aiconfigurator-core/src/perf_database/dsv4_megamoe.rs"]
report = {"scope": "independent AIC Rust same-geometry FP8-activation/MXFP4 routed module exact/midpoint queries",
          "aic_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=AIC, text=True).strip(),
          "aic_sources": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
          "model_geometry": {"hidden": 7168, "intermediate": 3072, "experts": 384, "topk": 6},
          "runtime_change": False, "formal_model_accuracy_accepted": False, "hardware": hardware,
          "limits": ["module query directly uses SGLang/GB200/GB300 source tables; it is not a target-vLLM builder result",
                     "gate/topk, shared experts, attention, mHC and draft are outside this routed-module diagnostic",
                     "matching the same original operator rows validates import/query semantics, not independent serving accuracy",
                     "full same-model/dtype/mixed-step AIC comparison remains incomplete; source kernel revision remains unknown"]}
destination = STAGE / "aic-module-followup.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
for sku, row in hardware.items():
    print(sku, row["case_count"], "queries; measured", row["exact_measured_cases"], "max abs difference(us)", row["max_abs_difference_us"])
