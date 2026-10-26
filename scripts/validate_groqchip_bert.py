"""Price one BERT-base encoder layer on GroqChip v1 and compare with ASAP 2022.

Ahmed et al., "Answer Fast: Accelerating BERT on the Tensor Streaming
Processor" (ASAP 2022), measure a mean of 128.9 microseconds for BERT-base at
batch 1 and sequence length 128, of which 83% (about 107 microseconds, or 8.9
per layer) is the 12-layer encoder.  This script prices one encoder layer with
TokenSim's generic Groq model and with the LP30 unit method under GroqChip v1
rates.  See docs/groqchip-v1-bert-validation.md for the recorded results.
"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.hardware.context import HardwareContext
from TokenSim.operator_data.analytical.base import analytical_model_for
from TokenSim.operator_data.analytical.lp30 import estimate_lp30
from scripts.search_lpx_native import write_json

TOKENS, HIDDEN, INTERMEDIATE, HEADS, HEAD_DIM, LAYERS = 128, 768, 3072, 12, 64, 12
MEASURED_ENCODER_US = 0.83 * 128.9
# GroqChip v1: 320 lanes of 16 VXM ALUs at 900 MHz, and the 20 TB/s operand
# bandwidth into the MXM and VXM stated by the BERT paper.
V1_VECTOR_OPS_PER_S = 320 * 16 * 0.9e9
V1_OPERAND_BYTES_PER_S = 20e12
# The LP30 study charges exponentials at 1/20 of the vector rate.
EXP_FRACTION = 1 / 20
CASES = (
    ("generic Groq model, device defaults", None),
    ("LP30 method, study efficiencies", dict(gemm_mfu=0.3, attention_mfu=0.1)),
    ("LP30 method, attention efficiency 0.5", dict(gemm_mfu=0.3, attention_mfu=0.5)),
    ("LP30 method, device-default efficiencies", dict(gemm_mfu=0.8, attention_mfu=0.6)),
)


def operators(matrix_dtype, activation_dtype):
    gemm = lambda n, k: dict(m=TOKENS, n=n, k=k, dtype=matrix_dtype)
    vector = lambda op, width: dict(
        op_name=op, num_tokens=TOKENS, hidden_size=width, dtype=activation_dtype
    )
    attention = dict(
        batch_size=1,
        input_seq_len=TOKENS,
        num_heads=HEADS,
        num_kv_heads=HEADS,
        head_dim=HEAD_DIM,
        attn_dtype=matrix_dtype,
        kv_cache_dtype=matrix_dtype,
    )
    return (
        ("qkv_gemm", "gemm", gemm(3 * HIDDEN, HIDDEN)),
        ("attention_core", "context_attention", attention),
        ("output_gemm", "gemm", gemm(HIDDEN, HIDDEN)),
        ("layernorm_1", "elementwise", vector("layernorm", HIDDEN)),
        ("ffn_up_gemm", "gemm", gemm(INTERMEDIATE, HIDDEN)),
        ("gelu", "elementwise", vector("gelu", INTERMEDIATE)),
        ("ffn_down_gemm", "gemm", gemm(HIDDEN, INTERMEDIATE)),
        ("layernorm_2", "elementwise", vector("layernorm", HIDDEN)),
    )


def estimator(device, efficiencies):
    if efficiencies is None:
        model = analytical_model_for(device.family)
        return (lambda table, key: model.estimate(table, key, device)), "fp16"
    pipeline = dict(
        activation_dtype="fp32",
        vector_ops_per_s=V1_VECTOR_OPS_PER_S,
        exp_ops_per_s=V1_VECTOR_OPS_PER_S * EXP_FRACTION,
        shuffle_bytes_per_s=V1_OPERAND_BYTES_PER_S,
        vector_efficiency=0.5,
        shuffle_efficiency=0.8,
        vector_bytes=320,
        attention_tile_keys=256,
        fill_drain_us=0.05,
        schedule="tiled",
    )
    analytical = {
        **device.analytical,
        **efficiencies,
        "lpx_pipeline": pipeline,
        "memory_efficiency": 0.9,
        "kernel_launch_us": 0.0,
    }
    device = replace(device, analytical=analytical)
    return (lambda table, key: estimate_lp30(table, key, device)), "fp32"


def run_case(device, name, efficiencies):
    estimate, activation_dtype = estimator(device, efficiencies)
    rows = []
    for operator, table, key in operators("int8", activation_dtype):
        result = estimate(table, key)
        rows.append(
            dict(
                operator=operator,
                table=table,
                latency_us=result.latency_us,
                bottleneck=getattr(result, "bottleneck", None),
            )
        )
    serialized = sum(row["latency_us"] for row in rows)
    matrix = sum(row["latency_us"] for row in rows if row["table"] != "elementwise")
    return dict(
        case=name,
        efficiencies=efficiencies,
        operators=rows,
        layer_serialized_us=serialized,
        layer_matrix_only_us=matrix,
        encoder_serialized_us=LAYERS * serialized,
        encoder_matrix_only_us=LAYERS * matrix,
        matrix_only_error=LAYERS * matrix / MEASURED_ENCODER_US - 1,
        serialized_error=LAYERS * serialized / MEASURED_ENCODER_US - 1,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    device = HardwareContext.load(ROOT / "data").device("groqchip_v1")
    results = [run_case(device, name, efficiencies) for name, efficiencies in CASES]
    for result in results:
        print(
            f"{result['case']}: layer {result['layer_matrix_only_us']:.2f} us matrix-only,"
            f" {result['layer_serialized_us']:.2f} us serialized; encoder error"
            f" {result['matrix_only_error']:+.1%} / {result['serialized_error']:+.1%}"
        )
    if args.summary:
        write_json(
            args.summary,
            dict(
                measured_encoder_us=MEASURED_ENCODER_US,
                measured_total_us=128.9,
                v1_vector_ops_per_s=V1_VECTOR_OPS_PER_S,
                v1_operand_bytes_per_s=V1_OPERAND_BYTES_PER_S,
                exp_fraction=EXP_FRACTION,
                cases=results,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
