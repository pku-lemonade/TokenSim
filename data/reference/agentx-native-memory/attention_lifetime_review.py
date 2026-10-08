"""Execute pinned attention/Indexer Python bodies with shape-only kernel stubs.

The probe validates Python ownership and storage alias lifetimes. CUDA streams,
allocator reservation, kernel internal scratch and real numerical work are
outside its scope. It never modifies runtime or existing historical evidence.
"""

import ast
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[2]
sys.path.insert(0, str(ROOT))
from TokenSim.config.deepseek_v4_buffers import live_phases
from TokenSim.config.parallel_config import ParallelConfig
from TokenSim.hardware.context import HardwareContext

SOURCE = Path("/home/zhuohang/Serving/vllm-note/vllm/models/deepseek_v4")
ATTN = SOURCE / "attention.py"
OUTPUT = SOURCE / "nvidia/ops/o_proj.py"
PROBE = None
ITEMS = {"bf16": 2, "fp8": 1, "fp32": 4, "int32": 4, "uint8": 1, "int64": 8}


class Storage:
    def __init__(self, shape, dtype, name, persistent=False):
        self.probe = PROBE
        self.identity = next(PROBE.ids)
        self.tracked = not persistent
        if not self.tracked:
            return
        self.entry = {"id": self.identity, "name": name, "shape": list(shape),
                      "dtype": dtype, "bytes": math.prod(shape) * ITEMS[dtype]}
        PROBE.live[self.identity] = self.entry
        PROBE.allocations.append(self.entry)
        total = sum(row["bytes"] for row in PROBE.live.values())
        if total > PROBE.maximum:
            PROBE.maximum = total
            PROBE.peak = [dict(row) for row in PROBE.live.values()]

    def __del__(self):
        if self.tracked:
            self.probe.live.pop(self.identity)


class Tensor:
    def __init__(self, shape, dtype="bf16", name="kernel_output", *, storage=None, persistent=False):
        self.shape, self.dtype, self.device = tuple(shape), dtype, "stub"
        self.storage = storage if storage is not None else Storage(shape, dtype, name, persistent)

    @property
    def data(self):
        return self

    @property
    def T(self):
        return Tensor(tuple(reversed(self.shape)), self.dtype, storage=self.storage)

    def view(self, *shape):
        shape = tuple(math.prod(self.shape) // math.prod(s for s in shape if s != -1)
                      if s == -1 else s for s in shape)
        assert math.prod(shape) == math.prod(self.shape)
        return Tensor(shape, self.dtype, storage=self.storage)

    def flatten(self, start):
        return self.view(*self.shape[:start], math.prod(self.shape[start:]))

    def split(self, sizes, dim):
        result = []
        for size in sizes:
            shape = list(self.shape)
            shape[dim] = size
            result.append(Tensor(shape, self.dtype, storage=self.storage))
        assert sum(sizes) == self.shape[dim]
        return result

    def __getitem__(self, slices):
        if not isinstance(slices, tuple):
            slices = (slices,)
        shape = list(self.shape)
        for axis, part in enumerate(slices):
            if isinstance(part, slice):
                shape[axis] = len(range(*part.indices(shape[axis])))
        return Tensor(shape, self.dtype, storage=self.storage)


class Torch:
    Tensor = Tensor
    bfloat16, float8_e4m3fn, float32, int32 = "bf16", "fp8", "fp32", "int32"
    cuda = SimpleNamespace(is_current_stream_capturing=lambda: True)

    @staticmethod
    def empty(shape, *, dtype="bf16", device=None):
        return Tensor(shape, dtype, "torch.empty")

    @staticmethod
    def mm(x, weight, *, out_dtype):
        return Tensor((x.shape[0], weight.shape[1]), out_dtype, "compressor_mm")


def linear(output, name):
    def apply(x, *args):
        return Tensor((x.shape[0], output), "bf16", name)
    return apply


def execute_in_parallel(main, functions, *args, **kwargs):
    # One permitted sequential ordering; physical parallel co-liveness is not
    # established by this Python-only test. Return values still retain owners.
    return main(), [function() if function is not None else None for function in functions]


def maybe_execute_in_parallel(main, other, *args, **kwargs):
    return main(), other()


def qkv_norm(q, kv, *args):
    return Tensor(q.shape, name="normalized_q_lora"), Tensor(kv.shape, name="normalized_kv")


def quant_index(positions, q, rope, weights, *args, **kwargs):
    rows, heads, width = q.shape
    return (Tensor((rows, heads, width // 2), "uint8", "indexer_packed_q"),
            Tensor((rows, heads, width // 32), "uint8", "indexer_q_scales")), \
        Tensor(weights.shape, "fp32", "indexer_output_weights")


def output_quant(o, *args, n_groups, quant_group_size, **kwargs):
    rows, heads, width = o.shape
    return Tensor((rows, n_groups, heads * width // n_groups), "fp8", "inverse_rope_values"), \
        Tensor((rows, heads), "int32", "inverse_rope_scales")


class QuantizedActivation:
    def __init__(self, data, scale):
        self.data, self.scale = data, scale


def noop(*args, **kwargs):
    return None


def source_body(path, cls, name, exported):
    tree = ast.parse(path.read_text())
    parent = tree if cls is None else next(node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == cls)
    function = next(node for node in parent.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.name, function.decorator_list = exported, []
    return function


functions = [source_body(ATTN, "DeepseekV4Attention", name, name) for name in
             ("forward", "_split_qkv_and_norm", "_prepare_and_attn", "_run_parallel_input_projections",
              "_sparse_indexer_and_attn")]
functions.append(source_body(ATTN, "DeepseekV4Indexer", "forward", "indexer_forward"))
functions.append(source_body(OUTPUT, None, "deep_gemm_fp8_o_proj", "deep_gemm_fp8_o_proj"))
namespace = {
    "torch": Torch, "QuantizedActivation": QuantizedActivation,
    "execute_in_parallel": execute_in_parallel, "maybe_execute_in_parallel": maybe_execute_in_parallel,
    "fused_q_kv_rmsnorm": qkv_norm, "fused_indexer_q_rope_quant": quant_index,
    "fused_inv_rope_fp8_quant": output_quant, "fp8_einsum": noop,
    "get_forward_context": lambda: SimpleNamespace(attn_metadata=None),
    "envs": SimpleNamespace(VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=256),
}
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                         *functions], type_ignores=[])
ast.fix_missing_locations(module)
exec(compile(module, "<pinned-attention-source-probe>", "exec"), namespace)


def bind(owner, name):
    setattr(owner, name, lambda *args, **kwargs: namespace[name](owner, *args, **kwargs))


def compressor(width):
    class Compressor:
        fused_wkv_wgate = SimpleNamespace(weight=Tensor((width, 7168), persistent=True))

        def __call__(self, *args):
            return None
    return Compressor()


def owner(rows, tp, kind):
    heads, groups = 128 // tp, 16 // tp
    indexer = None
    if kind == "csa":
        class Indexer:
            def __call__(self, *args):
                return namespace["indexer_forward"](self, *args)
        indexer = Indexer()
        indexer.compressor = compressor(512)
        indexer.weights_proj = lambda x: (Tensor((rows, 64), name="indexer_input_weights"), None)
        indexer._wq_b_proj = linear(64 * 128, "indexer_bf16_q")
        indexer.n_head, indexer.head_dim, indexer.softmax_scale = 64, 128, 128**-.5
        indexer.use_fp4_kv, indexer.ln_events, indexer.aux_stream = True, [None, None], None
        indexer.indexer_op = noop
    result = SimpleNamespace(
        n_local_heads=heads, head_dim=512, n_local_groups=groups,
        q_lora_rank=1536, q_norm=SimpleNamespace(weight=Tensor((1536,), persistent=True)),
        kv_norm=SimpleNamespace(weight=Tensor((512,), persistent=True)), eps=1e-6,
        indexer=indexer, compressor=compressor(2048 if kind == "csa" else 1024) if kind != "swa" else None,
        aux_stream_list=[None, None, None], ln_events=[None] * 4,
        rotary_emb=SimpleNamespace(cos_sin_cache=None), indexer_rotary_emb=SimpleNamespace(cos_sin_cache=None),
        _fused_wqa_wkv_gemm=linear(2048, "raw_qkv"),
        _wq_b_proj=linear(heads * 512, "bf16_query"),
        _fused_qnorm_rope_kv_insert=lambda q, *args: Tensor(q.shape, "fp8", "fp8_query"),
        forward_mqa=noop,
    )
    if tp == 1:
        result._alloc_attn_out = lambda *args: QuantizedActivation(
            Tensor((rows, groups, 8 * 512), "fp8", "fused_fp8_output"),
            Tensor((rows, groups, 8), "int32", "fused_output_scales"))
    else:
        result._alloc_attn_out = lambda *args: Tensor((rows, heads, 512), name="bf16_attention_output")
    weight = Tensor((groups * 1024, 8 * 512), "fp8", persistent=True)
    wo_a = SimpleNamespace(weight=weight, weight_scale=None)
    result._o_proj = lambda output, positions: namespace["deep_gemm_fp8_o_proj"](
        output, positions, None, wo_a, linear(7168, "output_hidden"), n_groups=groups,
        heads_per_group=8, nope_dim=448, rope_dim=64, o_lora_rank=1024,
        einsum_recipe=(1, 1, 128), tma_aligned_scales=True)
    for method in ("_split_qkv_and_norm", "_prepare_and_attn", "_run_parallel_input_projections", "_sparse_indexer_and_attn"):
        bind(result, method)
    result._prepare_and_attn_fn = result._prepare_and_attn
    return result


cases = []
model = HardwareContext.load(ROOT / "data").model("DeepSeek-V4-Pro")
for rows in (20, 196, 384, 8192, 16384):
    for tp in (1, 4, 8):
        for kind in ("csa", "hca", "swa"):
            PROBE = SimpleNamespace(ids=itertools.count(), live={}, allocations=[], maximum=0, peak=[])
            attention = owner(rows, tp, kind)
            inputs = Tensor((rows, 7168), persistent=True)
            positions = Tensor((rows,), "int64", persistent=True)
            result = namespace["forward"](attention, positions, inputs)
            assert result.shape == (rows, 7168)
            assert sum(row["bytes"] for row in PROBE.live.values()) == rows * 7168 * 2
            del result, attention, inputs, positions
            gc.collect()
            assert not PROBE.live
            parallel = ParallelConfig(tensor_parallel_size=tp, data_parallel_size=8 // tp,
                                      data_parallel_size_local=8 // tp, enable_expert_parallel=True)
            bound = live_phases(model, parallel, rows, 49, kind=kind, draft=kind == "swa")
            assert PROBE.maximum <= bound["peak_bytes"]
            cases.append({"rows": rows, "tp": tp, "kind": kind,
                          "python_shape_stub_peak_bytes": PROBE.maximum, "peak_owners": PROBE.peak,
                          "allocations": PROBE.allocations, "output_is_only_retained_transient": True,
                          "all_transient_storages_released": True,
                          "runtime_full_live_phase_bound_bytes": bound["peak_bytes"],
                          "partial_attention_peak_is_within_full_bound": True})

report = {
    "scope": "pinned attention/Indexer/output-projection Python body shape-stub ownership, independent of runtime live-phase interval code",
    "runtime_change": False, "formal_accuracy_evidence": False,
    "source_checks": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in (ATTN, OUTPUT, ROOT / "TokenSim/config/deepseek_v4_buffers.py")],
    "cases": cases,
    "limits": [
        "sequential ordering of parallel callbacks; simultaneous GPU branch peak is not observed",
        "normalization/quant/linear numeric kernels are shape stubs; internal quant staging and scratch excluded",
        "sparse attention/indexer numeric kernels, context logits and persistent workspaces are not allocated in this probe",
        "hidden input is externally owned; mHC caller residual/hidden and graph/allocator reservation are not included",
        "does not measure kernel timing, absolute GPU live/reserved bytes or reference serving memory",
        "full-bound comparison includes mHC/model/FFN budget absent from this partial probe; passing does not prove completeness or allocator upper bound",
    ],
}
destination = STAGE / "attention-lifetime-probe.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
print(len(cases), "source-body shape/alias cases; output-only retention and complete release verified")
