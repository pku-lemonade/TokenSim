"""Probe Python storage identities in pinned model/layer/mHC bodies without CUDA.

Only the source assignment/lifetime topology is executed. Kernels, attention,
FFN and model entry are stubs: this is not an allocator or peak-byte profile.
"""

import ast
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace


STAGE = Path(__file__).resolve().parent
SOURCE = Path("/home/zhuohang/Serving/vllm-note/vllm")
MODEL = SOURCE / "models/deepseek_v4/nvidia/model.py"
MHC = SOURCE / "model_executor/kernels/mhc/tilelang.py"
REMOVED_IMPORTS = []


class ImportResolver(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        REMOVED_IMPORTS.append(ast.unparse(node))
        return ast.copy_location(ast.Pass(), node)


def source_function(path, function_name, *, class_name=None, exported_name=None):
    tree = ast.parse(path.read_text())
    parent = tree if class_name is None else next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    function = next(node for node in parent.body
                    if isinstance(node, ast.FunctionDef) and node.name == function_name)
    function.name = exported_name or function_name
    function.decorator_list = []
    return ImportResolver().visit(function)


PROBE = None


class Storage:
    def __init__(self, shape, dtype, *, parameter=False):
        self.probe = PROBE
        self.identity = next(self.probe.identities)
        self.hyper = not parameter and dtype == "bf16" and tuple(shape) == self.probe.hyper_shape
        if self.hyper:
            self.probe.live.add(self.identity)
            self.probe.allocations += 1
            self.probe.maximum = max(self.probe.maximum, len(self.probe.live))

    def __del__(self):
        if self.hyper:
            self.probe.live.remove(self.identity)


class Tensor:
    def __init__(self, shape, dtype="bf16", *, storage=None, parameter=False):
        self.shape, self.dtype, self.device = tuple(shape), dtype, "stub"
        self.storage = storage if storage is not None else Storage(shape, dtype, parameter=parameter)

    @property
    def data(self):
        return self

    def view(self, *shape):
        if -1 in shape:
            known = math.prod(size for size in shape if size != -1)
            shape = tuple(math.prod(self.shape) // known if size == -1 else size for size in shape)
        assert math.prod(shape) == math.prod(self.shape)
        return Tensor(shape, self.dtype, storage=self.storage)

    def squeeze(self, dimension):
        dimension %= len(self.shape)
        shape = tuple(size for index, size in enumerate(self.shape)
                      if index != dimension or size != 1)
        return Tensor(shape, self.dtype, storage=self.storage)

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return True

    def to(self, dtype):
        return self if dtype == self.dtype else Tensor(self.shape, dtype)


class Torch:
    bfloat16, float32, int64 = "bf16", "fp32", "int64"

    @staticmethod
    def empty(*shape, dtype="bf16", device=None):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Tensor(shape, dtype)

    @staticmethod
    def empty_like(tensor, dtype=None):
        return Tensor(tensor.shape, dtype or tensor.dtype)


def unused_kernel(*args, **kwargs):
    return None


def fused_kernel(comb, residual, post, x, fn, mult, hidden, mix_size):
    rows = residual.shape[0]
    # Kernel scratch shapes are placeholders; only distinct residual identity
    # is under audit, independent of the internal split workspace size.
    return Tensor((1, rows, mix_size), "fp32"), Tensor((1, rows), "fp32"), Torch.empty_like(residual)


def broadcast(x, fn, scale, base, *args, **kwargs):
    rows, hidden = x.shape
    return (Tensor((rows, 4, hidden)), Tensor((rows, 4, 1), "fp32"),
            Tensor((rows, 4, 4), "fp32"), Tensor((rows, hidden)))


functions = [source_function(MHC, name) for name in
             ("_hc_prenorm_gemm_outputs", "mhc_fused_post_pre_tilelang",
              "mhc_post_tilelang", "hc_head_fused_kernel_tilelang")]
functions += [source_function(MODEL, "forward", class_name="DeepseekV4DecoderLayer", exported_name="layer_forward"),
              source_function(MODEL, "forward", class_name="DeepseekV4Model", exported_name="model_forward")]
namespace = {
    "torch": Torch, "is_deep_gemm_supported": lambda: False,
    "compute_num_split": lambda *args: 1, "cdiv": lambda a, b: (a + b - 1) // b,
    "_HC_PRENORM_GEMM_TILELANG_KERNEL": unused_kernel,
    "_MHC_POST_TILELANG_KERNEL": unused_kernel,
    "_MHC_PRE_BIG_FUSE_TILELANG_KERNEL": unused_kernel,
    "_HC_HEAD_FUSED_TILELANG_KERNEL": unused_kernel,
    "_MHC_FUSED_TILELANG_KERNEL": fused_kernel,
    "mhc_pre_broadcast_tilelang": broadcast,
    "get_pp_group": lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    "prepare_mega_gate_routing_metadata": lambda *args, **kwargs: None,
    "IMAGE_SENTINEL_BASE_ID": 0, "islice": itertools.islice,
}
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                         *functions], type_ignores=[])
ast.fix_missing_locations(module)
exec(compile(module, "<pinned-source-storage-probe>", "exec"), namespace)


def layer(rows, fused):
    hidden, mult, mix = 7168, 4, 24
    norm = SimpleNamespace(weight=Tensor((hidden,), parameter=True), variance_epsilon=1e-6)
    owner = SimpleNamespace(
        attn_norm=norm, ffn_norm=norm, use_sequence_parallel=False,
        hc_attn_fn=Tensor((mix, mult * hidden), "fp32", parameter=True),
        hc_ffn_fn=Tensor((mix, mult * hidden), "fp32", parameter=True),
        hc_attn_fn_broadcast=Tensor((mix, hidden), "fp32", parameter=True),
        hc_attn_scale=Tensor((3,), "fp32", parameter=True),
        hc_ffn_scale=Tensor((3,), "fp32", parameter=True),
        hc_attn_base=Tensor((mix,), "fp32", parameter=True),
        hc_ffn_base=Tensor((mix,), "fp32", parameter=True),
        rms_norm_eps=1e-6, hc_eps=1e-6, hc_post_alpha=2.0, hc_sinkhorn_iters=20,
        attn=lambda positions, x, _: Tensor((rows, hidden)),
        ffn=lambda x, ids, meta: Tensor((rows, hidden)),
    )
    return lambda *args, **kwargs: namespace["layer_forward"](owner, *args, **kwargs)


cases = []
for rows in (196, 384, 8192, 16384):
    for num_layers in (3, 61):
        for fused in (False, True):
            PROBE = SimpleNamespace(identities=itertools.count(), live=set(), maximum=0,
                                    allocations=0, hyper_shape=(rows, 4, 7168))
            namespace["mhc_fused_post_pre_split_config"] = lambda *args: (1, 1, 128) if fused else None
            model = SimpleNamespace(
                embed_input_ids=lambda ids: Tensor((rows, 7168)), use_mega_moe=True,
                use_sequence_parallel=False, has_local_hash_moe=True,
                config=SimpleNamespace(vision_n_layers=0),
                collect_remote_aux_hidden_states=lambda _: [],
                start_layer=0, end_layer=num_layers, aux_hidden_state_layers=[],
                layers=[layer(rows, fused) for _ in range(num_layers)],
                _mtp_hidden_buffer=None,
                hc_head_fn=Tensor((4, 28672), "fp32", parameter=True),
                hc_head_scale=Tensor((1,), "fp32", parameter=True),
                hc_head_base=Tensor((4,), "fp32", parameter=True),
                rms_norm_eps=1e-6, hc_eps=1e-6, norm=lambda x: Tensor((rows, 7168)),
            )
            result = namespace["model_forward"](model, Tensor((rows,), "int64"),
                                                Tensor((rows,), "int64"), None)
            assert result.shape == (rows, 7168)
            del result, model
            gc.collect()
            assert PROBE.maximum == 3 and not PROBE.live
            cases.append({"rows": rows, "num_layers": num_layers, "fused_kernel_stub": fused,
                          "maximum_distinct_hyper_residual_storages": PROBE.maximum,
                          "hyper_residual_allocations": PROBE.allocations,
                          "reference_release_complete": True})

report = {
    "scope": "actual pinned model/layer/mHC Python bodies with import resolution and numerical kernels stubbed",
    "runtime_change": False, "formal_accuracy_evidence": False,
    "source_checks": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in (MODEL, MHC)],
    "resolved_imports": sorted(set(REMOVED_IMPORTS)), "cases": cases,
    "findings": [
        "outer model assignment retains prior residual while the layer RHS executes",
        "second mHC transition can simultaneously own caller prior, attention residual and new FFN residual: three distinct storages",
        "three/61 layers have the same maximum residual multiplicity; do not multiply this residual maximum by layer count",
    ],
    "limits": [
        "this is reference-topology evidence, not an eager/capture GPU allocator measurement or additive byte peak",
        "kernel scratch, attention, FFN, streams, sampler, metadata, SP and compiler optimizations are outside this probe",
        "persistent MTP hidden buffer is excluded from transient probe and must be counted separately",
        "fused branch scratch shapes are stubs; only residual identity/multiplicity is claimed",
    ],
}
destination = STAGE / "memory-lifetime-probe.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
print(len(cases), "source-body cases; maximum residual multiplicity=3; releases complete")
