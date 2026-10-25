"""Temporary semantic scopes and profiler configuration inside vLLM workers."""

from functools import wraps
import os
from pathlib import Path
import re
import socket


def module_annotation(name):
    boundaries = {"model.norm": "final_norm", "model.embed_tokens": "embedding_lookup"}
    if name in boundaries:
        return "tokensim.model." + boundaries[name]
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.(.*))?$", name)
    if not match:
        return "tokensim.forward" if name == "" else None
    layer, suffix = match.groups()
    operations = {
        "self_attn.qkv_proj": "qkv",
        "self_attn.q_proj": "q_proj",
        "self_attn.k_proj": "k_proj",
        "self_attn.v_proj": "v_proj",
        "self_attn.o_proj": "output",
        "self_attn.q_norm": "q_norm",
        "self_attn.k_norm": "k_norm",
        "self_attn.attn": "attention",
        "input_layernorm": "attention_norm",
        "post_attention_layernorm": "ffn_norm",
        "mlp.gate_up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
        "mlp.act_fn": "activation",
        "mlp.gate": "router",
        "mlp.experts": "moe_module",
    }
    if suffix is None:
        return f"tokensim.layer.{layer}"
    operation = operations.get(suffix)
    return f"tokensim.layer.{layer}.{operation}" if operation else None


class AnnotationSession:
    """Restore every hook and method after the structure-only capture."""

    def __init__(self):
        self.handles = []
        self.methods = []
        self.labels = []

    def module(self, module, label):
        import torch

        contexts = []

        def before(module, args):
            context = torch.profiler.record_function(label)
            context.__enter__()
            contexts.append(context)

        def after(module, args, result):
            if contexts:
                contexts.pop().__exit__(None, None, None)

        self.handles.append(module.register_forward_pre_hook(before))
        self.handles.append(module.register_forward_hook(after, always_call=True))
        self.labels.append(label)

    def method(self, owner, name, label):
        import torch

        original = getattr(owner, name, None)
        if original is None:
            return
        owned = name in vars(owner)

        @wraps(original)
        def observed(*args, **kwargs):
            with torch.profiler.record_function(label):
                return original(*args, **kwargs)

        self.methods.append((owner, name, original, owned))
        setattr(owner, name, observed)
        self.labels.append(label)

    def close(self):
        for handle in reversed(self.handles):
            handle.remove()
        for owner, name, original, owned in reversed(self.methods):
            if owned:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.handles.clear()
        self.methods.clear()


def annotate_model(model, runner=None):
    session = AnnotationSession()
    try:
        for name, module in model.named_modules():
            label = module_annotation(name)
            if label:
                session.module(module, label)
        for name in ("compute_logits", "compute_logits_local"):
            session.method(model, name, "tokensim.model.lm_head")
        sampler = getattr(runner, "sampler", None)
        if sampler is not None:
            if hasattr(sampler, "register_forward_hook"):
                session.module(sampler, "tokensim.model.token_selection")
            else:
                session.method(sampler, "sample", "tokensim.model.token_selection")
        return session
    except BaseException:
        session.close()
        raise


class ProfilerWorkerExtension:
    def tokensim_annotations(self, enabled):
        previous = getattr(self, "_tokensim_annotations", None)
        if previous is not None:
            previous.close()
            self._tokensim_annotations = None
        if enabled:
            self._tokensim_annotations = annotate_model(
                self.get_model(), self.model_runner
            )
            return self._tokensim_annotations.labels
        return []

    def tokensim_configure_profiler(self, directory, structure, record_shapes=False):
        profiler = getattr(self, "profiler", None)
        if profiler is not None:
            if profiler.is_running:
                raise RuntimeError("cannot reconfigure a running profiler")
            profiler.shutdown()
        self.profiler = None
        self.profiler_config.torch_profiler_dir = directory
        self.profiler_config.torch_profiler_record_shapes = record_shapes
        self.profiler_config.torch_profiler_activities = (
            ["CPU", "CUDA"] if structure else ["CUDA"]
        )
        return self.tokensim_identity()

    def tokensim_identity(self):
        import torch

        torch.cuda.synchronize()

        def read(path):
            file = Path(path)
            return file.read_text().strip() if file.exists() else None

        return dict(
            rank=self.rank,
            hostname=socket.gethostname(),
            boot_id=read("/proc/sys/kernel/random/boot_id"),
            time_namespace=(
                os.readlink("/proc/self/ns/time")
                if Path("/proc/self/ns/time").exists()
                else None
            ),
            cpu_affinity=(
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            annotations_active=getattr(self, "_tokensim_annotations", None) is not None,
        )
