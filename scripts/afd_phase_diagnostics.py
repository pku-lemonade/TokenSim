"""Native wall intervals only: no synchronize, fitting, or business changes."""
import functools
import inspect
import time

active = False
rows = []
sources = {}


def wrap(owner, name, label):
    original = getattr(owner,name)
    source = inspect.unwrap(original)
    sources[label] = dict(file=inspect.getsourcefile(source),line=inspect.getsourcelines(source)[1])
    @functools.wraps(original)
    def measured(*args,**kwargs):
        if not active:
            return original(*args,**kwargs)
        start=time.perf_counter_ns()
        try:
            return original(*args,**kwargs)
        finally:
            rows.append(dict(name=label,start_ns=start,end_ns=time.perf_counter_ns()))
    setattr(owner,name,measured)


def install():
    import torch.distributed as dist
    from afd_plugin.v1.worker.attention_model_runner import AFDAttentionModelRunner
    from afd_plugin.connectors.gpu import p2p
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
    for name in ('execute_model','_prepare_inputs','_build_attention_metadata','_model_forward','sample_tokens'):
        wrap(AFDAttentionModelRunner,name,name)
    wrap(p2p,'send_control_payload','control_payload')
    wrap(dist,'send','torch_send')
    wrap(dist,'isend','torch_isend')
    wrap(Qwen3MoeDecoderLayer,'forward','decoder_layer')
