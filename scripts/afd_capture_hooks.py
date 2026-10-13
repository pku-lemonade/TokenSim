"""Opt-in diagnostic hooks for an isolated plugin copy; never timing calibration.

AFD_CAPTURE_DIR is shared by A/F on one host. The A driver creates capture.enabled
after warmup. Rows are bounded; tensor copies intentionally perturb diagnostics.
"""
import contextvars
import json
import os
from pathlib import Path
import time

_layer = contextvars.ContextVar('afd_capture_layer', default=None)


def install_codecs(role):
    from afd_plugin.connectors import metadata
    name = 'encode_control_payload' if role == 'a' else 'decode_control_payload'
    original = getattr(metadata, name)
    count = 0
    def codec(payload, *args, **kwargs):
        nonlocal count
        config = _configuration()
        enabled = config is not None and (config[0]/'capture.enabled').exists() and count < config[2]+1
        start = time.perf_counter_ns() if enabled else 0
        result = original(payload, *args, **kwargs)
        end = time.perf_counter_ns() if enabled else 0
        if enabled:
            count += 1
            # Function interval is diagnostic until tensor readback in encode
            # has been excluded. Do not import this interval as host service.
            _write(config[0], role, dict(kind='codec_interval', operation=name,
                duration_us=(end-start)/1000, contains_wait='unknown',
                source='native codec wrapper; encode may convert tensor scalars'))
        return result
    setattr(metadata, name, codec)


def _configuration():
    root = os.environ.get('AFD_CAPTURE_DIR')
    if not root:
        return None
    return Path(root), int(os.environ['AFD_CAPTURE_BATCH']), int(os.environ.get('AFD_CAPTURE_STEPS', '31'))


def _write(root, role, record):
    record.update(pid=os.getpid(), clock='perf_counter_ns_same_host', cpu_ns=time.perf_counter_ns(),
                  timing_scope='diagnostic_inclusive_not_service')
    with (root / f'capture-{role}-{os.getpid()}.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')


def install_attention(cls):
    if _configuration() is None:
        return
    install_codecs('a')
    original = cls._prepare_inputs
    count = 0
    def prepare(self, *args, **kwargs):
        nonlocal count
        result = original(self, *args, **kwargs)
        root, batch, limit = _configuration()
        if count >= limit or not (root/'capture.enabled').exists():
            return result
        input_batch = self.input_batch
        requests = list(input_batch.req_ids[:input_batch.num_reqs])
        scheduler = args[0] if args else kwargs['scheduler_output']
        scheduled = scheduler.num_scheduled_tokens
        if len(requests) != batch or any(scheduled.get(request) != 1 for request in requests):
            return result
        rows = [dict(row=i, native_request_id=request,
                     token_id=int(input_batch.num_computed_tokens_cpu[i]))
                for i, request in enumerate(requests)]
        _write(root, 'a', dict(kind='row_identity', decode_step=count, rows=rows))
        count += 1
        return result
    cls._prepare_inputs = prepare


def install_ffn(cls, router_cls):
    if _configuration() is None:
        return
    install_codecs('f')
    original = cls._execute_eager_mode
    select = router_cls.select_experts
    count = -1
    def execute(self, hidden_states, layer_idx, *args, **kwargs):
        nonlocal count
        root, batch, limit = _configuration()
        enabled = (root/'capture.enabled').exists() and int(hidden_states.shape[0]) == batch
        if enabled and layer_idx == 0:
            count += 1
        active = enabled and 0 <= count < limit
        token = _layer.set((root, count, int(layer_idx)) if active else None)
        try:
            result = original(self, hidden_states, layer_idx, *args, **kwargs)
            if active:
                import torch
                _write(root, 'f', dict(kind='layer_buffers', decode_step=count, layer_id=int(layer_idx),
                    input_shape=list(hidden_states.shape), output_shape=list(result.shape),
                    input_bytes=hidden_states.numel()*hidden_states.element_size(),
                    output_bytes=result.numel()*result.element_size(),
                    allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    memory_scope='process allocator snapshot; peak includes startup, not unique tensor sum'))
            return result
        finally:
            _layer.reset(token)
    def route(self, *args, **kwargs):
        result = select(self, *args, **kwargs)
        context = _layer.get()
        if context is not None:
            root, step, layer = context
            weights, ids = result
            _write(root, 'f', dict(kind='routing', decode_step=step, layer_id=layer,
                expert_ids=ids.detach().cpu().tolist(), routing_weights=weights.detach().cpu().tolist()))
        return result
    cls._execute_eager_mode = execute
    router_cls.select_experts = route
