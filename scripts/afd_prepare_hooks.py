"""Ready-input operation measurement in an isolated native runner copy."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time


def install(cls):
    if os.environ.get('AFD_PREPARE_MODE') == 'compute_cpu':
        from afd_compute_cpu_hooks import install as install_compute
        install_compute(cls,'a')
        return
    if os.environ.get('AFD_PREPARE_MODE') == 'bookkeeping':
        install_bookkeeping(cls)
        return
    if os.environ.get('AFD_PREPARE_MODE') == 'sampled_tokens':
        install_sampled_tokens(cls)
        return
    if os.environ.get('AFD_PREPARE_MODE') == 'metadata':
        install_metadata(cls, '_build_attention_metadata', 'a_attention_metadata', 'a', True)
        return
    original = cls._prepare_inputs
    source = inspect.getsource(original)
    rows = []

    def prepare(self, scheduler_output, num_scheduled_tokens):
        root = Path(os.environ['AFD_PREPARE_DIR'])
        batch = int(os.environ['AFD_PREPARE_BATCH'])
        active = ((root/'measure.enabled').exists() and len(rows) < 200
                  and self.input_batch.num_reqs == batch
                  and scheduler_output.total_num_scheduled_tokens == batch
                  and all(int(n) == 1 for n in num_scheduled_tokens))
        if not active:
            return original(self, scheduler_output, num_scheduled_tokens)
        import torch
        context = int(self.input_batch.num_computed_tokens_cpu[0]) + 1
        before_sync = time.perf_counter_ns()
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        result = original(self, scheduler_output, num_scheduled_tokens)
        returned = time.perf_counter_ns()
        torch.cuda.synchronize()
        finish = time.perf_counter_ns()
        rows.append(dict(context=context, service_us=(finish-start)/1000,
                         api_us=(returned-start)/1000, completion_tail_us=(finish-returned)/1000,
                         excluded_prior_sync_us=(start-before_sync)/1000))
        if len(rows) == 200:
            with (root/'prepare-samples.json').open('x') as stream:
                json.dump(dict(operation='a_input_prepare', batch=batch,
                    window=int(os.environ['AFD_PREPARE_WINDOW']), samples=rows,
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    source_file=inspect.getsourcefile(original),
                    source_line=inspect.getsourcelines(original)[1],
                    timing_scope='operation_service', external_wait_excluded=True,
                    completion='device_synchronize_after_operation',
                    prior_work_synchronize_outside_timer=True, profiler_enabled=False,
                    gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
                    limitation='isolated completion timing includes explicit completion sync; not native step timing'), stream)
        return result

    cls._prepare_inputs = prepare


def install_metadata(cls, method, operation, role, device):
    original = getattr(cls,method)
    source = inspect.getsource(original)
    rows = []
    def measured(self,*args,**kwargs):
        root = Path(os.environ['AFD_PREPARE_DIR'])
        batch = int(os.environ['AFD_PREPARE_BATCH'])
        tokens = (int(args[0] if args else kwargs['num_tokens']) if role=='a' else
                  sum(int(n) for n in args[0].num_tokens_across_dp_cpu))
        active = (root/'measure.enabled').exists() and len(rows)<200 and tokens==batch
        if not active:
            return original(self,*args,**kwargs)
        import torch
        if device:
            torch.cuda.synchronize()
        start = time.perf_counter_ns()
        result = original(self,*args,**kwargs)
        returned = time.perf_counter_ns()
        if device:
            torch.cuda.synchronize()
        end = time.perf_counter_ns()
        rows.append(dict(service_us=(end-start)/1000,api_us=(returned-start)/1000,
                         completion_tail_us=(end-returned)/1000))
        if len(rows)==200:
            with (root/f'{role}-metadata-samples.json').open('x') as stream:
                json.dump(dict(operation=operation,batch=batch,window=int(os.environ['AFD_PREPARE_WINDOW']),
                    samples=rows,timing_scope='operation_service',external_wait_excluded=True,
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    source_file=inspect.getsourcefile(original),source_line=inspect.getsourcelines(original)[1],
                    completion='device_synchronize_after_operation' if device else 'cpu_api_return_data_ready',
                    prior_work_synchronize_outside_timer=device,profiler_enabled=False,
                    gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES']),stream)
        return result
    setattr(cls,method,measured)


def install_ffn(cls):
    if os.environ.get('AFD_PREPARE_MODE') == 'compute_cpu':
        from afd_compute_cpu_hooks import install as install_compute
        install_compute(cls,'f')
        return
    if os.environ.get('AFD_PREPARE_MODE') == 'metadata':
        install_metadata(cls,'_make_ffn_dp_metadata','f_dp_metadata','f',False)


def install_sampled_tokens(cls):
    original = cls._to_list
    source = inspect.getsource(original)
    rows = []
    def measured(self, sampled_token_ids):
        root = Path(os.environ['AFD_PREPARE_DIR'])
        batch = int(os.environ['AFD_PREPARE_BATCH'])
        context = int(self.input_batch.num_computed_tokens_cpu[0]) + 1
        enabled = ((root/'measure.enabled').exists() and len(rows)<200
                   and sampled_token_ids.shape[0]==batch and 129<=context<=159)
        if not enabled:
            return original(self,sampled_token_ids)
        import torch
        torch.cuda.synchronize()  # Exclude all prior compute/NCCL, outside timer.
        begin=time.perf_counter_ns()
        result=original(self,sampled_token_ids)
        end=time.perf_counter_ns()
        assert len(result)==batch
        rows.append(dict(context=context,service_us=(end-begin)/1000,
            shape=list(sampled_token_ids.shape),dtype=str(sampled_token_ids.dtype)))
        if len(rows)==200:
            with (root/'sampled-token-samples.json').open('x') as f:
                json.dump(dict(operation='sampled_token_readback',batch=batch,
                    window=int(os.environ['AFD_PREPARE_WINDOW']),samples=rows,
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    source_file=inspect.getsourcefile(original),source_line=inspect.getsourcelines(original)[1],
                    timing_scope='operation_service',external_wait_excluded=True,
                    input_ready='sampled_tokens_device_ready',completion='cpu_api_return_data_ready',
                    prior_work_synchronize_outside_timer=True,profiler_enabled=False,
                    gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES']),f)
        return result
    cls._to_list=measured


def install_bookkeeping(cls):
    original = cls._bookkeeping_sync
    original_readback = cls._to_list
    rows = []
    nested = None

    def readback(self, *args, **kwargs):
        if nested is None:
            return original_readback(self,*args,**kwargs)
        start=time.perf_counter_ns()
        result=original_readback(self,*args,**kwargs)
        nested.append((time.perf_counter_ns()-start)/1000)
        return result

    def measured(self, *args, **kwargs):
        nonlocal nested
        root=Path(os.environ['AFD_PREPARE_DIR'])
        batch=int(os.environ['AFD_PREPARE_BATCH'])
        context=int(self.input_batch.num_computed_tokens_cpu[0])+1
        if not ((root/'measure.enabled').exists() and len(rows)<200
                and self.input_batch.num_reqs==batch and 129<=context<=159):
            return original(self,*args,**kwargs)
        if self.use_async_scheduling or self.routed_experts_initialized:
            raise ValueError('bookkeeping CPU-only audit excludes async/routing D2H')
        import torch
        torch.cuda.synchronize()  # Prior GPU work excluded before outer timer.
        nested=[]
        start=time.perf_counter_ns()
        try:
            result=original(self,*args,**kwargs)
            end=time.perf_counter_ns()
            excluded=list(nested)
        finally:
            nested=None
        if len(excluded)!=1 or result[1] is not None or result[3]:
            raise ValueError('expected one readback and no logprob device work')
        wall_us=(end-start)/1000
        service_us=wall_us-excluded[0]
        if service_us<0:raise ValueError('invalid nested CPU measurement')
        rows.append(dict(context=context,service_us=service_us,wall_us=wall_us,
            excluded_readback_us=excluded[0],readback_calls=1))
        if len(rows)==200:
            source=inspect.unwrap(original)
            with (root/'bookkeeping-samples.json').open('x') as stream:
                json.dump(dict(operation='runner_bookkeeping_cpu',batch=batch,
                    window=int(os.environ['AFD_PREPARE_WINDOW']),samples=rows,
                    source_sha256=hashlib.sha256(inspect.getsource(source).encode()).hexdigest(),
                    source_file=inspect.getsourcefile(source),source_line=inspect.getsourcelines(source)[1],
                    timing_scope='host_service',external_wait_excluded=True,
                    prior_work_synchronize_outside_timer=True,profiler_enabled=False,
                    readback_excluded=True,async_scheduling=False,routing_d2h=False,logprobs=False,
                    gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES']),stream)
        return result
    cls._to_list=readback
    cls._bookkeeping_sync=measured
