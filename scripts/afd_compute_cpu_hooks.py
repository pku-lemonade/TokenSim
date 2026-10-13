"""Ready-input native compute CPU cost with CUDA API interval union removed."""
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import time


class Result(ctypes.Structure):
    _fields_=[(name,ctypes.c_uint64) for name in ('wall_ns','api_ns','api_calls','unmatched')]


class ApiRow(ctypes.Structure):
    _fields_=[('name',ctypes.c_char*160),('ns',ctypes.c_uint64),('calls',ctypes.c_uint64),('copy_kinds',ctypes.c_uint64)]


def install(cls, role):
    original_load=cls.load_model

    def load(self,*args,**kwargs):
        result=original_load(self,*args,**kwargs)
        wall_control=os.environ.get('AFD_CPU_WALL_CONTROL')=='1'
        api_audit=os.environ.get('AFD_CPU_API_AUDIT')=='1'
        if api_audit and wall_control:
            raise RuntimeError('API audit requires CUDA callbacks')
        lib=None
        if not wall_control:
            lib=ctypes.PyDLL(os.environ['AFD_CPU_COUNTER_LIB'])
            lib.afd_init.restype=ctypes.c_int
            lib.afd_end.argtypes=[ctypes.POINTER(Result)]
            code=lib.afd_init()
            if code:raise RuntimeError(f'CUPTI subscription failed: {code}')
            if api_audit:
                lib.afd_api_count.restype=ctypes.c_uint
                lib.afd_api_row.argtypes=[ctypes.c_uint]
                lib.afd_api_row.restype=ctypes.POINTER(ApiRow)
        root=Path(os.environ['AFD_PREPARE_DIR'])
        batch=int(os.environ['AFD_PREPARE_BATCH'])
        windows={}
        sources={}
        overhead=[]
        for _ in range(100):
            if wall_control:
                start=time.perf_counter_ns();overhead.append((time.perf_counter_ns()-start)/1000)
            else:
                sample=Result();lib.afd_begin();lib.afd_end(ctypes.byref(sample))
                overhead.append(sample.wall_ns/1000)

        def wrap(owner,method,key):
            original=getattr(owner,method)
            source=inspect.unwrap(original)
            sources[key]=dict(file=inspect.getsourcefile(source),line=inspect.getsourcelines(source)[1],
                sha256=hashlib.sha256(inspect.getsource(source).encode()).hexdigest())
            windows[key]=[]
            def measured(*args,**kwargs):
                import torch
                tensors=[v for v in (*args,*kwargs.values()) if isinstance(v,torch.Tensor)]
                eligible=(root/'measure.enabled').exists() and len(windows[key])<200 and any(
                    v.ndim and v.shape[0]==batch for v in tensors)
                if not eligible:return original(*args,**kwargs)
                torch.cuda.synchronize()
                sample=Result()
                if wall_control:start=time.perf_counter_ns()
                else:lib.afd_begin()
                try:
                    value=original(*args,**kwargs)
                finally:
                    if wall_control:sample.wall_ns=time.perf_counter_ns()-start
                    else:lib.afd_end(ctypes.byref(sample))
                if sample.unmatched or sample.api_ns>sample.wall_ns or (not wall_control and not sample.api_calls):
                    raise RuntimeError(f'unqualified API coverage in {key}')
                windows[key].append(dict(service_us=(sample.wall_ns-sample.api_ns)/1000,
                    wall_us=sample.wall_ns/1000,excluded_cuda_api_us=sample.api_ns/1000,
                    cuda_api_calls=sample.api_calls))
                if api_audit:
                    api_rows=[lib.afd_api_row(i).contents for i in range(lib.afd_api_count())]
                    windows[key][-1]['outer_cuda_apis']=[dict(name=r.name.decode(),duration_us=r.ns/1000,calls=r.calls,copy_kinds=r.copy_kinds) for r in api_rows]
                    if sum(r.ns for r in api_rows)!=sample.api_ns:
                        raise RuntimeError('API audit union mismatch')
                if all(len(v)==200 for v in windows.values()):
                    path=root/f'{role}-compute-cpu-samples.json'
                    if not path.exists():
                        with path.open('x') as stream:json.dump(dict(role=role,batch=batch,
                            window=int(os.environ['AFD_PREPARE_WINDOW']),groups=windows,sources=sources,
                            empty_probe_us=overhead,timing_scope=('diagnostic_api_audit_not_fit' if api_audit else 'diagnostic_wall_not_fit' if wall_control else 'host_service_excluding_cuda_api_union'),
                            prior_sync_outside_timer=True,cuda_runtime_and_driver_callbacks=not wall_control,
                            profiler_enabled=False,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES']),stream)
                return value
            setattr(owner,method,measured)

        for name,module in self.model.named_modules():
            if role=='a':
                if re.search(r'layers\.\d+\.(input_layernorm|self_attn|post_attention_layernorm)$',name):
                    wrap(module,'forward',name)
                elif name.endswith(('model.embed_tokens','model.norm')):
                    wrap(module,'forward',name)
            elif re.search(r'layers\.\d+\.mlp$',name):
                wrap(module,'forward',name)
        if role=='a':
            wrap(self.model,'compute_logits','lm_head')
            wrap(self,'_sample','token_selection')
        if len(windows)!=(10 if role=='a' else 2):
            raise RuntimeError(f'unexpected native compute groups {tuple(windows)}')
        return result
    cls.load_model=load
