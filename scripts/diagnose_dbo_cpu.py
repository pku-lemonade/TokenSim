"""CPU-only timing of the installed plugin's DBO guard; no GPU work."""
import cProfile
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
from afd_plugin.v1.worker import dbo

def measure(fn):
    for _ in range(10): fn()
    values=[]
    for _ in range(200):
        start=time.perf_counter_ns(); fn(); values.append((time.perf_counter_ns()-start)/1000)
    return dict(median_us=statistics.median(values),min_us=min(values),max_us=max(values))

result=dict(installed_guard=measure(dbo._yield_if_dbo_enabled),native_dbo_check=measure(dbo.dbo_enabled))
try:
    importlib.import_module('afd_plugin.v1.worker.npu.ubatching')
    result['npu_import']='success'
except ImportError as e:
    result['npu_import']=str(e)
result['npu_module_cached']='afd_plugin.v1.worker.npu.ubatching' in sys.modules
prof=cProfile.Profile()
prof.enable()
for _ in range(100): dbo._yield_if_dbo_enabled()
prof.disable()
stats=prof.getstats()
result['cpu_profile_top']=[dict(function=str(s.code),calls=s.callcount,total_s=s.totaltime,self_s=s.inlinetime)
                           for s in sorted(stats,key=lambda s:s.totaltime,reverse=True)[:15]]
path=Path(sys.argv[1])
path.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
