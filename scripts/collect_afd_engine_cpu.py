"""Six isolated native CPU scheduler/output windows on the authorized GPU pair."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

if __package__:
    from .collect_afd_cpu_attribution import once
else:
    from collect_afd_cpu_attribution import once


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    base,root = args.base.resolve(),args.output.resolve()
    if root.exists() or base not in root.parents:
        raise ValueError('new output directory within workspace required')
    source = base/'profiling/afd-native-collection-20261007-v1/timing-b1-w0'
    driver = (source/'attention.py').read_text()
    driver = '\n'.join(line for line in driver.splitlines() if 'profile incomplete' not in line)+'\n'
    driver = once(driver,'import faulthandler','import afd_engine_cpu_hooks as cpu\nimport faulthandler')
    driver = once(driver,'    llm = LLM(**kwargs)','    cpu.install()\n    llm = LLM(**kwargs)')
    driver = once(driver,'    steps = []','    steps = []\n    enabled = False\n    cpu_samples = []')
    driver = once(driver,'        output = original_step()', '''        capture = enabled and bool(steps) and len(cpu_samples)<200
        cpu.events.clear()
        cpu.active = capture
        try:
            output = original_step()
        finally:
            cpu.active = False
        if capture:
            cpu_samples.append(cpu.finish_step())''')
    driver = once(driver,'    nv.nvmlInit()','    enabled = True\n    nv.nvmlInit()')
    driver = once(driver,'    duration_s = (t1-t0)/1e9', '''    assert len(cpu_samples)==200
    (directory/'engine-cpu-samples.json').write_text(json.dumps(dict(
        batch=cfg['concurrency'],window=int(os.environ['AFD_CPU_WINDOW']),samples=cpu_samples,
        sources=cpu.sources,timing_scope='host_service',contains_wait=False,
        profiler_enabled=False,execution_mode='eager',async_scheduling=False,
        detokenize=False,kv_transfer_enabled=False,
        coverage=['Scheduler.schedule','Scheduler.update_from_output','OutputProcessor.process_outputs'],
        excludes=['GPU submission','sample_tokens','_bookkeeping_sync','_to_list','input preparation','metadata','codec'],
        gpu_uuids=cfg['gpu_uuids'])))
    duration_s = (t1-t0)/1e9''')
    runner = (source/'run_curve.py').read_text()
    runner = once(runner,'dict(ENABLE=1,WAIT=32','dict(ENABLE=0,WAIT=32')
    runner = '\n'.join(line for line in runner.splitlines() if "assert len(list(directory.glob('*/*.pt.trace.json')))==2" not in line)+'\n'
    ast.parse(driver); ast.parse(runner)
    plugin = base/'src/afd-plugin-dbo-cache-20261007'
    hashes = {str(f.relative_to(plugin)):hashlib.sha256(f.read_bytes()).hexdigest() for f in plugin.rglob('*.py')}
    root.mkdir()
    cases = []
    for window in range(3):
        for batch in ((6,) if os.environ.get('AFD_COMPONENT_VALIDATION')=='6' else (1,8)):
            case = root/f'b{batch}-w{window}'
            case.mkdir()
            (case/'attention.py').write_text(driver)
            (case/'run_curve.py').write_text(runner)
            shutil.copy2(Path(__file__).with_name('afd_engine_cpu_hooks.py'),case/'afd_engine_cpu_hooks.py')
            env = {k:v for k,v in os.environ.items() if not k.startswith(('AFD_CAPTURE_','AFD_PREPARE_'))}
            env.update(AF_WORKSPACE=str(base),AFD_ISOLATED_PLUGIN=str(plugin),AFD_CPU_WINDOW=str(window))
            with (case/'collection.log').open('x') as log:
                subprocess.run([str(base/'envs/afd026/bin/python'),str(case/'run_curve.py'),'--concurrency',str(batch)],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            result = json.loads((case/f'c{batch}/result.json').read_text())
            assert result['repeat_token_min_match']==1
            cases.append(dict(batch=batch,window=window,repeat_token_min_match=1))
            print(json.dumps(cases[-1]),flush=True)
    assert hashes=={str(f.relative_to(plugin)):hashlib.sha256(f.read_bytes()).hexdigest() for f in plugin.rglob('*.py')}
    (root/'manifest.json').write_text(json.dumps(dict(source_sha256=hashes,cases=cases),indent=2))


if __name__ == '__main__': main()
