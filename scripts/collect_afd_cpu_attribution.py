"""Fresh paired CPU attribution/control runs; diagnostic timings never fitted."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('driver source drift: '+old)
    return source.replace(old,new,1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phases-only',action='store_true')
    a=p.parse_args()
    base,root=a.base.resolve(),a.output.resolve()
    if root.exists() or base not in root.parents:
        raise ValueError('new output directory inside workspace required')
    source=base/'profiling/afd-native-collection-20261007-v1/timing-b1-w0'
    driver=(source/'attention.py').read_text()
    driver='\n'.join(line for line in driver.splitlines() if 'profile incomplete' not in line)+'\n'
    runner=(source/'run_curve.py').read_text()
    runner=once(runner,'dict(ENABLE=1,WAIT=32','dict(ENABLE=0,WAIT=32')
    runner='\n'.join(line for line in runner.splitlines() if "assert len(list(directory.glob('*/*.pt.trace.json')))==2" not in line)+'\n'
    profiled=once(driver,'import faulthandler','import cProfile\nimport pstats\nimport faulthandler')
    profiled=once(profiled,'    steps = []','    steps = []\n    cpu_profile = cProfile.Profile()\n    enabled = False\n    measured_steps = 0')
    profiled=once(profiled,'        output = original_step()', '''        nonlocal measured_steps
        capture = enabled and bool(steps)
        if capture:
            cpu_profile.enable()
        try:
            output = original_step()
        finally:
            if capture:
                cpu_profile.disable()
                measured_steps += 1''')
    profiled=once(profiled,'    nv.nvmlInit()','    enabled = True\n    nv.nvmlInit()')
    extra='''    cpu_profile.dump_stats(str(directory/'decode-cpu.pstats'))
    stats = pstats.Stats(cpu_profile)
    entries = [dict(file=k[0],line=k[1],function=k[2],calls=v[1],
        calls_per_step=v[1]/measured_steps,self_us_per_step=v[2]*1e6/measured_steps,
        inclusive_us_per_step=v[3]*1e6/measured_steps) for k,v in stats.stats.items()]
    report = dict(profiled_decode_steps=measured_steps,
        scope='cProfile A process decode; inclusive rows overlap; diagnostic not calibration',
        total_self_us_per_step=sum(r['self_us_per_step'] for r in entries),
        top_self=sorted(entries,key=lambda r:r['self_us_per_step'],reverse=True)[:60],
        top_inclusive=sorted(entries,key=lambda r:r['inclusive_us_per_step'],reverse=True)[:80],
        all_entries=entries)
    (directory/'cpu-summary.json').write_text(json.dumps(report,indent=2))
'''
    profiled=once(profiled,'    duration_s = (t1-t0)/1e9',extra+'    duration_s = (t1-t0)/1e9')
    ast.parse(runner); ast.parse(driver); ast.parse(profiled)
    phased=once(driver,'import faulthandler','import afd_phase_diagnostics as diag\nimport faulthandler')
    phased=once(phased,'    llm = LLM(**kwargs)','    diag.install()\n    llm = LLM(**kwargs)')
    phased=once(phased,'    steps = []','    steps = []\n    enabled = False\n    phase_steps = []')
    phased=once(phased,'        output = original_step()', '''        capture = enabled and bool(steps)
        if capture:
            diag.rows.clear()
            diag.active = True
        try:
            output = original_step()
        finally:
            diag.active = False
            if capture:
                phase_steps.append(dict(start_ns=before,end_ns=time.perf_counter_ns(),events=list(diag.rows)))''')
    phased=once(phased,'    nv.nvmlInit()','    enabled = True\n    nv.nvmlInit()')
    phased=once(phased,'    duration_s = (t1-t0)/1e9', '''    (directory/'phase-samples.json').write_text(json.dumps(dict(
        scope='native CPU call wall intervals, includes waiting and own device work; not fixed service',
        sources=diag.sources,steps=phase_steps)))
    duration_s = (t1-t0)/1e9''')
    ast.parse(phased)
    plugin=base/'src/afd-plugin-dbo-cache-20261007'
    hashes={str(p.relative_to(plugin)):hashlib.sha256(p.read_bytes()).hexdigest() for p in plugin.rglob('*.py')}
    root.mkdir()
    manifest=dict(source_sha256=hashes,cases=[])
    for batch in (1,8):
        modes=[('phases',phased)] if a.phases_only else [('control',driver),('cprofile',profiled)]
        for mode,code in modes:
            case=root/f'b{batch}-{mode}'
            case.mkdir()
            (case/'attention.py').write_text(code)
            (case/'run_curve.py').write_text(runner)
            if a.phases_only:
                shutil.copy2(Path(__file__).with_name('afd_phase_diagnostics.py'),case/'afd_phase_diagnostics.py')
            env={k:v for k,v in os.environ.items() if not k.startswith(('AFD_CAPTURE_','AFD_PREPARE_'))}
            env.update(AF_WORKSPACE=str(base),AFD_ISOLATED_PLUGIN=str(plugin))
            with (case/'collection.log').open('x') as log:
                subprocess.run([str(base/'envs/afd026/bin/python'),str(case/'run_curve.py'),'--concurrency',str(batch)],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((case/f'c{batch}/result.json').read_text())
            assert result['repeat_token_min_match']==1
            manifest['cases'].append(dict(batch=batch,mode=mode,repetitions=result['repetitions']))
            print(json.dumps(manifest['cases'][-1]),flush=True)
    assert hashes=={str(p.relative_to(plugin)):hashlib.sha256(p.read_bytes()).hexdigest() for p in plugin.rglob('*.py')}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
