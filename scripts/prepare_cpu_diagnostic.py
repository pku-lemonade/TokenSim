"""Isolated AFD driver with decode-only cProfile; installed plugin unchanged."""
from pathlib import Path

base=Path('/home/youwei/autodl-tmp/af-workspace/profiling')
source=base/'afd-gpu23-rerun-20261007'
target=base/'afd-cpu-diagnostic-20261007'
target.mkdir(exist_ok=False)
code=(source/'run_curve.py').read_text().replace('17839','17939').replace('17840','17940').replace('18893','18993')
(target/'run_curve.py').write_text(code)
code=(source/'attention.py').read_text()
code=code.replace('import faulthandler','import cProfile\nimport pstats\nimport faulthandler')
code=code.replace('    steps = []','    steps = []\n    cpu_profile = cProfile.Profile()\n    profile_enabled = False\n    profiled_decode_steps = 0')
old='        output = original_step()'
new='''        nonlocal profiled_decode_steps
        capture = profile_enabled and bool(steps)
        if capture:
            cpu_profile.enable()
        try:
            output = original_step()
        finally:
            if capture:
                cpu_profile.disable()
                profiled_decode_steps += 1'''
assert code.count(old)==1
code=code.replace(old,new)
code=code.replace('    nv.nvmlInit()','    profile_enabled = True\n    nv.nvmlInit()')
marker='    duration_s = (t1-t0)/1e9'
extra='''    cpu_profile.dump_stats(str(directory/'decode-cpu.pstats'))
    stats = pstats.Stats(cpu_profile)
    entries = [dict(file=k[0], line=k[1], function=k[2], calls=v[1],
                    calls_per_step=v[1]/profiled_decode_steps,
                    self_us_per_step=v[2]*1e6/profiled_decode_steps,
                    inclusive_us_per_step=v[3]*1e6/profiled_decode_steps)
               for k,v in stats.stats.items()]
    report = dict(profiled_decode_steps=profiled_decode_steps,
                  scope='A process decode CPU cProfile; intrusive; inclusive rows overlap',
                  top_self=sorted(entries,key=lambda r:r['self_us_per_step'],reverse=True)[:35],
                  top_inclusive=sorted(entries,key=lambda r:r['inclusive_us_per_step'],reverse=True)[:40],
                  afd_functions=[r for r in entries if 'afd_plugin' in r['file']])
    (directory/'cpu-summary.json').write_text(json.dumps(report,indent=2))
'''
assert marker in code
code=code.replace(marker,extra+marker)
(target/'attention.py').write_text(code)
print(target)
