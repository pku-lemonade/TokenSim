"""Read-only single-process timeline diagnostics; do not add overlapping spans."""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def union_us(intervals):
    total = 0
    end = float('-inf')
    for a,b in sorted(intervals):
        total += max(0,b-max(a,end))
        end=max(end,b)
    return total


root=Path(sys.argv[1])
rows=[]
for batch in (1,8,4):
    for role in ('attention','ffn'):
        events=json.loads(next((root/f'c{batch}'/role).glob('*.pt.trace.json')).read_text())['traceEvents']
        steps=sorted([e for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('ProfilerStep#')],key=lambda e:e['ts'])[1:]
        assert len(steps)==31
        metrics=[]
        per_name=defaultdict(list)
        counts=defaultdict(list)
        for s in steps:
            lo,hi=s['ts'],s['ts']+s['dur']
            selected=[e for e in events if e.get('ph')=='X' and e.get('tid')==s['tid'] and
                      e.get('pid')==s['pid'] and e.get('cat') in ('cpu_op','cuda_runtime','cuda_driver') and lo<=e['ts']<hi]
            intervals=[(e['ts'],min(hi,e['ts']+e['dur'])) for e in selected]
            sync=[e for e in selected if e.get('cat') in ('cuda_runtime','cuda_driver') and
                  any(t in e['name'].lower() for t in ('synchronize','eventquery'))]
            sums=defaultdict(float)
            nums=defaultdict(int)
            for e in selected:
                key=e['cat']+'::'+e['name']
                sums[key]+=e['dur']; nums[key]+=1
            for k,v in sums.items(): per_name[k].append(v); counts[k].append(nums[k])
            metrics.append(dict(step_us=s['dur'], cpu_traced_union_us=union_us(intervals),
                                cpu_unattributed_us=s['dur']-union_us(intervals),
                                cuda_sync_union_us=union_us([(e['ts'],e['ts']+e['dur']) for e in sync])))
        top=sorted([dict(name=k,median_inclusive_us=statistics.median(v),steps_present=len(v),
                         median_calls=statistics.median(counts[k])) for k,v in per_name.items()],key=lambda x:x['median_inclusive_us'],reverse=True)[:22]
        row=dict(batch=batch,role=role,median={k:statistics.median(m[k] for m in metrics) for k in metrics[0]},
                 top_cpu_inclusive_not_additive=top)
        rows.append(row)
(root/'timeline-diagnostic.json').write_text(json.dumps(rows,indent=2))
for row in rows: print(json.dumps(row))
