"""Compact real-engine rerun evidence; never interpret NCCL occupancy as service."""
import hashlib
import json
import statistics
import sys
from pathlib import Path
from summarize_compute_alignment import summarize

root = Path(sys.argv[1])
previous = root.parent/'afd-compute-gpu03-20261006'
rows = []
for batch in (1, 8, 4):
    case = root/f'c{batch}'
    calls = json.loads((case/'calls.json').read_text())
    outputs = json.loads((case/'outputs.json').read_text())
    old_outputs = json.loads((previous/f'c{batch}/outputs.json').read_text())
    reference = outputs[0]
    row = dict(batch=batch, repetitions=len(calls),
               engine_decode_median_us=statistics.median(v*1000 for c in calls for v in c['step_ms'][1:]),
               engine_decode_p10_p90_us=[statistics.quantiles([v*1000 for c in calls for v in c['step_ms'][1:]],n=10)[i] for i in (0,8)],
               repeated_tokens_equal=all(o == reference for o in outputs),
               tokens_equal_to_gpu03=all(o == reference for o in old_outputs),
               token_sha256=hashlib.sha256(json.dumps(reference).encode()).hexdigest(),
               case=json.loads((case/'case.json').read_text()),
               result=json.loads((case/'result.json').read_text()))
    for role in ('attention', 'ffn'):
        path = next((case/role).glob('*.pt.trace.json'))
        row[role] = summarize(path, role, batch)
        events = json.loads(path.read_text())['traceEvents']
        steps = sorted([e for e in events if e.get('cat') == 'user_annotation' and
                       e.get('name','').startswith('ProfilerStep#')],key=lambda e:e['ts'])
        row[role]['profiled_step_median_us_excluding_first'] = statistics.median(e['dur'] for e in steps[1:])
    rows.append(row)
base = root.parents[1]
plugin = Path(sys.argv[2]) if len(sys.argv)>2 else base/'src/afd-plugin'
sources = [plugin/'afd_plugin/model_executor/models/qwen3_moe.py',
           plugin/'afd_plugin/connectors/gpu/p2p.py', plugin/'afd_plugin/v1/worker/dbo.py',
           base/'models/qwen3-moe-tiny-random/config.json', root/'run_curve.py',root/'attention.py',Path(__file__)]
summary = dict(rows=rows, source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               scope='tiny Qwen decode; kernel sums and unprofiled engine latency are different scopes; NCCL occupancy includes waiting')
(root/'rerun-summary.json').write_text(json.dumps(summary,indent=2))
for r in rows:
    print(json.dumps({k:v for k,v in r.items() if k not in ('case','result')}))
