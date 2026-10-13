"""Inspect one decode step on the server, without exporting full traces."""
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
for role in ('attention', 'ffn'):
    events = json.loads(next((root/'c1'/role).glob('*.pt.trace.json')).read_text())['traceEvents']
    steps = sorted([e for e in events if e.get('cat') == 'user_annotation' and
                    e.get('name', '').startswith('ProfilerStep#')], key=lambda e: e['ts'])
    s = steps[1]
    print(role, 'annotations', Counter(e['name'] for e in events if e.get('cat') == 'user_annotation'))
    origins = {e['args']['External id']: e for e in events if e.get('cat') == 'cpu_op'
               and 'External id' in e.get('args', {})}
    runtimes = {e['args']['correlation']: e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver')
                and 'correlation' in e.get('args', {})}
    for e in events:
        if e.get('cat') != 'kernel' or 'nccl' in e['name'].lower():
            continue
        o = origins.get(e.get('args', {}).get('External id')) or runtimes.get(e.get('args', {}).get('correlation'))
        if o and s['ts'] <= o['ts'] < s['ts'] + s['dur']:
            print(json.dumps(dict(cpu=o['name'], dims=o.get('args', {}).get('Input Dims'),
                                  cpu_ts=o['ts']-s['ts'], kernel=e['name'][:220], us=e['dur'])))
