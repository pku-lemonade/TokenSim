import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
events = json.loads(next((root/'c1/attention').glob('*.pt.trace.json')).read_text())['traceEvents']
k = next(e for e in events if e.get('cat') == 'kernel' and 'slot_mapping' in e['name'])
print('KERNEL', json.dumps(k))
for e in events:
    a = e.get('args', {})
    if (('External id' in k['args'] and a.get('External id') == k['args']['External id']) or
        a.get('correlation') == k['args'].get('correlation')):
        print(json.dumps(e)[:1800])
