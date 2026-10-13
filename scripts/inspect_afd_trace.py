"""Read large remote profiles locally on the server; emit compact diagnostics."""
import collections
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    for role in ('attention', 'ffn'):
        path = next((root / role).glob('*.pt.trace.json'))
        events = json.loads(path.read_text())['traceEvents']
        cats = collections.Counter(e.get('cat') for e in events)
        users = collections.Counter(e.get('name') for e in events if e.get('cat') == 'user_annotation')
        kernels = collections.Counter(e.get('name') for e in events if e.get('cat') == 'kernel')
        print(json.dumps({'role': role, 'cats': cats, 'ranges': users.most_common(16),
                          'kernels': [(k[:140], v) for k, v in kernels.most_common(12)]}))
        for cat in ('kernel', 'cuda_runtime', 'user_annotation'):
            sample = next((e for e in events if e.get('cat') == cat), None)
            print(json.dumps({'role': role, 'sample': sample})[:1200])


if __name__ == '__main__':
    main()
