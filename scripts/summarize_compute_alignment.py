"""Compact server-side compute trace summaries, with explicit attribution gaps."""
import argparse
import bisect
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(path, role, batch):
    events = json.loads(path.read_text())['traceEvents']
    native = role == 'control'
    steps = sorted((e for e in events if e.get('ph') == 'X' and e.get('cat') == 'user_annotation' and
                    e.get('name', '').startswith('compute_step::' if native else 'ProfilerStep#')),
                   key=lambda e: e['ts'])
    starts = [e['ts'] for e in steps]
    origins = {e['args']['External id']: e for e in events if e.get('ph') == 'X'
               and e.get('cat') == 'cpu_op' and 'External id' in e.get('args', {})}
    runtimes = {e['args']['correlation']: e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver')
                and 'correlation' in e.get('args', {})}
    annotations = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'user_annotation' and
                   e.get('name', '').startswith('compute_control::')]
    def step_of(e):
        i = bisect.bisect_right(starts, e['ts']) - 1
        return i if i >= 0 and e['ts'] < steps[i]['ts'] + steps[i]['dur'] else None
    decode, prefill = set(), set()
    if native:
        decode = {i for i, e in enumerate(steps) if '::decode::' in e['name']}
        prefill = set(range(len(steps))) - decode
    for e in events:
        if e.get('cat') != 'cpu_op':
            continue
        dims = e.get('args', {}).get('Input Dims', [])
        i = step_of(e)
        if i is None:
            continue
        if role == 'attention' and e['name'] == 'aten::mm' and dims and dims[0] == [128*batch, 256]:
            prefill.add(i)
        if role == 'ffn' and e['name'].endswith('topk_softmax') and dims and dims[0]:
            (decode if dims[0][0] == batch else prefill).add(i)
    values = defaultdict(lambda: defaultdict(float))
    unmatched = defaultdict(int)
    fallback = 0
    for e in events:
        if e.get('cat') != 'kernel':
            continue
        origin = origins.get(e.get('args', {}).get('External id'))
        if origin is None:
            origin = runtimes.get(e.get('args', {}).get('correlation'))
            fallback += origin is not None
        i = step_of(origin) if origin else None
        if i is None:
            unmatched[e['name']] += 1
            continue
        name = e['name'].lower()
        if role == 'attention' and 'splitkv' in name:
            decode.add(i)
        category = 'nccl_occupancy' if 'nccl' in name else 'compute'
        values[i][category] += e['dur']
        if native and category == 'compute':
            ranges = [a for a in annotations if a['tid'] == origin['tid'] and
                      a['ts'] <= origin['ts'] < a['ts'] + a['dur']]
            label = min(ranges, key=lambda a: a['dur'])['name'].split('::')[-1] if ranges else 'other'
            values[i][label] += e['dur']
            values[i]['f_compute' if label == 'moe' else 'a_compute'] += e['dur']
    decode -= prefill
    if not decode:
        raise RuntimeError(f'No decode steps: {path}')
    if len(decode) != 31:
        raise RuntimeError(f'Expected 31 decode steps, got {len(decode)}: {path}')
    if any(values[i]['compute'] <= 0 for i in decode):
        raise RuntimeError(f'Empty compute step in {path}')
    labels = sorted({k for i in decode for k in values[i]})
    return dict(decode_steps=len(decode), prefill_steps=len(prefill),
                runtime_fallback_kernels=fallback, unmatched_kernels=dict(unmatched),
                median_us={k: statistics.median(values[i][k] for i in decode) for k in labels},
                range_us={k: [min(values[i][k] for i in decode), max(values[i][k] for i in decode)] for k in labels})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--control', action='store_true')
    args = p.parse_args()
    rows = []
    for batch in (1, 8, 4):
        if args.control:
            rows.append(dict(batch=batch, windows=[summarize(path, 'control', batch)
                        for path in sorted(args.root.glob(f'b{batch}-r*.json'))]))
        else:
            rows.append(dict(batch=batch, **{role: summarize(next((args.root/f'c{batch}'/role).glob('*.pt.trace.json')), role, batch)
                                             for role in ('attention', 'ffn')}))
    (args.root/'compute-summary.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
