"""Attribute existing tiny Qwen AFD kernels to validated per-layer boundaries.

Specific to two-layer H256/Q4/KV2/top2 traces, not a generic model parser.
CPU call order gives layer identity; CUDA correlation gives launch attribution.
"""
import bisect
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def summarize(path, role, batch):
    events = json.loads(path.read_text())['traceEvents']
    steps = sorted([e for e in events if e.get('cat') == 'user_annotation' and
                    e['name'].startswith('ProfilerStep#')], key=lambda e: e['ts'])
    starts = [e['ts'] for e in steps]
    def step_of(e):
        i = bisect.bisect_right(starts, e['ts']) - 1
        return i if i >= 0 and e['ts'] < steps[i]['ts'] + steps[i]['dur'] else None
    ops = [e for e in events if e.get('cat') == 'cpu_op' and e.get('ph') == 'X']
    external = {e['args']['External id']: e for e in ops if 'External id' in e.get('args', {})}
    runtime = {e['args']['correlation']: e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver')
               and 'correlation' in e.get('args', {})}
    anchors = {}
    for i, s in enumerate(steps):
        local = sorted([e for e in ops if step_of(e) == i], key=lambda e: e['ts'])
        if role == 'attention':
            norms = [e for e in local if e['name'] in ('_C::rms_norm', '_C::fused_add_rms_norm') and
                     e.get('args', {}).get('Input Dims', [None])[0] == [batch, 256]]
            if not norms:
                continue  # prefill shape differs
            assert len(norms) == 5, (path, i, len(norms))
            assert norms[0]['name'] == '_C::rms_norm'
            assert all(e['name'] == '_C::fused_add_rms_norm' for e in norms[1:])
            anchors[i] = norms
        else:
            moes = [e for e in local if e['name'] == 'vllm::moe_forward' and
                    e.get('args', {}).get('Input Dims', [None])[0] == [batch, 256]]
            if not moes:
                continue
            assert len(moes) == 2, (path, i, len(moes))
            anchors[i] = moes
    assert len(anchors) == 31, (path, len(anchors))
    values = {i: defaultdict(float) for i in anchors}
    counts = {i: defaultdict(int) for i in anchors}
    unknown = defaultdict(int)
    copy_shapes = defaultdict(int)
    for k in events:
        if k.get('cat') != 'kernel' or 'nccl' in k['name'].lower():
            continue
        cpu = external.get(k.get('args', {}).get('External id'))
        launch = runtime.get(k.get('args', {}).get('correlation')) or cpu
        assert launch is not None, k['name']
        i = step_of(launch)
        if i not in anchors:
            continue
        a = anchors[i]
        op = cpu['name'] if cpu else launch['name']
        dims = cpu.get('args', {}).get('Input Dims', []) if cpu else []
        label = 'outside'
        layer = None
        if role == 'attention':
            for j in (0, 1):
                if a[2*j]['ts'] <= launch['ts'] < a[2*j+1]['ts'] + a[2*j+1]['dur']:
                    layer = j
                    break
            if layer is not None:
                if op in ('_C::rms_norm', '_C::fused_add_rms_norm'):
                    if dims[0] == [batch, 256]:
                        label = 'input_norm_previous_residual' if cpu['ts'] == a[2*layer]['ts'] else 'post_norm_attention_residual'
                    else:
                        label = 'q_norm' if dims[0] == [batch, 4, 64] else 'k_norm'
                elif op == 'aten::mm':
                    label = 'qkv_projection' if dims[1] == [256, 512] else 'output_projection'
                elif op == '_C::rotary_embedding':
                    label = 'rope'
                elif 'reshape_and_cache' in op:
                    label = 'kv_cache_write'
                elif 'varlen_fwd' in op:
                    label = 'attention_core'
                elif op == 'aten::copy_':
                    label = 'layout_copy'
                    copy_shapes[str(dims)] += 1
                else:
                    label = 'unclassified'
            else:
                if op == 'aten::index_select' and dims[0] == [128, 256]:
                    label = 'embedding'
                elif cpu and cpu['ts'] == a[4]['ts']:
                    label = 'final_norm_last_residual'
                elif op == 'aten::mm' and dims[1] == [256, 128]:
                    label = 'lm_head'
                elif op == 'aten::argmax':
                    label = 'argmax'
                elif 'slot_mapping' in k['name']:
                    label = 'kv_slot_mapping'
                else:
                    label = 'auxiliary'
        else:
            matches = [j for j, m in enumerate(a) if m['ts'] <= launch['ts'] < m['ts']+m['dur']]
            assert len(matches) == 1, (path, op, k['name'])
            layer = matches[0]
            topk = [e for e in ops if e['name'].endswith('topk_softmax') and
                    a[layer]['ts'] <= e['ts'] < a[layer]['ts']+a[layer]['dur']]
            assert len(topk) == 1
            if launch['ts'] < topk[0]['ts'] + topk[0]['dur']:
                label = 'router'
            elif 'moe_sum' in op:
                label = 'combine_sum'
            else:
                label = 'expert_and_routing_layout'
        if label == 'unclassified':
            unknown[op] += 1
        key = f'layer{layer}' if layer is not None else 'outside'
        values[i][key] += k['dur']
        values[i][key+'/'+label] += k['dur']
        values[i]['total'] += k['dur']
        counts[i][key] += 1
    keys = sorted({k for v in values.values() for k in v})
    return dict(decode_steps=31, unclassified=dict(unknown), layer_copy_shapes=dict(copy_shapes),
                mean_us={k: statistics.mean(v[k] for v in values.values()) for k in keys},
                median_us={k: statistics.median(v[k] for v in values.values()) for k in keys},
                min_max_us={k: [min(v[k] for v in values.values()), max(v[k] for v in values.values())] for k in keys},
                kernel_count_ranges={k: [min(v[k] for v in counts.values()), max(v[k] for v in counts.values())]
                                     for k in sorted({k for v in counts.values() for k in v})})


root = Path(sys.argv[1])
rows = []
for b in (1, 8, 4):
    rows.append(dict(batch=b, **{r: summarize(next((root/f'c{b}'/r).glob('*.pt.trace.json')), r, b)
                                for r in ('attention', 'ffn')}))
(root/'layer-summary.json').write_text(json.dumps(rows, indent=2))
for row in rows:
    print(json.dumps(row))
