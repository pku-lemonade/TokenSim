"""Rebuild native routing evidence and verify every modeled wire message."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from scripts.convert_afd_capture import convert


def audit(batch):
    root = Path('outputs/afd-frozen-validation-20261007-v1') if batch == 4 else Path(f'outputs/afd-live-capture-20261007/b{batch}')
    replay_path = root/'replay-b4.json' if batch == 4 else root.parent/f'replay-b{batch}.json'
    replay = json.loads(replay_path.read_text(encoding='utf-8'))
    files = sorted(root.glob('capture-*.jsonl'))
    assert len(files) == 2
    records = [json.loads(line) for path in files for line in path.read_text(encoding='utf-8').splitlines() if line]
    rebuilt = convert(records, runtime=replay['runtime_fingerprint'], environment=replay['environment'],
        source=replay['source'], batch=batch, layers=2, steps=31)
    assert rebuilt == replay
    events = json.loads(Path(f'outputs/afd-preparation-replay-20261007-b{batch}/tokensim-b{batch}-decode-events.json').read_text(encoding='utf-8'))
    source = json.loads(Path('docs/plans/afd-native-source-audit-20261007.json').read_text(encoding='utf-8'))
    wire = next(e for e in source['examples'] if e['batch'] == batch)
    for step in range(31):
        current = [e for e in events if e['iteration_id'] == step]
        controls = [e for e in current if e['event_type'] == 'control_transfer']
        assert Counter(e['payload_bytes'] for e in controls) == Counter([8, 179])
        assert all(e['direction'] == 'a_to_f' and e['details']['transport_backend'] == 'torch_distributed_nccl' for e in controls)
        payload = next(e for e in controls if e['payload_bytes'] == 179)
        assert payload['details']['payload_utf8'] == wire['payload_utf8']
        data = [e for e in current if e['event_type'] == 'transfer']
        assert Counter((e['layer_id'], e['direction'], e['payload_bytes']) for e in data) == Counter(
            (layer, direction, batch*256*2) for layer in (0,1) for direction in ('a_to_f','f_to_a'))
        for layer in (0,1):
            expected = Counter(a['expert_id'] for a in replay['assignments'] if a['decode_step'] == step and a['layer_id'] == layer)
            compute = [e for e in current if e['event_type'] == 'compute' and e['stage'] == 'routed_expert' and e['layer_id'] == layer]
            assert len(compute) == 1
            actual = {int(k):v for k,v in compute[0]['details']['expert_workload'].items()}
            assert actual == dict(expected), (actual, expected)
    return dict(batch=batch, decode_steps=31, routing_assignments=len(replay['assignments']),
        native_layer_buffers=62, verified_histograms=62, modeled_data_messages=124, modeled_control_messages=62,
        captured_codec_calls=Counter(r.get('operation') for r in records if r['kind']=='codec_interval'),
        capture_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        scope='逐行路由与buffer实采；metadata字节来自原序列化函数；调用次数按源码循环及codec实采核对，不是API时间戳回放')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=[audit(b) for b in (1,8,4)]
    with a.output.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
