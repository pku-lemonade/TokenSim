"""Join captured native A row identities and actual F top-k; fail on gaps."""
import argparse
import json
from pathlib import Path
from TokenSim.af.calibration import content_fingerprint
from TokenSim.af.routing_replay import RoutingReplay
from TokenSim.moe.routing import RoutingAssignment, LayerRoutingSnapshot


def convert(records, *, runtime, environment, source, batch, layers, steps, context=129, experts=4, top_k=2):
    a, f, buffers = {}, {}, {}
    for row in records:
        if row['kind'] == 'codec_interval':
            continue  # diagnostic only, deliberately not converted to service
        if row['kind'] not in ('row_identity', 'routing', 'layer_buffers'):
            raise ValueError('unknown capture record')
        target = a if row['kind'] == 'row_identity' else f if row['kind'] == 'routing' else buffers
        key = row['decode_step'] if target is a else (row['decode_step'], row['layer_id'])
        if key in target:
            raise ValueError('duplicate capture record')
        target[key] = row
    expected = {(s,l) for s in range(steps) for l in range(layers)}
    if set(a) != set(range(steps)) or set(f) != expected or set(buffers) != expected:
        raise ValueError('capture incomplete: require every step/layer, A row identity and F buffers')
    first = a[0]['rows']
    if len(first) != batch or len({r['native_request_id'] for r in first}) != batch:
        raise ValueError('invalid first request cohort')
    ids = {r['native_request_id']:i for i,r in enumerate(first)}
    assignments = []
    for step in range(steps):
        rows = a[step]['rows']
        if len(rows) != batch or {r['row'] for r in rows} != set(range(batch)) or {r['native_request_id'] for r in rows} != set(ids):
            raise ValueError('capture changed request cohort/row coverage')
        rows = sorted(rows, key=lambda r:r['row'])
        if any(r['token_id'] != context-1+step for r in rows):
            raise ValueError('capture token position disagrees with decode context')
        for layer in range(layers):
            routed = f[step,layer]
            if len(routed['expert_ids']) != batch or len(routed['routing_weights']) != batch:
                raise ValueError('routing batch mismatch')
            layer_records = []
            for row, selected, weights in zip(rows, routed['expert_ids'], routed['routing_weights']):
                if len(selected) != top_k or len(weights) != top_k or any(type(e) is not int or not 0 <= e < experts for e in selected):
                    raise ValueError('invalid expert selection')
                for k,(expert,weight) in enumerate(zip(selected, weights)):
                    record = dict(phase='decode', decode_step=step, layer_id=layer,
                        request_id=ids[row['native_request_id']], token_id=row['token_id'],
                        top_k_index=k, expert_id=expert, routing_weight=weight, a_dp_rank=0, a_tp_rank=0)
                    layer_records.append(RoutingAssignment(iteration_id=step, **record))
                    assignments.append(record)
            LayerRoutingSnapshot(step, 'decode', step, layer, top_k,
                tuple((ids[r['native_request_id']],r['token_id']) for r in rows), tuple(layer_records))
            buf = buffers[step,layer]
            if buf['input_shape'] != [batch,256] or buf['output_shape'] != [batch,256] or buf['input_bytes'] != batch*256*2 or buf['output_bytes'] != batch*256*2:
                raise ValueError('native tiny Qwen BF16 hidden buffer contract mismatch')
    replay = dict(schema_version=1, runtime_fingerprint=runtime, environment=environment,
                  source=source, kind='captured', assignments=assignments)
    RoutingReplay(replay, expected_runtime=runtime, expected_environment=environment)
    return replay


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('capture', type=Path)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--steps', type=int, default=31)
    args=parser.parse_args()
    files=sorted(args.capture.glob('capture-*.jsonl'))
    records=[json.loads(line) for path in files for line in path.read_text().splitlines() if line]
    replay=convert(records, runtime=json.loads(args.runtime.read_text())['runtime_fingerprint'],
        environment=json.loads(args.environment.read_text()), source=str(args.capture.resolve())+':'+content_fingerprint(records),
        batch=args.batch, layers=2, steps=args.steps)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(replay,stream,indent=2)


if __name__=='__main__': main()
