"""Audit actual operation charges in saved native-routing simulator replays."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def audit(root, profile, codec):
    summaries = []
    for batch in (1, 8):
        directory = root / f'afd-operation-profile-20261007-b{batch}'
        events = json.loads((directory/f'tokensim-b{batch}-decode-events.json').read_text())
        result = json.loads((directory/f'tokensim-b{batch}.json').read_text())
        assert result['routing_replay'] and result['operation_profile']
        iterations = sorted({e['iteration_id'] for e in events})
        assert len(iterations) == 31
        for iteration in iterations:
            step = [e for e in events if e['iteration_id'] == iteration]
            operations = [e for e in step if e['event_type'] == 'operation_service']
            assert Counter(e['operation'] for e in operations) == Counter({op:1 for op in profile['operations']})
            assert not any(e['event_type'] == 'control_device' and e['operation'] in profile['operations'] for e in step)
            for event in operations:
                assert math.isclose(event['service_duration']*1e6, profile['operations'][event['operation']]['fixed_us'], abs_tol=1e-8)
                assert event['details']['calibrated'] and event['source_id'].startswith('operation-profile:')
                assert event['resource_id'].startswith('host:')
            for name, entry in codec['costs'].items():
                charged = [e for e in step if e['details'].get('cost_field') == name]
                assert len(charged) == 1 and charged[0]['details']['calibrated']
                assert math.isclose(charged[0]['service_duration']*1e6, entry['fixed_us'], abs_tol=1e-8)
        summaries.append(dict(batch=batch, steps=31, operations_per_step=3, codec_calls_per_step=2,
                              mean_step_us=result['decode']['mean_step_us'],
                              critical_chain_us=result['decode']['mean_critical_chain_us']))
    return dict(status='partial_operation_calibration_verified_not_full_engine', results=summaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(Path('outputs'), json.loads(Path('docs/plans/afd-operation-profile-20261007.json').read_text()),
                   json.loads(Path('docs/plans/afd-measured-codec-profile-20261007.json').read_text()))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
