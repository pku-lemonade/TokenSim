"""Generate the compute-only evidence bundle from compact local summaries."""
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
measured = root/'outputs/afd-compute-gpu03-20261006'
afd = json.loads((measured/'compute-summary.json').read_text())
controls = {g: json.loads((measured/f'control-gpu{g}/compute-summary.json').read_text()) for g in (0, 3)}
sim = {b: json.loads((root/f'outputs/afd-compute-baseline-20261006/tokensim-b{b}.json').read_text()) for b in (1, 8, 4)}
rows = []
for row in afd:
    b = row['batch']
    for role, endpoint in [('attention', 'a-pool:0'), ('ffn', 'f-pool:0')]:
        predicted = sim[b]['decode']['compute_service_by_endpoint_us'][endpoint]
        actual = row[role]['median_us']['compute']
        assert row[role]['decode_steps'] == 31 and not row[role]['unmatched_kernels']
        rows.append(dict(batch=b, role=role, simulator_us=predicted, native_us=actual,
                         native_over_simulator=actual/predicted))
bundle = dict(scope='tiny Qwen3 MoE decode compute-only; no fitted parameters',
              comparison=rows, native=afd, control=controls, simulator=sim,
              validation=json.loads((measured/'validation-summary.json').read_text()),
              manifest=json.loads((measured/'manifest.json').read_text()))
dest = root/'docs/plans/afd-compute-alignment-20261006.json'
dest.write_text(json.dumps(bundle, indent=2), encoding='utf-8')
for r in rows:
    print(f"| {r['batch']} | {r['role']} | {r['simulator_us']:.2f} | {r['native_us']:.2f} | {r['native_over_simulator']:.2f}× |")
for g, samples in controls.items():
    for r in samples:
        assert len(r['windows']) == 3
        assert all(w['decode_steps'] == 31 and not w['unmatched_kernels'] for w in r['windows'])
        print('control', g, r['batch'], {k: round(statistics.median(w['median_us'][k] for w in r['windows']), 3)
                                       for k in ('a_compute', 'f_compute', 'attention', 'other')})
