"""Summarize paired samples without conflating receiver occupancy with service."""
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
assert (root/'COMPLETE').exists()
ranks = [json.loads((root/f'rank{r}.json').read_text()) for r in (0, 1)]
rows = []
for sender in (0, 1):
    keys = sorted({(r['mode'],r['bytes'],r['receiver_delay_ms'],r['gate_cycles']) for r in ranks[sender] if r['sender'] == sender})
    for mode, size, delay, gate in keys:
        groups = [[r for r in rank if (r['mode'],r['bytes'],r['receiver_delay_ms'],r['gate_cycles'],r['sender'])
                   == (mode,size,delay,gate,sender)] for rank in ranks]
        assert all(len(g) == 3 and all(r['n'] == 30 for r in g) for g in groups)
        s, recv = groups[sender], groups[1-sender]
        rows.append(dict(sender=sender, mode=mode, bytes=size, delay_ms=delay, gate_cycles=gate,
                    sender_median_us=statistics.median(r['median_us'] for r in s),
                    sender_window_medians_us=[r['median_us'] for r in s],
                    sender_min_max_us=[min(r['min_us'] for r in s),max(r['max_us'] for r in s)],
                    receiver_occupancy_median_us=statistics.median(r['median_us'] for r in recv),
                    host_api_median_us=statistics.median(r['host_api_median_us'] for r in s)))
manifest = json.loads((root/'manifest.json').read_text())
result = dict(manifest=manifest, complete=True, rows=rows,
              caveat=('Ready sender interval is data plus 32-byte acknowledgement RTT, NOT one-way service. '
                      if manifest.get('ack_bytes') else
                      'Sender-local completion does NOT establish receiver readiness; rejected as delivery-service calibration. ')
                     +'Delayed/receiver intervals contain wait. No fitted profile.')
(root/'summary.json').write_text(json.dumps(result,indent=2))
for r in rows:
    print(json.dumps(r))
