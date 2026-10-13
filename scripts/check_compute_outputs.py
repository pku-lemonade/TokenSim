"""Compare generated tokens server-side; emit no full token/trace arrays."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
controls = {g: json.loads((root/f'control-gpu{g}/outputs.json').read_text()) for g in (0, 3)}
rows = []
for batch in (1, 8, 4):
    outputs = json.loads((root/f'c{batch}/outputs.json').read_text())
    calls = json.loads((root/f'c{batch}/calls.json').read_text())
    reference = outputs[0]
    matches = {f'gpu{g}-r{r}': controls[g][f'b{batch}-r{r}'] == reference
               for g in controls for r in range(3)}
    rows.append(dict(batch=batch, afd_repetitions=len(outputs), afd_repeat_equal=all(o == reference for o in outputs),
                     control_equal_to_afd=matches,
                     token_sha256=hashlib.sha256(json.dumps(reference).encode()).hexdigest(),
                     unprofiled_engine_decode_median_us=statistics.median(v*1000 for c in calls for v in c['step_ms'][1:])))
files = [root/'run_curve.py', root/'attention.py', root.parent/'run_native_compute_control.py',
         root.parent/'summarize_compute_alignment.py', root.parent/'check_compute_outputs.py']
result = dict(rows=rows, script_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(root/'validation-summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
