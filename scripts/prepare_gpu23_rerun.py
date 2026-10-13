"""Copy the known working original GPU2/3 driver into a fresh experiment."""
import hashlib
import json
from pathlib import Path

base = Path('/home/youwei/autodl-tmp/af-workspace')
source = base/'profiling/afd-alignment-20261006'
target = base/'profiling/afd-gpu23-rerun-20261007'
target.mkdir(exist_ok=False)
for name in ('run_curve.py', 'attention.py'):
    code = (source/name).read_text().replace('17639','17839').replace('17640','17840').replace('18693','18893')
    (target/name).write_text(code)
(target/'manifest.json').write_text(json.dumps(dict(source=str(source), gpus=[2,3],
    scope='original NVLink pair; tiny Qwen decode comparison; no parameter changes',
    driver_sha256={name: hashlib.sha256((target/name).read_bytes()).hexdigest() for name in ('run_curve.py','attention.py')}), indent=2))
print(target)
