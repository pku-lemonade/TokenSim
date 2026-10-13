"""Verify isolated plugin change and output parity against original GPU2/3."""
import hashlib
import json
from pathlib import Path

base=Path('/home/youwei/autodl-tmp/af-workspace')
original=base/'src/afd-plugin'
fixed=base/'src/afd-plugin-dbo-cache-20261007'
manifest=json.loads((fixed/'dbo-fix-manifest.json').read_text())
assert hashlib.sha256((original/manifest['changed_file']).read_bytes()).hexdigest()==manifest['original_sha256']
differences=[]
for path in (original/'afd_plugin').rglob('*.py'):
    rel=path.relative_to(original)
    if path.read_bytes()!=(fixed/rel).read_bytes(): differences.append(str(rel))
assert differences==['afd_plugin/v1/worker/dbo.py'], differences
rows=[]
for batch in (1,8,4):
    case=base/f'profiling/afd-dbo-fixed-20261007/c{batch}'
    reference=json.loads((base/f'profiling/afd-gpu23-rerun-20261007/c{batch}/outputs.json').read_text())[0]
    outputs=json.loads((case/'outputs.json').read_text())
    assert all(o==reference for o in outputs)
    actual=json.loads((case/'loaded-plugin.json').read_text())['dbo_file']
    assert Path(actual)==fixed/manifest['changed_file']
    rows.append(dict(batch=batch,repeats=len(outputs),outputs_equal=True,loaded_dbo=actual))
result=dict(manifest=manifest,original_unchanged=True,changed_python_files=differences,rows=rows,
            cpu_guard_after=json.loads((fixed/'dbo-cpu-after.json').read_text()))
out=base/'profiling/afd-dbo-fixed-20261007/fix-audit.json'
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
