"""Prepare a new GPU2/3 benchmark using only the isolated fixed plugin."""
import json
from pathlib import Path

base=Path('/home/youwei/autodl-tmp/af-workspace')
source=base/'profiling/afd-gpu23-rerun-20261007'
target=base/'profiling/afd-dbo-fixed-20261007'
plugin=base/'src/afd-plugin-dbo-cache-20261007'
assert (plugin/'dbo-fix-manifest.json').exists()
target.mkdir(exist_ok=False)
code=(source/'run_curve.py').read_text()
assert 'src/afd-plugin' in code
code=code.replace('src/afd-plugin','src/afd-plugin-dbo-cache-20261007')
code=code.replace('17839','18039').replace('17840','18040').replace('18893','19093')
(target/'run_curve.py').write_text(code)
code=(source/'attention.py').read_text()
old="    cfg = json.loads((directory/'case.json').read_text())"
extra="""    from afd_plugin.v1.worker import dbo
    actual_plugin = Path(dbo.__file__).resolve()
    assert 'afd-plugin-dbo-cache-20261007' in actual_plugin.parts, actual_plugin
    (directory/'loaded-plugin.json').write_text(json.dumps(dict(dbo_file=str(actual_plugin))))
"""
assert old in code
(target/'attention.py').write_text(code.replace(old,extra+old))
(target/'manifest.json').write_text((plugin/'dbo-fix-manifest.json').read_text())
print(target)
