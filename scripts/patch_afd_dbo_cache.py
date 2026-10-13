"""Apply a narrowly scoped DBO fix to a NEW sibling plugin copy only."""
import argparse
import difflib
import hashlib
import json
import shutil
from pathlib import Path

OLD_GUARD = '''def _yield_if_dbo_enabled() -> None:
    try:
        from afd_plugin.v1.worker.npu.ubatching import (
            dbo_enabled as ascend_dbo_enabled,
        )
        from afd_plugin.v1.worker.npu.ubatching import (
            dbo_yield as ascend_dbo_yield,
        )
    except ImportError:
        ascend_dbo_enabled = None
        ascend_dbo_yield = None

    if (
        ascend_dbo_enabled is not None
        and ascend_dbo_yield is not None
        and ascend_dbo_enabled()
    ):
        ascend_dbo_yield()
        return

    if dbo_enabled():
        dbo_yield()
'''

NEW_GUARD = '''@lru_cache(maxsize=1)
def _get_ascend_dbo_helpers():
    # Backend availability is process-stable; never cache dynamic DBO state.
    try:
        from afd_plugin.v1.worker.npu.ubatching import (
            dbo_enabled as ascend_dbo_enabled,
        )
        from afd_plugin.v1.worker.npu.ubatching import (
            dbo_yield as ascend_dbo_yield,
        )
    except ImportError:
        return None, None
    return ascend_dbo_enabled, ascend_dbo_yield


def _yield_if_dbo_enabled() -> None:
    ascend_dbo_enabled, ascend_dbo_yield = _get_ascend_dbo_helpers()
    if (
        ascend_dbo_enabled is not None
        and ascend_dbo_yield is not None
        and ascend_dbo_enabled()
    ):
        ascend_dbo_yield()
        return

    if dbo_enabled():
        dbo_yield()
'''


def patch_source(source):
    if source.count(OLD_GUARD) != 1 or source.count('import torch\n') != 1:
        raise ValueError('DBO source differs from reviewed version; refusing to patch')
    if '_get_ascend_dbo_helpers' in source:
        raise ValueError('DBO helper already present')
    return source.replace('import torch\n', 'from functools import lru_cache\n\nimport torch\n', 1).replace(OLD_GUARD, NEW_GUARD, 1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    args=p.parse_args()
    source,dest=args.source.resolve(),args.destination.resolve()
    if source == dest or dest.parent != source.parent or dest.exists():
        raise ValueError('Destination must be a new sibling directory')
    rel=Path('afd_plugin/v1/worker/dbo.py')
    original=(source/rel).read_bytes()
    before=original.decode('utf-8')
    after=patch_source(before)
    compile(after,str(dest/rel),'exec')
    shutil.copytree(source,dest,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc','.venv'))
    (dest/rel).write_text(after,encoding='utf-8')
    assert (source/rel).read_bytes()==original
    manifest=dict(source=str(source),destination=str(dest),changed_file=str(rel),
                  original_sha256=hashlib.sha256(original).hexdigest(),
                  patched_sha256=hashlib.sha256((dest/rel).read_bytes()).hexdigest(),
                  scope='cache optional backend resolution only; dynamic enabled checks unchanged')
    (dest/'dbo-fix-manifest.json').write_text(json.dumps(manifest,indent=2))
    (dest/'dbo-cache.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
                                            fromfile='a/'+str(rel),tofile='b/'+str(rel))))
    print(json.dumps(manifest))


if __name__=='__main__':
    main()
