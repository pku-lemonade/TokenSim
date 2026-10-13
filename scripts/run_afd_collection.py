"""Execute prepared cases sequentially; GPU idle checks live in each case driver."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_cases(manifest,phase,profile):
    heldout=phase.endswith('heldout')
    if heldout and profile is None:
        raise ValueError('freeze a profile before running any held-out batch4 case')
    mode='capture' if phase.startswith('capture-') else 'timing'
    return [case for case in manifest['cases'] if case['mode']==mode and (case['batch']==4)==heldout]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--phase',choices=['train','heldout','capture-train','capture-heldout'],required=True)
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    cases=selected_cases(manifest,args.phase,args.profile)
    root=args.manifest.resolve().parent
    frozen=None
    if args.profile is not None:
        document=json.loads(args.profile.read_text())
        if document.get('schema_version')!=1 or document.get('contains_wait') is not False:
            raise ValueError('invalid frozen host profile')
        frozen=dict(sha256=sha(args.profile),path=str(args.profile.resolve()),profile=document)
    if args.dry_run:
        print(json.dumps(dict(phase=args.phase,cases=len(cases),profile_sha256=frozen['sha256'] if frozen else None),indent=2))
        return
    lock=root/'collection.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        if frozen:
            freeze=root/'frozen-profile.json'
            if freeze.exists():
                if json.loads(freeze.read_text())['sha256']!=frozen['sha256']:
                    raise ValueError('held-out experiment already bound to another profile; prepare a fresh experiment')
            else:
                with freeze.open('x') as stream:json.dump(frozen,stream,indent=2)
        for case in cases:
            directory=Path(case['command'][1]).parent
            if root not in directory.resolve().parents:
                raise ValueError('case escapes prepared experiment')
            for name,expected in case['driver_hashes'].items():
                if sha(directory/name)!=expected:raise ValueError('prepared driver drift')
            plugin=Path(case['environment']['AFD_ISOLATED_PLUGIN'])
            for name,expected in manifest['isolated_hashes'].items():
                if sha(plugin/name)!=expected:raise ValueError('isolated plugin drift')
            if sha(plugin/'afd_capture_hooks.py')!=manifest['hooks_sha256']:
                raise ValueError('capture hook drift')
            env=os.environ.copy()
            for name in list(env):
                if name.startswith('AFD_CAPTURE_'):env.pop(name)
            env.update(case['environment'])
            subprocess.run(case['command'],env=env,check=True)
            if frozen and sha(args.profile)!=frozen['sha256']:
                raise ValueError('profile changed during held-out run')
    finally:
        lock.unlink()


if __name__=='__main__':main()
