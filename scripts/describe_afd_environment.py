"""CPU-only fingerprint of the prepared plugin, drivers and installed software."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('manifest',type=Path)
    args=parser.parse_args()
    root=args.manifest.resolve().parent
    manifest=json.loads(args.manifest.read_text())
    versions={name:importlib.metadata.version(name) for name in ('vllm','torch','transformers')}
    evidence=dict(versions=versions,python=platform.python_version(),
        isolated_hashes=manifest['isolated_hashes'],hooks_sha256=manifest['hooks_sha256'],
        driver_hashes=manifest['cases'][0]['driver_hashes'])
    fingerprint=hashlib.sha256(json.dumps(evidence,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    environment=dict(execution_mode='eager',software_sha256=fingerprint,
        vllm_version=versions['vllm'],torch_version=versions['torch'],
        gpu_pair='GPU-8df5e331-8f9a-3cdc-af30-42b94f908018,GPU-76ca30e6-17c9-9a38-cb06-57aebeaa6973')
    for name,document in (('environment-evidence.json',evidence),('environment.json',environment)):
        with (root/name).open('x',encoding='utf-8') as stream:json.dump(document,stream,indent=2)
    print(json.dumps(environment))


if __name__=='__main__':main()
