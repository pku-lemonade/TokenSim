"""Finish an already launched A800 calibration job, then run CPU curves.

Waits only for the two specific measurement PIDs in this output directory.
It never stops other jobs and is not an availability monitor.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu-python', type=Path, required=True)
    parser.add_argument('--cpu-python', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root/'finish.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def status(stage, **extra):
        value = dict(stage=stage, pid=os.getpid(), updated=time.strftime('%Y-%m-%d %H:%M:%S %z'), **extra)
        temporary = root/'pipeline-status.json.new'
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(root/'pipeline-status.json')
        print(json.dumps(value), flush=True)

    try:
        status('waiting_for_own_compute_measurements')
        deadline = time.monotonic()+2700
        for role in ('A', 'F'):
            directory = root/('compute-'+role)
            pid = json.loads((directory/'launch.json').read_text())['pid']
            while Path(f'/proc/{pid}').exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('Compute job did not finish within 45 minutes')
                time.sleep(5)
            contract = json.loads((directory/'measurement-contract.json').read_text())
            cases = json.loads((directory/'cases.json').read_text())
            assert contract.get('failed') == 0 and contract.get('completed') == len(cases), role+' measurements incomplete'
        scripts = root/'source/scripts'
        status('measuring_two_gpu_p2p')
        with (root/'communication.log').open('w') as log:
            subprocess.run([str(args.gpu_python), str(scripts/'measure_mla_p2p.py'),
                '--root', str(root/'communication'), '--gpus',
                'GPU-dfe6761a-23f4-c34c-9c03-01da02020e5f', 'GPU-ce81ab13-d8a0-a49d-c908-f876b8eb2087'],
                stdout=log, stderr=subprocess.STDOUT, check=True, cwd=root/'source')
        status('running_cpu_tokensim_curves')
        with (root/'simulation.log').open('w') as log:
            subprocess.run([str(args.cpu_python), str(scripts/'run_mla_measured_curves.py'), '--root', str(root)],
                stdout=log, stderr=subprocess.STDOUT, check=True, cwd=root/'source')
        status('complete', summary='simulation/summary.json', figure='simulation/curves.png')
    except Exception:
        status('failed', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
