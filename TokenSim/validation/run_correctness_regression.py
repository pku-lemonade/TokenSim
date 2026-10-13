"""Run current correctness regressions with explicit workspace/snapshot scope."""
from pathlib import Path
import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--label', default='workspace')
    parser.add_argument('--specialists', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    temp = source/'.correctness-test-temp'
    temp.mkdir(exist_ok=True)
    env = dict(os.environ, TMP=str(temp), TEMP=str(temp), PYTHONUTF8='1')
    output = source/'results'/f'communication_correctness_{args.label}.log'
    output.parent.mkdir(exist_ok=True)
    with output.open('w',encoding='utf-8') as log:
        log.write(f'Validation root: {args.root.resolve()}\n')
        log.flush()
        suites = [['TokenSim.validation.test_communication_domains','TokenSim.validation.test_communication_correctness',
                   'TokenSim.validation.test_placement_correctness']]
        if not args.specialists:
            suites.insert(0,['discover','-s','tests','-t','.'])
        for suite in suites:
            result = subprocess.run(['python','-X','utf8','-B','-m','unittest',*suite],
                                    cwd=args.root,env=env,stdout=log,stderr=log)
            log.flush()
            if result.returncode:
                raise SystemExit(result.returncode)
    print('Passed:',output)


if __name__ == '__main__':
    main()
