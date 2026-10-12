"""Run repository and new domain tests with a portable UTF-8 evidence log."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
output=ROOT/'TokenSim/validation/results/full_regression.log'
output.parent.mkdir(exist_ok=True)
with output.open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
    suite=unittest.TestLoader().discover(str(ROOT/'tests'),pattern='test_*.py',top_level_dir=str(ROOT))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName('TokenSim.validation.test_communication_domains'))
    result=unittest.TextTestRunner(stream=log,verbosity=1).run(suite)
print(f'{result.testsRun} tests; failures={len(result.failures)}, errors={len(result.errors)}; {output}')
sys.exit(not result.wasSuccessful())
