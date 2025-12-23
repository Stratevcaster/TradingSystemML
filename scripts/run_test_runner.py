from pathlib import Path
import sys
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
import tester
import parameters

print('Calling tester.test with N_DAYS_STEP=', parameters.N_DAYS_STEP)
res = tester.test(parameters.N_DAYS_STEP)
print('tester.test returned (length):', len(res))
print('Values:', res)
