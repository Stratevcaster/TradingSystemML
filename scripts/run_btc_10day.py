"""Run a short training and test for BTC with a 10-day horizon.

Usage:
  python scripts/run_btc_10day.py [EPOCHS]

This script will:
 - set `parameters.ticker` to `BTC-USD`
 - set `parameters.N_DAYS_STEP` to 10
 - run a short training (uses `scripts/run_real_train.py` main())
 - run the test pipeline (`tester.test(10)`) and print the predicted prices
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import parameters
from scripts import run_real_train
import tester

def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    # Configure for BTC 10-day horizon
    parameters.ticker = 'BTC-USD'
    parameters.N_DAYS_STEP = 10
    print(f"Running short BTC training: epochs={epochs}, ticker={parameters.ticker}, N_DAYS_STEP={parameters.N_DAYS_STEP}")
    # Call the short training script with appropriate argv
    old_argv = sys.argv[:]
    sys.argv = [sys.argv[0], str(epochs), parameters.ticker]
    try:
        run_real_train.main()
    finally:
        sys.argv = old_argv

    # Run the test pipeline (will use parameters.N_DAYS_STEP=10 now)
    print("Running tester.test for 10 days ahead...")
    preds = tester.test(parameters.N_DAYS_STEP)
    print("Predicted prices for next 10 days:", preds)

if __name__ == '__main__':
    main()
