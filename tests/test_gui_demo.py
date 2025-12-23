import os
import subprocess
import sys
import unittest

class TestGUIDemo(unittest.TestCase):
    def test_demo_runs_and_creates_artifacts(self):
        # Run the demo script (should be fast and deterministic)
        rv = subprocess.run([sys.executable, 'scripts/gui_demo.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(rv.stdout)
        self.assertEqual(rv.returncode, 0, msg=f"Demo failed: {rv.stderr}")
        # Check preset file
        self.assertTrue(os.path.exists('presets/demo_preset.json'))
        # Check demo preds
        self.assertTrue(os.path.exists('results/demo_preds.csv'))
        # Check test preds file
        self.assertTrue(any('gui_test_preds_DEMO' in f for f in os.listdir('results')))

if __name__ == '__main__':
    unittest.main()
