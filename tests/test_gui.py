import os
import json
import unittest
from pathlib import Path

import scripts.gui_app as gui


class TestGUI(unittest.TestCase):
    def setUp(self):
        # ensure presets dir exists
        os.makedirs('presets', exist_ok=True)
        # instantiate app (do not call mainloop)
        self.app = gui.App()

    def tearDown(self):
        try:
            self.app.destroy()
        except Exception:
            pass

    def test_import_and_instantiation(self):
        self.assertTrue(hasattr(self.app, 'ticker_var'))

    def test_preset_save_and_load(self):
        # create a sample preset file
        preset = {
            'ticker': 'TESTCO',
            'cell': 'GRU',
            'target': 'price',
            'days': 5,
            'epochs': 2,
            'units': 32,
            'layers': 1,
            'dropout': 0.1,
            'bidirectional': False
        }
        fname = os.path.join('presets', 'test_preset.json')
        with open(fname, 'w') as f:
            json.dump(preset, f)

        # patch filedialog to return this file path
        orig = gui.filedialog.askopenfilename
        gui.filedialog.askopenfilename = lambda **kwargs: fname
        try:
            self.app._load_preset()
            self.assertEqual(self.app.ticker_var.get(), 'TESTCO')
            self.assertEqual(self.app.cell_var.get(), 'GRU')
            self.assertEqual(self.app.target_var.get(), 'price')
            self.assertEqual(self.app.days_var.get(), 5)
        finally:
            gui.filedialog.askopenfilename = orig

    def test_display_preds_plot_smoke(self):
        # create a tiny preds CSV
        os.makedirs('results', exist_ok=True)
        csv_path = os.path.join('results', 'test_preds.csv')
        with open(csv_path, 'w') as f:
            f.write('anchor,true_price,pred_price\n')
            f.write('100,101,102\n')
            f.write('101,102,103\n')
        # should not raise
        self.app._display_plot_from_preds(csv_path)


if __name__ == '__main__':
    unittest.main()
