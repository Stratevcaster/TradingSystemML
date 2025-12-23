"""Demo runner for `scripts/gui_app.py` that uses mocked train/test functions to
exercise the GUI flows programmatically (save/load preset, train, test, show plot).

This script is intended to run in CI or headless environments for reproducible demos.
"""
import os
import time
import json
import threading

import importlib.util
import sys

# Import gui_app by file path to avoid package/import issues in minimal test env
spec = importlib.util.spec_from_file_location('gui_app', os.path.join(os.path.dirname(__file__), 'gui_app.py'))
gui = importlib.util.module_from_spec(spec)
sys.modules['gui_app'] = gui
spec.loader.exec_module(gui)

# Ensure results and presets exist
os.makedirs('results', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)
os.makedirs('presets', exist_ok=True)

# Mock implementations

def mock_train_step(step, model_name):
    # Simulate some work and write a preds CSV that the GUI will pick up
    out = os.path.join('results', 'demo_preds.csv')
    with open(out, 'w') as fh:
        fh.write('anchor,true_price,pred_price\n')
        # write a small incremental series
        for i in range(10):
            fh.write(f"{100+i},{100+i+1},{100+i+1.5}\n")
    print(f"[mock_train_step] wrote {out} (model_name={model_name})")
    time.sleep(0.1)


def mock_tester_test(days):
    # Return simple preds (list of lists) and write a CSV file
    preds = [[100 + i, 101 + i] for i in range(days)]
    out = os.path.join('results', f'gui_test_preds_DEMO_n{days}.csv')
    with open(out, 'w') as fh:
        fh.write('anchor,pred\n')
        for row in preds:
            fh.write(','.join(map(str, row)) + '\n')
    print(f"[mock_tester_test] wrote {out}")
    return preds

# Inject mocks into module
gui.train_step = mock_train_step
gui.tester_test = mock_tester_test

print('Instantiating GUI App (headless demo, no mainloop)')
app = gui.App()

# Set some UI parameters
app.ticker_var.set('DEMO')
app.cell_var.set('GRU')
app.target_var.set('price')
app.days_var.set(3)
app.epochs_var.set(1)
app.units_var.set(16)
app.layers_var.set(1)
app.dropout_var.set(0.1)
app.bidirectional_var.set(False)

# Save a preset by monkeypatching filedialog to return a known filename
preset_path = os.path.abspath(os.path.join('presets', 'demo_preset.json'))
orig_save = gui.filedialog.asksaveasfilename
gui.filedialog.asksaveasfilename = lambda **kwargs: preset_path
print('Saving preset to', preset_path)
app._save_preset()
# restore
gui.filedialog.asksaveasfilename = orig_save

# Now load the preset by returning the same path
orig_open = gui.filedialog.askopenfilename
gui.filedialog.askopenfilename = lambda **kwargs: preset_path
print('Loading preset from', preset_path)
app._load_preset()
gui.filedialog.askopenfilename = orig_open

# Trigger training (runs in background thread)
print('Starting training (mock)')
# The GUI _on_train implementation creates a small Toplevel progress dialog which
# may not work reliably from a headless demo running outside the mainloop. To
# keep the demo deterministic and reproducible, run the mocked training loop
# directly here instead of invoking `app._on_train()`.
days = int(app.days_var.get())
for step in range(1, days + 1):
    model_name = f"demo_model_step_{step}"
    print(f"[demo] training step {step}/{days} -> {model_name}")
    gui.train_step(step, model_name)

# After training, trigger GUI to show latest preds
try:
    app._show_latest_plot()
except Exception as e:
    print('Failed to show latest plot in demo:', e)

print('Training (mock) completed')

# The GUI should have auto-refreshed latest plot (from demo_preds.csv). Verify file exists
preds_file = os.path.join('results', 'demo_preds.csv')
print('Preds file exists:', os.path.exists(preds_file))
if os.path.exists(preds_file):
    print('Preds file sample:')
    print(open(preds_file).read())

# Trigger test
print('Starting test (mock)')
app._on_test()
while threading.active_count() > 1:
    time.sleep(0.1)
print('Test (mock) completed')

# List results directory
print('\nResults dir contents:')
for f in sorted(os.listdir('results')):
    print(' -', f)

# Clean-up: destroy app
try:
    app.destroy()
except Exception:
    pass

print('\nDemo finished successfully')
