# GUI for TradingSystemML

This small GUI lets you run training and testing pipelines interactively and saves artifacts under `results/`. 
All required dependencies are auto-installed on first run (or you can set up manually using conda).

Quickstart
---------

### Option 1: Auto-setup (recommended) — one command!

From the repo root, run:

```powershell
py -3 launch_gui.py
```

Or on Windows, double-click `launch_gui.bat`.

Dependencies auto-install if missing. The GUI opens and you're ready to train/test.

### Option 2: Manual setup with conda

1. Install Miniforge from https://github.com/conda-forge/miniforge/releases if you don't have conda.

2. Create and activate the environment:

```powershell
conda env create -f environment.yml
conda activate trading
```

3. Run the GUI:

```powershell
python scripts\gui_app.py
```

### Option 3: Manual setup with pip (if conda unavailable)

1. Ensure Python 3.10+ is available.

2. Install packages:

```powershell
py -3 -m pip install --upgrade pip
py -3 -m pip install tensorflow pandas matplotlib pillow numpy scikit-learn
```

3. Run the GUI:

```powershell
py -3 scripts\gui_app.py
```

Using the GUI
--------------

Once launched, set the desired options on the left panel:
- **Ticker**: stock/crypto ticker (e.g., BTC-USD, AAPL)
- **Model cell**: LSTM or GRU
- **Target**: price or returns
- **Days to predict**: number of steps (e.g., 10)
- **Epochs**: training epochs (e.g., 10)
- **Units**: LSTM/GRU units (e.g., 128)
- **Layers**: number of layers (e.g., 1-2)
- **Dropout**: dropout rate (e.g., 0.2)
- **Bidirectional**: enable/disable bidirectional model

Then press:
- **Train** — run training for selected days
- **Test** — run evaluation and save predictions
- **Train + Test** — run both in sequence

Plot viewing & presets
----------------------

- **Show latest plot**: displays the most recent prediction output inline (prefers `*_preds.csv` for Matplotlib rendering with true vs predicted lines; fallback to PNG if available)
- **Choose plot...**: manually select a PNG to display
- **Save preset** / **Load preset**: save/load parameter presets to/from `presets/` folder
- **Manage presets**: rename or delete saved presets
- Auto-refresh: after training completes, the GUI automatically shows the latest plot

Files & directories
-------------------
- `launch_gui.py` — auto-installer + launcher (one command!)
- `launch_gui.bat` — Windows batch launcher (double-click)
- `environment.yml` — conda environment definition
- `scripts/gui_app.py` — main GUI application
- `presets/` — saved preset JSON files
- `results/` — training outputs and prediction CSVs
- `results/plots/` — prediction plots

Troubleshooting
---------------
- "train_step not available": install TensorFlow or use the auto-launcher (`py -3 launch_gui.py`)
- "matplotlib not available": install with `py -3 -m pip install matplotlib pillow`
- GUI won't open (no display): use the headless demo: `py -3 scripts/gui_demo.py`
- Packages fail to install via pip: use conda: `conda env create -f environment.yml`

Demo script (headless)
----------------------

For a quick headless demo that simulates training and testing without opening a GUI:

```powershell
py -3 scripts/gui_demo.py
```

This writes demo artifacts to `presets/demo_preset.json`, `results/demo_preds.csv`, and `results/gui_test_preds_DEMO_n3.csv`.

Files
-----
- `scripts/gui_app.py` — the GUI implementation
- `presets/` — folder for JSON preset files (created on first save)
- `results/plots/` — prediction plots saved here by the pipeline

Notes
-----
- Pillow (`PIL`) is used when available for more robust PNG handling; the GUI falls back to Tk `PhotoImage` if PIL is absent.
- The GUI uses the existing `train` and `tester` functions from the project; it doesn't reimplement model logic.
