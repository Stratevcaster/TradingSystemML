# TradingSystemML

## ⚡ Fastest Setup: One-Click Bootstrap

If you don't have conda or a virtual environment set up, run the bootstrap script:

**Windows PowerShell:**
```powershell
py -3 bootstrap.py
```

Or double-click `bootstrap.bat` on Windows.

This will:
1. Check if conda is installed; if not, offer to install Miniforge.
2. Or, fall back to creating a Python 3.10 venv + pip setup.
3. Guide you through the steps with clear prompts.

---

## Quick Start: Launch the GUI 🚀

After bootstrap completes (or if you already have an environment):

**If you just ran bootstrap with conda:**
```powershell
conda activate trading
python scripts/gui_app.py
```

**If you just ran bootstrap with venv:**
```powershell
.\.venv\Scripts\Activate.ps1
python scripts/gui_app.py
```

**Auto-launcher (if Python 3.10/3.11):**
```powershell
py -3 launch_gui.py
```

---

## Manual Setup (advanced)

### Option 1: With conda (if conda already installed)
```powershell
conda create -n trading -y --file environment.yml
conda activate trading
python scripts/gui_app.py
```

### Option 2: With pip + venv (Python 3.10/3.11)
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/gui_app.py
```

### Option 3: Auto-launcher (attempts auto-install)
```powershell
py -3 launch_gui.py
```

---

## Setup (for development)

If you prefer a manual venv setup (for development):

1. Create a virtual environment (Windows PowerShell):

```powershell
py -3.10 -m venv .venv
```

2. Activate the venv (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Run training or testing:

```powershell
python .\orquestratorTrain.py
python .\orquestadorTest.py
```

Or launch the GUI:
```powershell
python scripts/gui_app.py
```

If you prefer `cmd.exe` for activation use `.venv\Scripts\activate.bat`.

---

## Troubleshooting

### Python version incompatibility

If you're on Python 3.12+, TensorFlow may not have wheels available on PyPI. The **bootstrap script** handles this automatically by:
- Installing Miniforge (which provides prebuilt TensorFlow packages for Python 3.10/3.11), or
- Using Python 3.10 with pip (if you install it separately).

If you manually encounter a "no matching distribution for tensorflow" error, use the bootstrap script or install Python 3.10/3.11.

---

## GUI Usage

See `scripts/gui_README.md` for full GUI documentation and features.


### Conda setup (preferred for TensorFlow) 🐍🔧

If you want to use conda (recommended for easier TensorFlow installation), install Miniconda or Miniforge first (https://docs.conda.io/en/latest/miniconda.html or https://github.com/conda-forge/miniforge).

After conda is available, run:

```powershell
conda create -n trading python=3.11 -y
conda activate trading
conda install -c conda-forge tensorflow -y
pip install -r requirements.txt
```

This will install a TensorFlow build compatible with the selected Python version and then install the remaining dependencies with pip.

---

Automated setup script (Bash) 🧰

I've added a Bash script that automates Miniforge/conda + environment setup for this project:

* `scripts/setup_conda_env.sh` — downloads and installs Miniforge (if missing), creates a conda env, installs TensorFlow from `conda-forge`, and installs the project's `requirements.txt`.

Usage (Unix, macOS, or WSL/Git-Bash on Windows):

```bash
chmod +x scripts/setup_conda_env.sh
./scripts/setup_conda_env.sh trading 3.11
```

The script is idempotent with respect to Miniforge and will create the `trading` environment (or whatever name you pass as the first argument).

Notes:
- On native Windows PowerShell you can follow the `Conda setup` section above instead, or run the script inside WSL/Git-Bash.
- For native Windows, there's now a PowerShell setup script that automates the Windows-specific steps (Miniforge install, Visual C++ redistributable install, conda env creation and package installs):

	* `scripts/setup_windows_env.ps1` — use PowerShell (preferably run as Administrator) to install Miniforge (if missing), install the Visual C++ 2015-2022 Redistributable (x64) silently, create the `trading` env, install TensorFlow and other dependencies, and run a quick TensorFlow import test.

	Usage (PowerShell, run as Administrator):

	```powershell
	Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
	.\scripts\setup_windows_env.ps1 -EnvName trading -PythonVersion 3.11
	```

	The script prints progress and runs `scripts/tf_import_test.py` at the end to verify TensorFlow imports correctly.
- The script will print status and will exit on errors (useful for CI or automated setups).

---

Quick smoke tests (no network) 🚀

To quickly validate the training and prediction pipeline without downloading market data or training large models, use the included quick scripts which use synthetic data:

```powershell
# Train a tiny model (1 epoch, small network)
conda activate trading
python scripts/run_quick_train.py

# Run a prediction using the trained quick model
python scripts/run_quick_test.py
```

These scripts are useful for CI, debugging, or confirming your environment is correctly configured for TensorFlow and Keras.

---

Full (short) training example — real market data (3 epochs) 🔁

You can run a short real training job (uses real market data via `yahoo_fin`) to verify the full
end-to-end training loop works. This will download market data and train for a small number of
epochs so it finishes quickly.

```powershell
conda activate trading
python scripts/run_real_train.py 3   # runs 3 epochs
```

Notes & cautions:
- This downloads data using `yahoo_fin` (network required). If Yahoo returns an empty response the script will raise an error — retry after a short while or check your network.
- The full orchestrator (`orquestratorTrain.py`) runs multiple steps and uses the repository `parameters.py`. For quick testing prefer `scripts/run_real_train.py` which overrides `EPOCHS` to a short value.

Behavior when market data cannot be fetched
------------------------------------------

If `scripts/run_real_train.py` cannot fetch market data from `yahoo_fin` (or the `yfinance` fallback
fails), the script will automatically **fall back to a synthetic dataset** and continue training. This
ensures the pipeline (data preparation, model creation, training and saving) can be verified end-to-end
even when your machine or environment cannot reach the data provider (this is what I used when theFD
remote data source returned an empty response during testing).

When the synthetic fallback is used, the saved model filename will include `-synthetic-fallback` to
make it clear in `./results` that the run used generated data (e.g. `...-synthetic-fallback.h5`).

After running the short real training, the model will be saved under `./results` with a name matching the project's naming format (timestamp, ticker, config). You can then run the test orchestration or the quick test script to exercise prediction logic.

```powershell
# Example: run the test flow (this will attempt to fetch market-data for testing)
python .\orquestadorTest.py

# Or use the synthetic prediction quick test (no network)
python scripts/run_quick_test.py
```

GUI (Tkinter)
-------------

There is a simple Tkinter GUI available at `scripts/gui_app.py` that provides an easy way to run training and testing flows from a desktop UI. It supports:

- Selecting `ticker`, model type (`LSTM`/`GRU`), `target` (`price`/`returns`), days to predict, epochs, units, layers and dropout.
- Buttons: **Train**, **Test**, **Train + Test**, **Show latest plot**, **Choose plot...**, **Save preset**, **Load preset**, **Open results folder**.
- Inline plot viewing of prediction images saved in `results/plots/` and saving/loading presets in `presets/`.
 - Inline plot viewing of prediction images saved in `results/plots/` and rendering of prediction CSVs (`*_preds.csv`) with Matplotlib inline in the GUI.
 - Presets manager (rename/delete) and auto-refresh: the GUI can automatically show the latest prediction plot when training completes and includes a preset manager for rename/delete of saved presets.

Run it from your trading conda env:

```powershell
& "$env:USERPROFILE\miniforge3\envs\trading\python.exe" scripts/gui_app.py
```

See `scripts/gui_README.md` for a short guide and examples.

---

Changelog (what I changed) 📝

- Added `requirements.txt` containing runtime dependencies.
- Added setup scripts:
	- `scripts/setup_conda_env.sh` (Bash / WSL / macOS)
	- `scripts/setup_windows_env.ps1` (PowerShell for Windows; installs VC redistributable)
- Added quick smoke scripts (synthetic, no network):
	- `scripts/run_quick_train.py` — trains a tiny model for 1 epoch to exercise the pipeline
	- `scripts/run_quick_test.py` — runs prediction using the quick model
- Added diagnostic script: `scripts/tf_import_test.py` (checks TF import)
- Added `scripts/run_real_train.py` — runs a short realistic training (3 epochs default) using real market data
- Fixed runtime issues:
	- Removed accidental test-only import (`pandas.tests.frame.test_validate`) that required `pytest`.
	- Handled model-loading and inverse-transform edge cases in quick test.

Makefile / convenience targets

I added a `Makefile` with common convenience targets so you can run the main flows from one place (requires `make`):

```bash
make help
make setup-conda            # run Bash setup (Miniforge + env creation)
make setup-windows          # run PowerShell setup (Windows)
make install-reqs           # pip install -r requirements.txt into the conda env
make tf-test                # run TensorFlow import diagnostic
make quick-train            # run synthetic quick training
make quick-test             # run synthetic quick prediction
make real-train EPOCHS=3 TICKER=AAPL  # run short real training
```

On Windows you can run `make` from WSL, Git-Bash, or any environment which provides GNU `make`; otherwise run the individual scripts in the `scripts/` folder directly from PowerShell.

