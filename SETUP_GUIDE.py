"""
Visual guide: How to set up and run the Trading GUI

STEP 1: BOOTSTRAP (one-time setup)
==================================

User runs:
  py -3 bootstrap.py
  or double-clicks: bootstrap.bat

bootstrap.py does:
  ├─ Checks: Is conda installed?
  │  ├─ YES → Setup conda env from environment.yml
  │  └─ NO → Ask user to install Miniforge
  │          (or fallback to venv+pip)
  ├─ Downloads + installs packages
  ├─ Creates environment (trading env or .venv)
  └─ Says "Done! Activate environment and run GUI"

Result: Environment is ready with all dependencies


STEP 2: ACTIVATE & LAUNCH (every time you want to use GUI)
==========================================================

Option A (if conda):
  conda activate trading
  python scripts/gui_app.py

Option B (if venv):
  .\.venv\Scripts\Activate.ps1
  python scripts/gui_app.py

Option C (auto-launcher, if env already active or deps present):
  py -3 launch_gui.py


STEP 3: USE GUI
==============

GUI window opens with controls:
  - Set ticker, model type, days, epochs, etc.
  - Click "Train" → runs training
  - Click "Test" → runs evaluation
  - Click "Train + Test" → runs both
  - Results saved to results/ and presets/ folders


FILES INVOLVED
==============

Setup files:
  bootstrap.py           ← User runs this once (orchestrates setup)
  bootstrap.bat          ← Windows users can double-click instead
  environment.yml        ← Conda env definition (Python 3.10 + deps)
  requirements.txt       ← Pip packages (for venv fallback)

GUI & launcher files:
  scripts/gui_app.py     ← Main GUI application
  launch_gui.py          ← Auto-launcher (checks deps, then opens GUI)
  launch_gui.bat         ← Windows wrapper for launcher

Helper files:
  scripts/gui_demo.py    ← Headless demo (mocked training, no GUI window)
  scripts/gui_README.md  ← GUI documentation
  GET_STARTED.md         ← This file (beginner-friendly guide)
  README.md              ← Project README with setup info

Results directories (created automatically):
  results/               ← Training outputs, predictions
  presets/               ← Saved GUI presets (JSON)


DEPENDENCY INSTALLATION LOGIC (bootstrap.py)
==============================================

┌─────────────────────────────────────┐
│ bootstrap.py starts                 │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ Check: conda available?             │
└─────────────────────────────────────┘
      ↙                            ↖
    YES                           NO
      ↓                            ↓
   ┌──────────────┐      ┌────────────────────┐
   │ Use conda    │      │ Ask: Install       │
   │ env          │      │ Miniforge? [y/n]   │
   └──────────────┘      └────────────────────┘
      ↓                      ↙            ↖
      │                     YES          NO
      │                      ↓            ↓
      │                  ┌─────────┐  ┌──────────┐
      │                  │ Download│  │ Fallback │
      │                  │ Install │  │ to venv  │
      │                  │Miniforge│  │ + pip    │
      │                  └─────────┘  └──────────┘
      │                      ↓            ↓
      └──────────────────────┴────────────┘
                    ↓
        ┌──────────────────────┐
        │ Create env &         │
        │ Install packages     │
        │ (TF, pandas, etc)    │
        └──────────────────────┘
                ↓
        ┌──────────────────────┐
        │ Print instructions   │
        │ Activate env &       │
        │ Run GUI              │
        └──────────────────────┘


QUICK REFERENCE
===============

For the impatient:
  1. py -3 bootstrap.py
  2. Follow prompts (usually just press Enter)
  3. When done: conda activate trading (or .\.venv\Scripts\Activate.ps1)
  4. python scripts/gui_app.py


For conda experts:
  conda env create -f environment.yml
  conda activate trading
  python scripts/gui_app.py


For pip + venv experts:
  py -3.10 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  python scripts/gui_app.py


For CI/CD or headless (no GUI window):
  py -3 scripts/gui_demo.py
  (produces demo_preds.csv and demo_preset.json)
"""

print(__doc__)
