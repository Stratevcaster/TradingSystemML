# Get Started — Trading System GUI

## TL;DR: Three steps to launch the GUI

1. **Open PowerShell** in the repo folder (`c:\Users\Yani\TradingSystemML`)

2. **Run the bootstrap script:**
   ```powershell
   py -3 bootstrap.py
   ```
   Or double-click `bootstrap.bat`

3. **Follow the prompts** — bootstrap will:
   - Install Miniforge (conda) if needed, OR
   - Set up Python 3.10 venv + pip as fallback
   - Install all dependencies automatically

4. **Activate environment and launch GUI:**
   ```powershell
   # If conda was set up:
   conda activate trading
   python scripts/gui_app.py
   
   # If venv was set up:
   .\.venv\Scripts\Activate.ps1
   python scripts/gui_app.py
   ```

---

## What is bootstrap.py?

`bootstrap.py` is a one-command setup that handles all environment setup for you:

- ✅ Detects if conda (Miniforge) is installed
- ✅ If not installed, downloads and installs Miniforge automatically
- ✅ Falls back to Python 3.10 + venv + pip if conda install fails
- ✅ Installs all dependencies (TensorFlow, pandas, matplotlib, etc.)
- ✅ Guides you through each step with clear messages

**Why?** TensorFlow doesn't have wheels for Python 3.12+, so bootstrap handles version compatibility automatically.

---

## Common scenarios

### Scenario A: You have Python 3.10 or 3.11 locally
- Run `py -3 bootstrap.py`
- Choose fallback venv option when prompted
- Bootstrap creates `.venv` and installs packages via pip
- Done! Activate venv and run the GUI

### Scenario B: You have Python 3.12+ (no older Python)
- Run `py -3 bootstrap.py`
- Choose to install Miniforge when prompted
- Bootstrap downloads and installs Miniforge (one-time)
- Miniforge installs with Python 3.10 + TensorFlow prebuilt
- Close and reopen PowerShell, then run bootstrap again
- Done! Use conda environment

### Scenario C: Miniforge/Conda already installed
- Run `py -3 bootstrap.py`
- Bootstrap detects conda and skips to env creation
- Done! Activates trading env automatically

---

## Files provided

| File | Purpose |
|------|---------|
| `bootstrap.py` | Main setup script (run this!) |
| `bootstrap.bat` | Windows wrapper (double-click alternative) |
| `launch_gui.py` | Auto-launcher once environment is ready |
| `launch_gui.bat` | Windows wrapper for launcher |
| `environment.yml` | Conda environment definition (Python 3.10 + all deps) |
| `requirements.txt` | Pip packages list (for venv option) |

---

## Next: Use the GUI

Once bootstrap finishes and environment is activated:

```powershell
python scripts/gui_app.py
```

The GUI window opens. Set parameters (ticker, model, days, epochs, etc.) and click:
- **Train** → Run training
- **Test** → Run evaluation
- **Train + Test** → Both sequentially

See `scripts/gui_README.md` for full GUI documentation.

---

## Troubleshooting

**Q: Bootstrap says "conda not found" — what do I do?**
- Choose "y" to install Miniforge
- Bootstrap downloads and runs the installer
- Close and reopen PowerShell (important!)
- Run `py -3 bootstrap.py` again
- Done!

**Q: Bootstrap fails to install packages with pip**
- This usually means pip isn't compatible with your Python version
- Try installing Miniforge instead (choose "y" when prompted)
- Or manually install Python 3.10 from python.org

**Q: I still get "no matching distribution for tensorflow"**
- You likely have Python 3.12+ without Miniforge
- Install Miniforge: https://github.com/conda-forge/miniforge/releases
- Or install Python 3.10/3.11 separately

**Q: Can I just run the GUI without setting up an environment?**
- No, TensorFlow and other packages must be installed first
- Use bootstrap to do this automatically

---

## Still stuck?

1. Paste the exact error message you see
2. Tell me which Python version you have: `py -3 --version`
3. I'll help debug!
