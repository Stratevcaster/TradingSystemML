"""Launch the Tkinter GUI briefly, take a screenshot, save to results/gui_screenshot.png, then exit.

This will only work if the environment has a graphical desktop (Windows session). The script
handles errors gracefully and logs them to stdout.
"""
import os
import time
import threading
import traceback

try:
    from PIL import ImageGrab
except Exception:
    ImageGrab = None

import importlib.util
import sys
spec = importlib.util.spec_from_file_location('gui_app', os.path.join(os.path.dirname(__file__), 'gui_app.py'))
gui = importlib.util.module_from_spec(spec)
sys.modules['gui_app'] = gui
spec.loader.exec_module(gui)

os.makedirs('results', exist_ok=True)
out = os.path.join('results', 'gui_screenshot.png')

app = gui.App()

succeeded = False

# Run the GUI in a thread so we can take a screenshot
def run_app():
    try:
        app.after(2000, app.quit)  # run mainloop for ~2s
        app.mainloop()
    except Exception:
        traceback.print_exc()

thr = threading.Thread(target=run_app, daemon=True)
thr.start()

# Wait a bit for GUI to appear
time.sleep(1.5)

try:
    if ImageGrab is None:
        print('PIL.ImageGrab not available; cannot capture screenshot here.')
    else:
        img = ImageGrab.grab()
        img.save(out)
        print('Screenshot saved to', out)
        succeeded = True
except Exception as e:
    print('Failed to capture screenshot:', e)

# Ensure GUI stopped
try:
    thr.join(timeout=3)
except Exception:
    pass

if not succeeded:
    print('Screenshot was not captured. If you are running this on a remote/CI machine without a GUI, this will not work.')
else:
    print('Done.')
