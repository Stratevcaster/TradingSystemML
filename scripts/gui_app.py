"""Simple Tkinter GUI to run training and testing pipelines interactively.

Features:
- Specify ticker, model type (LSTM/GRU), target (price/returns), days to predict, epochs, units, layers, dropout, bidirectional
- Buttons: Train, Test, Train+Test
- Shows live logs in a text box and saves artifacts to `results/`

Usage:
  python scripts/gui_app.py
"""
import threading
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import glob
try:
    import pandas as pd
except Exception:
    pd = None
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    Figure = None
    FigureCanvasTkAgg = None

try:
    import parameters
except Exception:
    # Fallback dummy parameters when the real module (and heavy deps like tensorflow) aren't available
    class _DummyParameters:
        ticker = 'BTC-USD'
        TARGET = 'returns'
        date_now = 'nodate'
        LOSS = 'mse'
        activation = 'linear'
        normalizer = 'minmax'
        CELL = type('Cell', (), {'__name__': 'LSTM'})
        N_STEPS = 1
        NUM_LAYERS = 1
        UNITS = 32
        bidirectional = True

    parameters = _DummyParameters()

# Try to wire up the real training and testing functions if available in the repo
try:
    from train import train as train_step  # train.step signature: (step, model_name)
except Exception:
    try:
        from trainLin import train as train_step
    except Exception:
        train_step = None

try:
    from tester import test as tester_test  # tester.test signature: (N_DAYS_STEP)
except Exception:
    try:
        # legacy module name
        from test import test as tester_test
    except Exception:
        tester_test = None


class TextLogger:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, msg):
        try:
            self.text_widget.configure(state='normal')
            self.text_widget.insert('end', str(msg))
            self.text_widget.see('end')
            self.text_widget.configure(state='disabled')
        except Exception:
            pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Trading GUI')
        frm = ttk.Frame(self)
        frm.pack(fill='both', expand=True, padx=8, pady=8)
        left = ttk.Frame(frm, width=240)
        left.pack(side='left', fill='y')
        # Ticker
        ttk.Label(left, text='Ticker:').pack(anchor='w')
        self.ticker_var = tk.StringVar(value=getattr(parameters, 'ticker', 'BTC-USD'))
        ttk.Entry(left, textvariable=self.ticker_var, width=18).pack(anchor='w')
        ttk.Label(left, text='Model cell:').pack(anchor='w', pady=(8,0))
        self.cell_var = tk.StringVar(value='LSTM')
        ttk.Combobox(left, textvariable=self.cell_var, values=['LSTM','GRU'], width=18).pack(anchor='w')

        ttk.Label(left, text='Target:').pack(anchor='w', pady=(8,0))
        self.target_var = tk.StringVar(value=getattr(parameters, 'TARGET', 'returns'))
        ttk.Combobox(left, textvariable=self.target_var, values=['price','returns'], width=18).pack(anchor='w')

        ttk.Label(left, text='Days to predict:').pack(anchor='w', pady=(8,0))
        self.days_var = tk.IntVar(value=getattr(parameters, 'N_DAYS_STEP', 10))
        ttk.Entry(left, textvariable=self.days_var, width=8).pack(anchor='w')

        ttk.Label(left, text='Epochs:').pack(anchor='w', pady=(8,0))
        self.epochs_var = tk.IntVar(value=10)
        ttk.Entry(left, textvariable=self.epochs_var, width=8).pack(anchor='w')

        ttk.Label(left, text='Units:').pack(anchor='w', pady=(8,0))
        self.units_var = tk.IntVar(value=getattr(parameters, 'UNITS', 128))
        ttk.Entry(left, textvariable=self.units_var, width=8).pack(anchor='w')

        ttk.Label(left, text='Layers:').pack(anchor='w', pady=(8,0))
        self.layers_var = tk.IntVar(value=getattr(parameters, 'NUM_LAYERS', 2))
        ttk.Entry(left, textvariable=self.layers_var, width=8).pack(anchor='w')

        ttk.Label(left, text='Dropout:').pack(anchor='w', pady=(8,0))
        self.dropout_var = tk.DoubleVar(value=getattr(parameters, 'DROPOUT', 0.2))
        ttk.Entry(left, textvariable=self.dropout_var, width=8).pack(anchor='w')

        self.bidirectional_var = tk.BooleanVar(value=getattr(parameters, 'bidirectional', True))
        ttk.Checkbutton(left, text='Bidirectional', variable=self.bidirectional_var).pack(anchor='w', pady=(8,0))

        ttk.Button(left, text='Train', command=self._on_train).pack(fill='x', pady=(12,4))
        ttk.Button(left, text='Test', command=self._on_test).pack(fill='x', pady=(0,4))
        ttk.Button(left, text='Train + Test', command=self._on_train_test).pack(fill='x', pady=(0,4))
        ttk.Button(left, text='Open results folder', command=self._open_results).pack(fill='x', pady=(8,4))
        # status and progress
        ttk.Label(left, text='Status:').pack(anchor='w', pady=(8,0))
        self.status_var = tk.StringVar(value='Idle')
        ttk.Label(left, textvariable=self.status_var).pack(anchor='w')
        self.progress = ttk.Progressbar(left, mode='determinate', maximum=100)
        self.progress.pack(fill='x', pady=(6,0))

        # Right: logs
        right = ttk.Frame(frm)
        right.pack(side='right', fill='both', expand=True)

        # Split right side into logs (top) and image (bottom)
        self.log_frame = ttk.Frame(right)
        self.log_frame.pack(side='top', fill='both', expand=True)
        self.log_text = tk.Text(self.log_frame, state='disabled', height=18)
        self.log_text.pack(fill='both', expand=True)

        self.image_frame = ttk.Frame(right, height=250)
        self.image_frame.pack(side='bottom', fill='x')
        self.image_label = ttk.Label(self.image_frame, text='[Plot will appear here]')
        self.image_label.pack(fill='both', expand=True)
        # placeholders for inline image/canvas
        self._current_image = None
        self._canvas = None

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill='x', pady=(8,0))
        ttk.Button(btn_frame, text='Show latest plot', command=self._show_latest_plot).pack(side='left', padx=(0,6))
        ttk.Button(btn_frame, text='Choose plot...', command=self._choose_and_show_plot).pack(side='left')
        ttk.Button(btn_frame, text='Manage presets', command=self._manage_presets).pack(side='left', padx=(6,0))

        # Preset save/load
        preset_frame = ttk.Frame(left)
        preset_frame.pack(fill='x', pady=(8,0))
        ttk.Button(preset_frame, text='Save preset', command=self._save_preset).pack(fill='x', pady=(4,2))
        ttk.Button(preset_frame, text='Load preset', command=self._load_preset).pack(fill='x', pady=(0,2))

        self.orig_stdout = sys.stdout
        self.logger = TextLogger(self.log_text)

    def _set_parameters_from_ui(self):
        parameters.ticker = self.ticker_var.get()
        parameters.TARGET = self.target_var.get()
        # set CELL
        if self.cell_var.get().upper() == 'GRU':
            try:
                parameters.CELL = __import__('tensorflow.keras.layers', fromlist=['GRU']).GRU
            except Exception:
                parameters.CELL = type('Cell', (), {'__name__': 'GRU'})
        else:
            try:
                parameters.CELL = __import__('tensorflow.keras.layers', fromlist=['LSTM']).LSTM
            except Exception:
                parameters.CELL = type('Cell', (), {'__name__': 'LSTM'})
        parameters.N_DAYS_STEP = int(self.days_var.get())
        parameters.EPOCHS = int(self.epochs_var.get())
        parameters.UNITS = int(self.units_var.get())
        parameters.NUM_LAYERS = int(self.layers_var.get())
        parameters.DROPOUT = float(self.dropout_var.get())
        parameters.bidirectional = bool(self.bidirectional_var.get())

    def _run_threaded(self, target, *args, **kwargs):
        t = threading.Thread(target=self._run_and_capture, args=(target,)+args, kwargs=kwargs, daemon=True)
        t.start()

    def _run_and_capture(self, func, *args, **kwargs):
        try:
            sys.stdout = self.logger
            sys.stderr = self.logger
            func(*args, **kwargs)
        except Exception:
            traceback.print_exc()
        finally:
            sys.stdout = self.orig_stdout
            sys.stderr = sys.__stderr__

    def _set_status(self, text, progress_pct=None):
        try:
            self.status_var.set(text)
            if progress_pct is None:
                self.progress.config(mode='indeterminate')
                try:
                    self.progress.start(10)
                except Exception:
                    pass
            else:
                self.progress.config(mode='determinate')
                self.progress['value'] = progress_pct
        except Exception:
            pass

    def _clear_image_area(self):
        # remove a Matplotlib canvas or static image
        try:
            if getattr(self, '_canvas', None) is not None:
                self._canvas.get_tk_widget().destroy()
                self._canvas = None
        except Exception:
            pass
        try:
            self.image_label.configure(image='', text='[Plot will appear here]')
            self._current_image = None
        except Exception:
            pass

    def _show_latest_plot(self):
        """Search results/plots for the newest PNG and display it inline."""
        try:
            # Prefer preds CSV files to render with Matplotlib
            preds_files = glob.glob(os.path.join('results', '*_preds.csv'))
            if preds_files:
                latest_preds = max(preds_files, key=os.path.getmtime)
                self._display_plot_from_preds(latest_preds)
                return

            files = glob.glob(os.path.join('results', 'plots', '*.png'))
            if not files:
                self._log('No plot or preds files found in results/')
                return
            latest = max(files, key=os.path.getmtime)
            self._display_image(latest)
        except Exception as e:
            self._log(f'Error showing latest plot: {e}')

    def _choose_and_show_plot(self):
        path = filedialog.askopenfilename(title='Choose plot', filetypes=[('PNG Images','*.png'),('All files','*.*')], initialdir=os.path.abspath('results'))
        if path:
            self._display_image(path)

    def _display_image(self, path):
        try:
            # Try to use PIL for reliable PNG support
            try:
                from PIL import Image, ImageTk
                img = Image.open(path)
                img.thumbnail((800, 300))
                photo = ImageTk.PhotoImage(img)
            except Exception:
                # Fallback to Tk PhotoImage
                photo = tk.PhotoImage(file=path)

            # keep reference to avoid GC
            self._clear_image_area()
            self._current_image = photo
            self.image_label.configure(image=photo, text='')
            self._log(f'Displayed plot: {path}')
        except Exception as e:
            self._log(f'Failed to display image: {e}')

    def _display_plot_from_preds(self, csv_path):
        """Load preds CSV and render matplotlib figure inline with legend."""
        try:
            self._clear_image_area()
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                # Fallback CSV reader when pandas isn't available in the test env
                import csv
                with open(csv_path, 'r', newline='') as fh:
                    reader = csv.reader(fh)
                    rows = list(reader)
                if not rows:
                    df = None
                else:
                    cols = rows[0]
                    data = {c: [] for c in cols}
                    for row in rows[1:]:
                        for c, v in zip(cols, row):
                            try:
                                data[c].append(float(v))
                            except Exception:
                                data[c].append(v)
                    # Simple object with dict-like access
                    class _DF:
                        def __init__(self, d):
                            self._d = d
                            self.columns = list(d.keys())
                        def __getitem__(self, key):
                            return self._d.get(key, [])
                    df = _DF(data)
            # If matplotlib isn't available, show a text placeholder instead
            if Figure is None or FigureCanvasTkAgg is None:
                self._clear_image_area()
                self.image_label.configure(text=f'Plot preview unavailable (matplotlib missing): {os.path.basename(csv_path)}')
                self._log(f'Would display preds plot: {csv_path} (matplotlib not available)')
                return

            fig = Figure(figsize=(8,3))
            ax = fig.add_subplot(111)
            if 'true_price' in df.columns and 'pred_price' in df.columns:
                ax.plot(df['true_price'].values, label='True')
                ax.plot(df['pred_price'].values, label='Predicted')
            else:
                # fallback: plot first two columns
                cols = df.columns.tolist()
                if len(cols) >= 2:
                    ax.plot(df[cols[0]].values, label=cols[0])
                    ax.plot(df[cols[1]].values, label=cols[1])
            ax.legend()
            ax.set_title(os.path.basename(csv_path))
            canvas = FigureCanvasTkAgg(fig, master=self.image_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            self._canvas = canvas
            self._log(f'Displayed preds plot: {csv_path}')
        except Exception as e:
            self._log(f'Failed to display preds plot: {e}')

    def _log(self, message):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', str(message) + '\n')
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def _save_preset(self):
        import json
        preset = {
            'ticker': self.ticker_var.get(),
            'cell': self.cell_var.get(),
            'target': self.target_var.get(),
            'days': int(self.days_var.get()),
            'epochs': int(self.epochs_var.get()),
            'units': int(self.units_var.get()),
            'layers': int(self.layers_var.get()),
            'dropout': float(self.dropout_var.get()),
            'bidirectional': bool(self.bidirectional_var.get())
        }
        os.makedirs('presets', exist_ok=True)
        fname = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON','*.json')], initialdir=os.path.abspath('presets'), title='Save preset as')
        if not fname:
            return
        with open(fname, 'w') as f:
            json.dump(preset, f, indent=2)
        self._log(f'Saved preset to {fname}')

    def _load_preset(self):
        import json
        os.makedirs('presets', exist_ok=True)
        fname = filedialog.askopenfilename(title='Load preset', filetypes=[('JSON','*.json')], initialdir=os.path.abspath('presets'))
        if not fname:
            return
        with open(fname, 'r') as f:
            preset = json.load(f)
        # apply to UI
        self.ticker_var.set(preset.get('ticker', self.ticker_var.get()))
        self.cell_var.set(preset.get('cell', self.cell_var.get()))
        self.target_var.set(preset.get('target', self.target_var.get()))
        self.days_var.set(preset.get('days', self.days_var.get()))
        self.epochs_var.set(preset.get('epochs', self.epochs_var.get()))
        self.units_var.set(preset.get('units', self.units_var.get()))
        self.layers_var.set(preset.get('layers', self.layers_var.get()))
        self.dropout_var.set(preset.get('dropout', self.dropout_var.get()))
        self.bidirectional_var.set(preset.get('bidirectional', self.bidirectional_var.get()))
        self._log(f'Loaded preset from {fname}')

    def _manage_presets(self):
        """Open a small dialog to list / rename / delete preset JSON files."""
        top = tk.Toplevel(self)
        top.title('Manage presets')
        top.geometry('400x300')

        listbox = tk.Listbox(top)
        listbox.pack(fill='both', expand=True, padx=8, pady=8)

        def refresh():
            listbox.delete(0, 'end')
            os.makedirs('presets', exist_ok=True)
            files = sorted(glob.glob(os.path.join('presets', '*.json')))
            for f in files:
                listbox.insert('end', os.path.basename(f))

        def delete():
            sel = listbox.curselection()
            if not sel:
                return
            name = listbox.get(sel[0])
            path = os.path.join('presets', name)
            try:
                os.remove(path)
                self._log(f'Deleted preset {name}')
                refresh()
            except Exception as e:
                self._log(f'Failed to delete preset: {e}')

        def rename():
            sel = listbox.curselection()
            if not sel:
                return
            name = listbox.get(sel[0])
            new = tk.simpledialog.askstring('Rename preset', 'New name (without .json):', parent=top)
            if not new:
                return
            src = os.path.join('presets', name)
            dst = os.path.join('presets', new + '.json')
            try:
                os.rename(src, dst)
                self._log(f'Renamed preset {name} -> {os.path.basename(dst)}')
                refresh()
            except Exception as e:
                self._log(f'Failed to rename preset: {e}')

        btn_frame = ttk.Frame(top)
        btn_frame.pack(fill='x', padx=8, pady=(0,8))
        ttk.Button(btn_frame, text='Delete', command=delete).pack(side='left', padx=(0,6))
        ttk.Button(btn_frame, text='Rename', command=rename).pack(side='left')
        ttk.Button(btn_frame, text='Close', command=top.destroy).pack(side='right')

        refresh()

    def _on_train(self):
        self._set_parameters_from_ui()
        days = int(self.days_var.get())
        epochs = int(self.epochs_var.get())
        def do_train():
            print('Starting training sequence...')
            self._set_status('Training...', progress_pct=0)
            cancel_flag = {'stop': False}

            # show progress dialog
            progress_top = tk.Toplevel(self)
            progress_top.title('Training progress')
            ttk.Label(progress_top, text='Training progress').pack(padx=8, pady=6)
            prog = ttk.Progressbar(progress_top, maximum=days, mode='determinate')
            prog.pack(fill='x', padx=8, pady=(0,6))
            status_label = ttk.Label(progress_top, text='Starting...')
            status_label.pack(padx=8, pady=(0,6))
            cancel_btn = ttk.Button(progress_top, text='Cancel')
            cancel_btn.pack(padx=8, pady=(0,8))

            def cancel():
                cancel_flag['stop'] = True
                status_label.config(text='Cancel requested — finishing current step...')

            cancel_btn.config(command=cancel)

            for step in range(1, days+1):
                if cancel_flag['stop']:
                    print('Training cancelled by user — stopping after current step')
                    break

                model_name = f"{parameters.date_now}_{parameters.ticker}-{parameters.LOSS}-{parameters.activation}-{parameters.normalizer}-{parameters.CELL.__name__}-seq-{parameters.N_STEPS}-step-{step}-layers-{parameters.NUM_LAYERS}-units-{parameters.UNITS}"
                if parameters.bidirectional:
                    model_name += 'bidirectional'

                status_text = f'Step {step}/{days}'
                print(status_text, 'training with model_name=', model_name)
                status_label.config(text=status_text)
                self._set_status(status_text, progress_pct=int((step/days)*100))
                # call the real training step (defined elsewhere)
                if train_step is None:
                    # If no real training function is available, run a simulated training step
                    self._log('Real training unavailable; running simulated training step')
                    try:
                        # simple simulation: write a demo preds CSV and optionally update a plot
                        self._simulate_train_step(step, model_name)
                    except Exception as e:
                        self._log(f'Simulated training failed: {e}')
                        break
                else:
                    train_step(step, model_name)
                prog['value'] = step

            try:
                progress_top.destroy()
            except Exception:
                pass

            # Auto-refresh latest plot when training finishes
            try:
                self._show_latest_plot()
            except Exception:
                self._log('Failed to auto-show latest plot after training')

            self._set_status('Idle', progress_pct=0)

        self._run_threaded(do_train)

    def _on_train_test(self):
        self._on_train()
        # chain test after a short delay to allow training thread to finish
        def delayed_test():
            # wait until training threads finish by polling
            import time
            while threading.active_count() > 1:
                time.sleep(1)
            self._on_test()
        threading.Thread(target=delayed_test, daemon=True).start()

    def _on_test(self):
        self._set_parameters_from_ui()
        days = int(self.days_var.get())

        def do_test():
            print('Starting test...')
            try:
                preds = tester_test(days)
                print('Test finished. Predictions:', preds)
                # offer to save predictions to file
                out = os.path.join('results', f'gui_test_preds_{parameters.ticker}_{parameters.date_now}.csv')
                try:
                    import numpy as np
                    np.savetxt(out, preds, delimiter=',')
                    print('Saved predictions to', out)
                except Exception:
                    # If numpy isn't available or preds is not an array, write CSV manually
                    os.makedirs('results', exist_ok=True)
                    with open(out, 'w') as fh:
                        for row in (preds or []):
                            fh.write(','.join(map(str, row)) + '\n')
                    print('Saved predictions to', out)
            except NameError:
                print('tester_test not available in test environment; skipping actual test')

        self._run_threaded(do_test)

    def _open_results(self):
        path = os.path.abspath('results')
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        try:
            if sys.platform.startswith('win'):
                os.startfile(path)
            else:
                import subprocess
                subprocess.run(['xdg-open', path])
        except Exception as e:
            messagebox.showinfo('Open results', f'Location: {path}')


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
