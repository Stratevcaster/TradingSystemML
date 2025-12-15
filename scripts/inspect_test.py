import importlib
from pathlib import Path
import sys
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
import tester as t
importlib.reload(t)
print('module tester file =', getattr(t, '__file__', None))
print('has load_weights_with_fallback =', hasattr(t, 'load_weights_with_fallback'))
print('attributes starting with load:', [n for n in dir(t) if n.startswith('load')])
