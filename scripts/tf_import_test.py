import os
import time, sys

# Try to force CPU-only behavior and reduce TF logging to avoid GPU/DLL issues
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

start = time.time()
try:
    print('start import')
    sys.stdout.flush()
    import tensorflow as tf
    print('OK', tf.__version__)
except Exception as e:
    print('EXC', repr(e))
    import traceback
    traceback.print_exc()
finally:
    print('elapsed', time.time() - start)
    sys.stdout.flush()
