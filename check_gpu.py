"""Quick check that your GPU is visible to Python."""
import sys

print("Python:", sys.executable)
print()

# PyTorch (you have this in requirements)
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("  CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  Device:", torch.cuda.get_device_name(0))
        print("  Device count:", torch.cuda.device_count())
    else:
        print("  (PyTorch sees no CUDA device)")
except ImportError:
    print("PyTorch: not installed")

print()

# CuPy (optional)
try:
    import cupy as cp
    x = cp.array([1.0, 2.0, 3.0])
    y = cp.array([4.0, 5.0, 6.0])
    z = x + y
    print("CuPy: installed, GPU OK (test array sum:", float(z.sum()), ")")
except ImportError:
    print("CuPy: not installed (optional)")
except Exception as e:
    print("CuPy: installed but GPU error:", e)
