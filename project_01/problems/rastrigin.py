import numpy as np

def rastrigin(x):
    """Hàm Rastrigin cho vector NumPy. Tối ưu là 0.0 tại x = [0, ..., 0]."""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

