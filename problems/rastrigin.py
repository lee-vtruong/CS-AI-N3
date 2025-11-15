import numpy as np

def rastrigin(x):
    """Rastrigin function for NumPy vector. Optimal value is 0.0 at x = [0, ..., 0]."""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

