import numpy as np
import matplotlib.pyplot as plt

def get_eigvals(n):
    A = (np.random.randn(n, n) + 1j*np.random.randn(n, n)) / np.sqrt(2*n)
    return np.linalg.eigvals(A)

sizes = [100, 500, 1000, 2000]

fig, axes = plt.subplots(1, 4, figsize=(15, 5), dpi=200)

theta = np.linspace(0, 2*np.pi, 500)
circle_x, circle_y = np.cos(theta), np.sin(theta)

for ax, n in zip(axes, sizes):
    eigvals = get_eigvals(n)
    ax.scatter(eigvals.real, eigvals.imag, s=5, color='black', alpha=0.5)
    ax.plot(circle_x, circle_y, 'r', lw=1)
    ax.set_aspect('equal')
    ax.set_title(f"n = {n}")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.grid(True, ls=':')

plt.tight_layout()
plt.show()
