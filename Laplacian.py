"""Discretization of the time-independent one-dimensional free Schroedinger equation with periodic boundary conditions."""
import numpy as np
from scipy import linalg as la
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'

def free_hamiltonian():
    L = 10.
    n = 101
    dx = L / (n - 1)

    diag = -2. * np.ones(n - 1)
    off_diag = np.ones(n - 2)
    a = -1. / (2. * dx**2)

    H_free = a * (np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1))
    H_free[0, -1] = a
    H_free[-1, 0] = a

    # print(H_free)

    # eigenvalues, eigenvectors = np.linalg.eig(H_free)
    eigenvalues, eigenvectors = la.eig(H_free)

    # ft_eigenvalues = np.fft.fft(eigenvalues)

    x = np.arange(stop=L, step=dx)

    for i in range(1, 6):
        fig = px.scatter(x=x, y=eigenvectors[:, i])
        fig.show()

if __name__ == "__main__":
    free_hamiltonian()
    print(f"{__file__} complete!")