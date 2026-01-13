'''Discretization of the time-independent one-dimensional free Schroedinger equation with periodic boundary conditions.'''
import numpy as np
from scipy import linalg as la
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio

pio.renderers.default = 'browser'

def free_hamiltonian():
    L = 500
    n = L + 1
    dx = L / (n - 1) # we want unit spacing between the particles (nodes in the mesh)

    # solved using finite differences with periodic boundary conditions
    diag = -2. * np.ones(n - 1)
    off_diag = np.ones(n - 2)
    a = -1. / (2. * dx**2)

    H = a * (np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1))
    H[0, -1] = a
    H[-1, 0] = a

    # print(H)
    # print()

    # eigenvalues, eigenvectors = np.linalg.eig(H)
    eigenvalues, eigenvectors = la.eig(H, check_finite=False)

    # x = np.arange(stop=L, step=dx)

    # plot the first 3 eigenvectors of H
    # for i in range(1, 4):
    #     fig = px.scatter(x=x, y=eigenvectors[:, i])
    #     fig.show()
    
    # here we restrict H to a smaller domain (L - 2)
    # H_res = H[1:-1, 1:-1]
    # print(H_res)
    # print()

    # we add a row above and below to H_res, thus yielding a rectangular matrix
    H_res_rect = H[:, 1:-1]
    # print(H_res_rect)
    # print()

    # Singular Value Decomposition: A = U * S * VT (A is any m by n matrix), where the columns of U (m by m) are eigenvectors of A * AT
    # and the columns of V (n by n) are eigenvectors of AT * A. And S is diagonal (but rectangular, m by n). 
    # The r singular values on the diagonal of S are the square roots of the nonzero eigenvalues of both A * AT and AT * A.
    U, S, VT = la.svd(H_res_rect, compute_uv=True, check_finite=False)

    hist_data = [np.float64(eigenvalues), S]
    group_labels = ['Eigenvalues free Hamiltonian', 'Singular values rectangular Hamiltonian']
 
    fig = ff.create_distplot(
        hist_data=hist_data, 
        group_labels=group_labels,
        show_hist=False
        )
    
    fig.show()

if __name__ == '__main__':
    free_hamiltonian()
    print(f'{__file__} complete!')