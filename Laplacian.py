'''Discretization of the time-independent one-dimensional free Schroedinger equation with periodic boundary conditions.'''
import numpy as np
from scipy import linalg as la
import seaborn as sns
# import plotly.express as px
# import plotly.io as pio

# pio.renderers.default = 'browser'

def middle_value(k):
    '''
    Find the middle value of a positive integer. 

    The middle value of a positive integer k is defined like so: if k odd, then the middle value is (k + 1) / 2, otherwise it is k / 2.
    
    Parameters
    ----------
    k : int
        Positive integer.

    Returns
    -------
    int
        Middle value of integer k.
    '''
    if k % 2 == 0:
        return int(k / 2)
    else:
        return int((k + 1) / 2)

def rectangular_section(a, m, n, shift=0):
    '''
    Extract a rectangular section from the diagonal of a square matrix.
    
    Parameters
    ----------
    a : ndarray
        Square matrix from which the rectangular section will be extracted.
    m : int
        Number of rows of the rectangular section.
    n : int
        Number of columns of the rectangular section.
    shift : int, optional
        Location along the main diagonal from where the rectangular section will be extracted. Default is 0.
        If 0, the rectangular section will be centered around the middle value of the main diagonal of the matrix.
        else, the center of the rectangular section will be shifted by j steps about its middle value along the diagonal.
        The middle value of the diagonal is defined like so: if the number of rows k of the matrix is odd, then the middle value is the ((k + 1) / 2)th value, 
        otherwise it is the (k / 2)th value.

    Returns
    -------
    ndarray
        Rectangular section.
    '''
    k = a.shape[0]  # number of rows of matrix a
    diag_mid_val_idx = middle_value(k)
    m_mid_val = middle_value(m)
    n_mid_val = middle_value(n)

    left_row_idx = diag_mid_val_idx - m_mid_val + shift
    right_row_idx = diag_mid_val_idx + m_mid_val + shift - (m % 2)
    left_column_idx = diag_mid_val_idx - n_mid_val + shift
    right_column_idx = diag_mid_val_idx + n_mid_val + shift - (n % 2)

    if left_row_idx < 0 or left_column_idx < 0 or right_row_idx > k or right_column_idx > k:
        raise ValueError('The size and/or location of the rectangular section is not compatible with the matrix.')
    else:
        return a[left_row_idx : right_row_idx, left_column_idx : right_column_idx]

def free_hamiltonian():
    L = 11
    n = L + 1
    dx = L / (n - 1) # we want unit spacing between the particles (nodes in the mesh)

    # solved using finite differences with periodic boundary conditions
    diag = -2. * np.ones(n - 1)
    off_diag = np.ones(n - 2)
    a = -0.5 / dx**2

    H = a * (np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1))
    H[0, -1] = a
    H[-1, 0] = a

    H = H + np.diag(np.arange(n - 1))

    print(H)
    print()

    H_rect = rectangular_section(H, m=7, n=5, shift=0)
    print(H_rect)
    print()

    # eigenvalues, eigenvectors = np.linalg.eig(H)
    eigenvalues, eigenvectors = la.eig(H, check_finite=False)

    # x = np.arange(stop=L, step=dx)

    # plot the first 3 eigenvectors of H
    # for i in range(1, 4):
    #     fig = px.scatter(x=x, y=eigenvectors[:, i])
    #     fig.show()

    # Singular Value Decomposition: A = U * S * VT (A is any m by n matrix), where the columns of U (m by m) are eigenvectors of A * AT
    # and the columns of V (n by n) are eigenvectors of AT * A. And S is diagonal (but rectangular, m by n). 
    # The r singular values on the diagonal of S are the square roots of the nonzero eigenvalues of both A * AT and AT * A.
    # S = la.svd(H_rect, compute_uv=False, check_finite=False)

    hist_data = {
        'Eigenvalues free Hamiltonian': np.float64(eigenvalues),
        # 'Singular values rectangular Hamiltonian': S
        }
    
    fig = sns.displot(
        data=hist_data,
        binwidth=0.01,
        rug=False,
        rug_kws={'height':-0.02, 'clip_on':False},
        # kind='kde',
        # cut=0
        )

    fig.savefig(fname='dist_plot_2.png')

if __name__ == '__main__':
    free_hamiltonian()
    print(f'{__file__} complete!')