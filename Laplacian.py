'''Discretization of the time-independent one-dimensional free Schroedinger equation with periodic boundary conditions.'''
import numpy as np
from scipy import linalg as la
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
# import plotly.express as px
# import plotly.io as pio

# pio.renderers.default = 'browser'

# read directory from pathlib library (returns PosixPath object)
ROOT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = ROOT_DIR / 'plots'

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

def extract_rectangular_section(a, m, n, shift=0):
    '''
    Extract a rectangular section from a square matrix.
    
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
    
def make_hamiltonian(length, perturb_H=False) -> np.ndarray:
    '''
    Construct Hamiltonian matrix.
    
    Parameters
    ----------
    length : int
        Space length.
    perturb_H : bool, optional
        Whether to perturb Hamiltonian.
        If True, small random values are added to / subtracted from the nonzero entries of the matrix. Default is False.

    Returns
    -------
    H : ndarray
        Hamiltonian matrix
    '''
    n = length + 1
    dx = length / (n - 1) # we want unit spacing between the particles (nodes in the mesh)

    # solved using finite differences with periodic boundary conditions
    diag = -2. * np.ones(n - 1)
    off_diag = np.ones(n - 2)
    a = -0.5 / dx**2

    H = a * (np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1))
    H[0, -1] = a
    H[-1, 0] = a

    if perturb_H:
        rng = np.random.default_rng()
        # perturbs the main diagonal
        H = H + np.diag(rng.uniform(low=-0.25, high=0.25, size=n - 1))
        # perturbs the diagonals right above and below the main one 
        H = H + np.diag(rng.uniform(low=-0.1, high=0.1, size=n - 2), k=1) + np.diag(rng.uniform(low=-0.1, high=0.1, size=n - 2), k=-1)
        # perturbs the upper right corner and the lower left corner
        H[0, -1] = H[0, -1] + rng.uniform(low=-0.1, high=0.1)
        H[-1, 0] = H[-1, 0] + rng.uniform(low=-0.1, high=0.1)
    
    # H = H + np.diag(np.arange(n - 1))
    # print(H)
    # print()
    
    return H

def compute_eigenvalues_and_singular_values(H, rectangular_sections_specs={}) -> tuple[np.ndarray, dict]:
    '''
    Compute eigenvalues and singular values of H and rectangular sections extracted from it, respectively.

    Parameters
    ----------
    H : ndarray
        Hamiltonian matrix.
    rectangular_sections_specs : dict, optional
        Specifications for the different rectangular sections to be extracted from H.

    Returns
    -------
    H_eigenvalues : ndarray
        Eigenvalues of H.
    H_sections : dict
        Rectangular sections of H together with their singular values.
    '''
    H_eigenvalues, _ = la.eig(H, check_finite=False)

    H_sections = {}
    for k, v in rectangular_sections_specs.items():
        section = extract_rectangular_section(H, m=v['m'], n=v['n'], shift=v['shift'])

        # Singular Value Decomposition: A = U * S * VT (A is any m by n matrix), where the columns of U (m by m) are eigenvectors of A * AT
        # and the columns of V (n by n) are eigenvectors of AT * A. And S is diagonal (but rectangular, m by n). 
        # The r singular values on the diagonal of S are the square roots of the nonzero eigenvalues of both A * AT and AT * A.
        s = la.svd(section, compute_uv=False, check_finite=False)

        H_sections[k] = {'matrix': section, 'sv': s}

    return H_eigenvalues, H_sections

def make_histogram(hist_data, fname):
    '''Make histogram.'''
    n_subplots = 1
    for k in hist_data.keys():
        if k.startswith('Singular values'):
            n_subplots += 1

    fig, axs = plt.subplots(
        nrows=n_subplots, 
        sharex=True
        )
    
    if type(axs) == matplotlib.axes._axes.Axes:
        axs = np.array([axs], dtype=object)
    
    palette = sns.color_palette('colorblind', n_colors=n_subplots)

    idx = 0
    for k, v in hist_data.items():
        sns.histplot(
            x=v,
            ax=axs[idx],
            binwidth=0.01,
            color=palette[idx]
            )
        
        axs[idx].set_title(k)
        
        idx += 1
    
    fig.savefig(
        fname=fname,
        dpi=800
        )

def generate_histogram(length, H_perturbed, H_eigenvalues, H_sections):
    '''
    Prepare plotting data and create histogram.

    Parameters
    ----------
    length : int
        Space length.
    H_perturbed : bool
        Whether H has been perturbed.
    H_eigenvalues : ndarray
        Eingenvalues of H.
    H_sections : dict
        Rectangular sections of H together with their singular values.
    '''
    if H_perturbed:
        eigenvalues_plot_title = 'Eigenvalues perturbed Hamiltonian'
        fname = f'{PLOTS_DIR}/hist_plot_perturbed_{length}.png'
    else: 
        eigenvalues_plot_title = 'Eigenvalues nonperturbed Hamiltonian'
        fname = f'{PLOTS_DIR}/hist_plot_nonperturbed_{length}.png'

    eigenvalues = np.sort(np.float32(H_eigenvalues)) # sort and convert to floats

    hist_data = {eigenvalues_plot_title: eigenvalues}

    for k, v in H_sections.items():
        hist_data[f'Singular values {k}'] = v['sv']
    
    make_histogram(hist_data, fname)

def free_hamiltonian():
    '''
    Compute the spectrum of the Hamiltonian of the one-dimensional time-independent free Schroedinger equation with periodic boundary conditions.
    '''
    L = 100
    perturb_H = False

    H = make_hamiltonian(length=L, perturb_H=perturb_H)

    rectangular_sections_specs = {
        'first section': dict(m=13, n=11, shift=0),
        'second section': dict(m=13, n=11, shift=10)
    }
    H_eigenvalues, H_sections = compute_eigenvalues_and_singular_values(H, rectangular_sections_specs)

    generate_histogram(L, perturb_H, H_eigenvalues, H_sections)

if __name__ == '__main__':
    free_hamiltonian()
    print(f'{__file__} complete!')
