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
    
def is_symmetric(a, tol=1.e-10) -> bool:
    '''Check if square matrix is symmetric.'''
    return np.allclose(a, a.T, atol=tol)

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

def compute_eigenvalues_and_singular_values(H, sections_specs={}) -> tuple[np.ndarray, dict]:
    '''
    Compute eigenvalues and singular values of H and rectangular sections extracted from it, respectively.

    Parameters
    ----------
    H : ndarray
        Hamiltonian matrix.
    sections_specs : dict, optional
        Specifications for the different rectangular sections to be extracted from H.

    Returns
    -------
    H_eigenvalues : ndarray
        Eigenvalues of H.
    H_sections : dict
        Rectangular sections of H together with their singular values.
    '''
    if is_symmetric(H):
        try:
            # uses a more efficient algorithm for symmetric matrices
            H_eigenvalues = la.eigvalsh(H, check_finite=False)
        except la.LinAlgError:
            print('Eigenvalue computation did not converge.')
    else:
        try:
            # uses a general algorithm
            H_eigenvalues = la.eigvals(H, check_finite=False)
        except la.LinAlgError:
            print('Eigenvalue computation did not converge.')

        H_eigenvalues = np.float64(H_eigenvalues) # convert to floats

    H_sections = {}
    for k, v in sections_specs.items():
        section = extract_rectangular_section(H, m=v['m'], n=v['n'], shift=v['shift'])

        # Singular Value Decomposition: A = U * S * VT (A is any m by n matrix), where the columns of U (m by m) are eigenvectors of A * AT
        # and the columns of V (n by n) are eigenvectors of AT * A. And S is diagonal (but rectangular, m by n). 
        # The r singular values on the diagonal of S are the square roots of the nonzero eigenvalues of both A * AT and AT * A.
        s = la.svd(section, compute_uv=False, check_finite=False)

        H_sections[k] = {'matrix': section, 'sv': s}

    return H_eigenvalues, H_sections

def _create_figure(hist_data, fname):
    '''Create figure and set of subplots.'''
    n_subplots = 1
    height_ratios = None
    if any(key.startswith('Singular') for key in hist_data):
        n_subplots = 2
        height_ratios = [8, 2]

    fig, axs = plt.subplots(
        nrows=n_subplots, 
        sharex=True,
        height_ratios=height_ratios
        )
    
    if type(axs) == matplotlib.axes._axes.Axes:
        axs = np.array([axs], dtype=object)
    
    palette = sns.color_palette('colorblind', as_cmap=True)

    color_idx = 0
    for k, v in hist_data.items():
        if k.startswith('Eigenvalues'):
            # sns.histplot(
            #     x=v,
            #     ax=axs[0],
            #     binwidth=2/10,
            #     color=palette[0],
            #     )

            # Mirror data points near boundaries, calculate KDE and then ignore reflected part in order to fix Boundary Bias.
            left_boundary = 0
            right_boundary = 2
            right_reflected_v = 2 * right_boundary - v
            left_reflected_v = 2 * left_boundary - v
            reflected_v = np.append(left_reflected_v, np.append(v, right_reflected_v))

            sns.kdeplot(
                x=reflected_v,
                ax=axs[0],
                color=palette[0],
                fill=True,
                clip=(0, 2) # Do not evaluate the density outside of these limits.
            )
            
            axs[0].set_title(k)

        else: # Singular values
            sns.rugplot(
                x=v,
                ax=axs[1],
                color=palette[color_idx],
                height=0.75
            )

            color_idx += 1
    
    if n_subplots == 2:
        axs[1].set_title('Singular values')
        axs[1].set_yticks([])

    fig.tight_layout()
    
    fig.savefig(
        fname=fname,
        dpi=800
        )

def generate_plot(length, H_perturbed, H_eigenvalues, H_sections):
    '''
    Prepare plotting data and create figure.

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
        fname = f'{PLOTS_DIR}/hist_perturbed_L={length}.png'
    else: 
        eigenvalues_plot_title = 'Eigenvalues nonperturbed Hamiltonian'
        fname = f'{PLOTS_DIR}/hist_nonperturbed_L={length}.png'

    hist_data = {eigenvalues_plot_title: H_eigenvalues}

    for k, v in H_sections.items():
        hist_data[f'Singular values {k}'] = v['sv']
    
    _create_figure(hist_data, fname)

def free_hamiltonian():
    '''
    Compute the spectrum of the Hamiltonian of the one-dimensional time-independent free Schroedinger equation with periodic boundary conditions.
    '''
    L = 500
    perturb_H = False

    H = make_hamiltonian(length=L, perturb_H=perturb_H)

    sections_specs = {
        # 'first section': dict(m=13, n=11, shift=0),
        # 'second section': dict(m=13, n=11, shift=30)
    }
    H_eigenvalues, H_sections = compute_eigenvalues_and_singular_values(H, sections_specs)

    generate_plot(L, perturb_H, H_eigenvalues, H_sections)

if __name__ == '__main__':
    free_hamiltonian()
    print(f'{__file__} complete!')
