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

def _determine_indices_for_rectangular_section(k, m, n, shift=0) -> tuple[int, int, int, int]:
    '''
    Helper function for determining indices of parent matrix that are occupied by rectangular section.

    Parameters
    ----------
    k : int
        Number of rows / columns of parent matrix.
    m : int
        Number of rows of rectangular section.
    n : int
        Number of columns of rectangular section.
    shift : int, optional
        Location along the main diagonal from where the rectangular section will be extracted. Default is 0.

    Returns
    -------
    tuple
        Indices of parent matrix that are occupied by rectangular section.
    '''
    diag_mid_val_idx = middle_value(k)
    m_mid_val = middle_value(m)
    n_mid_val = middle_value(n)

    left_row_idx = diag_mid_val_idx - m_mid_val + shift
    right_row_idx = diag_mid_val_idx + m_mid_val + shift - (m % 2)
    left_column_idx = diag_mid_val_idx - n_mid_val + shift
    right_column_idx = diag_mid_val_idx + n_mid_val + shift - (n % 2)

    return left_row_idx, right_row_idx, left_column_idx, right_column_idx

def extract_rectangular_section(A, m, n, shift=0):
    '''
    Extract a rectangular section from a square matrix.
    
    Parameters
    ----------
    A : ndarray
        Square matrix from which the rectangular section will be extracted.
    m : int
        Number of rows of rectangular section.
    n : int
        Number of columns of rectangular section.
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
    k = A.shape[0]  # number of rows / columns of matrix A
    left_row_idx, right_row_idx, left_column_idx, right_column_idx = _determine_indices_for_rectangular_section(k, m, n, shift)

    if left_row_idx < 0 or left_column_idx < 0 or right_row_idx > k or right_column_idx > k:
        raise ValueError('The size and/or location of the rectangular section is not compatible with the matrix.')
    else:
        return A[left_row_idx : right_row_idx, left_column_idx : right_column_idx]
    
def select_rectangular_sections(H, m, n, d) -> dict:
    '''
    Generate dictionary of specifications for multiple rectangular sections of size m by n.

    Parameters
    ----------
    H : ndarray
        Hamiltonian matrix.
    m : int
        Number of rows of each rectangular section.
    n : int
        Number of columns of each rectangular section.
    d : int
        Separation distance between the centers of consecutive rectangular sections.

    Returns
    -------
    sections_specs : dict
        Specifications for the different rectangular sections to be extracted from H.
    '''
    k = H.shape[0] # number of rows / columns in H
    left_row_idx, right_row_idx, left_column_idx, right_column_idx = _determine_indices_for_rectangular_section(k, m, n)

    def is_valid_shift(j_val):
        shift = j_val * d
        return (0 <= left_row_idx + shift and right_row_idx + shift <= k and
                0 <= left_column_idx + shift and right_column_idx + shift <= k)

    sections_specs = {}
    for direction in [1, -1]:
        j = 0 if direction == 1 else -1
        while is_valid_shift(j):
            sections_specs[f'section {j}'] = {'m': m, 'n': n, 'shift': j * d}
            j += direction

    return sections_specs
    
def make_free_hamiltonian(length, perturb_H=False, random_rng=(-0.1, 0.1)) -> np.ndarray:
    '''
    Construct free Hamiltonian matrix.
    
    Parameters
    ----------
    length : int
        Space length.
    perturb_H : bool, optional
        Whether to perturb the Hamiltonian.
        If True, random values are added to / subtracted from the nonzero entries of the matrix. Default is False.
    random_rng : tuple[float, float], optional
        Minimum and maximum values for range of random values used in the perturbation of H. Default is (-0.1, 0.1).

    Returns
    -------
    H : ndarray
        (Perturbed) Free Hamiltonian matrix
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
        H = H + np.diag(rng.uniform(low=random_rng[0], high=random_rng[1], size=n - 1))
        # perturbs the diagonals right above and below the main one 
        H = H + np.diag(rng.uniform(low=random_rng[0], high=random_rng[1], size=n - 2), k=1) + np.diag(rng.uniform(low=random_rng[0], high=random_rng[1], size=n - 2), k=-1)
        # perturbs the upper right corner and the lower left corner
        H[0, -1] = H[0, -1] + rng.uniform(low=random_rng[0], high=random_rng[1])
        H[-1, 0] = H[-1, 0] + rng.uniform(low=random_rng[0], high=random_rng[1])
    
    # H = H + np.diag(np.arange(n - 1))
    # print(H)
    # print()
    
    return H

def compute_eigenvalues_and_singular_values(H, sections_specs={}) -> tuple[np.ndarray, np.ndarray, dict]:
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
        Eigenvalues of H, each repeated according to its multiplicity.
    H_eigenvectors : ndarray
        Eigenvectors of H. The normalized eigenvector corresponding to the eigenvalue w[i] is the column v[:,i].
    H_sections : dict
        Rectangular sections of H together with their singular values.
    '''
    if is_symmetric(H):
        try:
            # uses a more efficient algorithm for symmetric matrices
            H_eigenvalues, H_eigenvectors = la.eigh(H, eigvals_only=False, check_finite=False)
        except la.LinAlgError:
            print('Eigenvalue computation did not converge.')
    else:
        try:
            # uses a general algorithm
            H_eigenvalues, H_eigenvectors = la.eig(H, right=True, check_finite=False)
        except la.LinAlgError:
            print('Eigenvalue computation did not converge.')

        H_eigenvalues = np.float64(H_eigenvalues) # convert to floats

    H_sections = {}
    for k, v in sections_specs.items():
        section = extract_rectangular_section(H, m=v['m'], n=v['n'], shift=v['shift'])

        # Singular Value Decomposition: A = U * S * VT (A is any m by n matrix), where the columns of U (m by m) are eigenvectors of A * AT
        # and the columns of V (n by n) are eigenvectors of AT * A. And S is diagonal (but rectangular, m by n). 
        # The r singular values on the diagonal of S are the square roots of the nonzero eigenvalues of both A * AT and AT * A.
        U, s, Vh = la.svd(section, compute_uv=True, check_finite=False)

        H_sections[k] = {'matrix': section, 'U': U, 'sv': s, 'V': Vh.T}

    return H_eigenvalues, H_eigenvectors, H_sections

def dist_lambda_spec_H(lmbd, H_eigenvalues):
    '''
    Calculate d(λ, Spec(H)).

    Returns
    -------
    ndarray
        Distances between λ and Spec(H). Sorted in increasing order.
    '''
    return np.sort(np.abs(lmbd - H_eigenvalues), kind='stable')

def _mirror_array(arr) -> tuple[np.ndarray, tuple]:
    '''
    Extend array by mirroring it across its boundaries.
    
    Parameters
    ----------
    arr : ndarray
        One-dimensional array of floats.

    Returns
    -------
    ndarray
        Extended array.
    tuple
        Left and right boundaries of original array.
    '''
    lb = arr.min()
    rb = arr.max()
    left_mirrored_arr = 2 * lb - arr
    right_mirrored_arr = 2 * rb - arr
    return np.append(left_mirrored_arr, np.append(arr, right_mirrored_arr)), (lb, rb)

def _create_figure(hist_data, fname):
    '''Create figure and set of subplots.'''
    n_subplots = 1
    height_ratios = None
    if any(key.startswith('Singular') for key in hist_data):
        n_subplots = 2
        # height_ratios = (8, 2)
        height_ratios = (1, 1)

    fig, axs = plt.subplots(
        nrows=n_subplots, 
        sharex=True,
        height_ratios=height_ratios
        )
    
    if type(axs) == matplotlib.axes._axes.Axes:
        axs = np.array([axs], dtype=object)
    
    palette = sns.color_palette('colorblind', as_cmap=True)

    # color_idx = 0
    sv_array = np.array([], dtype=np.float64)
    for k, v in hist_data.items():
        if k.startswith('Eigenvalues'):
            # sns.histplot(
            #     x=v,
            #     ax=axs[0],
            #     binwidth=2/10,
            #     color=palette[0],
            #     )

            # Mirror data points near boundaries, calculate KDE and then ignore reflected part in order to fix Boundary Bias.
            reflected_eig, boundaries_eig = _mirror_array(v)

            sns.kdeplot(
                x=reflected_eig,
                ax=axs[0],
                color=palette[0],
                fill=True,
                clip=boundaries_eig # Do not evaluate the density outside of these limits.
            )
            
            axs[0].set_title(k)

        else: # Singular values
            # sns.rugplot(
            #     x=v,
            #     ax=axs[1],
            #     color=palette[color_idx],
            #     height=0.75
            # )

            # color_idx += 1
            sv_array = np.append(sv_array, v)

    reflected_sv, boundaries_sv = _mirror_array(sv_array)

    sns.kdeplot(
            x=reflected_sv,
            ax=axs[1],
            color=palette[1],
            fill=True,
            clip=boundaries_sv # Do not evaluate the density outside of these limits.
        )    
    
    if n_subplots == 2:
        axs[1].set_title('Singular values')
        # axs[1].set_yticks([])

    fig.tight_layout()
    
    fig.savefig(
        fname=fname,
        dpi=800
        )

def generate_plot(length, H_perturbed, H_eigenvalues, H_sections, plots_subfolder):
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
    plots_subfolder : str
        Name of directory where plot will be saved.
    '''
    FIG_DIR = PLOTS_DIR / plots_subfolder
    FIG_DIR.mkdir(exist_ok=True)

    if H_perturbed:
        eigenvalues_plot_title = 'Eigenvalues perturbed Hamiltonian'
        fname = f'{FIG_DIR}/kde_perturbed_L={length}.png'
    else: 
        eigenvalues_plot_title = 'Eigenvalues nonperturbed Hamiltonian'
        fname = f'{FIG_DIR}/kde_nonperturbed_L={length}.png'

    fig_data = {eigenvalues_plot_title: H_eigenvalues}

    for k, v in H_sections.items():
        fig_data[f'Singular values {k}'] = v['sv']
    
    _create_figure(fig_data, fname)

def free_hamiltonian():
    '''
    Compute the spectrum of the Hamiltonian of the one-dimensional time-independent free Schroedinger equation with periodic boundary conditions.
    '''
    L = 100
    perturb_H = False
    plots_subfolder = 'free_Hamiltonian'

    H = make_free_hamiltonian(length=L, perturb_H=perturb_H, random_rng=(-0.1, 0.1))

    # sections_specs = {
    #     # 'first section': dict(m=13, n=11, shift=0),
    #     # 'second section': dict(m=13, n=11, shift=30)
    # }

    sections_specs = select_rectangular_sections(H, m=52, n=50, d=5)

    H_eigenvalues, H_eigenvectors, H_sections = compute_eigenvalues_and_singular_values(H, sections_specs)

    generate_plot(L, perturb_H, H_eigenvalues, H_sections, plots_subfolder)

def free_hamiltonian_lambda():
    '''
    Compute the spectrum of (H - λ), where H is the Hamiltonian of the one-dimensional time-independent free Schroedinger equation with periodic boundary conditions
    and λ is any real number. 
    '''
    L = 1000
    perturb_H = False
    plots_subfolder='H_lambda'

    H = make_free_hamiltonian(length=L, perturb_H=perturb_H, random_rng=(-0.2, 0.2))

    lmbd = 2.5  # λ
    H_lambda = H - lmbd * np.identity(n=H.shape[0])  # H - λ

    n = 50  # in Hege's notation: 2 * L
    m = n + 2  # in Hege's notation: 2 * (L + m). 
    # I'm assuming that the range of my operator is m = 1, i.e. H_xy = 0 for d(x,y) > m. Is this a good assumption?
    sections_specs = {
        '0': dict(m=m, n=n, shift=0)
    }

    # sections_specs = select_rectangular_sections(H_lambda, m=52, n=50, d=38)

    H_eigenvalues, H_eigenvectors, H_sections = compute_eigenvalues_and_singular_values(H)

    H_lambda_eigenvalues, H_lambda_eigenvectors, H_lambda_sections = compute_eigenvalues_and_singular_values(H_lambda, sections_specs)

    dist = dist_lambda_spec_H(lmbd, H_eigenvalues)
    print('d(λ, Spec(H))')
    for i in range(5):
        print(f'd{i + 1}: {dist[i]}')

    print()

    s = np.sort(H_lambda_sections['0']['sv'], kind='stable')
    for i in range(5):
        print(f's{i + 1}: {s[i]}')

    # generate_plot(L, perturb_H, H_lambda_eigenvalues, H_lambda_sections, plots_subfolder)

def free_hamiltonian_lambda_with_variations_of_uneven_sections():
    '''
    Compute the spectrum of (H - λ), where H is the Hamiltonian of the one-dimensional time-independent free Schroedinger equation with periodic boundary conditions
    and λ is any real number. 
    Calculate d(λ, Spec(H)) (as an array) and compare this against the singular values of various uneven sections.
    '''
    length = 1000  # space length
    dx = 1.0  # step size
    perturb_H = True
    
    H = make_free_hamiltonian(length=length, perturb_H=perturb_H, random_rng=(-0.2, 0.2))
    H_eigenvalues, H_eigenvectors, _ = compute_eigenvalues_and_singular_values(H)
    H_rows = H.shape[0]
    H_diag_middle_idx = middle_value(H_rows)
    I = np.identity(n=H_rows)

    for n in range(50, 300, 50):  # in Hege's notation: 2 * L
        for lambda_idx in range(1, 6):
            # lmbd = lambda_idx * 0.5
            lmbd = 3
            for shift in range(-200, 250, 50):
                m = n + 2  # in Hege's notation: 2 * (L + m). 
                # I'm assuming that the range of my operator is m = 1, i.e. H_xy = 0 for d(x,y) > m. Is this a good assumption?
                sections_specs = {
                    'my section': dict(m=m, n=n, shift=shift)
                }

                H_lambda = H - lmbd * I  # H - λ
                H_lambda_eigenvalues, H_lambda_eigenvectors, H_lambda_sections = compute_eigenvalues_and_singular_values(H_lambda, sections_specs)

                dist = dist_lambda_spec_H(lmbd, H_eigenvalues)  # d(λ, Spec(H))
                s = np.sort(H_lambda_sections['my section']['sv'], kind='stable')
                L = 0.5 * n
                x = (H_diag_middle_idx + shift) * dx
                
                print()
                print(f'L: {L}, λ: {lmbd}, x: {x}')
                print()
                print(f'    d(λ, Spec(H)):              Singular values of Q_Lλx:')
                for i in range(5):
                    print(f'{i + 1} - {dist[i]}       {s[i]}')
                
                print()

if __name__ == '__main__':
    # free_hamiltonian()
    # free_hamiltonian_lambda()
    free_hamiltonian_lambda_with_variations_of_uneven_sections()
    print(f'{__file__} complete!')
