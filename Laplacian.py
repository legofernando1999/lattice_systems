'''
Discretization of the one-dimensional time-independent Schroedinger equation for a free particle with periodic boundary conditions.
'''
import numpy as np
from scipy import linalg as la
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
from hamiltonian import Hamiltonian
# import plotly.express as px
# import plotly.io as pio

# pio.renderers.default = 'browser'

# read directory from pathlib library (returns PosixPath object)
ROOT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = ROOT_DIR / 'plots'
HAMILTONIANS_DIR = ROOT_DIR / 'hamiltonians'

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
    
def choose_x(perturb_H, y_axis, N, r):
    '''
    Choose an x based on the index location of the highest value of `y_axis`.
    '''
    if perturb_H:
        x = np.argsort(y_axis)[-1]
        x = int(x)

        while x - r <= 0:
            x = x + 1

        while x + r >= N:
            x = x - 1
    else:
        x = middle_value(N)

    return x
    
def is_hermitian(a: np.ndarray, tol=1.e-10) -> bool:
    '''Determine if square matrix is Hermitian.'''
    return np.allclose(a, a.conj().T, atol=tol)

def _determine_indices_for_uneven_section(N, nrows, ncols, shift=0) -> tuple[int, int, int, int]:
    '''
    Helper function for determining indices of parent matrix that are occupied by uneven section.

    Parameters
    ----------
    N : int
        Number of rows / columns of parent matrix.
    nrows : int
        Number of rows of uneven section.
    ncols : int
        Number of columns of uneven section.
    shift : int, optional
        Location along the main diagonal from where the uneven section will be extracted. Default is 0.

    Returns
    -------
    tuple
        Indices of parent matrix that are occupied by uneven section.
    '''
    diag_mid_val_idx = middle_value(N)
    nrows_mid_val = middle_value(nrows)
    ncols_mid_val = middle_value(ncols)

    left_row_idx = diag_mid_val_idx - nrows_mid_val + shift
    right_row_idx = diag_mid_val_idx + nrows_mid_val + shift - (nrows % 2)
    left_column_idx = diag_mid_val_idx - ncols_mid_val + shift
    right_column_idx = diag_mid_val_idx + ncols_mid_val + shift - (ncols % 2)

    return left_row_idx, right_row_idx, left_column_idx, right_column_idx

def extract_uneven_section(matrix, nrows, ncols, shift=0) -> np.ndarray:
    '''
    Extract an uneven section from a square matrix.
    
    Parameters
    ----------
    matrix : ndarray
        Matrix from which the uneven section will be extracted.
    nrows : int
        Number of rows of uneven section.
    ncols : int
        Number of columns of uneven section.
    shift : int, optional
        Location along the main diagonal from where the uneven section will be extracted. Default is 0.
        If 0, the uneven section will be centered around the middle value of the main diagonal of the matrix.
        else, the center of the uneven section will be shifted by j steps about its middle value along the diagonal.
        The middle value of the diagonal is defined like so: if the number of rows k of the matrix is odd, then the middle value is the ((k + 1) / 2)th value, 
        otherwise it is the (k / 2)th value.

    Returns
    -------
    ndarray
        Uneven section.
    '''
    N = matrix.shape[0]  # number of rows / columns of matrix
    left_row_idx, right_row_idx, left_column_idx, right_column_idx = _determine_indices_for_uneven_section(N, nrows, ncols, shift)

    if left_row_idx < 0 or left_column_idx < 0 or right_row_idx > N or right_column_idx > N:
        raise ValueError('The size and/or location of the rectangular section is not compatible with the matrix.')
    else:
        return matrix[left_row_idx : right_row_idx, left_column_idx : right_column_idx]
    
def select_uneven_sections(matrix_shape, nrows, ncols, d) -> dict:
    '''
    Generate dictionary of specifications for multiple uneven sections of size nrows by ncols.

    Parameters
    ----------
    matrix_shape : ndarray
        Shape of matrix from which the uneven sections will be taken.
    nrows : int
        Number of rows of each uneven section.
    ncols : int
        Number of columns of each uneven section.
    d : int
        Separation distance between the centers of consecutive uneven sections.

    Returns
    -------
    sections_specs : dict
        Specifications for the different uneven sections to be extracted from matrix.
    '''
    N = matrix_shape[0] # Number of rows / columns of matrix
    left_row_idx, right_row_idx, left_column_idx, right_column_idx = _determine_indices_for_uneven_section(N, nrows, ncols)

    def is_valid_shift(j_val):
        shift = j_val * d
        return (0 <= left_row_idx + shift and right_row_idx + shift <= N and
                0 <= left_column_idx + shift and right_column_idx + shift <= N)

    sections_specs = {}
    for direction in [1, -1]:
        j = 0 if direction == 1 else -1
        while is_valid_shift(j):
            sections_specs[f'section {j}'] = {'nrows': nrows, 'ncols': ncols, 'shift': j * d}
            j += direction

    return sections_specs

def svd_uneven_sections(H, sections_specs, singular_vals_only=False) -> dict:
    '''
    Perform the singular value decomposition of the uneven sections extracted from H.

    Parameters
    ----------
    H : ndarray
        Hamiltonian matrix.
    sections_specs : dict
        Specifications for the different uneven sections to be extracted from H.
    singular_vals_only : bool, optional
        If True, only compute singular values of uneven sections. If False, compute U, S, and V. Default is `False`.

    Returns
    -------
    sections : dict
        A dictionary of SVD results for each spec. Each entry contains:
        - 'A': The extracted sub-matrix.
        - 'S': The singular values.
        - 'U', 'V' (ndarray, optional): The left and right singular vectors. Only present if `sing_vals_only` is `False`. Note: 'V' is returned as the matrix of vectors, not the adjoint (VH).

    Raises
    ------
    RuntimeError
        If the SVD computation fails to converge.

    Notes
    -----
    1. SVD implementation uses `full_matrices = False` to optimize memory usage for rectangular sections.
    '''
    sections = {}
    for k, v in sections_specs.items():
        section = extract_uneven_section(H, nrows=v['nrows'], ncols=v['ncols'], shift=v['shift'])

        # Singular Value Decomposition: A = U * S * VT (A is any m by n matrix), where the columns of U (m by m) are eigenvectors of A * AT
        # and the columns of V (n by n) are eigenvectors of AT * A. And S is diagonal (but rectangular, m by n). 
        # The r singular values on the diagonal of S are the square roots of the nonzero eigenvalues of both A * AT and AT * A.
        if singular_vals_only:
            s = la.svd(section, compute_uv=False, check_finite=False)
            sections[k] = {'A': section, 'S': s}
        else:
            # `full_matrices = False` is almost always better for rectangular SVD
            u, s, vh = la.svd(section, full_matrices=False, check_finite=False)
            sections[k] = {'A': section, 'U': u, 'S': s, 'V': vh.conj().T}

    return sections

def dist_lambda_spec_H(lmbd, spectrum) -> np.ndarray:
    '''
    Calculate d(λ, σ(H)).

    Returns
    -------
    ndarray
        Distances between λ and the spectrum of H.
    '''
    return np.abs(lmbd - spectrum)

def spectral_gap_bound(epsilon_r_lmbd, r, H, m, q, n=1) -> float:
    '''
    Compute the Spectral Gap Bound.

    Parameters
    ----------
    epsilon_r_lmbd : float
        Smallest singular value of Q_rλx.
    r : int
        Window size of Q_rλx.
    H : ndarray
        Hamiltonian matrix.
    m : float
        Finite range (maximal hopping length).
    q : float
        Packing radius of uniformly discrete set.
    n : int
        Dimension of space.

    Returns
    -------
    float
        Spectral Gap Bound.
    '''
    def bound_constant(H: np.ndarray, m: float, q: float, n: int) -> float:
        '''Compute the constant appearing in the Spectral Gap Bound.'''
        M = np.max(np.abs(H))
        return m * M * (36 * m / q) ** (0.5 * n)
    
    C = bound_constant(H, m, q, n)
    return epsilon_r_lmbd - C / r

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

def _create_figure(hist_data, fname) -> None:
    '''Create and save figure.'''
    n_subplots = 1
    height_ratios = None
    if any(key.startswith('Singular') for key in hist_data):
        n_subplots = 2
        height_ratios = (1, 1)

    fig, axs = plt.subplots(
        nrows=n_subplots, 
        sharex=True,
        height_ratios=height_ratios,
        # figsize=(5, 3)
        )
    
    if type(axs) == matplotlib.axes._axes.Axes:
        axs = np.array([axs], dtype=object)
    
    palette = sns.color_palette('colorblind', as_cmap=True)

    sv_array = np.array([], dtype=np.float64)
    for k, v in hist_data.items():
        if k.startswith('Eigenvalues'):
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
            sv_array = np.append(sv_array, v)

    if sv_array.size != 0: 
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

    fig.tight_layout()
    
    fig.savefig(
        fname=fname,
        dpi=800
        )

def generate_plot(L, H_perturbed, H_eigenvalues, H_sections, plots_subfolder) -> None:
    '''
    Prepare plotting data and create figure.

    Parameters
    ----------
    L : int
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
        fname = f'{FIG_DIR}/kde_perturbed_L={L}.png'
    else: 
        eigenvalues_plot_title = 'Eigenvalues nonperturbed Hamiltonian'
        fname = f'{FIG_DIR}/kde_nonperturbed_L={L}.png'

    fig_data = {eigenvalues_plot_title: H_eigenvalues}

    for k, v in H_sections.items():
        fig_data[f'Singular values {k}'] = v['S']
    
    _create_figure(fig_data, fname)

def free_hamiltonian():
    '''
    Compute the spectrum of the Hamiltonian of the one-dimensional time-independent free Schroedinger equation with periodic boundary conditions.
    '''
    L = 1000  # space length
    dx = 1.0  # step size
    perturb_H = True
    plots_subfolder = 'free_Hamiltonian'

    hamiltonian = Hamiltonian.construct_free_hamiltonian(L=L, dx=dx, perturb_H=perturb_H, random_rng=(-0.1, 0.1), eigvals_only=True)
    H = hamiltonian.matrix

    r = 50
    m = 1  # maximal hopping length
    ncols = 2 * r
    nrows = 2 * (r + m)
    sections_specs = select_uneven_sections(hamiltonian.shape, nrows=nrows, ncols=ncols, d=5)
    # sections_specs = {
    #     'section 1': dict(nrows=nrows, ncols=ncols, shift=0)
    # }

    H_sections = svd_uneven_sections(H, sections_specs, singular_vals_only=True)

    generate_plot(L, perturb_H, hamiltonian.eigenvalues, H_sections, plots_subfolder)

def free_hamiltonian_lambda():
    '''
    Compute the spectrum of (H - λ), where H is the Hamiltonian of the free one-dimensional time-independent Schroedinger equation with periodic boundary conditions
    and λ is any real number. 
    '''
    L = 1000  # space length
    dx = 1.0  # step size
    perturb_H = False
    lmbd = -0.5  # λ
    r = 150  # uneven section window size
    m = 1  # maximal hopping length
    num_eig = 5
    plots_subfolder='H_lambda'
    save_hamiltonian = False
    hamiltonian_filename = 'hamiltonian_3.json'

    # Construct new Hamiltonian
    hamiltonian = Hamiltonian.construct_free_hamiltonian(L=L, dx=dx, perturb_H=perturb_H, random_rng=(-0.2, 0.2))

    # Retrieve Hamiltonian from JSON file
    # from_json_path = HAMILTONIANS_DIR / 'hamiltonian_1.json'
    # hamiltonian = Hamiltonian.from_json(from_json_path)

    H = hamiltonian.matrix
    N = hamiltonian.shape[0]  # Number of rows / columns of H
    H_eigenvalues, H_eigenvectors = hamiltonian.eigenvalues, hamiltonian.eigenvectors

    dist = dist_lambda_spec_H(lmbd, H_eigenvalues)  # d(λ, σ(H))
    dist_sorted = np.sort(dist, kind='stable')
    dist_sorted_idx = np.argsort(dist)

    x_axis = np.linspace(start=0, stop=L, num=N)

    # Plot the absolute square of the eigenvectors corresponding to the 5 closest eigenvalues to λ
    title = f'Eigenvectors of the eigenvalues closest to λ = {lmbd}'
    fig, ax = plt.subplots(figsize=(8, 6))
    for j in range(num_eig):
        eig_idx = dist_sorted_idx[j]
        eigvalj = H_eigenvalues[eig_idx]
        yj_axis = np.abs(H_eigenvectors[:, eig_idx]) ** 2
        # yj_axis = H_eigenvectors[:, eig_idx]
        if j == 0:
            # Choose an x for the uneven section based on the highest value of the eigenvector corresponding to the closest eingenvalue to λ
            x = choose_x(perturb_H, yj_axis, N, r)
        
        sns.scatterplot(y=yj_axis, x=x_axis, ax=ax, label=f'{j + 1}: {eigvalj:.5f}')

    # Highlight the X-axis interval from X = x - (r + m) to X = x + (r + m)
    ax.axvspan(x - r - m, x + r + m, color='yellow', alpha=0.1, label='Uneven Section')
    ax.set_title(title)
    ax.legend(title='Eigenvalues:')
    ax.set_xlabel('Space length')
    ax.set_ylabel('Absolute square of eigenvector')

    plt.show()

    # Convert Hamiltonian to JSON file.
    if save_hamiltonian:
        to_json_path = HAMILTONIANS_DIR / hamiltonian_filename
        hamiltonian.to_json(to_json_path)

    # Calculate shift from x
    # x = (H_diag_middle_idx + shift) * dx
    shift = int(x / dx) - middle_value(N)
    
    ncols = 2 * r
    nrows = 2 * (r + m)

    sections_specs = {
        '(H - λ) section': dict(nrows=nrows, ncols=ncols, shift=shift)
    }

    I = np.identity(n=N)
    hamiltonian_lambda = Hamiltonian((H - lmbd * I), eigvals_only=True)  # H - λ
    H_lambda = hamiltonian_lambda.matrix

    H_lambda_sections = svd_uneven_sections(H_lambda, sections_specs, singular_vals_only=True)

    s_sorted = np.sort(H_lambda_sections['(H - λ) section']['S'], kind='stable')
    
    print()
    print(f'r: {r}, λ: {lmbd}, x: {x}')
    print()
    print(f'   d(λ, σ(H)):        Singular values of Q_rλx:')
    for i in range(num_eig):
        print(f'{i + 1}: {dist_sorted[i]:.8f}        {s_sorted[i]:.8f}')
    
    print()

    # generate_plot(L, perturb_H, hamiltonian_lambda.eigenvalues, H_lambda_sections, plots_subfolder)

def lower_norm_fct_bounds():
    '''
    Compute the Pseudospectral Inclusion Bound (upper bound) and the Spectral Gap Bound (lower bound) for the Hamiltonian of
    the free one-dimensional time-independent Schrödinger equation with periodic boundary conditions.
    '''
    L = 1000  # space length
    dx = 1.0  # step size
    perturb_H = False
    r_range = range(50, 300, 50)  # uneven section window size
    lmbd_range = (-0.1, 0.5, 1.2, 2.3)  # λ
    x = 501
    m = 1  # maximal hopping length

    # Construct new Hamiltonian
    hamiltonian = Hamiltonian.construct_free_hamiltonian(L=L, dx=dx, perturb_H=perturb_H, random_rng=(-0.2, 0.2), eigvals_only=True)

    H = hamiltonian.matrix
    N = hamiltonian.shape[0]  # Number of rows / columns of H
    I = np.identity(n=N)  # Identity matrix with the same shape as H
    H_eigenvalues = hamiltonian.eigenvalues

    print()
    print('| {:^8} | {:^10} | {:^11} | {:^11} |'.format('(r, λ)', 'd(λ, σ(H))', 'Upper bound', 'Lower bound'))
    print()

    for lmbd in lmbd_range:
        for r in r_range:
            dist = dist_lambda_spec_H(lmbd, H_eigenvalues)  # d(λ, σ(H))
            dist_sorted = np.sort(dist, kind='stable')

            # Calculate shift from x
            shift = int(x / dx) - middle_value(N)
            
            ncols = 2 * r
            nrows = 2 * (r + m)

            sections_specs = {
                '(H - λ) section': dict(nrows=nrows, ncols=ncols, shift=shift)
            }
            
            hamiltonian_lambda = Hamiltonian((H - lmbd * I), eigvals_only=True)  # H - λ
            H_lambda = hamiltonian_lambda.matrix

            H_lambda_sections = svd_uneven_sections(H_lambda, sections_specs, singular_vals_only=True)
            s_sorted = np.sort(H_lambda_sections['(H - λ) section']['S'], kind='stable')

            epsilon_r_lmbd_x = s_sorted.min()
            epsilon_r_lmbd = epsilon_r_lmbd_x  # Due to periodicity
            lower_bound = spectral_gap_bound(epsilon_r_lmbd=epsilon_r_lmbd, r=r, H=H_lambda, m=m, q=dx)

            # LaTeX format
            r_lmbd_str = f'$({r}, {lmbd})$'
            print('{} & {:.6f} & {:.6f} & {:.6f} \\\\'.format(r_lmbd_str, dist_sorted[0], s_sorted[0], lower_bound))
        
        print('\\hline')

if __name__ == '__main__':
    # free_hamiltonian()
    # free_hamiltonian_lambda()
    lower_norm_fct_bounds()
    print(f'{__file__} complete!')
