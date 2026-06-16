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

def _determine_indices_for_rectangular_section(N, nrows, ncols, shift=0) -> tuple[int, int, int, int]:
    '''
    Helper function for determining indices of parent matrix that are occupied by rectangular section.

    Parameters
    ----------
    N : int
        Number of rows / columns of parent matrix.
    nrows : int
        Number of rows of rectangular section.
    ncols : int
        Number of columns of rectangular section.
    shift : int, optional
        Location along the main diagonal from where the rectangular section will be extracted. Default is 0.

    Returns
    -------
    tuple
        Indices of parent matrix that are occupied by rectangular section.
    '''
    diag_mid_val_idx = middle_value(N)
    nrows_mid_val = middle_value(nrows)
    ncols_mid_val = middle_value(ncols)

    left_row_idx = diag_mid_val_idx - nrows_mid_val + shift
    right_row_idx = diag_mid_val_idx + nrows_mid_val + shift - (nrows % 2)
    left_column_idx = diag_mid_val_idx - ncols_mid_val + shift
    right_column_idx = diag_mid_val_idx + ncols_mid_val + shift - (ncols % 2)

    return left_row_idx, right_row_idx, left_column_idx, right_column_idx

def extract_rectangular_section(A, nrows, ncols, shift=0):
    '''
    Extract a rectangular section from a square matrix.
    
    Parameters
    ----------
    A : ndarray
        Square matrix from which the rectangular section will be extracted.
    nrows : int
        Number of rows of rectangular section.
    ncols : int
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
    N = A.shape[0]  # number of rows / columns of matrix A
    left_row_idx, right_row_idx, left_column_idx, right_column_idx = _determine_indices_for_rectangular_section(N, nrows, ncols, shift)

    if left_row_idx < 0 or left_column_idx < 0 or right_row_idx > N or right_column_idx > N:
        raise ValueError('The size and/or location of the rectangular section is not compatible with the matrix.')
    else:
        return A[left_row_idx : right_row_idx, left_column_idx : right_column_idx]
    
def select_rectangular_sections(H, nrows, ncols, d) -> dict:
    '''
    Generate dictionary of specifications for multiple rectangular sections of size nrows by ncols.

    Parameters
    ----------
    H : ndarray
        Hamiltonian matrix.
    nrows : int
        Number of rows of each rectangular section.
    ncols : int
        Number of columns of each rectangular section.
    d : int
        Separation distance between the centers of consecutive rectangular sections.

    Returns
    -------
    sections_specs : dict
        Specifications for the different rectangular sections to be extracted from H.
    '''
    N = H.shape[0] # Number of rows / columns of H
    left_row_idx, right_row_idx, left_column_idx, right_column_idx = _determine_indices_for_rectangular_section(N, nrows, ncols)

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
    
def make_free_hamiltonian(L, dx, perturb_H=False, random_rng=(-0.1, 0.1)) -> np.ndarray:
    '''
    Construct free Hamiltonian matrix.
    
    Parameters
    ----------
    L : int
        Space length.
    dx : float
        Step size. Or spacing between particles.
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
    N = int(L / dx) + 1
    # dx = L / (N - 1)

    # solved using finite differences with periodic boundary conditions
    diag = -2. * np.ones(N)
    off_diag = np.ones(N - 1)
    a = -0.5 / dx ** 2

    H = a * (np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1))
    H[0, -1] = a
    H[-1, 0] = a

    if perturb_H:
        # Random values should be applied symmetrically, otherwise the matrix becomes non-Hermitian
        rng = np.random.default_rng()
        
        # perturbs the main diagonal
        H = H + np.diag(rng.uniform(low=random_rng[0], high=random_rng[1], size=N))
        
        # perturbs the sub-diagonals
        random_vals_sub_diag = rng.uniform(low=random_rng[0], high=random_rng[1], size=N - 1)
        H = H + np.diag(random_vals_sub_diag, k=1) + np.diag(random_vals_sub_diag, k=-1)
        
        # perturbs the upper right corner and the lower left corner
        # I think I shouldn't mess with these corners as they encode the boundary condition. I'm not sure this is an issue, though.
        # As long as I add the same random value to both corners, it should be fine, right?
        random_val_corner = rng.uniform(low=random_rng[0], high=random_rng[1])
        H[0, -1] = H[0, -1] + random_val_corner
        H[-1, 0] = H[-1, 0] + random_val_corner
    
    return H

def compute_eigenvalues_and_singular_values(H, sections_specs=None, eigvals_only=False, sing_vals_only=False) -> dict:
    '''
    Compute the eigenvalues and eigenvectors of H, and the singular value decomposition of uneven sections extracted from H.

    Parameters
    ----------
    H : ndarray
        Hamiltonian matrix.
    sections_specs : dict, optional
        Specifications for the different uneven sections to be extracted from H. Defaults to `None` (no sections processed).
    eigvals_only : bool, optional
        If True, only compute eigenvalues of H. If False, compute eigenvectors as well. Default is `False`.
    sing_vals_only : bool, optional
        If True, only compute singular values of uneven sections. If False, compute U, S, and V. Default is `False`.

    Returns
    -------
    results : dict
        A dictionary containing the following keys:
        - 'eigenvalues' (ndarray): The eigenvalues of H, each repeated according to its multiplicity.
        - 'eigenvectors' (ndarray, optional): The normalized eigenvectors of H. The eigenvector corresponding to the eigenvalue w[i] is the column v[:,i]. Only present if `eigvals_only` is `False`.
        - 'sections' (dict, optional): A dictionary of SVD results for each spec. Each entry contains:
           - 'A': The extracted sub-matrix.
           - 'S': The singular values.
           - 'U', 'V' (ndarray, optional): The left and right singular vectors. Only present if `sing_vals_only` is `False`. Note: 'V' is returned as the matrix of vectors, not the adjoint (VH).

    Raises
    ------
    RuntimeError
        If the eigenvalue or SVD computation fails to converge.

    Notes
    -----
    1. SVD implementation uses `full_matrices = False` to optimize memory usage for rectangular sections.
    '''
    results = {}
    sections_specs = sections_specs or {}

    try:
        if is_hermitian(H):
            # uses a more efficient algorithm for Hermitian matrices
            if eigvals_only:
                results['eigenvalues'] = la.eigvalsh(H, check_finite=False)
            else:
                eigvals, eigvecs = la.eigh(H, check_finite=False)
                results['eigenvalues'], results['eigenvectors'] = eigvals, eigvecs
        else:
            # uses a general algorithm
            if eigvals_only:
                eigvals = la.eigvals(H, check_finite=False)
            else:
                eigvals, eigvecs = la.eig(H, check_finite=False)
                results['eigenvectors'] = eigvecs

            # Non-Hermitian eigenvalues might be complex; only cast to float if purely real
            results['eigenvalues'] = np.real_if_close(eigvals)

    except la.LinAlgError as e:
        raise RuntimeError(f'Eigenvalue computation failed to converge: {e}')

    if sections_specs:
        sections = {}
        for k, v in sections_specs.items():
            section = extract_rectangular_section(H, nrows=v['nrows'], ncols=v['ncols'], shift=v['shift'])

            # Singular Value Decomposition: A = U * S * VT (A is any m by n matrix), where the columns of U (m by m) are eigenvectors of A * AT
            # and the columns of V (n by n) are eigenvectors of AT * A. And S is diagonal (but rectangular, m by n). 
            # The r singular values on the diagonal of S are the square roots of the nonzero eigenvalues of both A * AT and AT * A.
            if sing_vals_only:
                s = la.svd(section, compute_uv=False, check_finite=False)
                sections[k] = {'A': section, 'S': s}
            else:
                # `full_matrices = False` is almost always better for rectangular SVD
                u, s, vh = la.svd(section, full_matrices=False, check_finite=False)
                sections[k] = {'A': section, 'U': u, 'S': s, 'V': vh.conj().T}
            
        results['sections'] = sections

    return results


def dist_lambda_spec_H(lmbd, spectrum):
    '''
    Calculate d(λ, σ(H)).

    Returns
    -------
    ndarray
        Distances between λ and the spectrum of H.
    '''
    return np.abs(lmbd - spectrum)

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
            # axs[1].set_yticks([])

    fig.tight_layout()
    
    fig.savefig(
        fname=fname,
        dpi=800
        )

def generate_plot(L, H_perturbed, H_eigenvalues, H_sections, plots_subfolder):
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
    perturb_H = False
    plots_subfolder = 'free_Hamiltonian'

    H = make_free_hamiltonian(L=L, dx=dx, perturb_H=perturb_H, random_rng=(-0.1, 0.1))

    r = 50
    m = 1  # maximal hopping length
    ncols = 2 * r
    nrows = 2 * (r + m)
    sections_specs = select_rectangular_sections(H, nrows=nrows, ncols=ncols, d=5)
    # sections_specs = {}

    results = compute_eigenvalues_and_singular_values(H, sections_specs, eigvals_only=True, sing_vals_only=True)
    H_sections = results.get('sections', {})

    generate_plot(L, perturb_H, results['eigenvalues'], H_sections, plots_subfolder)

def free_hamiltonian_lambda():
    '''
    Compute the spectrum of (H - λ), where H is the Hamiltonian of the one-dimensional time-independent free Schroedinger equation with periodic boundary conditions
    and λ is any real number. 
    '''
    L = 1000  # space length
    dx = 1.0  # step size
    perturb_H = True
    lmbd = 0.5  # λ
    plots_subfolder='H_lambda'

    H = make_free_hamiltonian(L=L, dx=dx, perturb_H=perturb_H, random_rng=(-0.2, 0.2))
    H_results = compute_eigenvalues_and_singular_values(H)
    H_eigenvalues, H_eigenvectors = H_results['eigenvalues'], H_results['eigenvectors']

    # Uneven section "radius"
    r = 100
    m = 1  # maximal hopping length
    ncols = 2 * r
    nrows = 2 * (r + m)

    dist = dist_lambda_spec_H(lmbd, H_eigenvalues)  # d(λ, σ(H))
    dist_sorted_idx = np.argsort(dist)

    N = H.shape[0]  # Number of rows / columns of H
    x_axis = np.linspace(start=0, stop=L, num=N)

    # Plot the absolute square of the eigenvectors corresponding to the 5 closest eigenvalues to λ
    num_eig = 5
    title = f'Eigenvectors of the eigenvalues closest to λ = {lmbd}'
    fig, ax = plt.subplots(figsize=(8, 6))
    for j in range(num_eig):
        eig_idx = dist_sorted_idx[j]
        eigvalj = H_eigenvalues[eig_idx]
        yj_axis = np.abs(H_eigenvectors[:, eig_idx]) ** 2
        if j == 0:
            # Choose an x for the uneven section based on the highest value of the eigenvector corresponding to the closest eingenvalue to λ
            x = choose_x(perturb_H, yj_axis, N, r)
        
        sns.scatterplot(y=yj_axis, x=x_axis, ax=ax, label=f'{j + 1} - {eigvalj:.5f}')

    ax.set_title(title)
    ax.legend(title='Eigenvalues:')
    ax.set_xlabel('Space length')
    ax.set_ylabel('Absolute square of eigenvector')

    plt.show()

    # Calculate shift from x
    # x = (H_diag_middle_idx + shift) * dx
    shift = int(x / dx) - middle_value(N)
    
    sections_specs = {
        '(H - λ) section': dict(nrows=nrows, ncols=ncols, shift=shift)
    }

    I = np.identity(n=N)
    H_lambda = H - lmbd * I  # H - λ

    H_lambda_results = compute_eigenvalues_and_singular_values(H_lambda, sections_specs, eigvals_only=True, sing_vals_only=True)
    H_lambda_sections = H_lambda_results['sections']

    d = np.sort(dist, kind='stable')
    s = np.sort(H_lambda_sections['(H - λ) section']['S'], kind='stable')
    
    print()
    print(f'r: {r}, λ: {lmbd}, x: {x}')
    print()
    print(f'    d(λ, σ(H)):        Singular values of Q_rλx:')
    for i in range(5):
        print(f'{i + 1} - {d[i]:.8f}         {s[i]:.8f}')
    
    print()

    # generate_plot(L, perturb_H, H_lambda_results['eigenvalues'], H_lambda_sections, plots_subfolder)

if __name__ == '__main__':
    # free_hamiltonian()
    free_hamiltonian_lambda()
    print(f'{__file__} complete!')
