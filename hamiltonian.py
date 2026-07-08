import numpy as np
from scipy import linalg as la
from pathlib import Path
import json

class Hamiltonian:
    
    def __init__(self, matrix, is_hermitian=None, eigvals_only=False) -> None:
        '''
        Class containing the Hamiltonian matrix and some information about it.
        
        Parameters
        ----------
        matrix : ndarray
            Matrix representation of Hamiltonian operator.
        is_hermitian : bool
            Indicates whether the Hamiltonian is Hermitian (self-adjoint).
        eigvals_only : bool
            If True, only compute eigenvalues of Hamiltonian. If False, compute eigenvectors as well. Default is False.
        '''
        self.matrix = matrix
        self.shape = matrix.shape
        self.is_hermitian = is_hermitian if is_hermitian is not None else self._is_hermitian_fct()
        results = self._solve_eigenvalue_problem(eigvals_only=eigvals_only)
        self.eigenvalues = results.get('eigenvalues')
        self.eigenvectors = results.get('eigenvectors')

    @classmethod
    def construct_free_hamiltonian(cls, L, dx, perturb_H=False, random_rng=(-0.1, 0.1),  eigvals_only=False):
        '''
        Construct the matrix representation of the free-particle Hamiltonian
        H_free ψ(x) = -1/2 d^2/dx^2 ψ(x) = E ψ(x), for 0 ≤ x ≤ L, 
        using the finite difference method with periodic boundary conditions ψ(0) = ψ(L).

        Parameters
        ----------
        L : int
            Space length.
        dx : float
            Discretization step size.
        perturb_H : bool, optional
            Whether to perturb the Hamiltonian.
            If True, random values are added to / subtracted from the nonzero entries of the matrix. Default is False.
        random_rng : tuple[float, float], optional
            Minimum and maximum values for range of random values used in the perturbation of H. Default is (-0.1, 0.1).
        eigvals_only : bool, optional
            If True, only compute eigenvalues of Hamiltonian. If False, compute eigenvectors as well. Default is `False`.

        Returns
        -------
        Instance of 'Hamiltonian` class.
        '''
        N = int(L / dx) + 1
        # dx = L / (N - 1)

        diag = -2. * np.ones(N)
        off_diag = np.ones(N - 1)
        a = -0.5 / dx ** 2

        H = a * (np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1))
        H[0, -1] = a
        H[-1, 0] = a

        if perturb_H:
            # Random values should be applied symmetrically, otherwise the matrix becomes non-Hermitian
            rng = np.random.default_rng()
            
            # Perturb the main diagonal. This represents the addition of a random potential
            H = H + np.diag(rng.uniform(low=random_rng[0], high=random_rng[1], size=N))
            
            # # Perturb sub-diagonals
            # random_vals_sub_diag = rng.uniform(low=random_rng[0], high=random_rng[1], size=N - 1)
            # H = H + np.diag(random_vals_sub_diag, k=1) + np.diag(random_vals_sub_diag, k=-1)
            
            # # Perturb upper right and lower left corners
            # # I think I shouldn't mess with these corners as they encode the periodic boundary conditions. I'm not sure this is an issue, though.
            # # As long as I perturb both corners equally, it should be fine, right?
            # random_val_corner = rng.uniform(low=random_rng[0], high=random_rng[1])
            # H[0, -1] = H[0, -1] + random_val_corner
            # H[-1, 0] = H[-1, 0] + random_val_corner

        return cls(matrix=H, is_hermitian=True, eigvals_only=eigvals_only)

    def _is_hermitian_fct(self, tol=1.e-10):
        '''Determine if Hamiltonian is Hermitian.'''
        H = self.matrix
        return np.allclose(H, H.conj().T, atol=tol)
    
    def _solve_eigenvalue_problem(self, eigvals_only=False) -> dict:
        '''
        Compute the eigenvalues and eigenvectors of the Hamiltonian matrix.

        Parameters
        ----------
        eigvals_only : bool, optional
            If True, only compute eigenvalues of Hamiltonian. If False, compute eigenvectors as well. Default is `False`.

        Returns
        -------
        results : dict
            A dictionary containing the following keys:
            - 'eigenvalues' (ndarray): The eigenvalues of the Hamiltonian, each repeated according to its multiplicity.
            - 'eigenvectors' (ndarray, optional): The normalized eigenvectors of the Hamiltonian. The eigenvector corresponding to the eigenvalue w[i] is the column v[:,i]. 
              Only present if `eigvals_only` is `False`.
        
        Raises
        ------
        RuntimeError
            If the eigenvalue computation fails to converge.
        '''
        results = {}
        H = self.matrix

        try:
            if self.is_hermitian:
                # Use a more efficient algorithm for Hermitian matrices
                if eigvals_only:
                    results['eigenvalues'] = la.eigvalsh(H, check_finite=False)
                else:
                    eigvals, eigvecs = la.eigh(H, check_finite=False)
                    results['eigenvalues'], results['eigenvectors'] = eigvals, eigvecs
            else:
                # Use a general algorithm
                if eigvals_only:
                    eigvals = la.eigvals(H, check_finite=False)
                else:
                    eigvals, eigvecs = la.eig(H, check_finite=False)
                    results['eigenvectors'] = eigvecs

                # Non-Hermitian eigenvalues might be complex; only cast to float if purely real
                results['eigenvalues'] = np.real_if_close(eigvals)

        except la.LinAlgError as e:
            raise RuntimeError(f'Eigenvalue computation failed to converge: {e}')
        
        return results

    def to_json(self, file_path=None):
        '''
        Serialize class instance to a JSON-formatted string.

        Parameters
        ----------
        file_path : None or Path
            - if None, returns JSON-formatted string.
            - if Path, saves JSON-formatted string to Path.

        Returns
        -------
        JSON formatted string.
        '''
        H = self.matrix
        if self.is_hermitian:
            diag = H.diagonal().tolist()
            subdiag = H.diagonal(-1).tolist()
            lower_left_corner = float(H[-1, 0])

            d = {
                'shape': self.shape,
                'is_Hermitian': True,
                'diagonal': diag,
                'subdiagonal': subdiag,
                'lower-left corner': lower_left_corner
            }

        else:
            d = {
                'shape': self.shape,
                'is_Hermitian': False,
                'H': H.tolist()
            }

        json_str = json.dumps(d, indent=2)

        if file_path is None:
            return json_str
        else:
            with open(file_path, 'w') as f:
                f.write(json_str)

    @classmethod
    def from_json(cls, json_data, eigvals_only=False):
        '''
        Deserialize JSON data (from a JSON string or JSON file) to a class instance.

        Parameters
        ----------
        json_data : str, dict
            The input JSON data, which can be a JSON string or a file path to a JSON file
        eigvals_only : bool, optional
            If True, only compute eigenvalues of Hamiltonian. If False, compute eigenvectors as well. Default is `False`.

        Returns
        -------
        Hamiltonian
            An instance of the Hamiltonian class constructed from the input JSON data.

        Raises
        ------
        ValueError
            If the input is an invalid JSON string or invalid file path.
        '''
        if isinstance(json_data, Path):
            try:
                with open(json_data, 'r') as f:
                    json_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise ValueError("Invalid file path.") from e

        elif isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON string.") from e
            
        # json_data is now a dict
        is_hermitian = json_data['is_Hermitian']
        
        if is_hermitian:
            H = np.diag(json_data['diagonal']) + np.diag(json_data['subdiagonal'], k=1) + np.diag(json_data['subdiagonal'], k=-1)
            H[0, -1] = json_data['lower-left corner']
            H[-1, 0] = json_data['lower-left corner']
        
        else:
            H = np.array(json_data['H'])

        return cls(H, is_hermitian, eigvals_only)
    

if __name__ == "__main__":
    print("`Hamiltonian` class.")
