# Lattice Systems: Spectrum Analysis of the Schrödinger Hamiltonian

A computational physics project focused on analyzing the spectrum of the one-dimensional time-independent Schrödinger equation with periodic boundary conditions, particularly for investigating lattice and quasicrystal systems.

## Overview

This project implements numerical methods to compute and visualize eigenvalues and singular values of discrete Hamiltonian matrices arising from finite difference discretizations of the free Schrödinger equation. A key focus is understanding the relationship between local perturbations in the Hamiltonian and the spectrum through singular value analysis.

## Main Features

- **Hamiltonian Construction**: Creates discretized Hamiltonian matrices using finite differences with periodic boundary conditions
- **Spectral Analysis**: Computes eigenvalues and eigenvectors using efficient algorithms for Hermitian matrices
- **Rectangular Section Extraction**: Extracts multiple rectangular subsections from the Hamiltonian matrix with configurable spacing
- **Singular Value Decomposition**: Performs SVD on extracted sections to analyze local spectral properties
- **Visualization**: Generates publication-quality plots using kernel density estimation (KDE) with boundary bias correction
- **Perturbation Analysis**: Supports random perturbations to the Hamiltonian for studying localization effects

## Project Structure

```
.
├── Laplacian.py              # Main computational module
├── environment.yml           # Conda environment specification
├── README.md                 # This file
├── meetings_with_Teufel.md  # Meeting notes and research progress
├── plots/                    # Output directory for generated plots
│   ├── free_Hamiltonian/    # Plots from free Hamiltonian spectrum analysis
│   └── H_lambda/            # Plots from (H - λ) analysis
└── docs/                     # Supporting documentation and references
    ├── articles/            # Research papers on quasicrystals and spectral theory
    └── theses/              # Related academic theses
```

## Installation

### Prerequisites

- Python 3.14+
- Anaconda/Miniconda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/legofernando1999/lattice_systems.git
cd lattice_systems
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate QMenv
```

## Usage

### Basic Example: Free Hamiltonian Spectrum

```python
from Laplacian import free_hamiltonian

# Compute and plot the spectrum of the free Hamiltonian
free_hamiltonian()
```

This function:
1. Constructs a free Hamiltonian matrix (L=1000, dx=1.0)
2. Applies random perturbations to simulate a non-periodic system
3. Extracts multiple rectangular sections (52×50) with spacing d=5
4. Computes eigenvalues and singular values
5. Generates KDE plots with boundary bias correction

### Advanced Example: Distance Analysis (H - λ)

```python
from Laplacian import free_hamiltonian_lambda

# Analyze distance from arbitrary λ to the spectrum
free_hamiltonian_lambda()
```

This function:
1. Constructs a perturbed Hamiltonian (L=500, dx=1.0)
2. Computes all eigenvalues and eigenvectors
3. Selects a test value λ = 0.5
4. Constructs (H - λ) and extracts a central section
5. Compares distances d(λ, Spec(H)) with singular values of the (H - λ) section
6. Prints results for verification against theoretical predictions

## Key Functions

### Core Computational Functions

#### `make_free_hamiltonian(length, dx, perturb_H=False, random_rng=(-0.1, 0.1))`

Constructs a discretized free particle Hamiltonian using finite differences with periodic boundary conditions.

**Parameters:**
- `length` (int): Spatial domain length
- `dx` (float): Grid spacing
- `perturb_H` (bool): Whether to apply random perturbations
- `random_rng` (tuple): Range of random values for perturbation

**Returns:**
- `H` (ndarray): Hermitian Hamiltonian matrix

#### `extract_rectangular_section(A, m, n, shift=0)`

Extracts an m×n rectangular section from a k×k matrix, optionally shifted along the diagonal.

**Parameters:**
- `A` (ndarray): Input square matrix
- `m` (int): Number of rows in section
- `n` (int): Number of columns in section
- `shift` (int): Diagonal offset for section center

**Returns:**
- Section (ndarray): The extracted m×n matrix

#### `select_rectangular_sections(H, m, n, d)`

Generates specifications for multiple overlapping rectangular sections covering the full Hamiltonian.

**Parameters:**
- `H` (ndarray): Hamiltonian matrix
- `m` (int): Row count per section
- `n` (int): Column count per section
- `d` (int): Spacing between section centers

**Returns:**
- `sections_specs` (dict): Configuration for section extraction

#### `compute_eigenvalues_and_singular_values(H, sections_specs=None, eigvals_only=False, sing_vals_only=False)`

Comprehensive spectral analysis function combining eigendecomposition and SVD.

**Parameters:**
- `H` (ndarray): Hamiltonian matrix
- `sections_specs` (dict, optional): Section specifications for SVD
- `eigvals_only` (bool): If True, compute only eigenvalues
- `sing_vals_only` (bool): If True, compute only singular values

**Returns:**
- `results` (dict): Dictionary containing:
  - `eigenvalues` (ndarray): Eigenvalues of H
  - `eigenvectors` (ndarray): Eigenvectors of H (if `eigvals_only=False`)
  - `sections` (dict): SVD results for each section (if provided)

#### `generate_plot(length, H_perturbed, H_eigenvalues, H_sections, plots_subfolder)`

Creates publication-quality KDE plots with boundary bias correction.

**Parameters:**
- `length` (int): Spatial domain length (for figure title)
- `H_perturbed` (bool): Whether Hamiltonian was perturbed
- `H_eigenvalues` (ndarray): Eigenvalues to plot
- `H_sections` (dict): Sections dictionary with singular values
- `plots_subfolder` (str): Subdirectory for output plots

### Utility Functions

- `middle_value(k)`: Computes the middle index of an integer (handles even/odd cases)
- `is_hermitian(a, tol=1e-10)`: Verifies matrix Hermiticity
- `dist_lambda_spec_H(lmbd, H_eigenvalues)`: Computes distances from λ to all eigenvalues
- `_mirror_array(arr)`: Extends array by mirroring for KDE boundary bias correction
- `_determine_indices_for_rectangular_section(k, m, n, shift=0)`: Helper for section extraction

## Algorithm Details

### Hamiltonian Discretization

The continuous Schrödinger operator is discretized using the central finite difference scheme:

```
H = -1/(2dx²) [2I - D₊ - D₋ - P.B.C.]
```

where:
- D₊, D₋ are forward/backward difference operators
- Periodic boundary conditions couple corners: H[0,-1] = H[-1,0]
- Optional random perturbations maintain Hermiticity by symmetric application

### Spectral Computation

- **Hermitian matrices**: Uses `scipy.linalg.eigh()` (efficient O(n³) algorithm)
- **Non-Hermitian matrices**: Uses `scipy.linalg.eig()` (standard QR algorithm)
- **SVD**: Uses `scipy.linalg.svd()` with `full_matrices=False` for memory efficiency

### Visualization

KDE plots employ reflection boundary correction to eliminate boundary bias:

1. Reflect data points across domain boundaries
2. Compute KDE on extended domain
3. Clip visualization to original domain

This ensures accurate density estimation near spectrum edges.

## Research Context

This project is part of a master's thesis investigating:

1. **Spectral properties of quasicrystalline systems** following Lagarias' geometric models
2. **Computing pseudospectra of infinite-volume operators** (Hege et al., 2025)
3. **Localization phenomena** in disordered lattices (increased randomness → more localized eigenfunctions)
4. **Generalization of Theorem 22** (Hege et al., 2022) relating singular values to spectral distances

### Key Research Objectives

- Understand when local singular values accurately approximate spectral distances
- Study effect of Hamiltonian perturbations on eigenfunction localization
- Develop efficient methods for computing spectra of infinite-dimensional operators
- Apply techniques to analyze spectral gaps in quasicrystals

## Dependencies

The project requires Python 3.14+ with:

- **numpy**: Numerical array operations
- **scipy**: Linear algebra (eigendecomposition, SVD)
- **matplotlib**: Plotting backend
- **seaborn**: Statistical data visualization
- **pandas**: Data manipulation (optional, for extended analysis)
- **plotly**: Interactive 3D visualization support (optional, currently unused)

For complete dependency list, see `environment.yml`.

## Output

### Plot Files

Generated plots are saved in `plots/` subdirectories:

- **`kde_nonperturbed_L=XXX.png`**: KDE of eigenvalues for unperturbed Hamiltonian
- **`kde_perturbed_L=XXX.png`**: KDE of eigenvalues + singular values for perturbed Hamiltonian

Plots include:
- Upper panel: Eigenvalue density (normalized kernel density estimation)
- Lower panel (if applicable): Singular value density from all extracted sections

## Research Notes

See `meetings_with_Teufel.md` for detailed research progress, including:

- January 16, 2026: Initial investigations on Hamiltonian discretization
- January 27, 2026: KDE boundary correction and overlapping sections
- April 22, 2026: Theorem 22 analysis and generalization
- May 1, 2026: Eigenfunction localization and scientific document writing

## Mathematical Background

### References

The project is grounded in the following key works:

1. **Quasicrystal Geometry** (Lagarias, 1996-1999)
   - Delone sets and finite local complexity
   - Meyer's quasicrystal concept

2. **Spectral Theory** (Damanik, 2007)
   - Fractal dimension of Fibonacci Hamiltonian spectrum
   - Aperiodic order and spectral properties

3. **Numerical Methods** (Hege et al., 2022-2025)
   - Computing spectra and pseudospectra
   - Theorem 22: Relating singular values to spectral distances

All references are located in `docs/articles/` and `docs/theses/`.

## License

This project is part of academic research. Please refer to the main lattice_systems repository for licensing information.

## Author

Fernando Munoz  
Research focus: Spectral analysis of quasicrystalline systems and infinite-volume operators

## Contributing

For contributions or questions, please contact the repository maintainers or refer to the main [lattice_systems](https://github.com/legofernando1999/lattice_systems) repository.

---

**Last Updated**: May 26, 2026  
**Environment**: QMenv (Python 3.14, SciPy 1.16, NumPy 2.3, Matplotlib 3.10)
