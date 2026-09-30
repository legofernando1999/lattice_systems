# Lattice Systems: Error-Controlled Spectra via Uneven Sections

Numerical experiments that test the **uneven sections method** of Hege, Moscolari & Teufel (2025), *Computing the Spectrum and Pseudospectrum of Infinite-Volume Operators from Local Patches*. The method gives rigorous two-sided bounds on the lower norm function $\rho_H(\lambda)$ of a discrete operator:

| Bound | Statement | Where it comes from |
|---|---|---|
| **Pseudospectral Inclusion Bound (PIB)**, upper | $\rho_H(\lambda) \le \varepsilon_{r,\lambda,x}$ | smallest singular value of the uneven section $Q_{r,\lambda,x}$ |
| **Spectral Gap Bound (SGB)**, lower | $\rho_H(\lambda) \ge \varepsilon_{r,\lambda} - C/r$ | infimum of $\varepsilon_{r,\lambda,x}$ over all centres $x$ |

For a normal (e.g. Hermitian) operator, $\rho_H(\lambda) = d(\lambda, \sigma(H))$, so both bounds can be checked directly against the exact eigenvalues.

The test case is the **free one-dimensional time-independent Schrödinger equation with periodic boundary conditions**, discretised by finite differences. Random diagonal perturbations (a random potential) are also supported.

This repository is the code and write-up for a **Scientific Project** (M.Sc., Mathematics Department, University of Tübingen, Summer Semester 2026), supervised by Prof. Dr. Stefan Teufel. The work is expected to continue into a Master's thesis.

---

## Table of Contents

1. [Mathematical Background](#mathematical-background)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Code Reference](#code-reference)
6. [Output Files](#output-files)
7. [Results](#results)
8. [Building the Scientific Project Report](#building-the-scientific-project-report)
9. [Known Limitations and Caveats](#known-limitations-and-caveats)
10. [Future Work](#future-work)
11. [References](#references)

---

## Mathematical Background

### The model

The equation is restricted to $[0, L]$ with $\psi(0) = \psi(L)$:

$$
-\tfrac{1}{2}\,\psi''(x) = E\,\psi(x), \qquad 0 \le x \le L .
$$

On a uniform mesh $x_i = i\,\Delta x$ with $N = L/\Delta x + 1$ points, the centred second difference gives the periodic tridiagonal matrix

$$
H_N = -\frac{1}{2\Delta x^2}
\begin{pmatrix}
-2 & 1 & & & 1\\
1 & -2 & 1 & & \\
 & \ddots & \ddots & \ddots & \\
 & & 1 & -2 & 1\\
1 & & & 1 & -2
\end{pmatrix}.
$$

When $\Delta x = 1$, the diagonal is $1$, the off-diagonals are $-\tfrac12$, and the spectrum lies in $[0, 2]$. In the paper's language:

- The uniformly discrete set is $\Gamma = \mathbb{Z}$, with packing radius $q = \Delta x = 1$.
- The operator has finite range, with maximal hopping length $m = 1$.
- The space dimension is $n = 1$.

### Uneven sections

For a centre $x$, a window radius $r$ and a spectral parameter $\lambda$, the uneven section is

$$
Q_{r,\lambda,x} = \mathbf{1}_{B_{r+m}(x)}\,(H - \lambda)\big|_{\mathcal{H}_{B_r(x)}},
$$

This is the rectangular block of $H - \lambda I$ with $2(r+m)$ rows and $2r$ columns, centred on the diagonal at $x$. Its smallest singular value $\varepsilon_{r,\lambda,x}$ is the PIB.

### Spectral Gap Bound constant

$$
C = m\,M\left(\frac{36\,m}{q}\right)^{n/2}, \qquad M = \sup_{x,y}|H_{xy}| .
$$

With periodic boundary conditions and no perturbation, $H_N$ is translation invariant. So $\varepsilon_{r,\lambda,x}$ does not depend on $x$, and one centre is enough to get the infimum $\varepsilon_{r,\lambda}$.

The full derivation is in [scientific_project/scientific_project.pdf](scientific_project/scientific_project.pdf).

---

## Repository Structure

```
lattice_systems/
├── hamiltonian.py               # Hamiltonian class: construction, eigen-solver, JSON (de)serialisation
├── main_script.py               # Uneven sections, SVD, bounds, plotting, and the three experiment drivers
├── environment.yml              # Conda environment "QMenv" (Python 3.14, NumPy, SciPy, Matplotlib, seaborn)
├── meetings_with_supervisor.md  # Dated log of supervisor meetings and action items
├── .gitignore                   # Ignores plots/, hamiltonians/, __pycache__/
├── scientific_project/          # LaTeX write-up of the project
│   ├── scientific_project.tex   # Main document
│   ├── scientific_project.pdf   # Compiled report
│   ├── sample.bib               # Bibliography (biblatex)
│   └── images/                  # Figures used in the report
└── docs/                        # Reference material (not code)
    ├── Questions.docx           # Reading notes / summary of Hege et al. (2025)
    ├── articles/                # Papers on quasicrystals, Delone sets, (pseudo)spectra
    └── theses/                  # Other theses, kept for reference
```

Two more directories are created automatically at runtime. Both are git-ignored:

- `plots/` is where KDE figures are saved, in the subfolders `free_Hamiltonian/` and `H_lambda/`.
- `hamiltonians/` is where saved Hamiltonians are stored as JSON.

---

## Installation

### Prerequisites

- Anaconda or Miniconda
- A LaTeX distribution with `latexmk` and `biber`, needed only to rebuild the report

### Setup

```bash
git clone https://github.com/legofernando1999/lattice_systems.git
cd lattice_systems

conda env create -f environment.yml
conda activate QMenv
```

`environment.yml` is a full export with Linux build strings and a machine-specific `prefix:`. If it doesn't resolve on your platform, a minimal environment is enough:

```bash
conda create -n QMenv python=3.14 numpy scipy matplotlib seaborn
```

The code imports only `numpy`, `scipy.linalg`, `matplotlib`, `seaborn`, and the standard library modules `json` and `pathlib`.

---

## Usage

All experiments live in [main_script.py](main_script.py). To choose one, uncomment the call you want in the `if __name__ == '__main__':` block at the bottom of the file, then run:

```bash
python main_script.py
```

By default, only `lower_norm_fct_bounds()` is enabled.

Each experiment's parameters (`L`, `dx`, `r`, `λ`, perturbation, etc.) are hard-coded at the top of its driver function. To change them, edit them there.

**Reproducing a perturbed run.** When the Hamiltonian is perturbed, the drivers print `Perturbation seed: <number>`. To get exactly the same random potential again, set `seed = <number>` at the top of the driver. With `seed = None`, every run draws a new perturbation.

### 1. `lower_norm_fct_bounds()`: PIB and SGB table

This driver computes the upper and lower bounds on $\rho_H(\lambda)$ for the unperturbed free Hamiltonian ($L = 1000$, $\Delta x = 1$, centre $x = 501$). It loops over:

- $r \in \{50, 100, 150, 200, 250\}$
- $\lambda \in \{-0.1, 0.5, 1.2, 1.7, 2.3\}$

Output goes to stdout as LaTeX table rows, ready to paste into the report:

```
$(50, -0.1)$ & 0.100000 & 0.100507 & -0.031493 \\
...
\hline
```

The columns are $(r, \lambda)$, $d(\lambda, \sigma(H))$, PIB, and SGB.

### 2. `free_hamiltonian_lambda()`: eigenvectors near λ versus singular values

For a fixed $\lambda$ (default $-0.5$) and window size $r$ (default $150$), this driver:

1. Finds the 5 eigenvalues of $H$ closest to $\lambda$.
2. Plots $|v_j|^2$ for their eigenvectors in an interactive window (`plt.show()`). The uneven-section window is highlighted.
3. Centres the uneven section on the peak of the closest eigenvector's density. This happens only when the Hamiltonian is perturbed. Otherwise the section sits at the middle of the domain.
4. Prints $d(\lambda, \sigma(H))$ next to the smallest singular values of $Q_{r,\lambda,x}$.

Set `perturb_H = True` to study localisation. Stronger randomness gives more localised eigenvectors.

Set `save_hamiltonian = True` to write the Hamiltonian to `hamiltonians/<hamiltonian_filename>`. Use this to keep perturbed Hamiltonians that make good case studies. To reload one later, use the commented-out `Hamiltonian.from_json(...)` lines.

### 3. `free_hamiltonian()`: spectrum and singular-value densities

This driver builds a perturbed free Hamiltonian ($L = 1000$, perturbations in $[-0.1, 0.1]$ on the diagonal). It then tiles the diagonal with uneven sections spaced `d = 5` apart ($r = 50$, $m = 1$) and computes all of their singular values.

The result is a two-panel KDE figure, saved to `plots/free_Hamiltonian/kde_perturbed_L=1000.png`:

- top panel: the density of the eigenvalues
- bottom panel: the pooled density of the singular values

### Using the `Hamiltonian` class directly

```python
from pathlib import Path
import numpy as np
from hamiltonian import Hamiltonian

# Free Hamiltonian with a random diagonal potential in [-0.2, 0.2]
ham = Hamiltonian.construct_free_hamiltonian(L=1000, dx=1.0, perturb_H=True, random_rng=(-0.2, 0.2))
ham.eigenvalues            # sorted real eigenvalues (eigh)
ham.eigenvectors           # columns are normalised eigenvectors

# Any matrix works; Hermiticity is detected automatically if not given
shifted = Hamiltonian(ham.matrix - 0.5 * np.eye(ham.shape[0]), eigvals_only=True)

# Round-trip through JSON
ham.to_json(Path('hamiltonians/example.json'))
same = Hamiltonian.from_json(Path('hamiltonians/example.json'))
```

---

## Code Reference

### [hamiltonian.py](hamiltonian.py): `class Hamiltonian`

| Member | Description |
|---|---|
| `Hamiltonian(matrix, is_hermitian=None, eigvals_only=False)` | Wraps a square matrix and solves the eigenproblem immediately. If `is_hermitian` is `None`, it is found with `np.allclose(H, H^*)`. Attributes: `matrix`, `shape`, `is_hermitian`, `eigenvalues`, `eigenvectors` (`None` if `eigvals_only`). |
| `construct_free_hamiltonian(L, dx, perturb_H=False, random_rng=(-0.1, 0.1), eigvals_only=False, seed=None)` | *Classmethod.* Builds the periodic finite-difference matrix $H_N$ with $N = L/dx + 1$. With `perturb_H=True`, it adds i.i.d. uniform noise from `random_rng` to the **main diagonal only**, which models a random potential and keeps $H$ Hermitian. Pass an int `seed` for a fixed perturbation; with `seed=None` a fresh seed is drawn. Either way, the seed used is stored in the returned instance's `seed` attribute (`None` when unperturbed). |
| `_solve_eigenvalue_problem(eigvals_only)` | Hermitian matrices use `scipy.linalg.eigh` / `eigvalsh`. Other matrices use `eig` / `eigvals`, followed by `np.real_if_close`. Raises `RuntimeError` if the solver does not converge. |
| `to_json(file_path=None)` | Returns a JSON string, or writes it to `file_path`. A real, symmetric, periodic tridiagonal matrix is stored compactly as `diagonal`, `subdiagonal` and the `lower-left corner`. Any other matrix is stored in full as `H`, so nothing is lost. Raises `TypeError` for complex matrices. |
| `from_json(json_data, eigvals_only=False)` | *Classmethod.* Rebuilds the matrix from a path-like object such as `pathlib.Path` (a file), a `str` (always parsed as JSON text, never as a file path), or a `dict`. Raises `ValueError` for a missing file or invalid JSON, and `TypeError` for any other input type. |

### [main_script.py](main_script.py)

**Uneven sections**

| Function | Description |
|---|---|
| `middle_value(k)` | Middle index convention: `k/2` if `k` is even, `(k+1)/2` if it is odd. |
| `extract_uneven_section(matrix, nrows, ncols, shift=0)` | Returns the `nrows × ncols` block centred on the main diagonal, shifted by `shift` along it. Raises `ValueError` if the block falls outside the matrix. |
| `select_uneven_sections(matrix_shape, nrows, ncols, d)` | Builds specs `{'section j': {nrows, ncols, shift=j*d}}` for every section that fits, walking out from the centre in both directions. |
| `svd_uneven_sections(H, sections_specs, singular_vals_only=False)` | Extracts each section and runs `scipy.linalg.svd` with `full_matrices=False`. Returns `{'A', 'S'}`, plus `'U'` and `'V'` when `singular_vals_only` is `False`. `'V'` is the matrix of right singular vectors, not `V^H`. |
| `_determine_indices_for_uneven_section(N, nrows, ncols, shift)` | Helper that computes the row and column slice bounds. |

**Bounds**

| Function | Description |
|---|---|
| `dist_lambda_spec_H(lmbd, spectrum)` | Returns $\lvert\lambda - E_i\rvert$ for every eigenvalue. Its minimum is $d(\lambda, \sigma(H))$. |
| `spectral_gap_bound(epsilon_r_lmbd, r, H, m, q, n=1)` | Returns $\varepsilon_{r,\lambda} - C/r$, where $C = mM(36m/q)^{n/2}$ and $M = \max\lvert H_{xy}\rvert$ of the matrix passed in. |
| `choose_x(perturb_H, y_axis, N, r)` | Picks the section centre. For a perturbed $H$ it uses the argmax of `y_axis`, clamped so that the window fits. Otherwise it uses the middle of the domain. |
| `is_hermitian(a, tol)` | Stand-alone Hermiticity check. |

**Plotting**

| Function | Description |
|---|---|
| `generate_plot(L, H_perturbed, H_eigenvalues, H_sections, plots_subfolder)` | Collects eigenvalues and singular values and saves `plots/<subfolder>/kde_{perturbed,nonperturbed}_L=<L>.png`. |
| `_create_figure(hist_data, fname)` | Draws a seaborn KDE with a colour-blind palette at 800 dpi. It uses one panel, or two when singular values are present. |
| `_mirror_array(arr)` | Reflects the data across its minimum and maximum before the KDE is computed, then clips to the original range. This corrects boundary bias. |

**Experiment drivers:** `lower_norm_fct_bounds()`, `free_hamiltonian_lambda()`, `free_hamiltonian()`. See [Usage](#usage).

---

## Output Files

| Path | Produced by | Contents |
|---|---|---|
| `plots/free_Hamiltonian/kde_{perturbed,nonperturbed}_L=<L>.png` | `free_hamiltonian()` | KDE of the eigenvalues, plus the pooled singular values of the uneven sections |
| `plots/H_lambda/...` | `free_hamiltonian_lambda()` (the `generate_plot` call is currently commented out) | Same layout, for $H - \lambda$ |
| `hamiltonians/<name>.json` | `free_hamiltonian_lambda()` with `save_hamiltonian = True`, or `Hamiltonian.to_json` | Serialised Hamiltonian |
| stdout | `lower_norm_fct_bounds()`, `free_hamiltonian_lambda()` | Bound tables and distance / singular-value comparisons |

Example JSON for a Hermitian Hamiltonian:

```json
{
  "shape": [1001, 1001],
  "is_Hermitian": true,
  "diagonal": [1.0, 1.0, "..."],
  "subdiagonal": [-0.5, -0.5, "..."],
  "lower-left corner": -0.5
}
```

---

## Results

The table below was produced by `lower_norm_fct_bounds()` for the unperturbed free Hamiltonian ($L = 1000$, $\Delta x = 1$). It is reproduced from the report. Selected rows:

| $(r, \lambda)$ | $d(\lambda, \sigma(H))$ | PIB (upper) | SGB (lower) |
|---|---|---|---|
| (50, −0.1) | 0.100000 | 0.100507 | −0.031493 |
| (250, −0.1) | 0.100000 | 0.100020 | 0.073620 |
| (50, 0.5) | 0.000906 | 0.026657 | −0.033343 |
| (250, 0.5) | 0.000906 | 0.005409 | −0.006591 |
| (50, 2.3) | 0.300005 | 0.300494 | 0.144494 |
| (250, 2.3) | 0.300005 | 0.300020 | 0.268820 |

What the results show:

- **Both bounds hold in every case:** SGB ≤ $d(\lambda,\sigma(H))$ ≤ PIB.
- **Both bounds tighten as $r$ grows.** The PIB decreases and the SGB increases, because a larger window captures more of the support of the relevant eigenvector.
- **For $\lambda$ outside the spectral range $[0, 2]$, the PIB is much sharper.** This is likely because edge states are more strongly localised.

The full table and a discussion are in the report.

---

## Building the Scientific Project Report

```bash
cd scientific_project
latexmk -pdf scientific_project.tex    # runs pdflatex + biber as needed
```

Or compile by hand:

```bash
pdflatex scientific_project && biber scientific_project && pdflatex scientific_project && pdflatex scientific_project
```

Figures are read from `scientific_project/images/`. To regenerate `kde_nonperturbed_L=1000.png`:

1. Set `perturb_H = False` in `free_hamiltonian()` and run it.
2. Copy the file from `plots/free_Hamiltonian/` into `images/`.

---

## Known Limitations and Caveats

- **Dense linear algebra.** Matrices are stored as dense `ndarray`s and fully diagonalised. Memory grows as $O(N^2)$ and time as $O(N^3)$. $N \approx 10^3$ is fast, but much larger systems would need sparse or banded solvers such as `scipy.linalg.eig_banded` or `scipy.sparse.linalg`.
- **Complex Hamiltonians cannot be saved to JSON.** `to_json` raises `TypeError` for them, because JSON has no complex number type.
- **The SGB constant $M$ is taken from the matrix passed in.** In `lower_norm_fct_bounds()`, that matrix is $H - \lambda I$, not $H$. The report's table was generated this way.
- **The eigenvalue solver runs on construction.** Building `Hamiltonian(H - λI)` just to take its matrix still diagonalises it. Pass `eigvals_only=True`, as the drivers do, to limit the cost.
- Experiments are configured by editing source code. There is no CLI and no config file.

---

## Future Work

Taken from the report's conclusion and [meetings_with_supervisor.md](meetings_with_supervisor.md):

- Test the PIB and SGB on non-trivial operators: random (Anderson-type) Hamiltonians, and Hamiltonians from quasicrystal systems (Fibonacci, jump potentials, Penrose tilings).
- For operators without translation invariance, compute $\varepsilon_{r,\lambda} = \inf_x \varepsilon_{r,\lambda,x}$ over a finite set of centres, using finite local complexity.
- Handle the overcounting of singular values where uneven sections overlap.
- Optionally, build a GUI for moving the uneven-section window interactively.
- Long term: software that can rigorously prove spectral gaps of infinite-volume one-body operators. This is the planned direction for the Master's thesis.

---

## References

The main references are cited in [scientific_project/sample.bib](scientific_project/sample.bib). PDFs of most of the articles are in [docs/articles/](docs/articles/).

1. P. Hege, M. Moscolari, S. Teufel. *Computing the Spectrum and Pseudospectrum of Infinite-Volume Operators from Local Patches*, 2025.
2. P. Hege et al. *Finding Spectral Gaps in Quasicrystals*, 2022.
3. M. J. Colbrook, B. Roman, A. C. Hansen. *How to Compute Spectra with Error Control*, 2019.
4. L. N. Trefethen, M. Embree. *Spectra and Pseudospectra: The Behavior of Nonnormal Matrices and Operators*.
5. H. J. Landau. *The Notion of Approximate Eigenvalues Applied to an Integral Equation of Laser Theory*, 1977.
6. S. Beckus et al. *On the Spectrum of Operator Families on Discrete Groups over Minimal Dynamical Systems*, 2017.
7. J. C. Lagarias. *Meyer's Concept of Quasicrystal and Quasiregular Sets* (1996); *Geometric Models for Quasicrystals I & II* (1999).
8. A. Besbes, M. Boshernitzan, D. Lenz. *Delone Sets with Finite Local Complexity*, 2013.
9. D. Damanik et al. *The Fractal Dimension of the Spectrum of the Fibonacci Hamiltonian*, 2007.
10. E. Cancès et al. *Numerical Computation of the Density of States*, 2025.
11. G. W. Stewart, *Matrix Algorithms Vol. I*; G. Strang, *Linear Algebra and Its Applications*; J. Dugundji, *Topology*; F. W. Byron & R. W. Fuller, *Mathematics of Classical and Quantum Physics*.

---

## Author

**Fernando Muñoz Martínez**, University of Tübingen
Supervisor: Prof. Dr. Stefan Teufel
