# Meetings with Stefan Teufel

## 01/05/2026
+ Plot some eigenfunctions to try and see where their support is, and whether moving the uneven sections to this region gives you a singular value that is closer to the distance between your lambda and the spectrum of H.
+ Fact: Increasing randomness in the Hamiltonian makes eigenfunctions more localized.
+ Begin writing a document (scientific project) describing what the problem is, what I have done and what I would do next in the master's thesis. Be precise, use mathematics, make it approx. 10 pages long, include references, plots, code, etc.

## 22/04/2026
+ Study Theorem 22 in Paul Hege's paper (Computing the spectrum and pseudospectrum...) and run simulations to understand whether it is possible to generalize it.
+ Think of ways to deal with the overcounting of eigenfunctions in the regions where the uneven sections overlap.
+ Reduce overlapping of uneven sections.
+ Compute singular values of uneven sections of (H - lambda). Where lambda is any number in the range of eigenvalues of H. More info in Paul Hege's paper.

## 27/01/2026
+ Fix KDE plots boundary issue.
+ Take overlapping rectangular sections covering the whole domain, and plot their singular values on the same subplot accounting for the overcounting of observations in the overlaps.

## 16/01/2026
+ Reduce size of average windows of histogram curves.
+ Verify computation of eigenvalues of Hamiltonian.
+ Add small random values to entries of Hamiltonian to make it non-periodic.
+ Reduce size of rectangular matrix -- to 50 by 51 for example -- and move it along the diagonal of the original matrix.