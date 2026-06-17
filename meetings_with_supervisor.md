# Meetings with Stefan Teufel

## 17/06/2026
+ FINISH THE SCIENTIFIC PROJECT. You need to start working on your Master's Thesis ASAP.
+ Save exactly those perturbed free Hamiltonians which give a good case study for what we are trying to do. This means, if H is such that the smallest singular values (first 2 or so) of the uneven section are approximately equal (>=) to the distances d(λ, σ(H)) of the closest eigenvalues to λ, then save it. This should happen when the eigenvalue corresponds to an eigenvector whose support lies completely within the window of the uneven section. Try choosing a λ that is outside the range of eigenvalues. That way you will detect the eigenvectors corresponding to the eigenvalues lying at the boundary, and these are highly localized. 
+ Play with different window sizes, e.g. 50, 100, 150, to see if this has an effect on the singular values.
+ Optional but nice to have: Make a graphic user interface (GUI) that allows the user to move the window of the uneven section around.

## 11/06/2026
+ Plot the eigenvectors corresponding to the closest ~5 eigenvalues to lambda (use different colors), and check if the second singular value of the uneven section "contains" the eigenvalues corresponding to the eigenvectors whose support lies completely within the uneven section window.
+ Complete and submit soon the scientific project.
+ In order to test the spectral gap bound you could center an uneven section at each point x of the domain and then look at their singular values.

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