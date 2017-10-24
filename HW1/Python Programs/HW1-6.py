from numpy.linalg import eig as eig

A = [[4, 0, 0],
     [0, 2, 2],
     [0, 9, -5]]

Lambda, Phi = eig(A)
print("Eigenvalues:\n", Lambda)
print("Eigenvectors:\n", Phi)

B = [[2, 3],
     [1, 5]]

Lambda, Phi = eig(B)
print("Eigenvalues:\n", Lambda)
print("Eigenvectors:\n", Phi)
