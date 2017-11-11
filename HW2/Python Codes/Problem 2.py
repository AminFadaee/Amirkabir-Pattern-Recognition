import numpy
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as normal

mu1 = [4, 16]
mu2 = [16, 4]
cov1 = cov2 = 4 * numpy.eye(2)
P1 = 0.6
P2 = 0.4

w1 = normal(mu1, cov1, size=100)
w2 = normal(mu2, cov2, size=100)
decision = lambda x: x - 0.1351

plt.scatter(w1[:, 0], w1[:, 1],label='Salmon (w1)')
plt.scatter(w2[:, 0], w2[:, 1],label='Sea Bass (w2)')
plt.plot(numpy.array([0, 20]), decision(numpy.array([0, 20])),label='Decision Boundary')
plt.legend()
plt.show()
