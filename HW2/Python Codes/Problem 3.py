import numpy
import matplotlib.pyplot as plt
from numpy.linalg import inv as inverse
from numpy.linalg import det
from math import log

w1 = numpy.array([[0, 0],
                  [0, 1],
                  [2, 2],
                  [3, 1],
                  [3, 2],
                  [3, 3]])

w2 = numpy.array([[6, 9],
                  [8, 9],
                  [9, 8],
                  [9, 9],
                  [9, 10],
                  [8, 11]])

mu1 = numpy.matrix(numpy.mean(w1, axis=0)).T
mu2 = numpy.matrix(numpy.mean(w2, axis=0)).T
cov1 = numpy.cov(w1.T)
cov2 = numpy.cov(w2.T)
print(mu1)
print(mu2)
print(cov1)
print(cov2)
###################################################
Wi = (-1 / 2) * inverse(cov1)
wi = inverse(cov1) * mu1
wi0 = (-1 / 2) * mu1.T * inverse(cov1) * mu1 - (1 / 2) * log(det(cov1)) + log(1 / 2)
print(Wi)
print(wi)
print(wi0)
###################################################
Wi = (-1 / 2) * inverse(cov2)
wi = inverse(cov2) * mu2
wi0 = (-1 / 2) * mu2.T * inverse(cov2) * mu2 - (1 / 2) * log(det(cov2)) + log(1 / 2)
print(Wi)
print(wi)
print(wi0)
###################################################
plt.scatter(w1[:, 0], w1[:, 1])
plt.scatter(w2[:, 0], w2[:, 1])
x = numpy.linspace(-15, 35, 100)
y = numpy.linspace(-15, 35, 100)
x, y = numpy.meshgrid(x, y)
plt.contour(x, y, (x * (59.9853 + y * (-9.65439)) + x ** 2 + 4.4487 * y ** 2 + 79.5307 * y - 667.274), [1])
plt.show()
