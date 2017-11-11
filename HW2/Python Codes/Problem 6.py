import numpy
import matplotlib.pyplot as plt
from numpy.linalg import inv as inverse
from numpy.linalg import det
from math import log

w1 = numpy.array([[2, 3],
                  [0.5, 2],
                  [1.5, 2],
                  [2.5, 2],
                  [1, 1],
                  [2, 1],
                  [1.5, 0]])

w2 = numpy.array([[-2, -1],
                  [-1, -1],
                  [1, -1],
                  [0.5, -0.5],
                  [1.5, -0.5],
                  [-1.5, 0],
                  [-0.5,0.5],
                  [0.5 ,0.5],
                  [1.5 ,0.5],
                  [-1.5, 1]])


mu1 = numpy.matrix(numpy.mean(w1, axis=0)).T
mu2 = numpy.matrix(numpy.mean(w2, axis=0)).T
cov1 = numpy.cov(w1.T)
cov2 = numpy.cov(w2.T)
p1 = 7/17
p2 = 10/17
# print(mu1)
# print(mu2)
# print(cov1)
# print(cov2)
###################################################
Wi = (-1 / 2) * inverse(cov1)
wi = inverse(cov1) * mu1
wi0 = (-1 / 2) * mu1.T * inverse(cov1) * mu1 - (1 / 2) * log(det(cov1)) + log(p1)
print(Wi)
print(wi)
print(wi0)
###################################################
Wi = (-1 / 2) * inverse(cov2)
wi = inverse(cov2) * mu2
wi0 = (-1 / 2) * mu2.T * inverse(cov2) * mu2 - (1 / 2) * log(det(cov2)) + log(p2)
print(Wi)
print(wi)
print(wi0)
###################################################
plt.scatter(w1[:, 0], w1[:, 1],color='red')
plt.scatter(w2[:, 0], w2[:, 1],color='k')
x = numpy.linspace(-2.5, 4, 100)
y = numpy.linspace(-7, 7, 100)
x, y = numpy.meshgrid(x, y)
plt.contour(x, y, (x**2 + x*(-0.331536*y - 3.78581) - 0.413482*y**2 - 1.78816*y +3.94877), [1])
plt.show()