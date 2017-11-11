import numpy
from numpy.linalg import inv as inverse, det
from math import log, sqrt, e
import matplotlib.pyplot as plt

mu1 = numpy.matrix([0, 0]).T
mu2 = numpy.matrix([1, 1]).T
cov1 = numpy.eye(2)
cov2 = numpy.eye(2)
p1 = 1 / 2
p2 = 1 / 2

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

def bhattacharyya_bound(mu1, mu2, cov1, cov2, P1, P2):
    k = float((1 / 8) * (mu2 - mu1).T * inverse((cov1 + cov2) / 2) * (mu2 - mu1) + (1 / 2) * log(
        det((cov1 + cov2) / 2) / sqrt(det(cov1) * det(cov2))))
    return round(sqrt(P1 * P2) * e ** (-k), 4)


print("Bhattacharyya Bound=", bhattacharyya_bound(mu1, mu2, cov1, cov2, p1, p2))


###################################################
def k(beta, mu1, mu2, cov1, cov2):
    return float(
        beta * (1 - beta) / 2 * (mu2 - mu1).T * inverse(beta * cov1 + (1 - beta) * cov2) * (mu2 - mu1) + 1 / 2 * log(
            det(beta * cov1 + (1 - beta) * cov2) / (det(cov1) ** beta * det(cov2) ** (1 - beta))))


def chernoff_bound(mu1, mu2, cov1, cov2,p1,p2):
    x = numpy.linspace(0, 1, 1001)
    y = list(map(lambda beta: e ** (-k(beta, mu1, mu2, cov1, cov2)), x))
    plt.plot(x, y, label='e^{-k(beta)}')
    plt.plot([0, 1], [numpy.min(y)] * 2, label='Best')
    plt.plot([x[numpy.argmin(y)]] * 2, [0.70, numpy.min(y)], linestyle=':', label=x[numpy.argmin(y)])
    plt.legend()
    plt.show()
    beta = x[numpy.argmin(y)]
    return (p1**beta)*(p2**(1-beta))*numpy.min(y)


print("Bhernoff Bound=",chernoff_bound(mu1, mu2, cov1, cov2,0.5,0.5))

#################################################################################################################

mu1 = numpy.matrix([0, 0]).T
mu2 = numpy.matrix([1, 1]).T
cov1 = numpy.matrix([[2, 0.5], [0.5, 2]])
cov2 = numpy.matrix([[5, 2], [2, 5]])
p1 = 1 / 2
p2 = 1 / 2

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


print("Bhattacharyya Bound=", bhattacharyya_bound(mu1, mu2, cov1, cov2, p1, p2))

###################################################


print("Bhernoff Bound=",chernoff_bound(mu1, mu2, cov1, cov2,0.5,0.5))
