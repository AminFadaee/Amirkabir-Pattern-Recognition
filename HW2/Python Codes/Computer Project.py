import numpy
import matplotlib.pyplot as plt
from numpy.linalg import inv as inverse, det
from math import log, sqrt, e
from numpy.random import multivariate_normal


def generate_sample_points(mu, cov, n, label):
    mu = numpy.squeeze(numpy.asarray(mu))
    samples = multivariate_normal(mu, cov, size=n)
    return list(zip(samples, [label] * len(samples)))


def k(beta, mu1, mu2, cov1, cov2):
    return float(
        beta * (1 - beta) / 2 * (mu2 - mu1).T * inverse(beta * cov1 + (1 - beta) * cov2) * (mu2 - mu1) + 1 / 2 * log(
            det(beta * cov1 + (1 - beta) * cov2) / (det(cov1) ** beta * det(cov2) ** (1 - beta))))


def chernoff_bound(mu1, mu2, cov1, cov2, p1, p2):
    x = numpy.linspace(0, 1, 1001)
    y = list(map(lambda beta: e ** (-k(beta, mu1, mu2, cov1, cov2)), x))
    beta = x[numpy.argmin(y)]
    return (p1 ** beta) * (p2 ** (1 - beta)) * numpy.min(y)


def bhattacharyya_bound(mu1, mu2, cov1, cov2, P1, P2):
    k = float((1 / 8) * (mu2 - mu1).T * inverse((cov1 + cov2) / 2) * (mu2 - mu1) + (1 / 2) * log(
        det((cov1 + cov2) / 2) / sqrt(det(cov1) * det(cov2))))
    return round(sqrt(P1 * P2) * e ** (-k), 4)


mu1 = numpy.matrix([-1, 0]).T
mu2 = numpy.matrix([1, 0]).T
cov1 = cov2 = numpy.eye(2)
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

print("x1 = 0")


def generate_empirical_data(mu1, mu2, cov1, cov2, n):
    samples = generate_sample_points(mu1, cov1, n // 2, 1) + generate_sample_points(mu2, cov2, n // 2, 2)
    correct = 0
    for instance in samples:
        point, label = instance
        prediction = 1 if point[0] < 0 else 2
        correct += prediction == label

    return 1 - (correct / n), samples


err, samples = generate_empirical_data(mu1, mu2, cov1, cov2, 100)
print("Error for 100 Samples:", err)
for instance in samples:
    point, label = instance
    plt.scatter(point[0], point[1], color=['blue', 'orange'][label == 2])
plt.plot([0, 0], [-4, 4], linestyle=':', linewidth=3)
plt.fill_between([-4, 0], [4, 4], color='blue', alpha=0.3)
plt.fill_between([-4, 0], [-4, -4], color='blue', alpha=0.3)
plt.fill_between([0, 4], [4, 4], color='orange', alpha=0.3)
plt.fill_between([0, 4], [-4, -4], color='orange', alpha=0.3)
plt.title('Error for 100 Samples:' + str(round(err, 3)))
plt.show()

sizes = numpy.linspace(100, 1000, 10)
errors = []
for n in sizes:
    err, _ = generate_empirical_data(mu1, mu2, cov1, cov2, int(n))
    errors.append(err)

plt.plot(sizes, errors, label='Errors')
plt.xlabel('Sizes')
plt.ylabel('Errors')
plt.title('Error Rate for Different Sizes')
bhat = bhattacharyya_bound(mu1, mu2, cov1, cov2, 1 / 2, 1 / 2)
cher = chernoff_bound(mu1, mu2, cov1, cov2, 0.5, 0.5)
plt.plot([100, 1000], [cher] * 2, label='Chernoff Bound = {0}'.format(round(cher,4)), linewidth=3)
plt.plot([100, 1000], [bhat] * 2, label='Bhattacharyya Bound = {0}'.format(round(bhat,4)), linewidth=3, linestyle=':',color='k')
plt.legend()
plt.show()
