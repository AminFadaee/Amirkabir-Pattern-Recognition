import numpy
import matplotlib.pyplot as plt
from numpy.random import normal
from math import e, sqrt, pi, log10


def gaussian_window(x, x_i, h):
    '''
    Implements the Gaussian density function

    Args:
        x: center of the distribution 
        x_i: desired point for obtaining the density
        h: parzen size parameter

    Returns:
        density at point x_i
    '''
    x = numpy.array(x)
    x_i = numpy.array(x_i)
    return (1 / sqrt(2 * pi)) * e ** (numpy.dot(-(x - x_i), x - x_i) / h)


H = [0.01, 0.1, 1, 10]
N = (200, 2000)
for n in N:
    samples1 = normal(20, 5, n // 2)
    samples2 = normal(35, 5, n // 2)
    samples = list(samples1) + list(samples2)
    plt.subplot(2, 1, (n == 2000) + 1)
    plt.scatter(samples, [0] * n, alpha=0.3, color='k')
    for h in H:
        X = numpy.linspace(0, 55, 56)
        Y = list(sum(gaussian_window(x, x_i, h) for x_i in samples) / (n * h) for x in X)
        plt.title('n = {0}'.format(n))
        plt.axis('off')
        plt.plot(X, Y / sum(Y), label='h = {0}'.format(h), linewidth=log10(h) + 3)

    if n == 200:
        plt.legend()

plt.show()
