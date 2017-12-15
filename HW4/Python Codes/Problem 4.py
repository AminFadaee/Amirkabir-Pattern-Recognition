import numpy
import matplotlib.pyplot as plt
from numpy.random import normal
from math import e, sqrt, pi


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


n = 100
samples = normal(0, 1, n)

H = [0.001, 0.01, 0.1, 1]

for i in range(4):
    for j in range(1, 5):
        h = round(j * H[i],4)
        X = numpy.linspace(-4, 4, 50)
        Y = list(sum(gaussian_window(x, x_i, h) for x_i in samples) / (n * h) for x in X)
        plt.subplot(4, 4, i * 4 + j)
        plt.title('h = {0}'.format(str(h)))
        plt.scatter(samples, [0] * n, alpha=0.3, color='k')
        plt.axis('off')
        plt.plot(X, Y)

plt.show()
