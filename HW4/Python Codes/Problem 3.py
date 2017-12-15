import numpy
from math import e


def spherical_gaussian(x, x_i, h):
    '''
    Implements the Spherical Gaussian density function
    
    Args:
        x: center of the distribution 
        x_i: desired point for obtaining the density
        h: parzen size parameter

    Returns:
        density at point x_i
    '''
    x = numpy.array(x)
    x_i = numpy.array(x_i)
    return e ** (numpy.dot(-(x - x_i), x - x_i) / (2 * h ** 2))


def p_x(D, x, h):
    '''
    Finds the conditional probability based on D at point x
    Args:
        D: data points
        x: arbitrary point
        h: parzen size parameter

    Returns:
        p(x|w_D)
    '''
    n = len(D)
    d = len(D[0])
    phi = sum(list(spherical_gaussian(x, point, h) for point in D))
    return phi / (n * (h ** d))


w1 = [[0.28, 1.31, -6.2],
      [0.07, 0.58, -0.78],
      [1.54, 2.01, -1.63],
      [-0.44, 1.18, -4.32],
      [-0.81, 0.21, 5.73],
      [1.52, 3.16, 2.77],
      [2.20, 2.42, -0.19],
      [0.91, 1.94, 6.21],
      [0.65, 1.93, 4.38],
      [-0.26, 0.82, -0.96]]

w2 = [[0.011, 1.03, -0.21],
      [1.27, 1.28, 0.08],
      [0.13, 3.12, 0.16],
      [-0.21, 1.23, -0.11],
      [-2.18, 1.39, -0.19],
      [0.34, 1.96, -0.16],
      [-1.38, 0.94, 0.45],
      [-0.12, 0.82, 0.17],
      [-1.44, 2.31, 0.14],
      [0.26, 1.94, 0.08]]

w3 = [[1.36, 2.17, 0.14],
      [1.41, 1.45, -0.38],
      [1.22, 0.99, 0.69],
      [2.46, 2.19, 1.31],
      [0.68, 0.79, 0.87],
      [2.51, 3.22, 1.35],
      [0.60, 2.44, 0.92],
      [0.64, 0.13, 0.97],
      [0.85, 0.58, 0.99],
      [0.66, 0.51, 0.88]]

W = [w1, w2, w3]

for h in (1, 0.1):
    for x in ([0.5, 1.0, 0.0], [0.31, 1.51, -0.5], [-0.3, 0.44, -0.1]):
        posteriors = list(p_x(W[i], x, h) for i in range(3))
        print('Class of {0} is {1}'.format(tuple(x), numpy.argmax(posteriors) + 1))
