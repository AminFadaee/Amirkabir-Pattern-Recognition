import numpy
from numpy.linalg import inv as inverse, det
from numpy import log as ln
from math import pi


def generate_random_sample(mu, cov, n):
    mu = numpy.array(mu)
    cov = numpy.matrix(cov)
    return numpy.random.multivariate_normal(mean=mu, cov=cov, size=n)


def g(x, mu, cov, P):
    x = numpy.matrix(x).T  # column vector
    mu = numpy.matrix(mu).T  # column vector
    cov = numpy.matrix(cov)
    d = x.shape[0]
    return float((-1 / 2) * (x - mu).T * inverse(cov) * (x - mu)- (d/2)*ln(2*pi) - (1 / 2) * ln(det(cov)) + ln(P))


def euclidean_distance(X, Y):
    X = numpy.array(X)
    Y = numpy.array(Y)
    return numpy.sqrt(numpy.sum((X - Y) * (X - Y).T))


def mahalanobis_distance(X, mu, cov):
    X = numpy.matrix(X).T  # column vector
    mu = numpy.matrix(mu).T  # column vector
    cov = numpy.matrix(cov)
    return float(numpy.sqrt((X - mu).T * inverse(cov) * (X - mu)))
