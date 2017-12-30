import numpy
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.style.use('ggplot')

X = numpy.array([[12.1, 6.6, 11.5, 8.7, 15.4, 6.9, 8.8, 10.1, 19.1, 15.6, 22.2, 10.2, 15.7, 10.0, 7.2, 11.8, 11.3, 12.5,
                  11.4, 12.9],
                 [6.9, 7.5, 6.1, 8.1, 9.0, 6.0, 8.5, 8.3, 11.2, 7.6, 9.3, 7.5, 9.2, 6.4, 6.3, 7.7, 8.3, 7.5, 9.5, 9.3]])


def cov(X):
    '''
    Finds the -without mean subtraction- covariance of X

    Args:
        X: a |2xn| numpy.array 

    Returns:
        2x2 covariance matrix
    '''
    result = numpy.array([[0, 0], [0, 0]])
    for xi in X.T:
        result = numpy.add(result, numpy.matmul(numpy.array([xi]).T, numpy.array([xi])))
    return result / X.shape[1]


def eigens(C):
    '''
    Returns the sorted eigen-vectors and eigen-values of C
    
    Args:
        C: covariance matrix

    Returns:

    '''
    evals, evects = numpy.linalg.eig(cov(C))
    return -numpy.sort(-evals), evects.T[numpy.argsort(-evals)].T


def center(X):
    '''
    Centers X to mean
    
    Args:
        X: a |2xn| numpy.array 

    Returns:
        centered data
    '''
    mu = numpy.mean(X, axis=1)
    CX = numpy.array(X)
    for c in range(CX.shape[1]):
        CX[:, c] -= mu
    return CX


def plot(X, color):
    vals, vecs = eigens(cov(X))
    pc = vecs[:, 0]
    Y = numpy.matmul(pc, X)

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    plt.subplot(gs[0])
    plt.scatter(X[0], X[1], color=color)
    plt.ylabel('Original Data')
    plt.subplot(gs[1])
    plt.scatter(Y, [0] * len(Y), color=color)
    plt.ylabel('Projected Data')
    plt.yticks([])

    plt.show()

# print(numpy.linalg.eig(cov(X)))
# print(numpy.linalg.eig(cov(center(X))))
print(cov(X))
E = eigens(cov(X))
print('Eigen-Values:',E[0])
print('Eigen-Vectors:')
print(E[1])
plot(X, '#0E6C75')
print('='*100)
print(cov(center(X)))
E2 = eigens(center(X))
print('Eigen-Values:',E2[0])
print('Eigen-Vectors:')
print(E2[1])
plot(center(X), '#BB3356')
