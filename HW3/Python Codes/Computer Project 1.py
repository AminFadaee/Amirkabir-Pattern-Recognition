import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn
import sklearn

w1 = numpy.array([[0.42, -0.087, 0.58],
                  [-0.2, -3.3, -3.4],
                  [1.3, -0.32, 1.7],
                  [0.39, 0.71, 0.23],
                  [-1.6, -5.3, -0.15],
                  [-0.029, 0.89, -4.7],
                  [-0.23, 1.9, 2.2],
                  [0.27, -0.3, -0.87],
                  [-1.9, 0.76, -2.1],
                  [0.87, -1.0, -2.6]])

w2 = numpy.array([[-0.4, 0.58, 0.089],
                  [-0.31, 0.27, -0.04],
                  [0.38, 0.055, -0.035],
                  [-0.15, 0.53, 0.011],
                  [-0.35, 0.47, 0.034],
                  [0.17, 0.69, 0.1],
                  [-0.011, 0.55, -0.18],
                  [-0.27, 0.61, 0.12],
                  [-0.065, 0.49, 0.0012],
                  [-0.12, 0.054, -0.063]])

w3 = numpy.array([[0.83, 1.6, -0.014],
                  [1.1, 1.6, 0.48],
                  [-0.44, -0.41, 0.32],
                  [0.047, -0.45, 1.4],
                  [0.28, 0.35, 3.1],
                  [-0.39, -0.48, 0.11],
                  [0.34, -0.079, 0.14],
                  [-0.3, -0.22, 2.2],
                  [1.1, 1.2, -0.46],
                  [0.18, -0.11, -0.49]])


def MLE_mu(data):
    '''
    Computes the maximum likelihood mu for a D-dimensional data.
    Args:
        data: |NxD| data

    Returns:
        mu numpy vector/scalar
    
    >>> MLE_mu([1,2,3,4])
    2.5
    >>> MLE_mu([[1,2],[3,4]])
    [2. 3.]
    '''
    data = numpy.array(data)
    N = len(data)
    return numpy.sum(data, axis=0) / N


def MLE_sigma2(data):
    '''
    Computes the maximum likelihood variance for a collection
    Args:
        data: |N| collection

    Returns:
        sigma2 scalar
    
    >>> MLE_sigma2([1,2,3,4])
    1.25
    '''
    data = numpy.array(data)
    N = len(data)
    mu = numpy.mean(data)
    return numpy.sum((data - mu) ** 2) / N


def MLE_cov(data):
    '''
    Computes the maximum likelihood covariance matrix for the data
    Args:
        data: |NxD| matrix of data

    Returns:
        covariance matrix
    >>> MLE_cov([[3, 2], [1, 2], [2, 0],[4,2]])
    [[ 1.25  0.25]
     [ 0.25  0.75]]    
    '''
    N, D = len(data), len(data[0])
    mu = numpy.mean(data, axis=0)  # mean along columns
    K = numpy.matrix(numpy.zeros((N, D)))
    for i in range(N):
        K[i] = numpy.array(data[i]) - mu
    return K.T * K / N

def MLE_diagonal_cov(data):
    '''
    Computes the maximum likelihood covariance matrix for the data supposing that it is a diagonal matrix
    Args:
        data: |NxD| matrix of data

    Returns:
        covariance matrix
    >>> MLE_diagonal_cov([[3, 2], [1, 2], [2, 0],[4,2]])
    [[ 1.25  0.]
     [ 0.  0.75]] 
    '''
    data = numpy.array(data)
    N, D = len(data), len(data[0])
    cov = numpy.zeros((D,D))
    for i in range(D):
        cov[i][i] = MLE_sigma2(data[:,i])
    return cov

for i in range(3):
    print(MLE_mu(w1[:, i]), MLE_sigma2(w1[:, i]))

for i, j in ((0, 1), (0, 2), (1, 2)):
    print(MLE_mu(w1[:, (i, j)]))
    print(MLE_cov(w1[:, (i, j)]))

print(MLE_mu(w1))
print(MLE_cov(w1))


print(MLE_mu(w2))
print(MLE_diagonal_cov(w2))
