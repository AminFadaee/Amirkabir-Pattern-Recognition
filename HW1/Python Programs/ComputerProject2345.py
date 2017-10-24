import numpy
from numpy.linalg import inv as inverse, det
from numpy import log as ln
import pandas
from math import pi, e, sqrt

w1 = pandas.DataFrame([[-5.01, -8.12, -3.68],
                       [-5.43, -3.48, -3.54],
                       [1.08, -5.52, 1.66],
                       [0.86, -3.78, -4.11],
                       [-2.67, 0.63, 7.39],
                       [4.94, 3.29, 2.08],
                       [-2.51, 2.09, -2.59],
                       [-2.25, -2.13, -6.94],
                       [5.56, 2.86, -2.26],
                       [1.03, -3.33, 4.33]])
w2 = pandas.DataFrame([[-0.91, -0.18, -0.05],
                       [1.30, -2.06, -3.53],
                       [-7.75, -4.54, -0.95],
                       [-5.47, 0.50, 3.92],
                       [6.14, 5.72, -4.85],
                       [3.60, 1.26, 4.36],
                       [5.37, -4.63, -3.65],
                       [7.18, 1.46, -6.66],
                       [-7.39, 1.17, 6.30],
                       [-7.50, -6.32, -0.31]])
w3 = pandas.DataFrame([[5.35, 2.26, 8.13],
                       [5.12, 3.22, -2.66],
                       [-1.34, -5.31, -9.87],
                       [4.48, 3.42, 5.19],
                       [7.11, 2.39, 9.21],
                       [7.17, 4.33, -0.98],
                       [5.75, 3.97, 6.65],
                       [0.77, 0.27, 2.41],
                       [0.90, -0.43, -8.71],
                       [3.52, -0.36, 6.43]])

mu1 = w1.apply(numpy.mean, axis=0)  # mean along the columns
mu2 = w2.apply(numpy.mean, axis=0)
mu3 = w3.apply(numpy.mean, axis=0)

K = numpy.matrix(w1.apply(lambda row: numpy.array(row) - mu1, axis=1))
cov1 = K.T * K / (K.shape[0]-1)

K = numpy.matrix(w2.apply(lambda row: numpy.array(row) - mu2, axis=1))
cov2 = K.T * K / (K.shape[0]-1)

K = numpy.matrix(w3.apply(lambda row: numpy.array(row) - mu3, axis=1))
cov3 = K.T * K / (K.shape[0]-1)

def mahalanobis_distance(X, mu, cov):
    X = numpy.matrix(X).T  # column vector
    mu = numpy.matrix(mu).T  # column vector
    cov = numpy.matrix(cov)
    return float(sqrt((X - mu).T * inverse(cov) * (X - mu)))


def bhattacharyya_bound(mu1, mu2, cov1, cov2, P1, P2):
    mu1 = numpy.matrix(mu1).T  # column vector
    mu2 = numpy.matrix(mu2).T  # column vector
    cov1 = numpy.matrix(cov1)
    cov2 = numpy.matrix(cov2)
    k = float((1 / 8) * (mu2 - mu1).T * inverse((cov1 + cov2) / 2) * (mu2 - mu1) + (1 / 2) * ln(
        det((cov1 + cov2) / 2) / sqrt(det(cov1) * det(cov2))))
    return round(sqrt(P1 * P2) * e ** (-k), 4)


def g(x, mu, cov, P):
    x = numpy.matrix(x).T  # column vector
    mu = numpy.matrix(mu).T  # column vector
    cov = numpy.matrix(cov)
    d = x.shape[0]
    return float(
        (-1 / 2) * (x - mu).T * inverse(cov) * (x - mu) - (d / 2) * ln(2 * pi) - (1 / 2) * ln(det(cov)) + ln(P))


def dichotomizer(data, labels, mu1, mu2, cov1, cov2, P1, P2):
    # Class 1 is True and class 2 is False.
    predictions = data.apply(lambda x: (g(x, mu1, cov1, P1) - g(x, mu2, cov2, P2)) > 0, axis=1)
    result = dict()
    result['accuracy'] = float(str(round((predictions == labels).sum() / len(labels), 5)))
    result['error'] = float(str(round(1 - result['accuracy'], 5)))
    result['bound'] = bhattacharyya_bound(mu1, mu2, cov1, cov2, P1, P2)
    return result


def trichotomizer(data, mu1, mu2, mu3, cov1, cov2, cov3, P1, P2, P3):
    # Class 1 is -1, class 2 is 0 and class 3 is 1
    predictions = data.apply(lambda x: numpy.argmax([g(x, mu1, cov1, P1),
                                                     g(x, mu2, cov2, P2),
                                                     g(x, mu3, cov3, P3)])+ 1, axis=1)
    return predictions


# Problem 2: a,b,c
data = pandas.DataFrame(pandas.concat([w1[0], w2[0]]).reset_index(drop=True))
labels = pandas.Series([True] * 10 + [False] * 10)
C1 = cov1[0, 0]
C2 = cov2[0, 0]
m1 = mu1[0]
m2 = mu2[0]
P1 = P2 = 1 / 2
print(dichotomizer(data, labels, m1, m2, C1, C2, P1, P2))

# Problem 2: d
data = pandas.concat([w1[[0, 1]], w2[[0, 1]]]).reset_index(drop=True)
labels = pandas.Series([True] * 10 + [False] * 10)
C1 = cov1[0:2, 0:2]
C2 = cov2[0:2, 0:2]
m1 = mu1[0:2]
m2 = mu2[0:2]
P1 = P2 = 1 / 2
print(dichotomizer(data, labels, m1, m2, C1, C2, P1, P2))

# Problem 2: e
data = pandas.concat([w1, w2]).reset_index(drop=True)
labels = pandas.Series([True] * 10 + [False] * 10)
C1 = cov1
C2 = cov2
m1 = mu1
m2 = mu2
P1 = P2 = 1 / 2
print(dichotomizer(data, labels, m1, m2, C1, C2, P1, P2))

# Problem 3: a,b,c
data = pandas.DataFrame(pandas.concat([w1[0], w3[0]]).reset_index(drop=True))
labels = pandas.Series([True] * 10 + [False] * 10)
C1 = cov1[0, 0]
C2 = cov3[0, 0]
m1 = mu1[0]
m2 = mu3[0]
P1 = P2 = 1 / 2
print(dichotomizer(data, labels, m1, m2, C1, C2, P1, P2))

# Problem 3: d
data = pandas.concat([w1[[0, 1]], w3[[0, 1]]]).reset_index(drop=True)
labels = pandas.Series([True] * 10 + [False] * 10)
C1 = cov1[0:2, 0:2]
C2 = cov3[0:2, 0:2]
m1 = mu1[0:2]
m2 = mu3[0:2]
P1 = P2 = 1 / 2
print(dichotomizer(data, labels, m1, m2, C1, C2, P1, P2))

# Problem 3: e
data = pandas.concat([w1, w3]).reset_index(drop=True)
labels = pandas.Series([True] * 10 + [False] * 10)
C1 = cov1
C2 = cov3
m1 = mu1
m2 = mu3
P1 = P2 = 1 / 2
print(dichotomizer(data, labels, m1, m2, C1, C2, P1, P2))

# Problem 4: a,b,c
data = pandas.DataFrame(pandas.concat([w2[0], w3[0]]).reset_index(drop=True))
labels = pandas.Series([True] * 10 + [False] * 10)
C1 = cov2[0, 0]
C2 = cov3[0, 0]
m1 = mu2[0]
m2 = mu3[0]
P1 = P2 = 1 / 2
print(dichotomizer(data, labels, m1, m2, C1, C2, P1, P2))

# Problem 4: d
data = pandas.concat([w2[[0, 1]], w3[[0, 1]]]).reset_index(drop=True)
labels = pandas.Series([True] * 10 + [False] * 10)
C1 = cov2[0:2, 0:2]
C2 = cov3[0:2, 0:2]
m1 = mu2[0:2]
m2 = mu3[0:2]
P1 = P2 = 1 / 2
print(dichotomizer(data, labels, m1, m2, C1, C2, P1, P2))

# Problem 4: e
data = pandas.concat([w2, w3]).reset_index(drop=True)
labels = pandas.Series([True] * 10 + [False] * 10)
C1 = cov2
C2 = cov3
m1 = mu2
m2 = mu3
P1 = P2 = 1 / 2
print(dichotomizer(data, labels, m1, m2, C1, C2, P1, P2))

# Problem 5: a
X = [[1, 2, 1],
     [5, 3, 2],
     [0, 0, 0],
     [1, 0, 0]]
for i in range(4):
    for j in range(3):
        print('Distance of {0} from {1} is:\t'.format(tuple(X[i]), ['mu1', 'mu2', 'mu3'][j]), end='')
        print(round(mahalanobis_distance(X[i], [mu1, mu2, mu3][j], [cov1, cov2, cov3][j]), 4))

# Problem 5: b
data = pandas.DataFrame(X)
C1 = cov1
C2 = cov2
C3 = cov3
m1 = mu1
m2 = mu2
m3 = mu3
P1 = P2 = P3 = 1 / 3
print(trichotomizer(data, m1, m2, m3, C1, C2, C3, P1, P2, P3))

# Problem 5: c
data = pandas.DataFrame(X)
C1 = cov1
C2 = cov2
C3 = cov3
m1 = mu1
m2 = mu2
m3 = mu3
P1 = 0.8
P2 = P3 = 0.1
print(trichotomizer(data, m1, m2, m3, C1, C2, C3, P1, P2, P3))
