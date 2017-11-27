import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn
import sklearn
import scipy

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


def T(delta, mu, x):
    if abs(x - mu) < delta:
        return (delta - abs(x - mu)) / (delta ** 2)
    return 0


def p_x_D(A, B, D, x, steps):
    MU = numpy.linspace(A, B, steps)
    DELTA = numpy.linspace(A, B, steps)
    P = 0
    a = MU[1] - MU[0]
    b = DELTA[1] - DELTA[0]
    for delta in DELTA:
        for mu in MU:
            p = T(delta, mu, x)
            for k in range(len(D)):
                p *= T(delta, mu, D[k])
            P += p*a*b
    return P


A, B = (min(w2[:, 1]), max(w2[:, 1]))
D = w2[:, 1]  # feature 2 of w2
y = []
for x in numpy.linspace(-0.6, 1.4, 1000):
    y.append(p_x_D(A, B, D, x, 1000))
    print(x, y[-1])
plt.plot(numpy.linspace(-0.6, 1.4, 1000), y)
plt.show()
print(y)
