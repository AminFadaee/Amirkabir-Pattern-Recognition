import numpy
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.style.use('ggplot')


def plot(vector, mult):
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    plt.subplot(gs[0])
    plt.scatter(X1[0], X1[1], color='#2D725A', alpha=0.2)
    plt.scatter(X2[0], X2[1], color='#D8156B', alpha=0.2)
    plt.plot([mu[0] - vector[0] * mult, mu[0] + vector[0] * mult],
             [mu[1] - vector[1] * mult, mu[1] + vector[1] * mult],
             color='#E08130', linewidth=3)
    plt.ylabel('Original Data')
    plt.subplot(gs[1])
    Y1 = numpy.matmul(numpy.array([vector]), X1)
    Y2 = numpy.matmul(numpy.array([vector]), X2)
    plt.scatter(Y1, [0] * 1000, color='#2D725A', alpha=0.2)
    plt.scatter(Y2, [0] * 1000, color='#D8156B', alpha=0.2)
    plt.yticks([])
    plt.ylabel('Projected Data')
    plt.show()


def pca_reconstruction_error(X, pc):
    Y = numpy.matmul(numpy.array([pc]), X)
    reconstructed_X = numpy.matmul(numpy.array([pc]).T, Y)
    difference = numpy.subtract(X, reconstructed_X)
    s = 0
    for d in difference.T:
        s += numpy.linalg.norm(d)
    return s


mu1 = numpy.array([10, 10])
mu2 = numpy.array([22, 10])
cov = numpy.array([[4, 4],
                   [4, 9]])

X1 = numpy.random.multivariate_normal(mu1, cov, 1000).T
X2 = numpy.random.multivariate_normal(mu2, cov, 1000).T
X = numpy.concatenate((X1, X2), axis=1)
mu = numpy.mean(X, axis=1)
# Plotting the points

# Finding the PCA
evals, evects = numpy.linalg.eig(numpy.cov(X))
evals, evects = -numpy.sort(-evals), evects.T[numpy.argsort(-evals)].T
print('Principle Component: {0}^T'.format(evects[:, 0]))
print(pca_reconstruction_error(X, evects[:, 0]))
plot(vector=evects[:, 0], mult=13)

S1 = numpy.cov(X1) * 999
S2 = numpy.cov(X2) * 999
SW = numpy.add(S1, S2)
SB = numpy.matmul(numpy.array([mu]).T, numpy.array([mu]))
v = numpy.matmul(numpy.linalg.inv(SW), numpy.array([numpy.subtract(mu1, mu2)]).T)
print('LDA Projection Vector: {0}^T'.format(v.T[0]))
plot(vector=v.T[0], mult=4000)
