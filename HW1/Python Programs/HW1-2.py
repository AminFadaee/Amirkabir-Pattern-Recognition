import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy
from scipy.stats import multivariate_normal


def plot_multi_normal(mu, cov, i):
    color_map=[cm.winter,cm.autumn,cm.copper]
    cov = numpy.array(cov)
    cov = cov*numpy.eye(2)
    a = -5.0
    x, y = numpy.mgrid[-a:a:30j, -a:a:30j]
    xy = numpy.column_stack([x.flat, y.flat])
    z = multivariate_normal.pdf(xy, mean=mu, cov=cov)
    z = z.reshape(x.shape)
    ax = plt.subplot(13 * 10 + i, projection='3d')
    ax.plot_surface(x, y, z,cmap=color_map[i-1])
    ax.set_zlim(0,0.06)


mu = [[1, 1], [-1, 1], [1, 0]]
cov = [[[2, 0],
        [0, 2]],
       [[2, 0],
        [0, 5]],
       [[2, 5],
        [5, 3]]]

for i in range(3):
    print(mu[i])
    print(cov[i])
    plot_multi_normal(mu=mu[i], cov=cov[i], i=i + 1)
plt.show()
