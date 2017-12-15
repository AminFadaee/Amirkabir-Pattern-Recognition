import numpy
import matplotlib.pyplot as plt

# w1 = sunny
w1 = [4, 1, 5]

# w2 = cloudy
w2 = [3, 2]

# parzen window width
h = 1


def phi(x, h, D):
    '''
    Implements the rectangular window function for parzen window estimation of 1D data points
    
    Args:
        x: center 
        h: width of parzen window
        D: data

    Returns:
        integer, number of points in the rectangular window
    '''
    k = 0
    for d in D:
        if abs(x - d) < h / 2:
            k += 1
    return k


X = numpy.linspace(-5, 10, 1000)
for D in (w1, w2):
    n = len(D)
    p = []
    for x in X:
        k = phi(x, h, D)
        p.append(k / (n * h))

    plt.subplot(2, 1, (D == w2) + 1)
    plt.title(['Sunny Data','Cloudy Data'][(D == w2)])
    plt.scatter(D, [0] * n, s=100)
    plt.plot(X, p)

plt.show()
