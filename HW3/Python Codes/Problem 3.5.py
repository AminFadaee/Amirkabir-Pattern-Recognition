import numpy
import scipy.special
import matplotlib.pyplot as plt

P = lambda T, p, d: (p ** (T * d)) * ((1 - p) ** (d - T * d)) * scipy.special.binom(d, T * d)


def plot(factor, count, prior, colors):
    x = numpy.linspace(0, 1, 1000)
    for c in range(count):
        plt.subplot(count,1,c+1)
        plt.plot(x, P(x, prior, d=factor * (10 ** c)), color=colors[c])
        plt.plot(x, P(x, 1 - prior, d=factor * (10 ** c)), color=colors[c])
    plt.show()


plot(10, 3, 0.6, ['#E9BE48', '#D3460F', '#602D27', '#0E2931', '#27361F'])


