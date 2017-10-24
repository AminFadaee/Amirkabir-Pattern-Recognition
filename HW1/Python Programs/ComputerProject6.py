import numpy
from random import random, randint
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn
from openpyxl.chart import title


def generate_uniform_random(x1, xu, n):
    results = numpy.zeros(n)
    for i in range(n):
        results[i] = random() * (xu - x1) + x1
    return results


def generate_random_parameters():
    x1 = randint(-100, 100)
    xu = randint(-100, 100)
    if x1 > xu:
        x1, xu = xu, x1
    n = randint(1, 1000)
    return x1, xu, n


def plotNormal(mu, sd):
    x = numpy.linspace(mu - 3 * sd, mu + 3 * sd, 100)
    plt.plot(x, mlab.normpdf(x, mu, sd), 'k')


def generate_and_plot():
    x1, xu, n = generate_random_parameters()
    iters = [10000, 100000,1000000]
    for j in range(3):
        R = numpy.zeros(iters[j])
        for i in range(iters[j]):
            R[i] = generate_uniform_random(x1, xu, n).mean()
            print(i)
        sd = R.std()  # standard deviation
        mu = R.mean()
        plt.subplot(1, 3, j + 1, title=iters[j])
        plotNormal(mu, sd)
        seaborn.distplot(R, norm_hist=True, color=['#0D9AB6', '#E2CD30', '#C76A2D'][j],
                         kde=False, hist_kws={'alpha': 1})
    plt.show()


generate_and_plot()
