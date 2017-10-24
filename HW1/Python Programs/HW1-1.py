import numpy
import matplotlib.pyplot as plt
import seaborn
from math import sqrt

mu = -1
variances = [0.25, 0.5, 1]
n = 1000
colors = ['#0D9AB6', '#E2CD30', '#C76A2D']
samples = []
for standardDeviation in map(sqrt, variances):
    samples.append(numpy.random.normal(mu, standardDeviation, n))

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.xlim(-5, 3)
    plt.ylim(0, 1)
    plt.title('Variance = {0}'.format(variances[i]))
    seaborn.distplot(samples[i], color=colors[i], hist_kws={'alpha': 1}, kde_kws={'color': 'k'})
plt.plot()
plt.show()
