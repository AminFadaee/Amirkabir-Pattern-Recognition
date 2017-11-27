import numpy
import matplotlib.pyplot as plt
from math import e

x = numpy.linspace(0, 10, 100)
y = e ** (-x)
plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title('p(x|θ) versus x for θ = 1')
plt.subplot(1, 2, 2)
theta = numpy.linspace(0, 5, 100)
y = theta * e ** (-theta * 2)
plt.plot(theta, y)
plt.title('p(x|θ) versus θ, (0 ≤ θ ≤ 5), for x = 2')
plt.show()

x = numpy.linspace(0, 10, 100)
y = e ** (-x)
samples = numpy.random.exponential(1,10000000)
est =  len(samples)/sum(samples)
plt.plot(x, y,label='Density')
plt.xticks(list(plt.xticks()[0])+[est])
plt.plot([est, est], [0,e ** (-est)],label='Estimation')
plt.title('θ = 1 and Estimate = {0} for {1} Samples'.format(round(est,5),'10e+6'))
plt.legend()
plt.show()

