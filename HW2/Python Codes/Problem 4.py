import numpy
import matplotlib.pyplot as plt
from scipy.stats import norm as normal

mu1 = 3
std1 = 3
mu2 = 7
std2 = 4

x = numpy.linspace(-15, 30, 100)
normal1 = normal(mu1, std1)
normal2 = normal(mu2, std2)

# generating n sums of random variables
n = 10000
samples1 = normal1.rvs(size=n)
samples2 = normal2.rvs(size=n)
samples = samples1 + samples2

# finding the mean and std of samples
mu = numpy.mean(samples)
std = numpy.std(samples)
emp = normal(mu, std)

# plotting the data:
plt.plot(x, normal1.pdf(x), label='First distribution')
plt.plot(x, normal2.pdf(x), label='Second distribution')
plt.scatter(samples, [0] * len(samples), alpha=0.1, color='k', label='Samples')
plt.plot(x, emp.pdf(x),linestyle=':',linewidth=3, label='Empirical distribution')
plt.legend()
plt.title('Sample Mean:{0}, Sample Variance:{1}'.format(round(mu, 2), round(std**2, 2)))
plt.show()
