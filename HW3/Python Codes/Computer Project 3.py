import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn
import sklearn
import scipy.stats
from math import sqrt


def p_x_D(D, mu_0, sigma_0, sigma,label):
    '''
    Plots the density p(x|D) given mu0, sigma0, sigma.
    
    Args:
        D: the samples
        mu_0: mean of the normal distribution for mu
        sigma_0: standard deviation of the normal distribution for mu
        sigma: the standard deviation of p(x|D)
        label: the label for the plot legend
        
    Returns:
        None
    '''
    n = len(D)
    x_bar = numpy.mean(D)
    mu_n = ((n * sigma_0 ** 2) / (n * sigma_0 ** 2 + sigma ** 2)) * x_bar + (sigma ** 2) / (
    n * sigma_0 ** 2 + sigma ** 2) * mu_0
    sigma_2_n = (sigma_0 ** 2 * sigma ** 2) / (n * sigma_0 ** 2 + sigma ** 2)
    # P(mu|D)~N(mu_n,sigma_2_n)
    # P(x|D)~N(mu_n,sigma**2+sigma_2_n)
    X = numpy.linspace(-5, 5, 1000)
    Y = scipy.stats.norm(mu_n, sqrt(sigma**2+sigma_2_n)).pdf(X)

    plt.plot(X, Y,label=label)


w3 = numpy.array([[0.83, 1.6, -0.014],
                  [1.1, 1.6, 0.48],
                  [-0.44, -0.41, 0.32],
                  [0.047, -0.45, 1.4],
                  [0.28, 0.35, 3.1],
                  [-0.39, -0.48, 0.11],
                  [0.34, -0.079, 0.14],
                  [-0.3, -0.22, 2.2],
                  [1.1, 1.2, -0.46],
                  [0.18, -0.11, -0.49]])

D = w3[:, 1]  # feature 2 of w3
sigma = numpy.std(D)
for i in (0.1,1.0,10,100):
    sigma_0 = sqrt((sigma**2)/i)
    p_x_D(D=D, mu_0=-1, sigma_0=sigma_0, sigma=sigma,label='Dogmatism = {0}'.format(i))
plt.legend()
plt.show()