import numpy
import matplotlib.pyplot as plt
from math import e

x = numpy.linspace(0, 5, 1000)
plt.plot(x, list(map(lambda x: 0.25 if 2 <= x <= 4 else 0, x)))
plt.plot(x, list(map(lambda x: (1 / 2) * e ** (-x), x)))
plt.plot([2, 2], [0, 0.5], linestyle=':', linewidth=3, alpha=0.3)
plt.plot([4, 4], [0, 0.5], linestyle=':', linewidth=3, alpha=0.3)
plt.fill_between([2, 4], [0.5, 0.5], alpha=0.1)
plt.fill_between([0, 2], [0.5, 0.5], color='orange', alpha=0.1)
plt.fill_between([4, 5], [0.5, 0.5], color='orange', alpha=0.1)
plt.show()

plt.plot(x, list(map(lambda x: 0.25 if 2 <= x <= 4 else 0, x)))
plt.plot(x, list(map(lambda x: (1 / 2) * e ** (-x), x)))
plt.show()

plt.plot(x, list(map(lambda x: 0.25 if 2 <= x <= 4 else 0, x)))
plt.plot(x, list(map(lambda x: (1 / 2) * e ** (-x), x)))
plt.fill_between(numpy.linspace(2, 4, 1000), list(map(lambda x: (1 / 2) * e ** (-x), numpy.linspace(2, 4, 1000))),
                 color='red')
plt.show()
