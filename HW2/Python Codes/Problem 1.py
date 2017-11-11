import numpy
import matplotlib.pyplot as plt

w1 = lambda x: 2 * x
w2 = lambda x: 2 - 2 * x

x = numpy.linspace(0, 1, 2)
y1 = w1(x)
y2 = w2(x)
plt.subplot(1,2,1)
plt.plot(x, y1,label='w1')
plt.plot(x, y2,label='w2')
plt.legend()
plt.title("Densities")
plt.subplot(1,2,2)
plt.plot(x, y1,label='w1')
plt.plot(x, y2,label='w2')
plt.plot([1/2,1/2], [0,2],label='Decision Boundary')
plt.legend()
plt.title("Decision Boundary")
plt.show()
