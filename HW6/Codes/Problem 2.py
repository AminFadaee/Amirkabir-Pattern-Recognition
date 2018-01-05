import numpy
import matplotlib.pyplot as plt

plt.style.use('ggplot')

X1 = numpy.array([(4, 1), (2, 4), (2, 3), (3, 6), (4, 4)]).T
X2 = numpy.array([(9, 10), (6, 8), (9, 5), (8, 7), (10, 8)]).T

plt.scatter(X1[0], X1[1], color='#24877D', label='Class 1')
plt.scatter(X2[0], X2[1], color='#FE6005', label='Class 2')
plt.legend()
plt.show()

m = (numpy.sum(X1, axis=1) + numpy.sum(X2, axis=1)) / 10
plt.scatter(X1[0], X1[1], color='#24877D', label='Class 1')
plt.scatter(X2[0], X2[1], color='#FE6005', label='Class 2')
plt.arrow(m[0], m[1], -0.44046095, -0.18822023, color='#0C2B43', width=0.05, length_includes_head=True)
plt.plot([-0.44046095*8.5+m[0],0.44046095*9.8+m[0]], [-0.18822023*8.5+m[1],0.18822023*9.8+m[1]], color='#DECE2A')
plt.legend()
plt.show()

plt.scatter(numpy.matmul([-0.44046095,-0.18822023],X1),[0]*5, color='#24877D', label='Class 1')
plt.scatter(numpy.matmul([-0.44046095,-0.18822023],X2),[0]*5, color='#FE6005', label='Class 2')
plt.legend()
plt.yticks([])
plt.show()