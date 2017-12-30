import numpy
import matplotlib.pyplot as plt

plt.style.use('ggplot')
X = numpy.array([[3, 2, 4, 0, 6, 3, 1, 5, -1, 7],
                 [1, 3, -1, 7, -5, 1, 0, 2, -1, 3]])
plt.plot(X[0], X[1], 'ko')
plt.show()

Y = (X.T - numpy.mean(X, axis=1)).T
# [[ 0. -1.  1. -3.  3.  0. -2.  2. -4.  4.]
#  [ 0.  2. -2.  6. -6.  0. -1.  1. -2.  2.]]
plt.plot(Y[0], Y[1], 'ko')
plt.show()

C = numpy.cov(Y)
# [[  6.6  -2.2]
#  [ -2.2  10. ]]
print(numpy.linalg.eig(C.T))
# Λ = [5.55555556,  11.11111111]
# Φ = [[-0.89442719,  0.4472136 ],
#     [-0.4472136 , -0.89442719]]
print(numpy.sum(Y ** 2, axis=1) / 9)

phi1 = [0.44, -0.89]
phi2 = [0.89, 0.43]
plt.plot(Y[0], Y[1], 'ko')
plt.quiver([0], [0], [phi2[0]], [phi2[1]], angles='xy', scale_units='xy', scale=0.5, color='#124B57')
plt.quiver([0], [0], [phi1[0]], [phi1[1]], angles='xy', scale_units='xy', scale=0.5, color='#EB3F3D')
plt.show()

Y1 = numpy.matmul(Y.T, numpy.array(phi1).T)
Y2 = numpy.matmul(Y.T, numpy.array(phi2).T)
plt.scatter(Y1, [2] * 10, color='#124B57', label='Phi1 Points')
plt.scatter(Y2, [1] * 10, color='#EB3F3D', label='Phi2 Points')
plt.yticks([])
plt.legend()
plt.show()
