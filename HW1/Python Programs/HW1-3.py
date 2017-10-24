import numpy

x = numpy.array([0, 1, 2])
probs = numpy.array([0.25, 0.5, 0.25])
square = lambda x: x ** 2
g = lambda x: 3*x+2
print('a.', 'E(X)=', numpy.sum(x * probs.T))
print('b.','E(X^2)=',numpy.sum(numpy.vectorize(square)(x)*probs.T))
print('c.','var(X)= E(X^2) - E(X)^2 =',numpy.sum(numpy.vectorize(square)(x)*probs.T)-numpy.sum(x * probs.T)**2)
print('d.','E(g(x))=',numpy.sum(numpy.vectorize(g)(x)*probs.T))



