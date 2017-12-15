import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def distance(x, y):
    '''
    Computes the euclidean distance between x,y
    Args:
        x: point in a space
        y: another point in the same space

    Returns:
        euclidean distance
    '''
    x = numpy.array(x)
    y = numpy.array(y)
    return numpy.sqrt(numpy.sum((x - y) ** 2))


def nearest_neighbor(x, w1, w2, w3=None):
    '''
    Implements 2 or 3 class nearest neighbor classification of point x

    Args:
        x: point to classify
        w1: data of class 1 
        w2: data of class 2
        w3: data of class 3 or None for 2 class classification

    Returns:
        1 or 2 (or 3)
    '''
    w1_best = min(list(distance(y, x) for y in w1))
    w2_best = min(list(distance(y, x) for y in w2))
    if w3 is not None:
        w3_best = min(list(distance(y, x) for y in w3))
        if w3_best <= w2_best and w3_best <= w1_best:
            return 3
    return (w2_best < w1_best) + 1


def plot_decision_boundary(w1, w2, w3=None):
    '''
    Plots nearest neighbor decision boundary for 2 or 3 classes based on matplotlib's pcolormesh
    
    Args:
        w1: data for class 1 
        w2: data for class 2
        w3: data of class 3 or None for 2 class plot

    Returns:
        None
    '''
    plt.figure(figsize=(5, 5))
    xx, yy = numpy.meshgrid(numpy.arange(-6, 11, 0.1), numpy.arange(-11, 11, 0.1))
    Z = list(nearest_neighbor(x, w1, w2, w3) for x in numpy.c_[xx.ravel(), yy.ravel()])
    Z = numpy.array(Z).reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=ListedColormap(['#D3A752', '#7BC2B7','#87B477']))
    plt.scatter(w1[:, 0], w1[:, 1], color='#891D1B')
    plt.scatter(w2[:, 0], w2[:, 1], color='#324738')
    if w3 is not None:
        plt.scatter(w3[:, 0], w3[:, 1], color='#2C3248')

    plt.show()


w1 = numpy.array([[10, 0],
                  [0, -10],
                  [5, -2]])

w2 = numpy.array([[5, 10],
                  [0, 5],
                  [5, 5]])

w3 = numpy.array([[2, 8],
                  [-5, 2],
                  [10, -4]])

plot_decision_boundary(w1, w2)
plot_decision_boundary(w1, w3)
plot_decision_boundary(w2, w3)
plot_decision_boundary(w1, w2, w3)
