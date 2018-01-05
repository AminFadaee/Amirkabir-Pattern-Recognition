import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Plotting Aesthetics
plt.style.use('ggplot')
colormap = LinearSegmentedColormap.from_list('custom', ['#F45B19', '#44A17C', '#A12834', '#ECDF2E', '#3E1443'])


def find_new_w(w, data, labels, rate=1):
    aug_data = numpy.multiply(numpy.concatenate((numpy.ones((1, data.shape[1])), data)), labels)
    for i in range(data.shape[1]):
        if numpy.matmul(aug_data[:, i], w) < 0:
            w = w + rate * aug_data[:, i]
    return w


data = numpy.array([[0.1, 0.15, 0.3, 0.35, 0.45, 0.6, 0.7, 0.9],
                    [0.7, 1, 0.55, 0.95, 0.15, 0.3, 0.65, 0.45]])
labels = numpy.array([1, -1, 1, -1, 1, 1, -1, -1])

w0 = numpy.array([0.2, 1, -1])
w1 = find_new_w(w0, data, labels)
w2 = find_new_w(w1, data, labels)
w3 = find_new_w(w2, data, labels)
w4 = find_new_w(w3, data, labels)

plt.figure(figsize=(8, 8))
cols = ['#F45B19', '#44A17C', '#A12834', '#ECDF2E', '#0D8EBF']
for i, w in enumerate((w0, w1, w2, w3, w4)):
    plt.scatter(data[0], data[1], c=labels, cmap=colormap)
    m = -w[1] / w[2]
    b = -w[0] / w[2]
    plt.plot([0, 1], [b, m + b], label='Iter {}'.format(i),color=cols[i],linewidth=1+(0.5*i))
plt.legend()
plt.show()
