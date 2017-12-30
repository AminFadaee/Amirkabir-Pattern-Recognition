import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn
import sklearn
import scipy
from scipy.io import loadmat
import statistics


def distance(p1, p2):
    '''
    Computes the euclidean distance between p1 and p2

    Args:
        p1: an array like collection representing an n dimensional point
        p2: an array like collection representing an n dimensional point

    Returns:
        float, the euclidean distance
    '''
    p1 = numpy.array(p1)
    p2 = numpy.array(p2)
    return numpy.sqrt(numpy.sum((p1 - p2) ** 2))


def mode(L):
    D = dict()
    for l in L:
        if l in D:
            D[l] += 1
        else:
            D[l] = 1
    return max(D, key=lambda i: D[i])


def KNN(train_X, train_Y, instance, k):
    distances = []
    for i in range(train_X.shape[1]):
        data = train_X[:, i]
        distances.append((distance(data, instance), i))
    knn = sorted(distances, key=lambda i: i[0])[0:k]
    knn_labels = list(train_Y[i[1]] for i in knn)
    return mode(knn_labels)


def pca(X_train):
    N2, M = X_train.shape
    A = numpy.array(X_train)
    # ============ Centering X_train ============
    mu = numpy.mean(A, axis=1)
    for m in range(A.shape[1]):
        A[:, m] -= mu
    # ========= Finding Eigens of A^T A =========
    ATA = numpy.matmul(A.T, A)
    ATA_eigen_vals, ATA_eigen_vecs = numpy.linalg.eig(ATA)
    # ========== Finding Eigens of AAT ==========
    eigen_vals = ATA_eigen_vals
    eigen_vecs = numpy.matmul(A, ATA_eigen_vecs)
    for v in range(eigen_vecs.shape[1]):
        norm = numpy.linalg.norm(eigen_vecs[:, v])
        eigen_vecs[:, v] /= norm
    # === Sorting The Vectors Based on Values ===
    eigen_vals, eigen_vecs = -numpy.sort(-eigen_vals), eigen_vecs.T[numpy.argsort(-eigen_vals)].T
    return eigen_vals, eigen_vecs


train = loadmat('Subset1YaleFaces.mat')
validation = loadmat('Subset2YaleFaces.mat')
test = loadmat('Subset3YaleFaces.mat')
X_train, Y_train = train['X'].T / 256, train['Y'].T[0]
X_valid, Y_valid = validation['X'].T / 256, validation['Y'].T[0]
X_test, Y_test = test['X'].T / 256, test['Y'].T[0]

eigen_vals, eigen_vecs = pca(X_train)
for M in [60,20, 10, 5]:
    reduced_X_train = numpy.matmul(eigen_vecs[:, :M].T, X_train)
    reduced_X_valid = numpy.matmul(eigen_vecs[:, :M].T, X_valid)
    reduced_X_test = numpy.matmul(eigen_vecs[:, :M].T, X_test)
    accuracies = []
    for k in range(1, 11):
        correct = 0
        for i in range(reduced_X_valid.shape[1]):
            img = reduced_X_valid[:, i]
            prediction = KNN(reduced_X_train, Y_train, img, k)
            correct += int(prediction == Y_valid[i])
        accuracy = correct / 70
        accuracies.append(accuracy)
    plt.plot(range(1, 11), accuracies, label='M={0}'.format(M))
    print(accuracies)
plt.legend()
plt.show()

M = 60
k = 3
reduced_X_train = numpy.matmul(eigen_vecs[:, :M].T, X_train)
reduced_X_test = numpy.matmul(eigen_vecs[:, :M].T, X_test)
correct = 0
for i in range(reduced_X_test.shape[1]):
    img = reduced_X_test[:, i]
    prediction = KNN(reduced_X_train, Y_train, img, k)
    correct += int(prediction == Y_test[i])
accuracy = correct / 70
print("Test Accuracy:", accuracy)
