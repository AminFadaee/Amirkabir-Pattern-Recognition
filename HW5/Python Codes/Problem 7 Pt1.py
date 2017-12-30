import numpy
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ============ Reading the Data =============
train = loadmat('Subset1YaleFaces.mat')
X_train = train['X'].T / 256
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
print(list(float('%.3f' %round(e,3)) for e in eigen_vals))
# ============ Display Mean Face ============
plt.imshow(numpy.reshape(mu, (50, 50), order='F'), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Mean Face')
plt.show()

# ========== Display 9 Eigen Faces ==========
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(numpy.reshape(eigen_vecs[:, i], (50, 50), order='F'), cmap='gray')
    plt.title('{0}{1} Eigen Face'.format(i + 1, {1: 'st', 2: 'nd', 3: 'rd'}[i + 1] if i < 3 else 'th'))
    plt.xticks([])
    plt.yticks([])
plt.show()

# ============ Reduction Example ============
for index in [11, 42, 25]:
    img = X_train[:, index]
    plt.subplot(2, 3, 1)
    plt.imshow(numpy.reshape(img, (50, 50), order='F'), cmap='gray')
    plt.title("Original")
    plt.xticks([])
    plt.yticks([])
    i = 2
    for M in [65, 45, 25, 15, 10]:
        E = eigen_vecs[:, 0:M]
        rec_img = numpy.zeros((1, 2500))
        for j in range(M):
            wj = numpy.matmul(E[:, j], img)
            rec_img = numpy.add(rec_img, wj * E[:, j])
        plt.subplot(2, 3, i)
        plt.imshow(numpy.reshape(rec_img + mu, (50, 50), order='F'), cmap='gray')
        plt.title('M={0}'.format(M))
        plt.xticks([])
        plt.yticks([])
        i += 1
    plt.show()
