import numpy as np
from scipy.special import expit as sigmoid
from sklearn.base import BaseEstimator, ClassifierMixin

from algos.utils import to_sign, to_binary


def kernel_ridge_regression(K, Y, W=None, lbda=1.0):
    n = len(K)

    if W is None:
        W = np.ones(n)

    Wh = np.sqrt(np.diag(W))
    L = Wh @ K @ Wh + n * lbda * np.eye(n)
    alpha = Wh @ np.linalg.solve(L, Wh @ Y)

    return alpha


class KernelLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, iter_max=20, tolerance=1E-6, lbda=1.0):

        self.lbda = lbda
        self.tolerance = tolerance
        self.iter_max = iter_max

    @property
    def _pairwise(self):
        return True

    def fit(self, K, y):
        alpha_t = np.zeros(len(K))
        y = to_sign(y)

        for iter in range(self.iter_max):
            m = K @ alpha_t
            P_t = - sigmoid(- y * m)
            W_t = sigmoid(m) * sigmoid(-m)
            z_t = m - P_t * y / W_t

            alpha_p = kernel_ridge_regression(K, z_t, W_t, lbda=self.lbda)

            delta = np.linalg.norm(alpha_p - alpha_t)
            alpha_t = alpha_p

            if delta < self.tolerance:
                break

        self.alpha_ = alpha_t
        self.fitted_ = True
        self.K_fit_ = K

        return self

    def predict(self, K):
        return to_binary(np.sign(K @ self.alpha_))
