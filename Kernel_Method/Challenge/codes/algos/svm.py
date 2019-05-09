import numpy as np
import qpsolvers
from sklearn.base import BaseEstimator, ClassifierMixin

from algos.utils import to_sign, to_binary


class SVM(BaseEstimator, ClassifierMixin):
    def __init__(self, lbda=1.0):
        self.lbda = lbda

    @property
    def _pairwise(self):
        return True

    def fit(self, K, y):
        y = to_sign(y)

        n = len(y)

        q = - y.astype(float)
        P = K
        G = np.zeros((2 * n, n))
        G[:n, :] = - np.diag(y)
        G[n:, :] = np.diag(y)
        h = np.zeros(2 * n)
        h[n:] = 1 / (2 * self.lbda * n)

        alpha = qpsolvers.solve_qp(P, q, G, h, solver='cvxopt')

        self.alpha_ = alpha
        self.fitted_ = True
        self.K_fit_ = K

        return self

    def predict(self, K):
        return to_binary(np.sign(K @ self.alpha_))
