import random
import numpy as np
from scipy.sparse import lil_matrix


class JointTopicModel:
    def __init__(self, K, alpha, beta, max_iter, verbose=0):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, V):
        self._X = X
        self._T = len(X)  # number of vocab types
        self._N = len(X[0])  # number of documents
        self._V = V  # number of vocabularies for each t
        self.Z = self._init_Z()
        self.ndk, self.nkv = self._init_params()
        nk = {}
        for t in range(self._T):
            nk[t] = self.nkv[t].sum(axis=1)

        remained_iter = self.max_iter
        while True:
            if self.verbose:
                print(remained_iter)
            for t in np.random.choice(self._T, self._T, replace=False):
                for d in np.random.choice(self._N, self._N, replace=False):
                    for i in np.random.choice(len(self._X[t][d]), len(self._X[t][d]), replace=False):
                        k = self.Z[t][d][i]
                        v = self._X[t][d][i]

                        self.ndk[t][d][k] -= 1
                        self.nkv[t][k][v] -= 1
                        nk[t][k] -= 1

                        self.Z[t][d][i] = self._sample_z(t, d, v, nk[t])

                        self.ndk[t][d][self.Z[t][d][i]] += 1
                        self.nkv[t][self.Z[t][d][i]][v] += 1
                        nk[t][self.Z[t][d][i]] += 1
            remained_iter -= 1
            if remained_iter <= 0:
                break
        return self

    def _init_Z(self):
        Z = {}
        for t in range(self._T):
            Z[t] = []
            for d in range(len(self._X[t])):
                Z[t].append(np.random.randint(
                    low=0, high=self.K, size=len(self._X[t][d])))
        return Z

    def _init_params(self):
        ndk = {}
        nkv = {}
        for t in range(self._T):
            ndk[t] = np.zeros((self._N, self.K)) + self.alpha
            nkv[t] = np.zeros((self.K, self._V[t])) + self.beta
            for d in range(self._N):
                for i in range(len(self._X[t][d])):
                    k = self.Z[t][d][i]
                    v = self._X[t][d][i]
                    ndk[t][d, k] += 1
                    nkv[t][k, v] += 1
        return ndk, nkv

    def _sample_z(self, t, d, v, nk):
        nkv = self.nkv[t][:, v]  # k-dimensional vector
        prob = (sum([self.ndk[t][d] for t in range(self._T)]) -
                self.alpha*(self._T-1)) * (nkv/nk)
        prob = prob/prob.sum()
        z = np.random.multinomial(n=1, pvals=prob).argmax()
        return z
