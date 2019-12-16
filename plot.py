import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import beta, dirichlet, gamma, multinomial
from mpl_toolkits.mplot3d import Axes3D


class Plot():
    def __init__(self, alpha=np.array([1, 1, 1])):
        self.alpha = alpha

    def dirichlet(self, x):
        array = np.array([x, self.alpha])
        self.param = np.sum(array, axis=0)
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # triangle mesh grid (0,0)-(1,0)-(0,1)
        xx = np.array([[0.01*a*0.01*(100-b) for a in range(1, 100)]
                       for b in range(1, 100)])
        yy = np.array([[0.01*b] * 99 for b in range(1, 100)])

        # Dirichlet PDF on mesh grid ((0,0)->(0,0,1), (1,0)->(1,0,0), (0,1)->(0,1,0))
        di = dirichlet(alpha=self.param)
        Z = di.pdf([xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # transform isosceles right triangle mesh into equilateral triangle
        xx2 = np.array([x + (0.5 - np.average(x)) for x in xx])
        yy2 = yy * np.sqrt(3) / 2

        ax.plot_surface(xx2, yy2, Z, cmap=cm.coolwarm, antialiased=False)
        plt.show()

    def posterior(self, x):
        def f(i, j):
            return gamma(i+j)/gamma(j)
        prod = []
        for i, j in x, self.alpha:
            prod.append(f(i, j))
        return np.prod(prod)

    def fit(self, n, x):
        """
        n = len(data)
        x = [x1,x2,x3]
        """
        A = sum(self.alpha)
        b = gamma(A)/gamma(n+A)
        model = b*self.posterior(x)
        return model
