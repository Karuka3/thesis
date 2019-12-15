import numpy as np
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def Dirichlet(alpha=[3, 1, 1]):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    a, b, c = alpha
    # triangle mesh grid (0,0)-(1,0)-(0,1)
    xx = np.array([[0.01*a*0.01*(100-b) for a in range(1, 100)]
                   for b in range(1, 100)])
    yy = np.array([[0.01*b] * 99 for b in range(1, 100)])

    # Dirichlet PDF on mesh grid ((0,0)->(0,0,1), (1,0)->(1,0,0), (0,1)->(0,1,0))
    di = dirichlet([a+1, b+1, c+1])
    Z = di.pdf([xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # transform isosceles right triangle mesh into equilateral triangle
    xx2 = np.array([x + (0.5 - np.average(x)) for x in xx])
    yy2 = yy * np.sqrt(3) / 2

    # 3D plot
    ax.plot_surface(xx2, yy2, Z, cmap=cm.coolwarm, antialiased=False)
    plt.show()
