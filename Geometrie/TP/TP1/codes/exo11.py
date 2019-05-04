#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: tong zhao

from utils import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    n = 3
    d = 2
    m = 2
    sigma = .5
    y = np.random.rand(n, d)
    z = np.random.rand(n, d)
    gamma = z - y
    t = np.linspace(0, 1, 20)
    X1, X2 = np.meshgrid(t, t)
    V1, V2 = InterpGrid2D(X1, X2, y, gamma, gauss(sigma))
    plt.figure()
    plt.title("appraiement, sigma=" + str(sigma))
    plt.scatter(y[:, 0], y[:, 1], label = 'y')
    plt.scatter(z[:, 0], z[:, 1], label = 'z')
    plt.quiver(y[:, 0], y[:, 1], gamma[:, 0], gamma[:, 1])
    plt.quiver(X1, X2, V1, V2, color = 'b')
    plt.legend()
    plt.show()