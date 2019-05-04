#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: tong zhao

from utils import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # tests pour differentes valeurs de sigma et de lambda
    n = 30
    lsigma = [0.1, 0.25]
    #llambda = [0, 0.1, 0.001, 0.0001]
    lambd = 0.001
    y = np.linspace(0, 1, n)[:, np.newaxis]
    x = np.linspace(-0.1, 1.1, 1000)[:, np.newaxis]
    c = np.cos(6 * y) + .05 * np.random.randn(n, 1)

    plt.plot()

    for i, sigma in enumerate(lsigma):
        fx = Interp_inexact(x, y, c, gauss(sigma), lambd) 
        plt.subplot(1, 2 , i + 1)
        plt.plot(y, c, 'o')                 
        plt.plot(x, fx)
        plt.title("sigma = %f" % sigma)

    plt.show()

    sigma = 0.1
    llambda = [0, 0.0001, 0.001, 0.1]

    plt.plot()

    for i, lambd in enumerate(llambda):
        fx = Interp_inexact(x, y, c, gauss(sigma), lambd) 
        plt.subplot(2, 2 , i + 1)
        plt.plot(y, c, 'o')                 
        plt.plot(x, fx)
        plt.title("lambda = %f" % lambd)

    plt.show()