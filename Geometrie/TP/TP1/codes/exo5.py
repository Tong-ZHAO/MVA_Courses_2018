#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: tong zhao

from utils import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    lsigma = [0.2, 2, 4, 10]
    y = np.array([0, 2, 4, 8])[:, np.newaxis]     # vecteur [0,2,4,8] sous forme de tableau 4x1
    c = np.random.randn(4, 1)            # valeurs tirees aleatoirement
    x = np.linspace(-2, 10, 1000)[:, np.newaxis] # points xj ou evaluer l'interpolation

    plt.figure()

    for i, sigma in enumerate(lsigma):

        fx = Interp(x, y, c, gauss(sigma))     # calcul de l'interpolation
        plt.subplot(2, 2, i + 1)
        plt.plot(y, c, 'o')                   # affichage
        plt.plot(x, fx)
        plt.title("gauss, sigma = % f" % sigma)


    plt.figure()

    for i, sigma in enumerate(lsigma):

        fx = Interp(x, y, c, cauchy(sigma))     # calcul de l'interpolation
        plt.subplot(2, 2, i + 1)
        plt.plot(y, c, 'o')                   # affichage
        plt.plot(x, fx)
        plt.title("cauchy, sigma = % f" % sigma)

    plt.show()