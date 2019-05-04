#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: tong zhao

from utils import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

    n = 10
    d = 2
    sigma = .25
    y = np.random.rand(n, d)    # 10 points tireÌs aleÌatoirement dans [0,1]^2
    c = np.random.randn(n, 1)   # 10 valeurs alÃ©atoires
    t = np.linspace(0, 1, 100)
    X1, X2 = np.meshgrid(t, t)  # grille uniforme de 50*50 points
    Z = InterpGrid(X1, X2, y, c, gauss(sigma))
    
    fig = plt.figure()
    plt.title("interpolation de fonction 2D")
    ax = plt.axes(projection = '3d')
    ax.plot_surface(X1, X2, Z) 
    ax.scatter3D(y[:, 0], y[:, 1], c, c = 'r', depthshade = False)
    plt.show()