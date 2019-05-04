#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: tong zhao

from utils import *
import numpy as np

if __name__ == "__main__":

    sigma = 1.5
    y = np.array([0, 2, 4, 8])[:, np.newaxis]     # vecteur [0,2,4,8] sous forme de tableau 4x1
    c = np.random.randn(4, 1)                     # valeurs tirees aleatoirement

    fg_kernel = func_kernel(y, c, gauss(sigma))
    fh_kernel = func_kernel(y, c, hkernel(sigma))
    fc_kernel = func_kernel(y, c, cauchy(sigma))

    print("Norm of h1 kernel: %f" % norm_h(fh_kernel, sigma))
    print("Norm of gauss kernel: %f" % norm_h(fg_kernel, sigma))
    print("Norm of cauchy kernel: %f" % norm_h(fc_kernel, sigma))