#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: tong zhao

from utils import KH1, H1ScalProd
import numpy as np

if __name__ == "__main__":
    
    sigma = 2
    x = 0.5
    f = lambda t: np.exp(- t ** 2)
    g = lambda t: KH1(t, x, sigma)

    fx = f(x)
    gx = H1ScalProd(f, g, sigma)

    print("x = %f, fx = %f (gauss)" % (x, fx))
    print("x = %f, gx = %f (Q1)" % (x, gx))