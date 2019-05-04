#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: tong zhao

from utils import *
import numpy as np

if __name__ == "__main__":
    
    x = np.random.rand(10,2)    # 10 points tireÌs aleÌatoirement dans [0,1]^2
    y = np.random.rand(15,2)    # 15 points tireÌs aleÌatoirement dans [0,1]^2
    sigma = 1.5                 # echelle du noyau
    Kxy = KernelMatrix(x,y,gauss(sigma))
    print(Kxy)