#!/usr/bin/env python3
# -*- coding: utf-8 -*

import numpy as np
import matplotlib.pyplot as plt

import torch
from utils_torch import *

if __name__ == "__main__":

    flag_fish = False

    if not flag_fish:
        lmk1 = np.random.rand(5, 2)
        lmk2 = lmk1 + .1 * np.random.randn(5, 2)
    else:
        _, _, lmk1, lmk2 = utils.load('fish.pckl')

    sigma = .25
    lr = 0.00001 if flag_fish else 0.01

    h = GaussKernel(sigma)
    q0 = torch.tensor(lmk1, dtype = torch.double).requires_grad_(True)
    p0 = torch.randn(lmk1.shape[0], lmk1.shape[1], dtype = torch.double).requires_grad_(True)

    p, q, Q = Shooting(p0, q0, h)

    PlotQ(Q, lmk1, lmk2)
    plt.show()