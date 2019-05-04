#!/usr/bin/env python3
# -*- coding: utf-8 -*

import numpy as np
import matplotlib.pyplot as plt

import torch
from utils_torch import *
import utils


if __name__ == "__main__":

    flag_fish = True

    if not flag_fish:
        lmk1 = np.random.rand(5, 2)
        lmk2 = lmk1 + .1 * np.random.randn(5, 2)
    else:
        _, _, lmk1, lmk2 = utils.load('fish.pckl')

    sigma = .25
    

    h = GaussKernel(sigma)
    q0 = torch.tensor(lmk1, dtype = torch.double).requires_grad_(True)
    p0 = torch.randn(lmk1.shape[0], lmk1.shape[1], dtype = torch.double).requires_grad_(True)
    z = torch.tensor(lmk2)

    flag_lbfgs = True

    if not flag_lbfgs:
        lr = 0.001 if flag_fish else 0.05
        # simple gradient descent
        for i in range(1000):
            J = loss(p0, q0, z, h)
            GJ = grad(J, p0)[0]
            p0 = p0 -  lr * GJ
    else:
        lr = 0.005 if flag_fish else 0.05
        # lbfgs gradient descent 
        optimizer = torch.optim.LBFGS([p0])
        Niter = 50
        print('performing optimization...')
        for _ in range(Niter):
            def closure():
                optimizer.zero_grad()
                #L = loss(p0, q0)
                L = loss(p0, q0, z, h)
                L.backward()
                return L
            optimizer.step(closure)

    p, q, Q = Shooting(p0, q0, h)

    PlotQ(Q, lmk1, lmk2)
    plt.show()
