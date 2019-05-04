#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of diffeomorphic matchings of landmarks, curves and surfaces
We perform LDDMM matching using the geodesic shooting algorithm
"""

import numpy as np

import matplotlib.pyplot as plt
import torch

from utils import *


if __name__ == "__main__":

    x = torch.load('synth_20.pt')
    q0 = x[0].clone().detach().requires_grad_(True) # initialization, barycenter may be better

    Kv = GaussKernel(sigma = 0.25)

    Dataloss = [losslmk(elem) for elem in x]
    loss = AtlasLDDMMloss(Kv, Dataloss)

    # initialize momentum vectors
    p0 = [torch.zeros(q0.shape, requires_grad = True) for _ in range(len(x))]

    # perform optimization
    optimizer = torch.optim.LBFGS(p0 + [q0])
    Niter = 5

    print('performing optimization...')
    for _ in range(Niter):
        def closure():
            optimizer.zero_grad()
            L = loss(p0, q0)
            L.backward()
            return L
        optimizer.step(closure)

    for i in range(len(x)):

        plotfun = PlotRes2D(x[i])
        plotfun(q0, p0[i], Kv)
        #plt.savefig(str(i) + '.png')

    print("Done.")
    plt.show(block = True)

