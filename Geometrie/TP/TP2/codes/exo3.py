#!/usr/bin/env python3
# -*- coding: utf-8 -*

import numpy as np
import matplotlib.pyplot as plt

from utils_torch import *
import exo1
import utils

import torch
from torch.autograd import grad


if __name__ == "__main__":

    flag_fish = False

    if not flag_fish:
        lmk1 = np.random.rand(5, 2)
        lmk2 = lmk1 + .1 * np.random.randn(5, 2)
    else:
        _, _, lmk1, lmk2 = utils.load('fish.pckl')

    sigma = .25
    lr = 0.00001 if flag_fish else 0.01

    h = gauss(sigma)
    Q = exo1.LinTraj(lmk1, lmk2, 5)
    Q = torch.tensor(Q, requires_grad = True)

    optim_Q = oracle_sgd(Q, h, max_iter = 2000, lr = lr) 
    PlotTrajs(optim_Q, lmk1, lmk2)
    plt.show()