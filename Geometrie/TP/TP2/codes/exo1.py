#!/usr/bin/env python3
# -*- coding: utf-8 -*

import numpy as np
import matplotlib.pyplot as plt

from utils import *

def LinTraj(y, z, nt = 10):
    # renvoie Q tableau de taille (n,d,nt) donnant les coordonnees de points
    # q_i^t de R^d regulierement espaces le long des segments [y_i,z_i]
    # (trajectoires lineaires entre ces points)
    Q = np.dot(y[:, :, None], np.linspace(1, 0, nt)[None, :]) + \
        np.dot(z[:, :, None], np.linspace(0, 1, nt)[None, :])

    return Q


def MatchingSteps(Q, h):
    # appariement de points labellises par composition d'appariement lineaires
    # Q est un tableau de taille (n,d,nt) donnant les coordonnees des points
    # q_i^t de R^d, pour 1 <= i <= n et 1 <= t <= nt
    # ... a completer
    lphi = [MatchingLinear(Q[:, :, i], Q[: ,:, i + 1], h) for i in range(Q.shape[-1] - 1)]
    def my_phi(x):
        phi_x = x
        for phi in lphi:
            phi_x = phi(phi_x)
        return phi_x
    
    return my_phi


if __name__ == "__main__":

    lmk1 = np.random.rand(5, 2)
    lmk2 = lmk1 + .1 * np.random.randn(5, 2)
    sigma = .25

    lnt = [3, 5, 10]

    for nt in lnt:
        Q = LinTraj(lmk1, lmk2, nt)
        phi = MatchingSteps(Q, gauss(sigma))
        PlotResMatching(phi, lmk1, lmk2, title = "nt = %d" % nt)