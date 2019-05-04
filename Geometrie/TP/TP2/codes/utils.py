#!/usr/bin/env python3
# -*- coding: utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
import pickle


def KernelMatrix(x, y, h):
    # calcul de la matrice noyau K(x,y) de taille (m,n)
    # de coefficients K(x_i,y_j) = h(||x_i-y_j||) 
    # ou h est une fonction scalaire,
    # les x_i sont m points dans R^d, et les y_j n points dans R^d,
    # donnes sous forme de tableaux x, y de tailles (m,d) et (n,d) 
    return h(np.sqrt(np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis = 2)))


def MatchingLinear(y, z, h, l = 0):
    # appariement lineaire de points labellisÃ©s dans R^d
    # y et z sont des tableaux de taille (n,d) donnant les coordonnees
    # de n points y_i (points sources) 
    # et leurs correspondants z_i (points cibles)
    # h est la fonction scalaire definissant le noyau,
    # l est le parametre de relaxation,
    # retourne la transformation phi:R^d->R^d realisant l'appariement
    a = np.linalg.solve(KernelMatrix(y, y, h) + l * np.eye(y.shape[0]), z - y)
    def phi(x):
        return x + np.dot(KernelMatrix(x, y, h), a)
    return phi


def gauss(sigma) :
    # noyau de Gauss h(r)=exp(-r^2/sigma^2)
    def f(u) :
        return np.exp(-u ** 2 / sigma ** 2)
    return f


def cauchy(sigma) :
    # noyau de Cauchy h(r)=1/(1+r^2/sigma^2)
    def f(u) :
        return 1 / (1 + u ** 2 / sigma ** 2)
    return f


def MatchingTPS(y, z, l = 0):
    # fonction similaire a MatchingLinear mais la methode utilisee ici 
    # est la methode Thin Plate Splines (cf TP1, questions 13 et 14)
    # remarque : la fonction noyau h(r)=r^2log(r) n'est valable en theorie
    # que pour des donnees en dimension 2.
    def TPSfun(r):
        r[r == 0] = 1
        return r ** 2 * np.log(r)
    h = TPSfun
    n, d = y.shape
    Kyy =  KernelMatrix(y, y, h) + l * np.eye(n)
    yt = np.concatenate((np.ones((n, 1)), y), axis = 1)
    M1 = np.concatenate((Kyy, yt), axis = 1)
    M2 = np.concatenate((yt.T, np.zeros((d + 1, d + 1))), axis = 1)
    M = np.concatenate((M1, M2))  
    c = z - y
    ct = np.concatenate((c, np.zeros((d + 1, c.shape[1]))))
    a = np.linalg.solve(M, ct)
    def phi(x):
        Kxy = KernelMatrix(x, y, h)
        nx = x.shape[0]
        xt = np.concatenate((np.ones((nx, 1)), x), axis = 1)
        N = np.concatenate((Kxy, xt), axis = 1)
        return x + np.dot(N, a)
    return phi


def Id(x):
    return x


def PlotConfig(lmk, pts = None, clr = 'b', phi = Id, withgrid = True):
    if type(pts) == type(None):
        pts = lmk
    plt.axis('equal')
    if withgrid:
        # definition d'une grille carree adaptee aux points
        mn, mx = lmk.min(axis = 0), lmk.max(axis = 0)
        c, sz = (mn + mx) / 2, 1.2 * (mx - mn).max()
        a, b = c - sz / 2, c + sz / 2
        ng = 50
        X1, X2 = np.meshgrid(np.linspace(a[0], b[0], ng), np.linspace(a[1], b[1], ng))
        x = np.concatenate((X1.reshape(ng * ng, 1), X2.reshape(ng * ng, 1)), axis = 1)
        x = phi(x)
        X1 = x[:, 0].reshape(ng, ng)
        X2 = x[:, 1].reshape(ng, ng)
        plt.plot(X1, X2, 'k', linewidth = .25)
        plt.plot(X1.T, X2.T, 'k', linewidth = .25)
    phipts = phi(pts)
    philmk = phi(lmk)
    plt.plot(phipts[:, 0], phipts[:, 1], '.' + clr, markersize = .1)
    plt.plot(philmk[:, 0], philmk[:, 1], 'o' + clr)


def PlotResMatching(phi, lmk1, lmk2, pts1 = None, pts2 = None, withgrid = True, title = None):
    plt.figure()
    plt.subplot(1, 3, 1)
    PlotConfig(lmk1, pts1)
    plt.subplot(1, 3, 2)
    PlotConfig(lmk2, pts2, 'r', withgrid = False)
    plt.subplot(1, 3, 3)
    PlotConfig(lmk1, pts1, phi = phi)
    if title is not None:
        plt.suptitle(title)
    plt.show()


def load(fname = 'store.pckl'):
    f = open(fname, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

