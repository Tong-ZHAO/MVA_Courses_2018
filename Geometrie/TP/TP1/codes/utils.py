#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: tong zhao

import numpy as np

def KH1(x, y, sigma = 1):
    """fonction pour definir le noyau 

    Params:
        x (np.array): of size n x d
        y (np.array): of size n x d
        sigma (float)

    Returns:
        result (np.array): K(x_i, y_i) for i in {1, ..., n}
    """
    return np.exp(-np.abs(x - y) / sigma) / (2 * sigma)


def H1ScalProd(f, g, sigma = 1):
    """fonction pour definir le produit scalaire dans l'espace H_sigma^1

    Params:
        f (func): func 1
        g (func): func 2
        sigma (float): param of the hilbert space

    Returns
        prod (float): <f,g>_H1 = \int f(t) * g(t) + sigma^2 f'(t) * g'(t) dt
    """

    t = np.linspace(-100., 100., 50000)
    tol = t[1] - t[0]

    ft, gt = f(t), g(t)
    dft = np.gradient(ft, tol)
    dgt = np.gradient(gt, tol)

    prod = ft * gt + sigma ** 2 * dft * dgt
    value = np.sum(prod) * tol

    return value


def KernelMatrix(x, y, h):
    """kernel matrix by calculating pairwise distance

    Params:
        x (np.array): of size n x d
        y (np.array): of size n x d
        h (func): the invariant kernel function

    Returns:
        dist (np.array): of size n x n, K_{ij} = K (x_i, y_j)
    """

    x = np.expand_dims(x, axis = 1)
    y = np.expand_dims(y, axis = 0)

    return h(np.linalg.norm(x - y, axis = -1))


def Interp(x, y, c, h):
    """résoudre le problème d'interpolation f(y_i) = c_i, évaluer aux point x_i

    Params:
        x (np.array): of size n1 x d, a linear space
        y (np.array): of size n2 x d
        c (np.array): of size n2 x 1
        h (func): kernel function

    Returns:
        result (np.array): of size n1 x d, interpolation result
    """

    Kh_yy = KernelMatrix(y, y, h)
    A = np.linalg.solve(Kh_yy, c)

    Kh_xy = KernelMatrix(x, y, h)
    return Kh_xy.dot(A)


def cauchy(sigma):
    """cauchy kernel, 1 / (1 + u^2 / sigma ^2)

    Params:
        sigma (float): param of kernel

    Returns:
        func: kernel function
    """
    def f(u):
        return 1. / (1 + u ** 2 / sigma ** 2)
    return f


def gauss(sigma):
    """gauss kernel, e^{- u^2 / sigma^2}

    Params:
        sigma (float): param of kernel

    Returns:
        func: kernel function
    """
    def f(u):
        return np.exp(-u ** 2 / sigma ** 2)
    return f


def hkernel(sigma):
    """h1 kernel, (1 / 2sigma) e^{-|u| / sigma}

    Params:
        sigma (float): param of kernel

    Returns: 
        func: kernel function
    """
    def f(u):
        return np.exp(-np.abs(u) / sigma) / (2 * sigma)
    return f 


def norm_h(f, sigma):
    """norme de la fonction dans l'espace H_sigma^1

    Params:
        f (func): function
        sigma (float): param of hilbert space

    Returns:
        norm (float): \sqrt(<f, f>_H)
    """
    return np.sqrt(H1ScalProd(f, f, sigma))


def func_kernel(y, c, h):
    """renvoyer une fonction de noyau

    Params:
        y (np.array): of size n x d
        c (np.array): of size n x 1
        h (func): kernel function

    Returns:
        func: a function of x
    """
    def f(u):
        return np.squeeze(Interp(u[:, np.newaxis], y, c, h))
    return f


def InterpGrid(X1, X2, y, c, h) :
    """renvoyer l'evaluation de l'interpolant a noyau sur les points de la grille X1, X2

    Params:
        X1 (np.array): of size n1 x n2
        X2 (np.array): of size n1 x n2
        y (np.array): of size n x 2
        c (np.array): of size n x 1
        h (np.array): kernel function

    Returns:
        result (np.array): of size n1 x n2, interpolation result on grid (X1, X2)
    """

    Kh_yy = KernelMatrix(y, y, h)
    A = np.linalg.solve(Kh_yy, c)

    Kh_xy = KernelMatrix(np.stack((X1, X2), axis = -1).reshape((-1, 2)), y, h)
    
    return Kh_xy.dot(A).reshape(X1.shape)


def Interp_inexact(x, y, c, h, l = 0):
    """revoyer l'evaluation de l'interpolation inexacte sur x

    Params:
        x (np.array): of size n1 x 1
        y (np.array): of size n2 x 1
        c (np.array): of size n2 x 1
        h (func): kernel function
        l (float): the coeff of regularization term

    Returns:
        result (np.array): of size n1 x 1, evaluations on x
    """
    
    Kh_yy = KernelMatrix(y, y, h) + l * np.eye(y.shape[0])
    A = np.linalg.solve(Kh_yy, c)

    Kh_xy = KernelMatrix(x, y, h)
    return Kh_xy.dot(A)


def InterpGrid2D(X1, X2, y, c, h, l = 0) :
    """modification de InterpGrid devant renvoyer un interpolant vectoriel et non scalaire

    Params:
        X1 (np.array): of size n1 x n2
        X2 (np.array): of size n1 x n2
        y (np.array): of size n x 2
        c (np.array): of size n x 1
        h (func): kernel function
        l (float): the coeff of regularization term

    Returns:
        V1 (np.array): interpolation result on X1
        V2 (np.array): interpolation result on X2
    """

    Kh_yy = KernelMatrix(y, y, h) + l * np.eye(y.shape[0])
    A = np.linalg.solve(Kh_yy, c)

    Kh_xy = KernelMatrix(np.stack((X1, X2), axis = -1).reshape((-1, 2)), y, h)
    V =  Kh_xy.dot(A).reshape((X1.shape[0], X1.shape[1], 2))   

    return V[:, :, 0], V[:, :, 1] 