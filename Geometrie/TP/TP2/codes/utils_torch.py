#!/usr/bin/env python3
# -*- coding: utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad


def gauss(sigma):
    # version PyTorch de la fonction gauss
    def f(u) :
        return torch.exp(-u / sigma ** 2)
    return f

def KernelMatrix(x, y, h):
    # version Pytorch de la fonction KernelMatrix
    return h(torch.sum(torch.pow(x.unsqueeze(1) - y.unsqueeze(0), 2), 2))
    
def Energy(Q, h):
    # calcul de la quantite J definie a la question 2
    # Q est un tableau de taille (n,d,nt) de type torch.tensor
    # h est une fonction scalaire
    # ... a completer

    loss = 0.
    q_diff = Q[:, :, 1:] - Q[:, :, :-1] 

    for i in range(Q.shape[2] - 1):
        k_inv = torch.inverse(KernelMatrix(Q[:, :, i], Q[:, :, i], h))
        loss = loss + torch.sum(q_diff[:, :, i] * (k_inv @ q_diff[:, :, i]))

    return loss


def oracle_sgd(init_Q, h, max_iter, lr):
    # descent de gradient 
    Q = init_Q

    for i in range(max_iter):
        J = Energy(Q, h)
        GJ = grad(J, Q)[0]
        Q[:, :, 1:-1] -= lr * GJ[:, :, 1:-1]

    return Q


# fonction utile pour l'affichage des trajectoires q_i^t
def PlotTrajs(Q, lmk1, lmk2):
    # affichage des trajectoires q_i^t
    plt.plot(Q[:,0,:].detach().numpy().T,Q[:,1,:].detach().numpy().T,'y')
    plt.scatter(lmk1[:, 0], lmk1[:, 1], label = "lmk1")
    plt.scatter(lmk2[:, 0], lmk2[:, 1], label = "lmk2")
    plt.legend()


# define Gaussian kernel (K(x,y)b)_i = sum_j exp(-|xi-yj|^2)bj
def GaussKernel(sigma):
    coeff = 1 / sigma ** 2
    def K(x,y,b):
        return torch.exp(-coeff * torch.sum((x[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2, dim = 2)) @ b
    return K


def Hamiltonian(p, q, K):
    return 0.5 * (p * K(q, q, p)).sum()


def HamiltonianSys(p, q, K):
    H = Hamiltonian(p, q, K)
    Gp, Gq = grad(H, (p, q), create_graph = True)
    return -Gq, Gp


def Shooting(p0, q0, K, nt = 5):

    Q = [[p0.detach().numpy(), q0.detach().numpy()]]
    rate = 1. / nt
    p, q = p0, q0
    
    for i in range(nt):

        grad_p, grad_q = HamiltonianSys(p, q, K)
        p = p + grad_p * rate
        q = q + grad_q * rate

        Q.append([p.detach().numpy(), q.detach().numpy()])

    return p, q, Q
    

def loss(p0, q0, z, K):

    q = Shooting(p0, q0, K)[1]
    
    return ((q - z) ** 2).sum()


def PlotQ(Q, lmk1, lmk2):
    # affichage des trajectoires q_i^t
    qs = np.array([item[1] for item in Q]) 
    plt.plot(qs[:, :, 0], qs[:, :, 1], 'y')
    plt.scatter(lmk1[:, 0], lmk1[:, 1], label = "lmk1")
    plt.scatter(lmk2[:, 0], lmk2[:, 1], label = "lmk2")
    plt.legend()







