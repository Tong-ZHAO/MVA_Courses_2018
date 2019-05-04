#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 11:41:28 2018

@author: glaunes
"""

import numpy as np
import matplotlib.pyplot as plt


#%% fonctions Ã  charger, adaptees des methodes vues au TP1

def KernelMatrix(x,y,h):
    # calcul de la matrice noyau K(x,y) de taille (m,n)
    # de coefficients K(x_i,y_j) = h(||x_i-y_j||) 
    # ou h est une fonction scalaire,
    # les x_i sont m points dans R^d, et les y_j n points dans R^d,
    # donnes sous forme de tableaux x, y de tailles (m,d) et (n,d) 
    return h(np.sqrt(np.sum((x[:,None,:]-y[None,:,:])**2,axis=2)))

def MatchingLinear(y,z,h,l=0):
    # appariement lineaire de points labellisÃ©s dans R^d
    # y et z sont des tableaux de taille (n,d) donnant les coordonnees
    # de n points y_i (points sources) 
    # et leurs correspondants z_i (points cibles)
    # h est la fonction scalaire definissant le noyau,
    # l est le parametre de relaxation,
    # retourne la transformation phi:R^d->R^d realisant l'appariement
    a = np.linalg.solve(KernelMatrix(y,y,h)+l*np.eye(y.shape[0]),z-y)
    def phi(x):
        return x+np.dot(KernelMatrix(x,y,h),a)
    return phi

def gauss(sigma) :
    # noyau de Gauss h(r)=exp(-r^2/sigma^2)
    def f(u) :
        return np.exp(-u**2/sigma**2)
    return f

def cauchy(sigma) :
    # noyau de Cauchy h(r)=1/(1+r^2/sigma^2)
    def f(u) :
        return 1/(1+u**2/sigma**2)
    return f

def MatchingTPS(y,z,l=0):
    # fonction similaire a MatchingLinear mais la methode utilisee ici 
    # est la methode Thin Plate Splines (cf TP1, questions 13 et 14)
    # remarque : la fonction noyau h(r)=r^2log(r) n'est valable en theorie
    # que pour des donnees en dimension 2.
    def TPSfun(r):
        r[r==0]=1
        return r**2 * np.log(r)
    h = TPSfun
    n,d = y.shape
    Kyy =  KernelMatrix(y,y,h) + l*np.eye(n)
    yt = np.concatenate((np.ones((n,1)),y),axis=1)
    M1 = np.concatenate((Kyy,yt),axis=1)
    M2 = np.concatenate((yt.T,np.zeros((d+1,d+1))),axis=1)
    M = np.concatenate((M1,M2))  
    c = z-y
    ct = np.concatenate((c,np.zeros((d+1,c.shape[1]))))
    a = np.linalg.solve(M,ct)
    def phi(x):
        Kxy = KernelMatrix(x,y,h)
        nx = x.shape[0]
        xt = np.concatenate((np.ones((nx,1)),x),axis=1)
        N = np.concatenate((Kxy,xt),axis=1)
        return x+np.dot(N,a)
    return phi



#%% fonctions d'affichage pour le TP2
    
def Id(x):
    return x

def PlotConfig(lmk,pts=None,clr='b',phi=Id,withgrid=True):
    if type(pts)==type(None):
        pts = lmk
    plt.axis('equal')
    if withgrid:
        # dÃ©finition d'une grille carrÃ©e adaptÃ©e aux points
        mn, mx = lmk.min(axis=0), lmk.max(axis=0)
        c, sz = (mn+mx)/2, 1.2*(mx-mn).max()
        a, b = c-sz/2, c+sz/2
        ng = 50
        X1, X2 = np.meshgrid(np.linspace(a[0],b[0],ng),np.linspace(a[1],b[1],ng))
        x = np.concatenate((X1.reshape(ng*ng,1),X2.reshape(ng*ng,1)),axis=1)
        x = phi(x)
        X1 = x[:,0].reshape(ng,ng)
        X2 = x[:,1].reshape(ng,ng)
        plt.plot(X1,X2,'k',linewidth=.25)
        plt.plot(X1.T,X2.T,'k',linewidth=.25)
    phipts = phi(pts)
    philmk = phi(lmk)
    plt.plot(phipts[:,0],phipts[:,1],'.'+clr,markersize=.1)
    plt.plot(philmk[:,0],philmk[:,1],'o'+clr)

def PlotResMatching(phi,lmk1,lmk2,pts1=None,pts2=None,withgrid=True):
    plt.figure()
    plt.subplot(1,3,1)
    PlotConfig(lmk1,pts1)
    plt.subplot(1,3,2)
    PlotConfig(lmk2,pts2,'r',withgrid=False)
    plt.subplot(1,3,3)
    PlotConfig(lmk1,pts1,phi=phi)
    plt.show()

"""
#%% Exemple d'appariement lineaire avec points aleatoires

lmk1 = np.random.rand(5,2)
lmk2 = lmk1 + .1 * np.random.randn(5,2)
sigma = .25
phi = MatchingLinear(lmk1,lmk2,gauss(sigma))
PlotResMatching(phi,lmk1,lmk2)
"""

#%% Chargement des donnees "poissons"

def load(fname='store.pckl'):
    import pickle
    f = open(fname, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj
"""
pts1,pts2,lmk1,lmk2 = load('fish.pckl')
plt.figure()
plt.axis('equal')
PlotConfig(lmk1,pts1)
plt.figure()
plt.axis('equal')
PlotConfig(lmk2,pts2)

#%% Appariement lineaire avec les donnees "poissons"
sigma = .25
phi = MatchingLinear(lmk1,lmk2,gauss(sigma))
PlotResMatching(phi,lmk1,lmk2,pts1,pts2)

#%% Appariement par la methode Thin Plate Splines
pts1,pts2,lmk1,lmk2 = load('fish.pckl')
phi = MatchingTPS(lmk1,lmk2)
PlotResMatching(phi,lmk1,lmk2,pts1,pts2)

"""



#%% Question 1

def LinTraj(y,z,nt=10):
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


# test avec une configuration aleatoire
lmk1 = np.random.rand(5, 2)
lmk2 = lmk1 + .1 * np.random.randn(5, 2)
sigma = .25
Q = LinTraj(lmk1, lmk2)
#phi = MatchingSteps(Q, gauss(sigma))
#PlotResMatching(phi, lmk1, lmk2)



#%% Test du module d'autodifferenciation de pytorch
import torch
from torch.autograd import grad

# definition d'une variable x de valeur 4, par rapport a laquelle 
# on pourra calculer des gradients:
x = torch.tensor(4.0, requires_grad=True)

# On peut aussi partir d'un vecteur numpy et le convertir en variable pytorch :
x_np = np.array(4.0)
x = torch.tensor(x_np, requires_grad=True)
 
# fonction de x qu'on va cherche a differentier :
def f(x):
    return torch.cos((x**2-2)**2)

# calcul automatique du gradient de f(x) par rapport a x :
# remarque : le [0] a la fin est necessaire car la sortie de grad est un tuple
gfx = grad(f(x), x)[0]
print("gradient auto = ", gfx.item())

# on compare avec la formule derivee a la main :
gfx_bis = 2 * (x * x-2) * (2 * x) * (-torch.sin((x ** 2-2) ** 2))
print("gradient mano = ", gfx_bis.item())


#%% Question 2

def gauss2_torch(sigma):
    # version PyTorch de la fonction gauss
    def f(u) :
        return torch.exp(-u / sigma ** 2)
    return f

def KernelMatrix_torch(x,y,h):
    # version Pytorch de la fonction KernelMatrix
    return h(torch.sum(torch.pow(x.unsqueeze(1) - y.unsqueeze(0), 2), 2))
    
def Energy(Q,h):
    # calcul de la quantite J definie a la question 2
    # Q est un tableau de taille (n,d,nt) de type torch.tensor
    # h est une fonction scalaire
    # ... a completer

    loss = 0.
    q_diff = Q[:, :, 1:] - Q[:, :, :-1] 

    for i in range(Q.shape[2] - 1):
        k_inv = torch.inverse(KernelMatrix_torch(Q[:, :, i], Q[:, :, i], h))
        loss = loss + torch.sum( q_diff[:, :, i] * (k_inv @ q_diff[:, :, i]) )

    return loss

# test de la differenciation automatique
h = gauss2_torch(sigma)
Q = torch.tensor(Q, requires_grad = True)
gradE = grad(Energy(Q, h), Q)[0]
print("gradE = ", gradE)


#... a completer : descente de gradient et test
def oracle_sgd(init_Q, h, max_iter, lr):

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


optim_Q = oracle_sgd(Q, h, max_iter = 1000, lr = 0.01) 
PlotTrajs(optim_Q, lmk1, lmk2)
plt.show()




