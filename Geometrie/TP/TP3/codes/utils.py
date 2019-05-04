#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of diffeomorphic matchings of landmarks, curves and surfaces
We perform LDDMM matching using the geodesic shooting algorithm
"""

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.autograd import grad

# define Gaussian kernel (K(x,y)b)_i = sum_j exp(-|xi-yj|^2)bj
def GaussKernel(sigma):
    oos2 = 1 / sigma ** 2
    def K(x,y,b):
        return torch.exp(-oos2 * torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim = 2)) @ b
    return K

# custom ODE solver, for ODE systems which are defined on tuples
def RalstonIntegrator(nt = 10):
    def f(ODESystem, x0, deltat = 1.0):
        x = tuple(map(lambda x: x.clone(), x0))
        dt = deltat / nt
        for i in range(nt):
            xdot = ODESystem(*x)
            xi = tuple(map(lambda x, xdot: x + (2 * dt / 3) * xdot, x, xdot))
            xdoti = ODESystem(*xi)
            x = tuple(map(lambda x, xdot, xdoti: x + (.25 * dt) * (xdot + 3 * xdoti), x, xdot, xdoti))
        return x
    return f

# LDDMM implementation
def Hamiltonian(K):
    def H(p,q):
        return .5 * (p * K(q, q, p)).sum()
    return H

def HamiltonianSystem(K):
    H = Hamiltonian(K)
    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph = True)
        return -Gq, Gp
    return HS

def Shooting(p0, q0, K, deltat = 1.0, Integrator = RalstonIntegrator()):
    return Integrator(HamiltonianSystem(K), (p0, q0), deltat)

def Flow(x0, p0, q0, K, deltat = 1.0, Integrator = RalstonIntegrator()):
    HS = HamiltonianSystem(K)
    def FlowEq(x, p, q):
        return (K(x, q, p),) + HS(p, q)
    return Integrator(FlowEq, (x0, p0, q0), deltat)[0]

def LDDMMloss(K, dataloss, gamma = 0):
    def loss(p0, q0):
        p, q = Shooting(p0, q0, K)
        return gamma * Hamiltonian(K)(p0, q0) + dataloss(q)
    return loss

def losslmk(z):
    def loss(q):
        return ((q-z) ** 2).sum()
    return loss
    
def lossmeas(z, Kw):
    nz = z.shape[0]
    wz = torch.ones(nz, 1)
    cst = (1 / nz ** 2) * Kw(z, z, wz).sum()
    def loss(q):
        nq = q.shape[0]
        wq = torch.ones(nq, 1)
        return cst + (1 / nq ** 2) * Kw(q, q, wq).sum() + (-2 / (nq * nz)) * Kw(q, z, wz).sum()
    return loss

# define "Gaussian-CauchyBinet" kernel (K(x,y,u,v)b)_i = sum_j exp(-|xi-yj|^2) <ui,vj>^2 bj
def GaussLinKernel(sigma, lib = "keops"):
    oos2 = 1 / sigma ** 2
    def K(x, y, u, v, b):
        Kxy = torch.exp(-oos2 * torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim = 2))
        Sxy = torch.sum(u[:, None, :] * v[None, :, :], dim = 2) ** 2
        return (Kxy*Sxy)@b
    return K

# Varifold data attachment loss for surfaces
# VT: vertices coordinates of target surface, 
# FS,FT : Face connectivity of source and target surfaces
# K kernel
def lossVarifoldSurf(FS,VT,FT,K):
    def CompCLNn(F,V):
        V0, V1, V2 = V.index_select(0,F[:,0]), V.index_select(0,F[:,1]), V.index_select(0,F[:,2])
        C, N = .5*(V0+V1+V2), .5*torch.cross(V1-V0,V2-V0)
        L = (N**2).sum(dim=1)[:,None].sqrt()
        return C,L,N/L
    CT,LT,NTn = CompCLNn(FT,VT)
    cst = (LT*K(CT,CT,NTn,NTn,LT)).sum()
    def loss(VS):
        CS,LS,NSn = CompCLNn(FS,VS)
        return cst + (LS*K(CS,CS,NSn,NSn,LS)).sum() - 2*(LS*K(CS,CT,NSn,NTn,LT)).sum()
    return loss

def AtlasLDDMMloss(K, dataloss, gamma = 0):
    def loss(p0, q0):
        my_loss = []
        for i in range(len(p0)):
            p, q = Shooting(p0[i], q0, K)
            my_loss.append(gamma * Hamiltonian(K)(p0[i], q0) + dataloss[i](q))
        return sum(my_loss)
    return loss

def AtlasLDDMMloss2(K, dataloss, gamma = 0):
    def loss(p0, c, xbar):
        my_loss = []
        for i in range(len(p0)):
            phixbar = Flow(xbar, p0[i], c, Kv)
            my_loss.append(gamma * Hamiltonian(K)(p0[i], c) + dataloss[i](q))
        return sum(my_loss)
    return loss

def PlotRes2D(z,pts=None):
    def plotfun(q0,p0,Kv):
        plt.figure()
        plt.title('LDDMM matching example')  
        p,q = Shooting(p0,q0,Kv)
        q0np, qnp = q0.data.numpy(), q.data.numpy()
        q0np, qnp, znp = q0.data.numpy(), q.data.numpy(), z.data.numpy()
        plt.plot(q0np[:,0],q0np[:,1],'o')
        plt.plot(qnp[:,0],qnp[:,1],'+')
        plt.plot(znp[:,0],znp[:,1],'x')
        plt.axis('equal')
        ng = 50
        lsp = np.linspace(0,1,ng,dtype=np.float32)
        X1, X2 = np.meshgrid(lsp,lsp)
        x = np.concatenate((X1.reshape(ng**2,1),X2.reshape(ng**2,1)),axis=1)
        phix = Flow(torch.from_numpy(x),p0,q0,Kv).detach().numpy()
        X1 = phix[:,0].reshape(ng,ng)
        X2 = phix[:,1].reshape(ng,ng)
        plt.plot(X1,X2,'k',linewidth=.25)
        plt.plot(X1.T,X2.T,'k',linewidth=.25); 
        n,d = q0.shape
        nt = 20
        Q = np.zeros((n,d,nt))
        for i in range(nt):
            t = (i-1)/(nt-1)
            Q[:,:,i] = Shooting(t*p0,q0,Kv)[1].data.numpy()
        plt.plot(Q[:,0,:].T,Q[:,1,:].T,'y')
        if type(pts)!=type(None):
            phipts = Flow(pts,p0,q0,Kv).data
            plt.plot(phipts.numpy()[:,0],phipts.numpy()[:,1],'.b',markersize=.1)
    return plotfun

def PlotResSurf(VS,FS,VT,FT):
    def plotfun(q0,p0,Kv):
        fig = plt.figure()
        plt.title('LDDMM matching example')  
        p,q = Shooting(p0,q0,Kv)
        q0np, qnp = q0.data.numpy(), q.data.numpy()
        FSnp,VTnp, FTnp = FS.data.numpy(),  VT.data.numpy(), FT.data.numpy()    
        ax = Axes3D(fig)
        ax.axis('equal')
        ax.plot_trisurf(q0np[:,0],q0np[:,1],q0np[:,2],triangles=FSnp,alpha=.5)
        ax.plot_trisurf(qnp[:,0],qnp[:,1],qnp[:,2],triangles=FSnp,alpha=.5)
        ax.plot_trisurf(VTnp[:,0],VTnp[:,1],VTnp[:,2],triangles=FTnp,alpha=.5)
    return plotfun