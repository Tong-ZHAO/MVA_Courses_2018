#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: tong zhao

from utils import *
import numpy as np
import matplotlib.pyplot as plt
import pickle


def save(obj,fname = 'store.pckl'):
    f = open(fname, 'wb')
    pickle.dump(obj, f)
    f.close()


def load(fname = 'store.pckl'):
    f = open(fname, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


def onpick1(event):
    # get landmarks for hand1
    if len(lmk1) < 35:
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind[0]
        point = tuple((xdata[ind], ydata[ind]))
        lmk1.append(point)
        print('onpick point:', point)
    else:
        print('already have {} points'.format(len(lmk1)))
        print(lmk1)


def onpick2(event):
    # get landmarks for hand2
    if len(lmk2) < 35:
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind[0]
        point = tuple((xdata[ind], ydata[ind]))
        lmk2.append(point)
        print('onpick point:', point)
    else:
        print('already have {} points'.format(len(lmk2)))
        print(lmk2)


if __name__ == "__main__":

    fname = "../try.pckl"
    lmk1, lmk2 = [], []
    if False:
        import scipy.io
        C1 = scipy.io.loadmat('../hand1.mat')['C1']
        C2 = scipy.io.loadmat('../hand2.mat')['C2']
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.set_title('hand1')
        ax1.plot(C1[:,0], C1[:,1], picker=1) 
        fig1.canvas.mpl_connect('pick_event', onpick1)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.set_title('hand2')
        ax2.plot(C2[:,0], C2[:,1], picker=1) 
        fig2.canvas.mpl_connect('pick_event', onpick2)
        plt.show()
        lmk1 = np.array(lmk1)
        lmk2 = np.array(lmk2)
        save((C1, C2, lmk1, lmk2), fname)
    else:
        C1, C2, lmk1, lmk2 = load("../hands_more.pckl")

    y = lmk1
    z = lmk2
    gamma = lmk2 - lmk1
    x = C1
    phix = x + Interp_inexact(x, y, gamma, hkernel(sigma = 0.8), l = 0.1)
    plt.figure()
    plt.plot(y[:, 0], y[:, 1], 'ob', label='y')
    plt.plot(z[:, 0], z[:, 1], 'xr', label = 'z')
    plt.plot(x[:, 0], x[:, 1], 'b', label = 'x')
    plt.plot(C2[:, 0], C2[:, 1], 'r', label = 'C2')
    plt.plot(phix[:, 0], phix[:, 1], 'g', label = 'C1+u')
    plt.legend()
    plt.show()