# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:42:49 2024

@author: JosephVermeil
"""

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

from matplotlib import ticker
import matplotlib
matplotlib.use("Qt5Cairo")

import os
import re

import UtilityFunctions as ufun
import GraphicStyles as gs


# %% Compression

gs.set_mediumText_options_jv()

def sigmoid(x, vi, vf, k):
    x2 = (2*x/np.max(x)) - 1
    y = vi + (vf-vi) * 1/(1+np.exp(-k*x2))
    return(y)

# fig, ax = plt.subplots(1, 1)
# XX = np.arange(0, 2000, 10)
# YY = sigmoid(XX, 5, 2, 6)
# ax.plot(XX, YY)

A1 = np.matmul(np.ones((1, 10)).T, [np.arange(-1,2)]).T
A2 = np.arange(50, 6050, 600)
A3 = A2 + 50*A1
TTrest1 = A3.flatten(order='F')
Tfrest1 = 6000
Brest1 = 5 * np.ones_like(TTrest1)
Zrest1 = (70 + 0.5*A1).flatten(order='F')

TTsigmo = np.arange(0, 2000, 100)
Tfsigmo = 8000
Bsigmo = sigmoid(TTsigmo, 5, 2, 6) #### CHECK LABVIEW
Zsigmo = 70 * np.ones_like(TTsigmo)

TTct = np.arange(0, 2000, 100)
Tfct = 10000
Bct = 2 * np.ones_like(TTct)
Zct = 70 * np.ones_like(TTct)

TTcomp = np.arange(0, 1500, 10)
Tfcomp = 11500
Bcomp = 2 + (55-2)*(TTcomp/np.max(TTcomp))**2
Zcomp = 70 * np.ones_like(TTcomp)

TTrelax = np.arange(0, 1500, 100)
Tfrelax = 13500
Brelax = 5 + (55-5)*((np.max(TTrelax)-TTrelax)/np.max(TTrelax))**2
Zrelax = 70 * np.ones_like(TTrelax)

TTrest2 = A3.flatten(order='F')
Tfrest2 = 19000
Brest2 = np.copy(Brest1)
Zrest2 = np.copy(Zrest1)


TTagg = np.concatenate((TTrest1, Tfrest1 + TTsigmo, Tfsigmo + TTct, Tfct + TTcomp, Tfcomp + TTrelax, Tfrelax + TTrest2))
TTagg /= 1000
Bagg = np.concatenate((Brest1, Bsigmo, Bct, Bcomp, Brelax, Brest2))
Zagg = np.concatenate((Zrest1, Zsigmo, Zct, Zcomp, Zrelax, Zrest2))

fig, axes = plt.subplots(2, 1, figsize=(17/2.54, 10/2.54), sharex=True)

ax = axes[0]
ax.plot(TTagg, Bagg, c='skyblue', ls='-', lw = 1, antialiased=True)
ax.plot(TTagg, Bagg, 'b|', ms = 5, antialiased=True)
ax.set_ylabel('B (mT)')
ax.set_ylim([0, 60])
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.grid()

ax = axes[1]
ax.plot(TTagg, Zagg, c='lightgreen', ls='-', lw = 1, antialiased=True)
ax.plot(TTagg, Zagg, 'g.', ms = 4, antialiased=True)
ax.set_ylabel('Z (µm)')
ax.set_xlabel('T (s)')
ax.set_xlim([-1, 20])
ax.set_ylim([69, 71])
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.grid()

fig.tight_layout()

path = "C:/Users/JosephVermeil/Desktop/Manuscrit/Mat&Meth/NouvellesFigures"
ufun.simpleSaveFig(fig, 'LoopStruct_BZT', path, '.png', 300)

plt.show()



# %% Constant

gs.set_mediumText_options_jv()

A1 = np.matmul(np.ones((1, 5)).T, [np.arange(-1,2)]).T
A2 = np.arange(50, 3050, 600) + 57000
A3 = A2 + 50*A1

TTrest1 = A3.flatten(order='F')
Tfrest1 = int(A3[0,-1])
Brest1 = 5 * np.ones_like(TTrest1)
Zrest1 = (70 + 0.5*A1).flatten(order='F')
ShutterRest1 = np.zeros_like(TTrest1)

TTFluo = Tfrest1 + np.array([300, 350, 350, 600, 600, 650, 750])
BFluo = 5 * np.ones_like(TTFluo)
ZFluo = np.array([70, 73.7, 73.7, 73.7, 73.7, 70, 70])
ShutterFluo = np.array([0, 0, 1, 1, 0, 0, 0])

TTimagefluo = Tfrest1 + np.array([420, 530])
Bimagefluo = [5, 5]
Zimagefluo = [73.7, 73.7]
Shutterimagefluo = [1, 1]

TTagg = np.concatenate((TTrest1, TTFluo))/1000
Bagg = np.concatenate((Brest1, BFluo))
Zagg = np.concatenate((Zrest1, ZFluo))
Shutteragg = np.concatenate((ShutterRest1, ShutterFluo))

fig, axes = plt.subplots(3, 1, figsize=((17/2.54)/1.5, 12/2.54), sharex=True)

ax = axes[0]
ax.plot(TTagg, Bagg, c='skyblue', ls='-', lw = 1.5)
ax.plot(TTrest1/1000, Brest1, 'b|', ms = 10)
ax.plot(TTimagefluo/1000, Bimagefluo, 'b-', lw=8)
ax.set_ylabel('B (mT)')
ax.set_ylim([0, 60])
# ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.grid()

ax = axes[1]
ax.plot(TTagg, Zagg, c='lightgreen', ls='-', lw = 1.5)
ax.plot(TTrest1/1000, Zrest1, 'g.', ms = 8)
ax.plot(TTimagefluo/1000, Zimagefluo, 'g-', lw=8)
ax.set_ylabel('Z (µm)')

# ax.set_xlim([-1, 20])
ax.set_ylim([69, 74.5])
# ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

ax.grid()

ax = axes[2]
ax.plot(TTagg, Shutteragg, c='red', ls='-', lw = 1.5)
ax.plot(TTrest1/1000, ShutterRest1, c='darkred', ls='', marker='|', ms = 10)
ax.plot(TTimagefluo/1000, Shutterimagefluo, c='darkred', ls='-', lw=8)
ax.set_ylabel('Shutter\nposition')
ax.set_ylim([-0.5, 1.5])
ax.set_yticks([0, 1])
ax.set_yticklabels(['TL', 'RL'])
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_xlabel('T (s)')
ax.grid()

fig.tight_layout()

path = "C:/Users/JosephVermeil/Desktop/Manuscrit/Mat&Meth/NouvellesFigures"
ufun.simpleSaveFig(fig, 'CstFieldStruct_BZTS_02', path, '.png', 300)

plt.show()