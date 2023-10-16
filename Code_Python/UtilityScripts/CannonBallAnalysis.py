# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:46:24 2023

@author: JosephVermeil
"""

# %% Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import re

from scipy.optimize import curve_fit

#### Constants

mu0 = np.pi*4e-7

R = 4.490/2 # µm
Volume = (4/3)*np.pi*(R*1e-6)**3 # V volume in m^3
scale = 15.8 * 0.63 # pix/µm
freq = 50 # Hz 
visco = 1e-3 # Pa.s-1

#### Dir

dataDir = "D://MagneticPincherData//Raw//23.09.12_Cannonball"

# %% Coil tension-to-field coefficient

df_V2B = pd.read_csv(dataDir + '//' + 'Coil_V2B' + '.csv', sep = ';')

a, b = np.polyfit(df_V2B.U, df_V2B.B, 1)
xx = np.linspace(0, 5, 50)
yy = a*xx + b



df_B2gradB = pd.read_csv(dataDir + '//' + 'Coil_B2gradB' + '.csv', sep = ';')

c, d = np.polyfit(df_B2gradB.dx, df_B2gradB.B, 1)
ii = np.linspace(-4, 4, 50)
jj = c*ii + d

fig, ax = plt.subplots(1, 1)

df_V2B.plot('U', 'B', fig=fig, ax=ax,
            marker = '.', color = 'k', ls = '',
            label = 'measured')

ax.plot(xx, yy, 'r--', label = 'y = {:.3f} * x + {:.3f}'.format(a, b))
ax.set_xlabel('U (V)')
ax.set_ylabel('B (mT)')
ax.legend()



fig2, ax2 = plt.subplots(1, 1)

df_B2gradB.plot('dx', 'B', fig=fig2, ax=ax2,
            marker = '.', color = 'k', ls = '',
            label = 'measured')

ax2.plot(ii, jj, 'r--', label = 'y = {:.3f} * x + {:.3f}'.format(c, d))
ax2.set_xlabel('dx (mm)')
ax2.set_ylabel('B (mT)')
ax2.legend()

plt.tight_layout()

plt.show()

def V2B(x):
    # Volt to mT
    return(a*x + b)

def B2gB(x):
    # mT to mT/mm or T/m
    return(np.abs(c)*x/np.abs(d))


# %% Reading the data

file = 'mPEG_BSA_Results'
# file = 'M450-2025_BSA_Results'

df_CB = pd.read_csv(dataDir + '//' + file + '.csv', sep = ';') # '\t'
df_CB['B'] = V2B(df_CB.V)
df_CB['gradB'] = B2gB(df_CB.B)

df_CB['dX_pix'] = np.abs(df_CB.Length * np.cos(np.pi*df_CB.Angle/180))
df_CB['dT_pix'] = np.abs(df_CB.Length * np.sin(np.pi*df_CB.Angle/180))

df_CB['dX'] = df_CB['dX_pix'] * (1/scale) # µm
df_CB['dT'] = df_CB['dT_pix'] * (1/freq) # sec

df_CB['U'] = (1e-6 * df_CB['dX']) / df_CB['dT'] # m/s

df_CB['Fmes'] = 6 * np.pi * visco * (R*1e-6) * df_CB['U'] # N
df_CB['Fmes'] = df_CB['Fmes'] * 1e12 # pN

df_CB['M'] = df_CB['Fmes'] / (Volume * df_CB['gradB']*1e12)

fig2, ax2 = plt.subplots(1, 1)

df_CB.plot('B', 'U', fig=fig2, ax=ax2,
            marker = '.', color = 'k', ls = '',
            label = '')

ax2.set_xlabel('B (mT)')
ax2.set_ylabel('U (m/s)')
ax2.legend()

plt.tight_layout()

plt.show()

# %%

# df_CB = df_CB[df_CB['V'] != 3.5]

B0 = np.linspace(0, np.max(df_CB.B), 100)
M0 = 1600 * (0.001991*B0**3 + 17.54*B0**2 + 153.4*B0) / (B0**2 + 35.53*B0 + 158.1) 

def B2M(x, k):
    M = 1600 * k * (0.001991*x**3 + 17.54*x**2 + 153.4*x) / (x**2 + 35.53*x + 158.1)
    return(M)
    
[k_exp], covM = curve_fit(B2M, df_CB['B'], df_CB['M'])
fig, ax = plt.subplots(1, 1)

fig.suptitle(file)

ax.plot(B0, M0, 'r--', label='Litterature formula')
ax.plot(df_CB['B'], df_CB['M'], 'k+', label='Measured for M450 - 2025')
ax.plot(B0, B2M(B0, k_exp), 'g--', label='New fit with k = {:.3f}'.format(k_exp))
ax.set_xlabel('B (mT)')
ax.set_ylabel('M (SI)')
ax.legend()

plt.tight_layout()

plt.show()


# %% Reanalyse Valentin M270

mu0 = np.pi*4e-7

R = 2.690/2 # µm
Volume = (4/3)*np.pi*(R*1e-6)**3 # V volume in m^3
scale = 15.8 * 0.63 # pix/µm
freq = 50 # Hz 
visco = 1e-3 # Pa.s-1

# %%% Reading the data

dataDir = "D://MagneticPincherData//Raw//20.08.01_CanonBall_M270_VL"

# file = 'mPEG_BSA_All-V_Results'
file = 'All_24-08_KimoMin'

df_CB = pd.read_csv(dataDir + '//' + file + '.txt', sep = '\t') # '\t'
# df_CB['B'] = V2B(df_CB.V)
df_CB['gradB (mT/mm)'] = B2gB(df_CB['B (mT)'])

# df_CB['dX_pix'] = np.abs(df_CB.Length * np.cos(np.pi*df_CB.Angle/180))
# df_CB['dT_pix'] = np.abs(df_CB.Length * np.sin(np.pi*df_CB.Angle/180))

# df_CB['dX'] = df_CB['dX_pix'] * (1/scale) # µm
# df_CB['dT'] = df_CB['dT_pix'] * (1/freq) # sec

df_CB['U'] = (1e-6 * df_CB['X (µm)']) / df_CB['T (s)'] # m/s

df_CB['Fmes'] = 6 * np.pi * visco * (R*1e-6) * df_CB['U'] # N
df_CB['Fmes'] = df_CB['Fmes'] * 1e12 # pN

df_CB['M'] = df_CB['Fmes'] / (Volume * df_CB['gradB (mT/mm)']*1e12)



fig2, ax2 = plt.subplots(1, 1)

df_CB.plot('B (mT)', 'U', fig=fig2, ax=ax2,
            marker = '.', color = 'k', ls = '',
            label = '')

ax2.set_xlabel('B (mT)')
ax2.set_ylabel('U (m/s)')
ax2.legend()

plt.tight_layout()

plt.show()

# %%%

# df_CB = df_CB[df_CB['V'] != 3.5]

B0 = np.linspace(0, np.max(df_CB['B (mT)']), 100)
M0 = 1600 * (0.001991*B0**3 + 17.54*B0**2 + 153.4*B0) / (B0**2 + 35.53*B0 + 158.1) 

def B2M(x, k):
    M = 1600 * k * (0.001991*x**3 + 17.54*x**2 + 153.4*x) / (x**2 + 35.53*x + 158.1)
    return(M)
    
[k_exp], covM = curve_fit(B2M, df_CB['B (mT)'], df_CB['M'])
fig, ax = plt.subplots(1, 1)

fig.suptitle(file)

ax.plot(B0, M0, 'r--', label='Litterature formula')
ax.plot(df_CB['B (mT)'], df_CB['M'], 'k+', label='Measured for M270 - Valentin')
ax.plot(B0, B2M(B0, k_exp), 'g--', label='New fit with k = {:.3f}'.format(k_exp))
ax.set_xlabel('B (mT)')
ax.set_ylabel('M (SI)')
ax.legend()

plt.tight_layout()

plt.show()
