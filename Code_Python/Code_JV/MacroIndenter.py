# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:22:07 2023

@author: JosephVermeil
"""

# %% Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import re

from scipy.optimize import curve_fit
from scipy.signal import find_peaks, argrelextrema

import UtilityFunctions as ufun
import GraphicStyles as gs

gs.set_default_options_jv()

# %% Imports

dataDir = "D://MagneticPincherData//Raw//23.10.10_MacroIndenter_VCI"

D1, D2, D3 = 1.401, 2.73, 4.004
R1, R2, R3 = D1/2, D2/2, D3/2
f1, f2, f3 = 'Specimen_RawData_2.csv', 'Specimen_RawData_4.csv', 'Specimen_RawData_5.csv'
rD = {'Temps' : 'T', 'Deplacement' : 'Z', 'Charge' : 'F'}
df1 = pd.read_csv(dataDir + '//' + f1, engine = 'python', sep = ';', skiprows=[1]).rename(columns = rD)
df2 = pd.read_csv(dataDir + '//' + f2, engine = 'python', sep = ';', skiprows=[1]).rename(columns = rD)
df3 = pd.read_csv(dataDir + '//' + f3, engine = 'python', sep = ';', skiprows=[1]).rename(columns = rD)
list_df = [df1, df2, df3]
list_R = [R1, R2, R3]

# Reduced Young Modulus : Y = E / (1 - v**2)
# F = (4/3) * Y * R**0.5 * h**1.5


# %% First one

dataDir = "D://MagneticPincherData//Raw//23.10.10_MacroIndenter_VCI"

D = 1.401
R = D/2
f = 'Specimen_RawData_2.csv'
rD = {'Temps' : 'T', 'Deplacement' : 'Z', 'Charge' : 'F'}
df = pd.read_csv(dataDir + '//' + f, engine = 'python', sep = ';', skiprows=[1]).rename(columns = rD)

early_points = 30

dict_curves = {}

imax = find_peaks(df['Z'], distance=150)[0]
imin = find_peaks((-1) * df['Z'], distance=150)[0]
imin = [0, imin[0], len(df)]

Ncycles = len(imax)
fig, axes = plt.subplots(2, Ncycles, figsize = (8, 6))
fig.suptitle('Curves for D = {:.3f} mm'.format(D))

for j in range(len(imax)):
    dict_curves[2*j] = {}
    dict_curves[2*j]['type'] = 'comp'
    dict_curves[2*j]['num'] = j+1
    dict_curves[2*j]['df'] = df.iloc[imin[j]:imax[j], :]
    
    dict_curves[2*j + 1] = {}
    dict_curves[2*j + 1]['type'] = 'relax'
    dict_curves[2*j + 1]['num'] = j+1
    dict_curves[2*j + 1]['df'] = df.iloc[imax[j]:imin[j+1], :]
    
for i in range(len(dict_curves)):
    i_ax = int(dict_curves[i]['type'] == 'relax')
    j_ax = i//2
    try:
        ax = axes[i_ax, j_ax]
    except:
        ax = axes[i_ax]
        
    ax.set_title(dict_curves[i]['type'] + ' ' + str(dict_curves[i]['num']))
    if dict_curves[i]['type'] == 'comp':
        F = dict_curves[i]['df']['F'].values
        Z = dict_curves[i]['df']['Z'].values
    elif dict_curves[i]['type'] == 'relax':
        F = dict_curves[i]['df']['F'].values[::-1]
        Z = dict_curves[i]['df']['Z'].values[::-1]
    
    F_moy = np.average(F[:early_points])
    F_std = np.std(F[:early_points])
    
    #
    F = F - F_moy
    F_moy = 0
    #
    
    ax.plot(Z, F)
    ax.set_xlabel('Z (mm)')
    ax.set_ylabel('F (N)')
    ax.grid(axis='y')
    
    #` Function
    
    def HertzFit(z, K, Z0):
        f = (4/3) * K * R**0.5 * (z - Z0)**1.5
        return(f)
    
    #### First Fit
    
    i_start = len(F) - ufun.findFirst(True, F[::-1] < F_moy + 1 * F_std)
    Z0_sup = Z[i_start]
    
    # ax.axhline(F_moy, color = 'orange', ls = '--')
    # ax.axhline(F_moy - 1 * F_std, color = 'cyan', ls = '--')
    # ax.axhline(F_moy + 1 * F_std, color = 'cyan', ls = '--', zorder = 5)
    # ax.axvline(Z[i_start], ls = '--', color = 'orange', zorder = 5)

    DZmax = 0.1
    i_stop = ufun.findFirst(True, Z >= Z0_sup + DZmax)
    
    ax.plot(Z[i_start:i_stop], F[i_start:i_stop], color = 'green', zorder = 4)
    
    binf = [0, 0]
    bsup = [np.inf, Z[i_start]]
    p0=[0, 0.99*Z[i_start]]
    [K, Z0], covM = curve_fit(HertzFit, Z[i_start:i_stop], F[i_start:i_stop], p0=p0, bounds=(binf, bsup))
    
    Z_fit = np.linspace(Z0, Z[i_stop], 100)
    F_fit = HertzFit(Z_fit, K, Z0)
    ax.plot(Z_fit, F_fit, color = 'orange', ls = '--', zorder = 6, label = '1st Hertz fit, K = {:.0f}kPa, z0 = {:.3f}mm'.format(K*1e3, Z0))
    ax.legend()
    
    #### Second Fit
    
    i_start_bis = ufun.findFirst(True, Z >= Z0)

    # ax.axvline(Z[i_start], ls = '--', color = 'red', zorder = 5)
   
    ax.plot(Z[i_start_bis:i_stop], F[i_start_bis:i_stop], color = 'turquoise', zorder = 3)
    
    binf = [0, 0]
    bsup = [np.inf, Z0_sup]
    p0=[0, Z0]

    [K2, Z02], covM = curve_fit(HertzFit, Z[i_start_bis:i_stop], F[i_start_bis:i_stop], p0=[K, Z0])
    
    Z_fit = np.linspace(Z02, Z[i_stop], 100)
    F_fit = HertzFit(Z_fit, K2, Z02)
    ax.plot(Z_fit, F_fit, color = 'red', ls = '--', zorder = 6, label = '2nd Hertz fit, K = {:.0f}kPa, z0 = {:.3f}mm'.format(K2*1e3, Z02))
    ax.legend()
    
plt.tight_layout()
plt.show()

# %% Second one

dataDir = "D://MagneticPincherData//Raw//23.10.10_MacroIndenter_VCI"

D = 2.73
R = D/2
f = 'Specimen_RawData_4.csv'
rD = {'Temps' : 'T', 'Deplacement' : 'Z', 'Charge' : 'F'}
df = pd.read_csv(dataDir + '//' + f, engine = 'python', sep = ';', skiprows=[1]).rename(columns = rD)

early_points = 30

dict_curves = {}

imax = find_peaks(df['Z'], distance=150)[0]
imin = find_peaks((-1) * df['Z'], distance=150)[0]
imin = [0, imin[0], len(df)]

Ncycles = len(imax)
fig, axes = plt.subplots(2, Ncycles, figsize = (8, 6))
fig.suptitle('Curves for D = {:.3f} mm'.format(D))

for j in range(len(imax)):
    dict_curves[2*j] = {}
    dict_curves[2*j]['type'] = 'comp'
    dict_curves[2*j]['num'] = j+1
    dict_curves[2*j]['df'] = df.iloc[imin[j]:imax[j], :]
    
    dict_curves[2*j + 1] = {}
    dict_curves[2*j + 1]['type'] = 'relax'
    dict_curves[2*j + 1]['num'] = j+1
    dict_curves[2*j + 1]['df'] = df.iloc[imax[j]:imin[j+1], :]
    
for i in range(len(dict_curves)):
    i_ax = int(dict_curves[i]['type'] == 'relax')
    j_ax = i//2
    try:
        ax = axes[i_ax, j_ax]
    except:
        ax = axes[i_ax]
        
    ax.set_title(dict_curves[i]['type'] + ' ' + str(dict_curves[i]['num']))
    if dict_curves[i]['type'] == 'comp':
        F = dict_curves[i]['df']['F'].values
        Z = dict_curves[i]['df']['Z'].values
    elif dict_curves[i]['type'] == 'relax':
        F = dict_curves[i]['df']['F'].values[::-1]
        Z = dict_curves[i]['df']['Z'].values[::-1]
    
    F_moy = np.average(F[:early_points])
    F_std = np.std(F[:early_points])
    
    #
    F = F - F_moy
    F_moy = 0
    #
    
    ax.plot(Z, F)
    ax.set_xlabel('Z (mm)')
    ax.set_ylabel('F (N)')
    ax.grid(axis='y')
    
    #` Function
    
    def HertzFit(z, K, Z0):
        f = (4/3) * K * R**0.5 * (z - Z0)**1.5
        return(f)
    
    #### First Fit
    
    i_start = len(F) - ufun.findFirst(True, F[::-1] < F_moy + 1 * F_std)
    Z0_sup = Z[i_start]
    
    # ax.axhline(F_moy, color = 'orange', ls = '--')
    # ax.axhline(F_moy - 1 * F_std, color = 'cyan', ls = '--')
    # ax.axhline(F_moy + 1 * F_std, color = 'cyan', ls = '--', zorder = 5)
    # ax.axvline(Z[i_start], ls = '--', color = 'orange', zorder = 5)

    DZmax = 0.1
    i_stop = ufun.findFirst(True, Z >= Z0_sup + DZmax)
    
    ax.plot(Z[i_start:i_stop], F[i_start:i_stop], color = 'green', zorder = 4)
    
    binf = [0, 0]
    bsup = [np.inf, Z[i_start]]
    p0=[0, 0.99*Z[i_start]]
    [K, Z0], covM = curve_fit(HertzFit, Z[i_start:i_stop], F[i_start:i_stop], p0=p0, bounds=(binf, bsup))
    
    Z_fit = np.linspace(Z0, Z[i_stop], 100)
    F_fit = HertzFit(Z_fit, K, Z0)
    ax.plot(Z_fit, F_fit, color = 'orange', ls = '--', zorder = 6, label = '1st Hertz fit, K = {:.0f}kPa, z0 = {:.3f}mm'.format(K*1e3, Z0))
    ax.legend()
    
    #### Second Fit
    
    i_start_bis = ufun.findFirst(True, Z >= Z0)

    # ax.axvline(Z[i_start], ls = '--', color = 'red', zorder = 5)
   
    ax.plot(Z[i_start_bis:i_stop], F[i_start_bis:i_stop], color = 'turquoise', zorder = 3)
    
    binf = [0, 0]
    bsup = [np.inf, Z0_sup]
    p0=[0, Z0]

    [K2, Z02], covM = curve_fit(HertzFit, Z[i_start_bis:i_stop], F[i_start_bis:i_stop], p0=[K, Z0])
    
    Z_fit = np.linspace(Z02, Z[i_stop], 100)
    F_fit = HertzFit(Z_fit, K2, Z02)
    ax.plot(Z_fit, F_fit, color = 'red', ls = '--', zorder = 6, label = '2nd Hertz fit, K = {:.0f}kPa, z0 = {:.3f}mm'.format(K2*1e3, Z02))
    ax.legend()
    
plt.tight_layout()
plt.show()

        
# %% Third one

dataDir = "D://MagneticPincherData//Raw//23.10.10_MacroIndenter_VCI"

D = 4.004
R = D/2
f = 'Specimen_RawData_5.csv'
rD = {'Temps' : 'T', 'Deplacement' : 'Z', 'Charge' : 'F'}
df = pd.read_csv(dataDir + '//' + f, engine = 'python', sep = ';', skiprows=[1]).rename(columns = rD)

early_points = 30

dict_curves = {}

imax = find_peaks(df['Z'], distance=150)[0]
imin = find_peaks((-1) * df['Z'], distance=150)[0]
imin = [0, imin[0], len(df)]

Ncycles = len(imax)
fig, axes = plt.subplots(2, Ncycles, figsize = (8, 6))
fig.suptitle('Curves for D = {:.3f} mm'.format(D))

for j in range(len(imax)):
    dict_curves[2*j] = {}
    dict_curves[2*j]['type'] = 'comp'
    dict_curves[2*j]['num'] = j+1
    dict_curves[2*j]['df'] = df.iloc[imin[j]:imax[j], :]
    
    dict_curves[2*j + 1] = {}
    dict_curves[2*j + 1]['type'] = 'relax'
    dict_curves[2*j + 1]['num'] = j+1
    dict_curves[2*j + 1]['df'] = df.iloc[imax[j]:imin[j+1], :]
    
for i in range(len(dict_curves)):
    i_ax = int(dict_curves[i]['type'] == 'relax')
    j_ax = i//2
    try:
        ax = axes[i_ax, j_ax]
    except:
        ax = axes[i_ax]
        
    ax.set_title(dict_curves[i]['type'] + ' ' + str(dict_curves[i]['num']))
    if dict_curves[i]['type'] == 'comp':
        F = dict_curves[i]['df']['F'].values
        Z = dict_curves[i]['df']['Z'].values
    elif dict_curves[i]['type'] == 'relax':
        F = dict_curves[i]['df']['F'].values[::-1]
        Z = dict_curves[i]['df']['Z'].values[::-1]
    
    F_moy = np.average(F[:early_points])
    F_std = np.std(F[:early_points])
    
    #
    F = F - F_moy
    F_moy = 0
    #
    
    ax.plot(Z, F)
    ax.set_xlabel('Z (mm)')
    ax.set_ylabel('F (N)')
    ax.grid(axis='y')
    
    #` Function
    
    def HertzFit(z, K, Z0):
        f = (4/3) * K * R**0.5 * (z - Z0)**1.5
        return(f)
    
    #### First Fit
    
    i_start = len(F) - ufun.findFirst(True, F[::-1] < F_moy + 1 * F_std)
    Z0_sup = Z[i_start]
    
    # ax.axhline(F_moy, color = 'orange', ls = '--')
    # ax.axhline(F_moy - 1 * F_std, color = 'cyan', ls = '--')
    # ax.axhline(F_moy + 1 * F_std, color = 'cyan', ls = '--', zorder = 5)
    # ax.axvline(Z[i_start], ls = '--', color = 'orange', zorder = 5)

    DZmax = 0.1
    i_stop = ufun.findFirst(True, Z >= Z0_sup + DZmax)
    
    ax.plot(Z[i_start:i_stop], F[i_start:i_stop], color = 'green', zorder = 4)
    
    binf = [0, 0]
    bsup = [np.inf, Z[i_start]]
    p0=[0, 0.99*Z[i_start]]
    [K, Z0], covM = curve_fit(HertzFit, Z[i_start:i_stop], F[i_start:i_stop], p0=p0, bounds=(binf, bsup))
    
    Z_fit = np.linspace(Z0, Z[i_stop], 100)
    F_fit = HertzFit(Z_fit, K, Z0)
    ax.plot(Z_fit, F_fit, color = 'orange', ls = '--', zorder = 6, label = '1st Hertz fit, K = {:.0f}kPa, z0 = {:.3f}mm'.format(K*1e3, Z0))
    ax.legend()
    
    #### Second Fit
    
    i_start_bis = ufun.findFirst(True, Z >= Z0)

    # ax.axvline(Z[i_start], ls = '--', color = 'red', zorder = 5)
   
    ax.plot(Z[i_start_bis:i_stop], F[i_start_bis:i_stop], color = 'turquoise', zorder = 3)
    
    binf = [0, 0]
    bsup = [np.inf, Z0_sup]
    p0=[0, Z0]

    [K2, Z02], covM = curve_fit(HertzFit, Z[i_start_bis:i_stop], F[i_start_bis:i_stop], p0=[K, Z0])
    
    Z_fit = np.linspace(Z02, Z[i_stop], 100)
    F_fit = HertzFit(Z_fit, K2, Z02)
    ax.plot(Z_fit, F_fit, color = 'red', ls = '--', zorder = 6, label = '2nd Hertz fit, K = {:.0f}kPa, z0 = {:.3f}mm'.format(K2*1e3, Z02))
    ax.legend()
    
plt.tight_layout()
plt.show()
        