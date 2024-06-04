# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:25:12 2024

@author: JosephVermeil
"""

# %% > Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import os
import re
import json
import pickle
import datetime
import matplotlib

from scipy import signal
from scipy.interpolate import Akima1DInterpolator # CubicSpline, PchipInterpolator, 
from scipy.optimize import curve_fit

import CellShapeSolver as css
import IndentHertzianFit as ihf
import UtilityFunctions as ufun
import GraphicStyles as gs
import CortexPaths as cp

mainFigDir = os.path.join(cp.DirDataFig, 'ChiaroAnalysis')

gs.set_mediumText_options_jv()

#### Utility function
def dateFormat(d):
    d2 = d[0:2]+'.'+d[3:5]+'.'+d[6:8]
    return(d2)

def checkForbiddenWords(s, fwL):
    res = True
    for w in fwL:
        if w.lower() in s.lower():
            res = False
    return(res)

def checkCompulsaryWords(s, fwL):
    res = True
    for w in fwL:
        if w.lower() not in s.lower():
            res = False
    return(res)

def saveAsPickle(d, path):
    head, tail = os.path.split(path)
    if not os.path.exists(head):
        os.mkdir(head)
    with open(path, "wb") as fp:
        pickle.dump(d, fp)

def loadPickle(path):
    if os.path.exists(path):
        with open(path, "rb") as fp:
            d = pickle.load(fp)
    else:
        d = None
    return(d)

# %% Constants

mainDir = "D://MagneticPincherData//Data_Analysis//Chiaro//FakeCells//FC1_R1-10_H0-12_Rp-26_Ka-05-_T0-03"
cellName = 'FakeCell-01'

R1=10
H0=12
Rp=26
Ka = 0.5
T0 = 0.3

delta_values = np.linspace(0, H0/2, 400) + 0.01

pickle_shape_path = os.path.join(mainDir, cellName + '_shape.pkl')
pickle_poly_path = os.path.join(mainDir, cellName + '_poly.pkl')

pickle_shape_path_error = os.path.join(mainDir, cellName + '_shape_error01.pkl')
pickle_poly_path_error = os.path.join(mainDir, cellName + '_poly_error01.pkl')

# %% TRUE SHAPE

shape_resDict, shape_resDf = css.CSS_sphere_general(R1, H0, Rp, delta_values)
       
css.plot_contours(shape_resDict, shape_resDf, 
                  save = True, save_path = mainDir, fig_name_prefix = cellName)
poly_dict = css.plot_curves(shape_resDict, shape_resDf, return_polynomial_fits = True, 
                            save = True, save_path = mainDir, fig_name_prefix = cellName)

saveAsPickle(shape_resDict,  pickle_shape_path)
saveAsPickle(poly_dict,  pickle_poly_path)

# %% TRUE TENSION

res_dict = loadPickle(pickle_shape_path)
poly_dict = loadPickle(pickle_poly_path)

P_alpha = poly_dict['P_alpha']
P_Lc = poly_dict['P_Lc']


delta_comp = delta_values
alpha_comp = np.polyval(P_alpha, delta_comp)
Lc_comp = np.polyval(P_Lc, delta_comp)
T_comp = Ka * alpha_comp + T0

F_comp = T_comp * 2*np.pi * Lc_comp


params_full = [T0, Ka]



#### 5.2 Plot tension
figT, axesT = plt.subplots(3, 2, figsize = (10, 9), sharex='col')
ax=axesT[0,0]
ax.plot(delta_comp, alpha_comp*100, 'g-', lw=3)
# ax.set_xlabel('$\\delta$ (µm)')
ax.set_ylabel('$\\alpha$ (%)')
ax.grid()

ax=axesT[1,0]
ax.plot(delta_comp, F_comp, 'r-', lw=1.5)
# ax.set_xlabel('$\\delta$ (µm)')
ax.set_ylabel('F (nN)')
ax.grid()

ax=axesT[2,0]
ax.plot(delta_comp, Lc_comp, 'c-', lw=3)
ax.plot(delta_comp, np.polyval(P_Lc, delta_comp), 'r--', lw=1)
ax.set_xlabel('$\\delta$ (µm)')
ax.set_ylabel('$L_c$ (µm)')
ax.grid()


ax=axesT[0,1]
ax.plot(alpha_comp, Lc_comp, c='royalblue', ls='-', lw=3)
# ax.set_xlabel('$\\alpha$')
ax.set_ylabel('$L_c$ (µm)')
ax.grid()

ax=axesT[1,1]
ax.plot(alpha_comp, F_comp, c='darkred', ls='-')
# ax.set_xlabel('$\\alpha$')
ax.set_ylabel('F (nN)')
ax.grid()

ax=axesT[2,1]
ax.plot(alpha_comp, T_comp)
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('$F / 2\\pi.L_c$ (nN/µm)')
ax.set_ylim([0, 0.8])
ax.grid()

s = f'Fit, all curve - F/$L_c$ = {params_full[1]:.3e}$\\alpha$ + {params_full[0]:.3e}'
# s += f'$R^2$ = {r_squared:.3f}'
ax.plot(alpha_comp, alpha_comp*params_full[1] + params_full[0], 'r--', label = s)


ax.legend()

plt.tight_layout()
plt.show()

name = cellName + '_tensionFit'
ufun.archiveFig(figT, name = name, figDir = mainDir)


# %% ERROR ON SHAPE

R1_err=R1
H0_err=H0+0.5
Rp_err=Rp

shape_resDict, shape_resDf = css.CSS_sphere_general(R1_err, H0_err, Rp_err, delta_values)
       
css.plot_contours(shape_resDict, shape_resDf, 
                  save = True, save_path = mainDir, fig_name_prefix = cellName + '_ERROR-01')
poly_dict = css.plot_curves(shape_resDict, shape_resDf, return_polynomial_fits = True, 
                            save = True, save_path = mainDir, fig_name_prefix = cellName + '_ERROR-01')

P_alpha = poly_dict['P_alpha']
P_Lc = poly_dict['P_Lc']


# pickle_path = os.path.join(indent_dir, indentID + '_results.pkl')
# pickle_hertz_path = os.path.join(indent_dir, 'pickles', indentID + '_hertz.pkl')
pickle_shape_path_error = os.path.join(mainDir, cellName + '_shape_error01.pkl')
pickle_poly_path_error = os.path.join(mainDir, cellName + '_poly_error01.pkl')
# pickle_tension_path = os.path.join(indent_dir, 'pickles', indentID + '_tension.pkl')

saveAsPickle(shape_resDict,  pickle_shape_path_error)
saveAsPickle(poly_dict,  pickle_poly_path_error)


# %% RESULTING ERROR ON TENSION

shape_resDict_err = loadPickle(pickle_shape_path_error)
poly_dict_err = loadPickle(pickle_poly_path_error)
# shape_resDict_err = loadPickle(pickle_shape_path)
# poly_dict_err = loadPickle(pickle_poly_path)

P_alpha = poly_dict_err['P_alpha']
P_Lc = poly_dict_err['P_Lc']

delta_comp = delta_values
F_comp = F_comp

F0_error = 0.5 # µm

if F0_error>0:
    i_thresh = ufun.findFirst(True, delta_comp > F0_error)
    F_comp_err = F_comp[i_thresh:] - F_comp[i_thresh] + 0.01
    delta_comp_err = delta_comp[i_thresh:] - delta_comp[i_thresh] + 0.01
else:
    F_comp_err = F_comp
    delta_comp_err = delta_comp


alpha_comp = np.polyval(P_alpha, delta_comp_err)
Lc_comp = np.polyval(P_Lc, delta_comp_err)
T_comp = F_comp_err / (2*np.pi*Lc_comp)

params_full, results_full = ufun.fitLineHuber(alpha_comp, T_comp)
CI_full = results_full.conf_int(alpha=0.05)
CIW_full = CI_full[:,1] - CI_full[:,0]



#### 5.2 Plot tension
figT, axesT = plt.subplots(3, 2, figsize = (10, 9), sharex='col')
ax=axesT[0,0]
ax.plot(delta_comp_err, alpha_comp*100, 'g-', lw=3)
# ax.set_xlabel('$\\delta$ (µm)')
ax.set_ylabel('$\\alpha$ (%)')
ax.grid()

ax=axesT[1,0]
ax.plot(delta_comp_err, F_comp_err, 'r-', lw=1.5)
# ax.set_xlabel('$\\delta$ (µm)')
ax.set_ylabel('F (nN)')
ax.grid()

ax=axesT[2,0]
ax.plot(delta_comp_err, Lc_comp, 'c-', lw=3)
ax.set_xlabel('$\\delta$ (µm)')
ax.set_ylabel('$L_c$ (µm)')
ax.grid()


ax=axesT[0,1]
ax.plot(alpha_comp, Lc_comp, c='royalblue', ls='-', lw=3)
# ax.set_xlabel('$\\alpha$')
ax.set_ylabel('$L_c$ (µm)')
ax.grid()

ax=axesT[1,1]
ax.plot(alpha_comp, F_comp_err, c='darkred', ls='-')
# ax.set_xlabel('$\\alpha$')
ax.set_ylabel('F (nN)')
ax.grid()

ax=axesT[2,1]
ax.plot(alpha_comp, T_comp)
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('$F / 2\\pi.L_c$ (nN/µm)')
ax.set_ylim([0, 0.8])
ax.grid()

s = f'Fit, all curve - F/$L_c$ = {params_full[1]:.3e}$\\alpha$ + {params_full[0]:.3e}'
# s += f'$R^2$ = {r_squared:.3f}'
ax.plot(alpha_comp, alpha_comp*params_full[1] + params_full[0], 'r--', label = s)

ax.legend()

suptitle = f'$R_1$ = {R1_err:.1f}, $H_0$ = {H0_err:.1f}, $R_p$ = {Rp_err:.1f} (µm), Error on $\\delta,F_0$ = {F0_error:.1f} (µm)'
figT.suptitle(suptitle)

figT.tight_layout()
plt.show()

name = cellName + '_ERROR-01' + '_tensionFit'
ufun.archiveFig(figT, name = name, figDir = mainDir)


# %% Plot all errors

def computeErrorVectors(dirPath, F_comp):
    pickle_shape_path_error = os.path.join(dirPath, 'FakeCell-01_shape_error01.pkl')
    pickle_poly_path_error = os.path.join(dirPath, 'FakeCell-01_poly_error01.pkl')
    shape_resDict_err = loadPickle(pickle_shape_path_error)
    poly_dict_err = loadPickle(pickle_poly_path_error)
    
    P_alpha = poly_dict_err['P_alpha']
    P_Lc = poly_dict_err['P_Lc']
    
    delta_comp = delta_values
    F_comp = F_comp
    
    F0_error = 0.5 # µm
    
    alpha_comp = np.polyval(P_alpha, delta_comp)
    Lc_comp = np.polyval(P_Lc, delta_comp)
    T_comp = F_comp / (2*np.pi*Lc_comp)
    
    params_full, results_full = ufun.fitLineHuber(alpha_comp, T_comp)
    CI_full = results_full.conf_int(alpha=0.05)
    CIW_full = CI_full[:,1] - CI_full[:,0]
    
    return(alpha_comp, Lc_comp, T_comp, params_full)



def plotErrorTension(fig, axes, delta_comp, F_comp, alpha_comp, Lc_comp, T_comp, params_full,
                     color='b', label=''):
    ax=axes[0,0]
    ax.plot(delta_comp, alpha_comp*100, lw=3, label=label, c=color)
    # ax.set_xlabel('$\\delta$ (µm)')
    ax.set_ylabel('$\\alpha$ (%)')
    ax.grid()
    ax.legend()

    ax=axes[1,0]
    ax.plot(delta_comp, F_comp, lw=1.5, label=label, c=color)
    # ax.set_xlabel('$\\delta$ (µm)')
    ax.set_ylabel('F (nN)')
    ax.grid()
    ax.legend()
    
    ax=axes[2,0]
    ax.plot(delta_comp, Lc_comp,  lw=3, label=label, c=color)
    ax.set_xlabel('$\\delta$ (µm)')
    ax.set_ylabel('$L_c$ (µm)')
    ax.grid()
    ax.legend()


    ax=axes[0,1]
    ax.plot(alpha_comp, Lc_comp, ls='-', lw=3, label=label, c=color)
    # ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$L_c$ (µm)')
    ax.grid()
    ax.legend()

    ax=axes[1,1]
    ax.plot(alpha_comp, F_comp, ls='-', label=label, c=color)
    # ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('F (nN)')
    ax.grid()
    ax.legend()

    ax=axes[2,1]
    ax.plot(alpha_comp, T_comp, 'k:')
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$F / 2\\pi.L_c$ (nN/µm)')
    ax.grid()

    s = f'Fit {label} - F/$L_c$ = {params_full[1]:.3e}$\\alpha$ + {params_full[0]:.3e}'
    # s += f'$R^2$ = {r_squared:.3f}'
    ax.plot(alpha_comp, alpha_comp*params_full[1] + params_full[0], c=color, ls='--', label = s)
    ax.legend()

    return(fig, axes)

    

    

mainDir = "D://MagneticPincherData//Data_Analysis//Chiaro//FakeCells//FC1_R1-10_H0-12_Rp-26_Ka-05-_T0-03"
allFiles = os.listdir(mainDir)

pickle_shape_path = os.path.join(mainDir, 'FakeCell-01_shape.pkl')
pickle_poly_path = os.path.join(mainDir, 'FakeCell-01_poly.pkl')
res_dict = loadPickle(pickle_shape_path)
poly_dict = loadPickle(pickle_poly_path)

P_alpha = poly_dict['P_alpha']
P_Lc = poly_dict['P_Lc']

delta_comp = delta_values
alpha_comp = np.polyval(P_alpha, delta_comp)
Lc_comp = np.polyval(P_Lc, delta_comp)
T_comp = Ka * alpha_comp + T0

F_comp = T_comp * 2*np.pi * Lc_comp

params_full = [T0, Ka]

colorList = gs.colorList10
figT, axesT = plt.subplots(3, 2, figsize = (10, 9), sharex='col')
figT, axesT = plotErrorTension(figT, axesT, delta_comp, F_comp, alpha_comp, Lc_comp, T_comp, params_full,
                     color=colorList[0], label='Truth')

suptitle = ''
figT.suptitle(suptitle)
figT.tight_layout()
ic = 0

for f in allFiles:
    if f.startswith("error_H0"):
        ic += 1
        label = f.split('_')[-1] + ' um'
        errorPath = os.path.join(mainDir, f)
        alpha_comp, Lc_comp, T_comp, params_full = computeErrorVectors(errorPath, F_comp)
        figT, axesT = plotErrorTension(figT, axesT, delta_comp, F_comp, alpha_comp, Lc_comp, T_comp, params_full,
                             color=colorList[ic], label=label)