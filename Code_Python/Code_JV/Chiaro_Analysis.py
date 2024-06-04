# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 20:37:38 2024

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

# %% Run the analysis block by block

# %%% Some parameters

Ref_Height = 20 # µm


# %%% Paths

date, manip, cell, indent = '24-04-11', 'M3', 'C2', '003'
date2 = dateFormat(date)
specif1 = '_25um' # indenter size
specif2 = '-20um' # height above substrate

mainDir = f'D://MagneticPincherData//Raw//{date2}_NanoIndent_ChiaroData{specif1}'
manipDir = os.path.join(mainDir, f'{manip}')

indentID = f'{dateFormat(date)}_{manip}_{cell}_I{int(indent)}'


# %%% Refresh Geometry


fgeom_path = os.path.join(mainDir, f'{date}_cellGeometry.csv')
dfG = pd.read_csv(fgeom_path)
dfG['R1'] = (dfG['W_R1'] + dfG['H_R1'])/4

dfG_manipID = dfG['date'] + '_' +  dfG['manip']
dfG_cellID = dfG['date'] + '_' +  dfG['manip'] + '_P1_' + dfG['cell']
dfG_indentID = dfG['date'] + '_' +  dfG['manip'] + '_' + dfG['cell'] + '_' + dfG['indent']
dfG['manipID'] = dfG_manipID
dfG['cellID'] = dfG_cellID
dfG['indentID'] = dfG_indentID

dfG.to_csv(fgeom_path, index = False)


# %%% Get the indentation hertzian fit

findent_path = os.path.join(manipDir, f'{cell}//Indentations//{cell}{specif2} Indentation_{indent}.txt')
findent = findent_path.split('//')[-1]

# 1. Read all the Indentation files for that cell
dI = ihf.initializeFitHertz(findent_path, date, manip, cell, indent) #

# 2. Analyze them
dfI = dI['df']
hertzFitResults = ihf.FitHertz(dI, mode = 'dZmax', dZmax = 2,
                         plot = 1, save_plot = True, save_path = manipDir)
# nm to µm # µN to pN

# %%% Get the radius table

dir_radius_path = f'D://MagneticPincherData//Raw//{date}_NanoIndent_ChiaroData{specif1}//{manip}//{cell}//FilmData//'
f_radius_list = os.listdir(dir_radius_path)

list_dfR = [pd.read_csv(os.path.join(dir_radius_path, f), index_col=0) for f in f_radius_list]

def plot_R0table(dfR, fig, ax):
    T, R, zone = dfR['T'], dfR['R'], dfR['zone'].astype(int)
    color_list = plt.cm.rainbow(np.linspace(0.2, 0.8, np.max(zone) + 1))[::-1]
    color_array = np.array([matplotlib.colors.rgb2hex(color_list[zone[k]]) for k in range(len(R))])
    params, results = ufun.fitLineHuber(T[zone == 1], R[zone == 1])
    R_fit = T[zone == 1]*params[1] + params[0]
    
    R_init = np.min(R[zone == 0])
    R_high = np.mean(R[zone == 2])
    T_init = (R_init-params[0])/params[1]
    T_high = (R_high-params[0])/params[1]

    ax.scatter(T, R, c=color_array, 
            marker = 'o', edgecolor='k', s = 50)
    ax.plot(T[zone == 1], R_fit, 'b--', label = f'R = {params[1]:.2f}*T + {params[0]:.1f}')
    ax.axhline(R_init, c='r', ls='--', lw=1)
    ax.axhline(R_high, c='r', ls='--', lw=1)
    ax.plot([T_init, T_high], [R_init, R_high], 'ro')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('R0 (µm)')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
N = len(list_dfR)
avg_dfR = list_dfR[0].copy(deep=True)
if N >= 2:
    for i in range(1, N):
        avg_dfR += list_dfR[i]
    avg_dfR /= N

fig_R, axes_R = plt.subplots(1,2, figsize=(12,6), sharey=True)
for dfR in list_dfR:
    plot_R0table(dfR, fig_R, axes_R[0])

plot_R0table(avg_dfR, fig_R, axes_R[1])

T, R, zone = avg_dfR['T'], avg_dfR['R'], avg_dfR['zone']
params, results = ufun.fitLineHuber(T[zone == 1], R[zone == 1])
R_fit = T[zone == 1]*params[1] + params[0]

R_init = np.min(R[zone == 0])
R_high = np.mean(R[zone == 2])
T_init = (R_init-params[0])/params[1]
T_high = (R_high-params[0])/params[1]



# %%% Geometry

date2 = dateFormat(date)
fgeom_path = os.path.join(mainDir, f'{date2}_cellGeometry.csv')
dfG = pd.read_csv(fgeom_path)
dfG['R1'] = (dfG['W_R1'] + dfG['H_R1'])/4

manipID_dfG = dfG['date'] + '_' +  dfG['manip']
cellID_dfG = dfG['date'] + '_' +  dfG['manip'] + '_P1_' + dfG['cell']
indentID_dfG = dfG['date'] + '_' +  dfG['manip'] + '_' + dfG['cell'] + '_' + dfG['indent']
dfG['manipID'] = manipID_dfG
dfG['cellID'] = cellID_dfG
dfG['indentID'] = indentID_dfG

# dfG.to_csv(fgeom_path, index = False)


# %%% Compute the compression

Z0 = hertzFitResults['Z0_d']
i_start = hertzFitResults['i_start']
# i_stop = hertzFitResults['i_stop']
i_stop = ufun.findFirst(dfI['step'], 1)
selected_indices = np.arange(i_start, i_stop)

Tcomp = dfI.loc[selected_indices, 'Time (s)'].values # s
Tcomp = Tcomp - np.min(Tcomp)
Zcomp = dfI.loc[selected_indices, 'Z Tip - denoised'].values*1e-3 # µm
Fcomp = dfI.loc[selected_indices, 'Load - denoised'].values*1e3 # nN

delta_comp = Zcomp - np.min(Zcomp) + 0.01
F_comp = Fcomp - np.min(Fcomp) + 0.01
maxDelta = np.max(delta_comp)

# plt.plot(Tcomp, delta)

R1 = dfG.loc[dfG['indentID'] == indentID, 'R1'].values[0] # Cell basal radius, µm
H0 = dfG.loc[dfG['indentID'] == indentID, 'Href'].values[0] - Z0 # Cell height, µm
Rp = dI['R'] # Probe radius, µm

print(R1, H0, Rp)

N_points = 200
delta_values = np.linspace(0, maxDelta, N_points) + 0.01

res_dict, res_df = css.CSS_sphere_general(R1, H0, Rp, delta_values)
       
css.plot_contours(res_dict, res_df)
css.plot_curves(res_dict, res_df, return_polynomial_fits = True)

# %%% Investigate R0(delta) comparison

time_microscope = dfR['T'][dfR['zone']==1].values
time_microscope = time_microscope - (np.max(time_microscope) - np.max(Tcomp)) - 0.1
delta_microscope = Akima1DInterpolator(Tcomp, delta_comp)(time_microscope)

R0_microscope = dfR['R'][dfR['zone']==1].values

fig_c, ax_c = plt.subplots(1, 1, figsize = (6, 6))
ax=ax_c
# ax.plot(Tcomp, delta_comp, 'b-', lw = 0.75)
# ax.plot(time_microscope, delta_microscope, 'ro')

ax.plot(res_df['delta'], res_df['R0'], 'b-', lw = 0.75, label='computed')
ax.plot(delta_microscope, R0_microscope, 'ro', label='microscope')
ax.set_xlabel('$\\delta$ (µm)')
ax.set_ylabel('$R_0$ (µm)')

fig_c.suptitle('Comparison between expected and observed R0', fontsize=12)
fig_c.tight_layout()
plt.show()


# %%% Compute the tension and fit Ka & T0

#### Some more shit
f0 = (res_df['error']==0)
def cubic_for_fit(x, K3, K2, K1):
    return(K3*x**3 + K2*x**2 + K1*x**1)

delta = res_df[f0]['delta']
alpha_raw = res_df[f0]['alpha']
K_alpha, covM = curve_fit(cubic_for_fit, delta, alpha_raw, p0=(1,1,1))
P_alpha = np.poly1d([k for k in K_alpha] + [0])

Lc_raw = (R1**2 * res_df[f0]['curvature'].values - R1*np.sin(res_df[f0]['phi'].values))

K_Lc, covM = curve_fit(cubic_for_fit, delta, Lc_raw, p0=(1,1,1))
P_Lc = np.poly1d([k for k in K_Lc] + [0])


# figLc, axLc = plt.subplots(1, 1, figsize = (6, 6))
# ax = axLc
# ax.plot(delta, Lc_raw, 'wo', mec='c', ms=10)
# ax.plot(delta, np.polyval(P_Lc, delta), 'r--')
# ax.set_xlabel('$\\delta$ (µm)')
# ax.set_ylabel('$L_c$ (µm)')
# ax.grid()
# plt.show()

i_thresh_Lc = ufun.findFirst(True, np.polyval(P_Lc, delta_comp) > 0.1)

delta_comp_f = delta_comp[i_thresh_Lc:]
Fcomp_f = F_comp[i_thresh_Lc:]

M = np.array([delta_comp_f, Fcomp_f]).T
M = M[M[:, 0].argsort()]
[delta_comp_f, Fcomp_f] = M.T

alpha_comp_f = np.polyval(P_alpha, delta_comp_f)
Lc_comp_f = np.polyval(P_Lc, delta_comp_f)


params_full, results_full = ufun.fitLineHuber(alpha_comp_f, Fcomp_f/(2*np.pi*Lc_comp_f))
f_a = (alpha_comp_f>0.01) & (alpha_comp_f<0.1)
params_filtered, results_filtered = ufun.fitLineHuber(alpha_comp_f[f_a], Fcomp_f[f_a]/(2*np.pi*Lc_comp_f[f_a]))


fig, axes = plt.subplots(3, 2, figsize = (10, 9), sharex='col')
ax=axes[0,0]
ax.plot(delta_comp_f, alpha_comp_f*100, 'g-', lw=3)
# ax.set_xlabel('$\\delta$ (µm)')
ax.set_ylabel('$\\alpha$ (%)')
ax.grid()

ax=axes[1,0]
ax.plot(delta_comp_f, Fcomp_f, 'r-', lw=1.5)
# ax.set_xlabel('$\\delta$ (µm)')
ax.set_ylabel('F (nN)')
ax.grid()

ax=axes[2,0]
ax.plot(delta_comp_f, Lc_comp_f, 'c-', lw=3)
ax.plot(delta_comp_f, np.polyval(P_Lc, delta_comp_f), 'r--', lw=1)
ax.set_xlabel('$\\delta$ (µm)')
ax.set_ylabel('$L_c$ (µm)')
ax.grid()


ax=axes[1,1]
ax.plot(alpha_comp_f, Fcomp_f, c='darkred', ls='-')
# ax.set_xlabel('$\\alpha$')
ax.set_ylabel('F (nN)')
ax.grid()

ax=axes[2,1]
ax.plot(alpha_comp_f, Fcomp_f/(2*np.pi*Lc_comp_f))
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('$F / 2\\pi.L_c$ (nN/µm)')
ax.grid()

s = f'Fit, all curve - F/$L_c$ = {params_full[1]:.3e}$\\alpha$ + {params_full[0]:.3e}'
# s += f'$R^2$ = {r_squared:.3f}'
ax.plot(alpha_comp_f, alpha_comp_f*params_full[1] + params_full[0], 'r--', label = s)

s = f'Fit, $\\alpha$ > 0.01 - F/$L_c$ = {params_filtered[1]:.3e}$\\alpha$ + {params_filtered[0]:.3e}'
# s += f'$R^2$ = {r_squared:.3f}'
ax.plot(alpha_comp_f[f_a], alpha_comp_f[f_a]*params_filtered[1] + params_filtered[0], c='gold', ls='--', label = s)

ax.legend()

plt.tight_layout()
plt.show()



# %% Do it as a function
        
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
    

# test_path = ".//test//01.pkl"
# data = [{'a':1, 'b':2}, np.arange(10), pd.DataFrame({'a':np.arange(10), 'b':2*np.arange(10)})]
# saveAsPickle(data, test_path)
# data_opened = loadPickle(test_path)


def indentations_hertz(indent_path, date, manip, cell, indent,
                       mode = 'dZmax', dZmax = 2, redo = False, 
                       save_plots=True, figures_path=''):
    
    indent_dir, indent_file = os.path.split(indent_path)
    
    #### LOAD JSON / PICKLE
    manipID = f'{date}_{manip}'
    cellID = f'{date}_{manip}_{cell}'
    indentID = f'{date}_{manip}_{cell}_I{int(indent)}'
    
    pickle_hertz_path = os.path.join(indent_dir, 'pickles', indentID + '_hertz.pkl')

    loaded_pickle = False
    try:
        hertz_resDict = loadPickle(pickle_hertz_path)
        if hertz_resDict != None:
            loaded_pickle = True            
    except:
        pass

    #### 1. Fit Hertz
    error = False
    if redo or not loaded_pickle:
        dI = ihf.initializeFitHertz(indent_path, date, manip, cell, indent) #
        dfI = dI['df']
        try:
            hertz_resDict = ihf.FitHertz(dI, mode = 'dZmax', dZmax = 2,
                                     plot = 1, save_plot = save_plots, save_path = figures_path)
        except:
            error = True
    
    #### SAVE JSON / PICKLE
    if not error:
        saveAsPickle(hertz_resDict, pickle_hertz_path)
    
        otherSaveDir = os.path.join(cp.DirDataAnalysis, 'Chiaro', manipID, cellID)
        head, tail = os.path.split(otherSaveDir)
        if not os.path.exists(head):
            os.mkdir(head)
        if not os.path.exists(otherSaveDir):
            os.mkdir(otherSaveDir) 
            
        saveAsPickle(hertz_resDict,  os.path.join(otherSaveDir, indentID + '_hertz.pkl'))
    
        return(hertz_resDict)




def indentations_apt(indent_path, date, manip, cell, indent,
                     R1, Href='glass-indent', save_plots=True, figures_path='', **kwargs):
    # tentative name for now
    # apt stands for "all purpose tool"
    
    VALID = True
    
    tasks = {'Fit Hertz':True,
             'Compute shape':True,
             'Fit tension':True}
    tasks.update(kwargs)
    
    indent_dir, indent_file = os.path.split(indent_path)
    
    #### LOAD JSON / PICKLE
    manipID = f'{date}_{manip}'
    cellID = f'{date}_{manip}_{cell}'
    indentID = f'{date}_{manip}_{cell}_I{int(indent)}'
    
    pickle_path = os.path.join(indent_dir, indentID + '_results.pkl')
    pickle_hertz_path = os.path.join(indent_dir, 'pickles', indentID + '_hertz.pkl')
    pickle_shape_path = os.path.join(indent_dir, 'pickles', indentID + '_shape.pkl')
    pickle_poly_path = os.path.join(indent_dir, 'pickles', indentID + '_poly.pkl')
    pickle_tension_path = os.path.join(indent_dir, 'pickles', indentID + '_tension.pkl')
    # print(pickle_path, pickle_shape_path)
    loaded_pickle = False
    try:
        jar = loadPickle(pickle_path)
        if jar != None:
            [dI, hertz_resDict, shape_resDict, poly_dict, tension_dict] = jar
            dfI = dI['df']
            loaded_pickle = True            
    except:
        pass
    

    #### 1. Fit Hertz
    if tasks['Fit Hertz'] or not loaded_pickle:
        # try:
        dI = ihf.initializeFitHertz(indent_path, date, manip, cell, indent) #
        dfI = dI['df']
        hertz_resDict = ihf.FitHertz(dI, mode = 'dZmax', dZmax = 2,
                                 plot = 1, save_plot = save_plots, save_path = figures_path)
        # except:
            # VALID = False
            # print(gs.ORANGE + "Failed to run Hertz Analysis" + gs.NORMAL)
            # return
            
        #### 1.2 Height of the cell
        if Href == 'glass-indent':
            glass_cellID = f'{date}_{manip}_P1_{cell}-glass'
            glass_indentID = f'{date}_{manip}_{cell}-glass_I{int(indent)}'
            glass_indent_path = os.path.join(manipDir, f'{cell}-glass//Indentations//{cell}-glass{specif2} Indentation_{indent}.txt')
            
            dI_glass = ihf.initializeFitHertz(glass_indent_path, date, manip, cell, indent)
            Zc = ihf.findHardSurface(dI_glass, plot = True, save_plot = save_plots, save_path = figures_path)
            
            Href = Zc
            
        hertz_resDict['Href'] = Href
            
    
    #### 2. Contact Point
    Z0 = hertz_resDict['Z0_d']
    try:
        Href = hertz_resDict['Href']
    except:
        pass
    

    #### 3. Prepare the data
    i_start = hertz_resDict['i_start']
    # i_stop = hertzFitResults['i_stop']
    i_stop = ufun.findFirst(dfI['step'], 1)
    selected_indices = np.arange(i_start, i_stop)

    T_fullComp = dfI.loc[selected_indices, 'Time (s)'].values # s
    T_fullComp = T_fullComp - np.min(T_fullComp)
    Z_fullComp = dfI.loc[selected_indices, 'Z Tip - denoised'].values*1e-3 # µm
    F_fullComp = dfI.loc[selected_indices, 'Load - denoised'].values*1e3 # nN
    F_fullComp = F_fullComp - np.min(F_fullComp)

    delta_fullComp = Z_fullComp - Z0 #np.min(Z_fullComp)
    max_delta = np.max(delta_fullComp)
    
    if max_delta < 1:
        VALID = False
        print(gs.ORANGE + "Indentation too shallow" + gs.NORMAL)
        return
    
    # N_fullComp = len(F_fullComp)
    # F1, delta1 = np.median(F_fullComp[-N_fullComp//40:]), np.median(delta_fullComp[-N_fullComp//40:])

    R1 = R1 # Cell basal radius, µm
    H0 = Href - Z0 # Cell height, µm
    Rp = dI['R'] # Probe radius, µm
    V_comp = dI['V_comp']
    print('\nIndent ' + indentID)
    print(f'R1 = {R1:.1f} | H0 = {H0:.1f} | Rp = {Rp:.1f} | V_comp = {V_comp:.1f}')


    #### 4. Compute shape
    if tasks['Compute shape'] or not loaded_pickle:
        N_points = 200
        delta_values = np.linspace(0, max_delta, N_points) + 0.01
        shape_resDict, shape_resDf = css.CSS_sphere_general(R1, H0, Rp, delta_values)
        css.plot_contours(shape_resDict, shape_resDf, 
                          save = save_plots, save_path = figures_path, fig_name_prefix = indentID)
        poly_dict = css.plot_curves(shape_resDict, shape_resDf, return_polynomial_fits = True, 
                                    save = save_plots, save_path = figures_path, fig_name_prefix = indentID)
        

    #### 5. Fit tension
    if tasks['Fit tension'] or not loaded_pickle:
        P_alpha = poly_dict['P_alpha']
        P_Lc = poly_dict['P_Lc']
    
        i_thresh_Lc = ufun.findFirst(True, np.polyval(P_Lc, delta_fullComp) > 0.1)
    
        delta_comp_f = delta_fullComp[i_thresh_Lc:]
        F_comp_f = F_fullComp[i_thresh_Lc:]
    
        M = np.array([delta_comp_f, F_comp_f]).T
        M = M[M[:, 0].argsort()]
        [delta_comp_f, F_comp_f] = M.T
    
        alpha_comp_f = np.polyval(P_alpha, delta_comp_f)
        Lc_comp_f = np.polyval(P_Lc, delta_comp_f)
        Tension_comp_f = F_comp_f/(2*np.pi*Lc_comp_f)
    
        params_full, results_full = ufun.fitLineHuber(alpha_comp_f, Tension_comp_f)
        CI_full = results_full.conf_int(alpha=0.05)
        CIW_full = CI_full[:,1] - CI_full[:,0]
        
        filter_alpha = (alpha_comp_f>0.01) & (alpha_comp_f<0.1)
        params_filtered, results_filtered = ufun.fitLineHuber(alpha_comp_f[filter_alpha], Tension_comp_f[filter_alpha])
        CI_filtered = results_filtered.conf_int(alpha=0.05)
        CIW_filtered = CI_filtered[:,1] - CI_filtered[:,0]
        # print(results_filtered.summary())
        
        tension_dict = {'i_thresh_Lc': i_thresh_Lc,
                        'tension_coeffs_full': params_full,
                        'tension_ciw_full': CIW_full,
                        'tension_results_full': results_full,
                        'tension_coeffs_filtered': params_filtered,
                        'tension_ciw_filtered': CIW_filtered,
                        'tension_results_filtered': results_filtered,}
    
        #### 5.2 Plot tension
        figT, axesT = plt.subplots(3, 2, figsize = (10, 9), sharex='col')
        ax=axesT[0,0]
        ax.plot(delta_comp_f, alpha_comp_f*100, 'g-', lw=3)
        # ax.set_xlabel('$\\delta$ (µm)')
        ax.set_ylabel('$\\alpha$ (%)')
        ax.grid()
    
        ax=axesT[1,0]
        ax.plot(delta_comp_f, F_comp_f, 'r-', lw=1.5)
        # ax.set_xlabel('$\\delta$ (µm)')
        ax.set_ylabel('F (nN)')
        ax.grid()
    
        ax=axesT[2,0]
        ax.plot(delta_comp_f, Lc_comp_f, 'c-', lw=3)
        ax.plot(delta_comp_f, np.polyval(P_Lc, delta_comp_f), 'r--', lw=1)
        ax.set_xlabel('$\\delta$ (µm)')
        ax.set_ylabel('$L_c$ (µm)')
        ax.grid()
        
        
        ax=axesT[0,1]
        ax.plot(alpha_comp_f, Lc_comp_f, c='royalblue', ls='-', lw=3)
        # ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$L_c$ (µm)')
        ax.grid()
    
        ax=axesT[1,1]
        ax.plot(alpha_comp_f, F_comp_f, c='darkred', ls='-')
        # ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('F (nN)')
        ax.grid()
    
        ax=axesT[2,1]
        ax.plot(alpha_comp_f, Tension_comp_f)
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$F / 2\\pi.L_c$ (nN/µm)')
        ax.grid()
    
        s = f'Fit, all curve - F/$L_c$ = {params_full[1]:.3e}$\\alpha$ + {params_full[0]:.3e}'
        # s += f'$R^2$ = {r_squared:.3f}'
        ax.plot(alpha_comp_f, alpha_comp_f*params_full[1] + params_full[0], 'r--', label = s)
    
        s = f'Fit, $\\alpha$ in 1-10% - F/$L_c$ = {params_filtered[1]:.3e}$\\alpha$ + {params_filtered[0]:.3e}'
        # s += f'$R^2$ = {r_squared:.3f}'
        ax.plot(alpha_comp_f[filter_alpha], alpha_comp_f[filter_alpha]*params_filtered[1] + params_filtered[0], c='gold', ls='--', label = s)
    
        ax.legend()
    
        plt.tight_layout()
        plt.show()
        
        if save_plots:
            name = indentID + '_tensionFit'
            ufun.archiveFig(figT, name = name, figDir = figures_path)
            
    
        #### 6. Have a look at relaxation
        F0 = hertz_resDict['F_moy']*1e-3 # pN to nN
        F_relax = dfI[dfI['step'] == 1]['Load - denoised']*1e3 # µN to nN
        F_relax = F_relax - F0 # Reset the zero force
        N_relax = len(F_relax)
        F2 = np.median(F_relax[-N_relax//40:]) # nN
        
        delta_relax = dfI[dfI['step'] == 1]['Z Tip - denoised']*1e-3 - Z0 # nm to µm
        delta2 = np.median(delta_relax[-N_relax//40:]) # µm
        
        M = np.array([delta_relax, F_relax]).T
        M = M[M[:, 0].argsort()]
        [delta_relax, F_relax] = M.T
        
        alpha_relax = np.polyval(P_alpha, delta_relax)
        Lc_relax = np.polyval(P_Lc, delta_relax)
        Tension_relax = F_relax/(2*np.pi*Lc_relax)
        alpha2, Lc2 = np.polyval(P_alpha, delta2), np.polyval(P_Lc, delta2)
        Tension2 = F2/(2*np.pi*Lc2)
        
        tension_dict['T0_relax'] = Tension2
    
        
        #### 6.2 Plot tension - relaxation
        figR, axesR = plt.subplots(1, 3, figsize = (15, 5))
        
        ax=axesR[2]
        ax.plot(alpha_comp_f, Tension_comp_f, c=gs.colorList10[0], label='compression')
        ax.plot(alpha_relax, Tension_relax, c=gs.colorList10[3], label='relaxation')
        ax.plot(alpha2, Tension2, c=gs.colorList10[3], ls='', marker = 'o', ms=10, mec='k', label='relaxed state', zorder = 6)
        aa = np.linspace(0, np.max(alpha_comp_f), 10)
        s = f'Fit, $\\alpha$ in 1-10% - F/$L_c$ = {params_filtered[1]:.3e}$\\alpha$ + {params_filtered[0]:.3e}'
        ax.plot(aa, aa*params_filtered[1] + params_filtered[0], c='orange', ls='--', label = s)
        ax.hlines(y=Tension2, xmin=0, xmax=alpha2, color='red', ls='--', label = 'final tension')
        
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$F / 2\\pi.L_c$ (nN/µm)')
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.grid()
        ax.legend()
        
        ax=axesR[0]
        ax.plot(alpha_comp_f, Lc_comp_f, c=gs.colorList10[0], label='compression')
        ax.plot(alpha_relax, Lc_relax, c=gs.colorList10[3], label='relaxation')
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$L_c$ (µm)')
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.grid()
        ax.legend()
        
        ax=axesR[1]
        ax.plot(alpha_comp_f, F_comp_f, c=gs.colorList10[0], label='compression')
        ax.plot(alpha_relax, F_relax, c=gs.colorList10[3], label='relaxation')
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$F$ (nN)')
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.grid()
        ax.legend()
        
        figR.suptitle('Tension in compression & relaxation', size = 12)
    
        plt.tight_layout()
        plt.show()
        
        if save_plots:
            name = indentID + '_tensionRelax'
            ufun.archiveFig(figR, name = name, figDir = figures_path)
        
        
    #### SAVE JSON / PICKLE
    jar = [dI, hertz_resDict, shape_resDict, poly_dict, tension_dict]
    saveAsPickle(jar, pickle_path)
    # print(pickle_shape_path)
    saveAsPickle(hertz_resDict, pickle_hertz_path)
    saveAsPickle(shape_resDict, pickle_shape_path)
    saveAsPickle(poly_dict, pickle_poly_path)
    saveAsPickle(tension_dict, pickle_tension_path)
    
    otherSaveDir = os.path.join(cp.DirDataAnalysis, 'Chiaro', manipID, cellID)
    head, tail = os.path.split(otherSaveDir)
    if not os.path.exists(head):
        os.mkdir(head)
    if not os.path.exists(otherSaveDir):
        os.mkdir(otherSaveDir) 
        
    saveAsPickle(jar, os.path.join(otherSaveDir, indentID + '_results.pkl'))
    saveAsPickle(hertz_resDict,  pickle_hertz_path)
    saveAsPickle(shape_resDict,  pickle_shape_path)
    saveAsPickle(poly_dict,  pickle_poly_path)
    saveAsPickle(tension_dict,  pickle_tension_path)
        
    return(jar)

# %% Run the function on 1 file

# date, manip, cell, indent = '24-04-11', 'M1', 'C3', '001'
# date2 = dateFormat(date)
# specif1 = '_25um' # indenter size
# specif2 = '' # height above substrate

for strI in ['001', '002']:

    date, manip, cell, indent = '24-04-11', 'M1', 'C5', strI
    date2 = '24.04.11'
    specif1 = '_25um' # indenter size
    specif2 = '' # height above substrate
    
    mainDir = f'D://MagneticPincherData//Raw//{date2}_NanoIndent_ChiaroData{specif1}'
    manipDir = os.path.join(mainDir, f'{manip}')
    cellID = f'{date}_{manip}_P1_{cell}'
    indentID = f'{date}_{manip}_{cell}_I{int(indent)}'
    indent_path = os.path.join(manipDir, f'{cell}-cell//Indentations//{cell}-cell{specif2} Indentation_{indent}.txt')
    # indent_path = os.path.join(manipDir, f'{cell}//Indentations//{cell}{specif2} Indentation_{indent}.txt')
    figures_path = os.path.join(mainFigDir, date)
    
    date2 = dateFormat(date)
    fgeom_path = os.path.join(mainDir, f'{date}_cellGeometry.csv')
    dfG = pd.read_csv(fgeom_path)
    
    # Href = 20 # µm
    R1 = dfG.loc[dfG['cellID'] == cellID, 'R1'].values[0] # Cell basal radius, µm
    
    
    tasks = {'Fit Hertz':True,
             'Compute shape':True,
             'Fit tension':True}
    
    jar = indentations_apt(indent_path, date, manip, cell, indent,
                         R1, Href='glass-indent', save_plots=True, figures_path=figures_path, **tasks)

# %% Run the HERTZ analysis on many files

plt.ioff()

date, manip = '24-04-11', 'M2'
date2 = dateFormat(date)
specif1 = '_25um' # indenter size
specif2 = '' # height above substrate

Excluded = ['calib', 'test']

mainDir = f'D://MagneticPincherData//Raw//{date2}_NanoIndent_ChiaroData{specif1}'
manipDir = os.path.join(mainDir, f'{manip}')
cellDirList = [cd for cd in os.listdir(manipDir) if ((os.path.isdir(manipDir + '//' + cd)) and checkForbiddenWords(cd, Excluded))] #  and len(f) in [2, 7, 9]
print(cellDirList)

liste_df = []

for cd in cellDirList:
    if 'glass' in cd:
        cell = cd[:8]
    elif 'combo' in cd:
        cell = cd[:8]
    elif 'cell' in cd:
        cell = cd[:2]
    else:
        cell = cd[:2]
    cellDirPath = os.path.join(manipDir, cd)
    indentDirPath = os.path.join(cellDirPath, 'Indentations')
    indentFiles = [f for f in os.listdir(indentDirPath) if (('Indentation' in f ) and ('.txt' in f ))]
    for f in indentFiles:
        indent_path = os.path.join(indentDirPath, f)
        indent_root = '.'.join(f.split('.')[:-1])
        indent = indent_root[-3:]
        
        manipID = f'{date}_{manip}'
        cellID = f'{date}_{manip}_P1_{cell}'
        indentID = f'{date}_{manip}_{cell}_I{int(indent)}'
        # print(indentID)
        
        figures_path = manipDir #, indentID)

        indentations_hertz(indent_path, date, manip, cell, indent,
                               mode = 'dZmax', dZmax = 2, redo = True, 
                               save_plots=True, figures_path=figures_path)
        
plt.close('all')
plt.ion()

# %% Run the function on many files


date, manip = '24-04-11', 'M1'
date2 = dateFormat(date)
specif1 = '_25um' # indenter size
specif2 = '' # height above substrate

tasks = {'Fit Hertz':False,
         'Compute shape':False,
         'Fit tension':True}

Excluded = ['calib', 'test', 'C1', 'C2', 'glass', 'combo']
Compulsary = ['cell']

mainDir = f'D://MagneticPincherData//Raw//{date2}_NanoIndent_ChiaroData{specif1}'
manipDir = os.path.join(mainDir, f'{manip}')
cellDirList = [cd for cd in os.listdir(manipDir) if ((os.path.isdir(manipDir + '//' + cd)) and checkForbiddenWords(cd, Excluded))]
                                                                                           # and checkCompulsaryWords(cd, Compulsary))] #  and len(f) in [2, 7, 9]
print(cellDirList)

liste_df = []

for cd in cellDirList:
    cell = cd[:2]
    cellDirPath = os.path.join(manipDir, cd)
    indentDirPath = os.path.join(cellDirPath, 'Indentations')
    indentFiles = [f for f in os.listdir(indentDirPath) if (('Indentation' in f )and ('.txt' in f ))]
    for f in indentFiles:
        indent_path = os.path.join(indentDirPath, f)
        indent_root = '.'.join(f.split('.')[:-1])
        indent = indent_root[-3:]
        
        manipID = f'{date}_{manip}'
        cellID = f'{date}_{manip}_P1_{cell}'
        indentID = f'{date}_{manip}_{cell}_I{int(indent)}'
        # print(indentID)
        
        figures_path = os.path.join(mainFigDir, date)
        
        fgeom_path = os.path.join(mainDir, f'{date}_cellGeometry.csv')
        dfG = pd.read_csv(fgeom_path)

        # Href = 20 # µm
        R1 = dfG.loc[dfG['cellID'] == cellID, 'R1'].values[0] # Cell basal radius, µm
        
        if '-cell' in cd:
            Href = 'glass-indent'
        else:
            Href = 20
        
        jar = indentations_apt(indent_path, date, manip, cell, indent,
                              R1, Href = Href, save_plots=True, figures_path=figures_path, **tasks)
        
        if jar != None:
            [dI, hertz_resDict, shape_resDict, poly_dict, tension_dict] = jar
            
            # Validity criteria
            delta_Zstart = hertz_resDict['Z0_d'] - hertz_resDict['Z_start']
            delta_Zstop = hertz_resDict['Z_stop'] - hertz_resDict['Z0_d']
            hertz_resDict['valid_Z0'] = (delta_Zstart > 1) & (delta_Zstop > hertz_resDict['dZmax'])
            
            # Fit quality
            R_full = tension_dict['tension_results_full']
            CI_full = R_full.conf_int(alpha=0.05)
            CIW_full = CI_full[:,1] - CI_full[:,0]
            
            
            # dict for table
            syntetic_dict = {# ID
                             'date':date,
                             'manipID':manipID,
                             'cellID':cellID,
                             'indentID':indentID,
                             # General
                             'Rp':shape_resDict['Rp'],
                             'Href':hertz_resDict['Href'],
                             # Hertz
                             't0':dI['time'],
                             'Xpos':dI['X0'],
                             'Ypos':dI['Y0'],
                             'Zpos':dI['Z0'],
                             'V_comp':dI['V_comp'],
                             'T_rest':dI['dT_rest'],
                             'V_relax':dI['V_relax'],
                             'Z0':hertz_resDict['Z0_d'],
                             'Kh':hertz_resDict['K_d'],
                             'dZmax':hertz_resDict['dZmax'],
                             'valid_hertz_Z0':hertz_resDict['valid_Z0'],
                             # Shape
                             'H0':shape_resDict['H0'],
                             'R1':shape_resDict['R1'],
                             'A0':shape_resDict['A0'],
                             'V0':shape_resDict['V0'],
                             'max_r1':np.max(shape_resDict['r1']),
                             'min_R0':np.min(shape_resDict['R0']),
                             'max_R0':np.max(shape_resDict['R0']),
                             'max_phi':np.max(shape_resDict['phi']),
                             'max_delta':np.max(shape_resDict['delta']),
                             'max_area':np.max(shape_resDict['area']),
                             'max_alpha':np.max(shape_resDict['alpha']),
                             # Tension
                             'T0':tension_dict['tension_coeffs_filtered'][0],
                             'T0_ciw':tension_dict['tension_ciw_filtered'][0],
                             'Ka':tension_dict['tension_coeffs_filtered'][1],
                             'Ka_ciw':tension_dict['tension_ciw_filtered'][1],
                             'T0_relax':tension_dict['T0_relax'],
                             }
            
            syntetic_df = pd.DataFrame(syntetic_dict, index=[1])
            liste_df.append(syntetic_df)
        
global_df = pd.concat(liste_df, ignore_index = True)
SavePath = os.path.join(mainDir, manipID + '_syn.csv')
otherSavePath = os.path.join(cp.DirDataAnalysis, 'Chiaro', manipID, manipID + '_syn.csv')
global_df.to_csv(SavePath, index=False)
global_df.to_csv(otherSavePath, index=False)



