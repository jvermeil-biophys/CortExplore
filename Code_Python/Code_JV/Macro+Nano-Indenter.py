# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:22:07 2023

@author: JosephVermeil
"""

# %% Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import re

from scipy.optimize import curve_fit
from scipy.signal import find_peaks, argrelextrema

import UtilityFunctions as ufun
import GraphicStyles as gs

gs.set_default_options_jv()

# %% Combine the data from the nanoIndent & the Pincher

dataDir = "D://MagneticPincherData//Data_Analysis"


# %%% Format the pincher table

df_pincher = pd.read_csv(os.path.join(dataDir, 'MecaData_NanoIndent_23-11.csv'), sep=';')

df_pincher.insert(5, 'compressionSet', df_pincher['cellName'].apply(lambda x : int(x.split('-')[0][-1])))
df_pincher['cellRawNum'] = df_pincher['cellName'].apply(lambda x : x.split('_')[-1])
df_pincher['cellName'] = df_pincher['cellName'].apply(lambda x : x.split('-')[0][:-1])
df_pincher['cellID'] = df_pincher['date'] + '_' + df_pincher['cellName']
df_pincher['cellCode'] = df_pincher['cellRawNum'].apply(lambda x : x[:2] + x[3:])

dates = ['23-11-15']
Filters = np.array([df_pincher['date'].apply(lambda x : x in dates).values,
                    (df_pincher['cellName'] != 'M1_P1_C1').values,
                    (df_pincher['valid_Full'] == True).values,
                    ])
totalFilter = np.all(Filters, axis = 0)
df_pincher_f = df_pincher[totalFilter]


# %%% Format the indenter table

df_indenter = pd.read_csv(os.path.join(dataDir, 'ChiaroData_NanoIndent_23-11.csv'), sep=';')

df_indenter['cellName'] = df_indenter['manip'] + '_P1_' + df_indenter['cell']
df_indenter['cellID'] = df_indenter['date'] + '_' + df_indenter['cellName']

df_indenter['indentName'] = df_indenter['cellName'] + '_' + df_indenter['indent']
df_indenter['indentID'] = df_indenter['cellID'] + '_' + df_indenter['indent']

dates = ['23-11-15']
Filters = np.array([df_indenter['date'].apply(lambda x : x in dates).values,
                    (df_indenter['cellName'] != 'M1_P1_C1').values,
                    (df_indenter['Rsq2'] >= 0.7).values,
                    ])
totalFilter = np.all(Filters, axis = 0)
df_indenter_f = df_indenter[totalFilter]

# %%% Start plotting

# %%%% First Plot

fig1, axes1 = plt.subplots(2,1, figsize = (8, 6))
fig = fig1

parm = 'bestH0'
ax = axes1[0]
dictPlot = {'data' : df_pincher_f,
            'ax' : ax,
            'x' : 'cellCode',
            'y' : parm,
            'hue' : 'compressionSet'
            }
sns.boxplot(**dictPlot, fliersize = 0, width=0.6)
sns.swarmplot(**dictPlot, dodge = True, size=4, edgecolor='black', linewidth=1)
ax.set_ylim([0, ax.get_ylim()[1]])
ax.get_legend().remove()

parm = 'K2'
ax = axes1[1]
dictPlot = {'data' : df_indenter_f,
            'ax' : ax,
            'x' : 'cell',
            'y' : parm,
            }
# sns.boxplot(**dictPlot, fliersize = 0, width=0.6)
sns.swarmplot(**dictPlot, dodge = True, size=6, edgecolor='black', linewidth=1)
ax.set_ylim([0, ax.get_ylim()[1]])
# ax.get_legend().remove()

plt.show()

# %%%% Second Plot

fig2, axes2 = plt.subplots(2,1, figsize = (8, 6))
fig = fig2

parm = 'E_Full'
ax = axes2[0]
dictPlot = {'data' : df_pincher_f,
            'ax' : ax,
            'x' : 'cellCode',
            'y' : parm,
            'hue' : 'compressionSet'
            }
sns.boxplot(**dictPlot, fliersize = 0, width=0.6)
sns.swarmplot(**dictPlot, dodge = True, size=4, edgecolor='black', linewidth=1)
ax.set_ylim([0, ax.get_ylim()[1]])
ax.get_legend().remove()

parm = 'K2'
ax = axes2[1]
dictPlot = {'data' : df_indenter_f,
            'ax' : ax,
            'x' : 'cell',
            'y' : parm,
            }
sns.boxplot(**dictPlot, fliersize = 0, width=0.6)
sns.swarmplot(**dictPlot, dodge = True, size=4, edgecolor='black', linewidth=1)
ax.set_ylim([0, ax.get_ylim()[1]])
# ax.get_legend().remove()

plt.show()

# %%% Second style of plots : Average shit together


# %%%% Format the pincher table

df_pincher = pd.read_csv(os.path.join(dataDir, 'MecaData_NanoIndent_23-11.csv'), sep=';')

df_pincher.insert(5, 'compressionSet', df_pincher['cellName'].apply(lambda x : int(x.split('-')[0][-1])))
df_pincher['cellRawNum'] = df_pincher['cellName'].apply(lambda x : x.split('_')[-1])
df_pincher['cellName'] = df_pincher['cellName'].apply(lambda x : x.split('-')[0][:-1])
df_pincher['cellID'] = df_pincher['date'] + '_' + df_pincher['cellName']
df_pincher['cellCode'] = df_pincher['cellRawNum'].apply(lambda x : x[:2] + x[3:])

dates = ['23-11-15']
Filters = np.array([df_pincher['date'].apply(lambda x : x in dates).values,
                    (df_pincher['cellName'] != 'M1_P1_C1').values,
                    (df_pincher['valid_Full'] == True).values,
                    (df_pincher['compressionSet'] == 1).values,
                    ])
totalFilter = np.all(Filters, axis = 0)
df_pincher_f = df_pincher[totalFilter]

g = df_pincher_f.groupby('cellName')
df_pincher_g = g.agg({'bestH0':'mean', 'E_Full':'mean', 'compNum': 'count'}).reset_index()

# %%%% Format the indenter table

df_indenter = pd.read_csv(os.path.join(dataDir, 'ChiaroData_NanoIndent_23-11.csv'), sep=';')

df_indenter['cellName'] = df_indenter['manip'] + '_P1_' + df_indenter['cell']
df_indenter['cellID'] = df_indenter['date'] + '_' + df_indenter['cellName']

df_indenter['indentName'] = df_indenter['cellName'] + '_' + df_indenter['indent']
df_indenter['indentID'] = df_indenter['cellID'] + '_' + df_indenter['indent']

dates = ['23-11-15']
Filters = np.array([df_indenter['date'].apply(lambda x : x in dates).values,
                    (df_indenter['cellName'] != 'M1_P1_C1').values,
                    (df_indenter['Rsq2'] >= 0.7).values,
                    ])
totalFilter = np.all(Filters, axis = 0)
df_indenter_f = df_indenter[totalFilter]

g = df_indenter_f.groupby('cellName')
df_indenter_g = g.agg({'K2':'mean', 'indent': 'count'}).reset_index()



# %%%% Merge

df_merge_g = df_pincher_g.merge(df_indenter_g, on='cellName')
df_merge_g['k'] = df_merge_g['bestH0'] * df_merge_g['E_Full']

# %%% Start plotting

fig3, axes3 = plt.subplots(1,3, figsize = (15, 6))
fig = fig3

ax = axes3[0]
dictPlot = {'data' : df_merge_g,
            'ax' : ax,
            'x' : 'bestH0',
            'y' : 'K2',
            'hue' : 'cellName',
            'sizes':(10,10),
            'edgecolor':'k'
            }
sns.scatterplot(**dictPlot)

ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])

ax = axes3[1]
dictPlot = {'data' : df_merge_g,
            'ax' : ax,
            'x' : 'E_Full',
            'y' : 'K2',
            'hue' : 'cellName',
            'sizes':(10,10),
            'edgecolor':'k'
            }
sns.scatterplot(**dictPlot)

ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])

ax = axes3[2]
dictPlot = {'data' : df_merge_g,
            'ax' : ax,
            'x' : 'k',
            'y' : 'K2',
            'hue' : 'cellName',
            'sizes':(10,10),
            'edgecolor':'k'
            }
sns.scatterplot(**dictPlot)

ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])

plt.show()

# %% Cell indents

# %%% Analysis 1 by 1

# %%%% Pick a folder and open the indentations inside

C = 'C9'
dataDir = "D://MagneticPincherData//Raw//23.11.15_NanoIndent_ChiaroData//M1//" + C + "//Indentations"

R = 24.5
headerFormat = 'Time (s)	Load (uN)	Indentation (nm)	Cantilever (nm)	Piezo (nm)	Auxiliary'

def getLineWithString(lines, string):
    i = 0
    while string not in lines[i]:
        i+=1
    return(lines[i], i)
    

rawListFiles = os.listdir(dataDir)
listIndent = []
for f in rawListFiles:
    if f.endswith('.txt'):
        # indent_number = int(f.split('_')[-1][:3])
        fpath = os.path.join(dataDir, f)
        fopen = open(fpath, 'r')
        flines = fopen.readlines()
        s, i = getLineWithString(flines, headerFormat)
        metadataText = flines[:i]
        df_indent = pd.read_csv(fpath, skiprows = i-1, sep='\t')
        dirIndent = {'indent_name':f, 'path':fpath, 'metadata':metadataText, 'df':df_indent}
        
        R = float(getLineWithString(flines, 'Tip radius (um)')[0][16:])
        dirIndent['R'] = R
        
        text_start_times, j = getLineWithString(flines, 'Step absolute start times')
        text_end_times, j = getLineWithString(flines, 'Step absolute end times')
        start_times = np.array(text_start_times[30:].split(',')).astype(float)
        end_times = np.array(text_end_times[28:].split(',')).astype(float)
        
        dirIndent['ti'], dirIndent['tf'] = start_times, end_times
        
        listIndent.append(dirIndent)
        fopen.close()
        


# %%%% Define functions

def FitHertz_VJojo(Z, F, fraction = 0.4):
    early_points = 1000
    F_moy = np.average(F[:early_points])
    # F = F - F_moy
    F_std = np.std(F[:early_points])
    F_max = np.max(F)
    
    # Function
    def HertzFit(z, K, Z0):
        zeroInd = 1e-9 * np.ones_like(z)
        d = z - Z0
        d = np.where(d>0, d, zeroInd)
        f = (4/3) * K * (R**0.5) * (d**1.5) + F_moy
        return(f)
    
    #### First Fit Zone
    i_start1 = len(F) - ufun.findFirst(True, F[::-1] < F_moy + 1 * F_std)
    # i_start1 = len(F) - ufun.findFirst(True, F[::-1] < 0 + 1 * F_std)
    Z0_sup = Z[i_start1]

    # upper_threshold = F_moy + fraction * (F_max - F_moy)
    upper_threshold = F_moy + fraction * (F_max - F_moy)
    i_stop = ufun.findFirst(True, F >= upper_threshold)
    
    #### First Fit
    binf = [0, 0]
    bsup = [5000, Z0_sup]
    p0=[100, 1] # 0.99*Z0_sup
    [K1, Z01], covM1 = curve_fit(HertzFit, Z[i_start1:i_stop], F[i_start1:i_stop], p0=p0, bounds=(binf, bsup))
    Z_fit1 = np.linspace(Z01, Z[i_stop], 100)
    F_fit1 = HertzFit(Z_fit1, K1, Z01)
    
    #### Second Fit Zone
    i_start2 = ufun.findFirst(True, Z >= Z01)
    
    #### Second Fit
    binf = [0, 0]
    bsup = [5000, Z0_sup]
    p0=[K1, 1]
    
    [K2, Z02], covM2 = curve_fit(HertzFit, Z[i_start2:i_stop], F[i_start2:i_stop], p0=p0, bounds=(binf, bsup))
    Z_fit2 = np.linspace(Z02, Z[i_stop], 100)
    F_fit2 = HertzFit(Z_fit2, K2, Z02)
    
    
    #### Third Fit
    arraySteps = np.array([-0.25, +0.25])
    Z0_error = arraySteps + Z02
    # print(Z0_test)
    resError = []
    for i in range(len(arraySteps)):
        Z0 = Z0_error[i]
        # function
        def HertzFit_1parm(z, K):
            zeroInd = 1e-9 * np.ones_like(z)
            d = z - Z0
            d = np.where(d>0, d, zeroInd)
            f = (4/3) * K * R**0.5 * (d)**1.5 + F_moy
            return(f)
        # zone
        i_start_test = len(Z) - ufun.findFirst(True, Z[::-1] <= Z0)
        # print(Z0, np.min(Z[i_start_test:i_stop]))
        # fit
        binf = [0]
        bsup = [np.inf]
        p0=[K2]
        [K_error], covM_error = curve_fit(HertzFit_1parm, Z[i_start_test:i_stop], F[i_start_test:i_stop], p0=p0, bounds=(binf, bsup))
        resError.append(K_error)
        
    # print(resError)
    K_error = np.array(resError)
    print(Z0_error)
    print(K_error)
    
    #### Results
    results = {'F_moy':F_moy,
               'F_std':F_std,
               'F_max':F_max,
               'i_start1':i_start1,
               'i_start2':i_start2,
               'i_stop':i_stop,
               'K1':K1,
               'Z01':Z01,
               'covM1':covM1,
               'Z_fit1':Z_fit1,
               'F_fit1':F_fit1,
               'K2':K2,
               'Z02':Z02,
               'covM2':covM2,
               'Z_fit2':Z_fit2,
               'F_fit2':F_fit2,
               'Z0_error':Z0_error,
               'K_error':K_error,
               }
    
    return(results)


# %%%% Analysis

for I in listIndent[:]:
    df = I['df']
    df['Z Tip (nm)'] = df['Piezo (nm)'] - df['Cantilever (nm)']
    R = I['R']
    
    fig, axes = plt.subplots(1,2, figsize = (12,6))
    ax=axes[0]
    ax.plot(df['Time (s)'], df['Piezo (nm)'], 'b-', label='Displacement')
    ax.plot(df['Time (s)'], df['Z Tip (nm)'], color='gold', ls='-', label='Z Tip')
    ax.plot(df['Time (s)'], df['Cantilever (nm)'], 'g-', label='Deflection')
    ax.plot(df['Time (s)'], df['Indentation (nm)'], 'r-', label='Indentation')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (nm)')
    ax.legend()
    
    
    i = 0
    for ti, tf in zip(I['ti'][::-1], I['tf'][::-1]):
        i += 1
        step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
    
    
    try:
        ti, tf = I['ti'][0], I['tf'][0]
        compression_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
        Z = df.loc[step_indices, 'Z Tip (nm)'].values
        F = df.loc[step_indices, 'Load (uN)'].values
        # Z = df.loc[step_indices, 'Z Tip (nm)']
        # F = df.loc[step_indices, 'Load (uN)']
        results = FitHertz_VJojo(Z * 1e-3, F * 1e6, fraction = 0.6)  # nm to µm # µN to pN
        
        ax=axes[1]
        colorList = gs.colorList10[:len(I['ti'])]
        i = 0
        for ti, tf, c in zip(I['ti'][::-1], I['tf'][::-1], colorList[::-1]):
            i += 1
            step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
            # print(step_indices)
            ax.plot(df.loc[step_indices, 'Z Tip (nm)'], df.loc[step_indices, 'Load (uN)'], color=c)
        
        ax.set_xlabel('Distance (nm)')
        ax.set_ylabel('Load (uN)')
        
        ax.axvline(Z[results['i_start1']], ls = '--', color = 'yellow', zorder = 4)
        ax.axvline(Z[results['i_start2']], ls = '--', color = 'orange', zorder = 4)
        ax.axvline(Z[results['i_stop']], ls = '--', color = 'red', zorder = 4)
        ax.axhline((results['F_moy'])*1e-6, ls = '-', color = 'k', zorder = 4)
        ax.axhline((results['F_moy']+results['F_std'])*1e-6, ls = '--', color = 'k', zorder = 4)
        ax.plot(results['Z_fit1']*1e3, results['F_fit1']*1e-6, ls = '-', color = 'orange', zorder = 5,
                label = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\nEeff = {:.0f} +/- {:.0f} Pa'.format(results['Z01'], results['covM1'][1,1]**0.5,
                                                                                                  results['K1'], results['covM1'][0,0]**0.5))
        ax.plot(results['Z_fit2']*1e3, results['F_fit2']*1e-6, ls = '-', color = 'red', zorder = 5,
                label = 'Second fit\nZ0 = {:.2f} +/- {:.2f} µm\nEeff = {:.0f} +/- {:.0f} Pa'.format(results['Z02'], results['covM2'][1,1]**0.5,
                                                                                                    results['K2'], results['covM2'][0,0]**0.5))
        Z0_error, K_error = results['Z0_error'], results['K_error']
        for j in range(len(Z0_error)):
            Z0, K = Z0_error[j], K_error[j]
            def Hertz(z, Z0, K):
                zeroInd = 1e-9 * np.ones_like(z)
                d = z - Z0
                d = np.where(d>0, d, zeroInd)
                f = (4/3) * K * R**0.5 * (d)**1.5 + results['F_moy']
                return(f)
            Zplot = np.linspace(Z0*1e3, Z[results['i_stop']], 100) * 1e-3
            Ffit = Hertz(Zplot, Z0, K)
            ax.plot(Zplot*1e3, Ffit*1e-6, ls = '--', color = 'cyan', zorder = 4)
        dZ0 = np.abs(Z0_error[1] - Z0_error[0])/2
        ax.plot([], [], ls = '--', color = 'cyan', zorder = 4, label = 'Effect of error on Z0\n+/-{:.2f} µm =>\nEeff = {:.0f}~{:.0f} Pa'.format(dZ0, K_error[0], K_error[1]))
    
        
        
    except:
        pass
    
    ax.legend()
    
    fig.suptitle(I['indent_name'])
    plt.tight_layout()
    plt.show()
    ufun.simpleSaveFig(fig, I['indent_name'], dataDir, '.png', 150)
    
# %%% Multiple files analysis

# %%%% Function 

def FitHertz_V1(Z, F, R, mode = 'dZmax', fractionF = 0.4, dZmax = 3):
    early_points = 1000
    F_moy = np.average(F[:early_points])
    F_std = np.std(F[:early_points])
    F_max = np.max(F)
    
    Z_start = np.min(Z)
    Z_stop = np.max(Z)
    
    # Function
    def HertzFit(z, K, Z0):
        zeroInd = 1e-9 * np.ones_like(z)
        d = z - Z0
        d = np.where(d>0, d, zeroInd)
        f = (4/3) * K * (R**0.5) * (d**1.5) + F_moy
        return(f)
    
    #### First Fit Zone
    # i_start1 = len(F) - ufun.findFirst(True, F[::-1] < F_moy + 1 * F_std)
    # i_start1 = len(F) - ufun.findFirst(True, F[::-1] < 0 + 1 * F_std)
    # Z0_sup = Z[i_start1]
    Z0_sup = np.max(Z)
    
    if mode == 'fractionF':
        # upper_threshold = F_moy + fraction * (F_max - F_moy)
        upper_threshold = F_moy + fractionF * (F_max - F_moy)
        i_stop = ufun.findFirst(True, F >= upper_threshold)
    
    #### First Fit
    binf = [0, 0]
    bsup = [5000, Z0_sup]
    p0=[100, 1] # 0.99*Z0_sup
    # Z1, F1 = Z[i_start1:i_stop], F[i_start1:i_stop]
    Z1, F1 = Z, F
    [K1, Z01], covM1 = curve_fit(HertzFit, Z1, F1, p0=p0, bounds=(binf, bsup))
    Z_fit1 = np.linspace(np.min(Z), np.max(Z), 200)
    F_fit1 = HertzFit(Z_fit1, K1, Z01)
    Rsq1 = ufun.get_R2(F1, HertzFit(Z1, K1, Z01))
    
    #### Second Fit Zone
    i_start = ufun.findFirst(True, Z >= Z01-0.5)
    if mode == 'fractionF':
        upper_threshold = F_moy + fractionF * (F_max - F_moy)
        i_stop = ufun.findFirst(True, F >= upper_threshold)
    elif mode == 'dZmax':
        i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z01 + dZmax) - 1
    
    #### Second Fit
    binf = [0, 0]
    bsup = [5000, Z0_sup]
    p0=[K1, 1]
    
    Z2, F2 = Z[i_start:i_stop], F[i_start:i_stop]
    [K2, Z02], covM2 = curve_fit(HertzFit, Z2, F2, p0=p0, bounds=(binf, bsup))
    Z_fit2 = np.linspace(Z02, Z[i_stop], 100)
    F_fit2 = HertzFit(Z_fit2, K2, Z02)
    Rsq2 = ufun.get_R2(F2, HertzFit(Z2, K2, Z02))
    
    
    #### Third Fit
    arraySteps = np.array([-0.25, +0.25])
    Z0_error = arraySteps + Z02
    # print(Z0_test)
    resError = []
    for i in range(len(arraySteps)):
        Z0 = Z0_error[i]
        # function
        def HertzFit_1parm(z, K):
            zeroInd = 1e-9 * np.ones_like(z)
            d = z - Z0
            d = np.where(d>0, d, zeroInd)
            f = (4/3) * K * R**0.5 * (d)**1.5 + F_moy
            return(f)
        # zone
        i_start_test = len(Z) - ufun.findFirst(True, Z[::-1] <= Z0)
        # print(Z0, np.min(Z[i_start_test:i_stop]))
        # fit
        binf = [0]
        bsup = [np.inf]
        p0=[K2]
        [K_error], covM_error = curve_fit(HertzFit_1parm, Z[i_start_test:i_stop], F[i_start_test:i_stop], p0=p0, bounds=(binf, bsup))
        resError.append(K_error)
        
    # print(resError)
    K_error = np.array(resError)
    
    #### Results
    results = {'Z_start':Z_start,
               'Z_stop':Z_stop,
               'F_moy':F_moy,
               'F_std':F_std,
               'F_max':F_max,
               # 'i_start1':i_start1,
               # 'i_start2':i_start2,
               'i_start':i_start,
               'i_stop':i_stop,
               'K1':K1,
               'Z01':Z01,
               'covM1':covM1,
               'Rsq1':Rsq1,
               'Z_fit1':Z_fit1,
               'F_fit1':F_fit1,
               'K2':K2,
               'Z02':Z02,
               'covM2':covM2,
               'Rsq2':Rsq2,
               'Z_fit2':Z_fit2,
               'F_fit2':F_fit2,
               'Z0_error':Z0_error,
               'K_error':K_error,
               }
    
    return(results)



def plotFitHertz_V1(dictIndent, fitResults, show = True, save = False, savePath = ''):
    plt.ioff()
    df = dictIndent['df']
    df['Z Tip (nm)'] = df['Piezo (nm)'] - df['Cantilever (nm)']
    # R = dictIndent['R']
    fig, axes = plt.subplots(1,2, figsize = (12,6))
    ax=axes[0]
    ax.plot(df['Time (s)'], df['Piezo (nm)'], 'b-', label='Displacement')
    ax.plot(df['Time (s)'], df['Z Tip (nm)'], color='gold', ls='-', label='Z Tip')
    ax.plot(df['Time (s)'], df['Cantilever (nm)'], 'g-', label='Deflection')
    ax.plot(df['Time (s)'], df['Indentation (nm)'], 'r-', label='Indentation')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (nm)')
    ax.legend()
    
    ax=axes[1]
    colorList = gs.colorList10[:len(dI['ti'])]
    i = 0
    for ti, tf, c in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]):
        i += 1
        step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
        # print(step_indices)
        ax.plot(df.loc[step_indices, 'Z Tip (nm)'], df.loc[step_indices, 'Load (uN)'], color=c)
    
    ax.set_xlabel('Distance (nm)')
    ax.set_ylabel('Load (uN)')
    
    # ax.axvline(Z[fitResults['i_start1']], ls = '--', color = 'yellow', zorder = 4)
    # ax.axvline(Z[fitResults['i_start2']], ls = '--', color = 'orange', zorder = 4)
    ax.axvline(fitResults['Z01']*1000, ls = '--', color = 'orange', zorder = 4)
    ax.axvline(Z[fitResults['i_stop']], ls = '--', color = 'red', zorder = 4)
    ax.axhline((fitResults['F_moy'])*1e-6, ls = '-', color = 'k', zorder = 4)
    ax.axhline((fitResults['F_moy']+fitResults['F_std'])*1e-6, ls = '--', color = 'k', zorder = 4)
    labelFit1 = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\nEeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(fitResults['Z01'], 
                                                                                                   fitResults['covM1'][1,1]**0.5,
                                                                                                   fitResults['K1'], 
                                                                                                   fitResults['covM1'][0,0]**0.5,
                                                                                                   fitResults['Rsq1'],)
    ax.plot(fitResults['Z_fit1']*1e3, fitResults['F_fit1']*1e-6, 
            ls = '-', color = 'orange', zorder = 5, label = labelFit1)
    labelFit2 = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\nEeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(fitResults['Z02'], 
                                                                                           fitResults['covM2'][1,1]**0.5,
                                                                                           fitResults['K2'], 
                                                                                           fitResults['covM2'][0,0]**0.5,
                                                                                           fitResults['Rsq2'],)
    ax.plot(fitResults['Z_fit2']*1e3, fitResults['F_fit2']*1e-6, 
            ls = '-', color = 'red', zorder = 5, label = labelFit2)
    
    Z0_error, K_error = fitResults['Z0_error'], fitResults['K_error']
    for j in range(len(Z0_error)):
        Z0, K = Z0_error[j], K_error[j]
        def Hertz(z, Z0, K):
            zeroInd = 1e-9 * np.ones_like(z)
            d = z - Z0
            d = np.where(d>0, d, zeroInd)
            f = (4/3) * K * R**0.5 * (d)**1.5 + fitResults['F_moy']
            return(f)
        Zplot = np.linspace(Z0*1e3, Z[fitResults['i_stop']], 100) * 1e-3
        Ffit = Hertz(Zplot, Z0, K)
        ax.plot(Zplot*1e3, Ffit*1e-6, ls = '--', color = 'cyan', zorder = 4)
    dZ0 = np.abs(Z0_error[1] - Z0_error[0])/2
    labelFitErrors = 'Effect of error on Z0\n+/-{:.2f} µm =>\nEeff = {:.0f}~{:.0f} Pa'.format(dZ0, K_error[0], K_error[1])
    ax.plot([], [], ls = '--', color = 'cyan', zorder = 4, label = labelFitErrors)

    ax.legend()

    fig.suptitle(dI['indent_name'])
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close('all')
    if save:
        ufun.simpleSaveFig(fig, dI['indent_name'], savePath, '.png', 150)
        
    plt.ion()
    

# %%%% Script


mainDir = 'D://MagneticPincherData//Raw//23.11.15_NanoIndent_ChiaroData//M1//'
fileList = [f for f in os.listdir(mainDir) if (os.path.isdir(mainDir + '//' + f) and len(f) == 2)]

date = '23-11-15'
manip = 'M1'

resDir = {'date':[],
          'manip':[],
          'cell':[],
          'indent':[],
          'indentID':[],
          # 'path':[],
          'time':[],
          #
          'Z_start':[],
          'Z_stop':[],
          #
          'F_moy':[],
          'F_std':[],
          'F_max':[],
          #
          'K1':[],
          'Z01':[],
          'Rsq1':[],
          'K2':[],
          'Z02':[],
          'Rsq2':[],
          }

for f in fileList:
    dataDir = os.path.join(mainDir, f, 'indentations')
    cell = f
    
    
    # 1. Read all the Indentation files for that cell
    headerFormat = 'Time (s)	Load (uN)	Indentation (nm)	Cantilever (nm)	Piezo (nm)	Auxiliary'
    
    def getLineWithString(lines, string):
        i = 0
        while string not in lines[i]:
            i+=1
        return(lines[i], i)
        
    rawListFiles = os.listdir(dataDir)
    listIndent = []
    for f in rawListFiles:
        if f.endswith('.txt'):
            indent_number = int(f[:-4].split('_')[-1])
            indent = f'I{indent_number:.0f}'
            fpath = os.path.join(dataDir, f)
            fopen = open(fpath, 'r')
            flines = fopen.readlines()
            s, i = getLineWithString(flines, headerFormat)
            metadataText = flines[:i]
            df_indent = pd.read_csv(fpath, skiprows = i-1, sep='\t')
            dict_indent = {'indent_name':f, 'path':fpath, 'metadata':metadataText, 'df':df_indent}
            
            dict_indent['date'] = date
            dict_indent['manip'] = manip
            dict_indent['cell'] = cell
            dict_indent['indent'] = indent
            
            time = flines[0].split('\t')[3]
            dict_indent['time'] = time
            R = float(getLineWithString(flines, 'Tip radius (um)')[0][16:])
            dict_indent['R'] = R
            
            text_start_times, j = getLineWithString(flines, 'Step absolute start times')
            text_end_times, j = getLineWithString(flines, 'Step absolute end times')
            start_times = np.array(text_start_times[30:].split(',')).astype(float)
            end_times = np.array(text_end_times[28:].split(',')).astype(float)
            
            dict_indent['ti'], dict_indent['tf'] = start_times, end_times
            
            listIndent.append(dict_indent)
            fopen.close()

    # 2. Analyze them
    for dI in listIndent[:]:
        df = dI['df']
        df['Z Tip (nm)'] = df['Piezo (nm)'] - df['Cantilever (nm)']
        R = dI['R']
        
        i = 0
        for ti, tf in zip(dI['ti'][::-1], dI['tf'][::-1]):
            i += 1
            step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
        
        try:
            ti, tf = dI['ti'][0], dI['tf'][0]
            compression_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
            Z = df.loc[step_indices, 'Z Tip (nm)'].values
            F = df.loc[step_indices, 'Load (uN)'].values
            fitResults = FitHertz_V1(Z * 1e-3, F * 1e6, R, mode = 'dZmax', dZmax = 3)  # nm to µm # µN to pN
            
            plotFitHertz_V1(dI, fitResults, show=True, save=True, savePath=mainDir)
            
            resDir['date'].append(date)
            resDir['manip'].append(manip)
            resDir['cell'].append(cell)
            resDir['indent'].append(dI['indent'])
            resDir['indentID'].append(f'{date}_{manip}_{cell}_{indent}')
            # resDir['path'].append(fpath)
            resDir['time'].append(dI['time'])
            resDir['Z_start'].append(fitResults['Z_start'])
            resDir['Z_stop'].append(fitResults['Z_stop'])
            resDir['F_moy'].append(fitResults['F_moy'])
            resDir['F_std'].append(fitResults['F_std'])
            resDir['F_max'].append(fitResults['F_max'])
            resDir['K1'].append(fitResults['K1'])
            resDir['Z01'].append(fitResults['Z01'])
            resDir['Rsq1'].append(fitResults['Rsq1'])
            resDir['K2'].append(fitResults['K2'])
            resDir['Z02'].append(fitResults['Z02'])
            resDir['Rsq2'].append(fitResults['Rsq2'])

        except:
            pass
        
resDf = pd.DataFrame(resDir)
resDf.to_csv(f'{mainDir}//results.csv', sep=';')

# %% VCI Macro-indents
# %%% Imports

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


# %%% First one

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

# %%% Second one

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

        
# %%% Third one

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
        