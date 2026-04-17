# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:27:15 2025

@author: josep
"""

# %% > Imports and constants

#### Main imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import os
import re
import sys
import time
import random
import numbers
import warnings
import itertools
import matplotlib

from cycler import cycler
from scipy import interpolate
from scipy import odr
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
from matplotlib.gridspec import GridSpec
from scipy.stats import f_oneway, shapiro, mannwhitneyu
from scipy.optimize import curve_fit

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)

import ArticlePlotMaker as apm
import UtilityFunctions as ufun
import TrackAnalyser as taka
import TrackAnalyser_V2 as taka2
import TrackAnalyser_V3 as taka3

#### Potentially useful lines of code
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# cp.DirDataFigToday

#### Pandas
pd.set_option('display.max_columns', None)
# pd.reset_option('display.max_columns')
pd.set_option('display.max_rows', None)
pd.reset_option('display.max_rows')

#### Graphic options
cm_in = 2.52
apm.setGraphicOptions(mode = 'screen', 
                      palette = 'Set2', 
                      colorList = apm.cL_Set21)

figDir = 'C:/Users/josep/Desktop/Papier/NewFigs'


# %% > Data import & export

# %%% MecaData_Phy
# MecaData_Phy = taka3.getMergedTable('MecaData_Physics')
# MecaData_Phy2 = taka3.getMergedTable('MecaData_Physics_V2')
MecaData_Phy3 = taka3.getMergedTable('MecaData_Physics_V3')

# MecaData_Phy2 = MecaData_Phy2.dropna(axis=0, subset='date')
MecaData_Phy3 = MecaData_Phy3.dropna(axis=0, subset='date')

MecaData_Phy = MecaData_Phy3

 # %%% Check content

print('Dates')
print([x for x in MecaData_Phy['date'].unique()])
print('')

print('Cell types')
print([x for x in MecaData_Phy['cell type'].unique()])
print('')

print('Cell subtypes')
print([x for x in MecaData_Phy['cell subtype'].unique()])
print('')

print('Drugs')
print([x for x in MecaData_Phy['drug'].unique()])
print('')

print('Substrates')
print([x for x in MecaData_Phy['substrate'].unique()])
print('')

print('Resting Fields')
print([x for x in MecaData_Phy['normal field'].unique()])
print('')

CountByCond, CountByCell = apm.makeCountDf(MecaData_Phy, 'date')

# %% Developping New Thickness & Stiffness Plots 2025

# %%% Investigate the median num of comps for dates

df = MecaData_Phy
dates = df['date'].unique()
medians = []
for date in dates:
    df_date = df[df['date']==date]
    groups = df_date.groupby('cellID').agg({'compNum':'max'})
    med = int(np.median(groups.compNum.values))
    medians.append(med)
    
df_res = pd.DataFrame({'date':dates, 'medianCompNum':medians})
   
# 	  date	     medianCompNum
# 0	  23-02-16	 10
# 3	  23-03-16	 8
# 6	  23-04-26	 8
# 16  24-12-11	 20

    
# %%% Tests on diverse dates

# %%%% 1. Standard plots E400

apm.set_defense_options_jv(palette = 'Set2')

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
# figname = 'bestH0' + drugSuffix
XCol = 'bestH0'
YCol = 'E_f_<_400'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x in ['23-03-16'])),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           ]

df_f = apm.filterDf(df, Filters)
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)

# Order
co_order = []

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(2,1, figsize=(17/apm.cm_in,20/apm.cm_in), sharex=True)

# LinLog
ax = axes[0]
ax.set_xscale('log')
ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 0.5)


sns.scatterplot(ax = ax, x=df_f['bestH0'].values, y=df_f['E_f_<_400'].values/1000, 
                marker = 'o', s = 15, color = 'gray', alpha = 0.5)
Xfit, Yfit = np.log10(df_f['bestH0'].values), np.log10(df_f['E_f_<_400'].values/1000)
# [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
# A, k = np.exp(b), a
# R2 = w_results.rsquared
# Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
# Yplot = A * Xplot**k
# ax.plot(Xplot, Yplot, ls = '--', c = 'k',
#         label = 'Fit $y = A.x^k$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + f'\n$R^2$  = {R2:.2f}')

[b, a], results = ufun.fitLine(Xfit, Yfit)
A, k = np.exp(b), a
R2 = results.rsquared
pval = results.pvalues[1]
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 1.5,
        label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')

ax.legend(fontsize = 9, loc = 'lower left')
ax.set_title(f'All compressions - N = {len(Xfit)}')
ax.set_ylabel('$E_{400}$ (kPa)')
ax.set_xlabel('')

color = apm.cL_Set2[0]

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 0.5)

sns.scatterplot(ax = ax, x=df_plot['bestH0'].values, y=df_plot['E_f_<_400_wAvg'].values/1000, 
                marker = 'o', s = 45, color = color, alpha = 0.5)
Xfit, Yfit = np.log10(df_plot['bestH0'].values), np.log10(df_plot['E_f_<_400_wAvg'].values/1000)
# [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
# A, k = np.exp(b), a
# Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
# Yplot = A * Xplot**k
# ax.plot(Xplot, Yplot, ls = '--', c = 'k',
#         label = 'Fit $y = A.x^k$' + '\n' + f'A = {A:.1e}\nk  = {k:.2f}')

[b, a], results = ufun.fitLine(Xfit, Yfit)
A, k = np.exp(b), a
R2 = results.rsquared
pval = results.pvalues[1]
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 2.5,
        label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')

ax.legend(fontsize = 9, loc = 'lower left')
ax.set_title(f'Average per cell - N = {len(Xfit)}')
ax.set_ylabel('$E_{400}$ (kPa)')
ax.set_xlabel('$H_0$ (nm)')

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    ax.set_xlim([50, 1100])

# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
figSubDir = 'E-h'
name = 'E400_vs_h0_V2'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%%% 2. Split per cell

apm.set_defense_options_jv(palette = 'Set2')

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
# figname = 'bestH0' + drugSuffix
XCol = 'bestH0'
YCol = 'E_f_<_400'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x in ['23-03-16', '23-04-20'])),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           ]

df_f = apm.filterDf(df, Filters)
df_f['cellCode'] = df_f['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])
df_f['cellNum'] = df_f['cellCode'].apply(lambda x : int(x[1:]))
CID_list = df_f['cellID'].unique()

CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)

# Order
co_order = []

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

dictFit = {'cellID':[], 'A':[], 'k':[], 'pv':[], 'R2':[], 'H_logmean':[], 'E_logmean':[], 'NLR_mean':[]}

# Plot
for cid in CID_list:
    df_cell = df_f[df_f['cellID'] == cid]
    
    fig, axes = plt.subplots(1,1, figsize=(17/apm.cm_in, 15/apm.cm_in), sharex=True)
    
    # LinLog
    ax = axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    # fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
    #                 XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
    #                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
    #                 figSizeFactor = 1, markersizeFactor = 0.5)
    
    
    sns.scatterplot(ax = ax, x=df_cell['bestH0'].values, y=df_cell['E_f_<_400'].values/1000, 
                    marker = 'o', s = 15, alpha = 0.5)
    Xfit, Yfit = np.log10(df_cell['bestH0'].values), np.log10(df_cell['E_f_<_400'].values/1000)
    # [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
    # A, k = np.exp(b), a
    # R2 = w_results.rsquared
    # Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    # Yplot = A * Xplot**k
    # ax.plot(Xplot, Yplot, ls = '--', c = 'k',
    #         label = 'Fit $y = A.x^k$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + f'\n$R^2$  = {R2:.2f}')
    
    [b, a], results = ufun.fitLine(Xfit, Yfit)
    A, k = np.exp(b), a
    R2 = results.rsquared
    pval = results.pvalues[1]
    Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = A * Xplot**k
    ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 1.5,
            label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                    f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
    
    ax.legend(fontsize = 9, loc = 'lower left')
    ax.set_title('DMSO')
    ax.set_ylabel('$E_{400}$ (kPa)')
    ax.set_xlabel('')
    
    
    ax.grid(visible=True, which='major', axis='both')
    ax.set_xlim([50, 1100])
    
    # Show
    plt.tight_layout()
    plt.show()
    
    # Count
    CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)


# %%%% 3. Split per cell 2

apm.set_defense_options_jv(palette = 'Set2')

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
# figname = 'bestH0' + drugSuffix
XCol = 'bestH0'
YCol = 'E_f_<_400'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x in ['23-03-16', '23-04-20'])),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           ]

df_f = apm.filterDf(df, Filters)
df_f['cellCode'] = df_f['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])
df_f['cellNum'] = df_f['cellCode'].apply(lambda x : int(x[1:]))
CID_list = df_f['cellID'].unique()

CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)

# Order
co_order = []

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

dictFit = {'cellID':[], 'A':[], 'k':[], 'pv':[], 'R2':[], 'H_logmean':[], 'E_logmean':[], 'NLR_mean':[]}
k_list = []

fig, axes = plt.subplots(1,1, figsize=(17/apm.cm_in, 15/apm.cm_in), sharex=True)


# Plot
for cid in CID_list:
    df_cell = df_f[df_f['cellID'] == cid]
    if len(df_cell) >= 6:
    
        
        
        # LinLog
        ax = axes
        ax.set_xscale('log')
        ax.set_yscale('log')
        # fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
        #                 XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
        #                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
        #                 figSizeFactor = 1, markersizeFactor = 0.5)
        
        
        
        Xfit, Yfit = np.log10(df_cell['bestH0'].values), np.log10(df_cell['E_f_<_400'].values/1000)
        # [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
        # A, k = np.exp(b), a
        # R2 = w_results.rsquared
        # Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
        # Yplot = A * Xplot**k
        # ax.plot(Xplot, Yplot, ls = '--', c = 'k',
        #         label = 'Fit $y = A.x^k$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + f'\n$R^2$  = {R2:.2f}')
        
        [b, a], results = ufun.fitLine(Xfit, Yfit)
        A, k = np.exp(b), a
        R2 = results.rsquared
        pval = results.pvalues[1]
        Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
        Yplot = A * Xplot**k
        
        if pval < 0.05:
            sns.scatterplot(ax = ax, x=df_cell['bestH0'].values, y=df_cell['E_f_<_400'].values/1000, 
                            marker = 'o', s = 15, alpha = 0.9)
            ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 1.5,
                    label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                            f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
            k_list.append(k)
        
        ax.legend(fontsize = 9, loc = 'lower left').set_visible(False)
        
        ax.set_ylabel('$E_{400}$ (kPa)')
        ax.set_xlabel('')
        
        
        ax.grid(visible=True, which='major', axis='both')
        ax.set_xlim([50, 1100])

mean_k = np.mean(k_list)
ax.set_title(f'DMSO - Mean k = {mean_k:.2f}')
        
# Show
plt.tight_layout()
plt.show()


# %%%% 4. Split per date 1

apm.set_defense_options_jv(palette = 'Set2')

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
# figname = 'bestH0' + drugSuffix
XCol = 'bestH0'
YCol = 'E_f_<_400'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x in ['23-03-16', '23-04-20'])),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           ]

df_f = apm.filterDf(df, Filters)
df_f['cellCode'] = df_f['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])
df_f['cellNum'] = df_f['cellCode'].apply(lambda x : int(x[1:]))
CID_list = df_f['cellID'].unique()

CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)

# Order
co_order = []

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

dictFit = {'cellID':[], 'A':[], 'k':[], 'pv':[], 'R2':[], 'H_logmean':[], 'E_logmean':[], 'NLR_mean':[]}
k_list = []

fig, axes = plt.subplots(1,1, figsize=(17/apm.cm_in, 15/apm.cm_in), sharex=True)


# Plot
for cid in CID_list:
    df_cell = df_f[df_f['cellID'] == cid]
    if len(df_cell) >= 6:
    
        
        
        # LinLog
        ax = axes
        ax.set_xscale('log')
        ax.set_yscale('log')
        # fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
        #                 XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
        #                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
        #                 figSizeFactor = 1, markersizeFactor = 0.5)
        
        
        
        Xfit, Yfit = np.log(df_cell['bestH0'].values), np.log(df_cell['E_f_<_400'].values/1000)
        # [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
        # A, k = np.exp(b), a
        # R2 = w_results.rsquared
        # Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
        # Yplot = A * Xplot**k
        # ax.plot(Xplot, Yplot, ls = '--', c = 'k',
        #         label = 'Fit $y = A.x^k$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + f'\n$R^2$  = {R2:.2f}')
        
        [b, a], results = ufun.fitLine(Xfit, Yfit)
        A, k = np.exp(b), a
        R2 = results.rsquared
        pval = results.pvalues[1]
        Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
        Yplot = A * Xplot**k
        
        if pval < 0.05:
            sns.scatterplot(ax = ax, x=df_cell['bestH0'].values, y=df_cell['E_f_<_400'].values/1000, 
                            marker = 'o', s = 15, alpha = 0.9)
            ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 1.5,
                    label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                            f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
            k_list.append(k)
        
        ax.legend(fontsize = 9, loc = 'lower left').set_visible(False)
        
        ax.set_ylabel('$E_{400}$ (kPa)')
        ax.set_xlabel('')
        
        
        ax.grid(visible=True, which='major', axis='both')
        ax.set_xlim([50, 1100])

mean_k = np.mean(k_list)
ax.set_title(f'DMSO - Mean k = {mean_k:.2f}')
        
# Show
plt.tight_layout()
plt.show()

# %%% Tests on 24-12-11 - E(h) long series

figDir = os.path.join(cp.DirDataFig, 'Paper')


# %%% Functions to analyze and plot

dict_code = {'bestH0': 'H0',
             'surroundingThickness': 'H5mT',
             'E_f_<_400': 'E400',
             'E_f_<_500': 'E500',
             'E_f_<_600': 'E600',
             'E_eff': 'Eeff',
             }

dict_axisLabels = {'bestH0': '$H_{0}$ (nm)',
                    'surroundingThickness': '$H_{5mT}$ (nm)',
                    'E_f_<_400': '$E_{400}$ (kPa)',
                    'E_f_<_500': '$E_{500}$ (kPa)',
                    'E_f_<_600': '$E_{600}$ (kPa)',
                    'E_eff': '$E_{eff}$ (kPa)',
                    }

def compute_Eh_Exponent(df, XCol = 'bestH0', YCol = 'E_f_<_400',
                        crit_NcompsMin = 10,
                        crit_pvalFit = 0.1,
                        crit_thickCV = 0.5,
                        activeCrits = ['NcompsMin', 'pvalFit', 'thickCV'],
                        modeFit = 'OLS'):

    df, condCol = apm.makeCompositeCol(df, cols=['date'])
    CountByCond, CountByCell = apm.makeCountDf(df, condCol)
    df_f = df
    df_f = apm.computeNLMetrics_V2(df_f, th_NLI = np.log10(2), ref_strain = 0.2)
    
    CID_longSeries = CountByCell[CountByCell['compCount'] >= 3].reset_index()['cellID'].values
    Ncells = len(CID_longSeries)
    global_crit = ''
    
    for s in activeCrits:
        global_crit += s
        global_crit += '__'
    global_crit = global_crit[:-2]
    
    codeX, codeY = dict_code[XCol], dict_code[YCol]
    codeXY = '_' + codeX + '_' + codeY
    
    # Group By
    df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
    df_fg = df_fg[[XCol]]
    df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                          valCol = YCol, weightCol = 'ciw'+YCol, weight_method = 'ciw^2')
    df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

    dictFit = {'cellID':[], 
               'A'+codeXY:[], 'alpha'+codeXY:[], 'alphaCiw'+codeXY:[], 'pval'+codeXY:[], 'R2'+codeXY:[], 
               codeX+'_logmean':[], codeY+'_logmean':[], 'NLR_mean':[],
               'crit_NcompsMin':[crit_NcompsMin]*Ncells, 'valid_NcompsMin'+codeXY:[], 
               'crit_pvalFit':[crit_pvalFit]*Ncells, 'valid_pvalFit'+codeXY:[], 
               'crit_thickCV':[crit_thickCV]*Ncells, 'valid_thickCV'+codeXY:[],
               'activeCrits':[global_crit]*Ncells, 'valid_global'+codeXY:[],}
    k_list = []

    # Plot
    for i in range(Ncells):
        cid = CID_longSeries[i]
        df_cell = df_f[df_f['cellID'] == cid]
        Ncomps = len(df_cell)
        
        Xfit, Yfit = np.log(df_cell[XCol].values), np.log(df_cell[YCol].values/1000)
        # [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
        # A, k = np.exp(b), a
        # R2 = w_results.rsquared
        # Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
        # Yplot = A * Xplot**k
        
        if modeFit == 'OLS':
            perc, dof, = 0.975, len(Yfit)-2
            q = st.t.ppf(perc, dof)
            
            [b, a], results = ufun.fitLine(Xfit, Yfit)
            A, alpha = np.exp(b), a
            alphaCiw = results.HC3_se[1] * q
            R2 = results.rsquared
            pval = results.pvalues[1]
            # Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
            # Yplot = A * Xplot**k
        
        elif modeFit == 'ODR':
            
            def funFit(B, X):
                return(B[0]*X + B[1])
            
            linear = odr.Model(funFit)
            mydata = odr.Data(Xfit, Yfit, wd=1, we=1)
            myodr = odr.ODR(mydata, linear, beta0=[-1.5, 2.])
            myoutput = myodr.run()
            a, b = myoutput.beta
            A, alpha = np.exp(b), a
            perc, dof, = 0.975, len(Yfit)-2
            q = st.t.ppf(perc, dof)
            alphaCiw = myoutput.sd_beta[0] * q
            R2 = 1
            pval = 0
        
        H_logmean = np.mean(Xfit)
        E_logmean = np.mean(Yfit)
        NLR_mean  = np.mean(df_cell['NLI_mod'])
        thickCV = np.std(Xfit)/H_logmean
        
        dictFit['cellID'].append(cid)
        dictFit['A'+codeXY].append(A)
        dictFit['alpha'+codeXY].append(alpha)
        dictFit['alphaCiw'+codeXY].append(alphaCiw)
        dictFit['pval'+codeXY].append(pval)
        dictFit['R2'+codeXY].append(R2)
        dictFit[codeX + '_logmean'].append(H_logmean)
        dictFit[codeY + '_logmean'].append(E_logmean)
        dictFit['NLR_mean'].append(NLR_mean)
        dictFit['valid_NcompsMin'+codeXY].append(Ncomps >= crit_NcompsMin)
        dictFit['valid_pvalFit'+codeXY].append(pval <= crit_pvalFit)
        dictFit['valid_thickCV'+codeXY].append(thickCV >= crit_thickCV)
        check_all_crit = np.all([dictFit['valid_'+s+codeXY][-1] for s in activeCrits])
        dictFit['valid_global'+codeXY].append(check_all_crit)
        
    res_df = pd.DataFrame(dictFit)
    return(res_df, df_plot)
        


def plotEh_perCell(df, XCol = 'bestH0', YCol = 'E_f_<_400',
                    crit_NcompsMin = 10,
                    crit_pvalFit = 0.4,
                    crit_thickCV = 0.025,
                    activeCrits = ['NcompsMin', 'pvalFit', 'thickCV'],
                    dstDir = '', figNameRoot = '', modeFit = 'OLS'):
    
    df, condCol = apm.makeCompositeCol(df, cols=['date'])
    CountByCond, CountByCell = apm.makeCountDf(df, condCol)
    df_f = df
    df_f = apm.computeNLMetrics_V2(df_f, th_NLI = np.log10(2), ref_strain = 0.2)
    
    CID_longSeries = CountByCell[CountByCell['compCount'] >= 3].reset_index()['cellID'].values
    Ncells = len(CID_longSeries)
    global_crit = ''
    
    for s in activeCrits:
        global_crit += s
        global_crit += '__'
    global_crit = global_crit[:-2]
    
    codeX, codeY = dict_code[XCol], dict_code[YCol]
    codeXY = '_' + codeX + '_' + codeY
    
    df_res, df_plot = compute_Eh_Exponent(df, XCol = XCol, YCol = YCol,
                            crit_NcompsMin = crit_NcompsMin,
                            crit_pvalFit = crit_pvalFit,
                            crit_thickCV = crit_thickCV,
                            activeCrits = activeCrits,
                            modeFit = modeFit)
    df_res = df_res.set_index(['cellID'])
    
    # dictFit = {'cellID':[], 'A'+codeXY:[], 'alpha'+codeXY:[], 'pval'+codeXY:[], 'R2'+codeXY:[], 
    #            codeX+'_logmean':[], codeY+'_logmean':[], 'NLR_mean':[],
    #            'crit_NcompsMin':[crit_NcompsMin]*Ncells, 'valid_NcompsMin'+codeXY:[], 
    #            'crit_pvalFit':[crit_pvalFit]*Ncells, 'valid_pvalFit'+codeXY:[], 
    #            'crit_thickCV':[crit_thickCV]*Ncells, 'valid_thickCV'+codeXY:[],
    #            'activeCrits':[global_crit]*Ncells, 'valid_global'+codeXY:[],}

    #### Plot 1
    ## Initialize
    ncols = 5
    nrows = 1 + (Ncells-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(35/apm.cm_in, nrows*6/apm.cm_in), sharex=True, sharey=True)
    axes_f = axes.flatten()
    
    ## Make the plot
    for i in range(Ncells):
        ### Data
        cid = CID_longSeries[i]        
        df_cell = df_f[df_f['cellID'] == cid]
        Xfit, Yfit = np.log(df_cell[XCol].values), np.log(df_cell[YCol].values/1000)
        A, alpha, alphaCiw = df_res.loc[cid, 'A'+codeXY], df_res.loc[cid, 'alpha'+codeXY], df_res.loc[cid, 'alphaCiw'+codeXY] #np.exp(b), a
        R2, pval = df_res.loc[cid, 'R2'+codeXY], df_res.loc[cid, 'pval'+codeXY] #results.rsquared, results.pvalues[1]
        valid = df_res.loc[cid, 'valid_global'+codeXY]
        Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
        Yplot = A * Xplot**alpha
        
        ### Plot
        ax = axes_f[i]
        ax.set_xscale('log')
        ax.set_yscale('log')
        if valid:
            color = apm.cL_Set2[0]
        else:
            color = apm.cL_Set2[1]
            
        sns.scatterplot(ax = ax, x=df_cell[XCol].values, y=df_cell[YCol].values/1000, 
                        marker = 'o', s = 40, color = color, alpha = 0.9, zorder=6)
        ax.plot(Xplot, Yplot, ls = '-', c = 'dimgray', lw = 2.5, zorder=7,
                label = \
                        # r'$\bf{Fit\ y\ =\ A.x^k}$' + \
                        # f'\nA = {A:.1e}' + \
                        f'$\\alpha$  = {alpha:.2f} $\\pm $ {alphaCiw:.2f}' # + \
                        )
            
        ### Format
        ax.legend(fontsize = 9, loc = 'lower left')#.set_visible(False)
        ax.set_ylabel(dict_axisLabels[YCol])
        ax.set_xlabel(dict_axisLabels[XCol])
        ax.grid(visible=True, which='major', axis='both')
        ax.set_xlim([50, 1100])
        ax.set_title(cid, fontsize = 10)


        
    #### Plot 2
    ## Initialize
    fig2 = plt.figure(figsize=(30/cm_in, 20/cm_in))
    spec = fig2.add_gridspec(2, 3)
    ax21 = fig2.add_subplot(spec[0:2, 0:2])
    ax22 = fig2.add_subplot(spec[0, 2])
    ax23 = fig2.add_subplot(spec[1, 2])
    
    ax21.set_xscale('log')
    ax21.set_yscale('log')
    ax23.set_xscale('log')
    ax23.set_yscale('log')
    
    ## Consider only validated cells
    CID_validCells = df_res[df_res['valid_global'+codeXY] == True].index.values
    df_f2 = df_f[df_f['cellID'].apply(lambda x : x in CID_validCells)]
    df_res2 = df_res[df_res['valid_global'+codeXY] == True]
    Ncells_valid = len(CID_validCells)
    
    ## Color palette
    ColPal = sns.color_palette("husl", Ncells_valid)
    
    ## Subplot 2.1
    ax = ax21
    
    ### For each cell, plot all compressions in that cell color
    for i in range(Ncells_valid):
        # Data
        cid = CID_validCells[i]     
        c = ColPal[i]
        df_cell = df_f[df_f['cellID'] == cid]
        Xfit, Yfit = np.log(df_cell[XCol].values), np.log(df_cell[YCol].values/1000)
        A, alpha, alphaCiw = df_res.loc[cid, 'A'+codeXY], df_res.loc[cid, 'alpha'+codeXY], df_res.loc[cid, 'alphaCiw'+codeXY] #np.exp(b), a
        R2, pval = df_res.loc[cid, 'R2'+codeXY], df_res.loc[cid, 'pval'+codeXY] #results.rsquared, results.pvalues[1]
        valid = df_res.loc[cid, 'valid_global'+codeXY]
        Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
        Yplot = A * Xplot**alpha

        sns.scatterplot(ax = ax, x=df_cell[XCol].values, y=df_cell[YCol].values/1000, 
                        marker = 'o', s = 20, alpha = 0.9, color = c)
        ax.plot(Xplot, Yplot, ls = '--', c = c, lw = 1.5, alpha = 0.8,)
                # label =  f'$\\alpha$  = {alpha:.2f} $\\pm $ {alphaCiw:.2f}')
    
    ### Plot the fit on all compressions
    Xfit, Yfit = np.log(df_f2[XCol].values), np.log(df_f2[YCol].values/1000)
    [b, a], results = ufun.fitLine(Xfit, Yfit)
    A, alpha = np.exp(b), a
    perc, dof, = 0.975, len(Yfit)-2
    q = st.t.ppf(perc, dof)
    alphaCiw = results.HC3_se[1] * q
    R2 = results.rsquared
    pval = results.pvalues[1]
    
    Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = A * Xplot**alpha
    ax.plot(Xplot, Yplot, ls = '-', c = 'dimgray', lw = 2.0,
             label =  f'$\\alpha$  = {alpha:.2f} $\\pm $ {alphaCiw:.2f}')
            # label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\n$\\alpha$  = {alpha:.2f}' + \
            #         f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
            
    ### Format
    ax.legend(fontsize = 9, loc = 'lower left')#.set_visible(False)
    ax.set_ylabel(dict_axisLabels[YCol])
    ax.set_xlabel(dict_axisLabels[XCol])
    ax.grid(visible=True, which='major', axis='both')
    ax.set_xlim([50, 1100])
    
    
    ## Subplot 2.2
    ### Plot the exponents distribution (power law slopes)
    ax = ax22
    sns.boxplot(data = df_res2, ax = ax, y='alpha'+codeXY, 
                width=0.4, color='.9', showfliers = False,
                boxprops={"facecolor": (.7, .7, .7, .9), "edgecolor": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
                whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                )
    sns.swarmplot(data = df_res2, ax = ax, y='alpha'+codeXY,
                  size = 10, hue = 'cellID', palette = ColPal, legend=False)
    
    ### Format
    ax.grid(axis='y')
    ax.set_ylabel('Exponent $\\alpha $')
    
    
    ## Subplot 2.3
    ax = ax23
    
    ### Plot the cell averages
    Xfit, Yfit = (df_res2[codeX+'_logmean'].values), (df_res2[codeY+'_logmean'].values)
    
    sns.scatterplot(ax = ax, x = np.exp(Xfit), y = np.exp(Yfit), 
                    marker = 'o', s = 100, hue = df_res2.index, palette = ColPal, legend=False)
    
    ### Plot the fit on all cell averages
    [b, a], results = ufun.fitLine(Xfit, Yfit)
    A, alpha = np.exp(b), a
    perc, dof, = 0.975, len(Yfit)-2
    q = st.t.ppf(perc, dof)
    alphaCiw = results.HC3_se[1] * q
    R2 = results.rsquared
    pval = results.pvalues[1]
    
    Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = A * Xplot**alpha
    ax.plot(Xplot, Yplot, ls = '-', c = 'dimgray', lw = 2.0,
             label =  f'$\\alpha$  = {alpha:.2f} $\\pm $ {alphaCiw:.2f}')
    
    ### Format
    ax.legend(fontsize = 9, loc = 'lower left')#.set_visible(False)
    ax.set_ylabel(dict_axisLabels[YCol])
    ax.set_xlabel(dict_axisLabels[XCol])
    ax.grid(visible=True, which='major', axis='both')
    ax.set_xlim([50, 1100])
        
    plt.show()
    
    if dstDir != '':
        figName01 = figNameRoot + '_ExploreCells'
        figName02 = figNameRoot + '_Summary'
        ufun.archiveFig(fig, name = figName01, ext = '.png', dpi = 100,
                        figDir = os.path.join(cp.DirDataFig, 'Paper'), figSubDir = dstDir, cloudSave = 'flexible')
        ufun.archiveFig(fig2, name = figName02, ext = '.png', dpi = 100,
                        figDir = os.path.join(cp.DirDataFig, 'Paper'), figSubDir = dstDir, cloudSave = 'flexible')
        
def plotEh_compareDates(df, XCol = 'bestH0', YCol = 'E_f_<_400',
                    crit_NcompsMin = 10,
                    crit_pvalFit = 0.4,
                    crit_thickCV = 0.025,
                    activeCrits = ['NcompsMin', 'pvalFit', 'thickCV'],
                    PLOT = True, dstDir = '', figNameRoot = ''):
    
    df, condCol = apm.makeCompositeCol(df, cols=['date'])
    CountByCond, CountByCell = apm.makeCountDf(df, condCol)
    df_f = df
    df_f = apm.computeNLMetrics_V2(df_f, th_NLI = np.log10(2), ref_strain = 0.2)
    
    CID_longSeries = CountByCell[CountByCell['compCount'] >= 3].reset_index()['cellID'].values
    Ncells = len(CID_longSeries)
    global_crit = ''
    
    for s in activeCrits:
        global_crit += s
        global_crit += '__'
    global_crit = global_crit[:-2]
    
    codeX, codeY = dict_code[XCol], dict_code[YCol]
    codeXY = '_' + codeX + '_' + codeY
    
    dates = df_f['date'].unique()
    list_res = []
    
    #### Compute results for all cells
    for date in dates:
        df_date = df_f[df_f['date']==date]
        df_res, df_plot = compute_Eh_Exponent(df_date, XCol = XCol, YCol = YCol,
                                crit_NcompsMin = crit_NcompsMin,
                                crit_pvalFit = crit_pvalFit,
                                crit_thickCV = crit_thickCV,
                                activeCrits = activeCrits)
        df_res = df_res.set_index(['cellID'])
        df_res['date'] = [date]*len(df_res)
        list_res.append(df_res)
        
    concat_res = pd.concat(list_res)
    
    #### Consider only validated cells
    CID_validCells = concat_res[concat_res['valid_global'+codeXY] == True].index.values
    df_f2 = df_f[df_f['cellID'].apply(lambda x : x in CID_validCells)]
    concat_res2 = concat_res[concat_res['valid_global'+codeXY] == True]
    Ncells_valid = len(CID_validCells)
        
    #### Compute results for each date
    dict_byDate = {'date':dates, 
                   'alpha_allComps':[], 'alpha_allComps_ciw':[],
                   'alpha_allCells':[], 'alpha_allCells_ciw':[]}
    for date in dates:
        df_date = df_f2[df_f2['date']==date]
        res_date = concat_res2[concat_res2['date']==date]
        
        ### Do the fit on all compressions
        Xfit, Yfit = np.log(df_date[XCol].values), np.log(df_date[YCol].values/1000)
        [b, a], results = ufun.fitLine(Xfit, Yfit)
        A, alpha = np.exp(b), a
        perc, dof = 0.975, len(Yfit)-2
        q = st.t.ppf(perc, dof)
        alphaCiw = results.HC3_se[1] * q
        # R2 = results.rsquared
        # pval = results.pvalues[1]
        dict_byDate['alpha_allComps'].append(alpha)
        dict_byDate['alpha_allComps_ciw'].append(alphaCiw)
        
        ### Do the fit on all cell average
        Xfit, Yfit = (res_date[codeX+'_logmean'].values), (res_date[codeY+'_logmean'].values)
        [b, a], results = ufun.fitLine(Xfit, Yfit)
        A, alpha = np.exp(b), a
        perc, dof = 0.975, len(Yfit)-2
        q = st.t.ppf(perc, dof)
        alphaCiw = results.HC3_se[1] * q
        # R2 = results.rsquared
        # pval = results.pvalues[1]
        dict_byDate['alpha_allCells'].append(alpha)
        dict_byDate['alpha_allCells_ciw'].append(alphaCiw)
        
    df_byDate = pd.DataFrame(dict_byDate)
        
    if PLOT:
        fig, ax = plt.subplots(1,1, figsize=(20/cm_in, 14/cm_in))
        sns.boxplot(data = concat_res2, ax = ax, x='date', y='alpha'+codeXY, 
                    width=0.4, color='.9', showfliers = False,
                    boxprops={"facecolor": (.7, .7, .7, .9), "edgecolor": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                    medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
                    whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                    capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                    )
        sns.swarmplot(data = concat_res2, ax = ax, x='date', y='alpha'+codeXY,
                      size = 10, hue = 'date', legend=False)
        sns.pointplot(
            data=df_byDate, ax = ax, x="date", y="alpha_allComps", errorbar=None,
            linestyle="none", marker="_", markersize=20, markeredgewidth=4,
            color='green', zorder = 5, label = 'Exponent fitted on all compressions'
        )
        sns.pointplot(
            data=df_byDate, ax = ax, x="date", y="alpha_allCells", errorbar=None,
            linestyle="none", marker="_", markersize=20, markeredgewidth=4,
            color='blue', zorder = 5, label = 'Exponent fitted on cells log-mean'
        )

        ### Format
        ax.grid(axis='y')
        ax.set_ylabel('Exponent $\\alpha $')
        ax.legend()
        ax.set_title(f'Exponents fitted with {codeX} vs. {codeY}')
        
        plt.show()
        
        
        return(concat_res)
        
    


def plotEh_fitVals(df, XCol = 'bestH0', YCol = 'E_f_<_400',
                    crit_NcompsMin = 10,
                    crit_pvalFit = 0.4,
                    crit_thickCV = 0.025,
                    activeCrits = ['NcompsMin', 'pvalFit', 'thickCV']):
    
    df, condCol = apm.makeCompositeCol(df, cols=['date'])
    CountByCond, CountByCell = apm.makeCountDf(df, condCol)
    df_f = df
    df_f = apm.computeNLMetrics_V2(df_f, th_NLI = np.log10(2), ref_strain = 0.2)
    
    CID_longSeries = CountByCell[CountByCell['compCount'] >= 3].reset_index()['cellID'].values
    Ncells = len(CID_longSeries)
    global_crit = ''
    
    for s in activeCrits:
        global_crit += s
        global_crit += '__'
    global_crit = global_crit[:-2]
    
    codeX, codeY = dict_code[XCol], dict_code[YCol]
    codeXY = '_' + codeX + '_' + codeY
    
    df_res, df_plot = compute_Eh_Exponent(df, XCol = XCol, YCol = YCol,
                            crit_NcompsMin = crit_NcompsMin,
                            crit_pvalFit = crit_pvalFit,
                            crit_thickCV = crit_thickCV,
                            activeCrits = activeCrits)
    df_res = df_res.set_index(['cellID'])
    
    # Initialize the plot
    fig, axes = plt.subplots(2, 2, figsize=(35/apm.cm_in, 25/apm.cm_in))
    
    # 1st plot
    ax = axes[0, 0]
    sns.boxplot(data = df_res, ax = ax, y='alpha'+codeXY, 
                width=0.5, color='.9')
    sns.swarmplot(data = df_res, ax = ax, y='alpha'+codeXY,
                  size = 10)
    ax.grid(axis='y')
    
    # 2nd plot
    ax = axes[0, 1]
    sns.scatterplot(data = df_res, ax=ax, x=codeX+'_logmean', y='alpha'+codeXY)
    ax.grid(axis='both')
    
    # 2nd plot
    ax = axes[1, 0]
    sns.scatterplot(data = df_res, ax=ax, x=codeY+'_logmean', y='alpha'+codeXY)
    ax.grid(axis='both')
    
    # 2nd plot
    ax = axes[1, 1]
    sns.scatterplot(data = df_res, ax=ax, x='NLR_mean', y='alpha'+codeXY)
    ax.grid(axis='both')
    
    plt.show()

# %%% Use the functions for 24-12-11 - E(h) long series

figSubDir = '24-12-11_longSeries'

df = MecaData_Phy
dates = ['24-12-11']
XCol = 'bestH0'
YCol = 'E_f_<_500'

cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x in dates)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_500'] <= 2e4),
           (df['valid_f_<_500'] == True), 
           ]

df = apm.filterDf(df, Filters)

res_df, df_plot = compute_Eh_Exponent(df, XCol = 'bestH0', YCol = 'E_f_<_500',
                        crit_NcompsMin = 10,
                        crit_pvalFit = 0.4,
                        crit_thickCV = 0.025,
                        activeCrits = ['NcompsMin', 'thickCV'], modeFit = 'ODR')

plotEh_perCell(df, XCol = 'bestH0', YCol = 'E_f_<_500',
                    crit_NcompsMin = 10,
                    crit_pvalFit = 0.4,
                    crit_thickCV = 0.02,
                    activeCrits = ['NcompsMin', 'thickCV'],
                    dstDir = 'E-h_perDate', figNameRoot = '24-12-11_E500vH0', modeFit = 'ODR')

# plotEh_fitVals(df, XCol = 'bestH0', YCol = 'E_f_<_400',
#                     crit_NcompsMin = 10,
#                     crit_pvalFit = 0.4,
#                     crit_thickCV = 0.025,
#                     activeCrits = ['NcompsMin', 'thickCV'])






# %%%  Use the functions for 23-02-16 - E(h)

figSubDir = '23-02-16'

df = MecaData_Phy
dates = ['23-02-16']
XCol = 'bestH0'
YCol = 'E_f_<_500'

cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x in dates)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_500'] <= 2e4),
           (df['valid_f_<_500'] == True), 
           ]

df = apm.filterDf(df, Filters)

res_df, df_plot = compute_Eh_Exponent(df, XCol = 'bestH0', YCol = 'E_f_<_500',
                        crit_NcompsMin = 7,
                        crit_pvalFit = 0.4,
                        crit_thickCV = 0.025,
                        activeCrits = ['NcompsMin', 'pvalFit', 'thickCV'])

plotEh_perCell(df, XCol = 'bestH0', YCol = 'E_f_<_500',
                    crit_NcompsMin = 7,
                    crit_pvalFit = 0.4,
                    crit_thickCV = 0.02,
                    activeCrits = ['NcompsMin', 'thickCV'],
                    dstDir = 'E-h_perDate', figNameRoot = '23-02-16_E500vH0')


# %%%  Use the functions for 23-03-16 - E(h)

figSubDir = '23-03-16'

df = MecaData_Phy
dates = ['23-03-16']
XCol = 'bestH0'
YCol = 'E_f_<_500'

cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x in dates)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_500'] <= 2e4),
           (df['valid_f_<_500'] == True), 
           ]

df = apm.filterDf(df, Filters)

res_df, df_plot = compute_Eh_Exponent(df, XCol = 'bestH0', YCol = 'E_f_<_500',
                        crit_NcompsMin = 7,
                        crit_pvalFit = 0.4,
                        crit_thickCV = 0.025,
                        activeCrits = ['NcompsMin', 'pvalFit', 'thickCV'])

plotEh_perCell(df, XCol = 'bestH0', YCol = 'E_f_<_500',
                    crit_NcompsMin = 7,
                    crit_pvalFit = 0.4,
                    crit_thickCV = 0.02,
                    activeCrits = ['NcompsMin', 'thickCV'],
                    dstDir = 'E-h_perDate', figNameRoot = '23-03-16_E500vH0')


# %%%  Use the functions for 23-04-26 - E(h)

figSubDir = '23-04-26'

df = MecaData_Phy
dates = ['23-04-26']
XCol = 'bestH0'
YCol = 'E_f_<_500'

cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x in dates)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_500'] <= 2e4),
           (df['valid_f_<_500'] == True), 
           ]

df = apm.filterDf(df, Filters)

res_df, df_plot = compute_Eh_Exponent(df, XCol = 'bestH0', YCol = 'E_f_<_500',
                        crit_NcompsMin = 6,
                        crit_pvalFit = 0.4,
                        crit_thickCV = 0.025,
                        activeCrits = ['NcompsMin', 'pvalFit', 'thickCV'])

plotEh_perCell(df, XCol = 'bestH0', YCol = 'E_f_<_500',
                    crit_NcompsMin = 6,
                    crit_pvalFit = 0.4,
                    crit_thickCV = 0.02,
                    activeCrits = ['NcompsMin', 'thickCV'],
                    dstDir = 'E-h_perDate', figNameRoot = '23-04-26_E500vH0')


# %%% Work on a common data frame

df = MecaData_Phy
dates = ['23-02-16', '23-03-16', '23-04-26', '24-12-11']

XCol = 'bestH0'
YCol = 'E_f_<_500'

cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x in dates)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_500'] <= 2e4),
           (df['valid_f_<_500'] == True), 
           ]

df = apm.filterDf(df, Filters)

concat_res = plotEh_compareDates(df, XCol = 'bestH0', YCol = 'E_f_<_500',
                    crit_NcompsMin = 6,
                    crit_pvalFit = 0.4,
                    crit_thickCV = 0.025,
                    activeCrits = ['NcompsMin', 'pvalFit', 'thickCV'],
                    PLOT = True, dstDir = '', figNameRoot = '')

# %% ------

# %% Implement a different type of fit (Total Least Squares)

# %%% Dataset -> 24-12-11

figSubDir = '24-12-11_longSeries'

df = MecaData_Phy
dates = ['24-12-11']
XCol = 'bestH0'
YCol = 'E_f_<_500'

cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x in dates)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_500'] <= 2e4),
           (df['valid_f_<_500'] == True), 
           ]

df = apm.filterDf(df, Filters)

# %%% Ordinary Least Square

dict_code = {'bestH0': 'H0',
             'surroundingThickness': 'H5mT',
             'E_f_<_400': 'E400',
             'E_f_<_500': 'E500',
             'E_f_<_600': 'E600',
             'E_eff': 'Eeff',
             }

dict_axisLabels = {'bestH0': '$H_{0}$ (nm)',
                    'surroundingThickness': '$H_{5mT}$ (nm)',
                    'E_f_<_400': '$E_{400}$ (kPa)',
                    'E_f_<_500': '$E_{500}$ (kPa)',
                    'E_f_<_600': '$E_{600}$ (kPa)',
                    'E_eff': '$E_{eff}$ (kPa)',
                    }

XCol = 'bestH0'
YCol = 'E_f_<_500'
crit_NcompsMin = 10
crit_pvalFit = 0.4
crit_thickCV = 0.02
activeCrits = ['NcompsMin', 'thickCV']
dstDir = 'E-h_perDate'
figNameRoot = '24-12-11_E500vH0'

df, condCol = apm.makeCompositeCol(df, cols=['date'])
CountByCond, CountByCell = apm.makeCountDf(df, condCol)
df_f = df
df_f = apm.computeNLMetrics_V2(df_f, th_NLI = np.log10(2), ref_strain = 0.2)

CID_longSeries = CountByCell[CountByCell['compCount'] >= 3].reset_index()['cellID'].values
Ncells = len(CID_longSeries)
global_crit = ''

for s in activeCrits:
    global_crit += s
    global_crit += '__'
global_crit = global_crit[:-2]

codeX, codeY = dict_code[XCol], dict_code[YCol]
codeXY = '_' + codeX + '_' + codeY

df_res, df_plot = compute_Eh_Exponent(df, XCol = XCol, YCol = YCol,
                        crit_NcompsMin = crit_NcompsMin,
                        crit_pvalFit = crit_pvalFit,
                        crit_thickCV = crit_thickCV,
                        activeCrits = activeCrits)
df_res = df_res.set_index(['cellID'])

#### Plot 1
## Initialize
ncols = 5
nrows = 1 + (Ncells-1)//ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(35/apm.cm_in, nrows*6/apm.cm_in), sharex=True, sharey=True)
axes_f = axes.flatten()

## Make the plot
for i in range(Ncells):
    ### Data
    cid = CID_longSeries[i]        
    df_cell = df_f[df_f['cellID'] == cid]
    Xfit, Yfit = np.log(df_cell[XCol].values), np.log(df_cell[YCol].values/1000)
    A, alpha, alphaCiw = df_res.loc[cid, 'A'+codeXY], df_res.loc[cid, 'alpha'+codeXY], df_res.loc[cid, 'alphaCiw'+codeXY] #np.exp(b), a
    R2, pval = df_res.loc[cid, 'R2'+codeXY], df_res.loc[cid, 'pval'+codeXY] #results.rsquared, results.pvalues[1]
    valid = df_res.loc[cid, 'valid_global'+codeXY]
    Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = A * Xplot**alpha
    
    ### Plot
    ax = axes_f[i]
    ax.set_xscale('log')
    ax.set_yscale('log')
    if valid:
        color = apm.cL_Set2[0]
    else:
        color = apm.cL_Set2[1]
        
    sns.scatterplot(ax = ax, x=df_cell[XCol].values, y=df_cell[YCol].values/1000, 
                    marker = 'o', s = 40, color = color, alpha = 0.9, zorder=6)
    ax.plot(Xplot, Yplot, ls = '-', c = 'dimgray', lw = 2.5, zorder=7,
            label = \
                    # r'$\bf{Fit\ y\ =\ A.x^k}$' + \
                    # f'\nA = {A:.1e}' + \
                    f'$\\alpha$  = {alpha:.2f} $\\pm $ {alphaCiw:.2f}' # + \
                    )
        
    ### Format
    ax.legend(fontsize = 9, loc = 'lower left')#.set_visible(False)
    ax.set_ylabel(dict_axisLabels[YCol])
    ax.set_xlabel(dict_axisLabels[XCol])
    ax.grid(visible=True, which='major', axis='both')
    ax.set_xlim([50, 1100])
    ax.set_title(cid, fontsize = 10)


#### Plot
## Initialize
fig2 = plt.figure(figsize=(30/cm_in, 20/cm_in))
spec = fig2.add_gridspec(2, 3)
ax21 = fig2.add_subplot(spec[0:2, 0:2])
ax22 = fig2.add_subplot(spec[0, 2])
ax23 = fig2.add_subplot(spec[1, 2])

ax21.set_xscale('log')
ax21.set_yscale('log')
ax23.set_xscale('log')
ax23.set_yscale('log')

## Consider only validated cells
CID_validCells = df_res[df_res['valid_global'+codeXY] == True].index.values
df_f2 = df_f[df_f['cellID'].apply(lambda x : x in CID_validCells)]
df_res2 = df_res[df_res['valid_global'+codeXY] == True]
Ncells_valid = len(CID_validCells)

## Color palette
ColPal = sns.color_palette("husl", Ncells_valid)

## Subplot 2.1
ax = ax21

### For each cell, plot all compressions in that cell color
for i in range(Ncells_valid):
    # Data
    cid = CID_validCells[i]     
    c = ColPal[i]
    df_cell = df_f[df_f['cellID'] == cid]
    Xfit, Yfit = np.log(df_cell[XCol].values), np.log(df_cell[YCol].values/1000)
    A, alpha, alphaCiw = df_res.loc[cid, 'A'+codeXY], df_res.loc[cid, 'alpha'+codeXY], df_res.loc[cid, 'alphaCiw'+codeXY] #np.exp(b), a
    R2, pval = df_res.loc[cid, 'R2'+codeXY], df_res.loc[cid, 'pval'+codeXY] #results.rsquared, results.pvalues[1]
    valid = df_res.loc[cid, 'valid_global'+codeXY]
    Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = A * Xplot**alpha

    sns.scatterplot(ax = ax, x=df_cell[XCol].values, y=df_cell[YCol].values/1000, 
                    marker = 'o', s = 20, alpha = 0.9, color = c)
    ax.plot(Xplot, Yplot, ls = '--', c = c, lw = 1.5, alpha = 0.8,)
            # label =  f'$\\alpha$  = {alpha:.2f} $\\pm $ {alphaCiw:.2f}')

### Plot the fit on all compressions
Xfit, Yfit = np.log(df_f2[XCol].values), np.log(df_f2[YCol].values/1000)
[b, a], results = ufun.fitLine(Xfit, Yfit)
A, alpha = np.exp(b), a
perc, dof, = 0.975, len(Yfit)-2
q = st.t.ppf(perc, dof)
alphaCiw = results.HC3_se[1] * q
R2 = results.rsquared
pval = results.pvalues[1]

Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**alpha
ax.plot(Xplot, Yplot, ls = '-', c = 'dimgray', lw = 2.0,
         label =  f'$\\alpha$  = {alpha:.2f} $\\pm $ {alphaCiw:.2f}')
        # label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\n$\\alpha$  = {alpha:.2f}' + \
        #         f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
        
### Format
ax.legend(fontsize = 9, loc = 'lower left')#.set_visible(False)
ax.set_ylabel(dict_axisLabels[YCol])
ax.set_xlabel(dict_axisLabels[XCol])
ax.grid(visible=True, which='major', axis='both')
ax.set_xlim([50, 1100])


## Subplot 2.2
### Plot the exponents distribution (power law slopes)
ax = ax22
sns.boxplot(data = df_res2, ax = ax, y='alpha'+codeXY, 
            width=0.4, color='.9', showfliers = False,
            boxprops={"facecolor": (.7, .7, .7, .9), "edgecolor": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
            medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
            whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
            capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
            )
sns.swarmplot(data = df_res2, ax = ax, y='alpha'+codeXY,
              size = 10, hue = 'cellID', palette = ColPal, legend=False)

### Format
ax.grid(axis='y')
ax.set_ylabel('Exponent $\\alpha $')


## Subplot 2.3
ax = ax23

### Plot the cell averages
Xfit, Yfit = (df_res2[codeX+'_logmean'].values), (df_res2[codeY+'_logmean'].values)

sns.scatterplot(ax = ax, x = np.exp(Xfit), y = np.exp(Yfit), 
                marker = 'o', s = 100, hue = df_res2.index, palette = ColPal, legend=False)

### Plot the fit on all cell averages
[b, a], results = ufun.fitLine(Xfit, Yfit)
A, alpha = np.exp(b), a
perc, dof, = 0.975, len(Yfit)-2
q = st.t.ppf(perc, dof)
alphaCiw = results.HC3_se[1] * q
R2 = results.rsquared
pval = results.pvalues[1]

Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**alpha
ax.plot(Xplot, Yplot, ls = '-', c = 'dimgray', lw = 2.0,
         label =  f'$\\alpha$  = {alpha:.2f} $\\pm $ {alphaCiw:.2f}')

### Format
ax.legend(fontsize = 9, loc = 'lower left')#.set_visible(False)
ax.set_ylabel(dict_axisLabels[YCol])
ax.set_xlabel(dict_axisLabels[XCol])
ax.grid(visible=True, which='major', axis='both')
ax.set_xlim([50, 1100])
    
plt.show()



# %%% Orthogonal Distance Regression



dict_code = {'bestH0': 'H0',
             'surroundingThickness': 'H5mT',
             'E_f_<_400': 'E400',
             'E_f_<_500': 'E500',
             'E_f_<_600': 'E600',
             'E_eff': 'Eeff',
             }

dict_axisLabels = {'bestH0': '$H_{0}$ (nm)',
                    'surroundingThickness': '$H_{5mT}$ (nm)',
                    'E_f_<_400': '$E_{400}$ (kPa)',
                    'E_f_<_500': '$E_{500}$ (kPa)',
                    'E_f_<_600': '$E_{600}$ (kPa)',
                    'E_eff': '$E_{eff}$ (kPa)',
                    }

XCol = 'bestH0'
YCol = 'E_f_<_500'
crit_NcompsMin = 10
crit_pvalFit = 0.4
crit_thickCV = 0.02
activeCrits = ['NcompsMin', 'thickCV']
dstDir = 'E-h_perDate'
figNameRoot = '24-12-11_E500vH0'

df, condCol = apm.makeCompositeCol(df, cols=['date'])
CountByCond, CountByCell = apm.makeCountDf(df, condCol)
df_f = df
df_f = apm.computeNLMetrics_V2(df_f, th_NLI = np.log10(2), ref_strain = 0.2)

CID_longSeries = CountByCell[CountByCell['compCount'] >= 3].reset_index()['cellID'].values
Ncells = len(CID_longSeries)
global_crit = ''

for s in activeCrits:
    global_crit += s
    global_crit += '__'
global_crit = global_crit[:-2]

codeX, codeY = dict_code[XCol], dict_code[YCol]
codeXY = '_' + codeX + '_' + codeY

df_res, df_plot = compute_Eh_Exponent(df, XCol = XCol, YCol = YCol,
                        crit_NcompsMin = crit_NcompsMin,
                        crit_pvalFit = crit_pvalFit,
                        crit_thickCV = crit_thickCV,
                        activeCrits = activeCrits,
                        modeFit = 'ODR')
df_res = df_res.set_index(['cellID'])

#### Plot 1
## Initialize
ncols = 5
nrows = 1 + (Ncells-1)//ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(35/apm.cm_in, nrows*6/apm.cm_in), sharex=True, sharey=True)
axes_f = axes.flatten()

## Make the plot
for i in range(Ncells):
    ### Data
    cid = CID_longSeries[i]        
    df_cell = df_f[df_f['cellID'] == cid]
    Xfit, Yfit = np.log(df_cell[XCol].values), np.log(df_cell[YCol].values/1000)
    A, alpha, alphaCiw = df_res.loc[cid, 'A'+codeXY], df_res.loc[cid, 'alpha'+codeXY], df_res.loc[cid, 'alphaCiw'+codeXY] #np.exp(b), a
    R2, pval = df_res.loc[cid, 'R2'+codeXY], df_res.loc[cid, 'pval'+codeXY] #results.rsquared, results.pvalues[1]
    valid = df_res.loc[cid, 'valid_global'+codeXY]
    Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = A * Xplot**alpha
    
    ### Plot
    ax = axes_f[i]
    ax.set_xscale('log')
    ax.set_yscale('log')
    if valid:
        color = apm.cL_Set2[0]
    else:
        color = apm.cL_Set2[1]
        
    if alpha > -5:
        sns.scatterplot(ax = ax, x=df_cell[XCol].values, y=df_cell[YCol].values/1000, 
                        marker = 'o', s = 40, color = color, alpha = 0.9, zorder=6)
        ax.plot(Xplot, Yplot, ls = '-', c = 'dimgray', lw = 2.5, zorder=7,
                label = \
                        # r'$\bf{Fit\ y\ =\ A.x^k}$' + \
                        # f'\nA = {A:.1e}' + \
                        f'$\\alpha$  = {alpha:.2f} $\\pm $ {alphaCiw:.2f}' # + \
                        )
        
        ### Format
        ax.legend(fontsize = 9, loc = 'lower left')#.set_visible(False)
        ax.set_ylabel(dict_axisLabels[YCol])
        ax.set_xlabel(dict_axisLabels[XCol])
        ax.grid(visible=True, which='major', axis='both')
        ax.set_xlim([50, 1100])
        ax.set_title(cid, fontsize = 10)


#### Plot 2
## Initialize
fig2 = plt.figure(figsize=(30/cm_in, 20/cm_in))
spec = fig2.add_gridspec(2, 3)
ax21 = fig2.add_subplot(spec[0:2, 0:2])
ax22 = fig2.add_subplot(spec[0, 2])
ax23 = fig2.add_subplot(spec[1, 2])

ax21.set_xscale('log')
ax21.set_yscale('log')
ax23.set_xscale('log')
ax23.set_yscale('log')

## Consider only validated cells
CID_validCells = df_res[df_res['valid_global'+codeXY] == True].index.values
df_f2 = df_f[df_f['cellID'].apply(lambda x : x in CID_validCells)]
df_res2 = df_res[df_res['valid_global'+codeXY] == True]
Ncells_valid = len(CID_validCells)

## Color palette
ColPal = sns.color_palette("husl", Ncells_valid)

## Subplot 2.1
ax = ax21

### For each cell, plot all compressions in that cell color
for i in range(Ncells_valid):
    # Data
    cid = CID_validCells[i]     
    c = ColPal[i]
    df_cell = df_f[df_f['cellID'] == cid]
    Xfit, Yfit = np.log(df_cell[XCol].values), np.log(df_cell[YCol].values/1000)
    # A, alpha, alphaCiw = df_res.loc[cid, 'A'+codeXY], df_res.loc[cid, 'alpha'+codeXY], df_res.loc[cid, 'alphaCiw'+codeXY] #np.exp(b), a
    # R2, pval = df_res.loc[cid, 'R2'+codeXY], df_res.loc[cid, 'pval'+codeXY] #results.rsquared, results.pvalues[1]
    # valid = df_res.loc[cid, 'valid_global'+codeXY]
    # Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    # Yplot = A * Xplot**alpha
    
    def funFit(B, X):
        return(B[0]*X + B[1])
    
    linear = odr.Model(funFit)
    # mydata = odr.Data(Xfit, Yfit, wd=1, we=1)
    mydata = odr.RealData(Xfit, Yfit, sx=1, sy=1)
    # mydata = Data(x, y, wd=1./power(sx,2), we=1./power(sy,2))
    # mydata = RealData(x, y, sx=sx, sy=sy)
    myodr = odr.ODR(mydata, linear, beta0=[-1.5, 2.])
    myoutput = myodr.run()
    # myoutput.pprint()
    a, b = myoutput.beta
    A, alpha = np.exp(b), a
    Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = A * Xplot**alpha

    sns.scatterplot(ax = ax, x=df_cell[XCol].values, y=df_cell[YCol].values/1000, 
                    marker = 'o', s = 20, alpha = 0.9, color = c)
    ax.plot(Xplot, Yplot, ls = '--', c = c, lw = 1.5, alpha = 0.8,)
            # label =  f'$\\alpha$  = {alpha:.2f} $\\pm $ {alphaCiw:.2f}')

### Plot the fit on all compressions
Xfit, Yfit = np.log(df_f2[XCol].values), np.log(df_f2[YCol].values/1000)
# [b, a], results = ufun.fitLine(Xfit, Yfit)
# A, alpha = np.exp(b), a
# perc, dof, = 0.975, len(Yfit)-2
# q = st.t.ppf(perc, dof)
# alphaCiw = results.HC3_se[1] * q
# R2 = results.rsquared
# pval = results.pvalues[1]

linear = odr.Model(funFit)
mydata = odr.Data(Xfit, Yfit, wd=1, we=1)
myodr = odr.ODR(mydata, linear, beta0=[-1.5, 2.])
myoutput = myodr.run()
a, b = myoutput.beta
A, alpha = np.exp(b), a
perc, dof, = 0.975, len(Yfit)-2
q = st.t.ppf(perc, dof)
alphaCiw = myoutput.sd_beta[0] * q

Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**alpha
ax.plot(Xplot, Yplot, ls = '-', c = 'dimgray', lw = 2.0,
         label =  f'$\\alpha$  = {alpha:.2f} $\\pm $ {alphaCiw:.2f}')
        # label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\n$\\alpha$  = {alpha:.2f}' + \
        #         f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
        
### Format
ax.legend(fontsize = 9, loc = 'lower left')#.set_visible(False)
ax.set_ylabel(dict_axisLabels[YCol])
ax.set_xlabel(dict_axisLabels[XCol])
ax.grid(visible=True, which='major', axis='both')
ax.set_xlim([50, 1100])


## Subplot 2.2
### Plot the exponents distribution (power law slopes)
ax = ax22
sns.boxplot(data = df_res2, ax = ax, y='alpha'+codeXY, 
            width=0.4, color='.9', showfliers = False,
            boxprops={"facecolor": (.7, .7, .7, .9), "edgecolor": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
            medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
            whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
            capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
            )
sns.swarmplot(data = df_res2, ax = ax, y='alpha'+codeXY,
              size = 10, hue = 'cellID', palette = ColPal, legend=False)

### Format
ax.grid(axis='y')
ax.set_ylabel('Exponent $\\alpha $')


## Subplot 2.3
ax = ax23

### Plot the cell averages
Xfit, Yfit = (df_res2[codeX+'_logmean'].values), (df_res2[codeY+'_logmean'].values)

sns.scatterplot(ax = ax, x = np.exp(Xfit), y = np.exp(Yfit), 
                marker = 'o', s = 100, hue = df_res2.index, palette = ColPal, legend=False)

### Plot the fit on all cell averages
# [b, a], results = ufun.fitLine(Xfit, Yfit)
# A, alpha = np.exp(b), a
# perc, dof, = 0.975, len(Yfit)-2
# q = st.t.ppf(perc, dof)
# alphaCiw = results.HC3_se[1] * q
# R2 = results.rsquared
# pval = results.pvalues[1]

linear = odr.Model(funFit)
mydata = odr.Data(Xfit, Yfit, wd=1, we=1)
myodr = odr.ODR(mydata, linear, beta0=[-1.5, 2.])
myoutput = myodr.run()
a, b = myoutput.beta
A, alpha = np.exp(b), a
perc, dof, = 0.975, len(Yfit)-2
q = st.t.ppf(perc, dof)
alphaCiw = myoutput.sd_beta[0] * q

Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**alpha
ax.plot(Xplot, Yplot, ls = '-', c = 'dimgray', lw = 2.0,
         label =  f'$\\alpha$  = {alpha:.2f} $\\pm $ {alphaCiw:.2f}')

### Format
ax.legend(fontsize = 9, loc = 'lower left')#.set_visible(False)
ax.set_ylabel(dict_axisLabels[YCol])
ax.set_xlabel(dict_axisLabels[XCol])
ax.grid(visible=True, which='major', axis='both')
ax.set_xlim([50, 1100])
    
plt.show()

# %%% Apply it to total dataset

# %%%% Figure NC6 - Thickness & Stiffness 1. E500 v H0
figDir = os.path.join(cp.DirDataFig, 'Paper')
apm.set_defense_options_jv(palette = 'Set2')

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix
XCol = 'bestH0'
YCol = 'E_f_<_500'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x not in excluded_dates)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_500'] <= 10e4),
           (df['valid_f_<_500'] == True), 
           ]

df_f = apm.filterDf(df, Filters)
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)

# Order
co_order = []

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_500', weightCol = 'ciwE_f_<_500', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,1, figsize=(17/apm.cm_in,10/apm.cm_in), sharex=True)

# LinLog
ax = axes
ax.set_xscale('log')
ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 0.5)


sns.scatterplot(ax = ax, x=df_f['bestH0'].values, y=df_f['E_f_<_500'].values/1000, 
                marker = 'o', s = 15, color = 'gray', alpha = 0.5)
Xfit, Yfit = np.log(df_f['bestH0'].values), np.log(df_f['E_f_<_500'].values/1000)
# [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
# A, k = np.exp(b), a
# R2 = w_results.rsquared
# Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
# Yplot = A * Xplot**k
# ax.plot(Xplot, Yplot, ls = '--', c = 'k',
#         label = 'Fit $y = A.x^k$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + f'\n$R^2$  = {R2:.2f}')

def funFit(B, X):
    return(B[0]*X + B[1])

linear = odr.Model(funFit)
mydata = odr.Data(Xfit, Yfit, wd=1, we=1)
myodr = odr.ODR(mydata, linear, beta0=[-1.5, 2.])
myoutput = myodr.run()
a, b = myoutput.beta
A, k = np.exp(b), a
perc, dof, = 0.975, len(Yfit)-2
q = st.t.ppf(perc, dof)
alphaCiw = myoutput.sd_beta[0] * q

# [b, a], results = ufun.fitLine(Xfit, Yfit)
# A, k = np.exp(b), a
# R2 = results.rsquared
# pval = results.pvalues[1]
R2 = -1
pval = -1

Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k

ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 1.5,
        label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')

ax.legend(fontsize = 9, loc = 'lower left')
ax.set_title(f'All compressions - N = {len(Xfit)}')
ax.set_ylabel('$E_{500}$ (kPa)')
ax.set_xlabel('')

color = apm.cL_Set2[0]

ax = axes.inset_axes([0.6, 0.6, 0.35, 0.35])
ax.set_xscale('log')
ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 0.5)

sns.scatterplot(ax = ax, x=df_plot['bestH0'].values, y=df_plot['E_f_<_500_wAvg'].values/1000, 
                marker = 'o', s = 20, color = color, edgecolor = 'k', alpha = 0.5)

Xfit, Yfit = np.log(df_plot['bestH0'].values), np.log(df_plot['E_f_<_500_wAvg'].values/1000)
# [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
# A, k = np.exp(b), a
# Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
# Yplot = A * Xplot**k
# ax.plot(Xplot, Yplot, ls = '--', c = 'k',
#         label = 'Fit $y = A.x^k$' + '\n' + f'A = {A:.1e}\nk  = {k:.2f}')
# [b, a], results = ufun.fitLine(Xfit, Yfit)
# A, k = np.exp(b), a
# R2 = results.rsquared
# pval = results.pvalues[1]

def funFit(B, X):
    return(B[0]*X + B[1])

linear = odr.Model(funFit)
mydata = odr.Data(Xfit, Yfit, wd=1, we=1)
myodr = odr.ODR(mydata, linear, beta0=[-1.5, 2.])
myoutput = myodr.run()
a, b = myoutput.beta
A, k = np.exp(b), a
perc, dof, = 0.975, len(Yfit)-2
q = st.t.ppf(perc, dof)
alphaCiw = myoutput.sd_beta[0] * q
R2 = -1
pval = -1

Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k


ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 2.5,
        label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')

# ax.legend(fontsize = 9, loc = 'lower left')
# ax.set_title(f'Average per cell - N = {len(Xfit)}')
# ax.set_ylabel('$E_{500}$ (kPa)')
# ax.set_xlabel('$H_0$ (nm)')

for ax in [axes]:
    ax.grid(visible=True, which='major', axis='both')
    ax.set_xlim([50, 1100])

# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
figSubDir = 'E-h'
name = 'E500_vs_h0_V2'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = '', cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 600,
                figDir = figDir, figSubDir = '', cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, '', name+'_count.txt'), sep='\t')


# %%%% Figure NC6 - Compare fits
figDir = os.path.join(cp.DirDataFig, 'Paper')
apm.set_defense_options_jv(palette = 'Set2')

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix
XCol = 'bestH0'
YCol = 'E_f_<_500'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x not in excluded_dates)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_500'] <= 2e4),
           (df['valid_f_<_500'] == True), 
           ]

df_f = apm.filterDf(df, Filters)
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)

# Order
co_order = []

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_500', weightCol = 'ciwE_f_<_500', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(2,1, figsize=(17/apm.cm_in,20/apm.cm_in), sharex=True)

#### Part 1 - All comps

ax = axes[0]
ax.set_xscale('log')
ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 0.5)


sns.scatterplot(ax = ax, x=df_f['bestH0'].values, y=df_f['E_f_<_500'].values/1000, 
                marker = 'o', s = 15, color = 'gray', alpha = 0.5)
Xfit, Yfit = np.log(df_f['bestH0'].values), np.log(df_f['E_f_<_500'].values/1000)

# ODR
def funFit(B, X):
    return(B[0]*X + B[1])
linear = odr.Model(funFit)
mydata = odr.Data(Xfit, Yfit, wd=1, we=1)
myodr = odr.ODR(mydata, linear, beta0=[-1.5, 2.])
myoutput = myodr.run()
a, b = myoutput.beta
A, k = np.exp(b), a
perc, dof, = 0.975, len(Yfit)-2
q = st.t.ppf(perc, dof)
alphaCiw = myoutput.sd_beta[0] * q
R2 = -1
pval = -1

Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
ax.plot(Xplot, Yplot, ls = '--', c = 'black', lw = 1.5,
        label =  r'$\bf{Orthogonal\ Fit}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')

# OLS
[b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
A, k = np.exp(b), a
R2 = w_results.rsquared
[b, a], results = ufun.fitLine(Xfit, Yfit)
pval = results.pvalues[1]

Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 1.5,
        label =  r'$\bf{Ordinary\ Fit}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')

ax.legend(fontsize = 9, loc = 'lower left')
ax.set_title(f'All compressions - N = {len(Xfit)}')
ax.set_ylabel('$E_{500}$ (kPa)')
ax.set_xlabel('')

color = apm.cL_Set2[0]



#### Part 2 - Per cell

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')

sns.scatterplot(ax = ax, x=df_plot['bestH0'].values, y=df_plot['E_f_<_500_wAvg'].values/1000, 
                marker = 'o', s = 45, color = color, alpha = 0.5)

Xfit, Yfit = np.log(df_plot['bestH0'].values), np.log(df_plot['E_f_<_500_wAvg'].values/1000)

def funFit(B, X):
    return(B[0]*X + B[1])
linear = odr.Model(funFit)
mydata = odr.Data(Xfit, Yfit, wd=1, we=1)
myodr = odr.ODR(mydata, linear, beta0=[-1.5, 2.])
myoutput = myodr.run()
a, b = myoutput.beta
A, k = np.exp(b), a
perc, dof, = 0.975, len(Yfit)-2
q = st.t.ppf(perc, dof)
alphaCiw = myoutput.sd_beta[0] * q
R2 = -1
pval = -1

Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 2.5,
        label =  r'$\bf{Orthogonal\ Fit}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
    

[b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
A, k = np.exp(b), a
[b, a], results = ufun.fitLine(Xfit, Yfit)
R2 = results.rsquared
pval = results.pvalues[1]

Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 2.5,
        label =  r'$\bf{Ordinary\ Fit}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
    
ax.legend(fontsize = 9, loc = 'lower left')
ax.set_title(f'Average per cell - N = {len(Xfit)}')
ax.set_ylabel('$E_{500}$ (kPa)')
ax.set_xlabel('$H_0$ (nm)')

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    ax.set_xlim([50, 1100])

# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
figSubDir = 'E-h'
name = 'E500_vs_h0_CompareFits'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = '', cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                figDir = figDir, figSubDir = '', cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, '', name+'_count.txt'), sep='\t')


# %%%% Figure NC6 - Compare binning approach with fits
figDir = os.path.join(cp.DirDataFig, 'Paper')
apm.set_defense_options_jv(palette = 'Set2')

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix
XCol = 'bestH0'
YCol = 'E_f_<_500'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x not in excluded_dates)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
            (df['E_f_<_500'] <= 2e4),
           # (df['E_f_<_500'] <= 1e5),
           (df['valid_f_<_500'] == True), 
           ]

df_f = apm.filterDf(df, Filters)
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)

# Order
co_order = []

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_500', weightCol = 'ciwE_f_<_500', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, ax = plt.subplots(1,1, figsize=(24/apm.cm_in, 16/apm.cm_in), sharex=True)

#### All comps

ax = ax
ax.set_xscale('log')
ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 0.5)


sns.scatterplot(ax = ax, x=df_f['bestH0'].values, y=df_f['E_f_<_500'].values/1000, 
                marker = 'o', s = 15, color = 'gray', alpha = 0.5)
Xfit, Yfit = np.log(df_f['bestH0'].values), np.log(df_f['E_f_<_500'].values/1000)


# OLS with X
[b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
A, k = np.exp(b), a
R2 = w_results.rsquared
[b, a], results = ufun.fitLine(Xfit, Yfit)
pval = results.pvalues[1]

Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
ax.plot(Xplot, Yplot, ls = '--', c = 'blue', lw = 1.0,
        label =  r'$\bf{Ordinary\ Fit\ in\ X}$' + f'\nk  = {k:.2f}')

color = apm.cL_Set2[0]

# OLS with Y
[b, a], results, w_results = ufun.fitLineHuber(Yfit, Xfit, with_wlm_results = True)
A, k = np.exp(b), 1/a
R2 = w_results.rsquared
[b, a], results = ufun.fitLine(Yfit, Xfit)
pval = results.pvalues[1]

Yplot2 = np.exp(np.linspace(min(Yfit), max(Yfit), 50))
Xplot2 = A * Yplot2**a
ax.plot(Xplot2, Yplot2, ls = '--', c = 'red', lw = 1.0,
        label =  r'$\bf{Ordinary\ Fit\ in\ Y}$' + f'\nk  = {k:.2f}')

# ax.legend(fontsize = 9, loc = 'lower left')
# ax.set_title(f'All compressions - N = {len(Xfit)}')
# ax.set_ylabel('$E_{500}$ (kPa)')
# ax.set_xlabel('')

color = apm.cL_Set2[0]

# Binning
NperBin = 40
Ntotal = len(Xfit)
Nbins = round(Ntotal / NperBin)
Mat = np.array([Xfit, Yfit]).T
MatS = Mat[Mat[:, 0].argsort()]
XfitS, YfitS = MatS[:, 0].T, MatS[:, 1].T
Xbins = [XfitS[k*NperBin:min(k*NperBin+NperBin, Ntotal)] for k in range(Nbins)]
Ybins = [YfitS[k*NperBin:min(k*NperBin+NperBin, Ntotal)] for k in range(Nbins)]
XbinAvg = np.array([np.mean(b) for b in Xbins])
YbinAvg = np.array([np.mean(b) for b in Ybins])
[b, a], results, w_results = ufun.fitLineHuber(XbinAvg, YbinAvg, with_wlm_results = True)
k = a

ax.plot(np.exp(XbinAvg), np.exp(YbinAvg), ls='', 
        marker='o', markerfacecolor = 'white', markeredgecolor = 'b', ms = 7,
        label = r'$\bf{Bin\ along\ X\ and\ Fit}$' + f'\nk  = {k:.2f}')

MatS = Mat[Mat[:, 1].argsort()]
XfitS, YfitS = MatS[:, 0].T, MatS[:, 1].T
Xbins = [XfitS[k*NperBin:min(k*NperBin+NperBin, Ntotal)] for k in range(Nbins)]
Ybins = [YfitS[k*NperBin:min(k*NperBin+NperBin, Ntotal)] for k in range(Nbins)]
XbinAvg = np.array([np.mean(b) for b in Xbins])
YbinAvg = np.array([np.mean(b) for b in Ybins])
[b, a], results, w_results = ufun.fitLineHuber(XbinAvg, YbinAvg, with_wlm_results = True)
k = a

ax.plot(np.exp(XbinAvg), np.exp(YbinAvg), ls='', 
        marker='o', markerfacecolor = 'white', markeredgecolor = 'r', ms = 7,
        label = r'$\bf{Bin\ along\ Y\ and\ Fit}$' + f'\nk  = {k:.2f}')


# ODR
def funFit(B, X):
    return(B[0]*X + B[1])
linear = odr.Model(funFit)
mydata = odr.Data(Xfit, Yfit, wd=1, we=1)
myodr = odr.ODR(mydata, linear, beta0=[-1.5, 2.])
myoutput = myodr.run()
a, b = myoutput.beta
A, k = np.exp(b), a
perc, dof, = 0.975, len(Yfit)-2
q = st.t.ppf(perc, dof)
alphaCiw = myoutput.sd_beta[0] * q
R2 = -1
pval = -1

Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
ax.plot(Xplot, Yplot, ls = '--', c = 'black', lw = 1.0,
        label =  r'$\bf{Orthogonal\ Fit}$' + f'\nk  = {k:.2f}')

ax.legend(fontsize = 9, loc = 'lower left', ncols = 3)
ax.set_title(f'All compressions - N = {len(Xfit)}')
ax.set_ylabel('$E_{500}$ (kPa)')
ax.set_xlabel('')
ax.grid(visible=True, which='major', axis='both')
ax.set_xlim([50, 1100])
ax.set_ylim([0.8, 21])

# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
figSubDir = 'E-h'
name = 'E500_vs_h0_CompareFitsAndBins'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = '', cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                figDir = figDir, figSubDir = '', cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, '', name+'_count.txt'), sep='\t')


# %% ------

# %% Working with stress or strain ranges ?

# %%% Explore Stress Ranges

df = MecaData_Phy
dates = ['23-02-16', '23-03-16', '23-04-26', '24-12-11']

XCol = 'bestH0'
YCol = 'E_f_<_500'

cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x in dates)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
            (df['minStress'] >= 0),
            (df['maxStress'] <= 1e4), 
           ]

df_f = apm.filterDf(df, Filters)

fig, ax = apm.StressRange_2D_V2(df_f, fig=None, ax=None, colorList = apm.colorList30, 
                      condition='cell type', defaultMode = True)

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x in dates)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['minStrain'] >= 0),
           (df['maxStrain'] <= 0.8), 
           ]

df_f = apm.filterDf(df, Filters)

fig2, ax2 = apm.StrainRange_2D_V2(df_f, fig=None, ax=None, colorList = apm.colorList30, 
                      condition='cell type', defaultMode = True)

plt.show()

# %% ------

# %% Beads size
MecaData_BeadSizes = taka3.getMergedTable('MecaData_Physics_BeadSizes_V2')
MecaData_BeadSizes = MecaData_Phy.dropna(axis=0, subset='date')

# %%% Non-linearity

apm.set_manuscript_options_jv()

# Define
df = MecaData_BeadSizes
# 
# 
drugs = ['none']
dates = ['21-10-18', '21-10-25', '21-12-08', '21-12-16', '22-01-12']
substrate = '20um fibronectin discs'

df, condCol = apm.makeCompositeCol(df, cols=['inside bead type'])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['bestH0'] <= 1000),
           ]
df_f = apm.filterDf(df, Filters)

# Order
co_order = ['M270-2022', 'M450-2022']


# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Ncells = CountByCond.cellCount.values[0]
# Ncomps = CountByCond.compCount.values[0]

# Group By
# df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(3, 1, figsize=(17/apm.cm_in, 12/apm.cm_in), sharex='col', sharey='col')

intervals = ['200_500', '300_700', '400_900']
cL = plt.cm.plasma(np.linspace(0.1, 0.9, len(intervals)))

rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbi 10µM', 
      'blebbistatin & 50.0' : 'Blebbi 50µM', 
      'blebbistatin & 100.0' : 'Blebbi 100µM',
      'blebbistatin & 250.0' : 'Blebbi 250µM', 
      }

for k, cond in enumerate(co_order):  
    for i, interval in enumerate(intervals):
        ax = axes[i]
        ax.set_yscale('log')
        df_fc = df_f[df_f[condCol] == cond]        
        
        # colorDict = {cond:cL[i]}
        # labelDict = {cond:f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"}
        colorDict = {cond:apm.styleDict[cond]['color']}
        labelDict = {cond:rD[cond]}
        apm.plotPopKS_V2(df_fc, fig = fig, ax = ax, condition = condCol, co_order = [], 
                     colorDict = colorDict, labelDict = labelDict, markersizefactor = 0.75,
                     fitType = 'stressGaussian', fitWidth=75, mode = interval, Sinf = 0, Ssup = np.inf)
    
        # Prettify
        ax.grid(visible=True, which='both', axis='y')
        # legendTitle = r'$\bf{Whole\  dataset}$ | ' + f'{Ncells} cells | {Ncomps} comp'
        legendTitle = r"$\sigma \in$" + f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"
        ax.legend(loc='lower right', fontsize = 6, title = legendTitle, title_fontsize = 8)
        # ax.legend().set_visible(False)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0.9, 30)
        if i != len(intervals)-1:
            ax.set_xlabel('')
            
      

# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
# figSubDir = 'Blebbi'
# drugSuffix = '_Blebbi'
# name = 'K-sigma-3ranges' + drugSuffix
# figSubDir = ''
# drugPrefix = 'Blebbi_'
# name = drugPrefix + 'K-sigma_and_Kbp'
# ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
#                 figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
# ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
#                 figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
# CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %% ------

# %% Drugs






# %% ------

# %% Cell Types

# %%% Data import & export - MecaData_Cells

MecaData_Cells = taka3.getMergedTable('MecaData_CellTypes_V2')
# MecaData_Cells = taka3.getMergedTable('MecaData_CellTypes_wMDCK')

MecaData_Phy = taka3.getMergedTable('MecaData_Physics_V3')

# MecaData_HeLa = taka3.getMergedTable('MecaData_HeLaFucci_V1')

# %%% Merge

MecaData_CellsFull = pd.concat([MecaData_Cells, MecaData_Phy])

# MecaData_Cells2['Indent_ID'] = MecaData_Cells2['cellID'] + '_' + MecaData_Cells2['compNum'].astype('str')
# MecaData_Cells2 = MecaData_Cells2.drop_duplicates(subset='Indent_ID')

# path = DirDataAnalysis + "/MecaData_CellTypes_V3.csv"
# MecaData_Cells2.to_csv(path, index=False)


# %%% Check content

print('Dates')
print([x for x in MecaData_Cells['date'].unique()])
print('')

print('Cell types')
print([x for x in MecaData_Cells['cell type'].unique()])
print('')

print('Cell subtypes')
print([x for x in MecaData_Cells['cell subtype'].unique()])
print('')

print('Drugs')
print([x for x in MecaData_Cells['drug'].unique()])
print('')

print('Substrates')
print([x for x in MecaData_Cells['substrate'].unique()])
print('')

print('Resting Fields')
print([x for x in MecaData_Cells['normal field'].unique()])
print('')

# %%% Plots - Cell types - Paper

figDir = os.path.join(cp.DirDataFig, 'Paper')
figSubDir = 'CellTypes'

df = MecaData_Cells
print(df['cell type'].unique())
print(df['cell subtype'].unique())

co_order = ['3T3 & Atcc-2023', 
            'HeLa & fucci', 
            'MDCK & WT',
            'DC & mouse-primary', 
            'HoxB8-Macro & ctrl', 
            'Dicty & DictyBase-WT',
            ]

rD = {'3T3 & Atcc-2023'      : '3T3 ATCC', 
      'HeLa & fucci'         : 'HeLa Fucci',  
      'DC & mouse-primary'   : 'Primary DC',  
      'Dicty & DictyBase-WT' : 'Dictys Ax3',  
      'HoxB8-Macro & ctrl'   : 'HoxB8 Macro',  
      'MDCK & WT'            : 'MDCK',
      }

colorsD = {'3T3 & Atcc-2023'      : apm.cL_Set2[0], 
           'HeLa & fucci'         : apm.cL_Set2[1],  
           'DC & mouse-primary'   : apm.cL_Set2[2],  
           'Dicty & DictyBase-WT' : apm.cL_Set2[3],  
           'HoxB8-Macro & ctrl'   : apm.cL_Set2[4],  
           'MDCK & WT'            : apm.cL_Set2[5],
           }


# %%% E_400 vs H0

apm.set_manuscript_options_jv()

# Define
df = MecaData_Cells
excluded_subtypes = ['tko', 'aSFL-A11']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs', 'bare glass']

XCol = 'bestH0'
YCol = 'E_f_<_400'

# Filter
Filters = [(df['validatedThickness'] == True), 
           # (df['substrate'].apply(lambda x : x in substrates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['valid_f_<_400'] == True),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = apm.filterDf(df, Filters)
df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

CellSubstrates = ['3T3 & 20um fibronectin discs',
                  'HeLa & 20um fibronectin discs',
                  'MDCK & 20um fibronectin discs',
                  'DC & BSA coated glass',
                  'HoxB8-Macro & bare glass',
                  'Dicty & BSA coated glass',
                  ]
df_f, cc = apm.makeCompositeCol(df_f, cols=['cell type', 'substrate'])
Filters2 = [
            (df_f['cell type & substrate'].apply(lambda x : x in CellSubstrates)),
            (df_f[YCol] <=  1e5),
           ]
df_f = apm.filterDf(df_f, Filters2)



df_f, condCol = apm.makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])
df_f = apm.computeNLMetrics_V2(df_f, th_NLI = np.log10(2), ref_strain = 0.2)

# Order
## See the style cell

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = YCol, weightCol = 'ciw' + YCol, weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(2, 3, figsize=(17/apm.cm_in, 12/apm.cm_in), sharex=True, sharey=True)
axes = axes.flatten('C')

# ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

YCol += '_wAvg'
df_plot[YCol] /= 1000

for i in range(len(axes)):
    ax = axes[i]
    ax.set_xscale('log')
    ax.set_yscale('log')
    df_fc = df_plot[df_plot[condCol] == co_order[i]]
    print(co_order[i], len(df_fc))
    # color = apm.cL_Set2[i]
    color = colorsD[co_order[i]]
    
    # fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
    #                 XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
    #                 modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
    #                 figSizeFactor = 1, markersizeFactor = 0.5)
    
        
    sns.scatterplot(ax = ax, x=df_fc[XCol].values, y=df_fc[YCol].values, 
                    marker = 'o', s = 20, color = color, edgecolor = 'k', linewidth =0.5, alpha = 0.5,
                    zorder = 3)
    Xfit, Yfit = np.log(df_fc[XCol].values), np.log(df_fc[YCol].values)
    
    [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
    A, k = np.exp(b), a
    k_cihw = (results.conf_int(0.05)[1, 1] - results.conf_int(0.05)[1, 0])/2
    R2 = w_results.rsquared
    pval = results.pvalues[1]
    Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = A * Xplot**k
    
    # [b, a], results = ufun.fitLine(Xfit, Yfit)
    # R2 = results.rsquared
    # pval = results.pvalues[1]
    # Xplot = (np.linspace(min(Xfit), max(Xfit), 50))
    # Yplot = a * Xplot + b
    
    ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 2.0, zorder = 6,
            label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + \
                        f'\nA = {A:.1e}' + \
                        f'\nk  = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
                        f'\n$R^2$ = {R2:.2f}' + \
                        f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 6, loc = 'best', handlelength=1)
    ax.set_xlabel('$H_{0}$ (nm)')
    ax.set_ylabel('$E_{400}$ (kPa)')
    ax.set_title(co_order[i])
    if i%3 != 0:
        ax.set_ylabel('')
           
# Prettify
rD.update({'E_eff_wAvg':'E_{eff} (kPa)'})

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder=0)
    apm.renameAxes(ax, rD, format_xticks = False)
    # apm.renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD)
    ax.set_xlim(50, 2000)
    ax.set_ylim(0.4, 300)

# axes[0].set_xlabel('')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
name = 'CellTypes_HE400'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%% E_500 vs H0

apm.set_manuscript_options_jv()

# Define
df = MecaData_Cells
excluded_subtypes = ['tko', 'aSFL-A11']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs']

XCol = 'bestH0'
YCol = 'E_f_<_500'


# Filter
Filters = [(df['validatedThickness'] == True), 
           # (df['substrate'].apply(lambda x : x in substrates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['valid_f_<_500'] == True),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = apm.filterDf(df, Filters)

df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

CellSubstrates = ['3T3 & 20um fibronectin discs',
                  'HeLa & 20um fibronectin discs',
                  'MDCK & 20um fibronectin discs',
                  'DC & BSA coated glass',
                  'HoxB8-Macro & bare glass',
                  'Dicty & BSA coated glass',
                  ]
df_f, cc = apm.makeCompositeCol(df_f, cols=['cell type', 'substrate'])
Filters2 = [
            (df_f['cell type & substrate'].apply(lambda x : x in CellSubstrates)),
            (df_f[YCol] <=  1e5),
           ]
df_f = apm.filterDf(df_f, Filters2)

df_f, condCol = apm.makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])
df_f = apm.computeNLMetrics_V2(df_f, th_NLI = np.log10(2), ref_strain = 0.2)

# YCol = 'E_eff'

# Filter
Filters = [
           (df_f[YCol] <=  1e5),
           ]
df_f = apm.filterDf(df_f, Filters)

# Order
## See the style cell

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = YCol, weightCol = 'ciw' + YCol, weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(2, 3, figsize=(17/apm.cm_in, 12/apm.cm_in), sharex=True, sharey=True)
axes = axes.flatten('C')

# ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

YCol += '_wAvg'
df_plot[YCol] /= 1000

for i in range(len(axes)):
    ax = axes[i]
    ax.set_xscale('log')
    ax.set_yscale('log')
    df_fc = df_plot[df_plot[condCol] == co_order[i]]
    print(co_order[i], len(df_fc))
    # color = apm.cL_Set2[i]
    color = colorsD[co_order[i]]
    
    # fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
    #                 XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
    #                 modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
    #                 figSizeFactor = 1, markersizeFactor = 0.5)
    
        
    sns.scatterplot(ax = ax, x=df_fc[XCol].values, y=df_fc[YCol].values, 
                    marker = 'o', s = 20, color = color, edgecolor = 'k', linewidth =0.5, alpha = 0.5,
                    zorder = 3)
    Xfit, Yfit = np.log(df_fc[XCol].values), np.log(df_fc[YCol].values)
    
    [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
    A, k = np.exp(b), a
    k_cihw = (results.conf_int(0.05)[1, 1] - results.conf_int(0.05)[1, 0])/2
    R2 = w_results.rsquared
    pval = results.pvalues[1]
    Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = A * Xplot**k
    
    # [b, a], results = ufun.fitLine(Xfit, Yfit)
    # R2 = results.rsquared
    # pval = results.pvalues[1]
    # Xplot = (np.linspace(min(Xfit), max(Xfit), 50))
    # Yplot = a * Xplot + b
    
    ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 2.0, zorder = 6,
            label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + \
                        f'\nA = {A:.1e}' + \
                        f'\nk  = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
                        f'\n$R^2$ = {R2:.2f}' + \
                        f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 6, loc = 'best', handlelength=1)
    ax.set_xlabel('$H_{0}$ (nm)')
    ax.set_ylabel('$E_{500}$ (kPa)')
    ax.set_title(co_order[i])
    if i%3 != 0:
        ax.set_ylabel('')
           
# Prettify
rD.update({'E_eff_wAvg':'E_{eff} (kPa)'})

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder=0)
    apm.renameAxes(ax, rD, format_xticks = False)
    # apm.renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD)
    ax.set_xlim(50, 2000)
    ax.set_ylim(0.4, 300)

# axes[0].set_xlabel('')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
name = 'CellTypes_HE500'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')



# %%% E_600 vs H0

apm.set_manuscript_options_jv()

# Define
df = MecaData_Cells
excluded_subtypes = ['tko', 'aSFL-A11']
drugs = ['dmso', 'none']
# substrates = ['BSA coated glass', '20um fibronectin discs']
substrates = ['bare glass']

XCol = 'bestH0'
YCol = 'E_f_<_600'


# Filter
Filters = [(df['validatedThickness'] == True), 
            # (df['substrate'].apply(lambda x : x in substrates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['valid_f_<_600'] == True),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = apm.filterDf(df, Filters)

df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = apm.makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])
df_f = apm.computeNLMetrics_V2(df_f, th_NLI = np.log10(2), ref_strain = 0.2)

# YCol = 'E_eff'

# Filter
Filters = [
           (df_f[YCol] <=  1e5),
           ]
df_f = apm.filterDf(df_f, Filters)

# Order
## See the style cell

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = YCol, weightCol = 'ciw' + YCol, weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(2, 3, figsize=(17/apm.cm_in, 12/apm.cm_in), sharex=True, sharey=True)
axes = axes.flatten('C')

# ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

YCol += '_wAvg'
df_plot[YCol] /= 1000

for i in range(len(axes)):
    ax = axes[i]
    ax.set_xscale('log')
    ax.set_yscale('log')
    try:
        df_fc = df_plot[df_plot[condCol] == co_order[i]]
        print(co_order[i], len(df_fc))
        # color = apm.cL_Set2[i]
        color = colorsD[co_order[i]]
        
        # fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
        #                 XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
        #                 modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
        #                 figSizeFactor = 1, markersizeFactor = 0.5)
        
            
        sns.scatterplot(ax = ax, x=df_fc[XCol].values, y=df_fc[YCol].values, 
                        marker = 'o', s = 20, color = color, edgecolor = 'k', linewidth =0.5, alpha = 0.5,
                        zorder = 3)
        Xfit, Yfit = np.log(df_fc[XCol].values), np.log(df_fc[YCol].values)
        
        [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
        A, k = np.exp(b), a
        k_cihw = (results.conf_int(0.05)[1, 1] - results.conf_int(0.05)[1, 0])/2
        R2 = w_results.rsquared
        pval = results.pvalues[1]
        Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
        Yplot = A * Xplot**k
        
        # [b, a], results = ufun.fitLine(Xfit, Yfit)
        # R2 = results.rsquared
        # pval = results.pvalues[1]
        # Xplot = (np.linspace(min(Xfit), max(Xfit), 50))
        # Yplot = a * Xplot + b
        
        ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 2.0, zorder = 6,
                label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + \
                            f'\nA = {A:.1e}' + \
                            f'\nk  = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
                            f'\n$R^2$ = {R2:.2f}' + \
                            f'\np-val = {pval:.2f}')
    
        ax.legend(fontsize = 6, loc = 'best', handlelength=1)
        ax.set_xlabel('$H_{0}$ (nm)')
        ax.set_ylabel('$E_{600}$ (kPa)')
        ax.set_title(co_order[i])
        if i%3 != 0:
            ax.set_ylabel('')
            
    except:
        pass
           
# Prettify
rD.update({'E_eff_wAvg':'E_{eff} (kPa)'})

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder=0)
    apm.renameAxes(ax, rD, format_xticks = False)
    # apm.renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD)
    ax.set_xlim(50, 2000)
    ax.set_ylim(0.4, 300)

# axes[0].set_xlabel('')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
name = 'CellTypes_HE600'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %% ------



# %% Error analysis


def fitChadwick_hf(h, f, D, PLOT = False):
    """
    Fit the Chadwick model on a force-thickness curve, using the inversed model.
    This means the X-variable is f and the Y-variable is h.

    Parameters
    ----------
    h : numpy array
        Array of cortical thickness in µm.
    f : numpy array
        Array of pinching forces in pN.
    D : float
        Diameter of the beads indenting the cortex, in µm.

    Returns
    -------
    params : (2 x 1) numpy array
        Parameters values as: [E, H0].
    ses : (2 x 1) numpy array
        Standard errors for the parameters: [se(E), se(H0)].
    ciw : (2 x 1) numpy array
        Confidence interval width: [ciw(E), ciw(H0)].
    istats : (2 x 1) numpy array
        Statistical indicators: [R2, Chi2].
    error : bool
        Error during the fit.
        
    Note
    -------
    Units in the fits: nm, pN, µPa; that is why the modulus will be multiplied by 1e6.
    """
    
    R = D/2
    Npts = len(h)
    error = False
    
    def chadwickModel(h, E, H0):
        f = (np.pi*E*R*((H0-h)**2))/(3*H0)
        return(f)

    def inversedChadwickModel(f, E, H0):
        h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
        return(h)

    # try:
    # some initial parameter values - must be within bounds
    initH0 = max(h) # H0 ~ h_max
    initE = (3*max(h)*max(f))/(np.pi*(R)*(max(h)-min(h))**2) # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
    
    initialParameters = [initE, initH0]
    
    # bounds on parameters - initial parameters must be within these
    lowerBounds = (0, 0)
    upperBounds = (np.inf, np.inf)
    parameterBounds = [lowerBounds, upperBounds]
    
    # params = [E, H0] ; ses = [seE, seH0]
    params, covM = curve_fit(inversedChadwickModel, f, h, p0=initialParameters, bounds = parameterBounds)
    ses = np.array([covM[0,0]**0.5, covM[1,1]**0.5])
    params[0], ses[0] = params[0]*1e6, ses[0]*1e6 # Convert E & seE to Pa
    
    # Ciw calculation
    E, H0 = params
    
    hPredict = inversedChadwickModel(f, E/1e6, H0)
    y, yPredict = h, hPredict
    
    seE, seH0 = ses
    alpha = 0.975
    nbPts = len(y)
    dof = nbPts - len(params)
    q = st.t.ppf(alpha, dof) # Student coefficient
    err_chi2 = 0.007
    R2 = ufun.get_R2(y, yPredict)
    Chi2 = ufun.get_Chi2(y, yPredict, dof, err_chi2)        
    ciwE = q*seE
    ciwH0 = q*seH0
    ciw = (ciwE, ciwH0)
    istats = (R2, Chi2)
        
    # except:
    #     error = True
    #     params = np.ones(2) * np.nan
    #     ses = np.ones(2) * np.nan
    #     ciw = np.ones(2) * np.nan
    #     istats = np.ones(2) * np.nan
    
    if PLOT:
        fig, ax = plt.subplots(1, 1)
        ax.plot(h, f, 'b.')
        ax.plot(hPredict, f, 'r-', lw=0.5)
        plt.show()
        
    res = (params, ses, ciw, istats, error)
        
    return(res)


def chadwickModel(h, E, H0, DIAMETER):
    """
    Implement the Chadwick formula with force as a function of thickness.

    Parameters
    ----------
    h : numpy array
        Array of cortical thickness in µm.
    E : float
        Elastic modulus of the cortex, in Pa.
    H0 : float
        Thickness of the cortex in the non-deformed configuration, in µm.
    DIAMETER : float
        Diameter of the beads indenting the cortex, in µm.

    Returns
    -------
    f : numpy array
        Array of pinching forces in pN.
    
    Reference
    -------
    Axisymmetric Indentation of a Thin Incompressible Elastic Layer, 
    R. S. Chadwick, 2002, https://doi.org/10.1137/S0036139901388222

    Note
    -------
    Compatible units: µm, pN, Pa ; or nm, pN, µPa.
    """
    
    R = DIAMETER/2
    f = (np.pi*E*R*((H0-h)**2))/(3*H0)
    return(f)


def inversedChadwickModel(f, E, H0, DIAMETER):
    """
    Implement the Chadwick formula with thickness as a function of force.

    Parameters
    ----------
    f : numpy array
        Array of pinching forces in pN.
    E : float
        Elastic modulus of the cortex, in Pa.
    H0 : float
        Thickness of the cortex in the non-deformed configuration, in µm.
    DIAMETER : float
        Diameter of the beads indenting the cortex, in µm.

    Returns
    -------
    h : numpy array
        Array of cortical thickness in µm.
        
    Reference
    -------
    Axisymmetric Indentation of a Thin Incompressible Elastic Layer, 
    R. S. Chadwick, 2002, https://doi.org/10.1137/S0036139901388222

    Note
    -------
    Compatible units: µm, pN, Pa ; or nm, pN, µPa.
    """
    
    R = DIAMETER/2
    h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
    return(h)

# %%% 0. Sanity checks

#### Check 1

# DIAMETER = 4.5
# H0 = 0.4
# E = 5000

# h_set = np.linspace(H0-0.02, H0/2, 50)
# f_set = np.linspace(50, 800, 50)

# h_comp = inversedChadwickModel(f_set, E, H0, DIAMETER)
# f_comp =         chadwickModel(h_set, E, H0, DIAMETER)

# fig, ax = plt.subplots(1, 1)
# ax.plot(h_set, f_comp, 'b.')
# ax.plot(h_comp, f_set, 'r-', lw=0.5)
# plt.show()

#### Check 2

# DIAMETER = 4.5
# H0 = 0.4
# E = 5000
# h_err = 0.05

# f_set = np.linspace(50, 800, 50)

# h_comp = inversedChadwickModel(f_set, E, H0, DIAMETER)
# h_bias = h_comp + h_err

# fig, ax = plt.subplots(1, 1)
# ax.plot(h_comp, f_set, 'b.')
# ax.plot(h_bias, f_set, 'g.')
# plt.show()

# params1, ses1, ciw1, istats1, error1 = fitChadwick_hf(h_comp*1000, f_set, DIAMETER*1000)
# params2, ses2, ciw2, istats2, error2 = fitChadwick_hf(h_bias*1000, f_set, DIAMETER*1000)
 
# E1, H01 = params1
# E2, H02 = params2

# f1 = chadwickModel(h_comp, E1, H01/1000, DIAMETER)
# f2 = chadwickModel(h_bias, E2, H02/1000, DIAMETER)

# ax.plot(h_comp, f1, 'r-', lw=0.5)
# ax.plot(h_bias, f2, 'r-', lw=0.5)
# plt.show()


#### Check 3

# DIAMETER = 4.5
# H0 = 0.4
# EE = np.arange(1000, 8500, 500)

# f_set = np.linspace(0, 800, 1000)

# fig, ax = plt.subplots(1, 1)

# for E in EE:
#     h_comp = inversedChadwickModel(f_set, E, H0, DIAMETER)
#     ax.plot(h_comp, f_set)

# plt.show()


#### Check 4

DIAMETER = 4.5
H0 = 0.4
E = 5000
h_err = 0.025

f_set = np.linspace(50, 800, 50)

h_comp = inversedChadwickModel(f_set, E, H0, DIAMETER)
h_bias = h_comp + h_err
f_bias = f_set  * ((DIAMETER+h_comp)**4/(DIAMETER+h_bias)**4)

fig, ax = plt.subplots(1, 1)
ax.plot(h_comp, f_set, 'b.')
ax.plot(h_bias, f_set, 'c.')
ax.plot(h_bias, f_bias, 'g.')
plt.show()

params1, ses1, ciw1, istats1, error1 = fitChadwick_hf(h_comp*1000, f_set, DIAMETER*1000)
params2, ses2, ciw2, istats2, error2 = fitChadwick_hf(h_bias*1000, f_bias, DIAMETER*1000)
 
E1, H01 = params1
E2, H02 = params2

f1 = chadwickModel(h_comp, E1, H01/1000, DIAMETER)
f2 = chadwickModel(h_bias, E2, H02/1000, DIAMETER)

ax.plot(h_comp, f1, 'r-', lw=0.5)
ax.plot(h_bias, f2, 'r-', lw=0.5)
plt.show()

# %%% 1. Fake data

DIAMETER = 4.5


E  = 5000
# H0 = 0.3
H0H0 = np.linspace(0.2, 0.6, 21)

f = np.linspace(50, 600, 50)
h = np.zeros((len(H0H0), len(f)))
# f = np.zeros((len(H0H0), 50))

#### Make the fake data

for k in range(len(H0H0)):
    H0 = H0H0[k]
    
    # f = np.linspace(50, 600, 50)
    # h = np.linspace(H0, H0/2, 50)
    
    # f[k, :] = chadwickModel(h, E, H0, DIAMETER)
    h[k, :] = inversedChadwickModel(f, E, H0, DIAMETER)
    

#### Fit the fake data

EfEf = []
H0fH0f = []
errorH = 0.025

for k in range(len(H0H0)):
    H0 = H0H0[k]
    h_fit = h[k, :] + errorH
    # f_fit = f * ((DIAMETER+h[k, :])**4/(DIAMETER+h_fit)**4)
    f_fit = f
    
    PLOT = (k%10 == 0)
    
    params, ses, ciw, istats, error = fitChadwick_hf(h_fit, f_fit, DIAMETER, PLOT)
    Efit, H0fit = params
    EfEf.append(Efit/1e6)
    H0fH0f.append(H0fit)

EfEf = np.array(EfEf)
Eratio = EfEf/E
Ediff = EfEf - E

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(H0H0, Eratio) 
ax[0].plot(H0H0, 1 + errorH/H0H0)
ax[1].plot(H0H0, Ediff)
plt.show()

H0fH0f = np.array(H0fH0f)
H0ratio = H0fH0f/H0H0
H0diff = H0fH0f - H0H0
    
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(H0H0, H0ratio) 
ax[1].plot(H0H0, H0diff)
plt.show()

# %%% 2. Actual dataset




