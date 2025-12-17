# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:21:04 2024

@author: JosephVermeil
"""

# %% > Imports and constants

#### Main imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
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
from scipy.stats import mannwhitneyu
from scipy import odr
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
# from matplotlib.gridspec import GridSpec

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

figDir = 'C:/Users/josep/Desktop/Seafile/PapierDensité/NewFigs'


# !!! Bug to correct !

# def fitLineTLS(X, Y):
#     """

#     """
#     def linearFun(B, X):
#         return(B[0]*X + B[1])
#     linear = odr.Model(linearFun)
#     data = odr.Data(X, Y, wd=1, we=1)
#     fit = odr.ODR(data, linear, beta0=[0, 0])
#     output = fit.run()
    
#     a, b = output.beta
#     sd_params = [k for k in output.sd_beta]
#     perc, dof, = 0.975, len(Y)-2
#     q = st.t.ppf(perc, dof)
#     ciw = [sd * q for sd in output.sd_beta]
    
#     beta_0 = 0  # test if slope is significantly different from zero
#     t_stat = [(output.beta[j] - beta_0) / output.sd_beta[j] for j in range(len(output.beta))]  # t statistic for the slope parameter
#     pvalues = [st.t.sf(np.abs(ts), dof) * 2 for ts in t_stat]
    
#     R2 = ufun.get_R2(Y, a*X+b)
#     results = ([a, b], sd_params, ciw, pvalues, R2)

#     out = ([a, b], results)
    
#     return(out)

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

# CountByCond, CountByCell = makeCountDf(MecaData_Phy, 'date')

# %% -------

# %% Plots EvH

# %%% 1. E500 small

# Save
SAVE = False
figSubDir = 'E-h'
name = 'E500_vs_h0_small'

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
           (df[XCol] < 1000),
           (df['normal field'] == 5),
           (df[YCol] <= 1e5),
           (df['valid' + YCol[1:]] == True), 
           ]

df_f = apm.filterDf(df, Filters)
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)

# Order
co_order = []

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[[XCol]]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = YCol, weightCol = 'ciw'+YCol, weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12/cm_in, 11/cm_in))
win, hin = 0.35, 0.35*(11/12)
xin, yin = 0.95-win, 0.93-hin 
ax_in = ax.inset_axes([xin, yin, win, hin])

ax = ax
ax.set_xscale('log')
ax.set_yscale('log')

sns.scatterplot(ax = ax, x=df_f[XCol].values, y=df_f[YCol].values/1000, 
                marker = 'o', s = 20, color = 'gray', alpha = 0.5, label='All compressions')
Xfit, Yfit = np.log(df_f[XCol].values), np.log(df_f[YCol].values/1000)

[a, b], results = ufun.fitLineTLS(Xfit, Yfit)
A, k = np.exp(b), a
pval = results.pvalues[0]
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
text_pval = apm.pval2text(pval, n_digits = 2, space = True)
ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 1.5,
        label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
                f'\nk  = {k:.2f}' + '\n' + text_pval)

ax.legend().set_visible(False)
# ax.legend(fontsize = 9, loc = 'lower left')
# ax.set_title('Average per cell')
ax.set_ylabel('$E_{500}$ (kPa)')
ax.set_xlabel('$H_0$ (nm)')
ax.grid(visible=True, which='major', axis='both')
ax.set_xlim([50, 1100])
ax.set_ylim([0.5, 500])
# ax.tick_params(axis='both', direction='in', which='both')
    
    
#### Inset
ax = ax_in
ax.set_xscale('log')
ax.set_yscale('log')

color = apm.cL_Set2[0]

sns.scatterplot(ax = ax, x=df_plot[XCol].values, y=df_plot[YCol+'_wAvg'].values/1000, 
                marker = 'o', s = 20, color = apm.cL_Set2[0], alpha = 0.6)
Xfit, Yfit = np.log(df_plot[XCol].values), np.log(df_plot[YCol+'_wAvg'].values/1000)

[a, b], results = ufun.fitLineTLS(Xfit, Yfit)
A, k = np.exp(b), a
pval = results.pvalues[0]
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
text_pval = apm.pval2text(pval, n_digits = 2, space = True)
ax.plot(Xplot, Yplot, ls = '--', c = apm.lightenColor(color, 0.7), lw = 1.5,)
        # label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
        #         f'\n$R^2$  = {R2:.2f}' + '\n' + text_pval)

# ax.legend(fontsize = 9, loc = 'lower left')
ax.set_title('Average per cell', fontsize=10)
ax.grid()
# ax.set_ylabel('$E_{500}$ (kPa)')
# ax.set_xlabel('$H_0$ (nm)')
ax.set_xlim([80, 1100])
ax.set_ylim([0.5, 50])
ax.tick_params(axis='both', direction='in', which='both', labelsize=9)
# ax.set_xticklabels(fontsize=9)
# ax.set_yticklabels(fontsize=9)

# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%% 1. E500 big

# Save
SAVE = False
figSubDir = 'E-h'
name = 'E500_vs_h0_big'

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
           (df[XCol] < 1000),
           (df['normal field'] == 5),
           (df[YCol] <= 1e5),
           (df['valid' + YCol[1:]] == True), 
           ]

df_f = apm.filterDf(df, Filters)
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)

# Order
co_order = []

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[[XCol]]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = YCol, weightCol = 'ciw'+YCol, weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, ax = plt.subplots(1, 1, figsize=(17/cm_in, 12/cm_in))
win, hin = 0.35, 0.35*(12/17)+0.05
xin, yin = 0.95-win, 0.93-hin 
ax_in = ax.inset_axes([xin, yin, win, hin])

ax = ax
ax.set_xscale('log')
ax.set_yscale('log')

sns.scatterplot(ax = ax, x=df_f[XCol].values, y=df_f[YCol].values/1000, 
                marker = 'o', s = 20, color = 'gray', alpha = 0.5, label='All compressions')
Xfit, Yfit = np.log(df_f[XCol].values), np.log(df_f[YCol].values/1000)

[a, b], results = ufun.fitLineTLS(Xfit, Yfit)
A, k = np.exp(b), a
pval = results.pvalues[0]
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
text_pval = apm.pval2text(pval, n_digits = 2, space = True)
ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 1.5,
        label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
                f'\nk  = {k:.2f}' + '\n' + text_pval)

# ax.legend().set_visible(False)
ax.legend(fontsize = 9, loc = 'lower left')
# ax.set_title('Average per cell')
ax.set_ylabel('$E_{500}$ (kPa)')
ax.set_xlabel('$H_0$ (nm)')
ax.grid(visible=True, which='major', axis='both')
ax.set_xlim([50, 1100])
ax.set_ylim([0.5, 500])
# ax.tick_params(axis='both', direction='in', which='both')
    
    
#### Inset
ax = ax_in
ax.set_xscale('log')
ax.set_yscale('log')

color = apm.cL_Set2[0]

sns.scatterplot(ax = ax, x=df_plot[XCol].values, y=df_plot[YCol+'_wAvg'].values/1000, 
                marker = 'o', s = 20, color = apm.cL_Set2[0], alpha = 0.6)
Xfit, Yfit = np.log(df_plot[XCol].values), np.log(df_plot[YCol+'_wAvg'].values/1000)

[a, b], results = ufun.fitLineTLS(Xfit, Yfit)
A, k = np.exp(b), a
pval = results.pvalues[0]
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
text_pval = apm.pval2text(pval, n_digits = 2, space = True)
ax.plot(Xplot, Yplot, ls = '--', c = apm.lightenColor(color, 0.7), lw = 1.5,)
        # label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
        #         f'\n$R^2$  = {R2:.2f}' + '\n' + text_pval)

# ax.legend(fontsize = 9, loc = 'lower left')
ax.set_title('Average per cell', fontsize=10)
ax.grid()
# ax.set_ylabel('$E_{500}$ (kPa)')
# ax.set_xlabel('$H_0$ (nm)')
ax.set_xlim([80, 1100])
ax.set_ylim([0.5, 50])
ax.tick_params(axis='both', direction='in', which='both', labelsize=9)
# ax.set_xticklabels(fontsize=9)
# ax.set_yticklabels(fontsize=9)

# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')