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
from scipy.stats import mannwhitneyu, shapiro
from scipy import odr
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
# from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)

import ArticlePlotMaker as apm
import UtilityFunctions as ufun
import TrackAnalyser_V3 as taka3
import TrackAnalyser_VPapierDensite as takaP


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
apm.setGraphicOptions(mode = 'print', 
                      palette = 'Set2', 
                      colorList = apm.cL_Set21)

figDir = 'C:/Users/josep/Desktop/Seafile/PapierDensité/DraftsFigs'


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
MecaData_Phy3 = takaP.getMergedTable('MecaData_Physics_V3')

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

# CountByCond, CountByCell =apm.makeCountDf(MecaData_Phy, 'date')

# %% -------

# %% Small diverse tasks

# %%% Get some data for Julien

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
Filters = [
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x not in excluded_dates)),
           (df['normal field'] == 5),
           ]
            
            # (df['validatedThickness'] == True), 
           # (df[XCol] < 1000)
           # (df[YCol] <= 1e5)
           # (df['valid' + YCol[1:]] == True)

df_f = apm.filterDf(df, Filters)

dstDir = os.path.join(cp.DirData, 'Data for Julien')
dstPath = os.path.join(dstDir, 'Main_EvH_Curve_Data.csv')

# df_f.to_csv(dstPath, index=False)

#### Part 2

list_Cid = df_f['cellID'].unique()

dstDir = os.path.join(cp.DirData, 'Data for Julien', 'Timeseries')
srcDir = cp.DirDataTimeseries
# dstDir = os.path.join(cp.DirData, 'Data for Julien', 'Timeseries_stress-strain')
# srcDir = cp.DirDataTimeseriesStressStrain

list_tsF = os.listdir(srcDir)
list_files_to_copy = []
for f in list_tsF:
    Cid = '_'.join(f.split('_')[:4])
    if Cid in list_Cid:
        list_files_to_copy.append(f)
    
for f in list_files_to_copy:
    # path = os.path.join(srcDir, f)
    # ufun.copyFile(srcDir, dstDir, f)
    continue
    

    
# %%% Sort data in t2 or t4

#### Get dataset

df = MecaData_Phy
# cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
# drugs = ['dmso'] #['none', 'dmso']
# substrate = '20um fibronectin discs'
# parameter = 'bestH0'
# df, condCol = apm.makeCompositeCol(df, cols=['drug'])
# excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# # figname = 'bestH0' + drugSuffix
# XCol = 'bestH0'
# YCol = 'E_f_<_500'

# # Filter
# Filters = [
#            (df['substrate'] == substrate),
#            (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
#            (df['drug'].apply(lambda x : x in drugs)),
#            (df['date'].apply(lambda x : x not in excluded_dates)),
#            (df['normal field'] == 5),
#            ]
            
#             # (df['validatedThickness'] == True), 
#            # (df[XCol] < 1000)
#            # (df[YCol] <= 1e5)
#            # (df['valid' + YCol[1:]] == True)

# df_f = apm.filterDf(df, Filters)
df_f = df


def drawPowerLine(ax, point, slope, **kwargs):
    k = slope
    x1, y1 = point
    A = y1/(x1**k)
    limX = ax.get_xlim()
    limY = ax.get_ylim()
    plotX = np.array([0.9*limX[0], 1.1*limX[1]])
    plotY = A*(plotX**k)
    ax.plot(plotX, plotY, **kwargs)
    ax.set_xlim(limX)
    ax.set_ylim(limY)
    

#### Look at tsDf

# Option 1. Take the CellIds from one table
# list_Cid = df_f['cellID'].unique()
# dict_Cid2File = {}

# srcDir = cp.DirDataTimeseries

# list_tsF = os.listdir(srcDir)
# for f in list_tsF:
#     Cid = '_'.join(f.split('_')[:4])
#     if Cid in list_Cid:
#         dict_Cid2File[Cid] = f

# dict_Bt_Exponent = {
#     'CellID'  : list(dict_Cid2File.keys()),
#     'TsFile'  : list(dict_Cid2File.values()),
#     'Exponent': np.ones(len(dict_Cid2File)),
#     }
# list_files_to_check = dict_Bt_Exponent['TsFile']

# Option 2. Take the filenames from the folder
list_tsF = os.listdir(srcDir)
dict_Cid2File = {}
for f in list_tsF:
    Cid = '_'.join(f.split('_')[:4])
    if not Cid in dict_Cid2File.keys():
        dict_Cid2File[Cid] = f

dict_Bt_Exponent = {
    'CellID'  : list(dict_Cid2File.keys()),
    'TsFile'  : list(dict_Cid2File.values()),
    'Exponent': np.ones(len(dict_Cid2File)),
    }
list_files_to_check = dict_Bt_Exponent['TsFile']
    
for i in range(len(list_files_to_check)):
    try:
        f = list_files_to_check[i]
        path = os.path.join(srcDir, f)
        tsDf = pd.read_csv(path, sep=';')
        T = tsDf[tsDf['idxAnalysis']==1]['T'].values
        B = tsDf[tsDf['idxAnalysis']==1]['B'].values
        i_Bmax = np.argmax(B)
        Bmax = B[i_Bmax]
        T = T[:i_Bmax]
        B = B[:i_Bmax]
        Tn = (T - np.min(T))/(np.max(T) - np.min(T))
        Bn = (B - np.min(B))/(np.max(B) - np.min(B))
        fitTn = np.log(Tn[10:])
        fitBn = np.log(Bn[10:])
        parms, results = ufun.fitLine(fitTn, fitBn)
        k = parms[1]
        k_round = int(np.round(k, decimals=0))
        dict_Bt_Exponent['Exponent'][i] = k_round
        # T1, B1 = Tn[30], Bn[30]
        # fig, ax = plt.subplots(1, 1)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # ax.plot(Tn[10:], Bn[10:])
        # drawPowerLine(ax, (T1, B1), 2)
        # drawPowerLine(ax, (T1, B1), 4)
    except:
        dict_Bt_Exponent['Exponent'][i] = -1 

df_Bt_Exponent = pd.DataFrame(dict_Bt_Exponent)

# T4
# 22-02-09_M2
# 22-02-09_M3
# 22-03-28
# 22-03-30
# 22-05-03
# 22-05-05
# 22-08-26
# 22-10-05
# 22-10-06

# %% Plots EvH

# %%% 1. E500_H0 
# %%%% 1.1 small

# Save
SAVE = False
figSubDir = 'E-h'
name = 'E500_vs_h0_small'

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
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

wd=1/(np.std(Xfit)) # **2
we=1/(np.std(Yfit)) # **2

[a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
A, k = np.exp(b), a
pval = results.pvalue_pearson
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
[k_ciw, b_ciw] = results.params_ciw
text_pval = apm.pval2text(pval, n_digits = 3, space = True)
ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 1.5,
        label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
                f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + '\n' + text_pval)

ax.legend().set_visible(False)
# ax.legend(fontsize = 9, loc = 'lower left')
# ax.set_title('Average per cell')
ax.set_ylabel('$E_{500}$ (kPa)')
ax.set_xlabel('$H_0$ (nm)')
ax.grid(visible=True, which='major', axis='both')
ax.set_xlim([50, 1100])
ax.set_ylim([0.5, 500])
# ax.tick_params(axis='both', direction='in', which='both')

hM, hL, hH = ufun.getLogNDistributionDescriptors(df_f[XCol].values)
EM, EL, EH = ufun.getLogNDistributionDescriptors(df_f[YCol].values/1000)
print(f'By compression, N = {len(df_f):.0f}')
print(f'For {XCol} vs {YCol}')
print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')
    
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
pval = results.pvalue_pearson
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
text_pval = apm.pval2text(pval, n_digits = 3, space = True)
ax.plot(Xplot, Yplot, ls = '--', c = apm.lightenColor(color, 0.7), lw = 1.5,)
        # label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + \
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

hM, hL, hH = ufun.getLogNDistributionDescriptors(df_plot[XCol].values)
EM, EL, EH = ufun.getLogNDistributionDescriptors(df_plot[YCol+'_wAvg'].values/1000)
print(f'By cells, N = {len(df_plot):.0f}')
print(f'For {XCol} vs {YCol}')
print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')

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

# %%%% 1.2 big

# Save
SAVE = False
figSubDir = 'E-h'
name = 'E500_vs_h0_big'

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
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

wd=1/(np.std(Xfit)) # **2
we=1/(np.std(Yfit)) # **2

[a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
A, k = np.exp(b), a
pval = results.pvalue_pearson
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
[k_ciw, b_ciw] = results.params_ciw
text_pval = apm.pval2text(pval, n_digits = 3, space = True)
ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 1.5,
        label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
                f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + '\n' + text_pval)

# ax.legend().set_visible(False)
ax.legend(fontsize = 9, loc = 'lower left')
# ax.set_title('Average per cell')
ax.set_ylabel('$E_{500}$ (kPa)')
ax.set_xlabel('$H_0$ (nm)')
ax.grid(visible=True, which='major', axis='both')
ax.set_xlim([50, 1100])
ax.set_ylim([0.5, 500])
# ax.tick_params(axis='both', direction='in', which='both')

hM, hL, hH = ufun.getLogNDistributionDescriptors(df_f[XCol].values)
EM, EL, EH = ufun.getLogNDistributionDescriptors(df_f[YCol].values/1000)
print(f'By compression, N = {len(df_f):.0f}')
print(f'For {XCol} vs {YCol}')
print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')
    
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
pval = results.pvalue_pearson
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
text_pval = apm.pval2text(pval, n_digits = 3, space = True)
ax.plot(Xplot, Yplot, ls = '--', c = apm.lightenColor(color, 0.7), lw = 1.5,)
        # label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + \
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

hM, hL, hH = ufun.getLogNDistributionDescriptors(df_plot[XCol].values)
EM, EL, EH = ufun.getLogNDistributionDescriptors(df_plot[YCol+'_wAvg'].values/1000)
print(f'By cells, N = {len(df_plot):.0f}')
print(f'For {XCol} vs {YCol}')
print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')

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
    
    
# %%% 2. E500_H500

# %%%% 2.0 check 


# Save
SAVE = False
figSubDir = 'E-h'
name = 'h500_vs_h0'

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix
XCol = 'bestH0'
YCol = 'H0_f_<_500'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x not in excluded_dates)),
           (df[XCol] < 1000),
           (df['normal field'] == 5),
           (df[YCol] <= 1e5),
           (df['valid' + YCol[2:]] == True), 
           ]

df_f = apm.filterDf(df, Filters)
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)

# Order
co_order = []

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()

# Plot
fig, axes = plt.subplots(2, 2, figsize=(17/cm_in, 12/cm_in), sharex=True, sharey='row')

ax = axes[0, 0]
ax.plot(df_f[XCol], df_f[YCol], c='gray', ls='', marker='.', markersize=5, alpha=0.3, zorder=6)
ax.axline((0, 0), slope = 1, ls='--', color='darkorange')

ax = axes[0, 1]
ax.plot(df_fg[XCol], df_fg[YCol], c=apm.cL_Set21[0], ls='', marker='.', markersize=5, alpha=0.4, zorder=6)
ax.axline((0, 0), slope = 1, ls='--', color='darkorange')

ax = axes[1, 0]
ax.plot(df_f[XCol], df_f[YCol]/df_f[XCol], c='gray', ls='', marker='.', markersize=5, alpha=0.3, zorder=6)
ax.axhline(1, ls='--', color='darkorange')

ax = axes[1, 1]
ax.plot(df_fg[XCol], df_fg[YCol]/df_fg[XCol], c=apm.cL_Set21[0], ls='', marker='.', markersize=5, alpha=0.4, zorder=6)
ax.axhline(1, ls='--', color='darkorange')

for ax in axes.flatten():
    ax.grid()


plt.show()

# %%%% Log normality

#### Save
SAVE = False
figSubDir = 'E-h'
name = 'h500 & E500 log-normality'

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix
XCol = 'H0_f_<_500'
YCol = 'E_f_<_500'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x not in excluded_dates)),
           (df[XCol] < 1000),
           (df['normal field'] == 5),
           (df[YCol] <= 2e5),
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


#### Plots

# df_qq = df_f
df_qq = df_plot
fig, axes = plt.subplots(1, 2, figsize=(17/cm_in, 12/cm_in), sharex=True, sharey='row')

for k, Col in enumerate([XCol, YCol+'_wAvg']): # +'_wAvg'
    
    data_lin = df_qq[Col].values
    data_log = np.log(df_qq[Col].values)
    
    ax = axes[k]
    ax.axline((0, 0), slope=1, color="k", linestyle='--', linewidth=1, zorder=6)
    
    data=data_lin
    shap_stat, shap_pval = shapiro(data)
    sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = apm.cL_Set21[5], markeredgecolor = 'None', markersize=6)
    ax.plot([], [], label=f'Normal distribution: {shap_pval:.2f}', ls='', marker='o', 
            markerfacecolor = apm.cL_Set21[5], markeredgecolor = 'None', markersize=4)
    
    data=data_log
    shap_stat, shap_pval = shapiro(data)
    sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = apm.cL_Set21[0], markeredgecolor = 'None', markersize=6)
    ax.plot([], [], label=f'Log-normal distribution: {shap_pval:.2f}', ls='', marker='o', 
            markerfacecolor = apm.cL_Set21[0], markeredgecolor = 'None', markersize=4)
    
    ax.set_aspect('equal')
    ax.set_xlim([-3.5,3.5])
    ax.set_ylim([-3.5,3.5])
    ax.legend(fontsize=7, title_fontsize=7, title = 'Shapiro–Wilk p-value', loc='lower right')
    ax.grid()
    
axes[0].set_title('Q-Q plots for $H_0$')
axes[1].set_title('Q-Q plots for $E_{500}$')
fig.tight_layout()
plt.show()


# %%%% 2.1 small

# Save
SAVE = True
figSubDir = 'E-h'
name = 'E500_vs_h500_small'

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix
XCol = 'H0_f_<_500'
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

#### Plot
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

wd=1/(np.std(Xfit)) # **2
we=1/(np.std(Yfit)) # **2

[a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
A, k = np.exp(b), a
pval = results.pvalue_pearson
[k_ciw, b_ciw] = results.params_ciw
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
text_pval = apm.pval2text(pval, n_digits = 3, space = True)
ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 1.5,
        label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
                f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + '\n' + text_pval)

# ax.legend().set_visible(False)
ax.legend(fontsize = 9, loc = 'lower left')
# ax.set_title('Average per cell')
ax.set_ylabel('$E_{500}$ (kPa)')
ax.set_xlabel('$H_{500}$ (nm)')
ax.grid(visible=True, which='major', axis='both')
ax.set_xlim([50, 1100])
ax.set_ylim([0.5, 500])
# ax.tick_params(axis='both', direction='in', which='both')

hM, hL, hH = ufun.getLogNDistributionDescriptors(df_f[XCol].values)
EM, EL, EH = ufun.getLogNDistributionDescriptors(df_f[YCol].values/1000)
print(f'By compression, N = {len(df_f):.0f}')
print(f'For {XCol} vs {YCol}')
print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')
    
#### Inset
ax = ax_in
ax.set_xscale('log')
ax.set_yscale('log')

color = apm.cL_Set2[0]

sns.scatterplot(ax = ax, x=df_plot[XCol].values, y=df_plot[YCol+'_wAvg'].values/1000, 
                marker = 'o', s = 20, color = apm.cL_Set2[0], alpha = 0.6)
Xfit, Yfit = np.log(df_plot[XCol].values), np.log(df_plot[YCol+'_wAvg'].values/1000)

wd=1/(np.std(Xfit)) # **2
we=1/(np.std(Yfit)) # **2

[a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
A, k = np.exp(b), a
pval = results.pvalue_pearson
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
text_pval = apm.pval2text(pval, n_digits = 3, space = True)
ax.plot(Xplot, Yplot, ls = '--', c = apm.lightenColor(color, 0.7), lw = 1.5,)
        # label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + \
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

hM, hL, hH = ufun.getLogNDistributionDescriptors(df_plot[XCol].values)
EM, EL, EH = ufun.getLogNDistributionDescriptors(df_plot[YCol+'_wAvg'].values/1000)
print(f'By cells, N = {len(df_plot):.0f}')
print(f'For {XCol} vs {YCol}')
print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')

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
    
    
    

# %%%% 2.2 big

# Save
SAVE = True
figSubDir = 'E-h'
name = 'E500_vs_h500_big'

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix
XCol = 'H0_f_<_500'
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

#### Plot
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

wd=1/(np.std(Xfit)) # **2
we=1/(np.std(Yfit)) # **2

[a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
A, k = np.exp(b), a
pval = results.pvalue_pearson
[k_ciw, b_ciw] = results.params_ciw
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
text_pval = apm.pval2text(pval, n_digits = 3, space = True)
ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 1.5,
        label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
                f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + '\n' + text_pval)

# ax.legend().set_visible(False)
ax.legend(fontsize = 9, loc = 'lower left')
# ax.set_title('Average per cell')
ax.set_ylabel('$E_{500}$ (kPa)')
ax.set_xlabel('$H_{500}$ (nm)')
ax.grid(visible=True, which='major', axis='both')
ax.set_xlim([50, 1100])
ax.set_ylim([0.5, 500])
# ax.tick_params(axis='both', direction='in', which='both')

hM, hL, hH = ufun.getLogNDistributionDescriptors(df_f[XCol].values)
EM, EL, EH = ufun.getLogNDistributionDescriptors(df_f[YCol].values/1000)
print(f'By compression, N = {len(df_f):.0f}')
print(f'For {XCol} vs {YCol}')
print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')
    
    
#### Inset
ax = ax_in
ax.set_xscale('log')
ax.set_yscale('log')

color = apm.cL_Set2[0]

sns.scatterplot(ax = ax, x=df_plot[XCol].values, y=df_plot[YCol+'_wAvg'].values/1000, 
                marker = 'o', s = 20, color = apm.cL_Set2[0], alpha = 0.6)
Xfit, Yfit = np.log(df_plot[XCol].values), np.log(df_plot[YCol+'_wAvg'].values/1000)

wd=1/(np.std(Xfit)) # **2
we=1/(np.std(Yfit)) # **2

[a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
A, k = np.exp(b), a
pval = results.pvalue_pearson
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
text_pval = apm.pval2text(pval, n_digits = 3, space = True)
ax.plot(Xplot, Yplot, ls = '--', c = apm.lightenColor(color, 0.7), lw = 1.5,)
        # label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + \
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

hM, hL, hH = ufun.getLogNDistributionDescriptors(df_plot[XCol].values)
EM, EL, EH = ufun.getLogNDistributionDescriptors(df_plot[YCol+'_wAvg'].values/1000)
print(f'By cells, N = {len(df_plot):.0f}')
print(f'For {XCol} vs {YCol}')
print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')

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
    
    
    
    

    
    

# %% --------

# %% Main Figure 1

# %%% h(t) and F(t)

plot_stressCenters = [ii for ii in range(100, 4000, 50)]
stressHalfWidths = [50, 75, 100]

fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['pts_15',
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_15',
                'doChadwickFit' : True,
                'ChadwickFitMethods' : ['Full', 'f_<_500'],
                'doVWCFit' : False,
                'doStressRegionFits' : False,
                'doStressGaussianFits' : False,
                'centers_StressFits' : plot_stressCenters,
                'halfWidths_StressFits' : stressHalfWidths,
                'doNPointsFits' : False,
                'nbPtsFit' : 33,
                'overlapFit' : 21,
                # NEW - Numi
                'doLogFits' : False,
                # NEW - Jojo
                'doStrainGaussianFits' : False,
                }

plot_stressCenters = [ii for ii in range(100, 2050, 100)]
plot_stressHalfWidth = 75

plotSettings = {# ON/OFF switchs plot by plot
                        'Plots_Papier':True,
                        'FH(t)':False,
                        'F(H)':False,
                        'F(H)_Dimitriadis':False,
                        'F(H)_VWC':False, # NEW - Numi
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':False,
                        'K(S)_stressGaussian':False,
                        'plotStressCenters':plot_stressCenters,
                        'plotStressHW':plot_stressHalfWidth,
                        'S(e)_nPoints':False,
                        'K(S)_nPoints':False,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        'Plot_Ratio':False
                        }

# =============================================================================
# # task = '24-03-13_M1_P1_C15'
# # task = '23-03-17_M4_P1_C15 & 23-03-17_M4_P1_C14 & 23-03-17_M4_P1_C8 & 24-07-04_M4_P1_C16' # 23-03-16_M1_P1_C2 & 
# # task = '24-07-04_M4_P1_C15'
# # task = '24-07-04_M6'
# # task = '23-03-17_M4'
# # task = '24-07-04_M6_P1_C11'
# # task = '23-03-09_M4_P1_C2 & 23-03-09_M4_P1_C5 & 23-03-09_M4_P1_C12'
# # task += ' & 23-03-09_M4_P1_C4 & 23-03-09_M4_P1_C8 & 23-03-09_M4_P1_C9'
# # task += ' & 23-03-09_M4_P1_C15 & 23-03-09_M4_P1_C14'
# # task = '23-03-17_M4_P1_C3 & 23-03-17_M4_P1_C9 & 23-03-17_M4_P1_C11'
# # task = '23-03-16_M1_P1_C4'
# =============================================================================

task = '24-04-11_M3_P1_C1'

res = takaP.computeGlobalTable_meca(mode = 'fromScratch', task = task, fileName = 'test', 
                                    save = False, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting'


# %%% Distribution H et E

# Source : Plotter_AtccPhysics - Figure NC1.1 - V4 - Thickness & Stiffness LOG SMALL

# Save
SAVE = True
figSubDir = 'F1'
name = 'F1_Distrib_H&E500'

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix
XCol = 'H0_f_<_500'
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
df_fgw2[YCol + '_wAvg'] /= 1000
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

#### Init fig
fig, axes = plt.subplots(1, 2, figsize=(12/cm_in, 6/cm_in), sharey=True)
color = apm.cL_Set2[0]

#### 01 - Best H0
ax = axes[0]

ax.set_title('Thickness')
ax.set_xlabel('Fitted $H_0$ (nm)')
ax.set_ylabel('Count (cells)')

ax, histo, logbins = apm.plot_logstairs(ax, df_fg[XCol].values, bins=12, logbins = [], 
                                    normalized = False, fill = True, color = color, label = '')
ax.set_xlim([0, ax.get_xlim()[1]])
medianH0 = np.median((df_fg[XCol].values))
ax.axvline(medianH0, c='darkred', label = f'Median\n$H_0$ = {medianH0:.0f} nm')
ax.set_xlim([50, 1000])
# ax.legend()

print(np.median(df_fg[XCol].values))
print(np.mean(np.log10(df_fg[XCol].values)))
print(10**(np.mean(np.log10(df_fg[XCol].values))))
print(np.std(np.log10(df_fg[XCol].values)))
print(10**(np.std(np.log10(df_fg[XCol].values))))

#### 02 - E_500
ax = axes[1]

ax.set_title('Stiffness')
ax.set_xlabel('$E_{500}$ (kPa)')
ax, histo, logbins = apm.plot_logstairs(ax, df_fgw2[YCol + '_wAvg'].values, bins=12, logbins = [], 
                                    normalized = False, fill = True, color = color, label = '')
ax.set_xlim([0, ax.get_xlim()[1]])
medianE500 = np.median(df_fgw2[YCol + '_wAvg'].values)
ax.axvline(medianE500, c='darkred', label = 'Median\n$E_{500}$ = ' + f'{medianE500:.2f} kPa')
ax.set_xlim([0.5, 50])
ax.set_ylim([0, 40])
# ax.legend()

print(np.median(df_fgw2[YCol + '_wAvg'].values))
print(np.mean(np.log10(df_fgw2[YCol + '_wAvg'].values)))
print(10**(np.mean(np.log10(df_fgw2[YCol + '_wAvg'].values))))
print(np.std(np.log10(df_fgw2[YCol + '_wAvg'].values)))
print(10**(np.std(np.log10(df_fgw2[YCol + '_wAvg'].values))))
    
# Prettify
rD = {
      'none' : 'No drug',
      'dmso' : 'DMSO', 
      XCol : 'Fitted $H_0$ (nm)',
      YCol + '_wAvg' : '$E_{500}$ (kPa)'
      }

for ax in axes[:]:
    apm.renameAxes(ax, rD, format_xticks = False)
    # apm.renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, renameDict)
    # ax.grid(visible=True, which='major', axis='y')
    
    
# Show
plt.tight_layout()
plt.show()

# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')




# %% Main Figure 2

# %%% 2A

# Save
SAVE = True
figSubDir = 'F2'
name = 'F2A_E500-vs-h500'

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix
XCol = 'H0_f_<_500'
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

#### Plot
fig, ax = plt.subplots(1, 1, figsize=(9/cm_in, 8.5/cm_in))
win, hin = 0.32, 0.35
xin, yin = 0.95-win, 0.91-hin 
ax_in = ax.inset_axes([xin, yin, win, hin])

ax = ax
ax.set_xscale('log')
ax.set_yscale('log')
c_base = 'gray'
c_dark = apm.lightenColor(c_base, 0.7)

sns.scatterplot(ax = ax, x=df_f[XCol].values, y=df_f[YCol].values/1000, 
                marker = 'o', s = 15, color = c_base, alpha = 0.33) #, label='All compressions')
Xfit, Yfit = np.log(df_f[XCol].values), np.log(df_f[YCol].values/1000)

wd=1/(np.std(Xfit)) # **2
we=1/(np.std(Yfit)) # **2

[a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
A, k = np.exp(b), a
pval = results.pvalue_pearson
[k_ciw, b_ciw] = results.params_ciw
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
text_pval = apm.pval2text(pval, n_digits = 3, space = True)
ax.plot(Xplot, Yplot, ls = '--', c = c_dark, lw = 2.0,
        label = r'$\bf{Fit\ y\ =\ A\cdot x^k}$' + \
                # f'\nA = {A:.1e}' + \
                f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + '\n' + text_pval)

# ax.legend().set_visible(False)
ax.legend(loc = 'lower left', handlelength = 1)
ax.set_title('All compressions', color = c_dark, weight = 'bold')
ax.set_ylabel('$E_{500}$ (kPa)')
ax.set_xlabel('$H_{500}$ (nm)')
ax.grid(visible=True, which='major', axis='both')
ax.set_xlim([50, 1100])
ax.set_ylim([0.5, 200])
# ax.tick_params(axis='both', direction='in', which='both')

hM, hL, hH = ufun.getLogNDistributionDescriptors(df_f[XCol].values)
EM, EL, EH = ufun.getLogNDistributionDescriptors(df_f[YCol].values/1000)
print(f'By compression, N = {len(df_f):.0f}')
print(f'For {XCol} vs {YCol}')
print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')
    
    
#### Inset
ax = ax_in
ax.set_xscale('log')
ax.set_yscale('log')
c_base = apm.cL_Set2[0]
c_dark = apm.lightenColor(c_base, 0.7)

sns.scatterplot(ax = ax, x=df_plot[XCol].values, y=df_plot[YCol+'_wAvg'].values/1000, 
                marker = 'o', s = 15, color = c_base, alpha = 0.3)
Xfit, Yfit = np.log(df_plot[XCol].values), np.log(df_plot[YCol+'_wAvg'].values/1000)

wd=1/(np.std(Xfit)) # **2
we=1/(np.std(Yfit)) # **2

[a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
A, k = np.exp(b), a
pval = results.pvalue_pearson
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
text_pval = apm.pval2text(pval, n_digits = 3, space = True)
ax.plot(Xplot, Yplot, ls = '--', c = c_dark, lw = 1.5,
        label =  text_pval)
        # label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + \
        #         f'\n$R^2$  = {R2:.2f}' + '\n' + text_pval)

ax.legend(fontsize = 6, handlelength = 1)
ax.set_title('Average per cell', color = c_dark, weight = 'bold')
ax.grid()
# ax.set_ylabel('$E_{500}$ (kPa)')
# ax.set_xlabel('$H_0$ (nm)')
ax.set_xlim([80, 1100])
ax.set_ylim([0.5, 200])
ax.tick_params(axis='both', direction='in', which='both')
# ax.set_xticklabels(fontsize=9)
# ax.set_yticklabels(fontsize=9)

hM, hL, hH = ufun.getLogNDistributionDescriptors(df_plot[XCol].values)
EM, EL, EH = ufun.getLogNDistributionDescriptors(df_plot[YCol+'_wAvg'].values/1000)
print(f'By cells, N = {len(df_plot):.0f}')
print(f'For {XCol} vs {YCol}')
print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')

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
    
    
    
    
    
    
    
# %% Main Figure 4

# %%% ~~~ Investigate the median num of comps for dates

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
            wd=1/(np.std(Xfit)) # **2
            we=1/(np.std(Yfit)) # **2
            mydata = odr.Data(Xfit, Yfit, wd=wd, we=we)
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
        print(cid, thickCV, pval)
        
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
                        crit_pvalFit = 0.2,
                        crit_thickCV = 0.04,
                        activeCrits = ['NcompsMin', 'pvalFit', 'thickCV'], modeFit = 'ODR')

plotEh_perCell(df, XCol = 'bestH0', YCol = 'E_f_<_500',
                    crit_NcompsMin = 10,
                    crit_pvalFit = 0.2,
                    crit_thickCV = 0.04,
                    activeCrits = ['NcompsMin', 'pvalFit', 'thickCV'],
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
    


# %% --------

# %% Supp Figure 1

# %%% Normal Distribution

apm.setGraphicOptions(mode = 'print', 
                      palette = 'Set2', 
                      colorList = apm.cL_Set21)

# Save
SAVE = True
figSubDir = 'S1'
name = 'S1_LogNormalDist'

#### Part 1 - Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']

HCol = 'H0_f_<_500'
ECol = 'E_f_<_500'
suffix = '_f_<_500'
parameter = HCol

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x not in excluded_dates)),
           (df['normal field'] == 5),
           (df[HCol] < 1000),
           (df[ECol] <= 2e4),
           (df['valid' + suffix] == True), 
           ]

df_f = apm.filterDf(df, Filters)
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)

# Group By for H0
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0', 'ctFieldFluctuAmpli'], aggFun = 'mean')

# Group By for E<400
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = ECol, weightCol = 'ciw' + ECol, weight_method = 'ciw^2')
df_fgw2[ECol + '_wAvg'] /= 1000

#### Init fig
fig, axes = plt.subplots(2, 1, figsize=(5.5/cm_in, 11/cm_in))
color = apm.styleDict['dmso']['color']

#### Part 2

#### 03 - Normality test H0

ax = axes[0]

data_lin = df_fg[df_fg['drug'] == 'dmso']['bestH0'].values
data_log = np.log(df_fg[df_fg['drug'] == 'dmso']['bestH0'].values)

data=data_lin
shap_stat, shap_pval = shapiro(data)
sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = apm.cL_Set21[5], markeredgecolor = 'None', markersize=6)
ax.plot([], [], label=f'N: {shap_pval:.2f}', ls='', marker='o', 
        markerfacecolor = apm.cL_Set21[5], markeredgecolor = 'None', markersize=4)
# ax.grid()
# ax.set_title('Q-Q plots for $H_0$')


#### 04 - normality test E

ax = axes[1]

data_lin = df_fgw2[df_fgw2['drug'] == 'dmso'][ECol + '_wAvg'].values
data_log = np.log(df_fgw2[df_fgw2['drug'] == 'dmso'][ECol + '_wAvg'].values)

data=data_lin
shap_stat, shap_pval = shapiro(data)
sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = apm.cL_Set21[5], markeredgecolor = 'None', markersize=6)
ax.plot([], [], label=f'N: {shap_pval:.2f}', ls='', marker='o', 
        markerfacecolor = apm.cL_Set21[5], markeredgecolor = 'None', markersize=4)
# ax.grid()
# ax.set_title('Q-Q plots for $E_{500}$')

#### Part 3

#### 05 - Log-Normality test H0

ax = axes[0]

data_lin = df_fg[df_fg['drug'] == 'dmso']['bestH0'].values
data_log = np.log(df_fg[df_fg['drug'] == 'dmso']['bestH0'].values)

data=data_log
shap_stat, shap_pval = shapiro(data)
ax.axline((0, 0), slope=1, color="k", linestyle='--', linewidth=1, zorder=6)
ax.set_aspect('equal')
ax.set_xlim([-3.5,3.5])
ax.set_ylim([-3.5,3.5])
sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = apm.cL_Set21[0], markeredgecolor = 'None', markersize=6)
ax.plot([], [], label=f'LogN: {shap_pval:.2f}', ls='', marker='o', 
        markerfacecolor = apm.cL_Set21[0], markeredgecolor = 'None', markersize=4)
ax.legend(fontsize=6, title_fontsize=6, title = 'Shapiro–Wilk\np-values', loc='lower right')
ax.grid()
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax.text(-3, 2.5, '$H_0$ Q-Q plot', va='center', ha='left', 
        fontsize=10.0, backgroundcolor='w')

#### 06 - Log-normality test E

ax = axes[1]

data_lin = df_fgw2[df_fgw2['drug'] == 'dmso'][ECol + '_wAvg'].values
data_log = np.log(df_fgw2[df_fgw2['drug'] == 'dmso'][ECol + '_wAvg'].values)

data=data_log
shap_stat, shap_pval = shapiro(data)
ax.axline((0, 0), slope=1, color="k", linestyle='--', linewidth=1, zorder=6)
ax.set_aspect('equal')
ax.set_xlim([-3.5,3.5])
ax.set_ylim([-3.5,3.5])
sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = apm.cL_Set21[0], markeredgecolor = 'None', markersize=6)
ax.plot([], [], label=f'LogN: {shap_pval:.2f}', ls='', marker='o', 
        markerfacecolor = apm.cL_Set21[0], markeredgecolor = 'None', markersize=4)
ax.legend(fontsize=6, title_fontsize=6, title = 'Shapiro–Wilk\np-values', loc='lower right')
ax.grid()
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax.text(-3, 2.5, '$E_{500}$ Q-Q plot', va='center', ha='left', 
        fontsize=10.0, backgroundcolor='w')

# Show
plt.tight_layout()
plt.show()

# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%% Choice of F < 500

# %%%% 1. Get a data set of F-h to fit

plot_stressCenters = [ii for ii in range(100, 4000, 50)]
stressHalfWidths = [50, 75, 100]

fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['pts_15',
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_15',
                'doChadwickFit' : True,
                'ChadwickFitMethods' : ['Full'],
                }


# =============================================================================
# # task = '24-03-13_M1_P1_C15'
# # task = '23-03-17_M4_P1_C15 & 23-03-17_M4_P1_C14 & 23-03-17_M4_P1_C8 & 24-07-04_M4_P1_C16' # 23-03-16_M1_P1_C2 & 
# # task = '24-07-04_M4_P1_C15'
# # task = '24-07-04_M6'
# # task = '23-03-17_M4'
# # task = '24-07-04_M6_P1_C11'
# # task = '23-03-09_M4_P1_C2 & 23-03-09_M4_P1_C5 & 23-03-09_M4_P1_C12'
# # task += ' & 23-03-09_M4_P1_C4 & 23-03-09_M4_P1_C8 & 23-03-09_M4_P1_C9'
# # task += ' & 23-03-09_M4_P1_C15 & 23-03-09_M4_P1_C14'
# # task = '23-03-17_M4_P1_C3 & 23-03-17_M4_P1_C9 & 23-03-17_M4_P1_C11'
# # task = '23-03-16_M1_P1_C4'
# =============================================================================

phyTask = '23-02-16_M1 & 23-02-23_M1 & 23-02-23_M3 & 23-03-08_M3 & 23-03-16_M1 & 23-03-17_M4' # Dmso & none 1/4
phyTask += ' & 23-04-20_M1 & 23-04-20_M4 & 23-04-20_M5 & 23-04-26_M2 & 23-04-28_M1 & 23-07-17_M3' # Dmso & none 2/4
phyTask += ' & 23-07-17_M4 & 23-07-17_M6 & 23-07-20_M2 & 23-09-06_M3 & 23-09-11_M1 & 23-09-19_M1 & 23-11-26_M2 & 23-12-03_M1' # Dmso & none 3/4
phyTask += ' & 24-07-04_M2 & 24-07-04_M6' # Dmso & none 4/4
phyTask += ' & 23-03-09_M4' # Pattern sizes JV
phyTask += ' & 24-12-11'

Id_comps, Comps = takaP.getCompressions(task = phyTask,
                                        fitSettings = fitSettings)


# %%%% 2. Compute several goodness of fit metrics and plot for F < Fmax

def fitChadwick_hf(h, f, D, err_chi2):
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

    try:
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
        # params[0], ses[0] = params[0]*1e6, ses[0]*1e6 # Convert E & seE to Pa
        
    except:
        error = True
        params = np.ones(2) * np.nan
        ses = np.ones(2) * np.nan
        
    if not error:
        E, H0 = params
        hPredict = inversedChadwickModel(f, E, H0)
        x, y, yPredict = f, h, hPredict

        seE, seH0 = ses

        alpha = 0.975
        dof = len(y)-len(params)
        q = st.t.ppf(alpha, dof) # Student coefficient
        R2 = ufun.get_R2(y, yPredict)
        Chi2 = ufun.get_Chi2(y, yPredict, dof, err_chi2)        

        ciwE = q*seE
        ciwH0 = q*seH0
        
        nbPts = len(y)
        
    else:
        R2 = 0
        Chi2 = 0
        
    res = (error, R2, Chi2)
        
    return(res)



def getCurvature(h, f, D):
    R = D/2
    Npts = len(h)
    error = False
    
    def chadwickModel(h, E, H0):
        f = (np.pi*E*R*((H0-h)**2))/(3*H0)
        return(f)

    def inversedChadwickModel(f, E, H0, k):
        h = H0 - ((3*H0*f)/(np.pi*E*R))**k
        return(h)

    # try:
    # some initial parameter values - must be within bounds
    initH0 = max(h) # H0 ~ h_max
    initE = (3*max(h)*max(f))/(np.pi*(R)*(max(h)-min(h))**2) # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
    initk = 0.5
    
    initialParameters = [initE, initH0, initk]
    
    # bounds on parameters - initial parameters must be within these
    lowerBounds = (0, 0, 0.33)
    upperBounds = (np.inf, 2000, 1)
    parameterBounds = [lowerBounds, upperBounds]
    
    
    # params = [E, H0] ; ses = [seE, seH0]
    params, covM = curve_fit(inversedChadwickModel, f, h, 
                             p0=initialParameters, 
                             bounds = parameterBounds,
                             maxfev = 2000)
    ses = np.array([covM[0,0]**0.5, covM[1,1]**0.5,  covM[2,2]**0.5])
    # params[0], ses[0] = params[0]*1e6, ses[0]*1e6 # Convert E & seE to Pa
        
    # except:
    #     error = True
    #     params = np.ones(3) * np.nan
    #     ses = np.ones(3) * np.nan
        
    if not error:
        E, H0, k = params
        hPredict = inversedChadwickModel(f, E, H0, k)
        x, y, yPredict = f, h, hPredict

        seE, seH0, sek = ses

        alpha = 0.975
        dof = len(y)-len(params)
        q = st.t.ppf(alpha, dof) # Student coefficient
        R2 = ufun.get_R2(y, yPredict)
        Chi2 = ufun.get_Chi2(y, yPredict, dof, err_chi2)        

        ciwE = q*seE
        ciwH0 = q*seH0
        
        nbPts = len(y)
        
    else:
        R2 = 0
        Chi2 = 0
        
    res = (E, H0, k)
        
    return(res)



list_Fmax = np.arange(150, 1100, 50)
# list_D = Id_comps[:][2]
all_Chi2 = []
all_R2   = []
err_chi2 = 8
# err_chi2_test1 = 5
# err_chi2_test2 = 20


# list_E  = []
# list_H0 = []
# list_k  = []
# for Fmax in list_Fmax:
#     for k in range(20,40): # len(Comps)
#         # print(k)
#         D = Id_comps[k][2]
#         h, f = Comps[k]
#         index = (f < Fmax)
#         h_fit, f_fit = h[index], f[index]
#         (E, H0, k) = getCurvature(h_fit, f_fit, D)
#         list_E.append(E)
#         list_H0.append(H0)
#         list_k.append(k)


for Fmax in list_Fmax:
    list_Chi2, list_R2 = [], []
    for k in range(len(Comps)): # len(Comps)
        D = Id_comps[k][2]
        h, f = Comps[k]
        index = (f < Fmax)
        h_fit, f_fit = h[index], f[index]
        res = fitChadwick_hf(h_fit, f_fit, D, err_chi2)
        error, r2, chi2 = res
        
        if (not error) and (chi2 > 0) and (r2 < 1):
            list_Chi2.append(chi2)
            list_R2.append(r2)
    
    all_Chi2.append(list_Chi2)
    all_R2.append(list_R2)
    
    
# %%%% 3. Compute statistics
    
avg_R2 = [np.mean(list_R2) for list_R2 in all_R2]
std_R2 = [np.std(list_R2) for list_R2 in all_R2]
median_R2 = [np.median(list_R2) for list_R2 in all_R2]
D1_R2 = [np.percentile(list_R2, 25) for list_R2 in all_R2]
D9_R2 = [np.percentile(list_R2, 75) for list_R2 in all_R2]

avg_Chi2 = [np.mean(list_Chi2) for list_Chi2 in all_Chi2]
std_Chi2 = [np.std(list_Chi2) for list_Chi2 in all_Chi2]
median_Chi2 = [np.median(list_Chi2) for list_Chi2 in all_Chi2]
D1_Chi2 = [np.percentile(list_Chi2, 25) for list_Chi2 in all_Chi2]
D9_Chi2 = [np.percentile(list_Chi2, 75) for list_Chi2 in all_Chi2]


# %%%% 4. Plot the results

apm.setGraphicOptions(mode = 'print', 
                      palette = 'Set2', 
                      colorList = apm.cL_Set21)

# Save
SAVE = True
figSubDir = 'S1'
name = 'S1_Choice_500pN'

c1 = apm.cL_Set2[0]
c2 = apm.cL_Set2[1]

fig, ax = plt.subplots(1, 1, figsize=(6/cm_in, 6/cm_in))
ax2 = ax.twinx()
ax.plot(list_Fmax, median_R2, color=apm.lightenColor(c1, 0.9), lw=2)
ax.plot(list_Fmax, D1_R2, ls=':', color=apm.lightenColor(c1, 1.1), lw=1.25)
ax.plot(list_Fmax, D9_R2, ls='--', color=apm.lightenColor(c1, 1.1), lw=1.25)
ax2.plot(list_Fmax, median_Chi2, color=apm.lightenColor(c2, 0.9), lw=2)
ax2.plot(list_Fmax, D1_Chi2, ls=':', color=apm.lightenColor(c2, 1.1), lw=1.25)
ax2.plot(list_Fmax, D9_Chi2, ls='--', color=apm.lightenColor(c2, 1.1), lw=1.25)

ax.set_xlabel('Upper Bound of F')
ax.set_ylabel(r'$\bf{R^2}$', color=apm.lightenColor(c1, 0.75), weight='bold')
ax2.set_ylabel(r'$\bf{\chi^2}$', color=apm.lightenColor(c2, 0.75), weight='bold')

ax.axvline(500, color='gray', lw=1, ls='-.')
ax.set_xlim([0, 1100])
ax.set_ylim([0, 1.05])
ax2.set_ylim([0, 2.1])
plt.show()


# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')




# %%% Cell Identity

# dates = ['23-03-09',
#          '23-09-19',
#          '23-12-03',
#          '24-12-11',
#          ]

# %%%% ~~~ Normality Tests

#### Define

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO'] # 
substrate = '20um fibronectin discs'

dates = ['23-03-09',
         '23-09-19',
         # '23-12-03',
         '24-12-11',
         ]

df, condCol = apm.makeCompositeCol(df, cols=['drug'])
# figname = 'bestH0' + drugSuffix

HCol = 'H0_f_<_500'
ECol = 'E_f_<_500'
parameter = HCol

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df[HCol] < 1000),
           (df[ECol] <= 2e4),
           (df['valid_f_<_500'] == True), 
           (df['date'].apply(lambda x : x in dates)),
           ]

df_f = apm.filterDf(df, Filters)

# Order
co_order = []

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
df_fg[parameter + '_pop_spread'] = df_fg[parameter] / np.mean(df_fg[parameter].values)

# Merge
df_m = pd.merge(left=df_f, right=df_fg, on = 'cellID', how='left', suffixes = (None, '_grouped'))
df_m[parameter + '_indiv_spread'] = df_m[parameter] / df_m[parameter + '_grouped']

# Sort by cellID
df_f.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_fg.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_m.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)


#### Apply log

dfLOG_f = apm.filterDf(df, Filters)

dfLOG_f[parameter] = np.log10(dfLOG_f[parameter])
# df_f[parameter] = np.log(df_f[parameter])

# Group By
dfLOG_fg = apm.dataGroup(dfLOG_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
dfLOG_fg[parameter + '_pop_spread'] = dfLOG_fg[parameter] / np.mean(dfLOG_fg[parameter].values)

# Merge
dfLOG_m = pd.merge(left=dfLOG_f, right=dfLOG_fg, on = 'cellID', how='left', suffixes = (None, '_grouped'))
dfLOG_m[parameter + '_indiv_spread'] = dfLOG_m[parameter] / dfLOG_m[parameter + '_grouped']

# Sort by cellID
dfLOG_f.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)
dfLOG_fg.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)
dfLOG_m.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)


#### Normality tests

data_lin = df_fg[parameter].values
data_log = dfLOG_fg[parameter].values

fig_test, axes_test = plt.subplots(1, 2, figsize = (12, 5))

data=data_lin
ax = axes_test[0]
shap_stat, shap_pval = shapiro(data)
sm.qqplot(data, fit=True, line='45', ax=ax)
ax.plot([], [], label=f'Shapiro–Wilk p-value = {shap_pval:.2f}', ls='', marker='o')
ax.legend(fontsize=11)
ax.set_title(f'Normality test and\nQ-Q plot of parm: {parameter}', fontsize=11)

data=data_log
ax = axes_test[1]
shap_stat, shap_pval = shapiro(data)
sm.qqplot(data, fit=True, line='45', ax=ax)
ax.plot([], [], label=f'Shapiro–Wilk p-value = {shap_pval:.2f}', ls='', marker='o')
ax.legend(fontsize=11)
ax.set_title(f'Log-normality test and\nQ-Q plot of parm: {parameter}', fontsize=11)

fig_test.tight_layout()
plt.show()


# # %%%% One way ANOVA
# list_cellIDs = df_f['cellID'].unique()
# values_per_cell = [df_f.loc[df_f['cellID'] == ID, parameter].values for ID in list_cellIDs]

# f_oneway(*values_per_cell)


#### Compute CV by cell and by pop

list_cellIDs = dfLOG_f['cellID'].unique()
values_per_cell = [dfLOG_f.loc[dfLOG_f['cellID'] == ID, parameter].values for ID in list_cellIDs]
CV_per_cell = [np.std(cell_vals) / np.mean(cell_vals) for cell_vals in values_per_cell if len(cell_vals) >= 4]
CV_per_cell_avg = np.mean(CV_per_cell)
CV_population = np.std(dfLOG_fg[parameter]) / np.mean(dfLOG_fg[parameter])

print('CV of all cells: ', *[f'{cv:.4f}' for cv in CV_per_cell])
print(f'Mean CV per cell: {CV_per_cell_avg:.4f}')
print(f'CV of full pop: {CV_population:.4f}')
print(f'N cells = {len(list_cellIDs):.0f}')



# %%%% Thickness

apm.setGraphicOptions(mode = 'print', 
                      palette = 'Set2', 
                      colorList = apm.cL_Set21)

# Save
SAVE = True
figSubDir = 'S1'
name = 'S1_Identity_H500'

#### Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO'] # 
substrate = '20um fibronectin discs'

dates = ['23-03-09',
         '23-09-19',
         # '23-12-03',
         '24-12-11',
         ]

HCol = 'H0_f_<_500'
ECol = 'E_f_<_500'
suffix = '_f_<_500'
parameter = HCol

df, condCol = apm.makeCompositeCol(df, cols=['date'])
df['cellCode'] = df['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df[HCol] < 1000),
           (df[ECol] <= 2e4),
           (df['valid_f_<_500'] == True), 
           (df['date'].apply(lambda x : x in dates)),
           (df['cellCode'] != 'C1801'),
           ]
df_f = apm.filterDf(df, Filters)
df_f['cellNum'] = df_f['cellCode'].apply(lambda x : int(x[1:]))

# Order
co_order = dates

# Count
CountByCond, CountByCell =apm.makeCountDf(df_f, condCol)
Manipe = df_f['manipID'].values[0]
Ncells = CountByCond['cellCount'].values[0]
Ncomps = CountByCond['compCount'].values[0]

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol, 'cellCode', 'cellNum'], numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
df_fg[parameter + '_pop_spread'] = df_fg[parameter] / np.mean(df_fg[parameter].values)

# Merge
df_m = pd.merge(left=df_f, right=df_fg, on = 'cellID', how='left', suffixes = (None, '_grouped'))
df_m[parameter + '_indiv_spread'] = df_m[parameter] / df_m[parameter + '_grouped']

# Sort by cellID
df_f.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_fg.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_m.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)




#### Apply log

dfLOG_f = apm.filterDf(df, Filters)
dfLOG_f['cellNum'] = dfLOG_f['cellCode'].apply(lambda x : int(x[1:]))

dfLOG_f[parameter] = np.log10(dfLOG_f[parameter])
# df_f[parameter] = np.log(df_f[parameter])

# Group By
dfLOG_fg = apm.dataGroup(dfLOG_f, groupCol = 'cellID', idCols = [condCol, 'cellCode', 'cellNum'], 
                     numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
dfLOG_fg[parameter + '_pop_spread'] = dfLOG_fg[parameter] / np.mean(dfLOG_fg[parameter].values)

# Merge
dfLOG_m = pd.merge(left=dfLOG_f, right=dfLOG_fg, on = 'cellID', how='left', suffixes = (None, '_grouped'))
dfLOG_m[parameter + '_indiv_spread'] = dfLOG_m[parameter] / dfLOG_m[parameter + '_grouped']

# Sort by cellID
dfLOG_f.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)
dfLOG_fg.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)
dfLOG_m.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)


#### Compute CV
CV_per_cell_avg = []
CV_population = []
for co in df_f[condCol].unique():
    dfLOG_f_c  = dfLOG_f[dfLOG_f[condCol] == co]
    dfLOG_fg_c = dfLOG_fg[dfLOG_fg[condCol] == co]
    list_cellIDs = dfLOG_f['cellID'].unique()
    values_per_cell = [dfLOG_f_c.loc[dfLOG_f_c['cellID'] == ID, parameter].values for ID in list_cellIDs]
    CV_per_cell = [np.std(cell_vals) / np.mean(cell_vals) for cell_vals in values_per_cell if len(cell_vals) >= 4]
    CV_per_cell_avg.append(np.mean(CV_per_cell))
    CV_population.append(np.std(dfLOG_fg_c[parameter]) / np.mean(dfLOG_fg_c[parameter]))


# =============================================================================
# print('CV of all cells: ', *[f'{cv:.4f}' for cv in CV_per_cell])
# print(f'Mean CV per cell: {CV_per_cell_avg:.4f}')
# print(f'CV of full pop: {CV_population:.4f}')
# print(f'N cells = {len(list_cellIDs):.0f}')
# =============================================================================


#### Start plot
fig = plt.figure(figsize=(14/cm_in, 7/cm_in))
spec = fig.add_gridspec(1, 2)


#### Plot 1
ax = fig.add_subplot(spec[0])
ax.set_ylim([0.7, 1.3])

df_m['cellCode'] = df_m['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])

# sns.swarmplot(ax = ax, data = dfLOG_m, x = 'date', y = parameter + '_indiv_spread', # hue = 'cellCode', 
#               s=2, edgecolor='w', linewidth=0) # 
sns.violinplot(ax = ax, data = dfLOG_m, x = 'date', y = parameter + '_indiv_spread', 
               hue = 'date', inner="quart",
               edgecolor='w', order = co_order, linewidth=1.1) # s=2, 
for xt in [0, 1, 2]:
    ax.text(xt, 1.265, f'CV = {CV_per_cell_avg[xt]*100:.1f} %', fontsize=7,
            horizontalalignment = 'center', verticalalignment = 'center')

ax.axhline(1, color='k', lw=0.5, zorder=2)
ax.grid(visible=True, which='major', axis='y')
apm.renameAxes(ax, apm.renameDict, format_xticks = True, rotation = 0)
ax.set_ylabel('$Log(H_0)$ distribution\naround cell average')
ax.set_title('Intra-cell Variability')
ax.set_xlabel('')
ax.set_xticklabels(['Expt 1', 'Expt 2', 'Expt 3'])


#### Plot 2
ax = fig.add_subplot(spec[1])
ax.set_ylim([0.7, 1.3])

# sns.swarmplot(ax = ax, data = dfLOG_fg, 
#                x = 'date', y = parameter + '_pop_spread', hue = 'cellID', 
#               edgecolor='w', linewidth=0, s=8) #, legend=False)
sns.violinplot(ax = ax, data = dfLOG_fg, 
               x = 'date', y = parameter + '_pop_spread', hue = 'date', inner="quart", # hue = 'cellID', 
               order = co_order, edgecolor='w', linewidth=1.1, zorder=2) #, s=8) #, legend=False)
for xt in [0, 1, 2]:
    ax.text(xt, 1.27, f'CV = {CV_population[xt]*100:.1f} %', fontsize=7,
            horizontalalignment = 'center', verticalalignment = 'center')
# LegendMark = mlines.Line2D([], [], color='gray', ls='', marker='o', 
#                            markersize=0, markeredgecolor='w', markeredgewidth=0,
#                            label=f'Inter-cell CV = {CV_population*100:.1f} %')
# ax.legend(handles=[LegendMark])
# ax.plot([], [], marker = 'o', ls='', c='gray', markersize=4, markeredgecolor='w', markeredgewidth=0.75, 
#         label = f'Population CV = {CV_population*100:.1f}')
ax.grid(visible=True, which='major', axis='y', zorder=1)
ax.axhline(1, color='k', lw=0.5, zorder=3)
apm.renameAxes(ax, apm.renameDict, format_xticks = True, rotation = 0)
ax.set_ylabel('$Log(H_0)$ distribution\naround population average')
ax.set_title('Inter-cell Variability')
ax.set_xlabel('')
ax.set_xticklabels(['Expt 1', 'Expt 2', 'Expt 3'])

# print(df_fg.cellID.unique())


#### Finalize
# Count
# CountByCond, CountByCell =apm.makeCountDf(df_f, condCol)

# Show
# fig.suptitle('Thickness')
fig.tight_layout()
plt.show()

# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')





# %%%% Stiffness

apm.setGraphicOptions(mode = 'print', 
                      palette = 'Set2', 
                      colorList = apm.cL_Set21)

# Save
SAVE = True
figSubDir = 'S1'
name = 'S1_Identity_E500'

#### Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO'] # 
substrate = '20um fibronectin discs'

HCol = 'H0_f_<_500'
ECol = 'E_f_<_500'
suffix = '_f_<_500'
parameter = ECol

df, condCol = apm.makeCompositeCol(df, cols=['date'])
df['cellCode'] = df['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df[HCol] < 1000),
           (df[ECol] <= 2e4),
           (df['valid_f_<_500'] == True), 
           (df['date'].apply(lambda x : x in dates)),
           (df['cellCode'] != 'C1801'),
           ]
df_f = apm.filterDf(df, Filters)
df_f['cellNum'] = df_f['cellCode'].apply(lambda x : int(x[1:]))

df_f[ECol] = df_f[ECol]/1000


# Order
co_order = dates


# Count
CountByCond, CountByCell =apm.makeCountDf(df_f, condCol)
Manipe = df_f['manipID'].values[0]
Ncells = CountByCond['cellCount'].values[0]
Ncomps = CountByCond['compCount'].values[0]

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol, 'cellCode', 'cellNum'], numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
df_fg[parameter + '_pop_spread'] = df_fg[parameter] / np.mean(df_fg[parameter].values)

# Merge
df_m = pd.merge(left=df_f, right=df_fg, on = 'cellID', how='left', suffixes = (None, '_grouped'))
df_m[parameter + '_indiv_spread'] = df_m[parameter] / df_m[parameter + '_grouped']

# Sort by cellID
df_f.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_fg.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_m.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)




#### Apply log

dfLOG_f = apm.filterDf(df, Filters)
dfLOG_f['cellNum'] = dfLOG_f['cellCode'].apply(lambda x : int(x[1:]))

dfLOG_f[parameter] = np.log10(dfLOG_f[parameter])
# df_f[parameter] = np.log(df_f[parameter])

# Group By
dfLOG_fg = apm.dataGroup(dfLOG_f, groupCol = 'cellID', idCols = [condCol, 'cellCode', 'cellNum'], 
                     numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
dfLOG_fg[parameter + '_pop_spread'] = dfLOG_fg[parameter] / np.mean(dfLOG_fg[parameter].values)

# Merge
dfLOG_m = pd.merge(left=dfLOG_f, right=dfLOG_fg, on = 'cellID', how='left', suffixes = (None, '_grouped'))
dfLOG_m[parameter + '_indiv_spread'] = dfLOG_m[parameter] / dfLOG_m[parameter + '_grouped']

# Sort by cellID
dfLOG_f.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)
dfLOG_fg.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)
dfLOG_m.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)

# df_m['cellCode'] = df_m['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])
# dfLOG_m['cellCode'] = dfLOG_m['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])





#### Compute CV
CV_per_cell_avg = []
CV_population = []
for co in df_f[condCol].unique():
    dfLOG_f_c  = dfLOG_f[dfLOG_f[condCol] == co]
    dfLOG_fg_c = dfLOG_fg[dfLOG_fg[condCol] == co]
    list_cellIDs = dfLOG_f['cellID'].unique()
    values_per_cell = [dfLOG_f_c.loc[dfLOG_f_c['cellID'] == ID, parameter].values for ID in list_cellIDs]
    CV_per_cell = [np.std(cell_vals) / np.mean(cell_vals) for cell_vals in values_per_cell if len(cell_vals) >= 4]
    CV_per_cell_avg.append(np.mean(CV_per_cell))
    CV_population.append(np.std(dfLOG_fg_c[parameter]) / np.mean(dfLOG_fg_c[parameter]))


# =============================================================================
# print('CV of all cells: ', *[f'{cv:.4f}' for cv in CV_per_cell])
# print(f'Mean CV per cell: {CV_per_cell_avg:.4f}')
# print(f'CV of full pop: {CV_population:.4f}')
# print(f'N cells = {len(list_cellIDs):.0f}')
# =============================================================================

#### Start plot
fig = plt.figure(figsize=(14/cm_in, 7/cm_in))
spec = fig.add_gridspec(1, 2)


#### Plot 1
ax = fig.add_subplot(spec[0])
ax.set_ylim([0.7, 1.3])

df_m['cellCode'] = df_m['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])

# sns.swarmplot(ax = ax, data = dfLOG_m, x = 'date', y = parameter + '_indiv_spread', # hue = 'cellCode', 
#               s=2, edgecolor='w', linewidth=0) # 
sns.violinplot(ax = ax, data = dfLOG_m, x = 'date', y = parameter + '_indiv_spread', 
               hue = 'date', inner="quart",
               edgecolor='w', order = co_order, linewidth=1.1) # s=2, 
for xt in [0, 1, 2]:
    ax.text(xt, 1.265, f'CV = {CV_per_cell_avg[xt]*100:.1f} %', fontsize=7, 
            horizontalalignment = 'center', verticalalignment = 'center')

ax.axhline(1, color='k', lw=0.5, zorder=2)
ax.grid(visible=True, which='major', axis='y')
apm.renameAxes(ax, apm.renameDict, format_xticks = True, rotation = 0)
ax.set_ylabel('$Log(E_{500})$ distribution\naround cell average')
ax.set_title('Intra-cell Variability')
ax.set_xlabel('')
ax.set_xticklabels(['Expt 1', 'Expt 2', 'Expt 3'])

# print(df_m.cellID.unique())


#### Plot 2
ax = fig.add_subplot(spec[1])
ax.set_ylim([0.7, 1.3])

# sns.swarmplot(ax = ax, data = dfLOG_fg, 
#                x = 'date', y = parameter + '_pop_spread', hue = 'cellID', 
#               edgecolor='w', linewidth=0, s=8) #, legend=False)
sns.violinplot(ax = ax, data = dfLOG_fg, 
               x = 'date', y = parameter + '_pop_spread', hue = 'date', inner="quart", # hue = 'cellID', 
               order = co_order, edgecolor='w', linewidth=1.1, zorder=2) #, s=8) #, legend=False)
for xt in [0, 1, 2]:
    ax.text(xt, 1.27, f'CV = {CV_population[xt]*100:.1f} %', fontsize=7,
            horizontalalignment = 'center', verticalalignment = 'center')

ax.grid(visible=True, which='major', axis='y', zorder=1)
ax.axhline(1, color='k', lw=0.5, zorder=3)
apm.renameAxes(ax, apm.renameDict, format_xticks = True, rotation = 0)
ax.set_ylabel('$Log(E_{500})$ distribution\naround population average')
ax.set_title('Inter-cell Variability')
ax.set_xlabel('')
ax.set_xticklabels(['Expt 1', 'Expt 2', 'Expt 3'])



#### Finalize
# Count
# CountByCond, CountByCell =apm.makeCountDf(df_f, condCol)

# Show
# fig.suptitle('Thickness')
fig.tight_layout()
plt.show()

# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')



# %%% Cortex Plasticity
# LOG Successive compressions

apm.setGraphicOptions(mode = 'print', 
                      palette = 'Set2', 
                      colorList = apm.cL_Set21)

# Save
SAVE = True
figSubDir = 'S1'
name = 'S1_Plasticity_E500-H500'


#### Define

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO'] # 
substrate = '20um fibronectin discs'

dates = ['23-03-09',
         '23-09-19',
         # '23-12-03',
         # '23-03-16', 
         # '23-03-17',
         '24-12-11',
         ]

df, condCol = apm.makeCompositeCol(df, cols=['drug'])
# figname = 'bestH0' + drugSuffix

HCol = 'H0_f_<_500'
ECol = 'E_f_<_500'
parameter = HCol

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df[HCol] < 1000),
           (df[ECol] <= 2e4),
           (df['compNum'] <= 10),
           (df['valid_f_<_500'] == True), 
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['none'])),
           # (df['cellName'].apply(lambda x : not '-2' in x)),
           ]

df_f = apm.filterDf(df, Filters)

# New data
# ManipTime
df_f['ManipTime'] = df_f['compAbsStartTime']
cellID_list = df_f['cellID'].unique()
manipID_list = df_f['manipID'].unique()
for mid in manipID_list:
    index_mid = df_f[df_f['manipID']==mid].index
    firstManipTime = np.min(df_f[df_f['manipID']==mid]['compAbsStartTime'].values)
    df_f.loc[index_mid, 'ManipTime'] -= firstManipTime
df_f['ManipTime'] /= 60

# relative H0
df_f['relative H0'] = df_f[HCol]
for cid in cellID_list:
    index_cid = df_f[df_f['cellID']==cid].index
    firstH0 = df_f[df_f['cellID']==cid][HCol].values[0]
    df_f.loc[index_cid, 'relative H0'] = np.log(df_f['relative H0'])/np.log(firstH0)
    # meanlogH0 = np.mean(np.log(df_f[df_f['cellID']==cid]['bestH0'].values))
    # df_f.loc[index_cid, 'relative H0'] = np.log(df_f['relative H0'])/meanlogH0

# relative E
df_f['relative E'] = df_f[ECol]
for cid in cellID_list:
    index_cid = df_f[df_f['cellID']==cid].index
    firstE = df_f[df_f['cellID']==cid][ECol].values[0]
    df_f.loc[index_cid, 'relative E'] = np.log(df_f['relative E'])/np.log(firstE)
    # meanlogE = np.mean(np.log(df_f[df_f['cellID']==cid]['E_f_<_400'].values))
    # df_f.loc[index_cid, 'relative E'] = np.log(df_f['relative E'])/meanlogE
    
# group
# df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], numCols = ['bestH0'], aggFun = 'mean')
# df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], 
#                                       valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
# df_fgw2['E_f_<_400_wAvg'] /= 1000

# df_fg = df_fg.drop_duplicates(subset='ManipTime')
# df_fgw2 = df_fgw2.drop_duplicates(subset='ManipTime')

Nmanips = len(manipID_list)
fig, axes = plt.subplots(2, Nmanips+1, figsize=(17/cm_in, 9/cm_in), sharex='col', sharey='row')

titles = ['Experiment ' + str(i+1) for i in range(Nmanips)]


for i in range(Nmanips):
    mid = manipID_list[i]
    df_f_mid = df_f[df_f['manipID'] == mid]
    cellID_list = df_f_mid['cellID'].unique()
    Nc = len(cellID_list)

    axcol = axes[:, i]
    P = sns.color_palette("husl", Nc)
    cL = matplotlib.colors.ListedColormap(P, 'my_cmap').colors
    
    #### Plot H0
    ax = axcol[0]
    ax.set_title(titles[i])
    # ax.set_title(mid + ' - $N_{cells}$ = ' + f'{len(cellID_list)}')
    # sns.scatterplot(ax=ax, data=df_f_mid, x='compNum', y='relative H0', hue='cellID', alpha=0.8)
    for j, cid in enumerate(cellID_list):
        df_f_cid = df_f_mid[df_f_mid['cellID']==cid]
        c = cL[j]
        X = df_f_cid['compNum'].values
        Y = df_f_cid['relative H0'].values
        # ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
        ax.axhline(1, color='k', ls='-', lw=0.75, zorder=2)
        ax.plot(X, Y, ls='-', c=c, lw=1.0, alpha=0.75, zorder=3, label='',
                marker = '.', markersize = 6, markerfacecolor = c, markeredgecolor = 'dimgray', markeredgewidth = 0.75)
        
    # ax.set_xlabel('Compression #')
    if i == 0:
        ax.set_ylabel('relative $log(H_0)$')
    tickloc = matplotlib.ticker.MultipleLocator(1)
    ax.xaxis.set_major_locator(tickloc)
    ax.grid(visible=True, which='major', axis='y')
    
    #### Plot E
    ax = axcol[1]
    # sns.scatterplot(ax=ax, data=df_f_mid, x='compNum', y='relative E', hue='cellID', alpha=0.8)
    for j, cid in enumerate(cellID_list):
        df_f_cid = df_f_mid[df_f_mid['cellID']==cid]
        c = cL[j]
        X = df_f_cid['compNum'].values
        Y = df_f_cid['relative E'].values
        # ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
        ax.axhline(1, color='k', ls='-', lw=0.75, zorder=2)
        ax.plot(X, Y, ls='-', c=c, lw=1.0, alpha=0.75, zorder=3, label='',
                marker = '.', markersize = 6, markerfacecolor = c, markeredgecolor = 'dimgray', markeredgewidth = 0.75)

    ax.set_xlabel('Compression #')
    if i == 0:
        ax.set_ylabel('relative $log(E_{500})$')
    tickloc = matplotlib.ticker.MultipleLocator(1)
    ax.xaxis.set_major_locator(tickloc)
    ax.grid(visible=True, which='major', axis='y')
    
    # for ax in axcol:
        # ax.legend().set_visible(False)
        

        
axcol = axes[:, -1]

ax = axcol[0]
ax.set_title('Average')
df_f_g = df_f[['compNum', 'relative H0']].groupby('compNum').agg(['mean', 'std'])
Xavg = df_f_g.index.values
Yavg = df_f_g['relative H0', 'mean'].values
Yerr = df_f_g['relative H0', 'std'].values
ax.errorbar(Xavg, Yavg, Yerr, ls='-', c='dimgray', lw=2, alpha=0.8, zorder=4, label='Average',
        marker = 'o', markersize = 6, markerfacecolor = 'w', markeredgecolor = 'dimgray', markeredgewidth = 2,
        elinewidth = 1.0, capsize=3, capthick=1, )

tickloc = matplotlib.ticker.MultipleLocator(1)
ax.xaxis.set_major_locator(tickloc)
ax.grid(visible=True, which='major', axis='y')


ax = axcol[1]
ax.set_xlabel('Compression #')
df_f_g = df_f[['compNum', 'relative E']].groupby('compNum').agg(['mean', 'std'])
Xavg = df_f_g.index.values
Yavg = df_f_g['relative E', 'mean'].values
Yerr = df_f_g['relative E', 'std'].values
ax.errorbar(Xavg, Yavg, Yerr, ls='-', c='dimgray', lw=2, alpha=0.8, zorder=4, label='Average',
        marker = 'o', markersize = 6, markerfacecolor = 'w', markeredgecolor = 'dimgray', markeredgewidth = 2,
        elinewidth = 1.0, capsize=3, capthick=1, )

tickloc = matplotlib.ticker.MultipleLocator(1)
ax.xaxis.set_major_locator(tickloc)
ax.grid(visible=True, which='major', axis='y')
        

# Show

fig.tight_layout()
plt.show()

# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')




# %%% Beads In-In vs Out-Out


# Save
SAVE = True
figSubDir = 'S1'
name = 'S1_InIn_OutOut'


# srcDirOut = "D:/MagneticPincherData/Raw/Control_InIn_OutOut"
srcDirOut = "C:/Users/josep/Documents/MagneticPincherData/Raw/Control_InIn_OutOut"
srcDirIn = srcDirOut
# srcDirIn  = "D:/MagneticPincherData/Raw/Control_InIn"
dirIn = os.path.join(srcDirIn, 'Timeseries_IN')
dirOut = os.path.join(srcDirOut, 'Timeseries_OUT')
dict_tsdf_in =  dict([(ufun.findInfosInFileName(f, 'cellID'), 
                       pd.read_csv(os.path.join(dirIn,  f), sep=';')) \
                      for f in os.listdir(dirIn)  if f.endswith('.csv')])
dict_tsdf_out = dict([(ufun.findInfosInFileName(f, 'cellID'), 
                       pd.read_csv(os.path.join(dirOut, f), sep=';')) \
                      for f in os.listdir(dirOut) if f.endswith('.csv')])
custom_cycler = (cycler(color=apm.cL_Set21))


Din = 4.463 # 4.493
Dout = 4.506

# ufun.findInfosInFileName(f, infoType)

for df in dict_tsdf_in.values():
    df['h'] = (df['D3']-Din)*1000
for df in dict_tsdf_out.values():
    df['h'] = (df['D3']-Dout)*1000

fig, axes = plt.subplots(1, 2, figsize=(11/cm_in, 6/cm_in), sharey=True)

count = 0
ax = axes[0]
ax.set_prop_cycle(custom_cycler)
for cid in dict_tsdf_out.keys():
    tsdf = dict_tsdf_out[cid]
    tsdf0 = tsdf[tsdf['idxAnalysis']==0]
    group = tsdf0[['idxLoop', 'T', 'h']].groupby('idxLoop')
    df = group.agg({'T':'mean', 'h':'mean'})
    # ax.plot(df.loc[df['idxAnalysis']==0, 'T'], df.loc[df['idxAnalysis']==0, 'h'], 
    #         ls='', marker = 'o', mec='w', mew=0.5, label=cid)
    if max(df['h']) > 80:
        continue
    else:
        print(cid, df['h'])
        count += 1
        ax.plot(df['T'].values, df['h'].values, ls='-', lw=1,
                marker = 'o', mec='w', mew=0.5, markersize = 8, zorder=6, # label=cid
                )
ax.plot([], [], ls='-', lw=1, c = 'gray',
        marker = 'o', mec='w', mew=0.5, markersize = 8, zorder=6,
        label=f'N = {(count):.0f}')
ax.axhline(0, ls='-', c='k', lw=1.5)
ax.set_title('Pair of beads outside')
ax.set_xlim([0, 100])
ax.set_ylim([-100, 100])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Measured\nthickness (nm)')
ax.grid(which='major', axis='both')
ax.legend(fontsize=8)


count = 0
ax = axes[1]
ax.set_prop_cycle(custom_cycler)
for cid in dict_tsdf_in.keys():
    tsdf = dict_tsdf_in[cid]
    tsdf0 = tsdf[tsdf['idxAnalysis']==0]
    tsdf0 = tsdf0[tsdf0['idxLoop']<=5]
    group = tsdf0[['idxLoop', 'T', 'h']].groupby('idxLoop')
    df = group.agg({'T':'mean', 'h':'median'})
    # ax.plot(df.loc[df['idxAnalysis']==0, 'T'], df.loc[df['idxAnalysis']==0, 'h'], 
    #         ls='', marker = 'o', mec='w', mew=0.5, label=cid)
    if max(df['h']) > 80:
        continue
    else:
        count += 1
        ax.plot(df['T'], df['h'], ls='-', lw=1,
                marker = 'o', mec='w', mew=0.5, markersize = 8, zorder=6, # label=cid
                )
    # ax.plot(df['T'], df['h'], ls='-', lw = 1, c='gray')
ax.plot([], [], ls='-', lw=1, c = 'gray',
        marker = 'o', mec='w', mew=0.5, markersize = 8, zorder=6,
        label=f'N = {(count):.0f}')
ax.axhline(0, ls='-', c='k', lw=1.5)
ax.set_title('Pair of beads inside')
ax.set_xlim([0, 100])
ax.set_xlabel('Time (s)')
# ax.set_ylabel('Thickness (nm)')
ax.grid(which='major', axis='both')
ax.legend(fontsize=8)



plt.show()

# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %% Supp Figure 2

# %%% 3e essai

# Save
SAVE = True
figSubDir = 'S2'
name = 'S2A_E_vs_h_ManyMetrics'

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix

XCols = ['ctFieldThickness', 'surroundingThickness', 'bestH0', 'H0_f_<_500']
YCols = ['E_f_<_400', 'E_f_<_500', 'E_f_<_600', 'E_Full', ]

dict_Xlabels = {'ctFieldThickness' : r'$H_{5mT}$', 
                'surroundingThickness' : r'$H_{surrounding}$', 
                'bestH0' : r'$H_{15\%}$', 
                'H0_f_<_500' : r'$H_{500}$',
                }

dict_Ylabels = {'E_f_<_400' : r'$E_{400}$', 
                'E_f_<_500' : r'$E_{500}$', 
                'E_f_<_600' : r'$E_{600}$',
                'E_Full' : r'$E_{full}$',
                }

nX = len(XCols)
nY = len(YCols)

fig, axes = plt.subplots(nY, nX, figsize = (17/cm_in, 12/cm_in), sharey='row', sharex='col')

for j, XCol in enumerate(XCols):
    for i, YCol in enumerate(YCols):
        # Filter
        Filters = [(df['validatedThickness'] == True), 
                   (df['substrate'] == substrate),
                   (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
                   (df['drug'].apply(lambda x : x in drugs)),
                   (df['date'].apply(lambda x : x not in excluded_dates)),
                   (df[XCol] < 1000),
                   (df['normal field'] == 5),
                   (df[YCol] <= 8e5),
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
        
        #### Plot
        # fig, ax = plt.subplots(1, 1, figsize=(12/cm_in, 11/cm_in))
        ax = axes[i, j]
        
        # win, hin = 0.35, 0.35*(11/12)
        # xin, yin = 0.95-win, 0.93-hin 
        # ax_in = ax.inset_axes([xin, yin, win, hin])
        
        ax = ax
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        sns.scatterplot(ax = ax, x=df_plot[XCol].values, y=df_plot[YCol+'_wAvg'].values/1000, 
                        marker = 'o', s = 17, color = apm.cL_Set2[0], alpha = 0.33)
        Xfit, Yfit = np.log(df_plot[XCol].values), np.log(df_plot[YCol+'_wAvg'].values/1000)
        
        wd=1/(np.std(Xfit)) # **2
        we=1/(np.std(Yfit)) # **2

        [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
        
        A, k = np.exp(b), a
        pval = results.pvalue_pearson
        # Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
        # Yplot = A * Xplot**k
        [k_ciw, b_ciw] = results.params_ciw
        text_pval = apm.pval2text(pval, n_digits = 4, space = True)
        # ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 2.0,
        #         label = text_pval)
        colorFit = apm.lightenColor(apm.cL_Set2[0], 0.7)
        apm.drawPowerLine(ax, (1, A), k, ls = '--', c = colorFit, lw = 2.0)
                # label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
                #         f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + '\n' + text_pval)
        LegendMark = mlines.Line2D([], [], color = colorFit, ls='-', 
                                   label = text_pval)
        # LegendMark = mlines.Line2D([], [], color = colorFit, ls='-', 
        #                            label = f'p-val = {pval:.2e}')
        ax.legend(handles=[LegendMark], handlelength = 0.8)
        
        # ax.legend()#.set_visible(False)
        # ax.legend(fontsize = 9, loc = 'lower left')
        # ax.set_title('Average per cell')
        if j==0:
            ax.set_ylabel(dict_Ylabels[YCol], fontsize=matplotlib.rcParams['axes.titlesize']+2)
            ax.tick_params(axis='y', labelsize=matplotlib.rcParams['ytick.labelsize']+2)
        else:
            ax.set_ylabel('')
        if i==3:
            ax.set_xlabel(dict_Xlabels[XCol], fontsize=matplotlib.rcParams['axes.titlesize']+2)
            ax.tick_params(axis='x', labelsize=matplotlib.rcParams['xtick.labelsize']+2)
        else:
            ax.set_xlabel('')
        ax.grid(visible=True, which='major', axis='both')
        ax.set_xlim([50, 1100])
        ax.set_ylim([0.5, 500])
        # ax.tick_params(axis='both', direction='in', which='both')
        
        
        hM, hL, hH = ufun.getLogNDistributionDescriptors(df_f[XCol].values)
        EM, EL, EH = ufun.getLogNDistributionDescriptors(df_f[YCol].values/1000)
        print(f'For {XCol} vs {YCol}')
        print(f'By compression, N = {len(df_f):.0f}')
        print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
        print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
        print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
        print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')
            
        # #### Inset
        # ax = ax_in
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        
        # color = apm.cL_Set2[0]
        
        # sns.scatterplot(ax = ax, x=df_plot[XCol].values, y=df_plot[YCol+'_wAvg'].values/1000, 
        #                 marker = 'o', s = 20, color = apm.cL_Set2[0], alpha = 0.6)
        # Xfit, Yfit = np.log(df_plot[XCol].values), np.log(df_plot[YCol+'_wAvg'].values/1000)
        
        # [a, b], results = ufun.fitLineTLS(Xfit, Yfit)
        # A, k = np.exp(b), a
        # pval = results.pvalue_pearson
        # Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
        # Yplot = A * Xplot**k
        # text_pval = apm.pval2text(pval, n_digits = 3, space = True)
        # ax.plot(Xplot, Yplot, ls = '--', c = apm.lightenColor(color, 0.7), lw = 1.5,)
        #         # label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + \
        #         #         f'\n$R^2$  = {R2:.2f}' + '\n' + text_pval)
        
        # # ax.legend(fontsize = 9, loc = 'lower left')
        # ax.set_title('Average per cell', fontsize=10)
        # ax.grid()
        # # ax.set_ylabel('$E_{500}$ (kPa)')
        # # ax.set_xlabel('$H_0$ (nm)')
        # ax.set_xlim([80, 1100])
        # ax.set_ylim([0.5, 50])
        # ax.tick_params(axis='both', direction='in', which='both', labelsize=9)
        # # ax.set_xticklabels(fontsize=9)
        # # ax.set_yticklabels(fontsize=9)
        

        
        # Count
        # CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
        
# Show
plt.tight_layout()
plt.show()
        
        
# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')