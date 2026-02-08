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
from scipy.stats import mannwhitneyu, shapiro
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

# %% Diverse tasks

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


# %% -------


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
SAVE = False
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
SAVE = False
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
    
    
    
    
# %% Plots EvH version chocolat

# %%% Premier essai

# Save
SAVE = False
figSubDir = 'E-h'
name = 'E_vs_h_CHOCOLATE'

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
                'surroundingThickness' : r'$H_{surr}$', 
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

fig, axes = plt.subplots(nY, nX, figsize = (5*nX, 4.5*nY), sharex='row', sharey='col')

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
        # fig, ax = plt.subplots(1, 1, figsize=(12/cm_in, 11/cm_in))
        ax = axes[i, j]
        
        # win, hin = 0.35, 0.35*(11/12)
        # xin, yin = 0.95-win, 0.93-hin 
        # ax_in = ax.inset_axes([xin, yin, win, hin])
        
        ax = ax
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        sns.scatterplot(ax = ax, x=df_f[XCol].values, y=df_f[YCol].values/1000, 
                        marker = 'o', s = 25, color = 'blue', edgecolor = 'None', alpha = 0.3, 
                        label='All compressions')
        Xfit, Yfit = np.log(df_f[XCol].values), np.log(df_f[YCol].values/1000)
        
        wd=1/(np.std(Xfit)) # **2
        we=1/(np.std(Yfit)) # **2

        [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
        
        A, k = np.exp(b), a
        pval = results.pvalue_pearson
        Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
        Yplot = A * Xplot**k
        text_pval = apm.pval2text(pval, n_digits = 4, space = True)
        ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 1.5,
                label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
                        f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + '\n' + text_pval)
        
        ax.legend()#.set_visible(False)
        # ax.legend(fontsize = 9, loc = 'lower left')
        # ax.set_title('Average per cell')
        if j==0:
            ax.set_ylabel(dict_Ylabels[YCol])
        else:
            ax.set_ylabel('')
        if i==3:
            ax.set_xlabel(dict_Xlabels[XCol])
        else:
            ax.set_xlabel('')
        ax.grid(visible=True, which='major', axis='both')
        ax.set_xlim([50, 1100])
        ax.set_ylim([0.5, 500])
        # ax.tick_params(axis='both', direction='in', which='both')
            
            
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
    
    
# %%% 2e essai

# Save
SAVE = False
figSubDir = 'E-h'
name = 'E_vs_h_CHOCOLATE'

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
                'surroundingThickness' : r'$H_{surr}$', 
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

fig, axes = plt.subplots(nY, nX, figsize = (3*nX, 2.5*nY), sharey='row', sharex='col')

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
        
        # Plot
        # fig, ax = plt.subplots(1, 1, figsize=(12/cm_in, 11/cm_in))
        ax = axes[i, j]
        
        # win, hin = 0.35, 0.35*(11/12)
        # xin, yin = 0.95-win, 0.93-hin 
        # ax_in = ax.inset_axes([xin, yin, win, hin])
        
        ax = ax
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        sns.scatterplot(ax = ax, x=df_f[XCol].values, y=df_f[YCol].values/1000, 
                        marker = 'o', s = 25, color = 'blue', edgecolor = 'None', alpha = 0.1, 
                        label='')
        Xfit, Yfit = np.log(df_f[XCol].values), np.log(df_f[YCol].values/1000)
        
        wd=1/(np.std(Xfit)) # **2
        we=1/(np.std(Yfit)) # **2

        [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
        
        A, k = np.exp(b), a
        pval = results.pvalue_pearson
        Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
        Yplot = A * Xplot**k
        text_pval = apm.pval2text(pval, n_digits = 4, space = True)
        ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 2.0,
                label = text_pval)
                # label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
                #         f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + '\n' + text_pval)
        
        ax.legend()#.set_visible(False)
        # ax.legend(fontsize = 9, loc = 'lower left')
        # ax.set_title('Average per cell')
        if j==0:
            ax.set_ylabel(dict_Ylabels[YCol], fontsize=18)
        else:
            ax.set_ylabel('')
        if i==3:
            ax.set_xlabel(dict_Xlabels[XCol], fontsize=18)
        else:
            ax.set_xlabel('')
        ax.grid(visible=True, which='major', axis='both')
        ax.set_xlim([50, 1100])
        ax.set_ylim([0.5, 500])
        # ax.tick_params(axis='both', direction='in', which='both')
            
            
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
    
    
# %%% 3e essai

# Save
SAVE = False
figSubDir = 'E-h'
name = 'E_vs_h_CHOCOLATE'

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
                'surroundingThickness' : r'$H_{surr}$', 
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

fig, axes = plt.subplots(nY, nX, figsize = (3*nX, 2.5*nY), sharey='row', sharex='col')

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
                        marker = 'o', s = 20, color = apm.cL_Set2[0], alpha = 0.6)
        Xfit, Yfit = np.log(df_plot[XCol].values), np.log(df_plot[YCol+'_wAvg'].values/1000)
        
        wd=1/(np.std(Xfit)) # **2
        we=1/(np.std(Yfit)) # **2

        [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
        
        A, k = np.exp(b), a
        pval = results.pvalue_pearson
        # Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
        # Yplot = A * Xplot**k
        text_pval = apm.pval2text(pval, n_digits = 4, space = True)
        # ax.plot(Xplot, Yplot, ls = '--', c = 'dimgray', lw = 2.0,
        #         label = text_pval)
        apm.drawPowerLine(ax, (1, A), k, ls = '--', c = 'dimgray', lw = 2.0,
                label = text_pval)
                # label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
                #         f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + '\n' + text_pval)
        
        ax.legend()#.set_visible(False)
        # ax.legend(fontsize = 9, loc = 'lower left')
        # ax.set_title('Average per cell')
        if j==0:
            ax.set_ylabel(dict_Ylabels[YCol], fontsize=18)
        else:
            ax.set_ylabel('')
        if i==3:
            ax.set_xlabel(dict_Xlabels[XCol], fontsize=18)
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
    
    

