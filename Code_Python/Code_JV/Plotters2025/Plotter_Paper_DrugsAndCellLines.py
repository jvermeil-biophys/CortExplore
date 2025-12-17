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
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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


# %% > Data import & export

# %%% > Drugs

# MecaData_DrugV3 = taka2.getMergedTable('MecaData_Drugs_V3')
# MecaData_DrugV3 = MecaData_DrugV3[MecaData_DrugV3['substrate'] == '20um fibronectin discs']

# MecaData_DrugV4 = taka2.getMergedTable('MecaData_Drugs_V4')
# MecaData_DrugV4 = MecaData_DrugV4[MecaData_DrugV4['substrate'] == '20um fibronectin discs']

MecaData_DrugV5 = taka2.getMergedTable('MecaData_Drugs_V5')
MecaData_DrugV5 = MecaData_DrugV5[MecaData_DrugV5['substrate'] == '20um fibronectin discs']

MecaData_Drug = MecaData_DrugV5

# %%% > Cell types

MecaData_CellTypesV2 = taka2.getMergedTable('MecaData_CellTypes_V2')

MecaData_Phy3 = taka3.getMergedTable('MecaData_Physics_V3')
MecaData_Phy3 = MecaData_Phy3.dropna(axis=0, subset='date')

MecaData_CellTypes = pd.concat([MecaData_CellTypesV2, MecaData_Phy3])
MecaData_CellTypes['Indent_ID'] = MecaData_CellTypes['cellID'] + '_' + MecaData_CellTypes['compNum'].astype('str')
MecaData_CellTypes = MecaData_CellTypes.drop_duplicates(subset='Indent_ID')


# %%% Check content

def checkContent(MecaData):

    print(apm.CYAN + 'Dates' + apm.NORMAL)
    print([x for x in MecaData['date'].unique()])
    print('')
    
    print(apm.CYAN + 'Cell types' + apm.NORMAL)
    print([x for x in MecaData['cell type'].unique()])
    print('')
    
    print(apm.CYAN + 'Cell subtypes' + apm.NORMAL)
    print([x for x in MecaData['cell subtype'].unique()])
    print('')
    
    print(apm.CYAN + 'Drugs' + apm.NORMAL)
    print([x for x in MecaData['drug'].unique()])
    print('')
    
    print(apm.CYAN + 'Substrates' + apm.NORMAL)
    print([x for x in MecaData['substrate'].unique()])
    print('')
    
    print(apm.CYAN + 'Resting Fields' + apm.NORMAL)
    print([x for x in MecaData['normal field'].unique()])
    print('')

print(apm.ORANGE + 'Drugs' + apm.NORMAL)
checkContent(MecaData_Drug)

print(apm.ORANGE + 'Cell Types' + apm.NORMAL)
checkContent(MecaData_CellTypes)

# CountByCond, CountByCell = makeCountDf(MecaData_Drug, 'date')


# %% -------

# %% New plots with drugs DataFrame

# %%% 1. Plot with average per drug

# %%%% Dataset

df = MecaData_Drug

drugs = ['dmso', 'blebbistatin', 'none', 'Y27', 'ck666', 'latrunculinA', 'LIMKi']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

# XCol = 'ctFieldThickness'
XCol = 'bestH0'
YCol = 'E_f_<_500'

df, condCol = apm.makeCompositeCol(df, cols=['drug', 'concentration'])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           # (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df[XCol] > 50),
           (df[YCol] < 1e5),
           ]

df_f = apm.filterDf(df, Filters)

logMean = lambda x : np.exp(np.mean(np.log(x)))
logStd = lambda x : np.exp(np.std(np.log(x)))
# x = np.array([1, 10, 100, 1000, 10000])
# print(np.std(x))
# print(logStd(x))

# Group By Step 1
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = YCol, weightCol = 'ciw'+YCol, weight_method = 'ciw^2')
df_gC = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Group By Step 2
df_fg_1 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = ['bestH0'],
                   aggFun = logMean) #.drop(columns=['cellID']).reset_index()
df_fg_2 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = ['bestH0'],
                   aggFun = 'std') #.drop(columns=['cellID']).reset_index()
df_fg_3 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = ['bestH0'],
                   aggFun = 'count') #.drop(columns=['cellID']).reset_index()
df_fg_1 = df_fg_1[['bestH0']].rename(columns={'bestH0': "H0_mean"})
df_fg_2 = df_fg_2[['bestH0']].rename(columns={'bestH0': "H0_std"})
df_fg_3 = df_fg_3[['bestH0']].rename(columns={'bestH0': "count"})
df_fg2_1 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [YCol + '_wAvg'],
                   aggFun = logMean)
df_fg2_2 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [YCol + '_wAvg'],
                   aggFun = 'std')
df_fg2_1 = df_fg2_1.rename(columns={YCol + '_wAvg': "E_mean"})
df_fg2_2 = df_fg2_2[[YCol + '_wAvg']].rename(columns={YCol + '_wAvg': "E_std"})
df_gD = pd.merge(left=df_fg_3, right=df_fg_1, on=condCol, how='inner')
df_gD = pd.merge(left=df_gD, right=df_fg_2, on=condCol, how='inner')
df_gD = pd.merge(left=df_gD, right=df_fg2_1, on=condCol, how='inner')
df_gD = pd.merge(left=df_gD, right=df_fg2_2, on=condCol, how='inner')
df_gD['H0_sem'] = df_gD['H0_std']/np.power(df_gD['count'], 0.5)
df_gD['E_sem'] = df_gD['E_std']/np.power(df_gD['count'], 0.5)

# df_gD = df_gD.reset_index()

# %%%% Plot 1

fig, ax = plt.subplots(1, 1, figsize = (10, 10))
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_yscale('log')
# ax.errorbar(df_gD['H0_mean'], df_gD['E400_mean'], 
#             xerr=df_gD['H0_std'], yerr=df_gD['E400_std'], 
#             ls = '', marker = 'o', ms=1, color='grey', zorder=3)
conds = df_gD.index.unique()
for i, cond in enumerate(conds):
    color = apm.styleDict[cond]['color']
    marker = apm.styleDict[cond]['marker']
    ax.errorbar(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3, 
                xerr=df_gD.loc[cond, 'H0_sem'], yerr=df_gD.loc[cond, 'E_sem']/1e3, 
                ls = '', marker = 'o', ms=1, color=color, 
                elinewidth = 3, capsize = 5, capthick = 3, zorder=3)
    ax.plot(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3,
            marker = marker, color = color, 
            ms=12, mec='k', label = apm.renameDict[cond], lw=0.5, zorder=6)

ax.grid()
# ax.set_xlim([0, 600])
ax.set_xlim([100, 1000])
ax.set_xlabel('$H_0$ (nm)')
# ax.set_ylim([0, 30])
ax.set_ylim([1, 50])
ax.set_ylabel('$E_{500}$ (kPa)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 12)
fig.tight_layout()

plt.show()


# %%%% Plot 1 bis 

# Save
SAVE = True
figSubDir = 'E-h'
name = 'E500_vs_h0_drugs'

sD = apm.styleDict_V2
rD = apm.renameDict

fig, ax = plt.subplots(1, 1, figsize = (17/cm_in, 12/cm_in))
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_yscale('log')
# ax.errorbar(df_gD['H0_mean'], df_gD['E400_mean'], 
#             xerr=df_gD['H0_std'], yerr=df_gD['E400_std'], 
#             ls = '', marker = 'o', ms=1, color='grey', zorder=3)
# conds = df_gD.index.unique()
conds = ['dmso & 0.0']
for i, cond in enumerate(conds):
    color = sD[cond]['color']
    marker = sD[cond]['marker']
    
    df_c = df_gC[df_gC[condCol] == cond]
    X, Y = df_c['bestH0'].values, df_c[YCol + '_wAvg'].values/1000
    
    ax.plot(X, Y,
            marker = marker, color = color, ls='',
            ms=6, alpha = 0.3, zorder=3)
    ax.errorbar(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3, 
                xerr=df_gD.loc[cond, 'H0_sem'], yerr=df_gD.loc[cond, 'E_sem']/1e3, 
                ls = '', marker = 'o', ms=1, color=color, 
                elinewidth = 3, capsize = 5, capthick = 3, zorder=5)
    ax.plot(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3,
            marker = marker, color = color, ls='',
            ms=10, mec='k', label = rD[cond], lw=0.5, zorder=6)

conds = ['Y27 & 50.0', 'ck666 & 50.0', 'latrunculinA & 0.5', 'LIMKi & 20.0']
for i, cond in enumerate(conds):
    color = sD[cond]['color']
    marker = sD[cond]['marker']
    
    df_c = df_gC[df_gC[condCol] == cond]
    X, Y = df_c['bestH0'].values, df_c[YCol + '_wAvg'].values/1000
    
    ax.plot(X, Y,
            marker = marker, color = color, ls='',
            ms=6, mec='w', mew=0.5, zorder=4)
    ax.errorbar(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3, 
                xerr=df_gD.loc[cond, 'H0_sem'], yerr=df_gD.loc[cond, 'E_sem']/1e3, 
                ls = '', marker = 'o', ms=1, color=apm.lightenColor(color, 0.8), 
                elinewidth = 3, capsize = 5, capthick = 3, zorder=5)
    ax.plot(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3,
            marker = marker, color = apm.lightenColor(color, 0.8), ls='',
            ms=10, mec='k', label = rD[cond], lw=0.5, zorder=6)

ax.grid()
# ax.set_xlim([0, 600])
ax.set_xlim([50, 2000])
ax.set_xlabel('$H_0$ (nm)')
# ax.set_ylim([0, 30])
ax.set_ylim([0.2, 150])
ax.set_ylabel('$E_{500}$ (kPa)')
ax.legend(fontsize = 12) # loc='center left', bbox_to_anchor=(1, 0.5), 
fig.tight_layout()

plt.show()

# Count
# CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    # CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%%% Plot 2 bis

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
ax.set_xscale('log')
ax.set_yscale('log')

# ax.errorbar(df_gD['H0_mean'], df_gD['E400_mean'], 
#             xerr=df_gD['H0_std'], yerr=df_gD['E400_std'], 
#             ls = '', marker = 'o', ms=1, color='grey', zorder=3)
conds = df_gC[condCol].unique()


conds = ['dmso & 0.0']
for i, cond in enumerate(conds):
    color = apm.styleDict[cond]['color']
    marker = apm.styleDict[cond]['marker']
    df_c = df_gC[df_gC[condCol] == cond]
    
    X, Y = df_c['bestH0'].values, df_c[YCol + '_wAvg'].values/1000
    Xfit, Yfit = np.log(X), np.log(Y)
    [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
    A, k = np.exp(b), a
    R2 = w_results.rsquared
    Xplot = np.exp(np.linspace(1, 15, 50))
    Yplot = A * Xplot**k
    
    parms, results = ufun.fitLine(Xfit, Yfit)
    pval = results.pvalues[1]
    
    ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 3.0, zorder=4, alpha = 0.6,
            label = apm.renameDict[cond] + '\n' + f'$R^2$={R2:.2f}' + ', ' + apm.pval2text(pval, n_digits = 2, space=False) + '\n')
    # ax.plot(X, Y,
    #         marker = marker, color = color, ls='',
    #         ms=6, mec='k', mew=0.5, zorder=4)
    ax.plot(X, Y,
            marker = marker, color = color, ls='',
            ms=9, alpha = 0.4, zorder=3)


conds = ['blebbistatin & 50.0', 'Y27 & 50.0',
       'ck666 & 50.0', 'latrunculinA & 0.5', 'LIMKi & 20.0']
for i, cond in enumerate(conds):
    color = apm.styleDict[cond]['color']
    marker = apm.styleDict[cond]['marker']
    df_c = df_gC[df_gC[condCol] == cond]
    
    X, Y = df_c['bestH0'].values, df_c[YCol + '_wAvg'].values/1000
    Xfit, Yfit = np.log(X), np.log(Y)
    [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
    A, k = np.exp(b), a
    R2 = w_results.rsquared
    Xplot = np.exp(np.linspace(np.log(75), np.log(1250), 50))
    Yplot = A * Xplot**k
    
    parms, results = ufun.fitLine(Xfit, Yfit)
    pval = results.pvalues[1]
    
    ax.plot(Xplot, Yplot, ls = '-', c = color, lw = 3.0, zorder=6,
            label = apm.renameDict[cond] + '\n' + f'$R^2$={R2:.2f}' + ', ' + apm.pval2text(pval, n_digits = 2, space=False) + '\n')
    ax.plot(X, Y,
            marker = marker, color = color, ls='',
            ms=8, mec='w', mew=0.5, zorder=4)

        # f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
    # ax.errorbar(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E400_mean']/1e3, 
    #             xerr=df_gD.loc[cond, 'H0_std'], yerr=df_gD.loc[cond, 'E400_std']/1e3, 
    #             ls = '', marker = 'o', ms=1, color=color, zorder=3)
    # ax.plot(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E400_mean']/1e3,
    #         marker = marker, color = color, 
    #         ms=10, mec='k', label = apm.renameDict[cond], lw=0.5, zorder=6)

ax.grid()
# ax.set_xlim([0, 900])
ax.set_xlim([50, 2000])
ax.set_xlabel('$H_0$ (nm)')
# ax.set_ylim([0, 40])
ax.set_ylim([0.2, 150])
ax.set_ylabel('$E_{500}$ (kPa)')
ax.legend(title = r'$\bf{Fit\ y\ =\ A.x^k}$', loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 12)
fig.tight_layout()

plt.show()


# %%%% Plot 3

#### Function

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
        
    print(x, y)
    cov = np.cov(x, y)
    print(cov)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


fig, ax = plt.subplots(1, 1, figsize = (15, 10))
# ax.set_xscale('log')
# ax.set_yscale('log')


conds = df_gC[condCol].unique()
for i, cond in enumerate(conds[:]):
    color = apm.styleDict[cond]['color']
    marker = apm.styleDict[cond]['marker']
    df_c = df_gC[df_gC[condCol] == cond]
    
    Xfit, Yfit = np.log10(df_c['bestH0'].values), np.log10(df_c[YCol + '_wAvg'].values/1000)
    [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
    A, k = np.exp(np.log(10)*b), a
    R2 = w_results.rsquared
    # Xplot = np.exp(np.log(10)*np.linspace(min(Xfit), max(Xfit), 50))
    # Yplot = A * Xplot**k
    Xplot = np.linspace(min(Xfit), max(Xfit), 50)
    Yplot = b + Xplot*a
    
    parms, results = ufun.fitLine(Xfit, Yfit)
    pval = results.pvalues[1]
    
    ax.plot(Xplot, Yplot, ls = '-', c = color, lw = 2.0,
            label = apm.renameDict[cond] + ' - ' + f'$R^2$={R2:.2f}' + f' p-val={pval:.2f}')
        # f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
    # ax.errorbar(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E400_mean']/1e3, 
    #             xerr=df_gD.loc[cond, 'H0_std'], yerr=df_gD.loc[cond, 'E400_std']/1e3, 
    #             ls = '', marker = 'o', ms=1, color=color, zorder=3)
    # ax.plot(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E400_mean']/1e3,
    #         marker = marker, color = color, 
    #         ms=10, mec='k', label = apm.renameDict[cond], lw=0.5, zorder=6)
    
    confidence_ellipse(Xfit, Yfit, ax, n_std=3.0, facecolor='none', edgecolor=color) #, **kwargs)


ax.grid()
# ax.set_xlim([0, 900])
ax.set_xlabel('$H_0$ (nm)')
# ax.set_ylim([0, 40])
ax.set_ylabel('$E_{500}$ (kPa)')
# ax.legend(title = r'$\bf{Fit\ y\ =\ A.x^k}$', loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 12)
fig.tight_layout()

plt.show()

# %%%% Best one - 1 bis

# Save
SAVE = False
figSubDir = 'E-h'
name = 'E500_vs_h0_drugs'

df = MecaData_Drug

drugs = ['dmso', 'blebbistatin', 'none', 'Y27', 'ck666', 'latrunculinA', 'LIMKi']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

# XCol = 'ctFieldThickness'
XCol = 'bestH0'
YCol = 'E_f_<_500'

df, condCol = apm.makeCompositeCol(df, cols=['drug', 'concentration'])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           # (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df[XCol] > 50),
           (df[YCol] < 10e4),
           ]

df_f = apm.filterDf(df, Filters)

logMean = lambda x : np.exp(np.mean(np.log(x)))
logStd = lambda x : np.exp(np.std(np.log(x)))
# x = np.array([1, 10, 100, 1000, 10000])
# print(np.std(x))
# print(logStd(x))

# Group By Step 1
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = YCol, weightCol = 'ciw'+YCol, weight_method = 'ciw^2')
df_gC = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Group By Step 2
df_fg_1 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = ['bestH0'],
                   aggFun = logMean) #.drop(columns=['cellID']).reset_index()
df_fg_2 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = ['bestH0'],
                   aggFun = 'std') #.drop(columns=['cellID']).reset_index()
df_fg_3 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = ['bestH0'],
                   aggFun = 'count') #.drop(columns=['cellID']).reset_index()
df_fg_1 = df_fg_1[['bestH0']].rename(columns={'bestH0': "H0_mean"})
df_fg_2 = df_fg_2[['bestH0']].rename(columns={'bestH0': "H0_std"})
df_fg_3 = df_fg_3[['bestH0']].rename(columns={'bestH0': "count"})
df_fg2_1 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [YCol + '_wAvg'],
                   aggFun = logMean)
df_fg2_2 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [YCol + '_wAvg'],
                   aggFun = 'std')
df_fg2_1 = df_fg2_1.rename(columns={YCol + '_wAvg': "E_mean"})
df_fg2_2 = df_fg2_2[[YCol + '_wAvg']].rename(columns={YCol + '_wAvg': "E_std"})
df_gD = pd.merge(left=df_fg_3, right=df_fg_1, on=condCol, how='inner')
df_gD = pd.merge(left=df_gD, right=df_fg_2, on=condCol, how='inner')
df_gD = pd.merge(left=df_gD, right=df_fg2_1, on=condCol, how='inner')
df_gD = pd.merge(left=df_gD, right=df_fg2_2, on=condCol, how='inner')
df_gD['H0_sem'] = df_gD['H0_std']/np.power(df_gD['count'], 0.5)
df_gD['E_sem'] = df_gD['E_std']/np.power(df_gD['count'], 0.5)

# df_gD = df_gD.reset_index()

sD = apm.styleDict_V2
rD = apm.renameDict

fig, ax = plt.subplots(1, 1, figsize = (17/cm_in, 12/cm_in))
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_yscale('log')
# ax.errorbar(df_gD['H0_mean'], df_gD['E400_mean'], 
#             xerr=df_gD['H0_std'], yerr=df_gD['E400_std'], 
#             ls = '', marker = 'o', ms=1, color='grey', zorder=3)
# conds = df_gD.index.unique()
conds = ['dmso & 0.0']
for i, cond in enumerate(conds):
    color = sD[cond]['color']
    marker = sD[cond]['marker']
    
    df_c = df_gC[df_gC[condCol] == cond]
    X, Y = df_c['bestH0'].values, df_c[YCol + '_wAvg'].values/1000
    
    ax.plot(X, Y,
            marker = marker, color = color, ls='',
            ms=6, alpha = 0.3, zorder=3)
    ax.errorbar(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3, 
                xerr=df_gD.loc[cond, 'H0_sem'], yerr=df_gD.loc[cond, 'E_sem']/1e3, 
                ls = '', marker = 'o', ms=1, color=color, 
                elinewidth = 3, capsize = 5, capthick = 3, zorder=5)
    ax.plot(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3,
            marker = marker, color = color, ls='',
            ms=10, mec='k', label = rD[cond], lw=0.5, zorder=6)

conds = ['Y27 & 50.0', 'ck666 & 50.0', 'latrunculinA & 0.5', 'LIMKi & 20.0']
for i, cond in enumerate(conds):
    color = sD[cond]['color']
    marker = sD[cond]['marker']
    
    df_c = df_gC[df_gC[condCol] == cond]
    X, Y = df_c['bestH0'].values, df_c[YCol + '_wAvg'].values/1000
    
    ax.plot(X, Y,
            marker = marker, color = color, ls='',
            ms=6, mec='w', mew=0.5, zorder=4)
    ax.errorbar(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3, 
                xerr=df_gD.loc[cond, 'H0_sem'], yerr=df_gD.loc[cond, 'E_sem']/1e3, 
                ls = '', marker = 'o', ms=1, color=apm.lightenColor(color, 0.8), 
                elinewidth = 3, capsize = 5, capthick = 3, zorder=5)
    ax.plot(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3,
            marker = marker, color = apm.lightenColor(color, 0.8), ls='',
            ms=10, mec='k', label = rD[cond], lw=0.5, zorder=6)

ax.grid()
# ax.set_xlim([0, 600])
ax.set_xlim([50, 2000])
ax.set_xlabel('$H_0$ (nm)')
# ax.set_ylim([0, 30])
ax.set_ylim([0.2, 150])
ax.set_ylabel('$E_{500}$ (kPa)')
ax.legend(fontsize = 12) # loc='center left', bbox_to_anchor=(1, 0.5), 
fig.tight_layout()

plt.show()

# Count
# CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    # CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')




# %% --------------------------

# %% New plots with cell types DataFrame

# %%%% Six cell types

# Save
SAVE = False
figSubDir = 'E-h'
name = 'E500_vs_h0_6celltypes'

df = MecaData_CellTypes

df, condCol = apm.makeCompositeCol(df, cols=['cell type', 'cell subtype'])

# XCol = 'ctFieldThickness'
XCol = 'bestH0'
YCol = 'E_f_<_500'

# Define
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# substrates = ['BSA coated glass', '20um fibronectin discs']

CountByCond, CountByCell = apm.makeCountDf(df, condCol)
dfDC = df[df['cell type'] == 'DC']

# Filter
Filters = [(df['validatedThickness'] == True), 
           # (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x not in excluded_dates)),
           (df[XCol] < 1000),
           # (df['normal field'] == 5),
           (df[YCol] <= 1e5),
           (df['valid' + YCol[1:]] == True), 
           ]
df_f = apm.filterDf(df, Filters)

df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = apm.makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])
CountByCond2, CountByCell2 = apm.makeCountDf(df_f, condCol)


# Filter 2
Case_A1 = (df_f['cell type'].apply(lambda x : x in ['HoxB8-Macro']))
Case_A2 = (df_f['substrate'] == 'bare glass')
Case_B1 = (df_f['cell type'].apply(lambda x : x in ['DC', 'Dicty']))
Case_B2 = (df_f['substrate'] == 'BSA coated glass')
Case_C1 = (df_f['cell type'].apply(lambda x : x in ['3T3', 'MDCK', 'HeLa']))
Case_C2 = (df_f['substrate'] == '20um fibronectin discs')
Case_D1 = (df_f['cell type'].apply(lambda x : x in ['MDCK', 'DC']))
Case_D2 = (df_f['normal field'] == 5)
Filters = [((Case_A1 & Case_A2) | (Case_B1 & Case_B2) | (Case_C1 & Case_C2)),
           (Case_D1 | Case_D2),
           ]
df_f = apm.filterDf(df_f, Filters)


# Order
co_order = ['3T3 & Atcc-2023', 
            'HeLa & fucci', 
            'MDCK & WT',
            'DC & mouse-primary', 
            'HoxB8-Macro & ctrl', 
            'Dicty & DictyBase-WT', 
            ]

colorsD = {'3T3 & Atcc-2023'     : apm.cL_Set2[0], 
          'HeLa & fucci'         : apm.cL_Set2[1],  
          'DC & mouse-primary'   : apm.cL_Set2[2],  
          'Dicty & DictyBase-WT' : apm.cL_Set2[3],  
          'HoxB8-Macro & ctrl'   : apm.cL_Set2[4],  
          'MDCK & WT'            : apm.cL_Set2[5],
          }

rD = {'3T3 & Atcc-2023'      :  '3T3 ATCC', 
      'HeLa & fucci'         :  'HeLa FUCCI',  
      'DC & mouse-primary'   :  'Primary DC',  
      'Dicty & DictyBase-WT' :  'Dictys Ax3',  
      'HoxB8-Macro & ctrl'   :  'HoxB8 Macro',  
      'MDCK & WT'            :  'MDCK',
      }

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[[XCol]]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = YCol, weightCol = 'ciw' + YCol, weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(2, 3, figsize=(17/cm_in, 12/cm_in), sharex=True, sharey=True)
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
    # color = gs.cL_Set2[i]
    color = colorsD[co_order[i]]
    
    # fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
    #                 XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
    #                 modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
    #                 figSizeFactor = 1, markersizeFactor = 0.5)
    
    medianX = np.median(df_fc[XCol].values)
    medianY = np.median(df_fc[YCol].values)
    
    if i == 0:
        alpha = 0.3
        s = 10
        medianX = 259
        medianY = 4.8
    else:
        alpha = 0.5
        s = 20
    
    sns.scatterplot(ax = ax, x=df_fc[XCol].values, y=df_fc[YCol].values, 
                    marker = 'o', s = s, color = color, edgecolor = 'k', linewidth =0.5, alpha = alpha,
                    zorder = 3) # , label = 'Median $H_0$ = ' + f'{medianX:.0f} nm'\
                                  #      f'\nMedian $E$ = ' + f'{medianY:.1f} kPa')
    Xfit, Yfit = np.log(df_fc[XCol].values), np.log(df_fc[YCol].values)
    
    [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=4, we=1)
    A, k = np.exp(b), a
    pval = results.pvalues[0]
    Xplot = np.exp(np.linspace(1, 9, 50))
    Yplot = A * Xplot**k
    text_pval = apm.pval2text(pval, n_digits = 2, space = True)
        
    # [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
    # A, k = np.exp(b), a
    # k_cihw = (results.conf_int(0.05)[1, 1] - results.conf_int(0.05)[1, 0])/2
    # R2 = w_results.rsquared
    # pval = results.pvalues[1]
    # Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    # Yplot = A * Xplot**k
    
    # [b, a], results = ufun.fitLine(Xfit, Yfit)
    # R2 = results.rsquared
    # pval = results.pvalues[1]
    # Xplot = (np.linspace(min(Xfit), max(Xfit), 50))
    # Yplot = a * Xplot + b
    
    # ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 2.0, zorder = 6,
    #         label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + \
    #                     f'\nA = {A:.1e}' + \
    #                     f'\nk  = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
    #                     f'\n$R^2$ = {R2:.2f}' + \
    #                     f'\np-val = {pval:.2f}')
        
    ax.plot(Xplot, Yplot, ls = '--', c = apm.lightenColor(color, 0.8), lw = 1.5, zorder = 6,
            label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
                    f'\nk  = {k:.2f}' + '\n' + text_pval)

    ax.legend(fontsize = 7, loc = 'best', handlelength=1)
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
    # renameAxes(ax, renameDict, format_xticks = False)
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
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%%% Four cell types

# Save
SAVE = True
figSubDir = 'E-h'
name = 'E500_vs_h0_4celltypes'

df = MecaData_CellTypes

df, condCol = apm.makeCompositeCol(df, cols=['cell type', 'cell subtype'])

# XCol = 'ctFieldThickness'
XCol = 'bestH0'
YCol = 'E_f_<_500'

# Define
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# substrates = ['BSA coated glass', '20um fibronectin discs']

CountByCond, CountByCell = apm.makeCountDf(df, condCol)
dfDC = df[df['cell type'] == 'DC']

# Filter
Filters = [(df['validatedThickness'] == True), 
           # (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x not in excluded_dates)),
           (df[XCol] < 1000),
           # (df['normal field'] == 5),
           (df[YCol] <= 1e5),
           (df['valid' + YCol[1:]] == True), 
           ]
df_f = apm.filterDf(df, Filters)

df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = apm.makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])
CountByCond2, CountByCell2 = apm.makeCountDf(df_f, condCol)


# Filter 2
Case_A1 = (df_f['cell type'].apply(lambda x : x in ['HoxB8-Macro']))
Case_A2 = (df_f['substrate'] == 'bare glass')
Case_B1 = (df_f['cell type'].apply(lambda x : x in ['DC', 'Dicty']))
Case_B2 = (df_f['substrate'] == 'BSA coated glass')
Case_C1 = (df_f['cell type'].apply(lambda x : x in ['3T3', 'MDCK', 'HeLa']))
Case_C2 = (df_f['substrate'] == '20um fibronectin discs')
Case_D1 = (df_f['cell type'].apply(lambda x : x in ['MDCK', 'DC']))
Case_D2 = (df_f['normal field'] == 5)
Filters = [((Case_A1 & Case_A2) | (Case_B1 & Case_B2) | (Case_C1 & Case_C2)),
           (Case_D1 | Case_D2),
           ]
df_f = apm.filterDf(df_f, Filters)


# Order
co_order = [
            # '3T3 & Atcc-2023', 
            'HeLa & fucci', 
            'MDCK & WT',
            'DC & mouse-primary', 
            # 'HoxB8-Macro & ctrl', 
            'Dicty & DictyBase-WT', 
            ]

colorsD = {
          # '3T3 & Atcc-2023'     : apm.cL_Set2[0], 
          'HeLa & fucci'         : apm.cL_Set2[1],  
          'DC & mouse-primary'   : apm.cL_Set2[2],  
          'Dicty & DictyBase-WT' : apm.cL_Set2[3],  
          # 'HoxB8-Macro & ctrl'   : apm.cL_Set2[4],  
          'MDCK & WT'            : apm.cL_Set2[5],
          }

rD = {
      # '3T3 & Atcc-2023'      :  '3T3 ATCC', 
      'HeLa & fucci'         :  'HeLa FUCCI',  
      'DC & mouse-primary'   :  'Primary DC',  
      'Dicty & DictyBase-WT' :  'Dictys Ax3',  
      # 'HoxB8-Macro & ctrl'   :  'HoxB8 Macro',  
      'MDCK & WT'            :  'MDCK',
      }

# Group By
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[[XCol]]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = YCol, weightCol = 'ciw' + YCol, weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(2, 2, figsize=(17/cm_in, 17/cm_in), sharex=True, sharey=True)
axes = axes.flatten('C')


YCol += '_wAvg'
df_plot[YCol] /= 1000

df_ctrl = df_plot[df_plot[condCol] == '3T3 & Atcc-2023']
Xctrl, Yctrl = df_ctrl[XCol].values, df_ctrl[YCol].values
Xctrl_fit, Yctrl_fit = np.log(Xctrl), np.log(Yctrl)

[a, b], results = ufun.fitLineTLS(Xctrl_fit, Yctrl_fit, wd=4, we=1)
A, k = np.exp(b), a
pval = results.pvalues[0]
Xctrl_plot = np.exp(np.linspace(1, 9, 50))
Yctrl_plot = A * Xplot**k
text_pval = apm.pval2text(pval, n_digits = 2, space = True)
# ax.plot(Xplot, Yplot, ls = '--', c = apm.lightenColor(color, 0.8), lw = 1.5, zorder = 6,
#         label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
#                 f'\nk  = {k:.2f}' + '\n' + text_pval)

for i in range(len(axes)):
    ax = axes[i]
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    color = sD[cond]['color']
    marker = sD[cond]['marker']
    
    sns.scatterplot(ax = ax, x=df_ctrl[XCol].values, y=df_ctrl[YCol].values, 
                    marker = 'o', s = 25, color = 'dimgray', alpha = 0.2,
                    zorder = 3)
    ax.plot(Xctrl_plot, Yctrl_plot, ls = '--', color = 'dimgray', 
            lw = 2.0, zorder = 6, alpha = 0.4)
            # label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
            #         f'\nk  = {k:.2f}' + '\n' + text_pval)

    
    df_fc = df_plot[df_plot[condCol] == co_order[i]]
    print(co_order[i], len(df_fc))
    color = colorsD[co_order[i]]
    
    medianX = np.median(df_fc[XCol].values)
    medianY = np.median(df_fc[YCol].values)
    
    alpha = 1
    s = 30
    
    sns.scatterplot(ax = ax, x=df_fc[XCol].values, y=df_fc[YCol].values, 
                    marker = 'o', s = s, color = color, edgecolor = 'k', linewidth=0.5, alpha = alpha,
                    zorder = 3) # , label = 'Median $H_0$ = ' + f'{medianX:.0f} nm'\
                                  #      f'\nMedian $E$ = ' + f'{medianY:.1f} kPa')
    Xfit, Yfit = np.log(df_fc[XCol].values), np.log(df_fc[YCol].values)
    
    [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=4, we=1)
    A, k = np.exp(b), a
    pval = results.pvalues[0]
    Xplot = np.exp(np.linspace(1, 9, 50))
    Yplot = A * Xplot**k
    text_pval = apm.pval2text(pval, n_digits = 2, space = True)

        
    ax.plot(Xplot, Yplot, ls = '--', c = apm.lightenColor(color, 0.75), lw = 2.0, zorder = 6,
            label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
                    f'\nk  = {k:.2f}' + '\n' + text_pval)

    ax.legend(fontsize = 9, loc = 'best', handlelength=1)
    ax.set_xlabel('$H_{0}$ (nm)')
    ax.set_ylabel('$E_{500}$ (kPa)')
    ax.set_title(co_order[i])
    if i%2 != 0:
        ax.set_ylabel('')
           
# Prettify
rD.update({'E_eff_wAvg':'E_{eff} (kPa)'})

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder=0)
    apm.renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
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
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')



# %% --------------------------