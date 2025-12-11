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

figDir = 'C:/Users/josep/Desktop/Papier/NewFigs'


# %% > Data import & export

# MecaData_DrugV3 = taka2.getMergedTable('MecaData_Drugs_V3')
# MecaData_DrugV3 = MecaData_DrugV3[MecaData_DrugV3['substrate'] == '20um fibronectin discs']

# MecaData_DrugV4 = taka2.getMergedTable('MecaData_Drugs_V4')
# MecaData_DrugV4 = MecaData_DrugV4[MecaData_DrugV4['substrate'] == '20um fibronectin discs']

MecaData_DrugV5 = taka2.getMergedTable('MecaData_Drugs_V5')
MecaData_DrugV5 = MecaData_DrugV5[MecaData_DrugV5['substrate'] == '20um fibronectin discs']

MecaData_Drug = MecaData_DrugV5

# %%% Check content

print('Dates')
print([x for x in MecaData_Drug['date'].unique()])
print('')

print('Cell types')
print([x for x in MecaData_Drug['cell type'].unique()])
print('')

print('Cell subtypes')
print([x for x in MecaData_Drug['cell subtype'].unique()])
print('')

print('Drugs')
print([x for x in MecaData_Drug['drug'].unique()])
print('')

print('Substrates')
print([x for x in MecaData_Drug['substrate'].unique()])
print('')

print('Resting Fields')
print([x for x in MecaData_Drug['normal field'].unique()])
print('')

# CountByCond, CountByCell = makeCountDf(MecaData_Drug, 'date')

# %% -------



# %% New plots for paper

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
           (df[YCol] < 10e4),
           ]

df_f = apm.filterDf(df, Filters)

# Group By Step 1
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = YCol, weightCol = 'ciw'+YCol, weight_method = 'ciw^2')
df_gC = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Group By Step 2
df_fg_1 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = ['bestH0'],
                   aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg_2 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = ['bestH0'],
                   aggFun = 'std') #.drop(columns=['cellID']).reset_index()
df_fg_3 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = ['bestH0'],
                   aggFun = 'count') #.drop(columns=['cellID']).reset_index()
df_fg_1 = df_fg_1[['bestH0']].rename(columns={'bestH0': "H0_mean"})
df_fg_2 = df_fg_2[['bestH0']].rename(columns={'bestH0': "H0_std"})
df_fg_3 = df_fg_3[['bestH0']].rename(columns={'bestH0': "count"})
df_fg2_1 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [YCol + '_wAvg'],
                   aggFun = 'mean')
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

fig, ax = plt.subplots(1, 1, figsize = (10, 10))
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_yscale('log')
# ax.errorbar(df_gD['H0_mean'], df_gD['E400_mean'], 
#             xerr=df_gD['H0_std'], yerr=df_gD['E400_std'], 
#             ls = '', marker = 'o', ms=1, color='grey', zorder=3)
# conds = df_gD.index.unique()
conds = ['dmso & 0.0']
for i, cond in enumerate(conds):
    color = apm.styleDict[cond]['color']
    marker = apm.styleDict[cond]['marker']
    
    df_c = df_gC[df_gC[condCol] == cond]
    X, Y = df_c['bestH0'].values, df_c[YCol + '_wAvg'].values/1000
    
    ax.plot(X, Y,
            marker = marker, color = color, ls='',
            ms=9, alpha = 0.4, zorder=3)
    ax.errorbar(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3, 
                xerr=df_gD.loc[cond, 'H0_sem'], yerr=df_gD.loc[cond, 'E_sem']/1e3, 
                ls = '', marker = 'o', ms=1, color=color, 
                elinewidth = 3, capsize = 5, capthick = 3, zorder=5)
    ax.plot(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3,
            marker = marker, color = color, 
            ms=12, mec='k', label = apm.renameDict[cond], lw=0.5, zorder=6)

conds = ['blebbistatin & 50.0', 'Y27 & 50.0',
       'ck666 & 50.0', 'latrunculinA & 0.5', 'LIMKi & 20.0']
for i, cond in enumerate(conds):
    color = apm.styleDict[cond]['color']
    marker = apm.styleDict[cond]['marker']
    
    df_c = df_gC[df_gC[condCol] == cond]
    X, Y = df_c['bestH0'].values, df_c[YCol + '_wAvg'].values/1000
    
    ax.plot(X, Y,
            marker = marker, color = color, ls='',
            ms=8, mec='w', mew=0.5, zorder=4)
    ax.errorbar(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3, 
                xerr=df_gD.loc[cond, 'H0_sem'], yerr=df_gD.loc[cond, 'E_sem']/1e3, 
                ls = '', marker = 'o', ms=1, color=color, 
                elinewidth = 3, capsize = 5, capthick = 3, zorder=5)
    ax.plot(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3,
            marker = marker, color = color, 
            ms=12, mec='k', label = apm.renameDict[cond], lw=0.5, zorder=6)

ax.grid()
# ax.set_xlim([0, 600])
ax.set_xlim([50, 2000])
ax.set_xlabel('$H_0$ (nm)')
# ax.set_ylim([0, 30])
ax.set_ylim([0.2, 150])
ax.set_ylabel('$E_{500}$ (kPa)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 12)
fig.tight_layout()

plt.show()


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








# %%%% --------------------------