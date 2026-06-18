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
apm.setGraphicOptions(mode = 'print', 
                      palette = 'Set2', 
                      colorList = apm.cL_Set21)

figDir = 'C:/Users/josep/Desktop/Seafile/PapierDensité/FiguresMain'

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

MecaData_Phy = MecaData_Phy3


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


# %% --------------

# %% Main Fig 2

def prepTableForDrugPlot(df_f, XCol, YCol, condCol):
    logMean = lambda x : np.exp(np.mean(np.log(x)))
    logStd_Inf = lambda x : np.exp(np.mean(np.log(x)) - np.std(np.log(x)))
    logStd_Sup = lambda x : np.exp(np.mean(np.log(x)) + np.std(np.log(x)))
    # logSem_Inf = lambda x : np.exp(np.mean(np.log(x)) - (np.std(np.log(x))/len(x)**2))
    # logSem_Sup = lambda x : np.exp(np.mean(np.log(x)) + (np.std(np.log(x))/len(x)**2))
    
    aggFuns = (logMean, logStd_Inf, logStd_Sup)
    aggXCols1 = [XCol + f'_<lambda_{i:.0f}>' for i in range(len(aggFuns))]
    aggXCols2 = [XCol + f'_{s}' for s in ('mean', 'logStd_inf', 'logStd_sup',)] 
                                           # 'logSem_inf', 'logSem_sup')]
    aggYCols1 = [YCol + '_wAvg' + f'_<lambda_{i:.0f}>' for i in range(len(aggFuns))]
    aggYCols2 = [YCol + f'_{s}' for s in ('mean', 'logStd_inf', 'logStd_sup', )]
                                           # 'logSem_inf', 'logSem_sup')]
    
    # Group By Step 1
    df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
    df_fg = df_fg[[XCol]]
    df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                          valCol = YCol, weightCol = 'ciw'+YCol, weight_method = 'ciw^2')
    df_gC = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')
    
    # Group By Step 2
    df_fg_1 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [XCol],
                       aggFun = aggFuns) #.drop(columns=['cellID']).reset_index()
    df_fg_2 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [XCol],
                       aggFun = 'std') #.drop(columns=['cellID']).reset_index()
    df_fg_3 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [XCol],
                       aggFun = 'count') #.drop(columns=['cellID']).reset_index()
    
    df_fg_1.columns = ufun.flattenPandasIndex(df_fg_1.columns)
    df_fg_1 = df_fg_1[aggXCols1].rename(columns={s1:s2 for (s1, s2) \
                                                 in zip(aggXCols1, aggXCols2)})
    df_fg_2 = df_fg_2[[XCol]].rename(columns={XCol: XCol + "_std"})
    df_fg_3 = df_fg_3[[XCol]].rename(columns={XCol: "count"})
    
    
    df_fg2_1 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [YCol + '_wAvg'],
                       aggFun = aggFuns)
    df_fg2_2 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [YCol + '_wAvg'],
                       aggFun = 'std')
    
    df_fg2_1.columns = ufun.flattenPandasIndex(df_fg2_1.columns)
    df_fg2_1 = df_fg2_1[aggYCols1].rename(columns={s1:s2 for (s1, s2) \
                                                 in zip(aggYCols1, aggYCols2)})
    df_fg2_2 = df_fg2_2[[YCol + '_wAvg']].rename(columns={YCol + '_wAvg': YCol + "_std"})
    
    df_gD = pd.merge(left=df_fg_3, right=df_fg_1, on=condCol, how='inner')
    df_gD = pd.merge(left=df_gD, right=df_fg_2, on=condCol, how='inner')
    df_gD = pd.merge(left=df_gD, right=df_fg2_1, on=condCol, how='inner')
    df_gD = pd.merge(left=df_gD, right=df_fg2_2, on=condCol, how='inner')
    df_gD[XCol + '_sem'] = df_gD[XCol + '_std']/np.power(df_gD['count'], 0.5)
    df_gD[YCol + '_sem'] = df_gD[YCol + '_std']/np.power(df_gD['count'], 0.5)
    
    return(df_gC, df_gD)

# %%% F2B

# Save
SAVE = True
figSubDir = 'F2'
name = 'F2_B'

print('\n------\nF2_B - Drugs, by cells')

df = MecaData_Drug
df_ctrl = MecaData_Phy

drugs = ['dmso', 'blebbistatin', 'Y27', 'ck666', 'latrunculinA', 'LIMKi']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']

# XCol = 'ctFieldThickness'
XCol = 'H0_f_<_500'
YCol = 'E_f_<_500'

df, condCol = apm.makeCompositeCol(df, cols=['drug', 'concentration'])
df_ctrl, condCol = apm.makeCompositeCol(df_ctrl, cols=['drug', 'concentration'])

AllConds = ['dmso & 0.0', 'latrunculinA & 0.5', 'Y27 & 50.0', 'ck666 & 50.0', 'LIMKi & 20.0']

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df[condCol].apply(lambda x : x in AllConds)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           # (df['date'].apply(lambda x : x not in excluded_dates)),
           (df[XCol] < 1100),
           (df['normal field'] == 5),
           (df[YCol] <= 1e5),
           (df['valid' + YCol[1:]] == True), 
           ]

Filters_ctrl = [(df_ctrl['validatedThickness'] == True), 
               (df_ctrl['substrate'] == substrate),
               (df_ctrl[condCol].apply(lambda x : x in AllConds)),
               (df_ctrl['cell subtype'].apply(lambda x : x in subtypes)),
               (df_ctrl['date'].apply(lambda x : x not in excluded_dates)),
               (df_ctrl[XCol] < 1100),
               (df_ctrl['normal field'] == 5),
               (df_ctrl[YCol] <= 1e5),
               (df_ctrl['valid' + YCol[1:]] == True), 
               ]

df_f = apm.filterDf(df, Filters)
df_ctrl_f = apm.filterDf(df_ctrl, Filters_ctrl)

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
CountByCond_ctrl, CountByCell_ctrl = apm.makeCountDf(df_ctrl_f, condCol)

#### Plot

sD = apm.styleDict_V2
rD = apm.renameDict

# fig, ax = plt.subplots(1, 1, figsize=(9/cm_in, 8.5/cm_in))
fig, ax = plt.subplots(1, 1, figsize=(8/cm_in, 12/cm_in))
ax.set_xscale('log')
ax.set_yscale('log')

ms_small = 5
ms_big = 7

#### Controls

conds = ['dmso & 0.0']
df_gC, df_gD = prepTableForDrugPlot(df_ctrl_f, XCol, YCol, condCol)

for i, cond in enumerate(conds):
    color = sD[cond]['color']
    marker = sD[cond]['marker']
    
    df_c = df_gC[df_gC[condCol] == cond]
    X, Y = df_c[XCol].values, df_c[YCol + '_wAvg'].values/1000
    
    ax.plot(X, Y,
            marker = marker, color = apm.lightenColor(color, factor=1.0), ls='',
            ms=ms_small, alpha = 0.5, zorder=3, mew=0)
    
    ax.errorbar(df_gD.loc[cond, XCol + '_mean'], df_gD.loc[cond, YCol + '_mean']/1e3, 
                xerr=df_gD.loc[cond, XCol + '_sem'], yerr=df_gD.loc[cond, YCol + '_sem']/1e3, 
                ls = '', marker = 'o', ms=1, color=apm.lightenColor(color, factor=0.6), 
                elinewidth = 1.0, capsize = 2.0, capthick = 1.0, zorder=5)
    ax.plot(df_gD.loc[cond, XCol + '_mean'], df_gD.loc[cond, YCol + '_mean']/1e3,
            marker = marker, color = apm.lightenColor(color, factor=0.9), ls='',
            ms=ms_big, mec='k', label = rD[cond], mew=0.5, zorder=6)
    
    # XERR = [df_gD.loc[cond, XCol + '_mean'] - df_gD.loc[cond, XCol + '_logStd_inf'], 
    #         df_gD.loc[cond, XCol + '_logStd_sup'] - df_gD.loc[cond, XCol + '_mean']]
    # YERR = [df_gD.loc[cond, YCol + '_mean'] - df_gD.loc[cond, YCol + '_logStd_inf'], 
    #         df_gD.loc[cond, YCol + '_logStd_sup'] - df_gD.loc[cond, YCol + '_mean']]
    # XERR, YERR = np.array([XERR]).T, (np.array([YERR]).T)/1e3
    # ax.errorbar(df_gD.loc[cond, XCol + '_mean'], 
    #             df_gD.loc[cond, YCol + '_mean']/1e3, 
    #             xerr = XERR, yerr = YERR, 
    #             color=apm.lightenColor(color, factor=0.6), 
    #             ls = '', marker = 'o', ms=1, 
    #             elinewidth = 1.0, capsize = 2.0, capthick = 1.0, 
    #             zorder=5)
    # ax.plot(df_gD.loc[cond, XCol + '_mean'], df_gD.loc[cond, YCol + '_mean']/1e3,
    #         marker = marker, color = apm.lightenColor(color, factor=0.9), ls='',
    #         ms=ms_big, mec='k', label = rD[cond], mew=0.5, zorder=6)
    
    hM, hL, hH = ufun.getLogNDistributionDescriptors(X)
    EM, EL, EH = ufun.getLogNDistributionDescriptors(Y)
    print(f"\n F2_B Drug {cond} - by cells")
    print(f"n = {CountByCond_ctrl.loc[cond, 'compCount']:.0f}, " + \
          f"N = {CountByCond_ctrl.loc[cond, 'cellCount']:.0f}, " + \
          f"M = {CountByCond_ctrl.loc[cond, 'manipsCount']:.0f}")
    print(f'For {XCol} vs {YCol}')
    print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
    print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')

#### Drugs

conds = ['latrunculinA & 0.5', 'Y27 & 50.0', 'ck666 & 50.0', 'LIMKi & 20.0']
df_gC, df_gD = prepTableForDrugPlot(df_f, XCol, YCol, condCol)

for i, cond in enumerate(conds):
    color = sD[cond]['color']
    marker = sD[cond]['marker']
    
    df_c = df_gC[df_gC[condCol] == cond]
    X, Y = df_c[XCol].values, df_c[YCol + '_wAvg'].values/1000
    
    ax.plot(X, Y,
            marker = marker, color = apm.lightenColor(color, factor=1.2), ls='',
            ms=ms_small, mew=0.25, zorder=4, mec='w') 
    
    ax.errorbar(df_gD.loc[cond, XCol + '_mean'], df_gD.loc[cond, YCol + '_mean']/1e3, 
                xerr=df_gD.loc[cond, XCol + '_sem'], yerr=df_gD.loc[cond, YCol + '_sem']/1e3, 
                ls = '', marker = 'o', ms=1, color=apm.lightenColor(color, 0.6), 
                elinewidth = 1.0, capsize = 2.0, capthick = 1.0, zorder=5)
    ax.plot(df_gD.loc[cond, XCol + '_mean'], df_gD.loc[cond, YCol + '_mean']/1e3,
            marker = marker, color = apm.lightenColor(color, 0.9), ls='',
            ms=ms_big, mec='k', label = rD[cond], mew=0.5, zorder=6)
    
    # XERR = [df_gD.loc[cond, XCol + '_mean'] - df_gD.loc[cond, XCol + '_logStd_inf'], 
    #         df_gD.loc[cond, XCol + '_logStd_sup'] - df_gD.loc[cond, XCol + '_mean']]
    # YERR = [df_gD.loc[cond, YCol + '_mean'] - df_gD.loc[cond, YCol + '_logStd_inf'], 
    #         df_gD.loc[cond, YCol + '_logStd_sup'] - df_gD.loc[cond, YCol + '_mean']]
    # XERR, YERR = np.array([XERR]).T, (np.array([YERR]).T)/1e3
    # ax.errorbar(df_gD.loc[cond, XCol + '_mean'], 
    #             df_gD.loc[cond, YCol + '_mean']/1e3, 
    #             xerr = XERR, yerr = YERR, 
    #             color=apm.lightenColor(color, factor=0.6), 
    #             ls = '', marker = 'o', ms=1, 
    #             elinewidth = 1.0, capsize = 2.0, capthick = 1.0, 
    #             zorder=5)
    # ax.plot(df_gD.loc[cond, XCol + '_mean'], df_gD.loc[cond, YCol + '_mean']/1e3,
    #         marker = marker, color = apm.lightenColor(color, 0.9), ls='',
    #         ms=ms_big, mec='k', label = rD[cond], mew=0.5, zorder=6)
    
    hM, hL, hH = ufun.getLogNDistributionDescriptors(X)
    EM, EL, EH = ufun.getLogNDistributionDescriptors(Y)
    print(f'\nDrug {cond} - by cells')
    print(f"n = {CountByCond.loc[cond, 'compCount']:.0f}, " + \
          f"N = {CountByCond.loc[cond, 'cellCount']:.0f}, " + \
          f"M = {CountByCond.loc[cond, 'manipsCount']:.0f}")
    print(f'For {XCol} vs {YCol}')
    print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
    print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
    

ax.grid()
ax.set_title('Drug treatments', weight='bold')

ax.set_xlim([40, 2000])
ax.set_ylim([0.5, 200])
ax.set_xlabel('$H_0$ (nm)')
ax.set_ylabel('$E$ (kPa)')
ax.legend(loc='upper right') # loc='center left', bbox_to_anchor=(1, 0.5), 
fig.tight_layout()
print('\n---------------')

plt.show()


# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')
    CountByCond_ctrl.to_csv(os.path.join(figDir, figSubDir, name+'_ctrl_count.txt'), sep='\t')


    
# %%% F2C
 
# Save
SAVE = True
figSubDir = 'F2'
name = 'F2_C'

print('\n------\nF2_C - Drugs by compressions')

df = MecaData_Drug
df_ctrl = MecaData_Phy

drugs = ['dmso', 'blebbistatin', 'none', 
         'Y27', 'ck666', 'latrunculinA', 'LIMKi']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
# excluded_dates = ['23-03-08', '23-02-23', '23-11-26']

# XCol = 'ctFieldThickness'
XCol = 'H0_f_<_500'
YCol = 'E_f_<_500'

df, condCol = apm.makeCompositeCol(df, cols=['drug', 'concentration'])
df_ctrl, condCol = apm.makeCompositeCol(df_ctrl, cols=['drug', 'concentration'])

concentrations = ['dmso & 0.0', 'latrunculinA & 0.5', 
         'Y27 & 50.0', 'ck666 & 50.0', 'LIMKi & 20.0']

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['drug'].apply(lambda x : x in drugs)),
           (df[condCol].apply(lambda x : x in concentrations)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           # (df['date'].apply(lambda x : x not in excluded_dates)),
           # (df[XCol] > 50),
           (df[XCol] < 1000),
           (df['normal field'] == 5),
           (df[YCol] <= 1e5),
           (df['valid' + YCol[1:]] == True), 
           ]

Filters_ctrl = [(df_ctrl['validatedThickness'] == True), 
               (df_ctrl['substrate'] == substrate),
               (df_ctrl['drug'].apply(lambda x : x in drugs)),
               (df_ctrl['cell subtype'].apply(lambda x : x in subtypes)),
               (df_ctrl['date'].apply(lambda x : x not in excluded_dates)),
               (df_ctrl[XCol] < 1000),
               (df_ctrl['normal field'] == 5),
               (df_ctrl[YCol] <= 1e5),
               (df_ctrl['valid' + YCol[1:]] == True), 
               ]

df_f = apm.filterDf(df, Filters)
df_ctrl_f = apm.filterDf(df_ctrl, Filters_ctrl)

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
CountByCond_ctrl, CountByCell_ctrl = apm.makeCountDf(df_ctrl_f, condCol)

#### Plot

sD = apm.styleDict_V2
rD = apm.renameDict

fig1, axes1 = plt.subplots(1, 2, figsize = (0.5*17/cm_in, 6/cm_in), 
                         sharex=True, sharey=True, layout='constrained')
fig2, axes2 = plt.subplots(1, 2, figsize = (0.5*17/cm_in, 6/cm_in), 
                         sharex=True, sharey=True, layout='constrained')
axes_f = np.concatenate([axes1, axes2])
dLabels = {}


#### Controls

conds = ['dmso & 0.0']
df_gC, df_gD = prepTableForDrugPlot(df_ctrl_f, XCol, YCol, condCol)

for i, cond in enumerate(conds):
    for j in range(len(axes_f)):
        ax = axes_f[j]
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        color = sD[cond]['color']
        marker = sD[cond]['marker']
        
        # df_c = df_gC[df_gC[condCol] == cond]
        # X, Y = df_c[XCol].values, df_c[YCol + '_wAvg'].values/1000
        
        df_c = df_ctrl_f[df_ctrl_f[condCol] == cond]
        X, Y = df_c[XCol].values, df_c[YCol].values/1000
        
        # ax.plot(X, Y,
        #         marker = marker, color = color, ls='',
        #         ms=3, alpha = 0.1, zorder=3, label = rD[cond])
        
        Xfit, Yfit = np.log(X), np.log(Y)
        wd=1/(np.std(Xfit)) # **2
        we=1/(np.std(Yfit)) # **2

        [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
        A, k = np.exp(b), a
        pval = results.pval
        [k_ciw, b_ciw] = results.params_ciw
        
        Xplot = np.exp(np.linspace(1, 1e2, 50))
        Yplot = A * Xplot**k
        text_pval = apm.pval2text(pval, n_digits = 3, space = True)
        # dLabels[cond] = {'A':A, 'k':k, 'k_ciw':k_ciw, 'pval':pval, 'text_pval':text_pval}
        if pval < 0.05:
            dLabels[cond] = f'{rD[cond]}' + \
                    f'\nk  = {k:.2f}  ' + \
                    r'$\pm$' + f' {(k_ciw/2):.2f}' + \
                    '\n' + text_pval
        else:
            dLabels[cond] = f'{rD[cond]}' + \
                            '\n' + 'NS fit' + \
                            '\n' + text_pval
        ax.plot(Xplot, Yplot, ls = '-', c = color, lw = 2,)
        
        if j==0:
            hM, hL, hH = ufun.getLogNDistributionDescriptors(df_c[XCol].values)
            EM, EL, EH = ufun.getLogNDistributionDescriptors(df_c[YCol].values/1000)
            A_Low = np.exp(b-(b_ciw/2))
            A_High = np.exp(b+(b_ciw/2))
            print(f'{cond} - by compressions')
            print(f"n = {CountByCond_ctrl.loc[cond, 'compCount']:.0f}, " + \
                  f"N = {CountByCond_ctrl.loc[cond, 'cellCount']:.0f}, " + \
                  f"M = {CountByCond_ctrl.loc[cond, 'manipsCount']:.0f}")
            print(f'For {XCol} vs {YCol}')
            print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
            print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
            print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
            print(f'Power-law constant & Ci : {A:.2e} [{A_Low:.2e}-{A_High:.2e}]')
            print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')
            

#### Drugs

conds = concentrations[1:]
df_gC, df_gD = prepTableForDrugPlot(df_f, XCol, YCol, condCol)

for i, cond in enumerate(conds):
    ax = axes_f[i]
    
    color = sD[cond]['color']
    marker = sD[cond]['marker']
    
    # Par cellule juste pour la p-value
    df_c = df_gC[df_gC[condCol] == cond]
    X, Y = df_c[XCol].values, df_c[YCol + '_wAvg'].values/1000
    Xfit, Yfit = np.log(X), np.log(Y)
    wd=1/(np.std(Xfit)) # **2
    we=1/(np.std(Yfit)) # **2
    [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
    A, k = np.exp(b), a
    pval = results.pval
    [k_ciw, b_ciw] = results.params_ciw
    Xplot = np.exp(np.linspace(1, 1e2, 50))
    Yplot = A * Xplot**k
    text_pval = apm.pval2text(pval, n_digits = 3, space = True)
    # print(cond, pval)
    
    
    # Par compressions
    df_c = df_f[df_f[condCol] == cond]
    X, Y = df_c[XCol].values, df_c[YCol].values/1000
    
    ax.plot(X, Y,
            marker = marker, color = color, ls='',
            ms=5, mec='w', mew=0.5, zorder=4, alpha=0.75) #, label = rD[cond])
    
    Xfit, Yfit = np.log(X), np.log(Y)
    wd=1/(np.std(Xfit)) # **2
    we=1/(np.std(Yfit)) # **2

    [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
    A, k = np.exp(b), a
    # pval = results.pval
    [k_ciw, b_ciw] = results.params_ciw
    Xplot = np.exp(np.linspace(1, 1e2, 50))
    Yplot = A * Xplot**k
    
    ax.set_title(rD[cond], color = color, weight = 'bold')
    # text_pval = apm.pval2text(pval, n_digits = 3, space = True)
    # dLabels[cond] = {'A':A, 'k':k, 'k_ciw':k_ciw, 'pval':pval, 'text_pval':text_pval}
    if pval < 0.05:
        dLabels[cond] = f'{rD[cond]}' + \
                f'\nk  = {k:.2f}  ' + \
                r'$\pm$' + f' {(k_ciw/2):.2f}' + \
                '\n' + text_pval
        ax.plot(Xplot, Yplot, ls = '-', c = apm.lightenColor(color, 0.8), 
                lw = 2, zorder=6)
    else:
        dLabels[cond] = f'{rD[cond]}' + \
                        '\n' + 'Fit not significant' + \
                        '\n' + text_pval
    
    hM, hL, hH = ufun.getLogNDistributionDescriptors(df_c[XCol].values)
    EM, EL, EH = ufun.getLogNDistributionDescriptors(df_c[YCol].values/1000)
    A_Low = np.exp(b-(b_ciw/2))
    A_High = np.exp(b+(b_ciw/2))
    print(f'Drug {cond} - by compressions')
    print(f"n = {CountByCond.loc[cond, 'compCount']:.0f}, " + \
          f"N = {CountByCond.loc[cond, 'cellCount']:.0f}, " + \
          f"M = {CountByCond.loc[cond, 'manipsCount']:.0f}")
    print(f'For {XCol} vs {YCol}')
    print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
    print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
    print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
    print(f'Power-law constant & Ci : {A:.2e} [{A_Low:.2e}-{A_High:.2e}]')
    print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')
    

#### Legend

# =============================================================================
# conds = ['dmso & 0.0', 'Y27 & 50.0', 'ck666 & 50.0', 'latrunculinA & 0.5', 'LIMKi & 20.0']
# CommonLegendMarks = []
# for i, cond in enumerate(conds):
#     ax = axes_f[1]
#     color = sD[cond]['color']
#     
#     label = dLabels[cond]
#     LegendMark = matplotlib.lines.Line2D([], [], color=color, ls='--', lw = 2.5,
#                                          label = label)
#     k, k_ciw  = dLabels[cond]['k'], dLabels[cond]['k_ciw'], 
#     pval, text_pval = dLabels[cond]['pval'], dLabels[cond]['text_pval']
#     if pval < 0.05:
#         LegendMark = matplotlib.lines.Line2D([], [], color=color, ls='--', lw = 2.5,
#                                              label = f'{rD[cond]}' + \
#                                                      f'\nk  = {k:.2f}  ' + \
#                                                      r'$\pm$' + f' {(k_ciw/2):.2f}' + \
#                                                      '\n' + text_pval)
#     else:
#         LegendMark = matplotlib.lines.Line2D([], [], color=color, ls='--', lw = 2.5,
#                                              label = label)
#         dLabels[cond] = f'{rD[cond]}' + \
#                         '\n' + 'NS fit' + \
#                         '\n' + text_pval
#     CommonLegendMarks.append(LegendMark)
# =============================================================================
    
    
#### Format
    
    
for k in range(len(axes_f)):
    ax = axes_f[k]
    ax.grid()
    # ax.set_xlim([0, 600])
    ax.set_xlim([50, 2000])
    # ax.set_ylim([0, 30])
    ax.set_ylim([0.2, 150])
    # if k//2==1:
    ax.set_xlabel('$H_0$ (nm)')
    if k%2==0:
        ax.set_ylabel('$E$ (kPa)')


plt.show()
print('\n---------------')


if SAVE:
    ufun.archiveFig(fig1, name = name+'_1', ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig1, name = name+'_1', ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig2, name = name+'_2', ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig2, name = name+'_2', ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')
    CountByCond_ctrl.to_csv(os.path.join(figDir, figSubDir, name+'_ctrl_count.txt'), sep='\t')




# %% Main Fig 4


# %%% F4B

# Save
SAVE = True
figSubDir = 'F4'
name = 'F4_B'

print('\n------\nF4_B - Cell types, by compressions')

df = MecaData_CellTypes
df_ctrl = MecaData_Phy

df, condCol = apm.makeCompositeCol(df, cols=['cell type', 'cell subtype'])
df_ctrl, condCol = apm.makeCompositeCol(df_ctrl, cols=['cell type', 'cell subtype'])

XCol = 'H0_f_<_500'
YCol = 'E_f_<_500'
ctrl_cond = '3T3 & Atcc-2023' 

# Define
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
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

Filters_ctrl = [(df_ctrl['validatedThickness'] == True), 
               (df_ctrl['substrate'] == '20um fibronectin discs'),
               (df_ctrl['drug'].apply(lambda x : x in ['dmso'])),
               (df_ctrl['cell subtype'].apply(lambda x : x in ['Atcc-2023', 'Atcc-2023-LaGFP'])),
               (df_ctrl['date'].apply(lambda x : x not in excluded_dates)),
               (df_ctrl[XCol] < 1000),
               (df_ctrl['normal field'] == 5),
               (df_ctrl[YCol] <= 1e5),
               (df_ctrl['valid' + YCol[1:]] == True), 
               ]

df_f = apm.filterDf(df, Filters)
df_ctrl_f = apm.filterDf(df_ctrl, Filters_ctrl)

df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'
df_ctrl_f.loc[df_ctrl_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'


# Filter 2
Case_A1 = (df_f['cell type'].apply(lambda x : x in ['HoxB8-Macro']))
Case_A2 = (df_f['substrate'] == 'bare glass')
Case_B1 = (df_f['cell type'].apply(lambda x : x in ['DC', 'Dicty']))
Case_B2 = (df_f['substrate'] == 'BSA coated glass')
Case_C1 = (df_f['cell type'].apply(lambda x : x in ['3T3', 'MDCK', 'HeLa']))
Case_C2 = (df_f['substrate'] == '20um fibronectin discs')
Case_D1 = (df_f['cell type'].apply(lambda x : x in ['3T3', 'MDCK', 'DC']))
Case_D2 = (df_f['normal field'] == 5)
Filters = [((Case_A1 & Case_A2) | (Case_B1 & Case_B2) | (Case_C1 & Case_C2)),
           (Case_D1 | Case_D2),
           ]

df_f = apm.filterDf(df_f, Filters)


# Count
CountByCond_ctrl, CountByCell_ctrl = apm.makeCountDf(df_ctrl_f, condCol)
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)

# Order
co_order = [
            # '3T3 & Atcc-2023', 
            'HeLa & fucci', 
            'MDCK & WT',
            'DC & mouse-primary', 
            'Dicty & DictyBase-WT', 
            ]

colorsD = {
          '3T3 & Atcc-2023'     : 'dimgray', 
          'HeLa & fucci'         : apm.cL_Set2[3], 
          'MDCK & WT'            : apm.cL_Set2[5],
          'DC & mouse-primary'   : apm.cL_Set2[2],  
          'Dicty & DictyBase-WT' : apm.cL_Set2[4],            
          }

rD = {
      '3T3 & Atcc-2023'      :  '3T3 ATCC', 
      'HeLa & fucci'         :  'HeLa FUCCI',  
      'DC & mouse-primary'   :  'Primary DC',  
      'Dicty & DictyBase-WT' :  'Dictys Ax3',  
      'MDCK & WT'            :  'MDCK',
      }



#### Plot
fig, axes = plt.subplots(2, 2, figsize=(8/cm_in, 12/cm_in), 
                         sharex=True, sharey=True, layout='compressed')
axes = axes.flatten('C')
dLabels = {}

# Per comp
df_plot = df_f
df_ctrl_plot = df_ctrl_f
# YCol += '_wAvg'
df_plot[YCol] /= 1000
df_ctrl_plot[YCol] /= 1000


# Fit for the controls
Xctrl, Yctrl = df_ctrl_plot[XCol].values, df_ctrl_plot[YCol].values
Xctrl_fit, Yctrl_fit = np.log(Xctrl), np.log(Yctrl)
wd=1/(np.std(Xctrl_fit)) # **2
we=1/(np.std(Yctrl_fit)) # **2
[a, b], results = ufun.fitLineTLS(Xctrl_fit, Yctrl_fit, wd=wd, we=we)
A, k = np.exp(b), a
pval = results.pval
[k_ciw, b_ciw] = results.params_ciw
Xctrl_plot = np.exp(np.linspace(1, 9, 50))
Yctrl_plot = A * Xctrl_plot**k
text_pval = apm.pval2text(pval, n_digits = 3, space = True)

dLabels[ctrl_cond] = f'{rD[ctrl_cond]}' + \
               '\n' + f'k  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + \
               '\n' + text_pval



for i in range(len(axes)):
    ax = axes[i]
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    #### Controls
    ax.plot(Xctrl_plot, Yctrl_plot, ls = '-', color = 'dimgray', 
            lw = 2.0, zorder = 6, alpha = 0.6)
            
    if i == 0:
        hM, hL, hH = ufun.getLogNDistributionDescriptors(df_ctrl_plot[XCol].values)
        EM, EL, EH = ufun.getLogNDistributionDescriptors(df_ctrl_plot[YCol].values)
        A_Low = np.exp(b-(b_ciw/2))
        A_High = np.exp(b+(b_ciw/2))
        
        print('Cell type 3T3 ATCC')
        print(f"n = {CountByCond_ctrl.loc['3T3 & Atcc-2023', 'compCount']:.0f}, " + \
              f"N = {CountByCond_ctrl.loc['3T3 & Atcc-2023', 'cellCount']:.0f}, " + \
              f"M = {CountByCond_ctrl.loc['3T3 & Atcc-2023', 'manipsCount']:.0f}")
        print(f'For {XCol} vs {YCol}')
        print(f'For {XCol} vs {YCol}')
        print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
        print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
        print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
        print(f'Power-law constant & Ci : {A:.2e} [{A_Low:.2e}-{A_High:.2e}]')
        print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')
        

    #### Cell type i
    df_fc = df_plot[df_plot[condCol] == co_order[i]]
    color = colorsD[co_order[i]]

    medianX = np.median(df_fc[XCol].values)
    medianY = np.median(df_fc[YCol].values)
    
    X=df_fc[XCol].values
    Y=df_fc[YCol].values
    
    marker = 'o'
    alpha = 0.7
    ms = 4.5
    
    ax.plot(X, Y,
            marker = marker, color = color, ls='',
            ms=ms, mec='w', mew=0.5, zorder=6, alpha=alpha)
    
    Xfit, Yfit = np.log(df_fc[XCol].values), np.log(df_fc[YCol].values)
    wd=1/(np.std(Xfit)) # **2
    we=1/(np.std(Yfit)) # **2

    [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
    A, k = np.exp(b), a
    pval = results.pval
    [k_ciw, b_ciw] = results.params_ciw
    
    Xplot = np.exp(np.linspace(1, 9, 50))
    Yplot = A * Xplot**k
    text_pval = apm.pval2text(pval, n_digits = 3, space = True)
    dLabels[co_order[i]] = f'{rD[co_order[i]]}' + \
                   '\n' + f'k  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + \
                   '\n' + text_pval
        
    ax.plot(Xplot, Yplot, ls = '-', c = apm.lightenColor(color, 0.75), 
            lw = 2.0, zorder = 7,)

    ax.set_xlabel('$H_{0}$ (nm)\n', labelpad=0.5)
    ax.set_ylabel('$E$ (kPa)', labelpad=0.5)
    # ax.set_title(co_order[i])
    if i%2 != 0:
        ax.set_ylabel('')
    if i//2 == 0:
        ax.set_xlabel('')
    
    ax.text(300, 220, f'{rD[co_order[i]]}', va='center', ha='left',  weight = 'bold',
            color=colorsD[co_order[i]], fontsize=7.0, zorder=3)
    ax.text(300, 150, f'{rD[ctrl_cond]}', va='center', ha='left', weight = 'bold',
            color=colorsD[ctrl_cond], fontsize=7.0, zorder=3)
    ax.add_patch(plt.Rectangle((270, 110), 1600, 180, 
                               fc="white", zorder=2, alpha=0.85))
    
    hM, hL, hH = ufun.getLogNDistributionDescriptors(df_fc[XCol].values)
    EM, EL, EH = ufun.getLogNDistributionDescriptors(df_fc[YCol].values)
    A_Low = np.exp(b-(b_ciw/2))
    A_High = np.exp(b+(b_ciw/2))
    
    print(f'Cell type {co_order[i]}')
    print(f"n = {CountByCond.loc[co_order[i], 'compCount']:.0f}, " + \
          f"N = {CountByCond.loc[co_order[i], 'cellCount']:.0f}, " + \
          f"M = {CountByCond.loc[co_order[i], 'manipsCount']:.0f}")
    print(f'For {XCol} vs {YCol}')
    print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
    print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
    print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
    print(f'Power-law constant & Ci : {A:.2e} [{A_Low:.2e}-{A_High:.2e}]')
    print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')
    
#### Format
rD.update({'E_eff_wAvg':'E_{eff} (kPa)'})

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder=0)
    apm.renameAxes(ax, rD, format_xticks = False)
    ax.set_xlim(50, 2000)
    ax.set_ylim(0.4, 300)
    
fig.get_layout_engine().set(w_pad=2e-2, h_pad=2e-2, 
                            hspace=5e-3, wspace=3e-3)

# Show
plt.show()

print('\n---------------')

# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')





# %% --------------

















# %% Supp Fig 2

# %%% Fig S2B - Split with fits - By cells
 
# Save
SAVE = True
figSubDir = 'S2'
name = 'E500_vs_h0_drugs_wFit_split_byCell'

df = MecaData_Drug
df_ctrl = MecaData_Phy

drugs = ['dmso', 'blebbistatin', 'none', 'Y27', 'ck666', 'latrunculinA', 'LIMKi']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']

# XCol = 'ctFieldThickness'
XCol = 'H0_f_<_500'
YCol = 'E_f_<_500'

df, condCol = apm.makeCompositeCol(df, cols=['drug', 'concentration'])
df_ctrl, condCol = apm.makeCompositeCol(df_ctrl, cols=['drug', 'concentration'])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['date'].apply(lambda x : x not in excluded_dates)),
           # (df[XCol] > 50),
           (df[XCol] < 1000),
           (df['normal field'] == 5),
           (df[YCol] <= 1e5),
           (df['valid' + YCol[1:]] == True), 
           ]

Filters_ctrl = [(df_ctrl['validatedThickness'] == True), 
               (df_ctrl['substrate'] == substrate),
               (df_ctrl['drug'].apply(lambda x : x in drugs)),
               (df_ctrl['cell subtype'].apply(lambda x : x in subtypes)),
               (df_ctrl['date'].apply(lambda x : x not in excluded_dates)),
               (df_ctrl[XCol] < 1000),
               (df_ctrl['normal field'] == 5),
               (df_ctrl[YCol] <= 1e5),
               (df_ctrl['valid' + YCol[1:]] == True), 
               ]

df_f = apm.filterDf(df, Filters)
df_ctrl_f = apm.filterDf(df_ctrl, Filters_ctrl)


def prepTableForDrugPlot(df_f, XCol, YCol, condCol):
    logMean = lambda x : np.exp(np.mean(np.log(x)))

    # Group By Step 1
    df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
    df_fg = df_fg[[XCol]]
    df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                          valCol = YCol, weightCol = 'ciw'+YCol, weight_method = 'ciw^2')
    df_gC = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')
    
    # Group By Step 2
    df_fg_1 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [XCol],
                       aggFun = logMean) #.drop(columns=['cellID']).reset_index()
    df_fg_2 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [XCol],
                       aggFun = 'std') #.drop(columns=['cellID']).reset_index()
    df_fg_3 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [XCol],
                       aggFun = 'count') #.drop(columns=['cellID']).reset_index()
    df_fg_1 = df_fg_1[[XCol]].rename(columns={XCol: "H0_mean"})
    df_fg_2 = df_fg_2[[XCol]].rename(columns={XCol: "H0_std"})
    df_fg_3 = df_fg_3[[XCol]].rename(columns={XCol: "count"})
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
    
    return(df_gC, df_gD)

#### Plot

sD = apm.styleDict_V2
rD = apm.renameDict

fig, axes = plt.subplots(2, 2, figsize = (12/cm_in, 8/cm_in), 
                         sharex=True, sharey=True, layout='constrained')
axes_f = axes.flatten()
dLabels = {}


#### Controls

conds = ['dmso & 0.0']
df_gC, df_gD = prepTableForDrugPlot(df_ctrl_f, XCol, YCol, condCol)

for i, cond in enumerate(conds):
    for k in range(len(axes_f)):
        ax = axes_f[k]
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        color = sD[cond]['color']
        marker = sD[cond]['marker']
        
        df_c = df_gC[df_gC[condCol] == cond]
        X, Y = df_c[XCol].values, df_c[YCol + '_wAvg'].values/1000
        
        ax.plot(X, Y,
                marker = marker, color = color, ls='',
                ms=6, alpha = 0.3, zorder=3, label = rD[cond])
        
        Xfit, Yfit = np.log(X), np.log(Y)
        wd=1/(np.std(Xfit)) # **2
        we=1/(np.std(Yfit)) # **2

        [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
        A, k = np.exp(b), a
        pval = results.pval
        [k_ciw, b_ciw] = results.params_ciw
        Xplot = np.exp(np.linspace(1, 1e2, 50))
        Yplot = A * Xplot**k
        text_pval = apm.pval2text(pval, n_digits = 3, space = True)
        # dLabels[cond] = {'A':A, 'k':k, 'k_ciw':k_ciw, 'pval':pval, 'text_pval':text_pval}
        if pval < 0.05:
            dLabels[cond] = f'{rD[cond]}' + \
                    f'\nk  = {k:.2f}  ' + \
                    r'$\pm$' + f' {(k_ciw/2):.2f}' + \
                    '\n' + text_pval
        else:
            dLabels[cond] = f'{rD[cond]}' + \
                            '\n' + 'NS fit' + \
                            '\n' + text_pval
        ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 1.5,)
            

#### Drugs

conds = ['Y27 & 50.0', 'ck666 & 50.0', 'latrunculinA & 0.5', 'LIMKi & 20.0']
df_gC, df_gD = prepTableForDrugPlot(df_f, XCol, YCol, condCol)

for i, cond in enumerate(conds):
    ax = axes_f[i]
    
    color = sD[cond]['color']
    marker = sD[cond]['marker']
    
    df_c = df_gC[df_gC[condCol] == cond]
    X, Y = df_c[XCol].values, df_c[YCol + '_wAvg'].values/1000
    
    ax.plot(X, Y,
            marker = marker, color = color, ls='',
            ms=6, mec='w', mew=0.5, zorder=4, label = rD[cond])
    
    Xfit, Yfit = np.log(X), np.log(Y)
    wd=1/(np.std(Xfit)) # **2
    we=1/(np.std(Yfit)) # **2

    [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
    A, k = np.exp(b), a
    pval = results.pval
    [k_ciw, b_ciw] = results.params_ciw
    Xplot = np.exp(np.linspace(1, 1e2, 50))
    Yplot = A * Xplot**k
    text_pval = apm.pval2text(pval, n_digits = 3, space = True)
    # dLabels[cond] = {'A':A, 'k':k, 'k_ciw':k_ciw, 'pval':pval, 'text_pval':text_pval}
    if pval < 0.05:
        dLabels[cond] = f'{rD[cond]}' + \
                f'\nk  = {k:.2f}  ' + \
                r'$\pm$' + f' {(k_ciw/2):.2f}' + \
                '\n' + text_pval
        ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 1.5,)
        
    else:
        dLabels[cond] = f'{rD[cond]}' + \
                        '\n' + 'Fit not significant' + \
                        '\n' + text_pval
    
    

#### Legend

conds = ['dmso & 0.0', 'Y27 & 50.0', 'ck666 & 50.0', 'latrunculinA & 0.5', 'LIMKi & 20.0']
CommonLegendMarks = []
for i, cond in enumerate(conds):
    ax = axes_f[1]
    color = sD[cond]['color']
    
    label = dLabels[cond]
    LegendMark = matplotlib.lines.Line2D([], [], color=color, ls='--', lw = 2.5,
                                         label = label)
        
    # k, k_ciw  = dLabels[cond]['k'], dLabels[cond]['k_ciw'], 
    # pval, text_pval = dLabels[cond]['pval'], dLabels[cond]['text_pval']
    # if pval < 0.05:
    #     LegendMark = matplotlib.lines.Line2D([], [], color=color, ls='--', lw = 2.5,
    #                                          label = f'{rD[cond]}' + \
    #                                                  f'\nk  = {k:.2f}  ' + \
    #                                                  r'$\pm$' + f' {(k_ciw/2):.2f}' + \
    #                                                  '\n' + text_pval)
    # else:
    #     LegendMark = matplotlib.lines.Line2D([], [], color=color, ls='--', lw = 2.5,
    #                                          label = label)
    #     dLabels[cond] = f'{rD[cond]}' + \
    #                     '\n' + 'NS fit' + \
    #                     '\n' + text_pval
    
    
    CommonLegendMarks.append(LegendMark)
    
plt.figlegend(handles=CommonLegendMarks, fontsize = 8, 
              loc='outside right upper',
              title = 'Power-law fits', title_fontproperties = {'weight':'bold'},
              labelspacing = 1.15, handletextpad=0.4, handlelength = 1)
    
#### Format
    
    
for k in range(len(axes_f)):
    ax = axes_f[k]
    ax.grid()
    # ax.set_xlim([0, 600])
    ax.set_xlim([50, 2000])
    # ax.set_ylim([0, 30])
    ax.set_ylim([0.2, 150])
    if k//2==1:
        ax.set_xlabel('$H_{500}$ (nm)')
    if k%2==0:
        ax.set_ylabel('$E_{500}$ (kPa)')
    ax.legend(handletextpad=0.2) # loc='center left', bbox_to_anchor=(1, 0.5), 
    
# fig.tight_layout()

plt.show()

if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    # CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%% Alternate Fig S2B - HORIZONTAL Split with fits - By cells
 
# Save
SAVE = True
figSubDir = 'S2'
name = 'E500_vs_h0_drugs_wFit_split_byCell_HORIZ'

df = MecaData_Drug
df_ctrl = MecaData_Phy

drugs = ['dmso', 'blebbistatin', 'none', 'Y27', 'ck666', 'latrunculinA', 'LIMKi']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']

# XCol = 'ctFieldThickness'
XCol = 'H0_f_<_500'
YCol = 'E_f_<_500'

df, condCol = apm.makeCompositeCol(df, cols=['drug', 'concentration'])
df_ctrl, condCol = apm.makeCompositeCol(df_ctrl, cols=['drug', 'concentration'])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['date'].apply(lambda x : x not in excluded_dates)),
           # (df[XCol] > 50),
           (df[XCol] < 1000),
           (df['normal field'] == 5),
           (df[YCol] <= 1e5),
           (df['valid' + YCol[1:]] == True), 
           ]

Filters_ctrl = [(df_ctrl['validatedThickness'] == True), 
               (df_ctrl['substrate'] == substrate),
               (df_ctrl['drug'].apply(lambda x : x in drugs)),
               (df_ctrl['cell subtype'].apply(lambda x : x in subtypes)),
               (df_ctrl['date'].apply(lambda x : x not in excluded_dates)),
               (df_ctrl[XCol] < 1000),
               (df_ctrl['normal field'] == 5),
               (df_ctrl[YCol] <= 1e5),
               (df_ctrl['valid' + YCol[1:]] == True), 
               ]

df_f = apm.filterDf(df, Filters)
df_ctrl_f = apm.filterDf(df_ctrl, Filters_ctrl)


def prepTableForDrugPlot(df_f, XCol, YCol, condCol):
    logMean = lambda x : np.exp(np.mean(np.log(x)))

    # Group By Step 1
    df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
    df_fg = df_fg[[XCol]]
    df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                          valCol = YCol, weightCol = 'ciw'+YCol, weight_method = 'ciw^2')
    df_gC = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')
    
    # Group By Step 2
    df_fg_1 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [XCol],
                       aggFun = logMean) #.drop(columns=['cellID']).reset_index()
    df_fg_2 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [XCol],
                       aggFun = 'std') #.drop(columns=['cellID']).reset_index()
    df_fg_3 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [XCol],
                       aggFun = 'count') #.drop(columns=['cellID']).reset_index()
    df_fg_1 = df_fg_1[[XCol]].rename(columns={XCol: "H0_mean"})
    df_fg_2 = df_fg_2[[XCol]].rename(columns={XCol: "H0_std"})
    df_fg_3 = df_fg_3[[XCol]].rename(columns={XCol: "count"})
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
    
    return(df_gC, df_gD)

#### Plot

sD = apm.styleDict_V2
rD = apm.renameDict

fig, axes = plt.subplots(1, 4, figsize = (17/cm_in, 6/cm_in), 
                         sharex=True, sharey=True, layout='constrained')
axes_f = axes.flatten()
dLabels = {}


#### Controls

conds = ['dmso & 0.0']
df_gC, df_gD = prepTableForDrugPlot(df_ctrl_f, XCol, YCol, condCol)

for i, cond in enumerate(conds):
    for k in range(len(axes_f)):
        ax = axes_f[k]
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        color = sD[cond]['color']
        marker = sD[cond]['marker']
        
        df_c = df_gC[df_gC[condCol] == cond]
        X, Y = df_c[XCol].values, df_c[YCol + '_wAvg'].values/1000
        
        ax.plot(X, Y,
                marker = marker, color = color, ls='',
                ms=6, alpha = 0.3, zorder=3, label = rD[cond])
        
        Xfit, Yfit = np.log(X), np.log(Y)
        wd=1/(np.std(Xfit)) # **2
        we=1/(np.std(Yfit)) # **2

        [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
        A, k = np.exp(b), a
        pval = results.pval
        [k_ciw, b_ciw] = results.params_ciw
        Xplot = np.exp(np.linspace(1, 1e2, 50))
        Yplot = A * Xplot**k
        text_pval = apm.pval2text(pval, n_digits = 3, space = True)
        # dLabels[cond] = {'A':A, 'k':k, 'k_ciw':k_ciw, 'pval':pval, 'text_pval':text_pval}
        if pval < 0.05:
            dLabels[cond] = f'{rD[cond]}' + \
                    f'\nk={k:.2f}' + \
                    r'$\pm$' + f'{(k_ciw/2):.2f}' + \
                    '\n' + text_pval
        else:
            dLabels[cond] = f'{rD[cond]}' + \
                            '\n' + 'NS fit' + \
                            '\n' + text_pval
        ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 1.5,)
            

#### Drugs

conds = ['Y27 & 50.0', 'ck666 & 50.0', 'latrunculinA & 0.5', 'LIMKi & 20.0']
df_gC, df_gD = prepTableForDrugPlot(df_f, XCol, YCol, condCol)

for i, cond in enumerate(conds):
    ax = axes_f[i]
    
    color = sD[cond]['color']
    marker = sD[cond]['marker']
    
    df_c = df_gC[df_gC[condCol] == cond]
    X, Y = df_c[XCol].values, df_c[YCol + '_wAvg'].values/1000
    
    ax.plot(X, Y,
            marker = marker, color = color, ls='',
            ms=6, mec='w', mew=0.5, zorder=4, label = rD[cond])
    
    Xfit, Yfit = np.log(X), np.log(Y)
    wd=1/(np.std(Xfit)) # **2
    we=1/(np.std(Yfit)) # **2

    [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
    A, k = np.exp(b), a
    pval = results.pval
    [k_ciw, b_ciw] = results.params_ciw
    Xplot = np.exp(np.linspace(1, 1e2, 50))
    Yplot = A * Xplot**k
    text_pval = apm.pval2text(pval, n_digits = 3, space = True)
    # dLabels[cond] = {'A':A, 'k':k, 'k_ciw':k_ciw, 'pval':pval, 'text_pval':text_pval}
    if pval < 0.05:
        dLabels[cond] = f'{rD[cond]}' + \
                f'\nk={k:.2f}' + \
                r'$\pm$' + f'{(k_ciw/2):.2f}' + \
                '\n' + text_pval
        ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 1.5,)
        
    else:
        dLabels[cond] = f'{rD[cond]}' + \
                        '\n' + 'Fit not significant' + \
                        '\n' + text_pval
    
    

#### Legend

conds = ['dmso & 0.0', 'Y27 & 50.0', 'ck666 & 50.0', 'latrunculinA & 0.5', 'LIMKi & 20.0']
CommonLegendMarks = []
for i, cond in enumerate(conds):
    ax = axes_f[1]
    color = sD[cond]['color']
    
    label = dLabels[cond]
    LegendMark = matplotlib.lines.Line2D([], [], color=color, ls='--', lw = 2.5,
                                         label = label)
        
    # k, k_ciw  = dLabels[cond]['k'], dLabels[cond]['k_ciw'], 
    # pval, text_pval = dLabels[cond]['pval'], dLabels[cond]['text_pval']
    # if pval < 0.05:
    #     LegendMark = matplotlib.lines.Line2D([], [], color=color, ls='--', lw = 2.5,
    #                                          label = f'{rD[cond]}' + \
    #                                                  f'\nk  = {k:.2f}  ' + \
    #                                                  r'$\pm$' + f' {(k_ciw/2):.2f}' + \
    #                                                  '\n' + text_pval)
    # else:
    #     LegendMark = matplotlib.lines.Line2D([], [], color=color, ls='--', lw = 2.5,
    #                                          label = label)
    #     dLabels[cond] = f'{rD[cond]}' + \
    #                     '\n' + 'NS fit' + \
    #                     '\n' + text_pval
    
    
    CommonLegendMarks.append(LegendMark)
    
plt.figlegend(handles=CommonLegendMarks, fontsize = 8, 
              loc='outside lower center', ncols = 5,
              # title = 'Power-law fits', title_fontproperties = {'weight':'bold'},
              labelspacing = 0.85, handletextpad=0.4, handlelength = 1)
    
#### Format
    
    
for k in range(len(axes_f)):
    ax = axes_f[k]
    ax.grid()
    # ax.set_xlim([0, 600])
    ax.set_xlim([50, 2000])
    # ax.set_ylim([0, 30])
    ax.set_ylim([0.2, 150])
    ax.set_xlabel('$H_{500}$ (nm)')
    if k==0:
        ax.set_ylabel('$E_{500}$ (kPa)')
    ax.legend(loc='lower left', handletextpad=0.2) # loc='center left', bbox_to_anchor=(1, 0.5), 
    
# fig.tight_layout()

plt.show()

if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    # CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


    
# %%% Alternate fig S2B - Split with fits - By comps
 
# Save
SAVE = True
figSubDir = 'S2'
name = 'E500_vs_h0_drugs_wFit_split_byComps'

df = MecaData_Drug
df_ctrl = MecaData_Phy

drugs = ['dmso', 'blebbistatin', 'none', 'Y27', 'ck666', 'latrunculinA', 'LIMKi']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']

# XCol = 'ctFieldThickness'
XCol = 'H0_f_<_500'
YCol = 'E_f_<_500'

df, condCol = apm.makeCompositeCol(df, cols=['drug', 'concentration'])
df_ctrl, condCol = apm.makeCompositeCol(df_ctrl, cols=['drug', 'concentration'])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['date'].apply(lambda x : x not in excluded_dates)),
           # (df[XCol] > 50),
           (df[XCol] < 1000),
           (df['normal field'] == 5),
           (df[YCol] <= 1e5),
           (df['valid' + YCol[1:]] == True), 
           ]

Filters_ctrl = [(df_ctrl['validatedThickness'] == True), 
               (df_ctrl['substrate'] == substrate),
               (df_ctrl['drug'].apply(lambda x : x in drugs)),
               (df_ctrl['cell subtype'].apply(lambda x : x in subtypes)),
               (df_ctrl['date'].apply(lambda x : x not in excluded_dates)),
               (df_ctrl[XCol] < 1000),
               (df_ctrl['normal field'] == 5),
               (df_ctrl[YCol] <= 1e5),
               (df_ctrl['valid' + YCol[1:]] == True), 
               ]

df_f = apm.filterDf(df, Filters)
df_ctrl_f = apm.filterDf(df_ctrl, Filters_ctrl)


def prepTableForDrugPlot(df_f, XCol, YCol, condCol):
    logMean = lambda x : np.exp(np.mean(np.log(x)))

    # Group By Step 1
    df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
    df_fg = df_fg[[XCol]]
    df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                          valCol = YCol, weightCol = 'ciw'+YCol, weight_method = 'ciw^2')
    df_gC = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')
    
    # Group By Step 2
    df_fg_1 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [XCol],
                       aggFun = logMean) #.drop(columns=['cellID']).reset_index()
    df_fg_2 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [XCol],
                       aggFun = 'std') #.drop(columns=['cellID']).reset_index()
    df_fg_3 = apm.dataGroup(df_gC, groupCol = condCol, idCols = [], numCols = [XCol],
                       aggFun = 'count') #.drop(columns=['cellID']).reset_index()
    df_fg_1 = df_fg_1[[XCol]].rename(columns={XCol: "H0_mean"})
    df_fg_2 = df_fg_2[[XCol]].rename(columns={XCol: "H0_std"})
    df_fg_3 = df_fg_3[[XCol]].rename(columns={XCol: "count"})
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
    
    return(df_gC, df_gD)

#### Plot

sD = apm.styleDict_V2
rD = apm.renameDict

fig, axes = plt.subplots(2, 2, figsize = (12/cm_in, 8/cm_in), 
                         sharex=True, sharey=True, layout='constrained')
axes_f = axes.flatten()
dLabels = {}

#### Controls

conds = ['dmso & 0.0']
df_gC, df_gD = prepTableForDrugPlot(df_ctrl_f, XCol, YCol, condCol)

for i, cond in enumerate(conds):
    for k in range(len(axes_f)):
        ax = axes_f[k]
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        color = sD[cond]['color']
        marker = sD[cond]['marker']
        
        df_c = df_f[df_f[condCol] == cond]
        X, Y = df_c[XCol].values, df_c[YCol].values/1000
        
        ax.plot(X, Y,
                marker = marker, color = color, ls='',
                ms=4, alpha = 0.15, zorder=3, label = rD[cond])
        # ax.errorbar(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3, 
        #             xerr=df_gD.loc[cond, 'H0_sem'], yerr=df_gD.loc[cond, 'E_sem']/1e3, 
        #             ls = '', marker = 'o', ms=1, color=color, 
        #             elinewidth = 3, capsize = 5, capthick = 3, zorder=5)
        # ax.plot(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3,
        #         marker = marker, color = color, ls='',
        #         ms=10, mec='k', label = rD[cond], lw=0.5, zorder=6)
        
        Xfit, Yfit = np.log(X), np.log(Y)
        wd=1/(np.std(Xfit)) # **2
        we=1/(np.std(Yfit)) # **2

        [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
        A, k = np.exp(b), a
        pval = results.pval
        [k_ciw, b_ciw] = results.params_ciw
        Xplot = np.exp(np.linspace(1, 1e2, 50))
        Yplot = A * Xplot**k
        text_pval = apm.pval2text(pval, n_digits = 3, space = True)
        dLabels[cond] = {'A':A, 'k':k, 'k_ciw':k_ciw, 'text_pval':text_pval}
        ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 1.5,)
                # label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
                #         f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + '\n' + text_pval)
                # label = r'Control' + f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + '\n' + text_pval)
            
#### Drugs

conds = ['Y27 & 50.0', 'ck666 & 50.0', 'latrunculinA & 0.5', 'LIMKi & 20.0']
df_gC, df_gD = prepTableForDrugPlot(df_f, XCol, YCol, condCol)

for i, cond in enumerate(conds):
    ax = axes_f[i]
    
    color = sD[cond]['color']
    marker = sD[cond]['marker']
    
    df_c = df_f[df_f[condCol] == cond]
    X, Y = df_c[XCol].values, df_c[YCol].values/1000
    
    ax.plot(X, Y,
            marker = marker, color = color, ls='',
            ms=4, mec='w', mew=0.3, zorder=4, label = rD[cond])
    # ax.errorbar(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3, 
    #             xerr=df_gD.loc[cond, 'H0_sem'], yerr=df_gD.loc[cond, 'E_sem']/1e3, 
    #             ls = '', marker = 'o', ms=1, color=apm.lightenColor(color, 0.8), 
    #             elinewidth = 3, capsize = 5, capthick = 3, zorder=5)
    # ax.plot(df_gD.loc[cond, 'H0_mean'], df_gD.loc[cond, 'E_mean']/1e3,
    #         marker = marker, color = apm.lightenColor(color, 0.8), ls='',
    #         ms=10, mec='k', label = rD[cond], lw=0.5, zorder=6)
    
    Xfit, Yfit = np.log(X), np.log(Y)
    wd=1/(np.std(Xfit)) # **2
    we=1/(np.std(Yfit)) # **2

    [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
    A, k = np.exp(b), a
    pval = results.pval
    [k_ciw, b_ciw] = results.params_ciw
    Xplot = np.exp(np.linspace(1, 1e2, 50))
    Yplot = A * Xplot**k
    text_pval = apm.pval2text(pval, n_digits = 3, space = True)
    dLabels[cond] = {'A':A, 'k':k, 'k_ciw':k_ciw, 'text_pval':text_pval}
    ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 1.5,)
            # label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
            #         f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + '\n' + text_pval)
            # label = f'{cond}' + f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + '\n' + text_pval  + '\n')

#### Legend

conds = ['dmso & 0.0', 'Y27 & 50.0', 'ck666 & 50.0', 'latrunculinA & 0.5', 'LIMKi & 20.0']
CommonLegendMarks = []
for i, cond in enumerate(conds):
    ax = axes_f[1]
    color = sD[cond]['color']
    k, k_ciw, text_pval = dLabels[cond]['k'], dLabels[cond]['k_ciw'], dLabels[cond]['text_pval']
    # ax.plot([], [], ls = '--', c = color, lw = 1.5,
    #         label = f'{cond}'
    #                 f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + '\n' + text_pval  + '\n')
    
    LegendMark = matplotlib.lines.Line2D([], [], color=color, ls='--', lw = 1.5,
                                         label = f'{rD[cond]}' + \
                                                 f'\nk  = {k:.2f}  ' + \
                                                 r'$\pm$' + f' {(k_ciw/2):.2f}' + \
                                                 '\n' + text_pval)
    CommonLegendMarks.append(LegendMark)
    
plt.figlegend(handles=CommonLegendMarks, loc='outside right upper',
              # bbox_to_anchor=(0, 1),
              labelspacing = 1.15, handlelength = 1)
    
#### Format
    
for k in range(len(axes_f)):
    ax = axes_f[k]
    ax.grid()
    # ax.set_xlim([0, 600])
    ax.set_xlim([50, 2000])
    # ax.set_ylim([0, 30])
    ax.set_ylim([0.2, 150])
    if k//2==1:
        ax.set_xlabel('$H_{500}$ (nm)')
    if k%2==0:
        ax.set_ylabel('$E_{500}$ (kPa)')
    ax.legend(fontsize = 5, ncol = 1) # loc='center left', bbox_to_anchor=(1, 0.5), 
    
# fig.tight_layout()

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





# %%% Fig S2C - Dose dep graphs
 
# Save
SAVE = True
figSubDir = 'S2'
name = 'Dose_LatA'

# Quantif 1 - 
# ctrl - 1.6175219043739675 0.09831368478048526
# 0.1x - 1.3117575053020685 0.05152019788610142
# 0.5x - 1.1553079524030232 0.04339410726074064
# 1x - 1.1334584265835206 0.042784119056154486
# 2x - 1.0881305698985289 0.05720244677786476
# 10x - 0.9468779097526224 0.02352824625082879
# gs.set_default_options_jv()

listCond = ['Ctrl', '0.1X', '0.5X', '1X', '2X', '10X']
listConc = [0, 0.05, 0.25, 0.5, 1.0, 5.0] # µM
listMean = [1.6175, 1.3117, 1.1553, 1.1334, 1.0881, 0.9469]
listSte = [0.0983, 0.05152, 0.04339, 0.04278, 0.05720, 0.02352]
    
df = pd.DataFrame({'co name':listCond, 
                   'concentration':listConc, 
                   'mean':listMean, 
                   'ste':listSte})

color = apm.styleDict_V2['latrunculinA']['color']
lc = apm.lightenColor(color, 1.25)
mc = apm.lightenColor(color, 0.75)

fig, ax = plt.subplots(1,1, figsize = (7/cm_in, 5/cm_in))
ax.axhline(1, lw=1.15, color='k')
ax.errorbar(df['concentration'], df['mean'], yerr=df['ste'],
            lw = 1.5, color = lc,
            marker = 'o', markerfacecolor = mc, mec = 'None',
            ecolor = mc, elinewidth = 1.5, capsize = 5)

# ax.text(df['concentration'].values, df['mean'].values, df['co name'].values)
# for k in range(df.shape[0]):
#     ax.text(df['concentration'].values[k]-0.06, df['mean'].values[k]-0.04, 
#             df['co name'].values[k], fontsize = 10, ha='right')

# ax.set_xscale('log')
# ax.plot(ax.get_xlim(), [1,1], 'k--', lw=0.8)
ax.grid(axis='y')
ax.set_xlim([-0.3, 5.2])
ax.set_ylim([0.5, 1.75])
ax.set_xlabel('Concentration of LatA (µM)')
ax.set_ylabel('Ratio Fluo\nCortex/Cytoplasm')

plt.show()

if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    

    








# %% Supp Fig 4


# %%% FS4C - Four cell types - all cells 

# Save
SAVE = True
figSubDir = 'S4'
name = 'S4_E500_vs_h500_4celltypes_perComp'

df = MecaData_CellTypes
df_ctrl = MecaData_Phy

df, condCol = apm.makeCompositeCol(df, cols=['cell type', 'cell subtype'])
df_ctrl, condCol = apm.makeCompositeCol(df_ctrl, cols=['cell type', 'cell subtype'])

# XCol = 'ctFieldThickness'
# XCol = 'bestH0'
XCol = 'H0_f_<_500'
YCol = 'E_f_<_500'
ctrl_cond = '3T3 & Atcc-2023' 

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

Filters_ctrl = [(df_ctrl['validatedThickness'] == True), 
               (df_ctrl['substrate'] == '20um fibronectin discs'),
               (df_ctrl['drug'].apply(lambda x : x in ['dmso'])),
               (df_ctrl['cell subtype'].apply(lambda x : x in ['Atcc-2023', 'Atcc-2023-LaGFP'])),
               (df_ctrl['date'].apply(lambda x : x not in excluded_dates)),
               (df_ctrl[XCol] < 1000),
               (df_ctrl['normal field'] == 5),
               (df_ctrl[YCol] <= 1e5),
               (df_ctrl['valid' + YCol[1:]] == True), 
               ]

df_f = apm.filterDf(df, Filters)
df_ctrl_f = apm.filterDf(df_ctrl, Filters_ctrl)

df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'
df_ctrl_f.loc[df_ctrl_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = apm.makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])
df_ctrl_f, condCol = apm.makeCompositeCol(df_ctrl_f, cols=['cell type', 'cell subtype'])
CountByCond2, CountByCell2 = apm.makeCountDf(df_f, condCol)


# Filter 2
Case_A1 = (df_f['cell type'].apply(lambda x : x in ['HoxB8-Macro']))
Case_A2 = (df_f['substrate'] == 'bare glass')
Case_B1 = (df_f['cell type'].apply(lambda x : x in ['DC', 'Dicty']))
Case_B2 = (df_f['substrate'] == 'BSA coated glass')
Case_C1 = (df_f['cell type'].apply(lambda x : x in ['3T3', 'MDCK', 'HeLa']))
Case_C2 = (df_f['substrate'] == '20um fibronectin discs')
Case_D1 = (df_f['cell type'].apply(lambda x : x in ['3T3', 'MDCK', 'DC']))
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
          '3T3 & Atcc-2023'     : 'dimgray', 
          'HeLa & fucci'         : apm.cL_Set2[1],  
          'DC & mouse-primary'   : apm.cL_Set2[2],  
          'Dicty & DictyBase-WT' : apm.cL_Set2[3],  
          # 'HoxB8-Macro & ctrl'   : apm.cL_Set2[4],  
          'MDCK & WT'            : apm.cL_Set2[5],
          }

rD = {
      '3T3 & Atcc-2023'      :  '3T3 ATCC', 
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


df_ctrl_fg = apm.dataGroup(df_ctrl_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_ctrl_fg = df_ctrl_fg[[XCol]]
df_ctrl_fgw2 = apm.dataGroup_weightedAverage(df_ctrl_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = YCol, weightCol = 'ciw' + YCol, weight_method = 'ciw^2')
df_ctrl_plot = pd.merge(left=df_ctrl_fg, right=df_ctrl_fgw2, on='cellID', how='inner')



#### Plot
fig, axes = plt.subplots(2, 2, figsize=(17/cm_in, 12/cm_in), 
                         sharex=True, sharey=True, layout='constrained')
axes = axes.flatten('C')
dLabels = {}

# Per cell / per comp
# df_plot = df_f
# df_ctrl_plot = df_ctrl_f
# YCol += '_wAvg'
# df_plot[YCol] /= 1000
df_f[YCol] /= 1000
# df_ctrl_plot[YCol] /= 1000


# Fit for the controls

Xctrl, Yctrl = df_ctrl_f[XCol].values, df_ctrl_f[YCol].values/1000
Xctrl_fit, Yctrl_fit = np.log(Xctrl), np.log(Yctrl)
wd=1/(np.std(Xctrl_fit)) # **2
we=1/(np.std(Yctrl_fit)) # **2
[a, b], results = ufun.fitLineTLS(Xctrl_fit, Yctrl_fit, wd=wd, we=we)
A, k = np.exp(b), a
pval = results.pval
[k_ciw, b_ciw] = results.params_ciw
Xctrl_plot = np.exp(np.linspace(1, 9, 50))
Yctrl_plot = A * Xctrl_plot**k
text_pval = apm.pval2text(pval, n_digits = 3, space = True)
print('expo ctrl', k, results.params_ciw[1]/2)

dLabels[ctrl_cond] = f'{rD[ctrl_cond]}' + \
               '\n' + f'k  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + \
               '\n' + text_pval



for i in range(len(axes)):
    ax = axes[i]
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # color = sD[cond]['color']
    # marker = sD[cond]['marker']
    # color = colorsD[cond]
    # marker = 'o'
    
    #### Controls
    # sns.scatterplot(ax = ax, x=df_ctrl_plot[XCol].values, y=df_ctrl_plot[YCol].values, 
    #                 marker = 'o', s = 25, color = 'dimgray', alpha = 0.3,
    #                 zorder = 3)
    ax.plot(Xctrl_plot, Yctrl_plot, ls = '--', color = 'dimgray', 
            lw = 2.0, zorder = 1, alpha = 0.9)
            # label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
            #         f'\nk  = {k:.2f}' + '\n' + text_pval)
            
    # if i == 0:
    #     hM, hL, hH = ufun.getLogNDistributionDescriptors(df_ctrl_plot[XCol].values)
    #     EM, EL, EH = ufun.getLogNDistributionDescriptors(df_ctrl_plot[YCol].values)
    #     print(f'Cell type 3T3 ATCC, N = {len(df_ctrl_plot):.0f}')
    #     print(f'For {XCol} vs {YCol}')
    #     print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
    #     print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
    #     print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
    #     print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')
        

    #### Cell type i
    # df_fc = df_plot[df_plot[condCol] == co_order[i]]
    df_fc = df_f[df_f[condCol] == co_order[i]]
    color = colorsD[co_order[i]]

    medianX = np.median(df_fc[XCol].values)
    medianY = np.median(df_fc[YCol].values)
    
    alpha = 0.5
    s = 30
    
    sns.scatterplot(ax = ax, x=df_fc[XCol].values, y=df_fc[YCol].values, 
                    marker = 'o', s = s, color = color, edgecolor = 'k', linewidth=0.5, alpha = alpha,
                    zorder = 3) # , label = 'Median $H_0$ = ' + f'{medianX:.0f} nm'\
                                  #      f'\nMedian $E$ = ' + f'{medianY:.1f} kPa')
                                  
    Xfit, Yfit = np.log(df_fc[XCol].values), np.log(df_fc[YCol].values)
    # print(np.std(Xfit), np.std(Yfit), np.std(Xfit)/np.std(Yfit))
    wd=1/(np.std(Xfit)) # **2
    we=1/(np.std(Yfit)) # **2

    [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
    A, k = np.exp(b), a
    pval = results.pval
    [k_ciw, b_ciw] = results.params_ciw
    
    Xplot = np.exp(np.linspace(1, 9, 50))
    Yplot = A * Xplot**k
    text_pval = apm.pval2text(pval, n_digits = 3, space = True)
    dLabels[co_order[i]] = f'{rD[co_order[i]]}' + \
                            '\n' + f'k  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + \
                            '\n' + text_pval
        
    ax.plot(Xplot, Yplot, ls = '--', c = apm.lightenColor(color, 0.75), 
            lw = 2.0, zorder = 6,)
            # label = r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + \
            #         f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + '\n' + text_pval)

    # ax.legend(fontsize = 9, loc = 'best', handlelength=1)
    ax.set_xlabel('$H_{500}$ (nm)')
    ax.set_ylabel('$E_{500}$ (kPa)')
    # ax.set_title(co_order[i])
    if i%2 != 0:
        ax.set_ylabel('')
    
    ax.text(820, 110, f'{rD[co_order[i]]}', va='center', ha='center',  weight = 'bold',
            color=apm.lightenColor(colorsD[co_order[i]], 0.75), fontsize=10.0, backgroundcolor='w', zorder=2)
    # ax.text(400, 80, f'{rD[ctrl_cond]}', va='center', ha='left', 
    #         color=colorsD[ctrl_cond], fontsize=6.0, backgroundcolor='w', zorder=2)
    
    hM, hL, hH = ufun.getLogNDistributionDescriptors(df_fc[XCol].values)
    EM, EL, EH = ufun.getLogNDistributionDescriptors(df_fc[YCol].values)
    print(f'Cell type {co_order[i]}, N = {len(df_fc):.0f}')
    print(f'For {co_order[i]}, {XCol} vs {YCol}')
    print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
    print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
    print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
    print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')
    
#### Legend
conds = ['3T3 & Atcc-2023'] + co_order
# conds = co_order
CommonLegendMarks = []
for i, cond in enumerate(conds):
    ax = axes[1]
    color = colorsD[cond]
    
    label = dLabels[cond]
    LegendMark = matplotlib.lines.Line2D([], [], color=color, ls='--', lw = 2.5,
                                         label = label)
        
    # k, k_ciw  = dLabels[cond]['k'], dLabels[cond]['k_ciw'], 
    # pval, text_pval = dLabels[cond]['pval'], dLabels[cond]['text_pval']
    # if pval < 0.05:
    #     LegendMark = matplotlib.lines.Line2D([], [], color=color, ls='--', lw = 2.5,
    #                                          label = f'{rD[cond]}' + \
    #                                                  f'\nk  = {k:.2f}  ' + \
    #                                                  r'$\pm$' + f' {(k_ciw/2):.2f}' + \
    #                                                  '\n' + text_pval)
    # else:
    #     LegendMark = matplotlib.lines.Line2D([], [], color=color, ls='--', lw = 2.5,
    #                                          label = label)
    #     dLabels[cond] = f'{rD[cond]}' + \
    #                     '\n' + 'NS fit' + \
    #                     '\n' + text_pval
    
    
    CommonLegendMarks.append(LegendMark)
    
plt.figlegend(handles=CommonLegendMarks, fontsize = 8, 
              loc='outside right center',
              title = 'Power-law fits', title_fontproperties = {'weight':'bold'},
              labelspacing = 1.15, handletextpad=0.4, handlelength = 1)
    
#### Format
rD.update({'E_eff_wAvg':'E_{eff} (kPa)'})

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder=0)
    apm.renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD)
    ax.set_xlim(50, 2000)
    ax.set_ylim(0.4, 300)

# Show
# plt.tight_layout()
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









