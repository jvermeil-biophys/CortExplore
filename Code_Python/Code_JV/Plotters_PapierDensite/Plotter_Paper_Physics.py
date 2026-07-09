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
from matplotlib.gridspec import GridSpec
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

figDir = 'C:/Users/josep/Desktop/Seafile/PapierDensité/FiguresMain'


# %% > Data import & export

# %%% MecaData_Phy
# MecaData_Phy = taka3.getMergedTable('MecaData_Physics')
# MecaData_Phy2 = taka3.getMergedTable('MecaData_Physics_V2')
MecaData_Phy3 = takaP.getMergedTable('MecaData_Physics_V3')
MecaData_Phy4 = takaP.getMergedTable('MecaData_Physics_V4bis')
MecaData_Phy5 = takaP.getMergedTable('MecaData_Physics_V5_Dimi')

# MecaData_Phy2 = MecaData_Phy2.dropna(axis=0, subset='date')
MecaData_Phy3 = MecaData_Phy3.dropna(axis=0, subset='date')
MecaData_Phy4 = MecaData_Phy4.dropna(axis=0, subset='date')
MecaData_Phy5 = MecaData_Phy5.dropna(axis=0, subset='date')

MecaData_Phy = MecaData_Phy4


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

# Find an appropriate day to test Dimitriadis
# for M in MecaData_Phy['manipID'].unique():
#     df_m = MecaData_Phy[MecaData_Phy['manipID'] == M]
#     Q1 = np.percentile(df_m['surroundingThickness'], 25)
#     med = np.median(df_m['surroundingThickness'])
#     Q3 = np.percentile(df_m['surroundingThickness'], 75)
#     if med > 250:
#         print(M, f'{Q1:.0f}', f'{med:.0f}', f'{Q3:.0f}', len(df_m['cellID'].unique()))


# %%% Utility scripts

# %%%% Compare content

MecaData_Phy3['full_comp_id'] = MecaData_Phy3['cellID'] + '_' + MecaData_Phy3['compNum'].astype(str)
full_comp_id_L3 = MecaData_Phy3['full_comp_id'].values

MecaData_Phy4['full_comp_id'] = MecaData_Phy4['cellID'] + '_' + MecaData_Phy4['compNum'].astype(str)
full_comp_id_L4 = MecaData_Phy4['full_comp_id'].values

print('in 3, not in 4')
for s in full_comp_id_L3:
    if s not in full_comp_id_L4:
        print(s)

print('in 4, not in 3')
for s in full_comp_id_L4:
    if s not in full_comp_id_L3:
        print(s)
        
        
        
MecaData_Phy3['full_comp_id'] = MecaData_Phy3['cellID'] + '_' + MecaData_Phy3['compNum'].astype(str)
full_comp_id_L3 = MecaData_Phy3['full_comp_id'].values

MecaData_Phy5['full_comp_id'] = MecaData_Phy5['cellID'] + '_' + MecaData_Phy5['compNum'].astype(str)
full_comp_id_L5 = MecaData_Phy5['full_comp_id'].values

print('in 3, not in 5')
for s in full_comp_id_L3:
    if s not in full_comp_id_L5:
        print(s)

print('in 5, not in 3')
for s in full_comp_id_L5:
    if s not in full_comp_id_L3:
        print(s)

# %%%% Compare content 2

df = MecaData_Phy3
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26', '24-12-18']
# figname = 'bestH0' + drugSuffix
XCol = 'H0_f_<_500'
YCol = 'E_f_<_500'

df['full_comp_id'] = df['cellID'] + '_' + df['compNum'].astype(str)



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

df_f3 = apm.filterDf(df, Filters)
full_comp_id_L3 = df_f3['full_comp_id'].values



df = MecaData_Phy4
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26', '24-12-18']
# figname = 'bestH0' + drugSuffix
XCol = 'H0_f_<_500'
YCol = 'E_f_<_500'

df['full_comp_id'] = df['cellID'] + '_' + df['compNum'].astype(str)

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

df_f4 = apm.filterDf(df, Filters)
full_comp_id_L4 = df_f4['full_comp_id'].values



df = MecaData_Phy5
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26', '24-12-18']
# figname = 'bestH0' + drugSuffix
XCol = 'H0_f_<_500'
YCol = 'E_f_<_500'

df['full_comp_id'] = df['cellID'] + '_' + df['compNum'].astype(str)

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

df_f5 = apm.filterDf(df, Filters)
full_comp_id_L5 = df_f5['full_comp_id'].values



L3 = []
L4 = []
L5 = []

print('in 3, not in 4')
for s in full_comp_id_L3:
    if s not in full_comp_id_L4:
        print(s)
        print(MecaData_Phy4.loc[MecaData_Phy4['full_comp_id']==s, 'valid' + YCol[1:]].values[0])
        print(MecaData_Phy4.loc[MecaData_Phy4['full_comp_id']==s, 'issue' + YCol[1:]].values[0])
        L3.append(s)

print('in 4, not in 3')
for s in full_comp_id_L4:
    if s not in full_comp_id_L3:
        print(s)
        L4.append(s)
        
print(len(L3), len(L4))


print('in 3, not in 5')
for s in full_comp_id_L3:
    if s not in full_comp_id_L5:
        print(s)
        print(MecaData_Phy5.loc[MecaData_Phy5['full_comp_id']==s, 'valid' + YCol[1:]].values[0])
        print(MecaData_Phy5.loc[MecaData_Phy5['full_comp_id']==s, 'issue' + YCol[1:]].values[0])
        L3.append(s)

print('in 5, not in 3')
for s in full_comp_id_L5:
    if s not in full_comp_id_L3:
        print(s)
        L5.append(s)
        
print(len(L3), len(L5))


# %%%% Correct some weird stuff

df = MecaData_Phy3
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26', '24-12-18']
# figname = 'bestH0' + drugSuffix
XCol = 'H0_f_<_500'
YCol = 'E_f_<_500'

df['full_comp_id'] = df['cellID'] + '_' + df['compNum'].astype(str)



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

df_f3 = apm.filterDf(df, Filters)
full_comp_id_L3 = df_f3['full_comp_id'].values



df = MecaData_Phy4
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26', '24-12-18']
# figname = 'bestH0' + drugSuffix
XCol = 'H0_f_<_500'
YCol = 'E_f_<_500'

df['full_comp_id'] = df['cellID'] + '_' + df['compNum'].astype(str)

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

df_f4 = apm.filterDf(df, Filters)
full_comp_id_L4 = df_f4['full_comp_id'].values

L3 = []
L4 = []

print('in 3, not in 4')
for s in full_comp_id_L3:
    if s not in full_comp_id_L4:
        MecaData_Phy4.loc[MecaData_Phy4['full_comp_id']==s, 'valid' + YCol[1:]] = True
        MecaData_Phy4.loc[MecaData_Phy4['full_comp_id']==s, 'issue' + YCol[1:]] = 'corrected'
        L3.append(s)

print('in 4, not in 3')
for s in full_comp_id_L4:
    if s not in full_comp_id_L3:
        print(s)
        L4.append(s)
        
print(len(L3), len(L4))

path = 'C:/Users/josep/Documents/MagneticPincherData/Data_Analysis/MecaData_Physics_V4bis.csv'
MecaData_Phy4.to_csv(path, sep=';', index=False)


    
    

# %% --------

# %% Main Figure 1

# %%% Fig 1D & G & Supp 9A & C - h(t) and F(t)

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

# task = '24-04-11_M3_P1_C1'
task = '23-03-09_M4_P1_C5' # -> for Pplot_Timeseries_V3()


res = takaP.computeGlobalTable_meca(mode = 'fromScratch', task = task, fileName = 'test', 
                                    save = True, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting'




# %%% Fig 1E - DH/H Before-After/Before

# Save
SAVE = True
figSubDir = 'F1'
name = 'F1_E' # 

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

df_f['ratioDh_Hi'] = df_f['Dh_BeforeAfter'].values/df_f['previousThickness'].values

color_med = 'darkred'

#### Plot
fig, axes = plt.subplots(1, 1, figsize=(6/cm_in, 6/cm_in), layout='constrained')
ax = axes
# sns.swarmplot(ax=ax, data = df_f, x='cell type', y='Dh_BeforeAfter', size=1)
ax.hist(df_f['ratioDh_Hi'].values, 
        bins=300, color='gray', zorder=3)

median = np.median(df_f['ratioDh_Hi'].values)
# ax.axvline(median, color='darkred', ls='-', lw=1,
#            label=f'Median = {median:.3f}', zorder=3)
# ax.text(x=0.2, y=200, s='Median\n' + f'{median:.3f} nm', size=7, c=color_med)
# ax.text(x=0.10, y=275, s=r'$\Delta H$ = ', size=7, c='k', va='center')
# ax.text(x=0.40, y=275, s=r'$H_{final}$', size=7, c='red', va='center')
# ax.text(x=0.65, y=275, s=r' - ', size=7, c='k', va='center')
# ax.text(x=0.75, y=275, s=r'$H_{init}$', size=7, c='deepskyblue', va='center')
# ax.legend(handlelength = 1.25)
ax.grid()
ax.axvline(0, color='k', ls='-', lw=0.75, zorder=5)
ax.axhline(0, color='k', ls='-', lw=1, zorder=5)
ax.set_xlim(-1, 1)
ax.set_xlabel(r'$\Delta H/H_{init}$ (ratio)', labelpad=0.1)
ax.set_ylabel('N compressions')
# ax.set_title('Thickness at 5mT\n(H_after - H_before) / H_before')


# Show
# plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')





# %%% Fig 1EF - DH/H Before-After/Before + Peak delay

# Save
SAVE = True
figSubDir = 'F1'
name = 'F1_EF' # 

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

df_f['ratioDh_Hi'] = df_f['Dh_BeforeAfter'].values/df_f['previousThickness'].values

color_med = 'darkred'

#### Plot
fig, axes = plt.subplots(2, 1, figsize=(6/cm_in, 6/cm_in), layout='constrained')
ax = axes[0]
# sns.swarmplot(ax=ax, data = df_f, x='cell type', y='Dh_BeforeAfter', size=1)
ax.hist(df_f['ratioDh_Hi'].values, 
        bins=300, color='gray', zorder=3)

median = np.median(df_f['ratioDh_Hi'].values)
# ax.axvline(median, color='darkred', ls='-', lw=1,
#            label=f'Median = {median:.3f}', zorder=3)
# ax.text(x=0.2, y=200, s='Median\n' + f'{median:.3f} nm', size=7, c=color_med)
# ax.legend(handlelength = 1.25)
## V1
# ax.text(x=0.10, y=250, s=r'$\Delta H$ = ', size=7, c='k', va='center')
# ax.text(x=0.40, y=250, s=r'$H_{final}$', size=7, c='red', va='center')
# ax.text(x=0.65, y=250, s=r' - ', size=7, c='k', va='center')
# ax.text(x=0.75, y=250, s=r'$H_{init}$', size=7, c='deepskyblue', va='center')
## V2
x0, y0 = 0.3, 290
ax.text(x=x0+0.15, y=y0, s=r'$\Delta H$ = ', size=7, c='k', va='center')
ax.text(x=x0, y=y0-50, s=r'$H_{final}$', size=7, c='red', va='center')
ax.text(x=x0+0.23, y=y0-50, s=r' $-$ ', size=7, c='k', va='center')
ax.text(x=x0+0.37, y=y0-50, s=r'$H_{init}$', size=7, c='deepskyblue', va='center')
ax.add_patch(plt.Rectangle((x0-0.025, y0-70), 0.6, 100, fc="white",
                           zorder=2, alpha=0.75))

ax.grid()
ax.axvline(0, color='k', ls='-', lw=0.75, zorder=5)
ax.axhline(0, color='k', ls='-', lw=1, zorder=5)
ax.set_xlim(-1, 1)
ax.set_xlabel(r'$\Delta H/H_{init}$ (ratio)', labelpad=0.6)
ax.set_ylabel('N compressions', fontsize=6, labelpad=0.6)
# ax.set_title('Thickness at 5mT\n(H_after - H_before) / H_before')

ax = axes[1]
ax.hist(df_f['peakDelay'].values, bins=30, color='gray', zorder=3)

median = np.median(df_f['peakDelay'].values)
# ax.axvline(median, color='darkred', ls='-', lw=1,
#            label=f'Median = {median:.2f} s', zorder=3)
# ax.text(x=0.2, y=225, s='Median\n' + f'{median*1e3:.0f} ms', size=7, c=color_med)
# ax.legend(handlelength = 1.25, loc='upper left')
ax.grid()
ax.axvline(0, color='k', ls='-', lw=0.75, zorder=5)
ax.axhline(0, color='k', ls='-', lw=1, zorder=5)
ax.set_xlabel(r'Force-thickness peak delay $\delta T$ (s)', labelpad=0.6)
ax.set_ylabel('N compressions', fontsize=6, labelpad=0.6)
ax.set_xlim(-1, 1)
# ax.set_title(r'Time delay between max force and min thickness')

# fig.supylabel('N compressions')


# Show
# plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%% Fig 1H - Distribution H et E

# Source : Plotter_AtccPhysics - Figure NC1.1 - V4 - Thickness & Stiffness LOG SMALL

apm.setGraphicOptions(mode = 'print', 
                      palette = 'Set2', 
                      colorList = apm.cL_Set21)

# Save
SAVE = True
figSubDir = 'F1'
name = 'F1_H_1-1'

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
fig, axes = plt.subplots(1, 2, figsize=(11/cm_in, 5/cm_in), sharey=True, layout='compressed')
color = apm.cL_Set2[0]
color = 'dimgray'
color_med = 'darkred'

#### 01 - Best H0
ax = axes[0]

ax.set_title('Thickness')
ax.set_xlabel('$H_{0}$ (nm)', labelpad=0.5)
ax.set_ylabel('Count (cells)', labelpad=1)

ax, histo, logbins = apm.plot_logstairs(ax, df_fg[XCol].values, bins=12, logbins = [], 
                                    normalized = False, fill = True, color = color, label = '')
ax.set_xlim([0, ax.get_xlim()[1]])
medianH0 = np.median((df_fg[XCol].values))
ax.axvline(medianH0, c=color_med, label = f'Median\n$H_0$ = {medianH0:.0f} nm')
ax.text(x=400, y=32, s='Median\n' + f'{medianH0:.0f} nm', size=7, c=color_med)
ax.set_xlim([10, 10000])
# ax.legend()

print(np.median(df_fg[XCol].values))
print(np.mean(np.log10(df_fg[XCol].values)))
print(10**(np.mean(np.log10(df_fg[XCol].values))))
print(np.std(np.log10(df_fg[XCol].values)))
print(10**(np.std(np.log10(df_fg[XCol].values))))

#### 02 - E_500
ax = axes[1]

ax.set_title('Stiffness')
ax.set_xlabel('$E$ (kPa)', labelpad=0.5)
ax, histo, logbins = apm.plot_logstairs(ax, df_fgw2[YCol + '_wAvg'].values, bins=12, logbins = [], 
                                    normalized = False, fill = True, color = color, label = '')
ax.set_xlim([0, ax.get_xlim()[1]])
medianE500 = np.median(df_fgw2[YCol + '_wAvg'].values)
ax.axvline(medianE500, c=color_med, label = 'Median\n$E_{500}$ = ' + f'{medianE500:.2f} kPa')
ax.set_xlim([0.1, 100])
ax.set_ylim([0, 40])
ax.text(x=10, y=32, s='Median\n' + f'{medianE500:.2f} kPa', size=7, c=color_med)
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
# plt.tight_layout()
plt.show()

# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')




# %% Main Figure 2

# %%% Fig 2A

# Save
SAVE = True
figSubDir = 'F2'
name = 'F2_A' #

print('\n------\n') 

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
           (df[XCol] < 1100),
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
# fig, ax = plt.subplots(1, 1, figsize=(9/cm_in, 8.5/cm_in))
# win, hin = 0.32, 0.35
# xin, yin = 0.95-win, 0.91-hin 
# ax_in = ax.inset_axes([xin, yin, win, hin])
fig, ax = plt.subplots(1, 1, figsize=(9/cm_in, 12/cm_in))
win, hin = 0.34, 0.35
xin, yin = 0.93-win, 0.93-hin 
ax_in = ax.inset_axes([xin, yin, win, hin])

ax = ax
ax.set_xscale('log')
ax.set_yscale('log')
c_base = 'gray'
c_dark = apm.lightenColor(c_base, 0.7)

sns.scatterplot(ax = ax, x=df_f[XCol].values, y=df_f[YCol].values/1000, 
                marker = 'o', s = 15, color = c_base, alpha = 0.33, zorder=4) #, label='All compressions')
Xfit, Yfit = np.log(df_f[XCol].values), np.log(df_f[YCol].values/1000)

wd=1/(np.std(Xfit)) # **2
we=1/(np.std(Yfit)) # **2

[a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
A, k = np.exp(b), a
pval = results.pval # results.pval
[k_ciw, b_ciw] = results.params_ciw
Xplot = np.exp(np.linspace(min(Xfit)*0.1, max(Xfit)*10, 50))
Yplot = A * Xplot**k
text_pval = apm.pval2text(pval, n_digits = 3, space = True)
ax.plot(Xplot, Yplot, ls = '-', c = c_dark, lw = 2.0, zorder=6,
        label = r'$\bf{Fit\ y\ =\ A\cdot x^k}$' + \
                # f'\nA = {A:.1e}' + \
                f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + '\n' + text_pval)

# ax.legend().set_visible(False)
# ax.legend(loc = 'lower left', handlelength = 1)
ax.set_title('All compressions', color = c_dark, weight = 'bold')
ax.set_ylabel('$E$ (kPa)')
ax.set_xlabel('$H_{0}$ (nm)')
ax.grid(visible=True, which='major', axis='both')
# ax.set_xlim([50, 1100])
ax.set_xlim([40, 3000])
ax.set_ylim([0.5, 200])
# ax.tick_params(axis='both', direction='in', which='both')

hM, hL, hH = ufun.getLogNDistributionDescriptors(df_f[XCol].values)
EM, EL, EH = ufun.getLogNDistributionDescriptors(df_f[YCol].values/1000)
A_Low = np.exp(b-(b_ciw/2))
A_High = np.exp(b+(b_ciw/2))

print('F2_A - Control case')
print(f"n = {CountByCond.loc['dmso', 'compCount']:.0f}, " + \
      f"N = {CountByCond.loc['dmso', 'cellCount']:.0f}, " + \
      f"M = {CountByCond.loc['dmso', 'manipsCount']:.0f}")
print(f'For {XCol} vs {YCol}\n')
print('---')
print('By compression')
print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
print(f'Power-law constant & Ci : {A:.2e} [{A_Low:.2e}-{A_High:.2e}]')
print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')
    
    
#### Inset
ax.add_patch(plt.Rectangle((360, 12), 2200, 170, 
                           fc="white", zorder=2, alpha=0.8))
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
pval = results.pval # results.pval
Xplot = np.exp(np.linspace(0.1*min(Xfit), 10*max(Xfit), 50))
Yplot = A * Xplot**k
text_pval = apm.pval2text(pval, n_digits = 3, space = True)
ax.plot(Xplot, Yplot, ls = '-', c = c_dark, lw = 1.5,
        label =  text_pval)
        # label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}  ' + r'$\pm$' + f' {(k_ciw/2):.2f}' + \
        #         f'\n$R^2$  = {R2:.2f}' + '\n' + text_pval)

# ax.legend(fontsize = 6, handlelength = 1)
ax.set_title('Average per cell', color = c_dark, weight = 'bold')
ax.grid()
# ax.set_ylabel('$E_{500}$ (kPa)')
# ax.set_xlabel('$H_0$ (nm)')
# ax.set_xlim([50, 1100])
ax.set_xlim([40, 3000])
ax.set_ylim([0.5, 200])
ax.tick_params(axis='both', direction='in', which='both')
# ax.set_xticklabels(fontsize=9)
# ax.set_yticklabels(fontsize=9)

hM, hL, hH = ufun.getLogNDistributionDescriptors(df_plot[XCol].values)
EM, EL, EH = ufun.getLogNDistributionDescriptors(df_plot[YCol+'_wAvg'].values/1000)
A_Low = np.exp(b-(b_ciw/2))
A_High = np.exp(b+(b_ciw/2))

print('---')
print('By cells')
print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
print(f'Power-law constant & Ci : {A:.2e} [{A_Low:.2e}-{A_High:.2e}]')
print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')
print('---------------')

# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)

# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')
    
    

    
# %% Main Figure 4


# %%% Functions to analyze and plot

dict_code = {'bestH0': 'H0',
             'surroundingThickness': 'H5mT',
             'H0_f_<_400': 'H400',
             'H0_f_<_500': 'H500',
             'H0_f_<_600': 'H600',
             'E_f_<_400': 'E400',
             'E_f_<_500': 'E500',
             'E_f_<_600': 'E600',
             'E_eff': 'Eeff',
             }

dict_axisLabels = {'bestH0': '$H_{0}$ (nm)',
                    'surroundingThickness': '$H_{5mT}$ (nm)',
                    'H0_f_<_400': '$H_{400}$ (nm)',
                    'H0_f_<_500': '$H_{0}$ (nm)',
                    'H0_f_<_600': '$H_{600}$ (nm)',
                    'E_f_<_400': '$E_{400}$ (kPa)',
                    'E_f_<_500': '$E$ (kPa)',
                    'E_f_<_600': '$E_{600}$ (kPa)',
                    'E_eff': '$E_{eff}$ (kPa)',
                    }

def compute_Eh_Exponent(df, XCol = 'H0_f_<_500', YCol = 'E_f_<_500',
                        crit_NcompsMin = 10,
                        crit_pvalFit = 0.1,
                        crit_thickCV = 0.5,
                        activeCrits = ['NcompsMin', 'pvalFit', 'thickCV'],
                        modeFit = 'OLS'):

    df, condCol = apm.makeCompositeCol(df, cols=['date'])
    CountByCond, CountByCell = apm.makeCountDf(df, condCol)
    df_f = df
    # df_f = apm.computeNLMetrics_V2(df_f, th_NLI = np.log10(2), ref_strain = 0.2)
    
    CID_longSeries = CountByCell[CountByCell['compCount'] >= 5].reset_index()['cellID'].values
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
               'A'+codeXY:[], 'alpha'+codeXY:[], 'alpha_ciw'+codeXY:[], 
               'pval'+codeXY:[], 'R2'+codeXY:[], 'thickCV'+codeXY:[], 
               codeX+'_logmean':[], codeY+'_logmean':[], #'NLR_mean':[],
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
            [b, a], results = ufun.fitLine(Xfit, Yfit)
            A, alpha = np.exp(b), a
            perc, dof, = 0.975, len(Yfit)-2
            q = st.t.ppf(perc, dof)
            params_sd = [results.cov_HC3[k, k]**0.5 for k in range(len(results.params))]
            params_ciw = [q * sd for sd in params_sd]
            alpha_ciw = results.HC3_se[1] * q
            R2 = results.rsquared
            pval = results.pvalues[1]
            
            # print(cid)
            # print(alpha, A)
            # print([a, b])
            # print(params_sd[::-1])
            # print(params_ciw[::-1])
            # print('---')
        
        elif modeFit == 'ODR':
            wd=1/(np.std(Xfit)) # **2
            we=1/(np.std(Yfit)) # **2
            
            params, results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
            a, b = params
            alpha, A = a, np.exp(b)
            alpha_ciw, _ = results.params_ciw
            pval = results.pval
            R2 = results.R2
            
            # print(cid)
            # print(alpha, A)
            # print(params)
            # print(results.params_sd)
            # print(results.params_ciw)
            # print('---')
        
        H_logmean = np.mean(Xfit)
        E_logmean = np.mean(Yfit)
        # NLR_mean  = np.mean(df_cell['NLI_mod'])
        thickCV = np.std(Xfit)/H_logmean
        # print(cid, f'{thickCV:.3f}', f'{pval:.3f}', f'{R2:.3f}')
        
        dictFit['cellID'].append(cid)
        dictFit['A'+codeXY].append(A)
        dictFit['alpha'+codeXY].append(alpha)
        dictFit['alpha_ciw'+codeXY].append(alpha_ciw)
        dictFit['thickCV'+codeXY].append(thickCV)
        dictFit['pval'+codeXY].append(pval)
        dictFit['R2'+codeXY].append(R2)
        dictFit[codeX + '_logmean'].append(H_logmean)
        dictFit[codeY + '_logmean'].append(E_logmean)
        # dictFit['NLR_mean'].append(NLR_mean)
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
        A, alpha, alpha_ciw = df_res.loc[cid, 'A'+codeXY], df_res.loc[cid, 'alpha'+codeXY], df_res.loc[cid, 'alpha_ciw'+codeXY] #np.exp(b), a
        R2 = df_res.loc[cid, 'R2'+codeXY]
        pval = df_res.loc[cid, 'pval'+codeXY]
        CV = df_res.loc[cid, 'thickCV'+codeXY]
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
                        f'$\\alpha$  = {alpha:.2f} $\\pm $ {alpha_ciw:.2f}\n' + \
                         f'CV={CV:.2f}, pv={pval:.2f}, R2={R2:.2f}',
                )
            
        ### Format
        ax.legend(fontsize = 8, loc = 'lower left', handlelength=1)#.set_visible(False)
        ax.set_ylabel(dict_axisLabels[YCol])
        ax.set_xlabel(dict_axisLabels[XCol])
        ax.grid(visible=True, which='major', axis='both')
        ax.set_xlim([50, 1100])
        ax.set_ylim([1, 100])
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
        A, alpha, alpha_ciw = df_res.loc[cid, 'A'+codeXY], df_res.loc[cid, 'alpha'+codeXY], df_res.loc[cid, 'alpha_ciw'+codeXY] #np.exp(b), a
        R2, pval = df_res.loc[cid, 'R2'+codeXY], df_res.loc[cid, 'pval'+codeXY] #results.rsquared, results.pvalues[1]
        valid = df_res.loc[cid, 'valid_global'+codeXY]
        Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
        Yplot = A * Xplot**alpha

        sns.scatterplot(ax = ax, x=df_cell[XCol].values, y=df_cell[YCol].values/1000, 
                        marker = 'o', s = 20, alpha = 0.9, color = c)
        ax.plot(Xplot, Yplot, ls = '--', c = c, lw = 1.5, alpha = 0.8,)
                # label =  f'$\\alpha$  = {alpha:.2f} $\\pm $ {alpha_ciw:.2f}')
    
    ### Plot the fit on all compressions
    Xfit, Yfit = np.log(df_f2[XCol].values), np.log(df_f2[YCol].values/1000)
    [b, a], results = ufun.fitLine(Xfit, Yfit)
    A, alpha = np.exp(b), a
    perc, dof, = 0.975, len(Yfit)-2
    q = st.t.ppf(perc, dof)
    alpha_ciw = results.HC3_se[1] * q
    R2 = results.rsquared
    pval = results.pvalues[1]
    
    Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = A * Xplot**alpha
    ax.plot(Xplot, Yplot, ls = '-', c = 'dimgray', lw = 2.0,
             label =  f'$\\alpha$  = {alpha:.2f} $\\pm $ {alpha_ciw:.2f}')
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
    alpha_ciw = results.HC3_se[1] * q
    R2 = results.rsquared
    pval = results.pvalues[1]
    
    Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = A * Xplot**alpha
    ax.plot(Xplot, Yplot, ls = '-', c = 'dimgray', lw = 2.0,
             label =  f'$\\alpha$  = {alpha:.2f} $\\pm $ {alpha_ciw:.2f}')
    
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
        alpha_ciw = results.HC3_se[1] * q
        # R2 = results.rsquared
        # pval = results.pvalues[1]
        dict_byDate['alpha_allComps'].append(alpha)
        dict_byDate['alpha_allComps_ciw'].append(alpha_ciw)
        
        ### Do the fit on all cell average
        Xfit, Yfit = (res_date[codeX+'_logmean'].values), (res_date[codeY+'_logmean'].values)
        [b, a], results = ufun.fitLine(Xfit, Yfit)
        A, alpha = np.exp(b), a
        perc, dof = 0.975, len(Yfit)-2
        q = st.t.ppf(perc, dof)
        alpha_ciw = results.HC3_se[1] * q
        # R2 = results.rsquared
        # pval = results.pvalues[1]
        dict_byDate['alpha_allCells'].append(alpha)
        dict_byDate['alpha_allCells_ciw'].append(alpha_ciw)
        
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



def compute_Eeq(df, df_expo, h_ref = 300, inferred_exponent = 'median',
                XCol = 'H0_f_<_500', YCol = 'E_f_<_500',
                PLOT = False):
    
    codeX, codeY = dict_code[XCol], dict_code[YCol]
    codeXY = '_' + codeX + '_' + codeY
    
    df_expo['valid_global'+codeXY] = df_expo['valid_global'+codeXY] & (df_expo['alpha'+codeXY] < -1)
    df_expo_valid = df_expo[df_expo['valid_global'+codeXY] == True]
    valid_cells = df_expo_valid['cellID'].values
    Ncells = len(valid_cells)
    
    df, condCol = apm.makeCompositeCol(df, cols=['date'])
    CountByCond, CountByCell = apm.makeCountDf(df, condCol)
    df_f = df[df['cellID'].apply(lambda x : x in valid_cells)]
    
    Xfit, Yfit = np.log(df_f[XCol].values), np.log(df_f[YCol].values/1000)
    wd=1/(np.std(Xfit)) # **2
    we=1/(np.std(Yfit)) # **2
    params, results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
    a, b = params
    global_expo = a
    
    list_expos = df_expo_valid['alpha'+codeXY].values
    mean_expo = np.mean(list_expos)
    median_expo = np.median(list_expos)
    
    expos = {'mean' : mean_expo,
             'median' : median_expo,
             'global' : global_expo,
             }
    
    print(expos)
    
    # Group By
    # df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
    # df_fg = df_fg[[XCol]]
    # df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
    #                                       valCol = YCol, weightCol = 'ciw'+YCol, weight_method = 'ciw^2')
    # df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

    # dictFit = {'cellID':[], 
    #            'A'+codeXY:[], 'alpha'+codeXY:[], 'alpha_ciw'+codeXY:[], 
    #            'pval'+codeXY:[], 'R2'+codeXY:[], 'thickCV'+codeXY:[], 
    #            codeX+'_logmean':[], codeY+'_logmean':[], #'NLR_mean':[],
    #            'crit_NcompsMin':[crit_NcompsMin]*Ncells, 'valid_NcompsMin'+codeXY:[], 
    #            'crit_pvalFit':[crit_pvalFit]*Ncells, 'valid_pvalFit'+codeXY:[], 
    #            'crit_thickCV':[crit_thickCV]*Ncells, 'valid_thickCV'+codeXY:[],
    #            'activeCrits':[global_crit]*Ncells, 'valid_global'+codeXY:[],}
    A_list = []
    Eeq_list = []
    inferred_expo = expos[inferred_exponent]   
    
    for i in range(Ncells):
        cid = valid_cells[i]
        df_cell = df_f[df_f['cellID'] == cid]
        # Ncomps = len(df_cell)
        
        X = df_cell[XCol].values
        Y = df_cell[YCol].values/1000
        Xfit = np.log(np.copy(X))
        Yfit = np.log(np.copy(Y)) / inferred_expo
        
        wd=1/(np.std(Xfit)) # **2
        we=1/(np.std(Yfit)) # **2
        
        params, results = ufun.fitConstantTLS(Xfit, Yfit, wd=wd, we=we)
        [B] = params
        A = np.exp(B * inferred_expo)        
        E_eq = A * (h_ref**inferred_expo)
        
        A_list.append(A)
        Eeq_list.append(E_eq)
    
    return(inferred_expo, A_list, Eeq_list)




# %%% Fig 4A

#### 1. Settings

apm.setGraphicOptions(mode = 'print', 
                      palette = 'Set2', 
                      colorList = apm.cL_Set21)

# Save
SAVE = False
figSubDir = 'F4'
name = 'F4_A_1-1'

df = MecaData_Phy3
dates = ['23-02-16', '23-03-16', '23-04-26', '24-12-11']
# dates = ['23-02-16', '23-04-26', '24-12-11']
XCol = 'H0_f_<_500'
YCol = 'E_f_<_500'
suffix = '_f_<_500'

# crit_NcompsMin = 7
# crit_pvalFit = 0.4
# crit_thickCV = 0.025
crit_NcompsMin = 7
crit_pvalFit = 0.4
crit_thickCV = 0.0275
activeCrits = ['NcompsMin', 'pvalFit', 'thickCV']
dstDir = ''
figNameRoot = ''
modeFit = 'ODR'

ColoredCells = ['24-12-11_M1_P1_C12', 
                '24-12-11_M1_P1_C17', 
                '24-12-11_M1_P1_C18', 
                '24-12-11_M1_P1_C3', 
                '24-12-11_M2_P1_C10-1', 
                '24-12-11_M2_P1_C6']

#### 2. Filter

cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
df, condCol = apm.makeCompositeCol(df, cols=['cell type'])

Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df[XCol] < 1100),
           (df['normal field'] == 5),
           (df[YCol] <= 2e4),
           (df['valid' + suffix] == True), 
           ]

df_f = apm.filterDf(df, Filters)

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)


CID_longSeries = CountByCell[CountByCell['compCount'] >= 5].reset_index()['cellID'].values
Ncells = len(CID_longSeries)
global_crit = ''

for s in activeCrits:
    global_crit += s
    global_crit += '__'
global_crit = global_crit[:-2]

codeX, codeY = dict_code[XCol], dict_code[YCol]
codeXY = '_' + codeX + '_' + codeY

df_res, df_plot = compute_Eh_Exponent(df_f, XCol = XCol, YCol = YCol,
                        crit_NcompsMin = crit_NcompsMin,
                        crit_pvalFit = crit_pvalFit,
                        crit_thickCV = crit_thickCV,
                        activeCrits = activeCrits,
                        modeFit = modeFit)
df_res['date'] = df_res['cellID'].apply(lambda x : x.split('_')[0])
df_res.sort_values(by='cellID', ascending=True, inplace=True)
df_res = df_res.set_index(['cellID'])

## Consider only validated cells
df_res['valid_global'+codeXY] = df_res['valid_global'+codeXY] & (df_res['alpha'+codeXY] < -1)
CID_validCells = df_res[df_res['valid_global'+codeXY] == True].index.values
Ncells_valid = len(CID_validCells)
df_res = df_res[df_res['valid_global'+codeXY] == True]
df_f = df_f[df_f['cellID'].apply(lambda x : x in CID_validCells)]

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)


#### 3. Main Plot
# Initialize
fig, ax = plt.subplots(1, 1, figsize=(9/cm_in, 10/cm_in), layout='compressed')
win, hin = 0.275, 0.325
xin, yin = 0.69, 0.605
ax_in = ax.inset_axes([xin, yin, win, hin], zorder = 11)

ax.add_patch(plt.Rectangle((530, 13), 2500, 180, fc="white", 
                           zorder=2, alpha=0.75))
ax.text(x = 670, y = 97.5, s = 'Each cell exponent $\\alpha $', 
        fontsize=6, va='top', ha='left')

ax.set_xscale('log')
ax.set_yscale('log')

## Subplot 2.1
ax = ax
i0 = ufun.findFirst('24-12-11', df_res['date'].values)
index_color = np.array([i for i in range(Ncells_valid) if (CID_validCells[i] in ColoredCells)])
index_grey = np.array([i for i in range(Ncells_valid) if i not in index_color])

## Color palette
# ColPal = sns.color_palette("husl", Ncells_valid)
BrightPal = sns.color_palette("husl", len(index_color))
ColPal = []
k = 0
for i in range(Ncells_valid):
    if i in index_color:
        ColPal.append(BrightPal[ufun.findFirst(i, index_color)])
    else:
        ColPal.append('gray')

### For each cell, plot all compressions in that cell color
for i in range(Ncells_valid):
    # Data
    cid = CID_validCells[i]     
    c = ColPal[i]
    df_cell = df_f[df_f['cellID'] == cid]
    
    A         = df_res.loc[cid, 'A'+codeXY]
    alpha_fit = df_res.loc[cid, 'alpha'+codeXY]
    alpha_ciw = df_res.loc[cid, 'alpha_ciw'+codeXY]
    R2        = df_res.loc[cid, 'R2'+codeXY], 
    pval      = df_res.loc[cid, 'pval'+codeXY]
    valid     = df_res.loc[cid, 'valid_global'+codeXY]
    
    Xfit, Yfit = np.log(df_cell[XCol].values), np.log(df_cell[YCol].values/1000)
    Xplot = np.exp(np.linspace(min(Xfit)-0.1, max(Xfit)+0.1, 50))
    Yplot = A * Xplot**alpha_fit
    if i in index_color:
        # print(cid)
        s, alpha_plot, zorder = 40, 0.9, 8
        ax.plot(Xplot, Yplot, ls = '-', c = apm.lightenColor(c, 0.75), 
                lw = 2.0, alpha = alpha_plot*0.9, zorder = zorder-1)
    else:
        s, alpha_plot, zorder = 20, 0.55, 5
        
    sns.scatterplot(ax = ax, x=df_cell[XCol].values, y=df_cell[YCol].values/1000, 
                    marker = 'o', s = s, alpha = alpha_plot, color = c, zorder = zorder)
    
            # label =  f'$\\alpha$  = {alpha:.2f} $\\pm $ {alpha_ciw:.2f}')

### Plot the fit on all compressions
Xfit, Yfit = np.log(df_f[XCol].values), np.log(df_f[YCol].values/1000)

wd=1/(np.std(Xfit)) # **2
we=1/(np.std(Yfit)) # **2

params, results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
a, b = params
alpha, A = a, np.exp(b)
alpha_ciw, b_ciw = results.params_ciw
pval = results.pval
R2 = results.R2

Xplot = np.exp(np.linspace(min(Xfit)/10, max(Xfit)*10, 50))
Yplot = A * Xplot**alpha
ax.plot(Xplot, Yplot, ls = '-', c = 'dimgray', lw = 2, zorder=3,
         label =  f'All cells exponent\n$\\alpha$  = {alpha:.2f} $\\pm $ {alpha_ciw:.2f}')

#### Text
hM, hL, hH = ufun.getLogNDistributionDescriptors(df_f[XCol].values)
EM, EL, EH = ufun.getLogNDistributionDescriptors(df_f[YCol].values/1000)
A_Low = np.exp(b-(b_ciw/2))
A_High = np.exp(b+(b_ciw/2))
text_pval = apm.pval2text(pval, n_digits = 4, space = True)

print('---\n')
print('F4_A Long Series - By compression')
print(f"n = {CountByCond.loc['3T3', 'compCount']:.0f}, " + \
      f"N = {CountByCond.loc['3T3', 'cellCount']:.0f}, " + \
      f"M = {CountByCond.loc['3T3', 'manipsCount']:.0f}")
print(f'For {XCol} vs {YCol}\n')
print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
print(f'Power-law exponent & Ci : {alpha:.2f} +- {(alpha_ciw/2):.2f}')
print(f'Power-law constant & Ci : {A:.2e} [{A_Low:.2e}-{A_High:.2e}]')
print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')
print('---------------')
    

# Format
# ax.legend(loc = 'lower left', handlelength=3.5)#.set_visible(False)
ax.set_ylabel('$E$ (kPa)', labelpad=0.5)
ax.set_xlabel('$H_0$ (nm)', labelpad=0.5)
ax.grid(visible=True, which='major', axis='both')
# ax.set_xlim([40, 3000])
# ax.set_ylim([0.2, 50])
ax.set_xlim([40, 3000])
ax.set_ylim([0.7, 110])


#### 4. Subplot Inset
### Plot the exponents distribution (power law slopes)
ax = ax_in
expos = df_res['alpha'+codeXY].values
Q1, med, Q3 = np.percentile(expos, 25), np.percentile(expos, 50), np.percentile(expos, 75)
# ax.errorbar([0], [med], 
#             ls='', marker='_', markerfacecolor=(1, 1, 1, 0.0),
#             mec = 'k', mew = 0.75, ms = 30,
#             xerr=None, yerr=[[med-Q1], [Q3-med]],
#             ecolor = 'k', elinewidth=0.75, capsize=6, zorder=5)
# sns.boxplot(data = df_res, ax = ax, y='alpha'+codeXY, 
#             width=0.8, color='.9', showfliers = False,
#             boxprops={"facecolor": (.7, .7, .7, .9), "edgecolor": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
#             medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
#             whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
#             capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
#             )
ax.axhline(med, color = 'darkred', lw=2.0, alpha = 0.8)
print(med)
sns.swarmplot(data = df_res, ax = ax, y='alpha'+codeXY,
              size = 5.0, edgecolor='w', linewidth=0.25, hue = 'cellID', 
              palette = ColPal, legend=False, zorder=4)
ax.set_ylim([-3.0, -1])
ax.yaxis.set_tick_params(length=3, pad=0.75)
ax.xaxis.set_tick_params(length=0, pad=0.75)
### Format
ax.grid(axis='y')
ax.set_ylabel('')

fig.get_layout_engine().set(w_pad=2e-2, h_pad=2e-2, 
                            hspace=5e-3, wspace=3e-3)

plt.show()


h_ref = 300
inferred_expo, A_list, Eeq_list = compute_Eeq(df_f, df_res.reset_index(), h_ref = h_ref, 
                               inferred_exponent = 'median',
                               XCol = 'H0_f_<_500', YCol = 'E_f_<_500')



# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%% S4_AB

#### 1. Settings

apm.setGraphicOptions(mode = 'print', 
                      palette = 'Set2', 
                      colorList = apm.cL_Set21)

# Save
SAVE = True
figDir = 'C:/Users/josep/Desktop/Seafile/PapierDensité/FiguresSupp'
figSubDir = 'S4'
name = 'S4_AB_1-0'

nCols = 6
nRows = Ncells_valid//nCols + 1
fig, axes = plt.subplots(nRows, nCols, figsize=(2.5*nCols/cm_in, 3*nRows/cm_in),
                         sharex=True, sharey=True, layout='compressed')

for i in range(Ncells_valid):
    iR = i//nCols
    iC = i%nCols
    
    cid = CID_validCells[i]
    df_cell = df_f[df_f['cellID'] == cid]
    # Ncomps = len(df_cell)
    
    X = df_cell[XCol].values
    Y = df_cell[YCol].values/1000
    
    A = A_list[i]
    E_eq = Eeq_list[i]
    
    ax = axes[iR, iC]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(X, Y, ls='', 
            marker='o', ms=4, mec='w', 
            mew=0.1, alpha=0.5, zorder=6)
    Xplot = np.array([40,2100])
    Yplot = A * (Xplot**inferred_expo)
    ax.plot(Xplot, Yplot, ls='-')
    
    ax.axvline(h_ref, ls='-', lw=1.5, color='dimgray')
    ax.plot([h_ref], [E_eq], 'kx', 
            label='$E_{eq}$=' + f'{E_eq:.2f}kPa')
    
    ax.legend(loc='upper right', fontsize=6, handlelength = 1)
    ax.set_xlim([50, 2000])
    ax.set_ylim([0.5, 100])
    ax.grid()
    
    


# ax.grid()

fig2, axes2 = plt.subplots(1, 2, figsize=(10/cm_in, 8/cm_in),
                         layout='compressed')
ax=axes2[0]
ax.set_title('Computing $E_{eq}$ at 300 nm\nExample on 4 cells')

for k, i in enumerate([2, 10, 6, 30]):
    color = apm.cL_Set2[k]
    cid = CID_validCells[i]
    df_cell = df_f[df_f['cellID'] == cid]
    # Ncomps = len(df_cell)
    
    X = df_cell[XCol].values
    Y = df_cell[YCol].values/1000
    
    A = A_list[i]
    E_eq = Eeq_list[i]
    
    ax = ax
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(X, Y, ls='', 
            marker='o', ms=4, color=color,
            mec='w', mew=0.1, alpha=1, zorder=3)
    Xplot = np.array([40,2100])
    Yplot = A * (Xplot**inferred_expo)
    ax.plot(Xplot, Yplot, ls='-', lw=1.0, color=apm.lightenColor(color, 0.8), zorder=4)
    ax.plot([h_ref], [E_eq], marker='x', color=apm.lightenColor(color, 0.8),
            label=f'{E_eq:.2f}kPa', zorder=5)
    
ax.axvline(h_ref, ls='-', lw=1.5, color='dimgray', zorder=2)
ax.legend(title='$E_{eq}$ at \n300 nm', loc='upper right', fontsize=6, handlelength = 1)
ax.set_xlim([50, 2000])
ax.set_ylim([0.5, 100])
ax.set_ylabel('$E$ (kPa)', labelpad=0.5)
ax.set_xlabel('$H_0$ (nm)', labelpad=0.5)
ax.grid()


ax = axes2[1]
ax.set_xlim([-0.5, +0.5])
log_Eeq     = np.log(Eeq_list)
logmean_Eeq = np.mean(log_Eeq)
logstd_Eeq  = np.std(log_Eeq)
print(logmean_Eeq, np.exp(logmean_Eeq), )
print(np.exp(logmean_Eeq - logstd_Eeq), np.exp(logmean_Eeq + logstd_Eeq))
print(logstd_Eeq, np.exp(logstd_Eeq), np.exp(logstd_Eeq)**0.5)
data = pd.DataFrame({'Eeq':Eeq_list,})
sns.swarmplot(ax=ax, data=data, y='Eeq', color=(0, 0, 0, 0.5), size=6)
f = 1
colorcross = 'darkred'
ax.errorbar([0], np.exp(logmean_Eeq), 
            ls='', marker='_', markerfacecolor='k',
            mec = colorcross, mew = 1.5*f, ms = 12*f,
            xerr=None, 
            yerr=[[np.exp(logmean_Eeq) - np.exp(logmean_Eeq - logstd_Eeq)], 
                  [np.exp(logmean_Eeq + logstd_Eeq) - np.exp(logmean_Eeq)]],
            ecolor = colorcross, elinewidth=1.5*f, capsize=3*f, zorder=10)
ax.set_ylim([0.5, 20])

ax.grid(axis='y')
ax.set_yscale('log')
ax.set_ylabel('$E_{eq}$ (kPa)', labelpad=0.5)
ax.set_title('Complete distibution\nof $E_{eq}$ at 300 nm')
# ax.grid()

plt.show()

if SAVE:
    ufun.archiveFig(fig2, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig2, name = name, ext = '.png', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %% --------

figDir = 'C:/Users/josep/Desktop/Seafile/PapierDensité/FiguresSupp'

# %% Supp Figure 1 & 1bis

# %%% Normal Distribution

apm.setGraphicOptions(mode = 'print', 
                      palette = 'Set2', 
                      colorList = apm.cL_Set21)

# Save
SAVE = False
figDir = 'C:/Users/josep/Desktop/Seafile/PapierDensité/FiguresSupp'
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
df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [HCol], aggFun = 'mean')

# Group By for E<400
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = ECol, weightCol = 'ciw' + ECol, weight_method = 'ciw^2')
df_fgw2[ECol + '_wAvg'] /= 1000


#### Plots
X = df_fg[df_fg['drug'] == 'dmso'][HCol].values
Y = df_fgw2[df_fgw2['drug'] == 'dmso'][ECol + '_wAvg'].values
titles = ['$H_{500}$ Q-Q plot', '$E_{500}$ Q-Q plot']

fig, axes = plt.subplots(2, 1, figsize=(6/cm_in, 12/cm_in))

for k, data in enumerate([X, Y]): # +'_wAvg'
    
    data_lin = data
    data_log = np.log(data)
    
    ax = axes[k]
    ax.axline((0, 0), slope=1, color="k", linestyle='-.', linewidth=1, zorder=6)
    
    data=data_lin
    shap_stat, shap_pval = shapiro(data)
    sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = apm.cL_Set21[1], 
              markeredgecolor = 'None', markersize=4, alpha=0.75)
    ax.plot([], [], label=f'N: {shap_pval:.2f}', ls='', marker='o', 
            markerfacecolor = apm.cL_Set21[1], markeredgecolor = 'None', markersize=5)
    
    data=data_log
    shap_stat, shap_pval = shapiro(data)
    sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = apm.cL_Set21[0], 
              markeredgecolor = 'None', markersize=4, alpha=0.75)
    ax.plot([], [], label=f'LogN: {shap_pval:.2f}', ls='', marker='o', 
            markerfacecolor = apm.cL_Set21[0], markeredgecolor = 'None', markersize=5)
    
    ax.legend(title_fontsize=6, title = 'Shapiro–Wilk\np-values', loc='lower right')
    
    ax.grid()
    ax.set_aspect('equal')
    ax.set_xlim([-3.5,3.5])
    ax.set_ylim([-3.5,3.5])
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.text(-3.2, 3.2, titles[k], va='top', ha='left', 
            fontsize=9.0, backgroundcolor='w')
    
plt.show()

# #### 03 - Normality test H0

# ax = axes[0]

# data_lin = df_fg[df_fg['drug'] == 'dmso']['bestH0'].values
# data_log = np.log(df_fg[df_fg['drug'] == 'dmso']['bestH0'].values)

# data=data_lin
# shap_stat, shap_pval = shapiro(data)
# sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = apm.cL_Set21[5], markeredgecolor = 'None', markersize=6)
# ax.plot([], [], label=f'N: {shap_pval:.2f}', ls='', marker='o', 
#         markerfacecolor = apm.cL_Set21[5], markeredgecolor = 'None', markersize=4)
# # ax.grid()
# # ax.set_title('Q-Q plots for $H_0$')


# #### 04 - normality test E

# ax = axes[1]

# data_lin = df_fgw2[df_fgw2['drug'] == 'dmso'][ECol + '_wAvg'].values
# data_log = np.log(df_fgw2[df_fgw2['drug'] == 'dmso'][ECol + '_wAvg'].values)

# data=data_lin
# shap_stat, shap_pval = shapiro(data)
# sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = apm.cL_Set21[5], markeredgecolor = 'None', markersize=6)
# ax.plot([], [], label=f'N: {shap_pval:.2f}', ls='', marker='o', 
#         markerfacecolor = apm.cL_Set21[5], markeredgecolor = 'None', markersize=4)
# # ax.grid()
# # ax.set_title('Q-Q plots for $E_{500}$')

# #### Part 3

# #### 05 - Log-Normality test H0

# ax = axes[0]

# data_lin = df_fg[df_fg['drug'] == 'dmso']['bestH0'].values
# data_log = np.log(df_fg[df_fg['drug'] == 'dmso']['bestH0'].values)

# data=data_log
# shap_stat, shap_pval = shapiro(data)
# ax.axline((0, 0), slope=1, color="k", linestyle='--', linewidth=1, zorder=6)

# sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = apm.cL_Set21[0], 
#           markeredgecolor = 'None', markersize=3)
# ax.plot([], [], label=f'LogN: {shap_pval:.2f}', ls='', marker='o', 
#         markerfacecolor = apm.cL_Set21[0], markeredgecolor = 'None', markersize=5)


# #### 06 - Log-normality test E

# ax = axes[1]

# data_lin = df_fgw2[df_fgw2['drug'] == 'dmso'][ECol + '_wAvg'].values
# data_log = np.log(df_fgw2[df_fgw2['drug'] == 'dmso'][ECol + '_wAvg'].values)

# data=data_log
# shap_stat, shap_pval = shapiro(data)
# ax.axline((0, 0), slope=1, color="k", linestyle='--', linewidth=1, zorder=6)
# ax.set_aspect('equal')
# ax.set_xlim([-3.5,3.5])
# ax.set_ylim([-3.5,3.5])
# sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = apm.cL_Set21[1], 
#           markeredgecolor = 'None', markersize=3)
# ax.plot([], [], label=f'LogN: {shap_pval:.2f}', ls='', marker='o', 
#         markerfacecolor = apm.cL_Set21[1], markeredgecolor = 'None', markersize=5)
# ax.legend(title_fontsize=6, title = 'Shapiro–Wilk\np-values', loc='lower right')
# ax.grid()
# ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
# ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
# ax.text(-3, 2.5, '', va='center', ha='left', 
#         fontsize=10.0, backgroundcolor='w')

# # Show
# plt.tight_layout()
# plt.show()

# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%% Choice of F < 500

# %%%% Other df

path = "C:\\Users\\josep\\Documents\\MagneticPincherData\\Main_EvH_Curve_Data.csv"
Julien_df = pd.read_csv(path)

print('Dates')
print([x for x in Julien_df['date'].unique()])
print('')

print('Manips')
print([x for x in Julien_df['manipID'].unique()])
print('')

print('Cell types')
print([x for x in Julien_df['cell type'].unique()])
print('')

print('Cell subtypes')
print([x for x in Julien_df['cell subtype'].unique()])
print('')

print('Drugs')
print([x for x in Julien_df['drug'].unique()])
print('')

print('Substrates')
print([x for x in Julien_df['substrate'].unique()])
print('')

print('Resting Fields')
print([x for x in Julien_df['normal field'].unique()])
print('')

# CountByCond, CountByCell =apm.makeCountDf(Julien_df, 'date')


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

# V1
# phyTask = '23-02-16_M1 & 23-02-23_M1 & 23-02-23_M3 & 23-03-08_M3 & 23-03-16_M1 & 23-03-17_M4' # Dmso & none 1/4
# phyTask += ' & 23-04-20_M1 & 23-04-20_M4 & 23-04-20_M5 & 23-04-26_M2 & 23-04-28_M1 & 23-07-17_M3' # Dmso & none 2/4
# phyTask += ' & 23-07-17_M4 & 23-07-17_M6 & 23-07-20_M2 & 23-09-06_M3 & 23-09-11_M1 & 23-09-19_M1 & 23-11-26_M2 & 23-12-03_M1' # Dmso & none 3/4
# phyTask += ' & 24-07-04_M2 & 24-07-04_M6' # Dmso & none 4/4
# phyTask += ' & 23-03-09_M4' # Pattern sizes JV
# phyTask += ' & 24-12-11'

# V2
phyTask = '23-02-16_M1 & 23-03-16_M1 & 23-03-17_M4 & 23-04-20_M1 & 23-04-20_M4 & '
phyTask += '23-04-20_M5 & 23-04-26_M2 & 23-04-28_M1 & 23-07-17_M3 & '
phyTask += '23-07-20_M2 & 23-09-06_M3 & 23-09-19_M1 & 23-12-03_M1 & 24-07-04_M2'

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
        yResid = yPredict - y

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
        yResid = np.ones(len(h), dtype=float) * np.nan
        
    res = (error, R2, Chi2, yResid)
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

list_Fmax = np.arange(100, 1100, 50)
# list_D = Id_comps[:][2]
all_Chi2 = []
all_R2   = []
all_M    = []
all_Mf   = []
all_Msq  = []
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
    list_M = []
    list_Mf = []
    list_Msq = []
    for k in range(len(Comps)): # len(Comps)
        D = Id_comps[k][2]
        h, f = Comps[k]
        index = (f < Fmax)
        h_fit, f_fit = h[index], f[index]
        results = fitChadwick_hf(h_fit, f_fit, D, err_chi2)
        error, r2, chi2, resid = results

        if (not error) and (chi2 > 0) and (r2 < 1):
            list_Chi2.append(chi2)
            list_R2.append(r2)
            N = len(resid)
            n = N//3
            M1, M2, M3 = np.median(resid[:n]), np.median(resid[n:2*n]), np.median(resid[2*n:])
            list_M.append((M1, M2, M3))
            
            F_3 = np.max(f_fit)/3
            r1 = resid[f_fit<F_3], 
            r2 = resid[(f_fit>=F_3) & (f_fit<2*F_3)]
            r3 = resid[(f_fit>=2*F_3)]
            Mf1 = np.median(r1)
            Mf2 = np.median(r2)
            Mf3 = np.median(r3)
            list_Mf.append((Mf1, Mf2, Mf3))
            
            Msq1 = Mf1 / len(r1)**0.5
            Msq2 = Mf2 / len(r2)**0.5
            Msq3 = Mf3 / len(r3)**0.5
            list_Msq.append((Msq1, Msq2, Msq3))
            
    all_Chi2.append(list_Chi2)
    all_R2.append(list_R2)
    all_M.append(list_M)
    all_Mf.append(list_Mf)
    all_Msq.append(list_Msq)
    
    
# %%%% Save

list_Fmax = np.array(list_Fmax).astype(int).tolist()
for x in all_R2:
    x = np.array(x).astype(float).tolist()
for x in all_Chi2:
    x = np.array(x).astype(float).tolist()
    
dstPath = os.path.join("C:\\Users\\josep\\Desktop\\Seafile\\PapierDensité\\DraftsFigs", 'S1')
ufun.list2json(list_Fmax, dstPath, 'list_Fmax_V2')
ufun.list2json(all_R2, dstPath, 'all_R2_V2')
ufun.list2json(all_Chi2, dstPath, 'all_Chi2_V2')
ufun.list2json(all_M, dstPath, 'all_M_V2')
ufun.list2json(all_Mf, dstPath, 'all_Mf_V2')
ufun.list2json(all_Msq, dstPath, 'all_Msq_V2')


# %%%% Open

srcPath = os.path.join("C:\\Users\\josep\\Desktop\\Seafile\\PapierDensité\\DraftsFigs", 'S1')
list_Fmax = ufun.json2list(srcPath, 'list_Fmax_V2')
all_R2 = ufun.json2list(srcPath, 'all_R2_V2')
all_Chi2 = ufun.json2list(srcPath, 'all_Chi2_V2')
all_M = ufun.json2list(srcPath, 'all_M_V2')
all_Mf = ufun.json2list(srcPath, 'all_Mf_V2')
all_Msq = ufun.json2list(srcPath, 'all_Msq_V2')
    
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

avg_Med = np.array([np.nanmean(np.array(list_M), axis=0) for list_M in all_M])
avg_MedF = np.array([np.nanmean(np.array(list_Mf), axis=0) for list_Mf in all_Mf])
avg_MedSq = np.array([np.nanmean(np.array(list_Msq), axis=0) for list_Msq in all_Msq])


# %%%% 4. Plot the results

apm.setGraphicOptions(mode = 'print', 
                      palette = 'Set2', 
                      colorList = apm.cL_Set21)

# Save
SAVE = False
figDir = 'C:/Users/josep/Desktop/Seafile/PapierDensité/FiguresSupp'
figSubDir = 'S1'
name = 'S1_Choice_500pN'

c1 = apm.cL_Set2[0]
c2 = apm.cL_Set2[1]

fig, ax = plt.subplots(1, 1, figsize=(6/cm_in, 6/cm_in))#, layout='compressed')
ax2 = ax.twinx()
ax.plot(list_Fmax, median_R2, color=apm.lightenColor(c1, 0.9), lw=2)
ax.plot(list_Fmax, D1_R2, ls=':', color=apm.lightenColor(c1, 1.1), lw=1.25)
ax.plot(list_Fmax, D9_R2, ls='--', color=apm.lightenColor(c1, 1.1), lw=1.25)
ax2.plot(list_Fmax, median_Chi2, color=apm.lightenColor(c2, 0.9), lw=2)
ax2.plot(list_Fmax, D1_Chi2, ls=':', color=apm.lightenColor(c2, 1.1), lw=1.25)
ax2.plot(list_Fmax, D9_Chi2, ls='--', color=apm.lightenColor(c2, 1.1), lw=1.25)

ax.set_xlabel('Upper Bound of F')
ax.set_title(' ')
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

# %%%% 5. Plot better results

apm.setGraphicOptions(mode = 'print', 
                      palette = 'Set2', 
                      colorList = apm.cL_Set21)

# Save
SAVE = True
figDir = 'C:/Users/josep/Desktop/Seafile/PapierDensité/FiguresSupp'
figSubDir = 'S1'
name = 'S1_Choice_500pN_V2'

c1 = apm.cL_Set2[0]
c2 = apm.cL_Set2[1]
c3 = apm.cL_Set2[2]

fig, axes = plt.subplots(1, 2, figsize=(12/cm_in, 6/cm_in))#, layout='compressed')
ax = axes[0]
ax.plot(list_Fmax, avg_MedF[:, 0], color=c1, lw=2, label=r'F < 1/3.$F_{max}$')
ax.plot(list_Fmax, avg_MedF[:, 1], color=c2, lw=2, label=r'1/3.$F_{max}$ < F < 2/3.$F_{max}$')
ax.plot(list_Fmax, avg_MedF[:, 2], color=c3, lw=2, label=r'F > 2/3.$F_{max}$')

ax.set_xlabel('Selected $F_{max}$')
ax.set_title(' ')
ax.set_ylabel(r'Residuals')

ax.axhline(0, color='gray', lw=1, ls='-')
ax.axvline(500, color='gray', lw=1, ls='-.')
ax.set_xlim([0, 1100])
# ax.set_ylim([0, 1.05])
ax.set_xticks([k for k in range(100, 1100, 200)])
ax.xaxis.set_tick_params(rotation=0)
ax.legend(fontsize=5)

ax= axes[1]
ax.plot(list_Fmax[1:], avg_MedSq[1:, 0], color=c1, lw=2, label=r'F < 1/3.$F_{max}$')
ax.plot(list_Fmax[1:], avg_MedSq[1:, 1], color=c2, lw=2, label=r'1/3.$F_{max}$ < F < 2/3.$F_{max}$')
ax.plot(list_Fmax[1:], avg_MedSq[1:, 2], color=c3, lw=2, label=r'F > 2/3.$F_{max}$')

ax.set_xlabel('Selected $F_{max}$')
ax.set_title(' ')
ax.set_ylabel(r'$\langle Resids \rangle \ /\ \sqrt{N}$')

ax.axhline(0, color='gray', lw=1, ls='-')
ax.axvline(500, color='gray', lw=1, ls='-.')
ax.set_xlim([0, 1100])
# ax.set_ylim([0, 1.05])
ax.set_xticks([k for k in range(100, 1100, 200)])
ax.xaxis.set_tick_params(rotation=0)
ax.legend(fontsize=5)

# ax.yaxis.set_tick_params(length=3, pad=0.75)
# ax.xaxis.set_tick_params(length=0, pad=0.75)

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
figDir = 'C:/Users/josep/Desktop/Seafile/PapierDensité/FiguresSupp'
figSubDir = 'S1'
name = 'S1_Identity_H500'
# palette = [apm.cL_Set1[0], apm.cL_Set1[1], apm.cL_Set1[2]]
palette = apm.cL_Set2

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
for co in df_f[condCol].unique():
    df_fm  = df_f[df_f[condCol] == co]
    CbCond, CbCells = apm.makeCountDf(df_fm, condCol)
    M = df_fm['manipID'].values[0]
    M_cells = CbCond['cellCount'].values[0]
    M_comps = CbCond['compCount'].values[0]
    print(M, M_cells, M_comps)

#### Apply log
dfLOG_f = apm.filterDf(df, Filters)
dfLOG_f['cellNum'] = dfLOG_f['cellCode'].apply(lambda x : int(x[1:]))
dfLOG_f[parameter] = np.log10(dfLOG_f[parameter])
dfLOG_f.sort_values(by='manipID', ascending=True, inplace=True)
Manip_list = dfLOG_f['manipID'].unique()


# Group By
dfLOG_fg = apm.dataGroup(dfLOG_f, groupCol = 'cellID', idCols = [condCol, 'cellCode', 'cellNum'], 
                         numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
# Merge
dfLOG_m = pd.merge(left=dfLOG_f, right=dfLOG_fg, on = 'cellID', how='left', 
                   suffixes = (None, '_grouped'))

for mid in Manip_list:
    mid_index_fg = dfLOG_fg[dfLOG_fg['manipID'] == mid].index
    mid_index_m = dfLOG_m[dfLOG_m['manipID'] == mid].index
    dfLOG_fg.loc[mid_index_fg, parameter + '_pop_spread'] = dfLOG_fg.loc[mid_index_fg, parameter] / np.mean(dfLOG_fg.loc[mid_index_fg, parameter].values)
    dfLOG_m.loc[mid_index_m, parameter + '_indiv_spread'] = dfLOG_m.loc[mid_index_m, parameter] / dfLOG_m.loc[mid_index_m, parameter + '_grouped']


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
fig = plt.figure(figsize=(12/cm_in, 6/cm_in))
spec = fig.add_gridspec(1, 2)


#### Plot 1
ax = fig.add_subplot(spec[0])
ax.set_ylim([0.7, 1.3])

df_m['cellCode'] = df_m['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])

# sns.swarmplot(ax = ax, data = dfLOG_m, x = 'date', y = parameter + '_indiv_spread', # hue = 'cellCode', 
#               s=2, edgecolor='w', linewidth=0) # 
sns.violinplot(ax = ax, data = dfLOG_m, x = 'date', y = parameter + '_indiv_spread', 
               hue = 'date', inner="quart", palette = palette,
               edgecolor='w', order = co_order, linewidth=1.1) # s=2, 
for xt in [0, 1, 2]:
    ax.text(xt, 1.265, f'{CV_per_cell_avg[xt]*100:.1f}%', fontsize=7,
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
sns.violinplot(ax = ax, data = dfLOG_fg, palette = palette,
               x = 'date', y = parameter + '_pop_spread', hue = 'date', inner="quart", # hue = 'cellID', 
               order = co_order, edgecolor='w', linewidth=1.1, zorder=2) #, s=8) #, legend=False)
for xt in [0, 1, 2]:
    ax.text(xt, 1.265, f'{CV_population[xt]*100:.1f}%', fontsize=7,
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
figDir = 'C:/Users/josep/Desktop/Seafile/PapierDensité/FiguresSupp'
figSubDir = 'S1'
name = 'S1_Identity_E500'
palette = apm.cL_Set2

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

#### Apply log
dfLOG_f = apm.filterDf(df, Filters)
dfLOG_f['cellNum'] = dfLOG_f['cellCode'].apply(lambda x : int(x[1:]))
dfLOG_f[parameter] = np.log10(dfLOG_f[parameter])
dfLOG_f.sort_values(by='manipID', ascending=True, inplace=True)
Manip_list = dfLOG_f['manipID'].unique()


# Group By
dfLOG_fg = apm.dataGroup(dfLOG_f, groupCol = 'cellID', idCols = [condCol, 'cellCode', 'cellNum'], 
                         numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
# Merge
dfLOG_m = pd.merge(left=dfLOG_f, right=dfLOG_fg, on = 'cellID', how='left', 
                   suffixes = (None, '_grouped'))

for mid in Manip_list:
    mid_index_fg = dfLOG_fg[dfLOG_fg['manipID'] == mid].index
    mid_index_m = dfLOG_m[dfLOG_m['manipID'] == mid].index
    dfLOG_fg.loc[mid_index_fg, parameter + '_pop_spread'] = dfLOG_fg.loc[mid_index_fg, parameter] / np.mean(dfLOG_fg.loc[mid_index_fg, parameter].values)
    dfLOG_m.loc[mid_index_m, parameter + '_indiv_spread'] = dfLOG_m.loc[mid_index_m, parameter] / dfLOG_m.loc[mid_index_m, parameter + '_grouped']


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
fig = plt.figure(figsize=(12/cm_in, 6/cm_in))
spec = fig.add_gridspec(1, 2)


#### Plot 1
ax = fig.add_subplot(spec[0])
ax.set_ylim([0.7, 1.3])

df_m['cellCode'] = df_m['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])

# sns.swarmplot(ax = ax, data = dfLOG_m, x = 'date', y = parameter + '_indiv_spread', # hue = 'cellCode', 
#               s=2, edgecolor='w', linewidth=0) # 
sns.violinplot(ax = ax, data = dfLOG_m, x = 'date', y = parameter + '_indiv_spread', 
               hue = 'date', inner="quart", palette = palette,
               edgecolor='w', order = co_order, linewidth=1.1) # s=2, 
for xt in [0, 1, 2]:
    ax.text(xt, 1.265, f'{CV_per_cell_avg[xt]*100:.1f}%', fontsize=7, 
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
sns.violinplot(ax = ax, data = dfLOG_fg,  palette = palette,
               x = 'date', y = parameter + '_pop_spread', hue = 'date', inner="quart", # hue = 'cellID', 
               order = co_order, edgecolor='w', linewidth=1.1, zorder=2) #, s=8) #, legend=False)
for xt in [0, 1, 2]:
    ax.text(xt, 1.265, f'{CV_population[xt]*100:.1f}%', fontsize=7,
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


# %%% DH - Before-After


# Save
SAVE = True
figSubDir = 'S1'
name = 'DH_Before-After' # 

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
fig, ax = plt.subplots(1, 1, figsize=(8/cm_in, 6/cm_in))
# sns.swarmplot(ax=ax, data = df_f, x='cell type', y='Dh_BeforeAfter', size=1)
ax.hist(df_f['Dh_BeforeAfter'].values, bins=40, color='dimgray', zorder=3)

median = np.median(df_f['Dh_BeforeAfter'].values)
ax.axvline(median, color='darkorange', ls='--', lw=1,
           label=f'Median = {median:.1f} nm', zorder=3)
ax.legend(handlelength = 1.25)
ax.grid(zorder=1)
ax.set_xlabel(r'$\Delta H_{5mT}$ (nm)')
ax.set_ylabel('# compressions')
ax.set_title(r'$\Delta H_{5mT}$ - Before/after compression')

# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')
    
# %%% DH/H - Before-After/Before

# Save
SAVE = True
figSubDir = 'S1'
name = 'DH_Before-After_ratio' # 

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

df_f['ratioDh_Hi'] = df_f['Dh_BeforeAfter'].values/df_f['previousThickness'].values

# Order
# co_order = []

# Group By
# df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
# df_fg = df_fg[[XCol]]
# df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
#                                       valCol = YCol, weightCol = 'ciw'+YCol, weight_method = 'ciw^2')
# df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

#### Plot
fig, ax = plt.subplots(1, 1, figsize=(6/cm_in, 6/cm_in))
# sns.swarmplot(ax=ax, data = df_f, x='cell type', y='Dh_BeforeAfter', size=1)
ax.hist(df_f['ratioDh_Hi'].values, 
        bins=300, color='dimgray', zorder=3)

median = np.median(df_f['ratioDh_Hi'].values)
ax.axvline(median, color='darkorange', ls='-.', lw=1,
           label=f'Median = {median:.3f}', zorder=3)
print(f'Median = {median:.3f}')
# ax.legend(handlelength = 1.25)
ax.grid(zorder=1)
ax.set_xlim(-1, 1)
ax.set_xlabel(r'$\Delta H/H_{init}$')
ax.set_ylabel('N compressions')
ax.set_title('Thickness at 5mT\n(H_after - H_before) / H_before')

# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%% DH - Precompression

# Save
SAVE = True
figSubDir = 'S1'
name = 'DH_Precompression' # 

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
fig, ax = plt.subplots(1, 1, figsize=(8/cm_in, 6/cm_in))
# sns.swarmplot(ax=ax, data = df_f, x='cell type', y='Dh_BeforeAfter', size=1)
ax.hist(df_f['Dh_Precomp'].values, bins=40, color='dimgray', zorder=3)

median = np.median(df_f['Dh_Precomp'].values)
ax.axvline(median, color='darkorange', ls='--', lw=1,
           label=f'Median = {median:.1f} nm', zorder=3)
ax.legend(handlelength = 1.25)
ax.grid(zorder=1)
ax.set_xlabel(r'$\Delta H$ (nm)')
ax.set_ylabel('# compressions')
ax.set_title(r'$\Delta H$ - Initial relaxation')

# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%% DH/H - Precompression

apm.setGraphicOptions(mode = 'print', 
                      palette = 'Set2', 
                      colorList = apm.cL_Set21)


# Save
SAVE = True
figSubDir = 'S1'
figDir = 'C:/Users/josep/Desktop/Seafile/PapierDensité/FiguresSupp'
name = 'DH_Precompression_ratio' # 

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
fig, ax = plt.subplots(1, 1, figsize=(5/cm_in, 5/cm_in))
# sns.swarmplot(ax=ax, data = df_f, x='cell type', y='Dh_BeforeAfter', size=1)
ax.hist(df_f['Dh_Precomp'].values/df_f['previousThickness'].values, 
        bins=360, color='dimgray', zorder=3)

median = np.median(df_f['Dh_Precomp'].values/df_f['previousThickness'].values)
ax.axvline(median, color='darkred', ls='-', lw=1,
           label=f'Median = {median:.2f}', zorder=3)
ax.legend(handlelength = 1.25)
ax.grid(zorder=1)
ax.set_xlim([-0.0, 1.2])
ax.set_xlabel(r'$\Delta H_{relax}/H_{init}$')
ax.set_ylabel('N compressions')
# ax.set_title('Initial relaxation / Initial thickness')

# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')




# %%% Fig S1EF - DH/H Before-After/Before + Peak delay

# Save
SAVE = True
figDir = 'C:/Users/josep/Desktop/Seafile/PapierDensité/FiguresSupp'
figSubDir = 'S1'
name = 'S1_Dh-h' # 

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

df_f['ratioDh_Hi'] = df_f['Dh_BeforeAfter'].values/df_f['previousThickness'].values

color_med = 'darkred'

#### Plot
fig, axes = plt.subplots(2, 1, figsize=(6/cm_in, 10/cm_in), layout='constrained')
ax = axes[0]
# sns.swarmplot(ax=ax, data = df_f, x='cell type', y='Dh_BeforeAfter', size=1)
ax.hist(df_f['ratioDh_Hi'].values, 
        bins=300, color='gray', zorder=3)

median = np.median(df_f['ratioDh_Hi'].values)
# ax.axvline(median, color='darkred', ls='-', lw=1,
#            label=f'Median = {median:.3f}', zorder=3)
# ax.text(x=0.2, y=200, s='Median\n' + f'{median:.3f} nm', size=7, c=color_med)
# ax.legend(handlelength = 1.25)
## V1
# ax.text(x=0.10, y=250, s=r'$\Delta H$ = ', size=7, c='k', va='center')
# ax.text(x=0.40, y=250, s=r'$H_{final}$', size=7, c='red', va='center')
# ax.text(x=0.65, y=250, s=r' - ', size=7, c='k', va='center')
# ax.text(x=0.75, y=250, s=r'$H_{init}$', size=7, c='deepskyblue', va='center')
## V2
x0, y0 = 0.3, 270
ax.text(x=x0+0.15, y=y0, s=r'$\Delta H$ = ', size=7, c='k', va='center')
ax.text(x=x0, y=y0-20, s=r'$H_{final}$', size=7, c='red', va='center')
ax.text(x=x0+0.23, y=y0-20, s=r' $-$ ', size=7, c='k', va='center')
ax.text(x=x0+0.37, y=y0-20, s=r'$H_{init}$', size=7, c='deepskyblue', va='center')
ax.add_patch(plt.Rectangle((x0-0.03, y0-35), 0.65, 55, fc="white",
                           zorder=2, alpha=0.75))

ax.grid()
ax.axvline(0, color='k', ls='-', lw=0.75, zorder=5)
ax.axhline(0, color='k', ls='-', lw=1, zorder=5)
ax.set_xlim(-1, 1)
ax.set_xlabel(r'$\Delta H/H_{init}$ (ratio)', labelpad=0.6)
ax.set_ylabel('N compressions', fontsize=6, labelpad=0.6)
# ax.set_title('Thickness at 5mT\n(H_after - H_before) / H_before')

ax = axes[1]
ax.hist(df_f['peakDelay'].values, bins=30, color='gray', zorder=3)

median = np.median(df_f['peakDelay'].values)
# ax.axvline(median, color='darkred', ls='-', lw=1,
#            label=f'Median = {median:.2f} s', zorder=3)
# ax.text(x=0.2, y=225, s='Median\n' + f'{median*1e3:.0f} ms', size=7, c=color_med)
# ax.legend(handlelength = 1.25, loc='upper left')
ax.grid()
ax.axvline(0, color='k', ls='-', lw=0.75, zorder=5)
ax.axhline(0, color='k', ls='-', lw=1, zorder=5)
ax.set_xlabel(r'Force-thickness peak delay $\delta T$ (s)', labelpad=0.6)
ax.set_ylabel('N compressions', fontsize=6, labelpad=0.6)
ax.set_xlim(-1, 1)
# ax.set_title(r'Time delay between max force and min thickness')

# fig.supylabel('N compressions')


# Show
# plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')



# %%% E500 vs Eprecomp - Precompression

# Save
SAVE = True
figSubDir = 'S1'
name = 'E500_vs_Eprecomp' # 

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
fig, ax = plt.subplots(1, 1, figsize=(6/cm_in, 6/cm_in))
# sns.swarmplot(ax=ax, data = df_f, x='cell type', y='Dh_BeforeAfter', size=1)
ax.grid()
ax.plot(df_f['E_f_<_500'].values/1e3, df_f['E_Precomp'].values/1e3, ls='', 
        marker='o', ms=4, color='dimgray', mec='none', alpha = 0.3)
parms, res = ufun.fitLineHuber(df_f['E_f_<_500'].values/1e3, df_f['E_Precomp'].values/1e3, 
                               with_intercept=False)
ax.axline((0, 0), slope=1., color=apm.cL_Set2[0], ls='-', label='y = x')
ax.axline((0, 0), slope=parms[0], color=apm.cL_Set2[1], ls='-', 
          label=f'Fit y = k.x\nk = {parms[0]:.2f}')
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_xlabel('$E_{500}$ (kPa)')
ax.set_ylabel('$E_{init\_relax}$ (kPa)')
ax.legend(handlelength=1.25)


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%% Peak Delay


# Save
SAVE = True
figSubDir = 'S1'
name = 'Peak_delay' # 

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
fig, ax = plt.subplots(1, 1, figsize=(8/cm_in, 6/cm_in))
# sns.swarmplot(ax=ax, data = df_f, x='cell type', y='Dh_BeforeAfter', size=1)
ax.hist(df_f['peakDelay'].values, bins=40, color='dimgray', zorder=3)

median = np.median(df_f['peakDelay'].values)
ax.axvline(median, color='darkorange', ls='--', lw=1,
           label=f'Median = {median:.2f} s', zorder=3)
ax.legend(handlelength = 1.25, loc='upper left')
ax.grid(zorder=1)
ax.set_xlabel(r'$\delta T$ (s)')
ax.set_ylabel('# compressions')
ax.set_title(r'Time delay between max force and min thickness')

# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = apm.makeCountDf(df_f, condCol)
# Save
if SAVE:
    ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    ufun.archiveFig(fig, name = name, ext = '.png', dpi = 500,
                    figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
    CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%%% Testing dimitriadis fit for S1 or S2

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
                'doDimitriadisFit' : True,
                'DimitriadisFitMethods' : ['Valid'],
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
                        'Plots_Papier':False,
                        'FH(t)':False,
                        'F(H)':False,
                        'F(H)_Dimitriadis':True,
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

# task = '23-12-03_M1' # -> for Pplot_Timeseries_V3()
task = '24-07-04_M6'


res = takaP.computeGlobalTable_meca(mode = 'fromScratch', task = task, fileName = 'testDimi', 
                                    save = True, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting'



# %% Supp Figure 2 & 2bis

# %%% Fig S2A - 4 x 4 with many metrics

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
        pval = results.pval
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
        # pval = results.pval
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
    
    


# %%% Fig S2A - 1 x 4 with Dimitriadis

# Save
SAVE = True
figSubDir = 'S2'
name = 'S2A_E_vs_h_Dimi'

#### Dataset

df = MecaData_Phy5
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix

XCols = ['ctFieldThickness', 'surroundingThickness', 'bestH0', 'H0_f_<_500', 'H0_Dimi_Valid']
YCols = ['E_Dimi_Valid']

dict_Xlabels = {'ctFieldThickness' : r'$H_{5mT}$', 
                'surroundingThickness' : r'$H_{surrounding}$', 
                'bestH0' : r'$H_{15\%}$', 
                'H0_f_<_500' : r'$H_{500}$',
                'H0_Dimi_Valid' : r'$H_{Dimi}$',
                }

dict_Ylabels = {'E_Dimi_Valid' : r'$E_{Dimi}$', 
                }

nX = len(XCols)
nY = len(YCols)

fig, axes = plt.subplots(nY, nX, figsize = (17/cm_in, 4/cm_in), sharey='row', sharex='col')

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
                   (df['error' + YCol[1:]] == False), 
                   (df['R2' + YCol[1:]] > 0.4), 
                   (df['Chi2' + YCol[1:]] < 50), 
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
        ax = axes[j]
        
        # win, hin = 0.35, 0.35*(11/12)
        # xin, yin = 0.95-win, 0.93-hin 
        # ax_in = ax.inset_axes([xin, yin, win, hin])
        
        ax = ax
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        sns.scatterplot(ax = ax, x=df_plot[XCol].values, y=df_plot[YCol+'_wAvg'].values/1000, 
                        marker = 'o', s = 17, color = apm.cL_Set2[0], alpha = 0.5)
        Xfit, Yfit = np.log(df_plot[XCol].values), np.log(df_plot[YCol+'_wAvg'].values/1000)
        
        wd=1/(np.std(Xfit)) # **2
        we=1/(np.std(Yfit)) # **2

        [a, b], results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
        
        A, k = np.exp(b), a
        pval = results.pval
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
        ax.legend(handles=[LegendMark], handlelength = 0.8).set_visible(False)
        
        # ax.legend()#.set_visible(False)
        # ax.legend(fontsize = 9, loc = 'lower left')
        # ax.set_title('Average per cell')
        if j==0:
            ax.set_ylabel(dict_Ylabels[YCol], fontsize=matplotlib.rcParams['axes.titlesize']+2)
            ax.tick_params(axis='y', labelsize=matplotlib.rcParams['ytick.labelsize']+2)
        else:
            ax.set_ylabel('')
        if i==(nY-1):
            ax.set_xlabel(dict_Xlabels[XCol], fontsize=matplotlib.rcParams['axes.titlesize']+2)
            ax.tick_params(axis='x', labelsize=matplotlib.rcParams['xtick.labelsize']+2)
        else:
            ax.set_xlabel('')
        ax.grid(visible=True, which='major', axis='both')
        ax.set_xlim([50, 1100])
        ax.set_ylim([0.05, 500])
        # ax.tick_params(axis='both', direction='in', which='both')
        
        
        hM, hL, hH = ufun.getLogNDistributionDescriptors(df_f[XCol].values)
        EM, EL, EH = ufun.getLogNDistributionDescriptors(df_f[YCol].values/1000)
        print(f'For {XCol} vs {YCol}')
        print(f'By compression, N = {len(df_f):.0f}')
        print(f'Typical values for H : {hM:.0f} [{hL:.0f}-{hH:.0f}]')
        print(f'Typical values for E : {EM:.2f} [{EL:.2f}-{EH:.2f}]')
        print(f'Power-law exponent & Ci : {k:.2f} +- {(k_ciw/2):.2f}')
        print(f'Actual p-value : {pval:.2e} | ' + text_pval + '\n')
        
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
    
    
# %%% Fig S2A - Investigate Dimitriadis depth range

# Save
SAVE = False
figSubDir = 'S2'
name = ''

#### Dataset

df = MecaData_Phy5
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
df, condCol = apm.makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix

XCols = ['ctFieldThickness', 'surroundingThickness', 'bestH0', 'H0_f_<_500', 'H0_Dimi_Valid']
YCols = ['E_Dimi_Valid']

dict_Xlabels = {'ctFieldThickness' : r'$H_{5mT}$', 
                'surroundingThickness' : r'$H_{surrounding}$', 
                'bestH0' : r'$H_{15\%}$', 
                'H0_f_<_500' : r'$H_{500}$',
                'H0_Dimi_Valid' : r'$H_{Dimi}$',
                }

dict_Ylabels = {'E_Dimi_Valid' : r'$E_{Dimi}$', 
                }


#### TBD !!!!

# %% Supp Figure 4


# %%% Functions to analyze and plot

dict_code = {'bestH0': 'H0',
             'surroundingThickness': 'H5mT',
             'H0_f_<_400': 'H400',
             'H0_f_<_500': 'H500',
             'H0_f_<_600': 'H600',
             'E_f_<_400': 'E400',
             'E_f_<_500': 'E500',
             'E_f_<_600': 'E600',
             'E_eff': 'Eeff',
             }

dict_axisLabels = {'bestH0': '$H_{0}$ (nm)',
                    'surroundingThickness': '$H_{5mT}$ (nm)',
                    'H0_f_<_400': '$H_{400}$ (nm)',
                    'H0_f_<_500': '$H_{0}$ (nm)',
                    'H0_f_<_600': '$H_{600}$ (nm)',
                    'E_f_<_400': '$E_{400}$ (kPa)',
                    'E_f_<_500': '$E$ (kPa)',
                    'E_f_<_600': '$E_{600}$ (kPa)',
                    'E_eff': '$E_{eff}$ (kPa)',
                    }

def compute_Eh_Exponent(df, XCol = 'H0_f_<_500', YCol = 'E_f_<_500',
                        crit_NcompsMin = 10,
                        crit_pvalFit = 0.1,
                        crit_thickCV = 0.5,
                        activeCrits = ['NcompsMin', 'pvalFit', 'thickCV'],
                        modeFit = 'OLS'):

    df, condCol = apm.makeCompositeCol(df, cols=['date'])
    CountByCond, CountByCell = apm.makeCountDf(df, condCol)
    df_f = df
    # df_f = apm.computeNLMetrics_V2(df_f, th_NLI = np.log10(2), ref_strain = 0.2)
    
    CID_longSeries = CountByCell[CountByCell['compCount'] >= 5].reset_index()['cellID'].values
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
               'A'+codeXY:[], 'alpha'+codeXY:[], 'alpha_ciw'+codeXY:[], 
               'pval'+codeXY:[], 'R2'+codeXY:[], 'thickCV'+codeXY:[], 
               codeX+'_logmean':[], codeY+'_logmean':[], #'NLR_mean':[],
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
            [b, a], results = ufun.fitLine(Xfit, Yfit)
            A, alpha = np.exp(b), a
            perc, dof, = 0.975, len(Yfit)-2
            q = st.t.ppf(perc, dof)
            params_sd = [results.cov_HC3[k, k]**0.5 for k in range(len(results.params))]
            params_ciw = [q * sd for sd in params_sd]
            alpha_ciw = results.HC3_se[1] * q
            R2 = results.rsquared
            pval = results.pvalues[1]
            
            # print(cid)
            # print(alpha, A)
            # print([a, b])
            # print(params_sd[::-1])
            # print(params_ciw[::-1])
            # print('---')
        
        elif modeFit == 'ODR':
            wd=1/(np.std(Xfit)) # **2
            we=1/(np.std(Yfit)) # **2
            
            params, results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
            a, b = params
            alpha, A = a, np.exp(b)
            alpha_ciw, _ = results.params_ciw
            pval = results.pval
            R2 = results.R2
            
            # print(cid)
            # print(alpha, A)
            # print(params)
            # print(results.params_sd)
            # print(results.params_ciw)
            # print('---')
        
        H_logmean = np.mean(Xfit)
        E_logmean = np.mean(Yfit)
        # NLR_mean  = np.mean(df_cell['NLI_mod'])
        thickCV = np.std(Xfit)/H_logmean
        # print(cid, f'{thickCV:.3f}', f'{pval:.3f}', f'{R2:.3f}')
        
        dictFit['cellID'].append(cid)
        dictFit['A'+codeXY].append(A)
        dictFit['alpha'+codeXY].append(alpha)
        dictFit['alpha_ciw'+codeXY].append(alpha_ciw)
        dictFit['thickCV'+codeXY].append(thickCV)
        dictFit['pval'+codeXY].append(pval)
        dictFit['R2'+codeXY].append(R2)
        dictFit[codeX + '_logmean'].append(H_logmean)
        dictFit[codeY + '_logmean'].append(E_logmean)
        # dictFit['NLR_mean'].append(NLR_mean)
        dictFit['valid_NcompsMin'+codeXY].append(Ncomps >= crit_NcompsMin)
        dictFit['valid_pvalFit'+codeXY].append(pval <= crit_pvalFit)
        dictFit['valid_thickCV'+codeXY].append(thickCV >= crit_thickCV)
        check_all_crit = np.all([dictFit['valid_'+s+codeXY][-1] for s in activeCrits])
        dictFit['valid_global'+codeXY].append(check_all_crit)
        
    res_df = pd.DataFrame(dictFit)
    return(res_df, df_plot)

        


def compute_Eeq(df, df_expo, h_ref = 300, inferred_exponent = 'median',
                XCol = 'H0_f_<_500', YCol = 'E_f_<_500',
                PLOT = False):
    
    codeX, codeY = dict_code[XCol], dict_code[YCol]
    codeXY = '_' + codeX + '_' + codeY
    
    df_expo['valid_global'+codeXY] = df_expo['valid_global'+codeXY] & (df_expo['alpha'+codeXY] < -1)
    df_expo_valid = df_expo[df_expo['valid_global'+codeXY] == True]
    valid_cells = df_expo_valid['cellID'].values
    Ncells = len(valid_cells)
    
    df, condCol = apm.makeCompositeCol(df, cols=['date'])
    CountByCond, CountByCell = apm.makeCountDf(df, condCol)
    df_f = df[df['cellID'].apply(lambda x : x in valid_cells)]
    
    Xfit, Yfit = np.log(df_f[XCol].values), np.log(df_f[YCol].values/1000)
    wd=1/(np.std(Xfit)) # **2
    we=1/(np.std(Yfit)) # **2
    params, results = ufun.fitLineTLS(Xfit, Yfit, wd=wd, we=we)
    a, b = params
    global_expo = a
    
    list_expos = df_expo_valid['alpha'+codeXY].values
    mean_expo = np.mean(list_expos)
    median_expo = np.median(list_expos)
    
    expos = {'mean' : mean_expo,
             'median' : median_expo,
             'global' : global_expo,
             }
    
    print(expos)
    
    # Group By
    # df_fg = apm.dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
    # df_fg = df_fg[[XCol]]
    # df_fgw2 = apm.dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
    #                                       valCol = YCol, weightCol = 'ciw'+YCol, weight_method = 'ciw^2')
    # df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

    # dictFit = {'cellID':[], 
    #            'A'+codeXY:[], 'alpha'+codeXY:[], 'alpha_ciw'+codeXY:[], 
    #            'pval'+codeXY:[], 'R2'+codeXY:[], 'thickCV'+codeXY:[], 
    #            codeX+'_logmean':[], codeY+'_logmean':[], #'NLR_mean':[],
    #            'crit_NcompsMin':[crit_NcompsMin]*Ncells, 'valid_NcompsMin'+codeXY:[], 
    #            'crit_pvalFit':[crit_pvalFit]*Ncells, 'valid_pvalFit'+codeXY:[], 
    #            'crit_thickCV':[crit_thickCV]*Ncells, 'valid_thickCV'+codeXY:[],
    #            'activeCrits':[global_crit]*Ncells, 'valid_global'+codeXY:[],}
    A_list = []
    Eeq_list = []
    inferred_expo = expos[inferred_exponent]   
    
    for i in range(Ncells):
        cid = valid_cells[i]
        df_cell = df_f[df_f['cellID'] == cid]
        # Ncomps = len(df_cell)
        
        X = df_cell[XCol].values
        Y = df_cell[YCol].values/1000
        Xfit = np.log(np.copy(X))
        Yfit = np.log(np.copy(Y)) / inferred_expo
        
        wd=1/(np.std(Xfit)) # **2
        we=1/(np.std(Yfit)) # **2
        
        params, results = ufun.fitConstantTLS(Xfit, Yfit, wd=wd, we=we)
        [B] = params
        A = np.exp(B * inferred_expo)        
        E_eq = A * (h_ref**inferred_expo)
        
        A_list.append(A)
        Eeq_list.append(E_eq)
    
    
    if PLOT:
        nCols = 6
        nRows = Ncells//nCols + 1
        fig, axes = plt.subplots(nRows, nCols, figsize=(2.5*nCols/cm_in, 3*nRows/cm_in),
                                 sharex=True, sharey=True, layout='compressed')
        
        for i in range(Ncells):
            iR = i//nCols
            iC = i%nCols
            
            cid = valid_cells[i]
            df_cell = df_f[df_f['cellID'] == cid]
            # Ncomps = len(df_cell)
            
            X = df_cell[XCol].values
            Y = df_cell[YCol].values/1000
            
            A = A_list[i]
            E_eq = Eeq_list[i]
            
            ax = axes[iR, iC]
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.plot(X, Y, ls='', 
                    marker='o', ms=4, mec='w', 
                    mew=0.1, alpha=0.5, zorder=6)
            Xplot = np.array([40,2100])
            Yplot = A * (Xplot**inferred_expo)
            ax.plot(Xplot, Yplot, ls='-')
            
            ax.axvline(h_ref, ls='-', lw=1.5, color='dimgray')
            ax.plot([h_ref], [E_eq], 'kx', 
                    label='$E_{eq}$=' + f'{E_eq:.2f}kPa')
            
            ax.legend(loc='upper right', fontsize=6, handlelength = 1)
            ax.set_xlim([50, 2000])
            ax.set_ylim([0.5, 100])
            ax.grid()
            
            
            
        fig2, ax2 = plt.subplots(1, 1, figsize=(5/cm_in, 5/cm_in),
                                 layout='compressed')
        ax=ax2
        log_Eeq     = np.log(Eeq_list)
        logmean_Eeq = np.mean(log_Eeq)
        logstd_Eeq  = np.std(log_Eeq)
        print(logmean_Eeq, np.exp(logmean_Eeq), )
        print(logstd_Eeq, np.exp(logstd_Eeq), np.exp(logstd_Eeq)**0.5)
        data = pd.DataFrame({'Eeq':Eeq_list,})
        sns.swarmplot(ax=ax, data=data, y='Eeq')
        f = 0.75
        ax.errorbar([0], np.exp(logmean_Eeq), 
                    ls='', marker='_', markerfacecolor='k',
                    mec = 'k', mew = 1.5*f, ms = 12*f,
                    xerr=None, 
                    yerr=[[np.exp(logmean_Eeq - logstd_Eeq)], [np.exp(logmean_Eeq + logstd_Eeq)]],
                    ecolor = 'k', elinewidth=1.5*f, capsize=3*f, zorder=10)
        
        ax.set_yscale('log')
        ax.set_ylabel('$E_{eq}$ (kPa)')
        # ax.grid()
        
        plt.show()
        
        
    return(A_list, Eeq_list)



# %%% E_equivalent


#### 1. Settings

apm.setGraphicOptions(mode = 'print', 
                      palette = 'Set2', 
                      colorList = apm.cL_Set21)

# Save
SAVE = False
figDir = 'C:/Users/josep/Desktop/Seafile/PapierDensité/FiguresSupp'
figSubDir = 'S4'
name = 'S4_A_1-0'

df = MecaData_Phy3
dates = ['23-02-16', '23-03-16', '23-04-26', '24-12-11']
# dates = ['24-12-11']
# dates = ['23-02-16', '23-04-26', '24-12-11']
XCol = 'H0_f_<_500'
YCol = 'E_f_<_500'
suffix = '_f_<_500'

# crit_NcompsMin = 7
# crit_pvalFit = 0.4
# crit_thickCV = 0.025
crit_NcompsMin = 7
crit_pvalFit = 0.4
crit_thickCV = 0.0275
activeCrits = ['NcompsMin', 'pvalFit', 'thickCV']
dstDir = ''
figNameRoot = ''
modeFit = 'ODR'

ColoredCells = ['24-12-11_M1_P1_C12', 
                '24-12-11_M1_P1_C17', 
                '24-12-11_M1_P1_C18', 
                '24-12-11_M1_P1_C3', 
                '24-12-11_M2_P1_C10-1', 
                '24-12-11_M2_P1_C6']

#### 2. Filter

cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
df, condCol = apm.makeCompositeCol(df, cols=['cell type'])

Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df[XCol] < 1100),
           (df['normal field'] == 5),
           (df[YCol] <= 2e4),
           (df['valid' + suffix] == True), 
           ]

df_f = apm.filterDf(df, Filters)

codeX, codeY = dict_code[XCol], dict_code[YCol]
codeXY = '_' + codeX + '_' + codeY

df_res, df_plot = compute_Eh_Exponent(df_f, XCol = XCol, YCol = YCol,
                                        crit_NcompsMin = crit_NcompsMin,
                                        crit_pvalFit = crit_pvalFit,
                                        crit_thickCV = crit_thickCV,
                                        activeCrits = activeCrits,
                                        modeFit = modeFit)

df_res['date'] = df_res['cellID'].apply(lambda x : x.split('_')[0])
df_res.sort_values(by='cellID', ascending=True, inplace=True)


A_list, Eeq_list = compute_Eeq(df_f, df_res, h_ref = 300, inferred_exponent = 'median',
                               XCol = 'H0_f_<_500', YCol = 'E_f_<_500',
                               PLOT = True)

