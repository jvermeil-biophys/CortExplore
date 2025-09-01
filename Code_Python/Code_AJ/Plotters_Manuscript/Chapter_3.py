# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:22:32 2024

@author: anumi
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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings('ignore', message='Warning: converting a masked element to nan')

import os
import sys
import time
import random
import warnings
import itertools
import matplotlib
import distinctipy

from copy import copy
from cycler import cycler
from datetime import date
import matplotlib.lines as lines
import PlottingFunctions_AJ as pf
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec
from scipy.stats import mannwhitneyu, wilcoxon
# from statannotations.Annotator import Annotator

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)

import GraphicStyles as gs
import UtilityFunctions as ufun
import TrackAnalyser_V3 as taka


#### Potentially useful lines of code
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# cp.DirDataFigToday

#### Pandas


pd.set_option('display.max_columns', None)
# pd.reset_option('display.max_columns')
pd.set_option('display.max_rows', None)
pd.reset_option('display.max_rows')


####  Matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})

#### Graphic options
gs.set_default_options_jv()

#### Bokeh
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Range1d
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10
from bokeh.layouts import gridplot
output_notebook()

#### Markers
my_default_marker_list = ['o', 's', 'D', '>', '^', 'P', 'X', '<', 'v', 'p']
markerList10 = ['o', 's', 'D', '>', '^', 'P', 'X', '<', 'v', 'p']

todayFigDir = cp.DirDataFigToday
experimentalDataDir = cp.DirRepoExp

plotLabels = 25
plotTicks = 18 
plotTitle = 25
plotLegend = 25
fontColour = '#ffffff'

plt.rcParams.update({
    'font.family': 'Arial',   # Choose your font
})

SCALE_px_cm = 2.60



# %%%% Default settings for mechanics analysis

#### HOW TO USE    
# Example : I don't want to do a whole curve Chawick Fit anymore, but I want a Dimitriadis one.
# I can create a dict fitSettings = {'doChadwickFit' : False, 'doDimitriadisFit' : True}
# And pass it as an argument in computeGlobalTable_meca

#### DETAILS
# See ufun.updateDefaultSettingsDict(settingsDict, defaultSettingsDict)
# And the 'Settings' flag in analyseTimeSeries_meca() 

#### AS REFERENCE ONLY, here is a copy of the default settings.

#### 1. For Fits
DEFAULT_stressCenters = [ii for ii in range(100, 1550, 50)]
DEFAULT_stressHalfWidths = [50, 75, 100]

DEFAULT_strainCenters = [ii/10000 for ii in range(125, 3750, 125)]
DEFAULT_strainHalfWidths = [0.0125, 0.025, 0.05]

DEFAULT_fitSettings = {# H0
                       'methods_H0':['Chadwick', 'Dimitriadis'],
                       'zones_H0':['%f_10', '%f_20'],
                       'method_bestH0':'Chadwick',
                       'zone_bestH0':'%f_10',
                       # Global fits
                       'doVWCFit' : True,
                       'VWCFitMethods' : ['Full'],
                       'doDimitriadisFit' : False,
                       'DimitriadisFitMethods' : ['Full'],
                       'doChadwickFit' : True,
                       'ChadwickFitMethods' : ['Full', 'f_<_400', 'f_in_400_800'],
                       'doDimitriadisFit' : False,
                       'DimitriadisFitMethods' : ['Full'],
                       # Local fits
                       'doStressRegionFits' : False,
                       'doStressGaussianFits' : False,
                       'centers_StressFits' : DEFAULT_stressCenters,
                       'halfWidths_StressFits' : DEFAULT_stressHalfWidths,
                       'doNPointsFits' : False,
                       'nbPtsFit' : 13,
                       'overlapFit' : 3,
                       # NEW - Numi
                       'doLogFits' : False,
                       'nbPtsFitLog' : 10,
                       'overlapFitLog' : 5,
                       # NEW - Jojo
                       'doStrainGaussianFits' : False,
                       'centers_StrainFits' : DEFAULT_strainCenters,
                       'halfWidths_StrainFits' : DEFAULT_strainHalfWidths,
                       # TEST - Jojo
                       'do3partsFits' : False,
                       }

#### 2. For Validation

DEFAULT_crit_nbPts = 8 # sup or equal to
DEFAULT_crit_R2 = 0.6 # sup or equal to
DEFAULT_crit_Chi2 = 1 # inf or equal to
DEFAULT_str_crit = 'nbPts>{:.0f} - R2>{:.2f} - Chi2<{:.1f}'.format(DEFAULT_crit_nbPts, 
                                                                   DEFAULT_crit_R2, 
                                                                   DEFAULT_crit_Chi2)

DEFAULT_fitValidationSettings = {'crit_nbPts': DEFAULT_crit_nbPts, 
                                 'crit_R2': DEFAULT_crit_R2, 
                                 'crit_Chi2': DEFAULT_crit_Chi2,
                                 'str': DEFAULT_str_crit}

#### 3. For Plots
DEFAULT_plot_stressCenters = [ii for ii in range(100, 1550, 50)]
DEFAULT_plot_stressHalfWidth = 75

DEFAULT_plot_strainCenters = [ii/10000 for ii in range(125, 3750, 125)]
DEFAULT_plot_strainHalfWidth = 0.0125

DEFAULT_plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'F(H)_VWC':True,
                        'S(e)_stressRegion':True,
                        'K(S)_stressRegion':True,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'S(e)_nPoints':True,
                        'K(S)_nPoints':True,
                        'S(e)_Log':True, # NEW - Numi
                        'K(S)_Log':True, # NEW - Numi
                        'S(e)_strainGaussian':True, # NEW - Jojo
                        'K(S)_strainGaussian':True, # NEW - Jojo
                        'Plot_Ratio':True, # NEW
                        'F(h)_log-log':True,
                        'F(h)_Chadwick_bonded':True,
                        # Fits plotting parameters
                        # Stress
                        'plotStressCenters':DEFAULT_plot_stressCenters,
                        'plotStressHW':DEFAULT_plot_stressHalfWidth,
                        # Strain
                        'plotStrainCenters':DEFAULT_plot_strainCenters,
                        'plotStrainHW':DEFAULT_plot_strainHalfWidth,
                        # Points
                        'plotPoints':str(DEFAULT_fitSettings['nbPtsFit']) \
                                     + '_' + str(DEFAULT_fitSettings['overlapFit']),
                        'plotLog':str(DEFAULT_fitSettings['nbPtsFitLog']) \
                                     + '_' + str(DEFAULT_fitSettings['overlapFitLog']),
                        }
    

#%% Mechanics


plot_stressCenters = [ii for ii in range(100, 4000, 50)]
stressHalfWidths = [50, 75, 100]

fitSettings = {# H0
                'methods_H0':['Chadwick', 'VWC', 'Dimitriadis'],
                'zones_H0':['%f_10', '%f_15', '%f_100'],
                'method_bestH0':'VWC', 
                'zone_bestH0':'%f_100',
                'doVWCFit' : True,
                'VWCFitMethods' : ['Full'],
                'doChadwickFit' : True,
                'ChadwickFitMethods' : ['Full', 'f_<_400'],
                'doChadwick-BondedFit' : True,
                'Chadwick-BondedFitMethods' : ['Full'],
                'doChadwick-log' : True,
                'Chadwick-logFitMethods' : ['Full'],
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
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

plot_stressCenters = [ii for ii in range(100, 4000, 100)]
plot_stressHalfWidth = 75

plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'F(H)_VWC':True,
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
                        'Plot_Ratio':False,
                        'F(h)_log-log':True,
                        'F(h)_Chadwick_bonded':True
                        }


Task = '24-12-14 & 24-12-20 & 25-01-14 & 25-01-23 & 25-01-21 & 25-02-28 & 25-03-12'
# Task = '25-01-21_M2_P2_C3'

fitsSubDir = 'Chadwick_bonded_Indentation_Tests'
# Task = '25-03-12'

# fitsSubDir = 'VWC-Chadwick_Chapter-3_25-03-12'

GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', 
                            fileName = fitsSubDir, save = True, PLOT = True, source = 'Python',
                            fitSettings = fitSettings, plotSettings = plotSettings,
                            fitsSubDir = fitsSubDir) # task = 'updateExisting'


#%%  Calling Data - Global Activation with I-OptoRhoA
"""
Task = '24-12-14 & 24-12-20 & 25-01-14 & 25-01-21 & 25-01-23 & 25-02-28 & 25-03-12'
        
"""

filename = 'VWC-Chadwick_Chapter-3_25-04-09'

GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/NextCloud/Anumita - Manuscript/Figures/Chapter_3/GlobalActivation_IOptoRhoa/'
# dirToSave = 'D:/Anumita/MagneticPincherData/Figures/FiguresForManuscript/Chapter_3'
dataActPath = 'D:/Anumita/MagneticPincherData/Raw/ActivationData/25-01-20_ActivationData.csv'
dataBlebPath = os.path.join(cp.DirDataAnalysis, '25-05-25_BlebbingCells_IOptoRhoA.csv')



data = pf.createDataTable(GlobalTable, dataActPath=dataActPath, dataBlebPath = dataBlebPath,
                          fitsSubDir = filename)

plt.style.use('seaborn-v0_8')

#%%%% Filters

styleActivation =  {'no light':{'color': '#808080','marker':'o', 'label': 'No Light'}, 
                   'global':{'color': '#0000ff','marker':'o', 'label': 'Global\nActivation'},
                    }

styleDf = pd.DataFrame(styleActivation)
styleDf = styleDf.transpose()

labels = list(styleDf['label'].values)
activation = list(styleActivation.keys())

celltypes = ['optoRhoA-NS']
magField = [5.0]
drugs = ['doxy'] #, 'doxy_act']
activationfreq = [0] #, 3]
firstAct = [-1] #, 1]
dates = ['24-12-14',  '25-01-14', '25-01-21', '25-01-23',  '25-02-28' ,'25-03-12']

# Here I am not considering the experiment where I activated the same cells 
# at different regions, i.e. 24-12-20

# chosenPairs = ['24-12-14_P2_C6',
#                '24-12-14_P1_C5',
#                '24-12-14_P2_C8',
#                '24-12-14_P2_C2',
#                '24-12-14_P2_C1',
#                '24-12-14_P2_C4',
#                '24-12-14_P1_C1',
#                '24-12-14_P1_C6',
#                '24-12-14_P1_C8', 
#                '24-12-14_P1_C2', 
#                '24-12-21_P1_C8',
#                '25-01-23_P1_C3',
#                '25-01-21_P1_C7',
#                '25-01-21_P2_C2',
#                '25-01-23_P1_C2']

blebCells = ['24-12-14_M2_P2_C9',
            '24-12-14_M2_P1_C4',
            '24-12-14_M2_P2_C7',
            '24-12-14_M2_P2_C5',
            '24-12-14_M2_P1_C7',
            '25-01-21_M4_P1_C6',
            '25-01-23_M4_P1_C3']


Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            (data['error_vwc_Full'] == False),
        
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] >= 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_eff'] <= 30000),
            (data['blebStatus'] == 0),
            (data['normal field'].apply(lambda x : x in magField)),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            (data['activation type'].apply(lambda x : x in activation)),
            (data['drug'].apply(lambda x : x in drugs)),
            (data['activation frequency'].apply(lambda x : x in activationfreq)),
            (data['first activation'].apply(lambda x : x in firstAct)),
            (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)

# mask = df['blebStatus'] == 1
# df = df[~mask]

df = pf.NLIcorr(df)
    
condCol, condCat = 'activation type', activation

styleDf = pd.DataFrame(styleActivation)
styleDf = styleDf.transpose()
labels = list(styleDf['label'].values)

palette_cond = pf.getSnsPalette(condCat, styleActivation)

pairs = [condCat]

plotChars = {'color' : '#000000', 'fontsize' : 13}
plotTicks = {'color' : '#000000', 'fontsize' : 13}

avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[(avgDf[('compNum', 'count')] > 2)]

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


#%%% E-400 vs E-eff (log-log)


plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(figsize=(12/SCALE_px_cm, 12/SCALE_px_cm))


plottingParams = {'data':df,
                  'x':'E_f_<_400_log',
                  'y':'E_eff_log',
                  'hue':'NLI_Plot',
                  'legend':False
                    }

ax = sns.scatterplot(**plottingParams)

params, results = ufun.fitLineOLS(df['E_f_<_400_log'].values, 
                                    df['E_eff_log'].values)

b, a = params[0], params[1]

x = (np.linspace(df['E_f_<_400_log'].min(), df['E_f_<_400_log'].max()))
Y=a*x+b


fitParamsText = 'Fit y = ax + b\n'
fitParamsText += 'a = {:.2f}, b = {:.2f}\n'.format(a, b)
fitParamsText += 'R2 = {:.3f}\n'.format((results.rsquared))

ax.plot(x, Y, color = 'grey', label = fitParamsText)
Y2 = x
ax.plot(x, Y2, color = 'black', ls = '--')

categories = ['linear', 'intermediate', 'non-linear']
colors = ['b', 'g', 'r']

for i, j in zip(categories, colors):
    df_plot = df.loc[df['NLI_Plot'] == i]
    params, results = ufun.fitLineOLS(df_plot['E_f_<_400_log'].values, 
                                      df_plot['E_eff_log'].values)
    
    
    b, a = params[0], params[1]

    x = (np.linspace(df['E_f_<_400_log'].min(), df['E_f_<_400_log'].max()))
    Y=a*x+b
    fitParamsText = ''
    fitParamsText += i + '\n'
    fitParamsText += 'a = {:.2f}, b = {:.2f}\n'.format(a, b)
    fitParamsText += 'R2 = {:.3f}\n'.format((results.rsquared))

    ax.plot(x, Y, color = j, label = fitParamsText)

plt.ylim(2, 4.5)
plt.xlim(2, 4.5)

plt.legend()
plt.show()

#%%% E-400 vs E-eff


plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(figsize=(12/SCALE_px_cm, 12/SCALE_px_cm))


plottingParams = {'data':df,
                  'x':'E_f_<_400',
                  'y':'E_eff',
                  'hue':'NLI_Plot',
                  # 'legend':False
                    }

ax = sns.scatterplot(**plottingParams)

params, results = ufun.fitLineOLS(df['E_f_<_400'].values, 
                                    df['E_eff'].values)

b, a = params[0], params[1]

x = (np.linspace(df['E_f_<_400'].min(), df['E_f_<_400'].max()))
Y=a*x+b


fitParamsText = 'Fit y = ax + b\n'
fitParamsText += 'a = {:.2f}, b = {:.2f}\n'.format(a, b)
fitParamsText += 'R2 = {:.3f}\n'.format((results.rsquared))

ax.plot(x, Y, color = 'grey', label = fitParamsText)
Y2 = x
ax.plot(x, Y2, color = 'black', ls = '--')

categories = ['linear', 'intermediate', 'non-linear']
colors = ['b', 'g', 'r']

for i, j in zip(categories, colors):
    df_plot = df.loc[df['NLI_Plot'] == i]
    params, results = ufun.fitLineOLS(df_plot['E_f_<_400'].values, 
                                      df_plot['E_eff'].values)
    
    
    b, a = params[0], params[1]

    x = (np.linspace(df['E_f_<_400'].min(), df['E_f_<_400'].max()))
    Y=a*x+b
    fitParamsText = ''
    fitParamsText += i + '\n'
    fitParamsText += 'a = {:.2f}, b = {:.2f}\n'.format(a, b)
    fitParamsText += 'R2 = {:.3f}\n'.format((results.rsquared))

    ax.plot(x, Y, color = j, label = fitParamsText)

# plt.ylim(2, 4.5)
# plt.xlim(2, 4.5)

plt.legend()
plt.show()

#%%% NLR vs NLR corrected with CIW

plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(figsize=(12/SCALE_px_cm, 12/SCALE_px_cm))

# Ensure consistent ID
df['cellID_comp'] = df['cellID'] + '_' + df['compNum'].astype(str)

# Prepare Y and K data
df_y = pd.concat([
    df[['cellID_comp', 'Y_vwc_Full']].rename(columns={'Y_vwc_Full': 'Y'}).assign(fitType='og'),
    df[['cellID_comp', 'Y_NLImod']].rename(columns={'Y_NLImod': 'Y'}).assign(fitType='corr')
])

df_k = pd.concat([
    df[['cellID_comp', 'K_vwc_Full']].rename(columns={'K_vwc_Full': 'K'}).assign(fitType='og'),
    df[['cellID_comp', 'K_NLImod']].rename(columns={'K_NLImod': 'K'}).assign(fitType='corr')
])

# Merge Y and K on index
df_combined = df_y.merge(df_k, on=['cellID_comp', 'fitType'])

palette = distinctipy.get_colors(len(df['cellID_comp'].unique()))
# Draw line from og → corr for each cellID_comp
for cid in df_combined['cellID_comp'].unique():
    sub = df_combined[df_combined['cellID_comp'] == cid]
    if len(sub) == 2:
        ax.plot(sub['Y'], sub['K'], color='gray', linewidth=1, alpha=0.4)
        
# sns.scatterplot(data=df_combined[df_combined['fitType'] == 'og'], x='Y', y='K', style='cellID_comp',
#                 palette = palette, color = 'blue', s = 100, linewidth = 1, ax=ax, alpha = 0.4)
# sns.scatterplot(data=df_combined[df_combined['fitType'] == 'corr'], x='Y', y='K', style='cellID_comp', 
#                 palette = palette, color = 'red', s = 175, linewidth = 1, ax=ax, alpha = 0.4)

marker_style = 'o'  # or any shape: 's', '^', 'D', etc.

# First plot: fitType == 'og' (blue)
for cid in df_combined['cellID_comp'].unique():
    subset = df_combined[(df_combined['fitType'] == 'og') & (df_combined['cellID_comp'] == cid)]
    sns.scatterplot(data=subset, x='Y', y='K', marker=marker_style,
                    color='blue', s=80, linewidth=1, ax=ax, alpha=0.5, label=f'OG - {cid}')

# Second plot: fitType == 'corr' (red)
for cid in df_combined['cellID_comp'].unique():
    subset = df_combined[(df_combined['fitType'] == 'corr') & (df_combined['cellID_comp'] == cid)]
    sns.scatterplot(data=subset, x='Y', y='K', marker=marker_style,
                    color='red', s=80, linewidth=1, ax=ax, alpha=0.5, label=f'Corr - {cid}')
    
# 1:1 reference line
ax.axline((1, 1), slope=1, color='C0', linestyle='--', linewidth=1)

# Log scales
ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)

# Optionally clean up legend or adjust
ax.get_legend().remove()
# ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# # Adjust layout to make space for the legend
# fig.subplots_adjust(right=0.6)
#%%% NLI vs. Compression

plt.style.use('seaborn-v0_8')
measure = 'NLI_mod'

# toPlot_avg = avgDf[(avgDf[('compNum', 'count')] > 5)]
# dfPairs, pairedCells = pf.dfCellPairs(toPlot_avg)

cellsChosen = dfPairs['cellID'].values
toPlot = df[df['cellID'].apply(lambda x : x in cellsChosen)]
toPlot.loc[toPlot['drug'] == 'doxy', 'compNum'] -= 6


colorblind_type = "Deuteranomaly"
palette_dateCell = distinctipy.get_colors(len(toPlot['dateCell'].unique()), colorblind_type=colorblind_type)

split_point=0
toPlot['new_compNum'] = toPlot['compNum']
toPlot['new_compNum'].loc[toPlot['compNum'] < split_point] = 0
toPlot['new_compNum'].loc[toPlot['compNum'] >= split_point] = toPlot['compNum'] - split_point


compCount = toPlot.groupby('compNum').count()
toPlot = toPlot[toPlot['new_compNum'] < 7] #After the 7th compression, number of cells go from 19 (max) to 8

# Apply this to each cellID group
toPlot_NLR = toPlot.dropna(subset = [measure])
toPlot_NLR = toPlot_NLR.groupby('dateCell', group_keys=False).apply(pf.NLR_normalize_by_first_six, measure)
measure_new = measure + '_norm'

#Remove compressions where 

plottingParams1 = {'data':toPlot_NLR,
                  'x':'new_compNum', 
                  'y':measure_new, 
                  'hue':'dateCell',
                  'marker':'o',
                  'palette':palette_dateCell,
                  'alpha':0.5,
                 }


plottingParams2 = {'data':toPlot_NLR,
                  'x':'new_compNum', 
                  'y':measure_new, 
                  'marker':'o',
                  'color':'black',
                  'estimator':'mean',
                  'errorbar': 'ci',
                 }


fig, ax = plt.subplots(figsize = (15/SCALE_px_cm,10/SCALE_px_cm))
ax = sns.lineplot( **plottingParams1)
ax = sns.lineplot(**plottingParams2)

# N = 2  # You can change N to any value
# for i in range(0, 3):
#     plt.axvline(x= 3.1, color='blue', linestyle='-', 
#                 ymin=0, ymax=0.1, linewidth=2)
# plt.xlim(1, 6)

N = 2  # You can change N to any value
for i in range(0, 3):
    plt.axvline(x= 0.2 + (i*3), color='blue', linestyle='-', 
                ymin=0, ymax=0.1, linewidth=2)
# plt.xlim(-5, 9)
    
# ax.legend(loc='upper left',fontsize = 6, bbox_to_anchor=(1, 1))
plt.xticks(**plotTicks)
plt.yticks(**plotTicks)
ax.get_legend().remove()

plt.ylim(-4, 4)
plt.xlim(-0.1, 6.1)

plt.axhline(y= 0, ls = (0, (5, 10)), color = 'red', lw = 2)
plt.ylabel('')
plt.xlabel('Compression No.')
plt.title('Mean NLR per cell (A.U.)', fontweight='bold', **plotChars)
plt.tight_layout()

# plt.savefig(os.path.join(dirToSave, measure+'vComp_IOpto_NoBleb_'+str(activation)+'.pdf'), dpi = 100)
# 
plt.show()

#%%% H0 vs. Compression

measure = 'bestH0_log'
toPlot_mean = toPlot.groupby('dateCell', group_keys=False).apply(pf.NLR_normalize_by_first_six, measure)

measure_new = measure + '_norm'

plottingParams1 = {'data':toPlot_mean,
                  'x':'new_compNum', 
                  'y':measure_new, 
                  'hue':'dateCell',
                  'palette':palette_dateCell,
                  'marker':'o',
                  'alpha':0.5,
                 }

plottingParams2 = {'data':toPlot_mean,
                  'x':'new_compNum', 
                  'y':measure_new, 
                  'marker':'o',
                  'color':'black',
                  'estimator':'mean'
                 }


fig, ax = plt.subplots(figsize = (15/SCALE_px_cm,10/SCALE_px_cm))
ax = sns.lineplot( **plottingParams1)
ax = sns.lineplot(**plottingParams2)

# for i in range(0, 3):
#     plt.axvline(x= 3.1 , color='blue', linestyle='-', 
#                 ymin=0, ymax=0.1, linewidth=2)
# plt.xlim(1, 6)

N = 2  # You can change N to any value
for i in range(0, 3):
    plt.axvline(x= 0.2 + (i*3), color='blue', linestyle='-', 
                ymin=0, ymax=0.1, linewidth=2)
# plt.xlim(-5, 9)

# ax.set_yscale('log')
# y_labels = np.asarray([100 ,250, 500, 1000, 1500, 1500])
# y_ticks = np.log10(np.asarray(y_labels))
# ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)

# plt.ylim(0, 2200)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(**plotTicks)
ax.get_legend().remove()
plt.axhline(y= 0, ls = (0, (5, 10)), color = 'red', lw = 2)

y_labels = np.linspace(1, 4, 4)
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels = np.asarray(y_labels),**plotTicks)


plt.xlim(-0.1, 6.1)
plt.ylabel('')
plt.xlabel('Compression No.')
plt.title('Mean H0 per cell (nm)', fontweight='bold', **plotChars)
plt.tight_layout()

# plt.savefig(os.path.join(dirToSave, measure+'vComp_IOpto_NoBleb_'+str(activation)+'.pdf'), dpi = 100)

plt.show()

#%%% surroundinthickness vs. Compression

measure = 'surroundingDx'
toPlot_mean = toPlot.groupby('dateCell', group_keys=False).apply(pf.median_normalize_by_first_six, measure)
measure_new = measure + '_norm'

plottingParams1 = {'data':toPlot_mean,
                  'x':'new_compNum', 
                  'y':measure_new, 
                  'hue':'dateCell',
                  'palette':palette_dateCell,
                  'marker':'o',
                  'alpha':0.5,
                 }

plottingParams2 = {'data':toPlot_mean,
                  'x':'new_compNum', 
                  'y':measure_new, 
                  'marker':'o',
                  'color':'black'
                 }


fig, ax = plt.subplots(figsize = (15/SCALE_px_cm,10/SCALE_px_cm))
ax = sns.lineplot( **plottingParams1)
ax = sns.lineplot(**plottingParams2)

N = 2  # You can change N to any value
for i in range(0, 3):
    plt.axvline(x= 0.2 + (i*3), color='blue', linestyle='-', 
                ymin=0, ymax=0.1, linewidth=2)
plt.xlim(-0.1, 6.1)

ax.get_legend().remove()

# ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(**plotTicks)
plt.yticks(**plotTicks)
plt.ylim(-0,2)

plt.ylabel('')
plt.xlabel('Compression No.')
# plt.title('Mean H0 per cell (nm)', fontweight='bold', **plotChars)
plt.tight_layout()


plt.savefig(os.path.join(dirToSave, measure+'vComp_IOpto_NoBleb_'+str(activation)+'.pdf'), dpi = 200)


#%%% E_eff vs. Compression

measure = 'E_eff_log'
toPlot_mean = toPlot.groupby('dateCell', group_keys=False).apply(pf.NLR_normalize_by_first_six, measure)

measure_new = measure + '_norm'

plottingParams1 = {'data':toPlot_mean,
                  'x':'new_compNum', 
                  'y':measure_new, 
                  'hue':'dateCell',
                  'palette':palette_dateCell,
                  'marker':'o',
                  'alpha':0.5,
                 }

plottingParams2 = {'data':toPlot_mean,
                  'x':'new_compNum', 
                  'y':measure_new, 
                  'marker':'o',
                  
                  'color':'black'
                 }


fig, ax = plt.subplots(figsize = (15/SCALE_px_cm,10/SCALE_px_cm))
ax = sns.lineplot( **plottingParams1)
ax = sns.lineplot(**plottingParams2)

# N = 2  # You can change N to any value
# for i in range(0, 3):
#     plt.axvline(x= 3.1 + (i*3), color='blue', linestyle='-', 
#                 ymin=0, ymax=0.1, linewidth=2)
# plt.xlim(1, 6)

N = 2  # You can change N to any value
for i in range(0, 3):
    plt.axvline(x= 0.2 + (i*3), color='blue', linestyle='-', 
                ymin=0, ymax=0.1, linewidth=2)
# plt.xlim(-5, 9)

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.yscale('log')
plt.xticks(**plotTicks)

ax.get_legend().remove()

y_labels = np.asarray([0.1, 0.5, 1, 2, 3, 4])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels = np.asarray(y_labels),**plotTicks)

plt.xlim(-0.1, 6.1)

plt.ylabel('')
plt.xlabel('Compression No.')
plt.title('Mean Effective Elasticity per cell (kPa)', fontweight='bold', **plotChars)
plt.tight_layout()

plt.savefig(os.path.join(dirToSave, measure+'vComp_IOpto_NoBleb_'+str(activation)+'.pdf'), dpi = 100)

plt.show()

#%%% E_eff vs. Compression

measure = 'E_f_<_400_log'

plottingParams1 = {'data':toPlot,
                  'x':'compNum', 
                  'y':measure, 
                  'hue':'dateCell',
                  'palette':palette_dateCell,
                  'marker':'o',
                  'alpha':0.5,
                 }

plottingParams2 = {'data':toPlot,
                  'x':'compNum', 
                  'y':measure, 
                  'marker':'o',
                  
                  'color':'black'
                 }


fig, ax = plt.subplots(figsize = (15/SCALE_px_cm,10/SCALE_px_cm))
ax = sns.lineplot( **plottingParams1)
ax = sns.lineplot(**plottingParams2)

# N = 2  # You can change N to any value
# for i in range(0, 3):
#     plt.axvline(x= 6 + (i*3), color='blue', linestyle='-', 
#                 ymin=0, ymax=0.1, linewidth=2)

N = 2  # You can change N to any value
for i in range(0, 3):
    plt.axvline(x= 6.1 + (i*3), color='blue', linestyle='-', 
                ymin=0, ymax=0.1, linewidth=2)
plt.xlim(1, 15)

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.yscale('log')
plt.xticks(**plotTicks)
plt.xlim(1, 14)

y_labels = [100, 500, 1000, 2500, 5000, 10000, 25000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels = np.asarray(y_labels)/1000,**plotTicks)

plt.ylabel('')
plt.xlabel('Compression No.')
plt.title('Mean Effective Elasticity per cell (kPa)', fontweight='bold', **plotChars)
plt.tight_layout()

# plt.savefig(os.path.join(dirToSave, measure+'vComp_IOpto'+str(activation)+'.pdf'), dpi = 100)

plt.show()
#%%% Chadwick / Dimitriadis Model (I-OptoRhoA (before and After Activation))

fullCurve = pf.plotPopKS(df,  fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, 
                    styleDict = styleActivation, condCol = condCol, mode = 'wholeCurve', 
                    scale = 'lin', printText = False, returnData = 1, returnCount = 1)

fig1, ax1, exportDf1, countDf1 = fullCurve[0]
ax1.set_ylim(0, 15)
# plt.savefig(os.path.join(dirToSave, 'KvS_FullRange_IOptoGlobal.pdf'), dpi = 100)
plt.show()

rangeCurve = pf.plotPopKS(df,  fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, 
                    styleDict = styleActivation, condCol = condCol, mode = '150_500', 
                    scale = 'lin', printText = False, returnData = 1, returnCount = 1)

fig2, ax2, exportDf2, countDf2 = rangeCurve[0]
ax2.set_ylim(0,8)
plt.savefig(os.path.join(dirToSave, 'KvS_200_600_I-Opto'+str(activation)+'.pdf'), dpi = 100)
plt.show()

#%%% Mean bestH0 boxplots

y = 'bestH0_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':6,
                  "linewidth": 0.5,
                  'edgecolor':'#ffffff',
                  'dodge':True
                    }




fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)
fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)


plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)

plt.ylabel('')
plt.xlabel('')
plt.title('Mean Cortical Thickness (nm)', fontweight='bold', **plotChars)
plt.show()
plt.savefig((dirToSave + 'Mean-{:}_I-OptoRhoA_Global.pdf'.format(y)))


#%%% NLI - Rainplot 

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'alpha':0.7
                    }

fig, ax = plt.subplots(figsize=(18/SCALE_px_cm, 15/SCALE_px_cm))

fig, ax, medians = pf.rainplot(fig, ax, condCat, palette = palette_cond, 
                             labels = labels, pairs = None, shiftBox = 0.15, shiftSwarm = -0.07,
                             colorScheme = 'white', test = 'non-param' ,pointSize = 15,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

plt.ylim(-4,3)
plt.xticks(color = 'black', fontsize = 13)
plt.title('NLR per Compression', fontweight='bold', **plotChars)
plt.yticks(**plotTicks)
plt.ylabel('NLR', **plotChars)
plt.xlabel(' ', **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + 'NLRainplot_I-OptoRhoA_'+str(activation)+'.pdf'))
plt.show()

#%%% E-effective - Rainplot 

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff_log',
                  'order' : condCat,
                  'alpha':0.7
                    }

fig, ax = plt.subplots(figsize=(18/SCALE_px_cm, 15/SCALE_px_cm))

fig, ax, medians = pf.rainplot(fig, ax, condCat, palette = palette_cond, 
                             labels = labels, pairs = None, shiftBox = 0.15, shiftSwarm = -0.07,
                             colorScheme = 'white', test = 'non-param' ,pointSize = 15,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

# plt.ylim(-4,3)
plt.xticks(color = 'black', fontsize = 13)

plt.title('Effective Elasticity per Compression', fontweight='bold', **plotChars)

ax.set_yscale('log')
y_labels = np.asarray([100, 500, 1000, 3000, 10000, 50000])
y_ticks = np.log10((y_labels))
ax.set_yticks(y_ticks, labels = (y_labels)/1000,**plotTicks)

plt.xlabel(' ', **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + 'E_eff_Rainplot_I-OptoRhoA_'+str(activation)+'.pdf'))
plt.show()

#%%% E-effective - Rainplot 

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'bestH0_log',
                  'order' : condCat,
                  'alpha':0.7
                    }

fig, ax = plt.subplots(figsize=(18/SCALE_px_cm, 15/SCALE_px_cm))

fig, ax, medians = pf.rainplot(fig, ax, condCat, palette = palette_cond, 
                             labels = labels, pairs = None, shiftBox = 0.15, shiftSwarm = -0.07,
                             colorScheme = 'white', test = 'non-param' ,pointSize = 15,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

# plt.ylim(-4,3)
plt.xticks(color = 'black', fontsize = 13)

plt.title('Thickness per Compression', fontweight='bold', **plotChars)

ax.set_yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)

plt.xlabel(' ', **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + 'bestH0_log_Rainplot_I-OptoRhoA_'+str(activation)+'.pdf'))
plt.show()



#%%% E vs H0

toPlot = df[df['dateCell'].apply(lambda x : x in pairedCells)]

fig, ax = plt.subplots(figsize=(18/SCALE_px_cm, 10/SCALE_px_cm))
fig, ax, dataSlopes = pf.EvsH0_perCompression(fig, ax, toPlot, condCat, condCol, hueType = condCol,
                                              colorScheme = 'white', palette = palette_cond)
# fig.suptitle(str(dates), **plotChars)
plt.xticks( **plotTicks)
plt.yticks(**plotTicks)
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_EvH_GlobalActivation_Ioptorhoa.pdf').format(str(dates), str(condCat)))

fig, axes = plt.subplots(figsize = (13,9))
fig, axes, avgDf = pf.EvH0_LogCellAvg(fig, axes,  dfPairs, condCat, condCol, hueType = 'condCol',
                                      colorScheme = 'white', palette = palette_cond)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()

plt.savefig((dirToSave + '(2a)_{:}_{:}_logAvgEvH_NLI.png').format(str(dates), str(condCat)))


avgDf = pf.createAvgDf(dataSlopes, condCol, dataFluoPath = None, e_norm = True)

#%%% Plotnine paired plots
#%%%% NLI - Pairedplot
measure = 'NLI_mod'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot_woHisto(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, test= 'less',
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2.1,2))



plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per cell (A.U.)\n', fontweight='bold', **plotChars)
plt.axhline(y= 0, ls = (0, (5, 10)), color = 'red', lw = 2, alpha  =0.4)

plt.show()
plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_Paired_woBlebs.pdf'))

#%%%% NLR - Normalized Pairedplot
measure = 'NLI_mod'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                          figsize = (10/SCALE_px_cm,10/SCALE_px_cm),
                     pairs = pairs, stat = stat, palette = palette_cond, 
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2.5, 2.5))


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')

plt.xlabel('')
plt.title('Normalized NLR per Cell', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_NormPaired-woBlebs.pdf'))

#%%%% NLR - correlation
measure = 'NLI_corr'
stat = 'first'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot_woHisto(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     figsize = (10/SCALE_px_cm,10/SCALE_px_cm), test = 'greater',
                     pairs = pairs, stat = stat, palette = palette_cond, 
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')

plt.xlabel('')
plt.title('NLR Correlation', fontweight='bold', **plotChars)

plt.show()
# plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_Paired-woBlebs.pdf'))


#%%%% NLR - correlation
measure = 'NLI_corr'
stat = 'first'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     figsize = (10/SCALE_px_cm,10/SCALE_px_cm), test = 'greater',
                     pairs = pairs, stat = stat, palette = palette_cond, 
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')

plt.xlabel('')
plt.title('NLR Correlation', fontweight='bold', **plotChars)

plt.show()
# plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_normPaired-woBlebs.pdf'))



#%%%% BestH0 - Pairedplot
measure = 'bestH0_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     test = 'greater',
                     pairs = pairs, stat = stat, palette = palette_cond, logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean H0 per cell (nm)\n ', fontweight='bold', **plotChars)

plt.show()
# plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_Paired_woBlebs.pdf'))

#%%%% BestH0 - Normalized Pairedplot
measure = 'bestH0_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot, df_out = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, 
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 2))


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Normalized Thickness per Cell', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_NormPaired-woBlebs.pdf'))


#%%%% E_eff - Pairedplot
measure = 'E_eff_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, test = 'greater', logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean Effective Elasticity (kPa)\n ', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_Paired_woBlebs.pdf'))

#%%%% E_eff - Normalized Pairedplot
measure = 'E_eff_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, test = 'greater',
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0,3))


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Normalized Effective Elasticity (kPa)\n ', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_normPaired_woBlebs.pdf'))

#%%%% E_normalized
measure = 'E_norm'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot_woHisto(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, test = 'greater',
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Normalized Effective Elasticity (kPa)\n ', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_Paired.pdf'))

#%%%% E_normalized, normalized
measure = 'E_norm'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, test = 'two-sided',
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Normalized Effective Elasticity (kPa)\n ', fontweight='bold', **plotChars)


plt.show()
plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_NormPaired.pdf'))

#%%% Pointplots

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()

N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)

testH0 = 'less'
testE = 'less'
testNli = 'less'

stats = 'mean'

plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', stats),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = None, 
                                          pairs = pairs, normalize = False, marker = stats, colorScheme = 'white',
                                          test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# ax.get_legend().remove()

plt.show()
plt.savefig((dirToSave + '(9a)_{:}_{:}_{:}_NLImodPointplot.png').format(str(dates), str(condCat), stats))

#%%% Cell-dependent NLR Variability

plotter = df

plotter = plotter.groupby('cellID', \
                          group_keys=False).apply(pf.NLR_normalize_by_first_six,
                                                  'NLI_mod', clip = (0, 6),
                                                  col = 'compNum')


plotter2 = avgDf[('cellID', 'first')]
colorblind_type = "Deuteranomaly"

unique_cellIDs = plotter['cellID'].unique()
Ncells = len(unique_cellIDs)
palette = distinctipy.get_colors(Ncells, colorblind_type=colorblind_type)

fig, ax = plt.subplots(figsize = (20/SCALE_px_cm,10/SCALE_px_cm))
cellID_to_color = {cell: palette[i] for i, cell in enumerate(unique_cellIDs)}

plottingParams = {'data':plotter, 
                  'x' : ('cellID'), 
                  'y' : ('NLI_mod_norm'),
                  'hue':'cellID',
                  'palette':cellID_to_color,
                  's':4
                 }

sns.swarmplot(**plottingParams, ax = ax)
for collection in ax.collections:
    collection.set_edgecolor('k')
    collection.set_linewidth(0.5) 


avg_var = plotter['NLI_mod_std'].mean()

for _, row in plotter.iterrows():
    plt.errorbar(x=row['cellID'], y=row['NLI_mod_ref'], yerr=row['NLI_mod_std'],
                 fmt="none", color=cellID_to_color[row['cellID']],
                 capsize=5, alpha=0.4, lw = 0.3,  capthick=1)
    
    
text = 'Average STD of Cells = {:.2f}'.format(avg_var)

plt.text(0, 3, text, fontsize=12)
x_labels = ['C'+str(i+1) for i in range(Ncells)]
ax.set_xticks(ax.get_xticks(), labels =x_labels, rotation = 90, **plotTicks)
plt.ylim(-3, 3.5)
# ax.get_legend().remove()
plt.savefig((dirToSave + '{:}_NLI_StandardDev.pdf').format(condCat), dpi = 200)

#%%% Cell-dependent NLR evolution

plotter = avgDf

fig, ax = plt.subplots(figsize = (20/SCALE_px_cm,10/SCALE_px_cm))

plottingParams = {'data':plotter, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'hue':('cellID','first'),
                  'palette':cellID_to_color,
                  's':10,
                 
                 }

sns.swarmplot(**plottingParams, ax = ax)
plt.ylim(-3, 3.5)

var_pop = plotter[('NLI_mod', 'mean')].var()
text = 'STD of the Population = {:.2f}'.format(var_pop)
plt.text(0, 3, text, fontsize=12)
ax.get_legend().remove()
plt.savefig((dirToSave + '{:}_NLI_StdDevofMean.pdf').format(condCat), dpi = 200)



#%%% Bleb status

df_global = df[df['dateCell'].apply(lambda x : x in pairedCells)]
df_global = df_global[df_global['activation type'] == 'global']
df_notblebbing = df_global[~df_global['cellID'].apply(lambda x : x in blebCells)]

desired_compNum = list(range(1, 10))

padded_rows = []

for cell_id in blebCells:
    print(cell_id)
    # Get the subset of rows for this cellID
    cell_data = df_global[df_global['cellID'] == cell_id]
    
    # Create a DataFrame for the full range of compNum (1 to 9)
    temp = pd.DataFrame({'cellID': [cell_id] * len(desired_compNum), 'compNum': desired_compNum})
    
    # Merge with the original data to fill missing compNum values with NaN
    merged = pd.merge(temp, cell_data, on=['cellID', 'compNum'], how='left')
    
    # Append the result to the expanded_data DataFrame
    padded_rows.append(merged)
    
# Combine only the padded results into a new DataFrame
df_global_padded = pd.concat(padded_rows + [df_notblebbing], ignore_index=True)

df_global_padded['blebStatus'] = df_global_padded['blebStatus'].fillna(1)

grouped = df_global_padded.groupby('compNum')

agg_dict = {'blebStatus':['sum', 'mean', 'count'],
            'cellID':['first', 'count'],
            
            }
bleb_metrics = grouped.agg(agg_dict)

bleb_metrics[('blebStatus', 'percent')] = np.round((bleb_metrics['blebStatus', 'sum'].values)/len(pairedCells)*100, 0)

fig, ax = plt.subplots()

plottingParams = {'data':bleb_metrics, 
                  'x' : 'compNum', 
                  'y' : ('blebStatus', 'percent'),
                  'linewidth' : 2,
                  'markersize' : 6,
                  'markeredgecolor':'black', 
                  'color' : 'k',
                  'marker' : 'o'
                   }

plottingParams2 = {'data':bleb_metrics, 
                  'x' : 'compNum', 
                  'y' : ('cellID', 'count'),
                  'linewidth' : 2,
                  'markersize' : 6,
                  'markeredgecolor':'red', 
                  'color' : 'r',
                  'marker' : 'o'
                   }

ax = sns.lineplot(ax = ax, **plottingParams)
y_labels_1 = np.linspace(0, 50, 6)

ax.set_yticks(y_labels_1, labels = (y_labels_1),**plotTicks)

ax2 = ax.twinx()
# Example of plotting another line on the twin axis
ax2 = sns.lineplot(ax=ax2, **plottingParams2)

# plt.ylabel('(%) Percentage of Blebbing Cells', **plotChars)
# plt.xlabel('Compressions Post-Activation', **plotChars)
y_labels_2 = np.linspace(0, 25, 6)

ax.set_ylim(-1, 50)
ax2.set_ylim(-1, 25)
ax2.set_yticks(y_labels_2, labels = y_labels_2,**plotTicks)


# plt.xticks(**plotTicks)
# plt.yticks(**plotTicks)
N = 2  # You can change N to any value
for i in range(0, 3):
    plt.axvline(x= 0.3 + (i*3), color='blue', linestyle='-', 
                ymin=0, ymax=0.1, linewidth=2)
    
plt.xlim(0, 9)
plt.savefig(os.path.join(dirToSave, 'BlebbingCellsvComp_IOpto.pdf'), dpi = 200)


#%%  Calling Data - Noninduced vs. Induced
"""
Task = '24-12-14 & 24-12-20 & 25-01-14 & 25-01-21 & 25-01-23 & 25-02-28 & 25-03-12'
        
"""

filename = 'VWC-Chadwick_Chapter-3_25-04-09'


GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/NextCloud/Anumita - Manuscript/Figures/Chapter_3/'
# dirToSave = 'D:/Anumita/MagneticPincherData/Figures/FiguresForManuscript/Chapter_3'
dataActPath = 'D:/Anumita/MagneticPincherData/Raw/ActivationData/25-01-20_ActivationData.csv'

data = pf.createDataTable(GlobalTable, dataActPath=dataActPath, fitsSubDir = filename)
plt.style.use('seaborn-v0_8')

#%%%% Filters

styleDox =  {#'none':{'color': '#000000','marker':'o', 'label': 'Non-induced'},
             'doxy':{'color': '#808080','marker':'o', 'label': 'Induced'},
            }

styleDf = pd.DataFrame(styleDox)
styleDf = styleDf.transpose()

labels = list(styleDf['label'].values)
drugs = list(styleDox.keys())

celltypes = ['optoRhoA-NS']
activation = ['no light']
magField = [5.0]
activationfreq = [0]
firstAct = [-1]
dates = ['24-12-14',  '24-12-20' , '25-01-14', '25-01-21', '25-01-23',  '25-02-28' ,'25-03-12']


Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] >= 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_eff'] <= 30000),
            # (data['compNum'] < 4),
            (data['normal field'].apply(lambda x : x in magField)),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            (data['activation type'].apply(lambda x : x in activation)),
            (data['drug'].apply(lambda x : x in drugs)),
            (data['activation frequency'].apply(lambda x : x in activationfreq)),
            (data['first activation'].apply(lambda x : x in firstAct)),
            (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)

condCol, condCat = 'drug', drugs
df = pf.NLIcorr(df)

palette_cond = pf.getSnsPalette(condCat, styleDox)

pairs = [condCat]

plotChars = {'color' : '#000000', 'fontsize' : 13}
plotTicks = {'color' : '#000000', 'fontsize' : 13}

avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[(avgDf[('compNum', 'count')] > 2)]

dfPairs, pairedCells = pf.dfCellPairs(avgDf)

#%%% Mean NLR Boxplot

y = 'NLI_mod' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':6,
                  "linewidth": 0.5,
                  'edgecolor':'#ffffff',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)
fig, ax, dfmedians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
plt.ylim(-3, 3.5)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean NLR per Cell', fontweight='bold', **plotChars)
plt.show()
plt.savefig((dirToSave + 'Mean-{:}_I-OptoRhoA_NI-Ind.pdf'.format(y)))

#%%% Mean bestH0 boxplots

y = 'bestH0_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':6,
                  "linewidth": 0.5,
                  'edgecolor':'#ffffff',
                  'dodge':True
                    }


fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)
fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)

plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)

plt.ylabel('')
plt.xlabel('')
plt.title('Mean Cortical Thickness (nm)', fontweight='bold', **plotChars)
plt.show()
plt.savefig((dirToSave + 'Mean-{:}_I-OptoRhoA_NI-Ind.pdf'.format(y)))

#%%% Mean Elastic Modulus 

y = 'E_eff_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':6,
                  "linewidth": 0.5,
                  'edgecolor':'#ffffff',
                  'dodge':True
                    }


fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)
fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)

plt.yscale('log')
y_labels = np.asarray([100, 500, 1000, 3000, 10000, 50000])
y_ticks = np.log10((y_labels))
ax.set_yticks(y_ticks, labels = (y_labels)/1000,**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean Effective Elasticity (kPa)', fontweight='bold', **plotChars)
plt.show()
plt.savefig((dirToSave + 'Mean-{:}_I-OptoRhoA_NI-Ind.pdf'.format(y)))

#%%% NLI correlation

y = 'NLI_corr' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'first'),
                  'order' : condCat,
                  's':6,
                  "linewidth": 0.5,
                  'edgecolor':'#ffffff',
                  'dodge':True
                    }

fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)
fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)

# plt.yscale('log')
# y_labels = np.asarray([100 ,250, 500, 1000, 1500])
# y_ticks = np.log10(np.asarray(y_labels))
# ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)

plt.ylabel('')
plt.xlabel('')
plt.title('NLI Correlation', fontweight='bold', **plotChars)
plt.show()
plt.savefig((dirToSave + 'Mean-{:}_I-OptoRhoA_NI-Ind.pdf'.format(y)))

#%%% Cell-dependent NLR Variability

plotter = df[df['date'] == '24-12-14']

plotter = plotter.groupby('cellID', \
                          group_keys=False).apply(pf.NLR_normalize_by_first_six,
                                                  'NLI_mod', clip = (0, 6),
                                                  col = 'compNum')


plotter2 = avgDf[('cellID', 'first')]
colorblind_type = "Deuteranomaly"

unique_cellIDs = plotter['cellID'].unique()
Ncells = len(unique_cellIDs)
palette = distinctipy.get_colors(Ncells, colorblind_type=colorblind_type)

fig, ax = plt.subplots(figsize = (20/SCALE_px_cm,10/SCALE_px_cm))
cellID_to_color = {cell: palette[i] for i, cell in enumerate(unique_cellIDs)}

plottingParams = {'data':plotter, 
                  'x' : ('cellID'), 
                  'y' : ('NLI_mod_norm'),
                  'hue':'cellID',
                  'palette':cellID_to_color,
                  's':4
                 }

sns.swarmplot(**plottingParams, ax = ax)
for collection in ax.collections:
    collection.set_edgecolor('k')
    collection.set_linewidth(0.5) 


avg_var = plotter['NLI_mod_std'].mean()

for _, row in plotter.iterrows():
    plt.errorbar(x=row['cellID'], y=row['NLI_mod_ref'], yerr=row['NLI_mod_std'],
                 fmt="none", color=cellID_to_color[row['cellID']],
                 capsize=5, alpha=0.4, lw = 0.3,  capthick=1)
    
    
text = 'Average STD of Cells = {:.2f}'.format(avg_var)

plt.text(0, 3, text, fontsize=12)
x_labels = ['C'+str(i+1) for i in range(Ncells)]
ax.set_xticks(ax.get_xticks(), labels =x_labels, rotation = 90, **plotTicks)
plt.ylim(-3, 3.5)
# ax.get_legend().remove()
plt.savefig((dirToSave + '{:}-IOptoRhoA_NLI_StandardDev.pdf').format(condCat), dpi = 200)

#%%% Cell-dependent NLR evolution

plotter = avgDf[avgDf[('date', 'first')] == '24-12-14']

fig, ax = plt.subplots(figsize = (20/SCALE_px_cm,10/SCALE_px_cm))

plottingParams = {'data':plotter, 
                  'x' : ('date', 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'hue':('cellID','first'),
                  'palette':cellID_to_color,
                  's':10,
                 
                 }

sns.swarmplot(**plottingParams, ax = ax)
plt.ylim(-3, 3.5)

var_pop = plotter[('NLI_mod', 'mean')].var()
text = 'STD of the Population = {:.2f}'.format(var_pop)
plt.text(0, 3, text, fontsize=12)
ax.get_legend().remove()
plt.savefig((dirToSave + '{:}-IOptoRhoA_NLI_StdDevofMean.pdf').format(condCat), dpi = 200)

#%%% Fluctuations vs Thickness

fig, ax = plt.subplots(figsize = (10/SCALE_px_cm,10/SCALE_px_cm))

plottingParams = {'data':avgDf, 
                  'x' :('bestH0' , 'mean'),
                  'y':('ctFieldFluctuAmpli' , 'first'),
                  's':30
                  }

params, results = ufun.fitLineHuber(avgDf[('bestH0' , 'mean')], avgDf[('ctFieldFluctuAmpli' , 'first')])
pval = results.pvalues[1]
a, k = params[1], params[0]
x = np.linspace(100, 1200, 10)
y = a*x + k

eqnText = ''
eqnText += " Fit y = m * x + c\n".format(a, k)
eqnText += " y = {:.1e} * x + {:.1f}\n".format(a, k)
eqnText += " p-val = {:.3f}".format(pval)

ax = sns.scatterplot(label = eqnText, color = styleActivation[activation[0]]['color'],  **plottingParams)
plt.plot(x, y, color = styleActivation[activation[0]]['color'])

plt.legend(fontsize=12)
plt.xticks(**plotTicks)
plt.yticks(**plotTicks)
ax.set_ylim(0, 1000)
plt.show()
plt.savefig((dirToSave + 'Fluctuations_bestH0_{:}_{:}.pdf').format(celltypes, drugs), dpi = 200)

#%%% Coefficient of variation between compressions and within cells of a population

groupedDf = df.groupby('cellID')

plottingParams = {'data':bleb_metrics, 
                  'x' : 'compNum', 
                  'y' : ('blebStatus', 'percent'),
                  'linewidth' : 2,
                  'markersize' : 6,
                  'markeredgecolor':'black', 
                  'color' : 'k',
                  'marker' : 'o'
                   }


ax2 = sns.lineplot(ax=ax2, **plottingParams2)
