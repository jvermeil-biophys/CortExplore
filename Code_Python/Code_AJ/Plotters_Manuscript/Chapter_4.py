# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 09:53:05 2025

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

import TrackAnalyser_V3_Manuscript_AJ as taka_ms

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
                'method_bestH0':'Chadwick', 
                'zone_bestH0':'%f_15',
                'doVWCFit' : True,
                'VWCFitMethods' : ['Full'],
                'doChadwickFit' : True,
                'ChadwickFitMethods' : ['Full', 'f_<_400'],
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
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
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


Task = '24-12-14'
# Task = '25-01-21_M2_P2_C3'


fitsSubDir = 'IntroPlots_25-06-12'
# Task = '25-03-12'

# fitsSubDir = 'VWC-Chadwick_Chapter-3_25-03-12'

GlobalTable_meca = taka_ms.computeGlobalTable_meca(task = Task, mode = 'fromScratch', 
                            fileName = fitsSubDir, save = True, PLOT = True, source = 'Python',
                            fitSettings = fitSettings, plotSettings = plotSettings,
                            fitsSubDir = fitsSubDir) # task = 'updateExisting'


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
                        'Plot_Ratio':False
                        }


Task = '24-12-14 & 24-12-20 & 25-01-14 & 25-01-23 & 25-01-21 & 25-02-28 & 25-03-12'
# Task = '25-01-21_M2_P2_C3'


fitsSubDir = 'VWC-Chadwick_Chapter-4_25-06-12'
# Task = '25-03-12'

# fitsSubDir = 'VWC-Chadwick_Chapter-3_25-03-12'

GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', 
                            fileName = fitsSubDir, save = True, PLOT = False, source = 'Python',
                            fitSettings = fitSettings, plotSettings = plotSettings,
                            fitsSubDir = fitsSubDir) # task = 'updateExisting'


#%%  Calling Data - Polarization with I-OptoRhoA
"""
Task = '24-12-14 & 24-12-20 & 25-01-14 & 25-01-21 & 25-01-23 & 25-02-28 & 25-03-12'
       'VWC-Chadwick_Chapter-3_25-05-09' 
       'VWC_optoRhoAVB-NS_updated_25-03-24' 
       'VWC-Chadwick_Chapter-3_25-04-09' 
filename =  
"""

filename =  'VWC-Chadwick_Chapter-4_25-06-12' 


GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/NextCloud/Anumita - Manuscript/Figures/Chapter_4/'
# dirToSave = 'D:/Anumita/MagneticPincherData/Figures/FiguresForManuscript/Chapter_3'
dataActPath = 'D:/Anumita/MagneticPincherData/Raw/ActivationData/25-01-20_ActivationData.csv'
dataBlebPath = os.path.join(cp.DirDataAnalysis, '25-05-25_BlebbingCells_IOptoRhoA.csv')

data = pf.createDataTable(GlobalTable, dataActPath=dataActPath,  dataBlebPath = dataBlebPath,
                          fitsSubDir = filename)

plt.style.use('seaborn-v0_8')

#%%%% Filters

styleActivation =  {'no light':{'color': '#808080','marker':'o', 'label': 'Non-Polarized'},
                    'away from beads':{'color': '#512DA8','marker':'o', 'label': 'Polarized\nFront'},
                    # 'side':{'color': '#BA1E74','marker':'o', 'label': 'Middle'},
                    # 'at beads':{'color': '#fcb001','marker':'o', 'label': 'Polarized\nRear'},
                  #  'global':{'color': '#0000ff','marker':'o', 'label': 'Global\nActivation'},
                    }

styleDf = pd.DataFrame(styleActivation)
styleDf = styleDf.transpose()

labels = list(styleDf['label'].values)
activation = list(styleActivation.keys())

celltypes = ['optoRhoA-NS']

magField = [5.0]
drugs = ['doxy', 'doxy_act']
activationfreq = [0, 3]
firstAct = [-1, 1]
dates = ['24-12-14',  '24-12-20' , '25-01-14', '25-01-21', '25-01-23',  '25-02-28' ,'25-03-12']

# dates = ['24-12-14',  '25-01-14', '25-01-21', '25-01-23',  '25-02-28' ,'25-03-12']


Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] >= 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_eff'] <= 30000),
            (data['blebStatus'] == 0),
            (data['compNum'] < 10),
            (data['normal field'].apply(lambda x : x in magField)),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            (data['activation type'].apply(lambda x : x in activation)),
            (data['activation frequency'].apply(lambda x : x in activationfreq)),
            (data['first activation'].apply(lambda x : x in firstAct)),
            (data['drug'].apply(lambda x : x in drugs)),
            (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)


mask = (data['cellID'] == '25-01-14_M2_P2_C4')
df = df[~mask]

df = pf.NLIcorr(df)

condCol, condCat = 'activation type', activation

styleDf = pd.DataFrame(styleActivation)
styleDf = styleDf.transpose()
labels = list(styleDf['label'].values)

palette_cond = pf.getSnsPalette(condCat, styleActivation)

# pairs = [['no light', 'away from beads'], ['no light', 'side'], ['no light', 'at beads'],
#          ['no light', 'global'], ['at beads', 'away from beads']]

pairs = [condCat]

plotChars = {'color' : '#000000', 'fontsize' : 13}
plotTicks = {'color' : '#000000', 'fontsize' : 13}

avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[(avgDf[('compNum', 'count')] > 2)]

dfPairs, pairedCells = pf.dfCellPairs(avgDf)

cellsChosen = dfPairs['cellID'].values
toPlot = df[df['cellID'].apply(lambda x : x in cellsChosen)]
toPlot.loc[toPlot['drug'] == 'doxy', 'compNum'] -= 6


#%%% Histogram thickness

plottingParams = {'data':toPlot,
                  'x':'bestH0_log'
                 }

fig, ax = plt.subplots(figsize = (15/SCALE_px_cm,10/SCALE_px_cm))

median_thickness = toPlot['bestH0_log'].median()

sns.histplot(**plottingParams)
plt.axvline(x= median_thickness, color='red')
plt.xscale('log')

x_labels = np.asarray([100 ,250, 500, 1000, 1500])
x_ticks = np.log10(np.asarray(x_labels))
ax.set_xticks(x_ticks, labels = np.asarray(x_labels),**plotTicks)


#%%% Z-difference vs BestH0

df['D2'] =  (np.sqrt(df['surroundingDx']**2 + df['surroundingDy']**2)) - 4500
df['ampliDz'] = np.abs(df['surroundingDz'])

plottingParams = {'data':df,
                  'x':'D2',
                  'y':'ampliDz',
                  'hue':condCol
                 }

fig, ax = plt.subplots(figsize = (15/SCALE_px_cm,10/SCALE_px_cm))



sns.scatterplot(**plottingParams)
# plt.xscale('log')

# x_labels = np.asarray([100 ,250, 500, 1000, 1500])
# x_ticks = np.log10(np.asarray(x_labels))
# ax.set_xticks(x_ticks, labels = np.asarray(x_labels),**plotTicks)


#%%% NLI vs. Compression

measure = 'NLI_mod'

# toPlot_avg = avgDf[(avgDf[('compNum', 'count')] > 5)]
# dfPairs, pairedCells = pf.dfCellPairs(toPlot_avg)

colorblind_type = "Deuteranomaly"
palette_dateCell = distinctipy.get_colors(len(toPlot['dateCell'].unique()), colorblind_type=colorblind_type)

split_point = 0
toPlot['new_compNum'] = toPlot['compNum']
toPlot['new_compNum'].loc[toPlot['compNum'] < split_point] = 0
toPlot['new_compNum'].loc[toPlot['compNum'] >= split_point] = toPlot['compNum'] - split_point


# Apply this to each cellID group
toPlot_norm = toPlot.groupby('dateCell', group_keys=False).apply(pf.NLR_normalize_by_first_six, measure)

measure_new = measure + '_norm'

compCount = toPlot.groupby('compNum').count()
try:
    compCount_max_index = compCount.loc[(compCount['new_compNum'] < int(compCount['new_compNum'].max() / 2))].index[0]
    toPlot_norm = toPlot_norm[toPlot_norm['new_compNum'] < compCount_max_index]# except:
except:
    pass
    
plottingParams1 = {'data':toPlot_norm,
                  'x':'new_compNum', 
                  'y':measure_new, 
                  'hue':'dateCell',
                  'marker':'o',
                  'palette':palette_dateCell,
                  'alpha':0.5,
                 }

plottingParams2 = {'data':toPlot_norm,
                  'x':'new_compNum', 
                  'y':measure_new, 
                  'marker':'o',
                  'color':'black',
                  'estimator':'mean'
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
    
ax.legend(loc='upper left',bbox_to_anchor=(1, 1))
# ax.get_legend().remove()

plt.xticks(**plotTicks)
plt.yticks(**plotTicks)

plt.xlim(-0.1, 9.5)

plt.ylim(-3, 2.5)
plt.axhline(y= 0, ls = (0, (5, 10)), color = 'red', lw = 2)

plt.ylabel('')
plt.xlabel('Compression No.')
plt.title('Mean NLR per cell (A.U.)', fontweight='bold', **plotChars)
plt.tight_layout()

plt.savefig(os.path.join(dirToSave, measure+'vComp_IOpto'+str(activation)+'.pdf'), dpi = 100)

plt.show()

#%%% H0 vs. Compression

measure = 'bestH0_log'

toPlot_norm = toPlot.groupby('dateCell', group_keys=False).apply(pf.mean_normalize_by_first_six, measure)
measure_new = measure + '_norm'

compCount = toPlot.groupby('compNum').count()
try:
    compCount_max_index = compCount.loc[(compCount['new_compNum'] < int(compCount['new_compNum'].max() / 2))].index[0]
    toPlot_norm = toPlot_norm[toPlot_norm['new_compNum'] < compCount_max_index]# except:
except:
    pass

plottingParams1 = {'data':toPlot_norm,
                  'x':'new_compNum', 
                  'y':measure_new, 
                  'hue':'dateCell',
                  'palette':palette_dateCell,
                  'marker':'o',
                  'alpha':0.5,
                 }

plottingParams2 = {'data':toPlot_norm,
                  'x':'new_compNum', 
                  'y':measure_new, 
                  'marker':'o',
                  'color':'black'
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

# plt.yscale('log')
# y_labels = np.asarray([100 ,250, 500, 1000, 1500])
# y_ticks = np.log10(np.asarray(y_labels))
# ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)


ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(**plotTicks)
plt.xlim(-0.1, 9.5)
plt.ylim(0.8, 1.3)

plt.axhline(y=1, ls = (0, (5, 10)), color = 'red', lw = 2)

plt.ylabel('')
plt.xlabel('Compression No.')
plt.title('Cortical Thickness (nm)', fontweight='bold', **plotChars)
plt.tight_layout()

plt.savefig(os.path.join(dirToSave, measure+'vComp_IOpto'+str(activation)+'.pdf'), dpi = 100)

plt.show()

#%%% surroundinthickness vs. Compression

measure = 'surroundingDz'

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
plt.xlim(-5, 9)
plt.axhline(y=1, ls = (0, (5, 10)), color = 'red', lw = 2)

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(**plotTicks)
plt.yticks(**plotTicks)
plt.xlim(-0.1, 9.5)
# plt.ylim(-1200, 1200)

plt.ylabel('')
plt.xlabel('Compression No.')
# plt.title('Mean H0 per cell (nm)', fontweight='bold', **plotChars)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(dirToSave, measure+'vComp_IOpto'+str(activation)+'.pdf'), dpi = 200)

#%%% compCount vs. Compression


toPlot_count = toPlot.groupby(['compNum', condCol]).count().reset_index()

toPlot_count = toPlot_count[['compNum', 'new_compNum', condCol]]

plottingParams1 = {'data':toPlot_count,
                  'x':'compNum', 
                  'y':'new_compNum', 
                  'hue':'activation type',
                  # 'palette':palette_dateCell,
                  'marker':'o',
                  'alpha':0.5,
                 }


fig, ax = plt.subplots(figsize = (15/SCALE_px_cm,10/SCALE_px_cm))
ax = sns.lineplot( **plottingParams1)

N = 2  # You can change N to any value
for i in range(0, 3):
    plt.axvline(x= 0.2 + (i*3), color='blue', linestyle='-', 
                ymin=0, ymax=0.1, linewidth=2)
# plt.xlim(-5, 9)
plt.axhline(y= 0, ls = (0, (5, 10)), color = 'red', lw = 2)

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(**plotTicks)
plt.yticks(**plotTicks)
# plt.xlim(-0.1, 9.5)
# plt.ylim(-1200, 1200)

plt.ylabel('')
plt.xlabel('Compression No.')
# plt.title('Mean H0 per cell (nm)', fontweight='bold', **plotChars)
plt.tight_layout()

plt.savefig(os.path.join(dirToSave, measure+'vComp_IOpto'+str(activation)+'.pdf'), dpi = 200)



#%%% E_eff vs. Compression

measure = 'E_eff_log'
toPlot_mean = toPlot.groupby('dateCell', group_keys=False).apply(pf.mean_normalize_by_first_six, measure)

measure_new = measure + '_norm'
compCount = toPlot.groupby('compNum').count()
try:
    compCount_max_index = compCount.loc[(compCount['new_compNum'] < int(compCount['new_compNum'].max() / 2))].index[0]
    toPlot_norm = toPlot_norm[toPlot_norm['new_compNum'] < compCount_max_index]# except:
except:
    pass
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
plt.axhline(y= 0, ls = (0, (5, 10)), color = 'red', lw = 2)

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.yscale('log')
# plt.xticks(**plotTicks)

# y_labels = [100, 500, 1000, 2500, 5000, 10000, 25000]
# y_ticks = np.log10(np.asarray(y_labels))
# ax.set_yticks(y_ticks, labels = np.asarray(y_labels)/1000,**plotTicks)
plt.xlim(-0.1, 9.5)
plt.ylim(0.5, 1.5)

plt.ylabel('')
plt.xlabel('Compression No.')
plt.title('Mean Effective Elasticity per cell (kPa)', fontweight='bold', **plotChars)
plt.tight_layout()

plt.savefig(os.path.join(dirToSave, measure+'vComp_IOpto'+str(activation)+'.pdf'), dpi = 100)

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

#%%% NLI - Rainplot 

plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'alpha':0.7,
                  'linewidth':0,
                    }

# fig, ax = plt.subplots(figsize=(22/SCALE_px_cm, 8/SCALE_px_cm))
fig, ax = plt.subplots(figsize = (12/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)

fig, ax, medians = pf.rainplot(fig, ax, condCat, palette = palette_cond, 
                             labels = labels, pairs = None, shiftBox = 0.15, shiftSwarm = -0.07,
                             colorScheme = 'white', test = 'non-param' ,pointSize = 15,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

plt.ylim(-4,4)
plt.axhline(y= 0, color = 'red', lw = 1, alpha = 0.5)

plt.xticks(color = 'black', fontsize = 9)
# plt.title('NLR per Compression', fontweight='bold', **plotChars)
plt.yticks(**plotTicks)
plt.ylabel('NLR', **plotChars)
plt.xlabel(' ', **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + 'NLRainplot_I-OptoRhoA_'+str(activation)+'.pdf'))


#%%% E-effective - Rainplot 

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff',
                  'order' : condCat,
                  'alpha':0.7
                    }

fig, ax = plt.subplots(figsize=(18/SCALE_px_cm, 10/SCALE_px_cm))

fig, ax, medians = pf.rainplot(fig, ax, condCat, palette = palette_cond, 
                             labels = labels, pairs = None, shiftBox = 0.15, shiftSwarm = -0.07,
                             colorScheme = 'white', test = 'non-param' ,pointSize = 15,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

# plt.ylim(-4,3)
plt.xticks(color = 'black', fontsize = 13)

plt.title('Effective Elasticity per Compression', fontweight='bold', **plotChars)

# ax.set_yscale('log')
# y_labels = np.asarray([100, 500, 1000, 3000, 10000, 50000])
# y_ticks = np.log10((y_labels))
# ax.set_yticks(y_ticks, labels = (y_labels)/1000,**plotTicks)

plt.xlabel(' ', **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + 'E_eff_Rainplot_I-OptoRhoA_'+str(activation)+'.pdf'))
plt.show()

#%%% BestH0 - Rainplot 

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



#%%% NLR, Displot

y = 'NLI_mod'

df['category'] = pd.Categorical(df[condCol], categories=condCat, ordered=True)
# Sort by the categorical order
toPlot = df.sort_values('category')

plottingParams = {
    'data': toPlot,
    # 'col': f'{condCol}_first',  
    'y': y,      
    'stat':'percent',
    'bins':25,
    'hue': condCol,  
    'col': condCol,  
    'palette': palette_cond,
    'line_kws':{"linewidth": 3},
    'kde':True,
    'aspect':1,
    }

fig, ax = plt.subplots(figsize=(10/SCALE_px_cm, 10/SCALE_px_cm))

ax = sns.displot(legend = False, **plottingParams)

ax.tick_params(axis='both', labelsize=plotTicks['fontsize'])

plt.ylabel('NLR per Compression', fontweight='bold', **plotChars)
plt.axhline(y= 0, ls = (0, (5, 10)), color = 'red', lw = 2)

plt.tight_layout()
# fig.title('Mean NLR per cell', fontweight='bold', **plotChars)


plt.savefig((dirToSave + 'I-OptoRhoA_Displot_{:}.pdf').format( str(condCat)), dpi = 200)
plt.show()

#%%% Mean NLR Boxplot

plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}

y = 'NLI_mod' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4.5,
                  "linewidth": 0.3,
                  'edgecolor':'k',
                  
                  'dodge':True
                    }




fig, ax = plt.subplots(figsize = (12/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)
fig, ax, dfmedians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)



plt.xlim(-0.5, 4.5)
plt.axhline(y= 0, color = 'red', lw = 1, alpha = 0.5)

plt.ylim(-4, 4)
plt.ylabel('')
plt.xlabel('')
# plt.title('Mean NLR per Cell', fontweight='bold', **plotChars)
plt.show()
plt.savefig((dirToSave + 'Mean-{}_I-OptoRhoA_'+str(y)+'.pdf'))

#%%% Mean bestH0 boxpltos

plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}

y = 'bestH0_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.1,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (12/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)
fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)

plt.xlim(-0.5, 4.5)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)

plt.ylabel('')
plt.xlabel('')
plt.title('Mean Cortical Thickness (nm)', fontweight='bold', **plotChars)
plt.show()
# plt.savefig((dirToSave + 'Mean-{}_I-OptoRhoA_'+str(y)+'.pdf'))

#%%% Elastic Modulus 

plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}

y = 'E_eff_log' 

plottingParams = {'data':df, 
                  'x' : condCol,
                  'y' : y, 
                  'order' : condCat,
                  
                  's':4,
                  "linewidth": 0.1
                  ,
                  'edgecolor':'k',
                  'dodge':True
                    }




fig, ax = plt.subplots(figsize = (12/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)
fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)



plt.xlim(-0.5, 4.5)
plt.yscale('log')
y_labels = np.asarray([100, 500, 1000, 3000, 10000, 50000])
y_ticks = np.log10((y_labels))
ax.set_yticks(y_ticks, labels = (y_labels)/1000,**plotTicks)
plt.ylabel('')
plt.xlabel('')
# plt.title('Mean Effective Elasticity (kPa)', fontweight='bold', **plotChars)
plt.show()
# plt.savefig((dirToSave + 'Mean-{}_I-OptoRhoA_'+str(y)+'.pdf'))

#%%% Mean Elastic Modulus 

plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}

y = 'E_eff_log' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  
                  's':4,
                  "linewidth": 0.1
                  ,
                  'edgecolor':'k',
                  'dodge':True
                    }




fig, ax = plt.subplots(figsize = (12/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)
fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)



plt.xlim(-0.5, 4.5)
plt.yscale('log')
y_labels = np.asarray([100, 500, 1000, 3000, 10000, 50000])
y_ticks = np.log10((y_labels))
ax.set_yticks(y_ticks, labels = (y_labels)/1000,**plotTicks)
plt.ylabel('')
plt.xlabel('')
# plt.title('Mean Effective Elasticity (kPa)', fontweight='bold', **plotChars)
plt.show()
# plt.savefig((dirToSave + 'Mean-{}_I-OptoRhoA_'+str(y)+'.pdf'))

#%%% E vs H0

toPlot = df[df['dateCell'].apply(lambda x : x in pairedCells)]

fig, ax = plt.subplots(figsize=(18/SCALE_px_cm, 10/SCALE_px_cm))
fig, ax, dataSlopes = pf.EvsH0_perCompression(fig, ax, toPlot, condCat, condCol,
                                              hueType = condCol, h_ref = 500,
                                              colorScheme = 'white', 
                                              palette = palette_cond)
# fig.suptitle(str(dates), **plotChars)
plt.xticks(**plotTicks)
plt.yticks(**plotTicks)
plt.show()
plt.savefig((dirToSave + '{:}_{:}_EvH_NLI.pdf').format(str(dates), str(condCat)))


# fig, axes = plt.subplots(figsize = (13,9))
# fig, axes, avgDf_EvH = pf.EvH0_LogCellAvg(fig, axes,  dfPairs, condCat, condCol, hueType = condCol,
#                                       colorScheme = 'white', palette = palette_cond)
# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(2a)_{:}_{:}_logAvgEvH_NLI.png').format(str(dates), str(condCat)))

avgDf =  pf.createAvgDf(dataSlopes, condCol, dataFluoPath = None, e_norm = True)


#%%% EvH0 Paired

toPlot = df[df['dateCell'].apply(lambda x : x in pairedCells)]
N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvH_Pairs(fig, ax, toPlot, condCol, condCat, palette = palette_cell_point, 
                colorScheme = 'white', metric = 'compression',plotChars = plotTicks)

y_labels = np.asarray([100, 500, 1000, 3000, 10000, 25000])
y_ticks = np.log10((y_labels))
ax.set_yticks(y_ticks, labels = (y_labels)/1000,**plotTicks)

x_labels = np.asarray([100 ,250, 500, 1000, 1500])
x_ticks = np.log10(np.asarray(x_labels))
ax.set_xticks(x_ticks, labels =x_labels,**plotTicks)

#%%% Mean E_norm boxpltos

plotChars = {'color' : '#000000', 'fontsize' : 10}
plotTicks = {'color' : '#000000', 'fontsize' : 10}

y = 'E_norm' 

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : (y, 'mean'),
                  'order' : condCat,
                  's':4,
                  "linewidth": 0.1,
                  'edgecolor':'k',
                  'dodge':True
                    }



fig, ax = plt.subplots(figsize = (12/SCALE_px_cm,10/SCALE_px_cm), tight_layout = True)
fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, labels = labels,
                             palette = palette_cond,colorScheme = 'white',
                             plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)

plt.xlim(-0.5, 4.5)
plt.yscale('log')

y_labels = [100, 500, 1000, 2500, 5000, 10000, 25000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels = np.asarray(y_labels)/1000,**plotTicks)


plt.ylabel('')
plt.xlabel('')
plt.title('Normalized Elasticity', fontweight='bold', **plotChars)
plt.show()
plt.savefig((dirToSave + 'Mean-{}_I-OptoRhoA_'+str(y)+'.pdf'))

#%%% Plotnine paired plots

#%%%% Z-Difference

measure = 'surroundingDz'
stat = 'median'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, test= 'two-sided',
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
# plt.axhline(y= 0, ls = (0, (5, 10)), color = 'red', alpha = 0.4, lw = 2)

plt.title('Z-Difference (nm)', fontweight='bold', **plotChars)
plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_normPaired.pdf'))

plt.show()

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
plt.axhline(y= 0, ls = (0, (5, 10)), color = 'red', alpha = 0.4, lw = 2)

plt.title('Mean NLR per cell (A.U.)\n', fontweight='bold', **plotChars)
plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_Paired.pdf'))

plt.show()

#%%%% NLR - Normalized Pairedplot
measure = 'NLI_mod'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot, dfplots = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, 
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2.1, 1))


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Normalized NLR per Cell', fontweight='bold', **plotChars)
plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_NormPaired.pdf'))

plt.show()


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
plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_Paired-woBlebs.pdf'))



#%%%% BestH0 - Paired
#%%%% BestH0 - Pairedplot
measure = 'bestH0_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot_woHisto(dfPairs, condCol = condCol, condCat = condCat, measure = measure,  test = 'two-sided',
                     pairs = pairs, stat = stat, palette = palette_cond, logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Cortical Thickness (nm)\n ', fontweight='bold', **plotChars)
# plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_Paired.pdf'))

plt.show()

#%%%% BestH0 - Normalized Pairedplot
measure = 'bestH0_log'
stat = 'nanmean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot, dfPairsPlot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, 
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0,3))


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Normalized Thickness per Cell', fontweight='bold', **plotChars)
plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_NormPaired.pdf'))

plt.show()


#%%%% E_eff - Pairedplot
measure = 'E_eff_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot_woHisto(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, test = 'two-sided', logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 25))


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Mean Effective Elasticity (kPa)\n ', fontweight='bold', **plotChars)
plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_Paired.pdf'))

plt.show()

#%%%% E_eff - Normalized Pairedplot
measure = 'E_eff_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot, dfPairsPlot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, test = 'two-sided',
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 4))


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Normalized Effective Elasticity (kPa)\n ', fontweight='bold', **plotChars)
# plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_NormPaired.pdf'))

plt.show()
# plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_NormPaired.pdf'))


#%%%% E_normalized
measure = 'E_norm_log'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.pairedplot_woHisto(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, test = '                     -sided',
                     logScale = True,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
# plt.title('Mean Elasticity (kPa)\n ', fontweight='bold', **plotChars)

plt.show()
plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_Paired.pdf'))

#%%%% E_normalized, normalized
measure = 'E_norm'
stat = 'mean'

dfPairs, pairedCells = pf.dfCellPairs(avgDf)


plot = pf.norm_pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, palette = palette_cond, test = 'two-sided',
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 3.8))


plt.xticks([1,2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.ylabel('')
plt.xlabel('')
plt.title('Normalized Effective Elasticity (kPa)\n ', fontweight='bold', **plotChars)


plt.show()
plt.savefig((dirToSave + measure + '_I-OptoRhoA_'+str(activation)+'_NormPaired.pdf'))

#%%% Thickness vs. time

measure = 'bestH0'
activationTime = []

fig1, axes = plt.subplots(1,1, figsize=(15,10))
flatui = ["#000000", "#0000ff"]

x = (df['compNum']-1)*20
# ax = sns.lineplot(x = x, y = measure, data = df, hue = 'drug', style = 'cellID',  marker='o')
ax = sns.lineplot(x = x, y = measure, data = df, hue = condCol)

plt.ylim(0,1200)
plt.xlim(0,140)

# plt.savefig(dirToSave + '/Thickness/'+str(dates)+'_'+measure+'vsCompr'+str(manips)+'.png')
ax.get_legend().remove()

# ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(**plotChars)
plt.yticks(**plotChars)
plt.show()


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


#%%% Point-line plots

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()
N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)

#%%
testH0 = 'two-sided'
testE = 'less'
testNli = 'two-sided'
stats = 'mean'


plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('bestH0', 'mean'),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim =None, 
                                          pairs = pairs, normalize = False, marker = stats,colorScheme = 'white',
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)


# ax.get_legend().remove()
plt.show()
# plt.savefig((dirToSave + '(9a)_{:}_{:}_{:}_NLImodPointplot.png').format(str(dates), str(condCat), stats))
