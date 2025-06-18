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

import TrackAnalyser as taka0
import TrackAnalyser_V3_Manuscript_AJ as taka


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
                        }



Task = '23-04-25_M1 & 23-04-25_M2 & 23-05-10_M3 & 23-05-10_M4 & 23-05-10_M5'
        
fitsSubDir = 'VWC-Chadwick_Chapter-2_C-OptoRhoA_SequentialGlobal_15mT'

GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', 
                            fileName = fitsSubDir, save = True, PLOT = False, source = 'Python',
                            fitSettings = fitSettings, plotSettings = plotSettings,
                            fitsSubDir = fitsSubDir) # task = 'updateExisting'



#%% Calling data - Only C-OptoRhoA, testing effects of 5mT ad 15mT on mechanics

"""
Task = '22-03-31'

 'VWC-Chadwick_Chapter-2_C-OptoRhoA_OnePulseGlobal_22-03-31'
        
"""

filename =  'VWC-Chadwick_Chapter-2_C-OptoRhoA_OnePulseGlobal_22-03-31'
GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/NextCloud/Anumita - Manuscript/Figures/Chapter_2/'
data = pf.createDataTable(GlobalTable)

fitsSubDir =  'VWC-Chadwick_Chapter-2_C-OptoRhoA_OnePulseGlobal_22-03-31'

#%%% Create dataframe for plotting

styleDict =  {500:{'color': '#2986CC','marker':'o', 'label': '800ms'},
              800:{'color': '#144366','marker':'o', 'label': '500ms'},
               }

styleAct =  {'global':{'color': '#2986CC','marker':'o', 'label': 'Global Activation'},
              'away from beads':{'color': '#144366','marker':'o', 'label': 'Polarized Front'},
              'at beads':{'color': '#144366','marker':'o', 'label': 'Polarized Rear'},
               }

styleDf = pd.DataFrame(styleAct)
styleDf = styleDf.transpose()

activation_exp = [500, 800]

celltypes = ['optoRhoA. high expressing']
activation_type = ['at beads']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            # (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            # (data['R2_vwc_Full'] >= 0.90),
            # (data['H0_vwc_Full'] <= 1500) & (data['H0_vwc_Full'] > 100),
            (data['E_f_<_400'] <= 30000),
            # (data['compNum'] <= 16),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            # (data['activation exp'].apply(lambda x : x in activation_exp)),
            (data['activation type'].apply(lambda x : x in activation_type)),

            ]

df = pf.filterDf(Filters, data)
df_h0 = taka.getMatchingFits(df, fitsSubDir = fitsSubDir, fitType = 'H0')
h0_method = ['Chadwick', '%f_10']
df_h0 = df_h0[(df_h0['method'] == h0_method[0]) & (df_h0['zone'] == h0_method[1])]
df = pd.merge(df, df_h0[['cellID', 'compNum', 'H0', 'nbPts']], on = ['cellID', 'compNum'], how = 'left')
df = df.rename(columns={'H0':'{:}_{:}_H0'.format(h0_method[0], h0_method[1])})

condCol, condCat = 'activation type', activation_type


labels = list(styleDf['label'].values)

palette_cond = pf.getSnsPalette(condCat, styleAct)

pairs = [condCat]

plotChars = {'color' : '#000000', 'fontsize' : 11}
plotTicks = {'color' : '#000000', 'fontsize' : 11}

avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[(avgDf[('compNum', 'count')] > 2)]

#%%% H0 @ 10% Force vs. Compression


cellsChosen = avgDf['cellID'].values
toPlot = df[df['cellID'].apply(lambda x : x in cellsChosen)]
colorblind_type = "Deuteranomaly"
palette = distinctipy.get_colors(len(toPlot['cellID'].unique()), colorblind_type=colorblind_type)

split_point = 5
y = 'Chadwick_%f_10_H0'

toPlot['new_compNum'] = toPlot['compNum']
toPlot['new_compNum'].loc[toPlot['compNum'] < split_point] = 0
toPlot['new_compNum'].loc[toPlot['compNum'] >= split_point] = toPlot['compNum'] - (split_point-1)

toPlot = toPlot.groupby('cellID', group_keys=False).apply(pf.normalize_by_first_five, y)

y_new = y + '_norm'

plottingParams1 = {'data':toPlot,
                  'x':'new_compNum', 
                  'hue':'cellID',
                  'kind':'point',
                  'markersize' : 15,
                  'palette':palette,
                  'marker':'.',
                  'linewidth': 2,
                  'errorbar':'sd',
                  'errwidth':2,
                  'alpha':0.8,
                  'aspect':1, 
                  'capsize':.05
                 }


plottingParams2 = {'data':toPlot,
                  'x':'new_compNum', 
                   
                  'hue':'cellID',
                  'marker': None,
                  'linewidth':2,
                  'palette':palette,
                  'alpha':1,
                 }
g = sns.catplot(y = y, legend_out = False,  **plottingParams1)
g.fig.set_size_inches(8.26, 2.6)
ax1 = g.ax
ax1 = sns.lineplot(y = y,   **plottingParams2)

plottingParams3 = {'data':toPlot,
                  'x':'new_compNum', 
                   'ls':'--',
                  'hue':'cellID',
                  'marker': None,
                  'linewidth':2,
                  'palette':palette,
                  'alpha':0.5,
                 }

ax2 = ax1.twinx()
ax2 = sns.lineplot(y = y_new, **plottingParams3)

for i in range(1):
    plt.axvline(x= 0.1, color='blue', linestyle='-', 
                ymin=0, ymax=0.15, linewidth=4)
    
    
# plt.legend(labelcolor='linecolor', bbox_to_anchor=(1.05, 1))

ax1.set_ylim(0, 1500)
ax2.set_ylim(0, 3)

for ax in [ax1, ax2]:

    ax.tick_params(axis='both', labelsize = plotTicks['fontsize'])
   
    ax.set_xlabel('Compressions Post-Activation', **plotChars)

ax1.set_ylabel('Cortical Thickness (nm)', **plotChars)

ax2.set_ylabel('Normalized Cortical Thickness', rotation = 270, va = 'bottom', **plotChars)

# plt.title('Cortical thickness @ 10% Force (nm)', fontweight='bold', **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + y_new + 'vsComp_{:}-{:}_C-Opto.pdf').format( str(activation_exp), str(activation_type)), dpi = 200)

plt.show()

#%% Calling data - Only C-OptoRhoA, testing sequential activation (big blast then every loop)

"""
Task = '22-06-21 & 22-05-31'

 'VWC-Chadwick_Chapter-2_C-OptoRhoA_Sequential_22-06-21'
        
"""

filename = 'VWC-Chadwick_Chapter-2_C-OptoRhoA_Sequential_22-06-21'
GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/NextCloud/Anumita - Manuscript/Figures/Chapter_2/'
data = pf.createDataTable(GlobalTable)

fitsSubDir =  'VWC-Chadwick_Chapter-2_C-OptoRhoA_Sequential_22-06-21'

#%%% Create dataframe for plotting


styleAct =  { 'away from beads':{'color': '#144366','marker':'o', 'label': 'Polarized Front'},
              'at beads':{'color': '#144366','marker':'o', 'label': 'Polarized Rear'},
               }

styleDf = pd.DataFrame(styleAct)
styleDf = styleDf.transpose()

labels = list(styleDf['label'].values)


celltypes = ['optoRhoA']
activation_type = list(styleAct.keys())

manipIDs = ['22-05-31_M2', '22-05-31_M4', '22-05-31_M7'] #, '22-06-21_M2']
# manipIDs = ['22-06-21_M2']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            # (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            # (data['R2_vwc_Full'] >= 0.90),
            # (data['H0_vwc_Full'] <= 1500) & (data['H0_vwc_Full'] > 100),
            (data['E_f_<_400'] <= 30000),
            # (data['compNum'] <= 16),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            (data['manipID'].apply(lambda x : x in manipIDs)),
            (data['activation type'].apply(lambda x : x in activation_type)),

            ]

df = pf.filterDf(Filters, data)
df_h0 = taka.getMatchingFits(df, fitsSubDir = fitsSubDir, fitType = 'H0')
h0_method = ['Chadwick', '%f_10']
df_h0 = df_h0[(df_h0['method'] == h0_method[0]) & (df_h0['zone'] == h0_method[1])]
df = pd.merge(df, df_h0[['cellID', 'compNum', 'H0', 'nbPts']], on = ['cellID', 'compNum'], how = 'left')
df = df.rename(columns={'H0':'{:}_{:}_H0'.format(h0_method[0], h0_method[1])})

condCol, condCat = 'activation type', activation_type

labels = list(styleDf['label'].values)

palette_cond = pf.getSnsPalette(condCat, styleAct)

pairs = [condCat]

plotChars = {'color' : '#000000', 'fontsize' : 11}
plotTicks = {'color' : '#000000', 'fontsize' : 11}


avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[(avgDf[('compNum', 'count')] > 4)]

#%%% H0 @ 10% Force vs. Compression


cellsChosen = avgDf['cellID'].values
toPlot = df[df['cellID'].apply(lambda x : x in cellsChosen)]
colorblind_type = "Deuteranomaly"
palette = distinctipy.get_colors(len(toPlot['cellID'].unique()), colorblind_type=colorblind_type)

split_point = 5
y = 'Chadwick_%f_10_H0'

toPlot['new_compNum'] = toPlot['compNum']
toPlot['new_compNum'].loc[toPlot['compNum'] < split_point] = 0
toPlot['new_compNum'].loc[toPlot['compNum'] >= split_point] = toPlot['compNum'] - (split_point-1)

# Apply this to each cellID group
toPlot = toPlot.groupby('cellID', group_keys=False).apply(pf.normalize_by_first_five, y)

y_new = y + '_norm'

plottingParams1 = {'data':toPlot,
                  'x':'new_compNum', 
                  'hue':'cellID',
                  'kind':'point',
                  'markersize' : 15,
                  'palette':palette,
                  'marker':'.',
                  'linewidth': 2,
                  'errorbar':'sd',
                  'errwidth':2,
                  'alpha':0.8,
                  'aspect':1, 
                  'capsize':.05
                 }


plottingParams2 = {'data':toPlot,
                  'x':'new_compNum', 
                   
                  'hue':'cellID',
                  'marker': None,
                  'linewidth':2,
                  'palette':palette,
                  'alpha':1,
                 }
g = sns.catplot(y = y, legend_out = False,  **plottingParams1)
g.fig.set_size_inches(8.26, 2.6)
ax1 = g.ax
ax1 = sns.lineplot(y = y, legend = False,  **plottingParams2)

plottingParams3 = {'data':toPlot,
                  'x':'new_compNum', 
                   'ls':'--',
                  'hue':'cellID',
                  'marker': None,
                  'linewidth':2,
                  'palette':palette,
                  'alpha':0.5,
                 }

ax2 = ax1.twinx()
ax2 = sns.lineplot(y = y_new, **plottingParams3)

for i in range(1):
    plt.axvline(x= 0.1, color='blue', linestyle='-', 
                ymin=0, ymax=0.15, linewidth=4)
    
for i in range(10):
    plt.axvline(x= 1.1 + i, color='blue', linestyle='-', 
                    ymin=0, ymax=0.1, linewidth=2)
    
# plt.legend(labelcolor='linecolor', bbox_to_anchor=(1.05, 1))

ax1.set_ylim(0, 1500)
ax2.set_ylim(0, 3)

plt.xlim(-0.1, 10)


for ax in [ax1, ax2]:

    ax.tick_params(axis='both', labelsize = plotTicks['fontsize'])
   
    ax.set_xlabel('Compressions Post-Activation', **plotChars)

ax1.set_ylabel('Cortical Thickness (nm)', **plotChars)

ax2.set_ylabel('Normalized Cortical Thickness', rotation = 270, va = 'bottom', **plotChars)


plt.tight_layout()
plt.savefig((dirToSave +y_new+ 'vsComp_Sequence-{:}_C-Opto.pdf').format( str(activation_type)), dpi = 100)

plt.show()


plt.show()

#%% Calling data - Only C-OptoRhoA, testing sequential activation (big blast then every loop)

"""
Task = '22-10-06'

  'VWC-Chadwick_Chapter-2_C-OptoRhoA_Sequential-Fields_22-10-06'
        
"""

filename =  'VWC-Chadwick_Chapter-2_C-OptoRhoA_Sequential-Fields_22-10-06'
GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/NextCloud/Anumita - Manuscript/Figures/Chapter_2/'
data = pf.createDataTable(GlobalTable)

fitsSubDir =  'VWC-Chadwick_Chapter-2_C-OptoRhoA_Sequential-Fields_22-10-06'

#%%% Create dataframe for plotting

styleDict =  {5.0:{'color': '#2986CC','marker':'o', 'label': '5mT'},
              15.0:{'color': '#144366','marker':'o', 'label': '15mT'},
               }

styleDf = pd.DataFrame(styleDict)
styleDf = styleDf.transpose()

labels = list(styleDf['label'].values)


celltypes = ['optoRhoA']
magFields = [15.0]

# manipIDs = ['22-05-31_M2', '22-05-31_M4', '22-05-31_M7'] 

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            # (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            # (data['R2_vwc_Full'] >= 0.90),
            # (data['H0_vwc_Full'] <= 1500) & (data['H0_vwc_Full'] > 100),
            (data['E_f_<_400'] <= 30000),
            (data['compNum'] <= 16),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            # (data['manipID'].apply(lambda x : x in manipIDs)),
            (data['normal field'].apply(lambda x : x in magFields)),

            ]

df = pf.filterDf(Filters, data)
df_h0 = taka.getMatchingFits(df, fitsSubDir = fitsSubDir, fitType = 'H0')
h0_method = ['Chadwick', '%f_10']
df_h0 = df_h0[(df_h0['method'] == h0_method[0]) & (df_h0['zone'] == h0_method[1])]
df = pd.merge(df, df_h0[['cellID', 'compNum', 'H0', 'nbPts']], on = ['cellID', 'compNum'], how = 'left')
df = df.rename(columns={'H0':'{:}_{:}_H0'.format(h0_method[0], h0_method[1])})

condCol, condCat = 'normal field', magFields

palette_cond = pf.getSnsPalette(condCat, styleDict)

pairs = [condCat]

plotChars = {'color' : '#000000', 'fontsize' : 11}
plotTicks = {'color' : '#000000', 'fontsize' : 11}

avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[(avgDf[('compNum', 'count')] > 4)]

#%%% H0 @ 10% Force vs. Compression

dfPairs, pairedCells = pf.dfCellPairs(avgDf)

cellsChosen = dfPairs['cellID'].values
toPlot = df[df['cellID'].apply(lambda x : x in cellsChosen)]

#Because activation done at compression 6, to avoid having different activation points, I remove the first point and keep the last 5
toPlot = toPlot.drop(toPlot[(toPlot['cellID'] == '22-10-06_M5_P3_C3') & (toPlot['compNum'] == 1)].index)
mask = (toPlot['cellID'] == '22-10-06_M5_P3_C3') & toPlot['compNum'].between(2, 6)
toPlot.loc[mask, 'compNum'] -= 1

toPlot.loc[toPlot['drug'] == 'activation', 'compNum'] += 5

split_point = 5
y = 'Chadwick_%f_10_H0'
colorblind_type = "Deuteranomaly"
palette = distinctipy.get_colors(len(toPlot['dateCell'].unique()), colorblind_type=colorblind_type)

toPlot['new_compNum'] = toPlot['compNum']
toPlot['new_compNum'].loc[toPlot['compNum'] < split_point] = 0
toPlot['new_compNum'].loc[toPlot['compNum'] >= split_point] = toPlot['compNum'] - split_point

# Apply this to each cellID group
toPlot = toPlot.groupby('dateCell', group_keys=False).apply(pf.normalize_by_first_five, y)

y_new = y + '_norm'

plottingParams1 = {'data':toPlot,
                  'x':'new_compNum', 
                  'hue':'dateCell',
                  'kind':'point',
                  'markersize' : 15,
                  'palette':palette,
                  'marker':'.',
                  'linewidth': 2,
                  'errorbar':'sd',
                  'errwidth':2,
                  'alpha':0.8,
                  'aspect':1, 
                  'capsize':.05
                 }


plottingParams2 = {'data':toPlot,
                  'x':'new_compNum', 
                   
                  'hue':'dateCell',
                  'marker': None,
                  'linewidth':2,
                  'palette':palette,
                  'alpha':1,
                 }
g = sns.catplot(y = y, legend_out = False,  **plottingParams1)
g.fig.set_size_inches(8.26, 2.6)
ax1 = g.ax
ax1 = sns.lineplot(y = y, legend = False,  **plottingParams2)

plottingParams3 = {'data':toPlot,
                  'x':'new_compNum', 
                   'ls':'--',
                  'hue':'dateCell',
                  'marker': None,
                  'linewidth':2,
                  'palette':palette,
                  'alpha':0.5,
                 }
                 

ax2 = ax1.twinx()
ax2 = sns.lineplot(y = y_new, legend = False, **plottingParams3)

for i in range(10):
    plt.axvline(x= 0.1 + i, color='blue', linestyle='-', 
                    ymin=0, ymax=0.1, linewidth=2)
    
# plt.legend(labelcolor='linecolor', bbox_to_anchor=(1.05, 1))


ax1.set_ylim(0, 1500)
ax2.set_ylim(0, 3)

plt.xlim(-0.1, 10)

 
for ax in [ax1, ax2]:

    ax.tick_params(axis='both', labelsize = plotTicks['fontsize'])
   
    ax.set_xlabel('Compressions Post-Activation', **plotChars)

ax1.set_ylabel('Cortical Thickness (nm)', **plotChars)

ax2.set_ylabel('Normalized Cortical Thickness', rotation = 270, va = 'bottom', **plotChars)



plt.title('Cortical thickness @ 10% Force (nm)', fontweight='bold', **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + y_new + 'vsComp_Sequence-{:}_C-Opto.pdf').format(str(magFields) ), dpi = 100)

plt.show()

#%% Calling data - Only C-OptoRhoA, testing sequential activation (big blast then every loop)

"""
Task = '23-05-23'

'VWC-Chadwick_Chapter-2_C-OptoRhoA_Sequential_15mT'
        
"""

filename ='VWC-Chadwick_Chapter-2_C-OptoRhoA_Sequential_15mT'
GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/NextCloud/Anumita - Manuscript/Figures/Chapter_2/'
data = pf.createDataTable(GlobalTable)

fitsSubDir ='VWC-Chadwick_Chapter-2_C-OptoRhoA_Sequential_15mT'

#%%% Create dataframe for plotting



celltypes = ['optoRhoA']
magFields = [15.0]
manips = ['M1', 'M3']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] >= 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_f_<_400'] <= 30000),
            (data['compNum'] <= 16),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            (data['manip'].apply(lambda x : x in manips)),
            (data['normal field'].apply(lambda x : x in magFields)),

            ]

df = pf.filterDf(Filters, data)
df_h0 = taka.getMatchingFits(df, fitsSubDir = fitsSubDir, fitType = 'H0')
h0_method = ['Chadwick', '%f_10']
df_h0 = df_h0[(df_h0['method'] == h0_method[0]) & (df_h0['zone'] == h0_method[1])]
df = pd.merge(df, df_h0[['cellID', 'compNum', 'H0', 'nbPts']], on = ['cellID', 'compNum'], how = 'left')
df = df.rename(columns={'H0':'{:}_{:}_H0'.format(h0_method[0], h0_method[1])})

plotChars = {'color' : '#000000', 'fontsize' : 13}
plotTicks = {'color' : '#000000', 'fontsize' : 13}

avgDf = pf.createAvgDf(df, condCol = 'activation type')
avgDf = avgDf[(avgDf[('compNum', 'count')] > 4)]

#%%% H0 @ 10% Force vs. Compression

dfPairs, pairedCells = pf.dfCellPairs(avgDf)

cellsChosen = dfPairs['cellID'].values
toPlot = df[df['cellID'].apply(lambda x : x in cellsChosen)]

#Because activation done at compression 6, to avoid having different activation points, I remove the first point and keep the last 5
# toPlot = toPlot.drop(toPlot[(toPlot['cellID'] == '23-05-23_M1_P1_C1') & (toPlot['compNum'] == 1)].index)
# mask = (toPlot['cellID'] == '23-05-23_M1_P1_C1') & toPlot['compNum'].between(2, 6)
# toPlot.loc[mask, 'compNum'] -= 1

toPlot.loc[toPlot['drug'] == 'activation', 'compNum'] += 6

split_point = 6
y = 'Chadwick_%f_10_H0'

palette = distinctipy.get_colors(len(toPlot['dateCell'].unique()))

toPlot['new_compNum'] = toPlot['compNum']
toPlot['new_compNum'].loc[toPlot['compNum'] < split_point] = 0
toPlot['new_compNum'].loc[toPlot['compNum'] >= split_point] = toPlot['compNum'] - split_point

# Apply this to each cellID group
toPlot = toPlot.groupby('dateCell', group_keys=False).apply(pf.normalize_by_first_five, y)

y_new = y + '_norm'

plottingParams1 = {'data':toPlot,
                  'x':'new_compNum', 
                  'hue':'dateCell',
                  'markersize' : 15,
                  'palette':palette,
                  'linewidth': 2,
                  'errorbar':'sd',
                  'errwidth':2,
                  'alpha':0.4,
                  'capsize':.05
                 }


plottingParams2 = {'data':toPlot,
                  'x':'new_compNum', 
                  'linewidth':2,
                  'palette':palette,
                  'markersize' : 15,
                  'marker':'.',
                  'color' : 'k',
                  'alpha':1,
                 }

plottingParams3 = {'data':toPlot,
                  'x':'new_compNum', 
                  'hue':'dateCell',
                  'markersize' : 15,
                  'palette':palette,
                  'linewidth': 2,
                  'alpha':0.4,
                 }

fig = plt.figure(figsize = (25/SCALE_px_cm, 9/SCALE_px_cm))

ax1 = fig.add_subplot(121)
g = sns.pointplot(y = y, legend = False, marker='.', ax = ax1, **plottingParams1)
ax2 = sns.lineplot(y = y, marker = None, legend = False,  **plottingParams3)
ax1 = sns.lineplot(y = y, legend = False,  **plottingParams2)


ax2 = fig.add_subplot(122)
g = sns.pointplot(y = y_new,legend = False, marker='.', ax = ax2, **plottingParams1)

ax2 = sns.lineplot(y = y_new, marker = None,   **plottingParams3)

ax2 = sns.lineplot(y = y_new,   **plottingParams2)


for ax in [ax1, ax2]:

    ax.tick_params(axis='both', labelsize = plotTicks['fontsize'])
    ax.set_xlabel('')
    for i in range(3):
        ax.axvline(x= 0.1 + 3*i, color='blue', linestyle='-', 
                        ymin=0, ymax=0.1, linewidth=2)
ax1.set_ylabel('Cortical Thickness (nm)', **plotChars)

ax2.set_ylabel('Normalized Cortical Thickness',  **plotChars)
ax1.set_ylim(0, 1500)
ax2.set_ylim(0, 3.2)

plt.xlim(-0.1, 10)
fig.supxlabel('Compressions Post-Activation',**plotChars)

# plt.title('Cortical thickness @ 10% Force (nm)', fontweight='bold', **plotChars)

plt.tight_layout(pad=1.0)
plt.savefig((dirToSave + y_new + 'vsComp_Sequence-{:}-23-05-23_C-Opto.pdf').format(str(magFields) ), dpi = 200)

plt.show()

#%% Calling data - Only C-OptoRhoA, testing sequential activation (big blast then every loop)

"""
Task = '23-04-25_M1 & 23-04-25_M2 & 23-05-10_M3 & 23-05-10_M4 & 23-05-10_M5'

'VWC-Chadwick_Chapter-2_C-OptoRhoA_SequentialGlobal_15mT'
        
"""

filename ='VWC-Chadwick_Chapter-2_C-OptoRhoA_SequentialGlobal_15mT'
GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/NextCloud/Anumita - Manuscript/Figures/Chapter_2/'
data = pf.createDataTable(GlobalTable)

fitsSubDir ='VWC-Chadwick_Chapter-2_C-OptoRhoA_SequentialGlobal_15mT'

#%%% Create dataframe for plotting



celltypes = ['optoRhoA']
magFields = [15.0]
# manips = ['M3', 'M4', 'M5']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] >= 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_f_<_400'] <= 30000),
            (data['compNum'] <= 16),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            # (data['manip'].apply(lambda x : x in manips)),
            (data['normal field'].apply(lambda x : x in magFields)),

            ]

df = pf.filterDf(Filters, data)
df_h0 = taka.getMatchingFits(df, fitsSubDir = fitsSubDir, fitType = 'H0')
h0_method = ['Chadwick', '%f_10']
df_h0 = df_h0[(df_h0['method'] == h0_method[0]) & (df_h0['zone'] == h0_method[1])]
df = pd.merge(df, df_h0[['cellID', 'compNum', 'H0', 'nbPts']], on = ['cellID', 'compNum'], how = 'left')
df = df.rename(columns={'H0':'{:}_{:}_H0'.format(h0_method[0], h0_method[1])})

plotChars = {'color' : '#000000', 'fontsize' : 13}
plotTicks = {'color' : '#000000', 'fontsize' : 13}

avgDf = pf.createAvgDf(df, condCol = 'activation type')
avgDf = avgDf[(avgDf[('compNum', 'count')] > 4)]

#%%% H0 @ 10% Force vs. Compression

dfPairs, pairedCells = pf.dfCellPairs(avgDf)

cellsChosen = dfPairs['cellID'].values
toPlot = df[df['cellID'].apply(lambda x : x in cellsChosen)]

#To avoid having different activation points, I remove this cell
toPlot = toPlot.drop(toPlot[(toPlot['dateCell'] == '23-04-25_P1_C7')].index)
# mask = (toPlot['cellID'] == '23-05-23_M1_P1_C1') & toPlot['compNum'].between(2, 6)
# toPlot.loc[mask, 'compNum'] -= 1

toPlot.loc[toPlot['drug'] == 'activation', 'compNum'] += 6

split_point = 6
y = 'Chadwick_%f_10_H0'
palette = distinctipy.get_colors(len(toPlot['dateCell'].unique()))

toPlot['new_compNum'] = toPlot['compNum']
toPlot['new_compNum'].loc[toPlot['compNum'] < split_point] = 0
toPlot['new_compNum'].loc[toPlot['compNum'] >= split_point] = toPlot['compNum'] - split_point

# Apply this to each cellID group
toPlot = toPlot.groupby('dateCell', group_keys=False).apply(pf.normalize_by_first_five, y)

y_new = y + '_norm'

plottingParams1 = {'data':toPlot,
                  'x':'new_compNum', 
                  'hue':'dateCell',
                  'markersize' : 15,
                  'palette':palette,
                  'linewidth': 2,
                  'errorbar':'sd',
                  'errwidth':2,
                  'alpha':0.4,
                  'capsize':.05
                 }


plottingParams2 = {'data':toPlot,
                  'x':'new_compNum', 
                  'linewidth':2,
                  'palette':palette,
                  'markersize' : 15,
                  'marker':'.',
                  'color' : 'k',
                  'alpha':1,
                 }

plottingParams3 = {'data':toPlot,
                  'x':'new_compNum', 
                  'hue':'dateCell',
                  'markersize' : 15,
                  'palette':palette,
                  'linewidth': 2,
                  'alpha':0.4,
                 }

fig = plt.figure(figsize = (25/SCALE_px_cm, 9/SCALE_px_cm))

ax1 = fig.add_subplot(121)
g = sns.pointplot(y = y, legend = False, marker='.', ax = ax1, **plottingParams1)
ax2 = sns.lineplot(y = y, marker = None, legend = False,  **plottingParams3)
ax1 = sns.lineplot(y = y, legend = False,  **plottingParams2)


ax2 = fig.add_subplot(122)
g = sns.pointplot(y = y_new,legend = False, marker='.', ax = ax2, **plottingParams1)

ax2 = sns.lineplot(y = y_new, marker = None,   **plottingParams3)

ax2 = sns.lineplot(y = y_new,   **plottingParams2)

     
     
for ax in [ax1, ax2]:

    ax.tick_params(axis='both', labelsize = plotTicks['fontsize'])
    ax.set_xlabel('')
    for i in range(3):
        ax.axvline(x= 0.1 + 3*i, color='blue', linestyle='-', 
                        ymin=0, ymax=0.1, linewidth=2)
ax1.set_ylabel('Cortical Thickness (nm)', **plotChars)

ax2.set_ylabel('Normalized Cortical Thickness',  **plotChars)
ax1.set_ylim(0, 1500)
ax2.set_ylim(0, 3.2)

plt.xlim(-0.1, 10)
fig.supxlabel('Compressions Post-Activation',**plotChars)

# plt.title('Cortical thickness @ 10% Force (nm)', fontweight='bold', **plotChars)

plt.tight_layout(pad=1.0)
plt.savefig((dirToSave + y_new + 'vsComp_SequenceGlobal-{:}-23-05-23_C-Opto.pdf').format(str(magFields) ), dpi = 200)

plt.show()