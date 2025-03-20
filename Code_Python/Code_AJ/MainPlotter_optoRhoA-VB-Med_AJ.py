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
                'methods_H0':['Chadwick', 'VWC'],
                'zones_H0':['%f_100'],
                'method_bestH0':'VWC', 
                'zone_bestH0':'%f_100',
                'doVWCFit' : True,
                'VWCFitMethods' : ['Full'],
                'doStressRegionFits' : False,
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

plot_stressCenters = [ii for ii in range(100, 4000, 100)]
plot_stressHalfWidth = 100

plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':False,
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
                        }
    
# Task = '24-12-14 & 24-12-20 & 25-01-14 & 25-01-23 & 25-01-21'
# Task = '25-01-23_M1_P1_C6'

Task = '24-12-14 & 24-12-20_M1 & 24-12-20_M2 & & 24-12-20_M5 & 25-01-14 & 25-01-23 & 25-01-21  & 25-02-28'

fitsSubDir = 'VWC_optoRhoAVB-NS_updated_25-03-04'
# fitsSubDir = 'VWC_optoRhoAVB-NS_ForBoxplots'

GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', 
                            fileName = fitsSubDir, save = True, PLOT = False, source = 'Python',
                            fitSettings = fitSettings, plotSettings = plotSettings,
                            fitsSubDir = fitsSubDir) # task = 'updateExisting'

#%% Calling data - Using FACsd cells (medium expression), first mechanics experiments on VB cell line
# Dates avilable : 24-09-26

filename = 'VWC_optoRhoAVB-Med_24-09-26_24-10-01'

GlobalTable = taka.getMergedTable(filename)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/OptoRhoA-VB/24.09.26/24-10-01_Plots/'

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)
dates = ['24-09-26']
drugs = ['doxy', 'doxy_act']

manips = ['M1', 'M2', 'M3', 'M4']
labels = ['Dox', 'Global Activation']

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1500),
            # (data['E_eff'] <= 30000),
            # (data['compNum'] <= 6),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['drug'].apply(lambda x : x in drugs)),
            ]

df = pf.filterDf(Filters, data)
condCol, condCat = 'drug', drugs
avgDf = pf.createAvgDf(df, condCol)

df = df.drop(df[(df['drug'] == 'doxy_act') & (df['compNum'] == 1)].index)

pairs = [['doxy', 'doxy_act']] 


plotChars = {'color' : '#ffffff', 'fontsize' : 20}
plotTicks = {'color' : '#ffffff', 'fontsize' : 18}
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
palette_cond = ['#7f7fff', '#0000FF']

swarmPointSize = 6


#%%%% Plot NLImod


########################################

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax = plt.subplots(figsize=(7, 6))

fig, ax = pf.rainplot(fig, ax, condCat, palette = palette_cond, 
                             labels = labels, pairs = pairs, shiftBox = 0.1, shiftSwarm = 0.0,
                             colorScheme = 'black', test = 'non-param' ,pointSize = 40,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

plt.ylim(-4,4.5)
plt.ylabel('NLR', **plotChars)
plt.xlabel(' ', **plotChars)
# plt.tight_layout()
plt.savefig((dirToSave + '(0a)_{:}_{:}_NLRrainplot.png').format(str(dates), str(condCat)))
plt.show()


########################################

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                                    hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(-3,3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLImodPLot.png').format(str(dates), str(condCat)))
plt.show()

######## vs. Compressions #########

fig, ax = plt.subplots(figsize = (13,9))

condition = 'doxy_act'
df_comp = df[df['drug'] == condition]

plottingParams = {'data':df_comp, 
                  'x' : 'compNum', 
                  'y' : 'NLI_mod',
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), 
                                    pairs = None, hueType = 'date', plotType = 'swarm',
                                    labels = [], plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1b)_{:}_{:}_NLImodPLot-Comps.png').format(str(dates), condition))


######## cell average #########

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1c)_{:}_{:}_NLImodPLot_cellAvg.png').format(str(dates), str(condCat)))
plt.show()

######## Paired plot #########

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1d)_{:}_{:}_NLImodPLot_paired.png').format(str(dates), str(condCat)))
plt.show()

######## coloured Compressions #########

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'linewidth' : 1,
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, plotType = 'swarm',
                                    hueType = 'compNum', labels = [], plottingParams = plottingParams,
                                    plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1e)_{:}_{:}_{:}_NLImodPLot-DiffComps.png').format(str(dates), condition, str(condCat)))

######## averaged by date #########

fig, ax = plt.subplots(figsize = (13,9))
plt.style.use('default')
fig.patch.set_facecolor('black')

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'hue' : 'date',
                  'linewidth' : 1, 
                  'markersize' : 100, 
                  'errorbar':('ci', 95),
                  'alpha' : 0.6
                    }

ax = sns.pointplot(**plottingParams)

plt.ylim(-1.5, 1.5)
fig.suptitle(str(dates), **plotChars)
if labels != []:
    xticks = np.arange(len(condCat))
    plt.xticks(xticks, labels, rotation = 25, **plotChars)
    
plt.yticks(**plotChars)
plt.legend(fontsize = 16)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(1f)_{:}_{:}_NLImodPLot-DateAverage.png').format(str(dates), str(condCat)))

plt.style.use('default')

#%%%% NLImod vs Compression
# fig, ax = plt.subplots(figsize = (13,9))

# plottingParams = {'data':df, 
#                   'x' : 'compNum', 
#                   'y' : 'NLI_mod',   
#                   'linewidth' : 1, 
#                   }
                    
# ax = sns.lineplot(**plottingParams)
# plt.ylim(-1, 2)

# plt.show()


#%%%% Plot NLI

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.plotNLI(fig, ax, df, condCat, condCol,
                            pairs = pairs, labels = labels, **plotChars)

fig.suptitle(str(dates), **plotChars)
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(1b)_{:}_{:}_NLIPLot.png').format(str(dates), str(condCat)))

plt.style.use('default')


####### vs. Compressions #########
condition = 'doxy_act'
df_comp = df[df.drug == condition]

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.plotNLI(fig, ax, df_comp, condCat =  np.sort(df_comp.compNum.unique()), condCol = 'compNum',
                            pairs = None, labels = [], **plotChars)

fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(1b)_{:}_{:}_{:}_NLIPLot-Comps.png').format(str(dates), condition, str(condCat)))

plt.style.use('default')

#%%%% E vs H0
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol, hueType = 'NLI_Plot')
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_EvH_NLI.png').format(str(dates), str(condCat)))


fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2c)_{:}_{:}_EvH_CellID.png').format(str(dates), str(condCat)))

#%%%% E vs H0 log average
 
fig, axes = plt.subplots(figsize = (13,9))
fig, axes, avgDf = pf.EvH0_LogCellAvg(fig, axes,  avgDf, condCat, condCol, hueType = 'NLI_Plot')
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_logAvgEvH_NLI.png').format(str(dates), str(condCat)))


fig, axes = plt.subplots(figsize = (13,9))
fig, axes, avgDf = pf.EvH0_LogCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_logAvgEvH_Conditions.png').format(str(dates), str(condCat)))

# fig, axes = plt.subplots(1, 2, figsize = (18,10))
# fig, axes = pf.EvH0_LogCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
# plt.legend(fontsize = 6, ncol = len(condCat))
# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(2c)_{:}_{:}_wAvgEvH_CellID.png').format(str(dates), str(condCat)))

################## E_Normalised #####################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_norm', 'logAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :4, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond, labels = labels,
                             plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(2d)_{:}_{:}_E-NormBoxplot_logAvg.png').format(str(dates), str(condCat)))
plt.show()

 #%%%% E vs H0 weighted average
 
# fig, axes = plt.subplots(1, 2, figsize = (18,10))
# fig, axes, avgDf = pf.EvH0_wCellAvg(fig, axes,  avgDf, condCat, condCol, hueType = 'NLI_Plot')
# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(2a)_{:}_{:}_wAvgEvH_NLI.png').format(str(dates), str(condCat)))


fig, axes = plt.subplots(1, 2, figsize = (18,10))
fig, axes, avgDf = pf.EvH0_wCellAvg(fig, axes, avgDf, condCat, condCol, h_ref = 500,
                                    palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
# plt.savefig((dirToSave + '(2b)_{:}_{:}_wAvgEvH_Conditions.png').format(str(dates), str(condCat)))

# # fig, axes = plt.subplots(1, 2, figsize = (18,10))
# # fig, axes = pf.EvH0_wCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
# # plt.legend(fontsize = 6, ncol = len(condCat))
# # fig.suptitle(str(dates), **plotChars)
# # plt.tight_layout()
# # plt.show()
# # plt.savefig((dirToSave + '(2c)_{:}_{:}_wAvgEvH_CellID.png').format(str(dates), str(condCat)))

# ################## E_Normalised #####################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_norm', 'wAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond, labels = labels,
                             plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
# plt.yscale('log')
# y_labels = [100, 500, 5000, 10000, 50000]
# y_ticks = np.log10(np.asarray(y_labels))
# ax.set_yticks(y_ticks, labels =y_labels,**plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(2d)_{:}_{:}_E-NormBoxplot_wAvg.png').format(str(dates), str(condCat)))
plt.show()

#%%% Box plots - H0

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'bestH0_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 4, 
                   }

######################## Hue type 'CellID'#######################
# fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)


# fig.suptitle(str(dates), **plotChars)
# plt.legend(fontsize = 6, ncol = 6)
# # plt.ylim(0,2500)
# plt.yscale('log')
# y_ticks = [100, 250, 500, 1000, 2500]
# ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(3a)_{:}_{:}_H0Boxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'bestH0_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'hue_order' : ['non-linear', 'intermediate', 'linear'],
                  'size' : swarmPointSize,  
                    }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(3c)_{:}_{:}_H0Boxplot_NLI.png').format(str(dates), str(condCat)))

####################### H0 vs compression ############################

fig, ax = plt.subplots(figsize = (13,9))
condition = 'doxy_act'
df_comp = df[df['drug'] == condition]

plottingParams = {'data':df_comp, 
                  'x' : 'compNum', 
                  'y' : 'bestH0_log',
                  'linewidth' : 1, 
                  'size' :6, 
                   }

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), pairs = None, hueType = 'date',
                                    labels = [], plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
# plt.yscale('log')
plt.ylim(0,1500)
# y_ticks = [100, 250, 500, 1000, 2500]
# ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.savefig((dirToSave + '(3d)_{:}_{:}_H0Boxplot_CompNum.png').format(str(dates), str(condCat)))
plt.show()

####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('bestH0_log', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.savefig((dirToSave + '(3e)_{:}_{:}_H0Boxplot_meancellAverage.png').format(str(dates), str(condCat)))
plt.show()



#%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 4, 
                   }

ylim = 20000

######################## Hue type 'CellID'#######################
# N = len(df['cellID'].unique())
# palette_cell = distinctipy.get_colors(N)
# fig, ax = plt.subplots(figsize = (13,9))
# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)

# # plt.ylim(0, ylim)
# plt.legend(fontsize = 6, ncol = 6)
# fig.suptitle(str(dates), **plotChars)
# # plt.yscale('log')
# # plt.ylim(100, 30000)
# # y_ticks = [100, 500, 5000, 10000, 30000]
# ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff_log',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : 4,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

######################## Paired 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff_log',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : 4,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff_log', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_cellAverage.png').format(str(dates), str(condCat)))
plt.show()


####################### cell weighted average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'wAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1000, 50000)
y_ticks = [100, 500, 5000, 10000, 50000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_weightedCellAverage.png').format(str(dates), str(condCat)))
plt.show()


#%%% Boxplots - CtFieldThickness

df_ctField = df.drop_duplicates(subset = 'ctFieldThickness')

plottingParams = {'data':df_ctField, 
                  'x' : condCol, 
                  'y' : 'ctFieldThickness',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

ylim = 20000

######################## Hue type 'CellID'#######################
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
plt.legend(fontsize = 6, ncol = 6)
fig.suptitle(str(dates), **plotChars)
plt.ylim(0,1500)
plt.show()
plt.savefig((dirToSave + '(4a)_{:}_{:}_ctFieldThickness_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)

# plt.ylim(0,1500)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_ctFieldThickness_Conditions.png').format(str(dates), str(condCat)))


####################### cell average ############################

# fig, ax = plt.subplots(figsize = (13,9))
# dates = np.unique(df['date'].values)

# plottingParams = {'data':avgDf, 
#                   'x' : (condCol, 'first'), 
#                   'y' : ('E_eff', 'median'),
#                   'order' : condCat,
#                   'linewidth' : 1, 
#                   'size' :swarmPointSize, 
#                     }

# fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
#                              hueType = None, palette = palette_crosslink,
#                                     labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.yscale('log')
# plt.ylim(1000, 30000)
# y_ticks = [100, 500, 5000, 10000, 30000]
# ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

# plt.tight_layout()
# plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_cellAverage.png').format(str(dates), str(condCat)))
# plt.show()

#%%% Pointplots 

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()

N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)

testH0 = 'two-sided'
testE = 'less'
testNli = 'greater'

stats = 'mean'
plottingParams = { 'x' : (condCol, 'first'), 
                  'y' : ('H0_vwc_Full', stats),
                  'linewidth' : 1, 
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

ylim = 1500
fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.xlim((-2,3))
plt.tight_layout()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_H0Pointplot.png').format(str(dates), str(condCat), stats))
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_H = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,3), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7b)_{:}_{:}_{:}_H0Pointplot-Normalised.png').format(str(dates), str(condCat), stats))
plt.show()


ylim = 10000
plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'wAvg'),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = 'wAvg',
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(8a)_{:}_{:}_{:}_EPointplot.png').format(str(dates), str(condCat), stats))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_E = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
                                          pairs = pairs, normalize = True, marker = 'wAvg',
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(8b)_{:}_{:}_{:}_EPointplot-Normalised.png').format(str(dates), str(condCat), stats))

nonOutliers_E = np.asarray(dfP_E['dateCell', 'first'][(dfP_E[condCol, 'first'] == condCat[1]) & (dfP_E['normMeasure', 'wAvg'] < 3.0)].values)

plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', stats),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-3,3), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

ax.get_legend().remove()
plt.show()
plt.savefig((dirToSave + '(9a)_{:}_{:}_{:}_NLImodPointplot.png').format(str(dates), str(condCat), stats))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-3,3), styleType = ('dateCell', 'first'),
                                          pairs = pairs, normalize = False, marker = stats, hueType = ('date', 'first'),
                                          test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(9b)_{:}_{:}_{:}_NLImodPointplot_dates.png').format(str(dates), str(condCat), stats))


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-1.5,1.5), 
                                          pairs = pairs, normalize = False, marker = stats, hueType = ('date', 'first'),
                                          test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(9c)_{:}_{:}_{:}_NLImodPointplot_datesAvg.png').format(str(dates), str(condCat), stats))


# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP_nli = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-6,6), 
#                                           pairs = pairs, normalize = True, marker = stats,
#                                           test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(9b)_{:}_{:}_{:}_NLImodPointplot-Normalised.png').format(str(dates), str(condCat), stats))

#### E-normalised 

plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('E_norm', 'wAvg'),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim =(0,ylim), 
                                          pairs = pairs, normalize = False, marker = 'wAvg', 
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(10a)_{:}_{:}_E-norm.png').format(str(dates), str(condCat)))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
                                          pairs = pairs, normalize = True, marker = 'wAvg', 
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(10b)_{:}_{:}_E-doubleNorm.png').format(str(dates), str(condCat)))

#%%% NLI averaged paired plots vs fluctuations V.1

# dfPairs, pairedCells = pf.dfCellPairs(avgDf)

# fig, ax = plt.subplots(figsize = (13,9))

# N_point = len(dfPairs['dateCell', 'first'].unique())
# palette_cell_point = distinctipy.get_colors(N_point)

# plottingParams = {'x' : ('normFluctu', 'first'), 
#                   'y' : ('NLI_mod', 'mean'),
#                   'data' : dfPairs,
#                   'hue' : ('dateCell', 'first'),
#                   'palette' : palette_cell_point,
#                   'linewidth' : 1,
#                   'markersize' : 10,
#                   'markeredgecolor':'black', 
#                   'marker' : 'o',
#                    }

# ax = sns.lineplot(**plottingParams)

# plt.tight_layout()
# # plt.legend(loc=2, prop={'size': 7}, ncol =5)
# ax.get_legend().remove()
# plt.show()
# plt.savefig((dirToSave + '(11)_{:}_{:}_NLIParisvFluctu.png').format(str(dates), str(condCat)))

#%%% NLI averaged paired plots vs fluctuations V.2

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
fig, ax = plt.subplots(figsize = (13,9))
N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)

fig, ax = pf.NLIPairsvFluctu(fig, ax, dfPairs, condCol, condCat, palette = palette_cell_point, 
                plotChars = plotChars)


plt.xlim(0,2)
plt.show()
plt.savefig((dirToSave + '(11)_{:}_{:}_NLIParisvFluctu.png').format(str(dates), str(condCat)))


#%%% NLI vs Fluctuation

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'x' : ('normFluctu', 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'data' : avgDf,
                  'hue' : (condCol, 'first'),
                  'palette' : palette_cond,
                  'hue_order' : condCat,
                  's' : 100
                   }

fig, ax = pf.NLIvFluctu(fig, ax, plottingParams = plottingParams, plotChars = plotChars)

plt.xlim(0, 1.5)
plt.show()
plt.savefig((dirToSave + '(11a)_{:}_{:}_NLIvFluctu.png').format(str(dates), str(condCat)))

#%%%  NLI-corr vs Fluctuation

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'palette' : palette_cond,
                  's' : 100,
                  'hue' : condCol,
                  'hue_order' : condCat,
                   }

fig, ax, data_nli = pf.NLIcorrvFluctu(fig, ax, data = df, plottingParams = plottingParams, 
                                      plotChars = plotChars)


plt.xlim(0, 1.5)
plt.show()
plt.savefig((dirToSave + '(11b)_{:}_{:}_NLI-corrvFluctu.png').format(str(dates), str(condCat)))


#%% Calling data - Using FACsd cells (medium expression), first mechanics experiments on confocal

# Dates avilable : 24-10-21
filename = 'VWC_optoRhoAVB-Med_24-10-21'

GlobalTable = taka.getMergedTable(filename)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/OptoRhoA-VB/24.10.21/'

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)
dates = ['24-10-21']
drugs = ['doxy', 'doxy_act']

labels = []

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1500),
            (data['E_eff'] <= 30000),
            ]

df = pf.filterDf(Filters, data)
condCol, condCat = 'drug', drugs
avgDf = pf.createAvgDf(df, condCol)

pairs = [['doxy', 'doxy_act']]

plotChars = {'color' : '#ffffff', 'fontsize' : 20}
plotTicks = {'color' : '#ffffff', 'fontsize' : 18}

N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
palette_cond = ['#bbbbbb', '#0a75ad']

swarmPointSize = 6


#%%%% Plot NLImod


########################################

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax = plt.subplots(figsize=(7, 6))

fig, ax = pf.rainplot(fig, ax, condCat, palette = palette_cond, 
                             labels = labels, pairs = pairs, shiftBox = 0.1, shiftSwarm = 0.0,
                             colorScheme = 'black', test = 'non-param' ,pointSize = 40,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

plt.ylim(-4,4.5)
plt.ylabel('NLR', **plotChars)
plt.xlabel(' ', **plotChars)
# plt.tight_layout()
plt.savefig((dirToSave + '(0a)_{:}_{:}_NLRrainplot.png').format(str(dates), str(condCat)))
plt.show()


################### box plots #######################
fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                                    hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(-3,3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLImodPLot.png').format(str(dates), str(condCat)))
plt.show()

######## vs. Compressions #########

fig, ax = plt.subplots(figsize = (13,9))

condition = 'doxy_act'
df_comp = df[df['drug'] == condition]

plottingParams = {'data':df_comp, 
                  'x' : 'compNum', 
                  'y' : 'NLI_mod',
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), 
                                    pairs = None, hueType = 'date', plotType = 'swarm',
                                    labels = [], plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1b)_{:}_{:}_NLImodPLot-Comps.png').format(str(dates), condition))


######## cell average #########

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1c)_{:}_{:}_NLImodPLot_cellAvg.png').format(str(dates), str(condCat)))
plt.show()

######## Paired plot #########

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1d)_{:}_{:}_NLImodPLot_paired.png').format(str(dates), str(condCat)))
plt.show()

######## coloured Compressions #########

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'linewidth' : 1,
                  'size' :swarmPointSize, 
                    }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, plotType = 'swarm',
                                    hueType = 'compNum', labels = [], plottingParams = plottingParams,
                                    plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1e)_{:}_{:}_{:}_NLImodPLot-DiffComps.png').format(str(dates), condition, str(condCat)))


#%%%% NLImod vs Compression
# fig, ax = plt.subplots(figsize = (13,9))

# plottingParams = {'data':df, 
#                   'x' : 'compNum', 
#                   'y' : 'NLI_mod',   
#                   'linewidth' : 1, 
#                   }
                    
# ax = sns.lineplot(**plottingParams)
# plt.ylim(-1, 2)

# plt.show()


#%%%% Plot NLI

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.plotNLI(fig, ax, df, condCat, condCol,
                            pairs = pairs, labels = labels, **plotChars)

fig.suptitle(str(dates), **plotChars)
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(1b)_{:}_{:}_NLIPLot.png').format(str(dates), str(condCat)))

plt.style.use('default')


####### vs. Compressions #########
condition = 'doxy'
df_comp = df[df.drug == condition]

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.plotNLI(fig, ax, df_comp, condCat =  np.sort(df_comp.compNum.unique()), condCol = 'compNum',
                            pairs = None, labels = [], **plotChars)

fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(1b)_{:}_{:}_{:}_NLIPLot-Comps.png').format(str(dates), condition, str(condCat)))

plt.style.use('default')

#%%%% E vs H0
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol, hueType = 'NLI_Plot')
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_EvH_NLI.png').format(str(dates), str(condCat)))


fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
plt.legend(fontsize = 6, ncol = len(condCat))
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2c)_{:}_{:}_EvH_CellID.png').format(str(dates), str(condCat)))

#%%%% E vs H0 log average
 
fig, axes = plt.subplots(figsize = (13,9))
fig, axes, avgDf = pf.EvH0_LogCellAvg(fig, axes,  avgDf, condCat, condCol, hueType = 'NLI_Plot')
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_logAvgEvH_NLI.png').format(str(dates), str(condCat)))


fig, axes = plt.subplots(figsize = (13,9))
fig, axes, avgDf = pf.EvH0_LogCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_logAvgEvH_Conditions.png').format(str(dates), str(condCat)))

# fig, axes = plt.subplots(1, 2, figsize = (18,10))
# fig, axes = pf.EvH0_LogCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
# plt.legend(fontsize = 6, ncol = len(condCat))
# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(2c)_{:}_{:}_wAvgEvH_CellID.png').format(str(dates), str(condCat)))

################## E_Normalised #####################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_norm', 'logAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :4, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond, labels = labels,
                             plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**pltTicks)

plt.tight_layout()
plt.savefig((dirToSave + '(2d)_{:}_{:}_E-NormBoxplot_logAvg.png').format(str(dates), str(condCat)))
plt.show()

 #%%%% E vs H0 weighted average
 
# fig, axes = plt.subplots(1, 2, figsize = (18,10))
# fig, axes, avgDf = pf.EvH0_wCellAvg(fig, axes,  avgDf, condCat, condCol, hueType = 'NLI_Plot')
# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(2a)_{:}_{:}_wAvgEvH_NLI.png').format(str(dates), str(condCat)))


fig, axes = plt.subplots(1, 2, figsize = (18,10))
fig, axes, avgDf = pf.EvH0_wCellAvg(fig, axes, avgDf, condCat, condCol, h_ref = 500,
                                    palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
# plt.savefig((dirToSave + '(2b)_{:}_{:}_wAvgEvH_Conditions.png').format(str(dates), str(condCat)))

# # fig, axes = plt.subplots(1, 2, figsize = (18,10))
# # fig, axes = pf.EvH0_wCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
# # plt.legend(fontsize = 6, ncol = len(condCat))
# # fig.suptitle(str(dates), **plotChars)
# # plt.tight_layout()
# # plt.show()
# # plt.savefig((dirToSave + '(2c)_{:}_{:}_wAvgEvH_CellID.png').format(str(dates), str(condCat)))

# ################## E_Normalised #####################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_norm', 'wAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond, labels = labels,
                             plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
# plt.yscale('log')
# y_labels = [100, 500, 5000, 10000, 50000]
# y_ticks = np.log10(np.asarray(y_labels))
# ax.set_yticks(y_ticks, labels =y_labels,**plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(2d)_{:}_{:}_E-NormBoxplot_wAvg.png').format(str(dates), str(condCat)))
plt.show()

#%%% Box plots - H0

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'bestH0_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 4, 
                   }

######################## Hue type 'CellID'#######################
# fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)


# fig.suptitle(str(dates), **plotChars)
# plt.legend(fontsize = 6, ncol = 6)
# # plt.ylim(0,2500)
# plt.yscale('log')
# y_ticks = [100, 250, 500, 1000, 2500]
# ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(3a)_{:}_{:}_H0Boxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'bestH0_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'hue_order' : ['non-linear', 'intermediate', 'linear'],
                  'size' : swarmPointSize,  
                    }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(3c)_{:}_{:}_H0Boxplot_NLI.png').format(str(dates), str(condCat)))

####################### H0 vs compression ############################

fig, ax = plt.subplots(figsize = (13,9))
condition = 'doxy_act'
df_comp = df[df['drug'] == condition]

plottingParams = {'data':df_comp, 
                  'x' : 'compNum', 
                  'y' : 'bestH0_log',
                  'linewidth' : 1, 
                  'size' :6, 
                   }

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), pairs = None, hueType = 'date',
                                    labels = [], plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
# plt.yscale('log')
plt.ylim(0,1500)
# y_ticks = [100, 250, 500, 1000, 2500]
# ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.savefig((dirToSave + '(3d)_{:}_{:}_H0Boxplot_CompNum.png').format(str(dates), str(condCat)))
plt.show()

####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('bestH0_log', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.savefig((dirToSave + '(3e)_{:}_{:}_H0Boxplot_meancellAverage.png').format(str(dates), str(condCat)))
plt.show()



#%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 4, 
                   }

ylim = 20000

######################## Hue type 'CellID'#######################
# N = len(df['cellID'].unique())
# palette_cell = distinctipy.get_colors(N)
# fig, ax = plt.subplots(figsize = (13,9))
# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)

# # plt.ylim(0, ylim)
# plt.legend(fontsize = 6, ncol = 6)
# fig.suptitle(str(dates), **plotChars)
# # plt.yscale('log')
# # plt.ylim(100, 30000)
# # y_ticks = [100, 500, 5000, 10000, 30000]
# ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff_log',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : 4,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

######################## Paired 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff_log',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : 4,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff_log', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_cellAverage.png').format(str(dates), str(condCat)))
plt.show()


####################### cell weighted average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'wAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1000, 50000)
y_ticks = [100, 500, 5000, 10000, 50000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_weightedCellAverage.png').format(str(dates), str(condCat)))
plt.show()


#%%% Boxplots - CtFieldThickness

df_ctField = df.drop_duplicates(subset = 'ctFieldThickness')

plottingParams = {'data':df_ctField, 
                  'x' : condCol, 
                  'y' : 'ctFieldThickness',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

ylim = 20000

######################## Hue type 'CellID'#######################
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
plt.legend(fontsize = 6, ncol = 6)
fig.suptitle(str(dates), **plotChars)
plt.ylim(0,1500)
plt.show()
plt.savefig((dirToSave + '(4a)_{:}_{:}_ctFieldThickness_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)

# plt.ylim(0,1500)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_ctFieldThickness_Conditions.png').format(str(dates), str(condCat)))


####################### cell average ############################

# fig, ax = plt.subplots(figsize = (13,9))
# dates = np.unique(df['date'].values)

# plottingParams = {'data':avgDf, 
#                   'x' : (condCol, 'first'), 
#                   'y' : ('E_eff', 'median'),
#                   'order' : condCat,
#                   'linewidth' : 1, 
#                   'size' :swarmPointSize, 
#                     }

# fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
#                              hueType = None, palette = palette_crosslink,
#                                     labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.yscale('log')
# plt.ylim(1000, 30000)
# y_ticks = [100, 500, 5000, 10000, 30000]
# ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

# plt.tight_layout()
# plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_cellAverage.png').format(str(dates), str(condCat)))
# plt.show()

#%%% Pointplots 

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()

N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)

testH0 = 'two-sided'
testE = 'less'
testNli = 'greater'

stats = 'mean'
plottingParams = { 'x' : (condCol, 'first'), 
                  'y' : ('H0_vwc_Full', stats),
                  'linewidth' : 1, 
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

ylim = 1500
fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.xlim((-2,3))
plt.tight_layout()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_H0Pointplot.png').format(str(dates), str(condCat), stats))
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_H = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,3), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7b)_{:}_{:}_{:}_H0Pointplot-Normalised.png').format(str(dates), str(condCat), stats))
plt.show()


ylim = 10000
plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'wAvg'),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = 'wAvg',
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(8a)_{:}_{:}_{:}_EPointplot.png').format(str(dates), str(condCat), stats))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_E = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
                                          pairs = pairs, normalize = True, marker = 'wAvg',
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(8b)_{:}_{:}_{:}_EPointplot-Normalised.png').format(str(dates), str(condCat), stats))

nonOutliers_E = np.asarray(dfP_E['dateCell', 'first'][(dfP_E[condCol, 'first'] == condCat[1]) & (dfP_E['normMeasure', 'wAvg'] < 3.0)].values)

plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', stats),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-3,3), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

ax.get_legend().remove()
plt.show()
plt.savefig((dirToSave + '(9a)_{:}_{:}_{:}_NLImodPointplot.png').format(str(dates), str(condCat), stats))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-3,3), styleType = ('dateCell', 'first'),
                                          pairs = pairs, normalize = False, marker = stats, hueType = ('date', 'first'),
                                          test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(9b)_{:}_{:}_{:}_NLImodPointplot_dates.png').format(str(dates), str(condCat), stats))


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-1.5,1.5), 
                                          pairs = pairs, normalize = False, marker = stats, hueType = ('date', 'first'),
                                          test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(9c)_{:}_{:}_{:}_NLImodPointplot_datesAvg.png').format(str(dates), str(condCat), stats))


# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP_nli = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-6,6), 
#                                           pairs = pairs, normalize = True, marker = stats,
#                                           test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(9b)_{:}_{:}_{:}_NLImodPointplot-Normalised.png').format(str(dates), str(condCat), stats))

#### E-normalised 

plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('E_norm', 'wAvg'),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim =(0,ylim), 
                                          pairs = pairs, normalize = False, marker = 'wAvg', 
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(10a)_{:}_{:}_E-norm.png').format(str(dates), str(condCat)))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
                                          pairs = pairs, normalize = True, marker = 'wAvg', 
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(10b)_{:}_{:}_E-doubleNorm.png').format(str(dates), str(condCat)))

#%%% NLI averaged paired plots vs fluctuations V.1

# dfPairs, pairedCells = pf.dfCellPairs(avgDf)

# fig, ax = plt.subplots(figsize = (13,9))

# N_point = len(dfPairs['dateCell', 'first'].unique())
# palette_cell_point = distinctipy.get_colors(N_point)

# plottingParams = {'x' : ('normFluctu', 'first'), 
#                   'y' : ('NLI_mod', 'mean'),
#                   'data' : dfPairs,
#                   'hue' : ('dateCell', 'first'),
#                   'palette' : palette_cell_point,
#                   'linewidth' : 1,
#                   'markersize' : 10,
#                   'markeredgecolor':'black', 
#                   'marker' : 'o',
#                    }

# ax = sns.lineplot(**plottingParams)

# plt.tight_layout()
# # plt.legend(loc=2, prop={'size': 7}, ncol =5)
# ax.get_legend().remove()
# plt.show()
# plt.savefig((dirToSave + '(11)_{:}_{:}_NLIParisvFluctu.png').format(str(dates), str(condCat)))

#%%% NLI averaged paired plots vs fluctuations V.2

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
fig, ax = plt.subplots(figsize = (13,9))
N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)

fig, ax = pf.NLIPairsvFluctu(fig, ax, dfPairs, condCol, condCat, palette = palette_cell_point, 
                plotChars = plotChars)


plt.xlim(0,2)
plt.show()
plt.savefig((dirToSave + '(11)_{:}_{:}_NLIParisvFluctu.png').format(str(dates), str(condCat)))


#%%% NLI vs Fluctuation

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'x' : ('normFluctu', 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'data' : avgDf,
                  'hue' : (condCol, 'first'),
                  'palette' : palette_cond,
                  'hue_order' : condCat,
                  's' : 100
                   }

fig, ax = pf.NLIvFluctu(fig, ax, plottingParams = plottingParams, plotChars = plotChars)

plt.xlim(0, 1.5)
plt.show()
plt.savefig((dirToSave + '(11a)_{:}_{:}_NLIvFluctu.png').format(str(dates), str(condCat)))

#%%%  NLI-corr vs Fluctuation

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'palette' : palette_cond,
                  's' : 100,
                  'hue' : condCol,
                  'hue_order' : condCat,
                   }

fig, ax, data_nli = pf.NLIcorrvFluctu(fig, ax, data = df, plottingParams = plottingParams, 
                                      plotChars = plotChars)


plt.xlim(0, 1.5)
plt.show()
plt.savefig((dirToSave + '(11b)_{:}_{:}_NLI-corrvFluctu.png').format(str(dates), str(condCat)))

#%% Calling data - Using Clones

# Dates avilable : 24-11-13, 24-11-15
filename = 'VWC_optoRhoAVB-Med_update_24-11-18'

GlobalTable = taka.getMergedTable(filename)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/OptoRhoA-VB/24.11.13_24.11.15/'

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)
dates = ['24-11-13'] 
# drugs = [ 'doxy', 'doxy_act']
manips = ['M2', 'M3']

# wells = ['P2']

labels = ['Dox', 'Global Activation']

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1500),
            (data['E_eff'] <= 30000),
            (data['date'].apply(lambda x : x in dates)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['wellID'].apply(lambda x : x in wells))
            ]

df = pf.filterDf(Filters, data)
condCol, condCat = 'manip', manips
avgDf = pf.createAvgDf(df, condCol)

pairs = [['M2', 'M3']]


plotChars = {'color' : '#ffffff', 'fontsize' : 20}
plotTicks = {'color' : '#ffffff', 'fontsize' : 18}

N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
palette_cond = ['#4a4a4a', '#7f81fe', '#0004FD']

swarmPointSize = 6


#%%%% Plot NLImod

########################################

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax = plt.subplots(figsize=(7, 6))

fig, ax = pf.rainplot(fig, ax, condCat, palette = palette_cond, 
                             labels = labels, pairs = pairs, shiftBox = 0.1, shiftSwarm = 0.0,
                             colorScheme = 'black', test = 'non-param' ,pointSize = 40,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

plt.ylim(-4,4.5)
plt.ylabel('NLR', **plotChars)
plt.xlabel(' ', **plotChars)
# plt.tight_layout()
plt.savefig((dirToSave + '(0a)_{:}_{:}_NLRrainplot.png').format(str(dates), str(condCat)))
plt.show()


################### box plots #######################
fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                                    hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(-3,3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLImodPLot.png').format(str(dates), str(condCat)))
plt.show()

######## vs. Compressions #########

fig, ax = plt.subplots(figsize = (13,9))

condition = 'doxy_act'
df_comp = df[df['drug'] == condition]

plottingParams = {'data':df_comp, 
                  'x' : 'compNum', 
                  'y' : 'NLI_mod',
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), 
                                    pairs = None, hueType = 'date', plotType = 'swarm',
                                    labels = [], plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1b)_{:}_{:}_NLImodPLot-Comps.png').format(str(dates), condition))


######## cell average #########

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1c)_{:}_{:}_NLImodPLot_cellAvg.png').format(str(dates), str(condCat)))
plt.show()


######## coloured Compressions #########

# fig, ax = plt.subplots(figsize = (13,9))

# plottingParams = {'data':df, 
#                   'x' : condCol, 
#                   'y' : 'NLI_mod',
#                   'order' : condCat,
#                   'linewidth' : 1,
#                   'size' :swarmPointSize, 
#                     }

# fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, plotType = 'swarm',
#                                     hueType = 'compNum', labels = [], plottingParams = plottingParams,
#                                     plotChars = plotChars)

# plt.ylim(-3, 3)
# fig.suptitle(str(dates), **plotChars)
# plt.yticks(**pltTicks)
# plt.xticks(**pltTicks)
# plt.tight_layout()
# plt.savefig((dirToSave + '(1e)_{:}_{:}_{:}_NLImodPLot-DiffComps.png').format(str(dates), condition, str(condCat)))

######## coloured dates #########

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'linewidth' : 1,
                  'size' :swarmPointSize, 
                    }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, plotType = 'swarm',
                                    hueType = 'date', labels = [], plottingParams = plottingParams,
                                    plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1f)_{:}_{:}_NLImodPLot-Dates.png').format(str(dates), str(condCat)))

#%%%% NLImod vs Compression
# fig, ax = plt.subplots(figsize = (13,9))

# plottingParams = {'data':df, 
#                   'x' : 'compNum', 
#                   'y' : 'NLI_mod',   
#                   'linewidth' : 1, 
#                   }
                    
# ax = sns.lineplot(**plottingParams)
# plt.ylim(-1, 2)

# plt.show()

#%%%% Plot NLI

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.plotNLI(fig, ax, df, condCat, condCol,
                            pairs = pairs, labels = labels, **plotChars)

fig.suptitle(str(dates), **plotChars)
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(1b)_{:}_{:}_NLIPLot.png').format(str(dates), str(condCat)))

plt.style.use('default')


####### vs. Compressions #########
condition = 'doxy'
df_comp = df[df.drug == condition]

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.plotNLI(fig, ax, df_comp, condCat =  np.sort(df_comp.compNum.unique()), condCol = 'compNum',
                            pairs = None, labels = [], **plotChars)

fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(1b)_{:}_{:}_{:}_NLIPLot-Comps.png').format(str(dates), condition, str(condCat)))

plt.style.use('default')

#%%%% E vs H0
# fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
# fig, ax = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol, hueType = 'NLI_Plot')
# fig.suptitle(str(dates), **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(2a)_{:}_{:}_EvH_NLI.png').format(str(dates), str(condCat)))


fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))

# fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
# fig, ax = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
# plt.legend(fontsize = 6, ncol = len(condCat))
# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(2c)_{:}_{:}_EvH_CellID.png').format(str(dates), str(condCat)))

#%%%% E vs H0 log average
 
fig, axes = plt.subplots(figsize = (13,9))
fig, axes, avgDf = pf.EvH0_LogCellAvg(fig, axes,  avgDf, condCat, condCol, hueType = 'NLI_Plot')
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_logAvgEvH_NLI.png').format(str(dates), str(condCat)))


fig, axes = plt.subplots(figsize = (13,9))
fig, axes, avgDf = pf.EvH0_LogCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_logAvgEvH_Conditions.png').format(str(dates), str(condCat)))

# fig, axes = plt.subplots(1, 2, figsize = (18,10))
# fig, axes = pf.EvH0_LogCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
# plt.legend(fontsize = 6, ncol = len(condCat))
# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(2c)_{:}_{:}_wAvgEvH_CellID.png').format(str(dates), str(condCat)))

################## E_Normalised #####################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_norm', 'logAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :4, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond, labels = labels,
                             plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(2d)_{:}_{:}_E-NormBoxplot_logAvg.png').format(str(dates), str(condCat)))
plt.show()

 #%%%% E vs H0 weighted average
 
# fig, axes = plt.subplots(1, 2, figsize = (18,10))
# fig, axes, avgDf = pf.EvH0_wCellAvg(fig, axes,  avgDf, condCat, condCol, hueType = 'NLI_Plot')
# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(2a)_{:}_{:}_wAvgEvH_NLI.png').format(str(dates), str(condCat)))


fig, axes = plt.subplots(1, 2, figsize = (18,10))
fig, axes, avgDf = pf.EvH0_wCellAvg(fig, axes, avgDf, condCat, condCol, h_ref = 500,
                                    palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
# plt.savefig((dirToSave + '(2b)_{:}_{:}_wAvgEvH_Conditions.png').format(str(dates), str(condCat)))

# # fig, axes = plt.subplots(1, 2, figsize = (18,10))
# # fig, axes = pf.EvH0_wCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
# # plt.legend(fontsize = 6, ncol = len(condCat))
# # fig.suptitle(str(dates), **plotChars)
# # plt.tight_layout()
# # plt.show()
# # plt.savefig((dirToSave + '(2c)_{:}_{:}_wAvgEvH_CellID.png').format(str(dates), str(condCat)))

# ################## E_Normalised #####################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_norm', 'wAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond, labels = labels,
                             plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
# plt.yscale('log')
# y_labels = [100, 500, 5000, 10000, 50000]
# y_ticks = np.log10(np.asarray(y_labels))
# ax.set_yticks(y_ticks, labels =y_labels,**plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(2d)_{:}_{:}_E-NormBoxplot_wAvg.png').format(str(dates), str(condCat)))
plt.show()

#%%% Box plots - H0

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'bestH0_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 4, 
                   }

######################## Hue type 'CellID'#######################
# fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)


# fig.suptitle(str(dates), **plotChars)
# plt.legend(fontsize = 6, ncol = 6)
# # plt.ylim(0,2500)
# plt.yscale('log')
# y_ticks = [100, 250, 500, 1000, 2500]
# ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(3a)_{:}_{:}_H0Boxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))

ax.set_yticks(y_ticks, labels =y_labels,**pltTicks)
plt.xticks(**pltTicks)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'bestH0_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'hue_order' : ['non-linear', 'intermediate', 'linear'],
                  'size' : swarmPointSize,  
                    }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(3c)_{:}_{:}_H0Boxplot_NLI.png').format(str(dates), str(condCat)))

####################### H0 vs compression ############################

fig, ax = plt.subplots(figsize = (13,9))
condition = 'doxy_act'
df_comp = df[df['drug'] == condition]

plottingParams = {'data':df_comp, 
                  'x' : 'compNum', 
                  'y' : 'bestH0_log',
                  'linewidth' : 1, 
                  'size' :6, 
                   }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), pairs = None, hueType = 'date',
                                    labels = [], plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
# plt.yscale('log')
plt.ylim(0,1500)
# y_ticks = [100, 250, 500, 1000, 2500]
# ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.savefig((dirToSave + '(3d)_{:}_{:}_H0Boxplot_CompNum.png').format(str(dates), str(condCat)))
plt.show()

####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('bestH0_log', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.savefig((dirToSave + '(3e)_{:}_{:}_H0Boxplot_meancellAverage.png').format(str(dates), str(condCat)))
plt.show()



#%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 4, 
                   }

ylim = 20000

######################## Hue type 'CellID'#######################
# N = len(df['cellID'].unique())
# palette_cell = distinctipy.get_colors(N)
# fig, ax = plt.subplots(figsize = (13,9))
# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)

# # plt.ylim(0, ylim)
# plt.legend(fontsize = 6, ncol = 6)
# fig.suptitle(str(dates), **plotChars)
# # plt.yscale('log')
# # plt.ylim(100, 30000)
# # y_ticks = [100, 500, 5000, 10000, 30000]
# ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff_log',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : 4,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

######################## Paired 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff_log',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : 4,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff_log', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_cellAverage.png').format(str(dates), str(condCat)))
plt.show()


####################### cell weighted average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'wAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1000, 50000)
y_ticks = [100, 500, 5000, 10000, 50000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_weightedCellAverage.png').format(str(dates), str(condCat)))
plt.show()


#%%% Boxplots - CtFieldThickness

df_ctField = df.drop_duplicates(subset = 'ctFieldThickness')

plottingParams = {'data':df_ctField, 
                  'x' : condCol, 
                  'y' : 'ctFieldThickness',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

ylim = 20000

######################## Hue type 'CellID'#######################
# N = len(df['cellID'].unique())
# palette_cell = distinctipy.get_colors(N)
# fig, ax = plt.subplots(figsize = (13,9))
# fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)

# # plt.ylim(0, ylim)
# plt.legend(fontsize = 6, ncol = 6)
# fig.suptitle(str(dates), **plotChars)
# plt.ylim(0,1500)
# plt.yticks(**pltTicks)
# plt.xticks(**pltTicks)
# plt.show()
# plt.savefig((dirToSave + '(4a)_{:}_{:}_ctFieldThickness_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
# plt.ylim(0,1500)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_ctFieldThickness_Conditions.png').format(str(dates), str(condCat)))


####################### cell average ############################

# fig, ax = plt.subplots(figsize = (13,9))
# dates = np.unique(df['date'].values)

# plottingParams = {'data':avgDf, 
#                   'x' : (condCol, 'first'), 
#                   'y' : ('E_eff', 'median'),
#                   'order' : condCat,
#                   'linewidth' : 1, 
#                   'size' :swarmPointSize, 
#                     }

# fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
#                              hueType = None, palette = palette_crosslink,
#                                     labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.yscale('log')
# plt.ylim(1000, 30000)
# y_ticks = [100, 500, 5000, 10000, 30000]
# ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

# plt.tight_layout()
# plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_cellAverage.png').format(str(dates), str(condCat)))
# plt.show()

#%%% Pointplots 

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()

N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)

testH0 = 'two-sided'
testE = 'less'
testNli = 'greater'

stats = 'mean'
plottingParams = { 'x' : (condCol, 'first'), 
                  'y' : ('H0_vwc_Full', stats),
                  'linewidth' : 1, 
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

ylim = 1500
fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.xlim((-2,3))
plt.tight_layout()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_H0Pointplot.png').format(str(dates), str(condCat), stats))
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_H = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,3), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7b)_{:}_{:}_{:}_H0Pointplot-Normalised.png').format(str(dates), str(condCat), stats))
plt.show()


ylim = 10000
plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'wAvg'),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = 'wAvg',
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(8a)_{:}_{:}_{:}_EPointplot.png').format(str(dates), str(condCat), stats))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_E = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
                                          pairs = pairs, normalize = True, marker = 'wAvg',
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(8b)_{:}_{:}_{:}_EPointplot-Normalised.png').format(str(dates), str(condCat), stats))

nonOutliers_E = np.asarray(dfP_E['dateCell', 'first'][(dfP_E[condCol, 'first'] == condCat[1]) & (dfP_E['normMeasure', 'wAvg'] < 3.0)].values)

plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', stats),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-3,3), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# ax.get_legend().remove()
plt.show()
plt.savefig((dirToSave + '(9a)_{:}_{:}_{:}_NLImodPointplot.png').format(str(dates), str(condCat), stats))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-3,3), styleType = ('dateCell', 'first'),
                                          pairs = pairs, normalize = False, marker = stats, hueType = ('date', 'first'),
                                          test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(9b)_{:}_{:}_{:}_NLImodPointplot_dates.png').format(str(dates), str(condCat), stats))


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-1.5,1.5), 
                                          pairs = pairs, normalize = False, marker = stats, hueType = ('date', 'first'),
                                          test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(9c)_{:}_{:}_{:}_NLImodPointplot_datesAvg.png').format(str(dates), str(condCat), stats))



fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_nli = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-6,6), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(9d)_{:}_{:}_{:}_NLImodPointplot-Normalised.png').format(str(dates), str(condCat), stats))

#### E-normalised 

plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('E_norm', 'wAvg'),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim =(0,ylim), 
                                          pairs = pairs, normalize = False, marker = 'wAvg', 
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(10a)_{:}_{:}_E-norm.png').format(str(dates), str(condCat)))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
                                          pairs = pairs, normalize = True, marker = 'wAvg', 
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(10b)_{:}_{:}_E-doubleNorm.png').format(str(dates), str(condCat)))

#%%% NLI averaged paired plots vs fluctuations V.1

# dfPairs, pairedCells = pf.dfCellPairs(avgDf)

# fig, ax = plt.subplots(figsize = (13,9))

# N_point = len(dfPairs['dateCell', 'first'].unique())
# palette_cell_point = distinctipy.get_colors(N_point)

# plottingParams = {'x' : ('normFluctu', 'first'), 
#                   'y' : ('NLI_mod', 'mean'),
#                   'data' : dfPairs,
#                   'hue' : ('dateCell', 'first'),
#                   'palette' : palette_cell_point,
#                   'linewidth' : 1,
#                   'markersize' : 10,
#                   'markeredgecolor':'black', 
#                   'marker' : 'o',
#                    }

# ax = sns.lineplot(**plottingParams)

# plt.tight_layout()
# # plt.legend(loc=2, prop={'size': 7}, ncol =5)
# ax.get_legend().remove()
# plt.show()
# plt.savefig((dirToSave + '(11)_{:}_{:}_NLIParisvFluctu.png').format(str(dates), str(condCat)))

#%%% NLI averaged paired plots vs fluctuations V.2

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
fig, ax = plt.subplots(figsize = (13,9))
N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)

fig, ax = pf.NLIPairsvFluctu(fig, ax, dfPairs, condCol, condCat, palette = palette_cell_point, 
                plotChars = plotChars)


plt.xlim(0,2)
plt.show()
plt.savefig((dirToSave + '(11)_{:}_{:}_NLIParisvFluctu.png').format(str(dates), str(condCat)))


#%%% NLI vs Fluctuation

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'x' : ('normFluctu', 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'data' : avgDf,
                  'hue' : (condCol, 'first'),
                  'palette' : palette_cond,
                  'hue_order' : condCat,
                  's' : 100
                   }

fig, ax = pf.NLIvFluctu(fig, ax, plottingParams = plottingParams, plotChars = plotChars)

plt.xlim(0, 1.5)
plt.show()
plt.savefig((dirToSave + '(11a)_{:}_{:}_NLIvFluctu.png').format(str(dates), str(condCat)))

#%%%  NLI-corr vs Fluctuation

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'palette' : palette_cond,
                  's' : 100,
                  'hue' : condCol,
                  'hue_order' : condCat,
                   }

fig, ax, data_nli = pf.NLIcorrvFluctu(fig, ax, data = df, plottingParams = plottingParams, 
                                      plotChars = plotChars)


plt.xlim(0, 1.5)
plt.show()
plt.savefig((dirToSave + '(11b)_{:}_{:}_NLI-corrvFluctu.png').format(str(dates), str(condCat)))

#%%% Plotnine paired plots

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()
N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)

measure = 'E_eff'
stat = 'wAvg'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'two-sided', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks)


# plt.ylim(-2, 1)
# plt.yticks([-2, -1, 0, 1], [-2, -1, 0, 1], **plotTicks)

plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))

#%% Calling data - Using non-sorted cells with antibiotics

# Dates avilable : 24.12.24
filename = 'VWC_optoRhoAVB-NS_update_24-12-16'

GlobalTable = taka.getMergedTable(filename)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/OptoRhoA-VB/24.12.24'
dataFluoPath = 'D:/Anumita/MagneticPincherData/Raw/FluoData/24-12-14_FluoData.csv'
dataFluo = pd.read_csv(dataFluoPath, sep = ';')

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable, dataFluoPath)

dates = ['24-12-14'] 
drugs = ['doxy', 'doxy_act']

labels = ['Dox', 'Global Activation']

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1500),
            (data['E_eff'] <= 30000),
            (data['compNum'] <= 7),
            (data['drug'].apply(lambda x : x in drugs)),
            (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)

condCol, condCat = 'drug', drugs
avgDf = pf.createAvgDf(df, condCol, dataFluoPath)
# avgDf = avgDf[avgDf[('compNum', 'count')] > 3]

pairs = [[ 'doxy', 'doxy_act']]

plotChars = {'color' : '#ffffff', 'fontsize' : 20}
plotTicks = {'color' : '#ffffff', 'fontsize' : 18}

N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
# palette_cond = ['#4a4a4a', '#7f81fe', '#0004FD']
palette_cond = ['#7f81fe', '#0004FD']

swarmPointSize = 6


#%%%% Plot NLImod

########################################

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax = plt.subplots(figsize=(7, 6))

fig, ax = pf.rainplot(fig, ax, condCat, palette = palette_cond, 
                             labels = labels, pairs = pairs, shiftBox = 0.1, shiftSwarm = 0.0,
                             colorScheme = 'black', test = 'non-param' ,pointSize = 40,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

plt.ylim(-4,4.5)
plt.ylabel('NLR', **plotChars)
plt.xlabel(' ', **plotChars)
# plt.tight_layout()
plt.savefig((dirToSave + '(0a)_{:}_{:}_NLRrainplot.png').format(str(dates), str(condCat)))
plt.show()


################### box plots #######################
fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                                    hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(-3,3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLImodPLot.png').format(str(dates), str(condCat)))
plt.show()

######## vs. Compressions #########

fig, ax = plt.subplots(figsize = (13,9))

condition = 'doxy_act'
df_comp = df[df['drug'] == condition]

plottingParams = {'data':df_comp, 
                  'x' : 'compNum', 
                  'y' : 'NLI_mod',
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), 
                                    pairs = None, hueType = 'date', plotType = 'swarm',
                                    labels = [], plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1b)_{:}_{:}_NLImodPLot-Comps.png').format(str(dates), condition))


######## cell average #########

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1c)_{:}_{:}_NLImodPLot_cellAvg.png').format(str(dates), str(condCat)))
plt.show()


######## coloured Compressions #########

# fig, ax = plt.subplots(figsize = (13,9))

# plottingParams = {'data':df, 
#                   'x' : condCol, 
#                   'y' : 'NLI_mod',
#                   'order' : condCat,
#                   'linewidth' : 1,
#                   'size' :swarmPointSize, 
#                     }

# fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, plotType = 'swarm',
#                                     hueType = 'compNum', labels = [], plottingParams = plottingParams,
#                                     plotChars = plotChars)

# plt.ylim(-3, 3)
# fig.suptitle(str(dates), **plotChars)
# plt.yticks(**pltTicks)
# plt.xticks(**pltTicks)
# plt.tight_layout()
# plt.savefig((dirToSave + '(1e)_{:}_{:}_{:}_NLImodPLot-DiffComps.png').format(str(dates), condition, str(condCat)))

######## coloured dates #########

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'linewidth' : 1,
                  'size' :swarmPointSize, 
                    }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, plotType = 'swarm',
                                    hueType = 'date', labels = [], plottingParams = plottingParams,
                                    plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1f)_{:}_{:}_NLImodPLot-Dates.png').format(str(dates), str(condCat)))

#%%%% NLImod vs Compression

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : 'compNum', 
                  'y' : 'H0_vwc_Full',   
                  'linewidth' : 1, 
                  'hue' : condCol
                  }
                    
ax = sns.lineplot(**plottingParams)
# plt.ylim(-1, 2)

plt.show()

#%%%% Plot NLI

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.plotNLI(fig, ax, df, condCat, condCol,
                            pairs = pairs, labels = labels, **plotChars)

fig.suptitle(str(dates), **plotChars)
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(1b)_{:}_{:}_NLIPLot.png').format(str(dates), str(condCat)))

plt.style.use('default')


####### vs. Compressions #########
condition = 'doxy'
df_comp = df[df.drug == condition]

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.plotNLI(fig, ax, df_comp, condCat =  np.sort(df_comp.compNum.unique()), condCol = 'compNum',
                            pairs = None, labels = [], **plotChars)

fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(1b)_{:}_{:}_{:}_NLIPLot-Comps.png').format(str(dates), condition, str(condCat)))

plt.style.use('default')

#%%%% E vs H0
# fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
# fig, ax = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol, hueType = 'NLI_Plot')
# fig.suptitle(str(dates), **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(2a)_{:}_{:}_EvH_NLI.png').format(str(dates), str(condCat)))


fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))

# fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
# fig, ax = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
# plt.legend(fontsize = 6, ncol = len(condCat))
# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(2c)_{:}_{:}_EvH_CellID.png').format(str(dates), str(condCat)))

#%%%% E vs H0 log average
 
fig, axes = plt.subplots(figsize = (13,9))
fig, axes, avgDf = pf.EvH0_LogCellAvg(fig, axes,  avgDf, condCat, condCol, hueType = 'NLI_Plot')
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_logAvgEvH_NLI.png').format(str(dates), str(condCat)))


fig, axes = plt.subplots(figsize = (13,9))
fig, axes, avgDf = pf.EvH0_LogCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_logAvgEvH_Conditions.png').format(str(dates), str(condCat)))

# fig, axes = plt.subplots(1, 2, figsize = (18,10))
# fig, axes = pf.EvH0_LogCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
# plt.legend(fontsize = 6, ncol = len(condCat))
# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(2c)_{:}_{:}_wAvgEvH_CellID.png').format(str(dates), str(condCat)))

################## E_Normalised #####################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_norm', 'logAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :4, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond, labels = labels,
                             plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(2d)_{:}_{:}_E-NormBoxplot_logAvg.png').format(str(dates), str(condCat)))
plt.show()

 #%%%% E vs H0 weighted average
 
# fig, axes = plt.subplots(1, 2, figsize = (18,10))
# fig, axes, avgDf = pf.EvH0_wCellAvg(fig, axes,  avgDf, condCat, condCol, hueType = 'NLI_Plot')
# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(2a)_{:}_{:}_wAvgEvH_NLI.png').format(str(dates), str(condCat)))


fig, axes = plt.subplots(1, 2, figsize = (18,10))
fig, axes, avgDf = pf.EvH0_wCellAvg(fig, axes, avgDf, condCat, condCol, h_ref = 500,
                                    palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
# plt.savefig((dirToSave + '(2b)_{:}_{:}_wAvgEvH_Conditions.png').format(str(dates), str(condCat)))

# # fig, axes = plt.subplots(1, 2, figsize = (18,10))
# # fig, axes = pf.EvH0_wCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
# # plt.legend(fontsize = 6, ncol = len(condCat))
# # fig.suptitle(str(dates), **plotChars)
# # plt.tight_layout()
# # plt.show()
# # plt.savefig((dirToSave + '(2c)_{:}_{:}_wAvgEvH_CellID.png').format(str(dates), str(condCat)))

# ################## E_Normalised #####################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_norm', 'wAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond, labels = labels,
                             plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
# plt.yscale('log')
# y_labels = [100, 500, 5000, 10000, 50000]
# y_ticks = np.log10(np.asarray(y_labels))
# ax.set_yticks(y_ticks, labels =y_labels,**plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(2d)_{:}_{:}_E-NormBoxplot_wAvg.png').format(str(dates), str(condCat)))
plt.show()

#%%% Box plots - H0

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'bestH0_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 4, 
                   }

######################## Hue type 'CellID'#######################
# fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)


# fig.suptitle(str(dates), **plotChars)
# plt.legend(fontsize = 6, ncol = 6)
# # plt.ylim(0,2500)
# plt.yscale('log')
# y_ticks = [100, 250, 500, 1000, 2500]
# ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(3a)_{:}_{:}_H0Boxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))

ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)
plt.xticks(**plotTicks)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'bestH0_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'hue_order' : ['non-linear', 'intermediate', 'linear'],
                  'size' : swarmPointSize,  
                    }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(3c)_{:}_{:}_H0Boxplot_NLI.png').format(str(dates), str(condCat)))

####################### H0 vs compression ############################

fig, ax = plt.subplots(figsize = (13,9))
condition = 'doxy_act'
df_comp = df[df['drug'] == condition]

plottingParams = {'data':df_comp, 
                  'x' : 'compNum', 
                  'y' : 'bestH0_log',
                  'linewidth' : 1, 
                  'size' :6, 
                   }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), pairs = None, hueType = 'date',
                                    labels = [], plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
# plt.yscale('log')
plt.ylim(0,1500)
# y_ticks = [100, 250, 500, 1000, 2500]
# ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.savefig((dirToSave + '(3d)_{:}_{:}_H0Boxplot_CompNum.png').format(str(dates), str(condCat)))
plt.show()

####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('bestH0_log', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.savefig((dirToSave + '(3e)_{:}_{:}_H0Boxplot_meancellAverage.png').format(str(dates), str(condCat)))
plt.show()



#%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 4, 
                   }

ylim = 20000

######################## Hue type 'CellID'#######################
# N = len(df['cellID'].unique())
# palette_cell = distinctipy.get_colors(N)
# fig, ax = plt.subplots(figsize = (13,9))
# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)

# # plt.ylim(0, ylim)
# plt.legend(fontsize = 6, ncol = 6)
# fig.suptitle(str(dates), **plotChars)
# # plt.yscale('log')
# # plt.ylim(100, 30000)
# # y_ticks = [100, 500, 5000, 10000, 30000]
# ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff_log',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : 4,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

######################## Paired 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff_log',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : 4,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff_log', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_cellAverage.png').format(str(dates), str(condCat)))
plt.show()


####################### cell weighted average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'wAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1000, 50000)
y_ticks = [100, 500, 5000, 10000, 50000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_weightedCellAverage.png').format(str(dates), str(condCat)))
plt.show()


#%%% Boxplots - CtFieldThickness

df_ctField = df.drop_duplicates(subset = 'ctFieldThickness')

plottingParams = {'data':df_ctField, 
                  'x' : condCol, 
                  'y' : 'ctFieldThickness',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

ylim = 20000

######################## Hue type 'CellID'#######################
# N = len(df['cellID'].unique())
# palette_cell = distinctipy.get_colors(N)
# fig, ax = plt.subplots(figsize = (13,9))
# fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)

# # plt.ylim(0, ylim)
# plt.legend(fontsize = 6, ncol = 6)
# fig.suptitle(str(dates), **plotChars)
# plt.ylim(0,1500)
# plt.yticks(**pltTicks)
# plt.xticks(**pltTicks)
# plt.show()
# plt.savefig((dirToSave + '(4a)_{:}_{:}_ctFieldThickness_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**pltTicks)
plt.xticks(**pltTicks)
# plt.ylim(0,1500)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_ctFieldThickness_Conditions.png').format(str(dates), str(condCat)))


####################### cell average ############################

# fig, ax = plt.subplots(figsize = (13,9))
# dates = np.unique(df['date'].values)

# plottingParams = {'data':avgDf, 
#                   'x' : (condCol, 'first'), 
#                   'y' : ('E_eff', 'median'),
#                   'order' : condCat,
#                   'linewidth' : 1, 
#                   'size' :swarmPointSize, 
#                     }

# fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
#                              hueType = None, palette = palette_crosslink,
#                                     labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.yscale('log')
# plt.ylim(1000, 30000)
# y_ticks = [100, 500, 5000, 10000, 30000]
# ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

# plt.tight_layout()
# plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_cellAverage.png').format(str(dates), str(condCat)))
# plt.show()

#%%% Boxplots - NLI correlation

df_nli_corr = df.drop_duplicates(subset = 'cellID')


plottingParams = {'data':df_nli_corr, 
                  'x' : condCol, 
                  'y' : 'NLI_corr',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }



fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
# plt.ylim(0,1500)
plt.show()
plt.savefig((dirToSave + '(5a)_{:}_{:}_NLI_correlation_Conditions.png').format(str(dates), str(condCat)))



#%%% Pointplots 

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()

N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)

testH0 = 'two-sided'
testE = 'less'
testNli = 'greater'

stats = 'mean'
plottingParams = { 'x' : (condCol, 'first'), 
                  'y' : ('H0_vwc_Full', stats),
                  'linewidth' : 1, 
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

ylim = 1500
fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.xlim((-2,3))
plt.tight_layout()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_H0Pointplot.png').format(str(dates), str(condCat), stats))
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_H = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,3), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7b)_{:}_{:}_{:}_H0Pointplot-Normalised.png').format(str(dates), str(condCat), stats))
plt.show()


ylim = 10000
plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'wAvg'),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = 'wAvg',
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(8a)_{:}_{:}_{:}_EPointplot.png').format(str(dates), str(condCat), stats))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_E = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
                                          pairs = pairs, normalize = True, marker = 'wAvg',
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(8b)_{:}_{:}_{:}_EPointplot-Normalised.png').format(str(dates), str(condCat), stats))

nonOutliers_E = np.asarray(dfP_E['dateCell', 'first'][(dfP_E[condCol, 'first'] == condCat[1]) & (dfP_E['normMeasure', 'wAvg'] < 3.0)].values)

plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', stats),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-3,3), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# ax.get_legend().remove()
plt.show()
plt.savefig((dirToSave + '(9a)_{:}_{:}_{:}_NLImodPointplot.png').format(str(dates), str(condCat), stats))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-3,3), styleType = ('dateCell', 'first'),
                                          pairs = pairs, normalize = False, marker = stats, hueType = ('date', 'first'),
                                          test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(9b)_{:}_{:}_{:}_NLImodPointplot_dates.png').format(str(dates), str(condCat), stats))


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-1.5,1.5), 
                                          pairs = pairs, normalize = False, marker = stats, hueType = ('date', 'first'),
                                          test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(9c)_{:}_{:}_{:}_NLImodPointplot_datesAvg.png').format(str(dates), str(condCat), stats))



fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_nli = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-6,6), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(9d)_{:}_{:}_{:}_NLImodPointplot-Normalised.png').format(str(dates), str(condCat), stats))

#### E-normalised 

plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('E_norm', 'wAvg'),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim =(0,ylim), 
                                          pairs = pairs, normalize = False, marker = 'wAvg', 
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(10a)_{:}_{:}_E-norm.png').format(str(dates), str(condCat)))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
                                          pairs = pairs, normalize = True, marker = 'wAvg', 
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(10b)_{:}_{:}_E-doubleNorm.png').format(str(dates), str(condCat)))

#%%% NLI averaged paired plots vs fluctuations V.1

# dfPairs, pairedCells = pf.dfCellPairs(avgDf)

# fig, ax = plt.subplots(figsize = (13,9))

# N_point = len(dfPairs['dateCell', 'first'].unique())
# palette_cell_point = distinctipy.get_colors(N_point)

# plottingParams = {'x' : ('normFluctu', 'first'), 
#                   'y' : ('NLI_mod', 'mean'),
#                   'data' : dfPairs,
#                   'hue' : ('dateCell', 'first'),
#                   'palette' : palette_cell_point,
#                   'linewidth' : 1,
#                   'markersize' : 10,
#                   'markeredgecolor':'black', 
#                   'marker' : 'o',
#                    }

# ax = sns.lineplot(**plottingParams)

# plt.tight_layout()
# # plt.legend(loc=2, prop={'size': 7}, ncol =5)
# ax.get_legend().remove()
# plt.show()
# plt.savefig((dirToSave + '(11)_{:}_{:}_NLIParisvFluctu.png').format(str(dates), str(condCat)))

#%%% NLI averaged paired plots vs fluctuations V.2

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
fig, ax = plt.subplots(figsize = (13,9))
N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)

fig, ax = pf.NLIPairsvFluctu(fig, ax, dfPairs, condCol, condCat, palette = palette_cell_point, 
                plotChars = plotChars)


plt.xlim(0,2)
plt.show()
plt.savefig((dirToSave + '(11)_{:}_{:}_NLIParisvFluctu.png').format(str(dates), str(condCat)))


#%%% NLI vs Fluctuation

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'x' : ('normFluctu', 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'data' : avgDf,
                  'hue' : (condCol, 'first'),
                  'palette' : palette_cond,
                  'hue_order' : condCat,
                  's' : 100
                   }

fig, ax = pf.NLIvFluctu(fig, ax, plottingParams = plottingParams, plotChars = plotChars)

plt.xlim(0, 1.5)
plt.show()
plt.savefig((dirToSave + '(11a)_{:}_{:}_NLIvFluctu.png').format(str(dates), str(condCat)))


#%%%% E vs H0
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, df = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))

avgDf = avgDf = pf.createAvgDf(df, condCol, dataFluoPath = None, e_norm = True)

#%%%% Plotnine paired plots

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()
N_point = len(dfPairs['dateCell', 'first'].unique())


measure = 'NLI_mod'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'less', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2, 1))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'E_norm'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits =( 0, 10000))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))


measure = 'H0_vwc_Full'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits =( 0, 1500))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))

measure = 'E_eff'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits =( 0, 5000))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))




#%%% Plotnine paired plots with FLUORESENCE

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()
N_point = len(dfPairs['dateCell', 'first'].unique())

measure = 'NLI_mod'
stat = 'mean'
plot, pvals = pf.pairedplot_wfluo(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'less', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2, 1))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))

#%%% Expression intensity with NLI

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()

plottingParams = {'x' : ('mean_intensity', 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'data' : dfPairs,
                  
                  's' : 100
                   }

sns.scatterplot(**plottingParams)

plt.show()


#%% Calling data - Using non-sorted cells with antibiotics

# 'VWC_optoRhoAVB-NS_25-01-14_25-01-16'
filename = 'VWC_optoRhoAVB-NS_awayFromBeads_25-01-20' #25-01-14, 24-12-20

GlobalTable = taka.getMergedTable(filename)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/OptoRhoA-VB/25.01.14/'
dataActPath = 'D:/Anumita/MagneticPincherData/Raw/ActivationData/25-01-20_ActivationData.csv'
dataAct = pd.read_csv(dataActPath, sep = ';')

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable, dataActPath = dataActPath)

dates = ['25-01-14', '24-12-20'] 
# drugs = ['doxy', 'doxy_act']

# manips = ['M5', 'M1', 'M2', 'M3', 'M4']
# labels = ['Control', 'Dox', 'Dox + Away', 'Dox + At', 'Dox + Global']
# palette_cond = ['#000000', '#8d9095', '#8ec582', '#c66440', '#325eb6']


# manips = ['M1', 'M2']
activation = ['none', 'side', 'away from beads', 'at beads', 'global']
# activation = ['none', 'global']

labels = []
palette_cond = ['#8d9095', '#c66440', '#325eb6', '#8ec582', '#8d9095']


# manips = ['M1', 'M3']
# labels = ['Dox', 'Dox + At Beads']
# palette_cond = ['#8d9095', '#c66440']


# manips = ['M1', 'M2']
# labels = ['Dox', 'Dox + \nAway from Beads']
# palette_cond = ['#8d9095', '#8ec582']


Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1200) & (data['bestH0'] > 100),
            (data['E_eff'] <= 30000),
            # (data['ctFieldThickness'] <= 1200),
            # (data['compNum'] <= 7),
            (data['activation type'].apply(lambda x : x in activation)),
            # (data['manip'].apply(lambda x : x in manips)),
            (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)

condCol, condCat = 'activation type', activation
avgDf = pf.createAvgDf(df, condCol)
# avgDf = avgDf[avgDf[('compNum', 'count')] > 3]

pairs = [['none', 'side'], ['none', 'away from beads'], ['side', 'away from beads']]

# pairs = [activation] #, ['none', 'away from beads'], ['side', 'away from beads']]

# pairs = [['M5', 'M1'], ['M1', 'M2'], ['M1', 'M3'], ['M1', 'M4']]

plotChars = {'color' : '#ffffff', 'fontsize' : 20}
plotTicks = {'color' : '#ffffff', 'fontsize' : 16}

N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)

swarmPointSize = 6

#%%%% Plot NLImod

########################################

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax = plt.subplots(figsize=(10, 10))

fig, ax = pf.rainplot(fig, ax, condCat, palette = palette_cond, 
                             labels = labels, pairs = pairs, shiftBox = 0.1, shiftSwarm = 0.0,
                             colorScheme = 'black', test = 'non-param' ,pointSize = 40,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

# plt.ylim(-4,4.5)
plt.ylabel('NLR', **plotChars)
plt.xlabel(' ', **plotChars)
# plt.tight_layout()
plt.savefig((dirToSave + '(0a)_{:}_{:}_NLRrainplot.png').format(str(dates), str(condCat)))
plt.show()


################### box plots #######################
fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = None, 
                                    hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(-3,3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLImodPLot.png').format(str(dates), str(condCat)))
plt.show()

######## vs. Compressions #########

fig, ax = plt.subplots(figsize = (13,9))

condition = 'doxy_act'
df_comp = df[df['drug'] == condition]

plottingParams = {'data':df_comp, 
                  'x' : 'compNum', 
                  'y' : 'NLI_mod',
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), 
                                    pairs = None, hueType = 'date', plotType = 'swarm',
                                    labels = [], plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1b)_{:}_{:}_NLImodPLot-Comps.png').format(str(dates), condition))


######## cell average #########

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1c)_{:}_{:}_NLImodPLot_cellAvg.png').format(str(dates), str(condCat)))
plt.show()

######## coloured Compressions #########

# fig, ax = plt.subplots(figsize = (13,9))

# plottingParams = {'data':df, 
#                   'x' : condCol, 
#                   'y' : 'NLI_mod',
#                   'order' : condCat,
#                   'linewidth' : 1,
#                   'size' :swarmPointSize, 
#                     }

# fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, plotType = 'swarm',
#                                     hueType = 'compNum', labels = [], plottingParams = plottingParams,
#                                     plotChars = plotChars)

# plt.ylim(-3, 3)
# fig.suptitle(str(dates), **plotChars)
# plt.yticks(**pltTicks)
# plt.xticks(**pltTicks)
# plt.tight_layout()
# plt.savefig((dirToSave + '(1e)_{:}_{:}_{:}_NLImodPLot-DiffComps.png').format(str(dates), condition, str(condCat)))


######## coloured dates #########

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'linewidth' : 1,
                  'size' :swarmPointSize, 
                    }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, plotType = 'swarm',
                                    hueType = 'date', labels = [], plottingParams = plottingParams,
                                    plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1f)_{:}_{:}_NLImodPLot-Dates.png').format(str(dates), str(condCat)))


#%%%% Box plots - H0

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'bestH0_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 4, 
                   }

######################## Hue type 'CellID'#######################
# fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)


# fig.suptitle(str(dates), **plotChars)
# plt.legend(fontsize = 6, ncol = 6)
# # plt.ylim(0,2500)
# plt.yscale('log')
# y_ticks = [100, 250, 500, 1000, 2500]
# ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(3a)_{:}_{:}_H0Boxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))

ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)
plt.xticks(**plotTicks)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))

#%%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 4, 
                   }

ylim = 20000

######################## Hue type 'CellID'#######################
# N = len(df['cellID'].unique())
# palette_cell = distinctipy.get_colors(N)
# fig, ax = plt.subplots(figsize = (13,9))
# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)

# # plt.ylim(0, ylim)
# plt.legend(fontsize = 6, ncol = 6)
# fig.suptitle(str(dates), **plotChars)
# # plt.yscale('log')
# # plt.ylim(100, 30000)
# # y_ticks = [100, 500, 5000, 10000, 30000]
# ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))

#%%%% E vs H0
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, df = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))


avgDf = avgDf = pf.createAvgDf(df, condCol, dataFluoPath = None, e_norm = True)

#%%%% Plotnine paired plots

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()
N_point = len(dfPairs['dateCell', 'first'].unique())


measure = 'NLI_mod'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'less', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (-2.5, 1.5))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'E_norm'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits =( 0, 10000))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'H0_vwc_Full'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 1500))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))


measure = 'ctFieldThickness'
stat = 'first'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 1000))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'ctFieldFluctuAmpli'
stat = 'first'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'NLI_mod'
stat = 'std'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 2))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))


#%%%% Point plots

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()

N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)
testH0 = 'two-sided'
testE = 'less'
testNli = 'two-sided'

stats = 'mean'

plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', stats),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-3,3), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# ax.get_legend().remove()
plt.show()


plt.savefig((dirToSave + '(9a)_{:}_{:}_{:}_NLImodPointplot.png').format(str(dates), str(condCat), stats))


plottingParams = { 'x' : (condCol, 'first'), 
                  'y' : ('H0_vwc_Full', stats),
                  'linewidth' : 1, 
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

ylim = 1500
fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.xlim((-2,3))
plt.tight_layout()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_H0Pointplot.png').format(str(dates), str(condCat), stats))
plt.show()


# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP_nli = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-6,6), 
#                                           pairs = pairs, normalize = True, marker = stats,
#                                           test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(9b)_{:}_{:}_{:}_NLImodPointplot-Normalised.png').format(str(dates), str(condCat), stats))



#%%%% Thickness vs. time


measure = 'surroundingThickness'
activationTime = []

fig1, axes = plt.subplots(1,1, figsize=(15,10))
fig1.patch.set_facecolor('black')
flatui = ["#000000", "#0000ff"]

x = (df['compNum']-1)*20
ax = sns.lineplot(x = x, y = measure, data = df, hue = condCol)


plt.ylim(0,1200)

# plt.savefig(dirToSave + '/Thickness/'+str(dates)+'_'+measure+'vsCompr'+str(manips)+'.png')

plt.show()

#%% Calling data - Using non-sorted cells with antibiotics

# filename = 'VWC_optoRhoAVB-NS_updated_25-01-29' #25-01-14, 25-01-14, 24-12-20, 25-01-21, 25-01-23

filename = 'VWC_optoRhoAVB-NS_ForBoxplots'  

GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/OptoRhoA-VB/25-01-28_allExpts_Boxplots/'
dataActPath = 'D:/Anumita/MagneticPincherData/Raw/ActivationData/25-01-20_ActivationData.csv'
dataAct = pd.read_csv(dataActPath, sep = ';')

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable, dataActPath = dataActPath)

dates = ['24-12-20', '24-12-14', '25-01-14', '25-01-21', '25-01-23']
# dates = ['24-12-20', '25-01-21', '25-01-23'] 

drugs = ['doxy', 'doxy_act']

# manips = ['M5', 'M1', 'M2', 'M3', 'M4']
activation = ['none',   'away from beads', 'at beads', 'global']
labels = [ 'Dox',  'Dox + Away', 'Dox + At', 'Dox + Global']
palette_cond = ['#8d9095', '#c66440', '#8ec582', '#325eb6']
# palette_cond = ['#8d9095', '#ffb230',  '#c66440', '#8ec582', '#325eb6']

# activation = ['none', 'at beads'] #, 'at beads', 'global']
# labels = [ 'Dox',  'At beads'] #, 'Dox + At', 'Dox + Global']
# palette_cond = ['#8d9095', '#ffdf00']

# activation = ['none', 'away from beads'] #, 'at beads', 'global']
# labels = [ 'Dox',  'Away from Beads'] #, 'Dox + At', 'Dox + Global']
# palette_cond = ['#8d9095', '#ff6f61']

# activation = ['none', 'global'] #, 'at beads', 'global']
# labels = [ 'Dox',  'Global Activation'] #, 'Dox + At', 'Dox + Global']
# palette_cond = ['#8d9095', '#0000FF']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_eff'] <= 30000),
            # (data['ctFieldThickness'] <= 1200),
            # (data['compNum'] <= 6),
            (data['activation type'].apply(lambda x : x in activation)),
            (data['manip'].apply(lambda x : x in manips)),
            (data['drug'].apply(lambda x : x in drugs)),
            (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)

condCol, condCat = 'activation type', activation
avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[avgDf[('compNum', 'count')] > 2]

pairs = [['none', 'at beads'], ['none', 'away from beads'], ['none', 'global'],
         ['away from beads' , 'global']]

# pairs = [activation]

plotChars = {'color' : '#ffffff', 'fontsize' : 20}
plotTicks = {'color' : '#ffffff', 'fontsize' : 16}

N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)

swarmPointSize = 6

#%%%% Plot NLImod

########################################

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax = plt.subplots(figsize=(10, 10))

fig, ax = pf.rainplot(fig, ax, condCat, palette = palette_cond, 
                             labels = labels, pairs = pairs, shiftBox = 0.1, shiftSwarm = 0.0,
                             colorScheme = 'black', test = 'non-param' ,pointSize = 40,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

# plt.ylim(-4,4.5)
plt.ylabel('NLR', **plotChars)
plt.xlabel(' ', **plotChars)
# plt.tight_layout()
plt.savefig((dirToSave + '(0a)_{:}_{:}_NLRrainplot.png').format(str(dates), str(condCat)))
plt.show()


################### box plots #######################
fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = None, 
                                    hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(-3,3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLImodPLot.png').format(str(dates), str(condCat)))
plt.show()

######## vs. Compressions #########

fig, ax = plt.subplots(figsize = (13,9))

condition = 'doxy_act'
df_comp = df[df['drug'] == condition]

plottingParams = {'data':df_comp, 
                  'x' : 'compNum', 
                  'y' : 'NLI_mod',
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), 
                                    pairs = None, hueType = 'date', plotType = 'swarm',
                                    labels = [], plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1b)_{:}_{:}_NLImodPLot-Comps.png').format(str(dates), condition))


######## cell average #########

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1c)_{:}_{:}_NLImodPLot_cellAvg.png').format(str(dates), str(condCat)))
plt.show()

######## coloured Compressions #########

# fig, ax = plt.subplots(figsize = (13,9))

# plottingParams = {'data':df, 
#                   'x' : condCol, 
#                   'y' : 'NLI_mod',
#                   'order' : condCat,
#                   'linewidth' : 1,
#                   'size' :swarmPointSize, 
#                     }

# fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, plotType = 'swarm',
#                                     hueType = 'compNum', labels = [], plottingParams = plottingParams,
#                                     plotChars = plotChars)

# plt.ylim(-3, 3)
# fig.suptitle(str(dates), **plotChars)
# plt.yticks(**pltTicks)
# plt.xticks(**pltTicks)
# plt.tight_layout()
# plt.savefig((dirToSave + '(1e)_{:}_{:}_{:}_NLImodPLot-DiffComps.png').format(str(dates), condition, str(condCat)))


######## coloured dates #########

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'linewidth' : 1,
                  'size' :swarmPointSize, 
                    }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, plotType = 'swarm',
                                    hueType = 'date', labels = [], plottingParams = plottingParams,
                                    plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1f)_{:}_{:}_NLImodPLot-Dates.png').format(str(dates), str(condCat)))


#%%%% Box plots - H0

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'bestH0_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 4, 
                   }

######################## Hue type 'CellID'#######################
# fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)


# fig.suptitle(str(dates), **plotChars)
# plt.legend(fontsize = 6, ncol = 6)
# # plt.ylim(0,2500)
# plt.yscale('log')
# y_ticks = [100, 250, 500, 1000, 2500]
# ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(3a)_{:}_{:}_H0Boxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))

ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)
plt.xticks(**plotTicks)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))

####################### cell average #######################


fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('bestH0_log', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.savefig((dirToSave + '(3e)_{:}_{:}_H0Boxplot_meancellAverage.png').format(str(dates), str(condCat)))
plt.show()


#%%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 4, 
                   }

ylim = 20000

######################## Hue type 'CellID'#######################
# N = len(df['cellID'].unique())
# palette_cell = distinctipy.get_colors(N)
# fig, ax = plt.subplots(figsize = (13,9))
# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)

# # plt.ylim(0, ylim)
# plt.legend(fontsize = 6, ncol = 6)
# fig.suptitle(str(dates), **plotChars)
# # plt.yscale('log')
# # plt.ylim(100, 30000)
# # y_ticks = [100, 500, 5000, 10000, 30000]
# ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))


####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'wAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                             labels = labels, plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_cellAverage.png').format(str(dates), str(condCat)))
plt.show()

#%%%% E vs H0
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, df = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))


avgDf = avgDf = pf.createAvgDf(df, condCol, dataFluoPath = None, e_norm = True)

#%%%% Plotnine paired plots

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()
N_point = len(dfPairs['dateCell', 'first'].unique())


measure = 'NLI_mod'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'less', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'E_norm'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits =( 0, 10000))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'H0_vwc_Full'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 1500))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))


measure = 'ctFieldThickness'
stat = 'first'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 1000))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'ctFieldFluctuAmpli'
stat = 'first'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'NLI_mod'
stat = 'std'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 2))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))


#%%%% Point-line plots

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()
N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)


testH0 = 'two-sided'
testE = 'less'
testNli = 'two-sided'
stats = 'mean'

plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', stats),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-3,3), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# ax.get_legend().remove()
plt.show()
plt.savefig((dirToSave + '(9a)_{:}_{:}_{:}_NLImodPointplot.png').format(str(dates), str(condCat), stats))


plottingParams = { 'x' : (condCol, 'first'), 
                  'y' : ('H0_vwc_Full', stats),
                  'linewidth' : 1, 
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

ylim = 1200
fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.xlim((-2,3))
plt.tight_layout()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_H0Pointplot.png').format(str(dates), str(condCat), stats))
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_H = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,3), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7b)_{:}_{:}_{:}_H0Pointplot-Normalised.png').format(str(dates), str(condCat), stats))
plt.show()



plottingParams = { 'x' : (condCol, 'first'), 
                  'y' : ('ctFieldThickness', 'first'),
                  'linewidth' : 1, 
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

ylim = 1500
fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,1200), 
                                          pairs = pairs, normalize = False, marker = 'first',
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.xlim((-2,3))
plt.tight_layout()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_ctFieldThickness.png').format(str(dates), str(condCat), stats))
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_H = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,3), 
                                          pairs = pairs, normalize = True, marker = 'first',
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7b)_{:}_{:}_{:}__ctFieldThickness-Normalised.png').format(str(dates), str(condCat), stats))
plt.show()


plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'wAvg'),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,5000), 
                                          pairs = pairs, normalize = False, marker = 'wAvg',
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
y_ticks = [100, 250, 500, 1000, 2500, 5000]
ax.set_yticks(y_ticks, labels =y_ticks, fontsize=25, color = fontColour)
plt.savefig((dirToSave + '(8a)_{:}_{:}_{:}_EPointplot.png').format(str(dates), str(condCat), stats))



fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_E = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
                                          pairs = pairs, normalize = True, marker = 'wAvg',
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(8b)_{:}_{:}_{:}_EPointplot-Normalised.png').format(str(dates), str(condCat), stats))

plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('E_norm', 'mean'),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,5000), 
                                          pairs = pairs, normalize = False, marker = 'wAvg',
                                          test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
y_ticks = [100, 250, 500, 1000, 2500, 5000]
ax.set_yticks(y_ticks, labels =y_ticks, fontsize=25, color = fontColour)
plt.savefig((dirToSave + '(8a)_{:}_{:}_{:}_E-normPointplot.png').format(str(dates), str(condCat), stats))


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_E = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
                                          pairs = pairs, normalize = True, marker = 'wAvg',
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(8b)_{:}_{:}_{:}_E-normPointplot-Normalised.png').format(str(dates), str(condCat), stats))





# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP_nli = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-6,6), 
#                                           pairs = pairs, normalize = True, marker = stats,
#                                           test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)


# plt.show()
# plt.savefig((dirToSave + '(9b)_{:}_{:}_{:}_NLImodPointplot-Normalised.png').format(str(dates), str(condCat), stats))

#%%%% Pointplots with errorbars

dfPairs = df[df['dateCell'].apply(lambda x : x in pairedCells)]

plottingParams = {'x' : condCol, 
                  'y' : 'NLI_mod',
                  'data' : dfPairs,
                  'hue' : 'dateCell',
                  'dodge' : True,
                  'palette' : palette_cell_point,
                  'errorbar' : 'se'
                   }

fig, ax = plt.subplots(figsize = (13,10))
fig.patch.set_facecolor('black')

ax = sns.pointplot(**plottingParams)

plt.xticks(**plotChars)
plt.yticks(**plotChars)
plt.legend(fontsize = 15, labelcolor='linecolor', bbox_to_anchor=(1.05, 1))
plt.show()
plt.savefig((dirToSave + '(10a)_{:}_{:}_{:}_NLIPoint_Errorbar.png').format(str(dates), str(condCat), stats))

#%%%% E vs H0

dfPairs = df[df['dateCell'].apply(lambda x : x in pairedCells)]

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, df = pf.EvsH0_perCompression(fig, ax, dfPairs, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH-dfPairs_Conditions.png').format(str(dates), str(condCat)))


avgDf = avgDf = pf.createAvgDf(df, condCol, dataFluoPath = None, e_norm = True)

#%%%% Thickness vs. time


measure = 'surroundingThickness'
activationTime = []

fig1, axes = plt.subplots(1,1, figsize=(15,10))
fig1.patch.set_facecolor('black')
flatui = ["#000000", "#0000ff"]

x = (df['compNum']-1)*20
ax = sns.lineplot(x = x, y = measure, data = df, hue = condCol)


plt.ylim(0,1200)

# plt.savefig(dirToSave + '/Thickness/'+str(dates)+'_'+measure+'vsCompr'+str(manips)+'.png')

plt.show()

#%% Calling data - Using non-sorted cells with antibiotics

filename = 'VWC_optoRhoAVB-NS_updated_25-03-04' #25-01-14, 25-01-14, 24-12-20, 25-01-21, 25-01-23

GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/OptoRhoA-VB/25-03-04_allExpts/'
dataActPath = 'D:/Anumita/MagneticPincherData/Raw/ActivationData/25-01-20_ActivationData.csv'
dataAct = pd.read_csv(dataActPath, sep = ';')
# 
#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable, dataActPath = dataActPath)

# dates = ['24-12-20', '24-12-14', '25-01-14', '25-01-21', '25-01-23']
# dates = ['24-12-20', '25-01-21', '25-01-23'] 

# drugs = ['doxy', 'doxy_act']

# manips = ['M5', 'M1', 'M2', 'M3', 'M4']
# activation = ['none',   'side', 'away from beads', 'at beads', 'global']
# labels = [ 'Dox', 'Dox + Side', 'Dox + Away', 'Dox + At', 'Dox + Global']
# palette_cond = ['#8d9095', '#325eb6', '#c66440', '#8ec582', '#325eb6']
# palette_cond = ['#8d9095', '#ffb230',  '#c66440', '#8ec582', '#325eb6']

activation = ['none',   'away from beads'] #, 'away from beads', 'at beads', 'global']
labels = [ 'Dox', 'Dox + Away'] #, 'Dox + Away', 'Dox + At', 'Dox + Global']
palette_cond = ['#8d9095', '#325eb6'] #, '#c66440', '#8ec582', '#325eb6']

# activation = ['none', 'at beads'] #, 'at beads', 'global']
# labels = [ 'Dox',  'At beads'] #, 'Dox + At', 'Dox + Global']
# palette_cond = ['#8d9095', '#ffdf00']

# activation = ['none', 'away from beads'] #, 'at beads', 'global']
# labels = [ 'Dox',  'Away from Beads'] #, 'Dox + At', 'Dox + Global']
# palette_cond = ['#8d9095', '#ff6f61']

# activation = ['none', 'global'] #, 'at beads', 'global']
# labels = [ 'Dox',  'Global Activation'] #, 'Dox + At', 'Dox + Global']
# palette_cond = ['#8d9095', '#0000FF']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_eff'] <= 30000),
            # (data['ctFieldThickness'] <= 1200),
            (data['compNum'] <= 6),
            (data['activation type'].apply(lambda x : x in activation)),
            # (data['manip'].apply(lambda x : x in manips)),
            # (data['drug'].apply(lambda x : x in drugs)),
            # (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)

condCol, condCat = 'activation type', activation
avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[avgDf[('compNum', 'count')] > 2]

# pairs = [['none', 'at beads'], ['none', 'away from beads'], ['none', 'global'],
#          ['away from beads' , 'global']]

pairs = [activation]

plotChars = {'color' : '#ffffff', 'fontsize' : 20}
plotTicks = {'color' : '#ffffff', 'fontsize' : 16}

N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)

swarmPointSize = 6

#%%%% Plot NLImod

########################################

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax = plt.subplots(figsize=(10, 10))

fig, ax = pf.rainplot(fig, ax, condCat, palette = palette_cond, 
                             labels = labels, pairs = pairs, shiftBox = 0.1, shiftSwarm = 0.0,
                             colorScheme = 'black', test = 'non-param' ,pointSize = 40,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

# plt.ylim(-4,4.5)
plt.ylabel('NLR', **plotChars)
plt.xlabel(' ', **plotChars)
# plt.tight_layout()
plt.savefig((dirToSave + '(0a)_{:}_{:}_NLRrainplot.png').format(str(dates), str(condCat)))
plt.show()


################### box plots #######################
fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = None, 
                                    hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(-3,3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLImodPLot.png').format(str(dates), str(condCat)))
plt.show()

######## vs. Compressions #########

fig, ax = plt.subplots(figsize = (13,9))

condition = 'doxy_act'
df_comp = df[df['drug'] == condition]

plottingParams = {'data':df_comp, 
                  'x' : 'compNum', 
                  'y' : 'NLI_mod',
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), 
                                    pairs = None, hueType = 'date', plotType = 'swarm',
                                    labels = [], plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1b)_{:}_{:}_NLImodPLot-Comps.png').format(str(dates), condition))


######## cell average #########

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1c)_{:}_{:}_NLImodPLot_cellAvg.png').format(str(dates), str(condCat)))
plt.show()

######## coloured Compressions #########

# fig, ax = plt.subplots(figsize = (13,9))

# plottingParams = {'data':df, 
#                   'x' : condCol, 
#                   'y' : 'NLI_mod',
#                   'order' : condCat,
#                   'linewidth' : 1,
#                   'size' :swarmPointSize, 
#                     }

# fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, plotType = 'swarm',
#                                     hueType = 'compNum', labels = [], plottingParams = plottingParams,
#                                     plotChars = plotChars)

# plt.ylim(-3, 3)
# fig.suptitle(str(dates), **plotChars)
# plt.yticks(**pltTicks)
# plt.xticks(**pltTicks)
# plt.tight_layout()
# plt.savefig((dirToSave + '(1e)_{:}_{:}_{:}_NLImodPLot-DiffComps.png').format(str(dates), condition, str(condCat)))


######## coloured dates #########

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'linewidth' : 1,
                  'size' :swarmPointSize, 
                    }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, plotType = 'swarm',
                                    hueType = 'date', labels = [], plottingParams = plottingParams,
                                    plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.tight_layout()
plt.savefig((dirToSave + '(1f)_{:}_{:}_NLImodPLot-Dates.png').format(str(dates), str(condCat)))


#%%%% Box plots - H0

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'bestH0_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 4, 
                   }

######################## Hue type 'CellID'#######################
# fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)


# fig.suptitle(str(dates), **plotChars)
# plt.legend(fontsize = 6, ncol = 6)
# # plt.ylim(0,2500)
# plt.yscale('log')
# y_ticks = [100, 250, 500, 1000, 2500]
# ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(3a)_{:}_{:}_H0Boxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))

ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)
plt.xticks(**plotTicks)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))

####################### cell average #######################


fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('bestH0_log', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.savefig((dirToSave + '(3e)_{:}_{:}_H0Boxplot_meancellAverage.png').format(str(dates), str(condCat)))
plt.show()


#%%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 4, 
                   }

ylim = 20000

######################## Hue type 'CellID'#######################
# N = len(df['cellID'].unique())
# palette_cell = distinctipy.get_colors(N)
# fig, ax = plt.subplots(figsize = (13,9))
# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
#                                     hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)

# # plt.ylim(0, ylim)
# plt.legend(fontsize = 6, ncol = 6)
# fig.suptitle(str(dates), **plotChars)
# # plt.yscale('log')
# # plt.ylim(100, 30000)
# # y_ticks = [100, 500, 5000, 10000, 30000]
# ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
# plt.show()
# plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))


####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'wAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                             labels = labels, plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_cellAverage.png').format(str(dates), str(condCat)))
plt.show()

#%%%% E vs H0
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, df = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))


avgDf = avgDf = pf.createAvgDf(df, condCol, dataFluoPath = None, e_norm = True)

#%%%% Plotnine paired plots

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()
N_point = len(dfPairs['dateCell', 'first'].unique())


measure = 'NLI_mod'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'less', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'E_norm'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits =( 0, 10000))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'H0_vwc_Full'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 1500))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))


measure = 'ctFieldThickness'
stat = 'first'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 1000))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'ctFieldFluctuAmpli'
stat = 'first'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'NLI_mod'
stat = 'std'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 2))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))


#%%%% Point-line plots

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()
N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)


testH0 = 'two-sided'
testE = 'less'
testNli = 'two-sided'
stats = 'mean'

plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', stats),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-3,3), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# ax.get_legend().remove()
plt.show()
plt.savefig((dirToSave + '(9a)_{:}_{:}_{:}_NLImodPointplot.png').format(str(dates), str(condCat), stats))


plottingParams = { 'x' : (condCol, 'first'), 
                  'y' : ('H0_vwc_Full', stats),
                  'linewidth' : 1, 
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

ylim = 1200
fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.xlim((-2,3))
plt.tight_layout()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_H0Pointplot.png').format(str(dates), str(condCat), stats))
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_H = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,3), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7b)_{:}_{:}_{:}_H0Pointplot-Normalised.png').format(str(dates), str(condCat), stats))
plt.show()



plottingParams = { 'x' : (condCol, 'first'), 
                  'y' : ('ctFieldThickness', 'first'),
                  'linewidth' : 1, 
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

ylim = 1500
fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,1200), 
                                          pairs = pairs, normalize = False, marker = 'first',
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.xlim((-2,3))
plt.tight_layout()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_ctFieldThickness.png').format(str(dates), str(condCat), stats))
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_H = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,3), 
                                          pairs = pairs, normalize = True, marker = 'first',
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7b)_{:}_{:}_{:}__ctFieldThickness-Normalised.png').format(str(dates), str(condCat), stats))
plt.show()


plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'wAvg'),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,5000), 
                                          pairs = pairs, normalize = False, marker = 'wAvg',
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
y_ticks = [100, 250, 500, 1000, 2500, 5000]
ax.set_yticks(y_ticks, labels =y_ticks, fontsize=25, color = fontColour)
plt.savefig((dirToSave + '(8a)_{:}_{:}_{:}_EPointplot.png').format(str(dates), str(condCat), stats))



fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_E = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
                                          pairs = pairs, normalize = True, marker = 'wAvg',
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(8b)_{:}_{:}_{:}_EPointplot-Normalised.png').format(str(dates), str(condCat), stats))

plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('E_norm', 'mean'),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,5000), 
                                          pairs = pairs, normalize = False, marker = 'wAvg',
                                          test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
y_ticks = [100, 250, 500, 1000, 2500, 5000]
ax.set_yticks(y_ticks, labels =y_ticks, fontsize=25, color = fontColour)
plt.savefig((dirToSave + '(8a)_{:}_{:}_{:}_E-normPointplot.png').format(str(dates), str(condCat), stats))


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_E = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
                                          pairs = pairs, normalize = True, marker = 'wAvg',
                                          test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(8b)_{:}_{:}_{:}_E-normPointplot-Normalised.png').format(str(dates), str(condCat), stats))





# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP_nli = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-6,6), 
#                                           pairs = pairs, normalize = True, marker = stats,
#                                           test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)


# plt.show()
# plt.savefig((dirToSave + '(9b)_{:}_{:}_{:}_NLImodPointplot-Normalised.png').format(str(dates), str(condCat), stats))

#%%%% Pointplots with errorbars

dfPairs = df[df['dateCell'].apply(lambda x : x in pairedCells)]

plottingParams = {'x' : condCol, 
                  'y' : 'NLI_mod',
                  'data' : dfPairs,
                  'hue' : 'dateCell',
                  'dodge' : True,
                  'palette' : palette_cell_point,
                  'errorbar' : 'se'
                   }

fig, ax = plt.subplots(figsize = (13,10))
fig.patch.set_facecolor('black')

ax = sns.pointplot(**plottingParams)

plt.xticks(**plotChars)
plt.yticks(**plotChars)
plt.legend(fontsize = 15, labelcolor='linecolor', bbox_to_anchor=(1.05, 1))
plt.show()
plt.savefig((dirToSave + '(10a)_{:}_{:}_{:}_NLIPoint_Errorbar.png').format(str(dates), str(condCat), stats))

#%%%% E vs H0

dfPairs = df[df['dateCell'].apply(lambda x : x in pairedCells)]

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, df = pf.EvsH0_perCompression(fig, ax, dfPairs, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH-dfPairs_Conditions.png').format(str(dates), str(condCat)))


avgDf = avgDf = pf.createAvgDf(df, condCol, dataFluoPath = None, e_norm = True)

#%%%% Thickness vs. time


measure = 'surroundingThickness'
activationTime = []

fig1, axes = plt.subplots(1,1, figsize=(15,10))
fig1.patch.set_facecolor('black')
flatui = ["#000000", "#0000ff"]

x = (df['compNum']-1)*20
ax = sns.lineplot(x = x, y = measure, data = df, hue = condCol)


plt.ylim(0,1200)

# plt.savefig(dirToSave + '/Thickness/'+str(dates)+'_'+measure+'vsCompr'+str(manips)+'.png')

plt.show()
