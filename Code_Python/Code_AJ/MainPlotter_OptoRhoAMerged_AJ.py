# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:19:45 2024

@author: anumi
"""

# %% > Imports and constants



#### Main imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm

# import ptitprince as pt
# from plotnine import (
#     ggplot,
#     aes,
#     stage,
#     geom_violin,
#     geom_point,
#     geom_line,
#     geom_boxplot,
#     guides,
#     scale_fill_manual,
#     theme,
#     theme_classic,
#     facet_wrap,
#     xlim, ylim,
#     ggtitle
# )
# from plotnine.themes.elements import element_rect, element_text


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings(
    "ignore",
    message=".*is_categorical_dtype is deprecated and will be removed.*"
)


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
# 
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
                'methods_H0':['Chadwick'],
                'zones_H0':['%f_15'],
                'method_bestH0':'Chadwick', 
                'zone_bestH0':'%f_15',
                'doVWCFit' : True,
                'VWCFitMethods' : ['Full'],
                'doChadwickFit' : True,
                'ChadwickFitMethods' : ['Full'],
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
                        'Plot_Ratio':False,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        }
    

Task = '22-12-07_M4 & 23-02-02_M1 & 23-04-19_M1 & 23-04-25_M1 & 23-05-10_M1 & 23-05-10_M3 & 23-05-23_M1 & 23-07-07_M2 & 23-07-12_M1'

fitsSubDir = 'VWC_allOptoRhoA_NoActivation'

GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', 
                            fileName = fitsSubDir, save = True, PLOT = False, source = 'Python',
                            fitSettings = fitSettings, plotSettings = plotSettings,
                            fitsSubDir = fitsSubDir) # task = 'updateExisting'

#%% Calling data

#Dates available : '23-04-25 & 23-05-10_M3 & 23-05-10_M4 & 23-05-10_M5 & 22-12-07'
filename = 'VWC_60sGlobalActivation'

GlobalTable = taka.getMergedTable(filename)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/24-11-13_PielTeamMeeting/'

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)

dates = ['23-04-25' , '23-05-10'] #, '22-12-07']

# drugs = [ 'none', 'Y27_10', 'Y27_50']
# labels = ['Control', '10µM Y27', '50µM Y27']

drugs = [ 'none', 'activation']
labels = ['Control', 'Global Activation']

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            # (data['ctFieldThickness'] < 1000), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1200),
            (data['E_eff'] <= 30000),
            # (data['compNum'] <= 6),
            (data['date'].apply(lambda x : x in dates)),
            (data['drug'].apply(lambda x : x in drugs)),
            ]

df = pf.filterDf(Filters, data)
condCol, condCat = 'drug', drugs
avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[avgDf[('compNum', 'count')] > 3]

df = df.drop(df[(df['drug'] == 'activation') & (df['compNum'] == 1)].index)
plotChars = {'color' : '#ffffff', 'fontsize' : 20}
plotTicks = {'color' : '#ffffff', 'fontsize' : 18}

pairs = [['none', 'activation']]
# pairs = [[ 'none',  'Y27_10'], [ 'none',  'Y27_50']]

N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
# palette_cond_y27 = ['#929292', '#bf7fbf', '#800080'] 
palette_cond_act = ['#4a4a4a', '#0000ff'] 

swarmPointSize = 10

#%%%% Plot NLImod

########################################

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax = plt.subplots(figsize=(7, 6))

fig, ax = pf.rainplot(fig, ax, condCat, palette = palette_cond_act, 
                             labels = labels, pairs = pairs, shiftBox = 0.1, shiftSwarm = 0.0,
                             colorScheme = 'black', test = 'non-param',
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

plt.ylim(-4,4.5)
plt.ylabel('NLR', **plotChars)
plt.xlabel(' ', **plotChars)
# plt.tight_layout()
plt.savefig((dirToSave + '(0a)_{:}_{:}_NLRrainplot.png').format(str(dates), str(condCat)))
plt.show()

##################################
plottingParams = {'data':df, 
                  'hue' : condCol, 
                  'x' : 'NLI_mod',
                  'stat':'percent',
                  'hue_order':condCat
                    }

pf.NLR_distplot(condCat = condCat, pairs = pairs, colorScheme = 'white',
                                    palette = palette_cond_y27,  test = 'param',
                                    plottingParams = plottingParams, plotChars = plotChars)

plt.tight_layout()
plt.xlim(-3, 2.5)
plt.savefig((dirToSave + '(0a)_{:}_{:}_NLRDistplot.png').format(str(dates), str(condCat)))
plt.show()

########### NLR Violin/Swarm plots ####################
fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'split':True, 
                  # 'inner_kws':dict(box_width=15, whis_width=2, color=".8")
                    }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, colorScheme = 'black',
                                    hueType = None, palette = palette_cond, plotType = 'distplot', test = 'param',
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# ax.grid(axis='y')
# plt.xticks(**pltTicks, rotation = 45)
# plt.yticks(**pltTicks)
# plt.ylim(-4,4)
# fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLImodPLot.png').format(str(dates), str(condCat)))
plt.show()

######## vs. Compressions #########

fig, ax = plt.subplots(figsize = (13,9))

condition = 'activation'
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
plt.tight_layout()
plt.savefig((dirToSave + '(1b)_{:}_{:}_NLImodPLot-Comps.png').format(str(dates), condition))


######## cell average #########

fig, ax = plt.subplots(figsize = (7,7))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :10, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond_y27, test = 'param',
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
                             hueType = None, palette = palette_cond_act,
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

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, plotType = 'swarm',
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

#%%%% ctFieldThickness / ctFieldFluctuations

df_ctField = df.drop_duplicates(subset = 'ctFieldFluctuAmpli')

plottingParams = {'data':df_ctField, 
                  'x' : condCol, 
                  'y' : 'ctFieldFluctuAmpli',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

ylim = 20000


####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, 1500)
fig.suptitle(str(dates), **plotChars)

# plt.ylim(0,1500)
plt.yticks(**pltTicks)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_ctFieldThickness_Conditions.png').format(str(dates), str(condCat)))

#%%% NLI vs Fluctuation

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'x' : ('normFluctu', 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'data' : avgDf,
                  'hue' : (condCol, 'first'),
                  'palette' : palette_cond_y27,
                  'hue_order' : condCat,
                  's' : 100
                   }

fig, ax = pf.NLIvFluctu(fig, ax, plottingParams = plottingParams, plotChars = plotChars)

plt.xlim(0, 0.8)
plt.show()

#%%%  NLI-corr vs Fluctuation

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'palette' : palette_cond_y27,
                  's' : 100,
                  'hue' : condCol,
                  'hue_order' : condCat,
                   }

fig, ax, data_nli = pf.NLIcorrvFluctu(fig, ax, data = df, plottingParams = plottingParams, 
                                      plotChars = plotChars)


plt.xlim(0, 1.5)
plt.show()
plt.savefig((dirToSave + '(11b)_{:}_{:}_NLI-corrvFluctu.png').format(str(dates), str(condCat)))

#%%%% E vs H0 

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol,  palette = palette_cond_act, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))


fig, axes = plt.subplots(figsize = (13,9))
fig, axes, avgDf = pf.EvH0_LogCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cond_act, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_logAvgEvH_Conditions.png').format(str(dates), str(condCat)))

#%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 4, 
                   }

ylim = 20000

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond_act,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))


################# E-normalised ###################
fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_norm', 'logAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :4, 
                    }

fig, ax, pvals = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond_act, labels = labels,
                             plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(2d)_{:}_{:}_E-NormBoxplot_logAvg.png').format(str(dates), str(condCat)))
plt.show()


#%%% Plotnine paired plots

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()
N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)

measure = 'E_eff'
stat = 'wAvg'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'two-sided', palette = palette_cond_act,
                     plotChars = plotChars, plotTicks = plotTicks)


# plt.ylim(-2, 1)
# plt.yticks([-2, -1, 0, 1], [-2, -1, 0, 1], **plotTicks)

plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))

#%% Calling data

#Dates available : '22-12-07_M4 & 23-02-02_M1 & 23-04-19_M1 & 23-04-25_M1 & 
#23-05-10_M1 & 23-05-10_M3 & 23-05-23_M1 & 23-07-07_M2 & 23-07-12_M1'

filename = 'VWC_allOptoRhoA_NoActivation'
GlobalTable_AJ = taka.getMergedTable(filename)
# columns_AJ = GlobalTable_AJ.columns.values

filename2 = 'MecaData_CellTypes_V2_JV'
GlobalTable_JV = taka.getMergedTable(filename2)
# GlobalTable_JV_2 = GlobalTable_JV[[columns_AJ]]

GlobalTable = pd.concat([GlobalTable_AJ, GlobalTable_JV], join = 'inner')
columns = GlobalTable.columns.values

dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/24-11-13_PielTeamMeeting/'


#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable).reset_index(drop=True)

# data = GlobalTable.reset_index(drop=True)

dates = ['all'] #, '22-12-07']

# celltype = ['Atcc-2023', 'optoRhoA']
# labels = ['3T3 WT', 'OptoRhoA']

celltype = ['Atcc-2023', 'optoRhoA', 'WT', 'ctrl', 'DictyBase-WT', 'mouse-primary']
labels = ['3T3 WT', 'OptoRhoA', 'MDCK', 'MacroΦ', 'Dicty', 'DC']

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            ((data['substrate'] == '20um fibronectin discs') | (data['substrate'] == 'BSA coated glass')), 
            # (data['ctFieldThickness'] < 1000), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1500),
            (data['E_eff'] <= 30000),
            ((data['drug'] == 'none') | (data['drug'] == 'dmso')),
            # ((data['compression duration'] == '1.5s') | (data['compression duration'] == '1s') ),
            # ((data['normal field'] == 5)| (data['normal field'] == 15)),
            (data['cell subtype'].apply(lambda x : x in celltype)),
            ]

df = pf.filterDf(Filters, data)

condCol, condCat = 'cell subtype', celltype
avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[avgDf[('compNum', 'count')] > 3]

plotChars = {'color' : '#ffffff', 'fontsize' : 20}
plotTicks = {'color' : '#ffffff', 'fontsize' : 25}

pairs = [['Atcc-2023', 'optoRhoA'], ['Atcc-2023', 'WT'], ['Atcc-2023', 'DictyBase-WT'],
         ['Atcc-2023', 'mouse-primary'], ['mouse-primary', 'optoRhoA'], 
         ['Atcc-2023', 'MacroΦ'], ['DictyBase-WT', 'optoRhoA']]
pairs = None

N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
# palette_cond = ['#f4b676', '#929292']
palette_cond = ['#f4b676', '#929292','#5ec391',' #ff7f7f', '#B676F4', '#76B4F4']

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
                             colorScheme = 'black', test = 'non-param', pointSize = 10,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

plt.ylim(-4,4.5)
plt.ylabel('NLR', **plotChars)
plt.xlabel(' ', **plotChars)
# plt.tight_layout()
plt.savefig((dirToSave + '(0a)_{:}_{:}_NLRrainplot.png').format(str(dates), str(condCat)), dpi = 100)
plt.show()

######## cell average #########

plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(figsize = (13,13))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 8, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                             labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
# fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1c)_{:}_NLImodPLot_cellAvg.png').format(str(condCat)))
plt.show()

#%% Calling data

#Dates available : '23-04-24'
filename = 'VWC_Blebbistatin'
GlobalTable = taka.getMergedTable(filename)

dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/24-11-13_PielTeamMeeting/'


#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)

drugs = ['dmso_10', 'blebbi_10']
labels = ['DMSO', '10µM Blebbistatin']

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            # (data['ctFieldThickness'] < 1000), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1200),
            (data['E_eff'] <= 30000),
            (data['drug'].apply(lambda x : x in drugs)),
            ]

df = pf.filterDf(Filters, data)

condCol, condCat = 'drug', drugs
avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[avgDf[('compNum', 'count')] > 3]

plotChars = {'color' : '#ffffff', 'fontsize' : 20}
plotTicks = {'color' : '#ffffff', 'fontsize' : 18}

pairs = [['dmso_10', 'blebbi_10']]

N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
palette_cond = ['#929292', '#006666']

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
                             colorScheme = 'black', test = 'non-param', pointSize = 40,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

plt.ylim(-4,4.5)
plt.ylabel('NLR', **plotChars)
plt.xlabel(' ', **plotChars)
# plt.tight_layout()
plt.savefig((dirToSave + '(0a)_{:}_{:}_NLRrainplot.png').format(str(dates), str(condCat)))
plt.show()

######## cell average #########


fig, ax = plt.subplots(figsize = (7,7))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :10, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1c)_{:}_{:}_NLImodPLot_cellAvg.png').format(str(dates), str(condCat)))
plt.show()


