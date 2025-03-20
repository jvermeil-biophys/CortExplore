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
                'methods_H0':['Chadwick'],
                'zones_H0':['%f_15'],
                'method_bestH0':'Chadwick', 
                'zone_bestH0':'%f_15',
                'doVWCFit' : True,
                'VWCFitMethods' : ['Full'],
                'doChadwickFit' : True,
                'ChadwickFitMethods' : ['Full', 'f_<_400'],
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
                        'Plot_Ratio':False,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        }
Task = '24-12-27'


fitsSubDir = 'VWC_Fibcon_24-12-27'

GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', 
                            fileName = fitsSubDir, save = True, PLOT = False, source = 'Python',
                            fitSettings = fitSettings, plotSettings = plotSettings,
                            fitsSubDir = fitsSubDir) # task = 'updateExisting'

#%% Calling data - _24-11-28
#
filename = 'VWC_Fibcon_24-11-28'
GlobalTable = taka.getMergedTable(filename)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/Fibcon/24.11.28'

#Dates available : ['24-05-29', '24-02-21', '24-06-07', '24-06-08']

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)

dates = ['24-11-28']

manips = ['M1', 'M2', 'M3', 'M4'] 


labels = []

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1500),
            (data['E_eff'] <= 30000),
            (data['compNum'] < 6),
            (data['date'].apply(lambda x : x in dates)),
            # (data['wellID'].apply(lambda x : x in wells)),
            # (data['drug'].apply(lambda x : x in drugs)),
            ]

df = pf.filterDf(Filters, data)
# df = df.drop(df[(df['drug'] == 'doxy_act') & (df['compNum'] == 1)].index)
# df = df.drop(df[(df['drug'] == 'doxy_2_Y27_10_act') & (df['compNum'] == 1)].index)

pairs = [['doxy', 'doxy_act']] 

condCol, condCat = 'manip', manips
# condCol, condCat = 'date', dates

plotChars = {'color' : 'white', 'fontsize' : 18}
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
palette_cond = ['#808080', '#ffdb19', '#b29600', '#99c3cf', '#4d96ab' ,'#005f79']

swarmPointSize = 8

#%% Calling data - 24-12-27

filename = 'VWC_Fibcon_24-12-27'
GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/Fibcon/25.01.27_MeetingwithEMBL/'

#Dates available : [24-12-27]

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)

dates = ['24-12-27']
manips = ['M6', 'M1', 'M2', 'M5', 'M3', 'M4'] 
labels = ['7XFN\nNI', '7XFN\n+Dox', '7XFN\n+Dox\n+Light', '1XFN\nNI', '1XFN\n+Dox', '1XFN\n+Dox\n+Light']

# manips = [ 'M1', 'M2', 'M3', 'M4'] 
# labels = [ '7XFN+\nDox', '7XFN+\nDox+\nLight', '1XFN+\nDox', '1XFN+\nDox+\nLight']

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['UI_Valid'] == True),
            # (data['ctFieldThickness'] < 1000), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1500),
            (data['E_eff'] <= 30000),
            (data['compNum'] <= 6),
            (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)
condCol, condCat = 'manip', manips
avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[avgDf[('compNum', 'count')] > 2]

plotChars = {'color' : '#ffffff', 'fontsize' : 25}
plotTicks = {'color' : '#ffffff', 'fontsize' : 15}

pairs = [['M6', 'M1'], ['M1', 'M2'], ['M5', 'M3'], ['M3', 'M4']] 
# pairs = [ ['M1', 'M2'], ['M3', 'M4']] 

N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
palette_cond = ['#808080', '#ffdb19', '#b29600', '#c1dbe2', '#7fafbc' ,'#003948']
# palette_cond = [ '#ffdb19', '#b29600', '#7fafbc' ,'#003948']

swarmPointSize = 10


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

# plt.ylim(-4,4.5)
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
                                    palette = palette_cond,  test = 'param',
                                    plottingParams = plottingParams, plotChars = plotChars)

plt.tight_layout()
plt.xlim(-3, 2.5)
plt.savefig((dirToSave + '(0a)_{:}_{:}_NLRDistplot.png').format(str(dates), str(condCat)))
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
                             hueType = None, palette = palette_cond, test = 'param',
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1c)_{:}_{:}_NLImodPLot_cellAvg.png').format(str(dates), str(condCat)))
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


####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs,
                                           labels = labels, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
# plt.ylim(0,1500)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_ctFieldThickness_Conditions.png').format(str(dates), str(condCat)))


#%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 8, 
                   }

ylim = 20000

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                             labels = labels, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels = y_labels,**plotTicks)
ax.set_xticks([0,1,2,3,4,5], labels = labels, **plotTicks)

plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))

####################### cell weighted average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_norm', 'mean'),
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
plt.savefig((dirToSave + '(4d)_{:}_{:}_E-normBoxplot_weightedCellAverage.png').format(str(dates), str(condCat)))
plt.show()


#%%% Box plots - H0

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'bestH0_log',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : 8, 
                   }

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = np.asarray([100 ,250, 500, 1000, 1500, 2500])
y_ticks = np.log10(np.asarray(y_labels))

ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)
ax.set_xticks([0,1,2,3,4,5], labels = labels, **plotTicks)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))


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
ax.set_xticks([0,1,2,3,4,5], labels = labels, **plotTicks)

plt.savefig((dirToSave + '(3e)_{:}_{:}_H0Boxplot_meancellAverage.png').format(str(dates), str(condCat)))
plt.show()


#%%% E vs H0
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, df = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))


avgDf = avgDf = pf.createAvgDf(df, condCol, dataFluoPath = None, e_norm = True)

#%%% Plotnine paired plots

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()
N_point = len(dfPairs['dateCell', 'first'].unique())


measure = 'E_eff_log'
stat = 'mean'
test = 'greater'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = test, palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 8000))



plt.xticks([1, 2, 3, 4], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))


#%%% Point plots

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

