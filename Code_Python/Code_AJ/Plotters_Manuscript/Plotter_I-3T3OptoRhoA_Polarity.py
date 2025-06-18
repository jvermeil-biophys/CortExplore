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
                'methods_H0':['Chadwick', 'VWC', 'Dimitriadis'],
                'zones_H0':['%f_10', '%f_100'],
                'method_bestH0':'VWC', 
                'zone_bestH0':'%f_100',
                'doVWCFit' : True,
                'VWCFitMethods' : ['Full'],
                'doChadwickFit' : True,
                'ChadwickFitMethods' : ['Full', 'f_<_400', 'f_in_400_800'],
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
plot_stressHalfWidth = 100

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



Task = '24-12-14 & 24-12-20_M1 & 24-12-20_M2 & 24-12-20_M5 & 25-01-14 & 25-01-23 & 25-01-21 & 25-02-28 & 25-03-12'
# Task = '24-12-14 & 24-12-20 & & 25-01-14 & 25-01-23 & 25-01-21 & 25-02-28 & 25-03-12'
# Task = '25-03-12_M2_P1_C3'

fitsSubDir = 'VWC_optoRhoAVB-NS_updated_25-03-24'
# fitsSubDir = 'VWC_optoRhoAVB-NS_'+Task

GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', 
                            fileName = fitsSubDir, save = True, PLOT = False, source = 'Python',
                            fitSettings = fitSettings, plotSettings = plotSettings,
                            fitsSubDir = fitsSubDir) # task = 'updateExisting'


#%% Calling data - Using non-sorted cells with antibiotics
'VWC_optoRhoAVB-NS_updated_25-03-24' 
filename = 'VWC_optoRhoAVB-NS_updated_25-03-20' 
 
GlobalTable = taka.getMergedTable(filename, mergeUMS = True)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/FiguresForManuscript/3T3OptoRhoA-I_Polarity/'

dataActPath = 'D:/Anumita/MagneticPincherData/Raw/ActivationData/25-01-20_ActivationData.csv'

dataAnglesPath = 'D:/Anumita/MagneticPincherData/Data_Polarization/'

data = pf.createDataTable(GlobalTable, dataActPath = dataActPath, dataAnglesPath = dataAnglesPath)

#%%%% Create dataframe for plotting

dates = ['24-12-14', '24-12-20', '25-01-14', '25-01-23', '25-01-21', '25-02-28', '25-03-12']
# dates = ['25-02-28', '25-03-12']

drugs = ['doxy', 'doxy_act']

manips = ['M5', 'M1', 'M2', 'M3', 'M4']
activation = ['no light',  'side',  'away from beads',  'at beads', 'global']
labels = [ 'Dox','Dox + Side',  'Dox + Away', 'Dox + At', 'Dox + Global']
# labels = [ 'Control', 'Polarized\nFront', 'Side', 'Polarized\nRear', 'Global\nContraction']
# # labels = []
palette_cond = ['#8d9095', '#ffb230',  '#c66440', '#8ec582', '#325eb6']

# activation = ['none',   'away from beads'] 
# labels = [ 'Dox', 'Dox + Away'] 
# palette_cond = ['#8d9095', '#c66440'] 

# activation = ['no light', 'side'] 
# labels = [ 'Dox', 'Side'] 
# palette_cond = ['#8d9095','#ffb230']

# activation = ['none', 'global']
# labels = [ 'Dox',  'Global Activation']
# palette_cond = ['#8d9095', '#325eb6']

# activation = ['none', 'at beads']
# labels = [ 'Dox',  'At Beads']
# palette_cond = ['#8d9095', '#8ec582']

# drugs = ['none', 'doxy'] 
# labels = [ 'Control',  'Induced']
# palette_cond = ['#8d9095', '#0000FF']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] >= 0.90),
            (data['H0_vwc_Full'] <= 1200) & (data['H0_vwc_Full'] > 100),
            (data['E_eff'] <= 30000),
            (data['ctFieldThickness'] <= 1200),
            (data['compNum'] <= 10),
            (data['activation type'].apply(lambda x : x in activation)),
            (data['manip'].apply(lambda x : x in manips)),
            (data['drug'].apply(lambda x : x in drugs)),
            (data['date'].apply(lambda x : x in dates)),
            ]

df = pf.filterDf(Filters, data)
# df = df.drop(df[(df['drug'] == 'doxy_act') & (df['compNum'] == 1)].index)

condCol, condCat = 'activation type', activation
# condCol, condCat = 'drug', drugs


# mask = (data['cellID'] == '25-01-14_M2_P2_C4')
# df = df[~mask]

avgDf = pf.createAvgDf(df, condCol)#, dataAnglesPath = dataAnglesPath)
avgDf = avgDf[(avgDf[('compNum', 'count')] > 2)]

pairs = [['no light', 'at beads'],  ['no light' , 'side'], ['no light' , 'away from beads'], ['no light' , 'global']]
# pairs = None
# pairs = [activation]

plotChars = {'color' : '#000000', 'fontsize' : 30}
plotTicks = {'color' : '#000000', 'fontsize' : 28}

N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)

swarmPointSize = 6



#%%%% Chadwick / Dimitriadis model

#%%%% Plot NLImod

########################################

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax = plt.subplots(figsize=(15, 10))

fig, ax, medians = pf.rainplot(fig, ax, condCat, palette = palette_cond, 
                             labels = labels, pairs = pairs, shiftBox = 0.1, shiftSwarm = 0.0,
                             colorScheme = 'white', test = 'non-param' ,pointSize = 30,
                             plottingParams = plottingParams, plotTicks = plotTicks, 
                             plotChars = plotChars)

# plt.ylim(-4,4.5)
plt.ylabel('NLR', **plotChars)
plt.xlabel(' ', **plotChars)
# plt.tight_layout()
plt.savefig((dirToSave + '(0a)_{:}_{:}_NLRrainplot.pdf').format(str(dates), str(condCat)), dpi = 50)
plt.show()


################### box plots #######################
fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = None, colorScheme = 'white',
                                    hueType = None, palette = palette_cond, plotType = 'violin',
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(-3,3)
# fig.suptitle(str(dates), **plotChars)
plt.yticks(**plotTicks)
plt.xticks(**plotTicks)
plt.ylabel('NLR')
plt.xlabel(' ')
plt.tight_layout()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLImodPLot.png').format(str(dates), str(condCat)))
plt.show()

######## vs. Compressions #########

fig, ax = plt.subplots(figsize = (13,9))

condition = 'at beads'
df_comp = df[df[condCol] == condition]
df_comp = df_comp.drop_duplicates()

plottingParams = {'data':df_comp, 
                  'x' : 'compNum', 
                  'y' : 'NLI_mod',
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), 
                                    pairs = None, hueType = None, plotType = 'swarm',
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

# plt.ylim(-3, 3)
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
fig, ax, pvals = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                             labels = labels, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 2000, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)

# plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))
plt.show()


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

fig, ax, medians = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond, colorScheme = 'white',
                             labels = labels, plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
y_labels = [100, 500, 2000, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels,**plotTicks)
plt.tight_layout()
# plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_cellAverage.png').format(str(dates), str(condCat)))
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
# plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'E_eff_log'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'two-sided', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (2, 4.5))


plt.yscale('log')
plt.xticks([1, 2], labels, **plotTicks)
y_labels = [100, 500, 2000, 5000, 10000, 50000]
y_ticks = np.log10(np.asarray(y_labels))
plt.yticks(y_ticks, labels =y_labels,**plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'H0_vwc_Full'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'two-sided', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 1500))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))


measure = 'ctFieldThickness'
stat = 'first'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'two-sided', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 1000))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'ctFieldFluctuAmpli'
stat = 'first'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'two-sided', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = None)


plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))



measure = 'NLI_mod'
stat = 'std'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'two-sided', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits = (0, 2))



plt.xticks([1, 2], labels, **plotTicks)
plt.yticks(**plotTicks)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(12a)_{:}_{:}_{:}-{:}_PairedPlot.png').format(str(dates), str(condCat), measure, stat))

#%%%%  Pointplots / Pairedplots

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
                  'y' : ('E_eff_log', 'mean'),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (2,4.5), 
                                          pairs = pairs, normalize = False, marker = 'mean',
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.yscale('log')
y_labels = [100, 500, 2000, 5000, 10000]
y_ticks = np.log10(np.asarray(y_labels))
plt.yticks(y_ticks, labels =y_labels,**plotTicks)
plt.savefig((dirToSave + '(8a)_{:}_{:}_{:}_EPointplot.png').format(str(dates), str(condCat), stats))


# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP_E = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
#                                           pairs = pairs, normalize = True, marker = 'mean',
#                                           test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)


# plt.show()
# plt.savefig((dirToSave + '(8b)_{:}_{:}_{:}_EPointplot-Normalised.png').format(str(dates), str(condCat), stats))

# plottingParams = {'x' : (condCol, 'first'), 
#                   'y' : ('E_norm', 'mean'),
#                   'linewidth' : 1,
#                   'markersize' : 10,
#                   'markeredgecolor':'black', 
#                    }


# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,5000), 
#                                           pairs = pairs, normalize = False, marker = 'wAvg',
#                                           test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)

# plt.show()
# y_ticks = [100, 250, 500, 1000, 2500, 5000]
# ax.set_yticks(y_ticks, labels =y_ticks, fontsize=25, color = fontColour)
# plt.savefig((dirToSave + '(8a)_{:}_{:}_{:}_E-normPointplot.png').format(str(dates), str(condCat), stats))


# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP_E = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
#                                           pairs = pairs, normalize = True, marker = 'wAvg',
#                                           test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)


# plt.show()
# plt.savefig((dirToSave + '(8b)_{:}_{:}_{:}_E-normPointplot-Normalised.png').format(str(dates), str(condCat), stats))





# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP_nli = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-6,6), 
#                                           pairs = pairs, normalize = True, marker = stats,
#                                           test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)


# plt.show()
# plt.savefig((dirToSave + '(9b)_{:}_{:}_{:}_NLImodPointplot-Normalised.png').format(str(dates), str(condCat), stats))

plottingParams = { 'x' : (condCol, 'first'), 
                  'y' : ('surroundingDz', 'median'),
                  'linewidth' : 1, 
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

ylim = 1500
fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-1000,1200), 
                                          pairs = pairs, normalize = False, marker = 'first',
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.xlim((-2,3))
plt.tight_layout()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_surroundingDz.png').format(str(dates), str(condCat), stats))
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals, dfP_H = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-2,3), 
                                          pairs = pairs, normalize = True, marker = 'first',
                                          test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7b)_{:}_{:}_{:}__surroundingDz-Normalised.png').format(str(dates), str(condCat), stats))
plt.show()

#%%%% Pointplots with errorbars
dfPairs, pairedCells = pf.dfCellPairs(avgDf)
dfPairs = df[df['dateCell'].apply(lambda x : x in pairedCells)]

measure = 'NLI_mod'
plottingParams = {'x' : condCol, 
                  'y' : measure,
                  'data' : dfPairs,
                  'hue' : 'dateCell',
                  'dodge' : True,
                  'palette' : palette_cell,
                  'errorbar' : 'se'
                   }

fig, ax = plt.subplots(figsize = (13,10))

fig.patch.set_facecolor('black')

ax = sns.pointplot(**plottingParams)

plt.xticks(**plotChars)
plt.yticks(**plotChars)
plt.legend(fontsize = 15, labelcolor='linecolor', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(10a)_{:}_{:}_{:}_Errorbar.png').format(str(dates), str(condCat),  str(measure)))

#%%%% E vs H0

dfPairs, pairedCells = pf.dfCellPairs(avgDf)

dfPairs = df[df['dateCell'].apply(lambda x : x in pairedCells)]

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, df = pf.EvsH0_perCompression(fig, ax, dfPairs, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 16, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH-dfPairs_Conditions.png').format(str(dates), str(condCat)))

# dfPairs, pairedCells = pf.dfCellPairs(avgDf)
# fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
# fig, ax, df = pf.EvH0_LogCellAvg(fig, ax, dfPairs, condCat, condCol, palette = palette_cond)
# plt.legend(fontsize = 16, ncol = len(condCat))
# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH-dfPairs_wAvg.png').format(str(dates), str(condCat)))

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
plt.xlim(0,140)

# plt.savefig(dirToSave + '/Thickness/'+str(dates)+'_'+measure+'vsCompr'+str(manips)+'.png')

plt.xticks(**plotChars)
plt.yticks(**plotChars)
plt.show()

#%%%% NLR / Average / Delta NLR vs. Angle from activation

dfPairs, pairedCells = pf.dfCellPairs(avgDf)

toPlot = dfPairs.dropna(subset=[('angle_beads', 'first')])
# toPlot = toPlot.drop(toPlot[(toPlot['activation type'] != 'side')].index)

N_cols = distinctipy.get_colors(len(toPlot[('dateCell', 'first')].unique()))

plottingParams = {'data' : toPlot,
                  'x' : ('angle_beads', 'first'),
                  'y' : ('NLI_mod', 'mean'),
                  'hue' : ('dateCell', 'first'),
                  'palette' : N_cols,
                  's' : 100
                   }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.NLRvAngle(fig, ax, toPlot, condCat, condCol, pairedCells, colorScheme = 'white',
                pairs = pairs,  plotType = False, plottingParams = plottingParams,  
                palette = palette_cond, plotChars = plotChars)


# sns.scatterplot(**plottingParams)

#%%%% NLR per comp vs. Angle from activation

toPlot = df.dropna(subset=[('angle_beads')])
toPlot = toPlot.drop_duplicates(subset=['dateCell', 'compNum'])

# toPlot = toPlot.drop(toPlot[(toPlot['activation type'] != 'side')].index)

# N_cols = distinctipy.get_colors(86)

plottingParams = {'data' : df,
                  'x' : ('arc_length'),
                  'y' : ('NLI_mod'),
                  'hue' : ('dateCell'),
                  'palette' : N_cols,
                  's' : 100
                   }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.NLRvAngle(fig, ax, toPlot, condCat, condCol, pairedCells, colorScheme = 'white',
                pairs = pairs,  plotType = False, plottingParams = plottingParams,  
                palette = palette_cond, plotChars = plotChars)

ax.legend().set_visible(False)
# sns.scatterplot(**plottingParams)