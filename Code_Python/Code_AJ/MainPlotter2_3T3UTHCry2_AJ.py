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
from statannotations.Annotator import Annotator
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
                       'doChadwickFit' : False,
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
    
# Task = '24-09-05 & 24-08-26 & 24-06-07 & 24-06-08 & 24-05-29 & 24-02-21 & 24-07-15 & 24-09-12 & 24-09-24'
# Task = '24-05-29 & 24-09-05 & 24-02-21 & 24-09-24 & 24-09-12'
Task = '24-08-26 & 24-06-07 & 24-06-08 & 24-07-15'


fitsSubDir = 'VWC_AllCrosslinking-OkExpts_24-09-31'

GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', 
                            fileName = fitsSubDir, save = True, PLOT = False, source = 'Python',
                            fitSettings = fitSettings, plotSettings = plotSettings,
                            fitsSubDir = fitsSubDir) # task = 'updateExisting'

#%% Calling data - Analysing with VWC global tables - 24-06-25
#'VWC_AllUTHCry2_24-06-25_Y27_Crosslinking' : '24-02-21 & 24-05-29 & 24-06-07 & 24-06-08'
#'VWC_AllUTHCry2_24-07-17_Y27_Crosslinking' : '24-02-21 & 24-05-29 & 24-06-07 & 24-06-08 & 24-07-15'

filename = 'VWC_AllUTHCry2_24-07-17_Y27_Crosslinking'
GlobalTable = taka.getMergedTable(filename)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/Crosslinkers/Plots/24-08-14'

#Dates available : ['24-05-29', '24-02-21', '24-06-07', '24-06-08']

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)

dates = ['24-07-15']

drugs = ['doxy', 'doxy_act'] #, 'Y27_10', 'doxy_2_Y27_10', 'doxy_2_Y27_10_act']

# labels = ['None'] #'Dox', 'Dox + Light'] #, 'Y27', 'Dox + Y27', 'Dox + Y27 + Light']
labels = []

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1500),
            (data['E_eff'] <= 30000),
            (data['compNum'] > 5),
            (data['date'].apply(lambda x : x in dates)),
            # (data['wellID'].apply(lambda x : x in wells)),
            (data['drug'].apply(lambda x : x in drugs)),
            ]

df = pf.filterDf(Filters, data)
# df = df.drop(df[(df['drug'] == 'doxy_act') & (df['compNum'] == 1)].index)
# df = df.drop(df[(df['drug'] == 'doxy_2_Y27_10_act') & (df['compNum'] == 1)].index)

pairs = [['doxy', 'doxy_act']] #, ['none', 'Y27_10'], ['doxy_2_Y27_10', 'doxy_2_Y27_10_act']]
# pairs = [['24-02-21', '24-05-29'], ['24-02-21', '24-06-07'], ['24-05-29', '24-06-07'], 
#          ['24-07-15', '24-05-29'], ['24-07-15', '24-06-07'], ['24-07-15', '24-02-21']]

# pairedPairs = [['doxy', 'doxy_act'], ['doxy_2_Y27_10', 'doxy_2_Y27_10_act']]

condCol, condCat = 'drug', drugs
# condCol, condCat = 'date', dates


plotChars = {'color' : 'white', 'fontsize' : 18}
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
palette_cond = ['#808080', '#ffdb19', '#b29600', '#99c3cf', '#4d96ab' ,'#005f79']

swarmPointSize = 8

#%%%% Plot NLI - Scatterplot

plotSettings = {'markersize' : 20,
                'mec' : 'k',
                'sort' : False,
                'ls' : 'solid'
                }

marker_dates = {'24-05-29': 'o',
                '24-02-21': '^', 
                '24-06-07':'*', 
                '24-06-08' : '*', 
                '24-07-15' : 'P'}


fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.plotNLI_Scatter(fig, ax, df, dates, condCat, condCol, pairs, labels = labels, 
                                    plotSettings = plotSettings, marker_dates = marker_dates,
                                    plotChars = plotChars) 
           

fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLIPLot-Scatter.png').format(str(dates), str(condCat)))

#Plot averages across experiments
plotSettings = {'marker' : 'o', 
                'markersize' : 12,
                'sort' : False,
                'ls' : 'solid',
                'mec' : 'k'}


fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.plotNLI_Scatter_Avg(fig, ax, df,condCat, condCol, pairs, labels = labels, 
                                    plotSettings = plotSettings,
                                    plotChars = plotChars) 
            

fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLIPLot-Scatter_Avg.png').format(str(dates), str(condCat)))


#%%%% Plot NLI - Scatterplot (vs. compressions)

df_comp = df[df.drug == 'doxy_2_Y27_10']

plotSettings = {'markersize' : 20,
                'mec' : 'k',
                'sort' : True,
                }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.plotNLI_Scatter(fig, ax, df_comp, dates, condCat = np.sort(df_comp.compNum.unique()), 
                                    condCol = 'compNum', pairs = None, labels = labels, 
                                    plotSettings = plotSettings, marker_dates = marker_dates,
                                    plotChars = plotChars) 
            

fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLIPLot-Scatter-Comps.png').format(str(dates), str(condCat)))

# Plot averages across experiments
plotSettings = {'marker' : 'o', 
                'markersize' : 12,
                'sort' : True,
                'mec' : 'k'}

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.plotNLI_Scatter_Avg(fig, ax, df_comp, condCat = np.sort(df_comp.compNum.unique()), 
                                    condCol = 'compNum', pairs = None, labels = labels, 
                                    plotSettings = plotSettings, 
                                    plotChars = plotChars) 
            
           
           
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLIPLot-Scatter-Comps_Avg.png').format(str(dates), str(condCat)))

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
# df_comp = df[df.drug == 'doxy_2_Y27_10_act']

# plt.style.use('dark_background')

# fig, ax = plt.subplots(figsize = (13,9))
# fig, ax, pvals = pf.plotNLI(fig, ax, df_comp, condCat =  np.sort(df_comp.compNum.unique()), condCol = 'compNum',
#                             pairs = None, labels = [], **plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(1b)_{:}_{:}_NLIPLot-Comps.png').format(str(dates), str(condCat)))

# plt.style.use('default')

#%%%% E vs H0
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0(fig, ax, df, condCat, condCol, hueType = 'NLI_Plot')
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_EvH_NLI.png').format(str(dates), str(condCat)))


fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0(fig, ax, df, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))

#%%% Box plots - H0

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : swarmPointSize, 
                   }

######################## Hue type 'CellID'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)
# plt.ylim(0, 1500)
fig.suptitle(str(dates), **plotChars)
plt.legend(fontsize = 6, ncol = 6)
plt.show()
plt.savefig((dirToSave + '(3a)_{:}_{:}_H0Boxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)
# plt.ylim(0, 1500)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'hue_order' : ['non-linear', 'intermediate', 'linear'],
                  'size' : swarmPointSize,  
                    }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, 1500)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(3c)_{:}_{:}_H0Boxplot_NLI.png').format(str(dates), str(condCat)))

#%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

ylim = 20000

######################## Hue type 'CellID'#######################
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)
# plt.ylim(0, ylim)
plt.legend(fontsize = 6, ncol = 6)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)
# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : swarmPointSize,  
                    }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

#%%% Boxplots average per cell - E_eff

avgDf = pf.createAvgDf(df, condCol)

stats = 'median'
plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', stats),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : swarmPointSize, 
                  'edgecolor':'k', 
             
                   }

plotChars = {'fontsize' : 15, 
             'color' : '#ffffff',
             }

######################## Hue type 'CellID'#######################
N = len(avgDf['cellID', 'first'].unique())
palette_cell = distinctipy.get_colors(N)
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)
# plt.ylim(0, 20000)
plt.legend(fontsize = 6, ncol = 6)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(5a)_{:}_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat), stats))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)
# plt.ylim(0, 20000)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(5b)_{:}_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat), stats))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', stats),
                  'order' : condCat,
                  'hue_order' : ['linear', 'intermediate', 'non-linear'],
                  'linewidth' : 1, 
                  'size' : swarmPointSize, 
                  'edgecolor':'k', 
                    }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', labels = labels,
                              plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, 20000)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(5c)_{:}_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat), stats))

#%%% Boxplots average per cell - H0

avgDf = pf.createAvgDf(df, condCol)

stats = 'median'
plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('H0_vwc_Full', stats),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : swarmPointSize, 
                  'edgecolor':'k', 
             
                   }

ylim = 2000

######################## Hue type 'CellID'#######################
N = len(avgDf['cellID', 'first'].unique())
palette_cell = distinctipy.get_colors(N)
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)
plt.ylim(0, ylim)
plt.legend(fontsize = 6, ncol = 6)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(6a)_{:}_{:}_{:}_H0Boxplot_Avg_CellID.png').format(str(dates), str(condCat), stats))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)
plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(6b)_{:}_{:}_{:}_H0Boxplot_Avg_Conditions.png').format(str(dates), str(condCat), stats))


#%%% Pointplots 

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()

N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)

stats = 'mean'
plottingParams = {'data':dfPairs, 
                  'x' : (condCol, 'first'), 
                  'y' : ('H0_vwc_Full', stats),
                  'linewidth' : 1, 
                  'markersize' : 15,
                  'markeredgecolor':'black', 
                   }

ylim = 1000
fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_H0Pointplot.png').format(str(dates), str(condCat), stats))

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, condCatPoint, pairedCells, ylim = (0,3), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_H0Pointplot-Normalised.png').format(str(dates), str(condCat), stats))


ylim = 20000
plottingParams = {'data':dfPairs, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', stats),
                  'linewidth' : 1,
                  'markersize' : 15,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_EPointplot.png').format(str(dates), str(condCat), stats))

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, condCatPoint, pairedCells, ylim = (0,10), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_EPointplot-Normalised.png').format(str(dates), str(condCat), stats))

#%%% K vs Y

# plottingParams = {'data':df, 
#                   's' : 50
#                    }

# pf.KvY(hueType = 'NLI_Plot', condCat = condCat, condCol = condCol,
#                  plottingParams = plottingParams, plotChars = plotChars)
# plt.tight_layout()
# plt.show()

plottingParams = {'data':df, 
                  's' : 50
                   }
pf.KvY(condCat = condCat, condCol = condCol, pairs = pairs, plottingParams = plottingParams, plotChars = plotChars)
plt.show()


#%% Calling data - 24-08-20

filename = 'VWC_24-08-19_Y27_Crosslinking_24-08-20'
GlobalTable = taka.getMergedTable(filename)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/Crosslinkers/24-08-20_Mechanics'

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)
dates = '24-08-19'

drugs = ['doxy', 'doxy_act', 'doxy_2_Y27_10', 'doxy_2_Y27_10_act']

# labels = ['None'] #'Dox', 'Dox + Light'] #, 'Y27', 'Dox + Y27', 'Dox + Y27 + Light']
labels = []

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1500),
            (data['E_eff'] <= 30000),
            # (data['compNum'] <= 6),
            (data['date'].apply(lambda x : x in dates)),

            (data['drug'].apply(lambda x : x in drugs)),
            ]

df = pf.filterDf(Filters, data)
df = df.drop(df[(df['drug'] == 'doxy_act') & (df['compNum'] < 2)].index)
# df = df.drop(df[(df['drug'] == 'doxy_2_Y27_10_act') & (df['compNum'] < 2)].index)

pairs = [['doxy', 'doxy_act'],  ['doxy_2_Y27_10', 'doxy_2_Y27_10_act'], ['doxy', 'doxy_2_Y27_10']]

# pairedPairs = [['doxy', 'doxy_act'], ['doxy_2_Y27_10', 'doxy_2_Y27_10_act']]

condCol, condCat = 'drug', drugs

plotChars = {'color' : 'white', 'fontsize' : 18}
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
palette_cond = ['#808080', '#ffdb19', '#b29600', '#99c3cf', '#4d96ab' ,'#005f79']

swarmPointSize = 8

#%%%% Plot NLI

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.plotNLI(fig, ax, df, condCat, condCol,
                            pairs = pairs, labels = labels, **plotChars)

# fig.suptitle(str(dates), **plotChars)
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()
# plt.savefig((dirToSave + '(1b)_{:}_{:}_NLIPLot.png').format(str(dates), str(condCat)))

plt.style.use('default')


####### vs. Compressions #########
# df_comp = df[df.drug == 'doxy_2_Y27_10_act']

# plt.style.use('dark_background')

# fig, ax = plt.subplots(figsize = (13,9))
# fig, ax, pvals = pf.plotNLI(fig, ax, df_comp, condCat =  np.sort(df_comp.compNum.unique()), condCol = 'compNum',
#                             pairs = None, labels = [], **plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(1b)_{:}_{:}_NLIPLot-Comps.png').format(str(dates), str(condCat)))

# plt.style.use('default')

#%%%% E vs H0
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0(fig, ax, df, condCat, condCol, hueType = 'NLI_Plot')
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_EvH_NLI.png').format(str(dates), str(condCat)))


fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0(fig, ax, df, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))

#%%% Box plots - H0

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : swarmPointSize, 
                   }

######################## Hue type 'CellID'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)
# plt.ylim(0, 1500)
fig.suptitle(str(dates), **plotChars)
plt.legend(fontsize = 6, ncol = 6)
plt.show()
plt.savefig((dirToSave + '(3a)_{:}_{:}_H0Boxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)
# plt.ylim(0, 1500)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'hue_order' : ['non-linear', 'intermediate', 'linear'],
                  'size' : swarmPointSize,  
                    }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, 1500)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(3c)_{:}_{:}_H0Boxplot_NLI.png').format(str(dates), str(condCat)))

#%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

ylim = 20000

######################## Hue type 'CellID'#######################
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)
# plt.ylim(0, ylim)
plt.legend(fontsize = 6, ncol = 6)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)
# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : swarmPointSize,  
                    }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

#%% Calling data - 24-08-27

filename = 'VWC_24-08-26_Y27_Crosslinking_24-08-27'
GlobalTable = taka.getMergedTable(filename)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/Crosslinkers/24-08-26_Mechanics/'

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)
dates = '24-08-26'

# drugs = ['doxy', 'doxy_act']
manips = ['M1', 'M2', 'M3', 'M4']

labels = ['Low light', 'High light', 'High light + Crosslink', 'Control']
# labels = []

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1500),
            # (data['E_eff'] <= 30000),
            # (data['compNum'] <= 6),
            (data['date'].apply(lambda x : x in dates)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['drug'].apply(lambda x : x in drugs)),
            ]

df = pf.filterDf(Filters, data)
# df = df.drop(df[(df['manip'] == 'M3') & (df['compNum'] < 1)].index)
# df = df.drop(df[(df['drug'] == 'doxy_2_Y27_10_act') & (df['compNum'] < 3)].index)
avgDf = pf.createAvgDf(df, condCol)

pairs = [['M1', 'M2'], ['M1', 'M4'], ['M2', 'M3'],  ['M1', 'M3']]

condCol, condCat = 'manip', manips

plotChars = {'color' : 'white', 'fontsize' : 18}
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
palette_cond = ['#808080', '#ffdb19', '#b29600', '#99c3cf', '#4d96ab' ,'#005f79']

swarmPointSize = 8

#%%%% Plot NLImod

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = None,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLImodPLot.png').format(str(dates), str(condCat)))

######## vs. Compressions #########

# fig, ax = plt.subplots(figsize = (13,9))

# condition = 'M1'
# df_comp = df[df.manip == condition]

# plottingParams = {'data':df_comp, 
#                   'x' : 'compNum', 
#                   'y' : 'NLI_mod',
#                   'linewidth' : 1, 
#                   'size' :swarmPointSize, 
#                    }

# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), pairs = pairs, hueType = None,
#                                     labels = [], plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(-3, 3)
# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.savefig((dirToSave + '(1b)_{:}_{:}_{:}_NLImodPLot-Comps.png').format(str(dates), condition, str(condCat)))

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



fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, hueType = None,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1c)_{:}_NLImodPLot_cellAverage.png').format(str(condCat)))



#%%%% Plot NLI

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.plotNLI(fig, ax, df, condCat, condCol,
                            pairs = pairs, labels = labels, **plotChars)

fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(1b)_{:}_{:}_NLIPLot.png').format(str(dates), str(condCat)))

plt.style.use('default')


###### vs. Compressions #########
condition = 'M4'
df_comp = df[df.manip == condition]

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize = (13,9))
fig, ax, pvals = pf.plotNLI(fig, ax, df_comp, condCat =  np.sort(df_comp.compNum.unique()), condCol = 'compNum',
                            pairs = None, labels = [], **plotChars)

fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(1b)_{:}_{:}_{:}_NLIPLot-Comps.png').format(str(dates), condition, str(condCat)))

plt.style.use('default')

#%%% Box plots - H0

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : swarmPointSize, 
                   }

######################## Hue type 'CellID'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)
# plt.ylim(0, 1500)
fig.suptitle(str(dates), **plotChars)
plt.legend(fontsize = 6, ncol = 6)
plt.show()
plt.savefig((dirToSave + '(3a)_{:}_{:}_H0Boxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)
# plt.ylim(0, 1500)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'hue_order' : ['non-linear', 'intermediate', 'linear'],
                  'size' : swarmPointSize,  
                    }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, 1500)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(3c)_{:}_{:}_H0Boxplot_NLI.png').format(str(dates), str(condCat)))

#%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

ylim = 20000

######################## Hue type 'CellID'#######################
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)
# plt.ylim(0, ylim)
plt.legend(fontsize = 6, ncol = 6, loc = 'best')
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.pdf').format(str(dates), str(condCat)), format = 'pdf')

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)
# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.pdf').format(str(dates), str(condCat)), format = 'pdf')

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : swarmPointSize,  
                    }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.pdf').format(str(dates), str(condCat)), format = 'pdf')

#%%% E vs H0
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0(fig, ax, df, condCat, condCol, hueType = 'NLI_Plot')
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_EvH_NLI.png').format(str(dates), str(condCat)))


fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0(fig, ax, df, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0(fig, ax, df, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2c)_{:}_{:}_EvH_CellID.png').format(str(dates), str(condCat)))

#%%% Boxplots - New NLI modulus

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = None,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(3a)_{:}_{:}_NLI-mod_Boxplot.png').format(str(dates), str(condCat)))


#%% Calling data - All crosslinkers - 24-08-20

filename = 'VWC_AllCrosslinking_24-08-28'
GlobalTable = taka.getMergedTable(filename)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/Crosslinkers/AcrossExperiments_Mechanics/'

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)
# dates =  ['24-02-21'] #, '24-05-29'] #, '24-05-29', '24-06-08' , '24-08-26']
# drugs = ['none', 'doxy', 'doxy_act', 'Y27_10', 'doxy_2_Y27_10', 'doxy_2_Y27_10_act']
# labels = ['Control', 'Dox', 'Dox + Light', 'Y27 10μM', 'Y27 + Dox', 'Y27 + Dox + Light']

drugs = [ 'none', 'doxy', 'doxy_act'] #,  'doxy_2_Y27_10', 'doxy_2_Y27_10_act']
labels = [ 'Control', 'Dox', 'Dox + Light'] #, 'Y27 + Dox', 'Y27 + Dox + Light']

# dates = ['24-02-21', '24-05-29', '24-06-08' , '24-08-26']
# drugs = ['none']

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1500),
            (data['E_eff'] <= 30000),
            (data['compNum'] <= 6),
            (data['date'].apply(lambda x : x in dates)),
            (data['drug'].apply(lambda x : x in drugs)),
            ]

df = pf.filterDf(Filters, data)
df = df.drop(df[(df['drug'] == 'doxy_act') & (df['compNum'] == 1)].index)
df = df.drop(df[(df['drug'] == 'doxy_2_Y27_10_act') & (df['compNum'] ==1)].index)

condCol, condCat = 'drug', drugs
avgDf = pf.createAvgDf(df, condCol)

pairs = [['doxy', 'doxy_act']] #, ['Y27_10', 'doxy_2_Y27_10'],
          # ['doxy_2_Y27_10','doxy_2_Y27_10_act'], ['none', 'Y27_10']]

# pairs = [['doxy', 'doxy_act'], ['doxy_2_Y27_10','doxy_2_Y27_10_act']]
# 
# pairs = [['24-02-21', '24-05-29'], ['24-05-29', '24-06-07'], ['24-06-07' , '24-08-26'], ['24-07-15', '24-08-26']]
# pairs = [['24-02-21', '24-05-29'], ['24-05-29', '24-06-08'], ['24-06-08' , '24-08-26']]

# pairedPairs = [['doxy', 'doxy_act'], ['doxy_2_Y27_10', 'doxy_2_Y27_10_act']]


plotChars = {'color' : '#ffffff', 'fontsize' : 25}
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
palette_cond = ['#808080', '#ffdb19', '#b29600', '#99c3cf', '#4d96ab' ,'#005f79']
palette_cond = ['#808080', '#ffdb19', '#b29600', '#99c3cf', '#4d96ab' ,'#005f79']


swarmPointSize = 10


#%%%% Plot NLImod

palette_crosslink = ['#808080', '#faea93', '#dfc644'] #, '#add2c3', '#5aa688', '#23644a']
# palette_crosslink = ['#faea93', '#dfc644', '#5aa688', '#23644a']

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                                    hueType = None, palette = palette_crosslink,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(-3,3)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLImodPLot.png').format(str(dates), str(condCat)))
plt.show()

######## vs. Compressions #########

# fig, ax = plt.subplots(figsize = (13,9))

# condition = 'M1'
# df_comp = df[df.manip == condition]

# plottingParams = {'data':df_comp, 
#                   'x' : 'compNum', 
#                   'y' : 'NLI_mod',
#                   'linewidth' : 1, 
#                   'size' :swarmPointSize, 
#                    }

# fig, ax = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), pairs = pairs, hueType = None,
#                                     labels = [], plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(-3, 3)
# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.savefig((dirToSave + '(1b)_{:}_{:}_{:}_NLImodPLot-Comps.png').format(str(dates), condition, str(condCat)))

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
                             hueType = None, palette = palette_crosslink,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1c)_{:}_{:}_NLImodPLot_cellAverage.png').format(str(dates), str(condCat)))
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
                             hueType = None, palette = palette_crosslink,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1d)_{:}_{:}_NLImodPLot_pairedCellAvg.png').format(str(dates), str(condCat)))
plt.show()

#%%%% Plot NLI

# plt.style.use('dark_background')

# fig, ax = plt.subplots(figsize = (13,9))
# fig, ax, pvals = pf.plotNLI(fig, ax, df, condCat, condCol,
#                             pairs = pairs, labels = labels, **plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.xticks(rotation = 45)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(1b)_{:}_{:}_NLIPLot.png').format(str(dates), str(condCat)))

# plt.style.use('default')


####### vs. Compressions #########
# condition = 'doxy'
# df_comp = df[df.drug == condition]

# plt.style.use('dark_background')

# fig, ax = plt.subplots(figsize = (13,9))
# fig, ax, pvals = pf.plotNLI(fig, ax, df_comp, condCat =  np.sort(df_comp.compNum.unique()), condCol = 'compNum',
#                             pairs = None, labels = [], **plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.tight_layout()
# plt.show()
# plt.savefig((dirToSave + '(1b)_{:}_{:}_{:}_NLIPLot-Comps.png').format(str(dates), condition, str(condCat)))

# plt.style.use('default')

#%%%% E vs H0
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0(fig, ax, df, condCat, condCol, hueType = 'NLI_Plot')
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_EvH_NLI.png').format(str(dates), str(condCat)))


fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0(fig, ax, df, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0(fig, ax, df, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2c)_{:}_{:}_EvH_CellID.png').format(str(dates), str(condCat)))

#%%% Box plots - H0

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : swarmPointSize, 
                   }

######################## Hue type 'CellID'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.legend(fontsize = 6, ncol = 6)
# plt.ylim(0,2500)
plt.yscale('log')
y_ticks = [100, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(3a)_{:}_{:}_H0Boxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(0,2500)
y_ticks = [100, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
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
plt.ylim(0,2500)
y_ticks = [100, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(3c)_{:}_{:}_H0Boxplot_NLI.png').format(str(dates), str(condCat)))


####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('H0_vwc_Full', 'median'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_crosslink,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(0,2000)
y_ticks = [100, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.savefig((dirToSave + '(3d)_{:}_{:}_H0Boxplot_cellAverage.png').format(str(dates), str(condCat)))
plt.show()

#%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff',
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
plt.yscale('log')
plt.ylim(100, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')

plt.ylim(50, 60000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : swarmPointSize,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(100, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)


plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

######################## Paired 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : swarmPointSize,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1, 100000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)


plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'median'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_crosslink,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1000, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_cellAverage.png').format(str(dates), str(condCat)))
plt.show()

#%%% Pointplots 

dfPairs, pairedCells = pf.dfCellPairs(avgDf)
condCatPoint = dfPairs[condCol, 'first'].unique()

N_point = len(dfPairs['dateCell', 'first'].unique())
palette_cell_point = distinctipy.get_colors(N_point)

stats = 'mean'
plottingParams = { 'x' : (condCol, 'first'), 
                  'y' : ('H0_vwc_Full', stats),
                  'linewidth' : 1, 
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

ylim = 1500
fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
# plt.xlim((-2,3))
plt.tight_layout()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_H0Pointplot.png').format(str(dates), str(condCat), stats))
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,3), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_H0Pointplot-Normalised.png').format(str(dates), str(condCat), stats))
plt.show()


ylim = 15000
plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('E_eff', stats),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_EPointplot.png').format(str(dates), str(condCat), stats))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,6), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_EPointplot-Normalised.png').format(str(dates), str(condCat), stats))


#%%% Plotting K vs Y and K*, Y*

fig, ax = plt.subplots(figsize = (13,9))
df['cellID_comp'] = df['cellID'] + '_' + df['compNum'].astype(str)

df_y = pd.DataFrame(df, columns = ['cellID_comp', 'Y_vwc_Full'])
df_y_mod = pd.DataFrame(df, columns = ['cellID_comp', 'Y_NLImod'])
df_y.rename(columns={'Y_vwc_Full': 'Y'}, inplace=True)
df_y_mod.rename(columns={'Y_NLImod': 'Y'}, inplace=True)
df_k = pd.DataFrame(df, columns = ['cellID_comp', 'K_vwc_Full'])
df_k_mod = pd.DataFrame(df, columns = ['cellID_comp', 'K_NLImod'])
df_k.rename(columns={'K_vwc_Full': 'K'}, inplace=True)
df_k_mod.rename(columns={'K_NLImod': 'K'}, inplace=True)
df_y = pd.concat([df_y, df_y_mod])
df_k = pd.concat([df_k, df_k_mod])
dfToPlot = df_y.merge(df_k, on = 'cellID_comp')

palette_cell_mod = distinctipy.get_colors(len(dfToPlot['cellID_comp'].unique()))
sns.scatterplot(y = 'K', x = 'Y', data = dfToPlot, hue = 'cellID_comp',  palette = palette_cell_mod)
sns.lineplot(y = 'K', x = 'Y', data = dfToPlot, hue = 'cellID_comp',  palette = palette_cell_mod)

ax.axline((1, 1), slope=1, color='C0', label='by slope')

ax.get_legend().remove()
plt.yscale('log')
plt.xscale('log')

#%% Calling data - 24-09-05

filename = 'VWC_24-09-05_24-09-09'
GlobalTable = taka.getMergedTable(filename)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/Crosslinkers/24-09-05_Mechanics/'

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)

dates = ['24-09-05']
drugs = [ 'none', 'doxy', 'doxy_act'] 
labels = ['Control', 'Dox', 'Dox + Light']
manips = ['M0', 'M1', 'M3']

thinCells = ['P3_C7', 'P1_C8', 'P3_C2', 'P2_C3', 'P2_C4', 'P1_C6', 'P2_C7', 'P1_C7', 'P3_C5',
             'P3_C4', 'P3_C6']
# 
# transCells = ['P1_C9', 'P1_C10', 'P3_C8', 'P3_C11', 'P3_C5']

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1500),
            # (data['E_eff'] <= 30000),
            (data['compNum'] <= 6),
            (data['date'].apply(lambda x : x in dates)),
            # (data['manip'].apply(lambda x : x in manips)),
            # (data['cellCode'].apply(lambda x : x in thinCells)),
            (data['drug'].apply(lambda x : x in drugs)),
            ]

df = pf.filterDf(Filters, data)
condCol, condCat = 'drug', drugs
avgDf = pf.createAvgDf(df, condCol)

# df = df.drop(df[(df['drug'] == 'doxy_act') & (df['compNum'] == 1)].index)

pairs = [['none', 'doxy'], ['doxy', 'doxy_act']] 


plotChars = {'color' : '#ffffff', 'fontsize' : 25}
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
palette_cond = ['#808080', '#ffdb19', '#b29600', '#99c3cf', '#4d96ab' ,'#005f79']

swarmPointSize = 10

#%%%% Plot NLImod

palette_crosslink = ['#808080', '#faea93', '#dfc644'] #, '#add2c3', '#5aa688', '#23644a']
# palette_crosslink = ['#faea93', '#dfc644', '#5aa688', '#23644a']

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                                    hueType = None, palette = palette_crosslink,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(-3,3)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLImodPLot.png').format(str(dates), str(condCat)))
plt.show()


##### cell coloured ######
fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                                    hueType = 'cellID', palette = palette_cell,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(-3,3)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLImodPLot_cellID.png').format(str(dates), str(condCat)))
plt.show()


######## vs. Compressions #########

fig, ax = plt.subplots(figsize = (13,9))

condition = 'none'
df_comp = df[df[condCol] == condition]

plottingParams = {'data':df_comp, 
                  'x' : 'compNum', 
                  'y' : 'NLI_mod',
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), pairs = None, hueType = None,
                                    labels = [], plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1b)_{:}_{:}_{:}_NLImodPLot-Comps.png').format(str(dates), condition, str(condCat)))


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
                             hueType = None, palette = palette_crosslink,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1c)_{:}_{:}_NLImodPLot_cellAverage.png').format(str(dates), str(condCat)))
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
                             hueType = None, palette = palette_crosslink,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1d)_{:}_{:}_NLImodPLot_pairedCellAvg.png').format(str(dates), str(condCat)))
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

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                                    hueType = 'compNum', labels = [], plottingParams = plottingParams,
                                    plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1e)_{:}_{:}_{:}_NLImodPLot-DiffComps.png').format(str(dates), condition, str(condCat)))


#%%%% NLImod vs Compression
fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : 'compNum', 
                  'y' : 'NLI_mod',   
                  'linewidth' : 1, 
                  }
                    
ax = sns.lineplot(**plottingParams)
plt.ylim(-1, 2)

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
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0(fig, ax, df, condCat, condCol, hueType = 'NLI_Plot')
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_EvH_NLI.png').format(str(dates), str(condCat)))


fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0(fig, ax, df, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0(fig, ax, df, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2c)_{:}_{:}_EvH_CellID.png').format(str(dates), str(condCat)))

#%%% Box plots - H0

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : swarmPointSize, 
                   }

######################## Hue type 'CellID'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.legend(fontsize = 6, ncol = 6)
# plt.ylim(0,2500)
plt.yscale('log')
y_ticks = [100, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(3a)_{:}_{:}_H0Boxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(0,2500)
y_ticks = [100, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
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
plt.ylim(0,2500)
y_ticks = [100, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(3c)_{:}_{:}_H0Boxplot_NLI.png').format(str(dates), str(condCat)))


####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('H0_vwc_Full', 'median'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_crosslink,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(0,2000)
y_ticks = [100, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.savefig((dirToSave + '(3d)_{:}_{:}_H0Boxplot_cellAverage.png').format(str(dates), str(condCat)))
plt.show()

#%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff',
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
# plt.yscale('log')
# plt.ylim(100, 30000)
# y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')

plt.ylim(50, 60000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : swarmPointSize,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(100, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)


plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

######################## Paired 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : swarmPointSize,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1, 100000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)


plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'median'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_crosslink,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1000, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_cellAverage.png').format(str(dates), str(condCat)))
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
plt.yscale('log')
plt.ylim(100, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(4a)_{:}_{:}_ctFieldThickness_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')

plt.ylim(50, 60000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

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

stats = 'mean'
plottingParams = { 'x' : (condCol, 'first'), 
                  'y' : ('H0_vwc_Full', stats),
                  'linewidth' : 1, 
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

ylim = 1500
fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.xlim((-2,3))
plt.tight_layout()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_H0Pointplot.png').format(str(dates), str(condCat), stats))
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,3), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7b)_{:}_{:}_{:}_H0Pointplot-Normalised.png').format(str(dates), str(condCat), stats))
plt.show()


ylim = 10000
plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('E_eff', stats),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(8a)_{:}_{:}_{:}_EPointplot.png').format(str(dates), str(condCat), stats))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,6), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(8b)_{:}_{:}_{:}_EPointplot-Normalised.png').format(str(dates), str(condCat), stats))



plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', stats),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-3,3), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(9a)_{:}_{:}_{:}_NLImodPointplot.png').format(str(dates), str(condCat), stats))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-6,6), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(9b)_{:}_{:}_{:}_NLImodPointplot-Normalised.png').format(str(dates), str(condCat), stats))


#%% Calling data - 24-09-12

filename = 'VWC_24-09-12_24-09-13'
GlobalTable = taka.getMergedTable(filename)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/Crosslinkers/24-09-05_Mechanics/'

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)

dates = ['24-09-12']
drugs = [ 'doxy', 'doxy_act', 'doxy_2_Y27_10', 'doxy_2_Y27_10_act']
labels = []

excludedCells = ['24-09-12_M1_P1_C4']

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['UI_Valid'] == True),
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1500),
            # (data['E_eff'] <= 30000),
            (data['compNum'] <= 6),
            (data['date'].apply(lambda x : x in dates)),
            (data['cellID'].apply(lambda x : x not in excludedCells)),
            (data['drug'].apply(lambda x : x in drugs)),
            ]

df = pf.filterDf(Filters, data)
condCol, condCat = 'drug', drugs
avgDf = pf.createAvgDf(df, condCol)

# df = df.drop(df[(df['drug'] == 'doxy_act') & (df['compNum'] == 1)].index)

pairs = [ ['doxy', 'doxy_act'], ['doxy_2_Y27_10', 'doxy_2_Y27_10_act']] 
# pairs = [ ['doxy', 'doxy_act']] 


plotChars = {'color' : '#ffffff', 'fontsize' : 25}
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
palette_cond = ['#808080', '#ffdb19', '#b29600', '#99c3cf', '#4d96ab' ,'#005f79']

swarmPointSize = 10

#%%%% Plot NLImod

palette_crosslink = ['#808080', '#ffdb19', '#b29600', '#99c3cf'] #, '#4d96ab' ,'#005f79']

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                                    hueType = None, palette = palette_crosslink,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(-3,3)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1a)_{:}_{:}_NLImodPLot.png').format(str(dates), str(condCat)))
plt.show()

######## vs. Compressions #########

fig, ax = plt.subplots(figsize = (13,9))

condition = 'doxy'
df_comp = df[df[condCol] == condition]

plottingParams = {'data':df_comp, 
                  'x' : 'compNum', 
                  'y' : 'NLI_mod',
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                   }

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), pairs = None, hueType = None,
                                    labels = [], plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1b)_{:}_{:}_{:}_NLImodPLot-Comps.png').format(str(dates), condition, str(condCat)))


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
                             hueType = None, palette = palette_crosslink,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1c)_{:}_{:}_NLImodPLot_cellAverage.png').format(str(dates), str(condCat)))
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
                             hueType = None, palette = palette_crosslink,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3,3.5)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1d)_{:}_{:}_NLImodPLot_pairedCellAvg.png').format(str(dates), str(condCat)))
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

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                                    hueType = 'compNum', labels = [], plottingParams = plottingParams,
                                    plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1e)_{:}_{:}_{:}_NLImodPLot-DiffComps.png').format(str(dates), condition, str(condCat)))


#%%%% NLImod vs Compression
fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : 'compNum', 
                  'y' : 'NLI_mod',   
                  'linewidth' : 1, 
                  }
                    
ax = sns.lineplot(**plottingParams)
plt.ylim(-1, 2)

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
fig, ax = pf.EvsH0(fig, ax, df, condCat, condCol, hueType = 'NLI_Plot')
fig.suptitle(str(dates), **plotChars)
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_EvH_NLI.png').format(str(dates), str(condCat)))


fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0(fig, ax, df, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.EvsH0(fig, ax, df, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2c)_{:}_{:}_EvH_CellID.png').format(str(dates), str(condCat)))

#%%% Box plots - H0

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : swarmPointSize, 
                   }

######################## Hue type 'CellID'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.legend(fontsize = 6, ncol = 6)
# plt.ylim(0,2500)
plt.yscale('log')
y_ticks = [100, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(3a)_{:}_{:}_H0Boxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(0,2500)
y_ticks = [100, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
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
plt.ylim(0,2500)
y_ticks = [100, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(3c)_{:}_{:}_H0Boxplot_NLI.png').format(str(dates), str(condCat)))


####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('H0_vwc_Full', 'median'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_crosslink,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(0,2000)
y_ticks = [100, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.savefig((dirToSave + '(3d)_{:}_{:}_H0Boxplot_cellAverage.png').format(str(dates), str(condCat)))
plt.show()

#%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff',
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
# plt.yscale('log')
# plt.ylim(100, 30000)
# y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')

plt.ylim(50, 60000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : swarmPointSize,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(100, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)


plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

######################## Paired 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : swarmPointSize,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1, 100000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)


plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'median'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_crosslink,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1000, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(4d)_{:}_{:}_EBoxplot_cellAverage.png').format(str(dates), str(condCat)))
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
plt.yscale('log')
plt.ylim(100, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(4a)_{:}_{:}_ctFieldThickness_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')

plt.ylim(50, 60000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

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

stats = 'mean'
plottingParams = { 'x' : (condCol, 'first'), 
                  'y' : ('H0_vwc_Full', stats),
                  'linewidth' : 1, 
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }

ylim = 1500
fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
# plt.xlim((-2,3))
plt.tight_layout()
plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_H0Pointplot.png').format(str(dates), str(condCat), stats))
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,3), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

# fig.suptitle(str(dates), **plotChars)
plt.savefig((dirToSave + '(7b)_{:}_{:}_{:}_H0Pointplot-Normalised.png').format(str(dates), str(condCat), stats))
plt.show()


ylim = 10000
plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('E_eff', stats),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(8a)_{:}_{:}_{:}_EPointplot.png').format(str(dates), str(condCat), stats))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,6), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(8b)_{:}_{:}_{:}_EPointplot-Normalised.png').format(str(dates), str(condCat), stats))



plottingParams = {'x' : (condCol, 'first'), 
                  'y' : ('NLI_mod', stats),
                  'linewidth' : 1,
                  'markersize' : 10,
                  'markeredgecolor':'black', 
                   }


fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-3,3), 
                                          pairs = pairs, normalize = False, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(9a)_{:}_{:}_{:}_NLImodPointplot.png').format(str(dates), str(condCat), stats))

fig, ax = plt.subplots(figsize = (10,10))
fig, ax, pvals = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-6,6), 
                                          pairs = pairs, normalize = True, marker = stats,
                                          test = 'less', plottingParams = plottingParams,  palette = palette_cell_point,
                                          plotChars = plotChars)


plt.show()
plt.savefig((dirToSave + '(9b)_{:}_{:}_{:}_NLImodPointplot-Normalised.png').format(str(dates), str(condCat), stats))



#%% Calling data - All crosslinker experimets - 24-09-13
filename = 'VWC_AllCrosslinking_24-09-13'

GlobalTable = taka.getMergedTable(filename)
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/Projects/Crosslinkers/CrosslinkerMechanicsSummary/'

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)

dates = ['24-02-21', '24-05-29', '24-09-05', '24-09-12']
# drugs = ['none', 'doxy', 'doxy_act', 'Y27_10', 'doxy_2_Y27_10', 'doxy_2_Y27_10_act']
# labels = ['Control', 'Dox', 'Dox+Light', 'Y27 10μM', 'Y27+Dox', 'Y27+Dox+Light']
# manips = ['M0', 'M1', 'M3']

# drugs = ['Y27_10', 'doxy_2_Y27_10', 'doxy_2_Y27_10_act']# ['doxy', 'doxy_act'] #, 'doxy_2_Y27_10', 'doxy_2_Y27_10_act']
# labels = ['Y27 10μM', 'Y27+Dox', 'Y27+Dox+Light'] #, 'Y27+Dox', 'Y27+Dox+Light']

drugs = [ 'doxy_2_Y27_10', 'doxy_2_Y27_10_act']#[ 'doxy', 'doxy_act'] #, 'doxy_2_Y27_10', 'doxy_2_Y27_10_act']
labels = ['Y27+Dox', 'Y27+Dox+Light']  #['Y27+Dox', 'Y27+Dox+Light'] [ 'Dox', 'Dox+Light']  

# drugs = [ 'doxy_2_Y27_10', 'doxy_2_Y27_10_act']
# labels = [ 'Y27+Dox', 'Y27+Dox+Light'] #, 'Y27+Dox', 'Y27+Dox+Light']

E_Cells_Dox = ['24-02-21_P1_C1', '24-02-21_P1_C3', '24-02-21_P1_C4',
       '24-02-21_P1_C6', '24-02-21_P2_C1', '24-02-21_P2_C10',
       '24-02-21_P2_C11', '24-02-21_P2_C2', '24-02-21_P2_C4',
       '24-02-21_P2_C5', '24-02-21_P2_C6', '24-02-21_P2_C7',
       '24-05-29_P1_C10', '24-05-29_P1_C2', '24-05-29_P1_C3',
       '24-05-29_P1_C4', '24-05-29_P1_C5', '24-05-29_P1_C6',
       '24-05-29_P1_C8', '24-05-29_P1_C9', '24-09-05_P1_C1',
       '24-09-05_P1_C3', '24-09-05_P1_C4', '24-09-05_P1_C6',
       '24-09-05_P1_C7', '24-09-05_P1_C8', '24-09-05_P1_C9',
       '24-09-05_P2_C10', '24-09-05_P2_C3', '24-09-05_P2_C4',
       '24-09-05_P2_C6', '24-09-05_P2_C7', '24-09-05_P2_C8',
       '24-09-05_P3_C1', '24-09-05_P3_C10', '24-09-05_P3_C4',
       '24-09-05_P3_C5', '24-09-05_P3_C6', '24-09-05_P3_C9',
       '24-09-12_P1_C2', '24-09-12_P1_C3', '24-09-12_P1_C6',
       '24-09-12_P4_C2', '24-09-12_P4_C3']

# E_Cells_Dox = np.random.choice(E_Cells_Dox, 23, replace = True)

E_Cells_Dox_y27  = ['24-05-29_P2_C1', '24-05-29_P2_C10', '24-05-29_P2_C2',
       '24-05-29_P2_C3', '24-05-29_P2_C4', '24-05-29_P2_C5',
       '24-05-29_P2_C6', '24-05-29_P2_C9', '24-09-12_P2_C1',
       '24-09-12_P2_C2', '24-09-12_P2_C3', '24-09-12_P2_C4',
       '24-09-12_P2_C5', '24-09-12_P2_C6', '24-09-12_P2_C7',
       '24-09-12_P2_C8', '24-09-12_P2_C9', '24-09-12_P3_C1',
       '24-09-12_P3_C2', '24-09-12_P3_C4', '24-09-12_P3_C6',
       '24-09-12_P3_C7', '24-09-12_P3_C8']


excludedCond = ['24-09-05_M4', '24-09-05_M6', '24-09-05_M5', '24-09-05_M7']

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1500),
            (data['E_eff'] <= 30000),
            (data['compNum'] <= 6),
            # (data['dateCell'].apply(lambda x : x in E_Cells_Dox)),
            # (data['manip'].apply(lambda x : x in manips)),
            (data['manipID'].apply(lambda x : x not in excludedCond)),
            (data['date'].apply(lambda x : x in dates)),
            (data['drug'].apply(lambda x : x in drugs)),
            ]

df = pf.filterDf(Filters, data)
condCol, condCat = 'drug', drugs
avgDf = pf.createAvgDf(df, condCol)

# df = df.drop(df[(df['drug'] == 'doxy_act') & (df['compNum'] == 1)].index)

# pairs = [['24-02-21', '24-05-29'], ['24-05-29', '24-09-05'], ['24-02-21', '24-09-05']] 
pairs = [['doxy', 'doxy_act']] 
# pairs = [['doxy_2_Y27_10', 'doxy_2_Y27_10_act']]
# pairs = [['none', 'doxy'], ['doxy', 'doxy_act'], ['none', 'Y27_10'], ['Y27_10','doxy_2_Y27_10'],  ['doxy_2_Y27_10', 'doxy_2_Y27_10_act']]
# pairs = [ ['doxy', 'doxy_act'], ['doxy_2_Y27_10', 'doxy_2_Y27_10_act']] 


plotChars = {'color' : '#ffffff', 'fontsize' : 18}
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
# palette_cond = ['#808080', '#f8ae85', '#f47835', '#d1e6a8', '#8ec127', '#557317']
palette_cond = ['#8ec127', '#557317']

# palette_crosslink = ['#808080', '#faea93', '#dfc644'] #, '#add2c3', '#5aa688', '#23644a']
# palette_cond = [ '#d1e6a8', '#8ec127', '#557317']


swarmPointSize = 10


#%%%% Plot NLImod


fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                                    hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(-3,3)
fig.suptitle(str(dates), **plotChars)
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

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), pairs = None, hueType = None,
                                    labels = [], plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1b)_{:}_{:}_{:}_NLImodPLot-Comps.png').format(str(dates), condition, str(condCat)))


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
plt.savefig((dirToSave + '(1c)_{:}_{:}_NLImodPLot_cellAverage.png').format(str(dates), str(condCat)))
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
plt.savefig((dirToSave + '(1d)_{:}_{:}_NLImodPLot_pairedCellAvg.png').format(str(dates), str(condCat)))
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

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
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

plt.ylim(-3, 3)
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

#%%%% E vs H0 weighted average
 
fig, axes = plt.subplots(1, 2, figsize = (18,10))
fig, axes, avgDf = pf.EvH0_wCellAvg(fig, axes,  avgDf, condCat, condCol, hueType = 'NLI_Plot')
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_wAvgEvH_NLI.png').format(str(dates), str(condCat)))


fig, axes = plt.subplots(1, 2, figsize = (18,10))
fig, axes, avgDf = pf.EvH0_wCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_wAvgEvH_Conditions.png').format(str(dates), str(condCat)))

# fig, axes = plt.subplots(1, 2, figsize = (18,10))
# fig, axes = pf.EvH0_wCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
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
                  'y' : ('E_norm', 'wAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond, labels = labels,
                             plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1000, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(2d)_{:}_{:}_E-NormBoxplot_Trial10.png').format(str(dates), str(condCat)))
plt.show()



#%%% Box plots - H0

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : swarmPointSize, 
                   }

######################## Hue type 'CellID'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.legend(fontsize = 6, ncol = 6)
# plt.ylim(0,2500)
plt.yscale('log')
y_ticks = [100, 250, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(3a)_{:}_{:}_H0Boxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(0,2500)
y_ticks = [100 ,250, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
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
plt.ylim(0,2500)
y_ticks = [100, 250, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(3c)_{:}_{:}_H0Boxplot_NLI.png').format(str(dates), str(condCat)))

####################### H0 vs compression ############################

fig, ax = plt.subplots(figsize = (13,9))
condition = 'doxy_act'
df_comp = df[df['drug'] == condition]

plottingParams = {'data':df_comp, 
                  'x' : 'compNum', 
                  'y' : 'H0_vwc_Full',
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
                  'y' : ('H0_vwc_Full', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(0,2000)
y_ticks = [100, 250, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.savefig((dirToSave + '(3e)_{:}_{:}_H0Boxplot_meancellAverage.png').format(str(dates), str(condCat)))
plt.show()



#%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff',
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
# plt.yscale('log')
# plt.ylim(100, 30000)
# y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')

plt.ylim(50, 50000)
y_ticks = [100, 500, 5000, 10000, 50000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : swarmPointSize,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(100, 50000)
y_ticks = [100, 500, 5000, 10000, 50000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)


plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

######################## Paired 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : swarmPointSize,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1, 100000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)


plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'median'),
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

#%%% Boxplots - E_Normalised

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff',
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
# plt.yscale('log')
# plt.ylim(100, 30000)
# y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')

plt.ylim(50, 60000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : swarmPointSize,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(100, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)


plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

######################## Paired 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : swarmPointSize,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1, 100000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)


plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'median'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1000, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

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
plt.ylim(1000, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
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

plt.ylim(0,1500)
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
testNli = 'two-sided'

# stats = 'mean'
# plottingParams = { 'x' : (condCol, 'first'), 
#                   'y' : ('H0_vwc_Full', stats),
#                   'linewidth' : 1, 
#                   'markersize' : 10,
#                   'markeredgecolor':'black', 
#                    }

# ylim = 1500
# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
#                                           pairs = pairs, normalize = False, marker = stats,
#                                           test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)

# # fig.suptitle(str(dates), **plotChars)
# # plt.xlim((-2,3))
# plt.tight_layout()
# plt.savefig((dirToSave + '(7a)_{:}_{:}_{:}_H0Pointplot.png').format(str(dates), str(condCat), stats))
# plt.show()

# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP_H = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,3), 
#                                           pairs = pairs, normalize = True, marker = stats,
#                                           test = testH0, plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)

# # fig.suptitle(str(dates), **plotChars)
# plt.savefig((dirToSave + '(7b)_{:}_{:}_{:}_H0Pointplot-Normalised.png').format(str(dates), str(condCat), stats))
# plt.show()


ylim = 10000
# plottingParams = {'x' : (condCol, 'first'), 
#                   'y' : ('E_eff', 'wAvg'),
#                   'linewidth' : 1,
#                   'markersize' : 10,
#                   'markeredgecolor':'black', 
#                    }


# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,ylim), 
#                                           pairs = pairs, normalize = False, marker = 'wAvg',
#                                           test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)

# plt.show()
# plt.savefig((dirToSave + '(8a)_{:}_{:}_{:}_EPointplot.png').format(str(dates), str(condCat), stats))

# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP_E = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (0,4), 
#                                           pairs = pairs, normalize = True, marker = 'wAvg',
#                                           test = testE, plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)


# plt.show()
# plt.savefig((dirToSave + '(8b)_{:}_{:}_{:}_EPointplot-Normalised.png').format(str(dates), str(condCat), stats))

# nonOutliers_E = np.asarray(dfP_E['dateCell', 'first'][(dfP_E[condCol, 'first'] == condCat[1]) & (dfP_E['normMeasure', 'wAvg'] < 3.0)].values)

# plottingParams = {'x' : (condCol, 'first'), 
#                   'y' : ('NLI_mod', stats),
#                   'linewidth' : 1,
#                   'markersize' : 10,
#                   'markeredgecolor':'black', 
#                    }


# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-3,3), 
#                                           pairs = pairs, normalize = False, marker = stats,
#                                           test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)

# plt.show()
# plt.savefig((dirToSave + '(9a)_{:}_{:}_{:}_NLImodPointplot.png').format(str(dates), str(condCat), stats))

# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-3,3), styleType = ('dateCell', 'first'),
#                                           pairs = pairs, normalize = False, marker = stats, hueType = ('date', 'first'),
#                                           test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)

# plt.show()
# plt.savefig((dirToSave + '(9b)_{:}_{:}_{:}_NLImodPointplot_dates.png').format(str(dates), str(condCat), stats))


# fig, ax = plt.subplots(figsize = (10,10))
# fig, ax, pvals, dfP = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-1.5,1.5), 
#                                           pairs = pairs, normalize = False, marker = stats, hueType = ('date', 'first'),
#                                           test = testNli, plottingParams = plottingParams,  palette = palette_cell_point,
#                                           plotChars = plotChars)

# plt.show()
# plt.savefig((dirToSave + '(9c)_{:}_{:}_{:}_NLImodPointplot_datesAvg.png').format(str(dates), str(condCat), stats))


# # fig, ax = plt.subplots(figsize = (10,10))
# # fig, ax, pvals, dfP_nli = pf.pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, ylim = (-6,6), 
# #                                           pairs = pairs, normalize = True, marker = stats,
# #                                           test = 'greater', plottingParams = plottingParams,  palette = palette_cell_point,
# #                                           plotChars = plotChars)


# plt.show()
# plt.savefig((dirToSave + '(9b)_{:}_{:}_{:}_NLImodPointplot-Normalised.png').format(str(dates), str(condCat), stats))

#### E-normalised 
trial = 1

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
plt.savefig((dirToSave + '(10a)_{:}_{:}_E-norm_t{:}.png').format(str(dates), str(condCat)))

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
                  's' : 100
                   }

fig, ax = pf.NLIvFluctu(fig, ax, plottingParams = plottingParams, plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(11a)_{:}_{:}_NLIvFluctu.png').format(str(dates), str(condCat)))

#%%%  NLI-corr vs Fluctuation

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'palette' : palette_cond,
                  's' : 100,
                  'hue' : condCol
                   }

fig, ax, data_nli = pf.NLIcorrvFluctu(fig, ax, data = df, plottingParams = plottingParams, 
                                      plotChars = plotChars)


plt.xlim(0, 1.5)
plt.show()
plt.savefig((dirToSave + '(11b)_{:}_{:}_NLI-corrvFluctu.png').format(str(dates), str(condCat)))

#%% Calling data - Similar results experiments (All replicates, including Y27)
# Dates avilable : '24-05-29 & 24-09-05 & 24-02-21 & 24-09-24 & 24-09-12'

filename = 'VWC_AllCrosslinking-GoodExpts_24-09-27'

GlobalTable = taka.getMergedTable(filename)
dirToSave = 'C:/Users/anumi/OneDrive/Desktop/CortexMeeting_24-10-01/Crosslinkers/Mechanics/24-09-31/PosExpts_Sept/'

#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)
# '24-02-21', '24-05-29',
dates = [ '24-09-05', '24-09-12', '24-09-24']
drugs = [ 'none', 'doxy', 'doxy_act', 'Y27_10', 'doxy_2_Y27_10', 'doxy_2_Y27_10_act']
labels = [ 'Control','Dox', 'Dox+Light', 'Y27_10', 'Y27+Dox', 'Y27+Dox+Light']

# drugs = ['Y27_10', 'doxy_2_Y27_10', 'doxy_2_Y27_10_act']# ['doxy', 'doxy_act'] #,  ['doxy_2_Y27_10', 'doxy_2_Y27_10_act']
# labels = ['Y27', 'Y27+Dox', 'Y27+Dox+Light'] #['Dox', 'Dox+Light'] #,  ['Y27+Dox', 'Y27+Dox+Light']

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1500),
            (data['E_eff'] <= 30000),
            (data['compNum'] <= 6),
            (data['date'].apply(lambda x : x in dates)),
            (data['drug'].apply(lambda x : x in drugs)),
            ]

df = pf.filterDf(Filters, data)
condCol, condCat = 'drug', drugs
avgDf = pf.createAvgDf(df, condCol)

# df = df.drop(df[(df['drug'] == 'doxy_act') & (df['compNum'] == 1)].index)

pairs = [['none', 'doxy'], ['doxy', 'doxy_act'], ['none', 'Y27_10'], ['doxy_2_Y27_10', 'doxy_2_Y27_10_act']] 
# pairs = [['doxy', 'doxy_act']] 


plotChars = {'color' : '#ffffff', 'fontsize' : 18}
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
palette_cond = ['#808080', '#faea93', '#dfc644', '#add2c3', '#5aa688', '#23644a']

# palette_cond = ['#add2c3', '#5aa688', '#23644a']

swarmPointSize = 6


#%%%% Plot NLImod
fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                    }

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                                    hueType = None, palette = palette_cond, plotType = 'violin',
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(-3,3)
fig.suptitle(str(dates), **plotChars)
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

#%% Calling data - All 'okaish' experiments, wherein results were not so strong
# Dates available : '24-08-26 & 24-06-07 & 24-06-08 & 24-07-15'

filename = 'VWC_AllCrosslinking-OkExpts_24-09-31'

GlobalTable = taka.getMergedTable(filename)
dirToSave = 'C:/Users/anumi/OneDrive/Desktop/CortexMeeting_24-10-01/Crosslinkers/Mechanics/24-09-31/OkExptsPlots/'


#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)

dates = ['24-08-26', '24-06-07' , '24-06-08' , '24-07-15']
drugs = ['none', 'doxy', 'doxy_act', 'Y27_10', 'doxy_2_Y27_10', 'doxy_2_Y27_10_act']
labels = ['Control', 'Dox', 'Dox+Light', 'Y27_10', 'Y27+Dox', 'Y27+Dox+Light']

Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1500),
            (data['E_eff'] <= 30000),
            ((data['compNum'] <= 6)),
            (data['date'].apply(lambda x : x in dates)),
            (data['drug'].apply(lambda x : x in drugs)),
            ]

df = pf.filterDf(Filters, data)
condCol, condCat = 'drug', drugs
avgDf = pf.createAvgDf(df, condCol)

# df = df.drop(df[(df['drug'] == 'doxy_act') & (df['compNum'] == 1)].index)

pairs = [['none', 'doxy'], ['doxy', 'doxy_act'], ['none', 'Y27_10'], ['doxy_2_Y27_10', 'doxy_2_Y27_10_act']] 


plotChars = {'color' : '#ffffff', 'fontsize' : 18}
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
palette_cond = ['#808080', '#faea93', '#dfc644', '#add2c3', '#5aa688', '#23644a']



swarmPointSize = 6


#%%%% Plot NLImod

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'NLI_mod',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                                    hueType = None, palette = palette_cond, plotType = 'violin',
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-5,5)
fig.suptitle(str(dates), **plotChars)
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

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = np.sort(df_comp.compNum.unique()), pairs = None, hueType = None,
                                    labels = [], plottingParams = plottingParams, plotChars = plotChars)

plt.ylim(-3, 3)
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.savefig((dirToSave + '(1b)_{:}_{:}_{:}_NLImodPLot-Comp.png').format(str(dates), condition, str(condCat)))


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

fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
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

plt.ylim(-3, 3)
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

#%%%% E vs H0 weighted average
 
fig, axes = plt.subplots(1, 2, figsize = (18,10))
fig, axes, avgDf = pf.EvH0_wCellAvg(fig, axes,  avgDf, condCat, condCol, hueType = 'NLI_Plot')
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2a)_{:}_{:}_wAvgEvH_NLI.png').format(str(dates), str(condCat)))


fig, axes = plt.subplots(1, 2, figsize = (18,10))
fig, axes, avgDf = pf.EvH0_wCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_wAvgEvH_Conditions.png').format(str(dates), str(condCat)))

# fig, axes = plt.subplots(1, 2, figsize = (18,10))
# fig, axes = pf.EvH0_wCellAvg(fig, axes, avgDf, condCat, condCol,  palette = palette_cell, hueType = 'cellID')
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
                  'y' : ('E_norm', 'wAvg'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond, labels = labels,
                             plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1000, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

plt.tight_layout()
plt.savefig((dirToSave + '(2d)_{:}_{:}_E-NormBoxplot_Trial10.png').format(str(dates), str(condCat)))
plt.show()



#%%% Box plots - H0

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : swarmPointSize, 
                   }

######################## Hue type 'CellID'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.legend(fontsize = 6, ncol = 6)
# plt.ylim(0,2500)
plt.yscale('log')
y_ticks = [100, 250, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(3a)_{:}_{:}_H0Boxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)


fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(0,2500)
y_ticks = [100 ,250, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
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
plt.ylim(0,2500)
y_ticks = [100, 250, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(3c)_{:}_{:}_H0Boxplot_NLI.png').format(str(dates), str(condCat)))

####################### H0 vs compression ############################

fig, ax = plt.subplots(figsize = (13,9))
condition = 'doxy_act'
df_comp = df[df['drug'] == condition]

plottingParams = {'data':df_comp, 
                  'x' : 'compNum', 
                  'y' : 'H0_vwc_Full',
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
                  'y' : ('H0_vwc_Full', 'mean'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(0,2000)
y_ticks = [100, 250, 500, 1000, 2500]
ax.set_yticks(y_ticks, labels =y_ticks, **plotChars)
plt.savefig((dirToSave + '(3e)_{:}_{:}_H0Boxplot_meancellAverage.png').format(str(dates), str(condCat)))
plt.show()



#%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff',
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
# plt.yscale('log')
# plt.ylim(100, 30000)
# y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')

plt.ylim(50, 50000)
y_ticks = [100, 500, 5000, 10000, 50000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : swarmPointSize,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(100, 50000)
y_ticks = [100, 500, 5000, 10000, 50000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)


plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

######################## Paired 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : swarmPointSize,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1, 100000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)


plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'median'),
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

#%%% Boxplots - E_Normalised

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff',
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
# plt.yscale('log')
# plt.ylim(100, 30000)
# y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)
plt.show()
plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))

####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cond,
                             plottingParams = plottingParams, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')

plt.ylim(50, 60000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))

######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : swarmPointSize,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(100, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)


plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

######################## Paired 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                        'x' : condCol, 
                        'y' : 'E_eff',
                        'order' : condCat,
                        'linewidth' : 1, 
                        'size' : swarmPointSize,  
                          }

fig, ax = plt.subplots(figsize = (13,9))
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

# plt.ylim(0, ylim)
fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1, 100000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)


plt.show()
plt.savefig((dirToSave + '(4c)_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat)))

####################### cell average ############################

fig, ax = plt.subplots(figsize = (13,9))
dates = np.unique(df['date'].values)

plottingParams = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'median'),
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' :swarmPointSize, 
                    }

fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             hueType = None, palette = palette_cond,
                                    labels = labels, plottingParams = plottingParams, plotChars = plotChars)

fig.suptitle(str(dates), **plotChars)
plt.yscale('log')
plt.ylim(1000, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
ax.set_yticks(y_ticks, labels = y_ticks, **plotChars)

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
plt.ylim(1000, 30000)
y_ticks = [100, 500, 5000, 10000, 30000]
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

plt.ylim(0,1500)
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
                  's' : 100
                   }

fig, ax = pf.NLIvFluctu(fig, ax, plottingParams = plottingParams, plotChars = plotChars)

plt.show()
plt.savefig((dirToSave + '(11a)_{:}_{:}_NLIvFluctu.png').format(str(dates), str(condCat)))

#%%%  NLI-corr vs Fluctuation

fig, ax = plt.subplots(figsize = (13,9))

plottingParams = {'palette' : palette_cond,
                  's' : 100,
                  'hue' : condCol
                   }

fig, ax, data_nli = pf.NLIcorrvFluctu(fig, ax, data = df, plottingParams = plottingParams, 
                                      plotChars = plotChars)


plt.xlim(0, 1.5)
plt.show()
plt.savefig((dirToSave + '(11b)_{:}_{:}_NLI-corrvFluctu.png').format(str(dates), str(condCat)))
