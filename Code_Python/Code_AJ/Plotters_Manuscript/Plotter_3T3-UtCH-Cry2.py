# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 11:06:00 2025

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
from statannotations.Annotator import Annotator

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
                       'zones_H0':['%f_10', '%f_15', '%f_20'],
                       'method_bestH0':'Chadwick',
                       'zone_bestH0':'%f_15',
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

    
Task = '24-01-02'
fitsSubDir = 'VWC_3T3-UtCH-Cry2_AllExpts_Manuscript'

GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', 
                            fileName = fitsSubDir,save = True, PLOT = False, source = 'Python',
                            fitSettings = fitSettings, plotSettings = plotSettings,
                            fitsSubDir = fitsSubDir) # task = 'updateExisting'

#%% Calling data - 3T3 Reproducible Experiments

filename = 'VWC_3T3-UtCH-Cry2_AllExpts_Manuscript'
GlobalTable = taka.getMergedTable(filename)
fitsSubDir = filename
dirToSave = 'D:/Anumita/MagneticPincherData/Figures/FiguresForManuscript/3T3_UtCH-Cry2/'


#%%%% Create dataframe for plotting

data = pf.createDataTable(GlobalTable)

dates = ['24-01-02']
drugs = [ 'doxy_1', 'doxy_1_act']
labels = []


Filters = [(data['validatedThickness'] == True),
            (data['error_vwc_Full'] == False),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['R2_vwc_Full'] > 0.90),
            (data['bestH0'] <= 1200),
            (data['E_eff'] <= 30000),
            # (data['compNum'] <= 7),
            (data['date'].apply(lambda x : x in dates)),
            (data['drug'].apply(lambda x : x in drugs)),
            ]

df = pf.filterDf(Filters, data)

pairs = [ ['doxy_1', 'doxy_1_act']]

condCol, condCat = 'drug', drugs

plotChars = {'color' : 'white', 'fontsize' : 15}
plotTicks = {'color' : '#ffffff', 'fontsize' : 16}

avgDf = pf.createAvgDf(df, condCol)
avgDf = avgDf[avgDf[('compNum', 'count')] > 2]

N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
# palette_cond = ['#6A3E00', '#6A3E00', '#c5b2d2', '#8b66a5','#511978', '#8b66a5','#511978']
# palette_cond = ['#c5b2d2', '#8b66a5']
palette_cond = ['#6A3E00',  '#c5b2d2', '#8b66a5'] #,'#511978'] #, '#8b66a5','#511978']


swarmPointSize = 10

#%%%% Plot NLI
plt.style.use('dark_background')

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, pvals = pf.plotNLI(fig, ax, data, condCat, condCol, pairs, labels = labels, **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(1)_{:}_{:}_NLIPLot.png').format(str(dates), str(condCat)))

plt.style.use('default')

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


#%%% E vs H0

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax, df = pf.EvsH0_perCompression(fig, ax, df, condCat, condCol,  palette = palette_cond, hueType = condCol)
plt.legend(fontsize = 6, ncol = len(condCat))
fig.suptitle(str(dates), **plotChars)
plt.tight_layout()
plt.show()
plt.savefig((dirToSave + '(2b)_{:}_{:}_EvH_Conditions.png').format(str(dates), str(condCat)))


avgDf = avgDf = pf.createAvgDf(df, condCol, dataFluoPath = None, e_norm = True)

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
plt.ylim(0, 2000)
plt.legend(fontsize = 6, ncol = 6)
plt.show()
plt.savefig((dirToSave + '(3a)_{:}_{:}_H0Boxplot_CellID.png').format(str(dates), str(condCat)))


####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                             plottingParams = plottingParams, plotChars = plotChars)
plt.ylim(0, 2000)
plt.show()
plt.savefig((dirToSave + '(3b)_{:}_{:}_H0Boxplot_Conditions.png').format(str(dates), str(condCat)))


######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'H0_vwc_Full',
                  'order' : condCat,
                  'hue_order' : ['linear', 'intermediate', 'non-linear'],
                  'linewidth' : 1, 
                  'size' : swarmPointSize,  
                    }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

plt.ylim(0, 2000)
plt.show()
plt.savefig((dirToSave + '(3c)_{:}_{:}_H0Boxplot_NLI.png').format(str(dates), str(condCat)))


#%%% Boxplots - E_eff

plottingParams = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff',
                  'order' : condCat,
                  'linewidth' : 1, 
                  'size' : swarmPointSize, 
                   }


######################## Hue type 'CellID'#######################
N = len(df['cellID'].unique())
palette_cell = distinctipy.get_colors(N)
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)
plt.ylim(0, 40000)
plt.legend(fontsize = 6, ncol = 6)
plt.show()
plt.savefig((dirToSave + '(4a)_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat)))


####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, 
                             plottingParams = plottingParams, plotChars = plotChars)
plt.ylim(0, 40000)
plt.show()
plt.savefig((dirToSave + '(4b)_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat)))


######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':df, 
                  'x' : condCol, 
                  'y' : 'E_eff',
                  'order' : condCat,
                  'hue_order' : ['linear', 'intermediate', 'non-linear'],
                  'linewidth' : 1, 
                  'size' : 5,  
                    }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCompression(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', 
                                    labels = labels, plottingParams = plottingParams_nli, plotChars = plotChars)

plt.ylim(0, 40000)
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

######################## Hue type 'CellID'#######################
N = len(avgDf['cellID', 'first'].unique())
palette_cell = distinctipy.get_colors(N)
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, palette = palette_cell,
                                    hueType = 'cellID', plottingParams = plottingParams, plotChars = plotChars)
plt.ylim(0, 30000)
plt.legend(fontsize = 6, ncol = 6)
plt.show()
plt.savefig((dirToSave + '(5a)_{:}_{:}_{:}_EBoxplot_CellID.png').format(str(dates), str(condCat), stats))


####################### Hue type 'condCol'#######################
fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, 
                             plottingParams = plottingParams, plotChars = plotChars)
plt.ylim(0, 30000)
plt.show()
plt.savefig((dirToSave + '(5b)_{:}_{:}_{:}_EBoxplot_Conditions.png').format(str(dates), str(condCat), stats))


######################## Hue type 'NLI_Plot'#######################

plottingParams_nli  = {'data':avgDf, 
                  'x' : (condCol, 'first'), 
                  'y' : ('E_eff', 'mean'),
                  'order' : condCat,
                  'hue_order' : ['linear', 'intermediate', 'non-linear'],
                  'linewidth' : 1, 
                  'size' : swarmPointSize, 
                  'edgecolor':'k', 
                    }

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
fig, ax = pf.boxplot_perCell(fig, ax, condCat = condCat, pairs = pairs, hueType = 'NLI_Plot', labels = labels,
                              plottingParams = plottingParams_nli, plotChars = plotChars)

plt.ylim(0, 30000)
plt.show()
plt.savefig((dirToSave + '(5c)_{:}_{:}_{:}_EBoxplot_NLI.png').format(str(dates), str(condCat), stats))

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

#%%% Plotnine paired plots

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


measure = 'E_eff'
stat = 'mean'
plot, pvals = pf.pairedplot(dfPairs, condCol = condCol, condCat = condCat, measure = measure, 
                     pairs = pairs, stat = stat, test = 'greater', palette = palette_cond,
                     plotChars = plotChars, plotTicks = plotTicks, y_limits =( 0, 10000))



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

