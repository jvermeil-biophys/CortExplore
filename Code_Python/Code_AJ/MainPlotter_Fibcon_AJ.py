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
Task = '24-11-28_M3'


fitsSubDir = 'VWC_Fibcon_24-11-28'

GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', 
                            fileName = fitsSubDir, save = True, PLOT = True, source = 'Python',
                            fitSettings = fitSettings, plotSettings = plotSettings,
                            fitsSubDir = fitsSubDir) # task = 'updateExisting'

#%% Calling data - 
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

