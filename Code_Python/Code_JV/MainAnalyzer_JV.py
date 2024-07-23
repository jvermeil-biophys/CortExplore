# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:31:26 2022

@author: JosephVermeil
"""

# %% > Imports and constants

#### Main imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm



import os
import sys
import time
import random
import warnings
import itertools
import matplotlib

from copy import copy
from cycler import cycler
from datetime import date
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)


import GraphicStyles as gs
import UtilityFunctions as ufun
import TrackAnalyser as taka
import TrackAnalyser_V2 as taka2
import TrackAnalyser_V3 as taka3
import TrackAnalyser_VManuscript as takaM

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



# %% TimeSeries functions


# %%% List files
allTimeSeriesDataFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                          if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv"))]
print(allTimeSeriesDataFiles)


# %%% Get a time series

df = taka.getCellTimeSeriesData('22-02-09_M1_P1_C7')


# %%% Plot a time series

# taka.plotCellTimeSeriesData('21-02-10_M1_P1_C2')
taka.plotCellTimeSeriesData('22-03-21_M3_P1_C1_sin5-3_1Hz')

















# %% GlobalTables functions



# %%% Experimental conditions

expDf = ufun.getExperimentalConditions(cp.DirRepoExp, save=True)


# %%% Constant Field

# %%%% Update the table
# taka.computeGlobalTable_ctField(task='updateExisting', save=False)

# %%%% Refresh the whole table
# taka.computeGlobalTable_ctField(task = 'updateExisting', fileName = 'Global_CtFieldData_Py', save = True, source = 'Python') # task = 'updateExisting'

# %%%% Display

df = taka.getGlobalTable_ctField().head()











# %%% Mechanics


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
                       'doChadwickFit' : True,
                       'ChadwickFitMethods' : ['Full', 'f_<_400', 'f_in_400_800'],
                       'doDimitriadisFit' : False,
                       'DimitriadisFitMethods' : ['Full'],
                       # Local fits
                       'doStressRegionFits' : True,
                       'doStressGaussianFits' : True,
                       'centers_StressFits' : DEFAULT_stressCenters,
                       'halfWidths_StressFits' : DEFAULT_stressHalfWidths,
                       'doNPointsFits' : True,
                       'nbPtsFit' : 13,
                       'overlapFit' : 3,
                       # NEW - Numi
                       'doLogFits' : True,
                       'nbPtsFitLog' : 10,
                       'overlapFitLog' : 5,
                       # NEW - Jojo
                       'doStrainGaussianFits' : True,
                       'centers_StrainFits' : DEFAULT_strainCenters,
                       'halfWidths_StrainFits' : DEFAULT_strainHalfWidths,
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


# %%%% Create a table from scratch

res = taka2.computeGlobalTable_meca(mode = 'fromScratch', task = 'all', fileName = 'MecaData_All_JV', 
                                    save = False, PLOT = False, source = 'Python') # task = 'updateExisting'

# %%%% Refresh an entire table

res = taka2.computeGlobalTable_meca(mode = 'updateExisting', task = 'all', fileName = 'MecaData_All_JV', 
                                    save = True, PLOT = False, source = 'Python') # task = 'updateExisting'



# %%%% ATCC-2023

# %%%%% 23-09-19 - JLY

plot_stressCenters = [ii for ii in range(100, 4000, 50)]
stressHalfWidths = [50, 75, 100]

fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['pts_15',
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_15',
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'centers_StressFits' : plot_stressCenters,
                'halfWidths_StressFits' : stressHalfWidths,
                'doNPointsFits' : True,
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
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'plotStressCenters':plot_stressCenters,
                        'plotStressHW':plot_stressHalfWidth,
                        'S(e)_nPoints':True,
                        'K(S)_nPoints':True,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        }

AtccTask = '23-09-19'
res = taka2.computeGlobalTable_meca(mode = 'fromScratch', task = AtccTask, fileName = 'MecaData_Atcc_JLY', 
                                    save = True, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting'


# %%%%% CalA

plot_stressCenters = [ii for ii in range(100, 4000, 50)]
stressHalfWidths = [50, 75, 100]

fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['pts_15',
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_15',
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'centers_StressFits' : plot_stressCenters,
                'halfWidths_StressFits' : stressHalfWidths,
                'doNPointsFits' : True,
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
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'plotStressCenters':plot_stressCenters,
                        'plotStressHW':plot_stressHalfWidth,
                        'S(e)_nPoints':True,
                        'K(S)_nPoints':True,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        }

AtccTask = '23-07-17 & 23-07-20 & 23-09-06'
res = taka2.computeGlobalTable_meca(mode = 'updateExisting', task = AtccTask, fileName = 'MecaData_Atcc_CalA', 
                                    save = True, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting'



# %%%%% 23-04-26 & 23-04-28 - CK666

fitSettings = {# H0
                'methods_H0':['Chadwick', 'Dimitriadis'],
                'zones_H0':['pts_10', 'pts_20', 
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_10',
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'doNPointsFits' : True,
                # NEW - Numi
                'doLogFits' : False,
                # NEW - Jojo
                'doStrainGaussianFits' : False,
                }

plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'S(e)_nPoints':False,
                        'K(S)_nPoints':False,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        }

AtccTask = '23-04-26 & 23-04-28'
res = taka2.computeGlobalTable_meca(mode = 'updateExisting', task = AtccTask, fileName = 'MecaData_Atcc', 
                                    save = True, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting'


# %%%%% 23-03-16 & 23-03-17 & 23-04-20 - Blebbi

fitSettings = {# H0
                'methods_H0':['Chadwick', 'Dimitriadis'],
                'zones_H0':['pts_10', 'pts_20', 
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_10',
                'doStressRegionFits' : True,
                'doStressGaussianFits' : True,
                'doNPointsFits' : True,
                'doStrainGaussianFits' : True,
                }

plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'S(e)_nPoints':False,
                        'K(S)_nPoints':False,
                        'S(e)_strainGaussian':False, # NEW
                        'K(S)_strainGaussian':False, # NEW
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        }

AtccTask = '23-03-16 & 23-03-17 & 23-04-20'
res = taka2.computeGlobalTable_meca(mode = 'updateExisting', task = AtccTask, fileName = 'MecaData_Atcc', 
                                    save = True, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting'

# %%%%% 23-03-08 & 23-03-09_M4 - Y27

plot_stressCenters = [ii for ii in range(100, 2000, 50)]
stressHalfWidths = [50, 75, 100]

fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['pts_15',
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_15',
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'centers_StressFits' : plot_stressCenters,
                'halfWidths_StressFits' : stressHalfWidths,
                'doNPointsFits' : True,
                'nbPtsFit' : 23,
                'overlapFit' : 11,
                # NEW - Numi
                'doLogFits' : False,
                # NEW - Jojo
                'doStrainGaussianFits' : True,
                }

plot_stressCenters = [ii for ii in range(100, 2000, 100)]
plot_stressHalfWidth = 100

plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'plotStressCenters':plot_stressCenters,
                        'plotStressHW':plot_stressHalfWidth,
                        'S(e)_nPoints':True,
                        'K(S)_nPoints':True,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        }

AtccTask = '23-03-08 & 23-03-09'
res = taka2.computeGlobalTable_meca(mode = 'fromScratch', task = AtccTask, fileName = 'MecaData_Atcc', 
                                    save = True, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting'

# %%%%% 23-02-23 - PNB

fitSettings = {# H0
                'methods_H0':['Chadwick', 'Dimitriadis'],
                'zones_H0':['pts_10', 'pts_20', 'pts_30',
                            '%f_5', '%f_10', '%f_20', '%f_40'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_10',
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'doNPointsFits' : True,
                'doStrainGaussianFits' : True,
                }

plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'S(e)_nPoints':True,
                        'K(S)_nPoints':True,
                        'S(e)_strainGaussian':True, # NEW
                        'K(S)_strainGaussian':True, # NEW
                        }

AtccTask = '23-02-23'
res = taka2.computeGlobalTable_meca(mode = 'fromScratch', task = AtccTask, fileName = 'MecaData_Atcc', 
                                    save = False, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting'


# %%%%% 23-02-16 - Blebbi

fitSettings = {# H0
                'methods_H0':['Chadwick', 'Dimitriadis'],
                'zones_H0':['pts_10', 'pts_20', 
                            '%f_5', '%f_10', '%f_20'],
                'method_bestH0':'Chadwick',
                'zone_bestH0':'%f_10',
                
                }

AtccTask = '23-02-16'
res = taka2.computeGlobalTable_meca(mode = 'updateExisting', task = AtccTask, fileName = 'MecaData_Atcc', 
                                    save = True, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings) # task = 'updateExisting'

# %%%% NanoIndenter


plot_stressCenters = [ii for ii in range(100, 4000, 50)]
stressHalfWidths = [50, 75, 100]

fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['pts_15',
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_15',
                'doVWCFit' : True, # NEW - Numi
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'centers_StressFits' : plot_stressCenters,
                'halfWidths_StressFits' : stressHalfWidths,
                'doNPointsFits' : True,
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
                        'F(H)_VWC':True, # NEW - Numi
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

# NanoIndentTask = '23-11-13 & 23-11-15 & 23-12-07 & 23-12-10'
# NanoIndentTask = '24-02-26 & 24-02-28'
# NanoIndentTask = '24-02-26'
NanoIndentTask = '24-04-11_M2_P1_C3'
res = taka3.computeGlobalTable_meca(mode = 'fromScratch', task = NanoIndentTask, fileName = 'MecaData_testV3',# 'MecaData_NanoIndent_2023-2024', 
                                    save = True, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'fromScratch' // 'updateExisting'


# %%%% Drugs

fitSettings = {# H0
                'methods_H0':['Chadwick', 'Dimitriadis'],
                'zones_H0':['pts_10', 'pts_20', 
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_10',
                'doStressRegionFits' : True,
                'doStressGaussianFits' : True,
                'doNPointsFits' : True,
                # NEW - Numi
                'doLogFits' : False,
                # NEW - Jojo
                'doStrainGaussianFits' : True,
                # TEST - Jojo
                'do3partsFits' : False,
                }

plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':False,
                        'K(S)_stressGaussian':False,
                        'S(e)_nPoints':False,
                        'K(S)_nPoints':False,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        'S(e)_3parts':False, # TEST - Jojo
                        }


drugTask = '22-03-28 & 22-03-30 & 22-11-23'
res = taka2.computeGlobalTable_meca(mode = 'fromScratch', task = drugTask, fileName = 'MecaData_aSFL-Drugs', 
                                    save = True, PLOT = False, source = 'Python', fitSettings = fitSettings) # task = 'updateExisting'

# %%%% Drugs 2

plot_stressCenters = [ii for ii in range(100, 4000, 50)]
stressHalfWidths = [50, 75, 100]

fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['pts_15',
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_15',
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'doVWCFit' : True,
                'centers_StressFits' : plot_stressCenters,
                'halfWidths_StressFits' : stressHalfWidths,
                'doNPointsFits' : True,
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
                        'F(H)_VWC':True, # NEW - Numi
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'plotStressCenters':plot_stressCenters,
                        'plotStressHW':plot_stressHalfWidth,
                        'S(e)_nPoints':True,
                        'K(S)_nPoints':True,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        }


# drugTask  = '22-03-28 & 22-03-30' # Blebbi & LatA
# drugTask += ' & 22-11-23 & 23-11-26 & 23-12-03' # LatA
# drugTask += ' & 23-03-08 & 23-03-09' # Y27
# drugTask += ' & 23-02-16 & 23-02-23 & 23-03-16 & 23-03-17 & 23-04-20' # PNB & Blebbistatin
# drugTask += ' & 23-04-26 & 23-04-28' # Ck666
# drugTask += ' & 23-07-17_M1 & 23-07-17_M2 & 23-07-17_M3 & 23-07-20 & 23-09-06' # CalA
# drugTask += ' & 23-09-19' # JLY
# drugTask = '24-03-13' # Limki + Y27
drugTask = '24-07-04' # Limki + Y27 + Blebbi

# drugTask = '23-09-19'
res = taka3.computeGlobalTable_meca(mode = 'updateExisting', task = drugTask, fileName = 'MecaData_Drugs_V3', 
                                    save = True, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting' / 'fromScratch'

# %%%% Physics

plot_stressCenters = [ii for ii in range(100, 4000, 50)]
stressHalfWidths = [50, 75, 100]

fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['pts_15',
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_15',
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'doVWCFit' : True,
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

plot_stressCenters = [ii for ii in range(100, 2050, 100)]
plot_stressHalfWidth = 75

# plotSettings = {# ON/OFF switchs plot by plot
#                         'FH(t)':True,
#                         'F(H)':True,
#                         'F(H)_VWC':True, # NEW - Numi
#                         'S(e)_stressRegion':False,
#                         'K(S)_stressRegion':False,
#                         'S(e)_stressGaussian':True,
#                         'K(S)_stressGaussian':True,
#                         'plotStressCenters':plot_stressCenters,
#                         'plotStressHW':plot_stressHalfWidth,
#                         'S(e)_nPoints':False,
#                         'K(S)_nPoints':False,
#                         'S(e)_strainGaussian':False, # NEW - Jojo
#                         'K(S)_strainGaussian':False, # NEW - Jojo
#                         'S(e)_Log':False, # NEW - Numi
#                         'K(S)_Log':False, # NEW - Numi
#                         }

plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'F(H)_VWC':True, # NEW - Numi
                        'Plots_Manuscript':True,
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

phyTask = '21-07-08_M1-2 & 21-07-08_M2 & 21-07-08_M4' # Bead size
phyTask += ' & 21-10-18 & 21-10-25 & 21-12-08 & 21-12-16 & 22-01-12' # Bead size
# phyTask += '22-06-10 & 22-06-16 & 22-07-06 & 22-07-06 & 22-07-12 & 22-07-12' # Pattern sizes DB 1/2
# phyTask += ' & 22-07-22 & 22-07-29 & 22-08-24 & 22-08-24 & 22-08-24 & 22-08-24' # Pattern sizes DB 2/2
# phyTask += ' & 23-03-09' # Pattern sizes JV
# phyTask += ' & 23-07-06_M1 & 23-07-06_M2 & 23-07-06_M3 & 23-07-06_M4 & 23-07-06_M5' # Repeats
# phyTask += ' & 23-07-06_M6 & 23-07-06_M7 & 23-07-06_M8' # Various fields
# phyTask += ' & 23-02-16_M1 & 23-02-23_M1 & 23-02-23_M3 & 23-03-08_M3 & 23-03-16_M1 & 23-03-17_M4' # Dmso & none 1/4
# phyTask += ' & 23-04-20_M1 & 23-04-20_M4 & 23-04-20_M5 & 23-04-26_M2 & 23-04-28_M1 & 23-07-17_M3' # Dmso & none 2/4
# phyTask += ' & 23-07-17_M4 & 23-07-17_M6 & 23-07-20_M2 & 23-09-06_M3 & 23-09-11_M1 & 23-09-19_M1 & 23-11-26_M2 & 23-12-03_M1' # Dmso & none 3/4
# phyTask += ' & 24-07-04_M2 & 24-07-04_M6' # Dmso & none 4/4

res = takaM.computeGlobalTable_meca(mode = 'fromScratch', task = phyTask, fileName = 'MecaData_Physics_BeadSize', 
                                    save = True, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting' / 'fromScratch'

# %%%% Cell Types

plot_stressCenters = [ii for ii in range(100, 4000, 50)]
stressHalfWidths = [50, 75, 100]

fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['pts_15',
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_15',
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'centers_StressFits' : plot_stressCenters,
                'halfWidths_StressFits' : stressHalfWidths,
                'doNPointsFits' : True,
                'nbPtsFit' : 11,
                'overlapFit' : 5,
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
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'plotStressCenters':plot_stressCenters,
                        'plotStressHW':plot_stressHalfWidth,
                        'S(e)_nPoints':True,
                        'K(S)_nPoints':True,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        }

cellTask = '20-09 & 20-10' # Dictys
cellTask += ' & 18-08 & 18-09 & 18-10 & 18-12' # DC
cellTask += ' & 22-05-03 & 22-05-04 & 22-05-05' # HoxB8
cellTask += ' & 22-02-09'
cellTask += ' & 21-12-08_M2 & 21-12-16_M1' # 3t3 aSFL 1/2
cellTask += ' & 22-07-15_M4 & 22-07-20_M2' # 3t3 aSFL 2/2
cellTask += ' & 23-07-17_M6 & 23-07-17_M4 & 23-03-09_M4 & 23-02-23_M1 & 24-02-26 & 24-02-28' # 3t3 ATCC (WT & LaGFP)

res = taka3.computeGlobalTable_meca(mode = 'fromScratch', task = cellTask, fileName = 'MecaData_CellTypes', 
                                    save = True, PLOT = False, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting' / 'fromScratch'

# %%%% Remove empty rows

path = "D:/MagneticPincherData/Data_Analysis/MecaData_CellTypes.csv"
df = pd.read_csv(path, sep = ';')

# df2 = df.dropna(subset='date')
# df2.to_csv(path, sep=';', index=False)

# %%%% Chameleon Compression + FluoQuantif

plot_stressCenters = [ii for ii in range(100, 4000, 50)]
stressHalfWidths = [50, 75, 100]

fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['pts_15',
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_15',
                'doVWCFit' : True, # NEW - Numi
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'centers_StressFits' : plot_stressCenters,
                'halfWidths_StressFits' : stressHalfWidths,
                'doNPointsFits' : True,
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
                        'F(H)_VWC':True, # NEW - Numi
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'plotStressCenters':plot_stressCenters,
                        'plotStressHW':plot_stressHalfWidth,
                        'S(e)_nPoints':True,
                        'K(S)_nPoints':True,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        }

task = '24-06-14'
# task = '24-06-14_M2_P1_C2-2'
res = taka3.computeGlobalTable_meca(mode = 'fromScratch', task = task, fileName = 'MecaData_Chameleon_CompFluo', 
                                    save = True, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting' / 'fromScratch'

# %%%% Non-Lin

fitSettings = {# H0
                'methods_H0':['Chadwick', 'Dimitriadis'],
                'zones_H0':['pts_10', 'pts_20', 
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_10',
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'doNPointsFits' : True,
                # NEW - Numi
                'doLogFits' : False,
                # NEW - Jojo
                'doStrainGaussianFits' : False,
                }

plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'S(e)_nPoints':False,
                        'K(S)_nPoints':False,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        }

nonLinTask = '21-12-08 & 22-01-12 & 21-12-16 & 22-02-09'
res = taka2.computeGlobalTable_meca(mode = 'updateExisting', task = nonLinTask, fileName = 'MecaData_NonLin', 
                            save = True, PLOT = True, source = 'Python', 
                            fitSettings = fitSettings,
                            plotSettings = plotSettings) # task = 'updateExisting'

# %%%% MCA

# %%%%% 2021

# MCAtask = '21-01-18 & 21-01-21'
# res = taka2.computeGlobalTable_meca(mode = 'updateExisting', task = MCAtask, fileName = 'Global_MecaData_MCA', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'

# %%%%% 2022

# MCAtask = '22-07-15 & 22-07-20 & 22-07-27' # ' & 22-07-20 & 22-07-27' #' & 22-05-04 & 22-05-05'
# taka2.computeGlobalTable_meca(mode = 'fromScratch', task = MCAtask, fileName = 'Global_MecaData_MCA', 
#                             save = False, PLOT = False, source = 'Python') # task = 'updateExisting'

# %%%%% All MCA 2021 - 2022

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

all_MCAtask = ''
for d in all_dates[:-1]:
    all_MCAtask += d
    all_MCAtask += ' & '
all_MCAtask += all_dates[-1]

res = taka2.computeGlobalTable_meca(mode = 'fromScratch', task = all_MCAtask, fileName = 'MecaData_MCA', 
                            save = False, PLOT = False, source = 'Python') # task = 'updateExisting'

# %%%% HoxB8

# %%%%% HoxB8 only

HoxB8task = '22-05-03 & 22-05-04 & 22-05-05' #' & 22-05-04 & 22-05-05' '22-05-03 & 22-05-04 & 22-05-05'
res = taka2.computeGlobalTable_meca(mode = 'fromScratch', task = HoxB8task, fileName = 'MecaData_HoxB8', 
                                    save = False, PLOT = False, source = 'Python') # task = 'updateExisting'
# taka.computeGlobalTable_meca(task = MCAtask, fileName = 'Global_MecaData_MCA2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'

# %%%%% MCA & HoxB8

MCA_and_HoxB8_task = all_MCAtask + ' & ' + HoxB8task
res = taka2.computeGlobalTable_meca(mode = 'fromScratch', task = MCA_and_HoxB8_task, fileName = 'MecaData_MCA-HoxB8', 
                            save = False, PLOT = False, source = 'Python') # task = 'updateExisting'


# %%%%% Old Expts

DC_task = '18-08-28 & 18-09-24 & 18-09-24 & 18-10-30 & 18-12-12'
res = taka2.computeGlobalTable_meca(mode = 'fromScratch', task = DC_task, fileName = 'MecaData_DC', 
                            save = True, PLOT = True, source = 'Python') # task = 'updateExisting'

# %%%% Anumita's data

fitSettings = {# H0
                'methods_H0':['Chadwick', 'Dimitriadis'],
                'zones_H0':['pts_10', 'pts_20', 
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_10',
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'doNPointsFits' : True,
                # NEW - Numi
                'doLogFits' : False,
                # NEW - Jojo
                'doStrainGaussianFits' : True,
                }

plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'S(e)_nPoints':False,
                        'K(S)_nPoints':False,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        }


fromAJ_task = '22-12-07 & 23-02-02 & 23-03-24 & 23-03-28 & 23-04-25'
# testTask = '23-03-24_M2_P1_C22'
res = taka2.computeGlobalTable_meca(mode = 'updateExisting', task = fromAJ_task, fileName = 'MecaData_fromAJ', 
                            save = True, PLOT = True, source = 'Python', 
                            fitSettings = fitSettings,
                            plotSettings = plotSettings) # task


# %%%% N Repeats

plot_stressCenters = [ii for ii in range(100, 4000, 100)]
stressHalfWidths = [100]

fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['pts_15',
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_15',
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


task_repeats = '23-07-06_M1 & 23-07-06_M2 & 23-07-06_M3 & 23-07-06_M4'
# task_repeats = '23-07-06_M1_P1_C2'
res = taka2.computeGlobalTable_meca(mode = 'fromScratch', task = task_repeats, fileName = 'MecaData_repeats', 
                            save = False, PLOT = True, source = 'Python', 
                            fitSettings = fitSettings,
                            plotSettings = plotSettings) # task

# %%%% "3 Fields"

plot_stressCenters = [ii for ii in range(100, 4000, 100)]
stressHalfWidths = [100]

fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['pts_15',
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_15',
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'centers_StressFits' : plot_stressCenters,
                'halfWidths_StressFits' : stressHalfWidths,
                'doNPointsFits' : True,
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
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'plotStressCenters':plot_stressCenters,
                        'plotStressHW':plot_stressHalfWidth,
                        'S(e)_nPoints':True,
                        'K(S)_nPoints':True,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        }


task_3f = '23-07-06_M6 & 23-07-06_M7 & 23-07-06_M8'
# task_repeats = '23-07-06_M1_P1_C2'
res = taka2.computeGlobalTable_meca(mode = 'fromScratch', task = task_3f, fileName = 'MecaData_3fields', 
                            save = True, PLOT = True, source = 'Python', 
                            fitSettings = fitSettings,
                            plotSettings = plotSettings) # task


# %%%% Next job !







# %%%% Tools


# %%%%% Display

df = taka2.getGlobalTable_meca('Global_MecaData_Py2')

# %%%%% Plot a precise date


plot_stressCenters = [ii for ii in range(100, 4000, 50)]
stressHalfWidths = [50, 75, 100]

fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['pts_15',
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_15',
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'centers_StressFits' : plot_stressCenters,
                'halfWidths_StressFits' : stressHalfWidths,
                'doNPointsFits' : True,
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
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'plotStressCenters':plot_stressCenters,
                        'plotStressHW':plot_stressHalfWidth,
                        'S(e)_nPoints':True,
                        'K(S)_nPoints':True,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        }


# drugTask  = '22-03-28 & 22-03-30' # Blebbi & LatA
# drugTask += ' & 22-11-23 & 23-11-26 & 23-12-03' # LatA
# drugTask += ' & 23-03-08 & 23-03-09' # Y27
# drugTask += ' & 23-02-16 & 23-02-23 & 23-03-16 & 23-03-17 & 23-04-20' # PNB & Blebbistatin
# drugTask += ' & 23-04-26 & 23-04-28' # Ck666
# drugTask += ' & 23-07-17_M1 & 23-07-17_M2 & 23-07-17_M3 & 23-07-20 & 23-09-06' # CalA
# drugTask += ' & 23-09-19' # JLY
drugTask = '24-03-13' # JLY

# drugTask = '23-09-19'
res = taka2.computeGlobalTable_meca(mode = 'fromScratch', task = drugTask, fileName = 'MecaData_Drugs_24-03-13', 
                                    save = False, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting' / 'fromScratch'



# %%%%% Plot a precise date -- Manuscript


plot_stressCenters = [ii for ii in range(100, 4000, 50)]
stressHalfWidths = [50, 75, 100]

fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['pts_15',
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_15',
                'doVWCFit' : True,
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'centers_StressFits' : plot_stressCenters,
                'halfWidths_StressFits' : stressHalfWidths,
                'doNPointsFits' : True,
                'nbPtsFit' : 33,
                'overlapFit' : 21,
                # NEW - Numi
                'doLogFits' : False,
                # NEW - Jojo
                'doStrainGaussianFits' : True,
                }

plot_stressCenters = [ii for ii in range(100, 2050, 100)]
plot_stressHalfWidth = 75

plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':False,
                        'F(H)':False,
                        'F(H)_VWC':False, # NEW - Numi
                        'Plots_Manuscript':True,
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

# task = '24-03-13_M1_P1_C15'
# task = '23-03-17_M4_P1_C15 & 23-03-17_M4_P1_C14 & 23-03-17_M4_P1_C8 & 24-07-04_M4_P1_C16' # 23-03-16_M1_P1_C2 & 
# task = '24-07-04_M4_P1_C15'
# task = '24-07-04_M6'
# task = '23-03-17_M4'
# task = '24-07-04_M6_P1_C11'
# task = '23-03-09_M4_P1_C2 & 23-03-09_M4_P1_C5 & 23-03-09_M4_P1_C12'
# task += ' & 23-03-09_M4_P1_C4 & 23-03-09_M4_P1_C8 & 23-03-09_M4_P1_C9'
# task += ' & 23-03-09_M4_P1_C15 & 23-03-09_M4_P1_C14'
task = '23-03-17_M4_P1_C3 & 23-03-17_M4_P1_C9 & 23-03-17_M4_P1_C11'
# task = '23-03-16_M1_P1_C4'

res = takaM.computeGlobalTable_meca(mode = 'fromScratch', task = task, fileName = 'test', 
                                    save = False, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting'


task = '23-03-16_M1_P1_C4 & 23-03-16_M1_P1_C9 & 23-03-16_M1_P1_C16 & 23-03-16_M1_P1_C21'

res = takaM.computeGlobalTable_meca(mode = 'fromScratch', task = task, fileName = 'test', 
                                    save = False, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting'

# %%%%% Concat dataframes

fileNames = ['Global_MecaData_NonLin_Py', 'Global_MecaData_MCA-HoxB8']
cdf = taka.concatAnalysisTables(fileNames, save = True, saveName = 'Global_MecaData_MCA-HoxB8_2')













# %%% Fluorescence

# %%%% Display

df = taka.getFluoData().head()


