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
                       'doChadwickFit' : True,
                       'doDimitriadisFit' : False,
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

fitSettings = {# H0
                'methods_H0':['Chadwick', 'Dimitriadis'],
                'zones_H0':['pts_10', 'pts_20', 'pts_30',
                            '%f_5', '%f_10', '%f_20', '%f_40'],
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
                        'S(e)_nPoints':True,
                        'K(S)_nPoints':True,
                        'S(e)_strainGaussian':True, # NEW
                        'K(S)_strainGaussian':True, # NEW
                        }

AtccTask = '23-03-08 & 23-03-09_M4'
res = taka2.computeGlobalTable_meca(mode = 'updateExisting', task = AtccTask, fileName = 'MecaData_Atcc', 
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


# %%%% Next job !





# %%%% Tools


# %%%%% Display

df = taka2.getGlobalTable_meca('Global_MecaData_Py2')

# %%%%% Plot a precise date


fitSettings = {# H0
                'methods_H0':['Chadwick', 'Dimitriadis'],
                'zones_H0':['pts_10', 'pts_20', 
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_10',
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'doNPointsFits' : False,
                # NEW - Numi
                'doLogFits' : False,
                # NEW - Jojo
                'doStrainGaussianFits' : False,
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

AtccTask = '22-02-09_M1_P1_C9'
res = taka2.computeGlobalTable_meca(mode = 'fromScratch', task = AtccTask, fileName = 'MecaData_Atcc_ALT', 
                                    save = False, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting'


# %%%%% Concat dataframes

fileNames = ['Global_MecaData_NonLin_Py', 'Global_MecaData_MCA-HoxB8']
cdf = taka.concatAnalysisTables(fileNames, save = True, saveName = 'Global_MecaData_MCA-HoxB8_2')













# %%% Fluorescence

# %%%% Display

df = taka.getFluoData().head()


