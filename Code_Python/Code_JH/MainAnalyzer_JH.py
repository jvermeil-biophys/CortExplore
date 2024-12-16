# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:31:26 2022
@author: Joseph Vermeil

MainAnalyzer_##.py - Script to use the TrackAnalyzer program.
Please replace the "_NewUser" in the name of the file by "_##", 
a suffix corresponding to the user's name (ex: JV for Joseph Vermeil) 
Joseph Vermeil, 2022

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
allTimeseriesDataFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                          if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv"))]
    
print(allTimeseriesDataFiles)

# %%% Get a time series
# Example : df = taka.getCellTimeSeriesData('22-02-09_M1_P1_C7')


# %%% Plot a time series
#  Example : taka.plotCellTimeSeriesData('21-02-10_M1_P1_C2')




# #############################################################################
# %% GlobalTables functions

# =============================================================================
# %%% Experimental conditions
expDf = ufun.getExperimentalConditions()



# =============================================================================
# %%% Constant Field

# %%%% Update the table
# Example : taka.computeGlobalTable_ctField(task='updateExisting', fileName = 'Global_CtFieldData_Py', 
#                                           save = False, source = 'Python')

# %%%% Refresh the whole table
# Example : taka.computeGlobalTable_ctField(task = 'fromScratch', fileName = 'Global_CtFieldData_Py', 
#                                           save = True, source = 'Python')

# %%%% Display
# Example : df = taka.getGlobalTable_ctField().head()



# =============================================================================
# %%% Mechanics

# %%%% Update the table

# Example : 
# taka.computeGlobalTable_meca(task = 'updateExisting', fileName = 'Global_MecaData_Py', 
#                             save = False, PLOT = False, source = 'Matlab') # task = 'updateExisting'


# %%%% Refresh the whole table

# Example : 
# taka.computeGlobalTable_meca(task = 'updateExisting', fileName = 'Global_MecaData_Py2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'


# %%%% Specific task

plot_stressCenters = [ii for ii in range(100, 4000, 50)]
stressHalfWidths = [50, 75, 100]

fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['pts_15',
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_15',
                'doChadwickFit' : True,
                'ChadwickFitMethods' : ['Full', 'f_<_400', 'f_in_400_800'],
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

plot_stressCenters = [ii for ii in range(100, 4000, 100)]
plot_stressHalfWidth = 100

plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'F(H)_VWC':True, # NEW - Numi
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


# drugTask  = '22-03-28 & 22-03-30' # Blebbi & LatA

Task =  '24-02-26_M3_P1_C4' # Test

# drugTask = '23-09-19'
res = taka3.computeGlobalTable_meca(mode = 'fromScratch', task = Task, fileName = 'MecaData_test', 
                                    save = False, PLOT = True, source = 'Python', 
                                    fitSettings = fitSettings,
                                    plotSettings = plotSettings) # task = 'updateExisting' / 'fromScratch'






# Example : 
# Task_1 = '22-05-03'
# taka.computeGlobalTable_meca(task = Task_1, fileName = 'Global_MecaData_Demo', 
#                             save = True, PLOT = True, source = 'Python') # task = 'updateExisting'
# Task_2 = '22-05-03_M1 & 22-05-04_M2'
# taka.computeGlobalTable_meca(task = Task_2, fileName = 'Global_MecaData_MCA2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'
# Task_3 = '22-05-03 & 22-05-04 & 22-05-05'
# taka.computeGlobalTable_meca(task = Task_3, fileName = 'Global_MecaData_MCA2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'



# %%%% Display

# Example : df = taka.getGlobalTable_meca('Global_MecaData_Py2').tail()



# =============================================================================
# %%% Fluorescence

# %%%% Display

# Example : df = taka.getFluoData().head()


