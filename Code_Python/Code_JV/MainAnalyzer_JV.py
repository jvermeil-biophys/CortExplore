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
import matplotlib.pyplot as plt


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
import TrackAnalyser_dev3_AJJV as taka2

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


# #############################################################################
# %% GlobalTables functions



# %%% Experimental conditions

expDf = ufun.getExperimentalConditions(cp.DirRepoExp, save=True)




# =============================================================================
# %%% Constant Field

# %%%% Update the table
# taka.computeGlobalTable_ctField(task='updateExisting', save=False)

# %%%% Refresh the whole table
# taka.computeGlobalTable_ctField(task = 'updateExisting', fileName = 'Global_CtFieldData_Py', save = True, source = 'Python') # task = 'updateExisting'

# %%%% Display
df = taka.getGlobalTable_ctField().head()






# =============================================================================
# %%% Mechanics

# %%%% Update the table

taka.computeGlobalTable_meca(mode = 'updateExisting', task = 'all', fileName = 'Global_MecaData_Py2', 
                            save = True, PLOT = False, source = 'Python') # task = 'updateExisting'


# %%%% Refresh the whole table

# taka.computeGlobalTable_meca(mode = 'fromScratch', task = 'all', fileName = 'Global_MecaData_Py2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'

# %%%% Drugs

drugTask = '22-03-30'
taka2.computeGlobalTable_meca(mode = 'fromScratch', task = drugTask, fileName = 'Global_MecaData_Drugs_Py', 
                            save = True, PLOT = False, source = 'Python') # task = 'updateExisting'


# %%%% Non-Lin

nonLinTask = '21-12-08 & 22-01-12 & 22-02-09'
taka.computeGlobalTable_meca(mode = 'updateExisting', task = nonLinTask, fileName = 'Global_MecaData_NonLin_Py', 
                            save = False, PLOT = False, source = 'Python') # task = 'updateExisting'

# %%%% MCA

MCAtask = '21-01-18 & 21-01-21'
taka.computeGlobalTable_meca(mode = 'updateExisting', task = MCAtask, fileName = 'Global_MecaData_MCA', 
                            save = False, PLOT = False, source = 'Python') # task = 'updateExisting'
# taka.computeGlobalTable_meca(task = MCAtask, fileName = 'Global_MecaData_MCA2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'


# %%%% New MCA 2022

MCA2task = '22-07-15 & 22-07-20 & 22-07-27' # ' & 22-07-20 & 22-07-27' #' & 22-05-04 & 22-05-05'
taka.computeGlobalTable_meca(mode = 'fromScratch', task = MCA2task, fileName = 'Global_MecaData_MCA3', 
                            save = True, PLOT = True, source = 'Python') # task = 'updateExisting'
# taka.computeGlobalTable_meca(task = MCAtask, fileName = 'Global_MecaData_MCA2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'

# %%%% All MCA 2021 - 2022

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

MCA123task = ''
for d in all_dates[:-1]:
    MCA123task += d
    MCA123task += ' & '
MCA123task += all_dates[-1]

taka.computeGlobalTable_meca(mode = 'fromScratch', task = MCA123task, fileName = 'Global_MecaData_MCA123-new', 
                            save = True, PLOT = False, source = 'Python') # task = 'updateExisting'
# taka.computeGlobalTable_meca(task = MCAtask, fileName = 'Global_MecaData_MCA2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'


# %%%% HoxB8

HoxB8task = '22-05-03 & 22-05-04 & 22-05-05' #' & 22-05-04 & 22-05-05'
taka.computeGlobalTable_meca(task = HoxB8task, fileName = 'Global_MecaData_HoxB8_2', 
                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'
# taka.computeGlobalTable_meca(task = MCAtask, fileName = 'Global_MecaData_MCA2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'

# %%%% MCA & HoxB8

MCA_and_HoxB8_task = MCAtask + ' & ' + HoxB8task
taka.computeGlobalTable_meca(mode = 'updateExisting', task = MCA_and_HoxB8_task, fileName = 'Global_MecaData_MCA-HoxB8', 
                            save = True, PLOT = False, source = 'Python') # task = 'updateExisting'




# %%%% Tests

Test = '21-04-28' # '22-05-03_M3'#' & 22-05-04_M2' #' & 22-05-04 & 22-05-05'
taka.computeGlobalTable_meca(mode = 'fromScratch', task = Test, fileName = 'Global_MecaData_TEST', 
                            save = True, PLOT = False, source = 'Python') # task = 'updateExisting'

# taka.computeGlobalTable_meca(task = MCAtask, fileName = 'Global_MecaData_MCA2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'


# %%%% Precise dates (to plot)


PlotTask = '21-01-18_M2_P1_C3' # ' & 22-07-20 & 22-07-27' #' & 22-05-04 & 22-05-05'
gdf = taka.computeGlobalTable_meca(mode = 'fromScratch', task = PlotTask, fileName = 'aaa', 
                                   save = False, PLOT = True, source = 'Python')


# %%%% Display

df = taka.getGlobalTable_meca('Global_MecaData_Py2').tail()

# %%%% Test of Numi's CRAZY GOOD NEW STUFF :D

taka.computeGlobalTable_meca(task = '22-02-09_M1_P1_C3', fileName = 'aaa', 
                            save = False, PLOT = False, source = 'Python') # task = 'updateExisting'

# %%%% Concat dataframes

fileNames = ['Global_MecaData_NonLin_Py', 'Global_MecaData_MCA-HoxB8']
cdf = taka.concatAnalysisTables(fileNames, save = True, saveName = 'Global_MecaData_MCA-HoxB8_2')


# =============================================================================
# %%% Fluorescence

# %%%% Display

df = taka.getFluoData().head()


