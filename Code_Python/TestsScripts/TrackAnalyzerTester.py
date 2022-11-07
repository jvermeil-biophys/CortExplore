# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:36:39 2022

@author: Joseph
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
import numbers
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
# import TrackAnalyser_dev_AJ as taka

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
plt.ioff()

#### Graphic options
gs.set_default_options_jv()


# %%

path = cp.DirCloudTimeseries
ld = os.listdir(path)

expDf = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = cp.suffix)

f = '22-07-15_M1_P1_C5_disc20um_L40_PY.csv'
tsDf = taka2.getCellTimeSeriesData(f, fromCloud = True)

res = taka2.analyseTimeSeries_Dev(f, tsDf, expDf, PLOT = False, SHOW = False)

# task = '22-07-15_M2_P1_C1'
# df = taka2.computeGlobalTable_meca(task = task, fileName = 'test', 
#                             save = True, PLOT = False, 
#                             source = 'Python', 
#                             ui_fileSuffix = 'UserManualSelection_MecaData')

# %%

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

MCA123task = ''
for d in all_dates[:-1]:
    MCA123task += d
    MCA123task += ' & '
MCA123task += all_dates[-1]

taka2.computeGlobalTable_meca(mode = 'fromScratch', task = MCA123task, fileName = 'NEW_MecaData_MCA123', 
                            save = True, PLOT = False, source = 'Python') # task = 'updateExisting'
