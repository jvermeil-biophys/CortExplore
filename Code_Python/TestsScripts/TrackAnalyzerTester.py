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

#### Graphic options
gs.set_default_options_jv()


# %%

path = cp.DirCloudTimeseries
ld = os.listdir(path)

expDf = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = cp.suffix)

f = '22-07-15_M2_P1_C5_disc20um_L40_PY.csv'
tsDf = taka2.getCellTimeSeriesData(f, fromCloud = True)

dCM = taka2.dictColumnsMeca

taka2.analyseTimeSeries_Class(f, tsDf, expDf, dCM)

# %%


