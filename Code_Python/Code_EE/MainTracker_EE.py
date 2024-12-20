# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:50:53 2021
@author: Joseph Vermeil

MainTracker_##.py - Script to use the Tracking functions in the BeadTracker program.
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

# %% General imports

# 1. Imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt

import os
import sys
import matplotlib

# Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)


import GraphicStyles as gs
import GlobalConstants as gc
import UtilityFunctions as ufun

from BeadTracker_V4 import mainTracker_V4


# 2. Pandas settings
pd.set_option('mode.chained_assignment',None)

# 3. Graphical settings
gs.set_default_options_jv()


# 4. Import of the experimental conditions

expDf = ufun.getExperimentalConditions(DirExp = cp.DirRepoExp, save = True)

# %% Small things

#### close all
plt.close('all')



# %% Next Topic !!

# %%% Next experiment day

# %%%% Next manipe

# %%%% Next manipe




# %% Next Topic !!

# %%% Next experiment day

# %%%% Next manipe

# %%%% Next manipe




# %% Example Topic - HoxB8 macrophages

# %%% 22.05.05, compressionsLowStart of HoxB8 macrophages, M450, M1 = tko & glass, M2 = ctrl & glass
# %%%% 22.05.05_M1 C1 only
# dates = '22.05.05'
# manips, wells, cells = 2, 1, 2
# depthoNames = '22.05.05_M1_M450_step20_100X'

# output = mainTracker(dates, manips, wells, cells, depthoNames, expDf, 
#                      redoAllSteps = True, MatlabStyle = True, trackAll = False, 
#                      sourceField = 'default')

# %%% 24.04.24, First constant field experiment HeLa FUCCI on Zen

dates = '24.04.24'
manips, wells, cells = 1, 2, 2
depthoNames = '24.04.24_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoNames, expDf, metaDataFormatting = 'constantField',
                     redoAllSteps = True, trackAll = False)

# %%

dates = '24.05.23'
manips, wells, cells = 1, 'all', 'all'
depthoNames = '24.05.23_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoNames, expDf, redoAllSteps = True, trackAll = False)

#%%
dates = '24.06.04'
manips, wells, cells = 1, 2,'all'
depthoNames = '24.06.04_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoNames, expDf, redoAllSteps = True, trackAll = True)

#%%
dates = '24.07.18'
manips, wells, cells = 1, 2, 'all'
depthoNames = '24.07.18_M450_step20_100X_P2' 

output = mainTracker_V4(dates, manips, wells, cells, depthoNames, expDf, redoAllSteps = True, trackAll = False)