# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:50:53 2021

@author: Joseph Vermeil
"""

# %% > General imports

# 1. Imports
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

import os
import re
import time
import pyautogui
import matplotlib

from scipy import interpolate
from scipy import signal

from skimage import io, filters, exposure, measure, transform, util
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import linear_sum_assignment

# Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)


import GraphicStyles as gs
import GlobalConstants as gc
import UtilityFunctions as ufun

from BeadTracker import mainTracker
from BeadTracker_V2 import mainTracker_V2
from BeadTracker_V3 import mainTracker_V3
from BeadTracker_V4 import mainTracker_V4

# 2. Pandas settings
pd.set_option('mode.chained_assignment',None)
pd.set_option('display.max_columns', None)

# 3. Graphical settings
gs.set_default_options_jv()


# 4. Import of the experimental conditions

expDf = ufun.getExperimentalConditions(DirExp = cp.DirRepoExp, save = False, suffix = cp.suffix)

# %% Small things

#### Plot last traj

# fig, ax = plt.subplots(1,1)
# X1, Y1 = listTrajDicts[0]['X'], listTrajDicts[0]['Y']
# X2, Y2 = listTrajDicts[1]['X'], listTrajDicts[1]['Y']
# ax.plot(X1, Y1, 'r-')
# ax.plot(X2, Y2, 'b-')

# fig.show()

#### close all
plt.close('all')

# %% Next Topic !!

# %%% Next experiment day
# %%%% Next manipe
# %%%% Next manipe

# %% NANO-INDENTER

# %%% 23-11-15

dates = '23.11.15'
manips, wells, cells = 1, 1, 42
depthoName = '23.11.15_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'statusFile', redoAllSteps = True, trackAll = False)

# %%% 23-11-13

dates = '23.11.13'
manips, wells, cells = 1, 1, 'all'
depthoName = '23.11.13_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'statusFile', redoAllSteps = True, trackAll = False)

dates = '23.11.13'
manips, wells, cells = 2, 1, 'all'
depthoName = '23.11.13_M2_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'statusFile', redoAllSteps = True, trackAll = False)

# %% Re-analyze Valentin's experiments on DC with drugs

# %%% 18-12-12
# %%%% M1 - Ctrl 
dates = '18.09.24'
manips, wells, cells = 1, 'all', 'all'
depthoName = '18.08.28_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')



# %%% 18-10-30
# %%%% M2 - Blebbi 
dates = '18.09.24'
manips, wells, cells = 1, 'all', 'all'
depthoName = '18.08.28_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')
 
# %%%% M3 - CalA 
dates = '18.09.24'
manips, wells, cells = 1, 'all', 'all'
depthoName = '18.08.28_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% M4 - Ctrl
dates = '18.09.24'
manips, wells, cells = 1, 'all', 'all'
depthoName = '18.08.28_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')



# %%% 18-09-25
# %%%% M1 - LatA
dates = '18.09.25'
manips, wells, cells = 1, 'all', 'all'
depthoName = '18.08.28_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% M2 - Smifh2
dates = '18.09.25'
manips, wells, cells = 2, 'all', 'all'
depthoName = '18.08.28_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% M3 - Ctrl
dates = '18.09.25'
manips, wells, cells = 3, 'all', 'all'
depthoName = '18.08.28_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% M4 - CK666 
dates = '18.09.25'
manips, wells, cells = 4, 'all', 'all'
depthoName = '18.08.28_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')



# %%% 18-09-24
# %%%% M1 - CK666
dates = '18.09.24'
manips, wells, cells = 1, 'all', 'all'
depthoName = '18.08.28_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% M2 - Ctrl 
dates = '18.09.24'
manips, wells, cells = 2, 'all', 'all'
depthoName = '18.08.28_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% M3 - Smifh2
dates = '18.09.24'
manips, wells, cells = 3, 'all', 'all'
depthoName = '18.08.28_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')



# %%% 18-08-28
# %%%% M1 - Ctrl - 1 cell
expDf = ufun.getExperimentalConditions(DirExp = cp.DirRepoExp, save = False, suffix = cp.suffix)

dates = '18.08.28'
manips, wells, cells = 1, 1, 3
depthoName = '18.08.28_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%%% M1 - Ctrl - all
dates = '18.08.28'
manips, wells, cells = 1, 'all', 'all'
depthoName = '18.08.28_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% M2 - Ctrl 
dates = '18.08.28'
manips, wells, cells = 2, 'all', 'all'
depthoName = '18.08.28_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')





# %% HoxB8 macrophages

# %%% 22.05.05, compressionsLowStart of HoxB8 macrophages, M450, M1 = tko & glass, M2 = ctrl & glass
# %%%% 22.05.05_M1 C1 Seulement
dates = '22.05.05'
manips, wells, cells = 2, 1, 2
depthoName = '22.05.05_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.05.05_M1
dates = '22.05.05'
manips, wells, cells = 1, 1, 'all'
depthoName = '22.05.05_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.05.05_M2
dates = '22.05.05'
manips, wells, cells = 2, 1, 'all'
depthoName = '22.05.05_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%% 22.05.04, compressionsLowStart of HoxB8 macrophages, M450, M1 = ctrl & 20um discs, M2 = tko & 20um discs, M3 = tko & glass, M4 = ctrl & glass
# %%%% 22.05.04 one specific cell
dates = '22.05.04'
manips, wells, cells = 2, 1, 8
depthoName = '22.05.04_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.05.04_M1
dates = '22.05.04'
manips, wells, cells = 1, 1, 'all'
depthoName = '22.05.04_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.05.04_M2
dates = '22.05.04'
manips, wells, cells = 2, 1, 'all'
depthoName = '22.05.04_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.05.04_M3
dates = '22.05.04'
manips, wells, cells = 3, 1, 'all'
depthoName = '22.05.04_M3_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.05.04_M4
dates = '22.05.04'
manips, wells, cells = 4, 1, 'all'
depthoName = '22.05.04_M4_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')



# %%% 22.05.03, compressionsLowStart of HoxB8 macrophages, M450, M1 = ctrl & glass, M2 = tko & glass, M3 = tko & 20um discs, M4 = ctrl & 20um discs
# %%%% 22.05.03_M1 C1 Seulement
dates = '22.05.03'
manips, wells, cells = ['1-1'], 1, 1
depthoName = '22.05.03_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.05.03_M1
dates = '22.05.03'
manips, wells, cells = ['1-1', '1-2'], 1, 'all'
depthoName = '22.05.03_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.05.03_M2
dates = '22.05.03'
manips, wells, cells = 2, 1, 'all'
depthoName = '22.05.03_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.05.03_M3
dates = '22.05.03'
manips, wells, cells = 3, 1, 'all'
depthoName = '22.05.03_M3_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.05.03_M4
dates = '22.05.03'
manips, wells, cells = 4, 1, 'all'
depthoName = '22.05.03_M4_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %% Drugs & perturbation


# %%% Next experiment day
# %%%% Next manipe

# %%% 23.09.19 JLY
# M1 - DMSO // M2 - JLY - Jasp 8µM, LatB 5µM, Y27 10µM

# %%%% 23.09.19 M1 - C1 only

dates = '23.09.19'
manips, wells, cells = 1, 1, 4
depthoName = '23.09.19_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'statusFile', redoAllSteps = False, trackAll = False)

# %%%% 23.09.19 M1 - DMSO

dates = '23.09.19'
manips, wells, cells = 1, 1, 'all'
depthoName = '23.09.19_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'statusFile', redoAllSteps = False, trackAll = False)


# %%%% 23.09.19 M2 - JLY - Jasp 8µM, LatB 5µM, Y27 10µM

dates = '23.09.19'
manips, wells, cells = 2, 1, 'all'
depthoName = '23.09.19_M2_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'statusFile', redoAllSteps = False, trackAll = False)

# %%% 23.09.06 CalA low dose
# M1 = DMSO // M2 = 1nM CalA

# %%%% 23.09.11_M1 C1 Seulement


# %%%% 23.09.11_M1


# %%%% 23.09.11_M2


# %%% 23.09.06 CalA low dose
# M1 = 0.25nM CalA // M2 = 0.5 CalA // M3 = dmso // M4 = 2.0 CalA

# %%%% 23.09.06_M1 C1 Seulement

dates = '23.09.06'
manips, wells, cells = 1, 1, 1
depthoName = '23.09.06_M1_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%%% 23.09.06_M1

dates = '23.09.06'
manips, wells, cells = 1, 1, 'all'
depthoName = '23.09.06_M1_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# %%%% 23.09.06_M2

dates = '23.09.06'
manips, wells, cells = 2, 1, 'all'
depthoName = '23.09.06_M2_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# %%%% 23.09.06_M3

dates = '23.09.06'
manips, wells, cells = 3, 1, 'all'
depthoName = '23.09.06_M3_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 23.09.06_M4

dates = '23.09.06'
manips, wells, cells = 4, 1, 'all'
depthoName = '23.09.06_M4_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# %%% 23.07.20
# M1 = 2nM CalA // M2 = ctrl (dmso) // M3 = 1nM CalA

# %%%% 23.07.20_M1 C1 Seulement

dates = '23.07.20'
manips, wells, cells = 1, 1, 1
depthoName = '23.07.20_M1_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%%% 23.07.20_M1

dates = '23.07.20'
manips, wells, cells = 1, 1, 'all'
depthoName = '23.07.20_M1_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# %%%% 23.07.20_M2

dates = '23.07.20'
manips, wells, cells = 2, 1, 'all'
depthoName = '23.07.20_M2_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# %%%% 23.07.20_M3

dates = '23.07.20'
manips, wells, cells = 3, 1, 'all'
depthoName = '23.07.20_M3_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# %%% 23.07.17
# M1 = 1nM CalA // M2 = 0.5nM CalA // M3 = ctrl (dmso)
# M4 = ctrl (no drug) --> M5 = 0.5nM CalA time-dep
# M6 = ctrl (no drug) --> M7 = 2nM CalA time-dep

# %%%% 23.07.17_M1 C1 Seulement

dates = '23.07.17'
manips, wells, cells = 1, 1, 1
depthoName = '23.07.17_M1_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%%% 23.07.17_M1

dates = '23.07.17'
manips, wells, cells = 1, 1, 'all'
depthoName = '23.07.17_M1_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')


# %%%% 23.07.17_M2

dates = '23.07.17'
manips, wells, cells = 2, 1, 'all'
depthoName = '23.07.17_M1_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')


# %%%% 23.07.17_M3

dates = '23.07.17'
manips, wells, cells = 3, 1, 'all'
depthoName = '23.07.17_M1_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%%% 23.07.17_M4

dates = '23.07.17'
manips, wells, cells = 4, 1, 'all'
depthoName = '23.07.17_M1_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')


# %%%% 23.07.17_M5

dates = '23.07.17'
manips, wells, cells = 5, 1, 'all'
depthoName = '23.07.17_M1_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')


# %%%% 23.07.17_M6

dates = '23.07.17'
manips, wells, cells = 6, 1, 'all'
depthoName = '23.07.17_M1_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%%% 23.07.17_M7

dates = '23.07.17'
manips, wells, cells = 7, 1, 'all'
depthoName = '23.07.17_M1_M450_step20_100X'

output = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%% 23.07.06

# %%%% TEST

dates = '23.09.19'
manips, wells, cells = 1, 1, 4
depthoName = '23.09.19_M1_M450_step20_100X'

mainTracker_V3(dates, manips, wells, cells, depthoName, expDf, 
               metaDataFormatting = 'default', redoAllSteps = True, trackAll = False)


# %%%% 23.07.06 Une Cellule Seulement

dates = '23.07.06'
manips, wells, cells = 1, 1, 5
depthoName = '23.07.06_M1-5_M450_step20_100X'

logDf, log_UIxy = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%%% 23.07.06 M1-5

dates = '23.07.06'
manips, wells, cells = [1,2,3,4,5], 1, 'all'
depthoName = '23.07.06_M1-5_M450_step20_100X'

logDf, log_UIxy = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                                 redoAllSteps = True, trackAll = False, 
                                 sourceField = 'default')


# %%%% 23.07.06 M6-8

dates = '23.07.06'
manips, wells, cells = [6,7,8], 1, 'all'
depthoName = '23.07.06_M6-8_M450_step20_100X'

logDf, log_UIxy = mainTracker_V2(dates, manips, wells, cells, depthoName, expDf, 
                                 redoAllSteps = True, trackAll = False, 
                                 sourceField = 'default')


# %%% 23.04.26 + 23.04.28, compressionsLowStart of 3T3-ATCC-2023, M450, 
# 1st day
# M1 = ck666 50µM, M2 = dmso 4µL, M3 = ck666 100µM
# 2nd day
# M1 = dmso 4µL, M2 = ck666 100µM, M3 = ck666 50µM

# %%%% 23.04.26_M1 C1 Seulement

dates = '23.04.26'
manips, wells, cells = 1, 1, 1
depthoName = '23.04.26_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')


# %%%% 23.04.26_M1

dates = '23.04.26'
manips, wells, cells = 1, 1, 'all'
depthoName = '23.04.26_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# %%%% 23.04.26_M2

dates = '23.04.26'
manips, wells, cells = 2, 1, 'all'
depthoName = '23.04.26_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 23.04.26_M3

dates = '23.04.26'
manips, wells, cells = 3, 1, 'all'
depthoName = '23.04.26_M3_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# %%%% 23.04.28_M1

dates = '23.04.28'
manips, wells, cells = 1, 1, 'all'
depthoName = '23.04.28_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# %%%% 23.04.28_M2

dates = '23.04.28'
manips, wells, cells = 2, 1, 'all'
depthoName = '23.04.28_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 23.04.28_M3

dates = '23.04.28'
manips, wells, cells = 3, 1, 'all'
depthoName = '23.04.28_M3_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# %%% 23.04.20, compressionsLowStart of 3T3-ATCC-2023, M450, 
# 1st day
# M1 = dmso, M2 = Fresh blebbi 250uM, M3 = Fresh blebbi 50uM, M4 = dmso, M5 = dmso


# %%%% 23.04.20_M1 C1 Seulement

dates = '23.04.20'
manips, wells, cells = 1, 1, 1
depthoName = '23.04.20_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)


# %%%% 23.04.20_M1

dates = '23.04.20'
manips, wells, cells = 1, 1, 'all'
depthoName = '23.04.20_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)



# %%%% 23.04.20_M2

dates = '23.04.20'
manips, wells, cells = 2, 1, 'all'
depthoName = '23.04.20_M2_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)



# %%%% 23.04.20_M3

dates = '23.04.20'
manips, wells, cells = 3, 1, 'all'
depthoName = '23.04.20_M3_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)


# %%%% 23.04.20_M4

dates = '23.04.20'
manips, wells, cells = 4, 1, 'all'
depthoName = '23.04.20_M4_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)



# %%%% 23.04.20_M5

dates = '23.04.20'
manips, wells, cells = 5, 1, 'all'
depthoName = '23.04.20_M5_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)




# %%% 23.03.16 + 23.03.17, compressionsLowStart of 3T3-ATCC-2023, M450, 
# 1st day
# M1 = dmso 10µL, M2 = Fresh blebbi 50uM,
# 2nd day
# M3 = Fresh blebbi 10uM, M4 = dmso 10µL

# %%%% 23.03.16_M1 C1 Seulement

dates = '23.03.16'
manips, wells, cells = 1, 1, 3
depthoName = '23.03.16_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)



# %%%% 23.03.16_M1

dates = '23.03.16'
manips, wells, cells = 1, 1, 'all'
depthoName = '23.03.16_M1_M450_step20_100X'


output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)



# %%%% 23.03.16_M2

dates = '23.03.16'
manips, wells, cells = 2, 1, 'all'
depthoName = '23.03.16_M2_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)


# %%%% 23.03.17_M3 1 cell only

dates = '23.03.17'
manips, wells, cells = 3, 1, 6
depthoName = '23.03.17_M3_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)


# %%%% 23.03.17_M3

dates = '23.03.17'
manips, wells, cells = 3, 1, 'all'
depthoName = '23.03.17_M3_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)


# %%%% 23.03.17_M4

dates = '23.03.17'
manips, wells, cells = 4, 1, 'all'
depthoName = '23.03.17_M4_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)



# %%% 23.03.08 + 23.03.09, compressionsLowStart of 3T3-ATCC-2023, M450, 
# M1 = Y27 50uM, M2 = Y27 10uM, M3 = No drug, 
# M4 = No drug - next day, M5 = No drug - 15um discs - next day,  M6 = No drug - 25um discs - next day

# %%%% 23.03.08_M1 C1 Seulement

dates = '23.03.08'
manips, wells, cells = 1, 1, 1
depthoName = '23.03.08_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)


# %%%% 23.03.08_M1

dates = '23.03.08'
manips, wells, cells = 1, 1, 'all'
depthoName = '23.03.08_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)


# %%%% 23.03.08_M2

dates = '23.03.08'
manips, wells, cells = 2, 1, 'all'
depthoName = '23.03.08_M2_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%%% 23.03.08_M3

dates = '23.03.08'
manips, wells, cells = 3, 1, 'all'
depthoName = '23.03.08_M3_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%%% 23.03.09_M4

dates = '23.03.09'
manips, wells, cells = 4, 1, [1,2,3,4,5,6,7,8,9]
depthoName = '23.03.09_M4_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%%% 23.03.09_M5

dates = '23.03.09'
manips, wells, cells = 5, 1, 'all'
depthoName = '23.03.09_M4_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%%% 23.03.09_M6

dates = '23.03.09'
manips, wells, cells = 6, 1, 'all'
depthoName = '23.03.09_M4_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%% 23.02.23, compressionsLowStart of 3T3-ATCC-2023, M450, M1 = None, M2 = PNB 5X, M3 = DMSO 5X, M4 = PNB 1X

# %%%% 23.02.23_M1 C1 Seulement

dates = '23.02.23'
manips, wells, cells = 1, 1, 1
depthoName = '23.02.23_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)


# %%%% 23.02.23_M1

dates = '23.02.23'
manips, wells, cells = 1, 1, 'all'
depthoName = '23.02.23_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)


# %%%% 23.02.23_M2

dates = '23.02.23'
manips, wells, cells = 2, 1, 'all'
depthoName = '23.02.23_M2_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%%% 23.02.23_M3

dates = '23.02.23'
manips, wells, cells = 3, 1, 'all'
depthoName = '23.02.23_M3_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%%% 23.02.23_M4

dates = '23.02.23'
manips, wells, cells = 4, 1, 'all'
depthoName = '23.02.23_M4_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)


# %%% 23.02.16, compressionsLowStart of 3T3-ATCC-2023, M450, M1 = DMSO, M2 = Blebbi 1X

# %%%% 23.02.16_M1 C1 Seulement

dates = '23.02.16'
manips, wells, cells = 1, 1, 13
depthoName = '23.02.16_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)


# %%%% 23.02.16_M1

dates = '23.02.16'
manips, wells, cells = 1, 1, 'all'
depthoName = '23.02.16_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)


# %%%% 23.02.16_M2

dates = '23.02.16'
manips, wells, cells = 2, 1, 'all'
depthoName = '23.02.16_M2_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)


# %%% 22.11.23, compressionsLowStart of 3T3 LG +++, M450, M1 = DMSO, M2 = LatA - 5x, M3 = DMSO
# %%%% 22.11.23_M1 C1 Seulement
dates = '22.11.23'
manips, wells, cells = 1, 1, 1
depthoName = '22.11.23_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = True, trackAll = False)

# %%%% 22.11.23_M1
dates = '22.11.23'
manips, wells, cells = 1, 1, 'all'
depthoName = '22.11.23_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%%% 22.11.23_M2
dates = '22.11.23'
manips, wells, cells = 2, 1, 'all'
depthoName = '22.11.23_M2_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%%% 22.11.23_M3
dates = '22.11.23'
manips, wells, cells = 3, 1, 'all'
depthoName = '22.11.23_M3_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)



# %%% 22.03.30, compressionsLowStart of 3T3 LG +++, M450, M1 = Blebbi, M2 = LatA, M3 = Ctrl, M4 = DMSO
# %%%% 22.03.30_M1 C1 Seulement
dates = '22.03.30'
manips, wells, cells = 1, 1, 1
depthoName = '22.03.30_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%%% 22.03.30_M1
dates = '22.03.30'
manips, wells, cells = 1, 1, 'all'
depthoName = '22.03.30_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%%% 22.03.30_M2
dates = '22.03.30'
manips, wells, cells = 2, 1, 'all'
depthoName = '22.03.30_M2_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%%% 22.03.30_M3
dates = '22.03.30'
manips, wells, cells = 3, 1, 'all'
depthoName = '22.03.30_M3_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%%% 22.03.30_M4
dates = '22.03.30'
manips, wells, cells = 4, 1, 'all'
depthoName = '22.03.30_M4_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%% 22.03.28, compressionsLowStart of 3T3 LG +++, M450, M1 = DMSO, M2 = Blebbi, M3 = LatA
# %%%% 22.03.28_M1 C1 Seulement
dates = '22.03.28'
manips, wells, cells = 1, 1, 1
depthoName = '22.03.28_M1_M450_step20_100X'

# output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
#                      redoAllSteps = False, trackAll = False, 
#                      sourceField = 'default')

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%%% 22.03.28_M1
dates = '22.03.28'
manips, wells, cells = 1, 1, 'all'
depthoName = '22.03.28_M1_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%%% 22.03.28_M2
dates = '22.03.28'
manips, wells, cells = 2, 1, 'all'
depthoName = '22.03.28_M2_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)

# %%%% 22.03.28_M3
dates = '22.03.28'
manips, wells, cells = 3, 1, 'all'
depthoName = '22.03.28_M3_M450_step20_100X'

output = mainTracker_V4(dates, manips, wells, cells, depthoName, expDf, 
                        metaDataFormatting = 'loopStruct', redoAllSteps = False, trackAll = False)



# %% Mechanics & Non-linearity

# %%% Next experiment day
# %%%% Next manipe

# %%% 22.03.21, >>>> SINUS <<<< on 3T3 LG+++, M3, M450, various freq and ampli
# %%%% Test !
dates = '22.03.21'
manips, wells, cells = 3, 1, 1
depthoName = '22.03.21_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%% 22.03.21, >>>> BROKEN RAMP <<<< on 3T3 LG+++, M4, M450, 3 comp
# %%%% Test !
dates = '22.03.21'
manips, wells, cells = 4, 1, 3
depthoName = '22.03.21_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%% 22.02.09, compressionsLowStart of 3T3, M1 = M450, M2 = M450
# %%%% 22.02.09_M1 C1 Seulement
dates = '22.02.09'
manips, wells, cells = 1, 1, 1
depthoName = '22.02.09_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.02.09_M1
dates = '22.02.09'
manips, wells, cells = 1, 1, 'all'
depthoName = '22.02.09_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.01.12_M2
dates = '22.02.09'
manips, wells, cells = 2, 1, 'all'
depthoName = '22.02.09_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# C3 and C6 are shit xy analysis due to chainof beads plus X motion -> corrected, allowed me to do a LOT of debugging :)

# %%%% 22.01.12 _ Only a few cells
dates = '22.02.09'
manips, wells, cells = 1, 1, [7]
depthoName = '22.02.09_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# trackAll = False works even for NB = 4 -> Corrected, I think !
# trackAll = True seems to work for NB = 2 but do not when NB = 4



# %%%% 22.01.12_M3
dates = '22.02.09'
manips, wells, cells = 3, 1, 'all'
depthoName = '22.02.09_M3_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# %%% 22.01.12, compressionsLowStart of 3T3, M1 = M270, M2 = M450, M4 = M450, pas de M3
# %%%% 22.01.12_M1 C1 Seulement
dates = '22.01.12'
manips, wells, cells = 1, 1, 1
depthoName = '22.01.12_M1_M270_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.01.12_M1
dates = '22.01.12'
manips, wells, cells = 1, 1, 'all'
depthoName = '22.01.12_M1_M270_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.01.12_M2
dates = '22.01.12'
manips, wells, cells = 2, 1, 'all'
depthoName = '22.01.12_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# %%%% 22.01.12_M4
dates = '22.01.12'
manips, wells, cells = 4, 1, 'all'
depthoName = '22.01.12_M4_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%% 21.12.16, compressions of 3T3, M1 = M450, M2 = M270
# %%%% 21.12.16_M1 C1 Seulement
dates = '21.12.16'
manips, wells, cells = 1, 1, 9
depthoName = '21.12.16_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')


# %%%% 21.12.16_M1
dates = '21.12.16'
manips, wells, cells = 1, 1, 'all'
depthoName = '21.12.16_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.12.16_M2
dates = '21.12.16'
manips, wells, cells = 2, 1, 'all'
depthoName = '21.12.16_M2_M270_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# %%% 21.12.08, compressions of 3T3, M1 = M270, M2 = M450
# %%%% 21.12.08_M1 C1 Seulement
dates = '21.12.08'
manips, wells, cells = 1, 2, 4
depthoName = '21.12.08_M1_M270_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')


# %%%% 21.12.08_M1
dates = '21.12.08'
manips, wells, cells = 1, 2, 'all'
depthoName = '21.12.08_M1_M270_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.12.08_M2
dates = '21.12.08'
manips, wells, cells = 2, 1, 'all'
depthoName = '21.12.08_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# %%% 21.10.25, compressions of 3T3, M1 = M450, M2 = M270
# %%%% 21.10.25_M1 C1 Seulement
dates = '21.10.25'
manips, wells, cells = 1, 1, 1
depthoName = '21.10.25_M1_M450_100X_step20'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.10.25_M1
dates = '21.10.25'
manips, wells, cells = 1, 1, 'all'
depthoName = '21.10.25_M1_M450_100X_step20'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.10.25_M2
dates = '21.10.25'
manips, wells, cells = 2, 1, 'all'
depthoName = '21.10.25_M2_M270_100X_step20'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%% 21.10.18, compressions of 3T3, M1 = M270, M2 = M450
# %%%% 21.10.18_M1
dates = '21.10.18'
manips, wells, cells = 1, 1, 'all'
depthoName = '21.10.18_M1_M270_100X_step20'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.10.18_M2
dates = '21.10.18'
manips, wells, cells = 2, 1, 'all'
depthoName = '21.10.18_M2_M450_100X_step20'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')




# %% MCA project

# %%% Next experiment day
# %%%% Next manipe

# %%% 22.07.27, compressionsLowStart of 3T3aSFL - M1 = F8 ctrl ; M2 = F8 doxy ; M3 = E4 ctrl ; M4 = E4 doxy
# %%%% 22.07.27 one specific cell
dates = '22.07.27'
manips, wells, cells = 3, 1, [2, 5]
depthoName = '22.07.27_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.07.27_M1
dates = '22.07.27'
manips, wells, cells = 1, 1, 'all'
depthoName = '22.07.27_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.07.27_M2
dates = '22.07.27'
manips, wells, cells = 2, 1, 'all'
depthoName = '22.07.27_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.07.27_M3
dates = '22.07.27'
manips, wells, cells = 3, 1, 'all'
depthoName = '22.07.27_M3_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.07.27_M4
dates = '22.07.27'
manips, wells, cells = 4, 1, 'all'
depthoName = '22.07.27_M4_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%% 22.07.20, compressionsLowStart of 3T3aSFL - M1 = A11 doxy ; M2 = A11 ctrl ; M3 = E4 doxy ; M4 = E4 ctrl
# %%%% 22.07.20 one specific cell
dates = '22.07.20'
manips, wells, cells = 1, 1, 1
depthoName = '22.07.20_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.07.20_M1
dates = '22.07.20'
manips, wells, cells = 1, 1, 'all'
depthoName = '22.07.20_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.07.20_M2
dates = '22.07.20'
manips, wells, cells = 2, 1, 'all'
depthoName = '22.07.20_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.07.20_M3
dates = '22.07.20'
manips, wells, cells = 3, 1, 'all'
depthoName = '22.07.20_M3_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.07.20_M4
dates = '22.07.20'
manips, wells, cells = 4, 1, 'all'
depthoName = '22.07.20_M4_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# %%% 22.07.15, compressionsLowStart of 3T3aSFL - M1 = F8 doxy ; M2 = F8 ctrl ; M3 = A11 doxy ; M4 = A11 ctrl
# %%%% 22.07.15 one specific cell
dates = '22.07.15'
manips, wells, cells = 1, 1, 1
depthoName = '22.07.15_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = True, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.07.15_M1
dates = '22.07.15'
manips, wells, cells = 1, 1, 'all'
depthoName = '22.07.15_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.07.15_M2
dates = '22.07.15'
manips, wells, cells = 2, 1, 'all'
depthoName = '22.07.15_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.07.15_M3
dates = '22.07.15'
manips, wells, cells = 3, 1, 'all'
depthoName = '22.07.15_M3_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 22.07.15_M4
dates = '22.07.15'
manips, wells, cells = 4, 1, 'all'
depthoName = '22.07.15_M4_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%% 21.09.09, compressions of 3T3aSFL-A8-2, M450, M1 = doxy, M2 = control
# %%%% 21.09.09_M1
dates = '21.09.09'
manips, wells, cells = 1, 1, 'all'
depthoName = '21.09.09_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.09.09_M2
dates = '21.09.09'
manips, wells, cells = 2, 1, 'all'
depthoName = '21.09.09_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%% 21.09.08, compressions of 3T3aSFL-6FP-2, M450, M1 & M4 = doxy, M2 & M3 = control
# %%%% 21.09.08_M1
dates = '21.09.08'
manips, wells, cells = 1, 1, 'all'
depthoName = '21.09.08_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.09.08_M2
dates = '21.09.08'
manips, wells, cells = 2, 1, 'all'
depthoName = '21.09.08_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.09.08_M3
dates = '21.09.08'
manips, wells, cells = 3, 1, 'all'
depthoName = '21.09.08_M3_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.09.08_M4
dates = '21.09.08'
manips, wells, cells = 4, 1, 'all'
depthoName = '21.09.08_M4_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default') 



# %%% 21.09.02, compressions of 3T3aSFL-6FP-2, M450, M1 = smifh2, M2 = dmso, M3 = smifh2+doxycyclin, M4 = dmso+doxycyclin
# %%%% 21.09.02_M1
dates = '21.09.02'
manips, wells, cells = 1, 1, 'all'
depthoName = '21.09.02_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.09.02_M2
dates = '21.09.02'
manips, wells, cells = 2, 1, 'all'
depthoName = '21.09.02_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.09.02_M3
dates = '21.09.02'
manips, wells, cells = 3, 1, 'all'
depthoName = '21.09.02_M3_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.09.02_M4
dates = '21.09.02'
manips, wells, cells = 4, 1, 'all'
depthoName = '21.09.02_M4_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%% 21.09.01, compressions of 3T3aSFL with drugs, M450, M1 = dmso, M2 = smifh2
# %%%% 21.09.01_M1
dates = '21.09.01'
manips, wells, cells = 1, 1, 'all'
depthoName = '21.09.01_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.09.01_M2
dates = '21.09.01'
manips, wells, cells = 2, 1, 'all'
depthoName = '21.09.01_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%% 21.04.28, constant field of 3T3aSFL-6FP, M450, M1 = doxy, M2 = control
# %%%% 21.04.28_M1

dates = '21.04.28'
manips, wells, cells = 1, 1, 'all'
depthoName = '21.04.28_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.04.28_M2

dates = '21.04.28'
manips, wells, cells = 2, 1, 'all'
depthoName = '21.04.28_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%% 21.04.27, constant field of 3T3aSFL-6FP, M450, M1 = control, M2 = doxy
# %%%% 21.04.27_M1

dates = '21.04.27'
manips, wells, cells = 1, 1, 'all'
depthoName = '21.04.27_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.04.27_M2

dates = '21.04.27'
manips, wells, cells = 2, 1, 'all'
depthoName = '21.04.27_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%% 21.04.23, constant field of 3T3aSFL, M450, M1 = doxy, M2 = control
# %%%% 21.04.23_M1

dates = '21.04.23'
manips, wells, cells = 1, 1, 'all'
depthoName = '21.04.23_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.04.23_M2

dates = '21.04.23'
manips, wells, cells = 2, 1, 'all'
depthoName = '21.04.23_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%% 21.04.21, constant field of 3T3aSFL, M450, M1 = control, M2 = doxy
# %%%% 21.04.21_M1

dates = '21.04.21'
manips, wells, cells = 1, 1, [1, 2, 3, 4, 5]
depthoName = '21.04.21_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.04.21_M2

dates = '21.04.21'
manips, wells, cells = 2, 1, 2
depthoName = '21.04.21_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%% 21.02.15, constant field of 3T3aSFL, M450, M1 = control, M2 = doxy

# %%%% 21.02.15_M1_C3

dates = '21.02.15'
manips, wells, cells = 1, 1, 3
depthoName = '21.02.15_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.02.15_M1

dates = '21.02.15'
manips, wells, cells = 1, 1, 'all'
depthoName = '21.02.15_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.02.15_M2

dates = '21.02.15'
manips, wells, cells = 2, 1, 'all'
depthoName = '21.02.15_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.02.15_M3

dates = '21.02.15'
manips, wells, cells = 3, 1, 'all'
depthoName = '21.02.15_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%% 21.02.10, constant field of 3T3aSFL, M450, M1 = control, M2 = doxy

# %%%% 21.02.10_M1_C1

dates = '21.02.10'
manips, wells, cells = 1, 1, 1
depthoName = '21.02.10_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.02.10_M1

dates = '21.02.10'
manips, wells, cells = 1, 1, 'all'
depthoName = '21.02.10_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.02.10_M2

dates = '21.02.10'
manips, wells, cells = 2, 1, 'all'
depthoName = '21.02.10_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')



# %%% 21.01.21, compressions of 3T3aSFL, M450, M1 = doxy, M2 = control, M3 = doxy

# %%%% 21.01.21_M1 - juste une cellule pour commencer
dates = '21.01.21'
manips, wells, cells = 2, 1, [9]
depthoName = '21.01.21_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')


# %%%% 21.01.21_M1
dates = '21.01.21'
manips, wells, cells = 1, 1, 'all'
depthoName = '21.01.21_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.01.21_M2
dates = '21.01.21'
manips, wells, cells = 2, 1, 'all'
depthoName = '21.01.21_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.01.21_M3
dates = '21.01.21'
manips, wells, cells = 3, 1, 'all'
depthoName = '21.01.21_M3_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%% 21.01.18, compressions of 3T3aSFL, M450, M1 = control, M2 = doxy, M3 = control
# %%%% 21.01.18_M1
dates = '21.01.18'
manips, wells, cells = ['1-1', '1-2'], 1, 'all'
depthoName = '21.01.18_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.01.18_M1
dates = '21.01.18'
manips, wells, cells = ['1-2'], 1, 5
depthoName = '21.01.18_M1_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.01.18_M2
dates = '21.01.18'
manips, wells, cells = 2, 1, 'all'
depthoName = '21.01.18_M2_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')

# %%%% 21.01.18_M3
dates = '21.01.18'
manips, wells, cells = 3, 1, 'all'
depthoName = '21.01.18_M3_M450_step20_100X'

output = mainTracker(dates, manips, wells, cells, depthoName, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')



















