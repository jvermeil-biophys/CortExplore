# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:50:53 2021

@author: Anumita Jawahar
"""

# %% General imports

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

import GraphicStyles as gs
import GlobalConstants as gc
import UtilityFunctions as ufun

from BeadTracker_V3 import mainTracker_V3
from BeadTracker_V4 import mainTracker_V4

from BeadTracker import XYZtracking


# 2. Pandas settings
# pd.set_option('mode.chained_assignment',None)
# 
# 3. Graphical settings
# gs.set_default_options_jv()


# 4. Import of the experimental conditions

expDf = ufun.getExperimentalConditions(DirExp = cp.DirRepoExp, save = True, suffix = cp.suffix)


#%% Deptho from experiment 24-10-26 - First mechanics experiment 3T3 optoRhoAVB (++) on Chameleon

# %%%% M1 : 

dates = '24.10.21'
manips, wells, cells = 2, 1, 'all'
depthoNames = '24.10.21_P'+str(1)+'_M450_step20_100X'

mainTracker_V4(dates, manips, wells, cells, depthoNames, expDf, 
                                 redoAllSteps = True, trackAll = False)

#%% Deptho from experiment 24-11-13 - LE.E5 and HB.B5 (mostly)

# %%%% M1 : 

dates = '24.11.13'
manips, wells, cells = 3, 1, 'all'
depthoNames = '24.11.13_P'+str(wells)+'_M450_step20_100X'

mainTracker_V4(dates, manips, wells, cells, depthoNames, expDf, 
                                 redoAllSteps = False, trackAll = False)

#%% Deptho from experiment 24-11-15 - LE.E2 

# %%%% M1 : 

dates = '24.11.15'
manips, wells, cells = 4, 1, 'all'
depthoNames = '24.11.15_P'+str(wells)+'_M450_step20_100X'

mainTracker_V4(dates, manips, wells, cells, depthoNames, expDf, 
                                 redoAllSteps = False, trackAll = False)