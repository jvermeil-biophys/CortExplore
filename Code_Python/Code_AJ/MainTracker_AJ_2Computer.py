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

from BeadTracker_2Computer import mainTracker
from BeadTracker_2Computer import XYZtracking


# 2. Pandas settings
# pd.set_option('mode.chained_assignment',None)
# 
# 3. Graphical settings
# gs.set_default_options_jv()


# 4. Import of the experimental conditions

expDf = ufun.getExperimentalConditions(DirExp = cp.DirRepoExp, save = True, suffix = cp.suffix)

#%% First experiment on confocal 

# %%%% M1 : 

dates = '23.09.21'
manips, wells, cells = 'all', 'all', 'all'
depthoNames = '23.09.21_P1_M450_step20_100X'
  
output = mainTracker(dates, manips, wells, cells, depthoNames, expDf, 
                     redoAllSteps = False, trackAll = False, 
                     sourceField = 'default')