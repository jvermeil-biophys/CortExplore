# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:20:48 2021

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

from BeadTracker import depthoMaker
# from BeadTracker_2Computer import depthoMaker


# 2. Pandas settings
pd.set_option('mode.chained_assignment',None)

# 3. Graphical settings
# gs.set_default_options_jv()

# 6. Others
SCALE_100X = 15.8 # pix/µm
SCALE_Zen_100X = 7.4588 # pix/µm
SCALE_63X = 9.9 # pix/µm


#%% Deptho from experiment 24-10-21 - First mechanics experiment with optoRhoA-VB(++) on Chameleon

mainDirPath = 'D:/Anumita/MagneticPincherData/Raw/'

date = '24.10.21'

subdir = 'Deptho_P1'
depthoPath = os.path.join(mainDirPath, date + '_Deptho', subdir)
savePath = os.path.join(mainDirPath, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date +'_P1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = SCALE_100X # pix/µm

depthoMaker(depthoPath, savePath, specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'Deptho_P3'
depthoPath = os.path.join(mainDirPath, date + '_Deptho', subdir)
savePath = os.path.join(mainDirPath, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date +'_P3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = SCALE_100X # pix/µm

depthoMaker(depthoPath, savePath, specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)
