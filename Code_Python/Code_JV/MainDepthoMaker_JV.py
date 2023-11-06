# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:20:48 2021

@author: Joseph Vermeil
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
sys.path.append(cp.DirRepoPythonUser)

import GraphicStyles as gs
import GlobalConstants as gc
import UtilityFunctions as ufun

from BeadTracker_V3 import depthoMaker

# 2. Pandas settings
pd.set_option('mode.chained_assignment',None)

# 3. Graphical settings
gs.set_default_options_jv()


# %% Next depthos !

# %% Depthos 23.09.21 - CALIBRATION

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.09.21'

subdir = 'depthos'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_CalibrationM450-2025_Try02', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_CALIBRATION_M450-2025-BSA_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, plot = 0) # , d = 'HD'


# %% Depthos 23.09.19

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.09.19'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, plot = 2)

# %%

subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, plot = 2)


# %% Depthos 23.09.11

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.09.11'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% Depthos 23.09.08 - CALIBRATION M450-mPEG_Try02

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.09.08'

subdir = 'Depthos'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_CalibrationM450-mPEG_Try02', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_CALIBRATION_M450-mPEG_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% Depthos 23.09.06

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.09.06'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M4'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M4_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


# %% Depthos 23.09.05 - CALIBRATION

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.09.05'

subdir = 'Bare_depthos'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_CalibrationM450-2025_Try01', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_CALIBRATION_M450-2025-Bare_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'Fibro_depthos'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_CalibrationM450-2025_Try01', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_CALIBRATION_M450-2025-Fibro_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)




# %% Depthos 23.07.20

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.07.20'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% Depthos 23.07.17

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.07.17'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M4-5'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M4-5_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M6-7'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M6-7_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% Depthos 23.07.06

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.07.06'

subdir = 'M1-5'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1-5_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M6-8'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M6-8_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% Depthos from 23.04.28 3T3-ATCC 2023 expt

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.04.28'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


# %% Depthos from 23.04.26 3T3-ATCC 2023 expt

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.04.26'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


# %% Depthos from 23.04.20 3T3-ATCC 2023 expt

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.04.20'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M4'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M4_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M5'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M5_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% Depthos from 23.03.16 & 23.03.17 3T3-ATCC 2023 expt

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.03.16'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.03.17'

subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M4'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M4_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% Depthos from 23.03.08 & 23.03.09 3T3-ATCC 2023 expt

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.03.08'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.03.09'
subdir = 'M4'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M4_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% Depthos from 23-02-23 3T3-ATCC 2023 expt

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.02.23'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M4'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M4_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



# %% Depthos from 23-02-16 Blebbi 3T3-ATCC 2023 expt

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '23.02.16'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% Depthos from 22-11-23 LatA expt


DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.11.23'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)






# %% All depthos from old DC experiments by Valentin

# %%% 18.18.28 -> Valid also for 18-09-24 and 18-09-25

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '18.08.28'

DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho')
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm # CHECK IT'S CORRECT !!!!!

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %%% 18.10.30

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '18.10.30'

DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho')
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm # CHECK IT'S CORRECT !!!!!

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %%% 18.12.12

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '18.12.12'

DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho')
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm # CHECK IT'S CORRECT !!!!!

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)




# %% All depthos from 22.07.27 Long Linker experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.07.27'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M4'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M4_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% All depthos from 22.07.20 Long Linker experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.07.20'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M4'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M4_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% All depthos from 22.07.15 Long Linker experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.07.15'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M4'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M4_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)





# %% All depthos from 22.05.05 HoxB8 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.05.05'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.05.05'

subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% All depthos from 22.05.04 HoxB8 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.05.04'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.05.04'

subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.05.04'

subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.05.04'

subdir = 'M4'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M4_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% All depthos from 22.05.03 HoxB8 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.05.03'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.05.03'

subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.05.03'

subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.05.03'

subdir = 'M4'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M4_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% All depthos from 22.03.30 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.03.30'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M4'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M4_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% All depthos from 22.03.38 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.03.28'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)




# %% Depthos of new M450 for calibration - 22.04.29

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.04.29'

subdir = 'Depthos'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_CalibrationM450-2023_SecondTry', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_CALIBRATION_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


# %% Depthos of new M450 for calibration - 22.03.21

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.03.21'

subdir = 'Depthos'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_CalibrationM450-2023_FirstTry', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_CALIBRATION_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% All depthos from 22.03.21 3T3 experiments -> M1 is valid also for M3 & M4

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.03.21'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



# %% All depthos from 22.02.09 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.02.09'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% All depthos from 22.01.12 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '22.01.12'

subdir = 'M1_M270'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M270'
saveLabel = date + '_M1_M270_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M2_M450'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M4_M450'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M4_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)




# %% All depthos from 21.12.16 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '21.12.16'

subdir = 'M1_M450'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M2_M270'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M270'
saveLabel = date + '_M2_M270_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)





# %% All depthos from 21.12.08 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '21.12.08'

subdir = 'M1_M270'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M270'
saveLabel = date + '_M1_M270_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'M2_M450'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


# %% All depthos from 21.09.09 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '21.09.09'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


# %% All depthos from 21.09.08 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '21.09.08'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'M4'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M4_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)




# %% All depthos from 21.09.02 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '21.09.02'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'M4'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M4_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



# %% All depthos from 21.09.01 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '21.09.01'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% All depthos from 21.04.28 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '21.04.28'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


# %% All depthos from 21.04.27 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '21.04.27'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


# %% All depthos from 21.04.23 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '21.04.23'

subdir = 'Deptho_M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'Deptho_M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



# %% All depthos from 21.04.21 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '21.04.21'

subdir = 'Deptho_M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'Deptho_M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% All depthos from 21.02.15 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '21.02.15'

subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



# %% All depthos from 21.02.10 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '21.02.10'

subdir = 'Deptho_M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'Deptho_M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)




 
# %% All depthos from 21.01.21 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
date = '21.01.21'

subdir = 'Deptho_M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'Deptho_M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'Deptho_M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

# %% All depthos from 21.01.18 3T3 experiments

DirDataRaw = 'D://MagneticPincherData//Raw'


date = '21.01.18'
subdir = 'Deptho_M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'Deptho_M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)



subdir = 'Deptho_M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)