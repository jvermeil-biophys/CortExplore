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

from BeadTracker import depthoMaker

# 2. Pandas settings
pd.set_option('mode.chained_assignment',None)

# 3. Graphical settings
gs.set_default_options_jv()


# %% Next depthos !

# %% Next depthos !

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