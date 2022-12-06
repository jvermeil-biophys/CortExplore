# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 12:06:29 2022

@author: JosephVermeil
"""

# %% (0) Imports and settings

# 1. Imports
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import statsmodels.api as sm

import os
import re
import time
import shutil
import pyautogui
import matplotlib
import traceback
# import cv2

# import scipy
from scipy import interpolate
from scipy import signal

# import skimage
from skimage import io, filters, exposure, measure, transform, util, color
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import linear_sum_assignment
from matplotlib.gridspec import GridSpec
from datetime import date, datetime

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)

import GraphicStyles as gs
import GlobalConstants as gc
import UtilityFunctions as ufun

# 2. Pandas settings
pd.set_option('mode.chained_assignment', None)

# 3. Plot settings
gs.set_default_options_jv()


# %% (1) Functions

def Image3DCapture(srcPath='', dstDir='', improveContrast = True, dstExt = '.png', indexCapture = 'Auto'):
        
    Isrc = io.imread(srcPath)
    if not len(Isrc.shape) == 3:
        print(gs.BRIGHTRED + 'Invalid image format : not 3D !' + gs.NORMAL)
        
    N, ny, nx = Isrc.shape
    
    if indexCapture == 'Auto':
        ic = N//2
    else:
        ic = indexCapture
        
    Idst = Isrc[ic, :, :]
    
    if improveContrast:
        p2, p98 = np.percentile(Idst, (2, 98))
        Idst = exposure.rescale_intensity(Idst, in_range=(p2, p98))
    
    srcName = os.path.split(srcPath)[-1]
    splitName = srcName.split('.')
    dstName = ''.join(splitName[:-1]) + dstExt
    dstPath = os.path.join(dstDir, dstName) 
    
    io.imsave(dstPath, Idst)
    


# ufun.copyFile(DirSrc, DirDst, filename)
# ufun.copyFolder(DirSrc, DirDst, folderName)
    
def mainWrapper(dates = [], dstDir = '', 
                Timeseries = True, FluoImg = False, UMS = False):
    ReadMe_path = os.path.join(dstDir, 'Readme.txt')
    ReadMeText = "TITLE\n\n"
    ReadMeText += "Actin Cortex Analysis *** DataSet Description\n\n"
    Npart = 0
    
    if not os.path.exists(dstDir):
        os.makedirs(dstDir)
    
    dates2 = []
    for d in dates:
        YY, MM, DD = d.split('-')
        d2 = ('' + YY + '.' + MM + '.' + DD)
        dates2.append(d2)
    
    #### ExperimentalConditions
    Npart += 1
    ReadMeText += str(Npart) + ". ExperimentalConditions\n"
    ReadMeText += "This ExperimentalConditions.csv file contains a detailed list of the experimental conditions.\n"
    ReadMeText += "Some field are only useful for the algorithm. The interesting ones for the human reader are mostly:\n"
    ReadMeText += "tags, date, manip, experimentType, drug, substrate, cell type, cell subtype.\n"
    ReadMeText += "\n"
    
    DirDataExp = cp.DirRepoExp
    expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
    subExpDf = expDf[expDf['date'].apply(lambda x : x in dates)]
    subExpDf_path = os.path.join(dstDir, 'ExperimentalConditions.csv')
    subExpDf.to_csv(subExpDf_path, sep = ';')
    
    #### BrightFieldCaptures
    Npart += 1
    ReadMeText += str(Npart) + ". BrightFieldCaptures\n"
    ReadMeText += "This folder contains one single time point frame for each analysed cell, in .png format.\n"
    ReadMeText += "The idea is just to have a trace of the cell morphology and state.\n"
    ReadMeText += "In general, the frame was simply taken in the middle of the timelapse.\n"
    ReadMeText += "\n"
    
    BF_path = subExpDf_path = os.path.join(dstDir, 'BrightFieldCaptures')
    if not os.path.exists(BF_path):
        os.makedirs(BF_path)
        
    listFolders = [os.path.join(cp.DirDataRaw, d2) for d2 in dates2]
    for F in listFolders:
        listFiles = os.listdir(F)
        for f in listFiles:
            if f.endswith('.tif'):
                f_path = os.path.join(F, f)
                # Image3DCapture(srcPath=f_path, dstDir=BF_path)
    
    
    #### FluoCaptures
    if FluoImg:
        Npart += 1
        ReadMeText += str(Npart) + ". FluoFilms\n"
        ReadMeText += "This folder contains all fluo images captured during the experiment, in .tif format.\n"
        ReadMeText += "\n"
        
        Fluo_path = os.path.join(dstDir, 'FluoFilms')
        if not os.path.exists(Fluo_path):
            os.makedirs(Fluo_path)
            
        listFolders = [os.path.join(cp.DirDataRaw, d2 + '_Fluo') for d2 in dates2]
        for F in listFolders:
            if os.path.exists(F):
                listSubFolders = os.listdir(F)
                for subF in listSubFolders:
                    # pass
                    # copyFolder(DirSrc, DirDst, folderName)
                    ufun.copyFolder(F, Fluo_path, subF)
    
    #### Timeseries
    if Timeseries:
        Npart += 1
        ReadMeText += str(Npart) + ". Timeseries\n"
        ReadMeText += "This folder contains all Timeseries files, in .csv format.\n"
        ReadMeText += "These files are the building blocks of the quantitative analysis.\n"
        ReadMeText += "For each analyzed pair of beads, they contains the following data:\n"
        ReadMeText += "idxAnalysis: A field useful for the selection of the data. Its value is 0 when not compressing; -i for the 'precompression' n°i; +i for the compression n°i.\n"
        ReadMeText += "T: Time in seconds, with t0 being the beginning of this series of compressions.\n"
        ReadMeText += "Tabs: Absolute time in seconds, the raw info saved by the computer during the experiments. Useful to test if the state of the cells changes during a whole day of experiments.\n"
        ReadMeText += "B: Applied external magnetic field, in mT.\n"
        ReadMeText += "F: Applied pinching force, in pN. Computed using B, D3 and dx.\n"
        ReadMeText += "dx: Distance between the beads centers along the x-axis, in µm.\n"
        ReadMeText += "dy: Distance between the beads centers along the y-axis, in µm.\n"
        ReadMeText += "dz: Distance between the beads centers along the z-axis, in µm. Harder to compute than the other two, hence noisier.\n"
        ReadMeText += "D2: Distance between the beads centers in the xy-plane, in µm.\n"
        ReadMeText += "D3: Distance between the beads centers in 3D, in µm.\n"
        ReadMeText += "\n"
        ReadMeText += "NB: to find H the cortex thickness, one need to take D3 and substract the diameter of the beads.\n"
        ReadMeText += "D = 4.477µm for these data.\n"
        ReadMeText += "\n"
    
        TS_path = os.path.join(dstDir, 'Timeseries')
        if not os.path.exists(TS_path):
            os.makedirs(TS_path)
            
        listTSFiles = os.listdir(cp.DirDataTimeseries)
        for f in listTSFiles:
            if (f[:8] in dates) and (f.endswith('.csv')):
                # pass
                # copyFile(DirSrc, DirDst, filename)
                ufun.copyFile(cp.DirDataTimeseries, TS_path, f)
    
    #### UserManualSelection
    if UMS:
        Npart += 1
        ReadMeText += str(Npart) + ". UserManualSelection\n"
        ReadMeText += "This folder contains .csv files used to accept or refuse each compression.\n"
        ReadMeText += "For each compression, UI_Valid, which is True by default, can be set to False.\n"
        ReadMeText += "If set to False, the reason will be indicated in UI_comments.\n"
        ReadMeText += "\n"
        
        UMS_path = os.path.join(dstDir, 'UserManualSelection')
        if not os.path.exists(UMS_path):
            os.makedirs(UMS_path)
            
        listUMSFiles = os.listdir(cp.DirDataAnalysisUMS)
        for f in listUMSFiles:
            if (f[:8] in dates) and (f.endswith('.csv')):
                # pass
                # copyFile(DirSrc, DirDst, filename)
                ufun.copyFile(cp.DirDataAnalysisUMS, UMS_path, f)
                
    if not os.path.exists(ReadMe_path):
        ReadMe_file = open(ReadMe_path, 'w')
        ReadMe_file.write(ReadMeText)
        ReadMe_file.close()
    
    
    



# %% (2) Scripts

# %%% (2.1) HoxB8 dataBase for RP & PV

dates = ['22-05-03','22-05-04','22-05-05']
dstDir = 'C://Users//JosephVermeil//Desktop//HoxB8_Cortex_DataSet'

mainWrapper(dates = dates, dstDir = dstDir, 
            Timeseries = True, FluoImg = True, UMS = True)






