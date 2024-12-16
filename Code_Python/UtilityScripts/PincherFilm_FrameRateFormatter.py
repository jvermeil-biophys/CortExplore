# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:36:19 2024

@author: JosephVermeil
"""

#%% Import

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import re
import datetime as dt
from datetime import date
import sys
from skimage import io
import CortexPaths as cp
import TrackAnalyser_V3 as taka


import scipy.stats as st
import tifffile as tiff
import cv2
import GraphicStyles as gs

#%% Code to create videos with a decently constant frame rate with a timestamp

pathSave = 'C:/Users/JosephVermeil/Desktop/Manuscrit/Soutenance/Films/MagPincher_Indent'
path = 'D:/MagneticPincherData/Raw/FilmsToConvert/'
allFiles = os.listdir(path)
date = '24-04-11'
allTifs = [i for i in allFiles if date in i and '.tif' in i]
allFields = [i for i in allFiles if date in i and '_Field' in i]
allLogs = [i for i in allFiles if date in i and '_LogPY' in i]

activationFrames = np.asarray([])

for i in range(len(allTifs)):
    selectedFrames = []
    tif = allTifs[i]
    file = allFields[i]
    log = allLogs[i]

    print(tif)
    field = np.loadtxt(os.path.join(path, file), delimiter = '\t')
    status = pd.read_csv(os.path.join(path, log), delimiter = '\t')

    ctfield = status['iS'][status['idx_inNUp'] == 2.0].values
    selectedFrames.extend(ctfield)

    comp = status['iS'][status['Status'] == 'Action_main'].values
    comp_correct_rate = comp[::25]
    others = status['iS'][status['Status'] == 'Action'].values
    others_correct_rate = others[::6]
    
    all_correct_rate = np.concatenate((ctfield, comp_correct_rate, others_correct_rate))
    all_correct_index = np.sort(all_correct_rate) - 1
    
    
    # compId = np.asarray(np.linspace(0,199,5), dtype = 'int')
    # for j in range(len(comp)//200):
    #     selectedComp = comp[j*200 : (j+1)*200]
    #     idx = np.asarray(selectedComp[compId], dtype = 'int')
    #     selectedFrames.extend(idx)

    # selectedFrames = np.asarray(np.sort(selectedFrames))

    stack = tiff.imread(os.path.join(path, tif))

    new_stack = stack[all_correct_index, :, :]
    # times = (field[selectedFrames, 1] - field[0, 1])/1000

    ##
    
    # def f(x, A):
    #     return((x in A))


    # def array_for(X, A):
    #     return(np.array([f(xi, A) for xi in X]))

    
    # comp_mask = array_for(all_correct_index, comp_correct_rate - 1)
    
    for z in range(len(new_stack)):
        if all_correct_index[z]+1 in comp_correct_rate:
            text = "INDENT"
            cv2.putText(new_stack[z, :, :], text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2, cv2.LINE_AA)

        # if len(activationFrames) != 0 and z in activationSlices:
        #     cv2.circle(new_stack[z, :, :], (128, 157), 175, (0,0,255), 3)

    io.imsave(os.path.join(pathSave, tif[:-4] + '_anot.tif'), new_stack)