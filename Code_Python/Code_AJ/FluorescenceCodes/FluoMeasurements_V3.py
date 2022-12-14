# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:45:06 2022

@author: anumi


To use this code, run the following in the Anaconda Powershell:
    
conda create -n cellpose pytorch=1.8.2 cudatoolkit=10.2 -c pytorch-lts
conda activate cellpose
conda install seaborn pandas scikit-image
conda install -c conda-forge spyder-kernels
conda install -c anaconda statsmodels
pip install opencv-python
pip install cellpose

Make sure to have your grahical backend as Inline and not Qt, as cellpose affects the Qt package version.

When you quit Spyder, MAKE SURE:
    1) You set the iPython console to it's default interpretor
    2) Type 'conda deactivate cellpose' in Anaconda PowerShell 

"""
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import unravel_index
import matplotlib.pyplot as plt
from scipy import interpolate, signal
from skimage.transform import warp_polar


import cellpose 
from cellpose.io import imread
from cellpose import plot, models, utils

# from skimage import measure

import CortexPaths as cp
os.chdir(cp.DirRepoPython)
import GraphicStyles as gs
# import UtilityFunctions as ufun

# os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = "D:/Anumita/MagneticPincherData/DataFluorescence/CellposeModels"

#Font size for plots
ylabel = 25
xlabel = 25
axtitle = 25
figtitle = 30
font_ticks = 25

#%% Setting directories

date = '22.12.02'
channel = 'w3CSU640'
subDir = '3t3optorhoa_Fastact640'
dirFluoRaw = cp.DirData + '/DataFluorescence/Raw/' + date + '/' + subDir
dirProcessed = os.path.join(cp.DirData + '/DataFluorescence/Processed', date)

if not os.path.exists(dirProcessed):
    os.mkdir(dirProcessed)

if not os.path.exists(cp.DirDataFigToday):
    os.mkdir(cp.DirDataFigToday)

timeRes = 10 #in secs
firstActivation = 2 #in mins
#%% Preprocessing and saving stacks as individual images for cellpose to do its work

allCells = os.listdir(dirFluoRaw)
scale_resize = 2

# allCells = allCells
for currentCell in allCells:
    print(gs.GREEN + currentCell + gs.NORMAL)
    folderCell = (os.path.join(dirFluoRaw, currentCell))
    fileCell = os.path.join(folderCell, currentCell + '_' + channel + '.tif')
    stack = cv2.imreadmulti(fileCell, [], cv2.IMREAD_ANYDEPTH)[1]

    filenames = [(f"{i:04d}.tif") for i in range(1, len(stack), 1)]
        
    for (j, k) in zip(stack, filenames):
        original = np.uint8(j)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalised_img = clahe.apply(original)
        
        equalised_img = cv2.resize(equalised_img, (int(j.shape[1]/scale_resize), int(j.shape[0]/scale_resize)))

        medianBlur = cv2.medianBlur(equalised_img, 5)
        # gaussianBlur = cv2.GaussianBlur(medianBlur, (5,5), 0)
        
        saveCell = os.path.join(dirProcessed, currentCell)
        saveChannel = os.path.join(saveCell, channel)
        
        if not os.path.exists(saveCell):
            os.mkdir(saveCell)
        
        if not os.path.exists(saveChannel):
            os.mkdir(saveChannel)
        
        cv2.imwrite(os.path.join(saveChannel, k), medianBlur)
        
        # cv2.imshow('Equalised', gaussianBlur)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
        
cv2.destroyAllWindows()
    
#%% Importing the cellpose model

# currentCell = '22-12-02_M2_P1_C9_disc20um'
allCells = os.listdir(dirProcessed)

for currentCell in allCells:
    
    filePath = os.path.join(dirProcessed, currentCell, channel)
    files = os.listdir(filePath)
    
    model = models.Cellpose(gpu=False, model_type='cyto')
    imgs = [imread(os.path.join(filePath, f)) for f in files]
    
    masks, flows, styles, diams = model.eval(imgs, diameter = 116, channels=[0, 0],
                                             flow_threshold = 0.2, do_3D=False, normalize = True)
    
    for each in masks:
        plt.imshow(each)
        plt.show()
    
    plt.close('all')
    
    
    allKymo = []
    allMaxVals = []    
    
    R_in = 40 #in px
    cortexThickness = 10
    
    for i in range(len(masks)):
        mask = masks[i]
        img = imgs[i]
        bg = np.mean(img[0:50, 0:50])
        
        img = img - bg
        outlines = img.copy()
        
        bw_mask = cellpose.utils.masks_to_edges(mask)*1
        center = cellpose.utils.distance_to_boundary(mask)
        cX, cY = unravel_index(center.argmax(), center.shape)
    
        warped_img = warp_polar(img, center = (cX, cY), radius = 250)
        maskVerifyBounds = np.copy(warped_img)
        warped_copy = np.zeros(len(warped_img)-1)
        warped_mask = warp_polar(bw_mask, center = (cX, cY), radius = 250)*10**12
        
        maxValues = np.argmax(warped_img, axis = 1)
        maxInter = signal.savgol_filter(maxValues, 301, 3)
        innerMean = np.mean(warped_img[:, 0:cortexThickness])
        
        # print(i)
        for j in range(len(warped_img)-1):
            maxval = np.argmax(warped_mask[j,:])
            warped_copy[j] = np.average(warped_img[j, maxval - cortexThickness:maxval])
            maskVerifyBounds[j, maxval - cortexThickness], maskVerifyBounds[j, maxval] =  0, 0
            
            # maxval = int(maxInter[j])
            # warped_copy[j] = np.average(warped_img[j, maxval - cortexThickness:maxval + cortexThickness])
            # warped_mask[j, maxval - cortexThickness],  warped_mask[j, maxval + cortexThickness] = 0, 0
        
        final = warped_img
        plt.imshow(maskVerifyBounds)
        plt.show()
        
        allKymo.append(warped_copy)
    
    allKymo = np.asarray(allKymo).T
    
    plt.style.use('dark_background')
    kymo_norm = allKymo/allKymo[0]
    plt.imshow(kymo_norm)
    
    fig1, ax1 = plt.subplots(1, 1, figsize = (10,10))
    length = np.shape(kymo_norm)[1]
    xx = np.arange(length)
    cleanSize = 360
    yy = np.arange(0,cleanSize-1)
    f = interpolate.interp2d(xx, yy, kymo_norm, kind='cubic')
    xxnew =  np.linspace(0, length, 300)
    yynew = yy
    profileROI_hd = f(xxnew, yynew)
    
    im = ax1.imshow(profileROI_hd)
    
    xtickslocs = ax1.get_xticks()[1:]
    new_xticks = np.linspace(0, length, len(xtickslocs))
    new_xlabels = (np.round(new_xticks*timeRes/60, 2)) #Converting frames to second
    ax1.set_xticks(xtickslocs, new_xlabels)
    ax1.set_xlabel('Time (mins)', fontsize = xlabel)
    
    ax1.set_ylabel('Angle (degrees)', fontsize = ylabel)
    # ytickslabels = ax1.set_yticks()[1:-1]
    ax1.tick_params(axis = 'both', labelsize = font_ticks)
    ax1.set_title(currentCell, fontsize = axtitle)
    ax1.axvline(x = 2*length, color = 'red')
    cbar = fig1.colorbar(im)
    cbar.ax.tick_params(labelsize=20)
    fig1.tight_layout()
    
    plt.savefig(cp.DirDataFigToday + '/'+currentCell+'_Kymo.png')

# plt.close('all')

#%% Creating stacks from individual files

# expt = '20221202_rpe1-3t3_100x_ActivationTests'
# subDir = '3t3optorhoa_Fastact640'
# dirExt = 'F:/Cortex Experiments/Fluorescence Experiments/'
# dirSave = 'D:/Anumita/MagneticPincherData/DataFluorescence/Raw/'
# prefix = 'cell'
# channel = 'w4CSU561'

# excludedCells = ufun.AllMMTriplets2Stack(dirExt, dirSave, expt = expt, prefix = prefix, channel = channel, subDir = subDir)

#%% Extras


# x = np.linspace(0, len(warped_img)-1, len(warped_img))
# f = interpolate.interp1d(x, maxValues, kind='cubic')
# xnew = x
# maxInter = f(xnew)
# t, c, k = interpolate.splrep(x, maxValues, s=0, k=5)
# print('''\
# t: {}
# c: {}
# k: {}
# '''.format(t, c, k))
# N = 600
# xmin, xmax = x.min(), x.max()
# xx = np.linspace(xmin, xmax, N)
# f = interpolate.BSpline(t, c, k, extrapolate=False)
# maxInter = f(xnew)
# plt.plot(x, maxValues)
# plt.plot(x, maxInter)
# plt.show()

# contour = np.asarray(np.where(bw_mask == 1)).T

# r = np.asarray([np.hypot(i[0] - cX, i[1] - cY) for i in contour])
# r_in = r - R_in
# avg_r_in = np.mean(r_in)
# theta =  np.asarray([np.arctan2(i[0] - cX, i[1] - cY) for i in contour])

# xy = np.asarray([[i[0], i[1]] for i in contour])
# xy_in = np.asarray([[int(i*np.cos(j) + cX), int(i*np.sin(j) + cY)] for i, j in zip(r_in, theta)])
# xy_in_fit = np.asarray([[int(avg_r_in*np.cos(j) + cX), int(avg_r_in*np.sin(j) + cY)] for j in theta])






