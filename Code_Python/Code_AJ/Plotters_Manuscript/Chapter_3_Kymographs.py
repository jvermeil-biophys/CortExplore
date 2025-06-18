# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:45:06 2022

@author: anumi


To use this code, run the following in the Anaconda Powershell:
    
conda create -n cellpose pytorch=1.8.2 cudatoolkit=10.2 -c pytorch-lts
conda activate cellpose
conda install -c conda-forge spyder-kernels
conda install seaborn pandas scikit-image
conda install -c anaconda statsmodels
pip install opencv-python
pip install cellpose

Make sure to have your grahical backend as Inline and not Qt, as cellpose affects the Qt package version.

When you quit Spyder, MAKE SURE:
    1) You set the iPython console to it's default interpretor
    2) Type 'conda deactivate cellpose' in Anaconda PowerShell 

"""
import os
import re

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import unravel_index
import matplotlib.pyplot as plt
from scipy import interpolate, signal
from scipy.interpolate import RegularGridInterpolator as RGI
import matplotlib.patches as mpatches
from skimage.transform import warp_polar


import CortexPaths as cp
os.chdir(cp.DirRepoPython)
import GraphicStyles as gs
# import UtilityFunctions as ufun

import cellpose 
from skimage import io as skio
from cellpose.io import imread
from cellpose import plot, models, utils, io

# from skimage import measure

# os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = "D:/Anumita/MagneticPincherData/DataFluorescence/CellposeModels"

#Font size for plots
ylabel = 12
xlabel = 12
axtitle = 12
figtitle =12
font_ticks = 12

SCALE_px_cm = 2.60

#%% Functions

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def findInfosInFileName(f, infoType):
    """
    Return a given type of info from a file name.
    Inputs : f (str), the file name.
             infoType (str), the type of info wanted.
             
             infoType can be equal to : 
                 
             * 'M', 'P', 'C' -> will return the number of manip (M), well (P), or cell (C) in a cellID.
             ex : if f = '21-01-18_M2_P1_C8.tif' and infoType = 'C', the function will return 8.
             
             * 'manipID'     -> will return the full manip ID.
             ex : if f = '21-01-18_M2_P1_C8.tif' and infoType = 'manipID', the function will return '21-01-18_M2'.
             
             * 'cellID'     -> will return the full cell ID.
             ex : if f = '21-01-18_M2_P1_C8.tif' and infoType = 'cellID', the function will return '21-01-18_M2_P1_C8'.
             
             * 'substrate'  -> will return the string describing the disc used for cell adhesion.
             ex : if f = '21-01-18_M2_P1_C8_disc15um.tif' and infoType = 'substrate', the function will return 'disc15um'.
             
             * 'ordinalValue'  -> will return a value that can be used to order the cells. It is equal to M*1e6 + P*1e3 + C
             ex : if f = '21-01-18_M2_P1_C8.tif' and infoType = 'ordinalValue', the function will return "2'001'008".
    """
    infoString = ''
    try:
        if infoType in ['M', 'P', 'C']:
            acceptedChar = [str(i) for i in range(10)] + ['-']
            string = '_' + infoType
            iStart = re.search(string, f).end()
            i = iStart
            infoString = '' + f[i]
            while f[i+1] in acceptedChar and i < len(f)-1:
                i += 1
                infoString += f[i]
                
        elif infoType == 'date':
            datePos = re.search(r"[\d]{1,2}-[\d]{1,2}-[\d]{2}", f)
            date = f[datePos.start():datePos.end()]
            infoString = date
        
        elif infoType == 'manipID':
            datePos = re.search(r"[\d]{1,2}-[\d]{1,2}-[\d]{2}", f)
            date = f[datePos.start():datePos.end()]
            manip = 'M' + findInfosInFileName(f, 'M')
            infoString = date + '_' + manip
            
        elif infoType == 'cellName':
            infoString = 'M' + findInfosInFileName(f, 'M') + \
                         '_' + 'P' + findInfosInFileName(f, 'P') + \
                         '_' + 'C' + findInfosInFileName(f, 'C')
            
        elif infoType == 'cellID':
            datePos = re.search(r"[\d]{1,2}-[\d]{1,2}-[\d]{2}", f)
            date = f[datePos.start():datePos.end()]
            infoString = date + '_' + 'M' + findInfosInFileName(f, 'M') + \
                                '_' + 'P' + findInfosInFileName(f, 'P') + \
                                '_' + 'C' + findInfosInFileName(f, 'C')
                                
        elif infoType == 'substrate':
            try:
                pos = re.search(r"disc[\d]*um", f)
                infoString = f[pos.start():pos.end()]
            except:
                infoString = ''
                
        elif infoType == 'ordinalValue':
            M, P, C = findInfosInFileName(f, 'M'), findInfosInFileName(f, 'P'), findInfosInFileName(f, 'C')
            L = [M, P, C]
            for i in range(len(L)):
                s = L[i]
                if '-' in s:
                    s = s.replace('-', '.')
                    L[i] = s
            [M, P, C] = L
            ordVal = int(float(M)*1e9 + float(P)*1e6 + float(C)*1e3)
            infoString = str(ordVal)
            
    except:
        pass
                             
    return(infoString)


def AllMMTriplets2Stack(DirExt, DirSave, expt, prefix, channel, subDir = None):
    """
    Used for metamorph created files.
    Metamoprh does not save images in stacks but individual triplets. These individual triplets take time
    to open in FIJI.
    This function takes images of a sepcific channel and creates .tif stacks from them.
       
    """
    if subDir == None:
        DirCells = os.path.join(DirExt, expt)
    else:
        DirCells = os.path.join(DirExt, expt, subDir)
    
    allCells = os.listdir(DirCells)
    excludedCells = []
    for currentCell in allCells:
        dirImages = os.path.join(DirCells, currentCell)
        date = findInfosInFileName(currentCell, 'date')
        date = date.replace('-', '.')
        filename = currentCell+'_'+channel
        exptPath = DirSave+'/'+date
        
        if not os.path.exists(exptPath):
            os.mkdir(exptPath)
        
        if subDir == None:
            dirSave = os.path.join(exptPath, currentCell)
            if not os.path.exists(dirSave):
                os.mkdir(dirSave)
        else:
            dirSave = os.path.join(exptPath, subDir, currentCell)
            dirSaveSubdir = os.path.join(exptPath, subDir)
            if not os.path.exists(dirSaveSubdir):
                os.mkdir(dirSaveSubdir)
            
            if not os.path.exists(dirSave):
                os.mkdir(os.path.join(dirSaveSubdir, currentCell))
                
        allFiles = os.listdir(dirImages)
        date = findInfosInFileName(currentCell, 'date')
        # print(gs.YELLOW + currentCell + gs.NORMAL)
        
        allFiles = [dirImages+'/'+string for string in allFiles if 'thumb' not in string and '.TIF' in string and channel in string]
        
        if len(allFiles) == 0:
            print(gs.ORANGE + 'Error in loading files' + gs.NORMAL)
            break
        
        #+4 at the end corrosponds to the '_t' part to sort the array well
        limiter = len(dirImages)+len(prefix)+len(channel)+4
        
        try:
            allFiles.sort(key=lambda x: int(x[limiter:-4]))
        except:
            print(gs.ORANGE + 'Error in sorting files for ' + currentCell + gs.NORMAL)

        try:
            ic = skio.ImageCollection(allFiles, conserve_memory = True)
            stack = skio.concatenate_images(ic)
            skio.imsave(dirSave+'/'+filename+'.tif', stack, check_contrast=False)
            print(gs.GREEN + "Successfully saved "+currentCell + gs.NORMAL)
        except:
            excludedCells.append(currentCell)
            print(gs.ORANGE + "Unknown error in saving "+currentCell + gs.NORMAL)
            
    return excludedCells

# Helper function to create directories if they don't exist
def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
#%% Setting directories

DirData = 'D:\\Anumita\\MagneticPincherData\\'
date = '25.01.09'
channel = 'Actin'
# subDir = '3t3opthorhoa_Fastact640'
dirFluoRaw = 'D:\\Anumita\\MagneticPincherData\\Data_Fluorescence\\Raw\\25.01.09\\' #cp.DirData + '/DataFluorescence/Raw/' + date + '/' + subDir
dirProcessed = os.path.join(DirData, 'Data_Fluorescence', 'Processed', date)
dirSegment = DirData + 'Data_Fluorescence\\Segmentation' 
dirSave ='D:/Anumita/NextCloud/Anumita - Manuscript/Figures/Chapter_3/ActinFluorescenceKymographs'


if not os.path.exists(dirProcessed):
    os.mkdir(dirProcessed)

if not os.path.exists(cp.DirDataFigToday):
    os.mkdir(cp.DirDataFigToday)


timeRes = 10.149580600000002 #in secs
firstActivation = 6 #in timepoints


#%% Preprocessing and saving stacks as individual images for cellpose to do its work

allCells = os.listdir(dirFluoRaw)
scale_resize = 2

allCells = [i for i in allCells if channel in i]

for currentCell in allCells:
    print(gs.GREEN + currentCell + gs.NORMAL)
    fileCell = os.path.join(dirFluoRaw, currentCell)

    stack = cv2.imreadmulti(fileCell, [], cv2.IMREAD_ANYDEPTH)[1]

    filenames = [(f"{i:04d}.tif") for i in range(len(stack))]
        
    for (j, k) in zip(stack, filenames):
        j = np.fliplr(j)

        medianBlur = cv2.medianBlur(j, 3)
        
        saveCell = os.path.join(dirProcessed, currentCell)

        if not os.path.exists(saveCell):
            os.mkdir(saveCell)
        
        cv2.imwrite(os.path.join(saveCell, k), medianBlur)

#%% Processing activation region

act_img_path = os.path.join(dirFluoRaw, 'Info', 'Filter5_150ms_HEX37_ActivationRegion.tif')
act_img = cv2.imread(act_img_path)
act_img = np.fliplr(act_img)
act_img = cv2.cvtColor(act_img, cv2.COLOR_BGR2RGB)

# Read the image
image = cv2.imread(act_img_path)
image = np.fliplr(image)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blurred = cv2.blur(gray,(55,55))

central_profile = np.mean(gray[170:210, :], axis = 0)
# plt.plot(central_profile)

threshold_inner_contour = int((50 / 100) * (np.max(central_profile) - np.min(central_profile)))
threshold_outer_contour = int((10 / 100) * (np.max(central_profile) - np.min(central_profile)))

# Use Canny edge detection
ret_in,th_in = cv2.threshold(blurred,threshold_inner_contour,255,cv2.THRESH_BINARY)
ret_out,th_out = cv2.threshold(blurred,threshold_outer_contour,255,cv2.THRESH_BINARY)

# Find contours from the edges
contour_in, _ = cv2.findContours(th_in, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contour_out, _ = cv2.findContours(th_out, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# Draw the contour on the original image
cv2.drawContours(act_img, contour_in, -1, (0, 0,0), 1)
cv2.drawContours(act_img, contour_out, -1, (0, 255, 0), 1)

cv2.drawContours(blurred, contour_in, -1, (0, 0,0), 1)
cv2.drawContours(blurred, contour_out, -1, (0, 255, 0), 1)


# Convert BGR to RGB for display

# Display the image with contours
plt.imshow(blurred)
plt.show()

# plt.imshow(act_img)
# plt.axis('off')
# plt.show()

        
#%% Importing the cellpose model

allCells = os.listdir(dirProcessed)
allCells = [x for x in allCells if channel in x and 'Global' in x and 'C4' not in x]


fluoDict = {'cellID': [], 
            'fluoFront': [],
            'fluoBack': [], 
            'fluoTotal': [],
            'frame':[]}

allKymoNorm = []


for j in range(len(allCells)):
    currentCell = allCells[j]

    segFolderCell = os.path.join(dirSegment, date)
    segFolderCh = os.path.join(dirSegment, date, currentCell)

    if not os.path.exists(segFolderCell) or not os.path.exists(segFolderCh):
        print(gs.ORANGE + 'Segmentation not done for cell ' + currentCell)
        print('Running model and creating segmented masks..' + gs.NORMAL)

        if not os.path.exists(segFolderCell):
            os.makedirs(segFolderCell)

        if not os.path.exists(segFolderCh):
            os.makedirs(segFolderCh)

        filePath = os.path.join(dirProcessed, currentCell)
        files = os.listdir(filePath)

        model = models.Cellpose(gpu=False, model_type='cyto')
        imgs = [imread(os.path.join(filePath, f)) for f in files]

        channels = [0, 0]
        masks, flows, styles, diams = model.eval(
            imgs, diameter=112, channels=channels,
            flow_threshold=0.2, do_3D=False, normalize=True
        )

        # Updated segFilename
        segFilename = [os.path.join(segFolderCh, f"{k}") for k in range(len(masks))]
        
        saveMasks = [io.masks_flows_to_seg(
            imgs[k], masks[k], flows[k], diams, segFilename[k], channels) for k in range(len(masks))]
    else:
        print(gs.GREEN + 'Segmentation already done for cell ' + currentCell)
        print('Loading masks..' + gs.NORMAL)

        nMasks = len(os.listdir(segFolderCh))
        datMasks = [np.load(segFolderCh + '/' + str(x) + '_seg.npy', allow_pickle=True).item()['masks'] for x in range(nMasks)]
        datImgs = [np.load(segFolderCh + '/' + str(x) + '_seg.npy', allow_pickle=True).item()['img'] for x in range(nMasks)]
        
        masks = np.asarray(datMasks)
        imgs = np.asarray(datImgs)

    # for each in masks:
    #     plt.imshow(each)
    #     plt.show()
    
    # plt.close('all')

    allKymo = []
    allMaxValsFront = [] 
    allMaxValsBack = [] 
    kymoNorm = []
    kymoContour = []
    
    R_in = 40 #in px
    cortexThickness = 11
    
    for i in range(len(masks)):
        mask = masks[i]
        img = imgs[i]
        bg = np.mean(img[0:50, 0:50])
        
        img = img - bg
        h,w = img.shape
        size = (w,h)
        cnt_mask =  np.zeros((h, w), np.uint8)
        # cnt_mask = cv2.cvtColor(cnt_mask,cv2.COLOR_BGR2GRAY)
        cnt_mask = cv2.drawContours(cnt_mask, contour_in, -1, (255, 0, 0), 2)
        cnt_mask = cv2.drawContours(cnt_mask, contour_out, -1, (255, 0, 0), 2)
        
        bw_mask = cellpose.utils.masks_to_edges(mask)*1
        center = cellpose.utils.distance_to_boundary(mask)
        cX, cY = unravel_index(center.argmax(), center.shape)

        warped_img = warp_polar(img, center = (cX, cY), radius = 250)
        
        if i == 0:
            warped_contour = warp_polar(cnt_mask, center = (cX, cY), radius = 250)
        
        maskVerifyBounds = np.copy(warped_img)
   
        warped_copy = np.zeros(len(warped_img)-1)
        warped_copy_cont = np.zeros(len(warped_contour)-1)
        warped_mask = warp_polar(bw_mask, center = (cX, cY), radius = 250)
        
        maxValues = np.argmax(warped_img, axis = 1)
        maxInter = signal.savgol_filter(maxValues, 301, 3)
        # innerMean = np.mean(warped_img[:, 0:cortexThickness])
        
        # warped_imgClean = np.asarray([warped_img[:, k] - innerMean for k in range(np.shape(warped_img)[1])]).T
        # warped_imgClean = warped_img - innerMean
        
        for j in range(len(warped_img)-1):
            # maxval = int(maxInter[j])
            maxval = np.argmax(warped_mask[j,:])
            # innerMean = np.mean(warped_img[j, 0:cortexThickness])
            warped_copy[j] = np.max(warped_img[j, maxval - cortexThickness:maxval]) #- innerMean
            warped_copy_cont[j] = np.average(warped_contour[j, maxval - cortexThickness:maxval])
            maskVerifyBounds[j, maxval - cortexThickness], maskVerifyBounds[j, maxval] =  0, 0

        # plt.imshow(maskVerifyBounds)
        # plt.show()
        
        allKymo.append(warped_copy)
        kymoContour.append(warped_copy_cont)
    
    # plt.style.use('dark_background')
    plt.style.use('default')
    allKymo = np.asarray(allKymo)
    
    preActivationAvg = np.mean(allKymo[0:firstActivation], axis = 0)
    
    kymo_norm = np.array([allKymo[k] / preActivationAvg for k in range(np.shape(allKymo)[0])]).T
    kymo_norm = np.asarray(kymo_norm)
    allKymo = allKymo.T
    kymoContour = np.asarray(kymoContour).T
    allKymoNorm.append(kymo_norm)
    
    # fig1, ax = plt.subplots(1, 2, figsize=(16/SCALE_px_cm, 12/SCALE_px_cm))
    fig1, ax = plt.subplots(1, 1, figsize=(12/SCALE_px_cm, 12/SCALE_px_cm))

    
    x = np.arange(kymo_norm.shape[1])  # 18 columns
    y = np.arange(kymo_norm.shape[0])
    duration = (x * timeRes / 60)[-1]  # in minutes
    activations = np.asarray([1.0, 2.0, 3.0])
    # Create meshgrid for x and y coordinates
    interpolator = RGI((y, x), kymo_norm, method='linear', bounds_error=False)
    xtime = np.linspace(0, duration, 300)
    # New grid
    xnew = np.linspace(0, kymo_norm.shape[1] - 1, 300)  # Interpolated to 300 columns
    # ynew = np.linspace(0, kymo_norm.shape[0] - 1, kymo_norm.shape[0]) 
    xxnew, yynew = np.meshgrid(xnew, y, indexing='ij')
    # Interpolated data
    points = np.array([yynew.ravel(), xxnew.ravel()]).T  # Create a grid of (y, x) points
    kymo_interpolated = interpolator(points).reshape(yynew.shape)

    x2 = np.arange(allKymo.shape[1])  # 18 columns
    y2 = np.arange(allKymo.shape[0])
    interpolator = RGI((y2, x2), allKymo, method='linear', bounds_error=False)
    xtime2 = np.linspace(0, duration, 300)
    # New grid
    xnew2 = np.linspace(0, allKymo.shape[1] - 1, 300)  # Interpolated to 300 columns
    xxnew2, yynew2 = np.meshgrid(xnew2, y2, indexing='ij')
    # Interpolated data
    points2 = np.array([yynew2.ravel(), xxnew2.ravel()]).T  # Create a grid of (y, x) points
    allkymo_interpolated = interpolator(points2).reshape(yynew2.shape)
    # Plot the interpolated data
    # im2 = ax[0].imshow(allkymo_interpolated.T, aspect='auto', cmap='magma', origin='lower', vmin = 0, vmax = 80000)
    # im = ax[1].imshow(kymo_interpolated.T, aspect='auto', cmap='magma', origin='lower', vmin = 0, vmax = 2.0)
    im = ax.imshow(kymo_interpolated.T, aspect='auto', cmap='magma', origin='lower', vmin = 0, vmax = 2.0)
    
        
    if 'HalfActivation' in currentCell or 'Global' in currentCell:
        x1 = np.arange(kymoContour.shape[1])  # 18 columns
        y1 = np.arange(kymoContour.shape[0])
        interpolator_cnt = RGI((y1, x1), kymoContour, method='linear', bounds_error=False)
        
        # New grid
        xnew1 = np.linspace(0, kymoContour.shape[1] - 1, 300)  # Interpolated to 300 columns
        
        
        xxnew1, yynew1 = np.meshgrid(xnew1, y1, indexing='ij')
        # Interpolated data
        points1 = np.array([yynew1.ravel(), xxnew1.ravel()]).T  # Create a grid of (y, x) points
        kymo_cont_interpolated = interpolator_cnt(points1).reshape(yynew1.shape)
        kymo_cont_interpolated = cv2.normalize(kymo_cont_interpolated.T, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        _, kymo_cont_thresh = cv2.threshold(kymo_cont_interpolated,20,255,cv2.THRESH_BINARY)
    
        rows,cols = kymo_cont_interpolated.shape[:2]
        contours, _ = cv2.findContours(kymo_cont_thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        colors = ['#35b779', 'k', 'k', '#35b779']
        # colors = []
        # plt.imshow(cv2.drawContours(np.zeros(kymo_cont_interpolated.shape), contours, -1, (255, 0, 0), 1))
        angles, count = [], 0
        for cnt, colour in zip(contours, colors):
            M = cv2.moments(cnt)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            theta = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
            endx = cols * np.cos(theta) + center[0]
            endy = cols * np.sin(theta) + center[1]
            
            
            # ax[1].plot([0, cols], [center[1], endy], color = colour, linewidth = 1, linestyle = '--')
            ax.plot([0, cols], [center[1], endy], color = colour, linewidth = 1, linestyle = '--')
            
            if count == 1 or count == 2:
                angles.append((center[1], int(endy)))
                
            count = count + 1
    
        # for i in range(np.shape(kymo_norm)[1]-1):
        #     frames = np.linspace(0, np.shape(kymo_norm)[1]-1, np.shape(kymo_norm)[1])
        #     # medFront = np.average(kymo_norm[100:200, i])
        #     medFront = np.average(np.average(kymo_norm[0:angles[1][0], i]) + np.average(kymo_norm[angles[0][0]:359, i]))
        #     # fluoDict['fluoFront'].append(medFront)
        #     medBack = np.average(kymo_norm[angles[1][0]:angles[0][0], i])
        #     fluoDict['fluoBack'].append(medBack)
        #     fluoDict['fluoFront'].append(medFront)
        #     fluoDict['fluoTotal'].append(np.average(kymo_norm[:, i]))
        #     fluoDict['cellID'].append(currentCell)
        #     fluoDict['frame'].append(i)
        
        for i in range(np.shape(kymo_norm)[1]-1):
            frames = np.linspace(0, np.shape(kymo_norm)[1]-1, np.shape(kymo_norm)[1])
            total = np.average(kymo_norm[:, i])
            medBack = np.average(kymo_norm[angles[1][0]:angles[0][0], i])
            front_up, front_down = kymo_norm[0:angles[1][0], i], kymo_norm[angles[0][0]:359, i]
            whole_front = np.concatenate((front_up.flatten(), front_down.flatten()))
            medFront =   np.average(whole_front)
            fluoDict['fluoFront'].append(medFront)
            fluoDict['fluoBack'].append(medBack)
            fluoDict['fluoTotal'].append(total)
            fluoDict['cellID'].append(currentCell)
            fluoDict['frame'].append(i)
    
    
    xtick_positions = np.asarray([0, 1, 2, 3, 4])
    xtick_activations = np.asarray([np.abs(xtime - i).argmin() for i in activations])
    xticks = np.asarray([np.abs(xtime - i).argmin() for i in xtick_positions])
    xtick_labels = np.asarray([-1, 0, 1, 2, 3])

    ax.set_xticks(xticks, xtick_labels)
    # ax[1].set_xticks(xticks, xtick_labels)

    for i in xtick_activations:
        ax.axvline(x = i, color = '#0096FF', ymin = 0, ymax = 0.1)
        # ax[1].axvline(x = i, color = '#0096FF', ymin = 0, ymax = 0.1)
    
    # ax.set_xlim(0, cols)
    # ax[1].set_xlim(0, cols)
    
    ax.set_title('Normalised', fontsize = ylabel)
    fig1.colorbar(im, orientation='vertical', fraction = 0.055, pad = 0.04).ax.tick_params(labelsize=10)

    # ax[0].set_title('Not normalised', fontsize = ylabel)
    # fig1.colorbar(im2, orientation='vertical', fraction = 0.055, pad = 0.04).ax.tick_params(labelsize=10)
    
    ax.set_ylabel('Angle (degrees)', fontsize = ylabel)
    ax.set_xlabel('Time (mins)', fontsize = xlabel)
    # ax[1].set_xlabel('Time (mins)', fontsize = xlabel)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    # ax[1].tick_params(axis='both', which='major', labelsize=10)

    fig1.suptitle(currentCell, fontsize = 12)
    
    # axline = find_nearest(xxtime, firstActMin)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(dirSave, currentCell.split('.tif')[0]+'.pdf'), dpi = 200)



# plt.close('all')

#%% Plotting normalised actin fluroscence intensity in time
# plt.style.use('dark_background')
fluoDf = pd.DataFrame(fluoDict)

fig, ax = plt.subplots(figsize=(15/SCALE_px_cm,10/SCALE_px_cm))

# data = fluoDf[fluoDf['cellID'].str.contains('Partial-Half')]
data = fluoDf[fluoDf['cellID'].str.contains('Global')]


x = (data['frame']*timeRes)/60


# flatui =  ["#FFD700", "#ee82ee", "#1AFFC6"]
# flatui = ["#bb2fa6", "#000000"]
palette = ['#fdae61', '#000004']

sns.lineplot(data=data, x = x ,y="fluoFront", color=palette[1])
sns.lineplot(data=data, x = x ,y="fluoBack", color=palette[0])

sns.lineplot(data=data, x = x ,y="fluoFront", color=palette[1], style = 'cellID', alpha = 0.5)
sns.lineplot(data=data, x = x ,y="fluoBack", color=palette[0], style = 'cellID', alpha = 0.5)

rear = mpatches.Patch(color=palette[0], label='Polarised rear (Activated)')
front = mpatches.Patch(color=palette[1], label='Polarised front')



plt.legend(handles=[rear, front], fontsize = 11, loc = 'upper left')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.xlabel('Time (mins)')
plt.ylabel('', fontsize=11)
plt.title('Normalized Actin Fluorescence Intensity', fontweight = 'bold', fontsize=11)

x2 = np.asarray([1, 2, 3])
for i in x2:
    ax.axvline(x = i, color = "blue", linewidth=4, ymax=0.10)

plt.tight_layout()
plt.ylim(0.2, 1.5)
plt.show()

plt.savefig('{}/_{}_ActinRecruitmentvTime_Global.pdf'.format(cp.DirDataFigToday,channel), dpi = 200)


#%% Plotting total actin intensity
fig, ax = plt.subplots(figsize=(15/SCALE_px_cm,10/SCALE_px_cm))

sns.lineplot(data=data, x = x ,y="fluoTotal", color = "#3f0f3e")
sns.lineplot(data=data, x = x ,y="fluoTotal", color = "#3f0f3e", style = 'cellID')

plt.ylim(0.2, 1.5)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.xlabel('Time (mins)', fontsize=11)

x2 = np.asarray([1, 2, 3])
for i in x2:
    ax.axvline(x = i, color = "blue", linewidth=4, ymax=0.10)

plt.title('Total Actin Fluorescence Intensity', fontweight = 'bold', fontsize=11)
plt.tight_layout()
plt.legend().remove()
plt.show()
plt.savefig('{}/{}_TotalActinRecruitmentvTime_Global.pdf'.format(cp.DirDataFigToday, channel), dpi = 100)

