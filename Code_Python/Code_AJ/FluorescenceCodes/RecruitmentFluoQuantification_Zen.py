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
ylabel = 20
xlabel = 20
axtitle = 35
figtitle = 30
font_ticks = 20

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
dirSave = os.path.join(DirData, 'Data_Fluorescence', 'Kymographs', date)


if not os.path.exists(dirProcessed):
    os.mkdir(dirProcessed)

if not os.path.exists(cp.DirDataFigToday):
    os.mkdir(cp.DirDataFigToday)


timeRes = 10 #in secs
firstActivation = 6 #in timepoints
firstActMin = np.round(firstActivation*timeRes / 60, 1)


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
cv2.drawContours(act_img, contour_in, -1, (0, 255, 0), 1)
cv2.drawContours(act_img, contour_out, -1, (255, 0, 0), 1)

# Convert BGR to RGB for display

# Display the image with contours
plt.imshow(blurred)
plt.show()

# plt.imshow(act_img)
# plt.axis('off')
# plt.show()

        
#%% Importing the cellpose model

allCells = os.listdir(dirProcessed)
allCells = [x for x in allCells if channel in x and 'Partial-Half' in x]

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
            imgs, diameter=116, channels=channels,
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
    
    plt.close('all')

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
    
    
    fig1, ax = plt.subplots(1, 2, figsize=(10, 10))
    
    x = np.arange(kymo_norm.shape[1])  # 18 columns
    y = np.arange(kymo_norm.shape[0])
    duration = (x * timeRes / 60)[-1]  # in minutes
    
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
    im2 = ax[0].imshow(allkymo_interpolated.T, aspect='auto', cmap='magma', origin='lower', vmin = 0, vmax = 80000)
    im = ax[1].imshow(kymo_interpolated.T, aspect='auto', cmap='magma', origin='lower', vmin = 0, vmax = 2.0)
    
    if 'HalfActivation' in currentCell:
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
        colors = ['r', 'b', 'b', 'r']
        # plt.imshow(cv2.drawContours(np.zeros(kymo_cont_interpolated.shape), contours, -1, (255, 0, 0), 1))
        angles, count = [], 0
        for cnt, colour in zip(contours, colors):
            M = cv2.moments(cnt)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            theta = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
            endx = cols * np.cos(theta) + center[0]
            endy = cols * np.sin(theta) + center[1]
            
            
            ax[1].plot([0, cols], [center[1], endy], color = colour, linewidth = 3, linestyle = '--')
            ax[0].plot([0, cols], [center[1], endy], color = colour, linewidth = 3, linestyle = '--')
            
            if count == 1 or count == 2:
                angles.append((center[1], int(endy)))
                
            count = count + 1
    
        for i in range(np.shape(kymo_norm)[1]-1):
            frames = np.linspace(0, np.shape(kymo_norm)[1]-1, np.shape(kymo_norm)[1])
            # medFront = np.average(kymo_norm[100:200, i])
            medFront = np.average(np.average(kymo_norm[0:angles[1][0], i]) + np.average(kymo_norm[angles[0][0]:359, i]))
            # fluoDict['fluoFront'].append(medFront)
            medBack = np.average(kymo_norm[angles[1][0]:angles[0][0], i])
            fluoDict['fluoBack'].append(medBack)
            fluoDict['fluoTotal'].append(np.average(kymo_norm[:, i]))
            fluoDict['cellID'].append(currentCell)
            fluoDict['frame'].append(i)
    
    
    ax[0].set_xlim(0, cols)
    ax[1].set_xlim(0, cols)
    
    ax[1].set_title('Normalised', fontsize = ylabel)
    fig1.colorbar(im, orientation='vertical', fraction = 0.055, pad = 0.04)

    ax[0].set_title('Not normalised', fontsize = ylabel)
    fig1.colorbar(im2, orientation='vertical', fraction = 0.055, pad = 0.04)
    
    ax[0].set_ylabel('Angle (degrees)', fontsize = ylabel)
    ax[0].set_xlabel('Time (mins)', fontsize = xlabel)
    ax[1].set_xlabel('Time (mins)', fontsize = xlabel)
    
    ax[0].tick_params(axis='both', which='major', labelsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=20)


    fig1.suptitle(currentCell, fontsize = 16)
    
    # axline = find_nearest(xxtime, firstActMin)
    # ax[0].axvline(x = axline, color = 'red')
    # ax[1].axvline(x = axline, color = 'red')
    # plt.tight_layout()
    # plt.savefig(os.path.join(dirSave, currentCell.split('.tif')[0]+'.png'))
    # plt.show()
    
    


# plt.close('all')

#%% Plotting normalised actin fluroscence intensity in time
# plt.style.use('dark_background')
fluoDf = pd.DataFrame(fluoDict)
fig, ax = plt.subplots(figsize=(15,10))

data = fluoDf[fluoDf['cellID'].str.contains('M2')]
# data = fluoDf[fluoDf['cellID'].str.contains('C9') == False]
# data = fluoDf[fluoDf['cellID'].str.contains('C13') == False]

data = data[(data['frame'] < 27)]
# data = data[(data['frame'] > 11)]

x = data['frame']*timeRes


flatui =  ["#FFD700", "#ee82ee", "#1AFFC6"]
flatui = ["#bb2fa6", "#000000"]

sns.set_palette(flatui)

sns.lineplot(data=data, x = x ,y="fluoFront") #, hue ='cellID')
sns.lineplot(data=data, x = x ,y="fluoBack")

control = mpatches.Patch(color=flatui[0], label='Polarised rear (activated)')
activated = mpatches.Patch(color=flatui[1], label='Polarised front')

x2 = np.linspace(120, 260,7)

# plt.legend(handles=[activated, control], fontsize = 20, loc = 'upper left')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('Time (secs)', fontsize=30)
plt.ylabel('Normalised Actin fluoresence intensity', fontsize=30)
plt.axvline(x = 120, color = 'red')

for i in x2:
    ax.axvline(x = i, color = "#bb2fa6", linewidth=4, ymax=0.10)

plt.tight_layout()
plt.ylim(0.4, 1.8)

plt.savefig('{}/{}_{}_ActinRecruitmentvTime.png'.format(cp.DirDataFigToday, currentCell, channel), dpi = 100)

plt.show()

#%% Plotting total actin intensity
plt.figure(figsize=(15,10))
flatui =  ["#1AFFC6", "#FFD700", "#ee82ee"]
sns.set_palette(flatui)

sns.lineplot(data=data, x = x ,y="fluoTotal")
plt.ylim(0.4, 1.8)
plt.axvline(x = 120, color = 'red')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Time (secs)', fontsize=30)
# plt.ylabel('Total Normalised Actin fluoresence intensity', fontsize=30)
plt.axvline(x = 120, color = 'red')
plt.tight_layout()

plt.savefig('{}/{}_TotalActinRecruitmentvTime.png'.format(cp.DirDataFigToday, channel), dpi = 100)
plt.show()

#%% Plotting actin intensity box plots before / after activation
# fluoDf = pd.DataFrame(fluoDict)
# data = fluoDf[fluoDf['cellID'].str.contains('M3')]
# dataPreAct =  data[(data['frame'] < 11)]
# dataPostAct =  data[(data['frame'] < 27) & (data['frame'] > 11)]
# fig1, axes = plt.subplots((1, 2), figsize = (15,10))

# sns.pointplot(x = x, y = y, data=data, hue = 'cellID', ax = axes[0], dodge = True)


#%%
allMedFront = []
allMedBack = []
for i in range(np.shape(kymo_norm)[1]-1):
    medFront = np.median(kymo_norm[150:200, i])
    allMedFront.append(medFront)
    medBack = np.median(kymo_norm[300:350, i])
    allMedBack.append(medBack)

plt.plot(allMedFront)
plt.plot(allMedBack)
plt.show()
#%% Creating stacks from individual files

expt = '20230405_3t3optoLARG_ActinRecruitmentDynamics_SPY650'
# subDir = 'Rpe1Tiam_Fastact640'
dirExt = 'F:/Cortex Experiments/Fluorescence Experiments/'
dirSave = 'D:/Anumita/MagneticPincherData/DataFluorescence/Raw/'
prefix = 'cell'
channel = 'w4CSU561'

excludedCells = AllMMTriplets2Stack(dirExt, dirSave, expt = expt, prefix = prefix, channel = channel)

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




plt.imshow(allkymo_interpolated)
plt.imshow(kymo_cont_interpolated, origin='lower', cmap='Reds', alpha=0.1)
plt.show()