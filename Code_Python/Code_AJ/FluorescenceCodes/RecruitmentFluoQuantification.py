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
import re

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import unravel_index
import matplotlib.pyplot as plt
from scipy import interpolate, signal
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
ylabel = 15
xlabel = 15
axtitle = 25
figtitle = 30
font_ticks = 25

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

#%% Setting directories

date = '22.12.02'
channel = 'w3CSU640'
subDir = '3t3opthorhoa_Fastact640'
dirFluoRaw = cp.DirData + '/DataFluorescence/Raw/' + date + '/' + subDir
dirProcessed = os.path.join(cp.DirData + '/DataFluorescence/Processed', date)
dirSegment = cp.DirData + '/DataFluorescence/Segmentation' 

if not os.path.exists(dirProcessed):
    os.mkdir(dirProcessed)

if not os.path.exists(cp.DirDataFigToday):
    os.mkdir(cp.DirDataFigToday)


timeRes = 10 #in secs
firstActivation = 12 #in timepoints
firstActMin = np.round(firstActivation*timeRes / 60, 1)

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

# allCells = ['22-12-02_M3_P1_C19_disc20um']
allCells = os.listdir(dirProcessed)
# allCells = [x for x in allCells if 'P1' in x and 'M1' not in x ]

fluoDict = {'cellID': [], 
            'fluoFront': [],
            'fluoBack': [], 
            'frame':[]}

for j in range(len(allCells)):
    currentCell = allCells[j]
    
    segFolderCell = os.path.join(dirSegment, currentCell)
    segFolderCh = os.path.join(dirSegment, currentCell, channel)
    
    if not os.path.exists(segFolderCell) or not os.path.exists(segFolderCh):
        print(gs.ORANGE + 'Segmentation not done for cell ' + currentCell)
        print('Running model and creating segmented masks..' + gs.NORMAL)
        
        try:
            os.mkdir(segFolderCell)
        except:
            pass
        
        try:
            os.mkdir(segFolderCh)
        except:
            pass

        
        filePath = os.path.join(dirProcessed, currentCell, channel)
        files = os.listdir(filePath)
        
        model = models.Cellpose(gpu=False, model_type='cyto')
        imgs = [imread(os.path.join(filePath, f)) for f in files]
        
        channels = [0,0]
        masks, flows, styles, diams = model.eval(imgs, diameter = 116, channels=channels,
                                                 flow_threshold = 0.2, do_3D=False, normalize = True)
        
        
    
        segFilename = [segFolderCh + '/' + str(k) for k in range(len(masks))] 
        
        saveMasks = [io.masks_flows_to_seg(imgs[k], masks[k], flows[k], diams, segFilename[k], channels) \
                     for k in range(len(masks))]
    else: 

        print(gs.GREEN + 'Segmentation already done for cell ' + currentCell)
        print('Loading masks..' + gs.NORMAL)
        nMasks = len(os.listdir(segFolderCh))
        datMasks = [np.load(segFolderCh+'/'+str(x)+'_seg.npy', allow_pickle=True).item()['masks'] for x in range(nMasks)]
        datImgs = [np.load(segFolderCh+'/'+str(x)+'_seg.npy', allow_pickle=True).item()['img'] for x in range(nMasks)]

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
        # innerMean = np.mean(warped_img[:, 0:cortexThickness])
        
        # warped_imgClean = np.asarray([warped_img[:, k] - innerMean for k in range(np.shape(warped_img)[1])]).T
        # warped_imgClean = warped_img - innerMean
        
        # print(i)
        for j in range(len(warped_img)-1):
            # maxval = int(maxInter[j])
            maxval = np.argmax(warped_mask[j,:])
            # innerMean = np.mean(warped_img[j, 0:cortexThickness])
            warped_copy[j] = np.max(warped_img[j, maxval - cortexThickness:maxval]) #- innerMean
            maskVerifyBounds[j, maxval - cortexThickness], maskVerifyBounds[j, maxval] =  0, 0
            
            # maxval = int(maxInter[j])
            # warped_copy[j] = np.average(warped_img[j, maxval - cortexThickness:maxval + cortexThickness])
            # warped_mask[j, maxval - cortexThickness],  warped_mask[j, maxval + cortexThickness] = 0, 0
        
        final = warped_copy
        plt.imshow(maskVerifyBounds)
        plt.show()
        
        allKymo.append(warped_copy)
    
    
    plt.style.use('default')
    cmap = 'plasma'
    # cmap = 'viridis'
    allKymo = np.asarray(allKymo)
    
    preActivationAvg = np.mean(allKymo[0:firstActivation], axis = 0)
    
    kymo_norm = [kymoNorm.append(allKymo[k]/preActivationAvg) for k in range(np.shape(allKymo)[0])]
    
    kymo_norm = np.asarray(kymoNorm).T
    allKymo = np.asarray(allKymo).T
    
    for i in range(np.shape(kymo_norm)[1]-1):
        frames = np.linspace(0, np.shape(kymo_norm)[1]-1, np.shape(kymo_norm)[1])
        medFront = np.average(kymo_norm[150:200, i])
        fluoDict['fluoFront'].append(medFront)
        medBack = np.average(kymo_norm[300:350, i])
        fluoDict['fluoBack'].append(medBack)
        fluoDict['cellID'].append(currentCell)
        fluoDict['frame'].append(i)
        
    
    fig1, ax1 = plt.subplots(1, 2)
    length = np.shape(kymo_norm)[1]
    xx = np.arange(length)
    duration = (xx*timeRes/60)[-1] #in mins
    
    cleanSize = 360
    yy = np.arange(0,cleanSize-1)
    f = interpolate.interp2d(xx, yy, kymo_norm, kind='cubic')
    xxnew =  np.linspace(0, length, 300)
    xxtime = np.linspace(0, duration, 300)
    yynew = yy
    profileROI_hd = f(xxnew, yynew)
    
    length = np.shape(allKymo)[1]
    xx2 = np.arange(length)
    yy2 = np.arange(0,cleanSize-1)
    f2 = interpolate.interp2d(xx2, yy2, allKymo, kind='cubic')
    xxnew2 =  np.linspace(0, length, 300)
    xxtime2 = np.linspace(0, duration, 300)
    yynew2 = yy2
    profileROI_hd2 = f2(xxnew2, yynew2)
    
    im = ax1[0].imshow(profileROI_hd, cmap = cmap, vmin = 0)
    im2 = ax1[1].imshow(profileROI_hd2, cmap = cmap, vmin = 0)
    
    xtickslocs = ax1[0].get_xticks()[1:]
    xtickslocs[-1] = xtickslocs[-1] - 1
    new_xlabels = np.round(xxtime[(xtickslocs.astype(int))], 1) #Converting frames to second
    ax1[0].set_xticks(xtickslocs, new_xlabels)
    ax1[1].set_xticks(xtickslocs, new_xlabels)
    
    ax1[0].set_ylabel('Angle (degrees)', fontsize = ylabel)
    ax1[0].set_xlabel('Time (mins)', fontsize = xlabel)
    ax1[1].set_xlabel('Time (mins)', fontsize = xlabel)
    fig1.suptitle(currentCell, fontsize = 16)
    
    axline = find_nearest(xxtime, firstActMin)
    ax1[0].axvline(x = axline, color = 'red')
    ax1[1].axvline(x = axline, color = 'red')
    
    
    ax1[0].set_title('Normalised')
    fig1.colorbar(im, orientation='vertical', fraction = 0.055, pad = 0.04)
    
    ax1[1].set_title('Not normalised')
    fig1.colorbar(im2, orientation='vertical', fraction = 0.055, pad = 0.04)
    
    fig1.tight_layout()
    plt.savefig('{}/{}_{}_Kymo.png'.format(cp.DirDataFigToday, currentCell, channel), dpi = 100)
    plt.show()

# plt.close('all')

#%%

fluoDf = pd.DataFrame(fluoDict)
plt.figure(figsize=(15,10))

data = fluoDf[fluoDf['cellID'].str.contains('M2')]
# data = fluoDf[fluoDf['cellID'].str.contains('C9') == False]
# data = fluoDf[fluoDf['cellID'].str.contains('C13') == False]

# data = data[(data['frame'] < 27)]
# data = data[(data['frame'] > 11)]

x = data['frame']*timeRes


flatui =  ["#1111EE", "#ee1111"]
sns.set_palette(flatui)

sns.lineplot(data=data, x = x ,y="fluoFront") #, hue ='cellID')
sns.lineplot(data=data, x = x ,y="fluoBack")

control = mpatches.Patch(color=flatui[0], label='Activated rear')
activated = mpatches.Patch(color=flatui[1], label='Non-activated front')

plt.legend(handles=[activated, control], fontsize = 20, loc = 'upper left')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Time (secs)', fontsize=30)
plt.ylabel('Normalised Actin fluoresence intensity', fontsize=30)
plt.axvline(x = 120, color = 'red')
plt.tight_layout()

plt.savefig('{}/{}_{}_ActinRecruitmentvTime.png'.format(cp.DirDataFigToday, currentCell, channel), dpi = 100)

plt.show()



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

expt = '20221202_rpe1-3t3_100x_ActivationTests'
subDir = 'Rpe1Tiam_Fastact640'
dirExt = 'F:/Cortex Experiments/Fluorescence Experiments/'
dirSave = 'D:/Anumita/MagneticPincherData/DataFluorescence/Raw/'
prefix = 'cell'
channel = 'w4CSU561'

excludedCells = AllMMTriplets2Stack(dirExt, dirSave, expt = expt, prefix = prefix, channel = channel, subDir = subDir)

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






