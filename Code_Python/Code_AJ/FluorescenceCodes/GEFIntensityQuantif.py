# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:45:06 2022

@author: anumi
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

#%% Setting directories

date = '24.12.14'
dirFluo = 'D:/Anumita/MagneticPincherData/Data_Fluorescence/Chameleon/' + date
dfPathFluoro = os.path.join(cp.DirDataRaw, 'FluoData')
dfPathSegmentation = 'D:/Anumita/MagneticPincherData/Data_Fluorescence/Chameleon/24.12.14/Segmentations'

allFiles = os.listdir(dirFluo)
allMasks = [i for i in allFiles if '_seg.npy' in i]
allImages = [i for i in allFiles if '.tif' in i]

#%%

date = '24.12.14'
dirFluo = 'D:/Anumita/MagneticPincherData/Data_Fluorescence/Chameleon/' + date
dfPathFluoro = os.path.join(cp.DirDataRaw, 'FluoData')
dfPathSegmentation = 'D:/Anumita/MagneticPincherData/Data_Fluorescence/Chameleon/25.01.21/Segmentations'

allFiles = os.listdir(dirFluo)
allMasks = [i for i in allFiles if '_seg.npy' in i]
allImages = [i for i in allFiles if '.tif' in i]

#%%

fluoDf = pd.DataFrame({'dateCell' : [],
                       'mean_intensity_corr' : [],
                       'mean_background' : [],
                       'std_intensity' : [],
                       # 'mean_subtracted' : []
                       })

for img, mask in zip(allImages, allMasks):
    frame = io.imread(dirFluo+'/'+img)
    date = date.replace('.', '-')
    framename = img.split('-')[1][1:]
    mask = np.load(dirFluo+'/'+mask, allow_pickle=True).item()['masks']
    
    # frame = np.log10(frame)
    masks_binary = (mask > 0).astype(np.uint8)
    x, y, w, h = 0, 0, 20, 200
    roi_bckgd = frame[y:y+h, x:x+w]
    mean_bckgd = np.round(cv2.mean(roi_bckgd)[0], 4)
    
    # frame_no_bg = frame - mean_bckgd
    
    mean_intensity, std_intensity = np.round(cv2.meanStdDev((frame), mask=masks_binary)[0], 4)[0][0], np.round(cv2.meanStdDev(frame, mask=masks_binary)[1], 4)[0][0]
     
    
    
    dateCell = date + '_' + '_'.join(framename.split('_')[1:])
    
    new_row = pd.DataFrame({'mean_intensity_corr': [mean_intensity - mean_bckgd], 
                            'std_intensity' : [std_intensity],
                            'dateCell' : [dateCell],
                            'mean_background' : [(mean_bckgd)],
                            # 'mean_subtracted' : [mean_intensity - mean_bckgd]
                            })
     # Ensure mask is in the correct format
    fluoDf = pd.concat([fluoDf, new_row], ignore_index=True)

fluoDf.to_csv(os.path.join(dfPathFluoro, date + "_FluoData.csv"), index=False, sep = ';')

#%% Drawing segmented outlines

for img, mask in zip(allImages, allMasks):
    print(img)
    frame = io.imread(dirFluo+'/'+img)
    datMasks = np.load(dirFluo+'/'+mask, allow_pickle=True).item()
    
    # plot image with masks overlaid
    mask_RGB = plot.mask_overlay(frame, datMasks['masks'],
                            colors=np.array(datMasks['colors']))
    
    # plot image with outlines overlaid in red
    outlines = utils.outlines_list(datMasks['masks'])

    for o in outlines:
        plt.clf()  # Clear the figure (this will remove any old data from the figure)
        plt.cla()
        plt.imshow(frame)
        plt.plot(o[:,0], o[:,1], color='r')
        plt.title(img)
        plt.show()
        plt.savefig(dfPathSegmentation + '/' + img)
        
        
#%% Corrletaing blebby cells, contracting, no effect and their mean expression levels

blebPath = os.path.join(cp.DirDataAnalysis, '25-05-25_BlebbingCells_IOptoRhoA.csv')
fluoData = pd.read_csv('D:/Anumita/MagneticPincherData/Raw/FluoData/All_FluoData.csv', sep = ';')
blebData = pd.read_csv(blebPath, sep = ';')

cellid = list(blebData['cellID'].values)

plotTicks = {'color' : '#000000', 'fontsize' : 13}

dateCell = np.asarray([i.split('_')[0] + '_' + i.split('_')[2] +'_'+ i.split('_')[3] for i in cellid])

blebData['dateCell'] = dateCell

merged = fluoData.merge(blebData, on = 'dateCell', how = 'left')
merged['blebStatus'] =  merged['blebStatus'].fillna(0)

merged['mean_intensity_corr'] = np.log10(merged['mean_intensity_corr'].values)
merged['std_intensity'] = np.log10(merged['std_intensity'].values)

fig, ax = plt.subplots()

sns.pointplot(merged, x = 'dateCell', y  = 'mean_intensity_corr',
              palette = ['#0000FF', 'red'], linestyle='', hue = 'blebStatus')
# plt.errorbar(merged['dateCell'].values, merged['mean_intensity_corr'].values,fmt="none",
#              ecolor = 'k',  capsize=3, yerr = merged['std_intensity'].values)


y_labels = np.asarray([1, 10, 30, 100, 250, 500, 1000])
y_ticks = np.log10(np.asarray(y_labels))
ax.set_yticks(y_ticks, labels =y_labels, **plotTicks)
# plt.yscale('log')
x_labels = ['C'+str(i+1) for i in range(len(merged))]
ax.set_xticks(ax.get_xticks(), labels =x_labels, rotation = 90, **plotTicks)

plt.show()