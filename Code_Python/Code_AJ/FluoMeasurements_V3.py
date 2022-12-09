# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:45:06 2022

@author: anumi


To use this code, run the following in the Anaconda Powershell:
    
conda create -n cellpose pytorch=1.8.2 cudatoolkit=10.2 -c pytorch-lts
conda activate cellpose
pip install cellpose
conda install seaborn pandas syder-kernels opencv

Make sure to have your grahical backend as Inline and not Qt, as cellpose affects the Qt package version.

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
from cellpose import plot, models, utils
from cellpose.io import imread
# from skimage import measure

import CortexPaths as cp
os.chdir(cp.DirRepoPython)
import GraphicStyles as gs

# os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = "D:/Anumita/MagneticPincherData/DataFluorescence/CellposeModels"

#%%

dirFluo = cp.DirData + '/DataFluorescence/Test/stackblur4'

model = models.Cellpose(gpu=False, model_type='cyto')

files = os.listdir(dirFluo)
imgs = [imread(os.path.join(dirFluo, f)) for f in files]

#%%

masks, flows, styles, diams = model.eval(imgs, diameter = 165, channels=[0, 0],
                                         flow_threshold = 0.2, do_3D=False, normalize = True)

#%%
for each in masks:
    plt.imshow(each)
    plt.show()

plt.close('all')
#%%

allKymo = []
allKymo_norm = []

R_in = 40 #in px
innerCell = 5

spline_prev = 0

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
    warped_mask = np.zeros((warped_img.shape[0], warped_img.shape[1]))
    warped_copy = np.zeros(len(warped_img)-1)
    
    maxValues = np.argmax(warped_img, axis = 1)
    maxInter = signal.savgol_filter(maxValues, 301, 3)
    
    print(i)
    for j in range(len(warped_img)-1):
        # print(j)
        maxval = int(maxInter[j])
        warped_copy[j] = np.average(warped_img[j, maxval - innerCell:maxval + innerCell])
        warped_mask[j, maxval - innerCell:maxval + innerCell] = 1
    
    final = warped_img
    plt.imshow(final)
    plt.show()

    allKymo.append(warped_copy)

allKymo = np.asarray(allKymo).T

kymo_norm = allKymo/allKymo[0]
plt.imshow(kymo_norm)

plt.figure(figsize = (15,10))
length = np.shape(kymo_norm)[1]
xx = np.arange(length)
cleanSize = 360
yy = np.arange(0,cleanSize-1)
f = interpolate.interp2d(xx, yy, allKymo, kind='cubic')
xxnew =  np.linspace(0, length, 300)
yynew = yy
profileROI_hd = f(xxnew, yynew)


plt.imshow(profileROI_hd)
plt.colorbar()

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






