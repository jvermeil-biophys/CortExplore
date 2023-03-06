# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 18:31:48 2022

@author: Joseph
"""

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
import traceback

from scipy import interpolate
from scipy import signal

from skimage import io, filters, exposure, measure, transform, util, color, draw
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import linear_sum_assignment
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from datetime import date

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)

import GraphicStyles as gs
import GlobalConstants as gc
import UtilityFunctions as ufun


# 2. Pandas settings
pd.set_option('mode.chained_assignment', None)


# %% Stacks based analysis

SCALE_63X = 8.08 # pix/um
SCALE = SCALE_63X

wavelenght_str_GFP = 'CSU-488'

def getCellsROI(I):
    nz, ny, nx = I.shape
    
    ZProfile_totalIntensity = np.sum(I, axis = (1,2))
    maxIntensity = np.max(ZProfile_totalIntensity)
    A = ZProfile_totalIntensity[::-1] < 0.975*maxIntensity
    Z_target = nz - ufun.findFirst(False, A)
    I_target = I[Z_target,:,:]
    
    thresh = filters.threshold_li(I_target)
    I_bin1 = (I_target > thresh)
    I_bin2 = ndi.binary_fill_holes(I_bin1).astype(np.int64)
    
    I_labeled, num_features = ndi.label(I_bin2)
    I_labeled = I_labeled.astype(np.int64)
    
    all_props = measure.regionprops(I_labeled, intensity_image=None, cache=True, coordinates=None)
    valid_labels = [all_props[i].area > 1e4 for i in range(num_features)]
    
    mask_background = np.ones_like(I_bin2) - I_bin2
    mask_background = ndi.binary_erosion(mask_background, iterations = 25).astype(np.int64)
    list_background = np.array([np.median(I[i,:,:] * mask_background) for i in range(nz)])
    
    
    list_ROIs = []
    for i in range(num_features):
        if valid_labels[i]:
            dict_ROI = {'bbox' : all_props[i].bbox,
                        'mask' : (I_labeled == i+1),
                        'list_background' : list_background}
            list_ROIs.append(dict_ROI)
    
    ## PLOT 1
    # Z_plot = np.arange(nz)
    # plt.plot(Z_plot, ZProfile_totalIntensity)
    ## PLOT 2
    # fig, axes = plt.subplots(1,6, figsize = (18,4))
    # ax = axes[0]
    # ax.imshow(I_target, cmap = 'viridis', vmin = 0, vmax = 2500)
    # ax = axes[1]
    # ax.imshow(I_bin1, cmap = 'viridis')
    # ax = axes[2]
    # ax.imshow(I_bin2, cmap = 'viridis')        
    # ax = axes[3]
    # ax.imshow(I_labeled + 20*(I_labeled>0), cmap = 'viridis')        
    # ax = axes[4]
    # ax.imshow(list_ROIs[1]['mask'], cmap = 'viridis')
    # ax = axes[5]
    # I_bg_plot = I_target * mask_background
    # ax.imshow(I_bg_plot, cmap = 'viridis')
    # for ax in axes:
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])
    # plt.tight_layout()
    # plt.show()
    
    return(list_ROIs)


def findZRegion(I, list_background):
    nz, ny, nx = I.shape
    
    ZProfile_totalIntensity = np.mean(I, axis = (1,2)) - list_background
    maxIntensity = np.max(ZProfile_totalIntensity)
    A = ZProfile_totalIntensity[::-1] < 0.975*maxIntensity
    Z_start = nz - ufun.findFirst(False, A)
    B = ZProfile_totalIntensity[::-1] < 0.65*maxIntensity
    Z_stop = nz - ufun.findFirst(False, B)
    
    I_zProj = np.mean(I[Z_start:Z_stop,:,:], axis=0)
    
    # fig, axes = plt.subplots(1,2, figsize = (8,4))
    # ax = axes[0]
    # Z_plot = np.arange(nz)
    # ax.plot(Z_plot, ZProfile_totalIntensity)
    # ax.axvline(Z_start, c='k', ls='--', lw=1)
    # ax.axvline(Z_stop, c='k', ls='--', lw=1)
    # ax = axes[1]
    # ax.imshow(I_zProj, cmap = 'viridis', vmin = 0, vmax = 2500)
    
    return(Z_start, Z_stop)
    

def quantifCortexRatio(I):
    ny, nx = I.shape
    thresh = filters.threshold_otsu(I)
    I_bin1 = (I > thresh)
    I_bin2 = ndi.binary_fill_holes(I_bin1).astype(np.int64)
    
    I_labeled, num_features = ndi.label(I_bin2)
    I_labeled = I_labeled.astype(np.int64)
    
    all_props = measure.regionprops(I_labeled, intensity_image=None, cache=True, coordinates=None)
    cell_label = np.argmax([all_props[i].area for i in range(num_features)])
    cellSolidity = all_props[cell_label].solidity
    
    mask_cell = (I_labeled == cell_label+1).astype(np.int64)
    cellContour = measure.find_contours(mask_cell)[0]
    
    mask_cell_smooth = mask_cell[:,:]
    
    nb_it = 0
    if cellSolidity < 0.975:
        nb_it += 1
    if cellSolidity < 0.95:
        nb_it += 1
    if cellSolidity < 0.925:
        nb_it += 1
    if cellSolidity < 0.9:
        nb_it += 1
    
    ## PLOT
    # fig, axes = plt.subplots(1, nb_it + 1, figsize = (3*(nb_it + 1), 4))
    # ax = axes[0]
    # ax.imshow(I_cell, cmap = 'viridis')
    
    for it in range(nb_it):
    
        cellContour_smooth = measure.find_contours(mask_cell_smooth)[0]
        
        ellipse = measure.EllipseModel()
        ellipse.estimate(cellContour_smooth[:,::-1])
        xc, yc, a, b, theta = ellipse.params
        
        rr_ellipse, cc_ellipse = draw.ellipse(int(yc), int(xc), int(b), int(a), 
                                  shape=None, rotation=theta)
        yx_ellipse = np.array([rr_ellipse, cc_ellipse]).T
        
        mask_ellipse = np.zeros_like(mask_cell)
        mask_ellipse[rr_ellipse, cc_ellipse] = 1
        
        mask_cell_smooth = mask_cell_smooth * mask_ellipse
        
        ## PLOT
        # ax = axes[it+1]
        # ax.contour(mask_ellipse, [0.5], colors='r', linewidths = 1)
        # ax = axes[it+1]
        # ax.imshow(I, cmap = 'viridis')
        
    mask_cell_whole = mask_cell_smooth[:,:]
    cellContour_outer = measure.find_contours(mask_cell_whole)[0]
    mask_cell_inner = ndi.binary_erosion(mask_cell_whole, iterations = 12).astype(np.int64)
    cellContour_inner = measure.find_contours(mask_cell_inner)[0]
    mask_cell_outer = (mask_cell_whole - mask_cell_inner).astype(np.int64)
    
    fake_I_inner = (I * mask_cell_inner).flatten()
    p20, p80 = np.percentile(fake_I_inner, 20), np.percentile(fake_I_inner, 80)
    m = (fake_I_inner > 0)
    fake_I_inner = fake_I_inner[m]
    p20, p80 = np.percentile(fake_I_inner, 20), np.percentile(fake_I_inner, 80)
    if p20 < 0.5*p80:
        thresh_nucleus = 0.8*filters.threshold_li(fake_I_inner)
    else:
        thresh_nucleus = 0
    
    # thresh_otsu = filters.threshold_otsu(fake_I_inner)
    # print('otsu : ', thresh_otsu)
    # thresh_li = filters.threshold_li(fake_I_inner)
    # print('li : ', thresh_li)
    # thresh_yen = filters.threshold_yen(fake_I_inner)
    # print('yen : ', thresh_yen)
    # thresh_min = filters.threshold_minimum(fake_I_inner)
    # print('min : ', thresh_min)
    # thresh_mean = filters.threshold_mean(fake_I_inner)
    # print('mean : ', thresh_mean)
    # thresh_iso = filters.threshold_isodata(fake_I_inner)
    # print('iso : ', thresh_iso)
    # print('')
    
    mask_cell_nucleus = (I < thresh_nucleus)

    ## PLOT
    fig, axes = plt.subplots(1, 4, figsize = (4*4,5))
    ax = axes[0]
    ax.imshow(I, cmap = 'viridis', vmin = 0, vmax = 2500)
    ax.plot(cellContour[:,1], cellContour[:,0], 'r-')
    try:
        ax.contour(mask_ellipse, [0.5], colors = 'b', linestyles = '--', linewidths = 1)
        ax.set_title('nb_it = {:.0f}'.format(nb_it))
    except:
        pass
    
    # ax = axes[1]
    # ax.imshow(I_bin2, cmap = 'viridis')
    # ax = axes[2]
    # ax.imshow(I_labeled + 20*(I_labeled>0), cmap = 'viridis')        
    ax = axes[1]
    ax.imshow(mask_cell, cmap = 'viridis')  
    ax = axes[2]
    ax.imshow(I, cmap = 'viridis', vmin = 0, vmax = 2500)
    ax.plot(cellContour_outer[:,1], cellContour_outer[:,0], 'r-')
    ax.plot(cellContour_inner[:,1], cellContour_inner[:,0], 'b-')
    ax.contour(mask_cell_nucleus, [0.5], colors = 'y', linestyles = '-', linewidths = 1)
    ax = axes[3]
    ax.imshow(3*mask_cell_outer + 2 * mask_cell_inner - mask_cell_nucleus, cmap = 'viridis')

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.tight_layout()
    plt.show()
    

def main_CortexIntensity_Stacks(srcDir, wavelenght_str, scale = SCALE):
    files = os.listdir(srcDir)
    valid_files = [f for f in files if (wavelenght_str in f and f.endswith('.TIF'))]
    for f in valid_files:
        f_path = os.path.join(srcDir, f)
        Iraw_allCells = io.imread(f_path)
        list_ROIs = getCellsROI(Iraw_allCells)
        for ROI in list_ROIs:
            # ROI is a dict with keys:
            # 'mask' and 'bbox' = (y1,x1,y2,x2)
            (y1,x1,y2,x2) = ROI['bbox']
            (y1,x1,y2,x2) = (y1-20, x1-20, y2+20, x2+20) # Enlarge a bit
            Iraw_cell = Iraw_allCells[:,y1:y2,x1:x2]
            mask_cell = ROI['mask'][y1:y2,x1:x2]
            list_background = ROI['list_background']
            
            Z_start, Z_stop = findZRegion(Iraw_cell, list_background)
            I_zProj = np.mean(Iraw_cell[Z_start:Z_stop,:,:], axis=0)
            
            quantifCortexRatio(I_zProj)
        
# %%% Test
srcDir = 'C://Users//Joseph//Desktop//2022-11-24_DrugAssay_Y27_Hela-MyoGFP_6co//C1_ctrl//Stack1_63x_488_500ms_stack250nmX101imgs_laser10p'
# srcDir = 'C://Users//Joseph//Desktop//2022-11-22_DrugAssay_PNB_Hela-MyoGFP_6co//C1_ctrl//Stack1_63x_488_500ms_stack250nmX101imgs_laser10p'

main_CortexIntensity_Stacks(srcDir, wavelenght_str_GFP, scale = SCALE)



# %% Snapshot based analysis

# %%%
gs.set_smallText_options_jv()
dirPath = 'C://Users//Joseph//Desktop//2022-11-02_DrugAssay_LatA_3T3aSFL-LaGFP_6co//C1_Ctrl//UniqueCells'
files = os.listdir(dirPath)
N = len(files)
nColsPlot = 8
nRowsPlot =  + N//8

fig, ax = plt.subplots(nRowsPlot, nColsPlot)

listRatioOutIn = []
listDf = []
listCellId = []
for i in range(1,2):
    ip, jp = i//nColsPlot, i%nColsPlot
    f = files[i]
    filePath = os.path.join(dirPath, f)
    I = io.imread(filePath)
    thresh = filters.threshold_li(I)
    I_bin = (I>thresh)
    
    I_lab, num_features = ndi.label(I_bin)
    
    all_props = measure.regionprops(I_lab, intensity_image=None, cache=True, coordinates=None)
    
    i_lab = np.argmax([all_props[i].area for i in range(num_features)]) + 1
    
    I_cell = (I_lab == i_lab).astype(np.int64)
    I_cell = ndi.binary_fill_holes(I_cell).astype(np.int64)
    
    I_cell = ndi.binary_erosion(I_cell, iterations = 2).astype(np.int64)
    
    I_inside = ndi.binary_erosion(I_cell, iterations = 9).astype(np.int64)
    
    ax[ip, jp].imshow(I)
    ax[ip, jp].contour(I_cell, [0.5], colors='y', linewidths = 1)
    ax[ip, jp].contour(I_inside, [0.5], colors='r', linewidths = 1)
    ax[ip, jp].set_xticks([])
    ax[ip, jp].set_yticks([])
    ax[ip, jp].set_xticklabels([])
    ax[ip, jp].set_yticklabels([])
    axtitle = 'Cell {:.0f}'.format(i)
    
    I_cell_inout = (I_cell + I_inside)
    
    cell_props = measure.regionprops(I_cell, intensity_image=I, cache=True, coordinates=None)
    # cell_props[0].intensity_mean
    inout_props = measure.regionprops(I_cell_inout, intensity_image=I, cache=True, coordinates=None)
    out_props, in_props = inout_props
    
    AR = cell_props[0].axis_major_length / cell_props[0].axis_minor_length

    if AR > 1.3:
        axtitle+= '\nExcluded, AR = {:.2f} > 1.3'.format(AR)
        
    else:
        listProps = [cell_props[0], in_props, out_props]
        
        listRegions = ['whole', 'in', 'out']
        listArea = [p.area for p in listProps]
        listMeanInt = [p.intensity_mean for p in listProps]
        
        ratioOutIn = listMeanInt[2] / listMeanInt[1]
        listRatioOutIn.append(listMeanInt[2] / listMeanInt[1])
        listCellId.append(f)
        
        d = {'f':[f, f, f], 'region':listRegions, 'area':listArea, 'mean int':listMeanInt}
        df = pd.DataFrame(d)
        listDf.append(df)
        
        axtitle+= '\nRatio Out/In = {:.2f}'.format(ratioOutIn)
        
    
        
    ax[ip, jp].set_title(axtitle)

Nval = len(listRatioOutIn)
m = np.mean(listRatioOutIn)
se = np.std(listRatioOutIn)/Nval**0.5
    
print(m, se)
    

ax = np.reshape(ax, (nRowsPlot, nColsPlot))
plt.tight_layout()
plt.show()
    
    
# %%% Summary plot
# Quantif 1 - 
# ctrl - 1.6175219043739675 0.09831368478048526
# 0.1x - 1.3117575053020685 0.05152019788610142
# 0.5x - 1.1553079524030232 0.04339410726074064
# 1x - 1.1334584265835206 0.042784119056154486
# 2x - 1.0881305698985289 0.05720244677786476
# 10x - 0.9468779097526224 0.02352824625082879
gs.set_default_options_jv()

listCond = ['ctrl', '0.1x', '0.5x', '1x', '2x', '10x']
listConc = [0, 0.05, 0.25, 0.5, 1.0, 5.0] # µM
listMean = [1.6175, 1.3117, 1.1553, 1.1334, 1.0881, 0.9469]
listSte = [0.0983, 0.05152, 0.04339, 0.04278, 0.05720, 0.02352]
    
df = pd.DataFrame({'co name':listCond, 
                   'concentration':listConc, 
                   'mean':listMean, 
                   'ste':listSte})

fig, ax = plt.subplots(1,1, figsize = (12,6))
ax.errorbar(df['concentration'], df['mean'], yerr=df['ste'],
            lw = 0.5, color = 'k',
            marker = 'o', markerfacecolor = gs.colorList40[30], mec = 'None',
            ecolor = gs.colorList40[30], elinewidth = 1, capsize = 5)

# ax.text(df['concentration'].values, df['mean'].values, df['co name'].values)
for k in range(df.shape[0]):
    ax.text(df['concentration'].values[k]+0.05, df['mean'].values[k]+0.01, df['co name'].values[k], fontsize = 10)

# ax.set_xscale('log')
# ax.plot(ax.get_xlim(), [1,1], 'k--', lw=0.8)
ax.grid(axis='y')
ax.set_xlim([-0.2, 5.2])
ax.set_ylim([0, 1.75])
ax.set_xlabel('Concentration of LatA (µM)')
ax.set_ylabel('Ratio Fluo Cortex/Cytoplasm')

    
    
    
    