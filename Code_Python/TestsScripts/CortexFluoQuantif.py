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

from scipy import interpolate, signal

from skimage import io, filters, exposure, measure, transform, util, color, draw, morphology
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

# 3. Plot settings
gs.set_smallText_options_jv()

# %% Tests

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


# %% Stacks based analysis

SCALE_63X = 8.08 # pix/um
SCALE = SCALE_63X

wavelenght_str_GFP = 'CSU-488'

def getCellsROI(I, PLOT = 0):
    nz, ny, nx = I.shape
    
    ZProfile_totalIntensity = np.sum(I, axis = (1,2))
    # maxIntensity = np.max(ZProfile_totalIntensity)
    peaks, peaks_prop = signal.find_peaks(ZProfile_totalIntensity, distance = 15)
    if 0 in peaks:
        peaks = peaks[1:]
        peaks_prop = peaks_prop[1:]
    heights = np.array([ZProfile_totalIntensity[p] for p in peaks])
    idx_max = peaks[np.argmax(heights)]
    Int_max = ZProfile_totalIntensity[idx_max]
    
    A = ZProfile_totalIntensity[::-1] < 0.975*Int_max
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
    
    if PLOT == 1:
    ## PLOT 1
        fig, ax = plt.subplots(1,1, figsize = (4,4))
        Z_plot = np.arange(nz)
        ax.plot(Z_plot, ZProfile_totalIntensity)
        
    if PLOT == 2:
    ## PLOT 2
        fig, axes = plt.subplots(1,6, figsize = (18,4))
        ax = axes[0]
        ax.imshow(I_target, cmap = 'viridis', vmin = 0, vmax = 2500)
        ax = axes[1]
        ax.imshow(I_bin1, cmap = 'viridis')
        ax = axes[2]
        ax.imshow(I_bin2, cmap = 'viridis')        
        ax = axes[3]
        ax.imshow(I_labeled + 20*(I_labeled>0), cmap = 'viridis')        
        ax = axes[4]
        ax.imshow(list_ROIs[1]['mask'], cmap = 'viridis')
        ax = axes[5]
        I_bg_plot = I_target * mask_background
        ax.imshow(I_bg_plot, cmap = 'viridis')
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
    if PLOT:
        plt.tight_layout()
        plt.show()
    
    return(list_ROIs)


def findZRegion(I, list_background, nFrames = 3, PLOT = 0):
    nz, ny, nx = I.shape
    ZProfile_totalIntensity_raw = np.mean(I, axis = (1,2))
    ZProfile_totalIntensity_1 = ZProfile_totalIntensity_raw - ZProfile_totalIntensity_raw[-1]
    ZProfile_totalIntensity = ZProfile_totalIntensity_raw - list_background
    
    # maxIntensity = np.max(ZProfile_totalIntensity)
    peaks, peaks_prop = signal.find_peaks(ZProfile_totalIntensity, distance = 15)
    if 0 in peaks:
        peaks = peaks[1:]
        peaks_prop = peaks_prop[1:]
    heights = np.array([ZProfile_totalIntensity[p] for p in peaks])
    idx_max = peaks[np.argmax(heights)]
    Int_max = ZProfile_totalIntensity[idx_max]
    
    A = ZProfile_totalIntensity[::-1] < 0.90*Int_max
    Z_start = nz - ufun.findFirst(False, A)
    
    if nFrames == 'auto':
        B = ZProfile_totalIntensity[::-1] < 0.8*Int_max
        Z_stop = nz - ufun.findFirst(False, B)
        
    else:
        Z_stop = Z_start + nFrames
    
    I_zProj = np.mean(I[Z_start:Z_stop,:,:], axis=0)
    
    if PLOT == 1:
        fig, axes = plt.subplots(1,2, figsize = (8,4))
        ax = axes[0]
        Z_plot = np.arange(nz)
        # ax.plot(Z_plot, ZProfile_totalIntensity_raw, 'r--')
        # ax.plot(Z_plot, ZProfile_totalIntensity_1, 'g--')
        ax.plot(Z_plot, ZProfile_totalIntensity)
        ax.axvline(Z_start, c='k', ls='--', lw=1)
        ax.axvline(Z_stop, c='k', ls='--', lw=1)
        ax = axes[1]
        ax.imshow(I_zProj, cmap = 'viridis', vmin = 0, vmax = 2500)
        
    if PLOT:
        plt.tight_layout()
        plt.show()
        
    return(Z_start, Z_stop)


def openQA(srcDir):
    QApath = os.path.join(srcDir, 'QA.csv')
    if os.path.isfile(QApath):
        QA_df = pd.read_csv(QApath, sep = ';')
    else:
        d = {'f':[], 'i':[], 'Q':[], 'A':[]}
        QA_df = pd.DataFrame(d)
    return(QA_df)


def readQA(QA_df, f, i):
    df = QA_df.loc[(QA_df['f'] == f) & (QA_df['i'] == i)]
    Qs = df['Q'].values
    As = df['A'].values
    if len(Qs) > 0:
        d = {Qs[i] : As[i] for i in range(len(Qs))}
    else:
        d = {}
    return(d)


def writeQA(QA_df, f, i, QADict):
    d = {'f':[], 'i':[], 'Q':[], 'A':[]}
    N = len(QADict.keys())
    for j in range(N):
        d['f'].append(f)
        d['i'].append(i)
        d['Q'].append(list(QADict.keys())[j])
        d['A'].append(list(QADict.values())[j])
    QA_df = pd.concat([QA_df, pd.DataFrame(d)], ignore_index = True)
    return(QA_df)
    

def saveQA(QA_df, dstDir):
    path = os.path.join(dstDir, 'QA.csv')
    QA_df.to_csv(path, sep = ';', index = False)
    

def computeRatio(I, background, f, i, QA_df, PLOT = 0):
    ny, nx = I.shape
    thresh = filters.threshold_otsu(I)
    I_bin1 = (I > thresh)
    I_bin2 = ndi.binary_fill_holes(I_bin1).astype(np.int64)
    I_labeled, num_features = ndi.label(I_bin2)
    
    all_props = measure.regionprops(I_labeled, intensity_image=None, cache=True, coordinates=None)
    areas = np.array([p.area for p in all_props])
    num_large_features = np.sum(areas > 100)
    
    if num_large_features > 1:
        thresh = filters.threshold_li(I)
        I_bin1 = (I > thresh)
        I_bin2 = ndi.binary_fill_holes(I_bin1).astype(np.int64)
        I_labeled, num_features = ndi.label(I_bin2)
        
    all_props = measure.regionprops(I_labeled, intensity_image=None, cache=True, coordinates=None)
    areas = np.array([p.area for p in all_props])
    perms = np.array([p.perimeter for p in all_props])
    circs = 4*np.pi*areas / perms**2
    imaxArea = np.argmax(areas)
    if circs[imaxArea] < 0.8:
        I_bin2 = ndi.binary_dilation(I_bin2, iterations = 2).astype(np.int64)
        I_bin2 = ndi.binary_fill_holes(I_bin2).astype(np.int64)
        I_labeled, num_features = ndi.label(I_bin2)
            
    I_labeled = I_labeled.astype(np.int64)
    
    all_props = measure.regionprops(I_labeled, intensity_image=None, cache=True, coordinates=None)
    cell_label = np.argmax([all_props[i].area for i in range(num_features)])
    cellSolidity = all_props[cell_label].solidity
    
    mask_cell = (I_labeled == cell_label+1).astype(np.int64)
    cellContour = measure.find_contours(mask_cell)[0]
    
    mask_cell_smooth = mask_cell[:,:]
    
    nb_it = 0
    if cellSolidity < 0.95:
        nb_it += 1
    if cellSolidity < 0.9:
        nb_it += 1
    if cellSolidity < 0.85:
        nb_it += 1
    if cellSolidity < 0.8:
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
    try:
        cellContour_inner = measure.find_contours(mask_cell_inner)[0]
    except:
        f, a = plt.subplots(1,1)
        a.imshow(I_bin2)
    mask_cell_outer = (mask_cell_whole - mask_cell_inner).astype(np.int64)
    
    QADict = readQA(QA_df, f, i)
    
    Q1 = 'Is the cell ok?'
    Q2 = 'Is the nucleus visible?'
    
    if QADict == {}:
        # Ask UI
        Choices = ['Yes', 'No']
        choicesDict = {Q1 : Choices,
                       Q2 : Choices}
        
        fig_ui, ax_ui = plt.subplots(1, 1, figsize = (5,5))
        ax = ax_ui
        ax.imshow(I, cmap = 'viridis', vmin = 0, vmax = 2500)
        ax.plot(cellContour_outer[:,1], cellContour_outer[:,0], 'r-')
        ax.plot(cellContour_inner[:,1], cellContour_inner[:,0], 'b-')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.tight_layout()
        plt.show()
        
        QADict = ufun.makeMultiChoiceBox(choicesDict)
        A1, A2 = QADict[Q1], QADict[Q2]
        plt.close(fig_ui)
        QA_df = writeQA(QA_df, f, i, QADict)
        # End of UI
    
    else:
        A1, A2 = QADict[Q1], QADict[Q2]
    
    ValidCell = (A1 == 'Yes')
    VisibleNucleus = (A2 == 'Yes')
    
    # Deal with the nucleus
    mask_cell_inner_2 = ndi.binary_erosion(mask_cell_inner, iterations = 5).astype(np.int64)
    I_inner_2 = filters.rank.median(I, footprint = morphology.square(3), mask = mask_cell_inner)
    fake_I_inner = (I_inner_2).flatten()
    m = (fake_I_inner > 0)
    fake_I_inner = fake_I_inner[m]
    # p20, p80 = np.percentile(fake_I_inner, 20), np.percentile(fake_I_inner, 80)
    if VisibleNucleus: # or (p20 < 0.4*p80):
        thresh_nucleus = 0.95*filters.threshold_li(fake_I_inner)
    else:
        thresh_nucleus = 0
    
    mask_cell_nucleus = (I < thresh_nucleus) & mask_cell_whole
    
    
    
    if PLOT == 1:
        ## PLOT
        # if ValidCell:
        #     title = 'Valid - Ratio = ' + str(ratio)
        # else:
        #     title = 'Not Valid'
        fig, axes = plt.subplots(1, 4, figsize = (4*4,5))
        # fig.suptitle(title)
        ax = axes[0]
        ax.imshow(I, cmap = 'viridis', vmin = 0, vmax = 2500)
        ax.plot(cellContour[:,1], cellContour[:,0], 'r-')
        try:
            ax.contour(mask_ellipse, [0.5], colors = 'b', linestyles = '--', linewidths = 1)
            ax.set_title('nb_it = {:.0f}'.format(nb_it))
        except:
            pass
        ax = axes[1]
        ax.imshow(mask_cell, cmap = 'viridis')
        # ax.imshow(I_inner_2, cmap = 'viridis') 
        ax = axes[2]
        # ax.imshow(I_inner_2, cmap = 'viridis')
        # print(I_inner_2.shape)
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

    if PLOT:
        plt.tight_layout()
        plt.show()
        
    # Compute the ratio if valid
    if ValidCell:
        
        props_cortex = measure.regionprops(mask_cell_outer & np.logical_not(mask_cell_nucleus), 
                                           intensity_image = I, cache=True, coordinates=None)
        props_cyto = measure.regionprops(mask_cell_inner & np.logical_not(mask_cell_nucleus), 
                                           intensity_image = I, cache=True, coordinates=None)
        
        try:
            int_cortex = props_cortex[0].intensity_mean
            int_cyto = props_cyto[0].intensity_mean
            ratio = (int_cortex - background) / max((int_cyto - background), 1)
        except:
            print(len(props_cyto))
        
    else:
        ratio = np.nan
    
    return(ratio, QA_df)


    

def stacks_CortexQuantif(srcDir, wavelenght_str, scale = SCALE):
    files = os.listdir(srcDir)
    valid_files = [f for f in files if (wavelenght_str in f and f.endswith('.TIF'))]

    QA_df = openQA(srcDir)

    L_ratio = []
    
    for f in valid_files[:]:
        f_path = os.path.join(srcDir, f)
        Iraw_allCells = io.imread(f_path)
        list_ROIs = getCellsROI(Iraw_allCells)
        for i in range(len(list_ROIs)):            
            ROI = list_ROIs[i]
            
            # ROI is a dict with keys:
            # 'mask', 'bbox' = (y1,x1,y2,x2) and 'list_background'
            (y1,x1,y2,x2) = ROI['bbox']
            (y1,x1,y2,x2) = (y1-20, x1-20, y2+20, x2+20) # Enlarge a bit
            Iraw_cell = Iraw_allCells[:,y1:y2,x1:x2]
            mask_cell = ROI['mask'][y1:y2,x1:x2]
            list_background = ROI['list_background']
            
            nZ = Iraw_cell.shape[0]
            Z_start, Z_stop = findZRegion(Iraw_cell, list_background, nFrames = 3)
            # zProj_background = np.mean(list_background[Z_start:Z_stop])
            # I_zProj = np.mean(Iraw_cell[Z_start:Z_stop,:,:], axis=0)
            
            L = []
            
            if Z_start < 0.8*nZ:
                for z in range(Z_start, Z_stop):
                    try:
                        I_z = Iraw_cell[z,:,:]
                        background_z = list_background[z]
                        ratio, QA_df = computeRatio(I_z, background_z, f, i, QA_df, PLOT = (z == Z_start+1))
                    except:
                        ratio = np.nan
                        fig_err, ax_err = plt.subplots(1,1, figsize = (5, 5))
                        ax_err.imshow(I_z, cmap = 'viridis', vmin = 0, vmax = 2500)
                        fig_err.suptitle('Major Error')
                
                    if not pd.isnull(ratio):
                        L.append(ratio)
            
            if len(L) > 0:
                L_ratio.append(np.mean(L))
                
    A_ratio = np.array(L_ratio)
    
    mean, std, N = np.mean(A_ratio), np.std(A_ratio), len(A_ratio)
    
    summaryDict = {'mean':mean,
                    'std':std,
                    'N':N,
                    }
    
    saveQA(QA_df, srcDir)
    
    return(A_ratio, summaryDict)
                



def main_CortexQuantif(srcDir, name, conditions, wavelenght_str, SCALE):
    dstDir = os.path.join(srcDir, 'Results')
    
    resDict = {}
    summaryDict = {'condition':[], 'mean':[], 'std':[], 'N':[]}
    
    for co in conditions:
        coDir = srcDir + '//' + co
        coListDir = os.listdir(coDir)
        stackDirs = [d for d in coListDir if ((os.path.isdir(os.path.join(coDir, d))) and ('Stack' in d))] #  
        if len(stackDirs) == 1:
            stacksPath = os.path.join(coDir, stackDirs[0])
            A_ratio, sD = stacks_CortexQuantif(stacksPath, wavelenght_str_GFP, scale = SCALE)
        else:
            print('Folder content is not well organised ! All stacks in one folder please !')
            
        resDict[co] = A_ratio

        summaryDict['condition'].append(co)
        summaryDict['mean'].append(sD['mean'])
        summaryDict['std'].append(sD['std'])
        summaryDict['N'].append(sD['N'])

        plt.close('all')
    
    print(summaryDict)
    print(resDict)
    summaryDf = pd.DataFrame(summaryDict)
    summaryDf.to_csv(os.path.join(dstDir, name + '.csv'), sep = ';')
    

SCALE = SCALE_63X
wavelenght_str_GFP = 'CSU-488'
conditions = ['C1_ctrl', 'C2_10x', 'C3_2x', 'C4_1x', 'C5_0.5x', 'C6_0.1x']

# %%% All conditions

# %%%% Results PNB

srcDir = "C://Users//Joseph//Desktop//DrugAnalysis//2022-11-22_DrugAssay_PNB_Hela-MyoGFP_3co"
name = 'PNB'
conditions = ['C1_ctrl', 'C2_10X', 'C3_1X']
# conditions = ['C6_2x']
wavelenght_str_GFP = 'CSU-488'

main_CortexQuantif(srcDir, name, conditions, wavelenght_str_GFP, SCALE_63X)

# %%%%% Plot

rd = {'condition': ['C1_ctrl', 'C2_10X', 'C3_1X'], 
      'concentration': [0, 400, 40], 
      }

    
df = pd.DataFrame(rd)
df = df.sort_values(by = 'concentration')

df['ste'] = df['std']/df['N']**0.5

fig, ax = plt.subplots(1,1, figsize = (8,6))
ax.errorbar(df['concentration'], df['mean'], yerr=df['ste'],
            lw = 1, color = 'k',
            marker = 'o', markerfacecolor = gs.colorList40[30], mec = 'None',
            ecolor = gs.colorList40[30], elinewidth = 1, capsize = 5)

# ax.text(df['concentration'].values, df['mean'].values, df['co name'].values)
for k in range(df.shape[0]):
    ax.text(df['concentration'].values[k]+5, df['mean'].values[k]+0.05, df['condition'].values[k], fontsize = 10)

# ax.set_xscale('log')
# ax.plot(ax.get_xlim(), [1,1], 'k--', lw=0.8)
ax.grid(axis='y')
# ax.set_xlim([-0.2, 5.2])
ax.set_ylim([0, 3])
ax.set_xlabel('Concentration of Blebbistatin (µM)')
ax.set_ylabel('Ratio Fluo Cortex/Cytoplasm')



# %%%% Results Blebbi

srcDir = "C://Users//Joseph//Desktop//DrugAnalysis//2022-12-19_DrugAssay_Blebbi_Hela-MyoGFP_6co"
name = 'blebbistatin'
conditions = ['C1_ctrl', 'C4_1x', 'C5_0.5x', 'C6_2x']
# conditions = ['C6_2x']
wavelenght_str_GFP = 'CSU-488'

main_CortexQuantif(srcDir, name, conditions, wavelenght_str_GFP, SCALE_63X)

# %%%%% Plot

rd = {'condition': ['C1_ctrl', 'C4_1x', 'C5_0.5x', 'C6_2x'], 
      'concentration': [0, 50, 25, 100], 
      'mean': [2.2373394121539687, 2.4784364952835625, 1.6285331407854653, 1.8282847052047828], 
      'std': [0.923285260844784, 0.7932843459611554, 0.5979098474447112, 0.6133135911042249], 
      'N': [19, 15, 24, 31]}

    
df = pd.DataFrame(rd)
df = df.sort_values(by = 'concentration')

df['ste'] = df['std']/df['N']**0.5

fig, ax = plt.subplots(1,1, figsize = (8,6))
ax.errorbar(df['concentration'], df['mean'], yerr=df['ste'],
            lw = 1, color = 'k',
            marker = 'o', markerfacecolor = gs.colorList40[30], mec = 'None',
            ecolor = gs.colorList40[30], elinewidth = 1, capsize = 5)

# ax.text(df['concentration'].values, df['mean'].values, df['co name'].values)
for k in range(df.shape[0]):
    ax.text(df['concentration'].values[k]+5, df['mean'].values[k]+0.05, df['condition'].values[k], fontsize = 10)

# ax.set_xscale('log')
# ax.plot(ax.get_xlim(), [1,1], 'k--', lw=0.8)
ax.grid(axis='y')
# ax.set_xlim([-0.2, 5.2])
ax.set_ylim([0, 3])
ax.set_xlabel('Concentration of Blebbistatin (µM)')
ax.set_ylabel('Ratio Fluo Cortex/Cytoplasm')


# %%%% Results Y27

srcDir = "C://Users//Joseph//Desktop//DrugAnalysis//2022-11-24_DrugAssay_Y27_Hela-MyoGFP_6co"
name = 'Y27'
conditions = ['C1_ctrl', 'C1_0.1x', 'C1_0.5x', 'C1_1x', 'C1_2x', 'C1_10x']
# conditions = ['C6_2x']
wavelenght_str_GFP = 'CSU-488'

main_CortexQuantif(srcDir, name, conditions, wavelenght_str_GFP, SCALE_63X)

# %%%%% Raw

A_ratio = np.array([3.23683413, 3.13146958, 2.63819555, 2.69039732, 1.39740912,
        1.07066333, 1.3147351 , 2.92050174, 3.26253924, 2.42681703,
        2.81725977, 2.29895761, 2.13645476, 3.08746338, 2.27516426,
        1.87860776, 1.74865955, 2.16190069, 1.74812156,
        2.64292656, 2.05856366, 1.55661962, 1.46598205, 1.07848548,
        1.47946137])
          
mean, std, N = np.mean(A_ratio), np.std(A_ratio), len(A_ratio)

# C1
# {'C1_ctrl': {'mean': 2.1809676088, 'std': 0.675737315250374, 'N': 25}}
# {'C1_ctrl': array([3.23683413, 3.13146958, 2.63819555, 2.69039732, 1.39740912,
#        1.07066333, 1.3147351 , 2.92050174, 3.26253924, 2.42681703,
#        2.81725977, 2.29895761, 2.13645476, 3.08746338, 2.27516426,
#        1.87860776, 1.74865955, 2.16190069, 1.74812156,
#        2.64292656, 2.05856366, 1.55661962, 1.46598205, 1.07848548,
#        1.47946137])}

# 4.47610907, 

# C2
# {'C2_10x': {'mean': 1.1504738982137706, 'std': 0.07706920613434011, 'N': 13}}
# {'C2_10x': array([1.18528329, 1.18142398, 1.23721024, 1.25836515, 1.02576929,
#        1.15602254, 1.13931083, 1.04267435, 1.19837782, 1.08196677,
#        1.09937366, 1.08102726, 1.26935549])}

# C3
# {'C3_2x': {'mean': 1.2346581126149598, 'std': 0.05463472858717262, 'N': 24}}
# {'C3_2x': array([1.23792763, 1.26650248, 1.30504833, 1.32473531, 1.26383796,
#        1.27921931, 1.24355993, 1.13074372, 1.18924433, 1.24569422,
#        1.29192879, 1.1933749 , 1.18057645, 1.29391318, 1.19989459,
#        1.2853424 , 1.20199902, 1.16103473, 1.28171618, 1.15308795,
#        1.28837453, 1.25994693, 1.1554759 , 1.19861595])}

# C4
# {'C4_1x': {'mean': 1.1734189803955142, 'std': 0.07904788243336819, 'N': 14}}
# {'C4_1x': array([0.98279757, 1.11206443, 1.10601043, 1.15840632, 1.19878053,
#        1.2665215 , 1.27535678, 1.17116013, 1.23026342, 1.2586848 ,
#        1.12370546, 1.26165763, 1.13528979, 1.14716694])}

# C5
# {'C5_0.5x': {'mean': 1.2075531230942547, 'std': 0.07823196133748052, 'N': 19}}
# {'C5_0.5x': array([0.99812369, 1.30317142, 1.21857554, 1.24029143, 1.32973095,
#        1.23556649, 1.29757038, 1.32373535, 1.13420473, 1.2291747 ,
#        1.14789419, 1.18987663, 1.17815803, 1.22504529, 1.18454485,
#        1.15744863, 1.25906017, 1.12934795, 1.16198892])}


# C6
# {'C6_0.1x': {'mean': 1.358150332864929, 'std': 0.11524630181105491, 'N': 15}}
# {'C6_0.1x': array([1.47223653, 1.53994367, 1.33303321, 1.22742962, 1.40657092,
#        1.30616066, 1.39737098, 1.4915153 , 1.49719887, 1.31580211,
#        1.19192314, 1.1174486 , 1.3135526 , 1.40689419, 1.35517459])}

# %%%%% Plot

gs.set_default_options_jv()

listCond = ['ctrl', '0.1x', '0.5x', '1x', '2x', '10x']
listConc = np.array([0, 0.05, 0.25, 0.5, 1.0, 5.0])*100 # µM
listMean = [2.1809, 1.3581, 1.2075, 1.1734, 1.2346, 1.1504]
listSte = [0.6757, 0.1152, 0.078, 0.0790, 0.0546, 0.0770]
    
df = pd.DataFrame({'co name':listCond, 
                   'concentration':listConc, 
                   'mean':listMean, 
                   'ste':listSte})

fig, ax = plt.subplots(1,1, figsize = (8,6))
ax.errorbar(df['concentration'], df['mean'], yerr=df['ste'],
            lw = 1, color = 'k',
            marker = 'o', markerfacecolor = gs.colorList40[30], mec = 'None',
            ecolor = gs.colorList40[30], elinewidth = 1, capsize = 5)

# ax.text(df['concentration'].values, df['mean'].values, df['co name'].values)
for k in range(df.shape[0]):
    ax.text(df['concentration'].values[k]+5, df['mean'].values[k]+0.05, df['co name'].values[k], fontsize = 10)

# ax.set_xscale('log')
# ax.plot(ax.get_xlim(), [1,1], 'k--', lw=0.8)
ax.grid(axis='y')
# ax.set_xlim([-0.2, 5.2])
ax.set_ylim([0, 3])
ax.set_xlabel('Concentration of Y27 (µM)')
ax.set_ylabel('Ratio Fluo Cortex/Cytoplasm')






# %% Snapshot based analysis - LatA

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
for i in range(N):
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

fig, ax = plt.subplots(1,1, figsize = (8,6))
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

    
    
    
    