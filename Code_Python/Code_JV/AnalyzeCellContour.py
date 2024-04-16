# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 18:08:50 2023

@author: JosephVermeil
"""

# %% Imports

import os
import cv2

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import UtilityFunctions as ufun
import GraphicStyles as gs

# from skimage import io, transform, filters, draw, exposure, measure


import skimage as skm
# import cellpose as cpo
from cellpose import models
from scipy import interpolate
from scipy import signal

# from cellpose.io import imread

import shapely
from shapely.ops import polylabel
# from shapely import Polygon, LineString, get_exterior_ring, distance
from shapely.plotting import plot_polygon, plot_points, plot_line


gs.set_mediumText_options_jv()

SCALE_100X_ZEN = 7.4588

# %% Utility functions


#### Utility functions

def autocorr_np(x):
    """The good one"""
    x = np.array(x) 
    # Mean
    mean = np.mean(x)
    # Variance
    var = np.var(x)
    # Normalized data
    ndata = x - mean
    acorr = np.correlate(ndata, ndata, 'full')[len(ndata)-1:] 
    acorr = (acorr / var) / len(ndata)
    return(acorr)




def argmedian(x):
    if len(x)%2 == 0:
        x = x[:-1]
    return(np.argpartition(x, len(x) // 2)[len(x) // 2])



def looping_index(a, start, stop): 
    # fn to convert your start stop to a wrapped range
    if stop<=start:
        stop += len(a)
    return(np.arange(start, stop)%len(a))


# a = np.arange(1, 7, dtype = int)
# b = a[looping_index(a,4,1)]  # or
# c = np.take(a,looping_index(a,4,1))




#### A Viterbi tracking function

def ViterbiPathFinder(treillis):
    
    def node2node_distance(node1, node2):
        """Define a distance between nodes of the treillis graph"""
        d = (np.abs(node1['pos'] - node2['pos']))**2 # + np.abs(node1['val'] - node2['val'])
        return(d)
    
    N_row = len(treillis)
    for i in range(1, N_row):
        current_row = treillis[i]
        previous_row = treillis[i-1]
        
        # if len(current_row) == 0:
        #     current_row = [node.copy() for node in previous_row]
            
        for node1 in current_row:
            costs = []
            for node2 in previous_row:
                c = node2node_distance(node1, node2) + node2['accumCost']
                costs.append(c)
            best_previous_node = np.argmin(costs)
            accumCost = np.min(costs)
            node1['previous'] = best_previous_node
            node1['accumCost'] = accumCost
    
    final_costs = [node['accumCost'] for node in treillis[-1]]
    i_best_arrival = np.argmin(final_costs)
    best_path = [i_best_arrival]
    
    for i in range(N_row-1, 0, -1):
        node = treillis[i][best_path[-1]]
        i_node_predecessor = node['previous']
        best_path.append(i_node_predecessor)
    
    best_path = best_path[::-1]
    nodes_list = [treillis[k][best_path[k]] for k in range(len(best_path))]
            
    return(best_path, nodes_list)



#### Test

# A "treillis" is a type of graph
# My understanding of it is the following
# > It has N "rows" of nodes
# > Each row has M nodes
# > Each node of a given row are linked with all the nodes of the row before
#   as well as with all the nodes of the row after
#   by vertices of different weights.
# > The weights of the vertices correspond to the cost computed by the home-made cost function.
# In the case of the Viterbi tracking, the goal is to go through the treillis graph, row after row,
# and find the best path leading to each node of the current row.
# At the end of the graph, the node with the lowest "accumCost" is the best arrival point.
# One just have to back track from there to find the best path.

# treillis = []

# node_A1 = {'pos':0,'val':10,'t':0,'previous':-1,'accumCost':0}
# row_A = [node_A1]

# node_B1 = {'pos':-35,'val':10,'t':1,'previous':-1,'accumCost':0}
# node_B2 = {'pos':10,'val':10,'t':1,'previous':-1,'accumCost':0}
# node_B3 = {'pos':5,'val':10,'t':1,'previous':-1,'accumCost':0}
# row_B = [node_B1, node_B2, node_B3]

# node_C1 = {'pos':-20,'val':10,'t':2,'previous':-1,'accumCost':0}
# node_C2 = {'pos':40,'val':10,'t':2,'previous':-1,'accumCost':0}
# row_C = [node_C1, node_C2]

# node_D1 = {'pos':-35,'val':10,'t':3,'previous':-1,'accumCost':0}
# node_D2 = {'pos':15,'val':10,'t':3,'previous':-1,'accumCost':0}
# row_D = [node_D1, node_D2]

# row_E = [node.copy() for node in row_A]

# treillis = [row_A, row_B, row_C, row_D, row_E]

# ViterbiPathFinder(treillis)



#### Cell approx detection

def cellPose_centerFinder(I0,  model_type='TN1', diameter=110, 
                          flow_threshold = 0.4, cellprob_threshold = 0.4):
    model = models.CellposeModel(gpu=False, model_type=model_type)
    channels = [0, 0]
    [mask], [flow], [style] = model.eval([I0], diameter = diameter, channels=channels,
                                        flow_threshold = flow_threshold, cellprob_threshold = cellprob_threshold, 
                                        do_3D=False, normalize = True) # masks, flows, styles, diams
    Y, X = ndi.center_of_mass(mask, labels=mask, index=1)
    Y, X = round(Y), round(X)
    
    [contour] = skm.measure.find_contours(mask, 0.5)
    
    return((Y, X), mask, contour)





def findCellInnerCircle(I0, plot=False):
    th1 = skm.filters.threshold_isodata(I0)
    I0_bin = (I0 > th1)
    I0_bin = ndi.binary_opening(I0_bin, iterations = 2)
    I0_label, num_features = ndi.label(I0_bin)
    df = pd.DataFrame(skm.measure.regionprops_table(I0_label, I0, properties = ['label', 'area']))
    df = df.sort_values(by='area', ascending=False)
    i_label = df.label.values[0]
    I0_rawCell = (I0_label == i_label)
    I0_rawCell = ndi.binary_fill_holes(I0_rawCell)
    [contour_rawCell] = skm.measure.find_contours(I0_rawCell, 0.5)
    
    polygon_cell = shapely.Polygon(contour_rawCell[:, ::-1])
    center = polylabel(polygon_cell, tolerance=1)
    exterior_ring_cell = shapely.get_exterior_ring(polygon_cell)
    R = shapely.distance(center, exterior_ring_cell)
    circle = center.buffer(R)
    
    X, Y = list(center.coords)[0]
    
    if plot:
        fig, axes = plt.subplots(1,2, figsize = (8,4))
        ax = axes[0]
        ax.imshow(I0_rawCell, cmap='gray')
        ax = axes[1]
        ax.imshow(I0, cmap='gray')
        
        for ax in axes:
            plot_polygon(circle, ax=ax, add_points=False, color='green')
            plot_points(center, ax=ax, color='green', alpha=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.tight_layout()
        plt.show()
    
    Y, X = round(Y), round(X)
    return((Y, X), R)




def getValidROIBounds(I, x10, y10, x20, y20):
    ny, nx = I.shape
    Li = [x10, y10, x20, y20]
    Lm = [nx, ny, nx, ny]
    Lr = []
    for a, m in zip(Li, Lm):
        a = min(m, a)
        a = max(0, a)
        Lr.append(a)
    return(Lr)



# %% Run on all Zstacks

PLOT = 1
SAVE = 0
plt.ioff()

srcDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//"
fileList = os.listdir(srcDir)
stackList = []

dstDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//test1"

acorrMatrix = []
radiusList = []

for f in fileList:
    if '_ZS_' in f:
        stackList.append(f)

for f in stackList[:1]:
    filePath = os.path.join(srcDir, f)

    I = skm.io.imread(filePath)
    sumArray = np.sum(I, axis=(1,2))
    z0 = np.argmax(sumArray)
    
    print(f, z0)
    
    for kk in range(16, 24):# 30):
        
        I0 = I[z0+kk]
        
        #### Settings
        framesize = int(35*8/2) 
        inPix, outPix = 20, 10
        
        #### 1. Detect
        (Yc0, Xc0), mask, contour = cellPose_centerFinder(I0,  model_type='TN1', diameter=110, 
                                  flow_threshold = 0.4, cellprob_threshold = 0.4)
        
        I0 = I0[Yc0-framesize:Yc0+framesize, Xc0-framesize:Xc0+framesize]
        
        (Yc, Xc), mask, contour = cellPose_centerFinder(I0,  model_type='TN1', diameter=110, 
                                  flow_threshold = 0.4, cellprob_threshold = 0.4)
        
        # [contour] = skm.measure.find_contours(mask, 0.5) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
        listDistances = np.sqrt((contour[:,0] - Yc)**2 + (contour[:,1] - Xc)**2)
        meanRadius = np.mean(listDistances)
        print(f, kk, meanRadius)
        radiusList.append(meanRadius)
        
        fig_I, axes_I = plt.subplots(1,2, figsize = (12, 6), sharey = True)
        ax = axes_I[0]
        ax.imshow(I0, cmap='gray')
        ax.plot(contour[:,1], contour[:,0], 'r--', lw=1, zorder = 4)
        ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
        theta = np.linspace(0, 2*np.pi, 100)
        CircleY, CircleX = Yc + meanRadius*np.sin(theta), Xc + meanRadius*np.cos(theta)
        # ax.plot(CircleX, CircleY, c='cyan', ls='--', lw=1)
        
        ax = axes_I[1]
        ax.imshow(I0, cmap='gray')
        ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
        CircleYin, CircleXin = Yc + (meanRadius-inPix)*np.sin(theta), Xc + (meanRadius-inPix)*np.cos(theta)
        ax.plot(CircleXin, CircleYin, c='gold', ls='--', lw=1)
        CircleYout, CircleXout = Yc + (meanRadius+outPix+1)*np.sin(theta), Xc + (meanRadius+outPix+1)*np.cos(theta)
        ax.plot(CircleXout, CircleYout, c='gold', ls='--', lw=1)
        
        #### 2. Warp
        
        warped = skm.transform.warp_polar(I0, center=(Yc, Xc), radius=meanRadius*1.3, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        warped_mask = skm.transform.warp_polar(mask, center=(Yc, Xc), radius=meanRadius*1.3, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        
        [warped_contour] = skm.measure.find_contours(warped_mask) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
        mask_integer = [warped_contour[i, 0] == int(warped_contour[i, 0]) for i in range(len(warped_contour[:, 0]))]
        warped_contour_int = warped_contour[mask_integer]
        warped_contour_int = warped_contour_int.astype(int)
        
        ny, nx = warped.shape
        yy = np.arange(0, ny)
        warped_cortex_region = np.zeros((ny, 1 + inPix + outPix), dtype = np.float64)
        
        for i in range(ny):
            warped_cortex_region[i, :] = warped[i, warped_contour_int[i,1]-inPix:warped_contour_int[i,1]+outPix+1]
        interp_factor = 5
        warped_cortex_region_interp = ufun.resize_2Dinterp(warped_cortex_region, fx=interp_factor, fy=1)
        max_values = np.max(warped_cortex_region_interp, axis = 1)
        sum_values = np.sum(warped_cortex_region_interp, axis = 1)
        max_position_0 = np.argmax(warped_cortex_region_interp, axis = 1)
        max_position_1 = (max_position_0/interp_factor) + warped_contour_int[:,1] - inPix
        
        def deg2um(a):
            return(a * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN))
    
        def um2deg(x):
            return(x * (360/2*np.pi) / (meanRadius/SCALE_100X_ZEN))
        
        secax = ax.secondary_yaxis('right', functions=(deg2um, um2deg))
        secax.set_ylabel('curv abs [µm]')
        
        def autocorr(x):
            n = x.size
            norm = (x - np.mean(x))
            result = np.correlate(norm, norm, mode='same')
            acorr = result[n//2 + 1:] / x.var() # * np.arange(n-1, n//2, -1))
            return(acorr)
        
        values = sum_values
        
        
        TT = np.arange(0, 360, 1, dtype=int)
        bspl = interpolate.make_interp_spline(TT, values, k = 3)
        
        interp_factor = 10
        TT_new = np.arange(0, 360, 1/interp_factor, dtype=float)
        values_interp = bspl(TT_new)
        
        acorr_0 = autocorr(values_interp)
        values_12rep = np.array(list(values_interp) * 12)
        
        acorr_12rep = autocorr(values_12rep)
        acorr_1 = acorr_12rep[3*ny*interp_factor:3*ny*interp_factor + (interp_factor*ny)//2]

        # figtest, axtest = plt.subplots(1, 1)
        # axtest.plot(TT, values, 'b-')
        # axtest.plot(TT_new, values_interp, 'r-')
        # figtest.show()
        
        if PLOT:
            fig_I, axes_I = plt.subplots(1,2, figsize = (12, 6), sharey = True)
            ax = axes_I[0]
            ax.imshow(I0, cmap='gray')
            ax.plot(contour[:,1], contour[:,0], 'r--', lw=1, zorder = 4)
            ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
            theta = np.linspace(0, 2*np.pi, 100)
            CircleY, CircleX = Yc + meanRadius*np.sin(theta), Xc + meanRadius*np.cos(theta)
            # ax.plot(CircleX, CircleY, c='cyan', ls='--', lw=1)
            
            ax = axes_I[1]
            ax.imshow(I0, cmap='gray')
            ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
            CircleYin, CircleXin = Yc + (meanRadius-inPix)*np.sin(theta), Xc + (meanRadius-inPix)*np.cos(theta)
            ax.plot(CircleXin, CircleYin, c='gold', ls='--', lw=1)
            CircleYout, CircleXout = Yc + (meanRadius+outPix+1)*np.sin(theta), Xc + (meanRadius+outPix+1)*np.cos(theta)
            ax.plot(CircleXout, CircleYout, c='gold', ls='--', lw=1)
            
            fig_w, axes_w = plt.subplots(1,4, sharey = True)
            ax = axes_w[0]
            ax.imshow(warped, cmap='gray')
            ax.set_title('Raw warp')
            
            ax = axes_w[1]
            ax.imshow(warped, cmap='gray')
            ax.plot(warped_contour_int[:,1]-inPix, warped_contour_int[:,0], c='gold', ls='--', lw=1, zorder = 4)
            ax.plot(warped_contour_int[:,1]+outPix+1, warped_contour_int[:,0], c='gold', ls='--', lw=1, zorder = 4)
            ax.set_title('Segmentation')
            
            ax = axes_w[1]
            ax.plot(max_position_1, yy, c='cyan', ls='', marker ='.', ms=1)
            ax = axes_w[2]
            ax.plot(max_values, yy, 'r-')
            ax.set_title('Max')
            ax = axes_w[3]
            ax.plot(sum_values, yy, 'b-')
            ax.set_title('Sum')
            
            
            fig_c, ax_c = plt.subplots(1,1)
            ax = ax_c
            S = np.arange(0, int(len(acorr_1)/interp_factor), 1/interp_factor) * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN)
            ax.plot(S, acorr_1, label='Cyclic')
            # ax.plot(S[:-1], acorr_0, label='Naive')
            ax.set_xlabel('Curvilinear abs (µm)')
            ax.set_ylabel('Auto-correlation')
            ax.axhline(0, c='k', ls='-', lw=1)
            ax.axhline(0.75, c='b', ls='--', lw=1, zorder=1, label='75%')
            ax.axhline(0.5, c='g', ls='--', lw=1, zorder=1, label='50%')
            
            i_75p = ufun.findFirst(True, acorr_1<0.75)
            # ax.axvline(S[i_75p-1], c='b', ls='--', lw=1, zorder=1)
            ax.text(S[i_75p-1]+0.2, acorr_1[i_75p-1]+0.05, f'{S[i_75p-1]:.2f} µm')
            i_50p = ufun.findFirst(True, acorr_1<0.5)
            # ax.axvline(S[i_50p-1], c='g', ls='--', lw=1, zorder=1)
            ax.text(S[i_50p-1]+0.2, acorr_1[i_50p-1]+0.05, f'{S[i_50p-1]:.2f} µm')
            
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            
            if SAVE:
                ufun.simpleSaveFig(fig_I, f[:-4] + f'_I_{kk:.0f}', dstDir, '.png', 150)
                ufun.simpleSaveFig(fig_w, f[:-4] + f'_w_{kk:.0f}', dstDir, '.png', 150)
                ufun.simpleSaveFig(fig_c, f[:-4] + f'_c_{kk:.0f}', dstDir, '.png', 150)
            
        plt.close('all')
        acorrMatrix.append(acorr_1)    
    
acorrMatrix = np.array(acorrMatrix)

acorr_mean = np.mean(acorrMatrix, axis=0)
radius_mean = np.mean(radiusList)
S = np.arange(0, int(len(acorr_1)/interp_factor), 1/interp_factor) * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN)

fig_c, ax_c = plt.subplots(1,1)
ax = ax_c
ax.plot(S, acorr_mean, label='Mean auto-cor - Ncells = {:.0f}'.format(len(stackList)))
ax.set_xlabel('Curvilinear abs (µm)')
ax.set_ylabel('Auto-correlation')
ax.axhline(0, c='k', ls='-', lw=1)
ax.axhline(0.75, c='b', ls='--', lw=1, zorder=1, label='75%')
ax.axhline(0.5, c='g', ls='--', lw=1, zorder=1, label='50%')

i_75p = ufun.findFirst(True, acorr_mean<0.75)
# ax.axvline(S[i_75p-1], c='b', ls='--', lw=1, zorder=1)
ax.text(S[i_75p-1]+0.2, acorr_mean[i_75p-1]+0.05, f'{S[i_75p-1]:.2f} µm')
i_50p = ufun.findFirst(True, acorr_mean<0.5)
# ax.axvline(S[i_50p-1], c='g', ls='--', lw=1, zorder=1)
ax.text(S[i_50p-1]+0.2, acorr_mean[i_50p-1]+0.05, f'{S[i_50p-1]:.2f} µm')
ax.legend(loc='upper right')

plt.show()


# %% Develop PIA detection and check


srcDir = "D://MicroscopyData//2023-12-06_SpinningDisc_3T3-LifeAct_JV//"
fileList = os.listdir(srcDir)
stackList = []

dstDir = "D://MicroscopyData//2023-12-06_SpinningDisc_3T3-LifeAct_JV//test4"

acorrMatrix_allCells = []
radiusList_allCells = []

for f in fileList:
    if '_ZS_' in f:
        stackList.append(f)

for f in stackList[2:3]:
    filePath = os.path.join(srcDir, f)

    I = skm.io.imread(filePath)
    sumArray = np.sum(I, axis=(1,2))
    z0 = np.argmax(sumArray)
    
    frameIndex = z0 + 16 + np.arange(12)
    # frameIndex = [z0 + 3]
    
    fig1, axes1 = plt.subplots(3, 4, figsize = (12, 9))
    flataxes1 = axes1.flatten().T
    
    fig2, axes2 = plt.subplots(3, 4, figsize = (12, 9))
    flataxes2 = axes2.flatten().T
    
    for k in range(len(frameIndex)): #(16, 25):
        #### Algo Settings
        z = frameIndex[k]
        framesize = int(45*8/2) 
        inPix, outPix = 25, 10
        
        I0 = I[z]
        
        # skm.filters.try_all_threshold(I0)
        th1 = skm.filters.threshold_isodata(I0)
        I0_bin = (I0 > th1)
        I0_bin = ndi.binary_opening(I0_bin, iterations = 2)
        I0_label, num_features = ndi.label(I0_bin)
        df = pd.DataFrame(skm.measure.regionprops_table(I0_label, I0, properties = ['label', 'area']))
        df = df.sort_values(by='area', ascending=False)
        i_label = df.label.values[0]
        I0_rawCell = (I0_label == i_label)
        I0_rawCell = ndi.binary_fill_holes(I0_rawCell)
        [contour_rawCell] = skm.measure.find_contours(I0_rawCell, 0.5)
        
        polygon_cell = shapely.Polygon(contour_rawCell[:, ::-1])
        center = polylabel(polygon_cell, tolerance=1)
        exterior_ring_cell = shapely.get_exterior_ring(polygon_cell)
        R = shapely.distance(center, exterior_ring_cell)
        circle = center.buffer(R)
        
        ax = flataxes1[k]
        ax.imshow(I0_rawCell, cmap='gray')
        ax.text(25, 50, '+ ' + str((z-z0)*0.25) + ' µm', size = 10, color='white')
        # ax.plot(contour_rawCell[:, 1], contour_rawCell[:, 0], 'r--', lw=0.75)
        # plot_polygon(polygon_cell, ax=ax, add_points=False, color='red')
        plot_polygon(circle, ax=ax, add_points=False, color='green')
        plot_points(center, ax=ax, color='green', alpha=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        ax = flataxes2[k]
        ax.imshow(I0, cmap='gray')
        ax.text(25, 50, '+ ' + str((z-z0)*0.25) + ' µm', size = 10, color='white')
        # ax.plot(contour_rawCell[:, 1], contour_rawCell[:, 0], 'r--', lw=0.75)
        # plot_polygon(polygon_cell, ax=ax, add_points=False, color='red')
        plot_polygon(circle, ax=ax, add_points=False, color='green')
        plot_points(center, ax=ax, color='green', alpha=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        
        (Yc, Xc), R = findCellInnerCircle(I0, plot=False)
        
        Xi, Yi = round(Xc), round(Yc)
        x1, y1, x2, y2 = Xi-framesize, Yi-framesize, Xi+framesize+1, Yi+framesize+1
        x1, y1, x2, y2 = getValidROIBounds(I0, x1, y1, x2, y2)
        I0_crop = I0[y1:y2, x1:x2]
        
        (Yc, Xc), R = findCellInnerCircle(I0_crop, plot=True)
        
        
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()
        
    


# %% With Viterbi Smoothing

#### Graphics Settings
PLOT = 1
SAVE = 0
gs.set_mediumText_options_jv()
plt.ioff()

srcDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//"
fileList = os.listdir(srcDir)
stackList = []

dstDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//test2"

acorrMatrix_allCells = []
radiusList_allCells = []

for f in fileList:
    if '_ZS_' in f:
        stackList.append(f)

for f in stackList[:1]:
    filePath = os.path.join(srcDir, f)

    I = skm.io.imread(filePath)
    sumArray = np.sum(I, axis=(1,2))
    z0 = np.argmax(sumArray)
    
    acorrMatrix = []
    radiusList = []
    
    for kk in range(20, 21): #(16, 25):
        
        #### Algo Settings
        framesize = int(35*8/2) 
        inPix, outPix = 25, 10
        blur_parm = 4
        relative_height_virebi = 0.4
        interp_factor_x = 5
        interp_factor_y = 10
        VEW = 7 # Viterbi edge width # Odd number
        
        I0 = I[z0+kk]
        
        #### 1. Detect
        def cellPose_centerFinder(I0,  model_type='TN1', diameter=110, 
                                  flow_threshold = 0.4, cellprob_threshold = 0.4):
            model = models.CellposeModel(gpu=False, model_type=model_type)
            channels = [0, 0]
            [mask], [flow], [style] = model.eval([I0], diameter = diameter, channels=channels,
                                                flow_threshold = flow_threshold, cellprob_threshold = cellprob_threshold, 
                                                do_3D=False, normalize = True) # masks, flows, styles, diams
            Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)
            Yc0, Xc0 = round(Yc), round(Xc)
            
            [contour] = skm.measure.find_contours(mask, 0.5)
            
            return(Yc0, Xc0, contour)
            
            
            
        model = models.CellposeModel(gpu=False, model_type='TN1')
        channels = [0, 0]
        [mask], [flow], [style] = model.eval([I0], diameter = 110, channels=channels,
                                            flow_threshold = 0.4, cellprob_threshold = 0.4, 
                                            do_3D=False, normalize = True) # masks, flows, styles, diams
        Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)
        Yc0, Xc0 = int(Yc), int(Xc)
        
        I0 = I0[Yc0-framesize:Yc0+framesize, Xc0-framesize:Xc0+framesize]
        [mask], [flow], [style] = model.eval([I0], diameter = 110, channels=channels,
                                            flow_threshold = 0.4, cellprob_threshold = 0.4, 
                                            do_3D=False, normalize = True) # masks, flows, styles, diams
        Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)
        
        [contour] = skm.measure.find_contours(mask, 0.5) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
        listDistances = np.sqrt((contour[:,0] - Yc)**2 + (contour[:,1] - Xc)**2)
        meanRadius = np.mean(listDistances)
        radiusList.append(meanRadius)
        
        
        #### 2.1 Warp
        warped = skm.transform.warp_polar(I0, center=(Yc, Xc), radius=meanRadius*1.3, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        warped_mask = skm.transform.warp_polar(mask, center=(Yc, Xc), radius=meanRadius*1.3, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        
        [warped_contour] = skm.measure.find_contours(warped_mask) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
        mask_integer = [warped_contour[i, 0] == int(warped_contour[i, 0]) for i in range(len(warped_contour[:, 0]))]
        warped_contour_int = warped_contour[mask_integer]
        warped_contour_int = warped_contour_int[:,1].astype(int)
        
        w_ny, w_nx = warped.shape
        Angles = np.arange(0, w_ny, 1)
        
        
        #### 2.2. Interp in X
        warped_interp = ufun.resize_2Dinterp(warped, fx=interp_factor_x, fy=1)
        
        warped, w_nx = warped_interp, w_nx * interp_factor_x 
        warped_contour_int = warped_contour_int * interp_factor_x
        inPix, outPix = inPix * interp_factor_x, outPix * interp_factor_x
        
        
        #### 2.3. Max Segmentation
        warped_cortex_region = np.zeros((w_ny, 1 + inPix + outPix), dtype = np.float64)
        
        for i in range(w_ny):
            warped_cortex_region[i, :] = warped[i, warped_contour_int[i]-inPix:warped_contour_int[i]+outPix+1]
            
        max_values = np.max(warped_cortex_region, axis = 1)
        edge_max = np.argmax(warped_cortex_region, axis = 1) + warped_contour_int[:] - inPix
        # edge_max = (edge_max_interp/interp_factor_x)
        
            
        #### 3. Viterbi Smoothing
        
        # Create the TreillisGraph for Viterbi tracking
        # maxWarped = np.max(warped)
        # normWarped = warped * 6 * step/maxWarped
        
        
        AllPeaks = []
        TreillisGraph = []
        warped_blured = skm.filters.gaussian(warped, sigma=(1, blur_parm), mode='wrap')
        
        for a in Angles:
            inBorder = warped_contour_int[a] - inPix
            outBorder = warped_contour_int[a] + outPix
            profile = warped[a, :] - np.min(warped[a, inBorder:outBorder])
            peaks, peaks_props = signal.find_peaks(profile[inBorder:outBorder], 
                                                    # width = 4,
                                                    height = relative_height_virebi*np.max(profile[inBorder:outBorder]))
            AllPeaks.append(peaks + inBorder)
            
            TreillisRow = [{'angle':a, 'pos':p+inBorder, 'val':profile[p+inBorder], 'previous':0, 'accumCost':0} for p in peaks]
            TreillisGraph.append(TreillisRow)
            
        for k in range(len(TreillisGraph)):
            row = TreillisGraph[k]
            if len(row) == 0:
                TreillisGraph[k] = [node.copy() for node in TreillisGraph[k-1]]
        
        # Get a reasonable starting point
        starting_candidates = []
        for R in TreillisGraph:
            if len(R) == 1:
                starting_candidates.append(R[0])
        pos_start = np.median([p['pos'] for p in starting_candidates])
        starting_i = argmedian([p['pos'] for p in starting_candidates])
        starting_peak = starting_candidates[starting_i]
        
        
        # Pretreatment of the TreillisGraph for Viterbi tracking
        TreillisGraph = TreillisGraph[starting_i:] + TreillisGraph[:starting_i] # Set the strating point
        TreillisGraph.append([node.copy() for node in TreillisGraph[0]]) # Make the graph cyclical
        
        # Viterbi tracking
        best_path, nodes_list = ViterbiPathFinder(TreillisGraph)
        
        
        # angles_viterbi = [p['angle'] for p in nodes_list[:-1]]
        edge_viterbi = [p['pos'] for p in nodes_list[:-1]]
        edge_viterbi = edge_viterbi[1-starting_i:] + edge_viterbi[:1-starting_i] # Put it back in order
        viterbi_Npix_values = [np.sum(warped[Angles[i], edge_viterbi[i]-VEW//2:edge_viterbi[i] + (VEW//2 + 1)])/VEW for i in range(len(edge_viterbi))]
        
        
        #### 4. Choose the profile and compute autocorrelation
        
        values = viterbi_Npix_values
        acorr_raw = autocorr_np(values)
        
        bspl = interpolate.make_interp_spline(Angles, values, k = 3)
        
        Angles_interp = np.arange(0, w_ny, 1/interp_factor_y, dtype=float)
        values_interp = bspl(Angles_interp)
        
        acorr_interp = autocorr_np(values_interp)
        
        acorrMatrix.append(acorr_interp)
        
        # To compare
        acorr_max_raw = autocorr_np(max_values)
        bspl = interpolate.make_interp_spline(Angles, max_values, k = 3)
        Angles_interp = np.arange(0, w_ny, 1/interp_factor_y, dtype=float)
        max_values_interp = bspl(Angles_interp)
        acorr_max_interp = autocorr_np(max_values_interp)
        
        #### 4. Attempt at binning+autocor
        
        n_per_bin = 3
        n_bins = 360//n_per_bin
        
        A = np.array(values)
        B = A.reshape((n_bins, n_per_bin))
        values_binned = np.mean(B, axis = 1)
        Angles_binned = np.linspace(0, 360, n_bins, dtype = int, endpoint=False)
        acorr_binned = autocorr_np(values_binned)
        
        #### 5. Plot
        
        if PLOT:
            #### 5.1 Cellpose
            fig_I, axes_I = plt.subplots(1,2, figsize = (12, 6), sharey = True)
            ax = axes_I[0]
            ax.imshow(I0, cmap='gray')
            ax.plot(contour[:,1], contour[:,0], 'r--', lw=1, zorder = 4)
            ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
            theta = np.linspace(0, 2*np.pi, 100)
            CircleY, CircleX = Yc + meanRadius*np.sin(theta), Xc + meanRadius*np.cos(theta)
            # ax.plot(CircleX, CircleY, c='cyan', ls='--', lw=1)
            
            ax = axes_I[1]
            ax.imshow(I0, cmap='gray')
            ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
            CircleYin = Yc + (meanRadius-(inPix/interp_factor_x))*np.sin(theta)
            CircleXin = Xc + (meanRadius-(inPix/interp_factor_x))*np.cos(theta)
            ax.plot(CircleXin, CircleYin, c='gold', ls='--', lw=1)
            CircleYout = Yc + (meanRadius+(outPix/interp_factor_x)+1)*np.sin(theta)
            CircleXout = Xc + (meanRadius+(outPix/interp_factor_x)+1)*np.cos(theta)
            ax.plot(CircleXout, CircleYout, c='gold', ls='--', lw=1)
            
            
            #### 5.2 Warp and Viterbi
            step = 5
            cList_viridis = plt.cm.viridis(np.linspace(0, 1, w_ny//step))
            
            figV, axV = plt.subplots(1, 3, sharey=True, figsize = (9,8))
            ax = axV[0]
            ax.imshow(warped[:, :], cmap='gray', aspect='auto')
            ax.set_title('Raw warp')
            ax.set_ylabel('Angle (deg)')
            ax.set_prop_cycle(plt.cycler("color", cList_viridis))
            ax.plot(warped_contour_int, Angles, ls='', marker='o', c='red', ms = 2)
        
            ax = axV[1]
            ax.set_prop_cycle(plt.cycler("color", cList_viridis))
            ax.set_ylabel('Intensity profile (a.u.)')
            ax.set_xlabel('Radial position (pix)')
            
            ax = axV[2]
            ax.set_prop_cycle(plt.cycler("color", cList_viridis))
            ax.set_ylabel('Intensity profile (a.u.)')
            ax.set_xlabel('Radial position (pix)')
        
            maxWarped = np.max(warped)
            normWarped = warped * 6 * step/maxWarped
        
            for a in Angles[::step]:            
                ax = axV[0]
                ax.plot([inBorder], [a], ls='', marker='o', c='gold', ms = 2)
                ax.plot([outBorder], [a], ls='', marker='o', c='gold', ms = 2)
                
                
                profile = normWarped[a, :] - np.min(normWarped[a, :])
                
                RR = np.arange(len(profile))
                ax = axV[1]
                ax.plot(RR, a - profile, zorder=2)
                ax.plot([inBorder], [a], c='k', ls='', marker='|', ms = 6)
                ax.plot([outBorder], [a], c='k', ls='', marker='|', ms = 6)
                
                ax = axV[2]
                ax.plot(RR, a - profile)
                ax.plot([inBorder], [a], c='k', ls='', marker='|', ms = 6)
                ax.plot([outBorder], [a], c='k', ls='', marker='|', ms = 6)
                # peaks, peaks_props = signal.find_peaks(profile[inBorder:outBorder], 
                #                                         height = 0.7*np.max(profile[inBorder:outBorder])) #, width = 2)
                
            # for P, a in zip(AllPeaks, Angles):
                P = AllPeaks[a]
                P = np.array(P)
                A = a * np.ones(len(P))
                ax.plot(P, A, ls='', c='orange', marker='o', ms = 4, mec = 'k') #  t - profile[inPix + peaks[k]]
                
            # ax.plot([starting_peak['pos']], [starting_peak['angle']], ls='', c='darkred', marker='o', ms = 5, mec = 'gold', zorder=6) #  t - profile[inPix + peaks[k]]
            # for p in starting_candidates:
            #     ax.plot([p['pos']], [p['angle']], ls='', c='darkred', marker='o', ms = 5, mec = 'k', zorder = 5) #  t - profile[inPix + peaks[k]]
            
                node = nodes_list[a]
                ax = axV[2]
                ax.plot([node['pos']], [node['angle']], ls='', c='cyan', marker='o', ms = 4, mec = 'k') #  t - profile[inPix + peaks[k]]
            
            for node in nodes_list[::]:
                ax = axV[0]
                ax.plot([node['pos']], [node['angle']], ls='', c='cyan', marker='.', ms = 4) #  t - profile[inPix + peaks[k]]
            
            
            #### 5.3 Warp and Profiles
            fig_w, axes_w = plt.subplots(1,4, sharey = True, figsize = (12, 8))
            ax = axes_w[0]
            ax.imshow(warped_blured, cmap='gray', aspect='auto')
            ax.set_title('Raw warp')
            ax.set_ylabel('Angle (deg)')
            
            ax = axes_w[1]
            ax.imshow(warped, cmap='gray', aspect='auto')
            ax.plot(warped_contour_int[:]-inPix, Angles, c='gold', ls='--', lw=1, zorder = 4)
            ax.plot(warped_contour_int[:]+outPix+1, Angles, c='gold', ls='--', lw=1, zorder = 4)
            ax.set_title('Max Segmentation')
            ax.plot(edge_max, Angles, c='cyan', ls='', marker ='.', ms=2)
            
            ax = axes_w[2]
            ax.imshow(warped, cmap='gray', aspect='auto')
            ax.plot(warped_contour_int[:]-inPix, Angles, c='gold', ls='--', lw=1, zorder = 4)
            ax.plot(warped_contour_int[:]+outPix+1, Angles, c='gold', ls='--', lw=1, zorder = 4)
            ax.set_title('Viterbi Segmentation')
            ax.plot(edge_viterbi, Angles, c='cyan', ls='', marker ='.', ms=2)
            
            ax = axes_w[3]
            ax.set_title('Profile Values')
            ax.plot(max_values, Angles, 'r-', label = 'Max profile')
            ax.plot(viterbi_Npix_values, Angles, 'b-', label = 'Viterbi profile')
            ax.plot(values_binned, Angles_binned, 'g-', label = 'Viterbi-binned profile', ms=2)
            ax.legend()
            
            def deg2um(a):
                return(a * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN))
            def um2deg(x):
                return(x * (360/2*np.pi) / (meanRadius/SCALE_100X_ZEN))
            secax = ax.secondary_yaxis('right', functions=(deg2um, um2deg))
            secax.set_ylabel('curv abs [µm]')  
            
            #### 5.X Prediction
            def compute_pred(A, w_bin=3, d_bin=0):
                """
                w_bin is the width of the bin
                d_bin is the distance between the bin edges
                """
                pred = np.zeros_like(A)
                pred_percent = np.zeros_like(A)
                hw_bin = w_bin//2
                for i in range(360):
                    bin_i    = A[looping_index(A, i-hw_bin, i+hw_bin+1)]
                    bin_prev = A[looping_index(A, i-d_bin-w_bin-hw_bin,i-d_bin-hw_bin)]
                    bin_next = A[looping_index(A, i+d_bin+hw_bin+1,i+d_bin+w_bin+hw_bin+1)]
                    val      = np.mean(bin_i)
                    val_prev = np.mean(bin_prev)
                    val_next = np.mean(bin_next)
                    pred_val = (val_prev + val_next)/2
                    pred_val_percent = 100 * np.abs(pred_val - val) / val
                    pred[i] = pred_val
                    pred_percent[i] = pred_val_percent
                return(pred, pred_percent)
            
            A = np.array(viterbi_Npix_values)
            # A = A - np.mean(A)
            A = A
            
            fig_p, axes_p = plt.subplots(2,1, figsize = (12, 8))
            ax = axes_p[0]
            ax.plot(Angles, A, label = 'Truth')
            ax.set_ylabel('Pred value')
            ax.set_xlabel('Angle (deg)')
            ax = axes_p[1]
            ax.set_ylabel('Pred percent')
            ax.set_xlabel('Angle (deg)')
            
            P, Pp = compute_pred(A, w_bin=3, d_bin=0)
            ax = axes_p[0]
            ax.plot(Angles, P, label='Pred w=3 d=0')
            ax = axes_p[1]
            ax.plot(Angles, Pp, label='Pred w=3 d=0')
            print(np.mean(Pp))
            
            P, Pp = compute_pred(A, w_bin=5, d_bin=0)
            ax = axes_p[0]
            ax.plot(Angles, P, label='Pred w=5 d=0')
            ax = axes_p[1]
            ax.plot(Angles, Pp, label='Pred w=5 d=0')
            print(np.mean(Pp))
            
            P, Pp = compute_pred(A, w_bin=3, d_bin=1)
            ax = axes_p[0]
            ax.plot(Angles, P, label='Pred w=3 d=1')
            ax = axes_p[1]
            ax.plot(Angles, Pp, label='Pred w=3 d=1')
            print(np.mean(Pp))
            
            CV = 100*np.std(A)/np.mean(A)
            fig_p.suptitle(f'Cv = {CV:.0f}%')
            
            axes_p[0].legend()
            axes_p[1].legend()
            
            # ax = axes_p[1]
            # ax.imshow(warped, cmap='gray', aspect='auto')
            # ax.plot(warped_contour_int[:]-inPix, Angles, c='gold', ls='--', lw=1, zorder = 4)
            # ax.plot(warped_contour_int[:]+outPix+1, Angles, c='gold', ls='--', lw=1, zorder = 4)
            # ax.set_title('Max Segmentation')
            # ax.plot(edge_max, Angles, c='cyan', ls='', marker ='.', ms=2)
        
            
            
            
            #### 5.4 Autocorr
            fig_c, ax_c = plt.subplots(1,1)
            ax = ax_c
            S = Angles * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN)
            S_interp = Angles_interp * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN)
            S_binned = Angles_binned * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN)
            
            # ax.plot(S, acorr_raw, label='raw')
            ax.plot(S_interp, acorr_max_interp, label='max-interp')
            ax.plot(S_interp, acorr_interp, label='viterbi-interp')
            ax.plot(S_binned, acorr_binned, label='viterbi-binned')
            ax.set_xlabel('Curvilinear abs (µm)')
            ax.set_ylabel('Auto-correlation')
            ax.axhline(0, c='k', ls='-', lw=1)
            ax.axhline(0.75, c='b', ls='--', lw=1, zorder=1, label='75%')
            ax.axhline(0.5, c='g', ls='--', lw=1, zorder=1, label='50%')
            
            i_75p = ufun.findFirst(True, acorr_interp<0.75)
            # ax.axvline(S[i_75p-1], c='b', ls='--', lw=1, zorder=1)
            ax.text(S_interp[i_75p-1]+0.2, acorr_interp[i_75p-1]+0.05, f'{S_interp[i_75p-1]:.2f} µm')
            i_50p = ufun.findFirst(True, acorr_interp<0.5)
            # ax.axvline(S[i_50p-1], c='g', ls='--', lw=1, zorder=1)
            ax.text(S_interp[i_50p-1]+0.2, acorr_interp[i_50p-1]+0.05, f'{S_interp[i_50p-1]:.2f} µm')
            ax.legend(loc='upper right')
            
            if SAVE:
                ufun.simpleSaveFig(fig_I, f[:-4] + f'_I_{kk:.0f}', dstDir, '.png', 150)
                ufun.simpleSaveFig(fig_w, f[:-4] + f'_w_{kk:.0f}', dstDir, '.png', 150)
                ufun.simpleSaveFig(figV,  f[:-4] + f'_V_{kk:.0f}', dstDir, '.png', 150)
                ufun.simpleSaveFig(fig_c, f[:-4] + f'_c_{kk:.0f}', dstDir, '.png', 150)
                
            # plt.close('all')
    
    
    
    # Average Plot
    S_interp = Angles_interp * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN)
    acorrArray = np.array(acorrMatrix)
    
    acorr_mean = np.mean(acorrArray, axis=0)
    radius_mean = np.mean(radiusList)
    
    fig_ac, ax_ac = plt.subplots(1,1)
    ax = ax_ac
    ax.plot(S_interp, acorr_mean, label='Mean auto-cor - Ncells = {:.0f}'.format(len(stackList)))
    ax.set_xlabel('Curvilinear abs (µm)')
    ax.set_ylabel('Auto-correlation')
    ax.axhline(0, c='k', ls='-', lw=1)
    ax.axhline(0.75, c='b', ls='--', lw=1, zorder=1, label='75%')
    ax.axhline(0.5, c='g', ls='--', lw=1, zorder=1, label='50%')
    
    i_75p = ufun.findFirst(True, acorr_mean<0.75)
    # ax.axvline(S[i_75p-1], c='b', ls='--', lw=1, zorder=1)
    ax.text(S_interp[i_75p-1]+0.2, acorr_mean[i_75p-1]+0.05, f'{S_interp[i_75p-1]:.2f} µm')
    i_50p = ufun.findFirst(True, acorr_mean<0.5)
    # ax.axvline(S[i_50p-1], c='g', ls='--', lw=1, zorder=1)
    ax.text(S_interp[i_50p-1]+0.2, acorr_mean[i_50p-1]+0.05, f'{S_interp[i_50p-1]:.2f} µm')
    ax.legend(loc='upper right')
    
    if SAVE:
        ufun.simpleSaveFig(fig_ac,  f[:-4] + '_average_acor', dstDir, '.png', 150)
        
        
    acorrMatrix_allCells = acorrMatrix_allCells + acorrMatrix
    radiusList_allCells = radiusList_allCells + radiusList
    

acorrArray_allCells = np.array(acorrMatrix_allCells)

acorr_mean = np.mean(acorrArray_allCells, axis=0)
radius_mean = np.mean(radiusList_allCells)

fig_ac, ax_ac = plt.subplots(1,1)
ax = ax_ac
ax.plot(S_interp, acorr_mean, label='Mean auto-cor - Ncells = {:.0f}'.format(len(stackList)))
ax.set_xlabel('Curvilinear abs (µm)')
ax.set_ylabel('Auto-correlation')
ax.axhline(0, c='k', ls='-', lw=1)
ax.axhline(0.75, c='b', ls='--', lw=1, zorder=1, label='75%')
ax.axhline(0.5, c='g', ls='--', lw=1, zorder=1, label='50%')

i_75p = ufun.findFirst(True, acorr_mean<0.75)
# ax.axvline(S[i_75p-1], c='b', ls='--', lw=1, zorder=1)
ax.text(S_interp[i_75p-1]+0.2, acorr_mean[i_75p-1]+0.05, f'{S_interp[i_75p-1]:.2f} µm')
i_50p = ufun.findFirst(True, acorr_mean<0.5)
# ax.axvline(S[i_50p-1], c='g', ls='--', lw=1, zorder=1)
ax.text(S_interp[i_50p-1]+0.2, acorr_mean[i_50p-1]+0.05, f'{S_interp[i_50p-1]:.2f} µm')
ax.legend(loc='upper right')

if SAVE:
    ufun.simpleSaveFig(fig_ac,  '00_allCells_average_acor', dstDir, '.png', 150)


plt.tight_layout()
plt.show()



# %% Test Vertical With Viterbi Smoothing

#### Graphics Settings
PLOT = 1
SAVE = 1
gs.set_smallText_options_jv()
plt.ioff()

srcDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//"
fileList = os.listdir(srcDir)
stackList = []

dstDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//testZ"

acorrMatrix_allCells = []
radiusList_allCells = []

for f in fileList:
    if '_ZS_' in f:
        stackList.append(f)

for f in stackList[:]:
    filePath = os.path.join(srcDir, f)

    I = skm.io.imread(filePath)
    sumArray = np.sum(I, axis=(1,2))
    I_nz = I.shape[0]
    z0 = np.argmax(sumArray)
    
    acorrMatrix = []
    radiusList = []
    
    verticalPos = []
    verticalValues = []
    verticalStds = []
    
    for kk in range(10, I_nz-20):
        #### Algo Settings
        framesize = int(35*8/2) 
        inPix, outPix = 25, 10
        blur_parm = 4
        relative_height_virebi = 0.4
        interp_factor_x = 5
        interp_factor_y = 10
        VEW = 7 # Viterbi edge width # Odd number
        
        try:        
            I0 = I[z0+kk]
            
            #### 1. Detect
            model = models.CellposeModel(gpu=False, model_type='TN1')
            channels = [0, 0]
            [mask], [flow], [style] = model.eval([I0], diameter = 110, channels=channels,
                                                flow_threshold = 0.4, cellprob_threshold = 0.4, 
                                                do_3D=False, normalize = True) # masks, flows, styles, diams
            Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)
            Yc0, Xc0 = int(Yc), int(Xc)
            
            I0 = I0[Yc0-framesize:Yc0+framesize, Xc0-framesize:Xc0+framesize]
            [mask], [flow], [style] = model.eval([I0], diameter = 110, channels=channels,
                                                flow_threshold = 0.4, cellprob_threshold = 0.4, 
                                                do_3D=False, normalize = True) # masks, flows, styles, diams
            Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)
            
            [contour] = skm.measure.find_contours(mask, 0.5) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
            listDistances = np.sqrt((contour[:,0] - Yc)**2 + (contour[:,1] - Xc)**2)
            meanRadius = np.mean(listDistances)
            radiusList.append(meanRadius)
            
            
            #### 2.1 Warp
            warped = skm.transform.warp_polar(I0, center=(Yc, Xc), radius=meanRadius*1.3, 
                                              output_shape=None, scaling='linear', channel_axis=None)
            warped_mask = skm.transform.warp_polar(mask, center=(Yc, Xc), radius=meanRadius*1.3, 
                                              output_shape=None, scaling='linear', channel_axis=None)
            
            [warped_contour] = skm.measure.find_contours(warped_mask) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
            mask_integer = [warped_contour[i, 0] == int(warped_contour[i, 0]) for i in range(len(warped_contour[:, 0]))]
            warped_contour_int = warped_contour[mask_integer]
            warped_contour_int = warped_contour_int[:,1].astype(int)
            
            w_ny, w_nx = warped.shape
            Angles = np.arange(0, w_ny, 1)
            
            
            #### 2.2. Interp in X
            warped_interp = ufun.resize_2Dinterp(warped, fx=interp_factor_x, fy=1)
            
            warped, w_nx = warped_interp, w_nx * interp_factor_x 
            warped_contour_int = warped_contour_int * interp_factor_x
            inPix, outPix = inPix * interp_factor_x, outPix * interp_factor_x
            
            
            #### 2.3. Max Segmentation
            warped_cortex_region = np.zeros((w_ny, 1 + inPix + outPix), dtype = np.float64)
            
            for i in range(w_ny):
                warped_cortex_region[i, :] = warped[i, warped_contour_int[i]-inPix:warped_contour_int[i]+outPix+1]
                
            max_values = np.max(warped_cortex_region, axis = 1)
            edge_max = np.argmax(warped_cortex_region, axis = 1) + warped_contour_int[:] - inPix
            # edge_max = (edge_max_interp/interp_factor_x)
            
                
            #### 3. Viterbi Smoothing
            
            # Create the TreillisGraph for Viterbi tracking
            # maxWarped = np.max(warped)
            # normWarped = warped * 6 * step/maxWarped
            
            
            AllPeaks = []
            TreillisGraph = []
            warped_blured = skm.filters.gaussian(warped, sigma=(1, blur_parm), mode='wrap')
            
            for a in Angles:
                inBorder = warped_contour_int[a] - inPix
                outBorder = warped_contour_int[a] + outPix
                profile = warped[a, :] - np.min(warped[a, inBorder:outBorder])
                peaks, peaks_props = signal.find_peaks(profile[inBorder:outBorder], 
                                                        # width = 4,
                                                        height = relative_height_virebi*np.max(profile[inBorder:outBorder]))
                AllPeaks.append(peaks + inBorder)
                
                TreillisRow = [{'angle':a, 'pos':p+inBorder, 'val':profile[p+inBorder], 'previous':0, 'accumCost':0} for p in peaks]
                TreillisGraph.append(TreillisRow)
                
            for k in range(len(TreillisGraph)):
                row = TreillisGraph[k]
                if len(row) == 0:
                    TreillisGraph[k] = [node.copy() for node in TreillisGraph[k-1]]
            
            # Get a reasonable starting point
            starting_candidates = []
            for R in TreillisGraph:
                if len(R) == 1:
                    starting_candidates.append(R[0])
            pos_start = np.median([p['pos'] for p in starting_candidates])
            starting_i = argmedian([p['pos'] for p in starting_candidates])
            starting_peak = starting_candidates[starting_i]
            
            
            # Pretreatment of the TreillisGraph for Viterbi tracking
            TreillisGraph = TreillisGraph[starting_i:] + TreillisGraph[:starting_i] # Set the strating point
            TreillisGraph.append([node.copy() for node in TreillisGraph[0]]) # Make the graph cyclical
            
            # Viterbi tracking
            best_path, nodes_list = ViterbiPathFinder(TreillisGraph)
            
            
            # angles_viterbi = [p['angle'] for p in nodes_list[:-1]]
            edge_viterbi = [p['pos'] for p in nodes_list[:-1]]
            edge_viterbi = edge_viterbi[1-starting_i:] + edge_viterbi[:1-starting_i] # Put it back in order
            viterbi_Npix_values = [np.sum(warped[Angles[i], edge_viterbi[i]-VEW//2:edge_viterbi[i] + (VEW//2 + 1)])/VEW for i in range(len(edge_viterbi))]
            
            
            #### 4. Choose the profile and compute autocorrelation
            
            values = viterbi_Npix_values
            acorr_raw = autocorr_np(values)
            
            bspl = interpolate.make_interp_spline(Angles, values, k = 3)
            
            Angles_interp = np.arange(0, w_ny, 1/interp_factor_y, dtype=float)
            values_interp = bspl(Angles_interp)
            
            acorr_interp = autocorr_np(values_interp)
            
            acorrMatrix.append(acorr_interp)
            
            # To compare
            acorr_max_raw = autocorr_np(max_values)
            bspl = interpolate.make_interp_spline(Angles, max_values, k = 3)
            Angles_interp = np.arange(0, w_ny, 1/interp_factor_y, dtype=float)
            max_values_interp = bspl(Angles_interp)
            acorr_max_interp = autocorr_np(max_values_interp)
            
            #### 4.2 Attempt at binning+autocor
            
            n_per_bin = 3
            n_bins = 360//n_per_bin
            
            A = np.array(values)
            B = A.reshape((n_bins, n_per_bin))
            values_binned = np.mean(B, axis = 1)
            Angles_binned = np.linspace(0, 360, n_bins, dtype = int, endpoint=False)
            acorr_binned = autocorr_np(values_binned)
            
            #### 4.3 Save the profile
            verticalPos.append(0.25*kk)
            verticalValues.append(values_binned)
            verticalAngles = Angles_binned
            verticalStds.append(np.std(values_binned))
            
            #### 5. Plot
            
            if PLOT:
                #### 5.1 Cellpose
                fig_I, axes_I = plt.subplots(1,2, figsize = (12, 6), sharey = True)
                ax = axes_I[0]
                ax.imshow(I0, cmap='gray')
                ax.plot(contour[:,1], contour[:,0], 'r--', lw=1, zorder = 4)
                ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
                theta = np.linspace(0, 2*np.pi, 100)
                CircleY, CircleX = Yc + meanRadius*np.sin(theta), Xc + meanRadius*np.cos(theta)
                # ax.plot(CircleX, CircleY, c='cyan', ls='--', lw=1)
                
                ax = axes_I[1]
                ax.imshow(I0, cmap='gray')
                ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
                CircleYin = Yc + (meanRadius-(inPix/interp_factor_x))*np.sin(theta)
                CircleXin = Xc + (meanRadius-(inPix/interp_factor_x))*np.cos(theta)
                ax.plot(CircleXin, CircleYin, c='gold', ls='--', lw=1)
                CircleYout = Yc + (meanRadius+(outPix/interp_factor_x)+1)*np.sin(theta)
                CircleXout = Xc + (meanRadius+(outPix/interp_factor_x)+1)*np.cos(theta)
                ax.plot(CircleXout, CircleYout, c='gold', ls='--', lw=1)
                
                
                #### 5.2 Warp and Viterbi
                step = 5
                cList_viridis = plt.cm.viridis(np.linspace(0, 1, w_ny//step))
                
                fig_V, ax_V = plt.subplots(1, 3, sharey=True, figsize = (9,8))
                ax = ax_V[0]
                ax.imshow(warped_blured[:, :], cmap='gray', aspect='auto')
                ax.set_title('Raw warp')
                ax.set_ylabel('Angle (deg)')
                ax.set_prop_cycle(plt.cycler("color", cList_viridis))
                ax.plot(warped_contour_int, Angles, ls='', marker='o', c='red', ms = 2)
            
                ax = ax_V[1]
                ax.set_prop_cycle(plt.cycler("color", cList_viridis))
                ax.set_ylabel('Intensity profile (a.u.)')
                ax.set_xlabel('Radial position (pix)')
                
                ax = ax_V[2]
                ax.set_prop_cycle(plt.cycler("color", cList_viridis))
                ax.set_ylabel('Intensity profile (a.u.)')
                ax.set_xlabel('Radial position (pix)')
            
                maxWarped = np.max(warped)
                normWarped = warped * 6 * step/maxWarped
            
                for a in Angles[::step]:            
                    ax = ax_V[0]
                    ax.plot([inBorder], [a], ls='', marker='o', c='gold', ms = 2)
                    ax.plot([outBorder], [a], ls='', marker='o', c='gold', ms = 2)
                    
                    
                    profile = normWarped[a, :] - np.min(normWarped[a, :])
                    
                    RR = np.arange(len(profile))
                    ax = ax_V[1]
                    ax.plot(RR, a - profile, zorder=2)
                    ax.plot([inBorder], [a], c='k', ls='', marker='|', ms = 6)
                    ax.plot([outBorder], [a], c='k', ls='', marker='|', ms = 6)
                    
                    ax = ax_V[2]
                    ax.plot(RR, a - profile)
                    ax.plot([inBorder], [a], c='k', ls='', marker='|', ms = 6)
                    ax.plot([outBorder], [a], c='k', ls='', marker='|', ms = 6)
                    # peaks, peaks_props = signal.find_peaks(profile[inBorder:outBorder], 
                    #                                         height = 0.7*np.max(profile[inBorder:outBorder])) #, width = 2)
                    
                # for P, a in zip(AllPeaks, Angles):
                    P = AllPeaks[a]
                    P = np.array(P)
                    A = a * np.ones(len(P))
                    ax.plot(P, A, ls='', c='orange', marker='o', ms = 4, mec = 'k') #  t - profile[inPix + peaks[k]]
                    
                # ax.plot([starting_peak['pos']], [starting_peak['angle']], ls='', c='darkred', marker='o', ms = 5, mec = 'gold', zorder=6) #  t - profile[inPix + peaks[k]]
                # for p in starting_candidates:
                #     ax.plot([p['pos']], [p['angle']], ls='', c='darkred', marker='o', ms = 5, mec = 'k', zorder = 5) #  t - profile[inPix + peaks[k]]
                
                    node = nodes_list[a]
                    ax = ax_V[2]
                    ax.plot([node['pos']], [node['angle']], ls='', c='cyan', marker='o', ms = 4, mec = 'k') #  t - profile[inPix + peaks[k]]
                
                for node in nodes_list[::]:
                    ax = ax_V[0]
                    ax.plot([node['pos']], [node['angle']], ls='', c='cyan', marker='.', ms = 4) #  t - profile[inPix + peaks[k]]
                
                
                #### 5.3 Warp and Profiles
                fig_w, axes_w = plt.subplots(1,4, sharey = True, figsize = (12, 8))
                ax = axes_w[0]
                ax.imshow(warped, cmap='gray', aspect='auto')
                ax.set_title('Raw warp')
                ax.set_ylabel('Angle (deg)')
                
                ax = axes_w[1]
                ax.imshow(warped, cmap='gray', aspect='auto')
                ax.plot(warped_contour_int[:]-inPix, Angles, c='gold', ls='--', lw=1, zorder = 4)
                ax.plot(warped_contour_int[:]+outPix+1, Angles, c='gold', ls='--', lw=1, zorder = 4)
                ax.set_title('Max Segmentation')
                ax.plot(edge_max, Angles, c='cyan', ls='', marker ='.', ms=2)
                
                ax = axes_w[2]
                ax.imshow(warped, cmap='gray', aspect='auto')
                ax.plot(warped_contour_int[:]-inPix, Angles, c='gold', ls='--', lw=1, zorder = 4)
                ax.plot(warped_contour_int[:]+outPix+1, Angles, c='gold', ls='--', lw=1, zorder = 4)
                ax.set_title('Viterbi Segmentation')
                ax.plot(edge_viterbi, Angles, c='cyan', ls='', marker ='.', ms=2)
                
                ax = axes_w[3]
                ax.set_title('Profile Values')
                ax.plot(max_values, Angles, 'r-', label = 'Max profile')
                ax.plot(viterbi_Npix_values, Angles, 'b-', label = 'Viterbi profile')
                ax.plot(values_binned, Angles_binned, 'g-', label = 'Viterbi-binned profile', ms=2)
                ax.legend()
            
                
                def deg2um(a):
                    return(a * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN))
            
                def um2deg(x):
                    return(x * (360/2*np.pi) / (meanRadius/SCALE_100X_ZEN))
            
                secax = ax.secondary_yaxis('right', functions=(deg2um, um2deg))
                secax.set_ylabel('curv abs [µm]')  
                
                
                
                
                
                
                #### 5.4 Autocorr
                fig_c, ax_c = plt.subplots(1,1)
                ax = ax_c
                S = Angles * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN)
                S_interp = Angles_interp * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN)
                S_binned = Angles_binned * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN)
                
                # ax.plot(S, acorr_raw, label='raw')
                ax.plot(S_interp, acorr_max_interp, label='max-interp')
                ax.plot(S_interp, acorr_interp, label='viterbi-interp')
                ax.plot(S_binned, acorr_binned, label='viterbi-binned')
                ax.set_xlabel('Curvilinear abs (µm)')
                ax.set_ylabel('Auto-correlation')
                ax.axhline(0, c='k', ls='-', lw=1)
                ax.axhline(0.75, c='b', ls='--', lw=1, zorder=1, label='75%')
                ax.axhline(0.5, c='g', ls='--', lw=1, zorder=1, label='50%')
                
                i_75p = ufun.findFirst(True, acorr_interp<0.75)
                # ax.axvline(S[i_75p-1], c='b', ls='--', lw=1, zorder=1)
                ax.text(S_interp[i_75p-1]+0.2, acorr_interp[i_75p-1]+0.05, f'{S_interp[i_75p-1]:.2f} µm')
                i_50p = ufun.findFirst(True, acorr_interp<0.5)
                # ax.axvline(S[i_50p-1], c='g', ls='--', lw=1, zorder=1)
                ax.text(S_interp[i_50p-1]+0.2, acorr_interp[i_50p-1]+0.05, f'{S_interp[i_50p-1]:.2f} µm')
                ax.legend(loc='upper right')
                
                if SAVE:
                    ufun.simpleSaveFig(fig_I, f[:-4] + f'_I_{kk:.0f}', dstDir + '//' + f, '.png', 150)
                    ufun.simpleSaveFig(fig_w, f[:-4] + f'_w_{kk:.0f}', dstDir + '//' + f, '.png', 150)
                    ufun.simpleSaveFig(fig_V, f[:-4] + f'_V_{kk:.0f}', dstDir + '//' + f, '.png', 150)
                    ufun.simpleSaveFig(fig_c, f[:-4] + f'_c_{kk:.0f}', dstDir + '//' + f, '.png', 150)
                    
                plt.close('all')
                
                
            
        except:
            pass
    
    
    
    # Average Plot
    S_interp = Angles_interp * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN)
    acorrArray = np.array(acorrMatrix)
    
    acorr_mean = np.mean(acorrArray, axis=0)
    radius_mean = np.mean(radiusList)
    
    fig_ac, ax_ac = plt.subplots(1,1)
    ax = ax_ac
    ax.plot(S_interp, acorr_mean, label='Mean auto-cor - Ncells = {:.0f}'.format(len(stackList)))
    ax.set_xlabel('Curvilinear abs (µm)')
    ax.set_ylabel('Auto-correlation')
    ax.axhline(0, c='k', ls='-', lw=1)
    ax.axhline(0.75, c='b', ls='--', lw=1, zorder=1, label='75%')
    ax.axhline(0.5, c='g', ls='--', lw=1, zorder=1, label='50%')
    
    i_75p = ufun.findFirst(True, acorr_mean<0.75)
    # ax.axvline(S[i_75p-1], c='b', ls='--', lw=1, zorder=1)
    ax.text(S_interp[i_75p-1]+0.2, acorr_mean[i_75p-1]+0.05, f'{S_interp[i_75p-1]:.2f} µm')
    i_50p = ufun.findFirst(True, acorr_mean<0.5)
    # ax.axvline(S[i_50p-1], c='g', ls='--', lw=1, zorder=1)
    ax.text(S_interp[i_50p-1]+0.2, acorr_mean[i_50p-1]+0.05, f'{S_interp[i_50p-1]:.2f} µm')
    ax.legend(loc='upper right')
    
    if SAVE:
        ufun.simpleSaveFig(fig_ac,  f[:-4] + '_average_acor', dstDir + '//' + f, '.png', 150)
        
    #### Vertical Plots
    verticalArray = np.array(verticalValues)
    nZ, nA = verticalArray.shape    
    
    fig_ZS1, ax_ZS1 = plt.subplots(1,1, figsize = (10, 5))
    cList_viridis = plt.cm.viridis(np.linspace(0, 1, verticalArray.shape[0]))
    ax = ax_ZS1
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_prop_cycle(plt.cycler("color", cList_viridis))
    for vals in verticalValues:
        ax.plot(verticalAngles, vals)
    
    step = 5
    Ngraphs = nA//step
    Ncols = 4
    Nrows = Ngraphs//Ncols
    mini_va, Maxi_va = np.min(verticalArray[:, ::step]), np.max(verticalArray[:, ::step])
    fig_ZS2, ax_ZS2 = plt.subplots(Nrows, Ncols, figsize = (3*Ncols, 1.5*Nrows))
    cList_viridis = plt.cm.viridis(np.linspace(0, 1, verticalArray.shape[0]))
    for i in range(Ngraphs):
        angle = i * step
        ax = (ax_ZS2.T).flatten()[i]
        ax.set_prop_cycle(plt.cycler("color", cList_viridis))
        for j in range(nZ):
            z = verticalPos[j]
            val = verticalArray[j, angle]
            ax.plot(z, val, ls='', marker='o', ms = 4)
            ax.set_ylim([mini_va, Maxi_va])
            ax.set_ylabel(f'{angle}°')
            
    for c in range(Ncols):
        for ax in ax_ZS2[:-1,c]:
            ax.set_xticklabels([])
        ax = ax_ZS2[-1,c]
        ax.set_xlabel('Z (µm)')
        
    fig_ZS3, ax_ZS3 = plt.subplots(1, 1, figsize = (5, 2.5))
    ax = ax_ZS3
    cList_viridis = plt.cm.viridis(np.linspace(0, 1, len(verticalStds)))
    ax.set_prop_cycle(plt.cycler("color", cList_viridis))
    for j in range(nZ):
        z = verticalPos[j]
        std = verticalStds[j]
        ax.plot(z, std, ls='', marker='o', ms = 4)
    ax.set_ylabel('Std')
    ax.set_xlabel('Z (µm)')
        
    if SAVE:
        ufun.simpleSaveFig(fig_ZS1, '00_' + f[:-4] + '_AllProfiles', dstDir + '//' + f, '.png', 150)
        ufun.simpleSaveFig(fig_ZS2, '00_' + f[:-4] + '_ZprofilesByAngles', dstDir + '//' + f, '.png', 150)
        ufun.simpleSaveFig(fig_ZS3, '00_' + f[:-4] + '_Zstds', dstDir + '//' + f, '.png', 150)
    
    
    # if SAVE:
    #     ufun.simpleSaveFig(fig_ac,  f[:-4] + '_average_acor', dstDir, '.png', 150)
        
    # acorrMatrix_allCells = acorrMatrix_allCells + acorrMatrix
    # radiusList_allCells = radiusList_allCells + radiusList
    

# acorrArray_allCells = np.array(acorrMatrix_allCells)

# acorr_mean = np.mean(acorrArray_allCells, axis=0)
# radius_mean = np.mean(radiusList_allCells)

# fig_ac, ax_ac = plt.subplots(1,1)
# ax = ax_ac
# ax.plot(S_interp, acorr_mean, label='Mean auto-cor - Ncells = {:.0f}'.format(len(stackList)))
# ax.set_xlabel('Curvilinear abs (µm)')
# ax.set_ylabel('Auto-correlation')
# ax.axhline(0, c='k', ls='-', lw=1)
# ax.axhline(0.75, c='b', ls='--', lw=1, zorder=1, label='75%')
# ax.axhline(0.5, c='g', ls='--', lw=1, zorder=1, label='50%')

# i_75p = ufun.findFirst(True, acorr_mean<0.75)
# # ax.axvline(S[i_75p-1], c='b', ls='--', lw=1, zorder=1)
# ax.text(S_interp[i_75p-1]+0.2, acorr_mean[i_75p-1]+0.05, f'{S_interp[i_75p-1]:.2f} µm')
# i_50p = ufun.findFirst(True, acorr_mean<0.5)
# # ax.axvline(S[i_50p-1], c='g', ls='--', lw=1, zorder=1)
# ax.text(S_interp[i_50p-1]+0.2, acorr_mean[i_50p-1]+0.05, f'{S_interp[i_50p-1]:.2f} µm')
# ax.legend(loc='upper right')

# if SAVE:
#     ufun.simpleSaveFig(fig_ac,  '00_allCells_average_acor', dstDir, '.png', 150)


plt.tight_layout()
plt.show()



# %% Warp and save


srcDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//"
fileList = os.listdir(srcDir)
stackList = []

dstDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//ZStacks_Warp//"


for f in fileList:
    if '_ZS_' in f:
        stackList.append(f)

for f in stackList[:]:
    filePath = os.path.join(srcDir, f)

    I = skm.io.imread(filePath)
    sumArray = np.sum(I, axis=(1,2))
    z0 = np.argmax(sumArray)
    
    
    for kk in range(16, 25):
                                        
        I0 = I[z0+kk]
        
        #### Settings
        framesize = int(35*8/2) 
        inPix, outPix = 25, 10
        
        #### 1. Detect
        model = models.CellposeModel(gpu=False, model_type='TN1')
        channels = [0, 0]
        [mask], [flow], [style] = model.eval([I0], diameter = 110, channels=channels,
                                            flow_threshold = 0.4, cellprob_threshold = 0.4, 
                                            do_3D=False, normalize = True) # masks, flows, styles, diams
        Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)
        Yc0, Xc0 = int(Yc), int(Xc)
        
        I0 = I0[Yc0-framesize:Yc0+framesize, Xc0-framesize:Xc0+framesize]
        [mask], [flow], [style] = model.eval([I0], diameter = 110, channels=channels,
                                            flow_threshold = 0.4, cellprob_threshold = 0.4, 
                                            do_3D=False, normalize = True) # masks, flows, styles, diams
        Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)
        
        [contour] = skm.measure.find_contours(mask, 0.5) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
        listDistances = np.sqrt((contour[:,0] - Yc)**2 + (contour[:,1] - Xc)**2)
        meanRadius = np.mean(listDistances)
        
        
        #### 2.1 Warp
        warped = skm.transform.warp_polar(I0, center=(Yc, Xc), radius=meanRadius*1.3, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        
        saveName = f"{f[:-4]}_Z{kk}.tif"
        skm.io.imsave(dstDir + saveName, warped, check_contrast = False)
        
        # warped_mask = skm.transform.warp_polar(mask, center=(Yc, Xc), radius=meanRadius*1.3, 
        #                                   output_shape=None, scaling='linear', channel_axis=None)
        
        # [warped_contour] = skm.measure.find_contours(warped_mask) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
        # mask_integer = [warped_contour[i, 0] == int(warped_contour[i, 0]) for i in range(len(warped_contour[:, 0]))]
        # warped_contour_int = warped_contour[mask_integer]
        # warped_contour_int = warped_contour_int[:,1].astype(int)
        
        # w_ny, w_nx = warped.shape
        # Angles = np.arange(0, w_ny, 1)
        
# %% Run Viterbi on pre_warped images

gs.set_mediumText_options_jv()

PLOT = 1
SAVE = 1
plt.ioff()

#### 1. Get images pre-warped & croped

srcDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//ZStacks_Warp//Crops"
fileList = os.listdir(srcDir)
stackList = []

dstDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//ZStacks_Warp//Results"

acorrMatrix = []

for f in fileList:
    if '_ZS_' in f:
        stackList.append(f)

for f in stackList[:]:
    filePath = os.path.join(srcDir, f)
    warped = skm.io.imread(filePath)
    w_ny, w_nx = warped.shape
    Angles = np.arange(0, w_ny, 1, dtype=int)
    
    #### Settings
    inBorder = 35
    outBorder = 70
    meanRadius = 64.32 # pix
    
    #### 1.2. Interp in X
    interp_factor_x = 5
    warped_interp = ufun.resize_2Dinterp(warped, fx=interp_factor_x, fy=1)
    
    warped, w_nx = warped_interp, w_nx * interp_factor_x 
    # warped_contour_int = warped_contour_int * interp_factor_x
    inBorder, outBorder = inBorder * interp_factor_x, outBorder * interp_factor_x
    
    
    
    #### 1.3. Max Segmentation
    warped_cortex_region = warped[:,inBorder:outBorder]
    
    # for i in range(w_ny):
    #     warped_cortex_region[i, :] = warped[i, warped_contour_int[i]-inPix:warped_contour_int[i]+outPix+1]
        
    max_values = np.max(warped_cortex_region, axis = 1)
    edge_max = np.argmax(warped_cortex_region, axis = 1) + inBorder
    # edge_max = (edge_max_interp/interp_factor_x)
    
        
    #### 2. Viterbi Smoothing
    
    # Create the TreillisGraph for Viterbi tracking
    # maxWarped = np.max(warped)
    # normWarped = warped * 6 * step/maxWarped
    
    AllPeaks = []
    
    TreillisGraph = []
    warped_blur = skm.filters.gaussian(warped, sigma=(1, 4), mode='wrap')
    
    for a in Angles:
        profile = warped_blur[a, :] - np.min(warped_blur[a, inBorder:outBorder])
        profile = warped_blur[a, :] - np.min(warped_blur[a, inBorder:outBorder])
        peaks, peaks_props = signal.find_peaks(profile[inBorder:outBorder], 
                                                # width = 1,
                                                height = 0.40*np.max(profile[inBorder:outBorder]))
        AllPeaks.append(peaks + inBorder)
        
        TreillisRow = [{'angle':a, 'pos':p+inBorder, 'val':profile[p+inBorder], 'previous':0, 'accumCost':0} for p in peaks]
        TreillisGraph.append(TreillisRow)
        
    for k in range(len(TreillisGraph)):
        row = TreillisGraph[k]
        if len(row) == 0:
            TreillisGraph[k] = [node.copy() for node in TreillisGraph[k-1]]
    
    # # Get a reasonable starting point
    # starting_candidates = []
    # for R in TreillisGraph:
    #     if len(R) == 1:
    #         starting_candidates.append(R[0])
    # pos_start = np.median([p['pos'] for p in starting_candidates])
    # starting_i = argmedian([p['pos'] for p in starting_candidates])
    # starting_peak = starting_candidates[starting_i]
    
    
    # # Pretreatment of the TreillisGraph for Viterbi tracking
    # TreillisGraph = TreillisGraph[starting_i:] + TreillisGraph[:starting_i] # Set the starting point
    
    # Viterbi tracking
    best_path, nodes_list = ViterbiPathFinder(TreillisGraph)
    
    # angles_viterbi = [p['angle'] for p in nodes_list[:-1]]
    edge_viterbi = [p['pos'] for p in nodes_list]
    # edge_viterbi = edge_viterbi[1-starting_i:] + edge_viterbi[:1-starting_i] # Put it back in order
    viterbi_Npix_values = [np.sum(warped[Angles[i], edge_viterbi[i]-5:edge_viterbi[i]+6])/10 for i in range(len(edge_viterbi))]
    
    
    #### 3. Choose the profile and compute autocorrelation
    
    values = viterbi_Npix_values
    acorr_raw = autocorr_np(values)
    
    bspl = interpolate.make_interp_spline(Angles, values, k = 3)
    interp_factor_y = 10
    Angles_interp = np.arange(0, w_ny, 1/interp_factor_y, dtype=float)
    values_interp = bspl(Angles_interp)
    
    
    acorr_interp = autocorr_np(values_interp)
    
    angle_span = 45
    acorrMatrix.append(acorr_interp[:angle_span*interp_factor_y])
    
    # To compare
    acorr_max_raw = autocorr_np(max_values)
    
    bspl = interpolate.make_interp_spline(Angles, max_values, k = 3)
    interp_factor_y = 10
    Angles_interp = np.arange(0, w_ny, 1/interp_factor_y, dtype=float)
    max_values_interp = bspl(Angles_interp)
    
    
    acorr_max_interp = autocorr_np(max_values_interp)

    
    #### 5. Plot
    
    if PLOT:
        #### 5.1 Warp and Viterbi
        step = 3
        cList_viridis = plt.cm.viridis(np.linspace(0, 1, w_ny//step))
        
        figV, axV = plt.subplots(1, 3, sharey=True, figsize = (9,8))
        ax = axV[0]
        ax.imshow(warped_blur[:, :], cmap='gray', aspect='auto')
        ax.set_title('Raw warp')
        ax.set_ylabel('Angle (deg)')
        ax.set_prop_cycle(plt.cycler("color", cList_viridis))
        # ax.plot(warped_contour_int, Angles, ls='', marker='o', c='red', ms = 2)
    
        ax = axV[1]
        ax.set_prop_cycle(plt.cycler("color", cList_viridis))
        ax.set_ylabel('Intensity profile (a.u.)')
        ax.set_xlabel('Radial position (pix)')
        
        ax = axV[2]
        ax.set_prop_cycle(plt.cycler("color", cList_viridis))
        ax.set_ylabel('Intensity profile (a.u.)')
        ax.set_xlabel('Radial position (pix)')
    
        maxWarped = np.max(warped)
        normWarped = warped * 6 * step/maxWarped
    
        for a in Angles[::step]:            
            ax = axV[0]
            ax.plot([inBorder], [a], ls='', marker='o', c='gold', ms = 2)
            ax.plot([outBorder], [a], ls='', marker='o', c='gold', ms = 2)
            
            
            profile = normWarped[a, :] - np.min(normWarped[a, :])
            
            RR = np.arange(len(profile))
            ax = axV[1]
            ax.plot(RR, a - profile, zorder=2)
            ax.plot([inBorder], [a], c='k', ls='', marker='|', ms = 6)
            ax.plot([outBorder], [a], c='k', ls='', marker='|', ms = 6)
            
            ax = axV[2]
            ax.plot(RR, a - profile)
            ax.plot([inBorder], [a], c='k', ls='', marker='|', ms = 6)
            ax.plot([outBorder], [a], c='k', ls='', marker='|', ms = 6)
            # peaks, peaks_props = signal.find_peaks(profile[inBorder:outBorder], 
            #                                         height = 0.7*np.max(profile[inBorder:outBorder])) #, width = 2)
            
        # for P, a in zip(AllPeaks, Angles):
            P = AllPeaks[a]
            P = np.array(P)
            A = a * np.ones(len(P))
            ax.plot(P, A, ls='', c='orange', marker='o', ms = 4, mec = 'k') #  t - profile[inPix + peaks[k]]
            
        # ax.plot([starting_peak['pos']], [starting_peak['angle']], ls='', c='darkred', marker='o', ms = 5, mec = 'gold', zorder=6) #  t - profile[inPix + peaks[k]]
        # for p in starting_candidates:
        #     ax.plot([p['pos']], [p['angle']], ls='', c='darkred', marker='o', ms = 5, mec = 'k', zorder = 5) #  t - profile[inPix + peaks[k]]
        
            node = nodes_list[a]
            ax = axV[2]
            ax.plot([node['pos']], [node['angle']], ls='', c='cyan', marker='o', ms = 4, mec = 'k') #  t - profile[inPix + peaks[k]]
        
        for node in nodes_list[::]:
            ax = axV[0]
            ax.plot([node['pos']], [node['angle']], ls='', c='cyan', marker='.', ms = 4) #  t - profile[inPix + peaks[k]]
        
        
        #### 5.2 Warp and Profiles
        fig_w, axes_w = plt.subplots(1,4, sharey = True, figsize = (12, 8))
        ax = axes_w[0]
        ax.imshow(warped, cmap='gray', aspect='auto')
        ax.set_title('Raw warp')
        ax.set_ylabel('Angle (deg)')
        
        ax = axes_w[1]
        ax.imshow(warped, cmap='gray', aspect='auto')
        ax.plot([inBorder]*w_ny, Angles, c='gold', ls='--', lw=1, zorder = 4)
        ax.plot([outBorder]*w_ny, Angles, c='gold', ls='--', lw=1, zorder = 4)
        ax.set_title('Max Segmentation')
        ax.plot(edge_max, Angles, c='r', ls='', marker ='.', ms=4)
        
        ax = axes_w[2]
        ax.imshow(warped, cmap='gray', aspect='auto')
        ax.plot([inBorder]*w_ny, Angles, c='gold', ls='--', lw=1, zorder = 4)
        ax.plot([outBorder]*w_ny, Angles, c='gold', ls='--', lw=1, zorder = 4)
        ax.set_title('Viterbi Segmentation')
        ax.plot(edge_viterbi, Angles, c='cyan', ls='', marker ='.', ms=4)
        
        ax = axes_w[3]
        ax.set_title('Profile Values')
        ax.plot(max_values, Angles, c='r', ls='-', label = 'Max profile')
        ax.plot(viterbi_Npix_values, Angles, c='cyan', ls='-', label = 'Viterbi profile')
        ax.legend()
    
        
        def deg2um(a):
            return(a * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN))
    
        def um2deg(x):
            return(x * (360/2*np.pi) / (meanRadius/SCALE_100X_ZEN))
    
        secax = ax.secondary_yaxis('right', functions=(deg2um, um2deg))
        secax.set_ylabel('curv abs [µm]')  
        
        #### 5.3 Autocorr
        fig_c, ax_c = plt.subplots(1,1)
        S = Angles * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN)
        S_interp = Angles_interp * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN)
        ax = ax_c
        
        # ax.plot(S, acorr_raw, label='raw')
        ax.plot(S_interp, acorr_max_interp, label='max-interp')
        ax.plot(S_interp, acorr_interp, label='viterbi-interp')
        ax.set_xlabel('Curvilinear abs (µm)')
        ax.set_ylabel('Auto-correlation')
        ax.axhline(0, c='k', ls='-', lw=1)
        ax.axhline(0.75, c='b', ls='--', lw=1, zorder=1, label='75%')
        ax.axhline(0.5, c='g', ls='--', lw=1, zorder=1, label='50%')
        
        i_75p = ufun.findFirst(True, acorr_interp<0.75)
        # ax.axvline(S[i_75p-1], c='b', ls='--', lw=1, zorder=1)
        ax.text(S_interp[i_75p-1]+0.2, acorr_interp[i_75p-1]+0.05, f'{S_interp[i_75p-1]:.2f} µm')
        i_50p = ufun.findFirst(True, acorr_interp<0.5)
        # ax.axvline(S[i_50p-1], c='g', ls='--', lw=1, zorder=1)
        ax.text(S_interp[i_50p-1]+0.2, acorr_interp[i_50p-1]+0.05, f'{S_interp[i_50p-1]:.2f} µm')
        ax.legend(loc='upper right')
        
        if SAVE:
            ufun.simpleSaveFig(fig_w, f[:-4] + '_w', dstDir, '.png', 150)
            ufun.simpleSaveFig(figV,  f[:-4] + '_V', dstDir, '.png', 150)
            ufun.simpleSaveFig(fig_c, f[:-4] + '_c', dstDir, '.png', 150)
            
        plt.close('all')
        
# Average Plot
 
acorrArray = np.array(acorrMatrix)
acorr_mean = np.mean(acorrArray, axis=0)
S_interp = np.arange(0, len(acorr_mean)/interp_factor_y, 1/interp_factor_y) * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN)

fig_ac, ax_ac = plt.subplots(1,1)
ax = ax_ac
ax.plot(S_interp, acorr_mean, label='Mean auto-cor - Ncortices = {:.0f}'.format(len(stackList)))
ax.set_xlabel('Curvilinear abs (µm)')
ax.set_ylabel('Auto-correlation')
ax.axhline(0, c='k', ls='-', lw=1)
ax.axhline(0.75, c='b', ls='--', lw=1, zorder=1, label='75%')
ax.axhline(0.5, c='g', ls='--', lw=1, zorder=1, label='50%')

i_75p = ufun.findFirst(True, acorr_mean<0.75)
# ax.axvline(S[i_75p-1], c='b', ls='--', lw=1, zorder=1)
ax.text(S_interp[i_75p-1]+0.2, acorr_mean[i_75p-1]+0.05, f'{S_interp[i_75p-1]:.2f} µm')
i_50p = ufun.findFirst(True, acorr_mean<0.5)
# ax.axvline(S[i_50p-1], c='g', ls='--', lw=1, zorder=1)
ax.text(S_interp[i_50p-1]+0.2, acorr_mean[i_50p-1]+0.05, f'{S_interp[i_50p-1]:.2f} µm')
ax.legend(loc='upper right')

if SAVE:
    ufun.simpleSaveFig(fig_ac,  '00_Average_acor', dstDir, '.png', 150)
    
plt.tight_layout()
plt.show()
    

# %% Test Temporal Correl With Viterbi


srcDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//"
fileList = os.listdir(srcDir)
stackList = []

dstDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//testTL"

acorrMatrix = []
radiusList = []

PLOT = 1
SAVE = 0
gs.set_smallText_options_jv()
plt.ioff()

for f in fileList:
    if ('_TL_' in f) and not ('_bf' in f):
        stackList.append(f)

f = stackList[2]
filePath = os.path.join(srcDir, f)

I = skm.io.imread(filePath)
nt, ny, nx, nz = I.shape
I_down = I[:,:,:,0]
I_mid = I[:,:,:,1]
I_up = I[:,:,:,2]

timeLapse_profiles = np.zeros((nz, nt, 360))

I3z = [I_down, I_mid, I_up]

for z in range(3): #[I_down, I_mid, I_up]:
    I1z = I3z[z]
    
    for t in range(nt):
        I0 = I1z[t]
        
        #### Settings
        framesize = int(40*8/2) 
        inPix, outPix = 20, 15
        blur_parm = 4
        relative_height_virebi = 0.4
        interp_factor_x = 5
        interp_factor_y = 10
        VEW = 7 # Viterbi edge width # Odd number
        
        # try:
        
        #### 1. Detect
        # model = models.CellposeModel(gpu=False, model_type='TN1')
        # channels = [0, 0]
        # [mask], [flow], [style] = model.eval([I0], diameter = 120, channels=channels,
        #                                     flow_threshold = 0.4, cellprob_threshold = 0.1, 
        #                                     do_3D=False, normalize = True) # masks, flows, styles, diams
        # Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)
        # Yc0, Xc0 = int(Yc), int(Xc)
        
        # I0 = I0[Yc0-framesize:Yc0+framesize, Xc0-framesize:Xc0+framesize]
        
        # [mask], [flow], [style] = model.eval([I0], diameter = 120, channels=channels,
        #                                     flow_threshold = 0.4, cellprob_threshold = 0.1, 
        #                                     do_3D=False, normalize = True) # masks, flows, styles, diams
        # Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)
        
        # [contour] = skm.measure.find_contours(mask, 0.5) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
        # listDistances = np.sqrt((contour[:,0] - Yc)**2 + (contour[:,1] - Xc)**2)
        # meanRadius = np.mean(listDistances)
        
        (Yc, Xc), R = findCellInnerCircle(I0, plot=False)
        
        Xi, Yi = round(Xc), round(Yc)
        x1, y1, x2, y2 = Xi-framesize, Yi-framesize, Xi+framesize+1, Yi+framesize+1
        x1, y1, x2, y2 = getValidROIBounds(I0, x1, y1, x2, y2)
        I0 = I0[y1:y2, x1:x2]
        
        (Yc, Xc), Radius = findCellInnerCircle(I0, plot=False)
        
        radiusList.append(Radius)
        
        
        #### 2.1 Warp
        warped = skm.transform.warp_polar(I0, center=(Yc, Xc), radius=R*1.3, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        # warped_mask = skm.transform.warp_polar(mask, center=(Yc, Xc), radius=R*1.3, 
        #                                   output_shape=None, scaling='linear', channel_axis=None)
        
        # [warped_contour] = skm.measure.find_contours(warped_mask) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
        # mask_integer = [warped_contour[i, 0] == int(warped_contour[i, 0]) for i in range(len(warped_contour[:, 0]))]
        # warped_contour_int = warped_contour[mask_integer]
        # warped_contour_int = warped_contour_int[:,1].astype(int)
        
        w_ny, w_nx = warped.shape
        Angles = np.arange(0, w_ny, 1)
        
        
        #### 2.2. Interp in X
        warped_interp = ufun.resize_2Dinterp(warped, fx=interp_factor_x, fy=1)
        
        warped, w_nx = warped_interp, w_nx * interp_factor_x 
        # warped_contour_int = warped_contour_int * interp_factor_x
        inPix, outPix = inPix * interp_factor_x, outPix * interp_factor_x
        
        
        #### 2.3. Max Segmentation
        # warped_cortex_region = np.zeros((w_ny, 1 + inPix + outPix), dtype = np.float64) # 1 + inPix + outPix
        
        # for i in range(w_ny):
        #     warped_cortex_region[i, :] = warped[i, warped_contour_int[i]-inPix:warped_contour_int[i]+outPix+1]
            
        # max_values = np.max(warped_cortex_region, axis = 1)
        # edge_max = np.argmax(warped_cortex_region, axis = 1) + warped_contour_int[:] - inPix
        
        #### 3. Viterbi Smoothing
        
        # Create the TreillisGraph for Viterbi tracking
        # maxWarped = np.max(warped)
        # normWarped = warped * 6 * step/maxWarped
        
        
        AllPeaks = []
        TreillisGraph = []
        warped_blured = skm.filters.gaussian(warped, sigma=(1, blur_parm), mode='wrap')
        
        for a in Angles:
            # inBorder = warped_contour_int[a] - inPix
            # outBorder = warped_contour_int[a] + outPix
            inBorder = round(Radius * interp_factor_x) - inPix
            outBorder = round(Radius * interp_factor_x) + outPix
            profile = warped[a, :] - np.min(warped[a, inBorder:outBorder])
            peaks, peaks_props = signal.find_peaks(profile[inBorder:outBorder], 
                                                    # width = 4,
                                                    height = relative_height_virebi*np.max(profile[inBorder:outBorder]))
            AllPeaks.append(peaks + inBorder)
            
            TreillisRow = [{'angle':a, 'pos':p+inBorder, 'val':profile[p+inBorder], 'previous':0, 'accumCost':0} for p in peaks]
            TreillisGraph.append(TreillisRow)
            
        for k in range(len(TreillisGraph)):
            row = TreillisGraph[k]
            if len(row) == 0:
                TreillisGraph[k] = [node.copy() for node in TreillisGraph[k-1]]
        
        # Get a reasonable starting point
        starting_candidates = []
        for R in TreillisGraph:
            if len(R) == 1:
                starting_candidates.append(R[0])
        pos_start = np.median([p['pos'] for p in starting_candidates])
        starting_i = argmedian([p['pos'] for p in starting_candidates])
        starting_peak = starting_candidates[starting_i]
        
        
        # Pretreatment of the TreillisGraph for Viterbi tracking
        TreillisGraph = TreillisGraph[starting_i:] + TreillisGraph[:starting_i] # Set the strating point
        TreillisGraph.append([node.copy() for node in TreillisGraph[0]]) # Make the graph cyclical
        
        # Viterbi tracking
        best_path, nodes_list = ViterbiPathFinder(TreillisGraph)
        
        
        # angles_viterbi = [p['angle'] for p in nodes_list[:-1]]
        edge_viterbi = [p['pos'] for p in nodes_list[:-1]]
        edge_viterbi = edge_viterbi[1-starting_i:] + edge_viterbi[:1-starting_i] # Put it back in order
        viterbi_Npix_values = [np.sum(warped[Angles[i], edge_viterbi[i]-VEW//2:edge_viterbi[i] + (VEW//2 + 1)])/VEW for i in range(len(edge_viterbi))]
        
        timeLapse_profiles[z, t] = viterbi_Npix_values
        
        #### 5. Plot
        
        if PLOT:
            #### 5.1 Cellpose
            fig_I, axes_I = plt.subplots(1,2, figsize = (12, 6), sharey = True)
            ax = axes_I[0]
            ax.imshow(I0, cmap='gray')
            # ax.plot(contour[:,1], contour[:,0], 'r--', lw=1, zorder = 4)
            ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
            theta = np.linspace(0, 2*np.pi, 100)
            CircleY, CircleX = Yc + Radius*np.sin(theta), Xc + Radius*np.cos(theta)
            # ax.plot(CircleX, CircleY, c='cyan', ls='--', lw=1)
            
            ax = axes_I[1]
            ax.imshow(I0, cmap='gray')
            ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
            CircleYin = Yc + (Radius-(inPix/interp_factor_x))*np.sin(theta)
            CircleXin = Xc + (Radius-(inPix/interp_factor_x))*np.cos(theta)
            ax.plot(CircleXin, CircleYin, c='gold', ls='--', lw=1)
            CircleYout = Yc + (Radius+(outPix/interp_factor_x)+1)*np.sin(theta)
            CircleXout = Xc + (Radius+(outPix/interp_factor_x)+1)*np.cos(theta)
            ax.plot(CircleXout, CircleYout, c='gold', ls='--', lw=1)
            
            
            #### 5.2 Warp and Viterbi
            step = 5
            cList_viridis = plt.cm.viridis(np.linspace(0, 1, w_ny//step))
            
            fig_V, ax_V = plt.subplots(1, 3, sharey=True, figsize = (9,8))
            ax = ax_V[0]
            ax.imshow(warped_blured[:, :], cmap='gray', aspect='auto')
            ax.set_title('Raw warp')
            ax.set_ylabel('Angle (deg)')
            ax.set_prop_cycle(plt.cycler("color", cList_viridis))
            # ax.plot(warped_contour_int, Angles, ls='', marker='o', c='red', ms = 2)
        
            ax = ax_V[1]
            ax.set_prop_cycle(plt.cycler("color", cList_viridis))
            ax.set_ylabel('Intensity profile (a.u.)')
            ax.set_xlabel('Radial position (pix)')
            
            ax = ax_V[2]
            ax.set_prop_cycle(plt.cycler("color", cList_viridis))
            ax.set_ylabel('Intensity profile (a.u.)')
            ax.set_xlabel('Radial position (pix)')
        
            maxWarped = np.max(warped)
            normWarped = warped * 6 * step/maxWarped
        
            for a in Angles[::step]:            
                ax = ax_V[0]
                ax.plot([inBorder], [a], ls='', marker='o', c='gold', ms = 2)
                ax.plot([outBorder], [a], ls='', marker='o', c='gold', ms = 2)
                
                profile = normWarped[a, :] - np.min(normWarped[a, :])
                
                RR = np.arange(len(profile))
                ax = ax_V[1]
                ax.plot(RR, a - profile, zorder=2)
                ax.plot([inBorder], [a], c='k', ls='', marker='|', ms = 6)
                ax.plot([outBorder], [a], c='k', ls='', marker='|', ms = 6)
                
                ax = ax_V[2]
                ax.plot(RR, a - profile)
                ax.plot([inBorder], [a], c='k', ls='', marker='|', ms = 6)
                ax.plot([outBorder], [a], c='k', ls='', marker='|', ms = 6)
                # peaks, peaks_props = signal.find_peaks(profile[inBorder:outBorder], 
                #                                         height = 0.7*np.max(profile[inBorder:outBorder])) #, width = 2)
                
            # for P, a in zip(AllPeaks, Angles):
                P = AllPeaks[a]
                P = np.array(P)
                A = a * np.ones(len(P))
                ax.plot(P, A, ls='', c='orange', marker='o', ms = 4, mec = 'k') #  t - profile[inPix + peaks[k]]
                
            # ax.plot([starting_peak['pos']], [starting_peak['angle']], ls='', c='darkred', marker='o', ms = 5, mec = 'gold', zorder=6) #  t - profile[inPix + peaks[k]]
            # for p in starting_candidates:
            #     ax.plot([p['pos']], [p['angle']], ls='', c='darkred', marker='o', ms = 5, mec = 'k', zorder = 5) #  t - profile[inPix + peaks[k]]
            
                node = nodes_list[a]
                ax = ax_V[2]
                ax.plot([node['pos']], [node['angle']], ls='', c='cyan', marker='o', ms = 4, mec = 'k') #  t - profile[inPix + peaks[k]]
            
            for node in nodes_list[::]:
                ax = ax_V[0]
                ax.plot([node['pos']], [node['angle']], ls='', c='cyan', marker='.', ms = 4) #  t - profile[inPix + peaks[k]]
            
            
            #### 5.3 Warp and Profiles
            fig_w, axes_w = plt.subplots(1,4, sharey = True, figsize = (12, 8))
            ax = axes_w[0]
            ax.imshow(warped, cmap='gray', aspect='auto')
            ax.set_title('Raw warp')
            ax.set_ylabel('Angle (deg)')
            
            ax = axes_w[1]
            ax.imshow(warped, cmap='gray', aspect='auto')
            # ax.plot(warped_contour_int[:]-inPix, Angles, c='gold', ls='--', lw=1, zorder = 4)
            # ax.plot(warped_contour_int[:]+outPix+1, Angles, c='gold', ls='--', lw=1, zorder = 4)
            ax.set_title('Max Segmentation')
            # ax.plot(edge_max, Angles, c='cyan', ls='', marker ='.', ms=2)
            
            ax = axes_w[2]
            ax.imshow(warped, cmap='gray', aspect='auto')
            # ax.plot(warped_contour_int[:]-inPix, Angles, c='gold', ls='--', lw=1, zorder = 4)
            # ax.plot(warped_contour_int[:]+outPix+1, Angles, c='gold', ls='--', lw=1, zorder = 4)
            ax.set_title('Viterbi Segmentation')
            ax.plot(edge_viterbi, Angles, c='cyan', ls='', marker ='.', ms=2)
            
            ax = axes_w[3]
            ax.set_title('Profile Values')
            # ax.plot(max_values, Angles, 'r-', label = 'Max profile')
            ax.plot(viterbi_Npix_values, Angles, 'b-', label = 'Viterbi profile')
            ax.legend()
        
            
            def deg2um(a):
                return(a * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN))
        
            def um2deg(x):
                return(x * (360/2*np.pi) / (meanRadius/SCALE_100X_ZEN))
        
            secax = ax.secondary_yaxis('right', functions=(deg2um, um2deg))
            secax.set_ylabel('curv abs [µm]')  
            
            #### 5.4 Autocorr
            # fig_c, ax_c = plt.subplots(1,1)
            # ax = ax_c
            # S = Angles * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN)
            # S_interp = Angles_interp * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN)
            # S_binned = Angles_binned * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN)
            
            # # ax.plot(S, acorr_raw, label='raw')
            # ax.plot(S_interp, acorr_max_interp, label='max-interp')
            # ax.plot(S_interp, acorr_interp, label='viterbi-interp')
            # ax.set_xlabel('Curvilinear abs (µm)')
            # ax.set_ylabel('Auto-correlation')
            # ax.axhline(0, c='k', ls='-', lw=1)
            # ax.axhline(0.75, c='b', ls='--', lw=1, zorder=1, label='75%')
            # ax.axhline(0.5, c='g', ls='--', lw=1, zorder=1, label='50%')
            
            # i_75p = ufun.findFirst(True, acorr_interp<0.75)
            # # ax.axvline(S[i_75p-1], c='b', ls='--', lw=1, zorder=1)
            # ax.text(S_interp[i_75p-1]+0.2, acorr_interp[i_75p-1]+0.05, f'{S_interp[i_75p-1]:.2f} µm')
            # i_50p = ufun.findFirst(True, acorr_interp<0.5)
            # # ax.axvline(S[i_50p-1], c='g', ls='--', lw=1, zorder=1)
            # ax.text(S_interp[i_50p-1]+0.2, acorr_interp[i_50p-1]+0.05, f'{S_interp[i_50p-1]:.2f} µm')
            # ax.legend(loc='upper right')
            
            if SAVE:
                ufun.simpleSaveFig(fig_I, f[:-4] + f'_I_{kk:.0f}', dstDir + '//' + f, '.png', 150)
                ufun.simpleSaveFig(fig_w, f[:-4] + f'_w_{kk:.0f}', dstDir + '//' + f, '.png', 150)
                ufun.simpleSaveFig(fig_V, f[:-4] + f'_V_{kk:.0f}', dstDir + '//' + f, '.png', 150)
                ufun.simpleSaveFig(fig_c, f[:-4] + f'_c_{kk:.0f}', dstDir + '//' + f, '.png', 150)
                
            # plt.close('all')
                
        # except:
        #### 5.1 Cellpose
        # fig_Error, axes_Error = plt.subplots(1,1, figsize = (6, 6), sharey = True)
        # ax = axes_Error
        # ax.imshow(I0, cmap='gray')
        # # ax.plot(contour[:,1], contour[:,0], 'r--', lw=1, zorder = 4)
        # # ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
        # # theta = np.linspace(0, 2*np.pi, 100)
        # # CircleY, CircleX = Yc + meanRadius*np.sin(theta), Xc + meanRadius*np.cos(theta)
        # # ax.plot(CircleX, CircleY, c='cyan', ls='--', lw=1)
        # plt.show()
        

figTL, axesTL = plt.subplots(3, 1, figsize=(9, 9))
LabelZ = ['Down', 'Mid', 'Up']
for z in range(3):
    ax = axesTL[-1-z]
    LZ = LabelZ[z]
    cList_viridis = plt.cm.viridis(np.linspace(0, 1, nt))
    ax.set_prop_cycle(plt.cycler("color", cList_viridis))
    ax.set_ylabel(f'Intensity profile - {LZ}')
    ax.set_xlabel('Radial position (pix)')
    profiles_z = timeLapse_profiles[z]
    for t in range(len(profiles_z)):
        ax.plot(Angles, profiles_z[t])

for z in range(3):
    profiles_z = timeLapse_profiles[z]
    nA = 360
    step = 10
    Ngraphs = nA//step
    Ncols = 4
    Nrows = Ngraphs//Ncols
    mini_va, Maxi_va = np.min(profiles_z[:, ::step]), np.max(profiles_z[:, ::step])
    figTL2, axTL2 = plt.subplots(Nrows, Ncols, figsize = (3*Ncols, 1.5*Nrows))
    cList_viridis = plt.cm.viridis(np.linspace(0, 1, nt))
    for i in range(Ngraphs):
        angle = i * step
        ax = (axTL2.T).flatten()[i]
        ax.set_prop_cycle(plt.cycler("color", cList_viridis))
        for j in range(nt):
            T = 4 * j
            val = profiles_z[j, angle]
            ax.plot(T, val, ls='', marker='o', ms = 4)
            ax.set_ylim([mini_va, Maxi_va])
            ax.set_ylabel(f'{angle}°')
            
    for c in range(Ncols):
        for ax in axTL2[:-1,c]:
            ax.set_xticklabels([])
        ax = axTL2[-1,c]
        ax.set_xlabel('T (s)')  
        
    plt.tight_layout()
    plt.show()

        

plt.tight_layout()
plt.show()


# %% Many autocorrelation functions

# def autocorr_np(x):
#     n = x.size
#     norm = (x - np.mean(x))
#     result = np.correlate(norm, norm, mode='same')
#     acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
#     return(acorr)

# def autocorr_np2(x):
#     """The good one"""
    
#     x = np.array(x) 

#     # Mean
#     mean = np.mean(x)
    
#     # Variance
#     var = np.var(x)
    
#     # Normalized data
#     ndata = x - mean
    
#     acorr = np.correlate(ndata, ndata, 'full')[len(ndata)-1:] 
#     acorr = (acorr / var) / len(ndata)
    
#     return(acorr)

# def autocorr_np3(x):
#     x = np.array(x) 

#     # Mean
#     mean = np.mean(x)
    
#     # Variance
#     var = np.var(x)
    
#     # Normalized data
#     ndata = (x - mean) / var
    
#     acorr = np.correlate(ndata, ndata, 'full')[len(ndata)-1:] 
#     acorr = (acorr) / len(ndata)
#     return(acorr)

# def autocorr_custom(x):
#     lags = np.arange(0, len(x)//2)
#     # Pre-allocate autocorrelation table
#     acorr = np.zeros(len(lags))
    
#     # Mean
#     mean = sum(x) / len(x) 
    
#     # Variance
#     var = sum([(i - mean)**2 for i in x]) / len(x) 
    
#     # Normalized data
#     ndata = [i - mean for i in x]
    
    
#     # Go through lag components one-by-one
#     for l in lags:
#         c = 1 # Self correlation
        
#         if (l > 0):
#             tmp = [ndata[l:][i] * ndata[:-l][i] for i in range(len(x) - l)]
#             c = (sum(tmp) / len(x)) / var
            
#         acorr[l] = c
    
#     return(acorr)


# %% Old versions

# %%% Old V1

srcDir = "D://MicroscopyData//2022-11-02_DrugAssay_LatA_3T3aSFL-LaGFP_6co//TestAnalysisPython//C1_ctrl"
fileName = "C1_S2_w1CSU-488-Em-525-30-2_other.tif"

filePath = os.path.join(srcDir, fileName)

I = skm.io.imread(filePath)
sumArray = np.sum(I, axis=(1,2))
z0 = np.argmax(sumArray)
I0 = I[z0]


fig, ax = plt.subplots(1,1)
ax.imshow(I0, cmap='gray')
plt.show()

factor = 4
I0 = skm.exposure.rescale_intensity(I0, 'image')
I0 = ufun.resize_2Dinterp(I0, fx=factor, fy=factor)
I0 = np.max(I0) - I0

th_otsu = skm.filters.threshold_otsu(I0)
I0_bin = np.array(I0 < th_otsu).astype(int)
H = skm.transform.hough_circle(I0, factor*10*15.8*0.4/1.2)
[accum], [cx], [cy], [rad] = skm.transform.hough_circle_peaks(H, [int(factor*10*15.8*0.4/1.2)], min_xdistance=int(factor*16*0.4/1.2), min_ydistance=int(factor*16*0.4/1.2), 
                                                  threshold=None, num_peaks=1, total_num_peaks=1, normalize=False)
print(accum, cx, cy, rad)
rr, cc = skm.draw.circle_perimeter(cy, cx, rad, method='bresenham')

# I0[rr, cc] = 0
fig, ax = plt.subplots(1,1)
ax.imshow(I0, cmap='gray')
I0_bin[rr, cc] = 1
fig, ax = plt.subplots(1,1)
ax.imshow(I0_bin, cmap='gray')

warped = skm.transform.warp_polar(I0, center=(cy, cx), radius=rad*1.3, output_shape=None, scaling='linear', channel_axis=None)

fig, ax = plt.subplots(1,1)
ax.imshow(warped, cmap='gray')

max_position = np.argmax(warped[:, factor*46:factor*54], axis = 1)

vals = []
for i in range(len(max_position)):
    ax.plot(factor*45+max_position[i], i, marker = '.', markersize = 1, ls='', color = 'b')
    vals.append(warped[i, factor*45+max_position[i]])
    
fig, ax = plt.subplots(1,1)
ax.plot([i for i in range(len(vals))], vals)


# original = np.uint8(I0)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# equalised_img = clahe.apply(original)

# fig, ax = plt.subplots(1,1)
# ax.imshow(equalised_img, cmap='gray')

model = models.Cellpose(gpu=False, model_type='TN1')
imgs = [I0]

channels = [0,0]
masks, flows, styles, diams = model.eval(imgs, diameter = 110, channels=channels,
                                          flow_threshold = 0.4, cellprob_threshold = 1.0, do_3D=False, normalize = True)

fig, ax = plt.subplots(1,1)
ax.imshow(masks[0], cmap='gray')
print(np.sum(masks))

# ax.contour(mask, levels = 0, linestyles = ('--'), colors = ('r'))


# %%% Test Temporal Correl 01


srcDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//"
fileList = os.listdir(srcDir)
stackList = []

dstDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//testTL"

acorrMatrix = []
radiusList = []

PLOT = 1
SAVE = 0

for f in fileList:
    if '_TL_' in f:
        stackList.append(f)

f = stackList[2]
filePath = os.path.join(srcDir, f)

I = skm.io.imread(filePath)
nt, ny, nx, nz = I.shape
I_down = I[:,:,:,0]
I_mid = I[:,:,:,1]
I_up = I[:,:,:,2]

for I in [I_up]: #[I_down, I_mid, I_up]:
    
    t = 0
    I0 = I[t]

    #### Settings
    framesize = int(35*8/2) 
    inPix, outPix = 20, 10
    
    #### I - Test on first frame
    
    #### I.1. Detect
    model = models.CellposeModel(gpu=False, model_type='TN1')
    channels = [0, 0]
    [mask], [flow], [style] = model.eval([I0], diameter = 110, channels=channels,
                                        flow_threshold = 0.4, cellprob_threshold = 0.4, 
                                        do_3D=False, normalize = True) # masks, flows, styles, diams
    Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)
    Yc0, Xc0 = int(Yc), int(Xc)
    
    I0 = I0[Yc0-framesize:Yc0+framesize, Xc0-framesize:Xc0+framesize]
    [mask], [flow], [style] = model.eval([I0], diameter = 110, channels=channels,
                                        flow_threshold = 0.4, cellprob_threshold = 0.4, 
                                        do_3D=False, normalize = True) # masks, flows, styles, diams
    Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)
    
    [contour] = skm.measure.find_contours(mask, 0.5) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
    listDistances = np.sqrt((contour[:,0] - Yc)**2 + (contour[:,1] - Xc)**2)
    meanRadius = np.mean(listDistances)
    radiusList.append(meanRadius)
    
    
    #### I.2. Warp
    
    warped = skm.transform.warp_polar(I0, center=(Yc, Xc), radius=meanRadius*1.3, 
                                      output_shape=None, scaling='linear', channel_axis=None)
    warped_mask = skm.transform.warp_polar(mask, center=(Yc, Xc), radius=meanRadius*1.3, 
                                      output_shape=None, scaling='linear', channel_axis=None)
    
    [warped_contour] = skm.measure.find_contours(warped_mask) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
    mask_integer = [warped_contour[i, 0] == int(warped_contour[i, 0]) for i in range(len(warped_contour[:, 0]))]
    warped_contour_int = warped_contour[mask_integer]
    warped_contour_int = warped_contour_int.astype(int)
    
    ny, nx = warped.shape
    yy = np.arange(0, ny)
    warped_cortex_region = np.zeros((ny, 1 + inPix + outPix), dtype = np.float64)
    
    for i in range(ny):
        warped_cortex_region[i, :] = warped[i, warped_contour_int[i,1]-inPix:warped_contour_int[i,1]+outPix+1]
    interp_factor = 5
    warped_cortex_region_interp = ufun.resize_2Dinterp(warped_cortex_region, fx=interp_factor, fy=1)
    max_values = np.max(warped_cortex_region_interp, axis = 1)
    sum_values = np.sum(warped_cortex_region_interp, axis = 1)
    max_position_0 = np.argmax(warped_cortex_region_interp, axis = 1)
    max_position_1 = (max_position_0/interp_factor) + warped_contour_int[:,1] - inPix
            
    if PLOT:
        fig_I, axes_I = plt.subplots(1,2, figsize = (12, 6), sharey = True)
        ax = axes_I[0]
        ax.imshow(I0, cmap='gray')
        ax.plot(contour[:,1], contour[:,0], 'r--', lw=1, zorder = 4)
        ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
        theta = np.linspace(0, 2*np.pi, 100)
        CircleY, CircleX = Yc + meanRadius*np.sin(theta), Xc + meanRadius*np.cos(theta)
        # ax.plot(CircleX, CircleY, c='cyan', ls='--', lw=1)
        
        ax = axes_I[1]
        ax.imshow(I0, cmap='gray')
        ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
        CircleYin, CircleXin = Yc + (meanRadius-inPix)*np.sin(theta), Xc + (meanRadius-inPix)*np.cos(theta)
        ax.plot(CircleXin, CircleYin, c='gold', ls='--', lw=1)
        CircleYout, CircleXout = Yc + (meanRadius+outPix+1)*np.sin(theta), Xc + (meanRadius+outPix+1)*np.cos(theta)
        ax.plot(CircleXout, CircleYout, c='gold', ls='--', lw=1)
        
        fig_w, axes_w = plt.subplots(1,4, sharey = True)
        ax = axes_w[0]
        ax.imshow(warped, cmap='gray')
        ax.set_title('Raw warp')
        
        ax = axes_w[1]
        ax.imshow(warped, cmap='gray')
        ax.plot(warped_contour_int[:,1]-inPix, warped_contour_int[:,0], c='gold', ls='--', lw=1, zorder = 4)
        ax.plot(warped_contour_int[:,1]+outPix+1, warped_contour_int[:,0], c='gold', ls='--', lw=1, zorder = 4)
        ax.set_title('Segmentation')
        
        ax = axes_w[1]
        ax.plot(max_position_1, yy, c='cyan', ls='', marker ='.', ms=1)
        ax = axes_w[2]
        ax.plot(max_values, yy, 'r-')
        ax.set_title('Max')
        ax = axes_w[3]
        ax.plot(sum_values, yy, 'b-')
        ax.set_title('Sum')
        
        
        plt.tight_layout()
    
    plt.show()
    
    #### II - Iterate along time
    
    valsMatrix = []
    warpMatrix = []
    timeList = []
    
    for t in range(nt):
        I0 = I[t]
        
        #### Settings
        framesize = int(35*8/2) 
        inPix, outPix = 20, 10
        
        #### 1. Detect
        model = models.CellposeModel(gpu=False, model_type='TN1')
        channels = [0, 0]
        [mask], [flow], [style] = model.eval([I0], diameter = 110, channels=channels,
                                            flow_threshold = 0.4, cellprob_threshold = 0.4, 
                                            do_3D=False, normalize = True) # masks, flows, styles, diams
        Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)
        Yc0, Xc0 = int(Yc), int(Xc)
        
        I0 = I0[Yc0-framesize:Yc0+framesize, Xc0-framesize:Xc0+framesize]
        [mask], [flow], [style] = model.eval([I0], diameter = 110, channels=channels,
                                            flow_threshold = 0.4, cellprob_threshold = 0.4, 
                                            do_3D=False, normalize = True) # masks, flows, styles, diams
        Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)
        
        [contour] = skm.measure.find_contours(mask, 0.5) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
        listDistances = np.sqrt((contour[:,0] - Yc)**2 + (contour[:,1] - Xc)**2)
        meanRadius = np.mean(listDistances)
        radiusList.append(meanRadius)
        
        
        #### 2. Warp
        
        warped = skm.transform.warp_polar(I0, center=(Yc, Xc), radius=meanRadius*1.3, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        warped_mask = skm.transform.warp_polar(mask, center=(Yc, Xc), radius=meanRadius*1.3, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        
        [warped_contour] = skm.measure.find_contours(warped_mask) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
        mask_integer = [warped_contour[i, 0] == int(warped_contour[i, 0]) for i in range(len(warped_contour[:, 0]))]
        warped_contour_int = warped_contour[mask_integer]
        warped_contour_int = warped_contour_int.astype(int)
        
        ny, nx = warped.shape
        yy = np.arange(0, ny)
        warped_cortex_region = np.zeros((ny, 1 + inPix + outPix), dtype = np.float64)
        
        for i in range(ny):
            warped_cortex_region[i, :] = warped[i, warped_contour_int[i,1]-inPix:warped_contour_int[i,1]+outPix+1]
            
        #### BUG HERE!!
            
        interp_factor = 5
        warped_cortex_region_interp = ufun.resize_2Dinterp(warped_cortex_region, fx=interp_factor, fy=1)
        max_values = np.max(warped_cortex_region_interp, axis = 1)
        sum_values = np.sum(warped_cortex_region_interp, axis = 1)
        max_position_0 = np.argmax(warped_cortex_region_interp, axis = 1)
        max_position_1 = (max_position_0/interp_factor) + warped_contour_int[:,1] - inPix
        
        valsMatrix.append(max_values)
        warpMatrix.append(warped[:, int(meanRadius-30):int(meanRadius+15)])
        timeList.append(t)
        
    
    cList_viridis = plt.cm.viridis(np.linspace(0, 1, len(valsMatrix)))
        
    fig_s, axes_s = plt.subplots(2, 1, sharex = True, figsize = (16, 6))
    fig_s.suptitle(f[:-4] + '_ZSum')
    sumWarp = np.sum(warpMatrix, axis=0)
    ny, nx = sumWarp.shape
    yy = np.arange(0, ny)
    
    ax = axes_s[0]
    ax.set_title('Warped cell')
    ax.imshow(sumWarp.T, cmap='viridis')
    
    ax = axes_s[1]
    ax.set_title('Max')
    ax.set_prop_cycle(plt.cycler("color", cList_viridis))
    for k in range(len(valsMatrix)):
        time_sec = timeList[k] * 4
        ax.plot(yy, valsMatrix[k], label = f'$+{time_sec:.0f} sec$')
        
    ax.legend()
        
    # ufun.simpleSaveFig(fig_s, f[:-4] + '_ZSum', dstDir, '.png', 150)



# %%% Test Vertical Correl 01

gs.set_smallText_options_jv()

PLOT = 1
SAVE = 1
plt.ioff()

srcDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//"
fileList = os.listdir(srcDir)
stackList = []

dstDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//testZ"

acorrMatrix = []
radiusList = []

for f in fileList:
    if '_ZS_' in f:
        stackList.append(f)

for f in stackList[5:]: # 
    filePath = os.path.join(srcDir, f)

    I = skm.io.imread(filePath)
    sumArray = np.sum(I, axis=(1,2))
    z0 = np.argmax(sumArray)
    
    valsMatrix = []
    warpMatrix = []
    offsetList = []
    
    print(f, z0)
    
    for kk in range(16, 25):# 30):
        
        I0 = I[z0+kk]
        
        #### Settings
        framesize = int(35*8/2) 
        inPix, outPix = 20, 10
        
        #### 1. Detect
        model = models.CellposeModel(gpu=False, model_type='TN1')
        channels = [0, 0]
        [mask], [flow], [style] = model.eval([I0], diameter = 110, channels=channels,
                                            flow_threshold = 0.4, cellprob_threshold = 0.4, 
                                            do_3D=False, normalize = True) # masks, flows, styles, diams
        Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)
        Yc0, Xc0 = int(Yc), int(Xc)
        
        I0 = I0[Yc0-framesize:Yc0+framesize, Xc0-framesize:Xc0+framesize]
        [mask], [flow], [style] = model.eval([I0], diameter = 110, channels=channels,
                                            flow_threshold = 0.4, cellprob_threshold = 0.4, 
                                            do_3D=False, normalize = True) # masks, flows, styles, diams
        Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)
        
        [contour] = skm.measure.find_contours(mask, 0.5) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
        listDistances = np.sqrt((contour[:,0] - Yc)**2 + (contour[:,1] - Xc)**2)
        meanRadius = np.mean(listDistances)
        print(f, kk, meanRadius)
        radiusList.append(meanRadius)
        
        fig_I, axes_I = plt.subplots(1,2, figsize = (12, 6), sharey = True)
        ax = axes_I[0]
        ax.imshow(I0, cmap='gray')
        ax.plot(contour[:,1], contour[:,0], 'r--', lw=1, zorder = 4)
        ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
        theta = np.linspace(0, 2*np.pi, 100)
        CircleY, CircleX = Yc + meanRadius*np.sin(theta), Xc + meanRadius*np.cos(theta)
        # ax.plot(CircleX, CircleY, c='cyan', ls='--', lw=1)
        
        ax = axes_I[1]
        ax.imshow(I0, cmap='gray')
        ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
        CircleYin, CircleXin = Yc + (meanRadius-inPix)*np.sin(theta), Xc + (meanRadius-inPix)*np.cos(theta)
        ax.plot(CircleXin, CircleYin, c='gold', ls='--', lw=1)
        CircleYout, CircleXout = Yc + (meanRadius+outPix+1)*np.sin(theta), Xc + (meanRadius+outPix+1)*np.cos(theta)
        ax.plot(CircleXout, CircleYout, c='gold', ls='--', lw=1)
        
        #### 2. Warp
        
        warped = skm.transform.warp_polar(I0, center=(Yc, Xc), radius=meanRadius*1.3, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        warped_mask = skm.transform.warp_polar(mask, center=(Yc, Xc), radius=meanRadius*1.3, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        
        [warped_contour] = skm.measure.find_contours(warped_mask) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
        mask_integer = [warped_contour[i, 0] == int(warped_contour[i, 0]) for i in range(len(warped_contour[:, 0]))]
        warped_contour_int = warped_contour[mask_integer]
        warped_contour_int = warped_contour_int.astype(int)
        
        ny, nx = warped.shape
        yy = np.arange(0, ny)
        warped_cortex_region = np.zeros((ny, 1 + inPix + outPix), dtype = np.float64)
        
        for i in range(ny):
            warped_cortex_region[i, :] = warped[i, warped_contour_int[i,1]-inPix:warped_contour_int[i,1]+outPix+1]
        interp_factor = 5
        warped_cortex_region_interp = ufun.resize_2Dinterp(warped_cortex_region, fx=interp_factor, fy=1)
        max_values = np.max(warped_cortex_region_interp, axis = 1)
        sum_values = np.sum(warped_cortex_region_interp, axis = 1)
        max_position_0 = np.argmax(warped_cortex_region_interp, axis = 1)
        max_position_1 = (max_position_0/interp_factor) + warped_contour_int[:,1] - inPix
        
        def deg2um(a):
            return(a * (2*np.pi/360) * (meanRadius/SCALE_100X_ZEN))
    
        def um2deg(x):
            return(x * (360/2*np.pi) / (meanRadius/SCALE_100X_ZEN))
        
        secax = ax.secondary_yaxis('right', functions=(deg2um, um2deg))
        secax.set_ylabel('curv abs [µm]')        
        
        valsMatrix.append(max_values)
        warpMatrix.append(warped[:, int(meanRadius-30):int(meanRadius+15)])
        offsetList.append(kk)
        
        # if PLOT:
        #     fig_I, axes_I = plt.subplots(1,2, figsize = (12, 6), sharey = True)
        #     ax = axes_I[0]
        #     ax.imshow(I0, cmap='gray')
        #     ax.plot(contour[:,1], contour[:,0], 'r--', lw=1, zorder = 4)
        #     ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
        #     theta = np.linspace(0, 2*np.pi, 100)
        #     CircleY, CircleX = Yc + meanRadius*np.sin(theta), Xc + meanRadius*np.cos(theta)
        #     # ax.plot(CircleX, CircleY, c='cyan', ls='--', lw=1)
            
        #     ax = axes_I[1]
        #     ax.imshow(I0, cmap='gray')
        #     ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
        #     CircleYin, CircleXin = Yc + (meanRadius-inPix)*np.sin(theta), Xc + (meanRadius-inPix)*np.cos(theta)
        #     ax.plot(CircleXin, CircleYin, c='gold', ls='--', lw=1)
        #     CircleYout, CircleXout = Yc + (meanRadius+outPix+1)*np.sin(theta), Xc + (meanRadius+outPix+1)*np.cos(theta)
        #     ax.plot(CircleXout, CircleYout, c='gold', ls='--', lw=1)
            
        #     fig_w, axes_w = plt.subplots(1,4, sharey = True)
        #     ax = axes_w[0]
        #     ax.imshow(warped, cmap='gray')
        #     ax.set_title('Raw warp')
            
        #     ax = axes_w[1]
        #     ax.imshow(warped, cmap='gray')
        #     ax.plot(warped_contour_int[:,1]-inPix, warped_contour_int[:,0], c='gold', ls='--', lw=1, zorder = 4)
        #     ax.plot(warped_contour_int[:,1]+outPix+1, warped_contour_int[:,0], c='gold', ls='--', lw=1, zorder = 4)
        #     ax.set_title('Segmentation')
            
        #     ax = axes_w[1]
        #     ax.plot(max_position_1, yy, c='cyan', ls='', marker ='.', ms=1)
        #     ax = axes_w[2]
        #     ax.plot(max_values, yy, 'r-')
        #     ax.set_title('Max')
        #     ax = axes_w[3]
        #     ax.plot(sum_values, yy, 'b-')
        #     ax.set_title('Sum')
            
            
        #     plt.tight_layout()
            
        #     if SAVE:
        #         ufun.simpleSaveFig(fig_I, f[:-4] + f'_I_{kk:.0f}', dstDir, '.png', 150)
        #         ufun.simpleSaveFig(fig_w, f[:-4] + f'_w_{kk:.0f}', dstDir, '.png', 150)
            
        plt.close('all')
    
    cList_viridis = plt.cm.viridis(np.linspace(0, 1, len(valsMatrix)))

    fig_w, axes_w = plt.subplots(len(valsMatrix), 2, sharex = True, figsize = (8, 16))
    fig_w.suptitle(f[:-4] + f'_ZScan')
    axes_w[0, 0].set_title('Warped cell')
    axes_w[0, 1].set_title('Max')
    for k in range(len(valsMatrix)):
        warped = warpMatrix[k]
        ny, nx = warped.shape
        yy = np.arange(0, ny)
        
        ax = axes_w[k, 0]
        ax.imshow(warped.T, cmap='viridis')
        offset_um = offsetList[k] * 0.25
        ax.set_ylabel(f'$+{offset_um:.2f}µm$')
        
        ax = axes_w[k, 1]
        ax.plot(yy, valsMatrix[k], c = cList_viridis[k])
        
    ufun.simpleSaveFig(fig_w, f[:-4] + f'_ZScan', dstDir, '.png', 150)
        
    fig_s, axes_s = plt.subplots(2, 1, sharex = True, figsize = (16, 6))
    fig_s.suptitle(f[:-4] + f'_ZSum')
    sumWarp = np.sum(warpMatrix, axis=0)
    ny, nx = sumWarp.shape
    yy = np.arange(0, ny)
    
    ax = axes_s[0]
    ax.set_title('Warped cell')
    ax.imshow(sumWarp.T, cmap='viridis')
    
    ax = axes_s[1]
    ax.set_title('Max')
    ax.set_prop_cycle(plt.cycler("color", cList_viridis))
    for k in range(len(valsMatrix)):
        offset_um = offsetList[k] * 0.25
        ax.plot(yy, valsMatrix[k], label = f'$+{offset_um:.2f}µm$')
        
    ax.legend()
        
    ufun.simpleSaveFig(fig_s, f[:-4] + f'_ZSum', dstDir, '.png', 150)

plt.show()

# %%% Test with cellPose

srcDir = "D://MicroscopyData//2023-12-06_3T3-LifeAct_JV//SingleFramesForTests"
# fileName = "3T3-LifeActGFP_TL_2min_4sec_400ms_10p-03-1.tif"
# fileName = "3T3-LifeActGFP_TL_2min_4sec_400ms_10p-02-1.tif"
fileName = "3T3-LifeActGFP_TL_2min_4sec_400ms_10p-01-1.tif"

filePath = os.path.join(srcDir, fileName)
I0 = skm.io.imread(filePath)

framesize = int(35*8/2) 

inPix, outPix = 20, 4



#### 1. Detect

fig, axes_I = plt.subplots(1,2, figsize = (12, 6), sharey = True)

model = models.CellposeModel(gpu=False, model_type='TN1')
channels = [0, 0]
[mask], [flow], [style] = model.eval([I0], diameter = 110, channels=channels,
                                    flow_threshold = 0.4, cellprob_threshold = 1.0, 
                                    do_3D=False, normalize = True) # masks, flows, styles, diams
Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)
Yc0, Xc0 = int(Yc), int(Xc)

I0 = I0[Yc0-framesize:Yc0+framesize, Xc0-framesize:Xc0+framesize]
[mask], [flow], [style] = model.eval([I0], diameter = 110, channels=channels,
                                    flow_threshold = 0.4, cellprob_threshold = 1.0, 
                                    do_3D=False, normalize = True) # masks, flows, styles, diams
Yc, Xc = ndi.center_of_mass(mask, labels=mask, index=1)

[contour] = skm.measure.find_contours(mask, 0.5) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
listDistances = np.sqrt((contour[:,0] - Yc)**2 + (contour[:,1] - Xc)**2)
meanRadius = np.mean(listDistances)

ax = axes_I[0]
ax.imshow(I0, cmap='gray')
ax.plot(contour[:,1], contour[:,0], 'r--', lw=1, zorder = 4)
ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
theta = np.linspace(0, 2*np.pi, 100)
CircleY, CircleX = Yc + meanRadius*np.sin(theta), Xc + meanRadius*np.cos(theta)
# ax.plot(CircleX, CircleY, c='cyan', ls='--', lw=1)

ax = axes_I[1]
ax.imshow(I0, cmap='gray')
ax.plot(Xc, Yc, marker = 'o', markersize = 5, color = 'gold')
CircleYin, CircleXin = Yc + (meanRadius-inPix)*np.sin(theta), Xc + (meanRadius-inPix)*np.cos(theta)
ax.plot(CircleXin, CircleYin, c='gold', ls='--', lw=1)
CircleYout, CircleXout = Yc + (meanRadius+outPix+1)*np.sin(theta), Xc + (meanRadius+outPix+1)*np.cos(theta)
ax.plot(CircleXout, CircleYout, c='gold', ls='--', lw=1)



#### 2. Warp

warped = skm.transform.warp_polar(I0, center=(Yc, Xc), radius=meanRadius*1.3, 
                                  output_shape=None, scaling='linear', channel_axis=None)
warped_mask = skm.transform.warp_polar(mask, center=(Yc, Xc), radius=meanRadius*1.3, 
                                  output_shape=None, scaling='linear', channel_axis=None)

[warped_contour] = skm.measure.find_contours(warped_mask) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]
mask_integer = [warped_contour[i, 0] == int(warped_contour[i, 0]) for i in range(len(warped_contour[:, 0]))]
warped_contour_int = warped_contour[mask_integer]
warped_contour_int = warped_contour_int.astype(int)

ny, nx = warped.shape
yy = np.arange(0, ny)
warped_cortex_region = np.zeros((ny, 1 + inPix + outPix), dtype = np.float64)

fig_w, axes_w = plt.subplots(1,3, sharey = True)
ax = axes_w[0]
ax.imshow(warped, cmap='gray')
# ax.axvline(meanRadius - 10, c='cyan', ls='--', lw=1)
# ax.axvline(meanRadius + 10, c='cyan', ls='--', lw=1)
# ax.plot(warped_contour_int[:,1], warped_contour_int[:,0], c='r', ls='--', zorder = 4)
ax.plot(warped_contour_int[:,1]-inPix, warped_contour_int[:,0], c='gold', ls='--', lw=1, zorder = 4)
ax.plot(warped_contour_int[:,1]+outPix+1, warped_contour_int[:,0], c='gold', ls='--', lw=1, zorder = 4)


for i in range(ny):
    warped_cortex_region[i, :] = warped[i, warped_contour_int[i,1]-inPix:warped_contour_int[i,1]+outPix+1]
interp_factor = 5
warped_cortex_region_interp = ufun.resize_2Dinterp(warped_cortex_region, fx=interp_factor, fy=1)
max_values = np.max(warped_cortex_region_interp, axis = 1)
sum_values = np.sum(warped_cortex_region_interp, axis = 1)
max_position_0 = np.argmax(warped_cortex_region_interp, axis = 1)
max_position_1 = (max_position_0/interp_factor) + warped_contour_int[:,1] - inPix
    
fig, axes_r = plt.subplots(1,3, sharey = True)
ax = axes_r[0]
ax.imshow(warped_cortex_region_interp, cmap='gray')
ax = axes_r[1]
ax.plot(max_values, yy, 'r-')
ax = axes_r[2]
ax.plot(sum_values, yy, 'b-')

ax = axes_w[0]
ax.plot(max_position_1, yy, c='cyan', ls='', marker ='.', ms=1)
ax = axes_w[1]
ax.plot(max_values, yy, 'r-')
ax = axes_w[2]
ax.plot(sum_values, yy, 'b-')

def autocorr(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
    return(acorr)

values = max_values
acorr_0 = autocorr(values)
values_12rep = np.array(list(values) * 12)

acorr_12rep = autocorr(values_12rep)
acorr_1 = acorr_12rep[3*ny:3*ny + ny//2]
    
fig, ax_c = plt.subplots(1,1)
ax = ax_c
ax.plot(np.arange(0, len(acorr_1)), acorr_1)
ax.plot(np.arange(0, len(acorr_0)), acorr_0)

plt.tight_layout()
plt.show()
