# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:07:53 2024

@author: JosephVermeil
"""

# %% Imports

import os

import numpy as np
import matplotlib.pyplot as plt
import skimage as skm
import scipy.ndimage as ndi

# %% Imports 2

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
SCALE_100X = 15.8

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



# %% Try on an example

src_dir = "D://MagneticPincherData//Raw//24.02.26_NanoIndent_ChiaroData_25um//M3_Films//Crops"
src_file = "C1-1.tif"
src_path = os.path.join(src_dir, src_file)

Iraw = skm.io.imread(src_path)

fig1, ax1 = plt.subplots(1, 1, figsize = (8, 8))
ax = ax1
ax.imshow(Iraw[0], cmap='Greys_r')



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
        