# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:28:43 2024

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
from matplotlib.patches import Rectangle

# from cellpose.io import imread

import shapely
from shapely.ops import polylabel
# from shapely import Polygon, LineString, get_exterior_ring, distance
from shapely.plotting import plot_polygon, plot_points, plot_line

from mpl_toolkits.axes_grid1 import make_axes_locatable


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
        d = (np.abs(node1['pos'] - node2['pos']))**2 #+ np.abs(node1['val'] - node2['val'])
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
    # mask =  I0_rawCell
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



# %% Cell Mappemonde


srcDir = "D:/MicroscopyData/2023-12-06_SpinningDisc_3T3-LifeAct_JV/"
fileRoot = "3T3-LifeActGFP_ZS_250nm_wholeCell_400ms_10p"
fileList = os.listdir(srcDir)
stackList = []

dstDir = "D:/MicroscopyData/2023-12-06_SpinningDisc_3T3-LifeAct_JV/Session2_Test1"

radiusList_allCells = []

for f in fileList:
    if '_ZS_' in f:
        stackList.append(f)

for f in stackList[2:3]:
    filePath = os.path.join(srcDir, f)
    cellCode = 'C' + (f[:-4])[-1:]
    # saveDirPath = os.path.join(srcDir, cellCode)
    # if not os.path.exists(saveDirPath):
    #     os.mkdir(saveDirPath)
    
    
    I = skm.io.imread(filePath)
    sumArray = np.sum(I, axis=(1,2))
    z0 = np.argmax(sumArray)
    
    # acorrMatrix = []
    # radiusList = []
    intensityMap = []
    
    Zi, Zf = 16, 26
    
    for kk in range(Zi, Zf): #(16, 25):
        #### 1. Algo Settings
        framesize = int(35*8/2) 
        inPix, outPix = 20, 5
        blur_parm = 4
        relative_height_virebi = 0.3
        interp_factor_x = 5
        interp_factor_y = 10
        VEW = 7 # Viterbi edge width # Odd number
        
        I0 = I[z0+kk]
        (Yc, Xc), approxRadius = findCellInnerCircle(I0, plot=False)
        Yc0, Xc0, R0 = round(Yc), round(Xc), round(approxRadius)
        
        #### 2.1 Warp
        warped = skm.transform.warp_polar(I0, center=(Yc, Xc), radius=approxRadius*1.3, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        # warped_mask = skm.transform.warp_polar(mask, center=(Yc, Xc), radius=approxRadius*1.3, 
        #                                   output_shape=None, scaling='linear', channel_axis=None)
        
        # [warped_contour] = skm.measure.find_contours(warped_mask) # contour = [(r1, c1), (r2, c2), ..., (rn, cn)]        
        # mask_integer = [warped_contour[i, 0] == int(warped_contour[i, 0]) for i in range(len(warped_contour[:, 0]))]
        # warped_contour_int = warped_contour[mask_integer]
        # warped_contour_int = warped_contour_int[:,1].astype(int)
        
        # warped_contour = [(r, R0) for r in range(warped.shape[0])]
        warped_contour_int = np.array([R0 for r in range(warped.shape[0])])
        
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
        
        intensityMap.append(viterbi_Npix_values)
        
        #### 5. Plot
        PLOT = 1
        if PLOT:
            #### 5.2 Warp and Viterbi
            step = 5
            cList_viridis = plt.cm.viridis(np.linspace(0.05, 0.95, w_ny//step))
            
            figV, axV = plt.subplots(1, 4, sharey=True, figsize = (9,8))
            ax = axV[0]
            ax.imshow(warped[:, :], cmap='gray', aspect='auto')
            ax.set_title('Raw warp')
            ax.set_ylabel('Angle (deg)')
            ax.set_xlabel('Radial pix')
            
            ax = axV[1]
            ax.imshow(warped[:, :], cmap='gray', aspect='auto')
            ax.plot(warped_contour_int, Angles, ls='--', c='red', lw=1)
            ax.plot(warped_contour_int - inPix, Angles, ls='--', c='gold', lw=1)
            ax.plot(warped_contour_int + outPix, Angles, ls='--', c='gold', lw=1)
            ax.plot(edge_viterbi, Angles, ls='', c='cyan', marker='.', ms = 2)
            ax.set_title('Viterbi Edge')
            ax.set_xlabel('Radial pix')
        
            ax = axV[2]
            ax.set_prop_cycle(plt.cycler("color", cList_viridis))
            ax.set_title('Viterbi profiles')
            ax.set_xlabel('Radial pix')
            
            ax = axV[3]
            ax.set_title('Profile Values')
            ax.plot(max_values, Angles, 'r-', label = 'Max profile')
            ax.plot(viterbi_Npix_values, Angles, 'b-', label = 'Viterbi profile')
            ax.legend()
        
            maxWarped = np.max(warped)
            normWarped = warped * 6 * step/maxWarped
            for a in Angles[::step]:            
                profile = normWarped[a, :] - np.min(normWarped[a, :])
                RR = np.arange(len(profile))
                ax = axV[2]
                ax.plot(RR, a - profile)
                ax.plot([inBorder], [a], c='k', ls='', marker='|', ms = 6)
                ax.plot([outBorder], [a], c='k', ls='', marker='|', ms = 6)
                # P = AllPeaks[a]
                # P = np.array(P)
                # A = a * np.ones(len(P))
                # ax.plot(P, A, ls='', c='orange', marker='o', ms = 4, mec = 'k') #  t - profile[inPix + peaks[k]]
                node = nodes_list[a]
                ax.plot([node['pos']], [node['angle']], ls='', c='cyan', marker='^', ms = 4, mec = 'k', zorder=5) #  t - profile[inPix + peaks[k]]

            ufun.archiveFig(figV, name = f'{cellCode}_ViterbiEdge_Z{kk:.0f}', ext = '.png', dpi = 100,
                           figDir = dstDir, cloudSave = 'none')
            
    
    intensityMap = np.array(intensityMap)
    np.savetxt(os.path.join(dstDir, f'{cellCode}_MapIntensity.txt'), intensityMap)
    fig, ax = plt.subplots(1, 1, figsize = (12, 3))
    im = ax.imshow(intensityMap, cmap = 'viridis', vmin=0)
    fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1, aspect=80)
    fig.tight_layout()
    ufun.archiveFig(fig, name = f'{cellCode}_MapIntensity_Z{Zi:.0f}to{Zf:.0f}', ext = '.png', dpi = 100,
                   figDir = dstDir, cloudSave = 'none')
    plt.show()
    

# %% Pixel Guesser

def pixelGuesser1D(mapCell, pos, marginA, method='linear'):
    Z0, A0 = pos
    mapCell = mapCell[:Z0+1,:]
    nz, na = mapCell.shape
    # TBD


def pixelGuesser2D(mapCell, pos, marginZ, marginA, method='linear',
                   plot = False):
    Z0, A00 = pos
    nZ, nA = mapCell.shape

    if (A00 < (4*marginA)) or (nA-A00 < (4*marginA)):
        mapCell = np.roll(mapCell, 7*marginA, axis=1)
        A0 = (A00+7*marginA)%nA
        print(A0)
    else:
        A0 = A00
        
    mask = np.zeros_like(mapCell, dtype=bool)
    mask[Z0-marginZ:Z0+1, A0-marginA:A0+marginA+1] = True
    mapCell_cut = mapCell[:Z0+1,:]
    mask_cut = mask[:Z0+1,:]
    nZ_cut = mapCell_cut.shape[0]

    aa, zz = np.meshgrid(np.arange(nA), np.arange(nZ_cut))

    known_z = zz[~mask_cut]
    known_a = aa[~mask_cut]
    known_v = mapCell_cut[~mask_cut]
    missing_z = zz[mask_cut]
    missing_a = aa[mask_cut]
    
    known_points = np.array([known_z, known_a]).T
    missing_points = np.array([missing_z, missing_a]).T
    
    try:
        interp_values = interpolate.griddata(
            known_points, known_v, missing_points,
            method=method
        )
    
        interp_values_reshaped = np.reshape(interp_values, (marginZ+1, 2*marginA+1))
        value_pos = interp_values_reshaped[marginZ, marginA]
    except:
        print(Z0, A0, A00)
    
    if plot:
        value_real = mapCell[Z0, A0]
        rel_error = 100 * (value_pos - value_real) / value_real

        mask = np.zeros_like(mapCell, dtype=bool)
        mask[Z0-marginZ:Z0+1, A0-marginA:A0+marginA+1] = True

        mapCell2 = np.copy(mapCell)
        np.place(mapCell2, mask, interp_values_reshaped.flatten())

        mini_map = mapCell[Z0-marginZ-1:Z0+1, A0-3*marginA:A0+3*marginA+1]*100
        mini_map2 = mapCell2[Z0-marginZ-1:Z0+1, A0-3*marginA:A0+3*marginA+1]*100
        newZ0, newA0 = mini_map.shape[0], mini_map.shape[1] // 2

        fig, axes = plt.subplots(1, 3, figsize = (16, 2))
        ax = axes[0]
        ax.imshow(mini_map)
        ax.set_title('real')
        ax.plot([3*marginA], [1+marginZ], 'r+')
        ax.add_patch(Rectangle((newA0 - marginA - 0.5, newZ0 - marginZ - 1.5), 
                               2*marginA + 1, 1*marginZ + 1, 
                               edgecolor='red', facecolor='none', lw=1))

        ax = axes[1]
        ax.imshow(mini_map2)
        ax.set_title('predicted')
        ax.plot([3*marginA], [1+marginZ], 'r+')
        ax.add_patch(Rectangle((newA0 - marginA - 0.5, newZ0 - marginZ - 1.5), 
                               2*marginA + 1, 1*marginZ + 1, 
                               edgecolor='red', facecolor='none', lw=1))

        ax = axes[2]
        percent_map = 100*(mini_map2 - mini_map)/mini_map
        # abs_max = np.max(np.abs(percent_map))
        abs_max = 20
        im = ax.imshow(percent_map, vmin = -abs_max, vmax = +abs_max, cmap = 'RdBu_r')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(im, cax=cax, orientation='vertical') #, fraction=.03, aspect=20)
        ax.set_title('rel error (%)')
        ax.plot([3*marginA], [1+marginZ], 'k+')

        fig.suptitle(f'Error at target point = {rel_error:.2f} %')
        fig.tight_layout()

        plt.show()
    
    return(value_pos, interp_values_reshaped)
    
    
# %% Test 1

dataDir = "D:/MicroscopyData/2023-12-06_SpinningDisc_3T3-LifeAct_JV/Session2_Test1"
cellCode = 'C3'
mapPath = os.path.join(dstDir, f'{cellCode}_MapIntensity.txt')
mapCell = np.loadtxt(os.path.join(dstDir, f'{cellCode}_MapIntensity.txt'))
    

Z0, A0 = (6, 349)
pos = (Z0, A0)
marginZ, marginA = 2, 3
value_interp, iv = pixelGuesser2D(mapCell, pos, marginZ, marginA, method='cubic',
                                  plot=True)


# %% Test 2

dataDir = "D:/MicroscopyData/2023-12-06_SpinningDisc_3T3-LifeAct_JV/Session2_Test1"
cellCode = 'C3'
mapPath = os.path.join(dstDir, f'{cellCode}_MapIntensity.txt')
mapCell = np.loadtxt(os.path.join(dstDir, f'{cellCode}_MapIntensity.txt'))

marginZ, marginA = 2, 3

nZ, nA = mapCell.shape
mapCell_pred = np.zeros((nZ, nA))

for iZ in range(2, nZ):
    for iA in range(nA):
        pos = (iZ, iA)
        value_interp, iv = pixelGuesser2D(mapCell, pos, marginZ, marginA, method='cubic',
                                          plot=False)
        mapCell_pred[iZ, iA] = value_interp
        
fig, axes = plt.subplots(1, 3, figsize = (16, 2))
ax = axes[0]
ax.imshow(mapCell)
ax.set_title('real')

ax = axes[1]
ax.imshow(mapCell_pred)
ax.set_title('predicted')

ax = axes[2]
percent_map = 100*(mapCell_pred - mapCell)/mapCell
# abs_max = np.max(np.abs(percent_map))
abs_max = 30
im = ax.imshow(percent_map, vmin = -abs_max, vmax = +abs_max, cmap = 'RdBu_r')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical') #, fraction=.03, aspect=20)
ax.set_title('rel error (%)')

fig.tight_layout()

plt.show()

# %% 
A = np.array([[1,2,3], [4,5,6]])
M = np.array([[1,0,0], [1,0,0]]).astype(bool)
B = np.array([10, 40])

C = np.roll(A, 2, axis=1)

# %%

import numpy as np
def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2


grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]


rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:,0], points[:,1])


from scipy.interpolate import griddata
grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

import matplotlib.pyplot as plt
plt.subplot(221)
plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
plt.plot(points[:,0], points[:,1], 'k.', ms=1)
plt.title('Original')
plt.subplot(222)
plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
plt.title('Nearest')
plt.subplot(223)
plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
plt.title('Linear')
plt.subplot(224)
plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
plt.title('Cubic')
plt.gcf().set_size_inches(6, 6)
plt.show()
    
# %%

from scipy import interpolate
import numpy as np

def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )
    
    
# %% Tests

print(os.path.join(dstDir, f'{cellCode}_MapIntensity.txt'))

AA= np.loadtxt(os.path.join(dstDir, f'{cellCode}_MapIntensity.txt'))

intensityMap_interp = ufun.resize_2Dinterp(intensityMap, fx=1, fy=1.363636) #
fig, ax = plt.subplots(1, 1, figsize = (12, 3))
im = ax.imshow(intensityMap_interp, cmap = 'viridis', vmin=0)
fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1, aspect=80)
fig.tight_layout()
ufun.archiveFig(fig, name = f'{cellCode}_MapIntensity_Z{Zi:.0f}to{Zf:.0f}_resized', ext = '.png', dpi = 100,
               figDir = dstDir, cloudSave = 'none')
plt.show()