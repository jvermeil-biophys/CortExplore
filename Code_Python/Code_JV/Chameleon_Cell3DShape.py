# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:38:29 2024

@author: JosephVermeil
"""

# %% Imports

import os
import re
import cv2

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import UtilityFunctions as ufun
import GraphicStyles as gs

# from skimage import io, transform, filters, draw, exposure, measure
from PIL import Image
from PIL.TiffTags import TAGS

from mpl_toolkits.mplot3d import axes3d

import skimage as skm
# import cellpose as cpo
from cellpose import models
from scipy import interpolate, signal, optimize
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

def fitCircle(contour, loss = 'huber'):   # Contour = [[Y, X], [Y, X], [Y, X], ...] 
    x, y = contour[:,1], contour[:,0]
    x_m = np.mean(x)
    y_m = np.mean(y)
    
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return(((x-xc)**2 + (y-yc)**2)**0.5)


    def f_2(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return(Ri - np.mean(Ri))

    center_estimate = x_m, y_m
    result = optimize.least_squares(f_2, center_estimate, loss=loss)
    center = result.x
    R = np.mean(calc_R(*center))
    
    return(center, R)


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


def circularMask(shape, center, radius):
    ny, nx  = shape
    yc, xc = center
    x, y = np.arange(0, nx), np.arange(0, ny)
    mask = np.zeros((nx, ny), dtype=bool)
    pos = (x[np.newaxis,:]-xc)**2 + (y[:,np.newaxis]-yc)**2 < radius**2
    mask[pos] = True
    return(mask)

def distFromCenter(center, mask):
    yc, xc = center
    ny, nx = mask.shape
    x, y = np.arange(0, nx), np.arange(0, ny)
    xx, yy = np.meshgrid(x, y)
    D = np.sqrt(np.square(yy - yc) + np.square(xx - xc))
    D = D*mask
    return(D)


def findCellInnerCircle(I0, withBeads = True, plot=False):
    th1 = skm.filters.threshold_isodata(I0)
    I0_bin = (I0 > th1)
    I0_bin = ndi.binary_opening(I0_bin, iterations = 2)
    I0_label, num_features = ndi.label(I0_bin)
    df = pd.DataFrame(skm.measure.regionprops_table(I0_label, I0, properties = ['label', 'area']))
    df = df.sort_values(by='area', ascending=False)
    i_label = df.label.values[0]
    I0_rawCell = (I0_label == i_label)
    I0_rawCell = ndi.binary_fill_holes(I0_rawCell)
    
    if withBeads == True:
        I0_rawCell = skm.morphology.convex_hull_image(I0_rawCell)
    
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


def get_CZT_fromTiff(filePath):
    img = Image.open(filePath)
    meta_dict = {TAGS[key] : img.tag[key] for key in img.tag_v2}
    # md_str = str(meta_dict['ImageJMetaData'].decode('UTF-8'))
    md_str = str(meta_dict['ImageJMetaData'].decode('UTF-8'))[1:-3:2]
    md_array = np.array(md_str.split('#')).astype('<U100')
    channel_str = r'c:\d+/\d+'
    slice_str = r'z:\d+/\d+'
    time_str = r't:\d+/\d+'
    line0 = md_array[0]
    c_match = re.search(channel_str, line0)
    c_tot = int(line0[c_match.start():c_match.end():].split('/')[-1])
    z_match = re.search(slice_str, line0)
    z_tot = int(line0[z_match.start():z_match.end():].split('/')[-1])
    t_match = re.search(time_str, line0)
    t_tot = int(line0[t_match.start():t_match.end():].split('/')[-1])
    czt_shape = [c_tot, z_tot, t_tot]

    czt_seq = []
    for line in md_array:
        if len(line) > 12:
            c_match = re.search(channel_str, line)
            c_raw = line[c_match.start():c_match.end():]
            c = c_raw.split('/')[0].split(':')[-1]
            z_match = re.search(slice_str, line)
            z_raw = line[z_match.start():z_match.end():]
            z = z_raw.split('/')[0].split(':')[-1]
            t_match = re.search(time_str, line)
            t_raw = line[t_match.start():t_match.end():]
            t = t_raw.split('/')[0].split(':')[-1]
            czt_seq.append([int(c), int(z), int(t)])
            
    return(czt_shape, czt_seq)



# %% Cell 3D contour

srcDir = "D:/MicroscopyData/2024-02-27_SpinningDisc_3T3-LifeAct_wPincher_JV/ZS_Fluo-crop"
fileName = "3T3-LifeActGFP_ZS_C10"

dstDir = os.path.join(srcDir, fileName + '_results')
warpSavePath = os.path.join(srcDir, fileName + '_results', 'Warps')
contourSavePath = os.path.join(srcDir, fileName + '_results', 'Contours')

filePath = os.path.join(srcDir, fileName + '.tif')
cellId = fileName.split('_')[-1]

I = skm.io.imread(filePath)
nz, ny, nx = I.shape

sumArray = np.sum(I, axis=(1,2))
z0 = np.argmax(sumArray)

Zi, Zf = z0 + 13, z0 + 73
# Zi, Zf = z0 + 10, z0 + 16
PLOT_WARP = False
PLOT_CONTOUR = False


Zcontour = []
Xcontour = []
Ycontour = []
XYcenter = []

for z in range(Zi, Zf): #(16, 25):
    #### 1. Algo Settings
    framesize = int(35*8/2) 
    inPix, outPix = 30, 5
    blur_parm = 4
    relative_height_virebi = 0.6
    interp_factor_x = 5
    interp_factor_y = 10
    VEW = 7 # Viterbi edge width # Odd number
    
    I0 = I[z0+z]
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
    edge_viterbi = np.array(edge_viterbi[1-starting_i:] + edge_viterbi[:1-starting_i]) # Put it back in order
    viterbi_Npix_values = [np.sum(warped[Angles[i], edge_viterbi[i]-VEW//2:edge_viterbi[i] + (VEW//2 + 1)])/VEW for i in range(len(edge_viterbi))]
    
    # Contour - unwarped
    # warped = skm.transform.warp_polar(I0, center=(Yc, Xc), radius=approxRadius*1.3, 
    #                                   output_shape=None, scaling='linear', channel_axis=None)
    # Angles / edge_viterbi
    X_cell = (edge_viterbi/interp_factor_x) * np.cos(Angles*np.pi/180) + Xc
    Y_cell = (edge_viterbi/interp_factor_x) * np.sin(Angles*np.pi/180) + Yc
    contour_viterbi = np.array([X_cell, Y_cell]).T

    Zcontour.append(z)
    Xcontour.append(X_cell)
    Ycontour.append(Y_cell)
    XYcenter.append(np.array([np.mean(X_cell), np.mean(Y_cell)]))
    
    # intensityMap.append(viterbi_Npix_values)
    
    #### 5. Plot
    if PLOT_WARP:
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
        
        figV.suptitle(f'Warp - {cellId} - z = {z+1:.0f}/{nz:.0f}')
        figName = f'{cellId}_warp_z{z+1:.0f}'
        ufun.simpleSaveFig(figV, figName, warpSavePath, '.png', 150)
        
        
    if PLOT_CONTOUR:
        figC, axesC = plt.subplots(1,2, figsize = (8,4))
        ax = axesC[0]
        ax.imshow(I0, cmap='gray')
        ax = axesC[1]
        ax.imshow(I0, cmap='gray')
        ax.plot(Xc, Yc, 'ro')
        ax.plot(X_cell, Y_cell, 'r-', lw=1)
        
        for ax in axesC:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
        figC.suptitle(f'Contour - {cellId} - z = {z+1:.0f}/{nz:.0f}')
        figC.tight_layout()
        
        figName = f'{cellId}_contour_z{z+1:.0f}'
        ufun.simpleSaveFig(figC, figName, contourSavePath, '.png', 150)
        
        plt.show()

# intensityMap = np.array(intensityMap)
# np.savetxt(os.path.join(dstDir, f'{cellCode}_MapIntensity.txt'), intensityMap)
# fig, ax = plt.subplots(1, 1, figsize = (12, 3))
# im = ax.imshow(intensityMap, cmap = 'viridis', vmin=0)
# fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1, aspect=80)
# fig.tight_layout()
# ufun.archiveFig(fig, name = f'{cellCode}_MapIntensity_Z{Zi:.0f}to{Zf:.0f}', ext = '.png', dpi = 100,
#                figDir = dstDir, cloudSave = 'none')
# plt.show()

plt.close('all')

XY_scale = 7.4588
Z_scale = 4/0.875

Zcontour = np.array(Zcontour)
Xcontour = np.array(Xcontour)
Ycontour = np.array(Ycontour)
XYcenter = np.array(XYcenter)

XYZcenter = np.array([[XYcenter[i, 0], XYcenter[i, 1], Zcontour[i]] for i in range(len(Zcontour))])

np.savetxt(os.path.join(dstDir, f'{cellId}_Zcontour.txt'), Zcontour)
np.savetxt(os.path.join(dstDir, f'{cellId}_Xcontour.txt'), Xcontour)
np.savetxt(os.path.join(dstDir, f'{cellId}_Ycontour.txt'), Ycontour)
np.savetxt(os.path.join(dstDir, f'{cellId}_XYZcenter.txt'), XYZcenter)

cList_viridis = plt.cm.viridis(np.linspace(0.1, 0.9, len(Zcontour)))

fig3d = plt.figure()
ax3d = fig3d.add_subplot(projection='3d')
ax = ax3d
ax.set_prop_cycle(plt.cycler("color", cList_viridis))

for i in range(len(Zcontour)):
    xx, yy = Xcontour[i]/XY_scale, Ycontour[i]/XY_scale
    z = Zcontour[i]/Z_scale
    zz = z * np.ones_like(xx)
    ax.plot(xx, yy, zz, label=f'z = {z/Z_scale:.1f}µm')
    
ax.plot(XYZcenter[:,0]/XY_scale, XYZcenter[:,1]/XY_scale, XYZcenter[:,2]/Z_scale, 'r-')
    
ax.set_aspect('equal', adjustable='box')    
# ax.legend(fontsize = 6)

figName = f'{cellId}_3D'
ufun.simpleSaveFig(fig3d, figName, dstDir, '.png', 150)

plt.show()


    
# %% Fit ellipsoid

srcDir = "D:/MicroscopyData/2024-02-27_SpinningDisc_3T3-LifeAct_wPincher_JV/ZS_Fluo-crop"
fileName = "3T3-LifeActGFP_ZS_C10"

dstDir = os.path.join(srcDir, fileName + '_results')
filePath = os.path.join(srcDir, fileName + '.tif')
cellId = fileName.split('_')[-1]

Zcontour = np.loadtxt(os.path.join(dstDir, f'{cellId}_Zcontour.txt'))
Xcontour = np.loadtxt(os.path.join(dstDir, f'{cellId}_Xcontour.txt'))
Ycontour = np.loadtxt(os.path.join(dstDir, f'{cellId}_Ycontour.txt'))
XYZcenter = np.loadtxt(os.path.join(dstDir, f'{cellId}_XYZcenter.txt'))
XYcenter_avg = np.mean(XYZcenter[10:-10, :2], axis=0)

XX = (Xcontour-XYcenter_avg[0])/XY_scale
YY = (Ycontour-XYcenter_avg[1])/XY_scale
Z = Zcontour/Z_scale
ZZ = np.repeat(Z, XX.shape[1], axis=0).reshape(XX.shape)

D2 = np.square(XX) + np.square(YY)

d2 = D2.flatten()
zz = ZZ.flatten()

def ellipsoid1(d2, R, C, g):
    return(np.sqrt(C**2 - (C**2/R**2)*d2) + g)

def ellipsoid2(zz, R, C, g):
    return(R**2 * (1 - (np.square(zz-g))/C**2))

def residuals_ellipsoid(p, zz, d2):
    return(d2 - ellipsoid2(zz, *p))


p0 = [10, 12, 0]
popt, pcov = optimize.leastsq(residuals_ellipsoid, p0, args=(zz, d2))

[R, C, g] = popt
print(f'R = {R:.2f}, C = {C:.2f}, g = {g:.2f} [µm]')
d2_fit = ellipsoid2(zz, R, C, g)
D2_fit = d2_fit.reshape(D2.shape)

AA = np.repeat(np.linspace(0, 2*np.pi, 360), XX.shape[0], axis=0).reshape(XX.T.shape).T
XX_fit = np.sqrt(D2_fit) * np.cos(AA)
YY_fit = np.sqrt(D2_fit) * np.sin(AA)

cList_viridis = plt.cm.viridis(np.linspace(0.1, 0.9, len(Zcontour)))
fig3D, axes3D = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(15,5))

# Real points
ax = axes3D[0]
ax.set_prop_cycle(plt.cycler("color", cList_viridis))
for i in range(len(Zcontour)):
    xx, yy = (Xcontour[i])/XY_scale, (Ycontour[i])/XY_scale
    z = Zcontour[i]/Z_scale
    zz = z * np.ones_like(xx)
    ax.plot(xx, yy, zz, label=f'z = {z/Z_scale:.1f}µm')
ax.plot(XYZcenter[:,0]/XY_scale, XYZcenter[:,1]/XY_scale, XYZcenter[:,2]/Z_scale, 'r-')
ax.set_aspect('equal', adjustable='box')    

# Fitted Ellipsoid
ax = axes3D[1]
ax.plot_wireframe(XX_fit, YY_fit, ZZ, rstride=10, cstride=10, color='r', lw=1)
ax.set_aspect('equal', adjustable='box')

# Both
ax = axes3D[2]
ax.plot_wireframe(XX_fit, YY_fit, ZZ, rstride=10, cstride=10, color='r', lw=1)
ax.set_prop_cycle(plt.cycler("color", cList_viridis))
for i in range(len(Zcontour)):
    xx, yy = (Xcontour[i]-XYcenter_avg[0])/XY_scale, (Ycontour[i]-XYcenter_avg[1])/XY_scale
    z = Zcontour[i]/Z_scale
    zz = z * np.ones_like(xx)
    ax.plot(xx, yy, zz, label=f'z = {z/Z_scale:.1f}µm')
ax.set_aspect('equal', adjustable='box')

fig3D.suptitle(f'3D view - {cellId}\nR = {R:.2f}, C = {C:.2f}, $\\gamma$ = {g:.2f} [µm]')
fig3D.tight_layout()

figName = f'{cellId}_3D'
ufun.simpleSaveFig(fig3D, figName, dstDir, '.png', 150)

plt.show()


# ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')


# %% Fit ellipsoid manuscript
gs.set_manuscript_options_jv()
srcDir = "D:/MicroscopyData/2024-02-27_SpinningDisc_3T3-LifeAct_wPincher_JV/ZS_Fluo-crop"

fileName = "3T3-LifeActGFP_ZS_C10"

dstDir = os.path.join(srcDir, fileName + '_results')
filePath = os.path.join(srcDir, fileName + '.tif')
cellId = fileName.split('_')[-1]

Zcontour = np.loadtxt(os.path.join(dstDir, f'{cellId}_Zcontour.txt'))
Xcontour = np.loadtxt(os.path.join(dstDir, f'{cellId}_Xcontour.txt'))
Ycontour = np.loadtxt(os.path.join(dstDir, f'{cellId}_Ycontour.txt'))
XYZcenter = np.loadtxt(os.path.join(dstDir, f'{cellId}_XYZcenter.txt'))
XYcenter_avg = np.mean(XYZcenter[10:-10, :2], axis=0)

XX = (Xcontour-XYcenter_avg[0])/XY_scale
YY = (Ycontour-XYcenter_avg[1])/XY_scale
Z = Zcontour/Z_scale
ZZ = np.repeat(Z, XX.shape[1], axis=0).reshape(XX.shape)

D2 = np.square(XX) + np.square(YY)

d2 = D2.flatten()
zz = ZZ.flatten()

def ellipsoid1(d2, R, C, g):
    return(np.sqrt(C**2 - (C**2/R**2)*d2) + g)

def ellipsoid2(zz, R, C, g):
    return(R**2 * (1 - (np.square(zz-g))/C**2))

def residuals_ellipsoid(p, zz, d2):
    return(d2 - ellipsoid2(zz, *p))


p0 = [10, 12, 0]
popt, pcov = optimize.leastsq(residuals_ellipsoid, p0, args=(zz, d2))

[R, C, g] = popt
print(f'R = {R:.2f}, C = {C:.2f}, g = {g:.2f} [µm]')
d2_fit = ellipsoid2(zz, R, C, g)
D2_fit = d2_fit.reshape(D2.shape)

AA = np.repeat(np.linspace(0, 2*np.pi, 360), XX.shape[0], axis=0).reshape(XX.T.shape).T
XX_fit = np.sqrt(D2_fit) * np.cos(AA)
YY_fit = np.sqrt(D2_fit) * np.sin(AA)

cList_viridis = plt.cm.viridis(np.linspace(0.1, 0.9, len(Zcontour)))
fig3D, axes3D = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(6/gs.cm_in,5/gs.cm_in))

# Real points
ax = axes3D
ax.set_prop_cycle(plt.cycler("color", cList_viridis))
for i in range(len(Zcontour)):
    xx, yy = (Xcontour[i])/XY_scale, (Ycontour[i])/XY_scale
    z = Zcontour[i]/Z_scale
    zz = z * np.ones_like(xx)
    ax.plot(xx, yy, zz, label=f'z = {z/Z_scale:.1f}µm', lw=0.75)
# ax.plot(XYZcenter[:,0]/XY_scale, XYZcenter[:,1]/XY_scale, XYZcenter[:,2]/Z_scale, 'r-')
ax.set_aspect('equal', adjustable='box')    


figName = f'{cellId}_3D'
fig3D.tight_layout()


ufun.simpleSaveFig(fig3D, figName, dstDir, '.pdf', 150)
plt.show()


# ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')



# %% Flatten the ellipse

XYcenter_avg = np.mean(XYZcenter[10:-10, :2], axis=0)

XX = (Xcontour-XYcenter_avg[0])/XY_scale
YY = (Ycontour-XYcenter_avg[1])/XY_scale
ZZ = Zcontour/Z_scale

cList_viridis = plt.cm.viridis(np.linspace(0.1, 0.9, 180))
figE, axE = plt.subplots(1, 1, figsize=(12,12))
ax = axE
ax.set_prop_cycle(plt.cycler("color", cList_viridis))

zp = np.concatenate((ZZ, ZZ))
m_dp = []

for i in range(180):
    x1 = XX[:,i]
    x2 = XX[:,i+180]

    y1 = YY[:,i]
    y2 = YY[:,i+180]

    d1 = np.sqrt(np.square(x1) + np.square(y1))
    d2 = - np.sqrt(np.square(x2) + np.square(y2))
    dp = np.concatenate((d1, d2))
    m_dp.append(dp)
    
    ax.plot(dp, zp, ls='', marker = '.', ms=1)
    
m_dp = np.array(m_dp)
median_dp = np.median(m_dp, axis=0)

ax.plot(median_dp, zp, 'ro', ms=3)   

ax.grid()
ax.set_aspect('equal', adjustable='box')
plt.show()