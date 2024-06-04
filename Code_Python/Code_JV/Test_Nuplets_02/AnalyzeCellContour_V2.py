# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:28:43 2024

@author: JosephVermeil
"""

# %% Imports

import os
import re

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import UtilityFunctions as ufun
import GraphicStyles as gs
import matplotlib.path as mpltPath
import seaborn as sns

# from skimage import io, transform, filters, draw, exposure, measure
from PIL import Image
from PIL.TiffTags import TAGS

import skimage as skm
# import cellpose as cpo
from cellpose import models
from scipy import interpolate, signal, optimize
from matplotlib.patches import Rectangle

from scipy.interpolate import Akima1DInterpolator # CubicSpline, PchipInterpolator

# from cellpose.io import imread

import shapely
from shapely.ops import polylabel
# from shapely import Polygon, LineString, get_exterior_ring, distance
from shapely.plotting import plot_polygon, plot_points, plot_line

from mpl_toolkits.axes_grid1 import make_axes_locatable

from BeadTracker_Fluo import smallTracker

gs.set_mediumText_options_jv()

SCALE_100X_ZEN = 7.4588

# %% Utility functions


# %%% For image processing

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

def warpXY(X, Y, Xcw, Ycw):
    """
    X, Y is the point of interest
    Xcw, Ycw are the coordinates of the center of the warp
    skimage.transform.warp_polar
    """
    R = ((Y-Ycw)**2 + (X-Xcw)**2)**0.5
    angle = (np.arccos((X-Xcw)/R * np.sign(Y-Ycw+0.001)) + np.pi*((Y-Ycw)<0)) * 180/np.pi  # Radian -> Degree
    return(R, angle)

def unwarpRA(R, A, Xcw, Ycw):
    """
    R, A is the point of interest
    Xcw, Ycw are the coordinates of the center of the warp
    skimage.transform.warp_polar
    """
    X = Xcw + (R*np.cos(A*np.pi/180)) # Degree -> Radian
    Y = Ycw + (R*np.sin(A*np.pi/180)) # Degree -> Radian
    return(X, Y)
    
def circleContour(center, radius, shape):
    rr, cc = skm.draw.disk(center, radius, shape=shape)
    Itmp = np.zeros(shape)
    Itmp[rr, cc] = 1
    [circle] = skm.measure.find_contours(Itmp, 0.5)
    return(circle)

def circleContour_V2(center, radius):
    aa = np.linspace(0, 2*np.pi, 120)
    xx, yy = radius*np.cos(aa)+center[1], radius*np.sin(aa)+center[0]
    xx, yy = np.round(xx,1), np.round(yy,1)
    # xx, yy = np.unique(xx, axis=0), np.unique(yy, axis=0)
    circle = np.array([yy, xx]).T
    return(circle)


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



def ViterbiEdge(warped, Rc, inPix, outPix, blur_parm, relative_height_virebi):
    # Create the TreillisGraph for Viterbi tracking
    Angles = np.arange(0, 360)
    AllPeaks = []
    TreillisGraph = []
    warped_filtered = skm.filters.gaussian(warped, sigma=(1, 2), mode='wrap')
    inBorder = round(Rc) - inPix
    outBorder = round(Rc) + outPix
    for a in Angles:
        profile = warped_filtered[a, :] - np.min(warped_filtered[a, inBorder:outBorder])
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
    try:
        starting_candidates = []
        for R in TreillisGraph:
            if len(R) == 1:
                starting_candidates.append(R[0])
        # pos_start = np.median([p['pos'] for p in starting_candidates])
        # starting_i = argmedian([p['pos'] for p in starting_candidates])
        # starting_peak = starting_candidates[starting_i]
        starting_peak = starting_candidates[argmedian([p['pos'] for p in starting_candidates])]
        starting_i = starting_peak['angle']
        
    except:
        starting_candidates = []
        for R in TreillisGraph:
            if len(R) <= 3:
                R_pos = [np.abs(p['pos']-Rc) for p in R]
                i_best = np.argmin(R_pos)
                starting_candidates.append(R[i_best])
        starting_peak = starting_candidates[argmedian([p['pos'] for p in starting_candidates])]
        starting_i = starting_peak['angle']
    
    # Pretreatment of the TreillisGraph for Viterbi tracking
    TreillisGraph = TreillisGraph[starting_i:] + TreillisGraph[:starting_i] # Set the starting point
    TreillisGraph.append([node.copy() for node in TreillisGraph[0]]) # Make the graph cyclical
    
    # Viterbi tracking
    best_path, nodes_list = ViterbiPathFinder(TreillisGraph)
    
    edge_viterbi = [p['pos'] for p in nodes_list[:-1]]
    edge_viterbi = edge_viterbi[1-starting_i:] + edge_viterbi[:1-starting_i] # Put it back in order
    return(edge_viterbi)

# %%% For smallTracker

def makeMetaData(T_raw, B_set, loopStructure):
    # Columns from the field file
    metaDf = pd.DataFrame({'T_raw':T_raw, 'B_set':B_set})
    
    # Columns from the loopStructure
    N_Frames_total = metaDf.shape[0]
    N_Frames_perLoop = np.sum([phase['NFrames'] for phase in loopStructure])
    consistency_test = ((N_Frames_total/N_Frames_perLoop) == (N_Frames_total//N_Frames_perLoop))
    
    if not consistency_test:
        print('Error in the loop structure: the length of described loop do not match the Field file data!')
    
    else:
        N_Loops = (N_Frames_total//N_Frames_perLoop)
        StatusCol = []
        for phase in loopStructure:
            Status = phase['Status']
            NFrames = phase['NFrames']
            L = [Status] * NFrames # Repeat
            StatusCol += L # Append
        StatusCol *= N_Loops # Repeat
        StatusCol = np.array(StatusCol)
        
        iLCol = (np.ones((N_Frames_perLoop, N_Loops)) * np.arange(1, N_Loops+1)).flatten(order='F')
        
        metaDf['iL'] = iLCol.astype(int)
        metaDf['Status'] = StatusCol
    
    return(metaDf)


def makeMetaData_field(fieldPath, loopStructure):
    # Columns from the field file
    fieldDf = pd.read_csv(fieldPath, sep='\t', names=['B_meas', 'T_raw', 'B_set', 'Z_piezo'])
    metaDf = fieldDf[['T_raw', 'B_set']]
    
    # Columns from the loopStructure
    N_Frames_total = metaDf.shape[0]
    N_Frames_perLoop = np.sum([phase['NFrames'] for phase in loopStructure])
    consistency_test = ((N_Frames_total/N_Frames_perLoop) == (N_Frames_total//N_Frames_perLoop))
    
    if not consistency_test:
        print(os.path.split(fieldPath)[-1])
        print('Error in the loop structure: the length of described loop do not match the Field file data!')
    
    else:
        N_Loops = (N_Frames_total//N_Frames_perLoop)
        StatusCol = []
        for phase in loopStructure:
            Status = phase['Status']
            NFrames = phase['NFrames']
            L = [Status] * NFrames # Repeat
            StatusCol += L # Append
        StatusCol *= N_Loops # Repeat
        StatusCol = np.array(StatusCol)
        
        iLCol = (np.ones((N_Frames_perLoop, N_Loops)) * np.arange(1, N_Loops+1)).flatten(order='F')
        
        metaDf['iL'] = iLCol.astype(int)
        metaDf['Status'] = StatusCol
    
    return(metaDf)


def makeMetaData_fieldAndStatus(fieldPath, statusPath):
    # Columns from the field file
    fieldDf = pd.read_csv(fieldPath, sep='\t', names=['B_meas', 'T_raw', 'B_set', 'Z_piezo'])
    metaDf = fieldDf[['T_raw', 'B_set']]
    
    # Format the status file
    statusDf = pd.read_csv(statusPath, sep='_', names=['iL', 'Status', 'Status details'])
    Ns = len(statusDf)
    
    statusDf['Action type'] = np.array(['' for i in range(Ns)], dtype = '<U16')
    statusDf['deltaB'] = np.zeros(Ns, dtype = float)
    statusDf['B_diff'] = np.array(['' for i in range(Ns)], dtype = '<U4')
    
    indexAction = statusDf[statusDf['Status'] == 'Action'].index
    Bstart = statusDf.loc[indexAction, 'Status details'].apply(lambda x : float(x.split('-')[1]))
    Bstop = statusDf.loc[indexAction, 'Status details'].apply(lambda x : float(x.split('-')[2]))
    
    statusDf.loc[indexAction, 'deltaB'] =  Bstop - Bstart
    statusDf.loc[statusDf['deltaB'] == 0, 'B_diff'] =  'none'
    statusDf.loc[statusDf['deltaB'] > 0, 'B_diff'] =  'up'
    statusDf.loc[statusDf['deltaB'] < 0, 'B_diff'] =  'down'
    
    statusDf.loc[statusDf['Status details'].apply(lambda x : x.startswith('t^')), 'Action type'] = 'power'
    statusDf.loc[statusDf['Status details'].apply(lambda x : x.startswith('sigmoid')), 'Action type'] = 'sigmoid'
    statusDf.loc[statusDf['Status details'].apply(lambda x : x.startswith('constant')), 'Action type'] = 'constant'
    
    statusDf.loc[indexAction, 'Action type'] = statusDf.loc[indexAction, 'Action type'] + '_' + statusDf.loc[indexAction, 'B_diff']
    statusDf = statusDf.drop(columns=['deltaB', 'B_diff'])
    
    # Columns from the status file
    mainActionStep = 'power_up'
    metaDf['iL'] = statusDf['iL'].astype(int)
    metaDf['Status'] = statusDf['Status']
    metaDf.loc[statusDf['Action type'] == mainActionStep, 'Status'] = 'Action_main'
    return(metaDf)


def OMEReadField(parsed_str, target_str):
    target_str = r'' + target_str
    res = []
    matches = re.finditer(target_str, parsed_str)
    for m in matches:
        str_num = parsed_str[m.end():m.end()+30]
        m_num = re.search(r'[\d\.]+', str_num)
        val_num = float(str_num[m_num.start():m_num.end()])
        res.append(val_num)
    return(res)



def OMEDataParser(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
        
        nC, nT, nZ = OMEReadField(text, ' SizeC=')[0], OMEReadField(text, ' SizeT=')[0], OMEReadField(text, ' SizeZ=')[0]
        nC, nT, nZ = int(nC), int(nT), int(nZ)
        CTZ_tz = np.zeros((nC, nT, nZ, 2))
        
        lines = text.split('\n')
        plane_lines = []
        for line in lines:
            if line.startswith('<Plane'):
                plane_lines.append(line)
        
        for line in plane_lines:
            cIdx = int(OMEReadField(line, r' TheC=')[0])
            tIdx = int(OMEReadField(line, r' TheT=')[0])
            zIdx = int(OMEReadField(line, r' TheZ=')[0])
            
            tVal = OMEReadField(line, r' DeltaT=')[0]
            zVal = OMEReadField(line, r' PositionZ=')[0]
            
            CTZ_tz[cIdx, tIdx, zIdx] = [tVal, zVal]
        
        return(CTZ_tz)



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

            # ufun.archiveFig(figV, name = f'{cellCode}_ViterbiEdge_Z{kk:.0f}', ext = '.png', dpi = 100,
            #                figDir = dstDir, cloudSave = 'none')
            
    
    intensityMap = np.array(intensityMap)
    np.savetxt(os.path.join(dstDir, f'{cellCode}_MapIntensity.txt'), intensityMap)
    fig, ax = plt.subplots(1, 1, figsize = (12, 3))
    im = ax.imshow(intensityMap, cmap = 'viridis', vmin=0)
    fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1, aspect=80)
    fig.tight_layout()
    # ufun.archiveFig(fig, name = f'{cellCode}_MapIntensity_Z{Zi:.0f}to{Zf:.0f}', ext = '.png', dpi = 100,
    #                figDir = dstDir, cloudSave = 'none')
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
        # print(A0)
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
        print('Error')
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
        ax.plot([4*marginA], [1+marginZ], 'r+')
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
    
    
# %% Predict 1 pixel

dataDir = "D:/MicroscopyData/2023-12-06_SpinningDisc_3T3-LifeAct_JV/Session2_Test1"
cellCode = 'C3'
mapPath = os.path.join(dstDir, f'{cellCode}_MapIntensity.txt')
mapCell = np.loadtxt(os.path.join(dstDir, f'{cellCode}_MapIntensity.txt'))
    

Z0, A0 = (6, 200)
pos = (Z0, A0)
marginZ, marginA = 2, 3
value_interp, iv = pixelGuesser2D(mapCell, pos, marginZ, marginA, method='linear',
                                  plot=True)


# %% Predict the whole map

dataDir = "D:/MicroscopyData/2023-12-06_SpinningDisc_3T3-LifeAct_JV/Session2_Test1"
cellCode = 'C3'
mapPath = os.path.join(dstDir, f'{cellCode}_MapIntensity.txt')
mapCell = np.loadtxt(os.path.join(dstDir, f'{cellCode}_MapIntensity.txt'))
mapCell *= 100

marginZ, marginA = 2, 3

nZ, nA = mapCell.shape
mapCell_pred = np.zeros((nZ, nA))

for iZ in range(2, nZ):
    for iA in range(nA):
        pos = (iZ, iA)
        value_interp, iv = pixelGuesser2D(mapCell, pos, marginZ, marginA, method='linear',
                                          plot=False)
        mapCell_pred[iZ, iA] = value_interp
        

vmin, vmax = np.min(mapCell), np.max(mapCell)
    
fig, axes = plt.subplots(3, 1, figsize = (12, 8))
ax = axes[0]
im = ax.imshow(mapCell, vmin = vmin, vmax = vmax)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical') #, fraction=.03, aspect=20)
ax.set_title('real')

ax = axes[1]
im = ax.imshow(mapCell_pred, vmin = vmin, vmax = vmax)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical') #, fraction=.03, aspect=20)
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



# %% Cell Mappemonde - Mode ZT

cell = 'C9'

srcDir = "D:/MagneticPincherData/Raw/24.02.27_Chameleon/Clean_Fluo_BF"
fileName = f"3T3-LifeActGFP_PincherFluo_{cell}_off4um-01.tif"
# fileList = os.listdir(srcDir)

dstDir =  os.path.join(srcDir, f'{cell}_Results')
filePath = os.path.join(srcDir, fileName)

figSavePath = dstDir
cellId = f'24-02-27_{cell}'

I = skm.io.imread(filePath)
czt_shape, czt_seq = get_CZT_fromTiff(filePath)

[nC, nZ, nT] = czt_shape

for t in range(nT):
    intensityMap_t = []
    
    for z in range(nZ):
        i_zt = z + nT*t
        I_zt = I[i_zt]
        
        #### Parameters
        framesize = int(35*8/2) 
        inPix, outPix = 20, 5
        blur_parm = 4
        relative_height_virebi = 0.4
        interp_factor_x = 5
        interp_factor_y = 10
        VEW = 7 # Viterbi edge width # Odd number
        
        
        #### Locate cell
        (Yc, Xc), approxRadius = findCellInnerCircle(I_zt, withBeads = True, plot=False)
        Yc0, Xc0, R0 = round(Yc), round(Xc), round(approxRadius)
        
        #### 2.1 Warp
        warped = skm.transform.warp_polar(I_zt, center=(Yc, Xc), radius=approxRadius*1.3, 
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
        
        intensityMap_t.append(viterbi_Npix_values)
        
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
            
            figV.suptitle(f'Warp - {cellId} - t = {t+1:.0f}/{nT:.0f} - z = {z+1:.0f}/{nZ:.0f}')
            figName = f'{cellId}_Warp_t{t+1:.0f}_z{z+1:.0f}'
            ufun.simpleSaveFig(figV, figName, figSavePath, '.png', 150)

            
#             ufun.archiveFig(figV, name = f'{cellCode}_ViterbiEdge_Z{kk:.0f}', ext = '.png', dpi = 100,
#                             figDir = dstDir, cloudSave = 'none')
        

    intensityMap_t = np.array(intensityMap_t)
    np.savetxt(os.path.join(dstDir, f'{cellId}_Map_t{t+1:.0f}.txt'), intensityMap_t)
    figM, axM = plt.subplots(1, 1, figsize = (16, 2))
    ax = axM
    im = ax.imshow(intensityMap_t, cmap = 'viridis', vmin=0)
    # figM.colorbar(im, ax=ax, orientation='horizontal', fraction=.1, aspect=80)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    figM.colorbar(im, cax=cax, orientation='vertical') 
    ax.set_title(f'Map - {cellId} - t = {t+1:.0f}/{nT:.0f}')
    figM.tight_layout()
    figName = f'{cellId}_Map_t{t+1:.0f}'
    ufun.simpleSaveFig(figM, figName, figSavePath, '.png', 150)
    plt.show()
    
    
# %% Load maps and make the ones in t

srcDir = "D:/MagneticPincherData/Raw/24.02.27_Chameleon/Clean_Fluo"
fileName = "3T3-LifeActGFP_PincherFluo_C7_off4um-01.tif"

dstDir =  os.path.join(srcDir, 'C7_results')
filePath = os.path.join(srcDir, fileName)

figSavePath = dstDir
cellId = '24-02-27_C7'

I = skm.io.imread(filePath)
czt_shape, czt_seq = get_CZT_fromTiff(filePath)

[nC, nZ, nT] = czt_shape

allMaps_T = []
allMaps_Z = []

for t in range(nT):
    mapPath = os.path.join(dstDir, f'{cellId}_Map_t{t+1:.0f}.txt')
    intensityMap_t = np.loadtxt(mapPath)
    allMaps_T.append(intensityMap_t)
    
print(allMaps_T[0].shape)

for z in range(nZ):
    intensityMap_z = np.zeros_like(allMaps_T[0])
    for t in range(nT):
        intensityMap_z[t,:] = allMaps_T[t][z,:]
    figM, axM = plt.subplots(1, 1, figsize = (16, 2))
    ax = axM
    im = ax.imshow(intensityMap_z, cmap = 'viridis', vmin=0)
    # figM.colorbar(im, ax=ax, orientation='horizontal', fraction=.1, aspect=80)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    figM.colorbar(im, cax=cax, orientation='vertical') 
    ax.set_title(f'Map - {cellId} - z = {z+1:.0f}/{nZ:.0f}')
    figM.tight_layout()
    figName = f'{cellId}_Map_z{z+1:.0f}'
    ufun.simpleSaveFig(figM, figName, figSavePath, '.png', 150)
    allMaps_Z.append(intensityMap_z)
    
    
plt.show()
    



# %% Complete pipeline: tracking bf -> fluo

date = '24-02-27'
cell = 'C5'

# %%% Tracking bf

# %%%% Define paths

dictPaths = {'sourceDirPath' : 'D:/MagneticPincherData/Raw/24.02.27_Chameleon/Clean_Fluo_BF',
             'imageFileName' : f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_BF.tif',
             'resultsFileName' : f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_BF_Results.txt',
             'depthoDir':'D:/MagneticPincherData/Raw/DepthoLibrary',
             'depthoName':'24.02.27_Chameleon_M450_step20_100X',
             'resultDirPath' : 'D:/MagneticPincherData/Raw/24.02.27_Chameleon/Clean_Fluo_BF/Results_Tracker',
             }


# %%%% Define constants

dictConstants = {'microscope' : 'labview',
                 #
                 'inside bead type' : 'M450-2025', # 'M450' or 'M270'
                 'inside bead diameter' : 4493, # nm
                 'inside bead magnetization correction' : 0.969, # nm
                 'outside bead type' : 'M450-Strept', # 'M450' or 'M270'
                 'outside bead diameter' : 4506, # nm
                 'outside bead magnetization correction' : 1.056, # nm
                 #
                 'normal field multi images' : 11, # Number of images
                 'multi image Z step' : 400, # nm
                 'multi image Z direction' : 'downward', # Either 'upward' or 'downward'
                 #
                 'scale pixel per um' : 7.4588, # pixel/µm
                 'optical index correction' : 0.875, # ratio, without unit
                 'beads bright spot delta' : 0, # Rarely useful, do not change
                 'magnetic field correction' : 1.0, # ratio, without unit
                 }


# %%%% Additionnal options

dictOptions = {'redoAllSteps' : True,
               'trackAll' : False,
               'importLogFile' : True,
               'saveLogFile' : True,
               'saveFluo' : False,
               'importTrajFile' : True,
               'expandedResults' : True,
              }

# %%%% Make metaDataFrame # NEED TO CODE STH HERE !

OMEfilepath = os.path.join(dictPaths['sourceDirPath'], f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_OME.txt')

CTZ_tz = OMEDataParser(OMEfilepath)
BF_T = CTZ_tz[0,:,:,0].flatten()

NFrames = len(BF_T)
B_set = 5 * np.ones(NFrames)

loopStructure = [{'Status':'Passive',     'NFrames':NFrames}]  # Only one constant field phase
                
metaDf = makeMetaData(BF_T, B_set, loopStructure)

# %%%% Call mainTracker()

# tsdf = smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)


# %%% Mappemonde fluo

# %%%% Paths

srcDir = 'D:/MagneticPincherData/Raw/24.02.27_Chameleon/Clean_Fluo_BF'
fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_Fluo.tif'
# fileList = os.listdir(srcDir)

dstDir =  os.path.join(srcDir, 'Results_Fluo', f'{cell}')
fluoPath = os.path.join(srcDir, fluoName)

figSavePath = dstDir
cellId = f'{date}_{cell}'


bfName = f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_BF.tif'
bfPath = os.path.join(srcDir, bfName)
resBfDir = os.path.join(srcDir, 'Results_Tracker')
resBfPath = os.path.join(resBfDir, bfName[:-4] + '_PY.csv')

# %%%% Options

framesize = int(35*8/2) 
inPix_set, outPix_set = 20, 20
blur_parm = 4
relative_height_virebi = 0.2
interp_factor_x = 5
interp_factor_y = 10
VEW = 5 # Viterbi edge width # Odd number
VE_in, VE_out = 7, 3 # Viterbi edge in/out

Fluo_T = np.mean(CTZ_tz[1,:,:,0], axis=1)

SAVE_RESULTS = True
PLOT_WARP = False
PLOT_MAP = False

optical_index_correction = 0.875
dz = 0.4  * optical_index_correction # µm # Distance between each image of the stacks
DZb = 4.5 * optical_index_correction # µm # Distance between focal point of the deptho and the equatorial plane of the beads #### SHOULD BE RE-MEASURED
DZo = 4.0 * optical_index_correction # µm # Distance between the stack of BF images and the corresponding stack of Fluo images
Rbead = 4.5/2

# %%%% Compute map



#### Start

plt.ioff()
I = skm.io.imread(fluoPath)

czt_shape, czt_seq = get_CZT_fromTiff(fluoPath)
[nC, nZ, nT] = czt_shape
Angles = np.arange(0, 360, 1)

tsdf = pd.read_csv(resBfPath, sep=None, engine='python')

ColsContact = ['iT', 'T', 'X', 'Y', 'Z', 'R', 'A', 'Xc', 'Yc', 'Rc', 
               'Xin', 'Yin', 'Zin', 'Xout', 'Yout', 'Zout', 
               f'I_viterbi_w{VEW:.0f}', f'I_viterbi_inout{VE_in:.0f}-{VE_out:.0f}', 'I_viterbi_flexW', 'W_viterbi_flexW']

DfContact = pd.DataFrame(np.zeros((nT, len(ColsContact))), columns=ColsContact)
fluoCell = np.zeros((nT, nZ))
fluoCyto = np.zeros((nT, nZ))
fluoCytoStd = np.zeros((nT, nZ))
fluoBack = np.zeros((nT, nZ))
arrayViterbiContours = np.zeros((nT, nZ, 360, 2))

for t in range(nT):
    intensityMap_t = []
    tsd_t = tsdf.iloc[t,:].to_dict()
        
    #### 1.0 Compute position of contact point
    x_contact = (tsd_t['X_in']+tsd_t['X_out'])/2
    y_contact = (tsd_t['Y_in']+tsd_t['Y_out'])/2
    
    zr_contact = (tsd_t['Zr_in']+tsd_t['Zr_out'])/2
    CF_mid = - zr_contact + DZo - DZb 
    # EF = ED + DI + IF; ED = - DZb; for I=I_mid & F=F_mid, DI = - zr; and IF = DZo
    iz_mid = (nZ-1)//2 # If 11 points in Z, iz_mid = 5
    iz_contact = iz_mid - (CF_mid / dz)
    # print(iz_contact)
    
    x_in = tsd_t['X_in']
    y_in = tsd_t['Y_in']
    z_in = ((-1)*tsd_t['Zr_in']) + DZo - DZb
    
    x_out = tsd_t['X_out']
    y_out = tsd_t['Y_out']
    z_out = ((-1)*tsd_t['Zr_out']) + DZo - DZb
    
    DfContact.loc[t, ['iT', 'T', 'X', 'Y', 'Z']] = [t, Fluo_T[t], x_contact, y_contact, iz_contact*dz]
    DfContact.loc[t, ['Xin', 'Yin', 'Zin', 'Xout', 'Yout', 'Zout']] = [x_in, y_in, dz*iz_mid-z_in, x_out, y_out, dz*iz_mid-z_out]

    
    # Zr = (-1) * (tsd_t['Zr_in']+tsd_t['Zr_out'])/2 # Taking the opposite cause the depthograph is upside down for these experiments
    # izmid = (nZ-1)//2 # If 11 points in Z, izmid = 5
    # zz = dz * np.arange(-izmid, +izmid, 1) # If 11 points in Z, zz = [-5, -4, ..., 4, 5] * dz
    # zzf = zz + Zr # Coordinates zz of bf images with respect to Zf, the Z of the focal point of the deptho
    # zzeq = zz + Zr + DZo - DZb # Coordinates zz of fluo images with respect to Zeq, the Z of the equatorial plane of the bead
    
    # zi_contact = Akima1DInterpolator(zzeq, np.arange(len(zzeq)))(0)
    
    # contactSlice = False
    A_contact_map = 0
    foundContact = False
    
    for z in range(nZ):
        contactSlice = False
        i_zt = z + nT*t
        I_zt = I[i_zt]
        
        #### 2.0 Locate cell
        (Yc, Xc), Rc = findCellInnerCircle(I_zt, withBeads = True, plot=False)
        Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
        
        #### 2.1 Warp
        warped = skm.transform.warp_polar(I_zt, center=(Yc, Xc), radius=Rc*1.2, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        
        max_values = np.max(warped[:,Rc0-inPix_set:Rc0+outPix_set+1], axis=1)
            
        #### 2.2 Viterbi Smoothing
        edge_viterbi = ViterbiEdge(warped, Rc0, inPix_set, outPix_set, 2, relative_height_virebi)
        edge_viterbi_unwarped = np.array(unwarpRA(np.array(edge_viterbi), Angles, Xc, Yc)) # X, Y
        
        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(warped, cmap='gray', aspect='equal')
        # ax.plot(edge_viterbi, Angles, 'r--')
        # plt.show()


        #### 3.0 Locate cell
        (Xc, Yc), Rc = fitCircle(np.array([edge_viterbi_unwarped[1], edge_viterbi_unwarped[0]]).T, loss = 'huber')
        Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
        DfContact.loc[t, ['Xc', 'Yc', 'Rc']] += np.array([(Xc/SCALE_100X_ZEN)/(nZ), (Yc/SCALE_100X_ZEN)/(nZ), (Rc/SCALE_100X_ZEN)/(nZ)])
        
        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(I_zt, cmap='gray', aspect='equal')
        # ax.plot(edge_viterbi_unwarped[0], edge_viterbi_unwarped[1], 'r--')
        # ax.plot(Xc, Yc, 'bo')
        # circle_test = circleContour_V2((Yc, Xc), Rc)
        # ax.plot(circle_test[:,1], circle_test[:,0], 'g--')
        # plt.show()
        
        #### 3.1 Warp
        warped = skm.transform.warp_polar(I_zt, center=(Yc, Xc), radius=Rc*1.4, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        warped = skm.util.img_as_uint(warped)
        w_ny, w_nx = warped.shape
        Angles = np.arange(0, 360, 1)
        max_values = np.max(warped[:,Rc0-inPix_set:Rc0+outPix_set+1], axis=1)
        
        
        R_contact, A_contact = warpXY(x_contact*SCALE_100X_ZEN, y_contact*SCALE_100X_ZEN, Xc, Yc)
                
        if (z == np.round(iz_contact)) or (not foundContact and z == nZ-1):
            contactSlice = True
            foundContact = True
            # print(z, R_contact, A_contact)
            A_contact_map = A_contact
            DfContact.loc[t, ['R', 'A']] = [R_contact, A_contact]
        
        #### 3.2 Interp in X
        warped_interp = ufun.resize_2Dinterp(warped, fx=interp_factor_x, fy=1)
        
        warped = warped_interp
        w_nx, Rc0 = w_nx * interp_factor_x, Rc0 * interp_factor_x
        inPix, outPix = inPix_set * interp_factor_x, outPix_set * interp_factor_x
        
        R_contact = R_contact * interp_factor_x
        
        #### 3.3 Viterbi Smoothing
        edge_viterbi = ViterbiEdge(warped, Rc0, inPix, outPix, blur_parm, relative_height_virebi)
        edge_viterbi_unwarped = unwarpRA(np.array(edge_viterbi)/interp_factor_x, Angles, Xc, Yc)
        
        arrayViterbiContours[t, z, :] = np.array([edge_viterbi_unwarped[0], edge_viterbi_unwarped[1]]).T
        
        #### 4. Analyse fluo cyto & background
        # if COMPUTE_FLUO_INOUT:
        path_edge = mpltPath.Path(np.array(edge_viterbi_unwarped).T)
        xx, yy = np.meshgrid(np.arange(I_zt.shape[1]), np.arange(I_zt.shape[0]))
        pp = np.array([[x, y] for x, y in zip(xx.flatten(), yy.flatten())]) # .reshape(I_zt.shape[1], I_zt.shape[0], 2)
        inside = path_edge.contains_points(pp).reshape(I_zt.shape[0], I_zt.shape[1])
        
        # Inside
        cyto_margin = 10
        inside_eroded = ndi.binary_erosion(inside, iterations = cyto_margin)
        data_cyto = I_zt.flatten()[inside_eroded.flatten()]
        threshold_cyto = skm.filters.threshold_li(data_cyto)
        # threshold_cell = skm.filters.threshold_triangle(data_cyto) #[0.9*skm.filters.threshold_triangle(data_cell)]
        mask_cyto = (I_zt > 0.9*threshold_cyto).astype(bool)
        I_cyto = np.median(I_zt[mask_cyto & inside_eroded])
        Std_cyto = np.std(I_zt[mask_cyto & inside_eroded])
        
        # Outside
        background_margin = 140
        outside = ~ndi.binary_dilation(inside, iterations = background_margin)
        data = I_zt.flatten()[outside.flatten()]
        I_back = np.percentile(I_zt[outside], 30)
        
        # Whole cell
        cell_margin = 3
        cell_dilated = ndi.binary_dilation(inside, iterations = cell_margin)
        
        # hist, bins = np.histogram(data_cyto.ravel(), bins=255)
        # hist_s = ndi.gaussian_filter1d(hist, sigma=2)
        # peaks, props = signal.find_peaks(hist_s, prominence = 10, distance = 15, height = 0)
        # i_maxHeight = np.argmax(props['peak_heights'])
        # test_nPeaks = (i_maxHeight > 0)
        # test_doublePeaks = (len(peaks)==2 and min(props['peak_heights'])>60)
        # threshold_cell = 0
        # if test_nPeaks or test_doublePeaks:
        #     threshold_cell = skm.filters.threshold_minimum(hist = (hist, bins))
        #     if np.sum((cell_dilated.astype(bool) * (I_zt > 0.95*threshold_cell).astype(bool)).astype(int)) < 5000:
        #         threshold_cell = 0
                    
        # cell_filtered = cell_dilated.astype(bool) * (I_zt > 0.95*threshold_cell).astype(bool)
        

        mask_removed_from_cyto = inside & (~mask_cyto)
        cell_filtered = cell_dilated & (~mask_removed_from_cyto)
        
        nb_pix = np.sum(cell_filtered.astype(int))
        total_val = np.sum(I_zt.flatten()[cell_filtered.flatten()])
        I_cell = total_val/nb_pix
        
        # Store
        fluoCyto[t, z] = I_cyto
        fluoCytoStd[t, z] = Std_cyto
        fluoBack[t, z] = I_back
        fluoCell[t, z] = I_cell
        
        # Display
        # figT, axesT = plt.subplots(1,4, figsize = (16, 5))
        # ax = axesT[0]
        # ax.imshow(I_zt)
        # ax.set_title('Original')
        # ax = axesT[1]
        # ax.imshow((mask_cyto & inside_eroded).astype(int))
        # ax.set_title('Cytoplasm')
        # ax = axesT[2]
        # ax.imshow(cell_filtered.astype(int))
        # ax.set_title('Whole cell')
        # ax = axesT[3]
        # ax.imshow(outside.astype(int))
        # ax.set_title('Background')
        # # # hist, bins, patches = ax.hist(data_cyto.ravel(), bins=255)
        # # # peaks, props = signal.find_peaks(hist, prominence = 40, distance = 20)
        # # ax.plot(bins[:-1], hist_s, 'g-')
        # # ax.hist(data_cyto.ravel(), bins=255)
        # # if len(peaks) <= 1:
        # #     threshold_cyto2 = 0
            
        # # ax.axvline(threshold_cell, color='r')
        # # for p in peaks:
        # #     val = bins[p]
        # #     ax.axvline(val, color='gold')
            
        # # testI = inside.astype(int) + inside_eroded.astype(int) + (inside_eroded*mask_cyto).astype(int)
        # # testI -= outside.astype(int)
        # # testI = np.sum([(I_zt > th).astype(int) for th in thresholds_cell], axis=0)
        # figT.tight_layout()
        # plt.show()
        
        def normalize_Icyto(x):
            return((x - I_back)/(I_cyto - I_back))
    
        def normalize_Icell(x):
            return((x - I_back)/(I_cell - I_back))
        
        def normalize_Icell2(x):
            return((x - I_cyto)/(I_cell - I_back))
        
        # warped_n = (warped-I_back) / (I_cyto-I_back)
        
        #### 5. Read values
        max_values = normalize_Icell(max_values)
        
        viterbi_Npix_values = [np.sum(warped[Angles[i], edge_viterbi[i]-VEW//2:edge_viterbi[i] + (VEW//2 + 1)])/VEW for i in range(len(edge_viterbi))]
        viterbi_Npix_values = normalize_Icell(viterbi_Npix_values)
        intensityMap_t.append(viterbi_Npix_values)
        
        if contactSlice:
            iA_contact = round(A_contact)
            DfContact.loc[t, f'I_viterbi_w{VEW:.0f}'] = viterbi_Npix_values[iA_contact]
            
            viterbi_inout_value = np.sum(warped[iA_contact-1:iA_contact+2, 
                                                edge_viterbi[iA_contact]-VE_in*interp_factor_x:edge_viterbi[iA_contact]+VE_out*interp_factor_x])/(3*(VE_in+VE_out)*interp_factor_x)
            viterbi_inout_value = normalize_Icell(viterbi_inout_value)
            DfContact.loc[t, f'I_viterbi_inout{VE_in:.0f}-{VE_out:.0f}'] = viterbi_inout_value
            
            
            
            
            #### X. viterbi_flex_value
            viterbi_flex_value = 0
            angleOffsets = [-1,0,1]
            
            figW, axesW = plt.subplots(len(angleOffsets), 1, figsize = (5, 4*len(angleOffsets)))
            
            for jj in range(len(angleOffsets)):
                iAngle = angleOffsets[jj]
                r_edge = edge_viterbi[iA_contact+iAngle]
                viterbiContactProfile = normalize_Icell2(warped[iA_contact+iAngle, :])
                
                r_init  = r_edge - ufun.findFirst(True, viterbiContactProfile[:r_edge][::-1] < 0) # I_cyto is set to 0 with normalize_Icell2
                r_final = r_edge + ufun.findFirst(True, viterbiContactProfile[r_edge:] < 0) # I_cyto is set to 0 with normalize_Icell2
                
                viterbi_flex_value += np.sum(viterbiContactProfile[r_init:r_final]) * (1/len(angleOffsets))
                
                try:
                    ax = axesW[jj]
                except:
                    ax = axesW
                ax.plot(np.arange(len(viterbiContactProfile)), (normalize_Icell(warped[iA_contact+iAngle, :])))
                ax.axhline(normalize_Icell(I_cyto), c='r')
                # ax.axhline(I_cyto+Std_cyto, c='r', ls = '--')
                ax.axhline(normalize_Icell(I_cell), c='g')
                ax.axhline(normalize_Icell(I_back), c='b')
                ax.axvline(r_edge, c='gold')
                ax.axvline(r_init, c='gold', ls = '--')
                ax.axvline(r_final, c='gold', ls = '--')
            
            # print(viterbi_flex_value)
            # viterbi_inout_value = normalize_Icell(viterbi_inout_value)
            # print(viterbi_inout_value)
            
            DfContact.loc[t, 'I_viterbi_flexW'] = viterbi_flex_value         
            DfContact.loc[t, 'W_viterbi_flexW'] = r_final-r_init  
                
            figW.tight_layout()
            plt.show()
            
            
        
        
        #### 6. Plot
        #### 6.1 Compute beads positions
        draw_InCircle = False
        draw_OutCircle = False
        
        if PLOT_WARP:
            h_InCircle = Rbead + (z_in + dz*(z-iz_mid))
            h_OutCircle = Rbead + (z_out + dz*(z-iz_mid))
            
            if h_InCircle > 0 and h_InCircle < 2*Rbead:
                draw_InCircle = True
                a_InCircle = (h_InCircle*(2*Rbead-h_InCircle))**0.5
                # INbeadContour = circleContour((round(y_in*SCALE_100X_ZEN), round(x_in*SCALE_100X_ZEN)), 
                #                                a_InCircle*SCALE_100X_ZEN, I_zt.shape)
                INbeadContour = circleContour_V2((round(y_in*SCALE_100X_ZEN), round(x_in*SCALE_100X_ZEN)), 
                                                 a_InCircle*SCALE_100X_ZEN)
                XINbeadContour, YINbeadContour = INbeadContour[:, 1], INbeadContour[:, 0]
                RINbeadContour, AINbeadContour = warpXY(XINbeadContour, YINbeadContour, Xc, Yc)
                RINbeadContour *= interp_factor_x
                
            if h_OutCircle > 0 and h_OutCircle < 2*Rbead:
                draw_OutCircle = True
                a_OutCircle = (h_OutCircle*(2*Rbead-h_OutCircle))**0.5
                # OUTbeadContour = circleContour((round(y_out*SCALE_100X_ZEN), round(x_out*SCALE_100X_ZEN)), 
                #                                a_OutCircle*SCALE_100X_ZEN, I_zt.shape)
                OUTbeadContour = circleContour_V2((round(y_out*SCALE_100X_ZEN), round(x_out*SCALE_100X_ZEN)),
                                                  a_OutCircle*SCALE_100X_ZEN)
                XOUTbeadContour, YOUTbeadContour = OUTbeadContour[:, 1], OUTbeadContour[:, 0]
                ROUTbeadContour, AOUTbeadContour = warpXY(XOUTbeadContour, YOUTbeadContour, Xc, Yc)
                ROUTbeadContour *= interp_factor_x   
                
        #### 6.2 Warp and Viterbi   
        if PLOT_WARP:
            # Init the fig
            figV2 = plt.figure(figsize=(20, 10))#, layout="constrained")
            spec = figV2.add_gridspec(2, 7)
            
            inBorder = Rc0 - inPix
            outBorder = Rc0 + outPix
            Rc0A = np.ones(360)*Rc0
            inBorderA = np.ones(360)*inBorder
            outBorderA = np.ones(360)*outBorder
            
            
            # 1.1 - Show the cell
            ax = figV2.add_subplot(spec[0, :2])
            ax.imshow(I_zt, cmap='gray', aspect='equal')
            ax.set_title('Raw Image')
            
            # 1.2 - Show the cell with annotations
            ax = figV2.add_subplot(spec[1, :2])
            ax.imshow(I_zt, cmap='gray', aspect='equal')
            ax.plot(edge_viterbi_unwarped[0], edge_viterbi_unwarped[1], c='g', ls = '--')
            if draw_InCircle:
                ax.plot(XINbeadContour, YINbeadContour, c='r', ls=':')
            if draw_OutCircle:
                ax.plot(XOUTbeadContour, YOUTbeadContour, c='gold', ls=':')
            ax.set_title('Detected contours')
            
            # 2.1 - Show the warped cell
            ax1 = figV2.add_subplot(spec[:, 2])
            ax = ax1
            ax.imshow(warped[:, :], cmap='gray', aspect='auto')
            ax.set_title('Raw warp')
            ax.set_ylabel('Angle (deg)')
            ax.set_xlabel('Radial pix')
            
            # 2.2 - Show the warped cell with viterbi detection
            ax = figV2.add_subplot(spec[:, 3], sharey = ax1)
            ax.imshow(warped[:, :], cmap='gray', aspect='auto')
            ax.plot(Rc0A, Angles, ls='--', c='red', lw=1)
            ax.plot(inBorderA, Angles, ls='--', c='gold', lw=1)
            ax.plot(outBorderA, Angles, ls='--', c='gold', lw=1)
            ax.plot(edge_viterbi, Angles, ls='', c='cyan', marker='.', ms = 2)
            ax.plot(R_contact, A_contact, 'r+', ms=5)
            if draw_InCircle:
                ax.plot(RINbeadContour, AINbeadContour, c='b', ls=':')
            if draw_OutCircle:
                mask = (ROUTbeadContour < w_nx)
                ROUTbeadContour = ROUTbeadContour[mask]
                AOUTbeadContour = AOUTbeadContour[mask]
                ax.plot(ROUTbeadContour, AOUTbeadContour, c='cyan', ls=':')
            ax.set_title('Viterbi Edge')
            ax.set_xlabel('Radial pix')
            
            # 3.1 - Show the intensity profiles for Viterbi
            ax = figV2.add_subplot(spec[:, 4], sharey = ax1)
            step1 = 5
            ax.plot(inBorderA, Angles, c='k', ls='--', marker='')
            ax.plot(outBorderA, Angles, c='k', ls='--', marker='')
            ax.plot(edge_viterbi[::step1], Angles[::step1], ls='', c='cyan', marker='^', ms = 4, mec = 'k', zorder=5)
            ax.set_title('Viterbi profiles')
            ax.set_xlabel('Radial pix')
            
            cList_viridis = plt.cm.viridis(np.linspace(0.05, 0.95, w_ny//step1))
            ax.set_prop_cycle(plt.cycler("color", cList_viridis))
            maxWarped = np.max(warped)
            normWarped = warped * 6 * step1/maxWarped
            for a in Angles[::step1]:            
                profile = normWarped[a, :] - np.min(normWarped[a, :])
                RR = np.arange(len(profile))
                ax.plot(RR, a - profile)

            # 3.2 - Zoom on the intensity profiles for Viterbi
            ax = figV2.add_subplot(spec[:, 5], sharey = ax1)
            step2 = 20
            ax.plot(edge_viterbi[::step2], Angles[::step2], ls='', c='cyan', marker='^', ms = 4, mec = 'k', zorder=5)
            cList_viridis = plt.cm.viridis(np.linspace(0.05, 0.95, w_ny//step2))
            ax.set_prop_cycle(plt.cycler("color", cList_viridis))
            ax.set_title('Viterbi profiles')
            ax.set_xlabel('Radial pix')
            maxWarped = np.max(warped)
            normWarped = warped * 6 * step2/maxWarped
            for a in Angles[::step2]:            
                profile = normWarped[a, Rc0-inPix:Rc0+outPix] - np.min(normWarped[a, Rc0-inPix:Rc0+outPix])
                RR = np.arange(Rc0-inPix, Rc0+outPix, 1)
                ax.plot(RR, a - profile)

            # 4 - Show the intensity along the cell contour
            ax = figV2.add_subplot(spec[:, 6], sharey = ax1)
            ax.set_title('Profile Values')
            ax.plot(max_values, Angles, 'r-', label = 'Max profile')
            ax.plot(viterbi_Npix_values, Angles, 'b-', label = 'Viterbi profile')
            ax.legend()
            
            figV2.suptitle(f'Warp - {cellId} - t = {t+1:.0f}/{nT:.0f} - z = {z+1:.0f}/{nZ:.0f}')
            figName = f'{cellId}_Warp_t{t+1:.0f}_z{z+1:.0f}'
            figV2.tight_layout()
            ufun.simpleSaveFig(figV2, figName, figSavePath, '.png', 150)
            plt.close('all')
    
    intensityMap_t = np.array(intensityMap_t).astype(float)
    np.savetxt(os.path.join(dstDir, f'{cellId}_Map_t{t+1:.0f}.txt'), intensityMap_t, fmt='%.5f')
    
    if PLOT_MAP:
        #### 6.3 Map and contact point
        figM, axM = plt.subplots(1, 1, figsize = (16, 2))
        ax = axM
        im = ax.imshow(intensityMap_t, cmap = 'viridis', vmin=0, origin='lower')
        # figM.colorbar(im, ax=ax, orientation='horizontal', fraction=.1, aspect=80)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        figM.colorbar(im, cax=cax, orientation='vertical', aspect=20) 
        ax.set_title(f'Map - {cellId} - t = {t+1:.0f}/{nT:.0f}')
        ax.plot(A_contact_map, iz_contact+0.5, 'r+', ms=7)
        figM.tight_layout()
        figName = f'{cellId}_Map_t{t+1:.0f}'
        ufun.simpleSaveFig(figM, figName, figSavePath, '.png', 150)
        # plt.show()
        plt.close('all')

# np.savetxt(os.path.join(dstDir, f'{cellId}_Contact-XYZ.txt'), contactXYZ)        
       
plt.ion()
if SAVE_RESULTS:
    DfContact.to_csv(os.path.join(dstDir, f'{cellId}_DfContact.csv'), sep=';', index=False)
    np.save(os.path.join(dstDir, f'{cellId}_xyContours.npy'), arrayViterbiContours)
    np.savetxt(os.path.join(dstDir, f'{cellId}_fluoCyto.txt'), fluoCyto, fmt='%.1f')
    np.savetxt(os.path.join(dstDir, f'{cellId}_fluoCytoStd.txt'), fluoCytoStd, fmt='%.3f')
    np.savetxt(os.path.join(dstDir, f'{cellId}_fluoBack.txt'), fluoBack, fmt='%.1f')
    np.savetxt(os.path.join(dstDir, f'{cellId}_fluoCell.txt'), fluoCell, fmt='%.1f')
    

# %%% 

cell='C5'


# %%%% Load files

cellId = f'{date}_{cell}'

srcDir = 'D:/MagneticPincherData/Raw/24.02.27_Chameleon/Clean_Fluo_BF'
fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_Fluo.tif'
bfName = f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_BF.tif'

bfPath = os.path.join(srcDir, bfName)
resBfDir = os.path.join(srcDir, 'Results_Tracker')
resBfPath = os.path.join(resBfDir, bfName[:-4] + '_PY.csv')

fluoPath = os.path.join(srcDir, fluoName)
resFluoDir =  os.path.join(srcDir, 'Results_Fluo', f'{cell}')

resCrossDir = os.path.join(srcDir, 'Results_Cross', f'{cell}')

czt_shape, czt_seq = get_CZT_fromTiff(fluoPath)
[nC, nZ, nT] = czt_shape

tsdf = pd.read_csv(resBfPath, sep=None, engine='python')

DfContact = pd.read_csv(os.path.join(resFluoDir, f'{cellId}_DfContact.csv'), sep=None, engine='python')
fluoCyto = np.loadtxt(os.path.join(resFluoDir, f'{cellId}_fluoCyto.txt'))
fluoBack = np.loadtxt(os.path.join(resFluoDir, f'{cellId}_fluoBack.txt'))
fluoCytoStd = np.loadtxt(os.path.join(resFluoDir, f'{cellId}_fluoCytoStd.txt'))
arrayViterbiContours = np.load(os.path.join(resFluoDir, f'{cellId}_xyContours.npy'))

optical_index_correction = 0.875
dz = 0.4  * optical_index_correction # µm # Distance between each image of the stacks
DZb = 4.5 * optical_index_correction # µm # Distance between focal point of the deptho and the equatorial plane of the beads #### SHOULD BE RE-MEASURED
DZo = 4.0 * optical_index_correction # µm # Distance between the stack of BF images and the corresponding stack of Fluo images

# %%%% Function

def pixelGuesser(mapCell, pos, marginZ, marginA, method='linear',
                   plot = False):
    Z0, A00 = pos
    nZ, nA = mapCell.shape

    if (A00 < (5*marginA)) or (nA-A00 < (5*marginA)):
        mapCell = np.roll(mapCell, 7*marginA, axis=1)
        A0 = (A00+7*marginA)%nA
        # print(A0)
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
        print('Error')
        print(Z0, A0, A00)
    
    if plot:
        value_real = mapCell[Z0, A0]
        rel_error = 100 * (value_pos - value_real) / value_real

        mask = np.zeros_like(mapCell, dtype=bool)
        mask[Z0-marginZ:Z0+1, A0-marginA:A0+marginA+1] = True

        mapCell2 = np.copy(mapCell)
        np.place(mapCell2, mask, interp_values_reshaped.flatten())
        
        marginPlot = 11
        mini_map = mapCell[:, A0-marginPlot:A0+marginPlot+1]*100
        mini_map2 = mapCell2[:, A0-marginPlot:A0+marginPlot+1]*100
        newZ0, newA0 = mini_map.shape[0], mini_map.shape[1] // 2

        fig, axes = plt.subplots(1, 3, figsize = (16, 4))
        ax = axes[0]
        im = ax.imshow(mini_map)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(im, cax=cax, orientation='vertical') #, fraction=.03, aspect=20)
        ax.set_title('real')
        ax.plot([marginPlot], [Z0], 'r+')
        ax.add_patch(Rectangle((newA0 - marginA - 0.5, Z0 - marginZ - 0.5), 
                               2*marginA + 1, 1*marginZ + 1, 
                               edgecolor='red', facecolor='none', lw=1))

        ax = axes[1]
        im = ax.imshow(mini_map2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(im, cax=cax, orientation='vertical') #, fraction=.03, aspect=20)
        ax.set_title('predicted')
        ax.plot([marginPlot], [Z0], 'r+')
        ax.add_patch(Rectangle((newA0 - marginA - 0.5, Z0 - marginZ - 0.5), 
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
        ax.plot([marginPlot], [Z0], 'k+')

        fig.suptitle(f'Error at target point = {rel_error:.2f} %')
        fig.tight_layout()

        plt.show()
    
    return(value_pos, interp_values_reshaped)


def profileExplorer(fluoPath, DfContact, fluoCyto, fluoBack, arrayViterbiContours,
                    T, Z, A, r_in, r_out, normalizeInt = True,
                    plot = False):
    I = skm.io.imread(fluoPath)
    czt_shape, czt_seq = get_CZT_fromTiff(fluoPath)
    dictContact = DfContact.set_index('iT').iloc[T,:].to_dict()
    [nC, nZ, nT] = czt_shape
    Z_profiles = np.arange(Z-1, Z+2)
    A_margin = 2
    A_profiles = np.arange(A-A_margin, A+A_margin+1)%360
    print(A_profiles)
    # A_profiles = np.arange(0,360)
    ZZ, AA = np.meshgrid(Z_profiles, A_profiles)
    nR = r_out + r_in
    
    profiles = np.zeros((len(Z_profiles), 360, nR))
    
    for z in ZZ.flatten():    
        i_zt = z + nT*T
        I_zt = I[i_zt]
        
        I_cyto = fluoCyto[T, z]
        I_back = fluoBack[T, z]
        
        contourXY = arrayViterbiContours[T, z, :, :]
        Xv, Yv = contourXY[:,0], contourXY[:,1]
        (Xcv, Ycv), Rcv = fitCircle(np.array([Yv, Xv]).T, loss = 'huber')
        Rv, Av = warpXY(Xv, Yv, Xcv, Ycv)
        Av = np.round(Av).astype(int)       
        
        #### 2.1 Warp
        warped = skm.transform.warp_polar(I_zt, center=(Ycv, Xcv), radius=np.mean(Rv)*2, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        warped = skm.util.img_as_uint(warped)
        # warped_n = (warped-I_back)/(I_cyto-I_back)

        for a in A_profiles:
            iz = z-np.min(ZZ)
            ia = a-np.min(AA)
            r = round(Rv[a])
            profiles[iz, ia] = warped[a, r-r_in:r+r_out]
    
    ColsProfiles = ['R_px', 'R_um'] + [f'I_z{z:.0f}_a{a:.0f}' for z, a in zip(ZZ.flatten(), AA.flatten())]
    DfProfiles = pd.DataFrame(np.zeros((nR, len(ColsProfiles))), columns=ColsProfiles)
    R_px = np.arange(r_out+r_in).astype(int) + (np.mean(Rv)-r_in)
    R_um = R_px*SCALE_100X_ZEN
    DfProfiles['R_px'] = R_px
    DfProfiles['R_um'] = R_um
    for z, a in zip(ZZ.flatten(), AA.flatten()):
        iz, ia = z-np.min(ZZ), a-np.min(AA)
        DfProfiles[f'I_z{z:.0f}_a{a:.0f}'] = profiles[iz, ia]
        
    if plot:
        figI, axI = plt.subplots(1,1, figsize = (6,6))
        axI.imshow(I[Z + nT*T])
        axI.plot([Xcv], [Ycv], 'bo')
        def plotLineAngle(ax, Xc, Yc, r_min, r_max, A, c='r', ls='-'):
            Xi, Yi = Xc + (r_min)*np.cos(A*np.pi/180), Yc + (r_min)*np.sin(A*np.pi/180)
            Xo, Yo = Xc + (r_max)*np.cos(A*np.pi/180), Yc + (r_max)*np.sin(A*np.pi/180)
            ax.plot([Xi, Xo], [Yi, Yo], c=c, ls=ls)
            
        plotLineAngle(axI, Xcv, Ycv, r-r_in, r+r_out, A, c='r', ls='-')
        
        fig, axes = plt.subplots(len(Z_profiles), len(A_profiles), figsize = (len(A_profiles)*3,len(Z_profiles)*3))
        for z, a in zip(ZZ.flatten(), AA.flatten()):
            print(z,a)
            iz, ia = z-np.min(ZZ), a-np.min(AA)
            ax = axes[iz, ia]
            I_cyto = fluoCyto[T, z]
            I_back = fluoBack[T, z]
            I_cytoPlusStd = fluoCyto[T, z] + fluoCytoStd[T, z]
            
            intProfile = DfProfiles[f'I_z{z:.0f}_a{a:.0f}'].values
            print(intProfile)
            
            def normalize(x):
                return((x-I_back)/(I_cyto-I_back))
            
    
            
            if normalizeInt:
                intProfile = normalize(intProfile)
                I_cyto = normalize(I_cyto)
                I_cytoPlusStd = normalize(I_cytoPlusStd)
                I_back = normalize(I_back)
            
            print(I_cytoPlusStd)
                
            ax.plot(R_px, intProfile)
            ax.axhline(I_cyto, c='r')
            # ax.axhline(I_cytoPlusStd, c='r', ls='--')
            ax.axhline(I_back, c='b')
            
            ax.set_title(f'z={z:.0f} - a={a:.0f}')
            
        fig.tight_layout()
        plt.show()
    
    return(DfProfiles)



def profileExplorer_avg(fluoPath, DfContact, fluoCyto, fluoBack, arrayViterbiContours,
                    T, Z, A, r_in, r_out, 
                    plot = False):
    I = skm.io.imread(fluoPath)
    czt_shape, czt_seq = get_CZT_fromTiff(fluoPath)
    dictContact = DfContact.set_index('iT').iloc[T,:].to_dict()
    [nC, nZ, nT] = czt_shape
    Z_profiles = np.arange(Z-1, Z+2)
    A_margin = 30
    A_profiles = np.arange(A-A_margin, A+A_margin)%360
    # A_profiles = np.arange(0,360)
    ZZ, AA = np.meshgrid(Z_profiles, A_profiles)
    nR = r_out + r_in
    
    profiles = np.zeros((len(Z_profiles), 360, nR))
    
    for z in ZZ.flatten():    
        i_zt = z + nT*T
        I_zt = I[i_zt]
        
        I_cyto = fluoCyto[T, z]
        I_back = fluoBack[T, z]
        
        contourXY = arrayViterbiContours[T, z, :, :]
        Xv, Yv = contourXY[:,0], contourXY[:,1]
        (Xcv, Ycv), Rcv = fitCircle(np.array([Yv, Xv]).T, loss = 'huber')
        Rv, Av = warpXY(Xv, Yv, Xcv, Ycv)
        Av = np.round(Av).astype(int)       
        
        #### 2.1 Warp
        warped = skm.transform.warp_polar(I_zt, center=(Ycv, Xcv), radius=np.mean(Rv)*2, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        warped = skm.util.img_as_uint(warped)

        for a in A_profiles:
            iz = z-np.min(ZZ)
            r = round(Rv[a])
            profiles[iz, a] = warped[a, r-r_in:r+r_out]
    
            
    profiles = profiles[:, A_profiles]
    
    ColsProfiles = ['R_px', 'R_um'] + [f'I_z{z:.0f}' for z in Z_profiles]
    DfProfiles = pd.DataFrame(np.zeros((nR, len(ColsProfiles))), columns=ColsProfiles)
    R_px = np.arange(r_out+r_in).astype(int) + (np.mean(Rv)-r_in)
    R_um = R_px/SCALE_100X_ZEN
    DfProfiles['R_px'] = R_px
    DfProfiles['R_um'] = R_um
    
    for z in Z_profiles:
        iz = z-np.min(Z_profiles)
        # print(z, iz)
        # print(np.max(profiles[iz, :, :]))
        # print(np.mean(profiles[iz, :, :], axis = 0).shape)
        DfProfiles[f'I_z{z:.0f}'] = np.mean(profiles[iz, :, :], axis = 0)
        
    if plot:
        figI, axI = plt.subplots(1,1, figsize = (6,6))
        axI.imshow(I[Z + nT*T])
        def plotLineAngle(ax, Xc, Yc, r_min, r_max, A, c='r', ls='-'):
            Xi, Yi = Xc + (r_min)*np.cos(A*np.pi/180), Yc + (r_min)*np.sin(A*np.pi/180)
            Xo, Yo = Xc + (r_max)*np.cos(A*np.pi/180), Yc + (r_max)*np.sin(A*np.pi/180)
            ax.plot([Xi, Xo], [Yi, Yo], c=c, ls=ls)
            
        # Xi, Yi = Xc + (r-r_in)*np.cos(A*np.pi/180), Yc + (r-r_in)*np.sin(A*np.pi/180)
        # Xo, Yo = Xc + (r+r_out)*np.cos(A*np.pi/180), Yc + (r+r_out)*np.sin(A*np.pi/180)
        axI.plot([Xcv], [Ycv], 'bo')
        plotLineAngle(axI, Xcv, Ycv, r-r_in, r+r_out, A, c='r', ls='-')
        plotLineAngle(axI, Xcv, Ycv, r-r_in, r+r_out, A-A_margin, c='r', ls='--')
        plotLineAngle(axI, Xcv, Ycv, r-r_in, r+r_out, A+A_margin, c='r', ls='--')
        
        
        fig, axes = plt.subplots(len(Z_profiles), 2, figsize = (8, len(Z_profiles)*3))
        for z in Z_profiles:
            I_cyto = fluoCyto[T, z]
            I_back = fluoBack[T, z]
            def normalize(x):
                return((x-I_back)/(I_cyto-I_back))
            
            iz = z-np.min(Z_profiles)
            ax = axes[iz, 0]
            ax.plot(R_px, DfProfiles[f'I_z{z:.0f}'])
            ax.axhline(I_cyto, c='r')
            ax.axhline(I_cyto + fluoCytoStd[T, z], c='r', ls='--')
            ax.axhline(I_back, c='b')
            
            ax = axes[iz, 1]
            ax.plot(R_px, normalize(DfProfiles[f'I_z{z:.0f}']))
            ax.axhline(1, c='r')
            ax.axhline(normalize(I_cyto + fluoCytoStd[T, z]), c='r', ls='--')
            ax.axhline(0, c='b')
            ax.set_title(f'z={z:.0f}')
            
        fig.tight_layout()
        plt.show()
        
    return(DfProfiles)

# Predict 1 pixel
# t = 1
# intensityMap_t = np.loadtxt(os.path.join(resFluoDir, f'{cellId}_Map_t{t+1:.0f}.txt'))
# tsd_t = tsdf.iloc[t,:].to_dict()
# contact_t = DfContact.iloc[t,:].to_dict()
# print(contact_t['Z']/dz)
# print(round(contact_t['Z']/dz))
# pos = (round(contact_t['Z']/dz), round(contact_t['A']))
# marginZ, marginA = 1, 1
# value_interp, iv = pixelGuesser(intensityMap_t, pos, marginZ, marginA, method='linear',
#                                   plot=True)

# Plot profiles
T = 0
Zc_um, Ac_deg = DfContact.loc[T, 'Z'], round(DfContact.loc[T, 'A'])
# Ac_deg = 270
Zc_i = round(Zc_um/dz)
r_in_um, r_out_um = 7, 5
r_in, r_out = round(r_in_um*SCALE_100X_ZEN), round(r_out_um*SCALE_100X_ZEN)

DfProfiles = profileExplorer(fluoPath, DfContact, fluoCyto, fluoBack, arrayViterbiContours,
                              T, Zc_i, Ac_deg, r_in, r_out, normalizeInt = True, plot = True)

# DfProfiles = profileExplorer_avg(fluoPath, DfContact, fluoCyto, fluoBack, arrayViterbiContours,
#                              T, Zc_i, Ac_deg, r_in, r_out, plot = True)


# %%%% 

    
    
# %%%% 

# dataDir = "D:/MicroscopyData/2023-12-06_SpinningDisc_3T3-LifeAct_JV/Session2_Test1"
# cellCode = 'C3'
# mapPath = os.path.join(dstDir, f'{cellCode}_MapIntensity.txt')
# mapCell = np.loadtxt(os.path.join(dstDir, f'{cellCode}_MapIntensity.txt'))
    

# Z0, A0 = (6, 200)
# pos = (Z0, A0)
# marginZ, marginA = 2, 3
# value_interp, iv = pixelGuesser2D(mapCell, pos, marginZ, marginA, method='linear',
#                                   plot=True)



# %%% Plot many

srcDir = 'D:/MagneticPincherData/Raw/24.02.27_Chameleon/Clean_Fluo_BF'
date = '24-02-27'
cells = ['C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11']
Dmoy = 4.4995

# %%%% Load files

list_df = []

for cell in cells:
    cellId = f'{date}_{cell}'
    
    
    # fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_Fluo.tif'
    bfName = f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_BF.tif'
    
    bfPath = os.path.join(srcDir, bfName)
    resBfDir = os.path.join(srcDir, 'Results_Tracker')
    resBfPath = os.path.join(resBfDir, bfName[:-4] + '_PY.csv')
    tsdf = pd.read_csv(resBfPath, sep=None, engine='python')
    
    fluoPath = os.path.join(srcDir, fluoName)
    resFluoDir =  os.path.join(srcDir, 'Results_Fluo', f'{cell}')
    cdf = pd.read_csv(os.path.join(resFluoDir, f'{cellId}_DfContact.csv'), sep=None, engine='python')
    
    czt_shape, czt_seq = get_CZT_fromTiff(fluoPath)
    [nC, nZ, nT] = czt_shape
    
    resCrossDir = os.path.join(srcDir, 'Results_Cross')
    
    mdf = pd.merge(cdf, tsdf, left_index=True, right_index=True)
    mdf.to_csv(os.path.join(resCrossDir, f'{cellId}_DfMerged.csv'), sep=';', index=False)
    
    nT = len(mdf)
    cell_col = np.array([cell]*nT)
    df = mdf[['D2', 'D3', 'F', 'I_viterbi_w5', 'I_viterbi_inout7-3', 'I_viterbi_flexW']]
    df['cell'] = cell_col
    list_df.append(df)


global_df = pd.concat(list_df)
global_df = global_df.drop(global_df[(global_df['cell'] == 'C5') & (global_df['D2'] < 4.6)].index)

global_df['h2'] = global_df['D2']-Dmoy
global_df['h3'] = global_df['D3']-Dmoy

global_df.to_csv(os.path.join(resCrossDir, f'{date}_global_DfMerged.csv'), sep=';', index=False)

# %%%% Plot - Lin



fig, axes = plt.subplots(1, 3, figsize=(14, 5))

ax = axes[0]
sns.scatterplot(ax=ax, data=global_df, x='h3', y='I_viterbi_w5', hue='cell', s= 75, zorder=5) # , style='cellNum'
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 2.5])
ax.set_ylabel('Cortex Intensity on a narrow width')
ax.set_xlabel('Cortex thickness (nm)')
ax.legend()
ax.grid()

ax = axes[1]
sns.scatterplot(ax=ax, data=global_df, x='h3', y='I_viterbi_inout7-3', hue='cell', s= 75, zorder=5) # , style='cellNum'
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 1.5])
ax.set_ylabel('Cortex Intensity on a large width')
ax.set_xlabel('Cortex thickness (nm)')
ax.legend()
ax.grid()

ax = axes[2]
sns.scatterplot(ax=ax, data=global_df, x='h3', y='I_viterbi_flexW', hue='cell', s= 75, zorder=5) # , style='cellNum'
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 100])
ax.set_ylabel('Cortex Intensity on a variable window')
ax.set_xlabel('Cortex thickness (nm)')
ax.legend()
ax.grid()


fig.tight_layout()
plt.show()


# %%%% Plot - Log

def mypowerlaw(X, a, b):
    return(np.exp(b)*X**a)

global_df['h2'] = global_df['D2']-Dmoy
global_df['h3'] = global_df['D3']-Dmoy

log_df = np.log(global_df[['D2', 'D3', 'F', 'I_viterbi_w5', 'I_viterbi_inout7-3', 'I_viterbi_flexW', 'h2', 'h3']])
log_df['cell'] = global_df['cell']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
sns.scatterplot(ax=ax, data=global_df, x='h3', y='I_viterbi_w5', hue='cell', s= 75, zorder=5) # , style='cellNum'
Xfit, Yfit = log_df['h3'].values, log_df['I_viterbi_w5'].values
params, res = ufun.fitLineHuber(Xfit, Yfit)
Xfitplot = np.linspace(np.min(global_df['h3'].values), np.max(global_df['h3'].values), 100)
ax.plot(Xfitplot, mypowerlaw(Xfitplot, params[1], params[0]), 'k--', label=f'Fit: Y = {np.exp(params[0]):.2f}.X^{params[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Cortex Intensity on a narrow width')
ax.set_xlabel('Cortex thickness (nm)')
ax.legend()
ax.grid()

ax = axes[1]
sns.scatterplot(ax=ax, data=global_df, x='h3', y='I_viterbi_inout7-3', hue='cell', s= 75, zorder=5) # , style='cellNum'
Xfit, Yfit = log_df['h3'].values, log_df['I_viterbi_inout7-3'].values
params, res = ufun.fitLineHuber(Xfit, Yfit)
Xfitplot = np.linspace(np.min(global_df['h3'].values), np.max(global_df['h3'].values), 100)
ax.plot(Xfitplot, mypowerlaw(Xfitplot, params[1], params[0]), 'k--', label=f'Fit: Y = {np.exp(params[0]):.2f}.X^{params[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Cortex Intensity on a large width')
ax.set_xlabel('Cortex thickness (nm)')
ax.legend()
ax.grid()

# ax = axes[2]
# sns.scatterplot(ax=ax, data=global_df, x='h3', y='I_viterbi_flexW', hue='cell', s= 75, zorder=5) # , style='cellNum'
# Xfit, Yfit = log_df['h3'].values, log_df['I_viterbi_w5'].values
# params, res = ufun.fitLineHuber(Xfit, Yfit)
# Xfitplot = np.linspace(np.min(Xfit), np.max(Xfit), 100)
# ax.plot(Xfitplot, Xfitplot*params[1]+params[0], 'k--', label=f'Fit: log(Y) = {params[1]:.2f}.log(X) + {params[0]:.2f}')
# ax.legend()
# ax.grid()


fig.tight_layout()
plt.show()


# %%%% Plot - Log - Density

def mypowerlaw(X, a, b):
    return(np.exp(b)*X**a)


global_df['Density_narrow'] = global_df['I_viterbi_w5']/global_df['h3']
global_df['Density_large'] = global_df['I_viterbi_inout7-3']/global_df['h3']


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
sns.scatterplot(ax=ax, data=global_df, x='h3', y='Density_narrow', hue='cell', s= 75, zorder=5) # , style='cellNum'
Xfit, Yfit = np.log(global_df['h3'].values), np.log(global_df['Density_narrow'].values)
params, res = ufun.fitLineHuber(Xfit, Yfit)
Xfitplot = np.linspace(np.min(global_df['h3'].values), np.max(global_df['h3'].values), 100)
ax.plot(Xfitplot, mypowerlaw(Xfitplot, params[1], params[0]), 'k--', label=f'Fit: Y = {np.exp(params[0]):.2f}.X^{params[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Cortex Density on a narrow width')
ax.set_xlabel('Cortex thickness (nm)')
ax.legend()
ax.grid()

ax = axes[1]
sns.scatterplot(ax=ax, data=global_df, x='h3', y='Density_large', hue='cell', s= 75, zorder=5) # , style='cellNum'
Xfit, Yfit = np.log(global_df['h3'].values), np.log(global_df['Density_large'].values)
params, res = ufun.fitLineHuber(Xfit, Yfit)
Xfitplot = np.linspace(np.min(global_df['h3'].values), np.max(global_df['h3'].values), 100)
ax.plot(Xfitplot, mypowerlaw(Xfitplot, params[1], params[0]), 'k--', label=f'Fit: Y = {np.exp(params[0]):.2f}.X^{params[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Cortex Density on a large width')
ax.set_xlabel('Cortex thickness (nm)')
ax.legend()
ax.grid()


fig.tight_layout()
plt.show()



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

# %% Tests

fig, ax = plt.subplots(1,1)
s = '$\\alpha\\beta$'

ax.set_title(s)

plt.show()

# %% As a function

def repeatAnalysis(date, cell):
    
    dictPaths = {'sourceDirPath' : 'D:/MagneticPincherData/Raw/24.02.27_Chameleon/Clean_Fluo_BF',
                 'imageFileName' : f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_BF.tif',
                 'resultsFileName' : f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_BF_Results.txt',
                 'depthoDir':'D:/MagneticPincherData/Raw/DepthoLibrary',
                 'depthoName':'24.02.27_Chameleon_M450_step20_100X',
                 'resultDirPath' : 'D:/MagneticPincherData/Raw/24.02.27_Chameleon/Clean_Fluo_BF/Results_Tracker',
                 }
    
    
    # Define constants
    
    dictConstants = {'microscope' : 'labview',
                     #
                     'inside bead type' : 'M450-2025', # 'M450' or 'M270'
                     'inside bead diameter' : 4493, # nm
                     'inside bead magnetization correction' : 0.969, # nm
                     'outside bead type' : 'M450-Strept', # 'M450' or 'M270'
                     'outside bead diameter' : 4506, # nm
                     'outside bead magnetization correction' : 1.056, # nm
                     #
                     'normal field multi images' : 11, # Number of images
                     'multi image Z step' : 400, # nm
                     'multi image Z direction' : 'downward', # Either 'upward' or 'downward'
                     #
                     'scale pixel per um' : 7.4588, # pixel/µm
                     'optical index correction' : 0.875, # ratio, without unit
                     'beads bright spot delta' : 0, # Rarely useful, do not change
                     'magnetic field correction' : 1.0, # ratio, without unit
                     }
    
    
    # Additionnal options
    
    dictOptions = {'redoAllSteps' : True,
                   'trackAll' : False,
                   'importLogFile' : True,
                   'saveLogFile' : True,
                   'saveFluo' : False,
                   'importTrajFile' : True,
                   'expandedResults' : True,
                  }
    
    # Make metaDataFrame # NEED TO CODE STH HERE !
    
    OMEfilepath = os.path.join(dictPaths['sourceDirPath'], f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_OME.txt')
    
    CTZ_tz = OMEDataParser(OMEfilepath)
    BF_T = CTZ_tz[0,:,:,0].flatten()
    
    NFrames = len(BF_T)
    B_set = 5 * np.ones(NFrames)
    
    loopStructure = [{'Status':'Passive',     'NFrames':NFrames}]  # Only one constant field phase
                    
    metaDf = makeMetaData(BF_T, B_set, loopStructure)
    
    # Call mainTracker()
    
    # tsdf = smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)
    
    
    # Mappemonde fluo
    
    # Paths
    
    srcDir = 'D:/MagneticPincherData/Raw/24.02.27_Chameleon/Clean_Fluo_BF'
    fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_Fluo.tif'
    # fileList = os.listdir(srcDir)
    
    dstDir =  os.path.join(srcDir, 'Results_Fluo', f'{cell}')
    fluoPath = os.path.join(srcDir, fluoName)
    
    figSavePath = dstDir
    cellId = f'{date}_{cell}'
    
    
    bfName = f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_BF.tif'
    bfPath = os.path.join(srcDir, bfName)
    resBfDir = os.path.join(srcDir, 'Results_Tracker')
    resBfPath = os.path.join(resBfDir, bfName[:-4] + '_PY.csv')
    
    # Options
    
    framesize = int(35*8/2) 
    inPix_set, outPix_set = 20, 20
    blur_parm = 4
    relative_height_virebi = 0.4
    interp_factor_x = 5
    interp_factor_y = 10
    VEW = 5 # Viterbi edge width # Odd number
    VE_in, VE_out = 7, 3 # Viterbi edge in/out
    
    Fluo_T = np.mean(CTZ_tz[1,:,:,0], axis=1)
    
    SAVE_RESULTS = True
    PLOT_WARP = False
    PLOT_MAP = False
    
    optical_index_correction = 0.875
    dz = 0.4  * optical_index_correction # µm # Distance between each image of the stacks
    DZb = 4.5 * optical_index_correction # µm # Distance between focal point of the deptho and the equatorial plane of the beads #### SHOULD BE RE-MEASURED
    DZo = 4.0 * optical_index_correction # µm # Distance between the stack of BF images and the corresponding stack of Fluo images
    Rbead = 4.5/2
    
    # Compute map
    
    
    
    #### Start
    
    plt.ioff()
    I = skm.io.imread(fluoPath)
    
    czt_shape, czt_seq = get_CZT_fromTiff(fluoPath)
    [nC, nZ, nT] = czt_shape
    Angles = np.arange(0, 360, 1)
    
    tsdf = pd.read_csv(resBfPath, sep=None, engine='python')
    
    ColsContact = ['iT', 'T', 'X', 'Y', 'Z', 'R', 'A', 'Xc', 'Yc', 'Rc', 
                   'Xin', 'Yin', 'Zin', 'Xout', 'Yout', 'Zout', 
                   f'I_viterbi_w{VEW:.0f}', f'I_viterbi_inout{VE_in:.0f}-{VE_out:.0f}', 'I_viterbi_flexW', 'W_viterbi_flexW']
    
    DfContact = pd.DataFrame(np.zeros((nT, len(ColsContact))), columns=ColsContact)
    fluoCell = np.zeros((nT, nZ))
    fluoCyto = np.zeros((nT, nZ))
    fluoCytoStd = np.zeros((nT, nZ))
    fluoBack = np.zeros((nT, nZ))
    arrayViterbiContours = np.zeros((nT, nZ, 360, 2))
    
    for t in range(nT):
        
        intensityMap_t = []
        tsd_t = tsdf.iloc[t,:].to_dict()
            
        #### 1.0 Compute position of contact point
        x_contact = (tsd_t['X_in']+tsd_t['X_out'])/2
        y_contact = (tsd_t['Y_in']+tsd_t['Y_out'])/2
        
        zr_contact = (tsd_t['Zr_in']+tsd_t['Zr_out'])/2
        CF_mid = - zr_contact + DZo - DZb 
        # EF = ED + DI + IF; ED = - DZb; for I=I_mid & F=F_mid, DI = - zr; and IF = DZo
        iz_mid = (nZ-1)//2 # If 11 points in Z, iz_mid = 5
        iz_contact = iz_mid - (CF_mid / dz)
        # print(iz_contact)
        
        x_in = tsd_t['X_in']
        y_in = tsd_t['Y_in']
        z_in = ((-1)*tsd_t['Zr_in']) + DZo - DZb
        
        x_out = tsd_t['X_out']
        y_out = tsd_t['Y_out']
        z_out = ((-1)*tsd_t['Zr_out']) + DZo - DZb
        
        DfContact.loc[t, ['iT', 'T', 'X', 'Y', 'Z']] = [t, Fluo_T[t], x_contact, y_contact, iz_contact*dz]
        DfContact.loc[t, ['Xin', 'Yin', 'Zin', 'Xout', 'Yout', 'Zout']] = [x_in, y_in, dz*iz_mid-z_in, x_out, y_out, dz*iz_mid-z_out]
    
        
        # Zr = (-1) * (tsd_t['Zr_in']+tsd_t['Zr_out'])/2 # Taking the opposite cause the depthograph is upside down for these experiments
        # izmid = (nZ-1)//2 # If 11 points in Z, izmid = 5
        # zz = dz * np.arange(-izmid, +izmid, 1) # If 11 points in Z, zz = [-5, -4, ..., 4, 5] * dz
        # zzf = zz + Zr # Coordinates zz of bf images with respect to Zf, the Z of the focal point of the deptho
        # zzeq = zz + Zr + DZo - DZb # Coordinates zz of fluo images with respect to Zeq, the Z of the equatorial plane of the bead
        
        # zi_contact = Akima1DInterpolator(zzeq, np.arange(len(zzeq)))(0)
        
        # contactSlice = False
        A_contact_map = 0
        foundContact = False
        
        for z in range(nZ):
            contactSlice = False
            i_zt = z + nT*t
            I_zt = I[i_zt]
            
            #### 2.0 Locate cell
            (Yc, Xc), Rc = findCellInnerCircle(I_zt, withBeads = True, plot=False)
            Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
            
            #### 2.1 Warp
            warped = skm.transform.warp_polar(I_zt, center=(Yc, Xc), radius=Rc*1.2, 
                                              output_shape=None, scaling='linear', channel_axis=None)
            
            max_values = np.max(warped[:,Rc0-inPix_set:Rc0+outPix_set+1], axis=1)
                
            #### 2.2 Viterbi Smoothing
            edge_viterbi = ViterbiEdge(warped, Rc0, inPix_set, outPix_set, 2, relative_height_virebi)
            edge_viterbi_unwarped = np.array(unwarpRA(np.array(edge_viterbi), Angles, Xc, Yc)) # X, Y
            
            # fig, ax = plt.subplots(1, 1)
            # ax.imshow(warped, cmap='gray', aspect='equal')
            # ax.plot(edge_viterbi, Angles, 'r--')
            # plt.show()
    
    
            #### 3.0 Locate cell
            (Xc, Yc), Rc = fitCircle(np.array([edge_viterbi_unwarped[1], edge_viterbi_unwarped[0]]).T, loss = 'huber')
            Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
            DfContact.loc[t, ['Xc', 'Yc', 'Rc']] += np.array([(Xc/SCALE_100X_ZEN)/(nZ), (Yc/SCALE_100X_ZEN)/(nZ), (Rc/SCALE_100X_ZEN)/(nZ)])
            
            # fig, ax = plt.subplots(1, 1)
            # ax.imshow(I_zt, cmap='gray', aspect='equal')
            # ax.plot(edge_viterbi_unwarped[0], edge_viterbi_unwarped[1], 'r--')
            # ax.plot(Xc, Yc, 'bo')
            # circle_test = circleContour_V2((Yc, Xc), Rc)
            # ax.plot(circle_test[:,1], circle_test[:,0], 'g--')
            # plt.show()
            
            #### 3.1 Warp
            warped = skm.transform.warp_polar(I_zt, center=(Yc, Xc), radius=Rc*1.4, 
                                              output_shape=None, scaling='linear', channel_axis=None)
            warped = skm.util.img_as_uint(warped)
            w_ny, w_nx = warped.shape
            Angles = np.arange(0, 360, 1)
            max_values = np.max(warped[:,Rc0-inPix_set:Rc0+outPix_set+1], axis=1)
            
            
            R_contact, A_contact = warpXY(x_contact*SCALE_100X_ZEN, y_contact*SCALE_100X_ZEN, Xc, Yc)
                    
            if (z == np.round(iz_contact)) or (not foundContact and z == nZ-1):
                contactSlice = True
                foundContact = True
                
                A_contact_map = A_contact
                DfContact.loc[t, ['R', 'A']] = [R_contact, A_contact]
            
            #### 3.2 Interp in X
            warped_interp = ufun.resize_2Dinterp(warped, fx=interp_factor_x, fy=1)
            
            warped = warped_interp
            w_nx, Rc0 = w_nx * interp_factor_x, Rc0 * interp_factor_x
            inPix, outPix = inPix_set * interp_factor_x, outPix_set * interp_factor_x
            
            R_contact = R_contact * interp_factor_x
            
            #### 3.3 Viterbi Smoothing
            edge_viterbi = ViterbiEdge(warped, Rc0, inPix, outPix, blur_parm, relative_height_virebi)
            edge_viterbi_unwarped = unwarpRA(np.array(edge_viterbi)/interp_factor_x, Angles, Xc, Yc)
            
            arrayViterbiContours[t, z, :] = np.array([edge_viterbi_unwarped[0], edge_viterbi_unwarped[1]]).T
            
            #### 4. Analyse fluo cyto & background
            # if COMPUTE_FLUO_INOUT:
            path_edge = mpltPath.Path(np.array(edge_viterbi_unwarped).T)
            xx, yy = np.meshgrid(np.arange(I_zt.shape[1]), np.arange(I_zt.shape[0]))
            pp = np.array([[x, y] for x, y in zip(xx.flatten(), yy.flatten())]) # .reshape(I_zt.shape[1], I_zt.shape[0], 2)
            inside = path_edge.contains_points(pp).reshape(I_zt.shape[0], I_zt.shape[1])
            
            # Inside
            cyto_margin = 10
            inside_eroded = ndi.binary_erosion(inside, iterations = cyto_margin)
            data_cyto = I_zt.flatten()[inside_eroded.flatten()]
            threshold_cyto = skm.filters.threshold_li(data_cyto)
            # threshold_cell = skm.filters.threshold_triangle(data_cyto) #[0.9*skm.filters.threshold_triangle(data_cell)]
            mask_cyto = (I_zt > 0.9*threshold_cyto).astype(bool)
            I_cyto = np.median(I_zt[mask_cyto & inside_eroded])
            Std_cyto = np.std(I_zt[mask_cyto & inside_eroded])
            
            # Outside
            background_margin = 140
            outside = ~ndi.binary_dilation(inside, iterations = background_margin)
            data = I_zt.flatten()[outside.flatten()]
            I_back = np.percentile(I_zt[outside], 30)
            
            # Whole cell
            cell_margin = 3
            cell_dilated = ndi.binary_dilation(inside, iterations = cell_margin)
            
            # hist, bins = np.histogram(data_cyto.ravel(), bins=255)
            # hist_s = ndi.gaussian_filter1d(hist, sigma=2)
            # peaks, props = signal.find_peaks(hist_s, prominence = 10, distance = 15, height = 0)
            # i_maxHeight = np.argmax(props['peak_heights'])
            # test_nPeaks = (i_maxHeight > 0)
            # test_doublePeaks = (len(peaks)==2 and min(props['peak_heights'])>60)
            # threshold_cell = 0
            # if test_nPeaks or test_doublePeaks:
            #     threshold_cell = skm.filters.threshold_minimum(hist = (hist, bins))
            #     if np.sum((cell_dilated.astype(bool) * (I_zt > 0.95*threshold_cell).astype(bool)).astype(int)) < 5000:
            #         threshold_cell = 0
                        
            # cell_filtered = cell_dilated.astype(bool) * (I_zt > 0.95*threshold_cell).astype(bool)
            
    
            mask_removed_from_cyto = inside & (~mask_cyto)
            cell_filtered = cell_dilated & (~mask_removed_from_cyto)
            
            nb_pix = np.sum(cell_filtered.astype(int))
            total_val = np.sum(I_zt.flatten()[cell_filtered.flatten()])
            I_cell = total_val/nb_pix
            
            # Store
            fluoCyto[t, z] = I_cyto
            fluoCytoStd[t, z] = Std_cyto
            fluoBack[t, z] = I_back
            fluoCell[t, z] = I_cell
            
            # Display
            # figT, axesT = plt.subplots(1,4, figsize = (16, 5))
            # ax = axesT[0]
            # ax.imshow(I_zt)
            # ax.set_title('Original')
            # ax = axesT[1]
            # ax.imshow((mask_cyto & inside_eroded).astype(int))
            # ax.set_title('Cytoplasm')
            # ax = axesT[2]
            # ax.imshow(cell_filtered.astype(int))
            # ax.set_title('Whole cell')
            # ax = axesT[3]
            # ax.imshow(outside.astype(int))
            # ax.set_title('Background')
            # # # hist, bins, patches = ax.hist(data_cyto.ravel(), bins=255)
            # # # peaks, props = signal.find_peaks(hist, prominence = 40, distance = 20)
            # # ax.plot(bins[:-1], hist_s, 'g-')
            # # ax.hist(data_cyto.ravel(), bins=255)
            # # if len(peaks) <= 1:
            # #     threshold_cyto2 = 0
                
            # # ax.axvline(threshold_cell, color='r')
            # # for p in peaks:
            # #     val = bins[p]
            # #     ax.axvline(val, color='gold')
                
            # # testI = inside.astype(int) + inside_eroded.astype(int) + (inside_eroded*mask_cyto).astype(int)
            # # testI -= outside.astype(int)
            # # testI = np.sum([(I_zt > th).astype(int) for th in thresholds_cell], axis=0)
            # figT.tight_layout()
            # plt.show()
            
            def normalize_Icyto(x):
                return((x - I_back)/(I_cyto - I_back))
        
            def normalize_Icell(x):
                return((x - I_back)/(I_cell - I_back))
            
            def normalize_Icell2(x):
                return((x - I_cyto)/(I_cell - I_back))
            
            # warped_n = (warped-I_back) / (I_cyto-I_back)
            
            #### 5. Read values
            max_values = normalize_Icell(max_values)
            
            viterbi_Npix_values = [np.sum(warped[Angles[i], edge_viterbi[i]-VEW//2:edge_viterbi[i] + (VEW//2 + 1)])/VEW for i in range(len(edge_viterbi))]
            viterbi_Npix_values = normalize_Icell(viterbi_Npix_values)
            intensityMap_t.append(viterbi_Npix_values)
            
            if contactSlice:
                print(t, z, R_contact, A_contact)
                
                iA_contact = round(A_contact)
                DfContact.loc[t, f'I_viterbi_w{VEW:.0f}'] = viterbi_Npix_values[iA_contact]
                
                viterbi_inout_value = np.sum(warped[iA_contact-1:iA_contact+2, 
                                                    edge_viterbi[iA_contact]-VE_in*interp_factor_x:edge_viterbi[iA_contact]+VE_out*interp_factor_x])/(3*(VE_in+VE_out)*interp_factor_x)
                viterbi_inout_value = normalize_Icell(viterbi_inout_value)
                DfContact.loc[t, f'I_viterbi_inout{VE_in:.0f}-{VE_out:.0f}'] = viterbi_inout_value
                
                
            
                #### X. viterbi_flex_value
                viterbi_flex_value = 0
                angleOffsets = [-1,0,1]
                
                # figW, axesW = plt.subplots(len(angleOffsets), 1, figsize = (5, 4*len(angleOffsets)))
                
                for jj in range(len(angleOffsets)):
                    iAngle = angleOffsets[jj]
                    r_edge = edge_viterbi[iA_contact+iAngle]
                    viterbiContactProfile = normalize_Icell2(warped[iA_contact+iAngle, :])
                    
                    r_init  = r_edge - ufun.findFirst(True, viterbiContactProfile[:r_edge][::-1] < 0) # I_cyto is set to 0 with normalize_Icell2
                    r_final = r_edge + ufun.findFirst(True, viterbiContactProfile[r_edge:] < 0) # I_cyto is set to 0 with normalize_Icell2
                    
                    viterbi_flex_value += np.sum(viterbiContactProfile[r_init:r_final]) * (1/len(angleOffsets))
                    
                    # try:
                    #     ax = axesW[jj]
                    # except:
                    #     ax = axesW
                    # ax.plot(np.arange(len(viterbiContactProfile)), (normalize_Icell(warped[iA_contact+iAngle, :])))
                    # ax.axhline(normalize_Icell(I_cyto), c='r')
                    # # ax.axhline(I_cyto+Std_cyto, c='r', ls = '--')
                    # ax.axhline(normalize_Icell(I_cell), c='g')
                    # ax.axhline(normalize_Icell(I_back), c='b')
                    # ax.axvline(r_edge, c='gold')
                    # ax.axvline(r_init, c='gold', ls = '--')
                    # ax.axvline(r_final, c='gold', ls = '--')
                
                # print(viterbi_flex_value)
                # viterbi_inout_value = normalize_Icell(viterbi_inout_value)
                # print(viterbi_inout_value)
                
                DfContact.loc[t, 'I_viterbi_flexW'] = viterbi_flex_value         
                DfContact.loc[t, 'W_viterbi_flexW'] = r_final-r_init  
                    
                # figW.tight_layout()
                # plt.show()
            
            
            #### 6. Plot
            #### 6.1 Compute beads positions
            draw_InCircle = False
            draw_OutCircle = False
            
            if PLOT_WARP:
                h_InCircle = Rbead + (z_in + dz*(z-iz_mid))
                h_OutCircle = Rbead + (z_out + dz*(z-iz_mid))
                
                if h_InCircle > 0 and h_InCircle < 2*Rbead:
                    draw_InCircle = True
                    a_InCircle = (h_InCircle*(2*Rbead-h_InCircle))**0.5
                    # INbeadContour = circleContour((round(y_in*SCALE_100X_ZEN), round(x_in*SCALE_100X_ZEN)), 
                    #                                a_InCircle*SCALE_100X_ZEN, I_zt.shape)
                    INbeadContour = circleContour_V2((round(y_in*SCALE_100X_ZEN), round(x_in*SCALE_100X_ZEN)), 
                                                     a_InCircle*SCALE_100X_ZEN)
                    XINbeadContour, YINbeadContour = INbeadContour[:, 1], INbeadContour[:, 0]
                    RINbeadContour, AINbeadContour = warpXY(XINbeadContour, YINbeadContour, Xc, Yc)
                    RINbeadContour *= interp_factor_x
                    
                if h_OutCircle > 0 and h_OutCircle < 2*Rbead:
                    draw_OutCircle = True
                    a_OutCircle = (h_OutCircle*(2*Rbead-h_OutCircle))**0.5
                    # OUTbeadContour = circleContour((round(y_out*SCALE_100X_ZEN), round(x_out*SCALE_100X_ZEN)), 
                    #                                a_OutCircle*SCALE_100X_ZEN, I_zt.shape)
                    OUTbeadContour = circleContour_V2((round(y_out*SCALE_100X_ZEN), round(x_out*SCALE_100X_ZEN)),
                                                      a_OutCircle*SCALE_100X_ZEN)
                    XOUTbeadContour, YOUTbeadContour = OUTbeadContour[:, 1], OUTbeadContour[:, 0]
                    ROUTbeadContour, AOUTbeadContour = warpXY(XOUTbeadContour, YOUTbeadContour, Xc, Yc)
                    ROUTbeadContour *= interp_factor_x   
                    
            #### 6.2 Warp and Viterbi   
            if PLOT_WARP:
                # Init the fig
                figV2 = plt.figure(figsize=(20, 10))#, layout="constrained")
                spec = figV2.add_gridspec(2, 7)
                
                inBorder = Rc0 - inPix
                outBorder = Rc0 + outPix
                Rc0A = np.ones(360)*Rc0
                inBorderA = np.ones(360)*inBorder
                outBorderA = np.ones(360)*outBorder
                
                
                # 1.1 - Show the cell
                ax = figV2.add_subplot(spec[0, :2])
                ax.imshow(I_zt, cmap='gray', aspect='equal')
                ax.set_title('Raw Image')
                
                # 1.2 - Show the cell with annotations
                ax = figV2.add_subplot(spec[1, :2])
                ax.imshow(I_zt, cmap='gray', aspect='equal')
                ax.plot(edge_viterbi_unwarped[0], edge_viterbi_unwarped[1], c='g', ls = '--')
                if draw_InCircle:
                    ax.plot(XINbeadContour, YINbeadContour, c='r', ls=':')
                if draw_OutCircle:
                    ax.plot(XOUTbeadContour, YOUTbeadContour, c='gold', ls=':')
                ax.set_title('Detected contours')
                
                # 2.1 - Show the warped cell
                ax1 = figV2.add_subplot(spec[:, 2])
                ax = ax1
                ax.imshow(warped[:, :], cmap='gray', aspect='auto')
                ax.set_title('Raw warp')
                ax.set_ylabel('Angle (deg)')
                ax.set_xlabel('Radial pix')
                
                # 2.2 - Show the warped cell with viterbi detection
                ax = figV2.add_subplot(spec[:, 3], sharey = ax1)
                ax.imshow(warped[:, :], cmap='gray', aspect='auto')
                ax.plot(Rc0A, Angles, ls='--', c='red', lw=1)
                ax.plot(inBorderA, Angles, ls='--', c='gold', lw=1)
                ax.plot(outBorderA, Angles, ls='--', c='gold', lw=1)
                ax.plot(edge_viterbi, Angles, ls='', c='cyan', marker='.', ms = 2)
                ax.plot(R_contact, A_contact, 'r+', ms=5)
                if draw_InCircle:
                    ax.plot(RINbeadContour, AINbeadContour, c='b', ls=':')
                if draw_OutCircle:
                    mask = (ROUTbeadContour < w_nx)
                    ROUTbeadContour = ROUTbeadContour[mask]
                    AOUTbeadContour = AOUTbeadContour[mask]
                    ax.plot(ROUTbeadContour, AOUTbeadContour, c='cyan', ls=':')
                ax.set_title('Viterbi Edge')
                ax.set_xlabel('Radial pix')
                
                # 3.1 - Show the intensity profiles for Viterbi
                ax = figV2.add_subplot(spec[:, 4], sharey = ax1)
                step1 = 5
                ax.plot(inBorderA, Angles, c='k', ls='--', marker='')
                ax.plot(outBorderA, Angles, c='k', ls='--', marker='')
                ax.plot(edge_viterbi[::step1], Angles[::step1], ls='', c='cyan', marker='^', ms = 4, mec = 'k', zorder=5)
                ax.set_title('Viterbi profiles')
                ax.set_xlabel('Radial pix')
                
                cList_viridis = plt.cm.viridis(np.linspace(0.05, 0.95, w_ny//step1))
                ax.set_prop_cycle(plt.cycler("color", cList_viridis))
                maxWarped = np.max(warped)
                normWarped = warped * 6 * step1/maxWarped
                for a in Angles[::step1]:            
                    profile = normWarped[a, :] - np.min(normWarped[a, :])
                    RR = np.arange(len(profile))
                    ax.plot(RR, a - profile)
    
                # 3.2 - Zoom on the intensity profiles for Viterbi
                ax = figV2.add_subplot(spec[:, 5], sharey = ax1)
                step2 = 20
                ax.plot(edge_viterbi[::step2], Angles[::step2], ls='', c='cyan', marker='^', ms = 4, mec = 'k', zorder=5)
                cList_viridis = plt.cm.viridis(np.linspace(0.05, 0.95, w_ny//step2))
                ax.set_prop_cycle(plt.cycler("color", cList_viridis))
                ax.set_title('Viterbi profiles')
                ax.set_xlabel('Radial pix')
                maxWarped = np.max(warped)
                normWarped = warped * 6 * step2/maxWarped
                for a in Angles[::step2]:            
                    profile = normWarped[a, Rc0-inPix:Rc0+outPix] - np.min(normWarped[a, Rc0-inPix:Rc0+outPix])
                    RR = np.arange(Rc0-inPix, Rc0+outPix, 1)
                    ax.plot(RR, a - profile)
    
                # 4 - Show the intensity along the cell contour
                ax = figV2.add_subplot(spec[:, 6], sharey = ax1)
                ax.set_title('Profile Values')
                ax.plot(max_values, Angles, 'r-', label = 'Max profile')
                ax.plot(viterbi_Npix_values, Angles, 'b-', label = 'Viterbi profile')
                ax.legend()
                
                figV2.suptitle(f'Warp - {cellId} - t = {t+1:.0f}/{nT:.0f} - z = {z+1:.0f}/{nZ:.0f}')
                figName = f'{cellId}_Warp_t{t+1:.0f}_z{z+1:.0f}'
                figV2.tight_layout()
                ufun.simpleSaveFig(figV2, figName, figSavePath, '.png', 150)
                plt.close('all')
        
        intensityMap_t = np.array(intensityMap_t).astype(float)
        np.savetxt(os.path.join(dstDir, f'{cellId}_Map_t{t+1:.0f}.txt'), intensityMap_t, fmt='%.5f')
        
        if PLOT_MAP:
            #### 6.3 Map and contact point
            figM, axM = plt.subplots(1, 1, figsize = (16, 2))
            ax = axM
            im = ax.imshow(intensityMap_t, cmap = 'viridis', vmin=0, origin='lower')
            # figM.colorbar(im, ax=ax, orientation='horizontal', fraction=.1, aspect=80)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            figM.colorbar(im, cax=cax, orientation='vertical', aspect=20) 
            ax.set_title(f'Map - {cellId} - t = {t+1:.0f}/{nT:.0f}')
            ax.plot(A_contact_map, iz_contact+0.5, 'r+', ms=7)
            figM.tight_layout()
            figName = f'{cellId}_Map_t{t+1:.0f}'
            ufun.simpleSaveFig(figM, figName, figSavePath, '.png', 150)
            # plt.show()
            plt.close('all')
    
    # np.savetxt(os.path.join(dstDir, f'{cellId}_Contact-XYZ.txt'), contactXYZ)        
           
    plt.ion()
    if SAVE_RESULTS:
        DfContact.to_csv(os.path.join(dstDir, f'{cellId}_DfContact.csv'), sep=';', index=False)
        np.save(os.path.join(dstDir, f'{cellId}_xyContours.npy'), arrayViterbiContours)
        np.savetxt(os.path.join(dstDir, f'{cellId}_fluoCyto.txt'), fluoCyto, fmt='%.1f')
        np.savetxt(os.path.join(dstDir, f'{cellId}_fluoCytoStd.txt'), fluoCytoStd, fmt='%.3f')
        np.savetxt(os.path.join(dstDir, f'{cellId}_fluoBack.txt'), fluoBack, fmt='%.1f')
        np.savetxt(os.path.join(dstDir, f'{cellId}_fluoCell.txt'), fluoCell, fmt='%.1f')
    
# %% Run function
date = '24-02-27'
cells = ['C6', 'C7', 'C8', 'C9', 'C10', 'C11']

for cell in cells:
    print(cell)
    repeatAnalysis(date, cell)


# %% Outdated code

# %%% findCellInnerCircle_withBead


def findCellInnerCircle_withBead(I0, plot=False):
    th1 = skm.filters.threshold_isodata(I0)
    I0_bin = (I0 > th1)
    I0_bin = ndi.binary_opening(I0_bin, iterations = 2)
    I0_label, num_features = ndi.label(I0_bin)
    df = pd.DataFrame(skm.measure.regionprops_table(I0_label, I0, properties = ['label', 'area']))
    df = df.sort_values(by='area', ascending=False)
    i_label = df.label.values[0]
    I0_rawCell = (I0_label == i_label)
    I0_rawCell = ndi.binary_fill_holes(I0_rawCell)
        
    #### CORRECTION DUE TO BEAD PRESENCE
    I0_robust = ndi.binary_closing(I0_rawCell, iterations = 20)
    [contour_corr] = skm.measure.find_contours(I0_robust, 0.5)
    (xc, yc), rc = fitCircle(contour_corr)
    circle_fit = shapely.Point((xc, yc)).buffer(rc)
    mask = circularMask(I0.shape, (yc, xc), rc)
    [contour_circularMask] = skm.measure.find_contours(mask, 0.5)
    beadshape = (~I0_rawCell) & mask
    beadshape = ndi.binary_opening(beadshape, iterations = 3)
    beadshape = ndi.binary_fill_holes(beadshape)
    beadshape_label, num_features = ndi.label(beadshape)
    # beadshape_df = pd.DataFrame(skm.measure.regionprops_table(beadshape_label, 
    #                                                           properties = ['label', 'area']))
    # beadshape_df = beadshape_df.sort_values(by='area', ascending=False)
    # beadshape_i_label = beadshape_df.label.values[0]
    
    fig0, axes0 = plt.subplots(1,1, figsize = (4,4))
    axes0.imshow(I0_robust, cmap='gray')
    
    for i_feat in range(num_features):
        beadshape_i = (beadshape_label == i_feat)
        beadshape_r = distFromCenter((yc, xc), beadshape_i)
        confirmedBead = (np.min(beadshape_r) < 0.8*rc) and (np.sum(beadshape_i) > 10)
        if confirmedBead:
            beadshape_i = ndi.binary_dilation(beadshape_i, iterations = 2)
            I0_rawCell = I0_rawCell | beadshape_i
            I0_rawCell = ndi.binary_fill_holes(I0_rawCell)
            [contour_beadshape_i] = skm.measure.find_contours(beadshape_i, 0.5)
            axes0.plot(contour_beadshape_i[:,1], contour_beadshape_i[:,0], 'b--')
            
    plot_polygon(circle_fit, ax=axes0, add_points=False, color='red')        
    axes0.plot(contour_circularMask[:,1], contour_circularMask[:,0], 'r--')
    plt.show()
    #### END OF CORRECTION DUE TO BEAD PRESENCE
    
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
            # plot_polygon(circle_fit, ax=ax, add_points=False, color='blue')
            # ax.plot(contour_circularMask[:,1], contour_circularMask[:,0], 'b--')
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