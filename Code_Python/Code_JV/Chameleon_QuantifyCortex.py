# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:16:35 2024

@author: JosephVermeil
"""

# %% Imports

import os
import re

import numpy as np
import pandas as pd
import skimage as skm
import scipy.ndimage as ndi

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from PIL import Image
from PIL.TiffTags import TAGS

from scipy import interpolate, signal, optimize
# from scipy.interpolate import Akima1DInterpolator # CubicSpline, PchipInterpolator

import shapely
from shapely.ops import polylabel
from shapely.plotting import plot_polygon, plot_points, plot_line
# from shapely import Polygon, LineString, get_exterior_ring, distance

from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

#### Local imports
import sys
sys.path.append('C:/Users/JosephVermeil/Desktop/CortExplore/Code_Python')
sys.path.append('C:/Users/JosephVermeil/Desktop/CortExplore/Code_Python/Code_JV')

import GraphicStyles as gs
import UtilityFunctions as ufun
from Chameleon_BeadTracker import smallTracker

gs.set_mediumText_options_jv()

SCALE_100X_ZEN = 7.4588


def dateFormat(d):
    d2 = d[0:2]+'.'+d[3:5]+'.'+d[6:8]
    return(d2)


# %% Utility functions


def filterDf(df, F):
    F = np.array(F)
    totalF = np.all(F, axis = 0)
    df_f = df[totalF]
    return(df_f)


# %%% For image processing

def autocorr_np(x):
    """
    1D autocorrelation of signal in array x
    """
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
    """
    Find the argument of the median value in array x.
    """
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


def ViterbiPathFinder(treillis):
    """
    To understand the idea, see : https://www.youtube.com/watch?v=6JVqutwtzmo
    """
    
    def node2node_distance(node1, node2):
        """Define a distance between nodes of the treillis graph"""
        d = (np.abs(node1['pos'] - node2['pos']))**2 #+ np.abs(node1['val'] - node2['val'])
        return(d)
    
    N_row = len(treillis)
    for i in range(1, N_row):
        current_row = treillis[i]
        previous_row = treillis[i-1]
        
        if len(current_row) == 0:
            current_row = [node.copy() for node in previous_row]
            treillis[i] = current_row
            
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

#### Test of ViterbiPathFinder

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

def ViterbiEdge(warped, Rc, inPix, outPix, blur_parm, relative_height_virebi):
    """
    Wrapper around ViterbiPathFinder
    Use the principle of the Viterbi algorithm to smoothen the contour of the cell on a warped image.
    To understand the idea, see : https://www.youtube.com/watch?v=6JVqutwtzmo
    """
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


def circularMask(shape, center, radius):
    ny, nx  = shape
    yc, xc = center
    x, y = np.arange(0, nx), np.arange(0, ny)
    mask = np.zeros((nx, ny), dtype=bool)
    pos = (x[np.newaxis,:]-xc)**2 + (y[:,np.newaxis]-yc)**2 < radius**2
    mask[pos] = True
    return(mask)

# def distFromCenter(center, mask):
#     yc, xc = center
#     ny, nx = mask.shape
#     x, y = np.arange(0, nx), np.arange(0, ny)
#     xx, yy = np.meshgrid(x, y)
#     D = np.sqrt(np.square(yy - yc) + np.square(xx - xc))
#     D = D*mask
#     return(D)


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


def findCellInnerCircle(I0, withBeads = True, plot=False, k_th = 1.0):
    """
    On a picture of a cell in fluo, find the approximate position of the cell.
    The idea is to fit the largest circle which can be contained in a mask of the cell.
    It uses the library shapely, and more precisely the function polylabel,
    to find the "pole of inaccessibility" of the cell mask.
    See : https://github.com/mapbox/polylabel and https://sites.google.com/site/polesofinaccessibility/
    Interesting topic !
    """
    
    th1 = skm.filters.threshold_isodata(I0) * k_th
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

# %%% For data management

def dataGroup(df, groupCol = 'cellID', idCols = [], numCols = [], aggFun = 'mean'):
    agg_dict = {}
    for col in idCols:
        agg_dict[col] = 'first'
    for col in numCols:
        agg_dict[col] = aggFun
    
    all_cols = list(agg_dict.keys()) + [groupCol]
    group = df[all_cols].groupby(groupCol)
    df_agg = group.agg(agg_dict)
    return(df_agg)


def dataGroup_weightedAverage(df, groupCol = 'cellID', idCols = [], 
                              valCol = '', weightCol = '', weight_method = 'ciw^2'):   
    # idCols = ['date', 'cellName', 'cellID', 'manipID'] + idCols
    
    wAvgCol = valCol + '_wAvg'
    wVarCol = valCol + '_wVar'
    wStdCol = valCol + '_wStd'
    wSteCol = valCol + '_wSte'
    
    # 1. Compute the weights if necessary
    if weight_method == 'ciw^1':
        ciwCol = weightCol
        weightCol = valCol + '_weight'
        df[weightCol] = (df[valCol]/df[ciwCol])
    elif weight_method == 'ciw^2':
        ciwCol = weightCol
        weightCol = valCol + '_weight'
        df[weightCol] = (df[valCol]/df[ciwCol])**2
    
    df = df.dropna(subset = [weightCol])
    
    # 2. Group and average
    groupColVals = df[groupCol].unique()
    
    d_agg = {k:'first' for k in idCols}

    # In the following lines, the weighted average and weighted variance are computed
    # using new columns as intermediates in the computation.
    #
    # Col 'A' = K x Weight --- Used to compute the weighted average.
    # 'K_wAvg' = sum('A')/sum('weight') in each category (group by condCol and 'fit_center')
    #
    # Col 'B' = (K - K_wAvg)**2 --- Used to compute the weighted variance.
    # Col 'C' =  B * Weight     --- Used to compute the weighted variance.
    # 'K_wVar' = sum('C')/sum('weight') in each category (group by condCol and 'fit_center')
    
    # Compute the weighted mean
    df['A'] = df[valCol] * df[weightCol]
    grouped1 = df.groupby(by=[groupCol])
    d_agg.update({'A': ['count', 'sum'], weightCol: 'sum'})
    data_agg = grouped1.agg(d_agg).reset_index()
    data_agg.columns = ufun.flattenPandasIndex(data_agg.columns)
    data_agg[wAvgCol] = data_agg['A_sum']/data_agg[weightCol + '_sum']
    data_agg = data_agg.rename(columns = {'A_count' : 'count_wAvg'})
    
    # Compute the weighted std
    df['B'] = df[valCol]
    for co in groupColVals:
        weighted_avg_val = data_agg.loc[(data_agg[groupCol] == co), wAvgCol].values[0]
        index_loc = (df[groupCol] == co)
        col_loc = 'B'
        
        df.loc[index_loc, col_loc] = df.loc[index_loc, valCol] - weighted_avg_val
        df.loc[index_loc, col_loc] = df.loc[index_loc, col_loc] ** 2
            
    df['C'] = df['B'] * df[weightCol]
    grouped2 = df.groupby(by=[groupCol])
    data_agg2 = grouped2.agg({'C': 'sum', weightCol: 'sum'}).reset_index()
    data_agg2[wVarCol] = data_agg2['C']/data_agg2[weightCol]
    data_agg2[wStdCol] = data_agg2[wVarCol]**0.5
    
    # Combine all in data_agg
    data_agg[wVarCol] = data_agg2[wVarCol]
    data_agg[wStdCol] = data_agg2[wStdCol]
    data_agg[wSteCol] = data_agg[wStdCol] / data_agg['count_wAvg']**0.5
    
    # data_agg = data_agg.drop(columns = ['A_sum', weightCol + '_sum'])
    data_agg = data_agg.drop(columns = ['A_sum'])
    
    # data_agg = data_agg.drop(columns = [groupCol + '_first'])
    data_agg = data_agg.rename(columns = {k+'_first':k for k in idCols})

    return(data_agg)




# %% Complete pipeline: tracking bf -> fluo

# %%% Paths

# date = '24-02-27'
# date = '24-05-24'
date = '24-06-14'
date2 = dateFormat(date)
cell = 'C2'

#### Added here
# for i in range(10, 19):
#     if i != 3 and i != 11:
#         cell = 'C'+str(i)
        #### Added here
        
cellId = f'{date}_{cell}'
specif = '' #'_off4um' # '_off4um' # '_off3-5um' # '_off4um'
specifFolder = '' # 'M2-20um/B5mT/' # 'M2-20um/B5mT/' # 'M2-20um/Bstairs/' # 'M2-20um/B5mT/'
specifDeptho = '' #'-Andor_M2' # 'M2_'

srcDir = f'D:/MagneticPincherData/Raw/{date2}_Chameleon/{specifFolder}Clean_Fluo_BF/M1/'

bfName = f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_BF.tif'
bfPath = os.path.join(srcDir, bfName)
resBfDir = os.path.join(srcDir, 'Results_Tracker')
resBfPath = os.path.join(resBfDir, bfName[:-4] + '_PY.csv')

dictPaths = {'sourceDirPath' : srcDir,
             'imageFileName' : bfName,
             'resultsFileName' : bfName[:-4] + '_Results.txt',
             'depthoDir' : 'D:/MagneticPincherData/Raw/DepthoLibrary',
             'depthoName' : f'{date2}_Chameleon{specifDeptho}_M450_step20_100X',
             'resultDirPath' : resBfDir,
             }


fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_Fluo.tif'

dstDir =  os.path.join(srcDir, 'Results_Fluo', f'{cell}')
fluoPath = os.path.join(srcDir, fluoName)

figSavePath = dstDir

if not os.path.isdir(dstDir):
    os.mkdir(dstDir)



# %%% Tracking bf

# %%%% Define constants & Additionnal options & Make metaDataFrame

OMEfilepath = os.path.join(dictPaths['sourceDirPath'], f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_OME.txt')

CTZ_tz = OMEDataParser(OMEfilepath)
[nC, nT, nZ] = CTZ_tz.shape[:3]
BF_T = CTZ_tz[0,:,:,0].flatten() * 1000

dictConstants = {'microscope' : 'labview',
                 #
                 'inside bead type' : 'M450-2025', # 'M450' or 'M270'
                 'inside bead diameter' : 4493, # nm
                 'inside bead magnetization correction' : 0.969, # nm
                 'outside bead type' : 'M450-Strept', # 'M450' or 'M270'
                 'outside bead diameter' : 4506, # nm
                 'outside bead magnetization correction' : 1.056, # nm
                 #
                 'normal field multi images' : nZ, # Number of images
                 'multi image Z step' : 400, # nm
                 'multi image Z direction' : 'downward', # Either 'upward' or 'downward'
                 #
                 'scale pixel per um' : 7.4588, # pixel/µm
                 'optical index correction' : 0.875, # ratio, without unit
                 'beads bright spot delta' : 0, # Rarely useful, do not change
                 'magnetic field correction' : 1.0, # ratio, without unit
                 }


dictOptions = {'redoAllSteps' : True,
               'trackAll' : False,
               'importLogFile' : True,
               'saveLogFile' : True,
               'saveFluo' : False,
               'importTrajFile' : True,
               'expandedResults' : True,
              }

NFrames = len(BF_T)
B_set = 5 * np.ones(NFrames)

# Stairs of B field
# nup = dictConstants['normal field multi images']
# B_set = 5 * (np.ones(NFrames).reshape(nup, NFrames//nup) * np.arange(1, NFrames//nup + 1)).flatten('F')
# B_set[B_set == 25] = 20 # ONLY FOR 24-05-24_C11

loopStructure = [{'Status':'Passive',     'NFrames':NFrames}]  # Only one constant field phase
                
metaDf = makeMetaData(BF_T, B_set, loopStructure)

# %%%% Call mainTracker()

# tsdf = smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)


# %%% Mappemonde fluo

# %%%% Options

# framesize = int(35*8/2) 
# inPix_set, outPix_set = 18, 10
# blur_parm = 2
# relative_height_virebi = 0.4

# warp_radius = round(12.5*SCALE_100X_ZEN)
# N_profilesMatrix = 15

# interp_factor_x = 5
# interp_factor_y = 10


# Fluo_T = np.mean(CTZ_tz[1,:,:,0], axis=1)

# optical_index_correction = 0.875
# dz = 0.4  * optical_index_correction # µm # Distance between each image of the stacks
# DZb = 4.5 * optical_index_correction # µm # Distance between focal point of the deptho and the equatorial plane of the beads #### SHOULD BE RE-MEASURED
# DZo = 4.0 * optical_index_correction # µm # Distance between the stack of BF images and the corresponding stack of Fluo images
# Rbead = 4.5/2

# SAVE_RESULTS = True
# PLOT_WARP = True
# PLOT_MAP = True
# PLOT_NORMAL = True

framesize = int(35*8/2) 
inPix_set, outPix_set = 20, 10
blur_parm = 2
relative_height_virebi = 0.2

warp_radius = round(12.5*SCALE_100X_ZEN)
N_profilesMatrix = 15

interp_factor_x = 5
interp_factor_y = 10


Fluo_T = np.mean(CTZ_tz[1,:,:,0], axis=1)

optical_index_correction = 0.875
dz = 0.4  * optical_index_correction # µm # Distance between each image of the stacks
DZb = 4.5 * optical_index_correction # µm # Distance between focal point of the deptho and the equatorial plane of the beads #### SHOULD BE RE-MEASURED
DZo = 4.0 * optical_index_correction # µm # Distance between the stack of BF images and the corresponding stack of Fluo images
Rbead = 4.5/2

SAVE_RESULTS = False
PLOT_WARP = False
PLOT_MAP = False
PLOT_NORMAL = False
PLOTS_MANUSCRIPT = True


# %%%% New version

#### Start

plt.ioff()
I = skm.io.imread(fluoPath)

# czt_shape, czt_seq = get_CZT_fromTiff(fluoPath)
# [nC, nZ, nT] = czt_shape
[nC, nT, nZ] = CTZ_tz.shape[:3]

Angles = np.arange(0, 360, 1)
previousCircleXYR = (50,50,50)

tsdf = pd.read_csv(resBfPath, sep=None, engine='python')

colsContact = ['iT', 'T', 'X', 'Y', 'iZ', 'Z', 'R', 'A', 'Xc', 'Yc', 'Rc', 
               'Xin', 'Yin', 'Zin', 'Xout', 'Yout', 'Zout', 
               ]
dfContact = pd.DataFrame(np.zeros((nT, len(colsContact))), columns=colsContact)
dfBeforeContact = pd.DataFrame(np.zeros((nT, len(colsContact))), columns=colsContact)

fluoCell = np.zeros((nT, nZ))
fluoCyto = np.zeros((nT, nZ))
fluoCytoStd = np.zeros((nT, nZ))
fluoBack = np.zeros((nT, nZ))
fluoBeadIn = np.zeros((nT, nZ))

arrayViterbiContours = np.zeros((nT, nZ, 360, 2))
profileMatrix = np.zeros((nT, nZ, N_profilesMatrix, warp_radius))

for t in range(nT):#(nT):
    intensityMap_t = []
    tsd_t = tsdf.iloc[t,:].to_dict()
        
    #### 1.0 Compute position of contact point
    x_contact = (tsd_t['X_in']+tsd_t['X_out'])/2
    y_contact = (tsd_t['Y_in']+tsd_t['Y_out'])/2
    
    zr_contact = (tsd_t['Zr_in']+tsd_t['Zr_out'])/2
    CF_mid = - zr_contact + DZo - DZb 
    # EF = ED + DI + IF; ED = - DZb; for I=I_mid & F=F_mid, DI = - zr; and IF = DZo
    # CF = (EinF + EoutF) / 2 [eq planes of the 2 beads]
    # so CF = -(zrIn+zrOut)/2 - DZb + DZo = -zr_contact + DZo - DZb 
    iz_mid = (nZ-1)//2 # If 11 points in Z, iz_mid = 5
    iz_contact = iz_mid - (CF_mid / dz)
    # print(iz_contact)
    
    x_in = tsd_t['X_in']
    y_in = tsd_t['Y_in']
    z_in = ((-1)*tsd_t['Zr_in']) + DZo - DZb
    
    x_out = tsd_t['X_out']
    y_out = tsd_t['Y_out']
    z_out = ((-1)*tsd_t['Zr_out']) + DZo - DZb
    
    dfContact.loc[t, ['iT', 'T', 'X', 'Y', 'iZ', 'Z']] = [t, Fluo_T[t], x_contact, y_contact, iz_contact, iz_contact*dz]
    dfContact.loc[t, ['Xin', 'Yin', 'Zin', 'Xout', 'Yout', 'Zout']] = [x_in, y_in, dz*iz_mid-z_in, x_out, y_out, dz*iz_mid-z_out]
    dfBeforeContact.loc[t, ['iT', 'T', 'X', 'Y', 'iZ', 'Z']] = [t, Fluo_T[t], x_contact, y_contact, iz_contact, iz_contact*dz]
    dfBeforeContact.loc[t, ['Xin', 'Yin', 'Zin', 'Xout', 'Yout', 'Zout']] = [x_in, y_in, dz*iz_mid-z_in, x_out, y_out, dz*iz_mid-z_out]
    # Zr = (-1) * (tsd_t['Zr_in']+tsd_t['Zr_out'])/2 # Taking the opposite cause the depthograph is upside down for these experiments
    # izmid = (nZ-1)//2 # If 11 points in Z, izmid = 5
    # zz = dz * np.arange(-izmid, +izmid, 1) # If 11 points in Z, zz = [-5, -4, ..., 4, 5] * dz
    # zzf = zz + Zr # Coordinates zz of bf images with respect to Zf, the Z of the focal point of the deptho
    # zzeq = zz + Zr + DZo - DZb # Coordinates zz of fluo images with respect to Zeq, the Z of the equatorial plane of the bead
    
    # zi_contact = Akima1DInterpolator(zzeq, np.arange(len(zzeq)))(0)
    
    A_contact_map = 0
    foundContact = False
    foundBeforeContact = False
    
    for z in range(nZ):
        contactSlice = False
        beforeContactSlice = False
        i_zt = z + nZ*t
        I_zt = I[i_zt]
        
        #### 2.0 Locate cell
        (Yc, Xc), Rc = findCellInnerCircle(I_zt, withBeads = True, plot=False)
        print(z, Rc)
        if z >= 1 and Rc < 8.5*SCALE_100X_ZEN:
            print('done this')
            # (Xc, Yc, Rc) = previousCircleXYR
            (Xc, Yc, Rc) = (I_zt.shape[1]//2, I_zt.shape[0]//2, 9.5*SCALE_100X_ZEN)
        if z == 0 and Rc < 8.5*SCALE_100X_ZEN:
            print('done that')
            (Xc, Yc, Rc) = (I_zt.shape[1]//2, I_zt.shape[0]//2, 9.5*SCALE_100X_ZEN)
            
        Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
        
        #### 2.1 Warp
        warped = skm.transform.warp_polar(I_zt, center=(Yc, Xc), radius=warp_radius, #Rc*1.2, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        # max_values = np.max(warped[:,Rc0-inPix_set:Rc0+outPix_set+1], axis=1)
            
        #### 2.2 Viterbi Smoothing
        edge_viterbi = ViterbiEdge(warped, Rc0, inPix_set, outPix_set, 2, relative_height_virebi)
        edge_viterbi_unwarped = np.array(unwarpRA(np.array(edge_viterbi), Angles, Xc, Yc)) # X, Y
        
        N_it_viterbi = 2
        for i in range(N_it_viterbi):

            #### Extra iteration
            # x.0 Locate cell
            (Xc, Yc), Rc = fitCircle(np.array([edge_viterbi_unwarped[1], edge_viterbi_unwarped[0]]).T, loss = 'huber')        
            Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
            
            # x.1 Warp
            warped = skm.transform.warp_polar(I_zt, center=(Yc, Xc), radius=warp_radius, #Rc*1.4, 
                                              output_shape=None, scaling='linear', channel_axis=None)
            warped = skm.util.img_as_uint(warped)
            w_ny, w_nx = warped.shape
            Angles = np.arange(0, 360, 1)
            max_values = np.max(warped[:,Rc0-inPix_set:Rc0+outPix_set+1], axis=1)
            inPix, outPix = inPix_set, outPix_set
            
            # x.3 Viterbi Smoothing
            edge_viterbi = ViterbiEdge(warped, Rc0, inPix, outPix, blur_parm, relative_height_virebi)
            edge_viterbi_unwarped = unwarpRA(np.array(edge_viterbi), Angles, Xc, Yc)
            arrayViterbiContours[t, z, :] = np.array([edge_viterbi_unwarped[0], edge_viterbi_unwarped[1]]).T
        
        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(I_zt, cmap='gray', aspect='equal')
        # ax.plot(edge_viterbi_unwarped[0], edge_viterbi_unwarped[1], c='g', ls = '--')
        # ax.plot(Xc, Yc, 'go')
        # CC = circleContour_V2((Yc, Xc), Rc)
        # [yy_circle, xx_circle] = CC.T
        # ax.plot(xx_circle, yy_circle, 'b--')
        # ax.set_title('Detected contours')
        # fig.tight_layout()
        # plt.show()

        # #### 3.0 Locate cell
        # (Xc, Yc), Rc = fitCircle(np.array([edge_viterbi_unwarped[1], edge_viterbi_unwarped[0]]).T, loss = 'huber')
        # previousCircleXYR = (Xc, Yc, Rc)
        # Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
        
        # #### 3.1 Warp
        # warped = skm.transform.warp_polar(I_zt, center=(Yc, Xc), radius=warp_radius, #Rc*1.4, 
        #                                   output_shape=None, scaling='linear', channel_axis=None)
        # warped = skm.util.img_as_uint(warped)
        # w_ny, w_nx = warped.shape
        # Angles = np.arange(0, 360, 1)
        # max_values = np.max(warped[:,Rc0-inPix_set:Rc0+outPix_set+1], axis=1)
        
        # # #### 3.2 Interp in X
        # # warped_interp = ufun.resize_2Dinterp(warped, fx=interp_factor_x, fy=1)
        # # warped = warped_interp
        # # w_nx, Rc0 = w_nx * interp_factor_x, Rc0 * interp_factor_x
        # # inPix, outPix = inPix_set * interp_factor_x, outPix_set * interp_factor_x
        # # R_contact *= interp_factor_x
        # # blur_parm *= interp_factor_x
        
        # inPix, outPix = inPix_set, outPix_set
        
        # #### 3.3 Viterbi Smoothing
        # edge_viterbi = ViterbiEdge(warped, Rc0, inPix, outPix, blur_parm, relative_height_virebi)
        # # edge_viterbi_unwarped = unwarpRA(np.array(edge_viterbi)/interp_factor_x, Angles, Xc, Yc)
        # edge_viterbi_unwarped = unwarpRA(np.array(edge_viterbi), Angles, Xc, Yc)
        # arrayViterbiContours[t, z, :] = np.array([edge_viterbi_unwarped[0], edge_viterbi_unwarped[1]]).T
        
        
        R_contact, A_contact = warpXY(x_contact*SCALE_100X_ZEN, y_contact*SCALE_100X_ZEN, Xc, Yc)
        if (z == np.round(iz_contact)) or (not foundContact and z == nZ-1):
            contactSlice = True
            foundContact = True
            # print(z, R_contact, A_contact)
            A_contact_map = A_contact
            dfContact.loc[t, ['R', 'A']] = [R_contact, A_contact]
            
        if (z == np.round(iz_contact)-1) or (not foundContact and z == nZ-1):
            beforeContactSlice = True
            foundBeforeContact = True
            # print(z, R_contact, A_contact)
            A_BeforeContact_map = A_contact
            dfBeforeContact.loc[t, ['R', 'A']] = [R_contact, A_contact]
        
        dfContact.loc[t, ['Xc', 'Yc', 'Rc']] += np.array([(Xc/SCALE_100X_ZEN)/(nZ), (Yc/SCALE_100X_ZEN)/(nZ), (Rc/SCALE_100X_ZEN)/(nZ)])
        
        #### 4. Append the contact profile matrix
        # profileMatrix = np.zeros((nT, nZ, N_profilesMatrix, warp_radius))
        halfN = N_profilesMatrix//2
        tmpWarp = np.copy(warped)
        A_contact_r = round(A_contact)
        dA = A_contact_r - A_contact
        if   A_contact_r < halfN+3:
            tmpWarp = np.roll(tmpWarp, halfN+3, axis=0) 
            tmp_A_contact   = A_contact   + (halfN+3)
            tmp_A_contact_r = A_contact_r + (halfN+3)
            print('roll forward')
            print(A_contact_r, tmp_A_contact_r)
        elif A_contact_r > 360 - (halfN+3):
            tmpWarp = np.roll(tmpWarp, -(halfN+3), axis=0) 
            tmp_A_contact   = A_contact   - (halfN+3)
            tmp_A_contact_r = A_contact_r - (halfN+3)
            print('roll backward')
            print(A_contact_r, tmp_A_contact_r)
        else:
            tmp_A_contact   = A_contact
            tmp_A_contact_r = A_contact_r
    
        
        try:
            smallTmpWarp = tmpWarp[tmp_A_contact_r - (halfN+3):tmp_A_contact_r + (halfN+3)+1, :]
            tform = skm.transform.EuclideanTransform(translation=(0, dA))
            smallTmpWarp_tformed = skm.util.img_as_uint(skm.transform.warp(smallTmpWarp, tform))
            
            Ltmp = len(smallTmpWarp_tformed)
            smallWarp = smallTmpWarp_tformed[Ltmp//2 - halfN:Ltmp//2 + halfN+1, :]
            
            profileMatrix[t, z] = smallWarp
            
        except:
            print("profileMatrix Error")
            print(A_contact_r)
            print(dA)
            print(tmpWarp.shape)
            print(smallTmpWarp.shape)
            print(smallTmpWarp_tformed.shape)
            print(smallWarp.shape)
            
        #### 6.1 Compute beads positions
        draw_InCircle = False
        draw_OutCircle = False
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
            # RINbeadContour *= interp_factor_x
            
        if h_OutCircle > 0 and h_OutCircle < 2*Rbead:
            draw_OutCircle = True
            a_OutCircle = (h_OutCircle*(2*Rbead-h_OutCircle))**0.5
            # OUTbeadContour = circleContour((round(y_out*SCALE_100X_ZEN), round(x_out*SCALE_100X_ZEN)), 
            #                                a_OutCircle*SCALE_100X_ZEN, I_zt.shape)
            OUTbeadContour = circleContour_V2((round(y_out*SCALE_100X_ZEN), round(x_out*SCALE_100X_ZEN)),
                                              a_OutCircle*SCALE_100X_ZEN)
            XOUTbeadContour, YOUTbeadContour = OUTbeadContour[:, 1], OUTbeadContour[:, 0]
            ROUTbeadContour, AOUTbeadContour = warpXY(XOUTbeadContour, YOUTbeadContour, Xc, Yc)
            # ROUTbeadContour *= interp_factor_x
            
        
        #### 5. Analyse fluo cyto & background
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

        mask_removed_from_cyto = inside & (~mask_cyto)
        cell_filtered = cell_dilated & (~mask_removed_from_cyto)
        
        nb_pix = np.sum(cell_filtered.astype(int))
        total_val = np.sum(I_zt.flatten()[cell_filtered.flatten()])
        I_cell = total_val/nb_pix
        
        # Bead
        I_bead = np.nan
        if draw_InCircle:
            xx, yy = np.meshgrid(np.arange(I_zt.shape[1]), np.arange(I_zt.shape[0]))
            xc, yc = round(x_in*SCALE_100X_ZEN), round(y_in*SCALE_100X_ZEN)
            rc = a_InCircle*SCALE_100X_ZEN*0.9
            maskBead = ((xx-xc)**2 + (yy-yc)**2)**0.5 < rc
            I_bead = np.median(I_zt[maskBead])
        
        
        # Store
        fluoCyto[t, z] = I_cyto
        fluoCytoStd[t, z] = Std_cyto
        fluoBack[t, z] = I_back
        fluoCell[t, z] = I_cell
        fluoBeadIn[t, z] = I_bead
        
        #### 5. Define normalization functions
        def normalize_Icyto(x):
            return((x - I_back)/(I_cyto - I_back))
    
        def normalize_Icell(x):
            return((x - I_back)/(I_cell - I_back))
        
        def normalize_Icell2(x):
            return((x - I_cyto)/(I_cell - I_back))
        
        
        
        #### 5. Read values
        VEW=5
        max_values = normalize_Icell(max_values)
        viterbi_Npix_values = [np.sum(warped[Angles[i], edge_viterbi[i]-VEW//2:edge_viterbi[i] + (VEW//2 + 1)])/VEW for i in range(len(edge_viterbi))]
        viterbi_Npix_values = normalize_Icell(viterbi_Npix_values)
        intensityMap_t.append(viterbi_Npix_values)
        
        
        
        #### 6. Plot
        #### 6.1 Plot Normalization
        if PLOT_NORMAL and z==nZ//2 and t==nT//2:
            figN, axesN = plt.subplots(1,5, figsize = (20, 5))
            ax = axesN[0]
            ax.imshow(I_zt)
            ax.set_title('Original')
            ax = axesN[1]
            ax.imshow((mask_cyto & inside_eroded).astype(int))
            ax.set_title('Cytoplasm')
            ax = axesN[2]
            ax.imshow(cell_filtered.astype(int))
            ax.set_title('Whole cell')
            ax = axesN[3]
            ax.imshow(outside.astype(int))
            ax.set_title('Background')
            if draw_InCircle:
                ax = axesN[4]
                ax.imshow(maskBead.astype(int))
                ax.set_title(f'I_bead = {I_bead:.0f}')
            
            figN.tight_layout()
            ufun.simpleSaveFig(figN, 'ExampleNormalization', figSavePath, '.png', 150)
            
        if PLOTS_MANUSCRIPT:
            figDir = "D:/MagneticPincherData/Figures/PhysicsDataset"
            if beforeContactSlice:
                #### First M plot
                gs.set_manuscript_options_jv()
                Ibf = skm.io.imread(bfPath)
                Ibf_zt = Ibf[i_zt]
                
                
                figC, axes = plt.subplots(1, 3, figsize = (17/gs.cm_in, 5.5/gs.cm_in))
                
                ax = axes[0]
                vmin, vmax = np.percentile(Ibf_zt, (1, 99))
                ax.imshow(Ibf_zt, cmap='gray', aspect='equal', vmin=vmin, vmax=vmax)
                ax.set_title('Bright field image')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                
                ax = axes[1]
                ax.imshow(I_zt, cmap='gray', aspect='equal')
                ax.plot(edge_viterbi_unwarped[0], edge_viterbi_unwarped[1], c='chartreuse', ls = '--', zorder=6, lw=0.75)
                ax.plot(Xc, Yc, 'go', ms=4)
                CC = circleContour_V2((Yc, Xc), Rc)
                [yy_circle, xx_circle] = CC.T
                # ax.plot(xx_circle, yy_circle, ls='--', c='cyan', lw=0.5)
                style = ':'
                if draw_InCircle:
                    ax.plot(XINbeadContour, YINbeadContour, c='dodgerblue', ls=style, lw=0.75)
                if draw_OutCircle:
                    ax.plot(XOUTbeadContour, YOUTbeadContour, c='darkturquoise', ls=style, lw=0.75)
                X_contact, Y_contact = unwarpRA(R_contact, A_contact, Xc, Yc)
                ax.plot(X_contact, Y_contact, 'r+', ms=5, zorder=8)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_title('Fluo image & contours')
                
                ax = axes[2]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                ufun.archiveFig(figC, name = 'fluoCell' + f'_{cellId}_Warp_t{t+1:.0f}_z{z+1:.0f}', ext = '.pdf', dpi = 100,
                                figDir = figDir, figSubDir = 'E-h-Fluo', cloudSave = 'flexible')
                
                
                #### Second M plot
                gs.set_manuscript_options_jv()
                figV2 = plt.figure(figsize = (17/gs.cm_in, 14/gs.cm_in))#, layout="constrained")
                spec = figV2.add_gridspec(3, 4)
                
                inBorder = Rc0 - inPix
                outBorder = Rc0 + outPix
                Rc0A = np.ones(360)*Rc0
                inBorderA = np.ones(360)*inBorder
                outBorderA = np.ones(360)*outBorder
                
                # 1.1 - Show the cell
                ax = figV2.add_subplot(spec[0,0])
                ax.imshow(I_zt, cmap='gray', aspect='equal')
                ax.set_title('Original Image', fontsize = 8)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                
                # 1.2 - Show the cell with annotations
                ax = figV2.add_subplot(spec[1,0])
                ax.imshow(I_zt, cmap='gray', aspect='equal')
                ax.plot(edge_viterbi_unwarped[0], edge_viterbi_unwarped[1], c='chartreuse', ls = '--', zorder=6, lw=1)
                ax.plot(Xc, Yc, 'go', ms=4)
                CC = circleContour_V2((Yc, Xc), Rc)
                [yy_circle, xx_circle] = CC.T
                # ax.plot(xx_circle, yy_circle, ls='--', c='cyan', lw=0.5)
                style = ':'
                if draw_InCircle:
                    ax.plot(XINbeadContour, YINbeadContour, c='dodgerblue', ls=style, lw=1)
                if draw_OutCircle:
                    ax.plot(XOUTbeadContour, YOUTbeadContour, c='darkturquoise', ls=style, lw=1)
                ax.set_title('Detected contours', fontsize = 8)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                
                # 2.1 - Show the warped cell
                ax1 = figV2.add_subplot(spec[:2,1])
                ax = ax1
                ax.imshow(warped[:, :], cmap='gray', aspect='auto')
                ax.set_title('Warped image', fontsize = 8)
                ax.set_ylabel('Angle (°)', fontsize = 8)
                # ax.set_xlabel('Radial pix')
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.tick_params(axis='both', which='major', labelsize=6)
                
                # 2.2 - Show the warped cell with viterbi detection
                ax = figV2.add_subplot(spec[:2,2], sharey = ax1)
                ax.imshow(warped[:, :], cmap='gray', aspect='auto')
                # ax.plot(Rc0A, Angles, ls='--', c='red', lw=1)
                # ax.plot(inBorderA, Angles, ls='--', c='gold', lw=1)
                # ax.plot(outBorderA, Angles, ls='--', c='gold', lw=1)
                ax.plot(edge_viterbi, Angles, ls='-', c='chartreuse', marker='', ms = 2, lw=1)
                ax.plot(R_contact, A_contact, 'rP', ms=5)
                style = ':'
                marker = ''
                if draw_InCircle:
                    ax.plot(RINbeadContour, AINbeadContour, c='dodgerblue', ls=style, marker=marker, ms=2, lw=1)
                if draw_OutCircle:
                    mask = (ROUTbeadContour < w_nx)
                    ROUTbeadContour = ROUTbeadContour[mask]
                    AOUTbeadContour = AOUTbeadContour[mask]
                    ax.plot(ROUTbeadContour, AOUTbeadContour, c='darkturquoise', ls=style, marker=marker, ms=2, lw=1)
                ax.set_title('Viterbi edge', fontsize = 8)
                # ax.set_xlabel('Radial pix')
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.tick_params(axis='both', which='major', labelsize=6)
                
                # 4 - Show the intensity along the cell contour
                ax = figV2.add_subplot(spec[:2, 3], sharey = ax1)
                ax.set_title('Intensity profile', fontsize = 8)
                ax.plot(viterbi_Npix_values, Angles, c='forestgreen', ls='-', label = 'Intensity profile')
                ax.set_xlim([0, 3])
                ax.grid(axis='both')
                ax.tick_params(axis='both', which='major', labelsize=6)
                
                # figV2.suptitle(f'Warp - {cellId} - t = {t+1:.0f}/{nT:.0f} - z = {z+1:.0f}/{nZ:.0f}')
                figName = f'{cellId}_Warp_t{t+1:.0f}_z{z+1:.0f}'
                figV2.tight_layout()
                ufun.archiveFig(figV2, name = 'viterbiSegment' + f'_{cellId}_Warp_t{t+1:.0f}_z{z+1:.0f}', ext = '.pdf', dpi = 100,
                                figDir = figDir, figSubDir = 'E-h-Fluo', cloudSave = 'flexible')
            
        
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
            ax.plot(edge_viterbi_unwarped[0], edge_viterbi_unwarped[1], c='chartreuse', ls = '--', zorder=6)
            ax.plot(Xc, Yc, 'go')
            CC = circleContour_V2((Yc, Xc), Rc)
            [yy_circle, xx_circle] = CC.T
            ax.plot(xx_circle, yy_circle, ls='--', c='cyan', lw=0.5)
            if contactSlice:
                style = '-'
            else:
                style = ':'
            if draw_InCircle:
                ax.plot(XINbeadContour, YINbeadContour, c='r', ls=style)
            if draw_OutCircle:
                ax.plot(XOUTbeadContour, YOUTbeadContour, c='gold', ls=style)
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
            style = ''
            marker = '.'
            if draw_InCircle:
                ax.plot(RINbeadContour, AINbeadContour, c='dodgerblue', ls=style, marker=marker, ms=2)
            if draw_OutCircle:
                mask = (ROUTbeadContour < w_nx)
                ROUTbeadContour = ROUTbeadContour[mask]
                AOUTbeadContour = AOUTbeadContour[mask]
                ax.plot(ROUTbeadContour, AOUTbeadContour, c='darkturquoise', ls=style, marker=marker, ms=2)
            ax.set_title('Viterbi Edge')
            ax.set_xlabel('Radial pix')
            
            # 3.1 - Show the intensity profiles for Viterbi
            ax = figV2.add_subplot(spec[:, 4], sharey = ax1)
            step1 = 10
            ax.plot(inBorderA, Angles, c='k', ls='--', marker='')
            ax.plot(outBorderA, Angles, c='k', ls='--', marker='')
            ax.plot(edge_viterbi[::step1], Angles[::step1], ls='', c='cyan', marker='^', ms = 4, mec = 'k', zorder=5)
            ax.set_title('Viterbi profiles')
            ax.set_xlabel('Radial pix')
            
            cList_viridis = plt.cm.viridis(np.linspace(0.05, 0.95, w_ny//step1))
            ax.set_prop_cycle(plt.cycler("color", cList_viridis))
            for a in Angles[::step1]:
                try:
                    profile = normalize_Icell(warped[a,:])*step1
                    RR = np.arange(len(profile))
                    ax.plot(RR, a - profile)
                except:
                    pass

            # 3.2 - Zoom on the intensity profiles for Viterbi
            ax = figV2.add_subplot(spec[:, 5], sharey = ax1)
            step2 = 20
            ax.plot(edge_viterbi[::step2], Angles[::step2], ls='', c='cyan', marker='^', ms = 4, mec = 'k', zorder=5)
            cList_viridis = plt.cm.viridis(np.linspace(0.05, 0.95, w_ny//step2))
            ax.set_prop_cycle(plt.cycler("color", cList_viridis))
            ax.set_title('Viterbi profiles')
            ax.set_xlabel('Radial pix')
            maxWarped = np.max(warped)
            normWarped = warped * 10 * step2/maxWarped
            for a in Angles[::step2]:    
                try:
                    # profile = normWarped[a, Rc0-inPix:Rc0+outPix] - np.min(normWarped[a, Rc0-inPix:Rc0+outPix])
                    profile = normalize_Icell(warped[a,Rc0-inPix:Rc0+outPix])*step2
                    RR = np.arange(Rc0-inPix, Rc0+outPix, 1)
                    ax.plot(RR, a - profile)
                except:
                    pass

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
        

    #### Out of the Z loop here
    
    #### 6.3 Map and contact point
    if PLOT_MAP:
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
        
    if PLOTS_MANUSCRIPT:
        figDir = "D:/MagneticPincherData/Figures/PhysicsDataset"
        fig, ax = plt.subplots(1, 1, figsize = (17/gs.cm_in, 3/gs.cm_in))
        im = ax.imshow(intensityMap_t, cmap = 'viridis', vmin=0, origin='lower')
        # figM.colorbar(im, ax=ax, orientation='horizontal', fraction=.1, aspect=80)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)
        fig.colorbar(im, cax=cax, orientation='vertical', aspect=40) 
        # ax.set_title(f'Map - {cellId} - t = {t+1:.0f}/{nT:.0f}')
        ax.set_title(f'Surface intensity map', fontsize = 8)
        ax.plot(A_contact_map, iz_contact+0.5, 'r+', ms=7)
        ax.set_xlabel(('Angle (°)'))
        ax.set_ylabel(('Z (frame)'), fontsize = 6)
        fig.tight_layout()
        figName = f'{cellId}_Map_t{t+1:.0f}'
        # ufun.simpleSaveFig(figM, figName, figSavePath, '.png', 150)
        ufun.archiveFig(fig, name = 'fluoMap' + f'_{cellId}_Map_t{t+1:.0f}', ext = '.pdf', dpi = 100,
                        figDir = figDir, figSubDir = 'E-h-Fluo', cloudSave = 'flexible')
        # plt.show()
        plt.close('all')
    
    #### 7. Save results 1/2
    if SAVE_RESULTS:
        intensityMap_t = np.array(intensityMap_t).astype(float)
        np.savetxt(os.path.join(dstDir, f'{cellId}_Map_t{t+1:.0f}.txt'), intensityMap_t, fmt='%.5f')

#### Out of the T loop here

#### 7. Save results 2/2
if SAVE_RESULTS:
    dfContact.to_csv(os.path.join(dstDir, f'{cellId}_dfContact.csv'), sep=';', index=False)
    np.save(os.path.join(dstDir, f'{cellId}_xyContours.npy'), arrayViterbiContours)
    np.save(os.path.join(dstDir, f'{cellId}_profileMatrix.npy'), profileMatrix)
    np.savetxt(os.path.join(dstDir, f'{cellId}_fluoCyto.txt'), fluoCyto, fmt='%.1f')
    np.savetxt(os.path.join(dstDir, f'{cellId}_fluoCytoStd.txt'), fluoCytoStd, fmt='%.3f')
    np.savetxt(os.path.join(dstDir, f'{cellId}_fluoBack.txt'), fluoBack, fmt='%.1f')
    np.savetxt(os.path.join(dstDir, f'{cellId}_fluoCell.txt'), fluoCell, fmt='%.1f')
    np.savetxt(os.path.join(dstDir, f'{cellId}_fluoBeadIn.txt'), fluoBeadIn, fmt='%.1f')
    
    
plt.ion()

print(f'Done for {cellId} :D')


# %%%% As a function

def doFluoAnalysis(date, cell,
                   SAVE_RESULTS = True,
                   PLOT_WARP = True, PLOT_MAP = True, PLOT_NORMAL = True):
    
    date2 = dateFormat(date)

    cellId = f'{date}_{cell}'
    specif = '' # '_off4um' # '_off3-5um' # '_off4um'
    specifFolder = '' # 'M2-20um/B5mT/' # 'M2-20um/B5mT/' # 'M2-20um/Bstairs/' # 'M2-20um/B5mT/'
    specifDeptho = '-Andor_M3' # 'M2_'

    srcDir = f'D:/MagneticPincherData/Raw/{date2}_Chameleon/{specifFolder}Clean_Fluo_BF/M3'

    bfName = f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_BF.tif'
    bfPath = os.path.join(srcDir, bfName)
    resBfDir = os.path.join(srcDir, 'Results_Tracker')
    resBfPath = os.path.join(resBfDir, bfName[:-4] + '_PY.csv')

    dictPaths = {'sourceDirPath' : srcDir,
                 'imageFileName' : bfName,
                 'resultsFileName' : bfName[:-4] + '_Results.txt',
                 'depthoDir' : 'D:/MagneticPincherData/Raw/DepthoLibrary',
                 'depthoName' : f'{date2}_Chameleon{specifDeptho}_M450_step20_100X',
                 'resultDirPath' : resBfDir,
                 }


    fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_Fluo.tif'

    dstDir =  os.path.join(srcDir, 'Results_Fluo', f'{cell}')
    fluoPath = os.path.join(srcDir, fluoName)

    figSavePath = dstDir
    
    if not os.path.isdir(dstDir):
        os.mkdir(dstDir)
    
    OMEfilepath = os.path.join(dictPaths['sourceDirPath'], f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_OME.txt')
    
    CTZ_tz = OMEDataParser(OMEfilepath)
    [nC, nT, nZ] = CTZ_tz.shape[:3]
    BF_T = CTZ_tz[0,:,:,0].flatten() * 1000
    
    dictConstants = {'microscope' : 'labview',
                     #
                     'inside bead type' : 'M450-2025', # 'M450' or 'M270'
                     'inside bead diameter' : 4493, # nm
                     'inside bead magnetization correction' : 0.969, # nm
                     'outside bead type' : 'M450-Strept', # 'M450' or 'M270'
                     'outside bead diameter' : 4506, # nm
                     'outside bead magnetization correction' : 1.056, # nm
                     #
                     'normal field multi images' : nZ, # Number of images
                     'multi image Z step' : 400, # nm
                     'multi image Z direction' : 'downward', # Either 'upward' or 'downward'
                     #
                     'scale pixel per um' : 7.4588, # pixel/µm
                     'optical index correction' : 0.875, # ratio, without unit
                     'beads bright spot delta' : 0, # Rarely useful, do not change
                     'magnetic field correction' : 1.0, # ratio, without unit
                     }
    
    NFrames = len(BF_T)
    B_set = 5 * np.ones(NFrames)
    
    #### Options
    
    framesize = int(35*8/2) 
    inPix_set, outPix_set = 20, 10
    blur_parm = 2
    relative_height_virebi = 0.2
    
    warp_radius = round(12.5*SCALE_100X_ZEN)
    N_profilesMatrix = 15
    
    interp_factor_x = 5
    interp_factor_y = 10
    
    
    Fluo_T = np.mean(CTZ_tz[1,:,:,0], axis=1)
    
    optical_index_correction = 0.875
    dz = 0.4  * optical_index_correction # µm # Distance between each image of the stacks
    DZb = 4.5 * optical_index_correction # µm # Distance between focal point of the deptho and the equatorial plane of the beads #### SHOULD BE RE-MEASURED
    DZo = 4.0 * optical_index_correction # µm # Distance between the stack of BF images and the corresponding stack of Fluo images
    Rbead = 4.5/2
    
    #### Start
    
    plt.ioff()
    I = skm.io.imread(fluoPath)
    
    # czt_shape, czt_seq = get_CZT_fromTiff(fluoPath)
    # [nC, nZ, nT] = czt_shape
    [nC, nT, nZ] = CTZ_tz.shape[:3]
    
    Angles = np.arange(0, 360, 1)
    previousCircleXYR = (50,50,50)
    
    tsdf = pd.read_csv(resBfPath, sep=None, engine='python')
    
    colsContact = ['iT', 'T', 'X', 'Y', 'iZ', 'Z', 'R', 'A', 'Xc', 'Yc', 'Rc', 
                   'Xin', 'Yin', 'Zin', 'Xout', 'Yout', 'Zout', 
                   ]
    dfContact = pd.DataFrame(np.zeros((nT, len(colsContact))), columns=colsContact)
    
    fluoCell = np.zeros((nT, nZ))
    fluoCyto = np.zeros((nT, nZ))
    fluoCytoStd = np.zeros((nT, nZ))
    fluoBack = np.zeros((nT, nZ))
    fluoBeadIn = np.zeros((nT, nZ))
    
    arrayViterbiContours = np.zeros((nT, nZ, 360, 2))
    profileMatrix = np.zeros((nT, nZ, N_profilesMatrix, warp_radius))
    
    for t in range(nT):
        intensityMap_t = []
        tsd_t = tsdf.iloc[t,:].to_dict()
            
        #### 1.0 Compute position of contact point
        x_contact = (tsd_t['X_in']+tsd_t['X_out'])/2
        y_contact = (tsd_t['Y_in']+tsd_t['Y_out'])/2
        
        zr_contact = (tsd_t['Zr_in']+tsd_t['Zr_out'])/2
        CF_mid = - zr_contact + DZo - DZb 
        # EF = ED + DI + IF; ED = - DZb; for I=I_mid & F=F_mid, DI = - zr; and IF = DZo
        # CF = (EinF + EoutF) / 2 [eq planes of the 2 beads]
        # so CF = -(zrIn+zrOut)/2 - DZb + DZo = -zr_contact + DZo - DZb 
        iz_mid = (nZ-1)//2 # If 11 points in Z, iz_mid = 5
        iz_contact = iz_mid - (CF_mid / dz)
        # print(iz_contact)
        
        x_in = tsd_t['X_in']
        y_in = tsd_t['Y_in']
        z_in = ((-1)*tsd_t['Zr_in']) + DZo - DZb
        
        x_out = tsd_t['X_out']
        y_out = tsd_t['Y_out']
        z_out = ((-1)*tsd_t['Zr_out']) + DZo - DZb
        
        dfContact.loc[t, ['iT', 'T', 'X', 'Y', 'iZ', 'Z']] = [t, Fluo_T[t], x_contact, y_contact, iz_contact, iz_contact*dz]
        dfContact.loc[t, ['Xin', 'Yin', 'Zin', 'Xout', 'Yout', 'Zout']] = [x_in, y_in, dz*iz_mid-z_in, x_out, y_out, dz*iz_mid-z_out]
    
        
        # Zr = (-1) * (tsd_t['Zr_in']+tsd_t['Zr_out'])/2 # Taking the opposite cause the depthograph is upside down for these experiments
        # izmid = (nZ-1)//2 # If 11 points in Z, izmid = 5
        # zz = dz * np.arange(-izmid, +izmid, 1) # If 11 points in Z, zz = [-5, -4, ..., 4, 5] * dz
        # zzf = zz + Zr # Coordinates zz of bf images with respect to Zf, the Z of the focal point of the deptho
        # zzeq = zz + Zr + DZo - DZb # Coordinates zz of fluo images with respect to Zeq, the Z of the equatorial plane of the bead
        
        # zi_contact = Akima1DInterpolator(zzeq, np.arange(len(zzeq)))(0)
        
        A_contact_map = 0
        foundContact = False
        
        for z in range(nZ):
            contactSlice = False
            i_zt = z + nZ*t
            I_zt = I[i_zt]
            
            #### 2.0 Locate cell
            (Yc, Xc), Rc = findCellInnerCircle(I_zt, withBeads = True, plot=False)
            print(z, Rc)
            if z >= 1 and Rc < 8.5*SCALE_100X_ZEN:
                print('done this')
                # (Xc, Yc, Rc) = previousCircleXYR
                (Xc, Yc, Rc) = (I_zt.shape[1]//2, I_zt.shape[0]//2, 9.5*SCALE_100X_ZEN)
            if z == 0 and Rc < 8.5*SCALE_100X_ZEN:
                print('done that')
                (Xc, Yc, Rc) = (I_zt.shape[1]//2, I_zt.shape[0]//2, 9.5*SCALE_100X_ZEN)
                
            Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
            
            #### 2.1 Warp
            warped = skm.transform.warp_polar(I_zt, center=(Yc, Xc), radius=warp_radius, #Rc*1.2, 
                                              output_shape=None, scaling='linear', channel_axis=None)
            # max_values = np.max(warped[:,Rc0-inPix_set:Rc0+outPix_set+1], axis=1)
                
            #### 2.2 Viterbi Smoothing
            edge_viterbi = ViterbiEdge(warped, Rc0, inPix_set, outPix_set, 2, relative_height_virebi)
            edge_viterbi_unwarped = np.array(unwarpRA(np.array(edge_viterbi), Angles, Xc, Yc)) # X, Y
            
            N_it_viterbi = 2
            for i in range(N_it_viterbi):
    
                #### Extra iteration
                # x.0 Locate cell
                (Xc, Yc), Rc = fitCircle(np.array([edge_viterbi_unwarped[1], edge_viterbi_unwarped[0]]).T, loss = 'huber')        
                Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
                
                # x.1 Warp
                warped = skm.transform.warp_polar(I_zt, center=(Yc, Xc), radius=warp_radius, #Rc*1.4, 
                                                  output_shape=None, scaling='linear', channel_axis=None)
                warped = skm.util.img_as_uint(warped)
                w_ny, w_nx = warped.shape
                Angles = np.arange(0, 360, 1)
                max_values = np.max(warped[:,Rc0-inPix_set:Rc0+outPix_set+1], axis=1)
                inPix, outPix = inPix_set, outPix_set
                
                # x.3 Viterbi Smoothing
                edge_viterbi = ViterbiEdge(warped, Rc0, inPix, outPix, blur_parm, relative_height_virebi)
                edge_viterbi_unwarped = unwarpRA(np.array(edge_viterbi), Angles, Xc, Yc)
                arrayViterbiContours[t, z, :] = np.array([edge_viterbi_unwarped[0], edge_viterbi_unwarped[1]]).T
            
            # fig, ax = plt.subplots(1, 1)
            # ax.imshow(I_zt, cmap='gray', aspect='equal')
            # ax.plot(edge_viterbi_unwarped[0], edge_viterbi_unwarped[1], c='g', ls = '--')
            # ax.plot(Xc, Yc, 'go')
            # CC = circleContour_V2((Yc, Xc), Rc)
            # [yy_circle, xx_circle] = CC.T
            # ax.plot(xx_circle, yy_circle, 'b--')
            # ax.set_title('Detected contours')
            # fig.tight_layout()
            # plt.show()
    
            # #### 3.0 Locate cell
            # (Xc, Yc), Rc = fitCircle(np.array([edge_viterbi_unwarped[1], edge_viterbi_unwarped[0]]).T, loss = 'huber')
            # previousCircleXYR = (Xc, Yc, Rc)
            # Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
            
            # #### 3.1 Warp
            # warped = skm.transform.warp_polar(I_zt, center=(Yc, Xc), radius=warp_radius, #Rc*1.4, 
            #                                   output_shape=None, scaling='linear', channel_axis=None)
            # warped = skm.util.img_as_uint(warped)
            # w_ny, w_nx = warped.shape
            # Angles = np.arange(0, 360, 1)
            # max_values = np.max(warped[:,Rc0-inPix_set:Rc0+outPix_set+1], axis=1)
            
            # # #### 3.2 Interp in X
            # # warped_interp = ufun.resize_2Dinterp(warped, fx=interp_factor_x, fy=1)
            # # warped = warped_interp
            # # w_nx, Rc0 = w_nx * interp_factor_x, Rc0 * interp_factor_x
            # # inPix, outPix = inPix_set * interp_factor_x, outPix_set * interp_factor_x
            # # R_contact *= interp_factor_x
            # # blur_parm *= interp_factor_x
            
            # inPix, outPix = inPix_set, outPix_set
            
            # #### 3.3 Viterbi Smoothing
            # edge_viterbi = ViterbiEdge(warped, Rc0, inPix, outPix, blur_parm, relative_height_virebi)
            # # edge_viterbi_unwarped = unwarpRA(np.array(edge_viterbi)/interp_factor_x, Angles, Xc, Yc)
            # edge_viterbi_unwarped = unwarpRA(np.array(edge_viterbi), Angles, Xc, Yc)
            # arrayViterbiContours[t, z, :] = np.array([edge_viterbi_unwarped[0], edge_viterbi_unwarped[1]]).T
            
            
            R_contact, A_contact = warpXY(x_contact*SCALE_100X_ZEN, y_contact*SCALE_100X_ZEN, Xc, Yc)
            if (z == np.round(iz_contact)) or (not foundContact and z == nZ-1):
                contactSlice = True
                foundContact = True
                # print(z, R_contact, A_contact)
                A_contact_map = A_contact
                dfContact.loc[t, ['R', 'A']] = [R_contact, A_contact]
            
            dfContact.loc[t, ['Xc', 'Yc', 'Rc']] += np.array([(Xc/SCALE_100X_ZEN)/(nZ), (Yc/SCALE_100X_ZEN)/(nZ), (Rc/SCALE_100X_ZEN)/(nZ)])
            
            #### 4. Append the contact profile matrix
            # profileMatrix = np.zeros((nT, nZ, N_profilesMatrix, warp_radius))
            halfN = N_profilesMatrix//2
            tmpWarp = np.copy(warped)
            A_contact_r = round(A_contact)
            dA = A_contact_r - A_contact
            if   A_contact_r < halfN+3:
                tmpWarp = np.roll(tmpWarp, halfN+3, axis=0) 
                tmp_A_contact   = A_contact   + (halfN+3)
                tmp_A_contact_r = A_contact_r + (halfN+3)
                print('roll forward')
                print(A_contact_r, tmp_A_contact_r)
            elif A_contact_r > 360 - (halfN+3):
                tmpWarp = np.roll(tmpWarp, -(halfN+3), axis=0) 
                tmp_A_contact   = A_contact   - (halfN+3)
                tmp_A_contact_r = A_contact_r - (halfN+3)
                print('roll backward')
                print(A_contact_r, tmp_A_contact_r)
            else:
                tmp_A_contact   = A_contact
                tmp_A_contact_r = A_contact_r
        
            
            try:
                smallTmpWarp = tmpWarp[tmp_A_contact_r - (halfN+3):tmp_A_contact_r + (halfN+3)+1, :]
                tform = skm.transform.EuclideanTransform(translation=(0, dA))
                smallTmpWarp_tformed = skm.util.img_as_uint(skm.transform.warp(smallTmpWarp, tform))
                
                Ltmp = len(smallTmpWarp_tformed)
                smallWarp = smallTmpWarp_tformed[Ltmp//2 - halfN:Ltmp//2 + halfN+1, :]
                
                profileMatrix[t, z] = smallWarp
                
            except:
                print("profileMatrix Error")
                print(A_contact_r)
                print(dA)
                print(tmpWarp.shape)
                print(smallTmpWarp.shape)
                print(smallTmpWarp_tformed.shape)
                print(smallWarp.shape)
                
            #### 6.1 Compute beads positions
            draw_InCircle = False
            draw_OutCircle = False
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
                # RINbeadContour *= interp_factor_x
                
            if h_OutCircle > 0 and h_OutCircle < 2*Rbead:
                draw_OutCircle = True
                a_OutCircle = (h_OutCircle*(2*Rbead-h_OutCircle))**0.5
                # OUTbeadContour = circleContour((round(y_out*SCALE_100X_ZEN), round(x_out*SCALE_100X_ZEN)), 
                #                                a_OutCircle*SCALE_100X_ZEN, I_zt.shape)
                OUTbeadContour = circleContour_V2((round(y_out*SCALE_100X_ZEN), round(x_out*SCALE_100X_ZEN)),
                                                  a_OutCircle*SCALE_100X_ZEN)
                XOUTbeadContour, YOUTbeadContour = OUTbeadContour[:, 1], OUTbeadContour[:, 0]
                ROUTbeadContour, AOUTbeadContour = warpXY(XOUTbeadContour, YOUTbeadContour, Xc, Yc)
                # ROUTbeadContour *= interp_factor_x
                
            
            #### 5. Analyse fluo cyto & background
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
    
            mask_removed_from_cyto = inside & (~mask_cyto)
            cell_filtered = cell_dilated & (~mask_removed_from_cyto)
            
            nb_pix = np.sum(cell_filtered.astype(int))
            total_val = np.sum(I_zt.flatten()[cell_filtered.flatten()])
            I_cell = total_val/nb_pix
            
            # Bead
            I_bead = np.nan
            if draw_InCircle:
                xx, yy = np.meshgrid(np.arange(I_zt.shape[1]), np.arange(I_zt.shape[0]))
                xc, yc = round(x_in*SCALE_100X_ZEN), round(y_in*SCALE_100X_ZEN)
                rc = a_InCircle*SCALE_100X_ZEN*0.9
                maskBead = ((xx-xc)**2 + (yy-yc)**2)**0.5 < rc
                I_bead = np.median(I_zt[maskBead])
            
            
            # Store
            fluoCyto[t, z] = I_cyto
            fluoCytoStd[t, z] = Std_cyto
            fluoBack[t, z] = I_back
            fluoCell[t, z] = I_cell
            fluoBeadIn[t, z] = I_bead
            
            #### 5. Define normalization functions
            def normalize_Icyto(x):
                return((x - I_back)/(I_cyto - I_back))
        
            def normalize_Icell(x):
                return((x - I_back)/(I_cell - I_back))
            
            def normalize_Icell2(x):
                return((x - I_cyto)/(I_cell - I_back))
            
            
            
            #### 5. Read values
            VEW=5
            max_values = normalize_Icell(max_values)
            viterbi_Npix_values = [np.sum(warped[Angles[i], edge_viterbi[i]-VEW//2:edge_viterbi[i] + (VEW//2 + 1)])/VEW for i in range(len(edge_viterbi))]
            viterbi_Npix_values = normalize_Icell(viterbi_Npix_values)
            intensityMap_t.append(viterbi_Npix_values)
            
            
            
            #### 6. Plot
            #### 6.1 Plot Normalization
            if PLOT_NORMAL and z==nZ//2 and t==nT//2:
                figN, axesN = plt.subplots(1,5, figsize = (20, 5))
                ax = axesN[0]
                ax.imshow(I_zt)
                ax.set_title('Original')
                ax = axesN[1]
                ax.imshow((mask_cyto & inside_eroded).astype(int))
                ax.set_title('Cytoplasm')
                ax = axesN[2]
                ax.imshow(cell_filtered.astype(int))
                ax.set_title('Whole cell')
                ax = axesN[3]
                ax.imshow(outside.astype(int))
                ax.set_title('Background')
                if draw_InCircle:
                    ax = axesN[4]
                    ax.imshow(maskBead.astype(int))
                    ax.set_title(f'I_bead = {I_bead:.0f}')
                
                figN.tight_layout()
                ufun.simpleSaveFig(figN, 'ExampleNormalization', figSavePath, '.png', 150)
                   
            
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
                ax.plot(edge_viterbi_unwarped[0], edge_viterbi_unwarped[1], c='chartreuse', ls = '--', zorder=6)
                ax.plot(Xc, Yc, 'go')
                CC = circleContour_V2((Yc, Xc), Rc)
                [yy_circle, xx_circle] = CC.T
                ax.plot(xx_circle, yy_circle, ls='--', c='cyan', lw=0.5)
                if contactSlice:
                    style = '-'
                else:
                    style = ':'
                if draw_InCircle:
                    ax.plot(XINbeadContour, YINbeadContour, c='r', ls=style)
                if draw_OutCircle:
                    ax.plot(XOUTbeadContour, YOUTbeadContour, c='gold', ls=style)
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
                if contactSlice:
                    style = '-'
                else:
                    style = ':'
                if draw_InCircle:
                    ax.plot(RINbeadContour, AINbeadContour, c='dodgerblue', ls=style)
                if draw_OutCircle:
                    mask = (ROUTbeadContour < w_nx)
                    ROUTbeadContour = ROUTbeadContour[mask]
                    AOUTbeadContour = AOUTbeadContour[mask]
                    ax.plot(ROUTbeadContour, AOUTbeadContour, c='darkturquoise', ls=style)
                ax.set_title('Viterbi Edge')
                ax.set_xlabel('Radial pix')
                
                # 3.1 - Show the intensity profiles for Viterbi
                ax = figV2.add_subplot(spec[:, 4], sharey = ax1)
                step1 = 10
                ax.plot(inBorderA, Angles, c='k', ls='--', marker='')
                ax.plot(outBorderA, Angles, c='k', ls='--', marker='')
                ax.plot(edge_viterbi[::step1], Angles[::step1], ls='', c='cyan', marker='^', ms = 4, mec = 'k', zorder=5)
                ax.set_title('Viterbi profiles')
                ax.set_xlabel('Radial pix')
                
                cList_viridis = plt.cm.viridis(np.linspace(0.05, 0.95, w_ny//step1))
                ax.set_prop_cycle(plt.cycler("color", cList_viridis))
                for a in Angles[::step1]:
                    try:
                        profile = normalize_Icell(warped[a,:])*step1
                        RR = np.arange(len(profile))
                        ax.plot(RR, a - profile)
                    except:
                        pass
    
                # 3.2 - Zoom on the intensity profiles for Viterbi
                ax = figV2.add_subplot(spec[:, 5], sharey = ax1)
                step2 = 20
                ax.plot(edge_viterbi[::step2], Angles[::step2], ls='', c='cyan', marker='^', ms = 4, mec = 'k', zorder=5)
                cList_viridis = plt.cm.viridis(np.linspace(0.05, 0.95, w_ny//step2))
                ax.set_prop_cycle(plt.cycler("color", cList_viridis))
                ax.set_title('Viterbi profiles')
                ax.set_xlabel('Radial pix')
                maxWarped = np.max(warped)
                normWarped = warped * 10 * step2/maxWarped
                for a in Angles[::step2]:    
                    try:
                        # profile = normWarped[a, Rc0-inPix:Rc0+outPix] - np.min(normWarped[a, Rc0-inPix:Rc0+outPix])
                        profile = normalize_Icell(warped[a,Rc0-inPix:Rc0+outPix])*step2
                        RR = np.arange(Rc0-inPix, Rc0+outPix, 1)
                        ax.plot(RR, a - profile)
                    except:
                        pass
    
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
            
    
        #### Out of the Z loop here
        
        #### 6.3 Map and contact point
        if PLOT_MAP:
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
        
        #### 7. Save results 1/2
        if SAVE_RESULTS:
            intensityMap_t = np.array(intensityMap_t).astype(float)
            np.savetxt(os.path.join(dstDir, f'{cellId}_Map_t{t+1:.0f}.txt'), intensityMap_t, fmt='%.5f')
    
    #### Out of the T loop here
    
    #### 7. Save results 2/2
    if SAVE_RESULTS:
        dfContact.to_csv(os.path.join(dstDir, f'{cellId}_dfContact.csv'), sep=';', index=False)
        np.save(os.path.join(dstDir, f'{cellId}_xyContours.npy'), arrayViterbiContours)
        np.save(os.path.join(dstDir, f'{cellId}_profileMatrix.npy'), profileMatrix)
        np.savetxt(os.path.join(dstDir, f'{cellId}_fluoCyto.txt'), fluoCyto, fmt='%.1f')
        np.savetxt(os.path.join(dstDir, f'{cellId}_fluoCytoStd.txt'), fluoCytoStd, fmt='%.3f')
        np.savetxt(os.path.join(dstDir, f'{cellId}_fluoBack.txt'), fluoBack, fmt='%.1f')
        np.savetxt(os.path.join(dstDir, f'{cellId}_fluoCell.txt'), fluoCell, fmt='%.1f')
        np.savetxt(os.path.join(dstDir, f'{cellId}_fluoBeadIn.txt'), fluoBeadIn, fmt='%.1f')
        
        
    plt.ion()
    
    print(f'Done for {cellId} :D')
    

#### Call the function

# date = '24-02-27'

# # cell = 'C6'
# # doFluoAnalysis(date, cell,
# #                 SAVE_RESULTS = True,
# #                 PLOT_WARP = True, PLOT_MAP = True, PLOT_NORMAL = True)
    
# cells = ['C2', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']
# for cell in cells:
#     doFluoAnalysis(date, cell,
#                    SAVE_RESULTS = True,
#                    PLOT_WARP = True, PLOT_MAP = True, PLOT_NORMAL = True)

#### Call the function

# date = '24-06-14'

# cells = ['C' + str(i) for i in range(1, 9)]
# for cell in cells:
#     if cell != 'C10':
#         doFluoAnalysis(date, cell,
#                        SAVE_RESULTS = True,
#                        PLOT_WARP = True, PLOT_MAP = True, PLOT_NORMAL = True)



# %%% Open profiles and analyze

# %%%% Load files

# date = '24-02-27'
# date = '24-05-24'
date = '24-06-14'

date2 = dateFormat(date)

cell = 'C2'
cellId = f'{date}_{cell}'
specif = '' # '_off4um' #'_off4um' #'_off4um'
specifFolder = '' # 'M2-20um/B5mT/' # 'M2-20um/B5mT/' # 'M2-20um/Bstairs/' # 'M2-20um/B5mT/'

srcDir = f'D:/MagneticPincherData/Raw/{date2}_Chameleon/{specifFolder}Clean_Fluo_BF/M1'
# fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_Fluo.tif'

dstDir =  os.path.join(srcDir, 'Results_Fluo', f'{cell}')

fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_Fluo.tif'
fluoPath = os.path.join(srcDir, fluoName)

OMEfilepath = os.path.join(dictPaths['sourceDirPath'], f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_OME.txt')
CTZ_tz = OMEDataParser(OMEfilepath)
[nC, nT, nZ] = CTZ_tz.shape[:3]
# BF_T = CTZ_tz[0,:,:,0].flatten() * 1000

resFluoDir =  dstDir
figSavePath = dstDir

dfContact = pd.read_csv(os.path.join(resFluoDir, f'{cellId}_DfContact.csv'), sep=None, engine='python')
fluoCyto = np.loadtxt(os.path.join(resFluoDir, f'{cellId}_fluoCyto.txt'))
fluoBack = np.loadtxt(os.path.join(resFluoDir, f'{cellId}_fluoBack.txt'))
fluoCytoStd = np.loadtxt(os.path.join(resFluoDir, f'{cellId}_fluoCytoStd.txt'))
fluoCell = np.loadtxt(os.path.join(resFluoDir, f'{cellId}_fluoCell.txt'))
fluoBeadIn = np.loadtxt(os.path.join(resFluoDir, f'{cellId}_fluoBeadIn.txt'))
arrayViterbiContours = np.load(os.path.join(resFluoDir, f'{cellId}_xyContours.npy'))
profileMatrix = np.load(os.path.join(resFluoDir, f'{cellId}_profileMatrix.npy'))
# edgeMatrix = np.load(os.path.join(resFluoDir, f'{cellId}_edgeMatrix.npy'))

# %%%% Plot for Manuscript M&M

def plotActinQuantity(cellId, t, profileMatrix, dfContact,
                  fluoCell, fluoCyto, fluoBack, fluoBeadIn,
                  figSavePath):
    pM = profileMatrix[t]
    approxR = round(dfContact.loc[dfContact['iT']==t, 'R'].values[0])
    iz = dfContact.loc[dfContact['iT']==t, 'iZ'].values[0]
    izr = round(iz)
    # Icell, Icyto, Iback = fluoCell[t, izr], fluoCyto[t, izr], fluoBack[t, izr]
    
    list_Q_fbL = []
    list_Q_fbN = []
    list_Q_vb = []
    list_Q_gf = []
    list_W_vb = []
    list_S_gf = []
    list_R2_gf = []
    
    nA_scan = 5
    nZ_scan = 3
    nZ_scanCount = 0
    
    for z in range(nZ_scan):
        iiz = z - nZ_scan + 1
        
        if (izr + iiz) >= 0:
            nZ_scanCount += 1
            
            for a in range(nA_scan):
                iia = a - nA_scan//2
                Ac = pM.shape[1]//2
            
                # print(izr + iiz, Ac + iia)
                profile = pM[izr + iiz, Ac + iia, :] 
                eR = (approxR - 5) + np.argmax(profile[approxR-5:approxR+20])
                Np = len(profile)
                
                Icell, Icyto, Iback = fluoCell[t, izr + iiz], fluoCyto[t, izr + iiz], fluoBack[t, izr + iiz]
                Ibeadin = fluoBeadIn[t, izr + iiz]
                
                
                def normalize_Icell(x):
                    return((x - Iback)/(Icell - Iback))
                profile_n = normalize_Icell(profile)
    
                eV = profile_n[eR]
                
                bead_in_level = normalize_Icell(Ibeadin) * 0.8 + eV * 0.2
                # bead_out_level = normalize_Icell(Ibeadin)
                bead_out_level = 0.5*eV + 0.5*np.median(profile_n[-5:])
                ri = eR - ufun.findFirst(True, profile_n[:eR][::-1]<bead_in_level)
                rf = eR + ufun.findFirst(True, profile_n[eR:]<bead_out_level)
                # ri = eR - 10
                # rf = eR + 5
                
                # if eR-ri <= 5:
                #     bead_in_level = normalize_Icell(Ibeadin)
                #     ri = eR - ufun.findFirst(True, profile_n[:eR][::-1]<bead_in_level)
                # if rf-eR <= 3:
                #     bead_out_level = 0.5*eV + 0.5*np.median(profile_n[-5:])
                #     rf = eR + ufun.findFirst(True, profile_n[eR:]<bead_out_level)
                # print(eR, ri, rf)
                
                #### Calculation 1
                n_in, n_out = 10, 5
                Q_fbL = np.sum(profile_n[eR-n_in:eR+n_out+1])
                list_Q_fbL.append(Q_fbL)
                
                #### Calculation 1
                n_inN, n_outN = 5, 3
                Q_fbN = np.sum(profile_n[eR-n_inN:eR+n_outN+1])
                list_Q_fbN.append(Q_fbN)
                
                #### Calculation 2
                Q_vb = np.sum(profile_n[ri:rf+1])
                list_Q_vb.append(Q_vb)
                list_W_vb.append(1+rf-ri)
    
                #### Calculation 3
                ### bead_in_level = 0.9
                ### bead_out_level = 0.95
                ###(2*bead_in_level+eV)/3)
                def Gfit(x, m, Q, s):
                    # m = eR
                    return((Q/(s*(2*np.pi)**0.5)) * np.exp(-(x-m)**2/(2*s**2)))
                
                try:
                    x_data = np.arange(ri, rf)
                    y_data = profile_n[ri:rf]
                    popt, pcov = optimize.curve_fit(Gfit, np.arange(ri, rf), profile_n[ri:rf], p0=[eR, Q_vb, (rf-ri)],
                                                    bounds = ([ri, 0, 0], [rf, 2*Q_vb, 4*(rf-ri)]))
                    m_gf, Q_gf, s_gf = popt
                    xfit = np.linspace(eR-20, eR+20, 200)
                    yfit = Gfit(xfit, m_gf, Q_gf, s_gf)
                    residuals = y_data - Gfit(x_data, m_gf, Q_gf, s_gf)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y_data - np.mean(y_data))**2)
                    r_squared = 1 - (ss_res / ss_tot)
                    list_Q_gf.append(Q_gf)
                    list_S_gf.append(s_gf)
                    list_R2_gf.append(r_squared)
                    fitError = False
                    
                except:
                    fitError = True
                
                # ax = axes[nZ_scan-z-1, a]            
                # ax.plot(np.arange(Np), profile_n)
                # ax.axvline(eR, c='r', ls='--')
                # # 1
                # ax.axvline(eR-n_in, c='cyan', ls='--', lw=1)
                # ax.axvline(eR+n_out+1, c='cyan', ls='--', lw=1, label=f'Fixed boundaries: Q={Q_fbL:.1f}')
                # # 2
                # ax.axhline(bead_in_level, c='green', ls='-.', lw=1)
                # ax.axhline(bead_out_level, c='green', ls='-.', lw=1)
                # # ax.axhline(normalize_Icell(Ibeadin), c='orange', ls='-.', lw=0.8)
                # ax.axvline(ri, c='orange', ls='--', lw=1)
                # ax.axvline(rf, c='orange', ls='--', lw=1, label=f'Variable boundaries: Q={Q_vb:.1f}') #, label='Gaussian boundaries')
                # # 3
                # if not fitError:
                #     ax.plot(xfit, yfit, c='gold', label=f'Gaussian fit: Q={Q_gf:.1f}\nR2={r_squared:.2f}')
                
                # ax.legend(fontsize=8, loc='lower left')
                
                # if a==0:
                #     ax.set_ylabel(f'z={izr + iiz:.0f}; zr={iiz:.0f} (steps)')
                # if z==nZ_scan-1:
                #     ax.set_title(f'a = {iia:.0f} (°)')
    
    # fig.suptitle(f'{cellId} - t={t+1:.0f}')
    # fig.tight_layout()
    # ufun.simpleSaveFig(fig, f'ProfileMetrics_t{t+1:.0f}', figSavePath, '.png', 150)
    # plt.show()
    
    ### Manuscript Plot
    gs.set_manuscript_options_jv()
    figM = plt.figure(figsize = (10/gs.cm_in, 8/gs.cm_in))#, layout="constrained")
    spec = figM.add_gridspec(2, 3)
    # figM, axM = plt.subplots(1, 3, figsize=(17/gs.cm_in, 6/gs.cm_in))
    
    Ac = pM.shape[1]//2
    profile = pM[izr-1, Ac, :]  
    eR = (approxR - 5) + np.argmax(profile[approxR-5:approxR+20])
    Np = len(profile)
    
    Icell, Icyto, Iback = fluoCell[t, izr + iiz], fluoCyto[t, izr + iiz], fluoBack[t, izr + iiz]
    Ibeadin = fluoBeadIn[t, izr + iiz]
    
    def normalize_Icell(x):
        return((x - Iback)/(Icell - Iback))
    profile_n = normalize_Icell(profile)

    eV = profile_n[eR]
    
    bead_in_level = normalize_Icell(Ibeadin) * 0.8 + eV * 0.2
    bead_out_level = 0.5*eV + 0.5*np.median(profile_n[-5:])
    ri = eR - ufun.findFirst(True, profile_n[:eR][::-1]<bead_in_level)
    rf = eR + ufun.findFirst(True, profile_n[eR:]<bead_out_level)

    # Calculation 1
    n_in, n_out = 10, 5
    Q_fbL = np.sum(profile_n[eR-n_in:eR+n_out+1])
    
    # Calculation 1
    n_inN, n_outN = 5, 3
    Q_fbN = np.sum(profile_n[eR-n_inN:eR+n_outN+1])
    
    # Calculation 2
    Q_vb = np.sum(profile_n[ri:rf+1])

    # Calculation 3
    ### bead_in_level = 0.9
    ### bead_out_level = 0.95
    ###(2*bead_in_level+eV)/3)
    def Gfit(x, m, Q, s):
        # m = eR
        return((Q/(s*(2*np.pi)**0.5)) * np.exp(-(x-m)**2/(2*s**2)))
    
    try:                
        popt, pcov = optimize.curve_fit(Gfit, np.arange(ri, rf), profile_n[ri:rf], p0=[eR, Q_vb, (rf-ri)],
                                        bounds = ([ri, 0, 0], [rf, 2*Q_vb, 4*(rf-ri)]))
        m_gf, Q_gf, s_gf = popt
        xfit = np.linspace(eR-20, eR+20, 200)
        yfit = Gfit(xfit, m_gf, Q_gf, s_gf)
        list_Q_gf.append(Q_gf)
        list_S_gf.append(s_gf)
        fitError = False
    except:
        fitError = True
        
    # ax = axM[0]
    ax = figM.add_subplot(spec[0,:])
    ax.imshow(pM[izr-1])
    ax.axhline(Ac, color='r', ls='--', lw=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    ax = figM.add_subplot(spec[1,:])          
    ax.plot(np.arange(Np)/7.4588, profile_n)
    ax.axvline(eR/7.4588, c='r', ls='-', lw=1)
    # 1
    # ax.axvline(eR-n_in/7.4588, c='cyan', ls='--', lw=1)
    # ax.axvline((eR+n_out+1)/7.4588, c='cyan', ls='--', lw=1, label=f'Fixed boundaries: Q={Q_fbL:.1f}')
    # 2
    # ax.axhline(bead_in_level, c='green', ls='-.', lw=1)
    # ax.axhline(bead_out_level, c='green', ls='-.', lw=1)
    # ax.axhline(normalize_Icell(Ibeadin), c='orange', ls='-.', lw=0.8)
    # ax.axvline(ri, c='orange', ls='--', lw=1)
    # ax.axvline(rf, c='orange', ls='--', lw=1, label=f'Variable boundaries: Q={Q_vb:.1f}') #, label='Gaussian boundaries')
    # 3
    if not fitError:
        ax.plot(xfit/7.4588, yfit, c='brown', ls='--', lw = 1, label=f'Gaussian fit:\nQ={Q_gf:.1f}')
    
    ax.legend(fontsize=7)
    # ax.set_title('Actin quantification')
    ax.set_ylabel(f'Normalized intensity', fontsize = 8)
    ax.set_xlabel(f'r (µm)', fontsize = 8)
    ax.set_xlim([0, 12.3])
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.grid(axis = 'y')
    # if z==nZ_scan-1:
    #     ax.set_title(f'a = {iia:.0f} (°)')
    figDir = "D:/MagneticPincherData/Figures/PhysicsDataset"
    ufun.archiveFig(figM, name = f'ProfileMetrics_t{t+1:.0f}_V2', ext = '.pdf', dpi = 100,
                    figDir = figDir, figSubDir = 'E-h-Fluo', cloudSave = 'flexible')
    
    plt.show()
            
    avg_Q_fbL = np.mean(np.sort(list_Q_fbL)[2:-3])
    avg_Q_fbN = np.mean(np.sort(list_Q_fbN)[2:-3])
    avg_Q_vb = np.mean(np.sort(list_Q_vb)[2:-3])
    avg_Q_gf = np.mean(np.sort(list_Q_gf)[2:-3])
    avg_S_gf = np.mean(np.sort(list_S_gf)[2:-3])
    avg_W_vb = np.mean(np.sort(list_W_vb)[2:-3])
    
    dfContact.loc[dfContact['iT']==t, 'Q_fbL'] = avg_Q_fbL
    dfContact.loc[dfContact['iT']==t, 'Q_fbN'] = avg_Q_fbN
    dfContact.loc[dfContact['iT']==t, 'Q_vb'] = avg_Q_vb
    dfContact.loc[dfContact['iT']==t, 'Q_gf'] = avg_Q_gf
    dfContact.loc[dfContact['iT']==t, 'S_gf'] = avg_S_gf
    dfContact.loc[dfContact['iT']==t, 'W_vb'] = avg_W_vb
    dfContact.loc[dfContact['iT']==t, 'nA_scan'] = nA_scan
    dfContact.loc[dfContact['iT']==t, 'nZ_scan'] = nZ_scanCount
    
    return(dfContact)

plotActinQuantity(cellId, 1, profileMatrix, dfContact,
                  fluoCell, fluoCyto, fluoBack, fluoBeadIn,
                  figSavePath)

# plotActinQuantity(cellId, t, profileMatrix, dfContact,
#                   fluoCell, fluoCyto, fluoBack, fluoBeadIn,
#                   figSavePath, comparePlot = False, PLOT_M = False)

# %%%% actin Quantitif

# def normalize_Icyto(x):
#     return((x - Iback)/(Icyto - Iback))

# def normalize_Icell(x):
#     return((x - Iback)/(Icell - Iback))

# def normalize_Icell2(x):
#     return((x - Icyto)/(Icell - Iback))

# for t in range(nT):
#     actinQuantity(1, profileMatrix, dfContact,
#                   fluoCell, fluoCyto, fluoBack)


def actinQuantity(cellId, t, profileMatrix, dfContact,
                  fluoCell, fluoCyto, fluoBack, fluoBeadIn,
                  figSavePath, comparePlot = False, PLOT_M = False):
    pM = profileMatrix[t]
    approxR = round(dfContact.loc[dfContact['iT']==t, 'R'].values[0])
    iz = dfContact.loc[dfContact['iT']==t, 'iZ'].values[0]
    izr = round(iz)
    # Icell, Icyto, Iback = fluoCell[t, izr], fluoCyto[t, izr], fluoBack[t, izr]
    
    list_Q_fbL = []
    list_Q_fbN = []
    list_Q_vb = []
    list_Q_gf = []
    list_W_vb = []
    list_S_gf = []
    list_R2_gf = []
    
    nA_scan = 5
    nZ_scan = 3
    nZ_scanCount = 0
    fig, axes = plt.subplots(nZ_scan, nA_scan, figsize=(3*nA_scan, 2.5*nZ_scan), sharex='col', sharey='row')
    
    
    
    for z in range(nZ_scan):
        iiz = z - nZ_scan + 1
        
        if (izr + iiz) >= 0:
            nZ_scanCount += 1
            
            for a in range(nA_scan):
                iia = a - nA_scan//2
                Ac = pM.shape[1]//2
            
                # print(izr + iiz, Ac + iia)
                profile = pM[izr + iiz, Ac + iia, :] 
                eR = (approxR - 5) + np.argmax(profile[approxR-5:approxR+20])
                Np = len(profile)
                
                Icell, Icyto, Iback = fluoCell[t, izr + iiz], fluoCyto[t, izr + iiz], fluoBack[t, izr + iiz]
                Ibeadin = fluoBeadIn[t, izr + iiz]
                
                
                def normalize_Icell(x):
                    return((x - Iback)/(Icell - Iback))
                profile_n = normalize_Icell(profile)
    
                eV = profile_n[eR]
                
                bead_in_level = normalize_Icell(Ibeadin) * 0.8 + eV * 0.2
                # bead_out_level = normalize_Icell(Ibeadin)
                bead_out_level = 0.5*eV + 0.5*np.median(profile_n[-5:])
                ri = eR - ufun.findFirst(True, profile_n[:eR][::-1]<bead_in_level)
                rf = eR + ufun.findFirst(True, profile_n[eR:]<bead_out_level)
                # ri = eR - 10
                # rf = eR + 5
                
                # if eR-ri <= 5:
                #     bead_in_level = normalize_Icell(Ibeadin)
                #     ri = eR - ufun.findFirst(True, profile_n[:eR][::-1]<bead_in_level)
                # if rf-eR <= 3:
                #     bead_out_level = 0.5*eV + 0.5*np.median(profile_n[-5:])
                #     rf = eR + ufun.findFirst(True, profile_n[eR:]<bead_out_level)
                # print(eR, ri, rf)
                
                #### Calculation 1
                n_in, n_out = 10, 5
                Q_fbL = np.sum(profile_n[eR-n_in:eR+n_out+1])
                list_Q_fbL.append(Q_fbL)
                
                #### Calculation 1
                n_inN, n_outN = 5, 3
                Q_fbN = np.sum(profile_n[eR-n_inN:eR+n_outN+1])
                list_Q_fbN.append(Q_fbN)
                
                #### Calculation 2
                Q_vb = np.sum(profile_n[ri:rf+1])
                list_Q_vb.append(Q_vb)
                list_W_vb.append(1+rf-ri)
    
                #### Calculation 3
                ### bead_in_level = 0.9
                ### bead_out_level = 0.95
                ###(2*bead_in_level+eV)/3)
                def Gfit(x, m, Q, s):
                    # m = eR
                    return((Q/(s*(2*np.pi)**0.5)) * np.exp(-(x-m)**2/(2*s**2)))
                
                try:
                    x_data = np.arange(ri, rf)
                    y_data = profile_n[ri:rf]
                    popt, pcov = optimize.curve_fit(Gfit, np.arange(ri, rf), profile_n[ri:rf], p0=[eR, Q_vb, (rf-ri)],
                                                    bounds = ([ri, 0, 0], [rf, 2*Q_vb, 4*(rf-ri)]))
                    m_gf, Q_gf, s_gf = popt
                    xfit = np.linspace(eR-20, eR+20, 200)
                    yfit = Gfit(xfit, m_gf, Q_gf, s_gf)
                    residuals = y_data - Gfit(x_data, m_gf, Q_gf, s_gf)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y_data - np.mean(y_data))**2)
                    r_squared = 1 - (ss_res / ss_tot)
                    list_Q_gf.append(Q_gf)
                    list_S_gf.append(s_gf)
                    list_R2_gf.append(r_squared)
                    fitError = False
                    
                except:
                    fitError = True
                
                ax = axes[nZ_scan-z-1, a]            
                ax.plot(np.arange(Np), profile_n)
                ax.axvline(eR, c='r', ls='--')
                # 1
                ax.axvline(eR-n_in, c='cyan', ls='--', lw=1)
                ax.axvline(eR+n_out+1, c='cyan', ls='--', lw=1, label=f'Fixed boundaries: Q={Q_fbL:.1f}')
                # 2
                ax.axhline(bead_in_level, c='green', ls='-.', lw=1)
                ax.axhline(bead_out_level, c='green', ls='-.', lw=1)
                # ax.axhline(normalize_Icell(Ibeadin), c='orange', ls='-.', lw=0.8)
                ax.axvline(ri, c='orange', ls='--', lw=1)
                ax.axvline(rf, c='orange', ls='--', lw=1, label=f'Variable boundaries: Q={Q_vb:.1f}') #, label='Gaussian boundaries')
                # 3
                if not fitError:
                    ax.plot(xfit, yfit, c='gold', label=f'Gaussian fit: Q={Q_gf:.1f}\nR2={r_squared:.2f}')
                
                ax.legend(fontsize=8, loc='lower left')
                
                if a==0:
                    ax.set_ylabel(f'z={izr + iiz:.0f}; zr={iiz:.0f} (steps)')
                if z==nZ_scan-1:
                    ax.set_title(f'a = {iia:.0f} (°)')
    
    fig.suptitle(f'{cellId} - t={t+1:.0f}')
    fig.tight_layout()
    ufun.simpleSaveFig(fig, f'ProfileMetrics_t{t+1:.0f}', figSavePath, '.png', 150)
    plt.show()
    
    #### Manuscript Plot
    # if PLOT_M:
    #     figM, axM = plt.subplots(1, 3, figsize=(17/gs.cm_in, 5.5/gs.cm_in))
    #     Ac = pM.shape[1]//2
    #     profile = pM[izr + iiz, Ac + iia, :]  
    #     eR = (approxR - 5) + np.argmax(profile[approxR-5:approxR+20])
    #     Np = len(profile)
        
    #     Icell, Icyto, Iback = fluoCell[t, izr + iiz], fluoCyto[t, izr + iiz], fluoBack[t, izr + iiz]
    #     Ibeadin = fluoBeadIn[t, izr + iiz]
        
    #     def normalize_Icell(x):
    #         return((x - Iback)/(Icell - Iback))
    #     profile_n = normalize_Icell(profile)

    #     eV = profile_n[eR]
        
    #     bead_in_level = normalize_Icell(Ibeadin) * 0.8 + eV * 0.2
    #     bead_out_level = 0.5*eV + 0.5*np.median(profile_n[-5:])
    #     ri = eR - ufun.findFirst(True, profile_n[:eR][::-1]<bead_in_level)
    #     rf = eR + ufun.findFirst(True, profile_n[eR:]<bead_out_level)

    #     # Calculation 1
    #     n_in, n_out = 10, 5
    #     Q_fbL = np.sum(profile_n[eR-n_in:eR+n_out+1])
        
    #     # Calculation 1
    #     n_inN, n_outN = 5, 3
    #     Q_fbN = np.sum(profile_n[eR-n_inN:eR+n_outN+1])
        
    #     # Calculation 2
    #     Q_vb = np.sum(profile_n[ri:rf+1])

    #     # Calculation 3
    #     ### bead_in_level = 0.9
    #     ### bead_out_level = 0.95
    #     ###(2*bead_in_level+eV)/3)
    #     def Gfit(x, m, Q, s):
    #         # m = eR
    #         return((Q/(s*(2*np.pi)**0.5)) * np.exp(-(x-m)**2/(2*s**2)))
        
    #     try:                
    #         popt, pcov = optimize.curve_fit(Gfit, np.arange(ri, rf), profile_n[ri:rf], p0=[eR, Q_vb, (rf-ri)],
    #                                         bounds = ([ri, 0, 0], [rf, 2*Q_vb, 4*(rf-ri)]))
    #         m_gf, Q_gf, s_gf = popt
    #         xfit = np.linspace(eR-20, eR+20, 200)
    #         yfit = Gfit(xfit, m_gf, Q_gf, s_gf)
    #         list_Q_gf.append(Q_gf)
    #         list_S_gf.append(s_gf)
    #         fitError = False
    #     except:
    #         fitError = True
        
    #     ax = axM[2]            
    #     ax.plot(np.arange(Np)/7.4588, profile_n)
    #     ax.axvline(eR/7.4588, c='r', ls='-', lw=0.75)
    #     # 1
    #     # ax.axvline(eR-n_in/7.4588, c='cyan', ls='--', lw=1)
    #     # ax.axvline((eR+n_out+1)/7.4588, c='cyan', ls='--', lw=1, label=f'Fixed boundaries: Q={Q_fbL:.1f}')
    #     # 2
    #     # ax.axhline(bead_in_level, c='green', ls='-.', lw=1)
    #     # ax.axhline(bead_out_level, c='green', ls='-.', lw=1)
    #     # ax.axhline(normalize_Icell(Ibeadin), c='orange', ls='-.', lw=0.8)
    #     # ax.axvline(ri, c='orange', ls='--', lw=1)
    #     # ax.axvline(rf, c='orange', ls='--', lw=1, label=f'Variable boundaries: Q={Q_vb:.1f}') #, label='Gaussian boundaries')
    #     # 3
    #     if not fitError:
    #         ax.plot(xfit/7.4588, yfit, c='gold', ls='--', lw = 1, label=f'Gaussian fit:\nQ={Q_gf:.1f}')
        
    #     ax.legend(fontsize=6, loc='upper left')
    #     ax.set_title('Actin quantification')
    #     ax.set_ylabel(f'Normalized intensity', fontsize = 8)
    #     ax.set_xlabel(f'r (µm)', fontsize = 8)
    #     ax.tick_params(axis='both', which='major', labelsize=6)
    #     ax.grid(axis = 'y')
    #     # if z==nZ_scan-1:
    #     #     ax.set_title(f'a = {iia:.0f} (°)')
    #     figDir = "D:/MagneticPincherData/Figures/PhysicsDataset"
    #     ufun.archiveFig(figM, name = f'ProfileMetrics_t{t+1:.0f}', ext = '.pdf', dpi = 100,
    #                     figDir = figDir, figSubDir = 'E-h-Fluo', cloudSave = 'flexible')
            
            
    avg_Q_fbL = np.mean(np.sort(list_Q_fbL)[2:-3])
    avg_Q_fbN = np.mean(np.sort(list_Q_fbN)[2:-3])
    avg_Q_vb = np.mean(np.sort(list_Q_vb)[2:-3])
    avg_Q_gf = np.mean(np.sort(list_Q_gf)[2:-3])
    avg_S_gf = np.mean(np.sort(list_S_gf)[2:-3])
    avg_W_vb = np.mean(np.sort(list_W_vb)[2:-3])
    # print(f'Average score: Q_fb = {avg_Q_fb:.1f}')
    # print(f'Average score: Q_vb = {avg_Q_vb:.1f}')
    # print(f'Average score: Q_gf = {avg_Q_gf:.1f}')
    
    dfContact.loc[dfContact['iT']==t, 'Q_fbL'] = avg_Q_fbL
    dfContact.loc[dfContact['iT']==t, 'Q_fbN'] = avg_Q_fbN
    dfContact.loc[dfContact['iT']==t, 'Q_vb'] = avg_Q_vb
    dfContact.loc[dfContact['iT']==t, 'Q_gf'] = avg_Q_gf
    dfContact.loc[dfContact['iT']==t, 'S_gf'] = avg_S_gf
    dfContact.loc[dfContact['iT']==t, 'W_vb'] = avg_W_vb
    dfContact.loc[dfContact['iT']==t, 'nA_scan'] = nA_scan
    dfContact.loc[dfContact['iT']==t, 'nZ_scan'] = nZ_scanCount
    
    # Compare the metrics
    # if comparePlot:
    #     fig2, axes2 = plt.subplots(1, 3, figsize=(12, 5))
    #     ax=axes2[0]
    #     ax.plot(list_Q_fbL, list_Q_vb, 'bo')
    #     ax.set_xlabel('Fixed b.')
    #     ax.set_ylabel('Variable b.')
    #     ax=axes2[1]
    #     ax.plot(list_Q_fbL, list_Q_gf, 'go')
    #     ax.set_xlabel('Fixed b.')
    #     ax.set_ylabel('Gaussian f.')
    #     ax=axes2[2]
    #     ax.plot(list_Q_vb, list_Q_gf, 'ro')
    #     ax.set_xlabel('Variable b.')
    #     ax.set_ylabel('Gaussian f.')
    #     fig2.tight_layout()
    #     plt.show()
    
    return(dfContact)





# for t in range(nT):
#     dfContact = actinQuantity(t, profileMatrix, dfContact,
#                               fluoCell, fluoCyto, fluoBack, fluoBeadIn,
#                               figSavePath)
# dfContact.to_csv(os.path.join(dstDir, f'{cellId}_dfContact.csv'), sep=';', index=False)




# dfContact = actinQuantity(cellId, 5, profileMatrix, dfContact,
#                           fluoCell, fluoCyto, fluoBack, fluoBeadIn,
#                           figSavePath, PLOT_M = True)

# actinQuantity(cellId, t, profileMatrix, dfContact,
#                   fluoCell, fluoCyto, fluoBack, fluoBeadIn,
#                   figSavePath, comparePlot = False)

# %%%% As a function

def actinQuantity_multiCells(date, cell, specif = '', specifFolder = ''):
    date2 = dateFormat(date)

    cellId = f'{date}_{cell}'
    # specif = '_off4um' #'_off4um'
    # specifFolder = '' # 'M2-20um/B5mT/' # 'M2-20um/Bstairs/' # 'M2-20um/B5mT/'

    srcDir = f'D:/MagneticPincherData/Raw/{date2}_Chameleon/Clean_Fluo_BF/{specifFolder}'

    dstDir =  os.path.join(srcDir, 'Results_Fluo', f'{cell}')

    fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_Fluo.tif'
    fluoPath = os.path.join(srcDir, fluoName)

    OMEfilepath = os.path.join(srcDir, f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_OME.txt')
    CTZ_tz = OMEDataParser(OMEfilepath)
    [nC, nT, nZ] = CTZ_tz.shape[:3]

    resFluoDir =  dstDir
    figSavePath = dstDir

    dfContact = pd.read_csv(os.path.join(resFluoDir, f'{cellId}_DfContact.csv'), sep=None, engine='python')
    fluoCyto = np.loadtxt(os.path.join(resFluoDir, f'{cellId}_fluoCyto.txt'))
    fluoBack = np.loadtxt(os.path.join(resFluoDir, f'{cellId}_fluoBack.txt'))
    # fluoCytoStd = np.loadtxt(os.path.join(resFluoDir, f'{cellId}_fluoCytoStd.txt'))
    fluoCell = np.loadtxt(os.path.join(resFluoDir, f'{cellId}_fluoCell.txt'))
    fluoBeadIn = np.loadtxt(os.path.join(resFluoDir, f'{cellId}_fluoBeadIn.txt'))
    # arrayViterbiContours = np.load(os.path.join(resFluoDir, f'{cellId}_xyContours.npy'))
    profileMatrix = np.load(os.path.join(resFluoDir, f'{cellId}_profileMatrix.npy'))

    for t in range(nT):
        dfContact = actinQuantity(cellId, t, profileMatrix, dfContact,
                                  fluoCell, fluoCyto, fluoBack, fluoBeadIn,
                                  figSavePath)

    dfContact.to_csv(os.path.join(dstDir, f'{cellId}_dfContact.csv'), sep=';', index=False)
    
#### Call the function - 24-02-27

# date = '24-02-27'

# specif = '_off3-5um'
# specifFolder = ''
# cell = 'C2'
# actinQuantity_multiCells(date, cell, specif = specif, specifFolder = specifFolder)
# plt.close('all')


# specif = '_off4um'
# specifFolder = ''
# cells = ['C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12'] # 
# for cell in cells:
#     actinQuantity_multiCells(date, cell, specif = specif, specifFolder = specifFolder)
#     plt.close('all')


# date = '24-02-27'
# specif = '_off4um'
# specifFolder = ''
# cells = ['C6'] # 
# for cell in cells:
#     actinQuantity_multiCells(date, cell, specif = specif, specifFolder = specifFolder)
#     plt.close('all')
    
#### Call the function - 24-05-24

# date = '24-05-24'

# specif = ''
# specifFolder = 'M2-20um/B5mT/' # 'M2-20um/B5mT/' # 'M2-20um/Bstairs/' # 'M2-20um/B5mT/'
# cells = ['C1', 'C2', 'C4', 'C6', 'C7', 'C8', 'C13'] # 'C1', 'C2', 'C4', 
# for cell in cells:
#     actinQuantity_multiCells(date, cell, specif = specif, specifFolder = specifFolder)
#     plt.close('all')
    
#### Call the function - 24-05-24 - stairs

# date = '24-05-24'

# specif = ''
# specifFolder = 'M2-20um/Bstairs/' # 'M2-20um/B5mT/' # 'M2-20um/Bstairs/' # 'M2-20um/B5mT/'
# cells = ['C9', 'C10', 'C11', 'C12'] # 'C1', 'C2', 'C4', 
# for cell in cells:
#     actinQuantity_multiCells(date, cell, specif = specif, specifFolder = specifFolder)
#     plt.close('all')

#### Call the function - 24-06-14

date = '24-06-14'

specif = ''
specifFolder = 'M1/' # 'M2-20um/B5mT/' # 'M2-20um/Bstairs/' # 'M2-20um/B5mT/'
cells = ['C' + str(i) for i in range(1, 9)]
for cell in cells:
    # if cell != 'C10':
    try:
        actinQuantity_multiCells(date, cell, specif = specif, specifFolder = specifFolder)
        plt.close('all')
    except:
        continue

# %%%% Cross results - '24-06-14'

# srcDir = 'D:/MagneticPincherData/Raw/24.02.27_Chameleon/Clean_Fluo_BF'
date = '24-06-14'
specif = '' #'_off4um'
specifFolder = 'M1/' # 'M2-20um/B5mT/' # 'M2-20um/Bstairs/' # 'M2-20um/B5mT/'
date2 = dateFormat(date)
cells = ['C' + str(i) for i in range(1, 20)] #+ ['C' + str(i) for i in range(6, 9)]
Dmoy = 4.4995

list_df = []

for cell in cells:
    try:
        cellId = f'{date}_{cell}'
        srcDir = f'D:/MagneticPincherData/Raw/{date2}_Chameleon/Clean_Fluo_BF/{specifFolder}'
    
        dstDir =  os.path.join(srcDir, 'Results_Fluo', f'{cell}')
    
        fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_Fluo.tif'
        fluoPath = os.path.join(srcDir, fluoName)
    
        OMEfilepath = os.path.join(srcDir, f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_OME.txt')
        CTZ_tz = OMEDataParser(OMEfilepath)
        [nC, nT, nZ] = CTZ_tz.shape[:3]
        
        # fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_Fluo.tif'
        bfName = f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_BF.tif'
        
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
        # df = mdf[['D2', 'D3', 'F', 'I_viterbi_w5', 'I_viterbi_inout7-3', 'I_viterbi_flexW']]
        df = mdf[['D2', 'D3', 'F', 'Q_fbL', 'Q_fbN', 'Q_vb', 'W_vb', 'Q_gf', 'S_gf', 'nA_scan', 'nZ_scan']]
    
        df['cell'] = cell_col
        list_df.append(df)
    except:
        continue


global_df = pd.concat(list_df)
global_df = global_df.drop(global_df[(global_df['cell'] == 'C5') & (global_df['D2'] < 4.6)].index)

global_df['h2'] = global_df['D2']-Dmoy
global_df['h3'] = global_df['D3']-Dmoy

global_df.to_csv(os.path.join(resCrossDir, f'{date}_global_DfMerged.csv'), sep=';', index=False)



# %%%% Cross results - '24-02-27'

# srcDir = 'D:/MagneticPincherData/Raw/24.02.27_Chameleon/Clean_Fluo_BF'
date = '24-02-27'
specif = '_off4um' #'_off4um'
specifFolder = '' # 'M2-20um/B5mT/' # 'M2-20um/Bstairs/' # 'M2-20um/B5mT/'
date2 = dateFormat(date)
cells = ['C2'] + ['C' + str(i) for i in range(5, 13)]
# cells = ['C11']
Dmoy = 4.4995

list_df = []

for cell in cells:
    if cell == 'C2':
        specif = '_off3-5um'
    else:
        specif = '_off4um'
    cellId = f'{date}_{cell}'
    srcDir = f'D:/MagneticPincherData/Raw/{date2}_Chameleon/{specifFolder}Clean_Fluo_BF'

    dstDir =  os.path.join(srcDir, 'Results_Fluo', f'{cell}')

    fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_Fluo.tif'
    fluoPath = os.path.join(srcDir, fluoName)

    OMEfilepath = os.path.join(srcDir, f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_OME.txt')
    CTZ_tz = OMEDataParser(OMEfilepath)
    [nC, nT, nZ] = CTZ_tz.shape[:3]
    
    # fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_Fluo.tif'
    bfName = f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_BF.tif'
    
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
    # df = mdf[['D2', 'D3', 'F', 'I_viterbi_w5', 'I_viterbi_inout7-3', 'I_viterbi_flexW']]
    df = mdf[['D2', 'D3', 'F', 'Q_fbL', 'Q_fbN', 'Q_vb', 'W_vb', 'Q_gf', 'S_gf', 'nA_scan', 'nZ_scan']]

    df['cell'] = cell_col
    list_df.append(df)


global_df = pd.concat(list_df)
global_df = global_df.drop(global_df[(global_df['cell'] == 'C5') & (global_df['D2'] < 4.6)].index)

global_df['h2'] = global_df['D2']-Dmoy
global_df['h3'] = global_df['D3']-Dmoy

global_df.to_csv(os.path.join(resCrossDir, f'{date}_global_DfMerged.csv'), sep=';', index=False)



# %%%% Cross results - '24-05-24'

# srcDir = 'D:/MagneticPincherData/Raw/24.02.27_Chameleon/Clean_Fluo_BF'
date = '24-05-24'
specif = '' #'_off4um'
specifFolder = 'M2-20um/B5mT/' # 'M2-20um/B5mT/' # 'M2-20um/Bstairs/' # 'M2-20um/B5mT/'
date2 = dateFormat(date)
cells = ['C1', 'C2', 'C4', 'C6', 'C7', 'C8', 'C13'] # 
Dmoy = 4.4995

list_df = []

for cell in cells:
    cellId = f'{date}_{cell}'
    srcDir = f'D:/MagneticPincherData/Raw/{date2}_Chameleon/{specifFolder}Clean_Fluo_BF'

    dstDir =  os.path.join(srcDir, 'Results_Fluo', f'{cell}')

    fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_Fluo.tif'
    fluoPath = os.path.join(srcDir, fluoName)

    OMEfilepath = os.path.join(srcDir, f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_OME.txt')
    CTZ_tz = OMEDataParser(OMEfilepath)
    [nC, nT, nZ] = CTZ_tz.shape[:3]
    
    # fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_Fluo.tif'
    bfName = f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_BF.tif'
    
    bfPath = os.path.join(srcDir, bfName)
    resBfDir = os.path.join(srcDir, 'Results_Tracker')
    resBfPath = os.path.join(resBfDir, bfName[:-4] + '_PY.csv')
    tsdf = pd.read_csv(resBfPath, sep=None, engine='python')
    
    fluoPath = os.path.join(srcDir, fluoName)
    resFluoDir =  os.path.join(srcDir, 'Results_Fluo', f'{cell}')
    cdf = pd.read_csv(os.path.join(resFluoDir, f'{cellId}_DfContact.csv'), sep=None, engine='python')
    print(cdf.columns)
    
    # czt_shape, czt_seq = get_CZT_fromTiff(fluoPath)
    # [nC, nZ, nT] = czt_shape
    
    resCrossDir = os.path.join(srcDir, 'Results_Cross')
    
    mdf = pd.merge(cdf, tsdf, left_index=True, right_index=True)
    mdf.to_csv(os.path.join(resCrossDir, f'{cellId}_DfMerged.csv'), sep=';', index=False)
    
    nT = len(mdf)
    cell_col = np.array([cell]*nT)
    # df = mdf[['D2', 'D3', 'F', 'I_viterbi_w5', 'I_viterbi_inout7-3', 'I_viterbi_flexW']]
    df = mdf[['D2', 'D3', 'F', 'Q_fbL', 'Q_fbN', 'Q_vb', 'W_vb', 'Q_gf', 'S_gf', 'nA_scan', 'nZ_scan']]
    df['cell'] = cell_col
    list_df.append(df)


global_df = pd.concat(list_df)
global_df = global_df.drop(global_df[(global_df['cell'] == 'C5') & (global_df['D2'] < 4.6)].index)

global_df['h2'] = global_df['D2']-Dmoy
global_df['h3'] = global_df['D3']-Dmoy

global_df.to_csv(os.path.join(resCrossDir, f'{date}_global_DfMerged.csv'), sep=';', index=False)


# %%%% Cross results - '24-05-24' stairs

# srcDir = 'D:/MagneticPincherData/Raw/24.02.27_Chameleon/Clean_Fluo_BF'
date = '24-05-24'
specif = '' #'_off4um'
specifFolder = 'M2-20um/Bstairs/' # 'M2-20um/B5mT/' # 'M2-20um/Bstairs/' # 'M2-20um/B5mT/'
date2 = dateFormat(date)
cells = ['C9', 'C10', 'C11', 'C12'] # 
Dmoy = 4.4995

list_df = []

for cell in cells:
    cellId = f'{date}_{cell}'
    srcDir = f'D:/MagneticPincherData/Raw/{date2}_Chameleon/{specifFolder}Clean_Fluo_BF'

    dstDir =  os.path.join(srcDir, 'Results_Fluo', f'{cell}')

    fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_Fluo.tif'
    fluoPath = os.path.join(srcDir, fluoName)

    OMEfilepath = os.path.join(srcDir, f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_OME.txt')
    CTZ_tz = OMEDataParser(OMEfilepath)
    [nC, nT, nZ] = CTZ_tz.shape[:3]
    
    # fluoName = f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_Fluo.tif'
    bfName = f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_BF.tif'
    
    bfPath = os.path.join(srcDir, bfName)
    resBfDir = os.path.join(srcDir, 'Results_Tracker')
    resBfPath = os.path.join(resBfDir, bfName[:-4] + '_PY.csv')
    tsdf = pd.read_csv(resBfPath, sep=None, engine='python')
    
    fluoPath = os.path.join(srcDir, fluoName)
    resFluoDir =  os.path.join(srcDir, 'Results_Fluo', f'{cell}')
    cdf = pd.read_csv(os.path.join(resFluoDir, f'{cellId}_DfContact.csv'), sep=None, engine='python')
    
    # czt_shape, czt_seq = get_CZT_fromTiff(fluoPath)
    # [nC, nZ, nT] = czt_shape
    
    resCrossDir = os.path.join(srcDir, 'Results_Cross')
    
    mdf = pd.merge(cdf, tsdf, left_index=True, right_index=True)
    mdf.to_csv(os.path.join(resCrossDir, f'{cellId}_DfMerged.csv'), sep=';', index=False)
    
    nT = len(mdf)
    cell_col = np.array([cell]*nT)
    # df = mdf[['D2', 'D3', 'F', 'I_viterbi_w5', 'I_viterbi_inout7-3', 'I_viterbi_flexW']]
    df = mdf[['D2', 'D3', 'F', 'Q_fb', 'Q_vb', 'Q_gf', 'nA_scan', 'nZ_scan']]
    df['cell'] = cell_col
    list_df.append(df)


global_df = pd.concat(list_df)
global_df = global_df.drop(global_df[(global_df['cell'] == 'C5') & (global_df['D2'] < 4.6)].index)

global_df['h2'] = global_df['D2']-Dmoy
global_df['h3'] = global_df['D3']-Dmoy

global_df.to_csv(os.path.join(resCrossDir, f'{date}_global_DfMerged.csv'), sep=';', index=False)



# %% Plots -- Manuscript

# %%% Dataset

#### Standard
global_df = pd.read_csv("D:/MagneticPincherData/Data_Analysis/FluoQuantifs/All_global_DfMerged.csv", sep=None, engine='python')
global_df['manipID'] = global_df['date'] + '_' + global_df['manip']
global_df['cellID'] = global_df['date'] + '_' + global_df['manip'] + '_P1_' + global_df['cell']
# global_df.to_csv("D:/MagneticPincherData/Data_Analysis/FluoQuantifs/All_global_DfMerged.csv", sep = ';', index = False)

figDir = 'D:/MagneticPincherData/Figures/FluoAnalysis'
figSubDir = '24-07-21'

global_df['D_fbL'] = global_df['Q_fbL']/global_df['h3']
global_df['D_fbN'] = global_df['Q_fbN']/global_df['h3']
global_df['D_vb'] = global_df['Q_vb']/global_df['h3']
global_df['D_gf'] = global_df['Q_gf']/global_df['h3']

excludedCells = ['24-05-24_M2_P1_C4', '24-05-24_M2_P1_C13',
                 '24-06-14_M1_P1_C3', '24-06-14_M1_P1_C5', 
                 '24-06-14_M1_P1_C10', '24-06-14_M1_P1_C12', 
                 '24-06-14_M1_P1_C17', ] # '24-06-14_M3_P1_C4'

Filters = [(global_df['cellID'].apply(lambda x : x not in excludedCells)),
           ]

global_df = filterDf(global_df, Filters)

#### Wtih compressions
Filters = [(global_df['date'] == '24-06-14'),
           (global_df['manip'].apply(lambda x : x in ['M2', 'M3'])),
           (global_df['cellID'].apply(lambda x : x not in ['24-06-14_M2_P1_C1'])),
           ]
fluo_df_f = filterDf(global_df, Filters)
fluo_df_f = fluo_df_f.dropna(subset = ['Q_fbL', 'Q_fbN', 'Q_vb', 'Q_gf'])
fluo_df_f['D1_fbN'] = fluo_df_f['Q_fbN']/fluo_df_f['h3']
fluo_df_f['D1_fbL'] = fluo_df_f['Q_fbL']/fluo_df_f['h3']
fluo_df_f['D1_vb'] = fluo_df_f['Q_vb']/fluo_df_f['h3']
fluo_df_f['D1_gf'] = fluo_df_f['Q_gf']/fluo_df_f['h3']

fluo_df_fg = dataGroup(fluo_df_f, groupCol = 'cellID', idCols = ['date', 'manip', 'cell'], 
                       numCols = ['Q_fbL', 'Q_fbN', 'Q_vb', 'Q_gf', 'h3', 'D1_fbN', 'D1_fbL', 'D1_vb', 'D1_gf'],
                       aggFun = 'median')

meca_df = pd.read_csv("D:/MagneticPincherData/Data_Analysis/FluoQuantifs/MecaData_Chameleon_CompFluo.csv", sep=None, engine='python')
meca_df['cellID_2'] = meca_df['cellID'].apply(lambda x : '-'.join(x.split('-')[:-1]))
Filters = [(meca_df['date'] == '24-06-14'),
           (meca_df['surroundingThickness'] <= 900),
           ]

meca_df_f = filterDf(meca_df, Filters)

merged_df = fluo_df_fg.merge(meca_df_f, left_on = 'cellID', right_on = 'cellID_2', how = 'inner')

merged_df['D2_fbN'] = 1000*merged_df['Q_fbN']/merged_df['surroundingThickness']
merged_df['D2_fbL'] = 1000*merged_df['Q_fbL']/merged_df['surroundingThickness']
merged_df['D2_vb'] = 1000*merged_df['Q_vb']/merged_df['surroundingThickness']
merged_df['D2_gf'] = 1000*merged_df['Q_gf']/merged_df['surroundingThickness']

figDir = 'D:/MagneticPincherData/Figures/FluoAnalysis'
figSubDir = '24-07-23'

# %%% Plot quantity - per comp

gs.set_manuscript_options_jv()

Set2 = matplotlib.colormaps['Set2'].colors
cL = [Set2[0], Set2[1], Set2[4], Set2[5]]
cMap = matplotlib.colors.ListedColormap(cL, name='from_list')

fig, axes = plt.subplots(2, 1, figsize = (17/gs.cm_in, 17/gs.cm_in))#, layout="constrained")

# style = 'date'
# hue = 'cell'

hue = 'manipID'
style = 'cell'
metric = 'Q_gf'

# Filter global_df

Filters = [(global_df['h3'] < 1.1),
           (global_df['Q_gf'] < 40),
           (global_df['date'] != '24-02-27'),
           # (global_df['manipID'] == '24-06-14_M1'),
           ]

df_f = filterDf(global_df, Filters)
df_f = df_f.dropna(subset=metric)
df_f['h3'] *= 1000

dates = df_f['date'].unique()
manipes = df_f['manipID'].unique()
md = {manipes[i]:i for i in range(len(manipes))}
df_f['manipNum'] = df_f['manipID'].apply(lambda x : md[x])

bins = np.linspace(0, 1000, 10, endpoint=False)
df_f['h3_bin'] = np.digitize(df_f['h3'].values, bins = bins)
df_fg = df_f[[metric,'h3','h3_bin']].groupby('h3_bin').agg(['median', 'std'])
df_fg['h3_upper'] = df_fg.index*50
df_fg = df_fg.dropna()

#### Plot 1 - Lin scale

ax = axes[0]

X = df_f['h3'].values
Y = df_f[metric].values
C = df_f['manipNum'].values

ax.set_prop_cycle(color=cL)
ax.scatter(X, Y, c=C, marker='o', s=6, zorder=3, cmap = cMap) # , style='cellNum'

Xg = df_fg['h3', 'median'].values
Yg = df_fg[metric, 'median'].values
# Xerr = df_fg['h3', 'median'].values
Ygerr = df_fg[metric, 'std'].values

ax.errorbar(Xg, Yg, Ygerr, color = 'dimgray', zorder=5,
            lw = 1.5, ls = '-',
            marker = 'o', markersize = 6, markerfacecolor = 'w', markeredgecolor = 'dimgray', markeredgewidth = 1.5,
            elinewidth = 1, ecolor = 'dimgray', capsize = 3, capthick = 1,
            label = 'Binning & median')

# Xfit, Yfit = df_f['h3'].values, df_f[metric].values
# [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
# R2 = w_results.rsquared
# pval = results.pvalues[1]
# Xplot = np.linspace(min(Xfit), max(Xfit), 50)
# Yplot = b + a*Xplot

# ax.plot(Xplot, Yplot, ls = '--', c = 'k', zorder=4, lw = 1.0,
#         label =  r'$\bf{Fit\ y\ =\ ax+b}$' + f'\n$a$ = {a:.1e}' + f'\n$b$  = {b:.2f}' + \
#                 f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
    
# xfit = df_f['h3'].values
# yfit = df_f[metric].values
# params, res = ufun.fitLineHuber(xfit, yfit)
# print(params)
# xfitplot = np.linspace(np.min(df_f['h3'].values), np.max(df_f['h3'].values), 100)
# ax.plot(xfitplot, params[0] + xfitplot*params[1], 'k--', zorder = 4,
#         label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')

# LegendMark = mlines.Line2D([], [], color='k', ls='--', marker='', 
#                             # markersize=0, markeredgecolor='w', markeredgewidth=0,
#                             label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')
# ax.legend(handles=[LegendMark])

ax.legend(loc = 'upper left')

ax.set_xlim([0, 1100])
ax.set_ylim([0, 42])

#### Plot 2 - Log scale


ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')

X = df_f['h3'].values
Y = df_f[metric].values
C = df_f['manipNum'].values

ax.set_prop_cycle(color=cL)
ax.scatter(X, Y, c=C, marker='o', s=6, zorder=3, cmap = cMap) # , style='cellNum'

Xg = df_fg['h3', 'median'].values
Yg = df_fg[metric, 'median'].values
# Xerr = df_fg['h3', 'median'].values
Ygerr = df_fg[metric, 'std'].values

ax.errorbar(Xg, Yg, Ygerr, color = 'dimgray', zorder=5,
            lw = 1.5, ls = '-',
            marker = 'o', markersize = 6, markerfacecolor = 'w', markeredgecolor = 'dimgray', markeredgewidth = 1.5,
            elinewidth = 1, ecolor = 'dimgray', capsize = 3, capthick = 1,
            label = 'Binning & median')

n = np.log10(200)
m = np.log10(10)

expo = +1
A = (10**(m-n*expo))
# A = 0.1
Xplot = np.logspace(2, 3.1, 50)
Yplot = A*(Xplot**expo)
ax.plot(Xplot, Yplot, ls = '-.', c = 'gray', lw = 1.0,
        label =  r'$\bf{Line\ y\ =\ Ax}$')


# Xfit, Yfit = df_f['h3'].values, df_f[metric].values
# [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
# R2 = w_results.rsquared
# pval = results.pvalues[1]
# Xplot = np.linspace(min(Xfit), max(Xfit), 50)
# Yplot = b + a*Xplot

# ax.plot(Xplot, Yplot, ls = '--', c = 'k', lw = 2.0,
#         label =  r'$\bf{Fit\ y\ =\ ax+b}$' + f'\n$a$ = {a:.1e}' + f'\n$b$  = {b:.2f}' + \
#                 f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')

# sns.scatterplot(ax=ax, data=df_f, x='h3', y=metric, hue=hue, marker='o', s= 25, zorder=5) # , style='cellNum'

# xfit = df_f['h3'].values
# yfit = df_f[metric].values
# params, res = ufun.fitLineHuber(xfit, yfit)
# xfitplot = np.linspace(np.min(df_f['h3'].values), np.max(df_f['h3'].values), 100)
# ax.plot(xfitplot, params[0] + xfitplot*params[1], 'k--', label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')

# LegendMark = mlines.Line2D([], [], color='k', ls='--', marker='', 
#                             # markersize=0, markeredgecolor='w', markeredgewidth=0,
#                             label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')
# ax.legend(handles=[LegendMark])

ax.set_xlim([100, 1300])
ax.legend(loc = 'upper left')


for ax in axes:
    ax.set_ylabel('Actin Quantity (a.u.)')
    ax.set_xlabel('$H_{5mT}$ (nm)')
    ax.grid(which = 'both', alpha = 0.4)
    
# ax.legend().set_visible(False)


#### Save
figSubDir = 'Manuscript_E-h_fluo'
name = 'Qactin_GF_vs_h5mT'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



#### Plot 3 - log only

gs.set_defense_options_jv(palette = 'Set2')

fig, ax = plt.subplots(1, 1, figsize = (12/gs.cm_in, 12/gs.cm_in))#, layout="constrained")

ax.set_xscale('log')
ax.set_yscale('log')

X = df_f['h3'].values
Y = df_f[metric].values
C = df_f['manipNum'].values

ax.set_prop_cycle(color=cL)
ax.scatter(X, Y, c=C, marker='o', s=6, zorder=3, cmap = cMap) # , style='cellNum'

Xg = df_fg['h3', 'median'].values
Yg = df_fg[metric, 'median'].values
# Xerr = df_fg['h3', 'median'].values
Ygerr = df_fg[metric, 'std'].values

ax.errorbar(Xg, Yg, Ygerr, color = 'dimgray', zorder=5,
            lw = 1.5, ls = '-',
            marker = 'o', markersize = 6, markerfacecolor = 'w', markeredgecolor = 'dimgray', markeredgewidth = 1.5,
            elinewidth = 1, ecolor = 'dimgray', capsize = 3, capthick = 1,
            label = 'Binning & median')

n = np.log10(200)
m = np.log10(10)

expo = +1
A = (10**(m-n*expo))
# A = 0.1
Xplot = np.logspace(2, 3.1, 50)
Yplot = A*(Xplot**expo)
ax.plot(Xplot, Yplot, ls = '-.', c = 'gray', lw = 1.0,
        label =  r'$\bf{Line\ y\ =\ Ax}$')

ax.set_xlim([100, 1100])
ax.legend(loc = 'upper left')

ax.set_ylabel('Actin Quantity (a.u.)')
ax.set_xlabel('$H_{5mT}$ (nm)')
ax.grid(which = 'both', alpha = 0.4)


def makeCountDf_Fluo(df):
    cols_count_df = ['h3', 'cellID', 'manipID', 'date']
    count_df = df[cols_count_df]
    groupByCell = count_df.groupby('cellID')
    d_agg = {'h3':'count', 'date':'first', 'manipID':'first'}
    df_CountByCell = groupByCell.agg(d_agg).rename(columns={'h3':'pointCount'})    

    groupByCond = df_CountByCell.reset_index().groupby(np.ones(len(df_CountByCell)))
    d_agg = {'cellID': 'count', 'pointCount': 'sum', 
             'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
    d_rename = {'cellID':'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
    df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
    
    return(df_CountByCond, df_CountByCell)

CountByCond, CountByCell = makeCountDf_Fluo(df_f)
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

plt.tight_layout()
plt.show()


#### Save
figSubDir = 'Manuscript_E-h_fluo'
name = 'Qactin_GF_vs_h5mT_DEFENSE'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%% Plot quantity - One cell !

# gs.set_manuscript_options_jv()
gs.set_defense_options_jv()

Set2 = matplotlib.colormaps['Set2'].colors
cL = [Set2[0], Set2[1], Set2[4], Set2[5]]
cMap = matplotlib.colors.ListedColormap(cL, name='from_list')

# fig, ax = plt.subplots(1, 1, figsize = (6/gs.cm_in, 8/gs.cm_in))#, layout="constrained")
fig, ax = plt.subplots(1, 1, figsize = (7.5/gs.cm_in, 7.5/gs.cm_in))#, layout="constrained")

# style = 'date'
# hue = 'cell'

hue = 'manipID'
style = 'cell'
metric = 'Q_gf'


# Filter global_df

Filters = [
           #  (global_df['h3'] < 1.1),
           # (global_df['Q_gf'] < 40),
           # (global_df['date'] == '24-06-14'),
            (global_df['cellID'] == '24-06-14_M1_P1_C2'),
           ]

df_f = filterDf(global_df, Filters)
df_f = df_f.dropna(subset=metric)
df_f['h3'] *= 1000

dates = df_f['date'].unique()
manipes = df_f['manipID'].unique()
md = {manipes[i]:i for i in range(len(manipes))}
df_f['manipNum'] = df_f['manipID'].apply(lambda x : md[x])

bins = np.linspace(0, 1000, 10, endpoint=False)
df_f['h3_bin'] = np.digitize(df_f['h3'].values, bins = bins)
df_fg = df_f[[metric,'h3','h3_bin']].groupby('h3_bin').agg(['median', 'std'])
df_fg['h3_upper'] = df_fg.index*50
df_fg = df_fg.dropna()

#### Plot 1 - Lin scale

ax = ax

X = df_f['h3'].values
Y = df_f[metric].values
C = df_f['manipNum'].values

ax.set_prop_cycle(color=cL)
ax.scatter(X, Y, c=C, marker='o', s=40, zorder=3, cmap = cMap, ec='w', linewidth=0.5, alpha=0.75) # , style='cellNum'


# Xfit, Yfit = df_f['h3'].values, df_f[metric].values
# [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
# R2 = w_results.rsquared
# pval = results.pvalues[1]
# Xplot = np.linspace(min(Xfit), max(Xfit), 50)
# Yplot = b + a*Xplot

# ax.plot(Xplot, Yplot, ls = '--', c = 'k', zorder=4, lw = 1.0,
#         label =  r'$\bf{Fit\ y\ =\ ax+b}$' + f'\n$a$ = {a:.1e}' + f'\n$b$  = {b:.2f}' + \
#                 f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
    
# xfit = df_f['h3'].values
# yfit = df_f[metric].values
# params, res = ufun.fitLineHuber(xfit, yfit)
# print(params)
# xfitplot = np.linspace(np.min(df_f['h3'].values), np.max(df_f['h3'].values), 100)
# ax.plot(xfitplot, params[0] + xfitplot*params[1], 'k--', zorder = 4,
#         label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')

# LegendMark = mlines.Line2D([], [], color='k', ls='--', marker='', 
#                             # markersize=0, markeredgecolor='w', markeredgewidth=0,
#                             label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')
# ax.legend(handles=[LegendMark])

# ax.legend(loc = 'upper left')

ax.set_xlim([0, 550])
ax.set_ylim([0, 15])


ax.set_ylabel('Actin Quantity (a.u.)')
ax.set_xlabel('$H_{5mT}$ (nm)')
ax.grid(which = 'both', alpha = 0.4)
    
# ax.legend().set_visible(False)

plt.tight_layout()
plt.show()


#### Save
figSubDir = 'E-h-Fluo'
name = 'ONE_CELL_Qactin_GF_vs_h5mT_V2'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 300,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%% Plot density - per comp

gs.set_manuscript_options_jv()

Set2 = matplotlib.colormaps['Set2'].colors
cL = [Set2[0], Set2[1], Set2[4], Set2[5]]
cMap = matplotlib.colors.ListedColormap(cL, name='from_list')

fig, axes = plt.subplots(2, 1, figsize = (17/gs.cm_in, 17/gs.cm_in))#, layout="constrained")

# style = 'date'
# hue = 'cell'

hue = 'manipID'
style = 'cell'
metric = 'D_gf'

# Filter global_df

Filters = [(global_df['h3'] < 1.1),
           (global_df['Q_gf'] < 40),
           (global_df['date'] != '24-02-27'),
           # (global_df['manipID'] == '24-06-14_M1'),
           ]

df_f = filterDf(global_df, Filters)
df_f = df_f.dropna(subset=metric)
df_f['h3'] *= 1000

dates = df_f['date'].unique()
manipes = df_f['manipID'].unique()
md = {manipes[i]:i for i in range(len(manipes))}
df_f['manipNum'] = df_f['manipID'].apply(lambda x : md[x])

bins = np.linspace(0, 1000, 10, endpoint=False)
df_f['h3_bin'] = np.digitize(df_f['h3'].values, bins = bins)
df_fg = df_f[[metric,'h3','h3_bin']].groupby('h3_bin').agg(['median', 'std'])
df_fg['h3_upper'] = df_fg.index*50
df_fg = df_fg.dropna()

#### Plot 1 - Lin scale

ax = axes[0]

X = df_f['h3'].values
Y = df_f[metric].values
C = df_f['manipNum'].values

ax.set_prop_cycle(color=cL)
ax.scatter(X, Y, c=C, marker='o', s=6, zorder=3, cmap = cMap) # , style='cellNum'

Xg = df_fg['h3', 'median'].values
Yg = df_fg[metric, 'median'].values
# Xerr = df_fg['h3', 'median'].values
Ygerr = df_fg[metric, 'std'].values

ax.errorbar(Xg, Yg, Ygerr, color = 'dimgray', zorder=5,
            lw = 1.5, ls = '-',
            marker = 'o', markersize = 6, markerfacecolor = 'w', markeredgecolor = 'dimgray', markeredgewidth = 1.5,
            elinewidth = 1, ecolor = 'dimgray', capsize = 3, capthick = 1,
            label = 'Binning & median')

# Xfit, Yfit = df_f['h3'].values, df_f[metric].values
# [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
# R2 = w_results.rsquared
# pval = results.pvalues[1]
# Xplot = np.linspace(min(Xfit), max(Xfit), 50)
# Yplot = b + a*Xplot

# ax.plot(Xplot, Yplot, ls = '--', c = 'k', zorder=4, lw = 1.0,
#         label =  r'$\bf{Fit\ y\ =\ ax+b}$' + f'\n$a$ = {a:.1e}' + f'\n$b$  = {b:.2f}' + \
#                 f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
    
# xfit = df_f['h3'].values
# yfit = df_f[metric].values
# params, res = ufun.fitLineHuber(xfit, yfit)
# print(params)
# xfitplot = np.linspace(np.min(df_f['h3'].values), np.max(df_f['h3'].values), 100)
# ax.plot(xfitplot, params[0] + xfitplot*params[1], 'k--', zorder = 4,
#         label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')

# LegendMark = mlines.Line2D([], [], color='k', ls='--', marker='', 
#                             # markersize=0, markeredgecolor='w', markeredgewidth=0,
#                             label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')
# ax.legend(handles=[LegendMark])

ax.legend(loc = 'lower left')

ax.set_xlim([0, 1100])
ax.set_ylim([0, 62])



#### Plot 2 - Log scale

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')

X = df_f['h3'].values
Y = df_f[metric].values
C = df_f['manipNum'].values

ax.set_prop_cycle(color=cL)
ax.scatter(X, Y, c=C, marker='o', s=6, zorder=3, cmap = cMap) # , style='cellNum'

Xg = df_fg['h3', 'median'].values
Yg = df_fg[metric, 'median'].values
# Xerr = df_fg['h3', 'median'].values
Ygerr = df_fg[metric, 'std'].values

ax.errorbar(Xg, Yg, Ygerr, color = 'dimgray', zorder=5,
            lw = 1.5, ls = '-',
            marker = 'o', markersize = 6, markerfacecolor = 'w', markeredgecolor = 'dimgray', markeredgewidth = 1.5,
            elinewidth = 1, ecolor = 'dimgray', capsize = 3, capthick = 1,
            label = 'Binning & median')


Xfit, Yfit = np.log(X), np.log(Y)
[b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
A, k = np.exp(b), a
R2 = w_results.rsquared
pval = results.pvalues[1]
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k

ax.plot(Xplot, Yplot, ls = '--', c = 'k', lw = 1.0,
        label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')

# n = np.log10(100)
# m = np.log10(80)

# expo = -0.5
# A = (10**(m-n*expo))
# # A = 0.1
# Xplot = np.logspace(2, 3.1, 50)
# Yplot = A*(Xplot**expo)
# ax.plot(Xplot, Yplot, ls = ':', c = 'gray', lw = 1.0,
#         label =  r'$\bf{Line\ y\ =\ Ax^{-0.5}}$')

# expo = -1
# A = (10**(m-n*expo))
# # A = 0.1
# Xplot = np.logspace(2, 3.1, 50)
# Yplot = A*(Xplot**expo)
# ax.plot(Xplot, Yplot, ls = '-.', c = 'gray', lw = 1.0,
#         label =  r'$\bf{Line\ y\ =\ Ax^{-1}}$')



# expo = -2
# A = (10**(m-n*expo))
# # A = 0.1
# Xplot = np.logspace(2, 3.1, 50)
# Yplot = A*(Xplot**expo)
# ax.plot(Xplot, Yplot, ls = '--', c = 'gray', lw = 1.0,
#         label =  r'$\bf{Line\ y\ =\ Ax^{-2}}$')

ax.set_xlim([100, 1100])
ax.legend(loc = 'lower left')

# sns.scatterplot(ax=ax, data=df_f, x='h3', y=metric, hue=hue, marker='o', s= 25, zorder=5) # , style='cellNum'

# xfit = df_f['h3'].values
# yfit = df_f[metric].values
# params, res = ufun.fitLineHuber(xfit, yfit)
# xfitplot = np.linspace(np.min(df_f['h3'].values), np.max(df_f['h3'].values), 100)
# ax.plot(xfitplot, params[0] + xfitplot*params[1], 'k--', label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')

# LegendMark = mlines.Line2D([], [], color='k', ls='--', marker='', 
#                             # markersize=0, markeredgecolor='w', markeredgewidth=0,
#                             label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')
# ax.legend(handles=[LegendMark])

# ax.set_xlim([190, 1200])
# ax.set_ylim([5, 150])


# ax.set_ylabel('Actin Quantity')
# ax.set_xlabel('Cortex thickness (nm)')
# ax.legend().set_visible(False)
    
    
for ax in axes:
    ax.set_ylabel('Actin Density (a.u.)')
    ax.set_xlabel('$H_{5mT}$ (nm)')
    ax.grid(which = 'both', alpha = 0.4)
    
# ax.legend().set_visible(False)


#### Save
figSubDir = 'Manuscript_E-h_fluo'
name = 'Dactin_GF_vs_h5mT'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')




def makeCountDf_Fluo(df):
    cols_count_df = ['h3', 'cellID', 'manipID', 'date']
    count_df = df[cols_count_df]
    groupByCell = count_df.groupby('cellID')
    d_agg = {'h3':'count', 'date':'first', 'manipID':'first'}
    df_CountByCell = groupByCell.agg(d_agg).rename(columns={'h3':'pointCount'})    

    groupByCond = df_CountByCell.reset_index().groupby(np.ones(len(df_CountByCell)))
    d_agg = {'cellID': 'count', 'pointCount': 'sum', 
             'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
    d_rename = {'cellID':'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
    df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
    
    return(df_CountByCond, df_CountByCell)

CountByCond, CountByCell = makeCountDf_Fluo(df_f)
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')



#### Plot 3 - log only

gs.set_defense_options_jv(palette = 'Set2')

fig, ax = plt.subplots(1, 1, figsize = (12/gs.cm_in, 12/gs.cm_in))#, layout="constrained")


ax = ax
ax.set_xscale('log')
ax.set_yscale('log')

X = df_f['h3'].values
Y = df_f[metric].values
C = df_f['manipNum'].values

ax.set_prop_cycle(color=cL)
ax.scatter(X, Y, c=C, marker='o', s=6, zorder=3, cmap = cMap) # , style='cellNum'

Xg = df_fg['h3', 'median'].values
Yg = df_fg[metric, 'median'].values
# Xerr = df_fg['h3', 'median'].values
Ygerr = df_fg[metric, 'std'].values

ax.errorbar(Xg, Yg, Ygerr, color = 'dimgray', zorder=5,
            lw = 1.5, ls = '-',
            marker = 'o', markersize = 6, markerfacecolor = 'w', markeredgecolor = 'dimgray', markeredgewidth = 1.5,
            elinewidth = 1, ecolor = 'dimgray', capsize = 3, capthick = 1,
            label = 'Binning & median')


Xfit, Yfit = np.log(X), np.log(Y)
[b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
A, k = np.exp(b), a
R2 = w_results.rsquared
pval = results.pvalues[1]
Xplot = np.exp(np.linspace(4, 8, 50))
Yplot = A * Xplot**k

ax.plot(Xplot, Yplot, ls = '-.', c = 'k', lw = 1.25,
        label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}', zorder=8)

# n = np.log10(100)
# m = np.log10(80)

# expo = -0.5
# A = (10**(m-n*expo))
# # A = 0.1
# Xplot = np.logspace(2, 3.1, 50)
# Yplot = A*(Xplot**expo)
# ax.plot(Xplot, Yplot, ls = ':', c = 'gray', lw = 1.0,
#         label =  r'$\bf{Line\ y\ =\ Ax^{-0.5}}$')

# expo = -1
# A = (10**(m-n*expo))
# # A = 0.1
# Xplot = np.logspace(2, 3.1, 50)
# Yplot = A*(Xplot**expo)
# ax.plot(Xplot, Yplot, ls = '-.', c = 'gray', lw = 1.0,
#         label =  r'$\bf{Line\ y\ =\ Ax^{-1}}$')



# expo = -2
# A = (10**(m-n*expo))
# # A = 0.1
# Xplot = np.logspace(2, 3.1, 50)
# Yplot = A*(Xplot**expo)
# ax.plot(Xplot, Yplot, ls = '--', c = 'gray', lw = 1.0,
#         label =  r'$\bf{Line\ y\ =\ Ax^{-2}}$')

ax.set_xlim([100, 1300])
ax.set_ylim([7, 110])
ax.legend(loc = 'lower left', fontsize=8)

# sns.scatterplot(ax=ax, data=df_f, x='h3', y=metric, hue=hue, marker='o', s= 25, zorder=5) # , style='cellNum'

# xfit = df_f['h3'].values
# yfit = df_f[metric].values
# params, res = ufun.fitLineHuber(xfit, yfit)
# xfitplot = np.linspace(np.min(df_f['h3'].values), np.max(df_f['h3'].values), 100)
# ax.plot(xfitplot, params[0] + xfitplot*params[1], 'k--', label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')

# LegendMark = mlines.Line2D([], [], color='k', ls='--', marker='', 
#                             # markersize=0, markeredgecolor='w', markeredgewidth=0,
#                             label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')
# ax.legend(handles=[LegendMark])

# ax.set_xlim([190, 1200])
# ax.set_ylim([5, 150])


# ax.set_ylabel('Actin Quantity')
# ax.set_xlabel('Cortex thickness (nm)')
# ax.legend().set_visible(False)
    
    
ax.set_ylabel('Actin Density (a.u.)')
ax.set_xlabel('$H_{5mT}$ (nm)')
ax.grid(which = 'both', alpha = 0.4)

# ax.legend().set_visible(False)


#### Save
figSubDir = 'Manuscript_E-h_fluo'
name = 'Dactin_GF_vs_h5mT_DEFENSE_V2'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


plt.tight_layout()
plt.show()

# %%% Plots crossing Stiff & Density

# %%%% 1.

fig, axes = plt.subplots(2, 1, figsize=(17/gs.cm_in, 17/gs.cm_in))
# gs.set_manuscript_options_jv(palette = 'Set2')

# Filter global_df

df = merged_df.copy()
df = df.rename(columns={'surroundingThickness':'H 5mT'})

x = 'H 5mT'
y = 'D2_gf'
style = None
s = 60
alpha = 1
zo = 5
ec = 'None'

hue = 'E_f_<_400'
df[hue] /= 1000

ax = axes[0]

sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_xlabel('$H_{5mT}$ (nm)')
ax.set_ylabel('Cortex density (a.u.)')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()



ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_xlabel('$H_{5mT}$ (nm)')
ax.set_ylabel('Cortex density (a.u.)')
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

Xfit, Yfit = np.log(df[x].values), np.log(df[y].values)
[b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results=True)
A, k = np.exp(b), a
R2 = w_results.rsquared
pval = results.pvalues[1]
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
ax.plot(Xplot, Yplot, ls = '--', c = 'k', lw = 1.5,
        label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
ax.legend()

for ax in axes:
    ax.grid(which = 'both', alpha = 0.4)
    


# fig.suptitle('density v thickness 2 - with cell w. avg.')
fig.tight_layout()
plt.show()


#### Save
figSubDir = 'Manuscript_E-h_fluo'
name = 'Comp_Dactin_GF_vs_h5mT'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



def makeCountDf_Fluo(df):
    cols_count_df = ['h3', 'cellID', 'manipID', 'date']
    count_df = df[cols_count_df]
    groupByCell = count_df.groupby('cellID')
    d_agg = {'h3':'count', 'date':'first', 'manipID':'first'}
    df_CountByCell = groupByCell.agg(d_agg).rename(columns={'h3':'pointCount'})    

    groupByCond = df_CountByCell.reset_index().groupby(np.ones(len(df_CountByCell)))
    d_agg = {'cellID': 'count', 'pointCount': 'sum', 
             'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
    d_rename = {'cellID':'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
    df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
    
    return(df_CountByCond, df_CountByCell)

CountByCond, CountByCell = makeCountDf_Fluo(df_f)
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')



# %%%% 2.

fig, axes = plt.subplots(2, 1, figsize=(17/gs.cm_in, 17/gs.cm_in))
# gs.set_manuscript_options_jv(palette = 'Set2')

# Filter global_df

df = merged_df.copy()
df = df.rename(columns={'surroundingThickness':'H 5mT'})
hue = 'H 5mT'
# x = 'D2_fbL'
x = 'D2_gf'
y = 'E_f_<_400'
style = None
s = 60
alpha = 1
zo = 5
ec = 'None'

df[y] /= 1000
# df = df[df[x] < 40]


ax = axes[0]
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = True) # , style='cellNum'
ax.set_ylabel('$E_{400}$ (kPa)')
ax.set_xlabel('Cortex density (au)')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()



ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('$E_{400}$ (kPa)')
ax.set_xlabel('Cortex density (au)')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

Xfit, Yfit = np.log(df[x].values), np.log(df[y].values)
[b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results=True)
A, k = np.exp(b), a
R2 = w_results.rsquared
pval = results.pvalues[1]
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
ax.plot(Xplot, Yplot, ls = '--', c = 'k', lw = 1.5,
        label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
ax.legend()


for ax in axes:
    ax.grid(which = 'both', alpha = 0.4)

# fig.suptitle('Stiffness v density 2 - with cell w. avg.')
fig.tight_layout()
plt.show()

#### Save
figSubDir = 'Manuscript_E-h_fluo'
name = 'Comp_E_vs_Dactin_GF'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



def makeCountDf_Fluo(df):
    cols_count_df = ['h3', 'cellID', 'manipID', 'date']
    count_df = df[cols_count_df]
    groupByCell = count_df.groupby('cellID')
    d_agg = {'h3':'count', 'date':'first', 'manipID':'first'}
    df_CountByCell = groupByCell.agg(d_agg).rename(columns={'h3':'pointCount'})    

    groupByCond = df_CountByCell.reset_index().groupby(np.ones(len(df_CountByCell)))
    d_agg = {'cellID': 'count', 'pointCount': 'sum', 
             'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
    d_rename = {'cellID':'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
    df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
    
    return(df_CountByCond, df_CountByCell)

CountByCond, CountByCell = makeCountDf_Fluo(df_f)
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%%% 3.

fig, axes = plt.subplots(2, 1, figsize=(17/gs.cm_in, 17/gs.cm_in))
# gs.set_manuscript_options_jv(palette = 'Set2')

# Filter global_df

df = merged_df.copy()
df = df.rename(columns={'surroundingThickness':'H 5mT'})

hue = 'D2_gf'
x = 'H 5mT'
y = 'E_f_<_400'
style = None
s = 60
alpha = 1
zo = 5
ec = 'None'

df[y] /= 1000


ax = axes[0]
# ax.plot([], [], label=r'$\bf{Density\ (a.u.)}$', ls='-', color='w')
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = True) # , style='cellNum'
# ax.plot([], [], label=r' ', ls='-', color='w')
ax.set_ylabel('$E_{400}$ (kPa)')
ax.set_xlabel('$H_{5mT}$ (nm)')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 8, loc='upper right', 
          title = r'$\bf{Density\ (a.u.)}$', title_fontsize = 9, ncol = 2)
ax.grid()



ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('$E_{400}$ (kPa)')
ax.set_xlabel('$H_{5mT}$ (nm)')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

Xfit, Yfit = np.log(df[x].values), np.log(df[y].values)
[b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results=True)
A, k = np.exp(b), a
R2 = w_results.rsquared
pval = results.pvalues[1]
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
ax.plot(Xplot, Yplot, ls = '--', c = 'k', lw = 1.5,
        label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
ax.legend(fontsize = 8, loc='upper right', )


for ax in axes:
    ax.grid(which = 'both', alpha = 0.4)


# fig.suptitle('Stiffness v density 2 - with cell w. avg.')
fig.tight_layout()
plt.show()

#### Save
figSubDir = 'Manuscript_E-h_fluo'
name = 'Comp_E_vs_h5mT'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



def makeCountDf_Fluo(df):
    cols_count_df = ['h3', 'cellID', 'manipID', 'date']
    count_df = df[cols_count_df]
    groupByCell = count_df.groupby('cellID')
    d_agg = {'h3':'count', 'date':'first', 'manipID':'first'}
    df_CountByCell = groupByCell.agg(d_agg).rename(columns={'h3':'pointCount'})    

    groupByCond = df_CountByCell.reset_index().groupby(np.ones(len(df_CountByCell)))
    d_agg = {'cellID': 'count', 'pointCount': 'sum', 
             'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
    d_rename = {'cellID':'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
    df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
    
    return(df_CountByCond, df_CountByCell)

CountByCond, CountByCell = makeCountDf_Fluo(df_f)
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%%% 3. V2


fig, ax = plt.subplots(1, 1, figsize=(12/gs.cm_in, 7/gs.cm_in))
# gs.set_manuscript_options_jv(palette = 'Set2')

# Filter global_df

df = merged_df.copy()
df = df.rename(columns={'surroundingThickness':'H 5mT'})

hue = 'D2_gf'
x = 'H 5mT'
y = 'E_f_<_400'
style = None
s = 40
alpha = 1
zo = 5
ec = 'None'

df[y] /= 1000


# ax = axes[0]
# # ax.plot([], [], label=r'$\bf{Density\ (a.u.)}$', ls='-', color='w')
# sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
#                 ec = ec, alpha = alpha, zorder=zo, legend = True) # , style='cellNum'
# # ax.plot([], [], label=r' ', ls='-', color='w')
# ax.set_ylabel('$E_{400}$ (kPa)')
# ax.set_xlabel('$H_{5mT}$ (nm)')
# ax.set_ylim([0, ax.get_ylim()[-1]])
# ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='upper right', 
#           title = r'$\bf{Density\ (a.u.)}$', title_fontsize = 9, ncol = 2)
# ax.grid()



ax = ax
ax.set_xscale('log')
ax.set_yscale('log')
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = True) # , style='cellNum'
ax.set_ylabel('$E_{400}$ (kPa)')
ax.set_xlabel('$H_{5mT}$ (nm)')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([90, 1100])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

Xfit, Yfit = np.log(df[x].values), np.log(df[y].values)
[b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results=True)
A, k = np.exp(b), a
R2 = w_results.rsquared
pval = results.pvalues[1]
Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
Yplot = A * Xplot**k
# ax.plot(Xplot, Yplot, ls = '--', c = 'k', lw = 1.5,)
        # label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
        #         f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
# ax.legend(fontsize = 8, loc='upper right', )
ax.legend(fontsize = 8, loc='upper left', 
          title = r'$\bf{Density\ (a.u.)}$', title_fontsize = 9, ncol = 2)


ax.grid(which = 'both', alpha = 0.4)


# fig.suptitle('Stiffness v density 2 - with cell w. avg.')
fig.tight_layout()
plt.show()

#### Save
figSubDir = 'Manuscript_E-h_fluo'
name = 'Comp_E_vs_h5mT'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = '', cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 200,
                figDir = figDir, figSubDir = '', cloudSave = 'flexible')



def makeCountDf_Fluo(df):
    cols_count_df = ['h3', 'cellID', 'manipID', 'date']
    count_df = df[cols_count_df]
    groupByCell = count_df.groupby('cellID')
    d_agg = {'h3':'count', 'date':'first', 'manipID':'first'}
    df_CountByCell = groupByCell.agg(d_agg).rename(columns={'h3':'pointCount'})    

    groupByCond = df_CountByCell.reset_index().groupby(np.ones(len(df_CountByCell)))
    d_agg = {'cellID': 'count', 'pointCount': 'sum', 
              'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
    d_rename = {'cellID':'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
    df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
    
    return(df_CountByCond, df_CountByCell)

df = df.rename(columns={'date_x':'date'})


CountByCond, CountByCell = makeCountDf_Fluo(df)
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t') 

# %% Plots -- Test Manuscript

# %%% Plot -- standard

global_df = pd.read_csv("D:/MagneticPincherData/Data_Analysis/FluoQuantifs/All_global_DfMerged.csv", sep=None, engine='python')
global_df['manipID'] = global_df['date'] + '_' + global_df['manip']
global_df['cellID'] = global_df['date'] + '_' + global_df['manip'] + '_P1_' + global_df['cell']
# global_df.to_csv("D:/MagneticPincherData/Data_Analysis/FluoQuantifs/All_global_DfMerged.csv", sep = ';', index = False)

figDir = 'D:/MagneticPincherData/Figures/FluoAnalysis'
figSubDir = '24-07-19'

global_df['D_fbL'] = global_df['Q_fbL']/global_df['h3']
global_df['D_fbN'] = global_df['Q_fbN']/global_df['h3']
global_df['D_vb'] = global_df['Q_vb']/global_df['h3']
global_df['D_gf'] = global_df['Q_gf']/global_df['h3']


# %%% Plot quantity - per cell

gs.set_manuscript_options_jv()

Set2 = matplotlib.colormaps['Set2'].colors
cL = [Set2[0], Set2[1], Set2[4], Set2[5]]
cMap = matplotlib.colors.ListedColormap(cL, name='from_list')

fig, axes = plt.subplots(2, 1, figsize = (17/gs.cm_in, 17/gs.cm_in))#, layout="constrained")

# style = 'date'
# hue = 'cell'

hue = 'manipID'
style = 'cell'
metric = 'Q_gf'

# Filter global_df

Filters = [(global_df['h3'] < 1.1),
           (global_df['Q_gf'] < 40),
            (global_df['date'] != '24-02-27'),
           # (global_df['date'] == '24-06-14'),
           ]

df_f = filterDf(global_df, Filters)
df_f = df_f.dropna(subset=metric)
df_f['h3'] *= 1000

dates = df_f['date'].unique()
manipes = df_f['manipID'].unique()
md = {manipes[i]:i for i in range(len(manipes))}
df_f['manipNum'] = df_f['manipID'].apply(lambda x : md[x])

agg_dict = {'manipID':'first', 'manipNum':'first', 'h3':['mean', 'std'], metric:['mean', 'std']}
df_f = df_f[['cellID'] + list(agg_dict.keys())].groupby('cellID').agg(agg_dict)
df_f.columns = ["_".join(a) for a in df_f.columns.to_flat_index()]
df_f['h3_CV'] = df_f['h3_std']/df_f['h3_mean']
df_f[metric + '_CV'] = df_f[metric + '_std']/df_f[metric + '_mean']

figCV, axCV = plt.subplots(1, 1, figsize = (12/gs.cm_in, 12/gs.cm_in))
axCV.axline((0,0), slope=1, ls='--', color='dimgray')
axCV.scatter(df_f['h3_CV'].values, df_f[metric + '_CV'].values, c = df_f['manipNum_first'],
             marker='o', s=20, zorder=3, cmap = cMap)
axCV.set_xlim([0, 0.5])
axCV.set_ylim([0, 0.5])
axCV.set_xlabel('CV of the thickness')
axCV.set_ylabel('CV of the actin quantity')
figCV.tight_layout()


bins = np.linspace(0, 1000, 10, endpoint=False)
df_f['h3_bin'] = np.digitize(df_f['h3_mean'].values, bins = bins)
df_fg = df_f[[metric + '_mean', 'h3_mean', 'h3_bin']].groupby('h3_bin').agg(['median', 'std'])
df_fg['h3_upper'] = df_fg.index*50
df_fg = df_fg.dropna()

#### Plot 1 - Lin scale

ax = axes[0]

X = df_f['h3_mean'].values
Y = df_f[metric + '_mean'].values
C = df_f['manipNum_first'].values

ax.set_prop_cycle(color=cL)
ax.scatter(X, Y, c=C, marker='o', s=20, zorder=3, cmap = cMap) # , style='cellNum'

Xg = df_fg['h3_mean', 'median'].values
Yg = df_fg[metric + '_mean', 'median'].values
# Xerr = df_fg['h3', 'median'].values
Ygerr = df_fg[metric + '_mean', 'std'].values

ax.errorbar(Xg, Yg, Ygerr, color = 'dimgray', zorder=5,
            lw = 1.5, ls = '-',
            marker = 'o', markersize = 6, markerfacecolor = 'w', markeredgecolor = 'dimgray', markeredgewidth = 1.5,
            elinewidth = 1, ecolor = 'dimgray', capsize = 3, capthick = 1,
            label = 'Binning & median')

# Xfit, Yfit = df_f['h3_mean'].values, df_f[metric + '_mean'].values
# [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
# R2 = w_results.rsquared
# pval = results.pvalues[1]
# Xplot = np.linspace(min(Xfit), max(Xfit), 50)
# Yplot = b + a*Xplot

# ax.plot(Xplot, Yplot, ls = '--', c = 'k', zorder=4, lw = 1.0,
#         label =  r'$\bf{Fit\ y\ =\ ax+b}$' + f'\n$a$ = {a:.1e}' + f'\n$b$  = {b:.2f}' + \
#                 f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
    
# xfit = df_f['h3'].values
# yfit = df_f[metric].values
# params, res = ufun.fitLineHuber(xfit, yfit)
# print(params)
# xfitplot = np.linspace(np.min(df_f['h3'].values), np.max(df_f['h3'].values), 100)
# ax.plot(xfitplot, params[0] + xfitplot*params[1], 'k--', zorder = 4,
#         label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')

# LegendMark = mlines.Line2D([], [], color='k', ls='--', marker='', 
#                             # markersize=0, markeredgecolor='w', markeredgewidth=0,
#                             label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')
# ax.legend(handles=[LegendMark])

ax.legend()

ax.set_xlim([0, 1100])
ax.set_ylim([0, 30])



#### Plot 2 - Log scale

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')

X = df_f['h3_mean'].values
Y = df_f[metric + '_mean'].values
C = df_f['manipNum' + '_first'].values

ax.set_prop_cycle(color=cL)
ax.scatter(X, Y, c=C, marker='o', s=20, zorder=3, cmap = cMap) # , style='cellNum'

Xg = df_fg['h3_mean', 'median'].values
Yg = df_fg[metric + '_mean', 'median'].values
# Xerr = df_fg['h3', 'median'].values
Ygerr = df_fg[metric + '_mean', 'std'].values

ax.errorbar(Xg, Yg, Ygerr, color = 'dimgray', zorder=5,
            lw = 1.5, ls = '-',
            marker = 'o', markersize = 6, markerfacecolor = 'w', markeredgecolor = 'dimgray', markeredgewidth = 1.5,
            elinewidth = 1, ecolor = 'dimgray', capsize = 3, capthick = 1,
            label = 'Binning & median')

Xm, Ym = np.median(X), np.median(Y)
A = 1.1*min(Yg)/min(Xg)
# A = 0.1
Xplot = np.logspace(1.9, 3.1, 50)
Yplot = A*Xplot
ax.plot(Xplot, Yplot, ls = '-.', c = 'gray', lw = 1.0,
        label =  r'$\bf{Line\ y\ =\ Ax}$')


# Xfit, Yfit = df_f['h3'].values, df_f[metric].values
# [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
# R2 = w_results.rsquared
# pval = results.pvalues[1]
# Xplot = np.linspace(min(Xfit), max(Xfit), 50)
# Yplot = b + a*Xplot

# ax.plot(Xplot, Yplot, ls = '--', c = 'k', lw = 2.0,
#         label =  r'$\bf{Fit\ y\ =\ ax+b}$' + f'\n$a$ = {a:.1e}' + f'\n$b$  = {b:.2f}' + \
#                 f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')

# sns.scatterplot(ax=ax, data=df_f, x='h3', y=metric, hue=hue, marker='o', s= 25, zorder=5) # , style='cellNum'

# xfit = df_f['h3'].values
# yfit = df_f[metric].values
# params, res = ufun.fitLineHuber(xfit, yfit)
# xfitplot = np.linspace(np.min(df_f['h3'].values), np.max(df_f['h3'].values), 100)
# ax.plot(xfitplot, params[0] + xfitplot*params[1], 'k--', label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')

# LegendMark = mlines.Line2D([], [], color='k', ls='--', marker='', 
#                             # markersize=0, markeredgecolor='w', markeredgewidth=0,
#                             label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')
# ax.legend(handles=[LegendMark])

ax.set_xlim([190, 1200])
# ax.set_ylim([5, 150])
ax.legend()

# ax.set_ylabel('Actin Quantity')
# ax.set_xlabel('Cortex thickness (nm)')
# ax.legend().set_visible(False)
for ax in axes:
    ax.grid()


# %%%% Plot - Lin - all dates V1

gs.set_manuscript_options_jv()
# fig = plt.figure(figsize = (17/gs.cm_in, 20/gs.cm_in))#, layout="constrained")
# spec = fig.add_gridspec(4, 1)
# ax0 = fig.add_subplot(spec[0])
# ax1 = fig.add_subplot(spec[1])
# ax2 = fig.add_subplot(spec[2])
# ax3 = fig.add_subplot(spec[3])
# ax4 = fig.add_subplot(spec[2,1])

fig, axes = plt.subplots(4, 1, figsize = (17/gs.cm_in, 20/gs.cm_in), sharex = True)

# axes = [ax1, ax2, ax3] #, ax4]

cL = matplotlib.colormaps['Set2']
# style = 'date'
# hue = 'cell'

hue = 'manipID'
style = 'cell'
metric = 'Q_gf'

# Filter global_df

Filters = [(global_df['h3'] < 0.95),
           # (global_df['date'] != ''),
           ]
df_f = filterDf(global_df, Filters)
df_f['h3'] *= 1000
dates = df_f['date'].unique()

# LegendMark = mlines.Line2D([], [], color='gray', ls='', marker='o', 
#                            markersize=0, markeredgecolor='w', markeredgewidth=0,
#                            label=f'Average intra-cell CV = {CV_per_cell_avg*100:.1f} %')
# ax.legend(handles=[LegendMark])

ax = axes[0]
sns.scatterplot(ax=ax, data=df_f, x='h3', y=metric, hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 30])
ax.set_ylabel('Actin Quantity')
ax.set_xlabel('Cortex thickness (nm)')
ax.legend().set_visible(False)
ax.grid()

for i, ax in enumerate(axes[1:]):
    date = dates[i]
    sns.scatterplot(ax=ax, data=df_f[df_f['date']==date], x='h3', y=metric, hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
    # ax.set_xlim([0, 0.8])
    ax.set_ylim([0, 30])
    ax.set_ylabel('Actin Quantity')
    ax.set_xlabel('Cortex thickness (nm)')
    ax.legend().set_visible(False)
    ax.grid()



fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Lin_AllDates_GF_V1', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%%% Plot - Lin - all dates V2

gs.set_manuscript_options_jv()
fig = plt.figure(figsize = (17/gs.cm_in, 20/gs.cm_in))#, layout="constrained")
spec = fig.add_gridspec(3, 2)
ax0 = fig.add_subplot(spec[0,:])
ax1 = fig.add_subplot(spec[1,0])
ax2 = fig.add_subplot(spec[1,1])
ax3 = fig.add_subplot(spec[2,0])
ax4 = fig.add_subplot(spec[2,1])

# fig, axes = plt.subplots(4, 1, figsize = (17/gs.cm_in, 20/gs.cm_in), sharex = True)

axes = [ax1, ax2, ax3, ax4]

cL = matplotlib.colormaps['Set2'].colors
# style = 'date'
# hue = 'cell'

hue = 'manipID'
style = 'cell'
metric = 'Q_gf'

# Filter global_df
Filters = [(global_df['h3'] < 1),
            (global_df['date'] != '24-02-27'),
           ]
df_f = filterDf(global_df, Filters)
df_f['h3'] *= 1000
df_f = df_f.dropna(subset=metric)

dates = df_f['date'].unique()
manipes = df_f['manipID'].unique()




ax = ax0
sns.scatterplot(ax=ax, data=df_f, x='h3', y=metric, hue=hue, marker='o', s= 25, zorder=5) # , style='cellNum'

xfit = df_f['h3'].values
yfit = df_f[metric].values
params, res = ufun.fitLineHuber(xfit, yfit)
print(params)
xfitplot = np.linspace(np.min(df_f['h3'].values), np.max(df_f['h3'].values), 100)
ax.plot(xfitplot, params[0] + xfitplot*params[1], 'k--', label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')

LegendMark = mlines.Line2D([], [], color='k', ls='--', marker='', 
                            # markersize=0, markeredgecolor='w', markeredgewidth=0,
                            label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')
ax.legend(handles=[LegendMark])

ax.set_xlim([0, 1100])
ax.set_ylim([0, 30])
ax.set_ylabel('Actin Quantity')
ax.set_xlabel('Cortex thickness (nm)')
# ax.legend().set_visible(False)
ax.grid()

for i, ax in enumerate(axes):
    c = cL[i]
    manip = manipes[i]
    data = df_f[df_f['manipID']==manip]

    sns.scatterplot(ax=ax, data=data, x='h3', y=metric, color = c, marker='o', s= 25, zorder=5) # , style='cellNum'
    
    xfit = data['h3'].values
    yfit = data[metric].values
    params, res = ufun.fitLineHuber(xfit, yfit)
    print(params)
    xfitplot = np.linspace(np.min(data['h3'].values), np.max(data['h3'].values), 100)
    ax.plot(xfitplot, params[0] + xfitplot*params[1], 'k--', label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')
    
    ax.set_xlim([0, 1100])
    ax.set_ylim([0, 30])
    ax.set_ylabel('Actin Quantity')
    ax.set_xlabel('Cortex thickness (nm)')
    ax.legend(loc = 'upper left', title = manip, title_fontsize = 8)
    ax.grid()



fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Lin_AllDates_GF_V2', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%%% Plot - Lin - all dates V3

gs.set_manuscript_options_jv()
fig = plt.figure(figsize = (17/gs.cm_in, 20/gs.cm_in))#, layout="constrained")
spec = fig.add_gridspec(3, 2)
ax0 = fig.add_subplot(spec[0,:])
ax1 = fig.add_subplot(spec[1,0])
ax2 = fig.add_subplot(spec[1,1])
ax3 = fig.add_subplot(spec[2,0])
ax4 = fig.add_subplot(spec[2,1])

# fig, axes = plt.subplots(4, 1, figsize = (17/gs.cm_in, 20/gs.cm_in), sharex = True)

axes = [ax1, ax2, ax3, ax4]

cL = matplotlib.colormaps['Set2'].colors
# style = 'date'
# hue = 'cell'

hue = 'manipID'
style = 'cell'
metric = 'Q_gf'

# Filter global_df

Filters = [(global_df['h3'] < 0.6),
           (global_df['date'] != '24-02-27'),
           ]
df_f = filterDf(global_df, Filters)
df_f = df_f.dropna(subset=metric)
df_f['h3'] *= 1000


dates = df_f['date'].unique()
manipes = df_f['manipID'].unique()




ax = ax0
sns.scatterplot(ax=ax, data=df_f, x='h3', y=metric, hue=hue, marker='o', s= 25, zorder=5) # , style='cellNum'

xfit = df_f['h3'].values
yfit = df_f[metric].values
params, res = ufun.fitLineHuber(xfit, yfit)
print(params)
xfitplot = np.linspace(np.min(df_f['h3'].values), np.max(df_f['h3'].values), 100)
ax.plot(xfitplot, params[0] + xfitplot*params[1], 'k--', label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')

LegendMark = mlines.Line2D([], [], color='k', ls='--', marker='', 
                            # markersize=0, markeredgecolor='w', markeredgewidth=0,
                            label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')
ax.legend(handles=[LegendMark])

ax.set_xlim([0, 650])
ax.set_ylim([0, 30])
ax.set_ylabel('Actin Quantity')
ax.set_xlabel('Cortex thickness (nm)')
# ax.legend().set_visible(False)
ax.grid()

for i, ax in enumerate(axes):
    c = cL[i]
    manip = manipes[i]
    data = df_f[df_f['manipID']==manip]

    sns.scatterplot(ax=ax, data=data, x='h3', y=metric, color = c, marker='o', s= 25, zorder=5) # , style='cellNum'
    
    xfit = data['h3'].values
    yfit = data[metric].values
    params, res = ufun.fitLineHuber(xfit, yfit)
    print(params)
    xfitplot = np.linspace(np.min(data['h3'].values), np.max(data['h3'].values), 100)
    ax.plot(xfitplot, params[0] + xfitplot*params[1], 'k--', label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')
    
    ax.set_xlim([0, 650])
    ax.set_ylim([0, 30])
    ax.set_ylabel('Actin Quantity')
    ax.set_xlabel('Cortex thickness (nm)')
    ax.legend(loc = 'upper left', title = manip, title_fontsize = 8)
    ax.grid()



fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Lin_AllDates_GF_V3', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% Plot - Log - all dates V3

gs.set_manuscript_options_jv()
fig = plt.figure(figsize = (17/gs.cm_in, 20/gs.cm_in))#, layout="constrained")
spec = fig.add_gridspec(3, 2)
ax0 = fig.add_subplot(spec[0,:])
ax1 = fig.add_subplot(spec[1,0])
ax2 = fig.add_subplot(spec[1,1])
ax3 = fig.add_subplot(spec[2,0])
ax4 = fig.add_subplot(spec[2,1])

# fig, axes = plt.subplots(4, 1, figsize = (17/gs.cm_in, 20/gs.cm_in), sharex = True)

axes = [ax1, ax2, ax3, ax4]

for ax in ([ax0] + axes):
    ax.set_xscale('log')
    ax.set_yscale('log')

cL = matplotlib.colormaps['Set2'].colors
# style = 'date'
# hue = 'cell'

hue = 'manipID'
style = 'cell'
metric = 'Q_gf'

# Filter global_df

Filters = [(global_df['h3'] < 0.6),
           (global_df['date'] != '24-02-27'),
           ]
df_f = filterDf(global_df, Filters)
df_f = df_f.dropna(subset=metric)
df_f['h3'] *= 1000


dates = df_f['date'].unique()
manipes = df_f['manipID'].unique()




ax = ax0
sns.scatterplot(ax=ax, data=df_f, x='h3', y=metric, hue=hue, marker='o', s= 25, zorder=5) # , style='cellNum'

xfit = df_f['h3'].values
yfit = df_f[metric].values
params, res = ufun.fitLineHuber(xfit, yfit)
print(params)
xfitplot = np.linspace(np.min(df_f['h3'].values), np.max(df_f['h3'].values), 100)
ax.plot(xfitplot, params[0] + xfitplot*params[1], 'k--', label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')

LegendMark = mlines.Line2D([], [], color='k', ls='--', marker='', 
                            # markersize=0, markeredgecolor='w', markeredgewidth=0,
                            label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')
ax.legend(handles=[LegendMark])

ax.set_xlim([180, 650])
ax.set_ylim([6, 30])
ax.set_ylabel('Actin Quantity')
ax.set_xlabel('Cortex thickness (nm)')
# ax.legend().set_visible(False)
ax.grid()

for i, ax in enumerate(axes):
    c = cL[i]
    manip = manipes[i]
    data = df_f[df_f['manipID']==manip]

    sns.scatterplot(ax=ax, data=data, x='h3', y=metric, color = c, marker='o', s= 25, zorder=5) # , style='cellNum'
    
    xfit = data['h3'].values
    yfit = data[metric].values
    params, res = ufun.fitLineHuber(xfit, yfit)
    print(params)
    xfitplot = np.linspace(np.min(data['h3'].values), np.max(data['h3'].values), 100)
    ax.plot(xfitplot, params[0] + xfitplot*params[1], 'k--', label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')
    
    ax.set_xlim([180, 650])
    ax.set_ylim([6, 30])
    ax.set_ylabel('Actin Quantity')
    ax.set_xlabel('Cortex thickness (nm)')
    ax.legend(loc = 'upper left', title = manip, title_fontsize = 8)
    ax.grid()



fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Log_AllDates_GF_V3', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%%% Plot - Log Density - all dates V3

gs.set_manuscript_options_jv()
fig = plt.figure(figsize = (17/gs.cm_in, 20/gs.cm_in))#, layout="constrained")
spec = fig.add_gridspec(3, 2)
ax0 = fig.add_subplot(spec[0,:])
ax1 = fig.add_subplot(spec[1,0])
ax2 = fig.add_subplot(spec[1,1])
ax3 = fig.add_subplot(spec[2,0])
ax4 = fig.add_subplot(spec[2,1])

# fig, axes = plt.subplots(4, 1, figsize = (17/gs.cm_in, 20/gs.cm_in), sharex = True)

axes = [ax1, ax2, ax3, ax4]

for ax in ([ax0] + axes):
    ax.set_xscale('log')
    ax.set_yscale('log')

cL = matplotlib.colormaps['Set2'].colors
# style = 'date'
# hue = 'cell'

hue = 'manipID'
style = 'cell'
metric = 'D_gf'

# Filter global_df

Filters = [(global_df['h3'] < 1),
           (global_df['date'] != '24-02-27'),
           ]
df_f = filterDf(global_df, Filters)
df_f = df_f.dropna(subset=metric)
df_f['h3'] *= 1000


dates = df_f['date'].unique()
manipes = df_f['manipID'].unique()




ax = ax0
sns.scatterplot(ax=ax, data=df_f, x='h3', y=metric, hue=hue, marker='o', s= 25, zorder=5) # , style='cellNum'

xfit = df_f['h3'].values
yfit = df_f[metric].values
params, res = ufun.fitLineHuber(xfit, yfit)
print(params)
xfitplot = np.linspace(np.min(df_f['h3'].values), np.max(df_f['h3'].values), 100)
ax.plot(xfitplot, params[0] + xfitplot*params[1], 'k--', label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')

LegendMark = mlines.Line2D([], [], color='k', ls='--', marker='', 
                            # markersize=0, markeredgecolor='w', markeredgewidth=0,
                            label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')
ax.legend(handles=[LegendMark])

# ax.set_xlim([180, 650])
# ax.set_ylim([6, 30])
ax.set_ylabel('Actin Quantity')
ax.set_xlabel('Cortex thickness (nm)')
# ax.legend().set_visible(False)
ax.grid()

for i, ax in enumerate(axes):
    c = cL[i]
    manip = manipes[i]
    data = df_f[df_f['manipID']==manip]

    sns.scatterplot(ax=ax, data=data, x='h3', y=metric, color = c, marker='o', s= 25, zorder=5) # , style='cellNum'
    
    xfit = data['h3'].values
    yfit = data[metric].values
    params, res = ufun.fitLineHuber(xfit, yfit)
    print(params)
    xfitplot = np.linspace(np.min(data['h3'].values), np.max(data['h3'].values), 100)
    ax.plot(xfitplot, params[0] + xfitplot*params[1], 'k--', label=f'Slope = {params[1]:.3f}\nIntercept = {params[0]:.1f}')
    
    # ax.set_xlim([180, 650])
    # ax.set_ylim([6, 30])
    ax.set_ylabel('Actin Quantity')
    ax.set_xlabel('Cortex thickness (nm)')
    ax.legend(loc = 'upper left', title = manip, title_fontsize = 8)
    ax.grid()



fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Log_AllDates_Density_GF_V3', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %% Plots

# %%% Plot -- standard

global_df = pd.read_csv("D:/MagneticPincherData/Data_Analysis/FluoQuantifs/All_global_DfMerged.csv", sep=None, engine='python')
global_df['manipID'] = global_df['date'] + '_' + global_df['manip']
global_df['cellID'] = global_df['date'] + '_' + global_df['manip'] + '_P1_' + global_df['cell']
# global_df.to_csv("D:/MagneticPincherData/Data_Analysis/FluoQuantifs/All_global_DfMerged.csv", sep = ';', index = False)

figDir = 'D:/MagneticPincherData/Figures/FluoAnalysis'
figSubDir = '24-07-19'

# %%%% Plot - Lin - all dates V1

gs.set_manuscript_options_jv()
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
# style = 'date'
# hue = 'cell'

hue = 'manipID'
style = 'cell'

# Filter global_df
Filters = [(global_df['h3'] < 0.9),
           ]

df = filterDf(global_df, Filters)

ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbL', hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 30])
ax.set_ylabel('Actin Quantity on a large width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()

ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbN', hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 30])
ax.set_ylabel('Actin Quantity on a narrow width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize = 6, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_vb', hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 40])
ax.set_ylabel('Actin Quantity on a variable window')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()

ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_gf', hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 50])
ax.set_ylabel('Actin Quantity with a gaussian fit')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()


fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Lin_AllDates_AllMetrics_V1', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% Plot - Lin - all dates V2

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
style = 'date'
hue = 'cell'

# hue = 'date'
# style = 'cell'

# Filter global_df
Filters = [(global_df['h3'] < 0.9),
           ]

df = filterDf(global_df, Filters)

ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbL', hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 30])
ax.set_ylabel('Actin Quantity on a large width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()

ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbN', hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 30])
ax.set_ylabel('Actin Quantity on a narrow width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize = 6, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_vb', hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 40])
ax.set_ylabel('Actin Quantity on a variable window')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()

ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_gf', hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 50])
ax.set_ylabel('Actin Quantity with a gaussian fit')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()


fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Lin_AllDates_AllMetrics_V2', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% Plot - Lin - all dates - width

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
hue = 'date'
style = 'cell'

# Filter global_df
Filters = [(global_df['h3'] < 0.9),
           ]

df = filterDf(global_df, Filters)

ax = axes[0]
sns.scatterplot(ax=ax, data=df, x='h3', y='W_vb', hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
# ax.set_ylim([0, 30])
ax.set_ylabel('Width of the variable window')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()

ax = axes[1]
sns.scatterplot(ax=ax, data=df, x='h3', y='S_gf', hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
# ax.set_ylim([0, 20])
ax.set_ylabel('Std of the gaussian fit')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize = 6, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Lin_AllDates_WidthOfWindows', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% Plot - Lin - 24-02-27

# Filter global_df
Filters = [(global_df['h3'] < 0.9),
           (global_df['date'] == '24-02-27')]

df = filterDf(global_df, Filters)

hue = 'cell'
style = None
s = 75

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbL', hue=hue, style=style, s= s, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 30])
ax.set_ylabel('Actin Quantity on a large width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()

ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbN', hue=hue, style=style, s= s, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 30])
ax.set_ylabel('Actin Quantity on a narrow width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize = 6, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_vb', hue=hue, style=style, s= s, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 40])
ax.set_ylabel('Actin Quantity on a variable window')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()

ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_gf', hue=hue, style=style, s= s, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 50])
ax.set_ylabel('Actin Quantity with a gaussian fit')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()


fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Lin_24-02-27_AllMetrics', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%%% Plot - Lin - 24-05-24

# Filter global_df
Filters = [(global_df['h3'] < 0.9),
           (global_df['date'] == '24-05-24')]

df = filterDf(global_df, Filters)

hue = 'cell'
style = None
s = 50

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbL', hue=hue, style=style, s= s, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 30])
ax.set_ylabel('Actin Quantity on a large width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()

ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbN', hue=hue, style=style, s= s, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 30])
ax.set_ylabel('Actin Quantity on a narrow width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize = 6, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_vb', hue=hue, style=style, s= s, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 40])
ax.set_ylabel('Actin Quantity on a variable window')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()

ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_gf', hue=hue, style=style, s= s, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 50])
ax.set_ylabel('Actin Quantity with a gaussian fit')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()


fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Lin_24-05-24_AllMetrics', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%%% Plot - Lin - 24-06-14

# Filter global_df
Filters = [(global_df['h3'] < 0.9),
           (global_df['date'] == '24-06-14')]

df = filterDf(global_df, Filters)

hue = 'cell'
style = None
s = 50

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbL', hue=hue, style=style, s= s, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 30])
ax.set_ylabel('Actin Quantity on a large width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()

ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbN', hue=hue, style=style, s= s, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 30])
ax.set_ylabel('Actin Quantity on a narrow width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize = 6, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_vb', hue=hue, style=style, s= s, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 40])
ax.set_ylabel('Actin Quantity on a variable window')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()

ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_gf', hue=hue, style=style, s= s, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 50])
ax.set_ylabel('Actin Quantity with a gaussian fit')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()


fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Lin_24-06-14_AllMetrics', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% Plot - Lin - Compression effect

# Filter global_df
Filters = [(global_df['h3'] < 0.9),
           (global_df['date'] == '24-06-14')]

df = filterDf(global_df, Filters)

hue = 'manip'
style = 'cell'
s = 50

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbL', hue=hue, style=style, s= s, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 30])
ax.set_ylabel('Actin Quantity on a large width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()

ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbN', hue=hue, style=style, s= s, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 30])
ax.set_ylabel('Actin Quantity on a narrow width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize = 6, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_vb', hue=hue, style=style, s= s, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 40])
ax.set_ylabel('Actin Quantity on a variable window')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()

ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_gf', hue=hue, style=style, s= s, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 50])
ax.set_ylabel('Actin Quantity with a gaussian fit')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid()


fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Lin_24-06-14_effectOfcompressions', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%%% Plot - Log

def mypowerlaw(X, a, b):
    return(np.exp(b)*X**a)

# Filter global_df
Filters = [(global_df['h3'] < 0.9),
           (global_df['Q_fbL'].apply(lambda x: not pd.isnull(x))),
           (global_df['Q_vb'].apply(lambda x: not pd.isnull(x))),
           (global_df['Q_gf'].apply(lambda x: not pd.isnull(x))),
           (global_df['date'].apply(lambda x : x in ['24-05-24', '24-06-14'])),
           ]

df = filterDf(global_df, Filters)

# global_df['h2'] = global_df['D2']-Dmoy
# global_df['h3'] = global_df['D3']-Dmoy

log_df = np.log(df[['D2', 'D3', 'F', 'Q_fbL', 'Q_fbN', 'Q_vb', 'Q_gf', 'h2', 'h3']])
log_df[['cell', 'date', 'cellId']] = df[['cell', 'date', 'cellId']]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
hue='date'
style='cell'

ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbL', hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
Xfit, Yfit = log_df['h3'].values, log_df['Q_fbL'].values
params, res = ufun.fitLineHuber(Xfit, Yfit)
Xfitplot = np.linspace(np.min(df['h3'].values), np.max(df['h3'].values), 100)
ax.plot(Xfitplot, mypowerlaw(Xfitplot, params[1], params[0]), 'k--', label='')
ax.set_title(f'Fit: Y = {np.exp(params[0]):.2f}.X^{params[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Actin Quantity on a large width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid(which='both', axis='both')

ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbN', hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
Xfit, Yfit = log_df['h3'].values, log_df['Q_fbN'].values
params, res = ufun.fitLineHuber(Xfit, Yfit)
Xfitplot = np.linspace(np.min(df['h3'].values), np.max(df['h3'].values), 100)
ax.plot(Xfitplot, mypowerlaw(Xfitplot, params[1], params[0]), 'k--', label='')
ax.set_title(f'Fit: Y = {np.exp(params[0]):.2f}.X^{params[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Actin Quantity on a narrow width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize = 6, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(which='both', axis='both')

ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_vb', hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
Xfit, Yfit = log_df['h3'].values, log_df['Q_vb'].values
params, res = ufun.fitLineHuber(Xfit, Yfit)
Xfitplot = np.linspace(np.min(df['h3'].values), np.max(df['h3'].values), 100)
ax.plot(Xfitplot, mypowerlaw(Xfitplot, params[1], params[0]), 'k--', label='')
ax.set_title(f'Fit: Y = {np.exp(params[0]):.2f}.X^{params[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Actin Quantity on a variable window')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid(which='both', axis='both')

ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_gf', hue=hue, style=style, s= 75, zorder=5)
Xfit, Yfit = log_df['h3'].values, log_df['Q_gf'].values
params, res = ufun.fitLineHuber(Xfit, Yfit)
Xfitplot = np.linspace(np.min(df['h3'].values), np.max(df['h3'].values), 100)
ax.plot(Xfitplot, mypowerlaw(Xfitplot, params[1], params[0]), 'k--', label='')
ax.set_title(f'Fit: Y = {np.exp(params[0]):.2f}.X^{params[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Actin Quantity with a gaussian fit')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid(which='both', axis='both')

fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Log_AllDates_AllMetrics', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% Plot - Lin - Density

# date = '24-02-27'
# date = '24-05-24'
date = '24-06-14'
Filters = [(global_df['h3'] < 0.9),
           (global_df['date'] == date)]

df = filterDf(global_df, Filters)


def mypowerlaw(X, a, b):
    return(np.exp(b)*X**a)


df['Density_Q_fbL'] = df['Q_fbL']/df['h3']
df['Density_Q_fbN'] = df['Q_fbN']/df['h3']
df['Density_Q_vb'] = df['Q_vb']/df['h3']
df['Density_Q_gf'] = df['Q_gf']/df['h3']

hue = 'cell'
style = 'date'
s = 75


fig, axes = plt.subplots(2, 2, figsize=(10, 8))

ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_fbL', hue=hue, style=style, s=s, zorder=5) # , style='cellNum'
ax.set_ylabel('Cortex Density on a large width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid(which='both', axis='both')

ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_fbN', hue=hue, style=style, s=s, zorder=5) # , style='cellNum'
ax.set_ylabel('Cortex Density on a narrow width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize = 6, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(which='both', axis='both')

ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_vb', hue=hue, style=style, s=s, zorder=5) # , style='cellNum'
ax.set_ylabel('Cortex Density on a variable window')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid(which='both', axis='both')

ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_gf', hue=hue, style=style, s=s, zorder=5) # , style='cellNum'
ax.set_ylabel('Cortex Density with a Gaussian fit')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid(which='both', axis='both')



fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = f'Lin_{date}_Density', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%%% Plot - Lin - Density - All dates

# date = '24-02-27'
# date = '24-05-24'
# date = '24-06-14'
Filters = [(global_df['h3'] < 0.9),
          ]

df = filterDf(global_df, Filters)


def mypowerlaw(X, a, b):
    return(np.exp(b)*X**a)


df['Density_Q_fbL'] = df['Q_fbL']/df['h3']
df['Density_Q_fbN'] = df['Q_fbN']/df['h3']
df['Density_Q_vb'] = df['Q_vb']/df['h3']
df['Density_Q_gf'] = df['Q_gf']/df['h3']

style = 'cell'
hue = 'date'
s = 75


fig, axes = plt.subplots(2, 2, figsize=(10, 8))

ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_fbL', hue=hue, style=style, s=s, zorder=5) # , style='cellNum'
ax.set_ylabel('Cortex Density on a large width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid(which='both', axis='both')

ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_fbN', hue=hue, style=style, s=s, zorder=5) # , style='cellNum'
ax.set_ylabel('Cortex Density on a narrow width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize = 6, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(which='both', axis='both')

ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_vb', hue=hue, style=style, s=s, zorder=5) # , style='cellNum'
ax.set_ylabel('Cortex Density on a variable window')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid(which='both', axis='both')

ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_gf', hue=hue, style=style, s=s, zorder=5) # , style='cellNum'
ax.set_ylabel('Cortex Density with a Gaussian fit')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid(which='both', axis='both')



fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = f'Lin_AllDates_Density', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% Plot - Log - Density

# date = '24-02-27'
# date = '24-05-24'
date = '24-06-14'
Filters = [(global_df['h3'] < 0.9),
           (global_df['Q_fbL'].apply(lambda x: not pd.isnull(x))),
           (global_df['Q_fbN'].apply(lambda x: not pd.isnull(x))),
           (global_df['Q_vb'].apply(lambda x: not pd.isnull(x))),
           (global_df['Q_gf'].apply(lambda x: not pd.isnull(x))),
           (global_df['date'] == date)]

df = filterDf(global_df, Filters)


def mypowerlaw(X, a, b):
    return(np.exp(b)*X**a)

df['Density_Q_fbN'] = df['Q_fbN']/df['h3']
df['Density_Q_fbL'] = df['Q_fbL']/df['h3']
df['Density_Q_vb'] = df['Q_vb']/df['h3']
df['Density_Q_gf'] = df['Q_gf']/df['h3']

style = 'cell'
hue = 'date'
s = 75


fig, axes = plt.subplots(2, 2, figsize=(10, 8))

ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_fbL', hue=hue, style=style, s=s, zorder=5) # , style='cellNum'
Xfit, Yfit = np.log(df['h3'].values), np.log(df['Density_Q_fbL'].values)
params, res = ufun.fitLineHuber(Xfit, Yfit)
Xfitplot = np.linspace(np.min(df['h3'].values), np.max(df['h3'].values), 100)
ax.plot(Xfitplot, mypowerlaw(Xfitplot, params[1], params[0]), 'k--')
ax.set_title(f'Fit: Y = {np.exp(params[0]):.2f}.X^{params[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Cortex Density on a large width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid(which='both', axis='both')

ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_fbN', hue=hue, style=style, s=s, zorder=5) # , style='cellNum'
Xfit, Yfit = np.log(df['h3'].values), np.log(df['Density_Q_fbN'].values)
params, res = ufun.fitLineHuber(Xfit, Yfit)
Xfitplot = np.linspace(np.min(df['h3'].values), np.max(df['h3'].values), 100)
ax.plot(Xfitplot, mypowerlaw(Xfitplot, params[1], params[0]), 'k--')
ax.set_title(f'Fit: Y = {np.exp(params[0]):.2f}.X^{params[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Cortex Density on a narrow width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize = 6, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(which='both', axis='both')

ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_vb', hue=hue, style=style, s=s, zorder=5) # , style='cellNum'
Xfit, Yfit = np.log(df['h3'].values), np.log(df['Density_Q_vb'].values)
params, res = ufun.fitLineHuber(Xfit, Yfit)
Xfitplot = np.linspace(np.min(df['h3'].values), np.max(df['h3'].values), 100)
ax.plot(Xfitplot, mypowerlaw(Xfitplot, params[1], params[0]), 'k--')
ax.set_title(f'Fit: Y = {np.exp(params[0]):.2f}.X^{params[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Cortex Density on a variable window')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid(which='both', axis='both')

ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_gf', hue=hue, style=style, s=s, zorder=5) # , style='cellNum'
Xfit, Yfit = np.log(df['h3'].values), np.log(df['Density_Q_gf'].values)
params, res = ufun.fitLineHuber(Xfit, Yfit)
Xfitplot = np.linspace(np.min(df['h3'].values), np.max(df['h3'].values), 100)
ax.plot(Xfitplot, mypowerlaw(Xfitplot, params[1], params[0]), 'k--')
ax.set_title(f'Fit: Y = {np.exp(params[0]):.2f}.X^{params[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Cortex Density with a Gaussian fit')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid(which='both', axis='both')



fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = f'Log_{date}_Density', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% Plot - Log - Density - All dates

Filters = [(global_df['h3'] < 0.9),
           (global_df['Q_fbL'].apply(lambda x: not pd.isnull(x))),
           (global_df['Q_fbN'].apply(lambda x: not pd.isnull(x))),
           (global_df['Q_vb'].apply(lambda x: not pd.isnull(x))),
           (global_df['Q_gf'].apply(lambda x: not pd.isnull(x))),
           ]

df = filterDf(global_df, Filters)


def mypowerlaw(X, a, b):
    return(np.exp(b)*X**a)

df['Density_Q_fbN'] = df['Q_fbN']/df['h3']
df['Density_Q_fbL'] = df['Q_fbL']/df['h3']
df['Density_Q_vb'] = df['Q_vb']/df['h3']
df['Density_Q_gf'] = df['Q_gf']/df['h3']

style = 'cell'
hue = 'date'
s = 75


fig, axes = plt.subplots(2, 2, figsize=(10, 8))

ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_fbL', hue=hue, style=style, s=s, zorder=5) # , style='cellNum'
Xfit, Yfit = np.log(df['h3'].values), np.log(df['Density_Q_fbL'].values)
params, res = ufun.fitLineHuber(Xfit, Yfit)
Xfitplot = np.linspace(np.min(df['h3'].values), np.max(df['h3'].values), 100)
ax.plot(Xfitplot, mypowerlaw(Xfitplot, params[1], params[0]), 'k--')
ax.set_title(f'Fit: Y = {np.exp(params[0]):.2f}.X^{params[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Cortex Density on a large width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid(which='both', axis='both')

ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_fbN', hue=hue, style=style, s=s, zorder=5) # , style='cellNum'
Xfit, Yfit = np.log(df['h3'].values), np.log(df['Density_Q_fbN'].values)
params, res = ufun.fitLineHuber(Xfit, Yfit)
Xfitplot = np.linspace(np.min(df['h3'].values), np.max(df['h3'].values), 100)
ax.plot(Xfitplot, mypowerlaw(Xfitplot, params[1], params[0]), 'k--')
ax.set_title(f'Fit: Y = {np.exp(params[0]):.2f}.X^{params[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Cortex Density on a narrow width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize = 6, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(which='both', axis='both')

ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_vb', hue=hue, style=style, s=s, zorder=5) # , style='cellNum'
Xfit, Yfit = np.log(df['h3'].values), np.log(df['Density_Q_vb'].values)
params, res = ufun.fitLineHuber(Xfit, Yfit)
Xfitplot = np.linspace(np.min(df['h3'].values), np.max(df['h3'].values), 100)
ax.plot(Xfitplot, mypowerlaw(Xfitplot, params[1], params[0]), 'k--')
ax.set_title(f'Fit: Y = {np.exp(params[0]):.2f}.X^{params[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Cortex Density on a variable window')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid(which='both', axis='both')

ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_gf', hue=hue, style=style, s=s, zorder=5) # , style='cellNum'
Xfit, Yfit = np.log(df['h3'].values), np.log(df['Density_Q_gf'].values)
params, res = ufun.fitLineHuber(Xfit, Yfit)
Xfitplot = np.linspace(np.min(df['h3'].values), np.max(df['h3'].values), 100)
ax.plot(Xfitplot, mypowerlaw(Xfitplot, params[1], params[0]), 'k--')
ax.set_title(f'Fit: Y = {np.exp(params[0]):.2f}.X^{params[1]:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Cortex Density with a Gaussian fit')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend().set_visible(False)
ax.grid(which='both', axis='both')



fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = f'Log_AllDates_Density', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%% Plot -- stairs

stairs_df = pd.read_csv("D:/MagneticPincherData/Data_Analysis/FluoQuantifs/24-05-24_stairs_DfMerged.csv", sep=None, engine='python')
stairs_df['cellId'] = '24-05-24_' + stairs_df['cell']

# %%%% Plot - Lin - both dates

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
style = None
hue = 'cell'

# Filter global_df
Filters = [(global_df['h3'] < 900),
           ]

df = filterDf(stairs_df, Filters)

ax = axes[0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fb', hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 30])
ax.set_ylabel('Actin Quantity on a large width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize=6)
ax.grid()

ax = axes[1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_vb', hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 40])
ax.set_ylabel('Actin Quantity on a variable window')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize=6)
ax.grid()

ax = axes[2]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_gf', hue=hue, style=style, s= 75, zorder=5) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 50])
ax.set_ylabel('Actin Quantity with a gaussian fit')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize=6)
ax.grid()


fig.tight_layout()
plt.show()


# %%% Plot -- with compressions V1

fluo_df = pd.read_csv("D:/MagneticPincherData/Data_Analysis/FluoQuantifs/All_global_DfMerged.csv", sep=None, engine='python')
Filters = [(fluo_df['date'] == '24-06-14'),
           (fluo_df['manip'].apply(lambda x : x in ['M2', 'M3'])),
           (fluo_df['cellID'].apply(lambda x : x not in ['24-06-14_M2_P1_C1'])),
           ]
fluo_df_f = filterDf(fluo_df, Filters)
fluo_df_f = fluo_df_f.dropna(subset = ['Q_fbL', 'Q_fbN', 'Q_vb', 'Q_gf'])
fluo_df_f['Density_Q_fbN'] = fluo_df_f['Q_fbN']/fluo_df_f['h3']
fluo_df_f['Density_Q_fbL'] = fluo_df_f['Q_fbL']/fluo_df_f['h3']
fluo_df_f['Density_Q_vb'] = fluo_df_f['Q_vb']/fluo_df_f['h3']
fluo_df_f['Density_Q_gf'] = fluo_df_f['Q_gf']/fluo_df_f['h3']

fluo_df_fg = dataGroup(fluo_df_f, groupCol = 'cellID', idCols = ['date', 'manip', 'cell'], 
                       numCols = ['Q_fbL', 'Q_fbN', 'Q_vb', 'Q_gf', 'h3', 'Density_Q_fbN', 'Density_Q_fbL', 'Density_Q_vb', 'Density_Q_gf'],
                       aggFun = 'median')

meca_df = pd.read_csv("D:/MagneticPincherData/Data_Analysis/FluoQuantifs/MecaData_Chameleon_CompFluo.csv", sep=None, engine='python')
meca_df['cellID_2'] = meca_df['cellID'].apply(lambda x : '-'.join(x.split('-')[:-1]))
Filters = [(meca_df['date'] == '24-06-14'),
           (meca_df['bestH0'] <= 1200),
           ]

meca_df_f = filterDf(meca_df, Filters)
meca_df_fg1 = dataGroup(meca_df_f, groupCol = 'cellID_2', idCols = ['date'], 
                        numCols = ['surroundingThickness', 'bestH0'], aggFun = 'mean')
meca_df_fg2 = dataGroup_weightedAverage(meca_df_f, groupCol = 'cellID_2', idCols = ['date'], 
                                        valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
meca_df_fg = meca_df_fg1.merge(meca_df_fg2, on='cellID_2')

merged_df = fluo_df_fg.merge(meca_df_fg, left_on = 'cellID', right_on = 'cellID_2', how = 'inner')


figDir = 'D:/MagneticPincherData/Figures/FluoAnalysis'
figSubDir = '24-06-28'


# %%%% fluo - grouped vs ungrouped - Qtt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# gs.set_manuscript_options_jv(palette = 'Set2')

# Filter global_df

df = fluo_df_f
hue = 'cellID'
style = None
s = 25
alpha = 0.8
zo = 5
pal = sns.color_palette(gs.cL_Set21, len(fluo_df_fg))

ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbL', hue=hue, style=style, s= s, palette = pal,
                alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 20])
ax.set_xlim([0, 1.4])
ax.set_ylabel('Actin Quantity on a large width')
ax.set_xlabel('Cortex thickness (µm)')
ax.grid()

ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbN', hue=hue, style=style, s= s, palette = pal,
                alpha = alpha, zorder=zo) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 20])
ax.set_xlim([0, 1.4])
ax.set_ylabel('Actin Quantity on a narrow width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_vb', hue=hue, style=style, s= s, palette = pal,
                alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 30])
ax.set_xlim([0, 1.4])
ax.set_ylabel('Actin Quantity on a variable window')
ax.set_xlabel('Cortex thickness (µm)')
ax.grid()

ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_gf', hue=hue, style=style, s= s, palette = pal,
                alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 30])
ax.set_xlim([0, 1.4])
ax.set_ylabel('Actin Quantity with a gaussian fit')
ax.set_xlabel('Cortex thickness (µm)')
ax.grid()




df = fluo_df_fg
hue = 'cellID'
style = None
s = 60
alpha = 1.0
zo = 6
ec = 'k'


ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbL', hue=hue, style=style, palette = pal,
                s= s, ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
# ax.legend().set_visible(False)


ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_fbN', hue=hue, style=style, palette = pal,
                s= s, ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
# ax.legend().set_visible(False)


ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_vb', hue=hue, style=style, palette = pal,
                s= s, ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
# ax.legend(fontsize = 6, loc='center left', bbox_to_anchor=(1, 0.5))


ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Q_gf', hue=hue, style=style, palette = pal,
                s= s, ec = ec, alpha = alpha, zorder=zo) # , style='cellNum'
ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))




fig.suptitle('Quantity v thickness - with cell medians')
fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Comp_Qtt_Lin', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% fluo - grouped vs ungrouped - Density

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# gs.set_manuscript_options_jv(palette = 'Set2')

# Filter global_df

df = fluo_df_f
hue = 'cellID'
style = None
s = 25
alpha = 0.8
zo = 5
pal = sns.color_palette(gs.cL_Set21, len(fluo_df_fg))

ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_fbL', hue=hue, style=style, s= s, palette = pal,
                alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 40])
ax.set_xlim([0, 1.4])
ax.set_ylabel('Actin Density on a large width')
ax.set_xlabel('Cortex thickness (µm)')
ax.grid()

ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_fbN', hue=hue, style=style, s= s, palette = pal,
                alpha = alpha, zorder=zo) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 40])
ax.set_xlim([0, 1.4])
ax.set_ylabel('Actin Density on a narrow width')
ax.set_xlabel('Cortex thickness (µm)')
ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_vb', hue=hue, style=style, s= s, palette = pal,
                alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 40])
ax.set_xlim([0, 1.4])
ax.set_ylabel('Actin Density on a variable window')
ax.set_xlabel('Cortex thickness (µm)')
ax.grid()

ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_gf', hue=hue, style=style, s= s, palette = pal,
                alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
# ax.set_xlim([0, 0.8])
ax.set_ylim([0, 40])
ax.set_xlim([0, 1.4])
ax.set_ylabel('Actin Density with a gaussian fit')
ax.set_xlabel('Cortex thickness (µm)')
ax.grid()




df = fluo_df_fg
hue = 'cellID'
style = None
s = 60
alpha = 1.0
zo = 6
ec = 'k'


ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_fbL', hue=hue, style=style, palette = pal,
                s= s, ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
# ax.legend().set_visible(False)


ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_fbN', hue=hue, style=style, palette = pal,
                s= s, ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
# ax.legend().set_visible(False)


ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_vb', hue=hue, style=style, palette = pal,
                s= s, ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
# ax.legend(fontsize = 6, loc='center left', bbox_to_anchor=(1, 0.5))


ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df, x='h3', y='Density_Q_gf', hue=hue, style=style, palette = pal,
                s= s, ec = ec, alpha = alpha, zorder=zo) # , style='cellNum'
ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))




fig.suptitle('Density v thickness - with cell medians')
fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Comp_Density_Lin', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%%% meca - grouped vs ungrouped

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# gs.set_manuscript_options_jv(palette = 'Set2')

# Filter global_df

df = meca_df_f
hue = 'cellID_2'
x = 'surroundingThickness'
y = 'E_f_<_400'
style = None
s = 25
alpha = 0.8
zo = 5
pal = sns.color_palette(gs.cL_Set21, len(meca_df_fg))

ax = ax
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s, palette = pal,
                alpha = alpha, zorder=zo, legend = True) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex thickness (nm)')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()




df = meca_df_fg
hue = 'cellID_2'
x = 'surroundingThickness'
y = 'E_f_<_400_wAvg'
style = None
s = 60
alpha = 1.0
zo = 6
ec = 'k'

ax = ax
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, palette = pal,
                s= s, ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
# ax.legend().set_visible(False)




fig.suptitle('Stiffness v thickness - with cell w. avg.')
fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Comp_Stiff_Lin', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% cross - stiff vs density

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
# gs.set_manuscript_options_jv(palette = 'Set2')

# Filter global_df

df = merged_df
hue = 'cellID_2'
y = 'E_f_<_400_wAvg'
style = None
s = 60
alpha = 1
zo = 5
ec = 'k'
pal = sns.color_palette(gs.cL_Set21, len(merged_df))

ax = axes[0, 0]
x = 'Density_Q_fbL'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s, palette = pal,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - large window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[0, 1]
x = 'Density_Q_fbN'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s, palette = pal,
                ec = ec, alpha = alpha, zorder=zo, legend = True) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - narrow window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 0]
x = 'Density_Q_vb'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s, palette = pal,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - variable bounds')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 1]
x = 'Density_Q_gf'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s, palette = pal,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - gaussian fit')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()



fig.suptitle('Stiffness v density - with cell w. avg.')
fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Comp_Density-Stiff_Lin', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% cross - stiff vs density with color

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
# gs.set_manuscript_options_jv(palette = 'Set2')

# Filter global_df

df = merged_df
hue = 'h3'
y = 'E_f_<_400_wAvg'
style = None
s = 60
alpha = 1
zo = 5
ec = 'None'

ax = axes[0, 0]
x = 'Density_Q_fbL'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - large window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[0, 1]
x = 'Density_Q_fbN'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = True) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - narrow window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 0]
x = 'Density_Q_vb'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - variable bounds')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 1]
x = 'Density_Q_gf'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - gaussian fit')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()



fig.suptitle('Stiffness v density - with cell w. avg.')
fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Comp_H3-Density-Stiff_Lin', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% cross - stiff vs thick with color for density

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
# gs.set_manuscript_options_jv(palette = 'Set2')

# Filter
Filters = [
           (merged_df['cellID_2'].apply(lambda x : x not in ['24-06-14_M2_P1_C1',
                                                             '24-06-14_M2_P1_C9',
                                                             '24-06-14_M3_P1_C1'])),
           ]
df = filterDf(merged_df, Filters)

x = 'surroundingThickness'
y = 'E_f_<_400_wAvg'
style = None
s = 60
alpha = 1
zo = 5
ec = 'None'

ax = axes[0, 0]

hue = 'Density_Q_fbL'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo) # , style='cellNum'
ax.set_xlabel('Cortex thickness (nm)')
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_title('Cortex density (au) - large window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 6, title = 'Density\nLW', loc = 'upper left')
ax.grid()

ax = axes[0, 1]
hue = 'Density_Q_fbN'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo) # , style='cellNum'
ax.set_xlabel('Cortex thickness (nm)')
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_title('Cortex density (au) - narrow window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 6, title = 'Density\nNW', loc = 'upper left')
ax.grid()

ax = axes[1, 0]
hue = 'Density_Q_vb'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo) # , style='cellNum'
ax.set_xlabel('Cortex thickness (nm)')
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_title('Cortex density (au) - variable bounds')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 6, title = 'Density\nVB', loc = 'upper left')
ax.grid()

ax = axes[1, 1]
hue = 'Density_Q_gf'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo) # , style='cellNum'
ax.set_xlabel('Cortex thickness (nm)')
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_title('Cortex density (au) - gaussian fit')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 6, title = 'Density\nGF', loc = 'upper left')
ax.grid()



fig.suptitle('Stiffness v density - with cell w. avg.')
fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Comp_Density-SurrH-Stiff_Lin', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')









# %%% Plot -- with compressions V2

fluo_df = pd.read_csv("D:/MagneticPincherData/Data_Analysis/FluoQuantifs/All_global_DfMerged.csv", sep=None, engine='python')
Filters = [(fluo_df['date'] == '24-06-14'),
           (fluo_df['manip'].apply(lambda x : x in ['M2', 'M3'])),
           (fluo_df['cellID'].apply(lambda x : x not in ['24-06-14_M2_P1_C1'])),
           ]
fluo_df_f = filterDf(fluo_df, Filters)
fluo_df_f = fluo_df_f.dropna(subset = ['Q_fbL', 'Q_fbN', 'Q_vb', 'Q_gf'])
fluo_df_f['D1_fbN'] = fluo_df_f['Q_fbN']/fluo_df_f['h3']
fluo_df_f['D1_fbL'] = fluo_df_f['Q_fbL']/fluo_df_f['h3']
fluo_df_f['D1_vb'] = fluo_df_f['Q_vb']/fluo_df_f['h3']
fluo_df_f['D1_gf'] = fluo_df_f['Q_gf']/fluo_df_f['h3']

fluo_df_fg = dataGroup(fluo_df_f, groupCol = 'cellID', idCols = ['date', 'manip', 'cell'], 
                       numCols = ['Q_fbL', 'Q_fbN', 'Q_vb', 'Q_gf', 'h3', 'D1_fbN', 'D1_fbL', 'D1_vb', 'D1_gf'],
                       aggFun = 'median')

meca_df = pd.read_csv("D:/MagneticPincherData/Data_Analysis/FluoQuantifs/MecaData_Chameleon_CompFluo.csv", sep=None, engine='python')
meca_df['cellID_2'] = meca_df['cellID'].apply(lambda x : '-'.join(x.split('-')[:-1]))
Filters = [(meca_df['date'] == '24-06-14'),
           (meca_df['surroundingThickness'] <= 900),
           ]

meca_df_f = filterDf(meca_df, Filters)
# meca_df_fg1 = dataGroup(meca_df_f, groupCol = 'cellID_2', idCols = ['date'], 
#                         numCols = ['surroundingThickness', 'bestH0'], aggFun = 'mean')
# meca_df_fg2 = dataGroup_weightedAverage(meca_df_f, groupCol = 'cellID_2', idCols = ['date'], 
#                                         valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
# meca_df_fg = meca_df_fg1.merge(meca_df_fg2, on='cellID_2')

merged_df = fluo_df_fg.merge(meca_df_f, left_on = 'cellID', right_on = 'cellID_2', how = 'inner')

merged_df['D2_fbN'] = 1000*merged_df['Q_fbN']/merged_df['surroundingThickness']
merged_df['D2_fbL'] = 1000*merged_df['Q_fbL']/merged_df['surroundingThickness']
merged_df['D2_vb'] = 1000*merged_df['Q_vb']/merged_df['surroundingThickness']
merged_df['D2_gf'] = 1000*merged_df['Q_gf']/merged_df['surroundingThickness']


figDir = 'D:/MagneticPincherData/Figures/FluoAnalysis'
figSubDir = '24-07-02_OtherCross'


# %%%% cross - stiff vs density

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
# gs.set_manuscript_options_jv(palette = 'Set2')

# Filter global_df

df = merged_df
hue = 'cellID_2'
y = 'E_f_<_400'
style = None
s = 60
alpha = 1
zo = 5
ec = 'k'
pal = sns.color_palette(gs.cL_Set21, len(merged_df['cellID_2'].unique()))

ax = axes[0, 0]
x = 'D1_fbL'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s, palette = pal,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - large window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[0, 1]
x = 'D1_fbN'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s, palette = pal,
                ec = ec, alpha = alpha, zorder=zo, legend = True) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - narrow window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 0]
x = 'D1_vb'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s, palette = pal,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - variable bounds')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 1]
x = 'D1_gf'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s, palette = pal,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - gaussian fit')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()



fig.suptitle('Stiffness v density - with cell w. avg.')
fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Comp_Density-Stiff_Lin', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%%% cross - stiff vs density 2

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
# gs.set_manuscript_options_jv(palette = 'Set2')

# Filter global_df

df = merged_df
hue = 'cellID_2'
y = 'E_f_<_400'
style = None
s = 60
alpha = 1
zo = 5
ec = 'k'
pal = sns.color_palette(gs.cL_Set21, len(merged_df['cellID_2'].unique()))

ax = axes[0, 0]
x = 'D2_fbL'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s, palette = pal,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - large window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[0, 1]
x = 'D2_fbN'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s, palette = pal,
                ec = ec, alpha = alpha, zorder=zo, legend = True) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - narrow window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 0]
x = 'D2_vb'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s, palette = pal,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - variable bounds')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 1]
x = 'D2_gf'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s, palette = pal,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - gaussian fit')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()



fig.suptitle('Stiffness v density 2 - with cell w. avg.')
fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Comp_Density2-Stiff_Lin', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% cross - stiff vs density with color

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
# gs.set_manuscript_options_jv(palette = 'Set2')

# Filter global_df

df = merged_df
hue = 'h3'
y = 'E_f_<_400'
style = None
s = 60
alpha = 1
zo = 5
ec = 'None'

ax = axes[0, 0]
x = 'D1_fbL'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - large window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[0, 1]
x = 'D1_fbN'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = True) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - narrow window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 0]
x = 'D1_vb'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - variable bounds')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 1]
x = 'D1_gf'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - gaussian fit')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()



fig.suptitle('Stiffness v density - with cell w. avg.')
fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Comp_H3-Density-Stiff_Lin', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% cross 2 - stiff vs density with color

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
# gs.set_manuscript_options_jv(palette = 'Set2')

# Filter global_df

df = merged_df
hue = 'surroundingThickness'
y = 'E_f_<_400'
style = None
s = 60
alpha = 1
zo = 5
ec = 'None'

ax = axes[0, 0]
x = 'D2_fbL'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - large window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[0, 1]
x = 'D2_fbN'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = True) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - narrow window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 0]
x = 'D2_vb'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - variable bounds')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

ax = axes[1, 1]
x = 'D2_gf'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo, legend = False) # , style='cellNum'
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_xlabel('Cortex density (au) - gaussian fit')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
# ax.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()



fig.suptitle('Stiffness v density 2 - with cell w. avg.')
fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Comp_H3-Density2-Stiff_Lin', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% cross - stiff vs thick with color for density

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
# gs.set_manuscript_options_jv(palette = 'Set2')

# Filter
Filters = [
           (merged_df['cellID_2'].apply(lambda x : x not in ['24-06-14_M2_P1_C1',
                                                             '24-06-14_M2_P1_C9',
                                                             '24-06-14_M3_P1_C1'])),
           ]
df = filterDf(merged_df, Filters)

x = 'surroundingThickness'
y = 'E_f_<_400'
style = None
s = 60
alpha = 1
zo = 5
ec = 'None'

ax = axes[0, 0]

hue = 'D1_fbL'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo) # , style='cellNum'
ax.set_xlabel('Cortex thickness (nm)')
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_title('Cortex density (au) - large window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 6, title = 'Density\nLW', loc = 'upper left')
ax.grid()

ax = axes[0, 1]
hue = 'D1_fbN'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo) # , style='cellNum'
ax.set_xlabel('Cortex thickness (nm)')
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_title('Cortex density (au) - narrow window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 6, title = 'Density\nNW', loc = 'upper left')
ax.grid()

ax = axes[1, 0]
hue = 'D1_vb'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo) # , style='cellNum'
ax.set_xlabel('Cortex thickness (nm)')
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_title('Cortex density (au) - variable bounds')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 6, title = 'Density\nVB', loc = 'upper left')
ax.grid()

ax = axes[1, 1]
hue = 'D1_gf'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo) # , style='cellNum'
ax.set_xlabel('Cortex thickness (nm)')
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_title('Cortex density (au) - gaussian fit')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 6, title = 'Density\nGF', loc = 'upper left')
ax.grid()



fig.suptitle('Stiffness v density - with cell w. avg.')
fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Comp_Density-SurrH-Stiff_Lin', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%%% cross 2 - stiff vs thick with color for density

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
# gs.set_manuscript_options_jv(palette = 'Set2')

# Filter
Filters = [
           (merged_df['cellID_2'].apply(lambda x : x not in ['24-06-14_M2_P1_C1',
                                                             '24-06-14_M2_P1_C9',
                                                             '24-06-14_M3_P1_C1'])),
           ]
df = filterDf(merged_df, Filters)

x = 'surroundingThickness'
y = 'E_f_<_400'
style = None
s = 60
alpha = 1
zo = 5
ec = 'None'

ax = axes[0, 0]

hue = 'D2_fbL'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo) # , style='cellNum'
ax.set_xlabel('Cortex thickness (nm)')
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_title('Cortex density (au) - large window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 6, title = 'Density\nLW', loc = 'upper left')
ax.grid()

ax = axes[0, 1]
hue = 'D2_fbN'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo) # , style='cellNum'
ax.set_xlabel('Cortex thickness (nm)')
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_title('Cortex density (au) - narrow window')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 6, title = 'Density\nNW', loc = 'upper left')
ax.grid()

ax = axes[1, 0]
hue = 'D2_vb'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo) # , style='cellNum'
ax.set_xlabel('Cortex thickness (nm)')
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_title('Cortex density (au) - variable bounds')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 6, title = 'Density\nVB', loc = 'upper left')
ax.grid()

ax = axes[1, 1]
hue = 'D2_gf'
sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=hue, style=style, s= s,
                ec = ec, alpha = alpha, zorder=zo) # , style='cellNum'
ax.set_xlabel('Cortex thickness (nm)')
ax.set_ylabel('Cortex stiffness (Pa)')
ax.set_title('Cortex density (au) - gaussian fit')
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_xlim([0, ax.get_xlim()[-1]])
ax.legend(fontsize = 6, title = 'Density\nGF', loc = 'upper left')
ax.grid()



fig.suptitle('Stiffness v density 2 - with cell w. avg.')
fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Comp_Density2-SurrH-Stiff_Lin', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

















# %% Archives





#### Start

plt.ioff()
I = skm.io.imread(fluoPath)

# czt_shape, czt_seq = get_CZT_fromTiff(fluoPath)
# [nC, nZ, nT] = czt_shape
[nC, nT, nZ] = CTZ_tz.shape[:3]

Angles = np.arange(0, 360, 1)
previousCircleXYR = (50,50,50)

tsdf = pd.read_csv(resBfPath, sep=None, engine='python')

colsContact = ['iT', 'T', 'X', 'Y', 'iZ', 'Z', 'R', 'A', 'Xc', 'Yc', 'Rc', 
               'Xin', 'Yin', 'Zin', 'Xout', 'Yout', 'Zout', 
               ]
dfContact = pd.DataFrame(np.zeros((nT, len(colsContact))), columns=colsContact)

fluoCell = np.zeros((nT, nZ))
fluoCyto = np.zeros((nT, nZ))
fluoCytoStd = np.zeros((nT, nZ))
fluoBack = np.zeros((nT, nZ))
fluoBeadIn = np.zeros((nT, nZ))

arrayViterbiContours = np.zeros((nT, nZ, 360, 2))
profileMatrix = np.zeros((nT, nZ, N_profilesMatrix, warp_radius))

for t in range(nT):
    intensityMap_t = []
    tsd_t = tsdf.iloc[t,:].to_dict()
        
    #### 1.0 Compute position of contact point
    x_contact = (tsd_t['X_in']+tsd_t['X_out'])/2
    y_contact = (tsd_t['Y_in']+tsd_t['Y_out'])/2
    
    zr_contact = (tsd_t['Zr_in']+tsd_t['Zr_out'])/2
    CF_mid = - zr_contact + DZo - DZb 
    # EF = ED + DI + IF; ED = - DZb; for I=I_mid & F=F_mid, DI = - zr; and IF = DZo
    # CF = (EinF + EoutF) / 2 [eq planes of the 2 beads]
    # so CF = -(zrIn+zrOut)/2 - DZb + DZo = -zr_contact + DZo - DZb 
    iz_mid = (nZ-1)//2 # If 11 points in Z, iz_mid = 5
    iz_contact = iz_mid - (CF_mid / dz)
    # print(iz_contact)
    
    x_in = tsd_t['X_in']
    y_in = tsd_t['Y_in']
    z_in = ((-1)*tsd_t['Zr_in']) + DZo - DZb
    
    x_out = tsd_t['X_out']
    y_out = tsd_t['Y_out']
    z_out = ((-1)*tsd_t['Zr_out']) + DZo - DZb
    
    dfContact.loc[t, ['iT', 'T', 'X', 'Y', 'iZ', 'Z']] = [t, Fluo_T[t], x_contact, y_contact, iz_contact, iz_contact*dz]
    dfContact.loc[t, ['Xin', 'Yin', 'Zin', 'Xout', 'Yout', 'Zout']] = [x_in, y_in, dz*iz_mid-z_in, x_out, y_out, dz*iz_mid-z_out]

    
    # Zr = (-1) * (tsd_t['Zr_in']+tsd_t['Zr_out'])/2 # Taking the opposite cause the depthograph is upside down for these experiments
    # izmid = (nZ-1)//2 # If 11 points in Z, izmid = 5
    # zz = dz * np.arange(-izmid, +izmid, 1) # If 11 points in Z, zz = [-5, -4, ..., 4, 5] * dz
    # zzf = zz + Zr # Coordinates zz of bf images with respect to Zf, the Z of the focal point of the deptho
    # zzeq = zz + Zr + DZo - DZb # Coordinates zz of fluo images with respect to Zeq, the Z of the equatorial plane of the bead
    
    # zi_contact = Akima1DInterpolator(zzeq, np.arange(len(zzeq)))(0)
    
    A_contact_map = 0
    foundContact = False
    
    for z in range(nZ):
        contactSlice = False
        i_zt = z + nZ*t
        I_zt = I[i_zt]
        
        #### 2.0 Locate cell
        if z >= 1:
            (Xc, Yc, Rc) = previousCircleXYR
        if z == 0:
            (Yc, Xc), Rc = findCellInnerCircle(I_zt, withBeads = True, plot=False)
            if Rc < 8*SCALE_100X_ZEN:
                print('done that')
                (Xc, Yc, Rc) = (I_zt.shape[1]//2, I_zt.shape[0]//2, 9.5*SCALE_100X_ZEN)
            
        # (Yc, Xc), Rc = findCellInnerCircle(I_zt, withBeads = True, plot=False)     
        Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
        
        
        #### 2.1 Warp
        warped = skm.transform.warp_polar(I_zt, center=(Yc, Xc), radius=warp_radius, #Rc*1.2, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        # max_values = np.max(warped[:,Rc0-inPix_set:Rc0+outPix_set+1], axis=1)
            
        #### 2.2 Viterbi Smoothing
        edge_viterbi = ViterbiEdge(warped, Rc0, inPix_set, outPix_set, 2, relative_height_virebi)
        edge_viterbi_unwarped = np.array(unwarpRA(np.array(edge_viterbi), Angles, Xc, Yc)) # X, Y

        #### Extra iteration
        # x.0 Locate cell
        (Xc, Yc), Rc = fitCircle(np.array([edge_viterbi_unwarped[1], edge_viterbi_unwarped[0]]).T, loss = 'huber')        
        Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
        
        # x.1 Warp
        warped = skm.transform.warp_polar(I_zt, center=(Yc, Xc), radius=warp_radius, #Rc*1.4, 
                                          output_shape=None, scaling='linear', channel_axis=None)
        warped = skm.util.img_as_uint(warped)
        w_ny, w_nx = warped.shape
        Angles = np.arange(0, 360, 1)
        max_values = np.max(warped[:,Rc0-inPix_set:Rc0+outPix_set+1], axis=1)
        inPix, outPix = inPix_set, outPix_set
        
        # x.3 Viterbi Smoothing
        edge_viterbi = ViterbiEdge(warped, Rc0, inPix, outPix, blur_parm, relative_height_virebi)
        edge_viterbi_unwarped = unwarpRA(np.array(edge_viterbi), Angles, Xc, Yc)
        arrayViterbiContours[t, z, :] = np.array([edge_viterbi_unwarped[0], edge_viterbi_unwarped[1]]).T
        
        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(I_zt, cmap='gray', aspect='equal')
        # ax.plot(edge_viterbi_unwarped[0], edge_viterbi_unwarped[1], c='g', ls = '--')
        # ax.plot(Xc, Yc, 'go')
        # CC = circleContour_V2((Yc, Xc), Rc)
        # [yy_circle, xx_circle] = CC.T
        # ax.plot(xx_circle, yy_circle, 'b--')
        # ax.set_title('Detected contours')
        # fig.tight_layout()
        # plt.show()



        #### 3.0 Locate cell
        (Xc, Yc), Rc = fitCircle(np.array([edge_viterbi_unwarped[1], edge_viterbi_unwarped[0]]).T, loss = 'huber')
        if z >= 1 and Rc < 7.5*SCALE_100X_ZEN:
            print('done this')
            (Xc, Yc, Rc) = previousCircleXYR
        if z == 0 and Rc < 7.5*SCALE_100X_ZEN:
            print('done that')
            (Xc, Yc, Rc) = (I_zt.shape[1]//2, I_zt.shape[0]//2, 9.5*SCALE_100X_ZEN)
        
        previousCircleXYR = (Xc, Yc, Rc)
        Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
        
        #### 3.1 Warp
        warped = skm.transform.warp_polar(I_zt, center=(Yc, Xc), radius=warp_radius, #Rc*1.4, 
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
            dfContact.loc[t, ['R', 'A']] = [R_contact, A_contact]
        
        # #### 3.2 Interp in X
        # warped_interp = ufun.resize_2Dinterp(warped, fx=interp_factor_x, fy=1)
        # warped = warped_interp
        # w_nx, Rc0 = w_nx * interp_factor_x, Rc0 * interp_factor_x
        # inPix, outPix = inPix_set * interp_factor_x, outPix_set * interp_factor_x
        # R_contact *= interp_factor_x
        # blur_parm *= interp_factor_x
        
        inPix, outPix = inPix_set, outPix_set
        
        #### 3.3 Viterbi Smoothing
        edge_viterbi = ViterbiEdge(warped, Rc0, inPix, outPix, blur_parm, relative_height_virebi)
        # edge_viterbi_unwarped = unwarpRA(np.array(edge_viterbi)/interp_factor_x, Angles, Xc, Yc)
        edge_viterbi_unwarped = unwarpRA(np.array(edge_viterbi), Angles, Xc, Yc)
        arrayViterbiContours[t, z, :] = np.array([edge_viterbi_unwarped[0], edge_viterbi_unwarped[1]]).T
        
        
        
        dfContact.loc[t, ['Xc', 'Yc', 'Rc']] += np.array([(Xc/SCALE_100X_ZEN)/(nZ), (Yc/SCALE_100X_ZEN)/(nZ), (Rc/SCALE_100X_ZEN)/(nZ)])
        
        #### 4. Append the contact profile matrix
        # profileMatrix = np.zeros((nT, nZ, N_profilesMatrix, warp_radius))
        halfN = N_profilesMatrix//2
        tmpWarp = np.copy(warped)
        A_contact_r = round(A_contact)
        dA = A_contact_r - A_contact
        if   A_contact_r < halfN+3:
            tmpWarp = np.roll(tmpWarp, halfN+3, axis=0) 
            tmp_A_contact   = A_contact   + (halfN+3)
            tmp_A_contact_r = A_contact_r + (halfN+3)
            print('roll forward')
            print(A_contact_r, tmp_A_contact_r)
        elif A_contact_r > 360 - (halfN+3):
            tmpWarp = np.roll(tmpWarp, -(halfN+3), axis=0) 
            tmp_A_contact   = A_contact   - (halfN+3)
            tmp_A_contact_r = A_contact_r - (halfN+3)
            print('roll backward')
            print(A_contact_r, tmp_A_contact_r)
        else:
            tmp_A_contact   = A_contact
            tmp_A_contact_r = A_contact_r
    
        
        try:
            smallTmpWarp = tmpWarp[tmp_A_contact_r - (halfN+3):tmp_A_contact_r + (halfN+3)+1, :]
            tform = skm.transform.EuclideanTransform(translation=(0, dA))
            smallTmpWarp_tformed = skm.util.img_as_uint(skm.transform.warp(smallTmpWarp, tform))
            
            Ltmp = len(smallTmpWarp_tformed)
            smallWarp = smallTmpWarp_tformed[Ltmp//2 - halfN:Ltmp//2 + halfN+1, :]
            
            profileMatrix[t, z] = smallWarp
            
        except:
            print("profileMatrix Error")
            print(A_contact_r)
            print(dA)
            print(tmpWarp.shape)
            print(smallTmpWarp.shape)
            print(smallTmpWarp_tformed.shape)
            print(smallWarp.shape)
            
        #### 6.1 Compute beads positions
        draw_InCircle = False
        draw_OutCircle = False
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
            # RINbeadContour *= interp_factor_x
            
        if h_OutCircle > 0 and h_OutCircle < 2*Rbead:
            draw_OutCircle = True
            a_OutCircle = (h_OutCircle*(2*Rbead-h_OutCircle))**0.5
            # OUTbeadContour = circleContour((round(y_out*SCALE_100X_ZEN), round(x_out*SCALE_100X_ZEN)), 
            #                                a_OutCircle*SCALE_100X_ZEN, I_zt.shape)
            OUTbeadContour = circleContour_V2((round(y_out*SCALE_100X_ZEN), round(x_out*SCALE_100X_ZEN)),
                                              a_OutCircle*SCALE_100X_ZEN)
            XOUTbeadContour, YOUTbeadContour = OUTbeadContour[:, 1], OUTbeadContour[:, 0]
            ROUTbeadContour, AOUTbeadContour = warpXY(XOUTbeadContour, YOUTbeadContour, Xc, Yc)
            # ROUTbeadContour *= interp_factor_x
            
        
        #### 5. Analyse fluo cyto & background
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

        mask_removed_from_cyto = inside & (~mask_cyto)
        cell_filtered = cell_dilated & (~mask_removed_from_cyto)
        
        nb_pix = np.sum(cell_filtered.astype(int))
        total_val = np.sum(I_zt.flatten()[cell_filtered.flatten()])
        I_cell = total_val/nb_pix
        
        # Bead
        I_bead = np.nan
        if draw_InCircle:
            xx, yy = np.meshgrid(np.arange(I_zt.shape[1]), np.arange(I_zt.shape[0]))
            xc, yc = round(x_in*SCALE_100X_ZEN), round(y_in*SCALE_100X_ZEN)
            rc = a_InCircle*SCALE_100X_ZEN*0.9
            maskBead = ((xx-xc)**2 + (yy-yc)**2)**0.5 < rc
            I_bead = np.median(I_zt[maskBead])
        
        
        # Store
        fluoCyto[t, z] = I_cyto
        fluoCytoStd[t, z] = Std_cyto
        fluoBack[t, z] = I_back
        fluoCell[t, z] = I_cell
        fluoBeadIn[t, z] = I_bead
        
        #### 5. Define normalization functions
        def normalize_Icyto(x):
            return((x - I_back)/(I_cyto - I_back))
    
        def normalize_Icell(x):
            return((x - I_back)/(I_cell - I_back))
        
        def normalize_Icell2(x):
            return((x - I_cyto)/(I_cell - I_back))
        
        
        
        #### 5. Read values
        VEW=5
        max_values = normalize_Icell(max_values)
        viterbi_Npix_values = [np.sum(warped[Angles[i], edge_viterbi[i]-VEW//2:edge_viterbi[i] + (VEW//2 + 1)])/VEW for i in range(len(edge_viterbi))]
        viterbi_Npix_values = normalize_Icell(viterbi_Npix_values)
        intensityMap_t.append(viterbi_Npix_values)