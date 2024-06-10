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
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath

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



# %% Complete pipeline: tracking bf -> fluo

# %%% Paths

date = '24-05-24'
date2 = dateFormat(date)

cell = 'C6'
cellId = f'{date}_{cell}'
specif = '' # '_off4um'
specifFolder = 'M2-20um/B5mT/' # 'M2-20um/Bstairs/' # 'M2-20um/B5mT/'

srcDir = f'D:/MagneticPincherData/Raw/{date2}_Chameleon/{specifFolder}Clean_Fluo_BF'

bfName = f'3T3-LifeActGFP_PincherFluo_{cell}{specif}_BF.tif'
bfPath = os.path.join(srcDir, bfName)
resBfDir = os.path.join(srcDir, 'Results_Tracker')
resBfPath = os.path.join(resBfDir, bfName[:-4] + '_PY.csv')

dictPaths = {'sourceDirPath' : srcDir,
             'imageFileName' : bfName,
             'resultsFileName' : bfName[:-4] + '_Results.txt',
             'depthoDir' : 'D:/MagneticPincherData/Raw/DepthoLibrary',
             'depthoName' : f'{date2}_Chameleon_M2_M450_step20_100X',
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

framesize = int(35*8/2) 
inPix_set, outPix_set = 20, 20
blur_parm = 2
relative_height_virebi = 0.1

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

SAVE_RESULTS = True
PLOT_WARP = True
PLOT_MAP = True
PLOT_NORMAL = True


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
        if z >= 1 and Rc < 6.5*SCALE_100X_ZEN:
            print('done this')
            # (Xc, Yc, Rc) = previousCircleXYR
            (Xc, Yc, Rc) = (I_zt.shape[1]//2, I_zt.shape[0]//2, 9.5*SCALE_100X_ZEN)
        if z == 0 and Rc < 6.5*SCALE_100X_ZEN:
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
            ax.plot(edge_viterbi_unwarped[0], edge_viterbi_unwarped[1], c='g', ls = '--')
            ax.plot(Xc, Yc, 'go')
            CC = circleContour_V2((Yc, Xc), Rc)
            [yy_circle, xx_circle] = CC.T
            ax.plot(xx_circle, yy_circle, 'b--')
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

# %%%% Open profiles and analyze

# %%%%% Load files

# date = '24-02-27'
date = '24-05-24'

date2 = dateFormat(date)

cell = 'C6'
cellId = f'{date}_{cell}'
specif = '' #'_off4um'
specifFolder = 'M2-20um/B5mT/' # 'M2-20um/Bstairs/' # 'M2-20um/B5mT/'

srcDir = f'D:/MagneticPincherData/Raw/{date2}_Chameleon/{specifFolder}Clean_Fluo_BF'
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


# %%%%% Check it out

# def normalize_Icyto(x):
#     return((x - Iback)/(Icyto - Iback))

# def normalize_Icell(x):
#     return((x - Iback)/(Icell - Iback))

# def normalize_Icell2(x):
#     return((x - Icyto)/(Icell - Iback))

# for t in range(nT):
#     actinQuantity(1, profileMatrix, dfContact,
#                   fluoCell, fluoCyto, fluoBack)


def actinQuantity(t, profileMatrix, dfContact,
                  fluoCell, fluoCyto, fluoBack, fluoBeadIn):
    pM = profileMatrix[t]
    approxR = round(dfContact.loc[dfContact['iT']==t, 'R'].values[0])
    iz = dfContact.loc[dfContact['iT']==t, 'iZ'].values[0]
    izr = round(iz)
    # Icell, Icyto, Iback = fluoCell[t, izr], fluoCyto[t, izr], fluoBack[t, izr]
    
    list_Q_fb = []
    list_Q_vb = []
    list_Q_gf = []
    
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
                eR = (approxR - 5) + np.argmax(profile[approxR-5:approxR+10])
                Np = len(profile)
                
                Icell, Icyto, Iback = fluoCell[t, izr + iiz], fluoCyto[t, izr + iiz], fluoBack[t, izr + iiz]
                Ibeadin = fluoBeadIn[t, izr + iiz]
                
                
                def normalize_Icell(x):
                    return((x - Iback)/(Icell - Iback))
                profile_n = normalize_Icell(profile)
    
                eV = np.max(profile_n[approxR-5:approxR+5])
                
                bead_in_level = normalize_Icell(Ibeadin) * 0.8 + eV * 0.2
                bead_out_level = normalize_Icell(Ibeadin)
                ri = eR - ufun.findFirst(True, profile_n[:eR][::-1]<bead_in_level)
                rf = eR + ufun.findFirst(True, profile_n[eR:]<bead_out_level)
                
                # if eR-ri <= 5:
                #     bead_in_level = normalize_Icell(Ibeadin)
                #     ri = eR - ufun.findFirst(True, profile_n[:eR][::-1]<bead_in_level)
                if rf-eR <= 3:
                    bead_out_level = 0.2*eV + 0.8*np.median(profile_n[-5:])
                    rf = eR + ufun.findFirst(True, profile_n[eR:]<bead_out_level)
                # print(eR, ri, rf)
                
                #### Calculation 1
                n_in, n_out = 10, 5
                Q_fb = np.sum(profile_n[eR-n_in:eR+n_out+1])
                list_Q_fb.append(Q_fb)
                
                #### Calculation 2
                Q_vb = np.sum(profile_n[ri:rf+1])
                list_Q_vb.append(Q_vb)
    
                #### Calculation 3
                ### bead_in_level = 0.9
                ### bead_out_level = 0.95
                ###(2*bead_in_level+eV)/3)
                def Gfit(x, m, Q, s):
                    # m = eR
                    return((Q/(s*(2*np.pi)**0.5)) * np.exp(-(x-m)**2/(2*s**2)))
                
                try:                
                    popt, pcov = optimize.curve_fit(Gfit, np.arange(ri, rf), profile_n[ri:rf], p0=[eR, 20, 5],
                                                    bounds = ([ri, 0, 0], [rf, 2*Q_vb, 4*(rf-ri)]))
                    m_gf, Q_gf, s_gf = popt
                    xfit = np.linspace(eR-20, eR+20, 200)
                    yfit = Gfit(xfit, m_gf, Q_gf, s_gf)
                    list_Q_gf.append(Q_gf)
                    fitError = False
                except:
                    fitError = True
                
                ax = axes[nZ_scan-z-1, a]            
                ax.plot(np.arange(Np), profile_n)
                ax.axvline(eR, c='r', ls='--')
                # 1
                ax.axvline(eR-n_in, c='cyan', ls='--', lw=1)
                ax.axvline(eR+n_out+1, c='cyan', ls='--', lw=1, label=f'Fixed boundaries: Q={Q_fb:.1f}')
                # 2
                ax.axhline(bead_in_level, c='green', ls='-.', lw=1)
                ax.axhline(bead_out_level, c='green', ls='-.', lw=1)
                # ax.axhline(normalize_Icell(Ibeadin), c='orange', ls='-.', lw=0.8)
                ax.axvline(ri, c='orange', ls='--', lw=1)
                ax.axvline(rf, c='orange', ls='--', lw=1, label=f'Variable boundaries: Q={Q_vb:.1f}') #, label='Gaussian boundaries')
                # 3
                if not fitError:
                    ax.plot(xfit, yfit, c='gold', label=f'Gaussian fit: Q={Q_gf:.1f}')
                
                ax.legend(fontsize=8, loc='lower left')
                
                if a==0:
                    ax.set_ylabel(f'z={izr + iiz:.0f}; zr={iiz:.0f} (steps)')
                if z==nZ_scan-1:
                    ax.set_title(f'a = {iia:.0f} (°)')
    
    fig.suptitle(f'{cellId} - t={t+1:.0f}')
    fig.tight_layout()
    ufun.simpleSaveFig(fig, f'ProfileMetrics_t{t+1:.0f}', figSavePath, '.png', 150)
    plt.show()
            
    avg_Q_fb = np.mean(list_Q_fb)
    avg_Q_vb = np.mean(list_Q_vb)
    avg_Q_gf = np.mean(list_Q_gf)
    # print(f'Average score: Q_fb = {avg_Q_fb:.1f}')
    # print(f'Average score: Q_vb = {avg_Q_vb:.1f}')
    # print(f'Average score: Q_gf = {avg_Q_gf:.1f}')
    
    dfContact.loc[dfContact['iT']==t, 'Q_fb'] = avg_Q_fb
    dfContact.loc[dfContact['iT']==t, 'Q_vb'] = avg_Q_vb
    dfContact.loc[dfContact['iT']==t, 'Q_gf'] = avg_Q_gf
    dfContact.loc[dfContact['iT']==t, 'nA_scan'] = nA_scan
    dfContact.loc[dfContact['iT']==t, 'nZ_scan'] = nZ_scanCount
    
    # # Compare the metrics
    # fig2, axes2 = plt.subplots(1, 3, figsize=(12, 5))
    # ax=axes2[0]
    # ax.plot(list_Q_fb, list_Q_vb, 'bo')
    # ax.set_xlabel('Fixed b.')
    # ax.set_ylabel('Variable b.')
    # ax=axes2[1]
    # ax.plot(list_Q_fb, list_Q_gf, 'go')
    # ax.set_xlabel('Fixed b.')
    # ax.set_ylabel('Gaussian f.')
    # ax=axes2[2]
    # ax.plot(list_Q_vb, list_Q_gf, 'ro')
    # ax.set_xlabel('Variable b.')
    # ax.set_ylabel('Gaussian f.')
    # fig2.tight_layout()
    # plt.show()
    
    return(dfContact)


for t in range(nT):
    dfContact = actinQuantity(t, profileMatrix, dfContact,
                              fluoCell, fluoCyto, fluoBack, fluoBeadIn)


dfContact.to_csv(os.path.join(dstDir, f'{cellId}_dfContact.csv'), sep=';', index=False)


































# %%%% Compute map
VEW = 5 # Viterbi edge width # Odd number
VE_in, VE_out = 7, 3 # Viterbi edge in/out

#### Start

plt.ioff()
I = skm.io.imread(fluoPath)

czt_shape, czt_seq = get_CZT_fromTiff(fluoPath)
[nC, nZ, nT] = czt_shape
Angles = np.arange(0, 360, 1)

tsdf = pd.read_csv(resBfPath, sep=None, engine='python')

colsContact = ['iT', 'T', 'X', 'Y', 'Z', 'R', 'A', 'Xc', 'Yc', 'Rc', 
               'Xin', 'Yin', 'Zin', 'Xout', 'Yout', 'Zout', 
               f'I_viterbi_w{VEW:.0f}', f'I_viterbi_inout{VE_in:.0f}-{VE_out:.0f}', 'I_viterbi_flexW', 'W_viterbi_flexW']

dfContact = pd.DataFrame(np.zeros((nT, len(colsContact))), columns=colsContact)
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
    
    dfContact.loc[t, ['iT', 'T', 'X', 'Y', 'Z']] = [t, Fluo_T[t], x_contact, y_contact, iz_contact*dz]
    dfContact.loc[t, ['Xin', 'Yin', 'Zin', 'Xout', 'Yout', 'Zout']] = [x_in, y_in, dz*iz_mid-z_in, x_out, y_out, dz*iz_mid-z_out]

    
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


        #### 3.0 Locate cell
        (Xc, Yc), Rc = fitCircle(np.array([edge_viterbi_unwarped[1], edge_viterbi_unwarped[0]]).T, loss = 'huber')
        Yc0, Xc0, Rc0 = round(Yc), round(Xc), round(Rc)
        dfContact.loc[t, ['Xc', 'Yc', 'Rc']] += np.array([(Xc/SCALE_100X_ZEN)/(nZ), (Yc/SCALE_100X_ZEN)/(nZ), (Rc/SCALE_100X_ZEN)/(nZ)])
        
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
            dfContact.loc[t, ['R', 'A']] = [R_contact, A_contact]
        
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
        # figN, axesN = plt.subplots(1,4, figsize = (16, 5))
        # ax = axesN[0]
        # ax.imshow(I_zt)
        # ax.set_title('Original')
        # ax = axesN[1]
        # ax.imshow((mask_cyto & inside_eroded).astype(int))
        # ax.set_title('Cytoplasm')
        # ax = axesN[2]
        # ax.imshow(cell_filtered.astype(int))
        # ax.set_title('Whole cell')
        # ax = axesN[3]
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
        # figN.tight_layout()
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
            dfContact.loc[t, f'I_viterbi_w{VEW:.0f}'] = viterbi_Npix_values[iA_contact]
            
            viterbi_inout_value = np.sum(warped[iA_contact-1:iA_contact+2, 
                                                edge_viterbi[iA_contact]-VE_in*interp_factor_x:edge_viterbi[iA_contact]+VE_out*interp_factor_x])/(3*(VE_in+VE_out)*interp_factor_x)
            viterbi_inout_value = normalize_Icell(viterbi_inout_value)
            dfContact.loc[t, f'I_viterbi_inout{VE_in:.0f}-{VE_out:.0f}'] = viterbi_inout_value
            
            
            
            
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
            
            dfContact.loc[t, 'I_viterbi_flexW'] = viterbi_flex_value         
            dfContact.loc[t, 'W_viterbi_flexW'] = r_final-r_init  
                
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
            ax.set_title(f'Detected contours - Rc={Rc:.0f}')
            
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
    dfContact.to_csv(os.path.join(dstDir, f'{cellId}_dfContact.csv'), sep=';', index=False)
    np.save(os.path.join(dstDir, f'{cellId}_xyContours.npy'), arrayViterbiContours)
    np.savetxt(os.path.join(dstDir, f'{cellId}_fluoCyto.txt'), fluoCyto, fmt='%.1f')
    np.savetxt(os.path.join(dstDir, f'{cellId}_fluoCytoStd.txt'), fluoCytoStd, fmt='%.3f')
    np.savetxt(os.path.join(dstDir, f'{cellId}_fluoBack.txt'), fluoBack, fmt='%.1f')
    np.savetxt(os.path.join(dstDir, f'{cellId}_fluoCell.txt'), fluoCell, fmt='%.1f')
    