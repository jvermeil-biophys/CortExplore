# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:30:55 2022
@author: Joseph Vermeil

BeadsCalibration.py - 
Joseph Vermeil, 2022

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# %% (0) Imports and settings

# 1. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import time
# import cv2

# import scipy
from scipy import interpolate

# import skimage
from skimage import io
from scipy.optimize import curve_fit


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
gs.set_default_options_jv()

# %% (1) Functions

# %%% (1.1) Simple chain analysis

def mainChainAnalysis(mainPath, depthoPath, approxDiameter):
    """
    

    Parameters
    ----------
    mainPath : TYPE
        DESCRIPTION.
    depthoPath : TYPE
        DESCRIPTION.
    approxDiameter : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    start = time.time()
    
    #### 0. Load
    imagesFiles = []
    imagesFiles_Paths = []
    fileList = os.listdir(mainPath)
    for f in fileList:
        fPath = os.path.join(mainPath, f)
        if f.endswith('.tif') and os.path.isfile(fPath[:-4] + '_Results.txt'):
            imagesFiles.append(f)
            imagesFiles_Paths.append(os.path.join(mainPath, f))
            
    distanceDict = {'dz':[], 'D2':[], 'D3':[]}
    XYZdict = {'X' : np.array([]), 'Y' : np.array([]), 'Z' : np.array([]), 'Z_quality' : np.array([])}
    
    #### 1 - Import depthograph
    HDZfactor = 10
    deptho = io.imread(depthoPath+'_Deptho.tif')
    depthoMetadata = pd.read_csv(depthoPath+'_Metadata.csv', sep=';')
    depthoStep = depthoMetadata.loc[0,'step']
    depthoZFocus = depthoMetadata.loc[0,'focus']
    
    # increase the resolution of the deptho with interpolation
    # print('deptho shape check')
    # print(deptho.shape)
    nX, nZ = deptho.shape[1], deptho.shape[0]
    X, Z = np.arange(0, nX, 1), np.arange(0, nZ, 1)
    Z_HD = np.arange(0, nZ, 1/HDZfactor)
    # print(XX.shape, ZZ.shape, deptho.shape, ZZ_HD.shape)
    
    # fd = interpolate.interp2d(XX, ZZ, deptho, kind='cubic')
    # depthoHD = fd(XX, ZZ_HD)
    
    fd = interpolate.RectBivariateSpline(Z, X, deptho)
    ZZ_HD, XX = np.meshgrid(Z_HD, X, indexing='ij')
    depthoHD = fd(ZZ_HD, XX, grid=False)
    
    depthoStepHD = depthoStep/HDZfactor
    depthoZFocusHD = depthoZFocus*HDZfactor
    # print(depthoHD.shape)

    #### 2 - Begining of the Main Loop
    for i in range(len(imagesFiles)):
        f, fP = imagesFiles[i], imagesFiles_Paths[i]
        print(f, fP)

        I = io.imread(fP) # Approx 0.5s per image
 
        resFilePath = fP[:-4] + '_Results.txt'
        # fieldCols = ['Area', 'StdDev', 'XM', 'YM', 'Slice']
        resDf = pd.read_csv(resFilePath, sep = '\t') #, names = fieldCols) # '\t'

        resDf = resDf.sort_values(by=['Slice', 'XM'])  
        print(resDf)

        #### 3 - Compute XYZ
        
        X, Y, Z, Z_quality = oneChainXYZ(I, resDf, depthoHD, depthoZFocusHD, 
                                         depthoStepHD, approxDiameter)
        
        X = X / gc.SCALE_100X
        Y = Y / gc.SCALE_100X
        Z = Z * depthoStepHD/1000
        
        # All 3 coordinates are now in microns
        
        XYZdict['X'] = np.concatenate((XYZdict['X'], X))
        XYZdict['Y'] = np.concatenate((XYZdict['Y'], Y))
        XYZdict['Z'] = np.concatenate((XYZdict['Z'], Z))
        XYZdict['Z_quality'] = np.concatenate((XYZdict['Z_quality'], Z_quality))
        
        #### 4 - Compute distances
        dz_list, D2_list, D3_list = [], [], []
        nB = len(X)
        for ii in range(nB-1):
            dz = (Z[ii+1] - Z[ii])
            D2 = (((X[ii+1] - X[ii])**2 + (Y[ii+1] - Y[ii])**2)**0.5)
            D3 = (D2**2 + dz**2)**0.5
            dz_list.append(dz)
            D2_list.append(D2)
            D3_list.append(D3)
        
        # dz_array, D2_array, D3_array = np.array(dz_list), np.array(D2_list), np.array(D3_list)
        # distanceDict = {'dz':dz_array, 'D2':D2_array, 'D3':D3_array}
        
        distanceDict['dz'] += dz_list
        distanceDict['D2'] += D2_list
        distanceDict['D3'] += D3_list
        
    #### After the loop
    
    distanceDf = pd.DataFrame(distanceDict)
    xyzDf = pd.DataFrame(XYZdict)
    
    #### REMOVING SOME OUTLIERS
    # 1st mode
    # out = np.percentile(distanceDf.D3.values, 97)
    # distanceDf = distanceDf.drop(index = distanceDf.loc[distanceDf.D3 >= out].index)
    
    # 2nd mode
    outUP = np.percentile(distanceDf.D3.values, 99)
    outDOWN = np.percentile(distanceDf.D3.values, 1)
    distanceDf = distanceDf.drop(index = distanceDf.loc[distanceDf.D3 >= outUP].index)
    distanceDf = distanceDf.drop(index = distanceDf.loc[distanceDf.D3 <= outDOWN].index)
    
    
    # percentile list
    perc =[.10, .25, .50, .75, .90]
      
    # list of dtypes to include
    # include =['object', 'float', 'int']
      
    # calling describe method
    statsDf = distanceDf.describe(percentiles = perc)
    # statsDf = distanceDf.describe()

    #### PLOT
    fig, ax = plt.subplots(1,2, figsize = (12,8))
    sns.violinplot(y="D3", data=distanceDf, ax=ax[0], inner="quartile")
    sns.swarmplot(y="D3", data=distanceDf, ax=ax[1])
    plt.show()

    return(xyzDf, distanceDf, statsDf)


def oneChainXYZ(I, resDf, deptho, Zfocus, depthoStep, D):
    """
    Standard
    """
    X, Y = resDf['XM'].values, resDf['YM'].values
    nB = len(X)
    Z, Z_quality = [], []
    
    ImgZStep = 0

    for i in range(nB):
        x, y = X[i], Y[i]
        
        #### PLOT
        plot = 0
        z, z_quality = computeZ_V2(I, x, y, deptho, Zfocus, 
                                   depthoStep, ImgZStep, D, plot = plot)
        Z.append(z)
        Z_quality.append(z_quality)
    
    Z, Z_quality = np.array(Z), np.array(Z_quality)

    return(X, Y, Z, Z_quality)




# %%% (1.2) Stack chain analysis

def mainChainNupAnalysis(mainPath, depthoPath, approxDiameter):
    start = time.time()
    
    #### 0. Load
    imagesFiles = []
    imagesFiles_Paths = []
    fileList = os.listdir(mainPath)
    for f in fileList:
        fPath = os.path.join(mainPath, f)
        if f.endswith('.tif') and os.path.isfile(fPath[:-4] + '_Results.txt'):
            imagesFiles.append(f)
            imagesFiles_Paths.append(os.path.join(mainPath, f))
            
    distanceDict = {'dz':[], 'D2':[], 'D3':[]}
    XYZdict = {'X' : np.array([]), 'Y' : np.array([]), 'Z' : np.array([]), 'Z_quality' : np.array([])}
    
    #### 1 - Import depthograph
    HDZfactor = 10
    deptho = io.imread(depthoPath+'_Deptho.tif')
    depthoMetadata = pd.read_csv(depthoPath+'_Metadata.csv', sep=';')
    depthoStep = depthoMetadata.loc[0,'step']
    depthoZFocus = depthoMetadata.loc[0,'focus']
    
    # increase the resolution of the deptho with interpolation
    # print('deptho shape check')
    # print(deptho.shape)
    nX, nZ = deptho.shape[1], deptho.shape[0]
    X, Z = np.arange(0, nX, 1), np.arange(0, nZ, 1)
    Z_HD = np.arange(0, nZ, 1/HDZfactor)
    # print(XX.shape, ZZ.shape, deptho.shape, ZZ_HD.shape)
    
    # fd = interpolate.interp2d(XX, ZZ, deptho, kind='cubic')
    # depthoHD = fd(XX, ZZ_HD)
    fd = interpolate.RectBivariateSpline(Z, X, deptho)
    ZZ_HD, XX = np.meshgrid(Z_HD, X, indexing='ij')
    depthoHD = fd(ZZ_HD, XX, grid=False)
    
    depthoStepHD = depthoStep/HDZfactor
    depthoZFocusHD = depthoZFocus*HDZfactor
    # print(depthoHD.shape)

    #### 2 - Begining of the Main Loop
    for i in range(len(imagesFiles)):
        f, fP = imagesFiles[i], imagesFiles_Paths[i]
        print(f, fP)

        I = io.imread(fP) # Approx 0.5s per image
 
        resFilePath = fP[:-4] + '_Results.txt'
        # fieldCols = ['Area', 'StdDev', 'XM', 'YM', 'Slice']
        resDf = pd.read_csv(resFilePath, sep = '\t') #, names = fieldCols) # '\t'

        resDf = resDf.sort_values(by=['Slice', 'XM'])
        print(resDf)
        
        
        #### OPTIONS OF THE Z TRACKING HERE !
        # Get the nUplets
        nz, ny, nx = I.shape[0], I.shape[1], I.shape[2]
        dZ_nUplets = 0.5 # µm # Distance between slices of the Nuplet
        dZ_stack = 0.02 # µm # Distance between slices of the deptho
        dN_nUplets = int(dZ_nUplets/dZ_stack)
        
        z_max = 200
        max_I = 0
        for z in range(nz):
            if np.max(I[z]) > max_I:
                max_I = np.max(I[z])
                z_max = z
        
        center_Zoffset = [ii for ii in range (-50, 55, 25)]
        numberOfNupletPerChain = 5
        aaa = numberOfNupletPerChain//2
        
        
        # center_Zoffset = [0]
        # nUpletsIndices = [[z_max + center_Zoffset[ii] - dN_nUplets,
        #                     z_max + center_Zoffset[ii],
        #                     z_max + center_Zoffset[ii] + dN_nUplets] \
        #                    for ii in range(len(center_Zoffset))]
            
        nUpletsIndices = [[z_max + center_Zoffset[ii] + kk*dN_nUplets \
                            for kk in range(-aaa, aaa+1)] \
                            for ii in range(len(center_Zoffset))]
            
        nUpletsIndices = [[z_max]]
            
        nUpletsIndices = np.array(nUpletsIndices)
        

        #### 3 - Compute XYZ
        
        for nUplet in nUpletsIndices:
        
            X, Y, Z, Z_quality = oneChainNupXYZ(I, resDf, nUplet, depthoHD, depthoZFocusHD, 
                                     depthoStepHD, approxDiameter, ImgZStep = 1000*dZ_nUplets)
            
            X = X / gc.SCALE_100X
            Y = Y / gc.SCALE_100X
            Z = Z * depthoStepHD/1000
            
            # All 3 coordinates are now in microns
            
            XYZdict['X'] = np.concatenate((XYZdict['X'], X))
            XYZdict['Y'] = np.concatenate((XYZdict['Y'], Y))
            XYZdict['Z'] = np.concatenate((XYZdict['Z'], Z))
            XYZdict['Z_quality'] = np.concatenate((XYZdict['Z_quality'], Z_quality))
            
            #### 4 - Compute distances
            dz_list, D2_list, D3_list = [], [], []
            nB = len(X)
            for ii in range(nB-1):
                dz = (Z[ii+1] - Z[ii])
                D2 = (((X[ii+1] - X[ii])**2 + (Y[ii+1] - Y[ii])**2)**0.5)
                D3 = (D2**2 + dz**2)**0.5
                dz_list.append(dz)
                D2_list.append(D2)
                D3_list.append(D3)
        
            # dz_array, D2_array, D3_array = np.array(dz_list), np.array(D2_list), np.array(D3_list)
            # distanceDict = {'dz':dz_array, 'D2':D2_array, 'D3':D3_array}
            
            distanceDict['dz'] += dz_list
            distanceDict['D2'] += D2_list
            distanceDict['D3'] += D3_list
        
    #### After the loop
   
    distanceDf = pd.DataFrame(distanceDict)
    xyzDf = pd.DataFrame(XYZdict)
    out_up = np.percentile(distanceDf.D3.values, 97)
    out_down = np.percentile(distanceDf.D3.values, 3)
    distanceDf = distanceDf.drop(index = distanceDf.loc[distanceDf.D3 >= out_up].index)
    distanceDf = distanceDf.drop(index = distanceDf.loc[distanceDf.D3 <= out_down].index)
    
    # percentile list
    perc =[.10, .25, .50, .75, .90]
      
    # list of dtypes to include
    # include =['object', 'float', 'int']
      
    # calling describe method
    statsDf = distanceDf.describe(percentiles = perc)
    # statsDf = distanceDf.describe()

    #### PLOT
    fig, ax = plt.subplots(1,2, figsize = (12,8))
    sns.violinplot(y="D3", data=distanceDf, ax=ax[0], inner="quartile")
    sns.swarmplot(y="D3", data=distanceDf, ax=ax[1], size = 3)
    plt.show()

    return(xyzDf, distanceDf, statsDf)


def oneChainNupXYZ(I, resDf, slices, deptho, Zfocus, depthoStep, D, ImgZStep = 500):
    Nup = len(slices)
    
    sumStd = resDf.loc[resDf['Slice'].apply(lambda x : x in slices+1)].groupby('Slice').agg({'StdDev' : 'sum'})
    sliceMaxStd = sumStd.index[np.argmax(sumStd.StdDev.values)]
    
    best_X = resDf.loc[resDf['Slice'] == sliceMaxStd]['XM'].values
    best_Y = resDf.loc[resDf['Slice'] == sliceMaxStd]['YM'].values
    
    X, Y = [], []
    for k in range(Nup):
        X.append(resDf.loc[resDf['Slice'] == slices[k]+1]['XM'].values)
        Y.append(resDf.loc[resDf['Slice'] == slices[k]+1]['YM'].values)
        
    X, Y = np.array(X), np.array(Y)
    
    nB = len(best_X)
    Nimages = int(ImgZStep/depthoStep)
    Z, Z_quality = [], []
    I_Nup = I[slices]    

    for i in range(nB):
        Xb, Yb = X[:,i], Y[:,i]
        
        #### PLOT
        plot = 0
        z, z_quality = computeZ_V2(I_Nup, Xb, Yb, deptho, Zfocus, 
                                   depthoStep, ImgZStep, D, plot = plot)
        Z.append(z)
        Z_quality.append(z_quality)
    
    Z, Z_quality = np.array(Z), np.array(Z_quality)

    return(best_X, best_Y, Z, Z_quality)
    
    
    
# %%% (1.3) Z computation    
    
    
def computeZ_V2(I, X, Y, deptho, Zfocus, depthoStep, ImgZStep, D, plot = 0):
    #### Initialize
    
    # maxDz = 40
    scale = 15.8
    matchingDirection = 'downward'
    Ddz, Ddx = deptho.shape[0], deptho.shape[1]
    
    if len(I.shape) == 2:
        Nup = 1
        I = np.array([I])
    elif len(I.shape) == 3:
        Nup = I.shape[0]
    
    if len(np.array(X).shape) == 0:
        X, Y = [X], [Y]
    
    Nframes = Nup

    listXY = np.array([X, Y]).T

    cleanSize = ufun.getDepthoCleanSize(D, scale)
    hdSize = deptho.shape[1]
    depthoDepth = deptho.shape[0]
    listProfiles = np.zeros((Nframes, hdSize))
    listROI = []
    
    #### Loop on the Nuplet
    for i in range(Nframes):
        X = np.arange(0, 5)
        Y = np.arange(0, cleanSize)
        x0, y0 = int(np.round(listXY[i][0])), int(np.round(listXY[i][1]))
        # try:
            
        # Old behaviour
        # # > We could also try to recenter the image to keep a subpixel resolution here
        # profileROI = I[i, y-cleanSize//2:y+cleanSize//2+1, x-2:x+3]
        # # > line that is 5 pixels wide
        # f = interpolate.interp2d(xx, yy, profileROI, kind='cubic')
        # # Now use the obtained interpolation function and plot the result:
        # xxnew = xx
        # yynew = np.linspace(0, cleanSize, hdSize)
        # profileROI_hd = f(xxnew, yynew)
        
        # New behaviour
        profileROI = I[i, y0-cleanSize//2:y0+cleanSize//2+1, x0-2:x0+3]
        # > line that is 5 pixels wide
        Y_HD = np.linspace(0, cleanSize, hdSize)
        fd = interpolate.RectBivariateSpline(Y, X, profileROI)
        YY_HD, XX = np.meshgrid(Y_HD, X, indexing='ij')
        profileROI_hd = fd(YY_HD, XX, grid=False)
        
        # except: # If the vertical slice doesn't work, try the horizontal one
        #     print(ORANGE + 'error with the vertical slice -> trying with horizontal one')
        #     print('Roi')
        #     print(y-2, y+3, x-cleanSize//2, x+cleanSize//2+1)
        #     print('' + NORMAL)
            
        #     xx, yy = yy, xx
        #     X, Y = int(np.round(listXY[i][0])), int(np.round(listXY[i][1])) # > We could also try to recenter the image to keep a subpixel resolution here
        #     # line that is 5 pixels wide
        #     profileROI = I[i, y-2:y+3, x-cleanSize//2:x+cleanSize//2+1]
        #     f = interpolate.interp2d(xx, yy, profileROI, kind='cubic')
        #     # Now use the obtained interpolation function and plot the result:
        #     xxnew = np.linspace(0, cleanSize, hdSize)
        #     yynew = yy
        #     profileROI_hd = f(xxnew, yynew).T

        listROI.append(profileROI)

        listProfiles[i,:] = profileROI_hd[:,5//2] * (1/5)
        for j in range(1, 1 + 5//2):
            listProfiles[i,:] += profileROI_hd[:,5//2-j] * (1/5)
            listProfiles[i,:] += profileROI_hd[:,5//2+j] * (1/5)       

    listProfiles = listProfiles.astype(np.uint16)

    nVoxels = int(np.round(ImgZStep/depthoStep))
    
    listDistances = np.zeros((Nframes, depthoDepth))
    listZ = np.zeros(Nframes, dtype = int)
    for i in range(Nframes):
        listDistances[i] = ufun.squareDistance(deptho, listProfiles[i], normalize = True) # Utility functions       
        listZ[i] = np.argmin(listDistances[i])
        
    # Translate the profiles that must be translated (status_frame 1 & 3 if Nup = 3)
    # and don't move the others (status_frame 2 if Nup = 3 or the 1 profile when Nup = 1)
    if Nup > 1:
        finalDists = ufun.matchDists(listDistances, [i for i in range(1, Nup+1)], Nup, 
                                nVoxels, direction = matchingDirection)
    elif Nup == 1:
        finalDists = listDistances
        
    sumFinalD = np.sum(finalDists, axis = 0)
    
    z1 = np.argmin(sumFinalD)
    z2 = len(sumFinalD) - np.argmin(sumFinalD[::-1])
    z = int((z1 + z2)//2)
    zr = Zfocus - z
    
    #### Z quality computation
    
    #### First try with der2
    
    # dz_spline = int(depthoDepth/8) # 25 voxels for a 400 voxels deep deptho
    # # then scale with the HDfactor
       
    
    # sumFinalD_fitSpline = sumFinalD[z-dz_spline:z+dz_spline]

    # X_fitSpline = np.arange(z-dz_spline,z+dz_spline)
    # fitSpline = interpolate.splrep(x = X_fitSpline,
    #                                y = sumFinalD_fitSpline,
    #                                k = 3)
    
    # Y_fitSpline = interpolate.splev(X_fitSpline, fitSpline)
    
    # der100 = interpolate.splev(x = X_fitSpline,
    #                           tck = fitSpline,
    #                           der=1)
    
    # der10 = interpolate.splrep(x = X_fitSpline,
    #                            y = der100,
    #                            k = 1)
    
    # der1 = interpolate.splev(x = X_fitSpline,
    #                          tck = der10,
    #                          der=0)
    
    # der2 = interpolate.splev(x = X_fitSpline,
    #                          tck = der10,
    #                          der=1)
    
    # listX_fitSpline = []
    # listY_fitSpline = []
    # listDer1 = []
    # listDer2 = []
    # for i in range(Nframes):
    #     z_i = np.argmin(finalDists[i])
    #     finalDists_fitSpline_i = finalDists[i][z_i - dz_spline : z_i + dz_spline]
    #     X_fitSpline_i = np.arange(z_i-dz_spline, z_i+dz_spline)

        
    #     fitSpline_i = interpolate.splrep(x = X_fitSpline_i,
    #                                      y = finalDists_fitSpline_i,
    #                                      k = 3)
        
    #     listX_fitSpline.append(X_fitSpline_i)
        
    #     listY_fitSpline.append(interpolate.splev(X_fitSpline_i, fitSpline_i))
        
        
    #     der100_i = interpolate.splev(x = X_fitSpline_i,
    #                                tck = fitSpline_i,
    #                                der=1)
        
    #     der10_i = interpolate.splrep(x = X_fitSpline_i,
    #                                        y = der100_i,
    #                                        k = 1)
        
    #     der1_i = interpolate.splev(x = X_fitSpline_i,
    #                                tck = der10_i,
    #                                der=0)
        
    #     der2_i = interpolate.splev(x = X_fitSpline_i,
    #                                 tck = der10_i,
    #                                 der=1)
        
    #     listDer1.append(der1_i)
    #     listDer2.append(der2_i)
    
    #### Second try with k*x²
       
    dz_fitPoly = int(depthoDepth/32)

    Cost_fitPoly = sumFinalD[z - dz_fitPoly : z + dz_fitPoly+1]
    X_fitPoly = np.arange(-dz_fitPoly, dz_fitPoly + 1, dtype=int)
    
    def f_sq(x, k):
        return(k * x**2)

    popt, pcov = curve_fit(f_sq, X_fitPoly, Cost_fitPoly - sumFinalD[z], 
                           p0=[1], bounds=(-np.inf, np.inf))
    z_quality = popt[0]*1e3
    
    listX_fitPoly = []
    listY_fitPoly = []
    listParms_fitPoly = []
    
    for i in range(Nframes):
        z_i = np.argmin(finalDists[i])
        Cost_fitPoly_i = finalDists[i][z_i - dz_fitPoly : z_i + dz_fitPoly + 1]
        X_fitPoly_i = np.arange(-dz_fitPoly, dz_fitPoly + 1)
    
        popt_i, pcov_i = curve_fit(f_sq, X_fitPoly_i, Cost_fitPoly_i - finalDists[i][z_i], 
                                   p0=[1], bounds=(-np.inf, np.inf))
    
        listX_fitPoly.append(X_fitPoly_i)
        listY_fitPoly.append(Cost_fitPoly_i)
        listParms_fitPoly.append(popt_i[0])
    
        
    # X_fitSpline = np.arange(z-dz_spline,z+dz_spline)

    #### PLOT
    if plot > 0:
        fig, axes = plt.subplots(5, max(Nframes, 3), figsize = (max(Nframes, 3)*4,7.5))
        fig.tight_layout()
        im = I[0]
        X2, Y2 = listXY[0][0], listXY[0][1]
        
        pStart, pStop = np.percentile(im, (1, 99))
        axes[0,0].imshow(im, vmin = pStart, vmax = 1.5*pStop, cmap = 'gray')
        col_im = 'cyan'
        dx, dy = 10, 60
        axes[0,0].plot([X2], [Y2], marker = '+', color = 'red')
        axes[0,0].plot([X2-dx,X2-dx], [Y2-dy,Y2+dy], ls = '--', color = col_im)
        axes[0,0].plot([X2+dx,X2+dx], [Y2-dy,Y2+dy], ls = '--', color = col_im)
        axes[0,0].plot([X2-dx,X2+dx], [Y2-dy,Y2-dy], ls = '--', color = col_im)
        axes[0,0].plot([X2-dx,X2+dx], [Y2+dy,Y2+dy], ls = '--', color = col_im)
        
        axes[0,1].imshow(deptho)
        # TEST !!! # -> Works well !
        XL0, YL0 = axes[0,1].get_xlim(), axes[0,1].get_ylim()
        extent = (XL0[0], YL0[0]*(5/3), YL0[0], YL0[1])
        axes[0,1].imshow(deptho, extent = extent)
        # TEST !!! #

        pixLineHD = np.arange(0, hdSize, 1)
        zPos = np.arange(0, depthoDepth, 1)
        if Nframes == 3:
            color = ['orange', 'gold', 'green']
        else:
            color = gs.colorList30
            
        for i in range(Nframes):
            axes[1,i].imshow(listROI[i])
            #
            axes[2,i].plot(pixLineHD, listProfiles[i])
            #
            axes[3,i].plot(zPos, listDistances[i])
            limy3 = axes[3,i].get_ylim()
            min_i = np.argmin(listDistances[i])
            axes[3,i].plot([min_i,min_i],limy3,ls = '--', c = color[i])
            #
            axes[4,i].plot(zPos, finalDists[i])
            limy4 = axes[4,i].get_ylim()
            min_i = np.argmin(finalDists[i])
            
            axes[4,i].plot([min_i,min_i],limy4,ls = '--', c = color[i])
            
            # axes[4,i].plot(listX_fitSpline[i], listY_fitSpline[i], 'c-', lw = 0.8)
            # axes4itwin = axes[4,i].twinx()
            # axes4itwin.plot(listX_fitSpline[i], listDer1[i], color='blue', lw = 0.8)
            # axes4itwin.plot(listX_fitSpline[i], listDer2[i]*100, color='purple', lw = 0.8)
            
            axes[4,i].plot(listX_fitPoly[i] + min_i, 
                           listY_fitPoly[i], 
                           'c-', lw = 0.8)
            axes[4,i].plot(listX_fitPoly[i] + min_i, 
                           f_sq(listX_fitPoly[i], listParms_fitPoly[i]) + finalDists[i][min_i], 
                           'g-', lw = 0.8, label = '{:.2e}'.format(listParms_fitPoly[i]))
            axes[4,i].legend()
            
            
            axes[0,1].plot([axes[0,1].get_xlim()[0], axes[0,1].get_xlim()[1]-1], [listZ[i], listZ[i]], ls = '--', c = color[i])
            axes[0,1].plot([axes[0,1].get_xlim()[0], axes[0,1].get_xlim()[1]-1], [z,z], ls = '--', c = 'red')

            
        axes[0,2].plot(zPos, sumFinalD)
        
        # axes[0,2].plot(X_fitSpline, sumFinalD_fitSpline, 'c-', lw = 0.8)
        # axes02twin = axes[0,2].twinx()
        # axes02twin.plot(X_fitSpline, der1, color='blue', lw = 0.8)
        # axes02twin.plot(X_fitSpline, der2*100, color='purple', lw = 0.8)
        
        axes[0,2].plot(X_fitPoly + z, Cost_fitPoly, 'c-', lw = 0.8)
        axes[0,2].plot(X_fitPoly + z, f_sq(X_fitPoly, popt[0]) + sumFinalD[z], 
                       'g-', lw = 0.8, label = '{:.2e}'.format(popt[0]))
        
        limy0 = axes[0,2].get_ylim()
        axes[0,2].plot([z,z],limy0,ls = '-', c = 'red')
        axes[0,2].legend()

        # archiveFig(fig, axes, tempPlot, name='Z detection - Frames ' + str(iFNuplet), dpi = 300)
        
    
    
    return(zr, z_quality)

    
    
# %%% (1.4) Compute deptho quality

def computeDepthoQuality(depthoPath):
    #### 1 - Import depthograph
    HDZfactor = 10
    deptho = io.imread(depthoPath+'_Deptho.tif')
    depthoMetadata = pd.read_csv(depthoPath+'_Metadata.csv', sep=';')
    depthoStep = depthoMetadata.loc[0,'step']
    depthoZFocus = depthoMetadata.loc[0,'focus']
    
    # increase the resolution of the deptho with interpolation
    # print('deptho shape check')
    # print(deptho.shape)
    nX, nZ = deptho.shape[1], deptho.shape[0]
    X, Z = np.arange(0, nX, 1), np.arange(0, nZ, 1)
    Z_HD = np.arange(0, nZ, 1/HDZfactor)
    # print(XX.shape, ZZ.shape, deptho.shape, ZZ_HD.shape)
    
    # fd = interpolate.interp2d(XX, ZZ, deptho, kind='cubic')
    # depthoHD = fd(XX, ZZ_HD)
    fd = interpolate.RectBivariateSpline(Z, X, deptho)
    ZZ_HD, XX = np.meshgrid(Z_HD, X, indexing='ij')
    depthoHD = fd(ZZ_HD, XX, grid=False)
    
    
    depthoStepHD = depthoStep/HDZfactor
    depthoZFocusHD = depthoZFocus*HDZfactor
    # print(depthoHD.shape)
    
    #### 2.
    Ddz, Ddx = depthoHD.shape[0], depthoHD.shape[1]
    print(Ddz, Ddx)
    dz_fitPoly = int(Ddz/32)
    
    def f_sq(x, k):
        return(k * x**2)
    
    listDistances = np.zeros((Ddz, Ddz))
    listZQuality = np.zeros(Ddz)
    
    for z in range(Ddz):
        
        profile_z = depthoHD[z, :]
        listDistances[z] = ufun.squareDistance(depthoHD, profile_z, normalize = True) # Utility functions
        z_start = max(0, z - dz_fitPoly)
        z_stop = min(Ddz - 1, z + dz_fitPoly)
        # print(z, z_start, z_stop)
        Cost_fitPoly = listDistances[z][z_start : z_stop + 1]
        X_fitPoly = np.arange(z_start - z, z_stop - z + 1, dtype=int)
        popt, pcov = curve_fit(f_sq, X_fitPoly, Cost_fitPoly - listDistances[z][z], 
                               p0=[1], bounds=(-np.inf, np.inf))
        z_quality = popt[0]*1e3
        listZQuality[z] = z_quality
    
    Z = np.array([i for i in range(Ddz)]) - depthoZFocusHD
    plt.plot(Z, listZQuality)
    
    return(listZQuality)



# %% (3) Scripts

# %%% (3.1) SCRIPT for simple chains

mainPath = 'D://MagneticPincherData//Raw//23.09.08_CalibrationM450-mPEG_Try02//Chains'
depthoPath = 'D://MagneticPincherData//Raw//DepthoLibrary//23.09.08_CALIBRATION_M450-mPEG_step20_100X'


xyzDf_01, distanceDf_01, statsDf_01 = mainChainAnalysis(mainPath, depthoPath, 4.5)


# %%% (3.2) SCRIPT for deptho chains 2nd try

mainPath = 'D://MagneticPincherData//Raw//23.09.21_CalibrationM450-2025_Try02//BSA_chains-depthos'
depthoPath = 'D://MagneticPincherData//Raw//DepthoLibrary//23.09.21_CALIBRATION_M450-2025-BSA_step20_100X'

xyzDf_02, distanceDf_02, statsDf_02 = mainChainNupAnalysis(mainPath, depthoPath, 4.5)

# %%% (3.3)

mainPath = 'D://MagneticPincherData//Raw//22.04.29_CalibrationM450-2023_SecondTry//DepthoChains'
depthoPath = 'D://MagneticPincherData//Raw//DepthoLibrary//22.04.29_CALIBRATION_M450_step20_100X'

xyzDf_03, distanceDf_03, statsDf_03 = mainChainNupAnalysis(mainPath, depthoPath, 4.5)


# %%% plots

fig, ax = plt.subplots(1,1, figsize = (5, 1.6))
# sns.violinplot(x="D3", data=distanceDf_03, ax=ax, inner="quartile", orient = 'h',
#                 color = 'skyblue', bw = 'silverman')

sns.histplot(distanceDf_03, ax=ax, binwidth = 0.01, x="D3", color='skyblue')
Q1, M, Q3 = np.percentile(distanceDf_03.D3, 25), np.median(distanceDf_03.D3), np.percentile(distanceDf_03.D3, 75)
ax.axvline(Q1, ls='--', lw=1.5, color='darkred')
ax.axvline(M, ls='-', lw=1.5, color='darkred')
ax.axvline(Q3, ls='--', lw=1.5, color='darkred')

# sns.swarmplot(x="D3", data=distanceDf_03, ax=ax, color='gray', orient = 'h', size = 3)
ax.set_xlabel('Bead diameter (µm) - N = 351', fontsize = 10)
ax.set_xticklabels(ax.get_xticklabels(), fontsize = 9)
# ax.grid(axis = 'x')

ax.set_ylabel('Count', fontsize = 9)
ax.set_yticklabels(ax.get_yticklabels(), fontsize = 8)
ax.set_ylim([0, 90])

plt.tight_layout()

fig.savefig('C://Users//JosephVermeil//Desktop//MethodPaper//Figures//FigBeads//Fig.png', dpi=300)

# sns.swarmplot(y="D3", data=distanceDf_02, ax=ax[1], size = 3)
plt.show()


# %%% (3.2) SCRIPT for deptho chains 1st try

mainPath = 'D://MagneticPincherData//Raw//22.03.21_CalibrationM450-2023_FirstTry//DepthoChains//else'
depthoPath = 'D://MagneticPincherData//Raw//DepthoLibrary//22.03.21_CALIBRATION_M450_step20_100X'

xyzDf_02, distanceDf_02, statsDf_02 = mainChainNupAnalysis(mainPath, depthoPath, 4.5)

# %%% (3.3) SCRIPT for deptho qlt

# %%%% New obj 2

# mainPath = 'D://MagneticPincherData//Raw//22.04.29_CalibrationNewM450//DepthoChains'
depthoPath = 'D://MagneticPincherData//Raw//DepthoLibrary//22.04.29_CALIBRATION_M450_step20_100X'

listZQualityNewObj = computeDepthoQuality(depthoPath)

# %%%% New obj

# mainPath = 'D://MagneticPincherData//Raw//22.03.21_CalibrationNewM450//DepthoChains'
depthoPath = 'D://MagneticPincherData//Raw//DepthoLibrary//22.03.21_CALIBRATION_M450_step20_100X'

listZQualityNewObj = computeDepthoQuality(depthoPath)

# %%%% Old obj

depthoPath = 'D://MagneticPincherData//Raw//DepthoLibrary//22.02.09_M1_M450_step20_100X'

listZQualityOldObj = computeDepthoQuality(depthoPath)

#%% Numi's calibrations
#%%% 22.12.15 
# BEAD CALIBRATION - mPegBiotin + Streptavidin beads (4.5um), 1X and 10X mPEG concentrations

mainPath = 'D:/Anumita/MagneticPincherData//Raw//22.12.15_CalibrationStreptavidinPEG//1XPEG'
depthoPath = 'D:/Anumita/MagneticPincherData//Raw//DepthoLibrary//22.12.15CALIBRATION_P1_M450_step20_100X'


xyzDf_01, distanceDf_01, statsDf_01 = mainChainAnalysis(mainPath, depthoPath, 4.5)

mainPath = 'D:/Anumita/MagneticPincherData//Raw//22.12.15_CalibrationStreptavidinPEG//10XPEG'
depthoPath = 'D:/Anumita/MagneticPincherData//Raw//DepthoLibrary//22.12.15CALIBRATION_P2_M450_step20_100X'

xyzDf_02, distanceDf_02, statsDf_02 = mainChainAnalysis(mainPath, depthoPath, 4.5)

#%%% 23.09.08 
# BEAD CALIBRATION - mPegBiotin + Streptavidin beads (4.5um),  10X mPEG concentrations

mainPath = 'D:/Anumita/MagneticPincherData//Raw//23.09.08_BeadCalibration_Strept'
depthoPath = 'D:/Anumita/MagneticPincherData//Raw//DepthoLibrary//23.09.08_BeadCalibration_StreptPEG_M450_step20_100X'


xyzDf_01, distanceDf_01, statsDf_01 = mainChainAnalysis(mainPath, depthoPath, 4.5)
