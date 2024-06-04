# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 13:30:18 2023

@author: JosephVermeil
"""

# %% (0) Imports and settings

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import scipy.interpolate as si

import matplotlib
import matplotlib.pyplot as plt

import re
import os
import sys
import time
import random
import numbers
import warnings
import itertools


from copy import copy, deepcopy
from cycler import cycler
from datetime import date
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)

import GraphicStyles as gs
import UtilityFunctions as ufun

# %% Some functions

def getCellTimeSeriesData(cellID):
    mainPath = cp.DirDataTimeseriesStressStrain
    allTimeSeriesDataFiles = [f for f in os.listdir(mainPath) 
                          if (os.path.isfile(os.path.join(mainPath, f)))]
    fileFound = False
    nFile = len(allTimeSeriesDataFiles)
    iFile = 0
    while (not fileFound) and (iFile < nFile):
        f = allTimeSeriesDataFiles[iFile]
        if f.startswith(cellID):
            timeSeriesDataFilePath = os.path.join(mainPath, f)
            timeSeriesDataFrame = pd.read_csv(timeSeriesDataFilePath, sep=';')
            fileFound = True
        iFile += 1
    if not fileFound:
        timeSeriesDataFrame = pd.DataFrame([])
    else:
        for c in timeSeriesDataFrame.columns:
                if 'Unnamed' in c:
                    timeSeriesDataFrame = timeSeriesDataFrame.drop([c], axis=1)
    return(timeSeriesDataFrame)

def getCompressionDf(tsdf, i):
    DIAMETER = 4477
    tsdf_i = tsdf[tsdf['idxAnalysis'] == i]
    tsdf_i = tsdf_i.dropna(axis='index', subset='Stress')
    tsdf_i['h'] = tsdf_i['D3'] - DIAMETER
    return(tsdf_i)

# %% Mechanics

def CM_O1(h, E, H0, DIAMETER):
    """
    Note
    -------
    Compatible units: µm, pN, Pa ; or nm, pN, µPa.
    """
    
    R = DIAMETER/2
    f = (np.pi*E*R*((H0-h)**2))/(3*H0)
    return(f)

def CM_O2(h, E, H0, DIAMETER):
    """
    Note
    -------
    Compatible units: µm, pN, Pa ; or nm, pN, µPa.
    """
    
    R = DIAMETER/2
    f = (np.pi*E*R*((H0-h)**2))/(3*H0)
    return(f)

def invCM_O1(f, E, H0, DIAMETER):
    """
    Note
    -------
    Compatible units: µm, pN, Pa ; or nm, pN, µPa.
    """
    R = DIAMETER/2
    h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
    return(h)


def invCM_O2(f, E, H0, DIAMETER):
    """
    Note
    -------
    Compatible units: µm, pN, Pa ; or nm, pN, µPa.
    """
    R = DIAMETER/2
    h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
    return(h)


# %% Fit

def fitChadwick_hf(h, f, D):
    """
    Note
    -------
    Units in the fits: µm, pN, Pa; that is why the modulus will be multiplied by 1e6.
    """
    
    R = D/2000
    h = h/1000
    Npts = len(h)
    error = False
    
    def chadwickModel(h, Y, A, H0):
        f = (np.pi*Y*R*((H0-h)**2))/(2*H0) + (np.pi*A*R*((H0-h)**3))/(12*H0**2)
        return(f)

    try:
        # some initial parameter values - must be within bounds
        
        initY = (3*max(h)*max(f))/(np.pi*(R)*(max(h)-min(h))**2) # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
        initA = 1e-10 # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
        initH0 = 1.5*max(h) # H0 ~ h_max
        
        print(initY)
        
        initialParameters = [initY, initA, initH0]
    
        # bounds on parameters - initial parameters must be within these
        lowerBounds = (initY/2, 0, 1.1*np.max(h))
        upperBounds = (np.Inf, 0.0001, np.Inf)
        parameterBounds = [lowerBounds, upperBounds]


        # params = [E, H0] ; ses = [seE, seH0]
        params, covM = curve_fit(chadwickModel, h, f, p0=initialParameters, bounds = parameterBounds)
        ses = np.array([covM[0,0]**0.5, covM[1,1]**0.5, covM[2,2]**0.5])
        # params[0], ses[0] = params[0]*1e6, ses[0]*1e6 # Convert Y & seY to Pa
        # params[1], ses[1] = params[1]*1e6, ses[1]*1e6 # Convert A & seA to Pa
        
    except:
        error = True
        params = np.ones(3) * np.nan
        ses = np.ones(3) * np.nan
        
    res = (params, ses, error)
        
    return(res)



# %% Test

cellId = '23-07-06_M7_P1_C62'
tsdf = getCellTimeSeriesData(cellId)
tsdf_1 = getCompressionDf(tsdf, 2)

fig, ax = plt.subplots(1, 1)
ax.plot(tsdf_1.h, tsdf_1.F)


params, ses, error = fitChadwick_hf(tsdf_1.h, tsdf_1.F, 4477)

print(params)

def chadwickModel(h, Y, A, H0, R):
    f = (np.pi*Y*R*((H0-h)**2))/(2*H0) + (np.pi*A*R*((H0-h)**3))/(8*H0**2)
    return(f)

R = 4477/2000
Y, A, H0 = params

f = chadwickModel(tsdf_1.h/1000, Y, A, H0, R)

ax.plot(tsdf_1.h, f, 'k.')











