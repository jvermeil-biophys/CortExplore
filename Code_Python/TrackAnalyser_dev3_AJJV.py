# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:18:47 2022

@author: anumi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:07:45 2022

@author: JosephVermeil & AnumitaJawahar
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


from copy import copy
from cycler import cycler
from datetime import date
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression


#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)

import GraphicStyles as gs
import UtilityFunctions as ufun

# %%% Warniong setting



# %%% Smaller settings

# Pandas settings
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)

# Plot settings
gs.set_default_options_jv()


####

dictSubstrates = {}
for i in range(5,105,5):
    dictSubstrates['disc' + str(i) + 'um'] = str(i) + 'um fibronectin discs'
    dictSubstrates['disc{:02.0f}um'.format(i)] = str(i) + 'um fibronectin discs'

                    
# %% (1) TimeSeries functions

def getCellTimeSeriesData(cellID, fromPython = True, fromCloud = False):
    if not fromCloud:
        mainPath = cp.DirDataTimeseries
    else:
        mainPath = cp.DirCloudTimeseries
    if fromPython:
        allTimeSeriesDataFiles = [f for f in os.listdir(mainPath) 
                              if (os.path.isfile(os.path.join(mainPath, f)) 
                                  and f.endswith("PY.csv"))]
    else:
        allTimeSeriesDataFiles = [f for f in os.listdir(mainPath) 
                              if (os.path.isfile(os.path.join(mainPath, f)) 
                                  and f.endswith(".csv") and not f.endswith("PY.csv"))]
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

def plotCellTimeSeriesData(cellID, fromPython = True):
    X = 'T'
    Y = np.array(['B', 'F', 'dx', 'dy', 'dz', 'D2', 'D3'])
    units = np.array([' (mT)', ' (pN)', ' (µm)', ' (µm)', ' (µm)', ' (µm)', ' (µm)'])
    timeSeriesDataFrame = getCellTimeSeriesData(cellID, fromPython)
    print(timeSeriesDataFrame.shape)
    # my_default_color_cycle = cycler(color=my_default_color_list)
    # plt.rcParams['axes.prop_cycle'] = my_default_color_cycle
    if not timeSeriesDataFrame.size == 0:
#         plt.tight_layout()
#         fig.show() # figsize=(20,20)
        axes = timeSeriesDataFrame.plot(x=X, y=Y, kind='line', ax=None, subplots=True, sharex=True, sharey=False, layout=None, \
                       figsize=(8,10), use_index=True, title = cellID + ' - f(t)', grid=None, legend=False, style=None, logx=False, logy=False, \
                       loglog=False, xticks=None, yticks=None, xlim=None, ylim=None, rot=None, fontsize=None, colormap=None, \
                       table=False, yerr=None, xerr=None, secondary_y=False, sort_columns=False)
        plt.gcf().tight_layout()
        for i in range(len(Y)):
            axes[i].set_ylabel(Y[i] + units[i])
        # plt.gcf().show()
        plt.show()
    else:
        print('cell not found')
    # plt.rcParams['axes.prop_cycle'] = my_default_color_cycle
        
def getCellTrajData(cellID, Ntraj = 2):
    trajDir = os.path.join(cp.DirDataTimeseries, 'Trajectories')
    allTrajFiles = [f for f in os.listdir(trajDir) 
                    if (os.path.isfile(os.path.join(trajDir, f)) 
                        and f.endswith(".csv"))]
    fileFound = 0
    nFile = len(allTrajFiles)
    iFile = 0
    listTraj = []
    while (fileFound < Ntraj) and (iFile < nFile):
        f = allTrajFiles[iFile]
        if f.startswith(cellID):
            fileFound += 1
            trajFilePath = os.path.join(trajDir, f)
            trajDataFrame = pd.read_csv(trajFilePath, sep='\t')
            for c in trajDataFrame.columns:
                if 'Unnamed' in c:
                    trajDataFrame = trajDataFrame.drop([c], axis=1)
            
            pos = 'na'
            if 'In' in f:
                pos = 'in'
            elif 'Out' in f:
                pos = 'out'
            
            dictTraj = {}
            dictTraj['df'] = trajDataFrame
            dictTraj['pos'] = pos
            dictTraj['path'] = trajFilePath
            
            listTraj.append(dictTraj)
            
        iFile += 1

    return(listTraj)
        
def addExcludedCell(cellID, motive):
    f = open(os.path.join(cp.DirRepoExp, 'ExcludedCells.txt'), 'r')
    lines = f.readlines()
    nLines = len(lines)
    excludedCellsList = []
    for iLine in range(nLines):
        line = lines[iLine]
        splitLine = line[:-1].split(',')
        excludedCellsList.append(splitLine[0])
    if cellID in excludedCellsList:
        newlines = copy(lines)
        iLineOfInterest = excludedCellsList.index(cellID)
        if motive not in newlines[iLineOfInterest][:-1].split(','):
            newlines[iLineOfInterest] = newlines[iLineOfInterest][:-1] + ',' + motive + '\n'            
    else:
        newlines = copy(lines)
        newlines.append('' + cellID + ',' + motive + '\n')
    f.close()
    f = open(os.path.join(cp.DirRepoExp, 'ExcludedCells.txt'), 'w')
    f.writelines(newlines)
    
def getExcludedCells():
    f = open(os.path.join(cp.DirRepoExp, 'ExcludedCells.txt'), 'r')
    lines = f.readlines()
    nLines = len(lines)
    excludedCellsDict = {}
    for iLine in range(nLines):
        line = lines[iLine]
        splitLine = line[:-1].split(',')
        excludedCellsDict[splitLine[0]] = splitLine[1:]
    return(excludedCellsDict)

# %% (2) GlobalTables functions

# %%% (2.1) Exp conditions

# See UtilityFunctions.py

# %%% (2.2) Constant Field experiments

listColumnsCtField = ['date','cellName','cellID','manipID',\
                      'duration','medianRawB','medianThickness',\
                      '1stDThickness','9thDThickness','fluctuAmpli',\
                      'R2_polyFit','validated']

def analyseTimeSeries_ctField(f, tsDf, expDf):
    results = {}
    
    thisManipID = ufun.findInfosInFileName(f, 'manipID')
    thisExpDf = expDf.loc[expDf['manipID'] == thisManipID]
    # Deal with the asymmetric pair case : the diameter can be for instance 4503 (float) or '4503_2691' (string)
    diameters = thisExpDf.at[thisExpDf.index.values[0], 'bead diameter'].split('_')
    if len(diameters) == 2:
        DIAMETER = (int(diameters[0]) + int(diameters[1]))/2.
    else:
        DIAMETER = int(diameters[0])
    
    results['duration'] = np.max(tsDf['T'])
    results['medianRawB'] = np.median(tsDf.B)
    results['medianThickness'] = (1000*np.median(tsDf.D3))-DIAMETER
    results['1stDThickness'] = (1000*np.percentile(tsDf.D3, 10))-DIAMETER
    results['9thDThickness'] = (1000*np.percentile(tsDf.D3, 90))-DIAMETER
    results['fluctuAmpli'] = (results['9thDThickness'] - results['1stDThickness'])
    results['validated'] = (results['1stDThickness'] > 0)

    
    # R2 polyfit to see the regularity. Order = 5 !
    X, Y = tsDf['T'], tsDf['D3']
    p, residuals, rank, singular_values, rcond = np.polyfit(X, Y, deg=5, full=True)
    Y2 = np.zeros(len(X))
    for i in range(len(X)):
        deg = len(p)-1
        for k in range(deg+1):
            Y2[i] += p[k]*(X[i]**(deg-k))
    results['R2_polyFit'] = ufun.get_R2(Y, Y2)
    return(results)



def createDataDict_ctField(list_ctFieldFiles):
    tableDict = {}
    tableDict['date'], tableDict['cellName'], tableDict['cellID'], tableDict['manipID'] = [], [], [], []
    tableDict['duration'], tableDict['medianRawB'], tableDict['medianThickness'] = [], [], []
    tableDict['1stDThickness'], tableDict['9thDThickness'], tableDict['fluctuAmpli'] = [], [], []
    tableDict['R2_polyFit'], tableDict['validated'] = [], []
    expDf = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = cp.suffix)
    for f in list_ctFieldFiles:
        split_f = f.split('_')
        tableDict['date'].append(split_f[0])
        tableDict['cellName'].append(split_f[1] + '_' + split_f[2] + '_' + split_f[3])
        tableDict['cellID'].append(split_f[0] + '_' + split_f[1] + '_' + split_f[2] + '_' + split_f[3])
        tableDict['manipID'].append(split_f[0] + '_' + split_f[1])
        tS_DataFilePath = os.path.join(cp.DirDataTimeseries, f)
        current_tsDf = pd.read_csv(tS_DataFilePath, ';')
        current_resultDict = analyseTimeSeries_ctField(f, current_tsDf, expDf)
        for k in current_resultDict.keys():
            tableDict[k].append(current_resultDict[k])
    return(tableDict)


def computeGlobalTable_ctField(task = 'fromScratch', fileName = 'Global_CtFieldData', save = False,
                               source = 'Matlab'):
    """
    Compute the GlobalTable_ctField from the time series data files.
    > Option task='fromScratch' will analyse all the time series data files and construct 
    a new GlobalTable from them regardless of the existing GlobalTable.
    > Option task='updateExisting' will open the existing GlobalTable and determine 
    which of the time series data files are new ones, and will append the existing GlobalTable 
    with the data analysed from those new files.
    > Else, having task= a date, a cellID or a manipID will create a globalTable with this source only.
    """
    ctFieldFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                                  if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
                                      and ('thickness' in f))]
        
#     print(ctFieldFiles)

    suffixPython = '_PY'
    if source == 'Matlab':
        ctFieldFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                      if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
                      and ('thickness' in f) and not (suffixPython in f))]
        
    elif source == 'Python':
        ctFieldFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                      if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
                      and ('thickness' in f) and (suffixPython in f))]
        # print(list_mecaFiles)


    if task == 'fromScratch':
        # create a dict containing the data
        tableDict = createDataDict_ctField(ctFieldFiles) # MAIN SUBFUNCTION
        # create the table
        CtField_DF = pd.DataFrame(tableDict)
        
    elif task == 'updateExisting':
        # get existing table
        try:
            savePath = os.path.join(cp.DirDataAnalysis, (fileName + '.csv'))
            existing_CtField_DF = pd.read_csv(savePath, sep=';')
            for c in existing_CtField_DF.columns:
                if 'Unnamed' in c:
                    existing_CtField_DF = existing_CtField_DF.drop([c], axis=1)
        except:
            print('No existing table found')
        # find which of the time series files are new
        new_ctFieldFiles = []
        for f in ctFieldFiles:
            split_f = f.split('_')
            currentCellID = split_f[0] + '_' + split_f[1] + '_' + split_f[2] + '_' + split_f[3]
            if currentCellID not in existing_CtField_DF.cellID.values:
                new_ctFieldFiles.append(f)
        new_tableDict = createDataDict_ctField(new_ctFieldFiles) # MAIN SUBFUNCTION
        # create the table with new data
        new_CtField_DF = pd.DataFrame(new_tableDict)
        # fuse the two
        new_CtField_DF.index += existing_CtField_DF.shape[0]
        CtField_DF = pd.concat([existing_CtField_DF, new_CtField_DF])
        
    else: # If task is neither 'fromScratch' nor 'updateExisting'
    # Then task can be a substring that can be in some timeSeries file !
    # It will create a table with only these files, WITH OR WITHOUT SAVING IT !
    # And it can plot figs from it.
        # save = False
        new_ctFieldFiles = []
        for f in new_ctFieldFiles:
            split_f = f.split('_')
            currentCellID = split_f[0] + '_' + split_f[1] + '_' + split_f[2] + '_' + split_f[3]
            if task in currentCellID:
                new_ctFieldFiles.append(f)
        # create the dict with new data
        new_tableDict = createDataDict_ctField(new_ctFieldFiles) # MAIN SUBFUNCTION
        # create the dataframe from it
        CtField_DF = pd.DataFrame(new_tableDict)
    
    dateExemple = CtField_DF.loc[CtField_DF.index[0],'date']
    if re.match(ufun.dateFormatExcel, dateExemple):
        CtField_DF.loc[:,'date'] = CtField_DF.loc[:,'date'].apply(lambda x: x.split('/')[0] + '-' + x.split('/')[1] + '-' + x.split('/')[2][2:])
    
    for c in CtField_DF.columns:
        if 'Unnamed' in c:
            CtField_DF = CtField_DF.drop([c], axis=1)
    
    if save:
        saveName = fileName + '.csv'
        savePath = os.path.join(cp.DirDataAnalysis, saveName)
        CtField_DF.to_csv(savePath, sep=';')
        
    return(CtField_DF)



def getGlobalTable_ctField(fileName = 'Global_CtFieldData'):
    try:
        savePath = os.path.join(cp.DirDataAnalysis, (fileName + '.csv'))
        CtField_DF = pd.read_csv(savePath, sep=';')
        for c in CtField_DF.columns:
            if 'Unnamed' in c:
                CtField_DF = CtField_DF.drop([c], axis=1)
        print('Extracted a table with ' + str(CtField_DF.shape[0]) + ' lines and ' + str(CtField_DF.shape[1]) + ' columns.')
        
    except:
        print('No existing table found')
        
    dateExemple = CtField_DF.loc[CtField_DF.index[0],'date']
    if re.match(ufun.dateFormatExcel, dateExemple):
        print('dates corrected')
        CtField_DF.loc[:,'date'] = CtField_DF.loc[:,'date'].apply(lambda x: x.split('/')[0] + '-' + x.split('/')[1] + '-' + x.split('/')[2][2:])
#         mecaDF['ManipID'] = mecaDF['ExpDay'] + '_' + mecaDF['CellName'].apply(lambda x: x.split('_')[0])
    return(CtField_DF)

# %%% (2.3) Compressions experiments

#### Workflow
# * analyseTimeSeries_meca() analyse 1 file and return the dict (with the results of the analysis)
# * buildDf_meca() call the previous function on the given list of files and concatenate the results
# * computeGlobalTable_meca() call the previous function and convert the dict to a DataFrame


# %%%% TO DO LIST

#### 
# - When this is done and runs ok, think of the results and the plots.
# - Write the polynomial fit.


# class ResultsCompression:
#     def __init__(self, dictColumns, CC):
#         Ncomp = CC.Ncomp
        
#         main = {}
#         for k in dictColumnsMeca.keys():
#             main[k] = [dictColumnsMeca[k] for m in range(Ncomp)]
#         self.main = main
#         self.dictColumns = dictColumns
#         self.Ncomp = CC.Ncomp

# %%%% mechanical models

def chadwickModel(h, E, H0, DIAMETER):
    R = DIAMETER/2
    f = (np.pi*E*R*((H0-h)**2))/(3*H0)
    return(f)

def inversedChadwickModel(f, E, H0, DIAMETER):
    R = DIAMETER/2
    h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
    return(h)

def dimitriadisModel(h, E, H0, DIAMETER, v = 0, order = 2):
    R = DIAMETER/2
    delta = H0-h
    X = np.sqrt(R*delta)/h
    ks = ufun.getDimitriadisCoefs(v, order)
    poly = np.zeros_like(X)
    for i in range(order+1):
        poly = poly + ks[i] * X**i
    f = ((4 * E * R**0.5 * delta**1.5)/(3 * (1 - v**2))) * poly
    return(f)

def constitutiveRelation(strain, K, stress0):
    stress = (K * strain) + stress0
    return(stress)

def inversedConstitutiveRelation(stress, K, strain0):
    strain = (stress / K) + strain0
    return(strain)

    

# %%%% general fitting functions


def fitChadwick_hf(h, f, D):
    # f in pN, h & D in nm.
    
    R = D/2
    Npts = len(h)
    error = False
    
    def chadwickModel(h, E, H0):
        f = (np.pi*E*R*((H0-h)**2))/(3*H0)
        return(f)

    def inversedChadwickModel(f, E, H0):
        h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
        return(h)

    # some initial parameter values - must be within bounds
    initH0 = max(h) # H0 ~ h_max
    initE = (3*max(h)*max(f))/(np.pi*(R)*(max(h)-min(h))**2) # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
    
    initialParameters = [initE, initH0]

    # bounds on parameters - initial parameters must be within these
    lowerBounds = (0, 0)
    upperBounds = (np.Inf, np.Inf)
    parameterBounds = [lowerBounds, upperBounds]

    try:
    # params = [E, H0] ; ses = [seE, seH0]
        params, covM = curve_fit(inversedChadwickModel, f, h, p0=initialParameters, bounds = parameterBounds)
        ses = np.array([covM[0,0]**0.5, covM[1,1]**0.5])
        params[0], ses[0] = params[0]*1e6, ses[0]*1e6 # Convert E & seE to Pa
        
    except:
        error = True
        params = np.ones(2) * np.nan
        ses = np.ones(2) * np.nan
        
    res = (params, ses, error)
        
    return(res)
        


def fitDimitriadis_hf(h, f, D, order = 2):
    # f in pN, h & D in nm.
    
    R = D/2
    Npts = len(h)
    error = False
    v = 0 # Poisson
    
    def dimitriadisModel(h, E, H0):
        
        delta = H0-h
        X = np.sqrt(R*delta)/h
        ks = ufun.getDimitriadisCoefs(v, order)
        
        poly = np.zeros_like(X)
        
        for i in range(order+1):
            poly = poly + ks[i] * X**i
            
        f = ((4 * E * R**0.5 * delta**1.5)/(3 * (1 - v**2))) * poly
        return(f)

    # some initial parameter values - must be within bounds
    # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
    initE = (3*max(h)*max(f))/(np.pi*(R)*(max(h)-min(h))**2) 
    # H0 ~ h_max
    initH0 = max(h) 

    # initH0, initE = initH0*(initH0>0), initE*(initE>0)
    
    initialParameters = [initE, initH0]

    # bounds on parameters - initial parameters must be within these
    lowerBounds = (0, max(h))
    upperBounds = (np.Inf, np.Inf)
    parameterBounds = [lowerBounds, upperBounds]

    try:
        # params = [E, H0] ; ses = [seE, seH0]
        params, covM = curve_fit(dimitriadisModel, h, f, p0=initialParameters, bounds = parameterBounds)
        ses = np.array([covM[0,0]**0.5, covM[1,1]**0.5])
        params[0], ses[0] = params[0]*1e6, ses[0]*1e6 # Convert E & seE to Pa
        
    except:
        error = True
        params = np.ones(2) * np.nan
        ses = np.ones(2) * np.nan
        
    res = (params, ses, error)
        
    return(res)



def fitLinear_ss(stress, strain, weights = []):
    
    error = False
    
    if len(weights) == 0:
        weights = np.ones_like(stress, dtype = bool)
    
    masked = (np.all(weights.astype(bool) == weights)) # Are all the weights equal to 0 or 1 ? If so it is a mask
    weighted = not masked # The other case

    def constitutiveRelation(strain, K, stress0):
        stress = (K * strain) + stress0
        return(stress)
    
    def inversedConstitutiveRelation(stress, K, strain0):
        strain = (stress / K) + strain0
        return(strain)
    
    try:
    # if np.sum(weights) > 0:
        if masked:
            ols_model = sm.OLS(strain[weights], sm.add_constant(stress[weights]))
            results_ols = ols_model.fit()
            # try:
            S0, K = results_ols.params[0], 1/results_ols.params[1]
            seS0, seK = results_ols.HC3_se[0], results_ols.HC3_se[1]*K**2 # See note below
            # except:
            #     print(strain[weights])
            #     print(results_ols.params)
        
        if weighted:
            wls_model = sm.WLS(strain, sm.add_constant(stress), weights=weights)
            results_wls = wls_model.fit()
            S0, K = results_wls.params[0], 1/results_wls.params[1]
            seS0, seK = results_wls.HC3_se[0], results_wls.HC3_se[1]*K**2 # See note below
            
            # NB : Here we fit strain = L * stress + S0 ; L = 1/K
            # To find K, we just take 1/params[1] = 1/L
            # To find seK, there is a trick : se(K) / K = relative error = se(L) / L ; so se(K) = se(L) * K**2
            
        params = np.array([K, S0])
        ses = np.array([seK, seS0])
        
    except:
    # else:
        error = True
        params = np.ones(2) * np.nan
        ses = np.ones(2) * np.nan

    res = (params, ses, error)
        
    return(res)


# %%%% functions to store the results of fits

def makeDictFit_hf(params, ses, error, 
                   x, y, yPredict, 
                   err_chi2, dictFitValidation):
    if not error:
        E, H0 = params
        seE, seH0 = ses

        alpha = 0.975
        dof = len(y)-len(params)
        q = st.t.ppf(alpha, dof) # Student coefficient
        R2 = ufun.get_R2(y, yPredict)
        Chi2 = ufun.get_Chi2(y, yPredict, dof, err_chi2)        

        ciwE = q*seE
        ciwH0 = q*seH0
        
        nbPts = len(y)
        
        isValidated = (E > 0 and
                       nbPts >= dictFitValidation['crit_nbPts'] and
                       R2 >= dictFitValidation['crit_R2'] and 
                       Chi2 <= dictFitValidation['crit_Chi2'])
        issue = ''
        if isValidated:
            issue += 'none'
        else:
            if not E > 0:
                issue += 'E<0_'
            if not nbPts >= dictFitValidation['crit_nbPts']:
                issue += 'nbPts<{:.0f}_'.format(dictFitValidation['crit_nbPts'])
            if not nbPts >= dictFitValidation['crit_R2']:
                issue += 'R2<{:.2f}_'.format(dictFitValidation['crit_R2'])
            if not nbPts >= dictFitValidation['crit_Chi2']:
                issue += 'Chi2>{:.1f}_'.format(dictFitValidation['crit_Chi2'])
    
    else:
        E, seE, H0, seH0 = np.nan, np.nan, np.nan, np.nan
        R2, Chi2 = np.nan, np.nan
        ciwE, ciwH0 = np.nan, np.nan
        isValidated = False
        issue = 'error'

    res =  {'error':error,
            'nbPts':len(y),
            'E':E, 'seE':seE,
            'H0':H0, 'seH0':seH0,
            'R2':R2, 'Chi2':Chi2,
            'ciwE':ciwE, 
            'ciwH0':ciwH0,
            'x': x,
            'y': y,
            'yPredict': yPredict,
            'valid': isValidated,
            'issue': issue
            }
    
    return(res)


def makeDictFit_ss(params, ses, error, 
                   center, halfWidth, x, y, yPredict, 
                   err_chi2, dictFitValidation):
    
    nbPts = len(y)

    if (not error) and (nbPts > 0):
        K, S0 = params
        seK, seS0 = ses

        alpha = 0.975
        dof = len(y)-len(params)
        q = st.t.ppf(alpha, dof) # Student coefficient
        R2 = ufun.get_R2(y, yPredict)
        Chi2 = ufun.get_Chi2(y, yPredict, dof, err_chi2)        

        ciwK = q*seK
        
        center_y = np.median(y)
        
        isValidated = (K > 0 and
                       nbPts >= dictFitValidation['crit_nbPts'] and
                       R2 >= dictFitValidation['crit_R2'] and 
                       Chi2 <= dictFitValidation['crit_Chi2'])
        issue = ''
        if isValidated:
            issue += 'none'
        else:
            if not K > 0:
                issue += 'K<0_'
            if not nbPts >= dictFitValidation['crit_nbPts']:
                issue += 'nbPts<{:.0f}_'.format(dictFitValidation['crit_nbPts'])
            if not R2 >= dictFitValidation['crit_R2']:
                issue += 'R2<{:.2f}_'.format(dictFitValidation['crit_R2'])
            if not Chi2 <= dictFitValidation['crit_Chi2']:
                issue += 'Chi2>{:.1f}_'.format(dictFitValidation['crit_Chi2'])
    
    else:
        error = True
        K, seK = np.nan, np.nan
        R2, Chi2 = np.nan, np.nan
        nbPts = np.nan
        ciwK = np.nan
        center_y = np.nan
        isValidated = False
        issue = 'error'

    res =  {'error': error,
            'nbPts': nbPts,
            'K': K, 'seK': seK,
            'R2': R2, 'Chi2': Chi2,
            'ciwK': ciwK,
            'center_x': center,
            'halfWidth_x': halfWidth,
            'center_y': center_y,
            'x': x,
            'y': y,
            'yPredict': yPredict,
            'valid': isValidated,
            'issue': issue
            }
    
    return(res)

def nestedDict_to_DataFrame(D):
    """
    Convert D, a dict of dicts, into a pandas DataFrame, after removing non-numeric values.
    
    D = {'d1' : d1, 'd2' : d2, ..., 'dn' : dn} where each di = {'k1' : val1, 'k2' : val2, ..., 'kn' : valn},
    and the keys k1 ... kn are the same for all dicts d1 ... dn.
    
    In the resulting DataFrame, k1 ... kn will be the columns, and d1 ... dn the rows.
    
    This function remove any couple 'ki':vali where vali is not a number (e.g. a numy array).
    """
    if len(D) > 0:
        fits =  list(D.keys())
        Nfits = len(fits)
        f0 = fits[0]
        d0 = D[f0]
        
        cols_others = [k for k in d0.keys() if (k in ['valid', 'issue'])]
        cols_num = [k for k in d0.keys() if (isinstance(d0[k], numbers.Number)) \
                                            and (k not in cols_others)]
        

        D2 = {}
        index = [] # will contains all the names of the small dicts 'd'
        
        for c in cols_num:
            D2[c] = np.zeros(Nfits)
        for c in cols_others:
            D2[c] = []
    
        for jj in range(Nfits):
            f = fits[jj]
            d = D[f]
            for c in cols_others:
                D2[c].append(d[c])
            for c in cols_num:
                D2[c][jj] = d[c]
            index.append(f)
            
    else:
        D2 = {}
        
    df = pd.DataFrame(D2)
    df.insert(0, 'id', index)
    return(df)



        

# %%%% classes
        

class CellCompression:
    """
    This class deals with all that requires the whole array of compressions.
    """
    
    def __init__(self, cellID, timeseriesDf, thisExpDf, fileName):
        self.tsDf = timeseriesDf
        self.expDf = thisExpDf
        self.cellID = cellID
        self.fileName = fileName
        
        Ncomp = max(timeseriesDf['idxAnalysis'])
        
        diameters = thisExpDf.at[thisExpDf.index.values[0], 'bead diameter'].split('_')
        if len(diameters) == 2:
            D = (int(diameters[0]) + int(diameters[1]))/2.
        else:
            D = int(diameters[0])
        
        EXPTYPE = str(thisExpDf.at[thisExpDf.index.values[0], 'experimentType'])
        
        # Field infos
        normalField = int(thisExpDf.at[thisExpDf.index.values[0], 'normal field'])
        
        compField = thisExpDf.at[thisExpDf.index.values[0], 'ramp field'].split('_')
        minCompField = float(compField[0])
        maxCompField = float(compField[1])
        
        # Loop structure infos
        loopStruct = thisExpDf.at[thisExpDf.index.values[0], 'loop structure'].split('_')
        nUplet = thisExpDf.at[thisExpDf.index.values[0], 'normal field multi images']
        
        loop_totalSize = int(loopStruct[0])
        loop_rampSize = int(loopStruct[1])
        loop_ctSize = int((loop_totalSize - loop_rampSize)/nUplet)
        
        
        self.Ncomp = Ncomp
        self.DIAMETER = D
        self.EXPTYPE = EXPTYPE
        self.normalField = normalField
        self.minCompField = minCompField
        self.maxCompField = maxCompField
        self.loopStruct = loopStruct
        self.nUplet = nUplet
        self.loop_totalSize = loop_totalSize
        self.loop_rampSize = loop_rampSize
        self.loop_ctSize = loop_ctSize
        
        # These fields are to be filled by methods later on
        self.listIndent = []
        self.listJumpsD3 = np.zeros(Ncomp, dtype = float)
        # self.method_bestH0 = 'Dimitriadis' # default
        
        self.df_mainResults = pd.DataFrame({})
        self.df_stressRegions = pd.DataFrame({})
        self.df_stressGaussian = pd.DataFrame({})
        self.df_nPoints = pd.DataFrame({})

        
        
        
    def getMaskForCompression(self, i, task = 'compression'):
        Npts = len(self.tsDf['idxAnalysis'].values)
        iStart = ufun.findFirst(np.abs(self.tsDf['idxAnalysis']), i+1)
        iStop = iStart + np.sum((np.abs(self.tsDf['idxAnalysis']) == i+1))
        
        if task == 'compression':
            mask = (self.tsDf['idxAnalysis'] == i+1)
        elif task == 'precompression':
            mask = (self.tsDf['idxAnalysis'] == -(i+1))
        elif task == 'compression & precompression':
            mask = (np.abs(self.tsDf['idxAnalysis']) == i+1)
        elif task == 'previous':
            i1 = max(0, iStart-(self.loop_ctSize))
            i2 = iStart
            mask = np.array([((i >= i1) and (i < i2)) for i in range(Npts)])
        elif task == 'following':
            i1 = iStop
            i2 = min(Npts, iStop+(self.loop_ctSize))
            mask = np.array([((i >= i1) and (i < i2)) for i in range(Npts)])
        elif task == 'surrounding':
            i1 = max(0,    iStart-(self.loop_ctSize//2))
            i2 = min(Npts, iStop +(self.loop_ctSize//2))
            mask = np.array([((i >= i1) and (i < i2)) for i in range(Npts)])
        
        return(mask)
        
        
    def correctJumpForCompression(self, i):
        colToCorrect = ['dx', 'dy', 'dz', 'D2', 'D3']
        mask = self.getMaskForCompression(i, task = 'compression & precompression')
        iStart = ufun.findFirst(np.abs(self.tsDf['idxAnalysis']), i+1)
        for c in colToCorrect:
            jump = np.median(self.tsDf[c].values[iStart:iStart+5]) - np.median(self.tsDf[c].values[iStart-2:iStart])
            self.tsDf.loc[mask, c] -= jump
            if c == 'D3':
                D3corrected = True
                jumpD3 = jump
        self.listJumpsD3[i] = jumpD3
        
    

    
        
    def plot_Timeseries(self, plotSettings):
        fig, ax = plt.subplots(1,1,figsize=(self.tsDf.shape[0]*(1/200),4))
        fig.suptitle(self.cellID)
        
        # Distance axis
        color = 'blue'
        ax.set_xlabel('t (s)')
        ax.set_ylabel('h (nm)', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.plot(self.tsDf['T'].values, self.tsDf['D3'].values-self.DIAMETER, 
                color = color, ls = '-', linewidth = 1, zorder = 1)
        
        for ii in range(self.Ncomp):
            IC = self.listIndent[ii]
            fitError = IC.dictFitFH_Chadwick['error']
            if not fitError:
                ax.plot(IC.Df['T'].values, IC.Df['D3'].values-self.DIAMETER, 
                        color = 'chartreuse', linestyle = '-', linewidth = 1.25, zorder = 3)
                
            else:
                ax.plot(IC.Df['T'].values, IC.Df['D3'].values-self.DIAMETER, 
                        color = 'crimson', linestyle = '-', linewidth = 1.25, zorder = 3)
        
        (axm, axM) = ax.get_ylim()
        ax.set_ylim([min(0,axm), axM])
        if (max(self.tsDf['D3'].values-self.DIAMETER) > 200):
            ax.set_yticks(np.arange(0, max(self.tsDf['D3'].values-self.DIAMETER), 100))
        
        # Force axis
        ax.tick_params(axis='y', labelcolor='b')
        axbis = ax.twinx()
        color = 'firebrick'
        axbis.set_ylabel('F (pN)', color=color)
        axbis.plot(self.tsDf['T'].values,self. tsDf['F'].values, color=color)
        axbis.tick_params(axis='y', labelcolor=color)
        axbis.set_yticks([0,500,1000,1500])
        minh = np.min(self.tsDf['D3'].values-self.DIAMETER)
        ratio = min(1/abs(minh/axM), 5)
        (axmbis, axMbis) = axbis.get_ylim()
        axbis.set_ylim([0, max(axMbis*ratio, 3*max(self.tsDf['F'].values))])
        
        axes = [ax, axbis]
        fig.tight_layout()
        return(fig, axes)
    
    
    def plot_FH(self, plotSettings, plotH0 = True, plotFit = True):
        nColsSubplot = 5
        nRowsSubplot = ((self.Ncomp-1) // nColsSubplot) + 1
        fig, axes = plt.subplots(nRowsSubplot, nColsSubplot,
                               figsize = (3*nColsSubplot, 3*nRowsSubplot))
        figTitle = 'Thickness-Force of indentations\n'
        if plotH0:
            figTitle += 'with H0 detection (' + self.method_bestH0 + ') ; ' 
        if plotFit:
            figTitle += 'with fit (Chadwick)'
        
        fig.suptitle(figTitle)
        
        for i in range(self.Ncomp):
            colSp = (i) % nColsSubplot
            rowSp = (i) // nColsSubplot
            if nRowsSubplot == 1:
                ax = axes[colSp]
            elif nRowsSubplot >= 1:
                ax = axes[rowSp,colSp]
                
            IC = self.listIndent[i]
            IC.plot_FH(fig, ax, plotSettings, plotH0 = plotH0, plotFit = plotFit)
            
        fig.tight_layout()
        return(fig, axes)
    
    
    
    def plot_SS(self, plotSettings, plotFit = True, fitType = 'stressRegion'):
        nColsSubplot = 5
        nRowsSubplot = ((self.Ncomp-1) // nColsSubplot) + 1
        fig, axes = plt.subplots(nRowsSubplot, nColsSubplot,
                               figsize = (3*nColsSubplot, 3*nRowsSubplot))
        
        figTitle = 'Strain-Stress of compressions'
        if plotFit:
            figTitle += '\nfit type: ' + fitType
            if fitType in ['stressRegion', 'stressGaussian']:
                figTitle += ' - {:.0f}_{:.0f}'.format(plotSettings['plotCenters'].step, 
                                                      plotSettings['plotHW'])
            if fitType == 'nPoints':
                figTitle += (' - ' + plotSettings['plotPoints'])
        fig.suptitle(figTitle)
        
        for i in range(self.Ncomp):
            colSp = (i) % nColsSubplot
            rowSp = (i) // nColsSubplot
            if nRowsSubplot == 1:
                ax = axes[colSp]
            elif nRowsSubplot >= 1:
                ax = axes[rowSp,colSp]
                
            IC = self.listIndent[i]
            IC.plot_SS(fig, ax, plotSettings, plotFit = plotFit, fitType = fitType)
            
        fig.tight_layout()
        return(fig, axes)
    
    
    def plot_KS(self, plotSettings, fitType = 'stressRegion'):
        nColsSubplot = 5
        nRowsSubplot = ((self.Ncomp-1) // nColsSubplot) + 1
        fig, axes = plt.subplots(nRowsSubplot, nColsSubplot,
                               figsize = (3*nColsSubplot, 3*nRowsSubplot))
        figTitle = 'Tangeantial modulus'
        figTitle += '\nfit type: ' + fitType
        if fitType in ['stressRegion', 'stressGaussian']:
            figTitle += ' - {:.0f}_{:.0f}'.format(plotSettings['plotCenters'].step, 
                                                  plotSettings['plotHW'])
        if fitType == 'nPoints':
            figTitle += (' - ' + plotSettings['plotPoints'])
        fig.suptitle(figTitle)
        
        for i in range(self.Ncomp):
            colSp = (i) % nColsSubplot
            rowSp = (i) // nColsSubplot
            if nRowsSubplot == 1:
                ax = axes[colSp]
            elif nRowsSubplot >= 1:
                ax = axes[rowSp,colSp]
                
            IC = self.listIndent[i]
            IC.plot_KS(fig, ax, plotSettings, fitType = fitType)
            
        fig.tight_layout()
        return(fig, axes)
    
    
    def plot_and_save(self, plotSettings, dpi = 150, figSubDir = 'MecaAnalysis_allCells'):
        # See the definition of plotSettings below !
        suf = plotSettings['subfolder_suffix']
        if len(suf) > 0:
            suf = '_' + suf
            figSubDir = figSubDir + suf
        
        # 1.
        if plotSettings['FH(t)']:
            try:
                name = self.cellID + '_01_h(t)'
                fig, ax = self.plot_Timeseries(plotSettings)
                ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            except:
                pass
            
        # 2.
        if plotSettings['F(H)']:
            try:
                name = self.cellID + '_02_F(h)'
                fig, ax = self.plot_FH(plotSettings, plotH0 = True, plotFit = True)
                ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            except:
                pass
            
        # 3.
        if plotSettings['S(e)_stressRegion']:
            try:
                name = self.cellID + '_03-1_S(e)_stressRegion'
                fig, ax = self.plot_SS(plotSettings, plotFit = True, fitType = 'stressRegion')
                ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            except:
                pass
        if plotSettings['K(S)_stressRegion']:
            # try:
            name = self.cellID + '_03-2_K(S)_stressRegion'
            fig, ax = self.plot_KS(plotSettings, fitType = 'stressRegion')
            ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            # except:
            #     pass
            
        # 4.
        if plotSettings['S(e)_stressGaussian']:
            try:
                name = self.cellID + '_04-1_S(e)_stressGaussian'
                fig, ax = self.plot_SS(plotSettings, plotFit = True, fitType = 'stressGaussian')
                ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            except:
                pass
        if plotSettings['K(S)_stressGaussian']:
            try:
                name = self.cellID + '_04-2_K(S)_stressGaussian'
                fig, ax = self.plot_KS(plotSettings, fitType = 'stressGaussian')
                ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            except:
                pass
            
        # 5.
        if plotSettings['S(e)_nPoints']:
            try:
                name = self.cellID + '_05-1_S(e)_nPoints'
                fig, ax = self.plot_SS(plotSettings, plotFit = True, fitType = 'nPoints')
                ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            except:
                pass
        if plotSettings['K(S)_nPoints']:
            try:
                name = self.cellID + '_05-2_K(S)_nPoints'
                fig, ax = self.plot_KS(plotSettings, fitType = 'nPoints')
                ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            except:
                pass
            
            
    def exportTimeseriesWithStressStrain(self):
        Nrows = self.tsDf.shape[0]
        ts_H0 = np.full(Nrows, np.nan)
        ts_stress = np.full(Nrows, np.nan)
        ts_strain = np.full(Nrows, np.nan)
        for IC in self.listIndent:
            iStart = IC.i_tsDf + IC.jStart
            iStop = IC.i_tsDf + IC.jMax+1
            if not IC.error_bestH0:
                ts_H0[iStart:iStop].fill(IC.bestH0)
                ts_stress[iStart:iStop] = IC.stressCompr
                ts_strain[iStart:iStop] = IC.strainCompr
                
        self.tsDf['H0'] = ts_H0
        self.tsDf['Stress'] = ts_stress
        self.tsDf['Strain'] = ts_strain
        saveName = self.fileName[:-4] + '_stress-strain.csv'
        savePath = os.path.join(cp.DirDataTimeseriesStressStrain, saveName)
        self.tsDf.to_csv(savePath, sep=';', index = False)
        if cp.CloudSaving != '':
            cloudSavePath = os.path.join(cp.DirCloudTimeseriesStressStrain, saveName)
            self.tsDf.to_csv(cloudSavePath, sep=';', index = False)
    
    
    def make_mainResults(self, fitSettings):
        # fitSettings = {'doChadwickFit' : True, 'doDimitriadisFit' : False,
        #     'doStressRegionFits' : True, 'doStressGaussianFits' : True, 'doNPointsFits' : True
        #     }
        
        #### Important setting : dictColumnsMeca
        dictColumnsMeca = {'date':'',
                           'cellName':'',
                           'cellID':'',
                           'manipID':'',
                           'compNum':np.nan,
                           'compDuration':'',
                           'compStartTime':np.nan,
                           'compAbsStartTime':np.nan,
                           'compStartTimeThisDay':np.nan,
                           'initialThickness':np.nan,
                           'minThickness':np.nan,
                           'maxIndent':np.nan,
                           'previousThickness':np.nan,
                           'surroundingThickness':np.nan,
                           'surroundingDx':np.nan,
                           'surroundingDy':np.nan,
                           'surroundingDz':np.nan,
                           'validatedThickness':False, 
                           'jumpD3':np.nan,
                           'minForce':np.nan, 
                           'maxForce':np.nan, 
                           'ctFieldForce':np.nan,
                           'minStress':np.nan, 
                           'maxStress':np.nan, 
                           'minStrain':np.nan, 
                           'maxStrain':np.nan,
                           'ctFieldThickness':np.nan,
                           'ctFieldFluctuAmpli':np.nan,
                           'ctFieldMinThickness':np.nan,
                           'ctFieldMaxThickness':np.nan,
                           'ctFieldVarThickness':np.nan,
                           'ctFieldDX':np.nan,
                           'ctFieldDY':np.nan,
                           'ctFieldDZ':np.nan,
                           'bestH0':np.nan,
                           'error_bestH0':True,
                           'method_bestH0':'',
                           }
        
        if fitSettings['doChadwickFit']:
            method = 'Chadwick'
            d = {'error_'+ method : True,
                 'nbPts_'+ method : np.nan, 
                 'E_'+ method : np.nan, 
                 'ciwE_'+ method : np.nan, 
                 'H0_'+ method : np.nan, 
                 'R2_'+ method : np.nan,
                 'valid_'+ method: False,
                 }
            dictColumnsMeca = {**dictColumnsMeca, **d}
            
        if fitSettings['doDimitriadisFit']:
            method = 'Dimitriadis'
            d = {'error_'+ method : True,
                 'nbPts_'+ method : np.nan, 
                 'E_'+ method : np.nan, 
                 'ciwE_'+ method : np.nan, 
                 'H0_'+ method : np.nan, 
                 'R2_'+ method : np.nan,
                 'valid_'+ method: False,
                 }
            dictColumnsMeca = {**dictColumnsMeca, **d}   
        
        
        N = self.Ncomp
        
        results = {}
        for k in dictColumnsMeca.keys():
            results[k] = [dictColumnsMeca[k] for i in range(N)]
        
        # currentCellID = self.cellID
        ctFieldH = (self.tsDf.loc[self.tsDf['idxAnalysis'] == 0, 'D3'].values - self.DIAMETER)
        ctFieldThickness   = np.median(ctFieldH)
        ctFieldMinThickness = np.min(ctFieldH)
        ctFieldMaxThickness = np.max(ctFieldH)
        ctFieldVarThickness = np.var(ctFieldH)
        ctFieldF   = (self.tsDf.loc[self.tsDf['idxAnalysis'] == 0, 'F'].values)
        ctFieldForce   = np.median(ctFieldF)
        ctFieldFluctuAmpli = np.percentile(ctFieldH, 90) - np.percentile(ctFieldH,10)
        ctFieldDX = np.median(self.tsDf.loc[self.tsDf['idxAnalysis'] == 0, 'dx'].values)
        ctFieldDY = np.median(self.tsDf.loc[self.tsDf['idxAnalysis'] == 0, 'dy'].values)
        ctFieldDZ = np.median(self.tsDf.loc[self.tsDf['idxAnalysis'] == 0, 'dz'].values)
            
        for i in range(N):
            IC = self.listIndent[i]
            
            # Identifiers
            results['date'][i] = ufun.findInfosInFileName(self.cellID, 'date')
            results['manipID'][i] = ufun.findInfosInFileName(self.cellID, 'manipID')
            results['cellName'][i] = ufun.findInfosInFileName(self.cellID, 'cellName')
            results['cellID'][i] = self.cellID
            results['compNum'][i] = i+1
            
            # Time-related
            date_T0 = self.expDf.at[self.expDf.index.values[0], 'date_T0']
            results['compDuration'][i] = self.expDf.at[self.expDf.index.values[0], 'compression duration']
            results['compStartTime'][i] = IC.rawDf['T'].values[0]
            results['compAbsStartTime'][i] = IC.rawDf['Tabs'].values[0]
            results['compStartTimeThisDay'][i] = IC.rawDf['Tabs'].values[0] - date_T0
            
            
            # Thickness-related ( = D3-DIAMETER)
            previousMask = self.getMaskForCompression(i, task = 'previous')
            surroundingMask = self.getMaskForCompression(i, task = 'surrounding')
            previousThickness = np.median(self.tsDf.D3.values[previousMask] - self.DIAMETER)
            surroundingThickness = np.median(self.tsDf.D3.values[surroundingMask] - self.DIAMETER)
            surroundingDx = np.median(self.tsDf.dx.values[surroundingMask])
            surroundingDy = np.median(self.tsDf.dy.values[surroundingMask])
            surroundingDz = np.median(self.tsDf.dz.values[surroundingMask])
            
            results['previousThickness'][i] = previousThickness
            results['surroundingThickness'][i] = surroundingThickness
            results['surroundingDx'][i] = surroundingDx
            results['surroundingDy'][i] = surroundingDy
            results['surroundingDz'][i] = surroundingDz
            results['ctFieldDX'][i] = ctFieldDX
            results['ctFieldDY'][i] = ctFieldDY
            results['ctFieldDZ'][i] = ctFieldDZ
            results['ctFieldThickness'][i] = ctFieldThickness
            results['ctFieldMinThickness'][i] = ctFieldMinThickness
            results['ctFieldMaxThickness'][i] = ctFieldMaxThickness
            results['ctFieldVarThickness'][i] = ctFieldVarThickness
            results['ctFieldFluctuAmpli'][i] = ctFieldFluctuAmpli
            results['jumpD3'][i] = self.listJumpsD3[i]
            
            results['initialThickness'][i] = np.mean(IC.hCompr[0:3])
            results['minThickness'][i] = np.min(IC.hCompr)
            results['maxIndent'][i] = results['initialThickness'][i] - results['minThickness'][i]

            results['validatedThickness'][i] = np.min([results['initialThickness'][i],results['minThickness'][i],
                                          results['previousThickness'][i],results['surroundingThickness'][i],
                                          results['ctFieldThickness'][i]]) > 0
            
            # Force-related
            results['minForce'][i] = np.min(IC.fCompr)
            results['maxForce'][i] = np.max(IC.fCompr)
            results['ctFieldForce'][i] = ctFieldForce
            
            # Best H0 related
            results['bestH0'][i] = IC.bestH0
            results['method_bestH0'][i] = IC.method_bestH0
            results['error_bestH0'][i] = IC.error_bestH0
            
            # Strain-stress-related
            results['minStress'][i] = np.min(IC.stressCompr)
            results['maxStress'][i] = np.max(IC.stressCompr)
            results['minStrain'][i] = np.min(IC.strainCompr)
            results['maxStrain'][i] = np.max(IC.strainCompr)
            
            # Whole curve fits related
            if fitSettings['doChadwickFit'] and IC.isValidForAnalysis:
                try:
                    method = 'Chadwick'
                    results['error_'+ method][i] = IC.dictFitFH_Chadwick['error']
                    results['nbPts_'+ method][i] = IC.dictFitFH_Chadwick['nbPts']
                    results['E_'+ method][i] = IC.dictFitFH_Chadwick['E']
                    results['ciwE_'+ method][i] = IC.dictFitFH_Chadwick['ciwE']
                    results['H0_'+ method][i] = IC.dictFitFH_Chadwick['H0']
                    results['R2_'+ method][i] = IC.dictFitFH_Chadwick['R2']
                    results['valid_'+ method][i] = IC.dictFitFH_Chadwick['valid']
                except:
                    print(IC.dictFitFH_Chadwick)

                    
            if fitSettings['doDimitriadisFit'] and IC.isValidForAnalysis:
                method = 'Dimitriadis'
                results['error_'+ method][i] = IC.dictFitFH_Dimitriadis['error']
                results['nbPts_'+ method][i] = IC.dictFitFH_Dimitriadis['nbPts']
                results['E_'+ method][i] = IC.dictFitFH_Dimitriadis['E']
                results['ciwE_'+ method][i] = IC.dictFitFH_Dimitriadis['ciwE']
                results['H0_'+ method][i] = IC.dictFitFH_Dimitriadis['H0']
                results['R2_'+ method][i] = IC.dictFitFH_Dimitriadis['R2']
                results['valid_'+ method][i] = IC.dictFitFH_Dimitriadis['valid']
        
        df_mainResults = pd.DataFrame(results)
        self.df_mainResults = df_mainResults
    
    
    def make_localFitsResults(self, fitSettings):
        # fitSettings = {'doChadwickFit' : True, 'doDimitriadisFit' : False,
        #     'doStressRegionFits' : True, 'doStressGaussianFits' : True, 'doNPointsFits' : True
        #     }

        if fitSettings['doStressRegionFits']:
            df = pd.concat([IC.df_stressRegions for IC in self.listIndent], axis = 0)
            df.reset_index(drop=True, inplace=True)
            self.df_stressRegions = df
            
        if fitSettings['doStressGaussianFits']:
            df = pd.concat([IC.df_stressGaussian for IC in self.listIndent], axis = 0)
            df.reset_index(drop=True, inplace=True)
            self.df_stressGaussian = df
            
        if fitSettings['doNPointsFits']:
            df = pd.concat([IC.df_nPoints for IC in self.listIndent], axis = 0)
            df.reset_index(drop=True, inplace=True)
            self.df_nPoints = df
    
    
    #### METHODS IN DEVELOPMENT
    
    
    def getH0Df(self):
        L = []
        for IC in self.listIndent:
            L.append(IC.getH0Df())
        df = pd.concat(L, axis = 0)
        return(df)
    
    def plot_KS_smooth(self):
        nColsSubplot = 5
        nRowsSubplot = ((self.Ncomp-1) // nColsSubplot) + 1
        fig, ax = plt.subplots(1,1, figsize = (7,5))
        figTitle = 'Tangeantial modulus'
        fig.suptitle(figTitle)
        
        listArraysStrain = []
        listArraysStress = []
        listArraysK = []
        for i in range(self.Ncomp):
            IC = self.listIndent[i]
            if IC.computed_SSK_filteredDer:
                listArraysStress.append(IC.SSK_filteredDer[:, 0])
                listArraysStrain.append(IC.SSK_filteredDer[:, 1])
                listArraysK.append(IC.SSK_filteredDer[:, 2])
        
        stress = np.concatenate(listArraysStress)
        strain = np.concatenate(listArraysStrain)
        K = np.concatenate(listArraysK)
        
        mat_ssk = np.array([stress, strain, K]).T

        mat_ssk_stressSorted = mat_ssk[mat_ssk[:, 0].argsort()]

        stress_sorted = mat_ssk_stressSorted[:, 0]
        strain_sorted = mat_ssk_stressSorted[:, 1]
        K_sorted = mat_ssk_stressSorted[:, 2]
        
        it = 1
        frac = 0.2
        # delta = np.max(stress_sorted) * 0.0075
        SK_smoothed = sm.nonparametric.lowess(exog=stress_sorted, endog=K_sorted, 
                                              frac=frac, it=it)

        ax.plot(stress_sorted, K_sorted, c = 'g', ls = '', marker = 'o', markersize = 2)
        ax.plot(SK_smoothed[:, 0], SK_smoothed[:, 1], c = 'r', ls = '-')
            
        fig.tight_layout()
        return(fig, ax)
    
    

class IndentCompression:
    """
    This class deals with all that is done on a single compression.
    """
    
    def __init__(self, CC, indentDf, thisExpDf, i_indent, i_tsDf):
        self.rawDf = indentDf
        self.thisExpDf = thisExpDf
        self.i_indent = i_indent
        self.i_tsDf = i_tsDf
        
        self.rawT0 = self.rawDf['T'].values[0]
        
        self.cellID = CC.cellID
        self.DIAMETER = CC.DIAMETER
        self.EXPTYPE = CC.EXPTYPE
        self.normalField = CC.normalField
        self.minCompField = CC.minCompField
        self.maxCompField = CC.maxCompField
        self.loopStruct = CC.loopStruct
        self.nUplet = CC.nUplet
        self.loop_totalSize = CC.loop_totalSize
        self.loop_rampSize = CC.loop_rampSize
        self.loop_ctSize = CC.loop_ctSize
        
        # These fields are to be modified or filled by methods later on
        
        # validateForAnalysis()
        self.isValidForAnalysis = False
        
        # refineStartStop()
        self.isRefined = False
        self.jMax = np.argmax(self.rawDf.B)
        self.jStart = 0
        self.jStop = len(self.rawDf.D3.values)
        self.hCompr = (self.rawDf.D3.values[:self.jMax+1] - self.DIAMETER)
        self.hRelax = (self.rawDf.D3.values[self.jMax+1:] - self.DIAMETER)
        self.fCompr = (self.rawDf.F.values[:self.jMax+1])
        self.fRelax = (self.rawDf.F.values[self.jMax+1:])
        self.TCompr = (self.rawDf['T'].values[:self.jMax+1])
        self.TRelax = (self.rawDf['T'].values[self.jMax+1:])
        self.BCompr = (self.rawDf.B.values[:self.jMax+1])
        self.BRelax = (self.rawDf.B.values[self.jMax+1:])
        
        self.Df = self.rawDf
        
        # computeH0()
        self.dictH0 = {}
        
        # setBestH0()
        self.bestH0 = np.nan
        self.method_bestH0 = ''
        self.zone_bestH0 = ''
        self.error_bestH0 = True
        
        # computeStressStrain()
        self.deltaCompr = np.zeros_like(self.hCompr)*np.nan
        self.stressCompr = np.zeros_like(self.hCompr)*np.nan
        self.strainCompr = np.zeros_like(self.hCompr)*np.nan
        
        # fitFH_Chadwick() & fitFH_Dimitriadis()
        self.dictFitFH_Chadwick = {}
        self.dictFitFH_Dimitriadis = {}
        
        # fitSS_stressRegion() & fitSS_stressGaussian() & fitSS_nPoints()
        self.dictFitsSS_stressRegions = {}
        self.dictFitsSS_stressGaussian = {}
        self.dictFitsSS_nPoints = {}
        
        # dictFits_To_DataFrame()
        self.df_stressRegions = pd.DataFrame({})
        self.df_stressGaussian = pd.DataFrame({})
        self.df_nPoints = pd.DataFrame({})
        
        
        
        # test
        self.computed_SSK_filteredDer = False

    
        
    def validateForAnalysis(self):
        listB = self.rawDf.B.values
        
        # Test to check if most of the compression have not been deleted due to bad image quality 
        highBvalues = (listB > (self.maxCompField +self.minCompField)/2)
        N_highBvalues = np.sum(highBvalues)
        testHighVal = (N_highBvalues > 20)

        # Test to check if the range of B field is large enough
        minB, maxB = min(listB), max(listB)
        testRangeB = ((maxB-minB) > 0.7*(self.maxCompField - self.minCompField))
        thresholdB = (maxB-minB)/50
        thresholdDeltaB = (maxB-minB)/400

        # Is the curve ok to analyse ?
        isValidForAnalysis = testHighVal and testRangeB # Some criteria can be added here
        self.isValidForAnalysis = isValidForAnalysis
        
        return(isValidForAnalysis)
    
    
    
    
    def refineStartStop(self):
        listB = self.rawDf.B.values
        NbPtsRaw = len(listB)
        
        # Correct for bugs in the B data
        for k in range(1,len(listB)):
            B = listB[k]
            if B > 1.25*self.maxCompField:
                listB[k] = listB[k-1]

        offsetStart, offsetStop = 0, 0
        minB, maxB = min(listB), max(listB)
        thresholdB = (maxB-minB)/50
        thresholdDeltaB = (maxB-minB)/400 # NEW CONDITION for the beginning of the compression : 
        # remove the first points where the steps in B are very very small

        k = 0
        while (listB[k] < minB+thresholdB) or (listB[k+1]-listB[k] < thresholdDeltaB):
            offsetStart += int((listB[k] < minB+thresholdB) or ((listB[k+1]-listB[k]) < thresholdDeltaB))
            k += 1

        k = 0
        while (listB[-1-k] < minB+thresholdB):
            offsetStop += int(listB[-1-k] < minB+thresholdB)
            k += 1

        jMax = np.argmax(self.rawDf.B) # End of compression, beginning of relaxation

        #
        hCompr_raw = (self.rawDf.D3.values[:jMax+1] - self.DIAMETER)

        # Refinement of the compression delimitation.
        # Remove the 1-2 points at the begining where there is just the viscous relaxation of the cortex
        # because of the initial decrease of B and the cortex thickness increases.
        
        NptsCompr = len(hCompr_raw)
        k = offsetStart
        while (k<(NptsCompr//2)) and (hCompr_raw[k] < np.max(hCompr_raw[k+1:min(k+10, NptsCompr)])):
            k += 1
        offsetStart = k
        
        jStart = offsetStart # Beginning of compression
        jStop = self.rawDf.shape[0] - offsetStop # End of relaxation
        
        # Better compressions arrays
        self.hCompr = (self.rawDf.D3.values[jStart:jMax+1] - self.DIAMETER)
        self.hRelax = (self.rawDf.D3.values[jMax+1:jStop] - self.DIAMETER)
        self.fCompr = (self.rawDf.F.values[jStart:jMax+1])
        self.fRelax = (self.rawDf.F.values[jMax+1:jStop])
        self.BCompr = (self.rawDf.B.values[jStart:jMax+1])
        self.BRelax = (self.rawDf.B.values[jMax+1:jStop])
        self.TCompr = (self.rawDf['T'].values[jStart:jMax+1])
        self.TRelax = (self.rawDf['T'].values[jMax+1:jStop])
        
        self.jMax = jMax
        self.jStart = jStart
        self.jStop = jStop

        mask = np.array([((j >= jStart) and (j < jStop)) for j in range(NbPtsRaw)])
        
        self.Df = self.rawDf.loc[mask]
        self.isRefined = True
        
        
        
        
    def computeH0(self, method = 'Chadwick', zone = 'pts_15'):
        listAllMethods = ['Chadwick', 'Dimitriadis', 'NaiveMax']
        listAllZones = ['pts_15', 'pts_30', 
                        '%f_10', '%f_20', '%f_30', '%f_40', 
                        '%h_10', '%h_20', '%h_30', '%h_40']
        

        if method == 'all' and zone != 'all':
            for m in listAllMethods:
                self.computeH0(method = m, zone = zone)
        elif method != 'all' and zone == 'all':
            for z in listAllZones:
                self.computeH0(method = method, zone = z)
        elif method == 'all' and zone == 'all':
            for m in listAllMethods:
                if m != 'NaiveMax':
                    for z in listAllZones:
                        self.computeH0(method = m, zone = z)
                else:
                    self.computeH0(method = m, zone = '^_^')
                    
        else:
            
            [zoneType, zoneVal] = zone.split('_')
            
            if zoneType == 'pts':
                i1 = 0
                i2 = int(zoneVal)
                mask = np.array([((i >= i1) and (i < i2)) for i in range(len(self.fCompr))])    
            elif zoneType == '%f':
                pseudoForce = self.fCompr - np.min(self.fCompr)
                thresh = (int(zoneVal)/100.) * np.max(pseudoForce)
                mask = (pseudoForce < thresh)
            elif zoneType == '%h':
                pseudoDelta = np.max(self.hCompr) - self.hCompr
                thresh = (int(zoneVal)/100.) * np.max(pseudoDelta)
                mask = (pseudoDelta < thresh)
            
            if method == 'Chadwick':
                h, f, D = self.hCompr[mask], self.fCompr[mask], self.DIAMETER
                params, covM, error = fitChadwick_hf(h, f, D)
                H0, E = params[1], params[0]
                self.dictH0['H0_' + method + '_' + zone] = H0
                self.dictH0['E_' + method + '_' + zone] = E
                self.dictH0['error_' + method + '_' + zone] = error
                self.dictH0['nbPts_' + method + '_' + zone] = np.sum(mask)
                self.dictH0['fArray_' + method + '_' + zone] = f
                self.dictH0['hArray_' + method + '_' + zone] = inversedChadwickModel(f, E, H0/1000, D/1000)*1000
                
            elif method == 'Dimitriadis':
                h, f, D = self.hCompr[mask], self.fCompr[mask], self.DIAMETER
                params, covM, error = fitDimitriadis_hf(h, f, D)
                H0, E = params[1], params[0]
                self.dictH0['H0_' + method + '_' + zone] = H0
                self.dictH0['E_' + method + '_' + zone] = E
                self.dictH0['error_' + method + '_' + zone] = error
                self.dictH0['nbPts_' + method + '_' + zone] = np.sum(mask)
                self.dictH0['fArray_' + method + '_' + zone] = dimitriadisModel(h/1000, E, H0/1000, D/1000)
                self.dictH0['hArray_' + method + '_' + zone] = h
                
            elif method == 'NaiveMax':
                H0 = np.max(self.hCompr[:])
                self.dictH0['H0_' + method] = H0
                self.dictH0['E_' + method] = np.nan
                self.dictH0['error_' + method] = False
                self.dictH0['fArray_' + method] = np.array([])
                self.dictH0['hArray_' + method] = np.array([])
        
            
    def setBestH0(self, method, zone):
        self.bestH0 = self.dictH0['H0_' + method + '_' + zone]
        self.error_bestH0 = self.dictH0['error_' + method + '_' + zone]
        self.method_bestH0 = method
        self.zone_bestH0 = zone
        
        
    def getH0Df(self):
        d = {'CompNum':[],
             'method':[],
             'zone':[],
             'H0':[],
             'nbPts':[],
             'error':[],
             }
        
        for k in self.dictH0.keys():
            if k.startswith('H0'):
                infos = k.split('_') # k = method_zoneType_zoneVal
                if infos[1] in ['Chadwick', 'Dimitriadis']:
                    CompNum = self.i_indent
                    method = infos[1]
                    zone = infos[2] + '_' + infos[3]
                    H0 = self.dictH0[k]
                    nbPts = int(self.dictH0['nbPts_' + method + '_' + zone])
                    error = bool(self.dictH0['error_' + method + '_' + zone])
                    d['CompNum'].append(CompNum)
                    d['method'].append(method)
                    d['zone'].append(zone)
                    d['H0'].append(H0)
                    d['nbPts'].append(nbPts)
                    d['error'].append(error)
                    
        for k in d.keys():
            d[k] = np.array(d[k])
                    
        df = pd.DataFrame(d)
        return(df)
    
    
    def computeStressStrain(self, method = 'Chadwick'):
        if not self.error_bestH0:
            deltaCompr = (self.bestH0 - self.hCompr)
            
            if method == 'Chadwick':
                # pN and µm
                stressCompr = self.fCompr / (np.pi * (self.DIAMETER/2000) * deltaCompr/1000)
                strainCompr = (deltaCompr/1000) / (3*(self.bestH0/1000))
                
            self.deltaCompr = deltaCompr
            self.stressCompr = stressCompr
            self.strainCompr = strainCompr
        
    
    def fitFH_Chadwick(self, dictFitValidation, mask = []):
        if len(mask) == 0:
            mask = np.ones_like(self.hCompr, dtype = bool)
        h, f, D = self.hCompr[mask], self.fCompr[mask], self.DIAMETER
        params, ses, error = fitChadwick_hf(h, f, D)
        E, H0 = params
        hPredict = inversedChadwickModel(f, E, H0/1000, self.DIAMETER/1000)*1000
        x = f
        y, yPredict = h, hPredict
        #### err_Chi2 for distance (nm)
        err_chi2 = 30
        dictFit = makeDictFit_hf(params, ses, error, 
                                 x, y, yPredict, 
                                 err_chi2, dictFitValidation)
        self.dictFitFH_Chadwick = dictFit

                
    def fitFH_Dimitriadis(self, dictFitValidation, mask = []):
        if len(mask) == 0:
            mask = np.ones_like(self.hCompr, dtype = bool)
        h, f, D = self.hCompr[mask], self.fCompr[mask], self.DIAMETER
        params, ses, error = fitDimitriadis_hf(h, f, D)
        E, H0 = params
        fPredict = dimitriadisModel(h/1000, E, H0/1000, self.DIAMETER/1000, v = 0)
        x = h
        y, yPredict = f, fPredict
        #### err_Chi2 for force (pN)
        err_chi2 = 100
        dictFit = makeDictFit_hf(params, ses, error, 
                           x, y, yPredict, 
                           err_chi2, dictFitValidation)
        self.dictFitFH_Dimitriadis = dictFit
                
    
    def fitSS_stressRegion(self, center, halfWidth, dictFitValidation):
        id_range = str(center) + '_' + str(halfWidth)
        stress, strain = self.stressCompr, self.strainCompr
        
        lowS, highS = center - halfWidth, center + halfWidth
        mask = ((stress > lowS) & (stress < highS))
        
        params, ses, error = fitLinear_ss(stress, strain, weights = mask)
        
        K, strain0 = params
        strainPredict = inversedConstitutiveRelation(stress[mask], K, strain0)
        
        x = stress[mask]
        y, yPredict = strain[mask], strainPredict
        #### err_Chi2 for strain
        err_chi2 = 0.01
        
        dictFit = makeDictFit_ss(params, ses, error, 
                                 center, halfWidth, x, y, yPredict, 
                                 err_chi2, dictFitValidation)
        self.dictFitsSS_stressRegions[id_range] = dictFit  
                
    
    def fitSS_stressGaussian(self, center, halfWidth, dictFitValidation):
        id_range = str(center) + '_' + str(halfWidth)
        stress, strain = self.stressCompr, self.strainCompr
        
        X = stress.flatten(order='C')
        weights = np.exp( -((X - center) ** 2) / halfWidth ** 2)
        
        params, ses, error = fitLinear_ss(stress, strain, weights = weights)
        
        K, strain0 = params
        lowS, highS = center - halfWidth, center + halfWidth
        mask = ((stress > lowS) & (stress < highS))
        strainPredict = inversedConstitutiveRelation(stress[mask], K, strain0)
        
        x = stress[mask]
        y, yPredict = strain[mask], strainPredict
        #### err_Chi2 for strain
        err_chi2 = 0.01
        
        dictFit = makeDictFit_ss(params, ses, error, 
                                 center, halfWidth, x, y, yPredict, 
                                 err_chi2, dictFitValidation)
        self.dictFitsSS_stressGaussian[id_range] = dictFit
    
    
    def fitSS_nPoints(self, mask, dictFitValidation):
        iStart = ufun.findFirst(1, mask)
        iStop = iStart + np.sum(mask)
        id_range = str(iStart) + '_' +str(iStop)
        stress, strain = self.stressCompr, self.strainCompr
        params, ses, error = fitLinear_ss(stress, strain, weights = mask)
        
        K, strain0 = params
        strainPredict = inversedConstitutiveRelation(stress[mask], K, strain0)
        
        x = stress[mask]
        y, yPredict = strain[mask], strainPredict
        #### err_Chi2 for strain
        err_chi2 = 0.01
        center = np.median(x)
        halfWidth = (np.max(x) - np.min(x))/2
        
        dictFit = makeDictFit_ss(params, ses, error, 
                                 center, halfWidth, x, y, yPredict, 
                                 err_chi2, dictFitValidation)
        self.dictFitsSS_nPoints[id_range] = dictFit
    
    
    def dictFits_To_DataFrame(self, fitSettings):
        if fitSettings['doStressRegionFits']:
            df = nestedDict_to_DataFrame(self.dictFitsSS_stressRegions)
            nRows = df.shape[0]
            df.insert(0, 'compNum', np.ones(nRows) * (self.i_indent+1))
            df.insert(0, 'cellID', [self.cellID for i in range(nRows)])
            self.df_stressRegions = df
            
        if fitSettings['doStressGaussianFits']:
            df = nestedDict_to_DataFrame(self.dictFitsSS_stressGaussian)
            nRows = df.shape[0]
            df.insert(0, 'compNum', np.ones(nRows) * (self.i_indent+1))
            df.insert(0, 'cellID', [self.cellID for i in range(nRows)])
            self.df_stressGaussian = df
            
        if fitSettings['doNPointsFits']:
            df = nestedDict_to_DataFrame(self.dictFitsSS_nPoints)
            nRows = df.shape[0]
            df.insert(0, 'compNum', np.ones(nRows) * (self.i_indent+1))
            df.insert(0, 'cellID', [self.cellID for i in range(nRows)])
            self.df_nPoints = df
    
    
    
    def plot_FH(self, fig, ax, plotSettings, plotH0 = True, plotFit = True):
        if self.isValidForAnalysis:
            ax.plot(self.hCompr, self.fCompr,'b-', linewidth = 0.8)
            ax.plot(self.hRelax, self.fRelax,'r-', linewidth = 0.8)
            titleText = self.cellID + '__c' + str(self.i_indent + 1)
            legendText = ''
            ax.set_xlabel('h (nm)')
            ax.set_ylabel('f (pN)')
    
            if plotFit:
                dictFit = self.dictFitFH_Chadwick
                fitError = dictFit['error']
                    
                if not fitError:
                    H0, E, R2, Chi2 = dictFit['H0'], dictFit['E'], dictFit['R2'], dictFit['Chi2']
                    hPredict = dictFit['yPredict']
                    
                    legendText += 'H0 = {:.1f}nm\nE = {:.2e}Pa\nR2 = {:.3f}\nChi2 = {:.1f}'.format(H0, E, R2, Chi2)
                    ax.plot(hPredict, self.fCompr,'k--', linewidth = 0.8, 
                            label = legendText, zorder = 2)
                    ax.legend(loc = 'upper right', prop={'size': 6})
                else:
                    titleText += '\nFIT ERROR'
                    
            if plotH0:
                bestH0 = self.bestH0
                method = self.method_bestH0
                zone = self.zone_bestH0
                str_m_z = method + '_' + zone
                E_bestH0 = self.dictH0['E_' + method + '_' + zone]
                
                if (not self.error_bestH0) and (method not in ['NaiveMax']):
                    max_h = np.max(self.hCompr)
                    high_h = np.linspace(max_h, bestH0, 20)
                    low_f = dimitriadisModel(high_h/1000, E_bestH0, bestH0/1000, self.DIAMETER/1000)
                    
                    legendText = 'bestH0 = {:.2f}nm'.format(bestH0) + '\n' + str_m_z
                    plot_startH = np.concatenate((self.dictH0['hArray_' + str_m_z][::-1], high_h))
                    plot_startF = np.concatenate((self.dictH0['fArray_' + str_m_z][::-1], low_f))

                    ax.plot([bestH0], [0], ls = '', marker = 'o', color = 'skyblue', markersize = 5, 
                            label = legendText)
                    ax.plot(plot_startH, plot_startF, ls = '--', color = 'skyblue', linewidth = 1.2, zorder = 4)
                    ax.legend(loc = 'upper right', prop={'size': 6})
            
                ax.title.set_text(titleText)
                for item in ([ax.title, ax.xaxis.label, \
                              ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(9)
            
            
        
    def plot_SS(self, fig, ax, plotSettings, plotFit = True, fitType = 'stressRegion'):
        if self.isValidForAnalysis and not self.error_bestH0:
            titleText = self.cellID + '__c' + str(self.i_indent + 1)
            ax.title.set_text(titleText)
            ax.set_xlabel('Strain')
            ax.set_ylabel('Stress (Pa)')
            main_color = 'k'
            ls = '-'
            lw = 1.8
            
            if not self.error_bestH0:
                ax.plot(self.strainCompr, self.stressCompr, 
                        color = main_color, marker = 'o', 
                        markersize = 2, ls = '', alpha = 0.8)
                
                if plotFit:
                    if fitType == 'stressRegion':
                        # Read settings
                        HW = plotSettings['plotHW']
                        centers = plotSettings['plotCenters']
                        dictFit = self.dictFitsSS_stressRegions
                        id_ranges = [str(C) + '_' + str(HW) for C in centers]
                        # colorDict = {id_ranges[k]:gs.colorList30[k] for k in range(len(id_ranges))}
                        for k in range(len(id_ranges)):
                            idr = id_ranges[k]
                            d = dictFit[idr]
                            if not d['error']:
                                if not d['valid']:
                                    ls = '--'
                                x = d['x']
                                y = d['yPredict']
                                color = gs.colorList30[k]
                                ax.plot(y, x, color = color, ls = ls, lw = lw)
                                
                    if fitType == 'stressGaussian':
                        # Read settings
                        HW = plotSettings['plotHW']
                        centers = plotSettings['plotCenters']
                        dictFit = self.dictFitsSS_stressGaussian
                        id_ranges = [str(C) + '_' + str(HW) for C in centers]
                        # colorDict = {id_ranges[k]:gs.colorList30[k] for k in range(len(id_ranges))}
                        for k in range(len(id_ranges)):
                            idr = id_ranges[k]
                            d = dictFit[idr]
                            if not d['error']:
                                if not d['valid']:
                                    ls = '--'
                                x = d['x']
                                y = d['yPredict']
                                color = gs.colorList30[k]
                                ax.plot(y, x, color = color, ls = ls, lw = lw)
                                
                    if fitType == 'nPoints':
                        dictFit = self.dictFitsSS_nPoints
                        id_ranges = list(dictFit.keys())
                        # colorDict = {id_ranges[k]:gs.colorList30[k] for k in range(len(id_ranges))}
                        for k in range(len(id_ranges)):
                            idr = id_ranges[k]
                            d = dictFit[idr]
                            if not d['error']:
                                if not d['valid']:
                                    ls = '--'
                                x = d['x']
                                y = d['yPredict']
                                color = gs.colorList30[k]
                                ax.plot(y, x, color = color, ls = ls, lw = lw)
                    
    
                
                for item in ([ax.title, ax.xaxis.label, \
                              ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(9)
    
        
    def plot_KS(self, fig, ax, plotSettings, fitType = 'stressRegion'):
        if self.isValidForAnalysis and not self.error_bestH0:
            titleText = self.cellID + '__c' + str(self.i_indent + 1)
            ax.title.set_text(titleText)
            ax.set_ylabel('K (kPa)')
            ax.set_ylim([0, 16])
            
            if fitType == 'stressRegion':
                df = self.df_stressRegions
                ax.set_xlabel('Stress (Pa)')
                ax.set_xlim([0, 1200])
                
                # Read settings
                HW = plotSettings['plotHW']
                centers = plotSettings['plotCenters']
                N = len(centers)
                colors = gs.colorList30[:N]
                
                fltr = (df['center_x'].apply(lambda x : x in centers)) & \
                       (df['halfWidth_x'] == HW)
                
                df_fltr = df[fltr]

                for k in range(N):
                    X = df_fltr['center_x'].values[k]
                    Y = df_fltr['K'].values[k]/1000
                    Yerr = df_fltr['ciwK'].values[k]/1000
                    if df_fltr['valid'].values[k]:
                        color = colors[k]
                        mec = 'none'
                    else:
                        color = 'w'
                        mec = colors[k]
                    if (not pd.isnull(Y)) and (Y > 0):
                        ax.errorbar(X, Y, yerr = Yerr, color = color, marker = 'o', ms = 5, mec = mec, ecolor = colors[k]) 
            
            if fitType == 'stressGaussian':
                df = self.df_stressGaussian
                ax.set_xlabel('Stress (Pa)')
                ax.set_xlim([0, 1200])
                
                # Read settings
                HW = plotSettings['plotHW']
                centers = plotSettings['plotCenters']
                N = len(centers)
                colors = gs.colorList30[:N]
                
                fltr = (df['center_x'].apply(lambda x : x in centers)) & \
                       (df['halfWidth_x'] == HW)
                
                df_fltr = df[fltr]
                # relativeError[k] = (Err/K_fit)
                # # mec = None
                for k in range(N):
                    X = df_fltr['center_x'].values[k]
                    Y = df_fltr['K'].values[k]/1000
                    Yerr = df_fltr['ciwK'].values[k]/1000
                    if df_fltr['valid'].values[k]:
                        color = colors[k]
                        mec = 'none'
                    else:
                        color = 'w'
                        mec = colors[k]
                    if (not pd.isnull(Y)) and (Y > 0):
                        ax.errorbar(X, Y, yerr = Yerr, color = color, marker = 'o', ms = 5, mec = mec, ecolor = colors[k]) 
                
            if fitType == 'nPoints':
                df = self.df_nPoints
                ax.set_xlabel('Stress (Pa)')
                ax.set_xlim([0, 1200])
    
                N = df.shape[0]
                colors = gs.colorList30[:N]

                for k in range(N):
                    X = df['center_x'].values[k]
                    Y = df['K'].values[k]/1000
                    Yerr = df['ciwK'].values[k]/1000
                    if df['valid'].values[k]:
                        color = colors[k]
                        mec = 'none'
                    else:
                        color = 'w'
                        mec = colors[k]
                    if (not pd.isnull(Y)) and (Y > 0):
                        ax.errorbar(X, Y, yerr = Yerr, color = color, marker = 'o', 
                                    ms = 5, mec = mec, ecolor = colors[k])
            
            
            for item in ([ax.title, ax.xaxis.label, \
                          ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(9)
            
            
    
    #### METHODS IN DEVELOPMENT
            
    
    def fitSS_polynomial(self):
        T, B, F =  self.TCompr - self.rawT0, self.BCompr, self.fCompr
        stress, strain = self.stressCompr, self.strainCompr
        fake_strain = np.linspace(np.min(strain), np.max(strain), 100)
        fake_stress = np.linspace(np.min(stress), np.max(stress), 100)
        titleText = self.cellID + '__c' + str(self.i_indent + 1)
        
        if not self.error_bestH0:
            fig, axes = plt.subplots(1,5, figsize = (15,4))
            fig.suptitle(titleText, fontsize = 12)
            
            # 1. 
            ax = axes[0]
            ax.set_xlabel('T (s)')
            ax.set_ylabel('B (mT)')
            
            rawT = (self.rawDf['T'].values[:self.jMax+1])
            rawT = rawT-rawT[0]
            rawB = (self.rawDf.B.values[:self.jMax+1])
            rawF = self.rawDf['F'].values[:self.jMax+1]

            ax.plot(T, B, 
                    color = 'fuchsia', marker = 'o', 
                    markersize = 2, ls = '', alpha = 0.8)
            
            axbis = ax.twinx()
            
            axbis.plot(T, F, 
                    color = 'red', marker = 'o', 
                    markersize = 2, ls = '', alpha = 0.8)
            
            # ax.set_xscale('log')
            # ax.set_yscale('log')
            # axbis.set_yscale('log')
            axbis.set_ylabel('F (pN)')
            
            
            B0 = 1
            Bmax = np.max(rawB)
            T0 = 0
            Tmax = np.max(rawT)
            Bt2 = B0 + ((Bmax-B0)/(Tmax-T0)**2) * (T-T0)**2
            Bt3 = B0 + ((Bmax-B0)/(Tmax-T0)**3) * (T-T0)**3
            Bt4 = B0 + ((Bmax-B0)/(Tmax-T0)**4) * (T-T0)**4
            iStart = 0
            ols_model = sm.OLS(np.log(B[:]-B0), sm.add_constant(np.log(T[:])))
            results_ols = ols_model.fit()
            params = results_ols.params[::-1]
            # print(params)
            # params, covM = np.polyfit(np.log(T), np.log(B), 1, cov=True)
            # BPredict = B0 + np.exp(params[0]) * (T[iStart:]**(params[1]))
            BPredict = (T[:]**(params[1]))
            
            ax.plot(T, Bt2, 
                    color = gs.colorList40[16], ls = '-', lw = 0.8)
            
            ax.plot(T, Bt3, 
                    color = gs.colorList40[26], ls = '--', lw = 0.8)
            
            ax.plot(T, Bt4, 
                    color = gs.colorList40[36], ls = '-.', lw = 0.8)
            
            ax.set_xlim([0.1, 2.0])
            axbis.set_xlim([0.1, 2.0])
            
            
            # 2.
            ax = axes[1]
            
            params, covM = np.polyfit(strain, stress, 3, cov=True) # , rcond=None, full=False, w=None
            X = np.linspace(np.min(strain), np.max(strain), 100)
            YPredict = np.polyval(params, X)
            
            ax.set_xlabel('Strain')
            ax.set_ylabel('Stress (Pa)')
            main_color = 'k'
            
            ax.plot(strain, stress, 
                    color = main_color, marker = 'o', 
                    markersize = 2, ls = '', alpha = 0.8)
                
            ax.plot(X, YPredict, 
                    color = 'r', ls = '-')
            
            poly5_SS = np.poly1d(params)
            der_poly5_SS = np.polyder(poly5_SS)
            der_YPredict = np.polyval(der_poly5_SS, fake_strain)
            
            ax = axes[2]
            
            ax.plot(fake_stress, der_YPredict/1000,
                         color = 'b', ls = '-')
            
            
            ax.set_xlabel('Stress (Pa)')
            ax.set_ylabel('K (kPa)')
            main_color = 'k'
            
            
        
            # 3.
            # print(T)
            # print(strain)
            
            ax = axes[3]
            
            params, covM = np.polyfit(T, strain, 5, cov=True) # , rcond=None, full=False, w=None
            YPredict = np.polyval(params, T)
            
            titleText = self.cellID + '__c' + str(self.i_indent + 1)
            ax.set_xlabel('T (s)')
            ax.set_ylabel('Strain')
            main_color = 'k'
            
            ax.plot(T, strain, 
                    color = main_color, marker = 'o', 
                    markersize = 2, ls = '', alpha = 0.8)
                
            ax.plot(T, YPredict, 
                    color = 'r', ls = '-')
            
            poly5_TStrain = np.poly1d(params)
            der_poly5_TStrain = np.polyder(poly5_TStrain)
            der_YPredict = np.polyval(der_poly5_TStrain, T)
            
            ax.plot(T, der_YPredict, 
                    color = 'b', ls = '-')
        
            # 3.
            ax = axes[4]
            params, covM = np.polyfit(T, stress, 5, cov=True) # , rcond=None, full=False, w=None
            YPredict = np.polyval(params, T)
            
            
            titleText = self.cellID + '__c' + str(self.i_indent + 1)
            ax.set_xlabel('T (s)')
            ax.set_ylabel('Stress (Pa)')
            main_color = 'k'
            
            ax.plot(T, stress, 
                    color = main_color, marker = 'o', 
                    markersize = 2, ls = '', alpha = 0.8)
                
            ax.plot(T, YPredict, 
                    color = 'r', ls = '-')
            
            poly5_TStress = np.poly1d(params)
            der_poly5_TStress = np.polyder(poly5_TStress)
            der_YPredict = np.polyval(der_poly5_TStress, T)
            
            ax.plot(T, der_YPredict, 
                    color = 'b', ls = '-')
        
        
        for ax in axes:
            for item in ([ax.title, ax.xaxis.label, \
                          ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(9)
        ax = axbis
        for item in ([ax.title, ax.xaxis.label, \
                      ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(9)
                
        fig.tight_layout()
            
            
            
            
            
    def fitSS_smooth(self):
        if not self.error_bestH0:
            T, stress, strain = self.TCompr - self.TCompr[0], self.stressCompr, self.strainCompr
            fig, axes = plt.subplots(3,4, figsize = (16,12))
            titleText = self.cellID + '__c' + str(self.i_indent + 1)
            fig.suptitle(titleText)
            main_color = 'k'
            
            matSS = np.array([stress, strain]).T            
            matSS_stressSorted = matSS[matSS[:, 0].argsort()]
            stress_sorted = matSS_stressSorted[:, 0]
            strain_sorted = matSS_stressSorted[:, 1]
            
            
            it = 3
            frac = 0.2
            delta = np.max(stress) * 0.0075
            smoothed = sm.nonparametric.lowess(exog=stress, endog=strain, 
                                               frac=frac, it=it, delta=delta)
            smoothed_c = np.copy(smoothed)
            for ii in range(1, len(smoothed)):
                if smoothed_c[ii, 1] < smoothed_c[ii-1, 1]:
                    smoothed_c[ii, 1] = smoothed_c[ii-1, 1] + 1e-9
                    
            smoothed_c_der = np.zeros((smoothed_c.shape[0], 3), dtype = np.float64)
            smoothed_c_der[:, :2] = smoothed_c
            for ii in range(1, len(smoothed_c)):
                smoothed_c_der[ii, 2] = (smoothed_c[ii, 0] - smoothed_c[ii-1, 0]) / (smoothed_c[ii, 1] - smoothed_c[ii-1, 1])
            
            mask = (smoothed_c_der[:,2] > 0) & (smoothed_c_der[:,2] < 20e3)
            smoothed_c_der = smoothed_c_der[mask]
            
            # print(smoothed_c_der)
            
            self.computed_SSK_filteredDer = True
            self.SSK_filteredDer = smoothed_c_der
            
            
            
            # 1.
            
            matSS = np.array([stress, strain]).T            
            matSS_sorted = matSS[matSS[:, 0].argsort()]        
            
            window_length = [11, 21, 41]
            polyorder = [5, 5, 5]
            
            for k in range(3):
                
                wl, po = window_length[k], polyorder[k]
                
                axes[k,0].set_xlabel('Strain')
                axes[k,0].set_ylabel('Stress (Pa)')
            
                axes[k,0].plot(matSS_sorted[:, 1], matSS_sorted[:, 0], 
                        color = main_color, marker = 'o', 
                        markersize = 2, ls = '', alpha = 0.8)
            
                mode = 'interp'
                yPredict = savgol_filter(matSS_sorted[:, 1], wl, po, mode=mode)
                
                axes[k,0].plot(yPredict, matSS_sorted[:, 0],
                        color = 'r', ls = '-')
                
                axes[k,0].set_title('Savgol smoothing\nlen = {:.0f} - deg = {:.0f}'.format(wl, po))
            
            # 2.
            its = [0, 1, 3]
            fracs = [0.2, 0.2, 0.2]
            
            for k in range(3):
                it = its[k]
                frac = fracs[k]
                delta = np.max(stress) * 0.0075
                smoothed = sm.nonparametric.lowess(exog=stress, endog=strain, 
                                                    frac=frac, it=it, delta=delta)
                smoothed_c = np.copy(smoothed)
                for ii in range(1, len(smoothed)):
                    if smoothed_c[ii, 1] < smoothed_c[ii-1, 1]:
                        smoothed_c[ii, 1] = smoothed_c[ii-1, 1] + 1e-9
                        
                smoothed_c_der = np.zeros((smoothed_c.shape[0], 3), dtype = np.float64)
                smoothed_c_der[:, :2] = smoothed_c
                for ii in range(1, len(smoothed_c)):
                    smoothed_c_der[ii, 2] = (smoothed_c[ii, 0] - smoothed_c[ii-1, 0]) / (smoothed_c[ii, 1] - smoothed_c[ii-1, 1])
                    
                # print(smoothed_c_der)

                # print(smoothed)
                
                axes[k,1].set_xlabel('Strain')
                axes[k,1].set_ylabel('Stress (Pa)')
                axes[k,1].set_title('Lowess smoothing')
                
                axes[k,1].plot(strain, stress, 
                        color = main_color, marker = 'o', 
                        markersize = 2, ls = '', alpha = 0.8)
                
                axes[k,1].plot(smoothed_c[:, 1], smoothed_c[:, 0],
                              color = 'b', ls = '-')
                axes[k,1].plot(smoothed[:, 1], smoothed[:, 0],
                              color = 'r', ls = '-')
                
                axes[k,2].set_xlabel('Stress (Pa)')
                axes[k,2].set_ylabel('K (Pa)')
                # axes[k,2].set_title('')
                axes[k,2].plot(smoothed_c_der[1:, 0], smoothed_c_der[1:, 2],
                              color = 'g', ls = '', marker = 'o', markersize = 2)
                
                mask = (smoothed_c_der[:,2] > 0) & (smoothed_c_der[:,2] < 20e3)
                smoothed_c_der = smoothed_c_der[mask]
                axes[k,2].set_ylim([0, 20000])
                
                self.SSK_filteredDer = smoothed_c_der

                
                axes[k,1].set_title('Lowess smoothing\nfrac = {:.2f} - it = {:.1f}'.format(frac, it))
              
                
            # 3.
            sfacts = [0.0005,0.001,0.01]
            for k in range(3):
                sfact = sfacts[k]
                sp = si.splrep(stress_sorted, strain_sorted, s=sfact, k=5)
                strain_smoothed = si.splev(stress_sorted, sp)
                
                axes[k,3].set_xlabel('Strain')
                axes[k,3].set_ylabel('Stress (Pa)')
                axes[k,3].set_title('Spline smoothing')
                
                axes[k,3].plot(strain_sorted, stress_sorted, 
                        color = main_color, marker = 'o', 
                        markersize = 2, ls = '', alpha = 0.8)
                
                axes[k,3].plot(strain_smoothed, stress_sorted,
                              color = 'r', ls = '-')

                
                axes[k,3].set_title('Spline smoothing\ns = {:.4f}'.format(sfact))
            
            

        
# %%%% main function
        
def analyseTimeSeries_meca(f, tsDf, expDf, taskName = '', PLOT = False, SHOW = False):
    top = time.time()
    
    print(gs.BLUE + f + gs.NORMAL, end = ' ... ')
    plt.ioff()
    warnings.filterwarnings('ignore')
    
    #### 0. Settings
    #### 0.1 BestH0
    # listAllMethods = ['Chadwick', 'Dimitriadis', 'NaiveMax']
    # listAllZones = ['pts_15', 'pts_30', '%f_15', '%f_30', '%h_15', '%h_30']
    method_bestH0 = 'Dimitriadis'
    zone_bestH0 = '%f_20'
    
    #### 0.2 Fits settings
    fitSettings = {'doChadwickFit' : True,
                   'doDimitriadisFit' : False,
                   'doStressRegionFits' : True,
                   'doStressGaussianFits' : True,
                   'doNPointsFits' : True}
    
    #### 0.2.1 Options for StressRegionFits & StressGaussianFits
    centers = [ii for ii in range(100, 1550, 50)]
    halfWidths = [50, 75, 100]
    #### 0.2.2 Options for NPointsFits
    nbPtsFit = 13
    overlapFit = 3
    
    #### 0.2.3 Options for fits validation
    crit_nbPts = 8 # sup or equal to
    crit_R2 = 0.6 # sup or equal to
    crit_Chi2 = 1 # inf or equal to
    str_crit = 'nbPts>{:.0f} - R2>{:.2f} - Chi2<{:.1f}'.format(crit_nbPts, crit_R2, crit_Chi2)
    dictFitValidation = {'crit_nbPts': crit_nbPts, 
                         'crit_R2': crit_R2, 
                         'crit_Chi2': crit_Chi2,
                         'str': str_crit}
    
    
    #### 0.3 Plots
    plotSettings = {'FH(t)':True,
                    'F(H)':True,
                    'S(e)_stressRegion':True,
                    'K(S)_stressRegion':True,
                    'S(e)_stressGaussian':True,
                    'K(S)_stressGaussian':True,
                    'S(e)_nPoints':True,
                    'K(S)_nPoints':True,
                    #
                    'subfolder_suffix':taskName,
                    'plotCenters':range(100, 1500, 100),
                    'plotHW':75,
                    'plotPoints':str(nbPtsFit)+'_'+str(overlapFit)}
    
    #### 1. Import experimental infos
    tsDf.dx, tsDf.dy, tsDf.dz, tsDf.D2, tsDf.D3 = tsDf.dx*1000, tsDf.dy*1000, tsDf.dz*1000, tsDf.D2*1000, tsDf.D3*1000
    thisManipID = ufun.findInfosInFileName(f, 'manipID')
    thisExpDf = expDf.loc[expDf['manipID'] == thisManipID]
    Ncomp = max(tsDf['idxAnalysis'])
    cellID = ufun.findInfosInFileName(f, 'cellID')
    
    #### 2. Create CellCompression object
    CC = CellCompression(cellID, tsDf, thisExpDf, f)
    CC.method_bestH0 = method_bestH0

    
    #### 3. Start looping over indents
    for i in range(Ncomp):

        #### 3.1 Correct jumps
        CC.correctJumpForCompression(i)
            
        #### 3.2 Segment the i-th compression
        maskComp = CC.getMaskForCompression(i, task = 'compression')
        thisCompDf = tsDf.loc[maskComp,:]
        i_tsDf = ufun.findFirst(1, maskComp)
        
        #### 3.3 Create IndentCompression object
        IC = IndentCompression(CC, thisCompDf, thisExpDf, i, i_tsDf)
        CC.listIndent.append(IC)

        #### 3.4 State if i-th compression is valid for analysis
        doThisCompAnalysis = IC.validateForAnalysis()

        if doThisCompAnalysis:
            
            #### 3.5 Inside i-th compression, delimit the compression and relaxation phases            
            IC.refineStartStop()            
            
            #### 3.7 Fit with Chadwick model of the force-thickness curve     
            IC.fitFH_Chadwick(dictFitValidation)
                
            #### 3.8 Find the best H0
            IC.computeH0(method = method_bestH0, zone = zone_bestH0)
            # IC.computeH0(method = 'all', zone = 'all')
            # print(IC.dictH0)
            
            IC.setBestH0(method = method_bestH0, zone = zone_bestH0)

            #### 3.9 Compute stress and strain based on the best H0
            IC.computeStressStrain(method = 'Chadwick')            
            
            #### 3.10 Local fits of stress-strain curves
            
            #### 3.10.1 Local fits based on stress regions
            if fitSettings['doStressRegionFits']:
                for jj in range(len(halfWidths)):
                    for ii in range(len(centers)):
                        C, HW = centers[ii], halfWidths[jj]
                        validRange = ((C-HW) > 0)
                        if validRange:
                            IC.fitSS_stressRegion(C, HW, dictFitValidation)
            
            #### 3.10.2 Local fits based on sliding gaussian weights based on stress values
            if fitSettings['doStressGaussianFits']:
                for jj in range(len(halfWidths)):
                    for ii in range(len(centers)):
                        C, HW = centers[ii], halfWidths[jj]
                        validRange = ((C-HW) > 0)
                        if validRange:
                            IC.fitSS_stressGaussian(C, HW, dictFitValidation)
            
            #### 3.10.3 Local fits based on fixed number of points
            if fitSettings['doNPointsFits']:
                nbPtsTotal = len(IC.stressCompr)
                iStart = 0
                iStop = iStart + nbPtsFit
                
                while iStop < nbPtsTotal:
                    mask = np.array([((i >= iStart) and (i < iStop)) for i in range(nbPtsTotal)])
                    IC.fitSS_nPoints(mask, dictFitValidation)
                    
                    iStart = iStop - overlapFit
                    iStop = iStart + nbPtsFit
                    
            #### 3.10.4 Convert all dictFits into DataFrame that can be concatenated and exported after.
            IC.dictFits_To_DataFrame(fitSettings)
            
            
            #### 3.11 IN DEVELOPMENT - Trying to get a smoothed representation of the stress-strain curves
            # IC.fitSS_polynomial()
            # IC.fitSS_smooth()
            
    
    
    #### Plots
    
    # Tests
    # fig1, axes1 = CC.plot_Timeseries(plotSettings)
    # fig2, axes2 = CC.plot_FH(plotSettings)
    # fig31, axes31 = CC.plot_SS(plotSettings, fitType='stressRegion')
    # fig41, axes41 = CC.plot_KS(plotSettings, fitType='stressRegion')
    # fig32, axes32 = CC.plot_SS(plotSettings, fitType='stressGaussian')
    # fig42, axes42 = CC.plot_KS(plotSettings, fitType='stressGaussian')
    # fig33, axes33 = CC.plot_SS(plotSettings, fitType='nPoints')
    # fig43, axes43 = CC.plot_KS(plotSettings, fitType='nPoints')
    # fig5, axes5 = CC.plot_KS_smooth()
    # if SHOW:
    #     plt.show()
    # else:
    #     plt.close('all')
    
    if PLOT:        
        CC.plot_and_save(plotSettings, dpi = 150, figSubDir = 'MecaAnalysis_allCells')
        
        if SHOW:
            plt.show()
        else:
            plt.close('all')
    
    #### Results
    
    CC.exportTimeseriesWithStressStrain()
    
    CC.make_mainResults(fitSettings)
    CC.make_localFitsResults(fitSettings)
    
    df_H0 = CC.getH0Df()
    
    res = {'results_main' : CC.df_mainResults,
            'results_stressRegions' : CC.df_stressRegions,
            'results_stressGaussian' : CC.df_stressGaussian,
            'results_nPoints' : CC.df_nPoints,
            'results_H0' : df_H0
            }

    print(gs.GREEN + 'T = {:.3f}'.format(time.time() - top) + gs.NORMAL)
    warnings.filterwarnings('default')
    
    return(res)

# %%%% simple wrapper
# Simple script to call just the analysis on 1 timeseries file

# path = cp.DirDataTimeseries
# # path = cp.DirCloudTimeseries
# ld = os.listdir(path)

# expDf = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = cp.suffix)

# f = '22-07-15_M1_P1_C1_disc20um_L40_PY.csv'

# def simpleWrapper(f, expDf):
#     tsDf = getCellTimeSeriesData(f, fromCloud = False)
#     res = analyseTimeSeries_meca(f, tsDf, expDf, PLOT = True, SHOW = False)
#     return(res)
    
# res = simpleWrapper(f, expDf)

# %%%% complex wrapper


def buildDf_meca(list_mecaFiles, task, expDf, PLOT=False, SHOW = False):
    """
    Subfunction of computeGlobalTable_meca
    Create the dictionnary that will be converted in a pandas table in the end.
    """
    list_resultDf = []
    # Nfiles = len(list_mecaFiles)
    
    for f in list_mecaFiles: #[:10]:
        tS_DataFilePath = os.path.join(cp.DirDataTimeseries, f)
        current_tsDf = pd.read_csv(tS_DataFilePath, sep = ';')
        # MAIN SUBFUNCTION
        res_all = analyseTimeSeries_meca(f, current_tsDf, expDf, 
                                        taskName = task, PLOT = PLOT, SHOW = SHOW)
        for k in res_all.keys():
            
            df = res_all[k]
            if k == 'results_main':
                res_main = df
            else:
                if df.size > 0:
                    cellID = ufun.findInfosInFileName(f, 'cellID')
                    fileName = cellID + '_' + k + '.csv'
                    path = os.path.join(cp.DirDataAnalysisFits, fileName)
                    df.to_csv(path, sep=';', index=False)
                    if cp.CloudSaving != '':
                        cloudpath = os.path.join(cp.DirCloudAnalysisFits, fileName)
                        df.to_csv(cloudpath, sep=';', index=False)
                    
        list_resultDf.append(res_main)

    mecaDf = pd.concat(list_resultDf)
    return(mecaDf)


def updateUiDf_meca(ui_fileSuffix, mecaDf):
    """
    """
    listColumnsUI = ['date','cellName','cellID','manipID','compNum',
                     'UI_Valid','UI_Comments']
    
    listDates = mecaDf['date'].unique()
    
    for date in listDates:
        ui_fileName = date + '_' + ui_fileSuffix
        
        try:
            savePath = os.path.join(cp.DirDataAnalysisUMS, (ui_fileName + '.csv'))
            uiDf = pd.read_csv(savePath, sep='\t')
            fromScratch = False
            print(gs.GREEN + date + ' : imported existing UI table' + gs.NORMAL)
            
        except:
            print(gs.DARKGREEN + date + ' : no existing UI table found' + gs.NORMAL)
            fromScratch = True
    
        new_uiDf = mecaDf[mecaDf['date'] == date][listColumnsUI[:5]]
        if not fromScratch:
            existingCellId = uiDf['cellID'].values
            new_uiDf = new_uiDf.loc[new_uiDf['cellID'].apply(lambda x : x not in existingCellId), :]
        
        nrows = new_uiDf.shape[0]
        new_uiDf['UI_Valid'] = np.ones(nrows, dtype = bool)
        new_uiDf['UI_Comments'] = np.array(['' for i in range(nrows)])
        
        if not fromScratch:
            new_uiDf = pd.concat([uiDf, new_uiDf], axis = 0, ignore_index=True)
            
        savePath = os.path.join(cp.DirDataAnalysisUMS, (ui_fileName + '.csv'))
        new_uiDf.sort_values(by=['cellID', 'compNum'], inplace = True)
        new_uiDf.to_csv(savePath, sep='\t', index = False)
        
        if cp.CloudSaving != '':
            CloudTimeSeriesFilePath = os.path.join(cp.DirCloudAnalysisUMS, (ui_fileName + '.csv'))
            new_uiDf.to_csv(CloudTimeSeriesFilePath, sep = ';', index=False)







def computeGlobalTable_meca(mode = 'fromScratch', task = 'all',
                            fileName = 'Global_MecaData', 
                            save = False, PLOT = False, \
                            source = 'Python'):
    """
    Compute the GlobalTable_meca from the time series data files. \n
    > mode = 'fromScratch' or 'updateExisting':\n
    - 'fromScratch' will analyse all the time series data files and construct a new GlobalTable from them regardless of the existing GlobalTable.\n
    - 'updateExisting' will open the existing GlobalTable and determine which of the time series data files are new ones, and will append the existing GlobalTable with the data analysed from those new files.\n
    > task = 'all' or a string containing a file prefix, or several prefixes separated by ' & '.\n
    
    - 'all' will include in the analysis all the files that correspond to the chosen 'mode'.\n
    - giving file prefixes will include in the analysis all the files that start with this prefix and correspond to the chosen 'mode'. Examples: task = '22-05-03' or task = '22-05-03_M1 & 22-05-04_M2 & 22-05-05'.\n
    > fileName: the name of the table that will be created. If mode='updateExisting', it is this table that will be updated.\n
    > save: save the final table or not.\n
    > PLOT: save the plots or not.\n
    > source: 'Matlab' or 'Python', default is 'Python'.\n
    > dictColumnsMeca: dict that has to contain all the fields of the table that will be constructed AND their default values: dictColumnsMeca = {col1: defaultVal1, col2: defaultVal2, etc}.
    """
    top = time.time()
    
    ui_fileSuffix = 'UserManualSelection_MecaData'
    
    # 1. Initialization
    # 1.1 Get the experimental dataframe
    expDf = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = cp.suffix)
    
    # 1.2 Get the list of all meca files    
    suffixPython = '_PY'
    if source == 'Matlab':
        list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                      if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
                      and (('R40' in f) or ('L40' in f)) and not (suffixPython in f))]
        
    elif source == 'Python':
        list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                      if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
                      and (('R40' in f) or ('L40' in f)) and (suffixPython in f))]
    
    # 2. Get the existing table if necessary
    imported_mecaDf = False
    
    # 2.1
    if mode == 'fromScratch':
        pass

    # 2.2
    elif mode == 'updateExisting':
        try:
            imported_mecaDf = True
            savePath = os.path.join(cp.DirDataAnalysis, (fileName + '.csv'))
            existing_mecaDf = pd.read_csv(savePath, sep=';')
            
        except:
            print('No existing table found')
    # 2.3
    else:
        pass
        # existing_mecaDf = buildDf_meca([], 'fromScratch', expDf, PLOT=False)
    
    # 3. Select the files to analyse from the list according to the task
    list_taskMecaFiles = []
    
    # 3.1
    if task == 'all':
        list_taskMecaFiles = list_mecaFiles
    # 3.2
    else:
        task_list = task.split(' & ')
        for f in list_mecaFiles:
            currentCellID = ufun.findInfosInFileName(f, 'cellID')
            for t in task_list:
                if t in currentCellID:
                    list_taskMecaFiles.append(f)
                    break
                
    list_selectedMecaFiles = []
    # 3.3
    if mode == 'fromScratch':
        list_selectedMecaFiles = list_taskMecaFiles
        
    # 3.4
    elif mode == 'updateExisting':
        for f in list_taskMecaFiles:
            currentCellID = ufun.findInfosInFileName(f, 'cellID')
            if currentCellID not in existing_mecaDf.cellID.values:
                list_selectedMecaFiles.append(f)
                
    # Exclude the ones that are not in expDf
    listExcluded = []
    listManips = expDf['manipID'].values
    for i in range(len(list_selectedMecaFiles)): 
        f = list_selectedMecaFiles[i]
        manipID = ufun.findInfosInFileName(f, 'manipID')
        if not manipID in listManips:
            listExcluded.append(f)
            
    for f in listExcluded:
        list_selectedMecaFiles.remove(f)
            
    if len(listExcluded) > 0:
        textExcluded = 'The following files were excluded from analysis\nbecause no matching experimental data was found:'
        print(gs.ORANGE + textExcluded)
        for f in listExcluded:
            print(f)
        print(gs.NORMAL)
            
                
                
    # 4. Run the analysis on the files, by blocks of 10
    listMecaDf = []
    Nfiles = len(list_selectedMecaFiles)
    
    if imported_mecaDf:
        listMecaDf.append(existing_mecaDf)
        
    for i in range(0, Nfiles, 10):
        i_start, i_stop = i, min(i+10, Nfiles)
        bloc_selectedMecaFiles = list_selectedMecaFiles[i_start:i_stop]
        # MAIN SUBFUNCTION
        new_mecaDf = buildDf_meca(bloc_selectedMecaFiles, task, expDf, PLOT)
        listMecaDf.append(new_mecaDf) 
        mecaDf = pd.concat(listMecaDf)
        if save:
            saveName = fileName + '.csv'
            savePath = os.path.join(cp.DirDataAnalysis, saveName)
            mecaDf.to_csv(savePath, sep=';', index = False)
            if cp.CloudSaving != '':
                cloudSavePath = os.path.join(cp.DirCloudAnalysis, saveName)
                mecaDf.to_csv(cloudSavePath, sep=';', index = False)
            textSave = 'Intermediate save {:.0f}/{:.0f} successful !'.format((i+10)//10, ((Nfiles-1)//10)+1)
            print(gs.CYAN + textSave + gs.NORMAL)
         
            
    # 5. Final save
    if save:
        saveName = fileName + '.csv'
        savePath = os.path.join(cp.DirDataAnalysis, saveName)
        mecaDf.to_csv(savePath, sep=';', index = False)
        if cp.CloudSaving != '':
            cloudSavePath = os.path.join(cp.DirCloudAnalysis, saveName)
            mecaDf.to_csv(cloudSavePath, sep=';', index = False)
        print(gs.BLUE + 'Final save successful !' + gs.NORMAL)
    
    updateUiDf_meca(ui_fileSuffix, mecaDf)
    
    duration = time.time() - top
    print(gs.DARKGREEN + 'Total time: {:.0f}s'.format(duration) + gs.NORMAL)
    
    return(mecaDf)
            

    
def getGlobalTable_meca(fileName):
    try:
        savePath = os.path.join(cp.DirDataAnalysis, (fileName + '.csv'))
        mecaDf = pd.read_csv(savePath, sep=';')
        print('Extracted a table with ' + str(mecaDf.shape[0]) + ' lines and ' + str(mecaDf.shape[1]) + ' columns.')
    except:
        print('No existing table found')
        
    for c in mecaDf.columns:
        if 'Unnamed' in c:
            mecaDf = mecaDf.drop([c], axis=1)
        # if 'K_CIW_' in c:    
        #     mecaDf[c].apply(lambda x : x.strip('][').split(', ')).apply(lambda x : [float(x[0]), float(x[1])])
    
    if 'ExpDay' in mecaDf.columns:
        dateExemple = mecaDf.loc[mecaDf.index[0],'ExpDay']
        if not ('manipID' in mecaDf.columns):
            mecaDf['manipID'] = mecaDf['ExpDay'] + '_' + mecaDf['CellID'].apply(lambda x: x.split('_')[0])
            
    elif 'date' in mecaDf.columns:
        dateExemple = mecaDf.loc[mecaDf.index[0],'date']
        if re.match(ufun.dateFormatExcel, dateExemple):
            print('bad date')
        
    if not ('manipID' in mecaDf.columns):
        mecaDf['manipID'] = mecaDf['date'] + '_' + mecaDf['cellName'].apply(lambda x: x.split('_')[0])

        
    return(mecaDf)

# %%% (2.4) Fluorescence data

def getFluoData(save = False):
    # Getting the table
    fluoDataFile = 'FluoQuantification.csv'
    fluoDataFilePath = os.path.join(cp.DirDataAnalysis, fluoDataFile)
    fluoDF = pd.read_csv(fluoDataFilePath, sep=';',header=0)
    print('Extracted a table with ' + str(fluoDF.shape[0]) + ' lines and ' + str(fluoDF.shape[1]) + ' columns.')
    # Cleaning the table
    try:
        for c in fluoDF.columns:
            if 'Unnamed' in c:
                fluoDF = fluoDF.drop([c], axis=1)
        
    except:
        print('Unexpected bug with the cleaning step')

    if save:
        saveName = 'FluoQuantification.csv'
        savePath = os.path.join(cp.DirDataAnalysis, saveName)
        fluoDF.to_csv(savePath, sep=';')

    
    return(fluoDF)

# %%% (2.5) Oscillations

def analyseTimeSeries_sinus(f, tsDf, expDf, listColumns, PLOT, PLOT_SHOW):
    
    plotSmallElements = True
    
    #### (0) Import experimental infos
    split_f = f.split('_')
    tsDf.dx, tsDf.dy, tsDf.dz, tsDf.D2, tsDf.D3 = tsDf.dx*1000, tsDf.dy*1000, tsDf.dz*1000, tsDf.D2*1000, tsDf.D3*1000
    thisManipID = ufun.findInfosInFileName(f, 'manipID')
    thisExpDf = expDf.loc[expDf['manipID'] == thisManipID]

    # Deal with the asymmetric pair case : the diameter can be for instance 4503 (float) or '4503_2691' (string)
    diameters = thisExpDf.at[thisExpDf.index.values[0], 'bead diameter'].split('_')
    if len(diameters) == 2:
        DIAMETER = (int(diameters[0]) + int(diameters[1]))/2.
    else:
        DIAMETER = int(diameters[0])
    
    EXPTYPE = str(thisExpDf.at[thisExpDf.index.values[0], 'experimentType'])

    results = {}
    for c in listColumnsMeca:
        results[c] = []

    nUplet = thisExpDf.at[thisExpDf.index.values[0], 'normal field multi images']
    
    if 'sinus' in EXPTYPE:
        pass


def createDataDict_sinus(listFiles, listColumns, PLOT):
    """
    Subfunction of computeGlobalTable_meca
    Create the dictionnary that will be converted in a pandas table in the end.
    """
    expDf = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = cp.suffix)
    tableDict = {}
    Nfiles = len(listFiles)
    PLOT_SHOW = (Nfiles==1)
    PLOT_SHOW = 0
    if not PLOT_SHOW:
        plt.ioff()
    for c in listColumns:
        tableDict[c] = []
    for f in listFiles: #[:10]:
        tS_DataFilePath = os.path.join(cp.DirDataTimeseries, f)
        current_tsDf = pd.read_csv(tS_DataFilePath, sep = ';')
         # MAIN SUBFUNCTION
        current_resultDict = analyseTimeSeries_sinus(f, current_tsDf, expDf, 
                                                    listColumns, PLOT, PLOT_SHOW)
        for k in current_resultDict.keys():
            tableDict[k] += current_resultDict[k]
#     for k in tableDict.keys():
#         print(k, len(tableDict[k]))
    return(tableDict)

# def computeGlobalTable_sinus(task = 'fromScratch', fileName = 'Global_Sinus', save = False, PLOT = False, \
#                             source = 'Matlab', listColumns=listColumnsMeca):
#     """
#     Compute the GlobalTable_Sinus from the time series data files.
#     Option task='fromScratch' will analyse all the time series data files and construct a new GlobalTable from them regardless of the existing GlobalTable.
#     Option task='updateExisting' will open the existing GlobalTable and determine which of the time series data files are new ones, and will append the existing GlobalTable with the data analysed from those new fils.
#     listColumns have to contain all the fields of the table that will be constructed.
#     """
#     top = time.time()
    
# #     list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
# #                       if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
# #                       and ('R40' in f))] # Change to allow different formats in the future
    
#     suffixPython = '_PY'
#     if source == 'Matlab':
#         list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
#                       if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
#                       and ('R40' in f) and not (suffixPython in f))]
        
#     elif source == 'Python':
#         list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
#                       if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
#                       and ('sin' in f) and (suffixPython in f))]
#         # print(list_mecaFiles)
    
# #     print(list_mecaFiles)
    
#     if task == 'fromScratch':
#         # create a dict containing the data
#         tableDict = createDataDict_sinus(list_mecaFiles, listColumns, PLOT) # MAIN SUBFUNCTION
#         # create the dataframe from it
#         DF = pd.DataFrame(tableDict)
        
#         # last step: now that the dataFrame is complete, one can use "compStartTimeThisDay" col to compute the start time of each compression relative to the first one done this day.
#         allDates = list(DF['date'].unique())
#         for d in allDates:
#             subDf = DF.loc[DF['date'] == d]
#             experimentStartTime = np.min(subDf['compStartTimeThisDay'])
#             DF['compStartTimeThisDay'].loc[DF['date'] == d] = DF['compStartTimeThisDay'] - experimentStartTime
        
#     elif task == 'updateExisting':
#         # get existing table
#         try:
#             savePath = os.path.join(cp.DirDataAnalysis, (fileName + '.csv'))
#             existing_mecaDf = pd.read_csv(savePath, sep=';')
#         except:
#             print('No existing table found')
            
#         # find which of the time series files are new
#         new_list_mecaFiles = []
#         for f in list_mecaFiles:
#             split_f = f.split('_')
#             currentCellID = split_f[0] + '_' + split_f[1] + '_' + split_f[2] + '_' + split_f[3]
#             if currentCellID not in existing_mecaDf.cellID.values:
#                 new_list_mecaFiles.append(f)
                
#         # create the dict with new data
#         new_tableDict = createDataDict_meca(new_list_mecaFiles, listColumnsMeca, PLOT) # MAIN SUBFUNCTION
#         # create the dataframe from it
#         new_mecaDf = pd.DataFrame(new_tableDict)
#         # fuse the existing table with the new one
#         DF = pd.concat([existing_mecaDf, new_mecaDf])
        
#     else: # If task is neither 'fromScratch' nor 'updateExisting'
#     # Then task can be a substring that can be in some timeSeries file !
#     # It will create a table with only these files, WITHOUT SAVING IT !
#     # But it can plot figs from it.
#         # save = False
#         task_list = task.split(' & ')
#         new_list_mecaFiles = []
#         for f in list_mecaFiles:
#             split_f = f.split('_')
#             currentCellID = split_f[0] + '_' + split_f[1] + '_' + split_f[2] + '_' + split_f[3]
#             for t in task_list:
#                 if t in currentCellID:
#                     new_list_mecaFiles.append(f)
#                     break
#         # create the dict with new data
#         new_tableDict = createDataDict_meca(new_list_mecaFiles, listColumnsMeca, PLOT) # MAIN SUBFUNCTION
#         # create the dataframe from it
#         DF = pd.DataFrame(new_tableDict)
    
#     for c in DF.columns:
#             if 'Unnamed' in c:
#                 DF = DF.drop([c], axis=1)
    
#     if save:
#         saveName = fileName + '.csv'
#         savePath = os.path.join(cp.DirDataAnalysis, saveName)
#         DF.to_csv(savePath, sep=';')
    
#     delta = time.time() - top
#     print(delta)
    
#     return(DF)



# %% (3) General import functions
    
# %%% Main functions

def getAnalysisTable(fileName):
    if not fileName[-4:] == '.csv':
        ext = '.csv'
    else:
        ext = ''
        
    try:
        path = os.path.join(cp.DirDataAnalysis, (fileName + ext))
        df = pd.read_csv(path, sep=';')
        print(gs.CYAN + 'Analysis table has ' + str(df.shape[0]) + ' lines and ' + \
              str(df.shape[1]) + ' columns.' + gs.NORMAL)
    except:
        print(gs.BRIGHTRED + 'No analysis table found' + gs.NORMAL)
        return()
        
    for c in df.columns:
        if 'Unnamed' in c:
            df = df.drop([c], axis=1)
            
    if 'CellName' in df.columns and 'CellID' in df.columns:
        shortCellIdColumn = 'CellID'
    elif 'cellName' in df.columns and 'cellID' in df.columns:
        shortCellIdColumn = 'cellName'
    
    if 'ExpDay' in df.columns:
        dateColumn = 'ExpDay'
    elif 'date' in df.columns:
        dateColumn = 'date'
        
    # try:
    dateExemple = df.loc[df.index[0],dateColumn]
    df = ufun.correctExcelDatesInDf(df, dateColumn, dateExemple)
    # except:
    #     print(gs.ORANGE + 'Problem in date correction' + gs.NORMAL)
        
    try:
        if not ('manipID' in df.columns):
            df['manipID'] = df[dateColumn] + '_' + df[shortCellIdColumn].apply(lambda x: x.split('_')[0])
    except:
        print(gs.ORANGE + 'Could not infer Manip Ids' + gs.NORMAL)
        
    return(df)



def getMergedTable(fileName, DirDataExp = cp.DirRepoExp, suffix = cp.suffix,
                   mergeExpDf = True, mergFluo = False, mergeUMS = True,
                   findSubstrates = True):
    
    df = getAnalysisTable(fileName)
    
    if mergeExpDf:
        expDf = ufun.getExperimentalConditions(DirDataExp, suffix = suffix)
        df = pd.merge(expDf, df, how="inner", on='manipID', suffixes=("_x", "_y"),
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     copy=True,indicator=False,validate=None,
        )
        
    df = ufun.removeColumnsDuplicate(df)
        
    if mergFluo:
        fluoDf = getFluoData()
        df = pd.merge(df, fluoDf, how="left", on='cellID', suffixes=("_x", "_y"),
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     copy=True,indicator=False,validate=None,
        )
        
    df = ufun.removeColumnsDuplicate(df)
        
    if mergeUMS:
        if 'ExpDay' in df.columns:
            dateColumn = 'ExpDay'
        elif 'date' in df.columns:
            dateColumn = 'date'
        listDates = df[dateColumn].unique()
        
        listFiles_UMS = os.listdir(cp.DirDataAnalysisUMS)
        listFiles_UMS_matching = []
        # listPaths_UMS = [os.path.join(DirDataAnalysisUMS, f) for f in listFiles_UMS]
        
        for f in listFiles_UMS:
            for d in listDates:
                if d in f:
                    listFiles_UMS_matching.append(f)
            
        listPaths_UMS_matching = [os.path.join(cp.DirDataAnalysisUMS, f) for f in listFiles_UMS_matching]
        listDF_UMS_matching = [pd.read_csv(p, sep = '\t') for p in listPaths_UMS_matching]
        
        umsDf = pd.concat(listDF_UMS_matching)
        # f_filterCol = lambda x : x not in ['date', 'cellName', 'manipID']
        # umsCols = umsDf.columns[np.array([f_filterCol(c) for c in umsDf.columns])]   
        
        df = pd.merge(df, umsDf, how="left", on=['cellID', 'compNum'], suffixes=("_x", "_y"),
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     copy=True,indicator=False,validate=None,[umsCols]
        )
        
    df = ufun.removeColumnsDuplicate(df)
    
    if findSubstrates and 'substrate' in df.columns:
        vals_substrate = df['substrate'].values
        if 'diverse fibronectin discs' in vals_substrate:
            try:
                cellIDs = df[df['substrate'] == 'diverse fibronectin discs']['cellID'].values
                listFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                              if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv"))]
                for Id in cellIDs:
                    for f in listFiles:
                        if Id == ufun.findInfosInFileName(f, 'cellID'):
                            thisCellSubstrate = ufun.findInfosInFileName(f, 'substrate')
                            thisCellSubstrate = dictSubstrates[thisCellSubstrate]
                            if not thisCellSubstrate == '':
                                df.loc[df['cellID'] == Id, 'substrate'] = thisCellSubstrate
                print(gs.GREEN  + 'Automatic determination of substrate type SUCCEDED !' + gs.NORMAL)
                
            except:
                print(gs.RED  + 'Automatic determination of substrate type FAILED !' + gs.NORMAL)
        
    df = ufun.removeColumnsDuplicate(df)
    
    print(gs.CYAN + 'Merged table has ' + str(df.shape[0]) + ' lines and ' \
          + str(df.shape[1]) + ' columns.' + gs.NORMAL)
        
    return(df)



def getMatchingFits(mecaDf, fitType = 'stressGaussian', output = 'df',
                    filter_fitID = None):
    """
    Browse the fits tables in 'cp.DirDataAnalysisFits'
    and return all those that corresponds to cells inside 'mecaDf'.
    
    Parameters
    ----------
    mecaDf : pandas.DataFrame
        A table returned by getMergedTable
    
    fitType = 'stressRegion' | 'stressGaussian' | 'nPoints'
        The type of fit wanted
    
    output = 'list' | 'dict' | 'df'
        The type of output returned
            'list' : list of dataframes
            'dict' : dict of {'cellId:dataframe'} 
            'df' : one concatenated dataframe.
    
    filter_fitID : string, optionnal
        Filter the imported data according to the 'id' column of the tables.
        
    NB: Other filter_### inputs should be added in this function if needed.
      
    Returns
    -------
    res : list, dict or pandas.DataFrame
        The required data in the specified format.

    Examples
    --------
    >>> mecaDf = getMergedTable('Global_MecaData_Py')
    >>> fitsDf = getMatchingFits(mecaDf, fitType = 'nPoints', output = 'df',
                        filter_fitID = None)
    >>> # Return all fits obtained with a specified number of points, in one df.
    >>> fitsDf = getMatchingFits(mecaDf, fitType = 'gaussianStress', output = 'df',
                        filter_fitID = '200_75')
    >>> # Return only fits in the region 200+/-75 Pa obtained with 
    >>> # gaussian stress window and in one df.
    >>> fitsDf = getMatchingFits(mecaDf, fitType = 'regionStress', output = 'df',
                        filter_fitID = '_75')
    >>> # Return only fits in regions of half width 75 Pa, obtained with 
    >>> # discrete stress window, and in one df.
    """
    
    src_path = cp.DirDataAnalysisFits
    listCellIDs = mecaDf['cellID'].unique()
    listFitResults = os.listdir(src_path)
    
    if filter_fitID != None:
        if filter_fitID.startswith('_'):
            filter_fitID = r'\d{2,4}' + filter_fitID
        elif filter_fitID.endswith('_'):
            filter_fitID = filter_fitID + r'\d{2,4}'
        
    
    if output == 'df' or output == 'list':
        L = []
        for f in listFitResults:
            if fitType in f:
                cellID = ufun.findInfosInFileName(f, 'cellID')
                if cellID in listCellIDs:
                    f_path = os.path.join(src_path, f)
                    df = pd.read_csv(f_path, sep=None, engine='python')
                    if filter_fitID != None:
                        fltr = df['id'].apply(lambda x : re.match(filter_fitID, x) != None)
                        df = df[fltr]
                    L.append(df)
                    
        if output == 'list':
            res = L
        elif output == 'df':
            res = pd.concat(L, ignore_index=True)
                    
    
    elif output == 'dict':
        res = {}
        for f in listFitResults:
            if fitType in f:
                cellID = ufun.findInfosInFileName(f, 'cellID')
                if cellID in listCellIDs:
                    f_path = os.path.join(src_path, f)
                    df = pd.read_csv(f_path, sep=None, engine='python')
                    if filter_fitID != None:
                        fltr = df['id'].apply(lambda x : re.match(filter_fitID, x) != None)
                        df = df[fltr]
                    res[cellID] = df
                    
    return(res)
    
    


def getFitsInTable(mecaDf, fitType = 'stressGaussian', filter_fitID = None):
    """
    Merge a mecaDf with the required kind of fit data, by calling getMatchingFits().
    
    Parameters
    ----------
    mecaDf : pandas.DataFrame
        A table returned by getMergedTable
    
    fitType = 'stressRegion' | 'stressGaussian' | 'nPoints'
        The type of fit wanted
    
    filter_fitID : string, optionnal
        Filter the imported data according to the 'id' column of the tables.
      
    Returns
    -------
    mergedDf
        The resulting merged DataFrame
    
    Examples
    --------
    >>> import TrackAnalyser as taka
    >>> data_main = taka.getMergedTable('Global_MecaData_Py')
    >>> # Ex1.
    >>> mecaDf = taka.getFitsInTable(mecaDf_main, fitType = 'nPoints', filter_fitID = None)
    >>> # Merge with all fits obtained with 
    >>> # the nPoints method (specified number of points).
    
    >>> # Ex2.
    >>> mecaDf = taka.getFitsInTable(mecaDf_main, fitType = 'gaussianStress', filter_fitID = '200_75')
    >>> # Return only fits in the region 200+/-75 Pa obtained with 
    >>> # the gaussianStress method (gaussian stress window).
    
    >>> # Ex3.
    >>> mecaDf = taka.getFitsInTable(mecaDf_main, fitType = 'regionStress', filter_fitID = '_75')
    >>> # Return only fits in regions of half width 75 Pa, obtained with 
    >>> # the regionStress method (discrete stress window).

    """
    
    fitsDf = getMatchingFits(mecaDf, fitType = fitType, filter_fitID = filter_fitID,
                             output = 'df')
    mergeCols = ['cellID', 'compNum']
    rd = {c : 'fit_' + c for c in fitsDf.columns if c not in mergeCols}
    fitsDf = fitsDf.rename(columns = rd) 
    mergedDf = pd.merge(mecaDf, fitsDf, how="left", on=mergeCols, suffixes=("_x", "_y"),
    #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
    #     copy=True,indicator=False,validate=None,
    )
        
    mergedDf = ufun.removeColumnsDuplicate(mergedDf)
    
    return(mergedDf)
        
        

  
  
  
  
  
