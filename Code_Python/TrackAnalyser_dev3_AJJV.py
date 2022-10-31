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


import matplotlib
import matplotlib.pyplot as plt

import re
import os
import sys
import time
import random
import itertools

from copy import copy
from cycler import cycler
from datetime import date
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)

import GraphicStyles as gs
import UtilityFunctions as ufun

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

# def computeGlobalTable_meca(task = 'fromScratch', fileName = 'Global_MecaData', save = False, PLOT = False, \
#                             source = 'Matlab', listColumnsMeca=listColumnsMeca):
#     """
#     Compute the GlobalTable_meca from the time series data files.
#     Option task='fromScratch' will analyse all the time series data files and construct a new GlobalTable from them regardless of the existing GlobalTable.
#     Option task='updateExisting' will open the existing GlobalTable and determine which of the time series data files are new ones, and will append the existing GlobalTable with the data analysed from those new fils.
#     listColumnsMeca have to contain all the fields of the table that will be constructed.
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
#                       and (('R40' in f) or ('L40' in f)) and (suffixPython in f))]
#         # print(list_mecaFiles)

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
# * createMecaDataDict() call the previous function on the given list of files and concatenate the results
# * computeGlobalTable_meca() call the previous function and convert the dict to a DataFrame

#### H0_bestMethod
H0_bestMethod = 'H0_Chadwick15'

#### listColumnsMeca
listColumnsMeca = ['date','cellName','cellID','manipID',
                   'compNum','compDuration','compStartTime','compAbsStartTime','compStartTimeThisDay',
                   'initialThickness','minThickness','maxIndent','previousThickness',
                   'surroundingThickness','surroundingDx','surroundingDz',
                   'validatedThickness', 'jumpD3',
                   'minForce', 'maxForce', 'minStress', 'maxStress', 'minStrain', 'maxStrain',
                   'ctFieldThickness','ctFieldFluctuAmpli','ctFieldDX','ctFieldDZ',
                   'H0_Chadwick15', 'H0_Dimitriadis15', 
                   'H0Chadwick','EChadwick','R2Chadwick','EChadwick_CIWidth',
                   'hysteresis',
                   'critFit', 'validatedFit','comments'] # 'fitParams', 'H0_Dimitriadis15', 



#### SETTING ! Fit Selection R2 & Chi2
'''
The different Chi2 criteria below corrospond to the fact the the fits on the stress-strain curve are
done based on three different methods. The first linear fit on specific stress ranges fits on the strain
and the other two (K3 and K4) are fit on the stress.

'''
#
dictSelectionCurve = {'R2' : 0.6, 'K2_Chi2' : 0.1, 'K3_Chi2' :0.1, 'K4_Chi2' : 0.1, 'Error' : 0.02}



#### SETTING ! Change the region fits NAMES here 

#### >>> OPTION 1 - MANY FITS, FEW PLOTS

# fit_intervals = [S for S in range(0,450,50)] + [S for S in range(500,1100,100)] + [S for S in range(1500,2500,500)]
# regionFitsNames = []
# for ii in range(len(fit_intervals)-1):
#     for jj in range(ii+1, len(fit_intervals)):
#         regionFitsNames.append(str(fit_intervals[ii]) + '<s<' + str(fit_intervals[jj]))
        
# fit_toPlot = ['50<s<200', '200<s<350', '350<s<500', '500<s<700', '700<s<1000', '1000<s<1500', '1500<s<2000']


#### >>> OPTION 2 - TO LIGHTEN THE COMPUTATION
# regionFitsNames = fit_toPlot


#### >>> OPTION 3 - OLIVIA'S IDEA
# fitMin = [S for S in range(25,1225,50)]
# fitMax = [S+150 for S in fitMin]
# fitCenters = np.array([S+75 for S in fitMin])
# regionFitsNames = [str(fitMin[ii]) + '<s<' + str(fitMax[ii]) for ii in range(len(fitMin))]
# fit_toPlot = [regionFitsNames[ii] for ii in range(0, len(regionFitsNames), 2)]

# mask_fitToPlot = np.array(list(map(lambda x : x in fit_toPlot, regionFitsNames)))

# for rFN in regionFitsNames:
#     listColumnsMeca += ['KChadwick_'+rFN, 'K_CIW_'+rFN, 'R2Chadwick_'+rFN, 'K2Chadwick_'+rFN, 
#                         'H0Chadwick_'+rFN, 'Npts_'+rFN, 'validatedFit_'+rFN] 



#### >>> OPTION 4 - Longe ranges
# intervalSize = 250
# fitMin = [S for S in range(25,975,50)] + [S for S in range(975,2125,100)]
# fitMax = [S+intervalSize for S in fitMin]
# fitCenters = np.array([S+0.5*intervalSize for S in fitMin])
# regionFitsNames = [str(fitMin[ii]) + '<s<' + str(fitMax[ii]) for ii in range(len(fitMin))]
# fit_toPlot = [regionFitsNames[ii] for ii in range(0, len(regionFitsNames), 4)]

# mask_fitToPlot = np.array(list(map(lambda x : x in fit_toPlot, regionFitsNames)))

# for rFN in regionFitsNames:
#     listColumnsMeca += ['KChadwick_'+rFN, 'K_CIW_'+rFN, 'R2Chadwick_'+rFN, 'K2Chadwick_'+rFN, 
#                         'H0Chadwick_'+rFN, 'Npts_'+rFN, 'validatedFit_'+rFN] 
    # 'H0Chadwick_'+rFN, 'EChadwick_'+rFN
    
#### >>> OPTION 5 - Effect of range width

fitC =  np.array([S for S in range(100, 1150, 50)])
fitW = [100, 150, 200, 250, 300]

fitCenters = np.array([[int(S) for S in fitC] for w in fitW]).flatten()
fitWidth = np.array([[int(w) for S in fitC] for w in fitW]).flatten()

fitMin = np.array([[int(S-(w/2)) for S in fitC] for w in fitW]).flatten()
fitMax = np.array([[int(S+(w/2)) for S in fitC] for w in fitW]).flatten()

fitCenters, fitWidth = fitCenters[fitMin>0], fitWidth[fitMin>0]
fitMin, fitMax = fitMin[fitMin>0], fitMax[fitMin>0]

regionFitsNames = ['S='  + str(fitCenters[ii]) + '+/-' + str(int(fitWidth[ii]//2)) for ii in range(len(fitCenters))]

fit_toPlot = [regionFitsNames[ii] for ii in range(len(fitC), 2*len(fitC), 2)]
mask_fitToPlot = np.array(list(map(lambda x : x in fit_toPlot, regionFitsNames)))

for rFN in regionFitsNames:
    listColumnsMeca += ['KChadwick_'+rFN, 'H0Chadwick_'+rFN, 
                        'R2Chadwick_'+rFN, 'K2Chadwick_'+rFN, 'K2_CIW_'+rFN,  'Npts_'+rFN, 'K2_validatedFit_'+rFN,
                        'K3Chadwick_'+rFN, 'K3_CIW_'+rFN,  'K3_Chi2_'+rFN, 'K3_validatedFit_'+rFN,
                         ] 
    # 'H0Chadwick_'+rFN, 'EChadwick_'+rFN


#### >>> OPTION 6 - Effect of range width on stress values, with whole stress range / 100. 
#Not specific pre-defined values as in option 5
NbOfTargets = 10
overlap = np.round((20/100)*NbOfTargets) #in percentage
count = 0

for count in range(1,NbOfTargets):
    tP = str(int(count)*NbOfTargets)
    listColumnsMeca += ['K4Chadwick_'+tP, 'K4_CIW_'+tP,  'K4_Chi2_'+tP, 'K4_validatedFit_'+tP]


#### dictColumnsMeca
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
                   'validatedThickness':np.nan, 
                   'jumpD3':np.nan,
                   'minForce':np.nan, 
                   'maxForce':np.nan, 
                   'minStress':np.nan, 
                   'maxStress':np.nan, 
                   'minStrain':np.nan, 
                   'maxStrain':np.nan,
                   'ctFieldThickness':np.nan,
                   'ctFieldFluctuAmpli':np.nan,
                   'ctFieldDX':np.nan,
                   'ctFieldDY':np.nan,
                   'ctFieldDZ':np.nan,
                   'bestH0':np.nan,
                   'error_bestH0':True,
                   'H0_Chadwick15':np.nan, 
                   'H0_Dimitriadis15':np.nan, 
                   'H0Chadwick':np.nan,
                   'EChadwick':np.nan,
                   'R2Chadwick':np.nan,
                   'EChadwick_CIWidth':np.nan,
                   'hysteresis':np.nan,
                   'validatedFit_Chadwick':False,
                   'critFit':'',
                   'comments':''
                   }


for rfn in regionFitsNames:
          
     d = {'KChadwick_'+rfn:np.nan, 
          'H0Chadwick_'+rfn:np.nan,
          
          #Parameters for linear fits on pre-defined stress ranges
          'R2Chadwick_'+rfn:np.nan, 
          'K2Chadwick_'+rfn:np.nan,
          'K2_CIW_'+rfn:np.nan, 
          'Npts_'+rfn:np.nan, 
          'K2_validatedFit_'+rfn:False,
          
          #Parameters for Gaussian-weighted linear fits
          'K3Chadwick_'+rfn:np.nan,
          'K3_CIW_'+rfn:np.nan, 
          'K3_Chi2'+rfn:np.nan, 
          'K3_validatedFit_'+rfn:False,
          
          }
     
     dictColumnsMeca = {**dictColumnsMeca, **d}



for count in range(1, NbOfTargets):
    targetPoint = str(int(count*NbOfTargets))
          #Parameters for Gaussian-weighted linear fits
    d = {'K4Chadwick_'+targetPoint:np.nan,
         'K4_CIW_'+targetPoint:np.nan, 
         'K4_Chi2'+targetPoint:np.nan, 
         'K4_validatedFit_'+targetPoint:False,
         }
     
    dictColumnsMeca = {**dictColumnsMeca, **d}
     
dictColumnsRegionFit = {'regionFitNames' : '', 
                        'K' : np.nan, 
                        'H0' : np.nan,
                        
                        'R2' : np.nan,  
                        'K2' : np.nan, 
                        'K2_fitError' : True, 
                        'K2_validatedFit' : False, 
                        'Npts' : np.nan, 
                        'K2_CIW' : np.nan,
                        
                        'K3' : np.nan, 
                        'K3_fitError' : True, 
                        'K3_validatedFit' : False, 
                        'K3_Chi2' : np.nan, 
                        'K3_CIW' : np.nan,
                        
                        'K4' : np.nan, 
                        'K4_fitError' : True, 
                        'K4_validatedFit' : False, 
                        'K4_Chi2' : np.nan, 
                        'K4_CIW' : np.nan,
                        
                        } 

#%% Attempt at class

#### TO DO
# - When this is done and runs ok, think of the results and the plots.
# - Write the polynomial fit.


# class ResultsCompression:
#     def __init__(self, dictColumns, cellComp):
#         Ncomp = cellComp.Ncomp
        
#         main = {}
#         for k in dictColumnsMeca.keys():
#             main[k] = [dictColumnsMeca[k] for m in range(Ncomp)]
#         self.main = main
#         self.dictColumns = dictColumns
#         self.Ncomp = cellComp.Ncomp

# %%% mechanical models

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

# %%% general fitting functions


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
        params, covM = curve_fit(dimitriadisModel, h, f, p0=initialParameters, bounds = parameterBounds)
        ses = np.array([covM[0,0], covM[1,1]])
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
        ses = np.array([covM[0,0], covM[1,1]])
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
        if masked:
            ols_model = sm.OLS(strain[weights], sm.add_constant(stress[weights]))
            results_ols = ols_model.fit()
            S0, K = results_ols.params[0], 1/results_ols.params[1]
            seS0, seK = results_ols.HC3_se[0], results_ols.HC3_se[1]*K**2 # See note below
        
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
        error = True
        params = np.ones(2) * np.nan
        ses = np.ones(2) * np.nan

    res = (params, ses, error)
        
    return(res)


def makeDictFit_hf(params, ses, error, y, yPredict, err_chi2):
    if not error:
        E, H0 = params
        seE, seH0 = ses

        alpha = 0.975
        dof = len(y)-len(params)
        q = st.t.ppf(alpha, dof) # Student coefficient
        R2 = ufun.get_R2(y, yPredict)
        Chi2 = ufun.get_Chi2(y, yPredict, dof, err_chi2)        

        ciwE = q*seE
        # confIntEWidth = 2*q*seE

        ciwH0 = q*seH0
        # confIntH0Width = 2*q*seH0  
    
    else:
        E, seE, H0, seH0 = np.nan, np.nan, np.nan, np.nan
        R2, Chi2 = np.nan, np.nan
        ciwE, ciwH0 = np.nan, np.nan
        # fPredict = np.nan(Npts)

    res =  {'error':error,
            'nbPts':len(y),
            'E':E, 'seE':seE,
            'H0':H0, 'seH0':seH0,
            'R2':R2, 'Chi2':Chi2,
            'ciwE':ciwE, 
            'ciwH0':ciwH0,
            'yPredict':yPredict, 
            # 'fPredict':fPredict,
            }
    
    return(res)


def makeDictFit_ss(params, ses, error, y, yPredict, err_chi2):
    if not error:
        K, S0 = params
        seK, seS0 = ses

        alpha = 0.975
        dof = len(y)-len(params)
        q = st.t.ppf(alpha, dof) # Student coefficient
        R2 = ufun.get_R2(y, yPredict)
        Chi2 = ufun.get_Chi2(y, yPredict, dof, err_chi2)        

        ciwK = q*seK
        # confIntKWidth = 2*q*seK  
    
    else:
        K, seK = np.nan, np.nan
        R2, Chi2 = np.nan, np.nan
        confIntK = [np.nan, np.nan]

    res =  {'error':error,
            'nbPts':len(y),
            'K':K, 'seK':seK,
            'R2':R2, 'Chi2':Chi2,
            'ciwK':ciwK, 
            'yPredict':yPredict, 
            # 'fPredict':fPredict,
            }
    
    return(res)





        

# %%% classes
        

class CellCompression:
    def __init__(self, timeseriesDf, thisExpDf):
        self.tsDf = timeseriesDf
        
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
        
        self.ListJumpsD3 = np.zeros(Ncomp, dtype = float)
        
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
        self.ListJumpsD3[i] = jumpD3
    
        
        
    
    

class IndentCompression:
    
    def __init__(self, cellComp, indentDf, thisExpDf, i_indent):
        self.rawDf = indentDf
        self.thisExpDf = thisExpDf
        self.i_indent = i_indent
        
        self.DIAMETER = cellComp.DIAMETER
        self.EXPTYPE = cellComp.EXPTYPE
        self.normalField = cellComp.normalField
        self.minCompField = cellComp.minCompField
        self.maxCompField = cellComp.maxCompField
        self.loopStruct = cellComp.loopStruct
        self.nUplet = cellComp.nUplet
        self.loop_totalSize = cellComp.loop_totalSize
        self.loop_rampSize = cellComp.loop_rampSize
        self.loop_ctSize = cellComp.loop_ctSize
        
        self.isValidForAnalysis = False
        
        self.isRefined = False
        self.hCompr = (self.rawDf.D3.values[:] - self.DIAMETER)
        self.hRelax = (self.rawDf.D3.values[:] - self.DIAMETER)
        self.fCompr = (self.rawDf.F.values[:])
        self.fRelax = (self.rawDf.F.values[:])
        self.jMax = np.argmax(self.rawDf.B)
        self.jStart = 0
        self.jStop = len(self.rawDf.D3.values)
        self.Df = self.rawDf
        
        self.dictH0 = {}
        self.bestH0 = np.nan
        self.method_bestH0 = ''
        self.error_bestH0 = True
        
        self.dictFitsFH = {}
        self.dictFitsSS_stressRegions = {}
        self.dictFitsSS_stressGaussian = {}
        self.dictFitsSS_nPoints = {}
        
        self.deltaCompr = np.zeros_like(self.hCompr)*np.nan
        self.stressCompr = np.zeros_like(self.hCompr)*np.nan
        self.strainCompr = np.zeros_like(self.hCompr)*np.nan
        
    

    
        
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
        
        Npts = len(hCompr_raw)
        k = offsetStart
        while (k<(Npts//2)) and (hCompr_raw[k] < np.max(hCompr_raw[k+1:min(k+10, Npts)])):
            k += 1
        offsetStart = k
        
        jStart = offsetStart # Beginning of compression
        jStop = self.rawDf.shape[0] - offsetStop # End of relaxation
        
        # Better compressions arrays
        self.hCompr = (self.rawDf.D3.values[jStart:jMax+1] - self.DIAMETER)
        self.hRelax = (self.rawDf.D3.values[jMax+1:jStop] - self.DIAMETER)
        self.fCompr = (self.rawDf.F.values[jStart:jMax+1])
        self.fRelax = (self.rawDf.F.values[jMax+1:jStop])
        
        self.jMax = jMax
        self.jStart = jStart
        self.jStop = jStop
        
        self.Df = self.rawDf.loc[jStart:jStop, :]
        self.isRefined = True
        
        
        
        
        
    def computeH0(self, method = 'Chadwick', zone = 'pts_15'):
        listAllMethods = ['Chadwick', 'Dimitriadis', 'NaiveMax']
        [zoneType, zoneVal] = zone.split('_')
        if zoneType == 'pts':
            i1 = 0
            i2 = int(zoneVal)
            mask = np.array([((i >= i1) and (i < i2)) for i in range(len(self.fCompr))])    
        elif zoneType == '%f':
            thresh = (int(zoneVal)/100.) * np.max(self.fCompr)
            mask = (self.fCompr < thresh)
        
        if method == 'Chadwick':
            h, f, D = self.hCompr[mask], self.fCompr[mask], self.DIAMETER
            params, covM, error = fitChadwick_hf(h, f, D)
            H0 = params[1]
            self.dictH0['H0_' + method] = H0
            self.dictH0['error_' + method] = error
        elif method == 'Dimitriadis':
            h, f, D = self.hCompr[mask], self.fCompr[mask], self.DIAMETER
            params, covM, error = fitDimitriadis_hf(h, f, D)
            H0 = params[1]
            self.dictH0['H0_' + method] = H0
            self.dictH0['error_' + method] = error
        elif method == 'NaiveMax':
            H0 = np.max(self.hCompr[:])
            self.dictH0['H0_' + method] = H0
            self.dictH0['error_' + method] = False
        
        elif method == 'all':
            for m in listAllMethods:
                self.computeH0(method = m, zone = zone)
            
            
            
    def setBestH0(self, method):
        self.bestH0 = self.dictH0['H0_' + method]
        self.error_bestH0 = self.dictH0['error_' + method]
        self.method_bestH0 = method
    
    
    
    
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
        

    
    
    def fitFH(self, method = 'Chadwick'):
        listAllMethods = ['Chadwick']
        # makeDictFit_hf(params, covM, error, y, yPredict, err_chi2)
        
        if method == 'Chadwick':
            h, f, D = self.hCompr, self.fCompr, self.DIAMETER
            params, covM, error = fitChadwick_hf(h, f, D)
            E, H0 = params
            hPredict = inversedChadwickModel(f, E, H0, self.DIAMETER)
            y, yPredict = h, hPredict
            err_chi2 = 30
            fitDict = makeDictFit_hf(params, covM, error, y, yPredict, err_chi2)

        elif method == 'Dimitriadis':
            h, f, D = self.hCompr, self.fCompr, self.DIAMETER
            params, covM, error = fitDimitriadis_hf(h, f, D)
            E, H0 = params
            fPredict = dimitriadisModel(h, E, H0, self.DIAMETER, v = 0)
            y, yPredict = f, fPredict
            err_chi2 = 100
            fitDict = makeDictFit_hf(params, covM, error, y, yPredict, err_chi2)
            
            
        if method != 'all':
            self.dictFitsFH[method] = fitDict
            
        else:
            for m in listAllMethods:
                self.fitFH(method = m)
                
    
    def fitSS_stressRegion(self, center, halfWidth):
        # makeDictFit_ss(params, covM, error, y, yPredict, err_chi2)
        stress, strain = self.stressCompr, self.strainCompr
        lowS, highS = center - halfWidth, center + halfWidth
        mask = ((stress > lowS) & (stress < highS))
        
        params, covM, error = fitLinear_ss(stress, strain, weights = mask)
        
        K, strain0 = params
        strainPredict = inversedConstitutiveRelation(stress[mask], K, strain0)
        y, yPredict = strain[mask], strainPredict
        err_chi2 = 0.0035
        fitDict = makeDictFit_ss(params, covM, error, y, yPredict, err_chi2)
        return(fitDict)                
                
    
    def fitSS_stressGaussian(self, center, halfWidth):
        # makeDictFit_ss(params, covM, error, y, yPredict, err_chi2)
        stress, strain = self.stressCompr, self.strainCompr
        X = stress.flatten(order='C')
        weights = np.exp( -((X - center) ** 2) / halfWidth ** 2)
        
        params, covM, error = fitLinear_ss(stress, strain, weights = weights)
        
        K, strain0 = params
        lowS, highS = center - halfWidth, center + halfWidth
        mask = ((stress > lowS) & (stress < highS))
        strainPredict = inversedConstitutiveRelation(stress[mask], K, strain0)
        y, yPredict = strain[mask], strainPredict
        err_chi2 = 0.0035
        fitDict = makeDictFit_ss(params, covM, error, y, yPredict, err_chi2)
        return(fitDict)
    
    
    def fitSS_mask(self, mask):
        # makeDictFit_ss(params, covM, error, y, yPredict, err_chi2)
        stress, strain = self.stressCompr, self.strainCompr
        
        params, covM, error = fitLinear_ss(stress, strain, weights = mask)
        
        K, strain0 = params
        strainPredict = inversedConstitutiveRelation(stress[mask], K, strain0)
        y, yPredict = strain[mask], strainPredict
        err_chi2 = 0.0035
        fitDict = makeDictFit_ss(params, covM, error, y, yPredict, err_chi2)
        return(fitDict)
    
    
    def fitSS_polynomial():
        pass
    
    def plot_FH():
        pass
        
    def plot_SS():
        pass
        
    def plot_KS():
        pass
    
        #### TBC
        
# %% main function
        
def analyseTimeSeries_Class(f, tsDf, expDf, dictColumnsMeca):
    
    #### Settings
    method_bestH0 = 'Dimitriadis'
    
    #### (1) Import experimental infos
    tsDf.dx, tsDf.dy, tsDf.dz, tsDf.D2, tsDf.D3 = tsDf.dx*1000, tsDf.dy*1000, tsDf.dz*1000, tsDf.D2*1000, tsDf.D3*1000
    thisManipID = ufun.findInfosInFileName(f, 'manipID')
    thisExpDf = expDf.loc[expDf['manipID'] == thisManipID]
    
    Ncomp = max(tsDf['idxAnalysis'])
    EXPTYPE = str(thisExpDf.at[thisExpDf.index.values[0], 'experimentType'])
    diameters = thisExpDf.at[thisExpDf.index.values[0], 'bead diameter'].split('_')
    if len(diameters) == 2:
        DIAMETER = (int(diameters[0]) + int(diameters[1]))/2.
    else:
        DIAMETER = int(diameters[0])
    
    cellComp = CellCompression(tsDf, thisExpDf)
    
    # results = ResultsCompression(dictColumnsMeca, cellComp)
    results = {}
    for k in dictColumnsMeca.keys():
        results[k] = [dictColumnsMeca[k] for m in range(Ncomp)]

    #### (2) Get global values
    ctFieldH = (tsDf.loc[tsDf['idxAnalysis'] == 0, 'D3'].values - DIAMETER)
    ctFieldThickness   = np.median(ctFieldH)
    ctFieldFluctuAmpli = np.percentile(ctFieldH, 90) - np.percentile(ctFieldH,10)
    ctFieldDX = np.median(tsDf.loc[tsDf['idxAnalysis'] == 0, 'dx'].values)
    ctFieldDY = np.median(tsDf.loc[tsDf['idxAnalysis'] == 0, 'dy'].values)
    ctFieldDZ = np.median(tsDf.loc[tsDf['idxAnalysis'] == 0, 'dz'].values)
    
    for i in range(Ncomp):

        #### (3) Identifiers
        currentCellID = ufun.findInfosInFileName(f, 'cellID')
        
        results['date'][i] = ufun.findInfosInFileName(f, 'date')
        results['manipID'][i] = ufun.findInfosInFileName(f, 'manipID')
        results['cellName'][i] = ufun.findInfosInFileName(f, 'cellName')
        results['cellID'][i] = currentCellID
        
            
        cellComp.correctJumpForCompression(i)
            
            
        #### (4) Segment the compression n°i
        maskComp = cellComp.getMaskForCompression(i, task = 'compression')
        thisCompDf = tsDf.loc[maskComp,:]
        # iStart = (ufun.findFirst(tsDf['idxAnalysis'], i+1))
        # iStop = iStart+thisCompDf.shape[0]
        
        IC = IndentCompression(cellComp, thisCompDf, thisExpDf, i)

        # Easy-to-get parameters
        results['compNum'][i] = i+1
        results['compDuration'][i] = thisExpDf.at[thisExpDf.index.values[0], 'compression duration']
        results['compStartTime'][i] = thisCompDf['T'].values[0]
        results['compAbsStartTime'][i] = thisCompDf['Tabs'].values[0]

        doThisCompAnalysis = IC.validateForAnalysis()

        if doThisCompAnalysis:
            #### (3) Inside the compression n°i, delimit the compression and relaxation phases

            # Delimit the start of the increase of B (typically the moment when the field decrease from 5 to 3)
            # and the end of its decrease (typically when it goes back from 3 to 5)
            
            IC.refineStartStop()
            
            # Get the points of constant field preceding and surrounding the current compression
            # Ex : if the labview code was set so that there is 6 points of ct field before and after each compression,
            # previousPoints will contains D3[iStart-12:iStart]
            # surroundingPoints will contains D3[iStart-6:iStart] and D3[iStop:iStop+6]
            
            previousMask = cellComp.getMaskForCompression(i, task = 'previous')
            surroundingMask = cellComp.getMaskForCompression(i, task = 'surrounding')
            previousThickness = np.median(tsDf.D3.values[previousMask] - DIAMETER)
            surroundingThickness = np.median(tsDf.D3.values[surroundingMask] - DIAMETER)
            surroundingDx = np.median(tsDf.dx.values[surroundingMask])
            surroundingDy = np.median(tsDf.dy.values[surroundingMask])
            surroundingDz = np.median(tsDf.dz.values[surroundingMask])
            

            # Parameters relative to the thickness ( = D3-DIAMETER)
            results['previousThickness'][i] = previousThickness
            results['surroundingThickness'][i] = surroundingThickness
            results['surroundingDx'][i] = surroundingDx
            results['surroundingDy'][i] = surroundingDy
            results['surroundingDz'][i] = surroundingDz
            results['ctFieldDX'][i] = ctFieldDX
            results['ctFieldDY'][i] = ctFieldDY
            results['ctFieldDZ'][i] = ctFieldDZ
            results['ctFieldThickness'][i] = ctFieldThickness
            results['ctFieldFluctuAmpli'][i] = ctFieldFluctuAmpli
            results['jumpD3'][i] = cellComp.ListJumpsD3[i]
            
            results['initialThickness'][i] = np.mean(IC.hCompr[0:3])
            results['minThickness'][i] = np.min(IC.hCompr)
            results['maxIndent'][i] = results['initialThickness'][i] - results['minThickness'][i]

            results['validatedThickness'][i] = np.min([results['initialThickness'][i],results['minThickness'][i],
                                         results['previousThickness'][i],results['surroundingThickness'][i],
                                         results['ctFieldThickness'][i]]) > 0
            
            # Parameters relative to the force
            results['minForce'][i] = np.min(IC.fCompr)
            results['maxForce'][i] = np.max(IC.fCompr)


            #### (4) Fit with Chadwick model of the force-thickness curve
            
            #### Classic, all curve, Chadwick fit            
            IC.fitFH(method = 'Chadwick')
                
                
            #### (4.) Find the best H0
            
            IC.computeH0(method = 'all', zone = 'pts_15')
            # IC.computeH0(method = 'all', zone = '%f_10')
            # print(IC.dictH0)
            
            IC.setBestH0(method = method_bestH0)
            

            #### (4.1) Compute stress and strain based on the best H0
            
            IC.computeStressStrain(method = 'Chadwick')
                
            results['minStress'][i] = np.min(IC.stressCompr)
            results['maxStress'][i] = np.max(IC.stressCompr)
            results['minStrain'][i] = np.min(IC.strainCompr)
            results['maxStrain'][i] = np.max(IC.strainCompr)
            
            centers = [ii for ii in range(100, 1550, 50)]
            # halfWidths = [50, 75, 100]
            halfWidths = [50]
            
            for jj in range(len(halfWidths)):
                for ii in range(len(centers)):
                    C, HW = centers[ii], halfWidths[jj]
                    validRange = ((C-HW) > 0)
                    id_range = str(C) + '_' + str(HW)
                    
                    if validRange:
                        dictFit = IC.fitSS_stressRegion(C, HW)
                        isValidated = (dictFit['nbPts'] >= 8 and
                                       dictFit['R2'] >= 0.6 and 
                                       dictFit['Chi2'] >= 1)
                        dictFit['valid'] = isValidated
                        dictFit['center'] = C
                        dictFit['halfWidth'] = HW
                        IC.dictFitsSS_stressRegions[id_range] = dictFit
                        
                        
                        dictFit = IC.fitSS_stressGaussian(C, HW)
                        isValidated = (dictFit['nbPts'] >= 8 and
                                       dictFit['R2'] >= 0.6 and 
                                       dictFit['Chi2'] >= 1)
                        dictFit['valid'] = isValidated
                        dictFit['center'] = C
                        dictFit['halfWidth'] = HW
                        IC.dictFitsSS_stressGaussian[id_range] = dictFit
            
            nbPtsFit = 11
            overlapFit = 2
            nbPtsTotal = len(IC.stressCompr)
            iStart = 0
            iStop = iStart + nbPtsFit
            
            while iStop < nbPtsTotal:
                # iCenter = iStart + nbPtsFit//2
                mask = np.array([((i >= iStart) and (i < iStop)) for i in range(nbPtsTotal)])
                dictFit = IC.fitSS_mask(mask)
                isValidated = (dictFit['nbPts'] >= 8 and
                               dictFit['R2'] >= 0.6 and 
                               dictFit['Chi2'] >= 1)
                dictFit['valid'] = isValidated
                dictFit['center'] = np.mean(IC.stressCompr[iStart:iStop])
                dictFit['halfWidth'] = (np.max(IC.stressCompr[iStart:iStop]) - \
                                        np.min(IC.stressCompr[iStop]))/2
                    
                IC.dictFitsSS_nPoints[id_range] = dictFit
                
                iStart = iStop - overlapFit
                iStop = iStart + nbPtsFit
            
            # print(dictFit)
            
                        
            
            # center, hW = 600, 75
            # dict1 = IC.fitSS_stressRegion(center, hW)
            # dict2 = IC.fitSS_stressGaussian(center, hW)
            
            # print(dict1)
            # print(dict2)



    
    

#%% Chadwick Computing functions

def compressionFitChadwick(hCompr, fCompr, DIAMETER):
    
    error = False
    
    def chadwickModel(h, E, H0):
        R = DIAMETER/2
        f = (np.pi*E*R*((H0-h)**2))/(3*H0)
        return(f)

    def inversedChadwickModel(f, E, H0):
        R = DIAMETER/2
        h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
        return(h)

    # some initial parameter values - must be within bounds
    initH0 = max(hCompr) # H0 ~ h_max
    initE = (3*max(hCompr)*max(fCompr))/(np.pi*(DIAMETER/2)*(max(hCompr)-min(hCompr))**2) # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
    # initH0, initE = initH0*(initH0>0), initE*(initE>0)
    
    initialParameters = [initE, initH0]

    # bounds on parameters - initial parameters must be within these
    lowerBounds = (0, 0)
    upperBounds = (np.Inf, np.Inf)
    parameterBounds = [lowerBounds, upperBounds]

    try:
        params, covM = curve_fit(inversedChadwickModel, fCompr, hCompr, initialParameters, bounds = parameterBounds)

        # Previously I fitted with y=F and x=H, but it didn't work so well cause H(t) isn't monotonous:
        # params, covM = curve_fit(chadwickModel, hCompr, fCompr, initialParameters, bounds = parameterBounds)
        # Fitting with the 'inverse Chadwick model', with y=H and x=F is more convenient

        E, H0 = params
        hPredict = inversedChadwickModel(fCompr, E, H0)
        err = dictSelectionCurve['Error']
        
        comprMat = np.array([hCompr, fCompr]).T
        comprMatSorted = comprMat[comprMat[:, 0].argsort()]
        hComprSorted, fComprSorted = comprMatSorted[:, 0], comprMatSorted[:, 1]
        fPredict = chadwickModel(hComprSorted, E, H0)
        
        # Stress and strain
        deltaCompr = (H0 - hCompr)/1000 # µm
        stressCompr = fCompr / (np.pi * (DIAMETER/2000) * deltaCompr)
        strainCompr = deltaCompr / (3*(H0/1000)) 
        strainPredict = stressCompr / (E*1e6) #((H0 - hPredict)/1000) / (3*(H0/1000))
        
        # residuals_h = hCompr-hPredict
        # residuals_f = fComprSorted-fPredict

        alpha = 0.975
        dof = len(fCompr)-len(params)
        q = st.t.ppf(alpha, dof) # Student coefficient
        R2 = ufun.get_R2(hCompr, hPredict)
        
        Chi2 = ufun.get_Chi2(strainCompr, strainPredict, dof, err)

        varE = covM[0,0]
        seE = (varE)**0.5
        E, seE = E*1e6, seE*1e6
        confIntE = [E-q*seE, E+q*seE]
        confIntEWidth = 2*q*seE

        varH0 = covM[1,1]
        seH0 = (varH0)**0.5
        confIntH0 = [H0-q*seH0, H0+q*seH0]
        confIntH0Width = 2*q*seH0
        
        
    except:
        error = True
        E, H0, hPredict, R2, Chi2, confIntE, confIntH0 = -1, -1, np.ones(len(hCompr))*(-1), -1, -1, [-1,-1], [-1,-1]
    
    return(E, H0, hPredict, R2, Chi2, confIntE, confIntH0, error)


def compressionFitDimitriadis(hCompr, fCompr, DIAMETER, order = 2):
    
    error = False
    
    v = 0
    
    def dimitriadisModel(h, E, H0):
        
        R = DIAMETER/2
        delta = H0-h
        X = np.sqrt(R*delta)/h
        ks = ufun.getDimitriadisCoefs(v, order)
        
        poly = np.zeros_like(X)
        
        for i in range(order+1):
            poly = poly + ks[i] * X**i
            
        F = ((4 * E * R**0.5 * delta**1.5)/(3 * (1 - v**2))) * poly
        return(F)

    # some initial parameter values - must be within bounds
    # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
    initE = (3*max(hCompr)*max(fCompr))/(np.pi*(DIAMETER/2)*(max(hCompr)-min(hCompr))**2) 
    # H0 ~ h_max
    initH0 = max(hCompr) 

    # initH0, initE = initH0*(initH0>0), initE*(initE>0)
    
    initialParameters = [initE, initH0]

    # bounds on parameters - initial parameters must be within these
    lowerBounds = (0, max(hCompr))
    upperBounds = (np.Inf, np.Inf)
    parameterBounds = [lowerBounds, upperBounds]

    try:
        
        params, covM = curve_fit(dimitriadisModel, hCompr, fCompr, p0=initialParameters, bounds = parameterBounds)
        E, H0 = params
        fPredict = dimitriadisModel(hCompr, E, H0)
        err = 100
        
        comprMat = np.array([hCompr, fPredict]).T
        comprMatSorted = comprMat[comprMat[:, 0].argsort()]
        hComprSorted, fPredictSorted = comprMatSorted[:, 0], comprMatSorted[:, 1]

        alpha = 0.975
        dof = len(fCompr)-len(params)
        q = st.t.ppf(alpha, dof) # Student coefficient
        R2 = ufun.get_R2(fCompr, fPredict)
        
        Chi2 = ufun.get_Chi2(fCompr, fPredict, dof, err)
        
        varE = covM[0,0]
        seE = (varE)**0.5
        E, seE = E*1e6, seE*1e6
        confIntE = [E-q*seE, E+q*seE]
        confIntEWidth = 2*q*seE

        varH0 = covM[1,1]
        seH0 = (varH0)**0.5
        confIntH0 = [H0-q*seH0, H0+q*seH0]
        confIntH0Width = 2*q*seH0
        
        
    except:
        error = True
        E, H0, fPredict, R2, Chi2, confIntE, confIntH0 = -1, -1, np.ones(len(hCompr))*(-1), -1, -1, [-1,-1], [-1,-1]
    
    return(E, H0, fPredict, R2, Chi2, confIntE, confIntH0, error)


    

def compressionFitChadwick_StressStrain(hCompr, fCompr, H0, DIAMETER):
    
    error = False
    
    # def chadwickModel(h, E, H0):
    #     R = DIAMETER/2
    #     f = (np.pi*E*R*((H0-h)**2))/(3*H0)
    #     return(f)

    # def inversedChadwickModel(f, E, H0):
    #     R = DIAMETER/2
    #     h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
    #     return(h)
    
    def computeStress(f, h, H0):
        R = DIAMETER/2000
        delta = (H0 - h)/1000
        stress = f / (np.pi * R * delta)
        return(stress)
        
    def computeStrain(h, H0):
        delta = (H0 - h)/1000
        strain = delta / (3 * (H0/1000))
        return(strain)
    
    def constitutiveRelation(strain, K, stress0):
        stress = (K * strain) + stress0
        return(stress)
    
    def inversedConstitutiveRelation(stress, K, strain0):
        strain = (stress / K) + strain0
        return(strain)

    # some initial parameter values - must be within bounds
    initK = (3*max(hCompr)*max(fCompr))/(np.pi*(DIAMETER/2)*(max(hCompr)-min(hCompr))**2) # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
    init0 = 0
    
    initialParameters = [initK, init0]

    # bounds on parameters - initial parameters must be within these
    lowerBounds = (0, -np.Inf)
    upperBounds = (np.Inf, np.Inf)
    parameterBounds = [lowerBounds, upperBounds]


    try:
        strainCompr = computeStrain(hCompr, H0)
        stressCompr = computeStress(fCompr, hCompr, H0)
        
        params, covM = curve_fit(inversedConstitutiveRelation, stressCompr, strainCompr, initialParameters, bounds = parameterBounds)

        K, strain0 = params
        strainPredict = inversedConstitutiveRelation(stressCompr, K, strain0)
        err = dictSelectionCurve['Error']

        alpha = 0.975
        dof = len(stressCompr)-len(params)
        
        R2 = ufun.get_R2(strainCompr, strainPredict)
        
        Chi2 = ufun.get_Chi2(strainCompr, strainPredict, dof, err)

        varK = covM[0,0]
        seK = (varK)**0.5
        
        q = st.t.ppf(alpha, dof) # Student coefficient
        # K, seK = K*1e6, seK*1e6
        confIntK = [K-q*seK, K+q*seK]
        confIntKWidth = 2*q*seK
        
        
    except:
        error = True
        K, strainPredict, R2, Chi2, confIntK = -1, np.ones(len(strainCompr))*(-1), -1, -1, [-1,-1]
    
    return(K, strainPredict, R2, Chi2, confIntK, error)

def compressionFitChadwick_weightedLinearFit_V1(hCompr, fCompr, fitCentres_region, H0, DIAMETER):

    error = False
    K3Limit = 100 #Limiting the range of stressCompr to perform Chi2 tests with weighted stressPredict
    
    def computeStress(f, h, H0):
        R = DIAMETER/2000
        delta = (H0 - h)/1000
        stress = f / (np.pi * R * delta)
        return(stress)
        
    def computeStrain(h, H0):
        delta = (H0 - h)/1000
        strain = delta / (3 * (H0/1000))
        return(strain)
    
    def constitutiveRelation(strain, K, stress0):
        stress = (K * strain) + stress0
        return(stress)
    
    def gaussianWeights(fitCentres_region, stressCompr, HR):
        stressCompr = stressCompr.flatten(order='C')
        return np.exp ( -((stressCompr - fitCentres_region) ** 2) / HR ** 2)
    
    # def gaussianWeights(fitCentres_region, stressCompr, HR):
    #     stressCompr = stressCompr.flatten(order='C')
    #     return np.exp ( -(stressCompr - fitCentres_region) / HR )
    
    def weightedLinearFit(strainCompr, stressCompr, fitCentres_region, HR = 100):
        
        strainCompr = strainCompr.reshape(-1, 1)
        stressCompr = stressCompr.reshape(-1, 1)
        
        weights = gaussianWeights(fitCentres_region, stressCompr, HR)
        regr = LinearRegression()
        regr.fit(strainCompr, stressCompr, weights)
        K = regr.coef_[0]
        stress0 = regr.intercept_
        strainCompr = strainCompr.flatten()
        
        stressCompr_copy = np.copy(stressCompr).flatten()
        stressPredict = constitutiveRelation(strainCompr, K, stress0)
        stressPredict = np.asarray(stressPredict)
        

        return(K, stressPredict, stress0)
    
    strainCompr = computeStrain(hCompr, H0)
    stressCompr = computeStress(fCompr, hCompr, H0)
    maxStressCompr = np.max(stressCompr)
    
    if fitCentres_region <= maxStressCompr:
        K, stressPredict, stress0 = weightedLinearFit(strainCompr, stressCompr, fitCentres_region)
        K = K[0] #Because the way regr.fit() returns the slope is within an array by default (very weird)
        
        lowS3, highS3 = int(fitCentres_region - K3Limit),  int(fitCentres_region + K3Limit)
        
        fitConditions = np.where(np.logical_and(stressPredict >= lowS3, \
                                                  stressPredict <= highS3))
        
        stressCompr_fit = stressCompr[fitConditions]
        stressPredict_fit = stressPredict[fitConditions]
        
        covM = np.cov(stressPredict_fit, stressCompr_fit)
        
        err = dictSelectionCurve['Error']
        params = K, stress0
        
        alpha = 0.975
        
        # R2 = ufun.get_R2(strainCompr, stressPredict)
        dof = len(stressCompr_fit)-len(params)

        Chi2 = ufun.get_Chi2(stressCompr_fit, stressPredict_fit, dof, err)

        varK = covM[0,0]
        seK = (varK)**0.5
        
        q = st.t.ppf(alpha, dof) # Student coefficient
        # K, seK = K*1e6, seK*1e6
        confIntK = [K-q*seK, K+q*seK]
        confIntKWidth = 2*q*seK
        # print(Chi2)
        
        # confIntK = st.t.interval(0.95, len(stressPredict)-1, loc=np.mean(stressPredict),\
        #                          scale=st.sem(stressPredict))
        # confIntKWidth = confIntK[1] - confIntK[0]
       
    
    else:
    # except:
        error = True
        K, stressPredict, Chi2, confIntK = -1, np.ones(len(stressCompr))*(-1), -1, [-1, -1]

    
    return(K, stressPredict, Chi2, confIntK, error)

def compressionFitChadwick_weightedLinearFit(hCompr, fCompr, fitCentres_region, H0, DIAMETER):

    error = False
    K3Limit = 50 #Limiting the range of stressCompr to perform Chi2 tests with weighted stressPredict
    
    def computeStress(f, h, H0):
        R = DIAMETER/2000
        delta = (H0 - h)/1000
        stress = f / (np.pi * R * delta)
        return(stress)
        
    def computeStrain(h, H0):
        delta = (H0 - h)/1000
        strain = delta / (3 * (H0/1000))
        return(strain)
    
    def constitutiveRelation(strain, K, stress0):
        stress = (K * strain) + stress0
        return(stress)
    
    def gaussianWeights(fitCentres_region, stressCompr, HR):
        stressCompr = stressCompr.flatten(order='C')
        return np.exp ( -((stressCompr - fitCentres_region) ** 2) / HR ** 2)
    
        
    # def gaussianWeights(fitCentres_region, stressCompr, HR):
    #     stressCompr = stressCompr.flatten(order='C')
    #     return np.exp ( -(stressCompr - fitCentres_region) / HR )


    def getConfInt(Npts, Nparms, parm, se):
        alpha = 0.05
        df = Npts - Nparms
        # se = diag(cov)**0.5
        q = st.t.ppf(1 - alpha / 2, df)
        ConfInt = [parm - q*se, parm, parm + q*se]
        return(ConfInt)


    def weightedLinearFit(strainCompr, stressCompr, fitCentres_region, HR = 50):
        weights = gaussianWeights(fitCentres_region, stressCompr, HR)
        
        wls_model = sm.WLS(stressCompr, sm.add_constant(strainCompr), weights=weights)
        results_wls = wls_model.fit()
        stress0, K = results_wls.params
        cov0 = results_wls.cov_HC0
        seK = cov0[1,1]**0.5

        stressPredict = constitutiveRelation(strainCompr, K, stress0)

        return(K, stressPredict, stress0, seK)
    
    strainCompr = computeStrain(hCompr, H0)
    stressCompr = computeStress(fCompr, hCompr, H0)
    maxStressCompr = np.max(stressCompr)
    
    if fitCentres_region <= maxStressCompr:
        K3, K3_stressPredict, stress0, seK = weightedLinearFit(strainCompr, stressCompr, fitCentres_region)
        
        lowS3, highS3 = int(fitCentres_region - K3Limit),  int(fitCentres_region + K3Limit)
        
        fitConditions = np.where(np.logical_and(K3_stressPredict >= lowS3, \
                                                  K3_stressPredict <= highS3))
        
        stressCompr_fit = stressCompr[fitConditions]
        stressPredict_fit = K3_stressPredict[fitConditions]
        
        
        err = dictSelectionCurve['Error']
        params = K3, stress0
        
        alpha = 0.975
        
        dof = len(stressCompr_fit)-len(params)

        K3_Chi2 = ufun.get_Chi2(stressCompr_fit, stressPredict_fit, dof, err)
        q = st.t.ppf(alpha, dof) # Student coefficient
        K3_confInt = [K3-q*seK, K3+q*seK]
        confIntKWidth = 2*q*seK
        
        
    else:
        error = True
        
        K3, K3_stressPredict, K3_Chi2, K3_confInt = -1, np.ones(len(stressCompr))*(-1), -1, [-1, -1]

    return(K3, K3_stressPredict, K3_Chi2, K3_confInt, error)


def compressionFitChadwick_pointBased(hCompr, fCompr, NbOfTargets, overlap, int_count, H0, DIAMETER):
    
    error = False
    
    def computeStress(f, h, H0):
        R = DIAMETER/2000
        delta = (H0 - h)/1000
        stress = f / (np.pi * R * delta)
        return(stress)
        
    def computeStrain(h, H0):
        delta = (H0 - h)/1000
        strain = delta / (3 * (H0/1000))
        return(strain)
    
    def constitutiveRelation(strain, K, stress0):
        stress = (K * strain) + stress0
        return(stress)
    
    def inversedConstitutiveRelation(stress, K, strain0):
        strain = (stress / K) + strain0
        return(strain)

    # some initial parameter values - must be within bounds
    initK = (3*max(hCompr)*max(fCompr))/(np.pi*(DIAMETER/2)*(max(hCompr)-min(hCompr))**2) # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
    init0 = 0
    
    initialParameters = [initK, init0]

    # bounds on parameters - initial parameters must be within these
    lowerBounds = (0, -np.Inf)
    upperBounds = (np.Inf, np.Inf)
    parameterBounds = [lowerBounds, upperBounds]

    try:
        strainCompr = computeStrain(hCompr, H0)
        stressCompr = computeStress(fCompr, hCompr, H0)
        
        if int_count == 1:
            overlap = 0
            
        lowerLimit = (int_count-1)*NbOfTargets - overlap
        upperLimit = (int_count)*NbOfTargets + overlap
            
        stressCompr_region = stressCompr[lowerLimit:upperLimit]
        
        strainCompr_region = strainCompr[lowerLimit:upperLimit]
            
        params, covM = curve_fit(constitutiveRelation, strainCompr_region, \
                                 stressCompr_region, initialParameters, bounds = parameterBounds)
    
        K4, stress0 = params
        
        stressPredict = constitutiveRelation(strainCompr, K4, stress0)
        
        stressPredict_fit = stressPredict[lowerLimit:upperLimit]
        stressCompr_fit = stressCompr[lowerLimit:upperLimit]
        err = dictSelectionCurve['Error']
        
        alpha = 0.975
        dof = len(stressCompr_fit)-len(params)
        
        # R2 = ufun.get_R2(strainCompr, stressPredict)
        
        K4_Chi2 = ufun.get_Chi2(stressCompr_fit, stressPredict_fit, dof, err)
        
        varK = covM[0,0]
        seK = (varK)**0.5
        
        q = st.t.ppf(alpha, dof) # Student coefficient
        # K, seK = K*1e6, seK*1e6
        K4_confInt = [K4-q*seK, K4+q*seK]
        confIntKWidth = 2*q*seK
        
    except:
        error = True
        K4, stressPredict, K4_Chi2, K4_confInt = -1, np.ones(len(strainCompr))*(-1), -1, [-1,-1]
        
        
    return(K4, stressPredict, K4_Chi2, K4_confInt, error)  # R2, Chi2, confIntK,

#%% Main Function

def fitH0_allMethods(hCompr, fCompr, DIAMETER):
    dictH0 = {}
    
    # findH0_E, H0_Chadwick15, findH0_hPredict, findH0_R2, findH0_Chi2, findH0_confIntE, findH0_confIntH0, findH0_fitError
    Chadwick15_resultTuple = compressionFitChadwick(hCompr[:15], fCompr[:15], DIAMETER)
    H0_Chadwick15 = Chadwick15_resultTuple[1]
    dictH0['H0_Chadwick15'] = H0_Chadwick15
    dictH0['H0_Chadwick15_resTuple'] = Chadwick15_resultTuple
    
    
    Dimitriadis15_resultTuple = compressionFitDimitriadis(hCompr[:15], fCompr[:15], DIAMETER)
    H0_Dimitriadis15 = Dimitriadis15_resultTuple[1]
    dictH0['H0_Dimitriadis15'] = H0_Dimitriadis15
    dictH0['H0_Dimitriadis15_resTuple'] = Dimitriadis15_resultTuple
    
    return(dictH0)



def analyseTimeSeries_meca(f, tsDf, expDf, dictColumnsMeca, task, PLOT, PLOT_SHOW):
    
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
    Ncomp = max(tsDf['idxAnalysis'])
    
    # Field infos
    normalField = thisExpDf.at[thisExpDf.index.values[0], 'normal field']
    normalField = int(normalField)
    if 'compression' in EXPTYPE:
        compField = thisExpDf.at[thisExpDf.index.values[0], 'ramp field'].split('_')
        minCompField = float(compField[0])
        maxCompField = float(compField[1])
    
    # Loop structure infos
    loopStruct = thisExpDf.at[thisExpDf.index.values[0], 'loop structure'].split('_')
    nUplet = thisExpDf.at[thisExpDf.index.values[0], 'normal field multi images']
    if 'compression' in EXPTYPE:
        loop_totalSize = int(loopStruct[0])
        if len(loopStruct) >= 2:
            loop_rampSize = int(loopStruct[1])
        else:
            loop_rampSize = 0
        if len(loopStruct) >= 3:
            loop_excludedSize = int(loopStruct[2])
        else:
            loop_excludedSize = 0
        loop_ctSize = int((loop_totalSize - (loop_rampSize+loop_excludedSize))/nUplet)
        
    
    # Initialize the results
    results = {}
    # for c in listColumnsMeca:
    #     results[c] = []
        
    for k in dictColumnsMeca.keys():
        results[k] = [dictColumnsMeca[k] for m in range(Ncomp)]


    #### (1) Get global values
    # These values are computed once for the whole cell D3 time series, but since the table has 1 line per compression, 
    # that same value will be put in the table for each line corresponding to that cell
    ctFieldH = (tsDf.loc[tsDf['idxAnalysis'] == 0, 'D3'].values - DIAMETER)
    ctFieldDX = np.median(tsDf.loc[tsDf['idxAnalysis'] == 0, 'dx'].values)
    ctFieldDY = np.median(tsDf.loc[tsDf['idxAnalysis'] == 0, 'dy'].values)
    ctFieldDZ = np.median(tsDf.loc[tsDf['idxAnalysis'] == 0, 'dz'].values)
    ctFieldThickness   = np.median(ctFieldH)
    ctFieldFluctuAmpli = np.percentile(ctFieldH, 90) - np.percentile(ctFieldH,10)
    
    #### PLOT [1/4]
    # First part of the plot [mainly ax1 and ax1bis]
    if PLOT:
        # 1st plot - fig1 & ax1, is the F(t) curves with the different compressions colored depending of the analysis success
        fig1, ax1 = plt.subplots(1,1,figsize=(tsDf.shape[0]*(1/100),5))
        color = 'blue'
        ax1.set_xlabel('t (s)')
        ax1.set_ylabel('h (nm)', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        fig1.tight_layout()
        nColsSubplot = 5
        nRowsSubplot = ((Ncomp-1) // nColsSubplot) + 1
        
        # 2nd plot - fig2 & ax2, gather all the F(h) curves, and will be completed later in the code
        fig2, ax2 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
        
        # 3rd plot - fig3 & ax3, gather all the stress-strain curves, and will be completed later in the code
        fig3, ax3 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
        
        # 4th plot - fig4 & ax4, gather all the F(h) curves with local fits, and will be completed later in the code
        fig4, ax4 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
        alreadyLabeled4 = []
        
        # 5th plot - fig5 & ax5, gather all the stress-strain curves with local fits, and will be completed later in the code
        fig5, ax5 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
        alreadyLabeled5 = []
        
        # 6th plot - fig6 & ax6
        fig6, ax6 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
        
        if plotSmallElements:
            # 7th plot - fig7 & ax7, gather all the "small elements", and will be completed later in the code
            fig7, ax7 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))

        # 8th plot - fig8 & ax8 - Gaussian weighted linear fit on the stress-strain curves
        fig8, ax8 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
        alreadyLabeled8 = []
        
        # 9th plot - fig9 & ax9 - K3 modulus vs. Sigma plot to see non-linearity
        fig9, ax9 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
        
        # 10th plot - fig10 & ax10 - K4 modulus vs. Sigma plot - point-base tangential modulus
        fig10, ax10 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
        alreadyLabeled10 = []
        
        # 11th plot - fig11 & ax11 - K4 modulus vs. Sigma plot to see non-linearity
        fig11, ax11 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))

    for i in range(Ncomp):#Ncomp+1):

        #### (1) Identifiers        
        results['date'][i] = ufun.findInfosInFileName(f, 'date')
        results['cellName'][i] = ufun.findInfosInFileName(f, 'cellName')
        results['cellID'][i] = ufun.findInfosInFileName(f, 'cellID')
        results['manipID'][i] = ufun.findInfosInFileName(f, 'manipID')
        
        currentCellID = results['cellID'][i]
        
        #### Jump correction
        if EXPTYPE == 'compressionsLowStart' or normalField == minCompField:
            colToCorrect = ['dx', 'dy', 'dz', 'D2', 'D3']
            maskCompAndPrecomp = np.abs(tsDf['idxAnalysis']) == i+1
            iStart = ufun.findFirst(np.abs(tsDf['idxAnalysis']), i+1)
            for c in colToCorrect:
                jump = np.median(tsDf[c].values[iStart:iStart+5]) - tsDf[c].values[iStart-1]
                tsDf.loc[maskCompAndPrecomp, c] -= jump
                if c == 'D3':
                    D3corrected = True
                    jumpD3 = jump
        else:
            D3corrected = False
            jumpD3 = 0
            
        #### (2) Segment the compression n°i
        thisCompDf = tsDf.loc[tsDf['idxAnalysis'] == i+1,:]
        iStart = (ufun.findFirst(tsDf['idxAnalysis'], i+1))
        iStop = iStart+thisCompDf.shape[0]

        # Easy-to-get parameters
        results['compNum'][i] = i+1
        results['compDuration'][i] = thisExpDf.at[thisExpDf.index.values[0], 'compression duration']
        results['compStartTime'][i] = thisCompDf['T'].values[0]
        results['compAbsStartTime'][i] = thisCompDf['Tabs'].values[0]
        results['compStartTimeThisDay'][i] = thisCompDf['Tabs'].values[0]

        listB = thisCompDf.B.values
        
        # Test to check if most of the compression have not been deleted due to bad image quality 
        highBvalues = (listB > (maxCompField + minCompField)/2)
        N_highBvalues = np.sum(highBvalues)
        testHighVal = (N_highBvalues > 20)

        # Test to check if the range of B field is large enough
        minB, maxB = min(listB), max(listB)
        testRangeB = ((maxB-minB) > 0.7*(maxCompField - minCompField))
        thresholdB = (maxB-minB)/50
        thresholdDeltaB = (maxB-minB)/400

        # Is the curve ok to analyse ?
        doThisCompAnalysis = testHighVal and testRangeB # Some criteria can be added here

        if doThisCompAnalysis:
            #### (3) Inside the compression n°i, delimit the compression and relaxation phases

            # Delimit the start of the increase of B (typically the moment when the field decrease from 5 to 3)
            # and the end of its decrease (typically when it goes back from 3 to 5)
            
            try:
                # Correct for bugs in the B data
                if 'compressions' in EXPTYPE:
                    for k in range(1,len(listB)):
                        B = listB[k]
                        if B > 1.25*maxCompField:
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

                jStart = offsetStart # Beginning of compression
                jMax = np.argmax(thisCompDf.B) # End of compression, beginning of relaxation
                jStop = thisCompDf.shape[0] - offsetStop # End of relaxation
            
            except:
                print(listB)
                print(testRangeB, thresholdB, thresholdDeltaB)

            # Four arrays
            hCompr = (thisCompDf.D3.values[jStart:jMax+1] - DIAMETER)
            hRelax = (thisCompDf.D3.values[jMax+1:jStop] - DIAMETER)
            fCompr = (thisCompDf.F.values[jStart:jMax+1])
            fRelax = (thisCompDf.F.values[jMax+1:jStop])


            # Refinement of the compression delimitation.
            # Remove the 1-2 points at the begining where there is just the viscous relaxation of the cortex
            # because of the initial decrease of B and the cortex thickness increases.
            offsetStart2 = 0
            k = 0
            while (k<len(hCompr)-10) and (hCompr[k] < np.max(hCompr[k+1:min(k+10, len(hCompr))])):
                offsetStart2 += 1
                k += 1
            # Better compressions arrays
            hCompr = hCompr[offsetStart2:]
            fCompr = fCompr[offsetStart2:]
            
            # Get the points of constant field preceding and surrounding the current compression
            # Ex : if the labview code was set so that there is 6 points of ct field before and after each compression,
            # previousPoints will contains D3[iStart-12:iStart]
            # surroundingPoints will contains D3[iStart-6:iStart] and D3[iStop:iStop+6]
            previousPoints = (tsDf.D3.values[max(0,iStart-(loop_ctSize)):iStart]) - DIAMETER
            surroundingPoints = np.concatenate([tsDf.D3.values[max(0,iStart-(loop_ctSize//2)):iStart],tsDf.D3.values[iStop:iStop+(loop_ctSize//2)]]) - DIAMETER
            surroundingPointsX = np.concatenate([tsDf.dx.values[max(0,iStart-(loop_ctSize//2)):iStart],tsDf.dx.values[iStop:iStop+(loop_ctSize//2)]])
            surroundingPointsY = np.concatenate([tsDf.dy.values[max(0,iStart-(loop_ctSize//2)):iStart],tsDf.dy.values[iStop:iStop+(loop_ctSize//2)]])
            surroundingPointsZ = np.concatenate([tsDf.dz.values[max(0,iStart-(loop_ctSize//2)):iStart],tsDf.dz.values[iStop:iStop+(loop_ctSize//2)]])
            
            # Parameters relative to the thickness ( = D3-DIAMETER)
            results['initialThickness'][i] = np.mean(hCompr[0:3])
            results['minThickness'][i] = np.min(hCompr)
            results['maxIndent'][i] = results['initialThickness'][i] - results['minThickness'][i]
            results['previousThickness'][i] = np.median(previousPoints)
            results['surroundingThickness'][i] = np.median(surroundingPoints)
            results['surroundingDx'][i] = np.median(surroundingPointsX)
            results['surroundingDy'][i] = np.median(surroundingPointsY)
            results['surroundingDz'][i] = np.median(surroundingPointsZ)
            results['ctFieldDX'][i] = ctFieldDX
            results['ctFieldDY'][i] = ctFieldDY
            results['ctFieldDZ'][i] = ctFieldDZ
            results['ctFieldThickness'][i] = ctFieldThickness
            results['ctFieldFluctuAmpli'][i] = ctFieldFluctuAmpli
            results['jumpD3'][i] = jumpD3

            validatedThickness = np.min([results['initialThickness'][i],results['minThickness'][i],
                                         results['previousThickness'][i],results['surroundingThickness'][i],
                                         results['ctFieldThickness'][i]]) > 0
            
            results['validatedThickness'][i] = validatedThickness
            results['minForce'][i] = np.min(fCompr)
            results['maxForce'][i] = np.max(fCompr)




            #### (4) Fit with Chadwick model of the force-thickness curve
            
            #### Classic, all curve, Chadwick fit
            E, H0, hPredict, R2, Chi2, confIntE, confIntH0, fitError = compressionFitChadwick(hCompr, fCompr, DIAMETER) # IMPORTANT SUBFUNCTION
  
            R2CRITERION = dictSelectionCurve['R2']
            CHI2CRITERION = dictSelectionCurve['K2_Chi2']
            critFit = 'R2 > ' + str(R2CRITERION)
            
            results['critFit'][i] = critFit
            
            validatedFit = ((R2 > R2CRITERION) and (Chi2 < CHI2CRITERION))


            if not fitError:
                results['validatedFit_Chadwick'][i] = validatedFit
                if validatedFit:
                    results['comments'][i] = 'ok'
                else:
                    results['comments'][i] = 'R2 < ' + str(R2CRITERION)

                confIntEWidth = abs(confIntE[0] - confIntE[1])

                results['H0Chadwick'][i] = H0
                results['EChadwick'][i] = E
                results['R2Chadwick'][i] = R2
                results['EChadwick_CIWidth'][i] = confIntEWidth
                
            elif fitError:
                results['comments'][i] = 'fitFailure'
                
                
            #### (4.0) Find the best H0
            
            dictH0 = fitH0_allMethods(hCompr, fCompr, DIAMETER)
            
            H0_Chadwick15 = dictH0['H0_Chadwick15']
            H0_Dimitriadis15 = dictH0['H0_Dimitriadis15']
            try:
                results['H0_Chadwick15'][i] = H0_Chadwick15
                results['H0_Dimitriadis15'][i] = H0_Dimitriadis15
            except:
                pass
            
            bestH0 = dictH0[H0_bestMethod]
            resTuple_bestH0 = dictH0[H0_bestMethod + '_resTuple']
            error_bestH0 = resTuple_bestH0[-1]

            maxH0 = max(H0, bestH0)
            if max(hCompr) > maxH0:
                error_bestH0 = True   
            
            
            #### (4.1) Compute stress and strain based on the best H0
            if not error_bestH0:
                
                results['bestH0'][i] = bestH0
                results['error_bestH0'][i] = error_bestH0
                
                deltaCompr = (maxH0 - hCompr)/1000
                stressCompr = fCompr / (np.pi * (DIAMETER/2000) * deltaCompr)

                strainCompr = deltaCompr / (3*(maxH0/1000))
                validDelta = (deltaCompr > 0)
                
                
                results['minStress'][i] = np.min(stressCompr)
                results['maxStress'][i] = np.max(stressCompr)
                results['minStrain'][i] = np.min(strainCompr)
                results['maxStrain'][i] = np.max(strainCompr)
                
                

            #### (4.2) Fits on specific regions of the curve
            
                K2_list_strainPredict_fitToPlot = [[] for kk in range(len(fit_toPlot))]
                K3_list_stressPredict_fitToPlot = [[] for kk in range(len(fit_toPlot))]
                
                N_Fits = len(regionFitsNames)
                dictRegionFit = {}
                for k in dictColumnsRegionFit.keys():
                    dictRegionFit[k] = [dictColumnsRegionFit[k] for m in range(N_Fits)]


                all_masks_region = np.zeros((N_Fits, len(stressCompr)), dtype = bool)
                for kk in range(N_Fits):
                    ftP = regionFitsNames[kk]
                    lowS, highS = int(fitMin[kk]), int(fitMax[kk])
                    all_masks_region[kk,:] = ((stressCompr > lowS) & (stressCompr < highS))                
                    
                for ii in range(N_Fits):
                    regionFitName = regionFitsNames[ii]
                    mask_region = all_masks_region[ii]
                    Npts_region = np.sum(mask_region)
                    fCompr_region = fCompr[mask_region]
                    hCompr_region = hCompr[mask_region]
                    
                    if Npts_region > 5:
                        # fCompr_region = fCompr[mask_region]
                        # hCompr_region = hCompr[mask_region]
                        
                        # Vérifier la concordance !
                        resTuple_FHregionFit = compressionFitChadwick(hCompr_region, fCompr_region, DIAMETER)
                        K2_region, H0_region = resTuple_FHregionFit[0:2]
                        
                        K2_region, K2_strainPredict_region, K2_R2_region, K2_Chi2_region, K2_confIntK_region, \
                        K2_fitError_region = compressionFitChadwick_StressStrain(hCompr_region, fCompr_region, maxH0, DIAMETER)
                            
                        K2_confIntWidthK_region = np.abs(K2_confIntK_region[0] - K2_confIntK_region[1])
            
                    else:
                        
                        K2_fitError_region = True
                        
                        
                    if (regionFitName in fit_toPlot) and not K2_fitError_region:
                        kk = np.argmax(np.array(fit_toPlot) == regionFitName)
                        K2_list_strainPredict_fitToPlot[kk] = K2_strainPredict_region

                    
                    if not K2_fitError_region:
                        R2CRITERION = dictSelectionCurve['R2']
                        CHI2CRITERION = dictSelectionCurve['K2_Chi2']
                        K2_validatedFit_region = ((K2_R2_region > R2CRITERION) and 
                                                (K2_Chi2_region < CHI2CRITERION))
                        
                        # Fill dictRegionFit (reinitialized for each cell)
                        dictRegionFit['regionFitNames'][ii] = regionFitName
                        dictRegionFit['Npts'][ii] = Npts_region
                        dictRegionFit['K'][ii] = K2_region
                        dictRegionFit['K2_CIW'][ii] = K2_confIntWidthK_region
                        dictRegionFit['R2'][ii] = K2_R2_region
                        dictRegionFit['K2'][ii] = K2_region
                        dictRegionFit['H0'][ii] = H0_region
                        # dictRegionFit['E'][ii] = E_region
                        dictRegionFit['K2_fitError'][ii] = K2_fitError_region
                        dictRegionFit['K2_validatedFit'][ii] = K2_validatedFit_region
                        
                        # Fill results (memorize everything)
                        rFN = regionFitName
                        results['Npts_'+rFN][i] = Npts_region
                        results['KChadwick_'+rFN][i] = K2_region
                        results['K2_CIW_'+rFN][i] = K2_confIntWidthK_region
                        results['R2Chadwick_'+rFN][i] = K2_R2_region
                        results['K2Chadwick_'+rFN][i] = K2_region
                        results['H0Chadwick_'+rFN][i] = H0_region
                        # results['EChadwick_'+rFN][i] = dictRegionFit['E'][ii]
                        results['K2_validatedFit_'+rFN][i] = K2_validatedFit_region
                    
                    
                    #### (4.1.2) Gaussian-weight fits on specific regions of the curve based on predefined stress-ranges
                    
                    # try:
                    fitCentres_region = fitCenters[ii]
                    K3_region, K3_stressPredict_region, K3_Chi2_region, K3_confIntK_region, \
                    K3_fitError_region = compressionFitChadwick_weightedLinearFit(hCompr, fCompr, fitCentres_region, H0, DIAMETER)
                    
                    K3_confIntWidthK_region = np.abs(K3_confIntK_region[0] - K3_confIntK_region[1])
                    # except:
                    #     K3_fitError_region = True
                    
                    
                    if (regionFitName in fit_toPlot) and not K3_fitError_region:
                        kk = np.argmax(np.array(fit_toPlot) == regionFitName)
                        
                        K3_list_stressPredict_fitToPlot[kk] = K3_stressPredict_region
                    
                    if not K3_fitError_region:
                        CHI2CRITERION = dictSelectionCurve['K3_Chi2']
                        K3_validatedFit_region = ((K3_Chi2_region < CHI2CRITERION))
                        
                        # Fill dictRegionFit (reinitialized for each cell)
                        dictRegionFit['K3_CIW'][ii] = K3_confIntWidthK_region
                        dictRegionFit['K3'][ii] = K3_region
                        dictRegionFit['K3_fitError'][ii] = K3_fitError_region
                        dictRegionFit['K3_validatedFit'][ii] = K3_validatedFit_region
                        
                        # Fill results (memorize everything)
                        rFN = regionFitName
                        results['K3_CIW_'+rFN][i] = K3_confIntWidthK_region
                        results['K3Chadwick_'+rFN][i] = K3_region
                        results['K3_validatedFit_'+rFN][i] = K3_validatedFit_region
                        
                #### (4.1.3) Fits on specific regions of the curve based on the number of points
                
                K4_list_stressPredict_fitToPlot = [[] for kk in range(NbOfTargets)]
                validTargets = [0 for kk in range(NbOfTargets)]
                CHI2CRITERION = dictSelectionCurve['K4_Chi2']
                
                for count in range(1, NbOfTargets):
                    if count == 1:
                        overlap = 0
                        
                    K4_region, K4_stressPredict_region, K4_Chi2_region, K4_confInt_region, K4_fitError_region \
                        = compressionFitChadwick_pointBased(hCompr, fCompr, NbOfTargets, overlap, count, H0, DIAMETER)
                    
                    K4_confIntWidthK_region = np.abs(K4_confInt_region[0] - K4_confInt_region[1])
                    # print(K4_region)
                    if not K4_fitError_region:
                        kk = count - 1
                        K4_list_stressPredict_fitToPlot[kk] = K4_stressPredict_region
                        validTargets[kk] = count
                    
                        K4_validatedFit_region = (K4_Chi2_region < CHI2CRITERION)
                                                
                    else:
                        K4_validatedFit_region, K4_fitError_region = False, True
                        K4_region, K4_confInt_region = np.nan, [np.nan, np.nan]
                        K4_confIntWidthK_region = np.nan
                        
                    if not K4_fitError_region: 
                        kk = count - 1
                        targetPoint = str(int(count*NbOfTargets))
                        dictRegionFit['K4'][kk] = (K4_region)
                        dictRegionFit['K4_CIW'][kk] = (K4_confIntWidthK_region)
                        dictRegionFit['K4_fitError'][kk] = (K4_fitError_region)
                        dictRegionFit['K4_validatedFit'][kk] = (K4_validatedFit_region)
                        
                        results['K4Chadwick_'+targetPoint][i] = (K4_region)
                        results['K4_validatedFit_'+targetPoint][i] = (K4_validatedFit_region)
 
            
                for k in dictRegionFit.keys():
                    dictRegionFit[k] = np.array(dictRegionFit[k])
            

            #### PLOT [2/4]
            # Complete fig 1, 2, 3 with the results of the fit
            # print(dictRegionFit['K4'])
            if PLOT:
                
                #### fig1
                if not fitError:
                    ax1.plot(thisCompDf['T'].values, thisCompDf['D3'].values-DIAMETER, color = 'chartreuse', linestyle = '-', linewidth = 1.25)
                    
                    # if validatedFit:
                    #     ax1.plot(thisCompDf['T'].values, thisCompDf['D3'].values-DIAMETER, color = 'chartreuse', linestyle = '-', linewidth = 1.25)
                    # else:
                    #     ax1.plot(thisCompDf['T'].values, thisCompDf['D3'].values-DIAMETER, color = 'gold', linestyle = '-', linewidth = 1.25)
                else:
                    ax1.plot(thisCompDf['T'].values, thisCompDf['D3'].values-DIAMETER, color = 'crimson', linestyle = '-', linewidth = 1.25)
                
                # Display jumpD3 >>> DISABLED
                # if jumpD3 != 0:
                #     x = np.mean(thisCompDf['T'].values)
                #     y = np.mean(thisCompDf['D3'].values-DIAMETER) * 1.3
                #     ax1.text(x, y, '{:.2f}'.format(jumpD3), ha = 'center')

                fig1.suptitle(currentCellID)
                

                #### fig2 & fig3
                colSp = (i) % nColsSubplot
                rowSp = (i) // nColsSubplot
                # ax2[i-1] with the 1 line plot
                if nRowsSubplot == 1:
                    thisAx2 = ax2[colSp]
                    thisAx3 = ax3[colSp]
                elif nRowsSubplot >= 1:
                    thisAx2 = ax2[rowSp,colSp]
                    thisAx3 = ax3[rowSp,colSp]

                thisAx2.plot(hCompr,fCompr,'b-', linewidth = 0.8)
                thisAx2.plot(hRelax,fRelax,'r-', linewidth = 0.8)
                titleText = currentCellID + '__c' + str(i+1)
                legendText = ''
                thisAx2.set_xlabel('h (nm)')
                thisAx2.set_ylabel('f (pN)')


                if not fitError:
                    legendText += 'H0 = {:.1f}nm\nE = {:.2e}Pa\nR2 = {:.3f}\nChi2 = {:.1f}'.format(H0, E, R2, Chi2)
                    thisAx2.plot(hPredict,fCompr,'k--', linewidth = 0.8, label = legendText, zorder = 2)
                    thisAx2.legend(loc = 'upper right', prop={'size': 6})
                    # if not validatedFit:
                    #     titleText += '\nNON VALIDATED'
                        
                    if not error_bestH0:
                        thisAx3.plot(stressCompr, strainCompr, 'go', ms = 3)
                        # thisAx3.plot(stressCompr, trueStrainCompr, 'bo', ms = 2)
                        thisAx3.plot([np.percentile(stressCompr,10), np.percentile(stressCompr,90)], 
                                     [np.percentile(stressCompr,10)/E, np.percentile(stressCompr,90)/E],
                                     'k--', linewidth = 1.2, label = legendText)
                        thisAx3.legend(loc = 'lower right', prop={'size': 6})
                    thisAx3.set_xlabel('Stress (Pa)')
                    thisAx3.set_ylabel('Strain')
                    
                else:
                    titleText += '\nFIT ERROR'
                    
                if not error_bestH0:
                    # Computations to display the fit of the H0_Chadwick15
                    
                    bestH0_resTuple = dictH0[H0_bestMethod + '_resTuple']
                    # (E, H0, fPredict, R2, Chi2, confIntE, confIntH0, error)
                    NbPts_bestH0 = 15
                    E_bestH0 = bestH0_resTuple[0]
                    hPredict_bestH0 = bestH0_resTuple[2]
                    
                    min_f = np.min(fCompr)
                    low_f = np.linspace(0, min_f, 20)
                    R = DIAMETER/2
                    low_h = bestH0 - ((3*H0_Chadwick15*low_f)/(np.pi*(E_bestH0/1e6)*R))**0.5

                    
                    legendText2 = 'bestH0 = {:.2f}nm'.format(bestH0)
                    plot_startH = np.concatenate((low_h, hPredict_bestH0[:]))
                    plot_startF = np.concatenate((low_f, fCompr[:NbPts_bestH0]))
                    thisAx2.plot(plot_startH[0], plot_startF[0], ls = '', 
                                  marker = 'o', color = 'skyblue', markersize = 5, label = legendText2)
                    thisAx2.plot(plot_startH, plot_startF, ls = '--', color = 'skyblue', linewidth = 1.2, zorder = 1)
                    thisAx2.legend(loc = 'upper right', prop={'size': 6})
                
                
                multiAxes = [thisAx2, thisAx3]
                
                for ax in multiAxes:
                    ax.title.set_text(titleText)
                    for item in ([ax.title, ax.xaxis.label, \
                                  ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(9)
            
                
                if not error_bestH0:
                    
                    #### fig4 & fig5
                    
                    Npts_fitToPlot = dictRegionFit['Npts'][mask_fitToPlot]
                    K_fitToPlot = dictRegionFit['K'][mask_fitToPlot]
                    K2_CIW_fitToPlot =dictRegionFit['K2_CIW'][mask_fitToPlot]
                    K2_fitToPlot = dictRegionFit['K2'][mask_fitToPlot]
                    H0_fitToPlot = dictRegionFit['H0'][mask_fitToPlot]
                    R2_fitToPlot = dictRegionFit['R2'][mask_fitToPlot]
                    K2_fitError_fitToPlot = dictRegionFit['K2_fitError'][mask_fitToPlot]
                    K2_validatedFit_fitToPlot = dictRegionFit['K2_validatedFit'][mask_fitToPlot]
                    
                    fitToPlot_masks_region = np.array(all_masks_region)[mask_fitToPlot]
                    
                    # else:
                    #     fitError_fitToPlot = np.ones_like(fit_toPlot, dtype = bool)
                        
                    
                    
                    
                    fitToPlot_masks_region = np.array(all_masks_region)[mask_fitToPlot]
                    
                    # ax2[i] with the 1 line plot
                    if nRowsSubplot == 1:
                        thisAx4 = ax4[colSp]
                        thisAx5 = ax5[colSp]
                    elif nRowsSubplot >= 1:
                        thisAx4 = ax4[rowSp,colSp]
                        thisAx5 = ax5[rowSp,colSp]
                    
                    main_color = 'k' # colorList10[0]
                    
                    thisAx4.plot(hCompr,fCompr, color = main_color, marker = 'o', markersize = 2, ls = '', alpha = 0.8) # 'b-'
                    # thisAx4.plot(hRelax,fRelax, color = main_color, ls = '-', linewidth = 0.8, alpha = 0.5) # 'r-'
                    titleText = currentCellID + '__c' + str(i+1)
                    thisAx4.set_xlabel('h (nm)')
                    thisAx4.set_ylabel('f (pN)')
                    
    
                    for k in range(len(fit_toPlot)):
                        if not K2_fitError_fitToPlot[k]:
    
                            fit = fit_toPlot[k]
                            
                            Npts_fit = Npts_fitToPlot[k]
                            K_fit = K_fitToPlot[k]
                            K2_fit = K2_fitToPlot[k]
                            H0_fit = H0_fitToPlot[k]
                            R2_fit = R2_fitToPlot[k]
                            fitError_fit = K2_fitError_fitToPlot[k]
                            validatedFit_fit = K2_validatedFit_fitToPlot[k]
                            fitConditions_fit = fitToPlot_masks_region[k]
                            
                            R = DIAMETER/2
                            fCompr_fit = fCompr[fitConditions_fit]
                            
                            hPredict_fit2 = H0_fit - ((3*H0_fit*fCompr_fit)/(np.pi*(K2_fit/1e6)*R))**0.5
                            
                            color = gs.colorList30[k]
                            legendText4 = ''
                            
                            # hPredict_fit0 = H0_Chadwick15 - ((3*H0_Chadwick15*fCompr_fit)/(np.pi*(K_fit/1e6)*R))**0.5
                            # strainPredict_fit = K2_list_strainPredict_fitToPlot[k]
                            # hPredict_fit = H0_Chadwick15 * (1 - 3*strainPredict_fit)
                            # legendText4 += 'Range ' + fit + '\n'
                            # legendText4 += 'K = {:.2e}Pa'.format(K_fit)
                            # thisAx4.plot(hPredict_fit, fCompr_fit, color = color, ls = '--', linewidth = 1.8, label = legendText4)
                            
                            if fit not in alreadyLabeled4:
                                alreadyLabeled4.append(fit)
                                legendText4 += '' + fit + ''
                                thisAx4.plot(hPredict_fit2, fCompr_fit, color = color, ls = '-', linewidth = 1.8, label = legendText4)
                            elif fit in alreadyLabeled4:
                                thisAx4.plot(hPredict_fit2, fCompr_fit, color = color, ls = '-', linewidth = 1.8)
                        else:
                            pass
                    
                    
                    
                    
                    thisAx5.set_xlabel('Strain')
                    thisAx5.set_ylabel('Stress (Pa)')
                    if not fitError:
                        thisAx5.plot(strainCompr, stressCompr, color = main_color, marker = 'o', markersize = 2, ls = '', alpha = 0.8)
                        
                        for k in range(len(fit_toPlot)):
                            fit = fit_toPlot[k]
                            
                            Npts_fit = Npts_fitToPlot[k]
                            K_fit = K_fitToPlot[k]
                            R2_fit = R2_fitToPlot[k]
                            fitError_fit = K2_fitError_fitToPlot[k]
                            validatedFit_fit = K2_validatedFit_fitToPlot[k]
                            fitConditions_fit = fitToPlot_masks_region[k]
                            
                            stressCompr_fit = stressCompr[fitConditions_fit]
                            strainPredict_fit = K2_list_strainPredict_fitToPlot[k]
                            
                            color = gs.colorList30[k]
                            legendText5 = ''
                            
                            if not fitError_fit:
                                if fit not in alreadyLabeled5:
                                    alreadyLabeled5.append(fit)
                                    legendText5 += '' + fit + ''
                                    thisAx5.plot(strainPredict_fit, stressCompr_fit,  
                                                 color = color, ls = '-', linewidth = 1.8, label = legendText5)
                                elif fit in alreadyLabeled5:
                                    thisAx5.plot(strainPredict_fit, stressCompr_fit,  
                                                 color = color, ls = '-', linewidth = 1.8)
                            else:
                                pass
    
                    multiAxes = [thisAx4, thisAx5] # thisAx5 #
                    
                    for ax in multiAxes:
                        ax.title.set_text(titleText)
                        for item in ([ax.title, ax.xaxis.label, \
                                      ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                            item.set_fontsize(9)
                            
                            
                    #### fig6
                    
                    fitCentersPlot = fitCenters[mask_fitToPlot]
                    
                    if nRowsSubplot == 1:
                        thisAx6 = ax6[colSp]
                    elif nRowsSubplot >= 1:
                        thisAx6 = ax6[rowSp,colSp]
                    
                    
                    thisAx6.set_xlabel('sigma (Pa)')
                    thisAx6.set_xlim([0, 1200])
                    thisAx6.set_ylabel('K (kPa)')
                    
                    relativeError = np.zeros(len(K_fitToPlot))
                    if not fitError:
                        
                        for k in range(len(fit_toPlot)):
                            fit = fit_toPlot[k]
                            
                            K_fit = K_fitToPlot[k]
                            K_CIW_fit = K2_CIW_fitToPlot[k]
                            fitError_fit = K2_fitError_fitToPlot[k]
                            validatedFit_fit = K2_validatedFit_fitToPlot[k]
                            fitConditions_fit = fitToPlot_masks_region[k]
                            
                            stressCompr_fit = stressCompr[fitConditions_fit]
                            strainPredict_fit = K2_list_strainPredict_fitToPlot[k]
                            
                            color = gs.colorList30[k]
                            
                            if not fitError_fit:
                                
                                Err = K_CIW_fit
                                relativeError[k] = (Err/K_fit)
                                # mec = None
                                thisAx6.errorbar([fitCentersPlot[k]], [K_fit/1000], yerr = [(Err/2)/1000],
                                              color = color, marker = 'o', ms = 5) #, mec = mec)                           
                                
                            
                        multiAxes = [thisAx6]
                        
                        for ax in multiAxes:
                            ax.title.set_text(titleText)
                            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + \
                                         ax.get_xticklabels() + ax.get_yticklabels()):
                                item.set_fontsize(9)
                                
                    
                            
                    #### fig7
                    if plotSmallElements:
    
                        if nRowsSubplot == 1:
                            thisAx7 = ax7[colSp]
                        elif nRowsSubplot >= 1:
                            thisAx7 = ax7[rowSp,colSp]
                            
                        thisAx7.set_xlabel('epsilon')
                        thisAx7.set_ylabel('ratios')
                        legendText7 = ''
                        
                        # def def2delta(e):
                        #     d = 3*H0_Chadwick15*e
                        #     return(d)
                        
                        # def delta2def(d):
                        #     e = d/(3*H0_Chadwick15)
                        #     return(e)
                        
                        # secax = thisAx7.secondary_xaxis('top', functions=(def2delta, delta2def))
                        # secax.set_xlabel('delta (nm)')
                        
                        if not fitError:
                            
                            A = 2* ((deltaCompr*(DIAMETER/2000))**0.5)
                            largeX = A/(maxH0/1000)
                            smallX = deltaCompr/(DIAMETER/1000)
                            # legendText7 = 'H0_Chadwick15 = {:.2f}nm'.format(H0_Chadwick15)
                            
                            thisAx7.plot(strainCompr, largeX, color = 'red',     ls = '', marker = '+', label = 'a/H0',     markersize = 3)#, mec = 'k', mew = 0.5)
                            thisAx7.plot(strainCompr, smallX, color = 'skyblue', ls = '', marker = '+', label = 'delta/2R', markersize = 3)#, mec = 'k', mew = 0.5)
                            thisAx7.legend(loc = 'upper left', prop={'size': 5})
                            thisAx7.set_yscale('log')
                            thisAx7.set_ylim([5e-3,10])
                            minPlot, maxPlot = thisAx7.get_xlim()
                            thisAx7.set_xlim([0,maxPlot])
                            thisAx7.plot([0,maxPlot], [1, 1], color = 'k', ls = '--', lw = 0.5)
                            
                            # # thisAx7bis.tick_params(axis='y', labelcolor='b')
                            # thisAx7bis = thisAx7.twinx()
                            # color = 'firebrick'
                            # thisAx7bis.set_ylabel('F (pN)', color=color)
                            # thisAx7bis.plot(strainCompr, fCompr, color=color, lw = 0.5)
                            # thisAx7bis.set_ylim([0,1400])
                            # thisAx7bis.tick_params(axis='y', labelrotation = 50, labelsize = 10)
                            # # thisAx7bis.tick_params(axis='y', labelcolor=color)
                            # # thisAx7bis.set_yticks([0,500,1000,1500])
                            # # minh = np.min(tsDf['D3'].values-DIAMETER)
                            
                            
                            epsLim = (largeX < 1.0)
                            if len(strainCompr[epsLim]) > 0:
                                strainLimit = np.max(strainCompr[epsLim])
                                minPlot, maxPlot = thisAx5.get_ylim()
                                # thisAx5.plot([strainLimit, strainLimit], [minPlot, maxPlot], color = 'gold', ls = '--')
                                minPlot, maxPlot = thisAx7.get_ylim()
                                thisAx7.plot([strainLimit, strainLimit], [minPlot, maxPlot], color = 'gold', ls = '--')
                            
                                    
                            multiAxes = [thisAx7] #, thisAx7bis]
                            
                            for ax in multiAxes:
                                ax.title.set_text(titleText + '\nbestH0 = {:.2f}nm'.format(maxH0))
                                for item in ([ax.title, ax.xaxis.label, \
                                              ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                                    item.set_fontsize(9)
                                    
                     #### fig8
                     
                     
                    if nRowsSubplot == 1:
                        thisAx8 = ax8[colSp]
                    elif nRowsSubplot >= 1:
                        thisAx8 = ax8[rowSp,colSp]
                        
                    K_fitToPlot = dictRegionFit['K3'][mask_fitToPlot]
                    K3_CIW_fitToPlot =  dictRegionFit['K3_CIW'][mask_fitToPlot]
                    K3_fitError_fitToPlot = dictRegionFit['K3_fitError'][mask_fitToPlot]
                    K3_validatedFit_fitToPlot = dictRegionFit['K3_validatedFit'][mask_fitToPlot]
                    fitCentersPlot = fitCenters[mask_fitToPlot]
                    
                    thisAx8.set_xlabel('epsilon')
                    thisAx8.set_ylabel('sigma (Pa)')
                    K3Limit = 50
                    allFitConditions = []
                    if not fitError:
                        thisAx8.plot(strainCompr, stressCompr, color = main_color, marker = 'o', markersize = 3, ls = '', alpha = 0.8)
                    
                        for k in range(len(fit_toPlot)):
                            fit = fit_toPlot[k]
                            fitCenters_region = fitCentersPlot[k]
                            fitError_fit = K3_fitError_fitToPlot[k]
                            stressPredict_region = np.asarray(K3_list_stressPredict_fitToPlot[k])
                            # print(gs.ORANGE + 'strain predict' + gs.NORMAL)
                            # print(stressPredict_region)
                            lowS3, highS3 = int(fitCenters_region - K3Limit),  int(fitCenters_region + K3Limit)
                            
                            fitConditions = np.where(np.logical_and(stressPredict_region >= lowS3, \
                                                                      stressPredict_region <= highS3))
                            
                            
                            allFitConditions.extend(fitConditions)    
                                
                            strainPredict_fit = strainCompr[fitConditions]
                            stressPredict_region = stressPredict_region[fitConditions]
                            
                            
                            
                            color = gs.colorList30[k]
                            legendText8 = ''
                            
                            if not fitError_fit:             
                                if fit not in alreadyLabeled8:
                                    alreadyLabeled8.append(fit)
                                    legendText8 += '' + fit + ''
        
                                    thisAx8.plot(strainPredict_fit, stressPredict_region,  
                                                 color = color, ls = '-', linewidth = 1.8, label = legendText8)
                                elif fit in alreadyLabeled8:
                                    thisAx8.plot(strainPredict_fit, stressPredict_region,  
                                                 color = color, ls = '-', linewidth = 1.8)

                                
                    multiAxes = [thisAx8]
                            
                    for ax in multiAxes:
                        ax.title.set_text(titleText)
                        for item in ([ax.title, ax.xaxis.label, \
                                      ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                            item.set_fontsize(9)
                    
                    #### fig9
                
                    fitCentersPlot = fitCenters[mask_fitToPlot]
                    K3_fitToPlot = dictRegionFit['K3'][mask_fitToPlot]
                    
                        
                    if nRowsSubplot == 1:
                        thisAx9 = ax9[colSp]
                    elif nRowsSubplot >= 1:
                        thisAx9 = ax9[rowSp,colSp]
                    
                    thisAx9bis = thisAx9.twinx()
                    
                    thisAx9.set_xlabel('sigma (Pa)')
                    thisAx9.set_xlim([0,1000])
                    thisAx9.set_yscale('log')
                    thisAx9.set_ylabel('K (Pa)')
                    
                    # thisAx9bis.set_yscale('linear')
                    # thisAx9bis.set_ylabel('relative error', color='red')
                    # relErrFilter = 0.5
                    
                    # relativeError = np.zeros(len(list_stressPredictK3_fitToPlot))
                    # if not fitError and not findH0_fitError:
                    if not fitError:
                        for k in range(len(fit_toPlot)):
                        
                            fit = fit_toPlot[k]
                            K3_fit = K3_fitToPlot[k]
                            fitCenters_region = fitCentersPlot[k]
                            K3_CIW_fit = K3_CIW_fitToPlot[k]
                            K3_fitError_fit = K3_fitError_fitToPlot[k]
                            K3_validatedFit_fit = K3_validatedFit_fitToPlot[k]
    
                            color = gs.colorList30[k]
                            
                            thisAx9.plot([fitCentersPlot[k]], [K3_fit], color = color, marker = 'o') 
                            
                            # if not fitError_fit:
                                
                            #     E = K3_CIW_fit
                            #     relativeError[k] = (E/K3_fit)
                            #     mec = None
                            #     if (E/K3_fit) > relErrFilter:
                            #         # print('true')
                            #         mec = 'orangered'
                                
                            thisAx9.errorbar([fitCentersPlot[k]], [K3_fit], yerr = [E/2],
                                         color = color, marker = 'o', ms = 5) #, mec = mec)                           
                                # print(K3_fit)
                                # print('ciw')
                                # print(E)
                                # print('fit')
                                # print(K3_fit)
                        # print(fitCentersPlot)
                        # relativeError_subset = relativeError[relativeError != 0]
                        # fitCenters_subset = fitCentersPlot[relativeError != 0]
                        # thisAx9bis.plot([0,1000], [relErrFilter, relErrFilter], ls = '--', color = 'red', lw = 0.5)
                        # thisAx9bis.plot(fitCenters_subset, relativeError_subset, marker = 'd', ms = 3, color = 'red', ls = '')
                        
                        # thisAx9bis.set_ylim([0,2])
                        # thisAx9bis.tick_params(axis='y', labelcolor='red')
                        # thisAx9bis.set_yticks([0,0.5,1,1.5,2])
                                                    
                    multiAxes = [thisAx9] #, thisAx9bis]
                    
                    for ax in multiAxes:
                        ax.title.set_text(titleText)
                        for item in ([ax.title, ax.xaxis.label, \
                                      ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                            item.set_fontsize(9)
                
                    #### fig10
                    
                    if nRowsSubplot == 1:
                        thisAx10 = ax10[colSp]
                    elif nRowsSubplot >= 1:
                        thisAx10 = ax10[rowSp,colSp]
                        
                    thisAx10.set_xlabel('epsilon')
                    thisAx10.set_ylabel('sigma (Pa)')
                    K4Limit = 50
                    if not fitError:
                        thisAx10.plot(strainCompr, stressCompr, color = main_color, marker = 'o', markersize = 3, ls = '', alpha = 0.8)
                        for k in range(len(validTargets)):
                            target = validTargets[k]
                            # print(target)
                            if target != 0:
                                stressPredict_region = np.asarray(K4_list_stressPredict_fitToPlot[k])
                                lowerLimit = (target-1)*NbOfTargets - overlap
                                upperLimit = target*NbOfTargets + overlap
                                
                                strainPredict_region = strainCompr[lowerLimit:upperLimit]
                                stressPredict_region = stressPredict_region[lowerLimit:upperLimit]
                            
                                color = gs.colorList30[k]
                                legendText10 = ''
                                
                            # print('Compression {:.0f}'.format(i))
    
                                thisAx10.plot(strainPredict_region, stressPredict_region, \
                                                      color = color, ls = '-', linewidth = 1.8)
                
                    multiAxes = [thisAx10]
                                
                    for ax in multiAxes:
                        ax.title.set_text(titleText)
                        for item in ([ax.title, ax.xaxis.label, \
                                      ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                            item.set_fontsize(9)
                
                
                    ####fig11
                    
                    fitToPlot = (np.linspace(0, max(validTargets),max(validTargets) + 1))
                    fitToPlot = [int(x) for x in fitToPlot]
                    K4_fitToPlot = dictRegionFit['K4'][fitToPlot]
                    K4_CIW_fitToPlot = dictRegionFit['K4_CIW'][fitToPlot]
                    K4_fitError_fitToPlot = dictRegionFit['K4_fitError'][fitToPlot]
                    K4_validatedFit_fitToPlot = dictRegionFit['K4_validatedFit'][fitToPlot]
                    
                    if nRowsSubplot == 1:
                        thisAx11 = ax11[colSp]
                    elif nRowsSubplot >= 1:
                        thisAx11 = ax11[rowSp,colSp]
                    
                    thisAx11bis = thisAx11.twinx()
                    
                    thisAx11.set_xlabel('Target Point')
                    thisAx11.set_xlim([0,NbOfTargets])
                    thisAx11.set_yscale('log')
                    thisAx11.set_ylabel('K (Pa)')
                    
                    # thisAx11bis.set_yscale('linear')
                    # thisAx11bis.set_ylabel('relative error', color='red')
                    # relErrFilter = 0.5
                    
                    # relativeError = np.zeros(len(K4_list_stressPredict_fitToPlot))
                    validTargets = np.asarray(validTargets)
                    if not fitError:
                        for k in range(1, max(validTargets) + 1):
                            color = gs.colorList30[k]
                            target = k - 1
                            
                            # if validTargets[target] != 0:
                            K4_fit = K4_fitToPlot[target]
                            
                            K4_CIW_fit = K4_CIW_fitToPlot[target]
                            fitError_fit = K4_fitError_fitToPlot[target]
                            validatedFit_fit = K4_validatedFit_fitToPlot[target]

                            thisAx11.plot([validTargets[target]], [K4_fit], color = color, marker = 'o') 
                            # if not K4_fitError_fit:
                                
                            #     E = K4_CIW_fit
                            #     relativeError[target] = (E/K4_fit)
                            #     mec = None
                            #     if (E/K4_fit) > relErrFilter:
                            #         # print('true')
                            #         mec = 'orangered'
                            thisAx11.errorbar([validTargets[target]], [K4_fit], yerr = [E/2],
                                          color = color, marker = 'o', ms = 5) #, mec = mec)
                                
                    # relativeError_subset = relativeError[relativeError != 0]
                                     
                    # validTargets_subset = validTargets[relativeError != 0]
                    # thisAx11bis.plot([0,1000], [relErrFilter, relErrFilter], ls = '--', color = 'red', lw = 0.5)
                    # thisAx11bis.plot(validTargets_subset, relativeError_subset, marker = 'd', ms = 3, color = 'red', ls = '')
                    
                    # thisAx11bis.set_ylim([0,2])
                    # thisAx11bis.tick_params(axis='y', labelcolor='red')
                    # thisAx11bis.set_yticks([0,0.5,1,1.5,2])
                                            
                    multiAxes = [thisAx11] #, thisAx11bis]
                    
                    for ax in multiAxes:
                        ax.title.set_text(titleText)
                        for item in ([ax.title, ax.xaxis.label, \
                                      ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                            item.set_fontsize(9)        

            # #### (5) hysteresis (its definition may change)
            # try:
            #     # results['hysteresis'].append(hCompr[0] - hRelax[-1])
            #     results['hysteresis'][i] = hCompr[0] - hRelax[-1]
                
            # except:
            #     print('hysteresis computation failed, see hCompr and hRelax below:')
            #     print(hCompr, hRelax)
            #     print('Details of the delimitation of the curve')
            #     print(jStart, jMax, jStop, offsetStop, thisCompDf.shape[0])
            #     print(offsetStart2)
            #     print(thisCompDf.B.values)
                # results['hysteresis'].append(np.nan)
        
        #### (6) Deal with the non analysed compressions
        else: # The compression curve was detected as not suitable for analysis
            generalBug = True
            results['comments'][i] = 'Unspecified bug in the code'

        if not doThisCompAnalysis:
            print('Curve not suitable for analysis !')
            print(currentCellID)
            print('Compression no ' + str(i+1))

    for k in results.keys():
        results[k] = np.array(results[k])
    
    #### PLOT [3/4]
    
    if PLOT:
        #### fig1
        color = 'blue'
        ax1.plot(tsDf['T'].values, tsDf['D3'].values-DIAMETER, color = color, ls = '-', linewidth = 1, zorder = 1)
        (axm, axM) = ax1.get_ylim()
        ax1.set_ylim([min(0,axm), axM])
        if (max(tsDf['D3'].values-DIAMETER) > 200):
            ax1.set_yticks(np.arange(0, max(tsDf['D3'].values-DIAMETER), 100))
        
        twinAxis = True
        if twinAxis:
            ax1.tick_params(axis='y', labelcolor='b')
            ax1bis = ax1.twinx()
            color = 'firebrick'
            ax1bis.set_ylabel('F (pN)', color=color)
            ax1bis.plot(tsDf['T'].values, tsDf['F'].values, color=color)
            ax1bis.tick_params(axis='y', labelcolor=color)
            ax1bis.set_yticks([0,500,1000,1500])
            minh = np.min(tsDf['D3'].values-DIAMETER)
            ratio = min(1/abs(minh/axM), 5)
#             print(ratio)
            (axmbis, axMbis) = ax1bis.get_ylim()
            ax1bis.set_ylim([0, max(axMbis*ratio, 3*max(tsDf['F'].values))])
        
        
        #### fig3
        # Rescale fig3 axes
        eMin, eMax = 1, 0
        sMin, sMax = 1000, 0
        for i in range(Ncomp):
            colSp = i % nColsSubplot
            rowSp = i // nColsSubplot
            # ax2[i] with the 1 line plot
            if nRowsSubplot == 1:
                thisAx3 = ax3[colSp]
            elif nRowsSubplot >= 1:
                thisAx3 = ax3[rowSp,colSp]
                
            if thisAx3.get_ylim()[1] > eMax:
                eMax = thisAx3.get_ylim()[1]
            if thisAx3.get_xlim()[1] > sMax:
                sMax = thisAx3.get_xlim()[1]
                
        for i in range(Ncomp):
            colSp = i % nColsSubplot
            rowSp = i // nColsSubplot
            # ax2[i] with the 1 line plot
            if nRowsSubplot == 1:
                thisAx3 = ax3[colSp]
            elif nRowsSubplot >= 1:
                thisAx3 = ax3[rowSp,colSp]
                
            sMax = min(sMax, 2100)
            eMax = min(eMax, 0.5)
            
            thisAx3.set_xlim([0, sMax])
            thisAx3.set_ylim([0, eMax])
                
        #### fig4
        NL = len(alreadyLabeled4)
        titleVoid = ' '
        if NL > 16:
            titleVoid += '\n '
        fig4.suptitle(titleVoid)
        fig4.legend(loc='upper center', bbox_to_anchor=(0.5,1), ncol = min(8, NL))
                
        #### fig5
        
        NL = len(alreadyLabeled5)
        titleVoid = ' '
        if NL > 16:
            titleVoid += '\n '
        fig5.suptitle(titleVoid)
        fig5.legend(loc='upper center', bbox_to_anchor=(0.5,1), ncol = min(8, NL))
        
        # Rescale fig5 axes
        eMin, eMax = 1, 0
        sMin, sMax = 1000, 0
        for i in range(Ncomp):
            colSp = i % nColsSubplot
            rowSp = i // nColsSubplot
            # ax2[i-1] with the 1 line plot
            if nRowsSubplot == 1:
                thisAx5 = ax5[colSp]
            elif nRowsSubplot >= 1:
                thisAx5 = ax5[rowSp,colSp]
                
            if thisAx5.get_xlim()[1] > eMax:
                eMax = thisAx5.get_xlim()[1]
            if thisAx5.get_ylim()[1] > sMax:
                sMax = thisAx5.get_ylim()[1]
                
        for i in range(Ncomp):
            colSp = i % nColsSubplot
            rowSp = i // nColsSubplot
            # ax2[i-1] with the 1 line plot
            if nRowsSubplot == 1:
                thisAx5 = ax5[colSp]
            elif nRowsSubplot >= 1:
                thisAx5 = ax5[rowSp,colSp]
            
            sMax = min(sMax, 2100)
            eMax = min(eMax, 0.5)
            thisAx5.set_ylim([0, 1000])
            thisAx5.set_xlim([0, 0.20])
                
        #### fig6
        # Rescale fig6 axes
        
        sMax = 0
        Kmax = 0

        for i in range(Ncomp):
            colSp = i % nColsSubplot
            rowSp = i // nColsSubplot
            if nRowsSubplot == 1:
                thisAx6 = ax6[colSp]
            elif nRowsSubplot >= 1:
                thisAx6 = ax6[rowSp,colSp]

            if thisAx6.get_xlim()[1] > sMax:
                sMax = thisAx6.get_xlim()[1]
            if thisAx6.get_ylim()[1] > Kmax:
                Kmax = thisAx6.get_xlim()[1]
        
        for i in range(Ncomp):
            colSp = i % nColsSubplot
            rowSp = i // nColsSubplot
            if nRowsSubplot == 1:
                thisAx6 = ax6[colSp]
            elif nRowsSubplot >= 1:
                thisAx6 = ax6[rowSp,colSp]
                
            Kmax = min(Kmax, 20)

            thisAx6.set_xlim([0, sMax])
            thisAx6.set_ylim([0, 30])
        
        #### fig8 
        
        NL = len(alreadyLabeled8)
        titleVoid = ' '
        if NL > 16:
            titleVoid += '\n '
        fig8.suptitle(titleVoid)
        fig8.legend(loc='upper center', bbox_to_anchor=(0.5,1), ncol = min(8, NL))
        
        eMin, eMax = 1, 0
        sMin, sMax = 1000, 0
        for i in range(1, Ncomp+1):
            colSp = (i-1) % nColsSubplot
            rowSp = (i-1) // nColsSubplot
            # ax2[i-1] with the 1 line plot
            if nRowsSubplot == 1:
                thisAx8 = ax8[colSp]
            elif nRowsSubplot >= 1:
                thisAx8 = ax8[rowSp,colSp]
            title = thisAx8.title.get_text()
            if not 'NON VALIDATED' in title:
                if thisAx8.get_xlim()[0] < eMin:
                    eMin = thisAx8.get_xlim()[0]
                if thisAx8.get_xlim()[1] > eMax:
                    eMax = thisAx8.get_xlim()[1]
                if thisAx8.get_ylim()[0] < sMin:
                    sMin = thisAx8.get_ylim()[0]
                if thisAx8.get_ylim()[1] > sMax:
                    sMax = thisAx8.get_ylim()[1]
        for i in range(1, Ncomp+1):
            colSp = (i-1) % nColsSubplot
            rowSp = (i-1) // nColsSubplot
            # ax2[i-1] with the 1 line plot
            if nRowsSubplot == 1:
                thisAx8 = ax8[colSp]
            elif nRowsSubplot >= 1:
                thisAx8 = ax8[rowSp,colSp]
            title = thisAx8.title.get_text()
            if not 'NON VALIDATED' in title:
                sMin = max(0, sMin)
                eMin = max(0, eMin)
                eMax = min(0.5, eMax)
                thisAx8.set_ylim([sMin, sMax])
                thisAx8.set_xlim([eMin, eMax])
        
        #### fig9
        
        for i in range(1, Ncomp+1):
            colSp = (i-1) % nColsSubplot
            rowSp = (i-1) // nColsSubplot
            # ax2[i-1] with the 1 line plot
            if nRowsSubplot == 1:
                thisAx9 = ax9[colSp]
            elif nRowsSubplot >= 1:
                thisAx9 = ax9[rowSp,colSp]
            title = thisAx9.title.get_text()
            if not 'NON VALIDATED' in title:
                thisAx9.set_xlim([0,1000])
                # thisAx6.set_ylim([KMin, KMax])
                thisAx9.set_ylim([100, 5e4])
        
        #### fig10 
        
        NL = len(alreadyLabeled10)
        titleVoid = ' '
        if NL > 16:
            titleVoid += '\n '
        fig10.suptitle(titleVoid)
        fig10.legend(loc='upper center', bbox_to_anchor=(0.5,1), ncol = min(8, NL))
        
        eMin, eMax = 1, 0
        sMin, sMax = 1000, 0
        for i in range(1, Ncomp+1):
            colSp = (i-1) % nColsSubplot
            rowSp = (i-1) // nColsSubplot
            # ax2[i-1] with the 1 line plot
            if nRowsSubplot == 1:
                thisAx10 = ax10[colSp]
            elif nRowsSubplot >= 1:
                thisAx10 = ax10[rowSp,colSp]
            title = thisAx10.title.get_text()
            if not 'NON VALIDATED' in title:
                if thisAx10.get_xlim()[0] < eMin:
                    eMin = thisAx10.get_xlim()[0]
                if thisAx10.get_xlim()[1] > eMax:
                    eMax = thisAx10.get_xlim()[1]
                if thisAx10.get_ylim()[0] < sMin:
                    sMin = thisAx10.get_ylim()[0]
                if thisAx10.get_ylim()[1] > sMax:
                    sMax = thisAx10.get_ylim()[1]
                    
        for i in range(1, Ncomp+1):
            colSp = (i-1) % nColsSubplot
            rowSp = (i-1) // nColsSubplot
            # ax2[i-1] with the 1 line plot
            if nRowsSubplot == 1:
                thisAx10 = ax10[colSp]
            elif nRowsSubplot >= 1:
                thisAx10 = ax10[rowSp,colSp]
            title = thisAx10.title.get_text()
            if not 'NON VALIDATED' in title:
                sMin = max(0, sMin)
                eMin = max(0, eMin)
                eMax = min(0.5, eMax)
                thisAx10.set_ylim([sMin, sMax])
                thisAx10.set_xlim([eMin, eMax])
                
        #### fig11
        
        for i in range(1, Ncomp+1):
            colSp = (i-1) % nColsSubplot
            rowSp = (i-1) // nColsSubplot
            # ax2[i-1] with the 1 line plot
            if nRowsSubplot == 1:
                thisAx11 = ax11[colSp]
            elif nRowsSubplot >= 1:
                thisAx11 = ax11[rowSp,colSp]
            title = thisAx11.title.get_text()
            if not 'NON VALIDATED' in title:
                thisAx11.set_xlim([0, NbOfTargets])
                thisAx11.set_ylim([100, 5e4])
                
                
        Allfigs = [fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,fig10,fig11]
        
        for fig in Allfigs:
            fig.tight_layout()

    
    
    #### PLOT [4/4]
    # Save the figures
    if PLOT:
        
        dpi = 150       
        
        figSubDir = 'MecaAnalysis_allCells_' + task
        ufun.archiveFig(fig1, name = currentCellID + '_01_h(t)', figSubDir = figSubDir, dpi = dpi)
        ufun.archiveFig(fig2, name = currentCellID + '_02_F(h)', figSubDir = figSubDir, dpi = dpi)
        ufun.archiveFig(fig3, name = currentCellID + '_03_sig(eps)', figSubDir = figSubDir, dpi = dpi)
        ufun.archiveFig(fig4, name = currentCellID + '_04_F(h)_regionFits', figSubDir = figSubDir, dpi = dpi)
        ufun.archiveFig(fig5, name = currentCellID + '_05_sig(eps)_regionFits', figSubDir = figSubDir, dpi = dpi)
        ufun.archiveFig(fig6, name = currentCellID + '_06_K(s)', figSubDir = figSubDir, dpi = dpi)
        ufun.archiveFig(fig7, name = currentCellID + '_07_smallElements', figSubDir = figSubDir, dpi = dpi)
        ufun.archiveFig(fig8, name = currentCellID + '_08_sig(eps)_weightedRegionFits', figSubDir = figSubDir, dpi = dpi)
        ufun.archiveFig(fig9, name = currentCellID + '_09_K3(s)', figSubDir = figSubDir, dpi = dpi)
        ufun.archiveFig(fig10, name = currentCellID + '_10_sig(eps)_pointBasedFits', figSubDir = figSubDir, dpi = dpi)
        ufun.archiveFig(fig11, name = currentCellID + '_11_K4(s)', figSubDir = figSubDir, dpi = dpi)
        
        
        
        if PLOT_SHOW:
            Allfigs = [fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,fig10,fig11]
            # for fig in Allfigs:
                # fig.tight_layout()
                # fig.show()
            plt.show()
        else:
            plt.close('all')

    return(results)



# def createDataDict_meca(list_mecaFiles, listColumnsMeca, task, PLOT):
#     """
#     Subfunction of computeGlobalTable_meca
#     Create the dictionnary that will be converted in a pandas table in the end.
#     """
#     expDf = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = cp.suffix)
#     tableDict = {}
#     Nfiles = len(list_mecaFiles)
#     PLOT_SHOW = (Nfiles==1)
#     PLOT_SHOW = 0
#     if not PLOT_SHOW:
#         plt.ioff()
#     for c in listColumnsMeca:
#         tableDict[c] = []
#     for f in list_mecaFiles: #[:10]:
#         tS_DataFilePath = os.path.join(cp.DirDataTimeseries, f)
#         current_tsDf = pd.read_csv(tS_DataFilePath, sep = ';')
#          # MAIN SUBFUNCTION
#         current_resultDict = analyseTimeSeries_meca(f, current_tsDf, expDf, 
#                                                     dictColumnsMeca, task,
#                                                     PLOT, PLOT_SHOW)
#         for k in current_resultDict.keys():
#             tableDict[k] += current_resultDict[k]
# #     for k in tableDict.keys():
# #         print(k, len(tableDict[k]))
#     return(tableDict)

def buildDf_meca(list_mecaFiles, dictColumnsMeca, task, PLOT):
    """
    Subfunction of computeGlobalTable_meca
    Create the dictionnary that will be converted in a pandas table in the end.
    """
    expDf = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = cp.suffix)
    list_resultDf = []
    Nfiles = len(list_mecaFiles)
    PLOT_SHOW = 0
    for f in list_mecaFiles: #[:10]:
        tS_DataFilePath = os.path.join(cp.DirDataTimeseries, f)
        current_tsDf = pd.read_csv(tS_DataFilePath, sep = ';')
         # MAIN SUBFUNCTION
        current_resultDict = analyseTimeSeries_meca(f, current_tsDf, expDf, 
                                                    dictColumnsMeca, task,
                                                    PLOT, PLOT_SHOW)
        current_resultDf = pd.DataFrame(current_resultDict)
        list_resultDf.append(current_resultDf)
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



def computeGlobalTable_meca(task = 'fromScratch', fileName = 'Global_MecaData', 
                            save = False, PLOT = False, \
                            source = 'Matlab', dictColumnsMeca=dictColumnsMeca,
                            ui_fileSuffix = 'UserManualSelection_MecaData'):
    """
    Compute the GlobalTable_meca from the time series data files.
    Option task='fromScratch' will analyse all the time series data files and construct a new GlobalTable from them regardless of the existing GlobalTable.
    Option task='updateExisting' will open the existing GlobalTable and determine which of the time series data files are new ones, and will append the existing GlobalTable with the data analysed from those new fils.
    DEPRECATED : listColumnsMeca have to contain all the fields of the table that will be constructed.
    dictColumnsMeca have to contain all the fields of the table that will be constructed AND their default values !
    """
    top = time.time()
    
#     list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
#                       if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
#                       and ('R40' in f))] # Change to allow different formats in the future
    
    suffixPython = '_PY'
    if source == 'Matlab':
        list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                      if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
                      and (('R40' in f) or ('L40' in f)) and not (suffixPython in f))]
        
    elif source == 'Python':
        list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                      if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
                      and (('R40' in f) or ('L40' in f)) and (suffixPython in f))]
        # print(list_mecaFiles)
    
#     print(list_mecaFiles)
    
    if task == 'fromScratch':
        # # create a dict containing the data
        # tableDict = createDataDict_meca(list_mecaFiles, listColumnsMeca, task, PLOT) # MAIN SUBFUNCTION
        
        # # create the dataframe from it
        # mecaDf = pd.DataFrame(tableDict)
        
        mecaDf = buildDf_meca(list_mecaFiles, dictColumnsMeca, task, PLOT) # MAIN SUBFUNCTION
        
        updateUiDf_meca(ui_fileSuffix, mecaDf)
        
        # last step: now that the dataFrame is complete, one can use "compStartTimeThisDay" col to compute the start time of each compression relative to the first one done this day.
        allDates = list(mecaDf['date'].unique())
        for d in allDates:
            subDf = mecaDf.loc[mecaDf['date'] == d]
            experimentStartTime = np.min(subDf['compStartTimeThisDay'])
            mecaDf['compStartTimeThisDay'].loc[mecaDf['date'] == d] = mecaDf['compStartTimeThisDay'] - experimentStartTime
        
    elif task == 'updateExisting':
        # get existing table
        try:
            savePath = os.path.join(cp.DirDataAnalysis, (fileName + '.csv'))
            existing_mecaDf = pd.read_csv(savePath, sep=';')
        except:
            print('No existing table found')
            
        # find which of the time series files are new
        new_list_mecaFiles = []
        for f in list_mecaFiles:
            currentCellID = ufun.findInfosInFileName(f, 'cellID')
            if currentCellID not in existing_mecaDf.cellID.values:
                new_list_mecaFiles.append(f)
                
        # # create the dict with new data
        # new_tableDict = createDataDict_meca(new_list_mecaFiles, listColumnsMeca, task, PLOT) # MAIN SUBFUNCTION
        # # create the dataframe from it
        # new_mecaDf = pd.DataFrame(new_tableDict)
        
        new_mecaDf = buildDf_meca(new_list_mecaFiles, dictColumnsMeca, task, PLOT) # MAIN SUBFUNCTION
        # fuse the existing table with the new one
        mecaDf = pd.concat([existing_mecaDf, new_mecaDf])
        
        updateUiDf_meca(ui_fileSuffix, mecaDf)
        
    else: # If task is neither 'fromScratch' nor 'updateExisting'
    # Then task can be a substring that can be in some timeSeries file !
    # It will create a table with only these files !

        task_list = task.split(' & ')
        new_list_mecaFiles = []
        for f in list_mecaFiles:
            currentCellID = ufun.findInfosInFileName(f, 'cellID')
            for t in task_list:
                if t in currentCellID:
                    new_list_mecaFiles.append(f)
                    break
                
        # # create the dict with new data
        # new_tableDict = createDataDict_meca(new_list_mecaFiles, dictColumnsMeca, task, PLOT) # MAIN SUBFUNCTION
        # # create the dataframe from it
        # mecaDf = pd.DataFrame(new_tableDict)
        
        mecaDf = buildDf_meca(new_list_mecaFiles, dictColumnsMeca, task, PLOT) # MAIN SUBFUNCTION
        updateUiDf_meca(ui_fileSuffix, mecaDf)
    
    for c in mecaDf.columns:
            if 'Unnamed' in c:
                mecaDf = mecaDf.drop([c], axis=1)
    
    if save:
        saveName = fileName + '.csv'
        savePath = os.path.join(cp.DirDataAnalysis, saveName)
        mecaDf.to_csv(savePath, sep=';')
    
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



# %% (5) General import functions
    
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
                   mergeExpDf = True, mergFluo = False, mergeUMS = False):
    
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
    
    print(gs.CYAN + 'Merged table has ' + str(df.shape[0]) + ' lines and ' \
          + str(df.shape[1]) + ' columns.' + gs.NORMAL)
        
    return(df)



def getGlobalTable(kind, DirDataExp = cp.DirRepoExp):
    if kind == 'ctField':
        GlobalTable = getGlobalTable_ctField()
        expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
        fluoDf = getFluoData()
        GlobalTable = pd.merge(expDf, GlobalTable, how="inner", on='manipID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
        print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')
        
        # print(GlobalTable_ctField.head())
    
    elif kind == 'ctField_py':
        GlobalTable = getGlobalTable_ctField('Global_CtFieldData_Py')
        expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
        fluoDf = getFluoData()
        GlobalTable = pd.merge(expDf, GlobalTable, how="inner", on='manipID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
        print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')
        
        # print(GlobalTable_ctField.head())
        
        # return(GlobalTable_ctField)

    elif kind == 'meca_matlab':
        GlobalTable = getGlobalTable_meca('Global_MecaData')
        expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
        fluoDf = getFluoData()
        GlobalTable = pd.merge(GlobalTable, expDf, how="inner", on='manipID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", left_on='CellName', right_on='cellID'
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')
        
        # print(GlobalTable.tail())
        GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
        # return(GlobalTable_meca_Matlab)


    elif kind == 'meca_py':
        GlobalTable = getGlobalTable_meca('Global_MecaData_Py')
        expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
        fluoDf = getFluoData()
        GlobalTable = pd.merge(GlobalTable, expDf, how="inner", on='manipID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", left_on='cellID', right_on='cellID'
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')
        
        # print(GlobalTable_meca_Py.tail())
        GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
        # return(GlobalTable_meca_Py)


    elif kind == 'meca_py2':
        GlobalTable = getGlobalTable_meca('Global_MecaData_Py2')
        expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
        fluoDf = getFluoData()
        GlobalTable = pd.merge(GlobalTable, expDf, how="inner", on='manipID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", left_on='cellID', right_on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')

        # print(GlobalTable_meca_Py2.tail())
        GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
        # return(GlobalTable_meca_Py2)
    
    elif kind == 'meca_nonLin':
        GlobalTable = getGlobalTable_meca('Global_MecaData_NonLin_Py')
        expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
        fluoDf = getFluoData()
        GlobalTable = pd.merge(GlobalTable, expDf, how="inner", on='manipID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", left_on='cellID', right_on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')

        # print(GlobalTable_meca_nonLin.tail())
        GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
        # return(GlobalTable_meca_nonLin)
    
    elif kind == 'meca_MCA':
        GlobalTable = getGlobalTable_meca('Global_MecaData_MCA')
        expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
        fluoDf = getFluoData()
        GlobalTable = pd.merge(GlobalTable, expDf, how="inner", on='manipID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", left_on='cellID', right_on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')

        # print(GlobalTable_meca_nonLin.tail())
        GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
        # return(GlobalTable)
        
    elif kind == 'meca_HoxB8':
        GlobalTable = getGlobalTable_meca('Global_MecaData_HoxB8')
        expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
        fluoDf = getFluoData()
        ui_fileName = 'UserManualSelection_MecaData'
        # uiDf, success = get_uiDf(ui_fileName)
        GlobalTable = pd.merge(GlobalTable, expDf, how="inner", on='manipID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", left_on='cellID', right_on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
        # GlobalTable = pd.merge(GlobalTable, uiDf, 
        #                        how="left", left_on=['cellID', 'compNum'], right_on=['cellID', 'compNum'],
        # #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        # #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        # )
        GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
        print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')

        # return(GlobalTable)
    
    else:
        GlobalTable = getGlobalTable_meca(kind)
        expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
        fluoDf = getFluoData()
        GlobalTable = pd.merge(GlobalTable, expDf, how="inner", on='manipID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", left_on='cellID', right_on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')
    
        # print(GlobalTable_meca_nonLin.tail())
        GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
    
    if 'substrate' in GlobalTable.columns:
        vals_substrate = GlobalTable['substrate'].values
        if 'diverse fibronectin discs' in vals_substrate:
            try:
                cellIDs = GlobalTable[GlobalTable['substrate'] == 'diverse fibronectin discs']['cellID'].values
                listFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                              if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv"))]
                for Id in cellIDs:
                    for f in listFiles:
                        if Id == ufun.findInfosInFileName(f, 'cellID'):
                            thisCellSubstrate = ufun.findInfosInFileName(f, 'substrate')
                            thisCellSubstrate = dictSubstrates[thisCellSubstrate]
                            if not thisCellSubstrate == '':
                                GlobalTable.loc[GlobalTable['cellID'] == Id, 'substrate'] = thisCellSubstrate
                print('Automatic determination of substrate type SUCCEDED !')
                
            except:
                print('Automatic determination of substrate type FAILED !')
    
    return(GlobalTable)
