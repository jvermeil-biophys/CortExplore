# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:24:45 2023

@author: JosephVermeil
"""

# -*- coding: utf-8 -*-
"""
Main_3DTracker.py - Script to call the mainTracker function from SimpleBeadTracker.py.
Joseph Vermeil, 2023

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

# %% Imports

import os
import re
import sys
import time

import numpy as np
import pandas as pd

from BeadTracker_Fluo import smallTracker



# %% Utility functions


def makeMetaData(T_raw, B_set, loopStructure):
    # Columns from the field file
    metaDf = pd.DataFrame({'T_raw':T_raw, 'B_set':B_set})
    
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
            

        
        
        
        





# %% General template

# %%% Define paths

cell = 'C8'

dictPaths = {'sourceDirPath' : 'D://MagneticPincherData//Raw//24.02.27_Chameleon//Clean_Fluo_BF',
             'imageFileName' : f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_BF.tif',
             'resultsFileName' : f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_BF_Results.txt',
             'depthoDir':'D://MagneticPincherData//Raw//DepthoLibrary',
             'depthoName':'24.02.27_Chameleon_M450_step20_100X',
             'resultDirPath' : 'D://MagneticPincherData//Raw//24.02.27_Chameleon//Clean_Fluo_BF//Results',
             }


# %%% Define constants

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


# %%% Additionnal options

dictOptions = {'redoAllSteps' : True,
               'trackAll' : False,
               'importLogFile' : True,
               'saveLogFile' : True,
               'saveFluo' : False,
               'importTrajFile' : True,
               'expandedResults' : True,
              }

# %%% Make metaDataFrame # NEED TO CODE STH HERE !

OMEfilepath = os.path.join(dictPaths['sourceDirPath'], f'3T3-LifeActGFP_PincherFluo_{cell}_off4um_OME.txt')

CTZ_tz = OMEDataParser(OMEfilepath)
BF_T = CTZ_tz[0,:,:,0].flatten()
NFrames = len(BF_T)
B_set = 5 * np.ones(NFrames)

loopStructure = [{'Status':'Passive',     'NFrames':NFrames}]  # Only one constant field phase
                
metaDf = makeMetaData(BF_T, B_set, loopStructure)

# %%% Call mainTracker()

smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)












# %% Test with my data

# %%% Define paths

dictPaths = {'sourceDirPath' : 'D://MagneticPincherData//Raw//24.03.13',
             'imageFileName' : '24-03-13_M3_P1_C1_L50_disc20um_12ms.tif',
             'resultsFileName' : '24-03-13_M3_P1_C1_L50_disc20um_12ms_Results.txt',
             'depthoDir':'D://MagneticPincherData//Raw//DepthoLibrary',
             'depthoName':'24.03.13_M3_M450_step20_100X', # '23.11.01_In_M450_step20_100X',
             'resultDirPath' : 'D://MagneticPincherData//Data_Timeseries',
             }


# %%% Define constants

dictConstants = {'microscope' : 'labview',
                 # If 1 type of beads
                  # 'bead type' : 'M450', # 'M450' or 'M270'
                  # 'bead diameter' : 4493, # nm
                  # 'bead magnetization correction' : 0.969, # nm
                 # If 2 types of beads
                 'inside bead type' : 'M450-2025', # 'M450' or 'M270'
                 'inside bead diameter' : 4493, # nm
                 'inside bead magnetization correction' : 0.969, # nm
                 'outside bead type' : 'M450-Strept', # 'M450' or 'M270'
                 'outside bead diameter' : 4506, # nm
                 'outside bead magnetization correction' : 1.056, # nm
                 #
                 'normal field multi images' : 3, # Number of images
                 'multi image Z step' : 500, # nm
                 'multi image Z direction' : 'upward', # Either 'upward' or 'downward'
                 #
                 'scale pixel per um' : 15.8, # pixel/µm
                 'optical index correction' : 0.875, # ratio, without unit
                 'beads bright spot delta' : 0, # Rarely useful, do not change
                 'magnetic field correction' : 1.0, # ratio, without unit
                 }


# %%% Additionnal options

dictOptions = {'redoAllSteps' : False,
               'trackAll' : False,
               'importLogFile' : True,
               'saveLogFile' : True,
               'saveFluo' : False,
               'importTrajFile' : True,
              }

# %%% Make metaDataFrame
# metaDf # ['T_raw', 'B_set', 'iL', 'Status']

fieldPath = dictPaths['sourceDirPath'] + '//' + dictPaths['imageFileName'][:-4] + '_Field.txt'
statusPath = dictPaths['sourceDirPath'] + '//' + dictPaths['imageFileName'][:-4] + '_Status.txt'
metaDf = makeMetaData_fieldAndStatus(fieldPath, statusPath)

# %%% Make metaDataFrame V2
# metaDf # ['T_raw', 'B_set', 'iL', 'Status']

fieldPath = dictPaths['sourceDirPath'] + '//' + dictPaths['imageFileName'][:-4] + '_Field.txt'

loopStructure = [
                 {'Status':'Passive',     'NFrames':30},  # First constant field phase
                 {'Status':'Action',      'NFrames':40},  # Pre-compression: sigmoïd down + constant field
                 {'Status':'Action_main', 'NFrames':125}, # Compression
                 {'Status':'Action',      'NFrames':15},  # Relaxation
                 {'Status':'Passive',     'NFrames':30},  # Second constant field phase
                ]

metaDf = makeMetaData(fieldPath, loopStructure)


# %%% Call mainTracker()

tsDf = smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)


# %%% Plot stuff

iL = 1
tsDf_i = tsDf[(tsDf['idxLoop'] == iL)] #  & (tsDf['idxAnalysis'] == iL)
tsDf.plot('T', 'D3')


# %% Test for Anumita's files

# %%% Define paths

dictPaths = {'sourceDirPath' : 'C://Users//JosephVermeil//Desktop//SpinningDisc_ExampleData//Source',
             'imageFileName' : '23-09-21_M1_P1_C1_disc20um_L40_1.tif',
             'resultsFileName' : '23-09-21_M1_P1_C1_disc20um_L40_1_Results.txt',
             'depthoDir':'C://Users//JosephVermeil//Desktop//SpinningDisc_ExampleData//Deptho',
             'depthoName':'23.09.19_M2_M450_step20_100X',
             'resultDirPath' : 'C://Users//JosephVermeil//Desktop//SpinningDisc_ExampleData//Output',
             }


# %%% Define constants

dictConstants = {'microscope' : 'ZEN',
                 #
                 'bead type' : 'M450', # 'M450' or 'M270'
                 'bead diameter' : 4500, # nm
                 #
                 'normal field multi images' : 3, # Number of images
                 'multi image Z step' : 500, # nm
                 'multi image Z direction' : 'upward', # Either 'upward' or 'downward'
                 #
                 'scale pixel per um' : 15.8, # pixel/µm
                 'optical index correction' : 0.875, # ratio, without unit
                 'beads bright spot delta' : 0, # Rarely useful, do not change
                 'magnetic field correction' : 1.0, # ratio, without unit
                 }


# %%% Additionnal options

dictOptions = {'redoAllSteps' : False,
               'trackAll' : False,
               'importLogFile' : True,
               'saveLogFile' : True,
               'saveFluo' : False,
               'importTrajFile' : True,
              }

# %%% Make metaDataFrame

# metaDf # ['T_raw', 'B_set', 'iL', 'Status']

i_start, i_stop = 653, 803

sourceDirPath = dictPaths['sourceDirPath']
print(os.listdir(sourceDirPath))

f_root = dictPaths['imageFileName'][:-4]

resAcquisPath = os.path.join(sourceDirPath, f_root + ' _ResAcquis.txt')
resAcquisDf = pd.read_csv(resAcquisPath, sep=None, engine='python', names=['B_set', 'B', 'T_raw'])

metaDf = resAcquisDf[['T_raw', 'B_set']]
metaDf['iL'] = 1
metaDf['Status'] = 'Action'
# print(metaDf['Status'].values)

metaDf.loc[i_start:i_stop, 'Status'] = 'Action_main'



# %%% Call mainTracker()

smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)



