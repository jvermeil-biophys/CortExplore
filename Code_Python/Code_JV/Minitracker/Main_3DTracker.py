# -*- coding: utf-8 -*-
"""
Main_3DTracker.py - Script to call the mainTracker function from BeadTracker.py.
Joseph Vermeil, 2024


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
import numpy as np
import pandas as pd

from BeadTracker_V4 import smallTracker

# %% Utility functions

def dateFormat(d):
    d2 = d[0:2]+'.'+d[3:5]+'.'+d[6:]
    return(d2)


def makeMetaData_type1(T_raw, B_set, loopStructure):
    """
    Parameters
    ----------
    T_raw : array-like
        The list of all time points corresponding to the images. len = N_Frames_total.
    B_set : array-like
        The list of all magnetic field values corresponding to the images. len = N_Frames_total.
    loopStructure : list of dicts
        Describe the phases composing each loop: 'Status' is the type of phase, 'NFrames' is the number of images it contains.
        The sum of all 'NFrames' must be equal to N_Frames_perLoop = N_Frames_total//N_Loops.
        Example of a loopStructure: 
        loopStructure = [
                         {'Status':'Passive',     'NFrames':30},  # First constant field phase
                         {'Status':'Action',      'NFrames':40},  # Pre-compression: sigmoïd down + constant field
                         {'Status':'Action_main', 'NFrames':125}, # Compression
                         {'Status':'Action',      'NFrames':15},  # Relaxation
                         {'Status':'Passive',     'NFrames':30},  # Second constant field phase
                        ]

    Returns
    -------
    metaDf: pandas DataFrame
        Table with 4 columns: 'T_raw', 'B_set', 'iL', 'Status'. Each row correspond to one frame. 
        Its length is N_Frames_total = N_Loops * N_Frames_perLoop
        - T_raw is the list of all time points.
        - B_set is the list of all magnetic field values.
        - iL stands for 'index loop'. It is an integer that gets incremented by 1 for each new loop.
        - Status is the list of the status of each image, as given by the loopStructure list.

    """
    
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


def makeMetaData_type2(fieldPath, loopStructure):
    """
    Parameters
    ----------
    fieldPath : string
        Path toward the '_Field.txt' file generated with the Labview software.
        This file is a table with  4 columns: ['B_meas', 'T_raw', 'B_set', 'Z_piezo']; and one row for each frame.
        From these, 'T_raw' & 'B_set' are read and used as columns for metaDf
        
    loopStructure : list of dicts
        Describe the phases composing each loop: 'Status' is the type of phase, 'NFrames' is the number of images it contains.
        The sum of all 'NFrames' must be equal to N_Frames_perLoop = N_Frames_total//N_Loops.
        Example of a loopStructure: 
        loopStructure = [
                         {'Status':'Passive',     'NFrames':30},  # First constant field phase
                         {'Status':'Action',      'NFrames':40},  # Pre-compression: sigmoïd down + constant field
                         {'Status':'Action_main', 'NFrames':125}, # Compression
                         {'Status':'Action',      'NFrames':15},  # Relaxation
                         {'Status':'Passive',     'NFrames':30},  # Second constant field phase
                        ]

    Returns
    -------
    metaDf: pandas DataFrame
        Table with 4 columns: 'T_raw', 'B_set', 'iL', 'Status'. Each row correspond to one frame. 
        Its length is N_Frames_total = N_Loops * N_Frames_perLoop
        - T_raw is the list of all time points.
        - B_set is the list of all magnetic field values.
        - iL stands for 'index loop'. It is an integer that gets incremented by 1 for each new loop.
        - Status is the list of the status of each image, as given by the loopStructure list.

    """
    
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
        
        metaDf['iL'] = iLCol
        metaDf['Status'] = StatusCol
    
    return(metaDf)



def makeMetaData_type3(fieldPath, statusPath):
    """
    Parameters
    ----------
    fieldPath : string
        Path toward the '_Field.txt' file generated with the Labview software.
        This file is a table with  4 columns: ['B_meas', 'T_raw', 'B_set', 'Z_piezo']; and one row for each frame.
        From these, 'T_raw' & 'B_set' are read and used as columns for metaDf
        
    statusPath : string
        Path toward the '_Status.txt' file generated with the Labview software.
        This file has 1 column containing infos about each frame. The infos are separated with a '_'.
        The format is usually the following: 'iL_PhaseType_ExtraInfos'
        From these, iL the loop index and the phase type related infos are read to build the columns 'Status' & 'iL' in metaDf

    Returns
    -------
    metaDf: pandas DataFrame
        Table with 4 columns: 'T_raw', 'B_set', 'iL', 'Status'. Each row correspond to one frame. 
        Its length is N_Frames_total = N_Loops * N_Frames_perLoop
        - T_raw is the list of all time points.
        - B_set is the list of all magnetic field values.
        - iL stands for 'index loop'. It is an integer that gets incremented by 1 for each new loop.
        - Status is the list of the status of each image, as given by the loopStructure dict.

    """
    
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
    metaDf['iL'] = statusDf['iL']
    metaDf['Status'] = statusDf['Status']
    metaDf.loc[statusDf['Action type'] == mainActionStep, 'Status'] = 'Action_main'
    return(metaDf)


def makeMetaData_Chameleon(timePath, resAcquisPath, loopStructure):
    """
    Parameters
    ----------
    timePath : string
        
        
    resAcquisPath : string
        Path toward the '_Field.txt' file generated with the Labview software.
        This file is a table with  4 columns: ['B_meas', 'T_raw', 'B_set', 'Z_piezo']; and one row for each frame.
        From these, 'T_raw' & 'B_set' are read and used as columns for metaDf
        
    loopStructure : list of dicts
        Describe the phases composing each loop: 'Status' is the type of phase, 'NFrames' is the number of images it contains.
        The sum of all 'NFrames' must be equal to N_Frames_perLoop = N_Frames_total//N_Loops.
        Example of a loopStructure: 
        loopStructure = [
                         {'Status':'Passive',     'NFrames':30},  # First constant field phase
                         {'Status':'Action',      'NFrames':40},  # Pre-compression: sigmoïd down + constant field
                         {'Status':'Action_main', 'NFrames':125}, # Compression
                         {'Status':'Action',      'NFrames':15},  # Relaxation
                         {'Status':'Passive',     'NFrames':30},  # Second constant field phase
                        ]

    Returns
    -------
    metaDf: pandas DataFrame
        Table with 4 columns: 'T_raw', 'B_set', 'iL', 'Status'. Each row correspond to one frame. 
        Its length is N_Frames_total = N_Loops * N_Frames_perLoop
        - T_raw is the list of all time points.
        - B_set is the list of all magnetic field values.
        - iL stands for 'index loop'. It is an integer that gets incremented by 1 for each new loop.
        - Status is the list of the status of each image, as given by the loopStructure list.

    """
    # Columns from the time file
    timeDf = pd.read_csv(timePath, sep='\t', names=['T_corrected'])    
    # Columns from the field file
    fieldDf = pd.read_csv(resAcquisPath, sep='\t', names=['B_set', 'B_meas', 'T_raw'])
    
    metaDf = fieldDf[['T_raw', 'B_set']]
    metaDf['T_raw'] = timeDf['T_corrected'] * 1000
    
    # Columns from the loopStructure
    N_Frames_total = metaDf.shape[0]
    N_Frames_perLoop = np.sum([int(phase['NFrames']) for phase in loopStructure])
    consistency_test = ((N_Frames_total/N_Frames_perLoop) == (N_Frames_total//N_Frames_perLoop))
    
    if not consistency_test:
        print(os.path.split(resAcquisPath)[-1])
        print('Error in the loop structure: the length of described loop do not match the Field file data!')
    
    else:
        N_Loops = (N_Frames_total//N_Frames_perLoop)
        StatusCol = []
        for phase in loopStructure:
            Status = phase['Status']
            NFrames = int(phase['NFrames'])
            L = [Status] * NFrames # Repeat
            StatusCol += L # Append
        StatusCol *= N_Loops # Repeat
        StatusCol = np.array(StatusCol)
        
        iLCol = (np.ones((N_Frames_perLoop, N_Loops)) * np.arange(1, N_Loops+1)).flatten(order='F')
        
        metaDf['iL'] = iLCol
        metaDf['Status'] = StatusCol
    
    return(metaDf)

# %% Valentin's dictys

# %%%% First batch : Dictys WT DB - 20.09.23, 20.09.25, 20.10.06, 20.10.08

dates = ['20.10.08'] # '20.09.23',  '20.09.25', '20.10.06', 

# %%% Set dicts

# Define paths

dictPaths = {'sourceDirPath_root' : 'D:/MagneticPincherData/Raw_Dictys/',
             'sourceDirPath' : '',
             'imageFileName' : '',
             'resultsFileName' : '',
             'depthoDir':'D:/MagneticPincherData/Raw_Dictys/DepthoLibrary',
             'depthoName':'20.05.19_M450_step20_100X_Deptho.tif',
             'resultDirPath' : 'D:/MagneticPincherData/Data_Timeseries',
             }


# Define constants

dictConstants = {'microscope' : 'labview',
                 # If 1 type of beads
                    'bead type' : 'M450', # 'M450' or 'M270'
                    'bead diameter' : 4453, # nm
                    'bead magnetization correction' : 1.05, # nm
                 # If 2 types of beads
                  # 'inside bead type' : 'M450-2025', # 'M450' or 'M270'
                  # 'inside bead diameter' : 4493, # nm
                  # 'inside bead magnetization correction' : 0.969, # nm
                  # 'outside bead type' : 'M450-Strept', # 'M450' or 'M270'
                  # 'outside bead diameter' : 4506, # nm
                  # 'outside bead magnetization correction' : 1.056, # nm
                 #
                 'normal field multi images' : 3, # Number of images
                 'multi image Z step' : 500, # nm
                 'multi image Z direction' : 'upward', # Either 'upward' or 'downward'
                 #
                 'scale pixel per um' : 15.8, # pixel/µm
                 'optical index correction' : 0.85, # ratio, without unit
                 'magnetic field correction' : 1.0 #30/27.5, # ratio, without unit
                 }

loopStructure = [
                 {'Status':'Passive',     'NFrames':36},  # First constant field phase
                 {'Status':'Action_main', 'NFrames':95}, # Compression
                 {'Status':'Passive',     'NFrames':36},  # Second constant field phase
                ]


# Additionnal options

dictOptions = {'redoAllSteps' : False,
               'saveFluo' : False,
               'plotZ' : False,
               'plotZ_range' : [25, 45],
               }


# %%% Run the function

for date in dates:
    dictPaths['sourceDirPath'] = dictPaths['sourceDirPath_root'] + '/' + date
    listFiles = os.listdir(dictPaths['sourceDirPath'])
    listDeptho = os.listdir(dictPaths['depthoDir'])
    for f in listFiles:
        if f.endswith('.tif') and ('R40' in f):
            dictPaths['imageFileName'] = f
            
            f_root = '.'.join(f.split('.')[:-1])
            fieldPath = os.path.join(dictPaths['sourceDirPath'], f_root + '_Field.txt')
            metaDf = makeMetaData_type2(fieldPath, loopStructure)

            
            # 
            dictPaths['resultsFileName'] = f_root + '_Results.txt'
            smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)
    
    
    # smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)



# %%% Test

resultsPath = os.path.join(dictPaths['sourceDirPath'], dictPaths['resultsFileName'])
resultsDf = pd.read_csv(resultsPath, usecols=['Area', 'StdDev', 'XM', 'YM', 'Slice'], sep=None, engine='python')   









# %% Chameleon Compressions

# %%% Set dicts

# Define paths

dictPaths = {'sourceDirPath' : 'D:/MagneticPincherData/Raw/24.06.14_Chameleon/Compressions',
             'imageFileName' : '',
             'resultsFileName' : '',
             'depthoDir':'D:/MagneticPincherData/Raw/DepthoLibrary',
             'depthoName':'',
             'resultDirPath' : '',
             }


# Define constants

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
                 'normal field multi images' : 1, # Number of images
                 'multi image Z step' : 0, # nm
                 'multi image Z direction' : 'downward', # Either 'upward' or 'downward'
                 #
                 'scale pixel per um' : 15.8, # pixel/µm
                 'optical index correction' : 0.85, # ratio, without unit
                 'magnetic field correction' : 1.0 #30/27.5, # ratio, without unit
                 }

loopStructure = [
                 {'Status':'Passive',     'NFrames':1.0 * 100},  # First constant field phase
                 {'Status':'Action',      'NFrames':3.0 * 100},  # Pre-compression: sigmoïd down + constant field
                 {'Status':'Action_main', 'NFrames':1.5 * 100}, # Compression
                 {'Status':'Action',      'NFrames':1.5 * 100},  # Relaxation
                 {'Status':'Passive',     'NFrames':1.0 * 100},  # Second constant field phase
                ]


# Additionnal options

dictOptions = {'redoAllSteps' : True,
               'saveFluo' : False,
               'plotZ' : False,
               'plotZ_range' : [445, 455],
               }


# %%% Run the function

listFiles = os.listdir(dictPaths['sourceDirPath'])
listDeptho = os.listdir(dictPaths['depthoDir'])
count = 0
for f in listFiles:
    if f.endswith('.tif'): #  and count < 4:
        dictPaths['imageFileName'] = f
        if f.startswith('24-06-14_M2'):
            dictConstants['magnetic field correction'] = 30/27.5
        else:
            dictConstants['magnetic field correction'] = 1.0
        
        f_root = '.'.join(f.split('.')[:-1])
        timePath = os.path.join(dictPaths['sourceDirPath'], f_root + 'time.txt')
        resAcquisPath = os.path.join(dictPaths['sourceDirPath'], f_root + ' _ResAcquis.txt')
        metaDf = makeMetaData_Chameleon(timePath, resAcquisPath, loopStructure)
        
        split_f = f.split('_')
        DepthoPrefix = split_f[0] + '_Chameleon-Orca_' + split_f[1]
        DepthoPrefix = dateFormat(DepthoPrefix)
        depthoName = ''
        for fd in listDeptho:
            if fd.startswith(DepthoPrefix) and fd.endswith('.tif'):
                depthoName = fd # '_'.join(fd.split('_')[:-1])
                break
        dictPaths['depthoName'] = depthoName
        
        # 
        dictPaths['resultsFileName'] = f_root + '_Results.txt'
        dictPaths['resultDirPath'] = os.path.join(dictPaths['sourceDirPath'], 'Timeseries')
        smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)
        count += 1


# smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)





# %% In-Out Analysis

# %%% Set dicts

# Define paths

dictPaths = {'sourceDirPath' : 'D:/MagneticPincherData/Raw/Control_InIn_OutOut',
             'imageFileName' : '',
             'resultsFileName' : '',
             'depthoDir':'D:/MagneticPincherData/Raw/DepthoLibrary',
             'depthoName':'',
             'resultDirPath' : '',
             }


# Define constants

dictConstants = {'microscope' : 'labview',
                 # If 1 type of beads
                   'bead type' : 'M450', # 'M450' or 'M270'
                   'bead diameter' : 4493, # nm
                   'bead magnetization correction' : 0.969, # nm
                 # If 2 types of beads
                 # 'inside bead type' : 'M450-2025', # 'M450' or 'M270'
                 # 'inside bead diameter' : 4493, # nm
                 # 'inside bead magnetization correction' : 0.969, # nm
                 # 'outside bead type' : 'M450-Strept', # 'M450' or 'M270'
                 # 'outside bead diameter' : 4506, # nm
                 # 'outside bead magnetization correction' : 1.056, # nm
                 #
                 'normal field multi images' : 3, # Number of images
                 'multi image Z step' : 500, # nm
                 'multi image Z direction' : 'upward', # Either 'upward' or 'downward'
                 #
                 'scale pixel per um' : 15.8, # pixel/µm
                 'optical index correction' : 0.85, # ratio, without unit
                 'magnetic field correction' : 1.0, # ratio, without unit
                 }


# Additionnal options

dictOptions = {'redoAllSteps' : False,
               'saveFluo' : True,
               'plotZ' : False,
               'plotZ_range' : [0, 0],
               }

# %%% Run the function

listFiles = os.listdir(dictPaths['sourceDirPath'])
listDeptho = os.listdir(dictPaths['depthoDir'])
for f in listFiles:
    if f.endswith('.tif'):
        dictPaths['imageFileName'] = f
        
        f_root = '.'.join(f.split('.')[:-1])
        fieldPath = os.path.join(dictPaths['sourceDirPath'], f_root + '_Field.txt')
        statusPath = os.path.join(dictPaths['sourceDirPath'], f_root + '_Status.txt')
        metaDf = makeMetaData_type3(fieldPath, statusPath)
        
        DepthoPrefix = '_'.join(f.split('_')[:2])
        DepthoPrefix = dateFormat(DepthoPrefix)
        depthoName = ''
        for fd in listDeptho:
            if fd.startswith(DepthoPrefix) and fd.endswith('.tif'):
                depthoName = fd # '_'.join(fd.split('_')[:-1])
                break
        dictPaths['depthoName'] = depthoName
        
        # In
        dictPaths['resultsFileName'] = f_root + '_IN_Results.txt'
        dictPaths['resultDirPath'] = os.path.join(dictPaths['sourceDirPath'], 'Timeseries_IN')
        dictConstants['bead type'] = 'M450-2025'
        dictConstants['bead diameter'] = 4493
        dictConstants['bead magnetization correction'] = 0.969
        smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)
        
        # Out
        dictPaths['resultsFileName'] = f_root + '_OUT_Results.txt'
        dictPaths['resultDirPath'] = os.path.join(dictPaths['sourceDirPath'], 'Timeseries_OUT')
        dictConstants['bead type'] = 'M450-Strept'
        dictConstants['bead diameter'] = 4506
        dictConstants['bead magnetization correction'] = 1.056
        smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)

# smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)

# %% In-only Analysis

# %%% Set dicts

# Define paths

dictPaths = {'sourceDirPath' : 'D:/MagneticPincherData/Raw/Control_InIn',
             'imageFileName' : '',
             'resultsFileName' : '',
             'depthoDir':'D:/MagneticPincherData/Raw/DepthoLibrary',
             'depthoName':'',
             'resultDirPath' : '',
             }


# Define constants

dictConstants = {'microscope' : 'labview',
                 # If 1 type of beads
                   'bead type' : 'M450', # 'M450' or 'M270'
                   'bead diameter' : 4493, # nm
                   'bead magnetization correction' : 0.969, # nm
                 # If 2 types of beads
                 # 'inside bead type' : 'M450-2025', # 'M450' or 'M270'
                 # 'inside bead diameter' : 4493, # nm
                 # 'inside bead magnetization correction' : 0.969, # nm
                 # 'outside bead type' : 'M450-Strept', # 'M450' or 'M270'
                 # 'outside bead diameter' : 4506, # nm
                 # 'outside bead magnetization correction' : 1.056, # nm
                 #
                 'normal field multi images' : 3, # Number of images
                 'multi image Z step' : 500, # nm
                 'multi image Z direction' : 'upward', # Either 'upward' or 'downward'
                 #
                 'scale pixel per um' : 15.8, # pixel/µm
                 'optical index correction' : 0.85, # ratio, without unit
                 'magnetic field correction' : 1.0, # ratio, without unit
                 }


# Additionnal options

dictOptions = {'redoAllSteps' : False,
               'saveFluo' : True,
               'plotZ' : False,
               'plotZ_range' : [0, 0],
               }

# %%% Run the function

listFiles = os.listdir(dictPaths['sourceDirPath'])
listDeptho = os.listdir(dictPaths['depthoDir'])
for f in listFiles:
    if f.endswith('.tif'):
        dictPaths['imageFileName'] = f
        
        f_root = '.'.join(f.split('.')[:-1])
        fieldPath = os.path.join(dictPaths['sourceDirPath'], f_root + '_Field.txt')
        statusPath = os.path.join(dictPaths['sourceDirPath'], f_root + '_Status.txt')
        metaDf = makeMetaData_type3(fieldPath, statusPath)
        
        DepthoPrefix = '_'.join(f.split('_')[:2])
        DepthoPrefix = dateFormat(DepthoPrefix)
        depthoName = ''
        for fd in listDeptho:
            if fd.startswith(DepthoPrefix) and fd.endswith('.tif'):
                depthoName = fd # '_'.join(fd.split('_')[:-1])
                break
        dictPaths['depthoName'] = depthoName
        
        # In
        dictPaths['resultsFileName'] = f_root + '_Results.txt'
        dictPaths['resultDirPath'] = os.path.join(dictPaths['sourceDirPath'], 'Timeseries_IN')
        smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)

# smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)

# %% Reanalyze Macrophage

# %%% Set dicts

# Define paths

dictPaths = {'sourceDirPath' : 'C:/Users/JosephVermeil/Desktop/Manuscrit/Figures/BeadInternalization/Tracker',
             'imageFileName' : '22-05-04_M2_P1_C5_disc20um_L40.tif',
             'resultsFileName' : '22-05-04_M2_P1_C5_disc20um_L40_Results.txt',
             'depthoDir':'D:/MagneticPincherData/Raw/DepthoLibrary',
             'depthoName':'',
             'resultDirPath' : 'C:/Users/JosephVermeil/Desktop/Manuscrit/Figures/BeadInternalization/Tracker/Timeseries',
             }


# Define constants

dictConstants = {'microscope' : 'labview',
                 # If 1 type of beads
                   'bead type' : 'M450', # 'M450' or 'M270'
                   'bead diameter' : 4477, # nm
                   'bead magnetization correction' : 1.023, # nm
                 # If 2 types of beads
                 # 'inside bead type' : 'M450-2025', # 'M450' or 'M270'
                 # 'inside bead diameter' : 4493, # nm
                 # 'inside bead magnetization correction' : 0.969, # nm
                 # 'outside bead type' : 'M450-Strept', # 'M450' or 'M270'
                 # 'outside bead diameter' : 4506, # nm
                 # 'outside bead magnetization correction' : 1.056, # nm
                 #
                 'normal field multi images' : 3, # Number of images
                 'multi image Z step' : 500, # nm
                 'multi image Z direction' : 'upward', # Either 'upward' or 'downward'
                 #
                 'scale pixel per um' : 15.8, # pixel/µm
                 'optical index correction' : 0.85, # ratio, without unit
                 'magnetic field correction' : 0.954, # ratio, without unit
                 }

loopStructure = [
                 {'Status':'Passive',     'NFrames':18},  # First constant field phase
                 {'Status':'Action',      'NFrames':200}, # Pre-compression: sigmoïd down + constant field
                 {'Status':'Action_main', 'NFrames':100}, # Compression
                 {'Status':'Action',      'NFrames':100},  # Relaxation
                 {'Status':'Passive',     'NFrames':18},  # Second constant field phase
                 {'Status':'Fluo',     'NFrames':1},
                ]



# Additionnal options

dictOptions = {'redoAllSteps' : False,
               'saveFluo' : True,
               'plotZ' : False,
               'plotZ_range' : [0, 0],
               }

# %%% Run the function

listDeptho = os.listdir(dictPaths['depthoDir'])
f = dictPaths['imageFileName']

f_root = '.'.join(f.split('.')[:-1])
fieldPath = os.path.join(dictPaths['sourceDirPath'], f_root + '_Field.txt')
# statusPath = os.path.join(dictPaths['sourceDirPath'], f_root + '_Status.txt')
metaDf = makeMetaData_type2(fieldPath, loopStructure)

DepthoPrefix = '_'.join(f.split('_')[:2])
DepthoPrefix = dateFormat(DepthoPrefix)
depthoName = ''
for fd in listDeptho:
    if fd.startswith(DepthoPrefix) and fd.endswith('.tif'):
        depthoName = fd # '_'.join(fd.split('_')[:-1])
        break
dictPaths['depthoName'] = depthoName

# In
smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)

# smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)


