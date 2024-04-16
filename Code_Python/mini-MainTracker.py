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

# from BeadTracker_V3 import smallTracker
from BeadTracker_V4 import smallTracker

# %% General template

# %%% Define paths

dictPaths = {'sourceDirPath' : '',
             'imageFileName' : '',
             'resultsFileName' : '',
             'depthoDir':'',
             'depthoName':'',
             'resultDirPath' : '',
             }


# %%% Define constants

dictConstants = {'microscope' : 'labview',
                 #
                 'bead type' : 'M450', # 'M450' or 'M270'
                 'bead diameter' : 4500, # nm
                 #
                 'normal field multi images' : 3, # Number of images
                 'multi image Z step' : 500, # nm
                 'multi image Z direction' : 'upward', # Either 'upward' or 'downward'
                 #
                 'scale pixel per um' : 15.8, # pixel/µm
                 'optical index correction' : 0.85, # ratio, without unit
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

# %%% Make metaDataFrame # NEED TO CODE STH HERE !

# metaDf # ['T_raw', 'B_set', 'iL', 'Status']
metaDf = pd.DataFrame({})

# %%% Call mainTracker()

smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)






# %% Test with my data

# %%% Define paths

dictPaths = {'sourceDirPath' : 'D://MagneticPincherData//Raw//23.11.01_NanoIndent_PincherData//Pincher',
             'imageFileName' : '23-11-01_M1_P1_C2-2_L50_disc20um_nanoIndent.tif',
             'resultsFileName' : '23-11-01_M1_P1_C2-2_L50_disc20um_nanoIndent_Results.txt',
             'depthoDir':'D://MagneticPincherData//Raw//DepthoLibrary',
             'depthoName':'23.09.19_M1_M450_step20_100X', # '23.11.01_In_M450_step20_100X',
             'resultDirPath' : 'D://MagneticPincherData//Data_Timeseries',
             }


# %%% Define constants

dictConstants = {'microscope' : 'labview',
                 #
                  # 'bead type' : 'M450', # 'M450' or 'M270'
                  # 'bead diameter' : 4493, # nm
                  # 'bead magnetization correction' : 0.969, # nm
                 #
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
                 'optical index correction' : 0.85, # ratio, without unit
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

# %%% Make metaDataFrame # NEED TO CODE STH HERE !

# metaDf # ['T_raw', 'B_set', 'iL', 'Status']

fieldPath = dictPaths['sourceDirPath'] + '//' + dictPaths['imageFileName'][:-4] + '_Field.txt'
statusPath = dictPaths['sourceDirPath'] + '//' + dictPaths['imageFileName'][:-4] + '_Status.txt'

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
                 'optical index correction' : 0.85, # ratio, without unit
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



