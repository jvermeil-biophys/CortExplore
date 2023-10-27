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

from BeadTracker_V3 import smallTracker

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



