# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:50:16 2021
@authors: Joseph Vermeil, Anumita Jawahar

BeadTracker.py - contains the classes to perform bead tracking in a movie
(see the function mainTracker and the Tracker classes), and to make a Depthograph
(see the function depthoMaker and the Depthograph classes).
Joseph Vermeil, Anumita Jawahar, 2021

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
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

import os
import re
import time
import pyautogui
import matplotlib
import traceback

from scipy import interpolate
from scipy import signal

from skimage import io, filters, exposure, measure, transform, util, color
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import linear_sum_assignment
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from datetime import date

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)

import GraphicStyles as gs
import GlobalConstants as gc
import UtilityFunctions as ufun


# 2. Pandas settings
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)

# 3. Plot settings
gs.set_default_options_jv()



# %% (1) Utility functions

# NB: Please use this part of the code only for development purposes.
# Once a utility function have been successfully tried & tested, 
# please copy it in the "UtilityFunctions.py" file (imported as ufun, cause it's fun).


# %% (2) Tracker classes


# %%%% PincherTimeLapse

class PincherTimeLapse:
    """
    This class is initialised for each new .tif file analysed.

    It requires the following inputs :
    > I : the timelapse analysed.
    > cellID : the id of the cell currently analysed.
    > manipDict : the line of the experimental data table that concerns the current experiment.
    > NB : the number of beads of interest that will be tracked.

    It contains:
    * data about the 3D image I (dimensions = time, height, width),
    * a list of Frame objects listFrames, 1 per frame in the timelapse,
    * a list of Trajectory objects listTrajectories, 1 per bead of interest (Boi) in the timelapse,
    * a dictionnary dictLog, saving the idx_inNUp of each frame (see below)
                             and all of the user inputs (points and clicks) during the tracking,
    * a pandas DataFrame resultsDf, that contains the raw output of the bead tracking,
    * metadata about the experiment (cellID, expType, loopStruct, loop_mainSize, loop_rampSize,
                                        loop_excludedSize, nLoop, Nuplet, excludedFrames_inward).

    When a PincherTimeLapse is initialised, most of these variables are initialised to zero values.
    In order to compute the different fields, the following methods should be called in this order:
    - ptl.checkIfBlackFrames() : detect if there are black images at the end of each loop in the time lapse and
                                 classify them as not relevant by filling the appropriate fields.
    - ptl.saveFluoAside() : save the fluo images in an other folder and classify them as not relevant
                            for the rest of the image analysis.
    - ptl.determineFramesStatus_R40() : fill the idx_inNUp and idx_NUp column of the dictLog.
                                    in the idx_inNUp field: -1 means excluded ; 0 means ramp ; >0 means *position in* the n-uplet
                                    in the idx_NUp field: -1 means excluded ; 0 means ramp ; >0 means *number of* the n-uplet
    - ptl.uiThresholding() : Compute the threshold that will be used for segmentation in an interractive way.
    - ptl.saveMetaData() : Save the computed threshold along with a few other data.
    - ptl.makeFramesList() : Initialize the frame list.
    - ptl.detectBeads() : Detect all the beads or load their positions from a pre-existing '_Results.txt' file.
    - ptl.buildTrajectories() : Do the tracking of the beads of interest, with the user help, or load pre-existing trajectories.
      [In the meantime, Z computations and neighbours detections are performed on the Trajectory objects]
    - ptl.computeForces() : when the Trajectory objects are complete (with Z and neighbours), compute the forces.
                            Include recent corrections to the formula [October 2021].
    """

    def __init__(self, I, cellID, manipDict, NB = 2):
        # 1. Infos about the 3D image. The shape of the 3D image should be the following: T, Y, X !
        nS, ny, nx = I.shape[0], I.shape[1], I.shape[2]
        self.I = I
        self.nx = nx
        self.ny = ny
        self.nS = nS

        # 2. Infos about the experimental conditions, mainly from the DataFrame 'manipDict'.
        self.NB = NB # The number of beads of interest ! Typically 2 for a normal experiment, 4 for a triple pincher !
        self.cellID = cellID
        self.wFluoEveryLoop = manipDict['with fluo images']
        self.expType = manipDict['experimentType']
        self.scale = manipDict['scale pixel per um']
        self.OptCorrFactor = manipDict['optical index correction']
        self.MagCorrFactor = manipDict['magnetic field correction']
        self.Nuplet = manipDict['normal field multi images']
        self.Zstep = manipDict['multi image Z step']
        self.MagField = manipDict['normal field']
        self.BeadsZDelta = manipDict['beads bright spot delta']
        self.beadTypes = [bT for bT in str(manipDict['bead type']).split('_')]
        self.beadDiameters = [int(bD) for bD in str(manipDict['bead diameter']).split('_')]
        self.dictBeadDiameters = {}
        
        for k in range(len(self.beadTypes)):
            self.dictBeadDiameters[self.beadTypes[k]] = self.beadDiameters[k]

        self.microscope = manipDict['microscope']
        # self.excludedFrames_inward = np.zeros(self.nLoop, dtype = int)
        # self.excludedFrames_black = np.zeros(self.nLoop, dtype = int) 
        # self.excludedFrames_outward = np.zeros(self.nLoop, dtype = int)

        
        #### Import data from the optogen condition columns, if they exist
        fluo = False
        try:
            print(gs.ORANGE + 'Reading optogen parameters...' + gs.NORMAL)
            
            # "activationFirst" is the number of the loop 
            # at the end of which the first activ is
            # when you count the loop starting from 1
            
            
            self.activationFirst = int(manipDict['first activation'])
            self.activationLast = (manipDict['last activation'])
            self.activationFreq = int(manipDict['activation frequency'])
            self.activationExp = manipDict['activation exp']
            self.activationType = manipDict['activation type']
            
            if (not pd.isna(self.activationFreq)) and self.activationFreq > 0 and pd.isna(self.activationLast):
                print('case 1')
                self.LoopActivations = np.array([k-1 for k in range(self.activationFirst, self.nLoop, self.activationFreq)])
                # k-1 here cause we counted the loops starting from 1 but python start from 0.
            elif (not pd.isna(self.activationFreq)) and self.activationFreq > 0 and (not pd.isna(self.activationLast)):
                print('case 2')
                self.LoopActivations = np.array([k-1 for k in range(self.activationFirst, self.activationLast + 1, self.activationFreq)])
            
            else:
                print('case 3')
                self.LoopActivations = np.array([self.activationFirst-1])
                            
            fluo = True
                
        except:
            print(gs.ORANGE + 'No optogen parameters found' + gs.NORMAL)
            if self.wFluoEveryLoop:
                print(gs.ORANGE + 'Fluo every loop detection...' + gs.NORMAL)
                self.LoopActivations = np.arange(self.nLoop, dtype = int)
                
                fluo = True
        
        if fluo == False:
            print(gs.ORANGE + 'No fluo !' + gs.NORMAL)
            self.LoopActivations = np.array([])
            
        # if self.microscope == 'labview':
        #     self.totalActivationImages = np.array([np.sum(self.LoopActivations < kk) 
        #                                        for kk in range(self.nLoop)])
            
        #     self.excludedFrames_outward += self.totalActivationImages
            
            
        # else:
        #     pass


        # 3. Field that are just initialized for now and will be filled by calling different methods.
        self.listFrames = []
        self.listTrajectories = []

        # self.dictLog = {'iL' : np.zeros(nS, dtype = float),
        #                 'status_phase' : np.array(['Passive' for i in range(nS)]),
        #                 'iS' : np.array([i+1 for i in range(nS)]),
        #                 'iField' : np.array([i for i in range(nS)]),
        #                 'idx_inNUp' : np.zeros(nS, dtype = float),  # in the idx_inNUp field: -1 means excluded ; 0 means ramp ; >0 means position in the n-uplet
        #                 'idx_NUp' : np.zeros(nS, dtype = int), # in the idx_NUp field: -1 means excluded ; 0 means ramp ; >0 means number of the n-uplet
        #                 'UI' : np.zeros(nS, dtype = bool),
        #                 'UILog' : np.array(['' for i in range(nS)], dtype = '<U16'),
        #                 'UIxy' : np.zeros((nS,NB,2), dtype = int)}
        
        self.NLoops = 0
        self.logDf = pd.DataFrame({})
        self.log_UIxy = np.zeros((self.nS, self.NB, 2), dtype = int)
        self.nullFramesPerLoop = []
        self.fluoFramesPerLoop = []

        self.resultsDf = pd.DataFrame({'Area' : [],
                                               'StdDev' : [],
                                               'XM' : [],
                                               'YM' : [],
                                               'Slice' : []})

        
        self.modeNoUIactivated = False
        # End of the initialization !
       
    
    # def checkIfBlackFrames(self):
    #     """
    #     Check if some images in the time lapse are completely black.
    #     This happens typically when the computer is not able to save
    #     properly a series of large images with a high frequency.
    #     To detect them, compute the checkSum = np.sum(self.I[j]).
    #     Then modify the 'idx_inNUp' & 'idx_NUp' fields to '-1' in the dictLog.
    #     """
    #     if self.microscope == 'labview' or self.microscope == 'old-labview':
    #         offsets = np.array([np.sum(self.LoopActivations <= kk) 
    #                             for kk in range(self.nLoop)])
    #         # print(offsets)
    #         for i in range(self.nLoop):
    #             j = ((i+1)*self.loop_mainSize) - 1 + offsets[i]
    #             # print(j)
    #             checkSum = np.sum(self.I[j])
    #             while checkSum == 0:
    #                 print('Black image found')
    # #                 self.dictLog['Black'][j] = True
    #                 self.dictLog['idx_inNUp'][j] = -1
    #                 self.dictLog['idx_NUp'][j] = -1
    #                 self.excludedFrames_black[i] += 1
    #                 self.excludedFrames_inward[i] += 1 # More general
    #                 j -= 1
    #                 checkSum = np.sum(self.I[j])
            
    #     else:
    #         pass
        
        
    def initializeLogDf(self, statusPath):
        # Setting
        mainPhase = 'power'
        
        # Import status file
        statusCols = ['iL', 'status_phase', 'status_phase_details']
        statusDf = pd.read_csv(statusPath, sep = '_', names = statusCols)
        logDf = statusDf # statusCols = ['iL', 'status_phase', 'status_phase_details']
        
        #
        logDf['iField'] = np.arange(self.nS, dtype = int)
        logDf['iS'] = logDf['iField'].values + 1
        #
        logDf['idx_NUp'] = np.zeros(self.nS, dtype = int)
        logDf['idx_inNUp'] = np.zeros(self.nS, dtype = int)
        #
        logDf['status_action'] = np.array(['' for i in range(self.nS)], dtype = '<U16')
        #
        logDf['nullFrame'] = np.zeros(self.nS, dtype = int)
        logDf['trackFrame'] = np.ones(self.nS, dtype = bool)
        #
        logDf['idxAnalysis'] = np.zeros(self.nS, dtype = int)
        #
        logDf['UI'] = np.zeros(self.nS, dtype = bool)
        logDf['UILog'] = np.array(['' for i in range(self.nS)], dtype = '<U16')
        #
        log_UIxy = np.zeros((self.nS, self.NB, 2), dtype = int)
        #
        
        self.NLoops = np.max(logDf['iL'])
        Nuplet = self.Nuplet
        
        # Passive Part
        NPassive = logDf[logDf['status_phase'] == 'Passive'].shape[0]
        logDf.loc[logDf['status_phase'] == 'Passive', 'idx_inNUp'] = np.array([1 + i%Nuplet for i in range(NPassive)])
        logDf.loc[logDf['status_phase'] == 'Passive', 'idx_NUp'] = np.array([1 + i//Nuplet for i in range(NPassive)])
        
        # Fluo Part
        logDf[logDf['status_phase'] == 'Fluo']['idxAnalysis'] = -1
        
        # Action Part
        indexAction = logDf[logDf['status_phase'] == 'Action'].index
        Bstart = logDf.loc[indexAction, 'status_phase_details'].apply(lambda x : float(x.split('-')[1]))
        Bstop = logDf.loc[indexAction, 'status_phase_details'].apply(lambda x : float(x.split('-')[2]))
        logDf['deltaB'] = np.zeros(self.nS, dtype = float)
        logDf.loc[indexAction, 'deltaB'] =  Bstop - Bstart
        logDf['B_diff'] = np.array(['' for i in range(self.nS)], dtype = '<U4')
        logDf.loc[logDf['deltaB'] == 0, 'B_diff'] =  'none'
        logDf.loc[logDf['deltaB'] > 0, 'B_diff'] =  'up'
        logDf.loc[logDf['deltaB'] < 0, 'B_diff'] =  'down'
        
        logDf.loc[logDf['status_phase_details'].apply(lambda x : x.startswith('t^')), 'status_action'] = 'power'
        logDf.loc[logDf['status_phase_details'].apply(lambda x : x.startswith('sigmoid')), 'status_action'] = 'sigmoid'
        logDf.loc[logDf['status_phase_details'].apply(lambda x : x.startswith('constant')), 'status_action'] = 'constant'
        
        logDf.loc[indexAction, 'status_action'] = logDf.loc[indexAction, 'status_action'] + '_' + logDf.loc[indexAction, 'B_diff']
        logDf = logDf.drop(columns=['deltaB', 'B_diff'])
        
        # idxAnalysis
        logDf.loc[indexAction, 'idxAnalysis'] = (-1)*logDf.loc[indexAction, 'iL']
        indexMainPhase = logDf[logDf['status_action'].apply(lambda x : x.startswith(mainPhase))].index
        logDf.loc[indexMainPhase, 'idxAnalysis'] = (-1)*logDf.loc[indexMainPhase, 'idxAnalysis']
        
        # idxAnalysis for loops with repeated compressions
        previous_idx = 0
        for i in range(1, self.NLoops+1):
            indexLoop_i = logDf[logDf['iL'] == i].index
            maskActionPhase_i = logDf.loc[indexLoop_i, 'status_phase'].apply(lambda x : x.startswith('Action')).values.astype(int)
            maskMainPhase_i = logDf.loc[indexLoop_i, 'status_action'].apply(lambda x : x.startswith(mainPhase)).values.astype(int)
            maskActionPhase_nonMain_i = np.logical_xor(maskMainPhase_i, maskActionPhase_i)
            # print(maskMainPhase_i)
            
            lab_main, nlab_main = ndi.label(maskMainPhase_i)
            # print(lab, nlab)
            
            if nlab_main > 1: # This means there are repeated compressions. Nothing was modified before this test
                logDf.loc[indexLoop_i, 'idxAnalysis'] = (lab_main + (previous_idx*maskMainPhase_i))
                lab_nonMain, nlab_nonMain = ndi.label(maskActionPhase_nonMain_i)
                logDf.loc[indexLoop_i, 'idxAnalysis'] -= (lab_nonMain + (previous_idx*maskActionPhase_nonMain_i))
                # print(logDf.loc[indexLoop_i, 'idxAnalysis'].values
                previous_idx = np.max(lab_main)
                
                
        
        self.logDf = logDf
        self.log_UIxy = log_UIxy
        
        
        
    def detectNullFrames(self):
        """
        Check if some images in the time lapse are completely black.
        This happens typically when the computer is not able to save
        properly a series of large images with a high frequency.
        """
        # Setting
        fastestPhase = 'power_up'
        
        # print(self.nullFramesPerLoop)
        
        if self.microscope == 'labview' or self.microscope == 'old-labview':
            # logDf = self.logDf
            NLoops = self.NLoops
            for i in range(1, NLoops+1):
                nullFrames = []
                logDf_loop = self.logDf[self.logDf['iL'] == i]
                iS = logDf_loop['iS'].values[-1]
                
                logDf_fast = logDf_loop[logDf_loop['status_action'] == fastestPhase]
                iS_fast = logDf_fast['iS'].values[-1]
                
                while np.sum(self.I[iS-1]) == 0:
                    nullFrames.append(iS_fast)
                    self.logDf.loc[self.logDf['iS'] == iS_fast, 'nullFrame'] = 1
                    self.logDf.loc[self.logDf['iS'] == iS_fast, 'trackFrame'] = False
                    iS -= 1
                    iS_fast -= 1
                if len(nullFrames) > 0:
                    print('Loop {:.0f}: {:.0f} null image(s) found'.format(i, len(nullFrames)))
                    
                A = np.cumsum(self.logDf[self.logDf['iL'] == i]['nullFrame'].values)
                self.logDf.loc[self.logDf['iL'] == i, 'iS'] = self.logDf[self.logDf['iL'] == i]['iS'] - A
                # logDf_loop = self.logDf[self.logDf['iL'] == i]
                # A = np.cumsum(logDf_loop['nullFrame'].values)
                # logDf_loop['iS'] = logDf_loop['iS'] - A
                    
                self.nullFramesPerLoop.append(nullFrames[::-1])
                # print(self.nullFramesPerLoop)

    
        else:
            pass
        
            
    def detectFluoFrames(self, save = True, fluoDirPath = '', f = ''):
        """
        Find and save all of the fluo images.
        """
        # Setting
        self.fluoFramesPerLoop = [[] for i in range(self.NLoops)]
        
        if self.microscope == 'labview' or self.microscope == 'old-labview':
            indexFluo = self.logDf['status_phase'].apply(lambda x : x.startswith('Fluo'))
            fluoFrames_iS = self.logDf.loc[indexFluo, 'iS'].values
            fluoFrames_iL = self.logDf.loc[indexFluo, 'iL'].values
            for k in range(len(fluoFrames_iS)):
                iL, iS = fluoFrames_iL[k], fluoFrames_iS[k]
                self.fluoFramesPerLoop[iL].append(iS)
                self.logDf.loc[self.logDf['iS'] == iS, 'trackFrame'] = False
                
            if save:
                if not os.path.exists(fluoDirPath):
                    os.makedirs(fluoDirPath)
                for fluoFrames_iS in self.fluoFramesPerLoop:
                    for iS in fluoFrames_iS:
                        Ifluo = self.I[iS-1]
                        path = os.path.join(fluoDirPath, f[:-4] + '_Fluo_' + str(iS) + '.tif')
                        io.imsave(path, Ifluo, check_contrast=False)
        else:
            pass


    # def saveFluoAside(self, fluoDirPath, f):
    #     """
    #     If wFluo = True in the expDf, modify the 'idx_inNUp' & 'idx_NUp' fields to '-1' in the dictLog.
    #     And if the directory for the fluo images has not be created yet,
    #     find and save all of the fluo images there.
    #     """
        
    #     if self.microscope == 'labview' or self.microscope == 'old-labview':
    #         # print('excludedFrames_black')
    #         # print(self.excludedFrames_black)
    #         try:
    #             if self.activationFirst > 0:
                        
    #                 for iLoop in self.LoopActivations:
                        
    #                     totalExcludedOutward = np.sum(self.excludedFrames_outward[iLoop])
                        
    #                     j = int(((iLoop+1)*self.loop_mainSize) + totalExcludedOutward - self.excludedFrames_black[iLoop])
    #                     # print(j)
    #                     self.dictLog['idx_inNUp'][j] = -1
    #                     self.dictLog['idx_NUp'][j] = -1

    #         except:
    #             if self.wFluoEveryLoop:
    #                 for iLoop in self.LoopActivations:
    #                     totalExcludedOutward = self.excludedFrames_outward[iLoop]
    #                     j = int(((iLoop+1)*self.loop_mainSize) + totalExcludedOutward - self.excludedFrames_black[iLoop])
    #                     self.dictLog['idx_inNUp'][j] = -1
    #                     self.dictLog['idx_NUp'][j] = -1
                    
                        
                    
    #     if not os.path.exists(fluoDirPath):
    #         os.makedirs(fluoDirPath)
    #         for iLoop in self.LoopActivations:
    #             totalExcludedOutward = np.sum(self.excludedFrames_outward[iLoop])
    #             j = int(((iLoop+1)*self.loop_mainSize) + totalExcludedOutward - self.excludedFrames_black[iLoop])
    #             Ifluo = self.I[j]

    #             path = os.path.join(fluoDirPath, f[:-4] + '_Fluo_' + str(j+1) + '.tif')
    #             io.imsave(path, Ifluo, check_contrast=False)
            
            
            
    #         # try: # Optogenetic activations behaviour
    #         #     print(gs.ORANGE + 'Trying optogen fluo detection...' + gs.NORMAL)
    #         #     if self.activationFirst > 0 and not self.wFluoEveryLoop:
    #         #         if (not pd.isna(self.activationFreq)) and self.activationFreq > 0: # Meaning there is a repetition of activation
    #         #         # self.LoopActivations = np.array([k-1 for k in range(self.activationFirst, self.nLoop, self.activationFreq)])    
    #         #             for iLoopActivation in self.LoopActivations:
    #         #                 totalExcludedOutward = np.sum(self.excludedFrames_outward[iLoopActivation])
    #         #                 j = int(((iLoopActivation+1)*self.loop_mainSize) + totalExcludedOutward - self.excludedFrames_black[iLoopActivation])
    #         #                 self.dictLog['idx_inNUp'][j] = -1
    #         #                 self.dictLog['idx_NUp'][j] = -1
    #         #             print('Total excluded fluoro: '+str(totalExcludedOutward))
                        
                            
    #         #         else: # Set self.activationFreq = 0 to only detect one single activation
    #         #             iLoopActivation = self.activationFirst-1
    #         #             totalExcludedOutward = np.sum(self.excludedFrames_outward[iLoopActivation])
    #         #             # LoopActivations = [iLoopActivation]
                        
    #         #             j = int(((iLoopActivation+1)*self.loop_mainSize) + totalExcludedOutward - self.excludedFrames_black[iLoopActivation])
    #         #             self.dictLog['idx_inNUp'][j] = -1
    #         #             self.dictLog['idx_NUp'][j] = -1
                        
    #         #         if not os.path.exists(fluoDirPath):
    #         #             os.makedirs(fluoDirPath)
    #         #             for iLoopActivation in self.LoopActivations:
    #         #                 totalExcludedOutward = np.sum(self.excludedFrames_outward[iLoopActivation])
    #         #                 j = int(((iLoopActivation+1)*self.loop_mainSize) + totalExcludedOutward - self.excludedFrames_black[iLoopActivation])
    #         #                 Ifluo = self.I[j]
    #         #                 path = os.path.join(fluoDirPath, f[:-4] + '_Fluo_' + str(j) + '.tif')
    #         #                 io.imsave(path, Ifluo)
                            
    #         #     print(gs.ORANGE + '...success' + gs.NORMAL)
    
    #         # except: 
    #         #     print(gs.ORANGE + '...failure' + gs.NORMAL)
            
            
    #         # if self.wFluoEveryLoop: # Behaviour when fluo check every loop => Not Optogen
    #         #     print(gs.ORANGE + 'Doing classic fluo detection' + gs.NORMAL)
    #         #     for i in range(self.nLoop):
    #         #         j = int(((i+1)*self.loop_mainSize) + totalExcludedOutward - self.excludedFrames_black[i])
    #         #         self.dictLog['idx_inNUp'][j] = -1
    #         #         self.dictLog['idx_NUp'][j] = -1
    #         #         print(j)
    
    #         #     if not os.path.exists(fluoDirPath):
    #         #         os.makedirs(fluoDirPath)
    #         #         for i in range(self.nLoop):
    #         #             j = int(((i+1)*self.loop_mainSize) - self.excludedFrames_black[i])
    #         #             Ifluo = self.I[j]
    #         #             path = os.path.join(fluoDirPath, f[:-4] + '_Fluo_' + str(j) + '.tif')
    #         #             io.imsave(path, Ifluo)
                        

    # def determineFramesStatus_R40(self):
    #     #### Exp type dependance here
    #     """
    #     Fill the idx_inNUp and idx_NUp column of the dictLog, in the case of a compression (R40) or constant field (thickness) experiment
    #     > in the idx_inNUp field: -1 means excluded ; 0 means ramp ; 10 > x > 0 means *position in* the n-uplet.
    #     > in the idx_NUp field: -1 means excluded ; 0 means ramp ; >0 means *number of* the n-uplet.
    #     Not very elegant but very confortable to work with.
    #     """
    #     N0 = self.loop_mainSize
    #     Nramp0 = self.loop_rampSize
    #     # Nexclu = self.loop_excludedSize
    #     nUp = self.Nuplet
    #     # N = N0 - Nexclu
    #     Nct = N0 - Nramp0 # N
    #     i_nUp = 1
    #     # print(N0,Nramp0,nUp)

    #     for i in range(self.nLoop):
    #         totalExcludedOutward = np.sum(self.excludedFrames_outward[i])
    #         jstart = int(i*N0 + totalExcludedOutward)
    #         if Nramp0 == 0:
    #             for j in range(N0): # N
    #                 self.dictLog['idx_inNUp'][jstart + j] = 1 + j%self.Nuplet
    #                 self.dictLog['idx_NUp'][jstart + j] = i_nUp + j//self.Nuplet
    #         else:
    #             Nramp = Nramp0-self.excludedFrames_black[i]
    #             for j in range(Nct//2): # Ct field before ramp
    #                 self.dictLog['idx_inNUp'][jstart + j] = 1 + j%self.Nuplet
    #                 self.dictLog['idx_NUp'][jstart + j] = i_nUp + j//self.Nuplet
    #             i_nUp = max(self.dictLog['idx_NUp']) + 1
    #             jstart += int(Nct//2 + Nramp) # In the ramp it self, the two status stay equal to 0
    #             for j in range(Nct//2): # Ct field after ramp
    #                 self.dictLog['idx_inNUp'][jstart + j] = 1 + j%self.Nuplet
    #                 self.dictLog['idx_NUp'][jstart + j] = i_nUp + j//self.Nuplet
    #             i_nUp = max(self.dictLog['idx_NUp']) + 1
                

    # def determineFramesStatus_L40(self):
    #     #### Exp type dependance here
    #     """
    #     Fill the idx_inNUp and idx_NUp column of the dictLog, in the case of a compression with an initial decrease (L40)
    #     > in the idx_inNUp field: -1 means excluded ; 0 means ramp ; 0.1 means pre-ramp ; 10 > x > 0 means *position in* the n-uplet.
    #     > in the idx_NUp field: -1 means excluded ; 0 means not in a n-uplet ; x > 0 means *number of* the n-uplet.
    #     Not very elegant but very confortable to work with.
    #     """
        
    #     N0 = self.loop_mainSize
    #     Nramp0 = self.loop_rampSize
    #     # Nexclu = self.loop_excludedSize
    #     nUp = self.Nuplet
    #     # N = N0 - Nexclu
    #     Nct = N0 - 2*Nramp0 # N
    #     i_nUp = 1
        
        
    #     # if not self.wFluoEveryLoop:
    #     #     mask_notAlreadyExcluded = self.dictLog['idx_inNUp'] >= 0
    #     #     print(mask_notAlreadyExcluded)
    #     # else:
    #     #     mask_notAlreadyExcluded = np.ones(len(self.dictLog['idx_inNUp']), dtype = bool)
        
    #     for i in range(self.nLoop):
    #         totalExcludedOutward = self.excludedFrames_outward[i]
    #         jstart = int(i*N0 + totalExcludedOutward)
    #         # print(jstart)
    #         if Nramp0 == 0:
    #             for j in range(N0): # N
    #                 self.dictLog['idx_inNUp'][jstart + j] = 1 + j%self.Nuplet
    #                 self.dictLog['idx_NUp'][jstart + j] = i_nUp + j//self.Nuplet
                    
    #         else:
    #             Nramp = Nramp0-self.excludedFrames_black[i]
    #             for j in range(Nct//2): # Ct field before ramp
    #                 if self.dictLog['idx_inNUp'][jstart + j] != 0:
    #                     print(gs.ORANGE + 'Careful ! Rewriting on some data in position {:.0f} (starting from 0)'.format(jstart + j) + '' + gs.NORMAL)
    #                     print(gs.ORANGE + 'Writing {:.1f} instead of {:.1f}'.format(1 + j%self.Nuplet, self.dictLog['idx_inNUp'][jstart + j]) + '' + gs.NORMAL)
    #                 self.dictLog['idx_inNUp'][jstart + j] = 1 + j%self.Nuplet
    #                 self.dictLog['idx_NUp'][jstart + j] = i_nUp + j//self.Nuplet
                    
    #             jstart += int(Nct//2)
    #             for j in range(Nramp0): # Pre-ramp
    #                 self.dictLog['idx_inNUp'][jstart + j] = 0.1 
    #                 self.dictLog['idx_NUp'][jstart + j] = 0
    #             i_nUp = max(self.dictLog['idx_NUp']) + 1
    #             jstart += int(Nramp0 + Nramp) # In the ramp itself, the two status stay equal to 0
                
    #             for j in range(Nct//2): # Ct field after ramp
    #                 if self.dictLog['idx_inNUp'][jstart + j] != 0:
    #                     print(gs.ORANGE + 'Careful ! Rewriting on some data in position {:.0f} (starting from 0)'.format(jstart + j) + '' + gs.NORMAL)
    #                     print(gs.ORANGE + 'Writing {:.1f} instead of {:.1f}'.format(1 + j%self.Nuplet, self.dictLog['idx_inNUp'][jstart + j]) + '' + gs.NORMAL)
    #                 self.dictLog['idx_inNUp'][jstart + j] = 1 + j%self.Nuplet
    #                 self.dictLog['idx_NUp'][jstart + j] = i_nUp + j//self.Nuplet
    #             i_nUp = max(self.dictLog['idx_NUp']) + 1
                    

    # def determineFramesStatus_optoGen(self):
    # #### Exp type dependance here
    #     N0 = self.loop_mainSize
    #     Nramp0 = self.loop_rampSize
    #     # Nexclu = self.loop_excludedSize
    #     nUp = self.Nuplet
    #     # N = N0 - Nexclu
    #     i_nUp = 1

    #     # print(N0,Nramp0,Nexclu,nUp)
    #     if self.microscope == 'metamorph':
    #         for i in range(self.nLoop):
    #             jstart = int(i*N0)
    #             for j in range(N0): # N
    #                 self.dictLog['idx_inNUp'][jstart + j] = 1 + j%self.Nuplet
    #                 self.dictLog['idx_NUp'][jstart + j] = i_nUp + j//self.Nuplet
    #             i_nUp = max(self.dictLog['idx_NUp']) + 1
                
    #     elif self.microscope == 'labview' or self.microscope == 'old-labview':
    #         for i in range(self.nLoop):
    #             totalExcludedOutward = (self.excludedFrames_outward[i])
    #             jstart = int(i*N0 + totalExcludedOutward)
    #             for j in range(N0): # N
    #                 self.dictLog['idx_inNUp'][jstart + j] = 1 + j%self.Nuplet
    #                 self.dictLog['idx_NUp'][jstart + j] = i_nUp + j//self.Nuplet
    #             i_nUp = max(self.dictLog['idx_NUp']) + 1
            
        
                
    # def determineFramesStatus_Sinus(self):
    #     #### Exp type dependance here
    #     pass
        
    # def determineFramesStatus_BR(self):
    #     #### Exp type dependance here
    #     pass
    
    def makeOptoMetadata(self, fieldDf, display = 1, save = False, path = ''):
        try:
            actFreq = self.activationFreq
            actExp = self.activationExp
            actType = [self.activationType]
            microscope = self.microscope
            if microscope == 'labview':
                allActivationIndices = ufun.findActivation(fieldDf)[0]
                # actFirst = idxActivation//self.loop_mainSize
                timeScaleFactor = 1000
                
                print(fieldDf)
                actN = len(allActivationIndices)
                fieldToMeta = fieldDf['T_abs'][fieldDf.index.isin(allActivationIndices)]
                metadataDict = {}
                metadataDict['activationNo'] = np.linspace(1, actN, actN)
                metadataDict['Slice'] = allActivationIndices
                #timeScaleFactor converts the time to milliseconds for the labview code and keeps it the same if from Metamorph
                metadataDict['T_abs'] = fieldToMeta/timeScaleFactor
                metadataDict['T_0'] = [fieldDf['T_abs'][0]/timeScaleFactor]*actN
                # metadataDict['Exp'] = actExp*np.ones(actN, dtype = type(actN))
                metadataDict['Type'] = actType*actN
                print(len(fieldToMeta))
                print(len(metadataDict['activationNo']))
                print(len(metadataDict['Slice']))
                print(len(metadataDict['T_abs']))
                print(len(metadataDict['T_0']))
                
                metadataDf = pd.DataFrame(metadataDict)
                if save:
                    metadataDf.to_csv(path, sep='\t')
        except:
            pass
        
        # if display == 1:
        #     print('\n\n* Initialized Log Table:\n')
        # if display == 2:
        #     print('\n\n* Filled Log Table:\n')
        #     print(metadataDf[metadataDf['UI']])
            
    # def makeOptoMetadata_V1(self, fieldDf, display = 1, save = False, path = ''):
    #     actFreq = self.activationFreq
    #     actExp = self.activationExp
    #     actType = [self.activationType]
    #     microscope = self.microscope
    #     if microscope == 'labview' or microscope == 'old-labview':
    #         idxActivation = ufun.findActivation_V1(fieldDf)[0]
    #         actFirst = idxActivation//self.loop_mainSize
    #         timeScaleFactor = 1000
    #     elif microscope == 'metamorph':
    #         actFirst = self.activationFirst
    #         #+2 in idxAnalysis because the time included in the timeSeriesfile is the third of the triplet
    #         idxActivation = actFirst*self.loop_mainSize-1
    #         timeScaleFactor = 1
        
    #     actN = ((self.nLoop - actFirst))//actFreq
        
    #     metadataDict = {}
    #     metadataDict['Total'] = actN*np.ones(actN, dtype = type(actN))
    #     metadataDict['Slice'] = idxActivation
    #     #timeScaleFactor converts the time to milliseconds for the labview code and keeps it the same if from Metamorph
    #     metadataDict['T_abs'] = fieldDf['T_abs'][idxActivation]/timeScaleFactor 
    #     metadataDict['T_0'] = fieldDf['T_abs'][0]/timeScaleFactor 
    #     # metadataDict['Exp'] = actExp*np.ones(actN, dtype = type(actN))
    #     metadataDict['Type'] = actType*actN
        
    #     metadataDf = pd.DataFrame(metadataDict)
    #     if save:
    #         metadataDf.to_csv(path, sep='\t')
        
    #     if display == 1:
    #         print('\n\n* Initialized Log Table:\n')
    #     if display == 2:
    #         print('\n\n* Filled Log Table:\n')
    #         print(metadataDf[metadataDf['UI']])
            
    
    def saveLogDf(self, display = 1, save = False, path = ''):
        """
        Save the dictLog so that next time it can be directly reloaded to save time.
        """
        dL = {}
        for i in range(self.NB):
            dL['UIx'+str(i+1)] = self.log_UIxy[:,i,0]
            dL['UIy'+str(i+1)] = self.log_UIxy[:,i,1]
        df = pd.DataFrame(dL)
        logDf = pd.concat([self.logDf, df], axis = 1)
        
        if save:
            logDf.to_csv(path, sep='\t', index=False)

        if display == 1:
            print('\n\n* Initialized Log Table:\n')
            print(logDf)
        if display == 2:
            print('\n\n* Filled Log Table:\n')
            print(logDf[logDf['UI']])

    
                
    # def saveLog(self, display = 1, save = False, path = ''):
    #     """
    #     Save the dictLog so that next time it can be directly reloaded to save time.
    #     """
    #     dL = {}
    #     dL['iS'], dL['idx_inNUp'], dL['idx_NUp'] = \
    #         self.dictLog['iS'], self.dictLog['idx_inNUp'], self.dictLog['idx_NUp']

    #     dL['UI'], dL['UILog'] = \
    #         self.dictLog['UI'], self.dictLog['UILog']
    #     for i in range(self.NB):
    #         dL['UIx'+str(i+1)] = self.dictLog['UIxy'][:,i,0]
    #         dL['UIy'+str(i+1)] = self.dictLog['UIxy'][:,i,1]
    #     dfLog = pd.DataFrame(dL)
    #     if save:
    #         dfLog.to_csv(path, sep='\t')

    #     if display == 1:
    #         print('\n\n* Initialized Log Table:\n')
    #         print(dfLog)
    #     if display == 2:
    #         print('\n\n* Filled Log Table:\n')
    #         print(dfLog[dfLog['UI']])
    
    
            
    def importLogDf(self, path):
        """
        Import the dictLog.
        """
        logDf = pd.read_csv(path, sep='\t')
        for i in range(self.NB):
            xkey, ykey = 'UIx'+str(i+1), 'UIy'+str(i+1)
            self.log_UIxy[:,i,0] = logDf[xkey].values
            self.log_UIxy[:,i,1] = logDf[ykey].values
            logDf = logDf.drop(columns=[xkey, ykey])
        logDf['UILog'] = logDf['UILog'].astype(str)
        self.logDf = logDf
            

    # def importLog(self, path):
    #     """
    #     Import the dictLog.
    #     """
    #     dfLog = pd.read_csv(path, sep='\t')
    #     dL = dfLog.to_dict()
    #     self.dictLog['iL'], self.dictLog['status_phase'], self.dictLog['status_phase_details'] = \
    #         dfLog['iL'].values, dfLog['status_phase'].values, dfLog['status_phase_details'].values
    #     self.dictLog['iS'], self.dictLog['idx_inNUp'], self.dictLog['idx_NUp'] = \
    #         dfLog['iS'].values, dfLog['idx_inNUp'].values, dfLog['idx_NUp'].values
    #     self.dictLog['UI'], self.dictLog['UILog'] = \
    #         dfLog['UI'].values, dfLog['UILog'].values
    #     for i in range(self.NB):
    #         xkey, ykey = 'UIx'+str(i+1), 'UIy'+str(i+1)
    #         self.dictLog['UIxy'][:,i,0] = dfLog[xkey].values
    #         self.dictLog['UIxy'][:,i,1] = dfLog[ykey].values

        


    def makeFramesList(self):
        """
        Initialize the Frame objects and add them to the PTL.listFrames list.
        """
        self.logDf['iF'] = np.ones(self.nS, dtype = int) * (-1)
        iF = 0
        for i in range(self.nS):
            if self.logDf['trackFrame'].values[i]:
                iL = self.logDf['iL'].values[i]
                iS = self.logDf['iS'].values[i]
                idx_NUp = self.logDf['idx_NUp'].values[i]
                idx_inNUp = self.logDf['idx_inNUp'].values[i]
                Nup = (self.Nuplet * (idx_NUp > 0))  +  (1 * (idx_NUp <= 0))
                # The Nup field of a slice is = to self.Nuplet if the idx_inNUp indicates that the frame is part of a multi image n-uplet
                # Otherwise the image is "alone", like in a compression, and therefore Nup = 1
                
                resDf = self.resultsDf.loc[self.resultsDf['Slice'] == iS]
                frame = Frame(self.I[iS-1], iL, iS, self.NB, Nup, idx_inNUp, idx_NUp, self.scale, resDf)
                frame.makeListBeads()
                
                self.listFrames.append(frame)
                self.logDf.loc[i, 'iF'] = iF
                iF += 1
                
        iF_column = self.logDf.pop('iF')
        self.logDf.insert(7, iF_column.name, iF_column)

    # def detectBeads(self, resFileImported):
    #     """
    #     If no '_Results.txt' file has been previously imported, ask each Frame
    #     object in the listFrames to run its Frame.detectBeads() method.
    #     Then concatenate the small 'Frame.resDf' to the big 'PTL.resultsDf' DataFrame,
    #     so that in the end you'll get a DataFrame that has exactly the shape of a '_Results.txt' file made from IJ.
    #     *
    #     If a '_Results.txt' file has been previously imported, just assign to each Frame
    #     object in the listFrames the relevant resDf DataFrame
    #     (each resDf is, as said earlier, just a fragment of the PTL.resultsDf).
    #     """
    #     for frame in self.listFrames: #[:3]:
    #         if not resFileImported:
    #             plot = 0
    #             frame.detectBeads(plot)
    #             self.resultsDf = pd.concat([self.resultsDf, frame.resDf])
    #         else:
    #             resDf = self.resultsDf.loc[self.resultsDf['Slice'] == frame.iS+1]
    #             frame.resDf = resDf

    #         frame.makeListBeads()

    #     if not resFileImported:
    #         self.resultsDf = self.resultsDf.convert_dtypes()
    #         self.resultsDf.reset_index(inplace=True)
    #         self.resultsDf.drop(['index'], axis = 1, inplace=True)
            

    # def saveBeadsDetectResult(self, path):
    #     """
    #     Save the 'PTL.resultsDf' DataFrame.
    #     """
    #     self.resultsDf.to_csv(path, sep='\t', index = False)

    def importBeadsDetectResult(self, path=''):
        """
        Import the 'PTL.resultsDf' DataFrame.
        """
        df = pd.read_csv(path, sep='\t')
        for c in df.columns:
            if 'Unnamed' in c:
                df.drop([c], axis = 1, inplace=True)
        self.resultsDf = df

    def findBestStd(self):
        """
        Simpler and better than findBestStd_V0 using the idx_NUp column of the dictLog.
        ---
        For each frame of the timelapse that belongs to a N-uplet, I want to reconsititute this N-uplet
        (meaning the list of 'Nup' consecutive images numbered from 1 to Nup,
        minus the images eventually with no beads detected).
        Then for each N-uplet of images, i want to find the max standard deviation
        and report its position because it's for the max std that the X and Y detection is the most precise.
        ---
        This is very easy thanks to the 'idx_NUp', because it contains a different number for each N-Uplet.
        """

        Nup = self.Nuplet
        nT = self.listTrajectories[0].nT
        idx_NUp = self.listTrajectories[0].dict['idx_NUp']
        idx_inNUp = self.listTrajectories[0].dict['idx_inNUp']
        sum_std = np.zeros(nT)
        for i in range(self.NB):
            sum_std += np.array(self.listTrajectories[i].dict['StdDev'])
        
        bestStd = np.zeros(nT, dtype = bool)
        i = 0
        while i < nT:
            if idx_inNUp[i] == 0:
                bestStd[i] = True
                i += 1
            elif idx_inNUp[i] > 0:
                s2 = idx_NUp[i]
                L = [i]
                j = 0
                while i+j < nT-1 and idx_NUp[i+j+1] == s2: # lazy evaluation of booleans
                    j += 1
                    L.append(i+j)
                #print(L)
                loc_std = sum_std[L]
                i_bestStd = i + int(np.argmax(loc_std))
                bestStd[i_bestStd] = True
                L = []
                i = i + j + 1

        return(bestStd)
        
    
    def buildTrajectories(self, trackAll = False):
        """
        The main tracking function.
        *
        Note about the naming conventions here:
        - 'iF': index in the list of Frames ;
        - 'iB': index in a list of Beads or a list of Trajectories ;
        - 'iS': index of the slice in the image I (but here python starts with 0 and IJ starts with 1);
        - 'Boi' refers to the 'Beads of interest', ie the beads that are being tracked.
        """
        
        #### 1. Initialize the BoI position in the first image where they can be detect, thanks to user input.
        init_iF = 0
        init_ok = False
        while not init_ok:
            init_iS = self.listFrames[init_iF].iS
            if not self.logDf.loc[self.logDf['iS'] == init_iS, 'UI'].values[0]: # Nothing in the log yet
                self.listFrames[init_iF].show()
                mngr = plt.get_current_fig_manager()
                mngr.window.setGeometry(720, 50, 1175, 1000)
                QA = pyautogui.confirm(
                    text='Can you point the beads of interest\nin the image ' + str(init_iS) + '?',
                    title='Initialise tracker',
                    buttons=['Yes', 'Next Frame', 'Quit'])
                if QA == 'Yes':
                    init_ok = True
                    ui = plt.ginput(self.NB, timeout=0)
                    uiXY = ufun.ui2array(ui)
                    self.logDf.loc[self.logDf['iS'] == init_iS, 'UI'] = True
                    self.logDf.loc[self.logDf['iS'] == init_iS, 'UILog'] = 'init_' + QA
                    self.log_UIxy[init_iS-1] = uiXY
                elif QA == 'Next Frame':
                    self.logDf.loc[self.logDf['iS'] == init_iS, 'UI'] = True
                    self.logDf.loc[self.logDf['iS'] == init_iS, 'UILog'] = 'init_' + QA
                    init_iF += 1
                else:
                    fig = plt.gcf()
                    plt.close(fig)
                    return('Bug')

                fig = plt.gcf()
                plt.close(fig)

            else: # Action to do already in the log
                QA = self.logDf.loc[self.logDf['iS'] == init_iS, 'UILog'].values[0]
                if QA == 'init_Yes':
                    init_ok = True
                    uiXY = self.log_UIxy[init_iS-1]
                elif QA == 'init_Next Frame':
                    init_iF += 1
                else:
                    print('Strange event in the tracking init')

        init_BXY = self.listFrames[init_iF].beadsXYarray()
        M = ufun.compute_cost_matrix(uiXY,init_BXY)
        row_ind, col_ind = linear_sum_assignment(M) # row_ind -> clicks / col_ind -> listBeads
        
        # Sort the beads by growing X coordinates on the first image,
        # So that iB = 0 has a X inferior to iB = 1, etc.
        sortM = np.array([[init_BXY[col_ind[i],0], col_ind[i]] for i in range(len(col_ind))])
        sortM = sortM[sortM[:, 0].argsort()]
        
        # Initialise position of the beads
        init_iBoi = sortM[:, 1].astype(int)
        # init_BoiXY = sortM[:, 0]
        init_BoiXY = np.array([init_BXY[init_iBoi[i]] for i in range(len(init_iBoi))])
        
        
        #### 2. Creation of the Trajectory objects
        for iB in range(self.NB):
            self.listTrajectories.append(Trajectory(self.I, self.cellID, self.listFrames, self.scale, self.Zstep, iB))

            self.listTrajectories[iB].dict['Bead'].append(self.listFrames[init_iF].listBeads[init_iBoi[iB]])
            self.listTrajectories[iB].dict['iF'].append(init_iF)
            self.listTrajectories[iB].dict['iS'].append(self.listFrames[init_iF].iS)
            self.listTrajectories[iB].dict['iL'].append(self.listFrames[init_iF].iL)
            self.listTrajectories[iB].dict['iB_inFrame'].append(init_iBoi[iB])
            self.listTrajectories[iB].dict['X'].append(init_BoiXY[iB][0])
            self.listTrajectories[iB].dict['Y'].append(init_BoiXY[iB][1])
            self.listTrajectories[iB].dict['StdDev'].append(self.listFrames[init_iF].beadsStdDevarray()[init_iBoi[iB]])
            # self.listTrajectories[iB].dict['idx_inNUp'].append(self.listFrames[init_iF].idx_inNUp)
            # self.listTrajectories[iB].dict['idx_NUp'].append(self.listFrames[init_iF].idx_NUp)
            
            # #### >>> Exp type dependance here (01)
            # if 'compressions' in self.expType or 'constant field' in self.expType:
            #     self.listTrajectories[iB].dict['idxAnalysis'].append((self.listFrames[init_iF].idx_inNUp == 0))
                
            # #### TBC !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # elif 'sinus' in self.expType:
            #      print('Passed expt type')
            #      self.listTrajectories[iB].dict['idxAnalysis'].append(0)
                 
            # elif 'brokenRamp' in self.expType:
            #      self.listTrajectories[iB].dict['idxAnalysis'].append(0)
                 
            # elif 'optoGen' in self.expType:
            #      self.listTrajectories[iB].dict['idxAnalysis'].append(0)

        #### 3. Start the tracking
        previous_iF = init_iF
        previous_iBoi = init_iBoi
        previous_BXY = init_BXY
        previous_BoiXY = init_BoiXY
        
        
        for iF in range(init_iF+1, len(self.listFrames)):
            validFrame = True
            askUI = False
            
            #### 3.1 Check the number of detected objects
            if self.listFrames[iF].NBdetected < self.NB: # -> Next frame
                validFrame = False
                continue
            
            #### 3.2 Try an automatic tracking
            if not trackAll:
                trackXY = previous_BoiXY
                previous_iBoi = [i for i in range(self.NB)]
            elif trackAll:
                trackXY = previous_BXY
                
            BXY = self.listFrames[iF].beadsXYarray()
            M = ufun.compute_cost_matrix(trackXY,BXY)
            row_ind, col_ind = linear_sum_assignment(M)
            costs = np.array([M[row_ind[iB], col_ind[iB]] for iB in range(len(row_ind))])
            foundBoi = []
            for iBoi in previous_iBoi:
                searchBoi = np.flatnonzero(row_ind == iBoi)
                if len(searchBoi) == 1:
                    foundBoi.append(searchBoi[0])
                                   
            
            #### 3.3 Assess if asking user input is necessary
            
            highCost = ((np.max(costs)**0.5) * (1/self.scale) > 0.5)
            # True if the distance travelled by one of the BoI is greater than 0.5 um
            
            allBoiFound = (len(foundBoi) == self.NB)
            # False if one of the beads of interest have not been detected
            
            if highCost or not allBoiFound:
                askUI = True
                
            #### 3.4 If not, automatically assign the positions of the next beads

            if not askUI:
                try:
                    iBoi = [col_ind[iB] for iB in foundBoi]
                    BoiXY = np.array([BXY[iB] for iB in iBoi])
                    
                except:
                    askUI = True
                    print('Error for ' + str(iF))
                    print('M')
                    print(M)
                    print('row_ind, col_ind')
                    print(row_ind, col_ind)
                    print('previous_iBoi')
                    print(previous_iBoi)
                    print('costs')
                    print(costs)
                    

            #### 3.5 If one of the previous steps failed, ask for user input
            if askUI:
                iS = self.listFrames[iF].iS
                
                #### 3.5.1: Case when the UI has been previously saved in the dictLog.
                # Then just import the previous answer from the dictLog
                
                if self.logDf.loc[self.logDf['iS'] == iS, 'UI'].values[0]:
                    QA = self.logDf.loc[self.logDf['iS'] == iS, 'UILog'].values[0]
                    if QA == 'Yes':
                        uiXY = self.log_UIxy[iS-1]
                    elif QA == 'No' or QA == 'No to all':
                        validFrame = False
                        #fig = plt.gcf()
                        #plt.close(fig)
                
                
                #### 3.5.2: Case when the UI has NOT been previously saved in the dictLog
                # Then ask for UI ; and save it in the dictLog
                elif not self.logDf.loc[self.logDf['iS'] == iS, 'UI'].values[0]:
                    if self.modeNoUIactivated == False:
                        # Display the image, plot beads positions and current trajectories & ask the question
                        self.listFrames[iF].show()
                        for iB in range(self.NB):
                            T = self.listTrajectories[iB]
                            ax = plt.gca()
                            T.plot(ax, iB)
                        
                        mngr = plt.get_current_fig_manager()
                        mngr.window.setGeometry(720, 50, 1175, 1000)
                        QA = pyautogui.confirm(
                            text='Can you point the beads of interest\nin the image ' + str(iS + 1) + '?',
                            title='', 
                            buttons=['No', 'Yes', 'Abort!', 'No to all'])
                        
                        # According to the question's answer:
                        if QA == 'Yes':
                            ui = plt.ginput(self.NB, timeout=0)
                            uiXY = ufun.ui2array(ui)
                            self.logDf.loc[self.logDf['iS'] == iS, 'UI'] = True
                            self.logDf.loc[self.logDf['iS'] == iS, 'UILog'] = QA
                            self.log_UIxy[iS-1] = uiXY
                        elif QA == 'No':
                            validFrame = False
                            self.logDf.loc[self.logDf['iS'] == iS, 'UI'] = True
                            self.logDf.loc[self.logDf['iS'] == iS, 'UILog'] = QA
                        elif QA == 'Abort!':
                            validFrame = False
                            fig = plt.gcf()
                            plt.close(fig)
                            return('Bug')
                        elif QA == 'No to all':
                            validFrame = False
                            self.modeNoUIactivated = True
                            self.logDf.loc[self.logDf['iS'] == iS, 'UI'] = True
                            self.logDf.loc[self.logDf['iS'] == iS, 'UILog'] = QA
                        fig = plt.gcf()
                        plt.close(fig)
                        
                    elif self.modeNoUIactivated == True:
                    # This mode is in case you don't want to keep clicking 'No' for hours when
                    # you know for a fact that there is nothing else you can do with this TimeLapse.
                        iS = self.listFrames[iF].iS
                        QA = 'No'
                        validFrame = False
                        self.logDf.loc[self.logDf['iS'] == iS, 'UI'] = True
                        self.logDf.loc[self.logDf['iS'] == iS, 'UILog'] = QA
                
                #### 3.5.3: Outcome of the user input case
                if not validFrame: # -> Next Frame
                    continue
            
                else:
                    # Double matching here
                    # First you match the user's click positions with the bead positions detected on frame iF
                    # You know then that you have identified the NB Beads of interest.
                    # Then another matching between these two new UIfound_BoiXY and the previous_BoiXY
                    # to be sure to attribute each position to the good trajectory !
                    
                    # First matching
                    M = ufun.compute_cost_matrix(uiXY,BXY)
                    row_ind, col_ind = linear_sum_assignment(M)
                    UIfound_BoiXY = np.array([BXY[iB] for iB in col_ind])
                    
                    # Second matching
                    M2 = ufun.compute_cost_matrix(previous_BoiXY, UIfound_BoiXY)
                    row_ind2, col_ind2 = linear_sum_assignment(M2)

                    
                    iBoi = [col_ind[i] for i in col_ind2]
                    BoiXY = np.array([BXY[iB] for iB in iBoi])

                    
            #### 3.6 Create the 'idxAnalysis' field
            #### >>> Exp type dependance here (02)
            # if 'compressions' in self.expType or 'constant field' in self.expType:
            #     # idxAnalysis = 0 if not in a ramp, and = number of ramp else. Basically increase by 1 each time you have an interval between two ramps.
                
            #     if self.expType == 'compressions':
            #         idxAnalysis = (self.listFrames[iF].idx_inNUp == 0) \
            #             * (max(self.listTrajectories[iB].dict['idxAnalysis']) \
            #                + 1*(self.listTrajectories[iB].dict['idxAnalysis'][-1] == 0))
                            
            #     elif self.expType == 'compressionsLowStart': 
            #     # a pre-ramp has the same idxAnalysis than a ramp but in negative.
            #         idxAnalysis = (self.listFrames[iF].idx_inNUp == 0) \
            #             * (max(self.listTrajectories[iB].dict['idxAnalysis']) + 1*(self.listTrajectories[iB].dict['idxAnalysis'][-1] <= 0)) \
            #                 - (self.listFrames[iF].idx_inNUp == 0.1) \
            #             * (abs(min(self.listTrajectories[iB].dict['idxAnalysis']) - 1*(self.listTrajectories[iB].dict['idxAnalysis'][-1] == 0)))
                        
            #     elif self.expType == 'constant field':
            #         idxAnalysis = 0
                        
            # #### TBC !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # elif 'sinus' in self.expType:
            #      idxAnalysis = 0
                 
            # elif 'brokenRamp' in self.expType:
            #      idxAnalysis = 0
            
            # elif 'optoGen' in self.expType:
            #     # idxFirstActivation = findFirstActivation(fieldDf)
            #     # if self.expType == 'optoGen_compressions':
            #     # #idxAnalysis = 0 if not in a ramp, and = number of ramp else. Basically increase by 1 each time you have an interval between two ramps.
            #     #     idxAnalysis = (self.listFrames[iF].idx_inNUp == 0) \
            #     #         * (max(self.listTrajectories[iB].dict['idxAnalysis']) \
            #     #            + (self.listTrajectories[iB].dict['idxAnalysis'][-1] == 0))
                            
            #     # elif self.expType == 'optoGen_compressionsLowStart': 
            #     # # a pre-ramp has the same idxAnalysis than a ramp but in negative.
            #     #     idxAnalysis = (self.listFrames[iF].idx_inNUp == 0) \
            #     #         * (max(self.listTrajectories[iB].dict['idxAnalysis']) + (self.listTrajectories[iB].dict['idxAnalysis'][-1] <= 0)) \
            #     #             - (self.listFrames[iF].idx_inNUp == 0.1) \
            #     #         * (abs(min(self.listTrajectories[iB].dict['idxAnalysis']) - (self.listTrajectories[iB].dict['idxAnalysis'][-1] == 0))) \
            #     #             + self.listTrajectories[iB].dict['idxAnalysis']
                        
            #     # elif self.expType == 'optoGen_constant field':
            #     idxAnalysis = 0
            
            
            #### 3.7 Append the different lists of listTrajectories[iB].dict
            for iB in range(self.NB):
               
                self.listTrajectories[iB].dict['Bead'].append(self.listFrames[iF].listBeads[iBoi[iB]])
                self.listTrajectories[iB].dict['iL'].append(self.listFrames[iF].iL)
                self.listTrajectories[iB].dict['iF'].append(iF)
                self.listTrajectories[iB].dict['iS'].append(self.listFrames[iF].iS)
                self.listTrajectories[iB].dict['iB_inFrame'].append(iBoi[iB])
                self.listTrajectories[iB].dict['X'].append(BoiXY[iB][0])
                self.listTrajectories[iB].dict['Y'].append(BoiXY[iB][1])
                self.listTrajectories[iB].dict['StdDev'].append(self.listFrames[iF].beadsStdDevarray()[iBoi[iB]])
                # self.listTrajectories[iB].dict['idx_inNUp'].append(self.listFrames[iF].idx_inNUp)
                # self.listTrajectories[iB].dict['idx_NUp'].append(self.listFrames[iF].idx_NUp)
                # self.listTrajectories[iB].dict['idxAnalysis'].append(idxAnalysis)
            

            #### 3.8 Initialize the next passage in the loop
            previous_iF = iF
            previous_iBoi = iBoi
            previous_BXY = BXY
            previous_BoiXY = BoiXY
            
            
            
            #### 3.9 End of the loop
            
                
        for iB in range(self.NB):
            for k in self.listTrajectories[iB].dict.keys():
                self.listTrajectories[iB].dict[k] = np.array(self.listTrajectories[iB].dict[k])
                
        
        #### 4. Refine the trajectories
        
        nT = len(self.listTrajectories[0].dict['Bead'])
        
        #### 4.1 Black Images deletion in the trajectory

        # Add the pointer to the correct line of the _Field.txt file.
        # It's just exactly the iS already saved in the dict, except if there are black images at the end of loops.
        # In that case you have to skip the X lines corresponding to the end of the ramp part, X being the nb of black images at the end of the current loop
        # This is because when black images occurs, they do because of the high frame rate during ramp parts and thus replace these last ramp images.

        # For now : excludedFrames_inward = excludedFrames_black
        # For now : excludedFrames_outward = excludedFrames_fluo # self.excludedFrames_outward[iLoop] +

        # Nct = (self.loop_mainSize-self.loop_rampSize)//2

        for iB in range(self.NB):
            self.listTrajectories[iB].dict['Zr'] = np.zeros(nT)
            self.listTrajectories[iB].nT = nT
            
            array_iF = np.array(self.listTrajectories[iB].dict['iF'])
            
            Series_iF = pd.Series(array_iF, name='iF')
            cols_to_merge = ['iField',  'idx_inNUp', 'idx_NUp', 'idxAnalysis']
            df2 = self.logDf[['iF'] + cols_to_merge]
            df_merge = pd.merge(left=Series_iF, right=df2, how='inner', on='iF')

            for col in cols_to_merge:
                array = df_merge[col].values
                # print(array.shape)
                self.listTrajectories[iB].dict[col] = array
            
        #     iField = []
        #     for i in range(nT):
        #         iF = self.listTrajectories[iB].dict['iF'][i]
        #         iLoop = ((iF)//self.loop_mainSize)
        #         try:
        #             offset = self.excludedFrames_black[iLoop]
        #         except:
        #             print(iF)
        #             print(iLoop)
        #             print(self.excludedFrames_black)

        #         i_lim = iLoop*self.loop_mainSize + (self.loop_mainSize - (Nct) - (offset))

        #         # i_lim is the first index after the end of the ramp
        #         addOffset = (iF >= i_lim) # Is this ramp going further than it should, considering the black images ?
                
                
        #         # 'optoGen' or 'compressions' but probably necessary in all cases actually
        #         if 'optoGen' in self.expType or 'compressions' in self.expType:
        #             # print(iLoop)
        #             SField = iF + int(addOffset*offset) + self.excludedFrames_outward[iLoop]
        #         else:
        #             SField = iF + int(addOffset*offset)
        #             # if i < 350:
        #             #     print(iF)
        #             #     # print(SField)
                
                
        #         iField.append(SField)
                
                
        #         # except:
        #         #     print(i, nT, offset)
        #         #     iF = self.listTrajectories[iB].dict['iF'][i]
        #         #     iLoop = ((iF+1)//self.loop_mainSize)
        #         #     print(iF, self.loop_mainSize, iLoop)
        #         #     [].concat()

        #     self.listTrajectories[iB].dict['iField'] = iField
        
            

        #### 4.2 Find the image with the best std within each n-uplet
            
        bestStd = self.findBestStd()
        for i in range(self.NB):
            self.listTrajectories[i].dict['bestStd'] = bestStd


    def importTrajectories(self, path, iB):
        """
        """
        self.listTrajectories.append(Trajectory(self.I, self.cellID, self.listFrames, self.scale, self.Zstep, iB))
        traj_df = pd.read_csv(path, sep = '\t')
        cols = traj_df.columns.values
        cols_to_remove = []
        for c in cols:
            if 'Unnamed' in c:
                cols_to_remove.append(c)
        traj_df = traj_df.drop(columns = cols_to_remove)
        self.listTrajectories[-1].dict = traj_df.to_dict(orient = 'list')
        for i in range(len(self.listTrajectories[-1].dict['iF'])):
            iBoi =  self.listTrajectories[-1].dict['iB_inFrame'][i]
            iF =  self.listTrajectories[-1].dict['iF'][i]
            self.listTrajectories[-1].dict['Bead'][i] = self.listFrames[iF].listBeads[iBoi]


    def computeForces(self, traj1, traj2, B0, D3, dx):
        """
        """

        # Magnetization functions
        def computeMag_M270(B):
            M = 0.74257*1.05*1600 * (0.001991*B**3 + 17.54*B**2 + 153.4*B) / (B**2 + 35.53*B + 158.1)
            return(M)

        def computeMag_M450(B):
            M = 1.05*1600 * (0.001991*B**3 + 17.54*B**2 + 153.4*B) / (B**2 + 35.53*B + 158.1)
            return(M)

        dictMag = {'M270' : computeMag_M270, 'M450' : computeMag_M450}
        dictBeadTypes = {2.7 : 'M270', 4.5 : 'M450'}

        dictLogF = {'D3' : [], 'B0' : [], 'Btot_L' : [], 'Btot_R' : [], 'F00' : [], 'F0' : [], 'dF_L' : [], 'dF_R' : [], 'Ftot' : []}

        # Correction functions
        def Bind_neighbour(B, D_BoI, neighbourType):
            if neighbourType == '' or neighbourType == 'nan':
                return(0)

            else:
                D_neighbour = self.dictBeadDiameters[neighbourType]
                f_Mag = dictMag[neighbourType] # Appropriate magnetization function
                M_neighbour = f_Mag(B) # magnetization [A.m^-1]
                V_neighbour = (4/3)*np.pi*(D_neighbour/2)**3 # volume [nm^3]
                m_neighbour = M_neighbour*V_neighbour*1e-9 # magnetic moment [A.nm^2]

                D_tot = (D_BoI + D_neighbour)/2 # Center-to-center distance [nm]
                B_ind = 2e5*m_neighbour/(D_tot**3) # Inducted mag field [mT]
                return(B_ind)

        def deltaF_neighbour(m_BoI, B, D_BoI, D_BoI2, neighbourType):
            if neighbourType == '' or neighbourType == 'nan':
                return(0)

            else:
                D_neighbour = self.dictBeadDiameters[neighbourType]
                f_Mag = dictMag[neighbourType] # Appropriate magnetization function
                M_neighbour = f_Mag(B) # magnetization [A.m^-1]
                V_neighbour = (4/3)*np.pi*(D_neighbour/2)**3 # volume [nm^3]
                m_neighbour = M_neighbour*V_neighbour*1e-9 # magnetic moment [A.nm^2]

                D_tot = D_BoI/2 + D_BoI2 + D_neighbour/2
                deltaF = 3e5*m_BoI*m_neighbour/D_tot**4 # force [pN]
                return(deltaF)

        # Let's make sure traj1 is the left bead traj and traj2 the right one.
        avgX1 = np.mean(traj1.dict['X'])
        avgX2 = np.mean(traj2.dict['X'])
        if avgX1 < avgX2:
            traj_L, traj_R = traj1, traj2
        else:
            traj_L, traj_R = traj2, traj1

        # Get useful data
        BeadType_L, BeadType_R = dictBeadTypes[traj_L.D], dictBeadTypes[traj_R.D]
        Neighbours_BL = np.concatenate(([traj_L.dict['Neighbour_L']], [traj_L.dict['Neighbour_R']]), axis = 0)
        Neighbours_BR = np.concatenate(([traj_R.dict['Neighbour_L']], [traj_R.dict['Neighbour_R']]), axis = 0)
        D_L, D_R = self.dictBeadDiameters[BeadType_L], self.dictBeadDiameters[BeadType_R]

        nT = len(B0)
        D3nm = 1000*D3
        Dxnm = 1000*dx
        F = np.zeros(nT)

        # Maybe possible to process that faster on lists themselves
        for i in range(nT):
            # Appropriate magnetization functions
            f_Mag_L = dictMag[BeadType_L]
            f_Mag_R = dictMag[BeadType_R]

            # Btot = B0 + B inducted by potential left neighbour mag + B inducted by potential right neighbour mag
            Btot_L = B0[i] + Bind_neighbour(B0[i], D_L, Neighbours_BL[0,i]) + Bind_neighbour(B0[i], D_L, Neighbours_BL[1,i])
            Btot_R = B0[i] + Bind_neighbour(B0[i], D_R, Neighbours_BR[0,i]) + Bind_neighbour(B0[i], D_R, Neighbours_BR[1,i])

            # Magnetizations
            M_L = f_Mag_L(Btot_L)
            M_R = f_Mag_R(Btot_R)

            # Volumes
            V_L = (4/3)*np.pi*(D_L/2)**3 # volume [nm^3]
            V_R = (4/3)*np.pi*(D_R/2)**3 # volume [nm^3]

            # Magnetizations
            m_L = M_L * 1e-9 * V_L
            m_R = M_R * 1e-9 * V_R

            anglefactor = abs(3*(Dxnm[i]/D3nm[i])**2 - 1)

            # Forces
            F00 = 3e5*anglefactor * (f_Mag_L(B0[i])* 1e-9*V_L) * (f_Mag_R(B0[i])*1e-9*V_R) / (D3nm[i]**4)
            
            F0 = 3e5*anglefactor*m_L*m_R/D3nm[i]**4
            dF_L = deltaF_neighbour(m_L, B0[i], D_L, D_R, Neighbours_BR[1,i])
            dF_R = deltaF_neighbour(m_R, B0[i], D_R, D_L, Neighbours_BL[0,i])

            # Total force = force between beads involved in the pair (F0)
            #               + small force between B_L and B_R's potential right neighbour
            #               + small force between B_R and B_L's potential left neighbour
            F[i] = F0 + dF_L + dF_R

            dictLogF['D3'].append(D3nm[i]-(D_L+D_R)/2)
            dictLogF['B0'].append(B0[i])
            dictLogF['Btot_L'].append(Btot_L)
            dictLogF['Btot_R'].append(Btot_R)
            dictLogF['F00'].append(F00)
            dictLogF['F0'].append(F0)
            dictLogF['dF_L'].append(dF_L)
            dictLogF['dF_R'].append(dF_R)
            dictLogF['Ftot'].append(F[i])

        dfLogF = pd.DataFrame(dictLogF)

        return(F, dfLogF)


# %%%% Frame

class Frame:
    def __init__(self, F, iL, iS, NB, Nup, idx_inNUp, idx_NUp, scale, resDf):
        ny, nx = F.shape[0], F.shape[1]
        self.F = F # Note : Frame.F points directly to the i-th frame of the image I ! To have 2 different versions one should use np.copy(F)
        self.NBoi = NB
        self.NBdetected = 0
        self.nx = nx
        self.ny = ny
        self.iL = iL
        self.iS = iS
        self.listBeads = []
        self.trajPoint = []
        self.Nuplet = Nup
        self.idx_inNUp = idx_inNUp
        self.idx_NUp = idx_NUp
        self.scale = scale
        self.resDf = resDf

    def __str__(self):
        text = 'a'
        return(text)

    def show(self, strech = True):
        fig, ax = plt.subplots(1,1)
#         fig_size = plt.gcf().get_size_inches()
#         fig.set_size_inches(2 * fig_size)
        if strech:
            pStart, pStop = np.percentile(self.F, (1, 99))
            ax.imshow(self.F, cmap = 'gray', vmin = pStart, vmax = pStop)
        else:
            ax.imshow(self.F, cmap = 'gray')
        if len(self.listBeads) > 0:
            for B in self.listBeads:
                ax.plot([B.x], [B.y], c='orange', marker='+', markersize = 15)
        fig.show()

    def makeListBeads(self):
        self.NBdetected = self.resDf.shape[0]
        for i in range(self.NBdetected):
            d = {}
            for c in self.resDf.columns:
                d[c] = self.resDf[c].values[i]
            self.listBeads.append(Bead(d, self.F))

    def beadsXYarray(self):
        A = np.zeros((len(self.listBeads), 2))
        for i in range(len(self.listBeads)):
            b = self.listBeads[i]
            A[i,0], A[i,1] = b.x, b.y
        return(A)

    def beadsStdDevarray(self):
        A = np.zeros(len(self.listBeads))
        for i in range(len(self.listBeads)):
            b = self.listBeads[i]
            A[i] = b.std
        return(A)




# %%%% Bead

class Bead:
    def __init__(self, d, F):
        self.x = d['XM']
        self.y = d['YM']
        self.D = 0
        self.area = d['Area']
        self.std = d['StdDev']
        self.iS = d['Slice']-1
        self.idx_inNUp = ''
        self.Neighbour_L = ''
        self.Neighbour_R = ''
        self.F = F


    def show(self, strech = True):
        fig, ax = plt.subplots(1,1)
        if strech:
            pStart, pStop = np.percentile(self.F, (1, 99))
            ax.imshow(self.F, cmap = 'gray', vmin = pStart, vmax = pStop)
        else:
            ax.imshow(self.F, cmap = 'gray')
        ax.plot([self.x], [self.y], c='orange', marker='o')
        fig.show()

#

# %%%% Trajectory

class Trajectory:
    def __init__(self, I, cellID, listFrames, scale, Zstep, iB):
        nS, ny, nx = I.shape[0], I.shape[1], I.shape[2]
        self.I = I
        self.cellID = cellID
        self.listFrames = listFrames
        self.scale = scale
        self.nx = nx
        self.ny = ny
        self.nS = nS
        self.D = 0
        self.nT = 0
        self.iB = iB
        self.dict = {'X': [],'Y': [],'idxAnalysis': [],'StdDev': [],
                     'iL': [],'Bead': [],'idx_inNUp': [],'idx_NUp': [],'iF': [],'iS': [],'iB_inFrame' : [], 
                     'bestStd' : [], 'Zr' : [], 'Neighbour_L' : [], 'Neighbour_R' : []}
        # iF is the index in the listFrames
        # iS is the index of the slice in the raw image MINUS ONE
        self.beadInOut = ''
        self.deptho = []
        self.depthoPath = ''
        self.depthoStep = 20
        self.depthoZFocus = 200
        self.Zstep = Zstep # The step in microns between 2 consecutive frames in a multi-frame Nuplet
        
        #### Z detection settings here
        self.HDZfactor = 5
        self.maxDz_triplets = 60 # Max Dz allowed between images
        self.maxDz_singlets = 30
        self.HWScan_triplets = 1200 # Half width of the scans
        self.HWScan_singlets = 600
        
        
    def __str__(self):
        text = 'iS : ' + str(self.series_iS)
        text += '\n'
        text += 'XY : ' + str(self.seriesXY)
        return(text)

    def save(self, path):
        df = pd.DataFrame(self.dict)
        df.to_csv(path, sep = '\t', index = False)

    def computeZ(self, matchingDirection, plot = 0):
        

        if len(self.deptho) == 0:
            return('Error, no depthograph associated with this trajectory')

        else:
            Ddz, Ddx = self.deptho.shape[0], self.deptho.shape[1]
            iF = self.dict['iF'][0]
            previousZ = -1
            
            
            # ###################################################################
            
            while iF <= max(self.dict['iF']):
                
            #### Enable plots of Z detection  here
                
                plot = 0
                # if (iF >= 0 and iF <= 30) or (iF > 178 and iF <= 208):
                #     plot = 1

            # ###################################################################

                if iF not in self.dict['iF']: # this index isn't in the trajectory list => the frame was removed for some reason.
                    iF += 1 # Let's just go to the next index

                else:
                    F = self.listFrames[iF]
                    Nup = F.Nuplet
                    if Nup <= 1:
                        framesNuplet = [F]
                        iFNuplet = [iF]
                        iF += 1
                    elif Nup > 1:
                        framesNuplet = [F]
                        iFNuplet = [iF]
                        jF = 1
                        while iF+jF <= max(self.dict['iF']) and self.listFrames[iF+jF].idx_NUp == F.idx_NUp:
                            if iF+jF in self.dict['iF']: # One of the images of the triplet may be invalid,
                                # and we don't want to take it. With this test we won't
                                nextF = self.listFrames[iF+jF]
                                framesNuplet.append(nextF)
                                iFNuplet.append(iF+jF)
                            jF += 1
                            
                        iF += jF


                    Z = self.findZ_Nuplet(framesNuplet, iFNuplet, Nup, previousZ, 
                                          matchingDirection, plot)
                        
                        
                    previousZ = Z
                    # This Z_pix has no meaning in itself, it needs to be compared to the depthograph Z reference point,
                    # which is depthoZFocus.

                    Zr = self.depthoZFocus - Z # If you want to find it back, Z = depthoZFocus - Zr
                    # This definition was chosen so that when Zr > 0, the plane of observation of the bead is HIGHER than the focus
                    # and accordingly when Zr < 0, the plane of observation of the bead is LOWER than the focus

                    mask = np.array([(iF in iFNuplet) for iF in self.dict['iF']])
                    self.dict['Zr'][mask] = Zr
                
                

    def findZ_Nuplet(self, framesNuplet, iFNuplet, Nup, previousZ, 
                     matchingDirection, plot = False):
        # try:
        Nframes = len(framesNuplet)
        listStatus_1 = [F.idx_inNUp for F in framesNuplet]
        listXY = [[self.dict['X'][np.where(self.dict['iF']==iF)][0],
                   self.dict['Y'][np.where(self.dict['iF']==iF)][0]] for iF in iFNuplet]
        listiS = [self.dict['iS'][np.where(self.dict['iF']==iF)][0] for iF in iFNuplet]
        cleanSize = ufun.getDepthoCleanSize(self.D, self.scale)
        hdSize = self.deptho.shape[1]
        depthoDepth = self.deptho.shape[0]
        listProfiles = np.zeros((Nframes, hdSize))
        listROI = []
        listWholeROI = []
        for i in range(Nframes):
            if np.sum(framesNuplet[i].F) == 0:
                print('illegal')
            xx = np.arange(0, 5)
            yy = np.arange(0, cleanSize)
            try:
                X, Y = int(np.round(listXY[i][0])), int(np.round(listXY[i][1])) # > We could also try to recenter the image to keep a subpixel resolution here
                # line that is 5 pixels wide
                wholeROI = framesNuplet[i].F[Y-cleanSize//2:Y+cleanSize//2+1, X-cleanSize//2:X+cleanSize//2+1]
                profileROI = framesNuplet[i].F[Y-cleanSize//2:Y+cleanSize//2+1, X-2:X+3]
                f = interpolate.interp2d(xx, yy, profileROI, kind='cubic')
                # Now use the obtained interpolation function and plot the result:
                xxnew = xx
                yynew = np.linspace(0, cleanSize, hdSize)
                profileROI_hd = f(xxnew, yynew)

            except: # If the vertical slice doesn't work, try the horizontal one
                print(gs.ORANGE + 'error with the vertical slice -> trying with horizontal one')
                print('iFNuplet')
                print(iFNuplet)
                print('Roi')
                print(Y-2,Y+3, X-cleanSize//2,X+cleanSize//2+1)
                print('' + gs.NORMAL)

                xx, yy = yy, xx
                X, Y = int(np.round(listXY[i][0])), int(np.round(listXY[i][1])) # > We could also try to recenter the image to keep a subpixel resolution here
                # line that is 5 pixels wide
                wholeROI = framesNuplet[i].F[Y-cleanSize//2:Y+cleanSize//2+1, X-cleanSize//2:X+cleanSize//2+1]
                profileROI = framesNuplet[i].F[Y-2:Y+3, X-cleanSize//2:X+cleanSize//2+1]
                f = interpolate.interp2d(xx, yy, profileROI, kind='cubic')
                # Now use the obtained interpolation function and plot the result:
                xxnew = np.linspace(0, cleanSize, hdSize)
                yynew = yy
                profileROI_hd = f(xxnew, yynew).T

            listROI.append(profileROI)
            listWholeROI.append(wholeROI)

            listProfiles[i,:] = profileROI_hd[:,5//2] * (1/5)
            for j in range(1, 1 + 5//2):
                listProfiles[i,:] += profileROI_hd[:,5//2-j] * (1/5)
                listProfiles[i,:] += profileROI_hd[:,5//2+j] * (1/5)

        listProfiles = listProfiles.astype(np.uint16)



        # now use listStatus_1, listProfiles, self.deptho + data about the jump between Nuplets ! (TBA)
        # to compute the correlation function
        nVoxels = int(np.round(int(self.Zstep)/self.depthoStep))
        
        if previousZ == -1:
            Ztop = 0
            Zbot = depthoDepth
        
        elif Nup > 1:
            HW = self.HWScan_triplets
            halfScannedDepth_raw = int(HW / self.depthoStep)
            Ztop = max(0, previousZ - halfScannedDepth_raw) 
            Zbot = min(depthoDepth, previousZ + halfScannedDepth_raw)
            
        elif Nup == 1:
            HW = self.HWScan_singlets
            halfScannedDepth_raw = int(HW / self.depthoStep) 
            Ztop = max(0, previousZ - halfScannedDepth_raw) 
            Zbot = min(depthoDepth, previousZ + halfScannedDepth_raw)

        scannedDepth = Zbot - Ztop
        # print(Nup, depthoDepth, Ztop, Zbot, scannedDepth)
        
        listDistances = np.zeros((Nframes, scannedDepth))
        listZ = np.zeros(Nframes, dtype = int)
        Zscanned = np.arange(Ztop, Zbot, 1, dtype=int)
        
        subDeptho = self.deptho[Ztop:Zbot, :]
        
        for i in range(Nframes):
            
            listDistances[i] = ufun.squareDistance(subDeptho, listProfiles[i], normalize = True) # Utility functions
            listZ[i] = Ztop + np.argmin(listDistances[i])

        # Translate the profiles that must be translated (idx_inNUp 1 & 3 if Nup = 3)
        # and don't move the others (idx_inNUp 2 if Nup = 3 or the 1 profile when Nup = 1)
        if Nup > 1:
            finalDists = ufun.matchDists(listDistances, listStatus_1, Nup, 
                                        nVoxels, direction = matchingDirection)
        elif Nup == 1:
            finalDists = listDistances

        sumFinalD = np.sum(finalDists, axis = 0)


        #### Tweak this part to force the Z-detection to a specific range to prevent abnormal jumps
        if previousZ == -1: # First image => No restriction
            Z = np.argmin(sumFinalD)
            maxDz = 0
            
        else: # Not first image => Restriction
            if Nup > 1 and previousZ != -1: # Not first image AND Triplets => Restriction Triplets
                maxDz = self.maxDz_triplets
            elif Nup == 1 and previousZ != -1: # Not first image AND singlet => Restriction Singlet
                maxDz = self.maxDz_singlets
                
            limInf = max(previousZ - maxDz, 0) - Ztop
            limSup = min(previousZ + maxDz, depthoDepth) - Ztop
            Z = Ztop + limInf + np.argmin(sumFinalD[limInf:limSup])



        #### Fit quality
        # Ddz, Ddx = depthoHD.shape[0], depthoHD.shape[1]
        # print(Ddz, Ddx)
        # dz_fitPoly = int(Ddz/32)
        
        # def f_sq(x, k):
        #     return(k * x**2)
        
        # listDistances = np.zeros((Ddz, Ddz))
        # listZQuality = np.zeros(Ddz)
        
        # for z in range(Ddz):
            
        #     profile_z = depthoHD[z, :]
        #     listDistances[z] = ufun.squareDistance(depthoHD, profile_z, normalize = True) # Utility functions
        #     z_start = max(0, z - dz_fitPoly)
        #     z_stop = min(Ddz - 1, z + dz_fitPoly)
        #     # print(z, z_start, z_stop)
        #     Cost_fitPoly = listDistances[z][z_start : z_stop + 1]
        #     X_fitPoly = np.arange(z_start - z, z_stop - z + 1, dtype=int)
        #     popt, pcov = curve_fit(f_sq, X_fitPoly, Cost_fitPoly - listDistances[z][z], 
        #                            p0=[1], bounds=(-np.inf, np.inf))
        #     z_quality = popt[0]*1e3
        #     listZQuality[z] = z_quality
        
        # Z = np.array([i for i in range(Ddz)]) - depthoZFocusHD
        # plt.plot(Z, listZQuality)


        #### Important plotting option here
        if plot >= 1:
            plt.ioff()
            fig, axes = plt.subplots(5, 3, figsize = (16,16))
            
            cmap = 'magma'
            color_image = 'cyan'
            color_Nup = ['gold', 'darkorange', 'red']
            color_result = 'darkgreen'
            color_previousResult = 'turquoise'
            color_margin = 'aquamarine'
            
            im = framesNuplet[0].F
            X2, Y2 = listXY[0][0], listXY[0][1]
            
            deptho_zticks_list = np.arange(0, depthoDepth, 50*self.HDZfactor, dtype = int)
            deptho_zticks_loc = ticker.FixedLocator(deptho_zticks_list)
            deptho_zticks_format = ticker.FixedFormatter((deptho_zticks_list/self.HDZfactor).astype(int))

            
            if Nup == 1:
                direction = 'Single Image'
            else:
                direction = matchingDirection

            pStart, pStop = np.percentile(im, (1, 99))
            axes[0,0].imshow(im, vmin = pStart, vmax = 1.5*pStop, cmap = 'gray')
            images_ticks_loc = ticker.MultipleLocator(50)
            axes[0,0].xaxis.set_major_locator(images_ticks_loc)
            axes[0,0].yaxis.set_major_locator(images_ticks_loc)
            
            
            dx, dy = 50, 50
            axes[0,0].plot([X2], [Y2], marker = '+', c = 'red')
            axes[0,0].plot([X2-dx,X2-dx], [Y2-dy,Y2+dy], ls = '--', c = color_image, lw = 0.8)
            axes[0,0].plot([X2+dx,X2+dx], [Y2-dy,Y2+dy], ls = '--', c = color_image, lw = 0.8)
            axes[0,0].plot([X2-dx,X2+dx], [Y2-dy,Y2-dy], ls = '--', c = color_image, lw = 0.8)
            axes[0,0].plot([X2-dx,X2+dx], [Y2+dy,Y2+dy], ls = '--', c = color_image, lw = 0.8)

            # Plot the deptho then resize it better
            axes[0,1].imshow(self.deptho, cmap = cmap)
            XL0, YL0 = axes[0,1].get_xlim(), axes[0,1].get_ylim()
            extent = (XL0[0], YL0[0]*(5/3), YL0[0], YL0[1])
            axes[0,1].imshow(self.deptho, extent = extent, cmap = cmap)
            
            axes[0,1].yaxis.set_major_locator(deptho_zticks_loc)
            axes[0,1].yaxis.set_major_formatter(deptho_zticks_format)
            
            pixLineHD = np.arange(0, hdSize, 1)
            zPos = Zscanned
            
            
            for i in range(Nframes):
                idx_inNUp = int(framesNuplet[i].idx_inNUp)
                idx_inNUp += (idx_inNUp == 0)
                
                # Show the bead appearence
                axes[1,i].imshow(listWholeROI[i], cmap = cmap)
                images_ticks_loc = ticker.MultipleLocator(10)
                axes[1,i].xaxis.set_major_locator(images_ticks_loc)
                axes[1,i].yaxis.set_major_locator(images_ticks_loc)
                axes[1,i].set_title('Image {:.0f}/{:.0f} - '.format(idx_inNUp, Nup) + direction, 
                                    fontsize = 14)
                axes[1,i].plot([cleanSize//2,cleanSize//2],[0,cleanSize-1], c=color_Nup[i], ls='--', lw = 1)
                
                # Show the profile of the beads
                axes[2,i].plot(pixLineHD, listProfiles[i], c = color_Nup[i])
                axes[2,i].set_xlabel('Position along the profile\n(Y-axis)', 
                                     fontsize = 9)
                axes[2,i].set_ylabel('Pixel intensity', 
                                     fontsize = 9)
                axes[2,i].set_title('Profile {:.0f}/{:.0f} - '.format(idx_inNUp, Nup), 
                                    fontsize = 11)
                
                # Show the distance map to the deptho
                listDistances = np.array(listDistances)
                # inversed_listDistances = (listDistances[i] * (-1)) + np.max(listDistances[i])
                # peaks, peaks_prop = signal.find_peaks(inversed_listDistances, distance = self.HDZfactor * 20)
                axes[3,i].plot(zPos, listDistances[i])
                # axes[3,i].plot(zPos, inversed_listDistances, ls='--', lw=0.75, c='k')
                axes[3,i].xaxis.set_major_locator(deptho_zticks_loc)
                axes[3,i].xaxis.set_major_formatter(deptho_zticks_format)
                axes[3,i].set_xlabel('Position along the depthograph\n(Z-axis)', 
                                     fontsize = 9)
                axes[3,i].set_ylabel('Cost\n(Squared diff to deptho)', 
                                     fontsize = 9)
                axes[3,i].set_title('Cost curve {:.0f}/{:.0f}'.format(idx_inNUp, Nup), 
                                    fontsize = 11)
                
                limy3 = axes[3,i].get_ylim()
                min_i = zPos[np.argmin(listDistances[i])]
                axes[3,i].plot([min_i, min_i], limy3, ls = '--', c = color_Nup[i])
                # for p in peaks:
                #     p_i = zPos[int(p)]
                #     axes[3,i].plot([p_i], [np.mean(limy3)], ls = '',
                #                   marker = 'v',  c = 'orange', mec = 'k', markersize = 8)
                #     axes[3,i].text(p_i, np.mean(limy3)*1.1, str(p_i/self.HDZfactor), c = 'k')
                axes[3,i].set_xlim([0, depthoDepth])
                
                #
                axes[4,i].plot(zPos, finalDists[i])
                axes[4,i].xaxis.set_major_locator(deptho_zticks_loc)
                axes[4,i].xaxis.set_major_formatter(deptho_zticks_format)
                axes[4,i].set_xlabel('Corrected position along the depthograph\n(Z-axis)', 
                                     fontsize = 9)
                axes[4,i].set_ylabel('Cost\n(Squared diff to deptho)', 
                                     fontsize = 9)
                axes[4,i].set_title('Cost curve with corrected position {:.0f}/{:.0f}'.format(idx_inNUp, Nup), 
                                    fontsize = 11)
                
                limy4 = axes[4,i].get_ylim()
                min_i = zPos[np.argmin(finalDists[i])]
                axes[4,i].plot([min_i, min_i], limy4, ls = '--', c = color_Nup[i])
                # axes[4,i].text(min_i+5, np.mean(limy4), str(min_i/self.HDZfactor), c = 'k')
                axes[4,i].set_xlim([0, depthoDepth])

                axes[0,1].plot([axes[0,1].get_xlim()[0], axes[0,1].get_xlim()[1]-1], 
                               [listZ[i], listZ[i]], 
                               ls = '--', c = color_Nup[i])
                
                axes[0,1].plot([axes[0,1].get_xlim()[0], axes[0,1].get_xlim()[1]-1], 
                               [Z,Z], 
                               ls = '--', c = color_result)


            axes[0,2].plot(zPos, sumFinalD)
            axes[0,2].xaxis.set_major_locator(deptho_zticks_loc)
            axes[0,2].xaxis.set_major_formatter(deptho_zticks_format)
            limy0 = axes[0,2].get_ylim()
            axes[0,2].plot([Z, Z], limy0, ls = '-', c = color_result, label = 'Z', lw = 1.5)
            axes[0,2].plot([previousZ, previousZ], limy0, 
                           ls = '--', c = color_previousResult, label = 'previous Z', lw = 0.8)
            axes[0,2].plot([previousZ-maxDz, previousZ-maxDz], limy0,
                           ls = '--', c = color_margin, label = 'allowed margin', lw = 0.8)
            axes[0,2].plot([previousZ+maxDz, previousZ+maxDz], limy0,
                           ls = '--', c = color_margin, lw = 0.8)
            axes[0,2].set_xlim([0, depthoDepth])
            
            axes[0,2].set_xlabel('Position along the depthograph\n(Z-axis)', 
                                 fontsize = 9)
            axes[0,2].set_ylabel('Total Cost\n(Sum of Squared diff to deptho)', 
                                 fontsize = 9)
            axes[0,2].set_title('Sum of Cost curves with corrected position', 
                                fontsize = 11)
            axes[0,2].legend()
            
            for ax in axes.flatten():
                ax.tick_params(axis='x', labelsize=9)
                ax.tick_params(axis='y', labelsize=9)
            
            Nfig = plt.gcf().number
            iSNuplet = [F.iS for F in framesNuplet]
            
            fig.tight_layout()
            fig.subplots_adjust(top=0.94)
            
            fig.suptitle('Frames '+str(iFNuplet)+' - Slices '+str(iSNuplet)+' ; '+\
                         'Z = {:.1f} slices = '.format(Z/self.HDZfactor) + \
                         '{:.4f} µm'.format(Z*(self.depthoStep/1000)),
                         y=0.98)
            
            if not os.path.isdir(cp.DirTempPlots):
                os.mkdir(cp.DirTempPlots)
                
            thisCellTempPlots = os.path.join(cp.DirTempPlots, self.cellID)
            if not os.path.isdir(thisCellTempPlots):
                os.mkdir(thisCellTempPlots)
            
            saveName = 'ZCheckPlot_S{:.0f}_B{:.0f}.png'.format(iSNuplet[0], self.iB+1)
            savePath = os.path.join(thisCellTempPlots, saveName)
            fig.savefig(savePath)
            plt.close(fig)
        
        plt.ion()

        return(Z)

        # except Exception:
        #     print(gs.RED + '')
        #     traceback.print_exc()
        #     print('\n')
        #     print(gs.ORANGE + 'Error with the Z detection')
        #     print('iFNuplet')
        #     print(iFNuplet)
        #     print('Roi')
        #     print(Y-2,Y+3, X-cleanSize//2,X+cleanSize//2+1)
        #     print('Deptho shape')
        #     print(self.deptho.shape)
        #     print('Shapes of listDistances, finalDists, sumFinalD')
        #     print(listDistances.shape)
        #     print(finalDists.shape)
        #     print(sumFinalD.shape)
        #     print('previousZ, previousZ-maxDz, previousZ+maxDz')
        #     print(previousZ, previousZ-maxDz, previousZ+maxDz)
        #     print('' + gs.NORMAL)


    def keepBestStdOnly(self):
        dictBestStd = {}
        bestStd = self.dict['bestStd']
        nT = int(np.sum(bestStd))
        for k in self.dict.keys():
            A = np.array(self.dict[k])
            dictBestStd[k] = A[bestStd]
        self.dict = dictBestStd
        self.nT = nT


    def detectNeighbours_ui(self, Nimg, frequency, beadType): # NOT VERY WELL MADE FOR NOW
        # Plots to help the user to see the neighbour of each bead
        ncols = 4
        nrows = ((Nimg-1) // ncols) + 1
        fig, ax = plt.subplots(nrows, ncols)
        for i in range(Nimg):
            try:
                pos = np.searchsorted(self.dict['iS'], i*frequency, 'left')
                iS = self.dict['iS'][pos]
                iF = self.dict['iF'][pos]
                pStart, pStop = np.percentile(self.I[iS-1], (1, 99))
                if nrows > 1:
                    ax[i//ncols,i%ncols].imshow(self.I[iS-1], cmap = 'gray', vmin = pStart, vmax = pStop)
                    ax[i//ncols,i%ncols].set_title('Loop ' + str(i+1))
                    ax[i//ncols,i%ncols].plot([self.dict['X'][pos]],[self.dict['Y'][pos]], 'ro')
                elif nrows == 1:
                    ax[i].imshow(self.I[iS-1], cmap = 'gray', vmin = pStart, vmax = pStop)
                    ax[i].set_title('Loop ' + str(i+1))
                    ax[i].plot([self.dict['X'][pos]],[self.dict['Y'][pos]], 'ro')
            except:
                print(gs.RED  + 'ptit probleme dans le detectNeighbours_ui' + gs.NORMAL)
                
        plt.show()
        # Ask the question
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(720, 50, 1175, 1000)
        QA = pyautogui.confirm(
            text='Neighbours of the selected bead?',
            title='',
            buttons=['1', '2'])


        # According to the question's answer:
        if QA == '1':
            if self.iB%2 == 0: # the bead is on the left of a pair
                Neighbour_L, Neighbour_R = '', beadType
            elif self.iB%2 == 1: # the bead is on the right of a pair
                Neighbour_L, Neighbour_R = beadType, ''
        elif QA == '2':
            Neighbour_L, Neighbour_R = beadType, beadType

        plt.close(fig)
        listNeighbours = []

        for i in range(len(self.dict['iF'])):
            self.dict['Bead'][i].Neighbour_L = Neighbour_L
            self.dict['Bead'][i].Neighbour_R = Neighbour_R
            listNeighbours.append([Neighbour_L, Neighbour_R])

        arrayNeighbours = np.array(listNeighbours)
        self.dict['Neighbour_L'] = arrayNeighbours[:,0]
        self.dict['Neighbour_R'] = arrayNeighbours[:,1]


    def detectInOut_ui(self, Nimg, frequency): # NOT VERY WELL MADE FOR NOW
        # Almost copy paste of detectNeighbours_ui
        ncols = 4
        nrows = ((Nimg-1) // ncols) + 1

        fig, ax = plt.subplots(nrows, ncols)
        for i in range(Nimg):
            try:
                pos = np.searchsorted(self.dict['iS'], i*frequency, 'left')
                iS = self.dict['iS'][pos]
                iF = self.dict['iF'][pos]
                pStart, pStop = np.percentile(self.I[iS-1], (1, 99))
                if nrows > 1:
                    ax[i//ncols,i%ncols].imshow(self.I[iS-1], cmap = 'gray', vmin = pStart, vmax = pStop)
                    ax[i//ncols,i%ncols].set_title('Loop ' + str(i+1))
                    ax[i//ncols,i%ncols].plot([self.dict['X'][pos]],[self.dict['Y'][pos]], 'ro')
                elif nrows == 1:
                    ax[i].imshow(self.I[iS-1], cmap = 'gray', vmin = pStart, vmax = pStop)
                    ax[i].set_title('Loop ' + str(i+1))
                    ax[i].plot([self.dict['X'][pos]],[self.dict['Y'][pos]], 'ro')
            except:
                print(gs.RED  + 'error in detectInOut_ui' + gs.NORMAL)
        
        plt.show()
        # Ask the question
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(720, 50, 1175, 1000)
        QA = pyautogui.confirm(
            text='Is it an inside or outside bead?',
            title='',
            buttons=['In', 'Out'])

        self.beadInOut = QA
        plt.close(fig)
        return(QA)



    def plot(self, ax, i_color):
        colors = ['cyan', 'red', 'blue', 'orange']
        c = colors[i_color]
        ax.plot(self.dict['X'], self.dict['Y'], color=c, lw=0.5)

# %%%% Main

def mainTracker_V2(dates, manips, wells, cells, depthoNames, expDf, NB = 2,
                sourceField = 'default', redoAllSteps = False, trackAll = False,
                DirData = cp.DirData, 
                DirDataRaw = cp.DirDataRaw, 
                DirDataRawDeptho = cp.DirDataRawDeptho, 
                DirDataTimeseries = cp.DirDataTimeseries,
                CloudSaving = cp.CloudSaving,
                DirCloudTimeseries = cp.DirCloudTimeseries):
    
    start = time.time()

    #### 0. Initialization
    #### 0.1 - Make list of files to analyse

    fileRoots = []
    tifImagesPaths = []
    txtFieldPaths = []
    txtStatusPaths = []
    txtResultsPaths = []
    if not isinstance(dates, str):
        rawDirList = [os.path.join(DirDataRaw, d) for d in dates]
    else:
        rawDirList = [os.path.join(DirDataRaw, dates)]

        
    for rd in rawDirList:
        fileList = os.listdir(rd)
        for f in fileList:
            if ufun.isFileOfInterest(f, manips, wells, cells, mode = 'soft', suffix = '_Results.txt'): # See Utility Functions > isFileOfInterest
                validFileGroup = False
                f_Res = f # A result file was found
                f_root = f_Res[:-12]
                f_root_simple = ufun.simplifyCellId(f_root) # The common 'root' to the .tif and _Field.txt files.
                # With a call to ufun.simplifyCellId(), 'M1_P1_C2-1' become 'M1_P1_C2'
                # This is useful cause like that, the image and field files 
                # do not have to be duplicated if there are 2 results files.
                
                test_image = os.path.isfile(os.path.join(rd, f_root_simple + '.tif'))
                test_field = os.path.isfile(os.path.join(rd, f_root_simple + '_Field.txt'))
                test_status = os.path.isfile(os.path.join(rd, f_root_simple + '_Status.txt'))
                if test_image and test_field and test_status:
                    f_Tif = f_root_simple + '.tif'
                    f_Field = f_root_simple + '_Field.txt'
                    f_Status = f_root_simple + '_Status.txt'
                    fileRoots.append(f_root)
                    tifImagesPaths.append(os.path.join(rd, f_Tif))
                    txtFieldPaths.append(os.path.join(rd, f_Field))
                    txtStatusPaths.append(os.path.join(rd, f_Status))
                    txtResultsPaths.append(os.path.join(rd, f_Res))
                    validFileGroup = True
                
                else: # Retry in case there was a duplicated image
                # No call to the function that simplifies the names
                    test_image = os.path.isfile(os.path.join(rd, f_root + '.tif'))
                    test_field = os.path.isfile(os.path.join(rd, f_root + '_Field.txt'))
                    test_status = os.path.isfile(os.path.join(rd, f_root + '_Status.txt'))
                    if test_image and test_field and test_status:
                        f_Tif = f_root + '.tif'
                        f_Field = f_root + '_Field.txt'
                        f_Status = f_root + '_Status.txt'
                        fileRoots.append(f_root)
                        tifImagesPaths.append(os.path.join(rd, f_Tif))
                        txtFieldPaths.append(os.path.join(rd, f_Field))
                        txtStatusPaths.append(os.path.join(rd, f_Status))
                        txtResultsPaths.append(os.path.join(rd, f_Res))
                        validFileGroup = True
                
                if not validFileGroup:
                    print(gs.RED + 'Bizarre! ' + f_Res + ' seems to be a lonely Results.txt file!' + gs.NORMAL)
    
    #### 0.2 - Begining of the Main Loop (i)
    for i in range(len(fileRoots)):
        f = fileRoots[i]
        imagePath, fieldPath, resPath, statusPath = tifImagesPaths[i], txtFieldPaths[i], txtResultsPaths[i], txtStatusPaths[i]
        manipID = ufun.findInfosInFileName(f, 'manipID') # See Utility Functions > findInfosInFileName
        cellID = ufun.findInfosInFileName(f, 'cellID') # See Utility Functions > findInfosInFileName

        print('\n')
        print(gs.BLUE + 'Analysis of file {:.0f}/{:.0f} : {}'.format(i+1, len(fileRoots), f))
        print('Loading image and experimental data...' + gs.NORMAL)

        #### 0.3 - Load exp data (manipDict)
        if manipID not in expDf['manipID'].values:
            print(gs.RED + 'Error! No experimental data found for: ' + manipID + gs.NORMAL)
            break
        else:
            expDf_line = expDf.loc[expDf['manipID'] == manipID]
            manipDict = {}
            for c in expDf_line.columns.values:
                manipDict[c] = expDf_line[c].values[0]

        #### 0.4 - Load image and init PTL
        I = io.imread(imagePath) # Approx 0.5s per image
        PTL = PincherTimeLapse(I, cellID, manipDict, NB)
    
        #### 0.5 - Load field file (fieldDf)
        if sourceField == 'default':
            fieldCols = ['B_set', 'T_abs', 'B', 'Z']
            fieldDf = pd.read_csv(fieldPath, sep = '\t', names = fieldCols) # '\t'
        elif sourceField == 'fastImagingVI':
            fieldCols = ['B_set', 'B', 'T_abs']
            fieldDf = pd.read_csv(fieldPath, sep = '\t', names = fieldCols) # '\t'
            
        #### 0.6 - Load results file (PTL.resultsDf)
        resultsDf = pd.read_csv(resPath, sep='\t')
        for c in resultsDf.columns:
            if 'Unnamed' in c:
                resultsDf.drop([c], axis = 1, inplace=True)
        PTL.resultsDf = resultsDf
            
        #### 0.7 - Make the log table (PTL.logDf)
        logFilePath = resPath[:-12] + '_LogPY.txt'
        logFileImported = False
        if redoAllSteps:
            pass
        elif os.path.isfile(logFilePath):
            PTL.importLogDf(logFilePath)
            logFileImported = True
        else:
            pass
        
        if not logFileImported:
            PTL.initializeLogDf(statusPath)
        
        ## 0.51 Find index of first activation
        # Anumita's stuff
        # try:
        #     optoMetaPath = f_Res[:-12] + '_OptoMetadata.txt'
        #     PTL.makeOptoMetadata(fieldDf, display = 1, save = True, path = optoMetaPath)
        # except:
        #     pass

        print(gs.BLUE + 'OK!')
        print(gs.BLUE + 'Pretreating the image...' + gs.NORMAL)
        
        #### 0.8 - Detect fluo & black images
        
        current_date = ufun.findInfosInFileName(f, 'date')
        current_date = current_date.replace("-", ".")
        fluoDirPath = os.path.join(DirDataRaw, current_date + '_Fluo', f)
        
        PTL.detectNullFrames()
        PTL.detectFluoFrames(save = True, fluoDirPath = fluoDirPath, f = f)
        
        ## 0.8 - Detect fluo & black images
        # current_date = ufun.findInfosInFileName(f, 'date')
        # current_date = current_date.replace("-", ".")
        
        # fluoDirPath = os.path.join(DirDataRaw, current_date + '_Fluo', f)

        # PTL.checkIfBlackFrames()
        # PTL.saveFluoAside(fluoDirPath, f + '.tif')


        ## 0.8 - Sort slices
        ## ! Exp type dependance here !
        # if not logFileImported:
        #     # 1. Import values from status file
        #     for col in statusCols:
        #         PTL.dictLog[col] = statusDf[col].values
        #     PTL.dictLog['iField'] = np.arange(PTL.nS, dtype = int)
            
            # if ('R40' in f) or ('R80' in f):
            #     PTL.determineFramesStatus_R40()
            #     if PTL.expType == 'compressions and constant field':
            #         PTL.expType = 'compressions'
            # if ('thickness' in f):
            #     PTL.determineFramesStatus_R40()
            #     if PTL.expType == 'compressions and constant field':
            #         PTL.expType = 'constant field'
            # elif 'L40' in f:
            #     PTL.determineFramesStatus_L40()
            #     if PTL.expType == 'compressions and constant field':
            #         PTL.expType = 'compressions'
                    
            # elif 'sin' in f:
            #     PTL.determineFramesStatus_Sinus()
            # elif 'brokenRamp' in f:
            #     PTL.determineFramesStatus_BR()
            # elif 'disc20um' in f:
            #     PTL.determineFramesStatus_optoGen()
                


        #### 0.9 - Create list of Frame objects
        PTL.makeFramesList()
        
        PTL.saveLogDf(display = False, save = (not logFileImported), path = logFilePath)

        print(gs.BLUE + 'OK!' + gs.NORMAL)


    # #### 1. Detect beads

    #     print(gs.BLUE + 'Detecting all the bead objects...' + gs.NORMAL)
        Td = time.time()

    #     #### 1.1 - Check if a _Results.txt exists and import it if it's the case
    #     # resFilePath = imagePath[:-4] + '_Results.txt'
    #     resFileImported = False
    #     try:
    #         PTL.importBeadsDetectResult(resPath)
    #         resFileImported = True
    #     except:
    #         pass
        
#         #### 1.2 - Detect the beads
#         # Input the results in each Frame objects [if the results have been loaded at the previous step]
#         PTL.detectBeads(resFileImported)

        print(gs.BLUE + 'OK! dT = {:.3f}'.format(time.time()-Td) + gs.NORMAL)


    #### 2. Make trajectories for beads of interest
        # One of the main steps ! The tracking of the beads happens here !

        print(gs.BLUE + 'Tracking the beads of interest...' + gs.NORMAL)
        Tt = time.time()

        #### 2.1 - Check if some trajectories exist already
        trajDirRaw = os.path.join(DirDataTimeseries, 'Trajectories_raw')
        trajFilesExist_global = False
        trajFilesImported = False
        trajFilesExist_sum = 0
        
        if redoAllSteps:
            pass
        else:
            allTrajPaths = [os.path.join(trajDirRaw, f + '_rawTraj' + str(iB) + '' + '_PY.csv') for iB in range(PTL.NB)]
            allTrajPaths += [os.path.join(trajDirRaw, f + '_rawTraj' + str(iB) + '_In' + '_PY.csv') for iB in range(PTL.NB)]
            allTrajPaths += [os.path.join(trajDirRaw, f + '_rawTraj' + str(iB) + '_Out' + '_PY.csv') for iB in range(PTL.NB)]
            allTrajPaths = np.array(allTrajPaths)
            trajFilesExist = np.array([os.path.isfile(trajPath) for trajPath in allTrajPaths])
            trajFilesExist_sum = np.sum(trajFilesExist)

        #### 2.2 - If yes, load them
        if trajFilesExist_sum == PTL.NB:
            trajFilesImported = True
            trajPaths = allTrajPaths[trajFilesExist]
            for iB in range(PTL.NB):
                PTL.importTrajectories(trajPaths[iB], iB)
                # print(PTL.listTrajectories[iB].dict['X'][0], PTL.listTrajectories[iB].dict['X'][1])
            print(gs.GREEN + 'Raw traj files found and imported :)' + gs.NORMAL)

        #### 2.3 - If no, compute them by tracking the beads
        if not trajFilesImported:
            issue = PTL.buildTrajectories(trackAll = trackAll) 
            # Main tracking function !
            if issue == 'Bug':
                continue
            else:
                pass

        #### 2.4 - Save the user inputs
        PTL.saveLogDf(display = 0, save = True, path = logFilePath)

        print(gs.BLUE + 'OK! dT = {:.3f}'.format(time.time()-Tt) + gs.NORMAL)

        #### 2.5 - Sort the trajectories [Maybe unnecessary]


    #### 3. Qualify - Detect boi sizes and neighbours

        #### 3.1 - Infer or detect Boi sizes in the first image
        # [Detection doesn't work well !]

        if len(PTL.beadTypes) == 1:
            if 'M450' in PTL.beadTypes[0]:
                D = 4.5
            elif 'M270' in PTL.beadTypes[0]:
                D = 2.7

            first_iF = PTL.listTrajectories[0].dict['iF'][0]
            for B in PTL.listFrames[first_iF].listBeads:
                B.D = D
        else:
            PTL.listFrames[0].detectDiameter(plot = 0)

        # Propagate it across the trajectories
        for iB in range(PTL.NB):
            traj = PTL.listTrajectories[iB]
            B0 = traj.dict['Bead'][0]
            D = B0.D
            traj.D = D
            for B in traj.dict['Bead']:
                B.D = D

        #### 3.2 - Detect neighbours

        # Current way, with user input
        if redoAllSteps or not trajFilesImported:
            for iB in range(PTL.NB):
                traj = PTL.listTrajectories[iB]
                beadType = ''
                if len(PTL.beadTypes) == 1:
                    beadType = PTL.beadTypes[0] # M270 or M450
                elif len(PTL.beadTypes) == 2 and PTL.beadTypes[0] == PTL.beadTypes[1]:
                    beadType = PTL.beadTypes[0] # M270 or M450
                else:
                    beadType = 'detect'
                traj.detectNeighbours_ui(Nimg = PTL.NLoops, frequency = PTL.nS//PTL.NLoops, beadType = beadType)


        #### 3.3 - Detect in/out bead

        if redoAllSteps or not trajFilesImported:
            for iB in range(PTL.NB):
                traj = PTL.listTrajectories[iB]
                InOut = traj.detectInOut_ui(Nimg = PTL.NLoops, frequency = PTL.nS//PTL.NLoops)


    #### 4. Compute dz

        #### 4.1 - Import depthographs
        HDZfactor = PTL.listTrajectories[0].HDZfactor
        
        if len(PTL.beadTypes) == 1:
            depthoPath = os.path.join(DirDataRawDeptho, depthoNames)
#             depthoExist = os.path.exists(depthoPath+'_Deptho.tif')
            deptho = io.imread(depthoPath+'_Deptho.tif')
            depthoMetadata = pd.read_csv(depthoPath+'_Metadata.csv', sep=';')
            depthoStep = depthoMetadata.loc[0,'step']
            depthoZFocus = depthoMetadata.loc[0,'focus']

            # increase the resolution of the deptho with interpolation
            # print('deptho shape check')
            # print(deptho.shape)
            nX, nZ = deptho.shape[1], deptho.shape[0]
            XX, ZZ = np.arange(0, nX, 1), np.arange(0, nZ, 1)
            # print(XX.shape, ZZ.shape)
            fd = interpolate.interp2d(XX, ZZ, deptho, kind='cubic')
            ZZ_HD = np.arange(0, nZ, 1/HDZfactor)
            # print(ZZ_HD.shape)
            depthoHD = fd(XX, ZZ_HD)
            depthoStepHD = depthoStep/HDZfactor
            depthoZFocus = depthoZFocus*HDZfactor
            # print(depthoHD.shape)
            #
            for iB in range(PTL.NB):
                traj = PTL.listTrajectories[iB]
                traj.deptho = depthoHD
                traj.depthoPath = depthoPath
                traj.depthoStep = depthoStepHD
                traj.depthoZFocus = depthoZFocus
                traj.HDZfactor = HDZfactor

        if len(PTL.beadTypes) > 1:
            for dN in depthoNames:
                depthoPath = os.path.join(DirDataRawDeptho, dN)
                deptho = io.imread(depthoPath+'_Deptho.tif')
                depthoMetadata = pd.read_csv(depthoPath+'_Metadata.csv', sep=';')
                depthoStep = depthoMetadata.loc[0,'step']
                depthoZFocus = depthoMetadata.loc[0,'focus']

                # increase the resolution of the deptho with interpolation
                nX, nZ = deptho.shape[1], deptho.shape[0]
                XX, ZZ = np.arange(0, nX, 1), np.arange(0, nZ, 1)
                fd = interpolate.interp2d(XX, ZZ, deptho, kind='cubic')
                ZZ_HD = np.arange(0, nZ, 1/HDZfactor)
                depthoHD = fd(XX, ZZ_HD)
                depthoStepHD = depthoStep/HDZfactor
                depthoZFocusHD = depthoZFocus*HDZfactor
                #
                if 'M450' in dN:
                    depthoD = 4.5
                elif 'M270' in dN:
                    depthoD = 2.7
                for iB in range(PTL.NB):
                    traj = PTL.listTrajectories[iB]
                    if traj.D == depthoD:
                        traj.deptho = depthoHD
                        traj.depthoPath = depthoPath
                        traj.depthoStep = depthoStepHD
                        traj.depthoZFocus = depthoZFocusHD

        #### 4.2 - Compute z for each traj
        #### ! Expt dependence here !
        if PTL.microscope == 'metamorph':
            matchingDirection = 'downward' # Change when needed !!
            print(gs.ORANGE + "Deptho detection 'downward' mode" + gs.NORMAL)
        elif PTL.microscope == 'labview' or PTL.microscope == 'old-labview':
            matchingDirection = 'upward'
            print(gs.ORANGE + "Deptho detection 'upward' mode" + gs.NORMAL)
            
        if redoAllSteps or not trajFilesImported:
            for iB in range(PTL.NB):
                np.set_printoptions(threshold=np.inf)

                print(gs.BLUE + 'Computing Z in traj  {:.0f}...'.format(iB+1) + gs.NORMAL)
                Tz = time.time()
                traj = PTL.listTrajectories[iB]
                traj.computeZ(matchingDirection, plot = 0)
                print(gs.BLUE + 'OK! dT = {:.3f}'.format(time.time()-Tz) + gs.NORMAL)

        else:
            print(gs.BLUE + 'Computing Z...' + gs.NORMAL)
            print(gs.GREEN + 'Z had been already computed :)' + gs.NORMAL)
            
        
        
        #### 4.3 - Save the raw traj (before Std selection)
        if redoAllSteps or not trajFilesImported:
            for iB in range(PTL.NB):
                traj = PTL.listTrajectories[iB]
                traj_df = pd.DataFrame(traj.dict)
                trajPathRaw = os.path.join(DirDataTimeseries, 'Trajectories_raw', f + '_rawTraj' + str(iB) + '_' + traj.beadInOut + '_PY.csv')
                traj_df.to_csv(trajPathRaw, sep = '\t', index = False)

        #### 4.4 - Keep only the best std data in the trajectories
        for iB in range(PTL.NB):
            traj = PTL.listTrajectories[iB]
            traj.keepBestStdOnly()

        #### 4.5 - The trajectories won't change from now on. We can save their '.dict' field.
        if redoAllSteps or not trajFilesImported:
            for iB in range(PTL.NB):
                traj = PTL.listTrajectories[iB]
                traj_df = pd.DataFrame(traj.dict)
                trajPath = os.path.join(DirDataTimeseries, 'Trajectories', f + '_traj' + str(iB) + '_' + traj.beadInOut + '_PY.csv')
                traj_df.to_csv(trajPath, sep = '\t', index = False)
                
                # save in ownCloud
                # if ownCloud_timeSeriesDataDir != '':
                #     OC_trajPath = os.path.join(ownCloud_timeSeriesDataDir, 'Trajectories', f + '_traj' + str(iB) + '_' + traj.beadInOut + '_PY.csv')
                #     traj_df.to_csv(OC_trajPath, sep = '\t', index = False)
    
    
    #### 5. Define pairs and compute distances
        print(gs.BLUE + 'Computing distances...' + gs.NORMAL)

        #### 5.1 - In case of 1 pair of beads
        if PTL.NB == 2:
            traj1 = PTL.listTrajectories[0]
            traj2 = PTL.listTrajectories[1]
            nT = traj1.nT

            #### 5.1.1 - Create a dict to prepare the export of the results
            timeSeries = {
                'idxLoop' : np.zeros(nT),
                'idxAnalysis' : np.zeros(nT),
                'T' : np.zeros(nT),
                'Tabs' : np.zeros(nT),
                'B' : np.zeros(nT),
                'F' : np.zeros(nT),
                'dx' : np.zeros(nT),
                'dy' : np.zeros(nT),
                'dz' : np.zeros(nT),
                'D2' : np.zeros(nT),
                'D3' : np.zeros(nT),
            }

            #### 5.1.2 - Input common values:
            T0 = fieldDf['T_abs'].values[0]/1000 # From ms to s conversion
            timeSeries['idxLoop'] = traj1.dict['iL']
            timeSeries['idxAnalysis'] = traj1.dict['idxAnalysis']
            timeSeries['Tabs'] = (fieldDf['T_abs'][traj1.dict['iField']])/1000 # From ms to s conversion
            timeSeries['T'] = timeSeries['Tabs'].values - T0*np.ones(nT)
            timeSeries['B'] = fieldDf['B_set'][traj1.dict['iField']].values
            timeSeries['B'] *= PTL.MagCorrFactor

            #### 5.1.3 - Compute distances
            timeSeries['dx'] = (traj2.dict['X'] - traj1.dict['X'])/PTL.scale
            timeSeries['dy'] = (traj2.dict['Y'] - traj1.dict['Y'])/PTL.scale
            timeSeries['D2'] = (timeSeries['dx']**2 +  timeSeries['dy']**2)**0.5

            timeSeries['dz'] = (traj2.dict['Zr']*traj2.depthoStep - traj1.dict['Zr']*traj1.depthoStep)/1000
            timeSeries['dz'] *= PTL.OptCorrFactor
            timeSeries['D3'] = (timeSeries['D2']**2 +  timeSeries['dz']**2)**0.5

            #print('\n\n* timeSeries:\n')
            #print(timeSeries_DF[['T','B','F','dx','dy','dz','D2','D3']])
            print(gs.BLUE + 'OK!' + gs.NORMAL)


    #### 6. Compute forces
        print(gs.BLUE + 'Computing forces...' + gs.NORMAL)
        Tf = time.time()
        if PTL.NB == 2:
            print(gs.GREEN + '1 pair force computation' + gs.NORMAL)
            traj1 = PTL.listTrajectories[0]
            traj2 = PTL.listTrajectories[1]
            B0 = timeSeries['B']
            D3 = timeSeries['D3']
            dx = timeSeries['dx']
            F, dfLogF = PTL.computeForces(traj1, traj2, B0, D3, dx)
            # Main force computation function
            timeSeries['F'] = F

        print(gs.BLUE + 'OK! dT = {:.3f}'.format(time.time()-Tf) + gs.NORMAL)

            # Magnetization [A.m^-1]
            # M270
            # M = 0.74257*1.05*1600*(0.001991*B.^3+17.54*B.^2+153.4*B)./(B.^2+35.53*B+158.1)
            # M450
            # M = 1.05*1600*(0.001991*B.^3+17.54*B.^2+153.4*B)./(B.^2+35.53*B+158.1);


    #### 7. Export the results

        #### 7.1 - Save the tables !
        if PTL.NB == 2:
            timeSeries_DF = pd.DataFrame(timeSeries)
            timeSeriesFilePath = os.path.join(DirDataTimeseries, f + '_PY.csv')
            timeSeries_DF.to_csv(timeSeriesFilePath, sep = ';', index=False)
            
            if CloudSaving != '':
                CloudTimeSeriesFilePath = os.path.join(DirCloudTimeseries, f + '_PY.csv')
                timeSeries_DF.to_csv(CloudTimeSeriesFilePath, sep = ';', index=False)
    
    print(gs.BLUE + '\nTotal time:' + gs.NORMAL)
    print(gs.BLUE + str(time.time()-start) + gs.NORMAL)
    print(gs.BLUE + '\n' + gs.NORMAL)

    plt.close('all')


        #### 7.2 - Return the last objects, for optional verifications
    listTrajDicts = []
    for iB in range(PTL.NB):
        listTrajDicts.append(PTL.listTrajectories[iB].dict)
        

    #### output
    
    return(PTL.logDf, PTL.log_UIxy)


# %%%% Stand-alone functions

# Tracker stand-Alone Functions

# 1. XYZtracking do the tracking on any image, given
def XYZtracking(I, cellID, NB, manipDict, depthoDir, depthoNames):

    PTL = PincherTimeLapse(I, cellID, manipDict, NB = NB)
    PTL.determineFramesStatus_R40()
    PTL.uiThresholding(method = 'max_entropy', factorT = 1)#, increment = 0.05)
    PTL.makeFramesList()
    PTL.detectBeads(resFileImported = False, display = 1)
    issue = PTL.buildTrajectories()
    XYZtracking_assignBeadSizes(PTL)
    for iB in range(PTL.NB):
        traj = PTL.listTrajectories[iB]
        beadType = ''
        if len(PTL.beadTypes) == 1:
            beadType = PTL.beadTypes[0] # M270 or M450
        else:
            beadType = 'detect'
        traj.detectNeighbours(frequency = PTL.loop_mainSize, beadType = beadType)
    XYZtracking_computeZ(PTL, depthoDir, depthoNames, plot = 1)
    return(PTL)

# Subfuctions of XYZtracking
def XYZtracking_assignBeadSizes(PTL):
    if len(PTL.beadTypes) == 1:
            if 'M450' in PTL.beadTypes[0]:
                D = 4.5
            elif 'M270' in PTL.beadTypes[0]:
                D = 2.7
            for B in PTL.listFrames[0].listBeads:
                B.D = D
    else:
        PTL.listFrames[0].detectDiameter(plot = 0)

    # Propagate it across the trajectories
    for iB in range(PTL.NB):
        traj = PTL.listTrajectories[iB]
        B0 = traj.dict['Bead'][0]
        D = B0.D
        traj.D = D
        for B in traj.dict['Bead']:
            B.D = D

def XYZtracking_computeZ(PTL, depthoDir, depthoNames, plot = 0):
    HDZfactor = 10
    if len(PTL.beadTypes) == 1:
        depthoPath = os.path.join(depthoDir, depthoNames)
#             depthoExist = os.path.exists(depthoPath+'_Deptho.tif')
        deptho = io.imread(depthoPath+'_Deptho.tif')
        depthoMetadata = pd.read_csv(depthoPath+'_Metadata.csv', sep=';')
        depthoStep = depthoMetadata.loc[0,'step']
        depthoZFocus = depthoMetadata.loc[0,'focus']

        # increase the resolution of the deptho with interpolation
        nX, nZ = deptho.shape[1], deptho.shape[0]
        XX, ZZ = np.arange(0, nX, 1), np.arange(0, nZ, 1)
        fd = interpolate.interp2d(XX, ZZ, deptho, kind='cubic')
        ZZ_HD = np.arange(0, nZ, 1/HDZfactor)
        depthoHD = fd(XX, ZZ_HD)
        depthoStepHD = depthoStep/HDZfactor
        depthoZFocusHD = depthoZFocus*HDZfactor
        #
        for iB in range(PTL.NB):
            traj = PTL.listTrajectories[iB]
            traj.deptho = depthoHD
            traj.depthoPath = depthoPath
            traj.depthoStep = depthoStepHD
            traj.depthoZFocus = depthoZFocusHD
            plt.imshow(depthoHD)

    if len(PTL.beadTypes) > 1:
        for dN in depthoNames:
            depthoPath = os.path.join(depthoDir, dN)
            deptho = io.imread(depthoPath+'_Deptho.tif')
            depthoMetadata = pd.read_csv(depthoPath+'_Metadata.csv', sep=';')
            depthoStep = depthoMetadata.loc[0,'step']
            depthoZFocus = depthoMetadata.loc[0,'focus']

            # increase the resolution of the deptho with interpolation
            nX, nZ = deptho.shape[1], deptho.shape[0]
            XX, ZZ = np.arange(0, nX, 1), np.arange(0, nZ, 1)
            fd = interpolate.interp2d(XX, ZZ, deptho, kind='cubic')
            ZZ_HD = np.arange(0, nZ, 1/HDZfactor)
            depthoHD = fd(XX, ZZ_HD)
            depthoStepHD = depthoStep/HDZfactor
            depthoZFocusHD = depthoZFocus*HDZfactor
            #
            if 'M450' in dN:
                depthoD = 4.5
            elif 'M270' in dN:
                depthoD = 2.7
            for iB in range(PTL.NB):
                traj = PTL.listTrajectories[iB]
                if traj.D == depthoD:
                    traj.deptho = depthoHD
                    traj.depthoPath = depthoPath
                    traj.depthoStep = depthoStepHD
                    traj.depthoZFocus = depthoZFocusHD

    # Compute z for each traj
    matchingDirection = 'downward'

    for iB in range(PTL.NB):
        np.set_printoptions(threshold=np.inf)

        print('Computing Z in traj  {:.0f}...'.format(iB+1))

        Tz = time.time()
        traj = PTL.listTrajectories[iB]
        traj.computeZ(matchingDirection, plot)
        print('OK! dT = {:.3f}'.format(time.time()-Tz))

    # Keep only the best std data in the trajectories
    for iB in range(PTL.NB):
        print('Picking the images with the best StdDev in traj  {:.0f}...'.format(iB+1))

        Tstd = time.time()
        traj = PTL.listTrajectories[iB]
        traj.keepBestStdOnly()
        print('OK! dT = {:.3f}'.format(time.time()-Tstd))



# %% (3) Depthograph making classes & functions

# %%%% BeadDeptho

class BeadDeptho:
    def __init__(self, I, X0, Y0, S0, bestZ, scale, beadType, fileName):

        nz, ny, nx = I.shape[0], I.shape[1], I.shape[2]

        self.I = I
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.scale = scale
        self.X0 = X0
        self.Y0 = Y0
        self.S0 = S0
        self.XYm = np.zeros((self.nz, 2))
        self.XYm[S0-1, 0] = X0
        self.XYm[S0-1, 1] = Y0
        self.fileName = fileName

        self.beadType = beadType
        self.D0 = 4.5 * (beadType == 'M450') + 2.7 * (beadType == 'M270')
        # self.threshold = threshold
        self.I_cleanROI = np.array([])
#         self.cleanROI = np.zeros((self.nz, 4), dtype = int)

        self.validBead = True
        self.iValid = -1

        self.bestZ = bestZ
        self.validSlice = np.zeros(nz, dtype = bool)
        self.zFirst = 0
        self.zLast = nz
        self.validDepth = nz

        self.valid_v = True
        self.valid_h = True
        self.depthosDict = {}
        self.profileDict = {}
        self.ZfocusDict = {}


    def buildCleanROI(self, plot):
        # Determine if the bead is to close to the edge on the max frame
        D0 = self.D0 + 4.5*(self.D0 == 0)
        roughSize = np.floor(1.1*D0*self.scale)
        mx, Mx = np.min(self.X0 - 0.5*roughSize), np.max(self.X0 + 0.5*roughSize)
        my, My = np.min(self.Y0 - 0.5*roughSize), np.max(self.Y0 + 0.5*roughSize)
        testImageSize = mx > 0 and Mx < self.nx and my > 0 and My < self.ny

        # Aggregate the different validity test (for now only 1)
        validBead = testImageSize

        # If the bead is valid we can proceed
        if validBead:
            # Detect or infer the size of the beads we are measuring
            if self.beadType == 'detect' or self.D0 == 0:
                counts, binEdges = np.histogram(self.I[self.z_max,my:My,mx:Mx].ravel(), bins=256)
                peaks, peaksProp = find_peaks(counts, height=100, threshold=None, distance=None, prominence=None, \
                                   width=None, wlen=None, rel_height=0.5, plateau_size=None)
                peakThreshVal = 1000
                if counts[peaks[0]] > peakThreshVal:
                    self.D0 = 4.5
                    self.beadType = 'M450'
                else:
                    self.D0 = 2.7
                    self.beadType = 'M270'
        else:
            self.validBead = False

        if validBead:
            for z in range(self.bestZ, -1, -1):
                if not z in self.S0:
                    break
            zFirst = z
            for z in range(self.bestZ, self.nz, +1):
                if not z in self.S0:
                    break
            zLast = z-1

            roughSize = int(np.floor(1.15*self.D0*self.scale))
            roughSize += 1 + roughSize%2
            roughCenter = int((roughSize+1)//2)

            cleanSize = ufun.getDepthoCleanSize(self.D0, self.scale)

            I_cleanROI = np.zeros([self.nz, cleanSize, cleanSize])

            try:
                for i in range(zFirst, zLast):
                    xmi, ymi = self.XYm[i,0], self.XYm[i,1]
                    x1, y1, x2, y2, validBead = ufun.getROI(roughSize, xmi, ymi, self.nx, self.ny)
                    if not validBead:
                        if x1 < 0 or x2 > self.nx:
                            self.valid_h = False
                        if y1 < 0 or y2 > self.ny:
                            self.valid_v = False

        #                 fig, ax = plt.subplots(1,2)
        #                 ax[0].imshow(self.I[i])
                    xm1, ym1 = xmi-x1, ymi-y1
                    I_roughRoi = self.I[i,y1:y2,x1:x2]
        #                 ax[1].imshow(I_roughRoi)
        #                 fig.show()

                    translation = (xm1-roughCenter, ym1-roughCenter)

                    tform = transform.EuclideanTransform(rotation=0, \
                                                         translation = (xm1-roughCenter, ym1-roughCenter))

                    I_tmp = transform.warp(I_roughRoi, tform, order = 1, preserve_range = True)

                    I_cleanROI[i] = np.copy(I_tmp[roughCenter-cleanSize//2:roughCenter+cleanSize//2+1,\
                                                  roughCenter-cleanSize//2:roughCenter+cleanSize//2+1])

                if not self.valid_v and not self.valid_h:
                    self.validBead = False

                else:
                    self.zFirst = zFirst
                    self.zLast = zLast
                    self.validDepth = zLast-zFirst
                    self.I_cleanROI = I_cleanROI.astype(np.uint16)

                # VISUALISE
                if plot >= 2:
                    for i in range(zFirst, zLast, 50):
                        self.plotROI(i)

            except:
                print('Error for the file: ' + self.fileName)


    def buildDeptho(self, plot):
        preferedDeptho = 'v'
        side_ROI = self.I_cleanROI.shape[1]
        mid_ROI = side_ROI//2
        nbPixToAvg = 3 # Have to be an odd number
        deptho_v = np.zeros([self.nz, side_ROI], dtype = np.float64)
        deptho_h = np.zeros([self.nz, side_ROI], dtype = np.float64)
        deptho_HD = np.zeros([self.nz, side_ROI*5], dtype = np.float64)

        if self.valid_v:
            for z in range(self.zFirst, self.zLast):
                templine = side_ROI
                deptho_v[z] = self.I_cleanROI[z,:,mid_ROI] * (1/nbPixToAvg)
                for i in range(1, 1 + nbPixToAvg//2):
                    deptho_v[z] += self.I_cleanROI[z,:,mid_ROI - i] * (1/nbPixToAvg)
                    deptho_v[z] += self.I_cleanROI[z,:,mid_ROI + i] * (1/nbPixToAvg)
            deptho_v = deptho_v.astype(np.uint16)
            self.depthosDict['deptho_v'] = deptho_v

        if self.valid_h:
            for z in range(self.zFirst, self.zLast):
                templine = side_ROI
                deptho_h[z] = self.I_cleanROI[z,mid_ROI,:] * (1/nbPixToAvg)
                for i in range(1, 1 + nbPixToAvg//2):
                    deptho_h[z] += self.I_cleanROI[z,mid_ROI - i,:] * (1/nbPixToAvg)
                    deptho_h[z] += self.I_cleanROI[z,mid_ROI + i,:] * (1/nbPixToAvg)
            deptho_h = deptho_h.astype(np.uint16)
            self.depthosDict['deptho_h'] = deptho_h

        if preferedDeptho == 'v' and not self.valid_v:
            hdDeptho = 'h'
        elif preferedDeptho == 'h' and not self.valid_h:
            hdDeptho = 'v'
        else:
            hdDeptho = preferedDeptho

        if hdDeptho == 'v':
            for z in range(self.zFirst, self.zLast):
                x = np.arange(mid_ROI - 2, mid_ROI + 3)
                y = np.arange(0, side_ROI)
#                 xx, yy = np.meshgrid(x, y)
                vals = self.I_cleanROI[z, :, mid_ROI-2:mid_ROI+3]
                f = interpolate.interp2d(x, y, vals, kind='cubic')
                # Now use the obtained interpolation function and plot the result:

                xnew = x
                ynew = np.arange(0, side_ROI, 0.2)
                vals_new = f(xnew, ynew)
                deptho_HD[z] = vals_new[:,5//2] * (1/nbPixToAvg)
                for i in range(1, 1 + nbPixToAvg//2):
                    deptho_HD[z] += vals_new[:,5//2-i] * (1/nbPixToAvg)
                    deptho_HD[z] += vals_new[:,5//2+i] * (1/nbPixToAvg)
#                 if z == self.z_max:
#                     figInterp, axesInterp = plt.subplots(1,2)
#                     axesInterp[0].imshow(vals)
#                     axesInterp[0].plot([5//2, 5//2], [0, vals.shape[0]], 'r--')
#                     axesInterp[1].imshow(vals_new)
#                     axesInterp[1].plot([5//2, 5//2], [0, vals_new.shape[0]], 'r--')
#                     figInterp.show()
            deptho_HD = deptho_HD.astype(np.uint16)
            self.depthosDict['deptho_HD'] = deptho_HD

        elif hdDeptho == 'h':
            for z in range(self.zFirst, self.zLast):
                x = np.arange(0, side_ROI)
                y = np.arange(mid_ROI - 2, mid_ROI + 3)
#                 xx, yy = np.meshgrid(x, y)
                vals = self.I_cleanROI[z, mid_ROI-2:mid_ROI+3, :]
                f = interpolate.interp2d(x, y, vals, kind='cubic')
                # Now use the obtained interpolation function and plot the result:

                xnew = np.arange(0, side_ROI, 0.2)
                ynew = y
                vals_new = f(xnew, ynew)
                deptho_HD[z] = vals_new[5//2,:] * (1/nbPixToAvg)
                for i in range(1, 1 + nbPixToAvg//2):
                    deptho_HD[z] += vals_new[5//2-i,:] * (1/nbPixToAvg)
                    deptho_HD[z] += vals_new[5//2+i,:] * (1/nbPixToAvg)
#                 if z == self.z_max:
#                     figInterp, axesInterp = plt.subplots(1,2)
#                     axesInterp[0].imshow(vals)
#                     axesInterp[0].plot([0, vals.shape[1]], [5//2, 5//2], 'r--')
#                     axesInterp[1].imshow(vals_new)
#                     axesInterp[1].plot([0, vals_new.shape[1]], [5//2, 5//2], 'r--')
#                     figInterp.show()
            deptho_HD = deptho_HD.astype(np.uint16)
            self.depthosDict['deptho_HD'] = deptho_HD

        # 3D caracterisation
#         I_binary = np.zeros([self.I_cleanROI.shape[0], self.I_cleanROI.shape[1], self.I_cleanROI.shape[2]])
#         I_binary[self.zFirst:self.zLast] = (self.I_cleanROI[self.zFirst:self.zLast] > self.threshold)
#         Zm3D, Ym3D, Xm3D = ndi.center_of_mass(self.I_cleanROI, labels=I_binary, index=1)
#         self.ZfocusDict['Zm3D'] = Zm3D

        # Raw profiles
        mid_ROI_HD = deptho_HD.shape[1]//2
        Z = np.array([z for z in range(self.I_cleanROI.shape[0])])
#         intensity_tot = np.array([np.sum(self.I_cleanROI[z][I_binary[z].astype(bool)])/(1+np.sum(I_binary[z])) for z in range(self.I_cleanROI.shape[0])]).astype(np.float64)
        intensity_v = np.array([np.sum(deptho_v[z,:])/side_ROI for z in range(deptho_v.shape[0])]).astype(np.float64)
        intensity_h = np.array([np.sum(deptho_h[z,:])/side_ROI for z in range(deptho_h.shape[0])]).astype(np.float64)
        intensity_HD = np.array([np.sum(deptho_HD[z,mid_ROI_HD-5:mid_ROI_HD+6])/11 for z in range(deptho_HD.shape[0])]).astype(np.float64)
#
        Zm_v, Zm_h = np.argmax(intensity_v), np.argmax(intensity_h)
#         Zm_tot = np.argmax(intensity_tot)
        Zm_HD = np.argmax(intensity_HD)

        self.profileDict['intensity_v'] = intensity_v
        self.profileDict['intensity_h'] = intensity_h
        self.profileDict['intensity_HD'] = intensity_HD
#         self.profileDict['intensity_tot'] = intensity_tot
        self.ZfocusDict['Zm_v'] = Zm_v
        self.ZfocusDict['Zm_h'] = Zm_h
        self.ZfocusDict['Zm_HD'] = Zm_HD
#         self.ZfocusDict['Zm_tot'] = Zm_tot


        # Smoothed profiles
        Z_hd = np.arange(0, self.I_cleanROI.shape[0], 0.2)
        intensity_v_hd = np.interp(Z_hd, Z, intensity_v)
        intensity_h_hd = np.interp(Z_hd, Z, intensity_h)
        intensity_HD_hd = np.interp(Z_hd, Z, intensity_HD)
#         intensity_tot_hd = np.interp(Z_hd, Z, intensity_tot)

        intensity_v_smooth = savgol_filter(intensity_v_hd, 101, 5)
        intensity_h_smooth = savgol_filter(intensity_h_hd, 101, 5)
        intensity_HD_smooth = savgol_filter(intensity_HD_hd, 101, 5)
#         intensity_tot_smooth = savgol_filter(intensity_tot_hd, 101, 5)

        Zm_v_hd, Zm_h_hd = Z_hd[np.argmax(intensity_v_smooth)], Z_hd[np.argmax(intensity_h_smooth)]
#         Zm_tot_hd = Z_hd[np.argmax(intensity_tot_smooth)]
        Zm_HD_hd = Z_hd[np.argmax(intensity_HD_smooth)]

        self.profileDict['intensity_v_smooth'] = intensity_v_smooth
        self.profileDict['intensity_h_smooth'] = intensity_h_smooth
        self.profileDict['intensity_HD_smooth'] = intensity_HD_smooth
#         self.profileDict['intensity_tot_smooth'] = intensity_tot_smooth
        self.ZfocusDict['Zm_v_hd'] = Zm_v_hd
        self.ZfocusDict['Zm_h_hd'] = Zm_h_hd
        self.ZfocusDict['Zm_HD_hd'] = Zm_HD_hd
#         self.ZfocusDict['Zm_tot_hd'] = Zm_tot_hd

        # VISUALISE
        if plot >= 2:
            self.plotProfiles()


    def saveBeadDeptho(self, path, ID, step, bestDetphoType = 'HD', bestFocusType = 'HD_hd'):
        supDataDir = ID + '_supData'
        supDataDirPath = os.path.join(path, supDataDir)
        if not os.path.exists(supDataDirPath):
            os.makedirs(supDataDirPath)

        cleanROIName = ID + '_cleanROI.tif'
        cleanROIPath = os.path.join(path, cleanROIName)
        io.imsave(cleanROIPath, self.I_cleanROI, check_contrast=False)

        profilesRaw_keys = ['intensity_v', 'intensity_h', 'intensity_HD'] #, 'intensity_tot']
        profileDictRaw = {k: self.profileDict[k] for k in profilesRaw_keys}
        profileDictRaw_df = pd.DataFrame(profileDictRaw)
        profileDictRaw_df.to_csv(os.path.join(supDataDirPath, 'profiles_raw.csv'))

        profilesSmooth_keys = ['intensity_v_smooth', 'intensity_h_smooth', 'intensity_HD_smooth'] #, 'intensity_tot_smooth']
        profileDictSmooth = {k: self.profileDict[k] for k in profilesSmooth_keys}
        profileDictSmooth_df = pd.DataFrame(profileDictSmooth)
        profileDictSmooth_df.to_csv(os.path.join(supDataDirPath, 'profiles_smooth.csv'))

        ZfocusDict_df = pd.DataFrame(self.ZfocusDict, index = [1])
        ZfocusDict_df.to_csv(os.path.join(supDataDirPath, 'Zfoci.csv'))

        bestFocus = self.ZfocusDict['Zm_' + bestFocusType]
        metadataPath = os.path.join(path, ID + '_Metadata.csv')
        with open(metadataPath, 'w') as f:
            f.write('step;bestFocus')
#             for k in self.ZfocusDict.keys():
#                 f.write(';')
#                 f.write(k)
            f.write('\n')
            f.write(str(step) + ';' + str(bestFocus))
#             for k in self.ZfocusDict.keys():
#                 f.write(';')
#                 f.write(str(self.ZfocusDict[k]))

        depthoPath = os.path.join(path, ID + '_deptho.tif')
        bestDeptho = self.depthosDict['deptho_' + bestDetphoType]
        io.imsave(depthoPath, bestDeptho)



# Plot functions

    def plotXYm(self):
        fig, ax = plt.subplots(1,1)
        pStart, pStop = np.percentile(self.I[self.z_max], (1, 99))
        ax.imshow(self.I[self.z_max], cmap = 'gray', vmin = pStart, vmax = pStop)
        ax.plot(self.XYm[self.validSlice,0],self.XYm[self.validSlice,1],'r-')
        fig.show()

    def plotROI(self, i = 'auto'):
        if i == 'auto':
            i = self.z_max

        fig, ax = plt.subplots(1,3, figsize = (16,4))

        xm, ym = np.mean(self.XYm[self.validSlice,0]),  np.mean(self.XYm[self.validSlice,1])
        ROIsize_x = self.D*1.25*self.scale + (max(self.XYm[self.validSlice,0])-min(self.XYm[self.validSlice,0]))
        ROIsize_y = self.D*1.25*self.scale + (max(self.XYm[self.validSlice,1])-min(self.XYm[self.validSlice,1]))
        x1_ROI, y1_ROI, x2_ROI, y2_ROI = int(xm - ROIsize_x//2), int(ym - ROIsize_y//2), int(xm + ROIsize_x//2), int(ym + ROIsize_y//2)

        pStart, pStop = np.percentile(self.I[i], (1, 99))
        ax[0].imshow(self.I[i], cmap = 'gray', vmin = pStart, vmax = pStop)
        ax[0].plot([x1_ROI,x1_ROI], [y1_ROI,y2_ROI], 'c--')
        ax[0].plot([x1_ROI,x2_ROI], [y2_ROI,y2_ROI], 'c--')
        ax[0].plot([x2_ROI,x2_ROI], [y1_ROI,y2_ROI], 'c--')
        ax[0].plot([x1_ROI,x2_ROI], [y1_ROI,y1_ROI], 'c--')

        I_ROI = self.I[i,y1_ROI:y2_ROI,x1_ROI:x2_ROI]
        pStart, pStop = np.percentile(I_ROI, (1, 99))
        ax[1].imshow(I_ROI, cmap = 'gray', vmin = pStart, vmax = pStop)
        ax[1].plot(self.XYm[self.validSlice,0]-x1_ROI, self.XYm[self.validSlice,1]-y1_ROI, 'r-', lw=0.75)
        ax[1].plot(self.XYm[i,0]-x1_ROI, self.XYm[i,1]-y1_ROI, 'b+', lw=0.75)

        pStart, pStop = np.percentile(self.I_cleanROI[i], (1, 99))
        mid = self.I_cleanROI[i].shape[0]//2
        I_cleanROI_binary = (self.I_cleanROI[i] > self.threshold)
        y, x = ndi.center_of_mass(self.I_cleanROI[i], labels=I_cleanROI_binary, index=1)
        ax[2].imshow(self.I_cleanROI[i], cmap = 'gray', vmin = pStart, vmax = pStop)
        ax[2].plot([0,2*mid],[mid, mid], 'r--', lw = 0.5)
        ax[2].plot([mid, mid],[0,2*mid], 'r--', lw = 0.5)
        ax[2].plot([x],[y], 'b+')
        fig.show()

    def plotProfiles(self):
        Z = np.array([z for z in range(self.I_cleanROI.shape[0])])
        Z_hd = np.arange(0, self.I_cleanROI.shape[0], 0.2)
        intensity_v = self.profileDict['intensity_v']
        intensity_h = self.profileDict['intensity_h']
        intensity_HD = self.profileDict['intensity_HD']
        # intensity_tot = self.profileDict['intensity_tot']
        Zm_v = self.ZfocusDict['Zm_v']
        Zm_h = self.ZfocusDict['Zm_h']
        Zm_HD = self.ZfocusDict['Zm_HD']
        # Zm_tot = self.ZfocusDict['Zm_tot']
        intensity_v_smooth = self.profileDict['intensity_v_smooth']
        intensity_h_smooth = self.profileDict['intensity_h_smooth']
        intensity_HD_smooth = self.profileDict['intensity_HD_smooth']
        intensity_tot_smooth = self.profileDict['intensity_tot_smooth']
        Zm_v_hd = self.ZfocusDict['Zm_v_hd']
        Zm_h_hd = self.ZfocusDict['Zm_h_hd']
        Zm_HD_hd = self.ZfocusDict['Zm_HD_hd']
        # Zm_tot_hd = self.ZfocusDict['Zm_tot_hd']

        fig, ax = plt.subplots(1,2, figsize = (12, 4))
        ax[0].plot(Z, intensity_v)
        ax[1].plot(Z, intensity_h)
        # ax[2].plot(Z, (intensity_tot))
        ax[0].plot([Zm_v, Zm_v], [0, ax[0].get_ylim()[1]], 'r--', lw = 0.8, label = 'Zm_v = {:.2f}'.format(Zm_v))
        ax[1].plot([Zm_h, Zm_h], [0, ax[1].get_ylim()[1]], 'r--', lw = 0.8, label = 'Zm_h = {:.2f}'.format(Zm_h))
        # ax[2].plot([Zm_tot, Zm_tot], [0, ax[2].get_ylim()[1]], 'r--', lw = 0.8, label = 'Zm_tot = {:.2f}'.format(Zm_tot))
        ax[0].legend(loc = 'lower right')
        ax[1].legend(loc = 'lower right')
        # ax[2].legend(loc = 'lower right')

        # fig, ax = plt.subplots(1,4, figsize = (16, 4))
        # ax[0].plot(Z, intensity_v, 'b-')
        # ax[1].plot(Z, intensity_h, 'b-')
        # ax[2].plot(Z, intensity_HD, 'b-')
        # ax[3].plot(Z, (intensity_tot), 'b-')
        # ax[0].plot(Z_hd, intensity_v_smooth, 'k--')
        # ax[1].plot(Z_hd, intensity_h_smooth, 'k--')
        # ax[2].plot(Z_hd, intensity_HD_smooth, 'k--')
        # ax[3].plot(Z_hd, intensity_tot_smooth, 'k--')
        # ax[0].plot([Zm_v_hd, Zm_v_hd], [0, ax[0].get_ylim()[1]], 'r--', lw = 0.8, label = 'Zm_v_hd = {:.2f}'.format(Zm_v_hd))
        # ax[1].plot([Zm_h_hd, Zm_h_hd], [0, ax[1].get_ylim()[1]], 'r--', lw = 0.8, label = 'Zm_h_hd = {:.2f}'.format(Zm_h_hd))
        # ax[2].plot([Zm_HD_hd, Zm_HD_hd], [0, ax[2].get_ylim()[1]], 'r--', lw = 0.8, label = 'Zm_HD_hd = {:.2f}'.format(Zm_HD_hd))
        # ax[3].plot([Zm_tot_hd, Zm_tot_hd], [0, ax[3].get_ylim()[1]], 'r--', lw = 0.8, label = 'Zm_tot_hd = {:.2f}'.format(Zm_tot_hd))
        # ax[0].legend(loc = 'lower right')
        # ax[1].legend(loc = 'lower right')
        # ax[2].legend(loc = 'lower right')
        # ax[3].legend(loc = 'lower right')

        #         print('Zm_v = {:.2f}, Zm_h = {:.2f}, Zm_tot = {:.2f}'\
        #               .format(Zm_v, Zm_h, Zm_tot))
        #         print('Zm_v_hd = {:.2f}, Zm_h_hd = {:.2f}, Zm_tot_hd = {:.2f}'\
        #               .format(Zm_v_hd, Zm_h_hd, Zm_tot_hd))

        fig.show()


    def plotDeptho(self, d = 'HD'):
        fig, ax = plt.subplots(1,1, figsize = (4, 6))
        D = self.depthosDict['deptho_' + d]
        z_focus = self.ZfocusDict['Zm_' + d + '_hd']
        ny, nx = D.shape[0], D.shape[1]
        pStart, pStop = np.percentile(D, (1, 99))
        pStop = pStop + 0.3 * (pStop-pStart)
        ax.imshow(D, cmap='plasma', vmin = pStart, vmax = pStop)
        ax.plot([0, nx], [self.zFirst, self.zFirst], 'r--')
        ax.text(nx//2, self.zFirst - 10, str(self.zFirst), c = 'r')
        ax.plot([0, nx], [self.zLast, self.zLast], 'r--')
        ax.text(nx//2, self.zLast - 10, str(self.zLast), c = 'r')
        ax.plot([nx//2], [z_focus], 'c+')
        ax.text(nx//2, z_focus - 10, str(z_focus), c = 'c')
        fig.suptitle('File ' + self.fileName + ' - Bead ' + str(self.iValid))
        fig.show()

# %%%% depthoMaker

def depthoMaker(dirPath, savePath, specif, saveLabel, scale, beadType = 'M450', step = 20, d = 'HD', plot = 0):
    rawFileList = os.listdir(dirPath)
    listFileNames = [f[:-4] for f in rawFileList if (os.path.isfile(os.path.join(dirPath, f)) and f.endswith(".tif"))]
    L = []

    for f in listFileNames:
        test1 = (specif in f) or (specif == 'all')
        test2 = ((f + '_Results.txt') in os.listdir(dirPath))
        valid = test1 and test2
        if valid:
            L.append(f)

    listFileNames = L

    listBD = []
#     dictBD = {}

#     print(listFileNames)
    for f in listFileNames:
        filePath = os.path.join(dirPath, f)
        I = io.imread(filePath + '.tif')
        resDf = pd.read_csv((filePath + '_Results.txt'), sep = '\t').drop(columns = [' '])
        # Area,StdDev,XM,YM,Slice
        X0 = resDf['XM'].values
        Y0 = resDf['YM'].values
        S0 = resDf['Slice'].values
        bestZ = S0[np.argmax(resDf['StdDev'].values)] - 1 # The index of the image with the highest Std
        # This image will be more or less the one with the brightest spot

        # Create the BeadDeptho object
        BD = BeadDeptho(I, X0, Y0, S0, bestZ, scale, beadType, f)

        # Creation of the clean ROI where the center of mass is always perfectly centered.
        BD.buildCleanROI(plot)

        # If the bead was not acceptable (for instance too close to the edge of the image)
        # then BD.validBead will be False
        if not BD.validBead:
            print(gs.RED + 'Not acceptable file: ' + f + gs.NORMAL)

        # Else, we can proceed.
        else:
            print(gs.BLUE + 'Job done for the file: ' + f + gs.NORMAL)

            # Creation of the z profiles
            BD.buildDeptho(plot)

        listBD.append(BD)
        i = 1
        for BD in listBD:
#             BD_manipID = findInfosInFileName(BD.fileName, 'manipID')
            subFileSavePath = os.path.join(savePath, 'Intermediate_Py', saveLabel + '_step' + str(step))
            # BD.saveBeadDeptho(subFileSavePath, specif +  '_' + str(i), step = step, bestDetphoType = 'HD', bestFocusType = 'HD_hd')
            BD.saveBeadDeptho(subFileSavePath, f, step = step, bestDetphoType = 'HD', bestFocusType = 'HD_hd')
            i += 1

#
## If different sizes of bead at once #
#
#         for BD in listBD:
#             if BD.D0 not in dictBD.keys():
#                 dictBD[BD.D0] = [BD]
#             else:
#                 dictBD[BD.D0].append(BD)
#
#     for size in dictBD.keys():
#         listBD = dictBD[size]
#     ... go on with the code below with an indent added !


    maxAboveZm, maxBelowZm = 0, 0
    for BD in listBD:
        Zm = int(np.round(BD.ZfocusDict['Zm_' + d + '_hd']))
        if Zm - BD.zFirst > maxAboveZm:
            maxAboveZm = Zm - BD.zFirst
        if BD.zLast - Zm > maxBelowZm:
            maxBelowZm = BD.zLast - Zm
    maxAboveZm, maxBelowZm = int(maxAboveZm), int(maxBelowZm)
    Zfocus = maxAboveZm
    depthoWidth = listBD[0].depthosDict['deptho_' + d].shape[1]
    depthoHeight = maxAboveZm + maxBelowZm
    finalDeptho = np.zeros([depthoHeight, depthoWidth], dtype = np.float64)

    for z in range(1, maxAboveZm+1):
        count = 0
        for BD in listBD:
            Zm = int(np.round(BD.ZfocusDict['Zm_' + d + '_hd']))
            currentDeptho = BD.depthosDict['deptho_' + d]
            if Zm-z >= 0 and np.sum(currentDeptho[Zm-z,:] != 0):
                count += 1
        for BD in listBD:
            Zm = int(np.round(BD.ZfocusDict['Zm_' + d + '_hd']))
            currentDeptho = BD.depthosDict['deptho_' + d]
            if Zm-z >= 0 and np.sum(currentDeptho[Zm-z,:] != 0):
                finalDeptho[Zfocus-z,:] += currentDeptho[Zm-z,:]/count

    for z in range(0, maxBelowZm):
        count = 0
        for BD in listBD:
            Zm = int(np.round(BD.ZfocusDict['Zm_' + d + '_hd']))
            currentDeptho = BD.depthosDict['deptho_' + d]
#             print(currentDeptho.shape)
            if Zm+z >= 0 and Zm+z < currentDeptho.shape[0] and np.sum(currentDeptho[Zm+z,:] != 0):
                count += 1
        for BD in listBD:
            Zm = int(np.round(BD.ZfocusDict['Zm_' + d + '_hd']))
            currentDeptho = BD.depthosDict['deptho_' + d]
            if Zm+z >= 0 and Zm+z < currentDeptho.shape[0] and np.sum(currentDeptho[Zm+z,:] != 0):
                finalDeptho[Zfocus+z,:] += currentDeptho[Zm+z,:]/count

    # print(Zm, maxAboveZm, maxBelowZm)
    finalDeptho = finalDeptho.astype(np.uint16)

    fig, ax = plt.subplots(1,1)
    ax.imshow(finalDeptho)

    fig.suptitle(beadType)
    fig.show()

    depthoSavePath = os.path.join(savePath, saveLabel + '_Deptho.tif')
    io.imsave(depthoSavePath, finalDeptho)
    metadataPath = os.path.join(savePath, saveLabel + '_Metadata.csv')
    with open(metadataPath, 'w') as f:
        f.write('step;focus')
        f.write('\n')
        f.write(str(step) + ';' + str(Zfocus))

    print(gs.GREEN + 'ok' + gs.NORMAL)


# Finished !
