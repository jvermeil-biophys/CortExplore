# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:12:49 2022

@author: JosephVermeil
"""

import os
import re
import time
import pandas as pd
import numpy as np
from skimage import io
dateFormat1 = re.compile('\d{2}-\d{2}-\d{2}')

# Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)
import GraphicStyles as gs
# import GlobalConstants as gc
# import UtilityFunctions as ufun

# 2. Pandas settings
pd.set_option('mode.chained_assignment',None)

# 3. Graphical settings
gs.set_default_options_jv()


# %% Functions

def isWithin(x, y, a):
    """
    Returns if x is in [y-a, y+a], boundaries included.
    Works with numbers and numpy arrays.
    """
    return(np.abs(x-y) <= a)

def formatImageFilesFromValentin(srcPath, dstPath, tag = ''):
    expType = 'R80'
    list_all_files = os.listdir(srcPath)
    list_selected_files = [f for f in list_all_files if ((tag in f) and (expType in f))]
    for f in list_selected_files:
        if f.endswith('.tif'):
            T0 = time.time()
            f_s = f.split('.')
            f0 = ''.join([str(s) for s in f_s[:-1]])
            f_I = f0 + '.tif'
            f_F = f0 + '_Field.txt'
            f_R = f0 + '_Results.txt'
            p_I, p_F, p_R = os.path.join(srcPath, f_I), os.path.join(srcPath, f_F), os.path.join(srcPath, f_R)
            
            print(f0)
            
            ps_I, ps_F, ps_R = os.path.join(dstPath, f_I), os.path.join(dstPath, f_F), os.path.join(dstPath, f_R)
            # all_done = (os.path.isfile(ps_I) and os.path.isfile(ps_F) and os.path.isfile(ps_R))
            all_done = False
            
            if not all_done and os.path.isfile(p_F) and os.path.isfile(p_R):
                dfF_raw = pd.read_csv(p_F, sep = '\t', header = None, names=['field', 'T_raw'])
                dfR_raw = pd.read_csv(p_R, sep = '\t', header=0)
                dfR_raw = dfR_raw.drop(columns=dfR_raw.columns[0])
                I = io.imread(p_I)
                nz, ny, nx = I.shape
                print(dfF_raw.shape, I.shape)
                # out = (dfF, dfR)
                
                T_raw = dfF_raw.T_raw.values
                dT_raw = np.diff(T_raw)
                
                removeFirst = (dT_raw[0] > 535)
                
                tripletTemplate = np.array([520, 40, 40])
                correctLastTriplet = not np.all(isWithin(dT_raw[-3:], tripletTemplate, 10))
                
                print(removeFirst, correctLastTriplet)
                dfR = dfR_raw.loc[:,:]
                
                if removeFirst:
                    dfF = dfF_raw.drop(0, axis = 0)
                    # dfR = dfR_raw.drop(dfR_raw[dfR_raw['Slice'] == 1].index, axis = 0)
                    # dfR['Slice'] = dfR['Slice'] - 1
                    # dfR = dfR.reset_index(drop=True)

                    
                else:
                    dfF = dfF_raw.loc[:,:]
                    # dfR = dfR_raw.loc[:,:]
                    
                if correctLastTriplet:
                    idx = np.argmax(dT_raw[-3:] > 500)
                    
                    def add_fake_timepoint_at_end(dfF, N=1):
                        for i in range(N):
                            B_last, T_last = dfF.field.values[-1], dfF.T_raw.values[-1]
                            row_last = pd.DataFrame([[B_last, T_last+40]], columns=['field', 'T_raw'])
                            dfF = pd.concat([dfF, row_last])
                            dfF = dfF.reset_index(drop=True)
                        return(dfF)
                    
                    if idx == 1: # dT_raw[-3:] = [40, 520, 40], one image is missing
                        dfF = add_fake_timepoint_at_end(dfF, N=1)
                        
                    elif idx == 2: # dT_raw[-3:] = [40, 40, 520], two images are missing
                        dfF = add_fake_timepoint_at_end(dfF, N=2)
                        
                # print(dfF.shape, I.shape)
                
                # A = np.zeros(dfF.shape[0])
                # A[1:] = np.diff(dfF.T_raw.values)
                # dfF['dT'] = A
                # out = (dfF, I[-1,:,:])
                
                # ps_I, ps_F, ps_R = os.path.join(dstPath, f_I), os.path.join(dstPath, f_F), os.path.join(dstPath, f_R)
                
                if not os.path.isfile(ps_I):
                    io.imsave(ps_I, I)
                # if not os.path.isfile(ps_F):
                dfF.to_csv(ps_F, sep='\t', header = False, index = False)
                if not os.path.isfile(ps_R):
                    dfR.to_csv(ps_R, sep='\t', header = True, index = True)
                
                print(dfF.shape[0]/179, I.shape[0]/179)
                
            T1 = time.time()
            print(gs.GREEN + 'T = {:.3f}'.format(T1 - T0) + gs.NORMAL)
            
            
def formatImageFilesFromValentin_V2(srcPath, dstPath, tag = ''):
    expType = 'R80'
    list_all_files = os.listdir(srcPath)
    list_selected_files = [f for f in list_all_files if ((tag in f) and (expType in f))]
    for f in list_selected_files:
        if f.endswith('.tif'):
            T0 = time.time()
            f_s = f.split('.')
            f0 = ''.join([str(s) for s in f_s[:-1]])
            f_F = f0 + '_Field.txt'
            p_F = os.path.join(srcPath, f_F)
            
            print(f0)
            
            ps_F = os.path.join(dstPath, f_F)
            
            if os.path.isfile(p_F):
                dfF_raw = pd.read_csv(p_F, sep = '\t', header = None, names=['field', 'T_raw'])

                # out = (dfF, dfR)
                
                T_raw = dfF_raw.T_raw.values
                dT_raw = np.diff(T_raw)
                
                removeFirst = (dT_raw[0] > 535)
                
                tripletTemplate = np.array([520, 40, 40])
                correctLastTriplet = not np.all(isWithin(dT_raw[-3:], tripletTemplate, 10))
                
                print(removeFirst, correctLastTriplet)
                
                if removeFirst:
                    dfF = dfF_raw.drop(0, axis = 0)
                    # dfR = dfR_raw.drop(dfR_raw[dfR_raw['Slice'] == 1].index, axis = 0)
                    # dfR['Slice'] = dfR['Slice'] - 1
                    # dfR = dfR.reset_index(drop=True)

                    
                else:
                    dfF = dfF_raw.loc[:,:]
                    # dfR = dfR_raw.loc[:,:]
                    
                if correctLastTriplet:
                    idx = np.argmax(dT_raw[-3:] > 500)
                    
                    def add_fake_timepoint_at_end(dfF, N=1):
                        for i in range(N):
                            B_last, T_last = dfF.field.values[-1], dfF.T_raw.values[-1]
                            row_last = pd.DataFrame([[B_last, T_last+40]], columns=['field', 'T_raw'])
                            dfF = pd.concat([dfF, row_last])
                            dfF = dfF.reset_index(drop=True)
                        return(dfF)
                    
                    if idx == 1: # dT_raw[-3:] = [40, 520, 40], one image is missing
                        dfF = add_fake_timepoint_at_end(dfF, N=1)
                        
                    elif idx == 2: # dT_raw[-3:] = [40, 40, 520], two images are missing
                        dfF = add_fake_timepoint_at_end(dfF, N=2)
                        
                # print(dfF.shape, I.shape)
                
                # A = np.zeros(dfF.shape[0])
                # A[1:] = np.diff(dfF.T_raw.values)
                # dfF['dT'] = A
                # out = (dfF, I[-1,:,:])
                
                # ps_I, ps_F, ps_R = os.path.join(dstPath, f_I), os.path.join(dstPath, f_F), os.path.join(dstPath, f_R)
                
                dfF.to_csv(ps_F, sep='\t', header = False, index = False)
                
                print(dfF.shape[0]/179)
                
            T1 = time.time()
            print(gs.GREEN + 'T = {:.3f}'.format(T1 - T0) + gs.NORMAL)
    


# %% Script
        
srcDir0 = "D://MagneticPincherData//Raw"
# dates = ['18.08.28', '18.09.24', '18.09.25', '18.10.30', '18.12.12']
dates = ['18.09.24', '18.09.25', '18.10.30']
tag = ''

for d in dates[:]:
    srcDir = os.path.join(srcDir0, d)
    dstDir = os.path.join(srcDir0, d)
    out = formatImageFilesFromValentin_V2(srcDir, dstDir, tag = tag)

# dirTest = "D://MagneticPincherData//Raw_DC//Test"
# CellIdTest = '12-12-18_M1_P1_C5'

# out = formatImageFilesFromValentin(dirTest, tag = CellIdTest)


