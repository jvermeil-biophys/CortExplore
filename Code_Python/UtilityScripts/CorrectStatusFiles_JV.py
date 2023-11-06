# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 18:48:48 2023

@author: JosephVermeil
"""

import os

import numpy as np
import pandas as pd


# %%

# mainDir = "D://MagneticPincherData//Raw//23.07.06"

# files = os.listdir(mainDir)
# for f in files:
#     if f.endswith('_Status.txt'):
#         print(f)
#         fileName = f

# fileName = "23-07-06_M1_P1_C1_20um_5repeats_Status.txt"
# # fileName = "23-07-06_M6_P1_C1_20um_5mT_Status.txt"
# filePath = os.path.join(mainDir, fileName)

# df = pd.read_csv(filePath, '_', header = None, names = ['type', 'details'])

# typeCol = df.type.values

# iLCol =  np.zeros(df.shape[0], dtype = int)
# count = 0
# for i in range(df.shape[0]):
#     if typeCol[i] == 'StartLoop':
#         count += 1
#     iLCol[i] = count
    
# if count == 0:
#     iLCol = np.array([1 + i//265 for i in range(df.shape[0])])


# df['iL'] = iLCol

# df.insert(0, 'iL', df.pop('iL'))

# df = df.drop(index = df[df['type'] == 'StartLoop'].index)

# df.to_csv(filePath, sep='_', index = False, header = False)

# %%

mainDir = "D://MagneticPincherData//Raw//23.07.06"

files = os.listdir(mainDir)
for f in files:
    if f.endswith('_Status.txt') and f.startswith('23-07-06_M8') and (not (f.startswith('23-07-06_M6_P1_C1'))):
        print(f)
        fileName = f

# fileName = "23-07-06_M6_P1_C3_20um_5mT_Status.txt"
        filePath = os.path.join(mainDir, fileName)
        
        df = pd.read_csv(filePath, sep='_', header = None, names = ['iL', 'type', 'details'])
        
        
        iLCol = np.array([1 + i//235 for i in range(df.shape[0])])
        
        
        df['iL'] = iLCol
        
        # df.insert(0, 'iL', df.pop('iL'))
        
        # df = df.drop(index = df[df['type'] == 'StartLoop'].index)
        
        # df.to_csv(filePath, sep='_', index = False, header = False)
        
        
# %%

# mainDir = "D://MagneticPincherData//Data_Timeseries"

# dirTraj = "D://MagneticPincherData//Data_Timeseries//Trajectories"

# files = os.listdir(mainDir)
# for f in files:
#     if f.endswith('_PY.csv') and (f.startswith('23-07-17') or (f.startswith('23-07-20'))):
#         print(f)
#         fileName = f
        
        

# # fileName = "23-07-06_M6_P1_C3_20um_5mT_Status.txt"
#         filePath = os.path.join(mainDir, fileName)
        
#         df = pd.read_csv(filePath, sep=';', header = 0)
        
        
#         iLCol = np.array([1 + i//265 for i in range(df.shape[0])])
        
        
#         df['iL'] = iLCol
        
#         df.insert(0, 'iL', df.pop('iL'))
        
#         # df = df.drop(index = df[df['type'] == 'StartLoop'].index)
        
#         # df.to_csv(filePath, sep=';', index = False, header = True)
        
