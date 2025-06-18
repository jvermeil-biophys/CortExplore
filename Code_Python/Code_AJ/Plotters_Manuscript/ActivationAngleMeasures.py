# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 15:24:21 2025

@author: anumi
"""

import os
import math
import numpy as np
import pandas as pd

#%% Defintions

def angle_between_points(A, B, C):
    xa, ya = A
    xb, yb = B
    xc, yc = C
    
    d_ba = np.sqrt((xb - xa)**2 + (yb - ya)**2)
    d_ac = np.sqrt((xa - xc)**2 + (ya - yc)**2)
    d_bc = np.sqrt((xb - xc)**2 + (yb - yc)**2)
    
    angle_abc = np.arccos((d_ba**2 + d_bc**2 - d_ac**2) / (2*d_ba*d_bc))
    
    angle_abc = np.round(angle_abc*(180/np.pi), 2)
    
    
    return angle_abc

#%% To measure angle of beads from activation region

dates = ['24-12-20', '25-02-28', '25-03-12']

path = 'D:/Anumita/MagneticPincherData/Data_Polarization/'

for date in dates:
    cellCom = pd.read_csv(os.path.join(path, date+'_Side_CellArea.csv'))
    
    intersectPoints = pd.read_csv(os.path.join(path, date+'_Side_BeadIntersectionPoints.csv'))
    
    listOfCells = (cellCom.Label.str).split('-').str[-2]
    cellID = date + '_' + listOfCells.str[1:]
    
    cellCom['Label'] = cellID
    
    cellID_repeated = cellID.loc[cellID.index.repeat(3)].reset_index(drop=True)
    intersectPoints['Label'] = cellID_repeated
    dfBeads = intersectPoints.drop_duplicates(subset='Label', keep='first')
    
    
    intersectPoints['occurrence'] = intersectPoints.groupby('Label').cumcount() + 1
    dfActClosest = intersectPoints[intersectPoints['occurrence'] == 2].drop(columns='occurrence')
    dfActFarthest = intersectPoints.drop_duplicates(subset='Label', keep='last')
    ##### B is always the vertex, the center of the circle #########
    
    A = (dfBeads.X.values, dfBeads.Y.values)
    B = (cellCom.XM.values, cellCom.YM.values)
    C = (dfActClosest.X.values, dfActClosest.Y.values)
    D = (dfActFarthest.X.values, dfActFarthest.Y.values)
    
    angle_beads = angle_between_points(A, B, C)
    angle_act = angle_between_points(D, B, C)
    angle_theta = np.round((angle_act / 2  + angle_beads), 2)

    
    dateCell = date + '_P' + cellID.str.split('P').str[-1]
    dfToSave = pd.DataFrame({'cellID':cellID.values, 
                             'dateCell':dateCell.values, 
                             'angle_beads' : angle_beads,
                             'angle_act' : angle_act,
                             'angle_theta' : angle_theta})
    
    dfToSave.to_csv(os.path.join(path, date+'_Side_ComputedAngles.csv'), sep = ';', index=False)
    
    
#%% Creating merged files for compression number, angle between beads and activation region

path = 'D:/Anumita/MagneticPincherData/Data_Polarization/AngleEachCompression'
rawPath = 'D:/Anumita/MagneticPincherData/Raw'
savePath = 'D:/Anumita/MagneticPincherData/Data_Polarization'

allCells = os.listdir(path)
allCells = np.array(['_'.join(i.split('_')[:-1]) for i in allCells if '.csv' in i])

angleInfoCell = pd.DataFrame({'cellID': [], 
                         'dateCell': [], 
                         'compNum': [],
                         'angle_beads' : [],
                         'angle_act' : [],
                         'angle_theta' : [],
                         'cell_radius' : [],
                         'arc_length' : []
                         })

for cell in allCells:
    date = cell.split('_')[0].replace('-', '.')
    dateCell = date + '_P' + cell.split('P')[-1]  
    
    actArea = pd.read_csv(os.path.join(path, cell + '_ActivationArea.csv'), sep=None, engine='python')
    status = pd.read_csv(os.path.join(rawPath, date, cell + '_disc20um_L50_Status.txt'), header = None, sep = '_')
    cellArea = pd.read_csv(os.path.join(path, cell + '_CellArea.csv'), sep=None, engine='python')
    beadPos = pd.read_csv(os.path.join(path, cell + '_BeadPos.csv'), sep=None, engine='python')
    beadPos = beadPos[beadPos['Slice'].apply(lambda x : x in np.unique(cellArea['Slice'].values))]

    # Looking for the compression numbers of the slices where ROIs were drawn
    compNums = status[0].iloc[cellArea.Slice]
    nearestAct = actArea.drop_duplicates(subset='Slice', keep='first')
    furthestAct = actArea.drop_duplicates(subset='Slice', keep='last')

    beadPos = beadPos.groupby("Slice").mean()

    A = (nearestAct.X.values, nearestAct.Y.values)
    B = (cellArea.XM.values, cellArea.YM.values)
    C = (beadPos.XM.values, beadPos.YM.values)
    D = (furthestAct.X.values, furthestAct.Y.values)
    
    angle_beads = angle_between_points(A, B, C)
    angle_act = angle_between_points(D, B, C)
    angle_theta = np.round((angle_act / 2  + angle_beads), 2)
    cell_radius = np.sqrt((cellArea.Area.values/np.pi)) / 15.38 #in microns
    arc_length = np.round(2*np.pi*cell_radius*(angle_theta/360), 3)
    
    columnstoAdd = pd.DataFrame({'cellID': [cell]*len(beadPos), 
                             'dateCell': [dateCell.replace('.', '-')]*len(beadPos), 
                             'compNum': compNums,
                             'angle_beads' : angle_beads,
                             'angle_act' : angle_act,
                             'cell_radius' : cell_radius,
                             'angle_theta' : angle_theta,
                             'arc_length' : arc_length,
                             })

    
    angleInfoCell = pd.concat([angleInfoCell, columnstoAdd], ignore_index=True)

angleInfoCell.to_csv(os.path.join(savePath, date+'_Side_ComputedAngles.csv'), sep = ';', index=False)