# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 19:06:27 2023

@author: anumi
"""

import numpy as np
import os
import CortexPaths as cp
import calendar, time
from datetime import datetime

filesPath = 'D:/Anumita/23.11.22 HeLaFucci/23.11.22_ConstantField'
fieldValue = 5.0
date = '23.11.22'
savePath = os.path.join(cp.DirDataRaw+'/'+date)
allFiles = os.listdir(filesPath)
allComp = np.unique(np.asarray([i.split('.czi')[0] for i in allFiles]))
utc_time_epoch = datetime.utcfromtimestamp(0)

for j in allComp:
    fieldFile = []
    files = [i for i in allFiles if j in i]
    # cellPath = os.path.join(filesPath,j)
    for file in files:
        print(file)
        with open(os.path.join(filesPath,file), errors="ignore") as f:
            lines = str(f.readlines())
            sub1 = "<METADATA><Tags><AcquisitionTime>"
            split_lines = np.asarray(lines.split(sub1)[1:])
            times = np.asarray([split_lines[i][0:25] for i in range(len(split_lines))])
            times = np.asarray([(datetime.strptime(times[i], '%Y-%m-%dT%H:%M:%S.%f') - utc_time_epoch) for i in range(len(times))])
            times = [int(timestamp.total_seconds() * 1000.0) for timestamp in times]
        
            sub2 = "</DetectorState><FocusPosition>"
            split_lines = np.asarray(lines.split(sub2)[1:])
            zposition = np.asarray([split_lines[i][11:18] for i in range(len(split_lines))])
            
            magField = np.asarray([fieldValue]*len(times))
            cols = fieldFile.extend(np.float64(np.asarray([magField, times, magField, zposition]).T))
            
    fieldFile = np.asarray(fieldFile)        
    np.savetxt(savePath + '/' + j + '_disc20um_thickness_Field.txt', fieldFile,  fmt='%2.2f\t%2.2f\t%2.2f\t%2.2f')
   # np.savetxt(savePath + '/' + j + '_Field.txt', fieldFile,  fmt='%2.2f\t%2.2f\t%2.2f\t%2.2f')


#%% Placing .czi files in their right folders to make import to fiji easy

allFiles = os.listdir(filesPath)
allFiles = np.asarray([i for i in allFiles if i.endswith('.czi')])

allCells = np.unique(np.asarray([i.split('_disc20um')[0] for i in allFiles]))

for i in allCells:
    if not os.path.exists(os.path.join(filesPath, i)):
        os.mkdir(os.path.join(filesPath, i))
    selectedFiles = [k for k in allFiles if i in k]
    for j in selectedFiles:
        os.rename(os.path.join(filesPath, j), os.path.join(filesPath + '/' + i, j))