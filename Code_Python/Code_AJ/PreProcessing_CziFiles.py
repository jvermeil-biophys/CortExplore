# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:34:50 2023

@author: anumi
"""

import numpy as np
import os
import CortexPaths as cp
import calendar, time
from datetime import datetime


#%% (1) Create Field Files for all cells
"""
This part of the code generates a Field file for all .czi files obtained from Zen.
A Field file contains the magnetic field strength, the exact imaging time of the image and the Z-position.

This file is important in the analysis, mainly the 'time' column. 
The time is extracted by going through the CZI metadata for each image.

"""
filesPath = 'D:/Anumita/23.11.22 HeLaFucci/23.11.22_ConstantField'
fieldValue = 5.0
date = '23.11.22'

rawFilesPath = 'D:/Anumita/Raw'

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


#%% (2) Create folders for each field of view during Cannonball experiment

"""
This part of the code creates separate folders for each cell and places the .czi files in their respective folders.
This helps because we use FIJI after this to flatten the 3D stack to a 2D stack i.e. only in time.

We use the 'Import as Image Sequence...' function of FIJI which automatically opens the file as a 2D stack and it can
only open files that are placed in folders. It does not open individual .czi or .tif files.

Hence, this function to place all the files in separate folders.

"""

filesPath = 'D:/Anumita/23.11.22 HeLaFucci/23.11.22_ConstantField'

allFiles = os.listdir(filesPath)
allFiles = np.asarray([i for i in allFiles if i.endswith('.czi')])

allRegs = np.unique(np.asarray([i.split('.czi')[0] for i in allFiles]))

for i in allRegs:
    if not os.path.exists(os.path.join(filesPath, i)):
        os.mkdir(os.path.join(filesPath, i))
    selectedFiles = [k for k in allFiles if i in k]
    for j in selectedFiles:
        os.rename(os.path.join(filesPath, j), os.path.join(filesPath + '/' + i, j))
        
#%% (3) Open FIJI, run the CZItoTIFF.ijm to convert the .czi files to individual tif images
"""
Make sure to save the TIF images in a separate folder like : "23-11-22_ConstantField_TIFs"

"""
        
#%% (4) Run ImageProcessing.py to crop files
"""
After this step, you are ready to run the ImageProcessing.py code to crop the files. This step is mainly important
data storage saving. 
Save the cropped images in a separate folder with just the date as the name : "23.11.22"

"""

#%% (5) Create Results.txt files for all the cropped images.
"""
Make sure to check the scale is removed on FIJI (i.e. Analyze > Set Scale > Click to Remove Scale..)
Save the results file in the same folder as above : "23.11.22"

"""
 