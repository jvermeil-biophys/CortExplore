# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:49:32 2022

@author: anumi
"""

import os
import numpy as np


path = 'D:/Anumita/MagneticPincherData/Raw/22.12.07/M5_P3'
allFiles = os.listdir(path)

newNames = np.asarray([i.replace(i.split('_')[1], 'M6') for i in allFiles])

for i in range(len(allFiles)):
    os.rename(os.path.join(path, allFiles[i]), os.path.join(path, newNames[i]))
