# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:55:58 2024

@author: anumi
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import CortexPaths as cp
import  matplotlib.pyplot as plt


#%%

DirData = 'D:/Anumita/MagneticPincherData/Data_Cloning/24.11.07_Montages'
allFiles = os.listdir(DirData)
allFiles = [i for i in allFiles if '.txt' in i]
allTxtFiles = pd.DataFrame()


for i in allFiles:
    clone = i.split('_')[0]
    read = pd.read_csv(os.path.join(DirData, i), sep = '\t')
    read['clone'] = [clone]*len(read)
    allTxtFiles = pd.concat([allTxtFiles, read])
    
allTxtFiles = allTxtFiles.reset_index()

#%%

sns.histplot(data=allTxtFiles, x='Mean', hue = 'clone', kde=True)
plt.xlim(0, 2000)
plt.show()