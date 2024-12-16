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

pltTicks = {'color' : '#ffffff', 'fontsize' : 18}

#%% 24.11.07

date = '24.11.07'
DirData = 'D:/Anumita/MagneticPincherData/Data_Cloning/24.11.07_Montages'
allFiles = os.listdir(DirData)
allFiles = [i for i in allFiles if '.txt' in i and not 'E2' in i]
allTxtFiles = pd.DataFrame()
background = {'HE.B5' : [312.649], 'LE.E2' : [298.090], 'LE.E5' : [274.423]}
background = pd.DataFrame.from_dict(background)

for i in allFiles:
    clone = i.split('_')[0]
    read = pd.read_csv(os.path.join(DirData, i), sep = '\t')
    read['clone'] = [date + '_' + clone]*len(read)
    read['bkgd'] = [background[clone].values[0]]*len(read)
    read['Mean_bckgd'] = read['Mean'] - read['bkgd']
    allTxtFiles = pd.concat([allTxtFiles, read])
    
allTxtFiles_1 = allTxtFiles.reset_index()

plt.style.use('default')
fig, ax = plt.subplots(figsize = (13,9))
fig.patch.set_facecolor('black')

sns.histplot(data=allTxtFiles, x='Mean_bckgd', hue = 'clone', kde=True)

plt.xlim(0, 2000)

plt.xticks(**pltTicks)
plt.yticks(**pltTicks)
plt.show()

#%% 24.11.14

date = '24.11.14'

DirData = 'D:/Anumita/MagneticPincherData/Data_Cloning/24.11.14_Montages'
allFiles = os.listdir(DirData)
allFiles = [i for i in allFiles if '.txt' in i and not 'Antibiotics' in i and not 'E1' in i]
allTxtFiles = pd.DataFrame()
bkgd = pd.read_csv(os.path.join(DirData, 'BackgroundValues.csv'), sep=';')


for i in allFiles:
    clone = i.split('_')[0]
    read = pd.read_csv(os.path.join(DirData, i), sep = '\t')
    read['clone'] = [date + '_' + clone]*len(read)
    read['bkgd'] = [bkgd['value'][(bkgd['clone']==clone)].values[0]]*len(read)
    read['Mean_bckgd'] = read['Mean'] - read['bkgd']
    allTxtFiles = pd.concat([allTxtFiles, read])
    
allTxtFiles_2 = allTxtFiles.reset_index()

plt.style.use('default')
fig, ax = plt.subplots(figsize = (13,9))
fig.patch.set_facecolor('black')

sns.histplot(data=allTxtFiles, x='Mean_bckgd', hue = 'clone', kde=True)
plt.xlim(0, 2000)

plt.xticks(**pltTicks)
plt.yticks(**pltTicks)
plt.show()


#%% Merged 

palette = ['#99cc99',  '#008000'] #['#ccccff', '#0000FF'] #, '#99cc99',  '#008000']
finalTxt = pd.concat([allTxtFiles_1, allTxtFiles_2])
plt.style.use('default')
fig, ax = plt.subplots(figsize = (13,9))
fig.patch.set_facecolor('black')


bins = 20

finalTxt = finalTxt[finalTxt['clone'].str.contains('E5')]

sns.histplot(data=finalTxt, x='Mean_bckgd' ,  palette = palette,
              hue = 'clone',   stat = 'count', bins = bins)

# sns.kdeplot(data=finalTxt, x='Mean_bckgd', palette = palette,
#               hue = 'clone', multiple="stack", alpha=.5, linewidth=0)

# sns_plot = sns.displot(finalTxt, x='Mean_bckgd',  palette = palette,
#               hue = 'clone', kde=True, alpha = 0.6)

plt.xlim(0, 800)

plt.xticks(**pltTicks)
plt.yticks(**pltTicks)
plt.show()