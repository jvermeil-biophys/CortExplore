# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 12:10:21 2024

@author: anumi
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt

#%%

path = 'D:/Anumita/MagneticPincherData/Data_DoxyDoseResponse/24-08-02_DoseResponseDoxy_3t3-uth-cry2_20x'
bckgd = pd.read_csv(os.path.join(path, 'BackgroundValues_Montage.csv'))
bckgd['Condition'] = ['0.1x', '1x-a', '1x-b', '1x-c', '10x']

allHistos = []
for i in os.listdir(path):
    if 'Histogram' in i:
        
        cond = i.split('_')[-1]
        cond = cond[:-4]
        print(cond)
        histo = pd.read_csv(os.path.join(path, i), sep = '\t', index_col=False)
        histo['Condition'] = [cond]*len(histo)
        print(bckgd.Mean[bckgd.Condition == cond].values)
        histo['Mean'] = histo['Mean'].values - bckgd.Mean[bckgd.Condition == cond].values
        allHistos.append(histo)
        
allHistos = pd.concat(allHistos)
allHistos = allHistos.reset_index()

#%% Plot Histogram

sns.histplot(data=allHistos, x="Mean", hue = 'Condition', bins = 10, kde = True)

plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

plt.tight_layout()
plt.show()

#%% Expression histograms of experiment 24.08.26

path = 'D:/Anumita/MagneticPincherData/Raw/24.08.26_Fluo'

allHistos = []
for i in os.listdir(path):
    if '_Results' in i:
        cond = i.split('_')
        cond = cond[1]
        table = pd.read_csv(os.path.join(path, i), sep = '\t', index_col=False)
        table['Condition'] = [cond]*len(table)
        histo_vals = table[:-1]
        histo_vals['Mean_bckgd'] = histo_vals['Mean'] - table.Mean.iloc[-1]
        histo_vals.Mean_bckgd[histo_vals.Mean_bckgd < 0] = 0
        allHistos.append(histo_vals)
        
allHistos = pd.concat(allHistos)
allHistos = allHistos.reset_index()

#%% Plot Histogram


plt.hist(allHistos.Mean_bckgd[allHistos.Condition == 'M1'], bins = 5, label = 'M1 - Low light')
plt.hist(allHistos.Mean_bckgd[allHistos.Condition == 'M2'], bins = 5, label = 'M2 - High light')
plt.hist(allHistos.Mean_bckgd[allHistos.Condition == 'M3'], bins = 5, label = 'M3 - High light + Activation')


plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

plt.tight_layout()
plt.legend()
plt.show()
