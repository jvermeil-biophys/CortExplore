# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:52:58 2023

@author: anumi
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import CortexPaths as cp
import UtilityFunctions as ufun
import matplotlib.pyplot as plt

expDf = ufun.getExperimentalConditions(DirExp = cp.DirRepoExp, save = True, suffix = cp.suffix)

#%% Functions

def makeMergedDf(expDf, date):

    dict1 = {'cellID' : [],
            'medianThickness' : [],
            'fluctuations' : [],
            'cell phase' : [],
            'color' : []}
    
    df_final = pd.DataFrame(dict1)
    tsPath = os.path.join(cp.DirData, 'Data_TimeSeries')
    allFiles = os.listdir(tsPath)
    date2 = date.replace('.', '-')
    allFiles = [tsPath+'/'+i for i in allFiles if date2 in i]
    # print(allFiles)
    
    expDf = expDf[expDf['date'] == date2]
    pathCC = 'C:/Users/anumi/OneDrive/Desktop/CortExplore/Data_Experimental_AJ/HeLa_Fucci'
    ccdf = pd.read_csv(os.path.join(pathCC, date2 + '_CellCyclePhase.csv'), sep=None, engine = 'python')
    
    for file in allFiles:
        print(file)
        cellID = file.split('Data_TimeSeries/')[1]
        cellID = cellID.split('_disc20um')[0]
        
        tsdf = pd.read_csv(file, sep = ';')
        beadDia = expDf['bead diameter'].values.astype(float)[0] / 1000
        
        ts = (tsdf['D3'] - beadDia)
        medianThickness = np.round(ts.median(), 4)
        
        fluct_ampli_90 = np.percentile(ts.values, 90)
        fluct_ampli_10 = np.percentile(ts.values, 10)
        fluctuations = fluct_ampli_90 - fluct_ampli_10
        
        cellphase = ccdf['phase'][ccdf['cellID'] == cellID]
        cellcolor = ccdf['color'][ccdf['cellID'] == cellID]
        
        df_results = pd.DataFrame({'cellID' : cellID,
                'medianThickness' : medianThickness,
                'fluctuations' : fluctuations,
                'cell phase' : cellphase,
                'color' : cellcolor})
        
        df_results = df_results[df_results['medianThickness'] < 1.2]
        
        df_final = pd.concat([df_final, df_results])
    return df_final
    
#%% Plots


#%%%% 23.10.31 : Constant field, with Pelin Sar

df = makeMergedDf(expDf, '23.10.31')
y = 'medianThifckness'
x = 'cell phase'

sns.swarmplot(data = df, x = x, y = y, 
              linewidth = 1, color = 'k', edgecolor='k') 

sns.boxplot(data = df, x = x, y = y,
            medianprops={"color": 'darkred', "linewidth": 2},\
            boxprops={"edgecolor": 'k',"linewidth": 2, 'alpha' : 1})
    
plt.ylim(0, 1)

# sns.lmplot(data = df, x = 'medianThickness', y = 'fluctuations', hue = 'cell phase')    