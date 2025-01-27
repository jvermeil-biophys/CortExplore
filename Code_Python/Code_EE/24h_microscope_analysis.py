# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:43:06 2024

@author: BioMecaCell
"""

#%% import
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statistics as stat
import matplotlib.colors as mcolors
import matplotlib as mpl
import os
import sys

from scipy.optimize import curve_fit
import UtilityFunctions as ufun
#%% load data
analysis = 'series3_cells3-1'
path='D:/Eloise/Hela_FUCCI_24h_microscopeDamien/240704_helaFUCCI_IPGG'
dataRed=pd.read_csv(os.path.join(path,analysis+'_red.txt'), sep='\t')
dataRed=dataRed.rename({' ':'i'},axis='columns')
dataGreen=pd.read_csv(os.path.join(path,analysis+'_green.txt'), sep='\t')
dataGreen=dataGreen.rename({' ':'i'},axis='columns')
#%%
cell3=pd.DataFrame()
#%%

cell3=dataRed.merge(dataGreen,'outer',on='Frame',suffixes=('_red','_green'))
#%%

p1=cell3.plot.scatter(x='Frame',y='Mean_green',c='g')
cell3.plot.scatter(x='Frame',y='Mean_red',ax=p1,c='r')

#%%
analysis = 'series9_cells1-1'
path='D:/Eloise/Hela_FUCCI_24h_microscopeDamien/240704_helaFUCCI_IPGG'
dataRed2=pd.read_csv(os.path.join(path,analysis+'_red.txt'), sep='\t')
dataRed2=dataRed2.rename({' ':'i'},axis='columns')
dataGreen2=pd.read_csv(os.path.join(path,analysis+'_green.txt'), sep='\t')
dataGreen2=dataGreen2.rename({' ':'i'},axis='columns')

cell2=dataRed2.merge(dataGreen2,'outer',on='Frame',suffixes=('_red','_green'))

#%%
p2=cell2.plot.scatter(x='Frame',y='Mean_green',marker='.',c='g')
cell2.plot.scatter(x='Frame',y='Mean_red',ax=p2,marker='.',c='r')
#p2.set_xlim(0,160)
#%%

maxRed=np.max(cell2['Mean_red'])
idmaxR=cell2['Mean_red'].idxmax()
maxGreen=np.max(cell2['Mean_green'])
idmaxG=cell2['Mean_green'].idxmax()

#%%
cellcrop=cell2[(cell2.index > idmaxR) & (cell2.index<idmaxG)]
p3=cellcrop.plot.scatter(x='Frame',y='Mean_green',marker='.',c='g')
cellcrop.plot.scatter(x='Frame',y='Mean_red',ax=p3,marker='.',c='r')


#%%
path='D:/Eloise/Hela_FUCCI_24h_microscopeDamien/240704_helaFUCCI_IPGG'
i=9
j=1
analysis='series'+str(i)+'_cells'+str(j)
dataRed=pd.read_csv(os.path.join(path,analysis+'-1_red.txt'), sep='\t')
dataRed=dataRed.rename({' ':'i'},axis='columns')
dataRed2=pd.read_csv(os.path.join(path,analysis+'-2_red.txt'), sep='\t')
dataRed2=dataRed2.rename({' ':'i'},axis='columns')
dataGreen=pd.read_csv(os.path.join(path,analysis+'-1_green.txt'), sep='\t')
dataGreen=dataGreen.rename({' ':'i'},axis='columns')
dataGreen2=pd.read_csv(os.path.join(path,analysis+'-2_green.txt'), sep='\t')
dataGreen2=dataGreen2.rename({' ':'i'},axis='columns')

cell1=dataRed.merge(dataGreen,'outer',on='Frame',suffixes=('_red','_green')) 
cell2=dataRed2.merge(dataGreen2,'outer',on='Frame',suffixes=('_red','_green'))

cellID=cell2
cellID=cellID[(cellID['Frame']>30)]
maxRed=np.max(cellID['Mean_red'])
idmaxR=cellID['Mean_red'].idxmax()
maxGreen=np.max(cellID['Mean_green'])
idmaxG=cellID['Mean_green'].idxmax()
cellcrop=cellID[(cellID.index > idmaxR) & (cellID.index<idmaxG)]
        
 
#%%
def func(x,a,b):
    return a*x+b
#params,res=ufun.fitLine(cellcrop['Frame'],cellcrop['Mean_green'])
cellcrop1=cellcrop[cellcrop['Mean_green'].isna()==False]
popt, pcov = curve_fit(func, cellcrop1['Frame'], cellcrop1['Mean_green'])
#%%
fit_x = np.linspace(np.min(cellcrop1['Frame']), np.max(cellcrop1['Frame']), 50)
params,res=ufun.fitLine(cellcrop1['Frame'],cellcrop1['Mean_green'])
fit_y1=params[1]*fit_x+params[0]

fit_x = np.linspace(np.min(cellcrop1['Frame']), np.max(cellcrop1['Frame']), 50)
fit_y = popt[0] * fit_x+popt[1]
#%%
plt.figure()
plt.plot(fit_x,fit_y1,'k')
plt.plot(cellcrop['Frame'], cellcrop['Mean_green'])
plt.plot(cellcrop['Frame'], cellcrop['Mean_red'])


#%%
plt.figure()
path='D:/Eloise/Hela_FUCCI_24h_microscopeDamien/240704_helaFUCCI_IPGG'
allFiles = [f for f in os.listdir(path) if f.endswith(".txt")]
fitparams=pd.DataFrame()
fitparams.index=[allFiles[j][:16] for j in range(len(allFiles))]
fig,(ax1,ax2) = plt.subplots(1, 2)
#colors=['b','g','r','c','m','y','k','orange','lightgreen','gold','tomato','navy','pink','sienna']

colors = mpl.colormaps['tab20'].colors
for j in range(len(allFiles)):
    
    cellID=allFiles[j][:16]
    print(cellID)
    cellN=pd.DataFrame()
    
    
    dataRed=pd.read_csv(os.path.join(path,cellID+'_red.txt'), sep='\t')
    dataRed=dataRed.rename({' ':'i'},axis='columns')
    dataGreen=pd.read_csv(os.path.join(path,cellID+'_green.txt'), sep='\t')
    dataGreen=dataGreen.rename({' ':'i'},axis='columns')
    
    cell=dataRed.merge(dataGreen,'outer',on='Frame',suffixes=('_red','_green'))
    cellG=cell[(cell['Frame']>30)]
    maxRed=np.max(cell['Mean_red'])
    idmaxR=cell['Mean_red'].idxmax()
    maxGreen=np.max(cellG['Mean_green'])
    idmaxG=cell['Mean_green'].idxmax()
    cellcrop=cell[(cell.index > idmaxR) & (cell.index<idmaxG)]
    
    cellN=cellcrop
    cellN['time (h)']=cellN['Frame']/12
    cellN['Mean_green']=cellcrop['Mean_green']/maxGreen
    cellN['Mean_red']=cellcrop['Mean_red']/maxRed
    cellcropN=cellN[cellN['Mean_green'].isna()==False]
    params,res=ufun.fitLine(cellcropN['time (h)'],cellcropN['Mean_green'])
    fitparams.at[cellID,'params - a']=params.iloc[1]
    fitparams.at[cellID,'params - b']=params.iloc[0]
    fitparams.at[cellID,'R2']=res.rsquared
    
    ax1.plot(cell['Frame']/12, cell['Mean_green'],color=colors[j])
    ax2.plot(cell['Frame']/12, cell['Mean_red'],color=colors[j])

    
fitparamsR2=fitparams[fitparams['R2']>0.9]
slope=np.median(fitparamsR2['params - a'])
plt.show()

#%%
fitparams2=fitparams.iloc[::2, :]
fitparams2.to_csv(path + '/fitparams.txt', sep='\t')

#%%


fit_x = np.linspace(np.min(cellN['time (h)']), np.max(cellN['time (h)']), 50)
fit_y1=params[1]*fit_x+params[0]
plt.figure()
plt.plot(cell['Frame']/12, cell['Mean_green']/maxGreen,'g')
plt.plot(cell['Frame']/12, cell['Mean_red']/maxRed,'r')
plt.plot(fit_x,fit_y1,'k')
plt.title('Fluo intensity')
plt.xlabel('Time (h)')
plt.show()