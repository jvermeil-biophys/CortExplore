# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:53:22 2024

@author: EE
"""
#%% import

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import seaborn as sb

import CortexPaths as cp

#%%
# import utility functions
import UtilityFunctions as ufun
import TrackAnalyser as taka
#import getIntensity as GI
#%% import data
DirDataTimeseries_d = "D:/Eloise/MagneticPincherData/Data_Timeseries/24-04-24_data"
allTimeSeriesDataFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                          if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv"))]
print(allTimeSeriesDataFiles)





#%% use plot from TrackAnalyser
taka.plotCellTimeSeriesData('24-04-24_M1_P1_C2_disc20um-02')

#%% get experimental conditions from csv file
ExpConditions=ufun.getExperimentalConditions(DirExp = cp.DirRepoExp, save = False, suffix = cp.suffix)
ExpConditions=ExpConditions.to_numpy()

#%% get the list of thickness of cortex according to the beads sizes
df = taka.getCellTimeSeriesData('24-04-24_M1_P1_C11_disc20um-01')
bead_size=stat.mean([ExpConditions[1,14],ExpConditions[1,18]])*0.001
data_ex=df.to_numpy()
shape=np.shape(data_ex)
h_ex=data_ex[:,10]-bead_size



fig=plt.subplot()
fig.set_title('Cortex thickness')
plt.scatter(data_ex[:,2],h_ex)
plt.xlabel('time (s)')
plt.ylabel('thickness (µm)')

h_ex_mean=stat.mean(h_ex)

#h_means=np.zeros((len(allTimeSeriesDataFiles),2))
h_means=np.ones((len(allTimeSeriesDataFiles),1))
#h_means[:,1]=[allTimeSeriesDataFiles[i][12:18] for i in range(len(allTimeSeriesDataFiles))]
i=0
for file in allTimeSeriesDataFiles:
    df = taka.getCellTimeSeriesData(file[:-7])
    #h_means[i,1]=file[16:18]
    data=df.to_numpy()
    shape=np.shape(data)
    h=data[:,10]-bead_size
    h_mean=stat.mean(h)
    h_means[i,0]=h_mean
    i+=1
    
fig.clear()
fig=plt.subplot()
fig.set_title('Cortex thickness')
plt.boxplot(h_means)
plt.ylabel('thickness (µm)')
X=np.ones(np.shape(h_means))
plt.scatter(X,h_means)

#%% plot D2, dz, D3 and thickness in subplots and save
bead_size=stat.mean([ExpConditions[1,14],ExpConditions[1,18]])*0.001


#file=allTimeSeriesDataFiles[1]
for file in allTimeSeriesDataFiles :
    dataf = taka.getCellTimeSeriesData(file[:-7])
    #data=dataf.to_numpy()
    cellID=ufun.findInfosInFileName(file, 'cellID')
    
    dz_data = dataf['dz']
    D2_data = dataf['D2']
    D3_data = dataf['D3']
    h_data = D3_data - bead_size
    T= dataf['T']
    
    
    fig,axs=plt.subplots(4)
    axs[0].plot(T,D2_data)
    axs[0].scatter(T,D2_data,marker='.')
    axs[0].set(ylabel='D2 (µm)')
    
    axs[1].plot(T,dz_data)
    axs[1].scatter(T,dz_data,marker='.')
    axs[1].set(ylabel='dz (µm)')
    
    axs[2].plot(T,D3_data)
    axs[2].scatter(T,D3_data,marker='.')
    axs[2].set(ylabel='D3 (µm)')
    
    
    axs[3].plot(T,h_data)
    axs[3].scatter(T,h_data,marker='.')
    axs[3].set(ylabel='thickness (µm)')
    fig.suptitle('Data for '+cellID)
    plt.xlabel('Time (s)')
    fig.set_size_inches(10, 11)
    plt.savefig(cp.DirDataTimeseries + '/' + file[:-4] + '_plot.jpg')
    plt.close()

#%% get every h data for each cell
bead_size=stat.mean([ExpConditions[1,14],ExpConditions[1,18]])*0.001

h_storage=pd.DataFrame(columns=allTimeSeriesDataFiles)
h_means=pd.DataFrame()
for file in allTimeSeriesDataFiles :
    dataf = taka.getCellTimeSeriesData(file[:-7])
    cellID=ufun.findInfosInFileName(file, 'cellID')
    
    
    D3_data = dataf[['D3']]
    h_data = D3_data - bead_size
    h_storage[file]=h_data
    name_cell=cellID[12:]+file[-10:-7]
    h_storage=h_storage.rename(columns={file:name_cell})
    h_means[name_cell]=[h_storage[name_cell].mean()]

data=h_means.T
data['index']=data.index
sb.boxplot(data=h_means.T).set_title('Cortex thickness (µm)')
sb.swarmplot(data,hue='index')

#%%
fluctu=pd.DataFrame(index=['mean','max','min','range'],columns=h_storage.columns)
for j in h_storage.columns :
    fluctu.at['mean',j]=stat.mean(h_storage[j])
    fluctu.at['max',j]=max(h_storage[j])
    fluctu.at['min',j]=min(h_storage[j])
    fluctu.at['range',j]=fluctu.at['max',j]-fluctu.at['min',j]

#%% plot fluctu
p1=sb.scatterplot(fluctu.T['mean'],color='k')
p2=sb.scatterplot(fluctu.T['max'],color='r')
p3=sb.scatterplot(fluctu.T['min'],color='b')

#%%
from getIntensity import intensityCellv2
#%% get info of fluo
path_crop="D:/Eloise/MagneticPincherData/Raw/24.04.24_fluo/crop"
path="D:/Eloise/MagneticPincherData/Raw/24.04.24_fluo"
cell='24-04-24_M1_P1_C1'
fileInfotxt=cell +'_disc20um-01.czi_info.txt'
allcrops=os.listdir(path_crop)

cells_fluo=pd.DataFrame()

for j in range(len(allcrops)):
    name=allcrops[j][12:-19] 
    test =intensityCellv2(allcrops[j],fileInfotxt,path_crop,path)
    cells_fluo=pd.concat([cells_fluo,test])

#%%
p4=sb.scatterplot(fluctu.T['mean'],color='k')
p5=sb.scatterplot(fluctu.T['max'],color='k')
p6=sb.scatterplot(fluctu.T['min'],color='b')
#%%
# cells_h=fluctu.T
# for i in fluctu.columns:
#     if i in cells.index:
#         cells_h.at[i,'Color']=cells.at[i,'Color']
# p7=sb.scatterplot(x=cells_h.index,y=cells_h['mean'],hue=cells_h['Color'], palette=['g','r'])

    

#%%
path_fluo='D:/Eloise/MagneticPincherData/Raw/24.04.24_fluo'
data_fluo=pd.read_csv(os.path.join(path_fluo,'data_fluo_24-04.txt'),sep='\t')
#%%
data_fluo.loc[data_fluo['Color'] =='Green', ['phase']] = 'S/G2'
data_fluo.loc[data_fluo['Color'] =='Red', ['phase']] = 'G1'
data_fluo.loc[data_fluo['Color'] =='Yellow', ['phase']] = 'G1/S'

#%%
hmean=h_means.T
hmean=hmean.rename({0:'mean_h'},axis='columns')
#%%
for i in data_fluo.index :
    for j in hmean.index :
        if data_fluo.at[i,'name'][:-3]==j[:-3]:
            hmean.at[j,'Color']=data_fluo.at[i,'Color']
            hmean.at[j,'phase']=data_fluo.at[i,'phase']
            hmean.at[j,'ratio G/R']=data_fluo.at[i,'ratio G/R']
hmean.at['P1_C12-02','phase']='M'
#hmean.at['P1_C1-01','phase']='M'
#%%
p1=sb.boxplot(hmean,x='phase',y='mean_h',hue='phase',palette=['g','limegreen','r','gold'])
sb.swarmplot(hmean,x='phase',y='mean_h',hue=hmean.index)
p1.set_title('Data 24-04 confocal')

#%%
hmean.to_csv(path + '/data_24-04.txt', sep='\t')
    

