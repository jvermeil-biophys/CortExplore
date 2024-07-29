# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:31:46 2024

@author: EE


"""

#%% import
import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPythonUser)

import numpy as np
import os
import czifile as czi
import matplotlib.pyplot as plt
import getInfoFromCZI as giczi

#%%
import UtilityFunctions as ufun
import cv2 as cv
import pandas as pd
#import seaborn.objects as so
import seaborn as sns
import statistics as stat
#%%

#%%
# path="E:/Eloise/Data/24-04-24-fuccihela-constantfield-100x-4.5fibrostep/24-04-24/24-04-24_fluo"
# all_files=os.listdir(path)
# file=all_files[0]
# image=czi.imread(os.path.join(path,file))

#%% look at the image

# channels=np.shape(image)[0]
# #for i in range(channels):
# plt.imshow(image[0])

#%% get channels and plots histos
#
# exp_cond=giczi.exptimeChannel(os.path.join(path,file))
# histos=pd.DataFrame()
# for i in range(channels):
#     histo,bins=np.histogram(image[i],bins=65536)
#     histos[i]=[histo,bins]
    
# fig,axs=plt.subplots(2)
# axs[0].plot(histos[0][1][0:-1],histos[0][0],color='g')
# axs[0].set(ylabel=exp_cond.columns[0])


# axs[1].plot(histos[1][1][0:-1],histos[1][0],color='r')
# axs[1].set(ylabel=exp_cond.columns[1])
# fig.suptitle('histograms of fluo images')

#%% "normalize" by exposure time
# intensity_norm=pd.DataFrame()
# for i in range(len(histos)):
#     intensity_norm[i]=histos[i]/exp_cond[exp_cond.columns[i]].iloc[0]
    
# fig,axs=plt.subplots(2)
# axs[0].plot(intensity_norm[0][1][0:-1],intensity_norm[0][0],color='g')
# axs[0].set(ylabel=exp_cond.columns[0])


# axs[1].plot(intensity_norm[1][1][0:-1],intensity_norm[1][0],color='r')
# axs[1].set(ylabel=exp_cond.columns[1])
# fig.suptitle('histograms of fluo images normalized by exposure time')

#%% get cropped images

path_crop="D:/Eloise/MagneticPincherData/Raw/24.04.24_fluo/crop"
path="D:/Eloise/MagneticPincherData/Raw/24.04.24_fluo"
all_dircrops=os.listdir(path_crop)

#%% plot cropped images
crop=os.path.join(path_crop,all_dircrops[0])
#ig_g=plt.imread(crop)
#plt.imshow(ig_g)

#%% get green and red mean and make a ratio

ratios=[]
intens=pd.DataFrame()

for j in range(len(all_dircrops)):
    images_folder=os.path.join(path_crop,all_dircrops[j])
    images_crop=os.listdir(os.path.join(path_crop,images_folder))
    file=all_dircrops[j][:-5]
    channels=len(images_crop)
    histos=pd.DataFrame()
    exp_cond=giczi.exptimeChannel(os.path.join(path,file))
    images_rg=[]
    cellID=ufun.findInfosInFileName(file, 'cellID')
    name_cell=cellID[12:]+file[-7:-4]
    
    for i in range(channels):
        images_rg.append(plt.imread(os.path.join(images_folder,images_crop[i])))
        histo,bins=np.histogram(images_rg[i],bins=65536,range=(images_rg[i][images_rg[i] != 0].min(),images_rg[i].max()))
        histos[i]=[histo,bins]
    # fig,axs=plt.subplots(2)
    # axs[0].plot(histos[0][1][0:-1],histos[0][0],color='g')
    # axs[0].set(ylabel=exp_cond.columns[0])


    # axs[1].plot(histos[1][1][0:-1],histos[1][0],color='r')
    # axs[1].set(ylabel=exp_cond.columns[1])
    # fig.suptitle('histograms of fluo images '+name_cell)
    
    mean_g=images_rg[0][images_rg[0] != 0].mean()
    mean_r=images_rg[1][images_rg[1] != 0].mean()
    intens[name_cell]=[mean_g, mean_r]
    intens.index=[exp_cond.columns[i] for i in range(len(exp_cond.columns))]
   
    ratio=mean_g/mean_r
    ratios.append(ratio)
#%% background

#%% figure log
plt.figure()
p1 = so.Plot(intens.T,x='EGFP', y='DsRed')
p1.add(so.Dots()).scale(x='log',y="log")


intens_data=intens.T
intens_data['name_cell']=intens_data['EGFP']
for name in intens_data.index :
    print(name)
    intens_data.at[name,'name_cell']=name[:-3]
intens_data['ratio G/R']=intens_data['EGFP']/intens_data['DsRed']
#intens_data['ratio R/G']=intens_data['DsRed']/intens_data['EGFP']
intens_data['Color']=intens_data['ratio G/R']
intens_data.loc[intens_data['ratio G/R'] > 4, ['Color']] = 'Green'
intens_data.loc[intens_data['ratio G/R'] < 1, ['Color']] = 'Red'
intens_data.loc[(intens_data['ratio G/R'] > 1) & (intens_data['ratio G/R']<4), ['Color']] = 'Yellow'




intens_data['notes'] = intens_data['Color']
intens_data.at['P1_C1-01','notes']='mitotic'

plt.figure()
#p1=sns.scatterplot(intens_data, x='EGFP', y='DsRed', hue='notes', palette=['r','g','k','gold'])
p1=sns.scatterplot(intens_data, x='EGFP', y='DsRed', hue='Color', palette=['r','g','gold'])

plt.xscale('log')
plt.yscale('log')
plt.ylim(100, 10**5)
plt.xlim(100, 10**5)
# x = 10 ** np.arange(1, 10)
# plt.plot(x,x,color='k')

#p2=plt.scatter(intens_data.EGFP.P1_C7_bg, intens_data.DsRed.P1_C7_bg, color='k')
plt.title('Intensity of red vs green fluo for each cell')


#%% get mean of the two images files for each cell
intens_mean=pd.DataFrame(1, index=np.arange(18), columns=['name cell','EGFP mean','DsRed mean'])

for j in range(0,len(intens_data['name_cell']),2):
    print(intens_data['name_cell'][j])
    intens_mean.at[j//2,'name cell']=intens_data['name_cell'][j]
    intens_mean.at[j//2,'EGFP mean']=stat.mean([intens_data['EGFP'][j],intens_data['EGFP'][j+1]])
    intens_mean.at[j//2,'DsRed mean']=stat.mean([intens_data['DsRed'][j],intens_data['DsRed'][j+1]])

intens_mean['ratio G/R']=intens_mean['EGFP mean']/intens_mean['DsRed mean']
intens_mean['Color']=intens_mean['ratio G/R']
intens_mean.loc[intens_mean['ratio G/R'] > 3, ['Color']] = 'Green'
intens_mean.loc[intens_mean['ratio G/R'] < 0.3, ['Color']] = 'Red'
intens_mean.loc[(intens_mean['ratio G/R'] > 0.3) & (intens_mean['ratio G/R']<3), ['Color']] = 'Yellow'


plt.figure()
p3=sns.scatterplot(intens_mean, x='EGFP mean', y='DsRed mean', hue='Color', palette=['gold','g','r'])
plt.title('Intensity of fluo for each cell (not normalized)')
plt.xscale('log')
plt.yscale('log')
plt.ylim(100, 10**5)
plt.xlim(100, 10**5)


#%% get histo and their mean for each cell (my own data)
def intensity_cell(files, dir_crop, dir_data):
    import getInfoFromCZI as giczi
    intens=pd.DataFrame()
    for j in range(len(files)):
        images_folder=os.path.join(dir_crop,files[j])
        images_crop=os.listdir(os.path.join(dir_crop,images_folder))
        file=files[j][:-5]
        channels=len(images_crop)
        histos=pd.DataFrame()
        exp_cond=giczi.exptimeChannel(os.path.join(dir_data,file))
        images_rg=[]
        cellID=ufun.findInfosInFileName(file, 'cellID')
        name_cell=cellID[12:]+file[-7:-4]
        
        for i in range(channels):
            images_rg.append(plt.imread(os.path.join(images_folder,images_crop[i])))
            histo,bins=np.histogram(images_rg[i],bins=65536,range=(images_rg[i][images_rg[i] != 0].min(),images_rg[i].max()))
            histos[i]=[histo,bins]
        # fig,axs=plt.subplots(2)
        # axs[0].plot(histos[0][1][0:-1],histos[0][0],color='g')
        # axs[0].set(ylabel=exp_cond.columns[0])


        # axs[1].plot(histos[1][1][0:-1],histos[1][0],color='r')
        # axs[1].set(ylabel=exp_cond.columns[1])
        # fig.suptitle('histograms of fluo images '+name_cell)
        
        mean_g=images_rg[0][images_rg[0] != 0].mean()
        mean_r=images_rg[1][images_rg[1] != 0].mean()
        intens[name_cell]=[mean_g, mean_r]
        intens.index=[exp_cond.columns[i] for i in range(len(exp_cond.columns))]
    
    mean_g=images_rg[0][images_rg[0] != 0].mean()
    mean_r=images_rg[1][images_rg[1] != 0].mean()
    intens[name_cell]=[mean_g, mean_r]
    intens.index=[exp_cond.columns[i] for i in range(len(exp_cond.columns))]
    intens_data=intens.T
    intens_data['name_cell']=intens_data['EGFP']
    for name in intens_data.index :
        print(name)
        intens_data.at[name,'name_cell']=name[:-3]
    intens_data['ratio G/R']=intens_data['EGFP']/intens_data['DsRed']
   #intens_data['ratio R/G']=intens_data['DsRed']/intens_data['EGFP']
    intens_data['Color']=intens_data['ratio G/R']
    intens_data.loc[intens_data['ratio G/R'] > 3, ['Color']] = 'Green'
    intens_data.loc[intens_data['ratio G/R'] < 0.3, ['Color']] = 'Red'
    intens_data.loc[(intens_data['ratio G/R'] > 0.3) & (intens_data['ratio G/R']<3), ['Color']] = 'Yellow'
    
    intens_mean=pd.DataFrame(1, index=np.arange(len(intens_data)//2), columns=['name cell','EGFP mean','DsRed mean','ratio','Color'])

    for j in range(0,len(intens_data['name_cell']),2):
        print(intens_data['name_cell'][j])
        intens_mean.at[j//2,'name cell']=intens_data['name_cell'][j]
        intens_mean.at[j//2,'EGFP mean']=stat.mean([intens_data['EGFP'][j],intens_data['EGFP'][j+1]])
        intens_mean.at[j//2,'DsRed mean']=stat.mean([intens_data['DsRed'][j],intens_data['DsRed'][j+1]])
        intens_mean.at[j//2,'ratio']=intens_mean.at[j//2,'EGFP mean']/intens_mean.at[j//2,'DsRed mean']
        intens_mean.loc[intens_mean['ratio'] > 3, ['Color']] = 'Green'
        intens_mean.loc[intens_mean['ratio'] < 0.3, ['Color']] = 'Red'
        intens_mean.loc[(intens_mean['ratio'] > 0.3) & (intens_mean['ratio']<3), ['Color']] = 'Yellow'
    return intens_data, intens_mean
    
#%% input directory

path_crop="D:/Eloise/MagneticPincherData/Raw/24.04.24_fluo/crop"
path="D:/Eloise/MagneticPincherData/Raw/24.04.24_fluo"
cell= '24-04-24_M1_P1_C8'


#%% get intensities of different cells from a folder
alldircrops=os.listdir(path_crop)
intensities=pd.DataFrame()
intensities_m=pd.DataFrame()
for c in range(0,len(alldircrops),2) :
    files=[alldircrops[c],alldircrops[c+1]]
    intens_ex,intens_m=intensity_cell(files,path_crop,path)
    intensities_m=pd.concat([intensities_m,intens_m])
    intensities=pd.concat([intensities,intens_ex])
#%% plot
p5=sns.scatterplot(intensities_m, x='EGFP mean', y='DsRed mean', hue='Color', palette=['gold','g','r'], style='name cell')

plt.xscale('log')
plt.yscale('log')
plt.ylim(100, 10**5)
plt.xlim(100, 10**5)

#%% get intensity of cell from Pelin's data
def intensityCellv2(images,fileInfotxt,dir_crop,dir_file):
    import getInfoFromCZI as giczi
    exp_cond=giczi.metadatafromtxt(fileInfotxt,dir_file)
    for i in exp_cond.index :
        if i=='TL Brightfield':
            exp_cond=exp_cond.drop(index=i)
    intens=pd.DataFrame()
    #for j in range(len(images)):
    images_folder=os.path.join(dir_crop,images)
    images_crop=os.listdir(images_folder)
    file=images[:-5]
    print(file)
    channels=len(images_crop)
    histos=pd.DataFrame()
    images_rg=[]
    cellID=ufun.findInfosInFileName(file, 'cellID')
    name_cell=cellID[12:]+file[-7:-4]
    for i in range(channels):
        images_rg.append(plt.imread(os.path.join(images_folder,images_crop[i])))
        histo,bins=np.histogram(images_rg[i],bins=65536,range=(images_rg[i][images_rg[i] != 0].min(),images_rg[i].max()))
        histos[i]=[histo,bins]
    # fig,axs=plt.subplots(2)
    # axs[0].plot(histos[0][1][0:-1],histos[0][0],color='g')
    # axs[0].set(ylabel=exp_cond.columns[0])


    # axs[1].plot(histos[1][1][0:-1],histos[1][0],color='r')
    # axs[1].set(ylabel=exp_cond.columns[1])
    # fig.suptitle('histograms of fluo images '+name_cell)
    
    # mean_g=images_rg[0][images_rg[0] != 0].mean()
    # mean_r=images_rg[1][images_rg[1] != 0].mean()
    # intens[name_cell]=[mean_g, mean_r]
    # intens.index=[exp_cond.index[i] for i in range(len(exp_cond.index))]

    mean_g=images_rg[0][images_rg[0] != 0].mean()
    mean_r=images_rg[1][images_rg[1] != 0].mean()
    intens[name_cell]=[mean_g, mean_r]
    intens.index=[exp_cond.index[i] for i in range(len(exp_cond.index))]
    intens_data=intens.T
    #intens_data['name_cell']=intens_data.loc['EGFP']
    # for name in intens_data.index :
    #     print(name)
    #     intens_data.at[name,'name_cell']=name[:-3]
    intens_data['ratio G/R']=intens_data['EGFP']/intens_data['DsRed']
   #intens_data['ratio R/G']=intens_data['DsRed']/intens_data['EGFP']
    intens_data['Color']=intens_data['ratio G/R']
    intens_data.loc[intens_data['ratio G/R'] > 2, ['Color']] = 'Green'
    intens_data.loc[intens_data['ratio G/R'] < 0.5, ['Color']] = 'Red'
    intens_data.loc[(intens_data['ratio G/R'] > 0.5) & (intens_data['ratio G/R']<2), ['Color']] = 'Yellow'

    return intens_data
    
#%%
#import getInfoFromCZI as giczi
path_crop="D:/Eloise/MagneticPincherData/Raw/24.04.24_fluo/crop"
path="D:/Eloise/MagneticPincherData/Raw/24.04.24_fluo"
#cell='24-05-16_M1_P2_C1'
allcrops=os.listdir(path_crop)
fileInfotxt=[i[:-5]+'_info.txt' for i in allcrops]

cells=pd.DataFrame()

for j in range(len(allcrops)):
    name=allcrops[j][12:-21]+allcrops[j][-12:-9]
    test= intensityCellv2(allcrops[j],fileInfotxt[j],path_crop,path)
    cells=pd.concat([cells,test])
#%%
# intens_data_max=cells
# inds=intens_data_max.index
# k=0
# while k<len(intens_data_max.index):
#     ind=intens_data_max.index[k]
#     while sum(inds.str.count(ind[:-2]))!=1:
#         print(k)
#         if inds[k][:-2]==intens_data_max.index[k+1][:-2]:
#             print(inds[k][:-2])
#             if intens_data_max.iat[k,0]<intens_data_max.iat[k+1,0]:
#                 print('change')
#                 intens_data_max=intens_data_max.drop(intens_data_max.index[k])
#                 k=k+1
#                 ind=intens_data_max.index[k]
#             else :
#                 print('change 2')
#                 intens_data_max=intens_data_max.drop(intens_data_max.index[k+1])
#                 k=k+1
#                 ind=intens_data_max.index[k]
#         else :
#             k=k+1
#             ind=intens_data_max.index[k]
#     else :
#         k=k+1
#%%
intens=cells.copy()
intens=intens.T
intens_max=pd.DataFrame()
inds=intens.columns
k=0
while k <len(inds):
    ind=inds[k]
    while sum(inds.str.count(ind[:-2]))!=1:
        #print(sum(inds.str.count(ind[:-2])))
        if inds[k][:-2]==intens.columns[k+1][:-2]:
            print(inds[k][:-2])
            if intens.iat[0,k]<intens.iat[0,k+1]:
                print('change')
                intens_max[inds[k][:-2]]=intens[inds[k+1]]
                intens_max.at['name',inds[k][:-2]]=inds[k]
                k=k+1
            else :
                print('change 2')
                intens_max[inds[k][:-2]]=intens[inds[k]]
                intens_max.at['name',inds[k][:-2],]=inds[k]
                k=k+1
        else :
            print('loop')
            intens_max[inds[k][:-2]]=intens[inds[k]]
            intens_max.at['name',inds[k][:-2],]=inds[k]
            k=k+1
    else :
        print(k)
        intens_max[inds[k][:-2]]=intens[inds[k]]
        intens_max.at['name',inds[k][:-2]]=inds[k]
        k=k+1
#%%
cells_max=intens_max.T

#%%
p5=sns.scatterplot(cells_max, x='EGFP', y='DsRed', hue='Color', palette=['gold','g','r'])
plt.title('Intensity of each fluo (not normalized)')
plt.xscale('log')
plt.yscale('log')
plt.ylim(100, 10**5)
plt.xlim(100, 10**5)

#%%
path_crop_bg="D:/Eloise/MagneticPincherData/Raw/24.04.24_fluo/background"
path="D:/Eloise/MagneticPincherData/Raw/24.04.24_fluo"
#cell='24-05-16_M1_P2_C1'
allcrops_bg=os.listdir(path_crop_bg)
fileInfotxt_bg=[i[:-5]+'_info.txt' for i in allcrops_bg]

bg=pd.DataFrame()

for j in range(len(allcrops_bg)):
    name=allcrops_bg[j][12:-17]
    test= intensityCellv2(allcrops_bg[j],fileInfotxt_bg[j],path_crop_bg,path)
    bg=pd.concat([bg,test])
bg=bg.replace('Yellow','bg')
#%%
cells_max=intens_max.T
cells_norm=cells_max
for i in cells_max.index :
    for k in range(len(bg.index)):
        if cells_max.at[i,'name']==bg.index[k]:
            print(k)
            cells_norm.at[i,'EGFP']=(cells_max.at[i,'EGFP']-bg.at[bg.index[k],'EGFP'])/bg.at[bg.index[k],'EGFP']
            cells_norm.at[i,'DsRed']=(cells_max.at[i,'DsRed']-bg.at[bg.index[k],'DsRed'])/bg.at[bg.index[k],'DsRed']
            cells_norm.at[i,'ratio G/R']=cells_norm.at[i,'EGFP']/cells_norm.at[i,'DsRed']
cells_norm.loc[cells_norm['ratio G/R'] > 3, ['Color']] = 'Green'
cells_norm.loc[cells_norm['ratio G/R'] < 0.3, ['Color']] = 'Red'
cells_norm.loc[(cells_norm['ratio G/R'] > 0.3) & (cells_norm['ratio G/R']<3), ['Color']] = 'Yellow' 

#%%
cells_all=cells
cells_norm2=cells_all
for i in cells_all.index :
    for k in range(len(bg.index)):
        if i==bg.index[k]:
            print(k)
            cells_norm2.at[i,'EGFP']=(cells_all.at[i,'EGFP']-bg.at[bg.index[k],'EGFP'])
            cells_norm2.at[i,'DsRed']=(cells_all.at[i,'DsRed']-bg.at[bg.index[k],'DsRed'])
            cells_norm2.at[i,'ratio G/R']=cells_norm2.at[i,'EGFP']/cells_norm2.at[i,'DsRed']
cells_norm2.loc[cells_norm2['ratio G/R'] > 3, ['Color']] = 'Green'
cells_norm2.loc[cells_norm2['ratio G/R'] < 0.3, ['Color']] = 'Red'
cells_norm2.loc[(cells_norm2['ratio G/R'] > 0.3) & (cells_norm2['ratio G/R']<3), ['Color']] = 'Yellow' 

#%%
intens=cells_norm2.T
intens_max=pd.DataFrame()
inds=intens.columns
k=0
while k <len(inds):
    ind=inds[k]
    while sum(inds.str.count(ind[:-2]))!=1:
        #print(sum(inds.str.count(ind[:-2])))
        if inds[k][:-2]==intens.columns[k+1][:-2]:
            print(inds[k][:-2])
            if intens.iat[0,k]<intens.iat[0,k+1]:
                print('change')
                intens_max[inds[k][:-2]]=intens[inds[k+1]]
                intens_max.at['name',inds[k][:-2]]=inds[k]
                k=k+1
            else :
                print('change 2')
                intens_max[inds[k][:-2]]=intens[inds[k]]
                intens_max.at['name',inds[k][:-2],]=inds[k]
                k=k+1
        else :
            print('loop')
            intens_max[inds[k][:-2]]=intens[inds[k]]
            intens_max.at['name',inds[k][:-2],]=inds[k]
            k=k+1
    else :
        print(k)
        intens_max[inds[k][:-2]]=intens[inds[k]]
        intens_max.at['name',inds[k][:-2]]=inds[k]
        k=k+1
#%%
cells_maxN=intens_max.T
#%%
p6=sns.scatterplot(cells_maxN, x='EGFP', y='DsRed',hue='Color', palette=['gold','g','r'])
plt.xscale('log')
plt.yscale('log')
plt.title('Intensity of each fluo without background')
plt.ylim(1, 10**5)
plt.xlim(1, 10**5)
#%%
cells_maxN.to_csv(path + '/data.txt', sep='\t')

#%%


p5=sns.scatterplot(cells_maxN, x='EGFP', y='DsRed', hue='Color', palette=['g','r','gold'])
p6=sns.scatterplot(bg, x='EGFP', y='DsRed', hue='Color', palette=['k'])
plt.xscale('log')
plt.yscale('log')
# plt.ylim(10, 10**5)
# plt.xlim(10, 10**5)
#%%
exp_cond=giczi.metadatafromtxt(fileInfotxt,path)

#%%
cellID=ufun.findInfosInFileName(fileInfotxt[:-9], 'cellID')

#%%
def intensityCellShrimp(images,dir_crop,dir_file):
    intens=pd.DataFrame()
    #for j in range(len(images)):
    images_folder=os.path.join(dir_crop,images)
    images_crop=os.listdir(images_folder)
    file=images[:-5]
    print(file)
    channels=len(images_crop)
    exp_cond=pd.DataFrame()
    exp_cond.index=['EGFP','DsRed']
    histos=pd.DataFrame()
    images_rg=[]
    cellID=ufun.findInfosInFileName(file, 'cellID')
    name_cell=cellID[12:]
    for i in range(channels):
        images_rg.append(plt.imread(os.path.join(images_folder,images_crop[i])))
        histo,bins=np.histogram(images_rg[i],bins=65536,range=(images_rg[i][images_rg[i] != 0].min(),images_rg[i].max()))
        histos[i]=[histo,bins]
    # fig,axs=plt.subplots(2)
    # axs[0].plot(histos[0][1][0:-1],histos[0][0],color='g')
    # axs[0].set(ylabel=exp_cond.columns[0])


    # axs[1].plot(histos[1][1][0:-1],histos[1][0],color='r')
    # axs[1].set(ylabel=exp_cond.columns[1])
    # fig.suptitle('histograms of fluo images '+name_cell)
    
    # mean_g=images_rg[0][images_rg[0] != 0].mean()
    # mean_r=images_rg[1][images_rg[1] != 0].mean()
    # intens[name_cell]=[mean_g, mean_r]
    # intens.index=[exp_cond.index[i] for i in range(len(exp_cond.index))]

    mean_g=images_rg[0][images_rg[0] != 0].mean()
    mean_r=images_rg[1][images_rg[1] != 0].mean()
    intens[name_cell]=[mean_g, mean_r]
    intens.index=[exp_cond.index[i] for i in range(len(exp_cond.index))]
    intens_data=intens.T
    #intens_data['name_cell']=intens_data.loc['EGFP']
    # for name in intens_data.index :
    #     print(name)
    #     intens_data.at[name,'name_cell']=name[:-3]
    intens_data['ratio G/R']=intens_data['EGFP']/intens_data['DsRed']
   #intens_data['ratio R/G']=intens_data['DsRed']/intens_data['EGFP']
    intens_data['Color']=intens_data['ratio G/R']
    intens_data.loc[intens_data['ratio G/R'] > 3, ['Color']] = 'Green'
    intens_data.loc[intens_data['ratio G/R'] < 0.3, ['Color']] = 'Red'
    intens_data.loc[(intens_data['ratio G/R'] > 0.3) & (intens_data['ratio G/R']<3), ['Color']] = 'Yellow'
    intens_data.loc[intens_data['Color']=='Green', ['phase']] = 'S/G2'
    intens_data.loc[intens_data['Color']=='Red', ['phase']] = 'G1'
    intens_data.loc[intens_data['Color']=='Yellow', ['phase']] = 'G1/S'
    return intens_data
    
#%%

path_crop="D:/Eloise/MagneticPincherData/Raw/24.05.23_fluo/crop"
path="D:/Eloise/MagneticPincherData/Raw/24.05.23_fluo"
#cell='24-05-16_M1_P2_C1'
allcrops=os.listdir(path_crop)
fileInfotxt=[i[:-5]+'_info.txt' for i in allcrops]

cells=pd.DataFrame()

for j in range(len(allcrops)):
    name=allcrops[j][12:-17]
    test= intensityCellShrimp(allcrops[j],path_crop,path)
    cells=pd.concat([cells,test])

#%% background for shrimp
path_crop_bg="D:/Eloise/MagneticPincherData/Raw/24.05.23_fluo/background"
path="D:/Eloise/MagneticPincherData/Raw/24.05.23_fluo"
#cell='24-05-16_M1_P2_C1'
allcrops_bg=os.listdir(path_crop_bg)
fileInfotxt_bg=[i[:-5]+'_info.txt' for i in allcrops_bg]

bgS=pd.DataFrame()

for j in range(len(allcrops_bg)):
    name=allcrops_bg[j][12:-17]
    test= intensityCellShrimp(allcrops_bg[j],path_crop_bg,path)
    bgS=pd.concat([bgS,test])
bgS=bgS.replace('Yellow','bg')

#%%
def normalize_crop_bg(cells,bg):
    cells_all=cells.copy()
    cells_normS=cells_all.copy()
    for i in cells_all.index :
        for k in range(len(bgS.index)):
            if i==bgS.index[k]:
                print(k)
                cells_normS.at[i,'EGFP']=(cells_all.at[i,'EGFP']-bgS.at[bgS.index[k],'EGFP'])
                cells_normS.at[i,'DsRed']=(cells_all.at[i,'DsRed']-bgS.at[bgS.index[k],'DsRed'])
                cells_normS.at[i,'ratio G/R']=cells_normS.at[i,'EGFP']/cells_normS.at[i,'DsRed']
    cells_normS.loc[cells_normS['ratio G/R'] > 3, ['Color']] = 'Green'
    cells_normS.loc[cells_normS['ratio G/R'] < 0.3, ['Color']] = 'Red'
    cells_normS.loc[(cells_normS['ratio G/R'] > 0.3) & (cells_normS['ratio G/R']<3), ['Color']] = 'Yellow'
    cells_normS.loc[cells_normS['Color'] =='Green', ['phase']] = 'S/G2'
    cells_normS.loc[cells_normS['Color'] =='Red', ['phase']] = 'G1'
    cells_normS.loc[cells_normS['Color'] =='Yellow', ['phase']] = 'G1/S'
    return cells_normS

#%%
cells_norm=normalize_crop_bg(cells, bgS)
cells_norm.to_csv(path + '/data_fluo_24.05.23.txt', sep='\t')
#%%
# plt.figure(figsize=(8, 7))
# p5=sns.scatterplot(cells_normS, x='EGFP', y='DsRed', hue='Color', palette=['r','g','gold'])
# #p6=sns.scatterplot(bg, x='EGFP', y='DsRed', hue='Color', palette=['k'])
# plt.title('Intensity of each color wo background - cells from Shrimp 23-05')
# plt.xscale('log')
# plt.yscale('log')
# plt.ylim(10, 10**4)
# plt.xlim(10, 10**4)

# #%%

# cells_normS.to_csv(path + '/data.txt', sep='\t')


#%%
path_crop="D:/Eloise/MagneticPincherData/Raw/24.06.04_fluo/crop_wobg"
path="D:/Eloise/MagneticPincherData/Raw/24.06.04_fluo"
all_dircrops=os.listdir(path_crop)
allcrops=os.listdir(path_crop)
fileInfotxt=[i[:-5]+'_info.txt' for i in allcrops]

cells=pd.DataFrame()

for j in range(len(allcrops)):
    name=allcrops[j][12:-19]
    test= intensityCellShrimp(allcrops[j],path_crop,path)
    cells=pd.concat([cells,test])



#%% background for shrimp
path_crop_bg="D:/Eloise/MagneticPincherData/Raw/24.06.04_fluo/background"
path="D:/Eloise/MagneticPincherData/Raw/24.06.04_fluo"
#cell='24-05-16_M1_P2_C1'
allcrops_bg=os.listdir(path_crop_bg)
fileInfotxt_bg=[i[:-5]+'_info.txt' for i in allcrops_bg]

bgS=pd.DataFrame()

for j in range(len(allcrops_bg)):
    name=allcrops_bg[j][12:-17]
    test= intensityCellShrimp(allcrops_bg[j],path_crop_bg,path)
    bgS=pd.concat([bgS,test])
bgS=bgS.replace('Yellow','bg')

#%%
cells_norm=normalize_crop_bg(cells, bgS)
cells_norm.loc[cells_norm.index=='P2_C12', ['phase']] = 'M'
cells_norm.loc[cells_norm.index=='P2_C13', ['phase']] = 'M'
cells_norm.loc[cells_norm.index=='P2_C5', ['phase']] = 'M'

#%%
cells.to_csv(path + '/data_fluo_24.06.04_wobg.txt', sep='\t')

#%%
mitotic=['P2_C12','P2_C13','P2_C5']
for i in mitotic:
    cells.at[i,'phase']='M'
#%%
p2=sns.scatterplot(data=cells,x='EGFP',y='DsRed',hue='Color',palette=['gold','g','r'])
plt.xscale('log')
plt.yscale('log')
#%%
p1=sns.scatterplot(data=cells_norm,x='EGFP',y='DsRed',hue='Color',palette=['g','gold','r'])
p1.axhline(np.min(bgS['DsRed']))
p1.axvline(np.min(bgS['EGFP']))
plt.xscale('log')
plt.xlim(10,10**4)
plt.yscale('log')
plt.ylim(10,10**4)


#%%
path_crop="D:/Eloise/MagneticPincherData/Raw/24.06.12_fluo/crop"
path="D:/Eloise/MagneticPincherData/Raw/24.06.12_fluo"
all_dircrops=os.listdir(path_crop)
allcrops=os.listdir(path_crop)
fileInfotxt=[i[:-5]+'_info.txt' for i in allcrops]

cells=pd.DataFrame()

for j in range(len(allcrops)):
    name=allcrops[j][12:-19]
    test= intensityCellShrimp(allcrops[j],path_crop,path)
    cells=pd.concat([cells,test])
    

#%% background for shrimp
path_crop_bg="D:/Eloise/MagneticPincherData/Raw/24.06.12_fluo/background"
path="D:/Eloise/MagneticPincherData/Raw/24.06.12_fluo"
#cell='24-05-16_M1_P2_C1'
allcrops_bg=os.listdir(path_crop_bg)
fileInfotxt_bg=[i[:-5]+'_info.txt' for i in allcrops_bg]

bgS=pd.DataFrame()

for j in range(len(allcrops_bg)):
    name=allcrops_bg[j][12:-17]
    test= intensityCellShrimp(allcrops_bg[j],path_crop_bg,path)
    bgS=pd.concat([bgS,test])
bgS=bgS.replace('Yellow','bg')

#%%
cells_norm=normalize_crop_bg(cells, bgS)
#%%
mitotic=['P2_C8','P2_C9','P2_C4','P2_C3','P2_C15','P1_C7','P1_C5']
for i in mitotic:
    cells_norm.at[i,'phase']='M'

#%%
cells_norm.to_csv(path + '/data_fluo_24.06.12.txt', sep='\t')

#%%
p1=sns.scatterplot(data=cells_norm,x='EGFP',y='DsRed',hue='phase',palette=['gold','r','g','k'])
p1.axhline(np.min(bgS['DsRed']))
p1.axvline(np.min(bgS['EGFP']))
plt.xscale('log')
plt.xlim(10**0,10**4)
plt.yscale('log')
plt.ylim(10**0,10**4)



#%%
path_crop="D:/Eloise/MagneticPincherData/Raw/24.07.03_fluo/crop"
path="D:/Eloise/MagneticPincherData/Raw/24.07.03_fluo"
all_dircrops=os.listdir(path_crop)
allcrops=os.listdir(path_crop)
fileInfotxt=[i[:-5]+'_info.txt' for i in allcrops]

cells=pd.DataFrame()

for j in range(len(allcrops)):
    name=allcrops[j][12:-19]
    test= intensityCellShrimp(allcrops[j],path_crop,path)
    cells=pd.concat([cells,test])
    
#%%
path_crop_bg="D:/Eloise/MagneticPincherData/Raw/24.07.03_fluo/background"
path="D:/Eloise/MagneticPincherData/Raw/24.07.03_fluo"
#cell='24-05-16_M1_P2_C1'
allcrops_bg=os.listdir(path_crop_bg)
fileInfotxt_bg=[i[:-5]+'_info.txt' for i in allcrops_bg]

bgS=pd.DataFrame()

for j in range(len(allcrops_bg)):
    name=allcrops_bg[j][12:-17]
    test= intensityCellShrimp(allcrops_bg[j],path_crop_bg,path)
    bgS=pd.concat([bgS,test])
bgS=bgS.replace('Yellow','bg')

#%%
cells_norm=normalize_crop_bg(cells, bgS)
#%%
mitotic=['P2_C5','P2_C6','P2_C9','P3_C6','P3_C7']
for i in mitotic:
    cells_norm.at[i,'phase']='M'
    
#%%
p1=sns.scatterplot(data=cells,x='EGFP',y='DsRed',hue='phase',palette=['gold','r','g','k'])
#p1.axhline(np.min(bgS['DsRed']))
#p1.axvline(np.min(bgS['EGFP']))
plt.xscale('log')
#plt.xlim(10**0,10**4)
plt.yscale('log')
#plt.ylim(10**0,10**4)
plt.show()
#%%
cells.to_csv(path + '/data_fluo_24.07.11.txt', sep='\t')

#%%
path_crop="D:/Eloise/MagneticPincherData/Raw/24.07.11_fluo/crop_wobg"
path="D:/Eloise/MagneticPincherData/Raw/24.07.11_fluo"
all_dircrops=os.listdir(path_crop)
allcrops=os.listdir(path_crop)
fileInfotxt=[i[:-5]+'_info.txt' for i in allcrops]
#%%
cells=pd.DataFrame()

for j in range(len(allcrops)):
    name=allcrops[j][12:-19]
    test= intensityCellShrimp(allcrops[j],path_crop,path)
    cells=pd.concat([cells,test])


#%%
path_crop_bg="D:/Eloise/MagneticPincherData/Raw/24.07.11_fluo/bg"
path="D:/Eloise/MagneticPincherData/Raw/24.07.11_fluo"
#cell='24-05-16_M1_P2_C1'
allcrops_bg=os.listdir(path_crop_bg)
fileInfotxt_bg=[i[:-5]+'_info.txt' for i in allcrops_bg]

bgS=pd.DataFrame()

for j in range(len(allcrops_bg)):
    name=allcrops_bg[j][12:-17]
    test= intensityCellShrimp(allcrops_bg[j],path_crop_bg,path)
    bgS=pd.concat([bgS,test])
bgS=bgS.replace('Yellow','bg')

#%%
cells_norm=normalize_crop_bg(cells, bgS)
#%%
mitotic=['P2_C4','P2_C6','P2_C8','P2_C2','P1_C4']
for i in mitotic:
    cells.at[i,'phase']='M'

#%%
cells_norm.to_csv(path + '/data_fluo_24.07.11bg.txt', sep='\t')


#%%
path_crop="D:/Eloise/MagneticPincherData/Raw/24.07.18_fluo/crop_wobg"
path="D:/Eloise/MagneticPincherData/Raw/24.07.18_fluo"
all_dircrops=os.listdir(path_crop)
allcrops=os.listdir(path_crop)
fileInfotxt=[i[:-5]+'_info.txt' for i in allcrops]
#%%
cells=pd.DataFrame()

for j in range(len(allcrops)):
    name=allcrops[j][12:-19]
    test= intensityCellShrimp(allcrops[j],path_crop,path)
    cells=pd.concat([cells,test])
    
#%%
#%%
mitotic=['P1_C3','P1_C4','P2_C4']
for i in mitotic:
    cells.at[i,'phase']='M'
    
#%%
cells.to_csv(path + '/data_fluo_24.07.18.txt', sep='\t')
#%%
p1=sns.scatterplot(data=cells,x='EGFP',y='DsRed',hue='phase',hue_order=['G1/S','G1','S/G2','M'],palette=['gold','r','g','k'])
#p1.axhline(np.min(bgS['DsRed']))
#p1.axvline(np.min(bgS['EGFP']))
plt.xscale('log')
plt.xlim(10**1,10**4)
plt.yscale('log')
plt.ylim(10**1,10**4)
plt.show()