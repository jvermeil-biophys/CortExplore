# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:31:53 2022

@author: anumi
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
plt.close('all')
mainDir = 'D:/Anumita/MagneticPincherData/Fluorescence_Analysis'
scale = 7.58

dirActin = mainDir+'/LifeAct'
dirGef = mainDir+'/RhoGEF'

filesActin = os.listdir(dirActin)
filesGef = os.listdir(dirGef)

bg = pd.read_csv(mainDir+'/backgroundValues.csv')

#%%


for i,j in zip(filesActin, filesGef):
    fig1, axes = plt.subplots(2,1, figsize=(15,10))
    print(i)
    title = i.split('_disc20um')[0]
    
    bgCell = bg[bg['cell'] == title]
    bgAct = bgCell['LifeAct']
    bgGef = bgCell['RhoGef']
    
    dataActin = pd.read_csv(dirActin+'/'+i)
    dataGef = pd.read_csv(dirGef+'/'+j)
    
    dataGef['Intensity_bg'] = dataGef['Intensity'] - bgGef.values
    dataActin['Intensity_bg'] = dataActin['Intensity'] - bgAct.values
    
    
    slices = np.unique(dataActin['Slice'])
    
    dataActin['fluoNorm'] = [np.nan]*len(dataActin)
    dataGef['fluoNorm'] = [np.nan]*len(dataGef)
    
    for z in slices:
        ratio = (dataActin['Intensity_bg'][dataActin['Slice'] == z]).values/(dataActin['Intensity_bg'][dataActin['Slice'] == 1]).values
        dataActin['fluoNorm'][dataActin['Slice'] == z] = ratio
        
        ratio = (dataGef['Intensity_bg'][dataGef['Slice'] == z]).values/(dataGef['Intensity_bg'][dataGef['Slice'] == 1]).values
        dataGef['fluoNorm'][dataActin['Slice'] == z] = ratio
    
    yGef = dataGef['Intensity_bg']
    yAct = dataActin['Intensity_bg']
    
    xGef = dataGef['Distance (microns)']/scale
    xAct= dataActin['Distance']/scale
    
    axes[0] = sns.lineplot(data = dataActin, x = xAct, y = yAct, hue = 'Slice', ax = axes[0])
    axes[1] = sns.lineplot(data = dataGef, x = xGef, y = yGef, hue = 'Slice', ax = axes[1])
    
    fig1.suptitle(title+ " | All Curves Fluo")
    axes[0].set_xlabel('')
    axes[1].set_xlabel('Distance (microns)')
    
    axes[0].set_ylabel('LifeAct-iRFP Fluo Intensity')
    axes[1].set_ylabel('ArhGEF11-mCherry Fluo Intensity')
    
    plt.savefig(mainDir+'/Plots/AllCurves/'+title+'_Fluo')
    plt.show()
    
    fig2, axes = plt.subplots(2,1, figsize=(15,10))
    
    
    mask =  dataActin['Slice']
    mask = mask.where(mask%3 == 0)
    mask = np.isnan(mask)
    
    dfGef = dataGef[mask == False]
    dfActin = dataActin[mask == False]
    
    yGef = dfGef['Intensity_bg']
    yAct = dfActin['Intensity_bg']
    
    xGef = (dfGef['Distance (microns)']/scale)
    xAct= (dfActin['Distance']/scale)
    
    axes[0] = sns.lineplot(data = dataActin, x = xAct, y = yAct, hue = 'Slice', ax = axes[0])
    axes[1] = sns.lineplot(data = dataGef, x = xGef, y = yGef, hue = 'Slice', ax = axes[1])
    
    fig2.suptitle(title + " | Selected Curves Fluo")
    axes[0].set_xlabel('')
    axes[1].set_xlabel('Distance (microns)')
    
    axes[0].set_ylabel('LifeAct-iRFP Fluo Intensity')
    axes[1].set_ylabel('ArhGEF11-mCherry Fluo Intensity')
    
    plt.savefig(mainDir+'/Plots/SelectedCurves/'+title+'_Fluo')
    plt.show()
    
    fig3, axes = plt.subplots(2,1, figsize=(15,10))
    
    yGef = dataGef['fluoNorm']
    yAct = dataActin['fluoNorm']
    
    xGef = dataGef['Distance (microns)']/scale
    xAct= dataActin['Distance']/scale
    
    axes[0] = sns.lineplot(data = dataActin, x = xAct, y = yAct, hue = 'Slice', ax = axes[0])
    axes[1] = sns.lineplot(data = dataGef, x = xGef, y = yGef, hue = 'Slice', ax = axes[1])
    
    fig3.suptitle(title + " | Normalised Fluorescence (with t = 0)")
    axes[0].set_xlabel('')
    axes[1].set_xlabel('Distance (microns)')
    
    axes[0].set_ylabel('LifeAct-iRFP Fluo Intensity (A.U.)')
    axes[1].set_ylabel('ArhGEF11-mCherry Fluo Intensity (A.U.)')
    
    plt.savefig(mainDir+'/Plots/NormalisedCurves/'+title+'_Fluo')
    plt.show()