# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 21:27:55 2022

@author: JosephVermeil & AnumitaJawahar
"""

#!/usr/bin/env python
# coding: utf-8


# %% > Imports and constants

#### Main imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt


import os
import sys
import time
import random
import warnings
import itertools
import matplotlib

from copy import copy
from cycler import cycler
from datetime import date
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)

import GraphicStyles as gs
import UtilityFunctions as ufun
import TrackAnalyser as taka
import TrackAnalyser_dev_AJ as tka

#### Potentially useful lines of code
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# cp.DirDataFigToday

#### Pandas
pd.set_option('display.max_columns', None)
# pd.reset_option('display.max_columns')
pd.set_option('display.max_rows', None)
pd.reset_option('display.max_rows')


####  Matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})

#### Graphic options
gs.set_default_options_jv()

#### Bokeh
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Range1d
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10
from bokeh.layouts import gridplot
output_notebook()

#### Markers
my_default_marker_list = ['o', 's', 'D', '>', '^', 'P', 'X', '<', 'v', 'p']
markerList10 = ['o', 's', 'D', '>', '^', 'P', 'X', '<', 'v', 'p']

todayFigDir = cp.DirDataFigToday
experimentalDataDir = cp.DirRepoExp

# %% Reminders

# %%%  Stats
# 
# (A) How to compute confidence intervals of fitted parameters with (1-alpha) confidence:
# 
#     0) from scipy import stats
#     1) df = nb_pts - nb_parms ; se = diag(cov)**0.5
#     2) Student t coefficient : q = stat.t.ppf(1 - alpha / 2, df)
#     3) ConfInt = [params - q*se, params + q*se]

# ## 

# %% TimeSeries functions


# %%% List files
# allTimeSeriesDataFiles = [f for f in os.listdir(timeSeriesDataDir) \
#                           if (os.path.isfile(os.path.join(timeSeriesDataDir, f)) and f.endswith(".csv"))]
# print(allTimeSeriesDataFiles)


# %%% Get a time series

# df = aja.getCellTimeSeriesData('22-03-31_M9_P2_C2')


# # %%% Plot a time series

# aja.plotCellTimeSeriesData('22-03-31_M9_P2_C2')


# %%% Plot multiple time series

# allTimeSeriesDataFiles = [f for f in os.listdir(timeSeriesDataDir) if (os.path.isfile(os.path.join(timeSeriesDataDir, f)) and f.endswith(".csv"))]
# for f in allTimeSeriesDataFiles:
#     if '22-04-12_M2' in f:
#         aja.plotCellTimeSeriesData(f[:-4])


# # %%% Close all

# plt.close('all')


# %%% Experimental conditions

expDf = ufun.getExperimentalConditions(experimentalDataDir, save=True)

# =============================================================================
# %%% Constant Field

# %%%% Update the table

# aja.computeGlobalTable_ctField(task='updateExisting', fileName = '', save=False, source = 'Python')



# %%%% Refresh the whole table

# aja.computeGlobalTable_ctField(task = 'fromScratch', fileName = '', save = True, source = 'Python')


# # %%%% Display

# df = aja.getGlobalTable_ctField().head()



# =============================================================================
# %%% Mechanics

# %%%% Update the table

tka.computeGlobalTable_meca(task = 'updateExisting', fileName = 'Global_MecaData_AJ', 
                            save = True, PLOT = True, source = 'Python')


# %%%% Refresh the whole table

# aja.computeGlobalTable_meca(task = 'fromScratch', fileName = 'Global_MecaData_AJ', 
#                             save = True, PLOT = True, source = 'Python')

# %%%% Specific experiments

Task = '22-08-26_M7 & 22-08-26_M5 & 22-08-26_M10 & 22-08-26_M1 & 22-08-26_M3'
#'22-08-26_M7 & 22-08-26_M5 & 22-08-26_M10 & 22-08-26_M1 & 22-08-26_M3' # For instance '22-03-30 & '22-03-31'
tka.computeGlobalTable_meca(task = Task, fileName = 'Global_MecaData_AJ2', 
                            save = True, PLOT = True, source = 'Python') # task = 'updateExisting'


 # %%%% Precise dates (to plot)

date = '22-05-31 & 22-06-21' # For instance '22-03-30 & '22-03-31'
tka.computeGlobalTable_meca(task = date, fileName = 'Global_MecaData_AJ1', 
                            save = True, PLOT = True, source = 'Python') # task = 'updateExisting'

 # %%%% Precise dates (to plot)

date = '22-08-26' # For instance '22-03-30 & '22-03-31'
taka.computeGlobalTable_meca(task = date, fileName = 'Global_MecaData_AJ_22-08-26', 
                            save = True, PLOT = True, source = 'Python') # task = 'updateExisting'

# %%%% Display

# df = aja.getGlobalTable_meca('Global_MecaData_Py2').tail()


# =============================================================================
# %%% Fluorescence

# %%%% Display

df = tka.getFluoData().head()
# 



# #############################################################################
# %% > Data import & export

#### Data import

#### Display

# df1 = ufun.getExperimentalConditions().head()
# df2 = aja.getGlobalTable_ctField().head()
# df3 = aja.getGlobalTable_meca().head()
# df4 = aja.getFluoData().head()


# #### GlobalTable_ctField

# GlobalTable_ctField = aja.getGlobalTable(kind = 'ctField')
# GlobalTable_ctField.head()


# #### GlobalTable_ctField_Py

# GlobalTable_ctField_Py = aja.getGlobalTable(kind = 'ctField_py')
# GlobalTable_ctField_Py.head()


# #### GlobalTable_meca

# GlobalTable_meca = aja.getGlobalTable(kind = 'meca_matlab')
# GlobalTable_meca.tail()


#### GlobalTable_meca_Py

GlobalTable_meca = tka.getGlobalTable(kind = 'Global_MecaData_AJ_22-12-07')
GlobalTable_meca.head()


# #### GlobalTable_meca_Py2

# GlobalTable_meca_Py2 = aja.getGlobalTable(kind = 'meca_py2')
# GlobalTable_meca_Py2.head()


# #### Global_MecaData_NonLin_Py

# # GlobalTable_meca_nonLin = aja.getGlobalTable(kind = 'meca_nonLin')
# GlobalTable_meca_nonLin = aja.getGlobalTable(kind = 'Global_MecaData_NonLin2_Py')
# GlobalTable_meca_nonLin.head()


# %%% Custom data export

# %%%% 21-06-28 Export of E - h data for Julien

# GlobalTable_meca_Py_expJ = GlobalTable_meca_Py.loc[GlobalTable_meca_Py['validatedFit'] & GlobalTable_meca_Py['validatedThickness']]\
#                                                [['cell type', 'cell subtype', 'bead type', 'drug', 'substrate', 'compNum', \
#                                                  'EChadwick', 'H0Chadwick', 'surroundingThickness', 'ctFieldThickness']]
# GlobalTable_meca_Py_expJ = GlobalTable_meca_Py_expJ.reset_index()
# GlobalTable_meca_Py_expJ = GlobalTable_meca_Py_expJ.drop('index', axis=1)
# savePath = os.path.join(dataDir, 'mecanicsData_3T3.csv')
# GlobalTable_meca_Py_expJ.to_csv(savePath, sep=';')


# %% > Plotting Functions

def TmodulusVsCompression_V1(GlobalTable_meca, dates, selectedStressRange, activationType = 'all'):
    sns.set_style('darkgrid')
    
    fitC =  np.array([S for S in range(100, 1150, 50)])
    fitW = [100, 150, 200, 250, 300]

    fitCenters = np.array([[int(S) for S in fitC] for w in fitW]).flatten()
    fitWidth = np.array([[int(w) for S in fitC] for w in fitW]).flatten()

    fitMin = np.array([[int(S-(w/2)) for S in fitC] for w in fitW]).flatten()
    fitMax = np.array([[int(S+(w/2)) for S in fitC] for w in fitW]).flatten()

    fitCenters, fitWidth = fitCenters[fitMin>0], fitWidth[fitMin>0]
    fitMin, fitMax = fitMin[fitMin>0], fitMax[fitMin>0]

    stressRanges = ['S='  + str(fitCenters[ii]) + '+/-' + str(int(fitWidth[ii]//2)) for ii in range(len(fitCenters))]
    
    # stressRanges = ['100+/-100', '150+/-100', '200+/-100', '250+/-100', \
    #                '300+/-100', '350+/-100', '400+/-100', '450+/-100', \
    #                '500+/-100', '550+/-100', '600+/-100', '650+/-100', \
    #                '700+/-100', '750+/-100', '800+/-100', '850+/-100', \
    #                '950+/-100', '1000+/-100', '1050+/-100', '1100+/-100']
    
    if not dates == 'all':
        GlobalTable_meca = GlobalTable_meca[GlobalTable_meca['manipID'].str.contains(dates)]
        
    if not activationType == 'all':
        GlobalTable_meca = GlobalTable_meca[GlobalTable_meca['activation type'] == activationType]
    
    
    allFiles = np.unique(GlobalTable_meca['cellID'])
    # print(allFiles)
    lenSubplots = len(allFiles)
    rows= int(np.floor(np.sqrt(lenSubplots)))
    cols= int(np.ceil(lenSubplots/rows))
    fontsize = 15
    
    if selectedStressRange == 'all':
       for selectedStressRange in stressRanges:
           print(selectedStressRange)
           fig, axes = plt.subplots(nrows = rows, ncols = cols, figsize = (15,15))
           _axes = []
           
           for ax_array in axes:
               for ax in ax_array:
                   _axes.append(ax)

           for cellID, ax in zip(allFiles, _axes):
               GlobalTable_meca_spec = GlobalTable_meca[(GlobalTable_meca['cellID'] == cellID)]
               KChadwick = GlobalTable_meca_spec['KChadwick_'+selectedStressRange].values
               compNum = GlobalTable_meca_spec['compNum'].values
               firstActivation = GlobalTable_meca['first activation'].values[0] + 1
               categories = (GlobalTable_meca_spec['validatedFit_'+selectedStressRange] == True).values
               activationType = GlobalTable_meca['activation type'][GlobalTable_meca['cellID'] == cellID].values[0]
               print(activationType)
               if activationType == 'global':
                   markerColor = 'orange'
               elif activationType == 'at beads':
                   markerColor = 'blue'
               elif activationType == 'away from beads':   
                   markerColor = 'green'
               categories = np.asarray(categories*1)
               colormap = np.asarray(['r', markerColor])
               labels = np.asarray(['Not valid', activationType])
               ax.scatter(compNum, KChadwick, c = colormap[categories], label = labels[categories])
               ax.set_title(cellID+'-'+activationType, fontsize = 15)
               
               
               ax.set_ylim(0, 40000)
               ax.set_xlim(0, len(compNum)+1)
               ax.axvline(x = firstActivation, color = 'r')
               
               plt.setp(ax.get_xticklabels(), fontsize=fontsize)
               plt.setp(ax.get_yticklabels(), fontsize=fontsize)
               fig.suptitle('KChadwick_'+selectedStressRange+' vs. CompNum_'+activationType)
               try:
                   os.mkdir(todayFigDir+'/TModulusVsComp Plots')
               except:
                   pass
               
               plt.savefig(todayFigDir+'/'+dates+'_'+cellID+'_TModulusVsComp_'+(selectedStressRange[:-6])+'_'+activationType+'.png')
               plt.show()
               plt.close()
        
    else:
        fig, axes = plt.subplots(nrows = rows, ncols = cols, figsize = (15,15))
        _axes = []
        for ax_array in axes:
            for ax in ax_array:
                _axes.append(ax)
        for cellID, ax in zip(allFiles, _axes):
            print(cellID)
            GlobalTable_meca_spec = GlobalTable_meca[(GlobalTable_meca['cellID'] == cellID)]
            KChadwick = GlobalTable_meca_spec['KChadwick_'+selectedStressRange].values
            compNum = GlobalTable_meca_spec['compNum'].values
            firstActivation = GlobalTable_meca['first activation'].values[0] + 1
            categories = (GlobalTable_meca_spec['validatedFit_'+selectedStressRange] == True).values
            activationType = GlobalTable_meca['activation type'][GlobalTable_meca['cellID'] == cellID].values[0]
            if activationType == 'global':
                markerColor = 'orange'
            elif activationType == 'at beads':
                markerColor = 'blue'
            elif activationType == 'away from beads':   
                markerColor = 'green'
            categories = np.asarray(categories*1)
            colormap = np.asarray(['r', markerColor])
            labels = np.asarray(['Not valid', activationType])
            ax.scatter(compNum, KChadwick, c = colormap[categories], label = labels[categories])
            ax.set_title(cellID+'-'+activationType, fontsize = 15)
            
            
            ax.set_ylim(0,40000)
            ax.set_xlim(0, len(compNum)+1)
            ax.axvline(x = firstActivation, color = 'r')
            
            plt.setp(ax.get_xticklabels(), fontsize=fontsize)
            plt.setp(ax.get_yticklabels(), fontsize=fontsize)
            fig.suptitle('KChadwick_'+selectedStressRange+' vs. CompNum_'+activationType)
            try:
                os.mkdir(todayFigDir)
            except:
                pass
            
            plt.savefig(todayFigDir+'/'+dates+'_'+cellID+'_TModulusVsComp_'+(selectedStressRange[:-6])+'_'+activationType+'.png')
        plt.show()
        plt.close()
    
def bestH0vsCompression(GlobalTable_meca, dates, activationType = 'all'):
    
    allFiles = np.unique(GlobalTable_meca['cellID'])
    lenSubplots = len(allFiles)
    rows= int(np.floor(np.sqrt(lenSubplots)))
    cols= int(np.ceil(lenSubplots/rows))
    fontsize = 15
    
    if not dates == 'all':
        GlobalTable_meca = GlobalTable_meca[GlobalTable_meca['manipID'].str.contains(dates)]
        
    if not activationType == 'all':
        GlobalTable_meca = GlobalTable_meca[GlobalTable_meca['activation type'] == activationType]
    
    fig, axes = plt.subplots(nrows = rows, ncols = cols, figsize = (15,15))
    _axes = []
    for ax_array in axes:
        for ax in ax_array:
            _axes.append(ax)
            
    for cellID, ax in zip(allFiles, _axes):
        print(cellID)
        GlobalTable_meca_spec = GlobalTable_meca[(GlobalTable_meca['cellID'] == cellID)]
        firstActivation = GlobalTable_meca['first activation'].values[0] + 1
        c = GlobalTable_meca['activation type'][GlobalTable_meca['cellID'] == cellID].values[0]
        activationTag = GlobalTable_meca['activation type'][GlobalTable_meca['cellID'] == cellID].values[0]
        if activationTag == 'global':
            markerColor = 'orange'
        elif activationTag == 'at beads':
            markerColor = 'blue'
        elif activationTag == 'away from beads':   
            markerColor = 'green'
        
        bestH0 = GlobalTable_meca_spec['bestH0'].values
        compNum = GlobalTable_meca_spec['compNum'].values
        
        ax.scatter(compNum, bestH0, color = markerColor)
        ax.set_title(cellID+'-'+activationTag, fontsize = 15)
        
        
        ax.set_ylim(0,2000)
        ax.set_xlim(0, len(compNum)+1)
        ax.axvline(x = firstActivation, color = 'r')
        
        plt.setp(ax.get_xticklabels(), fontsize=fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize)
        fig.suptitle('bestH0 vs. CompNum_'+activationType)
        try:
            os.mkdir(todayFigDir)
        except:
            pass
        
        plt.savefig(todayFigDir+'/'+dates+'H0VsComp_'+activationType+'.png')
    plt.show()


def TmodulusVsCompression(figDir, GlobalTable_meca, dates, selectedStressRange, activationType = 'all'):
    sns.set_style('darkgrid')
    
    fitC =  np.array([S for S in range(100, 1150, 50)])
    fitW = [100, 150, 200, 250, 300]

    fitCenters = np.array([[int(S) for S in fitC] for w in fitW]).flatten()
    fitWidth = np.array([[int(w) for S in fitC] for w in fitW]).flatten()

    fitMin = np.array([[int(S-(w/2)) for S in fitC] for w in fitW]).flatten()
    fitMax = np.array([[int(S+(w/2)) for S in fitC] for w in fitW]).flatten()

    fitCenters, fitWidth = fitCenters[fitMin>0], fitWidth[fitMin>0]
    fitMin, fitMax = fitMin[fitMin>0], fitMax[fitMin>0]

    stressRanges = ['S='  + str(fitCenters[ii]) + '+/-' + str(int(fitWidth[ii]//2)) for ii in range(len(fitCenters))]

    if not dates == 'all':
        GlobalTable_meca = GlobalTable_meca[GlobalTable_meca['manipID'].str.contains(dates)]
        
    if not activationType == 'all':
        GlobalTable_meca = GlobalTable_meca[GlobalTable_meca['activation type'] == activationType]
    
    
    allFiles = np.unique(GlobalTable_meca['cellID'])
    # print(allFiles)
    lenSubplots = len(allFiles)
    rows= int(np.floor(np.sqrt(lenSubplots)))
    cols= int(np.ceil(lenSubplots/rows))
    fontsize = 15
    
    if selectedStressRange == 'all':
       for selectedStressRange in stressRanges:
           print(selectedStressRange)
           
           fig, axes = plt.subplots(nrows = rows, ncols = cols, figsize = (15,15))
           _axes = []
           
           for ax_array in axes:
               for ax in ax_array:
                   _axes.append(ax)

           for cellID, ax in zip(allFiles, _axes):
               print(cellID)
               GlobalTable_meca_spec = GlobalTable_meca[(GlobalTable_meca['cellID'] == cellID)]
               categories = (GlobalTable_meca_spec['validatedFit_'+selectedStressRange] == True).values
               # try: 
               firstActivation = GlobalTable_meca_spec['first activation'].values
               activationType = GlobalTable_meca_spec['activation type']
                
               if np.sum(activationType.isna().values) == 0 and len(activationType) != 0:
                   activationType = activationType.values[0]
               else:
                   figTitle = 'none'
                   activationType = 'none'

               KChadwick = GlobalTable_meca_spec['KChadwick_'+selectedStressRange].values
               compNum = GlobalTable_meca_spec['compNum'].values
              

               if activationType == 'global':
                   markerColor = 'orange'
               elif activationType == 'at beads':
                   markerColor = 'blue'
               elif activationType == 'away from beads':   
                   markerColor = 'green'
               elif activationType == 'none':
                   markerColor = 'black'
                   
               categories = np.asarray(categories*1)
               colormap = np.asarray(['r', markerColor])
               labels = np.asarray(['Not valid', activationType])
               ax.scatter(compNum, KChadwick, c = colormap[categories], label = labels[categories])
               ax.scatter(compNum, KChadwick, color = markerColor, label = activationType)
               ax.set_title(cellID[9:], fontsize = 15)
               
               
               ax.set_ylim(0, 40000)
               ax.set_xlim(0, 20)
               
               plt.setp(ax.get_xticklabels(), fontsize=fontsize)
               plt.setp(ax.get_yticklabels(), fontsize=fontsize)
               fig.suptitle('KChadwick_'+selectedStressRange+' vs. CompNum_'+dates+'_'+activationType)
               
               try:
                   os.mkdir(figDir)
               except:
                   pass
               
           fig.tight_layout()
           plt.savefig(figDir+'/TModulusVsComp_'+(selectedStressRange[:-6])+'_'+activationType+'.png')
           
           plt.show()
       plt.close('all')
    else:
        
        fig, axes = plt.subplots(nrows = rows, ncols = cols, figsize = (15,15))
        _axes = []
        
        for ax_array in axes:
            for ax in ax_array:
                _axes.append(ax)

        for cellID, ax in zip(allFiles, _axes):
            print(cellID)
            GlobalTable_meca_spec = GlobalTable_meca[(GlobalTable_meca['cellID'] == cellID)]
            
            try:
                firstActivation = GlobalTable_meca_spec['first activation'].values
                activationType = GlobalTable_meca_spec['activation type'].values
                
                figTitle = activationType 
                print(activationType)
                if activationType.isna or firstActivation.isna:
                    figTitle = 'none'
                    activationType = 'none'
                    
                else:
                    activationType = activationType[0]
                    firstActivation = firstActivation[0]
                    
            except:
                print('No activation parameters found for cell '+cellID)
            
            
            KChadwick = GlobalTable_meca_spec['KChadwick_'+selectedStressRange].values
            compNum = GlobalTable_meca_spec['compNum'].values
           
            
            if activationType == 'global':
                markerColor = 'orange'
            elif activationType == 'at beads':
                markerColor = 'blue'
            elif activationType == 'away from beads':   
                markerColor = 'green'
            elif activationType == 'none':
                markerColor = 'black'
            ax.scatter(compNum, KChadwick, color = markerColor, label = activationType)
            ax.set_title(cellID+'-'+figTitle, fontsize = 15)
            
            
            ax.set_ylim(0, 40000)
            ax.set_xlim(0, 20)
            
            plt.setp(ax.get_xticklabels(), fontsize=fontsize)
            plt.setp(ax.get_yticklabels(), fontsize=fontsize)
            fig.suptitle('KChadwick_'+cellID+'_'+selectedStressRange+' vs. CompNum_'+activationType)
            try:
                os.mkdir(figDir)
            except:
                pass
            
            plt.savefig(figDir+'/TModulusVsComp_'+(selectedStressRange[:-6])+'_'+activationType+'.png')
            plt.show()
        plt.close()

#%%%% Plotting TModulus vs. Compression Number
selectedStressRange = 'all'
dates = '22.03.31'
activationType = 'all'
figDir = todayFigDir+'/TModulusVsCompression Plots_'+dates
TmodulusVsCompression(figDir, GlobalTable_meca, dates, selectedStressRange, activationType)

#%%%%  

dates = '22.05.31'
activationType = 'all'
bestH0vsCompression(GlobalTable_meca, dates, activationType)



#%%%%

#%% Statistics

def w_std(x, w):
    m = np.average(x, weights=w)
    v = np.average((x-m)**2, weights=w)
    std = v**0.5
    return(std)

def nan2zero(x):
    if np.isnan(x):
        return(0)
    else:
        return(x)
    
def nan2neg(x):
    if np.isnan(x):
        return(-1)
    else:
        return(x)

def addStat_df(ax, data, box_pairs, param, cond, test = 'Mann-Whitney', percentHeight = 95):
    refHeight = np.percentile(data[param].values, percentHeight)
    currentHeight = refHeight
    scale = ax.get_yscale()
    xTicks = ax.get_xticklabels()
    dictXTicks = {xTicks[i].get_text() : xTicks[i].get_position()[0] for i in range(len(xTicks))}
    for bp in box_pairs:
        c1 = data[data[cond] == bp[0]][param] #.values
        c2 = data[data[cond] == bp[1]][param] #.values
        
        if test == 'Mann-Whitney' or test == 'Wilcox_2s' or test == 'Wilcox_greater' or test == 'Wilcox_less' or test == 't-test':
            if test=='Mann-Whitney':
                statistic, pval = st.mannwhitneyu(c1,c2)
            elif test=='Wilcox_2s':
                statistic, pval = st.wilcoxon(c1,c2, alternative = 'two-sided')
            elif test=='Wilcox_greater':
                statistic, pval = st.wilcoxon(c1,c2, alternative = 'greater')
            elif test=='Wilcox_less':
                statistic, pval = st.wilcoxon(c1,c2, alternative = 'less')
            elif test=='t-test':
                statistic, pval = st.ttest_ind(c1,c2)
            text = 'ns'
            if pval == np.nan:
                text = 'nan'
            if pval < 0.05 and pval > 0.01:
                text = '*'
            elif pval < 0.01 and pval > 0.001:
                text = '**'
            elif pval < 0.001 and pval < 0.001:
                text = '***'
            elif pval < 0.0001:
                print('yes')
                text = '****'
            
            # print('Pval')
            # print(pval)
            ax.plot([bp[0], bp[1]], [currentHeight, currentHeight], 'k-', lw = 1)
            XposText = (dictXTicks[bp[0]]+dictXTicks[bp[1]])/2
            
            if scale == 'log':
                power = 0.01* (text=='ns') + 0.000 * (text!='ns')
                YposText = currentHeight*(refHeight**power)
            else:
                factor = 0.03 * (text=='ns') + 0.000 * (text!='ns')
                YposText = currentHeight + factor*refHeight
            
            # if XposText == np.nan or YposText == np.nan:
            #     XposText = 0
            #     YposText = 0
                
            ax.text(XposText, YposText, text, ha = 'center', color = 'k')
    #         if text=='ns':
    #             ax.text(posText, currentHeight + 0.025*refHeight, text, ha = 'center')
    #         else:
    #             ax.text(posText, currentHeight, text, ha = 'center')
            if scale == 'log':
                currentHeight = currentHeight*(refHeight**0.05)
            else:
                currentHeight =  currentHeight + 0.15*refHeight
        # ax.set_ylim([ax.get_ylim()[0], currentHeight])

        if test == 'pairwise':
            ratio = (c2/c1)
            stdError = np.nanstd(ratio)/np.sqrt(np.size(c1))
            confInt = np.nanmean(ratio) - 1.96 * stdError
            print(stdError)
            print(confInt)
            return confInt



#%% Creating a summaryDf for experiments with transient activation
selectedStressRange = 'all' #['S=150'] 
dates = '22.03.31'
activationType = 'all'
save = True

GlobalTable_meca = tka.getGlobalTable(kind = 'Global_MecaData_AJ_22-03-31')
GlobalTable_meca.head()

fitC =  np.array([S for S in range(100, 1150, 50)])
fitW = [100,150,200] # [150] # [200, 250, 300]

fitCenters = np.array([[int(S) for S in fitC] for w in fitW]).flatten()
fitWidth = np.array([[int(w) for S in fitC] for w in fitW]).flatten()

fitMin = np.array([[int(S-(w/2)) for S in fitC] for w in fitW]).flatten()
fitMax = np.array([[int(S+(w/2)) for S in fitC] for w in fitW]).flatten()

fitCenters, fitWidth = fitCenters[fitMin>0], fitWidth[fitMin>0]
fitMin, fitMax = fitMin[fitMin>0], fitMax[fitMin>0]

stressRanges = ['S='  + str(fitCenters[ii]) + '+/-' + str(int(fitWidth[ii]//2)) for ii in range(len(fitCenters))]


try:
    directory =  os.path.join(todayFigDir, "Mechanics_PopulationSummaryPlots")
    os.mkdir(directory)
except:
    pass
    
if not dates == 'all':
    GlobalTable_meca = GlobalTable_meca[GlobalTable_meca['manipID'].str.contains(dates)]
    
if not activationType == 'all':
    GlobalTable_meca = GlobalTable_meca[GlobalTable_meca['activation type'] == activationType]

if not selectedStressRange == 'all':
    stressRanges = selectedStressRange

allCells = np.unique(GlobalTable_meca['cellID'])
# print(allFiles)
lenSubplots = len(allCells)
rows= int(np.floor(np.sqrt(lenSubplots)))
cols= int(np.ceil(lenSubplots/rows))
fontsize = 8

expDf = ufun.getExperimentalConditions(experimentalDataDir, save = False)
GlobalTable_meca['activationTag'] = np.nan*len(GlobalTable_meca)
 
GlobalTable_meca['activationTag'][(GlobalTable_meca['activation type'] == 'global') & \
                                  (GlobalTable_meca['compNum'] < 5)] = 'globalBefore'

GlobalTable_meca['activationTag'][(GlobalTable_meca['activation type'] == 'global') & \
                                  (GlobalTable_meca['compNum'] >= 5)] = 'globalActivated'

GlobalTable_meca['activationTag'][(GlobalTable_meca['activation type'] == 'at beads') & \
                                  (GlobalTable_meca['compNum'] < 5)] = 'atBeadsBefore'

GlobalTable_meca['activationTag'][(GlobalTable_meca['activation type'] == 'at beads') & \
                                  (GlobalTable_meca['compNum'] >= 5)] = 'atBeadsActivated'

GlobalTable_meca['activationTag'][(GlobalTable_meca['activation type'] == 'away from beads') & \
                                  (GlobalTable_meca['compNum'] < 5)] = 'awayBeadsBefore'

GlobalTable_meca['activationTag'][(GlobalTable_meca['activation type'] == 'away from beads') & \
                                  (GlobalTable_meca['compNum'] >= 5)] = 'awayBeadsActivated'
    

#%% Initialise dataframe
selectedStressRange = 'all' #['S=150'] 
dates = 'all'
activationType = 'all'
save = True

GlobalTable_meca = tka.getGlobalTable(kind = 'Global_MecaData_AJ2')
GlobalTable_meca.head()

fitC =  np.array([S for S in range(100, 1150, 50)])
# fitW = [100,150,200] # [150] # [200, 250, 300]
fitW = [100, 150, 200]

fitCenters = np.array([[int(S) for S in fitC] for w in fitW]).flatten()
fitWidth = np.array([[int(w) for S in fitC] for w in fitW]).flatten()

fitMin = np.array([[int(S-(w/2)) for S in fitC] for w in fitW]).flatten()
fitMax = np.array([[int(S+(w/2)) for S in fitC] for w in fitW]).flatten()

fitCenters, fitWidth = fitCenters[fitMin>0], fitWidth[fitMin>0]
fitMin, fitMax = fitMin[fitMin>0], fitMax[fitMin>0]

stressRanges = ['S='  + str(fitCenters[ii]) + '+/-' + str(int(fitWidth[ii]//2)) for ii in range(len(fitCenters))]


KChadwickLim = 50000

try:
    directory =  os.path.join(todayFigDir, "Mechanics_PopulationSummaryPlots")
    os.mkdir(directory)
except:
    pass
    
if not dates == 'all':
    GlobalTable_meca = GlobalTable_meca[GlobalTable_meca['manipID'].str.contains(dates)]
    
if not activationType == 'all':
    GlobalTable_meca = GlobalTable_meca[GlobalTable_meca['activation type'] == activationType]

if not selectedStressRange == 'all':
    stressRanges = selectedStressRange

allCells = np.unique(GlobalTable_meca['cellID'])
# print(allFiles)
lenSubplots = len(allCells)
rows= int(np.floor(np.sqrt(lenSubplots)))
cols= int(np.ceil(lenSubplots/rows))
fontsize = 8
allActivationTags = []
expDf = ufun.getExperimentalConditions(experimentalDataDir, save = False)

GlobalTable_meca['activationTag'] = [0]*len(GlobalTable_meca)
GlobalTable_meca['cellID_new'] = [0]*len(GlobalTable_meca)
GlobalTable_meca['manip_new'] = [0]*len(GlobalTable_meca)

plotDf = {'cellID': [],
          'manip_new' : [],
          'activationTag': [],
          'cellID_new': []}

plotDf2 = {'cellID': [],
           'activationTag': [],
           'minStress': [],
           'maxStress': [],
           'compNum': []}

for selectedStressRange in stressRanges:
    GlobalTable_meca['AvgKChadwick_'+selectedStressRange] = [np.nan]*len(GlobalTable_meca)
    toAdd = {'AvgKChadwick_'+selectedStressRange: []}
    toAddK2 = {'ratioKChadwick_'+selectedStressRange: []}
    plotDf.update(toAdd)
    plotDf.update(toAddK2)
    
    toAddK = {'KChadwick_'+selectedStressRange: []}
    toAddWeight = {'KWeight_'+selectedStressRange: []}
    toAddCIW = {'K_CIW_'+selectedStressRange: []}
    plotDf2.update(toAddK)
    plotDf2.update(toAddWeight)
    plotDf2.update(toAddCIW)

plotDf = pd.DataFrame(plotDf)
plotDf2 = pd.DataFrame(plotDf2)

plotDf['cellID'] = [np.nan]*len(allCells)
plotDf['manip_new'] = [np.nan]*len(allCells)
plotDf['activationTag'] = [np.nan]*len(allCells)
plotDf['cellID_new'] = [np.nan]*len(allCells)


# GlobalTable_meca['activationTag'][GlobalTable_meca['manip'] == 'M1'] = 'AtBeadsBefore'

GlobalTable_meca['activationTag'][GlobalTable_meca['manip'] == 'M1'] = 'Before'
GlobalTable_meca['activationTag'][GlobalTable_meca['manip'] == 'M10'] = 'AtBeadsActivated'
GlobalTable_meca['activationTag'][GlobalTable_meca['manip'] == 'M2'] = np.nan
GlobalTable_meca['activationTag'][GlobalTable_meca['manip'] == 'M5'] = 'AtBeadsActivated'
GlobalTable_meca['activationTag'][GlobalTable_meca['manip'] == 'M6'] = np.nan
# GlobalTable_meca['activationTag'][GlobalTable_meca['manip'] == 'M3'] = 'AwayBeadsBefore'
GlobalTable_meca['activationTag'][GlobalTable_meca['manip'] == 'M3'] = 'Before'

GlobalTable_meca['activationTag'][GlobalTable_meca['manip'] == 'M7'] = 'AwayBeadsActivated'


plotDf2['cellID'] = GlobalTable_meca['cellID']
plotDf2['minStress'] = GlobalTable_meca['minStress']
plotDf2['maxStress'] = GlobalTable_meca['maxStress']
plotDf2['compNum'] = GlobalTable_meca['compNum']
plotDf2['activationTag'] = GlobalTable_meca['activationTag']


plotDf2 = plotDf2[plotDf2['cellID'].str.contains('22-08-26_M6_P1_C1') == False]
plotDf2 = plotDf2[plotDf2['cellID'].str.contains('22-08-26_M2_P1_C1') == False]
plotDf2 = plotDf2[plotDf2['cellID'].str.contains('22-08-26_M5_P1_C2') == False]

for cell in allCells:
    cellSpec = GlobalTable_meca['cellID'] == cell
    # print(cell)
    manip = GlobalTable_meca['manip'][cellSpec].values[0]
    # print(manip)
    if manip == 'M1' or manip == 'M2' or manip == 'M3':    
        manip_new = manip
        GlobalTable_meca['cellID_new'][cellSpec] = cell
        GlobalTable_meca['manip_new'][cellSpec] = manip_new
    elif manip == 'M5':     
        cellID = cell.replace('M5', 'M1')
        manip_new = 'M1'
        GlobalTable_meca['cellID_new'][cellSpec] = cellID
        GlobalTable_meca['manip_new'][cellSpec] = manip_new
    elif manip == 'M10':
        cellID = cell.replace('M10', 'M1')
        manip_new = 'M1'
        GlobalTable_meca['cellID_new'][cellSpec] = cellID
        GlobalTable_meca['manip_new'][cellSpec] = manip_new
    elif manip == 'M7':
        manip_new = 'M3'
        cellID = cell.replace('M7', 'M3')
        GlobalTable_meca['cellID_new'][cellSpec] = cellID
        GlobalTable_meca['manip_new'][cellSpec] = manip_new
    elif manip == 'M6':
        manip_new = 'M2'
        cellID = cell.replace('M6', 'M2')
        GlobalTable_meca['cellID_new'][cellSpec] = cellID
        GlobalTable_meca['manip_new'][cellSpec] = manip_new
    
    
    for selectedStressRange in stressRanges:
        K = GlobalTable_meca['KChadwick_'+selectedStressRange][cellSpec]
        avgK = np.unique(np.nanmean(K))[0]
        GlobalTable_meca['AvgKChadwick_'+selectedStressRange][cellSpec] = avgK
    
            
#%%
for (cell, i) in zip(allCells, plotDf.index):
    cellSpec = GlobalTable_meca['cellID'] == cell
    plotDf.at[i, 'cellID'] = cell
    plotDf.at[i, 'cellID_new'] = np.unique(GlobalTable_meca['cellID_new'][cellSpec].values)[0]
    plotDf.at[i, 'manip_new'] =  np.unique(GlobalTable_meca['manip_new'][cellSpec].values)[0]
    plotDf.at[i, 'activationTag'] =  np.unique(GlobalTable_meca['activationTag'][cellSpec].values)[0]
    for selectedStressRange in stressRanges:
        ratioDf = GlobalTable_meca
        avgK = np.unique(np.nanmean(GlobalTable_meca['KChadwick_'+selectedStressRange][cellSpec]))[0]
        if avgK < KChadwickLim:
            plotDf.at[i, 'AvgKChadwick_'+selectedStressRange] =  avgK
        else:
            plotDf.at[i, 'AvgKChadwick_'+selectedStressRange] =  np.nan

#Removing rows with no pairs
plotDf = plotDf[plotDf['cellID'].str.contains('22-08-26_M1_P1_C3') == False]
plotDf = plotDf[plotDf['cellID'].str.contains('22-08-26_M1_P1_C4') == False]
plotDf = plotDf[plotDf['cellID'].str.contains('22-08-26_M1_P1_C5') == False]

plotDf = plotDf[plotDf['cellID'].str.contains('22-08-26_M5_P1_C2') == False]
plotDf = plotDf[plotDf['cellID'].str.contains('22-08-26_M1_P2_C1') == False]
plotDf = plotDf[plotDf['cellID'].str.contains('22-08-26_M1_P1_C6') == False]
plotDf = plotDf[plotDf['cellID'].str.contains('22-08-26_M2_P1_C1') == False]
plotDf = plotDf[plotDf['cellID'].str.contains('22-08-26_M6_P1_C1') == False]
plotDf = plotDf[plotDf['cellID'].str.contains('22-08-26_M3_P2_C6') == False]
plotDf = plotDf[plotDf['cellID'].str.contains('22-08-26_M3_P2_C7') == False]


plotDf['cellID'][plotDf['cellID'].str.contains('22-08-26_M10_P1_C2')] = '22-08-26_M5_P1_C2'
plotDf = plotDf.sort_values(by='cellID')

allCellsDf = plotDf['cellID'].values
allKRatio = []
for (cell, i) in zip(allCellsDf, plotDf.index):
    if "M1" in cell:
        print(cell)
        tag1 = 'AtBeadsBefore'
        tag2 = 'AtBeadsActivated'
        cellSpec = plotDf['cellID_new'] == cell
        ratioDf = plotDf[cellSpec]
        for selectedStressRange in stressRanges:
            tagBefore = ratioDf['activationTag'] == tag1
            tagAfter = ratioDf['activationTag'] == tag2
            KBefore = ratioDf['AvgKChadwick_'+selectedStressRange][tagBefore].values
            KActive = ratioDf['AvgKChadwick_'+selectedStressRange][tagAfter].values
           
            ratioBefore = np.round(KBefore/KBefore, 2)                                        
            ratioAfter = np.round(KActive/KBefore, 2)
            indices = ratioDf.index
            print(indices)
            idB = indices[0]
            idA = indices[1]
            # if cell == '22-08-26_M10_P1_C2':
            #     idB = indices[1]
                # idA = indices[0]
            plotDf.at[idB, 'ratioKChadwick_'+selectedStressRange] = ratioBefore
            plotDf.at[idA, 'ratioKChadwick_'+selectedStressRange] = ratioAfter
            
    elif "M3" in cell:
        print(cell)
        tag1 = 'AwayBeadsBefore'
        tag2 = 'AwayBeadsActivated'
        cellSpec = plotDf['cellID_new'] == cell
        ratioDf = plotDf[plotDf['cellID_new'] == cell]
        for selectedStressRange in stressRanges:
            tagBefore = ratioDf['activationTag'] == tag1
            tagAfter = ratioDf['activationTag'] == tag2
            KBefore = ratioDf['AvgKChadwick_'+selectedStressRange][tagBefore].values
            KActive = ratioDf['AvgKChadwick_'+selectedStressRange][tagAfter].values
            ratioBefore = np.round(KBefore/KBefore, 2)                                        
            ratioAfter = np.round(KActive/KBefore, 2)
            indices = ratioDf.index
            plotDf.at[indices[0], 'ratioKChadwick_'+selectedStressRange] = ratioBefore
            plotDf.at[indices[1], 'ratioKChadwick_'+selectedStressRange] = ratioAfter

#%%

K = 'K2'
GlobalTable_meca = tka.getGlobalTable(kind = 'Global_MecaData_AJ_22-12-07')
GlobalTable_meca.head()

plotDf2 = {'cellID': [],
           'activationTag': [],
           'minStress': [],
           'maxStress': [],
           'compNum': []}

KChadwickLim = 50000

fitC =  np.array([S for S in range(100, 1150, 50)])
fitW = [100,150,200] # [150] # [200, 250, 300]
# fitW = [150] #, 150, 200]

fitCenters = np.array([[int(S) for S in fitC] for w in fitW]).flatten()
fitWidth = np.array([[int(w) for S in fitC] for w in fitW]).flatten()

fitMin = np.array([[int(S-(w/2)) for S in fitC] for w in fitW]).flatten()
fitMax = np.array([[int(S+(w/2)) for S in fitC] for w in fitW]).flatten()

fitCenters, fitWidth = fitCenters[fitMin>0], fitWidth[fitMin>0]
fitMin, fitMax = fitMin[fitMin>0], fitMax[fitMin>0]

stressRanges = ['S='  + str(fitCenters[ii]) + '+/-' + str(int(fitWidth[ii]//2)) for ii in range(len(fitCenters))]

plotDf2['cellID'] = GlobalTable_meca['cellID']
plotDf2['date'] = GlobalTable_meca['date']
plotDf2['minStress'] = GlobalTable_meca['minStress']
plotDf2['maxStress'] = GlobalTable_meca['maxStress']
plotDf2['compNum'] = GlobalTable_meca['compNum']
plotDf2['activationTag'] = GlobalTable_meca['manip']
plotDf2['bestH0'] = GlobalTable_meca['bestH0']
plotDf2['surroundingThickness'] = GlobalTable_meca['surroundingThickness']

plotDf2 = pd.DataFrame(plotDf2)



for selectedStressRange in stressRanges:
    plotDf2['KChadwick_'+selectedStressRange] = GlobalTable_meca[K+'Chadwick_'+selectedStressRange]
    plotDf2['KChadwick_'+selectedStressRange] *= GlobalTable_meca[K+'Chadwick_'+selectedStressRange].apply(lambda x : (x<KChadwickLim))
    plotDf2['KChadwick_'+selectedStressRange] *= GlobalTable_meca['R2Chadwick_'+selectedStressRange].apply(lambda x : (x>1e-2))
    plotDf2['KChadwick_'+selectedStressRange] *= GlobalTable_meca[K+'_CIW_'+selectedStressRange].apply(lambda x : (x!=0))
    
    K_CIWidth = GlobalTable_meca[K+'_CIW_'+selectedStressRange] #.apply(lambda x : x.strip('][').split(', ')).apply(lambda x : (np.abs(float(x[0]) - float(x[1]))))
    plotDf2['K_CIW_'+selectedStressRange] = K_CIWidth
    plotDf2['K_CIW_'+selectedStressRange] *= GlobalTable_meca[K+'Chadwick_'+selectedStressRange].apply(lambda x : (x<KChadwickLim))
    plotDf2['K_CIW_'+selectedStressRange] *= GlobalTable_meca['R2Chadwick_'+selectedStressRange].apply(lambda x : (x>1e-2))
    plotDf2['K_CIW_'+selectedStressRange] *= GlobalTable_meca[K+'_CIW_'+selectedStressRange].apply(lambda x : (x!=0))
    
    
    KWeight = (GlobalTable_meca[K+'Chadwick_'+selectedStressRange]/K_CIWidth)**2
    plotDf2['KWeight_'+selectedStressRange] = KWeight
    plotDf2['KWeight_'+selectedStressRange] *= GlobalTable_meca[K+'Chadwick_'+selectedStressRange].apply(lambda x : (x<KChadwickLim))
    plotDf2['KWeight_'+selectedStressRange] *= GlobalTable_meca['R2Chadwick_'+selectedStressRange].apply(lambda x : (x>1e-2))
    plotDf2['KWeight_'+selectedStressRange] *= GlobalTable_meca[K+'_CIW_'+selectedStressRange].apply(lambda x : (x!=0))

    

#%%% Boxplots

plot = 2
for selectedStressRange in stressRanges:
    # print(selectedStressRange)
    
    if plot == 1:
        dfValid = GlobalTable_meca[GlobalTable_meca['validatedFit_'+selectedStressRange] == True]
        dfValid = GlobalTable_meca[GlobalTable_meca['R2Chadwick_'+selectedStressRange] > 0.02]
        
        ####Whole boxplots with all activation tags
        y = 'KChadwick_'+selectedStressRange
        
        fig1, axes = plt.subplots(1,1, figsize=(20,10))
        
        try:
            axes = sns.boxplot(x = 'activationTag', y=y, data=dfValid, ax=axes,
                               medianprops={"color": 'darkred', "linewidth": 2}, 
                               boxprops={"edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    
            axes = sns.swarmplot(x = 'activationTag', y=y, data=dfValid,ax=axes, color = 'black', size = 7)
        except:
            pass
        
        try:
            addStat_df(axes, dfValid, [('AtBeadsBefore', 'AtBeadsActivated')], 'KChadwick_'+selectedStressRange, \
                    test = 'Mann-Whitney', cond = 'activationTag')
        except:
            pass
        
        try:
            addStat_df(axes, dfValid, [('AwayBeadsBefore', 'AwayBeadsActivated')], 'KChadwick_'+selectedStressRange, \
                        test = 'Mann-Whitney', cond = 'activationTag')
            
        except:
            pass
         
        plt.rcParams.update({'font.size': 16})
        fig1.tight_layout()
        fig1.suptitle('KChadwick_'+selectedStressRange)
        
        for patch in axes.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
    
        if save == True:
            stressRangefile = selectedStressRange .replace('/', '_')
            plt.savefig(directory+'/BoxTModulus_All_'+str(stressRangefile)+'.png')
        
        plt.show()
        plt.close()
        
    #### Activationtag boxplots

    if plot == 2:
        fig2, axes = plt.subplots(1,2, figsize=(20,10))
        
        dfValid = plotDf[plotDf['manip_new'] == 'M1']
                
        sns.boxplot(x = "activationTag", y='AvgKChadwick_'+selectedStressRange, data=dfValid,\
                            order = ["AtBeadsBefore", "AtBeadsActivated"], ax = axes[0], \
                            medianprops={"color": 'darkred', "linewidth": 2},\
                            boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
            
        sns.swarmplot(x = 'activationTag', y='AvgKChadwick_'+selectedStressRange,\
                              data=dfValid, linewidth = 2, ax = axes[0],\
                              hue = 'cellID_new', edgecolor='k',\
                              size = 20)
            
        axes[0].set_ylim(0, 50000)
        axes[0].set_xticklabels(['AtBeadsBefore', 'AtBeadsActivated'])
        plt.setp(axes[0].get_legend().get_texts(), fontsize='10')

        addStat_df(axes[0], dfValid, [('AtBeadsBefore', 'AtBeadsActivated')], 'AvgKChadwick_'+selectedStressRange, \
                    test = 'Wilcox_greater', cond = 'activationTag')
        
        
        plt.rcParams.update({'font.size': 25})
        
        axes[0].set_title('At beads')
        
        
        dfValid = plotDf[plotDf['manip_new'] == 'M3']
        
        
                
        sns.boxplot(x = 'activationTag', y='AvgKChadwick_'+selectedStressRange, data=dfValid,\
                                  medianprops={"color": 'darkred', "linewidth": 2}, ax = axes[1], 
                                  boxprops={"edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
            
        sns.swarmplot(x = 'activationTag', y='AvgKChadwick_'+selectedStressRange,\
                             data=dfValid,ax = axes[1], linewidth = 2,\
                             hue = 'cellID_new', edgecolor='k',\
                             size = 20)
            
        axes[1].set_ylim(0, 50000)
        plt.setp(axes[1].get_legend().get_texts(), fontsize='10')

        
        addStat_df(axes[1], dfValid, [('AwayBeadsBefore', 'AwayBeadsActivated')], 'AvgKChadwick_'+selectedStressRange, \
                    test = 'Wilcox_greater', cond = 'activationTag')
        
            
        plt.rcParams.update({'font.size': 25})
        
        axes[1].set_title('Away from beads')
        
        fig2.suptitle('AvgKChadwick_'+selectedStressRange)
        fig2.tight_layout()
        
        if save == True:
            stressRangefile = selectedStressRange .replace('/', '_')
            plt.savefig(directory+'/ActivationTypeBased_BoxTModulus_'+str(stressRangefile)+'.png')

        plt.show()
        plt.close('all')
        
    # if plot == 3:

        
#%% Ratio of K Scatter plot


dfValid = plotDf[(plotDf['activationTag'] == 'AtBeadsActivated') | (plotDf['activationTag'] == 'AwayBeadsActivated')]

for selectedStressRange in stressRanges:
    plt.rcParams.update({'font.size': 25})
    fig, ax = plt.subplots(1,1, figsize=(20,10))
    y = dfValid['ratioKChadwick_'+selectedStressRange]
    x = dfValid['cellID_new']
    sns.scatterplot(x = 'cellID_new', y = 'ratioKChadwick_'+selectedStressRange, ax = ax,\
                    data = dfValid, hue = 'activationTag', s = 200, edgecolor = 'k', linewidth = 2)
    ax.set_xticks(dfValid['cellID_new'])
    ax.set_ylim(0,3)
    plt.axhline(y=1, color = 'red')
    ax.set_yticklabels(ax.get_yticks(), size = 15)
    fig.tight_layout()
    figName = selectedStressRange.replace('/','_')
    filePath = todayFigDir+'/Mechanics_PopulationSummaryPlots/RatioScatter/'
    try:
        os.mkdir(filePath)
    except:
        pass
    plt.savefig(filePath+figName+'.jpg')
    plt.show()
    plt.close()
    
#%% non-linear analysis

#### Whole curve
plt.style.use('dark_background')

width = [150, 200]
fitC =  np.array([S for S in range(100, 1150, 50)])
# fitW = [100,150,200] # [150] # [200, 250, 300]
fitW = [150]

fitCenters = np.array([[int(S) for S in fitC] for w in fitW]).flatten()
fitWidth = np.array([[int(w) for S in fitC] for w in fitW]).flatten()

fitMin = np.array([[int(S-(w/2)) for S in fitC] for w in fitW]).flatten()
fitMax = np.array([[int(S+(w/2)) for S in fitC] for w in fitW]).flatten()

fitCenters, fitWidth = fitCenters[fitMin>0], fitWidth[fitMin>0]
fitMin, fitMax = fitMin[fitMin>0], fitMax[fitMin>0]

stressRanges = ['S='  + str(fitCenters[ii]) + '+/-' + str(int(fitWidth[ii]//2)) for ii in range(len(fitCenters))]


# conditions = np.array(plotDf2['activationTag'].unique())
# # conditions = np.asarray(['AtBeadsBefore', 'AtBeadsActivated'])
# # conditions = np.asarray(['AwayBeadsBefore', 'AwayBeadsActivated'])


# cD = {'AtBeadsBefore':[gs.colorList40[10], gs.colorList40[10]],
#       'AtBeadsActivated':[gs.colorList40[30], gs.colorList40[30]],
#       'AwayBeadsBefore':[gs.colorList40[12], gs.colorList40[12]],
#       'AwayBeadsActivated':[gs.colorList40[32], gs.colorList40[32]]}

# conditions = np.array(plotDf2['activationTag'].unique())
# conditions = np.asarray(['AtBeadsBefore', 'AtBeadsActivated']) 
# conditions = np.asarray(['AwayBeadsBefore', 'AwayBeadsActivated'])
# conditions = np.asarray(['AwayBeadsBefore', 'AtBeadsBefore'])
conditions = ['Before']

cD = {'Before':[gs.colorList40[12], gs.colorList40[12]]}
    

dfValid = plotDf2

#%%

#### Whole curve
plt.style.use('dark_background')


fitC =  np.array([S for S in range(100, 1150, 50)])
# fitW = [100,150,200] # [150] # [200, 250, 300]
fitW = [150]

fitCenters = np.array([[int(S) for S in fitC] for w in fitW]).flatten()
fitWidth = np.array([[int(w) for S in fitC] for w in fitW]).flatten()

fitMin = np.array([[int(S-(w/2)) for S in fitC] for w in fitW]).flatten()
fitMax = np.array([[int(S+(w/2)) for S in fitC] for w in fitW]).flatten()

fitCenters, fitWidth = fitCenters[fitMin>0], fitWidth[fitMin>0]
fitMin, fitMax = fitMin[fitMin>0], fitMax[fitMin>0]

stressRanges = ['S='  + str(fitCenters[ii]) + '+/-' + str(int(fitWidth[ii]//2)) for ii in range(len(fitCenters))]


# conditions = np.array(GlobalTable_meca['activationTag'].unique())
conditions = ['atBeadsBefore', 'atBeadsActivated', 'globalBefore', 'globalActivated']

for selectedStressRange in stressRanges:
    K_CIWidth = GlobalTable_meca['K_CIW_'+selectedStressRange]
    KWeight = (GlobalTable_meca['KChadwick_'+selectedStressRange]/K_CIWidth)**2
    GlobalTable_meca['KWeight_'+selectedStressRange] = KWeight
    
# cD = {'atBeadsBefore':[gs.colorList40[10], gs.colorList40[10]],
#       'atBeadsActivated':[gs.colorList40[30], gs.colorList40[30]],
#       'awayBeadsBefore':[gs.colorList40[12], gs.colorList40[12]],
#       'awayBeadsActivated':[gs.colorList40[32], gs.colorList40[32]],
#       'globalBefore': [gs.colorList40[14], gs.colorList40[14]],
#       'globalActivated': [gs.colorList40[34], gs.colorList40[34]]}

cD = {'atBeadsBefore':[gs.colorList40[12], gs.colorList40[12]],
      'atBeadsActivated':[gs.colorList40[32], gs.colorList40[32]],
      'globalBefore': [gs.colorList40[14], gs.colorList40[14]],
      'globalActivated': [gs.colorList40[34], gs.colorList40[34]]}



dfValid = GlobalTable_meca
plotDf2 = GlobalTable_meca

#%%
# plt.style.use('dark_background')

GlobalTable_meca = tka.getGlobalTable(kind = 'Global_MecaData_AJ_22-12-07')

GlobalTable_meca = GlobalTable_meca[(GlobalTable_meca['manipID'].str.contains('M1')) | (GlobalTable_meca['manipID'].str.contains('M5'))]

# excluded = ['22-10-06_M6_P3_C5','22-10-06_M6_P3_C7', '22-10-06_M5_P3_C5', '22-10-06_M5_P3_C7', \
#             '22-10-06_M5_P3_C9', '22-10-06_M6_P3_C9', '22-10-06_M2_P1_C4', '22-10-06_M1_P1_C4', '22-10-06_M2_P1_C5', '22-10-06_M1_P1_C5', \
#                         '22-10-06_M2_P2_C5', '22-10-06_M1_P2_C5', '22-10-06_M1_P2_C7', '22-10-06_M1_P2_C8'\
#                         '22-10-06_M2_P2_C7', '22-10-06_M2_P2_C8']
    


# for i in excluded:
#     plotDf2 = plotDf2[plotDf2['cellID'].str.contains(i) == False]

# conditions = ['M1', 'M3', 'M5']
# conditions = ['M2', 'M6']
# conditions = ['M5', 'M6']
# conditions = ['M1', 'M5']
# conditions = ['M1', 'M2', 'M5', 'M6']
# conditions = ['22-10-06']
# conditions = ['M1', 'M2']
# conditions = ['M1']

# conditions = ['M1', 'M2', 'M3', 'M4', 'M5']
conditions = ['M4', 'M5']

plotDf = plotDf2

cD = {'M4':[gs.colorList40[8], gs.colorList40[8]],
         #'M2':[gs.colorList40[28], gs.colorList40[28]],
       # 'M7':[gs.colorList40[10], gs.colorList40[10]],
        #'M8':[gs.colorList40[30], gs.colorList40[30]],
         #  'M9':[gs.colorList40[12], gs.colorList40[12]],
            'M5':[gs.colorList40[32], gs.colorList40[32]]}

# cD = {'22-10-06':[gs.colorList40[32], gs.colorList40[32]]}



fitC =  np.array([S for S in range(150, 1150, 50)])
# fitW = [100,150,200] # [150] # [200, 250, 300]
fitW = [150] #, 150, 200]

fitCenters = np.array([[int(S) for S in fitC] for w in fitW]).flatten()
fitWidth = np.array([[int(w) for S in fitC] for w in fitW]).flatten()

fitMin = np.array([[int(S-(w/2)) for S in fitC] for w in fitW]).flatten()
fitMax = np.array([[int(S+(w/2)) for S in fitC] for w in fitW]).flatten()

fitCenters, fitWidth = fitCenters[fitMin>0], fitWidth[fitMin>0]
fitMin, fitMax = fitMin[fitMin>0], fitMax[fitMin>0]

stressRanges = ['S='  + str(fitCenters[ii]) + '+/-' + str(int(fitWidth[ii]//2)) for ii in range(len(fitCenters))]

#%%


plotDf = plotDf2
fig, axes = plt.subplots(2,1, figsize = (9,12))
ListDfWhole = []
    
# Kavg = []
# Kstd = []
# D10 = []
# D90 = []
# fitC = []
# N = []
plt.rcParams.update({'font.size': 25})

for co in conditions:
    
    Kavg = []
    Kstd = []
    D10 = []
    D90 = []
    N = []
    fitC = []
    
    # dfValid = plotDf2[plotDf2['activationTag'] == co]
    dfValid = plotDf[plotDf['activationTag'] == co]
    for selectedStressRange in stressRanges:
        
        print(selectedStressRange)
        variable = 'KChadwick_'+selectedStressRange
        weight = 'KWeight_'+selectedStressRange
        stressSplit = selectedStressRange.split('=')[1]
        stressSplit = stressSplit.split('+/-')
        centre = stressSplit[0]
        width = stressSplit[1]
        
        x = dfValid[variable].apply(nan2zero).values
        w = dfValid[weight].apply(nan2zero).values
        # print(x)
        S = centre
        # print(S)

        d = {'x' : x, 'w' : w}
        if int(S) > 900:
            break
        
        print(np.sum(d['w']))
        
        if np.sum(d['x']) != 0:
            
            m = np.average(x, weights=w)
            v = np.average((x-m)**2, weights=w)
            std = v**0.5
            
            d10, d90 = np.percentile(x[x != 0], (10, 90))
            n = len(x[x != 0])
            # n = len(x)
            Kavg.append(m)
            # print(m)
            Kstd.append(std)
            D10.append(d10)
            D90.append(d90)
            N.append(n)
            fitC.append(int(centre))
        
        
    Kavg = np.array(Kavg)
    Kstd = np.array(Kstd)
    D10 = np.array(D10)
    D90 = np.array(D90)
    N = np.array(N)
    fitC = np.array(fitC)
    Kste = Kstd / (N**0.5)
    
    alpha = 0.975
    dof = len(N)
    q = st.t.ppf(alpha, dof) # Student coefficient
    
    d_val = {'S' : fitC, 'Kavg' : Kavg, 'Kstd' : Kstd, 'D10' : D10, 'D90' : D90, 'N' : N}
    
    for ax in axes:
        # Weighted means -- Weighted ste 95% as error
        ax.errorbar(fitC, Kavg, yerr = q*Kste, marker = 'o', elinewidth = 0.8, color = cD[co][0], 
                       ecolor = cD[co][1], capsize = 3, label = co)
        
        ax.set_ylim([500,3e4])
        ax.set_xlim([0,1100])
        ax.set_title('K(s) - All compressions pooled (log)')
        
        ax.legend(loc = 'lower left')
        
        ax.set_xlabel('Stress (Pa)')
        ax.set_ylabel('K (Pa)')
        
        for kk in range(len(N)):
            ax.text(x=fitC[kk], y=Kavg[kk]**0.9, s='n='+str(N[kk]), fontsize = 8, color = cD[co][1])
        # for kk in range(len(N)):
        #     ax[k].text(x=fitCenters[kk]+oD[co][0], y=Kavg[kk]**oD[co][1], 
        #                s='n='+str(N[kk]), fontsize = 6, color = cD[co][k])
        
    axes[0].set_yscale('log')
    axes[0].set_ylim([500,3e4])
    
    axes[1].set_ylim([0,1e4])
    axes[1].set_title('K(s) - All compressions pooled (linear scale)')
    # axes[1].set_xlim([0, 700])
    
    width = stressSplit[1]
    # fig.suptitle('K(s)'+'\n(fits width: {:.0f}Pa)'.format(width))   
    fig.suptitle('K(s)_Width='+width)   
    
    df_val = pd.DataFrame(d_val)
    ListDfWhole.append(df_val)
    # dftest = pd.DataFrame(d)

plt.show()

# ufun.archiveFig(fig, name='Pooled_K(s)'+width, figDir = todayFigDir+'/Mechanics_PopulationSummaryPlots', dpi = 100)

#%%
#### Local zoom

Sinf, Ssup = 150, 400
extraFilters = [plotDf2['minStress'] <= Sinf, plotDf2['maxStress'] >= Ssup] # >= 800
fitCenters = fitCenters[(fitCenters>=(Sinf)) & (fitCenters<=Ssup)] # <800
fitWidth = np.array([[int(w) for S in fitCenters] for w in fitW]).flatten()

stressRanges = ['S='  + str(fitCenters[ii]) + '+/-' + str(int(fitWidth[ii]//2)) for ii in range(len(fitCenters))]

dfValid_allCo = plotDf2
globalExtraFilter = extraFilters[0]
for k in range(1, len(extraFilters)):
    globalExtraFilter = globalExtraFilter & extraFilters[k]
dfValid_allCo = dfValid_allCo[globalExtraFilter]  



fig, axes = plt.subplots(2,1, figsize = (9,12))
ListDfZoom = []

for co in conditions:
    
    Kavg = []
    Kstd = []
    D10 = []
    D90 = []
    N = []
    fitC = []
    
    dfValid = dfValid_allCo[dfValid_allCo['activationTag'] == co]

    for selectedStressRange in stressRanges:
        
        print(selectedStressRange)
        variable = 'KChadwick_'+selectedStressRange
        weight = 'KWeight_'+selectedStressRange
        stressSplit = selectedStressRange.split('=')[1]
        stressSplit = stressSplit.split('+/-')
        centre = stressSplit[0]
        width = stressSplit[1]
        
        x = dfValid[variable].apply(nan2zero).values
        w = dfValid[weight].apply(nan2zero).values
        # print(x)
        S = centre
        # print(S)

        d = {'x' : x, 'w' : w}
        if int(S) > Ssup:
            break
        
        print(np.sum(d['w']))
        
        if np.sum(d['x']) != 0:
            
            m = np.average(x, weights=w)
            v = np.average((x-m)**2, weights=w)
            std = v**0.5
            
            d10, d90 = np.percentile(x[x != 0], (10, 90))
            n = len(x[x != 0])
            # n = len(x)
            Kavg.append(m)
            # print(m)
            Kstd.append(std)
            D10.append(d10)
            D90.append(d90)
            N.append(n)
            fitC.append(int(centre))
        
        
    Kavg = np.array(Kavg)
    Kstd = np.array(Kstd)
    D10 = np.array(D10)
    D90 = np.array(D90)
    N = np.array(N)
    fitC = np.array(fitC)
    Kste = Kstd / (N**0.5)
    
    alpha = 0.975
    dof = len(N)
    q = st.t.ppf(alpha, dof) # Student coefficient
    
    d_val = {'S' : fitC, 'Kavg' : Kavg, 'Kstd' : Kstd, 'D10' : D10, 'D90' : D90, 'N' : N}
    
    for ax in axes:
        # Weighted means -- Weighted ste 95% as error
        
        ax.errorbar(fitC, Kavg, yerr = q*Kste, marker = 'o', elinewidth = 0.8, color = cD[co][0], 
                       ecolor = cD[co][1], capsize = 3, label = co)
        
        ax.set_ylim([500,3e4])
        ax.set_xlim([0, Ssup + 50])
        ax.set_title('K(s) - All compressions pooled (log)')
        
        ax.legend(loc = 'lower left')
        
        ax.set_xlabel('Stress (Pa)')
        ax.set_ylabel('K (Pa)')
        
        for kk in range(len(N)):
            ax.text(x=fitC[kk], y=Kavg[kk]**0.9, s='n='+str(N[kk]), fontsize = 8, color = cD[co][1])
        # for kk in range(len(N)):
        #     ax[k].text(x=fitCenters[kk]+oD[co][0], y=Kavg[kk]**oD[co][1], 
        #                s='n='+str(N[kk]), fontsize = 6, color = cD[co][k])
        
    axes[0].set_yscale('log')
    axes[0].set_ylim([500, 3e4])
    
    axes[1].set_ylim([0, 2.1e4])
    axes[1].set_title('K(s) - All compressions pooled (linear scale)')
    # axes[1].set_xlim([0, 700])
    axes[0].xaxis.set_tick_params(labelsize=25)
    axes[1].xaxis.set_tick_params(labelsize=25)
    
    axes[0].yaxis.set_tick_params(labelsize=25)
    axes[1].yaxis.set_tick_params(labelsize=25)
    
    width = stressSplit[1]
    # fig.suptitle('K(s)'+'\n(fits width: {:.0f}Pa)'.format(width))   
    fig.suptitle('Zoomed_K(s)_Width='+width)   
    
    df_val = pd.DataFrame(d_val)
    ListDfZoom.append(df_val)
    # dftest = pd.DataFrame(d)

plt.show()

# ufun.archiveFig(fig, name='Zoomed_Pooled_K(s)_'+width, figDir = todayFigDir+'/Mechanics_PopulationSummaryPlots', dpi = 100)

#%%

selectedStressRange = 'all'
dates = 'all'
activationType = 'at beads'
save = True

GlobalTable_meca = tka.getGlobalTable(kind = 'Global_MecaData_AJ_22-12-07')
GlobalTable_meca.head()


sns.set_style('white')
fitC =  np.array([S for S in range(150, 1150, 50)])
fitW = [100]
fitCenters = np.array([[int(S) for S in fitC] for w in fitW]).flatten()
fitWidth = np.array([[int(w) for S in fitC] for w in fitW]).flatten()
stressRanges = [str(fitCenters[ii]) + '+/-' + str(int(fitWidth[ii]//2)) for ii in range(len(fitCenters))]

try:
    os.mkdir(todayFigDir+'/Mechanics_PopulationSummaryPlots')
except:
    pass
    
if not dates == 'all':
    GlobalTable_meca = GlobalTable_meca[GlobalTable_meca['manipID'].str.contains(dates)]
    
if not activationType == 'all':
    GlobalTable_meca = GlobalTable_meca[GlobalTable_meca['activation type'] == activationType]

if not selectedStressRange == 'all':
    stressRanges = selectedStressRange

allCells = np.unique(GlobalTable_meca['cellID'])
# print(allFiles)
lenSubplots = len(allCells)
rows= int(np.floor(np.sqrt(lenSubplots)))
cols= int(np.ceil(lenSubplots/rows))
fontsize = 8
allActivationTags = []


expDf = ufun.getExperimentalConditions(experimentalDataDir, save = False)

GlobalTable_meca_spec = []
for currentCell in allCells:
    GlobalTable_meca_spec = GlobalTable_meca[(GlobalTable_meca['cellID'] == currentCell)]
    activationType = GlobalTable_meca_spec['activation type'].values[0]
    firstActivation = int(GlobalTable_meca_spec['first activation'].values[0])
    compNum = GlobalTable_meca_spec['compNum']
    
    activationTag = [0]*len(compNum)
    activationTag[:firstActivation] = ["Before"]*(firstActivation)
    activationTag[firstActivation:] = ["After"]*(len(compNum) - firstActivation)
    allActivationTags.extend(activationTag)
    GlobalTable_meca_spec['activationTag'] = activationTag
    
GlobalTable_meca['activationTag'] = allActivationTags

for selectedStressRange in stressRanges:
    print(selectedStressRange)
    dfValid = GlobalTable_meca[GlobalTable_meca['validatedFit_S='+selectedStressRange] == True]
    # try:
        
    dfAtBeads = dfValid[dfValid['activation type'] == 'at beads']
    dfAwayBeads = dfValid[dfValid['activation type'] == 'away from beads']

    y = 'KChadwick_S='+selectedStressRange
    y1 = dfAwayBeads[y]
    
    fig1, axes = plt.subplots(1,2, figsize=(20,10))
    
    axes[0] = sns.boxplot(x = 'activationTag', y=y, data=dfAtBeads, ax = axes[0])
    axes[0] = sns.swarmplot(x = 'activationTag', y=y, data=dfAtBeads, color = 'k', ax = axes[0])
    
    
    axes[1] = sns.boxplot(x = 'activationTag', y=y, data=dfAwayBeads, ax = axes[1])
    axes[1] = sns.swarmplot(x = 'activationTag', y=y, data=dfAwayBeads, color = 'k', ax = axes[1])
    
    axes[1].set_ylim([0,40000])
    axes[0].set_ylim([0,40000])
    
    plt.rcParams.update({'font.size': 22})
    fig1.tight_layout()
    fig1.suptitle('KChadwick_S='+selectedStressRange)
    
    for patch in axes[0].artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .2))
    
    for patch in axes[1].artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .2))
        
    
    plt.ylim(0,40000)
    
    addStat_df(axes[0], dfAtBeads, [('After', 'Before')], 'KChadwick_S='+selectedStressRange, \
                test = 'Mann-Whitney', cond = 'activationTag')
    
    
    
    addStat_df(axes[1], dfAwayBeads, [('After', 'Before')], 'KChadwick_S='+selectedStressRange, \
                test = 'Mann-Whitney', cond = 'activationTag')
    
        
    axes[0].set_title('At Beads')
    axes[1].set_title('Away from beads')
    
    if save == True:
        plt.savefig(todayFigDir+'/SummaryPlots/BoxTModulus_Width50_'+str(selectedStressRange[:-6])+'.png')
        print('Plot saved')
        
    # except:
    #     print('Cell data not good')
    #     pass
    
plt.show()
plt.close('all')


#%%%% H0 vs. Compression No

GlobalTable_meca = tka.getGlobalTable(kind = 'Global_MecaData_AJ_22-12-07')

# excluded = ['22-10-06_M6_P3_C5','22-10-06_M6_P3_C7', '22-10-06_M5_P3_C5', '22-10-06_M5_P3_C7', \
#             '22-10-06_M5_P3_C9', '22-10-06_M6_P3_C9']

# excluded = ['22-10-06_M2_P1_C4', '22-10-06_M1_P1_C4', '22-10-06_M2_P1_C5', '22-10-06_M1_P1_C5', \
#             '22-10-06_M2_P2_C5', '22-10-06_M1_P2_C5', '22-10-06_M1_P2_C7', '22-10-06_M1_P2_C8'\
#             '22-10-06_M2_P2_C7', '22-10-06_M2_P2_C8']

ctrl = 'M5'
active = 'M4'

# for i in excluded:
#     plotDf2 = plotDf2[plotDf2['cellID'].str.contains(i) == False]

df = plotDf2[(plotDf2['activationTag'] == ctrl) | (plotDf2['activationTag'] == active)]

noCells = len(np.unique(plotDf2['cellID'][plotDf2['activationTag'] == 'M4']))


# ch1 = 'M1'
# ch3 = 'M5'

# df = plotDf2[(plotDf2['activationTag'] == ch1)| (plotDf2['activationTag'] == ch3)]

fig1, axes = plt.subplots(1,1, figsize=(15,10))


x = df['compNum']*20
sns.lineplot(x = x, y = 'bestH0', data = df, hue = 'activationTag')

fig1.suptitle(' [15mT = 500pN] bestH0 (nm) vs. Compression No. | n = '+str(noCells))
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('Time (secs)', fontsize = 25)
plt.ylabel('Best H0 (nm)', fontsize = 25)

plt.savefig(todayFigDir + '/22-10-06_15mT_BestH0vsCompr.png')

plt.show()

#%%%% surroundingThickness vs. Compression No

GlobalTable_meca = tka.getGlobalTable(kind = 'Global_MecaData_AJ_22-10-06')


excluded = ['22-10-06_M6_P3_C5','22-10-06_M6_P3_C7', '22-10-06_M5_P3_C5', '22-10-06_M5_P3_C7', \
            '22-10-06_M5_P3_C9', '22-10-06_M6_P3_C9']

# excluded = ['22-10-06_M2_P1_C4', '22-10-06_M1_P1_C4', '22-10-06_M2_P1_C5', '22-10-06_M1_P1_C5', \
#             '22-10-06_M2_P2_C5', '22-10-06_M1_P2_C5', '22-10-06_M1_P2_C7', '22-10-06_M1_P2_C8'\
#             '22-10-06_M2_P2_C7', '22-10-06_M2_P2_C8']

ctrl = 'M5'
active = 'M6'



for i in excluded:
    plotDf2 = plotDf2[plotDf2['cellID'].str.contains(i) == False]

df = plotDf2[(plotDf2['activationTag'] == ctrl) | (plotDf2['activationTag'] == active)]


# ch1 = 'M1'
# ch3 = 'M5'

# df = plotDf2[(plotDf2['activationTag'] == ch1)| (plotDf2['activationTag'] == ch3)]


noCells = len(np.unique(plotDf2['cellID'][plotDf2['activationTag'] == 'M5']))

fig1, axes = plt.subplots(1,1, figsize=(15,10))

x = df['compNum']*20

sns.lineplot(x = x, y = 'surroundingThickness', data = df, hue = 'activationTag')

fig1.suptitle('[15mT = 500pN] Surrounding thickness (nm) vs. Compression No. | n = '+str(noCells))

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('Time (secs)', fontsize = 25)
plt.ylabel('Surrounding Thickness (nm)', fontsize = 25)
plt.savefig(todayFigDir + '/22-10-06_5mT_SurroundingThicknessvsCompr.png')
plt.show()
#%%%%
# # %%% Tests

# # H0 vs Compression Number
# expt = '20220331_100xoil_3t3optorhoa_4.5beads_15mT_Mechanics'
# f = '22-03-31_M8_P2_C2_disc20um_L40'
# date = '22.03.31'

# file = 'C:/Users/anumi/OneDrive/Desktop/ActinCortexAnalysis/Data_Analysis/TimeSeriesData/'+f+'_PY.csv'
# tsDF = pd.read_csv(file, sep=';')


# indices = GlobalTable_meca[GlobalTable_meca['cellID'] == ufun.findInfosInFileName(f, 'cellID')].index.tolist() 

# plt.figure(figsize=(20,10))
# plt.rcParams.update({'font.size': 35})
# plt.plot(GlobalTable_meca['compNum'][indices].values, GlobalTable_meca['bestH0'][indices].values)
# plt.axvline(x = 5.5, color = 'r', marker='.')
# plt.xlabel('Compression Number')
# plt.ylabel('bestH0 (nm)')
# plt.title('bestH0 vs Compression No. | '+f)
# plt.savefig('D:/Anumita/MagneticPincherData/Figures/Historique/2022-04-20/MecaAnalysis_allCells/'+f+'_H0vT.png')
# plt.show()



