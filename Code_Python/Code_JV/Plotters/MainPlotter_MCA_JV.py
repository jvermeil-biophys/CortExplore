# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:53:25 2022

@author: JosephVermeil
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 21:27:55 2022

@author: JosephVermeil
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

import numbers

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
import TrackAnalyser_dev3_AJJV as taka2
# import TrackAnalyser_dev_AJ as taka

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


# %% Reminders

# %%% Useful option

pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.options.display.precision = 3
# pd.options.display.float_format = '{:.2f}%'.format

# %%%  Stats
# 
# (A) How to compute confidence intervals of fitted parameters with (1-alpha) confidence:
# 
#     0) from scipy import stats
#     1) df = nb_pts - nb_parms ; se = diag(cov)**0.5
#     2) Student t coefficient : q = stat.t.ppf(1 - alpha / 2, df)
#     3) ConfInt = [params - q*se, params + q*se]

# ## 




# %% TimeSeries plots


# %%% List files
allTimeSeriesDataFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                          if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv"))]
print(allTimeSeriesDataFiles)


# %%% Get a time series

df = taka.getCellTimeSeriesData('22-02-09_M1_P1_C7')


# %%% Plot a time series

# taka.plotCellTimeSeriesData('21-02-10_M1_P1_C2')
taka.plotCellTimeSeriesData('22-03-21_M3_P1_C1_sin5-3_1Hz')

# %%% Plot a time series

taka.plotCellTimeSeriesData('22-03-21_M3_P1_C1_sin5-3_2Hz')

# %%% Variation on the plotCellTimeSeriesData function

# cellID = '22-02-09_M1_P1_C2'
# fromPython = True

# X = 'T'
# Y = np.array(['B', 'F'])
# units = np.array([' (mT)', ' (pN)'])
# timeSeriesDataFrame = taka.getCellTimeSeriesData(cellID, fromPython)
# print(timeSeriesDataFrame.shape)
# if not timeSeriesDataFrame.size == 0:
# #         plt.tight_layout()
# #         fig.show() # figsize=(20,20)
#     axes = timeSeriesDataFrame.plot(x=X, y=Y, kind='line', ax=None, subplots=True, sharex=True, sharey=False, layout=None, \
#                     figsize=(4,6), use_index=True, title = cellID + ' - f(t)', grid=None, legend=False, style=None, logx=False, logy=False, \
#                     loglog=False, xticks=None, yticks=None, xlim=None, ylim=None, rot=None, fontsize=None, colormap=None, \
#                     table=False, yerr=None, xerr=None, secondary_y=False, sort_columns=False)
#     plt.gcf().tight_layout()
#     for i in range(len(Y)):
#         axes[i].set_ylabel(Y[i] + units[i])
        
#     axes[0].set_yticks([0,2,4,6,8,10])
#     axes[0].grid(axis="y")
#     axes[1].set_yticks([0,10,20,30,40,50,100,150])
#     axes[1].grid(axis="y")
#     plt.gcf().show()
# else:
#     print('cell not found')
    
#### Nice display for B and F

# cellID = '22-01-12_M2_P1_C3'
# cellID = '22-01-12_M1_P1_C6'
cellID = '22-02-09_M1_P1_C9'
fromPython = True

X = 'T'
Y = np.array(['B', 'F'])
units = np.array([' (mT)', ' (pN)'])
tSDF = taka.getCellTimeSeriesData(cellID, fromPython)

defaultColorCycle = plt.rcParams['axes.prop_cycle']
customColorCycle = cycler(color=['purple', 'red'])
plt.rcParams['axes.prop_cycle'] = customColorCycle

if not tSDF.size == 0:
#         plt.tight_layout()
#         fig.show() # figsize=(20,20)
    axes = tSDF[tSDF['T']<=40].plot(x=X, y=Y, kind='line', 
                    subplots=True, sharex=True, sharey=False,
                    figsize=(4,4), use_index=True)
    
    plt.gcf().tight_layout()
    for i in range(len(Y)):
        axes[i].set_ylabel(Y[i] + units[i])
        
    axes[0].set_yticks([0,10,20,30,40,50])
    axes[0].tick_params(axis='y', labelrotation = 0, labelsize = 10)
    axes[0].grid(axis="y")
    axes[0].legend().set_visible(False)
    axes[1].set_yticks([k for k in range(0,1400,200)])
    axes[1].tick_params(axis='y', labelrotation = 0, labelsize = 10)
    axes[1].grid(axis="y")
    axes[1].legend().set_visible(False)
    axes[1].set_xticks([0,10,20,30,40])
    axes[1].tick_params(axis='x', labelrotation = 0, labelsize = 10)
    plt.gcf().show()
else:
    print('cell not found')
plt.rcParams['axes.prop_cycle'] = defaultColorCycle


# %%% Variation on the plotCellTimeSeriesData -> Sinus

# cellID = '22-01-12_M2_P1_C3'
# cellID = '22-01-12_M1_P1_C6'
cellID = '22-03-21_M3_P1_C1_sin5-3_1Hz'
fromPython = True

X = 'T'
Y = np.array(['B', 'F'])
units = np.array([' (mT)', ' (pN)'])
tSDF = taka.getCellTimeSeriesData(cellID, fromPython)

# defaultColorCycle = plt.rcParams['axes.prop_cycle']
# customColorCycle = cycler(color=['green', 'red'])
# plt.rcParams['axes.prop_cycle'] = customColorCycle

fig, ax = plt.subplots(1,1, figsize = (10,5))

if not tSDF.size == 0:
#         plt.tight_layout()
#         fig.show() # figsize=(20,20)
    tsDFplot = tSDF[(tSDF['T']>=3.04) & (tSDF['T']<=6.05)]
    
    ax.plot(tsDFplot['T'], 1000*(tsDFplot['D3']-4.503), color = gs.colorList40[30], label = 'Thickness')
    ax.set_ylabel('Thickness (nm)', color = gs.colorList40[30])
    ax.set_ylim([0, 350])
    ax.set_xlabel('Time (s)')

    axR = ax.twinx()
    axR.plot(tsDFplot['T'], tsDFplot['F'], color = gs.colorList40[23], label = 'Force')
    axR.set_ylabel('Force (pN)', color = gs.colorList40[23])
    axR.set_ylim([0, 150])

    # axes = tSDF[(tSDF['T']>=20) & (tSDF['T']<=60)].plot(x=X, y=Y, kind='line', 
    #                 subplots=True, sharex=True, sharey=False,
    #                 figsize=(4,4), use_index=True)
    
    # plt.gcf().tight_layout()
    # for i in range(len(Y)):
    #     axes[i].set_ylabel(Y[i] + units[i])
        
    # axes[0].set_yticks([0,10,20,30,40,50])
    # axes[0].tick_params(axis='y', labelrotation = 0, labelsize = 10)
    # axes[0].grid(axis="y")
    # axes[0].legend().set_visible(False)
    # axes[1].set_yticks([k for k in range(0,1400,200)])
    # axes[1].tick_params(axis='y', labelrotation = 0, labelsize = 10)
    # axes[1].grid(axis="y")
    # axes[1].legend().set_visible(False)
    # axes[1].set_xticks([0,10,20,30,40])
    # axes[1].tick_params(axis='x', labelrotation = 0, labelsize = 10)
    plt.gcf().show()
else:
    print('cell not found')
# plt.rcParams['axes.prop_cycle'] = defaultColorCycle

# %%% Variation on the plotCellTimeSeriesData -> Broken Ramp

# cellID = '22-01-12_M2_P1_C3'
# cellID = '22-01-12_M1_P1_C6'
cellID = '22-03-21_M4_P1_C3'
fromPython = True

X = 'T'
Y = np.array(['B', 'F'])
units = np.array([' (mT)', ' (pN)'])
tSDF = taka.getCellTimeSeriesData(cellID, fromPython)

# defaultColorCycle = plt.rcParams['axes.prop_cycle']
# customColorCycle = cycler(color=['green', 'red'])
# plt.rcParams['axes.prop_cycle'] = customColorCycle

fig, ax = plt.subplots(1,1, figsize = (6,3))

if not tSDF.size == 0:
#         plt.tight_layout()
#         fig.show() # figsize=(20,20)
    tsDFplot = tSDF[(tSDF['T']>=31) & (tSDF['T']<=34)]
    
    ax.plot(tsDFplot['T'], 1000*(tsDFplot['D3']-4.503), color = gs.colorList40[30], label = 'Thickness')
    ax.set_ylabel('Thickness (nm)', color = gs.colorList40[30])
    ax.set_ylim([0, 1200])
    ax.set_xlabel('Time (s)')

    axR = ax.twinx()
    axR.plot(tsDFplot['T'], tsDFplot['F'], color = gs.colorList40[23], label = 'Force')
    axR.set_ylabel('Force (pN)', color = gs.colorList40[23])
    axR.set_ylim([0, 1500])

    # axes = tSDF[(tSDF['T']>=20) & (tSDF['T']<=60)].plot(x=X, y=Y, kind='line', 
    #                 subplots=True, sharex=True, sharey=False,
    #                 figsize=(4,4), use_index=True)
    
    # plt.gcf().tight_layout()
    # for i in range(len(Y)):
    #     axes[i].set_ylabel(Y[i] + units[i])
        
    # axes[0].set_yticks([0,10,20,30,40,50])
    # axes[0].tick_params(axis='y', labelrotation = 0, labelsize = 10)
    # axes[0].grid(axis="y")
    # axes[0].legend().set_visible(False)
    # axes[1].set_yticks([k for k in range(0,1400,200)])
    # axes[1].tick_params(axis='y', labelrotation = 0, labelsize = 10)
    # axes[1].grid(axis="y")
    # axes[1].legend().set_visible(False)
    # axes[1].set_xticks([0,10,20,30,40])
    # axes[1].tick_params(axis='x', labelrotation = 0, labelsize = 10)
    plt.gcf().show()
else:
    print('cell not found')
# plt.rcParams['axes.prop_cycle'] = defaultColorCycle

# %%% Plot multiple time series

allTimeSeriesDataFiles = [f for f in os.listdir(cp.DirDataTimeseries) if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv"))]
for f in allTimeSeriesDataFiles:
    if '22-02-09_M3' in f:
        taka.plotCellTimeSeriesData(f[:-4])

allTimeSeriesDataFiles = [f for f in os.listdir(cp.DirDataTimeseries) if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv"))]
for f in allTimeSeriesDataFiles:
    if '22-02-09_M2' in f:
        taka.plotCellTimeSeriesData(f[:-4])


# %%% Close all

plt.close('all')


# %%% Functions acting on the trajectories

listeTraj = taka.getCellTrajData('21-12-16_M1_P1_C10', Ntraj = 2)
listeTraj[1]['pos']

def testTraj(D0):
    trajDir = os.path.join(cp.DirDataTimeseries, 'Trajectories')
    allTimeSeriesDataFiles = [f for f in os.listdir(cp.DirDataTimeseries) 
                              if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) 
                                  and f.endswith(".csv"))]
    cellIDList = []
    for f in allTimeSeriesDataFiles:
        cellID = ufun.findInfosInFileName(f, 'cellID')
        if '21-12-08' in cellID:
            cellIDList.append(cellID)

    print(cellIDList)

    fig, ax = plt.subplots(1,2, figsize = (8,4))
    width = 15
    ax[0].set_title('Inside bead')
    ax[0].axis([-width,width,-width,width])
    ax[1].set_title('Outside bead')
    ax[1].axis([-width,width,-width,width])

    for C in (cellIDList):
        listeTraj = taka.getCellTrajData(C)
        iOut = (listeTraj[1]['pos'] == 'out')
        iIn = 1 - iOut
        dfIn, dfOut = listeTraj[iIn]['df'], listeTraj[iOut]['df']
        Xin = dfIn.X.values - dfIn.X.values[0]
        Yin = dfIn.Y.values - dfIn.Y.values[0]
        Xout = dfOut.X.values - dfOut.X.values[0]
        Yout = dfOut.Y.values - dfOut.Y.values[0]
        npts = len(Xin)
        for iii in range(1, npts):
            D2in = ((Xin[iii]-Xin[iii-1])**2 + (Yin[iii]-Yin[iii-1])**2) ** 0.5
            D2out = ((Xout[iii]-Xout[iii-1])**2 + (Yout[iii]-Yout[iii-1])**2) ** 0.5
            if D2in > D0 and D2out > D0:
                dxCorrIn = Xin[iii-1]-Xin[iii]
                dyCorrIn = Yin[iii-1]-Yin[iii]
                dxCorrOut = Xout[iii-1]-Xout[iii]
                dyCorrOut = Yout[iii-1]-Yout[iii]
                Xin[iii:] += dxCorrIn
                Yin[iii:] += dyCorrIn
                Xout[iii:] += dxCorrOut
                Yout[iii:] += dyCorrOut
        if np.max(np.abs(Xin)) < width and np.max(np.abs(Yin)) < width and np.max(np.abs(Xout)) < width and np.max(np.abs(Yout)) < width:
            ax[0].plot(Xin, Yin)
            ax[1].plot(Xout, Yout)

    plt.show()
    
testTraj(1)
    
testTraj(1.5)







# #############################################################################
# %% > Data import & export

#### Data import


#### Global_MecaData_MCA123

# GlobalTable_meca_MCA = taka.getGlobalTable(kind = 'meca_MCA')
MecaData_MCA = taka.getMergedTable('MecaData_MCA', mergeUMS = True, mergeFluo = True)

#### 'Round' based selection

MecaData_MCA['round'] = MecaData_MCA['tags'].apply(lambda x : x.split('. ')[0])

#### Get a 'categorical fluo column'

def categoriesFluoColumn(df):
    filter01 = df['UI_Fluo'].isnull()
    df.loc[filter01,'UI_Fluo'] = 'none'
    
    filter02 = ((df['drug'] == 'doxycyclin') & (df['meanFluoPeakAmplitude'].apply(lambda x : not pd.isnull(x))))
    filter02_low  = (filter02 & (df['meanFluoPeakAmplitude'] < 200))
    filter02_mid  = (filter02 & (df['meanFluoPeakAmplitude'] > 200) & (df['meanFluoPeakAmplitude'] < 500))
    filter02_high = (filter02 & (df['meanFluoPeakAmplitude'] > 500))
    
    df.loc[filter02_low , 'UI_Fluo'] = 'low'
    df.loc[filter02_mid , 'UI_Fluo'] = 'mid'
    df.loc[filter02_high, 'UI_Fluo'] = 'high'
    
    return(df)

MecaData_MCA = categoriesFluoColumn(MecaData_MCA)


#### Fluo based selection

def fluoSelectionColumn(df):
    df['FluoSelection'] = np.zeros(df.shape[0], dtype=bool)
    filter01 = (df['drug'] == 'none')
    df.loc[filter01,'FluoSelection'] = True
    filter02 = (df['drug'] == 'doxycyclin') & (df['UI_Fluo'].apply(lambda x : x in ['mid', 'high']))
    df.loc[filter02,'FluoSelection'] = True
    return(df)

MecaData_MCA = fluoSelectionColumn(MecaData_MCA)


#### Option to pool F8 and E4

# def linkerTypeColumn(df):
#     df['linker type'] = np.zeros(df.shape[0], dtype='<U7')
#     filter01 = (df['cell subtype'] == 'aSFL-A11')
#     df.loc[filter01,'linker type'] = 'iMC'
#     filter02 = (df['cell subtype'].apply(lambda x : x in ['aSFL-F8', 'aSFL-E4']))
#     df.loc[filter02,'linker type'] = 'iMC-6FP'
#     return(df)

# GlobalTable_meca_MCA123 = linkerTypeColumn(GlobalTable_meca_MCA123)





# %%% Custom data export

# %%%% 21-06-28 Export of E - h data for Julien

# GlobalTable_meca_Py_expJ = GlobalTable_meca_Py.loc[GlobalTable_meca_Py['validatedFit'] & GlobalTable_meca_Py['validatedThickness']]\
#                                                [['cell type', 'cell subtype', 'bead type', 'drug', 'substrate', 'compNum', \
#                                                  'EChadwick', 'H0Chadwick', 'surroundingThickness', 'ctFieldThickness']]
# GlobalTable_meca_Py_expJ = GlobalTable_meca_Py_expJ.reset_index()
# GlobalTable_meca_Py_expJ = GlobalTable_meca_Py_expJ.drop('index', axis=1)
# savePath = os.path.join(cp.DirDataAnalysis, 'mecanicsData_3T3.csv')
# GlobalTable_meca_Py_expJ.to_csv(savePath, sep=';')


# %% > Plotting Functions

# %%% Objects declaration

# %%%% Generic

Styles = {''} # Project of automatic formatting according to the type of data

renameDict1 = {'SurroundingThickness':'Thickness at low force (nm)',
                'surroundingThickness':'Thickness at low force (nm)',
                'ctFieldThickness':'Thickness at low force (nm)',
                'bestH0':'Thickness from fit (nm)',
                'EChadwick': 'E Chadwick (Pa)',
                'medianThickness': 'Median Thickness (nm)',               
                'fluctuAmpli': 'Fluctuations Amplitude (nm)',               
                'meanFluoPeakAmplitude' : 'Fluo Intensity (a.u.)',               
                'none':'control',               
                'doxycyclin':'expressing iMC linker',               
                'none & BSA coated glass':'control & non adherent',               
                'doxycyclin & BSA coated glass':'iMC & non adherent',               
                'none & 20um fibronectin discs':'control & adherent on fibro',               
                'doxycyclin & 20um fibronectin discs':'iMC & adherent on fibro',               
                'BSA coated glass & none':'control & non adherent',               
                'BSA coated glass & doxycyclin':'iMC & non adherent',               
                '20um fibronectin discs & none':'control & adherent on fibro',               
                '20um fibronectin discs & doxycyclin':'iMC & adherent on fibro',               
                'aSFL & none':'aSFL control',
                'aSFL & doxycyclin':'aSFL iMC',
                'aSFL-6FP & none':'aSFL-6FP control',               
                'aSFL-6FP & doxycyclin':'aSFL-6FP long-iMC',               
                'aSFL-6FP-2 & none':'aSFL-6FP-2 control',               
                'aSFL-6FP-2 & doxycyclin':'aSFL-6FP-2 long-iMC',               
                'aSFL-A8 & none':'aSFL-A8 control',               
                'aSFL-A8 & doxycyclin':'aSFL-A8 iMC',               
                'aSFL-A8-2 & none':'aSFL-A8-2 control',               
                'aSFL-A8-2 & doxycyclin':'aSFL-A8-2 iMC',               
                'dmso' : 'DMSO, no linker', 
                'smifh2' : 'SMIFH2, no linker', 
                'dmso, doxycyclin' : 'DMSO, iMC linker', 
                'smifh2, doxycyclin' : 'SMIFH2, iMC linker'}

styleDict1 =  {'none & BSA coated glass':{'color':'#ff9896','marker':'^'},               
                'doxycyclin & BSA coated glass':{'color':'#d62728','marker':'^'},               
                'none & 20um fibronectin discs':{'color':'#aec7e8','marker':'o'},               
                'doxycyclin & 20um fibronectin discs':{'color':'#1f77b4','marker':'o'},               
                'none':{'color':'#aec7e8','marker':'o'},               
                'doxycyclin':{'color':'#1f77b4','marker':'o'},               
                'BSA coated glass & none':{'color':'#ff9896','marker':'^'},               
                'BSA coated glass & doxycyclin':{'color':'#d62728','marker':'^'},               
                '20um fibronectin discs & none':{'color':'#aec7e8','marker':'o'},               
                '20um fibronectin discs & doxycyclin':{'color':'#1f77b4','marker':'o'},               
                'aSFL':{'color':'gs.colorList40[10]','marker':'o'},               
                'aSFL-6FP':{'color':'#2ca02c','marker':'o'},               
                'aSFL-A8':{'color':'#ff7f0e','marker':'o'},                
                'aSFL & none':{'color':'gs.colorList40[10]','marker':'o'},               
                'aSFL & doxycyclin':{'color':'gs.colorList40[30]','marker':'o'},               
                'aSFL-6FP & none':{'color':'#98df8a','marker':'o'},               
                'aSFL-6FP & doxycyclin':{'color':'#2ca02c','marker':'o'},               
                'aSFL-A8 & none':{'color':'#ffbb78','marker':'o'},              
                'aSFL-A8 & doxycyclin':{'color':'#ff7f0e','marker':'o'},
               
                'aSFL-A11 & none':{'color':gs.colorList40[20],'marker':'o'},              
                'aSFL-A11 & doxycyclin':{'color':gs.colorList40[30],'marker':'o'},
                'aSFL-F8 & none':{'color':gs.colorList40[21],'marker':'o'},              
                'aSFL-F8 & doxycyclin':{'color':gs.colorList40[31],'marker':'o'},
                'aSFL-E4 & none':{'color':gs.colorList40[22],'marker':'o'},              
                'aSFL-E4 & doxycyclin':{'color':gs.colorList40[32],'marker':'o'},
               
                'DictyDB_M270':{'color':'lightskyblue','marker':'o'}, 
                'DictyDB_M450':{'color': 'maroon','marker':'o'},
                'M270':{'color':'lightskyblue','marker':'o'}, 
                'M450':{'color': 'maroon','marker':'o'},
                # HoxB8
                'bare glass & ctrl':{'color': gs.colorList40[10],'marker':'^'},
                'bare glass & tko':{'color': gs.colorList40[30],'marker':'^'},
                '20um fibronectin discs & ctrl':{'color': gs.colorList40[12],'marker':'o'},
                '20um fibronectin discs & tko':{'color': gs.colorList40[32],'marker':'o'}
                }


# %%%% MCA specific

renameDict_MCA3 = {'SurroundingThickness':'Thickness at low force (nm)',
               'surroundingThickness':'Thickness at low force (nm)',
               'ctFieldThickness':'Thickness at low force (nm)',
               'ctFieldThickness_normalized':'Thickness at low force (nm) [Ratio]',
               'bestH0':'Thickness from fit (nm)',
               'bestH0_normalized':'Thickness from fit (nm) [Ratio]',
               'EChadwick': 'E Chadwick (Pa)',
               'medianThickness': 'Median Thickness (nm)',               
               'fluctuAmpli': 'Fluctuations Amplitude (nm)',               
               'meanFluoPeakAmplitude' : 'Fluo Intensity (a.u.)',               
               'none':'control',               
               'doxycyclin':'expressing iMC linker',
               
                'aSFL-A11 & none':'clone A11 - ctrl',              
                'aSFL-A11 & doxycyclin':'clone A11 - iMC',
                'aSFL-F8 & none':'clone F8 - ctrl',              
                'aSFL-F8 & doxycyclin':'clone F8 - long iMC',
                'aSFL-E4 & none':'clone E4 - ctrl',              
                'aSFL-E4 & doxycyclin':'clone E4 - long iMC',
                'aSFL-A11 & MCA1 & none':'Round1: clone A11 - ctrl',              
                'aSFL-A11 & MCA1 & doxycyclin':'Round1: clone A11 - iMC',
                'aSFL-F8 & MCA2 & none':'Round2: clone F8 - ctrl',                 
                'aSFL-F8 & MCA2 & doxycyclin':'Round2: clone F8 - long iMC',
                'aSFL-E4 & MCA1 & none':'Round1: clone E4 - ctrl',                  
                'aSFL-E4 & MCA1 & doxycyclin':'Round1: clone E4 - long iMC',
                'aSFL-A11 & MCA3 & none':'Round3: clone A11 - ctrl',                  
                'aSFL-A11 & MCA3 & doxycyclin':'Round3: clone A11 - iMC',
                'aSFL-F8 & MCA3 & none':'Round3: clone F8 - ctrl',                  
                'aSFL-F8 & MCA3 & doxycyclin':'Round3: clone F8 - long iMC',
                'aSFL-E4 & MCA3 & none':'Round3: clone E4 - ctrl',               
                'aSFL-E4 & MCA3 & doxycyclin':'Round3: clone E4 - long iMC',
                
                'aSFL-A11 & 21-01-18':'A11\n21-01-18',
                 'aSFL-A11 & 21-01-21':'A11\n21-01-21',
                 'aSFL-A11 & 22-07-15':'A11\n22-07-15',
                 'aSFL-A11 & 22-07-20':'A11\n22-07-20',
                 'aSFL-F8 & 21-09-08':'F8\n21-09-08',
                 'aSFL-F8 & 22-07-15':'F8\n22-07-15',
                 'aSFL-F8 & 22-07-27':'F8\n22-07-27',
                 'aSFL-E4 & 21-04-27':'E4\n21-04-27',
                 'aSFL-E4 & 21-04-28':'E4\n21-04-28',
                 'aSFL-E4 & 22-07-20':'E4\n22-07-20',
                 'aSFL-E4 & 22-07-27':'E4\n22-07-27',
                 
                 'iMC & none':'(-) iMC',              
                 'iMC & doxycyclin':'(+) iMC',
                 'iMC-6FP & none':'(-) iMC-6FP',              
                 'iMC-6FP & doxycyclin':'(+) iMC-6FP',
               }


styleDict_MCA3 = { '3T3':{'color':gs.colorList40[20],
                               'marker':'o'},
                  'aSFL-A11':{'color':gs.colorList40[10],
                              'marker':'o'},
                  'aSFL-F8':{'color':gs.colorList40[11],
                             'marker':'o'},   
                  'aSFL-E4':{'color':gs.colorList40[12],
                             'marker':'o'}, 
                  'aSFL-A11 & none':{'color':gs.colorList40[10],
                                     'marker':'o'},              
                  'aSFL-A11 & doxycyclin':{'color':gs.colorList40[30],
                                           'marker':'o'},
                  'aSFL-F8 & none':{'color':gs.colorList40[11],
                                    'marker':'o'},              
                  'aSFL-F8 & doxycyclin':{'color':gs.colorList40[31],
                                          'marker':'o'},
                  'aSFL-E4 & none':{'color':gs.colorList40[12],
                                    'marker':'o'},              
                  'aSFL-E4 & doxycyclin':{'color':gs.colorList40[32],
                                          'marker':'o'},
                  'aSFL-A11 & MCA1 & none':{'color':gs.colorList40[10],
                                            'marker':'o'},              
                  'aSFL-A11 & MCA1 & doxycyclin':{'color':gs.colorList40[30],
                                                  'marker':'o'},
                  'aSFL-F8 & MCA2 & none':{'color':gs.colorList40[11],
                                           'marker':'o'},              
                  'aSFL-F8 & MCA2 & doxycyclin':{'color':gs.colorList40[31],
                                                 'marker':'o'},
                  'aSFL-E4 & MCA1 & none':{'color':gs.colorList40[12],
                                           'marker':'o'},              
                  'aSFL-E4 & MCA1 & doxycyclin':{'color':gs.colorList40[32],
                                                 'marker':'o'},
                  'aSFL-A11 & MCA3 & none':{'color':gs.colorList40[10],
                                            'marker':'o'},              
                  'aSFL-A11 & MCA3 & doxycyclin':{'color':gs.colorList40[30],
                                                  'marker':'o'},
                  'aSFL-F8 & MCA3 & none':{'color':gs.colorList40[11],
                                           'marker':'o'},              
                  'aSFL-F8 & MCA3 & doxycyclin':{'color':gs.colorList40[31],
                                                 'marker':'o'},
                  'aSFL-E4 & MCA3 & none':{'color':gs.colorList40[12],
                                           'marker':'o'},              
                  'aSFL-E4 & MCA3 & doxycyclin':{'color':gs.colorList40[32],
                                                 'marker':'o'},
                  
                  'aSFL-A11 & 21-01-18':{'color':gs.colorList40[00],
                                                 'marker':'o'},
                   'aSFL-A11 & 21-01-21':{'color':gs.colorList40[10],
                                                  'marker':'o'},
                   'aSFL-A11 & 22-07-15':{'color':gs.colorList40[20],
                                                  'marker':'o'},
                   'aSFL-A11 & 22-07-20':{'color':gs.colorList40[30],
                                                  'marker':'o'},
                   'aSFL-F8 & 21-09-08':{'color':gs.colorList40[11],
                                                  'marker':'o'},
                   'aSFL-F8 & 22-07-15':{'color':gs.colorList40[21],
                                                  'marker':'o'},
                   'aSFL-F8 & 22-07-27':{'color':gs.colorList40[31],
                                                  'marker':'o'},
                   'aSFL-E4 & 21-04-27':{'color':gs.colorList40[12],
                                                  'marker':'o'},
                   'aSFL-E4 & 22-07-20':{'color':gs.colorList40[22],
                                                  'marker':'o'},
                   'aSFL-E4 & 22-07-27':{'color':gs.colorList40[32],
                                                  'marker':'o'},
                   'none':{'color':gs.colorList40[9],
                                                  'marker':'o'},
                   'doxycyclin':{'color':gs.colorList40[29],
                                                  'marker':'o'},
                   
                   'iMC & none':{'color':gs.colorList40[10],
                                      'marker':'o'},            
                   'iMC & doxycyclin':{'color':gs.colorList40[30],
                                      'marker':'o'},
                   'iMC-6FP & none':{'color':gs.colorList40[13],
                                      'marker':'o'},
                   'iMC-6FP & doxycyclin':{'color':gs.colorList40[33],
                                      'marker':'o'},
                  }

splitterStyleDict_MCA = {'high':'^',
                          'mid':'D',   
                          'low':'v', 
                          'none':'X'}


# These functions use matplotlib.pyplot and seaborn libraries to display 1D categorical or 2D plots

# %%% Main functions


def D1Plot(data, fig = None, ax = None, CondCols=[], Parameters=[], Filters=[], 
           Boxplot=True, AvgPerCell=False, cellID='cellID', co_order=[],
           stats=True, statMethod='Mann-Whitney', box_pairs=[], 
           figSizeFactor = 1, markersizeFactor = 1, orientation = 'h', # useHue = False, 
           stressBoxPlot = False, bypassLog = False, 
           returnData = 0, returnCount = 0):
    
    data_filtered = data
    # for fltr in Filters:
    #     data_filtered = data_filtered.loc[fltr]

    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for kk in range(len(Filters)):
        globalFilter = globalFilter & Filters[kk]
    data_filtered = data_filtered[globalFilter]
    
    NCond = len(CondCols)
    
    if NCond == 1:
        CondCol = CondCols[0]
        
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += CondCols[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_filtered[newColName] = ''
        for i in range(NCond):
            data_filtered[newColName] += data_filtered[CondCols[i]].astype(str)
            data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x + ' & ')
        data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x[:-3])
        CondCol = newColName
        
    # define the count df
    cols_count_df = ['compNum', 'cellID', 'manipID', 'date', CondCol]
    count_df = data_filtered[cols_count_df]

    # average per cell if necessary
    if AvgPerCell:
        group = data_filtered.groupby(cellID)
        dictAggMean = getDictAggMean(data_filtered)
#         dictAggMean['EChadwick'] = 'median'
        data_filtered = group.agg(dictAggMean)
        
    data_filtered.sort_values(CondCol, axis=0, ascending=True, inplace=True)
    
    # define the export df
    cols_export_df = ['date', 'manipID', 'cellID']
    if not AvgPerCell: 
        cols_export_df.append('compNum')
    cols_export_df += ([CondCol] + Parameters)
    export_df = data_filtered[cols_export_df]
    
    NPlots = len(Parameters)
    Conditions = list(data_filtered[CondCol].unique())     
    
    if len(co_order) > 0:
        if len(co_order) != len(Conditions):
            delCo = [co for co in co_order if co not in Conditions]
            for co in delCo:
                co_order.remove(co)
                
        p = getStyleLists_Sns(co_order, styleDict_MCA3) # styleDict_MCA3
        
    else: # len(co_order) == 0
        p = sns.color_palette()
        co_order = Conditions

    if fig == None:
        if orientation == 'h':
            fig, ax = plt.subplots(1, NPlots, figsize = (5*NPlots*NCond*figSizeFactor, 5))
        elif orientation == 'v':
            fig, ax = plt.subplots(NPlots, 1, figsize = (5*NCond*figSizeFactor, 5*NPlots))
    else:
        pass
        
    markersize = 5*markersizeFactor
    
    if NPlots == 1:
        ax = np.array([ax])
    
    for k in range(NPlots):

        if (not bypassLog) and (('EChadwick' in Parameters[k]) or ('K_' in Parameters[k])):
            ax[k].set_yscale('log')

        if Boxplot:
            if stressBoxPlot:
                if stressBoxPlot == 2:
                    sns.boxplot(x=CondCol, y=Parameters[k], data=data_filtered, ax=ax[k], 
                                width = 0.5, showfliers = False, order= co_order, 
                                medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 4},
                                boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
            #                   boxprops={"color": color, "linewidth": 0.5},
                                whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                                capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4})
                                # scaley = scaley)
                elif stressBoxPlot == 1:
                    sns.boxplot(x=CondCol, y=Parameters[k], data=data_filtered, ax=ax[k], 
                                width = 0.5, showfliers = False, order= co_order, 
                                medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
                                boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
            #                   boxprops={"color": color, "linewidth": 0.5},
                                whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                                capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2})
                                # scaley = scaley)
            else:
                sns.boxplot(x=CondCol, y=Parameters[k], data=data_filtered, ax=ax[k], 
                            width = 0.5, showfliers = False, order= co_order, 
                            medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},
                            boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
        #                   boxprops={"color": color, "linewidth": 0.5},
                            whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                            capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2})
                            # scaley = scaley)
            
            # data_filtered.boxplot(column=Parameters[k], by = CondCol, ax=ax[k],showfliers = False) # linewidth = 2, width = 0.5

        if stats:
            if len(box_pairs) == 0:
                box_pairs = makeBoxPairs(co_order)
            addStat_df(ax[k], data_filtered, box_pairs, Parameters[k], CondCol, test = statMethod)
            # add_stat_annotation(ax[k], x=CondCol, y=Parameters[k], data=data_filtered,box_pairs = box_pairs,test=statMethod, text_format='star',loc='inside', verbose=2)
        

        sns.swarmplot(x=CondCol, y=Parameters[k], data=data_filtered, ax=ax[k], order = co_order,
                      size=markersize, edgecolor='k', linewidth = 1*markersizeFactor, palette = p)
        # else:
        #     sns.swarmplot(x=CondCol, y=Parameters[k], data=data_filtered, ax=ax[k], order = co_order,
        #                   size=markersize, edgecolor='k', linewidth = 1*markersizeFactor, palette = p, 
        #                   hue = 'manipID')
        #     legend = ax[k].legend()
        #     legend.remove()

        ax[k].set_xlabel('')
        ax[k].set_ylabel(Parameters[k])
        ax[k].tick_params(axis='x', labelrotation = 10)
        ax[k].yaxis.grid(True)
        if ax[k].get_yscale() == 'linear':
            ax[k].set_ylim([0, ax[k].get_ylim()[1]])
        
    plt.rcParams['axes.prop_cycle'] = gs.my_default_color_cycle
        
    output = (fig, ax)
    
    if returnData > 0:
        output += (export_df, )
    
    if returnCount > 0:
        groupByCell = count_df.groupby(cellID)
        d_agg = {'compNum':'count', CondCol:'first', 'date':'first', 'manipID':'first'}
        df_CountByCell = groupByCell.agg(d_agg).rename(columns={'compNum':'compCount'})

        groupByCond = df_CountByCell.reset_index().groupby(CondCol)
        d_agg = {cellID: 'count', 'compCount': 'sum', 
                 'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
        d_rename = {cellID:'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
        df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
        
        if returnCount == 1:
            output += (df_CountByCond, )
        elif returnCount == 2:
            output += (df_CountByCond, df_CountByCell)

    return(output)
    
    
    
def D1Plot_wInnerSplit(data, fig = None, ax = None, CondCols=[], InnerSplitCols = [], 
                       Parameters=[], Filters=[], 
                       Boxplot=True, AvgPerCell=False, cellID='cellID', co_order=[],
                       stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                       figSizeFactor = 1, markersizeFactor=1, orientation = 'h', useHue = False, 
                       stressBoxPlot = False, bypassLog = False, 
                       returnData = 0, returnCount = 0):
    
    data_filtered = data
    # for fltr in Filters:
    #     data_filtered = data_filtered.loc[fltr]

    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for kk in range(len(Filters)):
        globalFilter = globalFilter & Filters[kk]
    data_filtered = data_filtered[globalFilter]
    
    NCond = len(CondCols)
    if NCond == 1:
        CondCol = CondCols[0]
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += CondCols[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_filtered[newColName] = ''
        for i in range(NCond):
            data_filtered[newColName] += data_filtered[CondCols[i]].astype(str)
            data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x + ' & ')
        data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x[:-3])
        CondCol = newColName
        
    NSplits = len(InnerSplitCols)
    if NSplits == 1:
        InnerSplitCol = InnerSplitCols[0]
    elif NSplits > 1:
        newColName = ''
        for i in range(NSplits):
            newColName += InnerSplitCols[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_filtered[newColName] = ''
        for i in range(NSplits):
            data_filtered[newColName] += data_filtered[InnerSplitCol[i]].astype(str)
            data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x + ' & ')
        data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x[:-3])
        InnerSplitCol = newColName
    
    # define the count df
    count_df = data_filtered[['cellID', 'compNum', 'date', 'manipID', CondCol, InnerSplitCol]]
    
    if AvgPerCell:
        group = data_filtered.groupby(cellID)
        dictAggMean = getDictAggMean(data_filtered)
#         dictAggMean['EChadwick'] = 'median'
        data_filtered = group.agg(dictAggMean)
        
    data_filtered.sort_values(CondCol, axis=0, ascending=True, inplace=True)
    
    # define the export df
    cols_export_df = ['date', 'manipID', 'cellID']
    if not AvgPerCell: 
        cols_export_df.append('compNum')
    cols_export_df += ([CondCol, InnerSplitCol] + Parameters)
    export_df = data_filtered[cols_export_df]
    
    
    NPlots = len(Parameters)
    Conditions = list(data_filtered[CondCol].unique())  
    Splitters = list(data_filtered[InnerSplitCol].unique())  
    print(Splitters)
    
    if len(co_order) > 0:
        if len(co_order) != len(Conditions):
            delCo = [co for co in co_order if co not in Conditions]
            for co in delCo:
                co_order.remove(co)
                
        p = getStyleLists_Sns(co_order, styleDict_MCA3)
        
    else: # len(co_order) == 0
        p = sns.color_palette()
        co_order = Conditions

    if fig == None:
        if orientation == 'h':
            fig, ax = plt.subplots(1, NPlots, figsize = (5*NPlots*NCond*figSizeFactor, 5))
        elif orientation == 'v':
            fig, ax = plt.subplots(NPlots, 1, figsize = (5*NCond*figSizeFactor, 5*NPlots))
    else:
        pass
        
    markersize = 5*markersizeFactor
    
    if NPlots == 1:
        ax = np.array([ax])
    
    for k in range(NPlots):

        if not bypassLog and (('EChadwick' in Parameters[k]) or ('KChadwick' in Parameters[k])):
            ax[k].set_yscale('log')

        if Boxplot:
            if stressBoxPlot:
                if stressBoxPlot == 2:
                    sns.boxplot(x=CondCol, y=Parameters[k], data=data_filtered, ax=ax[k], 
                                width = 0.5, showfliers = False, order= co_order, 
                                medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 4},
                                boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
            #                   boxprops={"color": color, "linewidth": 0.5},
                                whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                                capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4})
                                # scaley = scaley)
                elif stressBoxPlot == 1:
                    sns.boxplot(x=CondCol, y=Parameters[k], data=data_filtered, ax=ax[k], 
                                width = 0.5, showfliers = False, order= co_order, 
                                medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
                                boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
            #                   boxprops={"color": color, "linewidth": 0.5},
                                whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                                capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2})
                                # scaley = scaley)
            else:
                sns.boxplot(x=CondCol, y=Parameters[k], data=data_filtered, ax=ax[k], 
                            width = 0.5, showfliers = False, order= co_order, 
                            medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},
                            boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
        #                   boxprops={"color": color, "linewidth": 0.5},
                            whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                            capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2})
                            # scaley = scaley)
        
        for split in Splitters:
            marker = splitterStyleDict_MCA[split]
            data_filtered_split = data_filtered[data_filtered[InnerSplitCol] == split]
            sns.stripplot(x=CondCol, y=Parameters[k], data=data_filtered_split, ax=ax[k], order = co_order,
                          jitter = True, marker=marker, 
                          size=markersize, edgecolor='k', linewidth = 1*markersizeFactor, 
                          palette = p)
            
            # data_filtered.boxplot(column=Parameters[k], by = CondCol, ax=ax[k],showfliers = False) # linewidth = 2, width = 0.5

        if stats:
            if len(box_pairs) == 0:
                box_pairs = makeBoxPairs(co_order)
            addStat_df(ax[k], data_filtered, box_pairs, Parameters[k], CondCol, test = statMethod)
            # add_stat_annotation(ax[k], x=CondCol, y=Parameters[k], data=data_filtered,box_pairs = box_pairs,test=statMethod, text_format='star',loc='inside', verbose=2)
        
        # if not useHue:
        #     sns.swarmplot(x=CondCol, y=Parameters[k], data=data_filtered, ax=ax[k], order = co_order,
        #                   size=markersize, edgecolor='k', linewidth = 1*markersizeFactor, palette = p)
        # else:
        #     sns.swarmplot(x=CondCol, y=Parameters[k], data=data_filtered, ax=ax[k], order = co_order,
        #                   size=markersize, edgecolor='k', linewidth = 1*markersizeFactor, palette = p, 
        #                   hue = 'manipID')
        #     legend = ax[k].legend()
        #     legend.remove()

        ax[k].set_xlabel('')
        ax[k].set_ylabel(Parameters[k])
        ax[k].tick_params(axis='x', labelrotation = 10)
        ax[k].yaxis.grid(True)
        if ax[k].get_yscale() == 'linear':
            ax[k].set_ylim([0, ax[k].get_ylim()[1]])
        
    plt.rcParams['axes.prop_cycle'] = gs.my_default_color_cycle
        
    
    output = (fig, ax)
    
    if returnData > 0:
        output += (export_df, )
    
    if returnCount > 0:
        groupByCell = count_df.groupby(cellID)
        d_agg = {'compNum':'count', CondCol:'first', 'date':'first', 'manipID':'first'}
        df_CountByCell = groupByCell.agg(d_agg).rename(columns={'compNum':'compCount'})

        groupByCond = df_CountByCell.reset_index().groupby(CondCol)
        d_agg = {cellID: 'count', 'compCount': 'sum', 
                 'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
        d_rename = {cellID:'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
        df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
        
        if returnCount == 1:
            output += (df_CountByCond, )
        elif returnCount == 2:
            output += (df_CountByCond, df_CountByCell)

    return(output)
    
    
    
    
    
def D1Plot_wNormalize(data, fig = None, ax = None, CondCols=[], Parameters=[], Filters=[], 
                      Boxplot=True, AvgPerCell=False, cellID='cellID', co_order=[],
                      stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                      normalizeCol = [], normalizeGroups=[],
                      figSizeFactor = 1, markersizeFactor=1, orientation = 'h', useHue = False, 
                      stressBoxPlot = False, bypassLog = False, 
                      returnData = 0, returnCount = 0):
    
    data_filtered = data
  
    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for kk in range(len(Filters)):
        globalFilter = globalFilter & Filters[kk]
    data_filtered = data_filtered[globalFilter]

    
    NCond = len(CondCols)
    
    if NCond == 1:
        CondCol = CondCols[0]
        
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += CondCols[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_filtered[newColName] = ''
        for i in range(NCond):
            data_filtered[newColName] += data_filtered[CondCols[i]].astype(str)
            data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x + ' & ')
        data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x[:-3])
        CondCol = newColName
        
    # define the count df
    count_df = data_filtered[[CondCol, 'cellID', 'compNum', 'date', 'manipID']]
    
    if AvgPerCell:
        group = data_filtered.groupby(cellID)
        dictAggMean = getDictAggMean(data_filtered)
#         dictAggMean['EChadwick'] = 'median'
        data_filtered = group.agg(dictAggMean)
        
    data_filtered.sort_values(CondCol, axis=0, ascending=True, inplace=True)
    
    
    #### Normalisation column
    
    if normalizeCol == []:
        normalizeCol = CondCol
        
    else:
        N_normalizeCols = len(normalizeCol)
        newColName = ''
        for i in range(N_normalizeCols):
            newColName += normalizeCol[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_filtered[newColName] = ''
        for i in range(N_normalizeCols):
            data_filtered[newColName] += data_filtered[normalizeCol[i]].astype(str)
            data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x + ' & ')
        data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x[:-3])
        normalizeCol = newColName
    
    if normalizeGroups == []:
        normalizeGroups = co_order
    
    
    
    
    NPlots = len(Parameters)
    Conditions = list(data_filtered[CondCol].unique())     
    
    if len(co_order) > 0:
        if len(co_order) != len(Conditions):
            delCo = [co for co in co_order if co not in Conditions]
            for co in delCo:
                co_order.remove(co)
                
        p = getStyleLists_Sns(co_order, styleDict_MCA3)
        
    else: # len(co_order) == 0
        p = sns.color_palette()
        co_order = Conditions

    if fig == None:
        if orientation == 'h':
            fig, ax = plt.subplots(1, NPlots, figsize = (5*NPlots*NCond*figSizeFactor, 5))
        elif orientation == 'v':
            fig, ax = plt.subplots(NPlots, 1, figsize = (5*NCond*figSizeFactor, 5*NPlots))
    else:
        pass
        
    markersize = 5*markersizeFactor
    
    if NPlots == 1:
        ax = np.array([ax])
    
    for k in range(NPlots):
        Parm = Parameters[k]
        
        #### Normalisation
        
        data_filtered[Parm + '_normalized'] = data_filtered[Parm]
        for nG in normalizeGroups:
            ref = nG[0]
            ref_median = np.median(data_filtered[data_filtered[normalizeCol] == ref][Parm])
            
            Rows = data_filtered[normalizeCol].apply(lambda x : x in nG)
            Cols = Parm + '_normalized'
            
            newValues = data_filtered.loc[Rows, Cols].apply(lambda x : x/ref_median)
            data_filtered.loc[Rows, Cols] = newValues
        
        #### Take the normalized parameter
        
        Parm = Parm + '_normalized'

        if not bypassLog and (('EChadwick' in Parm) or ('K_Chadwick' in Parm) or ('K_S=' in Parm)):
            ax[k].set_yscale('log')

        if Boxplot:
            if stressBoxPlot:
                if stressBoxPlot == 2:
                    sns.boxplot(x=CondCol, y=Parm, data=data_filtered, ax=ax[k], 
                                width = 0.5, showfliers = False, order= co_order, 
                                medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 4},
                                boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
            #                   boxprops={"color": color, "linewidth": 0.5},
                                whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                                capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4})
                                # scaley = scaley)
                elif stressBoxPlot == 1:
                    sns.boxplot(x=CondCol, y=Parm, data=data_filtered, ax=ax[k], 
                                width = 0.5, showfliers = False, order= co_order, 
                                medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
                                boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
            #                   boxprops={"color": color, "linewidth": 0.5},
                                whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                                capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2})
                                # scaley = scaley)
            else:
                sns.boxplot(x=CondCol, y=Parm, data=data_filtered, ax=ax[k], 
                            width = 0.5, showfliers = False, order= co_order, 
                            medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},
                            boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
        #                   boxprops={"color": color, "linewidth": 0.5},
                            whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                            capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2})
                            # scaley = scaley)
            
            # data_filtered.boxplot(column=Parm, by = CondCol, ax=ax[k],showfliers = False) # linewidth = 2, width = 0.5

        if stats:
            if len(box_pairs) == 0:
                box_pairs = makeBoxPairs(co_order)
            addStat_df(ax[k], data_filtered, box_pairs, Parm, CondCol, test = statMethod, percentHeight = 95)
            # add_stat_annotation(ax[k], x=CondCol, y=Parm, data=data_filtered,box_pairs = box_pairs,test=statMethod, text_format='star',loc='inside', verbose=2)
        
        if not useHue:
            sns.swarmplot(x=CondCol, y=Parm, data=data_filtered, ax=ax[k], order = co_order,
                          size=markersize, edgecolor='k', linewidth = 1*markersizeFactor, palette = p)
        else:
            sns.swarmplot(x=CondCol, y=Parm, data=data_filtered, ax=ax[k], order = co_order,
                          size=markersize, edgecolor='k', linewidth = 1*markersizeFactor, palette = p, 
                          hue = 'manipID')
            legend = ax[k].legend()
            legend.remove()

        ax[k].set_xlabel('')
        ax[k].set_ylabel(Parm)
        ax[k].tick_params(axis='x', labelrotation = 10)
        ax[k].yaxis.grid(True)
        if ax[k].get_yscale() == 'linear':
            ax[k].set_ylim([0, ax[k].get_ylim()[1]])
        
    plt.rcParams['axes.prop_cycle'] = gs.my_default_color_cycle
        
    
    output = (fig, ax)
    
    if returnData > 0:
        # define the export df
        cols_export_df = ['date', 'manipID', 'cellID']
        if not AvgPerCell: 
            cols_export_df.append('compNum')
        cols_export_df += ([CondCol, normalizeCol] + Parameters)
        cols_export_df += [Parameters[k]+ '_normalized' for k in range(NPlots)]
        cols_export_df = ufun.drop_duplicates_in_array(cols_export_df)        
        
        export_df = data_filtered[cols_export_df]
        print(export_df.columns)
        
        export_df = export_df.rename(columns = {normalizeCol : 'NormalizeWith_' + normalizeCol})
        print(export_df.columns)
        
        output += (export_df, )
    
    if returnCount > 0:
        groupByCell = count_df.groupby(cellID)
        d_agg = {'compNum':'count', CondCol:'first', 'date':'first', 'manipID':'first'}
        df_CountByCell = groupByCell.agg(d_agg).rename(columns={'compNum':'compCount'})

        groupByCond = df_CountByCell.reset_index().groupby(CondCol)
        d_agg = {cellID: 'count', 'compCount': 'sum', 
                 'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
        d_rename = {cellID:'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
        df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
        
        if returnCount == 1:
            output += (df_CountByCond, )
        elif returnCount == 2:
            output += (df_CountByCond, df_CountByCell)

    return(output)








def D1PlotDetailed(data, CondCol=[], Parameters=[], Filters=[], Boxplot=True, cellID='cellID', 
                   co_order=[], stats=True, statMethod='Mann-Whitney', box_pairs=[],
                   figSizeFactor = 1, markersizeFactor=1, orientation = 'h', showManips = True):
    
    data_f = data
    for fltr in Filters:
        data_f = data_f.loc[fltr]    
    NCond = len(CondCol)
    
    print(min(data_f['EChadwick'].values))
    
    if NCond == 1:
        CondCol = CondCol[0]
        
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += CondCol[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_f[newColName] = ''
        for i in range(NCond):
            data_f[newColName] += data_f[CondCol[i]].astype(str)
            data_f[newColName] = data_f[newColName].apply(lambda x : x + ' & ')
        data_f[newColName] = data_f[newColName].apply(lambda x : x[:-3])
        CondCol = newColName
        
    data_f_agg = getAggDf(data_f, cellID, CondCol, Parameters)
    
    NPlots = len(Parameters)
    Conditions = list(data_f[CondCol].unique())
        
    if orientation == 'h':
        fig, ax = plt.subplots(1, NPlots, figsize = (5*NPlots*NCond*figSizeFactor, 5))
    elif orientation == 'v':
        fig, ax = plt.subplots(NPlots, 1, figsize = (5*NCond*figSizeFactor, 5*NPlots))
    
    markersize = 5*markersizeFactor
    
    if NPlots == 1:
        ax = np.array([ax])
    
    # Colors and markers
    if len(co_order) > 0:
        Conditions = co_order
        gs.colorList, mL = getStyleLists(co_order, styleDict_MCA3)
    else:
        co_order = Conditions
        gs.colorList = gs.colorList10
    markerList = gs.markerList10
        
        
    for k in range(NPlots):
        
        Parm = Parameters[k]
        if 'EChadwick' in Parm or 'KChadwick' in Parm:
            ax[k].set_yscale('log')

        for i in range(len(Conditions)):
            c = Conditions[i]
            sub_data_f_agg = data_f_agg.loc[data_f_agg[CondCol] == c]
            Ncells = sub_data_f_agg.shape[0]
            
            color = gs.colorList[i]
            
            if showManips:
                allManipID = list(sub_data_f_agg['manipID'].unique())
                alreadyLabeledManip = []
                dictMarker = {}
                for mi in range(len(allManipID)):
                    dictMarker[allManipID[mi]] = markerList[mi]

                
            midPos = i
            startPos = i-0.4
            stopPos = i+0.4
            
            values = sub_data_f_agg[Parm + '_mean'].values
            errors = sub_data_f_agg[Parm + '_std'].values
            
            if Boxplot:
                ax[k].boxplot(values, positions=[midPos], widths=0.4, patch_artist=True,
                            showmeans=False, showfliers=False,
                            medianprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.5},
                            boxprops={"facecolor": color, "edgecolor": 'k',
                                      "linewidth": 1, 'alpha' : 0.5},
    #                         boxprops={"color": color, "linewidth": 0.5},
                            whiskerprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.5},
                            capprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.5},
                             zorder = 1)
                

            step = 0.8/(Ncells-1)
            
            for m in range(Ncells):
                
                thisCellID = sub_data_f_agg.index.values[m]
                thisCell_data = data_f.loc[data_f[cellID] == thisCellID]
                thisCell_allValues = thisCell_data[Parm].values
                nval = len(thisCell_allValues)
                
                if showManips:
                    thisManipID = sub_data_f_agg['manipID'].values[m]
                    marker = dictMarker[thisManipID]
                else:
                    marker = 'o'
                
                cellpos = startPos + m*step
                ax[k].errorbar([cellpos], [values[m]], color = color, marker = marker, markeredgecolor = 'k', 
                               yerr=errors[m], capsize = 2, ecolor = 'k', elinewidth=0.8, barsabove = False)
                
                if showManips and thisManipID not in alreadyLabeledManip:
                    alreadyLabeledManip.append(thisManipID)
                    textLabel = thisManipID
                    ax[k].plot([cellpos], [values[m]], 
                               color = color, 
                               marker = marker, markeredgecolor = 'k', markersize = markersize,
                               label = textLabel)
                    
                ax[k].plot(cellpos*np.ones(nval), thisCell_allValues, 
                           color = color, marker = '_', ls = '', markersize = markersize)

        ax[k].set_xlabel('')
        ax[k].set_xticklabels(labels = Conditions)
        ax[k].tick_params(axis='x', labelrotation = 10)
        
        ax[k].set_ylabel(Parameters[k])
        ax[k].yaxis.grid(True)
        
        ax[k].legend(fontsize = 6)
        
        if stats:
            renameDict = {Parameters[k] + '_mean' : Parameters[k]}
            if len(box_pairs) == 0:
                box_pairs = makeBoxPairs(co_order)
            addStat_df(ax[k], data_f_agg.rename(columns = renameDict), 
                    box_pairs, Parameters[k], CondCol, test = statMethod)
        
    plt.rcParams['axes.prop_cycle'] = gs.my_default_color_cycle
    return(fig, ax)




def D1PlotPaired(data, Parameters=[], Filters=[], Boxplot=True, cellID='cellID', 
                   co_order=[], stats=True, statMethod='Wilcox_greater', box_pairs=[],
                   figSizeFactor = 1, markersizeFactor=1, orientation = 'h', labels = []):
    
    data_f = data
    for fltr in Filters:
        data_f = data_f.loc[fltr]
    
    if orientation == 'h':
        fig, ax = plt.subplots(1, 1, figsize = (5*figSizeFactor, 5))
    elif orientation == 'v':
        fig, ax = plt.subplots(1, 1, figsize = (5*figSizeFactor, 5))
        
    markersize = 5*markersizeFactor
        
    if 'EChadwick' in Parameters[0]:
        ax.set_yscale('log')
    
    pos = 1
    start = pos - 0.4
    stop = pos + 0.4
    NParms = len(Parameters)
    width = (stop-start)/NParms
    
    if len(labels) == len(Parameters):
        tickLabels = labels
    else:
        tickLabels = Parameters
    
    posParms = [start + 0.5*width + i*width for i in range(NParms)]
    parmsValues = np.array([data_f[P].values for P in Parameters])
#     for P in Parameters:
#         print(len(data_f[P].values))
#         print(data_f[P].values)
#     print(parmsValues)
    NValues = parmsValues.shape[1]
    print('Number of values : {:.0f}'.format(NValues))
    
    for i in range(NParms):
        color = gs.my_default_color_list[i]
        
        ax.plot([posParms[i] for k in range(NValues)], parmsValues[i,:], 
                color = color, marker = 'o', markeredgecolor = 'k', markersize = markersize,
                ls = '', zorder = 3)
        
        if Boxplot:
            ax.boxplot([parmsValues[i]], positions=[posParms[i]], widths=width*0.8, patch_artist=True,
                        showmeans=False, showfliers=False,
                        medianprops={"color": 'k', "linewidth": 2, 'alpha' : 0.75},
                        boxprops={"facecolor": color, "edgecolor": 'k',
                                  "linewidth": 1, 'alpha' : 0.5},
#                         boxprops={"color": color, "linewidth": 0.5},
                        whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.5},
                        capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.5},
                         zorder = 1)
    
    for k in range(NValues):
        ax.plot(posParms, parmsValues[:,k], 'k', ls = '-', linewidth = 0.5, marker = '', zorder = 2)
        
    if stats:
        X = np.array(posParms)
        Y = parmsValues
        box_pairs = makeBoxPairs([i for i in range(NParms)])
        addStat_arrays(ax, X, Y, box_pairs, test = statMethod, percentHeight = 99)
        
    ax.set_xlabel('')
    ax.set_xlim([pos-0.5, pos+0.5])
    ax.set_xticks(posParms)
    ax.set_xticklabels(labels = tickLabels, fontsize = 10)
    ax.tick_params(axis='x', labelrotation = 10)
        
    return(fig, ax)



def D2Plot(data, fig = None, ax = None, XCol='', YCol='', CondCol='', Filters=[], 
           cellID='cellID', AvgPerCell=False, xscale = 'lin', yscale = 'lin', 
           figSizeFactor = 1, markers = [], markersizeFactor = 1):
    data_filtered = data
    for fltr in Filters:
        data_filtered = data_filtered.loc[fltr]
    
    NCond = len(CondCol)    
    if NCond == 1:
        CondCol = CondCol[0]
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += CondCol[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_filtered[newColName] = ''
        for i in range(NCond):
            data_filtered[newColName] += data_filtered[CondCol[i]].astype(str)
            data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x + ' & ')
        data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x[:-3])
        CondCol = newColName
    
    if AvgPerCell:
        group = data_filtered.groupby(cellID)
        dictAggMean = getDictAggMean(data_filtered)
        data_filtered = group.agg(dictAggMean.pop(cellID)) #.reset_index(level=0, inplace=True)
        data_filtered.reset_index(level=0, inplace=True)
        
    Conditions = list(data_filtered[CondCol].unique())
    
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize = (8*figSizeFactor,5))
    else:
        pass
    
    markersize = 5 * markersizeFactor
    
    if xscale == 'log':
        ax.set_xscale('log')
    if yscale == 'log':
        ax.set_yscale('log')
    
    current_color_list = getStyleLists(Conditions, styleDict_MCA3).as_hex()
    cc = cycler(color=current_color_list)
    ax.set_prop_cycle(cc)
    
    im = 0
    for c in Conditions:
        Xraw = data_filtered[data_filtered[CondCol] == c][XCol].values
        Yraw = data_filtered[data_filtered[CondCol] == c][YCol].values
        XYraw = np.array([Xraw,Yraw]).T
        XY = XYraw[~np.isnan(XYraw).any(axis=1), :]
        X, Y = XY[:,0], XY[:,1]
        if len(markers) == 0:
            m = 'o'
        else:
            m = markers[im]
            im += 1
        
        if len(X) == 0:
            ax.plot([], [])
#             if modelFit:
#                 ax.plot([], [])
                
        elif len(X) > 0:                    
            ax.plot(X, Y, m, markersize = markersize, markeredgecolor='k', markeredgewidth = 1, label=c)
            
    ax.set_xlabel(XCol)
    ax.set_xlim([min(0,1.1*np.min(data_filtered[XCol])), 1.1*np.max(data_filtered[XCol])])
    ax.set_ylabel(YCol)
    if not yscale == 'log':
        ax.set_ylim([min(0,1.1*np.min(data_filtered[YCol])), 1.1*np.max(data_filtered[YCol])])
    ax.legend(loc='upper left')
    return(fig, ax)




def D2Plot_wFit(data, fig = None, ax = None, 
                XCol='', YCol='', CondCol='', 
                Filters=[], cellID='cellID', co_order = [],
                AvgPerCell=False, showManips = True,
                modelFit=False, modelType='y=ax+b', writeEqn = True,
                xscale = 'lin', yscale = 'lin', 
                figSizeFactor = 1, markersizeFactor = 1):
    
    data_filtered = data
    for fltr in Filters:
        data_filtered = data_filtered.loc[fltr]
    
    NCond = len(CondCol)    
    if NCond == 1:
        CondCol = CondCol[0]
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += CondCol[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_filtered[newColName] = ''
        for i in range(NCond):
            data_filtered[newColName] += data_filtered[CondCol[i]].astype(str)
            data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x + ' & ')
        data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x[:-3])
        CondCol = newColName
    
    if AvgPerCell:
        group = data_filtered.groupby(cellID)
        dictAggMean = getDictAggMean(data_filtered)
        data_filtered = group.agg(dictAggMean.pop(cellID)) #.reset_index(level=0, inplace=True)
        data_filtered.reset_index(level=0, inplace=True)
        
    Conditions = list(data_filtered[CondCol].unique())
    
    if len(co_order) > 0:
        if len(co_order) != len(Conditions):
            delCo = [co for co in co_order if co not in Conditions]
            for co in delCo:
                co_order.remove(co)
                
        try:
            colorList, markerList = getStyleLists(co_order, styleDict_MCA3)
        except:
            colorList, markerList = gs.colorList30, gs.markerList10
    else:
        co_order = Conditions
        colorList, markerList = gs.colorList30, gs.markerList10
        
    
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize = (8*figSizeFactor,5))
    else:
        pass
    
    markersize = 5 * markersizeFactor
    
    if xscale == 'log':
        ax.set_xscale('log')
    if yscale == 'log':
        ax.set_yscale('log')
    
#     current_color_list = getStyleLists(Conditions, styleDict1).as_hex()
#     cc = cycler(color=current_color_list)
#     ax.set_prop_cycle(cc)
    
#     if modelFit:
#         # Tweak the style cycle to plot for each condition: the points ('o') and then the fit ('-') with the same color.
# #         current_prop_cycle = plt.rcParams['axes.prop_cycle']
# #         current_color_list = prop_cycle.by_key()['color']
#         ncustom_color_list = list(np.array([current_color_list, current_color_list]).flatten(order='F'))
#         # if new_color_list was ['red', 'green', blue'], custom_color_list is now ['red', 'red', 'green', 'green', blue', blue']
#         cc = cycler(color=ncustom_color_list)
#         ax.set_prop_cycle(cc)
    
    for i in range(len(co_order)):
        c = co_order[i]
        color = colorList[i]
#         marker = my_default_marker_list[i]
        Xraw = data_filtered[data_filtered[CondCol] == c][XCol].values
        Yraw = data_filtered[data_filtered[CondCol] == c][YCol].values
        Mraw = data_filtered[data_filtered[CondCol] == c]['manipID'].values
        XYraw = np.array([Xraw,Yraw]).T
        XY = XYraw[~np.isnan(XYraw).any(axis=1), :]
        X, Y = XY[:,0], XY[:,1]
        M = Mraw[~np.isnan(XYraw).any(axis=1)]
        if len(X) == 0:
            ax.plot([], [])
            if modelFit:
                ax.plot([], [])
                
        elif len(X) > 0:
            eqnText = ''

            if modelFit:
                print('Fitting condition ' + c + ' with model ' + modelType)
                if modelType == 'y=ax+b':
                    params, results = ufun.fitLine(X, Y) # Y=a*X+b ; params[0] = b,  params[1] = a
                    pval = results.pvalues[1] # pvalue on the param 'a'
                    eqnText += " ; Y = {:.1f} X + {:.1f}".format(params[1], params[0])
                    eqnText += " ; p-val = {:.3f}".format(pval)
                    print("Y = {:.5} X + {:.5}".format(params[1], params[0]))
                    print("p-value on the 'a' coefficient: {:.4e}".format(pval))
                    # fitY = params[1]*X + params[0]
                    # imin = np.argmin(X)
                    # imax = np.argmax(X)
                    # ax.plot([X[imin],X[imax]], [fitY[imin],fitY[imax]], '--', lw = '1',
                    #         color = color, zorder = 4)
                    fitX = np.linspace(np.min(X), np.max(X), 100)
                    fitY = params[1]*fitX + params[0]
                    ax.plot(fitX, fitY, '--', lw = '1', 
                            color = color, zorder = 4)

                elif modelType == 'y=A*exp(kx)':
                    params, results = ufun.fitLine(X, np.log(Y)) # Y=a*X+b ; params[0] = b,  params[1] = a
                    pval = results.pvalues[1] # pvalue on the param 'k'
                    eqnText += " ; Y = {:.1f}*exp({:.1f}*X)".format(params[0], params[1])
                    eqnText += " ; p-val = {:.3f}".format(pval)
                    print("Y = {:.5}*exp({:.5}*X)".format(np.exp(params[0]), params[1]))
                    print("p-value on the 'k' coefficient: {:.4e}".format(pval))
                    # fitY = np.exp(params[0])*np.exp(params[1]*X)
                    # imin = np.argmin(X)
                    # imax = np.argmax(X)
                    # ax.plot([X[imin],X[imax]], [fitY[imin],fitY[imax]], '--', lw = '1',
                    #         color = color, zorder = 4)
                    fitX = np.linspace(np.min(X), np.max(X), 100)
                    fitY = np.exp(params[0])*np.exp(params[1]*fitX)
                    ax.plot(fitX, fitY, '--', lw = '1', 
                            color = color, zorder = 4)
                    
                elif modelType == 'y=k*x^a':
                    posValues = ((X > 0) & (Y > 0))
                    X, Y = X[posValues], Y[posValues]
                    params, results = ufun.fitLine(np.log(X), np.log(Y)) # Y=a*X+b ; params[0] = b,  params[1] = a
                    k = np.exp(params[0])
                    a = params[1]
                    R2 = results.rsquared
                    pval = results.pvalues[1] # pvalue on the param 'a'
                    eqnText += " ; Y = {:.1e} * X^{:.1f}".format(k, a)
                    eqnText += " ; p-val = {:.3f}".format(pval)
                    eqnText += " ; R2 = {:.2f}".format(R2)
                    print("Y = {:.4e} * X^{:.4f}".format(k, a))
                    print("p-value on the 'a' coefficient: {:.4e}".format(pval))
                    print("R2 of the fit: {:.4f}".format(R2))
                    # fitY = k * X**a
                    # imin = np.argmin(X)
                    # imax = np.argmax(X)
                    # ax.plot([X[imin],X[imax]], [fitY[imin],fitY[imax]], '--', lw = '1', 
                    #         color = color, zorder = 4)
                    fitX = np.linspace(np.min(X), np.max(X), 100)
                    fitY = k * fitX**a
                    ax.plot(fitX, fitY, '--', lw = '1', 
                            color = color, zorder = 4)
                
                print('Number of values : {:.0f}'.format(len(Y)))
                print('\n')
            
#                 if showManips:
#                     allManipID = list(data_filtered[data_filtered[CondCol] == c]['manipID'].unique())
#                     dictMarker = {}
#                     markers = []
#                     for mi in range(len(allManipID)):
#                         dictMarker[allManipID[mi]] = my_default_marker_list[mi]
#                     for k in range(len(M)):
#                         thisManipID = M[k]
#                         markers = dictMarker[thisManipID]
#                 else:
#                     markers = ['o' for i in range(len(M))]
#                 markers = np.array(markers)
                
            labelText = c
            if writeEqn:
                labelText += eqnText

            ax.plot(X, Y, 
                    color = color, ls = '',
                    marker = 'o', markersize = markersize, markeredgecolor='k', markeredgewidth = 1, 
                    label = labelText)
            
            
    ax.set_xlabel(XCol)
    ax.set_xlim([0.9*np.min(data_filtered[XCol]), 1.1*np.max(data_filtered[XCol])])
    ax.set_ylabel(YCol)
    if not yscale == 'log':
        ax.set_ylim([0.9*np.min(data_filtered[YCol]), 1.1*np.max(data_filtered[YCol])])
    ax.legend(loc='upper left')
    
    
    return(fig, ax)





# %%% Subfunctions

    
def getDictAggMean(df):
    dictAggMean = {}
    for c in df.columns:
        # print(c)
        S = df[c].dropna()
        lenNotNan = S.size
        if lenNotNan == 0:
            dictAggMean[c] = 'first'
        else:
            S.infer_objects()
            if S.dtype == bool:
                dictAggMean[c] = np.nanmin
            else:
                # print(c)
                # print(pd.Series.all(S.apply(lambda x : isinstance(x, numbers.Number))))
                if pd.Series.all(S.apply(lambda x : isinstance(x, numbers.Number))):
                    dictAggMean[c] = np.nanmean
                else:
                    dictAggMean[c] = 'first'
                    
    if 'compNum' in dictAggMean.keys():
        dictAggMean['compNum'] = np.nanmax
                    
    return(dictAggMean)



# def getDictAggMean_V1(df):
#     dictAggMean = {}
#     for c in df.columns:
#     #         t = df[c].dtype
#     #         print(c, t)
#             try:
#                 if np.array_equal(df[c], df[c].astype(bool)):
#                     dictAggMean[c] = 'min'
#                 else:
#                     try:
#                         if not c.isnull().all():
#                             dictAggMean[c] = 'nanmean'
#                     except:
#                         dictAggMean[c] = 'first'
#             except:
#                     dictAggMean[c] = 'first'
#     return(dictAggMean)


def getAggDf(df, cellID, CondCol, Variables):
    
    allCols = df.columns.values
    group = df.groupby(cellID)
    
    dictAggCategories = {}
    Categories = ['date', 'cellName', 'manipID', 'experimentType', 'drug', 
                  'substrate', 'cell type', 'cell subtype', 'bead type', 'bead diameter']
    if CondCol not in Categories:
        Categories += [CondCol]
    for c in Categories:
        if c in allCols:
            dictAggCategories[c] = 'first'
    dfCats = group.agg(dictAggCategories)
    
    dictAggCount = {'compNum' : 'count'}
    renameDict = {'compNum' : 'Ncomp'}
    dfCount = group.agg(dictAggCount).rename(columns=renameDict)
    
    dictAggMeans = {}
    renameDict = {}
    for v in Variables:
        dictAggMeans[v] = 'mean'
        renameDict[v] = v + '_mean'
    dfMeans = group.agg(dictAggMeans).rename(columns=renameDict)
        
    dictAggStds = {}
    renameDict = {}
    for v in Variables:
        dictAggStds[v] = 'std'
        renameDict[v] = v + '_std'
    dfStds = group.agg(dictAggStds).rename(columns=renameDict)
    
    if 'ctFieldThickness' in Variables:
        dfStds['ctFieldThickness_std'] = group.agg({'ctFieldFluctuAmpli' : 'median'}).values
        #ctFieldThickness ; ctFieldFluctuAmpli
    
#     for v in Variables:
        
    
    dfAgg = pd.merge(dfCats, dfCount, how="inner", on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
    
    dfAgg = pd.merge(dfAgg, dfMeans, how="inner", on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
    
    dfAgg = pd.merge(dfAgg, dfStds, how="inner", on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
    
    return(dfAgg)


def getAggDf_weightedAvg(df, cellID, CondCol, Variables, WeightCols):
    
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
    
    dictWeights = {}
    for i in range(len(Variables)):
        v = Variables[i]
        w = WeightCols[i]
        dictWeights[v] = w

    for v in Variables:
        df[v] = df[v].apply(nan2zero)
        df[dictWeights[v]] = df[dictWeights[v]].apply(nan2zero)
    cellIDvals = df['cellID'].unique()
    for v in Variables:
        for c in cellIDvals:
            if np.sum(df.loc[df['cellID'] == c, v].values) == 0:
                df.loc[df['cellID'] == c, dictWeights[v]].apply(lambda x: 0)
    
    
    
    allCols = df.columns.values
    group = df.groupby(cellID)
    
    dictAggCategories = {}
    Categories = ['date', 'cellName', 'manipID', 'experimentType', 'drug', 
                  'substrate', 'cell type', 'cell subtype', 'bead type', 'bead diameter']
    if CondCol not in Categories:
        Categories += [CondCol]
    for c in Categories:
        if c in allCols:
            dictAggCategories[c] = 'first'
    dfCats = group.agg(dictAggCategories)
    
    dictAggCount = {'compNum' : 'count'}
    renameDict = {'compNum' : 'Ncomp'}
    dfCount = group.agg(dictAggCount).rename(columns=renameDict) 
    
    dfAgg = pd.merge(dfCats, dfCount, how="inner", on='cellID',
    #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
    #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
    )
        
    dictAggMeans = {}
    renameDict = {}
    for i in range(len(Variables)):
        v = Variables[i]
        w = WeightCols[i]
        df_i = df[[cellID, v, w]]
        group_i = df_i.groupby(cellID)

        def WMfun(x):
            try:
                return(np.average(x, weights=df.loc[x.index, dictWeights[v]]))
            except:
                return(np.nan)
            
        def WSfun(x):
            try:
                return(w_std(x, df.loc[x.index, dictWeights[v]].values))
            except:
                return(np.nan)
        
        def Countfun(x):
            try:
                return(len(x.values[x.values != 0]))
            except:
                return(0)
        
        dictAggMeans[v] = WMfun
        renameDict[v] = v + '_Wmean'
        dfWMeans_i = group_i.agg({v : WMfun}).rename(columns={v : v + '_Wmean'})
        dfWStd_i = group_i.agg({v : WSfun}).rename(columns={v : v + '_Wstd'})
        dfWCount_i = group_i.agg({v : Countfun}).rename(columns={v : v + '_Count'})
        dfAgg = pd.merge(dfAgg, dfWMeans_i, how="inner", on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        dfAgg = pd.merge(dfAgg, dfWStd_i, how="inner", on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        
        dfAgg = pd.merge(dfAgg, dfWCount_i, how="inner", on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )

    return(dfAgg)


def makeOrder(*args):
    order = []
    listeTuple = list(itertools.product(*args, repeat=1))
    for tup in listeTuple:
        tmpText = ''
        for word in tup:
            tmpText += word
            tmpText += ' & '
        tmpText = tmpText[:-3]
        order.append(tmpText)
    return(order)


def makeBoxPairs(O):
    return(list(itertools.combinations(O, 2)))


def renameAxes(axes, rD):
    try:
        N = len(axes)
    except:
        axes = [axes]
        N = 1
    for i in range(N):
        # set xticks
        xticksTextObject = axes[i].get_xticklabels()
        xticksList = [xticksTextObject[j].get_text() for j in range(len(xticksTextObject))]
        test_hasXLabels = (len(''.join(xticksList)) > 0)
        if test_hasXLabels:
            newXticksList = [rD.get(k, k) for k in xticksList]
            axes[i].set_xticklabels(newXticksList)
        
        # set xlabel
        xlabel = axes[i].get_xlabel()
        newXlabel = rD.get(xlabel, xlabel)
        axes[i].set_xlabel(newXlabel)
        # set ylabel
        ylabel = axes[i].get_ylabel()
        newYlabel = rD.get(ylabel, ylabel)
        axes[i].set_ylabel(newYlabel)
        

def addStat_df(ax, data, box_pairs, param, cond, test = 'Mann-Whitney', percentHeight = 98):
    refHeight = np.percentile(data[param].values, percentHeight)
    currentHeight = refHeight
    scale = ax.get_yscale()
    xTicks = ax.get_xticklabels()
    dictXTicks = {xTicks[i].get_text() : xTicks[i].get_position()[0] for i in range(len(xTicks))}
    for bp in box_pairs:
        c1 = data[data[cond] == bp[0]][param].values
        c2 = data[data[cond] == bp[1]][param].values
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
        if pval < 0.05 and pval > 0.01:
            text = '*'
        elif pval < 0.01 and pval > 0.001:
            text = '**'
        elif pval < 0.001 and pval < 0.001:
            text = '***'
        elif pval < 0.0001:
            text = '****'
        ax.plot([bp[0], bp[1]], [currentHeight, currentHeight], 'k-', lw = 1.5, zorder = 4)
        XposText = (dictXTicks[bp[0]]+dictXTicks[bp[1]])/2
        if scale == 'log':
            power = 0.01* (text=='ns') + 0.000 * (text!='ns')
            YposText = currentHeight*(refHeight**power)
        else:
            factor = 0.02 * (text=='ns') + 0.000 * (text!='ns')
            YposText = currentHeight + factor*refHeight
        ax.text(XposText, YposText, text, ha = 'center', color = 'k', size = 11, zorder = 4)
#         if text=='ns':
#             ax.text(posText, currentHeight + 0.025*refHeight, text, ha = 'center')
#         else:
#             ax.text(posText, currentHeight, text, ha = 'center')
        if scale == 'log':
            currentHeight = currentHeight*(refHeight**0.05)
        else:
            currentHeight =  currentHeight + 0.1*refHeight
    # ax.set_ylim([ax.get_ylim()[0], currentHeight])
    

def addStat_arrays(ax, X, Y, box_pairs, test = 'Mann-Whitney', percentHeight = 98):
    refHeight = np.percentile(Y, percentHeight)
    currentHeight = refHeight
    scale = ax.get_yscale()
    
    for bp in box_pairs:
        y1 = Y[bp[0], :]
        y2 = Y[bp[1], :]
        
        if test=='Mann-Whitney':
            statistic, pval = st.mannwhitneyu(y1, y2)
        elif test=='Wilcox_2s':
            statistic, pval = st.wilcoxon(y1, y2, alternative = 'two-sided')
        elif test=='Wilcox_greater':
            statistic, pval = st.wilcoxon(y1, y2, alternative = 'greater')
        elif test=='Wilcox_less':
            statistic, pval = st.wilcoxon(y1, y2, alternative = 'less')
        elif test=='t-test':
            statistic, pval = st.ttest_ind(y1, y2)
            
        text = 'ns'
        if pval < 0.05 and pval > 0.01:
            text = '*'
        elif pval < 0.01 and pval > 0.001:
            text = '**'
        elif pval < 0.001 and pval < 0.001:
            text = '***'
        elif pval < 0.0001:
            text = '****'
            
        ax.plot([X[bp[0]], X[bp[1]]], [currentHeight, currentHeight], 'k-', lw = 1)
        XposText = (X[bp[0]]+X[bp[1]])/2
        if scale == 'log':
            power = 0.01* (text=='ns') + 0.000 * (text!='ns')
            YposText = currentHeight*(refHeight**power)
        else:
            factor = 0.03 * (text=='ns') + 0.000 * (text!='ns')
            YposText = currentHeight + factor*refHeight
        ax.text(XposText, YposText, text, ha = 'center', color = 'k')
#         if text=='ns':
#             ax.text(posText, currentHeight + 0.025*refHeight, text, ha = 'center')
#         else:
#             ax.text(posText, currentHeight, text, ha = 'center')
        if scale == 'log':
            currentHeight = currentHeight*(refHeight**0.05)
        else:
            currentHeight =  currentHeight + 0.15*refHeight
    ax.set_ylim([ax.get_ylim()[0], currentHeight])


def getStyleCycle(co_order, styleDict):
    colors = []
    markers = []
    linestyles = []
    linewidth = []

    for co in co_order:
        coStyle = styleDict[co]
        if 'color' in coStyle.keys():
            colors.append(coStyle['color'])
        else:
            colors.append('')
        if 'marker' in coStyle.keys():
            markers.append(coStyle['marker'])
        else:
            markers.append('')
        if 'linestyle' in coStyle.keys():
            linestyles.append(coStyle['marker'])
        else:
            linestyles.append('')
        if 'linewidth' in coStyle.keys():
            linewidth.append(coStyle['linewidth'])
        else:
            linewidth.append(1)
            
    cc = (cycler(color=colors) +
          cycler(linestyle=linestyles) +
          cycler(marker=markers) +
          cycler(linewidth=linewidth))
            
    return(cc)


def getStyleLists_Sns(co_order, styleDict):
    colors = []
    markers = []
    try:
        for co in co_order:
            coStyle = styleDict[co]
            if 'color' in coStyle.keys():
                colors.append(coStyle['color'])
            else:
                colors.append('')
            if 'marker' in coStyle.keys():
                markers.append(coStyle['marker'])
            else:
                markers.append('')
        palette = sns.color_palette(colors)
    except:
        palette = sns.color_palette()
    return(palette)

def getStyleLists(co_order, styleDict):
    colors = []
    markers = []
    for co in co_order:
        coStyle = styleDict[co]
        if 'color' in coStyle.keys():
            colors.append(coStyle['color'])
        else:
            colors.append('')
        if 'marker' in coStyle.keys():
            markers.append(coStyle['marker'])
        else:
            markers.append('')
    print(colors, markers)
    return(colors, markers)


def buildStyleDictMCA():
    # TBC
    styleDict = {}
    return(styleDict)


# %%% Tests of plotting functions

#### Test getDictAggMean(df)

# df = GlobalTable_meca_MCA123
# # df = sns.load_dataset("tips")
# cellID = 'cellID'
# # cellID = 'day'

# group = df.groupby(cellID)
# dictAggMean = getDictAggMean(df)
# df_agg = group.agg(dictAggMean)

# group = df.groupby(cellID)
# dictAggMean = getDictAggMean_V1(df)
# df_agg2 = group.agg(dictAggMean)


# df = GlobalTable_meca_MCA123
# # df = sns.load_dataset("tips")
# cellID = 'cellID'
# # cellID = 'day'

# testCols = df.columns.drop('cellID')

# for c in testCols:
#     print(c)
#     subdf = df[['cellID', c]]
#     group = subdf.groupby('cellID')
#     dictAggMean = getDictAggMean(subdf)
#     subdf_agg = group.agg(dictAggMean)


#### Test getAggDf_weightedAvg(df, cellID, CondCol, Variables, WeightCols)

# data = GlobalTable_meca_nonLin

# dates = ['22-02-09'] #['21-12-08', '22-01-12'] ['21-01-18', '21-01-21', '21-12-08']

# filterList = [(data['validatedThickness'] == True),
#               (data['cell subtype'] == 'aSFL'), 
#               (data['bead type'].apply(lambda x : x == list(['M450']))),
#               (data['substrate'] == '20um fibronectin discs'),
#               (data['date'].apply(lambda x : x in dates))]  # (data['validatedFit'] == True), 
# globalFilter = filterList[0]
# for k in range(1, len(filterList)):
#     globalFilter = globalFilter & filterList[k]

# data_f = data[globalFilter]

# fitMin = [S for S in range(25,1225,50)]
# fitMax = [S+150 for S in fitMin]
# fitCenters = np.array([S+75 for S in fitMin])
# regionFitsNames = [str(fitMin[ii]) + '<s<' + str(fitMax[ii]) for ii in range(len(fitMin))]

# listColumnsMeca = []

# KChadwick_Cols = []
# KWeight_Cols = []

# for rFN in regionFitsNames:
#     listColumnsMeca += ['KChadwick_'+rFN, 'K_CIW_'+rFN, 'R2Chadwick_'+rFN, 'K2Chadwick_'+rFN, 
#                         'H0Chadwick_'+rFN, 'Npts_'+rFN, 'validatedFit_'+rFN]
#     KChadwick_Cols += [('KChadwick_'+rFN)]

#     K_CIWidth = data_f['K_CIW_'+rFN] #.apply(lambda x : x.strip('][').split(', ')).apply(lambda x : (np.abs(float(x[0]) - float(x[1]))))
#     KWeight = (data_f['KChadwick_'+rFN]/K_CIWidth) # **2
#     data_f['K_Weight_'+rFN] = KWeight
#     data_f['K_Weight_'+rFN] *= data_f['KChadwick_'+rFN].apply(lambda x : (x<1e6))
#     data_f['K_Weight_'+rFN] *= data_f['R2Chadwick_'+rFN].apply(lambda x : (x>1e-2))
#     data_f['K_Weight_'+rFN] *= data_f['K_CIW_'+rFN].apply(lambda x : (x!=0))
#     KWeight_Cols += [('K_Weight_'+rFN)]
    

# # dictWeights = {}
# # for i in range(len(Variables)):
# #     v = Variables[i]
# #     w = WeightCols[i]
# #     dictWeights[v] = w
# # def nan2zero(x):
# #     if np.isnan(x):
# #         return(0)
# #     else:
# #         return(x)

# # for v in Variables:
# #     df[v] = df[v].apply(nan2zero)
# #     df[dictWeights[v]] = df[dictWeights[v]].apply(nan2zero)
# # cellIDvals = df['cellID'].unique()
# # for v in Variables:
# #     for c in cellIDvals:
# #         if np.sum(df.loc[df['cellID'] == c, v].values) == 0:
# #             df.loc[df['cellID'] == c, dictWeights[v]].apply(lambda x: 1)
            
# # df

# CondCol = 'date'
# Variables = KChadwick_Cols
# WeightCols = KWeight_Cols
# data_f_agg = getAggDf_weightedAvg(data_f, 'cellID', CondCol, Variables, WeightCols)
# data_f_agg


#### Test getAggDf(df, cellID, CondCol, Variables)

# data = GlobalTable_meca_Py2

# Filters = [(data['validatedFit'] == True), (data['validatedThickness'] == True)]

# data_f = data
# for fltr in Filters:
#     data_f = data_f.loc[fltr]

# # dfA = getAggDf(data_f, 'cellID', 'bead type', ['surroundingThickness', 'EChadwick'])
# dfA = getAggDf(data_f, 'cellID', 'bead type', ['ctFieldThickness', 'EChadwick'])
# dfA


#### Test the D2Plot_wFit
# data = GlobalTable_meca_Py2

# Filters = [(data['validatedFit'] == True), 
#            (data['validatedThickness'] == True), 
#            (data['substrate'] == '20um fibronectin discs'),
#            (data['date'].apply(lambda x : x in ['21-12-16','21-12-08']))]

# fig, ax = D2Plot_wFit(data, XCol='ctFieldThickness', YCol='EChadwick', CondCol = ['bead type'],
#            Filters=Filters, cellID = 'cellID', AvgPerCell=True, xscale = 'log', yscale = 'log', 
#            modelFit=True, modelType = 'y=k*x^a')

# ax.set_ylabel('EChadwick (Pa)')
# ax.set_xlabel('Thickness at low force (nm)')
# fig.suptitle('3T3aSFL: E(h)')
# ax.legend(loc = 'upper right')

# # ufun.archiveFig(fig, ax, name='E(h)_3T3aSFL_Dec21_M450-5-13_vs_M270-14-54', figDir = cp.DirDataFigToday + '//' + figSubDir)
# plt.show()


#### Test of the averaging per cell routine

# data = GlobalTable_meca
# CondCol='drug'
# Parameters=['SurroundingThickness','EChadwick']
# Filters = [(GlobalTable_meca['Validated'] == 1)]
# AvgPerCell=True
# cellID='CellName'

# data_filtered = data
# for fltr in Filters:
#     data_filtered = data_filtered.loc[fltr]

# group = data_filtered.groupby(cellID)
# dictAggMean = getDictAggMean(data_filtered)
# data_filtered = group.agg(dictAggMean.pop(cellID)) #.reset_index(level=0, inplace=True)
# data_filtered.reset_index(level=0, inplace=True)
# data_filtered=data_filtered[[cellID]+[CondCol]+Parameters]
# print(data_filtered)


#### Test of a routine to remove points of a list of XY positions where at least 1 of the coordinates is 'nan'

# XYraw = np.array([[np.nan, 2, 3, np.nan, 5], [10,20,30,40,50]])
# XYraw = XYraw.T
# XY = XYraw[~np.isnan(XYraw).any(axis=1), :]
# X, Y = XY[:,0], XY[:,1]
# X, Y


#### Test of a routine to double each element in a list ; example [1, 2, 3] -> [1, 1, 2, 2, 3, 3]

# newnew_color_list = np.array([new_color_list, new_color_list])
# custom_color_list = list(np.array([new_color_list, new_color_list]).flatten(order='F'))
# custom_color_list


#### Test of makeOrder function

# print(makeOrder(['none','doxycyclin'],['BSA coated glass','20um fibronectin discs']))
# print(makeOrder(['A','B']))
# print(makeOrder(['A','B'], ['C','D']))
# print(makeOrder(['A','B'], ['C','D'], ['E','F']))


#### Test of makeBoxPairs function

# O = makeOrder(['none','doxycyclin'],['BSA coated glass','20um fibronectin discs'])
# makeBoxPairs(O)


#### Test of custom cycles

# cc = (cycler(color=list('rgb')) +
#       cycler(linestyle=['', '', '']) +
#       cycler(marker=['o','*','<']) +
#      cycler(linewidth=[1,1,1]))
# cc

# plt.rcParams['axes.prop_cycle'] = cc

# fig, ax = plt.subplots()
# for i in range(10):
#     ax.plot([0,1], [0,i])
# plt.show()



# %% Plots



# %%% MCA project - Matlab processing

# %%%%% 3T3aSFL on patterns: Ct Field


Filters = [(GlobalTable_ctField['validated'] == True), 
           (GlobalTable_ctField['medianThickness'] <= 1000)]

fig, ax = D1Plot(GlobalTable_ctField, CondCols=['drug','substrate'],
                 Parameters=['medianThickness','fluctuAmpli'],
                 Filters=Filters)

fig.suptitle('3T3aSFL on patterns: Ct Field')
plt.show()


# %%%%% 3T3aSFL on diverse substrates: Compressions


Filters = [(GlobalTable_meca['Validated'] == 1), 
           (GlobalTable_meca['cell subtype'] == 'aSFL')]

co_order = makeOrder(['none','doxycyclin'],['BSA coated glass','20um fibronectin discs'])
fig, ax = D1Plot(GlobalTable_meca, CondCols=['drug','substrate'],
                 Parameters=['SurroundingThickness','EChadwick'],
                 Filters=Filters,AvgPerCell=False,cellID='CellName', co_order=co_order,
                 markersizeFactor = 0.6)

fig.suptitle('3T3aSFL on diverse substrates: Compressions')
plt.show()

# %%%%% 3T3aSFL on patterns: Compressions


Filters = [(GlobalTable_meca['Validated'] == 1),
           (GlobalTable_meca['substrate'] == '20um fibronectin discs'),
           (GlobalTable_meca['cell subtype'] == 'aSFL-6FP')]

fig, ax = D1Plot(GlobalTable_meca, CondCols=['drug','substrate'],
                 Parameters=['SurroundingThickness','EChadwick'],
                 Filters=Filters, AvgPerCell=False, cellID='CellName',
                 markersizeFactor = 1, orientation = 'v')

fig.suptitle('3T3aSFL-6FP on patterns: Compressions')
plt.show()


# %%%%% 3T3aSFL on diverse substrates: Compressions


Filters = [(GlobalTable_meca['Validated'] == 1), 
           (GlobalTable_meca['substrate'] == '20um fibronectin discs')]

co_order = makeOrder([['aSFL','aSFL-6FP'],['none','doxycyclin']])
fig, ax = D1Plot(GlobalTable_meca, CondCols=['cell subtype','drug'],
                 Parameters=['SurroundingThickness','EChadwick'],
                 Filters=Filters,AvgPerCell=True,cellID='CellName',co_order=co_order)
fig.suptitle('3T3aSFL on diverse substrates: Compressions')
plt.show()


# %%%%% 3T3aSFL SHORT vs LONG linker: Compressions


Filters = [(GlobalTable_meca['Validated'] == 1), 
           (GlobalTable_meca['substrate'] == '20um fibronectin discs')]

co_order = makeOrder([['aSFL','aSFL-6FP'],['none','doxycyclin']])
fig, ax = D1Plot(GlobalTable_meca, CondCols=['cell subtype','drug'],
                 Parameters=['SurroundingThickness','EChadwick'],
                 Filters=Filters,AvgPerCell=True,cellID='CellName',co_order=co_order)
fig.suptitle('3T3aSFL SHORT vs LONG linker: Compressions')
plt.show()


# %%%%% 3T3aSFL - Dh = f(H)


Filters = [(GlobalTable_ctField['validated'] == True), 
           (GlobalTable_ctField['cell subtype'] == 'aSFL')]

fig, ax = D2Plot_wFit(GlobalTable_ctField, XCol='medianThickness',YCol='fluctuAmpli',
                 CondCol = ['drug'], Filters=Filters, modelFit=False)
fig.suptitle('3T3aSFL - Dh = f(H)')
# ufun.archiveFig(fig, ax, name='aSFL_Dh(h)_drug', figDir = cp.DirDataFigToday + '//' + 'ThicknessPlots')
plt.show()


# %%%%% 3T3aSFL - Dh = f(H) - medianThickness <= 700

# Same as above without the 2 thickest cells
Filters = [(GlobalTable_ctField['validated'] == True), 
           (GlobalTable_ctField['cell subtype'] == 'aSFL'), 
           (GlobalTable_ctField['medianThickness'] <= 700)]

fig, ax = D2Plot_wFit(GlobalTable_ctField, XCol='medianThickness',YCol='fluctuAmpli',
                 CondCol = ['drug'], Filters=Filters, modelFit=False)
fig.suptitle('3T3aSFL - Dh = f(H)')
# ufun.archiveFig(fig, ax, name='aSFL_Dh(h)_drug_wo2LastPoints', figDir = cp.DirDataFigToday + '//' + 'ThicknessPlots')
plt.show()


# %%%%% 3T3aSFL: E(h) - glass & fibro

Filters = [(GlobalTable_meca['Validated'] == True), 
           (GlobalTable_meca['cell subtype'] == 'aSFL')]

fig, ax = D2Plot_wFit(GlobalTable_meca, XCol='SurroundingThickness',YCol='EChadwick',
                 CondCol = ['substrate','drug'],Filters=Filters, cellID = 'CellName', 
                 AvgPerCell=True, modelFit=False, modelType='y=A*exp(kx)', yscale = 'log')

fig.suptitle('3T3aSFL: E(h)')
renameAxes(ax,renameDict1)
ax.legend(loc='upper right')
# ufun.archiveFig(fig, ax, name='aSFL_E(h)_drug&substrate_01', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%% 3T3aSFL: E(h) - only fibro

Filters = [(GlobalTable_meca['Validated'] == True), 
           (GlobalTable_meca['cell subtype'] == 'aSFL'), 
           (GlobalTable_meca['substrate'] == '20um fibronectin discs')]

fig, ax = D2Plot_wFit(GlobalTable_meca, XCol='SurroundingThickness',YCol='EChadwick',
                 CondCol = ['substrate','drug'],Filters=Filters,
                 cellID = 'CellName', AvgPerCell=False, modelFit=False, 
                 modelType='y=A*exp(kx)', xscale = 'log', yscale = 'log')

fig.suptitle('3T3aSFL: E(h)')
renameAxes(ax,renameDict1)
ax.set_xlim([90, 1500])
ax.legend(loc='upper right')
# ufun.archiveFig(fig, ax, name='aSFL_E(h)_drug&substrate_00_allComp', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%% 3T3aSFL: E(h) - only fibro - several cell subtype

Filters = [(GlobalTable_meca['Validated'] == True), 
           (GlobalTable_meca['substrate'] == '20um fibronectin discs')]

fig, ax = D2Plot_wFit(GlobalTable_meca, XCol='SurroundingThickness',YCol='EChadwick',
                 CondCol = ['cell subtype','drug'], Filters=Filters, 
                 cellID = 'CellName', AvgPerCell=True, modelFit=False, 
                 modelType='y=A*exp(kx)', yscale = 'log')

fig.suptitle('3T3aSFL: E(h)')
renameAxes(ax,renameDict1)
ax.legend(loc='upper right')
# ufun.archiveFig(fig, ax, name='aSFL_E(h)_drug&substrate_02', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%% 3T3aSFL: E(h) - only fibro - several cell subtype - all comp

Filters = [(GlobalTable_meca['Validated'] == True), 
           (GlobalTable_meca['substrate'] == '20um fibronectin discs')]

fig, ax = D2Plot_wFit(GlobalTable_meca, XCol='SurroundingThickness',YCol='EChadwick',
                 CondCol = ['cell subtype','drug'],Filters=Filters, 
                 cellID = 'CellName', AvgPerCell=False, modelFit=False, 
                 modelType='y=A*exp(kx)', yscale = 'log')

fig.suptitle('3T3aSFL: E(h)')
renameAxes(ax,renameDict1)
ax.legend(loc='upper right')
# ufun.archiveFig(fig, ax, name='aSFL_E(h)_drug&substrate_02_allComp', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%% Some test regarding plot style

co_o = ['BSA coated glass & none', 'BSA coated glass & doxycyclin', 
        '20um fibronectin discs & doxycyclin', '20um fibronectin discs & none']

sns.color_palette(['#ff9896', '#d62728', '#1f77b4', '#aec7e8'])

# getStyleLists(co_o, styleDict1)

# xticksTextObject = ax.get_xticklabels()
# xticksList = [xticksTextObject[j].get_text() for j in range(len(xticksTextObject))]
# newXticksList = [renameDict1.get(k, k) for k in xticksList]
# newXticksList


# %%%%% 3T3aSFL - All linker types: All Compressions

Filters = [(GlobalTable_meca_Py['validatedFit'] == True), 
           (GlobalTable_meca_Py['validatedThickness'] == True), 
           (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs')]

co_order = makeOrder(['aSFL','aSFL-A8','aSFL-6FP'], ['none','doxycyclin'])

box_pairs=[('aSFL & none', 'aSFL & doxycyclin'),
 ('aSFL-A8 & none', 'aSFL-A8 & doxycyclin'),
 ('aSFL-6FP & none', 'aSFL-6FP & doxycyclin'),
 ('aSFL & none', 'aSFL-A8 & none'),
 ('aSFL & none', 'aSFL-6FP & none')]

fig, ax = D1Plot(GlobalTable_meca_Py, CondCols=['cell subtype','drug'],
                 Parameters=['surroundingThickness','EChadwick'],Filters=Filters,
                 AvgPerCell=False,cellID='cellID',co_order=co_order,
                 box_pairs=box_pairs,stats=True,markersizeFactor = 0.5,
                 orientation = 'v')

renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL - All linker types: All Compressions')
plt.show()

# %%% MCA project - Py1 > half python half matlab processing

figSubDir = 'MCAproject'
# print(pairedPalette)
# pairedPalette


# %%%% 3T3 aSFL (A11) on and off pattern, with or without doxy -- (january 2021)

# %%%%%


data = GlobalTable_meca # 100% matlab

Filters = [(data['Validated'] == 1), (data['cell subtype'] == 'aSFL'), 
           (data['substrate'] == 'BSA coated glass'), (data['SurroundingThickness'] > 0)]

co_order = makeOrder(['BSA coated glass'],['none','doxycyclin'])

fig, ax = D1Plot(data, CondCols=['substrate','drug'],Parameters=['SurroundingThickness','EChadwick'], 
                 Filters=Filters,AvgPerCell=False, cellID='CellName', co_order=co_order, 
                 figSizeFactor = 0.5)

renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL non-adherent: Compressions')
# ufun.archiveFig(fig, ax, name='3T3aSFLonBSA_drug_SurroundingThickness&EChadwick_allComp', figSubDir = figSubDir)
plt.show()


# %%%%%


data = GlobalTable_meca # 100% matlab
dates = ['20-08-04', '20-08-05', '20-08-07', '21-01-18', '21-01-21']

Filters = [(data['Validated'] == 1), (data['cell subtype'] == 'aSFL'), 
           (data['SurroundingThickness'] > 0), (data['date'].apply(lambda x : x in dates))]

co_order = makeOrder(['BSA coated glass','20um fibronectin discs'],['none','doxycyclin'])
fig, ax = D1Plot(GlobalTable_meca, CondCols=['substrate','drug'],
                 Parameters=['SurroundingThickness','EChadwick'],
                 Filters=Filters,AvgPerCell=True, cellID='CellName', 
                 co_order=co_order, figSizeFactor = 0.8)

renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL on diverse substrates: Compressions')
# ufun.archiveFig(fig, ax, name='3T3aSFL_substrate&drug_SurroundingThickness&EChadwick', figSubDir = figSubDir)
plt.show()


# %%%%%


data = GlobalTable_meca_Py # Tracking matlab, postprocessing python
dates = ['20-08-04', '20-08-05', '20-08-07', '21-01-18', '21-01-21']

Filters = [(data['validatedFit'] == True), (data['validatedThickness'] == True), 
           (data['cell subtype'] == 'aSFL'), (data['date'].apply(lambda x : x in dates))]

co_order = makeOrder(['BSA coated glass','20um fibronectin discs'],['none','doxycyclin'])
fig, ax = D1Plot(data, CondCols=['substrate','drug'],Parameters=['surroundingThickness','EChadwick'],
                 Filters=Filters,AvgPerCell=True, cellID='cellID', co_order=co_order, 
                 figSizeFactor = 0.8,useHue = False, orientation = 'v')

renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL on diverse substrates: Compressions')
# ufun.archiveFig(fig, ax, name='3T3aSFL_substrate&drug_SurroundingThickness&EChadwick_NEWTABLE', figSubDir = figSubDir)
plt.show()


# %%%%%


data = GlobalTable_meca_Py # Tracking matlab, postprocessing python
dates = ['20-08-04', '20-08-05', '20-08-07', '21-01-18', '21-01-21']

Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True), 
           (data['cell subtype'] == 'aSFL'),
           (data['substrate'] == '20um fibronectin discs'),
           (data['bead type'] == 'M450'),
           (data['ctFieldThickness'] <= 600),
           (data['date'].apply(lambda x : x in dates))]

co_order = makeOrder(['none','doxycyclin'])
fig, ax = D1Plot(data, CondCols=['drug'],Parameters=['ctFieldThickness','EChadwick'],Filters=Filters,                 AvgPerCell=True, cellID='cellID', co_order=co_order, figSizeFactor = 1.0,
                 useHue = False, orientation = 'h', markersizeFactor=1.2)

ax[0].set_ylim([0, ax[0].get_ylim()[1]])
renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL on fibronectin discs: Compressions')
# ufun.archiveFig(fig, ax, name='3T3aSFL_substrate&drug_SurroundingThickness&EChadwick_NEWTABLE', figSubDir = figSubDir)
plt.show()


# %%%%%


data = GlobalTable_ctField # Tracking matlab, postprocessing python
dates = []

Filters = [(data['validated'] == True), (data['medianThickness'] <= 1000)]

co_order = makeOrder(['20um fibronectin discs'],['none','doxycyclin'])
fig, ax = D1Plot(data, CondCols=['substrate','drug'],Parameters=['medianThickness','fluctuAmpli'],                 Filters=Filters,stats=True,co_order=co_order,figSizeFactor=0.5)

ax[0].set_ylim([0,600])
ax[1].set_ylim([0,600])
renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL on patterns: Constant Field')
# ufun.archiveFig(fig, ax, name='3T3aSFL_drug_medianThickness', figSubDir = figSubDir)

plt.show()


# %%%%%


data = GlobalTable_meca # 100%  matlab
dates = ['21-01-18', '21-01-21']

Filters = [(data['Validated'] == True), 
           (data['date'].apply(lambda x : x in dates))]

# def D2Plot_wFit(data, XCol='', YCol='', CondCol='', Filters=[], cellID='cellID', 
#                 AvgPerCell=False, showManips = True,
#                 modelFit=False, modelType='y=ax+b',
#                 xscale = 'lin', yscale = 'lin', 
#                 figSizeFactor = 1):

fig, ax = D2Plot_wFit(data, XCol='meanFluoPeakAmplitude', YCol='EChadwick', CondCol = ['drug'], 
                      Filters=Filters, cellID = 'CellName', AvgPerCell=True, modelFit=False, 
                      modelType = 'y=ax+b')

renameAxes(ax,renameDict1)
fig.suptitle('aSFL expressing linker: E(fluo)')
ax.set_ylim([0,80000])
ax.set_xlim([-10,1000])
yt = ax.get_yticks()
ax.set_yticklabels((yt/1000).astype(int))
ax.set_ylabel('E Chadwick (kPa)')

# ufun.archiveFig(fig, ax, name='aSFL_iMC_E(fluo)', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='FluoPlots')
plt.show()


# %%%%%


data = GlobalTable_meca # 100%  matlab
dates = ['21-01-18', '21-01-21']

#Same as above without the lonely point
Filters = [(data['Validated'] == True), 
           (data['date'].apply(lambda x : x in dates)), 
           (data['EChadwick'] <= 80000)]

fig, ax = D2Plot_wFit(data, XCol='meanFluoPeakAmplitude', YCol='EChadwick', 
                      CondCol = ['cell subtype', 'drug'], Filters=Filters, 
                      cellID = 'CellName', AvgPerCell=True, modelFit=True)

renameAxes(ax,renameDict1)
fig.suptitle('aSFL expressing linker: E(fluo)')

# ufun.archiveFig(fig, ax, name='aSFL_iMC_E(fluo)_woLastPoint', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='FluoPlots')
plt.show()


# %%%%%


Filters = [(GlobalTable_meca['Validated'] == True), (GlobalTable_meca['cell subtype'] == 'aSFL')]
fig, ax = D2Plot(GlobalTable_meca, XCol='meanFluoPeakAmplitude',YCol='SurroundingThickness',
                 CondCol = ['cell subtype', 'drug'], Filters=Filters, cellID = 'CellName',AvgPerCell=True)

renameAxes(ax,renameDict1)
fig.suptitle('aSFL expressing linker: H(fluo)')

# ufun.archiveFig(fig, ax, name='aSFL_iMC_H(fluo)', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='FluoPlots')
plt.show()


# %%%%%


Filters = [(GlobalTable_ctField['validated'] == True)]
fig, ax = D2Plot(GlobalTable_ctField, XCol='meanFluoPeakAmplitude',YCol='medianThickness',
                 CondCol = ['cell subtype', 'drug'],Filters=Filters, cellID = 'cellID', AvgPerCell=True)

renameAxes(ax,renameDict1)
fig.suptitle('aSFL expressing linker: medianH(fluo)')

# ufun.archiveFig(fig, ax, name='aSFL_iMC_medianH(fluo)', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='FluoPlots')
plt.show()


# %%%% 3T3 aSFL : standard line (A11) versus different clones (A8 - high exp ; 6FP - long linker) -- First round (april - june 2021)

# %%%%%


Filters = [(GlobalTable_meca_Py['validatedThickness'] == True), (GlobalTable_meca_Py['ctFieldThickness'] <= 1000)]
co_order = makeOrder(['aSFL','aSFL-A8','aSFL-6FP'],['none','doxycyclin'])
box_pairs=[('aSFL & none', 'aSFL & doxycyclin'),
 ('aSFL-A8 & none', 'aSFL-A8 & doxycyclin'),
 ('aSFL-6FP & none', 'aSFL-6FP & doxycyclin'),
 ('aSFL & none', 'aSFL-A8 & none'),
 ('aSFL & none', 'aSFL-6FP & none')]
fig, ax = D1Plot(GlobalTable_meca_Py, CondCols=['cell subtype','drug'],Parameters=['ctFieldThickness','ctFieldFluctuAmpli'],                 Filters=Filters,AvgPerCell=True,stats=True,co_order=co_order,box_pairs=box_pairs,figSizeFactor=1)
renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL on patterns: H and DH from meca expe')
# ufun.archiveFig(fig, ax, name='3T3aSFL_drug_medianThickness_fromMeca_PYTHONTABLE', figSubDir = figSubDir)

plt.show()


# %%%%%


Filters = [(GlobalTable_meca['Validated'] == 1), (GlobalTable_meca['substrate'] == '20um fibronectin discs')]
co_order = makeOrder(['aSFL','aSFL-6FP'],['none','doxycyclin'])
fig, ax = D1Plot(GlobalTable_meca, CondCols=['cell subtype','drug'],Parameters=['SurroundingThickness','EChadwick'],                 Filters=Filters,AvgPerCell=True,cellID='CellName',co_order=co_order,stats=True)
renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL short vs long linker: Compressions')
# ufun.archiveFig(fig, ax, name='3T3aSFL_likerType&drug_SurroundingThickness&EChadwick', figSubDir = figSubDir)
plt.show()


# %%%%%


Filters = [(GlobalTable_meca['Validated'] == True), (GlobalTable_meca['substrate'] == '20um fibronectin discs')]
co_order = makeOrder(['aSFL','aSFL-A8','aSFL-6FP'],['none','doxycyclin'])
box_pairs=[('aSFL & none', 'aSFL & doxycyclin'),
 ('aSFL-A8 & none', 'aSFL-A8 & doxycyclin'),
 ('aSFL-6FP & none', 'aSFL-6FP & doxycyclin'),
 ('aSFL & none', 'aSFL-A8 & none'),
 ('aSFL & none', 'aSFL-6FP & none')]
fig, ax = D1Plot(GlobalTable_meca, CondCols=['cell subtype','drug'],Parameters=['SurroundingThickness','EChadwick'],                 Filters=Filters,AvgPerCell=True,cellID='CellName',co_order=co_order,box_pairs=box_pairs,stats=True)
renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL - All linker types: Compressions')
# ufun.archiveFig(fig, ax, name='3T3aSFL_likerType&drug_SurroundingThickness&EChadwick', figSubDir = figSubDir)
plt.show()


# %%%%%


Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['validatedThickness'] == True), (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs')]
co_order = makeOrder(['aSFL','aSFL-A8','aSFL-6FP'],['none','doxycyclin'])
box_pairs=[('aSFL & none', 'aSFL & doxycyclin'),
 ('aSFL-A8 & none', 'aSFL-A8 & doxycyclin'),
 ('aSFL-6FP & none', 'aSFL-6FP & doxycyclin'),
 ('aSFL & none', 'aSFL-A8 & none'),
 ('aSFL & none', 'aSFL-6FP & none')]
fig, ax = D1Plot(GlobalTable_meca_Py, CondCols=['cell subtype','drug'],Parameters=['ctFieldThickness','EChadwick'],                 Filters=Filters,AvgPerCell=True,cellID='cellID',co_order=co_order,box_pairs=box_pairs,stats=True)
renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL - All linker types: Compressions')
# ufun.archiveFig(fig, ax, name='3T3aSFL_likerType&drug_ctFThickness&EChadwick_PYTHONTABLE', figSubDir = figSubDir)
plt.show()


# %%%%%


Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['validatedThickness'] == True),           (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs'), (GlobalTable_meca_Py['drug'] == 'doxycyclin'),           (pd.isna(GlobalTable_meca_Py['meanFluoPeakAmplitude']) != True)]
co_order = makeOrder(['aSFL','aSFL-A8','aSFL-6FP'])
fig, ax = D1Plot(GlobalTable_meca_Py, CondCols=['cell subtype'],Parameters=['meanFluoPeakAmplitude'],                 Filters=Filters,AvgPerCell=True,cellID='cellID',co_order=co_order,stats=True,figSizeFactor=1.25)
renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL different cell lines\nLinker expression quantif by fluo')
ufun.archiveFig(fig, ax, name='3T3aSFL_likerType&_fluoExp_PYTHONTABLE', figSubDir = figSubDir)
plt.show()


# %%%%%


Filters = [(GlobalTable_meca_Py['validatedFit'] == True), ((GlobalTable_meca_Py['cell subtype'] == 'aSFL') | (GlobalTable_meca_Py['cell subtype'] == 'aSFL-A8')), (GlobalTable_meca_Py['drug'] == 'doxycyclin')]
fig, ax = D2Plot(GlobalTable_meca_Py, XCol='meanFluoPeakAmplitude', YCol='EChadwick', CondCol = ['cell subtype'],                  Filters=Filters, cellID = 'cellID', AvgPerCell=True, modelFit=True)
renameAxes(ax,renameDict1)
fig.suptitle('aSFL & aSFL-A8 expressing linker: E(fluo)')
# ufun.archiveFig(fig, ax, name='aSFL&A8_iMC_E(fluo)', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='FluoPlots')
plt.show()


# %%%%%


#Same as above without the lonely point
Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['drug'] == 'doxycyclin'),            ((GlobalTable_meca_Py['cell subtype'] == 'aSFL') | (GlobalTable_meca_Py['cell subtype'] == 'aSFL-A8')),            (GlobalTable_meca_Py['meanFluoPeakAmplitude'] <= 1200), (GlobalTable_meca_Py['EChadwick'] >= 2000)]
fig, ax = D2Plot(GlobalTable_meca_Py, XCol='meanFluoPeakAmplitude', YCol='EChadwick', CondCol = ['cell subtype'],                  Filters=Filters, cellID = 'cellID', AvgPerCell=True, modelFit=True)
renameAxes(ax,renameDict1)
fig.suptitle('aSFL & aSFL-A8 expressing linker: E(fluo)')
# ufun.archiveFig(fig, ax, name='aSFL&A8_iMC_E(fluo)_fluo-1200_&_E+2000', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='FluoPlots')
plt.show()


# %%%%%


Filters = [(GlobalTable_meca['Validated'] == True), (GlobalTable_meca['cell subtype'] == 'aSFL-6FP')]
fig, ax = D2Plot(GlobalTable_meca, XCol='meanFluoPeakAmplitude', YCol='EChadwick', CondCol = ['cell subtype', 'drug'],                  Filters=Filters, cellID = 'CellName', AvgPerCell=True, modelFit=True)
renameAxes(ax,renameDict1)
fig.suptitle('aSFL-6FP expressing long linker: E(fluo)')
# ufun.archiveFig(fig, ax, name='aSFL-6FP_iMC_E(fluo)', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='FluoPlots')
plt.show()


# %%%%%


#Same as above without the lonely point
Filters = [(GlobalTable_meca['Validated'] == True), (GlobalTable_meca['cell subtype'] == 'aSFL-6FP'), (GlobalTable_meca['meanFluoPeakAmplitude'] <= 1000)]
fig, ax = D2Plot(GlobalTable_meca, XCol='meanFluoPeakAmplitude', YCol='EChadwick', CondCol = ['cell subtype', 'drug'],                  Filters=Filters, cellID = 'CellName', AvgPerCell=True, modelFit=True)
renameAxes(ax,renameDict1)
fig.suptitle('aSFL-6FP expressing long linker: E(fluo)')
# ufun.archiveFig(fig, ax, name='aSFL-6FP_iMC_E(fluo)_woLastPoint', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='FluoPlots')
plt.show()


# %%%%%


Filters = [(GlobalTable_meca['Validated'] == True), (GlobalTable_meca['cell subtype'] == 'aSFL-6FP')]
fig, ax = D2Plot(GlobalTable_meca, XCol='meanFluoPeakAmplitude', YCol='SurroundingThickness', CondCol = ['cell subtype', 'drug'],                 Filters=Filters, cellID = 'CellName',AvgPerCell=True)
renameAxes(ax,renameDict1)
fig.suptitle('aSFL-6FP expressing long linker: H(fluo)')
# ufun.archiveFig(fig, ax, name='aSFL-6FP_iMC_H(fluo)', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='FluoPlots')
plt.show()


# %%%%  Diverse 2D plots

# %%%%%


Filters = [(GlobalTable_ctField['validated'] == True), (GlobalTable_ctField['cell subtype'] == 'aSFL')]
fig, ax = D2Plot(GlobalTable_ctField, XCol='medianThickness',YCol='fluctuAmpli',CondCol = ['cell subtype', 'drug'],                 Filters=Filters, modelFit=True, figSizeFactor = 0.8)
fig.suptitle('3T3aSFL - Dh = f(H)')
# ufun.archiveFig(fig, ax, name='aSFL_Dh(h)_drug', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='ThicknessPlots')
plt.show()


# %%%%%


# Same as above without the 2 thickest cells
Filters = [(GlobalTable_ctField['validated'] == True), (GlobalTable_ctField['cell subtype'] == 'aSFL'),           (GlobalTable_ctField['medianThickness'] <= 700)]
fig, ax = D2Plot(GlobalTable_ctField, XCol='medianThickness',YCol='fluctuAmpli',CondCol = ['cell subtype', 'drug'],                 Filters=Filters, modelFit=True, figSizeFactor = 0.8)
fig.suptitle('3T3aSFL - Dh = f(H)')
# ufun.archiveFig(fig, ax, name='aSFL_Dh(h)_drug_wo2LastPoints', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='ThicknessPlots')
plt.show()


# %%%%%


Filters = [(GlobalTable_meca['Validated'] == True), (GlobalTable_meca['cell subtype'] == 'aSFL')]
fig, ax = D2Plot(GlobalTable_meca, XCol='SurroundingThickness',YCol='EChadwick',CondCol = ['substrate','drug'],           Filters=Filters, cellID = 'CellName', AvgPerCell=True, modelFit=False, modelType='y=A*exp(kx)')
fig.suptitle('3T3aSFL: E(h)')
# ufun.archiveFig(fig, ax, name='aSFL_E(h)_drug&substrate', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='')
plt.show()





# %%%% Smifh2 vs Dmso -- (Sergio's visit september 2021)

# %%%%%


pd.set_option('max_columns', None)
# pd.reset_option('max_columns')
pd.set_option('max_rows', None)
# pd.reset_option('max_rows')


# %%%%%


Outliers = ['21-09-02_M4_P1_C3', '21-09-02_M3_P1_C2', '21-09-02_M3_P1_C9']
Filters = [(GlobalTable_meca_Py['drug'].apply(lambda x : x in ['dmso', 'smifh2', 'dmso. doxycyclin', 'smifh2. doxycyclin'])),            (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs'), (GlobalTable_meca_Py['cell subtype'] == 'aSFL'),            (GlobalTable_meca_Py['cellID'].apply(lambda x : x not in Outliers))]
data_filtered = GlobalTable_meca_Py
for fltr in Filters:
    data_filtered = data_filtered.loc[fltr]
data_filtered


# %%%%%


Outliers = ['21-09-02_M4_P1_C3', '21-09-02_M3_P1_C2', '21-09-02_M3_P1_C9']
Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['validatedThickness'] == True),            (GlobalTable_meca_Py['drug'].apply(lambda x : x in ['dmso', 'smifh2', 'dmso. doxycyclin', 'smifh2. doxycyclin'])),            (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs'), (GlobalTable_meca_Py['cell subtype'] == 'aSFL'),            (GlobalTable_meca_Py['cellID'].apply(lambda x : x not in Outliers))]
co_order = makeOrder(['dmso', 'smifh2', 'dmso. doxycyclin', 'smifh2. doxycyclin'])
fig, ax = D1Plot(GlobalTable_meca_Py, CondCols=['drug'],                 Parameters=['ctFieldThickness','EChadwick'], Filters=Filters,                 AvgPerCell=True, cellID='cellID', co_order=co_order, figSizeFactor = 1.8, orientation = 'v')
renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL smifh2')
ufun.archiveFig(fig, ax, name='3T3aSFL_smifh2&doxy_Thickness&EChadwick', figSubDir = figSubDir)
plt.show()
# 'ctFieldThickness', 'surroundingThickness', 'none', 'doxycyclin', 


# %%%% Long linker second clone (3T3 aSFL-6FP-2) -- (Sergio's visit september 2021)

# %%%%%


pd.set_option('max_columns', None)
pd.reset_option('max_columns')
pd.set_option('max_rows', None)
pd.reset_option('max_rows')


# %%%%%


# Outliers = []
# Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['validatedThickness'] == True), \
#            (GlobalTable_meca_Py['drug'].apply(lambda x : x in ['none', 'doxycyclin'])), \
#            (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs'), (GlobalTable_meca_Py['cell subtype'].apply(lambda x : x in ['aSFL', 'aSFL-6FP', 'aSFL-6FP-2'])), \
#            (GlobalTable_meca_Py['cellID'].apply(lambda x : x not in Outliers))]
# co_order = makeOrder(['aSFL', 'aSFL-6FP', 'aSFL-6FP-2'], ['none', 'doxycyclin'])
# box_pairs=[('aSFL & none', 'aSFL & doxycyclin'),
#  ('aSFL-6FP & none', 'aSFL-6FP & doxycyclin'),
#  ('aSFL-6FP-2 & none', 'aSFL-6FP-2 & doxycyclin'),
#  ('aSFL & none', 'aSFL-6FP & none'),
#  ('aSFL & none', 'aSFL-6FP-2 & none'),
#  ('aSFL-6FP & none', 'aSFL-6FP-2 & none')]
# fig, ax = D1Plot(GlobalTable_meca_Py, CondCols=['cell subtype', 'drug'],\
#                  Parameters=['ctFieldThickness','EChadwick'], Filters=Filters,\
#                  AvgPerCell=True, cellID='cellID', co_order=co_order, box_pairs = box_pairs, figSizeFactor = 1, orientation = 'v')
# ax[0].set_ylim([0, ax[0].get_ylim()[1]])
# renameAxes(ax,renameDict1)
# fig.suptitle('3T3aSFL 6FP-2')
# ufun.archiveFig(fig, ax, name='3T3aSFL-6FP-2_Thickness&EChadwick', figSubDir = figSubDir)
# plt.show()
# # 'ctFieldThickness', 'surroundingThickness', 'none', 'doxycyclin', 


# %%%%%


Outliers = []
Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['validatedThickness'] == True),            (GlobalTable_meca_Py['drug'].apply(lambda x : x in ['none', 'doxycyclin'])),            (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs'), (GlobalTable_meca_Py['cell subtype'].apply(lambda x : x in ['aSFL', 'aSFL-6FP-2'])),            (GlobalTable_meca_Py['cellID'].apply(lambda x : x not in Outliers))]
co_order = makeOrder(['aSFL', 'aSFL-6FP-2'], ['none', 'doxycyclin'])
box_pairs=[('aSFL & none', 'aSFL & doxycyclin'),
 ('aSFL-6FP-2 & none', 'aSFL-6FP-2 & doxycyclin'),
 ('aSFL & none', 'aSFL-6FP-2 & none')]
fig, ax = D1Plot(GlobalTable_meca_Py, CondCols=['cell subtype', 'drug'],                 Parameters=['ctFieldThickness','EChadwick'], Filters=Filters,                 AvgPerCell=True, cellID='cellID', co_order=co_order, box_pairs = box_pairs, figSizeFactor = 1, orientation = 'v')
ax[0].set_ylim([0, ax[0].get_ylim()[1]])
renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL 6FP-2')
ufun.archiveFig(fig, ax, name='3T3aSFL-6FP-2_Thickness&EChadwick', figSubDir = figSubDir)
plt.show()
# 'ctFieldThickness', 'surroundingThickness', 'none', 'doxycyclin', 


# %%%%%


Outliers = []
Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['validatedThickness'] == True),            (GlobalTable_meca_Py['drug'].apply(lambda x : x in ['none', 'doxycyclin'])),            (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs'), (GlobalTable_meca_Py['cell subtype'].apply(lambda x : x in ['aSFL-6FP-2'])),            (GlobalTable_meca_Py['cellID'].apply(lambda x : x not in Outliers))]
fig, ax = D2Plot(GlobalTable_meca_Py, XCol='meanFluoPeakAmplitude', YCol='EChadwick', CondCol = ['cell subtype', 'drug'],                  Filters=Filters, cellID = 'cellID', AvgPerCell=True, modelFit=True)

renameAxes(ax,renameDict1)
fig.suptitle('aSFL-6FP-2 expressing linker: E(fluo)')
yt = ax.get_yticks()
ax.set_yticklabels((yt/1000).astype(int))
ax.set_ylabel('E Chadwick (kPa)')

# ufun.archiveFig(fig, ax, name='aSFL_iMC_E(fluo)', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='FluoPlots')
plt.show()


# %%%%%


Outliers = []
Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['validatedThickness'] == True),            (GlobalTable_meca_Py['drug'].apply(lambda x : x in ['none', 'doxycyclin'])),            (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs'), 
           (GlobalTable_meca_Py['cell subtype'].apply(lambda x : x in ['aSFL-6FP'])), \
           (GlobalTable_meca_Py['cellID'].apply(lambda x : x not in Outliers))]
fig, ax = D2Plot(GlobalTable_meca_Py, XCol='meanFluoPeakAmplitude', YCol='EChadwick', CondCol = ['cell subtype', 'drug'],                  Filters=Filters, cellID = 'cellID', AvgPerCell=True, modelFit=True)

renameAxes(ax,renameDict1)
fig.suptitle('aSFL-6FP expressing linker: E(fluo)')
yt = ax.get_yticks()
ax.set_yticklabels((yt/1000).astype(int))
ax.set_ylabel('E Chadwick (kPa)')

# ufun.archiveFig(fig, ax, name='aSFL_iMC_E(fluo)', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='FluoPlots')
plt.show()


# %%%%%


Outliers = []
Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['validatedThickness'] == True),            (GlobalTable_meca_Py['drug'].apply(lambda x : x in ['none', 'doxycyclin'])),            (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs'), 
           (GlobalTable_meca_Py['cell subtype'].apply(lambda x : x in ['aSFL'])), \
           (GlobalTable_meca_Py['cellID'].apply(lambda x : x not in Outliers))]
fig, ax = D2Plot(GlobalTable_meca_Py, XCol='meanFluoPeakAmplitude', YCol='EChadwick', CondCol = ['cell subtype', 'drug'],                  Filters=Filters, cellID = 'cellID', AvgPerCell=True, modelFit=True)

renameAxes(ax,renameDict1)
fig.suptitle('aSFL expressing linker: E(fluo)')
yt = ax.get_yticks()
ax.set_yticklabels((yt/1000).astype(int))
ax.set_ylabel('E Chadwick (kPa)')

# ufun.archiveFig(fig, ax, name='aSFL_iMC_E(fluo)', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='FluoPlots')
plt.show()


# %%%% High expresser experiment, second version of the clone (aSFL A8-2) -- (Sergio's visit september 2021)

# %%%%%


# Outliers = []
# Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['validatedThickness'] == True),
#            (GlobalTable_meca_Py['drug'].apply(lambda x : x in ['none', 'doxycyclin'])),
#            (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs'), 
#            (GlobalTable_meca_Py['cell subtype'].apply(lambda x : x in ['aSFL', 'aSFL-A8', 'aSFL-A8-2'])),
#            (GlobalTable_meca_Py['cellID'].apply(lambda x : x not in Outliers))]
# co_order = makeOrder(['aSFL', 'aSFL-A8', 'aSFL-A8-2'], ['none', 'doxycyclin'])
# box_pairs=[('aSFL & none', 'aSFL & doxycyclin'),
#  ('aSFL-A8 & none', 'aSFL-A8 & doxycyclin'),
#  ('aSFL-A8-2 & none', 'aSFL-A8-2 & doxycyclin'),
#  ('aSFL & none', 'aSFL-A8 & none'),
#  ('aSFL & none', 'aSFL-A8-2 & none'),
#  ('aSFL-A8 & none', 'aSFL-A8-2 & none')]
# fig, ax = D1Plot(GlobalTable_meca_Py, CondCols=['cell subtype', 'drug'],\
#                  Parameters=['surroundingThickness','EChadwick'], Filters=Filters,\
#                  AvgPerCell=True, cellID='cellID', co_order=co_order, box_pairs = box_pairs, 
#                  figSizeFactor = 1, orientation = 'v', useHue = True)
# ax[0].set_ylim([0, ax[0].get_ylim()[1]])
# renameAxes(ax,renameDict1)
# fig.suptitle('3T3aSFL A8-2')
# # ufun.archiveFig(fig, ax, name='3T3aSFL-A8-2_Thickness&EChadwick', figSubDir = figSubDir)
# plt.show()
# # 'ctFieldThickness', 'surroundingThickness', 'none', 'doxycyclin', 


# %%%%%


Outliers = []
Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['validatedThickness'] == True),
           (GlobalTable_meca_Py['drug'].apply(lambda x : x in ['none', 'doxycyclin'])),
           (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs'), 
           (GlobalTable_meca_Py['cell subtype'].apply(lambda x : x in ['aSFL', 'aSFL-A8-2'])),
           (GlobalTable_meca_Py['cellID'].apply(lambda x : x not in Outliers))]
co_order = makeOrder(['aSFL', 'aSFL-A8-2'], ['none', 'doxycyclin'])
box_pairs=[('aSFL & none', 'aSFL & doxycyclin'),
 ('aSFL-A8-2 & none', 'aSFL-A8-2 & doxycyclin'),
 ('aSFL & none', 'aSFL-A8-2 & none')]
fig, ax = D1Plot(GlobalTable_meca_Py, CondCols=['cell subtype', 'drug'],                 Parameters=['surroundingThickness','EChadwick'], Filters=Filters,                 AvgPerCell=True, cellID='cellID', co_order=co_order, box_pairs = box_pairs, 
                 figSizeFactor = 1, orientation = 'v', useHue = False)
ax[0].set_ylim([0, ax[0].get_ylim()[1]])
renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL A8-2')
ufun.archiveFig(fig, ax, name='3T3aSFL-A8-2_Thickness&EChadwick', figSubDir = figSubDir)
plt.show()
# 'ctFieldThickness', 'surroundingThickness', 'none', 'doxycyclin', 


# %%%%%


Outliers = []
Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['validatedThickness'] == True),            (GlobalTable_meca_Py['drug'].apply(lambda x : x in ['none', 'doxycyclin'])),            (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs'), 
           (GlobalTable_meca_Py['cell subtype'].apply(lambda x : x in ['aSFL-A8-2'])), \
           (GlobalTable_meca_Py['cellID'].apply(lambda x : x not in Outliers))]
fig, ax = D2Plot(GlobalTable_meca_Py, XCol='meanFluoPeakAmplitude', YCol='EChadwick', CondCol = ['cell subtype', 'drug'],                  Filters=Filters, cellID = 'cellID', AvgPerCell=True, modelFit=True)

renameAxes(ax,renameDict1)
fig.suptitle('aSFL-A8-2 expressing linker: E(fluo)')
yt = ax.get_yticks()
ax.set_yticklabels((yt/1000).astype(int))
ax.set_ylabel('E Chadwick (kPa)')

ufun.archiveFig(fig, ax, name='aSFL-A8-2_iMC_E(fluo)', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='FluoPlots')
plt.show()


# %%%%%


Outliers = []
Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['validatedThickness'] == True),            (GlobalTable_meca_Py['drug'].apply(lambda x : x in ['none', 'doxycyclin'])),            (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs'), 
           (GlobalTable_meca_Py['cell subtype'].apply(lambda x : x in ['aSFL'])), \
           (GlobalTable_meca_Py['cellID'].apply(lambda x : x not in Outliers))]
fig, ax = D2Plot(GlobalTable_meca_Py, XCol='meanFluoPeakAmplitude', YCol='EChadwick', CondCol = ['cell subtype', 'drug'],                  Filters=Filters, cellID = 'cellID', AvgPerCell=True, modelFit=True)

renameAxes(ax,renameDict1)
fig.suptitle('aSFL expressing linker: E(fluo)')
yt = ax.get_yticks()
ax.set_yticklabels((yt/1000).astype(int))
ax.set_ylabel('E Chadwick (kPa)')

# ufun.archiveFig(fig, ax, name='aSFL_iMC_E(fluo)', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='FluoPlots')
plt.show()


# %%%%%


Outliers = []
Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['validatedThickness'] == True),            (GlobalTable_meca_Py['drug'].apply(lambda x : x in ['none', 'doxycyclin'])),            (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs'), 
           (GlobalTable_meca_Py['cell subtype'].apply(lambda x : x in ['aSFL', 'aSFL-A8-2'])), \
           (GlobalTable_meca_Py['cellID'].apply(lambda x : x not in Outliers))]
fig, ax = D2Plot(GlobalTable_meca_Py, XCol='meanFluoPeakAmplitude', YCol='EChadwick', CondCol = ['drug'],                  Filters=Filters, cellID = 'cellID', AvgPerCell=True, modelFit=True)

renameAxes(ax,renameDict1)
fig.suptitle('{aSFL & aSFL-A8-2 mixed} expressing linker: E(fluo)')
yt = ax.get_yticks()
ax.set_yticklabels((yt/1000).astype(int))
ax.set_ylabel('E Chadwick (kPa)')

# ufun.archiveFig(fig, ax, name='aSFL_A11+A8-2_iMC_E(fluo)', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir='FluoPlots')
plt.show()


# %%%%%


Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['validatedThickness'] == True),           (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs'), 
           (GlobalTable_meca_Py['drug'].apply(lambda x : x in ['doxycyclin'])), \
           (pd.isna(GlobalTable_meca_Py['meanFluoPeakAmplitude']) != True),
          (GlobalTable_meca_Py['cell subtype'].apply(lambda x : x in ['aSFL','aSFL-6FP','aSFL-A8','aSFL-6FP-2','aSFL-A8-2'])),]
co_order = makeOrder(['aSFL','aSFL-A8','aSFL-6FP','aSFL-A8-2','aSFL-6FP-2'])
box_pairs=[('aSFL', 'aSFL-A8'),
 ('aSFL', 'aSFL-6FP'),
 ('aSFL', 'aSFL-A8-2'),
 ('aSFL', 'aSFL-6FP-2')]
fig, ax = D1Plot(GlobalTable_meca_Py, CondCols=['cell subtype'],Parameters=['meanFluoPeakAmplitude'],                 Filters=Filters,AvgPerCell=True,cellID='cellID',co_order=co_order,box_pairs=box_pairs,stats=True,figSizeFactor=1.75)
renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL - different cell lines expressing the linker\nLinker expression quantification by fluo')
ufun.archiveFig(fig, ax, name='3T3aSFL_likerType&_fluoExp_NewVersion', figSubDir = figSubDir)
plt.show()


# %%%%%


Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['validatedThickness'] == True),           (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs'), 
           (GlobalTable_meca_Py['drug'].apply(lambda x : x in ['none', 'doxycyclin'])), \
           (pd.isna(GlobalTable_meca_Py['meanFluoPeakAmplitude']) != True),
          (GlobalTable_meca_Py['cell subtype'].apply(lambda x : x in ['aSFL','aSFL-6FP','aSFL-A8','aSFL-6FP-2','aSFL-A8-2'])),]
co_order = makeOrder(['aSFL','aSFL-A8','aSFL-6FP','aSFL-A8-2','aSFL-6FP-2'], ['none', 'doxycyclin'])
fig, ax = D1Plot(GlobalTable_meca_Py, CondCols=['cell subtype', 'drug'],Parameters=['meanFluoPeakAmplitude'],                 Filters=Filters,AvgPerCell=True,cellID='cellID',co_order=co_order,stats=False,figSizeFactor=1.05)
renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL different cell lines\nLinker expression quantif by fluo')
# ufun.archiveFig(fig, ax, name='3T3aSFL_likerType&_fluoExp_NewVersion_2', figSubDir = figSubDir)
plt.show()


# %%%%%


Filters = [(GlobalTable_meca_Py['validatedFit'] == True), (GlobalTable_meca_Py['validatedThickness'] == True),           (GlobalTable_meca_Py['substrate'] == '20um fibronectin discs'), 
           (GlobalTable_meca_Py['drug'].apply(lambda x : x in ['none', 'doxycyclin'])), \
           (pd.isna(GlobalTable_meca_Py['meanFluoPeakAmplitude']) != True),
          (GlobalTable_meca_Py['cell subtype'].apply(lambda x : x in ['aSFL','aSFL-6FP-2','aSFL-A8-2'])),]
co_order = makeOrder(['aSFL','aSFL-A8-2','aSFL-6FP-2'], ['none', 'doxycyclin'])
box_pairs=[('aSFL & doxycyclin', 'aSFL-A8-2 & doxycyclin'),
 ('aSFL & doxycyclin', 'aSFL-6FP-2 & doxycyclin')]
fig, ax = D1Plot(GlobalTable_meca_Py, CondCols=['cell subtype', 'drug'],Parameters=['meanFluoPeakAmplitude'],                 Filters=Filters,AvgPerCell=True,cellID='cellID',co_order=co_order,box_pairs=box_pairs,stats=True,
                 figSizeFactor=0.95)
renameAxes(ax,renameDict1)
fig.suptitle('3T3aSFL - different cell lines\nLinker expression quantification by fluo')
ufun.archiveFig(fig, ax, name='3T3aSFL_likerType&_fluoExp_NewVersion_2', figSubDir = figSubDir)
plt.show()


# %%% MCA project - Py2 > Full python processing

plt.close('all')
figSubDir = 'MCAproject'

# %%%% Entire table (meca_Py2)

# %%%%% Test of dataframe sorting

condition = 'drug'
co_order = ['none', 'doxycyclin']
order_dict = {}
for i in range(len(co_order)):
    order_dict[co_order[i]] = i+1

data = GlobalTable_meca_Py2

Filters = [(data['validatedFit'] == True), (data['validatedThickness'] == True),
          (data['date'].apply(lambda x : x in ['21-01-18', '21-01-21']))]

data_f = data
for F in Filters:
    data_f = data_f[F]

data_f_sorted = data_f.sort_values(by=[condition], key=lambda x: x.map(order_dict))
df0 = data_f_sorted


# %%%%% 3T3 aSFL - Jan 21 - Compression experiments


data = GlobalTable_meca_Py2

Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['date'].apply(lambda x : x in ['21-01-18', '21-01-21']))]

fig, ax = D1Plot(data, CondCols=['drug'], Parameters=['surroundingThickness', 'EChadwick'], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=['none', 'doxycyclin'], stats=True, statMethod='Mann-Whitney', 
               box_pairs=[], figSizeFactor = 0.9, markersizeFactor=1, orientation = 'h', AvgPerCell = True)
renameAxes(ax,renameDict1)
ax[0].set_ylim([0, 850])
# ax[0].legend(loc = 'upper right', fontsize = 8)
# ax[1].legend(loc = 'upper right', fontsize = 8)
fig.suptitle('3T3 aSFL - Jan 21 - Compression experiments')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_H&Echad_simple', figSubDir = figSubDir)
plt.show()


# %%%%% 3T3 aSFL - Jan 21 - Detailed compression experiments


data = GlobalTable_meca_Py2

Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['date'].apply(lambda x : x in ['21-01-18', '21-01-21']))]

fig, ax = D1PlotDetailed(data, CondCols=['drug'], Parameters=['bestH0', 'EChadwick'], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order = ['none', 'doxycyclin'], stats=True, statMethod='Mann-Whitney', 
                box_pairs=[], figSizeFactor = 1.8, markersizeFactor=1, orientation = 'v', showManips = True)

ax[0].legend(loc = 'upper right', fontsize = 8)
ax[1].legend(loc = 'upper right', fontsize = 8)
ax[0].set_ylim([0, 850])
ax[1].set_ylim([1, 2e5])
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_H&Echad_detailed', figSubDir = figSubDir)
plt.show()


# %%%%% 3T3aSFL on patterns: Constant Field


data = GlobalTable_ctField_Py # Tracking python, postprocessing python
dates = []

Filters = [(data['validated'] == True), (data['medianThickness'] <= 1000)]

co_order = makeOrder(['none','doxycyclin'])
fig, ax = D1Plot(data, CondCols=['drug'],Parameters=['medianThickness','fluctuAmpli'],
                 Filters=Filters,stats=True,co_order=co_order,figSizeFactor=0.9)

renameAxes(ax,renameDict1)
ax[0].set_ylim([0,600])
ax[1].set_ylim([0,600])
fig.suptitle('3T3aSFL on patterns: Constant Field')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_medianThickness', figSubDir = figSubDir)

plt.show()


# %%%%% Nonlinearity, first check

data = GlobalTable_meca_Py2

columns = data.columns.values
minS = []
maxS = []
centerS = []
colList = []

for c in columns:
    if 'KChadwick' in c and '<s<' in c:
        valS = c.split('_')[-1].split('<s<')
        valS[0], valS[1] = int(valS[0]), int(valS[1])
        minS.append(valS[0])
        maxS.append(valS[1])
        centerS.append(int(0.5*(valS[0]+valS[1])))
        colList.append(c)
#     elif 'Npts' in c and '<s<' in c:
        
        
Filters = [(data['validatedFit'] == True),
           (data['validatedThickness'] == True),
           (data['date'].apply(lambda x : x in ['21-01-18', '21-01-21']))]

F = Filters[0]
for fltr in Filters[1:]:
    F = F & fltr
    
data_f = data[F]

count = []
for i in range(len(colList)):
    c = colList[i]
    S = centerS[i]
    Vals = data_f[c].values
    countValidFit = len(Vals) - np.sum(np.isnan(Vals))
    count.append(countValidFit)
    
d = {'center_stress' : centerS, 'count_valid' : count}
dfCheck = pd.DataFrame(d)


print(data_f.shape[0])
df0 = dfCheck.T
print(dfCheck.T)


# %%%%% Many tangeantial moduli of aSFL 3T3 - control vs iMC linker

data = GlobalTable_meca_Py2

listeS = [200, 300, 400, 500, 600, 700, 800, 900, 1000]

fig, axes = plt.subplots(1, len(listeS), figsize = (20,6))
kk = 0

for S in listeS:
    interval = str(S-75) + '<s<' + str(S+75)

    Filters = [(data['validatedFit_'+interval] == True),
               (data['validatedThickness'] == True),
               (data['date'].apply(lambda x : x in ['21-09-09']))] #, '21-12-16' , '21-01-21'

    fig, ax = D1Plot(data, fig=fig, ax=axes[kk], CondCols=['drug'], Parameters=['KChadwick_'+interval], Filters=Filters, 
                   Boxplot=True, cellID='cellID', co_order=['none', 'doxycyclin'], stats=True, statMethod='Mann-Whitney', 
                   AvgPerCell = False, box_pairs=[], figSizeFactor = 1, markersizeFactor=0.7, orientation = 'h', 
                   stressBoxPlot=True, bypassLog = True)

#     axes[kk].legend(loc = 'upper right', fontsize = 8)
    label = axes[kk].get_ylabel()
    axes[kk].set_title(label.split('_')[-1] + 'Pa', fontsize = 10)
    # axes[kk].set_yscale('log')
    axes[kk].set_ylim([1e2, 2e4])
    axes[kk].tick_params(axis='x', labelrotation = 50, labelsize = 10)
    if kk == 0:
        axes[kk].set_ylabel(label.split('_')[0] + ' (Pa)')
        axes[kk].tick_params(axis='x', labelsize = 10)
    else:
        axes[kk].set_ylabel(None)
        axes[kk].set_yticklabels([])
    # ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_Detailed1DPlot', figSubDir = figSubDir)
    
    kk+=1

renameAxes(axes,{'none':'ctrl', 'doxycyclin':'iMC'})
fig.tight_layout()
fig.suptitle('Tangeantial modulus of aSFL 3T3 - control vs iMC linker')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_K_multipleIntervals', figSubDir = figSubDir)
plt.show()

# %%%%% Many tangeantial moduli of aSFL 3T3 - control vs iMC linker - Plot by plot


data = GlobalTable_meca_Py2

listeS = [200, 300, 400, 500, 600, 700, 800, 900, 1000]

for S in listeS:
    interval = str(S-75) + '<s<' + str(S+75)
    textForFileName = str(S-75) + '-' + str(S+75) + 'Pa'

    Filters = [(data['validatedFit_'+interval] == True),
               (data['validatedThickness'] == True),
               (data['date'].apply(lambda x : x in ['21-01-18', '21-01-21']))] #, '21-12-16'

    fig, ax = D1Plot(data, CondCols=['drug'], Parameters=['bestH0', 'KChadwick_'+interval], Filters=Filters, 
                   Boxplot=True, cellID='cellID', co_order=['none', 'doxycyclin'], stats=True, statMethod='Mann-Whitney', 
                   box_pairs=[], figSizeFactor = 1.0, markersizeFactor=1, orientation = 'h', AvgPerCell = True)

    # ax[0].legend(loc = 'upper right', fontsize = 6)
    # ax[1].legend(loc = 'lower right', fontsize = 6)
    ax[0].set_ylim([0,1200])
    ax[1].set_ylim([1e2, 3e4])
#     ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_H0-K_'+textForFileName, figSubDir = figSubDir)
    plt.show()
    

# %%%%% Many tangeantial moduli of aSFL 3T3 - control vs iMC linker - detailed plot by plot

data = GlobalTable_meca_Py2

listeS = [200, 300, 400, 500, 600, 700, 800, 900, 1000]

for S in listeS:
    interval = str(S-75) + '<s<' + str(S+75)
    textForFileName = str(S-75) + '-' + str(S+75) + 'Pa'

    Filters = [(data['validatedFit_'+interval] == True),
               (data['validatedThickness'] == True),
               (data['date'].apply(lambda x : x in ['21-01-18', '21-01-21']))] #, '21-12-16'

    fig, ax = D1PlotDetailed(data, CondCols=['drug'], Parameters=['bestH0', 'KChadwick_'+interval], Filters=Filters, 
                   Boxplot=True, cellID='cellID', co_order=['none', 'doxycyclin'], stats=True, statMethod='Mann-Whitney', 
                   box_pairs=[], figSizeFactor = 1.0, markersizeFactor=1, orientation = 'h', showManips = True)

    ax[0].legend(loc = 'upper right', fontsize = 6)
    ax[0].set_ylim([0,1200])
    ax[1].legend(loc = 'lower right', fontsize = 6)
    ax[1].set_ylim([1e2, 3e4])
    # ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_H0-K_'+textForFileName, figSubDir = figSubDir)
    plt.show()
    


# %%%%% aSFL 3T3 - control vs iMC linker - extremal stress


# ['21-12-08', '21-12-16', '22-01-12']
data = GlobalTable_meca_Py2

dates = ['21-01-18', '21-01-21'] # ['21-01-18', '21-01-21', '21-12-08']

filterList = [(data['validatedThickness'] == True),
              (data['cell subtype'] == 'aSFL'), 
              (data['bead type'] == 'M450'),
              (data['substrate'] == '20um fibronectin discs'),
              (data['date'].apply(lambda x : x in dates))]  # (data['validatedFit'] == True), 


fig, ax = plt.subplots(1, 1, figsize = (9, 6), tight_layout=True)

globalFilter = filterList[0]
for k in range(1, len(filterList)):
    globalFilter = globalFilter & filterList[k]

data_f = data[globalFilter]
    
condition = 'drug'
conditionValues = data_f[condition].unique()
conditionFilter = {}
for val in conditionValues:
    conditionFilter[val] = (data_f[condition] == val)
    
rD = {'none' : 'Ctrl', 'doxycyclin' : 'iMC linker'}

for val in conditionValues:
    if val == 'none':
        col1, col2, col3 = 'deepskyblue', 'skyblue', 'royalblue'
    if val == 'doxycyclin':
        col1, col2, col3 = 'orangered', 'lightsalmon', 'firebrick'
    X = data_f[conditionFilter[val]]['bestH0'].values
    Ncomp = len(X)
    Smin = data_f[conditionFilter[val]]['minStress'].values
    Smax = data_f[conditionFilter[val]]['maxStress'].values

    for i in range(Ncomp):
        if i == 0:
            textLabel = rD[val]
        else:
            textLabel = None
        ax.plot([X[i], X[i]], [Smin[i], Smax[i]], 
                ls = '-', color = col1, alpha = 0.3,
                zorder = 1, label = textLabel)
        ax.plot([X[i]], [Smin[i]], 
                marker = 'o', markerfacecolor = col2, markeredgecolor = 'k', markersize = 4,
                ls = '')
        ax.plot([X[i]], [Smax[i]], 
                marker = 'o', markerfacecolor = col3, markeredgecolor = 'k', markersize = 4,
                ls = '')

ax.set_xlabel('Cortical thickness (nm)')
ax.set_xlim([0, 1200])
locator = matplotlib.ticker.MultipleLocator(100)
ax.xaxis.set_major_locator(locator)

ax.set_ylabel('Extremal stress values (Pa)')
ax.set_ylim([3e1, 5e4])
ax.set_yscale('log')
# locator = matplotlib.ticker.MultipleLocator(100)
# ax.yaxis.set_major_locator(locator)
ax.yaxis.grid(True)

ax.legend(loc = 'upper right')

ax.set_title('Jan 21 - Extremal stress values vs. thickness, all compressions, ctrl vs. iMC')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_ctrlVsLinker_StressRanges', figSubDir = figSubDir)
print(Ncomp)
plt.show()


# %%%%% aSFL 3T3 - control vs iMC linker -- E(h) -- plot  by plot

data = GlobalTable_meca_Py2

listeS = [200, 300, 400, 500, 600, 700, 800, 900, 1000]

# fig, axes = plt.subplots(len(listeS), 1, figsize = (9,40))
# kk = 0

for S in listeS:
    interval = str(S-75) + '<s<' + str(S+75)
    textForFileName = str(S-75) + '-' + str(S+75) + 'Pa'

    Filters = [(data['validatedFit_'+interval] == True),
               (data['validatedThickness'] == True),
               (data['date'].apply(lambda x : x in ['21-01-18', '21-01-21']))] #, '21-12-16'

    fig, ax = D2Plot_wFit(data, #fig=fig, ax=axes[kk], 
                          XCol='bestH0', YCol='KChadwick_'+interval, CondCol = ['drug'], Filters=Filters, 
                          cellID = 'cellID', AvgPerCell=False, 
                          xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')
    
    # individual figs
    ax.set_xlim([8e1, 1.2e3])
    ax.set_ylim([3e2, 5e4])
    renameAxes(ax,{'none':'ctrl', 'doxycyclin':'iMC'})
    fig.set_size_inches(6,4)
    fig.tight_layout()
    # ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_ctrlVsLinker_E(h)_' + textForFileName, figSubDir = figSubDir)
    
    
    # one big fig
#     axes[kk].set_xlim([8e1, 1.2e3])
#     axes[kk].set_ylim([3e2, 5e4])
#     kk+=1


# renameAxes(axes,{'none':'ctrl', 'doxycyclin':'iMC'})
# fig.tight_layout()
plt.show()


# %%%%% aSFL 3T3 - control vs iMC linker -- E(h)


data = GlobalTable_meca_Py2

listeS = [200, 300, 400, 500, 600, 700, 800, 900, 1000]

fig, axes = plt.subplots(len(listeS), 1, figsize = (9,40))
kk = 0

for S in listeS:
    interval = str(S-75) + '<s<' + str(S+75)

    Filters = [(data['validatedFit_'+interval] == True),
               (data['validatedThickness'] == True),
               (data['date'].apply(lambda x : x in ['21-01-18', '21-01-21']))] #, '21-12-16'

    fig, ax = D2Plot_wFit(data, fig=fig, ax=axes[kk], 
                          XCol='surroundingThickness', YCol='KChadwick_'+interval, CondCol = ['drug'], Filters=Filters, 
                          cellID = 'cellID', AvgPerCell=True, 
                          xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')
    
    
    
#     axes[kk].legend(loc = 'upper right', fontsize = 8)
#     label = axes[kk].get_ylabel()
#     axes[kk].set_title(label.split('_')[-1] + 'Pa', fontsize = 10)
#     axes[kk].set_yscale('log')
    axes[kk].set_xlim([8e1, 1.2e3])
#     axes[kk].tick_params(axis='x', labelrotation = 50, labelsize = 10)
#     if kk == 0:
#         axes[kk].set_ylabel(label.split('_')[0] + ' (Pa)')
#         axes[kk].tick_params(axis='x', labelsize = 10)
#     else:
#         axes[kk].set_ylabel(None)
#         axes[kk].set_yticklabels([])
    # ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_Detailed1DPlot', figSubDir = figSubDir)
    
    kk+=1

renameAxes(axes,{'none':'ctrl', 'doxycyclin':'iMC'})
fig.tight_layout()
# fig.suptitle('Tangeantial modulus of aSFL 3T3 - control vs iMC linker')
plt.show()


# %%%%  Table with only MCA data


# %%%%% Check the number of valid compressions for each stress interval


data = GlobalTable_meca_MCA

columns = data.columns.values
minS = []
maxS = []
centerS = []
colList = []

for c in columns:
    if 'KChadwick' in c and '<s<' in c:
        valS = c.split('_')[-1].split('<s<')
        valS[0], valS[1] = int(valS[0]), int(valS[1])
        minS.append(valS[0])
        maxS.append(valS[1])
        centerS.append(int(0.5*(valS[0]+valS[1])))
        colList.append(c)
#     elif 'Npts' in c and '<s<' in c:
        
intervalSizes = np.array(maxS) - np.array(minS)
if len(np.unique(intervalSizes)) == 1:
    intervalSize = int(intervalSizes[0])
        
Filters = [(data['validatedFit'] == True),
           (data['validatedThickness'] == True),
           (data['date'].apply(lambda x : x in ['21-01-18', '21-01-21']))]

data_f = data
for F in Filters:
    data_f = data_f[F]

count = []
for i in range(len(colList)):
    c = colList[i]
    S = centerS[i]
    Vals = data_f[c].values
    countValidFit = len(Vals) - np.sum(np.isnan(Vals))
    count.append(countValidFit)
    
d = {'center_stress' : centerS, 'count_valid' : count}
dfCheck = pd.DataFrame(d)


print(data_f.shape[0])
print(dfCheck.T)


# %%%%% All stress ranges - K(stress), ctrl vs doxy

# '21-01-18', '21-01-21' > standard aSFL (A11)
# '21-04-27', '21-04-28' > long linker aSFL (6FP)
# '21-09-08' > long linker aSFL (6FP-2)
# '21-09-09' > 'high expresser' aSFL (A8)

data = GlobalTable_meca_MCA
dates = ['21-01-18', '21-01-21']
listeS = [i for i in range(100, 1100, 100)]
# listeS = [200, 300, 400, 500, 600, 700, 800, 900, 1000]


fig, axes = plt.subplots(1, len(listeS), figsize = (20,6))
kk = 0

for S in listeS:
    interval = str(S-intervalSize//2) + '<s<' + str(S+intervalSize//2)

    Filters = [(data['validatedFit_'+interval] == True),
               (data['validatedThickness'] == True),
               (data['date'].apply(lambda x : x in dates))]
    
    
    # axes[kk].set_yscale('log')
    # axes[kk].set_ylim([5e2, 5e4])
    # axes[kk].set_yscale('linear')
    axes[kk].set_ylim([-10, 3e4])

    fig, ax = D1Plot(data, fig=fig, ax=axes[kk], CondCols=['drug'], Parameters=['KChadwick_'+interval], 
                     Filters=Filters, Boxplot=True, cellID='cellID', co_order=['none', 'doxycyclin'], 
                     stats=True, statMethod='Mann-Whitney', AvgPerCell = False, box_pairs=[], 
                     figSizeFactor = 1, markersizeFactor=0.7, orientation = 'h', stressBoxPlot = True, 
                     bypassLog = True)

    # axes[kk].legend(loc = 'upper right', fontsize = 8)
    label = axes[kk].get_ylabel()
    axes[kk].set_title(label.split('_')[-1] + 'Pa', fontsize = 10)
    axes[kk].tick_params(axis='x', labelrotation = 50, labelsize = 10)
    axes[kk].set_ylim([-10, 3e4])
    if kk == 0:
        axes[kk].set_ylabel(label.split('_')[0] + ' (Pa)')
        axes[kk].tick_params(axis='x', labelsize = 10)
    else:
        axes[kk].set_ylabel(None)
        axes[kk].set_yticklabels([])
    
    kk+=1

renameAxes(axes,{'none':'ctrl', 'doxycyclin':'iMC'})
fig.tight_layout()
# fig.suptitle('Tangeantial modulus of aSFL 3T3 - control vs iMC linker')

plt.show()


# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_K_multipleIntervals_250pa_lin', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir = figSubDir)
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_K_multipleIntervals_250pa_lin', figDir = ownCloudTodayFigDir, figSubDir = figSubDir)

# %%%%% Only One Stress Range - K(stress), ctrl vs doxy

# '21-01-18', '21-01-21' > standard aSFL (A11)
# '21-04-27', '21-04-28' > long linker aSFL (6FP)
# '21-09-08' > long linker aSFL (6FP-2)
# '21-09-09' > 'high expresser' aSFL (A8)

data = GlobalTable_meca_MCA
dates = ['21-01-18', '21-01-21']
listeS = [300]
# listeS = [200, 300, 400, 500, 600, 700, 800, 900, 1000]


fig, axes = plt.subplots(1, len(listeS), figsize = (6,5))
kk = 0

intervalSize = 200

for S in listeS:
    interval = str(S-intervalSize//2) + '<s<' + str(S+intervalSize//2)

    Filters = [(data['validatedFit_'+interval] == True),
               (data['validatedThickness'] == True),
               (data['date'].apply(lambda x : x in dates))]
    
    
    # axes.set_yscale('log')
    # axes.set_ylim([5e2, 5e4])
    # axes.set_yscale('linear')
    axes.set_ylim([500, 3e4])

    fig, ax = D1Plot(data, fig=fig, ax=axes, CondCols=['drug'], Parameters=['KChadwick_'+interval], 
                     Filters=Filters, Boxplot=True, cellID='cellID', co_order=['none', 'doxycyclin'], 
                     stats=True, statMethod='Mann-Whitney', AvgPerCell = False, box_pairs=[], 
                     figSizeFactor = 1, markersizeFactor=1.1, orientation = 'h', stressBoxPlot = True, 
                     bypassLog = False)

    # axes.legend(loc = 'upper right', fontsize = 8)
    label = axes.get_ylabel()
    axes.set_title(label.split('_')[-1] + 'Pa\n', fontsize = 10)
    axes.tick_params(axis='x', labelrotation = 30, labelsize = 14)
    axes.set_ylim([-10, 3e4])
    
renameAxes(axes,{'none':'Control', 'doxycyclin':'Increased MCA', 'KChadwick_'+interval:'Tangeantial Modulus (Pa)'})
fig.tight_layout()
# fig.suptitle('Tangeantial modulus of aSFL 3T3 - control vs iMC linker')

plt.show()


# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_K_multipleIntervals_250pa_lin', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir = figSubDir)
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_K_multipleIntervals_250pa_lin', figDir = ownCloudTodayFigDir, figSubDir = figSubDir)




# %%%%% All stress ranges - K(stress), ctrl vs doxy - 200nm < H0 < 300nm

# '21-01-18', '21-01-21' > standard aSFL (A11)
# '21-04-27', '21-04-28' > long linker aSFL (6FP)
# '21-09-08' > long linker aSFL (6FP-2)
# '21-09-09' > 'high expresser' aSFL (A8)

data = GlobalTable_meca_MCA
dates = ['21-01-18', '21-01-21']
listeS = [i for i in range(200, 1100, 100)]
# listeS = [200, 300, 400, 500, 600, 700, 800, 900, 1000]


fig, axes = plt.subplots(1, len(listeS), figsize = (20,6))
kk = 0

for S in listeS:
    interval = str(S-intervalSize//2) + '<s<' + str(S+intervalSize//2)

    Filters = [(data['validatedFit_'+interval] == True),
               (data['validatedThickness'] == True),
               ((data['bestH0'] <= 300) & (data['bestH0'] >= 200)),
               (data['date'].apply(lambda x : x in dates))]
    
    fig, ax = D1Plot(data, fig=fig, ax=axes[kk], CondCols=['drug'], Parameters=['KChadwick_'+interval], 
                     Filters=Filters, Boxplot=True, cellID='cellID', co_order=['none', 'doxycyclin'], 
                     stats=True, statMethod='Mann-Whitney', AvgPerCell = False, box_pairs=[], 
                     figSizeFactor = 1, markersizeFactor=0.7, orientation = 'h', stressBoxPlot = True, 
                     bypassLog = True)

#     axes[kk].legend(loc = 'upper right', fontsize = 8)
    label = axes[kk].get_ylabel()
    axes[kk].set_title(label.split('_')[-1] + 'Pa', fontsize = 10)
    axes[kk].set_yscale('log')
    axes[kk].set_ylim([5e2, 5e4])
    axes[kk].tick_params(axis='x', labelrotation = 50, labelsize = 10)
    if kk == 0:
        axes[kk].set_ylabel(label.split('_')[0] + ' (Pa)')
        axes[kk].tick_params(axis='x', labelsize = 10)
    else:
        axes[kk].set_ylabel(None)
        axes[kk].set_yticklabels([])
   
    kk+=1

renameAxes(axes,{'none':'ctrl', 'doxycyclin':'iMC'})
fig.tight_layout()
fig.suptitle('Tangeantial modulus of aSFL 3T3 - control vs iMC linker')
plt.show()
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_K_multipleIntervals_250pa_200-300nm', figSubDir = figSubDir)

# %%%%% All stress ranges - K(stress), ctrl vs doxy -- With bestH0 distributions and E(h)

# '21-01-18', '21-01-21' > standard aSFL (A11)
# '21-04-27', '21-04-28' > long linker aSFL (6FP)
# '21-09-08' > long linker aSFL (6FP-2)
# '21-09-09' > 'high expresser' aSFL (A8)

data = GlobalTable_meca_MCA
dates = ['21-01-18', '21-01-21']
listeS = [i for i in range(100, 1100, 100)]
# listeS = [200, 300, 400, 500, 600, 700, 800, 900, 1000]


fig, axes = plt.subplots(3, len(listeS), figsize = (20,10))
kk = 0

for S in listeS:
    interval = str(S-intervalSize//2) + '<s<' + str(S+intervalSize//2)
    print(interval)
    
    Filters = [(data['validatedFit_'+interval] == True),
               (data['validatedThickness'] == True),
               (data['bestH0'] <= 1100),
               (data['date'].apply(lambda x : x in dates))]
    
    
    # axes[0, kk].set_yscale('log')
    axes[0, kk].set_ylim([5e2, 5e4])
    # axes[0, kk].set_yscale('linear')
    # axes[0, kk].set_ylim([0, 3e4])

    D1Plot(data, fig=fig, ax=axes[0, kk], CondCols=['drug'], Parameters=['KChadwick_'+interval], 
         Filters=Filters, Boxplot=True, cellID='cellID', co_order=['none', 'doxycyclin'], 
         stats=True, statMethod='Mann-Whitney', AvgPerCell = False, box_pairs=[], 
         figSizeFactor = 1, markersizeFactor=0.6, orientation = 'h', 
         stressBoxPlot = True)# , bypassLog = True)
    
    D1Plot(data, fig=fig, ax=axes[1, kk], CondCols=['drug'], Parameters=['bestH0'], 
         Filters=Filters, Boxplot=True, cellID='cellID', co_order=['none', 'doxycyclin'], 
         stats=True, statMethod='Mann-Whitney', AvgPerCell = False, box_pairs=[], 
         figSizeFactor = 1, markersizeFactor=0.6, orientation = 'h', stressBoxPlot = True)
    
    D2Plot_wFit(data, fig=fig, ax=axes[2, kk], Filters=Filters, 
          XCol='bestH0', YCol='KChadwick_'+interval, CondCol = ['drug'], co_order=['none', 'doxycyclin'], 
          cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', 
          modelFit=True, modelType='y=k*x^a', markersizeFactor = 0.6)
    
    # 0.
    # axes[0, kk].legend(loc = 'upper right', fontsize = 8)
    label0 = axes[0, kk].get_ylabel()
    axes[0, kk].set_title(label0.split('_')[-1] + 'Pa', fontsize = 10)
    axes[0, kk].set_ylim([5e2, 5e4])
    # axes[0, kk].set_ylim([1, 3e4])
    # axes[0, kk].set_xticklabels([])
    
    # 1.
    axes[1, kk].set_ylim([0, 1500])
    axes[1, kk].tick_params(axis='x', labelrotation = 50, labelsize = 10)
    
    # 2.
    # label2 = axes[2, kk].get_ylabel()
    axes[2, kk].set_xlim([8e1, 1.2e3])
    axes[2, kk].set_ylim([3e2, 5e4])
    axes[2, kk].tick_params(axis='x', labelrotation = 0, labelsize = 10)
    axes[2, kk].xaxis.label.set_size(10)
    axes[2, kk].legend().set_visible(False)
    
    for ll in range(axes.shape[0]):
        axes[ll, kk].tick_params(axis='y', labelsize = 10)
    
    if kk == 0:
        axes[0, kk].set_ylabel(label0.split('_')[0] + ' (Pa)')
        axes[1, kk].set_ylabel('Best H0 (nm)')
        axes[2, kk].set_ylabel('K_Chadwick (Pa)')
    else:
        for ll in range(axes.shape[0]):
            axes[ll, kk].set_ylabel(None)
            axes[ll, kk].set_yticklabels([])
    
    kk+=1

renameAxes(axes.flatten(),{'none':'ctrl', 'doxycyclin':'iMC', 'bestH0' : 'Best H0 (nm)'})
# fig.tight_layout()
# fig.suptitle('Tangeantial modulus of aSFL 3T3 - control vs iMC linker')

plt.show()


# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_K_multipleIntervals_250pa_lin', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir = figSubDir)
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_K_multipleIntervals_250pa_lin', figDir = ownCloudTodayFigDir, figSubDir = figSubDir)

# %%%%% All stress ranges - K(stress), ctrl vs doxy -- With bestH0 distributions and E(h) - effect of windows width


data = GlobalTable_meca_MCA
dates = ['21-01-18', '21-01-21']

fitC =  np.array([S for S in range(100, 1000, 100)])
fitW = [100, 150, 200, 250, 300]
fitCenters = np.array([[int(S) for S in fitC] for w in fitW])
fitWidth = np.array([[int(w) for S in fitC] for w in fitW])
fitMin = np.array([[int(S-(w/2)) for S in fitC] for w in fitW])
fitMax = np.array([[int(S+(w/2)) for S in fitC] for w in fitW])
# fitCenters, fitWidth = fitCenters[fitMin>0], fitWidth[fitMin>0]
# fitMin, fitMax = fitMin[fitMin>0], fitMax[fitMin>0]

for ii in range(len(fitW)):
    width = fitW[ii]
    fitMin_ii = fitMin[ii]
    N = len(fitMin_ii[fitMin_ii > 0])
    
    fig, axes = plt.subplots(3, N, figsize = (20,10))
    kk = 0
    
    for S in fitC:
        minS, maxS = int(S-width//2), int(S+width//2)
        interval = str(minS) + '<s<' + str(maxS)
        
        if minS > 0:
            print(interval)
            try:
                Filters = [(data['validatedFit_'+interval] == True),
                           (data['validatedThickness'] == True),
                           (data['bestH0'] <= 1100),
                           (data['date'].apply(lambda x : x in dates))]
                
                
                # axes[0, kk].set_yscale('log')
                axes[0, kk].set_ylim([5e2, 5e4])
                # axes[0, kk].set_yscale('linear')
                # axes[0, kk].set_ylim([0, 3e4])
            
                D1Plot(data, fig=fig, ax=axes[0, kk], CondCols=['drug'], Parameters=['KChadwick_'+interval], 
                     Filters=Filters, Boxplot=True, cellID='cellID', co_order=['none', 'doxycyclin'], 
                     stats=True, statMethod='Mann-Whitney', AvgPerCell = False, box_pairs=[], 
                     figSizeFactor = 1, markersizeFactor=0.6, orientation = 'h', 
                     stressBoxPlot = True)# , bypassLog = True)
                
                D1Plot(data, fig=fig, ax=axes[1, kk], CondCols=['drug'], Parameters=['bestH0'], 
                     Filters=Filters, Boxplot=True, cellID='cellID',  co_order=['none', 'doxycyclin'], 
                     stats=True, statMethod='Mann-Whitney', AvgPerCell = False, box_pairs=[], 
                     figSizeFactor = 1, markersizeFactor=0.6, orientation = 'h', stressBoxPlot = True)
                
                D2Plot_wFit(data, fig=fig, ax=axes[2, kk], Filters=Filters, 
                      XCol='bestH0', YCol='KChadwick_'+interval, CondCol = ['drug'],  co_order=['none', 'doxycyclin'], 
                      cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', 
                      modelFit=True, modelType='y=k*x^a', markersizeFactor = 0.6)
                
            except:
                pass
            
            # 0.
            # axes[0, kk].legend(loc = 'upper right', fontsize = 8)
            # label0 = axes[0, kk].get_ylabel()
            axes[0, kk].set_title('S = {:.0f} +/- {:.0f} Pa'.format(S, width//2), fontsize = 10)
            axes[0, kk].set_ylim([5e2, 5e4])
            # axes[0, kk].set_ylim([1, 3e4])
            # axes[0, kk].set_xticklabels([])
            
            # 1.
            axes[1, kk].set_ylim([0, 1500])
            axes[1, kk].tick_params(axis='x', labelrotation = 50, labelsize = 10)
            
            # 2.
            # label2 = axes[2, kk].get_ylabel()
            axes[2, kk].set_xlim([8e1, 1.2e3])
            axes[2, kk].set_ylim([3e2, 5e4])
            axes[2, kk].tick_params(axis='x', labelrotation = 0, labelsize = 10)
            axes[2, kk].xaxis.label.set_size(10)
            axes[2, kk].legend().set_visible(False)
            
            for ll in range(axes.shape[0]):
                axes[ll, kk].tick_params(axis='y', labelsize = 10)
            
            if kk == 0:
                axes[0, kk].set_ylabel(label0.split('_')[0] + ' (Pa)')
                axes[1, kk].set_ylabel('Best H0 (nm)')
                axes[2, kk].set_ylabel('K_Chadwick (Pa)')
            else:
                for ll in range(axes.shape[0]):
                    axes[ll, kk].set_ylabel(None)
                    axes[ll, kk].set_yticklabels([])
        
            kk+=1
            
        else:
            pass
    
    renameAxes(axes.flatten(),{'none':'ctrl', 'doxycyclin':'iMC', 'bestH0' : 'Best H0 (nm)'})
    # fig.tight_layout()
    # fig.suptitle('Tangeantial modulus of aSFL 3T3 - control vs iMC linker')
    
    fig.suptitle('Width = {:.0f}'.format(width))
    
    plt.show()


    # ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_K_multipleIntervals_250pa_lin', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir = figSubDir)
    # ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_K_multipleIntervals_250pa_lin', figDir = ownCloudTodayFigDir, figSubDir = figSubDir)

# %%%%% All stress ranges - K(stress), ctrl vs doxy -- effect of windows width -> 5width


data = GlobalTable_meca_MCA # Get the table in the folder "Width" or recompute it !
dates = ['21-01-18', '21-01-21']

fitC =  np.array([S for S in range(100, 1000, 100)])
fitW = [100, 150, 200, 250, 300]
fitCenters = np.array([[int(S) for S in fitC] for w in fitW])
fitWidth = np.array([[int(w) for S in fitC] for w in fitW])
fitMin = np.array([[int(S-(w/2)) for S in fitC] for w in fitW])
fitMax = np.array([[int(S+(w/2)) for S in fitC] for w in fitW])
# fitCenters, fitWidth = fitCenters[fitMin>0], fitWidth[fitMin>0]
# fitMin, fitMax = fitMin[fitMin>0], fitMax[fitMin>0]

N = len(fitC)
fig, axes = plt.subplots(5, N, figsize = (20,10))

for ii in range(len(fitW)):
    width = fitW[ii]
    fitMin_ii = fitMin[ii]
    
    kk = 0
    
    for S in fitC:
        minS, maxS = int(S-width//2), int(S+width//2)
        interval = str(minS) + '<s<' + str(maxS)
        
        if minS > 0:
            print(interval)

            Filters = [(data['validatedFit_'+interval] == True),
                       (data['validatedThickness'] == True),
                       (data['bestH0'] <= 1000),
                       (data['date'].apply(lambda x : x in dates))]
            
            
            # axes[0, kk].set_yscale('log')
            axes[0, kk].set_ylim([5e2, 5e4])
            # axes[0, kk].set_yscale('linear')
            # axes[0, kk].set_ylim([0, 3e4])
        
            D1Plot(data, fig=fig, ax=axes[ii, kk], CondCols=['drug'], Parameters=['KChadwick_'+interval], 
                 Filters=Filters, Boxplot=True, cellID='cellID', co_order=['none', 'doxycyclin'], 
                 stats=True, statMethod='Mann-Whitney', AvgPerCell = True, box_pairs=[], 
                 figSizeFactor = 1, markersizeFactor=0.6, orientation = 'h', 
                 stressBoxPlot = True)# , bypassLog = True)
            
            # D1Plot(data, fig=fig, ax=axes[1, kk], CondCols=['drug'], Parameters=['bestH0'], 
            #      Filters=Filters, Boxplot=True, cellID='cellID',  co_order=['none', 'doxycyclin'], 
            #      stats=True, statMethod='Mann-Whitney', AvgPerCell = False, box_pairs=[], 
            #      figSizeFactor = 1, markersizeFactor=0.6, orientation = 'h', stressBoxPlot = True)
            
            # D2Plot_wFit(data, fig=fig, ax=axes[2, kk], Filters=Filters, 
            #       XCol='bestH0', YCol='KChadwick_'+interval, CondCol = ['drug'],  co_order=['none', 'doxycyclin'], 
            #       cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', 
            #       modelFit=True, modelType='y=k*x^a', markersizeFactor = 0.6)
                

            
            # # 0.
            # # axes[0, kk].legend(loc = 'upper right', fontsize = 8)
            # # label0 = axes[0, kk].get_ylabel()
            # axes[0, kk].set_title('S = {:.0f} +/- {:.0f} Pa'.format(S, width//2), fontsize = 10)
            # axes[0, kk].set_ylim([5e2, 5e4])
            # # axes[0, kk].set_ylim([1, 3e4])
            # # axes[0, kk].set_xticklabels([])
            
            # # 1.
            # axes[1, kk].set_ylim([0, 1500])
            # axes[1, kk].tick_params(axis='x', labelrotation = 50, labelsize = 10)
            
            # # 2.
            # # label2 = axes[2, kk].get_ylabel()
            # axes[2, kk].set_xlim([8e1, 1.2e3])
            # axes[2, kk].set_ylim([3e2, 5e4])
            # axes[2, kk].tick_params(axis='x', labelrotation = 0, labelsize = 10)
            # axes[2, kk].xaxis.label.set_size(10)
            # axes[2, kk].legend().set_visible(False)  
        else:
            pass
        
        
        axes[ii, kk].set_title('S = {:.0f} +/- {:.0f} Pa'.format(S, width//2), fontsize = 10)
        axes[ii, kk].set_ylim([5e2, 5e4])
        axes[ii, kk].tick_params(axis='y', labelsize = 10)
        
        if kk == 0:
            axes[ii, kk].set_ylabel('K ; width={:.0f}Pa'.format(width), fontsize = 10)
        else:
            axes[ii, kk].set_ylabel(None)
            axes[ii, kk].set_yticklabels([])
            
        if ii == len(fitW)-1:
            axes[ii, kk].tick_params(axis='x', labelrotation = 50, labelsize = 10)
        else:
            axes[ii, kk].tick_params(axis='x', labelrotation = 50, labelsize = 10)
            axes[ii, kk].set_xlabel(None)
            axes[ii, kk].set_xticklabels([])
            
        kk+=1
    
    renameAxes(axes.flatten(),{'none':'ctrl', 'doxycyclin':'iMC', 'bestH0' : 'Best H0 (nm)'})
    # fig.tight_layout()
    # fig.suptitle('Tangeantial modulus of aSFL 3T3 - control vs iMC linker')
    
    fig.suptitle('Tangeantial moduli with different windows for the fit')
    
    plt.show()


    # ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_K_multipleIntervals_250pa_lin', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir = figSubDir)
    # ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_K_multipleIntervals_250pa_lin', figDir = ownCloudTodayFigDir, figSubDir = figSubDir)



# %%%%% extremalStress = f(bestH0) - whole dataset


# ['21-12-08', '21-12-16', '22-01-12']
data = GlobalTable_meca_Py2

dates = ['21-01-18', '21-01-21'] # ['21-01-18', '21-01-21', '21-12-08']

filterList = [(data['validatedThickness'] == True),
              (data['cell subtype'] == 'aSFL'), 
              (data['bead type'] == 'M450'),
              (data['substrate'] == '20um fibronectin discs'),
              (data['date'].apply(lambda x : x in dates))]  # (data['validatedFit'] == True), 


fig, ax = plt.subplots(1, 1, figsize = (9, 6), tight_layout=True)

globalFilter = filterList[0]
for k in range(1, len(filterList)):
    globalFilter = globalFilter & filterList[k]

data_f = data[globalFilter]
    
condition = 'drug'
conditionValues = data_f[condition].unique()
conditionFilter = {}
for val in conditionValues:
    conditionFilter[val] = (data_f[condition] == val)
    
rD = {'none' : 'Ctrl', 'doxycyclin' : 'iMC linker'}

for val in conditionValues:
    if val == 'none':
        col1, col2, col3 = 'deepskyblue', 'skyblue', 'royalblue'
    if val == 'doxycyclin':
        col1, col2, col3 = 'orangered', 'lightsalmon', 'firebrick'
    X = data_f[conditionFilter[val]]['bestH0'].values
    Ncomp = len(X)
    Smin = data_f[conditionFilter[val]]['minStress'].values
    Smax = data_f[conditionFilter[val]]['maxStress'].values

    for i in range(Ncomp):
        if i == 0:
            textLabel = rD[val]
        else:
            textLabel = None
        ax.plot([X[i], X[i]], [Smin[i], Smax[i]], 
                ls = '-', color = col1, alpha = 0.3,
                zorder = 1, label = textLabel)
        ax.plot([X[i]], [Smin[i]], 
                marker = 'o', markerfacecolor = col2, markeredgecolor = 'k', markersize = 4,
                ls = '')
        ax.plot([X[i]], [Smax[i]], 
                marker = 'o', markerfacecolor = col3, markeredgecolor = 'k', markersize = 4,
                ls = '')

ax.set_xlabel('Cortical thickness (nm)')
ax.set_xlim([0, 1200])
locator = matplotlib.ticker.MultipleLocator(100)
ax.xaxis.set_major_locator(locator)

ax.set_ylabel('Extremal stress values (Pa)')
ax.set_ylim([3e1, 5e4])
ax.set_yscale('log')
# locator = matplotlib.ticker.MultipleLocator(100)
# ax.yaxis.set_major_locator(locator)
ax.yaxis.grid(True)

ax.legend(loc = 'upper right')

ax.set_title('Jan 21 - Extremal stress values vs. thickness, all compressions, ctrl vs. iMC')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_ctrlVsLinker_StressRanges', figSubDir = figSubDir)
print(Ncomp)
plt.show()


# %%%%% extremalStress = f(bestH0) - just a slice


# ['21-12-08', '21-12-16', '22-01-12']
data = GlobalTable_meca_MCA

dates = ['21-01-18', '21-01-21'] # ['21-01-18', '21-01-21', '21-12-08']

filterList = [(data['validatedThickness'] == True),
              (data['cell subtype'] == 'aSFL'), 
              (data['bead type'] == 'M450'),
              ((data['bestH0'] <= 300) & (data['bestH0'] >= 200)),
              (data['substrate'] == '20um fibronectin discs'),
              (data['date'].apply(lambda x : x in dates))]  # (data['validatedFit'] == True), 


fig, ax = plt.subplots(1, 1, figsize = (9, 6), tight_layout=True)

globalFilter = filterList[0]
for k in range(1, len(filterList)):
    globalFilter = globalFilter & filterList[k]

data_f = data[globalFilter]
    
condition = 'drug'
conditionValues = data_f[condition].unique()
conditionFilter = {}
for val in conditionValues:
    conditionFilter[val] = (data_f[condition] == val)
    
rD = {'none' : 'Ctrl', 'doxycyclin' : 'iMC linker'}

for val in conditionValues:
    if val == 'none':
        col1, col2, col3 = 'deepskyblue', 'skyblue', 'royalblue'
    if val == 'doxycyclin':
        col1, col2, col3 = 'orangered', 'lightsalmon', 'firebrick'
    X = data_f[conditionFilter[val]]['bestH0'].values
    Ncomp = len(X)
    Smin = data_f[conditionFilter[val]]['minStress'].values
    Smax = data_f[conditionFilter[val]]['maxStress'].values

    for i in range(Ncomp):
        if i == 0:
            textLabel = rD[val]
        else:
            textLabel = None
        ax.plot([X[i], X[i]], [Smin[i], Smax[i]], 
                ls = '-', color = col1, alpha = 0.3,
                zorder = 1, label = textLabel)
        ax.plot([X[i]], [Smin[i]], 
                marker = 'o', markerfacecolor = col2, markeredgecolor = 'k', markersize = 4,
                ls = '')
        ax.plot([X[i]], [Smax[i]], 
                marker = 'o', markerfacecolor = col3, markeredgecolor = 'k', markersize = 4,
                ls = '')

ax.set_xlabel('Cortical thickness (nm)')
ax.set_xlim([0, 1200])
locator = matplotlib.ticker.MultipleLocator(100)
ax.xaxis.set_major_locator(locator)

ax.set_ylabel('Extremal stress values (Pa)')
ax.set_ylim([3e1, 5e4])
ax.set_yscale('log')
# locator = matplotlib.ticker.MultipleLocator(100)
# ax.yaxis.set_major_locator(locator)
ax.yaxis.grid(True)

ax.legend(loc = 'upper right')

ax.set_title('Jan 21 - Extremal stress values vs. thickness, all compressions, ctrl vs. iMC')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_ctrlVsLinker_StressRanges', figSubDir = figSubDir)
print(Ncomp)
plt.show()


# %%%%% "Three metrics" : Median H, Best H0 avg per cell, Best H0 all comp // change the dates !

# '21-01-18', '21-01-21' > standard aSFL (A11)
# '21-04-27', '21-04-28' > long linker aSFL (6FP)
# '21-09-08' > long linker aSFL (6FP-2)
# '21-09-09' > 'high expresser' aSFL (A8)


data = GlobalTable_meca_Py2 # Tracking python, postprocessing python
# data['bestH02'] = data['bestH0']**2
dates = ['21-01-18', '21-01-21']

fig, ax = plt.subplots(1, 3, figsize = (12,5))

#1
Filters = [(data['validatedThickness'] == True), 
            (data['ctFieldThickness'] <= 1000),
            (data['bead type'] == 'M450'),
            (data['substrate'] == '20um fibronectin discs'),
            (data['date'].apply(lambda x : x in dates))]

co_order = makeOrder(['none','doxycyclin'])

print(data[data['date'].apply(lambda x : x in dates)]['cell subtype'].unique())

fig, ax1 = D1Plot(data, fig=fig, ax=ax[0], CondCols=['drug'],Parameters=['ctFieldThickness'], AvgPerCell = True,
                 Filters=Filters,stats=True,co_order=co_order,figSizeFactor=1.2)
ax[0].set_title('Median thickness at nominal field B~5.5mT')
renameAxes(ax[0],renameDict1)


#2
Filters = [(data['validatedThickness'] == True),
            (data['bestH0'] <= 1000),
            (data['bead type'] == 'M450'),
            (data['substrate'] == '20um fibronectin discs'),
            (data['date'].apply(lambda x : x in dates))]

co_order = makeOrder(['none','doxycyclin'])

fig, ax2 = D1Plot(data, fig=fig, ax=ax[1], CondCols=['drug'],Parameters=['bestH0'], AvgPerCell = True,
                 Filters=Filters,stats=True,co_order=co_order,figSizeFactor=1.5)

ax[1].set_title('H0, average per cell')
renameAxes(ax[1],renameDict1)
renameAxes(ax[1],{'bestH0':'H0 (nm)'})

#3
# Filters = [(data['validatedThickness'] == True),
#             (data['bestH0'] <= 1000),
#             (data['bead type'] == 'M450'),
#             (data['substrate'] == '20um fibronectin discs'),
#             (data['date'].apply(lambda x : x in dates))]

co_order = makeOrder(['none','doxycyclin'])

fig, ax3 = D1Plot(data, fig=fig, ax=ax[2], CondCols=['drug'],Parameters=['bestH0'], AvgPerCell = False,
                 Filters=Filters,stats=True,co_order=co_order,figSizeFactor=1.5, stressBoxPlot = True, 
                 markersizeFactor = 0.5)

ax[2].set_title('H0, all compressions')
renameAxes(ax[2],renameDict1)
renameAxes(ax[2],{'bestH0':'Best H0 (nm)'})


multiAxes = ax #, thisAx7bis]
                        
for axis in multiAxes:
    axis.set_ylim([0,1050])
    for item in ([axis.title, axis.xaxis.label,axis.yaxis.label] + axis.get_xticklabels() + axis.get_yticklabels()):
        item.set_fontsize(11)

fig.suptitle('3T3aSFL on patterns - Several thickness metrics', fontsize = 14)

# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_3metricsThicknessFromMECA', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir = 'ProjectAMC')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_3metricsThicknessFromMECA', figDir = ownCloudTodayFigDir, figSubDir = 'ProjectAMC')
# ufun.archiveFig(fig, ax, cp.DirDataFigToday + '//MCAproject', name='ThreeThickMetrics', dpi = 100)


plt.show()



# %%%%% Best H0 from fits B~5mT - All comp


data = GlobalTable_meca_MCA # Tracking python, postprocessing python
dates = []

Filters = [(data['validatedThickness'] == True), (data['bestH0'] <= 900)]

fig, ax = plt.subplots(1, 1, figsize = (6,5))

co_order = makeOrder(['none','doxycyclin'])
fig, ax, dfcount = D1Plot(fig=fig, ax=ax, data=data, CondCols=['drug'],Parameters=['bestH0'], AvgPerCell = False,
                 Filters=Filters,stats=True, co_order=co_order, figSizeFactor=1.0, 
                 markersizeFactor = 0.7, stressBoxPlot = True, returnCount=1)


ax[0].set_ylim([0, 1000])
rD = {'none':'Control', 'doxycyclin':'Increased MCA', 'bestH0' : 'Thickness (nm)'}
renameAxes(ax,rD)
# ax[0].set_ylim([0,600])
# ax[1].set_ylim([0,600])
fig.suptitle('3T3aSFL on patterns\nBest H0 from fits\n', fontsize = 12)
fig.tight_layout()
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_medianThickness', figSubDir = figSubDir)

plt.show()


# %%%%% Best H0 from fits B~5mT - Avg per cell


data = GlobalTable_meca_MCA # Tracking python, postprocessing python
dates = []

Filters = [(data['validatedThickness'] == True), (data['bestH0'] <= 1000)]

co_order = makeOrder(['none','doxycyclin'])
fig, ax = D1Plot(data, CondCols=['drug'],Parameters=['bestH0'], AvgPerCell = True,
                 Filters=Filters,stats=True,co_order=co_order,figSizeFactor=1.5)
renameAxes(ax,renameDict1)
# ax[0].set_ylim([0,600])
# ax[1].set_ylim([0,600])
fig.suptitle('3T3aSFL on patterns\nBest H0 from fits\nB~5mT', fontsize = 12)
fig.tight_layout()
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_medianThickness', figSubDir = figSubDir)

plt.show()


# %%%%% E(h) for ranges of stress - plot by plot

data = GlobalTable_meca_Py2

listeS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
dates = ['21-01-18', '21-01-21']
# fig, axes = plt.subplots(len(listeS), 1, figsize = (9,40))
# kk = 0

for S in listeS:
    interval = str(S-75) + '<s<' + str(S+75)
    textForFileName = str(S-75) + '-' + str(S+75) + 'Pa'

    Filters = [(data['validatedFit_'+interval] == True),
               (data['validatedThickness'] == True),
               (data['date'].apply(lambda x : x in dates))] #, '21-12-16'

    fig, ax = D2Plot_wFit(data, #fig=fig, ax=axes[kk], 
                          XCol='surroundingThickness', YCol='KChadwick_'+interval, CondCol = ['drug'], Filters=Filters, 
                          cellID = 'cellID', AvgPerCell=False, 
                          xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')
    
    # individual figs
    ax.set_xlim([8e1, 1.2e3])
    ax.set_ylim([3e2, 5e4])
    renameAxes(ax,{'none':'ctrl', 'doxycyclin':'iMC'})
    fig.set_size_inches(6,4)
    fig.tight_layout()
#     ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_ctrlVsLinker_E(h)_' + textForFileName, figSubDir = figSubDir)
    
    
    # one big fig
#     axes[kk].set_xlim([8e1, 1.2e3])
#     axes[kk].set_ylim([3e2, 5e4])
#     kk+=1


# renameAxes(axes,{'none':'ctrl', 'doxycyclin':'iMC'})
# fig.tight_layout()
plt.show()
    

# %%%%% E(h) for ranges of stress - all in one


data = GlobalTable_meca_Py2

listeS = [200, 300, 400, 500, 600, 700, 800, 900, 1000]

fig, axes = plt.subplots(len(listeS), 1, figsize = (9,40))
kk = 0

for S in listeS:
    interval = str(S-75) + '<s<' + str(S+75)

    Filters = [(data['validatedFit_'+interval] == True),
               (data['validatedThickness'] == True),
               (data['date'].apply(lambda x : x in ['21-01-18', '21-01-21']))] #, '21-12-16'

    fig, ax = D2Plot_wFit(data, fig=fig, ax=axes[kk], 
                          XCol='surroundingThickness', YCol='KChadwick_'+interval, CondCol = ['drug'], Filters=Filters, 
                          cellID = 'cellID', AvgPerCell=True, 
                          xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')
    
    
    
#     axes[kk].legend(loc = 'upper right', fontsize = 8)
#     label = axes[kk].get_ylabel()
#     axes[kk].set_title(label.split('_')[-1] + 'Pa', fontsize = 10)
#     axes[kk].set_yscale('log')
    axes[kk].set_xlim([8e1, 1.2e3])
#     axes[kk].tick_params(axis='x', labelrotation = 50, labelsize = 10)
#     if kk == 0:
#         axes[kk].set_ylabel(label.split('_')[0] + ' (Pa)')
#         axes[kk].tick_params(axis='x', labelsize = 10)
#     else:
#         axes[kk].set_ylabel(None)
#         axes[kk].set_yticklabels([])
    # ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_Detailed1DPlot', figSubDir = figSubDir)
    
    kk+=1

renameAxes(axes,{'none':'ctrl', 'doxycyclin':'iMC'})
fig.tight_layout()
# fig.suptitle('Tangeantial modulus of aSFL 3T3 - control vs iMC linker')
plt.show()
    

# %%%%% Effective stiffness - many ranges of s to find K


data = GlobalTable_meca_MCA # Tracking python, postprocessing python
dates = ['21-01-18', '21-01-21']
figSubDir='MCAproject'

# data['EqStiffness_200Pa'] = (2/3) * (data['bestH0']/1000)**2 * data['KChadwick_75<s<325'] / 10
# data['EqStiffness_300Pa'] = (2/3) * (data['bestH0']/1000)**2 * data['KChadwick_175<s<425'] / 10
# data['EqStiffness_400Pa'] = (2/3) * (data['bestH0']/1000)**2 * data['KChadwick_275<s<525'] / 10
# data['EqStiffness_500Pa'] = (2/3) * (data['bestH0']/1000)**2 * data['KChadwick_375<s<625'] / 10

R = 10
data['bestH0^2'] = (data['bestH0']/1000)**2
listS = [300, 400, 500, 600]
fig, ax = plt.subplots(len(listS), 3, figsize = (12,4.5*len(listS)))

for i in range(len(listS)):
    S = listS[i]
    data['EqStiffness_'+str(S)+'Pa'] = (2/3)*(data['bestH0']/1000)**2 * data['KChadwick_'+str(S-75)+'<s<'+str(S+75)] / R
    
    
    Filters = [(data['validatedThickness'] == True), (data['bestH0'] <= 1000), 
               (data['validatedFit_'+str(S-75)+'<s<'+str(S+75)] == True),
               (data['date'].apply(lambda x : x in dates))]

    co_order = makeOrder(['none','doxycyclin'])
    
    fig, ax00 = D1Plot(data, fig=fig, ax=ax[i,0], CondCols=['drug'],Parameters=['bestH0^2'], 
                       AvgPerCell = False, markersizeFactor = 0.7,
                       Filters=Filters,stats=True,co_order=co_order,figSizeFactor=1.5, stressBoxPlot = True)
    ax[i,0].set_title('bestH0^2')
    renameAxes(ax[i,0],renameDict1)
    renameAxes(ax[i,0],{'bestH0^2':'H0 squared (nm²)', 
                        'EqStiffness_'+str(S)+'Pa':'k_eff at '+str(S)+'Pa (pN/µm)'})
    
    fig, ax01 = D1Plot(data, fig=fig, ax=ax[i,1], CondCols=['drug'],Parameters=['EqStiffness_'+str(S)+'Pa'], 
                       AvgPerCell = False, markersizeFactor = 0.7,
                       Filters=Filters,stats=True,co_order=co_order,figSizeFactor=1.5, stressBoxPlot = True)
    ax[i,1].set_title('Effective spring constant at '+str(S)+'Pa, all compressions')
    renameAxes(ax[i,1],renameDict1)
    renameAxes(ax[i,1],{'bestH0':'Best H0 from fit (nm)', 
                        'EqStiffness_'+str(S)+'Pa':'k_eff at '+str(S)+'Pa (pN/µm)'})


    fig, ax02 = D1Plot(data, fig=fig, ax=ax[i,2], CondCols=['drug'],Parameters=['EqStiffness_'+str(S)+'Pa'], 
                       AvgPerCell = True, markersizeFactor = 0.7,
                       Filters=Filters,stats=True,co_order=co_order,figSizeFactor=1.5)
#     ax[i,2].set_title('Effective spring constant at '+str(S)+'Pa, average per cell')
    renameAxes(ax[i,2],renameDict1)
    renameAxes(ax[i,2],{'bestH0':'Best H0 from fit (nm)', 
                        'EqStiffness_'+str(S)+'Pa':'k_eff at '+str(S)+'Pa (pN/µm)'})



multiAxes = ax.flatten() #, thisAx7bis]
                        
for axis in multiAxes:
#     axis.set_ylim([0,1050])
    for item in ([axis.title, axis.xaxis.label,axis.yaxis.label] + axis.get_xticklabels() + axis.get_yticklabels()):
        item.set_fontsize(11)

fig.suptitle('3T3aSFL on patterns - K_eff', fontsize = 14)

# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_effectiveShellSpring', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir = 'ProjectAMC')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_effectiveShellSpring', figDir = ownCloudTodayFigDir, figSubDir = 'ProjectAMC')


plt.show()

# %%%%% Effective stiffness - K(s=400Pa)


data = GlobalTable_meca_MCA # Tracking python, postprocessing python
dates = ['21-01-18', '21-01-21']
figSubDir='MCAproject'

# data['EqStiffness_200Pa'] = (2/3) * (data['bestH0']/1000)**2 * data['KChadwick_75<s<325'] / 10
# data['EqStiffness_300Pa'] = (2/3) * (data['bestH0']/1000)**2 * data['KChadwick_175<s<425'] / 10
# data['EqStiffness_400Pa'] = (2/3) * (data['bestH0']/1000)**2 * data['KChadwick_275<s<525'] / 10
# data['EqStiffness_500Pa'] = (2/3) * (data['bestH0']/1000)**2 * data['KChadwick_375<s<625'] / 10

R = 10
data['bestH0^2'] = (data['bestH0']/1000)**2
listS = [600]
width = 200

fig, ax = plt.subplots(len(listS), 2, figsize = (12,4.5*len(listS)))

for i in range(len(listS)):
    S = listS[i]
    interval = str(int(S-width//2))+'<s<'+str(int(S+width//2))
    data['EqStiffness_'+str(S)+'Pa'] = (2/3)*(data['bestH0']/1000)**2 * data['KChadwick_' + interval] / R
    
    
    Filters = [(data['validatedThickness'] == True), (data['bestH0'] <= 800), 
               (data['EqStiffness_'+str(S)+'Pa']<= 150),
               (data['validatedFit_' + interval] == True),
               (data['date'].apply(lambda x : x in dates))]

    co_order = makeOrder(['none','doxycyclin'])
    
    ax[0].set_ylim([0, 150])
    fig, ax01 = D1Plot(data, fig=fig, ax=ax[0], CondCols=['drug'],Parameters=['EqStiffness_'+str(S)+'Pa'], 
                       AvgPerCell = False, markersizeFactor = 0.7,
                       Filters=Filters,stats=True,co_order=co_order,figSizeFactor=0.8, stressBoxPlot = True)
    ax[0].set_title('Effective spring constant at '+str(S)+'Pa,\nall compressions')
    renameAxes(ax[0],renameDict1)
    renameAxes(ax[0],{'bestH0':'Best H0 from fit (nm)', 
                        'EqStiffness_'+str(S)+'Pa':'k_eff (pN/µm)'})
    ax[0].set_ylim([0, 150])
    

    ax[1].set_ylim([0, 150])
    fig, ax02 = D1Plot(data, fig=fig, ax=ax[1], CondCols=['drug'],Parameters=['EqStiffness_'+str(S)+'Pa'], 
                       AvgPerCell = True, markersizeFactor = 1.0,
                       Filters=Filters,stats=True,co_order=co_order,figSizeFactor=0.8)
    ax[1].set_title('Effective spring constant at '+str(S)+'Pa,\naverage per cell')
    renameAxes(ax[1],renameDict1)
    renameAxes(ax[1],{'bestH0':'Best H0 from fit (nm)', 
                        'EqStiffness_'+str(S)+'Pa':'k_eff (pN/µm)'})
    ax[1].set_ylim([0, 150])
    



multiAxes = ax.flatten() #, thisAx7bis]
                        
for axis in multiAxes:
#     axis.set_ylim([0,1050])
    for item in ([axis.title, axis.xaxis.label,axis.yaxis.label] + axis.get_xticklabels() + axis.get_yticklabels()):
        item.set_fontsize(11)

fig.suptitle('3T3aSFL on patterns - k_eff', fontsize = 14)

# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_effectiveShellSpring', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir = 'ProjectAMC')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_effectiveShellSpring', figDir = ownCloudTodayFigDir, figSubDir = 'ProjectAMC')


plt.show()

# %%%% MCA + nonLin curves

# %%%%% K(s) for 3T3aSFL ctrl vs doxy

#### making the Dataframe 

data = GlobalTable_meca_MCA

dates = ['21-01-18', '21-01-21'] #['21-12-08', '22-01-12'] ['21-01-18', '21-01-21', '21-12-08']

filterList = [(data['validatedThickness'] == True),
              (data['cell subtype'] == 'aSFL'), 
              (data['bead type'] == 'M450'),
              (data['substrate'] == '20um fibronectin discs'),
              (data['bestH0'] <= 1000),
              (data['date'].apply(lambda x : x in dates))]  # (data['validatedFit'] == True),

globalFilter = filterList[0]
for k in range(1, len(filterList)):
    globalFilter = globalFilter & filterList[k]

data_f = data[globalFilter]

width = 200 # 200
fitCenters =  np.array([S for S in range(200, 1000, 50)])
fitMin = np.array([int(S-(width/2)) for S in fitCenters])
fitMax = np.array([int(S+(width/2)) for S in fitCenters])

regionFitsNames = [str(fitMin[ii]) + '<s<' + str(fitMax[ii]) for ii in range(len(fitMin))]

listColumnsMeca = []

KChadwick_Cols = []
KWeight_Cols = []

for rFN in regionFitsNames:
    listColumnsMeca += ['KChadwick_'+rFN, 'K_CIW_'+rFN, 'R2Chadwick_'+rFN, 'K2Chadwick_'+rFN, 
                        'H0Chadwick_'+rFN, 'Npts_'+rFN, 'validatedFit_'+rFN]
    KChadwick_Cols += [('KChadwick_'+rFN)]

    K_CIWidth = data_f['K_CIW_'+rFN] #.apply(lambda x : x.strip('][').split(', ')).apply(lambda x : (np.abs(float(x[0]) - float(x[1]))))
    KWeight = (data_f['KChadwick_'+rFN]/K_CIWidth)**2
    data_f['K_Weight_'+rFN] = KWeight
    data_f['K_Weight_'+rFN] *= data_f['KChadwick_'+rFN].apply(lambda x : (x<1e6))
    data_f['K_Weight_'+rFN] *= data_f['R2Chadwick_'+rFN].apply(lambda x : (x>1e-2))
    data_f['K_Weight_'+rFN] *= data_f['K_CIW_'+rFN].apply(lambda x : (x!=0))
    KWeight_Cols += [('K_Weight_'+rFN)]
    

#### Whole curve

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

valStr = 'KChadwick_'
weightStr = 'K_Weight_'



regionFitsNames = [str(fitMin[ii]) + '<s<' + str(fitMax[ii]) for ii in range(len(fitMin))]

fig, ax = plt.subplots(2,1, figsize = (9,12))
conditions = ['none', 'doxycyclin']
cD = {'none':[gs.colorList40[10], gs.colorList40[11]], 'doxycyclin':[gs.colorList40[30], gs.colorList40[31]]}
oD = {'none': [-15, 1.02] , 'doxycyclin': [5, 0.97] }
lD = {'none': 'Control' , 'doxycyclin': 'iMC linker' }

for drug in conditions:
    
    Kavg = []
    Kstd = []
    D10 = []
    D90 = []
    N = []
    
    data_ff = data_f[data_f['drug'] == drug]
    for ii in range(len(fitCenters)):
        rFN = str(fitMin[ii]) + '<s<' + str(fitMax[ii])
        variable = valStr+rFN
        weight = weightStr+rFN
        
        x = data_ff[variable].apply(nan2zero).values
        w = data_ff[weight].apply(nan2zero).values
        
        if S == 250:
            d = {'x' : x, 'w' : w}
        
        m = np.average(x, weights=w)
        v = np.average((x-m)**2, weights=w)
        std = v**0.5
        
        d10, d90 = np.percentile(x[x != 0], (10, 90))
        n = len(x[x != 0])
        
        Kavg.append(m)
        Kstd.append(std)
        D10.append(d10)
        D90.append(d90)
        N.append(n)
    
    Kavg = np.array(Kavg)
    Kstd = np.array(Kstd)
    D10 = np.array(D10)
    D90 = np.array(D90)
    N = np.array(N)
    Kste = Kstd / (N**0.5)
    
    alpha = 0.975
    dof = N
    q = st.t.ppf(alpha, dof) # Student coefficient
    
    d_val = {'S' : fitCenters, 'Kavg' : Kavg, 'Kstd' : Kstd, 'D10' : D10, 'D90' : D90, 'N' : N}
    
    
    # Weighted means -- Weighted ste 95% as error
    ax[0].errorbar(fitCenters, Kavg, yerr = q*Kste, marker = 'o', color = cD[drug][0], 
                   ecolor = 'k', elinewidth = 0.8, capsize = 3, label = lD[drug])
    ax[0].set_ylim([500,2e4])
    ax[0].set_xlim([0,1000])
    ax[0].set_title('Stress stiffening - All compressions pooled')
    
    # Weighted means -- D9-D1 as error
    ax[1].errorbar(fitCenters, Kavg, yerr = [D10, D90], marker = 'o', color = cD[drug][1], 
                   ecolor = 'k', elinewidth = 0.8, capsize = 3, label = lD[drug]) 
    ax[1].set_ylim([500,2e4])
    ax[1].set_xlim([0,1000])
    
    for k in range(2):
        ax[k].legend(loc = 'upper left')
        ax[k].set_yscale('log')
        ax[k].set_xlabel('Stress (Pa)')
        ax[k].set_ylabel('K (Pa)')
        for kk in range(len(N)):
            ax[k].text(x=fitCenters[kk]+oD[drug][0], y=Kavg[kk]**oD[drug][1], s='n='+str(N[kk]), fontsize = 6, color = cD[drug][k])
    
    fig.suptitle('K(s) - All compressions pooled'+'\n(fits width: {:.0f}Pa)'.format(width))    
    
    df_val = pd.DataFrame(d_val)
    dftest = pd.DataFrame(d)

plt.show()

ufun.archiveFig(fig, ax, cp.DirDataFigToday + '//MCAproject', name='aSFL_ctrl-vs-doxy_K(s)', dpi = 100)

#### Local zoom


Sinf, Ssup = 300, 800
extraFilters = [data_f['minStress'] <= Sinf, data_f['maxStress'] >= Ssup] # >= 800
fitCenters = fitCenters[(fitCenters>=(Sinf)) & (fitCenters<=Ssup)] # <800
fitMin = np.array([int(S-(width/2)) for S in fitCenters])
fitMax = np.array([int(S+(width/2)) for S in fitCenters])

data2_f = data_f
globalExtraFilter = extraFilters[0]
for k in range(1, len(extraFilters)):
    globalExtraFilter = globalExtraFilter & extraFilters[k]
data2_f = data2_f[globalExtraFilter]  


fig, ax = plt.subplots(2,1, figsize = (7,12))
conditions = ['none', 'doxycyclin']
# cD = {'none':[gs.colorList30[10], gs.colorList30[11]], 'doxycyclin':[gs.colorList30[20], gs.colorList30[21]]}
# oD = {'none': 1.04 , 'doxycyclin': 0.96 }
# lD = {'none': 'Control' , 'doxycyclin': 'iMC linker' }


for drug in conditions:
    
    Kavg = []
    Kstd = []
    D10 = []
    D90 = []
    N = []
    
    data_ff = data2_f[data_f['drug'] == drug]
    
    for ii in range(len(fitCenters)):
        print(fitCenters[ii])
        rFN = str(fitMin[ii]) + '<s<' + str(fitMax[ii])
        variable = valStr+rFN
        weight = weightStr+rFN
        
        x = data_ff[variable].apply(nan2zero).values
        w = data_ff[weight].apply(nan2zero).values
        
        if S == 250:
            d = {'x' : x, 'w' : w}
        
        m = np.average(x, weights=w)
        v = np.average((x-m)**2, weights=w)
        std = v**0.5
        
        d10, d90 = np.percentile(x[x != 0], (10, 90))
        n = len(x[x != 0])
        
        Kavg.append(m)
        Kstd.append(std)
        D10.append(d10)
        D90.append(d90)
        N.append(n)
    
    Kavg = np.array(Kavg)
    Kstd = np.array(Kstd)
    D10 = np.array(D10)
    D90 = np.array(D90)
    N = np.array(N)
    Kste = Kstd / (N**0.5)
    
    alpha = 0.975
    dof = N
    q = st.t.ppf(alpha, dof) # Student coefficient
    
    d_val = {'S' : fitCenters, 'Kavg' : Kavg, 'Kstd' : Kstd, 'D10' : D10, 'D90' : D90, 'N' : N}
    
    
    # Weighted means -- Weighted ste 95% as error
    ax[0].errorbar(fitCenters, Kavg, yerr = q*Kste, marker = 'o', color = cD[drug][0], 
                   ecolor = 'k', elinewidth = 0.8, capsize = 3, label = lD[drug])
    ax[0].set_ylim([500,2e4])
    ax[0].set_xlim([200,900])
    ax[0].set_title('Stress stiffening\nOnly compressions including the [300,800]Pa range')
    
    # Weighted means -- D9-D1 as error
    ax[1].errorbar(fitCenters, Kavg, yerr = [D10, D90], marker = 'o', color = cD[drug][1], 
                   ecolor = 'k', elinewidth = 0.8, capsize = 3, label = lD[drug]) 
    ax[1].set_ylim([500,2e4])
    ax[1].set_xlim([200,900])
    
    for k in range(2):
        ax[k].legend(loc = 'upper left')
        ax[k].set_yscale('log')
        ax[k].set_xlabel('Stress (Pa)')
        ax[k].set_ylabel('K (Pa)')
        for kk in range(len(N)):
            ax[k].text(x=fitCenters[kk]+oD[drug][0], y=Kavg[kk]**oD[drug][1], s='n='+str(N[kk]), fontsize = 6, color = cD[drug][k])
    
    fig.suptitle('K(s) - All compressions pooled'+'\n(fits width: {:.0f}Pa)'.format(width))
    
    df_val = pd.DataFrame(d_val)
    dftest = pd.DataFrame(d)

plt.show()

ufun.archiveFig(fig, ax, cp.DirDataFigToday + '//MCAproject', name='aSFL_ctrl-vs-doxy_K(s)_localZoom', dpi = 100)



# %%%%%


# print(data_f.head())

# print(fitCenters)

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

valStr = 'KChadwick_'
weightStr = 'K_Weight_'

Kavg = []
Kstd = []
D10 = []
D90 = []
N = []

for S in fitCenters:
    rFN = str(S-75) + '<s<' + str(S+75)
    variable = valStr+rFN
    weight = weightStr+rFN
    
    x = data_f[variable].apply(nan2zero).values
    w = data_f[weight].apply(nan2zero).values
    
    if S == 250:
        d = {'x' : x, 'w' : w}
    
    m = np.average(x, weights=w)
    v = np.average((x-m)**2, weights=w)
    std = v**0.5
    
    d10, d90 = np.percentile(x[x != 0], (10, 90))
    n = len(x[x != 0])
    
    Kavg.append(m)
    Kstd.append(std)
    D10.append(d10)
    D90.append(d90)
    N.append(n)

Kavg = np.array(Kavg)
Kstd = np.array(Kstd)
D10 = np.array(D10)
D90 = np.array(D90)
N = np.array(N)
Kste = Kstd / (N**0.5)

alpha = 0.975
dof = N
q = st.t.ppf(alpha, dof) # Student coefficient

d_val = {'S' : fitCenters, 'Kavg' : Kavg, 'Kstd' : Kstd, 'D10' : D10, 'D90' : D90, 'N' : N}

fig, ax = plt.subplots(1,1, figsize = (9,6)) # (2,1, figsize = (9,12))

# ax[0]
ax.errorbar(fitCenters, Kavg, yerr = q*Kste, marker = 'o', color = gs.my_default_color_list[0], 
               ecolor = 'k', elinewidth = 0.8, capsize = 3, label = 'Weighted means\nWeighted ste 95% as error')
ax.set_ylim([500,2e4])

# ax[1].errorbar(fitCenters, Kavg, yerr = [D10, D90], marker = 'o', color = gs.my_default_color_list[3], 
#                ecolor = 'k', elinewidth = 0.8, capsize = 3, label = 'Weighted means\nD9-D1 as error')
# ax[1].set_ylim([500,1e6])

# for k in range(1): #2
#     ax[k].legend(loc = 'upper left')
#     ax[k].set_yscale('log')
#     ax[k].set_xlabel('Stress (Pa) [center of a 150Pa large interval]')
#     ax[k].set_ylabel('K (Pa) [tangeant modulus w/ Chadwick]')
#     for kk in range(len(N)):
#         ax[k].text(x=fitCenters[kk]+5, y=Kavg[kk]**0.98, s='n='+str(N[kk]), fontsize = 6)
ax.legend(loc = 'upper left')
ax.set_yscale('log')
ax.set_xlabel('Stress (Pa) [center of a 150Pa large interval]')
ax.set_ylabel('K (Pa) [tangeant modulus w/ Chadwick]')
for kk in range(len(N)):
    ax.text(x=fitCenters[kk]+5, y=Kavg[kk]**0.98, s='n='+str(N[kk]), fontsize = 6)

fig.suptitle('K(sigma)')
ax.set_title('From all compressions pooled\n22-02-09 experiment, 36 cells, 232 compression')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Feb22_CompressionsLowStart_K(s)globalAvg_V2', figSubDir = 'NonLin')
plt.show()

# df_val = pd.DataFrame(d_val)
# dftest = pd.DataFrame(d)

# df_val


# %%%%%


# print(data_f.head())

# print(fitCenters)

extraFilters = [data_f['minStress'] <= 100, data_f['maxStress'] >= 800]
fitCenters2 = fitCenters[fitCenters<850]

data_ff = data_f
globalExtraFilter = extraFilters[0]
for k in range(1, len(extraFilters)):
    globalExtraFilter = globalExtraFilter & extraFilters[k]

data_ff = data_ff[globalExtraFilter]
data_ff
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

valStr = 'KChadwick_'
weightStr = 'K_Weight_'

Kavg = []
Kstd = []
D10 = []
D90 = []
N = []

for S in fitCenters2:
    rFN = str(S-75) + '<s<' + str(S+75)
    variable = valStr+rFN
    weight = weightStr+rFN
    
    x = data_ff[variable].apply(nan2zero).values
    w = data_ff[weight].apply(nan2zero).values
    
    if S == 250:
        d = {'x' : x, 'w' : w}
    
    m = np.average(x, weights=w)
    v = np.average((x-m)**2, weights=w)
    std = v**0.5
    
    d10, d90 = np.percentile(x[x != 0], (10, 90))
    n = len(x[x != 0])
    
    Kavg.append(m)
    Kstd.append(std)
    D10.append(d10)
    D90.append(d90)
    N.append(n)

Kavg = np.array(Kavg)
Kstd = np.array(Kstd)
D10 = np.array(D10)
D90 = np.array(D90)
N = np.array(N)
Kste = Kstd / (N**0.5)

alpha = 0.975
dof = N
q = st.t.ppf(alpha, dof) # Student coefficient

d_val = {'S' : fitCenters, 'Kavg' : Kavg, 'Kstd' : Kstd, 'D10' : D10, 'D90' : D90, 'N' : N}

fig, ax = plt.subplots(1,1, figsize = (9,6)) # (2,1, figsize = (9,12))

# ax[0]
ax.errorbar(fitCenters2, Kavg, yerr = q*Kste, marker = 'o', color = gs.my_default_color_list[0], 
               ecolor = 'k', elinewidth = 0.8, capsize = 3, label = 'Weighted means\nWeighted ste 95% as error')
ax.set_ylim([500,2e4])

# ax[1].errorbar(fitCenters, Kavg, yerr = [D10, D90], marker = 'o', color = gs.my_default_color_list[3], 
#                ecolor = 'k', elinewidth = 0.8, capsize = 3, label = 'Weighted means\nD9-D1 as error')
# ax[1].set_ylim([500,1e6])

# for k in range(1): #2
#     ax[k].legend(loc = 'upper left')
#     ax[k].set_yscale('log')
#     ax[k].set_xlabel('Stress (Pa) [center of a 150Pa large interval]')
#     ax[k].set_ylabel('K (Pa) [tangeant modulus w/ Chadwick]')
#     for kk in range(len(N)):
#         ax[k].text(x=fitCenters[kk]+5, y=Kavg[kk]**0.98, s='n='+str(N[kk]), fontsize = 6)
ax.legend(loc = 'upper left')
ax.set_yscale('log')
ax.set_xlabel('Stress (Pa) [center of a 150Pa large interval]')
ax.set_ylabel('K (Pa) [tangeant modulus w/ Chadwick]')
for kk in range(len(N)):
    ax.text(x=fitCenters2[kk]+5, y=Kavg[kk]**0.98, s='n='+str(N[kk]), fontsize = 6)

fig.suptitle('K(sigma)')
ax.set_title('Only compressions including the [100, 800]Pa range')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Feb22_CompressionsLowStart_K(s)globalAvg_100to800', figSubDir = 'NonLin')
plt.show()

# df_val = pd.DataFrame(d_val)
# dftest = pd.DataFrame(d)

# df_val


# %%%%%


# print(data_f.head())

# print(fitCenters)

extraFilters = [data_f['minStress'] <= 100, data_f['maxStress'] >= 600]
fitCenters2 = fitCenters[fitCenters<650]

data_ff = data_f
globalExtraFilter = extraFilters[0]
for k in range(1, len(extraFilters)):
    globalExtraFilter = globalExtraFilter & extraFilters[k]

data_ff = data_ff[globalExtraFilter]
data_ff
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

valStr = 'KChadwick_'
weightStr = 'K_Weight_'

Kavg = []
Kstd = []
D10 = []
D90 = []
N = []

for S in fitCenters2:
    rFN = str(S-75) + '<s<' + str(S+75)
    variable = valStr+rFN
    weight = weightStr+rFN
    
    x = data_ff[variable].apply(nan2zero).values
    w = data_ff[weight].apply(nan2zero).values
    
    if S == 250:
        d = {'x' : x, 'w' : w}
    
    m = np.average(x, weights=w)
    v = np.average((x-m)**2, weights=w)
    std = v**0.5
    
    d10, d90 = np.percentile(x[x != 0], (10, 90))
    n = len(x[x != 0])
    
    Kavg.append(m)
    Kstd.append(std)
    D10.append(d10)
    D90.append(d90)
    N.append(n)

Kavg = np.array(Kavg)
Kstd = np.array(Kstd)
D10 = np.array(D10)
D90 = np.array(D90)
N = np.array(N)
Kste = Kstd / (N**0.5)

alpha = 0.975
dof = N
q = st.t.ppf(alpha, dof) # Student coefficient

d_val = {'S' : fitCenters, 'Kavg' : Kavg, 'Kstd' : Kstd, 'D10' : D10, 'D90' : D90, 'N' : N}

fig, ax = plt.subplots(1,1, figsize = (9,6)) # (2,1, figsize = (9,12))

# ax[0]
ax.errorbar(fitCenters2, Kavg, yerr = q*Kste, marker = 'o', color = gs.my_default_color_list[0], 
               ecolor = 'k', elinewidth = 0.8, capsize = 3, label = 'Weighted means\nWeighted ste 95% as error')
ax.set_ylim([500,2e4])

# ax[1].errorbar(fitCenters, Kavg, yerr = [D10, D90], marker = 'o', color = gs.my_default_color_list[3], 
#                ecolor = 'k', elinewidth = 0.8, capsize = 3, label = 'Weighted means\nD9-D1 as error')
# ax[1].set_ylim([500,1e6])

# for k in range(1): #2
#     ax[k].legend(loc = 'upper left')
#     ax[k].set_yscale('log')
#     ax[k].set_xlabel('Stress (Pa) [center of a 150Pa large interval]')
#     ax[k].set_ylabel('K (Pa) [tangeant modulus w/ Chadwick]')
#     for kk in range(len(N)):
#         ax[k].text(x=fitCenters[kk]+5, y=Kavg[kk]**0.98, s='n='+str(N[kk]), fontsize = 6)
ax.legend(loc = 'upper left')
ax.set_yscale('log')
ax.set_xlabel('Stress (Pa) [center of a 150Pa large interval]')
ax.set_ylabel('K (Pa) [tangeant modulus w/ Chadwick]')
for kk in range(len(N)):
    ax.text(x=fitCenters2[kk]+5, y=Kavg[kk]**0.98, s='n='+str(N[kk]), fontsize = 6)

fig.suptitle('K(sigma)')
ax.set_title('Only compressions including the [100, 600]Pa range')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Feb22_CompressionsLowStart_K(s)globalAvg_100to600', figSubDir = 'NonLin')
plt.show()

# df_val = pd.DataFrame(d_val)
# dftest = pd.DataFrame(d)

# df_val


Kavg_ref, Kste_ref, N_ref, fitCenters_ref = Kavg, Kste, N, fitCenters2

# %%% MCA project round 3 : long linker july 2022

# %%%% All conditions first plot

data = GlobalTable_meca_MCA3
dates = ['22-07-15', '22-07-20', '22-07-27']

StressRegion = '_S=300+/-100'
srs = StressRegion.split('+/-')[0]+'-'+StressRegion.split('+/-')[1]

Filters = [(data['validatedFit'+StressRegion] == True), 
           (data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['bead type'] == 'M450'),
           (data['bestH0'] <= 900),
           (data['date'].apply(lambda x : x in dates))]


# print(data['cell subtype'].unique())

co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'],['none','doxycyclin'])
box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
           ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
           ['aSFL-E4 & none','aSFL-E4 & doxycyclin'],
           ['aSFL-A11 & none','aSFL-F8 & none'],
           ['aSFL-A11 & none','aSFL-E4 & none']]

fig, ax, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['bestH0', 'K'+StressRegion],Filters=Filters,
                          AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 0.8, markersizeFactor=0.8, orientation = 'v', stressBoxPlot= True,
                          returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('MCA3 summary plot\nK = ' + StressRegion[1:] + ' Pa')

ufun.archiveFig(fig, name=('MCA3_H0 & K' + srs + '_allComps'), figDir = 'MCA3_project', dpi = 100)


fig, ax, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['bestH0', 'K'+StressRegion],Filters=Filters,
                          AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 0.8, markersizeFactor=0.8, orientation = 'v', stressBoxPlot= False, 
                          returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('MCA3 summary plot\nK = ' + StressRegion[1:] + ' Pa')

ufun.archiveFig(fig, name=('MCA3_H0 & K' + srs + '_cellAvg'), figDir = 'MCA3_project', dpi = 100)

plt.show()

# %%%% Six conditions only H0

data = GlobalTable_meca_MCA123
dates = ['22-07-15', '22-07-20', '22-07-27']


Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['bead type'] == 'M450'),
           (data['bestH0'] <= 900),
           (data['date'].apply(lambda x : x in dates))]



co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'],['none','doxycyclin'])
box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
           ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
           ['aSFL-E4 & none','aSFL-E4 & doxycyclin'],
           ['aSFL-A11 & none','aSFL-F8 & none'],
           ['aSFL-A11 & none','aSFL-E4 & none']]

fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['bestH0'],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=True,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('MCA3 - H0 from fits - allComps')

ufun.archiveFig(fig, name=('MCA3_H0 only' + '_allComps'), figDir = 'MCA3_project', dpi = 100)


fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['bestH0'],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('MCA3 - H0 from fits - cellAvg')

ufun.archiveFig(fig, name=('MCA3_H0 only' + '_cellAvg'), figDir = 'MCA3_project', dpi = 100)


plt.show()

# %%%% Six conditions only H0 - ratio

data = GlobalTable_meca_MCA123
dates = ['22-07-15', '22-07-20', '22-07-27']


Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['bead type'] == 'M450'),
           (data['bestH0'] <= 900),
           (data['date'].apply(lambda x : x in dates))]



co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'],['none','doxycyclin'])
box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
           ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
           ['aSFL-E4 & none','aSFL-E4 & doxycyclin'],
           ['aSFL-A11 & none','aSFL-F8 & none'],
           ['aSFL-A11 & none','aSFL-E4 & none']]

normalizeGroups = [['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
           ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
           ['aSFL-E4 & none','aSFL-E4 & doxycyclin']]

fig, ax, dfcount, dfcountcells = D1Plot_wNormalize(data, CondCols=['cell subtype','drug'], Parameters=['bestH0'],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 normalizeCol = ['cell subtype','drug'], normalizeGroups=normalizeGroups,
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=True,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('MCA3 - H0 from fits (r) - allComps')

ufun.archiveFig(fig, name=('MCA3_H0 only_ratio' + '_allComps'), figDir = 'MCA3_project', dpi = 100)


fig, ax, dfcount, dfcountcells = D1Plot_wNormalize(data, CondCols=['cell subtype','drug'], Parameters=['bestH0'],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 normalizeCol = ['cell subtype','drug'], normalizeGroups=normalizeGroups,
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('MCA3 - H0 from fits (r) - cellAvg')

ufun.archiveFig(fig, name=('MCA3_H0 only_ratio' + '_cellAvg'), figDir = 'MCA3_project', dpi = 100)


plt.show()

# %%%% Six conditions only surroundingThickness

# data = GlobalTable_meca_MCA3
# dates = ['22-07-15', '22-07-20', '22-07-27']


# Filters = [(data['validatedThickness'] == True), 
#            (data['UI_Valid'] == True),
#            (data['cell type'] == '3T3'), 
#            (data['bead type'] == 'M450'),
#            (data['bestH0'] <= 900),
#            (data['date'].apply(lambda x : x in dates))]



# co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'],['none','doxycyclin'])
# box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
#            ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
#            ['aSFL-E4 & none','aSFL-E4 & doxycyclin'],
#            ['aSFL-A11 & none','aSFL-F8 & none'],
#            ['aSFL-A11 & none','aSFL-E4 & none']]

# fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['surroundingThickness'],Filters=Filters,
#                  AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
#                  box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=True,
#                  returnCount = 2)

# renameAxes(ax,renameDict_MCA3)
# fig.suptitle('MCA3 surroundingThickness')

# ufun.archiveFig(fig, name=('MCA3_SurH only' + '_allComps'), figDir = 'MCA3_project', dpi = 100)


# fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['surroundingThickness'],Filters=Filters,
#                  AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
#                  box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
#                  returnCount = 2)

# renameAxes(ax,renameDict_MCA3)
# fig.suptitle('MCA3 surroundingThickness')

# ufun.archiveFig(fig, name=('MCA3_SurH only' + '_cellAvg'), figDir = 'MCA3_project', dpi = 100)


# plt.show()

# %%%% Six conditions only ctFieldThickness

data = GlobalTable_meca_MCA3
dates = ['22-07-15', '22-07-20', '22-07-27']


Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['bead type'] == 'M450'),
           (data['bestH0'] <= 800),
           (data['date'].apply(lambda x : x in dates))]



co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'],['none','doxycyclin'])
box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
           ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
           ['aSFL-E4 & none','aSFL-E4 & doxycyclin'],
           ['aSFL-A11 & none','aSFL-F8 & none'],
           ['aSFL-A11 & none','aSFL-E4 & none']]



fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['ctFieldThickness'],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('MCA3 - median thickness')

ufun.archiveFig(fig, name=('MCA3_ctFieldH only' + '_cellAvg'), figDir = 'MCA3_project', dpi = 100)


plt.show()


# %%%% Six conditions only ctFieldThickness - Ratio

data = GlobalTable_meca_MCA3
dates = ['22-07-15', '22-07-20', '22-07-27']


Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['bead type'] == 'M450'),
           (data['ctFieldThickness'] <= 800),
           (data['date'].apply(lambda x : x in dates))]



co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'],['none','doxycyclin'])
box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
           ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
           ['aSFL-E4 & none','aSFL-E4 & doxycyclin'],
           ['aSFL-A11 & none','aSFL-F8 & none'],
           ['aSFL-A11 & none','aSFL-E4 & none']]

normalizeGroups = [['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
           ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
           ['aSFL-E4 & none','aSFL-E4 & doxycyclin']]

# fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['ctFieldThickness'],Filters=Filters,
#                  AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
#                  box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
#                  returnCount = 2)

fig, ax, dfcount, dfcountcells = D1Plot_wNormalize(data, CondCols=['cell subtype','drug'], Parameters=['ctFieldThickness'],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 normalizeCol = ['cell subtype','drug'], normalizeGroups=normalizeGroups,
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('MCA3 - median thickness (r)')

ufun.archiveFig(fig, name=('MCA3_ctFieldH only_ratio' + '_cellAvg'), figDir = 'MCA3_project', dpi = 100)


plt.show()

# %%%% Six conditions only H0 INTRA-ExpDay

data = GlobalTable_meca_MCA3
dates = ['22-07-15', '22-07-20', '22-07-27']
pairs = [['aSFL-A11','aSFL-F8'],['aSFL-A11','aSFL-E4'],['aSFL-F8','aSFL-E4']]


for i in range(len(dates)):
    D = dates[i]
    P = pairs[i]

    Filters = [(data['validatedThickness'] == True), 
               (data['UI_Valid'] == True),
               (data['cell type'] == '3T3'), 
               (data['bead type'] == 'M450'),
               (data['bestH0'] <= 800),
               (data['bestH0'] >= 100),
               (data['date'] == D)]
    
    
    
    co_order = makeOrder(P,['none','doxycyclin'])

    
    fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['bestH0'],Filters=Filters,
                     AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                     box_pairs=[], figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=True,
                     returnCount = 2)
    
    renameAxes(ax,renameDict_MCA3)
    fig.suptitle('MCA3 - ' + D)
    
    ufun.archiveFig(fig, name=('MCA3_' + D + '_allComps'), figDir = 'MCA3_project', dpi = 100)
    
    
    fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['bestH0'],Filters=Filters,
                     AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                     box_pairs=[], figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                     returnCount = 2)
    
    renameAxes(ax,renameDict_MCA3)
    fig.suptitle('MCA3 - ' + D)
    
    ufun.archiveFig(fig, name=('MCA3_' + D + '_cellAvg'), figDir = 'MCA3_project', dpi = 100)


plt.show()


# %%%% Six conditions only H0 INTER-ExpDay

data = GlobalTable_meca_MCA3
dates = ['22-07-15', '22-07-20', '22-07-27']


# 1


Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['bead type'] == 'M450'),
           (data['drug'] == 'none'),
           (data['bestH0'] <= 800),
           (data['date'].apply(lambda x : x in dates))]


co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'], ['22-07-15', '22-07-20', '22-07-27'])
# co_order.remove('22-07-15 & aSFL-E4')
# co_order.remove('22-07-20 & aSFL-F8')
# co_order.remove('22-07-27 & aSFL-A11')

box_pairs = [['aSFL-A11 & 22-07-15', 'aSFL-A11 & 22-07-20'], 
             ['aSFL-F8 & 22-07-15', 'aSFL-F8 & 22-07-27'], 
             ['aSFL-E4 & 22-07-20', 'aSFL-E4 & 22-07-27']]

fig, ax, dfcount = D1Plot(data, CondCols=['cell subtype','date'], Parameters=['bestH0'],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('3T3aSFL non activated, by dates')

ufun.archiveFig(fig, name=('MCA3_H0_noDrug_dates_allComps'), figDir = 'MCA3_project', dpi = 100)

fig, ax, dfcount = D1Plot(data, CondCols=['cell subtype','date'], Parameters=['bestH0'],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('3T3aSFL non activated, by dates')

ufun.archiveFig(fig, name=('MCA3_H0_noDrug_dates_cellAvg'), figDir = 'MCA3_project', dpi = 100)

# 2

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['bead type'] == 'M450'),
           (data['drug'] == 'doxycyclin'),
           (data['bestH0'] <= 800),
           (data['date'].apply(lambda x : x in dates))]


co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'], ['22-07-15', '22-07-20', '22-07-27'])

box_pairs = [['aSFL-A11 & 22-07-15', 'aSFL-A11 & 22-07-20'], 
             ['aSFL-F8 & 22-07-15', 'aSFL-F8 & 22-07-27'], 
             ['aSFL-E4 & 22-07-20', 'aSFL-E4 & 22-07-27']]

fig, ax, dfcount = D1Plot(data, CondCols=['cell subtype','date'], Parameters=['bestH0'],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('3T3aSFL activated, by dates')

ufun.archiveFig(fig, name=('MCA3_H0_doxy_dates_allComps'), figDir = 'MCA3_project', dpi = 100)

fig, ax, dfcount = D1Plot(data, CondCols=['cell subtype','date'], Parameters=['bestH0'],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('3T3aSFL activated, by dates')

ufun.archiveFig(fig, name=('MCA3_H0_doxy_dates_cellAvg'), figDir = 'MCA3_project', dpi = 100)

plt.show()





# %%%% TO ADAPT Multiple boxplot K(s)

data = GlobalTable_meca_MCA3
dates = ['22-07-15', '22-07-20', '22-07-27']



# fig, ax, dfcount = D1Plot(data, CondCols=['substrate','cell subtype'], Parameters=['bestH0', 'KChadwick'+StressRegion],Filters=Filters,
#                           AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
#                           box_pairs=[], figSizeFactor = 0.8, markersizeFactor=0.8, orientation = 'v', returnCount = 1)

CondCol=['cell subtype', 'drug']
co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'],['none','doxycyclin'])
NCondCol = len(CondCol)
NCond = len(co_order)


listS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# listS = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
width = 150

fig, axes = plt.subplots(NCond, len(listS), figsize = (18,12))

for ii in range(NCond):
    co = co_order[ii]
    co_split = co.split(' & ')
        
    
    for kk in range(len(listS)):
        S = listS[kk]
        interval = 'S={:.0f}+/-{:.0f}'.format(S, width//2)
        fileNameInterval = 'S={:.0f}-{:.0f}'.format(S, width//2)
    
        Filters = [(data['validatedFit_'+interval] == True),
                   (data['validatedThickness'] == True),
                   (data['UI_Valid'] == True),
                   (data['bestH0'] <= 800),
                   (data['date'].apply(lambda x : x in dates))]
        
        for ll in range(NCondCol):
            Filters += [(data[CondCol[ll]] == co_split[ll])]
        
        # co_order = makeOrder(['bare glass','20um fibronectin discs'], ['ctrl','tko'])
    
        fig, ax = D1Plot(data, fig=fig, ax=axes[ii,kk], CondCol=CondCol, Parameters=['K_' + interval], 
                         Filters=Filters, Boxplot=True, cellID='cellID', co_order=copy(co_order), stats=False, statMethod='Mann-Whitney', 
                         AvgPerCell = False, box_pairs=[], figSizeFactor = 1, markersizeFactor=1, orientation = 'h', stressBoxPlot=True)
    
    #     axes[kk].legend(loc = 'upper right', fontsize = 8)
        label = axes[ii,kk].get_ylabel()
        # axes[ii,kk].set_title(label.split('_')[-1] + 'Pa', fontsize = 10)
        axes[ii,kk].set_yscale('log')
        axes[ii,kk].set_ylim([4e2, 2e4])
        axes[ii,kk].tick_params(axis='x', labelrotation = 50, labelsize = 10)
        axes[ii,kk].set_xticklabels([])
        if ii == 0:
            axes[ii,kk].set_title(interval + ' Pa', fontsize = 10)
        if ii == len(co_order)-1:
            axes[ii,kk].tick_params(axis='x', labelsize = 10)
        else:
            axes[ii,kk].set_xlabel(None)
            axes[ii,kk].set_xticklabels([])
            
        if kk == 0:
            # axes[ii,kk].set_ylabel(label.split('_')[0] + ' (Pa)')
            # axes[ii,kk].set_ylabel(co, fontsize = 10)
            axes[ii,kk].set_ylabel(co + '\n' + label.split('_')[0] + ' (Pa)', 
                                   fontsize = 11, color = styleDict1[co]['color'])
        else:
            axes[ii,kk].set_ylabel(None)
            axes[ii,kk].set_yticklabels([])


fig.tight_layout()
fig.suptitle('')
plt.show()

ufun.archiveFig(fig, name='MCA3_K(s)_MultiBoxPlots_to1000'.format(Sinf, Ssup), 
                figDir = 'MCA3_project', dpi = 100)



# %%%% K(s) for MCA3

#### Making the Dataframe 

data = GlobalTable_meca_MCA3

data['MCA3_Co'] = data['cell subtype'].values + \
                   np.array([' & ' for i in range(data.shape[0])]) + \
                   data['drug'].values

dates = ['22-07-15', '22-07-20', '22-07-27']

filterList = [(data['validatedThickness'] == True),
              (data['cell type'] == '3T3'), 
              (data['bead type'] == 'M450'),
              (data['UI_Valid'] == True),
              (data['bestH0'] <= 800),
              (data['date'].apply(lambda x : x in dates))]  # (data['validatedFit'] == True),

globalFilter = filterList[0]
for k in range(1, len(filterList)):
    globalFilter = globalFilter & filterList[k]

data_f = data[globalFilter]

# data_f['HoxB8_Co'] = data_f['substrate'].values + \
#                      np.array([' & ' for i in range(data_f.shape[0])]) + \
#                      data_f['cell subtype'].values

width = 150 # 200
fitCenters =  np.array([S for S in range(100, 1100, 50)])
# fitMin = np.array([int(S-(width/2)) for S in fitCenters])
# fitMax = np.array([int(S+(width/2)) for S in fitCenters])

# regionFitsNames = [str(fitMin[ii]) + '<s<' + str(fitMax[ii]) for ii in range(len(fitMin))]
regionFitsNames = ['S={:.0f}+/-{:.0f}'.format(fitCenters[ii], width//2) for ii in range(len(fitCenters))]

listColumnsMeca = []

KChadwick_Cols = []
KWeight_Cols = []

for rFN in regionFitsNames:
    listColumnsMeca += ['K_'+rFN, 'K_CIW_'+rFN, 'R2_'+rFN, 
                        'Npts_'+rFN, 'validatedFit_'+rFN]
    KChadwick_Cols += [('K_'+rFN)]

    K_CIWidth = data_f['K_CIW_'+rFN] #.apply(lambda x : x.strip('][').split(', ')).apply(lambda x : (np.abs(float(x[0]) - float(x[1]))))
    KWeight = (data_f['K_'+rFN]/K_CIWidth)**2
    data_f['K_Weight_'+rFN] = KWeight
    data_f['K_Weight_'+rFN] *= data_f['K_'+rFN].apply(lambda x : (x<1e6))
    data_f['K_Weight_'+rFN] *= data_f['R2_'+rFN].apply(lambda x : (x>1e-2))
    data_f['K_Weight_'+rFN] *= data_f['K_CIW_'+rFN].apply(lambda x : (x!=0))
    KWeight_Cols += [('K_Weight_'+rFN)]
    

#### Useful functions

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


#### Whole curve


valStr = 'K_'
weightStr = 'K_Weight_'



# regionFitsNames = [str(fitMin[ii]) + '<s<' + str(fitMax[ii]) for ii in range(len(fitMin))]
regionFitsNames = ['S={:.0f}+/-{:.0f}'.format(fitCenters[ii], width//2) for ii in range(len(fitCenters))]

fig, axes = plt.subplots(2,1, figsize = (9,12))
ListDfWhole = []

conditions = np.array(data_f['MCA3_Co'].unique())


cD = {co: [styleDict_MCA3[co]['color'], styleDict_MCA3[co]['color']] for co in conditions}

# oD = {'none': [-15, 1.02] , 'doxycyclin': [5, 0.97] }
# lD = {'naked glass & ctrl':', 
#       'naked glass & tko':'',
#       '20um fibronectin discs & tko':'',
#       '20um fibronectin discs & ctrl':''}

for co in conditions:
    
    Kavg = []
    Kstd = []
    D10 = []
    D90 = []
    N = []
    
    data_ff = data_f[data_f['MCA3_Co'] == co]
    
    
    for ii in range(len(fitCenters)):
        S = fitCenters[ii]
        rFN = 'S={:.0f}+/-{:.0f}'.format(S, width//2)
        variable = valStr+rFN
        weight = weightStr+rFN
        
        x = data_ff[variable].apply(nan2zero).values
        w = data_ff[weight].apply(nan2zero).values
        
        if S == 250:
            d = {'x' : x, 'w' : w}
        
        m = np.average(x, weights=w)
        v = np.average((x-m)**2, weights=w)
        std = v**0.5
        
        d10, d90 = np.percentile(x[x != 0], (10, 90))
        n = len(x[x != 0])
        
        Kavg.append(m)
        Kstd.append(std)
        D10.append(d10)
        D90.append(d90)
        N.append(n)
    
    Kavg = np.array(Kavg)
    Kstd = np.array(Kstd)
    D10 = np.array(D10)
    D90 = np.array(D90)
    N = np.array(N)
    Kste = Kstd / (N**0.5)
    
    alpha = 0.975
    dof = N
    q = st.t.ppf(alpha, dof) # Student coefficient
    
    d_val = {'S' : fitCenters, 'Kavg' : Kavg, 'Kstd' : Kstd, 'D10' : D10, 'D90' : D90, 'N' : N}
    
    for ax in axes:
        # Weighted means -- Weighted ste 95% as error
        ax.errorbar(fitCenters, Kavg, yerr = q*Kste, marker = 'o', color = cD[co][0], 
                       ecolor = cD[co][1], elinewidth = 0.8, capsize = 3, label = co)
        ax.set_ylim([500,2e4])
        ax.set_xlim([0,1100])
        ax.set_title('K(s) - All compressions pooled')
        
        ax.legend(loc = 'upper left')
        
        ax.set_xlabel('Stress (Pa)')
        ax.set_ylabel('K (Pa)')
        
        for kk in range(len(N)):
            ax.text(x=fitCenters[kk], y=Kavg[kk]**0.9, s='n='+str(N[kk]), 
                    fontsize = 8, color = cD[co][1])
        # for kk in range(len(N)):
        #     ax[k].text(x=fitCenters[kk]+oD[co][0], y=Kavg[kk]**oD[co][1], 
        #                s='n='+str(N[kk]), fontsize = 6, color = cD[co][k])
        
    axes[0].set_yscale('log')
    axes[0].set_ylim([500,2e4])
    
    axes[1].set_ylim([0,1.4e4])
    
    fig.suptitle('K(s)'+'\n(fits width: {:.0f}Pa)'.format(width))    
    
    df_val = pd.DataFrame(d_val)
    ListDfWhole.append(df_val)
    dftest = pd.DataFrame(d)

plt.show()

ufun.archiveFig(fig, name='MCA3_K(s)', figDir = 'MCA3_project', dpi = 100)



#### Local zoom

Sinf, Ssup = 100, 600
extraFilters = [data_f['minStress'] <= Sinf, data_f['maxStress'] >= Ssup] # >= 800
fitCenters = fitCenters[(fitCenters>=(Sinf)) & (fitCenters<=Ssup)] # <800
# fitMin = np.array([int(S-(width/2)) for S in fitCenters])
# fitMax = np.array([int(S+(width/2)) for S in fitCenters])

data2_f = data_f
globalExtraFilter = extraFilters[0]
for k in range(1, len(extraFilters)):
    globalExtraFilter = globalExtraFilter & extraFilters[k]
data2_f = data2_f[globalExtraFilter]  


fig, axes = plt.subplots(2,1, figsize = (9,12))

conditions = np.array(data_f['MCA3_Co'].unique())
print(conditions)


cD = {co: [styleDict_MCA3[co]['color'], styleDict_MCA3[co]['color']] for co in conditions}

listDfZoom = []

for co in conditions:
    
    Kavg = []
    Kstd = []
    D10 = []
    D90 = []
    N = []
    
    data_ff = data2_f[data_f['MCA3_Co'] == co]
    
    for ii in range(len(fitCenters)):
        S = fitCenters[ii]
        rFN = 'S={:.0f}+/-{:.0f}'.format(S, width//2)
        S = fitCenters[ii]
        variable = valStr+rFN
        weight = weightStr+rFN
        
        x = data_ff[variable].apply(nan2zero).values
        w = data_ff[weight].apply(nan2zero).values
        
        print(S)
        print(np.sum(w))
        
        m = np.average(x, weights=w)
        v = np.average((x-m)**2, weights=w)
        std = v**0.5
        
        d10, d90 = np.percentile(x[x != 0], (10, 90))
        n = len(x[x != 0])
        
        Kavg.append(m)
        Kstd.append(std)
        D10.append(d10)
        D90.append(d90)
        N.append(n)
    
    Kavg = np.array(Kavg)
    Kstd = np.array(Kstd)
    D10 = np.array(D10)
    D90 = np.array(D90)
    N = np.array(N)
    Kste = Kstd / (N**0.5)
    
    alpha = 0.975
    dof = N
    q = st.t.ppf(alpha, dof) # Student coefficient
    
    d_val = {'S' : fitCenters, 'Kavg' : Kavg, 'Kstd' : Kstd, 'D10' : D10, 'D90' : D90, 'N' : N}
    
    for ax in axes:
        # Weighted means -- Weighted ste 95% as error
        ax.errorbar(fitCenters, Kavg, yerr = q*Kste, marker = 'o', color = cD[co][0], 
                        ecolor = cD[co][1], elinewidth = 0.8, capsize = 3, label = co)
        
        ax.set_xlim([Sinf-50, Ssup+50])
        ax.set_title('K(s)\nOnly compressions including the [{:.0f},{:.0f}]Pa range'.format(Sinf, Ssup))
        
        # Weighted means -- D9-D1 as error
        # ax[1].errorbar(fitCenters, Kavg, yerr = [D10, D90], marker = 'o', color = cD[co][1], 
        #                 ecolor = 'k', elinewidth = 0.8, capsize = 3, label = lD[co]) 
        # ax[1].set_ylim([500,2e4])
        # ax[1].set_xlim([200,900])
        
        # for k in range(2):
        ax.legend(loc = 'upper left')
        
        ax.set_xlabel('Stress (Pa)')
        ax.set_ylabel('K (Pa)')
        for kk in range(len(N)):
            ax.text(x=fitCenters[kk], y=Kavg[kk]**0.9, s='n='+str(N[kk]), fontsize = 8, color = cD[co][1])
    
    axes[0].set_yscale('log')
    axes[0].set_ylim([500,2e4])
    
    axes[1].set_ylim([0,15000])
    
    fig.suptitle('K(s)'+' - (fits width: {:.0f}Pa)'.format(width))
    
    df_val = pd.DataFrame(d_val)
    listDfZoom.append(df_val)

plt.show()

ufun.archiveFig(fig, name='MCA3_K(s)_localZoom_{:.0f}-{:.0f}Pa'.format(Sinf, Ssup), 
                figDir = 'MCA3_project', dpi = 100)










# %%% MCA project rounds 1, 2, 3 - comparison of all experiments



# %%%% Fluo analysis

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3


Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['bead type'] == 'M450'),
           (data['drug'] == 'doxycyclin'),
           (data['bestH0'] <= 900),
           (data['date'].apply(lambda x : x in (dates_r1 + dates_r2))),
           (data['meanFluoPeakAmplitude'] >= 0),
           (data['meanFluoPeakAmplitude'] <= 1500),
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11', 'aSFL-F8', 'aSFL-E4']))]


# print(data['cell subtype'].unique())

co_order, box_pairs = ['aSFL-A11', 'aSFL-F8', 'aSFL-E4'], [['aSFL-F8', 'aSFL-E4']]
co_order, box_pairs = ['aSFL-A11', 'aSFL-F8', 'aSFL-E4'], []

# co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'],['none','doxycyclin'])
# box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
#            ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
#            ['aSFL-E4 & none','aSFL-E4 & doxycyclin'],
#            ['aSFL-A11 & none','aSFL-F8 & none'],
#            ['aSFL-A11 & none','aSFL-E4 & none']]


fig, ax, dfcount = D1Plot(data, CondCols=['cell subtype'], Parameters=['meanFluoPeakAmplitude'],Filters=Filters,
                          AvgPerCell=True, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor=1.5, orientation = 'v', stressBoxPlot= False,
                          returnCount = 1)

ax[0].plot(ax[0].get_xlim(), [200, 200], 'c--')
ax[0].plot(ax[0].get_xlim(), [500, 500], 'r--')

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA fluo plot (2021)')

ufun.archiveFig(fig, name=('MCA123_fluoPlot_oldExpts'), figDir = 'MCA_project_123', dpi = 100)


plt.show()


#### Get a 'categorical fluo column'

def categoriesFluoColumn(df):
    filter01 = df['UI_Fluo'].isnull()
    df.loc[filter01,'UI_Fluo'] = 'none'
    
    filter02 = ((df['drug'] == 'doxycyclin') & (df['meanFluoPeakAmplitude'].apply(lambda x : not pd.isnull(x))))
    filter02_low  = (filter02 & (df['meanFluoPeakAmplitude'] < 200))
    filter02_mid  = (filter02 & (df['meanFluoPeakAmplitude'] > 200) & (df['meanFluoPeakAmplitude'] < 500))
    filter02_high = (filter02 & (df['meanFluoPeakAmplitude'] > 500))
    
    df.loc[filter02_low , 'UI_Fluo'] = 'low'
    df.loc[filter02_mid , 'UI_Fluo'] = 'mid'
    df.loc[filter02_high, 'UI_Fluo'] = 'high'
    
    return(df)

GlobalTable_meca_MCA123 = categoriesFluoColumn(GlobalTable_meca_MCA123)

def countFluoCat(df):
    fdf = df[df['drug'] == 'doxycyclin']
    
    # Simple count
    groupByCat = fdf.groupby('UI_Fluo')
    df_CountByCat = groupByCat.agg({'UI_Fluo':'count'})
    
    # Detailed count
    groupByCell = fdf.groupby('cellID')
    df_ByCell = groupByCell.agg({'cell subtype':'first', 'UI_Fluo':'first'})
    
    groupByType = df_ByCell.groupby(['cell subtype', 'UI_Fluo'])
    df_CountBySubtype = groupByType.agg({'UI_Fluo':'count'})
    

    return(df_CountBySubtype)

df_countFluo = countFluoCat(GlobalTable_meca_MCA123)

#### Test the new 1D plot

fig, ax, dfcount = D1Plot_wInnerSplit(data, CondCols=['cell subtype'], Parameters=['meanFluoPeakAmplitude'],Filters=Filters,
                                      InnerSplitCol = ['UI_Fluo'], 
                          AvgPerCell=True, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor=1.5, orientation = 'v', stressBoxPlot= False,
                          returnCount = 1)

ax[0].plot(ax[0].get_xlim(), [200, 200], 'c--')
ax[0].plot(ax[0].get_xlim(), [500, 500], 'r--')

ufun.archiveFig(fig, name=('MCA123_fluoPlot_V2_oldExpts'), figDir = 'MCA_project_123', dpi = 100)

plt.show()




# %%%% Only A11 only H0

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'bestH0' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           # (data['FluoSelection'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11'])), 
           (data['bead type'] == 'M450'),
           (data[thicknessType] >= 50),
           (data[thicknessType] <= 800),
           (data['date'].apply(lambda x : x in all_dates))]



# co_order, box_pairs = [], []

co_order = makeOrder(['aSFL-A11'],['MCA1','MCA3'],['none','doxycyclin'])
box_pairs=[['aSFL-A11 & MCA1 & none','aSFL-A11 & MCA1 & doxycyclin'],
            ['aSFL-A11 & MCA1 & none','aSFL-A11 & MCA3 & none'],
            ['aSFL-A11 & MCA3 & none','aSFL-A11 & MCA3 & doxycyclin']]

fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','round','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 0.7, orientation = 'v', stressBoxPlot=True,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - A11 - '+ thicknessType +' - allComps')

# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_A11-1vs3' + '_allComps'), figDir = 'MCA_project_123', dpi = 100)


fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','round','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - A11 - '+ thicknessType +' - cellAvg')

# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_A11-1vs3' + '_cellAvg'), figDir = 'MCA_project_123', dpi = 100)



co_order = makeOrder(['aSFL-A11'],['none','doxycyclin'])
box_pairs=[]

fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 0.7, orientation = 'v', stressBoxPlot=True,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - A11 - '+ thicknessType +' - allComps')

# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_A11-all' +  '_allComps'), figDir = 'MCA_project_123', dpi = 100)



fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - A11 - '+ thicknessType +' - cellAvg')

# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_A11-all' +  '_cellAvg'), figDir = 'MCA_project_123', dpi = 100)



plt.show()




# %%%% Only A11 only H0 _ normalization


data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11'])), 
           (data['bead type'] == 'M450'),
           (data[thicknessType] >= 50),
           (data[thicknessType] <= 800),
           (data['date'].apply(lambda x : x in all_dates))]



# co_order, box_pairs = [], []

co_order = makeOrder(['aSFL-A11'],['MCA1','MCA3'],['none','doxycyclin'])
box_pairs=[['aSFL-A11 & MCA1 & none','aSFL-A11 & MCA1 & doxycyclin'],
            ['aSFL-A11 & MCA3 & none','aSFL-A11 & MCA3 & doxycyclin']]

normalizeGroups = [['aSFL-A11 & MCA1 & none','aSFL-A11 & MCA1 & doxycyclin'],
                   ['aSFL-A11 & MCA3 & none','aSFL-A11 & MCA3 & doxycyclin']]

fig, ax, dfcount = D1Plot_wNormalize(data, CondCols=['cell subtype','round','drug'], Parameters=[thicknessType],Filters=Filters,
                                     AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                                     box_pairs=box_pairs, normalizeCol=['cell subtype','round','drug'], normalizeGroups=normalizeGroups,
                                     figSizeFactor = 0.6, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                                     returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - A11 - '+ thicknessType +' (r) - cellAvg')
ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_A11-1vs3' + '_ratio_cellAvg'), figDir = 'MCA_project_123', dpi = 100)




co_order = makeOrder(['aSFL-A11'],['none','doxycyclin'])
box_pairs=[]

fig, ax, dfcount = D1Plot_wNormalize(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                                     AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                                     box_pairs=box_pairs, normalizeCol=['cell subtype','round','drug'], normalizeGroups=normalizeGroups,
                                     figSizeFactor = 0.6, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                                     returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - A11 - '+ thicknessType +' (r) - cellAvg')
ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_A11-all' +  '_ratio_cellAvg'), figDir = 'MCA_project_123', dpi = 100)

plt.show()



# %%%% Only F8 only H0

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'bestH0' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['FluoSelection'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-F8'])), 
           (data['bead type'] == 'M450'),
           (data[thicknessType] >= 50),
           (data[thicknessType] <= 800),
           (data['date'].apply(lambda x : x in all_dates))]



# co_order, box_pairs = [], []

co_order = makeOrder(['aSFL-F8'],['MCA2','MCA3'],['none','doxycyclin'])
box_pairs=[['aSFL-F8 & MCA2 & none','aSFL-F8 & MCA2 & doxycyclin'],
            ['aSFL-F8 & MCA2 & none','aSFL-F8 & MCA3 & none'],
            ['aSFL-F8 & MCA3 & none','aSFL-F8 & MCA3 & doxycyclin']]

fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','round','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 0.7, orientation = 'v', stressBoxPlot=True,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - F8 - '+ thicknessType +' - allComps')

# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_F8-1vs3' + '_allComps'), figDir = 'MCA_project_123', dpi = 100)


fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','round','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - F8 - '+ thicknessType +' - cellAvg')

# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_F8-1vs3' + '_cellAvg'), figDir = 'MCA_project_123', dpi = 100)



co_order = makeOrder(['aSFL-F8'],['none','doxycyclin'])
box_pairs=[]

fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 0.7, orientation = 'v', stressBoxPlot=True,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - F8 - '+ thicknessType +' - allComps')

# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_F8-all' +  '_allComps'), figDir = 'MCA_project_123', dpi = 100)


co_order = makeOrder(['aSFL-F8'],['none','doxycyclin'])
box_pairs=[]

fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - F8 - '+ thicknessType +' - cellAvg')

# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_F8-all' +  '_cellAvg'), figDir = 'MCA_project_123', dpi = 100)



plt.show()




# %%%% Only F8 only H0 _ normalization


data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-F8'])), 
           (data['bead type'] == 'M450'),
           (data[thicknessType] >= 50),
           (data[thicknessType] <= 800),
           (data['date'].apply(lambda x : x in all_dates))]



# co_order, box_pairs = [], []

co_order = makeOrder(['aSFL-F8'],['MCA2','MCA3'],['none','doxycyclin'])
box_pairs=[['aSFL-F8 & MCA2 & none','aSFL-F8 & MCA2 & doxycyclin'],
            ['aSFL-F8 & MCA3 & none','aSFL-F8 & MCA3 & doxycyclin']]

normalizeGroups = [['aSFL-F8 & MCA2 & none','aSFL-F8 & MCA2 & doxycyclin'],
                   ['aSFL-F8 & MCA3 & none','aSFL-F8 & MCA3 & doxycyclin']]

fig, ax, dfcount = D1Plot_wNormalize(data, CondCols=['cell subtype','round','drug'], Parameters=[thicknessType],Filters=Filters,
                                     AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                                     box_pairs=box_pairs, normalizeCol=['cell subtype','round','drug'], normalizeGroups=normalizeGroups,
                                     figSizeFactor = 0.6, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=True,
                                     returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - F8 - '+ thicknessType +' (ratio) - cellAvg')
ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_F8-1vs3_ratio' + '_cellAvg'), figDir = 'MCA_project_123', dpi = 100)




co_order = makeOrder(['aSFL-F8'],['none','doxycyclin'])
box_pairs=[]

fig, ax, dfcount = D1Plot_wNormalize(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                                     AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                                     box_pairs=box_pairs, normalizeCol=['cell subtype','round','drug'], normalizeGroups=normalizeGroups,
                                     figSizeFactor = 0.6, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=True,
                                     returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - F8 - '+ thicknessType +' (ratio) - cellAvg')
ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_F8-all_ratio' +  '_cellAvg'), figDir = 'MCA_project_123', dpi = 100)

plt.show()


# %%%% Only E4 only H0

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'bestH0' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['FluoSelection'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data[thicknessType] >= 50),
           (data[thicknessType] <= 800),
           (data['date'].apply(lambda x : x in all_dates))]



co_order, box_pairs = [], []

co_order = makeOrder(['aSFL-E4'],['MCA1','MCA3'],['none','doxycyclin'])
box_pairs=[['aSFL-E4 & MCA1 & none','aSFL-E4 & MCA1 & doxycyclin'],
            ['aSFL-E4 & MCA1 & none','aSFL-E4 & MCA3 & none'],
            ['aSFL-E4 & MCA3 & none','aSFL-E4 & MCA3 & doxycyclin']]

fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','round','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 0.7, orientation = 'v', stressBoxPlot=True,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - E4 - '+ thicknessType +' - allComps')

# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_E4-1vs3' + '_allComps'), figDir = 'MCA_project_123', dpi = 100)


fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','round','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - E4 - '+ thicknessType +' - cellAvg')

# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_E4-1vs3' + '_cellAvg'), figDir = 'MCA_project_123', dpi = 100)



co_order = makeOrder(['aSFL-E4'],['none','doxycyclin'])
box_pairs=[]

fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 0.7, orientation = 'v', stressBoxPlot=True,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - E4 - '+ thicknessType +' - allComps')

# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_E4-all' +  '_allComps'), figDir = 'MCA_project_123', dpi = 100)


co_order = makeOrder(['aSFL-E4'],['none','doxycyclin'])
box_pairs=[]

fig, ax, dfcount, dfcountcells = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                 returnCount = 2)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - E4 - '+ thicknessType +' - cellAvg')

# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_E4-all' +  '_cellAvg'), figDir = 'MCA_project_123', dpi = 100)



plt.show()



# %%%% Only E4 only H0 _ normalization


data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data[thicknessType] >= 50),
           (data[thicknessType] <= 800),
           (data['date'].apply(lambda x : x in all_dates))]



# co_order, box_pairs = [], []

co_order = makeOrder(['aSFL-E4'],['MCA1','MCA3'],['none','doxycyclin'])
box_pairs=[['aSFL-E4 & MCA1 & none','aSFL-E4 & MCA1 & doxycyclin'],
            ['aSFL-E4 & MCA3 & none','aSFL-E4 & MCA3 & doxycyclin']]

normalizeGroups = [['aSFL-E4 & MCA1 & none','aSFL-E4 & MCA1 & doxycyclin'],
                   ['aSFL-E4 & MCA3 & none','aSFL-E4 & MCA3 & doxycyclin']]

fig, ax, dfcount = D1Plot_wNormalize(data, CondCols=['cell subtype','round','drug'], Parameters=[thicknessType],Filters=Filters,
                                     AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                                     box_pairs=box_pairs, normalizeCol=['cell subtype','round','drug'], normalizeGroups=normalizeGroups,
                                     figSizeFactor = 0.5, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                                     returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - E4 - '+ thicknessType +' (r) - cellAvg')
ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_E4-1vs3' + '_ratio_cellAvg'), figDir = 'MCA_project_123', dpi = 100)




co_order = makeOrder(['aSFL-E4'],['none','doxycyclin'])
box_pairs=[]

fig, ax, dfcount = D1Plot_wNormalize(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                                     AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                                     box_pairs=box_pairs, normalizeCol=['cell subtype','round','drug'], normalizeGroups=normalizeGroups,
                                     figSizeFactor = 0.5, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                                     returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - E4 - '+ thicknessType +' (r) - cellAvg')
ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_E4-all' +  '_ratio_cellAvg'), figDir = 'MCA_project_123', dpi = 100)

plt.show()



# %%%% All 3 clones only H0 _ normalization


data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
            (data['FluoSelection'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11','aSFL-F8','aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data[thicknessType] >= 50),
           (data[thicknessType] <= 500),
           (data['date'].apply(lambda x : x in all_dates))]





co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'],['none','doxycyclin'])

box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
            ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
            ['aSFL-E4 & none','aSFL-E4 & doxycyclin']]

normalizeGroups = [['aSFL-A11 & MCA1 & none','aSFL-A11 & MCA1 & doxycyclin'],
                   ['aSFL-A11 & MCA3 & none','aSFL-A11 & MCA3 & doxycyclin'],
                   ['aSFL-F8 & MCA2 & none','aSFL-F8 & MCA2 & doxycyclin'],
                   ['aSFL-F8 & MCA3 & none','aSFL-F8 & MCA3 & doxycyclin'],
                   ['aSFL-E4 & MCA1 & none','aSFL-E4 & MCA1 & doxycyclin'],
                   ['aSFL-E4 & MCA3 & none','aSFL-E4 & MCA3 & doxycyclin']]

fig, ax, dfcount = D1Plot_wNormalize(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                                     AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                                     box_pairs=box_pairs, normalizeCol=['cell subtype','round','drug'], normalizeGroups=normalizeGroups,
                                     figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                                     returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - All lines - '+ thicknessType +' (r) - cellAvg')
# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_ALLclones-all' +  '_ratio_cellAvg'), figDir = 'MCA_project_123', dpi = 100)




thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
            (data['FluoSelection'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11','aSFL-F8','aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data[thicknessType] >= 50),
           (data[thicknessType] <= 500),
           (data['date'].apply(lambda x : x in all_dates))]


fig, ax, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                                     AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                                     box_pairs=box_pairs, 
                                     figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                                     returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - All lines - '+ thicknessType +' - cellAvg')
# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_ALLclones-all' +  '_cellAvg'), figDir = 'MCA_project_123', dpi = 100)



thicknessType = 'bestH0' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
            (data['FluoSelection'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11','aSFL-F8','aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data[thicknessType] >= 50),
           (data[thicknessType] <= 800),
           (data['date'].apply(lambda x : x in all_dates))]


fig, ax, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                                     AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                                     box_pairs=box_pairs, 
                                     figSizeFactor = 1.0, markersizeFactor = 0.5, orientation = 'v', stressBoxPlot=True,
                                     returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - All lines - '+ thicknessType +' - allComps')
# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_ALLclones-all' +  '_callComps'), figDir = 'MCA_project_123', dpi = 100)





plt.show()




# %%%% A11 - Test fluo splitting

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['drug'] == 'doxycyclin'), 
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11'])), 
           (data['bead type'] == 'M450'),
           (data['UI_Fluo'].apply(lambda x : x not in ['none'])),
           (data['ctFieldThickness'] >= 50),
           (data['bestH0'] <= 800),
           (data['date'].apply(lambda x : x in all_dates))]

# co_order = makeOrder(['aSFL-A11'],['MCA1','MCA3'])
# box_pairs=[]

# fig, ax, dfcount = D1Plot_wInnerSplit(data, CondCols=['cell subtype','round'], Parameters=[thicknessType],Filters=Filters,
#                  InnerSplitCol=['UI_Fluo'],
#                  AvgPerCell=True, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
#                  box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 2, orientation = 'v', stressBoxPlot=False,
#                  returnCount = 1)

co_order = makeOrder(['aSFL-A11'],['MCA1','MCA3'],['none','low','mid','high'])
box_pairs=[]

fig, ax, dfcount = D1Plot(data, CondCols=['cell subtype','round','UI_Fluo'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 2, orientation = 'v', stressBoxPlot=False,
                 returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('fluoSplitting - A11 - '+ thicknessType +' - cellAvg')

ufun.archiveFig(fig, name=('MCA123_fluoSplitting_A11_' + thicknessType + '_perCell'), figDir = 'MCA_project_123', dpi = 100)


#
thicknessType = 'bestH0'

fig, ax, dfcount = D1Plot(data, CondCols=['cell subtype','round','UI_Fluo'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1, orientation = 'v', stressBoxPlot=False,
                 returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('fluoSplitting - A11 - '+ thicknessType +' - allComps')

ufun.archiveFig(fig, name=('MCA123_fluoSplitting_A11_' + thicknessType + '_allComps'), figDir = 'MCA_project_123', dpi = 100)

# %%%% F8 - Test fluo splitting 

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['drug'] == 'doxycyclin'), 
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-F8'])), 
           (data['bead type'] == 'M450'),
           (data['UI_Fluo'].apply(lambda x : x not in ['none'])),
           (data['ctFieldThickness'] >= 50),
           (data['bestH0'] <= 800),
           (data['date'].apply(lambda x : x in all_dates))]

co_order = makeOrder(['aSFL-F8'],['MCA2','MCA3'],['low','mid','high'])
box_pairs=[]

fig, ax, dfcount = D1Plot(data, CondCols=['cell subtype','round','UI_Fluo'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 2, orientation = 'v', stressBoxPlot=False,
                 returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('fluoSplitting - F8 - '+ thicknessType +' - cellAvg')

ufun.archiveFig(fig, name=('MCA123_fluoSplitting_F8_' + thicknessType + '_perCell'), figDir = 'MCA_project_123', dpi = 100)


#
thicknessType = 'bestH0'

fig, ax, dfcount = D1Plot(data, CondCols=['cell subtype','round','UI_Fluo'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1, orientation = 'v', stressBoxPlot=False,
                 returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('fluoSplitting - F8 - '+ thicknessType +' - allComps')

ufun.archiveFig(fig, name=('MCA123_fluoSplitting_F8_' + thicknessType + '_allComps'), figDir = 'MCA_project_123', dpi = 100)

# %%%% E4 - Test fluo splitting

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['drug'] == 'doxycyclin'), 
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-E4'])), 
           (data['UI_Fluo'].apply(lambda x : x not in ['none'])),
           (data['bead type'] == 'M450'),
           (data['ctFieldThickness'] >= 50),
           (data['bestH0'] <= 800),
           (data['date'].apply(lambda x : x in all_dates))]

co_order = makeOrder(['aSFL-E4'],['MCA1','MCA3'],['none','low','mid','high'])
box_pairs=[]

fig, ax, dfcount = D1Plot(data, CondCols=['cell subtype','round','UI_Fluo'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 2, orientation = 'v', stressBoxPlot=False,
                 returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('fluoSplitting - E4 - '+ thicknessType +' - cellAvg')

ufun.archiveFig(fig, name=('MCA123_fluoSplitting_E4_' + thicknessType + '_perCell'), figDir = 'MCA_project_123', dpi = 100)


#
thicknessType = 'bestH0'

fig, ax, dfcount = D1Plot(data, CondCols=['cell subtype','round','UI_Fluo'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1, orientation = 'v', stressBoxPlot=False,
                 returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('fluoSplitting - E4 - '+ thicknessType +' - allComps')

ufun.archiveFig(fig, name=('MCA123_fluoSplitting_E4_' + thicknessType + '_allComps'), figDir = 'MCA_project_123', dpi = 100)


# %%%% Six conditions only H0 INTER-ExpDay

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3


# 1 - ctFieldThickness

thicknessType = 'ctFieldThickness'

fig, ax = plt.subplots(2,1, figsize = (12,8))


Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['bead type'] == 'M450'),
           (data['drug'] == 'none'),
           (data['bestH0'] <= 800),
           (data['ctFieldThickness'] >= 50),
           (data['ctFieldThickness'] <= 650),
           (data['date'].apply(lambda x : x in all_dates))]


co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'], all_dates)

box_pairs = [['aSFL-A11 & 22-07-15', 'aSFL-A11 & 22-07-20'], 
             ['aSFL-F8 & 22-07-15', 'aSFL-F8 & 22-07-27'], 
             ['aSFL-E4 & 22-07-20', 'aSFL-E4 & 22-07-27']]

fig, subAx, dfcount = D1Plot(data, fig=fig, ax=ax[0], CondCols=['cell subtype','date'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

ax[0].set_title('3T3aSFL non activated, by dates')




Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['bead type'] == 'M450'),
           (data['drug'] == 'doxycyclin'),
           (data['bestH0'] <= 800),
           (data['ctFieldThickness'] >= 50),
           (data['ctFieldThickness'] <= 650),
           (data['date'].apply(lambda x : x in all_dates))]


co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'], all_dates)

box_pairs = [['aSFL-A11 & 22-07-15', 'aSFL-A11 & 22-07-20'], 
             ['aSFL-F8 & 22-07-15', 'aSFL-F8 & 22-07-27'], 
             ['aSFL-E4 & 22-07-20', 'aSFL-E4 & 22-07-27']]

fig, subAx, dfcount = D1Plot(data, fig=fig, ax=ax[1], CondCols=['cell subtype','date'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

ax[1].set_title('3T3aSFL activated, by dates')

fig.suptitle('Median thickness at low force')
renameAxes(ax, renameDict_MCA3)
ufun.archiveFig(fig, name=('MCA123_interDayComp_ctFieldH'), figDir = 'MCA_project_123', dpi = 100)



# 2 - Averaged fit H0

thicknessType = 'bestH0'

fig, ax = plt.subplots(2,1, figsize = (12,8))

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['bead type'] == 'M450'),
           (data['drug'] == 'none'),
           (data['bestH0'] <= 800),
           (data['ctFieldThickness'] >= 50),
           (data['date'].apply(lambda x : x in all_dates))]


co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'], all_dates)

box_pairs = [['aSFL-A11 & 22-07-15', 'aSFL-A11 & 22-07-20'], 
             ['aSFL-F8 & 22-07-15', 'aSFL-F8 & 22-07-27'], 
             ['aSFL-E4 & 22-07-20', 'aSFL-E4 & 22-07-27']]

fig, subAx, dfcount = D1Plot(data, fig=fig, ax=ax[0], CondCols=['cell subtype','date'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

ax[0].set_title('3T3aSFL non activated, by dates')




Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['bead type'] == 'M450'),
           (data['drug'] == 'doxycyclin'),
           (data['bestH0'] <= 800),
           (data['ctFieldThickness'] >= 50),
           (data['date'].apply(lambda x : x in all_dates))]


co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'], all_dates)

box_pairs = [['aSFL-A11 & 22-07-15', 'aSFL-A11 & 22-07-20'], 
             ['aSFL-F8 & 22-07-15', 'aSFL-F8 & 22-07-27'], 
             ['aSFL-E4 & 22-07-20', 'aSFL-E4 & 22-07-27']]

fig, subAx, dfcount = D1Plot(data, fig=fig, ax=ax[1], CondCols=['cell subtype','date'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

ax[1].set_title('3T3aSFL activated, by dates')

fig.suptitle('H0 from fit, averaged per cell')
renameAxes(ax, renameDict_MCA3)
ufun.archiveFig(fig, name=('MCA123_interDayComp_H0fitAvg'), figDir = 'MCA_project_123', dpi = 100)




# 3 - Raw fit H0

thicknessType = 'bestH0'

fig, ax = plt.subplots(2,1, figsize = (12,8))

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['bead type'] == 'M450'),
           (data['drug'] == 'none'),
           (data['bestH0'] <= 800),
           (data['ctFieldThickness'] >= 50),
           (data['date'].apply(lambda x : x in all_dates))]


co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'], all_dates)

box_pairs = [['aSFL-A11 & 22-07-15', 'aSFL-A11 & 22-07-20'], 
             ['aSFL-F8 & 22-07-15', 'aSFL-F8 & 22-07-27'], 
             ['aSFL-E4 & 22-07-20', 'aSFL-E4 & 22-07-27']]

fig, subAx, dfcount = D1Plot(data, fig=fig, ax=ax[0], CondCols=['cell subtype','date'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 0.6, orientation = 'v', returnCount = 1)

ax[0].set_title('3T3aSFL non activated, by dates')




Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['bead type'] == 'M450'),
           (data['drug'] == 'doxycyclin'),
           (data['bestH0'] <= 800),
           (data['ctFieldThickness'] >= 50),
           (data['date'].apply(lambda x : x in all_dates))]


co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'], all_dates)

box_pairs = [['aSFL-A11 & 22-07-15', 'aSFL-A11 & 22-07-20'], 
             ['aSFL-F8 & 22-07-15', 'aSFL-F8 & 22-07-27'], 
             ['aSFL-E4 & 22-07-20', 'aSFL-E4 & 22-07-27']]

fig, subAx, dfcount = D1Plot(data, fig=fig, ax=ax[1], CondCols=['cell subtype','date'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 0.6, orientation = 'v', returnCount = 1)

ax[1].set_title('3T3aSFL activated, by dates')

fig.suptitle('H0 from fit, all values')
renameAxes(ax, renameDict_MCA3)
ufun.archiveFig(fig, name=('MCA123_interDayComp_H0fitAll'), figDir = 'MCA_project_123', dpi = 100)


plt.show()

# %%%% Paired plot per day

def D1Plot_PairedByDate(data, groupVar = 'date', splitVar = 'drug', superSplitVar = '', Filters=[],
                        Parameter = '', mode = 'non-parametric', plotAllVals = False,
                        byCell = True, stat = False, statMethod = 'Wilcox_lower',
                        splitOrder = []):
    
    data_filtered = data

    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for kk in range(len(Filters)):
        globalFilter = globalFilter & Filters[kk]
    data_filtered = data_filtered[globalFilter]
    
    allCols = data_filtered.columns.values
    aggDict = {}
    for col in allCols:
        if col == Parameter:
            aggDict[col] = 'mean'
        else:
            aggDict[col] = 'first'
    
    if byCell:
        groupByCell = data_filtered.groupby('cellID')
        df = groupByCell.agg(aggDict)
    else:
        df = data_filtered
        
    if superSplitVar != '':
        superSplitVarValues = df[superSplitVar].unique()
        NSS = len(superSplitVarValues)
        
        #### PLOT
        fig, axes = plt.subplots(1,NSS)
        if NSS == 1:
            axes = [axes]
        ListDf = [df[df[superSplitVar] == sSV] for sSV in superSplitVarValues]
        
    else:
        NSS == 1
        #### PLOT
        fig, axes = plt.subplots(1,1)
        axes = [axes]
        ListDf = [df]
        superSplitVarValues = ['']
    
    groupVarAllValues = df[groupVar].unique()
    NGall = len(groupVarAllValues)
    mD = {}
    for i in range(NGall):
        gV = groupVarAllValues[i]
        mD[gV] = gs.markerList10[i]
        
    countSplitGroups = 0
    
    for iss in range(NSS):
        ax = axes[iss]
        df = ListDf[iss]
        sSV = superSplitVarValues[iss]
                
        groupVarValues = df[groupVar].unique()
        if len(splitOrder) == 0:
            splitVarValues = df[splitVar].unique()
        else:
            splitVarValues = splitOrder
            
        NG = len(groupVarValues)
        NS = len(splitVarValues)
        countSplitGroups += NS
        
        cD = {}
        for i in range(NS):
            sV = splitVarValues[i]
            if superSplitVar != '':
                condition = sSV + ' & ' + sV
            else:
                condition = sV

            cD[sV] = styleDict_MCA3[condition]['color']
        
        
        for i in range(NG):
            gV = groupVarValues[i]
            df_group = df[df[groupVar] == gV]
            listCentralVals = []
            listDispersionVals = []
            for j in range(NS):
                sV = splitVarValues[j]
                df_group_split = df_group[df_group[splitVar] == sV]
                if mode == 'non-parametric':
                    CentralVal = np.median(df_group_split[Parameter].values)
                    listCentralVals.append(CentralVal)
                    DispersionVal = np.array([np.percentile(df_group_split[Parameter].values, 25), 
                                              np.percentile(df_group_split[Parameter].values, 75)])
                    listDispersionVals.append(DispersionVal)
                elif mode == 'gaussian':
                    CentralVal = np.mean(df_group_split[Parameter].values)
                    listCentralVals.append(CentralVal)
                    std = np.std(df_group_split[Parameter].values)
                    N = len(df_group_split[Parameter].values)
                    ste = std / (N**0.5)
                    DispersionVal = np.array([CentralVal-ste, 
                                              CentralVal+ste])
                    listDispersionVals.append(DispersionVal)
                
            jitter = (0.5-np.random.rand())*0.4
            df_group['x_jitter'] = 0
            for j in range(NS):
                sV = splitVarValues[j]
                if j == 0:
                    label = gV
                else:
                    label = None
                
                df_group.loc[df_group[splitVar]==sV,'x_jitter'] = j+jitter
                
                #### PLOT
                centralVal = listCentralVals[j]
                yerrTopBar = listDispersionVals[j][1] - centralVal
                yerrBottomBar = centralVal - listDispersionVals[j][0]
                ax.errorbar([j + jitter], [listCentralVals[j]], yerr = [[yerrBottomBar], [yerrTopBar]],
                        marker = mD[gV], markerfacecolor = cD[sV], markersize = 15, markeredgecolor = 'k', 
                        color = 'k', lw = 0.75, 
                        ecolor = cD[sV], elinewidth = 1, capsize = 3, zorder = 4)
                
                # For the legend
                ax.errorbar([], [], yerr = [[], []],
                        marker = mD[gV], markerfacecolor = cD[sV], markersize = 8, markeredgecolor = 'k', 
                        color = 'k', lw = 0.75, ecolor = cD[sV], elinewidth = 2, capsize = 2, zorder = 1,
                        label = label)
                
                
                
                if j < len(splitVarValues)-1:
                    #### PLOT
                    ax.plot([j + jitter, j+1 + jitter], listCentralVals[j:j+2], c = 'k', ls = '-', lw = 0.75)
                    
            #### PLOT
            sns.stripplot(data = df_group, x = 'x_jitter', y = Parameter, ax = ax,
                        marker = mD[gV], color = cD[sV], size = 6, edgecolor = 'gray', linewidth=0.5,
                        alpha = 0.6,
                        zorder = 1)
            
            # for j in range(NS):
            #     sV = splitVarValues[j]
            #     # data_small = df_group[df_group[splitVar] == sV]
            #     if plotAllVals:
            #         print(df_group['x_jitter'])
            #         sns.stripplot(data = df_group, x = 'x_jitter', y = Parameter, ax = ax,
            #                     marker = mD[gV], color = cD[sV], size = 6, edgecolor = 'gray', linewidth=0.5,
            #                     alpha = 0.6,
            #                     zorder = 1)
                
        
        xloc = matplotlib.ticker.FixedLocator([j for j in range(NS)])
        ax.xaxis.set_major_locator(xloc)
        xlab = matplotlib.ticker.FixedFormatter([splitVarValues[j] for j in range(NS)])
        ax.xaxis.set_major_formatter(xlab)
        ax.set_xticklabels([splitVarValues[j] for j in range(NS)], rotation=10)
        
        # ax.set_ylim([0,1.1*ax.get_ylim()[1]])
        ax.set_ylim([0, 750])
        ax.set_xlim([-0.5, NS-0.5])
        ax.legend(loc='lower right', fontsize = 11)
        ax.set_xlabel(splitVar)
        ax.set_title(sSV)
      
    fig.set_size_inches((3*countSplitGroups, 8))

    textSupTitle = ''
    if mode == 'non-parametric':
        textSupTitle += 'Center = median, Dispersion = quartiles'
    if mode == 'gaussian':
        textSupTitle += 'Center = mean, Dispersion = standard error'
        
    fig.suptitle(textSupTitle, fontsize = 11)
    
    return(fig, axes)
       
    
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

#### Filter

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           # (data['cell subtype'] == 'aSFL-A11'), 
           (data['bead type'] == 'M450'),
           (data['bestH0'] <= 900),
           (data['ctFieldThickness'] >= 50),
           (data['ctFieldThickness'] <= 700),
           (data['date'].apply(lambda x : x in all_dates))]


splitOrder = ['none', 'doxycyclin']

fig1, axes1 = D1Plot_PairedByDate(data, Filters = Filters, byCell = True, plotAllVals = True,
                    Parameter = 'ctFieldThickness', superSplitVar = 'cell subtype', mode = 'non-parametric',
                    splitOrder=splitOrder)

renameAxes(axes1, renameDict1)

fig2, axes2 = D1Plot_PairedByDate(data, Filters = Filters, byCell = True, plotAllVals = True,
                    Parameter = 'ctFieldThickness', superSplitVar = 'cell subtype', mode = 'gaussian',
                    splitOrder=splitOrder)

renameAxes(axes2, renameDict1)

plt.show()

ufun.archiveFig(fig1, name=('MCA123_interDayComp_BigPoints_NP'), figDir = 'MCA_project_123', dpi = 100)
ufun.archiveFig(fig2, name=('MCA123_interDayComp_BigPoints_G'), figDir = 'MCA_project_123', dpi = 100)


# %%%% Stat check

def compareDispersion(data, targetCols, groupVar = 'cellID'):
    pass


# %%%% Figure 1 - méca



# %%%% Poolons tous les long linkers ensemble :D XD


# %%%% Filtering out the low fluo

# %%%%% General Thickness - ratio

# %%%%% Thickness - ratio

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11','aSFL-F8','aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data['FluoSelection'] == True),
           # the line above does this:
           # ((data['drug'] == 'none') | (data['UI_Fluo'].apply(lambda x : x in ['mid','high']))), 
           (data[thicknessType] >= 50),
           (data[thicknessType] <= 500),
           (data['date'].apply(lambda x : x in all_dates))]





co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'],['none','doxycyclin'])

box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
            ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
            ['aSFL-E4 & none','aSFL-E4 & doxycyclin']]

normalizeGroups = [['aSFL-A11 & 21-01-18 & none','aSFL-A11 & 21-01-18 & doxycyclin'],
                   ['aSFL-A11 & 21-01-21 & none','aSFL-A11 & 21-01-21 & doxycyclin'],
                   ['aSFL-A11 & 22-07-15 & none','aSFL-A11 & 22-07-15 & doxycyclin'],
                   ['aSFL-A11 & 22-07-20 & none','aSFL-A11 & 22-07-20 & doxycyclin'],
                   ['aSFL-F8 & 21-09-08 & none','aSFL-F8 & 21-09-08 & doxycyclin'],
                   ['aSFL-F8 & 22-07-15 & none','aSFL-F8 & 22-07-15 & doxycyclin'],
                   ['aSFL-F8 & 22-07-27 & none','aSFL-F8 & 22-07-27 & doxycyclin'],
                   ['aSFL-E4 & 21-04-27 & none','aSFL-E4 & 21-04-27 & doxycyclin'],
                   ['aSFL-E4 & 22-07-20 & none','aSFL-E4 & 22-07-20 & doxycyclin'],
                   ['aSFL-E4 & 22-07-27 & none','aSFL-E4 & 22-07-27 & doxycyclin']]

fig, ax, dfcount = D1Plot_wNormalize(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                                     AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                                     box_pairs=box_pairs, normalizeCol=['cell subtype','date','drug'], normalizeGroups=normalizeGroups,
                                     figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                                     returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - All lines - '+ thicknessType +' (r) - cellAvg')
# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_ALLclones-all' +  '_ratio_cellAvg'), figDir = 'MCA_project_123', dpi = 100)





thicknessType = 'bestH0' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11','aSFL-F8','aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data['FluoSelection'] == True),
           # the line above does this:
           # ((data['drug'] == 'none') | (data['UI_Fluo'].apply(lambda x : x in ['mid','high']))), 
           (data[thicknessType] >= 50),
           (data[thicknessType] <= 500),
           (data['date'].apply(lambda x : x in all_dates))]





co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'],['none','doxycyclin'])

box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
            ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
            ['aSFL-E4 & none','aSFL-E4 & doxycyclin']]

normalizeGroups = [['aSFL-A11 & 21-01-18 & none','aSFL-A11 & 21-01-18 & doxycyclin'],
                   ['aSFL-A11 & 21-01-21 & none','aSFL-A11 & 21-01-21 & doxycyclin'],
                   ['aSFL-A11 & 22-07-15 & none','aSFL-A11 & 22-07-15 & doxycyclin'],
                   ['aSFL-A11 & 22-07-20 & none','aSFL-A11 & 22-07-20 & doxycyclin'],
                   ['aSFL-F8 & 21-09-08 & none','aSFL-F8 & 21-09-08 & doxycyclin'],
                   ['aSFL-F8 & 22-07-15 & none','aSFL-F8 & 22-07-15 & doxycyclin'],
                   ['aSFL-F8 & 22-07-27 & none','aSFL-F8 & 22-07-27 & doxycyclin'],
                   ['aSFL-E4 & 21-04-27 & none','aSFL-E4 & 21-04-27 & doxycyclin'],
                   ['aSFL-E4 & 21-04-28 & none','aSFL-E4 & 21-04-28 & doxycyclin'],
                   ['aSFL-E4 & 22-07-20 & none','aSFL-E4 & 22-07-20 & doxycyclin'],
                   ['aSFL-E4 & 22-07-27 & none','aSFL-E4 & 22-07-27 & doxycyclin']]

fig, ax, dfcount = D1Plot_wNormalize(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                                     AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                                     box_pairs=box_pairs, normalizeCol=['cell subtype','date','drug'], normalizeGroups=normalizeGroups,
                                     figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=False,
                                     returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('allMCA - All lines - '+ thicknessType +' (r) - cellAvg')
# ufun.archiveFig(fig, name=('MCA123_' + thicknessType + '_ALLclones-all' +  '_ratio_cellAvg'), figDir = 'MCA_project_123', dpi = 100)

plt.show()






















# %%% MCA project - paper figures

#### Main settings
filterFluo = True

# %%%% Fig 1 - Thickness

# %%%%% A11 - avgH0

# data = GlobalTable_meca_MCA123
data = MecaData_MCA
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'bestH0' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'
alias = 'Thickness from fits'

Filters = [(data['validatedThickness'] == True), 
           (data['bestH0'] <= 850),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11'])), 
           (data['inside bead type'].apply(lambda x : x.startswith('M450'))),
           (data['date'].apply(lambda x : x in all_dates))]


descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'bestH0'
alias = 'Thickness from fits'

Filters = [(data['validatedThickness'] == True), 
           (data['bestH0'] <= 850),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''


co_order = makeOrder(['aSFL-A11'],['none','doxycyclin'])
box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin']]

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 1.4, orientation = 'v', stressBoxPlot=1,
                 returnData = 1, returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('Fig 1 - ' + alias + ' - A11')

ufun.archiveFig(fig, name=('Fig1_' + alias + '_A11_cellAvg'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('Fig1_' + alias + '_A11_cellAvg'), 
                 sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)

plt.show()

# %%%%% A11 - ctFieldH

data = MecaData_MCA
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]


descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''


co_order = makeOrder(['aSFL-A11'],['none','doxycyclin'])
box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin']]

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 1.4, orientation = 'v', stressBoxPlot=True,
                 returnData = 1, returnCount = 1)

renameAxes(ax,renameDict_MCA3)
renameAxes(ax,{'clone A11 - ctrl' : '- iMC', 'clone A11 - iMC' : '+ iMC'})
# fig.suptitle('Fig 1 - ' + alias + ' - A11')
# fig.suptitle('3T3 aSFL thickness')

# ufun.archiveFig(fig, name=('Fig1_' + alias + '_A11_cellAvg'), figDir = 'MCA_Paper'+extDir, dpi = 100)
# ufun.archiveData(dfexport, name=('Fig1_' + alias + '_A11_cellAvg'), 
#                   sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)

plt.show()

# %%%%% A11 - ctFieldH - by dates

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]


descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''


co_order = makeOrder(['21-01-18', '21-01-21','22-07-15', '22-07-20'],['none','doxycyclin'])
box_pairs=[['21-01-18 & none','21-01-18 & doxycyclin'],
           ['21-01-21 & none','21-01-21 & doxycyclin'],
           ['22-07-15 & none','22-07-15 & doxycyclin'],
           ['22-07-20 & none','22-07-20 & doxycyclin'],
           ['21-01-18 & none','21-01-21 & none'],
           ['21-01-18 & none','22-07-15 & none'],
           ['21-01-18 & none','22-07-20 & none'],
           ['21-01-21 & none','22-07-15 & none'],
           ['21-01-21 & none','22-07-20 & none'],
           ['22-07-15 & none','22-07-20 & none'],]

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['date','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.4, orientation = 'v', stressBoxPlot=0,
                 returnData = 1, returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('BONUS Fig - ' + alias + ' - A11')

# for item in ([ax.title, ax.xaxis.label, \
#               ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#     item.set_fontsize(9)
L = []
for item in (ax[0].get_xticklabels()):
    txt = item.get_text()
    txt = txt.split(' & ')[0] + '\n' + txt.split(' & ')[1]
    item.set(text = txt)
    L.append(item)
ax[0].set_xticklabels(L)

ufun.archiveFig(fig, name=('SFigBONUS_' + alias + '_A11_byDate'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('SFigBONUS_' + alias + '_A11_byDate'), 
                  sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)

plt.show()

# %%%% Fig 1 - Stiffness

# %%%%% A11

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

StressRegion = '_S=200+/-100'
srs = StressRegion.split('+/-')[0]+'-'+StressRegion.split('+/-')[1]

Filters = [(data['validatedThickness'] == True), 
           (data['validatedFit'+StressRegion] == True), 
           (data['K'+StressRegion] < 12500), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]

descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

StressRegion = '_S=200+/-100'

Filters = [(data['validatedThickness'] == True), 
           (data['validatedFit'+StressRegion] == True), 
           (data['K'+StressRegion] < 12500), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''


co_order = makeOrder(['aSFL-A11'],['none','doxycyclin'])
box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin']]

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['K'+StressRegion],Filters=Filters,
                          AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor=0.8, orientation = 'v', stressBoxPlot=2,
                          returnData = 1, returnCount = 1)
ax[0].set_ylim([1e2,1.2e4])
renameAxes(ax,renameDict_MCA3)
fig.suptitle('Fig 1 - Stiffness - A11\n' + StressRegion[1:] + ' Pa')

# ufun.archiveFig(fig, name=('Fig1_K' + srs + '_A11_allComps'), figDir = 'MCA_Paper'+extDir, dpi = 100)
# ufun.archiveData(dfexport, name=('Fig1_K' + srs + '_A11_allComps'), 
#                  sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)


fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['K'+StressRegion],Filters=Filters,
                          AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor=1.2, orientation = 'v', stressBoxPlot= 1, 
                          returnData = 1, returnCount = 1)
ax[0].set_ylim([3e2,1.2e4])
renameAxes(ax,renameDict_MCA3)
fig.suptitle('Fig 1 - Stiffness - A11\n' + StressRegion[1:] + ' Pa')

ufun.archiveFig(fig, name=('Fig1_K' + srs + '_A11_cellAvg'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('Fig1_K' + srs + '_A11_cellAvg'), 
                 sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)

plt.show()





# %%%% Fig 4 - Thickness

# %%%%% F8

# ctFieldH

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-F8'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]


descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-F8'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''


co_order = makeOrder(['aSFL-F8'],['none','doxycyclin'])
box_pairs=[['aSFL-F8 & none','aSFL-F8 & doxycyclin']]

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 1.4, orientation = 'v', stressBoxPlot=1,
                 returnData = 1, returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('Fig 4 - ' + alias + ' - F8')

ufun.archiveFig(fig, name=('Fig4_' + alias + '_F8_cellAvg'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('Fig4_' + alias + '_F8_cellAvg'), 
                  sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)

plt.show()

# %%%%% E4

# ctFieldH

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]


descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''


co_order = makeOrder(['aSFL-E4'],['none','doxycyclin'])
box_pairs=[['aSFL-E4 & none','aSFL-E4 & doxycyclin']]

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 1.4, orientation = 'v', stressBoxPlot=1,
                 returnData = 1, returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('Fig 4 - ' + alias + ' - E4')

ufun.archiveFig(fig, name=('Fig4_' + alias + '_E4_cellAvg'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('Fig4_' + alias + '_E4_cellAvg'), 
                  sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)

plt.show()

# %%%%% F8+E4

# ctFieldH

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['linker type'].apply(lambda x : x in ['iMC-6FP'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]


descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['linker type'].apply(lambda x : x in ['iMC-6FP'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''


co_order = makeOrder(['iMC-6FP'],['none','doxycyclin'])
box_pairs=[['iMC-6FP & none','iMC-6FP & doxycyclin']]

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['linker type','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor = 1.4, orientation = 'v', stressBoxPlot=1,
                 returnData = 1, returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('Fig 4 - ' + alias + ' - F8 & E4')

ufun.archiveFig(fig, name=('Fig4_' + alias + '_F8+E4_cellAvg'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('Fig4_' + alias + '_F8+E4_cellAvg'), 
                  sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)

plt.show()


# %%%% Supplementary

# %%%%% Ratios ctFieldH

normalizeGroups = [['aSFL-A11 & 21-01-18 & none','aSFL-A11 & 21-01-18 & doxycyclin'],
                   ['aSFL-A11 & 21-01-21 & none','aSFL-A11 & 21-01-21 & doxycyclin'],
                   ['aSFL-A11 & 22-07-15 & none','aSFL-A11 & 22-07-15 & doxycyclin'],
                   ['aSFL-A11 & 22-07-20 & none','aSFL-A11 & 22-07-20 & doxycyclin'],
                   ['aSFL-F8 & 21-09-08 & none','aSFL-F8 & 21-09-08 & doxycyclin'],
                   ['aSFL-F8 & 22-07-15 & none','aSFL-F8 & 22-07-15 & doxycyclin'],
                   ['aSFL-F8 & 22-07-27 & none','aSFL-F8 & 22-07-27 & doxycyclin'],
                   ['aSFL-E4 & 21-04-27 & none','aSFL-E4 & 21-04-27 & doxycyclin'],
                   ['aSFL-E4 & 21-04-28 & none','aSFL-E4 & 21-04-28 & doxycyclin'],
                   ['aSFL-E4 & 22-07-20 & none','aSFL-E4 & 22-07-20 & doxycyclin'],
                   ['aSFL-E4 & 22-07-27 & none','aSFL-E4 & 22-07-27 & doxycyclin']]

# %%%%%% Ratio ctFieldH A11

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]

descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]

normalizeGroups = [['aSFL-A11 & 21-01-18 & none','aSFL-A11 & 21-01-18 & doxycyclin'],
                   ['aSFL-A11 & 21-01-21 & none','aSFL-A11 & 21-01-21 & doxycyclin'],
                   ['aSFL-A11 & 22-07-15 & none','aSFL-A11 & 22-07-15 & doxycyclin'],
                   ['aSFL-A11 & 22-07-20 & none','aSFL-A11 & 22-07-20 & doxycyclin']]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''

co_order = makeOrder(['aSFL-A11'],['none','doxycyclin'])
box_pairs=[]

normalizeGroups = [['aSFL-A11 & 21-01-18 & none','aSFL-A11 & 21-01-18 & doxycyclin'],
                   ['aSFL-A11 & 21-01-21 & none','aSFL-A11 & 21-01-21 & doxycyclin'],
                   ['aSFL-A11 & 22-07-15 & none','aSFL-A11 & 22-07-15 & doxycyclin'],
                   ['aSFL-A11 & 22-07-20 & none','aSFL-A11 & 22-07-20 & doxycyclin']]

fig, ax, dfexport, dfcount = D1Plot_wNormalize(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                                     AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                                     box_pairs=box_pairs, normalizeCol=['cell subtype','date','drug'], normalizeGroups=normalizeGroups,
                                     figSizeFactor = 0.6, markersizeFactor = 1.2, orientation = 'v', stressBoxPlot=1,
                                     returnData = 1, returnCount = 1)

ax[0].set_ylim([0, 2.1])
renameAxes(ax,renameDict_MCA3)
fig.suptitle('SFig 1 - ' + alias + ' - A11\nNormalized per date')

ufun.archiveFig(fig, name=('SFig1_' + alias + '_A11_RATIO'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('SFig1_' + alias + '_A11_RATIO'), 
                  sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)

plt.show()

# %%%%%% Ratio ctFieldH F8

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'


Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-F8'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]

descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-F8'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]

normalizeGroups = [['aSFL-F8 & 21-09-08 & none','aSFL-F8 & 21-09-08 & doxycyclin'],
                   ['aSFL-F8 & 22-07-15 & none','aSFL-F8 & 22-07-15 & doxycyclin'],
                   ['aSFL-F8 & 22-07-27 & none','aSFL-F8 & 22-07-27 & doxycyclin']]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''

co_order = makeOrder(['aSFL-F8'],['none','doxycyclin'])
box_pairs=[]

normalizeGroups = [['aSFL-F8 & 21-09-08 & none','aSFL-F8 & 21-09-08 & doxycyclin'],
                   ['aSFL-F8 & 22-07-15 & none','aSFL-F8 & 22-07-15 & doxycyclin'],
                   ['aSFL-F8 & 22-07-27 & none','aSFL-F8 & 22-07-27 & doxycyclin']]

fig, ax, dfexport, dfcount = D1Plot_wNormalize(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                                     AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                                     box_pairs=box_pairs, normalizeCol=['cell subtype','date','drug'], normalizeGroups=normalizeGroups,
                                     figSizeFactor = 0.6, markersizeFactor = 1.2, orientation = 'v', stressBoxPlot=1,
                                     returnData = 1, returnCount = 1)

ax[0].set_ylim([0, 2.1])
renameAxes(ax,renameDict_MCA3)
fig.suptitle('SFig 4 - ' + alias + ' - F8\nNormalized per date')

ufun.archiveFig(fig, name=('SFig4_' + alias + '_F8_RATIO'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('SFig4_' + alias + '_F8_RATIO'), 
                  sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)

plt.show()




# %%%%%% Ratio ctFieldH E4

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]

descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]

normalizeGroups = [['aSFL-E4 & 21-04-27 & none','aSFL-E4 & 21-04-27 & doxycyclin'],
                   ['aSFL-E4 & 21-04-28 & none','aSFL-E4 & 21-04-28 & doxycyclin'],
                   ['aSFL-E4 & 22-07-20 & none','aSFL-E4 & 22-07-20 & doxycyclin'],
                   ['aSFL-E4 & 22-07-27 & none','aSFL-E4 & 22-07-27 & doxycyclin']]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''

co_order = makeOrder(['aSFL-E4'],['none','doxycyclin'])
box_pairs=[]

normalizeGroups = [['aSFL-E4 & 21-04-27 & none','aSFL-E4 & 21-04-27 & doxycyclin'],
                   ['aSFL-E4 & 21-04-28 & none','aSFL-E4 & 21-04-28 & doxycyclin'],
                   ['aSFL-E4 & 22-07-20 & none','aSFL-E4 & 22-07-20 & doxycyclin'],
                   ['aSFL-E4 & 22-07-27 & none','aSFL-E4 & 22-07-27 & doxycyclin']]

fig, ax, dfexport, dfcount = D1Plot_wNormalize(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                                     AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                                     box_pairs=box_pairs, normalizeCol=['cell subtype','date','drug'], normalizeGroups=normalizeGroups,
                                     figSizeFactor = 0.6, markersizeFactor = 1.2, orientation = 'v', stressBoxPlot=1,
                                     returnData = 1, returnCount = 1)

ax[0].set_ylim([0, 2.6])
renameAxes(ax,renameDict_MCA3)
fig.suptitle('SFig 4 - ' + alias + ' - E4\nNormalized per date')

ufun.archiveFig(fig, name=('SFig4_' + alias + '_E4_RATIO'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('SFig4_' + alias + '_E4_RATIO'), 
                  sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)

plt.show()





# %%%%%% Ratio ctFieldH F8+E4

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['linker type'].apply(lambda x : x in ['iMC-6FP'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]

descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldMaxThickness'] <= 900),
           (data['ctFieldVarThickness'] <= 2e4),
           (data['ctFieldThickness'] >= 50),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['linker type'].apply(lambda x : x in ['iMC-6FP'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]

normalizeGroups = [['aSFL-F8 & 21-09-08 & none','aSFL-F8 & 21-09-08 & doxycyclin'],
                   ['aSFL-F8 & 22-07-15 & none','aSFL-F8 & 22-07-15 & doxycyclin'],
                   ['aSFL-F8 & 22-07-27 & none','aSFL-F8 & 22-07-27 & doxycyclin'],
                   ['aSFL-E4 & 21-04-27 & none','aSFL-E4 & 21-04-27 & doxycyclin'],
                   ['aSFL-E4 & 21-04-28 & none','aSFL-E4 & 21-04-28 & doxycyclin'],
                   ['aSFL-E4 & 22-07-20 & none','aSFL-E4 & 22-07-20 & doxycyclin'],
                   ['aSFL-E4 & 22-07-27 & none','aSFL-E4 & 22-07-27 & doxycyclin']]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''

co_order = makeOrder(['iMC-6FP'],['none','doxycyclin'])
box_pairs=[]

normalizeGroups = [['aSFL-F8 & 21-09-08 & none','aSFL-F8 & 21-09-08 & doxycyclin'],
                   ['aSFL-F8 & 22-07-15 & none','aSFL-F8 & 22-07-15 & doxycyclin'],
                   ['aSFL-F8 & 22-07-27 & none','aSFL-F8 & 22-07-27 & doxycyclin'],
                   ['aSFL-E4 & 21-04-27 & none','aSFL-E4 & 21-04-27 & doxycyclin'],
                   ['aSFL-E4 & 21-04-28 & none','aSFL-E4 & 21-04-28 & doxycyclin'],
                   ['aSFL-E4 & 22-07-20 & none','aSFL-E4 & 22-07-20 & doxycyclin'],
                   ['aSFL-E4 & 22-07-27 & none','aSFL-E4 & 22-07-27 & doxycyclin']]

fig, ax, dfexport, dfcount = D1Plot_wNormalize(data, CondCols=['linker type','drug'], Parameters=[thicknessType],Filters=Filters,
                                     AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                                     box_pairs=box_pairs, normalizeCol=['cell subtype','date','drug'], normalizeGroups=normalizeGroups,
                                     figSizeFactor = 0.6, markersizeFactor = 1.2, orientation = 'v', stressBoxPlot=1,
                                     returnData = 1, returnCount = 1)

ax[0].set_ylim([0, 2.6])
renameAxes(ax,renameDict_MCA3)
fig.suptitle('SFig 4 - ' + alias + ' - F8 & E4\nNormalized per date')

ufun.archiveFig(fig, name=('SFig4_' + alias + '_F8+E4_RATIO'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('SFig4_' + alias + '_F8+E4_RATIO'), 
                  sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)

plt.show()






# %%%%% Stiffness long linker

# %%%%%% F8

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

StressRegion = '_S=200+/-100'
srs = StressRegion.split('+/-')[0]+'-'+StressRegion.split('+/-')[1]

Filters = [(data['validatedThickness'] == True), 
           (data['validatedFit'+StressRegion] == True), 
           (data['K'+StressRegion] < 12500), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-F8'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]

descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

StressRegion = '_S=200+/-100'

Filters = [(data['validatedThickness'] == True), 
           (data['validatedFit'+StressRegion] == True), 
           (data['K'+StressRegion] < 12500), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-F8'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''


co_order = makeOrder(['aSFL-F8'],['none','doxycyclin'])
box_pairs=[['aSFL-F8 & none','aSFL-F8 & doxycyclin']]

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['K'+StressRegion],Filters=Filters,
                          AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor=0.8, orientation = 'v', stressBoxPlot= True,
                          returnData = 1, returnCount = 1)
ax[0].set_ylim([1e2,1.2e4])
renameAxes(ax,renameDict_MCA3)
fig.suptitle('SFig 4 - Stiffness - F8\n' + StressRegion[1:] + ' Pa')

# ufun.archiveFig(fig, name=('SFig4_K' + srs + '_F8_allComps'), figDir = 'MCA_Paper'+extDir, dpi = 100)
# ufun.archiveData(dfexport, name=('SFig4_K' + srs + '_F8_allComps'), 
#                  sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)


fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['K'+StressRegion],Filters=Filters,
                          AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor=1.2, orientation = 'v', stressBoxPlot= False, 
                          returnData = 1, returnCount = 1)
ax[0].set_ylim([3e2,1.2e4])
renameAxes(ax,renameDict_MCA3)
fig.suptitle('SFig 4 - Stiffness - F8\n' + StressRegion[1:] + ' Pa')

ufun.archiveFig(fig, name=('SFig4_K' + srs + '_F8_cellAvg'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('SFig4_K' + srs + '_F8_cellAvg'), 
                 sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)

plt.show()

# %%%%%% E4

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

StressRegion = '_S=200+/-100'
srs = StressRegion.split('+/-')[0]+'-'+StressRegion.split('+/-')[1]

Filters = [(data['validatedThickness'] == True), 
           (data['validatedFit'+StressRegion] == True), 
           (data['K'+StressRegion] < 12500), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]

descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

StressRegion = '_S=200+/-100'

Filters = [(data['validatedThickness'] == True), 
           (data['validatedFit'+StressRegion] == True), 
           (data['K'+StressRegion] < 12500), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''


co_order = makeOrder(['aSFL-E4'],['none','doxycyclin'])
box_pairs=[['aSFL-E4 & none','aSFL-E4 & doxycyclin']]

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['K'+StressRegion],Filters=Filters,
                          AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor=0.8, orientation = 'v', stressBoxPlot= True,
                          returnData = 1, returnCount = 1)
ax[0].set_ylim([1e2,1.2e4])
renameAxes(ax,renameDict_MCA3)
fig.suptitle('SFig 4 - Stiffness - E4\n' + StressRegion[1:] + ' Pa')

# ufun.archiveFig(fig, name=('SFig4_K' + srs + '_E4_allComps'), figDir = 'MCA_Paper'+extDir, dpi = 100)
# ufun.archiveData(dfexport, name=('SFig4_K' + srs + '_E4_allComps'), 
#                  sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)


fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['K'+StressRegion],Filters=Filters,
                          AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor=1.2, orientation = 'v', stressBoxPlot= False, 
                          returnData = 1, returnCount = 1)
ax[0].set_ylim([3e2,1.2e4])
renameAxes(ax,renameDict_MCA3)
fig.suptitle('SFig 4 - Stiffness - E4\n' + StressRegion[1:] + ' Pa')

ufun.archiveFig(fig, name=('SFig4_K' + srs + '_E4_cellAvg'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('SFig4_K' + srs + '_E4_cellAvg'), 
                 sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)

plt.show()

# %%%%%% F8+E4

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

StressRegion = '_S=200+/-100'
srs = StressRegion.split('+/-')[0]+'-'+StressRegion.split('+/-')[1]

Filters = [(data['validatedThickness'] == True), 
           (data['validatedFit'+StressRegion] == True), 
           (data['K'+StressRegion] < 12500), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['linker type'].apply(lambda x : x in ['iMC-6FP'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]

descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

StressRegion = '_S=200+/-100'

Filters = [(data['validatedThickness'] == True), 
           (data['validatedFit'+StressRegion] == True), 
           (data['K'+StressRegion] < 12500), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['linker type'].apply(lambda x : x in ['iMC-6FP'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''


co_order = makeOrder(['iMC-6FP'],['none','doxycyclin'])
box_pairs=[['iMC-6FP & none','iMC-6FP & doxycyclin']]

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['linker type','drug'], Parameters=['K'+StressRegion],Filters=Filters,
                          AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor=0.7, orientation = 'v', stressBoxPlot= True,
                          returnData = 1, returnCount = 1)
ax[0].set_ylim([1e2,1.2e4])
renameAxes(ax,renameDict_MCA3)
fig.suptitle('SFig 4 - Stiffness - F8 & E4\n' + StressRegion[1:] + ' Pa')

ufun.archiveFig(fig, name=('SFig4_K' + srs + '_F8+E4_allComps'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('SFig4_K' + srs + '_F8+E4_allComps'), 
                 sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)


fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['linker type','drug'], Parameters=['K'+StressRegion],Filters=Filters,
                          AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 0.6, markersizeFactor=1.2, orientation = 'v', stressBoxPlot= False, 
                          returnData = 1, returnCount = 1)
ax[0].set_ylim([3e2,1.2e4])
renameAxes(ax,renameDict_MCA3)
fig.suptitle('SFig 4 - Stiffness - F8 & E4\n' + StressRegion[1:] + ' Pa')

ufun.archiveFig(fig, name=('SFig4_K' + srs + '_F8+E4_cellAvg'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('SFig4_K' + srs + '_F8+E4_cellAvg'), 
                 sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)

plt.show()

# %%%% BONUS Figs

# %%%%% All cell lines - thickness

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldThickness'] >= 50),
           (data['ctFieldMaxThickness'] <= 900),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11','aSFL-F8','aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]


descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['ctFieldThickness'] >= 50),
           (data['ctFieldMaxThickness'] <= 900),
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11','aSFL-F8','aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''


co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'],['none','doxycyclin'])
box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
           ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
           ['aSFL-E4 & none','aSFL-E4 & doxycyclin'],
           ['aSFL-A11 & none','aSFL-F8 & none'],
           ['aSFL-A11 & none','aSFL-E4 & none'],
           ['aSFL-F8 & none','aSFL-E4 & none']]

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=1,
                 returnData = 1, returnCount = 1)

renameAxes(ax,renameDict_MCA3)

fig.suptitle('BONUS Fig - '+ alias +' - All Lines')

ufun.archiveFig(fig, name=('SFigBONUS_' + alias + '_AllLines_cellAvg'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('SFigBONUS_' + alias + '_AllLines_cellAvg'), 
                  sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)


# %%%%% All cell lines - ratio thickness


data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness' # 'surroundingThickness' # 'bestH0' # 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
            (data['FluoSelection'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11','aSFL-F8','aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data['ctFieldThickness'] >= 50),
           (data['ctFieldMaxThickness'] <= 650),
           (data['date'].apply(lambda x : x in all_dates))]


descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

thicknessType = 'ctFieldThickness'
alias = 'Thickness at low force'

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
            (data['FluoSelection'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11','aSFL-F8','aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data['ctFieldThickness'] >= 50),
           (data['ctFieldMaxThickness'] <= 650),
           (data['date'].apply(lambda x : x in all_dates))]

normalizeGroups = [['aSFL-A11 & 21-01-18 & none','aSFL-A11 & 21-01-18 & doxycyclin'],
                   ['aSFL-A11 & 21-01-21 & none','aSFL-A11 & 21-01-21 & doxycyclin'],
                   ['aSFL-A11 & 22-07-15 & none','aSFL-A11 & 22-07-15 & doxycyclin'],
                   ['aSFL-A11 & 22-07-20 & none','aSFL-A11 & 22-07-20 & doxycyclin'],
                   ['aSFL-F8 & 21-09-08 & none','aSFL-F8 & 21-09-08 & doxycyclin'],
                   ['aSFL-F8 & 22-07-15 & none','aSFL-F8 & 22-07-15 & doxycyclin'],
                   ['aSFL-F8 & 22-07-27 & none','aSFL-F8 & 22-07-27 & doxycyclin'],
                   ['aSFL-E4 & 21-04-27 & none','aSFL-E4 & 21-04-27 & doxycyclin'],
                   ['aSFL-E4 & 21-04-28 & none','aSFL-E4 & 21-04-28 & doxycyclin'],
                   ['aSFL-E4 & 22-07-20 & none','aSFL-E4 & 22-07-20 & doxycyclin'],
                   ['aSFL-E4 & 22-07-27 & none','aSFL-E4 & 22-07-27 & doxycyclin']]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''





co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'],['none','doxycyclin'])

box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
            ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
            ['aSFL-E4 & none','aSFL-E4 & doxycyclin']]

normalizeGroups = [['aSFL-A11 & 21-01-18 & none','aSFL-A11 & 21-01-18 & doxycyclin'],
                   ['aSFL-A11 & 21-01-21 & none','aSFL-A11 & 21-01-21 & doxycyclin'],
                   ['aSFL-A11 & 22-07-15 & none','aSFL-A11 & 22-07-15 & doxycyclin'],
                   ['aSFL-A11 & 22-07-20 & none','aSFL-A11 & 22-07-20 & doxycyclin'],
                   ['aSFL-F8 & 21-09-08 & none','aSFL-F8 & 21-09-08 & doxycyclin'],
                   ['aSFL-F8 & 22-07-15 & none','aSFL-F8 & 22-07-15 & doxycyclin'],
                   ['aSFL-F8 & 22-07-27 & none','aSFL-F8 & 22-07-27 & doxycyclin'],
                   ['aSFL-E4 & 21-04-27 & none','aSFL-E4 & 21-04-27 & doxycyclin'],
                   ['aSFL-E4 & 21-04-28 & none','aSFL-E4 & 21-04-28 & doxycyclin'],
                   ['aSFL-E4 & 22-07-20 & none','aSFL-E4 & 22-07-20 & doxycyclin'],
                   ['aSFL-E4 & 22-07-27 & none','aSFL-E4 & 22-07-27 & doxycyclin']]

fig, ax, dfexport, dfcount = D1Plot_wNormalize(data, CondCols=['cell subtype','drug'], Parameters=[thicknessType],Filters=Filters,
                                     AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                                     box_pairs=box_pairs, normalizeCol=['cell subtype','date','drug'], normalizeGroups=normalizeGroups,
                                     figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', stressBoxPlot=1,
                                     returnData = 1, returnCount = 1)

renameAxes(ax,renameDict_MCA3)
fig.suptitle('BONUS Fig - '+ alias +' - All Lines\nNormalized per date')

ufun.archiveFig(fig, name=('SFigBONUS_' + alias + '_AllLines_RATIO'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('SFigBONUS_' + alias + '_AllLines_RATIO'), 
                  sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)


# %%%%% All cell lines - thickness day by day

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3


# 1 - ctFieldThickness

thicknessType = 'ctFieldThickness'

fig, ax = plt.subplots(2,1, figsize = (12,8))


Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['bead type'] == 'M450'),
           (data['drug'] == 'none'),
           # (data['bestH0'] <= 800),
           (data['ctFieldThickness'] >= 50),
           (data['ctFieldMaxThickness'] <= 900),
           (data['date'].apply(lambda x : x in all_dates))]

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    
else:
    extDir = ''


co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'], all_dates)

box_pairs = [['aSFL-A11 & 22-07-15', 'aSFL-A11 & 22-07-20'], 
             ['aSFL-F8 & 22-07-15', 'aSFL-F8 & 22-07-27'], 
             ['aSFL-E4 & 22-07-20', 'aSFL-E4 & 22-07-27']]

fig, subAx, dfcount = D1Plot(data, fig=fig, ax=ax[0], CondCols=['cell subtype','date'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

ax[0].set_title('3T3aSFL CONTROL, by dates')
renameAxes(ax[0], renameDict_MCA3)


Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['bead type'] == 'M450'),
           (data['drug'] == 'doxycyclin'),
           # (data['bestH0'] <= 800),
           (data['ctFieldThickness'] >= 50),
           (data['ctFieldMaxThickness'] <= 900),
           (data['date'].apply(lambda x : x in all_dates))]


co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'], all_dates)

box_pairs = [['aSFL-A11 & 22-07-15', 'aSFL-A11 & 22-07-20'], 
             ['aSFL-F8 & 22-07-15', 'aSFL-F8 & 22-07-27'], 
             ['aSFL-E4 & 22-07-20', 'aSFL-E4 & 22-07-27']]

fig, subAx, dfcount = D1Plot(data, fig=fig, ax=ax[1], CondCols=['cell subtype','date'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

ax[1].set_title('3T3aSFL EXPRESSING LINKER, by dates')
renameAxes(ax[1], renameDict_MCA3)

fig.suptitle('BONUS Fig - Median thickness at low force')

plt.show()

ufun.archiveFig(fig, name=('SFigBONUS' + '_AllLines_DayByDay'), figDir = 'MCA_Paper'+extDir, dpi = 100)

# %%%%% All cell lines - Big Marker Joint Plot

def D1Plot_PairedByDate(data, groupVar = 'date', splitVar = 'drug', superSplitVar = '', Filters=[],
                        Parameter = '', mode = 'non-parametric', plotAllVals = False,
                        byCell = True, stat = False, statMethod = 'Wilcox_lower',
                        splitOrder = []):
    
    data_filtered = data

    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for kk in range(len(Filters)):
        globalFilter = globalFilter & Filters[kk]
    data_filtered = data_filtered[globalFilter]
    
    allCols = data_filtered.columns.values
    aggDict = {}
    for col in allCols:
        if col == Parameter:
            aggDict[col] = 'mean'
        else:
            aggDict[col] = 'first'
    
    if byCell:
        groupByCell = data_filtered.groupby('cellID')
        df = groupByCell.agg(aggDict)
    else:
        df = data_filtered
        
    if superSplitVar != '':
        superSplitVarValues = df[superSplitVar].unique()
        NSS = len(superSplitVarValues)
        
        #### PLOT
        fig, axes = plt.subplots(1,NSS)
        if NSS == 1:
            axes = [axes]
        ListDf = [df[df[superSplitVar] == sSV] for sSV in superSplitVarValues]
        
    else:
        NSS == 1
        #### PLOT
        fig, axes = plt.subplots(1,1)
        axes = [axes]
        ListDf = [df]
        superSplitVarValues = ['']
    
    groupVarAllValues = df[groupVar].unique()
    NGall = len(groupVarAllValues)
    mD = {}
    for i in range(NGall):
        gV = groupVarAllValues[i]
        mD[gV] = gs.markerList10[i]
        
    countSplitGroups = 0
    
    for iss in range(NSS):
        ax = axes[iss]
        df = ListDf[iss]
        sSV = superSplitVarValues[iss]
                
        groupVarValues = df[groupVar].unique()
        if len(splitOrder) == 0:
            splitVarValues = df[splitVar].unique()
        else:
            splitVarValues = splitOrder
            
        NG = len(groupVarValues)
        NS = len(splitVarValues)
        countSplitGroups += NS
        
        cD = {}
        for i in range(NS):
            sV = splitVarValues[i]
            if superSplitVar != '':
                condition = sSV + ' & ' + sV
            else:
                condition = sV

            cD[sV] = styleDict_MCA3[condition]['color']
        
        
        for i in range(NG):
            gV = groupVarValues[i]
            df_group = df[df[groupVar] == gV]
            listCentralVals = []
            listDispersionVals = []
            for j in range(NS):
                sV = splitVarValues[j]
                df_group_split = df_group[df_group[splitVar] == sV]
                if mode == 'non-parametric':
                    CentralVal = np.median(df_group_split[Parameter].values)
                    listCentralVals.append(CentralVal)
                    DispersionVal = np.array([np.percentile(df_group_split[Parameter].values, 25), 
                                              np.percentile(df_group_split[Parameter].values, 75)])
                    listDispersionVals.append(DispersionVal)
                elif mode == 'gaussian':
                    CentralVal = np.mean(df_group_split[Parameter].values)
                    listCentralVals.append(CentralVal)
                    std = np.std(df_group_split[Parameter].values)
                    N = len(df_group_split[Parameter].values)
                    ste = std / (N**0.5)
                    DispersionVal = np.array([CentralVal-ste, 
                                              CentralVal+ste])
                    listDispersionVals.append(DispersionVal)
                
            jitter = (0.5-np.random.rand())*0.4
            df_group['x_jitter'] = 0
            for j in range(NS):
                sV = splitVarValues[j]
                if j == 0:
                    label = gV
                else:
                    label = None
                
                df_group.loc[df_group[splitVar]==sV,'x_jitter'] = j+jitter
                
                #### PLOT
                centralVal = listCentralVals[j]
                yerrTopBar = listDispersionVals[j][1] - centralVal
                yerrBottomBar = centralVal - listDispersionVals[j][0]
                ax.errorbar([j + jitter], [listCentralVals[j]], yerr = [[yerrBottomBar], [yerrTopBar]],
                        marker = mD[gV], markerfacecolor = cD[sV], markersize = 15, markeredgecolor = 'k', 
                        color = 'k', lw = 0.75, 
                        ecolor = cD[sV], elinewidth = 1, capsize = 3, zorder = 4)
                
                # For the legend
                ax.errorbar([], [], yerr = [[], []],
                        marker = mD[gV], markerfacecolor = cD[sV], markersize = 8, markeredgecolor = 'k', 
                        color = 'k', lw = 0.75, ecolor = cD[sV], elinewidth = 2, capsize = 2, zorder = 1,
                        label = label)
                
                
                
                if j < len(splitVarValues)-1:
                    #### PLOT
                    ax.plot([j + jitter, j+1 + jitter], listCentralVals[j:j+2], c = 'k', ls = '-', lw = 0.75)
                    
            #### PLOT
            sns.stripplot(data = df_group, x = 'x_jitter', y = Parameter, ax = ax,
                        marker = mD[gV], color = cD[sV], size = 6, edgecolor = 'gray', linewidth=0.5,
                        alpha = 0.6,
                        zorder = 1)
            
            # for j in range(NS):
            #     sV = splitVarValues[j]
            #     # data_small = df_group[df_group[splitVar] == sV]
            #     if plotAllVals:
            #         print(df_group['x_jitter'])
            #         sns.stripplot(data = df_group, x = 'x_jitter', y = Parameter, ax = ax,
            #                     marker = mD[gV], color = cD[sV], size = 6, edgecolor = 'gray', linewidth=0.5,
            #                     alpha = 0.6,
            #                     zorder = 1)
                
        
        xloc = matplotlib.ticker.FixedLocator([j for j in range(NS)])
        ax.xaxis.set_major_locator(xloc)
        xlab = matplotlib.ticker.FixedFormatter([splitVarValues[j] for j in range(NS)])
        ax.xaxis.set_major_formatter(xlab)
        ax.set_xticklabels([splitVarValues[j] for j in range(NS)], rotation=10)
        
        # ax.set_ylim([0,1.1*ax.get_ylim()[1]])
        ax.set_ylim([0, 750])
        ax.set_xlim([-0.5, NS-0.5])
        ax.legend(loc='lower right', fontsize = 11)
        ax.set_xlabel(splitVar)
        ax.set_title(sSV)
      
    fig.set_size_inches((3*countSplitGroups, 8))

    textSupTitle = 'BONUS Fig - '
    if mode == 'non-parametric':
        textSupTitle += 'Center = median, Dispersion = quartiles'
    if mode == 'gaussian':
        textSupTitle += 'Center = mean, Dispersion = standard error'
        
    fig.suptitle(textSupTitle, fontsize = 12)
    
    return(fig, axes)
       
    
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

#### Filter

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           # (data['cell subtype'] == 'aSFL-A11'), 
           (data['bead type'] == 'M450'),
           (data['ctFieldThickness'] >= 50),
           (data['ctFieldMaxThickness'] <= 900),
           (data['date'].apply(lambda x : x in all_dates))]

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    
else:
    extDir = ''


splitOrder = ['none', 'doxycyclin']

fig1, axes1 = D1Plot_PairedByDate(data, Filters = Filters, byCell = True, plotAllVals = True,
                    Parameter = 'ctFieldThickness', superSplitVar = 'cell subtype', mode = 'non-parametric',
                    splitOrder=splitOrder)

renameAxes(axes1, renameDict1)

plt.show()

ufun.archiveFig(fig1, name=('SFigBONUS' + '_AllLines_DayByDay_AlternativeView'), figDir = 'MCA_Paper'+extDir, dpi = 100)

# %%%%% All cell lines - stiffness

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

StressRegion = '_S=200+/-100'
srs = StressRegion.split('+/-')[0]+'-'+StressRegion.split('+/-')[1]

Filters = [(data['validatedThickness'] == True), 
           (data['validatedFit'+StressRegion] == True), 
           (data['K'+StressRegion] < 12500), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11','aSFL-F8','aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]



descText = """
data = GlobalTable_meca_MCA123

dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

StressRegion = '_S=200+/-100'

Filters = [(data['validatedThickness'] == True), 
           (data['validatedFit'+StressRegion] == True), 
           (data['K'+StressRegion] < 12500), 
           (data['UI_Valid'] == True),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11','aSFL-F8','aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]
"""

if filterFluo:
    Filters.append((data['FluoSelection'] == True))
    extDir = '_filterFluo'
    descText += """\nWith Filter Fluo Enabled (no +iMC cell with low fluo)."""
    
else:
    extDir = ''

# print(data['cell subtype'].unique())

co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'],['none','doxycyclin'])
box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
           ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
           ['aSFL-E4 & none','aSFL-E4 & doxycyclin'],
           ['aSFL-A11 & none','aSFL-F8 & none'],
           ['aSFL-A11 & none','aSFL-E4 & none']]

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['K'+StressRegion],Filters=Filters,
                          AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor=0.6, orientation = 'v', stressBoxPlot= True,
                          returnData = 1, returnCount = 1)
# ax[0].set_ylim([1e2,1.2e4])
ax[0].set_yscale('log')
renameAxes(ax,renameDict_MCA3)
fig.suptitle('Fig 1 - Stiffness - all cell lines\n' + StressRegion[1:] + ' Pa')

ufun.archiveFig(fig, name=('Fig1_K' + srs + '_allLines_allComps'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('Fig1_K' + srs + '_allLines_allComps'), 
                 sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)


fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['K'+StressRegion],Filters=Filters,
                          AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor=1.0, orientation = 'v', stressBoxPlot= False, 
                          returnData = 1, returnCount = 1)
# ax[0].set_ylim([1e2,1.2e4])
ax[0].set_yscale('log')
ax[0].set_yscale('log')
renameAxes(ax,renameDict_MCA3)
fig.suptitle('BONUS Fig - Stiffness - all cell lines\n' + StressRegion[1:] + ' Pa')

ufun.archiveFig(fig, name=('SFigBONUS _K' + srs + '_allLines_cellAvg'), figDir = 'MCA_Paper'+extDir, dpi = 100)
ufun.archiveData(dfexport, name=('SFigBONUS _K' + srs + '_allLines_cellAvg'), 
                 sep = ';', saveDir = 'MCA_Paper'+extDir, descText = descText)

plt.show()

# %%%% Typical constant field force

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3


Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['ctFieldForce'] < 400),
           (data['cell type'] == '3T3'), 
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11','aSFL-F8','aSFL-E4'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]

fltr = Filters[0]
for i in range(1, len(Filters)):
    fltr = fltr & Filters[i]
print(np.median(data[fltr]['maxForce']))
# print(np.std(data[fltr]['ctFieldForce']))

co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'],['none','doxycyclin'])
box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
           ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
           ['aSFL-E4 & none','aSFL-E4 & doxycyclin'],
           ['aSFL-A11 & none','aSFL-F8 & none'],
           ['aSFL-A11 & none','aSFL-E4 & none']]

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype','drug'], Parameters=['minForce'],Filters=Filters,
                          AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor=1.0, orientation = 'v', stressBoxPlot= False, 
                          returnData = 1, returnCount = 1)

# %%%% Next

# %%% Fig for TAC

# %%%% E(h)

data = GlobalTable_meca_MCA123
dates_r1 = ['21-01-18', '21-01-21'] #, '22-07-15', '22-07-20']
# dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
# dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 # + dates_r2 + dates_r3
cellsSub = ['ctrl', 'aSFL-A11'] # ,'aSFL-F8','aSFL-E4']
drugs = ['none']


Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
            (data['surroundingThickness'] >= 80),
           (data['cell type'] == '3T3'), 
           (data['drug'].apply(lambda x : x in drugs)), 
           (data['cell subtype'].apply(lambda x : x in cellsSub)), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in all_dates))]

# fltr = Filters[0]
# for i in range(1, len(Filters)):
#     fltr = fltr & Filters[i]
# print(np.median(data[fltr]['maxForce']))
# print(np.std(data[fltr]['ctFieldForce']))

# co_order = makeOrder(['aSFL-A11','aSFL-F8','aSFL-E4'],['none','doxycyclin'])
# box_pairs=[['aSFL-A11 & none','aSFL-A11 & doxycyclin'],
#            ['aSFL-F8 & none','aSFL-F8 & doxycyclin'],
#            ['aSFL-E4 & none','aSFL-E4 & doxycyclin'],
#            ['aSFL-A11 & none','aSFL-F8 & none'],
#            ['aSFL-A11 & none','aSFL-E4 & none']]

# fig, ax = D2Plot(data, CondCols=['cell subtype','drug'], Parameters=['minForce'],Filters=Filters,
#                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
#                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor=1.0, orientation = 'v', stressBoxPlot= False, 
#                 returnData = 1, returnCount = 1)

fig, ax = D2Plot_wFit(data, XCol='surroundingThickness', YCol='EChadwick', CondCol=['cell type'], Filters=Filters, 
           cellID='cellID', AvgPerCell=True, co_order=['3T3'], modelFit=True, modelType='y=k*x^a', writeEqn = True,
           xscale = 'log', yscale = 'log', 
           figSizeFactor = 1, markersizeFactor = 1.5)
ax.legend(fontsize = 11)
ax.set_xlim([95, 1800])
renameAxes(ax,renameDict_MCA3)

plt.show()
