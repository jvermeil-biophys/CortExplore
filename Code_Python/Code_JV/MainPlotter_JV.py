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
import re
import sys
import time
import random
import numbers
import warnings
import itertools
import matplotlib

from cycler import cycler
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
# from matplotlib.gridspec import GridSpec

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)

import GraphicStyles as gs
import UtilityFunctions as ufun
import TrackAnalyser as taka
import TrackAnalyser_V2 as taka2

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

taka.plotCellTimeSeriesData('22-07-15_M1_P1_C2')

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
cellID = '22-03-21_M3_P1_C1_sin5-3_2Hz'
fromPython = True

X = 'T'

units = np.array([' (mT)', ' (pN)'])
tSDF = taka.getCellTimeSeriesData(cellID, fromPython)

# defaultColorCycle = plt.rcParams['axes.prop_cycle']
# customColorCycle = cycler(color=['green', 'red'])
# plt.rcParams['axes.prop_cycle'] = customColorCycle

fig, ax = plt.subplots(1,1, figsize = (10,5))

if not tSDF.size == 0:
#         plt.tight_layout()
#         fig.show() # figsize=(20,20)
    tsDFplot = tSDF[(tSDF['T']>=3.0) & (tSDF['T']<=6)]
    
    ax.plot(tsDFplot['T'], 1000*(tsDFplot['D3']-4.503), color = gs.colorList40[30], label = 'Thickness')
    ax.set_ylabel('Thickness (nm)', color = gs.colorList40[30])
    ax.set_ylim([700, 980])
    ax.set_xlabel('Time (s)')

    axR = ax.twinx()
    axR.plot(tsDFplot['T'], tsDFplot['F'], color = gs.colorList40[23], label = 'Force')
    axR.set_ylabel('Force (pN)', color = gs.colorList40[23])
    axR.set_ylim([0, 150])
    
    
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
    
    fig.suptitle('2Hz oscillations')
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

# %% > Data import & export

#### Data import

#### GlobalTable_ctField
CtFieldData_All_JV = taka2.getMergedTable('CtFieldData_All_JV', mergeUMS = False)

#### MecaData_All
# MecaData_All = taka2.getMergedTable('MecaData_All_JV')

#### MecaData_NonLin
MecaData_NonLin = taka2.getMergedTable('MecaData_NonLin')

#### MecaData_Drugs
MecaData_Drugs = taka2.getMergedTable('MecaData_Drugs')

#### MecaData_Atcc
MecaData_Atcc = taka2.getMergedTable('MecaData_Atcc')

#### MecaData_MCA
# MecaData_MCA = taka2.getMergedTable('MecaData_MCA')

#### MecaData_HoxB8
# MecaData_HoxB8 = taka2.getMergedTable('MecaData_HoxB8')

#### Test
# MecaData_Test = taka2.getMergedTable('Test')

MecaData_AtccDrugs = pd.concat([MecaData_Drugs, MecaData_Atcc])

# %%% Test of adding fits

# data_main = MecaData_Drugs
# fitType = 'stressRegion'
# fitId = '_75'

# data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

data_main = MecaData_Atcc
fitType = 'strainGaussian'
fitId = '_0.0125'

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)


# %% > Plotting Functions

# %%% Objects declaration

renameDict1 = {# Variables
               'SurroundingThickness':'Thickness at low force (nm)',
               'surroundingThickness':'Thickness at low force (nm)',
               'ctFieldThickness':'Thickness at low force (nm)',
               'EChadwick': 'E Chadwick (Pa)',
               'medianThickness': 'Median Thickness (nm)',               
               'fluctuAmpli': 'Fluctuations Amplitude (nm)',               
               'meanFluoPeakAmplitude' : 'Fluo Intensity (a.u.)', 
               'fit_K' : 'Tangeantial Modulus (Pa)',
               # Drugs
               'none':'Control',
               'dmso':'DMSO',
               'blebbistatin':'Blebbi',
               'latrunculinA':'LatA',
               'Y27':'Y27',
               }

styleDict1 =  {# Drugs
               'none':{'color': gs.colorList40[10],'marker':'o'},
               'none & 0.0':{'color': gs.colorList40[10],'marker':'o'},
               'dmso':{'color': gs.colorList40[19],'marker':'o'},
               'dmso & 0.0':{'color': gs.colorList40[19],'marker':'o'},
               'blebbistatin':{'color': gs.colorList40[22],'marker':'o'},
               'blebbistatin & 10.0':{'color': gs.colorList40[2],'marker':'o'},
               'blebbistatin & 50.0':{'color': gs.colorList40[22],'marker':'o'},
               'PNB & 50.0':{'color': gs.colorList40[25],'marker':'o'},
               'PNB & 250.0':{'color': gs.colorList40[35],'marker':'o'},
               'latrunculinA':{'color': gs.colorList40[13],'marker':'o'},
               'latrunculinA & 0.5':{'color': gs.colorList40[13],'marker':'o'},
               'latrunculinA & 2.5':{'color': gs.colorList40[33],'marker':'o'},
               'Y27':{'color': gs.colorList40[7],'marker':'o'},
               'Y27 & 10.0':{'color': gs.colorList40[7],'marker':'o'},
               'Y27 & 50.0':{'color': gs.colorList40[27],'marker':'o'},
               # Cell types
               '3T3':{'color': gs.colorList40[30],'marker':'o'},
               'HoxB8-Macro':{'color': gs.colorList40[32],'marker':'o'},
               'DC':{'color': gs.colorList40[33],'marker':'o'},
               # Drugs + cell types
               'aSFL-LG+++ & dmso':{'color': gs.colorList40[19],'marker':'o'},
               'aSFL-LG+++ & blebbistatin':{'color': gs.colorList40[32],'marker':'o'},
               'Atcc-2023 & dmso':{'color': gs.colorList40[19],'marker':'o'},
               'Atcc-2023 & blebbistatin':{'color': gs.colorList40[22],'marker':'o'},
               }

splitterStyleDict_MCA = {'high':'^',
                          'mid':'D',   
                          'low':'v', 
                          'none':'X'}

# %%% Main functions
# These functions use matplotlib.pyplot and seaborn libraries to display 1D categorical or 2D plots

# %%%% Main plots

def D1Plot(data, fig = None, ax = None, condCols=[], Parameters=[], Filters=[], 
           Boxplot=True, AvgPerCell=False, cellID='cellID', co_order=[],
           stats=True, statMethod='Mann-Whitney', box_pairs=[], statVerbose = False,
           figSizeFactor = 1, markersizeFactor = 1, orientation = 'h',
           stressBoxPlot = False, showMean = False, bypassLog = False, 
           returnData = 0, returnCount = 0):
    
    # Filter
    data_f = data
    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for kk in range(len(Filters)):
        globalFilter = globalFilter & Filters[kk]
    data_f = data_f[globalFilter]
    
    # Make cond col
    NCond = len(condCols)
    if NCond == 1:
        condCol = condCols[0]
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += condCols[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_f[newColName] = ''
        for i in range(NCond):
            data_f[newColName] += data_f[condCols[i]].astype(str)
            data_f[newColName] = data_f[newColName].apply(lambda x : x + ' & ')
        data_f[newColName] = data_f[newColName].apply(lambda x : x[:-3])
        condCol = newColName
        
    # Define the count df
    cols_count_df = ['compNum', 'cellID', 'manipID', 'date', condCol]
    count_df = data_f[cols_count_df]

    # Average per cell if necessary
    if AvgPerCell:
        group = data_f.groupby(cellID)
        dictAggMean = getDictAggMean(data_f)
#         dictAggMean['EChadwick'] = 'median'
        data_f = group.agg(dictAggMean)
    
    # Sort data
    data_f.sort_values(condCol, axis=0, ascending=True, inplace=True)
        
    # Define the export df
    cols_export_df = ['date', 'manipID', 'cellID']
    if not AvgPerCell: 
        cols_export_df.append('compNum')
    cols_export_df += ([condCol] + Parameters)
    export_df = data_f[cols_export_df]
    
    # Select style
    NPlots = len(Parameters)
    Conditions = list(data_f[condCol].unique()) 
    if len(co_order) > 0:
        if len(co_order) != len(Conditions):
            delCo = [co for co in co_order if co not in Conditions]
            for co in delCo:
                co_order.remove(co)
        p = getStyleLists_Sns(co_order, styleDict1)
    else: # len(co_order) == 0
        p = sns.color_palette()
        co_order = Conditions

    # Create fig if necessary
    if fig == None:
        if orientation == 'h':
            fig, ax = plt.subplots(1, NPlots, figsize = (5*NPlots*NCond*figSizeFactor, 5))
        elif orientation == 'v':
            fig, ax = plt.subplots(NPlots, 1, figsize = (5*NCond*figSizeFactor, 5*NPlots))
    else:
        pass
        
    markersize = 5*markersizeFactor
    axes = ufun.toList(ax)
    
    for k in range(NPlots):

        if (not bypassLog) and (('EChadwick' in Parameters[k]) or ('K_' in Parameters[k])):
            axes[k].set_yscale('log')
            
        swarmplot_parameters = {'data':    data_f,
                                'x':       condCol,
                                'y':       Parameters[k],
                                'order':   co_order,
                                'palette': p,
                                'size'    : markersize, 
                                'edgecolor'    : 'k', 
                                'linewidth'    : 1*markersizeFactor
                                }

        # sns.swarmplot(x=condCol, y=Parameters[k], data=data_f, ax=axes[k], order = co_order,
        #               size=markersize, edgecolor='k', linewidth = 1*markersizeFactor, palette = p)
        
        sns.swarmplot(ax=axes[k], **swarmplot_parameters)
        
        # if stats == 2:
        #     if len(box_pairs) == 0:
        #         box_pairs = makeBoxPairs(co_order)
        #     addStat_df(axes[k], data_f, box_pairs, Parameters[k], condCol, test = statMethod, verbose = statVerbose)
            
        if stats == 1:
            if len(box_pairs) == 0:
                box_pairs = makeBoxPairs(co_order)
            addStat_lib(ax, box_pairs, test = statMethod, verbose = statVerbose, **swarmplot_parameters)

        if Boxplot:
            boxplot_parameters = {'data':    data_f,
                                    'x':       condCol,
                                    'y':       Parameters[k],
                                    'order':   co_order,
                                    'width' : 0.5,
                                    'showfliers': False,
                                    }
            if stressBoxPlot:
                if stressBoxPlot == 2:
                    boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 4},
                                        boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                                        # boxprops={"color": color, "linewidth": 0.5},
                                        whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                                        capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4})
                                
                elif stressBoxPlot == 1:
                    boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
                                            boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                                            # boxprops={"color": color, "linewidth": 0.5},
                                            whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                                            capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2})
                    
            else:
                boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},
                                        boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                                        # boxprops={"color": color, "linewidth": 0.5},
                                        whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                                        capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2})
                
            if showMean:
                boxplot_parameters.update(meanline='True', showmeans='True',
                                          meanprops={"color": 'darkblue', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},)
                
            sns.boxplot(ax=axes[k], **boxplot_parameters)
            
            


        axes[k].set_xlabel('')
        axes[k].set_ylabel(Parameters[k])
        axes[k].tick_params(axis='x', labelrotation = 10)
        axes[k].yaxis.grid(True)
        if axes[k].get_yscale() == 'linear':
            axes[k].set_ylim([0, axes[k].get_ylim()[1]])
        
    plt.rcParams['axes.prop_cycle'] = gs.my_default_color_cycle
        
    
    # Make output
    
    output = (fig, axes)
    
    if returnData > 0:
        output += (export_df, )
    
    if returnCount > 0:
        groupByCell = count_df.groupby(cellID)
        d_agg = {'compNum':'count', condCol:'first', 'date':'first', 'manipID':'first'}
        df_CountByCell = groupByCell.agg(d_agg).rename(columns={'compNum':'compCount'})

        groupByCond = df_CountByCell.reset_index().groupby(condCol)
        d_agg = {cellID: 'count', 'compCount': 'sum', 
                 'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
        d_rename = {cellID:'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
        df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
        
        if returnCount == 1:
            output += (df_CountByCond, )
        elif returnCount == 2:
            output += (df_CountByCond, df_CountByCell)

    return(output)



def D1Plot_K(data, fig = None, ax = None, condCols=[], Filters=[], AvgPerCell=True, 
            parm = 'fit_K', weightParm = 'fit_ciwK',
            co_order=[], box_pairs=[], stats=True, statMethod='Mann-Whitney', statVerbose = False,
            Boxplot=True, stressBoxPlot = 1, showMean = False, styleDict = styleDict1,
            figSizeFactor = 1, markersizeFactor = 1, scale = 'log',
            returnData = 0, returnCount = 0):
    
    # Filter
    data_f = data
    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for kk in range(len(Filters)):
        globalFilter = globalFilter & Filters[kk]
    data_f = data_f[globalFilter]
    
    # Make cond col
    NCond = len(condCols)
    if NCond == 1:
        condCol = condCols[0]
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += condCols[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_f[newColName] = ''
        for i in range(NCond):
            data_f[newColName] += data_f[condCols[i]].astype(str)
            data_f[newColName] = data_f[newColName].apply(lambda x : x + ' & ')
        data_f[newColName] = data_f[newColName].apply(lambda x : x[:-3])
        condCol = newColName
        
    # Define the id df
    cols_data_id = ['compNum', 'cellID', 'manipID', 'date', condCol]
    data_id = data_f[cols_data_id]

    # Average per cell
    if AvgPerCell:
        valCol, weightCol = parm, weightParm
        data_wAvg = taka2.computeWeightedAverage(data_f, valCol, weightCol, 
                                     groupCol = 'cellID', weight_method = 'ciw')
        #### HERE CORRECTION ON THE WAY
        print(data_wAvg)
        data_wAvg = data_wAvg.drop(data_wAvg[data_wAvg['count_wAvg'] <= 2].index)
        data_wAvg = data_wAvg.drop(data_wAvg[data_wAvg['fit_K_weight_sum'] <= 1].index)
        
        data_id2 = data_id.groupby('cellID').agg('first')
        data_f = pd.merge(data_id2, data_wAvg, how="inner", on='cellID', suffixes=("_x", "_y"))
        data_f = ufun.removeColumnsDuplicate(data_f)
        data_f = data_f.rename(columns={parm + '_wAvg' : parm})
    
    # Sort data
    data_f.sort_values(condCol, axis=0, ascending=True, inplace=True)
    # Define the export df
    cols_export_df = ['date', 'manipID', 'cellID']
    if not AvgPerCell: 
        cols_export_df.append('compNum')
    cols_export_df += ([condCol, parm])
    export_df = data_f[cols_export_df]
    
    # Style & Co_order
    conditions = list(data_f[condCol].unique()) 
    if len(co_order) > 0:
        if len(co_order) != len(conditions):
            delCo = [co for co in co_order if co not in conditions]
            for co in delCo:
                co_order.remove(co) 
        p = getStyleLists_Sns(co_order, styleDict)
    else: # len(co_order) == 0
        p = sns.color_palette()
        co_order = conditions

    # Create fig if necessary
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize = (3*NCond*figSizeFactor, 5))
    else:
        pass
        
    markersize = 5*markersizeFactor
    
    #### HERE CORRECTION ON THE WAY
    data_f['fit_K'] = data_f['fit_K']/1000
    
    swarmplot_parameters = {'ax' : ax,
                            'data':    data_f,
                            'x':       condCol,
                            'y':       parm,
                            'order':   co_order,
                            'palette': p,
                            'size'    : markersize, 
                            'edgecolor'    : 'k', 
                            'linewidth'    : 1*markersizeFactor,
                            'orient' : 'v'
                            }

    sns.swarmplot(**swarmplot_parameters)
    
    if stats == 1:
        if len(box_pairs) == 0:
            box_pairs = makeBoxPairs(co_order)
        swarmplot_parameters.pop('ax')
        addStat_lib(ax, box_pairs, test = statMethod, verbose = statVerbose, **swarmplot_parameters)
        

    if Boxplot:
        boxplot_parameters = {'ax' : ax,
                              'data':    data_f,
                                'x':       condCol,
                                'y':       parm,
                                'order':   co_order,
                                'width' : 0.5,
                                'showfliers': False,
                                'orient' : 'v'
                                }
        if stressBoxPlot:
            if stressBoxPlot == 2:
                boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 4},
                                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                                    # boxprops={"color": color, "linewidth": 0.5},
                                    whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                                    capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                                    meanline='True', showmeans='True',
                                    meanprops={"color": 'darkblue', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},)
                            
            elif stressBoxPlot == 1:
                boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
                                        boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                                        # boxprops={"color": color, "linewidth": 0.5},
                                        whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                                        capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                                        meanline='True', showmeans='True',
                                        meanprops={"color": 'darkblue', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},)
                
        else:
            boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},
                                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                                    # boxprops={"color": color, "linewidth": 0.5},
                                    whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                                    capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                                    meanline='True', showmeans='True',
                                    meanprops={"color": 'darkblue', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},)
            
        if showMean:
            boxplot_parameters.update(meanline='True', showmeans='True',
                                      meanprops={"color": 'darkblue', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},)
            
        sns.boxplot(**boxplot_parameters)

    ax.set_xlabel('')
    ax.set_ylabel(parm)
    ax.tick_params(axis='x', labelrotation = 10)
    ax.yaxis.grid(True)
    if scale == 'log':
        ax.set_yscale('log')
    if scale == 'lin':
        ax.set_yscale('linear')       
    
    # Make output
    
    output = (fig, ax)
    
    if returnData > 0:
        output += (export_df, )
    
    if returnCount > 0:
        groupByCell = data_id.groupby(cellID)
        d_agg = {'compNum':'count', condCol:'first', 'date':'first', 'manipID':'first'}
        df_CountByCell = groupByCell.agg(d_agg).rename(columns={'compNum':'compCount'})

        groupByCond = df_CountByCell.reset_index().groupby(condCol)
        d_agg = {cellID: 'count', 'compCount': 'sum', 
                 'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
        d_rename = {cellID:'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
        df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
        
        if returnCount == 1:
            output += (df_CountByCond, )
        elif returnCount == 2:
            output += (df_CountByCond, df_CountByCell)

    return(output)




def D2Plot_wFit(data, fig = None, ax = None, 
                XCol='', YCol='', condCol='', 
                Filters=[], cellID='cellID', co_order = [],
                AvgPerCell=False, showManips = True,
                modelFit=False, modelType='y=ax+b', writeEqn = True,
                xscale = 'lin', yscale = 'lin', 
                figSizeFactor = 1, markersizeFactor = 1,
                returnData = False):
    
    data_f = data
    for fltr in Filters:
        data_f = data_f.loc[fltr]
    
    NCond = len(condCol)    
    if NCond == 1:
        condCol = condCol[0]
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += condCol[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_f[newColName] = ''
        for i in range(NCond):
            data_f[newColName] += data_f[condCol[i]].astype(str)
            data_f[newColName] = data_f[newColName].apply(lambda x : x + ' & ')
        data_f[newColName] = data_f[newColName].apply(lambda x : x[:-3])
        condCol = newColName
    
    if AvgPerCell:
        group = data_f.groupby(cellID)
        dictAggMean = getDictAggMean(data_f)
        data_f = group.agg(dictAggMean.pop(cellID)) #.reset_index(level=0, inplace=True)
        data_f.reset_index(level=0, inplace=True)
        
    conditions = list(data_f[condCol].unique())
    
    if len(co_order) > 0:
        if len(co_order) != len(conditions):
            delCo = [co for co in co_order if co not in conditions]
            for co in delCo:
                co_order.remove(co)
                
        print(co_order)
        try:
            colorList, markerList = getStyleLists(co_order, styleDict1)
        except:
            colorList, markerList = gs.colorList30, gs.markerList10
    else:
        co_order = conditions
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
    
    for i in range(len(co_order)):
        c = co_order[i]
        color = colorList[i]
#         marker = my_default_marker_list[i]
        Xraw = data_f[data_f[condCol] == c][XCol].values
        Yraw = data_f[data_f[condCol] == c][YCol].values
        Mraw = data_f[data_f[condCol] == c]['manipID'].values
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
                    ax.plot(fitX, fitY, '--', lw = '2', 
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
                    ax.plot(fitX, fitY, '--', lw = '2', 
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
                    ax.plot(fitX, fitY, '--', lw = '2', 
                            color = color, zorder = 4)
                
                print('Number of values : {:.0f}'.format(len(Y)))
                print('\n')
            
#                 if showManips:
#                     allManipID = list(data_f[data_f[condCol] == c]['manipID'].unique())
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
    ax.set_xlim([0.9*np.min(data_f[XCol]), 1.1*np.max(data_f[XCol])])
    ax.set_ylabel(YCol)
    if not yscale == 'log':
        ax.set_ylim([0.9*np.min(data_f[YCol]), 1.1*np.max(data_f[YCol])])
    ax.legend(loc='upper left')
    
    output = (fig, ax)
    
    if returnData:
        output += (data_f, )
    
    return(output)

# %%%% Special Plots

    
def D1Plot_wInnerSplit(data, fig = None, ax = None, condCol=[], InnerSplitCol = [], 
                       Parameters=[], Filters=[], 
                       Boxplot=True, AvgPerCell=False, cellID='cellID', co_order=[],
                       stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                       figSizeFactor = 1, markersizeFactor=1, orientation = 'h', useHue = False, 
                       stressBoxPlot = False, bypassLog = False, returnCount = 0):
    
    data_filtered = data
    # for fltr in Filters:
    #     data_filtered = data_filtered.loc[fltr]

    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for kk in range(len(Filters)):
        globalFilter = globalFilter & Filters[kk]
    data_filtered = data_filtered[globalFilter]
    
    NCond = len(condCol)
    if NCond == 1:
        condCol = condCol[0]
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += condCol[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_filtered[newColName] = ''
        for i in range(NCond):
            data_filtered[newColName] += data_filtered[condCol[i]].astype(str)
            data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x + ' & ')
        data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x[:-3])
        condCol = newColName
        
    NSplits = len(InnerSplitCol)
    if NSplits == 1:
        InnerSplitCol = InnerSplitCol[0]
    elif NSplits > 1:
        newColName = ''
        for i in range(NSplits):
            newColName += InnerSplitCol[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_filtered[newColName] = ''
        for i in range(NSplits):
            data_filtered[newColName] += data_filtered[InnerSplitCol[i]].astype(str)
            data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x + ' & ')
        data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x[:-3])
        InnerSplitCol = newColName
        
    small_df = data_filtered[['cellID', 'compNum', 'date', 'manipID', condCol, InnerSplitCol]]
    
    if AvgPerCell:
        group = data_filtered.groupby(cellID)
        dictAggMean = getDictAggMean(data_filtered)
#         dictAggMean['EChadwick'] = 'median'
        data_filtered = group.agg(dictAggMean)
        
    data_filtered.sort_values(condCol, axis=0, ascending=True, inplace=True)
    
    NPlots = len(Parameters)
    Conditions = list(data_filtered[condCol].unique())  
    Splitters = list(data_filtered[InnerSplitCol].unique())  
    print(Splitters)
    
    if len(co_order) > 0:
        if len(co_order) != len(Conditions):
            delCo = [co for co in co_order if co not in Conditions]
            for co in delCo:
                co_order.remove(co)
                
        p = getStyleLists_Sns(co_order, styleDict1)
        
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
                sns.boxplot(x=condCol, y=Parameters[k], data=data_filtered, ax=ax[k], 
                            width = 0.5, showfliers = False, order= co_order, 
                            medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 4},
                            boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
        #                   boxprops={"color": color, "linewidth": 0.5},
                            whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                            capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4})
                            # scaley = scaley)
            else:
                sns.boxplot(x=condCol, y=Parameters[k], data=data_filtered, ax=ax[k], 
                            width = 0.5, showfliers = False, order= co_order, 
                            medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 1},
                            boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 1},
        #                   boxprops={"color": color, "linewidth": 0.5},
                            whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 1},
                            capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 1})
                            # scaley = scaley)
        
        for split in Splitters:
            marker = splitterStyleDict_MCA[split]
            data_filtered_split = data_filtered[data_filtered[InnerSplitCol] == split]
            sns.stripplot(x=condCol, y=Parameters[k], data=data_filtered_split, ax=ax[k], order = co_order,
                          jitter = True, marker=marker, 
                          size=markersize, edgecolor='k', linewidth = 1*markersizeFactor, 
                          palette = p)
            
            # data_filtered.boxplot(column=Parameters[k], by = condCol, ax=ax[k],showfliers = False) # linewidth = 2, width = 0.5

        if stats:
            if len(box_pairs) == 0:
                box_pairs = makeBoxPairs(co_order)
            addStat_df(ax[k], data_filtered, box_pairs, Parameters[k], condCol, test = statMethod)
            # add_stat_annotation(ax[k], x=condCol, y=Parameters[k], data=data_filtered,box_pairs = box_pairs,test=statMethod, text_format='star',loc='inside', verbose=2)
        
        # if not useHue:
        #     sns.swarmplot(x=condCol, y=Parameters[k], data=data_filtered, ax=ax[k], order = co_order,
        #                   size=markersize, edgecolor='k', linewidth = 1*markersizeFactor, palette = p)
        # else:
        #     sns.swarmplot(x=condCol, y=Parameters[k], data=data_filtered, ax=ax[k], order = co_order,
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
        
    
    if returnCount > 0:
        groupByCell = small_df.groupby(cellID)
        d_agg = {'compNum':'count', condCol:'first', 'date':'first', 'manipID':'first'}
        df_CountByCell = groupByCell.agg(d_agg).rename(columns={'compNum':'compCount'})

        groupByCond = df_CountByCell.reset_index().groupby(condCol)
        d_agg = {cellID: 'count', 'compCount': 'sum', 
                 'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
        d_rename = {cellID:'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
        df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
        
        if returnCount == 1:
            return(fig, ax, df_CountByCond)
        elif returnCount == 2:
            return(fig, ax, df_CountByCond, df_CountByCell)
    
    else:
        return(fig, ax)
    
    
    
    
    
def D1Plot_wNormalize(data, fig = None, ax = None, condCol=[], Parameters=[], Filters=[], 
                      Boxplot=True, AvgPerCell=False, cellID='cellID', co_order=[],
                      stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                      normalizeCol = [], normalizeGroups=[],
                      figSizeFactor = 1, markersizeFactor=1, orientation = 'h', useHue = False, 
                      stressBoxPlot = False, bypassLog = False, returnCount = 0):
    
    data_filtered = data
    # for fltr in Filters:
    #     data_filtered = data_filtered.loc[fltr]
    
    
    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for kk in range(len(Filters)):
        globalFilter = globalFilter & Filters[kk]
    data_filtered = data_filtered[globalFilter]

    
    NCond = len(condCol)
    
    if NCond == 1:
        condCol = condCol[0]
        
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += condCol[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_filtered[newColName] = ''
        for i in range(NCond):
            data_filtered[newColName] += data_filtered[condCol[i]].astype(str)
            data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x + ' & ')
        data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x[:-3])
        condCol = newColName
        
    # print(data_filtered.shape)
    small_df = data_filtered[[condCol, 'cellID', 'compNum', 'date', 'manipID']]
    
    if AvgPerCell:
        group = data_filtered.groupby(cellID)
        dictAggMean = getDictAggMean(data_filtered)
#         dictAggMean['EChadwick'] = 'median'
        data_filtered = group.agg(dictAggMean)
        
    data_filtered.sort_values(condCol, axis=0, ascending=True, inplace=True)
    
    NPlots = len(Parameters)
    Conditions = list(data_filtered[condCol].unique())     
    
    if len(co_order) > 0:
        if len(co_order) != len(Conditions):
            delCo = [co for co in co_order if co not in Conditions]
            for co in delCo:
                co_order.remove(co)
                
        p = getStyleLists_Sns(co_order, styleDict1)
        
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
        
        if normalizeCol == []:
            normalizeCol = condCol
            
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
                sns.boxplot(x=condCol, y=Parm, data=data_filtered, ax=ax[k], 
                            width = 0.5, showfliers = False, order= co_order, 
                            medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 4},
                            boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
        #                   boxprops={"color": color, "linewidth": 0.5},
                            whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                            capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4})
                            # scaley = scaley)
            else:
                sns.boxplot(x=condCol, y=Parm, data=data_filtered, ax=ax[k], 
                            width = 0.5, showfliers = False, order= co_order, 
                            medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 1},
                            boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 1},
        #                   boxprops={"color": color, "linewidth": 0.5},
                            whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 1},
                            capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 1})
                            # scaley = scaley)
            
            # data_filtered.boxplot(column=Parm, by = condCol, ax=ax[k],showfliers = False) # linewidth = 2, width = 0.5

        if stats:
            if len(box_pairs) == 0:
                box_pairs = makeBoxPairs(co_order)
            addStat_df(ax[k], data_filtered, box_pairs, Parm, condCol, test = statMethod)
            # add_stat_annotation(ax[k], x=condCol, y=Parm, data=data_filtered,box_pairs = box_pairs,test=statMethod, text_format='star',loc='inside', verbose=2)
        
        if not useHue:
            sns.swarmplot(x=condCol, y=Parm, data=data_filtered, ax=ax[k], order = co_order,
                          size=markersize, edgecolor='k', linewidth = 1*markersizeFactor, palette = p)
        else:
            sns.swarmplot(x=condCol, y=Parm, data=data_filtered, ax=ax[k], order = co_order,
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
        
    
    if returnCount > 0:
        groupByCell = small_df.groupby(cellID)
        d_agg = {'compNum':'count', condCol:'first', 'date':'first', 'manipID':'first'}
        df_CountByCell = groupByCell.agg(d_agg).rename(columns={'compNum':'compCount'})

        groupByCond = df_CountByCell.reset_index().groupby(condCol)
        d_agg = {cellID: 'count', 'compCount': 'sum', 
                 'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
        d_rename = {cellID:'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
        df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
        
        if returnCount == 1:
            return(fig, ax, df_CountByCond)
        elif returnCount == 2:
            return(fig, ax, df_CountByCond, df_CountByCell)
    
    else:
        return(fig, ax)



def D1PlotDetailed(data, condCol=[], Parameters=[], Filters=[], Boxplot=True, cellID='cellID', 
                   co_order=[], stats=True, statMethod='Mann-Whitney', box_pairs=[],
                   figSizeFactor = 1, markersizeFactor=1, orientation = 'h', showManips = True):
    
    data_f = data
    for fltr in Filters:
        data_f = data_f.loc[fltr]    
    NCond = len(condCol)
    
    print(min(data_f['EChadwick'].values))
    
    if NCond == 1:
        condCol = condCol[0]
        
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += condCol[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_f[newColName] = ''
        for i in range(NCond):
            data_f[newColName] += data_f[condCol[i]].astype(str)
            data_f[newColName] = data_f[newColName].apply(lambda x : x + ' & ')
        data_f[newColName] = data_f[newColName].apply(lambda x : x[:-3])
        condCol = newColName
        
    data_f_agg = getAggDf(data_f, cellID, condCol, Parameters)
    
    NPlots = len(Parameters)
    Conditions = list(data_f[condCol].unique())
        
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
        gs.colorList, mL = getStyleLists(co_order, styleDict1)
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
            sub_data_f_agg = data_f_agg.loc[data_f_agg[condCol] == c]
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
                    box_pairs, Parameters[k], condCol, test = statMethod)
        
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



def D2Plot(data, fig = None, ax = None, XCol='', YCol='', condCol='', Filters=[], 
           cellID='cellID', AvgPerCell=False, xscale = 'lin', yscale = 'lin', 
           figSizeFactor = 1, markers = [], markersizeFactor = 1):
    data_filtered = data
    for fltr in Filters:
        data_filtered = data_filtered.loc[fltr]
    
    NCond = len(condCol)    
    if NCond == 1:
        condCol = condCol[0]
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += condCol[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_filtered[newColName] = ''
        for i in range(NCond):
            data_filtered[newColName] += data_filtered[condCol[i]].astype(str)
            data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x + ' & ')
        data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x[:-3])
        condCol = newColName
    
    if AvgPerCell:
        group = data_filtered.groupby(cellID)
        dictAggMean = getDictAggMean(data_filtered)
        data_filtered = group.agg(dictAggMean.pop(cellID)) #.reset_index(level=0, inplace=True)
        data_filtered.reset_index(level=0, inplace=True)
        
    Conditions = list(data_filtered[condCol].unique())
    
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize = (8*figSizeFactor,5))
    else:
        pass
    
    markersize = 5 * markersizeFactor
    
    if xscale == 'log':
        ax.set_xscale('log')
    if yscale == 'log':
        ax.set_yscale('log')
    
    current_color_list = getStyleLists(Conditions, styleDict1).as_hex()
    cc = cycler(color=current_color_list)
    ax.set_prop_cycle(cc)
    
    im = 0
    for c in Conditions:
        Xraw = data_filtered[data_filtered[condCol] == c][XCol].values
        Yraw = data_filtered[data_filtered[condCol] == c][YCol].values
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




def plotPopKS(data, fitType = 'stressRegion', fitWidth=75, Filters = [], condCol = '', 
              mode = 'wholeCurve', scale = 'lin', printText = True,
              returnData = 0, returnCount = 0):
    
    fig, ax = plt.subplots(1,1, figsize = (9,6))

    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for k in range(0, len(Filters)):
        globalFilter = globalFilter & Filters[k]
    data_f = data[globalFilter]
    
    if mode == 'wholeCurve':
        Sinf, Ssup = 0, np.Inf
        ax.set_xlim([0, 1050])  
        
    else:
        bounds = mode.split('_')
        Sinf, Ssup = int(bounds[0]), int(bounds[1])
        extraFilters = [data_f['minStress'] <= Sinf, data_f['maxStress'] >= Ssup] # >= 800
    
        globalExtraFilter = extraFilters[0]
        for k in range(1, len(extraFilters)):
            globalExtraFilter = globalExtraFilter & extraFilters[k]
        data_f = data_f[globalExtraFilter]
            
        ax.set_xlim([Sinf-50, Ssup+50])     
    
    fitId = '_' + str(fitWidth)
    data_ff = taka2.getFitsInTable(data_f, fitType=fitType, filter_fitID=fitId)
    
    # Filter the table
    data_ff = data_ff[(data_ff['fit_center'] >= Sinf) & (data_ff['fit_center'] <= Ssup)]    
    data_ff = data_ff.drop(data_ff[data_ff['fit_error'] == True].index)
    data_ff = data_ff.drop(data_ff[data_ff['fit_K'] < 0].index)
    data_ff = data_ff.dropna(subset = ['fit_ciwK'])
    
    conditions = np.array(data_ff[condCol].unique())
    
    
    # Compute the weights
    data_ff['weight'] = (data_ff['fit_K']/data_ff['fit_ciwK'])**2
    
    # In the following lines, the weighted average and weighted variance are computed
    # using new columns as intermediates in the computation.
    #
    # Col 'A' = K x Weight --- Used to compute the weighted average.
    # 'K_wAvg' = sum('A')/sum('weight') in each category (group by condCol and 'fit_center')
    #
    # Col 'B' = (K - K_wAvg)**2 --- Used to compute the weighted variance.
    # Col 'C' =  B * Weight     --- Used to compute the weighted variance.
    # 'K_wVar' = sum('C')/sum('weight') in each category (group by condCol and 'fit_center')
    
    # Compute the weighted mean
    data_ff['A'] = data_ff['fit_K'] * data_ff['weight']
    grouped1 = data_ff.groupby(by=[condCol, 'fit_center'])
    data_agg = grouped1.agg({'compNum' : 'count',
                            'A': 'sum', 'weight': 'sum'}).reset_index()
    data_agg['K_wAvg'] = data_agg['A']/data_agg['weight']
    data_agg = data_agg.rename(columns = {'compNum' : 'compCount'})
    
    # Compute the weighted std
    data_ff['B'] = data_ff['fit_K']
    for co in conditions:
        centers = np.array(data_ff[data_ff[condCol] == co]['fit_center'].unique())
        for ce in centers:
            weighted_mean_val = data_agg.loc[(data_agg[condCol] == co) & (data_agg['fit_center'] == ce), 'K_wAvg'].values[0]
            
            index_loc = (data_ff[condCol] == co) & (data_ff['fit_center'] == ce)
            col_loc = 'B'
            data_ff.loc[index_loc, col_loc] = data_ff.loc[index_loc, 'fit_K'] - weighted_mean_val
            data_ff.loc[index_loc, col_loc] = data_ff.loc[index_loc, col_loc] ** 2
            
    data_ff['C'] = data_ff['B'] * data_ff['weight']
    grouped2 = data_ff.groupby(by=[condCol, 'fit_center'])
    data_agg2 = grouped2.agg({'compNum' : 'count',
                              'C': 'sum', 'weight': 'sum'}).reset_index()
    data_agg2['K_wVar'] = data_agg2['C']/data_agg2['weight']
    data_agg2['K_wStd'] = data_agg2['K_wVar']**0.5
    
    # Combine all in data_agg
    data_agg['K_wVar'] = data_agg2['K_wVar']
    data_agg['K_wStd'] = data_agg2['K_wStd']
    data_agg['K_wSte'] = data_agg['K_wStd'] / data_agg['compCount']**0.5
    
    # Plot
    for co in conditions:
        df = data_agg[data_agg[condCol] == co]
        color = styleDict1[co]['color']
        centers = df['fit_center'].values
        Kavg = df['K_wAvg'].values
        Kste = df['K_wSte'].values
        N = df['compCount'].values
        
        dof = N
        alpha = 0.975
        q = st.t.ppf(alpha, dof) # Student coefficient
    
        if scale == 'lin':
            if co == conditions[0]:
                texty = Kavg + 1500
            else:
                texty = texty + 300
            ax.set_yscale('linear')
            ax.set_ylim([0, 10])
                
        elif scale == 'log':
            if co == conditions[0]:
                texty = Kavg**0.95
            else:
                texty = texty**0.98
            ax.set_yscale('log')
            
        # weighted means -- weighted ste 95% as error
        ax.errorbar(centers, Kavg/1000, yerr = q*Kste/1000, 
                    color = color, lw = 2, marker = 'o', markersize = 8, mec = 'k',
                    ecolor = color, elinewidth = 1.5, capsize = 6, capthick = 1.5, 
                    label = co)
        
        # ax.set_title('K(s) - All compressions pooled')
        
        ax.legend(loc = 'upper left', fontsize = 11)
        ax.set_xlabel('Stress (Pa)')
        ax.set_ylabel('K (kPa)')
        ax.grid(visible=True, which='major', axis='y')
        
        if printText:
            for kk in range(len(N)):
                ax.text(x=centers[kk], y=texty[kk]/1000, s='n='+str(N[kk]), fontsize = 8, color = color)
    
    
    # Define the count df
    cols_count_df = ['compNum', 'cellID', 'manipID', 'date', condCol]
    count_df = data_ff[cols_count_df]
    
    # Define the export df
    # cols_export_df = ['date', 'manipID', 'cellID', 'compNum', condCol]
    # export_df = data_ff[cols_export_df]
    cols_export_df = [c for c in data_agg.columns if c not in ['weights', 'A', 'B', 'C']]
    export_df = data_agg[cols_export_df]
    
    # Make output
    
    output = (fig, ax)
    
    if returnData > 0:
        output += (export_df, )
    
    #### NOT FINISHED
    if returnCount > 0:
        groupByCell = count_df.groupby('cellID')
        d_agg = {'compNum':'count', condCol:'first', 'date':'first', 'manipID':'first'}
        df_CountByCell = groupByCell.agg(d_agg).rename(columns={'compNum':'compCount'})

        groupByCond = df_CountByCell.reset_index().groupby(condCol)
        d_agg = {'cellID': 'count', 'compCount': 'sum', 
                 'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
        d_rename = {'cellID':'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
        df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
        
        if returnCount == 1:
            output += (df_CountByCond, )
        elif returnCount == 2:
            output += (df_CountByCond, df_CountByCell)

    return(output)


def plotPopKS_V2(data, fitType = 'stressRegion', fitWidth=75, Filters = [], condCols = [''], 
              mode = 'wholeCurve', scale = 'lin', printText = True, Sinf = 0, Ssup = np.Inf,
              returnData = 0, returnCount = 0):
    
    fig, ax = plt.subplots(1,1, figsize = (9,6))
    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))

    for k in range(0, len(Filters)):
        globalFilter = globalFilter & Filters[k]
    data_f = data[globalFilter]
    
    if mode == 'wholeCurve':
        if Ssup >= 1500:
            xmax = 1550
        else:
            xmax = Ssup + 50
        xmin = Sinf
        ax.set_xlim([xmin, xmax])  
        
    else:
        bounds = mode.split('_')
        Sinf, Ssup = int(bounds[0]), int(bounds[1])
        extraFilters = [data_f['minStress'] <= Sinf, data_f['maxStress'] >= Ssup] # >= 800
    
        globalExtraFilter = extraFilters[0]
        for k in range(1, len(extraFilters)):
            globalExtraFilter = globalExtraFilter & extraFilters[k]
        data_f = data_f[globalExtraFilter]
            
        ax.set_xlim([Sinf-50, Ssup+50])
        
    # Make cond col
    NCond = len(condCols)
    if NCond == 1:
        condCol = condCols[0]
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += condCols[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_f[newColName] = ''
        for i in range(NCond):
            data_f[newColName] += data_f[condCols[i]].astype(str)
            data_f[newColName] = data_f[newColName].apply(lambda x : x + ' & ')
        data_f[newColName] = data_f[newColName].apply(lambda x : x[:-3])
        condCol = newColName
    
    fitId = '_' + str(fitWidth)
    data_ff = taka2.getFitsInTable(data_f, fitType=fitType, filter_fitID=fitId)
    
    # Filter the table
    data_ff = data_ff[(data_ff['fit_center'] >= Sinf) & (data_ff['fit_center'] <= Ssup)]    
    data_ff = data_ff.drop(data_ff[data_ff['fit_error'] == True].index)
    data_ff = data_ff.drop(data_ff[data_ff['fit_K'] < 0].index)
    data_ff = data_ff.dropna(subset = ['fit_ciwK'])
    
    conditions = np.array(data_ff[condCol].unique())
    
    
    # Compute the weights
    data_ff['weight'] = (data_ff['fit_K']/data_ff['fit_ciwK'])**2
    
    # In the following lines, the weighted average and weighted variance are computed
    # using new columns as intermediates in the computation.
    #
    # Col 'A' = K x Weight --- Used to compute the weighted average.
    # 'K_wAvg' = sum('A')/sum('weight') in each category (group by condCol and 'fit_center')
    #
    # Col 'B' = (K - K_wAvg)**2 --- Used to compute the weighted variance.
    # Col 'C' =  B * Weight     --- Used to compute the weighted variance.
    # 'K_wVar' = sum('C')/sum('weight') in each category (group by condCol and 'fit_center')
    
    # Compute the weighted mean
    data_ff['A'] = data_ff['fit_K'] * data_ff['weight']
    grouped1 = data_ff.groupby(by=['cellID', 'fit_center'])
    data_agg_cells = grouped1.agg({'compNum':'count',
                                   'A':'sum', 
                                   'weight': 'sum',
                                   condCol:'first'})
    data_agg_cells = data_agg_cells.reset_index()
    data_agg_cells['K_wAvg'] = data_agg_cells['A']/data_agg_cells['weight']
    data_agg_cells = data_agg_cells.rename(columns = {'compNum' : 'compCount'})
    
    # 2nd selection
    data_agg_cells = data_agg_cells.drop(data_agg_cells[data_agg_cells['compCount'] <= 2].index)
    data_agg_cells = data_agg_cells.drop(data_agg_cells[data_agg_cells['weight'] <= 1].index)
    
    grouped2 = data_agg_cells.groupby(by=[condCol, 'fit_center'])
    data_agg_all = grouped2.agg({'compCount':['sum', 'count'],
                                   'K_wAvg':['mean', 'std', 'median']})
    data_agg_all = data_agg_all.reset_index()
    data_agg_all.columns = ufun.flattenPandasIndex(data_agg_all.columns)
    data_agg_all = data_agg_all.rename(columns = {'compCount_sum' : 'compCount', 'compCount_count' : 'cellCount'})
    data_agg_all['K_wAvg_ste'] = data_agg_all['K_wAvg_std']/data_agg_all['cellCount']**0.5
    
    
    # Plot
    i_color = 0
    
    for co in conditions:
        try:
            color = styleDict1[co]['color']
        except:
            color = gs.colorList10[i_color%10]
            i_color = i_color + 1
            
        df = data_agg_all[data_agg_all[condCol] == co]
        
        centers = df['fit_center'].values
        Kavg = df['K_wAvg_mean'].values
        Kste = df['K_wAvg_ste'].values
        N = df['cellCount'].values
        total_N = np.max(N)
        
        n = df['compCount'].values
        total_n = np.max(n)
        
        dof = N
        alpha = 0.975
        q = st.t.ppf(alpha, dof) # Student coefficient
    
        if scale == 'lin':
            if co == conditions[0]:
                texty = Kavg + 1500
            else:
                texty = texty + 300
            ax.set_yscale('linear')
            ax.set_ylim([0, 10])
                
        elif scale == 'log':
            if co == conditions[0]:
                texty = Kavg**0.95
            else:
                texty = texty**0.98
            ax.set_yscale('log')
            
        # weighted means -- weighted ste 95% as error
        ax.errorbar(centers, Kavg/1000, yerr = q*Kste/1000, 
                    color = color, lw = 2, marker = 'o', markersize = 8, mec = 'k',
                    ecolor = color, elinewidth = 1.5, capsize = 6, capthick = 1.5, 
                    label = co + ' | ' + str(total_N) + ' cells' + ' | ' + str(total_n) + ' comp')
        
        # ax.set_title('K(s) - All compressions pooled')
        
        ax.legend(loc = 'upper left', fontsize = 11)
        ax.set_xlabel('Stress (Pa)')
        ax.set_ylabel('K (kPa)')
        ax.grid(visible=True, which='major', axis='y')
        
        if printText:
            for kk in range(len(N)):
                ax.text(x=centers[kk], y=texty[kk]/1000, s='n='+str(N[kk]), fontsize = 8, color = color)
    
    
    # # Define the count df
    # cols_count_df = ['compNum', 'cellID', 'manipID', 'date', condCol]
    # count_df = data_ff[cols_count_df]
    
    # # Define the export df
    # # cols_export_df = ['date', 'manipID', 'cellID', 'compNum', condCol]
    # # export_df = data_ff[cols_export_df]
    # cols_export_df = [c for c in data_agg_all.columns if c not in ['weights', 'A', 'B', 'C']]
    # export_df = data_agg_all[cols_export_df]
    
    # Make output
    
    output = (fig, ax)
    
    # if returnData > 0:
    #     output += (export_df, )
    
    # #### NOT FINISHED
    # if returnCount > 0:
    #     groupByCell = count_df.groupby('cellID')
    #     d_agg = {'compNum':'count', condCol:'first', 'date':'first', 'manipID':'first'}
    #     df_CountByCell = groupByCell.agg(d_agg).rename(columns={'compNum':'compCount'})

    #     groupByCond = df_CountByCell.reset_index().groupby(condCol)
    #     d_agg = {'cellID': 'count', 'compCount': 'sum', 
    #               'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
    #     d_rename = {'cellID':'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
    #     df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
        
    #     if returnCount == 1:
    #         output += (df_CountByCond, )
    #     elif returnCount == 2:
    #         output += (df_CountByCond, df_CountByCell)

    return(output)



def plotAllH0(data, Filters = [], condCols = [], 
              co_order = [], box_pairs = [], AvgPerCell = True,
              stats = False, statMethod = 'Mann-Whitney', stressBoxPlot = 1,
              ratio = False, ref_ratio = ''):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    Filters : TYPE, optional
        DESCRIPTION. The default is [].
    condCols : TYPE, optional
        DESCRIPTION. The default is [].
    co_order : TYPE, optional
        DESCRIPTION. The default is [].
    box_pairs : TYPE, optional
        DESCRIPTION. The default is [].
    AvgPerCell : TYPE, optional
        DESCRIPTION. The default is True.
    stats : TYPE, optional
        DESCRIPTION. The default is False.
    statMethod : TYPE, optional
        DESCRIPTION. The default is 'Mann-Whitney'.
    stressBoxPlot : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    fig, ax
    
    Example
    -------
    
    >>> data_main = MecaData_Drugs
    >>> dates = ['22-03-28', '22-03-30']
    >>> Filters = [(data_main['validatedThickness'] == True),
    >>>            (data_main['bestH0'] <= 1200),
    >>>            (data_main['drug'].apply(lambda x : x in ['dmso'])),
    >>>            (data_main['date'].apply(lambda x : x in dates))]
    >>> fig, ax = plotAllH0(data_main, Filters = Filters, condCols = [], 
    >>>         co_order = [], box_pairs = [], AvgPerCell = True,
    >>>         stats = False, statMethod = 'Mann-Whitney', stressBoxPlot = 2)
    >>> plt.show()

    """

    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for k in range(0, len(Filters)):
        globalFilter = globalFilter & Filters[k]
    data_f = data[globalFilter]

    data_ff = taka2.getAllH0InTable(data_f)
    
    # Filter the table   
    data_ff = data_ff.drop(data_ff[data_ff['allH0_error'] == True].index)
    
    # Make cond col
    condCols = condCols + ['allH0_method', 'allH0_zone']
    NCond = len(condCols)
    newColName = ''
    for i in range(NCond):
        newColName += condCols[i]
        newColName += ' & '
    newColName = newColName[:-3]
    data_ff[newColName] = ''
    for i in range(NCond):
        data_ff[newColName] += data_ff[condCols[i]].astype(str)
        data_ff[newColName] = data_ff[newColName].apply(lambda x : x + ' & ')
    data_ff[newColName] = data_ff[newColName].apply(lambda x : x[:-3])
    condCol = newColName
    
    conditions = np.array(data_ff[condCol].unique())
    
    if ratio:
        # Make H0_m_z col
        H0_m_z = ['allH0_method', 'allH0_zone']
        NCond = len(condCols)
        newColName = 'allH0_method_zone'
        data_ff[newColName] = ''
        for i in range(2):
            data_ff[newColName] += data_ff[H0_m_z[i]].astype(str)
            data_ff[newColName] = data_ff[newColName].apply(lambda x : x + '_')
        data_ff[newColName] = data_ff[newColName].apply(lambda x : x[:-1])
        H0_m_z = newColName
        
        H0_types = np.array(data_ff[H0_m_z].unique())
        if ref_ratio == '':
            ref_ratio = H0_types[0]
        
        cells = np.array(data_ff['cellID'].unique())
        data_ff['allH0_ratioH0'] = data_ff['allH0_H0']
        for c in cells:
            ref_H0 = data_ff.loc[(data_ff['cellID'] == c) & (data_ff[H0_m_z] == ref_ratio), 'allH0_H0'].values[0]
            data_ff.loc[(data_ff['cellID'] == c), 'allH0_ratioH0'] = data_ff.loc[(data_ff['cellID'] == c), 'allH0_ratioH0']/ref_H0
            
    
    
    if len(co_order) > 0:
        if len(co_order) != len(conditions):
            delCo = [co for co in co_order if co not in conditions]
            for co in delCo:
                co_order.remove(co)
    else:
        co_order = conditions
    
    # Average per cell if necessary
    if AvgPerCell:
        dictAggMean = {}
        for c in data_ff.columns:
            try :
                if np.array_equal(data_ff[c], data_ff[c].astype(bool)):
                    dictAggMean[c] = 'min'
                else:
                    try:
                        if not c.isnull().all():
                            np.mean(data_ff[c])
                            dictAggMean[c] = 'mean'
                    except:
                        dictAggMean[c] = 'first'
            except:
                    dictAggMean[c] = 'first'
        
        group = data_ff.groupby(by = ['cellID', 'allH0_method', 'allH0_zone'])
        data_ff = group.agg(dictAggMean)
    
        
    # Sort data
    data_ff.sort_values(condCol, axis=0, ascending=True, inplace=True)

    # output = data_ff
    
    # Create fig
    fig, ax = plt.subplots(1, 1, figsize = (5*NCond, 5))
    
    if ratio:
        Y='allH0_ratioH0'
    else:
        Y='allH0_H0'

    if stressBoxPlot == 0:
        sns.boxplot(x=condCol, y=Y, data=data_ff, ax=ax,
                    width = 0.5, showfliers = False, order= co_order, 
                    medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},
                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
#                   boxprops={"color": color, "linewidth": 0.5},
                    whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                    capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2})
                    # scaley = scaley)
                        
    elif stressBoxPlot == 1:
        sns.boxplot(x=condCol, y=Y, data=data_ff, ax=ax, 
                    width = 0.5, showfliers = False, order= co_order, 
                    medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
#                   boxprops={"color": color, "linewidth": 0.5},
                    whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                    capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2})
                    # scaley = scaley)
                    
    if stressBoxPlot == 2:
        sns.boxplot(x=condCol, y=Y, data=data_ff, ax=ax, 
                    width = 0.5, showfliers = False, order= co_order, 
                    medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 4},
                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
#                   boxprops={"color": color, "linewidth": 0.5},
                    whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                    capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4})
                    # scaley = scaley)
                
    if stats:
        if len(box_pairs) == 0:
            box_pairs = makeBoxPairs(co_order)
        addStat_df(ax, data_ff, box_pairs, Parameters[k], condCol, test = statMethod)
    
    sns.swarmplot(x=condCol, y=Y, data=data_ff, ax=ax, order = co_order)

    ax.set_xlabel('')
    ax.set_ylabel('H0 (nm)')
    ax.tick_params(axis='x', labelrotation = 10)
    ax.yaxis.grid(True)
    if ax.get_yscale() == 'linear':
        ax.set_ylim([0, ax.get_ylim()[1]])

    # Make output    
    output = (fig, ax)

    return(output)


# %%% Subfunctions

def getDictAggMean(df):
    dictAggMean = {}
    for c in df.columns:
    #         t = df[c].dtype
    #         print(c, t)
            try :
                if np.array_equal(df[c], df[c].astype(bool)):
                    dictAggMean[c] = 'min'
                else:
                    try:
                        if not c.isnull().all():
                            np.mean(df[c])
                            dictAggMean[c] = 'mean'
                    except:
                        dictAggMean[c] = 'first'
            except:
                    dictAggMean[c] = 'first'
    return(dictAggMean)


def getAggDf(df, cellID, condCol, Variables):
    
    allCols = df.columns.values
    group = df.groupby(cellID)
    
    dictAggCategories = {}
    Categories = ['date', 'cellName', 'manipID', 'experimentType', 'drug', 
                  'substrate', 'cell type', 'cell subtype', 'bead type', 'bead diameter']
    if condCol not in Categories:
        Categories += [condCol]
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


def getAggDf_K_wAvg(df, cellID, condCol, Variables, weightCols):
    
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
    
    dictweights = {}
    for i in range(len(Variables)):
        v = Variables[i]
        w = weightCols[i]
        dictweights[v] = w

    for v in Variables:
        df[v] = df[v].apply(nan2zero)
        df[dictweights[v]] = df[dictweights[v]].apply(nan2zero)
    cellIDvals = df['cellID'].unique()
    for v in Variables:
        for c in cellIDvals:
            if np.sum(df.loc[df['cellID'] == c, v].values) == 0:
                df.loc[df['cellID'] == c, dictweights[v]].apply(lambda x: 0)
    
    
    
    allCols = df.columns.values
    group = df.groupby(cellID)
    
    dictAggCategories = {}
    Categories = ['date', 'cellName', 'manipID', 'experimentType', 'drug', 
                  'substrate', 'cell type', 'cell subtype', 'bead type', 'bead diameter']
    if condCol not in Categories:
        Categories += [condCol]
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
        w = weightCols[i]
        df_i = df[[cellID, v, w]]
        group_i = df_i.groupby(cellID)

        def WMfun(x):
            try:
                return(np.average(x, weights=df.loc[x.index, dictweights[v]]))
            except:
                return(np.nan)
            
        def WSfun(x):
            try:
                return(w_std(x, df.loc[x.index, dictweights[v]].values))
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


def renameAxes(axes, rD, format_xticks = True):
    try:
        N = len(axes)
    except:
        axes = [axes]
        N = 1
    for i in range(N):
        # set xlabel
        xlabel = axes[i].get_xlabel()
        newXlabel = rD.get(xlabel, xlabel)
        axes[i].set_xlabel(newXlabel)
        # set ylabel
        ylabel = axes[i].get_ylabel()
        newYlabel = rD.get(ylabel, ylabel)
        axes[i].set_ylabel(newYlabel)
        
        if format_xticks:
            # set xticks
            xticksTextObject = axes[i].get_xticklabels()
            xticksList = [xticksTextObject[j].get_text() for j in range(len(xticksTextObject))]
            test_hasXLabels = (len(''.join(xticksList)) > 0)
            if test_hasXLabels:
                newXticksList = [rD.get(k, k) for k in xticksList]
                axes[i].set_xticklabels(newXticksList)
                
                
                
def renameLegend(axes, rD):
    axes = ufun.toList(axes)
    N = len(axes)
    for i in range(N):
        ax = axes[i]
        L = ax.legend()
        Ltext = L.get_texts()
        M = len(Ltext)
        for j in range(M):
            T = Ltext[j].get_text()
            for s in rD.keys():
                if re.search(s, T):
                    Ltext[j].set_text(re.sub(s, rD[s], T))
    
                
        
def addStat_lib(ax, box_pairs, test = 'Mann-Whitney', verbose = False, **plotting_parameters):
    listTests = ['t-test_ind', 't-test_welch', 't-test_paired', 
                 'Mann-Whitney', 'Mann-Whitney-gt', 'Mann-Whitney-ls', 
                 'Levene', 'Wilcoxon', 'Kruskal', 'Brunner-Munzel']
    if test in listTests:
        annotator = Annotator(ax, box_pairs, **plotting_parameters)
        annotator.configure(test=test, verbose=verbose).apply_and_annotate()
    else:
        print(gs.BRIGHTORANGE + 'Dear Madam, dear Sir, i am the eternal god and i command that you have to define this stat test cause it is not in the list !' + gs.NORMAL)
    return(ax)

def addStat_lib_noR(box_pairs, test = 'Mann-Whitney', verbose = False, **plotting_parameters):
    listTests = ['t-test_ind', 't-test_welch', 't-test_paired', 
                 'Mann-Whitney', 'Mann-Whitney-gt', 'Mann-Whitney-ls', 
                 'Levene', 'Wilcoxon', 'Kruskal', 'Brunner-Munzel']
    if test in listTests:
        annotator = Annotator(box_pairs, **plotting_parameters)
        annotator.configure(test=test, verbose=verbose).apply_and_annotate()
    else:
        print(gs.BRIGHTORANGE + 'Dear Madam, dear Sir, i am the eternal god and i command that you have to define this stat test cause it is not in the list !' + gs.NORMAL)


def addStat_df(ax, data, box_pairs, param, cond, test = 'Mann-Whitney', percentHeight = 99, verbose = False):
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
        ax.plot([bp[0], bp[1]], [currentHeight, currentHeight], 'k-', lw = 1.5)
        XposText = (dictXTicks[bp[0]]+dictXTicks[bp[1]])/2
        if scale == 'log':
            power = 0.01* (text=='ns') + 0.000 * (text!='ns')
            YposText = currentHeight*(refHeight**power)
        else:
            factor = 0.025 * (text=='ns') + 0.000 * (text!='ns')
            YposText = currentHeight + factor*refHeight
        ax.text(XposText, YposText, text, ha = 'center', color = 'k', size = 12 + 2*(text!='ns'))
#         if text=='ns':
#             ax.text(posText, currentHeight + 0.025*refHeight, text, ha = 'center')
#         else:
#             ax.text(posText, currentHeight, text, ha = 'center')
        if scale == 'log':
            currentHeight = currentHeight*(refHeight**0.05)
        else:
            currentHeight =  currentHeight + 0.125*refHeight
            
        if verbose:
            print(bp)
            print(statistic, pval)
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


#### Test getAggDf_K_wAvg(df, cellID, condCol, Variables, weightCols)

# data = MecaData_NonLin

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
# Kweight_Cols = []

# for rFN in regionFitsNames:
#     listColumnsMeca += ['KChadwick_'+rFN, 'K_CIW_'+rFN, 'R2Chadwick_'+rFN, 'K2Chadwick_'+rFN, 
#                         'H0Chadwick_'+rFN, 'Npts_'+rFN, 'validatedFit_'+rFN]
#     KChadwick_Cols += [('KChadwick_'+rFN)]

#     K_CIWidth = data_f['K_CIW_'+rFN] #.apply(lambda x : x.strip('][').split(', ')).apply(lambda x : (np.abs(float(x[0]) - float(x[1]))))
#     Kweight = (data_f['KChadwick_'+rFN]/K_CIWidth) # **2
#     data_f['K_weight_'+rFN] = Kweight
#     data_f['K_weight_'+rFN] *= data_f['KChadwick_'+rFN].apply(lambda x : (x<1e6))
#     data_f['K_weight_'+rFN] *= data_f['R2Chadwick_'+rFN].apply(lambda x : (x>1e-2))
#     data_f['K_weight_'+rFN] *= data_f['K_CIW_'+rFN].apply(lambda x : (x!=0))
#     Kweight_Cols += [('K_weight_'+rFN)]
    

# # dictweights = {}
# # for i in range(len(Variables)):
# #     v = Variables[i]
# #     w = weightCols[i]
# #     dictweights[v] = w
# # def nan2zero(x):
# #     if np.isnan(x):
# #         return(0)
# #     else:
# #         return(x)

# # for v in Variables:
# #     df[v] = df[v].apply(nan2zero)
# #     df[dictweights[v]] = df[dictweights[v]].apply(nan2zero)
# # cellIDvals = df['cellID'].unique()
# # for v in Variables:
# #     for c in cellIDvals:
# #         if np.sum(df.loc[df['cellID'] == c, v].values) == 0:
# #             df.loc[df['cellID'] == c, dictweights[v]].apply(lambda x: 1)
            
# # df

# condCol = 'date'
# Variables = KChadwick_Cols
# weightCols = Kweight_Cols
# data_f_agg = getAggDf_K_wAvg(data_f, 'cellID', condCol, Variables, weightCols)
# data_f_agg


#### Test getAggDf(df, cellID, condCol, Variables)

# data = MecaData_All

# Filters = [(data['validatedFit'] == True), (data['validatedThickness'] == True)]

# data_f = data
# for fltr in Filters:
#     data_f = data_f.loc[fltr]

# # dfA = getAggDf(data_f, 'cellID', 'bead type', ['surroundingThickness', 'EChadwick'])
# dfA = getAggDf(data_f, 'cellID', 'bead type', ['ctFieldThickness', 'EChadwick'])
# dfA


#### Test the D2Plot_wFit
# data = MecaData_All

# Filters = [(data['validatedFit'] == True), 
#            (data['validatedThickness'] == True), 
#            (data['substrate'] == '20um fibronectin discs'),
#            (data['date'].apply(lambda x : x in ['21-12-16','21-12-08']))]

# fig, ax = D2Plot_wFit(data, XCol='ctFieldThickness', YCol='EChadwick', condCol = ['bead type'],
#            Filters=Filters, cellID = 'cellID', AvgPerCell=True, xscale = 'log', yscale = 'log', 
#            modelFit=True, modelType = 'y=k*x^a')

# ax.set_ylabel('EChadwick (Pa)')
# ax.set_xlabel('Thickness at low force (nm)')
# fig.suptitle('3T3aSFL: E(h)')
# ax.legend(loc = 'upper right')

# # ufun.archiveFig(fig, ax, name='E(h)_3T3aSFL_Dec21_M450-5-13_vs_M270-14-54', figDir = cp.DirDataFigToday + '//' + figSubDir)
# plt.show()


#### Test of the averaging per cell routine

# data = MecaData_All
# condCol='drug'
# Parameters=['SurroundingThickness','EChadwick']
# Filters = [(MecaData_All['Validated'] == 1)]
# AvgPerCell=True
# cellID='CellName'

# data_filtered = data
# for fltr in Filters:
#     data_filtered = data_filtered.loc[fltr]

# group = data_filtered.groupby(cellID)
# dictAggMean = getDictAggMean(data_filtered)
# data_filtered = group.agg(dictAggMean.pop(cellID)) #.reset_index(level=0, inplace=True)
# data_filtered.reset_index(level=0, inplace=True)
# data_filtered=data_filtered[[cellID]+[condCol]+Parameters]
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


# %%% In Dev

# %%%% D1Plot inter-intra variability

def D1Plot_vars(data, fig = None, ax = None, condCols=[], Parameters=[], Filters=[], 
           cellID='cellID', co_order=[],
           stats=True, statMethod='Mann-Whitney', box_pairs=[], statVerbose = False,
           figSizeFactor = 1, markersizeFactor = 1, stressBoxPlot = False,
           returnData = 0, returnCount = 0):
    
    # Filter
    data_filtered = data
    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for kk in range(len(Filters)):
        globalFilter = globalFilter & Filters[kk]
    data_filtered = data_filtered[globalFilter]
    
    # Make cond col
    NCond = len(condCols)
    if NCond == 1:
        condCol = condCols[0]
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += condCols[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_filtered[newColName] = ''
        for i in range(NCond):
            data_filtered[newColName] += data_filtered[condCols[i]].astype(str)
            data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x + ' & ')
        data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x[:-3])
        condCol = newColName

    # Average per cell if necessary
    # if AvgPerCell:
    #     group = data_filtered.groupby(cellID)
    #     dictAggMean = getDictAggMean(data_filtered)
    #     data_filtered = group.agg(dictAggMean)
    
    # Sort data
    data_filtered.sort_values(condCol, axis=0, ascending=True, inplace=True)

    # Select style
    NPlots = len(Parameters)
    Conditions = list(data_filtered[condCol].unique()) 
    if len(co_order) > 0:
        if len(co_order) != len(Conditions):
            delCo = [co for co in co_order if co not in Conditions]
            for co in delCo:
                co_order.remove(co)
        p = getStyleLists_Sns(co_order, styleDict1)
    else: 
        p = sns.color_palette()
        co_order = Conditions

    # Create fig if necessary
    if fig == None:
        fig, ax = plt.subplots(2, NPlots, figsize = (5*NPlots*NCond*figSizeFactor, 8))
    else:
        pass
        
    markersize = 5*markersizeFactor
    axes = ufun.toList(ax)
    axes = axes.flatten()
    
    # Make two new dataset
    
    # 1. Group by cellID    
    group1 = data_filtered.groupby(cellID)
    d_agg = {'compNum': 'count', 
             condCol: 'first', 
             'date': 'first', 
             'manipID': 'first'}
    for P in Parameters:
        d_agg[P] = [np.mean, np.median, np.std, ufun.interDeciles]
    
    df_Cells = group1.agg(d_agg)
    df_Cells = df_Cells.reset_index()
    df_Cells['id_within_co'] = [int(np.sum(df_Cells.loc[:i,condCol] == df_Cells.loc[i,condCol])) for i in range(df_Cells.shape[0])]
    df_Cells.columns = ufun.flattenPandasIndex(df_Cells.columns)
    
    d_rename = {'compNum_count': 'compCount',
                condCol + '_first': condCol,
                'date_first': 'date', 
                'manipID_first': 'manipID'}
    # for P in Parameters:
    #     d_rename[P + '_mean'] = P
    df_Cells = df_Cells.rename(columns=d_rename)

    # 2. Group by condition
    group2 = df_Cells.reset_index().groupby(condCol)
    d_agg = {cellID: 'count', 
             'compCount': 'sum', 
             'date': pd.Series.nunique, 
             'manipID': pd.Series.nunique}
    for P in Parameters:
        d_agg[P + '_mean'] = [np.mean, np.median, np.std, ufun.interDeciles]
        d_agg[P + '_median'] = [np.mean, np.median, np.std, ufun.interDeciles]
        d_agg[P + '_std'] = 'mean'
        d_agg[P + '_interDeciles'] = 'mean'
        
    df_Cond = group2.agg(d_agg)
    df_Cond = df_Cond.reset_index()

    df_Cond = group2.agg(d_agg)
    df_Cond = df_Cond.reset_index()
    df_Cond.columns = ufun.flattenPandasIndex(df_Cond.columns)
    
    d_rename = {cellID + '_count': 'cellCount',
                'compCount_sum': 'compCount', 
                'date_nunique': 'datesCount', 
                'manipID_nunique': 'manipsCount'}
    # for P in Parameters:
    #     d_rename[P + '_mean'] = P
    df_Cond = df_Cond.rename(columns=d_rename)
    
    data_filtered = data_filtered.merge(df_Cells[[cellID, 'id_within_co']], how='left', on=cellID)
    
    #### 1. Intercells
    i = 0
    for j in range(NPlots):
        k = 2*j+i
        
        #### HERE
        swarmplot_parameters = {'data':    df_Cells,
                                'x':       condCol,
                                'y':       Parameters[j] + '_mean',
                                'order':   co_order,
                                'palette': p,
                                'size'   : markersize, 
                                'edgecolor'    : 'k', 
                                'linewidth'    : 1*markersizeFactor
                                }
        
        sns.swarmplot(ax=axes[k], **swarmplot_parameters)
                    
        if stats:
            if len(box_pairs) == 0:
                box_pairs = makeBoxPairs(co_order)
            addStat_lib(axes[k], box_pairs, test = statMethod, verbose = statVerbose, **swarmplot_parameters)

        boxplot_parameters = {'data':    df_Cells,
                                'x':       condCol,
                                'y':       Parameters[j] + '_mean',
                                'order':   co_order,
                                'width' : 0.5,
                                'showfliers': False,
                                }
        if stressBoxPlot:
            if stressBoxPlot == 2:
                boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 4},
                                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                                    whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                                    capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                                    meanline='True', showmeans='True',
                                    meanprops={"color": 'darkblue', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},)
                            
            elif stressBoxPlot == 1:
                boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
                                        boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                                        whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                                        capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                                        meanline='True', showmeans='True',
                                        meanprops={"color": 'darkblue', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},)
                
        else:
            boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},
                                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                                    whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                                    capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                                    meanline='True', showmeans='True',
                                    meanprops={"color": 'darkblue', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},)
            
        sns.boxplot(ax=axes[k], **boxplot_parameters)
        
        for ii in range(len(co_order)):
            Co = co_order[ii]
            median = df_Cond[df_Cond[condCol] ==  Co][Parameters[j] + '_mean_median'].values[0]
            intercell_std = df_Cond[df_Cond[condCol] ==  Co][Parameters[j] + '_mean_std'].values[0]
            intercell_interD = df_Cond[df_Cond[condCol] ==  Co][Parameters[j] + '_mean_interDeciles'].values[0]
            intracell_std = df_Cond[df_Cond[condCol] ==  Co][Parameters[j] + '_std_mean'].values[0]
            intracell_interD = df_Cond[df_Cond[condCol] ==  Co][Parameters[j] + '_interDeciles_mean'].values[0]
            # print(median)
            x1 = ii - 0.3
            # axes[k].annotate('', xy=(x1, median - intercell_std), 
            #                  xytext=(x1, median + intercell_std), 
            #                  arrowprops=dict(arrowstyle='<->', color = 'red'),
            #                  )
            axes[k].annotate('', 
                             xy=(x1, median - intercell_interD/2), 
                              xytext=(x1, median + intercell_interD/2), 
                              arrowprops=dict(arrowstyle='<->', color = 'red'),
                              )
            axes[k].text(x = x1 - 0.01, y  = median, 
                         s = 'INTER\n{:.2f}'.format(intercell_interD),
                         ha='right', va='center', fontsize=8)
            
            x2 = ii + 0.3
            # axes[k].annotate('', xy=(x2, median - intracell_std), 
            #                  xytext=(x2, median + intracell_std), 
            #                  arrowprops=dict(arrowstyle='<->', color = 'blue'),
            #                  )
            axes[k].annotate('', 
                             xy=(x2, median - intracell_interD/2), 
                             xytext=(x2, median + intracell_interD/2), 
                             arrowprops=dict(arrowstyle='<->', color = 'blue'),
                             )
            axes[k].text(x = x2 + 0.01, y  = median, 
                         s = 'INTRA\n{:.2f}'.format(intracell_interD),
                         ha='left', va='center', fontsize=8)
        
        axes[k].legend().set_visible(False)
        axes[k].set_xlabel('')
        axes[k].set_ylabel(Parameters[j])
        axes[k].tick_params(axis='x', labelrotation = 10)
        axes[k].yaxis.grid(True)
        if axes[k].get_yscale() == 'linear':
            axes[k].set_ylim([0, axes[k].get_ylim()[1]])
            
    #### 2. Intracells
    i = 1
    for j in range(NPlots):
        k = 2*j+i
        
        #### HERE    
        boxplot_parameters = {'data':    data_filtered,
                                'x':       condCol,
                                'y':       Parameters[j],
                                'hue':     'id_within_co',
                                'order':   co_order,
                                'width' : 0.5,
                                'showfliers': False,
                                }

        boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 0.75, 'alpha' : 0.8, 'zorder' : 2},
                                boxprops={"linewidth": 0.75, 'alpha' : 1, 'zorder' : 2},
                                whiskerprops={"linewidth": 0.75, 'alpha' : 1, 'zorder' : 2},
                                capprops={"linewidth": 0.75, 'alpha' : 1, 'zorder' : 2},
                                meanline='True', showmeans='True',
                                meanprops={"color": 'darkblue', "linewidth": 0.75, 'alpha' : 0.8, 'zorder' : 2},)
            
        sns.boxplot(ax=axes[k], **boxplot_parameters)
        
        axes[k].legend().set_visible(False)
        axes[k].set_xlabel('')
        axes[k].set_ylabel(Parameters[j])
        axes[k].tick_params(axis='x', labelrotation = 10)
        axes[k].yaxis.grid(True)
        if axes[k].get_yscale() == 'linear':
            axes[k].set_ylim([0, axes[k].get_ylim()[1]])
        
    
    # Make output
    
    output = (fig, ax, df_Cells, df_Cond)
    
    
    
    if returnData > 0:
        # Define the export df
        cols_export_df = ['date', 'manipID', 'cellID']
        # if not AvgPerCell: 
        #     cols_export_df.append('compNum')
        cols_export_df += ([condCol] + Parameters)
        export_df = data_filtered[cols_export_df]
        
        output += (export_df, )
    
    if returnCount > 0:
        # Define the count df
        cols_count_df = ['compNum', 'cellID', 'manipID', 'date', condCol]
        count_df = data_filtered[cols_count_df]
        
        groupByCell = count_df.groupby(cellID)
        d_agg = {'compNum':'count', condCol:'first', 'date':'first', 'manipID':'first'}
        df_CountByCell = groupByCell.agg(d_agg).rename(columns={'compNum':'compCount'})

        groupByCond = df_CountByCell.reset_index().groupby(condCol)
        d_agg = {cellID: 'count', 'compCount': 'sum', 
                 'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
        d_rename = {cellID:'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
        df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
        
        if returnCount == 1:
            output += (df_CountByCond, )
        elif returnCount == 2:
            output += (df_CountByCond, df_CountByCell)

    return(output)




# %%%% TEST : Compare the controls 23-03-08 & 23-03-09

rD = {'none & 0.0' : 'No drug',
      'blebbistatin & 10.0' : 'blebbistatin\n(10µM)', 
      'blebbistatin & 50.0' : 'blebbistatin\n(50µM)', 
      'Thickness at low force (nm)' : 'Thickness (nm)'}

data_main = MecaData_Atcc
dates = ['23-03-08', '23-03-09'] # ['22-03-28', '22-03-30', '22-11-23']
substrate = '20um fibronectin discs'
thicknessType = 'surroundingThickness' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['substrate'] == substrate),
            # (data['drug'] == 'Y27'),
           (data['date'].apply(lambda x : x in dates))]

co_order = ['none', 'Y27']

out = D1Plot_vars(data, condCols=['drug'], Parameters=[thicknessType], Filters=Filters, 
                                        cellID='cellID', co_order=co_order, 
                                        stats=False, statMethod='Mann-Whitney', box_pairs=[], 
                                        figSizeFactor = 2.0, markersizeFactor=1.2, stressBoxPlot=1)

fig, ax1 = out[:2]

renameAxes(ax1, renameDict1)
renameAxes(ax1, rD)

plt.show()

# %%%% K = f(Strain)

def plotPopKStrain(data, fitType = 'strainGaussian', fitWidth=0.0125, Filters = [], condCols = [''], 
              mode = 'wholeCurve', scale = 'lin', printText = True, Sinf = 0, Ssup = np.Inf, 
              metric = 'mean',
              returnData = 0, returnCount = 0):
    
    fig, ax = plt.subplots(1,1, figsize = (9,6))

    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    print(len(Filters))
    for k in range(0, len(Filters)):
        globalFilter = globalFilter & Filters[k]
    data_f = data[globalFilter]
    
    if mode == 'wholeCurve':
        if Ssup >= 0.35:
            xmax = 0.35
        else:
            xmax = Ssup + 0.025
        xmin = Sinf
        ax.set_xlim([xmin, xmax])  
        
    else:
        bounds = mode.split('_')
        Sinf, Ssup = float(bounds[0]), float(bounds[1])
        extraFilters = [data_f['minStrain'] <= Sinf, data_f['maxStrain'] >= Ssup] # >= 800
    
        globalExtraFilter = extraFilters[0]
        for k in range(1, len(extraFilters)):
            globalExtraFilter = globalExtraFilter & extraFilters[k]
        data_f = data_f[globalExtraFilter]
            
        ax.set_xlim([Sinf-0.025, Ssup+0.025])
        
    # Make cond col
    NCond = len(condCols)
    if NCond == 1:
        condCol = condCols[0]
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += condCols[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_f[newColName] = ''
        for i in range(NCond):
            data_f[newColName] += data_f[condCols[i]].astype(str)
            data_f[newColName] = data_f[newColName].apply(lambda x : x + ' & ')
        data_f[newColName] = data_f[newColName].apply(lambda x : x[:-3])
        condCol = newColName
    
    fitId = '_' + str(fitWidth)
    data_ff = taka2.getFitsInTable(data_f, fitType=fitType, filter_fitID=fitId)
    
    # Filter the table
    data_ff = data_ff[(data_ff['fit_center'] >= Sinf) & (data_ff['fit_center'] <= Ssup)]    
    data_ff = data_ff.drop(data_ff[data_ff['fit_error'] == True].index)
    data_ff = data_ff.drop(data_ff[data_ff['fit_K'] < 0].index)
    data_ff = data_ff.dropna(subset = ['fit_ciwK'])
    
    conditions = np.array(data_ff[condCol].unique())
    
    
    # Compute the weights
    data_ff['weight'] = (data_ff['fit_K']/data_ff['fit_ciwK'])**2
    
    # In the following lines, the weighted average and weighted variance are computed
    # using new columns as intermediates in the computation.
    #
    # Col 'A' = K x Weight --- Used to compute the weighted average.
    # 'K_wAvg' = sum('A')/sum('weight') in each category (group by condCol and 'fit_center')
    #
    # Col 'B' = (K - K_wAvg)**2 --- Used to compute the weighted variance.
    # Col 'C' =  B * Weight     --- Used to compute the weighted variance.
    # 'K_wVar' = sum('C')/sum('weight') in each category (group by condCol and 'fit_center')
    
    # Compute the weighted mean
    data_ff['A'] = data_ff['fit_K'] * data_ff['weight']
    grouped1 = data_ff.groupby(by=['cellID', 'fit_center'])
    data_agg_cells = grouped1.agg({'compNum':'count',
                                   'A':'sum', 
                                   'weight': 'sum',
                                   condCol:'first'})
    data_agg_cells = data_agg_cells.reset_index()
    data_agg_cells['K_wAvg'] = data_agg_cells['A']/data_agg_cells['weight']
    data_agg_cells = data_agg_cells.rename(columns = {'compNum' : 'compCount'})
    
    # 2nd selection
    data_agg_cells = data_agg_cells.drop(data_agg_cells[data_agg_cells['compCount'] <= 2].index)
    data_agg_cells = data_agg_cells.drop(data_agg_cells[data_agg_cells['weight'] <= 1].index)
    
    grouped2 = data_agg_cells.groupby(by=[condCol, 'fit_center'])
    data_agg_all = grouped2.agg({'compCount':['sum', 'count'],
                                   'K_wAvg':['mean', 'std', 'median']})
    data_agg_all = data_agg_all.reset_index()
    data_agg_all.columns = ufun.flattenPandasIndex(data_agg_all.columns)
    data_agg_all = data_agg_all.rename(columns = {'compCount_sum' : 'compCount', 'compCount_count' : 'cellCount'})
    data_agg_all['K_wAvg_ste'] = data_agg_all['K_wAvg_std']/data_agg_all['cellCount']**0.5
    
    # Compute the weighted std
    # data_ff['B'] = data_ff['fit_K']
    # for co in conditions:
    #     centers = np.array(data_ff[data_ff[condCol] == co]['fit_center'].unique())
    #     for ce in centers:
    #         weighted_mean_val = data_agg.loc[(data_agg[condCol] == co) & (data_agg['fit_center'] == ce), 'K_wAvg'].values[0]
            
    #         index_loc = (data_ff[condCol] == co) & (data_ff['fit_center'] == ce)
    #         col_loc = 'B'
    #         data_ff.loc[index_loc, col_loc] = data_ff.loc[index_loc, 'fit_K'] - weighted_mean_val
    #         data_ff.loc[index_loc, col_loc] = data_ff.loc[index_loc, col_loc] ** 2
            
    # data_ff['C'] = data_ff['B'] * data_ff['weight']
    # grouped2 = data_ff.groupby(by=[condCol, 'fit_center'])
    # data_agg2 = grouped2.agg({'compNum' : 'count',
    #                           'C': 'sum', 'weight': 'sum'}).reset_index()
    # data_agg2['K_wVar'] = data_agg2['C']/data_agg2['weight']
    # data_agg2['K_wStd'] = data_agg2['K_wVar']**0.5
    
    # # Combine all in data_agg
    # data_agg['K_wVar'] = data_agg2['K_wVar']
    # data_agg['K_wStd'] = data_agg2['K_wStd']
    # data_agg['K_wSte'] = data_agg['K_wStd'] / data_agg['compCount']**0.5
    
    # Plot
    i_color = 0
    
    for co in conditions:
        try:
            color = styleDict1[co]['color']
        except:
            color = gs.colorList10[i_color%10]
            i_color = i_color + 1
            
        df = data_agg_all[data_agg_all[condCol] == co]
        
        centers = df['fit_center'].values
        if metric == 'mean':
            Kavg = df['K_wAvg_mean'].values
        elif metric == 'median':
            Kavg = df['K_wAvg_median'].values
            
        Kste = df['K_wAvg_ste'].values
        N = df['cellCount'].values
        total_N = np.max(N)
        
        dof = N
        alpha = 0.975
        q = st.t.ppf(alpha, dof) # Student coefficient
    
        if scale == 'lin':
            if co == conditions[0]:
                texty = Kavg + 1500
            else:
                texty = texty + 300
            ax.set_yscale('linear')
            ax.set_ylim([0, 10])
                
        elif scale == 'log':
            if co == conditions[0]:
                texty = Kavg**0.95
            else:
                texty = texty**0.98
            ax.set_yscale('log')
            
        # weighted means -- weighted ste 95% as error
        ax.errorbar(centers, Kavg/1000, yerr = q*Kste/1000, 
                    color = color, lw = 2, marker = 'o', markersize = 8, mec = 'k',
                    ecolor = color, elinewidth = 1.5, capsize = 6, capthick = 1.5, 
                    label = co + ' | ' + str(total_N) + ' cells')
        
        # ax.set_title('K(s) - All compressions pooled')
        
        ax.legend(loc = 'upper left', fontsize = 11)
        ax.set_xlabel('Strain (S.U.)')
        ax.set_ylabel('K (kPa)')
        ax.grid(visible=True, which='major', axis='y')
        
        if printText:
            for kk in range(len(N)):
                ax.text(x=centers[kk], y=texty[kk]/1000, s='n='+str(N[kk]), fontsize = 8, color = color)
    
    
    # # Define the count df
    # cols_count_df = ['compNum', 'cellID', 'manipID', 'date', condCol]
    # count_df = data_ff[cols_count_df]
    
    # # Define the export df
    # # cols_export_df = ['date', 'manipID', 'cellID', 'compNum', condCol]
    # # export_df = data_ff[cols_export_df]
    # cols_export_df = [c for c in data_agg_all.columns if c not in ['weights', 'A', 'B', 'C']]
    # export_df = data_agg_all[cols_export_df]
    
    # Make output
    
    output = (fig, ax)
    
    # if returnData > 0:
    #     output += (export_df, )
    
    # #### NOT FINISHED
    # if returnCount > 0:
    #     groupByCell = count_df.groupby('cellID')
    #     d_agg = {'compNum':'count', condCol:'first', 'date':'first', 'manipID':'first'}
    #     df_CountByCell = groupByCell.agg(d_agg).rename(columns={'compNum':'compCount'})

    #     groupByCond = df_CountByCell.reset_index().groupby(condCol)
    #     d_agg = {'cellID': 'count', 'compCount': 'sum', 
    #               'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
    #     d_rename = {'cellID':'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
    #     df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
        
    #     if returnCount == 1:
    #         output += (df_CountByCond, )
    #     elif returnCount == 2:
    #         output += (df_CountByCond, df_CountByCell)

    return(output)

# %%%% TEST


# MAKE SURE YOU GET THE SAME MEAN VALUES HERE !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

fitType = 'strainGaussian'
fitId = '0.1_0.0125'
c, hw = np.array(fitId.split('_')).astype(float)
# fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)
fitStr = 'Fit from {:.2f} to {:.2f} %'.format((c-hw)*100, (c+hw)*100)

rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27\n(10µM)', 
      'Y27 & 50.0' : 'Y27\n(50µM)', 
      'dmso & 0.0' : 'DMSO', 
      'blebbistatin & 10.0' : 'Blebbi\n(10µM)',  
      'blebbistatin & 50.0' : 'Blebbi\n(50µM)', 
      'Thickness at low force (nm)' : 'Thickness (nm)',
      'Tangeantial Modulus (Pa)' : 'Tangeantial Modulus (kPa)\n' + fitStr}


# Part 1.

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)
# data['fit_K'] = data['fit_K']/1e3

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           # (data['fit_valid'] == True),
            (data['fit_K'] <= 15e3),
            (data['fit_K'] > 0),
            (data['fit_error'] == False),
            (data['fit_ciwK'].apply(lambda x : pd.isnull(x) == False)),
            (data['substrate'] == substrate),
            (data['date'].apply(lambda x : x in dates))]

co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0', 'dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0']
box_pairs = [['none & 0.0', 'dmso & 0.0'],
              ['none & 0.0', 'Y27 & 10.0'],['none & 0.0', 'Y27 & 50.0'], ['Y27 & 10.0', 'Y27 & 50.0'], 
              ['dmso & 0.0', 'blebbistatin & 10.0'],['dmso & 0.0', 'blebbistatin & 50.0'], ['blebbistatin & 10.0', 'blebbistatin & 50.0']]


# fig1, ax1 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=['fit_K'], Filters=Filters, 
#                 Boxplot=True, cellID='cellID', co_order=co_order, 
#                 AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
#                 figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

fig11, ax11 = D1Plot_K(data, condCols=['drug', 'concentration'], Filters=Filters, AvgPerCell=True, 
                        parm = 'fit_K', weightParm = 'fit_ciwK',
                        co_order=co_order, box_pairs=box_pairs, stats=True, statMethod='Mann-Whitney', 
                        Boxplot=True, stressBoxPlot = 1, showMean = True, styleDict = styleDict1,
                        figSizeFactor = 1.4, markersizeFactor = 1, scale = 'lin',
                        returnData = 0, returnCount = 0)




# renameAxes(ax1, renameDict1)
# renameAxes(ax1, rD)
renameAxes(ax11, renameDict1)
renameAxes(ax11, rD)


data = MecaData_Atcc
dates = ['23-03-08', '23-03-09', '23-03-16', '23-03-17']
substrate = '20um fibronectin discs'

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
           (data['substrate'] == substrate),
           (data['date'].apply(lambda x : x in dates)),
          ]

out2 = plotPopKStrain(data, fitType = 'strainGaussian', fitWidth=0.025, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = '0.075_0.125', 
                                scale = 'lin', printText = False, metric = 'median',
                                returnData = 1, returnCount = 1)
fig2, ax2 = out2
ax2.set_ylim([0, 25])

out21 = plotPopKStrain(data, fitType = 'strainGaussian', fitWidth=0.025, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = '0.10_0.15', 
                                scale = 'lin', printText = False, metric = 'median',
                                returnData = 1, returnCount = 1)
fig21, ax21 = out21
ax21.set_ylim([0, 25])

out22 = plotPopKStrain(data, fitType = 'strainGaussian', fitWidth=0.025, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = '0.125_0.175', 
                                scale = 'lin', printText = False, metric = 'median',
                                returnData = 1, returnCount = 1)
fig22, ax22 = out22
ax22.set_ylim([0, 25])



plt.show()

# %% Plots

# %%%  2023 ATCC

# %%%% 23-02-16

# %%%%% H & K

# fig, axes = plt.subplots(2,2, figsize = (9,7))

# Part 1.

data_main = MecaData_Atcc
dates = ['23-02-16'] # ['22-03-28', '22-03-30', '22-11-23']

fitType = 'stressRegion'
fitId = '300_100'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

thicknessType = 'surroundingThickness' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data['fit_K'] = data['fit_K']/1e3

Filters = [(data['validatedThickness'] == True), 
           (data['valid_Chadwick'] == True),
           # (data['surroundingThickness'] <= 700),
           (data['fit_valid'] == True),
           # (data['fit_K'] <= 10),
            # (data['ctFieldMinThickness'] >= 200**2),
           (data['drug'].apply(lambda x : x in ['dmso', 'blebbistatin'])),
           (data['date'].apply(lambda x : x in dates))]

# co_order=['none', 'dmso', 'blebbistatin', 'latrunculinA']
co_order=['dmso & 0.0', 'blebbistatin & 50.0']

fig, axes = D1Plot(data, condCols=['drug', 'concentration'], Parameters=[thicknessType, 'fit_K'], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 0.4, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)


rD = {'dmso & 0.0' : 'DMSO', 
      'blebbistatin & 50.0' : 'Blebbi\n(50µM)', 
      'latrunculinA & 0.5' : 'LatA\n(0.5µM)', 
      'latrunculinA & 2.5' : 'LatA\n(2.5µM)',
      'Thickness at low force (nm)' : 'Thickness (nm)',
      'Tangeantial Modulus (Pa)' : 'Tangeantial Modulus (kPa)\n' + fitStr}

renameAxes(axes.flatten(), renameDict1)
renameAxes(axes.flatten(), rD)

# titles = ['Blebbistatin', 'LatrunculinA']
# tl = matplotlib.ticker.MultipleLocator(2)

# for k in range(2):
#     axes[0,k].set_ylim([0,900])
#     axes[1,k].set_ylim([0,12.5])
#     axes[k,1].set_ylabel('')
#     # axes[0,k].set_xticklabels([])
#     axes[0,k].set_title(titles[k])
#     axes[1,k].yaxis.set_major_locator(tl)



# ax[0].set_ylim([0, 1000])
# ax[0].legend(loc = 'upper right', fontsize = 8)
# ax[1].legend(loc = 'upper right', fontsize = 8)
# fig.suptitle('3T3aSFL & drugs\nPreliminary data')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_H&Echad_simple', figSubDir = figSubDir)
plt.show()



# %%%%% K(s)

data = MecaData_Atcc
dates = ['23-02-16']

Filters = [(data['validatedThickness'] == True),
            (data['UI_Valid'] == True),
            (data['date'].apply(lambda x : x in dates)),
            (data['drug'] != 'none'),
            ]

# out1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
#                                 condCol = 'drug', mode = '200_400', scale = 'lin', printText = True,
#                                 returnData = 1, returnCount = 1)
# fig1, ax1, exportDf1, countDf1 = out1
# ax1.set_ylim([0, 10])

out2 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'drug', mode = 'wholeCurve', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig2, ax2 = out2
ax2.set_ylim([0, 20])

out2 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'drug', mode = '300_500', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig2, ax2 = out2
ax2.set_ylim([0, 10])

# data_ff1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False)
# data_ff2 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = '150_550', scale = 'lin', printText = True)

# fig1.suptitle('3T3 vs. HoxB8 - stiffness')
# fig2.suptitle('3T3 vs. HoxB8 - stiffness')
plt.show()

# %%%%% K(s) for drugs


#### Making the Dataframe 

data = MecaData_Drugs
dates = ['22-03-30']
drugs = ['dmso', 'blebbistatin']

Filters = [(data['validatedThickness'] == True),
            (data['UI_Valid'] == True),
            (data['date'].apply(lambda x : x in dates)),
            (data['drug'].apply(lambda x : x in drugs)),
            ]

out1 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'drug', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)
fig1, ax1 = out1
ax1.set_ylim([0, 20])

out2 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'drug', mode = '300_500', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)
fig2, ax2 = out2
ax2.set_ylim([0, 10])

# data_ff1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False)
# data_ff2 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = '150_550', scale = 'lin', printText = True)

# fig1.suptitle('3T3 vs. HoxB8 - stiffness')
# fig2.suptitle('3T3 vs. HoxB8 - stiffness')
plt.show()

# %%%%% K(s)

data = MecaData_AtccDrugs.reset_index()
dates = ['22-03-28', '22-03-30', '23-02-16']
drugs = ['dmso', 'blebbistatin']

condCols = ['cell subtype', 'drug']

NCond = len(condCols)
if NCond == 1:
    condCol = condCols[0]
elif NCond > 1:
    newColName = ''
    for i in range(NCond):
        newColName += condCols[i]
        newColName += ' & '
    newColName = newColName[:-3]
    data[newColName] = ''
    for i in range(NCond):
        data[newColName] += data[condCols[i]].astype(str)
        data[newColName] = data[newColName].apply(lambda x : x + ' & ')
    data[newColName] = data[newColName].apply(lambda x : x[:-3])
    condCol = newColName

Filters = [(data['validatedThickness'] == True),
            (data['UI_Valid'] == True),
            (data['date'].apply(lambda x : x in dates)),
            (data['drug'].apply(lambda x : x in drugs))
            ]

# out1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
#                                 condCol = 'drug', mode = '200_400', scale = 'lin', printText = True,
#                                 returnData = 1, returnCount = 1)
# fig1, ax1, exportDf1, countDf1 = out1
# ax1.set_ylim([0, 10])

out = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'cell subtype & drug', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)
fig, ax = out
ax.set_ylim([0, 20])

out2 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'cell subtype & drug', mode = '300_500', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)
fig2, ax2 = out2
ax2.set_ylim([0, 10])

# data_ff1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False)
# data_ff2 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = '150_550', scale = 'lin', printText = True)

# fig1.suptitle('3T3 vs. HoxB8 - stiffness')
# fig2.suptitle('3T3 vs. HoxB8 - stiffness')
plt.show()


# %%%% 23-02-23

# %%%%% H and K PNB

# fig, axes = plt.subplots(2,2, figsize = (9,7))

fitType = 'stressGaussian'
fitId = '300_100'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)

rD = {'none & 0.0' : 'No drug',
      'dmso & 0.0' : 'DMSO', 
      'PNB & 50.0' : 'PNB\n(50µM)', 
      'PNB & 250.0' : 'PNB\n(250µM)', 
      'Thickness at low force (nm)' : 'Thickness (nm)',
      'Tangeantial Modulus (Pa)' : 'Tangeantial Modulus (kPa)\n' + fitStr}

# Part 1.1

data_main = MecaData_Atcc
dates = ['23-02-23'] # ['22-03-28', '22-03-30', '22-11-23']
thicknessType = 'surroundingThickness' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['date'].apply(lambda x : x in dates))]

co_order = ['none & 0.0', 'dmso & 0.0', 'PNB & 50.0', 'PNB & 250.0']

fig, ax1 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

renameAxes(ax1, renameDict1)
renameAxes(ax1, rD)

# Part 1.2

data_main = MecaData_Atcc
dates = ['23-02-23'] # ['22-03-28', '22-03-30', '22-11-23']
thicknessType = 'bestH0' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['date'].apply(lambda x : x in dates))]

co_order = ['none & 0.0', 'dmso & 0.0', 'PNB & 50.0', 'PNB & 250.0']

fig, ax11 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

renameAxes(ax11, renameDict1)
renameAxes(ax11, rD)

# Part 2.

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

data['fit_K'] = data['fit_K']/1e3

Filters = [(data['validatedThickness'] == True), 
           (data['fit_valid'] == True),
            (data['fit_K'] <= 10),
            # (data['ctFieldMinThickness'] >= 200**2),
            (data['date'].apply(lambda x : x in dates))]

co_order = ['none & 0.0', 'dmso & 0.0', 'PNB & 50.0', 'PNB & 250.0']

fig, ax2 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=['fit_K'], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)




renameAxes(ax2, renameDict1)
renameAxes(ax2, rD)

# titles = ['Blebbistatin', 'LatrunculinA']
# tl = matplotlib.ticker.MultipleLocator(2)

# for k in range(2):
#     axes[0,k].set_ylim([0,900])
#     axes[1,k].set_ylim([0,12.5])
#     axes[k,1].set_ylabel('')
#     # axes[0,k].set_xticklabels([])
#     axes[0,k].set_title(titles[k])
#     axes[1,k].yaxis.set_major_locator(tl)



# ax[0].set_ylim([0, 1000])
# ax[0].legend(loc = 'upper right', fontsize = 8)
# ax[1].legend(loc = 'upper right', fontsize = 8)
# fig.suptitle('3T3aSFL & drugs\nPreliminary data')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_H&Echad_simple', figSubDir = figSubDir)
plt.show()


# %%%%% H and K PNB 2 dates

# fig, axes = plt.subplots(2,2, figsize = (9,7))

fitType = 'stressGaussian'
fitId = '300_100'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)

rD = {'none & 0.0' : 'No drug',
      'dmso & 0.0' : 'DMSO', 
      'PNB & 50.0' : 'PNB\n(50µM)', 
      'PNB & 250.0' : 'PNB\n(250µM)', 
      'Thickness at low force (nm)' : 'Thickness (nm)',
      'Tangeantial Modulus (Pa)' : 'Tangeantial Modulus (kPa)\n' + fitStr}

# Part 1.1

data_main = MecaData_Atcc
dates = ['23-02-16', '23-02-23'] # ['22-03-28', '22-03-30', '22-11-23']
thicknessType = 'surroundingThickness' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['date'].apply(lambda x : x in dates))]

co_order = ['none & 0.0', 'dmso & 0.0', 'blebbistatin & 50.0', 'PNB & 50.0', 'PNB & 250.0']

fig, ax1 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

renameAxes(ax1, renameDict1)
renameAxes(ax1, rD)

# Part 1.2

data_main = MecaData_Atcc
dates = ['23-02-16', '23-02-23'] # ['22-03-28', '22-03-30', '22-11-23']
thicknessType = 'bestH0' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['date'].apply(lambda x : x in dates))]

co_order = ['none & 0.0', 'dmso & 0.0', 'blebbistatin & 50.0', 'PNB & 50.0', 'PNB & 250.0']

fig, ax11 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

renameAxes(ax11, renameDict1)
renameAxes(ax11, rD)

# Part 2.

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

data['fit_K'] = data['fit_K']/1e3

Filters = [(data['validatedThickness'] == True), 
           (data['fit_valid'] == True),
            (data['fit_K'] <= 10),
            # (data['ctFieldMinThickness'] >= 200**2),
            (data['date'].apply(lambda x : x in dates))]

co_order = ['none & 0.0', 'dmso & 0.0', 'blebbistatin & 50.0', 'PNB & 50.0', 'PNB & 250.0']

fig, ax2 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=['fit_K'], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)




renameAxes(ax2, renameDict1)
renameAxes(ax2, rD)

# titles = ['Blebbistatin', 'LatrunculinA']
# tl = matplotlib.ticker.MultipleLocator(2)

# for k in range(2):
#     axes[0,k].set_ylim([0,900])
#     axes[1,k].set_ylim([0,12.5])
#     axes[k,1].set_ylabel('')
#     # axes[0,k].set_xticklabels([])
#     axes[0,k].set_title(titles[k])
#     axes[1,k].yaxis.set_major_locator(tl)



# ax[0].set_ylim([0, 1000])
# ax[0].legend(loc = 'upper right', fontsize = 8)
# ax[1].legend(loc = 'upper right', fontsize = 8)
# fig.suptitle('3T3aSFL & drugs\nPreliminary data')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_H&Echad_simple', figSubDir = figSubDir)
plt.show()

# %%%%% H and K PNB 2 dates

# fig, axes = plt.subplots(2,2, figsize = (9,7))

fitType = 'stressGaussian'
fitId = '300_100'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)

rD = {'none & 0.0' : 'No drug',
      'dmso & 0.0' : 'DMSO', 
      'PNB & 50.0' : 'PNB\n(50µM)', 
      'PNB & 250.0' : 'PNB\n(250µM)', 
      'Thickness at low force (nm)' : 'Thickness (nm)',
      'Tangeantial Modulus (Pa)' : 'Tangeantial Modulus (kPa)\n' + fitStr}

# Part 1.1

data_main = MecaData_Atcc
dates = ['23-02-16', '23-02-23'] # ['22-03-28', '22-03-30', '22-11-23']
thicknessType = 'surroundingThickness' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['date'].apply(lambda x : x in dates)),
           (data['drug'].apply(lambda x : x in ['dmso']))]

co_order = ['dmso & 23-02-16', 'dmso & 23-02-23']

fig, ax1 = D1Plot(data, condCols=['drug', 'date'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

renameAxes(ax1, renameDict1)
renameAxes(ax1, rD)

# Part 1.2

data_main = MecaData_Atcc
dates = ['23-02-16', '23-02-23'] # ['22-03-28', '22-03-30', '22-11-23']
thicknessType = 'bestH0' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['date'].apply(lambda x : x in dates)),
           (data['drug'].apply(lambda x : x in ['dmso']))]

co_order = ['dmso & 23-02-16', 'dmso & 23-02-23']

fig, ax11 = D1Plot(data, condCols=['drug', 'date'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

renameAxes(ax11, renameDict1)
renameAxes(ax11, rD)


plt.show()

# %%%%% K(s) PNB

data = MecaData_Atcc
dates = ['23-02-23']

Filters = [(data['validatedThickness'] == True),
            (data['UI_Valid'] == True),
            (data['date'].apply(lambda x : x in dates)),
            ]

# out1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
#                                 condCol = 'drug', mode = '200_400', scale = 'lin', printText = True,
#                                 returnData = 1, returnCount = 1)
# fig1, ax1, exportDf1, countDf1 = out1
# ax1.set_ylim([0, 10])

out2 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = 'wholeCurve', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig2, ax2 = out2
ax2.set_ylim([0, 20])

out2 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = '300_500', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig2, ax2 = out2
ax2.set_ylim([0, 10])

# data_ff1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False)
# data_ff2 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = '150_550', scale = 'lin', printText = True)

# fig1.suptitle('3T3 vs. HoxB8 - stiffness')
# fig2.suptitle('3T3 vs. HoxB8 - stiffness')
plt.show()

# %%%%% Diverse H0

data_main = MecaData_Atcc

dfH0 = taka2.getMatchingFits(data_main, fitType = 'H0', output = 'df')

dates = ['23-02-23'] # ['22-03-28', '22-03-30']

data_wH0 = taka2.getAllH0InTable(data_main)

Filters = [(data_main['validatedThickness'] == True),
           (data_main['bestH0'] <= 1200),
           (data_main['drug'].apply(lambda x : x in ['none'])),
           (data_main['date'].apply(lambda x : x in dates))]

fig, ax = plotAllH0(data_main, Filters = Filters, condCols = [], 
        co_order = [], box_pairs = [], AvgPerCell = True,
        stats = False, statMethod = 'Mann-Whitney', stressBoxPlot = 2)

plt.show()

# %%%%% Diverse H0_ratio

data_main = MecaData_Atcc

dfH0 = taka2.getMatchingFits(data_main, fitType = 'H0', output = 'df')

dates = ['23-02-23'] # ['22-03-28', '22-03-30']

data_wH0 = taka2.getAllH0InTable(data_main)

Filters = [(data_main['validatedThickness'] == True),
           (data_main['bestH0'] <= 1200),
           (data_main['drug'].apply(lambda x : x in ['none'])),
           (data_main['date'].apply(lambda x : x in dates))]

fig1, ax1 = plotAllH0(data_main, Filters = Filters, condCols = [], 
        co_order = [], box_pairs = [], AvgPerCell = True,
        stats = False, statMethod = 'Mann-Whitney', stressBoxPlot = 2,
        ratio = True, ref_ratio= 'Dimitriadis_%f_10')

fig2, ax2 = plotAllH0(data_main, Filters = Filters, condCols = [], 
        co_order = [], box_pairs = [], AvgPerCell = True,
        stats = False, statMethod = 'Mann-Whitney', stressBoxPlot = 2,
        ratio = True, ref_ratio= 'Chadwick_%f_10')

plt.show()

# %%%% 23-03-08 & 23-03-09

# %%%%% Compare the controls 23-03-08 & 23-03-09

fitType = 'stressGaussian'
fitId = '300_100'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)

rD = {'none & 0.0' : 'No drug',
      'blebbistatin & 10.0' : 'blebbistatin\n(10µM)', 
      'blebbistatin & 50.0' : 'blebbistatin\n(50µM)', 
      'Thickness at low force (nm)' : 'Thickness (nm)',
      'Tangeantial Modulus (Pa)' : 'Tangeantial Modulus (kPa)\n' + fitStr}

# Part 1.1

data_main = MecaData_Atcc
dates = ['23-03-08', '23-03-09'] # ['22-03-28', '22-03-30', '22-11-23']
substrate = '20um fibronectin discs'
thicknessType = 'surroundingThickness' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['substrate'] == substrate),
           (data['drug'] == 'none'),
           (data['date'].apply(lambda x : x in dates))]

co_order = ['23-03-08 & none', '23-03-09 & none']

fig, ax1 = D1Plot(data, condCols=['date', 'drug'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

renameAxes(ax1, renameDict1)
renameAxes(ax1, rD)

# Part 1.2

data_main = MecaData_Atcc
dates = ['23-03-08', '23-03-09'] # ['22-03-28', '22-03-30', '22-11-23']
substrate = '20um fibronectin discs'
thicknessType = 'bestH0' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['substrate'] == substrate),
           (data['drug'] == 'none'),
           (data['date'].apply(lambda x : x in dates))]

co_order = ['23-03-08 & none', '23-03-09 & none']

fig, ax11 = D1Plot(data, condCols=['date', 'drug'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

renameAxes(ax11, renameDict1)
renameAxes(ax11, rD)

# Part 2.

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

data['fit_K'] = data['fit_K']/1e3

Filters = [(data['validatedThickness'] == True), 
           (data['fit_valid'] == True),
            (data['fit_K'] <= 10),
            (data['substrate'] == substrate),
            (data['drug'] == 'none'),
            (data['date'].apply(lambda x : x in dates))]

co_order = ['23-03-08 & none', '23-03-09 & none']

fig, ax2 = D1Plot(data, condCols=['date', 'drug'], Parameters=['fit_K'], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)




renameAxes(ax2, renameDict1)
renameAxes(ax2, rD)


# Part 3.

data = MecaData_Atcc
dates = ['23-03-08', '23-03-09']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
           (data['substrate'] == substrate),
           (data['drug'] == 'none'),
           (data['date'].apply(lambda x : x in dates)),
          ]

# out1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
#                                 condCol = 'drug', mode = '200_400', scale = 'lin', printText = True,
#                                 returnData = 1, returnCount = 1)
# fig1, ax1, exportDf1, countDf1 = out1
# ax1.set_ylim([0, 10])

out31 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['date', 'drug'], mode = '250_450', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig31, ax31 = out31
ax31.set_ylim([0, 20])

out32 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['date', 'drug'], mode = '350_550', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig32, ax32 = out32
ax32.set_ylim([0, 25])

out33 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['date', 'drug'], mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)
fig33, ax33 = out33
ax33.set_ylim([0, 25])

# data_ff1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False)
# data_ff2 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = '150_550', scale = 'lin', printText = True)

# fig1.suptitle('3T3 vs. HoxB8 - stiffness')
# fig2.suptitle(


plt.show()


# %%%%% H and K Y27

# fig, axes = plt.subplots(2,2, figsize = (9,7))

fitType = 'stressGaussian'
fitId = '300_100'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)

rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27\n(10µM)', 
      'Y27 & 50.0' : 'Y27\n(50µM)', 
      'Thickness at low force (nm)' : 'Thickness (nm)',
      'Tangeantial Modulus (Pa)' : 'Tangeantial Modulus (kPa)\n' + fitStr}

# Part 1.1

data_main = MecaData_Atcc
dates = ['23-03-08', '23-03-09'] # ['22-03-28', '22-03-30', '22-11-23']
substrate = '20um fibronectin discs'
thicknessType = 'surroundingThickness' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['substrate'] == substrate),
           (data['date'].apply(lambda x : x in dates))]

co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

fig, ax1 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)



renameAxes(ax1, renameDict1)
renameAxes(ax1, rD)

# Part 1.2

data_main = MecaData_Atcc
dates = ['23-03-08', '23-03-09'] # ['22-03-28', '22-03-30', '22-11-23']
substrate = '20um fibronectin discs'
thicknessType = 'bestH0' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['substrate'] == substrate),
           (data['date'].apply(lambda x : x in dates))]

co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

fig, ax11 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

renameAxes(ax11, renameDict1)
renameAxes(ax11, rD)

# Part 2.

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

data['fit_K'] = data['fit_K']/1e3

Filters = [(data['validatedThickness'] == True), 
           (data['fit_valid'] == True),
            (data['fit_K'] <= 10),
            (data['substrate'] == substrate),
            # (data['ctFieldMinThickness'] >= 200**2),
            (data['date'].apply(lambda x : x in dates))]

co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# fig, ax2 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=['fit_K'], Filters=Filters, 
#                 Boxplot=True, cellID='cellID', co_order=co_order, 
#                 AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
#                 figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

fig, ax2 = D1Plot_K(data, condCols=['drug', 'concentration'], Filters=Filters, AvgPerCell=True, 
            parm = 'fit_K', weightParm = 'fit_ciwK',
            co_order=co_order, box_pairs=[], stats=True, statMethod='Mann-Whitney', 
            Boxplot=True, stressBoxPlot = 1, styleDict = styleDict1,
            figSizeFactor = 1, markersizeFactor = 1, scale = 'lin',
            returnData = 0, returnCount = 0)




renameAxes(ax2, renameDict1)
renameAxes(ax2, rD)

# titles = ['Blebbistatin', 'LatrunculinA']
# tl = matplotlib.ticker.MultipleLocator(2)

# for k in range(2):
#     axes[0,k].set_ylim([0,900])
#     axes[1,k].set_ylim([0,12.5])
#     axes[k,1].set_ylabel('')
#     # axes[0,k].set_xticklabels([])
#     axes[0,k].set_title(titles[k])
#     axes[1,k].yaxis.set_major_locator(tl)



# ax[0].set_ylim([0, 1000])
# ax[0].legend(loc = 'upper right', fontsize = 8)
# ax[1].legend(loc = 'upper right', fontsize = 8)
# fig.suptitle('3T3aSFL & drugs\nPreliminary data')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_H&Echad_simple', figSubDir = figSubDir)
plt.show()



# %%%%% K(s) Y27


data = MecaData_Atcc
dates = ['23-03-08', '23-03-09']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
           (data['substrate'] == substrate),
           (data['date'].apply(lambda x : x in dates)),
          ]

out1 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = '300_500', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig1, ax1 = out1
ax1.set_ylim([0, 12])

out2 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = '300_700', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig2, ax2 = out2
ax2.set_ylim([0, 20])

out3 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)
fig3, ax3 = out3
ax3.set_ylim([0, 20])

# data_ff1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False)
# data_ff2 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = '150_550', scale = 'lin', printText = True)

# fig1.suptitle('3T3 vs. HoxB8 - stiffness')
# fig2.suptitle('3T3 vs. HoxB8 - stiffness')
plt.show()







# %%%%% Fluctuations

data = MecaData_Atcc
dates = ['23-03-08', '23-03-09']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
           (data['substrate'] == '20um fibronectin discs'),
           (data['ctFieldFluctuAmpli'] <= 500),
           (data['ctFieldThickness'] <= 480),
           (data['compNum'] >= 5),
           (data['date'].apply(lambda x : x in dates)),
          ]

co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

fig, ax = plt.subplots(1,1, figsize = (10,6))

fig, ax, df = D2Plot_wFit(data, fig = fig, ax = ax, 
                        XCol='ctFieldThickness', YCol='ctFieldFluctuAmpli', condCol=['drug', 'concentration'], 
                        Filters=Filters, cellID='cellID', co_order = co_order,
                        AvgPerCell=True, showManips = True,
                        modelFit=True, modelType='y=ax+b', writeEqn=True,
                        xscale = 'lin', yscale = 'lin', 
                        figSizeFactor = 1, markersizeFactor = 1.4,
                        returnData = True)



rD = {'fit_K':'Tangeantial Modulus (Pa)\n'+fitStr, 
      'ctFieldThickness':'Median thickness (nm)', 
      'ctFieldFluctuAmpli' : 'Thickness fluctuations amplitude (nm)'}

renameAxes(ax, rD, format_xticks=False)
# ax.set_xlim([0, 500])
# ax.set_ylim([0, 300])

fig.suptitle('Thickness - Fluctuation')

plt.show()



# %%%% 23-03-16 & 23-03-17

# %%%%% Compare the controls 23-03-16 & 23-03-17

fitType = 'stressGaussian'
fitId = '300_100'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)

rD = {'none & 0.0' : 'No drug',
      'blebbistatin & 10.0' : 'blebbistatin\n(10µM)', 
      'blebbistatin & 50.0' : 'blebbistatin\n(50µM)', 
      'Thickness at low force (nm)' : 'Thickness (nm)',
      'Tangeantial Modulus (Pa)' : 'Tangeantial Modulus (kPa)\n' + fitStr}

# Part 1.1

data_main = MecaData_Atcc
dates = ['23-03-16', '23-03-17'] # ['22-03-28', '22-03-30', '22-11-23']
substrate = '20um fibronectin discs'
thicknessType = 'surroundingThickness' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['substrate'] == substrate),
           (data['drug'] == 'dmso'),
           (data['date'].apply(lambda x : x in dates))]

co_order = ['23-03-16 & dmso', '23-03-17 & dmso']

fig, ax1 = D1Plot(data, condCols=['date', 'drug'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

renameAxes(ax1, renameDict1)
renameAxes(ax1, rD)

# Part 1.2

data_main = MecaData_Atcc
dates = ['23-03-16', '23-03-17'] # ['22-03-28', '22-03-30', '22-11-23']
substrate = '20um fibronectin discs'
thicknessType = 'bestH0' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['substrate'] == substrate),
           (data['drug'] == 'dmso'),
           (data['date'].apply(lambda x : x in dates))]

co_order = ['23-03-16 & dmso', '23-03-17 & dmso']

fig, ax11 = D1Plot(data, condCols=['date', 'drug'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

renameAxes(ax11, renameDict1)
renameAxes(ax11, rD)

# Part 2.

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

data['fit_K'] = data['fit_K']/1e3

Filters = [(data['validatedThickness'] == True), 
           (data['fit_valid'] == True),
            (data['fit_K'] <= 10),
            (data['substrate'] == substrate),
            (data['drug'] == 'dmso'),
            (data['date'].apply(lambda x : x in dates))]

co_order = ['23-03-16 & dmso', '23-03-17 & dmso']

fig, ax2 = D1Plot(data, condCols=['date', 'drug'], Parameters=['fit_K'], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)




renameAxes(ax2, renameDict1)
renameAxes(ax2, rD)


# Part 3.

data = MecaData_Atcc
dates = ['23-03-16', '23-03-17']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
           (data['substrate'] == substrate),
           (data['drug'] == 'dmso'),
           (data['date'].apply(lambda x : x in dates)),
          ]

# out1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
#                                 condCol = 'drug', mode = '200_400', scale = 'lin', printText = True,
#                                 returnData = 1, returnCount = 1)
# fig1, ax1, exportDf1, countDf1 = out1
# ax1.set_ylim([0, 10])

out31 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['date', 'drug'], mode = '250_600', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig31, ax31 = out31
ax31.set_ylim([0, 20])

out32 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['date', 'drug'], mode = '400_800', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig32, ax32 = out32
ax32.set_ylim([0, 25])

out33 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['date', 'drug'], mode = 'wholeCurve', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig33, ax33 = out33
ax33.set_ylim([0, 25])

# data_ff1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False)
# data_ff2 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = '150_550', scale = 'lin', printText = True)

# fig1.suptitle('3T3 vs. HoxB8 - stiffness')
# fig2.suptitle(


plt.show()


# %%%%% H and K Blebbi-2023

# fig, axes = plt.subplots(2,2, figsize = (9,7))

fitType = 'stressGaussian'
fitId = '400_100'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)

rD = {'none & 0.0' : 'No drug',
      'blebbistatin & 10.0' : 'blebbistatin\n(10µM)', 
      'blebbistatin & 50.0' : 'blebbistatin\n(50µM)', 
      'Thickness at low force (nm)' : 'Thickness (nm)',
      'Tangeantial Modulus (Pa)' : 'Tangeantial Modulus (kPa)\n' + fitStr}

# Part 1.1

data_main = MecaData_Atcc
dates = ['23-03-16', '23-03-17'] # ['22-03-28', '22-03-30', '22-11-23']
substrate = '20um fibronectin discs'
thicknessType = 'surroundingThickness' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['substrate'] == substrate),
           (data['date'].apply(lambda x : x in dates))]

co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0']

fig, ax1 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

renameAxes(ax1, renameDict1)
renameAxes(ax1, rD)

# Part 1.2

data_main = MecaData_Atcc
dates = ['23-03-16', '23-03-17'] # ['22-03-28', '22-03-30', '22-11-23']
substrate = '20um fibronectin discs'
thicknessType = 'bestH0' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['substrate'] == substrate),
           (data['date'].apply(lambda x : x in dates))]

co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0']

fig, ax11 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

renameAxes(ax11, renameDict1)
renameAxes(ax11, rD)

# Part 2.

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

data['fit_K'] = data['fit_K']/1e3

Filters = [(data['validatedThickness'] == True), 
           (data['fit_valid'] == True),
            (data['fit_K'] <= 10),
            (data['substrate'] == substrate),
            # (data['ctFieldMinThickness'] >= 200**2),
            (data['date'].apply(lambda x : x in dates))]

co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0']

D1Plot_K(data, condCols=['drug', 'concentration'], Filters=Filters, AvgPerCell=True, 
            parm = 'fit_K', weightParm = 'fit_ciwK',
            co_order=co_order, box_pairs=[], stats=True, statMethod='Mann-Whitney', 
            Boxplot=True, stressBoxPlot = 1, styleDict = styleDict1,
            figSizeFactor = 1, markersizeFactor = 1, scale = 'lin',
            returnData = 0, returnCount = 0)




renameAxes(ax2, renameDict1)
renameAxes(ax2, rD)

# titles = ['Blebbistatin', 'LatrunculinA']
# tl = matplotlib.ticker.MultipleLocator(2)

# for k in range(2):
#     axes[0,k].set_ylim([0,900])
#     axes[1,k].set_ylim([0,12.5])
#     axes[k,1].set_ylabel('')
#     # axes[0,k].set_xticklabels([])
#     axes[0,k].set_title(titles[k])
#     axes[1,k].yaxis.set_major_locator(tl)



# ax[0].set_ylim([0, 1000])
# ax[0].legend(loc = 'upper right', fontsize = 8)
# ax[1].legend(loc = 'upper right', fontsize = 8)
# fig.suptitle('3T3aSFL & drugs\nPreliminary data')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_H&Echad_simple', figSubDir = figSubDir)
plt.show()





# %%%%% K(s) Blebbi-2023

# fitType = 'stressGaussian'
# fitId = '300_100'
# c, hw = np.array(fitId.split('_')).astype(int)
# fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)

# rD = {'none & 0.0' : 'No drug',
#       'Y27 & 10.0' : 'Y27\n(10µM)', 
#       'Y27 & 50.0' : 'Y27\n(50µM)', 
#       'Thickness at low force (nm)' : 'Thickness (nm)',
#       'Tangeantial Modulus (Pa)' : 'Tangeantial Modulus (kPa)\n' + fitStr}

# # Part 1.1

# data_main = MecaData_Atcc
# dates = ['23-03-08', '23-03-09'] # ['22-03-28', '22-03-30', '22-11-23']
# substrate = '20um fibronectin discs'
# thicknessType = 'surroundingThickness' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = MecaData_Atcc
dates = ['23-03-16', '23-03-17']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
           (data['substrate'] == substrate),
           (data['date'].apply(lambda x : x in dates)),
          ]

# out1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
#                                 condCol = 'drug', mode = '200_400', scale = 'lin', printText = True,
#                                 returnData = 1, returnCount = 1)
# fig1, ax1, exportDf1, countDf1 = out1
# ax1.set_ylim([0, 10])

out1 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = '400_600', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig1, ax1 = out1
ax1.set_ylim([0, 20])

out2 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = '700_1000', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig2, ax2 = out2
ax2.set_ylim([0, 25])

out3 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = 'wholeCurve', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig3, ax3 = out3
ax3.set_ylim([0, 25])

# data_ff1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False)
# data_ff2 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = '150_550', scale = 'lin', printText = True)

# fig1.suptitle('3T3 vs. HoxB8 - stiffness')
# fig2.suptitle('3T3 vs. HoxB8 - stiffness')
plt.show()


# %%%%% Fluctuations

data = MecaData_Atcc
dates = ['23-03-16', '23-03-17']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
           (data['substrate'] == '20um fibronectin discs'),
           (data['compNum'] >= 4),
           (data['date'].apply(lambda x : x in dates)),
          ]

fig, ax = plt.subplots(1,1, figsize = (10,6))

co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0']

fig, ax, df = D2Plot_wFit(data, fig = fig, ax = ax, 
                        XCol='ctFieldThickness', YCol='ctFieldFluctuAmpli', condCol=['drug', 'concentration'], 
                        Filters=Filters, cellID='cellID', co_order = co_order,
                        AvgPerCell=True, showManips = True,
                        modelFit=True, modelType='y=ax+b', writeEqn=True,
                        xscale = 'lin', yscale = 'lin', 
                        figSizeFactor = 1, markersizeFactor = 1.4,
                        returnData = True)



rD = {'fit_K':'Tangeantial Modulus (Pa)\n'+fitStr, 
      'ctFieldThickness':'Median thickness (nm)', 
      'ctFieldFluctuAmpli' : 'Thickness fluctuations amplitude (nm)'}

renameAxes(ax, rD, format_xticks=False)
ax.set_xlim([0, 500])
ax.set_ylim([0, 300])

fig.suptitle('Thickness - Fluctuation')

plt.show()


# %%%%% Thickness - LowStress stiffness

data_main = MecaData_Atcc
fitType = 'stressGaussian'
fitId = '200_75'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit on [{:.0f}, {:.0f}] Pa'.format(c-hw, c+hw)
data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

dates = ['23-03-16', '23-03-17']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
           (data['substrate'] == '20um fibronectin discs'),
           (data['date'].apply(lambda x : x in dates)),
          ]

fig, ax = plt.subplots(1,1, figsize = (10,6))

fig, ax, df = D2Plot_wFit(data, fig = fig, ax = ax, 
                        XCol='ctFieldThickness', YCol='fit_K', condCol=['drug', 'concentration'], 
                        Filters=Filters, cellID='cellID', co_order = [],
                        AvgPerCell=True, showManips = True,
                        modelFit=True, modelType='y=k*x^a', writeEqn=True,
                        xscale = 'log', yscale = 'log', 
                        figSizeFactor = 1, markersizeFactor = 1.4,
                        returnData = True)


rD = {'fit_K':'Tangeantial Modulus (Pa)\n'+fitStr, 
      'ctFieldThickness':'Median thickness (nm)', 
      'ctFieldFluctuAmpli' : 'Thickness fluctuations amplitude (nm)'}

renameAxes(ax, rD)
ax.set_xlim([50, 1200])
ax.set_ylim([0, 30000])

fig.suptitle('Thickness - Stiffness relation at low stress')

plt.show()


# %%%% Y27 + Blebbi

# %%%%% Test new plots

# fig, axes = plt.subplots(2,2, figsize = (9,7))

fitType = 'stressGaussian'
fitId = '400_100'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)

rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27\n(10µM)', 
      'Y27 & 50.0' : 'Y27\n(50µM)', 
      'dmso & 0.0' : 'DMSO', 
      'blebbistatin & 10.0' : 'Blebbi\n(10µM)',  
      'blebbistatin & 50.0' : 'Blebbi\n(50µM)', 
      'Thickness at low force (nm)' : 'Thickness (nm)',
      'Tangeantial Modulus (Pa)' : 'Tangeantial Modulus (kPa)\n' + fitStr}

# Part 1.

data_main = MecaData_Atcc
dates = ['23-03-08', '23-03-09', '23-03-16', '23-03-17'] # ['22-03-28', '22-03-30', '22-11-23']
substrate = '20um fibronectin discs'
thicknessType = 'surroundingThickness' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['substrate'] == substrate),
           (data['date'].apply(lambda x : x in dates))]

co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0', 'dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0']
box_pairs = [['none & 0.0', 'dmso & 0.0'],
              ['none & 0.0', 'Y27 & 10.0'],['none & 0.0', 'Y27 & 50.0'], ['Y27 & 10.0', 'Y27 & 50.0'], 
              ['dmso & 0.0', 'blebbistatin & 10.0'],['dmso & 0.0', 'blebbistatin & 50.0'], ['blebbistatin & 10.0', 'blebbistatin & 50.0']]

out1 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=1, statMethod='Mann-Whitney', box_pairs=box_pairs, 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1,
                returnData = 1, returnCount = 1)

fig1, ax1, dfExp1, dfCount1 = out1

renameAxes(ax1, renameDict1)
renameAxes(ax1, rD)


out2 = D1Plot_vars(data, condCols=['drug', 'concentration'], Parameters=[thicknessType], Filters=Filters, 
                    cellID='cellID', co_order=co_order, 
                    stats = 1, statMethod='Mann-Whitney', box_pairs=box_pairs, 
                    figSizeFactor = 1.4, markersizeFactor=1.2, stressBoxPlot=1)

fig2, ax2 = out2[:2]

renameAxes(ax2, renameDict1)
renameAxes(ax2, rD)



# Part 2.

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)
data['fit_K'] = data['fit_K']/1e3
data['E_Chadwick'] = data['E_Chadwick']/1e3

Filters = [(data['validatedThickness'] == True), 
            (data['valid_Chadwick'] == True),
             (data['E_Chadwick'] <= 10),
           # (data['fit_valid'] == True),
           #  (data['fit_K'] <= 10),
            (data['substrate'] == substrate),
            (data['date'].apply(lambda x : x in dates))]

co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0', 'dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0']
box_pairs = [['none & 0.0', 'dmso & 0.0'],
              ['none & 0.0', 'Y27 & 10.0'],['none & 0.0', 'Y27 & 50.0'], ['Y27 & 10.0', 'Y27 & 50.0'], 
              ['dmso & 0.0', 'blebbistatin & 10.0'],['dmso & 0.0', 'blebbistatin & 50.0'], ['blebbistatin & 10.0', 'blebbistatin & 50.0']]

out1 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=['E_Chadwick'], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=1, statMethod='Mann-Whitney', box_pairs=box_pairs, 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1,
                returnData = 1, returnCount = 1)

fig1, ax1, dfExp1, dfCount1 = out1

renameAxes(ax1, renameDict1)
renameAxes(ax1, rD)


out2 = D1Plot_vars(data, condCols=['drug', 'concentration'], Parameters=['E_Chadwick'], Filters=Filters, 
                    cellID='cellID', co_order=co_order, 
                    stats = 1, statMethod='Mann-Whitney', box_pairs=box_pairs, 
                    figSizeFactor = 1.4, markersizeFactor=1.2, stressBoxPlot=1)

fig2, ax2 = out2[:2]

renameAxes(ax2, renameDict1)
renameAxes(ax2, rD)

plt.show()


# %%%%% H and K Y27 + Blebbi

# fig, axes = plt.subplots(2,2, figsize = (9,7))

fitType = 'stressGaussian'
fitId = '300_100'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)

rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27\n(10µM)', 
      'Y27 & 50.0' : 'Y27\n(50µM)', 
      'dmso & 0.0' : 'DMSO', 
      'blebbistatin & 10.0' : 'Blebbi\n(10µM)',  
      'blebbistatin & 50.0' : 'Blebbi\n(50µM)', 
      'Thickness at low force (nm)' : 'Thickness (nm)',
      'Tangeantial Modulus (Pa)' : 'Tangeantial Modulus (kPa)\n' + fitStr}

# Part 1.1

# Part 1.1

data_main = MecaData_Atcc
dates = ['23-03-08', '23-03-09', '23-03-16', '23-03-17'] # ['22-03-28', '22-03-30', '22-11-23']
substrate = '20um fibronectin discs'
thicknessType = 'surroundingThickness' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['substrate'] == substrate),
           (data['date'].apply(lambda x : x in dates))]

co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0', 'dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0']

fig, ax1 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=False, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

renameAxes(ax1, renameDict1)
renameAxes(ax1, rD)

# Part 1.2

data_main = MecaData_Atcc
dates = ['23-03-08', '23-03-09', '23-03-16', '23-03-17'] # ['22-03-28', '22-03-30', '22-11-23']
substrate = '20um fibronectin discs'
thicknessType = 'bestH0' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data = data_main

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['substrate'] == substrate),
           (data['date'].apply(lambda x : x in dates))]

co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0', 'dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0']
# box_pairs = [['none & 0.0', 'dmso & 0.0'],
#              ['none & 0.0', 'Y27 & 10.0'],['none & 0.0', 'Y27 & 50.0'], ['Y27 & 10.0', 'Y27 & 50.0'], 
#              ['dmso & 0.0', 'blebbistatin & 10.0'],['dmso & 0.0', 'blebbistatin & 50.0'], ['blebbistatin & 10.0', 'blebbistatin & 50.0']]
box_pairs = []

fig, ax11 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=False, statMethod='Mann-Whitney', box_pairs=box_pairs, 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

renameAxes(ax11, renameDict1)
renameAxes(ax11, rD)

# Part 2.

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

data['fit_K'] = data['fit_K']/1e3

Filters = [(data['validatedThickness'] == True), 
           (data['fit_valid'] == True),
            (data['fit_K'] <= 10),
            (data['substrate'] == substrate),
            # (data['ctFieldMinThickness'] >= 200**2),
            (data['date'].apply(lambda x : x in dates))]

co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0', 'dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0']

fig, ax2 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=['fit_K'], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=False, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)




renameAxes(ax2, renameDict1)
renameAxes(ax2, rD)

# titles = ['Blebbistatin', 'LatrunculinA']
# tl = matplotlib.ticker.MultipleLocator(2)

# for k in range(2):
#     axes[0,k].set_ylim([0,900])
#     axes[1,k].set_ylim([0,12.5])
#     axes[k,1].set_ylabel('')
#     # axes[0,k].set_xticklabels([])
#     axes[0,k].set_title(titles[k])
#     axes[1,k].yaxis.set_major_locator(tl)



# ax[0].set_ylim([0, 1000])
# ax[0].legend(loc = 'upper right', fontsize = 8)
# ax[1].legend(loc = 'upper right', fontsize = 8)
# fig.suptitle('3T3aSFL & drugs\nPreliminary data')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_H&Echad_simple', figSubDir = figSubDir)
plt.show()



# %%%%% K(s) Y27 + Blebbi


data = MecaData_Atcc
dates = ['23-03-08', '23-03-09', '23-03-16', '23-03-17']
substrate = '20um fibronectin discs'

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
           (data['substrate'] == substrate),
           (data['date'].apply(lambda x : x in dates)),
          ]

out1 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = '300_500', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)
fig1, ax1 = out1
ax1.set_ylim([0, 12])

out2 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = '500_700', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)
fig2, ax2 = out2
ax2.set_ylim([0, 20])

out3 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)
fig3, ax3 = out3
ax3.set_ylim([0, 20])

# data_ff1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False)
# data_ff2 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = '150_550', scale = 'lin', printText = True)

# fig1.suptitle('3T3 vs. HoxB8 - stiffness')
# fig2.suptitle('3T3 vs. HoxB8 - stiffness')
plt.show()


# %%%%% All K(s) For Anumita


data = MecaData_Atcc
substrate = '20um fibronectin discs'

rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27 10µM', 
      'Y27 & 50.0' : 'Y27 50µM', 
      'dmso & 0.0' : 'DMSO', 
      'blebbistatin & 10.0' : 'Blebbi 10µM',  
      'blebbistatin & 50.0' : 'Blebbi 50µM', 
      'Thickness at low force (nm)' : 'Thickness (nm)',
      'Tangeantial Modulus (Pa)' : 'Tangeantial Modulus (kPa)'}

# Only control

dates = ['23-03-08', '23-03-09']
Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
           (data['substrate'] == substrate),
           (data['date'].apply(lambda x : x in dates)),
           (data['drug'] == 'none'),
          ]

out1 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = '150_450', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)
fig1, ax1 = out1
ax1.set_ylim([0, 18])

# Control vs Y27
dates = ['23-03-08', '23-03-09']
Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
           (data['substrate'] == substrate),
           (data['date'].apply(lambda x : x in dates)),
          ]

out2 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = '300_500', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)
fig2, ax2 = out2
ax2.set_ylim([0, 10])

# Control vs Blebbi
dates = ['23-03-16', '23-03-17']
Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
           (data['substrate'] == substrate),
           (data['date'].apply(lambda x : x in dates)),
          ]

out3 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = '300_500', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)
fig3, ax3 = out3
ax3.set_ylim([0, 20])

# Y27 & Blebbi
dates = ['23-03-08', '23-03-09', '23-03-16', '23-03-17']

Filters = [(data['validatedThickness'] == True),
           (data['UI_Valid'] == True),
           (data['substrate'] == substrate),
           (data['date'].apply(lambda x : x in dates)),
          ]

out4 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCols = ['drug', 'concentration'], mode = '300_500', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)
fig4, ax4 = out4
ax4.set_ylim([0, 12])


for ax in [ax1, ax2, ax3, ax4]:
    ax.set_title('3T3 ATCC (no opto construct)')
    ax.legend(fontsize = 8)
    renameAxes(ax, renameDict1, format_xticks=False)
    renameAxes(ax, rD, format_xticks=False)
    renameLegend(ax, rD)


plt.show()


# %%% Non-linearity

gs.set_bigText_options_jv()

data = MecaData_All

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['drug'].apply(lambda x : x in ['none'])),
           (data['substrate'].apply(lambda x : x in ['20um fibronectin discs'])),
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11'])), # , 'aSFL-LG+++'
           (data['date'].apply(lambda x : x.startswith('22')))]

# out1 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
#                                 condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False,
#                                 Sinf = 0, Ssup = 1000,
#                                 returnData = 1, returnCount = 1)
# fig1, ax1 = out1
# ax1.set_ylim([0, 14])

out2 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'cell type', mode = '150_600', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)
fig2, ax2 = out2
ax2.set_ylim([0, 16])

renameAxes(ax2, {'K (kPa)' : 'Tangeantial Modulus (kPa)'})

# data_ff1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False)
# data_ff2 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = '150_550', scale = 'lin', printText = True)

# fig1.suptitle('3T3 vs. HoxB8 - stiffness')
# fig2.suptitle('3T3 vs. HoxB8 - stiffness')
plt.show()



# %%%  Thickness analysis

# %%%%  Thickness stiffness curves
gs.set_bigText_options_jv()

# %%%%%  Low stress

data_main = MecaData_All
fitType = 'stressGaussian'
fitId = '200_75'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit on [{:.0f}, {:.0f}] Pa'.format(c-hw, c+hw)
data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)


Filters = [(data['validatedThickness'] == True),
           (data['cell type'].apply(lambda x : x in ['3T3'])),
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11', 'aSFL-LG+++'])),
           (data['normal field'] == 5),
           (data['ctFieldThickness'] <= 1200),
           (data['drug'] == 'none'),
           (data['fit_error'] == False), 
           (data['fit_valid'] == True),
           (data['substrate'] == '20um fibronectin discs')]

fig, ax = plt.subplots(1,1, figsize = (10,6))

fig, ax, df = D2Plot_wFit(data, fig = fig, ax = ax, 
                        XCol='ctFieldThickness', YCol='fit_K', condCol=['cell type'], 
                        Filters=Filters, cellID='cellID', co_order = [],
                        AvgPerCell=True, showManips = True,
                        modelFit=True, modelType='y=k*x^a', writeEqn=True,
                        xscale = 'log', yscale = 'log', 
                        figSizeFactor = 1, markersizeFactor = 1.4,
                        returnData = True)

rD = {'fit_K':'Tangeantial Modulus (Pa)\n'+fitStr, 
      'ctFieldThickness':'Median thickness (nm)', 
      'ctFieldFluctuAmpli' : 'Thickness fluctuations amplitude (nm)'}

renameAxes(ax, rD)
ax.set_xlim([50, 1200])
ax.set_ylim([0, 30000])

fig.suptitle('Thickness - Stiffness relation at low stress')

plt.show()

# %%%%  Medium stress

data_main = MecaData_All
fitType = 'stressGaussian'
fitId = '400_75'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit on [{:.0f}, {:.0f}] Pa'.format(c-hw, c+hw)

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

Filters = [(data['validatedThickness'] == True),
           (data['cell type'].apply(lambda x : x in ['3T3'])),
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11', 'aSFL-LG+++'])),
           (data['normal field'] == 5),
           (data['ctFieldThickness'] <= 1200),
           (data['drug'] == 'none'),
           (data['fit_error'] == False), 
           (data['fit_valid'] == True),
           (data['substrate'] == '20um fibronectin discs')]

fig, ax = plt.subplots(1,1, figsize = (10,6))

fig, ax, df = D2Plot_wFit(data, fig = fig, ax = ax, 
                        XCol='ctFieldThickness', YCol='fit_K', condCol=['cell type'], 
                        Filters=Filters, cellID='cellID', co_order = [],
                        AvgPerCell=True, showManips = True,
                        modelFit=True, modelType='y=k*x^a', writeEqn=True,
                        xscale = 'log', yscale = 'log', 
                        figSizeFactor = 1, markersizeFactor = 1.4,
                        returnData = True)

rD = {'fit_K':'Tangeantial Modulus (Pa)\n'+fitStr, 
      'ctFieldThickness':'Median thickness (nm)', 
      'ctFieldFluctuAmpli' : 'Thickness fluctuations amplitude (nm)'}

renameAxes(ax, rD)
ax.set_xlim([50, 1200])
ax.set_ylim([0, 30000])

fig.suptitle('Thickness - Stiffness relation at medium stress')

plt.show()

# %%%%  Thickness stiffness curves

data_main = MecaData_All
fitType = 'stressGaussian'
fitId = '500_75'

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

Filters = [(data['validatedThickness'] == True),
           (data['cell type'].apply(lambda x : x in ['3T3'])),
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11', 'aSFL-LG+++'])),
           (data['ctFieldThickness'] <= 1200),
           (data['drug'] == 'none'),
           (data['fit_error'] == False), 
           (data['fit_valid'] == True),
           (data['substrate'] == '20um fibronectin discs')]

fig, ax, df = D2Plot_wFit(data, fig = None, ax = None, 
                        XCol='bestH0', YCol='fit_K', condCol=['cell type'], 
                        Filters=Filters, cellID='cellID', co_order = [],
                        AvgPerCell=False, showManips = True,
                        modelFit=True, modelType='y=k*x^a', writeEqn=True,
                        xscale = 'log', yscale = 'log', 
                        figSizeFactor = 1, markersizeFactor = 1,
                        returnData = True)

rD = {'ctFieldThickness':'Median thickness (nm)', 'ctFieldFluctuAmpli' : 'Thickness fluctuations amplitude (nm)'}
renameAxes(ax, rD)
# ax.set_xlim([0, 1200])
# ax.set_ylim([0, 1000])

fig.suptitle('Compression Experiments')

med, ampli = np.median(df['ctFieldThickness'].values), np.median(df['ctFieldFluctuAmpli'].values)

plt.show()


# %%%%  Median-Fluctu on compression expts
# 'aSFL-A11', 

data = MecaData_All

Filters = [(data['validatedThickness'] == True),
           (data['cell type'].apply(lambda x : x in ['3T3'])),
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11', 'aSFL-LG+++'])),
           (data['compNum'] >= 6),
           (data['ctFieldThickness'] <= 1200),
           (data['date'].apply(lambda x : x.startswith('22'))),
           (data['drug'] == 'none'),
           (data['substrate'] == '20um fibronectin discs')]

fig, ax, df = D2Plot_wFit(data, fig = None, ax = None, 
                        XCol='ctFieldThickness', YCol='ctFieldFluctuAmpli', condCol=['cell type'], 
                        Filters=Filters, cellID='cellID', co_order = [],
                        AvgPerCell=True, showManips = True,
                        modelFit=True, modelType='y=ax+b', writeEqn=True,
                        xscale = 'lin', yscale = 'lin', 
                        figSizeFactor = 1, markersizeFactor = 1,
                        returnData = True)

rD = {'ctFieldThickness':'Median thickness (nm)', 'ctFieldFluctuAmpli' : 'Thickness fluctuations amplitude (nm)'}
renameAxes(ax, rD)
ax.set_xlim([0, 1200])
ax.set_ylim([0, 1000])

fig.suptitle('Compression Experiments')

med, ampli = np.median(df['ctFieldThickness'].values), np.median(df['ctFieldFluctuAmpli'].values)
print(med, ampli)

plt.show()

# %%%%  Median-Fluctu on ct field expts

data = CtFieldData_All_JV

Filters = [(data['cell type'].apply(lambda x : x in ['3T3'])),
           (data['cell subtype'].apply(lambda x : x in ['aSFL-A11', 'aSFL-LG+++'])),
           (data['duration'] >= 300),
            (data['date'].apply(lambda x : x in ['21-02-10', '21-04-21', '21-04-23'])),
            (data['drug'] == 'none'),
           (data['substrate'] == '20um fibronectin discs')]

fig, ax, df = D2Plot_wFit(data, fig = None, ax = None, 
                        XCol='medianThickness', YCol='fluctuAmpli', condCol=['cell type'], 
                        Filters=Filters, cellID='cellID', co_order = [],
                        AvgPerCell=True, showManips = True,
                        modelFit=True, modelType='y=ax+b', writeEqn=True,
                        xscale = 'lin', yscale = 'lin', 
                        figSizeFactor = 1, markersizeFactor = 1,
                        returnData = True)

rD = {'medianThickness':'Median thickness (nm)', 'fluctuAmpli' : 'Thickness fluctuations amplitude (nm)'}
renameAxes(ax, rD)
ax.set_xlim([0, 1200])
ax.set_ylim([0, 1000])

fig.suptitle('Constant Field Experiments')

med, ampli = np.median(df['medianThickness'].values), np.median(df['fluctuAmpli'].values)
print(med, ampli)

plt.show()

# %%%  Region fits, non linearity


# %%%% On Jan 2021 data

# %%%%%


data = MecaData_All

Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True),
           (data['drug'] == 'none'),
           (data['substrate'] == '20um fibronectin discs'),
           (data['date'].apply(lambda x : x in ['21-01-18', '21-01-21']))]

data_f = data
for fltr in Filters:
    data_f = data_f.loc[fltr]
    
print(data_f[data_f['validatedFit_s<500Pa']].shape)
print(data_f[data_f['validatedFit_500<s<1000Pa']].shape)
print(data_f[data_f['validatedFit_s<500Pa'] & data_f['validatedFit_500<s<1000Pa']].shape)


# %%%%%


data = MecaData_All
dates = ['21-01-18', '21-01-21']
fits = ['s<500Pa', '250<s<750Pa', '500<s<1000Pa']

Filters = [(data['validatedFit'] == True),
           (data['validatedThickness'] == True),
           (data['drug'] == 'none'),
           (data['substrate'] == '20um fibronectin discs'),
           (data['date'].apply(lambda x : x in dates))]
Filters += [(data['validatedFit_' + f] == True) for f in fits]

Parameters = ['EChadwick_' + f for f in fits]

fig, ax = D1PlotPaired(data, Parameters=Parameters, Filters=Filters, Boxplot=True, cellID='cellID', 
                   co_order=[], stats=True, statMethod='Wilcox_less', box_pairs=[],
                   figSizeFactor = 1.5, markersizeFactor=1, orientation = 'h', labels = fits)
ax.set_ylabel('E_Chadwick (Pa)')
fig.suptitle('Jan 2021 data ; B = 3 -> 40 mT ; no drug')
plt.show()


# %%%%%


data = MecaData_All
dates = ['21-01-18', '21-01-21']
fits = ['s<500Pa', '250<s<750Pa', '500<s<1000Pa']

Filters = [(data['validatedFit'] == True),
           (data['validatedThickness'] == True),
           (data['drug'] == 'doxycyclin'),
           (data['substrate'] == '20um fibronectin discs'),
           (data['date'].apply(lambda x : x in dates))]
Filters += [(data['validatedFit_' + f] == True) for f in fits]

Parameters = ['EChadwick_' + f for f in fits]

fig, ax = D1PlotPaired(data, Parameters=Parameters, Filters=Filters, Boxplot=True, cellID='cellID', 
                   co_order=[], stats=True, statMethod='Wilcox_less', box_pairs=[],
                   figSizeFactor = 1.5, markersizeFactor=1, orientation = 'h', labels = fits)
ax.set_ylabel('E_Chadwick (Pa)')
fig.suptitle('Jan 2021 data ; B = 3 -> 40 mT ; doxy')
plt.show()


# %%%%%


data = MecaData_All
dates = ['21-01-18', '21-01-21']
fits = ['s<400Pa', '300<s<700Pa', '600<s<1000Pa']

Filters = [(data['validatedFit'] == True),
           (data['validatedThickness'] == True),
           (data['drug'] == 'none'),
           (data['substrate'] == '20um fibronectin discs'),
           (data['date'].apply(lambda x : x in dates))]
Filters += [(data['validatedFit_' + f] == True) for f in fits]

Parameters = ['EChadwick_' + f for f in fits]

fig, ax = D1PlotPaired(data, Parameters=Parameters, Filters=Filters, Boxplot=True, cellID='cellID', 
                   co_order=[], stats=True, statMethod='Wilcox_less', box_pairs=[],
                   figSizeFactor = 1.5, markersizeFactor=1, orientation = 'h', labels = fits)
ax.set_ylabel('E_Chadwick (Pa)')
fig.suptitle('Jan 2021 data ; B = 3 -> 40 mT ; no drug')
plt.show()


# %%%%%


data = MecaData_All
dates = ['21-12-08', '22-01-12']
fits = ['s<100Pa', '100<s<200Pa', '200<s<300Pa']

Filters = [(data['validatedFit'] == True),
           (data['validatedThickness'] == True),
           (data['drug'] == 'none'),
           (data['substrate'] == '20um fibronectin discs'),
           (data['date'].apply(lambda x : x in dates))]
Filters += [(data['validatedFit_' + f] == True) for f in fits]

Parameters = ['EChadwick_' + f for f in fits]

fig, ax = D1PlotPaired(data, Parameters=Parameters, Filters=Filters, Boxplot=True, cellID='cellID', 
                   co_order=[], stats=True, statMethod='Wilcox_less', box_pairs=[],
                   figSizeFactor = 1.5, markersizeFactor=1, orientation = 'h', labels = fits)
ax.set_ylabel('E_Chadwick (Pa)')
fig.suptitle('Jan 2022 data ; B = 1 -> 13 mT ; no drug')
plt.show()


# %%%%%


# ['21-12-08', '21-12-16', '22-01-12']
data = MecaData_All

dates = ['22-02-09'] #['21-12-08', '22-01-12'] ['21-01-18', '21-01-21', '21-12-08']

filterList = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True),\
           (data['cell subtype'] == 'aSFL'), 
           (data['bead type'] == 'M450'),
           (data['substrate'] == '20um fibronectin discs'),
           (data['date'].apply(lambda x : x in dates))]


fig, ax = plt.subplots(1, 1, figsize = (9, 6), tight_layout=True)

globalFilter = filterList[0]
for k in range(1, len(filterList)):
    globalFilter = globalFilter & filterList[k]

X = data[globalFilter]['surroundingThickness'].values
Smin = data[globalFilter]['minStress'].values
Smax = data[globalFilter]['maxStress'].values

for i in range(len(X)):
    ax.plot([X[i], X[i]], [Smin[i], Smax[i]], 
            ls = '-', color = 'deepskyblue', alpha = 0.3,
            label = 'Stress range', zorder = 1)
    ax.plot([X[i]], [Smin[i]], 
            marker = 'o', markerfacecolor = 'skyblue', markeredgecolor = 'k', markersize = 4,
            ls = '')
    ax.plot([X[i]], [Smax[i]], 
            marker = 'o', markerfacecolor = 'royalblue', markeredgecolor = 'k', markersize = 4,
            ls = '')

ax.set_xlabel('Cortical thickness (nm)')
ax.set_xlim([0, 1200])
locator = matplotlib.ticker.MultipleLocator(100)
ax.xaxis.set_major_locator(locator)

ax.set_ylabel('Extremal stress values (Pa)')
ax.set_ylim([0, 2500])
locator = matplotlib.ticker.MultipleLocator(100)
ax.yaxis.set_major_locator(locator)
ax.yaxis.grid(True)

ax.set_title('Jan 21 & jan 22 - Extremal stress values vs. thickness, for each compression')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_StressRanges', figSubDir = figSubDir)
plt.show()


# %%%%%


data = MecaData_All
dates = ['21-01-18', '21-01-21']
fit = 's<200Pa_included' # 's<400Pa', '300<s<700Pa', '600<s<1000Pa'

Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True),
           (data['validatedFit_'+fit] == True),
           (data['substrate'] == '20um fibronectin discs'),
           (data['drug'] == 'none'),
           (data['date'].apply(lambda x : x in dates))]

fig, ax = D2Plot_wFit(data, XCol='surroundingThickness',YCol='EChadwick_' + fit,condCol = ['bead type', 'date'],           Filters=Filters, cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')

ax.set_ylabel('EChadwick ['+ fit +']  (Pa)')
ax.set_xlabel('Thickness at low force (nm)')
fig.suptitle('3T3aSFL: E(h) // dates = ' + dates[0][:5] + ' // ' + fit + ' // All comp')
ax.legend(loc = 'upper right', fontsize = 8)
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_E(h)', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%%


data = MecaData_All
dates = ['21-01-18', '21-01-21']
fit = '100<s<300Pa_200included' # 's<400Pa', '300<s<700Pa', '600<s<1000Pa'

Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True),
           (data['bead type'] == 'M450'),
           (data['validatedFit_'+fit] == True),
           (data['Npts_'+fit] >= 15),
           (data['surroundingThickness'] <= 800),
           (data['substrate'] == '20um fibronectin discs'),
           (data['drug'] == 'none'),
           (data['date'].apply(lambda x : x in dates))]

fig, ax = D2Plot_wFit(data, XCol='surroundingThickness',YCol='EChadwick_' + fit,condCol = ['bead type', 'date'],           Filters=Filters, cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')

ax.set_ylabel('EChadwick ['+ fit +']  (Pa)', fontsize = 12)
ax.set_xlabel('Thickness at low force (nm)', fontsize = 12)
fig.suptitle('3T3aSFL: E(h) // dates = ' + dates[0][:5] + ' // ' + fit + ' // All comp', fontsize = 14)
ax.legend(loc = 'upper right', fontsize = 8)
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_E(h)', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%%


data = MecaData_All
dates = ['21-01-18', '21-01-21']
fit = '50<s<250Pa_150included' # 's<400Pa', '300<s<700Pa', '600<s<1000Pa'

Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True),
           (data['bead type'] == 'M450'),
           (data['validatedFit_'+fit] == True),
           (data['Npts_'+fit] >= 15),
           (data['surroundingThickness'] <= 800),
           (data['substrate'] == '20um fibronectin discs'),
           (data['drug'] == 'none'),
           (data['date'].apply(lambda x : x in dates))]

fig, ax = D2Plot_wFit(data, XCol='surroundingThickness',YCol='EChadwick_' + fit,condCol = ['bead type', 'date'],           Filters=Filters, cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')

ax.set_ylabel('EChadwick ['+ fit +']  (Pa)', fontsize = 12)
ax.set_xlabel('Thickness at low force (nm)', fontsize = 12)
fig.suptitle('3T3aSFL: E(h) // dates = ' + dates[0][:5] + ' // ' + fit + ' // All comp', fontsize = 14)
ax.legend(loc = 'upper right', fontsize = 8)
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_E(h)', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%%


data = MecaData_All
dates = ['22-01-12']
fit = '50<s<250Pa_150included' # 's<400Pa', '300<s<700Pa', '600<s<1000Pa'

Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True),
           (data['bead type'] == 'M450'),
           (data['validatedFit_'+fit] == True),
           (data['Npts_'+fit] >= 15),
           (data['surroundingThickness'] <= 800),
           (data['substrate'] == '20um fibronectin discs'),
           (data['drug'] == 'none'),
           (data['date'].apply(lambda x : x in dates))]

fig, ax = D2Plot_wFit(data, XCol='surroundingThickness',YCol='EChadwick_' + fit,condCol = ['bead type'],           Filters=Filters, cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')

ax.set_ylabel('EChadwick ['+ fit +']  (Pa)', fontsize = 12)
ax.set_xlabel('Thickness at low force (nm)', fontsize = 12)
fig.suptitle('3T3aSFL: E(h) // dates = ' + dates[0][:5] + ' // ' + fit + ' // All comp', fontsize = 14)
ax.legend(loc = 'upper right', fontsize = 8)
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_E(h)', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%%


data = MecaData_All
dates = ['21-01-18', '21-01-21', '22-01-12']
fit = '50<s<250Pa_150included' # 's<400Pa', '300<s<700Pa', '600<s<1000Pa'

Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True),
           (data['bead type'] == 'M450'),
           (data['validatedFit_'+fit] == True),
           (data['Npts_'+fit] >= 15),
           (data['surroundingThickness'] <= 800),
           (data['substrate'] == '20um fibronectin discs'),
           (data['drug'] == 'none'),
           (data['date'].apply(lambda x : x in dates))]

fig, ax = D2Plot_wFit(data, XCol='surroundingThickness',YCol='EChadwick_' + fit,condCol = ['bead type', 'date'],           Filters=Filters, cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')

ax.set_ylabel('EChadwick ['+ fit +']  (Pa)', fontsize = 12)
ax.set_xlabel('Thickness at low force (nm)', fontsize = 12)
fig.suptitle('3T3aSFL: E(h) // dates = ' + 'all' + ' // ' + fit + ' // All comp', fontsize = 14)
ax.legend(loc = 'upper right', fontsize = 8)
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_E(h)', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%%


data = MecaData_All
dates = ['21-01-18', '21-01-21']
fit = 's<300Pa_included' # 's<400Pa', '300<s<700Pa', '600<s<1000Pa'

Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True),
           (data['validatedFit_'+fit] == True),
           (data['substrate'] == '20um fibronectin discs'),
           (data['drug'] == 'none'),
           (data['date'].apply(lambda x : x in dates))]

fig, ax = D2Plot_wFit(data, XCol='surroundingThickness',YCol='EChadwick_' + fit,condCol = ['bead type', 'date'],           Filters=Filters, cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')

ax.set_ylabel('EChadwick ['+ fit +']  (Pa)')
ax.set_xlabel('Thickness at low force (nm)')
fig.suptitle('3T3aSFL: E(h) // dates = ' + dates[0][:5] + ' // ' + fit + ' // All comp')
ax.legend(loc = 'upper right', fontsize = 8)
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_E(h)', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%%


data = MecaData_All
dates = ['21-01-18', '21-01-21']
fit = '200<s<500Pa_350included' # 's<400Pa', '300<s<700Pa', '600<s<1000Pa'

Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True),
           (data['validatedFit_'+fit] == True),
           (data['substrate'] == '20um fibronectin discs'),
           (data['drug'] == 'none'),
           (data['date'].apply(lambda x : x in dates))]

fig, ax = D2Plot_wFit(data, XCol='surroundingThickness',YCol='EChadwick_' + fit,condCol = ['bead type', 'date'],           Filters=Filters, cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')

ax.set_ylabel('EChadwick ['+ fit +']  (Pa)')
ax.set_xlabel('Thickness at low force (nm)')
fig.suptitle('3T3aSFL: E(h) // dates = ' + dates[0][:5] + ' // ' + fit + ' // All comp')
ax.legend(loc = 'upper right', fontsize = 8)
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_E(h)', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%%


data = MecaData_All
dates = ['21-01-18', '21-01-21']
fit = 's<200Pa' # 's<400Pa', '300<s<700Pa', '600<s<1000Pa'

Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True),
           (data['validatedFit_'+fit] == True),
           (data['substrate'] == '20um fibronectin discs'),
           (data['drug'] == 'none'),
           (data['date'].apply(lambda x : x in dates))]

fig, ax = D2Plot_wFit(data, XCol='surroundingThickness',YCol='EChadwick_' + fit,condCol = ['bead type', 'date'],           Filters=Filters, cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')

ax.set_ylabel('EChadwick ['+ fit +']  (Pa)')
ax.set_xlabel('Thickness at low force (nm)')
fig.suptitle('3T3aSFL: E(h) // dates = ' + dates[0][:5] + ' // ' + fit + ' // All comp')
ax.legend(loc = 'upper right', fontsize = 8)
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_E(h)', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%%


data = MecaData_All
dates = ['21-01-18', '21-01-21']
fit = '250<s<750Pa' # 's<400Pa', '300<s<700Pa', '600<s<1000Pa'

Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True),
           (data['validatedFit_'+fit] == True),
           (data['substrate'] == '20um fibronectin discs'),
           (data['drug'] == 'none'),
           (data['date'].apply(lambda x : x in dates))]

fig, ax = D2Plot_wFit(data, XCol='surroundingThickness',YCol='EChadwick_' + fit,condCol = ['bead type', 'date'],           Filters=Filters, cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')

ax.set_ylabel('EChadwick ['+ fit +']  (Pa)')
ax.set_xlabel('Thickness at low force (nm)')
fig.suptitle('3T3aSFL: E(h) // dates = ' + dates[0][:5] + ' // ' + fit + ' // All comp', fontsize = 17)
ax.legend(loc = 'upper right', fontsize = 8)
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_E(h)', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%%


data = MecaData_All
dates = ['21-01-18', '21-01-21']
fit = '500<s<1000Pa' # 's<400Pa', '300<s<700Pa', '600<s<1000Pa'

Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True),
           (data['validatedFit_'+fit] == True),
           (data['substrate'] == '20um fibronectin discs'),
           (data['drug'] == 'none'),
           (data['date'].apply(lambda x : x in dates))]

fig, ax = D2Plot_wFit(data, XCol='surroundingThickness',YCol='EChadwick_' + fit,condCol = ['bead type', 'date'],           Filters=Filters, cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')

ax.set_ylabel('EChadwick ['+ fit +']  (Pa)')
ax.set_xlabel('Thickness at low force (nm)')
fig.suptitle('3T3aSFL: E(h) // dates = ' + dates[0][:5] + ' // ' + fit + ' // All comp', fontsize = 17)
ax.legend(loc = 'upper right', fontsize = 8)
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_E(h)', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%%


data = MecaData_All
dates = ['21-12-08', '21-12-16', '22-01-12']
fit = '' # ['s<100Pa', '100<s<200Pa', '200<s<300Pa']

Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True),
           (data['validatedFit'+fit] == True),
           (data['substrate'] == '20um fibronectin discs'),
           (data['drug'] == 'none'),
           (data['date'].apply(lambda x : x in dates))]

fig, ax = D2Plot_wFit(data, XCol='surroundingThickness',YCol='EChadwick' + fit,condCol = ['bead type', 'date'],           Filters=Filters, cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')

ax.set_ylabel('EChadwick ['+ fit +']  (Pa)')
ax.set_xlabel('Thickness at low force (nm)')
ax.legend(loc = 'upper right', fontsize = 8)
fig.suptitle('3T3aSFL: E(h) // dates = ' + dates[0][:5] + ' // ' + fit + ' // All comp')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_E(h)', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%%


data = MecaData_All
dates = ['22-01-12']
fit = '_s<400Pa' # ['s<100Pa', '100<s<200Pa', '200<s<300Pa']

Filters = [(MecaData_All['validatedFit'] == True), 
           (MecaData_All['validatedThickness'] == True), 
           (MecaData_All['substrate'] == '20um fibronectin discs'),
           (MecaData_All['date'].apply(lambda x : x in dates))]

fig, ax = D2Plot_wFit(data, XCol='surroundingThickness',YCol='EChadwick' + fit,condCol = ['bead type'],           Filters=Filters, cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')

ax.set_ylabel('EChadwick ['+ fit[1:] +']  (Pa)')
ax.set_xlabel('Thickness at low force (nm)')
fig.suptitle('3T3aSFL: E(h)')
ax.legend(loc = 'upper right', fontsize = 8)
fig.suptitle('3T3aSFL: E(h) // dates = ' + dates[0][:5] + ' // ' + fit[1:] + ' // All comp')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_E(h)', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%%


data = MecaData_All
dates = ['22-01-12']
fit = '_s<300Pa' # ['s<100Pa', '100<s<200Pa', '200<s<300Pa']

Filters = [(MecaData_All['validatedFit'] == True), 
           (MecaData_All['validatedThickness'] == True), 
           (MecaData_All['substrate'] == '20um fibronectin discs'),
           (MecaData_All['date'].apply(lambda x : x in dates))]

fig, ax = D2Plot_wFit(data, XCol='surroundingThickness',YCol='EChadwick' + fit,condCol = ['bead type'],           Filters=Filters, cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')

ax.set_ylabel('EChadwick ['+ fit[1:] +']  (Pa)')
ax.set_xlabel('Thickness at low force (nm)')
fig.suptitle('3T3aSFL: E(h)')
ax.legend(loc = 'upper right', fontsize = 8)
fig.suptitle('3T3aSFL: E(h) // dates = ' + dates[0][:5] + ' // ' + fit[1:] + ' // All comp')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_E(h)', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%%


data = MecaData_All
dates = ['21-12-08', '21-12-16', '22-01-12']
fit = '' # ['s<100Pa', '100<s<200Pa', '200<s<300Pa']

Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True), 
           (data['substrate'] == '20um fibronectin discs'),
           (data['date'].apply(lambda x : x in dates))]

fig, ax = D2Plot_wFit(MecaData_All, XCol='ctFieldThickness',YCol='EChadwick' + fit, condCol = ['bead type', 'date'],           Filters=Filters, cellID = 'cellID', AvgPerCell=True, xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')

ax.set_ylabel('EChadwick ['+ fit[1:] +']  (Pa)')
ax.set_xlabel('Thickness at low force (nm)')
fig.suptitle('3T3aSFL: E(h)')
ax.legend(loc = 'upper right', fontsize = 8)
fig.suptitle('3T3aSFL: E(h) // dates = ' + dates[0][:5] + ' // ' + fit[1:] + ' // All comp')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_E(h)_all3exp', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%%


data = MecaData_All
dates = ['21-12-08', '22-01-12']
fit = '_s<400Pa' # ['s<100Pa', '100<s<200Pa', '200<s<300Pa']

# Filters = [(data['validatedFit'] == True), 
#            (data['validatedThickness'] == True),
#            (data['validatedFit_'+fit] == True),
#            (data['substrate'] == '20um fibronectin discs'),
#            (data['drug'] == 'none'),
#            (data['date'].apply(lambda x : x in dates))]

# fig, ax = D2Plot_wFit(data, XCol='surroundingThickness', YCol='EChadwick_' + fit, condCol = ['bead type', 'date'],\
#            Filters=Filters, cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')


Filters = [(data['validatedFit'] == True), 
           (data['validatedThickness'] == True), 
           (data['substrate'] == '20um fibronectin discs'),
           (data['date'].apply(lambda x : x in dates))]

fig, ax = D2Plot_wFit(MecaData_All, XCol='surroundingThickness',YCol='EChadwick' + fit, condCol = ['bead type', 'date'],           Filters=Filters, cellID = 'cellID', AvgPerCell=False, xscale = 'log', yscale = 'log', modelFit=True, modelType='y=k*x^a')

ax.set_ylabel('EChadwick ['+ fit[1:] +']  (Pa)')
ax.set_xlabel('Thickness at low force (nm)')
fig.suptitle('3T3aSFL: E(h)')
ax.legend(loc = 'upper right', fontsize = 8)
fig.suptitle('3T3aSFL: E(h) // dates = ' + dates[0][:5] + ' // ' + fit[1:] + ' // All comp')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_E(h)_all3exp', figDir = cp.DirDataFigToday + '//' + figSubDir)
plt.show()


# %%%%%


# plt.close('all')
data = MecaData_All

Filters = [(data['validatedFit'] == True), (data['validatedThickness'] == True),
          (data['date'].apply(lambda x : x in ['21-12-08', '22-01-12']))] #, '21-12-16'

fig, ax = D1PlotDetailed(data, condCol=['bead type'], Parameters=['surroundingThickness', 'EChadwick'], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=[], stats=True, statMethod='Mann-Whitney', 
               box_pairs=[], figSizeFactor = 1.8, markersizeFactor=1, orientation = 'v', showManips = True)
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_Detailed1DPlot', figSubDir = figSubDir)
plt.show()


# %%%%%


plt.close('all')
data = MecaData_All

Filters = [(data['validatedFit'] == True), (data['validatedThickness'] == True),
          (data['date'].apply(lambda x : x in ['21-12-08', '21-12-16', '22-01-12']))] #, '21-12-16'

box_pairs=[('21-12-08 & M270', '21-12-08 & M450'),
             ('21-12-16 & M450', '21-12-16 & M270'),
             ('22-01-12 & M270', '22-01-12 & M450')]

fig, ax = D1PlotDetailed(data, condCol=['date', 'bead type'], Parameters=['surroundingThickness', 'EChadwick'], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=[], stats=True, statMethod='Mann-Whitney', 
               box_pairs=box_pairs, figSizeFactor = 0.9, markersizeFactor=1, orientation = 'v', showManips = True)
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan22_M450vsM270_CompressionsLowStart_Detailed1DPlot_all3exp', figSubDir = figSubDir)
plt.show()


# %%%% On Feb 2022 data

# %%%%%


data = MecaData_NonLin
data


# %%%%%


# ['21-12-08', '21-12-16', '22-01-12']
data = MecaData_NonLin

dates = ['22-02-09'] #['21-12-08', '22-01-12'] ['21-01-18', '21-01-21', '21-12-08']

filterList = [(data['validatedThickness'] == True),
              (data['cell subtype'] == 'aSFL'), 
              (data['bead type'] == 'M450'),
              (data['substrate'] == '20um fibronectin discs'),
              (data['date'].apply(lambda x : x in dates))]  # (data['validatedFit'] == True), 


fig, ax = plt.subplots(1, 1, figsize = (9, 6), tight_layout=True)

globalFilter = filterList[0]
for k in range(1, len(filterList)):
    globalFilter = globalFilter & filterList[k]


X = data[globalFilter]['bestH0'].values
Ncomp = len(X)
Smin = data[globalFilter]['minStress'].values
Smax = data[globalFilter]['maxStress'].values

for i in range(Ncomp):
    ax.plot([X[i], X[i]], [Smin[i], Smax[i]], 
            ls = '-', color = 'deepskyblue', alpha = 0.3,
            label = 'Stress range', zorder = 1)
    ax.plot([X[i]], [Smin[i]], 
            marker = 'o', markerfacecolor = 'skyblue', markeredgecolor = 'k', markersize = 4,
            ls = '')
    ax.plot([X[i]], [Smax[i]], 
            marker = 'o', markerfacecolor = 'royalblue', markeredgecolor = 'k', markersize = 4,
            ls = '')

ax.set_xlabel('Cortical thickness (nm)')
ax.set_xlim([0, 1200])
locator = matplotlib.ticker.MultipleLocator(100)
ax.xaxis.set_major_locator(locator)

ax.set_ylabel('Extremal stress values (Pa)')
ax.set_ylim([0, 2500])
locator = matplotlib.ticker.MultipleLocator(100)
ax.yaxis.set_major_locator(locator)
ax.yaxis.grid(True)

ax.set_title('Feb 22 - Extremal stress values vs. thickness, for each compression')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Feb22_CompressionsLowStart_StressRanges', figSubDir = 'NonLin')
print(Ncomp)
plt.show()


# %%%% First K(s) curves

# %%%%% Make the dataframe


# Make the dataframe

data = MecaData_NonLin

dates = ['22-02-09'] #['21-12-08', '22-01-12'] ['21-01-18', '21-01-21', '21-12-08']

filterList = [(data['validatedThickness'] == True),
              (data['cell subtype'] == 'aSFL'), 
              (data['bead type'] == 'M450'),
              (data['substrate'] == '20um fibronectin discs'),
              (data['date'].apply(lambda x : x in dates))]  # (data['validatedFit'] == True), 
globalFilter = filterList[0]
for k in range(1, len(filterList)):
    globalFilter = globalFilter & filterList[k]

data_f = data[globalFilter]

fitMin = [S for S in range(25,1025,50)]
fitMax = [S+150 for S in fitMin]
fitCenters = np.array([S+75 for S in fitMin])
regionFitsNames = [str(fitMin[ii]) + '<s<' + str(fitMax[ii]) for ii in range(len(fitMin))]

listColumnsMeca = []

KChadwick_Cols = []
Kweight_Cols = []

for rFN in regionFitsNames:
    listColumnsMeca += ['KChadwick_'+rFN, 'K_CIW_'+rFN, 'R2Chadwick_'+rFN, 'K2Chadwick_'+rFN, 
                        'H0Chadwick_'+rFN, 'Npts_'+rFN, 'validatedFit_'+rFN]
    KChadwick_Cols += [('KChadwick_'+rFN)]

    K_CIWidth = data_f['K_CIW_'+rFN] #.apply(lambda x : x.strip('][').split(', ')).apply(lambda x : (np.abs(float(x[0]) - float(x[1]))))
    Kweight = (data_f['KChadwick_'+rFN]/K_CIWidth)**2
    data_f['K_weight_'+rFN] = Kweight
    data_f['K_weight_'+rFN] *= data_f['KChadwick_'+rFN].apply(lambda x : (x<1e6))
    data_f['K_weight_'+rFN] *= data_f['R2Chadwick_'+rFN].apply(lambda x : (x>1e-2))
    data_f['K_weight_'+rFN] *= data_f['K_CIW_'+rFN].apply(lambda x : (x!=0))
    Kweight_Cols += [('K_weight_'+rFN)]
    
data_f.tail()


# %%%%% K(s) cell by cell


condCol = 'date'
Variables = KChadwick_Cols
weightCols = Kweight_Cols
data_f_agg = getAggDf_K_wAvg(data_f, 'cellID', condCol, Variables, weightCols)
data_f_agg.T
dictPlot = {'cellID' : [], 'Kavg' : [], 'Kstd' : [], 'Kcount' : []}
meanCols = []
stdCols = []
countCols = []
for col in data_f_agg.columns:
    if 'KChadwick' in col and 'Wmean' in col:
        meanCols.append(col)
    elif 'KChadwick' in col and '_Wstd' in col:
        stdCols.append(col)
    elif 'KChadwick' in col and '_Count' in col:
        countCols.append(col)
meanDf = data_f_agg[meanCols]
stdDf = data_f_agg[stdCols]
countDf = data_f_agg[countCols]
for c in data_f_agg.index:
    means = meanDf.T[c].values
    stds = stdDf.T[c].values
    counts = countDf.T[c].values
    dictPlot['cellID'].append(c)
    dictPlot['Kavg'].append(means)
    dictPlot['Kstd'].append(stds)
    dictPlot['Kcount'].append(counts)


fig, ax = plt.subplots(1,1, figsize = (9,6))
for i in range(len(dictPlot['cellID'])):
    c = dictPlot['cellID'][i]
    color = gs.colorList10[i%10]
#     ax.errorbar(fitCenters, dictPlot['Kavg'][i], yerr = dictPlot['Kstd'][i], color = color)
    ax.plot(fitCenters, dictPlot['Kavg'][i], color = color)
    low =  dictPlot['Kavg'][i] - (dictPlot['Kstd'][i] / (dictPlot['Kcount'][i]**0.5)) 
    high = dictPlot['Kavg'][i] + (dictPlot['Kstd'][i] / (dictPlot['Kcount'][i]**0.5))
    low = np.where(low < 10, 10, low)
    matplotlib.pyplot.fill_between(x=fitCenters, 
                                   y1=low, 
                                   y2=high,
                                   color = color, alpha = 0.1, zorder = 1)

ax.set_xlabel('Stress (Pa)')
ax.set_ylabel('K (Pa)')
ax.set_yscale('log')
ax.set_ylim([1e2, 1e5])
fig.suptitle('Stress stiffening - On each cell') #\n1 day, 3 expts, 36 cells, 232 compression
# ufun.archiveFig(fig, ax, cp.DirDataFigToday + '//NonLin', name='NonLin_K(s)_cellByCell', dpi = 100)
plt.show()


# %%%%% K(s) two different error types


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
weightStr = 'K_weight_'

Kavg = []
Kstd = []
D10 = []
D90 = []
N = []

for S in range(100,1100,50):
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

fig, ax = plt.subplots(2,1, figsize = (9,12))

ax[0].errorbar(fitCenters, Kavg, yerr = q*Kste, marker = 'o', color = gs.my_default_color_list[0], 
               ecolor = 'k', elinewidth = 0.8, capsize = 3, label = 'weighted means\nweighted ste 95% as error')
ax[0].set_ylim([50,1e5])

ax[1].errorbar(fitCenters, Kavg, yerr = [D10, D90], marker = 'o', color = gs.my_default_color_list[3], 
               ecolor = 'k', elinewidth = 0.8, capsize = 3, label = 'weighted means\nD9-D1 as error')
ax[1].set_ylim([500,1e6])

for k in range(2):
    ax[k].legend(loc = 'upper left')
    ax[k].set_yscale('log')
    ax[k].set_xlabel('Stress (Pa)')
    ax[k].set_ylabel('K (Pa)')
    for kk in range(len(N)):
        ax[k].text(x=fitCenters[kk]+5, y=Kavg[kk]**0.98, s='n='+str(N[kk]), fontsize = 6)

fig.suptitle('Stress stiffening - All compressions pooled') # \n1 day, 3 expts, 36 cells, 232 compression
# ufun.archiveFig(fig, ax, cp.DirDataFigToday + '//NonLin', name='NonLin_K(s)_twoErrorsTypes', dpi = 100)
plt.show()

df_val = pd.DataFrame(d_val)
dftest = pd.DataFrame(d)

df_val


# %%%%% K(s) with only ste for error bars


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
weightStr = 'K_weight_'

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

# FIT ?
X, Y = fitCenters, Kavg
eqnText = "Linear Fit"
params, results = ufun.fitLine(X, Y) # Y=a*X+b ; params[0] = b,  params[1] = a
pval = results.pvalues[1] # pvalue on the param 'a'
eqnText += " ; Y = {:.1f} X + {:.1f}".format(params[1], params[0])
eqnText += " ; p-val = {:.3f}".format(pval)
print("Y = {:.5} X + {:.5}".format(params[1], params[0]))
print("p-value on the 'a' coefficient: {:.4e}".format(pval))
fitY = params[1]*X + params[0]
imin = np.argmin(X)
imax = np.argmax(X)


fig, ax = plt.subplots(1,1, figsize = (9,6)) # (2,1, figsize = (9,12))

# ax[0]
ax.errorbar(fitCenters, Kavg/1000, yerr = q*Kste/1000, 
            marker = 'o', color = gs.my_default_color_list[0],
            markersize = 7.5, lw = 2,
            ecolor = 'k', elinewidth = 1.5, capsize = 5, 
            label = 'weighted means\nweighted ste 95% as error')
ax.set_ylim([0,1.6e4/1000])
ax.set_xlim([0,1150])

ax.plot([X[imin],X[imax]], [fitY[imin]/1000,fitY[imax]/1000], 
        ls = '--', lw = '1', color = 'darkorange', zorder = 1, label = eqnText)

# ax[1].errorbar(fitCenters, Kavg, yerr = [D10, D90], marker = 'o', color = gs.my_default_color_list[3], 
#                ecolor = 'k', elinewidth = 0.8, capsize = 3, label = 'weighted means\nD9-D1 as error')
# ax[1].set_ylim([500,1e6])

# for k in range(1): #2
#     ax[k].legend(loc = 'upper left')
#     ax[k].set_yscale('log')
#     ax[k].set_xlabel('Stress (Pa) [center of a 150Pa large interval]')
#     ax[k].set_ylabel('K (Pa) [tangeant modulus w/ Chadwick]')
#     for kk in range(len(N)):
#         ax[k].text(x=fitCenters[kk]+5, y=Kavg[kk]**0.98, s='n='+str(N[kk]), fontsize = 6)
ax.legend(loc = 'upper left')
ax.set_yscale('linear')
ax.set_xlabel('Stress (Pa)') #' [center of a 150Pa large interval]')
ax.set_ylabel('Tangeantial Modulus (kPa)') #' [tangeant modulus w/ Chadwick]')
for kk in range(len(N)):
    # ax.text(x=fitCenters[kk]+5, y=(Kavg[kk]**0.98), s='n='+str(N[kk]), fontsize = 10)
    ax.text(x=fitCenters[kk]+5, y=(Kavg[kk]-500)/1000, s='n='+str(N[kk]), fontsize = 8)

# fig.suptitle('K(sigma)')
ax.set_title('Stress-stiffening of the actin cortex\nALL compressions pooled') #  '\n22-02-09 experiment, 36 cells, 232 compression')
ufun.archiveFig(fig, ax, cp.DirDataFigToday + '//NonLin', name='NonLin_K(s)_lin', dpi = 100)
plt.show()

# df_val = pd.DataFrame(d_val)
# dftest = pd.DataFrame(d)

# df_val


# %%%%% 100-800Pa range


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
weightStr = 'K_weight_'

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
               ecolor = 'k', elinewidth = 0.8, capsize = 3, label = 'weighted means\nweighted ste 95% as error')
ax.set_ylim([0,1.4e4])

# ax[1].errorbar(fitCenters, Kavg, yerr = [D10, D90], marker = 'o', color = gs.my_default_color_list[3], 
#                ecolor = 'k', elinewidth = 0.8, capsize = 3, label = 'weighted means\nD9-D1 as error')
# ax[1].set_ylim([500,1e6])

# for k in range(1): #2
#     ax[k].legend(loc = 'upper left')
#     ax[k].set_yscale('log')
#     ax[k].set_xlabel('Stress (Pa) [center of a 150Pa large interval]')
#     ax[k].set_ylabel('K (Pa) [tangeant modulus w/ Chadwick]')
#     for kk in range(len(N)):
#         ax[k].text(x=fitCenters[kk]+5, y=Kavg[kk]**0.98, s='n='+str(N[kk]), fontsize = 6)
ax.legend(loc = 'upper left')
ax.set_yscale('linear')
ax.set_xlabel('Stress (Pa)')
ax.set_ylabel('K (Pa)')
for kk in range(len(N)):
    ax.text(x=fitCenters2[kk]+5, y=Kavg[kk]**0.98, s='n='+str(N[kk]), fontsize = 6)

# fig.suptitle('K(sigma)')
ax.set_title('Stress stiffening\nOnly compressions including the [100, 800]Pa range')
ufun.archiveFig(fig, ax, cp.DirDataFigToday + '//NonLin', name='NonLin_K(s)_100-800Pa_lin', dpi = 100)
plt.show()

# df_val = pd.DataFrame(d_val)
# dftest = pd.DataFrame(d)

# df_val


# %%%%% 100-700Pa range


# print(data_f.head())

# print(fitCenters)

extraFilters = [data_f['minStress'] <= 100, data_f['maxStress'] >= 700]
fitCenters2 = fitCenters[fitCenters<=700]

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
weightStr = 'K_weight_'

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
               ecolor = 'k', elinewidth = 0.8, capsize = 3, label = 'weighted means\nweighted ste 95% as error')
ax.set_ylim([0,1.4e4])

# ax[1].errorbar(fitCenters, Kavg, yerr = [D10, D90], marker = 'o', color = gs.my_default_color_list[3], 
#                ecolor = 'k', elinewidth = 0.8, capsize = 3, label = 'weighted means\nD9-D1 as error')
# ax[1].set_ylim([500,1e6])

# for k in range(1): #2
#     ax[k].legend(loc = 'upper left')
#     ax[k].set_yscale('log')
#     ax[k].set_xlabel('Stress (Pa) [center of a 150Pa large interval]')
#     ax[k].set_ylabel('K (Pa) [tangeant modulus w/ Chadwick]')
#     for kk in range(len(N)):
#         ax[k].text(x=fitCenters[kk]+5, y=Kavg[kk]**0.98, s='n='+str(N[kk]), fontsize = 6)
ax.legend(loc = 'upper left')
ax.set_yscale('linear')
ax.set_xlabel('Stress (Pa)')
ax.set_ylabel('K (Pa)')
for kk in range(len(N)):
    ax.text(x=fitCenters2[kk]+5, y=Kavg[kk]**0.98, s='n='+str(N[kk]), fontsize = 6)

# fig.suptitle('K(sigma)')
ax.set_title('Stress stiffening\nOnly compressions including the [100, 700]Pa range')
ufun.archiveFig(fig, ax, cp.DirDataFigToday + '//NonLin', name='NonLin_K(s)_100-700Pa_lin', dpi = 100)
plt.show()

# df_val = pd.DataFrame(d_val)
# dftest = pd.DataFrame(d)

# df_val


# %%%%% 100-600Pa range


# print(data_f.head())

# print(fitCenters)

extraFilters = [data_f['minStress'] <= 100, data_f['maxStress'] >= 550]
fitCenters2 = fitCenters[fitCenters<=550]

data_ff = data_f
globalExtraFilter = extraFilters[0]
for k in range(1, len(extraFilters)):
    globalExtraFilter = globalExtraFilter & extraFilters[k]

data_ff = data_ff[globalExtraFilter]
print(len(data_ff['cellID'].unique()), data_ff['cellID'].unique())
inspectDict = {}
for cId in data_ff['cellID'].unique():
    inspectDict[cId] = data_ff[data_ff['cellID'] == cId].shape[0]
print(inspectDict)

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
weightStr = 'K_weight_'

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
ax.errorbar(fitCenters2, Kavg/1000, yerr = q*Kste/1000, 
            marker = 'o', color = gs.my_default_color_list[0],
            markersize = 10, lw = 2,
            ecolor = 'k', elinewidth = 1.5, capsize = 5, 
            label = 'weighted means\nweighted ste 95% as error')
ax.set_ylim([0,1.0e4/1000])

# ax[1].errorbar(fitCenters, Kavg, yerr = [D10, D90], marker = 'o', color = gs.my_default_color_list[3], 
#                ecolor = 'k', elinewidth = 0.8, capsize = 3, label = 'weighted means\nD9-D1 as error')
# ax[1].set_ylim([500,1e6])

# for k in range(1): #2
#     ax[k].legend(loc = 'upper left')
#     ax[k].set_yscale('log')
#     ax[k].set_xlabel('Stress (Pa) [center of a 150Pa large interval]')
#     ax[k].set_ylabel('K (Pa) [tangeant modulus w/ Chadwick]')
#     for kk in range(len(N)):
#         ax[k].text(x=fitCenters[kk]+5, y=Kavg[kk]**0.98, s='n='+str(N[kk]), fontsize = 6)
ax.legend(loc = 'upper left')
ax.set_yscale('linear')
ax.set_xlabel('Stress (Pa)') #  [center of a 150Pa large interval]
ax.set_ylabel('Tangeantial Modulus (kPa)') # [tangeant modulus w/ Chadwick]
#### Decomment to get the number of compressions again
# for kk in range(len(N)):
#     ax.text(x=fitCenters2[kk]+5, y=Kavg[kk]**0.98, s='n='+str(N[kk]), fontsize = 6)

# fig.suptitle('K(sigma)')
ax.set_title('Stress-stiffening of the actin cortex')#'\nOnly compressions including the [100, 600]Pa range')

ufun.archiveFig(fig, ax, cp.DirDataFigToday + '//NonLin', name='NonLin_K(s)_100-600Pa_lin', dpi = 100)
plt.show()

# df_val = pd.DataFrame(d_val)
# dftest = pd.DataFrame(d)

# df_val


Kavg_ref, Kste_ref, N_ref, fitCenters_ref = Kavg, Kste, N, fitCenters2



# %%%%% Multiple boxplot K(s)

data = data_f

listeS = [100, 150, 200, 250, 300, 400, 500, 600, 700, 800]

fig, axes = plt.subplots(1, len(listeS), figsize = (14,6))
kk = 0

for S in listeS:
    interval = str(S-75) + '<s<' + str(S+75)

    Filters = [(data['validatedFit_'+interval] == True),
               (data['validatedThickness'] == True),
               (data['date'].apply(lambda x : x in ['22-02-09']))] #, '21-12-16'

    fig, ax = D1Plot(data, fig=fig, ax=axes[kk], condCol=['bead type'], Parameters=['KChadwick_'+interval], Filters=Filters, 
                   Boxplot=True, cellID='cellID', co_order=[], stats=True, statMethod='Mann-Whitney', AvgPerCell = True,
                   box_pairs=[], figSizeFactor = 1, markersizeFactor=1, orientation = 'h')

#     axes[kk].legend(loc = 'upper right', fontsize = 8)
    label = axes[kk].get_ylabel()
    axes[kk].set_title(label.split('_')[-1] + 'Pa', fontsize = 10)
    axes[kk].set_yscale('log')
    axes[kk].set_ylim([1e2, 5e4])
    axes[kk].tick_params(axis='x', labelrotation = 50, labelsize = 10)
    axes[kk].set_xticklabels([])
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
fig.suptitle('Tangeantial modulus of aSFL 3T3 - 0.5>50mT lowStartCompressions - avg per cell')
plt.show()
ufun.archiveFig(fig, ax, name='3T3aSFL_Feb22_nonLinK_multipleIntervals', figSubDir = 'NonLin')


# %%%%% Multiple K(h)



data = MecaData_NonLin

listeS = [100, 150, 200, 250, 300, 400, 500, 600, 700, 800]

# fig, axes = plt.subplots(len(listeS), 1, figsize = (9,40))
# kk = 0

for S in listeS:
    interval = str(S-75) + '<s<' + str(S+75)
    textForFileName = str(S-75) + '-' + str(S+75) + 'Pa'

    Filters = [(data['validatedFit_'+interval] == True),
               (data['validatedThickness'] == True),
               (data['bestH0'] <= 600),
               (data['date'].apply(lambda x : x in ['22-02-09']))] #, '21-12-16'

    fig, ax = D2Plot_wFit(data, #fig=fig, ax=axes[kk], 
                          XCol='bestH0', YCol='KChadwick_'+interval, condCol = ['bead type'], Filters=Filters, 
                          cellID = 'cellID', AvgPerCell=False, 
                          xscale = 'linear', yscale = 'log', modelFit=True, modelType='y=k*x^a')
    
    # individual figs
    ax.set_xlim([8e1, 700])
    ax.set_ylim([1e2, 5e4])
    renameAxes(ax,{'bestH0':'best H0 (nm)', 'doxycyclin':'iMC'})
    fig.set_size_inches(6,4)
    fig.tight_layout()
#     ufun.archiveFig(fig, ax, name='3T3aSFL_Feb22_nonLin_K(h)_' + textForFileName, figSubDir = 'NonLin')
    
    
    # one big fig
#     axes[kk].set_xlim([8e1, 1.2e3])
#     axes[kk].set_ylim([3e2, 5e4])
#     kk+=1


# renameAxes(axes,{'none':'ctrl', 'doxycyclin':'iMC'})
# fig.tight_layout()
plt.show()


# %%%%% Multiple K(h) - 2



data = MecaData_NonLin

listeS = [200]

# fig, axes = plt.subplots(len(listeS), 1, figsize = (9,40))
# kk = 0

for S in listeS:
    halfWidth = 125
    interval = str(S-halfWidth) + '<s<' + str(S+halfWidth)
    textForFileName = str(S-halfWidth) + '-' + str(S+halfWidth) + 'Pa'

    Filters = [(data['validatedFit_'+interval] == True),
               (data['validatedThickness'] == True),
               (data['bestH0'] <= 650),
               (data['date'].apply(lambda x : x in ['22-02-09']))] #, '21-12-16'

    fig, ax = D2Plot_wFit(data, #fig=fig, ax=axes[kk], 
                          XCol='bestH0', YCol='KChadwick_'+interval, 
                          condCol = ['bead type'], Filters=Filters, 
                          cellID = 'cellID', AvgPerCell=False, 
                          xscale = 'log', yscale = 'log', modelFit=True, 
                          modelType='y=k*x^a', markersizeFactor = 1.2)
    
    # individual figs
    ax.set_xlim([2e2, 700])
    ax.set_ylim([5e2, 3e4])
    # ax.set_ylim([0, 8000])
    fig.suptitle(interval)
    ax.get_legend().set_visible(False)
    rD = {'bestH0':'Thickness (nm)',
          'KChadwick_'+interval : 'Tangeantial Modulus (Pa)'}
    renameAxes(ax,rD)
    fig.set_size_inches(8,7)
    fig.tight_layout()
#     ufun.archiveFig(fig, ax, name='3T3aSFL_Feb22_nonLin_K(h)_' + textForFileName, figSubDir = 'NonLin')
    
    
    # one big fig
#     axes[kk].set_xlim([8e1, 1.2e3])
#     axes[kk].set_ylim([3e2, 5e4])
#     kk+=1


# renameAxes(axes,{'none':'ctrl', 'doxycyclin':'iMC'})
# fig.tight_layout()
plt.show()

# %%%%% test


# dftest['produit'] = dftest.x.values*dftest.w.values
# wm = np.average(dftest.x.values, weights = dftest.w.values)
# print(np.sum(dftest.x.values*dftest.w.values))
# print(np.sum(dftest['produit']))
# print(np.sum(dftest.w.values))
# print((np.sum(dftest['produit']))/np.sum(dftest.w.values))
# print(wm)
# print(max(dftest.produit.values))
# dftest

# %%%% Different range width

# %%%%% All stress ranges - K(stress) -- With bestH0 distributions and E(h)


data = MecaData_NonLin
dates = ['22-02-09']

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
            
                D1Plot(data, fig=fig, ax=axes[0, kk], condCol=['bead type'], Parameters=['KChadwick_'+interval], 
                     Filters=Filters, Boxplot=True, cellID='cellID', 
                     stats=True, statMethod='Mann-Whitney', AvgPerCell = True, box_pairs=[], 
                     figSizeFactor = 1, markersizeFactor=1, orientation = 'h', 
                     stressBoxPlot = True)# , bypassLog = True)
                
                D1Plot(data, fig=fig, ax=axes[1, kk], condCol=['bead type'], Parameters=['bestH0'], 
                     Filters=Filters, Boxplot=True, cellID='cellID', 
                     stats=True, statMethod='Mann-Whitney', AvgPerCell = True, box_pairs=[], 
                     figSizeFactor = 1, markersizeFactor=1, orientation = 'h', stressBoxPlot = True)
                
                D2Plot_wFit(data, fig=fig, ax=axes[2, kk], Filters=Filters, 
                      XCol='bestH0', YCol='KChadwick_'+interval, condCol = ['bead type'],
                      cellID = 'cellID', AvgPerCell=True, xscale = 'log', yscale = 'log', 
                      modelFit=True, modelType='y=k*x^a', markersizeFactor = 1)
                
            except:
                pass
            
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
            
        else:
            pass
    
    renameAxes(axes.flatten(),{'none':'ctrl', 'doxycyclin':'iMC', 'bestH0' : 'Best H0 (nm)'})
    # fig.tight_layout()
    # fig.suptitle('Tangeantial modulus of aSFL 3T3 - control vs iMC linker')
    
    plt.show()


    # ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_K_multipleIntervals_250pa_lin', figDir = cp.DirDataFigToday + '//' + figSubDir, figSubDir = figSubDir)
    # ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_K_multipleIntervals_250pa_lin', figDir = ownCloudTodayFigDir, figSubDir = figSubDir)


# %%%%% All stress ranges - K(stress)

#### Creation of data_new, with the different width piled on top of each other

data = MecaData_NonLin
dates = ['22-02-09']

fitC =  np.array([S for S in range(100, 1000, 100)])
fitW = [100, 150, 200, 250, 300]
fitCenters = np.array([[int(S) for S in fitC] for w in fitW])
fitWidth = np.array([[int(w) for S in fitC] for w in fitW])
fitMin = np.array([[int(S-(w/2)) for S in fitC] for w in fitW])
fitMax = np.array([[int(S+(w/2)) for S in fitC] for w in fitW])
# fitCenters, fitWidth = fitCenters[fitMin>0], fitWidth[fitMin>0]
# fitMin, fitMax = fitMin[fitMin>0], fitMax[fitMin>0]

cols = data.columns
mainCols = []
for c in cols:
    strC = str(c)
    if '<s<' not in strC:
        mainCols.append(c)
        
data_bloc = data[mainCols]
nRows = data_bloc.shape[0]

cols_fits = np.array([['validatedFit_S='+str(S), 'KChadwick_S='+str(S), 
                       'K_CIW_S='+str(S), 'R2Chadwick_S='+str(S), 
                       'Npts_S='+str(S)] for S in fitC]).flatten()
for c in cols_fits:
    if 'validated' in c:
        data_bloc[c] = np.zeros(nRows, dtype=bool)
    else:
        data_bloc[c] = np.ones(nRows) * np.nan
        
data_bloc['fitWidth'] = np.ones(nRows)
data_new = data_bloc[:]
i = 0
for width in fitW:
    data_bloc['fitWidth'] = np.ones(nRows, dtype = int) * int(width)
    for S in fitC:
        try:
            minS, maxS = int(S-width//2), int(S+width//2)
            interval = str(minS) + '<s<' + str(maxS)
            data_bloc['validatedFit_S='+str(S)] = data['validatedFit_' + interval]
            data_bloc['KChadwick_S='+str(S)] = data['KChadwick_' + interval]
            data_bloc['K_CIW_S='+str(S)] = data['K_CIW_' + interval]
            data_bloc['R2Chadwick_S='+str(S)] = data['R2Chadwick_' + interval]
            data_bloc['Npts_S='+str(S)] = data['Npts_' + interval]
        except:
            minS, maxS = int(S-width//2), int(S+width//2)
            interval = str(minS) + '<s<' + str(maxS)
            data_bloc['ValidatedFit_S='+str(S)] = np.zeros(nRows, dtype=bool)
            data_bloc['KChadwick_S='+str(S)] = np.ones(nRows) * np.nan
            data_bloc['K_CIW_S='+str(S)] = np.ones(nRows) * np.nan
            data_bloc['R2Chadwick_S='+str(S)] = np.ones(nRows) * np.nan
            data_bloc['Npts_S='+str(S)] = np.ones(nRows) * np.nan
    if i == 0:
        data_new = data_bloc[:]
    else:
        data_new = pd.concat([data_new, data_bloc])
        
    i += 1
    
data_new = data_new.reset_index(drop=True)

#### Plot

fitCPlot = np.array([S for S in range(200, 900, 100)])
N = len(fitCPlot)
fig, axes = plt.subplots(1, N, figsize = (25,8))
kk = 0

for S in fitCPlot:
    # minS, maxS = int(S-width//2), int(S+width//2)
    # interval = str(minS) + '<s<' + str(maxS)
    Filters = [(data_new['validatedFit_S='+str(S)] == True),
               (data_new['validatedThickness'] == True),
               (data_new['bestH0'] <= 1100),
               (data_new['date'].apply(lambda x : x in dates))]
    
    
    # axes[0, kk].set_yscale('log')
    axes[kk].set_ylim([1e2, 5e4])
    # axes[0, kk].set_yscale('linear')
    # axes[0, kk].set_ylim([0, 3e4])

    D1Plot(data_new, fig=fig, ax=axes[kk], condCol=['fitWidth'], Parameters=['KChadwick_S='+str(S)], 
         Filters=Filters, Boxplot=True, cellID='cellID', 
         stats=False, statMethod='Mann-Whitney', AvgPerCell = False, box_pairs=[], 
         figSizeFactor = 1, markersizeFactor=0.5, orientation = 'h', 
         stressBoxPlot = True)# , bypassLog = True)
    
    # D1Plot(data_new, fig=fig, ax=axes[1, kk], condCol=['fitWidth'], Parameters=['bestH0'], 
    #      Filters=Filters, Boxplot=True, cellID='cellID', 
    #      stats=False, statMethod='Mann-Whitney', AvgPerCell = False, box_pairs=[], 
    #      figSizeFactor = 1, markersizeFactor=0.5, orientation = 'h', stressBoxPlot = True)
        
    # 0.
    # axes[0, kk].legend(loc = 'upper right', fontsize = 8)
    label0 = axes[kk].get_ylabel()
    axes[kk].set_title(label0.split('_')[-1] + 'Pa', fontsize = 10)
    axes[kk].set_ylim([4e2, 5e4])
    axes[kk].tick_params(axis='y', labelsize = 10)
    
    axes[kk].set_xlabel('Width (Pa)')
    axes[kk].tick_params(axis='x', labelrotation = 0, labelsize = 10)
    
    
    if kk == 0:
        axes[kk].set_ylabel(label0.split('_')[0] + ' (Pa)')

    else:
        axes[kk].set_ylabel(None)
        axes[kk].set_yticklabels([])

    kk+=1
    
fig.suptitle('Effect of fitting window on tangeantial modulus, data from 22-02-09')
renameAxes(axes.flatten(),{'none':'ctrl', 'doxycyclin':'iMC', 'bestH0' : 'Best H0 (nm)'})
# fig.tight_layout()
# fig.suptitle('Tangeantial modulus of aSFL 3T3 - control vs iMC linker')

plt.show()


# %%%  Drug experiments

# %%%%% Summary

fig, axes = plt.subplots(2,2, figsize = (9,7))

# Part 1.

data_main = MecaData_Drugs
dates = ['22-03-30'] # ['22-03-28', '22-03-30', '22-11-23']

fitType = 'stressRegion'
fitId = '300_100'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

thicknessType = 'surroundingThickness' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data['fit_K'] = data['fit_K']/1e3

Filters = [(data['validatedThickness'] == True), 
           (data['valid_Chadwick'] == True),
           (data['surroundingThickness'] <= 700),
           (data['fit_valid'] == True),
           (data['fit_K'] <= 10),
            # (data['ctFieldMinThickness'] >= 200**2),
           (data['drug'].apply(lambda x : x in ['dmso', 'blebbistatin'])),
           (data['date'].apply(lambda x : x in dates))]

# co_order=['none', 'dmso', 'blebbistatin', 'latrunculinA']
co_order=['dmso & 0.0', 'blebbistatin & 50.0']

fig, a = D1Plot(data, fig=fig, ax = axes[:,0], condCols=['drug', 'concentration'], Parameters=[thicknessType, 'fit_K'], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 0.4, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

# Part 2.
data_main = MecaData_Drugs
dates = ['22-03-30', '22-11-23'] # ['22-03-28', '22-03-30', '22-11-23']

fitType = 'stressRegion'
fitId = '300_100'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

thicknessType = 'surroundingThickness' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

data['fit_K'] = data['fit_K']/1e3

Filters = [(data['validatedThickness'] == True), 
           (data['valid_Chadwick'] == True),
           (data['surroundingThickness'] <= 700),
           (data['fit_valid'] == True),
           (data['fit_K'] <= 10),
            # (data['ctFieldMinThickness'] >= 200**2),
           (data['drug'].apply(lambda x : x in ['dmso', 'latrunculinA'])),
           (data['date'].apply(lambda x : x in dates))]

# co_order=['none', 'dmso', 'blebbistatin', 'latrunculinA']
co_order=['dmso & 0.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']

fig, a = D1Plot(data, fig=fig, ax = axes[:,1], condCols=['drug', 'concentration'], Parameters=[thicknessType, 'fit_K'], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 0.4, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1)

# Formatting

# rD = {'none' : 'Control\n(nothing)', 'dmso' : 'Control\n(DMSO)', 
#       'blebbistatin' : 'Blebbistatin', 'latrunculinA' : 'LatA', 
#       'bestH0' : 'Thickness (nm)', 'K' + fit : 'fit_KTangeantial Modulus (Pa)'}

rD = {'dmso & 0.0' : 'DMSO', 
      'blebbistatin & 50.0' : 'Blebbi\n(50µM)', 
      'latrunculinA & 0.5' : 'LatA\n(0.5µM)', 
      'latrunculinA & 2.5' : 'LatA\n(2.5µM)',
      'Thickness at low force (nm)' : 'Thickness (nm)',
      'Tangeantial Modulus (Pa)' : 'Tangeantial Modulus (kPa)\n' + fitStr}

renameAxes(axes.flatten(), renameDict1)
renameAxes(axes.flatten(), rD)

titles = ['Blebbistatin', 'LatrunculinA']
tl = matplotlib.ticker.MultipleLocator(2)

for k in range(2):
    axes[0,k].set_ylim([0,900])
    axes[1,k].set_ylim([0,12.5])
    axes[k,1].set_ylabel('')
    # axes[0,k].set_xticklabels([])
    axes[0,k].set_title(titles[k])
    axes[1,k].yaxis.set_major_locator(tl)



# ax[0].set_ylim([0, 1000])
# ax[0].legend(loc = 'upper right', fontsize = 8)
# ax[1].legend(loc = 'upper right', fontsize = 8)
# fig.suptitle('3T3aSFL & drugs\nPreliminary data')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_H&Echad_simple', figSubDir = figSubDir)
plt.show()

# %%%% H0, Date by date

Avg = True
data_main = MecaData_Drugs

# fitType = 'stressRegion'
# fitId = '250_75'
# data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)
# valCol = 'fit_K'
# weightCol = 'fit_ciwK'

data = data_main

dates = ['22-03-30'] # ['22-03-28', '22-03-30']

Filters = [(data['drug'].apply(lambda x : x in ['none', 'dmso', 'blebbistatin', 'latrunculinA'])),
           (data['date'].apply(lambda x : x in dates))]

# co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
# co_order = []

fig, ax = D1Plot(data, condCols=['drug', 'concentration'], Parameters=['ctFieldThickness'], Filters=Filters, 
           Boxplot=True, AvgPerCell=Avg, cellID='cellID', co_order=co_order,
           stats=True, statMethod='Mann-Whitney', box_pairs=[], 
           figSizeFactor = 1, markersizeFactor = 1, orientation = 'h',
           stressBoxPlot = False, bypassLog = False, 
           returnData = 0, returnCount = 0)

fig.suptitle(dates[0])




dates = ['22-03-30'] # ['22-03-28', '22-03-30']


Filters = [(data['drug'].apply(lambda x : x in ['none', 'dmso', 'blebbistatin', 'latrunculinA'])),
           (data['date'].apply(lambda x : x in dates))]

# co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
# co_order = []

fig, ax = D1Plot(data, condCols=['drug', 'concentration'], Parameters=['ctFieldThickness'], Filters=Filters, 
           Boxplot=True, AvgPerCell=Avg, cellID='cellID', co_order=co_order,
           stats=True, statMethod='Mann-Whitney', box_pairs=[], 
           figSizeFactor = 1, markersizeFactor = 1, orientation = 'h',
           stressBoxPlot = False, bypassLog = False, 
           returnData = 0, returnCount = 0)

fig.suptitle(dates[0])



dates = ['22-11-23'] # ['22-03-28', '22-03-30']

Filters = [(data['drug'].apply(lambda x : x in ['none', 'dmso', 'blebbistatin', 'latrunculinA'])),
           (data['date'].apply(lambda x : x in dates))]

# co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
# co_order = []

fig, ax = D1Plot(data, condCols=['drug', 'concentration'], Parameters=['ctFieldThickness'], Filters=Filters, 
           Boxplot=True, AvgPerCell=Avg, cellID='cellID', co_order=co_order,
           stats=True, statMethod='Mann-Whitney', box_pairs=[], 
           figSizeFactor = 1, markersizeFactor = 1, orientation = 'h',
           stressBoxPlot = False, bypassLog = False, 
           returnData = 0, returnCount = 0)

fig.suptitle(dates[0])

plt.show()


# %%%% H0, Drug by drug

Avg = True
data_main = MecaData_Drugs


data = data_main


drugs = ['dmso']
dates = ['22-03-28', '22-03-30', '22-11-23']

Filters = [(data['drug'].apply(lambda x : x in ['none', 'dmso', 'blebbistatin', 'latrunculinA'])),
           (data['date'].apply(lambda x : x in dates))]

# co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
co_order=['22-03-28 & dmso & 0.0', '22-03-30 & dmso & 0.0', '22-11-23 & dmso & 0.0']
# co_order = []

fig, ax = D1Plot(data, condCols=['date', 'drug', 'concentration'], Parameters=['ctFieldThickness'], Filters=Filters, 
           Boxplot=True, AvgPerCell=Avg, cellID='cellID', co_order=co_order,
           stats=True, statMethod='Mann-Whitney', box_pairs=[], 
           figSizeFactor = 0.5, markersizeFactor = 1, orientation = 'h',
           stressBoxPlot = False, bypassLog = False, 
           returnData = 0, returnCount = 0)

fig.suptitle(drugs[0])




drugs = ['blebbistatin']
dates = ['22-03-28', '22-03-30', '22-11-23']

Filters = [(data['drug'].apply(lambda x : x in ['none', 'dmso', 'blebbistatin', 'latrunculinA'])),
           (data['date'].apply(lambda x : x in dates))]

# co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
co_order=['22-03-28 & blebbistatin & 50.0', '22-03-30 & blebbistatin & 50.0']
# co_order = []

fig, ax = D1Plot(data, condCols=['date', 'drug', 'concentration'], Parameters=['ctFieldThickness'], Filters=Filters, 
           Boxplot=True, AvgPerCell=Avg, cellID='cellID', co_order=co_order,
           stats=True, statMethod='Mann-Whitney', box_pairs=[], 
           figSizeFactor = 0.5, markersizeFactor = 1, orientation = 'h',
           stressBoxPlot = False, bypassLog = False, 
           returnData = 0, returnCount = 0)

fig.suptitle(drugs[0])



drugs = ['latrunculinA']
dates = ['22-03-28', '22-03-30', '22-11-23']

Filters = [(data['drug'].apply(lambda x : x in ['none', 'dmso', 'blebbistatin', 'latrunculinA'])),
           (data['date'].apply(lambda x : x in dates))]

# co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
co_order=['22-03-28 & latrunculinA & 0.5', '22-03-30 & latrunculinA & 0.5', '22-11-23 & latrunculinA & 2.5']
# co_order = []

fig, ax = D1Plot(data, condCols=['date', 'drug', 'concentration'], Parameters=['ctFieldThickness'], Filters=Filters, 
           Boxplot=True, AvgPerCell=Avg, cellID='cellID', co_order=co_order,
           stats=True, statMethod='Mann-Whitney', box_pairs=[], 
           figSizeFactor = 0.5, markersizeFactor = 1, orientation = 'h',
           stressBoxPlot = False, bypassLog = False, 
           returnData = 0, returnCount = 0)

fig.suptitle(drugs[0])

plt.show()





# %%%% K, Date by date

Avg = False
data_main = MecaData_Drugs

fitType = 'stressRegion'
fitId = '250_75'
data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)
valCol = 'fit_K'
weightCol = 'fit_ciwK'

dates = ['22-03-28'] # ['22-03-28', '22-03-30']

Filters = [(data['fit_error'] == False), 
           (data['fit_K'] > 0),
           (data['fit_R2'] > 0.2),
           (data['drug'].apply(lambda x : x in ['none', 'dmso', 'blebbistatin', 'latrunculinA'])),
           (data['date'].apply(lambda x : x in dates))]

# co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
# co_order = []

fig, ax = D1Plot_K(data, condCols=['drug', 'concentration'], Filters=Filters, AvgPerCell=Avg, 
            parm = 'fit_K', weightParm = 'fit_ciwK',
            co_order=co_order, box_pairs=[], stats=True, statMethod='Mann-Whitney', 
            Boxplot=True, stressBoxPlot = 2, styleDict = styleDict1,
            figSizeFactor = 0.6, markersizeFactor = 1, scale = 'lin',
            returnData = 0, returnCount = 0)

fig.suptitle(dates[0])




dates = ['22-03-30'] # ['22-03-28', '22-03-30']


Filters = [(data['fit_error'] == False), 
           (data['fit_K'] > 0),
           (data['fit_R2'] > 0.2),
           (data['drug'].apply(lambda x : x in ['none', 'dmso', 'blebbistatin', 'latrunculinA'])),
           (data['date'].apply(lambda x : x in dates))]

# co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
# co_order = []

fig, ax = D1Plot_K(data, condCols=['drug', 'concentration'], Filters=Filters, AvgPerCell=Avg, 
            parm = 'fit_K', weightParm = 'fit_ciwK',
            co_order=co_order, box_pairs=[], stats=True, statMethod='Mann-Whitney', 
            Boxplot=True, stressBoxPlot = 2, styleDict = styleDict1,
            figSizeFactor = 0.6, markersizeFactor = 1, scale = 'lin',
            returnData = 0, returnCount = 0)

fig.suptitle(dates[0])



dates = ['22-11-23'] # ['22-03-28', '22-03-30']

Filters = [(data['fit_error'] == False), 
           (data['fit_K'] > 0),
           (data['fit_R2'] > 0.2),
           (data['drug'].apply(lambda x : x in ['none', 'dmso', 'blebbistatin', 'latrunculinA'])),
           (data['date'].apply(lambda x : x in dates))]

# co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
# co_order = []

fig, ax = D1Plot_K(data, condCols=['drug', 'concentration'], Filters=Filters, AvgPerCell=Avg, 
            parm = 'fit_K', weightParm = 'fit_ciwK',
            co_order=co_order, box_pairs=[], stats=True, statMethod='Mann-Whitney', 
            Boxplot=True, stressBoxPlot = 2, styleDict = styleDict1,
            figSizeFactor = 0.5, markersizeFactor = 1, scale = 'lin',
            returnData = 0, returnCount = 0)

fig.suptitle(dates[0])

plt.show()


# %%%% K, Drug by drug

Avg = True
data_main = MecaData_Drugs

fitType = 'stressRegion'
fitId = '300_75'
data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)
valCol = 'fit_K'
weightCol = 'fit_ciwK'

drugs = ['dmso']
dates = ['22-03-28', '22-03-30', '22-11-23']

Filters = [(data['fit_error'] == False), 
           (data['fit_K'] > 0),
           (data['fit_R2'] > 0.2),
           (data['drug'].apply(lambda x : x in drugs)),
           (data['date'].apply(lambda x : x in dates))]

# co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
co_order=['22-03-28 & dmso & 0.0', '22-03-30 & dmso & 0.0', '22-11-23 & dmso & 0.0']
# co_order = []

fig, ax = D1Plot_K(data, condCols=['date', 'drug', 'concentration'], Filters=Filters, AvgPerCell=Avg, 
            parm = 'fit_K', weightParm = 'fit_ciwK',
            co_order=co_order, box_pairs=[], stats=True, statMethod='Mann-Whitney', 
            Boxplot=True, stressBoxPlot = 2, styleDict = styleDict1,
            figSizeFactor = 0.8, markersizeFactor = 1, scale = 'lin',
            returnData = 0, returnCount = 0)

fig.suptitle(drugs[0])




drugs = ['blebbistatin']
dates = ['22-03-28', '22-03-30', '22-11-23']

Filters = [(data['fit_error'] == False), 
           (data['fit_K'] > 0),
           (data['fit_R2'] > 0.2),
           (data['drug'].apply(lambda x : x in drugs)),
           (data['date'].apply(lambda x : x in dates))]

# co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
co_order=['22-03-28 & blebbistatin & 50.0', '22-03-30 & blebbistatin & 50.0']
# co_order = []

fig, ax = D1Plot_K(data, condCols=['date', 'drug', 'concentration'], Filters=Filters, AvgPerCell=Avg, 
            parm = 'fit_K', weightParm = 'fit_ciwK',
            co_order=co_order, box_pairs=[], stats=True, statMethod='Mann-Whitney', 
            Boxplot=True, stressBoxPlot = 2, styleDict = styleDict1,
            figSizeFactor = 0.8, markersizeFactor = 1, scale = 'lin',
            returnData = 0, returnCount = 0)

fig.suptitle(drugs[0])



drugs = ['latrunculinA']
dates = ['22-03-28', '22-03-30', '22-11-23']

Filters = [(data['fit_error'] == False), 
           (data['fit_K'] > 0),
           (data['fit_R2'] > 0.2),
           (data['drug'].apply(lambda x : x in drugs)),
           (data['date'].apply(lambda x : x in dates))]

# co_order=['none', 'dmso & 0.0', 'blebbistatin & 50.0', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
co_order=['22-03-28 & latrunculinA & 0.5', '22-03-30 & latrunculinA & 0.5', '22-11-23 & latrunculinA & 2.5']
# co_order = []

fig, ax = D1Plot_K(data, condCols=['date', 'drug', 'concentration'], Filters=Filters, AvgPerCell=Avg, 
            parm = 'fit_K', weightParm = 'fit_ciwK',
            co_order=co_order, box_pairs=[], stats=True, statMethod='Mann-Whitney', 
            Boxplot=True, stressBoxPlot = 2, styleDict = styleDict1,
            figSizeFactor = 0.8, markersizeFactor = 1, scale = 'lin',
            returnData = 0, returnCount = 0)

fig.suptitle(drugs[0])

plt.show()



# %%%% Test_2

data_main = MecaData_Drugs

dfH0 = taka2.getMatchingFits(data_main, fitType = 'H0', output = 'df')

dates = ['22-03-28', '22-03-30'] # ['22-03-28', '22-03-30']

data_wH0 = taka2.getAllH0InTable(data_main)

Filters = [(data_main['validatedThickness'] == True),
           (data_main['bestH0'] <= 1200),
           (data_main['drug'].apply(lambda x : x in ['dmso'])),
           (data_main['date'].apply(lambda x : x in dates))]

fig, ax = plotAllH0(data_main, Filters = Filters, condCols = [], 
        co_order = [], box_pairs = [], AvgPerCell = True,
        stats = False, statMethod = 'Mann-Whitney', stressBoxPlot = 2)

plt.show()



# %%%% K(s) for drugs

# %%%%% Date 2

#### Making the Dataframe 

data = MecaData_Drugs
dates = ['22-03-30']

Filters = [(data['validatedThickness'] == True),
            (data['UI_Valid'] == True),
            (data['date'].apply(lambda x : x in dates)),
            (data['drug'] != 'none'),
            ]

out1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'drug', mode = '200_400', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig1, ax1, exportDf1, countDf1 = out1
ax1.set_ylim([0, 10])

out2 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'drug', mode = '200_400', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig2, ax2 = out2
ax2.set_ylim([0, 10])

# data_ff1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False)
# data_ff2 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = '150_550', scale = 'lin', printText = True)

# fig1.suptitle('3T3 vs. HoxB8 - stiffness')
# fig2.suptitle('3T3 vs. HoxB8 - stiffness')
plt.show()

# %%%%% Date 1

#### Making the Dataframe 

data = MecaData_Drugs
dates = ['22-11-23']

Filters = [(data['validatedThickness'] == True),
            (data['UI_Valid'] == True),
            (data['date'].apply(lambda x : x in dates)),
            ]

out1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'drug', mode = '200_400', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig1, ax1, exportDf1, countDf1 = out1

out2 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'drug', mode = '200_400', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig2, ax2 = out2

# data_ff1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False)
# data_ff2 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = '150_550', scale = 'lin', printText = True)

# fig1.suptitle('3T3 vs. HoxB8 - stiffness')
# fig2.suptitle('3T3 vs. HoxB8 - stiffness')
plt.show()



# %%%  Cell types

# %%%% Naive plots -> GOOD FOR 24/01

# %%%%% 1

data_main = MecaData_All
# dates = 

fitType = 'stressGaussian'
fitId = '400_100'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)
data_wFits = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

# %%%%% Stiffness

Filters = [(data_wFits['validatedThickness'] == True), 
           (data_wFits['surroundingThickness'] <= 800),
           # (data['fit_valid'] == True),
           (data_wFits['fit_K'] > 0),
            (data_wFits['fit_K'] < 2e4),
           (data_wFits['fit_R2'] > 0.1),
           (data_wFits['drug'].apply(lambda x : x in ['none', 'dmso'])),
           (data_wFits['cell subtype'].apply(lambda x : x in ['mouse-primary', 'ctrl', 'aSFL-A11', 'aSFL-LG+++'])),
           (data_wFits['date'].apply(lambda x : (x.startswith('18') or x.startswith('22'))))]

co_order=[] 
box_pairs=[]

fig, ax = D1Plot_K(data_wFits, condCols=['cell type'], Filters=Filters, AvgPerCell = True, 
                    parm = 'fit_K', weightParm = 'fit_ciwK',
                    co_order=co_order, box_pairs=box_pairs, stats=True, statMethod='Mann-Whitney', 
                    Boxplot=True, stressBoxPlot = 2, styleDict = styleDict1,
                    figSizeFactor = 2, markersizeFactor = 1, scale = 'lin',
                    returnData = 0, returnCount = 0)

# fig, ax = D1Plot(data_wFits, condCols=['cell type'], Parameters=[thicknessType, 'fit_K'], Filters=Filters, 
#                 Boxplot=True, cellID='cellID', co_order=[], 
#                 AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
#                 figSizeFactor = 1, markersizeFactor=1, orientation = 'h')

# rD = {'none' : 'Control\n(nothing)', 'dmso' : 'Control\n(DMSO)', 
#       'blebbistatin' : 'Blebbistatin', 'latrunculinA' : 'LatA', 
#       'bestH0' : 'Thickness (nm)', 'K' + fit : 'fit_KTangeantial Modulus (Pa)'}

# renameAxes(ax, rD)
renameAxes(ax, renameDict1)

# ax[0].set_ylim([0, 1000])
# ax[0].legend(loc = 'upper right', fontsize = 8)
# ax[1].legend(loc = 'upper right', fontsize = 8)
fig.suptitle('Cell types\nPreliminary data')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_H&Echad_simple', figSubDir = figSubDir)
plt.show()


# %%%%% Thickness
data_main = MecaData_All
data = data_main
thicknessType = 'surroundingThickness' # 'bestH0', 'surroundingThickness', 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['bestH0'] <= 1000),
           (data['drug'].apply(lambda x : x in ['none', 'dmso'])),
           (data['cell subtype'].apply(lambda x : x in ['mouse-primary', 'ctrl', 'aSFL-A11'])), #, 'aSFL-LG+++'])),
           (data['date'].apply(lambda x : (x.startswith('18') or x.startswith('22'))))]

co_order=['3T3', 'DC', 'HoxB8-Macro']

fig, ax = D1Plot(data, condCols=['cell type'], Parameters=[thicknessType], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], stressBoxPlot=2,
                figSizeFactor = 1, markersizeFactor=1, orientation = 'h')

# rD = {'none' : 'Control\n(nothing)', 'dmso' : 'Control\n(DMSO)', 
#       'blebbistatin' : 'Blebbistatin', 'latrunculinA' : 'LatA', 
#       'bestH0' : 'Thickness (nm)', 'K' + fit : 'fit_KTangeantial Modulus (Pa)'}

# renameAxes(ax, rD)
renameAxes(ax, renameDict1)

# ax[0].set_ylim([0, 1000])
# ax[0].legend(loc = 'upper right', fontsize = 8)
# ax[1].legend(loc = 'upper right', fontsize = 8)
fig.suptitle('Cell types')
# ufun.archiveFig(fig, ax, name='3T3aSFL_Jan21_drug_H&Echad_simple', figSubDir = figSubDir)
plt.show()

# %%%%% K(s) - 1

data = data_main
Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['drug'].apply(lambda x : x in ['none', 'dmso'])),
           (data['substrate'].apply(lambda x : x in ['BSA coated glass', '20um fibronectin discs'])),
           (data['cell subtype'].apply(lambda x : x in ['mouse-primary', 'ctrl', 'aSFL-A11'])),
           (data['date'].apply(lambda x : (x.startswith('18') or x.startswith('22'))))]

out1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters,
                                condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

fig1, ax1, exportDf1, countDf1 = out1
ax1.set_ylim([0, 14])

out2 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'cell type', mode = '250_500', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)
fig2, ax2, exportDf2, countDf2 = out2
ax2.set_ylim([0, 14])

# data_ff1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False)
# data_ff2 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = '150_550', scale = 'lin', printText = True)

# fig1.suptitle('3T3 vs. HoxB8 - stiffness')
# fig2.suptitle('3T3 vs. HoxB8 - stiffness')
plt.show()


# %%%%% K(s) - 2

data = data_main
Filters = [(data['validatedThickness'] == True), 
           (data['surroundingThickness'] <= 800),
           (data['drug'].apply(lambda x : x in ['none', 'dmso'])),
           (data['substrate'].apply(lambda x : x in ['BSA coated glass', '20um fibronectin discs'])),
           (data['cell subtype'].apply(lambda x : x in ['mouse-primary', 'ctrl', 'aSFL-A11'])),
           (data['date'].apply(lambda x : (x.startswith('18') or x.startswith('22'))))]

out1 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False,
                                Sinf = 0, Ssup = 1000,
                                returnData = 1, returnCount = 1)
fig1, ax1 = out1
ax1.set_ylim([0, 14])

out2 = plotPopKS_V2(data, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'cell type', mode = '250_500', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)
fig2, ax2 = out2
ax2.set_ylim([0, 14])

# data_ff1 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False)
# data_ff2 = plotPopKS(data, fitType = 'stressGaussian', fitWidth=50, Filters = Filters, 
#                                 condCol = 'cell type', mode = '150_550', scale = 'lin', printText = True)

# fig1.suptitle('3T3 vs. HoxB8 - stiffness')
# fig2.suptitle('3T3 vs. HoxB8 - stiffness')
plt.show()

