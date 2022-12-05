# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:01:08 2022

@author: JosephVermeil
"""


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

#### Display

# df1 = ufun.getExperimentalConditions().head()
# df2 = taka.getGlobalTable_ctField().head()
# df3 = taka.getGlobalTable_meca().head()
# df4 = taka.getFluoData().head()


#### GlobalTable_ctField

# GlobalTable_ctField = taka.getGlobalTable(kind = 'ctField')
GlobalTable_ctField = taka.getMergedTable('Global_CtFieldData')

#### GlobalTable_ctField_Py

# GlobalTable_ctField_Py = taka.getGlobalTable(kind = 'ctField_py')
GlobalTable_ctField_Py = taka.getMergedTable('Global_CtFieldData_Py')

#### GlobalTable_meca

# GlobalTable_meca = taka.getGlobalTable(kind = 'meca_matlab')
GlobalTable_meca = taka.getMergedTable('Global_MecaData')

#### GlobalTable_meca_Py

# GlobalTable_meca_Py = taka.getGlobalTable(kind = 'meca_py')
GlobalTable_meca_Py = taka.getMergedTable('Global_MecaData_Py')

#### GlobalTable_meca_Py2

# GlobalTable_meca_Py2 = taka.getGlobalTable(kind = 'meca_py2')
GlobalTable_meca_Py2 = taka.getMergedTable('Global_MecaData_Py2', mergeUMS = True)


#### Global_MecaData_NonLin_Py

# GlobalTable_meca_nonLin = taka.getGlobalTable(kind = 'meca_nonLin')
GlobalTable_meca_nonLin = taka.getMergedTable('Global_MecaData_NonLin2_Py')


#### Global_MecaData_MCA

# GlobalTable_meca_MCA = taka.getGlobalTable(kind = 'meca_MCA')
GlobalTable_meca_MCA = taka.getMergedTable('Global_MecaData_MCA2')

#### Global_MecaData_MCA3

# GlobalTable_meca_MCA = taka.getGlobalTable(kind = 'meca_MCA')
GlobalTable_meca_MCA3 = taka.getMergedTable('Global_MecaData_MCA3', mergeUMS = True)

#### Global_MecaData_MCA123

# GlobalTable_meca_MCA = taka.getGlobalTable(kind = 'meca_MCA')
GlobalTable_meca_MCA123 = taka.getMergedTable('Global_MecaData_MCA123', mergeUMS = True, mergeFluo = True)
GlobalTable_meca_MCA123['round'] = GlobalTable_meca_MCA123['tags'].apply(lambda x : x.split('. ')[0])
# print(GlobalTable_meca_MCA123['round'])

#### Global_MecaData_HoxB8

# GlobalTable_meca_MCA = taka.getGlobalTable(kind = 'meca_MCA')
GlobalTable_meca_HoxB8 = taka.getMergedTable('Global_MecaData_HoxB8_2', mergeUMS = True)
GlobalTable_meca_HoxB8_new = taka.getMergedTable('Global_MecaData_HoxB8_3', mergeUMS = True)

#### Global_MecaData_MCA-HoxB8_2

# GlobalTable_meca_MCA = taka.getGlobalTable(kind = 'meca_MCA')
GlobalTable_meca_MCAHoxB8 = taka.getMergedTable('Global_MecaData_MCA-HoxB8_2', mergeUMS = True)

# %%%

tic = time.time()
data_main = GlobalTable_meca_HoxB8_new
data = taka2.getFitsInTable(data_main, fitType = 'stressGaussian', 
                            filter_fitID = '200_')
toc = time.time()

print(toc-tic)


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
               'fit_K':'Tangeantial Modulus (Pa)',
               
               # HoxB8
               'ctrl':'Control',
               'tko':'TKO',
               'bare glass & ctrl':'Control on bare glass',
               'bare glass & tko':'TKO on bare glass',
               '20um fibronectin discs & ctrl':'Control on fibronectin',
               '20um fibronectin discs & tko':'TKO on fibronectin',
               }

styleDict1 =  {'DictyDB_M270':{'color':'lightskyblue','marker':'o'}, 
               'DictyDB_M450':{'color': 'maroon','marker':'o'},
               'M270':{'color':'lightskyblue','marker':'o'}, 
               'M450':{'color': 'maroon','marker':'o'},
               
               # HoxB8
               'ctrl':{'color': gs.colorList40[10],'marker':'^'},
               'tko':{'color': gs.colorList40[30],'marker':'^'},
               'bare glass & ctrl':{'color': gs.colorList40[10],'marker':'^'},
               'bare glass & tko':{'color': gs.colorList40[30],'marker':'^'},
               '20um fibronectin discs & ctrl':{'color': gs.colorList40[12],'marker':'o'},
               '20um fibronectin discs & tko':{'color': gs.colorList40[32],'marker':'o'},
               
               # Comparison
               '3T3':{'color': gs.colorList40[31],'marker':'o'},
               'HoxB8-Macro':{'color': gs.colorList40[32],'marker':'o'},
               }




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
                
        p = getStyleLists_Sns(co_order, styleDict1) # styleDict_MCA3
        
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

        if (not bypassLog) and (('EChadwick' in Parameters[k]) or ('K_' in Parameters[k]) or ('fit_K' in Parameters[k])):
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
    
    
    
    
def D1Plot_wNormalize(data, fig = None, ax = None, CondCol=[], Parameters=[], Filters=[], 
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
        
    # print(data_filtered.shape)
    small_df = data_filtered[[CondCol, 'cellID', 'compNum', 'date', 'manipID']]
    
    if AvgPerCell:
        group = data_filtered.groupby(cellID)
        dictAggMean = getDictAggMean(data_filtered)
#         dictAggMean['EChadwick'] = 'median'
        data_filtered = group.agg(dictAggMean)
        
    data_filtered.sort_values(CondCol, axis=0, ascending=True, inplace=True)
    
    NPlots = len(Parameters)
    Conditions = list(data_filtered[CondCol].unique())     
    
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
                sns.boxplot(x=CondCol, y=Parm, data=data_filtered, ax=ax[k], 
                            width = 0.5, showfliers = False, order= co_order, 
                            medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 4},
                            boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
        #                   boxprops={"color": color, "linewidth": 0.5},
                            whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                            capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4})
                            # scaley = scaley)
            else:
                sns.boxplot(x=CondCol, y=Parm, data=data_filtered, ax=ax[k], 
                            width = 0.5, showfliers = False, order= co_order, 
                            medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 1},
                            boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 1},
        #                   boxprops={"color": color, "linewidth": 0.5},
                            whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 1},
                            capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 1})
                            # scaley = scaley)
            
            # data_filtered.boxplot(column=Parm, by = CondCol, ax=ax[k],showfliers = False) # linewidth = 2, width = 0.5

        if stats:
            if len(box_pairs) == 0:
                box_pairs = makeBoxPairs(co_order)
            addStat_df(ax[k], data_filtered, box_pairs, Parm, CondCol, test = statMethod)
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
        
    
    if returnCount > 0:
        groupByCell = small_df.groupby(cellID)
        d_agg = {'compNum':'count', CondCol:'first', 'date':'first', 'manipID':'first'}
        df_CountByCell = groupByCell.agg(d_agg).rename(columns={'compNum':'compCount'})

        groupByCond = df_CountByCell.reset_index().groupby(CondCol)
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
    
    current_color_list = getStyleLists(Conditions, styleDict1).as_hex()
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
                
        print(co_order)
        try:
            colorList, markerList = getStyleLists(co_order, styleDict1)
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





def plotPopKS(data, width, centers, Filters = [], condCol = '', 
              mode = 'wholeCurve', scale = 'lin', printText = True):

    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for k in range(0, len(Filters)):
        globalFilter = globalFilter & Filters[k]
    
    data_f = data[globalFilter]

    # regionFitsNames = [str(fitMin[ii]) + '<s<' + str(fitMax[ii]) for ii in range(len(fitMin))]
    regionFitsNames = ['S={:.0f}+/-{:.0f}'.format(centers[ii], width//2) for ii in range(len(centers))]

    fig, ax = plt.subplots(1,1, figsize = (9,6))

    ListDfWhole = []

    conditions = np.array(data_f[condCol].unique())

    valStr = 'KChadwick_'
    weightStr = 'K_Weight_'
    
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
    
    
    #### Whole curve / local part
    
    if mode == 'wholeCurve':
        data2_f = data_f
        ax.set_xlim([0, np.max(centers)+50])  
        
    else:
        bounds = mode.split('_')
        Sinf, Ssup = int(bounds[0]), int(bounds[1])
        extraFilters = [data_f['minStress'] <= Sinf, data_f['maxStress'] >= Ssup] # >= 800
        centers = centers[(centers>=(Sinf)) & (centers<=Ssup)] # <800
    
        data2_f = data_f
        globalExtraFilter = extraFilters[0]
        for k in range(1, len(extraFilters)):
            globalExtraFilter = globalExtraFilter & extraFilters[k]
        data2_f = data2_f[globalExtraFilter]
            
        ax.set_xlim([np.min(centers)-50, np.max(centers)+50])        
    
    for co in conditions:
        Kavg = []
        Kstd = []
        D10 = []
        D90 = []
        N = []
        
        data_ff = data2_f[data2_f[condCol] == co]
        color = styleDict1[co]['color']
        
        for ii in range(len(centers)):
            S = centers[ii]
            rFN = 'S={:.0f}+/-{:.0f}'.format(S, width//2)
            variable = valStr+rFN
            weight = weightStr+rFN
            
            x = data_ff[variable].apply(nan2zero).values
            w = data_ff[weight].apply(nan2zero).values
            
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
        
        d_val = {'S' : centers, 'Kavg' : Kavg, 'Kstd' : Kstd, 'D10' : D10, 'D90' : D90, 'N' : N}
        
        if scale == 'lin':
            if co == conditions[0]:
                texty = Kavg + 1500
            else:
                texty = texty + 300
            
            ax.set_yscale('linear')
            # ax.set_ylim([0,1.2e1])
            # ax.set_ylim([0,0.8e1])
                
        elif scale == 'log':
            if co == conditions[0]:
                texty = Kavg**0.95
            else:
                texty = texty**0.98
            
            ax.set_yscale('log')
            # ax.set_ylim([0.5,2e1])
            
        # Weighted means -- Weighted ste 95% as error
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
        
        df_val = pd.DataFrame(d_val)
        ListDfWhole.append(df_val)
        
    fig.suptitle('K(s)'+' - fits width: {:.0f}Pa'.format(width))  
    return(fig, ax)
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
            axes[i].set_xticklabels(newXticksList, fontsize=16)
        
        # set xlabel
        xlabel = axes[i].get_xlabel()
        newXlabel = rD.get(xlabel, xlabel)
        axes[i].set_xlabel(newXlabel, fontsize=16)
        # set ylabel
        ylabel = axes[i].get_ylabel()
        newYlabel = rD.get(ylabel, ylabel)
        axes[i].set_ylabel(newYlabel, fontsize=16)
        

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
        ax.plot([bp[0], bp[1]], [currentHeight, currentHeight], 'k-', lw = 1)
        XposText = (dictXTicks[bp[0]]+dictXTicks[bp[1]])/2
        if scale == 'log':
            power = 0.01* (text=='ns') + 0.000 * (text!='ns')
            YposText = currentHeight*(refHeight**power)
        else:
            factor = 0.02 * (text=='ns') + 0.000 * (text!='ns')
            YposText = currentHeight + factor*refHeight
        ax.text(XposText, YposText, text, ha = 'center', color = 'k', size = 14)
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


# %%% HoxB8 -- (july 2022)

# %%%% Four conditions first plot

data = GlobalTable_meca_HoxB8
dates = ['22-05-03', '22-05-04', '22-05-05']
StressRegion = '_S=300+/-100'
srs = StressRegion.split('+/-')[0]+'-'+StressRegion.split('+/-')[1]

Filters = [(data['validatedFit'+StressRegion] == True), 
           (data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == 'HoxB8-Macro'), 
           (data['bead type'] == 'M450'),
           (data['bestH0'] <= 800),
           (data['date'].apply(lambda x : x in dates))]



# co_order = makeOrder(['ctrl','tko'],['naked glass','20um fibronectin discs'])
co_order = makeOrder(['bare glass','20um fibronectin discs'],['ctrl','tko'])

fig, ax, dfcount = D1Plot(data, CondCol=['substrate','cell subtype'], Parameters=['bestH0', 'KChadwick'+StressRegion],Filters=Filters,
                          AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=[], figSizeFactor = 0.8, markersizeFactor=0.8, orientation = 'v', returnCount = 1,
                          stressBoxPlot=True)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 summary plot')

ufun.archiveFig(fig, name=('HoxB8_H0 & K' + srs + '_allComps'), figDir = 'HoxB8project', dpi = 100)


fig, ax, dfcount = D1Plot(data, CondCol=['substrate','cell subtype'], Parameters=['bestH0', 'KChadwick'+StressRegion],Filters=Filters,
                          AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=[], figSizeFactor = 0.8, markersizeFactor=0.8, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 summary plot')

ufun.archiveFig(fig, name=('HoxB8_H0 & K' + srs + '_cellAvg'), figDir = 'HoxB8project', dpi = 100)

plt.show()

# %%%% Four conditions only H0

data = GlobalTable_meca_HoxB8
dates = ['22-05-03', '22-05-04', '22-05-05']

thicknessType = 'bestH0' # 'bestH0', 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
            (data['UI_Valid'] == True),
            (data['cell type'] == 'HoxB8-Macro'), 
            (data['bead type'] == 'M450'),
            (data[thicknessType] <= 800),
            (data['date'].apply(lambda x : x in dates))]



# co_order = makeOrder(['ctrl','tko'],['naked glass','20um fibronectin discs'])
co_order = makeOrder(['bare glass','20um fibronectin discs'],['ctrl','tko'])

fig, ax, dfcount = D1Plot(data, CondCol=['substrate','cell subtype'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=[], figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1,
                 stressBoxPlot=True)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 H0 plot' + '\nall compressions')

ufun.archiveFig(fig, name=('HoxB8_'+thicknessType+' only' + '_allComps'), figDir = 'HoxB8project', dpi = 100)


fig, ax, dfcount = D1Plot(data, CondCol=['substrate','cell subtype'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=[], figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 H0 plot' + '\naverage per cell')

ufun.archiveFig(fig, name=('HoxB8_'+thicknessType+' only' + '_cellAvg'), figDir = 'HoxB8project', dpi = 100)


plt.show()


# %%%% Fibro only H0

data = GlobalTable_meca_HoxB8
dates = ['22-05-03', '22-05-04', '22-05-05']

thicknessType = 'surroundingThickness' # 'bestH0', 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
            (data['UI_Valid'] == True),
            (data['substrate'] == '20um fibronectin discs'),
            (data['cell type'] == 'HoxB8-Macro'), 
            (data['bead type'] == 'M450'),
            (data[thicknessType] <= 800),
            (data['date'].apply(lambda x : x in dates))]



# co_order = makeOrder(['ctrl','tko'],['naked glass','20um fibronectin discs'])
# co_order = makeOrder(['bare glass','20um fibronectin discs'],['ctrl','tko'])
co_order = makeOrder(['20um fibronectin discs'],['ctrl','tko'])

# fig, ax, dfcount = D1Plot(data, CondCol=['substrate','cell subtype'], Parameters=[thicknessType],Filters=Filters,
#                  AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
#                  box_pairs=[], figSizeFactor = 0.5, markersizeFactor = 1.0, orientation = 'v', returnCount = 1,
#                  stressBoxPlot=True)

# renameAxes(ax,renameDict1)
# fig.suptitle('HoxB8 H0 plot' + '\nall compressions')

# ufun.archiveFig(fig, name=('HoxB8_'+thicknessType+' only' + '_allComps'), figDir = 'HoxB8project', dpi = 100)


fig, ax, dfcount = D1Plot(data, CondCol=['substrate','cell subtype'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=[], figSizeFactor = 0.5, markersizeFactor = 1.5, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 thickness')

# ufun.archiveFig(fig, name=('HoxB8_'+thicknessType+' only' + '_cellAvg'), figDir = 'HoxB8project', dpi = 100)


plt.show()


# %%%% Four conditions only surroundingThickness

data = GlobalTable_meca_HoxB8
dates = ['22-05-03', '22-05-04', '22-05-05']

Filters = [(data['validatedThickness'] == True), 
            (data['UI_Valid'] == True),
            (data['cell type'] == 'HoxB8-Macro'), 
            (data['bead type'] == 'M450'),
            (data['bestH0'] <= 800),
            (data['date'].apply(lambda x : x in dates))]



# co_order = makeOrder(['ctrl','tko'],['naked glass','20um fibronectin discs'])
co_order = makeOrder(['bare glass','20um fibronectin discs'],['ctrl','tko'])

fig, ax, dfcount, dfcountcells = D1Plot(data, CondCol=['substrate','cell subtype'], Parameters=['surroundingThickness'],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=[], figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 2,
                 stressBoxPlot=True)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 surroundingThickness plot' + '\nall compressions')

ufun.archiveFig(fig, name=('HoxB8_surH only' + '_allComps'), figDir = 'HoxB8project', dpi = 100)


fig, ax, dfcount, dfcountcells = D1Plot(data, CondCol=['substrate','cell subtype'], Parameters=['surroundingThickness'],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=[], figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 2)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 surroundingThickness plot' + '\naverage per cell')

ufun.archiveFig(fig, name=('HoxB8_surH only' + '_cellAvg'), figDir = 'HoxB8project', dpi = 100)


plt.show()


# %%%% Four conditions only ctFieldThickness

data = GlobalTable_meca_HoxB8
dates = ['22-05-03', '22-05-04', '22-05-05']

Filters = [(data['validatedThickness'] == True), 
            (data['UI_Valid'] == True),
            (data['cell type'] == 'HoxB8-Macro'), 
            (data['bead type'] == 'M450'),
            (data['bestH0'] <= 800),
            (data['date'].apply(lambda x : x in dates))]



# co_order = makeOrder(['ctrl','tko'],['naked glass','20um fibronectin discs'])
co_order = makeOrder(['bare glass','20um fibronectin discs'],['ctrl','tko'])

fig, ax, dfcount, dfcountcells = D1Plot(data, CondCol=['substrate','cell subtype'], Parameters=['ctFieldThickness'],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=[], figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 2)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 ctFieldThickness plot' + '\naverage per cell')

ufun.archiveFig(fig, name=('HoxB8_ctFieldH only' + '_cellAvg'), figDir = 'HoxB8project', dpi = 100)


plt.show()


# %%%% Four conditions only H0 + by dates

data = GlobalTable_meca_HoxB8
dates = ['22-05-03', '22-05-04', '22-05-05']

# 1

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == 'HoxB8-Macro'), 
           (data['bead type'] == 'M450'),
           (data['substrate'] == 'bare glass'),
           (data['bestH0'] <= 800),
           (data['date'].apply(lambda x : x in dates))]

# co_order = makeOrder(['ctrl','tko'],['naked glass','20um fibronectin discs'])
co_order = makeOrder(['22-05-03', '22-05-04', '22-05-05'],['ctrl','tko'])

box_pairs = [['22-05-03 & ctrl', '22-05-03 & tko'], ['22-05-04 & ctrl', '22-05-04 & tko'], ['22-05-05 & ctrl', '22-05-05 & tko']]

fig, ax, dfcount = D1Plot(data, CondCol=['date','cell subtype'], Parameters=['bestH0'],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 on bare glass, data by date')

ufun.archiveFig(fig, name=('HoxB8_H0_bareGlass_dates_allComps'), figDir = 'HoxB8project', dpi = 100)

fig, ax, dfcount = D1Plot(data, CondCol=['date','cell subtype'], Parameters=['bestH0'],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 on bare glass, data by date')

ufun.archiveFig(fig, name=('HoxB8_H0_bareGlass_dates_cellAvg'), figDir = 'HoxB8project', dpi = 100)

# 2

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == 'HoxB8-Macro'), 
           (data['bead type'] == 'M450'),
           (data['substrate'] == '20um fibronectin discs'),
           (data['bestH0'] <= 800),
           (data['date'].apply(lambda x : x in dates))]

# co_order = makeOrder(['ctrl','tko'],['naked glass','20um fibronectin discs'])
co_order = makeOrder(['22-05-03', '22-05-04'],['ctrl','tko'])

box_pairs = [['22-05-03 & ctrl', '22-05-03 & tko'], ['22-05-04 & ctrl', '22-05-04 & tko']]

fig, ax, dfcount = D1Plot(data, CondCol=['date','cell subtype'], Parameters=['bestH0'],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 on 20um fibronectin discs, data by date')

ufun.archiveFig(fig, name=('HoxB8_H0_fibro_dates_allComps'), figDir = 'HoxB8project', dpi = 100)

fig, ax, dfcount = D1Plot(data, CondCol=['date','cell subtype'], Parameters=['bestH0'],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 on 20um fibronectin discs, data by date')

ufun.archiveFig(fig, name=('HoxB8_H0_fibro_dates_cellAvg'), figDir = 'HoxB8project', dpi = 100)

plt.show()


# %%%% Four conditions only H0 + by substrate & date

data = GlobalTable_meca_HoxB8
dates = ['22-05-03', '22-05-04']

# 1

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == 'HoxB8-Macro'), 
           (data['bead type'] == 'M450'),
           (data['cell subtype'] == 'ctrl'),
           (data['bestH0'] <= 800),
           (data['date'].apply(lambda x : x in dates))]

# co_order = makeOrder(['ctrl','tko'],['naked glass','20um fibronectin discs'])
co_order = makeOrder(['22-05-03', '22-05-04'],['bare glass','20um fibronectin discs'])

box_pairs = [['22-05-03 & bare glass', '22-05-03 & 20um fibronectin discs'], 
             ['22-05-04 & bare glass', '22-05-04 & 20um fibronectin discs']]

fig, ax, dfcount = D1Plot(data, CondCol=['date','substrate'], Parameters=['bestH0'],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('Ctrl HoxB8, data by date')

ufun.archiveFig(fig, name=('z_HoxB8_H0_CTRL_dates_allComps'), figDir = 'HoxB8project', dpi = 100)

fig, ax, dfcount = D1Plot(data, CondCol=['date','substrate'], Parameters=['bestH0'],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('Ctrl HoxB8, data by date')

ufun.archiveFig(fig, name=('z_HoxB8_H0_CTRL_dates_cellAvg'), figDir = 'HoxB8project', dpi = 100)

# 2

Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == 'HoxB8-Macro'), 
           (data['bead type'] == 'M450'),
           (data['cell subtype'] == 'tko'),
           (data['bestH0'] <= 800),
           (data['date'].apply(lambda x : x in dates))]

# co_order = makeOrder(['ctrl','tko'],['naked glass','20um fibronectin discs'])
co_order = makeOrder(['22-05-03', '22-05-04'],['bare glass','20um fibronectin discs'])

box_pairs = [['22-05-03 & bare glass', '22-05-03 & 20um fibronectin discs'], 
             ['22-05-04 & bare glass', '22-05-04 & 20um fibronectin discs']]

fig, ax, dfcount = D1Plot(data, CondCol=['date','substrate'], Parameters=['bestH0'],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('Tko HoxB8, data by date')

ufun.archiveFig(fig, name=('z_HoxB8_H0_TKO_dates_allComps'), figDir = 'HoxB8project', dpi = 100)

fig, ax, dfcount = D1Plot(data, CondCol=['date','substrate'], Parameters=['bestH0'],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('Tko HoxB8, data by date')

ufun.archiveFig(fig, name=('z_HoxB8_H0_TKO_dates_cellAvg'), figDir = 'HoxB8project', dpi = 100)

plt.show()


# %%%% Four conditions only KChadwick + by dates

data = GlobalTable_meca_HoxB8
dates = ['22-05-03', '22-05-04', '22-05-05']
StressRegion = '_S=600+/-75'
srs = StressRegion.split('+/-')[0]+'-'+StressRegion.split('+/-')[1]

# 1

Filters = [(data['validatedFit'+StressRegion] == True), 
           (data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == 'HoxB8-Macro'), 
           (data['bead type'] == 'M450'),
           (data['substrate'] == 'bare glass'),
           (data['bestH0'] <= 900),
           (data['date'].apply(lambda x : x in dates))]

# co_order = makeOrder(['ctrl','tko'],['naked glass','20um fibronectin discs'])
co_order = makeOrder(['22-05-03', '22-05-04', '22-05-05'],['ctrl','tko'])

box_pairs = [['22-05-03 & ctrl', '22-05-03 & tko'], ['22-05-04 & ctrl', '22-05-04 & tko'], ['22-05-05 & ctrl', '22-05-05 & tko']]

fig, ax, dfcount = D1Plot(data, CondCol=['date','cell subtype'], Parameters=['KChadwick'+StressRegion],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 on bare glass, data by date')

ufun.archiveFig(fig, name=('HoxB8_K' + srs + '_bareGlass_dates_allComps'), figDir = 'HoxB8project', dpi = 100)

fig, ax, dfcount = D1Plot(data, CondCol=['date','cell subtype'], Parameters=['KChadwick'+StressRegion],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 on bare glass, data by date')

ufun.archiveFig(fig, name=('HoxB8_K' + srs + '_bareGlass_dates_cellAvg'), figDir = 'HoxB8project', dpi = 100)

# 2

Filters = [(data['validatedFit'+StressRegion] == True), 
           (data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['cell type'] == 'HoxB8-Macro'), 
           (data['bead type'] == 'M450'),
           (data['substrate'] == '20um fibronectin discs'),
           (data['bestH0'] <= 900),
           (data['date'].apply(lambda x : x in dates))]

# co_order = makeOrder(['ctrl','tko'],['naked glass','20um fibronectin discs'])
co_order = makeOrder(['22-05-03', '22-05-04'],['ctrl','tko'])

box_pairs = [['22-05-03 & ctrl', '22-05-03 & tko'], ['22-05-04 & ctrl', '22-05-04 & tko']]

fig, ax, dfcount = D1Plot(data, CondCol=['date','cell subtype'], Parameters=['KChadwick'+StressRegion],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 on 20um fibronectin discs, data by date')

ufun.archiveFig(fig, name=('HoxB8_K' + srs + '_fibro_dates_allComps'), figDir = 'HoxB8project', dpi = 100)


fig, ax, dfcount = D1Plot(data, CondCol=['date','cell subtype'], Parameters=['KChadwick'+StressRegion],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor = 1.0, orientation = 'v', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 on 20um fibronectin discs, data by date')

ufun.archiveFig(fig, name=('HoxB8_K' + srs + '_fibro_dates_cellAvg'), figDir = 'HoxB8project', dpi = 100)

plt.show()


# %%%% Multiple boxplot K(s)

data = GlobalTable_meca_HoxB8
dates = ['22-05-03', '22-05-04', '22-05-05']



# fig, ax, dfcount = D1Plot(data, CondCol=['substrate','cell subtype'], Parameters=['bestH0', 'KChadwick'+StressRegion],Filters=Filters,
#                           AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
#                           box_pairs=[], figSizeFactor = 0.8, markersizeFactor=0.8, orientation = 'v', returnCount = 1)

CondCol=['substrate', 'cell subtype']
co_order = makeOrder(['bare glass','20um fibronectin discs'],['ctrl','tko'])
NCondCol = len(CondCol)
NCond = len(co_order)


listeS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# listS = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
width = 150

fig, axes = plt.subplots(NCond, len(listeS), figsize = (18,12))

for ii in range(NCond):
    co = co_order[ii]
    co_split = co.split(' & ')
        
    
    for kk in range(len(listeS)):
        S = listeS[kk]
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
    
        fig, ax = D1Plot(data, fig=fig, ax=axes[ii,kk], CondCol=['substrate','cell subtype'], Parameters=['KChadwick_' + interval], 
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

ufun.archiveFig(fig, name='HoxB8_K(s)_MultiBoxPlots_to1000', 
                figDir = 'HoxB8project', dpi = 100)

# %%%% Multiple 2D plot K(h)

data = GlobalTable_meca_HoxB8
dates = ['22-05-03', '22-05-04', '22-05-05']



# fig, ax, dfcount = D1Plot(data, CondCol=['substrate','cell subtype'], Parameters=['bestH0', 'KChadwick'+StressRegion],Filters=Filters,
#                           AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
#                           box_pairs=[], figSizeFactor = 0.8, markersizeFactor=0.8, orientation = 'v', returnCount = 1)

CondCol=['substrate', 'cell subtype']
co_order = makeOrder(['bare glass','20um fibronectin discs'],['ctrl','tko'])
NCondCol = len(CondCol)
NCond = len(co_order)


listS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# listS = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
width = 150

fig, axes = plt.subplots(NCond, len(listeS), figsize = (18,12))

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


        D2Plot_wFit(data, fig=fig, ax=axes[ii,kk], 
                    XCol='bestH0', YCol='KChadwick_'+interval, 
                    CondCol = ['substrate','cell subtype'], co_order = copy(co_order),
                    Filters=Filters, cellID = 'cellID', AvgPerCell=False, 
                    xscale = 'linear', yscale = 'log', 
                    modelFit=True, modelType='y=k*x^a', writeEqn = False)
        
    #     # individual figs
    #     ax.set_xlim([8e1, 700])
    #     ax.set_ylim([1e2, 5e4])
    #     # renameAxes(ax,{'bestH0':'best H0 (nm)', 'doxycyclin':'iMC'})
    #     fig.set_size_inches(6,4)
    
        # axes[ii,kk].legend(loc = 'upper right', fontsize = 8)
        axes[ii,kk].legend().set_visible(False)
        label = axes[ii,kk].get_ylabel()
        # axes[ii,kk].set_title(label.split('_')[-1] + 'Pa', fontsize = 10)
        axes[ii,kk].set_ylim([4e2, 2e4])
        axes[ii,kk].set_xlim([0, 900])
        axes[ii,kk].set_xticks([i for i in range(100, 1000, 200)])
        
        if ii == 0:
            axes[ii,kk].set_title(interval + ' Pa', fontsize = 10)
        if ii == len(co_order)-1:
            axes[ii,kk].tick_params(axis='x', labelrotation = 10, labelsize = 10)
        else:
            axes[ii,kk].set_xlabel(None)
            axes[ii,kk].set_xticklabels([])
            
        if kk == 0:
            # axes[ii,kk].set_ylabel(label.split('_')[0] + ' (Pa)')
            axes[ii,kk].set_ylabel(co + '\n' + label.split('_')[0] + ' (Pa)', 
                                   fontsize = 11, color = styleDict1[co]['color'])
        else:
            axes[ii,kk].set_ylabel(None)
            axes[ii,kk].set_yticklabels([])


fig.tight_layout()
fig.suptitle('')
plt.show()

ufun.archiveFig(fig, name='HoxB8_K(h)_Multi2DPlots', 
                figDir = 'HoxB8project', dpi = 100)







# %%%% K(s) for HoxB8

#### Making the Dataframe 

data = GlobalTable_meca_HoxB8

data['HoxB8_Co'] = data['substrate'].values + \
                   np.array([' & ' for i in range(data.shape[0])]) + \
                   data['cell subtype'].values

dates = ['22-05-03', '22-05-04', '22-05-05'] #['21-12-08', '22-01-12'] ['21-01-18', '21-01-21', '21-12-08']

filterList = [(data['validatedThickness'] == True),
              (data['cell type'] == 'HoxB8-Macro'), 
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


valStr = 'KChadwick_'
weightStr = 'K_Weight_'



# regionFitsNames = [str(fitMin[ii]) + '<s<' + str(fitMax[ii]) for ii in range(len(fitMin))]
regionFitsNames = ['S={:.0f}+/-{:.0f}'.format(fitCenters[ii], width//2) for ii in range(len(fitCenters))]

fig, axes = plt.subplots(2,1, figsize = (9,12))
ListDfWhole = []

conditions = np.array(data_f['HoxB8_Co'].unique())


cD = {'bare glass & ctrl':[gs.colorList40[10], gs.colorList40[10]],
      'bare glass & tko':[gs.colorList40[30], gs.colorList40[30]],
      '20um fibronectin discs & ctrl':[gs.colorList40[12], gs.colorList40[12]],
      '20um fibronectin discs & tko':[gs.colorList40[32], gs.colorList40[32]]}

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
    
    data_ff = data_f[data_f['HoxB8_Co'] == co]
    
    
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
    
    if co == conditions[0]:
        texty_1 = Kavg**0.95
        texty_2 = Kavg + 2500
    else:
        texty_1 = texty_1**0.98
        texty_2 = texty_2 + 750
        
    axes[0].set_yscale('log')
    axes[0].set_ylim([500,2e4])
    axes[1].set_ylim([0,1.4e4])
    
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
            if ax.get_yscale() == 'log':
                ax.text(x=fitCenters[kk], y=texty_1[kk], s='n='+str(N[kk]), fontsize = 8, color = cD[co][1])
            elif ax.get_yscale() == 'linear':
                ax.text(x=fitCenters[kk], y=texty_2[kk], s='n='+str(N[kk]), fontsize = 8, color = cD[co][1])
    
        

    
    fig.suptitle('K(s)'+'\n(fits width: {:.0f}Pa)'.format(width))    
    
    df_val = pd.DataFrame(d_val)
    ListDfWhole.append(df_val)
    dftest = pd.DataFrame(d)

plt.show()

ufun.archiveFig(fig, name='HoxB8_K(s)', figDir = 'HoxB8project', dpi = 100)



#### Local zoom

Sinf, Ssup = 500, 900
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

conditions = np.array(data_f['HoxB8_Co'].unique())
print(conditions)


cD = {'bare glass & ctrl':[gs.colorList40[10], gs.colorList40[10]],
      'bare glass & tko':[gs.colorList40[30], gs.colorList40[30]],
      '20um fibronectin discs & ctrl':[gs.colorList40[12], gs.colorList40[12]],
      '20um fibronectin discs & tko':[gs.colorList40[32], gs.colorList40[32]]}

listDfZoom = []

for co in conditions:
    
    Kavg = []
    Kstd = []
    D10 = []
    D90 = []
    N = []
    
    data_ff = data2_f[data_f['HoxB8_Co'] == co]
    
    for ii in range(len(fitCenters)):
        S = fitCenters[ii]
        rFN = 'S={:.0f}+/-{:.0f}'.format(S, width//2)
        S = fitCenters[ii]
        variable = valStr+rFN
        weight = weightStr+rFN
        
        x = data_ff[variable].apply(nan2zero).values
        w = data_ff[weight].apply(nan2zero).values
        
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
    
    if co == conditions[0]:
        texty_1 = Kavg**0.95
        texty_2 = Kavg - 1000
    else:
        texty_1 = texty_1**0.98
        texty_2 = texty_2 - 400
        
    
    axes[0].set_yscale('log')
    axes[0].set_ylim([500,1.1e4])
    
    axes[1].set_ylim([0,10000])
    
    for ax in axes:
        # Weighted means -- Weighted ste 95% as error
        ax.errorbar(fitCenters, Kavg, yerr = q*Kste, marker = 'o', color = cD[co][0], 
                        ecolor = cD[co][1], elinewidth = 0.8, capsize = 3, label = co)
        
        # ax.set_xlim([Sinf-50, Ssup+50])
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
        ax.set_xlim([0, 1100])
        
        for kk in range(len(N)):
            if ax.get_yscale() == 'log':
                ax.text(x=fitCenters[kk], y=texty_1[kk], s='n='+str(N[kk]), fontsize = 8, color = cD[co][1])
            elif ax.get_yscale() == 'linear':
                ax.text(x=fitCenters[kk], y=texty_2[kk], s='n='+str(N[kk]), fontsize = 8, color = cD[co][1])
    
    fig.suptitle('K(s)'+' - (fits width: {:.0f}Pa)'.format(width))
    
    df_val = pd.DataFrame(d_val)
    listDfZoom.append(df_val)

plt.show()

ufun.archiveFig(fig, name='HoxB8_K(s)_localZoom_{:.0f}-{:.0f}Pa'.format(Sinf, Ssup), 
                figDir = 'HoxB8project', dpi = 100)

# %%%%% Using the function

data = GlobalTable_meca_MCAHoxB8
data['HoxB8_Co'] = data['substrate'].values + \
                   np.array([' & ' for i in range(data.shape[0])]) + \
                   data['cell subtype'].values

dates = ['22-05-03', '22-05-04', '22-05-05'] #['21-12-08', '22-01-12'] ['21-01-18', '21-01-21', '21-12-08']

Filters = [(data['validatedThickness'] == True),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 800),
            (data['date'].apply(lambda x : x in dates)),
            (data['cell type'] == 'HoxB8-Macro'),
            (data['cell subtype'].apply(lambda x : x in ['ctrl', 'tko']))]  # (data['validatedFit'] == True),

globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]

data_f = data[globalFilter]

width = 150 # 200
centers =  np.array([S for S in range(100, 900, 50)])


fig1, ax1 = plotPopKS(data, width, centers, Filters = Filters, 
                    condCol = 'HoxB8_Co', mode = 'wholeCurve', scale = 'lin', printText = False)
fig2, ax2 = plotPopKS(data, width, centers, Filters = Filters, 
                    condCol = 'HoxB8_Co', mode = '300_800', scale = 'lin', printText = True)

fig1.suptitle('HoxB8 ctrl v. tko - stiffness')
fig2.suptitle('HoxB8 ctrl v. tko - stiffness')
plt.show()


# %%%% Comparison with 3T3aSFL

# %%%%% Comparison first plot

data = GlobalTable_meca_MCAHoxB8
dates = ['22-02-09', '22-05-03', '22-05-04', '22-05-05']
StressRegion = '_S=600+/-75'
srs = StressRegion.split('+/-')[0]+'-'+StressRegion.split('+/-')[1]



filterList = [(data['validatedFit'+StressRegion] == True), 
              (data['validatedThickness'] == True),
              (data['substrate'] == '20um fibronectin discs'), 
              (data['drug'] == 'none'), 
              (data['bead type'] == 'M450'),
              (data['UI_Valid'] == True),
              (data['bestH0'] <= 800),
              (data['date'].apply(lambda x : x in dates)),
              (data['cell subtype'].apply(lambda x : x in ['aSFL', 'ctrl']))]



# co_order = makeOrder(['ctrl','tko'],['naked glass','20um fibronectin discs'])
co_order = ['3T3','HoxB8-Macro']

# All comp
fig, ax, dfcount = D1Plot(data, CondCol=['cell type'], Parameters=['bestH0', 'KChadwick'+StressRegion],Filters=Filters,
                          AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=[], figSizeFactor = 0.8, markersizeFactor=0.8, orientation = 'h', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 summary plot')

ufun.archiveFig(fig, name=('3T3vsHoxB8_H0 & K' + srs + '_allComps'), figDir = 'HoxB8project', dpi = 100)


# Avg per cell
fig, ax, dfcount = D1Plot(data, CondCol=['cell type'], Parameters=['bestH0', 'KChadwick'+StressRegion],Filters=Filters,
                          AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=[], figSizeFactor = 0.8, markersizeFactor=0.8, orientation = 'h', returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 summary plot')

ufun.archiveFig(fig, name=('3T3vsHoxB8_H0 & K' + srs+ '_cellAvg'), figDir = 'HoxB8project', dpi = 100)

plt.show()


# %%%%% Comparison only H0

data = GlobalTable_meca_MCAHoxB8
dates = ['22-02-09', '22-05-03', '22-05-04', '22-05-05']
# StressRegion = '_S=600+/-75'
# srs = StressRegion.split('+/-')[0]+'-'+StressRegion.split('+/-')[1]



Filters = [(data['validatedThickness'] == True),
              (data['substrate'] == '20um fibronectin discs'), 
              (data['drug'] == 'none'), 
              (data['bead type'] == 'M450'),
              (data['UI_Valid'] == True),
              (data['bestH0'] <= 800),
              (data['date'].apply(lambda x : x in dates)),
              (data['cell subtype'].apply(lambda x : x in ['aSFL-A11', 'ctrl']))]


# co_order = makeOrder(['ctrl','tko'],['naked glass','20um fibronectin discs'])
co_order = ['3T3','HoxB8-Macro']

# All comp
fig, ax, dfcount = D1Plot(data, CondCol=['cell type'], Parameters=['surroundingThickness'],Filters=Filters,cellID='cellID',
                          AvgPerCell=False, co_order=co_order, stats=True, statMethod='Mann-Whitney', box_pairs=[],
                          figSizeFactor = 1.0, markersizeFactor=0.9, orientation = 'h', stressBoxPlot=2, 
                          returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 summary plot')

# ufun.archiveFig(fig, name=('3T3vsHoxB8_onlyH0_allComps'), figDir = 'HoxB8project', dpi = 100)
ufun.archiveFig(fig, name=('3T3vsHoxB8_onlyH0_allComps'), dpi = 100)


# Avg per cell
fig, ax, dfcount = D1Plot(data, CondCol=['cell type'], Parameters=['surroundingThickness'],Filters=Filters, cellID='cellID',
                          AvgPerCell=True,  co_order=co_order, stats=True, statMethod='Mann-Whitney', box_pairs=[],
                          figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=1,
                          returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('3T3 vs. HoxB8 - thickness')

# ufun.archiveFig(fig, name=('3T3vsHoxB8_onlyH0_cellAvg'), figDir = 'HoxB8project', dpi = 100)
ufun.archiveFig(fig, name=('3T3vsHoxB8_onlyH0_cellAvg'), dpi = 100)

plt.show()


# %%%%% K(s)

#### Making the Dataframe 

data = GlobalTable_meca_MCAHoxB8

dates = ['22-02-09', '22-05-03', '22-05-04', '22-05-05'] #['21-12-08', '22-01-12'] ['21-01-18', '21-01-21', '21-12-08']

filterList = [(data['validatedThickness'] == True),
              (data['substrate'] == '20um fibronectin discs'), 
              (data['drug'] == 'none'), 
              (data['bead type'] == 'M450'),
              (data['UI_Valid'] == True),
              (data['bestH0'] <= 800),
              (data['date'].apply(lambda x : x in dates)),
              (data['cell subtype'].apply(lambda x : x in ['aSFL-A11', 'ctrl']))]  # (data['validatedFit'] == True),

globalFilter = filterList[0]
for k in range(1, len(filterList)):
    globalFilter = globalFilter & filterList[k]

data_f = data[globalFilter]

# data_f['HoxB8_Co'] = data_f['substrate'].values + \
#                      np.array([' & ' for i in range(data_f.shape[0])]) + \
#                      data_f['cell subtype'].values

width = 150 # 200
fitCenters =  np.array([S for S in range(100, 900, 50)])
# fitMin = np.array([int(S-(width/2)) for S in fitCenters])
# fitMax = np.array([int(S+(width/2)) for S in fitCenters])

# regionFitsNames = [str(fitMin[ii]) + '<s<' + str(fitMax[ii]) for ii in range(len(fitMin))]
regionFitsNames = ['S={:.0f}+/-{:.0f}'.format(fitCenters[ii], width//2) for ii in range(len(fitCenters))]

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


valStr = 'KChadwick_'
weightStr = 'K_Weight_'



# regionFitsNames = [str(fitMin[ii]) + '<s<' + str(fitMax[ii]) for ii in range(len(fitMin))]
regionFitsNames = ['S={:.0f}+/-{:.0f}'.format(fitCenters[ii], width//2) for ii in range(len(fitCenters))]

fig, axes = plt.subplots(2,1, figsize = (9,12))
ListDfWhole = []

conditions = np.array(data_f['cell type'].unique())


cD = {'HoxB8-Macro':[gs.colorList40[22], gs.colorList40[22]],
      '3T3':[gs.colorList40[20], gs.colorList40[20]]}

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
    
    data_ff = data_f[data_f['cell type'] == co]
    
    
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
    
    if co == conditions[0]:
        texty_1 = Kavg**0.95
        texty_2 = Kavg + 2500
    else:
        texty_1 = texty_1**0.98
        texty_2 = texty_2 + 750
        
    axes[0].set_yscale('log')
    axes[0].set_ylim([500,2e4])
    axes[1].set_ylim([0,1.4e4])
    
    for ax in axes:
        # Weighted means -- Weighted ste 95% as error
        ax.errorbar(fitCenters, Kavg, yerr = q*Kste, marker = 'o', color = cD[co][0], 
                       ecolor = cD[co][1], elinewidth = 0.8, capsize = 3, label = co)
        ax.set_ylim([500,2e4])
        ax.set_xlim([0,950])
        ax.set_title('K(s) - All compressions pooled')
        
        ax.legend(loc = 'upper left')
        
        ax.set_xlabel('Stress (Pa)')
        ax.set_ylabel('K (Pa)')
        
        for kk in range(len(N)):
            if ax.get_yscale() == 'log':
                ax.text(x=fitCenters[kk], y=texty_1[kk], s='n='+str(N[kk]), fontsize = 8, color = cD[co][1])
            elif ax.get_yscale() == 'linear':
                ax.text(x=fitCenters[kk], y=texty_2[kk], s='n='+str(N[kk]), fontsize = 8, color = cD[co][1])
    
    # for ax in axes:
    #     # Weighted means -- Weighted ste 95% as error
    #     ax.errorbar(fitCenters, Kavg, yerr = q*Kste, marker = 'o', color = cD[co][0], 
    #                    ecolor = cD[co][1], elinewidth = 0.8, capsize = 3, label = co)
    #     ax.set_ylim([500,2e4])
    #     ax.set_xlim([0,1100])
    #     ax.set_title('K(s) - All compressions pooled')
        
    #     ax.legend(loc = 'upper left')
        
    #     ax.set_xlabel('Stress (Pa)')
    #     ax.set_ylabel('K (Pa)')
        
    #     for kk in range(len(N)):
    #         ax.text(x=fitCenters[kk], y=Kavg[kk]**0.9, s='n='+str(N[kk]), fontsize = 8, color = cD[co][1])
    #     # for kk in range(len(N)):
    #     #     ax[k].text(x=fitCenters[kk]+oD[co][0], y=Kavg[kk]**oD[co][1], 
    #     #                s='n='+str(N[kk]), fontsize = 6, color = cD[co][k])
        
    axes[0].set_yscale('log')
    axes[0].set_ylim([500,2e4])
    
    axes[1].set_ylim([0,1.4e4])
    
    fig.suptitle('K(s)'+'\n(fits width: {:.0f}Pa)'.format(width))    
    
    df_val = pd.DataFrame(d_val)
    ListDfWhole.append(df_val)
    dftest = pd.DataFrame(d)

plt.show()

# ufun.archiveFig(fig, name='3T3vsHoxB8_K(s)', figDir = 'HoxB8project', dpi = 100)
ufun.archiveFig(fig, name='3T3vsHoxB8_K(s)', dpi = 100)






#### Local zoom

Sinf, Ssup = 150, 450
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

maxK = 0

conditions = np.array(data_f['cell type'].unique())
print(conditions)


cD = {'HoxB8-Macro':[gs.colorList40[22], gs.colorList40[22]],
      '3T3':[gs.colorList40[20], gs.colorList40[20]]}

listDfZoom = []

for co in conditions:
    
    Kavg = []
    Kstd = []
    D10 = []
    D90 = []
    N = []
    
    data_ff = data2_f[data_f['cell type'] == co]
    
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
    
    if np.max(Kavg) > maxK:
        maxK = np.max(Kavg)
    
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
        
        for kk in range(len(N)):
            ax.text(x=fitCenters[kk], y=Kavg[kk]**0.9, s='n='+str(N[kk]), fontsize = 8, color = cD[co][1])
    

    
    df_val = pd.DataFrame(d_val)
    listDfZoom.append(df_val)

for ax in axes:
    ax.legend(loc = 'upper left')
    ax.set_xlabel('Stress (Pa)')
    ax.set_ylabel('K (Pa)')

axes[0].set_yscale('log')
axes[0].set_ylim([500,2e4])
axes[1].set_ylim([0, 1.2*maxK])

fig.suptitle('K(s)'+' - (fits width: {:.0f}Pa)'.format(width))

plt.show()

# ufun.archiveFig(fig, name='3T3vsHoxB8_K(s)_localZoom_{:.0f}-{:.0f}Pa'.format(Sinf, Ssup), 
#                 figDir = 'HoxB8project', dpi = 100)

ufun.archiveFig(fig, name='3T3vsHoxB8_K(s)_localZoom_{:.0f}-{:.0f}Pa'.format(Sinf, Ssup), 
                dpi = 100)



# %%%%% K(s) - as a function



def plotPopKS(data, width, centers, Filters = [], condCol = '', 
              mode = 'wholeCurve', scale = 'lin', printText = True):

    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for k in range(0, len(filterList)):
        globalFilter = globalFilter & filterList[k]
    
    data_f = data[globalFilter]

    # regionFitsNames = [str(fitMin[ii]) + '<s<' + str(fitMax[ii]) for ii in range(len(fitMin))]
    regionFitsNames = ['S={:.0f}+/-{:.0f}'.format(centers[ii], width//2) for ii in range(len(centers))]

    fig, ax = plt.subplots(1,1, figsize = (9,6))

    ListDfWhole = []

    conditions = np.array(data_f[condCol].unique())

    valStr = 'KChadwick_'
    weightStr = 'K_Weight_'
    
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
    
    
    #### Whole curve / local part
    
    if mode == 'wholeCurve':
        data2_f = data_f
        ax.set_xlim([0, np.max(centers)+50])  
        
    else:
        bounds = mode.split('_')
        Sinf, Ssup = int(bounds[0]), int(bounds[1])
        extraFilters = [data_f['minStress'] <= Sinf, data_f['maxStress'] >= Ssup] # >= 800
        centers = centers[(centers>=(Sinf)) & (centers<=Ssup)] # <800
    
        data2_f = data_f
        globalExtraFilter = extraFilters[0]
        for k in range(1, len(extraFilters)):
            globalExtraFilter = globalExtraFilter & extraFilters[k]
        data2_f = data2_f[globalExtraFilter]
            
        ax.set_xlim([np.min(centers)-50, np.max(centers)+50])        
    
    for co in conditions:
        Kavg = []
        Kstd = []
        D10 = []
        D90 = []
        N = []
        
        data_ff = data2_f[data2_f['cell type'] == co]
        color = styleDict1[co]['color']
        
        for ii in range(len(centers)):
            S = centers[ii]
            rFN = 'S={:.0f}+/-{:.0f}'.format(S, width//2)
            variable = valStr+rFN
            weight = weightStr+rFN
            
            x = data_ff[variable].apply(nan2zero).values
            w = data_ff[weight].apply(nan2zero).values
            
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
        
        d_val = {'S' : centers, 'Kavg' : Kavg, 'Kstd' : Kstd, 'D10' : D10, 'D90' : D90, 'N' : N}
        
        if scale == 'lin':
            if co == conditions[0]:
                texty = Kavg + 1500
            else:
                texty = texty + 300
            
            ax.set_yscale('linear')
            # ax.set_ylim([0,1.2e1])
            # ax.set_ylim([0,0.8e1])
                
        elif scale == 'log':
            if co == conditions[0]:
                texty = Kavg**0.95
            else:
                texty = texty**0.98
            
            ax.set_yscale('log')
            # ax.set_ylim([0.5,2e1])
            
        # Weighted means -- Weighted ste 95% as error
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
        
        df_val = pd.DataFrame(d_val)
        ListDfWhole.append(df_val)
        
    fig.suptitle('K(s)'+' - fits width: {:.0f}Pa'.format(width))  
    return(fig, ax)


#### Making the Dataframe 

data = GlobalTable_meca_MCAHoxB8

dates = ['22-02-09', '22-05-03', '22-05-04', '22-05-05'] #['21-12-08', '22-01-12'] ['21-01-18', '21-01-21', '21-12-08']

Filters = [(data['validatedThickness'] == True),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 800),
            (data['date'].apply(lambda x : x in dates)),
            (data['cell subtype'].apply(lambda x : x in ['aSFL-A11', 'ctrl']))]  # (data['validatedFit'] == True),

width = 150 # 200
centers =  np.array([S for S in range(100, 900, 50)])


fig1, ax1 = plotPopKS(data, width, centers, Filters = Filters, 
                    condCol = 'cell type', mode = 'wholeCurve', scale = 'lin', printText = False)
fig2, ax2 = plotPopKS(data, width, centers, Filters = Filters, 
                    condCol = 'cell type', mode = '150_450', scale = 'lin', printText = False)

fig1.suptitle('3T3 vs. HoxB8 - stiffness')
fig2.suptitle('3T3 vs. HoxB8 - stiffness')
plt.show()



# %%%% Plots pour Perrine Verdys

data = GlobalTable_meca_HoxB8
dates = ['22-05-03', '22-05-04', '22-05-05']

thicknessType = 'bestH0' # 'bestH0', 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
            (data['UI_Valid'] == True),
            (data['cell type'] == 'HoxB8-Macro'), 
            (data['substrate'] == 'bare glass'), 
            (data['bead type'] == 'M450'),
            (data[thicknessType] <= 800),
            (data['date'].apply(lambda x : x in dates))]


# co_order = makeOrder(['ctrl','tko'],['naked glass','20um fibronectin discs'])
co_order = makeOrder(['ctrl','tko'])

fig, ax, dfcount = D1Plot(data, CondCol=['cell subtype'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=[], figSizeFactor = 0.9, markersizeFactor = 1.2, orientation = 'v', returnCount = 1,
                 stressBoxPlot=True)

ax[0].tick_params(axis='y', which='major', labelsize=14)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 cortical thickness' + '\n(all compressions)')

ufun.archiveFig(fig, name=('4PV_HoxB8_'+thicknessType+' only' + '_allComps'), figDir = 'HoxB8project', dpi = 100)


thicknessType = 'ctFieldThickness' # 'bestH0', 'ctFieldThickness'

Filters = [(data['validatedThickness'] == True), 
            (data['UI_Valid'] == True),
            (data['cell type'] == 'HoxB8-Macro'), 
            (data['substrate'] == 'bare glass'), 
            (data['bead type'] == 'M450'),
            (data[thicknessType] <= 600),
            (data['date'].apply(lambda x : x in dates))]

fig, ax, dfcount = D1Plot(data, CondCol=['cell subtype'], Parameters=[thicknessType],Filters=Filters,
                 AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                 box_pairs=[], figSizeFactor = 0.9, markersizeFactor = 1.7, orientation = 'v', returnCount = 1,
                 stressBoxPlot=True)

ax[0].tick_params(axis='y', which='major', labelsize=14)

renameAxes(ax,renameDict1)
fig.suptitle('HoxB8 cortical thickness' + '\n(median per cell)')

ufun.archiveFig(fig, name=('4PV_HoxB8_'+thicknessType+' only' + '_cellAvg'), figDir = 'HoxB8project', dpi = 200)


plt.show()

#### Making the Dataframe 

data = GlobalTable_meca_HoxB8

data['HoxB8_Co'] = data['substrate'].values + \
                   np.array([' & ' for i in range(data.shape[0])]) + \
                   data['cell subtype'].values

dates = ['22-05-03', '22-05-04', '22-05-05'] #['21-12-08', '22-01-12'] ['21-01-18', '21-01-21', '21-12-08']

filterList = [(data['validatedThickness'] == True),
              (data['cell type'] == 'HoxB8-Macro'), 
              (data['substrate'] == 'bare glass'), 
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

#### Local zoom

valStr = 'KChadwick_'
weightStr = 'K_Weight_'

Sinf, Ssup = 300, 700
extraFilters = [data_f['minStress'] <= Sinf, data_f['maxStress'] >= Ssup] # >= 800
fitCenters = fitCenters[(fitCenters>=(Sinf)) & (fitCenters<=Ssup)] # <800
# fitMin = np.array([int(S-(width/2)) for S in fitCenters])
# fitMax = np.array([int(S+(width/2)) for S in fitCenters])

data2_f = data_f
globalExtraFilter = extraFilters[0]
for k in range(1, len(extraFilters)):
    globalExtraFilter = globalExtraFilter & extraFilters[k]
data2_f = data2_f[globalExtraFilter]  


fig, ax = plt.subplots(1,1, figsize = (8,5))

conditions = np.array(data_f['HoxB8_Co'].unique())
print(conditions)


cD = {'bare glass & ctrl':[gs.colorList40[10], gs.colorList40[10]],
      'bare glass & tko':[gs.colorList40[30], gs.colorList40[30]],
      '20um fibronectin discs & ctrl':[gs.colorList40[12], gs.colorList40[12]],
      '20um fibronectin discs & tko':[gs.colorList40[32], gs.colorList40[32]]}

listDfZoom = []

for co in conditions:
    
    Kavg = []
    Kstd = []
    D10 = []
    D90 = []
    N = []
    
    data_ff = data2_f[data_f['HoxB8_Co'] == co]
    
    for ii in range(len(fitCenters)):
        S = fitCenters[ii]
        rFN = 'S={:.0f}+/-{:.0f}'.format(S, width//2)
        S = fitCenters[ii]
        variable = valStr+rFN
        weight = weightStr+rFN
        
        x = data_ff[variable].apply(nan2zero).values
        w = data_ff[weight].apply(nan2zero).values
        
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
    
    if co == conditions[0]:
        texty_1 = Kavg**0.95
        texty_2 = Kavg - 1000
    else:
        texty_1 = texty_1**0.98
        texty_2 = texty_2 - 400
        
    
    # axes[0].set_yscale('log')
    # axes[0].set_ylim([500,1.1e4])
    
    ax.set_ylim([0,8])
    

    # Weighted means -- Weighted ste 95% as error
    ax.errorbar(fitCenters, Kavg/1000, yerr = q*Kste/1000, 
                linewidth = 2,
                marker = 'o', color = cD[co][0], markersize = 8, markeredgecolor = 'k',
                ecolor = cD[co][1], elinewidth = 1.5, capsize = 6, label = co)
    
    # ax.set_xlim([Sinf-50, Ssup+50])
    ax.set_title('K(stress) - compressions including the [{:.0f},{:.0f}] Pa range\n'.format(Sinf, Ssup))
    
    # Weighted means -- D9-D1 as error
    # ax[1].errorbar(fitCenters, Kavg, yerr = [D10, D90], marker = 'o', color = cD[co][1], 
    #                 ecolor = 'k', elinewidth = 0.8, capsize = 3, label = lD[co]) 
    # ax[1].set_ylim([500,2e4])
    # ax[1].set_xlim([200,900])
    
    # for k in range(2):
    ax.legend(loc = 'upper left')
    
    ax.set_xlabel('Stress (Pa)')
    ax.set_ylabel('K (kPa)')
    ax.set_xlim([Sinf-50, Ssup+50])
    
    # for kk in range(len(N)):
    #     if ax.get_yscale() == 'log':
    #         ax.text(x=fitCenters[kk], y=texty_1[kk]/1000, s='n='+str(N[kk]), fontsize = 8, color = cD[co][1])
    #     elif ax.get_yscale() == 'linear':
    #         ax.text(x=fitCenters[kk], y=texty_2[kk]/1000, s='n='+str(N[kk]), fontsize = 8, color = cD[co][1])
    
    # fig.suptitle('K(s)'+' - (fits width: {:.0f}Pa)'.format(width))
    
    df_val = pd.DataFrame(d_val)
    listDfZoom.append(df_val)

plt.show()

ufun.archiveFig(fig, name='4PV_HoxB8_K(s)_localZoom_{:.0f}-{:.0f}Pa'.format(Sinf, Ssup), 
                figDir = 'HoxB8project', dpi = 200)



# %%%% Typical constant field force

data = GlobalTable_meca_HoxB8_new
dates = ['22-05-03', '22-05-04', '22-05-05']


Filters = [(data['validatedThickness'] == True), 
           (data['UI_Valid'] == True),
           (data['ctFieldForce'] < 600),
           (data['cell type'] == 'HoxB8-Macro'), 
           (data['substrate'] == 'bare glass'),
           (data['cell subtype'].apply(lambda x : x in ['ctrl','tko'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in dates))]

fltr = Filters[0]
for i in range(1, len(Filters)):
    fltr = fltr & Filters[i]
print(np.median(data[fltr]['ctFieldForce']))
print(np.median(data[fltr]['minForce']))
print(np.median(data[fltr]['maxForce']))
print(np.std(data[fltr]['ctFieldForce']))

co_order = []
box_pairs=[]

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype'], Parameters=['ctFieldForce'],Filters=Filters,
                          AvgPerCell=True, cellID='cellID', co_order=co_order, stats=False, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 1.0, markersizeFactor=1.0, orientation = 'v', stressBoxPlot= False, 
                          returnData = 1, returnCount = 1)


plt.show()

# %%%% Figures for the paper

# %%%%% Thickness

data_main = GlobalTable_meca_HoxB8_new
data = data_main

dates = ['22-05-03', '22-05-04', '22-05-05']

Filters = [(data['validatedThickness'] == True),
           (data['surroundingThickness'] <= 900),
           (data['ctFieldThickness'] <= 1000),
           (data['UI_Valid'] == True),
           (data['cell type'] == 'HoxB8-Macro'), 
           (data['substrate'] == 'bare glass'),
           (data['cell subtype'].apply(lambda x : x in ['ctrl','tko'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in dates))]


descText = """
data_main = GlobalTable_meca_HoxB8_new
data = data_main

dates = ['22-05-03', '22-05-04', '22-05-05']

Filters = [(data['validatedThickness'] == True),
           (data['surroundingThickness'] <= 900),
           (data['ctFieldThickness'] <= 1000),
           (data['UI_Valid'] == True),
           (data['cell type'] == 'HoxB8-Macro'), 
           (data['substrate'] == 'bare glass'),
           (data['cell subtype'].apply(lambda x : x in ['ctrl','tko'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in dates))]
"""



co_order = ['ctrl', 'tko']
box_pairs=[]

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype'], Parameters=['surroundingThickness'],Filters=Filters,
                          AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 1, markersizeFactor=1, orientation = 'v', stressBoxPlot=2,
                          returnData = 1, returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('Fig 1 - Thickness - Bare Glass')

ufun.archiveFig(fig, name=('Fig1_surroundingH_BareGlass_allComps'), figDir = 'HoxB8_Paper', dpi = 100)
ufun.archiveData(dfexport, name=('Fig1_surroundingH_BareGlass_allComps'), 
                  sep = ';', saveDir = 'HoxB8_Paper', descText = descText)

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype'], Parameters=['ctFieldThickness'],Filters=Filters,
                          AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 1, markersizeFactor=1.2, orientation = 'v', stressBoxPlot= 1, 
                          returnData = 1, returnCount = 1)

renameAxes(ax,renameDict1)
fig.suptitle('Fig 1 - Thickness - Bare Glass')

ufun.archiveFig(fig, name=('Fig1_ctFieldH_BareGlass_cellAvg'), figDir = 'HoxB8_Paper', dpi = 100)
ufun.archiveData(dfexport, name=('Fig1_ctFieldH_BareGlass_cellAvg'), 
                  sep = ';', saveDir = 'HoxB8_Paper', descText = descText)



plt.show()


# %%%%% Stiffness

data_main = GlobalTable_meca_HoxB8_new
fitType = 'stressRegion'
fitId = '250_100'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

dates = ['22-05-03', '22-05-04', '22-05-05']

Filters = [(data['validatedThickness'] == True),
           (data['fit_valid'] == True),
           # (data['fit_K'] > 1000),
           (data['UI_Valid'] == True),
           (data['cell type'] == 'HoxB8-Macro'), 
           (data['substrate'] == 'bare glass'),
           (data['cell subtype'].apply(lambda x : x in ['ctrl','tko'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in dates))]


descText = """
data_main = GlobalTable_meca_HoxB8_new
fitType = 'stressRegion'
fitId = '250_100'

data = taka2.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)

dates = ['22-05-03', '22-05-04', '22-05-05']

Filters = [(data['validatedThickness'] == True),
           (data['fit_valid'] == True),
           (data['UI_Valid'] == True),
           (data['cell type'] == 'HoxB8-Macro'), 
           (data['substrate'] == 'bare glass'),
           (data['cell subtype'].apply(lambda x : x in ['ctrl','tko'])), 
           (data['bead type'] == 'M450'),
           (data['date'].apply(lambda x : x in dates))]
"""


co_order = ['ctrl', 'tko']
box_pairs=[]

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype'], Parameters=['fit_K'],Filters=Filters,
                          AvgPerCell=False, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 1, markersizeFactor=1, orientation = 'v', stressBoxPlot=2,
                          returnData = 1, returnCount = 1)
# ax[0].set_ylim([1e2,1.2e4])
renameAxes(ax,renameDict1)
fig.suptitle('Fig 1 - Stiffness - Bare Glass\n' + fitStr)

ufun.archiveFig(fig, name=('Fig1_K_' + fitId + '_BareGlass_allComps'), figDir = 'HoxB8_Paper', dpi = 100)
ufun.archiveData(dfexport, name=('Fig1_K_' + fitId + '_BareGlass_allComps'), 
                  sep = ';', saveDir = 'HoxB8_Paper', descText = descText)
print(dfcount[['cellCount', 'compCount']])

fig, ax, dfexport, dfcount = D1Plot(data, CondCols=['cell subtype'], Parameters=['fit_K'],Filters=Filters,
                          AvgPerCell=True, cellID='cellID', co_order=co_order, stats=True, statMethod='Mann-Whitney', 
                          box_pairs=box_pairs, figSizeFactor = 1, markersizeFactor=1.2, orientation = 'v', stressBoxPlot= 1, 
                          returnData = 1, returnCount = 1)
# ax[0].set_ylim([3e2,1.2e4])
renameAxes(ax,renameDict1)
fig.suptitle('Fig 1 - Stiffness - Bare Glass\n' + fitStr)

ufun.archiveFig(fig, name=('Fig1_K_' + fitId + '_BareGlass_cellAvg'), figDir = 'HoxB8_Paper', dpi = 100)
ufun.archiveData(dfexport, name=('Fig1_K_' + fitId + '_BareGlass_cellAvg'), 
                  sep = ';', saveDir = 'HoxB8_Paper', descText = descText)



plt.show()
