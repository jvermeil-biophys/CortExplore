# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:51:36 2023

@author: anumi
"""
# %% > Imports and constants

#### Main imports

import random
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



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
import matplotlib.lines as lines
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec
from scipy.stats import mannwhitneyu, wilcoxon, ranksums
from statannotations.Annotator import Annotator

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)

import GraphicStyles as gs
import UtilityFunctions as ufun
import TrackAnalyser_V3 as taka


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



#%% Functions

def dataGroup(df, groupCol = 'cellID', idCols = [], numCols = [], aggFun = 'mean'):
    agg_dict = {'date':'first',
                'cellName':'first',
                'cellID':'first',	
                'manipID':'first',	
                'compNum':'count',
                }
    for col in idCols:
        agg_dict[col] = 'first'
    for col in numCols:
        agg_dict[col] = aggFun
    
    all_cols = agg_dict.keys()
    group = df[all_cols].groupby(groupCol)
    df_agg = group.agg(agg_dict)
    return(df_agg)

def makeDirs(fitsSubDir, FIT_MODE):
    if not os.path.exists(todayFigDir):
        os.mkdir(todayFigDir)

    pathSubDir = todayFigDir+'/'+fitsSubDir
    if not os.path.exists(pathSubDir):
        os.mkdir(pathSubDir)
        
    pathFits = pathSubDir + '/' + FIT_MODE
    if not os.path.exists(pathFits):
        os.mkdir(pathFits)
        
    pathBoxPlots = pathFits + '/BoxPlots'
    if not os.path.exists(pathBoxPlots):
        os.mkdir(pathBoxPlots)

    pathNonlinDir = pathSubDir+'/NonLinPlots'
    if not os.path.exists(pathNonlinDir):
        os.mkdir(pathNonlinDir)
        
    pathSSPlots = pathSubDir+'/Stress-strain Plots'
    if not os.path.exists(pathSSPlots):
        os.mkdir(pathSSPlots)
        
    pathKSPlots = pathSubDir+'/KvsS Plots'
    if not os.path.exists(pathKSPlots):
        os.mkdir(pathKSPlots)
        
    return pathSubDir,pathFits,pathBoxPlots,pathNonlinDir,pathSSPlots,pathKSPlots

def plotCellTimeSeriesData(cellID, fromPython = True):
    X = 'T'
    Y = np.array(['B', 'F', 'dx', 'dy', 'dz', 'D2', 'D3'])
    units = np.array([' (mT)', ' (pN)', ' (µm)', ' (µm)', ' (µm)', ' (µm)', ' (µm)'])
    timeSeriesDataFrame = ufun.getCellTimeSeriesData(cellID, fromPython)
    print(timeSeriesDataFrame.shape)
    # my_default_color_cycle = cycler(color=my_default_color_list)
    # plt.rcParams['axes.prop_cycle'] = my_default_color_cycle
    if not timeSeriesDataFrame.size == 0:
#         plt.tight_layout()
#         fig.show() # figsize=(20,20)
        axes = timeSeriesDataFrame.plot(x=X, y=Y, kind='line', ax=None, subplots=True, sharex=True, sharey=False, layout=None, \
                       figsize=(8,10), use_index=True, title = cellID + ' - f(t)', grid=None, legend=False, style=None, logx=False, logy=False, \
                       loglog=False, xticks=None, yticks=None, xlim=None, ylim=None, rot=None, fontsize=None, colormap=None, \
                       table=False, yerr=None, xerr=None, secondary_y=False, sort_columns=False)
        plt.gcf().tight_layout()
        for i in range(len(Y)):
            axes[i].set_ylabel(Y[i] + units[i])
        # plt.gcf().show()
        plt.show()
    else:
        print('cell not found')

#%% Plotting Functions

def filterDf(Filters, data):
    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for k in range(0, len(Filters)):
        globalFilter = globalFilter & Filters[k]
    data_f = data[globalFilter]
    return data_f

def createAvgDf(data, condCol):
    group_by_cell = data.groupby(['cellID'])
    avgDf = group_by_cell.agg({'H0_vwc_Full':['var', 'std', 'mean', 'count', 'median'], 
                               'bestH0_log':['var', 'std', 'mean', 'count', 'median'],
                               'NLI_Ind':['var', 'std', 'mean', 'count'],
                               'E_eff':['var', 'std', 'mean', 'count', 'median'],
                               'E_eff_log':['var', 'std', 'mean', 'count', 'median'], 
                               'cellID' : 'first',
                               'cellName':'first', 
                               'NLI_Plot' : 'first', 
                               'dateCell':'first',
                               'ctFieldFluctuAmpli' : 'first', 
                               'normFluctu' : 'first',
                               'NLI_mod':['mean', 'count', 'std', 'var'], 
                               'date' : 'first',
                               condCol:'first', 'cellCode':'first', 'manip':'first'})
    
    avgDf_wE = group_by_cell.agg({'compNum' : ['count'], 'wE_eff':[ 'sum'], 
                                 'weights_E_eff': ['sum']})
    
    avgDf_wE[('E_eff', 'wAvg')] = avgDf_wE['wE_eff', 'sum'] / avgDf_wE['weights_E_eff', 'sum']
    
    avgDf = avgDf.join(avgDf_wE)
    
    return avgDf.copy()


def dfCellPairs(avgDf):
    dfCleaned = avgDf.drop_duplicates(subset = ('cellID', 'first'))
    group_by_cell = dfCleaned.groupby([('dateCell', 'first')])
    dfGrouped = group_by_cell.agg({('dateCell', 'first') : ['first', 'count']})

    pairedCells = dfGrouped[('dateCell', 'first', 'first')][dfGrouped[('dateCell', 'first', 'count')] == 2].values
    dfPairs = avgDf[(avgDf[('dateCell', 'first')].apply(lambda x : x in pairedCells))]
    return dfPairs.copy(), pairedCells

def createDataTable(GlobalTable):
    data_main = GlobalTable
    data_main['dateID'] = GlobalTable['date']
    data_main['manipId'] = GlobalTable['manipID']
    data_main['cellId'] = GlobalTable['cellID']
    data_main['dateCell'] = GlobalTable['date'] + '_' + GlobalTable['cellCode']
    data_main['wellID'] = GlobalTable['cellCode'].str.split('_').str[0]

    nBins = 11
    bins = np.linspace(1, 2000, nBins)
    data_main['H0_Bin'] = np.digitize(GlobalTable['bestH0'], bins, right = True)
    data_main['Thickness_Bin'] = np.digitize(GlobalTable['surroundingThickness'], bins, right = True)
    
    data_main['NLI_Plot'] = [np.nan]*len(data_main)
    data_main['NLI_Ind'] = [np.nan]*len(data_main)
    data_main['E_eff'] = [np.nan]*len(data_main)

    K, Y = data_main['K_vwc_Full'], data_main['Y_vwc_Full']
    E = Y + K*(0.8)**-4

    data_main['E_eff'] = E
    data_main['ciwE_eff'] = data_main['ciwY_vwc_Full'] + data_main['ciwK_vwc_Full']*(0.8)**-4
    data_main['weights_E_eff'] = data_main['E_eff'] / data_main['ciwE_eff']**2
    data_main['wE_eff'] =  data_main['E_eff'] * data_main['weights_E_eff']
    
    data_main['NLI'] = np.log10((0.8)**-4 * K/Y)
    
    data_main['bestH0_log'] = np.log10(GlobalTable['H0_vwc_Full'].values)
    data_main['E_eff_log'] = np.log10(E)
    
    NLItypes = ['linear', 'intermediate', 'non-linear']
    for i in NLItypes:
        if i == 'linear':
            index = data_main[data_main['NLI'] < -0.3].index
            ID = 1
        elif i =='non-linear':
            index =  data_main[data_main['NLI'] > 0.3].index
            ID = 0
        elif i =='intermediate':
            index = data_main[(data_main['NLI'] > -0.3) & (data_main['NLI'] < 0.3)].index
            ID = 0.5
        for j in index:
            data_main.loc[j, 'NLI_Plot'] = str(i)
            data_main.loc[j, 'NLI_Ind'] = ID
    
    data_main['Y_err_div10'], data_main['K_err_div10'] = data_main['ciwY_vwc_Full']/10, data_main['ciwK_vwc_Full']/10
    data_main['Y_NLImod'] = data_main[["Y_vwc_Full", "Y_err_div10"]].max(axis=1)
    data_main['K_NLImod'] = data_main[["K_vwc_Full", "K_err_div10"]].max(axis=1)
    Y_nli, K_nli = data_main['Y_NLImod'].values, data_main['K_NLImod'].values

    data_main['NLI_mod'] = np.log10((0.8)**-4 * K_nli/Y_nli)
    data_main['normFluctu'] = data_main['ctFieldFluctuAmpli'] /  data_main['ctFieldThickness']
    
    return data_main

def plotNLI_Scatter(fig, ax, data, dates, condCat, condCol, pairs, labels = [],  
                    palette = ['#b96a9b', '#d6c9bc', '#92bda4'], marker_dates = {},
            colorScheme = 'black', plotSettings = {}, plotChars = {}):
        
    if colorScheme == 'black':
        plt.style.use('default')
        fig.patch.set_facecolor('black')
        fontColor = '#ffffff'
        lineColor = '#000000'
    else:
        plt.style.use('default')
        fontColor = '#000000'
        lineColor = '#000000'
        
    NComps = data.groupby([condCol, 'date'])['NLI_Plot'].count().reset_index()    
    NComps = NComps.iloc[pd.Categorical(NComps[condCol], condCat).argsort()].reset_index(drop = True)    
    
    linear = data[data.NLI_Plot=='linear'].groupby([condCol, 'date'])['NLI_Plot'].count().reset_index().fillna(0)
    linear = linear.iloc[pd.Categorical(linear[condCol], condCat).argsort()].reset_index(drop = True)   
    
    nonlinear = data[data.NLI_Plot=='non-linear'].groupby([condCol, 'date'])['NLI_Plot'].count().reset_index().fillna(0)
    nonlinear = nonlinear.iloc[pd.Categorical(nonlinear[condCol], condCat).argsort()].reset_index(drop = True)   
    
    intermediate = data[data.NLI_Plot=='intermediate'].groupby([condCol, 'date'])['NLI_Plot'].count().reset_index().fillna(0)
    intermediate = intermediate.iloc[pd.Categorical(intermediate[condCol], condCat).argsort()].reset_index(drop = True)   

    linear['NLI_Plot'] = [i / j * 100 for i,j in zip(linear['NLI_Plot'], NComps['NLI_Plot'])]
    nonlinear['NLI_Plot'] = [i / j * 100 for i,j in zip(nonlinear['NLI_Plot'], NComps['NLI_Plot'])]
    intermediate['NLI_Plot'] = [i / j * 100 for i,j in zip(intermediate['NLI_Plot'], NComps['NLI_Plot'])]
    
    NComps['NLI_Plot'] = [i / j * 100 for i,j in zip(NComps['NLI_Plot'], NComps['NLI_Plot'])]

    for i in dates:
        print(i)
        print(linear)
        m = marker_dates.get(i)
        lin, nonlin, inter = linear, nonlinear, intermediate
        sns.lineplot(ax = ax, x=condCol, y="NLI_Plot", data=nonlin[nonlin.date == i], 
                     color=palette[0], marker = m, **plotSettings)
        sns.lineplot(ax = ax, x=condCol,  y="NLI_Plot", data=inter[inter.date == i], 
                     color=palette[1], marker = m,  **plotSettings)
        sns.lineplot(ax = ax, x=condCol, y="NLI_Plot", data=lin[lin.date == i], 
                     color=palette[2], marker = m, **plotSettings)

    if condCol == 'compNum':
        xticks = condCat -1
    else:
        xticks = np.arange(len(condCat))
        
    pvals = []
    
    if pairs != None:
        for pair in pairs:
            a1 = data['NLI'][data[condCol] == pair[0]].values
            b1 = data['NLI'][data[condCol] == pair[1]].values
            U1, p = mannwhitneyu(a1, b1, nan_policy = 'omit')
            # U1, p = ranksums(a1, b1, nan_policy = 'omit')
            pvals.append(p)
    
        annotator = Annotator(ax = ax, pairs = pairs, x=condCol,  y="NLI_Plot", data=NComps)
        annotator.configure(text_format="simple", color = lineColor, loc = 'outside')
        annotator.set_pvalues(pvals).annotate(line_offset_to_group = 0.10)
    
    if labels != []:
        plt.xticks(xticks, labels, **plotChars)
        
    plt.xticks(**plotChars)
    plt.yticks(**plotChars)
    plt.legend(fontsize = 15, labelcolor='linecolor')
    plt.ylim(0,100)

    return fig, ax, pvals

def plotNLI_Scatter_Avg(fig, ax, data, condCat, condCol, pairs, labels = [],  
                    palette = ['#b96a9b', '#d6c9bc', '#92bda4'],
            colorScheme = 'black', plotSettings = {}, plotChars = {}):
        
    if colorScheme == 'black':
        plt.style.use('default')
        fig.patch.set_facecolor('black')
        fontColor = '#ffffff'
        lineColor = '#000000'
    else:
        plt.style.use('default')
        fontColor = '#000000'
        lineColor = '#000000'
        
    NComps = data.groupby([condCol, 'date'])['NLI_Plot'].count().reset_index()    
    NComps = NComps.iloc[pd.Categorical(NComps[condCol], condCat).argsort()].reset_index(drop = True)    
    
    linear = data[data.NLI_Plot=='linear'].groupby([condCol, 'date'])['NLI_Plot'].count().reset_index()
    linear = linear.iloc[pd.Categorical(linear[condCol], condCat).argsort()].reset_index(drop = True)   
    
    nonlinear = data[data.NLI_Plot=='non-linear'].groupby([condCol, 'date'])['NLI_Plot'].count().reset_index()
    nonlinear = nonlinear.iloc[pd.Categorical(nonlinear[condCol], condCat).argsort()].reset_index(drop = True)   
    
    intermediate = data[data.NLI_Plot=='intermediate'].groupby([condCol, 'date'])['NLI_Plot'].count().reset_index()
    intermediate = intermediate.iloc[pd.Categorical(intermediate[condCol], condCat).argsort()].reset_index(drop = True)   

    linear['NLI_Plot'] = [i / j * 100 for i,j in zip(linear['NLI_Plot'], NComps['NLI_Plot'])]
    nonlinear['NLI_Plot'] = [i / j * 100 for i,j in zip(nonlinear['NLI_Plot'], NComps['NLI_Plot'])]
    intermediate['NLI_Plot'] = [i / j * 100 for i,j in zip(intermediate['NLI_Plot'], NComps['NLI_Plot'])]
    
    NComps['NLI_Plot'] = [i / j * 100 for i,j in zip(NComps['NLI_Plot'], NComps['NLI_Plot'])]


    sns.lineplot(ax = ax, x=condCol, y="NLI_Plot", data=nonlinear, color=palette[0], **plotSettings)
    sns.lineplot(ax = ax, x=condCol,  y="NLI_Plot", data=intermediate,  color=palette[1],   **plotSettings)
    sns.lineplot(ax = ax, x=condCol, y="NLI_Plot", data=linear, color=palette[2],  **plotSettings)

    
    if condCol == 'compNum':
        xticks = condCat - 1
    else:
        xticks = np.arange(len(condCat))
        
    pvals = []
    
    if pairs != None:
        for pair in pairs:
            a1 = data['NLI'][data[condCol] == pair[0]].values
            b1 = data['NLI'][data[condCol] == pair[1]].values
            U1, p = mannwhitneyu(a1, b1, nan_policy = 'omit')
            # U1, p = ranksums(a1, b1, nan_policy = 'omit')
            pvals.append(p)
    
        annotator = Annotator(ax = ax, pairs = pairs, x=condCol,  y="NLI_Plot", data=NComps)
        annotator.configure(text_format="simple", color = lineColor, loc = 'outside')
        annotator.set_pvalues(pvals).annotate(line_offset_to_group = 0.10)
    
    if labels != []:
        plt.xticks(xticks, labels, **plotChars)
        
    plt.xticks(**plotChars)
    plt.yticks(**plotChars)
    plt.legend(fontsize = 15, labelcolor='linecolor')
    
    plt.ylim(0,100)
    return fig, ax, pvals


def plotNLI(fig, ax, data, condCat, condCol, pairs, labels = [],  palette = ['#b96a9b', '#d6c9bc', '#92bda4'], 
            colorScheme = 'black', **plotChars):
        
    if colorScheme == 'black':
        plt.style.use('dark_background')
        fontColor = '#ffffff'
        lineColor = '#ffffff'
    else:
        plt.style.use('default')
        fontColor = '#000000'
        lineColor = '#000000'
        
    NComps = data.groupby(condCol)['NLI_Plot'].count().reindex(condCat, axis = 0).reset_index()
    linear = data[data.NLI_Plot=='linear'].groupby(condCol)['NLI_Plot'].count().reindex(condCat, axis = 0).reset_index().fillna(0)
    nonlinear = data[data.NLI_Plot=='non-linear'].groupby(condCol)['NLI_Plot'].count().reindex(condCat, axis = 0).reset_index().fillna(0)
    intermediate = data[data.NLI_Plot=='intermediate'].groupby(condCol)['NLI_Plot'].count().reindex(condCat, axis = 0).reset_index().fillna(0)

    linear['NLI_Plot'] = [i / j * 100 for i,j in zip(linear['NLI_Plot'], NComps['NLI_Plot'])]
    nonlinear['NLI_Plot'] = [i / j * 100 for i,j in zip(nonlinear['NLI_Plot'], NComps['NLI_Plot'])]
    intermediate['NLI_Plot'] = [i / j * 100 for i,j in zip(intermediate['NLI_Plot'], NComps['NLI_Plot'])]

    y1 = linear['NLI_Plot'].values
    y2 = intermediate['NLI_Plot'].values
    y3 = nonlinear['NLI_Plot'].values
    N = NComps['NLI_Plot'].values

    nonlinear['NLI_Plot'] = linear['NLI_Plot'] + nonlinear['NLI_Plot'] + intermediate['NLI_Plot']
    intermediate['NLI_Plot'] = intermediate['NLI_Plot'] + linear['NLI_Plot']

    sns.barplot(x=condCol,  y="NLI_Plot", data=nonlinear, color=palette[0],  ax = ax, order = condCat)
    sns.barplot(x=condCol,  y="NLI_Plot", data=intermediate, color=palette[1], ax = ax, order = condCat)
    sns.barplot(x=condCol,  y="NLI_Plot", data=linear, color=palette[2], ax = ax, order = condCat)

    if condCol == 'compNum':
        xticks = condCat  - 1
    else:
        xticks = np.arange(len(condCat))

    for xpos, ypos, yval in zip(xticks, y1/2, y1):
        plt.text(xpos, ypos, "%.1f"%yval + '%', ha="center", va="center", color = '#000000', fontsize = 16)
    for xpos, ypos, yval in zip(xticks, y1+y2/2, y2):
        plt.text(xpos, ypos, "%.1f"%yval+ '%', ha="center", va="center", color = '#000000', fontsize = 16)
    for xpos, ypos, yval in zip(xticks, y1+y2+y3/2, y3):
        plt.text(xpos, ypos, "%.1f"%yval+ '%', ha="center", va="center", color = '#000000', fontsize = 16)
    # add text annotation corresponding to the "total" value of each bar
    for xpos, ypos, yval in zip(xticks, y1+y2+y3+0.5, N):
        plt.text(xpos, ypos, "N=%d"%yval, ha="center", va="bottom", fontsize = 16)
        
    pvals = []
    if pairs != None:
        for pair in pairs:
            a1 = data['NLI'][data[condCol] == pair[0]].values
            b1 = data['NLI'][data[condCol] == pair[1]].values
            U1, p = mannwhitneyu(a1, b1, nan_policy = 'omit')
            # U1, p = ranksums(a1, b1, nan_policy = 'omit')
            pvals.append(p)
    
        annotator = Annotator(ax = ax, pairs = pairs, x=condCol,  y="NLI_Plot", data=linear, order = condCat)
        annotator.configure(text_format="simple", color = lineColor)
        annotator.set_pvalues(pvals).annotate()


    texts = ["Nonlinear", "Intermediate", "Linear"]
    patches = [mpatches.Patch(color=palette[i], label="{:s}".format(texts[i]) ) for i in range(len(texts)) ]
    
    if labels != []:
        plt.xticks(xticks, labels, **plotChars)
        
    plt.xticks(**plotChars)
    plt.yticks(**plotChars)
    plt.tight_layout()
    # plt.legend(handles = patches, loc = 'best', fontsize = 15, labelcolor='linecolor')
    plt.show()
    
    return fig, ax, pvals

def plotNLI_V0(fig, ax, data, condCat, condCol, pairs, palette = ['#b96a9b', '#d6c9bc', '#92bda4'], 
            colorScheme = 'black', setOffset = 0):
        
    if colorScheme == 'black':
        plt.style.use('dark_background')
        fontColor = '#ffffff'
        lineColor = '#ffffff'
    else:
        plt.style.use('default')
        fontColor = '#000000'
        lineColor = '#000000'
        
    linear = []
    nonlinear = []
    intermediate = []
    N = []
    frac = []
    dfStats = {}
    
    for i in condCat:
        frac = data[data[condCol] == i]
        dfStats.update({i : np.asarray([np.nan]*len(frac))})
        sLinear = np.sum(frac['NLI_Plot']=='linear')
        sNonlin = np.sum(frac['NLI_Plot']=='non-linear')
        sInter = np.sum(frac['NLI_Plot']=='intermediate')
        linear.append(sLinear)
        nonlinear.append(sNonlin)
        intermediate.append(sInter)
        N.append(sLinear + sNonlin + sInter)
        
        dfStats[i] = frac['NLI'].values
    
    
    N = np.asarray(N)
    linear = (np.asarray(linear)/N)*100
    intermediate = (np.asarray(intermediate)/N)*100
    nonlinear = (np.asarray(nonlinear)/N)*100
    
    plt.bar(condCat, linear, label='linear', color = palette[0])
    plt.bar(condCat, intermediate, bottom = linear, label='intermediate', color = palette[1])
    plt.bar(condCat, nonlinear, bottom = linear+intermediate, label='nonlinear', color = palette[2])
    
    y1 = linear
    y2 = intermediate
    y3 = nonlinear

    
    for xpos, ypos, yval in zip(condCat, y1/2, y1):
        plt.text(xpos, ypos, "%.1f"%yval + '%', ha="center", va="center", fontsize = 25, color = '#000000')
    for xpos, ypos, yval in zip(condCat, y1+y2/2, y2):
        plt.text(xpos, ypos, "%.1f"%yval+ '%', ha="center", va="center",fontsize = 25, color = '#000000')
    for xpos, ypos, yval in zip(condCat, y1+y2+y3/2, y3):
        plt.text(xpos, ypos, "%.1f"%yval+ '%', ha="center", va="center", fontsize = 25, color = '#000000')
    # add text annotation corresponding to the "total" value of each bar
    for xpos, ypos, yval in zip(condCat, y1+y2+y3+0.5, N):
        plt.text(xpos, ypos, "N=%d"%yval, ha="center", va="bottom", fontsize = 20)
    
    x_min, x_max = ax.get_xlim()
    xticks = [(tick - x_min)/(x_max - x_min) for tick in ax.get_xticks()]
    
    y_min, y_max = ax.get_ylim()
    yticks = [(tick - y_min)/(y_max - y_min) for tick in ax.get_yticks()]
    
    try:
        offset = len(pairs)*0.05 + setOffset
        
        for pair in pairs:
            a1 = data['NLI'][data[condCol] == condCat[pair[0]]].values
            b1 = data['NLI'][data[condCol] == condCat[pair[1]]].values
            U1, p = mannwhitneyu(a1, b1, nan_policy = 'omit')
            # U1, p = ranksums(a1, b1)
            
            fig.add_artist(lines.Line2D([xticks[pair[0]], xticks[pair[1]]], [ yticks[-2] - offset, yticks[-2] - offset], color = lineColor))
            xtxt = xticks[pair[0]] + (xticks[pair[1]] - xticks[pair[0]])/2.5
            ytxt = yticks[-2] - offset + 0.01
            ax.annotate('p = ' + str(np.round(p, 4)), xycoords='figure fraction', xy=(xtxt, ytxt), color = 'white')
            offset = offset  - 0.03
            ax.margins(y=0.3)
    except:
        pass
    
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xticks(fontsize=18, color = fontColor)
    plt.yticks(fontsize=30, color = fontColor)
    # plt.legend(bbox_to_anchor=(1.01,0.5), loc='center left', fontsize = 20, labelcolor='linecolor')
    plt.show()
    return fig, ax, dfStats



def plotNLImod(fig, ax, data, palette = sns.color_palette("tab10"),  colorScheme = 'black'):
    
    if colorScheme == 'black':
        plt.style.use('default')
        fig.patch.set_facecolor('black')
        fontColor = '#ffffff'
    else: 
        plt.style.use('default')
        fontColor = '#000000'
        
    
    ax = sns.lineplot(x = 'K_vwc_Full', y = 'Y_vwc_Full', data = data,  hue = ('dateCell', 'first'), 
                      marker = 'o', markersize = 12, markeredgecolor='black' )
    
    return fig, ax


def NLIvFluctu(fig, ax,  palette = sns.color_palette("tab10"), 
                colorScheme = 'black', plottingParams = {}, plotChars = {}):
    
    if colorScheme == 'black':
        plt.style.use('default')
        fig.patch.set_facecolor('black')
        fontColor = '#ffffff'
    else: 
        plt.style.use('default')
        fontColor = '#000000'
        
    sns.scatterplot(**plottingParams)
    
    plt.xticks(**plotChars)
    plt.yticks(**plotChars)
    ax.set_ylabel('Average NLI', **plotChars)
    ax.set_xlabel('Activity', **plotChars)
    return fig, ax

def NLIcorrvFluctu(fig, ax, data, palette = sns.color_palette("tab10"), 
                colorScheme = 'black', plottingParams = {}, plotChars = {}):
    
    if colorScheme == 'black':
        plt.style.use('default')
        fig.patch.set_facecolor('black')
        fontColor = '#ffffff'
    else: 
        plt.style.use('default')
        fontColor = '#000000'
        
    data_nli = data[['cellID', 'compNum', 'NLI_mod']]

    for i in data_nli['cellID'].values:
        diff = []
        dataCell = data_nli[data_nli.cellID == i]
        for j in range(1, dataCell.compNum.max()):
            if j in dataCell.compNum.values and j+1 in dataCell.compNum.values:
                diffNLI = dataCell.NLI_mod[dataCell['compNum'] == j+1].values - dataCell.NLI_mod[dataCell['compNum'] == j].values
                diff.extend(diffNLI)
            else:
                diff.append(np.nan)
                
        diff = np.ma.array(diff, mask=np.isnan(diff), fill_value=None)
        data_nli.loc[dataCell.index, 'NLI_corr'] = [np.sqrt(np.mean(diff**2) / len(dataCell))]*len(dataCell)
        print(len(dataCell))
    data['NLI_corr'] = data_nli['NLI_corr']
    ax = sns.scatterplot(data = data, x = 'normFluctu', y = 'NLI_corr', **plottingParams)
    
    plt.xticks(**plotChars)
    plt.yticks(**plotChars)
    ax.set_ylabel('NLI_correlation', **plotChars)
    ax.set_xlabel('Activity', **plotChars)
    
    return fig, ax, data.copy()

def NLIPairsvFluctu(fig, ax, dfPairs, condCol, condCat, palette = sns.color_palette("tab10"), 
                colorScheme = 'black', plotChars = {}):
    
    if colorScheme == 'black':
        plt.style.use('default')
        fig.patch.set_facecolor('black')
        fontColor = '#ffffff'
    else: 
        plt.style.use('default')
        fontColor = '#000000'
    
    idx = 0
    for i in np.unique(dfPairs['dateCell', 'first'].values):
        toPlot = dfPairs[dfPairs['dateCell', 'first'] == i]
        x1 = toPlot['normFluctu', 'first'][toPlot[condCol, 'first'] == condCat[0]]
        y1 = toPlot['NLI_mod', 'mean'][toPlot[condCol, 'first'] == condCat[0]]
        
        x2 = toPlot['normFluctu', 'first'][toPlot[condCol, 'first'] == condCat[1]]
        y2 = toPlot['NLI_mod', 'mean'][toPlot[condCol, 'first'] == condCat[1]]

        ax.scatter(x1, y1, marker = 'o', color = palette[idx], s = 100)
        ax.scatter(x2, y2, marker = '*',  color = palette[idx], s = 100)
        ax.plot([x1, x2], [y1, y2], color = palette[idx]) 
        
        idx = idx + 1
            
    plt.xticks(**plotChars)
    plt.yticks(**plotChars)
    fig.suptitle(str(condCat), **plotChars)
    ax.set_ylabel('Average NLI', **plotChars)
    ax.set_xlabel('Activity', **plotChars)
    return fig, ax
     
def EvsH0_perCompression(fig, ax, data, condCat, condCol, hueType, xlim = (100, 2*10**3), ylim = (100, 10**5),
          palette = sns.color_palette("tab10"),  colorScheme = 'black'):
    
    
    if colorScheme == 'black':
        plt.style.use('default')
        fig.patch.set_facecolor('black')
        fontColor = '#ffffff'
    else: 
        plt.style.use('default')
        fontColor = '#000000'

    idx = 0
    
    if hueType == 'cellID':
        N = len(data['cellID'].unique())
        palette = palette
        sns.scatterplot(data = data, x = 'H0_vwc_Full', y = 'E_eff', palette = palette, hue = hueType, s = 100)
        for m in condCat:
            eqnText = ''
            toPlot = data[(data[condCol] == m)]
            x, y = toPlot['H0_vwc_Full'].values, toPlot['E_eff'].values
            
            try:
                params, results = ufun.fitLineHuber((np.log(x)), np.log(y))
                k = np.exp(params[0])
                a = params[1]
                
                fit_x = np.linspace(np.min(x), np.max(x), 50)
                fit_y = k * fit_x**a
                
                ax.plot(fit_x, fit_y, label =  eqnText, linestyle = '--', linewidth = 6,
                        color = 'black')
            except:
                pass
            
        plt.legend(fontsize = 12, ncol = int(np.round(N/2)))

    if hueType == 'NLI_Plot':
        palette = ['#b96a9b', '#92bda4']
        mechanicsType = ['non-linear', 'linear']
        for m in mechanicsType:
            eqnText = ''
            toPlot = data[(data['NLI_Plot'] == m)]
            x, y = toPlot['H0_vwc_Full'].values, toPlot['E_eff'].values
            
            params, results = ufun.fitLineHuber((np.log(x)), np.log(y))
            k = np.exp(params[0])
            a = params[1]
            
            fit_x = np.linspace(np.min(x), np.max(x), 50)
            fit_y = k * fit_x**a
            
            pval = results.pvalues[1] # pvalue on the param 'a'
            eqnText += " Y = {:.1e} * X^{:.1f}".format(k, a)
            eqnText += "\np-val = {:.3f}".format(pval)
            ax.scatter(x , y, color = palette[idx], label = m, s = 100)
            ax.plot(fit_x, fit_y, label =  eqnText, linestyle = '--', linewidth = 6,
                    color = palette[idx])
            ax.plot(fit_x, fit_y, label =  eqnText, linestyle = '--', linewidth = 8,
                    color = 'k')
            
            
            idx = idx + 1
            
        plt.legend(fontsize = 12, ncol = len(mechanicsType))
        
    elif hueType == condCol:
        palette = palette
        for m in condCat:
            eqnText = ''
            toPlot = data[(data[condCol] == m)]
            toPlot = toPlot.dropna(subset=['H0_vwc_Full', 'E_eff'])
            x, y = toPlot['H0_vwc_Full'].values, toPlot['E_eff'].values
            
            try:
                params, results = ufun.fitLineHuber((np.log(x)), np.log(y))
                k = np.exp(params[0])
                a = params[1]
                
                fit_x = np.linspace(np.min(x), np.max(x), 50)
                fit_y = k * fit_x**a
                
                pval = results.pvalues[1] # pvalue on the param 'a'
                eqnText += " Y = {:.1e} * X^{:.1f}".format(k, a)
                eqnText += "\np-val = {:.3f}".format(pval)
                
                ax.plot(fit_x, fit_y, label =  eqnText, linewidth = 8,
                        color = 'k')
                ax.plot(fit_x, fit_y, label =  eqnText, linewidth = 6,
                        color = palette[idx])

            except:
                pass
            
            ax.scatter(x , y, color = palette[idx], label = m, s = 100, alpha = 0.5)
            idx = idx + 1
            
        # plt.legend(fontsize = 12, ncol = len(condCat))

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_ylabel('E_effective (Pa)', fontsize=30, color = fontColor)
    ax.set_xlabel('BestH0 (nm)', fontsize=30, color = fontColor)
    plt.xticks(fontsize=25, color = fontColor)
    plt.yticks(fontsize=25, color = fontColor)

    x_ticks = [100, 250, 500, 1000, 1500]
    ax.set_xticks(x_ticks, labels =x_ticks, fontsize=25, color = fontColor)

    y_ticks = [100, 1000, 5000, 10000, 50000, 10**5]
    ax.set_yticks(y_ticks, labels =y_ticks, fontsize=25, color = fontColor)

    plt.show()
    return fig, ax


def boxplot_perCompressionLog(fig, ax, condCat, hueType = None, palette = sns.color_palette("tab10"), labels = [],
                           pairs = None, colorScheme = 'black', plotType = 'swarm',
                           plottingParams = {}, plotChars = {}):
    
    if colorScheme == 'black':
        plt.style.use('default')
        fig.patch.set_facecolor('black')
        fontColor = plotChars['color']
    else: 
        plt.style.use('default')
        fontColor = plotChars['color']
    
    measure = plottingParams['y']
    condCol = plottingParams['x']
    df = plottingParams['data']
    
    if hueType == 'cellID':
        N = len(df['cellID'].unique())
        if plotType == 'swarm':
            ax = sns.swarmplot(hue = hueType, palette = palette, **plottingParams) 
            ax = sns.boxplot(data = df, x = condCol, y = measure, color = 'grey',  order = condCat,
                             medianprops={"color": 'darkred', "linewidth": 2},
                             boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.1})
            
                             
        elif plotType == 'violin':
            ax = sns.violinplot(hue = hueType, palette = palette,
                                inner_kws=dict(box_width=15, whis_width=2, color=".8"), 
                                **plottingParams) 
            
        plt.legend(loc=2, prop={'size': 7}, ncol = int(np.round(N/2)))
    
    elif hueType == 'NLI_Plot':
        palette = ['#CB799C', '#d6c9bc', '#79CBA8']
        if plotType == 'swarm':
            ax = sns.swarmplot(hue = hueType, palette = palette, **plottingParams) 
            ax = sns.boxplot(data = df, x = condCol, y = measure, color = 'grey',  order = condCat,
                             medianprops={"color": 'darkred', "linewidth": 2},
                             boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.1})
            
        elif plotType == 'violin':
            ax = sns.violinplot(hue = hueType, palette = palette,
                                inner_kws=dict(box_width=15, whis_width=2, color=".8"), 
                                **plottingParams) 
            
        plt.legend(loc=2, prop={'size': 16}, ncol = 3)
        
    elif hueType != 'NLI_Plot' and hueType != 'cellID':
        palette = palette
        if plotType == 'swarm':
            ax = sns.swarmplot(palette = palette, hue = hueType, **plottingParams) 
            ax = sns.boxplot(data = df, x = condCol, y = measure, color = 'grey',  order = condCat,
                             medianprops={"color": 'darkred', "linewidth": 2},
                             boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.1})
        elif plotType == 'violin':
            ax = sns.violinplot(hue = hueType, palette = palette,
                                inner_kws=dict(box_width=15, whis_width=2, color=".8"), 
                                **plottingParams) 
            
        plt.legend(loc=2, prop={'size': 16}, ncol = len(condCat))
        
        
    
    # if pairs != None:
    #     annotator = Annotator(ax = ax, pairs = pairs, x=condCol,  y=measure, data=df, order = condCat)
    #     annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside', 
    #                         show_test_name = False, color = '#000000')
    #     annotator.apply_and_annotate()
    
    pvals = []
    if pairs != None:
        for pair in pairs:
            a1 = df[measure][df[condCol] == pair[0]].values
            b1 = df[measure][df[condCol] == pair[1]].values
            U1, p = mannwhitneyu(a1, b1, nan_policy = 'omit')
            # U1, p = ranksums(a1, b1, nan_policy = 'omit')
            pvals.append(p)
    
        annotator = Annotator(ax = ax, pairs = pairs, x=condCol,  y=measure, data=df, order = condCat)
        annotator.configure(text_format="simple", color = '#000000', fontsize = plotChars['fontsize'])
        annotator.set_pvalues(pvals).annotate()
        
    if labels != []:
        xticks = np.arange(len(condCat))
        plt.xticks(xticks, labels, **plotChars)
        
    plt.xticks(**plotChars)
    plt.yticks(**plotChars)
    plt.ylabel(measure, **plotChars)
    plt.xlabel(condCol, **plotChars)
    
    return fig, ax

def boxplot_perCompression(fig, ax, condCat, hueType = None, palette = sns.color_palette("tab10"), labels = [],
                           pairs = None, colorScheme = 'black', plotType = 'swarm',
                           plottingParams = {}, plotChars = {}):
    
    if colorScheme == 'black':
        plt.style.use('default')
        fig.patch.set_facecolor('black')
        fontColor = plotChars['color']
    else: 
        plt.style.use('default')
        fontColor = plotChars['color']
    
    measure = plottingParams['y']
    condCol = plottingParams['x']
    df = plottingParams['data']
    
    if hueType == 'cellID':
        N = len(df['cellID'].unique())
        if plotType == 'swarm':
            ax = sns.swarmplot(hue = hueType, palette = palette, **plottingParams) 
            ax = sns.boxplot(data = df, x = condCol, y = measure, color = 'grey',  order = condCat,
                             medianprops={"color": 'darkred', "linewidth": 2},
                             boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.1})
            
                             
        elif plotType == 'violin':
            ax = sns.violinplot(hue = hueType, palette = palette,
                                inner_kws=dict(box_width=15, whis_width=2, color=".8"), 
                                **plottingParams) 
            
        plt.legend(loc=2, prop={'size': 7}, ncol = int(np.round(N/2)))
    
    elif hueType == 'NLI_Plot':
        palette = ['#CB799C', '#d6c9bc', '#79CBA8']
        if plotType == 'swarm':
            ax = sns.swarmplot(hue = hueType, palette = palette, **plottingParams) 
            ax = sns.boxplot(data = df, x = condCol, y = measure, color = 'grey',  order = condCat,
                             medianprops={"color": 'darkred', "linewidth": 2},
                             boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.1})
            
        elif plotType == 'violin':
            ax = sns.violinplot(hue = hueType, palette = palette,
                                inner_kws=dict(box_width=15, whis_width=2, color=".8"), 
                                **plottingParams) 
            
        plt.legend(loc=2, prop={'size': 16}, ncol = 3)
        
    elif hueType != 'NLI_Plot' and hueType != 'cellID':
        palette = palette
        if plotType == 'swarm':
            ax = sns.swarmplot(palette = palette, hue = hueType, **plottingParams) 
            ax = sns.boxplot(data = df, x = condCol, y = measure, color = 'grey',  order = condCat,
                             medianprops={"color": 'darkred', "linewidth": 2},
                             boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.1})
        elif plotType == 'violin':
            ax = sns.violinplot(hue = hueType, palette = palette,
                                inner_kws=dict(box_width=15, whis_width=2, color=".8"), 
                                **plottingParams) 
            
        plt.legend(loc=2, prop={'size': 16}, ncol = len(condCat))
        
        
    
    # if pairs != None:
    #     annotator = Annotator(ax = ax, pairs = pairs, x=condCol,  y=measure, data=df, order = condCat)
    #     annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside', 
    #                         show_test_name = False, color = '#000000')
    #     annotator.apply_and_annotate()
    
    pvals = []
    if pairs != None:
        for pair in pairs:
            a1 = df[measure][df[condCol] == pair[0]].values
            b1 = df[measure][df[condCol] == pair[1]].values
            U1, p = mannwhitneyu(a1, b1, nan_policy = 'omit')
            # U1, p = ranksums(a1, b1, nan_policy = 'omit')
            pvals.append(p)
    
        annotator = Annotator(ax = ax, pairs = pairs, x=condCol,  y=measure, data=df, order = condCat)
        annotator.configure(text_format="simple", color = '#000000', fontsize = plotChars['fontsize'])
        annotator.set_pvalues(pvals).annotate()
        
    if labels != []:
        xticks = np.arange(len(condCat))
        plt.xticks(xticks, labels, **plotChars)
        
    plt.xticks(**plotChars)
    plt.yticks(**plotChars)
    plt.ylabel(measure, **plotChars)
    plt.xlabel(condCol, **plotChars)
    
    return fig, ax

def boxplot_perCell(fig, ax, condCat, hueType = None, palette = sns.color_palette("tab10"), 
                    labels = [], pairs = None,  colorScheme = 'black', plottingParams = {}, 
                    plotChars = {}):
    
    if colorScheme == 'black':
        plt.style.use('default')
        fig.patch.set_facecolor('black')
        fontColor = plotChars['color']
    else: 
        plt.style.use('default')
        fontColor = plotChars['color']
        
    measure = plottingParams['y']
    condCol = plottingParams['x']
    avgDf = plottingParams['data']
    # avgDf = avgDf.reindex(condCat, axis = 0).reset_index()s
    
    if hueType == 'cellID':
        N = len(avgDf['cellID', 'first'].unique())
        palette = palette
        ax = sns.swarmplot(hue = (hueType, 'first'), palette = palette, **plottingParams) 
        plt.legend(loc=2, prop={'size': 7}, ncol = int(np.round(N/2)))
    
    elif hueType == 'NLI_Plot':
        palette = ['#CB799C', '#d6c9bc', '#79CBA8']
        ax = sns.swarmplot(hue = (hueType, 'first'), palette = palette, **plottingParams) 
        plt.legend(loc=2, prop={'size': 7}, ncol = 3)
        
    elif hueType == None:
        palette = palette
        ax = sns.swarmplot(palette = palette, **plottingParams) 
        plt.legend(loc=2, prop={'size': 7}, ncol = len(condCat))
        
    ax = sns.boxplot(data = avgDf, x = condCol, y = measure, color = 'grey',  order = condCat,
                     medianprops={"color": 'darkred', "linewidth": 2},
                     boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.1})
    
    
    annotator = Annotator(ax = ax, pairs = pairs, x=condCol,  y=measure, data=avgDf, order = condCat)
    annotator.configure(test='Mann-Whitney', text_format='simple', fontsize = plotChars['fontsize'],
                        loc='inside', show_test_name = False, color = '#000000')
    annotator.apply_and_annotate()
    
    if labels != []:
        xticks = np.arange(len(condCat))
        plt.xticks(xticks, labels, **plotChars)
        
    plt.xticks(**plotChars)
    plt.legend(fontsize = 16)
    plt.yticks(**plotChars)
    plt.ylabel(measure, **plotChars)
    plt.xlabel(condCol, **plotChars)
    
    return fig, ax

def EvH0_LogCellAvg(fig, ax, avgDf, condCat, condCol, hueType, h_ref = 400,
                  palette = sns.color_palette("tab10"), plotChars = {},
                  errorbar = False, pairs = None, colorScheme = 'black'):
    
    if colorScheme == 'black':
        plt.style.use('default')
        fig.patch.set_facecolor('black')
        fontColor = '#ffffff'
    else: 
        plt.style.use('default')
        fontColor = '#000000'
    
    
    h = ('bestH0_log', 'mean')
    e = ('E_eff_log', 'mean')
    
    
    # if errorbar == True:
    #     ax.errorbar(avgDf[h], avgDf[e], yerr = avgDf[('E_eff', 'std')], xerr = avgDf[('H0_vwc_Full', 'std')], 
    #                       linestyle='', color = '#a6a6a6') 
    
    idx = 0
    
    if hueType == 'NLI_Plot':
        mechanicsType = ['non-linear', 'linear']
        palette = ['#b96a9b', '#92bda4']
        for m in mechanicsType:
            eqnText = ''
            toPlot = avgDf[(avgDf[('NLI_Plot', 'first')] == m)]
            x, y = toPlot[h].values, toPlot[e].values
            
            params, results = ufun.fitLineHuber((np.log(x)), np.log(y))
            k = np.exp(params[0])
            t = params[1]
            
            fit_x = np.linspace(np.min(x), np.max(x), 50)
        
            fit_y = k * fit_x**t
            pval = results.pvalues[1] # pvalue on the param 'a'
            eqnText += " Y = {:.1e} * X^{:.1f}".format(k, t)
            eqnText += "\np-val = {:.3f}".format(pval)
            ax.scatter(x , y, color = palette[idx], label = m, s = 100, edgecolors="k", alpha = 0.7)
            ax.plot(fit_x, fit_y, label = eqnText, linestyle = '--', color = palette[idx])
            
            toPlot[('E_norm', 'logAvg')] = toPlot[e] * (h_ref/toPlot[h])**t
            
            # Xfit, Yfit = np.log(toPlot[('bestH0_log', 'mean')].values), np.log(toPlot[('E_norm', 'logAvg')].values)
            
            # [b, a], results = ufun.fitLine(Xfit, Yfit)
            # A, k = np.exp(b), a
            # R2 = results.rsquared
            # pval = results.pvalues[1]
            # Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
            # Yplot = A * Xplot**k
            # ax.plot(Xplot, Yplot, ls = '--', c = palette[idx], lw = 6.0,
            #         label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
            #                 f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
    
            idx = idx + 1
        
        plt.legend(fontsize = 10, ncol = len(mechanicsType))
        
    elif hueType == 'cellID':
        N = len(avgDf[('cellID', 'first')].unique())
        # palette = (sns.color_palette("Paired", np.round(N/2)) + sns.color_palette("husl",  np.round(N/2)))
        sns.scatterplot(ax = ax[0], data = avgDf, x = h, y = e, hue = ('cellName', 'first'), 
                          s = 150, palette = palette, edgecolor = 'k')
        
        for m in condCat:
            eqnText = ''
            toPlot = avgDf[(avgDf[(condCol, 'first')] == m)]
            x, y = toPlot[h].values, toPlot[e].values
            
            params, results = ufun.fitLineHuber((np.log(x)), np.log(y))
            k = np.exp(params[0])
            t = params[1]
            
            fit_x = np.linspace(np.min(x), np.max(x), 50)
        
            fit_y = k * fit_x**t
            pval = results.pvalues[1] # pvalue on the param 'a'
            eqnText += " Y = {:.1e} * X^{:.1f}".format(k, t)
            eqnText += "\np-val = {:.3f}".format(pval)
            ax.plot(fit_x, fit_y, label = eqnText, lw = 6, linestyle = '--', color = 'black')
            
            idx = idx + 1

            
        plt.legend(fontsize = 10, ncol = np.round(N/2))
          
    else:
        palette = palette
        # sns.scatterplot(ax = ax[0], data = avgDf, x = h, y = e, hue = (condCol, 'first'), 
        #                s = 150, palette = palette, edgecolor = 'k')
        
        for m in condCat:
            eqnText = ''
            toPlot = avgDf[(avgDf[(condCol, 'first')] == m)]

            x, y = toPlot[h].values, toPlot[e].values
            
            params, results = ufun.fitLineHuber((np.log(x)), np.log(y))
            k = np.exp(params[0])
            t = params[1]
            
            fit_x = np.linspace(np.min(x), np.max(x), 50)
        
            fit_y = k * fit_x**t
            pval = results.pvalues[1] # pvalue on the param 'a'
            eqnText += " Y = {:.1e} * X^{:.1f}".format(k, t)
            eqnText += "\np-val = {:.3f}".format(pval)
            ax.scatter(x , y, color = palette[idx], label = m, s = 100, alpha = 0.7)
            ax.plot(fit_x, fit_y,  lw = 6, linestyle = '--', color = 'k')

            ax.plot(fit_x, fit_y, label = eqnText, lw = 6.1, linestyle = '--', color = palette[idx])

            toPlot[('E_norm', 'logAvg')] = toPlot[e] * (h_ref/toPlot[h])**t

            
            # Xfit, Yfit = np.log(toPlot[('bestH0_log', 'mean')].values), np.log(toPlot[('E_norm', 'logAvg')].values)
            avgDf.loc[(avgDf[(condCol, 'first')] == m), ('E_norm', 'logAvg')] = toPlot[('E_norm', 'logAvg')]
            
            
            # [b, a], results = ufun.fitLine(Xfit, Yfit)
            # A, k = np.exp(b), a
            # R2 = results.rsquared
            # pval = results.pvalues[1]
            # Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
            # Yplot = A * Xplot**k
            # ax[1].plot(Xplot, Yplot,  lw = 6.1, linestyle = '--', color = 'k')

            # ax[1].plot(Xplot, Yplot, ls = '--', c = palette[idx], lw = 6.0,
            #         label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
            #                 f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
            
            idx = idx + 1
        
        plt.legend(fontsize = 10, ncol = len(condCat))
       
    # for i in range(len(ax)):
    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax[i].set_ylim(100, 10**5)
    # ax[i].set_xlim(100, 2*10**3)
    ax.set_ylabel('E_effective (Pa)', fontsize=30, color = fontColor)
    ax.set_xlabel('BestH0 (nm)', fontsize=30, color = fontColor)
    

    x_labels = np.asarray([100, 250, 500, 1000, 1500, 2500])
    x_ticks = np.log10(np.asarray(x_labels))
    ax.set_xticks(x_ticks, labels = x_labels,**plotChars)

    y_labels = np.asarray([100, 500, 2500, 10000, 20000, 50000])
    y_ticks = np.log10(np.asarray(y_labels))
    ax.set_yticks(y_ticks, labels = y_labels,**plotChars)
    
    plt.xticks(fontsize=25, color = fontColor)
    plt.yticks(fontsize=25, color = fontColor)
    plt.tight_layout()
    plt.show()
    return fig, ax, avgDf.copy()

def EvH0_wCellAvg(fig, ax, avgDf, condCat, condCol, hueType, h_ref = 400,
                  palette = sns.color_palette("tab10"), plotChars = {},
                  errorbar = False, pairs = None, colorScheme = 'black'):
    
    if colorScheme == 'black':
        plt.style.use('default')
        fig.patch.set_facecolor('black')
        fontColor = '#ffffff'
    else: 
        plt.style.use('default')
        fontColor = '#000000'
    
    
    h = ('H0_vwc_Full', 'mean')
    e = ('E_eff', 'wAvg')
    
    
    if errorbar == True:
        ax.errorbar(avgDf[h], avgDf[e], yerr = avgDf[('E_eff', 'std')], xerr = avgDf[('H0_vwc_Full', 'std')], 
                          linestyle='', color = '#a6a6a6') 
    
    idx = 0
    
    if hueType == 'NLI_Plot':
        mechanicsType = ['non-linear', 'linear']
        palette = ['#b96a9b', '#92bda4']
        for m in mechanicsType:
            eqnText = ''
            toPlot = avgDf[(avgDf[('NLI_Plot', 'first')] == m)]
            x, y = toPlot[h].values, toPlot[e].values
            
            params, results = ufun.fitLineHuber((np.log(x)), np.log(y))
            k = np.exp(params[0])
            t = params[1]
            
            fit_x = np.linspace(np.min(x), np.max(x), 50)
        
            fit_y = k * fit_x**t
            pval = results.pvalues[1] # pvalue on the param 'a'
            eqnText += " Y = {:.1e} * X^{:.1f}".format(k, t)
            eqnText += "\np-val = {:.3f}".format(pval)
            ax[0].scatter(x , y, color = palette[idx], label = m, s = 100, edgecolors="k", alpha = 0.5)
            ax[0].plot(fit_x, fit_y, label = eqnText, linestyle = '--', color = palette[idx])
            
            toPlot[('E_norm', 'wAvg')] = toPlot[e] * (h_ref/toPlot[h])**t
    
            ax[1].scatter(x = x, y = toPlot[('E_norm', 'wAvg')].values, s = 150,
                          color = palette[idx], edgecolor = 'k', alpha = 0.5)
            
            Xfit, Yfit = np.log(toPlot[('H0_vwc_Full', 'mean')].values), np.log(toPlot[('E_norm', 'wAvg')].values)
            
            [b, a], results = ufun.fitLine(Xfit, Yfit)
            A, k = np.exp(b), a
            R2 = results.rsquared
            pval = results.pvalues[1]
            Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
            Yplot = A * Xplot**k
            ax[1].plot(Xplot, Yplot, ls = '--', c = palette[idx], lw = 6.0,
                    label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                            f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
    
            idx = idx + 1
        
        plt.legend(fontsize = 10, ncol = len(mechanicsType))
        
    elif hueType == 'cellID':
        N = len(avgDf[('cellID', 'first')].unique())
        # palette = (sns.color_palette("Paired", np.round(N/2)) + sns.color_palette("husl",  np.round(N/2)))
        sns.scatterplot(ax = ax[0], data = avgDf, x = h, y = e, hue = ('cellName', 'first'), 
                          s = 150, palette = palette, edgecolor = 'k', alpha = 0.5)
        
        for m in condCat:
            eqnText = ''
            toPlot = avgDf[(avgDf[(condCol, 'first')] == m)]
            x, y = toPlot[h].values, toPlot[e].values
            
            params, results = ufun.fitLineHuber((np.log(x)), np.log(y))
            k = np.exp(params[0])
            t = params[1]
            
            fit_x = np.linspace(np.min(x), np.max(x), 50)
        
            fit_y = k * fit_x**t
            pval = results.pvalues[1] # pvalue on the param 'a'
            eqnText += " Y = {:.1e} * X^{:.1f}".format(k, t)
            eqnText += "\np-val = {:.3f}".format(pval)
            ax[0].plot(fit_x, fit_y, label = eqnText, lw = 6, linestyle = '--', color = 'black')
            
            idx = idx + 1

            
        plt.legend(fontsize = 10, ncol = np.round(N/2))
          
    else:
        palette = palette
        # sns.scatterplot(ax = ax[0], data = avgDf, x = h, y = e, hue = (condCol, 'first'), 
        #                s = 150, palette = palette, edgecolor = 'k')
        
        for m in condCat:
            eqnText = ''
            toPlot = avgDf[(avgDf[(condCol, 'first')] == m)]

            x, y = toPlot[h].values, toPlot[e].values
            
            params, results = ufun.fitLineHuber((np.log(x)), np.log(y))
            k = np.exp(params[0])
            t = params[1]
            
            fit_x = np.linspace(np.min(x), np.max(x), 50)
        
            fit_y = k * fit_x**t
            pval = results.pvalues[1] # pvalue on the param 'a'
            eqnText += " Y = {:.1e} * X^{:.1f}".format(k, t)
            eqnText += "\np-val = {:.3f}".format(pval)
            ax[0].scatter(x , y, color = palette[idx], label = m, s = 100,  alpha = 0.5)
            ax[0].plot(fit_x, fit_y,  lw = 6, linestyle = '--', color = 'k')

            ax[0].plot(fit_x, fit_y, label = eqnText, lw = 6.1, linestyle = '--', color = palette[idx])

            toPlot[('E_norm', 'wAvg')] = toPlot[e] * (h_ref/toPlot[h])**t
        
            ax[1].scatter(x = x, y = toPlot[('E_norm', 'wAvg')].values, s = 150, alpha = 0.5,
                          color = palette[idx])
            
            Xfit, Yfit = np.log(toPlot[('H0_vwc_Full', 'mean')].values), np.log(toPlot[('E_norm', 'wAvg')].values)
            avgDf.loc[(avgDf[(condCol, 'first')] == m), ('E_norm', 'wAvg')] = toPlot[('E_norm', 'wAvg')]
            
            
            [b, a], results = ufun.fitLine(Xfit, Yfit)
            A, k = np.exp(b), a
            R2 = results.rsquared
            pval = results.pvalues[1]
            Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
            Yplot = A * Xplot**k
            ax[1].plot(Xplot, Yplot,  lw = 6.1, linestyle = '--', color = 'k')

            ax[1].plot(Xplot, Yplot, ls = '--', c = palette[idx], lw = 6.0,
                    label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + f'\nA = {A:.1e}' + f'\nk  = {k:.2f}' + \
                            f'\n$R^2$  = {R2:.2f}' + f'\np-val = {pval:.3f}')
            idx = idx + 1
        
        plt.legend(fontsize = 10, ncol = len(condCat))
       
    for i in range(len(ax)):
        ax[i].set_yscale('log')
        ax[i].set_xscale('log')
        ax[i].set_ylim(100, 10**5)
        ax[i].set_xlim(100, 2*10**3)
        ax[i].set_ylabel('E_effective (Pa)', fontsize=30, color = fontColor)
        ax[i].set_xlabel('BestH0 (nm)', fontsize=30, color = fontColor)
        
    
        x_ticks = [100, 250, 500, 1000, 1500]
        ax[i].set_xticks(x_ticks, labels =x_ticks, fontsize=25, color = fontColor)
    
        y_ticks = [100, 1000, 5000, 10000, 50000]
        ax[i].set_yticks(y_ticks, labels =y_ticks, fontsize=25, color = fontColor)
    
    plt.xticks(fontsize=25, color = fontColor)
    plt.yticks(fontsize=25, color = fontColor)
    plt.tight_layout()
    plt.show()
    return fig, ax, avgDf.copy()

def pointplot_cellAverage(fig, ax, dfPairs, condCatPoint, pairedCells, marker, palette = sns.color_palette("tab10"),
                          ylim = (0,1000), pairs = None, normalize = False, styleType = None,
                          colorScheme = 'black', test = 'two-sided', hueType = ('dateCell', 'first'),
                          plottingParams = {}, plotChars = {}):
    
    if colorScheme == 'black':

        fig.patch.set_facecolor('black')
        fontColor = '#ffffff'
        
    else: 
        plt.style.use('default')
        fontColor = '#000000'
    
    measure = plottingParams['y']
    condCol = plottingParams['x']
    
    if normalize == False :
        # plt.sytle.use('seaborn')
        pvals = []
        for pair in pairs:
            a1 = dfPairs[measure][dfPairs[condCol] == pair[0]].values
            b1 = dfPairs[measure][dfPairs[condCol] == pair[1]].values
            res = wilcoxon(a1, b1, alternative=test, zero_method = 'wilcox')
            pvals.append(res[1])

        ax = sns.lineplot(palette = palette, data = dfPairs, hue = hueType, style = styleType,
                          marker = 'o', **plottingParams)

    if normalize == True:
        dfPairs['normMeasure', marker] = [np.nan]*len(dfPairs)
        pvals = []
        for pair in pairs:
            for cell in pairedCells:
                c1 = dfPairs[(measure)][(dfPairs[('dateCell', 'first')] == cell) & (dfPairs[condCol] == pair[0])].values
                c2 = dfPairs[(measure)][(dfPairs[('dateCell', 'first')] == cell) & (dfPairs[condCol] == pair[1])].values
                ratio = np.round(c2/c1, 3)
                
                dfPairs.loc[(dfPairs[('dateCell', 'first')] == cell) & (dfPairs[condCol] == pair[0]), ('normMeasure', marker)] = 1
                dfPairs.loc[(dfPairs[('dateCell', 'first')] == cell) & (dfPairs[condCol] == pair[1]), ('normMeasure', marker)] = ratio

            
            a1 = dfPairs[('normMeasure', marker)][dfPairs[condCol] == pair[0]].values
            b1 = dfPairs[('normMeasure', marker)][dfPairs[condCol] == pair[1]].values
            res = wilcoxon(a1, b1, alternative=test, zero_method = 'wilcox')
            pvals.append(res[1])
        
        ax = sns.lineplot(x = condCol, y = ('normMeasure',marker), data = dfPairs,  hue = hueType, 
                          marker = 'o',  markersize = 15, markeredgecolor = 'black', palette = palette)
        
        plt.axhline(y = 1, linestyle = '--', color = 'k')
        
        # annotator = Annotator(ax = ax, pairs = pairs, x=condCol,  y=('normMeasure', marker), 
        #                       order = condCatPoint, data=dfPairs)
        
        # annotator.configure(text_format="simple", color = '#000000', size = plotChars['fontsize'])
        # annotator.set_pvalues(pvals).annotate() 
        
    
    plt.xlim((-0.5,1.5))
    fig.suptitle('p = ' + str(np.round(pvals[0], 4)) + ' | ' + test , **plotChars)
    plt.xticks(**plotChars)
    plt.yticks(**plotChars)
    plt.ylabel(measure, **plotChars)
    plt.xlabel(condCol, **plotChars)
    plt.legend(loc=2, prop={'size': 7}, ncol = 6)
    plt.ylim(ylim)
    plt.show()

    return fig, ax, pvals, dfPairs.copy()

def KvY(condCat, condCol, pairs, plottingParams = {}, plotChars = {}):
   
    data = plottingParams['data']
    pointSize = plottingParams['s']
    ylim = -1000, 10000
    
    for pair in pairs:
        
        df = data[(data[condCol] == pair[0]) | (data[condCol] == pair[1])]
        df = df[df.NLI_Plot != 'intermediate']
        
        sns.jointplot(data=df, x="Y_vwc_Full", y="K_vwc_Full", hue=condCol)
        # sns.scatterplot(data=df, x="Y_vwc_Full", y="K_vwc_Full", hue=condCol)
        # sns.scatterplot(data=df, x="Y_vwc_Full", y="K_vwc_Full", hue='NLI_Plot')
        # plt.yscale('log')
        # plt.xscale('log')
        
        plt.title(pair)
    
        # xticks = np.arange(ylim[0], ylim[1], 1000)
        # yticks = np.arange(ylim[0], ylim[1], 1000)

        # plt.xticks(xticks)
        # plt.yticks(yticks)
        
        # plt.tight_layout()
        
        
    return


def KvY_V0(hueType, condCat, condCol, palette = sns.color_palette("tab10"), 
        plottingParams = {}, plotChars = {}):

    nColsSubplot = 2
    nRowsSubplot = ((len(condCat)-1) // nColsSubplot) + 1
                    
    # fig, axes = plt.subplots(nRowsSubplot, nColsSubplot,
    #                         figsize = (13,9))
    
    data = plottingParams['data']
    pointSize = plottingParams['s']
    
    for i in range(len(condCat)):
        # fig, ax = plt.subplots(figsize = (13,9))
        df = data[data[condCol] == condCat[i]]
                         
       
        # colSp = (i) % nColsSubplot
        # rowSp = (i) // nColsSubplot
        
        # if nRowsSubplot == 1:
        #     ax = axes[colSp]
        # elif nRowsSubplot >= 1:
        #     ax = axes[rowSp,colSp]
        
        if hueType == 'NLI_Plot':
            palette = ['#b96a9b', '#d6c9bc', '#92bda4']
            hue_order =  ['non-linear', 'intermediate', 'linear']
            
            sns.jointplot(data=df, x="Y_vwc_Full", y="K_vwc_Full", hue="NLI_Plot", hue_order = hue_order,
                          palette = palette)

        elif hueType == 'cellID':
            sns.jointplot(data=df, x="Y_vwc_Full", y="K_vwc_Full", hue="NLI_Plot", hue_order = hue_order,
                          palette = palette)
            
        
        plt.title(condCat[i])
        
        # x_ticks = [100, 1000, 5000, 10000, 50000, 10**5]
        # ax.set_xticks(ticks = x_ticks, labels = x_ticks, **plotChars)
        
        # y_ticks = [100, 1000, 5000, 10000, 50000, 10**5]
        # ax.set_yticks(ticks = y_ticks, labels =y_ticks, **plotChars)
                    
    return



