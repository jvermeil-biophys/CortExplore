# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:51:36 2023

@author: anumi
"""

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
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)

import GraphicStyles as gs
import UtilityFunctions as ufun
import TrackAnalyser_V2 as taka

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

plotLabels = 25
plotTicks = 18 
plotTitle = 25
plotLegend = 25
fontColour = '#000000'


#%% DEFAULT settings

DEFAULT_centers = [ii for ii in range(100, 1550, 50)]
DEFAULT_halfWidths = [50, 75, 100]

DEFAULT_fitSettings = {# H0
                       'methods_H0':['Dimitriadis'],
                       'zones_H0':['%f_20'],
                       'method_bestH0':'Dimitriadis',
                       'zone_bestH0':'%f_20',
                       # Global fits
                       'doChadwickFit' : True,
                       'doDimitriadisFit' : False,
                       # Local fits
                       'doStressRegionFits' : True,
                       'doStressGaussianFits' : True,
                       'centers_StressFits' : DEFAULT_centers,
                       'halfWidths_StressFits' : DEFAULT_halfWidths,
                       'doNPointsFits' : True,
                       'nbPtsFit' : 13,
                       'overlapFit' : 3,
                       'doLogFits' : True,
                       'nbPtsFitLog' : 10,
                       'overlapFitLog' : 5,
                       }

DEFAULT_crit_nbPts = 8 # sup or equal to
DEFAULT_crit_R2 = 0.6 # sup or equal to
DEFAULT_crit_Chi2 = 1 # inf or equal to
DEFAULT_str_crit = 'nbPts>{:.0f} - R2>{:.2f} - Chi2<{:.1f}'.format(DEFAULT_crit_nbPts, 
                                                                   DEFAULT_crit_R2, 
                                                                   DEFAULT_crit_Chi2)

DEFAULT_fitValidationSettings = {'crit_nbPts': DEFAULT_crit_nbPts, 
                                 'crit_R2': DEFAULT_crit_R2, 
                                 'crit_Chi2': DEFAULT_crit_Chi2,
                                 'str': DEFAULT_str_crit}


#%% Functions

def makeDirs():
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

def makeBoxPlotParametric(dfAllCells, condCol, labels, order, condSelection, colorPalette = None,
                          average = False, stats = False):
    
    fig1, ax = plt.subplots(1, 1, figsize = (15,10))
    df = dfAllCells

    if FIT_MODE == 'linlin':
        params = ['A', 'B']
        ax_titles = ['Slope (A)', 'Linear Intercept (B)']
    elif FIT_MODE == 'loglog':
        params = ['a', 'q']
        ax_titles = ['Exponent (a)', 'Coefficient (q)']
        
    condPairs = condSelection

    condCol = condCol
    df = df[[params[0], params[1], condCol, 'chosenIntercept', 'chosenInterceptStress', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

    df = df.drop_duplicates()
    df = df.dropna()

    plotIntercept = True

    if plotIntercept:
        y1, y2 = df[params[0]],  df['chosenIntercept']
        figExt = 'Intercept'+str(interceptStress)
        axtitle = 'Intercept at '+str(interceptStress)+'Pa'
    else:
        y1, y2 = df[params[0]],  df[params[1]]
        figExt = 'Coeff'
        axtitle = ax_titles[1]

    if average == True:
        group_by_cell = df.groupby(['cellID'])
        df_average = group_by_cell.agg({params[0]:['var', 'std', 'mean', 'count'], 'cellCode':'first', condCol:'first'})
        df_average = df_average[df_average['cellCode'].duplicated(keep = False)]
    
        condCol_box = condCol, 'first'
        x_box = df_average[(condCol_box)]
        hue_box = df_average[('cellCode', 'first')]
        y_box = df_average[(params[0], 'mean')]
       
        x = df[(condCol)]
        hue = df[('cellCode')]
        y = df[(params[0])]
        df = df_average
        
        sns.boxplot(x = x_box, y = y_box, data=df, ax = ax, order = order, 
                            medianprops={"color": 'darkred', "linewidth": 2},\
                            boxprops={"color" : 'grey',  "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})

        sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')

        
    else:
        
        x = df[condCol]
        order = order
        if colorPalette == None:
            sns.boxplot(x = x, y = y1, data=df, ax = ax, order = order,
                                medianprops={"color": 'darkred', "linewidth": 2},\
                                boxprops={"color" : 'grey',  "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})
        else:
            sns.boxplot(x = x, y = y1, data=df, ax = ax, order = order,palette = colorPalette,
                                medianprops={"color": 'darkred', "linewidth": 2},\
                                boxprops={"edgecolor": 'k',"linewidth": 2, 'alpha' : 0.8})

        sns.swarmplot(x = x, y = y1, data=df,  order = order,linewidth = 1, color = 'k', ax = ax, edgecolor='k') #, hue = 'cellCode')
    

    if stats == True:
        # condPairs = [condSelection[1], condSelection[0]]
        box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
        addStat_df(ax = ax, data = df, box_pairs = box_pairs, param = params[0], cond = condCol,\
                    test = 'ranksum_greater')

    if labels != []:
        ax.set_xticklabels(labels, fontsize = plotTicks)
        
    ax.yaxis.set_tick_params(labelsize=plotTicks)
    ax.xaxis.set_tick_params(labelsize=plotTicks)
    handles, labels = ax.get_legend_handles_labels()
    
    if average == True:
        newLabels = []
        for i in labels:
            count1 = df[(params[0], 'count')][(df['cellCode', 'first'] == i) & (df[condCol, 'first'] == condSelection[0])].values
            count2 = df[(params[0], 'count')][(df['cellCode', 'first'] == i) & (df[condCol, 'first'] == condSelection[1])].values
            newLabels.append('{}, No.: {}, {}'.format(i, str(count1), str(count2)))
    
        ax.legend(handles, newLabels)

    # fig1.suptitle(str(dates) + '_'+str(condPairs))
    plt.tight_layout()
    # plt.savefig(pathBoxPlots + '/' +params[0]+'_'+ str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
    plt.show()
    return fig1, ax
    
    

def makeBoxPlots(dfAllCells, measure, condCol, labels, order, condSelection, colorPalette = None, removeLegend = False,
                 average = False, stats = False):
    fig1, ax = plt.subplots(1, 1, figsize = (10,10))
    fig1.patch.set_facecolor('black')
    plt.style.use('default')

    df = dfAllCells

    if FIT_MODE == 'linlin':
        params = ['A', 'B']
        ax_titles = ['Slope (A)', 'Linear Intercept (B)']
    elif FIT_MODE == 'loglog':
        params = ['a', 'q']
        ax_titles = ['Exponent (a)', 'Coefficient (q)']
        

    condPairs = condSelection

    
    df = df[[measure, condCol, 'cellCode', 'cellId', 'compNum']]
    df = df.drop_duplicates()
    df = df.dropna()
    
    
    if measure == 'E_Chadwick':
        
        df_og = df
        # print(df_og)
        group_by_cell = df_og.groupby(['cellId'])   
        
        df_average = group_by_cell.agg({measure:['mean'], 'cellId':'first', condCol : 'first'})
        
        print(df_average)       
        
        condCol_box = condCol, 'first'
        x_box = df_average[condCol_box]
        hue_box = df_average[('cellId', 'first')]
        y_box = np.unique(df_average[(measure, 'mean')])
        

        x = df_og[condCol]
        hue = df_og[('cellCode')]
        y = df_og[(measure)]
        order = order
        
        if average == False:
            sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, palette = colorPalette,
                                medianprops={"color": 'darkred', "linewidth": 2},\
                                boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.9})
                
            sns.swarmplot(x = x_box, y = y_box, data=df_average, order = order,linewidth = 1, ax = ax, color = 'k', edgecolor='k') #, hue = 'cellCode')
            test = 'Mann-Whitney'
            
        elif average == True:
            sns.boxplot(x = x, y = y, data=df, ax = ax, order = order,
                                medianprops={"color": 'darkred', "linewidth": 2},\
                                boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})
            
            sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')
        
            test = 'ranksum_greater'
            
        if stats:
            box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
            print(box_pairs)
            addStat_df(ax = ax, data = df, box_pairs = box_pairs, param = measure, cond = condCol_box,
                       test = test)
            
    else:
        
        x = df[condCol]
        y = measure
        order = order
        
        # df = df[df['R2Fit'] > 0.70]
        
        # if mode == 'Compare activation' and condCol == 'manip':
        #     cellCodes = df['cellCode'][df['manip'] == manips[1]].values
        #     df = df[df['cellCode'].isin(cellCodes)]

        if average == False:
            sns.boxplot(x = x, y = y, data=df, ax = ax, order = order, palette = colorPalette,
                                medianprops={"color": 'darkred', "linewidth": 2},\
                                boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.9})
                
            sns.swarmplot(x = x, y = y, data=df, order = order,linewidth = 1, ax = ax, color = 'k', edgecolor='k') #, hue = 'cellCode')
        
        elif average == True:
            
            sns.boxplot(x = x, y = y, data=df, ax = ax, order = order,
                                medianprops={"color": 'darkred', "linewidth": 2},\
                                boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})
            
            sns.pointplot(x = x, y = y, data=df, order = order, hue = 'cellCode', ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')
               
            
        if stats:
            box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
            print(box_pairs)
            addStat_df(ax = ax, data = df, box_pairs = box_pairs, param = measure, cond = condCol)

    
    if labels != []:
        ax.set_xticklabels(labels, fontsize = plotTicks)

    color = "#ffffff"
    ax.yaxis.set_tick_params(labelsize=plotTicks, color = color)
    ax.xaxis.set_tick_params(labelsize=plotTicks, color = color)
    ax.set_ylabel(measure, fontsize = 20,  color = color)
    # ax.set_xlabel(' ', fontsize = 20, color = color)
    ax.tick_params(axis='both', colors= color) 
    # ax.set_ylim(0, 1000)
    
    if removeLegend == True:
        ax.get_legend().remove()

    plt.tight_layout()
    # plt.ylim(0,1200)
    plt.show()
    return fig1, ax


def plot2Params(data, Filters, fitSubDir, fitType, interceptStress, FIT_MODE, pathFits, plot = True):


    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for k in range(0, len(Filters)):
        globalFilter = globalFilter & Filters[k]
    data_f = data[globalFilter]

    data_ff = taka.getFitsInTable(data_f, fitsSubDir, fitType=fitType, filter_fitID=fitId)

    # Filter the table
    # data_ff = data_ff[(data_ff['fit_center'] >= Sinf) & (data_ff['fit_center'] <= Ssup)]    
    data_ff = data_ff.drop(data_ff[data_ff['fit_error'] == True].index)
    data_ff = data_ff.drop(data_ff[data_ff['fit_K'] < 0].index)
    data_ff = data_ff.drop(data_ff[data_ff['fit_K'] > 80000].index)
    data_ff = data_ff.drop(data_ff[data_ff['E_Chadwick'] > 80000].index)
    data_ff = data_ff.dropna(subset = ['fit_ciwK'])

    cells = data_ff['cellID'].unique()

    # Making fits and new table with fit parameter values

    dfAllCells = data_ff
    dfAllCells['fit_K_kPa'] = dfAllCells['fit_K']/1000
    dfAllCells['fit_ciwK_kPa'] = dfAllCells['fit_ciwK']/1000
    dfAllCells['q'] = [np.nan]*len(dfAllCells)
    dfAllCells['a'] = [np.nan]*len(dfAllCells)
    dfAllCells['chosenIntercept'] = [np.nan]*len(dfAllCells)
    dfAllCells['chosenInterceptStress'] = [np.nan]*len(dfAllCells)

    dfAllCells['A'] = [np.nan]*len(dfAllCells)
    dfAllCells['B'] = [np.nan]*len(dfAllCells)
    dfAllCells['R2Fit'] = [np.nan]*len(dfAllCells)

    for i in range(len(cells)):
        allSelectedCells = cells[i]
        print(gs.BLUE + allSelectedCells + gs.NORMAL)
        filtr = dfAllCells[dfAllCells['cellID'] == allSelectedCells].index.values
        idx = dfAllCells.index.isin(filtr)
        
        # try:
        N = dfAllCells['compNum'][idx].max()
        nColsSubplot = 4
        nRowsSubplot = (N // nColsSubplot) + 2
        
        if plot: 
            fig, axes = plt.subplots(nRowsSubplot, nColsSubplot, figsize = (20,20))
            _axes = []
            
            for ax_array in axes:
                for ax in ax_array:
                    _axes.append(ax)

        colors = gs.colorList30[:N]
        for ii in range(N):
            filtrComp = dfAllCells[(dfAllCells['cellID'] == allSelectedCells) & (dfAllCells['compNum'] == ii)].index.values
            idxComp = dfAllCells.index.isin(filtrComp)
            
            legendText = ''
            Y = dfAllCells['fit_K_kPa'][idxComp]
            X = dfAllCells['fit_center'][idxComp]
            Yerr = dfAllCells['fit_ciwK_kPa'][idxComp]
            validity = dfAllCells['fit_error'][idxComp]
            
            if plot == True:
                ax = _axes[ii]
               
            if len(X) >= 3:
                posValues = ((X > 0) & (Y > 0))
                X, Y = X[posValues], Y[posValues]
                weights = (Y / Yerr)**2
        
                #Weighted linear regression
                if FIT_MODE == 'loglog':
                    logX = np.log(X)
                    logY = np.log(Y)
                    logYerr = np.log(Yerr)
                    weights = (Y / Yerr)**2
                    params, results = ufun.fitLineWeighted(logX, logY, weights)
                    q = params[0]
                    a = params[1]
                    R2 = results.rsquared
                    pval = results.pvalues[1] # pvalue on the param 'a'
                    dfAllCells['q'][idxComp] = [q]*len(logX)
                    dfAllCells['a'][idxComp] = [a]*len(logX)
                    dfAllCells['R2Fit'][idxComp] = [R2]*len(X)
                    legendText += " Y = {:.3e} * X^{:.3f}".format(q, a)
                    fitX = np.linspace(np.min(X), np.max(X), 100)
                    fitY = np.exp(q) * fitX**a
                    chosenIntercept = np.exp(q) * interceptStress**a
                    dfAllCells['chosenIntercept'][idxComp] = [chosenIntercept]*len(X)
                    dfAllCells['chosenInterceptStress'][idxComp] = [interceptStress]*len(X)
                    label = legendText
                    if plot:
                        ax.errorbar(X, Y, yerr = Yerr, marker = 'o', ms = 5, ecolor = colors[ii])
                        ax.plot((fitX), (fitY), '--', lw = '1', color = 'red', zorder = 4, label = legendText)
                        ax.set_yscale('log')
                        ax.set_xscale('log')
                        ax.set_ylim(0, 12)
                        ax.set_xlim(0, 1500)
                        ax.set_title('CompNum: ' + str(ii))
                        ax.legend(loc = 'upper left', prop={'size': 15})
                        fig.suptitle(allSelectedCells, fontsize = 15)
                        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    
                elif FIT_MODE == 'linlin':
                    
                    params, results = ufun.fitLineWeighted(X, Y, weights)
                    A = params[1]
                    B = params[0]
                    R2 = results.rsquared
                    pval = results.pvalues[1] # pvalue on the param 'a'
                    dfAllCells['A'][idxComp] = [A]*len(X)
                    dfAllCells['B'][idxComp] = [B]*len(X)
                    dfAllCells['R2Fit'][idxComp] = [R2]*len(X)
                    legendText += " Y = {:.3f} X + {:.3f}".format(A, B)
                    legendText += " \n R2 = {:.2f}".format(R2)
                    fitX = np.round(np.linspace(np.min(X), np.max(X), 100))
                    fitY = A * fitX + B
                    chosenIntercept = A * interceptStress + B
                    dfAllCells['chosenIntercept'][idxComp] = [chosenIntercept]*len(X)
                    dfAllCells['chosenInterceptStress'][idxComp] = [interceptStress]*len(X)
                    label = legendText
                    if plot:
                        ax.errorbar(X, Y, yerr = Yerr, marker = 'o', ms = 5, ecolor = colors[ii])
                        ax.plot(fitX, fitY, '--', lw = '1', color = 'red', zorder = 4, label = legendText)
                        ax.set_ylim(0, 25)
                        ax.set_xlim(0, 1200)
                        ax.set_title('CompNum: ' + str(ii))
                        ax.legend(loc = 'upper left', prop={'size': 15})
                        fig.suptitle(allSelectedCells, fontsize = 15)
                    
        
        if plot:
            plt.tight_layout()
            plt.show()
            plt.savefig(pathFits + '/' + allSelectedCells)   
            
    return dfAllCells


def closest_value_index(input_list, input_value):
 
  arr = np.asarray(input_list)
 
  i = (np.abs(arr - input_value)).argmin()
 
  return i

def plotAllH0(data, fig, ax, fitsSubDir = '', Filters = [], maxH0 = np.Inf, condCols = [], 
              co_order = [], box_pairs = [], AvgPerCell = True,
              stats = False, statMethod = 'Mann-Whitney', stressBoxPlot = 1):
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

    data_ff = taka.getAllH0InTable(data_f, fitsSubDir = fitsSubDir)
    
    # Filter the table   
    data_ff = data_ff.drop(data_ff[data_ff['allH0_error'] == True].index)
    data_ff = data_ff.drop(data_ff[data_ff['allH0_H0'] >= maxH0].index)
    
    
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
    # fig, ax = plt.subplots(1, 1, figsize = (5*NCond, 5))

    if stressBoxPlot == 0:
        sns.boxplot(x=condCol, y='allH0_H0', data=data_ff, ax=ax,
                    width = 0.5, showfliers = False, order= co_order, 
                    medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},
                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
#                   boxprops={"color": color, "linewidth": 0.5},
                    whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                    capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2})
                    # scaley = scaley)
                        
    elif stressBoxPlot == 1:
        sns.boxplot(x=condCol, y='allH0_H0', data=data_ff, ax=ax, 
                    width = 0.5, showfliers = False, order= co_order, 
                    medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
#                   boxprops={"color": color, "linewidth": 0.5},
                    whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                    capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2})
                    # scaley = scaley)
                    
    if stressBoxPlot == 2:
        sns.boxplot(x=condCol, y='allH0_H0', data=data_ff, ax=ax, 
                    width = 0.5, showfliers = False, order= co_order, 
                    medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 4},
                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
#                   boxprops={"color": color, "linewidth": 0.5},
                    whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                    capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4})
                    # scaley = scaley)
                
    # if stats:
    #     if len(box_pairs) == 0:
    #         box_pairs = makeBoxPairs(co_order)
    #     addStat_df(ax, data_ff, box_pairs, Parameters[k], CondCol, test = statMethod)
    
    sns.swarmplot(x=condCol, y='allH0_H0', data=data_ff, ax=ax, order = co_order)

    ax.set_xlabel('')
    ax.set_ylabel('H0 (nm)')
    ax.tick_params(axis='x', labelrotation = 10)
    ax.yaxis.grid(True)
    if ax.get_yscale() == 'linear':
        ax.set_ylim([0, ax.get_ylim()[1]])

    # Make output    
    output = (fig, ax)

    return(output)

def plotPopKS(data, fig, ax, fitsSubDir = '',  fitType = 'stressRegion', fitWidth=75, Filters = [], condCol = '', 
              c_min = 0, c_max = np.Inf, legendLabels = [],
              mode = 'wholeCurve', scale = 'lin', printText = True,
              returnData = 0, returnCount = 0):
    
    # fig, ax = plt.subplots(1,1, figsize = (9,6))

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
    data_ff = taka.getFitsInTable(data_f, fitsSubDir, fitType=fitType, filter_fitID=fitId)
    
    # Filter the table
    data_ff = data_ff[(data_ff['fit_center'] >= Sinf) & (data_ff['fit_center'] <= Ssup)]    
    data_ff = data_ff.drop(data_ff[data_ff['fit_error'] == True].index)
    data_ff = data_ff.drop(data_ff[data_ff['fit_K'] < 0].index)
    data_ff = data_ff.dropna(subset = ['fit_ciwK'])
    
    conditions = np.array(data_ff[condCol].unique())

    
    # Compute the weights
    data_ff['weight'] = (data_ff['fit_K']/data_ff['fit_ciwK'])**2
    
    #### NOTE
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
        centers = np.array([ce for ce in centers if ((ce<c_max) and (ce>c_min))])
        
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
    
    if len(legendLabels) == 0:
        legendLabels = conditions
        
    for i in range(len(conditions)):
        co = conditions[i]
        df = data_agg[data_agg[condCol] == co]
        color = styleDict1[co]['color']
        marker = styleDict1[co]['marker']
        label = styleDict1[co]['label']
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
            ax.set_ylim([0, 18])
                
        elif scale == 'log':
            if co == conditions[0]:
                texty = Kavg**0.95
            else:
                texty = texty**0.98
            ax.set_yscale('log')
        
        
        cellCount = len(np.unique(data_ff['cellID'][data_ff[condCol] == co].values))
        # label = '{} | NCells = {} | NComp = {}'.format(co, cellCount, sum(N))
        
        
        label = '{} | NCells = {}'.format(label, cellCount)
            
        # weighted means -- weighted ste 95% as error
        ax.errorbar(centers, Kavg/1000, yerr = q*Kste/1000, 
                    color = color, lw = 2, marker = marker, markersize = 8, mec = 'k',
                    ecolor = color, elinewidth = 1.5, capsize = 6, capthick = 1.5, 
                    label = label)
        
        # ax.set_title('K(s) - All compressions pooled')
        color = '#ffffff'
        ax.legend(loc = 'upper left', fontsize = 11)
        ax.set_xlabel('Stress (Pa)', fontsize = 20,  color = color)
        ax.set_ylabel('K (kPa)', fontsize = 20, color = color)
        ax.tick_params(axis='both', colors= color) 
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.grid(visible=True, which='major', axis='y') #, color = '#3a3b3b')
        
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


    return(output, count_df)

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
        c1 = data[data[cond] == bp[0]][param].values
        c2 = data[data[cond] == bp[1]][param].values

        if test=='Mann-Whitney':
            res = st.mannwhitneyu(c1,c2)
            statistic, pval = res.statistic, res.pvalue
        elif test=='Wilcox_2s':
            res = st.wilcoxon(c1,c2, alternative = 'two-sided')
            
        elif test=='Wilcox_greater':
            res = st.wilcoxon(c1,c2, alternative = 'greater')
            statistic, pval = res.statistic, res.pvalue[0]
        elif test=='Wilcox_less':
            res = st.wilcoxon(c1,c2, alternative = 'less')
            statistic, pval = res.statistic, res.pvalue[0]
        elif test=='ranksum_greater':
            res = st.ranksums(c1,c2, alternative = 'greater')
            statistic, pval = res.statistic, res.pvalue[0]
        elif test=='ranksum_less':
            res = st.ranksums(c1,c2, alternative = 'less')
            statistic, pval = res.statistic, res.pvalue[0]
        elif test=='ranksum_2s':
            res = st.ranksums(c1,c2, alternative = 'two-sided')
        elif test=='t-test':
            res = st.ttest_ind(c1,c2)
                
        
        print(pval)
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
        



# %% Mechanics

# %%%% Update the table

# tka.computeGlobalTable_meca(task = 'updateExisting', fileName = 'Global_MecaData_AJ', 
#                             save = True, PLOT = True, source = 'Python')

# %%%% Specific experiments

newFitSettings = {# H0
                       'methods_H0':['Chadwick', 'Dimitriadis'],
                       'zones_H0':['%f_15'],
                       'method_bestH0':'Chadwick',
                       'zone_bestH0':'%f_15',
                        # 'centers_StressFits' : [ii for ii in range(100, 1550, 20)],
                        # 'halfWidths_StressFits' : [75],
                        # 'Plot_Ratio':False,
                       }

newfitValidationSettings = {'crit_nbPts': 6}


Task = '23-10-25'

#'22-08-26_M7 & 22-08-26_M5 & 22-08-26_M10 & 22-08-26_M1 & 22-08-26_M3' # For instance '22-03-30 & '22-03-31'
fitsSubDir = 'Chad_f15_23-10-25_23-11-06_HeLa'

GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', \
                            fileName = fitsSubDir,
                            save = True, PLOT = True, source = 'Python', fitSettings = newFitSettings,\
                               fitValidationSettings = newfitValidationSettings, fitsSubDir = fitsSubDir) # task = 'updateExisting'

#%% Plots for 23-05-10
GlobalTable = taka.getMergedTable('Chad_f15_23-05-10_23-05-16')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']

fitsSubDir = 'Chad_f15_23-05-10_23-05-16'


fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75


#%%%% Non linearity

########## Declare variables ##########
dates = ['23-05-10']
manips = ['M3', 'M4'] #, 'M3'] 
labels = ['Not activated', 'Activated']

stressRange = '200_550'
interceptStress = 250
plot = False
FIT_MODE = 'loglog'
condCol = 'manip'
order = None

condSelection = manips

styleDict1 =  {'M1':{'color': "#5c337f",'marker':'o'},
                'M2':{'color': gs.colorList40[23],'marker':'o'},
                'M3':{'color': gs.colorList40[20],'marker':'o'},
                'M4':{'color': "#008fb1",'marker':'o'},
                }

###########################################

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

# plt.style.use('seaborn-v0_8-bright')
# plt.style.use('seaborn')

data = data_main

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            # (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

# selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')


out1, cellDf1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir,  legendLabels = labels, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)


mainFig1, mainAx1, exportDf1, countDf1 = out1

plt.legend(fontsize = 20, loc = 'upper left')


mainFig2, mainAx2 = plt.subplots(1,1)
mainFig2.patch.set_facecolor('black')


out2, cellDf2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = condCol, mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

# plt.legend(fontsize = 15, loc = 'upper left')

plt.ylim(0,10)
plt.tight_layout()
# plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'250-600.png')  
plt.show()

dfAllCells = plot2Params(data, Filters, fitsSubDir, fitType, interceptStress, FIT_MODE, pathFits, plot = plot)

#%%%% Box plots, thickness
measure = 'surroundingThickness'
makeBoxPlots(dfAllCells, measure, condCol, labels, order, condSelection, average = True)

#%%%% Box plots, 2Parameter fit

makeBoxPlotParametric(dfAllCells, condCol, labels, order, condSelection, average = False)

#%%%% Thickness/bestH0 in time
plt.style.use('seaborn')
globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

fig1, axes = plt.subplots(1,1, figsize=(15,10))
fig1.patch.set_facecolor('black')

# sns.set_palette(sns.color_palette("tab10"))

flatui = ["#000000", "#1111EE"]
sns.set_palette(flatui)

x = (data_f['compNum']-1)*20
sns.lineplot(x = x, y = 'E_Chadwick', data = data_f, hue = 'manip', linewidth = 3)

# fig1.suptitle('[15mT = 500pN] bestH0 (nm) vs. Time (secs)', fontsize = 25)
plt.xticks(fontsize=30, color = '#ffffff')
plt.yticks(fontsize=30, color = '#ffffff')
plt.xlabel('Time (secs)', fontsize = 25, color = '#ffffff')
plt.ylabel('Cortex Thickness (nm)', fontsize = 25, color = '#ffffff')
plt.ylim(0, 1200)
plt.xlim(0, 120)

#%%%% Variable over time (comp number)

palette = ["#000000", "#0000FF"]
y = 'E_Chadwick'
param1 = 'M3'
param2 = 'M4'

# if y == 'E_Chadwick':
df = pd.DataFrame({y : dfAllCells[y], 'compNum': dfAllCells['compNum'], 
                   'cellId' : dfAllCells['cellID'], 'manip' : dfAllCells['manip']})
df = df.drop_duplicates()

cells = np.unique(cellDf2['cellID'])
cells = np.asarray([i for i in cells if param1 in i])

mainFig1, mainAx1 = plt.subplots(4,5, figsize = (15,10))
mainFig1.patch.set_facecolor('black')
mainAx1 = mainAx1.flatten()

for i in range(len(cells)):
    print(cells[i])
    dfPlot = df[(df['cellId'] == cells[i]) | (df['cellId'] == cells[i].replace(param1, param2))]
    x, y = 'compNum', y
    
    sns.lineplot(x = x, y = y, data = dfPlot, hue = 'manip', ax = mainAx1[i], marker = 'o')
    if i != 0:
        mainAx1[i].get_legend().remove()
    mainAx1[i].set_title(cells[i], color = '#ffffff', size = 10)
    mainAx1[i].tick_params(axis='both', colors = '#ffffff')
    if y == 'E_Chadwick':
        mainAx1[i].set_ylim(0,15000)
        mainAx1[i].set_xlim(0,11)
        # mainFig1.text(0.04, 0.5, y, va='center', rotation='vertical', size =  20, 
        #               color = '#ffffff')
        mainFig1.suptitle(y + ' vs Compression Number', color = '#ffffff')

    elif y == 'surroundingThickness':
        mainAx1[i].set_ylim(0,1200)
        mainAx1[i].set_xlim(0,11)
        # mainFig1.text(0.04, 0.5, y, va='center', rotation='vertical', size =  20, 
        #               color = '#ffffff')
        mainFig1.suptitle(y + ' vs Compression Number', color = '#ffffff')

    
# mainFig1.text(0.5, 0.04, 'Compression no.', ha='center', color = '#ffffff', size = 20)
plt.tight_layout()
plt.show()

#%% Plots for all experiments with 3T3 optoPRG

GlobalTable = taka.getMergedTable('Chad_f15_All3T3PRGControl_23-05-16')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']

fitsSubDir = 'Chad_f15_All3T3PRGControl_23-05-16'

nBins = 11
bins = np.linspace(0, 2000, nBins)
data_main['H0_Bin'] = np.digitize(GlobalTable['bestH0'], bins, right = True)
data_main['Thickness_Bin'] = np.digitize(GlobalTable['surroundingThickness'], bins, right = True)

fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75


#%%%% Non linearity

########## Declare variables ##########
dates = ['22-12-07']
manipIDs = ['23-01-23_M1', '23-05-10_M3']
manips = ['M4']
labels = []

stressRange = '200_550'
interceptStress = 250
plot = False
FIT_MODE = 'loglog'
condCol = 'Thickness_Bin'
order = np.linspace(1,len(bins), len(bins))

condSelection = ['none']

# styleDict1 =  {'M1':{'color': "#5c337f",'marker':'o'},
#                 'M2':{'color': gs.colorList40[23],'marker':'o'},
#                 'M3':{'color': gs.colorList40[20],'marker':'o'},
#                 
#                 }

# styleDict1 =  {'none':{'color': "#5c337f",'marker':'o'},
#                 }

# styleDict1 =  {'22-12-07':{'color': "#5c337f",'marker':'o'},
#                 '23-02-02':{'color': gs.colorList40[23],'marker':'o'},
#                 '23-05-10':{'color': gs.colorList40[20],'marker':'o'},
#                 '23-01-23':{'color': "#008fb1",'marker':'o'},
                
#                 }


styleDict1 = {}
keys = np.rint(np.linspace(1,nBins,nBins)).astype(int)
values = gs.colorList40[10:nBins+10]

for key, value, label in zip(keys, values, bins):
    key = int(key)
    styleDict1[key] = {'color' : value, 'marker' : 'o', 'label' : "<" + str(label)}
print(styleDict1)

###########################################

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

data = data_main

Filters = [(data['validatedThickness'] == True),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            # (data['UI_Valid'] == True),
            (data['bestH0'] <= 2000),
            # (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manip'].apply(lambda x : x in manips)),
            (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

# selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')


out1, cellDf1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir,  legendLabels = labels, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

plt.legend(fontsize = 20, loc = 'upper left')


mainFig2, mainAx2 = plt.subplots(1,1)
mainFig2.patch.set_facecolor('black')


out2, cellDf2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = condCol, mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

plt.legend(fontsize = 15, loc = 'upper left')

plt.ylim(0,10)
plt.tight_layout()
# plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'250-600.png')  
plt.show()

# dfAllCells = plot2Params(data, Filters, fitsSubDir, fitType, interceptStress, FIT_MODE, pathFits, plot = plot)

#%%%% Individual K vs. S plots to check averaging

data = data_main

cells = np.unique(cellDf2['cellID'])

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')

styleDict1 = {}

labels = []


for i in range(len(cells)):
    cell = cells[i]
    
    styleDict1[cell] = {'color': gs.colorList40[10+i] ,'marker':'o'}

    Filters = [(data['validatedThickness'] == True),
                # (data['substrate'] == '20um fibronectin discs'), 
                # (data['drug'] == 'none'), 
                (data['bead type'] == 'M450'),
                # (data['UI_Valid'] == True),
                (data['bestH0'] <= 1900),
                # (data['date'].apply(lambda x : x in dates)),
                (data['cellID'].apply(lambda x : x in cells)),
                # (data['manip'].apply(lambda x : x in manips)),
                # (data['manipId'].apply(lambda x : x in manipIDs)),
                ] 
    
out2, cellDf2 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
              fitWidth=75, Filters = Filters, condCol = 'cellID', mode = stressRange, scale = 'lin', printText = False,
                            returnData = 1, returnCount = 1)

#%%%% Box plots, thickness
measure = 'surroundingThickness'
makeBoxPlots(dfAllCells, measure, condCol, labels, order, condSelection, average = False)

#%%%% Box plots, 2Parameter fit

makeBoxPlotParametric(dfAllCells, condCol, labels, order, condSelection, average = False)

#%% Plots for ALL experiments
#Experiments included: 
# 23-05-10 
# 23-02-02 
# 23-01-23 
# 22-12-07 
# 23-03-24 
# 23-03-28 
# 23-04-25 
# 23-04-19
    
# GlobalTable = taka.getMergedTable('GlobalTable_MergeATCC_OptoPRG')
# fitsSubDir = 'GlobalTable_MergeATCC_OptoPRG'

GlobalTable = taka.getMergedTable('Chad_f15_AllExpts_23-05-30')
fitsSubDir = 'Chad_f15_AllExpts_23-05-30'

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
data_main['cellId'] = GlobalTable['cellID']




fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75


#%%%% Non linearity
#All 3T3PRG vs 3T3 optoLARG vs 3T3 WT
# manipIDs = ['22-12-07_M4', '23-02-02_M1', '23-05-10_M3', '23-03-28_M1']

########## Declare variables ##########
manipIDs = ['22-12-07_M1', '23-03-24_M1', '23-02-02_M1', '23-05-10_M3', '23-05-23_M1']
# manipIDs = [ '23-03-24_M1']

dates = ['23-02-02']
stressRange = '200_500'
interceptStress = 250
plot = False
FIT_MODE = 'loglog'
condCol = 'drug'
order = None
plt.style.use('seaborn')


# manipIDs = ['23-04-19_M1', '23-04-19_M2',
#             '23-02-02_M1',
#             '23-05-10_M3', '23-05-10_M4']
# # labels = ['Control-1', '10uM Y27-1',
#           'Control-2', 'GlobalAct-2', 
#           'Control-3', 'GlobalAct-3', '10uM Y27-3',
#           'Control-4', 'GlobalAct-4']
# styleDict1 =  {'23-04-19_M1':{'color': gs.colorList40[21],'marker':'o'},      #Control
#                 '23-04-19_M2':{'color': gs.colorList40[22],'marker':'o'},      #Global Activation
#                 '23-05-10_M3':{'color': gs.colorList40[31],'marker':'o'},     #Control
#                 '23-05-10_M4':{'color': gs.colorList40[32],'marker':'o'},     #Global Activation
#                 '23-04-25_M1':{'color': gs.colorList40[11],'marker':'o'},     #Control
#                 '23-04-25_M2':{'color': gs.colorList40[12],'marker':'o'},     #Global Activation
#                 '23-04-25_M3':{'color': gs.colorList40[14],'marker':'o'},     #Y27 10uM
#                 '23-02-02_M1':{'color': gs.colorList40[21],'marker':'o'},   #Control
#                 '23-02-02_M3':{'color': gs.colorList40[24],'marker':'o'},   #10uM Y27
#                 }


labels = [ '50uM Y27', 'Control',  '10uM Blebbistatin']

# labels = ['10uM Blebbistatin', 'DMSO']
condSelection = ['none', 'activation', 'Y27_10', 'Y27_50', 'Y27', 'dmso', 'blebbi_10']
styleDict1 =  {'none':{'color': "#000000",'marker':'o'},
                'activation':{'color': "#3232ff",'marker':'o'},
                'Y27_10':{'color': "#b266b2",'marker':'o'},
                'Y27_50':{'color': "#800080",'marker':'o'},
                'dmso':{'color': "#666666",'marker':'o'},
                'blebbi_10':{'color': "#008000",'marker':'o'},
                }

# manips = ['M1', 'M3', 'M5', 'M7']
# condSelection = manips
# styleDict1 =  {'M1':{'color': "#000000",'marker':'o'},
#                 'M3':{'color': "#3232ff",'marker':'o'},
#                 'M5':{'color': "#b266b2",'marker':'o'},
#                 'M7':{'color': "#800080",'marker':'o'},
#                 }

###########################################

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


data = data_main

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            # (data['UI_Valid'] == True),
            (data['bestH0'] <= 1600),
            # (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manip'].apply(lambda x : x in manips)),
            (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

# selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')


out1, cellDf1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir,  legendLabels = labels, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

plt.legend(fontsize = 20, loc = 'upper left')


mainFig2, mainAx2 = plt.subplots(1,1)
mainFig2.patch.set_facecolor('black')


out2, cellDf2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = condCol, mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

plt.legend(fontsize = 18, loc = 'upper left')

plt.ylim(0,8)
# plt.tight_layout()
# # plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'250-600.png')  
# plt.show()

dfAllCells = plot2Params(data, Filters, fitsSubDir, fitType, interceptStress, FIT_MODE, pathFits, plot = plot)

#%%%% Individual K vs. S plots to check averaging

data = data_main

cells = np.unique(cellDf2['cellID'])

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')

styleDict1 = {}

labels = []


for i in range(len(cells)):
    cell = cells[i]
    
    styleDict1[cell] = {'color': gs.colorList40[10+i] ,'marker':'o'}

    Filters = [(data['validatedThickness'] == True),
                # (data['substrate'] == '20um fibronectin discs'), 
                # (data['drug'] == 'none'), 
                (data['bead type'] == 'M450'),
                # (data['UI_Valid'] == True),
                (data['bestH0'] <= 1900),
                # (data['date'].apply(lambda x : x in dates)),
                (data['cellID'].apply(lambda x : x in cells)),
                # (data['manip'].apply(lambda x : x in manips)),
                # (data['manipId'].apply(lambda x : x in manipIDs)),
                ] 
    
out2, cellDf2 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
              fitWidth=75, Filters = Filters, condCol = 'cellId', mode = stressRange, scale = 'lin', printText = False,
                            returnData = 1, returnCount = 1)

plt.ylim(0,8)

mainAx1.get_legend().remove()

#%%%% Box plots, thickness
measure = 'E_Chadwick'

# labels = ['Control-1', 'Control-2', 'Control-3', 'Control-4',
#          'GlobalAct-2', 'GlobalAct-3', 'GlobalAct-4', 
#          '10uM Y27-1', '10uM Y27-3']



# order = ['23-02-02_M1', '23-04-19_M1', '23-04-25_M1', '23-05-10_M3',
#          '23-04-19_M2', '23-04-25_M2', '23-05-10_M4',  
#          '23-02-02_M3', '23-04-25_M3']

# colorPalette = [gs.colorList40[21], gs.colorList40[21], gs.colorList40[11], gs.colorList40[31],
#                 gs.colorList40[22],  gs.colorList40[12], gs.colorList40[32],
#                 gs.colorList40[24], gs.colorList40[14]]

labels = []
order=None
colorPalette = [gs.colorList40[21], gs.colorList40[22], gs.colorList40[24]]

df1, df2 = makeBoxPlots(dfAllCells, measure, condCol, labels, order, condSelection, colorPalette, average = False)

#%%%% Box plots, 2Parameter fit

labels = ['Control', 'Global Activation', '10uM Y27']
order=['none', 'activation', 'Y27'] 
fig1, ax = makeBoxPlotParametric(dfAllCells, condCol, labels, order, colorPalette, condSelection, average = False)

ax.set_ylabel('Log-log exponent', fontsize = plotTicks)
ax.set_ylim(-10, 6)


#%% Plots for all experiments compared to 3T3 ATCC WT

GlobalTable = taka.getMergedTable('GlobalTable_MergeATCC_OptoPRG')
fitsSubDir = 'GlobalTable_MergeATCC_OptoPRG'


data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']

nBins = 11
bins = np.linspace(0, 2000, nBins)

data_main['H0_Bin'] = np.digitize(GlobalTable['bestH0'], bins, right = True)
data_main['Thickness_Bin'] = np.digitize(GlobalTable['surroundingThickness'], bins, right = True)


fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75


#%%%% Non linearity
#All 3T3PRG vs 3T3 optoLARG vs 3T3 WT
# manipIDs = ['22-12-07_M4', '23-02-02_M1', '23-05-10_M3', '23-03-28_M1']

########## Declare variables ##########
# dates = ['23-04-19', '23-05-10']
stressRange = '200_550'
interceptStress = 250
plot = False
FIT_MODE = 'loglog'
condCol = 'H0_Bin'
order = None
plt.style.use('seaborn')

celltypes = ['Atcc-2023']
condSelection = ['Atcc-2023', 'optoRhoA']
# '23-01-23_M1', '23-04-19_M1', '23-02-02_M1', '23-05-10_M3',
# manipIDs = ['23-04-19_M1', '23-02-02_M1', '23-05-10_M3', ]



# '23-02-02_M1', '23-05-10_M3', '23-05-23_M1'
# labels = []
# styleDict1 =  {'optoRhoA':{'color': "#000000",'marker':'o'},
#                 'Atcc-2023':{'color': "#6f8d68",'marker':'o'},
#                 'Y27':{'color': gs.colorList40[24],'marker':'o'},
#                 }

styleDict1 = {}
keys = np.rint(np.linspace(1,nBins,nBins)).astype(int)
values = gs.colorList40[10:nBins+10]

for key, value, label in zip(keys, values, bins):
    key = int(key)
    styleDict1[key] = {'color' : value, 'marker' : 'o', 'label' : '>' + str(label)}
print(styleDict1)
###########################################

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

data = data_main

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

Filters = [(data['validatedThickness'] == True),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            # (data['UI_Valid'] == True),
            (data['bestH0'] <= 2000),
            # (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manip'].apply(lambda x : x in manips)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            (data['cell subtype'].apply(lambda x : x in celltypes))
            ]

# selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')


out1, cellDf1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir,  fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

plt.legend(fontsize = 20, loc = 'upper left')


mainFig2, mainAx2 = plt.subplots(1,1)
mainFig2.patch.set_facecolor('black')


out2, cellDf2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = condCol, mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

plt.legend(fontsize = 20, loc = 'upper left')

plt.ylim(0,10)
plt.tight_layout()
# plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'250-600.png')  
plt.show()

# dfAllCells = plot2Params(data, Filters, fitsSubDir, fitType, interceptStress, FIT_MODE, pathFits, plot = plot)

#%%%% Box plots, thickness
measure = 'surroundingThickness'

# labels = ['Control-1', 'Control-2', 'Control-3', 'Control-4',
#          'GlobalAct-2', 'GlobalAct-3', 'GlobalAct-4', 
#          '10uM Y27-1', '10uM Y27-3']


# order = ['23-02-02_M1', '23-04-19_M1', '23-04-25_M1', '23-05-10_M3',
#          '23-04-19_M2', '23-04-25_M2', '23-05-10_M4',  
#          '23-02-02_M3', '23-04-25_M3']

# colorPalette = [gs.colorList40[21], gs.colorList40[21], gs.colorList40[11], gs.colorList40[31],
#                 gs.colorList40[22],  gs.colorList40[12], gs.colorList40[32],
#                 gs.colorList40[24], gs.colorList40[14]]

labels = ['Control', 'Global Activation', '10uM Y27']
order=['none', 'activation', 'Y27']

colorPalette = [gs.colorList40[21], gs.colorList40[22], gs.colorList40[24]]

makeBoxPlots(dfAllCells, measure, condCol, labels, order, condSelection, colorPalette, average = False)

#%%%% Box plots, 2Parameter fit

# labels = ['Control', 'Global Activation', '10uM Y27']
# order=['none', 'activation', 'Y27'] 
fig1, ax = makeBoxPlotParametric(dfAllCells, condCol, labels, order, colorPalette, condSelection, average = False)

ax.set_ylabel('Log-log exponent', fontsize = plotTicks)
ax.set_ylim(-10, 6)

#%% Polarised rear experiments
# Expt 1: 'Chad_f15_23-05-23_23-06-27'
# Expt 2: 'Chad_f15_23-07-12_23-07-19'

GlobalTable = taka.getMergedTable('Chad_f15_23-07-12_23-07-19')
fitsSubDir = 'Chad_f15_23-07-12_23-07-19'

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
data_main['cellId'] = GlobalTable['cellID']

fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75


#%%%% Non linearity
#All 3T3PRG vs 3T3 optoLARG vs 3T3 WT
# manipIDs = ['22-12-07_M4', '23-02-02_M1', '23-05-10_M3', '23-03-28_M1']

########## Declare variables ##########
dates = ['23-07-12']
stressRange = '150_550'
interceptStress = 250
plot = False
FIT_MODE = 'loglog'
condCol = 'manip'
order = None


# manipIDs = ['23-01-23_M1', '23-04-19_M1', '23-02-02_M1', '23-05-10_M3', '23-02-23_M1']
manips = ['M1', 'M2']
condSelection = manips

labels = []
styleDict1 =  {'M1':{'color': gs.colorList40[21],'marker':'o'},
                'M2':{'color': "#000000",'marker':'o'},
                }

###########################################

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
    
pathTSPlots = pathSubDir+'/Timeseries Plots'
if not os.path.exists(pathTSPlots):
    os.mkdir(pathTSPlots)

# plt.style.use('seaborn-v0_8-bright')
# plt.style.use('seaborn')

data = data_main

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    

# allSelectedCells = np.asarray(intersected + controlCells)

# allSelectedCells = ['23-05-23_M3_P1_C6','23-05-23_M3_P2_C4',
#                     '23-05-23_M3_P3_C12', '23-05-23_M3_P1_C9',
#                     '23-05-23_M3_P2_C2', 
#                     '23-05-23_M1_P1_C6','23-05-23_M1_P2_C4',
#                     '23-05-23_M1_P3_C12', '23-05-23_M1_P1_C9',
#                     '23-05-23_M1_P2_C2']

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            # (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

# selRows = data[(data['manip'] == manips[0]) & (data['compNum'] < 1) & (data['compNum'] > 6)].index
# data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')


out1, cellDf1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir,  legendLabels = labels, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

plt.legend(fontsize = 20, loc = 'upper left')


mainFig2, mainAx2 = plt.subplots(1,1)
mainFig2.patch.set_facecolor('black')


out2, cellDf2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = condCol, mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

plt.legend(fontsize = 12, loc = 'upper left')

plt.ylim(0,10)
plt.tight_layout()
# plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'250-600.png')  
plt.show()

dfAllCells = plot2Params(data, Filters, fitsSubDir, fitType, interceptStress, FIT_MODE, pathFits, plot = plot)

#%%%% Individual K vs. S plots to check averaging

data = data_main

cells = np.unique(cellDf2['cellID'])

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')

styleDict1 = {}

labels = []


for i in range(len(cells)):
    cell = cells[i]
    
    styleDict1[cell] = {'color': gs.colorList40[10+i] ,'marker':'o'}

    Filters = Filters
    
out2, cellDf2 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
              fitWidth=75, Filters = Filters, condCol = 'cellId', mode = stressRange, scale = 'lin', printText = False,
                            returnData = 1, returnCount = 1)

mainAx1.get_legend().remove()
plt.ylim(0,10)
plt.show()

#%%%% Individual stress-strain super-imposed plots to check averaging

cells = np.unique(cellDf1['cellID'])

ssPath = "D:/Anumita/MagneticPincherData/Data_TimeSeries/Timeseries_stress-strain"
ssFiles = os.listdir(ssPath)
selectedFile = []
param1 = 'M1'
param2 = 'M2'
iter = 15
# palette = list((sns.color_palette("RdBu", iter).as_hex()))

for cell in cells:
    mainFig1, mainAx1 = plt.subplots(1,2, figsize = (15,8), sharex=True, sharey=True,)
    mainFig1.patch.set_facecolor('black')
    
    print(cell)
    cell1 = cell
    cell2 = cell.replace(param1, param2)
    
    
    df1 = pd.read_csv(os.path.join(ssPath, cell1 + '_disc20um_L40_PY_stress-strain.csv'), sep = ';').dropna()
    df1['cellName'] = np.asarray([cell]*len(df1))
    sns.lineplot(data = df1, y = "Stress", x = "Strain", ax = mainAx1[0], hue = 'idxAnalysis')

    try:
        df2 = pd.read_csv(os.path.join(ssPath, cell2 + '_disc20um_L40_PY_stress-strain.csv'), sep = ';').dropna()
        sns.lineplot(data = df2, y = "Stress", x = "Strain", ax = mainAx1[1], hue = 'idxAnalysis')

    except:
        mainAx1[1].set_title('Data non-existant')
    
    fontColor = '#ffffff'
    mainFig1.text(0.5, 0.04, 'Strain (%)', ha='center', color = fontColor, size = 20)
    mainFig1.text(0.04, 0.5, 'Stress (Pa)', va='center', rotation='vertical', size =  20, 
                  color = fontColor)
    
    mainAx1[0].tick_params(axis='both', colors = fontColor) 
    mainAx1[1].tick_params(axis='both', colors = fontColor) 
    mainAx1[1].xaxis.set_tick_params(labelsize=20)
    mainAx1[1].yaxis.set_tick_params(labelsize=20)
    mainAx1[0].xaxis.set_tick_params(labelsize=20)
    mainAx1[0].yaxis.set_tick_params(labelsize=20)
    mainAx1[0].set_title('Not activated', fontsize = 20, color = fontColor)
    mainAx1[1].set_title('Activated rear', fontsize = 20, color = fontColor)
    mainFig1.suptitle(cell1 + ' | ' + cell2, color = fontColor)

    plt.ylim(0, 1500)
    plt.xlim(0, 0.30)
    
    plt.savefig(os.path.join(pathSSPlots, cell + '.png'))

#%%%% Individual K vs S super-imposed plots to check averaging

cells = np.unique(cellDf1['cellID'])

fitsPath = 'D:/Anumita/MagneticPincherData/Data_Analysis/Fits/' + fitsSubDir
param1 = 'M1'
param2 = 'M2'
iter = 15
halfWidth = 100
# palette = list((sns.color_palette("RdBu", iter).as_hex()))

for cell in cells:
    mainFig1, mainAx1 = plt.subplots(1,2, figsize = (15,10), sharex=True, sharey=True,)
    mainFig1.patch.set_facecolor('black')
    
    print(cell)
    cell1 = cell
    cell2 = cell.replace(param1, param2)
    
    
    df1 = pd.read_csv(os.path.join(fitsPath, cell1 + '_results_stressGaussian.csv'), sep = ';').dropna()
    df1 = df1[(df1['halfWidth_x'] == halfWidth) & (df1['valid'] == True)]
    df1['compNum'] = df1['compNum'].astype(int)
    
    sns.lineplot(data = df1, y = "K", x = "center_x", ax = mainAx1[0], hue = 'compNum', marker = 'o')

    try:
        df2 = pd.read_csv(os.path.join(fitsPath, cell2 + '_results_stressGaussian.csv'), sep = ';').dropna()
        df2 = df2[(df2['halfWidth_x'] == halfWidth) & (df2['valid'] == True)]
        df2['compNum'] = df2['compNum'].astype(int)
        sns.lineplot(data = df2, y = "K", x = "center_x", ax = mainAx1[1], hue = 'compNum', marker = 'o')

    except:
        mainAx1[1].set_title('Data non-existant')
    
    fontColor = '#ffffff'
    mainFig1.text(0.5, 0.04, 'Stress (Pa)', ha='center', color = fontColor, size = 20)
    mainFig1.text(0.04, 0.5, 'K (Pa)', va='center', rotation='vertical', size =  20, 
                  color = fontColor)
    
    mainAx1[0].tick_params(axis='both', colors = fontColor) 
    mainAx1[1].tick_params(axis='both', colors = fontColor) 
    mainAx1[1].xaxis.set_tick_params(labelsize=20)
    mainAx1[1].yaxis.set_tick_params(labelsize=20)
    mainAx1[0].xaxis.set_tick_params(labelsize=20)
    mainAx1[0].yaxis.set_tick_params(labelsize=20)
    mainAx1[0].set_title('Not activated', fontsize = 20, color = fontColor)
    mainAx1[1].set_title('Activated rear', fontsize = 20, color = fontColor)
    mainFig1.suptitle(cell1 + ' | ' + cell2, color = fontColor)

    plt.ylim(0, 12000)
    plt.xlim(0, 1000)
    
    plt.savefig(os.path.join(pathKSPlots, cell + '_KvsS.png'))

#%%%% Box plots, thickness
measure = 'fit_K_kPa'
labels = []
order= None

makeBoxPlots(dfAllCells, measure, condCol, labels, order, condSelection, stats = False, average = True)

#%%%% Box plots, 2Parameter fit

# labels = ['Control', 'Global Activation', '10uM Y27']
# order=['none', 'activation', 'Y27'] 
fig1, ax = makeBoxPlotParametric(dfAllCells, condCol, labels, order, condSelection, average = False)

ax.set_ylabel('Log-log exponent', fontsize = plotTicks)
ax.set_ylim(-10, 6)


#%%%% Inidividual 3D trajectories
cells = np.unique(cellDf1['cellID'])

for i in cells:
    axes = taka.plotCellTimeSeriesData(i, save = True, savePath = pathTSPlots)


#%%%% Variable over time (comp number)

palette = ["#000000", "#0000FF"]
y = 'E_Chadwick'
param1 = 'M1'
param2 = 'M2'

# if y == 'E_Chadwick':
df = pd.DataFrame({'E_Chadwick' : dfAllCells['E_Chadwick'], 'surroundingThickness' : dfAllCells['surroundingThickness'],
                   'compNum': dfAllCells['compNum'],  'cellId' : dfAllCells['cellID'], 
                   'manip' : dfAllCells['manip'], 'bestH0' : dfAllCells['bestH0'],})
df = df.drop_duplicates()

cells = np.unique(cellDf2['cellID'])
cells = np.asarray([i for i in cells if param1 in i])

mainFig1, mainAx1 = plt.subplots(4,5, figsize = (15,10))
mainFig1.patch.set_facecolor('black')
mainAx1 = mainAx1.flatten()

for i in range(len(cells)):
    print(cells[i])
    dfPlot = df[(df['cellId'] == cells[i]) | (df['cellId'] == cells[i].replace(param1, param2))]
    # x, y = 'compNum', y
    x, y = 'E_Chadwick', 'bestH0'
    
    sns.lineplot(x = x, y = y, data = dfPlot, hue = 'manip', ax = mainAx1[i], marker = 'o')
    if i != 0:
        mainAx1[i].get_legend().remove()
    mainAx1[i].set_title(cells[i], color = '#ffffff', size = 10)
    mainAx1[i].tick_params(axis='both', colors = '#ffffff')
    if y == 'E_Chadwick':
        mainAx1[i].set_ylim(0,15000)
        mainAx1[i].set_xlim(0,11)
        # mainFig1.text(0.04, 0.5, y, va='center', rotation='vertical', size =  20, 
        #               color = '#ffffff')
        mainFig1.suptitle(y + ' vs Compression Number', color = '#ffffff')

    elif y == 'surroundingThickness':
        mainAx1[i].set_ylim(0,1200)
        mainAx1[i].set_xlim(0,11)
        # mainFig1.text(0.04, 0.5, y, va='center', rotation='vertical', size =  20, 
        #               color = '#ffffff')
        mainFig1.suptitle(y + ' vs Compression Number', color = '#ffffff')

    
# mainFig1.text(0.5, 0.04, 'Compression no.', ha='center', color = '#ffffff', size = 20)
plt.tight_layout()
plt.show()
# %%%% Plotting surrounding thickness / best H0 - manip wise

data = data_main
plt.style.use('seaborn')

# legendLabels = ['Y27-treated', 'Normally Expressing Pop.', 'Low Expressing Pop.', ]
dates = ['23-07-12']
condCol = 'manip'
manips = ['M1', 'M2']

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['compNum'] < 8),
            (data['E_Chadwick'] < 30000),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manipId'].apply(lambda x : x in manipsIDs)),
            ]


globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

fig1, axes = plt.subplots(1,1, figsize=(15,10))
fig1.patch.set_facecolor('black')

palette = ["#000000", "#0000FF"]

x = (data_f['compNum']-1)*20
sns.lmplot(x = 'bestH0', y = 'E_Chadwick', palette = palette, data = data_f, hue = 'manip')
# fig1.suptitle('[15mT = 500pN] Surrounding Thickness (nm) vs. Time (secs)')
# plt.xticks(fontsize=30, color = '#ffffff')
# plt.yticks(fontsize=30, color = '#ffffff')
# plt.xlabel('Time (secs)', fontsize = 25, color = '#ffffff')
# plt.ylabel('Surrounding thickness (nm)', fontsize = 25, color = '#ffffff')
# axes.get_legend().remove()

plt.ylim(0,30000)

plt.show()

#%% Experiments at 5mT and 15mT

# lowFieldData = taka.getMergedTable('Chad_f15_23-06-28_23-06-29')
# highFieldData = taka.getMergedTable('Chad_f15_All3T3PRGControl_23-05-16')


GlobalTable = taka.getMergedTable('Chad_f15_MagField_23-07-12')
fitsSubDir = 'Chad_f15_MagField_23-07-12'

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
data_main['cellId'] = GlobalTable['cellID']

nBins = 11
bins = np.linspace(0, 2000, nBins)
data_main['H0_Bin'] = np.digitize(GlobalTable['bestH0'], bins, right = True)
data_main['Thickness_Bin'] = np.digitize(GlobalTable['surroundingThickness'], bins, right = True)


fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75


#%%%% Non linearity
#All 3T3PRG vs 3T3 optoLARG vs 3T3 WT
# manipIDs = ['22-12-07_M4', '23-02-02_M1', '23-05-10_M3', '23-03-28_M1']

########## Declare variables ##########
dates = ['23-07-07']
stressRange = '150_450'
interceptStress = 250
plot = False
FIT_MODE = 'loglog'
condCol = 'Thickness_Bin'
order = None
# , '23-06-28_M1'
# manipIDs = ['23-06-28_M1']
manipIDs = ['23-06-28_M1', '23-07-07_M1' , '23-07-07_M2']
manips = ['M1', 'M2']
field = 15
condSelection = manips

labels = []
# styleDict1 =  {5:{'color': gs.colorList40[21],'marker':'o'},
#                 15:{'color': "#000000",'marker':'o'},
#                 }

# styleDict1 =  {'23-07-07_M1':{'color': gs.colorList40[22],'marker':'o'},
#                '23-07-07_M2':{'color': gs.colorList40[23],'marker':'o'},
#                 '23-06-28_M1':{'color': gs.colorList40[21],'marker':'o'},
#                 '23-05-23_M1':{'color': gs.colorList40[22],'marker':'o'},
#                 }

# styleDict1 =  {5:{'color': gs.colorList40[22],'marker':'o'},
#                15:{'color': gs.colorList40[23],'marker':'o'},
#                 }

styleDict1 = {}

keys = np.rint(np.linspace(1,nBins,nBins)).astype(int)
values = gs.colorList40[10:nBins+10]

for key, value in zip(keys, values):
    key = int(key)
    styleDict1[key] = {'color' : value, 'marker' : 'o'}
print(styleDict1)


###########################################

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

data = data_main

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    

# allSelectedCells = np.asarray(intersected + controlCells)


Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['normal field'] == 15),
            # (data['UI_Valid'] == True),
            (data['bestH0'] <= 2000),
            # (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manip'].apply(lambda x : x in manips)),
            (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')


out1, cellDf1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir,  legendLabels = labels, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

plt.legend(fontsize = 20, loc = 'upper left')


mainFig2, mainAx2 = plt.subplots(1,1)
mainFig2.patch.set_facecolor('black')


out2, cellDf2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = condCol, mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

plt.legend(fontsize = 12, loc = 'upper left')

plt.ylim(0,10)
plt.tight_layout()
# plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'250-600.png')  
plt.show()

dfAllCells = plot2Params(data, Filters, fitsSubDir, fitType, interceptStress, FIT_MODE, pathFits, plot = plot)

#%%%% Individual K vs. S plots to check averaging

data = data_main

cells = np.unique(cellDf2['cellID'])

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')

styleDict1 = {}

labels = []


for i in range(len(cells)):
    cell = cells[i]
    
    styleDict1[cell] = {'color': gs.colorList40[10+i] ,'marker':'o'}

    Filters = [(data['validatedThickness'] == True),
                # (data['substrate'] == '20um fibronectin discs'), 
                # (data['drug'] == 'none'), 
                (data['bead type'] == 'M450'),
                # (data['UI_Valid'] == True),
                (data['bestH0'] <= 1900),
                # (data['date'].apply(lambda x : x in dates)),
                (data['cellId'].apply(lambda x : x in cells)),
                # (data['manip'].apply(lambda x : x in manips)),
                # (data['manipId'].apply(lambda x : x in manipIDs)),
                ] 
    
out2, cellDf2 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
              fitWidth=75, Filters = Filters, condCol = 'cellId', mode = stressRange, scale = 'lin', printText = False,
                            returnData = 1, returnCount = 1)

mainAx1.get_legend().remove()
plt.ylim(0,10)
plt.show()

#%%%% Box plots, thickness
measure = 'ctFieldFluctuAmpli'

labels = []
order= None

colorPalette = [gs.colorList40[21],  gs.colorList40[22],  gs.colorList40[23]]
makeBoxPlots(dfAllCells, measure, condCol, labels, order, condSelection, 
             colorPalette=colorPalette, stats = False, average = False)

plt.ylim(0, 1000)
plt.show()

#%% All Experiments

GlobalTable = taka.getMergedTable('Chad_f15_AllExpts-ATCC_23-07-03')
fitsSubDir = 'Chad_f15_AllExpts-ATCC_23-07-03'

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
data_main['cellId'] = GlobalTable['cellID']


fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75


#%%%% Non linearity
#All 3T3PRG vs 3T3 optoLARG vs 3T3 WT
# manipIDs = ['22-12-07_M4', '23-02-02_M1', '23-05-10_M3', '23-03-28_M1']

########## Declare variables ##########
dates = ['23-05-23']
stressRange = '200_500'
interceptStress = 250
plot = False
FIT_MODE = 'loglog'
condCol = 'cell subtype'
order = None

manips = [ 'M1', 'M3']
manipIDs = [ '23-05-10_M3', '23-02-02_M1']
# manipIDs = [ ]
# manips = ['M1']
# condSelection = ['5', '15']

labels = []
# styleDict1 =  {'23-02-02_M1':{'color': gs.colorList40[21],'marker':'o'},
#                 '23-04-25_M3':{'color': gs.colorList40[21],'marker':'o'},
#                 '23-02-02_M1':{'color': gs.colorList40[23],'marker':'o'},
#                 '23-05-10_M3':{'color': gs.colorList40[22],'marker':'o'},
#                 '23-05-23_M1':{'color': gs.colorList40[30],'marker':'o'},
#                 '22-12-07_M4':{'color': gs.colorList40[30],'marker':'o'},
#                 '23-06-28_M1':{'color': "#000000",'marker':'o'},
#                 }

# styleDict1 =  {'M1':{'color': gs.colorList40[21],'marker':'o'},
#                'M3':{'color': gs.colorList40[22],'marker':'o'},

#                 }

styleDict1 =  {'Atcc-2023':{'color': gs.colorList40[21],'marker':'o'},
                'optoRhoA':{'color': '#000000','marker':'o'},
        
                }

# styleDict1 =  {5:{'color': gs.colorList40[21],'marker':'o'},
#                 15:{'color': gs.colorList40[22],'marker':'o'},
#                 }

###########################################

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

# plt.style.use('seaborn-v0_8-bright')
# plt.style.use('seaborn')

data = data_main

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    

# allSelectedCells = np.asarray(intersected + controlCells)


Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            # (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            # (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manip'].apply(lambda x : x in manips)),
            (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

# selRows = data[(data['manip'] == manips[0]) & (data['compNum'] < 1) & (data['compNum'] > 6)].index
# data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')


out1, cellDf1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir,  legendLabels = labels, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

plt.legend(fontsize = 20, loc = 'upper left')


mainFig2, mainAx2 = plt.subplots(1,1)
# mainFig2.patch.set_facecolor('black')


out2, cellDf2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = condCol, mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

plt.legend(fontsize = 12, loc = 'upper left')

plt.ylim(0,8)
plt.tight_layout()
# plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'250-600.png')  
plt.show()

# dfAllCells = plot2Params(data, Filters, fitsSubDir, fitType, interceptStress, FIT_MODE, pathFits, plot = plot)

#%%%% Individual K vs. S plots to check averaging

data = data_main

cells = np.unique(cellDf2['cellID'])

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')

styleDict1 = {}

labels = []


for i in range(len(cells)):
    cell = cells[i]
    
    styleDict1[cell] = {'color': gs.colorList40[10+i] ,'marker':'o'}

    Filters = [(data['validatedThickness'] == True),
                # (data['substrate'] == '20um fibronectin discs'), 
                # (data['drug'] == 'none'), 
                (data['bead type'] == 'M450'),
                # (data['UI_Valid'] == True),
                (data['bestH0'] <= 1900),
                # (data['date'].apply(lambda x : x in dates)),
                (data['cellId'].apply(lambda x : x in cells)),
                # (data['manip'].apply(lambda x : x in manips)),
                # (data['manipId'].apply(lambda x : x in manipIDs)),
                ] 
    
out2, cellDf2 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
              fitWidth=75, Filters = Filters, condCol = 'cellId', mode = stressRange, scale = 'lin', printText = False,
                            returnData = 1, returnCount = 1)

mainAx1.get_legend().remove()
plt.ylim(0,10)
plt.show()

#%%%% Box plots, thickness
measure = 'surroundingThickness'

labels = []
order = None

# colorPalette = 
makeBoxPlots(dfAllCells, measure, condCol, labels, order, condSelection, stats = False, average = False)
plt.ylim(0, 1000)
plt.show()

#%% Re-analysing experiments from 22-10-06

GlobalTable = taka.getMergedTable('Chad_f15_22-10-06_23-07-04')
fitsSubDir = 'Chad_f15_22-10-06_23-07-04'

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
data_main['cellId'] = GlobalTable['cellID']


fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75

#%%%% Non linearity
#All 3T3PRG vs 3T3 optoLARG vs 3T3 WT
# manipIDs = ['22-12-07_M4', '23-02-02_M1', '23-05-10_M3', '23-03-28_M1']

########## Declare variables ##########
dates = ['22-10-06']
stressRange = '150_400'
interceptStress = 250
plot = False
FIT_MODE = 'loglog'
condCol = 'manip'
order = None

manips = ['M1', 'M5']

labels = ['5mT', '15mT']
styleDict1 =  {'M1':{'color': "#000000",'marker':'o'},
                'M2':{'color': gs.colorList40[22],'marker':'o'},
                'M3':{'color': gs.colorList40[21],'marker':'o'},
                'M4':{'color': gs.colorList40[22],'marker':'o'},
                'M5':{'color':gs.colorList40[21],'marker':'o'},
                'M6':{'color': gs.colorList40[22],'marker':'o'},
                }

###########################################

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

# plt.style.use('seaborn-v0_8-bright')
# plt.style.use('seaborn')

data = data_main

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    

# allSelectedCells = np.asarray(intersected + controlCells)


Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            # (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

# selRows = data[(data['manip'] == manips[0]) & (data['compNum'] < 1) & (data['compNum'] > 6)].index
# data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')


out1, cellDf1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir,  legendLabels = labels, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

plt.legend(fontsize = 20, loc = 'upper left')


mainFig2, mainAx2 = plt.subplots(1,1)
mainFig2.patch.set_facecolor('black')


out2, cellDf2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = condCol, mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

plt.legend(fontsize = 14, loc = 'upper left')

# plt.ylim(0,10)
# plt.tight_layout()
# # plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'250-600.png')  
# plt.show()

dfAllCells = plot2Params(data, Filters, fitsSubDir, fitType, interceptStress, FIT_MODE, pathFits, plot = plot)

#%%%% Individual K vs. S plots to check averaging

data = data_main

cells = np.unique(cellDf2['cellID'])

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')

styleDict1 = {}

labels = []


for i in range(len(cells)):
    cell = cells[i]
    
    styleDict1[cell] = {'color': gs.colorList40[10+i] ,'marker':'o'}

    Filters = [(data['validatedThickness'] == True),
                # (data['substrate'] == '20um fibronectin discs'), 
                # (data['drug'] == 'none'), 
                (data['bead type'] == 'M450'),
                # (data['UI_Valid'] == True),
                (data['bestH0'] <= 1900),
                # (data['date'].apply(lambda x : x in dates)),
                (data['cellId'].apply(lambda x : x in cells)),
                # (data['manip'].apply(lambda x : x in manips)),
                # (data['manipId'].apply(lambda x : x in manipIDs)),
                ] 
    
out2, cellDf2 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
              fitWidth=75, Filters = Filters, condCol = 'cellId', mode = stressRange, scale = 'lin', printText = False,
                            returnData = 1, returnCount = 1)

mainAx1.get_legend().remove()
plt.ylim(0,10)
plt.show()

#%%%% Box plots, thickness
measure = 'E_Chadwick'

labels = []
order= None

makeBoxPlots(dfAllCells, measure, condCol, labels, order, condSelection, stats = False, average = False)
# plt.ylim(0, 1000)
# plt.show()

#%% Re-analysing Y27 experiments

GlobalTable = taka.getMergedTable('Chad_f15_AllExpts_23-05-30')
fitsSubDir = 'Chad_f15_AllExpts_23-05-30'

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
data_main['cellId'] = GlobalTable['cellID']

fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75


#%%%% Non linearity
#All 3T3PRG vs 3T3 optoLARG vs 3T3 WT
# manipIDs = ['22-12-07_M4', '23-02-02_M1', '23-05-10_M3', '23-03-28_M1']

########## Declare variables ##########
dates = ['22-12-07']
stressRange = '150_550'
interceptStress = 250
plot = False
FIT_MODE = 'loglog'
condCol = 'manip'
order = None


# manipIDs = ['23-01-23_M1', '23-04-19_M1', '23-02-02_M1', '23-05-10_M3', '23-02-23_M1']
manips = ['M4'] #, 'M5']
condSelection = manips

labels = []
styleDict1 =  {'M4':{'color': gs.colorList40[21],'marker':'o'},
               'M5':{'color': gs.colorList40[22],'marker':'o'},
                'M8':{'color': "#000000",'marker':'o'},
                }

###########################################

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

# plt.style.use('seaborn-v0_8-bright')
# plt.style.use('seaborn')

data = data_main

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    

# allSelectedCells = np.asarray(intersected + controlCells)

# allSelectedCells = ['23-05-23_M3_P1_C6','23-05-23_M3_P2_C4',
#                     '23-05-23_M3_P3_C12', '23-05-23_M3_P1_C9',
#                     '23-05-23_M3_P2_C2', 
#                     '23-05-23_M1_P1_C6','23-05-23_M1_P2_C4',
#                     '23-05-23_M1_P3_C12', '23-05-23_M1_P1_C9',
#                     '23-05-23_M1_P2_C2']

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['E_Chadwick'] < 40000),
            # (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

# selRows = data[(data['manip'] == manips[0]) & (data['compNum'] < 1) & (data['compNum'] > 6)].index
# data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')


out1, cellDf1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir,  legendLabels = labels, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

plt.legend(fontsize = 20, loc = 'upper left')


mainFig2, mainAx2 = plt.subplots(1,1)
mainFig2.patch.set_facecolor('black')


out2, cellDf2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = condCol, mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

plt.legend(fontsize = 12, loc = 'upper left')

# plt.ylim(0,10)
# plt.tight_layout()
# # plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'250-600.png')  
# plt.show()

dfAllCells = plot2Params(data, Filters, fitsSubDir, fitType, interceptStress, FIT_MODE, pathFits, plot = plot)

#%% Testing stress-strain plots

cells = np.unique(cellDf1['cellID'])

ssPath = "D:/Anumita/MagneticPincherData/Data_TimeSeries/Timeseries_stress-strain"
ssFiles = os.listdir(ssPath)
selectedFile = []
param1 = 'M4'
param2 = 'M5'
iter = 15
# palette = list((sns.color_palette("RdBu", iter).as_hex()))

for cell in cells:
    mainFig1, mainAx1 = plt.subplots(1,2, figsize = (15,8), sharex=True, sharey=True,)
    mainFig1.patch.set_facecolor('black')
    
    print(cell)
    cell1 = cell
    cell2 = cell.replace(param1, param2)
    
    
    df1 = pd.read_csv(os.path.join(ssPath, cell1 + '_disc20um_L40_PY_stress-strain.csv'), sep = ';').dropna()
    df1['cellName'] = np.asarray([cell]*len(df1))
    sns.lineplot(data = df1, y = "Stress", x = "Strain", ax = mainAx1[0], hue = 'idxAnalysis')

    try:
        df2 = pd.read_csv(os.path.join(ssPath, cell2 + '_disc20um_L40_PY_stress-strain.csv'), sep = ';').dropna()
        sns.lineplot(data = df2, y = "Stress", x = "Strain", ax = mainAx1[1], hue = 'idxAnalysis')

    except:
        mainAx1[1].set_title('Data non-existant')
    
    fontColor = '#ffffff'
    mainFig1.text(0.5, 0.04, 'Strain (%)', ha='center', color = fontColor, size = 20)
    mainFig1.text(0.04, 0.5, 'Stress (Pa)', va='center', rotation='vertical', size =  20, 
                  color = fontColor)
    
    mainAx1[0].tick_params(axis='both', colors = fontColor) 
    mainAx1[1].tick_params(axis='both', colors = fontColor) 
    mainAx1[1].xaxis.set_tick_params(labelsize=20)
    mainAx1[1].yaxis.set_tick_params(labelsize=20)
    mainAx1[0].xaxis.set_tick_params(labelsize=20)
    mainAx1[0].yaxis.set_tick_params(labelsize=20)
    mainAx1[0].set_title('Not activated', fontsize = 20, color = fontColor)
    mainAx1[1].set_title('Activated rear', fontsize = 20, color = fontColor)
    mainFig1.suptitle(cell1 + ' | ' + cell2, color = fontColor)

    plt.ylim(0, 1500)
    plt.xlim(0, 0.30)
    plt.show()
    plt.savefig(os.path.join(pathSSPlots, cell + '.png'))
    
# %%%% Plotting surrounding thickness / best H0 - manip wise


cellCond = pd.read_csv(os.path.join(experimentalDataDir, 'cellConditions_22-10-06&22-12-07.csv'))

plt.style.use('default')
# category = 'Polarised, beads together'
# category = 'Polarised, no lamellipodia'

# category = 'beads non-adhesive'
# cellCats = cellCond[category]

styleDict1 =  {'M1':{'color': gs.colorList40[1],'marker':'o'},
               'M2':{'color': gs.colorList40[11],'marker':'o'},
               'M3':{'color': gs.colorList40[21],'marker':'o'},
                'M4':{'color': gs.colorList40[2],'marker':'o'},
                'M5':{'color': gs.colorList40[22],'marker':'o'},
                'M6':{'color': gs.colorList40[32],'marker':'o'},
                'M7':{'color': gs.colorList40[22],'marker':'o'},
                # 'M8':{'color': gs.colorList40[11],'marker':'o'},
                # 'M9':{'color': gs.colorList40[11],'marker':'o'},
               }


data = data_main

oldManip = ['M7', 'M8', 'M9']

for i in oldManip:
    data['manip'][data['manip'] == i] = 'M5'

dates = ['22-12-07']
manips = ['M4', 'M5'] #, 'M6']# , 'M4', 'M5']

# control = 'M4'
# active = 'M5'
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

fig1, axes = plt.subplots(1,1, figsize=(15,10))

# sns.set_palette(sns.color_palette("tab10"))

flatui = ["#ee1111", "#1111EE"]
sns.set_palette(flatui)

x = data_f['compNum']*20
sns.lineplot(x = x, y = 'bestH0', data = data_f, hue = 'manip', linewidth = 3)

fig1.suptitle('[15mT = 500pN] bestH0 (nm) vs. Time (secs)', fontsize = 25)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('Time (secs)', fontsize = 25)
plt.ylabel('bestH0 (nm)', fontsize = 25)

activated = mpatches.Patch(color=flatui[1], label='Activated')
control = mpatches.Patch(color=flatui[0], label='Non-activated')
plt.legend(handles=[activated, control], fontsize = 20)


plt.ylim(0,1500)
plt.tight_layout()

plt.savefig(todayFigDir + '/22-12-07_surroundingThicknessvsCompr.png')

plt.show()
cells = np.unique(cellDf1['cellID'])

ssPath = "D:/Anumita/MagneticPincherData/Data_TimeSeries/Timeseries_stress-strain"
ssFiles = os.listdir(ssPath)
selectedFile = []
param1 = 'M1'
param2 = 'M2'
iter = 15
# palette = list((sns.color_palette("RdBu", iter).as_hex()))

for cell in cells:
    mainFig1, mainAx1 = plt.subplots(1,2, figsize = (15,8), sharex=True, sharey=True,)
    mainFig1.patch.set_facecolor('black')
    
    print(cell)
    cell1 = cell
    cell2 = cell.replace(param1, param2)
    
    
    df1 = pd.read_csv(os.path.join(ssPath, cell1 + '_disc20um_L40_PY_stress-strain.csv'), sep = ';').dropna()
    df1['cellName'] = np.asarray([cell]*len(df1))
    sns.lineplot(data = df1, y = "Stress", x = "Strain", ax = mainAx1[0], hue = 'idxAnalysis')

    try:
        df2 = pd.read_csv(os.path.join(ssPath, cell2 + '_disc20um_L40_PY_stress-strain.csv'), sep = ';').dropna()
        sns.lineplot(data = df2, y = "Stress", x = "Strain", ax = mainAx1[1], hue = 'idxAnalysis')

    except:
        mainAx1[1].set_title('Data non-existant')
    
    fontColor = '#ffffff'
    mainFig1.text(0.5, 0.04, 'Strain (%)', ha='center', color = fontColor, size = 20)
    mainFig1.text(0.04, 0.5, 'Stress (Pa)', va='center', rotation='vertical', size =  20, 
                  color = fontColor)
    
    mainAx1[0].tick_params(axis='both', colors = fontColor) 
    mainAx1[1].tick_params(axis='both', colors = fontColor) 
    mainAx1[1].xaxis.set_tick_params(labelsize=20)
    mainAx1[1].yaxis.set_tick_params(labelsize=20)
    mainAx1[0].xaxis.set_tick_params(labelsize=20)
    mainAx1[0].yaxis.set_tick_params(labelsize=20)
    mainAx1[0].set_title('Not activated', fontsize = 20, color = fontColor)
    mainAx1[1].set_title('Activated rear', fontsize = 20, color = fontColor)
    mainFig1.suptitle(cell1 + ' | ' + cell2, color = fontColor)

    plt.ylim(0, 1500)
    plt.xlim(0, 0.30)
    
    plt.savefig(os.path.join(pathSSPlots, cell + '.png'))

# %%%% Plotting surrounding thickness / best H0 - manip wise

data = data_main
plt.style.use('seaborn')

# legendLabels = ['Y27-treated', 'Normally Expressing Pop.', 'Low Expressing Pop.', ]
# dates = ['23-07-12']
# condCol = 'manip'
# manips = ['M1', 'M2']


globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

fig1, axes = plt.subplots(1,1, figsize=(15,10))
fig1.patch.set_facecolor('black')

palette = ["#000000", "#0000FF"]

x = (data_f['compNum']-1)*20
sns.lmplot(x = 'bestH0', y = 'E_Chadwick', palette = palette, data = data_f, hue = 'manip')
# fig1.suptitle('[15mT = 500pN] Surrounding Thickness (nm) vs. Time (secs)')
# plt.xticks(fontsize=30, color = '#ffffff')
# plt.yticks(fontsize=30, color = '#ffffff')
# plt.xlabel('Time (secs)', fontsize = 25, color = '#ffffff')
# plt.ylabel('Surrounding thickness (nm)', fontsize = 25, color = '#ffffff')
# axes.get_legend().remove()

plt.ylim(0,30000)

plt.show()

#%%%% Individual K vs. S plots to check averaging

data = data_main

cells = np.unique(cellDf1['cellID'])

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')

styleDict1 = {}

labels = []


for i in range(len(cells)):
    cell = cells[i]
    
    styleDict1[cell] = {'color': gs.colorList40[10+i] ,'marker':'o'}

    Filters = Filters
    
out2, cellDf2 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
              fitWidth=75, Filters = Filters, condCol = 'cellId', mode = stressRange, scale = 'lin', printText = False,
                            returnData = 1, returnCount = 1)

mainAx1.get_legend().remove()
plt.ylim(0,10)
plt.show()

#%%%% Individual stress-strain super-imposed plots to check averaging

cells = np.unique(cellDf2['cellID'])

ssPath = "D:/Anumita/MagneticPincherData/Data_TimeSeries/Timeseries_stress-strain"
ssFiles = os.listdir(ssPath)
selectedFile = []
param1 = 'M3'
param2 = 'M4'
iter = 15
# palette = list((sns.color_palette("RdBu", iter).as_hex()))

for cell in cells:
    mainFig1, mainAx1 = plt.subplots(1,2, figsize = (15,10), sharex=True, sharey=True,)
    mainFig1.patch.set_facecolor('black')
    
    print(cell)
    cell1 = cell
    cell2 = cell.replace(param1, param2)
    
    
    df1 = pd.read_csv(os.path.join(ssPath, cell1 + '_disc20um_L40_PY_stress-strain.csv'), sep = ';').dropna()
    sns.lineplot(data = df1, y = "Stress", x = "Strain", ax = mainAx1[0], hue = 'idxAnalysis')

    try:
        df2 = pd.read_csv(os.path.join(ssPath, cell2 + '_disc20um_L40_PY_stress-strain.csv'), sep = ';').dropna()
        sns.lineplot(data = df2, y = "Stress", x = "Strain", ax = mainAx1[1], hue = 'idxAnalysis')

    except:
        mainAx1[1].set_title('Data non-existant')
    
    fontColor = '#ffffff'
    mainFig1.text(0.5, 0.04, 'Strain (%)', ha='center', color = fontColor, size = 20)
    mainFig1.text(0.04, 0.5, 'Stress (Pa)', va='center', rotation='vertical', size =  20, 
                  color = fontColor)
    
    mainAx1[0].tick_params(axis='both', colors = fontColor) 
    mainAx1[1].tick_params(axis='both', colors = fontColor) 
    mainAx1[1].xaxis.set_tick_params(labelsize=20)
    mainAx1[1].yaxis.set_tick_params(labelsize=20)
    mainAx1[0].xaxis.set_tick_params(labelsize=20)
    mainAx1[0].yaxis.set_tick_params(labelsize=20)
    mainAx1[0].set_title('Not activated', fontsize = 20, color = fontColor)
    mainAx1[1].set_title('Activated rear', fontsize = 20, color = fontColor)
    mainFig1.suptitle(cell1 + ' | ' + cell2, color = fontColor)

    plt.ylim(0, 1500)
    plt.xlim(0, 0.30)
    
    plt.savefig(os.path.join(pathSSPlots, cell + '.png'))

#%%%% Individual K vs S super-imposed plots to check averaging

cells = np.unique(cellDf2['cellID'])

fitsPath = 'D:/Anumita/MagneticPincherData/Data_Analysis/Fits/' + fitsSubDir
param1 = 'M7'
param2 = 'M8'
iter = 15
halfWidth = 100
# palette = list((sns.color_palette("RdBu", iter).as_hex()))

for cell in cells:
    mainFig1, mainAx1 = plt.subplots(1,2, figsize = (15,10), sharex=True, sharey=True,)
    mainFig1.patch.set_facecolor('black')
    
    print(cell)
    cell1 = cell
    cell2 = cell.replace(param1, param2)
    
    
    df1 = pd.read_csv(os.path.join(fitsPath, cell1 + '_results_stressGaussian.csv'), sep = ';').dropna()
    df1 = df1[(df1['halfWidth_x'] == halfWidth) & (df1['valid'] == True)]
    df1['compNum'] = df1['compNum'].astype(int)
    
    sns.lineplot(data = df1, y = "K", x = "center_x", ax = mainAx1[0], hue = 'compNum', marker = 'o')

    try:
        df2 = pd.read_csv(os.path.join(fitsPath, cell2 + '_results_stressGaussian.csv'), sep = ';').dropna()
        df2 = df2[(df2['halfWidth_x'] == halfWidth) & (df2['valid'] == True)]
        df2['compNum'] = df2['compNum'].astype(int)
        sns.lineplot(data = df2, y = "K", x = "center_x", ax = mainAx1[1], hue = 'compNum', marker = 'o')

    except:
        mainAx1[1].set_title('Data non-existant')
    
    fontColor = '#ffffff'
    mainFig1.text(0.5, 0.04, 'Stress (Pa)', ha='center', color = fontColor, size = 20)
    mainFig1.text(0.04, 0.5, 'K (Pa)', va='center', rotation='vertical', size =  20, 
                  color = fontColor)
    
    mainAx1[0].tick_params(axis='both', colors = fontColor) 
    mainAx1[1].tick_params(axis='both', colors = fontColor) 
    mainAx1[1].xaxis.set_tick_params(labelsize=20)
    mainAx1[1].yaxis.set_tick_params(labelsize=20)
    mainAx1[0].xaxis.set_tick_params(labelsize=20)
    mainAx1[0].yaxis.set_tick_params(labelsize=20)
    mainAx1[0].set_title('Not activated', fontsize = 20, color = fontColor)
    mainAx1[1].set_title('Activated rear', fontsize = 20, color = fontColor)
    mainFig1.suptitle(cell1 + ' | ' + cell2, color = fontColor)

    plt.ylim(0, 12000)
    plt.xlim(0, 1000)
    
    plt.savefig(os.path.join(pathKSPlots, cell + '_KvsS.png'))

#%%%% Box plots, thickness
measure = 'fit_K_kPa'
labels = []
order= None

makeBoxPlots(dfAllCells, measure, condCol, labels, order, condSelection, stats = False, average = True)

#%%%% Box plots, 2Parameter fit

# labels = ['Control', 'Global Activation', '10uM Y27']
# order=['none', 'activation', 'Y27'] 
fig1, ax = makeBoxPlotParametric(dfAllCells, condCol, labels, order, condSelection, average = False)

ax.set_ylabel('Log-log exponent', fontsize = plotTicks)
ax.set_ylim(-10, 6)


#%%%% Inidividual 3D trajectori
cellIDs = dfAllCells['cellID'].unique()

for i in cellIDs:
    axes = taka.plotCellTimeSeriesData(i, save = True, savePath = todayFigDir + '/timeseries/')

# %%%% Plotting surrounding thickness / best H0 - manip wise

data = data_main
plt.style.use('seaborn')

# legendLabels = ['Y27-treated', 'Normally Expressing Pop.', 'Low Expressing Pop.', ]
dates = ['23-02-02']
condCol = 'manip'
manips = ['M1', 'M2']

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['compNum'] < 8),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manipId'].apply(lambda x : x in manipsIDs)),
            ]


globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

fig1, axes = plt.subplots(1,1, figsize=(15,10))
fig1.patch.set_facecolor('black')

palette = ["#000000", "#0000FF"]

x = (data_f['compNum']-1)*20
sns.lineplot(x = x, y = 'surroundingThickness', palette = palette, data = data_f, hue = 'manipId')
fig1.suptitle('[15mT = 500pN] Surrounding Thickness (nm) vs. Time (secs)')
plt.xticks(fontsize=30, color = '#ffffff')
plt.yticks(fontsize=30, color = '#ffffff')
plt.xlabel('Time (secs)', fontsize = 25, color = '#ffffff')
plt.ylabel('Surrounding thickness (nm)', fontsize = 25, color = '#ffffff')
axes.get_legend().remove()

plt.ylim(0,1200)

plt.show()

#%% Re-plotting data with new experiments, but only control optoPRG (15mT, 1.5s, 1-50mT)

GlobalTable = taka.getMergedTable('Chad_f15_optoPRGCtrl_23-10-11')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']

fitsSubDir = 'Chad_f15_optoPRGCtrl_23-10-11'

nBins = 11
bins = np.linspace(0, 2000, nBins)
data_main['H0_Bin'] = np.digitize(GlobalTable['bestH0'], bins, right = True)
data_main['Thickness_Bin'] = np.digitize(GlobalTable['surroundingThickness'], bins, right = True)

fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75


#%%%% Non linearity

########## Declare variables ##########


celltypes = ['optoRhoA']

stressRange = '200_550'
interceptStress = 250
plot = False
FIT_MODE = 'loglog'
condCol = 'H0_Bin'
order = np.linspace(1,len(bins), len(bins))
labels = []
condSelection = ['none']


styleDict1 = {}
keys = np.rint(np.linspace(1,nBins,nBins)).astype(int)
values = gs.colorList40[10:nBins+10]

for key, value, label in zip(keys, values, bins):
    key = int(key)
    styleDict1[key] = {'color' : value, 'marker' : 'o', 'label' : ">" + str(label)}
print(styleDict1)

# styleDict1 = {'optoRhoA' : {'color' : '#9ecae1', 'marker' : 'o', 'label' : 'optoRhoA'}}


###########################################

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

data = data_main

Filters = [(data['validatedThickness'] == True),
            (data['substrate'] == '20um fibronectin discs'), 
            (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            # (data['UI_Valid'] == True),
            (data['bestH0'] <= 2000),
            # (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manip'].apply(lambda x : x in manips)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            (data['cell subtype'].apply(lambda x : x in celltypes)),
            ]

# selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')


out1, cellDf1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir,  legendLabels = labels, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

plt.legend(fontsize = 20, loc = 'upper left')


mainFig2, mainAx2 = plt.subplots(1,1)
mainFig2.patch.set_facecolor('black')


out2, cellDf2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = condCol, mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

plt.legend(fontsize = 15, loc = 'upper left')

plt.ylim(0,10)
plt.tight_layout()
# plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'250-600.png')  
plt.show()

#%% Plots for 23-10-19
GlobalTable = taka.getMergedTable('Chad_f15_23-05-10_23-05-16')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']

fitsSubDir = 'Chad_f15_23-05-10_23-05-16'


fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75


#%%%% Non linearity

########## Declare variables ##########
dates = ['23-05-10']
manips = ['M3', 'M4'] #, 'M3'] 
labels = ['Not activated', 'Activated']

stressRange = '200_550'
interceptStress = 250
plot = False
FIT_MODE = 'loglog'
condCol = 'manip'
order = None

condSelection = manips

styleDict1 =  {'M1':{'color': "#5c337f",'marker':'o'},
                'M2':{'color': gs.colorList40[23],'marker':'o'},
                'M3':{'color': gs.colorList40[20],'marker':'o'},
                'M4':{'color': "#008fb1",'marker':'o'},
                }

###########################################

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

# plt.style.use('seaborn-v0_8-bright')
# plt.style.use('seaborn')

data = data_main

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            # (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

# selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')


out1, cellDf1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir,  legendLabels = labels, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)


mainFig1, mainAx1, exportDf1, countDf1 = out1

plt.legend(fontsize = 20, loc = 'upper left')


mainFig2, mainAx2 = plt.subplots(1,1)
mainFig2.patch.set_facecolor('black')


out2, cellDf2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = condCol, mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

# plt.legend(fontsize = 15, loc = 'upper left')

plt.ylim(0,10)
plt.tight_layout()
# plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'250-600.png')  
plt.show()
