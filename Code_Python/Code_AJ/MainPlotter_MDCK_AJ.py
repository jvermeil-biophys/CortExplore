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
fontColour = '#ffffff'


# %%%% Default settings for mechanics analysis

#### HOW TO USE    
# Example : I don't want to do a whole curve Chawick Fit anymore, but I want a Dimitriadis one.
# I can create a dict fitSettings = {'doChadwickFit' : False, 'doDimitriadisFit' : True}
# And pass it as an argument in computeGlobalTable_meca

#### DETAILS
# See ufun.updateDefaultSettingsDict(settingsDict, defaultSettingsDict)
# And the 'Settings' flag in analyseTimeSeries_meca() 

#### AS REFERENCE ONLY, here is a copy of the default settings.

#### 1. For Fits
DEFAULT_stressCenters = [ii for ii in range(100, 1550, 50)]
DEFAULT_stressHalfWidths = [50, 75, 100]

DEFAULT_strainCenters = [ii/10000 for ii in range(125, 3750, 125)]
DEFAULT_strainHalfWidths = [0.0125, 0.025, 0.05]

DEFAULT_fitSettings = {# H0
                       'methods_H0':['Chadwick', 'Dimitriadis'],
                       'zones_H0':['%f_10', '%f_20'],
                       'method_bestH0':'Chadwick',
                       'zone_bestH0':'%f_10',
                       # Global fits
                       'doChadwickFit' : True,
                       'doDimitriadisFit' : False,
                       # Local fits
                       'doStressRegionFits' : True,
                       'doStressGaussianFits' : True,
                       'centers_StressFits' : DEFAULT_stressCenters,
                       'halfWidths_StressFits' : DEFAULT_stressHalfWidths,
                       'doNPointsFits' : True,
                       'nbPtsFit' : 13,
                       'overlapFit' : 3,
                       # NEW - Numi
                       'doLogFits' : True,
                       'nbPtsFitLog' : 10,
                       'overlapFitLog' : 5,
                       # NEW - Jojo
                       'doStrainGaussianFits' : True,
                       'centers_StrainFits' : DEFAULT_strainCenters,
                       'halfWidths_StrainFits' : DEFAULT_strainHalfWidths,
                       }

#### 2. For Validation

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


#### 3. For Plots

DEFAULT_plot_stressCenters = [ii for ii in range(100, 1550, 50)]
DEFAULT_plot_stressHalfWidth = 75

DEFAULT_plot_strainCenters = [ii/10000 for ii in range(125, 3750, 125)]
DEFAULT_plot_strainHalfWidth = 0.0125

DEFAULT_plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'S(e)_stressRegion':True,
                        'K(S)_stressRegion':True,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'S(e)_nPoints':True,
                        'K(S)_nPoints':True,
                        'S(e)_Log':True, # NEW - Numi
                        'K(S)_Log':True, # NEW - Numi
                        'S(e)_strainGaussian':True, # NEW - Jojo
                        'K(S)_strainGaussian':True, # NEW - Jojo
                        'Plot_Ratio':True, # NEW
                        # Fits plotting parameters
                        # Stress
                        'plotStressCenters':DEFAULT_plot_stressCenters,
                        'plotStressHW':DEFAULT_plot_stressHalfWidth,
                        # Strain
                        'plotStrainCenters':DEFAULT_plot_strainCenters,
                        'plotStrainHW':DEFAULT_plot_strainHalfWidth,
                        # Points
                        'plotPoints':str(DEFAULT_fitSettings['nbPtsFit']) \
                                     + '_' + str(DEFAULT_fitSettings['overlapFit']),
                        'plotLog':str(DEFAULT_fitSettings['nbPtsFitLog']) \
                                     + '_' + str(DEFAULT_fitSettings['overlapFitLog']),
                        }

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
    
    

def makeBoxPlots(dfAllCells, measure, condCol, labels, order, condSelection, savePath, colorPalette = None, removeLegend = False,
                 compAverage = False, cellAverage = False, stats = False, logScale = False):
    fig1, ax = plt.subplots(1, 1, figsize = (10,10))
    fig1.patch.set_facecolor('black')
    plt.style.use('default')

    df = dfAllCells
    condPairs = condSelection

    
    df = df[[measure, condCol, 'cellCode', 'cellId', 'compNum']]
    df = df.drop_duplicates()
    df = df.dropna()   
    N = len(df['cellCode'].unique())
    if  cellAverage == True:
        
        df_og = df
        # print(df_og)
        group_by_cell = df_og.groupby(['cellId'])   
        
        df_average = group_by_cell.agg({measure:['mean'], 'cellCode':'first', 'cellId':'first', condCol : 'first'})
        
        
        condCol_box = condCol, 'first'
        x_box = df_average[condCol_box]
        hue_box = df_average[('cellCode', 'first')]
        # y_box = np.unique(df_average[(measure, 'mean')])
        y_box = (measure, 'mean')
        
        x = df_og[condCol]
        # hue = df_og[('cellCode')]
        hue = df_og[('cellId')]
        y = df_og[(measure)]
        order = order
        
        if compAverage == False:
            sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, color = 'grey',
                                medianprops={"color": 'darkred', "linewidth": 2},\
                                boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.9})
                
            sns.swarmplot(x = x_box, y = y_box, data=df_average, order = order,linewidth = 1, ax = ax,
                          edgecolor='k', size = 12, hue = hue_box, palette = sns.color_palette("Paired", N))
            # sns.swarmplot(x = x_box, y = y_box, data=df_average, order = order,linewidth = 1, ax = ax, edgecolor='k', hue = 'cellId')

            test = 'Mann-Whitney'
            
        
            
        elif compAverage == True:
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

        if compAverage == False:
            sns.boxplot(x = x, y = y, data=df, ax = ax, order = order, color = 'grey',
                                medianprops={"color": 'darkred', "linewidth": 2},\
                                boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.9})
                
            sns.swarmplot(x = x, y = y, data=df, order = order,linewidth = 1, ax = ax, edgecolor='k', size = 8, 
                          palette = sns.color_palette("Paired", N), hue = 'cellCode')
            # sns.swarmplot(x = x, y = y, data=df, order = order,linewidth = 1, ax = ax,  edgecolor='k', hue = 'cellId', size = 6)

        elif compAverage == True:
            
            sns.boxplot(x = x, y = y, data=df, ax = ax, order = order,
                                medianprops={"color": 'darkred', "linewidth": 2},\
                                boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})
            
            sns.pointplot(x = x, y = y, data=df, order = order,hue = 'cellID', ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')
            # sns.pointplot(x = x, y = y, data=df, order = order, hue = 'cellCode', ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')

            
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
    
    if logScale == True:
        plt.yscale('log')
        
    plt.tight_layout()
    # plt.savefig(savePath)
    # plt.ylim(0,1200)
    # plt.show()
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
    # data_ff = data_ff.drop(data_ff[data_ff['E_Chadwick'] > 80000].index)
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

def addStat_df(ax, data, box_pairs, param, cond, test = 'Wilcox_less', percentHeight = 95):
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
        



#%% Mechanics - data analysed before 24/04/2024


plot_stressCenters = [ii for ii in range(100, 4000, 50)]
stressHalfWidths = [50, 75, 100]

fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['pts_15',
                            '%f_5', '%f_10', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'%f_15',
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'centers_StressFits' : plot_stressCenters,
                'halfWidths_StressFits' : stressHalfWidths,
                'doNPointsFits' : True,
                'nbPtsFit' : 33,
                'overlapFit' : 21,
                # NEW - Numi
                'doLogFits' : False,
                # NEW - Jojo
                'doStrainGaussianFits' : False,
                }

plot_stressCenters = [ii for ii in range(100, 4000, 100)]
plot_stressHalfWidth = 75

plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'plotStressCenters':plot_stressCenters,
                        'plotStressHW':plot_stressHalfWidth,
                        'S(e)_nPoints':True,
                        'K(S)_nPoints':True,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        }

    
Task = '24-02-27'
fitsSubDir = 'Chad_f15_MDCK_24-04-02'

GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', 
                            fileName = fitsSubDir,save = True, PLOT = False, source = 'Python',
                            fitSettings = fitSettings, plotSettings = plotSettings,
                            fitsSubDir = fitsSubDir) # task = 'updateExisting'


#%% Mechanics - data analysed after 24/04/2024


# plot_stressCenters = [ii for ii in range(100, 4000, 50)]
# stressHalfWidths = [50, 75, 100]

plot_stressCenters = [ii for ii in range(100, 4000, 100)]
plot_stressHalfWidth = 50
stressHalfWidths = [50]


fitSettings = {# H0
                'methods_H0':['Chadwick'],
                'zones_H0':['lesser_400', '%f_15'],
                'method_bestH0':'Chadwick', # Chadwick
                'zone_bestH0':'lesser_400',
                'doStressRegionFits' : False,
                'doStressGaussianFits' : True,
                'centers_StressFits' : plot_stressCenters,
                'halfWidths_StressFits' : stressHalfWidths,
                'doNPointsFits' : True,
                'nbPtsFit' : 33,
                'overlapFit' : 21,
                # NEW - Numi
                'doLogFits' : False,
                # NEW - Jojo
                'doStrainGaussianFits' : False,
                }


plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'S(e)_stressRegion':False,
                        'K(S)_stressRegion':False,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'plotStressCenters':plot_stressCenters,
                        'plotStressHW':plot_stressHalfWidth,
                        'S(e)_nPoints':True,
                        'K(S)_nPoints':True,
                        'S(e)_strainGaussian':False, # NEW - Jojo
                        'K(S)_strainGaussian':False, # NEW - Jojo
                        'S(e)_Log':False, # NEW - Numi
                        'K(S)_Log':False, # NEW - Numi
                        }

    
Task = '24-02-27'
fitsSubDir = 'Chad_f15_MDCK_24-02-27_24-04-25'

GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', 
                            fileName = fitsSubDir,save = True, PLOT = True, source = 'Python',
                            fitSettings = fitSettings, plotSettings = plotSettings,
                            fitsSubDir = fitsSubDir) # task = 'updateExisting'

 #%%% Calling data

GlobalTable = taka.getMergedTable('Chad_f15_MDCK_24-02-27_24-04-25')
fitsSubDir = 'Chad_f15_MDCK_24-02-27_24-04-25'

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
data_main['cellId'] = GlobalTable['cellID']

nBins = 11
bins = np.linspace(1, 2000, nBins)
data_main['H0_Bin'] = np.digitize(GlobalTable['bestH0'], bins, right = True)
data_main['Thickness_Bin'] = np.digitize(GlobalTable['surroundingThickness'], bins, right = True)

fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75

dirToSave = 'D:/Anumita/MagneticPincherData/Collabs-Internships/Hugo/Meeting_24-04-25'



#%%%% Non - linearity
plt.style.use('seaborn')

data = data_main

########## Declare variables ##########
dates = ['24-02-27']
stressRange = '200_500'
interceptStress = 250
plot = False
FIT_MODE = 'loglog'
# condCol = 'ramp field'
order = None

method = 'f_<_400'
stiffnessType = 'E_' + method
data[stiffnessType + '_kPa'] = data[stiffnessType] / 1000

pathSubDir,pathFits,pathBoxPlots,pathNonlinDir,pathSSPlots,pathKSPlots = makeDirs(fitsSubDir, FIT_MODE)

###########################################

#All cells with standard compression conditions
# condCol = 'manip'
# manips = ['M1','M2', 'M3', 'M4', 'M5']
# # manips = ['M4', 'M5']
# labels = ['Ep. 1', 'Ep. 2', 'Ep. 3', '20um Patterns', 'Single, spread']
# condSelection = manips
# saveExt = manips
# styleDict1 =  {'M1':{'color': gs.colorList40[20] ,'marker':'o', 'label':'Epithelia; Conc. 1'},
#                 'M2':{'color':  gs.colorList40[21],'marker':'o', 'label':'Epithelia; Conc. 2'}, 
#                 'M3':{'color':  gs.colorList40[22],'marker':'o', 'label':'Epithelia; Conc. 3'},
#                 'M4':{'color':  gs.colorList40[23],'marker':'o', 'label':'20um Patterns'},
#                 'M5':{'color':  gs.colorList40[24],'marker':'o', 'label':'Spread single cells'},
#                 }


#All cells with standard compression conditions
condCol = 'substrate'
types = ['epithelia', '20um fibronectin discs', 'fibronectin']
labels = ['Epithelia', '20um fibronectin discs', 'Spread, single cells']
condSelection = labels
saveExt = types
styleDict1 =  {'epithelia':{'color': gs.colorList40[20] ,'marker':'o', 'label':'Epithelia'},
                '20um fibronectin discs':{'color':  gs.colorList40[21],'marker':'o', 'label':'20um Patterns'}, 
                'fibronectin':{'color':  gs.colorList40[22],'marker':'o', 'label':'Spread single cells'},
                }

# condCol = 'manip'
# manips = ['M1', 'M2', 'M3']
# labels = ['Conc. 1', 'Conc. 2', 'Conc. 3']
# condSelection = labels
# saveExt = manips
# styleDict1 =   {'M1':{'color': gs.colorList40[25] ,'marker':'o', 'label':'Epithelia; Conc. 1'},
#                 'M2':{'color':  gs.colorList40[23],'marker':'o', 'label':'Epithelia; Conc. 2'}, 
#                 'M3':{'color':  gs.colorList40[24],'marker':'o', 'label':'Epithelia; Conc. 3'},
#                 }

Filters = [(data['validatedThickness'] == True),
            (data['valid_' + method] == True),
            # (data[stiffnessType + '_kPa'] <= 100),
            # (data['bestH0'] <= 1500),
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            # (data['manip'].apply(lambda x : x in manips)),
            ]

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')

out1, cellDf1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

# plt.legend(fontsize = 20, loc = 'upper left')
# plt.legend(fontsize = 12, loc = 'upper left')
plt.ylim(0,20)
plt.tight_layout()
mainFig1.suptitle(condCol, color = fontColour)

plt.savefig(dirToSave + '/Nonlinearity/' + str(dates) + '_'+str(saveExt) + '_wholeCurve.png')  

mainFig2, mainAx2 = plt.subplots(1,1)
mainFig2.patch.set_facecolor('black')

out2, cellDf2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = condCol, mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

# plt.legend(fontsize = 12, loc = 'upper left')
# plt.ylim(0,30)
# mainFig2.suptitle(condCol, color = fontColour)
# plt.tight_layout()
plt.savefig(dirToSave + '/Nonlinearity/' + str(dates) + '_'+str(saveExt) + '_' + stressRange +'.png')  
# plt.show()

dfAllCells = plot2Params(data, Filters, fitsSubDir, fitType, interceptStress, FIT_MODE, pathFits, plot = plot)


#%%%% Thickness vs. time

plt.style.use('seaborn')


measure = 'surroundingThickness'
activationTime = []

globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]


fig1, axes = plt.subplots(1,1, figsize=(15,10))
fig1.patch.set_facecolor('black')
flatui = ["#000000", "#0000ff"]
sns.set_palette(flatui)

x = (data_f['compNum']-1)*20
ax = sns.lineplot(x = x, y = measure, data = data_f, hue = condCol)
# ax.axvline(x = 5, color = 'red')


# for cell in allCells:
#     # try:
#     meta = pd.read_csv(os.path.join(cp.DirDataRaw+'/'+dateDir, cell+'_disc20um_L40_OptoMetadata.txt'), sep = '\t')
#     times = meta['T_abs'] - meta['T_0']
#     activationTime.append(times.values)
#     # except:
#     #     print('No activation data')
# fig1.suptitle('[15mT = 500pN] '+measure+' (nm) vs. Time (secs)', color = fontColour)

plt.xticks(fontsize=40, color = fontColour)
plt.yticks(fontsize=40, color = fontColour)
plt.xlabel('Time (secs)', fontsize = 25, color = fontColour)
plt.ylabel(measure+' (nm)', fontsize = 25, color = fontColour)
plt.legend(fontsize = 30, loc = 'upper left')

plt.ylim(0,1200)

plt.savefig(dirToSave + '/Thickness_Fluctuations/'+str(dates)+'_'+measure+'vsCompr'+str(saveExt)+'.png')

plt.show()

#%%%% Thicknes/bestH0 boxplots each compression

measure = 'bestH0'
# order= ['Y27_50', 'Y27_10', 'none', 'LIMKi3_1', 'LIMKi3_0.2', 'LIMKi3_0.1', 'blebbi_10', 'dmso']
order = None
labels = labels


savePath = dirToSave + '/Thickness_Fluctuations/'+str(dates)+'_'+measure+'_boxplot_'+str(saveExt)+'.png'

# fig, ax = makeBoxPlots(dfAllCells, measure, condCol, labels = labels, order = order, 
#                         condSelection = condSelection, savePath= savePath, cellAverage = True)

fig, ax = makeBoxPlots(dfAllCells, measure, condCol, labels = labels, order = None,
                        condSelection = condSelection, savePath= savePath, cellAverage = False)

ax.set_ylim(0, 1400)
# ax.get_legend().remove()
plt.legend(loc=2, prop={'size': 6}, ncol = 10)

plt.savefig(savePath)
plt.show()
#%%%% Thickness to cell ID

plt.style.use('default')

y = 'E_Full'
df = dfAllCells
group_by_cell = df.groupby(['cellID'])
df_average = group_by_cell.agg({y:['var', 'std', 'mean', 'count'], 'bestH0':['var', 'std', 'mean', 'count'],
                                'cellCode':'first', condCol:'first'})

y_bestH0 = ('bestH0', 'mean')
y_E = (y, 'mean')
x = ('cellCode', 'first')
hue = (condCol, 'first')

fig, ax1 = plt.subplots(figsize = (18,10))
ax2 = ax1.twinx()

sns.scatterplot(data = df_average, x = x, y = y_bestH0, hue = hue, ax = ax1, s = 100)
sns.scatterplot(data = df_average, x = x, y = y_E, color = 'red', marker="*", s = 100, ax = ax2)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax2.set_yscale('log')
# ax2.set_ylim(0, 30000)

plt.tight_layout()
plt.show()



#%%%% Thickness/Fluctuations boxplots each cell

measure = 'ctFieldThickness'
order = types

savePath = dirToSave + '/Thickness_Fluctuations/'+str(dates)+'_'+measure+'_boxplot_'+str(saveExt)+'.png'
dfAllCellsCtField = dfAllCells.drop_duplicates(subset = 'ctFieldFluctuAmpli', keep = 'first')


fig, ax = makeBoxPlots(dfAllCellsCtField, measure, condCol, labels = labels, order = None,
                        condSelection = condSelection, savePath= savePath, cellAverage = False)

ax.set_ylim(0, 1200)
plt.legend(loc=2, prop={'size': 6}, ncol = 10)
plt.savefig(savePath)
plt.show()

measure = 'ctFieldFluctuAmpli'
order = types

savePath = dirToSave + '/Thickness_Fluctuations/'+str(dates)+'_'+measure+'_boxplot_'+str(saveExt)+'.png'
dfAllCellsCtField = dfAllCells.drop_duplicates(subset = 'ctFieldFluctuAmpli', keep = 'first')


fig, ax = makeBoxPlots(dfAllCellsCtField, measure, condCol, labels = labels, order = None,
                        condSelection = condSelection, savePath= savePath, cellAverage = False)

ax.set_ylim(0, 800)
plt.legend(loc=2, prop={'size': 6}, ncol = 10)

plt.savefig(savePath)
plt.show()

#%%%% E_F_<_400pN

measure = stiffnessType
order = condSelection
labels = labels

measureName = 'E_400pN'

savePath = dirToSave + '/E_400pN/'+str(dates)+'_'+measureName+'_'+str(saveExt)+'.png'


fig, ax = makeBoxPlots(dfAllCells, measure, condCol, labels = labels, order = None, logScale = True,
                        condSelection = condSelection, savePath= savePath, cellAverage = False)

# ax.set_ylim(0, 300000)
plt.legend(loc=2, prop={'size': 6}, ncol = 10)

plt.savefig(savePath)
plt.show()

#%%%% Slopes of K vs S

measure = 'ctFieldFluctuAmpli'
labels = []
order= None
makeBoxPlotParametric

#%%%% Fluctuations vs Thickness
y = 'ctFieldFluctuAmpli'
x = 'ctFieldThickness'
sns.lmplot(data = dfAllCells, x = x, y = y, hue = condCol,ci = 95, fit_reg = True, robust = True)
plt.legend(fontsize = 15, loc = 'upper left')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.show()


#%%%% E_400pN vs Thickness - compressoin  based

dfe = dfAllCells

dfe['bestH0_halved'] = dfe['bestH0']
allepithelia = (dfe['manip'] == 'M1') | (dfe['manip'] == 'M2') | (dfe['manip'] == 'M3')
dfe['bestH0_halved'][allepithelia] = dfe['bestH0_halved'][allepithelia].values / 2

dfe['bestH0_log'] = np.log(dfe['bestH0_halved'].values)

y = stiffnessType 
x = 'bestH0_halved'

sns.scatterplot(data = dfe, x = x, y = y, hue = condCol, style = 'cellId', s = 100)
# plt.show()

# sns.lmplot(data = dfe, x = x, y = y, hue = condCol, fit_reg = True, robust = True)
plt.yscale('log')
plt.xscale('log')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# plt.legend(fontsize = 15, loc = 'upper left')

# plt.yscale('log')
# plt.xscale('log')

# plt.tick_params(axis='y',labelleft='off')

# x_labels = np.asarray([90, 100, 200, 500, 800, 1200])
# xlocs = np.log(x_labels)

# y_labels = np.asarray([300, 500, 2000, 10000, 60000, 200000])
# ylocs = np.log(y_labels)

# plt.xticks(xlocs, (x_labels), fontsize=25)
# plt.yticks(ylocs, (y_labels), fontsize=25)

plt.show()

#%%%% E_400pN vs Thickness - cell based

df_og = dfAllCells
group_by_cell = df_og.groupby(['cellId'], as_index=False)   

df_average = group_by_cell.agg({stiffnessType:'mean', 'bestH0_halved': 'mean',
                                'cellId':'first', condCol : 'first'})

df_average['bestH0_log'] = np.log(df_average['bestH0_halved'].values)

y = stiffnessType 
x = 'bestH0_log'

sns.lmplot(data = df_average, x = x, y = y, hue = condCol, fit_reg = True, robust = True)
plt.legend(fontsize = 15, loc = 'upper left')


# xlocs, xlabels = plt.xticks()  
# x_labels = (np.exp(xlocs)).astype(int)

# ylocs, ylabels = plt.yticks()  
# y_labels = (np.exp(ylocs)).astype(int)


plt.yscale('log')
plt.xscale('log')

plt.tick_params(axis='y',labelleft='off')

x_labels = np.asarray([90, 100, 200, 500, 800, 1200])
xlocs = np.log(x_labels)

y_labels = np.asarray([300, 500, 2000, 10000, 60000, 200000])
ylocs = np.log(y_labels)

                
plt.xticks(xlocs, (x_labels), fontsize=25)
plt.yticks(ylocs, (y_labels), fontsize=25)

plt.show()

#%%%% E_400pN vs Thickness - cell based

# df_og = dfAllCells
# group_by_cell = df_og.groupby(['cellId'], as_index=False)   

# df_average = group_by_cell.agg({stiffnessType:'mean', 'bestH0_halved': 'mean',
#                                 'cellId':'first', condCol : 'first'})
# fig, ax = plt.subplots()

# x = df_average['bestH0_halved'].values
# y = df_average[stiffnessType].values


# wls_model = sm.WLS(x, y)
# results_wls = wls_model.fit()
# S0, K = results_wls.params[0], 1/results_wls.params[1]
# seS0, seK = results_wls.HC3_se[0], results_wls.HC3_se[1]*K**2