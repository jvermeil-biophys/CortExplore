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


# #%% DEFAULT settings

# DEFAULT_centers = [ii for ii in range(100, 1550, 50)]
# DEFAULT_halfWidths = [50, 75, 100]

# DEFAULT_fitSettings = {# H0
#                        'methods_H0':['Dimitriadis'],
#                        'zones_H0':['%f_20'],
#                        'method_bestH0':'Dimitriadis',
#                        'zone_bestH0':'%f_20',
#                        # Global fits
#                        'doChadwickFit' : True,
#                        'doDimitriadisFit' : False,
#                        # Local fits
#                        'doStressRegionFits' : True,
#                        'doStressGaussianFits' : True,
#                        'centers_StressFits' : DEFAULT_centers,
#                        'halfWidths_StressFits' : DEFAULT_halfWidths,
#                        'doNPointsFits' : True,
#                        'nbPtsFit' : 13,
#                        'overlapFit' : 3,
#                        'doLogFits' : True,
#                        'nbPtsFitLog' : 10,
#                        'overlapFitLog' : 5,
#                        }

# DEFAULT_crit_nbPts = 8 # sup or equal to
# DEFAULT_crit_R2 = 0.6 # sup or equal to
# DEFAULT_crit_Chi2 = 1 # inf or equal to
# DEFAULT_str_crit = 'nbPts>{:.0f} - R2>{:.2f} - Chi2<{:.1f}'.format(DEFAULT_crit_nbPts, 
#                                                                    DEFAULT_crit_R2, 
#                                                                    DEFAULT_crit_Chi2)

# DEFAULT_fitValidationSettings = {'crit_nbPts': DEFAULT_crit_nbPts, 
#                                  'crit_R2': DEFAULT_crit_R2, 
#                                  'crit_Chi2': DEFAULT_crit_Chi2,
#                                  'str': DEFAULT_str_crit}

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
                
    if stats:
        if len(box_pairs) == 0:
            box_pairs = makeBoxPairs(co_order)
        addStat_df(ax, data_ff, box_pairs, Parameters[k], CondCol, test = statMethod)
    
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
    
    if legendLabels == []:
        legendLabels = conditions
        
    for i in range(len(conditions)):
        co = conditions[i]
        df = data_agg[data_agg[condCol] == co]
        color = styleDict1[co]['color']
        marker = styleDict1[co]['marker']
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
        
        
        label = '{} | NCells = {}'.format(legendLabels[i], cellCount)
            
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

# newFitSettings = {# H0
#                        'methods_H0':['Chadwick', 'Dimitriadis'],
#                        'zones_H0':['%f_15'],
#                        'method_bestH0':'Chadwick',
#                        'zone_bestH0':'%f_15',
#                        # 'centers_StressFits' : [ii for ii in range(100, 1550, 20)],
#                        #  'halfWidths_StressFits' : [75],
#                        }

# newfitValidationSettings = {'crit_nbPts': 6}


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
plot_stressHalfWidth = 100

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

    
# Task = '22-10-06 & 22-10-05 & 22-12-07'
# Task = '22-12-07 & 23-02-02'
# Task = '23-02-02 & 23-01-23 & 22-12-07 & 22-12-07 & 23-03-28 & 23-03-24'
Task = '23-11-21 & 23-10-29'


#'22-08-26_M7 & 22-08-26_M5 & 22-08-26_M10 & 22-08-26_M1 & 22-08-26_M3' # For instance '22-03-30 & '22-03-31'
fitsSubDir = 'Chad_f15_LIMKi3_23-11-21_23-10-29'

GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', 
                            fileName = fitsSubDir,save = True, PLOT = False, source = 'Python',
                            fitSettings = fitSettings, fitsSubDir = fitsSubDir) # task = 'updateExisting'

# %%%% Specific experiments

# newFitSettings = {# H0
#                        'methods_H0':['Chadwick', 'Dimitriadis'],
#                        'zones_H0':['%f_15'],
#                        'method_bestH0':'Chadwick',
#                        'zone_bestH0':'%f_15',
#                        # 'centers_StressFits' : [ii for ii in range(100, 1550, 20)],
#                        #  'halfWidths_StressFits' : [75],
#                        }

# newfitValidationSettings = {'crit_nbPts': 6}

# fitSettings = ufun.updateDefaultSettingsDict(newFitSettings, DEFAULT_fitSettings)
# fitValidationSettings = ufun.updateDefaultSettingsDict(newfitValidationSettings, \
#                                                         DEFAULT_fitValidationSettings)
    
# Task = '23-03-28 & 22-12-07 & 23-02-02'

# #'22-08-26_M7 & 22-08-26_M5 & 22-08-26_M10 & 22-08-26_M1 & 22-08-26_M3' # For instance '22-03-30 & '22-03-31'
# fitsSubDir = 'Chad_f15_23-03-28 & 22-12-07 & 23-02-02_23-05-03'

# GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', \
#                             fileName = 'Global_MecaData_Chad_f15_23-03-2822-12-07&23-02-02_23-05-03', 
#                             save = True, PLOT = False, source = 'Python', fitSettings = fitSettings,\
#                                fitValidationSettings = fitValidationSettings, fitsSubDir = fitsSubDir) # task = 'updateExisting'

 # %%%% Precise dates (to plot)

date = '22-08-26' # For instance '22-03-30 & '22-03-31'
taka.computeGlobalTable_meca(task = date, mode = 'fromScratch', fileName = 'Global_MecaData_'+date, 
                            save = True, PLOT = True, source = 'Python') # task = 'updateExisting'

 # %%%% Plot all experiments

taka.computeGlobalTable_meca(task = 'all', mode = 'fromScratch', fileName = 'Global_MecaData_all', 
                            save = True, PLOT = True, source = 'Python', fitSettings = fitSettings) # task = 'updateExisting'

#%% Testing new methods here
newFitSettings = {# H0
                       'methods_H0':['Chadwick', 'Dimitriadis'],
                       'zones_H0':['%pts_15', '%f_15'],
                       'method_bestH0':'Dimitriadis',
                       'zone_bestH0':'%f_15',
                       'nbPtsFitLog' : 10,
                       'overlapFitLog' : 4,
                       }


newfitValidationSettings = {'crit_nbPts': 6}
# date = '23-02-02 & 23-01-23 & 22-12-07'

date = '22-12-07_M1'
# date = '22-10-05_M2_P1_C10'
fitsSubDir = '22-12-07_Dimi_f15_tka3'

fitSettings = ufun.updateDefaultSettingsDict(newFitSettings, DEFAULT_fitSettings)
fitValidationSettings = ufun.updateDefaultSettingsDict(newfitValidationSettings, \
                                                       DEFAULT_fitValidationSettings)

GlobalTable_meca =  tka3.computeGlobalTable_meca(task = date, mode = 'fromScratch', \
                            fileName = fitsSubDir, \
                            save = True, PLOT = True, source = 'Python', fitSettings = fitSettings,\
                            fitValidationSettings = fitValidationSettings, fitsSubDir = fitsSubDir) # task = 'updateExisting'

 # %% > Data import & export
#### GlobalTable_meca_Py

date = '22-08-26'
GlobalTable_meca = taka.getMergedTable('Global_MecaData_all')
GlobalTable_meca.head()
# %% Non-linear plots 

# data = taka.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)
# data = data.drop(data[data['fit_error'] == True].index)
# data = data.drop(data[data['fit_K'] < 0].index)
# data = data.dropna(subset = ['fit_ciwK'])

#%% Plots for 22-10-06 and 22-12-07

GlobalTable = taka.getMergedTable('Global_MecaData_22-12-07_Dimi_f15_0Offset') #'_tka3_offset15pts')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
fitsSubDir = '22-12-07_Dimi_f15_0Offset'

fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '200_75'
fitWidth = 75


# %%%% DISPLAY ALL THE H0

data = data_main

# oldManip = ['M7', 'M8', 'M9']

# for i in oldManip:
#     data['manip'][data['manip'] == i] = 'M5'

dates = ['22-10-06']
manips = ['M5', 'M6']

excluded = ['22-10-06_M6_P3_C5','22-10-06_M6_P3_C7', '22-10-06_M5_P3_C5', '22-10-06_M5_P3_C7', '22-10-06_M5_P3_C9', '22-10-06_M6_P3_C9', \
            '22-10-06_M2_P1_C4', '22-10-06_M1_P1_C4', '22-10-06_M2_P1_C5', '22-10-06_M1_P1_C5', \
                        '22-10-06_M2_P2_C5', '22-10-06_M1_P2_C5', '22-10-06_M1_P2_C7', '22-10-06_M1_P2_C8'\
                        '22-10-06_M2_P2_C7', '22-10-06_M2_P2_C8']

for i in excluded:
    data = data[data['cellID'].str.contains(i) == False]
        
Filters = [(data['validatedThickness'] == True),
            (data['bestH0'] <= 2500),
           (data['date'].apply(lambda x : x in dates)),
            (data['manip'].apply(lambda x : x in manips))]

# df_all_H0 = plotAllH0(data, fitsSubDir = 'Dimi_pts15_0Offset', Filters = Filters, condCols = [], 
#         co_order = [], box_pairs = [], AvgPerCell = True,
#         stats = False, statMethod = 'Mann-Whitney', stressBoxPlot = 2)

mainFig, mainAx = plt.subplots(2,1, figsize = (10,8))

plotAllH0(data, mainFig, mainAx[0], fitsSubDir = 'Dimi_f15_0Offset', Filters = Filters, maxH0 = 1000,
        condCols = ['manip'], co_order = [], box_pairs = [], AvgPerCell = False,
        stats = False, statMethod = 'Mann-Whitney', stressBoxPlot = 2)

plotAllH0(data, mainFig, mainAx[1], fitsSubDir = 'Dimi_f15_15Offset', Filters = Filters, maxH0 = 1000,
        condCols = ['manip'], co_order = [], box_pairs = [], AvgPerCell = False,
        stats = False, statMethod = 'Mann-Whitney', stressBoxPlot = 2)


mainAx[0].set_title('0Offset')
mainAx[1].set_title('15Offset')


plt.show()
plt.tight_layout()

# %%%% Plotting over dates, not looped over manip pairs

styleDict1 =  {'M1':{'color': gs.colorList40[10],'marker':'o'},
               'M2':{'color': gs.colorList40[11],'marker':'o'},
               'M3':{'color': gs.colorList40[12],'marker':'o'},
                'M4':{'color': gs.colorList40[30],'marker':'o'},
                'M5':{'color': gs.colorList40[31],'marker':'o'},
                'M6':{'color': gs.colorList40[32],'marker':'o'}
                # 'M7':{'color': gs.colorList40[22],'marker':'o'},
                # 'M8':{'color': gs.colorList40[11],'marker':'o'},
                # 'M9':{'color': gs.colorList40[11],'marker':'o'},
               }

data = data_main

excluded = ['22-10-06_M6_P3_C5','22-10-06_M6_P3_C7', '22-10-06_M5_P3_C5', '22-10-06_M5_P3_C7', '22-10-06_M5_P3_C9', '22-10-06_M6_P3_C9', \
            '22-10-06_M2_P1_C4', '22-10-06_M1_P1_C4', '22-10-06_M2_P1_C5', '22-10-06_M1_P1_C5', \
                        '22-10-06_M2_P2_C5', '22-10-06_M1_P2_C5', '22-10-06_M1_P2_C7', '22-10-06_M1_P2_C8'\
                        '22-10-06_M2_P2_C7', '22-10-06_M2_P2_C8']

for i in excluded:
    data = data[data['cellID'].str.contains(i) == False]


selRows = data[(data['manip'] == 'M6') & (data['compNum'] > 2)].index
data = data.drop(selRows, axis = 0)

dates = ['22-10-06']
manips = ['M5', 'M6'] #,'M6'] #, 'M4', 'M6', 'M5']


Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            # (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            (data['manip'].apply(lambda x : x in manips))]

mainFig, mainAx = plt.subplots(1,1)

out1 = plotPopKS(data, mainFig, mainAx, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'manip', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig, mainAx, exportDf1, countDf1 = out1

# out2 = plotPopKS(data, mainFig, mainAx, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
#                                 condCol = 'manip', mode = stressRange, scale = 'lin', printText = False,
#                                 returnData = 1, returnCount = 1)


    
plt.show()


# %%%% Comparing experiments with 3T3 OptoRhoA, mechanics at 14-15mT field/force range, no activation

cellCond = pd.read_csv(os.path.join(experimentalDataDir, 'cellConditions_22-10-06&22-12-07.csv'))

category = 'Polarised, beads together'
# category = 'Polarised, no lamellipodia'
# category = 'beads non-adhesive'

# control = 'M4'
# active = 'M5'
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

cellCats = cellCond[category]

styleDict1 =  {'M1':{'color': gs.colorList40[10],'marker':'o'},
                'M2':{'color': gs.colorList40[11],'marker':'o'},
                'M3':{'color': gs.colorList40[12],'marker':'o'},
                'M4':{'color': gs.colorList40[30],'marker':'o'},
                'M5':{'color': gs.colorList40[31],'marker':'o'},
                'M6':{'color': gs.colorList40[32],'marker':'o'},
                'M7':{'color': gs.colorList40[31],'marker':'o'}
                }

data = data_main

oldManip = ['M7', 'M8', 'M9']
for i in oldManip:
    data['manip'][data['manip'] == i] = 'M5'
    
selRows = data[(data['manip'] == 'M5') & (data['compNum'] > 3)].index
data = data.drop(selRows, axis = 0)

manips = ['M4', 'M5'] #, 'M6'] #, 'M3'] #, 'M3']
legendLabels = ['No activation', 'Activation at beads']
dates = ['22-12-07']
condCol = 'manip'



Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            # (data['bestH0'] <= 1500),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['date'].apply(lambda x : x in dates)),
            (data['manip'].apply(lambda x : x in manips))]
plt.style.use('default')

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig2, mainAx2 = plt.subplots(1,1)

out1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, legendLabels = legendLabels,
                 fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1


out2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = legendLabels,
                 fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = condCol, mode = '100_500', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

plt.show()


# %%%% Comparing experiments with 15mT field/force range, no activation

cellCond = pd.read_csv(os.path.join(experimentalDataDir, 'cellConditions_22-10-06&22-12-07.csv'))

# category = 'Polarised, beads together'
# category = 'Polarised, no lamellipodia'
category = 'beads non-adhesive'


cellCats = cellCond[category]

# manipPairs = [['M1', 'M2', 'M3'], ['M4', 'M5', 'M6'], ['M1', 'M4'], ['M2', 'M5'], ['M4', 'M5']]
manipPairs = [['M4', 'M5']]

dates = ['22-12-07']
figNames = ['Y27_ctrl-active_', 'nodrug_ctrl-active_', 'nodrug-Y27_ctrl_', 'nodrug-Y27_act_',\
            'nodrug_ctrl-atbeads_']
stressRange = '150_400'

# models = np.asarray(["Chad_pts15_0Offset", "Chad_f15_0Offset", 'Chad_pts15_15Offset', 'Chad_f15_15Offset',
#           'Dimi_pts15_0Offset', 'Dimi_f15_0Offset', 'Dimi_pts15_15Offset', 'Dimi_f15_15Offset'])

models = np.asarray(["Chad_pts15_0Offset", 'Dimi_f15_0Offset', 'Dimi_f15_15Offset'])

mecaFiles = ["Global_MecaData_all_"+i for i in models]

data = data_main

oldManip = ['M7', 'M8', 'M9']

for i in oldManip:
    data['manip'][data['manip'] == i] = 'M5'

styleDict1 =  {'M1':{'color': gs.colorList40[21],'marker':'o'},
                'M2':{'color': gs.colorList40[22],'marker':'o'},
                'M3':{'color': gs.colorList40[24],'marker':'o'},
                'M4':{'color': gs.colorList40[31],'marker':'o'},
                'M5':{'color': gs.colorList40[32],'marker':'o'},
                'M6':{'color': gs.colorList40[34],'marker':'o'}
                }

lenSubplots = len(models)
rows= int(np.floor(np.sqrt(lenSubplots)))
cols= int(np.ceil(lenSubplots/rows))

for i in range(len(manipPairs)):

    manips = manipPairs[i]
    mainFig1, mainAx1 = plt.subplots(nrows = rows, ncols = cols, figsize = (20,10))
    axes1 = []
    
    for k in mainAx1:
        for ax1 in k:
            axes1.append(ax1)
        
    for z, Ax1 in zip(range(lenSubplots), axes1):
        GlobalTable = taka.getMergedTable(mecaFiles[z]) 
        data_main = GlobalTable
        data_main['dateID'] = GlobalTable['date']
        data_main['manipId'] = GlobalTable['manipID']
        
        oldManip = ['M7', 'M8', 'M9']
        
        for p in oldManip:
            data_main['manip'][data_main['manip'] == p] = 'M5'
            
        fitsSubDir = models[z]
        print(mecaFiles[z])        
        data = data_main
        # control = 'M4'
        # active = 'M5'
        # intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
        # controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
        #               for x in intersected]
            
        # allSelectedCells = np.asarray(intersected + controlCells)
        
            
        Filters = [(data['validatedThickness'] == True),
                    # (data['substrate'] == '20um fibronectin discs'), 
                    # (data['drug'] == 'none'), 
                    (data['bead type'] == 'M450'),
                    (data['UI_Valid'] == True),
                    (data['bestH0'] <= 2000),
                    # (data['compNum'][data['manip'] == 'M4'] > 1), 
                    (data['compNum'][data['manip'] == 'M5'] > 2),
                    # (data['cellID'].apply(lambda x : x in allSelectedCells)),
                    (data['date'].apply(lambda x : x in dates)),
                    (data['manip'].apply(lambda x : x in manips))]
        

        out1 = plotPopKS(data, mainFig1, Ax1, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                        condCol = 'manip', mode = stressRange, scale = 'lin', printText = False,
                                        returnData = 1, returnCount = 1)
        
        mainFig1, Ax1, exportDf1, countDf1 = out1
        
        # out2 = plotPopKS(data, mainFig1, Ax1, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
        #                                 condCol = 'manip', mode = stressRange, scale = 'lin', printText = False,
        #                                 returnData = 1, returnCount = 1)
        
        # mainFig2, Ax2, exportDf2, countDf2 = out2
        

        
        
        Ax1.set_title(fitsSubDir)
        
    mainFig1.tight_layout()
    mainFig1.show()
    # mainFig1.savefig(os.path.join(cp.DirDataFig, 'CompleteFigures/Mechanics', dates[0]) + \
    #               '/' + figNames[i] +  stressRange + '.png')
    
        

# %%%% Comparing experiments with 5mT field/force range, no activation

cellCond = pd.read_csv(os.path.join(experimentalDataDir, 'cellConditions_22-10-06&22-12-07.csv'))

category = 'Polarised, beads split'
# category = 'Polarised, no lamellipodia'

cellCats = cellCond[category]

# manipPairs = [['M1', 'M2'], ['M5', 'M6'], ['M1', 'M5'], ['M2', 'M6']]
manipPairs = [['M1', 'M2']]


models = np.asarray(["Chad_pts15_0Offset", "Chad_f15_0Offset", 'Chad_pts15_15Offset', 'Chad_f15_15Offset',
          'Dimi_pts15_0Offset', 'Dimi_f15_0Offset', 'Dimi_pts15_15Offset', 'Dimi_f15_15Offset'])

mecaFiles = ["Global_MecaData_all_"+i for i in models]

dates = ['22-10-06']
figNames = ['5mT_ctrl-act_', '15mT_ctrl-act_', '5-15mT_ctrl_', '5-15mT_act_']
stressRange = '250_400'


styleDict1 =  {'M1':{'color': gs.colorList40[10],'marker':'o'},
                'M2':{'color': gs.colorList40[11],'marker':'o'},
                'M3':{'color': gs.colorList40[20],'marker':'o'},
                'M4':{'color': gs.colorList40[21],'marker':'o'},
                'M5':{'color': gs.colorList40[30],'marker':'o'},
                'M6':{'color': gs.colorList40[31],'marker':'o'}
                }

lenSubplots = len(models)
rows= int(np.floor(np.sqrt(lenSubplots)))
cols= int(np.ceil(lenSubplots/rows))

mainFig1, mainAx1 = plt.subplots(nrows = rows, ncols = cols, figsize = (20,15))
axes1 = []


for i in range(len(manipPairs)):

    manips = manipPairs[i]
    mainFig1, mainAx1 = plt.subplots(nrows = rows, ncols = cols, figsize = (20,10))
    axes1 = []
    
    for k in mainAx1:
        for ax1 in k:
            axes1.append(ax1)
        
    for z, Ax1 in zip(range(lenSubplots), axes1):
        GlobalTable = taka.getMergedTable(mecaFiles[z]) 
        data_main = GlobalTable
        data_main['dateID'] = GlobalTable['date']
        data_main['manipId'] = GlobalTable['manipID']
        
        
        
        fitsSubDir = models[z]
        print(mecaFiles[z])        
        data = data_main
        
        control = 'M1'
        active = 'M2'
        intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
        controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
                      for x in intersected]
            
        allSelectedCells = np.asarray(intersected + controlCells)
        
        Filters = [(data['validatedThickness'] == True),
                    # (data['substrate'] == '20um fibronectin discs'), 
                    # (data['drug'] == 'none'), 
                    (data['bead type'] == 'M450'),
                    (data['UI_Valid'] == True),
                    (data['bestH0'] <= 2000),
                    (data['cellID'].apply(lambda x : x in allSelectedCells)),
                    (data['date'].apply(lambda x : x in dates)),
                    (data['manip'].apply(lambda x : x in manips))]
        
        excluded = ['22-10-06_M6_P3_C5','22-10-06_M6_P3_C7', '22-10-06_M5_P3_C5', '22-10-06_M5_P3_C7', '22-10-06_M5_P3_C9', '22-10-06_M6_P3_C9', \
                    '22-10-06_M2_P1_C4', '22-10-06_M1_P1_C4', '22-10-06_M2_P1_C5', '22-10-06_M1_P1_C5', \
                                '22-10-06_M2_P2_C5', '22-10-06_M1_P2_C5', '22-10-06_M1_P2_C7', '22-10-06_M1_P2_C8'\
                                '22-10-06_M2_P2_C7', '22-10-06_M2_P2_C8']
            
        for j in excluded:
            data = data[data['cellID'].str.contains(j) == False]
        
        
        # oldManip = ['M7', 'M8', 'M9']
        
        # for p in oldManip:
        #     data['manip'][data['manip'] == p] = 'M5'
            
        
        
            
        out1 = plotPopKS(data, mainFig1, Ax1, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                        condCol = 'manip', mode = stressRange, scale = 'lin', printText = False,
                                        returnData = 1, returnCount = 1)
        
        mainFig1, Ax1, exportDf1, countDf1 = out1
        
        # out2 = plotPopKS(data, mainFig1, Ax1, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
        #                                 condCol = 'manip', mode = stressRange, scale = 'lin', printText = False,
        #                                 returnData = 1, returnCount = 1)
        
        # mainFig2, Ax2, exportDf2, countDf2 = out2
        

        
        
        Ax1.set_title(fitsSubDir)
        
    mainFig1.tight_layout()
    mainFig1.show()
    mainFig1.savefig(os.path.join(cp.DirDataFig, 'CompleteFigures/Mechanics', dates[0]) + \
                  '/' + figNames[i] + stressRange+'_BeadsSplit.png')
    
            
            
        # plt.show()

# %%%% Comparing experiments between 5mT and 15mT experiments

# GlobalTable = taka.getMergedTable('Global_MecaData_all')
# data_main = GlobalTable
# data_main['dateID'] = GlobalTable['date']

# fitType = 'stressRegion'
# # fitType = 'nPoints'
# fitId = '_75'

# # c, hw = np.array(fitId.split('_')).astype(int)
# # fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)


styleDict1 =  {'22-12-07_M1':{'color': gs.colorList40[10],'marker':'o'},
                '22-08-26':{'color': gs.colorList40[11],'marker':'o'},
                '22-10-05_M2':{'color': gs.colorList40[11],'marker':'o'},
                '22-12-07_M4':{'color': gs.colorList40[12],'marker':'o'}
                }


# manipIDs = ['22-12-07_M4', '22-08-26_M1', '22-08-26_M2', '22-08-26_M3']
manipIDs = ['22-12-07_M4', '22-12-07_M1', '22-10-05_M2']

# manipIDs = ['22-10-06_M1']
dates = ['22-12-07', '22-10-05']


# excludedCells = ['22-10-06_M6_P3_C5','22-10-06_M6_P3_C7', '22-10-06_M5_P3_C5', '22-10-06_M5_P3_C7', \
#             '22-10-06_M5_P3_C9', '22-10-06_M6_P3_C9', '22-10-06_M2_P1_C4', '22-10-06_M1_P1_C4', '22-10-06_M2_P1_C5', '22-10-06_M1_P1_C5', \
#                         '22-10-06_M2_P2_C5', '22-10-06_M1_P2_C5', '22-10-06_M1_P2_C7', '22-10-06_M1_P2_C8'\
#                         '22-10-06_M2_P2_C7', '22-10-06_M2_P2_C8']
    
# for i in excludedCells:
#     data = data[data['cellID'].str.contains(i) == False]

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig2, mainAx2 = plt.subplots(1,1)


Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            # (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            (data['manipID'].apply(lambda x : x in manipIDs))]


out1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'manipId', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)


mainFig1, mainAx1, exportDf1, countDf1 = out1


out2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'manipId', mode = stressRange, scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

plt.show()
# %%%% Plotting surrounding thickness / best H0 - date-wise

# GlobalTable = taka.getMergedTable('Global_MecaData_all')
# data_main = GlobalTable
# data_main['dateID'] = GlobalTable['date']


data = data_main

styleDict1 =  {'22-12-07':{'color': gs.colorList40[13],'marker':'o'},
                '22-08-26':{'color': gs.colorList40[14],'marker':'o'},
                '22-10-06':{'color': gs.colorList40[15],'marker':'o'}
                }


manipIDs = ['22-12-07_M4', '22-10-05_M2']
dates = ['22-12-07', '22-10-05']


excludedCells = ['22-10-06_M6_P3_C5','22-10-06_M6_P3_C7', '22-10-06_M5_P3_C5', '22-10-06_M5_P3_C7', \
            '22-10-06_M5_P3_C9', '22-10-06_M6_P3_C9', '22-10-06_M2_P1_C4', '22-10-06_M1_P1_C4', '22-10-06_M2_P1_C5', '22-10-06_M1_P1_C5', \
                        '22-10-06_M2_P2_C5', '22-10-06_M1_P2_C5', '22-10-06_M1_P2_C7', '22-10-06_M1_P2_C8'\
                        '22-10-06_M2_P2_C7', '22-10-06_M2_P2_C8']
    
for i in excludedCells:
    data = data[data['cellID'].str.contains(i) == False]
    

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 3000),
            (data['date'].apply(lambda x : x in dates)),
            (data['manipID'].apply(lambda x : x in manipIDs))]


globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

cellcount1 = len(np.unique(data_f['cellID'][data_f['manipId'] == manipIDs[0]].values))
cellcount2 = len(np.unique(data_f['cellID'][data_f['manipId'] == manipIDs[1]].values))

label1 = '{} | NCells = {} '.format(manipIDs[0], cellcount1)
label2 = '{} | NCells = {} '.format(manipIDs[1], cellcount2)


fig1, axes = plt.subplots(1,1, figsize=(15,10))

x = data_f['compNum']*20
sns.lineplot(x = x, y = 'bestH0', data = data_f, hue = 'dateID', ax = axes)

fig1.suptitle(' [15mT = 500pN] bestH0 (nm) vs. Compression No. ')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('Time (secs)', fontsize = 25)
plt.ylabel('Best H0 (nm)', fontsize = 25)
    
# axes.legend(labels=[label1, label2])

plt.savefig(todayFigDir + '/15mT_BestH0vsCompr.png')

plt.show()

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

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 2000),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips))]


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


# %%%% Box plots

# cellCond = pd.read_csv(os.path.join(experimentalDataDir, 'cellConditions_22-10-06&22-12-07.csv'))

# category = 'Polarised, beads together'
# # category = 'Polarised, no lamellipodia'
# # category = 'beads non-adhesive'


# cellCats = cellCond[category]

styleDict1 =  {'M1':{'color': gs.colorList40[10],'marker':'o'},
                'M2':{'color': gs.colorList40[11],'marker':'o'},
                'M3':{'color': gs.colorList40[20],'marker':'o'},
                'M4':{'color': gs.colorList40[21],'marker':'o'},
                'M5':{'color': gs.colorList40[30],'marker':'o'},
                'M6':{'color': gs.colorList40[31],'marker':'o'},
                'M7':{'color': gs.colorList40[31],'marker':'o'}
                }

data = data_main

oldManip = ['M7', 'M8', 'M9']
for i in oldManip:
    data['manip'][data['manip'] == i] = 'M7'
    
selRows = data[(data['manip'] == 'M5') & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)

manips = ['M4', 'M5'] #, 'M6'] #, 'M3'] #, 'M3']
dates = ['22-12-07']
condCol = 'manip'

# control = 'M4'
# active = 'M5'
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            # (data['bestH0'] <= 1500),
            (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (((data['manip'] == 'M5') & (data['compNum'] >= 2)) | (data['manip'] == 'M4')),
            (data['date'].apply(lambda x : x in dates)),
            (data['manip'].apply(lambda x : x in manips))]


globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter] 

fitId = '200_' + str(fitWidth)
data_ff = taka.getFitsInTable(data_f, fitsSubDir, fitType=fitType, filter_fitID=fitId)

# Filter the table
# data_ff = data_ff[(data_ff['fit_center'] >= Sinf) & (data_ff['fit_center'] <= Ssup)]    
data_ff = data_ff.drop(data_ff[data_ff['fit_error'] == True].index)
data_ff = data_ff.drop(data_ff[data_ff['fit_K'] < 0].index)
data_ff = data_ff.drop(data_ff[data_ff['fit_K'] > 9e3].index)
data_ff = data_ff.drop(data_ff[data_ff['fit_R2'] < 0.2].index)
data_ff = data_ff.dropna(subset = ['fit_ciwK'])



mainFig1, mainAx1 = plt.subplots(1,1)

dfValid = data_ff
# y = dfValid['E_Chadwick']

data = dfValid
x = 'manip'
y = 'fit_K'

cellAverage = True

if cellAverage:
    # Compute the weights
    data['weight'] = (data['fit_K']/data['fit_ciwK'])**2
    
    # Compute the weighted mean
    data['A'] = data['fit_K'] * data['weight']
    grouped1 = data.groupby(by=['cellID'])
    data_agg = grouped1.agg({'cellCode' : 'first',
                             x : 'first',
                             'compNum' : 'count',
                            'A': 'sum', 'weight': 'sum'})
    data_agg = data_agg.reset_index()
    data_agg['fit_K'] = data_agg['A']/data_agg['weight']
    data_agg = data_agg.rename(columns = {'compNum' : 'compCount'})
    data = data_agg

sns.boxplot(x = x, y = y, data=data, ax = mainAx1, \
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    
sns.lineplot(data=data, x=x, y=y, units="cellCode",  color = "0.7", estimator=None)

    
sns.swarmplot(x = x, y = y, data=data, linewidth = 1, ax = mainAx1, edgecolor='k')


# mainAx1.set_ylim(0, 50000)
# mainAx1.set_xticklabels(['AtBeadsBefore', 'AtBeadsActivated'])
# plt.setp(axes[0].get_legend().get_texts(), fontsize='10')

addStat_df(mainAx1, data, [('M4', 'M5')], y, test = 'Wilcox_less',  cond = x)

plt.show()

#%%%% K vs. S for each cell

data = data_main 


manips = ['M1', 'M2'] #, '22-12-07_M1'] 
# legendLabels = ['Y27-treated', 'Normally Expressing Pop.', 'Low Expressing Pop.', ]
dates = ['22-12-07']
condCol = 'manip'


Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 2000),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips))]


globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

 
data_ff = taka.getFitsInTable(data_f, fitsSubDir, fitType=fitType, filter_fitID=fitId)

# Filter the table
# data_ff = data_ff[(data_ff['fit_center'] >= Sinf) & (data_ff['fit_center'] <= Ssup)]    
data_ff = data_ff.drop(data_ff[data_ff['fit_error'] == True].index)
data_ff = data_ff.drop(data_ff[data_ff['fit_K'] < 0].index)
data_ff = data_ff.drop(data_ff[data_ff['fit_K'] > 30000].index)

data_ff = data_ff.dropna(subset = ['fit_ciwK'])


sns.lineplot(y = 'fit_K', x = 'fit_center', hue = 'cellCode', data = data_ff)


#%% Plots for 23-01-23

GlobalTable = taka.getMergedTable('Global_MecaData_23-01-23&22-12-07_Dimi_f15')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
fitsSubDir = '23-01-23&22-12-07_Dimi_f15'

fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75


#%%%% Non-linear plots for 23-01-23, experiment with low expressing cells

# cellCond = pd.read_csv(os.path.join(experimentalDataDir, 'cellConditions_22-10-06&22-12-07.csv'))

# category = 'Polarised, beads together'
# # category = 'Polarised, no lamellipodia'
# category = 'beads non-adhesive'


# cellCats = cellCond[category]

styleDict1 =  {'M1':{'color': gs.colorList40[10],'marker':'o'},
                'M2':{'color': gs.colorList40[11],'marker':'o'},
                'M3':{'color': gs.colorList40[12],'marker':'o'},
                'M4':{'color': gs.colorList40[30],'marker':'o'},
                }

data = data_main

selRows = data[(data['manip'] == 'M4') & (data['compNum'] > 3)].index
data = data.drop(selRows, axis = 0)

manips = ['M3', 'M4'] 
legendLabels = ['No activation', 'Activation at beads']
dates = ['23-01-23']
condCol = 'manip'

# control = 'M4'
# active = 'M5'
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            # (data['bestH0'] <= 1500),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['date'].apply(lambda x : x in dates)),
            (data['manip'].apply(lambda x : x in manips))]

plt.style.use('default')

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig2, mainAx2 = plt.subplots(1,1)

out1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, legendLabels = legendLabels,
                 fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1


out2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = legendLabels,
                 fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = condCol, mode = '150_450', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

plt.show()

# %%%% Plotting surrounding thickness / best H0 - manip wise


# cellCond = pd.read_csv(os.path.join(experimentalDataDir, 'cellConditions_22-10-06&22-12-07.csv'))

# plt.style.use('dark_background')
# category = 'Polarised, beads together'
# category = 'Polarised, no lamellipodia'

# category = 'beads non-adhesive'
# cellCats = cellCond[category]

styleDict1 =  {'M1':{'color': gs.colorList40[10],'marker':'o'},
                'M2':{'color': gs.colorList40[11],'marker':'o'},
                'M3':{'color': gs.colorList40[12],'marker':'o'},
                'M4':{'color': gs.colorList40[30],'marker':'o'},
                }

data = data_main

selRows = data[(data['manip'] == 'M4') & (data['compNum'] > 3)].index
data = data.drop(selRows, axis = 0)

manips = ['M3', 'M4'] 
legendLabels = ['No activation', 'Activation at beads']
dates = ['23-01-23']
condCol = 'manip'

# control = 'M4'
# active = 'M5'
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 2000),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips))]


globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

fig1, axes = plt.subplots(1,1, figsize=(15,10))

x = data_f['compNum']*20
sns.lineplot(x = x, y = 'bestH0', data = data_f, hue = 'manipId')

fig1.suptitle('[15mT = 500pN] bestH0 (nm) vs. Time (secs)')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('Time (secs)', fontsize = 25)
plt.ylabel('bestH0 (nm)', fontsize = 25)


plt.ylim(0,1600)

plt.savefig(todayFigDir + '/22-12-07_surroundingThicknessvsCompr.png')

plt.show()


#%%%% Non-linear plots for 23-01-23 and 22-12-07
# Comparing low-exporessing, normal and Y27-treated cells

# cellCond = pd.read_csv(os.path.join(experimentalDataDir, 'cellConditions_22-10-06&22-12-07.csv'))

# category = 'Polarised, beads together'
# # category = 'Polarised, no lamellipodia'
# category = 'beads non-adhesive'


# cellCats = cellCond[category]

styleDict1 =  {'23-01-23_M3':{'color': gs.colorList40[10],'marker':'o'},
                '22-12-07_M4':{'color': gs.colorList40[11],'marker':'o'},
                '22-12-07_M1':{'color': gs.colorList40[12],'marker':'o'},
                }

data = data_main


oldManip = ['23-01-23_M1']
for i in oldManip:
    data['manipId'][data['manipId'] == i] = '23-01-23_M3'
    
# selRows = data[(data['manip'] == 'M5') & (data['compNum'] > 3)].index
# data = data.drop(selRows, axis = 0)

manipsIDs = ['23-01-23_M3', '22-12-07_M4', '22-12-07_M1'] 
legendLabels = ['Y27-treated', 'Normally Expressing Pop.', 'Low Expressing Pop.', ]
dates = ['23-01-23', '22-12-07']
condCol = 'manipId'

# control = 'M4'
# active = 'M5'
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            # (data['bestH0'] <= 1500),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['date'].apply(lambda x : x in dates)),
            (data['manipId'].apply(lambda x : x in manipsIDs))]

plt.style.use('default')

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig2, mainAx2 = plt.subplots(1,1)

out1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, legendLabels = legendLabels,
                 fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1


out2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = legendLabels,
                 fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = condCol, mode = '150_450', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

plt.show()

# %%%% Plotting surrounding thickness / best H0 - manip wise


# cellCond = pd.read_csv(os.path.join(experimentalDataDir, 'cellConditions_22-10-06&22-12-07.csv'))

# plt.style.use('dark_background')
# category = 'Polarised, beads together'
# category = 'Polarised, no lamellipodia'

# category = 'beads non-adhesive'
# cellCats = cellCond[category]

styleDict1 =  {'23-01-23_M3':{'color': gs.colorList40[10],'marker':'o'},
                '22-12-07_M4':{'color': gs.colorList40[11],'marker':'o'},
                '22-12-07_M1':{'color': gs.colorList40[12],'marker':'o'},
                }

data = data_main

oldManip = ['23-01-23_M1']
for i in oldManip:
    data['manipId'][data['manipId'] == i] = '23-01-23_M3'
    
# selRows = data[(data['manip'] == 'M5') & (data['compNum'] > 3)].index
# data = data.drop(selRows, axis = 0)

manipsIDs = ['23-01-23_M3', '22-12-07_M4'] #, '22-12-07_M1'] 
# legendLabels = ['Y27-treated', 'Normally Expressing Pop.', 'Low Expressing Pop.', ]
dates = ['23-01-23', '22-12-07']
condCol = 'manipId'

# control = 'M4'
# active = 'M5'
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 2000),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manipId'].apply(lambda x : x in manipsIDs))]


globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

fig1, axes = plt.subplots(1,1, figsize=(15,10))

x = data_f['compNum']*20
sns.lineplot(x = x, y = 'surroundingThickness', data = data_f, hue = 'manipId')

fig1.suptitle('[15mT = 500pN] Surrounding Thickness (nm) vs. Time (secs)')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('Time (secs)', fontsize = 25)
plt.ylabel('Surrounding thickness (nm)', fontsize = 25)


plt.ylim(0,1500)

plt.savefig(todayFigDir + '/22-12-07_surroundingThicknessvsCompr.png')

plt.show()


#%%%% K vs. S for each cell

data = data_main 


manips = ['M3', 'M4'] #, '22-12-07_M1'] 
# legendLabels = ['Y27-treated', 'Normally Expressing Pop.', 'Low Expressing Pop.', ]
dates = ['23-01-23']
condCol = 'manip'


Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 2000),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips))]


globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

 
data_ff = taka.getFitsInTable(data_f, fitsSubDir, fitType=fitType, filter_fitID=fitId)

# Filter the table
# data_ff = data_ff[(data_ff['fit_center'] >= Sinf) & (data_ff['fit_center'] <= Ssup)]    
data_ff = data_ff.drop(data_ff[data_ff['fit_error'] == True].index)
data_ff = data_ff.drop(data_ff[data_ff['fit_K'] < 0].index)
data_ff = data_ff.drop(data_ff[data_ff['fit_K'] > 30000].index)

data_ff = data_ff.dropna(subset = ['fit_ciwK'])


sns.lineplot(y = 'fit_K', x = 'fit_center', hue = 'cellCode', data = data_ff)

#%%%% Fluctuations vs. median thickness

data = data_main 

oldManip = ['M1']
for i in oldManip:
    data['manip'][data['manip'] == i] = 'M3'
    
oldManip = ['M2']
for i in oldManip:
    data['manip'][data['manip'] == i] = 'M4'

manips = ['M3', 'M4'] #, '22-12-07_M1'] 
# legendLabels = ['Y27-treated', 'Normally Expressing Pop.', 'Low Expressing Pop.', ]
dates = ['23-01-23']
condCol = 'manip'


Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 2000),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips))]


globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

 
data_ff = taka.getFitsInTable(data_f, fitsSubDir, fitType=fitType, filter_fitID=fitId)

# Filter the table
# data_ff = data_ff[(data_ff['fit_center'] >= Sinf) & (data_ff['fit_center'] <= Ssup)]    
data_ff = data_ff.drop(data_ff[data_ff['fit_error'] == True].index)
data_ff = data_ff.drop(data_ff[data_ff['fit_K'] < 0].index)
data_ff = data_ff.drop(data_ff[data_ff['fit_K'] > 30000].index)

data_ff = data_ff.dropna(subset = ['fit_ciwK'])


sns.lmplot(y = 'ctFieldFluctuAmpli', x = 'ctFieldThickness', hue = 'cellCode', data = data_ff)

#%% Plots for 23-02-02, 22-12-07 and 23-01-23

GlobalTable = taka.getMergedTable('Global_MecaData_23-02-02&23-01-23&22-12-07_Dimi_f15') #'_tka3_offset15pts')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
fitsSubDir = '23-02-02&23-01-23&22-12-07_Dimi_f15'

fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75


# %%%% Plotting for each dates, not looped over manip pairs

styleDict1 =  {'M1':{'color': gs.colorList40[10],'marker':'o'},
               'M2':{'color': gs.colorList40[11],'marker':'o'},
               'M3':{'color': gs.colorList40[12],'marker':'o'},
                'M4':{'color': gs.colorList40[30],'marker':'o'},
                'M5':{'color': gs.colorList40[31],'marker':'o'},
                'M6':{'color': gs.colorList40[32],'marker':'o'},
                'M7':{'color': gs.colorList40[22],'marker':'o'},
                'M8':{'color': gs.colorList40[23],'marker':'o'}
               }

data = data_main

# manipIDs = ['23-02-02_M1', '23-02-02_M3', '23-02-02_M5', '23-02-02_M7', '22-12-07_M1']
dates = ['23-02-02']
manips = ['M1', 'M3', 'M5', 'M7'] #, 'M7'] #,'M6'] #, 'M4', 'M6', 'M5']

selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 4)].index
data = data.drop(selRows, axis = 0)

stressRange = '100_400'

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1900),
            (data['date'].apply(lambda x : x in dates)),
            (data['manip'].apply(lambda x : x in manips))]

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig2, mainAx2 = plt.subplots(1,1)

out1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'manip', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

out2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, fitType =  'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'manip', mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

    
plt.show()


# %%%% Comparing experiments with manipIDs

# cellCond = pd.read_csv(os.path.join(experimentalDataDir, 'cellConditions_22-10-06&22-12-07.csv'))

category = 'Polarised, beads together'
# category = 'Polarised, no lamellipodia'
# category = 'beads non-adhesive'


cellCats = cellCond[category]

styleDict1 =  {'23-02-02_M1':{'color': gs.colorList40[10],'marker':'o'},
                '22-12-07_M1':{'color': gs.colorList40[11],'marker':'o'},
                '22-12-07_M4':{'color': gs.colorList40[11],'marker':'o'},
                '23-02-02_M3':{'color': gs.colorList40[12],'marker':'o'},
                '23-02-02_M7':{'color': gs.colorList40[13],'marker':'o'},
                '23-02-02_M5':{'color': gs.colorList40[14],'marker':'o'},
                '23-02-02_M2':{'color': gs.colorList40[10],'marker':'o'},
                '22-12-07_M2':{'color': gs.colorList40[11],'marker':'o'},
                '23-02-02_M4':{'color': gs.colorList40[12],'marker':'o'},
                '23-02-02_M8':{'color': gs.colorList40[13],'marker':'o'},
                '23-01-23_M3':{'color': gs.colorList40[13],'marker':'o'},
                '23-02-02_M6':{'color': gs.colorList40[14],'marker':'o'}
                }

data = data_main

    
# selRows = data[(data['manip'] == 'M5') & (data['compNum'] > 3)].index
# data = data.drop(selRows, axis = 0)

manipIDs = ['23-02-02_M3', '22-12-07_M1', '23-02-02_M7'] #, '23-02-02_M8'] #,  '23-02-02_M5']
legendLabels = ['Ctrl', '50uM', '10uM', '1uM', '100nM']
# legendLabels = ['50uM', 'Ctrl', '10uM', '1uM']
dates = ['23-02-02', '22-12-07', '23-01-23']
condCol = 'manipId'

# control = 'M4'
# active = 'M5'
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1900),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['date'].apply(lambda x : x in dates)),
            (data['manipId'].apply(lambda x : x in manipIDs))]

plt.style.use('default')

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig2, mainAx2 = plt.subplots(1,1)

out1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, 
                 fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1


out2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir,
                 fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = condCol, mode = '200_400', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

plt.show()


#%%%% K vs. S for each cell

data = data_main 

manips = ['M7', 'M8'] 
# manipIDs = ['23-02-02_M3'] #, '22-12-07_M4', '23-01-23_M1', '23-01-23_M3'] 
# legendLabels = ['Y27-treated', 'Normally Expressing Pop.', 'Low Expressing Pop.', ]
# dates = ['23-02-02', '22-12-07', '23-']
condCol = 'manip'
cellCode = ['P1_C10']

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 2000),
            (data['cellCode'].apply(lambda x : x in cellCode)),
            # (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips))]
            # (data['manipId'].apply(lambda x : x in manipIDs))]


globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]
 
data_ff = taka.getFitsInTable(data_f, fitsSubDir, fitType=fitType, filter_fitID=fitId)

# Filter the table
# data_ff = data_ff[(data_ff['fit_center'] >= Sinf) & (data_ff['fit_center'] <= Ssup)]    
data_ff = data_ff.drop(data_ff[data_ff['fit_error'] == True].index)
data_ff = data_ff.drop(data_ff[data_ff['fit_K'] < 0].index)
data_ff = data_ff.drop(data_ff[data_ff['fit_K'] > 60000].index)
data_ff = data_ff.dropna(subset = ['fit_ciwK'])

sns.lineplot(y = 'fit_K', x = 'fit_center', hue = 'manip', style = 'cellCode', data = data_ff)

plt.show()

# %%%% Plotting surrounding thickness / best H0 - manip wise

data = data_main

# selRows = data[(data['manip'] == 'M5') & (data['compNum'] > 3)].index
# data = data.drop(selRows, axis = 0)

manips = ['M3', 'M4'] 
dates = ['23-02-02']
# manipIDs = ['23-02-02_M3', '22-12-07_M1', '23-02-02_M7']
condCol = 'manip'

# control = 'M4'
# active = 'M5'
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1900),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            (data['manip'].apply(lambda x : x in manips))
            ]

globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

fig1, axes = plt.subplots(1,1, figsize=(15,10))

x = data_f['compNum']*20
sns.lineplot(x = x, y = 'bestH0', data = data_f, hue = condCol)

fig1.suptitle('[15mT = 500pN] Surrounding Thickness (nm) vs. Time (secs)')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('Time (secs)', fontsize = 25)
plt.ylabel('Surrounding thickness (nm)', fontsize = 25)

plt.ylim(0,1500)

plt.savefig(todayFigDir + '/23-02-02_surroundingThicknessvsCompr.png')

plt.show()

#%%%% H0 boxplots

data = data_main

# selRows = data[(data['manip'] == 'M4') & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)

manips = ['M3', 'M4'] 
dates = ['23-02-02']
# manipIDs = ['23-02-02_M1', '22-12-07_M4', '23-01-23_M1', '23-01-23_M3'] 
condCol = 'manip'

# control = 'M4'
# active = 'M5'
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            # (data['bestH0'] <= 2000),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            (data['manip'].apply(lambda x : x in manips))]


mainFig, mainAx = plt.subplots(1,1, figsize = (10,8))

plotAllH0(data, mainFig, mainAx, fitsSubDir = fitsSubDir, Filters = Filters, maxH0 = 1000,
        condCols = ['manip'], co_order = [], box_pairs = [], AvgPerCell = False,
        stats = False, statMethod = 'Mann-Whitney', stressBoxPlot = 2)

fig1.suptitle('[15mT = 500pN] Surrounding Thickness (nm) vs. Time (secs)')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('Time (secs)', fontsize = 25)
plt.ylabel('Surrounding thickness (nm)', fontsize = 25)

plt.ylim(0,1500)

plt.savefig(todayFigDir + '/23-02-02_surroundingThicknessvsCompr.png')

plt.show()

#%%%% K boxplots

data = data_main

# selRows = data[(data['manip'] == 'M4') & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)

manips = ['M3', 'M4'] 
dates = ['23-02-02']
# manipIDs = ['23-02-02_M1', '22-12-07_M4', '23-01-23_M1', '23-01-23_M3'] 
condCol = 'manip'

# control = 'M4'

# active = 'M5'
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            # (data['bestH0'] <= 2000),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            (data['manip'].apply(lambda x : x in manips))]

globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]


fitId = '150_' + str(fitWidth)
data_ff = taka.getFitsInTable(data_f, fitsSubDir, fitType=fitType, filter_fitID=fitId)

# Filter the table
# data_ff = data_ff[(data_ff['fit_center'] >= Sinf) & (data_ff['fit_center'] <= Ssup)]    
data_ff = data_ff.drop(data_ff[data_ff['fit_error'] == True].index)
data_ff = data_ff.drop(data_ff[data_ff['fit_K'] < 0].index)
data_ff = data_ff.drop(data_ff[data_ff['fit_K'] > 9e3].index)
data_ff = data_ff.drop(data_ff[data_ff['fit_R2'] < 0.2].index)
data_ff = data_ff.dropna(subset = ['fit_ciwK'])

mainFig1, mainAx1 = plt.subplots(1,1)

dfValid = data_ff
# y = dfValid['E_Chadwick']

data = dfValid
x = 'manip'
y = 'fit_K'

cellAverage = True

if cellAverage:
    # Compute the weights
    data['weight'] = (data['fit_K']/data['fit_ciwK'])**2
    
    # Compute the weighted mean
    data['A'] = data['fit_K'] * data['weight']
    grouped1 = data.groupby(by=['cellID'])
    data_agg = grouped1.agg({'cellCode' : 'first',
                             x : 'first',
                             'compNum' : 'count',
                            'A': 'sum', 'weight': 'sum'})
    data_agg = data_agg.reset_index()
    data_agg['fit_K'] = data_agg['A']/data_agg['weight']
    data_agg = data_agg.rename(columns = {'compNum' : 'compCount'})
    data = data_agg

sns.boxplot(x = x, y = y, data=data, ax = mainAx1, \
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    
sns.lineplot(data=data, x=x, y=y, units="cellCode",  color = "0.7", estimator=None)

    
sns.swarmplot(x = x, y = y, data=data, linewidth = 1, ax = mainAx1, edgecolor='k')

# mainAx1.set_ylim(0, 50000)
# mainAx1.set_xticklabels(['AtBeadsBefore', 'AtBeadsActivated'])
# plt.setp(axes[0].get_legend().get_texts(), fontsize='10')

# addStat_df(mainAx1, data, [(manips[0], manips[1])], y, test = 'Wilcox_less',  cond = x)

plt.show()


#%% Plots for 23-02-02, 22-12-07 and 23-01-23

GlobalTable = taka.getMergedTable('Global_MecaData_all_Dimi_f15') #'_tka3_offset15pts')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
fitsSubDir = 'All_Dimi_f15'

# fitType = 'Log'
fitType = 'stressGaussian'
fitId = '_75' #  = '_75'
fitWidth = 75

#Plot decorations for different dates:

# For 22-12-07
    

#%%%% Non-linearity curves for 22-06-21


plt.style.use('seaborn')

flatui =  ["#e5c100", "#ad6aea", "#000000"]
legendLabels = ['Activation away from beads', 'No Activation']

styleDict1 =  {'M1':{'color': gs.colorList40[10],'marker':'o'},
               'M2':{'color': flatui[1],'marker':'o'},
               'M3':{'color': gs.colorList40[10],'marker':'o'},
                'M4':{'color': gs.colorList40[30],'marker':'o'},
                'M5':{'color': gs.colorList40[31],'marker':'o'},
                'M6':{'color': gs.colorList40[32],'marker':'o'},
                'M7':{'color': gs.colorList40[22],'marker':'o'},
                'M8':{'color': gs.colorList40[23],'marker':'o'}
                }

# styleDict1 =  {'22-05-31_M7_P1_C8':{'color': gs.colorList40[10],'marker':'o'},
#                '22-05-31_M7_P2_C1':{'color': gs.colorList40[11],'marker':'o'},
#                '22-05-31_M7_P2_C3':{'color': gs.colorList40[12],'marker':'o'},
#                }

data = data_main

# manipIDs = ['23-02-02_M1', '23-02-02_M3', '23-02-02_M5', '23-02-02_M7', '22-12-07_M1']
dates = ['22-06-21']
manips = ['M2'] #, 'M2'] #, 'M7'] #,'M6'] #, 'M4', 'M6', 'M5']

stressRange = '200_400'

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1900),
            (data['date'].apply(lambda x : x in dates)),
            (data['manip'].apply(lambda x : x in manips))]

selRows = data[(data['manip'] == manips[0]) & (data['compNum'] < 5)].index
data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')

# mainFig2, mainAx2 = plt.subplots(1,1)

out1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                  legendLabels = legendLabels, condCol = 'manip', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

styleDict1 =  {'M1':{'color': gs.colorList40[10],'marker':'o'},
               'M2':{'color':flatui[2],'marker':'o'},
               'M3':{'color': gs.colorList40[10],'marker':'o'},
                'M4':{'color': gs.colorList40[30],'marker':'o'},
                'M5':{'color': gs.colorList40[31],'marker':'o'},
                'M6':{'color': gs.colorList40[32],'marker':'o'},
                'M7':{'color': gs.colorList40[22],'marker':'o'},
                'M8':{'color': gs.colorList40[23],'marker':'o'}
                }

data = data_main

# manipIDs = ['23-02-02_M1', '23-02-02_M3', '23-02-02_M5', '23-02-02_M7', '22-12-07_M1']
dates = ['22-06-21']
manips = ['M2'] #, 'M2'] #, 'M7'] #,'M6'] #, 'M4', 'M6', 'M5']

stressRange = '200_400'

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1900),
            (data['date'].apply(lambda x : x in dates)),
            (data['manip'].apply(lambda x : x in manips))]

selRows = data[(data['manip'] == manips[0]) & (data['compNum'] > 5)].index
data = data.drop(selRows, axis = 0)


out2 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, fitType =  'stressGaussian', 
                 fitWidth=75, Filters = Filters, legendLabels = legendLabels,
                                condCol = 'manip', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

# mainFig1, mainAx2, exportDf2, countDf2 = out2


# atbeads = mpatches.Patch(color=flatui[0], label='Activation at beads')
# awaybeads = mpatches.Patch(color=flatui[1], label='Activation away from beads')
# control = mpatches.Patch(color=flatui[2], label='Control')

# plt.legend(handles=[awaybeads, control], fontsize = 20, loc = 'upper left')

plt.tight_layout()
plt.show()

mainFig1.savefig(cp.DirDataFigToday + '/temp.png')

#%%%% Non-linearity curves for 22-07-12


plt.style.use('seaborn')

flatui =  ["#e5c100", "#ad6aea", "#000000"]
legendLabels = ['Activation away from beads', 'No Activation']


styleDict1 =  {'23-02-02_M1':{'color': gs.colorList40[10],'marker':'o'},
                '22-12-07_M1':{'color': gs.colorList40[11],'marker':'o'},
                '22-12-07_M4':{'color': gs.colorList40[11],'marker':'o'},
                '23-02-02_M3':{'color': gs.colorList40[12],'marker':'o'},
                '23-02-02_M7':{'color': gs.colorList40[13],'marker':'o'},
                '23-02-02_M5':{'color': gs.colorList40[14],'marker':'o'},
                '23-02-02_M2':{'color': gs.colorList40[10],'marker':'o'},
                '22-12-07_M2':{'color': gs.colorList40[11],'marker':'o'},
                '23-02-02_M4':{'color': gs.colorList40[12],'marker':'o'},
                '23-02-02_M8':{'color': gs.colorList40[13],'marker':'o'},
                '23-01-23_M3':{'color': gs.colorList40[13],'marker':'o'},
                '23-02-02_M6':{'color': gs.colorList40[14],'marker':'o'}
                }


# styleDict1 =  {'M1':{'color': gs.colorList40[10],'marker':'o'},
#                'M2':{'color': gs.colorList40[11],'marker':'o'},
#                'M3':{'color': gs.colorList40[12],'marker':'o'},
#                 'M4':{'color': gs.colorList40[30],'marker':'o'},
#                 'M5':{'color': gs.colorList40[31],'marker':'o'},
#                 'M6':{'color': gs.colorList40[32],'marker':'o'},
#                 'M7':{'color': gs.colorList40[22],'marker':'o'},
#                 'M8':{'color': gs.colorList40[23],'marker':'o'}
#                }

data = data_main

manipIDs = ['23-02-02_M1', '23-02-02_M3', '23-02-02_M5', '23-02-02_M7', '22-12-07_M1']
dates = ['23-02-02']
# legendLabels = ['Ctrl', '10uM', '100nM', '1uM', '50nM']
legendLabels = ['50uM', 'Ctrl', '10uM', '100nM', '1uM']


manips = ['M1', 'M3', 'M5', 'M7'] #, 'M5', 'M7'] #, 'M7'] #,'M6'] #, 'M4', 'M6', 'M5']

# selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)

stressRange = '150_450'

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            # (data['date'].apply(lambda x : x in dates)),
            (data['manipId'].apply(lambda x : x in manipIDs))]

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig2, mainAx2 = plt.subplots(1,1)

out1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir,legendLabels = legendLabels, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'manipId', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

out2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = legendLabels, fitType =  'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'manipId', mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

    
plt.show()


#%%%%
styleDict1 =  {'M1':{'color': gs.colorList40[10],'marker':'o'},
               'M2':{'color':flatui[2],'marker':'o'},
               'M3':{'color': gs.colorList40[10],'marker':'o'},
                'M4':{'color': gs.colorList40[30],'marker':'o'},
                'M5':{'color': gs.colorList40[31],'marker':'o'},
                'M6':{'color': gs.colorList40[32],'marker':'o'},
                'M7':{'color': gs.colorList40[22],'marker':'o'},
                'M8':{'color': gs.colorList40[23],'marker':'o'}
                }

data = data_main

# manipIDs = ['23-02-02_M1', '23-02-02_M3', '23-02-02_M5', '23-02-02_M7', '22-12-07_M1']
dates = ['22-06-21']
manips = ['M2'] #, 'M2'] #, 'M7'] #,'M6'] #, 'M4', 'M6', 'M5']

stressRange = '200_400'

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1900),
            (data['date'].apply(lambda x : x in dates)),
            (data['manip'].apply(lambda x : x in manips))]

selRows = data[(data['manip'] == manips[0]) & (data['compNum'] > 5)].index
data = data.drop(selRows, axis = 0)


out2 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, fitType =  'stressGaussian', 
                 fitWidth=75, Filters = Filters, legendLabels = legendLabels,
                                condCol = 'manip', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

# mainFig1, mainAx2, exportDf2, countDf2 = out2


# atbeads = mpatches.Patch(color=flatui[0], label='Activation at beads')
# awaybeads = mpatches.Patch(color=flatui[1], label='Activation away from beads')
# control = mpatches.Patch(color=flatui[2], label='Control')

# plt.legend(handles=[awaybeads, control], fontsize = 20, loc = 'upper left')

plt.tight_layout()
plt.show()

mainFig1.savefig(cp.DirDataFigToday + '/temp.png')

# %%%% Plotting surrounding thickness / best H0 - manip wise

data = data_main

# selRows = data[(data['manip'] == 'M5') & (data['compNum'] > 3)].index
# data = data.drop(selRows, axis = 0)

dates = ['22-12-07']
manips = ['M1', 'M3'] #, 'M8']
manipIDs = ['23-02-02_M1', '23-02-02_M3', '23-02-02_M5', '23-02-02_M7', '22-12-07_M1']

# manipIDs = ['23-02-02_M3', '22-12-07_M1', '23-02-02_M7']
condCol = 'manipId'
measure = 'surroundingThickness'

# control = 'M4'
# active = 'M5'
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            # (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manipId'].apply(lambda x : x in manipIDs)),
            # (data['manip'].apply(lambda x : x in manips))
            ]

globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]


fig1, axes = plt.subplots(1,1, figsize=(15,10))
# fig1.patch.set_facecolor('black')


x = (data_f['compNum']-1)*20
ax = sns.lineplot(x = x, y = measure, data = data_f, hue = condCol)
# ax.axvline(x = 5, color = 'red')

fig1.suptitle('[15mT = 500pN] '+measure+' (nm) vs. Time (secs)', color = fontColour)
plt.xticks(fontsize=30, color = fontColour)
plt.yticks(fontsize=30, color = fontColour)
plt.xlabel('Time (secs)', fontsize = 25, color = fontColour)
plt.ylabel(measure+' (nm)', fontsize = 25, color = fontColour)
plt.legend(fontsize = 25, loc = 'upper left')

plt.ylim(0,1500)

plt.savefig(todayFigDir + '/'+str(dates)+'_'+measure+'vsCompr'+str(manips)+'.png')

plt.show()

#%%%% Non-linearity curves for 22-10-06

# %%%% Plotting for each dates, not looped over manip pairs

styleDict1 =  {'M1':{'color': gs.colorList40[10],'marker':'o'},
               'M2':{'color': gs.colorList40[11],'marker':'o'},
               'M3':{'color': gs.colorList40[12],'marker':'o'},
                'M4':{'color': gs.colorList40[30],'marker':'o'},
                'M5':{'color': gs.colorList40[31],'marker':'o'},
                'M6':{'color': gs.colorList40[32],'marker':'o'},
                'M7':{'color': gs.colorList40[22],'marker':'o'},
                'M8':{'color': gs.colorList40[23],'marker':'o'}
               }

data = data_main

# manipIDs = ['23-02-02_M1', '23-02-02_M3', '23-02-02_M5', '23-02-02_M7', '22-12-07_M1']
dates = ['22-10-06']
manips = ['M5', 'M6'] #, 'M7'] #,'M6'] #, 'M4', 'M6', 'M5']

selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 4)].index
data = data.drop(selRows, axis = 0)

stressRange = '150_400'

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1900),
            (data['date'].apply(lambda x : x in dates)),
            (data['manip'].apply(lambda x : x in manips))]

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig2, mainAx2 = plt.subplots(1,1)

out1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'manip', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

out2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, fitType =  'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = 'manip', mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

plt.show()

#%%%% Fluctuations vs. median thickness

data = data_main 
plt.style.use('seaborn')

# oldManip = ['M1']
# for i in oldManip:
#     data['manip'][data['manip'] == i] = 'M3'
    
# oldManip = ['M2']
# for i in oldManip:
#     data['manip'][data['manip'] == i] = 'M4'

manips = ['M1', 'M3', 'M5', 'M7'] #, '22-12-07_M1'] 
# legendLabels = ['Y27-treated', 'Normally Expressing Pop.', 'Low Expressing Pop.', ]
dates = ['23-02-02']
condCol = 'manip'


Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['ctFieldFluctuAmpli'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips))]


globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

 
data_ff = taka.getFitsInTable(data_f, fitsSubDir, fitType=fitType, filter_fitID=fitId)

# Filter the table
# data_ff = data_ff[(data_ff['fit_center'] >= Sinf) & (data_ff['fit_center'] <= Ssup)]    
data_ff = data_ff.drop(data_ff[data_ff['fit_error'] == True].index)
data_ff = data_ff.drop(data_ff[data_ff['fit_K'] < 0].index)
data_ff = data_ff.drop(data_ff[data_ff['fit_K'] > 30000].index)

data_ff = data_ff.dropna(subset = ['fit_ciwK'])


sns.lmplot(y = 'ctFieldFluctuAmpli', x = 'ctFieldThickness', hue = 'manip',
                data = data_ff)


#%%%% Plotting all K vs Stress, cell by cell

data = data_main 
plt.style.use('seaborn')


# oldManip = ['M1']
# for i in oldManip:
#     data['manip'][data['manip'] == i] = 'M3'
    
# oldManip = ['M2']
# for i in oldManip:
#     data['manip'][data['manip'] == i] = 'M4'

# manips = ['M1', 'M3', 'M5', 'M7'] #, '22-12-07_M1'] 
manips = ['M4']
# legendLabels = ['Y27-treated', 'Normally Expressing Pop.', 'Low Expressing Pop.', ]
dates = ['23-02-02']
condCol = 'manip'
dfCells =  data['cellID'][data['date'] == dates[0]]
cells = dfCells[dfCells.str.contains(manips[0])].unique()

nColsSubplot = 4
nRowsSubplot = (12 // nColsSubplot) + 1
fig, axes = plt.subplots(nRowsSubplot, nColsSubplot)
_axes = []

for ax_array in axes:
    for ax in ax_array:
        _axes.append(ax)

for i in range(len(cells)):
    allSelectedCells = [cells[i]]
    ax = _axes[i]
    Filters = [(data['validatedThickness'] == True),
                # (data['substrate'] == '20um fibronectin discs'), 
                # (data['drug'] == 'none'), 
                (data['bead type'] == 'M450'),
                (data['UI_Valid'] == True),
                (data['bestH0'] <= 1500),
                (data['date'].apply(lambda x : x in dates)),
                (data['cellID'].apply(lambda x : x in allSelectedCells)),
                (data['manip'].apply(lambda x : x in manips))]
    
    
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
    
    data_ff = data_ff.dropna(subset = ['fit_ciwK'])
    
    
    sns.scatterplot(y = 'fit_K', x = 'fit_center', hue = 'compNum', palette = 'flare',
                              data = data_ff, ax = ax, s = 20)
    ax.set_ylim(0, 30000)
    ax.set_xlim(0, 1200)
    ax.set_title(allSelectedCells[0])
    ax.legend(fontsize = 8)
    


plt.tight_layout()
plt.show()

#%%%% Plotting all K vs Stress, per compression with fits on K

plt.style.use('seaborn')

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
fitsSubDir = 'All_Dimi_f15'

# fitType = 'Log'
fitType = 'nPoints'
fitId = None
Sinf, Ssup = 150, 450
FIT_MODE = 'loglog'  # 'linlin', 'loglog'

data = data_main 

if not os.path.exists(todayFigDir):
    os.mkdir(todayFigDir)

pathSubDir = todayFigDir+'/'+fitsSubDir
if not os.path.exists(pathSubDir):
    os.mkdir(pathSubDir)
    
pathFits = pathSubDir + '/' + FIT_MODE
if not os.path.exists(pathFits):
    os.mkdir(pathFits)


#Only for 23-01-23
# oldManip = ['23-01-23_M1']
# for i in oldManip:
#     data['manipId'][data['manipId'] == i] = '23-01-23_M3'
    
# oldManip = ['M2']
# for i in oldManip:
#     data['manip'][data['manip'] == i] = 'M4'

manipIDs = ['23-02-02_M1', '23-02-02_M3', '23-02-02_M5', '23-02-02_M7', '22-12-07_M1']

manips = ['M1', 'M2', 'M3'] #, 'M5', 'M7'] #, 'M3'] #, 'M5', 'M7'] #, 'M6']
# manipIDs = ['23-01-23_M3', '22-12-07_M4', '23-02-02_M1']
# legendLabels = ['Y27-treated', 'Normally Expressing Pop.', 'Low Expressing Pop.', ]
dates = ['22-12-07']
condCol = 'manipId'
interceptStress = 250

# selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)
# selRows = data[(data['manip'] == 'M3') & (data['compNum'] < 2)].index
# data = data.drop(selRows, axis = 0)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            # (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manip'].apply(lambda x : x in manips)),
            (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

dfAllCells = plot2Params(data, Filters, interceptStress, FIT_MODE, plot = False)

fig1, ax = plt.subplots(1, 2, figsize = (20,20))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']
    
if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips

df = df[[params[0], params[1], condCol, 'chosenIntercept', 'chosenInterceptStress', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

# cellCodes = df['cellCode'][df['manip'] == manips[1]].values
# df = df[df['cellCode'].isin(cellCodes)]

# order = manips #, 'M7', 'M5']
order = ['22-12-07_M1', '23-02-02_M3', '23-02-02_M7', '23-02-02_M5', '23-02-02_M1']

x = df[condCol]
y1, y2 = df[params[0]],  df[params[1]]
# y1, y2 = df[params[0]],  df['chosenIntercept']
sns.boxplot(x = x, y = y1, data=df, ax = ax[0], order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    
sns.swarmplot(x = x, y = y1, data=df,  order = order,linewidth = 1, ax = ax[0], s = 7, edgecolor='k') #, hue = 'cellCode')

sns.boxplot(x = x, y = y2, data=df, ax = ax[1], order = order,
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    
    
sns.swarmplot(x = x, y = y2, data=df,  order = order, linewidth = 1, ax = ax[1], s = 7, edgecolor='k') #, hue = 'cellCode')

box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax[0], data = df, box_pairs = box_pairs, param = params[0], cond = condCol)
addStat_df(ax = ax[1], data = df, box_pairs = box_pairs, param = params[1], cond = condCol)


ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)
# ax[0].set_xticklabels(('1uM Y27', '1uM - Act. at beads'), fontsize = plotTicks)
# ax[0].set_xticklabels(('No drug', '10uM', '1uM', '100nM'), fontsize = plotTicks)
# ax[0].set_xticklabels(('50uM', '50uM - Act. at beads', '50uM - Act. away beads'), fontsize = plotTicks)
# ax[0].set_xticklabels(('50uM', '50uM - Act. at beads'), fontsize = plotTicks)
# ax[0].set_xticklabels(('Control', 'Activation at beads', 'Activation away beads'), fontsize = plotTicks)
# ax[0].set_xticklabels(('No drug', '50uM Y27'), fontsize = plotTicks)
ax[0].set_xticklabels(('50uM', '10uM', '1uM', '100nM', 'No drug'), fontsize = plotTicks)


# ax[1].set_title('K-value at '+str(interceptStress)+'Pa', fontsize = plotLabels)
ax[1].set_title(ax_titles[1], fontsize = plotLabels)
# ax[1].set_xticklabels(('No drug', '10uM', '1uM', '100nM'), fontsize = plotTicks)
# ax[1].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)
# ax[1].set_xticklabels(('1uM Y27', '1uM - Act. at beads'), fontsize = plotTicks)
# ax[1].set_xticklabels(('50uM', '50uM - Act. at beads', '50uM - Act. away beads'), fontsize = plotTicks)
# ax[1].set_xticklabels(('50uM', '50uM - Act. at beads'), fontsize = plotTicks)
# ax[1].set_xticklabels(('Control', 'Activation at beads', 'Activation away beads'), fontsize = plotTicks)
# ax[1].set_xticklabels(('No drug', '50uM Y27'), fontsize = plotTicks)
ax[1].set_xticklabels(('50uM', '10uM', '1uM', '100nM', 'No drug'), fontsize = plotTicks)


ax[0].yaxis.set_tick_params(labelsize=plotTicks)
ax[1].yaxis.set_tick_params(labelsize=plotTicks)
ax[0].xaxis.set_tick_params(labelsize=plotTicks)
ax[1].xaxis.set_tick_params(labelsize=plotTicks)

# ax[0].set_yscale('log')
# ax[0].set_ylim(-150, 100)
# ax[1].set_ylim(-10000, 15000)
# ax[1].get_legend().remove()
fig1.suptitle('{} | {}'.format(str(dates), str(condPairs)))
filename = '{}_{}_{}.png'.format(str(dates), FIT_MODE, str(condPairs))
plt.savefig(todayFigDir, filename)
plt.tight_layout()
plt.show()


# %%%% Plotting surrounding thickness / best H0 - box plots
plt.style.use('seaborn')

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
fitsSubDir = 'All_Dimi_f15'

# fitType = 'Log'
fitType = 'nPoints'
fitId = None
Sinf, Ssup = 150, 450

data = data_main 

#Only for 23-01-23
# oldManip = ['23-01-23_M1']
# for i in oldManip:
#     data['manipId'][data['manipId'] == i] = '23-01-23_M3'
    
# oldManip = ['M2']
# for i in oldManip:
#     data['manip'][data['manip'] == i] = 'M4'

manipIDs = ['23-02-02_M1', '23-02-02_M3', '23-02-02_M5', '23-02-02_M7', '22-12-07_M1']

manips = ['M4', 'M5', 'M6'] #, 'M5', 'M7'] #, 'M3'] #, 'M5', 'M7'] #, 'M6']
# order = manips
order = ['22-12-07_M1', '23-02-02_M3', '23-02-02_M7', '23-02-02_M5', '23-02-02_M1']
# legendLabels = ['Y27-treated', 'Normally Expressing Pop.', 'Low Expressing Pop.', ]
dates = ['22-12-07']
condCol = 'manipId'

if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips


# selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)
# selRows = data[(data['manip'] == manips[2]) & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)
# selRows = data[(data['manip'] == 'M3') & (data['compNum'] < 2)].index
# data = data.drop(selRows, axis = 0)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            # (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manip'].apply(lambda x : x in manips)),
            (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

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
data_ff = data_ff.dropna(subset = ['fit_ciwK'])


fig1, ax1 = plt.subplots(1,1, figsize=(15,10))
# fig1.patch.set_facecolor('black')

x = 'manipId'
y = 'surroundingThickness'

dfAllCells = data_ff[['cellCode', x, 'compNum', y]].drop_duplicates()

sns.boxplot(x = x, y = y, data=dfAllCells, ax = ax1, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    
    
sns.swarmplot(x = x, y = y, data=dfAllCells, order = order, linewidth = 1, \
              ax = ax1, s = 7, edgecolor='k') #, hue = 'cellCode')

box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax1, data = dfAllCells, box_pairs = box_pairs, param = y, cond = condCol)

ax1.set_xticklabels(('50uM', '10uM', '1uM', '100nM', 'No drug'), fontsize = plotTicks)
# ax1.set_xticklabels(('Control', 'Activation at beads', 'Activation away beads'), fontsize = plotTicks)
ax1.yaxis.set_tick_params(labelsize=plotTicks)
ax1.xaxis.set_tick_params(labelsize=plotTicks)

fig1.suptitle(str(dates)+'_[15mT = 500pN] '+y+' (nm) vs. Time (secs)_'+str(condPairs), color = fontColour)

plt.xlabel('Time (secs)', fontsize = 25, color = fontColour)
plt.ylabel(y+' (nm)', fontsize = 25, color = fontColour)
# plt.legend(fontsize = 25, loc = 'upper left')

# plt.ylim(0,1500)

plt.savefig(todayFigDir + '/'+str(dates)+'_'+y+'vsCompr'+str(condPairs)+'.png')

plt.show()


#%% Plots for 23-02-02, 22-12-07 - Testing different ways to plot

GlobalTable = taka.getMergedTable('Global_MecaData_Updated')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
fitsSubDir = 'Dimi_f15_Updated_OversamplingStressGaussian'

# fitType = 'Log'
fitType = 'stressGaussian'
fitId = '_75' #  = '_75'
fitWidth = 75

#%%%% Plotting all K vs Stress, per compression with fits on K
plt.style.use('seaborn')

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
data_main['cellId'] = GlobalTable['cellID']

fitsSubDir = fitsSubDir

fitType = 'stressGaussian'
fitId = '_75'
Sinf, Ssup = 150, 450
FIT_MODE = 'linlin'  # 'linlin', 'loglog'

data = data_main 

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


interceptStress = 150
condCol = 'manip'
# manipIDs = ['23-02-02_M1', '23-02-02_M3', '23-02-02_M5', '23-02-02_M7', '22-12-07_M1']
# order = ['22-12-07_M1', '23-02-02_M3', '23-02-02_M7', '23-02-02_M5', '23-02-02_M1']

manips = ['M1', 'M4']
order = manips #, 'M7', 'M5']

cellIDs = ['22-12-07_M4_P3_C6', '22-12-07_M4_P3_C1', '22-12-07_M8_P3_C6', '22-12-07_M5_P3_C1',\
           '22-12-07_M4_P3_C5', '22-12-07_M5_P3_C5', '22-12-07_M4_P3_C7', '22-12-07_M7_P3_C7']
# order = cellIDs #, 'M7', 'M5']

dates = ['22-12-07']
plot = False


for date in dates:
    if date == '22-12-07' and condCol == 'manip':
        oldManip = ['M7', 'M8', 'M9']
        for i in oldManip:
            data['manip'][(data['date'] == date) & (data['manip'] == i)] = 'M5'

mode = 'Compare activation'
if mode == 'Compare activation' and condCol == 'manip':
    print(gs.ORANGE + 'Considering values after 3rd compression for activate cells' + gs.NORMAL)
    selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
    data = data.drop(selRows, axis = 0)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['cellId'].apply(lambda x : x in cellIDs)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

dfAllCells = plot2Params(data, Filters, fitsSubDir, fitType, interceptStress, FIT_MODE, pathFits, plot = plot)

plt.close('all')

#%%%% Plot box plots
fig1, ax = plt.subplots(1, 2, figsize = (15,10))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']
    
if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs

# df = df[df['R2Fit'] > 0.70]
df = df[[params[0], params[1], condCol, 'chosenIntercept', 'chosenInterceptStress', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

# if mode == 'Compare activation' and condCol == 'manip':
#     cellCodes = df['cellCode'][df['manip'] == manips[1]].values
#     df = df[df['cellCode'].isin(cellCodes)]


plotIntercept = True

if plotIntercept:
    y1, y2 = df[params[0]],  df['chosenIntercept']
    figExt = 'Intercept'+str(interceptStress)
    axtitle = 'Intercept at '+str(interceptStress)+'Pa'
else:
    y1, y2 = df[params[0]],  df[params[1]]
    figExt = 'Coeff'
    axtitle = ax_titles[1]

x = df[condCol]

sns.boxplot(x = x, y = y1, data=dfAllCells, ax = ax[0], order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    
sns.swarmplot(x = x, y = y1, data=dfAllCells,  order = order,linewidth = 1, ax = ax[0], s = 7, edgecolor='k') #, hue = 'cellCode')

sns.boxplot(x = x, y = y2, data=dfAllCells, ax = ax[1], order = order,
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    

sns.swarmplot(x = x, y = y2, data=dfAllCells,  order = order, linewidth = 1, ax = ax[1], s = 7, edgecolor='k') #, hue = 'cellCode')

box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax[0], data = df, box_pairs = box_pairs, param = params[0], cond = condCol)
addStat_df(ax = ax[1], data = df, box_pairs = box_pairs, param = params[1], cond = condCol)


ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)
# ax[0].set_xticklabels(('1uM Y27', '1uM - Act. at beads'), fontsize = plotTicks)
# ax[0].set_xticklabels(('No drug', '10uM', '1uM', '100nM'), fontsize = plotTicks)
# ax[0].set_xticklabels(('50uM', '50uM - Act. at beads', '50uM - Act. away beads'), fontsize = plotTicks)
# ax[0].set_xticklabels(('50uM', '50uM - Act. at beads'), fontsize = plotTicks)
# ax[0].set_xticklabels(('Control', 'Activation at beads', 'Activation away beads'), fontsize = plotTicks)
ax[0].set_xticklabels(('50uM Y27', 'No drug'), fontsize = plotTicks)
# ax[0].set_xticklabels(('50uM', '10uM', '1uM', '100nM', 'No drug'), fontsize = plotTicks)


ax[1].set_title(axtitle, fontsize = plotLabels)
# ax[1].set_xticklabels(('No drug', '10uM', '1uM', '100nM'), fontsize = plotTicks)
ax[1].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)
# ax[1].set_xticklabels(('1uM Y27', '1uM - Act. at beads'), fontsize = plotTicks)
# ax[1].set_xticklabels(('50uM', '50uM - Act. at beads', '50uM - Act. away beads'), fontsize = plotTicks)
# ax[1].set_xticklabels(('50uM', '50uM - Act. at beads'), fontsize = plotTicks)
# ax[1].set_xticklabels(('Control', 'Activation at beads', 'Activation away beads'), fontsize = plotTicks)
ax[1].set_xticklabels(('50uM Y27', 'No drug'), fontsize = plotTicks)
# ax[1].set_xticklabels(('50uM', '10uM', '1uM', '100nM', 'No drug'), fontsize = plotTicks)


ax[0].yaxis.set_tick_params(labelsize=plotTicks)
ax[1].yaxis.set_tick_params(labelsize=plotTicks)
ax[0].xaxis.set_tick_params(labelsize=plotTicks)
ax[1].xaxis.set_tick_params(labelsize=plotTicks)


# ax[1].get_legend().remove()
# ax[0].get_legend().remove()

fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
plt.savefig(pathBoxPlots + '/' + str(dates) + '_'+str(condPairs) + '_' + figExt+'.png')  
plt.show()

#%%%% 2Param plots, testing intracellular quantities

fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']
    
if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs
    
condCol = 'manip'

# df = df[df['R2Fit'] > 0.70]
df = df[[params[0], params[1], condCol, 'chosenIntercept', 'chosenInterceptStress', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

cellCodes = df['cellCode'][df['manip'] == manips[1]].values
df = df[df['cellCode'].isin(cellCodes)]

plotIntercept = True

if plotIntercept:
    y1, y2 = df[params[0]],  df['chosenIntercept']
    figExt = 'Intercept'+str(interceptStress)
    axtitle = 'Intercept at '+str(interceptStress)+'Pa'
else:
    y1, y2 = df[params[0]],  df[params[1]]
    figExt = 'Coeff'
    axtitle = ax_titles[1]

group_by_cell = df.groupby(['cellID'])
df_average = group_by_cell.agg({params[0]:['var', 'std', 'mean', 'count'], 'cellCode':'first', 'manip':'first'})
df_average = df_average[df_average['cellCode'].duplicated(keep = False)]

condCol_box = 'manip', 'first'
x_box = df_average[(condCol_box)]
hue_box = df_average[('cellCode', 'first')]
y_box = df_average[(params[0], 'mean')]

condCol = 'manip'
x = df[(condCol)]
hue = df[('cellCode')]
y = df[(params[0])]


sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={"color" : 'grey',  "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})


# sns.lineplot(data=df_average, x=x, y=y, units=hue,  color = "0.7", ax = ax, estimator=None)

# sns.swarmplot(x = x, y = y, data=df_average, order = order,linewidth = 1, hue = hue, ax = ax, s = 7, edgecolor='k') #, hue = 'cellCode')

sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')


condPairs = [manips[1], manips[0]]
box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df_average, box_pairs = box_pairs, param = params[0], cond = condCol_box,\
            test = 'ranksum_greater')


# ax[0].set_title(ax_titles[0], fontsize = plotLabels)
ax.set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)
ax.set_ylabel(ax_titles[0], fontsize = plotTicks)


ax.yaxis.set_tick_params(labelsize=plotTicks)
ax.xaxis.set_tick_params(labelsize=plotTicks)
handles, labels = ax.get_legend_handles_labels()
newLabels = []
for i in labels:
    count1 = df_average[(params[0], 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[0])].values
    count2 = df_average[(params[0], 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[1])].values
    newLabels.append('{}, No.: {}, {}'.format(i, str(count1), str(count2)))

ax.legend(handles, newLabels)

# # # ax[1].get_legend().remove()
# # # ax[0].get_legend().remove()

fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
plt.savefig(pathBoxPlots + '/' +params[0]+'_'+ str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
plt.show()

#%%%% Whole-fit 'K', testing intracellular quantities

fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells
measure = 'fit_K_kPa'

df = df[[measure, 'dateID', 'manip', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

# if mode == 'Compare activation' and condCol == 'manip':
cellCodes = df['cellCode'][df['manip'] == manips[1]].values
df = df[df['cellCode'].isin(cellCodes)]


group_by_cell = df.groupby(['cellID'])
df_average = group_by_cell.agg({measure:['mean', 'count'], 'cellCode':'first', 'manip':'first'})
df_average = df_average[df_average['cellCode'].duplicated(keep = False)]


condCol_box = 'manip', 'first'
x_box = df_average[condCol_box]
hue_box = df_average[('cellCode', 'first')]
y_box = df_average[('fit_K_kPa', 'mean')]


condCol = 'manip'
x = df[condCol]
hue = df[('cellCode')]
y = df[('fit_K_kPa')]


sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={"color" : 'grey',  "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})


sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')


condPairs = [manips[0], manips[1]]
box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df_average, box_pairs = box_pairs, param = measure, cond = condCol_box,\
            test = 'ranksum_greater')


ax.set_title(measure, fontsize = plotLabels)
ax.set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)
# ax.set_ylabel(ax_titles[0], fontsize = plotTicks)

# ax[1].set_title(axtitle, fontsize = plotLabels)
# # # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax.yaxis.set_tick_params(labelsize=plotTicks)
ax.xaxis.set_tick_params(labelsize=plotTicks)

# handles, labels = ax.get_legend_handles_labels()
# newLabels = []
# for i in labels:
#     count1 = df_average[(measure, 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[0])].values
#     count2 = df_average[(measure, 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[1])].values
#     newLabels.append('{}, No.: {}, {}'.format(i, str(count1), str(count2)))

# ax.legend(handles, newLabels)

# fig1.suptitle(str(dates) + '_'+str(condPairs))
# plt.tight_layout()
# plt.savefig(pathBoxPlots + '/surroundingThickness_' + str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
# plt.show()


#%%%% Surrounding thickness, averaged by cell

fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells

df = df[['surroundingThickness', 'dateID', 'manip', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

# if mode == 'Compare activation' and condCol == 'manip':
cellCodes = df['cellCode'][df['manip'] == manips[1]].values
df = df[df['cellCode'].isin(cellCodes)]


group_by_cell = df.groupby(['cellID'])
df_average = group_by_cell.agg({'surroundingThickness':['mean', 'count'], 'cellCode':'first', 'manip':'first'})
df_average = df_average[df_average['cellCode'].duplicated(keep = False)]


condCol_box = 'manip', 'first'
x_box = df_average[condCol_box]
hue_box = df_average[('cellCode', 'first')]
y_box = df_average[('surroundingThickness', 'mean')]

condCol = 'manip'
x = df[condCol]
hue = df[('cellCode')]
y = df[('surroundingThickness')]

sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={"color" : 'grey',  "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})

# sns.lineplot(data=df_average, x=x, y=y, units=hue,  color = "0.7", ax = ax, estimator=None)

# sns.swarmplot(x = x, y = y, data=df_average, order = order,linewidth = 1, hue = hue, ax = ax, s = 7, edgecolor='k') #, hue = 'cellCode')

sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')

condCol = 'manip', 'first'
condPairs = [manips[1], manips[0]]
box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df_average, box_pairs = box_pairs, param = 'surroundingThickness', cond = condCol,\
            test = 'ranksum_greater')


# ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# # # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


# ax[1].set_title(axtitle, fontsize = plotLabels)
# # # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax.yaxis.set_tick_params(labelsize=plotTicks)
ax.xaxis.set_tick_params(labelsize=plotTicks)

handles, labels = ax.get_legend_handles_labels()
newLabels = []
for i in labels:
    count1 = df_average[('surroundingThickness', 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[0])].values
    count2 = df_average[('surroundingThickness', 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[1])].values
    newLabels.append('{}, No.: {}, {}'.format(i, str(count1), str(count2)))

ax.legend(handles, newLabels)

# fig1.suptitle(str(dates) + '_'+str(condPairs))
# plt.tight_layout()
plt.savefig(pathBoxPlots + '/surroundingThickness_' + str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
# plt.show()


#%%  For experiments 22-05-31, 22-03-31
date = '22-05-31'

GlobalTable = taka.getMergedTable('Global_MecaData_' + str(date))
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
fitsSubDir = 'Dimi_f15_'+str(date)



#%%%% Surrounding thickness
plt.style.use('seaborn')

data = data_main

dates = ['22-05-31']
manips = ['M7']

condCol = 'activation type'
measure = 'surroundingThickness'
force = '[5mT = 50pN]'
activationTime = []

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            # (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manip'].apply(lambda x : x in manips))
            ]

globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

allCells = data_f['cellID'].unique()
dateDir = date.replace('-', '.')

for cell in allCells:
    
    print(cell)
    meta = pd.read_csv(os.path.join(cp.DirDataRaw+'/'+dateDir, cell+'_disc20um_L40_OptoMetadata.txt'), sep = '\t')
    times = meta['T_abs'] - meta['T_0']
    activationTime.append(times.values)

fig1, axes = plt.subplots(1,1, figsize=(15,10))

x = (data_f['compNum']-1)*18
ax = sns.lineplot(x = x, y = measure, data = data_f, hue = 'cellCode')

for each in activationTime[0]:
    ax.axvline(x = each, ymax = .05, color = 'blue', lw = 5)

fig1.suptitle(force+'_'+measure+' (nm) vs. Time (secs)', color = fontColour)
plt.xticks(fontsize=30, color = fontColour)
plt.yticks(fontsize=30, color = fontColour)
plt.xlabel('Time (secs)', fontsize = 25, color = fontColour)
plt.ylabel(measure+' (nm)', fontsize = 25, color = fontColour)
plt.legend(fontsize = 25, loc = 'upper right')

plt.ylim(0,1500)

plt.savefig(todayFigDir + '/'+str(dates)+'_'+measure+'vsCompr_cellSpecific.png')

plt.show()

#%%%% Non-linear plots for 22-03-31

pathNonlinDir = pathSubDir+'/NonLinPlots'
if not os.path.exists(pathNonlinDir):
    os.mkdir(pathNonlinDir)

styleDict1 =  {'M3':{'color': gs.colorList40[10],'marker':'o'},
                'M4':{'color': gs.colorList40[10],'marker':'o'},
                'M5':{'color': gs.colorList40[20],'marker':'o'},
                'M6':{'color': gs.colorList40[10],'marker':'o'},
                'M7':{'color': gs.colorList40[11],'marker':'o'},
                'M8':{'color': gs.colorList40[12],'marker':'o'},
                'M9':{'color': gs.colorList40[14],'marker':'o'},
                '22-05-31':{'color': gs.colorList40[14],'marker':'o'},
                }


data = data_main


manips = ['M7'] #, 'M4'] 
# legendLabels = ['No activation', 'Activation at beads']
dates = ['22-05-31']
condCol = 'manip'
mode = '150_450'

#Before activation
selRows = data[(data['manip'] == manips[0]) & (data['compNum'] > 5)].index
data = data.drop(selRows, axis = 0)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            # (data['bestH0'] <= 1500),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # data['compNum'] < 5, 
            (data['date'].apply(lambda x : x in dates)),
            (data['manip'].apply(lambda x : x in manips)),
            ]

plt.style.use('default')

mainFig1, mainAx1 = plt.subplots(1,1)

out1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, #legendLabels = ['Before Activation'],
                 fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = condCol, mode = mode, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

data = data_main

styleDict1 =  {'M5':{'color': gs.colorList40[21],'marker':'o'},
                'M6':{'color': gs.colorList40[30],'marker':'o'},
                'M7':{'color': gs.colorList40[31],'marker':'o'},
                'M8':{'color': gs.colorList40[32],'marker':'o'},
                'M9':{'color': gs.colorList40[34],'marker':'o'},
                '22-05-31':{'color': gs.colorList40[15],'marker':'o'},

                }

#After activation
selRows = data[(data['manip'] ==  manips[0]) & (data['compNum'] < 5)].index
data = data.drop(selRows, axis = 0)


Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            # (data['bestH0'] <= 1500),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['date'].apply(lambda x : x in dates)),
            (data['manip'].apply(lambda x : x in manips))]


out2 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, legendLabels = ['After activation'],
                  fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = condCol, mode = mode, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf2, countDf2 = out2


plt.tight_layout()
plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'.png')  

plt.show()


# %%%% Plotting surrounding thickness / best H0 - manip wise
plt.style.use('seaborn')

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
data_main['cellId'] = GlobalTable['cellID']

fitsSubDir = fitsSubDir

fitType = 'Log'
fitId = None
Sinf, Ssup = 150, 450
FIT_MODE = 'loglog'  # 'linlin', 'loglog'

data = data_main 

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


interceptStress = 150
condCol = 'dateID'


manips = ['M4', 'M7', 'M5', 'M6']
order = manips #, 'M7', 'M5']

dates = ['22-05-31']
plot = True


for date in dates:
    if date == '22-12-07' and condCol == 'manip':
        oldManip = ['M7', 'M8', 'M9']
        for i in oldManip:
            data['manip'][(data['date'] == date) & (data['manip'] == i)] = 'M5'

mode = 'none'
if mode == 'Compare activation' and condCol == 'manip':
    selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
    data = data.drop(selRows, axis = 0)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['cellId'].apply(lambda x : x in cellIDs)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

dfAllCells = plot2Params(data, Filters, interceptStress, FIT_MODE, pathFits, plot = False)

plt.close('all')


#%%%% Plotting boxplots with above data

fig1, ax = plt.subplots(1, 2, figsize = (15,10))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']


# df = df[df['R2Fit'] > 0.70]
df = df[[params[0], params[1], condCol, 'chosenIntercept', 'chosenInterceptStress', 'dateID', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

activationTag = (np.zeros(len(df)).astype(str))
df['activationTag'] = activationTag
df['activationTag'][df['compNum'] > 5] = '1.0'

df = df.drop_duplicates()
df = df.dropna()

if mode == 'Compare activation' and condCol == 'manip':
    cellCodes = df['cellCode'][df['manip'] == manips[1]].values
    df = df[df['cellCode'].isin(cellCodes)]


plotIntercept = True

if plotIntercept:
    y1, y2 = df[params[0]],  df['chosenIntercept']
    figExt = 'Intercept'+str(interceptStress)
    axtitle = 'Intercept at '+str(interceptStress)+'Pa'
else:
    y1, y2 = df[params[0]],  df[params[1]]
    figExt = 'Coeff'
    axtitle = ax_titles[1]

condCol = 'activationTag'

x = df[condCol]
order = None
hue = 'cellCode'

if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs
elif condCol == 'activationTag':
    condPairs = ['0.0', '1.0']

sns.boxplot(x = x, y = y1, data=df, ax = ax[0], order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    
sns.swarmplot(x = x, y = y1, data=df, order = order,linewidth = 1, hue = hue, ax = ax[0], s = 7, edgecolor='k') #, hue = 'cellCode')

sns.boxplot(x = x, y = y2, data=df, ax = ax[1], order = order,
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    

sns.swarmplot(x = x, y = y2, data=df,  order = order, linewidth = 1, hue = hue, ax = ax[1], s = 7, edgecolor='k') #, hue = 'cellCode')

box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax[0], data = df, box_pairs = box_pairs, param = params[0], cond = condCol)
addStat_df(ax = ax[1], data = df, box_pairs = box_pairs, param = params[1], cond = condCol)


ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax[1].set_title(axtitle, fontsize = plotLabels)
# ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax[0].yaxis.set_tick_params(labelsize=plotTicks)
ax[1].yaxis.set_tick_params(labelsize=plotTicks)
ax[0].xaxis.set_tick_params(labelsize=plotTicks)
ax[1].xaxis.set_tick_params(labelsize=plotTicks)


# ax[1].get_legend().remove()
# ax[0].get_legend().remove()

fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
plt.savefig(pathBoxPlots + '/' + str(dates) + '_'+str(condPairs) + '_' + figExt+'.png')  
plt.show()

#%%%% Plotting boxplots with above data, averaged by cell

fig1, ax = plt.subplots(1, 2, figsize = (15,10))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']


# df = df[df['R2Fit'] > 0.70]
df = df[[params[0], params[1], 'chosenIntercept', 'chosenInterceptStress', 'dateID', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

activationTag = (np.zeros(len(df)).astype(str))
df['activationTag'] = activationTag
df['activationTag'][df['compNum'] > 5] = '1.0'

df = df.drop_duplicates()
df = df.dropna()

if mode == 'Compare activation' and condCol == 'manip':
    cellCodes = df['cellCode'][df['manip'] == manips[1]].values
    df = df[df['cellCode'].isin(cellCodes)]

condCol = 'activationTag', 'first'

if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs
elif condCol == 'activationTag':
    condPairs = ['0.0', '1.0']
    

group_by_cell = df.groupby(['cellCode', 'activationTag'])
df_average = group_by_cell.agg({params[0]:['mean', 'count'], 'chosenIntercept':['mean', 'count'], 
                                   'cellCode':'first', 'activationTag':'first'})
plotIntercept = True
if plotIntercept:
    y1, y2 = df_average[(params[0], 'mean')],  df_average[('chosenIntercept', 'mean')]
    figExt = 'Intercept'+str(interceptStress)
    axtitle = 'Intercept at '+str(interceptStress)+'Pa'
else:
    y1, y2 = df_average[params[0]],  df_average[params[1]]
    figExt = 'Coeff'
    axtitle = ax_titles[1]


x = df_average[condCol]
condPairs = ['0.0', '1.0']
hue = df_average[('cellCode', 'first')]

sns.boxplot(x = x, y = y1, data=df_average, ax = ax[0], order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})

sns.lineplot(data=df_average, x=x, y=y1, units=hue,  color = "0.7", ax = ax[0], estimator=None)

sns.swarmplot(x = x, y = y1, data=df_average, order = order,linewidth = 1, hue = hue, ax = ax[0], s = 7, edgecolor='k') #, hue = 'cellCode')

sns.boxplot(x = x, y = y2, data=df_average, ax = ax[1], order = order,
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})

sns.lineplot(data=df_average, x=x, y=y2, units=hue,  color = "0.7", estimator=None, ax = ax[1])


sns.swarmplot(x = x, y = y2, data=df_average,  order = order, linewidth = 1, hue = hue, ax = ax[1], s = 7, edgecolor='k') #, hue = 'cellCode')

# box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
# addStat_df(ax = ax[0], data = df, box_pairs = box_pairs, param = params[0], cond = condCol)
# addStat_df(ax = ax[1], data = df, box_pairs = box_pairs, param = params[1], cond = condCol)


ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax[1].set_title(axtitle, fontsize = plotLabels)
# # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax[0].yaxis.set_tick_params(labelsize=plotTicks)
ax[1].yaxis.set_tick_params(labelsize=plotTicks)
ax[0].xaxis.set_tick_params(labelsize=plotTicks)
ax[1].xaxis.set_tick_params(labelsize=plotTicks)


# # ax[1].get_legend().remove()
# # ax[0].get_legend().remove()

fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
plt.savefig(pathBoxPlots + '/' + str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
plt.show()

#%%%% fluctuations vs. median thickness

plt.style.use('seaborn')

data = data_main

dates = ['22-05-31']
manips = ['M5']

condCol = 'manip'
force = '[5mT = 50pN]'
activationTime = []

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manip'].apply(lambda x : x in manips))
            ]

globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

allCells = data_f['cellID'].unique()
dateDir = date.replace('-', '.')

plt.figure(figsize = (15,15))
measure = 'ctFieldThickness'
dataToPlot = data_f[[measure, 'ctFieldFluctuAmpli', 'cellCode', 'compNum', 'activation type']].drop_duplicates()
activationTag = (np.zeros(len(dataToPlot)).astype(str))
dataToPlot['activationTag'] = activationTag
dataToPlot['activationTag'][dataToPlot['compNum'] > 5] = '1.0'

sns.lmplot(y = 'ctFieldFluctuAmpli', x = measure, data = dataToPlot)


plt.title(force+'_'+measure+' (nm) vs. Fluctuations (nm)', color = fontColour)
plt.xticks(fontsize=30, color = fontColour)
plt.yticks(fontsize=30, color = fontColour)
plt.ylabel('Fluctuations (nm)', fontsize = 25, color = fontColour)
plt.xlabel(measure+' (nm)', fontsize = 25, color = fontColour)
plt.legend(fontsize = 25, loc = 'upper left')

plt.ylim(0,1500)

plt.savefig(todayFigDir + '/'+str(dates)+'_'+measure+'vsFluctu.png')

plt.show()

#%% Plots for old experiments ------ 22-08-26
date = '22-08-26'

GlobalTable = taka.getMergedTable('Global_MecaData_' + str(date))
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
fitsSubDir = 'Dimi_f15_'+str(date)

#%%%% Surrounding thickness

plt.style.use('seaborn')

data = data_main

dates = ['22-08-26']
manips = ['M1', 'M5']

condCol = 'activation type'
measure = 'surroundingThickness'
force = '[5mT = 50pN]'
activationTime = []

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            # (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips))
            ]

globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

allCells = data_f['cellID'].unique()
dateDir = date.replace('-', '.')

for cell in allCells:
    try:
        meta = pd.read_csv(os.path.join(cp.DirDataRaw+'/'+dateDir, cell+'_disc20um_L40_OptoMetadata.txt'), sep = '\t')
        times = meta['T_abs'] - meta['T_0']
        activationTime.append(times.values)
    except:
        print('No activation data')

fig1, axes = plt.subplots(1,1, figsize=(15,10))

x = (data_f['compNum']-1)*18
ax = sns.lineplot(x = x, y = measure, data = data_f, hue = 'cellCode')

# for each in activationTime[0]:
#     ax.axvline(x = each, ymax = .05, color = 'blue', lw = 5)

fig1.suptitle(force+'_'+measure+' (nm) vs. Time (secs)', color = fontColour)
plt.xticks(fontsize=30, color = fontColour)
plt.yticks(fontsize=30, color = fontColour)
plt.xlabel('Time (secs)', fontsize = 25, color = fontColour)
plt.ylabel(measure+' (nm)', fontsize = 25, color = fontColour)
plt.legend(fontsize = 25, loc = 'upper right')

# plt.ylim(0,1500)

plt.savefig(todayFigDir + '/'+str(dates)+'_'+measure+'vsCompr'+str(manips)+'cellCode.png')

plt.show()

#%%%% Non-linear plots for 22-08-26

if not os.path.exists(todayFigDir):
    os.mkdir(todayFigDir)

pathSubDir = todayFigDir+'/'+fitsSubDir
if not os.path.exists(pathSubDir):
    os.mkdir(pathSubDir)



pathNonlinDir = pathSubDir+'/NonLinPlots'
if not os.path.exists(pathNonlinDir):
    os.mkdir(pathNonlinDir)

plt.style.use('seaborn')

flatui =  ["#e5c100", "#ad6aea", "#000000"]
legendLabels = ['Activation away from beads', 'No Activation']

styleDict1 =  {'M1':{'color': gs.colorList40[10],'marker':'o'},
               'M2':{'color': flatui[1],'marker':'o'},
               'M3':{'color': gs.colorList40[10],'marker':'o'},
                'M4':{'color': gs.colorList40[30],'marker':'o'},
                'M5':{'color': gs.colorList40[31],'marker':'o'},
                'M6':{'color': gs.colorList40[32],'marker':'o'},
                'M7':{'color': gs.colorList40[22],'marker':'o'},
                'M8':{'color': gs.colorList40[23],'marker':'o'}
                }

styleDict1 =  {'22-08-26':{'color': gs.colorList40[10],'marker':'o'},
                '22-05-31_M7_P2_C1':{'color': gs.colorList40[11],'marker':'o'},
                '22-05-31_M7_P2_C3':{'color': gs.colorList40[12],'marker':'o'},
                }

data = data_main

# manipIDs = ['23-02-02_M1', '23-02-02_M3', '23-02-02_M5', '23-02-02_M7', '22-12-07_M1']
dates = ['22-08-26']
manips = ['M3', 'M1'] #, 'M2'] #, 'M7'] #,'M6'] #, 'M4', 'M6', 'M5']

stressRange = '150_450'

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1900),
            (data['date'].apply(lambda x : x in dates)),
            # (data['manip'].apply(lambda x : x in manips)),
            ]

# selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)

out1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = 'dateID', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

mainFig2, mainAx2 = plt.subplots(1,1)


out2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, 
                                condCol = 'dateID', mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2


# atbeads = mpatches.Patch(color=flatui[0], label='Activation at beads')
# awaybeads = mpatches.Patch(color=flatui[1], label='Activation away from beads')
# control = mpatches.Patch(color=flatui[2], label='Control')

# plt.legend(handles=[awaybeads, control], fontsize = 20, loc = 'upper left')


plt.tight_layout()
plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'150-450.png')  

plt.show()


# %%%% Plotting surrounding thickness / best H0 - manip wise
plt.style.use('seaborn')

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
data_main['cellId'] = GlobalTable['cellID']

fitsSubDir = fitsSubDir

fitType = 'Log'
fitId = None
Sinf, Ssup = 150, 450
FIT_MODE = 'linlin'  # 'linlin', 'loglog'

data = data_main 

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


interceptStress = 150
condCol = 'manip'


manips = ['M3', 'M7']
order = manips #, 'M7', 'M5']

dates = ['22-08-26']
plot = True


for date in dates:
    if date == '22-12-07' and condCol == 'manip':
        oldManip = ['M7', 'M8', 'M9']
        for i in oldManip:
            data['manip'][(data['date'] == date) & (data['manip'] == i)] = 'M5'

mode = 'none'
if mode == 'Compare activation' and condCol == 'manip':
    selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
    data = data.drop(selRows, axis = 0)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['cellId'].apply(lambda x : x in cellIDs)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

dfAllCells = plot2Params(data, Filters, interceptStress, FIT_MODE, pathFits, plot = True)

plt.close('all')


#%%%% Plotting boxplots with above data

fig1, ax = plt.subplots(1, 2, figsize = (15,10))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']


# df = df[df['R2Fit'] > 0.70]
df = df[[params[0], params[1], condCol, 'chosenIntercept', 'chosenInterceptStress', 'dateID', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

if mode == 'Compare activation' and condCol == 'manip':
    cellCodes = df['cellCode'][df['manip'] == manips[1]].values
    df = df[df['cellCode'].isin(cellCodes)]


plotIntercept = True

if plotIntercept:
    y1, y2 = df[params[0]],  df['chosenIntercept']
    figExt = 'Intercept'+str(interceptStress)
    axtitle = 'Intercept at '+str(interceptStress)+'Pa'
else:
    y1, y2 = df[params[0]],  df[params[1]]
    figExt = 'Coeff'
    axtitle = ax_titles[1]

condCol = 'manip'

x = df[condCol]
order = None
hue = 'cellCode'

if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs


sns.boxplot(x = x, y = y1, data=df, ax = ax[0], order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    
sns.swarmplot(x = x, y = y1, data=df, order = order,linewidth = 1, hue = hue, ax = ax[0], s = 7, edgecolor='k') #, hue = 'cellCode')

sns.boxplot(x = x, y = y2, data=df, ax = ax[1], order = order,
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    

sns.swarmplot(x = x, y = y2, data=df,  order = order, linewidth = 1, hue = hue, ax = ax[1], s = 7, edgecolor='k') #, hue = 'cellCode')

box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax[0], data = df, box_pairs = box_pairs, param = params[0], cond = condCol)
addStat_df(ax = ax[1], data = df, box_pairs = box_pairs, param = params[1], cond = condCol)


ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax[1].set_title(axtitle, fontsize = plotLabels)
# ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax[0].yaxis.set_tick_params(labelsize=plotTicks)
ax[1].yaxis.set_tick_params(labelsize=plotTicks)
ax[0].xaxis.set_tick_params(labelsize=plotTicks)
ax[1].xaxis.set_tick_params(labelsize=plotTicks)


# ax[1].get_legend().remove()
# ax[0].get_legend().remove()

fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
plt.savefig(pathBoxPlots + '/' + str(dates) + '_'+str(condPairs) + '_' + figExt+'.png')  
plt.show()

#%%%% Plotting boxplots with above data, averaged by cell

fig1, ax = plt.subplots(1, 2, figsize = (15,10))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']


# df = df[df['R2Fit'] > 0.70]
df = df[[params[0], params[1], 'chosenIntercept', 'chosenInterceptStress', 'dateID', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

activationTag = (np.zeros(len(df)).astype(str))
df['activationTag'] = activationTag
df['activationTag'][df['compNum'] > 5] = '1.0'

df = df.drop_duplicates()
df = df.dropna()

if mode == 'Compare activation' and condCol == 'manip':
    cellCodes = df['cellCode'][df['manip'] == manips[1]].values
    df = df[df['cellCode'].isin(cellCodes)]

condCol = 'activationTag', 'first'

if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs
elif condCol == 'activationTag':
    condPairs = ['0.0', '1.0']
    

group_by_cell = df.groupby(['cellCode', 'activationTag'])
df_average = group_by_cell.agg({params[0]:['mean', 'count'], 'chosenIntercept':['mean', 'count'], 
                                   'cellCode':'first', 'activationTag':'first'})
plotIntercept = True
if plotIntercept:
    y1, y2 = df_average[(params[0], 'mean')],  df_average[('chosenIntercept', 'mean')]
    figExt = 'Intercept'+str(interceptStress)
    axtitle = 'Intercept at '+str(interceptStress)+'Pa'
else:
    y1, y2 = df_average[params[0]],  df_average[params[1]]
    figExt = 'Coeff'
    axtitle = ax_titles[1]


x = df_average[condCol]
condPairs = ['0.0', '1.0']
hue = df_average[('cellCode', 'first')]

sns.boxplot(x = x, y = y1, data=df_average, ax = ax[0], order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})

sns.lineplot(data=df_average, x=x, y=y1, units=hue,  color = "0.7", ax = ax[0], estimator=None)

sns.swarmplot(x = x, y = y1, data=df_average, order = order,linewidth = 1, hue = hue, ax = ax[0], s = 7, edgecolor='k') #, hue = 'cellCode')

sns.boxplot(x = x, y = y2, data=df_average, ax = ax[1], order = order,
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})

sns.lineplot(data=df_average, x=x, y=y2, units=hue,  color = "0.7", estimator=None, ax = ax[1])


sns.swarmplot(x = x, y = y2, data=df_average,  order = order, linewidth = 1, hue = hue, ax = ax[1], s = 7, edgecolor='k') #, hue = 'cellCode')

# box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
# addStat_df(ax = ax[0], data = df, box_pairs = box_pairs, param = params[0], cond = condCol)
# addStat_df(ax = ax[1], data = df, box_pairs = box_pairs, param = params[1], cond = condCol)


ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax[1].set_title(axtitle, fontsize = plotLabels)
# # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax[0].yaxis.set_tick_params(labelsize=plotTicks)
ax[1].yaxis.set_tick_params(labelsize=plotTicks)
ax[0].xaxis.set_tick_params(labelsize=plotTicks)
ax[1].xaxis.set_tick_params(labelsize=plotTicks)


# # ax[1].get_legend().remove()
# # ax[0].get_legend().remove()

fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
plt.savefig(pathBoxPlots + '/' + str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
plt.show()

#%%%% fluctuations vs. median thickness

plt.style.use('seaborn')

data = data_main

dates = ['22-08-26']
manips = ['M1', 'M5']

condCol = 'manip'
force = '[5mT = 50pN]'
activationTime = []

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            # (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips))
            ]

globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

allCells = data_f['cellID'].unique()
dateDir = date.replace('-', '.')

# plt.figure(figsize = (15,15))
measure = 'ctFieldThickness'
# dataToPlot = data_f[[measure, 'ctFieldFluctuAmpli', 'cellCode', 'compNum', 'activation type']].drop_duplicates()

fig = sns.lmplot(y = 'ctFieldFluctuAmpli', x = measure, data = data_f, hue = 'manip')


plt.title(force+'_'+measure+' (nm) vs. Fluctuations (nm)', color = fontColour)
plt.xticks(fontsize=30, color = fontColour)
plt.yticks(fontsize=30, color = fontColour)
plt.ylabel('Fluctuations (nm)', fontsize = 25, color = fontColour)
plt.xlabel(measure+' (nm)', fontsize = 25, color = fontColour)
plt.legend(fontsize = 25, loc = 'upper left')

plt.ylim(0,2000)

plt.savefig(todayFigDir + '/'+str(dates)+'_'+measure+'vsFluctu'+str(manips)+'.png')

plt.show()

#%% Plots for old experiments ------ 22-10-05 / 22-10-06
date = '22-10-05&22-10-06'

GlobalTable = taka.getMergedTable('Global_MecaData_' + str(date))
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
fitsSubDir = 'Dimi_f15_'+str(date)

#%%%% Surrounding thickness

plt.style.use('seaborn')

data = data_main

dates = ['22-10-06']
manips = ['M5', 'M6']
manipIDs = ['22-10-05_M2', '22-10-06_M1', '22-10-06_M5']


condCol = 'manipId'
measure = 'surroundingThickness'
force = '[5mT = 50pN]'
activationTime = []

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            # (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manipId'].apply(lambda x : x in manipIDs)),
            # (data['manip'].apply(lambda x : x in manips))
            ]

globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

allCells = data_f['cellID'].unique()
dateDir = date.replace('-', '.')

# for cell in allCells:
#     try:
#         meta = pd.read_csv(os.path.join(cp.DirDataRaw+'/'+dateDir, cell+'_disc20um_L40_OptoMetadata.txt'), sep = '\t')
#         times = meta['T_abs'] - meta['T_0']
#         activationTime.append(times.values)
#     except:
#         print('No activation data')

fig1, axes = plt.subplots(1,1, figsize=(15,10))

x = (data_f['compNum']-1)*18
ax = sns.lineplot(x = x, y = measure, data = data_f, hue = condCol)

# for each in activationTime[0]:
#     ax.axvline(x = each, ymax = .05, color = 'blue', lw = 5)

fig1.suptitle(force+'_'+measure+' (nm) vs. Time (secs)', color = fontColour)
plt.xticks(fontsize=30, color = fontColour)
plt.yticks(fontsize=30, color = fontColour)
plt.xlabel('Time (secs)', fontsize = 25, color = fontColour)
plt.ylabel(measure+' (nm)', fontsize = 25, color = fontColour)
plt.legend(fontsize = 25, loc = 'upper right')

# plt.ylim(0,1500)
# ax.get_legend().remove()

plt.savefig(todayFigDir + '/'+str(dates)+'_'+measure+'vsCompr'+str(manips)+'_'+str(condCol)+'.png')

plt.show()

#%%%% Non-linear plots for 22-10-05 / 22-10-06

if not os.path.exists(todayFigDir):
    os.mkdir(todayFigDir)

pathSubDir = todayFigDir+'/'+fitsSubDir
if not os.path.exists(pathSubDir):
    os.mkdir(pathSubDir)



pathNonlinDir = pathSubDir+'/NonLinPlots'
if not os.path.exists(pathNonlinDir):
    os.mkdir(pathNonlinDir)

plt.style.use('seaborn')

flatui =  ["#e5c100", "#ad6aea", "#000000"]
legendLabels = ['Activation away from beads', 'No Activation']

styleDict1 =  {'M1':{'color': gs.colorList40[20],'marker':'o'},
               'M2':{'color': gs.colorList40[21],'marker':'o'},
               'M3':{'color': gs.colorList40[22],'marker':'o'},
                'M4':{'color': gs.colorList40[30],'marker':'o'},
                'M5':{'color': gs.colorList40[31],'marker':'o'},
                'M6':{'color': gs.colorList40[32],'marker':'o'},
                'M7':{'color': gs.colorList40[22],'marker':'o'},
                'M8':{'color': gs.colorList40[23],'marker':'o'}
                }

# styleDict1 =  {'22-10-05_M2':{'color': gs.colorList40[21],'marker':'o'},
#                 '22-10-06_M1':{'color': gs.colorList40[22],'marker':'o'},
#                 }

data = data_main
condCol = 'manip'
manipIDs = ['22-10-05_M2', '22-10-06_M1']
dates = ['22-10-05']
manips = ['M1', 'M2'] #, 'M2'] #, 'M7'] #,'M6'] #, 'M4', 'M6', 'M5']

stressRange = '150_450'

if condCol == 'manip':
    figExt = manips
elif condCol == 'manipId':
    figExt = manipIDs

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            (data['manip'].apply(lambda x : x in manips)),
            ]

selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)

out1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

mainFig2, mainAx2 = plt.subplots(1,1)


out2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, 
                                condCol = condCol, mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2


# atbeads = mpatches.Patch(color=flatui[0], label='Activation at beads')
# awaybeads = mpatches.Patch(color=flatui[1], label='Activation away from beads')
# control = mpatches.Patch(color=flatui[2], label='Control')

# plt.legend(handles=[awaybeads, control], fontsize = 20, loc = 'upper left')


plt.tight_layout()
mainFig1.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(condCol) +'_'+ str(figExt)+'_wholeCurve.png')  
mainFig2.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(condCol) +'_'+ str(figExt)+ '_' + stressRange + '.png')  


plt.show()


# %%%% Plotting surrounding thickness / best H0 - manip wise
plt.style.use('seaborn')

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
data_main['cellId'] = GlobalTable['cellID']

fitsSubDir = fitsSubDir

fitType = 'Log'
fitId = None
Sinf, Ssup = 150, 450
FIT_MODE = 'linlin'  # 'linlin', 'loglog'

data = data_main 

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


interceptStress = 150
condCol = 'manip'
# manipIDs = ['22-10-05_M2', '22-10-06_M1']



manips = ['M1', 'M5']
order = None #, 'M7', 'M5']

dates = ['22-10-06']
plot = True


for date in dates:
    if date == '22-12-07' and condCol == 'manip':
        oldManip = ['M7', 'M8', 'M9']
        for i in oldManip:
            data['manip'][(data['date'] == date) & (data['manip'] == i)] = 'M5'

mode = 'none'
if mode == 'Compare activation' and condCol == 'manip':
    selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
    data = data.drop(selRows, axis = 0)

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['cellId'].apply(lambda x : x in cellIDs)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

dfAllCells = plot2Params(data, Filters, interceptStress, FIT_MODE, pathFits, plot = True)

plt.close('all')


#%%%% Plotting boxplots with above data

fig1, ax = plt.subplots(1, 2, figsize = (15,10))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']


# df = df[df['R2Fit'] > 0.70]
df = df[[params[0], params[1], condCol, 'chosenIntercept', 'chosenInterceptStress', 'dateID', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

if mode == 'Compare activation' and condCol == 'manip':
    cellCodes = df['cellCode'][df['manip'] == manips[1]].values
    df = df[df['cellCode'].isin(cellCodes)]


plotIntercept = True

if plotIntercept:
    y1, y2 = df[params[0]],  df['chosenIntercept']
    figExt = 'Intercept'+str(interceptStress)
    axtitle = 'Intercept at '+str(interceptStress)+'Pa'
else:
    y1, y2 = df[params[0]],  df[params[1]]
    figExt = 'Coeff'
    axtitle = ax_titles[1]

condCol = 'manip'

x = df[condCol]
order = None
hue = 'cellCode'

if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs


sns.boxplot(x = x, y = y1, data=df, ax = ax[0], order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    
sns.swarmplot(x = x, y = y1, data=df, order = order,linewidth = 1, hue = hue, ax = ax[0], s = 7, edgecolor='k') #, hue = 'cellCode')

sns.boxplot(x = x, y = y2, data=df, ax = ax[1], order = order,
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    

sns.swarmplot(x = x, y = y2, data=df,  order = order, linewidth = 1, hue = hue, ax = ax[1], s = 7, edgecolor='k') #, hue = 'cellCode')

box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax[0], data = df, box_pairs = box_pairs, param = params[0], cond = condCol)
addStat_df(ax = ax[1], data = df, box_pairs = box_pairs, param = params[1], cond = condCol)


ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax[1].set_title(axtitle, fontsize = plotLabels)
# ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax[0].yaxis.set_tick_params(labelsize=plotTicks)
ax[1].yaxis.set_tick_params(labelsize=plotTicks)
ax[0].xaxis.set_tick_params(labelsize=plotTicks)
ax[1].xaxis.set_tick_params(labelsize=plotTicks)


# ax[1].get_legend().remove()
# ax[0].get_legend().remove()

fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
plt.savefig(pathBoxPlots + '/' + str(dates) + '_'+str(condPairs) + '_' + figExt+'.png')  
plt.show()

#%%%% Plotting boxplots with above data, averaged by cell

fig1, ax = plt.subplots(1, 2, figsize = (15,10))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']


# df = df[df['R2Fit'] > 0.70]
df = df[[params[0], params[1], 'chosenIntercept', 'chosenInterceptStress', 'dateID', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

activationTag = (np.zeros(len(df)).astype(str))
df['activationTag'] = activationTag
df['activationTag'][df['compNum'] > 5] = '1.0'

df = df.drop_duplicates()
df = df.dropna()

if mode == 'Compare activation' and condCol == 'manip':
    cellCodes = df['cellCode'][df['manip'] == manips[1]].values
    df = df[df['cellCode'].isin(cellCodes)]

condCol = 'activationTag', 'first'

if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs
elif condCol == 'activationTag':
    condPairs = ['0.0', '1.0']
    

group_by_cell = df.groupby(['cellCode', 'activationTag'])
df_average = group_by_cell.agg({params[0]:['mean', 'count'], 'chosenIntercept':['mean', 'count'], 
                                   'cellCode':'first', 'activationTag':'first'})
plotIntercept = True
if plotIntercept:
    y1, y2 = df_average[(params[0], 'mean')],  df_average[('chosenIntercept', 'mean')]
    figExt = 'Intercept'+str(interceptStress)
    axtitle = 'Intercept at '+str(interceptStress)+'Pa'
else:
    y1, y2 = df_average[params[0]],  df_average[params[1]]
    figExt = 'Coeff'
    axtitle = ax_titles[1]


x = df_average[condCol]
condPairs = ['0.0', '1.0']
hue = df_average[('cellCode', 'first')]

sns.boxplot(x = x, y = y1, data=df_average, ax = ax[0], order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})

sns.lineplot(data=df_average, x=x, y=y1, units=hue,  color = "0.7", ax = ax[0], estimator=None)

sns.swarmplot(x = x, y = y1, data=df_average, order = order,linewidth = 1, hue = hue, ax = ax[0], s = 7, edgecolor='k') #, hue = 'cellCode')

sns.boxplot(x = x, y = y2, data=df_average, ax = ax[1], order = order,
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})

sns.lineplot(data=df_average, x=x, y=y2, units=hue,  color = "0.7", estimator=None, ax = ax[1])


sns.swarmplot(x = x, y = y2, data=df_average,  order = order, linewidth = 1, hue = hue, ax = ax[1], s = 7, edgecolor='k') #, hue = 'cellCode')

# box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
# addStat_df(ax = ax[0], data = df, box_pairs = box_pairs, param = params[0], cond = condCol)
# addStat_df(ax = ax[1], data = df, box_pairs = box_pairs, param = params[1], cond = condCol)


ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax[1].set_title(axtitle, fontsize = plotLabels)
# # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax[0].yaxis.set_tick_params(labelsize=plotTicks)
ax[1].yaxis.set_tick_params(labelsize=plotTicks)
ax[0].xaxis.set_tick_params(labelsize=plotTicks)
ax[1].xaxis.set_tick_params(labelsize=plotTicks)


# # ax[1].get_legend().remove()
# # ax[0].get_legend().remove()

fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
plt.savefig(pathBoxPlots + '/' + str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
plt.show()

#%%%% fluctuations vs. median thickness

plt.style.use('seaborn')

data = data_main

dates = ['22-08-26']
manips = ['M1', 'M5']

condCol = 'manip'
force = '[5mT = 50pN]'
activationTime = []

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            # (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips))
            ]

globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

allCells = data_f['cellID'].unique()
dateDir = date.replace('-', '.')

# plt.figure(figsize = (15,15))
measure = 'ctFieldThickness'
# dataToPlot = data_f[[measure, 'ctFieldFluctuAmpli', 'cellCode', 'compNum', 'activation type']].drop_duplicates()

fig = sns.lmplot(y = 'ctFieldFluctuAmpli', x = measure, data = data_f, hue = 'manip')


plt.title(force+'_'+measure+' (nm) vs. Fluctuations (nm)', color = fontColour)
plt.xticks(fontsize=30, color = fontColour)
plt.yticks(fontsize=30, color = fontColour)
plt.ylabel('Fluctuations (nm)', fontsize = 25, color = fontColour)
plt.xlabel(measure+' (nm)', fontsize = 25, color = fontColour)
plt.legend(fontsize = 25, loc = 'upper left')

plt.ylim(0,2000)

plt.savefig(todayFigDir + '/'+str(dates)+'_'+measure+'vsFluctu'+str(manips)+'.png')

plt.show()

#%% Plots for 23-03-28 --> OptoLARG

GlobalTable = taka.getMergedTable('Global_MecaData_Chad_f15_23-03-2822-12-07&23-02-02_23-05-03')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
fitsSubDir = 'Chad_f15_23-03-28 & 22-12-07 & 23-02-02_23-05-03'

# fitType = 'Log'
fitType = 'stressGaussian'
fitId = '_75' #  = '_75'
fitWidth = 75

#%%%% Cell conditions

cellCond = pd.read_csv(os.path.join(experimentalDataDir, 'cellConditions_23-03-28.csv'))

category = 'phenotype'

cellCats = cellCond['cellID'][(cellCond['phenotype'] != 'none') & (cellCond['phenotype'] != 'blebbing')]
# cellCats = cellCond['cellID'][(cellCond['phenotype'] == 'blebbing')]
# cellCats = cellCond['cellID'][cellCond['phenotype'] != 'blebbing']
# cellCats = cellCond['cellID'][cellCond['phenotype'] == 'blebbing']



#%%%% Non-linear plots

if not os.path.exists(todayFigDir):
    os.mkdir(todayFigDir)

pathSubDir = todayFigDir+'/'+fitsSubDir
if not os.path.exists(pathSubDir):
    os.mkdir(pathSubDir)


pathNonlinDir = pathSubDir+'/NonLinPlots'
if not os.path.exists(pathNonlinDir):
    os.mkdir(pathNonlinDir)

plt.style.use('seaborn')

flatui =  ["#e5c100", "#ad6aea", "#000000"]
legendLabels = ['Activation away from beads', 'No Activation']

styleDict1 =  {'M1':{'color': gs.colorList40[10],'marker':'o'},
               'M2':{'color': flatui[1],'marker':'o'},
               'M3':{'color': gs.colorList40[10],'marker':'o'},
              
                }

data = data_main

dates = ['23-03-28']
manips = ['M1', 'M2'] #, 'M3'] 
manipIDs = ['']

stressRange = '250_650'

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

# allSelectedCells = np.asarray(['23-03-28_M1_P1_C12', '23-03-28_M2_P1_C12', '23-03-28_M3_P1_C12'
#                                '23-03-28_M1_P2_C5', '23-03-28_M2_P2_C5', '23-03-28_M3_P2_C5'])


Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1900),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips)),
            ]

selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)

out1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = 'manip', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

mainFig2, mainAx2 = plt.subplots(1,1)


out2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = 'manip', mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2


# atbeads = mpatches.Patch(color=flatui[0], label='Activation at beads')
# awaybeads = mpatches.Patch(color=flatui[1], label='Activation away from beads')
# control = mpatches.Patch(color=flatui[2], label='Control')

# plt.legend(handles=[awaybeads, control], fontsize = 20, loc = 'upper left')


plt.tight_layout()
# plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'150-450.png')  

plt.show()


#%%%% Plotting all K vs Stress, per compression with fits on K
plt.style.use('seaborn')

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
data_main['cellId'] = GlobalTable['cellID']

fitsSubDir = fitsSubDir

fitType = 'stressGaussian'
fitId = '_75'
Sinf, Ssup = 150, 500
FIT_MODE = 'loglog'  # 'linlin', 'loglog'

data = data_main 

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


interceptStress = 150
condCol = 'manip'

manips = ['M1', 'M2'] #, 'M3']
order = manips 

dates = ['23-03-28']
plot = False

mode = 'Compare activation'
if mode == 'Compare activation' and condCol == 'manip':
    print(gs.ORANGE + 'Considering values after 3rd compression for activate cells' + gs.NORMAL)
    selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
    data = data.drop(selRows, axis = 0)

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

# allSelectedCells = ['23_03-28_M1']

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['cellId'].apply(lambda x : x in cellIDs)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            ]


dfAllCells = plot2Params(data, Filters, fitsSubDir, fitType, interceptStress, FIT_MODE, pathFits, plot = plot)

plt.close('all')



#%%%% Plot box plots
fig1, ax = plt.subplots(1, 2, figsize = (15,10))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']
    
if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs

# df = df[df['R2Fit'] > 0.70]
df = df[[params[0], params[1], condCol, 'chosenIntercept', 'chosenInterceptStress', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

if mode == 'Compare activation' and condCol == 'manip':
    cellCodes = df['cellCode'][df['manip'] == manips[1]].values
    df = df[df['cellCode'].isin(cellCodes)]


plotIntercept = True

if plotIntercept:
    y1, y2 = df[params[0]],  df['chosenIntercept']
    figExt = 'Intercept'+str(interceptStress)
    axtitle = 'Intercept at '+str(interceptStress)+'Pa'
else:
    y1, y2 = df[params[0]],  df[params[1]]
    figExt = 'Coeff'
    axtitle = ax_titles[1]

x = df[condCol]

sns.boxplot(x = x, y = y1, data=dfAllCells, ax = ax[0], order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    
sns.swarmplot(x = x, y = y1, data=dfAllCells, hue = 'cellCode', order = order,linewidth = 1, ax = ax[0], s = 7, edgecolor='k') #, hue = 'cellCode')

sns.boxplot(x = x, y = y2, data=dfAllCells, ax = ax[1], order = order,
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    

sns.swarmplot(x = x, y = y2, hue = 'cellCode', data=dfAllCells,  order = order, linewidth = 1, ax = ax[1], s = 7, edgecolor='k') #, hue = 'cellCode')

box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax[0], data = df, box_pairs = box_pairs, param = params[0], cond = condCol)
addStat_df(ax = ax[1], data = df, box_pairs = box_pairs, param = params[1], cond = condCol)


ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax[1].set_title(axtitle, fontsize = plotLabels)
# ax[1].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax[0].yaxis.set_tick_params(labelsize=plotTicks)
ax[1].yaxis.set_tick_params(labelsize=plotTicks)
ax[0].xaxis.set_tick_params(labelsize=plotTicks)
ax[1].xaxis.set_tick_params(labelsize=plotTicks)


# ax[1].get_legend().remove()
# ax[0].get_legend().remove()

fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
plt.savefig(pathBoxPlots + '/' + str(dates) + '_'+str(condPairs) + '_' + figExt+'.png')  
plt.show()

#%%%% 2Param plots, testing intracellular quantities

fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']
    
if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs
    
condCol = 'manip'
df = df[[params[0], params[1], condCol, 'chosenIntercept', 'chosenInterceptStress', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

cellCodes = df['cellCode'][df['manip'] == manips[1]].values
df = df[df['cellCode'].isin(cellCodes)]

plotIntercept = True

if plotIntercept:
    y1, y2 = df[params[0]],  df['chosenIntercept']
    figExt = 'Intercept'+str(interceptStress)
    axtitle = 'Intercept at '+str(interceptStress)+'Pa'
else:
    y1, y2 = df[params[0]],  df[params[1]]
    figExt = 'Coeff'
    axtitle = ax_titles[1]

group_by_cell = df.groupby(['cellID'])
df_average = group_by_cell.agg({params[0]:['var', 'std', 'mean', 'count'], 'cellCode':'first', 'manip':'first'})
df_average = df_average[df_average['cellCode'].duplicated(keep = False)]

condCol_box = 'manip', 'first'
x_box = df_average[(condCol_box)]
hue_box = df_average[('cellCode', 'first')]
y_box = df_average[(params[0], 'mean')]

condCol = 'manip'
x = df[(condCol)]
hue = df[('cellCode')]
y = df[(params[0])]


sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={"color" : 'grey',  "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})

sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')


condPairs = [manips[1], manips[0]]
box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df_average, box_pairs = box_pairs, param = params[0], cond = condCol_box,\
            test = 'ranksum_greater')


# ax[0].set_title(ax_titles[0], fontsize = plotLabels)
ax.set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)
ax.set_ylabel(ax_titles[0], fontsize = plotTicks)


ax.yaxis.set_tick_params(labelsize=plotTicks)
ax.xaxis.set_tick_params(labelsize=plotTicks)
handles, labels = ax.get_legend_handles_labels()
newLabels = []
for i in labels:
    count1 = df_average[(params[0], 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[0])].values
    count2 = df_average[(params[0], 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[1])].values
    newLabels.append('{}, No.: {}, {}'.format(i, str(count1), str(count2)))

ax.legend(handles, newLabels)

fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
plt.savefig(pathBoxPlots + '/' +params[0]+'_'+ str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
plt.show()

#%%%% Whole-fit 'K', testing intracellular quantities

fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells
measure = 'fit_K_kPa'

df = df[[measure, 'dateID', 'manip', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()


# if mode == 'Compare activation' and condCol == 'manip':
cellCodes = df['cellCode'][df['manip'] == manips[1]].values
df = df[df['cellCode'].isin(cellCodes)]


group_by_cell = df.groupby(['cellID'])
df_average = group_by_cell.agg({measure:['mean', 'count'], 'cellCode':'first', 'manip':'first'})
df_average = df_average[df_average['cellCode'].duplicated(keep = False)]


condCol_box = 'manip', 'first'
x_box = df_average[condCol_box]
hue_box = df_average[('cellCode', 'first')]
y_box = df_average[('fit_K_kPa', 'mean')]


condCol = 'manip'
x = df[condCol]
hue = df[('cellCode')]
y = df[('fit_K_kPa')]


sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={"color" : 'grey',  "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})


sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')


condPairs = [manips[1], manips[0]]
box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df_average, box_pairs = box_pairs, param = measure, cond = condCol_box,\
            test = 'ranksum_less')


ax.set_title(measure, fontsize = plotLabels)
ax.set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)
# ax.set_ylabel(ax_titles[0], fontsize = plotTicks)

# ax[1].set_title(axtitle, fontsize = plotLabels)
# # # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax.yaxis.set_tick_params(labelsize=plotTicks)
ax.xaxis.set_tick_params(labelsize=plotTicks)

# handles, labels = ax.get_legend_handles_labels()
# newLabels = []
# for i in labels:
#     count1 = df_average[(measure, 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[0])].values
#     count2 = df_average[(measure, 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[1])].values
#     newLabels.append('{}, No.: {}, {}'.format(i, str(count1), str(count2)))

# ax.legend(handles, newLabels)

# fig1.suptitle(str(dates) + '_'+str(condPairs))
# plt.tight_layout()
# plt.savefig(pathBoxPlots + '/surroundingThickness_' + str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
plt.show()


#%%%% Surrounding thickness, averaged by cell

fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells

df = df[['surroundingThickness', 'dateID', 'manip', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

# if mode == 'Compare activation' and condCol == 'manip':
cellCodes = df['cellCode'][df['manip'] == manips[1]].values
df = df[df['cellCode'].isin(cellCodes)]


group_by_cell = df.groupby(['cellID'])
df_average = group_by_cell.agg({'surroundingThickness':['mean', 'count'], 'cellCode':'first', 'manip':'first'})
df_average = df_average[df_average['cellCode'].duplicated(keep = False)]


condCol_box = 'manip', 'first'
x_box = df_average[condCol_box]
hue_box = df_average[('cellCode', 'first')]
y_box = df_average[('surroundingThickness', 'mean')]

condCol = 'manip'
x = df[condCol]
hue = df[('cellCode')]
y = df[('surroundingThickness')]

sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={"color" : 'grey',  "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})

sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')

condCol = 'manip', 'first'
condPairs = [manips[0], manips[1]]
box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df_average, box_pairs = box_pairs, param = 'surroundingThickness', cond = condCol,\
            test = 'ranksum_less')


# ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# # # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


# ax[1].set_title(axtitle, fontsize = plotLabels)
# # # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax.yaxis.set_tick_params(labelsize=plotTicks)
ax.xaxis.set_tick_params(labelsize=plotTicks)

handles, labels = ax.get_legend_handles_labels()
newLabels = []
for i in labels:
    count1 = df_average[('surroundingThickness', 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[0])].values
    count2 = df_average[('surroundingThickness', 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[1])].values
    newLabels.append('{}, No.: {}, {}'.format(i, str(count1), str(count2)))

ax.legend(handles, newLabels)

# fig1.suptitle(str(dates) + '_'+str(condPairs))
# plt.tight_layout()
plt.savefig(pathBoxPlots + '/surroundingThickness_' + str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
plt.show()

#%%%% Surrounding thickness

plt.style.use('seaborn')

data = data_main

dates = ['23-03-28']
manips = ['M1', 'M2']


condCol = 'manip'
measure = 'surroundingThickness'
force = '[5mT = 50pN]'
activationTime = []

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            (data['manip'].apply(lambda x : x in manips))
            ]

globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

allCells = data_f['cellID'].unique()
# dateDir = date.replace('-', '.')

# for cell in allCells:
#     try:
#         meta = pd.read_csv(os.path.join(cp.DirDataRaw+'/'+dateDir, cell+'_disc20um_L40_OptoMetadata.txt'), sep = '\t')
#         times = meta['T_abs'] - meta['T_0']
#         activationTime.append(times.values)
#     except:
#         print('No activation data')

fig1, axes = plt.subplots(1,1, figsize=(15,10))

x = (data_f['compNum']-1)*18
ax = sns.lineplot(x = x, y = measure, data = data_f, hue = 'manip')

# for each in activationTime[0]:
#     ax.axvline(x = each, ymax = .05, color = 'blue', lw = 5)

fig1.suptitle(force+'_'+measure+' (nm) vs. Time (secs)', color = fontColour)
plt.xticks(fontsize=30, color = fontColour)
plt.yticks(fontsize=30, color = fontColour)
plt.xlabel('Time (secs)', fontsize = 25, color = fontColour)
plt.ylabel(measure+' (nm)', fontsize = 25, color = fontColour)
plt.legend(fontsize = 10, loc = 'upper right')

# plt.ylim(0,1500)
# ax.get_legend().remove()

plt.savefig(todayFigDir + '/'+str(dates)+'_'+measure+'vsCompr'+str(manips)+'_'+str(condCol)+'.png')

plt.show()


#%% Expriments with blebbi --> 23-03-24

GlobalTable = taka.getMergedTable('Global_MecaData_Dimi_f15_23-03-24')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
fitsSubDir = 'Dimi_f15_23-03-24'

# fitType = 'Log'
fitType = 'stressGaussian'
fitId = '_75' #  = '_75'
fitWidth = 75

#%% Plots for 23-02-02, 22-12-07 and 23-01-23

GlobalTable = taka.getMergedTable('Global_MecaData_Chad_f15_All_23-04-22') #'_tka3_offset15pts')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
fitsSubDir = 'Chad_f15_All_23-04-22'


fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75


#%%%% Non linearity

if not os.path.exists(todayFigDir):
    os.mkdir(todayFigDir)

pathSubDir = todayFigDir+'/'+fitsSubDir
if not os.path.exists(pathSubDir):
    os.mkdir(pathSubDir)


pathNonlinDir = pathSubDir+'/NonLinPlots'
if not os.path.exists(pathNonlinDir):
    os.mkdir(pathNonlinDir)

# plt.style.use('seaborn-v0_8-bright')
plt.style.use('seaborn')


# styleDict1 =  {'M1':{'color': "#5c337f",'marker':'o'},
#                 'M2':{'color': gs.colorList40[23],'marker':'o'},
#                 'M3':{'color': gs.colorList40[20],'marker':'o'},
#                 'M4':{'color': "#008fb1",'marker':'o'},
#                 'M5':{'color': gs.colorList40[30],'marker':'o'},
#                 'M6':{'color': gs.colorList40[33],'marker':'o'},
#                 'M7':{'color': gs.colorList40[31],'marker':'o'},
#                 'M8':{'color': gs.colorList40[32],'marker':'o'},
#                 }

# # Y27 Comparison
# styleDict1 =  {'23-02-02_M1':{'color':"#008fb1",'marker':'o'},
#                 '22-12-07_M1':{'color': "#5c337f",'marker':'o'},
#                 '23-02-02_M3':{'color': "#9452cc",'marker':'o'},
#                 '23-02-02_M7':{'color': "#b967ff",'marker':'o'},
#                 }
# manipIDs = ['22-12-07_M1', '23-02-02_M1', '23-02-02_M3', '23-02-02_M7']


# Blebbistatin Comparison
# styleDict1 =  {'22-12-07_M4':{'color': gs.colorList40[21],'marker':'o'},
#                 '23-03-24_M1':{'color': "#039960",'marker':'o'},
#                 '23-03-23_M3':{'color':"#212222",'marker':'o'},
#                 '23-02-02_M1':{'color': gs.colorList40[23],'marker':'o'},
#                 '23-03-28_M1':{'color': gs.colorList40[24],'marker':'o'},
                # '23-02-02_M3':{'color': gs.colorList40[23],'marker':'o'},
                # '23-02-02_M7':{'color': gs.colorList40[24],'marker':'o'},
                # }

styleDict1 =  {'M1':{'color': "#039960",'marker':'o'},
                'M2':{'color': gs.colorList40[30],'marker':'o'},
                'M3':{'color': "#989a99",'marker':'o'},
                'M4':{'color': gs.colorList40[34],'marker':'o'},
              }

labels = ['10uM Blebbi', 'DMSO']

#LARG Comparison
# styleDict1 =  {'22-12-07_M4':{'color': "#b9008d",'marker':'o'},
#                 '23-02-02_M1':{'color': "#008db9",'marker':'o'},
#                 '23-03-28_M1':{'color':"#8db900",'marker':'o'},
#                 }
# manipIDs = ['22-12-07_M4', '23-03-28_M1']
# labels = ['3T3 optoPRG', '3T3 optoLARG']

# Control Comparison
# styleDict1 =  {'22-12-07_M4':{'color': "#b9008d",'marker':'o'},
#                 '23-03-24_M3':{'color': "#039960",'marker':'o'},
#                 '23-02-02_M1':{'color':"#008db9",'marker':'o'},
#                 # '23-03-28_M1':{'color': gs.colorList40[24],'marker':'o'},
#                 }

# manipIDs = ['23-02-02_M1', '22-12-07_M4'] #, '23-02-02_M7']

data = data_main

dates = ['23-03-24']
manips = ['M1', 'M3'] #, 'M3'] 
# manipIDs = ['23-03-28_M1',  '23-02-02_M1']

# labels = ['Control', '50uM Y27']
# labels = ['50uM Y27', 'Control', '10uM Y27', '1uM Y27']
# 
stressRange = '200_500'

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

# allSelectedCells = np.asarray(['23-03-28_M1_P1_C12', '23-03-28_M2_P1_C12', '23-03-28_M3_P1_C12'
#                                '23-03-28_M1_P2_C5', '23-03-28_M2_P2_C5', '23-03-28_M3_P2_C5'])


Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1900),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

# selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')


out1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir,  legendLabels = labels, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = 'manip', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

plt.legend(fontsize = 20, loc = 'upper left')


mainFig2, mainAx2 = plt.subplots(1,1)
mainFig2.patch.set_facecolor('black')


out2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = 'manip', mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2


# atbeads = mpatches.Patch(color=flatui[0], label='Activation at beads')
# awaybeads = mpatches.Patch(color=flatui[1], label='Activation away from beads')
# control = mpatches.Patch(color=flatui[2], label='Control')

plt.legend(fontsize = 15, loc = 'upper left')

plt.ylim(0,10)
plt.tight_layout()
# plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'250-600.png')  

plt.show()

#%%%% Plotting all K vs Stress, per compression with fits on K
plt.style.use('seaborn')

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
data_main['cellId'] = GlobalTable['cellID']

fitsSubDir = fitsSubDir

fitType = 'stressGaussian'
fitId = '_75'
Sinf, Ssup = 150, 450
FIT_MODE = 'loglog'  # 'linlin', 'loglog'

data = data_main 

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


interceptStress = 150
condCol = 'manip'

dates = ['22-12-07']
manips = ['M4', 'M5']
manipIDs = ['22-12-07_M1', '23-02-02_M1', '23-02-02_M3', '23-02-02_M7']
# manipIDs = ['22-12-07_M4', '23-03-28_M1']
# manipIDs = ['23-02-02_M1', '22-12-07_M4']

# order = ['22-12-07_M1', '23-02-02_M3', '23-02-02_M7', '23-02-02_M1']

plot = False
mode = None
if mode == 'Compare activation' and condCol == 'manip':
    print(gs.ORANGE + 'Considering values after 3rd compression for activate cells' + gs.NORMAL)
    selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
    data = data.drop(selRows, axis = 0)
    


for date in dates:
    if date == '22-12-07' and condCol == 'manip':
        oldManip = ['M7', 'M8', 'M9']
        for i in oldManip:
            data['manip'][(data['date'] == date) & (data['manip'] == i)] = 'M5'

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

# allSelectedCells = ['23_03-28_M1']

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            # (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['cellId'].apply(lambda x : x in cellIDs)),
            # (data['manip'].apply(lambda x : x in manips)),
            (data['manipId'].apply(lambda x : x in manipIDs)),
            ]


dfAllCells = plot2Params(data, Filters, fitsSubDir, fitType, interceptStress, FIT_MODE, pathFits, plot = plot)

plt.close('all')

#%%%% Plot box plots
fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']
    
if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs

# df = df[df['R2Fit'] > 0.70]
df = df[[params[0], params[1], condCol, 'chosenIntercept', 'chosenInterceptStress', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()


# cellCodes = df['cellCode'][df['manip'] == manips[1]].values
# df = df[df['cellCode'].isin(cellCodes)]

plotIntercept = True

if plotIntercept:
    y1, y2 = df[params[0]],  df['chosenIntercept']
    figExt = 'Intercept'+str(interceptStress)
    axtitle = 'Intercept at '+str(interceptStress)+'Pa'
else:
    y1, y2 = df[params[0]],  df[params[1]]
    figExt = 'Coeff'
    axtitle = ax_titles[1]

x = df[condCol]

sns.boxplot(x = x, y = y1, data=dfAllCells, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    
sns.swarmplot(x = x, y = y1, data=dfAllCells, order = order,linewidth = 1, ax = ax, edgecolor='k') #, hue = 'cellCode')

# sns.boxplot(x = x, y = y2, data=dfAllCells, ax = ax[1], order = order,
#                     medianprops={"color": 'darkred', "linewidth": 2},\
#                     boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    

# sns.swarmplot(x = x, y = y2, hue = 'cellCode', data=dfAllCells,  order = order, linewidth = 1, ax = ax[1], edgecolor='k') #, hue = 'cellCode')

box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df, box_pairs = box_pairs, param = params[0], cond = condCol)
# addStat_df(ax = ax[1], data = df, box_pairs = box_pairs, param = params[1], cond = condCol)


# ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


# ax[1].set_title(axtitle, fontsize = plotLabels)
# # ax[1].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


# ax[0].yaxis.set_tick_params(labelsize=plotTicks)
# ax[1].yaxis.set_tick_params(labelsize=plotTicks)
# ax[0].xaxis.set_tick_params(labelsize=plotTicks)
# ax[1].xaxis.set_tick_params(labelsize=plotTicks)


# ax.get_legend().remove()
# ax[0].get_legend().remove()

# fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
# plt.savefig(pathBoxPlots + '/' + str(dates) + '_'+str(condPairs) + '_' + figExt+'.png')  
plt.show()

# %%%% Plotting surrounding thickness / best H0 - box plots

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
    
condCol = 'manipId'
if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs

# df = df[df['R2Fit'] > 0.70]
df = df[['surroundingThickness', condCol, 'chosenIntercept', 'chosenInterceptStress', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

# if mode == 'Compare activation' and condCol == 'manip':
#     cellCodes = df['cellCode'][df['manip'] == manips[1]].values
#     df = df[df['cellCode'].isin(cellCodes)]


x = df[condCol]
y = 'surroundingThickness'
order = ['22-12-07_M4','22-12-07_M1', '23-02-23_M1', '23-03-28_M1']
my_pal_y27 = l = {'22-12-07_M1':"#6f3d99", '23-02-02_M3':"#b967ff", '23-02-02_M7':"#d5a3ff", '23-02-02_M1':"#008fb1"}
# my_pal_blebbi = {'M1':"#039960", 'M3':"#989a99"}
# my_pal_larg = {'22-12-07_M4': "#b9008d", '23-02-02_M1':"#008db9", '23-03-28_M1':"#8db900"}
# order = None


sns.boxplot(x = x, y = y, data=df, ax = ax, order = order,
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.9})
    
sns.swarmplot(x = x, y = y, data=df, order = order,linewidth = 1, s = 7, ax = ax, color = 'k', edgecolor='k') #, hue = 'cellCode')

# sns.pointplot(x = x, y = y, data=df, order = order, hue = 'cellID', ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')


# box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
# addStat_df(ax = ax, data = df, box_pairs = box_pairs, param = params[0], cond = condCol)
# # addStat_df(ax = ax[1], data = df, box_pairs = box_pairs, param = params[1], cond = condCol)


# ax[0].set_title(ax_titles[0], fontsize = plotLabels)
ax.set_xticklabels(('optoPRG Control', 'optoPRG Control + 50uM Y27', '3T3 ATCC', 'optoLARG'), fontsize = plotTicks)
# ax.set_xticklabels(('3T3 optoPRG', '3T3 optoLARG'), fontsize = plotTicks)


# ax[1].set_title(axtitle, fontsize = plotLabels)
# # ax[1].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)

color = "#ffffff"
ax.yaxis.set_tick_params(labelsize=plotTicks, color = color)
ax.xaxis.set_tick_params(labelsize=plotTicks, color = color)
ax.set_ylabel('Surrounding Thickness (nm)', fontsize = 20,  color = color)
ax.set_xlabel(' ', fontsize = 20, color = color)
ax.tick_params(axis='both', colors= color) 
ax.set_ylim(0, 1000)

# ax.get_legend().remove()
# ax[0].get_legend().remove()

# fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
# plt.savefig(pathBoxPlots + '/' + str(dates) + '_'+str(condPairs) + '_' + figExt+'.png')  
plt.show()



#%%%% 2Param plots, testing intracellular quantities

fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']
    
if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs
    
condCol = 'manip'
df = df[[params[0], params[1], condCol, 'chosenIntercept', 'chosenInterceptStress', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

# cellCodes = df['cellCode'][df['manip'] == manips[1]].values
# df = df[df['cellCode'].isin(cellCodes)]

plotIntercept = True

if plotIntercept:
    y1, y2 = df[params[0]],  df['chosenIntercept']
    figExt = 'Intercept'+str(interceptStress)
    axtitle = 'Intercept at '+str(interceptStress)+'Pa'
else:
    y1, y2 = df[params[0]],  df[params[1]]
    figExt = 'Coeff'
    axtitle = ax_titles[1]

group_by_cell = df.groupby(['cellID'])
df_average = group_by_cell.agg({params[0]:['var', 'std', 'mean', 'count'], 'cellCode':'first', 'manip':'first'})
df_average = df_average[df_average['cellCode'].duplicated(keep = False)]

condCol_box = 'manip', 'first'
x_box = df_average[(condCol_box)]
hue_box = df_average[('cellCode', 'first')]
y_box = df_average[(params[0], 'mean')]

condCol = 'manip'
x = df[(condCol)]
hue = df[('cellCode')]
y = df[(params[0])]


sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={"color" : 'grey',  "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})

sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')


condPairs = [manips[1], manips[0]]
box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df_average, box_pairs = box_pairs, param = params[0], cond = condCol_box,\
            test = 'ranksum_greater')


# ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# ax.set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)
# ax.set_ylabel(ax_titles[0], fontsize = plotTicks)


ax.yaxis.set_tick_params(labelsize=plotTicks)
ax.xaxis.set_tick_params(labelsize=plotTicks)
handles, labels = ax.get_legend_handles_labels()
newLabels = []
for i in labels:
    count1 = df_average[(params[0], 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[0])].values
    count2 = df_average[(params[0], 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[1])].values
    newLabels.append('{}, No.: {}, {}'.format(i, str(count1), str(count2)))

ax.legend(handles, newLabels)

fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
plt.savefig(pathBoxPlots + '/' +params[0]+'_'+ str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
plt.show()

#%%%% Whole-fit 'K', testing intracellular quantities

fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells
measure = 'fit_K_kPa'

df = df[[measure, 'dateID', 'manip', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()


# if mode == 'Compare activation' and condCol == 'manip':
# cellCodes = df['cellCode'][df['manip'] == manips[1]].values
# df = df[df['cellCode'].isin(cellCodes)]


group_by_cell = df.groupby(['cellID'])
df_average = group_by_cell.agg({measure:['mean', 'count'], 'cellCode':'first', 'manip':'first'})
df_average = df_average[df_average['cellCode'].duplicated(keep = False)]


condCol_box = 'manip', 'first'
x_box = df_average[condCol_box]
hue_box = df_average[('cellCode', 'first')]
y_box = df_average[('fit_K_kPa', 'mean')]


condCol = 'manip'
x = df[condCol]
hue = df[('cellCode')]
y = df[('fit_K_kPa')]


sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={"color" : 'grey',  "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})


sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')


condPairs = [manips[1], manips[0]]
box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df_average, box_pairs = box_pairs, param = measure, cond = condCol_box,\
            test = 'ranksum_less')


ax.set_title(measure, fontsize = plotLabels)
# ax.set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)
# ax.set_ylabel(ax_titles[0], fontsize = plotTicks)

# ax[1].set_title(axtitle, fontsize = plotLabels)
# # # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax.yaxis.set_tick_params(labelsize=plotTicks)
ax.xaxis.set_tick_params(labelsize=plotTicks)

# handles, labels = ax.get_legend_handles_labels()
# newLabels = []
# for i in labels:
#     count1 = df_average[(measure, 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[0])].values
#     count2 = df_average[(measure, 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[1])].values
#     newLabels.append('{}, No.: {}, {}'.format(i, str(count1), str(count2)))

# ax.legend(handles, newLabels)

# fig1.suptitle(str(dates) + '_'+str(condPairs))
# plt.tight_layout()
# plt.savefig(pathBoxPlots + '/surroundingThickness_' + str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
plt.show()


#%%%% Surrounding thickness, averaged by cell

fig1, ax = plt.subplots(1, 1, figsize = (15,10))
fig1.patch.set_facecolor('black')
plt.style.use('seaborn')

df = dfAllCells

df = df[['surroundingThickness', 'dateID', 'manip', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

# if mode == 'Compare activation' and condCol == 'manip':
cellCodes = df['cellCode'][df['manip'] == manips[1]].values
df = df[df['cellCode'].isin(cellCodes)]


group_by_cell = df.groupby(['cellID'])
df_average = group_by_cell.agg({'surroundingThickness':['mean', 'count'], 'cellCode':'first', 'manip':'first'})
df_average = df_average[df_average['cellCode'].duplicated(keep = False)]


condCol_box = 'manip', 'first'
x_box = df_average[condCol_box]
hue_box = df_average[('cellCode', 'first')]
y_box = df_average[('surroundingThickness', 'mean')]

condCol = 'manip'
x = df[condCol]
hue = df[('cellCode')]
y = df[('surroundingThickness')]


sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={"color" : 'grey',  "edgecolor": '#ffffff',"linewidth": 2, 'alpha' : 1})

sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='se') #, hue = 'cellCode')

condCol = 'manip', 'first'
condPairs = [manips[0], manips[1]]
box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df_average, box_pairs = box_pairs, param = 'surroundingThickness', cond = condCol,\
            test = 'ranksum_less')


# ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# # # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


# ax[1].set_title(axtitle, fontsize = plotLabels)
ax.set_xticklabels(('No activation', 'Polarised rear'), fontsize = plotTicks, color = color)
ax.yaxis.set_tick_params(labelsize=plotTicks, color = color)
ax.set_ylabel('Surrounding Thickness (nm)', fontsize = 25, color = color)
ax.xaxis.set_tick_params(labelsize=plotTicks, color = color)
ax.tick_params(axis='both', colors= color) 

handles, labels = ax.get_legend_handles_labels()
newLabels = []
for i in labels:
    count1 = df_average[('surroundingThickness', 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[0])].values
    count2 = df_average[('surroundingThickness', 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[1])].values
    newLabels.append('{}, No.: {}, {}'.format(i, str(count1), str(count2)))

ax.legend(handles, newLabels)

# fig1.suptitle(str(dates) + '_'+str(condPairs))
# plt.tight_layout()
# plt.savefig(pathBoxPlots + '/surroundingThickness_' + str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
plt.show()

#%%%% Surrounding thickness

plt.style.use('seaborn')

data = data_main

dates = ['23-03-28']
manips = ['M1', 'M2']


condCol = 'manip'
measure = 'surroundingThickness'
force = '[5mT = 50pN]'
activationTime = []

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            (data['manip'].apply(lambda x : x in manips))
            ]

globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

allCells = data_f['cellID'].unique()
# dateDir = date.replace('-', '.')

# for cell in allCells:
#     try:
#         meta = pd.read_csv(os.path.join(cp.DirDataRaw+'/'+dateDir, cell+'_disc20um_L40_OptoMetadata.txt'), sep = '\t')
#         times = meta['T_abs'] - meta['T_0']
#         activationTime.append(times.values)
#     except:
#         print('No activation data')

fig1, axes = plt.subplots(1,1, figsize=(15,10))

x = (data_f['compNum']-1)*18
ax = sns.lineplot(x = x, y = measure, data = data_f, hue = 'manip')

# for each in activationTime[0]:
#     ax.axvline(x = each, ymax = .05, color = 'blue', lw = 5)

fig1.suptitle(force+'_'+measure+' (nm) vs. Time (secs)', color = fontColour)
plt.xticks(fontsize=30, color = fontColour)
plt.yticks(fontsize=30, color = fontColour)
plt.xlabel('Time (secs)', fontsize = 25, color = fontColour)
plt.ylabel(measure+' (nm)', fontsize = 25, color = fontColour)
plt.legend(fontsize = 10, loc = 'upper right')

# plt.ylim(0,1500)
# ax.get_legend().remove()

plt.savefig(todayFigDir + '/'+str(dates)+'_'+measure+'vsCompr'+str(manips)+'_'+str(condCol)+'.png')

plt.show()

#%% Plots for All

GlobalTable = taka.getMergedTable('Global_MecaData_Dimi_f15_All_23-04-11') #'_tka3_offset15pts')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
fitsSubDir = 'Dimi_f15_All_23-04-11'

fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75


#%%%% Non linearity

if not os.path.exists(todayFigDir):
    os.mkdir(todayFigDir)

pathSubDir = todayFigDir+'/'+fitsSubDir
if not os.path.exists(pathSubDir):
    os.mkdir(pathSubDir)


pathNonlinDir = pathSubDir+'/NonLinPlots'
if not os.path.exists(pathNonlinDir):
    os.mkdir(pathNonlinDir)

plt.style.use('seaborn')

styleDict1 =  {'M1':{'color': gs.colorList40[20],'marker':'o'},
               'M2':{'color': gs.colorList40[23],'marker':'o'},
               'M3':{'color': gs.colorList40[21],'marker':'o'},
               'M4':{'color': gs.colorList40[22],'marker':'o'},
               'M5':{'color': gs.colorList40[30],'marker':'o'},
                'M6':{'color': gs.colorList40[33],'marker':'o'},
                'M7':{'color': gs.colorList40[31],'marker':'o'},
                'M8':{'color': gs.colorList40[32],'marker':'o'},
                }

styleDict1 =  {'22-12-07_M1':{'color': gs.colorList40[21],'marker':'o'},
               '22-12-07_M4':{'color': gs.colorList40[20],'marker':'o'},
               '23-02-02_M1':{'color': gs.colorList40[30],'marker':'o'},
               '23-02-02_M3':{'color': gs.colorList40[21],'marker':'o'},
               '23-03-24_M1':{'color': gs.colorList40[31],'marker':'o'},
               '23-03-24_M3':{'color': gs.colorList40[15],'marker':'o'},
                }

data = data_main

dates = ['22-12-07']
manips = ['M4'] # 'M3'] 
# manipIDs = ['23-02-02_M3','23-02-02_M1', '23-03-24_M1',  '23-03-24_M3']
manipIDs = ['23-02-02_M3','23-02-02_M1', '23-03-24_M1',  '23-03-24_M3']


stressRange = '200_550'


Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1900),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

# selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)

out1, cellDf1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = 'manipId', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

mainFig2, mainAx2 = plt.subplots(1,1)


out2, cellDf2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = 'manipId', mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2


# atbeads = mpatches.Patch(color=flatui[0], label='Activation at beads')
# awaybeads = mpatches.Patch(color=flatui[1], label='Activation away from beads')
# control = mpatches.Patch(color=flatui[2], label='Control')

# plt.legend(handles=[awaybeads, control], fontsize = 20, loc = 'upper left')


plt.tight_layout()
plt.ylim(0,10)
# plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'150-450.png')  

plt.show()

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

#%%%% Plotting all K vs Stress, per compression with fits on K
plt.style.use('seaborn')

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
data_main['cellId'] = GlobalTable['cellID']

fitsSubDir = fitsSubDir

fitType = 'stressGaussian'
fitId = '_75'
Sinf, Ssup = 150, 450
FIT_MODE = 'loglog'  # 'linlin', 'loglog'

data = data_main 

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


interceptStress = 150
condCol = 'manip'
dates = ['23-03-24']

manipIDs = ['22-12-07_M1', '23-02-02_M3','23-02-02_M1', '23-03-24_M1',  '23-03-24_M3']

manips = ['M1', 'M3']
order = None

plot = False

mode = None
if mode == 'Compare activation' and condCol == 'manip':
    print(gs.ORANGE + 'Considering values after 3rd compression for activate cells' + gs.NORMAL)
    selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
    data = data.drop(selRows, axis = 0)
    


for date in dates:
    if date == '22-12-07' and condCol == 'manip':
        oldManip = ['M7', 'M8', 'M9']
        for i in oldManip:
            data['manip'][(data['date'] == date) & (data['manip'] == i)] = 'M5'

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

# allSelectedCells = ['23_03-28_M1']

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            # (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['cellId'].apply(lambda x : x in cellIDs)),
            # (data['manip'].apply(lambda x : x in manips)),
            (data['manipId'].apply(lambda x : x in manipIDs)),
            ]


dfAllCells = plot2Params(data, Filters, fitsSubDir, fitType, interceptStress, FIT_MODE, pathFits, plot = plot)

plt.close('all')

#%%%% Plot box plots
fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']
    
if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs

# df = df[df['R2Fit'] > 0.70]
df = df[[params[0], params[1], condCol, 'chosenIntercept', 'chosenInterceptStress', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()


# cellCodes = df['cellCode'][df['manip'] == manips[1]].values
# df = df[df['cellCode'].isin(cellCodes)]

plotIntercept = True

if plotIntercept:
    y1, y2 = df[params[0]],  df['chosenIntercept']
    figExt = 'Intercept'+str(interceptStress)
    axtitle = 'Intercept at '+str(interceptStress)+'Pa'
else:
    y1, y2 = df[params[0]],  df[params[1]]
    figExt = 'Coeff'
    axtitle = ax_titles[1]

x = df[condCol]

sns.boxplot(x = x, y = y1, data=dfAllCells, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    
sns.swarmplot(x = x, y = y1, data=dfAllCells, order = order,linewidth = 1, ax = ax, edgecolor='k') #, hue = 'cellCode')

# sns.boxplot(x = x, y = y2, data=dfAllCells, ax = ax[1], order = order,
#                     medianprops={"color": 'darkred', "linewidth": 2},\
#                     boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    

# sns.swarmplot(x = x, y = y2, hue = 'cellCode', data=dfAllCells,  order = order, linewidth = 1, ax = ax[1], edgecolor='k') #, hue = 'cellCode')

box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df, box_pairs = box_pairs, param = params[0], cond = condCol)
# addStat_df(ax = ax[1], data = df, box_pairs = box_pairs, param = params[1], cond = condCol)


# ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


# ax[1].set_title(axtitle, fontsize = plotLabels)
# # ax[1].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


# ax[0].yaxis.set_tick_params(labelsize=plotTicks)
# ax[1].yaxis.set_tick_params(labelsize=plotTicks)
# ax[0].xaxis.set_tick_params(labelsize=plotTicks)
# ax[1].xaxis.set_tick_params(labelsize=plotTicks)


# ax.get_legend().remove()
# ax[0].get_legend().remove()

# fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
# plt.savefig(pathBoxPlots + '/' + str(dates) + '_'+str(condPairs) + '_' + figExt+'.png')  
plt.show()

# %%%% Plotting surrounding thickness / best H0 - box plots

fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']
condCol = 'manipId'
if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs

# df = df[df['R2Fit'] > 0.70]
df = df[['surroundingThickness', condCol, 'chosenIntercept', 'chosenInterceptStress', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

if mode == 'Compare activation' and condCol == 'manip':
    cellCodes = df['cellCode'][df['manip'] == manips[1]].values
    df = df[df['cellCode'].isin(cellCodes)]


x = df[condCol]
y = 'surroundingThickness'
order = ['22-12-07_M1', '22-12-07_M4', '23-02-23_M1', '23-03-28_M1']

sns.boxplot(x = x, y = y, data=df, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.4})
    
sns.swarmplot(x = x, y = y, data=df, order = order,linewidth = 1, ax = ax, edgecolor='k') #, hue = 'cellCode')

# sns.pointplot(x = x, y = y, data=df, order = order, hue = 'cellID', ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')


# box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
# addStat_df(ax = ax, data = df, box_pairs = box_pairs, param = params[0], cond = condCol)
# addStat_df(ax = ax[1], data = df, box_pairs = box_pairs, param = params[1], cond = condCol)


# ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# ax.set_xticklabels(('22-12-07_optoPDZ', '23-02-02_optoPDZ', '23-03-28_optoLARG'), fontsize = plotTicks)


# ax[1].set_title(axtitle, fontsize = plotLabels)
# # ax[1].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax.yaxis.set_tick_params(labelsize=plotTicks)
ax.xaxis.set_tick_params(labelsize=plotTicks)
# ax[0].xaxis.set_tick_params(labelsize=plotTicks)
# ax[1].xaxis.set_tick_params(labelsize=plotTicks)


# ax.get_legend().remove()
# ax[0].get_legend().remove()

# fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
# plt.savefig(pathBoxPlots + '/' + str(dates) + '_'+str(condPairs) + '_' + figExt+'.png')  
plt.show()



#%%%% 2Param plots, testing intracellular quantities

fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']
    
if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs
    
condCol = 'manip'
df = df[[params[0], params[1], condCol, 'chosenIntercept', 'chosenInterceptStress', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

# cellCodes = df['cellCode'][df['manip'] == manips[1]].values
# df = df[df['cellCode'].isin(cellCodes)]

plotIntercept = True

if plotIntercept:
    y1, y2 = df[params[0]],  df['chosenIntercept']
    figExt = 'Intercept'+str(interceptStress)
    axtitle = 'Intercept at '+str(interceptStress)+'Pa'
else:
    y1, y2 = df[params[0]],  df[params[1]]
    figExt = 'Coeff'
    axtitle = ax_titles[1]

group_by_cell = df.groupby(['cellID'])
df_average = group_by_cell.agg({params[0]:['var', 'std', 'mean', 'count'], 'cellCode':'first', 'manip':'first'})
df_average = df_average[df_average['cellCode'].duplicated(keep = False)]

condCol_box = 'manip', 'first'
x_box = df_average[(condCol_box)]
hue_box = df_average[('cellCode', 'first')]
y_box = df_average[(params[0], 'mean')]

condCol = 'manip'
x = df[(condCol)]
hue = df[('cellCode')]
y = df[(params[0])]


sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={"color" : 'grey',  "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})

sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')


condPairs = [manips[1], manips[0]]
box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df_average, box_pairs = box_pairs, param = params[0], cond = condCol_box,\
            test = 'ranksum_greater')


# ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# ax.set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)
# ax.set_ylabel(ax_titles[0], fontsize = plotTicks)


ax.yaxis.set_tick_params(labelsize=plotTicks)
ax.xaxis.set_tick_params(labelsize=plotTicks)
handles, labels = ax.get_legend_handles_labels()
newLabels = []
for i in labels:
    count1 = df_average[(params[0], 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[0])].values
    count2 = df_average[(params[0], 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[1])].values
    newLabels.append('{}, No.: {}, {}'.format(i, str(count1), str(count2)))

ax.legend(handles, newLabels)

fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
plt.savefig(pathBoxPlots + '/' +params[0]+'_'+ str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
plt.show()

#%%%% Whole-fit 'K', testing intracellular quantities

fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells
measure = 'fit_K_kPa'

df = df[[measure, 'dateID', 'manip', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()


# if mode == 'Compare activation' and condCol == 'manip':
# cellCodes = df['cellCode'][df['manip'] == manips[1]].values
# df = df[df['cellCode'].isin(cellCodes)]


group_by_cell = df.groupby(['cellID'])
df_average = group_by_cell.agg({measure:['mean', 'count'], 'cellCode':'first', 'manip':'first'})
df_average = df_average[df_average['cellCode'].duplicated(keep = False)]


condCol_box = 'manip', 'first'
x_box = df_average[condCol_box]
hue_box = df_average[('cellCode', 'first')]
y_box = df_average[('fit_K_kPa', 'mean')]


condCol = 'manip'
x = df[condCol]
hue = df[('cellCode')]
y = df[('fit_K_kPa')]


sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={"color" : 'grey',  "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})


sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')


condPairs = [manips[1], manips[0]]
box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df_average, box_pairs = box_pairs, param = measure, cond = condCol_box,\
            test = 'ranksum_less')


ax.set_title(measure, fontsize = plotLabels)
# ax.set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)
# ax.set_ylabel(ax_titles[0], fontsize = plotTicks)

# ax[1].set_title(axtitle, fontsize = plotLabels)
# # # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax.yaxis.set_tick_params(labelsize=plotTicks)
ax.xaxis.set_tick_params(labelsize=plotTicks)

# handles, labels = ax.get_legend_handles_labels()
# newLabels = []
# for i in labels:
#     count1 = df_average[(measure, 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[0])].values
#     count2 = df_average[(measure, 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[1])].values
#     newLabels.append('{}, No.: {}, {}'.format(i, str(count1), str(count2)))

# ax.legend(handles, newLabels)

# fig1.suptitle(str(dates) + '_'+str(condPairs))
# plt.tight_layout()
# plt.savefig(pathBoxPlots + '/surroundingThickness_' + str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
plt.show()


#%%%% Surrounding thickness, averaged by cell

fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells

df = df[['surroundingThickness', 'dateID', 'manip', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

# if mode == 'Compare activation' and condCol == 'manip':
cellCodes = df['cellCode'][df['manip'] == manips[1]].values
df = df[df['cellCode'].isin(cellCodes)]


group_by_cell = df.groupby(['cellID'])
df_average = group_by_cell.agg({'surroundingThickness':['mean', 'count'], 'cellCode':'first', 'manip':'first'})
df_average = df_average[df_average['cellCode'].duplicated(keep = False)]


condCol_box = 'manip', 'first'
x_box = df_average[condCol_box]
hue_box = df_average[('cellCode', 'first')]
y_box = df_average[('surroundingThickness', 'mean')]

condCol = 'manip'
x = df[condCol]
hue = df[('cellCode')]
y = df[('surroundingThickness')]

sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={"color" : 'grey',  "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})

sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')

condCol = 'manip', 'first'
condPairs = [manips[0], manips[1]]
box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df_average, box_pairs = box_pairs, param = 'surroundingThickness', cond = condCol,\
            test = 'ranksum_less')


# ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# # # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


# ax[1].set_title(axtitle, fontsize = plotLabels)
# # # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax.yaxis.set_tick_params(labelsize=plotTicks)
ax.xaxis.set_tick_params(labelsize=plotTicks)

handles, labels = ax.get_legend_handles_labels()
newLabels = []
for i in labels:
    count1 = df_average[('surroundingThickness', 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[0])].values
    count2 = df_average[('surroundingThickness', 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[1])].values
    newLabels.append('{}, No.: {}, {}'.format(i, str(count1), str(count2)))

ax.legend(handles, newLabels)

# fig1.suptitle(str(dates) + '_'+str(condPairs))
# plt.tight_layout()
plt.savefig(pathBoxPlots + '/surroundingThickness_' + str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
plt.show()

#%%%% Surrounding thickness

plt.style.use('seaborn')

data = data_main

dates = ['23-03-28']
manips = ['M1', 'M2']


condCol = 'manip'
measure = 'bestH0'
force = '[15mT = 500pN]'
activationTime = []

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            (data['manip'].apply(lambda x : x in manips))
            ]

globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

allCells = data_f['cellID'].unique()
# dateDir = date.replace('-', '.')

# for cell in allCells:
#     try:
#         meta = pd.read_csv(os.path.join(cp.DirDataRaw+'/'+dateDir, cell+'_disc20um_L40_OptoMetadata.txt'), sep = '\t')
#         times = meta['T_abs'] - meta['T_0']
#         activationTime.append(times.values)
#     except:
#         print('No activation data')

fig1, axes = plt.subplots(1,1, figsize=(15,10))

x = (data_f['compNum']-1)*18
ax = sns.lineplot(x = x, y = measure, data = data_f, hue = 'manip')

# for each in activationTime[0]:
#     ax.axvline(x = each, ymax = .05, color = 'blue', lw = 5)

fig1.suptitle(force+'_'+measure+' (nm) vs. Time (secs)', color = fontColour)
plt.xticks(fontsize=30, color = fontColour)
plt.yticks(fontsize=30, color = fontColour)
plt.xlabel('Time (secs)', fontsize = 25, color = fontColour)
plt.ylabel(measure+' (nm)', fontsize = 25, color = fontColour)
plt.legend(fontsize = 10, loc = 'upper right')

# plt.ylim(0,1500)
# ax.get_legend().remove()

plt.savefig(todayFigDir + '/'+str(dates)+'_'+measure+'vsCompr'+str(manips)+'_'+str(condCol)+'.png')

plt.show()

#%% Plots for 23-04-19

GlobalTable = taka.getMergedTable('Global_MecaData_Chad_f15_23-04-19_23-04-21')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
fitsSubDir = 'Chad_f15_23-04-19_23-04-21'

fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75

#%%%% Non linearity

if not os.path.exists(todayFigDir):
    os.mkdir(todayFigDir)

pathSubDir = todayFigDir+'/'+fitsSubDir
if not os.path.exists(pathSubDir):
    os.mkdir(pathSubDir)


pathNonlinDir = pathSubDir+'/NonLinPlots'
if not os.path.exists(pathNonlinDir):
    os.mkdir(pathNonlinDir)

plt.style.use('seaborn')

styleDict1 =  {'M1':{'color': gs.colorList40[20],'marker':'o'},
               'M2':{'color': gs.colorList40[23],'marker':'o'},
               'M3':{'color': gs.colorList40[21],'marker':'o'},
               'M4':{'color': gs.colorList40[22],'marker':'o'},
               'M5':{'color': gs.colorList40[30],'marker':'o'},
                'M6':{'color': gs.colorList40[33],'marker':'o'},
                'M7':{'color': gs.colorList40[31],'marker':'o'},
                'M8':{'color': gs.colorList40[32],'marker':'o'},
                }

data = data_main

excludedCells = ['23-04-19_M1_P2_C2', '23-04-19_M3_P2_C2',
                  '23-04-19_M1_P2_C3','23-04-19_M3_P2_C3']
    
for i in excludedCells:
    data = data[data['cellID'].str.contains(i) == False]

dates = ['23-04-19']
manips = ['M1', 'M3'] # 'M3'] 
# manipIDs = ['23-02-02_M3','23-02-02_M1', '23-03-24_M1',  '23-03-24_M3']

stressRange = '150_450'

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

# allSelectedCells = np.asarray(['23-03-28_M1_P1_C12', '23-03-28_M2_P1_C12', '23-03-28_M3_P1_C12'
#                                '23-03-28_M1_P2_C5', '23-03-28_M2_P2_C5', '23-03-28_M3_P2_C5'])


Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1900),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

# selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)

out1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = 'manip', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

mainFig2, mainAx2 = plt.subplots(1,1)


out2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = 'manip', mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2


# atbeads = mpatches.Patch(color=flatui[0], label='Activation at beads')
# awaybeads = mpatches.Patch(color=flatui[1], label='Activation away from beads')
# control = mpatches.Patch(color=flatui[2], label='Control')

# plt.legend(handles=[awaybeads, control], fontsize = 20, loc = 'upper left')


plt.tight_layout()
# plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'150-450.png')  

plt.show()

#%%%% Plotting all K vs Stress, per compression with fits on K
plt.style.use('seaborn')

data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
data_main['cellId'] = GlobalTable['cellID']

fitsSubDir = fitsSubDir

fitType = 'stressGaussian'
fitId = '_75'
Sinf, Ssup = 150, 450
FIT_MODE = 'loglog'  # 'linlin', 'loglog'

data = data_main 
# excludedCells = ['23-04-19_M1_P2_C2', '23-04-19_M3_P2_C2',
#                   '23-04-19_M1_P2_C3','23-04-19_M3_P2_C3']
    
# for i in excludedCells:
#     data = data[data['cellID'].str.contains(i) == False]

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


interceptStress = 150
condCol = 'manip'
dates = ['23-04-19']


# manips = ['M1', 'M2']
manipIDs = ['23-02-23_M1', '22-12-07_M1', '22-12-07_M4', '23-03-28_M1']
order = None

plot = False

mode = None
if mode == 'Compare activation' and condCol == 'manip':
    print(gs.ORANGE + 'Considering values after 3rd compression for activate cells' + gs.NORMAL)
    selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
    data = data.drop(selRows, axis = 0)
    
# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

# allSelectedCells = ['23_03-28_M1']

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            # (data['UI_Valid'] == True),
            (data['bestH0'] <= 1900),
            # (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['cellId'].apply(lambda x : x in cellIDs)),
            # (data['manip'].apply(lambda x : x in manips)),
            (data['manipId'].apply(lambda x : x in manipIDs)),
            ]


dfAllCells = plot2Params(data, Filters, fitsSubDir, fitType, interceptStress, FIT_MODE, pathFits, plot = plot)

plt.close('all')


#%%%% Whole fit-K
fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells
measure = 'fit_K_kPa'

df = df[[measure, 'dateID', 'manip', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()


# if mode == 'Compare activation' and condCol == 'manip':
# cellCodes = df['cellCode'][df['manip'] == manips[1]].values
# df = df[df['cellCode'].isin(cellCodes)]


group_by_cell = df.groupby(['cellID'])
df_average = group_by_cell.agg({measure:['mean', 'count'], 'cellCode':'first', 'manip':'first'})
df_average = df_average[df_average['cellCode'].duplicated(keep = False)]


condCol_box = 'manip', 'first'
x_box = df_average[condCol_box]
hue_box = df_average[('cellCode', 'first')]
y_box = df_average[('fit_K_kPa', 'mean')]


condCol = 'manip'
x = df[condCol]
hue = df[('cellCode')]
y = df[('fit_K_kPa')]


sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={"color" : 'grey',  "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})


sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')


condPairs = [manips[1], manips[0]]
box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df_average, box_pairs = box_pairs, param = measure, cond = condCol_box,\
            test = 'Wilcox_less')


ax.set_title(measure, fontsize = plotLabels)
# ax.set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)
# ax.set_ylabel(ax_titles[0], fontsize = plotTicks)

# ax[1].set_title(axtitle, fontsize = plotLabels)
# # # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax.yaxis.set_tick_params(labelsize=plotTicks)
ax.xaxis.set_tick_params(labelsize=plotTicks)

# handles, labels = ax.get_legend_handles_labels()
# newLabels = []
# for i in labels:
#     count1 = df_average[(measure, 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[0])].values
#     count2 = df_average[(measure, 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[1])].values
#     newLabels.append('{}, No.: {}, {}'.format(i, str(count1), str(count2)))

# ax.legend(handles, newLabels)

# fig1.suptitle(str(dates) + '_'+str(condPairs))
# plt.tight_layout()
# plt.savefig(pathBoxPlots + '/surroundingThickness_' + str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
plt.show()


#%%%% Surrounding thickness, averaged by cell

fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells

df = df[['surroundingThickness', 'dateID', 'manip', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

# if mode == 'Compare activation' and condCol == 'manip':
cellCodes = df['cellCode'][df['manip'] == manips[1]].values
df = df[df['cellCode'].isin(cellCodes)]


group_by_cell = df.groupby(['cellID'])
df_average = group_by_cell.agg({'surroundingThickness':['mean', 'count'], 'cellCode':'first', 'manip':'first'})
df_average = df_average[df_average['cellCode'].duplicated(keep = False)]


condCol_box = 'manip', 'first'
x_box = df_average[condCol_box]
hue_box = df_average[('cellCode', 'first')]
y_box = df_average[('surroundingThickness', 'mean')]

condCol = 'manip'
x = df[condCol]
hue = df[('cellCode')]
y = df[('surroundingThickness')]

sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={"color" : 'grey',  "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})

sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')

condCol = 'manip', 'first'
condPairs = [manips[0], manips[1]]
box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df_average, box_pairs = box_pairs, param = 'surroundingThickness', cond = condCol,\
            test = 'ranksum_less')


# ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# # # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


# ax[1].set_title(axtitle, fontsize = plotLabels)
# # # ax[0].set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)


ax.yaxis.set_tick_params(labelsize=plotTicks)
ax.xaxis.set_tick_params(labelsize=plotTicks)

handles, labels = ax.get_legend_handles_labels()
newLabels = []
for i in labels:
    count1 = df_average[('surroundingThickness', 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[0])].values
    count2 = df_average[('surroundingThickness', 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[1])].values
    newLabels.append('{}, No.: {}, {}'.format(i, str(count1), str(count2)))

ax.legend(handles, newLabels)

# fig1.suptitle(str(dates) + '_'+str(condPairs))
# plt.tight_layout()
# plt.savefig(pathBoxPlots + '/surroundingThickness_' + str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
plt.show()

#%%%% Surrounding thickness

plt.style.use('seaborn')

data = data_main

dates = ['23-04-19']
manips = ['M1', 'M3']


condCol = 'manip'
measure = 'surroundingThickness'
force = '[15mT = 500pN]'
activationTime = []

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            (data['UI_Valid'] == True),
            (data['bestH0'] <= 1500),
            (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            (data['manip'].apply(lambda x : x in manips))
            ]

globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
for k in range(0, len(Filters)):
    globalFilter = globalFilter & Filters[k]
data_f = data[globalFilter]

allCells = data_f['cellID'].unique()
# dateDir = date.replace('-', '.')

# for cell in allCells:
#     try:
#         meta = pd.read_csv(os.path.join(cp.DirDataRaw+'/'+dateDir, cell+'_disc20um_L40_OptoMetadata.txt'), sep = '\t')
#         times = meta['T_abs'] - meta['T_0']
#         activationTime.append(times.values)
#     except:
#         print('No activation data')

fig1, axes = plt.subplots(1,1, figsize=(15,10))

x = (data_f['compNum']-1)*18
ax = sns.lineplot(x = x, y = measure, data = data_f, hue = 'manip')

# for each in activationTime[0]:
#     ax.axvline(x = each, ymax = .05, color = 'blue', lw = 5)

fig1.suptitle(force+'_'+measure+' (nm) vs. Time (secs)', color = fontColour)
plt.xticks(fontsize=30, color = fontColour)
plt.yticks(fontsize=30, color = fontColour)
plt.xlabel('Time (secs)', fontsize = 25, color = fontColour)
plt.ylabel(measure+' (nm)', fontsize = 25, color = fontColour)
plt.legend(fontsize = 10, loc = 'upper right')

# plt.ylim(0,1500)
# ax.get_legend().remove()

# plt.savefig(todayFigDir + '/'+str(dates)+'_'+measure+'vsCompr'+str(manips)+'_'+str(condCol)+'.png')

plt.show()

#%%%% 2Param plots, testing intracellular quantities

fig1, ax = plt.subplots(1, 1, figsize = (15,10))
df = dfAllCells

if FIT_MODE == 'linlin':
    params = ['A', 'B']
    ax_titles = ['Slope (A)', 'Linear Intercept (B)']
elif FIT_MODE == 'loglog':
    params = ['a', 'q']
    ax_titles = ['Exponent (a)', 'Coefficient (q)']
    
if condCol == 'manipId':
    condPairs = manipIDs
elif condCol == 'manip':
    condPairs = manips
elif condCol == 'cellId':
    condPairs = cellIDs
    
condCol = 'manip'
df = df[[params[0], params[1], condCol, 'chosenIntercept', 'chosenInterceptStress', 'R2Fit', 'cellCode', 'cellID', 'compNum']]

df = df.drop_duplicates()
df = df.dropna()

# cellCodes = df['cellCode'][df['manip'] == manips[1]].values
# df = df[df['cellCode'].isin(cellCodes)]

plotIntercept = True

if plotIntercept:
    y1, y2 = df[params[0]],  df['chosenIntercept']
    figExt = 'Intercept'+str(interceptStress)
    axtitle = 'Intercept at '+str(interceptStress)+'Pa'
else:
    y1, y2 = df[params[0]],  df[params[1]]
    figExt = 'Coeff'
    axtitle = ax_titles[1]

group_by_cell = df.groupby(['cellID'])
df_average = group_by_cell.agg({params[0]:['var', 'std', 'mean', 'count'], 'cellCode':'first', 'manip':'first'})
df_average = df_average[df_average['cellCode'].duplicated(keep = False)]

condCol_box = 'manip', 'first'
x_box = df_average[(condCol_box)]
hue_box = df_average[('cellCode', 'first')]
y_box = df_average[(params[0], 'mean')]

condCol = 'manip'
x = df[(condCol)]
hue = df[('cellCode')]
y = df[(params[0])]


sns.boxplot(x = x_box, y = y_box, data=df_average, ax = ax, order = order, 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={"color" : 'grey',  "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.2})

sns.pointplot(x = x, y = y, data=df, order = order, hue = hue, ax = ax, dodge = True, errorbar='sd') #, hue = 'cellCode')


condPairs = [manips[1], manips[0]]
box_pairs = [(a, b) for idx, a in enumerate(condPairs) for b in condPairs[idx + 1:]]
addStat_df(ax = ax, data = df_average, box_pairs = box_pairs, param = params[0], cond = condCol_box,\
            test = 'ranksum_greater')


# ax[0].set_title(ax_titles[0], fontsize = plotLabels)
# ax.set_xticklabels(('Control', 'Activation at beads'), fontsize = plotTicks)
# ax.set_ylabel(ax_titles[0], fontsize = plotTicks)


ax.yaxis.set_tick_params(labelsize=plotTicks)
ax.xaxis.set_tick_params(labelsize=plotTicks)
handles, labels = ax.get_legend_handles_labels()
newLabels = []
for i in labels:
    count1 = df_average[(params[0], 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[0])].values
    count2 = df_average[(params[0], 'count')][(df_average['cellCode', 'first'] == i) & (df_average['manip', 'first'] == manips[1])].values
    newLabels.append('{}, No.: {}, {}'.format(i, str(count1), str(count2)))

ax.legend(handles, newLabels)

fig1.suptitle(str(dates) + '_'+str(condPairs))
plt.tight_layout()
plt.savefig(pathBoxPlots + '/' +params[0]+'_'+ str(dates) + '_'+str(condPairs) + '_' + figExt+'_averaged.png')  
plt.show()


#%% Plots for 3T3 ATCC

GlobalTable = taka.getMergedTable('GlobalTable_MergeATCC_OptoPRG.csv', mergeUMS = False) #'_tka3_offset15pts')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']

fitsSubDir = 'JV_AJ_MergedData_ATCC-optoPRG'


fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75


#%%%% Non linearity

if not os.path.exists(todayFigDir):
    os.mkdir(todayFigDir)

pathSubDir = todayFigDir+'/'+fitsSubDir
if not os.path.exists(pathSubDir):
    os.mkdir(pathSubDir)


pathNonlinDir = pathSubDir+'/NonLinPlots'
if not os.path.exists(pathNonlinDir):
    os.mkdir(pathNonlinDir)

# plt.style.use('seaborn-v0_8-bright')
# plt.style.use('seaborn')


# styleDict1 =  {'M1':{'color': "#5c337f",'marker':'o'},
#                 'M2':{'color': gs.colorList40[23],'marker':'o'},
#                 'M3':{'color': gs.colorList40[20],'marker':'o'},
#                 'M4':{'color': "#008fb1",'marker':'o'},
#                 'M5':{'color': gs.colorList40[30],'marker':'o'},
#                 'M6':{'color': gs.colorList40[33],'marker':'o'},
#                 'M7':{'color': gs.colorList40[31],'marker':'o'},
#                 'M8':{'color': gs.colorList40[32],'marker':'o'},
#                 }

# # Y27 Comparison
# styleDict1 =  {'23-02-02_M1':{'color':"#008fb1",'marker':'o'},
#                 '22-12-07_M1':{'color': "#5c337f",'marker':'o'},
#                 '23-02-02_M3':{'color': "#9452cc",'marker':'o'},
#                 '23-02-02_M7':{'color': "#b967ff",'marker':'o'},
#                 }
# manipIDs = ['22-12-07_M1', '23-02-02_M1', '23-02-02_M3', '23-02-02_M7']


# Blebbistatin Comparison
# styleDict1 =  {'22-12-07_M4':{'color': gs.colorList40[21],'marker':'o'},
#                 '23-03-24_M1':{'color': "#039960",'marker':'o'},
#                 '23-03-23_M3':{'color':"#212222",'marker':'o'},
#                 '23-02-02_M1':{'color': gs.colorList40[23],'marker':'o'},
#                 '23-03-28_M1':{'color': gs.colorList40[24],'marker':'o'},
                # '23-02-02_M3':{'color': gs.colorList40[23],'marker':'o'},
                # '23-02-02_M7':{'color': gs.colorList40[24],'marker':'o'},
                # }

# styleDict1 =  {'M1':{'color': "#039960",'marker':'o'},
#                 'M2':{'color': gs.colorList40[30],'marker':'o'},
#                 'M3':{'color': "#989a99",'marker':'o'},
#                 'M4':{'color': gs.colorList40[34],'marker':'o'},
#               }

# styleDict1 =  {'M1':{'color': "#000000",'marker':'o'},
               
#               }

# labels = ['10uM Blebbi', 'DMSO']

#LARG Comparison
# styleDict1 =  {'22-12-07_M4':{'color': "#b9008d",'marker':'o'},
#                 '23-02-02_M1':{'color': "#008db9",'marker':'o'},
#                 '23-03-28_M1':{'color':"#8db900",'marker':'o'},
#                 }
# manipIDs = ['22-12-07_M4', '23-03-28_M1']
# labels = ['3T3 optoPRG', '3T3 optoLARG']

# Control Comparison
# styleDict1 =  {'22-12-07_M4':{'color': "#b9008d",'marker':'o'},
#                 '23-03-24_M3':{'color': "#039960",'marker':'o'},
#                 '23-02-02_M1':{'color':"#008db9",'marker':'o'},
#                 # '23-03-28_M1':{'color': gs.colorList40[24],'marker':'o'},
#                 }

# manipIDs = ['23-02-02_M1', '22-12-07_M4'] #, '23-02-02_M7']


#3T3 ATCC Comparison

styleDict1 =  {'22-12-07_M1':{'color':"#5c337f",'marker':'o'},
                '22-12-07_M4':{'color': "#008fb1",'marker':'o'},
                '23-02-23_M1':{'color': "#000000",'marker':'o'},
                
                }

manipIDs = ['22-12-07_M1', '22-12-07_M4', '23-02-23_M1']
labels = ['3T3 optoPRG + 50uM Y27', '3T3 optoPRG Control', '3T3 ATCC']

# manipIDs = ['22-12-07_M4', '23-02-23_M1']
# labels = ['3T3 optoPRG Control', '3T3 ATCC']


data = data_main

dates = ['23-02-23']
# manips = ['M1'] #, 'M3'] 
# manipIDs = ['23-03-28_M1',  '23-02-02_M1']


# labels = ['Control', '50uM Y27']
# labels = ['50uM Y27', 'Control', '10uM Y27', '1uM Y27']
# 
stressRange = '200_500'

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

# allSelectedCells = np.asarray(['23-03-28_M1_P1_C12', '23-03-28_M2_P1_C12', '23-03-28_M3_P1_C12'
#                                '23-03-28_M1_P2_C5', '23-03-28_M2_P2_C5', '23-03-28_M3_P2_C5'])


Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            # (data['UI_Valid'] == True),
            (data['bestH0'] <= 1900),
            # (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manip'].apply(lambda x : x in manips)),
            (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

# selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')


out1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir,  legendLabels = labels, fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = 'manipId', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

plt.legend(fontsize = 20, loc = 'upper left')


mainFig2, mainAx2 = plt.subplots(1,1)
mainFig2.patch.set_facecolor('black')


out2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = 'manipId', mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2


# atbeads = mpatches.Patch(color=flatui[0], label='Activation at beads')
# awaybeads = mpatches.Patch(color=flatui[1], label='Activation away from beads')
# control = mpatches.Patch(color=flatui[2], label='Control')

plt.legend(fontsize = 15, loc = 'upper left')

plt.ylim(0,10)
plt.tight_layout()
# plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'250-600.png')  

plt.show()

#%% Plots for 23-05-10 and 23-04-25

GlobalTable = taka.getMergedTable('Chad_f15_All_23-05-10_23-05-16', mergeUMS = False) #'_tka3_offset15pts')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']

fitsSubDir = 'Chad_f15_All_23-05-10_23-05-16'


fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75


#%%%% Non linearity

if not os.path.exists(todayFigDir):
    os.mkdir(todayFigDir)

pathSubDir = todayFigDir+'/'+fitsSubDir
if not os.path.exists(pathSubDir):
    os.mkdir(pathSubDir)


pathNonlinDir = pathSubDir+'/NonLinPlots'
if not os.path.exists(pathNonlinDir):
    os.mkdir(pathNonlinDir)

# plt.style.use('seaborn-v0_8-bright')
# plt.style.use('seaborn')


# styleDict1 =  {'M1':{'color': "#5c337f",'marker':'o'},
#                 'M2':{'color': gs.colorList40[23],'marker':'o'},
#                 'M3':{'color': gs.colorList40[20],'marker':'o'},
#                 'M4':{'color': "#008fb1",'marker':'o'},
#                 }

styleDict1 =  {'23-05-10_M3':{'color': "#5c337f",'marker':'o', 'label' : 'global'},
                '23-04-25_M1':{'color': gs.colorList40[23],'marker':'o', 'label' : 'global 2'},
                }



data = data_main

# dates = ['23-05-10']
# manips = ['M3'] #, 'M3'] 
manipIDs = ['23-04-25_M1',  '23-05-10_M3']


labels = []
# labels = ['50uM Y27', 'Control', '10uM Y27', '1uM Y27']

stressRange = '200_500'

# control = manips[0]
# active = manips[1]
# intersected = list(set(data['cellID'][data['manip'] == active]) & set(cellCats))
# controlCells = ['{}_{}_P{}_C{}'.format(dates[0], control, ufun.findInfosInFileName(x, 'P'), ufun.findInfosInFileName(x, 'C'))\
#               for x in intersected]
    
# allSelectedCells = np.asarray(intersected + controlCells)

# allSelectedCells = np.asarray(['23-03-28_M1_P1_C12', '23-03-28_M2_P1_C12', '23-03-28_M3_P1_C12'
#                                '23-03-28_M1_P2_C5', '23-03-28_M2_P2_C5', '23-03-28_M3_P2_C5'])


Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            # (data['UI_Valid'] == True),
            (data['bestH0'] <= 1900),
            # (data['date'].apply(lambda x : x in dates)),
            # (data['cellID'].apply(lambda x : x in allSelectedCells)),
            # (data['manip'].apply(lambda x : x in manips)),
            (data['manipId'].apply(lambda x : x in manipIDs)),
            ]

# selRows = data[(data['manip'] == manips[1]) & (data['compNum'] < 3)].index
# data = data.drop(selRows, axis = 0)

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')


out1, cellDf1 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir,  fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                   condCol = 'manipId', mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1

plt.legend(fontsize = 20, loc = 'upper left')


mainFig2, mainAx2 = plt.subplots(1,1)
mainFig2.patch.set_facecolor('black')


out2, cellDf2 = plotPopKS(data, mainFig2, mainAx2, fitsSubDir = fitsSubDir,  fitType =  'stressGaussian', 
                  fitWidth=75, Filters = Filters, condCol = 'manipId', mode = stressRange, scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2


# atbeads = mpatches.Patch(color=flatui[0], label='Activation at beads')
# awaybeads = mpatches.Patch(color=flatui[1], label='Activation away from beads')
# control = mpatches.Patch(color=flatui[2], label='Control')

plt.legend(fontsize = 15, loc = 'upper left')

plt.ylim(0,8)
plt.tight_layout()
# plt.savefig(pathNonlinDir + '/' + str(dates) + '_'+str(manips) + '_' + mode+'250-600.png')  

plt.show()

#%%%% Individual K vs. S plots to check averaging

data = data_main

dates = ['23-05-10']
manips = ['M3']

labels = []

stressRange = '200_550'
cells = data_ff['cellID'].unique()

mainFig1, mainAx1 = plt.subplots(1,1)
mainFig1.patch.set_facecolor('black')

styleDict1 = {}

for i in range(len(cells)):
    cell = cells[i]
    
    styleDict1[cell] = {'color': gs.colorList40[10+i] ,'marker':'o'}

Filters = [(data['validatedThickness'] == True),
            # (data['substrate'] == '20um fibronectin discs'), 
            # (data['drug'] == 'none'), 
            (data['bead type'] == 'M450'),
            # (data['UI_Valid'] == True),
            (data['bestH0'] <= 1900),
            (data['date'].apply(lambda x : x in dates)),
            (data['cellID'].apply(lambda x : x in cells)),
            (data['manip'].apply(lambda x : x in manips)),
            # (data['manipId'].apply(lambda x : x in manipIDs)),
            ] 
    
out2, cellDf2 = plotPopKS(data, mainFig1, mainAx1, fitsSubDir = fitsSubDir, legendLabels = labels, fitType =  'stressGaussian', 
              fitWidth=75, Filters = Filters, condCol = 'cellID', mode = stressRange, scale = 'lin', printText = False,
                            returnData = 1, returnCount = 1)


#%% Plots for 23-11-21 and 23-10-29

GlobalTable = taka.getMergedTable('Chad_f15_LIMKi3_23-11-21_23-10-29', mergeUMS = False) #'_tka3_offset15pts')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']

fitsSubDir = 'Chad_f15_LIMKi3_23-11-21_23-10-29'


fitType = 'stressGaussian'
# fitType = 'nPoints'
fitId = '_75'
fitWidth = 75

#%%%%
data = data_main


# rD = {'none & 0.0' : 'No drug',
#       'Y27 & 10.0' : 'Y27\n(10µM)', 
#       'Y27 & 50.0' : 'Y27\n(50µM)', 
#       'dmso & 0.0' : 'DMSO', 
#       'blebbistatin & 10.0' : 'Blebbi\n(10µM)',  
#       'blebbistatin & 50.0' : 'Blebbi\n(50µM)', 
#       'blebbistatin & 250.0' : 'Blebbi\n(250µM)', 
#       'ck666 & 50.0' : 'CK666\n(50µM)', 
#       'ck666 & 100.0' : 'CK666\n(100µM)', 
#       'Thickness at low force (nm)' : 'Thickness (nm)',
#       'E_f_<_400_kPa' : 'Elastic modulus (Pa)\nfor F < 400pN',
#       'Tangeantial Modulus (Pa)' : 'Tangeantial Modulus (kPa)\n' + fitStr}


method = 'f_<_400'
stiffnessType = 'E_' + method

data[stiffnessType + '_kPa'] = data[stiffnessType] / 1000

Filters = [(data['validatedThickness'] == True), 
           (data['valid_' + method] == True),
            (data[stiffnessType + '_kPa'] <= 20),
            (data['date'].apply(lambda x : x in dates)),
            (data['manipID'].apply(lambda x : x not in ['23-04-20_M1'])),]

# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 250.0']

# box_pairs = [('none & 0.0', 'Y27 & 10.0'), ('none & 0.0', 'Y27 & 50.0'), ('Y27 & 10.0', 'Y27 & 50.0')]


fig2, ax2 = D1Plot(data, condCols=['drug', 'concentration'], Parameters=[stiffnessType + '_kPa'], Filters=Filters, 
                Boxplot=True, cellID='cellID', co_order=co_order, 
                AvgPerCell = True, stats=True, statMethod='Mann-Whitney', box_pairs=[], 
                figSizeFactor = 1.0, markersizeFactor=1.2, orientation = 'h', stressBoxPlot=2)


renameAxes(ax2, renameDict1)
renameAxes(ax2, rD)

plt.show()