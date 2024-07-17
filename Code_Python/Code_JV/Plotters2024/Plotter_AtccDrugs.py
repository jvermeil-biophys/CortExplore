# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:21:04 2024

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
import matplotlib.patches as mpatches

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
from scipy.stats import mannwhitneyu
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

#### Pandasµ
pd.set_option('display.max_columns', None)
# pd.reset_option('display.max_columns')
pd.set_option('display.max_rows', None)
pd.reset_option('display.max_rows')


####  Matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=gs.colorList40) 

#### Graphic options
gs.set_default_options_jv()

figDir = "C://Users//JosephVermeil//Desktop//Joseph_Presentations//Drugs_SystematicSummary//SummaryFigs"



# %% > Objects declaration

renameDict = {# Variables
               'SurroundingThickness': 'Median Thickness (nm)',
               'surroundingThickness': 'Median Thickness (nm)',
               'ctFieldThickness': 'Median Thickness (nm)',
               'ctFieldFluctuAmpli' : 'Thickness Fluctuations; $D_9$-$D_1$ (nm)',
               'EChadwick': 'E Chadwick (Pa)',
               'medianThickness': 'Median Thickness (nm)',               
               'fluctuAmpli': 'Fluctuations Amplitude (nm)',               
               'meanFluoPeakAmplitude' : 'Fluo Intensity (a.u.)', 
               'fit_K' : 'Tangeantial Modulus (Pa)',
               'bestH0' : 'Fitted $H_0$ (nm)',
               'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
               'E_f_<_400_kPa' : 'Elastic modulus (kPa)\nfor F < 400pN',
               # Drugs
               'none':'Control',
               'dmso':'DMSO',
               'blebbistatin':'Blebbi',
               'latrunculinA':'LatA',
               'Y27':'Y27',
               }

styleDict =  {# Drugs
               'none':{'color': gs.colorList40[10],'marker':'o'},
               'none & 0.0':{'color': gs.colorList40[10],'marker':'o'},
               #
               'dmso':{'color': gs.colorList40[19],'marker':'o'},
               'dmso & 0.0':{'color': gs.colorList40[19],'marker':'o'},
               #
               'blebbistatin':{'color': gs.colorList40[22],'marker':'o'},
               'blebbistatin & 10.0':{'color': gs.colorList40[12],'marker':'o'},
               'blebbistatin & 50.0':{'color': gs.colorList40[26],'marker':'o'},
               'blebbistatin & 250.0':{'color': gs.colorList40[32],'marker':'o'},
               #
               'PNB & 50.0':{'color': gs.colorList40[25],'marker':'o'},
               'PNB & 250.0':{'color': gs.colorList40[35],'marker':'o'},
               #
               'latrunculinA':{'color': gs.colorList40[23],'marker':'o'},
               'latrunculinA & 0.1':{'color': gs.colorList40[13],'marker':'o'},
               'latrunculinA & 0.5':{'color': gs.colorList40[20],'marker':'o'},
               'latrunculinA & 2.5':{'color': gs.colorList40[33],'marker':'o'},
               #
               'calyculinA':{'color': gs.colorList40[23],'marker':'o'},
               'calyculinA & 0.25':{'color': gs.colorList40[15],'marker':'o'},
               'calyculinA & 0.5':{'color': gs.colorList40[22],'marker':'o'},
               'calyculinA & 1.0':{'color': gs.colorList40[30],'marker':'o'},
               'calyculinA & 2.0':{'color': gs.colorList40[33],'marker':'o'},
               #
               'Y27':{'color': gs.colorList40[17],'marker':'o'},
               'Y27 & 1.0':{'color': gs.colorList40[7],'marker':'o'},
               'Y27 & 10.0':{'color': gs.colorList40[15],'marker':'o'},
               'Y27 & 50.0':{'color': gs.colorList40[27],'marker':'o'},
               'Y27 & 100.0':{'color': gs.colorList40[37],'marker':'o'},
               #
               'LIMKi':{'color': gs.colorList40[22],'marker':'o'},
               'LIMKi & 10.0':{'color': gs.colorList40[22],'marker':'o'},
               'LIMKi & 20.0':{'color': gs.colorList40[31],'marker':'o'},
               #
               'JLY':{'color': gs.colorList40[23],'marker':'o'},
               'JLY & 8-5-10':{'color': gs.colorList40[23],'marker':'o'},
               #
               'ck666':{'color': gs.colorList40[25],'marker':'o'},
               'ck666 & 50.0':{'color': gs.colorList40[15],'marker':'o'},
               'ck666 & 100.0':{'color': gs.colorList40[38],'marker':'o'},
               
               # Cell types
               '3T3':{'color': gs.colorList40[30],'marker':'o'},
               'HoxB8-Macro':{'color': gs.colorList40[32],'marker':'o'},
               'DC':{'color': gs.colorList40[33],'marker':'o'},
               # Cell subtypes
               'aSFL':{'color': gs.colorList40[12],'marker':'o'},
               'Atcc-2023':{'color': gs.colorList40[10],'marker':'o'},
               'optoRhoA':{'color': gs.colorList40[13],'marker':'o'},
               
               # Drugs + cell types
               'aSFL-LG+++ & dmso':{'color': gs.colorList40[19],'marker':'o'},
               'aSFL-LG+++ & blebbistatin':{'color': gs.colorList40[32],'marker':'o'},
               'Atcc-2023 & dmso':{'color': gs.colorList40[19],'marker':'o'},
               'Atcc-2023 & blebbistatin':{'color': gs.colorList40[22],'marker':'o'},
               #
               'Atcc-2023 & none':{'color': gs.colorList40[0],'marker':'o'},
               'Atcc-2023 & none & 0.0':{'color': gs.colorList40[0],'marker':'o'},
               'Atcc-2023 & Y27': {'color': gs.colorList40[17],'marker':'o'},
               'Atcc-2023 & Y27 & 10.0': {'color': gs.colorList40[17],'marker':'o'},
               #
               'optoRhoA & none':{'color': gs.colorList40[13],'marker':'o'},
               'optoRhoA & none & 0.0':{'color': gs.colorList40[13],'marker':'o'},
               'optoRhoA & Y27': {'color': gs.colorList40[27],'marker':'o'},
               'optoRhoA & Y27 & 10.0': {'color': gs.colorList40[27],'marker':'o'},
               #
               'Atcc-2023 & dmso':{'color': gs.colorList40[9],'marker':'o'},
               'Atcc-2023 & dmso & 0.0':{'color': gs.colorList40[9],'marker':'o'},
               'Atcc-2023 & blebbistatin':{'color': gs.colorList40[12],'marker':'o'},
               'Atcc-2023 & blebbistatin & 10.0':{'color': gs.colorList40[2],'marker':'o'},
               'Atcc-2023 & blebbistatin & 50.0':{'color': gs.colorList40[12],'marker':'o'},
               #
               'optoRhoA & dmso':{'color': gs.colorList40[29],'marker':'o'},
               'optoRhoA & dmso & 0.0':{'color': gs.colorList40[29],'marker':'o'},
               'optoRhoA & blebbistatin':{'color': gs.colorList40[32],'marker':'o'},
               'optoRhoA & blebbistatin & 10.0':{'color': gs.colorList40[22],'marker':'o'},
               'optoRhoA & blebbistatin & 50.0':{'color': gs.colorList40[32],'marker':'o'},
               }



# %% > Data subfunctions

def filterDf(df, F):
    F = np.array(F)
    totalF = np.all(F, axis = 0)
    df_f = df[totalF]
    return(df_f)

def makeBoxPairs(O):
    return(list(itertools.combinations(O, 2)))

def makeCompositeCol(df, cols=[]):
    N = len(cols)
    if N > 1:
        newColName = ''
        for i in range(N):
            newColName += cols[i]
            newColName += ' & '
        newColName = newColName[:-3]
        df[newColName] = ''
        for i in range(N):
            df[newColName] += df[cols[i]].astype(str)
            df[newColName] = df[newColName].apply(lambda x : x + ' & ')
        df[newColName] = df[newColName].apply(lambda x : x[:-3])
    return(df, newColName)

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


def dataGroup_weightedAverage(df, groupCol = 'cellID', idCols = [], 
                              valCol = '', weightCol = '', weight_method = 'ciw^2'):   
    idCols = ['date', 'cellName', 'cellID', 'manipID'] + idCols
    
    wAvgCol = valCol + '_wAvg'
    wVarCol = valCol + '_wVar'
    wStdCol = valCol + '_wStd'
    wSteCol = valCol + '_wSte'
    
    # 1. Compute the weights if necessary
    if weight_method == 'ciw^1':
        ciwCol = weightCol
        weightCol = valCol + '_weight'
        df[weightCol] = (df[valCol]/df[ciwCol])
    elif weight_method == 'ciw^2':
        ciwCol = weightCol
        weightCol = valCol + '_weight'
        df[weightCol] = (df[valCol]/df[ciwCol])**2
    
    df = df.dropna(subset = [weightCol])
    
    # 2. Group and average
    groupColVals = df[groupCol].unique()
    
    d_agg = {k:'first' for k in idCols}

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
    df['A'] = df[valCol] * df[weightCol]
    grouped1 = df.groupby(by=[groupCol])
    d_agg.update({'A': ['count', 'sum'], weightCol: 'sum'})
    data_agg = grouped1.agg(d_agg).reset_index()
    data_agg.columns = ufun.flattenPandasIndex(data_agg.columns)
    data_agg[wAvgCol] = data_agg['A_sum']/data_agg[weightCol + '_sum']
    data_agg = data_agg.rename(columns = {'A_count' : 'count_wAvg'})
    
    # Compute the weighted std
    df['B'] = df[valCol]
    for co in groupColVals:
        weighted_avg_val = data_agg.loc[(data_agg[groupCol] == co), wAvgCol].values[0]
        index_loc = (df[groupCol] == co)
        col_loc = 'B'
        
        df.loc[index_loc, col_loc] = df.loc[index_loc, valCol] - weighted_avg_val
        df.loc[index_loc, col_loc] = df.loc[index_loc, col_loc] ** 2
            
    df['C'] = df['B'] * df[weightCol]
    grouped2 = df.groupby(by=[groupCol])
    data_agg2 = grouped2.agg({'C': 'sum', weightCol: 'sum'}).reset_index()
    data_agg2[wVarCol] = data_agg2['C']/data_agg2[weightCol]
    data_agg2[wStdCol] = data_agg2[wVarCol]**0.5
    
    # Combine all in data_agg
    data_agg[wVarCol] = data_agg2[wVarCol]
    data_agg[wStdCol] = data_agg2[wStdCol]
    data_agg[wSteCol] = data_agg[wStdCol] / data_agg['count_wAvg']**0.5
    
    # data_agg = data_agg.drop(columns = ['A_sum', weightCol + '_sum'])
    data_agg = data_agg.drop(columns = ['A_sum'])
    
    data_agg = data_agg.drop(columns = [groupCol + '_first'])
    data_agg = data_agg.rename(columns = {k+'_first':k for k in idCols})

    return(data_agg)


def makeCountDf(df, condition):   
    cols_count_df = ['compNum', 'cellID', 'manipID', 'date', condition]
    count_df = df[cols_count_df]
    groupByCell = count_df.groupby('cellID')
    d_agg = {'compNum':'count', condition:'first', 'date':'first', 'manipID':'first'}
    df_CountByCell = groupByCell.agg(d_agg).rename(columns={'compNum':'compCount'})

    groupByCond = df_CountByCell.reset_index().groupby(condition)
    d_agg = {'cellID': 'count', 'compCount': 'sum', 
             'date': pd.Series.nunique, 'manipID': pd.Series.nunique}
    d_rename = {'cellID':'cellCount', 'date':'datesCount', 'manipID':'manipsCount'}
    df_CountByCond = groupByCond.agg(d_agg).rename(columns=d_rename)
    
    return(df_CountByCond, df_CountByCell)
    

# %% > Graphic subfunctions

def getSnsPalette(conditions, styleDict):
    colors = []
    try:
        for co in conditions:
            coStyle = styleDict[co]
            if 'color' in coStyle.keys():
                colors.append(coStyle['color'])
            else:
                colors.append('')
        palette = sns.color_palette(colors)
    except:
        palette = sns.color_palette(gs.colorList10)
    return(palette)

def getStyleLists(conditions, styleDict):
    colors = []
    markers = []
    try:
        for co in conditions:
            coStyle = styleDict[co]
            colors.append(coStyle['color'])
            markers.append(coStyle['marker'])
    except:
        N = len(conditions)
        colors = gs.colorList10
        markers = ['o'] * N
        
    return(colors, markers)

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
                
def renameLegend(axes, rD, loc='best'):
    axes = ufun.toList(axes)
    N = len(axes)
    for i in range(N):
        ax = axes[i]
        L = ax.legend(loc = loc)
        Ltext = L.get_texts()
        M = len(Ltext)
        for j in range(M):
            T = Ltext[j].get_text()
            for s in rD.keys():
                if re.search(s, T):
                    Ltext[j].set_text(re.sub(s, rD[s], T))
                    Ltext[j].set_fontsize(8)
                
def addStat_lib(ax, box_pairs, test = 'Mann-Whitney', verbose = False, **plotting_parameters):
    #### STATS
    listTests = ['t-test_ind', 't-test_welch', 't-test_paired', 
                 'Mann-Whitney', 'Mann-Whitney-gt', 'Mann-Whitney-ls', 
                 'Levene', 'Wilcoxon', 'Kruskal', 'Brunner-Munzel']
    if test in listTests:
        annotator = Annotator(ax, box_pairs, **plotting_parameters)
        annotator.configure(test=test, verbose=verbose).apply_and_annotate() # , loc = 'outside'
    else:
        print(gs.BRIGHTORANGE + 'Dear Madam, dear Sir, i am the eternal god and i command that you define this stat test cause it is not in the list !' + gs.NORMAL)
    return(ax)


# %% > Plot Functions

def D1Plot(data, fig = None, ax = None, condition='', parameter='',
           co_order=[], boxplot=1, figSizeFactor = 1, markersizeFactor = 1,
           stats=True, statMethod='Mann-Whitney', box_pairs=[], statVerbose = False,
           showMean = False):
    
    #### Init
    co_values = data[condition].unique()
    Nco = len(co_values)
    if len(co_order) == 0:
        co_order = np.sort(co_values)
        
    if ax == None:
        figHeight = 5
        figWidth = 5*Nco*figSizeFactor
        fig, ax = plt.subplots(1,1, figsize=(figWidth, figHeight))
    markersize = 5 * markersizeFactor
        
    palette = getSnsPalette(co_order, styleDict)
    
    #### Swarmplot
    swarmplot_parameters = {'data':    data,
                            'x':       condition,
                            'y':       parameter,
                            'order':   co_order,
                            'palette': palette,
                            'size'    : markersize, 
                            'edgecolor'    : 'k', 
                            'linewidth'    : 0.75*markersizeFactor
                            }
    
    
    sns.swarmplot(ax=ax, **swarmplot_parameters)

    #### Stats    
    if stats:
        if len(box_pairs) == 0:
            box_pairs = makeBoxPairs(co_order)
        addStat_lib(ax, box_pairs, test = statMethod, verbose = statVerbose, **swarmplot_parameters)

    
    #### Boxplot
    if boxplot>0:
        boxplot_parameters = {'data':    data,
                                'x':       condition,
                                'y':       parameter,
                                'order':   co_order,
                                'width' : 0.5,
                                'showfliers': False,
                                }
        if boxplot==1:
            boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},
                                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                                    # boxprops={"color": color, "linewidth": 0.5},
                                    whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                                    capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2})
        
        elif boxplot==2:
            boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
                                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                                    # boxprops={"color": color, "linewidth": 0.5},
                                    whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                                    capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2})
        
        
        elif boxplot == 3:
            boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 4},
                                boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                                # boxprops={"color": color, "linewidth": 0.5},
                                whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                                capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4})

            
        if showMean:
            boxplot_parameters.update(meanline='True', showmeans='True',
                                      meanprops={"color": 'darkblue', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},)
            
        sns.boxplot(ax=ax, **boxplot_parameters)
        
    return(fig, ax)


def D2Plot_wFit(data, fig = None, ax = None, 
                XCol='', YCol='', condition='', co_order = [],
                modelFit=False, modelType='y=ax+b', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1):
    
    #### Init
    co_values = data[condition].unique()
    print(co_values)
    Nco = len(co_values)
    if len(co_order) == 0:
        co_order = np.sort(co_values)
        
    if ax == None:
        figHeight = 5
        figWidth = 5*Nco*figSizeFactor
        fig, ax = plt.subplots(1,1, figsize=(figWidth, figHeight))
    markersize = 5 * markersizeFactor
        
    colors, markers = getStyleLists(co_order, styleDict)
    
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize = (8*figSizeFactor,5))
    else:
        pass
    
    markersize = 5 * markersizeFactor
    
    #### Get fitting function
    if robust == False:
        my_fitting_fun = ufun.fitLine
    else:
        my_fitting_fun = ufun.fitLineHuber
    
    
    #### Get data for fit
    for i in range(Nco):
        cond = co_order[i]
        c = colors[i]
        m = markers[i]
        Xraw = data[data[condition] == cond][XCol].values
        Yraw = data[data[condition] == cond][YCol].values
        Mraw = data[data[condition] == cond]['manipID'].values
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
                print('Fitting condition ' + cond + ' with model ' + modelType)
                if modelType == 'y=ax+b':
                    params, results = my_fitting_fun(X, Y) 
                    # Y=a*X+b ; params[0] = b,  params[1] = a
                    pval = results.pvalues[1] # pvalue on the param 'a'
                    eqnText += " ; Y = {:.1f} X + {:.1f}".format(params[1], params[0])
                    eqnText += "\np-val = {:.3f}".format(pval)
                    print("Y = {:.5} X + {:.5}".format(params[1], params[0]))
                    print("p-value on the 'a' coefficient: {:.4e}".format(pval))
                    fitX = np.linspace(np.min(X), np.max(X), 100)
                    fitY = params[1]*fitX + params[0]
                    ax.plot(fitX, fitY, '--', lw = '2', 
                            color = c, zorder = 4)

                elif modelType == 'y=A*exp(kx)':
                    params, results = my_fitting_fun(X, np.log(Y)) 
                    # Y=a*X+b ; params[0] = b,  params[1] = a
                    pval = results.pvalues[1] # pvalue on the param 'k'
                    eqnText += " ; Y = {:.1f}*exp({:.1f}*X)".format(params[0], params[1])
                    eqnText += "\np-val = {:.3f}".format(pval)
                    print("Y = {:.5}*exp({:.5}*X)".format(np.exp(params[0]), params[1]))
                    print("p-value on the 'k' coefficient: {:.4e}".format(pval))
                    fitX = np.linspace(np.min(X), np.max(X), 100)
                    fitY = np.exp(params[0])*np.exp(params[1]*fitX)
                    ax.plot(fitX, fitY, '--', lw = '2', 
                            color = c, zorder = 4)
                    
                elif modelType == 'y=k*x^a':
                    posValues = ((X > 0) & (Y > 0))
                    X, Y = X[posValues], Y[posValues]
                    params, results = my_fitting_fun(np.log(X), np.log(Y)) 
                    # Y=a*X+b ; params[0] = b,  params[1] = a
                    k = np.exp(params[0])
                    a = params[1]
                    pval = results.pvalues[1] # pvalue on the param 'a'
                    eqnText += " ; Y = {:.1e} * X^{:.1f}".format(k, a)
                    eqnText += "\np-val = {:.3f}".format(pval)
                    print("Y = {:.4e} * X^{:.4f}".format(k, a))
                    print("p-value on the 'a' coefficient: {:.4e}".format(pval))
                    fitX = np.linspace(np.min(X), np.max(X), 100)
                    fitY = k * fitX**a
                    ax.plot(fitX, fitY, '--', lw = '2', 
                            color = c, zorder = 4)
                
                print('Number of values : {:.0f}'.format(len(Y)))
                print('\n')
            
            labelText = cond
            if writeEqn:
                labelText += eqnText
            if robust:
                labelText += ' (R)'

            ax.plot(X, Y, 
                    color = c, ls = '', 
                    marker = m, markersize = markersize, 
                    # markeredgecolor='k', markeredgewidth = 1, 
                    label = labelText)
            
    ax.set_xlabel(XCol)
    ax.set_ylabel(YCol)
    ax.legend()
    
    return(fig, ax)



def plotPopKS_V2(data, fig = None, ax = None, 
                 condition = '', co_order = [],
                 fitType = 'stressRegion', fitWidth=75,
                 mode = 'wholeCurve', scale = 'lin', Sinf = 0, Ssup = np.Inf):
    
    #### Init
    co_values = data[condition].unique()     
    if co_order == []:
        co_order = np.sort(co_values)
    if ax == None:
        figHeight = 5
        figWidth = 8
        fig, ax = plt.subplots(1,1, figsize=(figWidth, figHeight))
        
    # colors, markers = getStyleLists(co_order, styleDict)
    
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
        extraFilters = [data['minStress'] <= Sinf, data['maxStress'] >= Ssup] # >= 800
    
        globalExtraFilter = extraFilters[0]
        for k in range(1, len(extraFilters)):
            globalExtraFilter = globalExtraFilter & extraFilters[k]
        data = data[globalExtraFilter]
            
        ax.set_xlim([Sinf-50, Ssup+50])
        
    
    #### Pre-treatment of fitted values
    fitId = '_' + str(fitWidth)
    data_fits = taka2.getFitsInTable(data, fitType=fitType, filter_fitID=fitId)
    
    # Filter the table
    data_fits = data_fits[(data_fits['fit_center'] >= Sinf) & (data_fits['fit_center'] <= Ssup)]    
    data_fits = data_fits.drop(data_fits[data_fits['fit_error'] == True].index)
    data_fits = data_fits.drop(data_fits[data_fits['fit_K'] < 0].index)
    data_fits = data_fits.dropna(subset = ['fit_ciwK'])
    
    
    # Compute the weights
    data_fits['weight'] = (data_fits['fit_K']/data_fits['fit_ciwK'])**2
    
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
    data_fits['A'] = data_fits['fit_K'] * data_fits['weight']
    grouped1 = data_fits.groupby(by=['cellID', 'fit_center'])
    data_agg_cells = grouped1.agg({'compNum':'count',
                                   'A':'sum', 
                                   'weight': 'sum',
                                   condCol:'first'})
    data_agg_cells = data_agg_cells.reset_index()
    data_agg_cells['K_wAvg'] = data_agg_cells['A']/data_agg_cells['weight']
    data_agg_cells = data_agg_cells.rename(columns = {'compNum' : 'compCount'})
    
    # 2nd selection
    data_agg_cells = data_agg_cells.drop(data_agg_cells[data_agg_cells['compCount'] <= 1].index)
    data_agg_cells = data_agg_cells.drop(data_agg_cells[data_agg_cells['weight'] <= 1].index)
    
    grouped2 = data_agg_cells.groupby(by=[condCol, 'fit_center'])
    data_agg_all = grouped2.agg({'compCount':['sum', 'count'],
                                   'K_wAvg':['mean', 'std', 'median']})
    data_agg_all = data_agg_all.reset_index()
    data_agg_all.columns = ufun.flattenPandasIndex(data_agg_all.columns)
    data_agg_all = data_agg_all.rename(columns = {'compCount_sum' : 'compCount', 'compCount_count' : 'cellCount'})
    data_agg_all['K_wAvg_ste'] = data_agg_all['K_wAvg_std']/data_agg_all['cellCount']**0.5
    
    
    #### Plot
    i_color = 0
    
    for co in co_order:
        # try:
        try:
            color = styleDict[co]['color']
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
            
        # weighted means -- weighted ste 95% as error
        ax.errorbar(centers, Kavg/1000, yerr = q*Kste/1000, 
                    color = color, lw = 2, marker = 'o', markersize = 6, mec = 'k',
                    ecolor = color, elinewidth = 1.5, capsize = 6, capthick = 1.5, 
                    label = str(co) + ' | ' + str(total_N) + ' cells' + ' | ' + str(total_n) + ' comp')
        
        ax.legend(loc = 'upper left', fontsize = 8)
        ax.set_xlabel('Stress (Pa)')
        ax.set_ylabel('K (kPa)')
        ax.grid(visible=True, which='major', axis='y')
                    
        # except:
        #     pass


    return(fig, ax)


def StressRange_2D(data, condition='', split = False, defaultColors = False):
        
    co_values = list(data[condition].unique())
    Nco = len(co_values)
    
    if not defaultColors:
        try:
            colorList, markerList = getStyleLists(co_values, styleDict)
        except:
            colorList, markerList = gs.colorList30, gs.markerList10
    else:
        colorList, markerList = gs.colorList30, gs.markerList10

    if split:
        fig, axes = plt.subplots(Nco, 1, figsize = (8, 3*Nco), sharex = True)
    else:
        fig, axes = plt.subplots(1, 1, figsize = (8, 6))
    
    for i in range(Nco):
        if split:
            ax = axes[i]
        else:
            ax = axes
        
        co = co_values[i]
        data_co = data[data[condCol] == co]
        color = colorList[i]
        ax.scatter(data_co['bestH0'], data_co['minStress'], marker = 'o', s = 3, color = color, edgecolor = 'k', zorder=4)
        ax.scatter(data_co['bestH0'], data_co['maxStress'], marker = 'o', s = 3, color = color, edgecolor = 'k', zorder=4)
        ax.vlines(data_co['bestH0'], data_co['minStress'], data_co['maxStress'], color = colorList[i], label = co)
        ax.set_xlim([0, 800])
        ax.set_ylim([0, 7500])
        
    fig.legend()
    
    out = fig, axes    
    
    return(out)

# %% Numi's awesome plot

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
    linear = data[data.NLI_Plot=='linear'].groupby(condCol)['NLI_Plot'].count().reindex(condCat, axis = 0).reset_index()
    nonlinear = data[data.NLI_Plot=='non-linear'].groupby(condCol)['NLI_Plot'].count().reindex(condCat, axis = 0).reset_index()
    intermediate = data[data.NLI_Plot=='intermediate'].groupby(condCol)['NLI_Plot'].count().reindex(condCat, axis = 0).reset_index()

    linear['NLI_Plot'] = [i / j * 100 for i,j in zip(linear['NLI_Plot'], NComps['NLI_Plot'])]
    nonlinear['NLI_Plot'] = [i / j * 100 for i,j in zip(nonlinear['NLI_Plot'], NComps['NLI_Plot'])]
    intermediate['NLI_Plot'] = [i / j * 100 for i,j in zip(intermediate['NLI_Plot'], NComps['NLI_Plot'])]

    y1 = linear['NLI_Plot'].values
    y2 = intermediate['NLI_Plot'].values
    y3 = nonlinear['NLI_Plot'].values
    N = NComps['NLI_Plot'].values

    nonlinear['NLI_Plot'] = linear['NLI_Plot'] + nonlinear['NLI_Plot'] + intermediate['NLI_Plot']
    intermediate['NLI_Plot'] = intermediate['NLI_Plot'] + linear['NLI_Plot']

    sns.barplot(x=condCol,  y="NLI_Plot", data=nonlinear, color=palette[0],  ax = ax)
    sns.barplot(x=condCol,  y="NLI_Plot", data=intermediate, color=palette[1], ax = ax)
    sns.barplot(x=condCol,  y="NLI_Plot", data=linear, color=palette[2], ax = ax)

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

    try:
        pvals = []
        for pair in pairs:
            a1 = data['NLI'][data[condCol] == pair[0]].values
            b1 = data['NLI'][data[condCol] == pair[1]].values
            U1, p = mannwhitneyu(a1, b1, nan_policy = 'omit')
            pvals.append(p)

        annotator = Annotator(ax = ax, pairs = pairs, x=condCol,  y="NLI_Plot", data=linear)
        annotator.configure(text_format="simple", color = lineColor)
        annotator.set_pvalues(pvals).annotate()
    except:
        pass

    texts = ["Nonlinear", "Intermediate", "Linear"]
    patches = [mpatches.Patch(color=palette[i], label="{:s}".format(texts[i]) ) for i in range(len(texts)) ]

    if labels != []:
        plt.xticks(xticks, labels, **plotChars)

    plt.xticks(xticks, **plotChars)
    plt.yticks(**plotChars)
    plt.tight_layout()
    plt.legend(handles = patches, bbox_to_anchor=(1.01,0.5), fontsize = 15, labelcolor='linecolor')
    plt.show()

    return fig, ax, pvals


def createDataTable(GlobalTable):

    data_main = GlobalTable
    data_main['dateID'] = GlobalTable['date']
    data_main['manipId'] = GlobalTable['manipID']
    data_main['cellId'] = GlobalTable['cellID']
    data_main['dateCell'] = GlobalTable['date'] + '_' + GlobalTable['cellCode']

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
    data_main['NLI'] = np.log10((0.8)**-4 * K/Y)
    
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
            data_main['NLI_Plot'][j] = i
            data_main['NLI_Ind'][j] = ID
    
    return data_main

#%%% Analysing with VWC global tables

#%%%% Define data

MecaData_DrugV3 = taka2.getMergedTable('MecaData_Drugs_V3')

figSubDir = 'DrugSummary_LIMKi_02'
drugSuffix = '_LIMKi'
gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3
# dates = ['24-03-13', '24-07-04']
dates = ['24-07-04']
substrate = '20um fibronectin discs'
# df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
parameter = 'bestH0'
figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['compNum'] <= 5),
            (df['bestH0'] > 50),
            (df['bestH0'] < 1000),
           ]
df_f = filterDf(df, Filters)


df_f = createDataTable(df_f)



#%%%% Plot NLI

fig, ax = plt.subplots(figsize = (13,9), tight_layout = True)
# condCol, condCat = 'drug', drugs
condCat = ['dmso & 0.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0', 'LIMKi & 20.0', 'Y27 & 50.0', 'none & 0.0']
df_f, condCol = makeCompositeCol(df_f, cols=['drug', 'concentration'])

bp = [['dmso & 0.0', 'blebbistatin & 50.0'], ['blebbistatin & 50.0', 'blebbistatin & 100.0'], 
      ['dmso & 0.0', 'LIMKi & 20.0'], ['none & 0.0', 'Y27 & 50.0'],
      ['blebbistatin & 100.0', 'Y27 & 50.0'], ['LIMKi & 20.0', 'Y27 & 50.0']]

plotChars = {'color' : 'black', 'fontsize' : 15}

fig, ax, pvals = plotNLI(fig, ax, df_f, condCat, condCol, bp, labels = [], colorScheme = 'white', **plotChars)

plt.tight_layout()
plt.show()



# %% > Data import & export

MecaData_DrugV3 = taka2.getMergedTable('MecaData_Drugs_V3')



# %% Only some controls

figSubDir = 'DrugSummary_Controls'
drugSuffix = '_controls'

# %%%% 1. E400

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '22-11-23', '23-11-26', '23-12-03']
substrate = '20um fibronectin discs'
XCol = 'bestH0'
YCol = 'E_f_<_400'
# df, condCol = makeCompositeCol(df, cols=['drug'])
df, condCol = df, 'drug'
figname = 'H0vsE400' + drugSuffix

# Filter
Filters1 = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['none'])),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f1 = filterDf(df, Filters1)


Filters2 = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['dmso'])),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f2 = filterDf(df, Filters2)

# Order
# co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# Group By
df_fg1 = dataGroup(df_f1, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg1 = df_fg1[['bestH0']]
df_fgw1 = dataGroup_weightedAverage(df_f1, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot1 = pd.merge(left=df_fg1, right=df_fgw1, on='cellID', how='inner')

df_fg2 = dataGroup(df_f2, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg2 = df_fg2[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f2, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot2 = pd.merge(left=df_fg2, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6), sharey=True)

ax = axes[0]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot1, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot2, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'none' : 'No drug',
      'dmso' : 'DMSO',
      'E_f_<_400_wAvg':'Elastic modulus (Pa)\nfor F < 400pN - wAvg',
      # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    # ax.set_ylim(900, 12000)

# axes[0].set_xlabel('')

# Count
CountByCond1, CountByCell1 = makeCountDf(df_f1, condCol)
CountByCond2, CountByCell2 = makeCountDf(df_f2, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% 2. fit_K_250_100

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '22-11-23', '23-11-26', '23-12-03']
substrate = '20um fibronectin discs'
XCol = 'bestH0'
YCol = 'fit_K'
fitID='250_100'
# df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
df, condCol = df, 'drug'
figname = 'H0vsfit_K' + fitID + drugSuffix

# Filter
Filters1 = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['none'])),
           ]
df_f1 = filterDf(df, Filters1)


Filters2 = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['dmso'])),
           ]
df_f2 = filterDf(df, Filters2)


# Filters = [(df['validatedThickness'] == True), 
#            (df['substrate'] == substrate),
#            (df['date'].apply(lambda x : x in dates)),
#            ]
# df_f = filterDf(df, Filters)

# Order
# co_order = ['dmso & 0.0', 'latrunculinA & 0.1', 'latrunculinA & 0.5', 'latrunculinA & 2.5']

# Merge with extra data
df_ff1 = taka2.getFitsInTable(df_f1, fitType='stressGaussian', filter_fitID=fitID)

df_ff2 = taka2.getFitsInTable(df_f2, fitType='stressGaussian', filter_fitID=fitID)

# Filter again
Filters11 = [(df_ff1['fit_K'] >= 0), 
           (df_ff1['fit_K'] <= 1e5), 
           ]
df_ff1 = filterDf(df_ff1, Filters11)

Filters22 = [(df_ff2['fit_K'] >= 0), 
           (df_ff2['fit_K'] <= 1e5), 
           ]
df_ff2 = filterDf(df_ff2, Filters22)

# Group by
df_fg1 = dataGroup(df_ff1, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg1 = df_fg1[['bestH0']]
df_ffggw1 = dataGroup_weightedAverage(df_ff1, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'fit_K', weightCol = 'fit_ciwK', weight_method = 'ciw^2')
df_plot1 = pd.merge(left=df_fg1, right=df_ffggw1, on='cellID', how='inner')

df_fg2 = dataGroup(df_ff2, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg2 = df_fg2[['bestH0']]
df_ffggw2 = dataGroup_weightedAverage(df_ff2, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'fit_K', weightCol = 'fit_ciwK', weight_method = 'ciw^2')
df_plot2 = pd.merge(left=df_fg2, right=df_ffggw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

# ax = axes[0]
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# Model types :: 'y=A*exp(kx)' ;; 'y=k*x^a'

ax = axes[0]
# ax.set_xscale('log')
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot1, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
# ax.set_xscale('log')
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot2, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'none' : 'No drug',
      'dmso' : 'DMSO',
      'fit_K':'Tangeantial Modulus (Pa)',
      'fit_K_wAvg':'Tangeantial Modulus (Pa) - wAvg',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    ax.set_ylim(900, 20000)
    
# axes[0].set_xlabel('')
fig.suptitle(f"Fit on stress = {fitID.split('_')[0]}+/-{fitID.split('_')[1]} Pa")

# Count
CountByCond1, CountByCell1 = makeCountDf(df_f1, condCol)
CountByCond2, CountByCell2 = makeCountDf(df_f2, condCol)

# Show
plt.tight_layout()
plt.show()


# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %% Plots Y27

figSubDir = 'DrugSummary_Y27'
drugSuffix = '_Y27'

# %%% bestH0

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.2,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27\n(10µM)', 
      'Y27 & 50.0' : 'Y27\n(50µM)', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%% ctFieldThickness

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09']
substrate = '20um fibronectin discs'
parameter = 'ctFieldThickness'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'ctFieldThickness' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.2,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27\n(10µM)', 
      'Y27 & 50.0' : 'Y27\n(50µM)', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% E400 - Test Weighted avg

#### Test Weighted avg

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09']
substrate = '20um fibronectin discs'
parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'E400_TestAverageMethods' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
            (df['E_f_<_400'] <= 1e6),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fgw1 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^1')
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, axes = plt.subplots(1,3, figsize=(12,4))
for ax in axes:
    ax.set_yscale('log')
    
fig, axes[0] = D1Plot(df_fg, fig = fig, ax = axes[0], condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
fig, axes[1] = D1Plot(df_fgw1, fig = fig, ax = axes[1], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
fig, axes[2] = D1Plot(df_fgw2, fig = fig, ax = axes[2], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27\n(10µM)', 
      'Y27 & 50.0' : 'Y27\n(50µM)', 
      'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
      'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

titles = ['Simple Avg', 'Weighted Avg pow 1', 'Weighted Avg pow 2']
for i in range(len(axes)):
    ax = axes[i]
    t = titles[i]
    renameAxes(ax, rD, format_xticks = True)
    renameAxes(ax, renameDict, format_xticks = True)
    # renameLegend(ax, renameDict)
    ax.grid(visible=True, which='major', axis='y')
    ax.set_title(t)
    ax.set_xlabel('')
    if i >= 1:
        ax.set_ylabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


#### E400

# Define
figname = 'E400' + drugSuffix


# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
ax.set_yscale('log')

fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')




# %%% Fluctuations

#### All
# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09']
substrate = '20um fibronectin discs'
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'fluctuations_all' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(8,5))
    
fig, ax = D2Plot_wFit(df_fg, fig = fig, ax = ax, 
                XCol=XCol, YCol=YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27 (10µM)', 
      'Y27 & 50.0' : 'Y27 (50µM)', 
      # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, rD)

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### 1 by 1
# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09']
substrate = '20um fibronectin discs'
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'fluctuations_1by1' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,3, figsize=(16,5))

for i in range(len(axes)):
    ax = axes[i]
    fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
                    XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
                    modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                    figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27 (10µM)', 
      'Y27 & 50.0' : 'Y27 (50µM)', 
      # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = True)
    renameAxes(ax, renameDict, format_xticks = True)
    renameLegend(ax, rD, loc='upper left')
    ax.set_xlim([0, 600])
    ax.set_ylim([0, 300])

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%% Non-linearity

# %%%% 1. Global

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = 'wholeCurve'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
# co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))
    
plotPopKS_V2(df_f, fig = fig, ax = axes[0], 
                 fitType = 'stressGaussian', fitWidth=75, condition = condCol,
                  mode = mode, Sinf = 0, Ssup = np.Inf)
plotPopKS_V2(df_f, fig = fig, ax = axes[1], 
                 fitType = 'stressGaussian', fitWidth=75, condition = condCol,
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27 (10µM)', 
      'Y27 & 50.0' : 'Y27 (50µM)', 
      # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }


for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)

ax = axes[0]
ax.set_xlim(0, 1200)
ax.set_ylim(0, 20)
ax = axes[1]
ax.set_xlim(0, 600)
ax.set_ylim(0, 8)

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% 2. Local

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = '200_500'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
# co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,6))
    
plotPopKS_V2(df_f, fig = fig, ax = ax, 
                 fitType = 'stressGaussian', fitWidth=75, condition = condCol,
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27 (10µM)', 
      'Y27 & 50.0' : 'Y27 (50µM)', 
      # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)

ax.set_title(f"Stress Range = {mode.split('_')[0]} - {mode.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% Thickness & Stiffness

# %%%% 1. E400

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09']
substrate = '20um fibronectin discs'
XCol = 'bestH0'
YCol = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'H0vsE400' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           # (df['drug'] == 'dmso'),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

# Order
# co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

ax = axes[0]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'E_f_<_400_wAvg':'Elastic modulus (Pa)\nfor F < 400pN - wAvg',
      # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    # ax.set_ylim(900, 12000)

# axes[0].set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% 2. fit_K_250_100

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09']
substrate = '20um fibronectin discs'
XCol = 'bestH0'
YCol = 'fit_K'
fitID='250_100'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'H0vsfit_K' + fitID + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
# co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# Merge with extra data
df_ff = taka2.getFitsInTable(df_f, fitType='stressGaussian', filter_fitID=fitID)

# Filter again
Filters = [(df_ff['fit_K'] >= 0), 
           (df_ff['fit_K'] <= 1e5), 
           ]
df_ff = filterDf(df_ff, Filters)

# Group by
df_fg = dataGroup(df_ff, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_ffggw2 = dataGroup_weightedAverage(df_ff, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'fit_K', weightCol = 'fit_ciwK', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_ffggw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

ax = axes[0]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'fit_K':'Tangeantial Modulus (Pa)',
      'fit_K_wAvg':'Tangeantial Modulus (Pa) - wAvg',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    ax.set_ylim(900, 12000)
    
# axes[0].set_xlabel('')
fig.suptitle(f"Fit on stress = {fitID.split('_')[0]}+/-{fitID.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()


# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %% Plots LatA

figSubDir = 'DrugSummary_LatA'
drugSuffix = '_LatA'

# %%% bestH0 # LOWVALUES == ON !

# Define
df = MecaData_DrugV3
dates = ['22-11-23', '23-11-26', '23-12-03']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'bestH0_LOWVALUES' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           # (df['bestH0'] > 50),
           (df['bestH0'] < 450),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'latrunculinA & 0.1', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[3], bp[5]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 1,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'latrunculinA & 0.1' : 'LatA\n(0.1µM)', 
      'latrunculinA & 0.5' : 'LatA\n(0.5µM)', 
      'latrunculinA & 2.5' : 'LatA\n(2.5µM)', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% ctFieldThickness # LOWVALUES == ON !

# Define
df = MecaData_DrugV3
dates = ['22-11-23', '23-11-26', '23-12-03']
substrate = '20um fibronectin discs'
parameter = 'ctFieldThickness'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'ctFieldThickness_LOWVALUES' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['ctFieldThickness'] < 450),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'latrunculinA & 0.1', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[3], bp[5]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 1,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'latrunculinA & 0.1' : 'LatA\n(0.1µM)', 
      'latrunculinA & 0.5' : 'LatA\n(0.5µM)', 
      'latrunculinA & 2.5' : 'LatA\n(2.5µM)', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% E400 - Test Weighted avg

#### Test Weighted avg

# Define
df = MecaData_DrugV3
dates = ['22-11-23', '23-11-26', '23-12-03']
substrate = '20um fibronectin discs'
parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'E400_TestAverageMethods' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
            (df['E_f_<_400'] <= 1e6),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'latrunculinA & 0.1', 'latrunculinA & 0.5', 'latrunculinA & 2.5']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[3], bp[5]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fgw1 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^1')
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, axes = plt.subplots(1,3, figsize=(12,4))
for ax in axes:
    ax.set_yscale('log')
    
fig, axes[0] = D1Plot(df_fg, fig = fig, ax = axes[0], condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
fig, axes[1] = D1Plot(df_fgw1, fig = fig, ax = axes[1], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
fig, axes[2] = D1Plot(df_fgw2, fig = fig, ax = axes[2], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'latrunculinA & 0.1' : 'LatA\n(0.1µM)', 
      'latrunculinA & 0.5' : 'LatA\n(0.5µM)', 
      'latrunculinA & 2.5' : 'LatA\n(2.5µM)', 
      'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
      'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

titles = ['Simple Avg', 'Weighted Avg pow 1', 'Weighted Avg pow 2']
for i in range(len(axes)):
    ax = axes[i]
    t = titles[i]
    renameAxes(ax, rD, format_xticks = True)
    renameAxes(ax, renameDict, format_xticks = True)
    # renameLegend(ax, renameDict)
    ax.grid(visible=True, which='major', axis='y')
    ax.set_title(t)
    ax.set_xlabel('')
    if i >= 1:
        ax.set_ylabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


#### E400

# Define
figname = 'E400' + drugSuffix


# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
ax.set_yscale('log')

fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')




# %%% Fluctuations

#### All
# Define
df = MecaData_DrugV3
dates = ['22-11-23', '23-11-26', '23-12-03']
substrate = '20um fibronectin discs'
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'fluctuations_all' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df[XCol] > 20),
           # (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'latrunculinA & 0.1', 'latrunculinA & 0.5', 'latrunculinA & 2.5']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(8,6))
    
fig, ax = D2Plot_wFit(df_fg, fig = fig, ax = ax, 
                XCol=XCol, YCol=YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'latrunculinA & 0.1' : 'LatA (0.1µM)', 
      'latrunculinA & 0.5' : 'LatA (0.5µM)', 
      'latrunculinA & 2.5' : 'LatA (2.5µM)', 
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### 1 by 1
# Define
df = MecaData_DrugV3
dates = ['22-11-23', '23-11-26', '23-12-03']
substrate = '20um fibronectin discs'
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'fluctuations_1by1' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df[XCol] > 20),
           # (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'latrunculinA & 0.1', 'latrunculinA & 0.5', 'latrunculinA & 2.5']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,4, figsize=(16,6))

for i in range(len(axes)):
    ax = axes[i]
    fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
                    XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
                    modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                    figSizeFactor = 1, markersizeFactor = 1)
    if i >= 1:
        ax.set_ylabel('')
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'latrunculinA & 0.1 ; ' : 'LatA (0.1µM)\n', 
      'latrunculinA & 0.5 ; ' : 'LatA (0.5µM)\n', 
      'latrunculinA & 2.5 ; ' : 'LatA (2.5µM)\n', 
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD, loc='upper left')
    ax.set_xlim([0, 800])
    ax.set_ylim([0, 500])
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%% Non-linearity

# %%%% 1. Global

# Define
df = MecaData_DrugV3
dates = ['22-11-23', '23-11-26', '23-12-03']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = 'wholeCurve'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'latrunculinA & 0.1', 'latrunculinA & 0.5', 'latrunculinA & 2.5']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))
    
plotPopKS_V2(df_f, fig = fig, ax = axes[0], condition = condCol, co_order = co_order,
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
plotPopKS_V2(df_f, fig = fig, ax = axes[1], condition = condCol, co_order = co_order, 
                 fitType = 'stressGaussian', fitWidth=75,
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'latrunculinA & 0.1' : 'LatA (0.1µM)', 
      'latrunculinA & 0.5' : 'LatA (0.5µM)', 
      'latrunculinA & 2.5' : 'LatA (2.5µM)', 
      }


for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)

ax = axes[0]
ax.set_xlim(0, 1200)
ax.set_ylim(0, 20)
ax = axes[1]
ax.set_xlim(0, 600)
ax.set_ylim(0, 8)

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% 2. Local

# Define
df = MecaData_DrugV3
dates = ['22-11-23', '23-11-26', '23-12-03']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = '200_500'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'latrunculinA & 0.1', 'latrunculinA & 0.5', 'latrunculinA & 2.5']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,6))
    
plotPopKS_V2(df_f, fig = fig, ax = ax, condition = condCol, co_order = co_order,  
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'latrunculinA & 0.1' : 'LatA (0.1µM)', 
      'latrunculinA & 0.5' : 'LatA (0.5µM)', 
      'latrunculinA & 2.5' : 'LatA (2.5µM)', 
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)
ax.set_ylim(0, 15)

ax.set_title(f"Stress Range = {mode.split('_')[0]} - {mode.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% Thickness & Stiffness

# %%%% 1.1. E400

# Define
df = MecaData_DrugV3
dates = ['22-11-23', '23-11-26', '23-12-03']
substrate = '20um fibronectin discs'
XCol = 'bestH0'
YCol = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'H0vsE400' + drugSuffix + '_lin'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           # (df['drug'] == 'dmso'),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

# Order
# co_order = ['dmso & 0.0', 'latrunculinA & 0.1', 'latrunculinA & 0.5', 'latrunculinA & 2.5']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

# ax = axes[0]
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

ax = axes[0]
ax.set_xscale('linear')
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_xscale('linear')
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'latrunculinA & 0.1' : 'LatA (0.1µM)', 
      'latrunculinA & 0.5' : 'LatA (0.5µM)', 
      'latrunculinA & 2.5' : 'LatA (2.5µM)', 
      'E_f_<_400_wAvg':'Elastic modulus (Pa)\nfor F < 400pN - wAvg',
      # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    # ax.set_ylim(900, 12000)

# axes[0].set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% 1.2. E400

# Define
df = MecaData_DrugV3
dates = ['22-11-23', '23-11-26', '23-12-03']
substrate = '20um fibronectin discs'
XCol = 'bestH0'
YCol = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'H0vsE400' + drugSuffix + '_log'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           # (df['drug'] == 'dmso'),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

# Order
# co_order = ['dmso & 0.0', 'latrunculinA & 0.1', 'latrunculinA & 0.5', 'latrunculinA & 2.5']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

# ax = axes[0]
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

ax = axes[0]
ax.set_xscale('log')
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
                modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
                modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'latrunculinA & 0.1' : 'LatA (0.1µM)', 
      'latrunculinA & 0.5' : 'LatA (0.5µM)', 
      'latrunculinA & 2.5' : 'LatA (2.5µM)', 
      'E_f_<_400_wAvg':'Elastic modulus (Pa)\nfor F < 400pN - wAvg',
      # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    # ax.set_ylim(900, 12000)

# axes[0].set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% 2. fit_K_250_100

# Define
df = MecaData_DrugV3
dates = ['22-11-23', '23-11-26', '23-12-03']
substrate = '20um fibronectin discs'
XCol = 'bestH0'
YCol = 'fit_K'
fitID='250_100'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'H0vsfit_K' + fitID + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
# co_order = ['dmso & 0.0', 'latrunculinA & 0.1', 'latrunculinA & 0.5', 'latrunculinA & 2.5']

# Merge with extra data
df_ff = taka2.getFitsInTable(df_f, fitType='stressGaussian', filter_fitID=fitID)

# Filter again
Filters = [(df_ff['fit_K'] >= 0), 
           (df_ff['fit_K'] <= 1e5), 
           ]
df_ff = filterDf(df_ff, Filters)

# Group by
df_fg = dataGroup(df_ff, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_ffggw2 = dataGroup_weightedAverage(df_ff, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'fit_K', weightCol = 'fit_ciwK', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_ffggw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

# ax = axes[0]
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

ax = axes[0]
ax.set_xscale('log')
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
                modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
                modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'latrunculinA & 0.1' : 'LatA (0.1µM)', 
      'latrunculinA & 0.5' : 'LatA (0.5µM)', 
      'latrunculinA & 2.5' : 'LatA (2.5µM)', 
      'fit_K':'Tangeantial Modulus (Pa)',
      'fit_K_wAvg':'Tangeantial Modulus (Pa) - wAvg',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    ax.set_ylim(900, 20000)
    
# axes[0].set_xlabel('')
fig.suptitle(f"Fit on stress = {fitID.split('_')[0]}+/-{fitID.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()


# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
















# %% Plots Blebbi

figSubDir = 'DrugSummary_Blebbi'
drugSuffix = '_Blebbi'

# %%% bestH0 # LOWVALUES == 600 !

# Define
df = MecaData_DrugV3
dates = ['22-03-28', '23-02-16', '23-03-17', '23-04-20']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'bestH0_cutoff600' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['dmso', 'blebbistatin'])),
           # (df['bestH0'] > 50),
            (df['bestH0'] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 250.0']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[1], bp[2]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbi\n(10µM)', 
      'blebbistatin & 50.0' : 'Blebbi\n(50µM)', 
      'blebbistatin & 250.0' : 'Blebbi\n(250µM)', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% ctFieldThickness # LOWVALUES == 500 !

# Define
df = MecaData_DrugV3
dates = ['22-03-28', '23-02-16', '23-03-17', '23-04-20']
substrate = '20um fibronectin discs'
parameter = 'ctFieldThickness'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'ctFieldThickness_cutoff600' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['ctFieldThickness'] < 500),
           (df['drug'].apply(lambda x : x in ['dmso', 'blebbistatin'])),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 250.0']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[1], bp[2]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbi\n(10µM)', 
      'blebbistatin & 50.0' : 'Blebbi\n(50µM)', 
      'blebbistatin & 250.0' : 'Blebbi\n(250µM)', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% E400 - Test Weighted avg

#### Test Weighted avg

# Define
df = MecaData_DrugV3
dates = ['22-03-28', '23-02-16', '23-03-17', '23-04-20']
substrate = '20um fibronectin discs'
parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'E400_TestAverageMethods' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
            (df['E_f_<_400'] <= 1e6),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['dmso', 'blebbistatin'])),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 250.0']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[1], bp[2]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fgw1 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^1')
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, axes = plt.subplots(1,3, figsize=(12,4))
for ax in axes:
    ax.set_yscale('log')
    
fig, axes[0] = D1Plot(df_fg, fig = fig, ax = axes[0], condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
fig, axes[1] = D1Plot(df_fgw1, fig = fig, ax = axes[1], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
fig, axes[2] = D1Plot(df_fgw2, fig = fig, ax = axes[2], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbi\n(10µM)', 
      'blebbistatin & 50.0' : 'Blebbi\n(50µM)', 
      'blebbistatin & 250.0' : 'Blebbi\n(250µM)', 
      'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
      'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

titles = ['Simple Avg', 'Weighted Avg pow 1', 'Weighted Avg pow 2']
for i in range(len(axes)):
    ax = axes[i]
    t = titles[i]
    renameAxes(ax, rD, format_xticks = True)
    renameAxes(ax, renameDict, format_xticks = True)
    # renameLegend(ax, renameDict)
    ax.grid(visible=True, which='major', axis='y')
    ax.set_title(t)
    ax.set_xlabel('')
    if i >= 1:
        ax.set_ylabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


#### E400

# Define
figname = 'E400' + drugSuffix


# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
ax.set_yscale('log')

fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')




# %%% Fluctuations

#### All
# Define
df = MecaData_DrugV3
dates = ['22-03-28', '23-02-16', '23-03-17', '23-04-20']
substrate = '20um fibronectin discs'
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'fluctuations_all' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['dmso', 'blebbistatin'])),
           (df[XCol] > 20),
            (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 250.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(8,6))
    
fig, ax = D2Plot_wFit(df_fg, fig = fig, ax = ax, 
                XCol=XCol, YCol=YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbi (10µM)', 
      'blebbistatin & 50.0' : 'Blebbi (50µM)', 
      'blebbistatin & 250.0' : 'Blebbi (250µM)', 
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)

ax.set_xlim([0, 600])
ax.set_ylim([0, 350])

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### 1 by 1
# Define
df = MecaData_DrugV3
dates = ['22-03-28', '23-02-16', '23-03-17', '23-04-20']
substrate = '20um fibronectin discs'
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'fluctuations_1by1' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['dmso', 'blebbistatin'])),
           (df[XCol] > 20),
           # (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 250.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,4, figsize=(16,6))

for i in range(len(axes)):
    ax = axes[i]
    fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
                    XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
                    modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                    figSizeFactor = 1, markersizeFactor = 1)
    if i >= 1:
        ax.set_ylabel('')
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0 ; ' : 'Blebbi (10µM)\n', 
      'blebbistatin & 50.0 ; ' : 'Blebbi (50µM)\n', 
      'blebbistatin & 250.0 ; ' : 'Blebbi (250µM)\n', 
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD, loc='upper left')
    ax.set_xlim([0, 600])
    ax.set_ylim([0, 350])
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%% Non-linearity

# %%%% 1. Global

# Define
df = MecaData_DrugV3
dates = ['22-03-28', '23-02-16', '23-03-17', '23-04-20']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = 'wholeCurve'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['dmso', 'blebbistatin'])),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 250.0']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))
    
plotPopKS_V2(df_f, fig = fig, ax = axes[0], condition = condCol, co_order = co_order,
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
plotPopKS_V2(df_f, fig = fig, ax = axes[1], condition = condCol, co_order = co_order, 
                 fitType = 'stressGaussian', fitWidth=75,
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbi (10µM)', 
      'blebbistatin & 50.0' : 'Blebbi (50µM)', 
      'blebbistatin & 250.0' : 'Blebbi (250µM)', 
      }


for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)

ax = axes[0]
ax.set_xlim(0, 1200)
ax.set_ylim(0, 20)
ax = axes[1]
ax.set_xlim(0, 600)
ax.set_ylim(0, 8)

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% 2. Local

# Define
df = MecaData_DrugV3
dates = ['22-03-28', '23-02-16', '23-03-17', '23-04-20']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = '200_500'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['dmso', 'blebbistatin'])),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 250.0']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,6))
    
plotPopKS_V2(df_f, fig = fig, ax = ax, condition = condCol, co_order = co_order,  
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbi (10µM)', 
      'blebbistatin & 50.0' : 'Blebbi (50µM)', 
      'blebbistatin & 250.0' : 'Blebbi (250µM)', 
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)
ax.set_ylim(0, 15)

ax.set_title(f"Stress Range = {mode.split('_')[0]} - {mode.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% Thickness & Stiffness

# %%%% 1. E400

# Define
df = MecaData_DrugV3
dates = ['22-03-28', '23-02-16', '23-03-17', '23-04-20']
substrate = '20um fibronectin discs'
XCol = 'bestH0'
YCol = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'H0vsE400' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['dmso', 'blebbistatin'])),
           # (df['drug'] == 'dmso'),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 250.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

# LinLog
ax = axes[0]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
# LogLog
# ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbi (10µM)', 
      'blebbistatin & 50.0' : 'Blebbi (50µM)', 
      'blebbistatin & 250.0' : 'Blebbi (250µM)', 
      'E_f_<_400_wAvg':'Elastic modulus (Pa)\nfor F < 400pN - wAvg',
      # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    # ax.set_ylim(900, 12000)

# axes[0].set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% 2. fit_K_250_100

# Define
df = MecaData_DrugV3
dates = ['22-03-28', '23-02-16', '23-03-17', '23-04-20']
substrate = '20um fibronectin discs'
XCol = 'bestH0'
YCol = 'fit_K'
fitID='250_100'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'H0vsfit_K' + fitID + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['dmso', 'blebbistatin'])),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 250.0']

# Merge with extra data
df_ff = taka2.getFitsInTable(df_f, fitType='stressGaussian', filter_fitID=fitID)

# Filter again
Filters = [(df_ff['fit_K'] >= 0), 
           (df_ff['fit_K'] <= 1e5), 
           ]
df_ff = filterDf(df_ff, Filters)

# Group by
df_fg = dataGroup(df_ff, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_ffggw2 = dataGroup_weightedAverage(df_ff, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'fit_K', weightCol = 'fit_ciwK', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_ffggw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

ax = axes[0]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbi (10µM)', 
      'blebbistatin & 50.0' : 'Blebbi (50µM)', 
      'blebbistatin & 250.0' : 'Blebbi (250µM)', 
      'fit_K':'Tangeantial Modulus (Pa)',
      'fit_K_wAvg':'Tangeantial Modulus (Pa) - wAvg',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    ax.set_ylim(900, 20000)
    
# axes[0].set_xlabel('')
fig.suptitle(f"Fit on stress = {fitID.split('_')[0]}+/-{fitID.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()


# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')








# %% Plots JLY

figSubDir = 'DrugSummary_JLY'
drugSuffix = '_JLY'

# %%% bestH0 # LOWVALUES == 600 !

# Define
df = MecaData_DrugV3
dates = ['23-09-19']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
parameter = 'bestH0'
figname = 'bestH0_cutoff600' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           # (df['bestH0'] > 50),
            (df['bestH0'] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'JLY & 8-5-10']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.2,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'JLY & 8-5-10' : 'JLY\n(8/5/10µM)',
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% ctFieldThickness # LOWVALUES == 500 !

# Define
df = MecaData_DrugV3
dates = ['23-09-19']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
parameter = 'ctFieldThickness'
figname = 'ctFieldThickness_cutoff600' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['ctFieldThickness'] < 500),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'JLY & 8-5-10']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'JLY & 8-5-10' : 'JLY\n(8/5/10µM)',
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% E400 - Test Weighted avg

#### Test Weighted avg

# Define
df = MecaData_DrugV3
dates = ['23-09-19']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
parameter = 'E_f_<_400'
figname = 'E400_TestAverageMethods' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
            (df['E_f_<_400'] <= 1e6),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'JLY & 8-5-10']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fgw1 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^1')
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, axes = plt.subplots(1,3, figsize=(12,4))
for ax in axes:
    ax.set_yscale('log')
    
fig, axes[0] = D1Plot(df_fg, fig = fig, ax = axes[0], condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
fig, axes[1] = D1Plot(df_fgw1, fig = fig, ax = axes[1], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
fig, axes[2] = D1Plot(df_fgw2, fig = fig, ax = axes[2], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'JLY & 8-5-10' : 'JLY\n(8/5/10µM)',
      'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
      'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

titles = ['Simple Avg', 'Weighted Avg pow 1', 'Weighted Avg pow 2']
for i in range(len(axes)):
    ax = axes[i]
    t = titles[i]
    renameAxes(ax, rD, format_xticks = True)
    renameAxes(ax, renameDict, format_xticks = True)
    # renameLegend(ax, renameDict)
    ax.grid(visible=True, which='major', axis='y')
    ax.set_title(t)
    ax.set_xlabel('')
    if i >= 1:
        ax.set_ylabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


#### E400

# Define
figname = 'E400' + drugSuffix


# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
ax.set_yscale('log')

fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')




# %%% Fluctuations

#### All
# Define
df = MecaData_DrugV3
dates = ['23-09-19']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
figname = 'fluctuations_all' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df[XCol] > 20),
            (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'JLY & 8-5-10']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(8,6))
    
fig, ax = D2Plot_wFit(df_fg, fig = fig, ax = ax, 
                XCol=XCol, YCol=YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'JLY & 8-5-10' : 'JLY (8/5/10µM)',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)

ax.set_xlim([0, 600])
ax.set_ylim([0, 350])

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### 1 by 1
# Define
df = MecaData_DrugV3
dates = ['23-09-19']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
figname = 'fluctuations_1by1' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df[XCol] > 20),
           # (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'JLY & 8-5-10']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,2, figsize=(10,5))

for i in range(len(axes)):
    ax = axes[i]
    fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
                    XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
                    modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                    figSizeFactor = 1, markersizeFactor = 1)
    if i >= 1:
        ax.set_ylabel('')
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'JLY & 8-5-10' : 'JLY (8/5/10µM)',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD, loc='upper left')
    ax.set_xlim([0, 600])
    ax.set_ylim([0, 350])
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%% Non-linearity

# %%%% 1. Global

# Define
df = MecaData_DrugV3
dates = ['23-09-19']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = 'wholeCurve'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'JLY & 8-5-10']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))
    
plotPopKS_V2(df_f, fig = fig, ax = axes[0], condition = condCol, co_order = co_order,
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
plotPopKS_V2(df_f, fig = fig, ax = axes[1], condition = condCol, co_order = co_order, 
                 fitType = 'stressGaussian', fitWidth=75,
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'JLY & 8-5-10' : 'JLY (8/5/10µM)',
      }


for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)

ax = axes[0]
ax.set_xlim(0, 1200)
ax.set_ylim(0, 20)
ax = axes[1]
ax.set_xlim(0, 600)
ax.set_ylim(0, 8)

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% 2. Local

#### 200_500
# Define
df = MecaData_DrugV3
dates = ['23-09-19']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = '200_500'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'JLY & 8-5-10']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,6))
    
plotPopKS_V2(df_f, fig = fig, ax = ax, condition = condCol, co_order = co_order,  
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'JLY & 8-5-10' : 'JLY (8/5/10µM)',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)
ax.set_ylim(0, 15)

ax.set_title(f"Stress Range = {mode.split('_')[0]} - {mode.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### 400_700
# Define
df = MecaData_DrugV3
dates = ['23-09-19']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = '400_700'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'JLY & 8-5-10']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,6))
    
plotPopKS_V2(df_f, fig = fig, ax = ax, condition = condCol, co_order = co_order,  
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'JLY & 8-5-10' : 'JLY (8/5/10µM)',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)
ax.set_ylim(0, 15)

ax.set_title(f"Stress Range = {mode.split('_')[0]} - {mode.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% Thickness & Stiffness

# %%%% 1. E400

# Define
df = MecaData_DrugV3
dates = ['23-09-19']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'bestH0'
YCol = 'E_f_<_400'
figname = 'H0vsE400' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           # (df['drug'] == 'dmso'),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'JLY & 8-5-10']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

# LinLog
ax = axes[0]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
# LogLog
# ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'JLY & 8-5-10' : 'JLY (8/5/10µM)', 
      'E_f_<_400_wAvg':'Elastic modulus (Pa)\nfor F < 400pN - wAvg',
      # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    # ax.set_ylim(900, 12000)

# axes[0].set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% 2. fit_K_400_100

# Define
df = MecaData_DrugV3
dates = ['23-09-19']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'bestH0'
YCol = 'fit_K'
fitID='400_100'
figname = 'H0vsfit_K' + fitID + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'JLY & 8-5-10']

# Merge with extra data
df_ff = taka2.getFitsInTable(df_f, fitType='stressGaussian', filter_fitID=fitID)

# Filter again
Filters = [(df_ff['fit_K'] >= 0), 
           (df_ff['fit_K'] <= 1e5), 
           ]
df_ff = filterDf(df_ff, Filters)

# Group by
df_fg = dataGroup(df_ff, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_ffggw2 = dataGroup_weightedAverage(df_ff, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'fit_K', weightCol = 'fit_ciwK', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_ffggw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

ax = axes[0]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'JLY & 8-5-10' : 'JLY (8/5/10µM)',
      'fit_K':'Tangeantial Modulus (Pa)',
      'fit_K_wAvg':'Tangeantial Modulus (Pa) - wAvg',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    ax.set_ylim(900, 20000)
    
# axes[0].set_xlabel('')
fig.suptitle(f"Fit on stress = {fitID.split('_')[0]}+/-{fitID.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()


# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')





# %% Plots Ck666

figSubDir = 'DrugSummary_ck666'
drugSuffix = '_ck666'

# %%% bestH0 # LOWVALUES == 600 !

# Define
df = MecaData_DrugV3
dates = ['23-04-26', '23-04-28']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
parameter = 'bestH0'
figname = 'bestH0_cutoff600' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           # (df['bestH0'] > 50),
            (df['bestH0'] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'ck666 & 50.0', 'ck666 & 100.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.2,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'ck666 & 50.0' : 'CK666\n(50µM)',
      'ck666 & 100.0' : 'CK666\n(100µM)',
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% ctFieldThickness # LOWVALUES == 500 !

# Define
df = MecaData_DrugV3
dates = ['23-04-26', '23-04-28']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
parameter = 'ctFieldThickness'
figname = 'ctFieldThickness_cutoff600' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['ctFieldThickness'] < 500),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'ck666 & 50.0', 'ck666 & 100.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.2,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'ck666 & 50.0' : 'CK666\n(50µM)',
      'ck666 & 100.0' : 'CK666\n(100µM)',
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% E400 - Test Weighted avg

#### Test Weighted avg

# Define
df = MecaData_DrugV3
dates = ['23-04-26', '23-04-28']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
parameter = 'E_f_<_400'
figname = 'E400_TestAverageMethods' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
            (df['E_f_<_400'] <= 1e6),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'ck666 & 50.0', 'ck666 & 100.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fgw1 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^1')
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, axes = plt.subplots(1,3, figsize=(12,4))
for ax in axes:
    ax.set_yscale('log')
    
fig, axes[0] = D1Plot(df_fg, fig = fig, ax = axes[0], condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
fig, axes[1] = D1Plot(df_fgw1, fig = fig, ax = axes[1], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
fig, axes[2] = D1Plot(df_fgw2, fig = fig, ax = axes[2], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'ck666 & 50.0' : 'CK666\n(50µM)',
      'ck666 & 100.0' : 'CK666\n(100µM)',
      'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
      'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

titles = ['Simple Avg', 'Weighted Avg pow 1', 'Weighted Avg pow 2']
for i in range(len(axes)):
    ax = axes[i]
    t = titles[i]
    renameAxes(ax, rD, format_xticks = True)
    renameAxes(ax, renameDict, format_xticks = True)
    # renameLegend(ax, renameDict)
    ax.grid(visible=True, which='major', axis='y')
    ax.set_title(t)
    ax.set_xlabel('')
    if i >= 1:
        ax.set_ylabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


#### E400

# Define
figname = 'E400' + drugSuffix


# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
ax.set_yscale('log')

fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')




# %%% Fluctuations

#### All
# Define
df = MecaData_DrugV3
dates = ['23-04-26', '23-04-28']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
figname = 'fluctuations_all' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df[XCol] > 20),
            (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'ck666 & 50.0', 'ck666 & 100.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(8,6))
    
fig, ax = D2Plot_wFit(df_fg, fig = fig, ax = ax, 
                XCol=XCol, YCol=YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'ck666 & 50.0' : 'CK666 (50µM)',
      'ck666 & 100.0' : 'CK666 (100µM)',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)

ax.set_xlim([0, 600])
ax.set_ylim([0, 400])

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### 1 by 1
# Define
df = MecaData_DrugV3
dates = ['23-04-26', '23-04-28']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
figname = 'fluctuations_1by1' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df[XCol] > 20),
           # (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'ck666 & 50.0', 'ck666 & 100.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,3, figsize=(15,5))

for i in range(len(axes)):
    ax = axes[i]
    fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
                    XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
                    modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                    figSizeFactor = 1, markersizeFactor = 1)
    if i >= 1:
        ax.set_ylabel('')
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'ck666 & 50.0' : 'CK666 (50µM)',
      'ck666 & 100.0' : 'CK666 (100µM)',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD, loc='upper left')
    ax.set_xlim([0, 600])
    ax.set_ylim([0, 400])
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%% Non-linearity

# %%%% 1. Global

# Define
df = MecaData_DrugV3
dates = ['23-04-26', '23-04-28']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = 'wholeCurve'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'ck666 & 50.0', 'ck666 & 100.0']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))
    
plotPopKS_V2(df_f, fig = fig, ax = axes[0], condition = condCol, co_order = co_order,
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
plotPopKS_V2(df_f, fig = fig, ax = axes[1], condition = condCol, co_order = co_order, 
                 fitType = 'stressGaussian', fitWidth=75,
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'ck666 & 50.0' : 'CK666 (50µM)',
      'ck666 & 100.0' : 'CK666 (100µM)',
      }


for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)

ax = axes[0]
ax.set_xlim(0, 1200)
ax.set_ylim(0, 20)
ax = axes[1]
ax.set_xlim(0, 600)
ax.set_ylim(0, 8)

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% 2. Local

#### 200_500
# Define
df = MecaData_DrugV3
dates = ['23-04-26', '23-04-28']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = '200_500'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'ck666 & 50.0', 'ck666 & 100.0']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,6))
    
plotPopKS_V2(df_f, fig = fig, ax = ax, condition = condCol, co_order = co_order,  
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'ck666 & 50.0' : 'CK666 (50µM)',
      'ck666 & 100.0' : 'CK666 (100µM)',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)
ax.set_ylim(0, 15)

ax.set_title(f"Stress Range = {mode.split('_')[0]} - {mode.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### 400_700
# Define
df = MecaData_DrugV3
dates = ['23-04-26', '23-04-28']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = '400_700'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'ck666 & 50.0', 'ck666 & 100.0']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,6))
    
plotPopKS_V2(df_f, fig = fig, ax = ax, condition = condCol, co_order = co_order,  
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'ck666 & 50.0' : 'CK666 (50µM)',
      'ck666 & 100.0' : 'CK666 (100µM)',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)
ax.set_ylim(0, 15)

ax.set_title(f"Stress Range = {mode.split('_')[0]} - {mode.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% Thickness & Stiffness

# %%%% 1. E400

# Define
df = MecaData_DrugV3
dates = ['23-04-26', '23-04-28']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'bestH0'
YCol = 'E_f_<_400'
figname = 'H0vsE400' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           # (df['drug'] == 'dmso'),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'ck666 & 50.0', 'ck666 & 100.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

# LinLog
ax = axes[0]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
# LogLog
# ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'ck666 & 50.0' : 'CK666 (50µM)',
      'ck666 & 100.0' : 'CK666 (100µM)', 
      'E_f_<_400_wAvg':'Elastic modulus (Pa)\nfor F < 400pN - wAvg',
      # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    # ax.set_ylim(900, 12000)

# axes[0].set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% 2. fit_K_250_100

# Define
df = MecaData_DrugV3
dates = ['23-04-26', '23-04-28']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'bestH0'
YCol = 'fit_K'
fitID='250_100'
figname = 'H0vsfit_K' + fitID + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'ck666 & 50.0', 'ck666 & 100.0']

# Merge with extra data
df_ff = taka2.getFitsInTable(df_f, fitType='stressGaussian', filter_fitID=fitID)

# Filter again
Filters = [(df_ff['fit_K'] >= 0), 
           (df_ff['fit_K'] <= 1e5), 
           ]
df_ff = filterDf(df_ff, Filters)

# Group by
df_fg = dataGroup(df_ff, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_ffggw2 = dataGroup_weightedAverage(df_ff, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'fit_K', weightCol = 'fit_ciwK', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_ffggw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

ax = axes[0]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'ck666 & 50.0' : 'CK666 (50µM)',
      'ck666 & 100.0' : 'CK666 (100µM)',
      'fit_K':'Tangeantial Modulus (Pa)',
      'fit_K_wAvg':'Tangeantial Modulus (Pa) - wAvg',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    ax.set_ylim(900, 20000)
    
# axes[0].set_xlabel('')
fig.suptitle(f"Fit on stress = {fitID.split('_')[0]}+/-{fitID.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()


# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')









# %% Plots CalyculinA

figSubDir = 'DrugSummary_CalA'
drugSuffix = '_CalA'

# %%% bestH0 # LOWVALUES == 600 !

# Define
df = MecaData_DrugV3
dates = ['23-07-17', '23-07-20', '23-09-06']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
parameter = 'bestH0'
figname = 'bestH0_cutoff600' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           # (df['bestH0'] > 50),
            (df['bestH0'] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'calyculinA & 0.25', 'calyculinA & 0.5', 'calyculinA & 1.0', 'calyculinA & 2.0']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[1], bp[2], bp[3]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'calyculinA & 0.25' : 'CalA\n(0.25µM)',
      'calyculinA & 0.5' : 'CalA\n(0.5µM)',
      'calyculinA & 1.0' : 'CalA\n(1.0µM)',
      'calyculinA & 2.0' : 'CalA\n(2.0µM)',
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% ctFieldThickness # LOWVALUES == 500 !

# Define
df = MecaData_DrugV3
dates = ['23-07-17', '23-07-20', '23-09-06']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
parameter = 'ctFieldThickness'
figname = 'ctFieldThickness_cutoff600' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['ctFieldThickness'] < 500),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'calyculinA & 0.25', 'calyculinA & 0.5', 'calyculinA & 1.0', 'calyculinA & 2.0']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[1], bp[2], bp[3]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'calyculinA & 0.25' : 'CalA\n(0.25µM)',
      'calyculinA & 0.5' : 'CalA\n(0.5µM)',
      'calyculinA & 1.0' : 'CalA\n(1.0µM)',
      'calyculinA & 2.0' : 'CalA\n(2.0µM)',
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% E400 - Test Weighted avg

#### Test Weighted avg

# Define
df = MecaData_DrugV3
dates = ['23-07-17', '23-07-20', '23-09-06']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
parameter = 'E_f_<_400'
figname = 'E400_TestAverageMethods' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
            (df['E_f_<_400'] <= 1e6),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'calyculinA & 0.25', 'calyculinA & 0.5', 'calyculinA & 1.0', 'calyculinA & 2.0']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[1], bp[2], bp[3]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fgw1 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^1')
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, axes = plt.subplots(1,3, figsize=(12,4))
for ax in axes:
    ax.set_yscale('log')
    
fig, axes[0] = D1Plot(df_fg, fig = fig, ax = axes[0], condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
fig, axes[1] = D1Plot(df_fgw1, fig = fig, ax = axes[1], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
fig, axes[2] = D1Plot(df_fgw2, fig = fig, ax = axes[2], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'calyculinA & 0.25' : 'CalA\n0.25µM',
      'calyculinA & 0.5' : 'CalA\n0.5µM',
      'calyculinA & 1.0' : 'CalA\n1.0µM',
      'calyculinA & 2.0' : 'CalA\n2.0µM',
      'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
      'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

titles = ['Simple Avg', 'Weighted Avg pow 1', 'Weighted Avg pow 2']
for i in range(len(axes)):
    ax = axes[i]
    t = titles[i]
    renameAxes(ax, rD, format_xticks = True)
    renameAxes(ax, renameDict, format_xticks = True)
    # renameLegend(ax, renameDict)
    ax.grid(visible=True, which='major', axis='y')
    ax.set_title(t)
    ax.set_xlabel('')
    if i >= 1:
        ax.set_ylabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


#### E400

# Define
figname = 'E400' + drugSuffix


# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
ax.set_yscale('log')

fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')




# %%% Fluctuations

#### All
# Define
df = MecaData_DrugV3
dates = ['23-07-17', '23-07-20', '23-09-06']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
figname = 'fluctuations_all' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df[XCol] > 20),
            (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'calyculinA & 0.25', 'calyculinA & 0.5', 'calyculinA & 1.0', 'calyculinA & 2.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(8,6))
    
fig, ax = D2Plot_wFit(df_fg, fig = fig, ax = ax, 
                XCol=XCol, YCol=YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'calyculinA & 0.25' : 'CalA (0.25µM)',
      'calyculinA & 0.5' : 'CalA (0.5µM)',
      'calyculinA & 1.0' : 'CalA (1.0µM)',
      'calyculinA & 2.0' : 'CalA (2.0µM)',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)

ax.set_xlim([0, 600])
ax.set_ylim([0, 400])

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### 1 by 1
# Define
df = MecaData_DrugV3
dates = ['23-07-17', '23-07-20', '23-09-06']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
figname = 'fluctuations_1by1' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df[XCol] > 20),
           # (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'calyculinA & 0.25', 'calyculinA & 0.5', 'calyculinA & 1.0', 'calyculinA & 2.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,5, figsize=(18,5))

for i in range(len(axes)):
    ax = axes[i]
    fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
                    XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
                    modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                    figSizeFactor = 1, markersizeFactor = 1)
    if i >= 1:
        ax.set_ylabel('')
           
# Prettify
rD = {'dmso & 0.0 ; ' : 'DMSO\n',
      'calyculinA & 0.25 ; ' : 'CalA (0.25µM)\n',
      'calyculinA & 0.5 ; ' : 'CalA (0.5µM)\n',
      'calyculinA & 1.0 ; ' : 'CalA (1.0µM)\n',
      'calyculinA & 2.0 ; ' : 'CalA (2.0µM)\n',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD, loc='upper left')
    ax.set_xlim([0, 600])
    ax.set_ylim([0, 400])
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%% Non-linearity

# %%%% 1. Global

# Define
df = MecaData_DrugV3
dates = ['23-07-17', '23-07-20', '23-09-06']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = 'wholeCurve'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'calyculinA & 0.25', 'calyculinA & 0.5', 'calyculinA & 1.0', 'calyculinA & 2.0']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))
    
plotPopKS_V2(df_f, fig = fig, ax = axes[0], condition = condCol, co_order = co_order,
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
plotPopKS_V2(df_f, fig = fig, ax = axes[1], condition = condCol, co_order = co_order, 
                 fitType = 'stressGaussian', fitWidth=75,
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'calyculinA & 0.25' : 'CalA (0.25µM)',
      'calyculinA & 0.5' : 'CalA (0.5µM)',
      'calyculinA & 1.0' : 'CalA (1.0µM)',
      'calyculinA & 2.0' : 'CalA (2.0µM)',
      }


for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)

ax = axes[0]
ax.set_xlim(0, 1200)
ax.set_ylim(0, 20)
ax = axes[1]
ax.set_xlim(0, 600)
ax.set_ylim(0, 8)

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% 2. Local

#### 200_500
# Define
df = MecaData_DrugV3
dates = ['23-07-17', '23-07-20', '23-09-06']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = '200_500'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'calyculinA & 0.25', 'calyculinA & 0.5', 'calyculinA & 1.0', 'calyculinA & 2.0']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,6))
    
plotPopKS_V2(df_f, fig = fig, ax = ax, condition = condCol, co_order = co_order,  
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'calyculinA & 0.25' : 'CalA (0.25µM)',
      'calyculinA & 0.5' : 'CalA (0.5µM)',
      'calyculinA & 1.0' : 'CalA (1.0µM)',
      'calyculinA & 2.0' : 'CalA (2.0µM)',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)
ax.set_ylim(0, 15)

ax.set_title(f"Stress Range = {mode.split('_')[0]} - {mode.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### 400_700
# Define
df = MecaData_DrugV3
dates = ['23-07-17', '23-07-20', '23-09-06']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = '400_700'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'calyculinA & 0.25', 'calyculinA & 0.5', 'calyculinA & 1.0', 'calyculinA & 2.0']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,6))
    
plotPopKS_V2(df_f, fig = fig, ax = ax, condition = condCol, co_order = co_order,  
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'calyculinA & 0.25' : 'CalA (0.25µM)',
      'calyculinA & 0.5' : 'CalA (0.5µM)',
      'calyculinA & 1.0' : 'CalA (1.0µM)',
      'calyculinA & 2.0' : 'CalA (2.0µM)',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)
ax.set_ylim(0, 15)

ax.set_title(f"Stress Range = {mode.split('_')[0]} - {mode.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%% Thickness & Stiffness

# %%%% 1. E400

# Define
df = MecaData_DrugV3
dates = ['23-07-17', '23-07-20', '23-09-06']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'bestH0'
YCol = 'E_f_<_400'
figname = 'H0vsE400' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           # (df['drug'] == 'dmso'),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'calyculinA & 0.25', 'calyculinA & 0.5', 'calyculinA & 1.0', 'calyculinA & 2.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

# LinLog
ax = axes[0]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
# LogLog
# ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'calyculinA & 0.25' : 'CalA (0.25µM)',
      'calyculinA & 0.5' : 'CalA (0.5µM)',
      'calyculinA & 1.0' : 'CalA (1.0µM)',
      'calyculinA & 2.0' : 'CalA (2.0µM)',
      'E_f_<_400_wAvg':'Elastic modulus (Pa)\nfor F < 400pN - wAvg',
      # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    # ax.set_ylim(900, 12000)

# axes[0].set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% 2. fit_K_250_100

# Define
df = MecaData_DrugV3
dates = ['23-07-17', '23-07-20', '23-09-06']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'bestH0'
YCol = 'fit_K'
fitID='250_100'
figname = 'H0vsfit_K' + fitID + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'calyculinA & 0.25', 'calyculinA & 0.5', 'calyculinA & 1.0', 'calyculinA & 2.0']

# Merge with extra data
df_ff = taka2.getFitsInTable(df_f, fitType='stressGaussian', filter_fitID=fitID)

# Filter again
Filters = [(df_ff['fit_K'] >= 0), 
           (df_ff['fit_K'] <= 1e5), 
           ]
df_ff = filterDf(df_ff, Filters)

# Group by
df_fg = dataGroup(df_ff, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_ffggw2 = dataGroup_weightedAverage(df_ff, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'fit_K', weightCol = 'fit_ciwK', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_ffggw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

ax = axes[0]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'calyculinA & 0.25' : 'CalA (0.25µM)',
      'calyculinA & 0.5' : 'CalA (0.5µM)',
      'calyculinA & 1.0' : 'CalA (1.0µM)',
      'calyculinA & 2.0' : 'CalA (2.0µM)',
      'fit_K':'Tangeantial Modulus (Pa)',
      'fit_K_wAvg':'Tangeantial Modulus (Pa) - wAvg',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    ax.set_ylim(500, 20000)
    
# axes[0].set_xlabel('')
fig.suptitle(f"Fit on stress = {fitID.split('_')[0]}+/-{fitID.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()


# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')















# %% Plots LimKi

# %%% Plots LimKi -- Batch 01

figSubDir = 'DrugSummary_LIMKi'
drugSuffix = '_LIMKi'

# %%%% bestH0

# Define
df = MecaData_DrugV3
dates = ['24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
parameter = 'bestH0'
figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           # (df['bestH0'] > 50),
            # (df['bestH0'] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'Y27 & 50.0', 'LIMKi & 10.0', 'LIMKi & 20.0']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[1], bp[2], bp[4], bp[5]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27\n(50µM)',
      'LIMKi & 10.0' : 'LIMKi\n(10µM)',
      'LIMKi & 20.0' : 'LIMKi\n(20µM)',
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% ctFieldThickness

# Define
df = MecaData_DrugV3
dates = ['24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
parameter = 'ctFieldThickness'
figname = 'ctFieldThickness' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           # (df['ctFieldThickness'] < 500),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'Y27 & 50.0', 'LIMKi & 10.0', 'LIMKi & 20.0']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[1], bp[2], bp[4], bp[5]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27\n(50µM)',
      'LIMKi & 10.0' : 'LIMKi\n(10µM)',
      'LIMKi & 20.0' : 'LIMKi\n(20µM)',
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% E400 - Test Weighted avg

#### Test Weighted avg

# Define
df = MecaData_DrugV3
dates = ['24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
parameter = 'E_f_<_400'
figname = 'E400_TestAverageMethods' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
            (df['E_f_<_400'] <= 1e6),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'Y27 & 50.0', 'LIMKi & 10.0', 'LIMKi & 20.0']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[1], bp[2], bp[4], bp[5]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fgw1 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^1')
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, axes = plt.subplots(1,3, figsize=(12,4))
for ax in axes:
    ax.set_yscale('log')
    
fig, axes[0] = D1Plot(df_fg, fig = fig, ax = axes[0], condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
fig, axes[1] = D1Plot(df_fgw1, fig = fig, ax = axes[1], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
fig, axes[2] = D1Plot(df_fgw2, fig = fig, ax = axes[2], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27\n(50µM)',
      'LIMKi & 10.0' : 'LIMKi\n(10µM)',
      'LIMKi & 20.0' : 'LIMKi\n(20µM)',
      'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
      'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

titles = ['Simple Avg', 'Weighted Avg pow 1', 'Weighted Avg pow 2']
for i in range(len(axes)):
    ax = axes[i]
    t = titles[i]
    renameAxes(ax, rD, format_xticks = True)
    renameAxes(ax, renameDict, format_xticks = True)
    # renameLegend(ax, renameDict)
    ax.grid(visible=True, which='major', axis='y')
    ax.set_title(t)
    ax.set_xlabel('')
    if i >= 1:
        ax.set_ylabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


#### E400

# Define
figname = 'E400' + drugSuffix


# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
ax.set_yscale('log')

fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')




# %%%% Fluctuations

#### All
# Define
df = MecaData_DrugV3
dates = ['24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
figname = 'fluctuations_all' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df[XCol] > 20),
            (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'Y27 & 50.0', 'LIMKi & 10.0', 'LIMKi & 20.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(8,6))
    
fig, ax = D2Plot_wFit(df_fg, fig = fig, ax = ax, 
                XCol=XCol, YCol=YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27 (50µM)',
      'LIMKi & 10.0' : 'LIMKi (10µM)',
      'LIMKi & 20.0' : 'LIMKi (20µM)',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)

ax.set_xlim([0, 700])
ax.set_ylim([0, 400])

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### 1 by 1
# Define
df = MecaData_DrugV3
dates = ['24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
figname = 'fluctuations_1by1' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df[XCol] > 20),
           # (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'Y27 & 50.0', 'LIMKi & 10.0', 'LIMKi & 20.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,4, figsize=(18,5))

for i in range(len(axes)):
    ax = axes[i]
    fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
                    XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
                    modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                    figSizeFactor = 1, markersizeFactor = 1)
    if i >= 1:
        ax.set_ylabel('')
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27 (50µM)',
      'LIMKi & 10.0' : 'LIMKi (10µM)',
      'LIMKi & 20.0' : 'LIMKi (20µM)',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD, loc='upper left')
    ax.set_xlim([0, 700])
    ax.set_ylim([0, 400])
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% Non-linearity

# %%%%% 1. Global

# Define
df = MecaData_DrugV3
dates = ['24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = 'wholeCurve'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'Y27 & 50.0', 'LIMKi & 10.0', 'LIMKi & 20.0']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))
    
plotPopKS_V2(df_f, fig = fig, ax = axes[0], condition = condCol, co_order = co_order,
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
plotPopKS_V2(df_f, fig = fig, ax = axes[1], condition = condCol, co_order = co_order, 
                 fitType = 'stressGaussian', fitWidth=75,
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27 (50µM)',
      'LIMKi & 10.0' : 'LIMKi (10µM)',
      'LIMKi & 20.0' : 'LIMKi (20µM)',
      }


for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)

ax = axes[0]
ax.set_xlim(0, 1200)
ax.set_ylim(0, 24)
ax = axes[1]
ax.set_xlim(0, 600)
ax.set_ylim(0, 10)

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%%% 2. Local

#### 200_500
# Define
df = MecaData_DrugV3
dates = ['24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = '200_500'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'Y27 & 50.0', 'LIMKi & 10.0', 'LIMKi & 20.0']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,6))
    
plotPopKS_V2(df_f, fig = fig, ax = ax, condition = condCol, co_order = co_order,  
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27 (50µM)',
      'LIMKi & 10.0' : 'LIMKi (10µM)',
      'LIMKi & 20.0' : 'LIMKi (20µM)',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)
ax.set_ylim(0, 18)

ax.set_title(f"Stress Range = {mode.split('_')[0]} - {mode.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### 400_700
# Define
df = MecaData_DrugV3
dates = ['24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = '400_700'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'Y27 & 50.0', 'LIMKi & 10.0', 'LIMKi & 20.0']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,6))
    
plotPopKS_V2(df_f, fig = fig, ax = ax, condition = condCol, co_order = co_order,  
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27 (50µM)',
      'LIMKi & 10.0' : 'LIMKi (10µM)',
      'LIMKi & 20.0' : 'LIMKi (20µM)',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)
ax.set_ylim(0, 18)

ax.set_title(f"Stress Range = {mode.split('_')[0]} - {mode.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% Thickness & Stiffness

# %%%%% 1. E400

# Define
df = MecaData_DrugV3
dates = ['24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'bestH0'
YCol = 'E_f_<_400'
figname = 'H0vsE400' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           # (df['drug'] == 'dmso'),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'Y27 & 50.0', 'LIMKi & 10.0', 'LIMKi & 20.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

# LinLog
ax = axes[0]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
# LogLog
# ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27 (50µM)',
      'LIMKi & 10.0' : 'LIMKi (10µM)',
      'LIMKi & 20.0' : 'LIMKi (20µM)',
      'E_f_<_400_wAvg':'Elastic modulus (Pa)\nfor F < 400pN - wAvg',
      # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    # ax.set_ylim(900, 12000)

# axes[0].set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%%% 2. fit_K_400_100

# Define
df = MecaData_DrugV3
dates = ['24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'bestH0'
YCol = 'fit_K'
fitID='400_100'
figname = 'H0vsfit_K' + fitID + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'Y27 & 50.0', 'LIMKi & 10.0', 'LIMKi & 20.0']

# Merge with extra data
df_ff = taka2.getFitsInTable(df_f, fitType='stressGaussian', filter_fitID=fitID)

# Filter again
Filters = [(df_ff['fit_K'] >= 0), 
           (df_ff['fit_K'] <= 1e5), 
           ]
df_ff = filterDf(df_ff, Filters)

# Group by
df_fg = dataGroup(df_ff, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_ffggw2 = dataGroup_weightedAverage(df_ff, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'fit_K', weightCol = 'fit_ciwK', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_ffggw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

ax = axes[0]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27 (50µM)',
      'LIMKi & 10.0' : 'LIMKi (10µM)',
      'LIMKi & 20.0' : 'LIMKi (20µM)',
      'fit_K':'Tangeantial Modulus (Pa)',
      'fit_K_wAvg':'Tangeantial Modulus (Pa) - wAvg',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    ax.set_ylim(500, 20000)
    
# axes[0].set_xlabel('')
fig.suptitle(f"Fit on stress = {fitID.split('_')[0]}+/-{fitID.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()


# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% Little check : compare to the Y27 experiments

drugSuffix = '_Y27vsY27'

# %%%%% bestH0

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['date', 'drug', 'concentration'])
parameter = 'bestH0'
figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['none', 'dmso', 'Y27'])),
           # (df['bestH0'] > 50),
            # (df['bestH0'] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['23-03-09 & none & 0.0', '23-03-08 & Y27 & 50.0', '24-03-13 & dmso & 0.0', '24-03-13 & Y27 & 50.0']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[1], bp[4], bp[5]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'23-03-09 & none & 0.0':'(1) Ctrl\nno drug',
      '23-03-08 & Y27 & 50.0':'(1) Y27\n50µM',
      '24-03-13 & dmso & 0.0':'(2) Ctrl\nDMSO',
      '24-03-13 & Y27 & 50.0':'(2) Y27\n50µM',
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

fig.suptitle("(1) = March 2023; (2) = March 2024", fontsize=12)

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
fig.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%%% ctFieldThickness

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['date', 'drug', 'concentration'])
parameter = 'ctFieldThickness'
figname = 'ctFieldThickness' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['none', 'dmso', 'Y27'])),
           # (df['ctFieldThickness'] < 500),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['23-03-09 & none & 0.0', '23-03-08 & Y27 & 50.0', '24-03-13 & dmso & 0.0', '24-03-13 & Y27 & 50.0']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[1], bp[4], bp[5]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'23-03-09 & none & 0.0':'(1) Ctrl\nno drug',
      '23-03-08 & Y27 & 50.0':'(1) Y27\n50µM',
      '24-03-13 & dmso & 0.0':'(2) Ctrl\nDMSO',
      '24-03-13 & Y27 & 50.0':'(2) Y27\n50µM',
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%%% E400 - Test Weighted avg

#### Test Weighted avg

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['date', 'drug', 'concentration'])
parameter = 'E_f_<_400'
figname = 'E400_TestAverageMethods' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['E_f_<_400'] <= 1e6),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['none', 'dmso', 'Y27'])),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['23-03-09 & none & 0.0', '23-03-08 & Y27 & 50.0', '24-03-13 & dmso & 0.0', '24-03-13 & Y27 & 50.0']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[1], bp[4], bp[5]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fgw1 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^1')
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, axes = plt.subplots(1,3, figsize=(12,4))
for ax in axes:
    ax.set_yscale('log')
    
fig, axes[0] = D1Plot(df_fg, fig = fig, ax = axes[0], condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
fig, axes[1] = D1Plot(df_fgw1, fig = fig, ax = axes[1], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
fig, axes[2] = D1Plot(df_fgw2, fig = fig, ax = axes[2], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'23-03-09 & none & 0.0':'(1) Ctrl\nno drug',
      '23-03-08 & Y27 & 50.0':'(1) Y27\n50µM',
      '24-03-13 & dmso & 0.0':'(2) Ctrl\nDMSO',
      '24-03-13 & Y27 & 50.0':'(2) Y27\n50µM',
      'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
      'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

titles = ['Simple Avg', 'Weighted Avg pow 1', 'Weighted Avg pow 2']
for i in range(len(axes)):
    ax = axes[i]
    t = titles[i]
    renameAxes(ax, rD, format_xticks = True)
    renameAxes(ax, renameDict, format_xticks = True)
    # renameLegend(ax, renameDict)
    ax.grid(visible=True, which='major', axis='y')
    ax.set_title(t)
    ax.set_xlabel('')
    if i >= 1:
        ax.set_ylabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


#### E400

# Define
figname = 'E400' + drugSuffix


# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
ax.set_yscale('log')

fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')




# %%%%% Fluctuations

#### All
# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['date', 'drug', 'concentration'])
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
figname = 'fluctuations_all' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['none', 'dmso', 'Y27'])),
           (df[XCol] > 20),
            (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['23-03-09 & none & 0.0', '23-03-08 & Y27 & 50.0', '24-03-13 & dmso & 0.0', '24-03-13 & Y27 & 50.0']
Filters = [(df_f[condCol].apply(lambda x : x in co_order)),
           ]
df_f = filterDf(df_f, Filters)

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(8,6))
    
fig, ax = D2Plot_wFit(df_fg, fig = fig, ax = ax, 
                XCol=XCol, YCol=YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'23-03-09 & none & 0.0':'(1) Ctrl - no drug',
      '23-03-08 & Y27 & 50.0':'(1) Y27 - 50µM',
      '24-03-13 & dmso & 0.0':'(2) Ctrl - DMSO',
      '24-03-13 & Y27 & 50.0':'(2) Y27 - 50µM',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)

ax.set_xlim([0, 700])
ax.set_ylim([0, 400])

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### 1 by 1
# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['date', 'drug', 'concentration'])
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
figname = 'fluctuations_1by1' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['none', 'dmso', 'Y27'])),
           (df[XCol] > 20),
           # (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['23-03-09 & none & 0.0', '23-03-08 & Y27 & 50.0', '24-03-13 & dmso & 0.0', '24-03-13 & Y27 & 50.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,4, figsize=(18,5))

for i in range(len(axes)):
    ax = axes[i]
    fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
                    XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
                    modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                    figSizeFactor = 1, markersizeFactor = 1)
    if i >= 1:
        ax.set_ylabel('')
           
# Prettify
rD = {'23-03-09 & none & 0.0':'(1) Ctrl - no drug\n',
      '23-03-08 & Y27 & 50.0':'(1) Y27 - 50µM\n',
      '24-03-13 & dmso & 0.0':'(2) Ctrl - DMSO\n',
      '24-03-13 & Y27 & 50.0':'(2) Y27 - 50µM\n',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD, loc='upper left')
    ax.set_xlim([0, 700])
    ax.set_ylim([0, 400])
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')







# %%% Plots LimKi -- Batch 02

figSubDir = 'DrugSummary_LIMKi_02'
drugSuffix = '_LIMKi'
gs.set_manuscript_options_jv()

# %%%% bestH0

# Define
df = MecaData_DrugV3
# dates = ['24-03-13', '24-07-04']
dates = ['24-07-04']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
parameter = 'bestH0'
figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['compNum'] <= 5),
            (df['bestH0'] > 50),
            (df['bestH0'] < 1000),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0', 'LIMKi & 20.0', 'Y27 & 50.0', 'none & 0.0'] # , 'LIMKi & 10.0'
bp = [['dmso & 0.0', 'blebbistatin & 50.0'], ['blebbistatin & 50.0', 'blebbistatin & 100.0'], 
      ['dmso & 0.0', 'LIMKi & 20.0'], ['none & 0.0', 'Y27 & 50.0'],
      ['blebbistatin & 100.0', 'Y27 & 50.0'], ['LIMKi & 20.0', 'Y27 & 50.0']]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(10, 8))
fig, ax = D1Plot(df_f, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 3, figSizeFactor = 2, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27\n(50µM)',
      'LIMKi & 10.0' : 'LIMKi\n(10µM)',
      'LIMKi & 20.0' : 'LIMKi\n(20µM)',
      'blebbistatin & 50.0':'Blebbi\n(50µM)',
      'blebbistatin & 100.0':'Blebbi\n(100µM)',
      'none & 0.0':'No Drug'
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% ctFieldThickness

# Define
df = MecaData_DrugV3
# dates = ['24-03-13', '24-07-04']
dates = ['24-07-04']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
parameter = 'surroundingThickness'
figname = 'surroundingThickness' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           # (df['ctFieldThickness'] < 500),
           (df['compNum'] <= 5),
            (df['ctFieldThickness'] > 50),
            (df['ctFieldThickness'] < 1000),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0', 'LIMKi & 20.0', 'Y27 & 50.0', 'none & 0.0'] # , 'LIMKi & 10.0'
bp = [['dmso & 0.0', 'blebbistatin & 50.0'], ['blebbistatin & 50.0', 'blebbistatin & 100.0'], 
      ['dmso & 0.0', 'LIMKi & 20.0'], ['none & 0.0', 'Y27 & 50.0'],
      ['blebbistatin & 100.0', 'Y27 & 50.0'], ['LIMKi & 20.0', 'Y27 & 50.0']]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(10, 8))
fig, ax = D1Plot(df_f, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27\n(50µM)',
      'LIMKi & 10.0' : 'LIMKi\n(10µM)',
      'LIMKi & 20.0' : 'LIMKi\n(20µM)',
      'blebbistatin & 50.0':'Blebbi\n(50µM)',
      'blebbistatin & 100.0':'Blebbi\n(100µM)',
      'none & 0.0':'No Drug'
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% E400 - Test Weighted avg

#### Test Weighted avg

# Define
df = MecaData_DrugV3
dates = ['24-07-04']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
parameter = 'E_f_<_400'
figname = 'E400_TestAverageMethods' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
            (df['E_f_<_400'] <= 1e6),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0', 'LIMKi & 20.0', 'Y27 & 50.0', 'none & 0.0'] # , 'LIMKi & 10.0'
bp = [['dmso & 0.0', 'blebbistatin & 50.0'], ['blebbistatin & 50.0', 'blebbistatin & 100.0'], 
      ['dmso & 0.0', 'LIMKi & 20.0'], ['none & 0.0', 'Y27 & 50.0'],
      ['blebbistatin & 100.0', 'Y27 & 50.0'], ['LIMKi & 20.0', 'Y27 & 50.0']]

# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27\n(50µM)',
      'LIMKi & 10.0' : 'LIMKi\n(10µM)',
      'LIMKi & 20.0' : 'LIMKi\n(20µM)',
      'blebbistatin & 50.0':'Blebbi\n(50µM)',
      'blebbistatin & 100.0':'Blebbi\n(100µM)',
      'none & 0.0':'No Drug',
      'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
      'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }


#### E400 -- by comp

# Define
figname = 'E400_comps' + drugSuffix

Filters = [(df_f['valid_f_<_400'] == True), 
           ]
df_f = filterDf(df_f, Filters)


# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, ax = plt.subplots(1,1, figsize=(10, 8))
ax.set_yscale('log')

fig, ax = D1Plot(df_f, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


#### E400 -- by cell

# Define
figname = 'E400_cells' + drugSuffix


# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, ax = plt.subplots(1,1, figsize=(10, 8))
ax.set_yscale('log')

fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')




# %%%% Fluctuations

#### All
# Define
df = MecaData_DrugV3
# dates = ['24-03-13', '24-07-04']
dates = ['24-07-04']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
figname = 'fluctuations_all' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df[XCol] > 20),
           (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0', 'LIMKi & 20.0', 'Y27 & 50.0', 'none & 0.0'] # , 'LIMKi & 10.0'
bp = [['dmso & 0.0', 'blebbistatin & 50.0'], ['blebbistatin & 50.0', 'blebbistatin & 100.0'], 
      ['dmso & 0.0', 'LIMKi & 20.0'], ['none & 0.0', 'Y27 & 50.0'],
      ['blebbistatin & 100.0', 'Y27 & 50.0'], ['LIMKi & 20.0', 'Y27 & 50.0']]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(8,6))
    
fig, ax = D2Plot_wFit(df_fg, fig = fig, ax = ax, 
                XCol=XCol, YCol=YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27\n(50µM)',
      'LIMKi & 10.0' : 'LIMKi\n(10µM)',
      'LIMKi & 20.0' : 'LIMKi\n(20µM)',
      'blebbistatin & 50.0':'Blebbi\n(50µM)',
      'blebbistatin & 100.0':'Blebbi\n(100µM)',
      'none & 0.0':'No Drug',
      'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
      'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)

# ax.set_xlim([0, 700])
# ax.set_ylim([0, 400])

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### 1 by 1
# Define
df = MecaData_DrugV3
# dates = ['24-03-13', '24-07-04']
dates = ['24-07-04']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
figname = 'fluctuations_1by1' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df[XCol] > 20),
           # (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0', 'LIMKi & 20.0', 'Y27 & 50.0', 'none & 0.0'] # , 'LIMKi & 10.0'
bp = [['dmso & 0.0', 'blebbistatin & 50.0'], ['blebbistatin & 50.0', 'blebbistatin & 100.0'], 
      ['dmso & 0.0', 'LIMKi & 20.0'], ['none & 0.0', 'Y27 & 50.0'],
      ['blebbistatin & 100.0', 'Y27 & 50.0'], ['LIMKi & 20.0', 'Y27 & 50.0']]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,6, figsize=(18,5), sharey=True)

for i in range(len(axes)):
    ax = axes[i]
    fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
                    XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
                    modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                    figSizeFactor = 1, markersizeFactor = 1)
    if i >= 1:
        ax.set_ylabel('')
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27\n(50µM)',
      'LIMKi & 10.0' : 'LIMKi\n(10µM)',
      'LIMKi & 20.0' : 'LIMKi\n(20µM)',
      'blebbistatin & 50.0':'Blebbi\n(50µM)',
      'blebbistatin & 100.0':'Blebbi\n(100µM)',
      'none & 0.0':'No Drug',
      'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
      'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD, loc='upper left')
    # ax.set_xlim([0, 700])
    # ax.set_ylim([0, 400])
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% Non-linearity

# %%%%% 1. Global

# Define
df = MecaData_DrugV3
# dates = ['24-03-13', '24-07-04']
dates = ['24-07-04']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = 'wholeCurve'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0', 
            'LIMKi & 20.0', 'Y27 & 50.0', 'none & 0.0'] # , 'LIMKi & 10.0'

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))
    
plotPopKS_V2(df_f, fig = fig, ax = axes[0], condition = condCol, co_order = co_order,
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
plotPopKS_V2(df_f, fig = fig, ax = axes[1], condition = condCol, co_order = co_order, 
                 fitType = 'stressGaussian', fitWidth=75,
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27\n(50µM)',
      'LIMKi & 10.0' : 'LIMKi\n(10µM)',
      'LIMKi & 20.0' : 'LIMKi\n(20µM)',
      'blebbistatin & 50.0':'Blebbi\n(50µM)',
      'blebbistatin & 100.0':'Blebbi\n(100µM)',
      'none & 0.0':'No Drug',
      'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
      'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }


for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)

ax = axes[0]
ax.set_xlim(0, 1200)
ax.set_ylim(0, 24)
ax = axes[1]
ax.set_xlim(0, 600)
ax.set_ylim(0, 10)

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%%% 2. Local

#### 200_500
# Define
df = MecaData_DrugV3
# dates = ['24-03-13', '24-07-04']
dates = ['24-07-04']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = '200_500'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0', 
            'LIMKi & 20.0', 'Y27 & 50.0', 'none & 0.0'] # , 'LIMKi & 10.0'

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,6))
    
plotPopKS_V2(df_f, fig = fig, ax = ax, condition = condCol, co_order = co_order,  
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27\n(50µM)',
      'LIMKi & 10.0' : 'LIMKi\n(10µM)',
      'LIMKi & 20.0' : 'LIMKi\n(20µM)',
      'blebbistatin & 50.0':'Blebbi\n(50µM)',
      'blebbistatin & 100.0':'Blebbi\n(100µM)',
      'none & 0.0':'No Drug',
      'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
      'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)
ax.set_ylim(0, 18)

ax.set_title(f"Stress Range = {mode.split('_')[0]} - {mode.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### 400_700
# Define
df = MecaData_DrugV3
# dates = ['24-03-13', '24-07-04']
dates = ['24-07-04']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
mode = '400_700'
figname = 'nonLin_' + mode + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0', 
            'LIMKi & 20.0', 'Y27 & 50.0', 'none & 0.0'] # , 'LIMKi & 10.0'

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,6))
    
plotPopKS_V2(df_f, fig = fig, ax = ax, condition = condCol, co_order = co_order,  
                 fitType = 'stressGaussian', fitWidth=75, 
                  mode = mode, Sinf = 0, Ssup = np.Inf)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27\n(50µM)',
      'LIMKi & 10.0' : 'LIMKi\n(10µM)',
      'LIMKi & 20.0' : 'LIMKi\n(20µM)',
      'blebbistatin & 50.0':'Blebbi\n(50µM)',
      'blebbistatin & 100.0':'Blebbi\n(100µM)',
      'none & 0.0':'No Drug',
      'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
      'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)
ax.set_ylim(0, 18)

ax.set_title(f"Stress Range = {mode.split('_')[0]} - {mode.split('_')[1]} Pa")

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% Thickness & Stiffness

# %%%%% 1. E400

# Define
df = MecaData_DrugV3
# dates = ['24-03-13', '24-07-04']
dates = ['24-07-04']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
XCol = 'bestH0'
YCol = 'E_f_<_400'
figname = 'H0vsE400' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           # (df['drug'] == 'dmso'),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['dmso & 0.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0', 
            'LIMKi & 20.0', 'Y27 & 50.0', 'none & 0.0'] # , 'LIMKi & 10.0'

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))

# LinLog
ax = axes[0]
ax.set_xscale('log')
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
# LogLog
# ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = [],
#                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'Y27 & 50.0' : 'Y27\n(50µM)',
      'LIMKi & 10.0' : 'LIMKi\n(10µM)',
      'LIMKi & 20.0' : 'LIMKi\n(20µM)',
      'blebbistatin & 50.0':'Blebbi\n(50µM)',
      'blebbistatin & 100.0':'Blebbi\n(100µM)',
      'none & 0.0':'No Drug',
      'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
      'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    # ax.set_ylim(900, 12000)

# axes[0].set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%%% 2. fit_K_400_100

# # Define
# df = MecaData_DrugV3
# dates = ['24-03-13']
# substrate = '20um fibronectin discs'
# df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
# XCol = 'bestH0'
# YCol = 'fit_K'
# fitID='400_100'
# figname = 'H0vsfit_K' + fitID + drugSuffix

# # Filter
# Filters = [(df['validatedThickness'] == True), 
#            (df['substrate'] == substrate),
#            (df['date'].apply(lambda x : x in dates)),
#            (df[XCol] >= 50),
#            (df[XCol] <= 1000),
#            ]
# df_f = filterDf(df, Filters)

# # Order
# co_order = ['dmso & 0.0', 'Y27 & 50.0', 'LIMKi & 10.0', 'LIMKi & 20.0']

# # Merge with extra data
# df_ff = taka2.getFitsInTable(df_f, fitType='stressGaussian', filter_fitID=fitID)

# # Filter again
# Filters = [(df_ff['fit_K'] >= 0), 
#            (df_ff['fit_K'] <= 1e5), 
#            ]
# df_ff = filterDf(df_ff, Filters)

# # Group by
# df_fg = dataGroup(df_ff, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
# df_fg = df_fg[['bestH0']]
# df_ffggw2 = dataGroup_weightedAverage(df_ff, groupCol = 'cellID', idCols = [condCol], 
#                                       valCol = 'fit_K', weightCol = 'fit_ciwK', weight_method = 'ciw^2')
# df_plot = pd.merge(left=df_fg, right=df_ffggw2, on='cellID', how='inner')

# # Plot
# fig, axes = plt.subplots(1,2, figsize=(12,6))

# ax = axes[0]
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# ax = axes[1]
# ax.set_yscale('log')
# fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
#                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=A*exp(kx)', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)

# # ax = axes[0]
# # ax.set_xscale('log')
# # ax.set_yscale('log')
# # fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
# #                 XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
# #                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
# #                 figSizeFactor = 1, markersizeFactor = 1)

# # ax = axes[1]
# # ax.set_xscale('log')
# # ax.set_yscale('log')
# # fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
# #                 XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
# #                 modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
# #                 figSizeFactor = 1, markersizeFactor = 1)
           
# # Prettify
# rD = {'dmso & 0.0' : 'DMSO',
#       'Y27 & 50.0' : 'Y27 (50µM)',
#       'LIMKi & 10.0' : 'LIMKi (10µM)',
#       'LIMKi & 20.0' : 'LIMKi (20µM)',
#       'fit_K':'Tangeantial Modulus (Pa)',
#       'fit_K_wAvg':'Tangeantial Modulus (Pa) - wAvg',
#       }

# for ax in axes:
#     ax.grid(visible=True, which='major', axis='both')
#     renameAxes(ax, rD, format_xticks = False)
#     renameAxes(ax, renameDict, format_xticks = False)
#     renameLegend(ax, rD)
#     # ax.set_xlim(0, 400)
#     ax.set_ylim(500, 20000)
    
# # axes[0].set_xlabel('')
# fig.suptitle(f"Fit on stress = {fitID.split('_')[0]}+/-{fitID.split('_')[1]} Pa")

# # Count
# CountByCond, CountByCell = makeCountDf(df_f, condCol)

# # Show
# plt.tight_layout()
# plt.show()


# # Save
# ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
#                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% Little check : compare to the Y27 experiments

drugSuffix = '_Y27vsY27'

# %%%%% bestH0

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13', '24-07-04']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['date', 'drug', 'concentration'])
parameter = 'bestH0'
figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['none', 'dmso', 'Y27'])),
           # (df['bestH0'] > 50),
            # (df['bestH0'] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['23-03-09 & none & 0.0', '23-03-08 & Y27 & 50.0', '24-03-13 & dmso & 0.0', '24-03-13 & Y27 & 50.0',
            '24-07-04 & none & 0.0', '24-07-04 & dmso & 0.0', '24-07-04 & Y27 & 50.0']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[1], bp[7], bp[11], bp[-3], bp[-2], bp[-1]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(10, 8))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'23-03-09 & none & 0.0':'(1) Ctrl\nno drug',
      '23-03-08 & Y27 & 50.0':'(1) Y27\n50µM',
      '24-03-13 & dmso & 0.0':'(2) Ctrl\nDMSO',
      '24-03-13 & Y27 & 50.0':'(2) Y27\n50µM',
      '24-07-04 & none & 0.0':'(3) Ctrl\nno drug',
      '24-07-04 & dmso & 0.0':'(3) Ctrl\nDMSO',
      '24-07-04 & Y27 & 50.0':'(3) Y27\n50µM',
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

fig.suptitle("(1) = March 2023; (2) = March 2024; (3) = July 2024", fontsize=12)

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
fig.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%%% ctFieldThickness

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['date', 'drug', 'concentration'])
parameter = 'ctFieldThickness'
figname = 'ctFieldThickness' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['none', 'dmso', 'Y27'])),
           # (df['ctFieldThickness'] < 500),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['23-03-09 & none & 0.0', '23-03-08 & Y27 & 50.0', '24-03-13 & dmso & 0.0', '24-03-13 & Y27 & 50.0']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[1], bp[4], bp[5]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'23-03-09 & none & 0.0':'(1) Ctrl\nno drug',
      '23-03-08 & Y27 & 50.0':'(1) Y27\n50µM',
      '24-03-13 & dmso & 0.0':'(2) Ctrl\nDMSO',
      '24-03-13 & Y27 & 50.0':'(2) Y27\n50µM',
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%%% E400 - Test Weighted avg

#### Test Weighted avg

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['date', 'drug', 'concentration'])
parameter = 'E_f_<_400'
figname = 'E400_TestAverageMethods' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['E_f_<_400'] <= 1e6),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['none', 'dmso', 'Y27'])),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['23-03-09 & none & 0.0', '23-03-08 & Y27 & 50.0', '24-03-13 & dmso & 0.0', '24-03-13 & Y27 & 50.0']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[1], bp[4], bp[5]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fgw1 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^1')
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, axes = plt.subplots(1,3, figsize=(12,4))
for ax in axes:
    ax.set_yscale('log')
    
fig, axes[0] = D1Plot(df_fg, fig = fig, ax = axes[0], condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
fig, axes[1] = D1Plot(df_fgw1, fig = fig, ax = axes[1], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
fig, axes[2] = D1Plot(df_fgw2, fig = fig, ax = axes[2], condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'23-03-09 & none & 0.0':'(1) Ctrl\nno drug',
      '23-03-08 & Y27 & 50.0':'(1) Y27\n50µM',
      '24-03-13 & dmso & 0.0':'(2) Ctrl\nDMSO',
      '24-03-13 & Y27 & 50.0':'(2) Y27\n50µM',
      'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
      'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

titles = ['Simple Avg', 'Weighted Avg pow 1', 'Weighted Avg pow 2']
for i in range(len(axes)):
    ax = axes[i]
    t = titles[i]
    renameAxes(ax, rD, format_xticks = True)
    renameAxes(ax, renameDict, format_xticks = True)
    # renameLegend(ax, renameDict)
    ax.grid(visible=True, which='major', axis='y')
    ax.set_title(t)
    ax.set_xlabel('')
    if i >= 1:
        ax.set_ylabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


#### E400

# Define
figname = 'E400' + drugSuffix


# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
ax.set_yscale('log')

fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')




# %%%%% Fluctuations

#### All
# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['date', 'drug', 'concentration'])
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
figname = 'fluctuations_all' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['none', 'dmso', 'Y27'])),
           (df[XCol] > 20),
            (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['23-03-09 & none & 0.0', '23-03-08 & Y27 & 50.0', '24-03-13 & dmso & 0.0', '24-03-13 & Y27 & 50.0']
Filters = [(df_f[condCol].apply(lambda x : x in co_order)),
           ]
df_f = filterDf(df_f, Filters)

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(8,6))
    
fig, ax = D2Plot_wFit(df_fg, fig = fig, ax = ax, 
                XCol=XCol, YCol=YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1)
           
# Prettify
rD = {'23-03-09 & none & 0.0':'(1) Ctrl - no drug',
      '23-03-08 & Y27 & 50.0':'(1) Y27 - 50µM',
      '24-03-13 & dmso & 0.0':'(2) Ctrl - DMSO',
      '24-03-13 & Y27 & 50.0':'(2) Y27 - 50µM',
      }

ax.grid(visible=True, which='major', axis='both')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)

ax.set_xlim([0, 700])
ax.set_ylim([0, 400])

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### 1 by 1
# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['date', 'drug', 'concentration'])
XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
figname = 'fluctuations_1by1' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['none', 'dmso', 'Y27'])),
           (df[XCol] > 20),
           # (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['23-03-09 & none & 0.0', '23-03-08 & Y27 & 50.0', '24-03-13 & dmso & 0.0', '24-03-13 & Y27 & 50.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,4, figsize=(18,5))

for i in range(len(axes)):
    ax = axes[i]
    fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
                    XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
                    modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                    figSizeFactor = 1, markersizeFactor = 1)
    if i >= 1:
        ax.set_ylabel('')
           
# Prettify
rD = {'23-03-09 & none & 0.0':'(1) Ctrl - no drug\n',
      '23-03-08 & Y27 & 50.0':'(1) Y27 - 50µM\n',
      '24-03-13 & dmso & 0.0':'(2) Ctrl - DMSO\n',
      '24-03-13 & Y27 & 50.0':'(2) Y27 - 50µM\n',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD, loc='upper left')
    ax.set_xlim([0, 700])
    ax.set_ylim([0, 400])
    

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
               figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %% Other stuff


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
        p = getSnsPalette(co_order, styleDict)
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
