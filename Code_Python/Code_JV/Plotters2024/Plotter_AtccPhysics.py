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
import matplotlib.lines as mlines
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
from scipy import interpolate
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
from matplotlib.gridspec import GridSpec
from scipy.stats import f_oneway, shapiro
from scipy.optimize import curve_fit

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)

import GraphicStyles as gs
import UtilityFunctions as ufun
import TrackAnalyser as taka
import TrackAnalyser_V2 as taka2
import TrackAnalyser_V3 as taka3

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
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=gs.colorList40) 

#### Graphic options
gs.set_default_options_jv()

figDir = "D:/MagneticPincherData/Figures/PhysicsDataset"

cm_in = 2.52

# %% > Data import & export

# MecaData_DrugV3 = taka2.getMergedTable('MecaData_Drugs_V3')


# %% > Objects declaration

renameDict = {# Variables
               'SurroundingThickness': 'Median Thickness (nm)',
               'surroundingThickness': 'Median Thickness (nm)',
               'ctFieldThickness': 'Median Thickness (nm)',
               'ctFieldFluctuAmpli' : 'Thickness Fluctuations\n$D_9$-$D_1$ (nm)',
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
    else:
        newColName = cols[0]
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
    if not condition in ['compNum', 'cellID', 'manipID', 'date']:
        cols_count_df = ['compNum', 'cellID', 'manipID', 'date', condition]
        count_df = df[cols_count_df]
        groupByCell = count_df.groupby('cellID')
        d_agg = {'compNum':'count', condition:'first', 'date':'first', 'manipID':'first'}
        df_CountByCell = groupByCell.agg(d_agg).rename(columns={'compNum':'compCount'})
    
    else:
        cols_count_df = ['compNum', 'cellID', 'manipID', 'date']
        count_df = df[cols_count_df]
        groupByCell = count_df.groupby('cellID')
        d_agg = {'compNum':'count', 'date':'first', 'manipID':'first'}
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

def renameAxes(axes, rD, 
               format_xticks = True, rotation = 0):
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
                axes[i].set_xticklabels(newXticksList, rotation = rotation)
                
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
                                boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 4},
                                # boxprops={"color": color, "linewidth": 0.5},
                                whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 6},
                                capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 6})

            
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
                    eqnText += " ; p-val = {:.3f}".format(pval)
                    print("Y = {:.5} X + {:.5}".format(params[1], params[0]))
                    print("p-value on the 'a' coefficient: {:.4e}".format(pval))
                    fitX = np.linspace(np.min(X), np.max(X), 100)
                    fitY = params[1]*fitX + params[0]
                    ax.plot(fitX, fitY, '--', lw = '2', 
                            color = c, zorder = 6)

                elif modelType == 'y=A*exp(kx)':
                    params, results = my_fitting_fun(X, np.log(Y)) 
                    # Y=a*X+b ; params[0] = b,  params[1] = a
                    pval = results.pvalues[1] # pvalue on the param 'k'
                    eqnText += " ; Y = {:.1f}*exp({:.1f}*X)".format(params[0], params[1])
                    eqnText += " ; p-val = {:.3f}".format(pval)
                    print("Y = {:.5}*exp({:.5}*X)".format(np.exp(params[0]), params[1]))
                    print("p-value on the 'k' coefficient: {:.4e}".format(pval))
                    fitX = np.linspace(np.min(X), np.max(X), 100)
                    fitY = np.exp(params[0])*np.exp(params[1]*fitX)
                    ax.plot(fitX, fitY, '--', lw = '2', 
                            color = c, zorder = 6)
                    
                elif modelType == 'y=k*x^a':
                    posValues = ((X > 0) & (Y > 0))
                    X, Y = X[posValues], Y[posValues]
                    params, results = my_fitting_fun(np.log(X), np.log(Y)) 
                    # Y=a*X+b ; params[0] = b,  params[1] = a
                    k = np.exp(params[0])
                    a = params[1]
                    pval = results.pvalues[1] # pvalue on the param 'a'
                    eqnText += " ; Y = {:.1e} * X^{:.1f}".format(k, a)
                    eqnText += " ; p-val = {:.3f}".format(pval)
                    print("Y = {:.4e} * X^{:.4f}".format(k, a))
                    print("p-value on the 'a' coefficient: {:.4e}".format(pval))
                    fitX = np.linspace(np.min(X), np.max(X), 100)
                    fitY = k * fitX**a
                    ax.plot(fitX, fitY, '--', lw = '2', 
                            color = c, zorder = 6)
                
                print('Number of values : {:.0f}'.format(len(Y)))
                print('\n')
            
            labelText = cond
            if writeEqn:
                labelText += eqnText
            # if robust:
            #     labelText += ' (R)'

            ax.plot(X, Y, 
                    color = c, ls = '', 
                    marker = m, markersize = markersize, 
                    markeredgecolor='k', markeredgewidth = 0.5, 
                    label = labelText)
            
    ax.set_xlabel(XCol)
    ax.set_ylabel(YCol)
    ax.legend()
    
    return(fig, ax)



def plotPopKS_V2(data, fig = None, ax = None, 
                 condition = '', co_order = [], colorDict = {}, labelDict = {},
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
                                   condition:'first'})
    data_agg_cells = data_agg_cells.reset_index()
    data_agg_cells['K_wAvg'] = data_agg_cells['A']/data_agg_cells['weight']
    data_agg_cells = data_agg_cells.rename(columns = {'compNum' : 'compCount'})
    
    # 2nd selection
    data_agg_cells = data_agg_cells.drop(data_agg_cells[data_agg_cells['compCount'] <= 1].index)
    data_agg_cells = data_agg_cells.drop(data_agg_cells[data_agg_cells['weight'] <= 1].index)
    
    grouped2 = data_agg_cells.groupby(by=[condition, 'fit_center'])
    data_agg_all = grouped2.agg({'compCount':['sum', 'count'],
                                   'K_wAvg':['mean', 'std', 'median']})
    data_agg_all = data_agg_all.reset_index()
    data_agg_all.columns = ufun.flattenPandasIndex(data_agg_all.columns)
    data_agg_all = data_agg_all.rename(columns = {'compCount_sum' : 'compCount', 'compCount_count' : 'cellCount'})
    data_agg_all['K_wAvg_ste'] = data_agg_all['K_wAvg_std']/data_agg_all['cellCount']**0.5
    
    
    #### Plot
    i_color = 0
    for co in co_order:
        if co in colorDict.keys():
            pass
        else:
            try:
                colorDict[co] = styleDict[co]['color']
            except:
                colorDict[co] = gs.colorList10[i_color%10]
                i_color = i_color + 1
        
        if co in labelDict.keys():
            pass
        else:
            labelDict[co] = str(co)
            
    
    for co in co_order:
        
        color = colorDict[co]
        label_co = labelDict[co]
            
        df = data_agg_all[data_agg_all[condition] == co]
        
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
                    label = labelDict[co] + ' | ' + str(total_N) + ' cells' + ' | ' + str(total_n) + ' comp')
        
        ax.legend(loc = 'upper left', fontsize = 8)
        ax.set_xlabel('Stress (Pa)')
        ax.set_ylabel('K (kPa)')
        ax.grid(visible=True, which='major', axis='y')
                    
        # except:
        #     pass


    return(fig, ax)







def plotPopKS_V3(data, fig = None, ax = None, 
                 condition = '', co_order = [],
                 fitType = 'stressRegion', fitWidth=75,
                 mode = 'wholeCurve', scale = 'lin', Sinf = 0, Ssup = np.Inf):
    
    #### Init
    if ax == None:
        figHeight = 10/gs.cm_in
        figWidth = 17/gs.cm_in
        fig, ax = plt.subplots(1,1, figsize=(figWidth, figHeight))
        
    cl = matplotlib.colormaps['Set2'].colors[:-1] + matplotlib.colormaps['Set1'].colors[1:]
    ml = ['o', 'D', 'P', 'X', '^']
    
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
    data_fits = taka3.getFitsInTable(data, fitType=fitType, filter_fitID=fitId)
    
    # Filter the table
    data_fits = data_fits[(data_fits['fit_center'] >= Sinf) & (data_fits['fit_center'] <= Ssup)]   
    data_fits = data_fits.drop(data_fits[data_fits['fit_error'] == True].index)
    data_fits = data_fits.dropna(subset = ['fit_ciwK'])
    
    data_fits = data_fits.drop(data_fits[data_fits['fit_K'] < 0].index)
    data_fits = data_fits.drop(data_fits[data_fits['fit_K'] < data_fits['fit_ciwK']/2].index)
    
    
    
    # Compute the weights
    # data_fits['weight'] = (data_fits['fit_K']/data_fits['fit_ciwK'])**2
    
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
    # data_fits['A'] = data_fits['fit_K'] * data_fits['weight']
    # grouped1 = data_fits.groupby(by=['cellID', 'fit_center'])
    # data_agg_cells = grouped1.agg({'compNum':'count',
    #                                'A':'sum', 
    #                                'weight': 'sum',
    #                                condition:'first'})
    # data_agg_cells = data_agg_cells.reset_index()
    # data_agg_cells['K_wAvg'] = data_agg_cells['A']/data_agg_cells['weight']
    # data_agg_cells = data_agg_cells.rename(columns = {'compNum' : 'compCount'})
    
    # # 2nd selection
    # data_agg_cells = data_agg_cells.drop(data_agg_cells[data_agg_cells['compCount'] <= 1].index)
    # data_agg_cells = data_agg_cells.drop(data_agg_cells[data_agg_cells['weight'] <= 1].index)
    
    # grouped2 = data_agg_cells.groupby(by=[condition, 'fit_center'])
    # data_agg_all = grouped2.agg({'compCount':['sum', 'count'],
    #                                'K_wAvg':['mean', 'std', 'median']})
    # data_agg_all = data_agg_all.reset_index()
    # data_agg_all.columns = ufun.flattenPandasIndex(data_agg_all.columns)
    # data_agg_all = data_agg_all.rename(columns = {'compCount_sum' : 'compCount', 'compCount_count' : 'cellCount'})
    # data_agg_all['K_wAvg_ste'] = data_agg_all['K_wAvg_std']/data_agg_all['cellCount']**0.5
    
    
    data_fits['compID'] = data_fits['cellID'] + '_' + data_fits['compNum'].apply(lambda x : str(x))
    list_cellID = data_fits['cellID'].unique()
    Ncells = len(list_cellID)
    
    #### Plot
    i_color = 0
    
    for i, cid in enumerate(list_cellID):
        color = cl[i%len(cl)]
        dfcell = data_fits.loc[data_fits['cellID'] == cid]
        plot_parms = {'color':color, 'marker':'o', 'markersize':6, 'mec':'None',}
        ax.plot([], [], label = cid.split('_')[-1], color = plot_parms['color'], ls='-')
        
        for j in range(5):
            if (j+1) in dfcell['compNum'].values:
                marker = ml[j]
                plot_parms.update({'marker':marker, 'mec':'k', 'mew':0.5})
                errplot_parms = {**plot_parms, 'ecolor':color, 'elinewidth':1.5, 
                                 'capsize':6, 'capthick':1.5, 'alpha':0.9, 'zorder':3}
                dfcomp = dfcell.loc[dfcell['compNum'] == j+1]
    
                centers = dfcomp['fit_center'].values
                K = dfcomp['fit_K'].values
                Kciw = dfcomp['fit_ciwK'].values/2
    
                # ax.plot(centers, K/1000, **plot_parms)
                ax.errorbar(centers, K/1000, yerr = Kciw/1000, **errplot_parms)
                
    imax = i
    nskip = 5 - (imax%5 + 1)
    for k in range(nskip):
        ax.plot([], [], label=' ', c = 'None')
    
    for j in range(5):
        marker = ml[j]
        plot_parms = {'color':'gray', 'marker':marker, 'markersize':6, 'mec':'k', 'mew':0.5}
        ax.plot([], [], label = f'Comp n°{j+1:.0f}', **plot_parms)
        
    ax.legend(loc = 'upper left', fontsize = 8, ncols=3)
    ax.set_xlabel('Stress (Pa)')
    ax.set_ylabel('K (kPa)')
    ax.grid(visible=True, which='major', axis='y')

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
        data_co = data[data[condition] == co]
        color = colorList[i]
        ax.scatter(data_co['bestH0'], data_co['minStress'], marker = 'o', s = 3, color = color, edgecolor = 'k', zorder=4)
        ax.scatter(data_co['bestH0'], data_co['maxStress'], marker = 'o', s = 3, color = color, edgecolor = 'k', zorder=4)
        ax.vlines(data_co['bestH0'], data_co['minStress'], data_co['maxStress'], color = colorList[i], label = co)
        ax.set_xlim([0, 800])
        ax.set_ylim([0, 7500])

    out = fig, axes    
    
    return(out)


def StressRange_2D_V2(data, fig=None, ax=None, condition='', defaultMode = False):
        
    co_values = list(data[condition].unique())
    Nco = len(co_values)
    cl = matplotlib.colormaps['Set2'].colors[:-1] + matplotlib.colormaps['Set1'].colors[1:]
    
    # if not defaultColors:
    #     try:
    #         colorList, markerList = getStyleLists(co_values, styleDict)
    #     except:
    #         colorList, markerList = gs.colorList30, gs.markerList10
    # else:
    #     colorList, markerList = gs.colorList30, gs.markerList10

    if ax == None:
        figHeight = 10/gs.cm_in
        figWidth = 17/gs.cm_in
        fig, ax = plt.subplots(1,1, figsize=(figWidth, figHeight))
    
    for i in range(Nco):      
        co = co_values[i]
        data_co = data[data[condition] == co]
        if defaultMode:
            color = gs.colorList40[19]
            alpha = 0.4
            zo = 4
            s = 4
            ec = 'None'
            labels = ['Minimum stress', 'Maximum stress', 'Compression']
        else:
            color = cl[i]
            alpha = 1
            zo = 6
            s = 10
            ec = color
            labels = ['', '', '']
            
            
        ax.scatter(data_co['bestH0'], data_co['minStress'], marker = 'o', s = s, color = 'deepskyblue', edgecolor = ec, zorder=zo, 
                   label = labels[0])
        ax.scatter(data_co['bestH0'], data_co['maxStress'], marker = 'o', s = s, color = 'darkred', edgecolor = ec, zorder=zo,
                   label = labels[1])
        ax.vlines(data_co['bestH0'], data_co['minStress'], data_co['maxStress'], color = color, alpha = alpha, zorder=zo-1,
                  label = labels[2])
        
    # fig.legend()
    
    out = fig, axes    
    
    return(out)

# %% > Anumita's awesome plots

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
        plt.text(xpos, ypos, "%.1f"%yval + '%', ha="center", va="center", color = '#000000', fontsize = 8)
    for xpos, ypos, yval in zip(xticks, y1+y2/2, y2):
        plt.text(xpos, ypos, "%.1f"%yval+ '%', ha="center", va="center", color = '#000000', fontsize = 8)
    for xpos, ypos, yval in zip(xticks, y1+y2+y3/2, y3):
        plt.text(xpos, ypos, "%.1f"%yval+ '%', ha="center", va="center", color = '#000000', fontsize = 8)
    # add text annotation corresponding to the "total" value of each bar
    for xpos, ypos, yval in zip(xticks, y1+y2+y3+0.5, N):
        plt.text(xpos, ypos, "N=%d"%yval, ha="center", va="bottom", fontsize = 8)

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
    
    ax.set_ylim([0, 110])
    plt.xticks(xticks, **plotChars)
    plt.yticks(**plotChars)
    plt.tight_layout()
    plt.legend(handles = patches, bbox_to_anchor=(1.01, 0.5), fontsize = 10, labelcolor='linecolor')
    plt.show()

    return(fig, ax, pvals)


def computeNLMetrics(GlobalTable, th_NLI = np.log10(2)):

    data_main = GlobalTable
    data_main['dateID'] = GlobalTable['date']
    data_main['manipId'] = GlobalTable['manipID']
    data_main['cellId'] = GlobalTable['cellID']
    data_main['dateCell'] = GlobalTable['date'] + '_' + GlobalTable['cellCode']

    nBins = 10
    bins = np.linspace(0, 1000, nBins)
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
    th_NLI = np.abs(th_NLI)
    for i in NLItypes:
        if i == 'linear':
            index = data_main[data_main['NLI'] < -th_NLI].index
            ID = 1
        elif i =='non-linear':
            index =  data_main[data_main['NLI'] > th_NLI].index
            ID = 0
        elif i =='intermediate':
            index = data_main[(data_main['NLI'] > -th_NLI) & (data_main['NLI'] < th_NLI)].index
            ID = 0.5
        for j in index:
            data_main.loc[index, 'NLI_Plot'][j] = i
            data_main.loc[index, 'NLI_Ind'][j] = ID
    
    return(data_main)


# %% > Data import & export
# %%% MecaData_Phy
# MecaData_Phy = taka3.getMergedTable('MecaData_Physics')
MecaData_Phy = taka3.getMergedTable('MecaData_Physics_V2')

# %%% MecaData_Phy
MecaData_Phy = MecaData_Phy.dropna(axis=0, subset='date')

# %%% Check content

print('Dates')
print([x for x in MecaData_Phy['date'].unique()])
print('')

print('Cell types')
print([x for x in MecaData_Phy['cell type'].unique()])
print('')

print('Cell subtypes')
print([x for x in MecaData_Phy['cell subtype'].unique()])
print('')

print('Drugs')
print([x for x in MecaData_Phy['drug'].unique()])
print('')

print('Substrates')
print([x for x in MecaData_Phy['substrate'].unique()])
print('')

print('Resting Fields')
print([x for x in MecaData_Phy['normal field'].unique()])
print('')


# %% Plots - Manuscrit

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
substrate = '20um fibronectin discs'
df, condCol = makeCompositeCol(df, cols=['drug', 'date'])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           ]
df_f = filterDf(df, Filters)

CountByCond, CountByCell = makeCountDf(df_f, condCol)


# %%% Figure NC1.1 - V2 - Thickness & Stiffness

gs.set_manuscript_options_jv(palette = 'Set2')

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x not in excluded_dates)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           ]

df_f = filterDf(df, Filters)
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Group By for H0
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0', 'ctFieldFluctuAmpli'], aggFun = 'mean')

# Group By for E<400
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_fgw2['E_f_<_400_wAvg'] /= 1000


#### Init fig
gs.set_manuscript_options_jv()
fig, axes = plt.subplots(2, 2, figsize=(17/cm_in, 17/cm_in))
color = styleDict['dmso']['color']

#### 01 - Best H0
ax = axes[0,0]

# Order
# co_order = ['none', 'dmso']

# Plot
# fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter='bestH0',
#                  co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 0.6,
#                  stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
#                  showMean = False)
ax.set_title('Thickness')
ax.set_xlabel('Fitted $H_0$ (nm)')
ax.set_ylabel('Count (cells)')
N, bins, patches = ax.hist(x=df_fg['bestH0'].values, bins = 16, color = color)
ax.set_xlim([0, ax.get_xlim()[1]])
medianH0 = np.median(df_fg['bestH0'].values)
ax.axvline(medianH0, c='darkred', label = f'Median $H_0$ = {medianH0:.0f} nm')
ax.legend()


#### 02 - E_400
ax = axes[0,1]

# Order
# co_order = ['none', 'dmso']

# Plot
# fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter='E_f_<_400_wAvg',
#                  co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 0.6,
#                  stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
#                  showMean = False)

ax.set_title('Stiffness')
ax.set_xlabel('$E_{400}$ (kPa)')
ax.set_ylabel('Count (cells)')
N, bins, patches = ax.hist(x=df_fgw2['E_f_<_400_wAvg'].values, bins = 16, color = color)
ax.set_xlim([0, ax.get_xlim()[1]])
medianE400 = np.median(df_fgw2['E_f_<_400_wAvg'].values)
ax.axvline(medianE400, c='darkred', label = 'Median $E_{400}$ = ' + f'{medianE400:.2f} kPa')
ax.legend()
    
# Prettify
rD = {'none' : 'No drug',
      'dmso' : 'DMSO', 
      'bestH0' : 'Fitted $H_0$ (nm)',
      'E_f_<_400_wAvg' : '$E_{400}$ (kPa)'
      }

for ax in axes[0,:]:
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, renameDict)
    # ax.grid(visible=True, which='major', axis='y')
    
    

#### Part 2

#### 03 - Normality test H0

ax = axes[1,0]

data_lin = df_fg[df_fg['drug'] == 'dmso']['bestH0'].values
data_log = np.log(df_fg[df_fg['drug'] == 'dmso']['bestH0'].values)

data=data_lin
shap_stat, shap_pval = shapiro(data)
sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = gs.cL_Set21[5], markeredgecolor = 'None', markersize=6)
ax.plot([], [], label=f'Normal distribution: {shap_pval:.2f}', ls='', marker='o', 
        markerfacecolor = gs.cL_Set21[5], markeredgecolor = 'None', markersize=4)
ax.grid()
ax.set_title('Q-Q plots for $H_0$')


#### 04 - normality test E_400

ax = axes[1,1]

data_lin = df_fgw2[df_fgw2['drug'] == 'dmso']['E_f_<_400_wAvg'].values
data_log = np.log(df_fgw2[df_fgw2['drug'] == 'dmso']['E_f_<_400_wAvg'].values)

data=data_lin
shap_stat, shap_pval = shapiro(data)
sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = gs.cL_Set21[5], markeredgecolor = 'None', markersize=6)
ax.plot([], [], label=f'Normal distribution: {shap_pval:.2f}', ls='', marker='o', 
        markerfacecolor = gs.cL_Set21[5], markeredgecolor = 'None', markersize=4)
ax.grid()
ax.set_title('Q-Q plots for $E_{400}$')

#### Part 3

#### 05 - Log-Normality test H0

ax = axes[1,0]

data_lin = df_fg[df_fg['drug'] == 'dmso']['bestH0'].values
data_log = np.log(df_fg[df_fg['drug'] == 'dmso']['bestH0'].values)

data=data_log
shap_stat, shap_pval = shapiro(data)
ax.axline((0, 0), slope=1, color="k", linestyle='--', linewidth=1, zorder=6)
ax.set_aspect('equal')
ax.set_xlim([-3.5,3.5])
ax.set_ylim([-3.5,3.5])
sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = gs.cL_Set21[0], markeredgecolor = 'None', markersize=6)
ax.plot([], [], label=f'Log-normal distribution: {shap_pval:.2f}', ls='', marker='o', 
        markerfacecolor = gs.cL_Set21[0], markeredgecolor = 'None', markersize=4)
ax.legend(fontsize=7, title_fontsize=7, title = 'Shapiro–Wilk p-value', loc='lower right')
ax.grid()


#### 06 - Log-normality test E_400

ax = axes[1,1]

data_lin = df_fgw2[df_fgw2['drug'] == 'dmso']['E_f_<_400_wAvg'].values
data_log = np.log(df_fgw2[df_fgw2['drug'] == 'dmso']['E_f_<_400_wAvg'].values)

data=data_log
shap_stat, shap_pval = shapiro(data)
ax.axline((0, 0), slope=1, color="k", linestyle='--', linewidth=1, zorder=6)
ax.set_aspect('equal')
ax.set_xlim([-3.5,3.5])
ax.set_ylim([-3.5,3.5])
sm.qqplot(data, fit=True, line=None, ax=ax, markerfacecolor = gs.cL_Set21[0], markeredgecolor = 'None', markersize=6)
ax.plot([], [], label=f'Log-normal distribution: {shap_pval:.2f}', ls='', marker='o', 
        markerfacecolor = gs.cL_Set21[0], markeredgecolor = 'None', markersize=4)
ax.legend(fontsize=7, title_fontsize=7, title = 'Shapiro–Wilk p-value', loc='lower right')
ax.grid()

# Show
plt.tight_layout()
plt.show()

name = 'Fig_NC1-1_V2'
figDir = "D:/MagneticPincherData/Figures/PhysicsDataset"
figSubDir = 'Manuscript'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = 'Manuscript', cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%% Figure NC1.2 - Fluctuations

gs.set_manuscript_options_jv(palette = 'Set2')

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug'])
# excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           # (df['date'].apply(lambda x : x not in excluded_dates)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           ]

df_f = filterDf(df, Filters)
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Group By for H0
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], 
                  numCols = ['bestH0', 'surroundingThickness', 'ctFieldFluctuAmpli'], aggFun = 'mean')

# Group By for E<400
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_fgw2['E_f_<_400_wAvg'] /= 1000


#### Init fig
fig, ax = plt.subplots(1, 1, figsize=(17/cm_in, 10/cm_in))

#### 03 - Fluctuations

Filters = [(df_fg['compNum'] >= 7), 
           ]
df_fg2 = filterDf(df_fg, Filters)


# Plot
XCol = 'surroundingThickness'
YCol = 'ctFieldFluctuAmpli'
    
fig, ax = D2Plot_wFit(df_fg2, fig = fig, ax = ax, 
                XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
                modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 1.3)
ax.set_xlim([0, 500])
ax.set_ylim([0, 400])
ax.grid()

# Prettify
rD = {'none' : 'No drug',
      'dmso' : 'DMSO', 
      'surroundingThickness' : '$H_{5mT}$ (nm)',
      'E_f_<_400_wAvg' : '$E_{400}$ (nN)'
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
renameLegend(ax, renameDict)


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

name = 'Fig_NC1-2_V2'
figSubDir = 'Manuscript'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = 'Manuscript', cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%% Figure NC2 - Cell Identity

# %%%% Tests

#### Define

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO'] # 
substrate = '20um fibronectin discs'
parameter = 'bestH0'
# parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug'])
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['bestH0'] < 1000),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
            (df['date'] == '23-03-09'),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = []

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
df_fg[parameter + '_pop_spread'] = df_fg[parameter] / np.mean(df_fg[parameter].values)

# Merge
df_m = pd.merge(left=df_f, right=df_fg, on = 'cellID', how='left', suffixes = (None, '_grouped'))
df_m[parameter + '_indiv_spread'] = df_m[parameter] / df_m[parameter + '_grouped']

# Sort by cellID
df_f.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_fg.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_m.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)


#### Apply log

dfLOG_f = filterDf(df, Filters)

dfLOG_f[parameter] = np.log10(dfLOG_f[parameter])
# df_f[parameter] = np.log(df_f[parameter])

# Group By
dfLOG_fg = dataGroup(dfLOG_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
dfLOG_fg[parameter + '_pop_spread'] = dfLOG_fg[parameter] / np.mean(dfLOG_fg[parameter].values)

# Merge
dfLOG_m = pd.merge(left=dfLOG_f, right=dfLOG_fg, on = 'cellID', how='left', suffixes = (None, '_grouped'))
dfLOG_m[parameter + '_indiv_spread'] = dfLOG_m[parameter] / dfLOG_m[parameter + '_grouped']

# Sort by cellID
dfLOG_f.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)
dfLOG_fg.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)
dfLOG_m.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)


#### Normality tests

data_lin = df_fg[parameter].values
data_log = dfLOG_fg[parameter].values

fig_test, axes_test = plt.subplots(1, 2, figsize = (12, 5))

data=data_lin
ax = axes_test[0]
shap_stat, shap_pval = shapiro(data)
sm.qqplot(data, fit=True, line='45', ax=ax)
ax.plot([], [], label=f'Shapiro–Wilk p-value = {shap_pval:.2f}', ls='', marker='o')
ax.legend(fontsize=11)
ax.set_title(f'Normality test and\nQ-Q plot of parm: {parameter}', fontsize=11)

data=data_log
ax = axes_test[1]
shap_stat, shap_pval = shapiro(data)
sm.qqplot(data, fit=True, line='45', ax=ax)
ax.plot([], [], label=f'Shapiro–Wilk p-value = {shap_pval:.2f}', ls='', marker='o')
ax.legend(fontsize=11)
ax.set_title(f'Log-normality test and\nQ-Q plot of parm: {parameter}', fontsize=11)

fig_test.tight_layout()
plt.show()


# # %%%% One way ANOVA
# list_cellIDs = df_f['cellID'].unique()
# values_per_cell = [df_f.loc[df_f['cellID'] == ID, parameter].values for ID in list_cellIDs]

# f_oneway(*values_per_cell)


#### Compute CV by cell and by pop

list_cellIDs = dfLOG_f['cellID'].unique()
values_per_cell = [dfLOG_f.loc[dfLOG_f['cellID'] == ID, parameter].values for ID in list_cellIDs]
CV_per_cell = [np.std(cell_vals) / np.mean(cell_vals) for cell_vals in values_per_cell if len(cell_vals) >= 4]
CV_per_cell_avg = np.mean(CV_per_cell)
CV_population = np.std(dfLOG_fg[parameter]) / np.mean(dfLOG_fg[parameter])

print('CV of all cells: ', *[f'{cv:.4f}' for cv in CV_per_cell])
print(f'Mean CV per cell: {CV_per_cell_avg:.4f}')
print(f'CV of full pop: {CV_population:.4f}')
print(f'N cells = {len(list_cellIDs):.0f}')



# %%%% Plots -- Thickness

gs.set_manuscript_options_jv(palette = 'Set2')

#### Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO'] # 
substrate = '20um fibronectin discs'
parameter = 'bestH0'
# parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug'])
df['cellCode'] = df['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['bestH0'] < 1000),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           (df['date'] == '23-03-09'),
           (df['cellCode'] != 'C1801'),
           ]
df_f = filterDf(df, Filters)
df_f['cellNum'] = df_f['cellCode'].apply(lambda x : int(x[1:]))

# Order
co_order = []

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
Manipe = df_f['manipID'].values[0]
Ncells = CountByCond['cellCount'].values[0]
Ncomps = CountByCond['compCount'].values[0]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol, 'cellCode', 'cellNum'], numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
df_fg[parameter + '_pop_spread'] = df_fg[parameter] / np.mean(df_fg[parameter].values)

# Merge
df_m = pd.merge(left=df_f, right=df_fg, on = 'cellID', how='left', suffixes = (None, '_grouped'))
df_m[parameter + '_indiv_spread'] = df_m[parameter] / df_m[parameter + '_grouped']

# Sort by cellID
df_f.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_fg.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_m.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)




#### Apply log

dfLOG_f = filterDf(df, Filters)
dfLOG_f['cellNum'] = dfLOG_f['cellCode'].apply(lambda x : int(x[1:]))

dfLOG_f[parameter] = np.log10(dfLOG_f[parameter])
# df_f[parameter] = np.log(df_f[parameter])

# Group By
dfLOG_fg = dataGroup(dfLOG_f, groupCol = 'cellID', idCols = [condCol, 'cellCode', 'cellNum'], 
                     numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
dfLOG_fg[parameter + '_pop_spread'] = dfLOG_fg[parameter] / np.mean(dfLOG_fg[parameter].values)

# Merge
dfLOG_m = pd.merge(left=dfLOG_f, right=dfLOG_fg, on = 'cellID', how='left', suffixes = (None, '_grouped'))
dfLOG_m[parameter + '_indiv_spread'] = dfLOG_m[parameter] / dfLOG_m[parameter + '_grouped']

# Sort by cellID
dfLOG_f.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)
dfLOG_fg.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)
dfLOG_m.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)

# df_m['cellCode'] = df_m['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])
# dfLOG_m['cellCode'] = dfLOG_m['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])





#### Compute CV
list_cellIDs = dfLOG_f['cellID'].unique()
values_per_cell = [dfLOG_f.loc[dfLOG_f['cellID'] == ID, parameter].values for ID in list_cellIDs]
CV_per_cell = [np.std(cell_vals) / np.mean(cell_vals) for cell_vals in values_per_cell if len(cell_vals) >= 4]
CV_per_cell_avg = np.mean(CV_per_cell)
CV_population = np.std(dfLOG_fg[parameter]) / np.mean(dfLOG_fg[parameter])

print('CV of all cells: ', *[f'{cv:.4f}' for cv in CV_per_cell])
print(f'Mean CV per cell: {CV_per_cell_avg:.4f}')
print(f'CV of full pop: {CV_population:.4f}')
print(f'N cells = {len(list_cellIDs):.0f}')




#### Start plot
fig = plt.figure(figsize=(17/cm_in, 18/cm_in))
spec = fig.add_gridspec(3, 1)



#### Plot 1
ax = fig.add_subplot(spec[0])
ax.set_ylim([60, 1000])
ax.set_yscale('log')

sns.swarmplot(ax = ax, data = df_m, x = 'cellCode', y = parameter, hue = 'cellCode', 
              s=4, edgecolor='w', linewidth=0)
# LegendMark = mlines.Line2D([], [], color='gray', ls='', marker='o', 
#                            markersize=4, markeredgecolor='w', markeredgewidth=0,
#                            label=f'N cells = {Ncells:.0f}\nN compressions = {Ncomps:.0f}')
# ax.legend(loc='upper right', handles=[LegendMark])
ax.legend().set_visible(False)

ax.grid(visible=True, which='both', axis='y')
renameAxes(ax, renameDict, format_xticks = True, rotation = 60)
ax.set_ylabel('Fitted $H_0$ (nm)')
ax.set_xlabel('')

print(df_m.cellID.unique())



#### Plot 2
ax = fig.add_subplot(spec[1])
ax.set_ylim([0.75, 1.25])

df_m['cellCode'] = df_m['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])

sns.swarmplot(ax = ax, data = dfLOG_m, x = 'cellCode', y = parameter + '_indiv_spread', hue = 'cellCode', 
              s=4, edgecolor='w', linewidth=0)
LegendMark = mlines.Line2D([], [], color='gray', ls='', marker='o', 
                           markersize=0, markeredgecolor='w', markeredgewidth=0,
                           label=f'Average intra-cell CV = {CV_per_cell_avg*100:.1f} %')
ax.legend(handles=[LegendMark])
# ax.plot([], [], marker = 'o', ls='', c='gray', markersize=4, markeredgecolor='w', markeredgewidth=0.75, 
#         label = f'Average cell CV = {CV_per_cell_avg*100:.1f}')
ax.axhline(1, color='k', lw=0.5, zorder=2)
ax.grid(visible=True, which='major', axis='y')
renameAxes(ax, renameDict, format_xticks = True, rotation = 60)
ax.set_ylabel('$Log(H_0)$ distribution\naround cell average')
ax.set_xlabel('')

print(df_m.cellID.unique())




#### Plot 3
ax = fig.add_subplot(spec[2])
ax.set_ylim([0.75, 1.25])

sns.swarmplot(ax = ax, data = dfLOG_fg, x = 'date', y = parameter + '_pop_spread', hue = 'cellID', 
              edgecolor='w', linewidth=0, s=8) #, legend=False)
LegendMark = mlines.Line2D([], [], color='gray', ls='', marker='o', 
                           markersize=0, markeredgecolor='w', markeredgewidth=0,
                           label=f'Inter-cell CV = {CV_population*100:.1f} %')
ax.legend(handles=[LegendMark])
# ax.plot([], [], marker = 'o', ls='', c='gray', markersize=4, markeredgecolor='w', markeredgewidth=0.75, 
#         label = f'Population CV = {CV_population*100:.1f}')
ax.axhline(1, color='k', lw=0.5, zorder=2)
ax.grid(visible=True, which='major', axis='y')
ax.set_ylabel('$Log(H_0)$ distribution\naround population average')
ax.set_xlabel('')
ax.set_xticklabels([])

print(df_fg.cellID.unique())


#### Finalize
# Count
# CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
# fig.suptitle('Thickness')
fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Fig_NC2-1', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = 'Manuscript', cloudSave = 'flexible')


# %%%% Plots -- Stiffness

gs.set_manuscript_options_jv(palette = 'Set2')

#### Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO'] # 
substrate = '20um fibronectin discs'
parameter = 'E_f_<_400'
# parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug'])
df['cellCode'] = df['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['bestH0'] < 1000),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           (df['date'] == '23-03-09'),
           (df['cellCode'] != 'C1801'),
           ]
df_f = filterDf(df, Filters)
df_f['cellNum'] = df_f['cellCode'].apply(lambda x : int(x[1:]))
df_f['E_f_<_400'] = df_f['E_f_<_400']/1000
# Order
co_order = []

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
Manipe = df_f['manipID'].values[0]
Ncells = CountByCond['cellCount'].values[0]
Ncomps = CountByCond['compCount'].values[0]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol, 'cellCode', 'cellNum'], numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
df_fg[parameter + '_pop_spread'] = df_fg[parameter] / np.mean(df_fg[parameter].values)

# Merge
df_m = pd.merge(left=df_f, right=df_fg, on = 'cellID', how='left', suffixes = (None, '_grouped'))
df_m[parameter + '_indiv_spread'] = df_m[parameter] / df_m[parameter + '_grouped']

# Sort by cellID
df_f.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_fg.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_m.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)




#### Apply log

dfLOG_f = filterDf(df, Filters)
dfLOG_f['cellNum'] = dfLOG_f['cellCode'].apply(lambda x : int(x[1:]))

dfLOG_f[parameter] = np.log10(dfLOG_f[parameter])
# df_f[parameter] = np.log(df_f[parameter])

# Group By
dfLOG_fg = dataGroup(dfLOG_f, groupCol = 'cellID', idCols = [condCol, 'cellCode', 'cellNum'], 
                     numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
dfLOG_fg[parameter + '_pop_spread'] = dfLOG_fg[parameter] / np.mean(dfLOG_fg[parameter].values)

# Merge
dfLOG_m = pd.merge(left=dfLOG_f, right=dfLOG_fg, on = 'cellID', how='left', suffixes = (None, '_grouped'))
dfLOG_m[parameter + '_indiv_spread'] = dfLOG_m[parameter] / dfLOG_m[parameter + '_grouped']

# Sort by cellID
dfLOG_f.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)
dfLOG_fg.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)
dfLOG_m.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                  kind='quicksort', na_position='last', ignore_index=False, key=None)

# df_m['cellCode'] = df_m['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])
# dfLOG_m['cellCode'] = dfLOG_m['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])





#### Compute CV
list_cellIDs = dfLOG_f['cellID'].unique()
values_per_cell = [dfLOG_f.loc[dfLOG_f['cellID'] == ID, parameter].values for ID in list_cellIDs]
CV_per_cell = [np.std(cell_vals) / np.mean(cell_vals) for cell_vals in values_per_cell if len(cell_vals) >= 4]
CV_per_cell_avg = np.mean(CV_per_cell)
CV_population = np.std(dfLOG_fg[parameter]) / np.mean(dfLOG_fg[parameter])

print('CV of all cells: ', *[f'{cv:.4f}' for cv in CV_per_cell])
print(f'Mean CV per cell: {CV_per_cell_avg:.4f}')
print(f'CV of full pop: {CV_population:.4f}')
print(f'N cells = {len(list_cellIDs):.0f}')




#### Start plot
fig = plt.figure(figsize=(17/cm_in, 18/cm_in))
spec = fig.add_gridspec(3, 1)



#### Plot 1
ax = fig.add_subplot(spec[0])
# ax.set_ylim([60, 1000])
ax.set_yscale('log')

sns.swarmplot(ax = ax, data = df_m, x = 'cellCode', y = parameter, hue = 'cellCode', 
              s=4, edgecolor='w', linewidth=0)
# LegendMark = mlines.Line2D([], [], color='gray', ls='', marker='o', 
#                            markersize=4, markeredgecolor='w', markeredgewidth=0,
#                            label=f'N cells = {Ncells:.0f}\nN compressions = {Ncomps:.0f}')
# ax.legend(loc='upper right', handles=[LegendMark])
ax.legend().set_visible(False)

ax.grid(visible=True, which='both', axis='y')
renameAxes(ax, renameDict, format_xticks = True, rotation = 60)
ax.set_ylabel('Fitted $E_{400}$ (kPa)')
ax.set_xlabel('')

print(df_m.cellID.unique())



#### Plot 2
ax = fig.add_subplot(spec[1])
ax.set_ylim([0.85, 1.15])

df_m['cellCode'] = df_m['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])

sns.swarmplot(ax = ax, data = dfLOG_m, x = 'cellCode', y = parameter + '_indiv_spread', hue = 'cellCode', 
              s=4, edgecolor='w', linewidth=0)
LegendMark = mlines.Line2D([], [], color='w', ls='', marker='o', 
                           markersize=0, markeredgecolor='w', markeredgewidth=0,
                           label=f'Average intra-cell CV = {CV_per_cell_avg*100:.1f} %')
ax.legend(handles=[LegendMark])
# ax.plot([], [], marker = 'o', ls='', c='gray', markersize=4, markeredgecolor='w', markeredgewidth=0.75, 
#         label = f'Average cell CV = {CV_per_cell_avg*100:.1f}')
ax.axhline(1, color='k', lw=0.5, zorder=2)
ax.grid(visible=True, which='major', axis='y')
renameAxes(ax, renameDict, format_xticks = True, rotation = 60)
ax.set_ylabel('$Log(E_{400})$ distribution\naround cell average')
ax.set_xlabel('')

print(df_m.cellID.unique())




#### Plot 3
ax = fig.add_subplot(spec[2])
ax.set_ylim([0.85, 1.15])

sns.swarmplot(ax = ax, data = dfLOG_fg, x = 'date', y = parameter + '_pop_spread', hue = 'cellID', 
              edgecolor='w', linewidth=0, s=8) #, legend=False)
LegendMark = mlines.Line2D([], [], color='w', ls='', marker='o', 
                           markersize=0, markeredgecolor='w', markeredgewidth=0,
                           label=f'Inter-cell CV = {CV_population*100:.1f} %')
ax.legend(handles=[LegendMark])
# ax.plot([], [], marker = 'o', ls='', c='gray', markersize=4, markeredgecolor='w', markeredgewidth=0.75, 
#         label = f'Population CV = {CV_population*100:.1f}')
ax.axhline(1, color='k', lw=0.5, zorder=2)
ax.grid(visible=True, which='major', axis='y')
ax.set_ylabel('$Log(E_{400})$ distribution\naround population average')
ax.set_xlabel('')
ax.set_xticklabels([])

print(df_fg.cellID.unique())


#### Finalize
# Count
# CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
# fig.suptitle('Thickness')
fig.tight_layout()
plt.show()

ufun.archiveFig(fig, name = 'Fig_NC2-2', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = 'Manuscript', cloudSave = 'flexible')

# %%% Figure NC3 - Plasticity

# %%%% Figure NC3 - Prototypes

gs.set_mediumText_options_jv()
figSubDir = ''

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
# dates = ['23-09-19', '23-11-26', '23-12-03', '23-02-23', ]
dates = ['23-12-03']
# dates = ['23-03-09']
suffix = ''
substrate = '20um fibronectin discs'
color = 'lightseagreen' #'red' # 'lightseagreen'

parameter = 'bestH0'

# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['date'].apply(lambda x : x in dates)),
           (df['bestH0'] <= 900),
           (df['compNum'] <= 5),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           (df['cellName'].apply(lambda x : not '-2' in x)),
           ]
df_f = filterDf(df, Filters)

# New data
# ManipTime
df_f['ManipTime'] = df_f['compAbsStartTime']
cellID_list = df_f['cellID'].unique()
manipID_list = df_f['manipID'].unique()
for mid in manipID_list:
    index_mid = df_f[df_f['manipID']==mid].index
    firstManipTime = np.min(df_f[df_f['manipID']==mid]['compAbsStartTime'].values)
    df_f.loc[index_mid, 'ManipTime'] -= firstManipTime
df_f['ManipTime'] /= 60

# relative H0
df_f['relative H0'] = df_f['bestH0']
for cid in cellID_list:
    index_cid = df_f[df_f['cellID']==cid].index
    firstH0 = df_f[df_f['cellID']==cid]['bestH0'].values[0]
    df_f.loc[index_cid, 'relative H0'] /= firstH0

# relative E
df_f['relative E'] = df_f['E_f_<_400']
for cid in cellID_list:
    index_cid = df_f[df_f['cellID']==cid].index
    firstE = df_f[df_f['cellID']==cid]['E_f_<_400'].values[0]
    df_f.loc[index_cid, 'relative E'] /= firstE
    
# group
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], numCols = ['bestH0'], aggFun = 'mean')
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

df_fgw2['E_f_<_400_wAvg'] /= 1000

fig, axes = plt.subplots(2, 2, figsize=(17/cm_in, 10/cm_in), sharex='col')
#### Plot 1
ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df_f, x='compNum', y='relative H0', hue='cellID', alpha=0.8)
for cid in cellID_list:
    X = df_f[df_f['cellID']==cid]['compNum'].values
    Y = df_f[df_f['cellID']==cid]['relative H0'].values
    ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
ax.set_xlabel('Compression #')
ax.set_ylabel('relative $H_0$')
tickloc = matplotlib.ticker.MultipleLocator(1)
ax.xaxis.set_major_locator(tickloc)
ax.grid(visible=True, which='major', axis='y')

ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df_f, x='compNum', y='relative E', hue='cellID', alpha=0.8)
for cid in cellID_list:
    X = df_f[df_f['cellID']==cid]['compNum'].values
    Y = df_f[df_f['cellID']==cid]['relative E'].values
    ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
ax.set_xlabel('Compression #')
ax.set_ylabel('relative $E$')
tickloc = matplotlib.ticker.MultipleLocator(1)
ax.xaxis.set_major_locator(tickloc)
ax.grid(visible=True, which='major', axis='y')

# # Prettify
# rD = {'manipID' : 'Exp #',
#       }
# for ax in axes:
#     renameAxes(ax, rD, format_xticks = True)
#     renameLegend(ax, rD)
#     ax.grid(visible=True, which='major', axis='y')
    
# # Save
# figname = 'cortex_vs_compnum' + suffix
# ufun.archiveFig(fig, name = figname, ext = '.pdf', dpi = 100,
#                 figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### Plot 2
ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df_fg, x='ManipTime', y='bestH0', color=color, alpha=0.8)
ax.set_xlabel('Time from experiment start (min)')
ax.set_ylabel('$H_0$ (nm)')
for mid in manipID_list:
    X = df_fg[df_fg['manipID']==mid]['ManipTime'].values
    Y = df_fg[df_fg['manipID']==mid]['bestH0'].values
    M = np.array([X,Y]).T
    M = ufun.sortMatrixByCol(M, col=0, direction = 1)
    [X, Y] = M.T
    ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
ax.grid(visible=True, which='major', axis='y')

ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df_fgw2, x='ManipTime', y='E_f_<_400_wAvg', color=color, alpha=0.8)
ax.set_xlabel('Time from experiment start (min)')
ax.set_ylabel('$E$ (kPa)')
for mid in manipID_list:
    X = df_fgw2[df_fgw2['manipID']==mid]['ManipTime'].values
    Y = df_fgw2[df_fgw2['manipID']==mid]['E_f_<_400_wAvg'].values
    M = np.array([X,Y]).T
    M = ufun.sortMatrixByCol(M, col=0, direction = 1)
    [X, Y] = M.T
    ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
ax.grid(visible=True, which='major', axis='y')

# Prettify
rD = {'manipID' : 'Exp #',
      }
for ax in axes[:, 0]:
    ax.legend().set_visible(False)
    ax.grid(visible=True, which='major', axis='y')
    

for ax in axes[:, 1]:
    ax.legend().set_visible(False)
    ax.grid(visible=True, which='major', axis='y')


# Show
fig.suptitle(manipID_list[0])
plt.tight_layout()
plt.show()

# Save
# figname = 'cortex_vs_time' + suffix
# ufun.archiveFig(fig, name = figname, ext = '.pdf', dpi = 100,
#                 figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% Figure NC3-1 - Manip Time

gs.set_manuscript_options_jv()
figSubDir = ''

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
# dates = ['23-09-19', '23-11-26', '23-12-03', '23-02-23', ] # dates = ['23-03-09']
dates = ['23-03-16', '23-03-17', '23-07-20', '23-09-19', '24-07-04'] # , '23-07-17'
# dates = ['23-09-19']
substrate = '20um fibronectin discs'

parameter = 'bestH0'

# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['dmso'])),
           (df['bestH0'] <= 900),
           # (df['compNum'] <= 5),
           (df['E_f_<_400'] <= 2e4),
           # (df['valid_f_<_400'] == True), 
           (df['cellName'].apply(lambda x : not '-2' in x)),
           ]
df_f = filterDf(df, Filters)

# New data
# ManipTime
df_f['ManipTime'] = df_f['compAbsStartTime']
cellID_list = df_f['cellID'].unique()
manipID_list = df_f['manipID'].unique()
for mid in manipID_list:
    index_mid = df_f[df_f['manipID']==mid].index
    firstManipTime = np.min(df_f[df_f['manipID']==mid]['compAbsStartTime'].values)
    df_f.loc[index_mid, 'ManipTime'] -= firstManipTime
df_f['ManipTime'] /= 60


# # relative H0
# df_f['relative H0'] = df_f['bestH0']
# for cid in cellID_list:
#     index_cid = df_f[df_f['cellID']==cid].index
#     firstH0 = df_f[df_f['cellID']==cid]['bestH0'].values[0]
#     df_f.loc[index_cid, 'relative H0'] /= firstH0

# # relative E
# df_f['relative E'] = df_f['E_f_<_400']
# for cid in cellID_list:
#     index_cid = df_f[df_f['cellID']==cid].index
#     firstE = df_f[df_f['cellID']==cid]['E_f_<_400'].values[0]
#     df_f.loc[index_cid, 'relative E'] /= firstE
    
# group
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], numCols = ['bestH0'], aggFun = 'mean')
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_fgw2['E_f_<_400_wAvg'] /= 1000

df_fg = df_fg.drop_duplicates(subset='ManipTime')
df_fgw2 = df_fgw2.drop_duplicates(subset='ManipTime')




fig, axes = plt.subplots(2, 1, figsize=(17/cm_in, 12/cm_in), sharex='col')
cL = [matplotlib.colormaps['Set2_r'].colors[i] for i in [2,3,5,6,7]]

ax = axes[0]
# sns.scatterplot(ax=ax, data=df_fg, x='ManipTime', y='bestH0', alpha=0.8)
# ax.set_xlabel('Time from experiment start (min)')
ax.set_ylabel('$H_0$ (nm)')
ax.set_yscale('log')
ax.set_ylim([70, 1100])
for mid, c in zip(manipID_list, cL):
    df_fg_mid = df_fg[df_fg['manipID']==mid]
    # sns.scatterplot(ax=ax, data=df_fg_mid, x='ManipTime', y='bestH0', alpha=0.8, color = c, label=mid)
    X = df_fg_mid['ManipTime'].values
    Y = df_fg_mid['bestH0'].values
    M = np.array([X,Y]).T
    M = ufun.sortMatrixByCol(M, col=0, direction = 1)
    [X, Y] = M.T
    ax.plot(X, Y, ls='-', c=c, lw=1.5, alpha=0.75, zorder=0, label=mid + '\n$N_{cells}$ = ' + f'{len(X):.0f}',
            marker = '.', markersize = 6, markerfacecolor = 'w', markeredgecolor = 'gray')
ax.grid(visible=True, which='major', axis='y')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 8)

ax = axes[1]
# sns.scatterplot(ax=ax, data=df_fgw2_mid, x='ManipTime', y='E_f_<_400_wAvg', alpha=0.8)
ax.set_xlabel('Time from experiment start (min)')
ax.set_ylabel('$E$ (kPa)')
ax.set_yscale('log')
for mid, c in zip(manipID_list, cL):
    df_fgw2_mid = df_fgw2[df_fgw2['manipID']==mid]
    # sns.scatterplot(ax=ax, data=df_fgw2_mid, x='ManipTime', y='E_f_<_400_wAvg', alpha=0.8, color = c, label='')
    X = df_fgw2_mid['ManipTime'].values
    Y = df_fgw2_mid['E_f_<_400_wAvg'].values
    M = np.array([X,Y]).T
    M = ufun.sortMatrixByCol(M, col=0, direction = 1)
    [X, Y] = M.T
    # ax.plot(X, Y, ls='-', c=c, lw=0.8, alpha=0.8, zorder=0)
    ax.plot(X, Y, ls='-', c=c, lw=1.5, alpha=0.75, zorder=0,
            marker = '.', markersize = 6, markerfacecolor = 'w', markeredgecolor = 'gray')
ax.grid(visible=True, which='major', axis='y')
# ax.legend().set_visible(False)

# Prettify
rD = {'manipID' : 'Exp #',
      }


for ax in axes:
    ax.grid(visible=True, which='major', axis='y')


fig.legend(loc='upper center', bbox_to_anchor=(0.535, 0.99), ncols=5, fontsize = 7.5)

# Show
# fig.legend()
fig.tight_layout(rect=(0,0,1,0.92))
plt.show()

# Count
# df_f = filterDf(df, Filters)
CountByCond, CountByCell = makeCountDf(df_f, 'manipID')
# Save
figSubDir = 'Plasticity'
name = 'manipTime_expt'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')




# %%%% Figure NC3-2 - Successive compressions LIN

gs.set_mediumText_options_jv()
figSubDir = ''

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
# dates = ['23-09-19', '23-11-26', '23-12-03', '23-02-23', ] # dates = ['23-03-09']
dates = ['23-03-16', '23-03-17'] # , '23-07-17'
# dates = ['23-09-19']
substrate = '20um fibronectin discs'

parameter = 'bestH0'

# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['dmso'])),
           (df['bestH0'] <= 900),
            (df['compNum'] <= 10),
           (df['E_f_<_400'] <= 2e4),
           # (df['valid_f_<_400'] == True), 
           (df['cellName'].apply(lambda x : not '-2' in x)),
           ]
df_f = filterDf(df, Filters)

# New data
# ManipTime
df_f['ManipTime'] = df_f['compAbsStartTime']
cellID_list = df_f['cellID'].unique()
manipID_list = df_f['manipID'].unique()
for mid in manipID_list:
    index_mid = df_f[df_f['manipID']==mid].index
    firstManipTime = np.min(df_f[df_f['manipID']==mid]['compAbsStartTime'].values)
    df_f.loc[index_mid, 'ManipTime'] -= firstManipTime
df_f['ManipTime'] /= 60

# relative H0
df_f['relative H0'] = df_f['bestH0']
for cid in cellID_list:
    index_cid = df_f[df_f['cellID']==cid].index
    firstH0 = df_f[df_f['cellID']==cid]['bestH0'].values[0]
    df_f.loc[index_cid, 'relative H0'] /= firstH0

# relative E
df_f['relative E'] = df_f['E_f_<_400']
for cid in cellID_list:
    index_cid = df_f[df_f['cellID']==cid].index
    firstE = df_f[df_f['cellID']==cid]['E_f_<_400'].values[0]
    df_f.loc[index_cid, 'relative E'] /= firstE
    
# group
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], numCols = ['bestH0'], aggFun = 'mean')
# df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], 
#                                       valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
# df_fgw2['E_f_<_400_wAvg'] /= 1000

# df_fg = df_fg.drop_duplicates(subset='ManipTime')
# df_fgw2 = df_fgw2.drop_duplicates(subset='ManipTime')

Ncols = len(manipID_list)
fig, axes = plt.subplots(2, Ncols, figsize=(17/cm_in, 12/cm_in), sharex='col')

for i in range(Ncols):
    mid = manipID_list[i]
    df_f_mid = df_f[df_f['manipID'] == mid]
    
    axcol = axes[:, i]
    
    #### Plot H0
    ax = axcol[0]
    sns.scatterplot(ax=ax, data=df_f_mid, x='compNum', y='relative H0', hue='cellID', alpha=0.8)
    for cid in cellID_list:
        df_f_cid = df_f_mid[df_f_mid['cellID']==cid]
        X = df_f_cid['compNum'].values
        Y = df_f_cid['relative H0'].values
        ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
    ax.set_xlabel('Compression #')
    ax.set_ylabel('relative $H_0$')
    tickloc = matplotlib.ticker.MultipleLocator(1)
    ax.xaxis.set_major_locator(tickloc)
    ax.grid(visible=True, which='major', axis='y')
    
    #### Plot E
    ax = axcol[1]
    sns.scatterplot(ax=ax, data=df_f_mid, x='compNum', y='relative E', hue='cellID', alpha=0.8)
    for cid in cellID_list:
        df_f_cid = df_f_mid[df_f_mid['cellID']==cid]
        X = df_f_cid['compNum'].values
        Y = df_f_cid['relative E'].values
        ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
    ax.set_xlabel('Compression #')
    ax.set_ylabel('relative $E$')
    tickloc = matplotlib.ticker.MultipleLocator(1)
    ax.xaxis.set_major_locator(tickloc)
    ax.grid(visible=True, which='major', axis='y')
    
    for ax in axcol:
        ax.legend().set_visible(False)

# Show


fig.tight_layout()
plt.show()

# Count
# df_f = filterDf(df, Filters)
CountByCond, CountByCell = makeCountDf(df_f, 'manipID')
# Save
figSubDir = 'Plasticity'
name = 'compNum_cell_LIN'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%%% Figure NC3-2 - Successive compressions LOG

gs.set_mediumText_options_jv()
figSubDir = ''

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
# dates = ['23-09-19', '23-11-26', '23-12-03', '23-02-23', ] # dates = ['23-03-09']
dates = ['23-03-16', '23-03-17'] # , '23-07-17'
# dates = ['23-09-19']
substrate = '20um fibronectin discs'

parameter = 'bestH0'

# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in ['dmso'])),
           (df['bestH0'] <= 900),
            (df['compNum'] <= 10),
           (df['E_f_<_400'] <= 2e4),
           # (df['valid_f_<_400'] == True), 
           (df['cellName'].apply(lambda x : not '-2' in x)),
           ]
df_f = filterDf(df, Filters)

# New data
# ManipTime
df_f['ManipTime'] = df_f['compAbsStartTime']
cellID_list = df_f['cellID'].unique()
manipID_list = df_f['manipID'].unique()
for mid in manipID_list:
    index_mid = df_f[df_f['manipID']==mid].index
    firstManipTime = np.min(df_f[df_f['manipID']==mid]['compAbsStartTime'].values)
    df_f.loc[index_mid, 'ManipTime'] -= firstManipTime
df_f['ManipTime'] /= 60

# relative H0
df_f['relative H0'] = df_f['bestH0']
for cid in cellID_list:
    index_cid = df_f[df_f['cellID']==cid].index
    firstH0 = df_f[df_f['cellID']==cid]['bestH0'].values[0]
    df_f.loc[index_cid, 'relative H0'] = np.log(df_f['relative H0'])/np.log(firstH0)

# relative E
df_f['relative E'] = df_f['E_f_<_400']
for cid in cellID_list:
    index_cid = df_f[df_f['cellID']==cid].index
    firstE = df_f[df_f['cellID']==cid]['E_f_<_400'].values[0]
    df_f.loc[index_cid, 'relative E'] = np.log(df_f['relative E'])/np.log(firstE)
    
# group
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], numCols = ['bestH0'], aggFun = 'mean')
# df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], 
#                                       valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
# df_fgw2['E_f_<_400_wAvg'] /= 1000

# df_fg = df_fg.drop_duplicates(subset='ManipTime')
# df_fgw2 = df_fgw2.drop_duplicates(subset='ManipTime')

Ncols = len(manipID_list)
fig, axes = plt.subplots(2, Ncols, figsize=(17/cm_in, 12/cm_in), sharex='col', sharey='row')

for i in range(Ncols):
    mid = manipID_list[i]
    df_f_mid = df_f[df_f['manipID'] == mid]
    
    axcol = axes[:, i]
    
    #### Plot H0
    ax = axcol[0]
    ax.set_title(mid)
    sns.scatterplot(ax=ax, data=df_f_mid, x='compNum', y='relative H0', hue='cellID', alpha=0.8)
    for cid in cellID_list:
        df_f_cid = df_f_mid[df_f_mid['cellID']==cid]
        X = df_f_cid['compNum'].values
        Y = df_f_cid['relative H0'].values
        ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
    # ax.set_xlabel('Compression #')
    if i == 0:
        ax.set_ylabel('relative $log(H_0)$')
    tickloc = matplotlib.ticker.MultipleLocator(1)
    ax.xaxis.set_major_locator(tickloc)
    ax.grid(visible=True, which='major', axis='y')
    
    #### Plot E
    ax = axcol[1]
    sns.scatterplot(ax=ax, data=df_f_mid, x='compNum', y='relative E', hue='cellID', alpha=0.8)
    for cid in cellID_list:
        df_f_cid = df_f_mid[df_f_mid['cellID']==cid]
        X = df_f_cid['compNum'].values
        Y = df_f_cid['relative E'].values
        ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
    ax.set_xlabel('Compression #')
    if i == 0:
        ax.set_ylabel('relative $log(E_{400})$')
    tickloc = matplotlib.ticker.MultipleLocator(1)
    ax.xaxis.set_major_locator(tickloc)
    ax.grid(visible=True, which='major', axis='y')
    
    for ax in axcol:
        ax.legend().set_visible(False)

# Show


fig.tight_layout()
plt.show()

# Count
# df_f = filterDf(df, Filters)
CountByCond, CountByCell = makeCountDf(df_f, 'manipID')
# Save
figSubDir = 'Plasticity'
name = 'compNum_cell_LOG'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')



# %%% Figure NC4 - Solidity

# %%%% NC4-1 Distributions

gs.set_manuscript_options_jv()

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
substrate = '20um fibronectin discs'

# df, condCol = makeCompositeCol(df, cols=['date', 'drug'])
# dates = ['23-03-16','23-03-17']
# figname = 'bestH0' + drugSuffix
condCol = 'drug'

# New parameter
df['dH'] = df['initialThickness'] - df['previousThickness']
df['H0/H5'] = df['initialThickness'] / df['previousThickness']
df['dH/H5'] = df['dH'] / df['previousThickness']
parameter = 'dH'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['drug'] == 'dmso'),
           # (df['date'].apply(lambda x : x in dates)),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['normal field'] == 5),
           (df['ramp field'].apply(lambda x : x.startswith('2.5'))),
           (df['bestH0'] <= 900),
           (df['dH/H5'] <= 3),
           (df['dH'] >= -30),
           ]

df_f = filterDf(df, Filters)

# Order
# co_order = ['none', 'dmso']
co_order = []

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['dH', 'dH/H5'], aggFun = 'mean')

median_dH = np.median(df_f['dH'].values)
median_dH_ratio = np.median(df_f['dH/H5'].values)


# Plot
fig, axes = plt.subplots(1, 2, figsize=(17/gs.cm_in, 6/gs.cm_in), sharey=True)
color = styleDict['dmso']['color']

ax = axes[0]
# ax.set_title('Thickness')
ax.set_xlabel('$\delta H$ (nm)')
ax.set_ylabel('Count (# compressions)')
N, bins, patches = ax.hist(x=df_f['dH'].values, bins=20, color = color)
ax.set_xlim([-40, ax.get_xlim()[1]])
ax.axvline(median_dH, c='darkred', label = f'Median = {median_dH:.0f} nm')
ax.legend(fontsize=8)


ax = axes[1]
# ax.set_title('Thickness')
ax.set_xlabel('Ratio $\delta H/H_{5mT}$')
# ax.set_ylabel('Count (# cells)')
N, bins, patches = ax.hist(x=df_f['dH/H5'].values, bins=20, color = color)
ax.set_xlim([-0.2, ax.get_xlim()[1]])
ax.axvline(median_dH_ratio, c='darkred', label = f'Median = {median_dH_ratio:.2f}')
ax.legend(fontsize=8)
           
# Prettify
for ax in axes:
    rD = {'none' : 'No drug',
          'dmso' : 'DMSO', 
          }
    
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False, rotation = 30)
    # renameLegend(ax, renameDict)
    ax.grid(visible=True, which='major', axis='y')
    # ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

figSubDir = 'Solidity'
name = 'DeltaH_Distribution'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%%% NC4-2 Tendances

gs.set_manuscript_options_jv()

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
substrate = '20um fibronectin discs'

# dates = ['23-03-16','23-03-17']
# figname = 'bestH0' + drugSuffix

# New parameters
df['dH'] = df['initialThickness'] - df['previousThickness']
df['H0/H5'] = df['initialThickness'] / df['previousThickness']
df['dH/H5'] = df['dH'] / df['previousThickness']
parameter = 'dH'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['drug'] == 'dmso'),
           # (df['date'].apply(lambda x : x in dates)),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['normal field'] == 5),
           (df['ramp field'].apply(lambda x : x.startswith('2.5'))),
           (df['bestH0'] <= 900),
           (df['dH/H5'] <= 3),
           (df['dH'] >= -30),
           ]

df_f = filterDf(df, Filters)

CountByCond, CountByCell = makeCountDf(df_f, 'manipID')
excludedManip = CountByCond[CountByCond['cellCount']<13].index.values

# Filter
Filters = [(df_f['manipID'].apply(lambda x : not x in excludedManip)),
            (df_f['cellName'].apply(lambda x : not '-2' in x)),
           ]
df_f = filterDf(df_f, Filters)

cellID_list = df_f['cellID'].unique()
manipID_list = df_f['manipID'].unique()
Nm = len(manipID_list)

# ManipTime
df_f['ManipTime'] = df_f['compAbsStartTime']

for mid in manipID_list:
    index_mid = df_f[df_f['manipID']==mid].index
    firstManipTime = np.min(df_f[df_f['manipID']==mid]['compAbsStartTime'].values)
    df_f.loc[index_mid, 'ManipTime'] -= firstManipTime
df_f['ManipTime'] /= 60

# Group
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], numCols = ['bestH0', 'dH', 'dH/H5'], aggFun = 'mean')

fig, axes = plt.subplots(Nm, 1, figsize=(17/gs.cm_in, 20/gs.cm_in), sharex=True)
PLOT_FIT = True

#### Plot 1
ax = ax
cL = gs.cL_Set21

ax.set_xlabel('Time from experiment start (min)')
ax.set_ylabel('$\delta H$ / $H_{5mT}$')
for i in range(Nm):
    ax = axes[i]
    mid = manipID_list[i]
    color = cL[i]
    print(mid)
    
    X = df_fg[df_fg['manipID']==mid]['ManipTime'].values
    Y = df_fg[df_fg['manipID']==mid]['dH/H5'].values
    M = np.array([X,Y]).T
    M = ufun.sortMatrixByCol(M, col=0, direction = 1)
    [X, Y] = M.T
    ax.scatter(X, Y, color=color, alpha=0.8, label=mid)
    ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
    ax.set_ylim([0, 1.5])
    ax.set_ylabel('$\delta H/H_{5mT}$')
    
    if PLOT_FIT:
        df_fit = df_fg[df_fg['manipID']==mid].copy()
        params, results = ufun.fitLineHuber(df_fit['ManipTime'].values, df_fit['dH/H5'].values)
        [b, a] = params
        pval = results.pvalues[1]
        fitX = np.linspace(min(df_fit['ManipTime'].values), max(df_fit['ManipTime'].values), 10)
        fitY = a*fitX + b
        ax.plot(fitX, fitY, ls='--', c=color, lw=1.5,
                label = f'{a:.3f} x + {b:.1f}\np-val = {pval:.2f}')
        
    ax.grid(visible=True, which='major', axis='y')
    
    # Prettify
    rD = {'manipID' : 'Exp #',
          'dH/H5' : '$\delta H$ / $H_H_{5mT}$'
          }

    ax.legend(loc='upper right') # (loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(visible=True, which='major', axis='y')

axes[-1].set_xlabel('Time (min)')
# if PLOT_FIT:
#     params, results = ufun.fitLineHuber(df_fg['ManipTime'].values, df_fg['dH/H5'].values)
#     [b, a] = params
#     pval = results.pvalues[1]
#     fitX = np.linspace(min(df_fg['ManipTime'].values), max(df_fg['ManipTime'].values), 10)
#     fitY = a*fitX + b
#     ax.plot(fitX, fitY, ls='--', c='k', lw=2,
#             label = 'Global Fit\n' + f'{a:.3f} x + {b:.1f}\np-val = {pval:.2f}')

# Prettify
# rD = {'manipID' : 'Exp #',
#       'dH/H5' : '$\delta H$ / $H_H_{5mT}$'
#       }

# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax.grid(visible=True, which='major', axis='y')

# Show
plt.tight_layout()
plt.show()

# Save
figSubDir
figname = 'DeltaH_Time'
ufun.archiveFig(fig, name = figname, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = 'Solidity', cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, figname+'_count.txt'), sep='\t')


# %%% Figure NC5 - Stress Stiffening



# %%%% Figure NC5-1 - Residuals -> with takaM

# %%%% Figure NC5-2 - Stress-strain fits by parts -> with takaM

# %%%% Figure NC5-3 - Curve zoology V1

gs.set_manuscript_options_jv(palette = 'Set1')

# Define
df = MecaData_Phy
substrate = '20um fibronectin discs'
condCol = 'drug' # 'date'
# mode = 'wholeCurve'

          
# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['manipID'] == '23-03-09_M4'),
           ]
df_f = filterDf(df, Filters)

df_f['cellComp'] = df_f['cellName'] + '_c' + df_f['compNum'].apply(lambda x : str(int(x)))
cells =     []
cellComps = ['M4_P1_C2_c4', 'M4_P1_C5_c4', 'M4_P1_C12_c3', 'M4_P1_C4_c5', 'M4_P1_C8_c3', 
             'M4_P1_C9_c5', 'M4_P1_C14_c1'] # 'M4_P1_C15_c2', 

Filters = [(df_f['cellName'].apply(lambda x : x in cells)) | (df_f['cellComp'].apply(lambda x : x in cellComps)),
           ]
df_f = filterDf(df_f, Filters)

df_f['cellCode'] = df_f['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])
df_f['cellNum'] = df_f['cellCode'].apply(lambda x : int(x[1:]))
df_f.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)

# Order
co_order = []

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

#### Plot 1
fig, axes = plt.subplots(2,1, figsize=(17/gs.cm_in, 17/gs.cm_in))

ax = axes[0]
ax.set_yscale('log')
fig, ax = plotPopKS_V3(df_f, fig = fig, ax = ax, 
                 condition = '', co_order = [],
                 fitType = 'stressRegion', fitWidth=75,
                 mode = 'wholeCurve', scale = 'lin', Sinf = 0, Ssup = 1100)
ax.set_ylim([0.8, 120])



#### Plot 2
ax = axes[1]
StressRange_2D_V2(df_f, fig=fig, ax=ax, condition='cellID', defaultMode = False)

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['drug'] == 'dmso'),
           (df['valid_f_<_400'] == True), 
           (df['minStress'] > 0),
           (df['maxStress'] < 1e4), 
           (df['bestH0'] < 1000), 
           ]
df_f2 = filterDf(df, Filters)


ax.set_yscale('log')
StressRange_2D_V2(df_f2, fig = fig, ax = ax, condition='drug', defaultMode = True)
ax.set_ylim([1e1, 1e4])
ax.set_ylabel('Stress range (Pa)')
ax.set_xlim([0, 1050])
ax.set_xlabel('$H_0$ (nm)')
ax.grid(axis='both', which='major')
ax.legend(loc='upper right')

fig.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f2, condCol)
# Save
figSubDir = 'NonLinearity'
name = 'CompressionsZoo_V2'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%%% Figure NC5-3 - Curve zoology V1

gs.set_manuscript_options_jv(palette = 'Set1')

# Define
df = MecaData_Phy
substrate = '20um fibronectin discs'
condCol = 'drug' # 'date'
# mode = 'wholeCurve'

          
# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['manipID'] == '23-03-09_M4'),
           ]
df_f = filterDf(df, Filters)

df_f['cellComp'] = df_f['cellName'] + '_c' + df_f['compNum'].apply(lambda x : str(int(x)))
cells =     []
cellComps = ['M4_P1_C2_c4', 'M4_P1_C5_c4', 'M4_P1_C12_c3', 'M4_P1_C4_c5', 'M4_P1_C8_c3', 
             'M4_P1_C9_c5', 'M4_P1_C14_c1'] # 'M4_P1_C15_c2', 

Filters = [(df_f['cellName'].apply(lambda x : x in cells)) | (df_f['cellComp'].apply(lambda x : x in cellComps)),
           ]
df_f = filterDf(df_f, Filters)

df_f['cellCode'] = df_f['cellName'].apply(lambda x : x.split('_')[-1].split('-')[0])
df_f['cellNum'] = df_f['cellCode'].apply(lambda x : int(x[1:]))
df_f.sort_values('cellNum', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)

# Order
co_order = []

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

#### Plot 1
fig, axes = plt.subplots(2,1, figsize=(17/gs.cm_in, 17/gs.cm_in))

ax = axes[0]
ax.set_yscale('log')
fig, ax = plotPopKS_V3(df_f, fig = fig, ax = ax, 
                 condition = '', co_order = [],
                 fitType = 'stressRegion', fitWidth=75,
                 mode = 'wholeCurve', scale = 'lin', Sinf = 0, Ssup = 1100)
ax.set_ylim([0.8, 120])



#### Plot 2
# ax = axes[1]
# StressRange_2D_V2(df_f, fig=fig, ax=ax, condition='cellID', defaultMode = False)

# # Filter
# Filters = [(df['validatedThickness'] == True), 
#            (df['substrate'] == substrate),
#            (df['drug'] == 'dmso'),
#            (df['valid_f_<_400'] == True), 
#            (df['minStress'] > 0),
#            (df['maxStress'] < 1e4), 
#            (df['bestH0'] < 1000), 
#            ]
# df_f2 = filterDf(df, Filters)


# ax.set_yscale('log')
# StressRange_2D_V2(df_f2, fig = fig, ax = ax, condition='drug', defaultMode = True)
# ax.set_ylim([1e1, 1e4])
# ax.set_ylabel('Stress range (Pa)')
# ax.set_xlim([0, 1050])
# ax.set_xlabel('$H_0$ (nm)')
# ax.grid(axis='both', which='major')
# ax.legend(loc='upper right')

fig.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f2, condCol)
# Save
figSubDir = 'NonLinearity'
name = 'CompressionsZoo_V3-1'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')



# %%%% Figure NC5-4 - Average behavior

gs.set_manuscript_options_jv(palette = 'Set2')

# Define
df = MecaData_Phy
substrate = '20um fibronectin discs'
condCol = 'drug' # 'date'
# mode = 'wholeCurve'


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           # (df['date'].apply(lambda x : x in dates)),
           (df['drug'] == 'dmso'),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = []

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(17/gs.cm_in,12/gs.cm_in))
ax.set_yscale('log')

intervals = ['100_300', '150_400', '200_500', '300_1000', '500_1200']
cL = plt.cm.plasma(np.linspace(0.1, 0.9, len(intervals)))

for i, interval in enumerate(intervals):
    colorDict = {'dmso':cL[i]}
    labelDict = {'dmso':f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"}
    plotPopKS_V2(df_f, fig = fig, ax = ax, condition = 'drug', co_order = co_order, 
                 colorDict = colorDict, labelDict = labelDict,
                 fitType = 'stressGaussian', fitWidth=75, mode = interval, Sinf = 0, Ssup = np.Inf)

           
# Prettify
rD = {'dmso' : 'DMSO',
      }


ax.grid(visible=True, which='both', axis='y')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)

ax = ax
ax.set_xlim(0, 1300)
ax.set_ylim(0.8, 30)
ax.legend(loc='upper left')

# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
figSubDir = 'NonLinearity'
name = 'nonLin_dmso_w75_V2'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')







# %%% Figure NC6 - Thickness & Stiffness

# %%%% 1. E400

gs.set_manuscript_options_jv(palette = 'Set2')

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix
XCol = 'bestH0'
YCol = 'E_f_<_400'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x not in excluded_dates)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           ]

df_f = filterDf(df, Filters)
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Order
co_order = []

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(1,2, figsize=(17/gs.cm_in,8/gs.cm_in))

# LinLog
ax = axes[0]
ax.set_xscale('log')
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_f, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 0.5)

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 0.5)
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
# rD = {'dmso & 0.0' : 'DMSO',
#       'Y27 & 50.0' : 'Y27\n(50µM)',
#       'LIMKi & 10.0' : 'LIMKi\n(10µM)',
#       'LIMKi & 20.0' : 'LIMKi\n(20µM)',
#       'blebbistatin & 50.0':'Blebbi\n(50µM)',
#       'blebbistatin & 100.0':'Blebbi\n(100µM)',
#       'none & 0.0':'No Drug',
#       'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
#       'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
#       # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
#       # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
#       }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    ax.legend(fontsize = 6)
    # renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD)
    # # ax.set_xlim(0, 400)
    # # ax.set_ylim(900, 12000)

# axes[0].set_xlabel('')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
figSubDir = 'E-h'
name = 'E-h-01'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%%% 2. fit_K_400_100

gs.set_manuscript_options_jv(palette = 'Set2')

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix
XCol = 'bestH0'
YCol = 'fit_K'
fitID='300_100'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x not in excluded_dates)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           # (df[XCol] >= 50),
           # (df[XCol] < 1000),
           ]

df_f = filterDf(df, Filters)


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
fig, axes = plt.subplots(1,2, figsize=(17/gs.cm_in,8/gs.cm_in))

ax = axes[0]
ax.set_xscale('log')
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_ff, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol, condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 0.5)

ax = axes[1]
ax.set_xscale('log')
ax.set_yscale('log')
fig, ax = D2Plot_wFit(df_plot, fig = fig, ax = ax, 
                XCol = XCol, YCol = YCol + '_wAvg', condition=condCol, co_order = co_order,
                modelFit=True, modelType='y=k*x^a', writeEqn = True, robust = True,
                figSizeFactor = 1, markersizeFactor = 0.5)

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
# rD = {'dmso & 0.0' : 'DMSO',
#       'Y27 & 50.0' : 'Y27 (50µM)',
#       'LIMKi & 10.0' : 'LIMKi (10µM)',
#       'LIMKi & 20.0' : 'LIMKi (20µM)',
#       'fit_K':'Tangeantial Modulus (Pa)',
#       'fit_K_wAvg':'Tangeantial Modulus (Pa) - wAvg',
#       }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both')
    ax.legend(fontsize=6)
    # renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD)
    # ax.set_xlim(0, 400)
    ax.set_ylim(500, 20000)
    
# axes[0].set_xlabel('')
fig.suptitle(f"Fit on stress = {fitID.split('_')[0]}+/-{fitID.split('_')[1]} Pa")

# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
figSubDir = 'E-h'
name = 'K-200-400-h-01'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')






# %%% Figure MM - Plots Beads In-In vs Out-Out
gs.set_manuscript_options_jv()
custom_cycler = (cycler(color=gs.cL_Set21))


srcDirOut = "D:/MagneticPincherData/Raw/Control_InIn_OutOut"
srcDirIn  = "D:/MagneticPincherData/Raw/Control_InIn"
dirIn = os.path.join(srcDirIn, 'Timeseries_IN')
dirOut = os.path.join(srcDirOut, 'Timeseries_OUT')
dict_tsdf_in =  dict([(ufun.findInfosInFileName(f, 'cellID'), pd.read_csv(os.path.join(dirIn,  f), sep=';')) for f in os.listdir(dirIn)  if f.endswith('.csv')])
dict_tsdf_out = dict([(ufun.findInfosInFileName(f, 'cellID'), pd.read_csv(os.path.join(dirOut, f), sep=';')) for f in os.listdir(dirOut) if f.endswith('.csv')])

Din = 4.493 # 4.493
Dout = 4.516

# ufun.findInfosInFileName(f, infoType)

for df in dict_tsdf_in.values():
    df['h'] = (df['D3']-Din)*1000
for df in dict_tsdf_out.values():
    df['h'] = (df['D3']-Dout)*1000

fig, axes = plt.subplots(1, 2, figsize=(17/cm_in, 8/cm_in), sharey=True)

count = 0
ax = axes[1]
ax.set_prop_cycle(custom_cycler)
for cid in dict_tsdf_in.keys():
    tsdf = dict_tsdf_in[cid]
    tsdf0 = tsdf[tsdf['idxAnalysis']==0]
    tsdf0 = tsdf0[tsdf0['idxLoop']<=5]
    group = tsdf0[['idxLoop', 'T', 'h']].groupby('idxLoop')
    df = group.agg({'T':'mean', 'h':'median'})
    # ax.plot(df.loc[df['idxAnalysis']==0, 'T'], df.loc[df['idxAnalysis']==0, 'h'], 
    #         ls='', marker = 'o', mec='w', mew=0.5, label=cid)
    if max(df['h']) > 80:
        continue
    else:
        count += 1
        ax.plot(df['T'], df['h'], ls='-', lw=1,
                marker = 'o', mec='w', mew=0.5, markersize = 8, zorder=6, # label=cid
                )
    # ax.plot(df['T'], df['h'], ls='-', lw = 1, c='gray')
ax.plot([], [], ls='-', lw=1, c = 'gray',
        marker = 'o', mec='w', mew=0.5, markersize = 8, zorder=6,
        label=f'N = {(count):.0f}')
ax.axhline(0, ls='-', c='k', lw=1.5)
ax.set_title('Pair of beads inside')
ax.set_xlim([0, 100])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Thickness (nm)')
ax.grid(which='major', axis='both')
ax.legend(fontsize=8)

count = 0
ax = axes[0]
ax.set_prop_cycle(custom_cycler)
for cid in dict_tsdf_out.keys():
    tsdf = dict_tsdf_out[cid]
    tsdf0 = tsdf[tsdf['idxAnalysis']==0]
    group = tsdf0[['idxLoop', 'T', 'h']].groupby('idxLoop')
    df = group.agg({'T':'mean', 'h':'mean'})
    # ax.plot(df.loc[df['idxAnalysis']==0, 'T'], df.loc[df['idxAnalysis']==0, 'h'], 
    #         ls='', marker = 'o', mec='w', mew=0.5, label=cid)
    if max(df['h']) > 80:
        continue
    else:
        count += 1
        ax.plot(df['T'], df['h'], ls='-', lw=1,
                marker = 'o', mec='w', mew=0.5, markersize = 8, zorder=6, # label=cid
                )
ax.plot([], [], ls='-', lw=1, c = 'gray',
        marker = 'o', mec='w', mew=0.5, markersize = 8, zorder=6,
        label=f'N = {(count):.0f}')
ax.axhline(0, ls='-', c='k', lw=1.5)
ax.set_title('Pair of beads outside')
ax.set_xlim([0, 100])
ax.set_ylim([-60, 85])
ax.set_xlabel('Time (s)')
# ax.set_ylabel('Thickness (nm)')
ax.grid(which='major', axis='both')
ax.legend(fontsize=8)

plt.show()

# Save
figname = 'InIn_OutOut_V2'
ufun.archiveFig(fig, name = figname, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = 'Internalization', cloudSave = 'flexible')


# %%% Figure MM - Shear rate

def SelectFile(f, list_strings):
    res = False
    for s in list_strings:
        if f.startswith(s):
            res = True
    return(res)

def strainRateCheck_V2(list_tsdf, names_list):
    gs.set_manuscript_options_jv('Set1')
    N = len(list_tsdf)
    
    fig, axes = plt.subplots(N, 2, figsize=(17/gs.cm_in, 5*N/gs.cm_in), sharex='col')
    
    for k in range(N):
        tsdf = list_tsdf[k]
        cellName = names_list[k]
        Nc = min(np.max(tsdf['idxAnalysis']), 5)
        coi = ['T', 'B', 'F', 'H0', 'Stress', 'Strain']
        
        rate_matrix = []
        
        for i in range(1, Nc+1):
            df_i = tsdf[tsdf['idxAnalysis'] == i][coi].copy()
            df_i = df_i.dropna()
            # i400 = ufun.findFirst(True, df_i['F'].values>400)
            # t400 = df_i['T'].values[i400]-df_i['T'].values[0]
            
            x, y = df_i['T']-df_i['T'].values[0], df_i['Strain']
                
            tck_s = interpolate.splrep(x, y, s=0.01)
            Bspl = interpolate.BSpline(*tck_s)
            Bspl_der = Bspl.derivative(1)
            rate_matrix.append(Bspl_der(x))
            
            ax = axes[k, 0]
            ax.plot(x, y)
            # ax.axvline(t400, color='gray', alpha=0.5, ls='--')
            ax.set_ylim([0, 0.25])
            ax.set_ylabel(cellName, fontsize = 10)
            tickloc = matplotlib.ticker.MultipleLocator(0.05)
            ax.yaxis.set_major_locator(tickloc)
            ax.grid(axis='y')
            
            ax = axes[k, 1]
            ax.plot(x, Bspl_der(x), label=f'#{i:.0f} - $H_0$ = {df_i.H0.values[0]:.0f} nm',)
            ax.legend(fontsize=6, loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_ylim([0, 0.25])
            tickloc = matplotlib.ticker.MultipleLocator(0.05)
            ax.yaxis.set_major_locator(tickloc)
            ax.grid(axis='y')
    
    axes[0,0].set_title('Strain - $\epsilon (t)$')
    axes[0,1].set_title('Strain rate - $d\epsilon /dt$ $(s^{-1})$')
    axes[-1,0].set_xlabel('Time (s)')
    axes[-1,1].set_xlabel('Time (s)')
    
    fig.tight_layout()
    plt.show()
    
    # ufun.archiveFig(fig, name = name + '_allStrainRates', ext = '.pdf', dpi = 100,
    #                 figDir = figDir, figSubDir = '', cloudSave = 'flexible')
    
    return(fig, axes)



# specifs = ['23-03-09_M4_P1_C3', '23-03-09_M4_P1_C4', '23-03-09_M4_P1_C5'] #, '24-04-11_M3_P1_C5', '22-06-10_M1_P1_C1']
# specifs = ['23-12-03_M1_P1_C3', '23-12-03_M1_P1_C6', '23-12-03_M1_P1_C11', '23-12-03_M1_P1_C13', '23-12-03_M1_P1_C17', '23-12-03_M1_P1_C20', '23-12-03_M1_P1_C21']
specifs = ['24-07-04_M2_P1_C2', '24-07-04_M2_P1_C8', '24-07-04_M2_P1_C13']
dirPath = "D:/MagneticPincherData/Data_Timeseries/Timeseries_stress-strain"
allFiles = os.listdir(dirPath)
selectedFiles = []

for f in allFiles:
    if SelectFile(f, specifs):
        selectedFiles.append(f)
        
ts_list = []
names_list = []
for f in selectedFiles:
    path = os.path.join(dirPath, f)
    ts_list.append(pd.read_csv(path, sep=None, engine='python'))
    names_list.append(f.split('_')[3])


fig, axes = strainRateCheck_V2(ts_list, names_list)
ufun.archiveFig(fig, name = 'ExampleStrainRates_24-07-04_M2', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = 'ViscoElasticity', cloudSave = 'flexible')


# %% Plots - All controls

# %%% bestH0

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug'])
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['none', 'dmso']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 0.75,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'none' : 'No drug',
      'dmso' : 'DMSO', 
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
# ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
#                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%% E_400

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
substrate = '20um fibronectin discs'
parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug'])
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['none', 'dmso']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 0.75,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'none' : 'No drug',
      'dmso' : 'DMSO', 
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
# ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
#                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %% Plots - All controls - by date

# %%% bestH0

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug', 'date'])
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['normal field'] == 5),
           (df['drug'] == 'none'),
           ]
df_f1 = filterDf(df, Filters)

Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['normal field'] == 5),
           (df['drug'] == 'dmso'),
           ]
df_f2 = filterDf(df, Filters)

# Order
# co_order = ['none', 'dmso']
co_order = []

# Group By
df_fg1 = dataGroup(df_f1, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fg2 = dataGroup(df_f2, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(2,1, figsize=(16, 8))
fig, ax = D1Plot(df_fg1, fig = fig, ax = axes[0], condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 0.75,
                 stats=False, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
fig, ax = D1Plot(df_fg2, fig = fig, ax = axes[1], condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 0.75,
                 stats=False, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'none' : 'No drug',
      'dmso' : 'DMSO', 
      }
for ax in axes:
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
# ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
#                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%% E_400

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
substrate = '20um fibronectin discs'
parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'date'])
# figname = 'bestH0' + drugSuffix


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           (df['normal field'] == 5),
           (df['drug'] == 'none'),
           ]
df_f1 = filterDf(df, Filters)

Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           (df['normal field'] == 5),
           (df['drug'] == 'dmso'),
           ]
df_f2 = filterDf(df, Filters)

# Order
# co_order = ['none', 'dmso']
co_order = []

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fgw1 = dataGroup_weightedAverage(df_f1, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_fgw2 = dataGroup_weightedAverage(df_f2, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

# Plot
fig, axes = plt.subplots(2,1, figsize=(16, 8))
fig, ax = D1Plot(df_fgw1, fig = fig, ax = axes[0], condition=condCol, parameter=parameter + '_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 0.75,
                 stats=False, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
fig, ax = D1Plot(df_fgw2, fig = fig, ax = axes[1], condition=condCol, parameter=parameter + '_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 0.75,
                 stats=False, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'none' : 'No drug',
      'dmso' : 'DMSO', 
      }

for ax in axes:
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
# ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
#                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %% Plots - Repeated compressions

# %%% Check
# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
dates = ['23-03-09', '23-09-19', '23-11-26', '23-12-03']
substrate = '20um fibronectin discs'
parameter = 'ctFieldThickness'
df, condCol = makeCompositeCol(df, cols=['date'])
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['date'].apply(lambda x : x in dates)),
           (df['bestH0'] <= 900),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['23-03-09', '23-09-19', '23-11-26', '23-12-03']


# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.2,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'none' : 'No drug',
      'dmso' : 'DMSO', 
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
# ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
#                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%% Manuscript plot - repeated compressions

gs.set_mediumText_options_jv()
figSubDir = ''

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
# dates = ['23-09-19', '23-11-26', '23-12-03', '23-02-23', ]
dates = ['23-12-03']
# dates = ['23-03-09']
suffix = '_3'
substrate = '20um fibronectin discs'
color = 'lightseagreen' #'red' # 'lightseagreen'

parameter = 'bestH0'

# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['date'].apply(lambda x : x in dates)),
           (df['bestH0'] <= 900),
           (df['compNum'] <= 5),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           (df['cellName'].apply(lambda x : not '-2' in x)),
           ]
df_f = filterDf(df, Filters)

# New data
# ManipTime
df_f['ManipTime'] = df_f['compAbsStartTime']
cellID_list = df_f['cellID'].unique()
manipID_list = df_f['manipID'].unique()
for mid in manipID_list:
    index_mid = df_f[df_f['manipID']==mid].index
    firstManipTime = np.min(df_f[df_f['manipID']==mid]['compAbsStartTime'].values)
    df_f.loc[index_mid, 'ManipTime'] -= firstManipTime
df_f['ManipTime'] /= 60

# relative H0
df_f['relative H0'] = df_f['bestH0']
for cid in cellID_list:
    index_cid = df_f[df_f['cellID']==cid].index
    firstH0 = df_f[df_f['cellID']==cid]['bestH0'].values[0]
    df_f.loc[index_cid, 'relative H0'] /= firstH0

# relative E
df_f['relative E'] = df_f['E_f_<_400']
for cid in cellID_list:
    index_cid = df_f[df_f['cellID']==cid].index
    firstE = df_f[df_f['cellID']==cid]['E_f_<_400'].values[0]
    df_f.loc[index_cid, 'relative E'] /= firstE
    
# group
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], numCols = ['bestH0'], aggFun = 'mean')
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

df_fgw2['E_f_<_400_wAvg'] /= 1000

fig, axes = plt.subplots(2, 2, figsize=(17/cm_in, 10/cm_in), sharex='col')
#### Plot 1
ax = axes[0, 0]
sns.scatterplot(ax=ax, data=df_f, x='compNum', y='relative H0', hue='cellID', alpha=0.8)
for cid in cellID_list:
    X = df_f[df_f['cellID']==cid]['compNum'].values
    Y = df_f[df_f['cellID']==cid]['relative H0'].values
    ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
ax.set_xlabel('Compression #')
ax.set_ylabel('relative $H_0$')
tickloc = matplotlib.ticker.MultipleLocator(1)
ax.xaxis.set_major_locator(tickloc)
ax.grid(visible=True, which='major', axis='y')

ax = axes[1, 0]
sns.scatterplot(ax=ax, data=df_f, x='compNum', y='relative E', hue='cellID', alpha=0.8)
for cid in cellID_list:
    X = df_f[df_f['cellID']==cid]['compNum'].values
    Y = df_f[df_f['cellID']==cid]['relative E'].values
    ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
ax.set_xlabel('Compression #')
ax.set_ylabel('relative $E$')
tickloc = matplotlib.ticker.MultipleLocator(1)
ax.xaxis.set_major_locator(tickloc)
ax.grid(visible=True, which='major', axis='y')

# # Prettify
# rD = {'manipID' : 'Exp #',
#       }
# for ax in axes:
#     renameAxes(ax, rD, format_xticks = True)
#     renameLegend(ax, rD)
#     ax.grid(visible=True, which='major', axis='y')
    
# # Save
# figname = 'cortex_vs_compnum' + suffix
# ufun.archiveFig(fig, name = figname, ext = '.pdf', dpi = 100,
#                 figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

#### Plot 2
ax = axes[0, 1]
sns.scatterplot(ax=ax, data=df_fg, x='ManipTime', y='bestH0', color=color, alpha=0.8)
ax.set_xlabel('Time from experiment start (min)')
ax.set_ylabel('$H_0$ (nm)')
for mid in manipID_list:
    X = df_fg[df_fg['manipID']==mid]['ManipTime'].values
    Y = df_fg[df_fg['manipID']==mid]['bestH0'].values
    M = np.array([X,Y]).T
    M = ufun.sortMatrixByCol(M, col=0, direction = 1)
    [X, Y] = M.T
    ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
ax.grid(visible=True, which='major', axis='y')

ax = axes[1, 1]
sns.scatterplot(ax=ax, data=df_fgw2, x='ManipTime', y='E_f_<_400_wAvg', color=color, alpha=0.8)
ax.set_xlabel('Time from experiment start (min)')
ax.set_ylabel('$E$ (kPa)')
for mid in manipID_list:
    X = df_fgw2[df_fgw2['manipID']==mid]['ManipTime'].values
    Y = df_fgw2[df_fgw2['manipID']==mid]['E_f_<_400_wAvg'].values
    M = np.array([X,Y]).T
    M = ufun.sortMatrixByCol(M, col=0, direction = 1)
    [X, Y] = M.T
    ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
ax.grid(visible=True, which='major', axis='y')

# Prettify
rD = {'manipID' : 'Exp #',
      }
for ax in axes[:, 0]:
    ax.legend().set_visible(False)
    ax.grid(visible=True, which='major', axis='y')
    

for ax in axes[:, 1]:
    ax.legend().set_visible(False)
    ax.grid(visible=True, which='major', axis='y')


# Show
fig.suptitle(manipID_list[0])
plt.tight_layout()
plt.show()

# Save
figname = 'cortex_vs_time' + suffix
ufun.archiveFig(fig, name = figname, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%% Fast compressions






# %% Plots - Cell viscosity

# cellID = '22-06-10_M1_P1_C1'
cellID = '24-04-11_M3_P1_C1'
taka3.plotCellTimeSeriesData(cellID)


# %%% Load a dataset

def SelectFile(f, list_strings):
    res = False
    for s in list_strings:
        if f.startswith(s):
            res = True
    return(res)

def normalizeVect(A):
    B = (A-np.min(A))/(np.max(A)-np.min(A))
    return(B)


specifs = ['24-04-11_M3_P1_C1', '24-04-11_M3_P1_C2', '24-04-11_M3_P1_C4'] #, '24-04-11_M3_P1_C5', '22-06-10_M1_P1_C1']
dirPath = "D:/MagneticPincherData/Data_Timeseries/Timeseries_stress-strain"
allFiles = os.listdir(dirPath)
selectedFiles = []

for f in allFiles:
    if SelectFile(f, specifs):
        selectedFiles.append(f)
        
ts_list = []
names_list = []
for f in selectedFiles:
    path = os.path.join(dirPath, f)
    ts_list.append(pd.read_csv(path, sep=None, engine='python'))
    names_list.append(f)
    
# %%% function to check epsilon of t 2

def strainRateCheck(tsdf, name = '', normalize = False):
    Nc = np.max(tsdf['idxAnalysis'])
    coi = ['T', 'B', 'F', 'H0', 'Stress', 'Strain']
    
    fig, axes = plt.subplots(Nc, 2, figsize=(8*2/gs.cm_in, 4*Nc/gs.cm_in), sharex='col')
    
    rate_matrix = []
    
    for i in range(1, Nc+1):
        df_i = tsdf[tsdf['idxAnalysis'] == i][coi].copy()
        df_i = df_i[df_i['F'] <= 400]
        df_i = df_i.dropna()
        
        x, y = df_i['T']-df_i['T'].values[0], df_i['Strain']
            
        tck_s = interpolate.splrep(x, y, s=1)
        Bspl = interpolate.BSpline(*tck_s)
        Bspl_der = Bspl.derivative(1)
        
        rate_matrix.append(Bspl_der(x))
        
        ax = axes[i-1, 0]
        ax.plot(x, y)
        # ax.plot(x, Bspl(x))
        ax.set_ylim([0, 0.2])
        
        ax = axes[i-1, 1]
        ax.plot(x, Bspl_der(x))
        ax.set_ylim([0, 0.2])
    
    
    # rate_matrix = np.array(rate_matrix).reshape((Nc, 200))
    # rate_avg = np.mean(rate_matrix, axis=0)
    # fig_avg, ax_avg = plt.subplots(1, 1, figsize=(8/gs.cm_in, 4/gs.cm_in))
    # ax_avg.plot(x, rate_avg)
    # fig_avg.tight_layout()
    
    fig.tight_layout()
    plt.show()
    
    # ufun.archiveFig(fig, name = name + '_allStrainRates', ext = '.pdf', dpi = 100,
    #                 figDir = figDir, figSubDir = '', cloudSave = 'flexible')
    
    # ufun.archiveFig(fig_avg, name = name + '_avgStrainRates', ext = '.pdf', dpi = 100,
    #                 figDir = figDir, figSubDir = '', cloudSave = 'flexible')
        
    

for tsdf, name in zip(ts_list[:], names_list[:]):
    strainRateCheck(tsdf, name = name, normalize = False)



# %%% Many

def SelectFile(f, list_strings):
    res = False
    for s in list_strings:
        if f.startswith(s):
            res = True
    return(res)

def strainRateCheck_V2(list_tsdf, names_list):
    gs.set_manuscript_options_jv('Set1')
    N = len(list_tsdf)
    
    fig, axes = plt.subplots(N, 2, figsize=(17/gs.cm_in, 5*N/gs.cm_in), sharex='col')
    
    for k in range(N):
        tsdf = list_tsdf[k]
        cellName = names_list[k]
        Nc = min(np.max(tsdf['idxAnalysis']), 5)
        coi = ['T', 'B', 'F', 'H0', 'Stress', 'Strain']
        
        rate_matrix = []
        
        for i in range(1, Nc+1):
            df_i = tsdf[tsdf['idxAnalysis'] == i][coi].copy()
            df_i = df_i.dropna()
            # i400 = ufun.findFirst(True, df_i['F'].values>400)
            # t400 = df_i['T'].values[i400]-df_i['T'].values[0]
            
            x, y = df_i['T']-df_i['T'].values[0], df_i['Strain']
                
            tck_s = interpolate.splrep(x, y, s=0.01)
            Bspl = interpolate.BSpline(*tck_s)
            Bspl_der = Bspl.derivative(1)
            rate_matrix.append(Bspl_der(x))
            
            ax = axes[k, 0]
            ax.plot(x, y)
            # ax.axvline(t400, color='gray', alpha=0.5, ls='--')
            ax.set_ylim([0, 0.25])
            ax.set_ylabel(cellName, fontsize = 10)
            tickloc = matplotlib.ticker.MultipleLocator(0.05)
            ax.yaxis.set_major_locator(tickloc)
            ax.grid(axis='y')
            
            ax = axes[k, 1]
            ax.plot(x, Bspl_der(x), label=f'#{i:.0f} - $H_0$ = {df_i.H0.values[0]:.0f} nm',)
            ax.legend(fontsize=6, loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_ylim([0, 0.25])
            tickloc = matplotlib.ticker.MultipleLocator(0.05)
            ax.yaxis.set_major_locator(tickloc)
            ax.grid(axis='y')
    
    axes[0,0].set_title('Strain - $\epsilon (t)$')
    axes[0,1].set_title('Strain rate - $d\epsilon /dt$ $(s^{-1})$')
    axes[-1,0].set_xlabel('Time (s)')
    axes[-1,1].set_xlabel('Time (s)')
    
    fig.tight_layout()
    plt.show()
    
    # ufun.archiveFig(fig, name = name + '_allStrainRates', ext = '.pdf', dpi = 100,
    #                 figDir = figDir, figSubDir = '', cloudSave = 'flexible')
    
    return(fig, axes)



# specifs = ['23-03-09_M4_P1_C3', '23-03-09_M4_P1_C4', '23-03-09_M4_P1_C5'] #, '24-04-11_M3_P1_C5', '22-06-10_M1_P1_C1']
# specifs = ['23-12-03_M1_P1_C3', '23-12-03_M1_P1_C6', '23-12-03_M1_P1_C11', '23-12-03_M1_P1_C13', '23-12-03_M1_P1_C17', '23-12-03_M1_P1_C20', '23-12-03_M1_P1_C21']
specifs = ['24-07-04_M2_P1_C2', '24-07-04_M2_P1_C8', '24-07-04_M2_P1_C13']
dirPath = "D:/MagneticPincherData/Data_Timeseries/Timeseries_stress-strain"
allFiles = os.listdir(dirPath)
selectedFiles = []

for f in allFiles:
    if SelectFile(f, specifs):
        selectedFiles.append(f)
        
ts_list = []
names_list = []
for f in selectedFiles:
    path = os.path.join(dirPath, f)
    ts_list.append(pd.read_csv(path, sep=None, engine='python'))
    names_list.append(f.split('_')[3])


fig, axes = strainRateCheck_V2(ts_list, names_list)
ufun.archiveFig(fig, name = 'ExampleStrainRates_24-07-04_M2', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = 'ViscoElasticity', cloudSave = 'flexible')

# %% Plots - Cell identity

# %%% Thickness

# %%%% Define

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
substrate = '20um fibronectin discs'

df, condCol = makeCompositeCol(df, cols=['date', 'drug'])
parameter = 'bestH0'
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['normal field'] == 5),
           (df['bestH0'] <= 1000),
           (df['date'] == '23-03-09'),
           ]

df_f = filterDf(df, Filters)

# Order
co_order = []

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
df_fg[parameter + '_pop_spread'] = df_fg[parameter] / np.mean(df_fg[parameter].values)

# Merge
df_m = pd.merge(left=df_f, right=df_fg, on = 'cellID', how='left', suffixes = (None, '_grouped'))
df_m[parameter + '_indiv_spread'] = df_m[parameter] / df_m[parameter + '_grouped']

# Sort by cellID
df_f.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_fg.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_m.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)


# %%%% Normality tests

data_lin = df_fg[parameter].values
data_log = np.log(df_fg[parameter].values)

fig_test, axes_test = plt.subplots(1, 2, figsize = (12, 5))

data=data_lin
ax = axes_test[0]
shap_stat, shap_pval = shapiro(data)
sm.qqplot(data, fit=True, line='45', ax=ax)
ax.plot([], [], label=f'Shapiro–Wilk p-value = {shap_pval:.2f}', ls='', marker='o')
ax.legend(fontsize=11)
ax.set_title(f'Normality test and\nQ-Q plot of parm: {parameter}', fontsize=11)

data=data_log
ax = axes_test[1]
shap_stat, shap_pval = shapiro(data)
sm.qqplot(data, fit=True, line='45', ax=ax)
ax.plot([], [], label=f'Shapiro–Wilk p-value = {shap_pval:.2f}', ls='', marker='o')
ax.legend(fontsize=11)
ax.set_title(f'Log-normality test and\nQ-Q plot of parm: {parameter}', fontsize=11)

fig_test.tight_layout()
plt.show()


# %%%% One way ANOVA
list_cellIDs = df_f['cellID'].unique()
values_per_cell = [df_f.loc[df_f['cellID'] == ID, parameter].values for ID in list_cellIDs]

f_oneway(*values_per_cell)


# %%%% Compute CV by cell and by pop
list_cellIDs = df_f['cellID'].unique()
values_per_cell = [df_f.loc[df_f['cellID'] == ID, parameter].values for ID in list_cellIDs]
CV_per_cell = [np.std(cell_vals) / np.mean(cell_vals) for cell_vals in values_per_cell]
CV_per_cell_avg = np.mean(CV_per_cell)
CV_population = np.std(df_fg[parameter]) / np.mean(df_fg[parameter])

print('CV of all cells: ', *[f'{cv:.4f}' for cv in CV_per_cell])
print(f'Mean CV per cell: {CV_per_cell_avg:.4f}')
print(f'CV of full pop: {CV_population:.4f}')



# %%%% Plots

# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig = plt.figure(figsize=(14, 8))
spec = fig.add_gridspec(2, 2)


# Plot 1
ax = fig.add_subplot(spec[0, 0])

boxplot_parameters = {'width' : 0.5, 'showfliers': False}
boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 2.0, 'alpha' : 0.8, 'zorder' : 2},
                        boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2},
                        whiskerprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2},
                        capprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2})

sns.swarmplot(ax = ax, data = df_f, x = 'date', y = parameter, hue='cellID')
sns.boxplot(ax = ax, data = df_f, x = 'date', y = parameter, **boxplot_parameters)

ax.legend().set_visible(False)
ax.grid(visible=True, which='major', axis='y')
ax.set_ylim([0, ax.get_ylim()[1]])

print(df_f.cellID.unique())


# Plot 2
ax = fig.add_subplot(spec[1, :])

# boxplot_parameters = {'width' : 0.5, 'showfliers': False}
# boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 2.0, 'alpha' : 0.8, 'zorder' : 2},
#                         boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2},
#                         whiskerprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2},
#                         capprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2})

# sns.boxplot(ax = ax, data = df_m, x = 'cellID', y = parameter + '_spread', **boxplot_parameters)

sns.swarmplot(ax = ax, data = df_m, x = 'cellID', y = parameter + '_indiv_spread', hue = 'cellID')

ax.set_ylim([0, ax.get_ylim()[1]*1.1])
ax.legend().set_visible(False)
ax.grid(visible=True, which='major', axis='y')
renameAxes(ax, renameDict, format_xticks = True, rotation = 30)

print(df_m.cellID.unique())

# # Plot 3
ax = fig.add_subplot(spec[0, 1])

sns.swarmplot(ax = ax, data = df_fg, x = 'date', y = parameter + '_pop_spread', hue = 'cellID')

ax.set_ylim([0, 2])
ax.legend().set_visible(False)
ax.grid(visible=True, which='major', axis='y')
# renameAxes(ax, renameDict, format_xticks = True, rotation = 30)

print(df_fg.cellID.unique())

# Prettify
# rD = {'none' : 'No drug',
#       'dmso' : 'DMSO', 
#       }

# renameAxes(ax, rD, format_xticks = True)
# renameAxes(ax, renameDict, format_xticks = True, rotation = 30)
# renameLegend(ax, renameDict)
# ax.grid(visible=True, which='major', axis='y')
# ax.set_xlabel('')

# Count
# CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
fig.suptitle('Thickness')
fig.tight_layout()
plt.show()


# %%% Stiffness

# %%%% Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
substrate = '20um fibronectin discs'

df, condCol = makeCompositeCol(df, cols=['date', 'drug'])
parameter = 'E_f_<_400'
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['normal field'] == 5),
           (df['E_f_<_400'] <= 1e4),
           (df['valid_f_<_400'] == True),
           (df['date'] == '23-03-09'),
           ]

df_f = filterDf(df, Filters)

# Order
# co_order = ['none', 'dmso']
co_order = []

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
df_fg[parameter + '_pop_spread'] = df_fg[parameter] / np.mean(df_fg[parameter].values)
# Merge
df_m = pd.merge(left=df_f, right=df_fg, on = 'cellID', how='left', suffixes = (None, '_grouped'))
df_m[parameter + '_indiv_spread'] = df_m[parameter] / df_m[parameter + '_grouped']

# Sort by cellID
df_f.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_fg.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_m.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)


# %%%% Normality tests

data_lin = df_fg[parameter].values
data_log = np.log(df_fg[parameter].values)

fig_test, axes_test = plt.subplots(1, 2, figsize = (12, 5))

data=data_lin
ax = axes_test[0]
shap_stat, shap_pval = shapiro(data)
sm.qqplot(data, fit=True, line='45', ax=ax)
ax.plot([], [], label=f'Shapiro–Wilk p-value = {shap_pval:.2f}', ls='', marker='o')
ax.legend(fontsize=11)
ax.set_title(f'Normality test and\nQ-Q plot of parm: {parameter}', fontsize=11)

data=data_log
ax = axes_test[1]
shap_stat, shap_pval = shapiro(data)
sm.qqplot(data, fit=True, line='45', ax=ax)
ax.plot([], [], label=f'Shapiro–Wilk p-value = {shap_pval:.2f}', ls='', marker='o')
ax.legend(fontsize=11)
ax.set_title(f'Log-normality test and\nQ-Q plot of parm: {parameter}', fontsize=11)

fig_test.tight_layout()
plt.show()


# %%%% Apply log

# df_f[parameter] = np.log10(df_f[parameter])
df_f[parameter] = np.log(df_f[parameter])

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean').reset_index(drop=True)
df_fg[parameter + '_pop_spread'] = df_fg[parameter] / np.mean(df_fg[parameter].values)

# Merge
df_m = pd.merge(left=df_f, right=df_fg, on = 'cellID', how='left', suffixes = (None, '_grouped'))
df_m[parameter + '_indiv_spread'] = df_m[parameter] / df_m[parameter + '_grouped']

# Sort by cellID
df_f.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_fg.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)
df_m.sort_values('cellID', axis=0, ascending=True, inplace=True, 
                 kind='quicksort', na_position='last', ignore_index=False, key=None)


# %%%% One way ANOVA
list_cellIDs = df_f['cellID'].unique()
values_per_cell = [df_f.loc[df_f['cellID'] == ID, parameter].values for ID in list_cellIDs]

f_oneway(*values_per_cell)


# %%%% Compute CV by cell and by pop
list_cellIDs = df_f['cellID'].unique()
values_per_cell = [df_f.loc[df_f['cellID'] == ID, parameter].values for ID in list_cellIDs]
CV_per_cell = [np.std(cell_vals) / np.mean(cell_vals) for cell_vals in values_per_cell]
CV_per_cell_avg = np.mean(CV_per_cell)
CV_population = np.std(df_fg[parameter]) / np.mean(df_fg[parameter])

print('CV of all cells: ', *[f'{cv:.4f}' for cv in CV_per_cell])
print(f'Mean CV per cell: {CV_per_cell_avg:.4f}')
print(f'CV of full pop: {CV_population:.4f}')



# %%%% Plots

# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig = plt.figure(figsize=(14, 8))
spec = fig.add_gridspec(2, 2)


# Plot 1
ax = fig.add_subplot(spec[0, 0])

boxplot_parameters = {'width' : 0.5, 'showfliers': False}
boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 2.0, 'alpha' : 0.8, 'zorder' : 2},
                        boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2},
                        whiskerprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2},
                        capprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2})

sns.swarmplot(ax = ax, data = df_f, x = 'date', y = parameter, hue='cellID')
sns.boxplot(ax = ax, data = df_f, x = 'date', y = parameter, **boxplot_parameters)

ax.legend().set_visible(False)
ax.grid(visible=True, which='major', axis='y')
# ax.set_ylim([0, ax.get_ylim()[1]])

print(df_f.cellID.unique())


# Plot 2
ax = fig.add_subplot(spec[1, :])

# boxplot_parameters = {'width' : 0.5, 'showfliers': False}
# boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 2.0, 'alpha' : 0.8, 'zorder' : 2},
#                         boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2},
#                         whiskerprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2},
#                         capprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2})

# sns.boxplot(ax = ax, data = df_m, x = 'cellID', y = parameter + '_spread', **boxplot_parameters)

sns.swarmplot(ax = ax, data = df_m, x = 'cellID', y = parameter + '_indiv_spread', hue = 'cellID')

# ax.set_ylim([0, 2])
ax.legend().set_visible(False)
ax.grid(visible=True, which='major', axis='y')
renameAxes(ax, renameDict, format_xticks = True, rotation = 30)

print(df_m.cellID.unique())

# # Plot 3
ax = fig.add_subplot(spec[0, 1])

sns.swarmplot(ax = ax, data = df_fg, x = 'date', y = parameter + '_pop_spread', hue = 'cellID')

# ax.set_ylim([0, 2])
ax.legend().set_visible(False)
ax.grid(visible=True, which='major', axis='y')
# renameAxes(ax, renameDict, format_xticks = True, rotation = 30)

print(df_fg.cellID.unique())

# Prettify
# rD = {'none' : 'No drug',
#       'dmso' : 'DMSO', 
#       }

# renameAxes(ax, rD, format_xticks = True)
# renameAxes(ax, renameDict, format_xticks = True, rotation = 30)
# renameLegend(ax, renameDict)
# ax.grid(visible=True, which='major', axis='y')
# ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
fig.suptitle('Stiffness - Log10 values')
fig.tight_layout()
plt.show()


# %%% Test


# Importing library
from scipy.stats import f_oneway, shapiro
 
# Performance when each of the engine 
# oil is applied
performance1 = [89, 89, 88, 78, 79]
performance2 = [93, 92, 94, 89, 88]
performance3 = [20, 21, 22, 23, 24]
performance4 = [ 1,  2,  3,  4,  5]


# Conduct the one-way ANOVA
f_oneway(performance1, performance2, performance3, performance4)

 
# # Performance when each of the engine 
# # oil is applied
# performance1 = [89, 89, 88, 87, 92, 95, 94, 56]
# performance2 = [93, 92, 94, 89, 88]
# performance3 = [89, 88, 89, 93, 90]
# performance4 = [91, 88, 91, 92, 92]

 
# # Conduct the one-way ANOVA
# f_oneway(performance1, performance2, performance3, performance4)





# %% Plots - Cortex solidity

# %%% Delta

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
substrate = '20um fibronectin discs'

df, condCol = makeCompositeCol(df, cols=['date', 'drug'])
# dates = ['23-03-16','23-03-17']
# figname = 'bestH0' + drugSuffix

# New parameter
df['dH'] = df['initialThickness'] - df['previousThickness']
df['H0/H5'] = df['initialThickness'] / df['previousThickness']
df['dH/H5'] = df['dH'] / df['previousThickness']
parameter = 'dH'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['drug'] == 'dmso'),
           # (df['date'].apply(lambda x : x in dates)),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['normal field'] == 5),
           (df['ramp field'].apply(lambda x : x.startswith('2.5'))),
           (df['bestH0'] <= 900),
           (df['dH/H5'] <= 3),
           # (df['dH/H5'] >= 0.5),
           ]

df_f = filterDf(df, Filters)

# Order
# co_order = ['none', 'dmso']
co_order = []

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1, 1, figsize=(17/gs.cm_in, 10/gs.cm_in))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 2, markersizeFactor = 1.25,
                 stats=False, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'none' : 'No drug',
      'dmso' : 'DMSO', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True, rotation = 30)
# renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()



# %%% Ratio

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
substrate = '20um fibronectin discs'

df, condCol = makeCompositeCol(df, cols=['date', 'drug'])
# figname = 'bestH0' + drugSuffix

# New parameter
df['dH_relax'] = df['initialThickness'] - df['previousThickness']
df['dH_ratio'] = df['initialThickness'] / df['previousThickness']
# df['dH_ratio'] = df['dH_relax'] / df['previousThickness']
parameter = 'dH_ratio'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['normal field'] == 5),
           (df['ramp field'].apply(lambda x : x.startswith('2.5'))),
           (df['bestH0'] <= 800),
           (df['dH_ratio'] <= 3),
           (df['dH_ratio'] >= 0.5),
           ]

df_f = filterDf(df, Filters)

# Order
# co_order = ['none', 'dmso']
co_order = []

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(12, 4))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 2, markersizeFactor = 1.25,
                 stats=False, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
rD = {'none' : 'No drug',
      'dmso' : 'DMSO', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True, rotation = 30)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()





# %%% 2D

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
substrate = '20um fibronectin discs'

df, condCol = makeCompositeCol(df, cols=['date', 'drug'])
# figname = 'bestH0' + drugSuffix

# New parameter
df['dH_relax'] = df['initialThickness'] - df['previousThickness']
df['dH_ratio'] = df['initialThickness'] / df['previousThickness']
# df['dH_ratio'] = df['dH_relax'] / df['previousThickness']
parameter = 'dH_ratio'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['valid_f_<_400'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['normal field'] == 5),
           # (df['ramp field'].apply(lambda x : x.startswith('2.5_'))),
            (df['ramp field'].apply(lambda x : x.startswith('2_'))),
           (df['valid_f_<_400'] == True), 
           (df['dH_ratio'] <= 3),
           (df['dH_ratio'] >= 0.5),
           ]

df_f = filterDf(df, Filters)

# Order
# co_order = ['none', 'dmso']
co_order = []

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol, 'date'], 
                  numCols = ['bestH0', parameter], aggFun = 'mean').reset_index(drop=True)

df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')

df_m = pd.merge(df_fg, df_fgw2, on='cellID', suffixes=(None, '_y'))
df_m['E_f_<_400_wAvg'] /= 1000
Filters = [(df_m['E_f_<_400_wAvg'] <= 14),
           ]

df_mf = filterDf(df_m, Filters)

# Plot
fig, ax = plt.subplots(1,1, figsize=(12, 8))
# fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
#                  co_order = co_order, boxplot = 2, figSizeFactor = 2, markersizeFactor = 1.25,
#                  stats=False, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
#                  showMean = False)
sns.scatterplot(ax=ax, data=df_mf, x = 'E_f_<_400_wAvg', y = 'dH_ratio', hue = 'date', s=50, palette='Set2')

params, results = ufun.fitLineHuber(df_mf['E_f_<_400_wAvg'].values, df_mf['dH_ratio'].values)
[b, a] = params
fitX = np.linspace(min(df_mf['E_f_<_400_wAvg'].values), max(df_mf['E_f_<_400_wAvg'].values), 10)
fitY = a*fitX + b
ax.plot(fitX, fitY, ls='--', c='k', label = f'{a:.3f} x + {b:.1f}')
           
# Prettify
rD = {'none' : 'No drug',
      'dmso' : 'DMSO', 
      }

# renameAxes(ax, rD, format_xticks = True)
# renameAxes(ax, renameDict, format_xticks = True, rotation = 30)
renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
# ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()


# %%% In time

gs.set_mediumText_options_jv()
figSubDir = ''

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
# dates = ['23-09-19', '23-11-26', '23-12-03', '23-02-23', ]
# dates = ['23-12-03']
dates = ['23-03-09']
# dates = ['23-03-17']
# dates = ['23-04-28']
suffix = '_3'
substrate = '20um fibronectin discs'
color = 'lightseagreen' #'red' # 'lightseagreen'

parameter = 'bestH0'

# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['date'].apply(lambda x : x in dates)),
           (df['bestH0'] <= 900),
           (df['compNum'] <= 10),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           (df['cellName'].apply(lambda x : not '-2' in x)),
           ]
df_f = filterDf(df, Filters)

# New data
# dH
df_f['dH_relax'] = df_f['initialThickness'] - df_f['previousThickness']
df_f['dH_ratio'] = df_f['initialThickness'] / df_f['previousThickness']
# df['dH_ratio'] = df['dH_relax'] / df['previousThickness']
parameter = 'dH_ratio'

# ManipTime
df_f['ManipTime'] = df_f['compAbsStartTime']
cellID_list = df_f['cellID'].unique()
manipID_list = df_f['manipID'].unique()
for mid in manipID_list:
    index_mid = df_f[df_f['manipID']==mid].index
    firstManipTime = np.min(df_f[df_f['manipID']==mid]['compAbsStartTime'].values)
    df_f.loc[index_mid, 'ManipTime'] -= firstManipTime
df_f['ManipTime'] /= 60


# Re-filter
Filters = [(df_f['dH_ratio'] <= 3),
           (df_f['dH_ratio'] >= 0.5),
           ]

df_f = filterDf(df_f, Filters)

    
# group
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], numCols = ['bestH0', 'dH_relax', 'dH_ratio'], aggFun = 'mean')


fig, axes = plt.subplots(2, 1, figsize=(10/cm_in, 12/cm_in))
PLOT_FIT = True

#### Plot 1
ax = axes[0]
sns.scatterplot(ax=ax, data=df_f, x='compNum', y='dH_ratio', hue='cellID', alpha=0.8, legend=False)
for cid in cellID_list:
    X = df_f[df_f['cellID']==cid]['compNum'].values
    Y = df_f[df_f['cellID']==cid]['dH_ratio'].values
    ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
ax.set_xlabel('Compression #')
ax.set_ylabel('$\delta H$ / $H_{5mT}$')
tickloc = matplotlib.ticker.MultipleLocator(1)
ax.xaxis.set_major_locator(tickloc)
ax.grid(visible=True, which='major', axis='y')

if PLOT_FIT:
    params, results = ufun.fitLineHuber(df_f['compNum'].values, df_f['dH_ratio'].values)
    [b, a] = params
    pval = results.pvalues[1]
    fitX = np.linspace(min(df_mf['compNum'].values), max(df_mf['compNum'].values), 10)
    fitY = a*fitX + b
    ax.plot(fitX, fitY, ls='--', c='k', lw=1,
            label = f'{a:.3f} x + {b:.1f} - p-val = {pval:.2f}')

#### Plot 2
ax = axes[1]
sns.scatterplot(ax=ax, data=df_fg, x='ManipTime', y='dH_ratio', color=color, alpha=0.8)
ax.set_xlabel('Time from experiment start (min)')
ax.set_ylabel('$\delta H$ / $H_{5mT}$')
for mid in manipID_list:
    X = df_fg[df_fg['manipID']==mid]['ManipTime'].values
    Y = df_fg[df_fg['manipID']==mid]['dH_ratio'].values
    M = np.array([X,Y]).T
    M = ufun.sortMatrixByCol(M, col=0, direction = 1)
    [X, Y] = M.T
    ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
ax.grid(visible=True, which='major', axis='y')

if PLOT_FIT:
    params, results = ufun.fitLineHuber(df_fg['ManipTime'].values, df_fg['dH_ratio'].values)
    [b, a] = params
    pval = results.pvalues[1]
    fitX = np.linspace(min(df_fg['ManipTime'].values), max(df_fg['ManipTime'].values), 10)
    fitY = a*fitX + b
    ax.plot(fitX, fitY, ls='--', c='k', lw=1,
            label = f'{a:.3f} x + {b:.1f} - p-val = {pval:.2f}')

# Prettify
rD = {'manipID' : 'Exp #',
      'dH_ratio' : '$\delta H$ / $H_H_{5mT}$'
      }
for ax in axes:
    ax.legend().set_visible(True)
    ax.grid(visible=True, which='major', axis='y')

# Show
fig.suptitle(manipID_list[0])
plt.tight_layout()
plt.show()

# Save
# figname = 'cortex_vs_time' + suffix
# ufun.archiveFig(fig, name = figname, ext = '.pdf', dpi = 100,
#                 figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%% In time - many dates - only CompNum

gs.set_mediumText_options_jv()
figSubDir = ''

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
# dates = ['23-09-19', '23-11-26', '23-12-03', '23-02-23', '23-12-03', '23-03-09', '23-03-17', '23-04-28']
suffix = '_allComps'
substrate = '20um fibronectin discs'
color = 'lightseagreen' #'red' # 'lightseagreen'

parameter = 'bestH0'

# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           # (df['date'].apply(lambda x : x in dates)),
           (df['bestH0'] <= 900),
            (df['compNum'] <= 10),
           (df['compNum'] >= 1),
           # (df['E_f_<_400'] <= 2e4),
           # (df['valid_f_<_400'] == True), 
           (df['normal field'] == 5),
           (df['ramp field'].apply(lambda x : x.startswith('2.5'))),
           (df['cellName'].apply(lambda x : not '-2' in x)),
           ]
df_f = filterDf(df, Filters)

# New data
# dH
df_f['dH_relax'] = df_f['initialThickness'] - df_f['previousThickness']
df_f['dH_ratio'] = df_f['initialThickness'] / df_f['previousThickness']
# df['dH_ratio'] = df['dH_relax'] / df['previousThickness']
parameter = 'dH_ratio'

# ManipTime
df_f['ManipTime'] = df_f['compAbsStartTime']
cellID_list = df_f['cellID'].unique()
manipID_list = df_f['manipID'].unique()
for mid in manipID_list:
    index_mid = df_f[df_f['manipID']==mid].index
    firstManipTime = np.min(df_f[df_f['manipID']==mid]['compAbsStartTime'].values)
    df_f.loc[index_mid, 'ManipTime'] -= firstManipTime
df_f['ManipTime'] /= 60


# Re-filter
Filters = [(df_f['dH_ratio'] <= 3),
           (df_f['dH_ratio'] >= 0.5),
           ]

df_f = filterDf(df_f, Filters)

    
# group
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], numCols = ['bestH0', 'dH_relax', 'dH_ratio'], aggFun = 'mean')


fig, ax = plt.subplots(1, 1, figsize=(17/cm_in, 10/cm_in))

PLOT_FIT = True

#### Plot 1
ax = ax
cL = gs.cL_Set21

ax.set_xlabel('# Compression')
ax.set_ylabel('$\delta H$ / $H_{5mT}$')
for mid, color in zip(manipID_list, cL):
    df_fit = df_f[df_f['manipID']==mid].copy()
    cellID_list = df_fit['cellID'].unique()
    for cid in cellID_list:
        X = df_f[df_f['cellID']==cid]['compNum'].values
        Y = df_f[df_f['cellID']==cid]['dH_ratio'].values
        ax.scatter(X, Y, color='None', edgecolor=color, alpha=1.0)
        ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0, alpha=0.5)
    
    if PLOT_FIT:
        params, results = ufun.fitLineHuber(df_fit['compNum'].values, df_fit['dH_ratio'].values)
        [b, a] = params
        pval = results.pvalues[1]
        fitX = np.linspace(min(df_fit['compNum'].values), max(df_fit['compNum'].values), 10)
        fitY = a*fitX + b
        ax.plot(fitX, fitY, ls='--', c=color, lw=2,
                label = f'{mid}\n{a:.3f} x + {b:.1f} - p-val = {pval:.2f}')
    
ax.grid(visible=True, which='major', axis='y')

if PLOT_FIT:
    params, results = ufun.fitLineHuber(df_f['compNum'].values, df_f['dH_ratio'].values)
    [b, a] = params
    pval = results.pvalues[1]
    fitX = np.linspace(min(df_f['compNum'].values), max(df_f['compNum'].values), 10)
    fitY = a*fitX + b
    ax.plot(fitX, fitY, ls='--', c='k', lw=2,
            label = 'Global Fit\n' + f'{a:.3f} x + {b:.1f} - p-val = {pval:.2f}')

# Prettify
rD = {'manipID' : 'Exp #',
      'dH_ratio' : '$\delta H$ / $H_H_{5mT}$'
      }

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(visible=True, which='major', axis='y')
# ax.set_xlim([0.75, 2.25])

# Show
plt.tight_layout()
plt.show()

# Save
figname = 'relaxH_vs_compNum' + suffix
ufun.archiveFig(fig, name = figname, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%% In time - many dates - TEST E400

gs.set_mediumText_options_jv()
figSubDir = ''

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
# dates = ['23-09-19', '23-11-26', '23-12-03', '23-02-23', '23-12-03', '23-03-09', '23-03-17', '23-04-28']
suffix = ''
substrate = '20um fibronectin discs'
color = 'lightseagreen' #'red' # 'lightseagreen'

parameter = 'bestH0'

# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           # (df['date'].apply(lambda x : x in dates)),
           (df['bestH0'] <= 900),
           (df['compNum'] <= 2),
           (df['compNum'] >= 1),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           (df['normal field'] == 5),
           (df['ramp field'].apply(lambda x : x.startswith('2.5'))),
           (df['cellName'].apply(lambda x : not '-2' in x)),
           ]
df_f = filterDf(df, Filters)

# New data
# dH
df_f['dH_relax'] = df_f['initialThickness'] - df_f['previousThickness']
df_f['dH_ratio'] = df_f['initialThickness'] / df_f['previousThickness']
# df['dH_ratio'] = df['dH_relax'] / df['previousThickness']
parameter = 'dH_ratio'

# ManipTime
df_f['ManipTime'] = df_f['compAbsStartTime']
cellID_list = df_f['cellID'].unique()
manipID_list = df_f['manipID'].unique()
for mid in manipID_list:
    index_mid = df_f[df_f['manipID']==mid].index
    firstManipTime = np.min(df_f[df_f['manipID']==mid]['compAbsStartTime'].values)
    df_f.loc[index_mid, 'ManipTime'] -= firstManipTime
df_f['ManipTime'] /= 60


# Re-filter
Filters = [(df_f['dH_ratio'] <= 3),
           (df_f['dH_ratio'] >= 0.5),
           ]

df_f = filterDf(df_f, Filters)

    
# group
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], numCols = ['bestH0', 'dH_relax', 'dH_ratio'], aggFun = 'mean')


fig, ax = plt.subplots(1, 1, figsize=(17/cm_in, 10/cm_in))

PLOT_FIT = True

#### Plot 1
ax = ax
cL = gs.cL_Set12

ax.set_xlabel('# Compression')
# ax.set_ylabel('$\delta H$ / $H_{5mT}$')
for mid, color in zip(manipID_list, cL):
    df_fit = df_f[df_f['manipID']==mid].copy()
    cellID_list = df_fit['cellID'].unique()
    for cid in cellID_list:
        X = df_f[df_f['cellID']==cid]['compNum'].values
        Y = df_f[df_f['cellID']==cid]['E_f_<_400'].values
        ax.scatter(X, Y, color='None', edgecolor=color, alpha=1.0)
        ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0, alpha=0.5)
    
    if PLOT_FIT:
        params, results = ufun.fitLineHuber(df_fit['compNum'].values, df_fit['E_f_<_400'].values)
        [b, a] = params
        pval = results.pvalues[1]
        fitX = np.linspace(min(df_fit['compNum'].values), max(df_fit['compNum'].values), 10)
        fitY = a*fitX + b
        ax.plot(fitX, fitY, ls='--', c=color, lw=2,
                label = f'{mid}\n{a:.3f} x + {b:.1f} - p-val = {pval:.2f}')
    
ax.grid(visible=True, which='major', axis='y')

if PLOT_FIT:
    params, results = ufun.fitLineHuber(df_f['compNum'].values, df_f['E_f_<_400'].values)
    [b, a] = params
    pval = results.pvalues[1]
    fitX = np.linspace(min(df_f['compNum'].values), max(df_f['compNum'].values), 10)
    fitY = a*fitX + b
    ax.plot(fitX, fitY, ls='--', c='k', lw=2,
            label = 'Global Fit\n' + f'{a:.3f} x + {b:.1f} - p-val = {pval:.2f}')

# Prettify
rD = {'manipID' : 'Exp #',
      'dH_ratio' : '$\delta H$ / $H_{5mT}$'
      }
renameAxes(ax, renameDict, format_xticks = True, rotation = 0)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(visible=True, which='major', axis='y')

# Show
plt.tight_layout()
plt.show()

# Save
figname = 'E400_vs_compNum' + suffix
ufun.archiveFig(fig, name = figname, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')





# %%% In time - many dates - only ManipTime

gs.set_mediumText_options_jv()
figSubDir = ''

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
# dates = ['23-09-19', '23-11-26', '23-12-03', '23-02-23', '23-12-03', '23-03-09', '23-03-17', '23-04-28']
suffix = '_3'
substrate = '20um fibronectin discs'
color = 'lightseagreen' #'red' # 'lightseagreen'

parameter = 'bestH0'

# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           # (df['date'].apply(lambda x : x in dates)),
           (df['bestH0'] <= 900),
           (df['compNum'] <= 10),
           # (df['E_f_<_400'] <= 2e4),
           # (df['valid_f_<_400'] == True), 
           (df['normal field'] == 5),
           (df['ramp field'].apply(lambda x : x.startswith('2.5'))),
           (df['cellName'].apply(lambda x : not '-2' in x)),
           ]
df_f = filterDf(df, Filters)

# New data
# dH
df_f['dH_relax'] = df_f['initialThickness'] - df_f['previousThickness']
# df_f['dH_ratio'] = df_f['initialThickness'] / df_f['previousThickness']
df['dH_ratio'] = df['dH_relax'] / df['previousThickness']
parameter = 'dH_ratio'

# ManipTime
df_f['ManipTime'] = df_f['compAbsStartTime']
cellID_list = df_f['cellID'].unique()
manipID_list = df_f['manipID'].unique()
for mid in manipID_list:
    index_mid = df_f[df_f['manipID']==mid].index
    firstManipTime = np.min(df_f[df_f['manipID']==mid]['compAbsStartTime'].values)
    df_f.loc[index_mid, 'ManipTime'] -= firstManipTime
df_f['ManipTime'] /= 60


# Re-filter
Filters = [(df_f['dH_ratio'] <= 3),
           (df_f['dH_ratio'] >= 0.5),
           ]

df_f = filterDf(df_f, Filters)

    
# Group
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = ['manipID', 'ManipTime'], numCols = ['bestH0', 'dH_relax', 'dH_ratio'], aggFun = 'mean')



fig, ax = plt.subplots(1, 1, figsize=(17/cm_in, 10/cm_in))
PLOT_FIT = True

#### Plot 1
ax = ax
cL = gs.cL_Set21

ax.set_xlabel('Time from experiment start (min)')
ax.set_ylabel('$\delta H$ / $H_{5mT}$')
for mid, color in zip(manipID_list, cL):
    X = df_fg[df_fg['manipID']==mid]['ManipTime'].values
    Y = df_fg[df_fg['manipID']==mid]['dH_ratio'].values
    M = np.array([X,Y]).T
    M = ufun.sortMatrixByCol(M, col=0, direction = 1)
    [X, Y] = M.T
    ax.scatter(X, Y, color=color, alpha=0.8, label=mid)
    ax.plot(X, Y, ls='-', c='gray', lw=0.8, zorder=0)
    
    if PLOT_FIT:
        df_fit = df_fg[df_fg['manipID']==mid].copy()
        params, results = ufun.fitLineHuber(df_fit['ManipTime'].values, df_fit['dH_ratio'].values)
        [b, a] = params
        pval = results.pvalues[1]
        fitX = np.linspace(min(df_fit['ManipTime'].values), max(df_fit['ManipTime'].values), 10)
        fitY = a*fitX + b
        ax.plot(fitX, fitY, ls='--', c=color, lw=2,
                label = f'{a:.3f} x + {b:.1f} - p-val = {pval:.2f}')
    
ax.grid(visible=True, which='major', axis='y')

if PLOT_FIT:
    params, results = ufun.fitLineHuber(df_fg['ManipTime'].values, df_fg['dH_ratio'].values)
    [b, a] = params
    pval = results.pvalues[1]
    fitX = np.linspace(min(df_fg['ManipTime'].values), max(df_fg['ManipTime'].values), 10)
    fitY = a*fitX + b
    ax.plot(fitX, fitY, ls='--', c='k', lw=2,
            label = 'Global Fit\n' + f'{a:.3f} x + {b:.1f} - p-val = {pval:.2f}')

# Prettify
rD = {'manipID' : 'Exp #',
      'dH_ratio' : '$\delta H$ / $H_H_{5mT}$'
      }

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(visible=True, which='major', axis='y')

# Show
plt.tight_layout()
plt.show()

# Save
figname = 'relaxH_vs_maniptime' + suffix
ufun.archiveFig(fig, name = figname, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %% Plots - Resting mag field

# %%% Thickness

gs.set_mediumText_options_jv()
figSubDir = ''

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
dates = ['23-07-06']
suffix = ''
substrate = '20um fibronectin discs'

parameter = 'surroundingThickness'
condCol = 'normal field'

# Filter
Filters = [
           # (df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['date'].apply(lambda x : x in dates)),
           (df['bestH0'] <= 900),
           (df['compNum'] <= 10),
           # (df['E_f_<_400'] <= 2e4),
           # (df['valid_f_<_400'] == True), 
           ]
df_f = filterDf(df, Filters)

# New id
df_f['cellNo'] = df_f['cellName'].apply(lambda x : x.split('_')[-1])

# Order
co_order = [5, 10, 15]
box_pairs = []

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol, 'cellNo'], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
# fig, ax = D1Plot(df_f, fig = fig, ax = ax, condition=condCol, parameter=parameter,
#                  co_order = co_order, boxplot = 2, figSizeFactor = 0.8, markersizeFactor =1,
#                  stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
#                  showMean = False)
plot_parms = {'data' : df_fg, 'x' : condCol, 'y' : parameter, 'order': co_order,
             }

swarmplot_parms = {**plot_parms}
swarmplot_parms.update({'hue' : 'cellNo', 'size' : 7, 'palette' : 'Set1',
                        'edgecolor' : 'gray', 'linewidth' : 0.75})
sns.swarmplot(ax=ax, **swarmplot_parms)

boxplot_parms = {**plot_parms}
boxplot_parms.update(width = 0.5, showfliers = False,
                    medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},
                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                    # boxprops={"color": color, "linewidth": 0.5},
                    whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                    capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2})
sns.boxplot(ax=ax, **boxplot_parms)

if len(box_pairs) == 0:
    box_pairs = makeBoxPairs(co_order)
addStat_lib(ax, box_pairs, test = 'Mann-Whitney', verbose = False, **plot_parms)
           
# Prettify
ax.grid(visible=True, which='major', axis='y')
# ax.legend().set_visible(False)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()


# %%% Stiffness

gs.set_mediumText_options_jv()
figSubDir = ''

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
dates = ['23-07-06']
suffix = ''
substrate = '20um fibronectin discs'

parameter = 'E_f_<_400'
condCol = 'normal field'

# Filter
Filters = [
           # (df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['date'].apply(lambda x : x in dates)),
           # (df['bestH0'] <= 900),
           # (df['compNum'] <= 10),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           ]
df_f = filterDf(df, Filters)

# New id
df_f['cellNo'] = df_f['cellName'].apply(lambda x : x.split('_')[-1])

# Order
co_order = [5, 10, 15]
box_pairs = []

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol, 'cellNo'], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
# fig, ax = D1Plot(df_f, fig = fig, ax = ax, condition=condCol, parameter=parameter,
#                  co_order = co_order, boxplot = 2, figSizeFactor = 0.8, markersizeFactor =1,
#                  stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
#                  showMean = False)
plot_parms = {'data' : df_fg, 'x' : condCol, 'y' : parameter, 'order': co_order,
             }

swarmplot_parms = {**plot_parms}
swarmplot_parms.update({'hue' : 'cellNo', 'size' : 7, 'palette' : 'Set1',
                        'edgecolor' : 'gray', 'linewidth' : 0.75})
sns.swarmplot(ax=ax, **swarmplot_parms)

boxplot_parms = {**plot_parms}
boxplot_parms.update(width = 0.5, showfliers = False,
                    medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},
                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                    # boxprops={"color": color, "linewidth": 0.5},
                    whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                    capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2})
sns.boxplot(ax=ax, **boxplot_parms)

if len(box_pairs) == 0:
    box_pairs = makeBoxPairs(co_order)
addStat_lib(ax, box_pairs, test = 'Mann-Whitney', verbose = False, **plot_parms)
           
# Prettify
ax.grid(visible=True, which='major', axis='y')
# ax.legend().set_visible(False)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()



# %%% Non-linearity - Global

gs.set_mediumText_options_jv()
figSubDir = ''

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
dates = ['23-07-06']
suffix = ''
substrate = '20um fibronectin discs'
mode = 'wholeCurve'
# mode = '300_500'

condCol = 'normal field'

# Filter
Filters = [
           (df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['date'].apply(lambda x : x in dates)),
           # (df['bestH0'] <= 900),
           # (df['compNum'] <= 10),
           # (df['E_f_<_400'] <= 2e4),
           # (df['valid_f_<_400'] == True), 
           ]
df_f = filterDf(df, Filters)

# New id
df_f['cellNo'] = df_f['cellName'].apply(lambda x : x.split('_')[-1])

# Order
co_order = [5, 10, 15]
box_pairs = []

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol, 'cellNo'], numCols = [parameter], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))
    
plotPopKS_V2(df_f, fig = fig, ax = axes[0], condition = condCol, co_order = co_order,
                 fitType = 'stressGaussian', fitWidth=100, 
                  mode = mode, Sinf = 0, Ssup = 800)
plotPopKS_V2(df_f, fig = fig, ax = axes[1], condition = condCol, co_order = co_order, 
                 fitType = 'stressGaussian', fitWidth=100,
                  mode = mode, Sinf = 0, Ssup = 800)
           
# Prettify
# ax.grid(visible=True, which='major', axis='y')
# ax.legend().set_visible(False)
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# ax = axes[0]
# ax.set_xlim(0, 1200)
# ax.set_ylim(0, 20)
# ax = axes[1]
# ax.set_xlim(0, 600)
# ax.set_ylim(0, 8)

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
# ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
#                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%% Solidity Ratio

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO']
substrate = '20um fibronectin discs'
dates = ['23-07-06']

# df, condCol = makeCompositeCol(df, cols=['date', 'drug'])
# figname = 'bestH0' + drugSuffix
condCol = 'normal field'

# New parameter
df['dH_relax'] = df['initialThickness'] - df['previousThickness']
df['dH_ratio'] = df['initialThickness'] / df['previousThickness']
# df['dH_ratio'] = df['dH_relax'] / df['previousThickness']
parameter = 'dH_ratio'

# Filter
Filters = [(df['date'].apply(lambda x : x in dates)),
           (df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           # (df['normal field'] == 5),
           # (df['ramp field'].apply(lambda x : x.startswith('2.5'))),
           (df['bestH0'] <= 900),
           (df['dH_ratio'] <= 3),
           (df['dH_ratio'] >= 0.5),
           ]


df_f = filterDf(df, Filters)

# New id
df_f['cellNo'] = df_f['cellName'].apply(lambda x : x.split('_')[-1])

# Order
co_order = [5, 10, 15]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol, 'cellNo'], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(6, 4))
# fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
#                  co_order = co_order, boxplot = 2, figSizeFactor = 2, markersizeFactor = 1.25,
#                  stats=False, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
#                  showMean = False)

plot_parms = {'data' : df_fg, 'x' : condCol, 'y' : parameter, 'order': co_order,
             }

swarmplot_parms = {**plot_parms}
swarmplot_parms.update({'hue' : 'cellNo', 'size' : 7, 'palette' : 'Set1',
                        'edgecolor' : 'gray', 'linewidth' : 0.75})
sns.swarmplot(ax=ax, **swarmplot_parms)

boxplot_parms = {**plot_parms}
boxplot_parms.update(width = 0.5, showfliers = False,
                    medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},
                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                    # boxprops={"color": color, "linewidth": 0.5},
                    whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
                    capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2})
sns.boxplot(ax=ax, **boxplot_parms)

if len(box_pairs) == 0:
    box_pairs = makeBoxPairs(co_order)
addStat_lib(ax, box_pairs, test = 'Mann-Whitney', verbose = False, **plot_parms)
           
# Prettify
ax.grid(visible=True, which='major', axis='y')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()




# %% Plots - Pattern sizes


# %% Plots - Non Lin

# %%%% Figure NC5 - Bonus - Clues

gs.set_manuscript_options_jv(palette = 'Set2')

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['drug'].apply(lambda x : x in drugs)),
            (df['date'].apply(lambda x : x not in excluded_dates)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['E_f_<_400'] <= 3e4),
           (df['valid_f_<_400'] == True), 
           (df['E_f_in_400_800'] <= 3e4),
           (df['valid_f_in_400_800'] == True), 
           ]

df_f = filterDf(df, Filters)
CountByCond, CountByCell = makeCountDf(df_f, condCol)

df_fgw400 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_fgw400['E_f_<_400_wAvg'] /= 1000

df_fgw800 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_in_400_800', weightCol = 'ciwE_f_in_400_800', weight_method = 'ciw^2')
df_fgw800['E_f_in_400_800_wAvg'] /= 1000

df_fgwM = pd.merge(df_fgw400, df_fgw800, on='cellID', how='inner', suffixes=('', '_2'))

Filters = [(df_fgwM['count_wAvg'] > 1), 
           (df_fgwM['count_wAvg_2'] > 1), 
           ]

# df_fgwM = filterDf(df_fgwM, Filters)
# CountByCond, CountByCell = makeCountDf(df_fgwM, condCol)

#### Plot df
V_cellID = df_fgwM['cellID'].values
V_400 = df_fgwM['E_f_<_400_wAvg'].values
V_800 = df_fgwM['E_f_in_400_800_wAvg'].values
Nc = len(V_cellID)
kind400 = np.array(['.<400']*Nc)
kind800 = np.array(['400<.<800']*Nc)
dict400 = {'cellID':V_cellID, 'E':V_400, 'range':kind400}
dict800 = {'cellID':V_cellID, 'E':V_800, 'range':kind800}
df_plot = pd.concat([pd.DataFrame(dict400), pd.DataFrame(dict800)]).reset_index(drop=True)


co_values = df_plot['range'].unique()
Nco = len(co_values)
co_order = np.sort(co_values)
box_pairs = makeBoxPairs(co_order)
    

fig, axes = plt.subplots(1, 1, figsize=(17/gs.cm_in, 10/gs.cm_in))

#### Plot 1 - Stiffnesses
ax = axes
plot_parms = {'data':df_plot, 'x':'range', 'y':'E'}
sns.swarmplot(ax=ax, **plot_parms)
addStat_lib(ax, box_pairs, test = 'Mann-Whitney', verbose = False, **plot_parms)

boxplot_parms = {**plot_parms, 'showfliers': False}
boxplot_parms.update(medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
                     boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                     # boxprops={"color": color, "linewidth": 0.5},
                     whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
                     capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2})
sns.boxplot(ax=ax, **boxplot_parms)

fig.tight_layout()
plt.show()

figSubDir = 'NonLinearity'
name = 'Clue_E400-800'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.cellCount.values[0] = Nc
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')



# %%%% Figure NC5 - Bonus - Strain ?

# Define
df = MecaData_Phy
substrate = '20um fibronectin discs'
condCol = 'drug' # 'date'
mode = 'wholeCurve'
figname = 'nonLin_' + mode + 'dmso'
dates = ['23-03-09']


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
            (df['date'].apply(lambda x : x in dates)),
           # (df['drug'] == 'dmso'),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = []

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,6))
    
plotPopKS_V2(df_f, fig = fig, ax = axes[0], condition = 'drug', co_order = co_order,
                 fitType = 'strainGaussian', fitWidth=0.0125, 
                  mode = 'wholeCurve', Sinf = 0, Ssup = np.Inf)
plotPopKS_V2(df_f, fig = fig, ax = axes[1], condition = 'drug', co_order = co_order, 
                 fitType = 'strainGaussian', fitWidth=0.025,
                  mode = 'wholeCurve', Sinf = 0, Ssup = np.Inf)
           
# Prettify
# rD = {'dmso & 0.0' : 'DMSO',
#       'Y27 & 50.0' : 'Y27\n(50µM)',
#       'LIMKi & 10.0' : 'LIMKi\n(10µM)',
#       'LIMKi & 20.0' : 'LIMKi\n(20µM)',
#       'blebbistatin & 50.0':'Blebbi\n(50µM)',
#       'blebbistatin & 100.0':'Blebbi\n(100µM)',
#       'none & 0.0':'No Drug',
#       'E_f_<_400' : 'Elastic modulus (Pa)\nfor F < 400pN',
#       'E_f_<_400_wAvg' : 'Elastic modulus (Pa)\nfor F < 400pN',
#       }


# for ax in axes:
#     ax.grid(visible=True, which='major', axis='both')
#     renameAxes(ax, rD, format_xticks = False)
#     renameAxes(ax, renameDict, format_xticks = False)
#     renameLegend(ax, rD)

ax = axes[0]
ax.set_xlim(0, 0.6)
ax.set_ylim(0, 24)
ax = axes[1]
ax.set_xlim(0, 0.6)
ax.set_ylim(0, 24)

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
# ufun.archiveFig(fig, name = figname, ext = '.png', dpi = 100,
#                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')




# %%%% Figure NC5 - Bonus - Bead sizes ?




# %% Plots - Non Lin VW

# %%%%% 1.

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO'] # 
substrate = '20um fibronectin discs'
parameter = 'bestH0'
# parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug'])
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           # (df['ramp field'].apply(lambda x : x.startswith('2'))),
           # (df['E_f_<_400'] <= 2e4),
           # (df['valid_f_<_400'] == True), 
           (df['drug'] == 'dmso'),
           ]
df_f = filterDf(df, Filters)

#### Default threshold
th_NLI = np.log10(10)
df_fNL = computeNLMetrics(df_f, th_NLI = th_NLI)

fig, ax = plt.subplots(figsize = (10/gs.cm_in,10/gs.cm_in), tight_layout = True)
# condCol, condCat = 'drug', drugs

df_fNL, condCol = makeCompositeCol(df_fNL, cols=['drug'])

condCol = 'drug'
condCat = ['dmso']
bp = []

plotChars = {'color' : 'black', 'fontsize' : 8}

fig, ax, pvals = plotNLI(fig, ax, df_fNL, condCat, condCol, bp, labels = [], colorScheme = 'white', **plotChars)
ax.set_xticklabels(['DMSO'])
ax.set_xlabel('')
ax.set_ylabel('Compressions behavior (%)')
ax.get_legend().set_title(f'Thesh = $\pm${th_NLI:.1f}')

plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_fNL, condCol)
# Save
figSubDir = 'NonLinearity'
name = 'nonLin_VW_allDmso_th03'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')





#### Other threshold
th_NLI = 8
df_fNL = computeNLMetrics(df_f, th_NLI = th_NLI)

fig, ax = plt.subplots(figsize = (10/gs.cm_in,10/gs.cm_in), tight_layout = True)
# condCol, condCat = 'drug', drugs

df_fNL, condCol = makeCompositeCol(df_fNL, cols=['drug'])

condCol = 'drug'
condCat = ['dmso']
bp = []

plotChars = {'color' : 'black', 'fontsize' : 8}

fig, ax, pvals = plotNLI(fig, ax, df_fNL, condCat, condCol, bp, labels = [], colorScheme = 'white', **plotChars)
ax.set_xticklabels(['DMSO'])
ax.set_xlabel('')
ax.set_ylabel('Compressions behavior (%)')
ax.get_legend().set_title(f'Thesh = $\pm${th_NLI:.1f}')

plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_fNL, condCol)
# Save
figSubDir = 'NonLinearity'
name = 'nonLin_VW_allDmso_th6'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%%%% 2.

gs.set_manuscript_options_jv(palette = 'Set2')

# Define
df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO'] # 
substrate = '20um fibronectin discs'
parameter = 'bestH0'
# parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug'])
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           # (df['ramp field'].apply(lambda x : x.startswith('2'))),
           # (df['E_f_<_400'] <= 2e4),
           # (df['valid_f_<_400'] == True), 
           (df['drug'] == 'dmso'),
           ]
df_f = filterDf(df, Filters)
df_f = computeNLMetrics(df_f, th_NLI = np.log10(10)) # np.log10(2)
CountByCond, CountByCell = makeCountDf(df_f, 'NLI_Plot')



#### Plot 1 - lin

# Order
co_order = []

# Filter again
Filters = [(df_f['NLI_Plot'] == 'linear'), 
           ]
df_f1 = filterDf(df_f, Filters)

# Plot
fig1, ax = plt.subplots(1,1, figsize=(17/gs.cm_in,12/gs.cm_in))
ax.set_yscale('log')

intervals = ['100_300', '150_400', '200_500', '300_1000', '500_1200']
cL = plt.cm.plasma(np.linspace(0.1, 0.9, len(intervals)))

for i, interval in enumerate(intervals):
    colorDict = {'dmso':cL[i]}
    labelDict = {'dmso':f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"}
    plotPopKS_V2(df_f1, fig = fig, ax = ax, condition = 'drug', co_order = co_order, 
                 colorDict = colorDict, labelDict = labelDict,
                 fitType = 'stressGaussian', fitWidth=75, mode = interval, Sinf = 0, Ssup = np.Inf)

# Prettify
rD = {'dmso' : 'DMSO',
      }

ax.grid(visible=True, which='both', axis='y')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)

ax.set_title('Linear NLI')
ax.set_xlim(0, 1300)
ax.set_ylim(0.8, 30)
ax.legend(loc='upper left')

# Count
CountByCond1, CountByCell1 = makeCountDf(df_f1, condCol)






#### Plot 2 - non-lin
# Order
co_order = []

# Filter again
Filters = [(df_f['NLI_Plot'] == 'non-linear'), 
           ]
df_f2 = filterDf(df_f, Filters)

# Plot
fig2, ax = plt.subplots(1,1, figsize=(17/gs.cm_in,12/gs.cm_in))
ax.set_yscale('log')

intervals = ['100_300', '150_400', '200_500', '300_1000', '500_1200']
cL = plt.cm.plasma(np.linspace(0.1, 0.9, len(intervals)))

for i, interval in enumerate(intervals):
    colorDict = {'dmso':cL[i]}
    labelDict = {'dmso':f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"}
    plotPopKS_V2(df_f2, fig = fig, ax = ax, condition = 'drug', co_order = co_order, 
                 colorDict = colorDict, labelDict = labelDict,
                 fitType = 'stressGaussian', fitWidth=75, mode = interval, Sinf = 0, Ssup = np.Inf)

# Prettify
rD = {'dmso' : 'DMSO',
      }

ax.grid(visible=True, which='both', axis='y')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)

ax.set_title('Non-linear NLI')
ax.set_xlim(0, 1300)
ax.set_ylim(0.8, 30)
ax.legend(loc='upper left')

# Count
CountByCond2, CountByCell2 = makeCountDf(df_f2, condCol)






#### Plot 3 - intermediate
# Order
co_order = []

# Filter again
Filters = [(df_f['NLI_Plot'] == 'intermediate'), 
           ]
df_f3 = filterDf(df_f, Filters)

# Plot
fig3, ax = plt.subplots(1,1, figsize=(17/gs.cm_in,12/gs.cm_in))
ax.set_yscale('log')

intervals = ['100_300', '150_400', '200_500', '300_1000', '500_1200']
cL = plt.cm.plasma(np.linspace(0.1, 0.9, len(intervals)))

for i, interval in enumerate(intervals):
    colorDict = {'dmso':cL[i]}
    labelDict = {'dmso':f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"}
    plotPopKS_V2(df_f3, fig = fig, ax = ax, condition = 'drug', co_order = co_order, 
                 colorDict = colorDict, labelDict = labelDict,
                 fitType = 'stressGaussian', fitWidth=75, mode = interval, Sinf = 0, Ssup = np.Inf)

# Prettify
rD = {'dmso' : 'DMSO',
      }

ax.grid(visible=True, which='both', axis='y')
renameAxes(ax, rD, format_xticks = False)
renameAxes(ax, renameDict, format_xticks = False)
renameLegend(ax, rD)

ax.set_title('Intermediate NLI')
ax.set_xlim(0, 1300)
ax.set_ylim(0.8, 30)
ax.legend(loc='upper left')

# Count
CountByCond3, CountByCell3 = makeCountDf(df_f3, condCol)





#### Show
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
plt.show()

#### Save
figSubDir = 'NonLinearity'

# fig = fig1
# name = 'nonLin_dmso_w75_NLIsort_LIN'
# ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
#                 figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
# CountByCond1.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# fig = fig2
# name = 'nonLin_dmso_w75_V2_NLIsort_NONLIN'
# ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
#                 figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
# CountByCond1.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# fig = fig3
# name = 'nonLin_dmso_w75_V2_NLIsort_INTER'
# ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
#                 figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
# CountByCond1.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%%%% 3.

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO'] # 
substrate = '20um fibronectin discs'
parameter = 'bestH0'
# parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug'])
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['ramp field'].apply(lambda x : x.startswith('2'))),
           # (df['E_f_<_400'] <= 2e4),
           # (df['valid_f_<_400'] == True), 
           (df['drug'] == 'dmso'),
           ]

df_f = filterDf(df, Filters)
df_f = computeNLMetrics(df_f)
df_f['H0_BinStr'] = '<' + (df_f['H0_Bin']*100).astype(str)

condCol = 'H0_BinStr'
CountByCond, CountByCell = makeCountDf(df_f, condCol)
validCond = CountByCond[CountByCond['compCount'] > 30].index

fig, ax = plt.subplots(figsize = (17/gs.cm_in,10/gs.cm_in), tight_layout = True)
# condCol, condCat = 'drug', drugs
# df_f, condCol = makeCompositeCol(df_f, cols=['drug'])


condCat = validCond
bp = []

plotChars = {'color' : 'black', 'fontsize' : 8}

fig, ax, pvals = plotNLI(fig, ax, df_f, condCat, condCol, bp, labels = [], colorScheme = 'white', **plotChars)
ax.set_xlabel('Bins of $H_0$ (nm)')
ax.set_ylabel('Compressions behavior (%)')

fig.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
figSubDir = 'NonLinearity'
name = 'nonLin_VW_allDmso_byH0'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')



# %%%%% X. Check NLI distribution

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['none', 'DMSO'] # 
substrate = '20um fibronectin discs'
parameter = 'bestH0'
# parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug'])
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['ramp field'].apply(lambda x : x.startswith('2'))),
           # (df['E_f_<_400'] <= 2e4),
           # (df['valid_f_<_400'] == True), 
           (df['drug'] == 'dmso'),
           ]

df_f = filterDf(df, Filters)
df_f = computeNLMetrics(df_f, th_NLI = np.log10(2))

# Filter again
Filters = [(df_f['NLI'] <  25), 
           (df_f['NLI'] > -25), 
           ]
df_f = filterDf(df_f, Filters)

fig, ax = plt.subplots(1,1, figsize = (17/gs.cm_in,10/gs.cm_in), tight_layout = True)
# condCol, condCat = 'drug', drugs
# df_f, condCol = makeCompositeCol(df_f, cols=['drug'])

# ax.set_yscale('log')
# sns.swarmplot(data=df_f, ax=ax, x='drug', y='NLI')
N, bins, patches = ax.hist(x=df_f['NLI'].values, bins = 50, color = color)
ax.axvline(np.log10(2))
ax.axvline(-np.log10(2))
ax.set_xlabel('Cells')
ax.set_ylabel('NLI')

fig.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'NonLinearity'
# name = 'nonLin_VW_allDmso_byH0'
# ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
#                 figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
# CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%%%% 4. E_eff per cell

gs.set_manuscript_options_jv(palette = 'Set2')

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['ramp field'].apply(lambda x : x.startswith('2'))),
           # (df['E_f_<_400'] <= 2e4),
           # (df['valid_f_<_400'] == True), 
           (df['drug'] == 'dmso'),
           ]

df_f = filterDf(df, Filters)
df_f = computeNLMetrics(df_f)

# Group By for H0
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0', 'E_eff', 'NLI_Ind'], aggFun = 'mean')

# Group By for E<400
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
# df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
#                                       valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
# df_fgw2['E_f_<_400_wAvg'] /= 1000

df_fg['E_eff'] /= 1000

#### Init fig
gs.set_manuscript_options_jv()
fig, axes = plt.subplots(2, 1, figsize=(17/cm_in, 17/cm_in))
color = styleDict['dmso']['color']

#### 01 - Best H0
ax = axes[0]

# Order
# co_order = ['none', 'dmso']

# Plot
# fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter='bestH0',
#                  co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 0.6,
#                  stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
#                  showMean = False)
# ax.set_title('Thickness')
ax.set_xlabel('$E_{eff}$ (kPa)')
ax.set_ylabel('Count (cells)')
N, bins, patches = ax.hist(x=df_fg['E_eff'].values, bins = 15, color = color) # , bins = 20
ax.set_xlim([0, ax.get_xlim()[1]])
median_E_eff = np.median(df_fg['E_eff'].values)
ax.axvline(median_E_eff, c='darkred', label = 'Median $E_{eff}$' + f' = {median_E_eff:.2f} kPa')
ax.legend()


#### 02 - E_400
ax = axes[1]

# Order
# co_order = ['none', 'dmso']

# Plot
# fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter='E_f_<_400_wAvg',
#                  co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 0.6,
#                  stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
#                  showMean = False)

ax.set_xscale('log')
ax.set_yscale('log')
sns.scatterplot(data=df_fg, x='bestH0', y='E_eff', hue='NLI_Ind')
ax.set_xlabel('Fitted $H_0$ (nm)')
ax.set_ylabel('$E_{eff}$ (kPa)')


# Prettify
rD = {'none' : 'No drug',
      'dmso' : 'DMSO', 
      'bestH0' : 'Fitted $H_0$ (nm)',
      'E_f_<_400_wAvg' : '$E_{400}$ (kPa)'
      }

for ax in axes:
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, renameDict)
    # ax.grid(visible=True, which='major', axis='y')
    
    
    
fig.tight_layout()
plt.show()
    

# %%%%% 4. E_eff per compression

gs.set_manuscript_options_jv(palette = 'Set2')

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['ramp field'].apply(lambda x : x.startswith('2'))),
           # (df['E_f_<_400'] <= 2e4),
           # (df['valid_f_<_400'] == True), 
           (df['drug'] == 'dmso'),
           ]

df_f = filterDf(df, Filters)
df_f = computeNLMetrics(df_f)

# Filter again
Filters = [(df_f['E_eff'] < 4e4), 
           ]
df_f = filterDf(df_f, Filters)

# Group By for H0
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0', 'E_eff', 'NLI_Ind'], aggFun = 'mean')

# Group By for E<400
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
# df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
#                                       valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
# df_fgw2['E_f_<_400_wAvg'] /= 1000

df_f['E_eff'] /= 1000

#### Init fig
gs.set_manuscript_options_jv()
fig, axes = plt.subplots(2, 1, figsize=(17/cm_in, 17/cm_in))
color = styleDict['dmso']['color']

#### 01 - Best H0
ax = axes[0]

# Order
# co_order = ['none', 'dmso']

# Plot
# fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter='bestH0',
#                  co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 0.6,
#                  stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
#                  showMean = False)
# ax.set_title('Thickness')
ax.set_xlabel('$E_{eff}$ (kPa)')
ax.set_ylabel('Count (cells)')
N, bins, patches = ax.hist(x=df_f['E_eff'].values, bins = 40, color = color) # , bins = 20
ax.set_xlim([0, ax.get_xlim()[1]])
median_E_eff = np.median(df_f['E_eff'].values)
ax.axvline(median_E_eff, c='darkred', label = 'Median $E_{eff}$' + f' = {median_E_eff:.2f} kPa')
ax.legend()


#### 02 - E_400
ax = axes[1]

# Order
# co_order = ['none', 'dmso']

# Plot
# fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter='E_f_<_400_wAvg',
#                  co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 0.6,
#                  stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
#                  showMean = False)

ax.set_xscale('log')
ax.set_yscale('log')
sns.scatterplot(data=df_f, x='bestH0', y='E_eff', hue='NLI_Ind')
ax.set_xlabel('Fitted $H_0$ (nm)')
ax.set_ylabel('$E_{eff}$ (kPa)')


# Prettify
rD = {'none' : 'No drug',
      'dmso' : 'DMSO', 
      'bestH0' : 'Fitted $H_0$ (nm)',
      'E_f_<_400_wAvg' : '$E_{400}$ (kPa)'
      }

for ax in axes:
    renameAxes(ax, rD, format_xticks = False)
    renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, renameDict)
    # ax.grid(visible=True, which='major', axis='y')
    
    
fig.tight_layout()
plt.show()


# %%%%% 4. E_eff vs E400

gs.set_manuscript_options_jv(palette = 'Set2')

#### Dataset

df = MecaData_Phy
cell_subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
drugs = ['dmso'] #['none', 'dmso']
substrate = '20um fibronectin discs'
parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug'])
excluded_dates = ['23-03-08', '23-02-23', '23-11-26']
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['cell subtype'].apply(lambda x : x in cell_subtypes)),
           (df['bestH0'] < 1000),
           (df['normal field'] == 5),
           (df['ramp field'].apply(lambda x : x.startswith('2'))),
            (df['E_f_<_400'] <= 2e4),
            (df['valid_f_<_400'] == True), 
           (df['drug'] == 'dmso'),
           ]

df_f = filterDf(df, Filters)
df_f = computeNLMetrics(df_f)

# Filter again
Filters = [(df_f['E_eff'] < 4e4), 
           ]
df_f = filterDf(df_f, Filters)

# Group By for H0
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0', 'E_eff', 'NLI_Ind'], aggFun = 'mean')

# Group By for E<400
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
# df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
#                                       valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
# df_fgw2['E_f_<_400_wAvg'] /= 1000

df_f['E_eff'] /= 1000
df_f['E_f_<_400'] /= 1000

#### Init fig
gs.set_manuscript_options_jv()
fig, ax = plt.subplots(1, 1, figsize=(17/cm_in, 17/cm_in))


# Order
# co_order = ['none', 'dmso']

# Plot
# fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter='E_f_<_400_wAvg',
#                  co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 0.6,
#                  stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
#                  showMean = False)

ax.set_xscale('log')
ax.set_yscale('log')
sns.scatterplot(data=df_f, ax=ax, x='E_f_<_400', y='E_eff', hue='NLI_Ind', palette='viridis')
# ax.axline((0,0), slope=1, ls='--', color='r')
ax.set_xlabel('$E_{400}$ (kPa)')
ax.set_ylabel('$E_{eff}$ (kPa)')
ax.grid()
    
fig.tight_layout()
plt.show()


