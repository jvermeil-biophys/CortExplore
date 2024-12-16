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

figDir = "D:/MagneticPincherData/Figures/CellTypesDataset_V2"

cm_in = 2.52

# %% > Data import & export

# MecaData_DrugV3 = taka2.getMergedTable('MecaData_Drugs_V3')


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
               
               '3T3 & Atcc-2023':{'color': gs.cL_Set21[0],'marker':'o'},
               '3T3 & aSFL-A11':{'color': gs.cL_Set21[1],'marker':'o'},
               'DC & mouse-primary':{'color': gs.cL_Set21[2],'marker':'o'},
               'Dicty & DictyBase-WT':{'color': gs.cL_Set21[3],'marker':'o'},
               'HoxB8-Macro & ctrl':{'color': gs.cL_Set21[4],'marker':'o'},
               'MDCK & WT':{'color': gs.cL_Set21[5],'marker':'o'},
               
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

def renameAxes(axes, rD, format_xticks = True, rotation = 0):
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
        # set title
        axtitle = axes[i].get_title()
        newaxtitle = rD.get(axtitle, axtitle)
        axes[i].set_title(newaxtitle)
        
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
                
def addStat_lib(ax, box_pairs, test = 'Mann-Whitney', verbose = False, statpos = 'in', 
                **plotting_parameters):
    #### STATS
    listTests = ['t-test_ind', 't-test_welch', 't-test_paired', 
                 'Mann-Whitney', 'Mann-Whitney-gt', 'Mann-Whitney-ls', 
                 'Levene', 'Wilcoxon', 'Kruskal', 'Brunner-Munzel']
    if test in listTests:
        annotator = Annotator(ax, box_pairs, **plotting_parameters)
        annotator.configure(test=test, verbose=verbose, fontsize = 9, loc = statpos+'side',
                            line_height = 0.01).apply_and_annotate() # , loc = 'outside' , hide_non_significant = True
    else:
        print(gs.BRIGHTORANGE + 'Dear Madam, dear Sir, i am the eternal god and i command that you define this stat test cause it is not in the list !' + gs.NORMAL)
    return(ax)


# %% > Plot Functions

def D1Plot(data, fig = None, ax = None, condition='', parameter='',
           co_order=[], boxplot=1, figSizeFactor = 1, markersizeFactor = 1,
           stats=True, statMethod='Mann-Whitney', box_pairs=[], statVerbose = False, statpos = 'in',
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
        addStat_lib(ax, box_pairs, test = statMethod, verbose = statVerbose, statpos = statpos,
                    **swarmplot_parameters)

    
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




def D1Plot_violin(data, fig = None, ax = None, condition='', parameter='',
           co_order=[], figSizeFactor = 1, 
           stats=True, statMethod='Mann-Whitney', box_pairs=[], statVerbose = False):
    
    #### Init
    co_values = data[condition].unique()
    Nco = len(co_values)
    if len(co_order) == 0:
        co_order = np.sort(co_values)
        
    if ax == None:
        figHeight = 5
        figWidth = 5*Nco*figSizeFactor
        fig, ax = plt.subplots(1,1, figsize=(figWidth, figHeight))    
        
        
    palette = getSnsPalette(co_order, styleDict)
    
    #### Swarmplot
    violinplot_parameters = {'data':    data,
                            'x':       condition,
                            'y':       parameter,
                            'order':   co_order,
                            'palette': palette,
                            }
    
    sns.violinplot(ax=ax, **violinplot_parameters, zorder = 8)

    #### Stats    
    if stats:
        if len(box_pairs) == 0:
            box_pairs = makeBoxPairs(co_order)
        addStat_lib(ax, box_pairs, test = statMethod, verbose = statVerbose, **violinplot_parameters)

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
                            color = c, zorder = 4)

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
                    eqnText += " ; p-val = {:.3f}".format(pval)
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
                 condition = '', co_order = [], colorDict = {}, labelDict = {},
                 fitType = 'stressRegion', fitWidth=75, markersizefactor = 1,
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
                    color = color, lw = 2, marker = 'o', markersize = 6*markersizefactor, mec = 'k', mew = 1*markersizefactor,
                    ecolor = color, elinewidth = 1.5, capsize = 6, capthick = 1.5, 
                    label = labelDict[co] + ' | ' + str(total_N) + ' cells' + ' | ' + str(total_n) + ' comp')
        
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
        fig, axes = plt.subplots(1, Nco, figsize = (17/gs.cm_in, 4*Nco/gs.cm_in), sharex = True)
    else:
        fig, axes = plt.subplots(1, 1, figsize = (17/gs.cm_in, 10/gs.cm_in))
    
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
        
    fig.legend()
    
    out = fig, axes    
    print(out)
    print(axes)
    return(out)


def StressRange_2D_V2(data, fig=None, ax=None, colorList = gs.colorList30, 
                      condition='', defaultMode = False):
        
    co_values = list(data[condition].unique())
    Nco = len(co_values)
    # cl = matplotlib.colormaps['Set1'].colors[1:] + matplotlib.colormaps['Set2'].colors[:-1]
    cl = colorList
    
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
            s = 15
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


def StressRange_2D_V3(data, fig=None, ax=None):
    
    Npp = 50
    Ntot = len(data['bestH0'].values)
    Nb = Ntot//Npp
    step = 100/Nb
    Lp = step * np.arange(Nb)
    bins = np.percentile(data['bestH0'].values, Lp)
    data['H0_Bin'] = np.digitize(data['bestH0'], bins, right = True)
    
    agg_dict = {'compNum':'count',
                'bestH0':['mean', 'std'],
                'minStress':['mean', 'std'],
                'maxStress':['mean', 'std'],
                }
    all_cols = ['H0_Bin'] + list(agg_dict.keys())
    group = data[all_cols].groupby('H0_Bin')
    data_g = group.agg(agg_dict)
    return(data_g)
    
    if ax == None:
        figHeight = 10/gs.cm_in
        figWidth = 17/gs.cm_in
        fig, ax = plt.subplots(1,1, figsize=(figWidth, figHeight))
    
    for i in range(Nco):      
        co = co_values[i]
        data_co = data[data[condition] == co]
        color = gs.colorList40[19]
        alpha = 0.4
        zo = 4
        s = 4
        ec = 'None'
        labels = ['Minimum stress', 'Maximum stress', 'Compression']
        
            
            
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
    plt.legend(handles = patches, bbox_to_anchor=(1.01, 0.5), fontsize = 8, labelcolor='linecolor')
    plt.show()

    return(fig, ax, pvals)


def computeNLMetrics(GlobalTable, th_NLI = np.log10(2), ref_strain = 0.2):

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
    E = Y + (K*(1 - ref_strain)**-4)

    data_main['E_eff'] = E
    data_main['NLI'] = np.log10(((1 - ref_strain)**-4 * K)/Y)
    
    ciwK, ciwY = data_main['ciwK_vwc_Full'], data_main['ciwY_vwc_Full']
    ciwE = ciwY + (ciwK*(1 - ref_strain)**-4)
    data_main['ciwE_eff'] = ciwE
    
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
        # for j in index:
        data_main.loc[index, 'NLI_Plot'] = i
        data_main.loc[index, 'NLI_Ind'] = ID
    
    return(data_main)

def computeNLMetrics_V2(GlobalTable, th_NLI = np.log10(2), ref_strain = 0.2):
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
    E = Y + K*(1 - ref_strain)**-4

    data_main['E_eff'] = E
    data_main['NLI'] = np.log10((1 - ref_strain)**-4 * K/Y)
    
    ciwK, ciwY = data_main['ciwK_vwc_Full'], data_main['ciwY_vwc_Full']
    ciwE = ciwY + (ciwK*(1 - ref_strain)**-4)
    data_main['ciwE_eff'] = ciwE
    

    data_main['Y_err_div10'], data_main['K_err_div10'] = data_main['ciwY_vwc_Full']/10, data_main['ciwK_vwc_Full']/10
    data_main['Y_NLImod'] = data_main[["Y_vwc_Full", "Y_err_div10"]].max(axis=1)
    data_main['K_NLImod'] = data_main[["K_vwc_Full", "K_err_div10"]].max(axis=1)
    Y_nli, K_nli = data_main['Y_NLImod'].values, data_main['K_NLImod'].values

    data_main['NLI_mod'] = np.log10((1 - ref_strain)**-4 * K_nli/Y_nli)
    
    NLItypes = ['linear', 'intermediate', 'non-linear']
    for i in NLItypes:
        if i == 'linear':
            index = data_main[data_main['NLI_mod'] < -th_NLI].index
            ID = 1
        elif i =='non-linear':
            index =  data_main[data_main['NLI_mod'] > th_NLI].index
            ID = 0
        elif i =='intermediate':
            index = data_main[(data_main['NLI_mod'] > -th_NLI) & (data_main['NLI_mod'] < th_NLI)].index
            ID = 0.5
        for j in index:
            data_main['NLI_Plot'][j] = i
            data_main['NLI_Ind'][j] = ID


    return(data_main)



# %% > Data import & export
# %%% MecaData_Cells
# MecaData_Cells = taka3.getMergedTable('MecaData_CellTypes')
MecaData_Cells = taka3.getMergedTable('MecaData_CellTypes_wMDCK')

MecaData_Phy = taka3.getMergedTable('MecaData_Physics_V2')

# %%%


MecaData_Cells2 = pd.concat([MecaData_Cells, MecaData_Phy])
MecaData_Cells2['Indent_ID'] = MecaData_Cells2['cellID'] + '_' + MecaData_Cells2['compNum'].astype('str')
MecaData_Cells2 = MecaData_Cells2.drop_duplicates(subset='Indent_ID')

path = "D:/MagneticPincherData/Data_Analysis/MecaData_CellTypes_V2.csv"
MecaData_Cells2.to_csv(path, index=False)


# %%%



# %%% Check content

print('Dates')
print([x for x in MecaData_Cells['date'].unique()])
print('')

print('Cell types')
print([x for x in MecaData_Cells['cell type'].unique()])
print('')

print('Cell subtypes')
print([x for x in MecaData_Cells['cell subtype'].unique()])
print('')

print('Drugs')
print([x for x in MecaData_Cells['drug'].unique()])
print('')

print('Substrates')
print([x for x in MecaData_Cells['substrate'].unique()])
print('')

print('Resting Fields')
print([x for x in MecaData_Cells['normal field'].unique()])
print('')

# %% Plots -- HoxB8

MecaData_HoxB8 = taka3.getMergedTable('MecaData_HoxB8')
figDir = 'D:/MagneticPincherData/Figures/CellTypesDataset'
figSubDir = 'HoxB8_project'

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

styleDict =  {'DictyDB_M270':{'color':'lightskyblue','marker':'o'}, 
               'DictyDB_M450':{'color': 'maroon','marker':'o'},
               'M270':{'color':'lightskyblue','marker':'o'}, 
               'M450':{'color': 'maroon','marker':'o'},
               
               # HoxB8
               'ctrl':{'color': gs.colorList40[12],'marker':'^'},
               'tko':{'color': gs.colorList40[32],'marker':'^'},
               'bare glass & ctrl':{'color': gs.colorList40[10],'marker':'^'},
               'bare glass & tko':{'color': gs.colorList40[30],'marker':'^'},
               '20um fibronectin discs & ctrl':{'color': gs.colorList40[12],'marker':'o'},
               '20um fibronectin discs & tko':{'color': gs.colorList40[32],'marker':'o'},
               
               # Comparison
               '3T3':{'color': gs.colorList40[31],'marker':'o'},
               'HoxB8-Macro':{'color': gs.colorList40[32],'marker':'o'},
               }

# %%% bestH0 & K-200-400

gs.set_manuscript_options_jv()

# Define 1.
df = MecaData_HoxB8
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs']
parameter = 'bestH0'

# figname = 'bestH0' + drugSuffix

dates = ['22-05-03', '22-05-04', '22-05-05']

Filters = [(df['validatedThickness'] == True),
           (df['surroundingThickness'] <= 900),
           (df['ctFieldThickness'] <= 1000),
           (df['UI_Valid'] == True),
           (df['cell type'] == 'HoxB8-Macro'), 
           (df['substrate'] == 'bare glass'),
           (df['cell subtype'].apply(lambda x : x in ['ctrl','tko'])), 
           # (data['bead type'] == 'M450'),
           (df['date'].apply(lambda x : x in dates))]


df_f = filterDf(df, Filters)

df_f, condCol = makeCompositeCol(df_f, cols=['cell subtype'])
print(df_f[condCol].unique())

# Order
co_order = ['ctrl', 'tko']
box_pairs=[]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
CountByCond.to_csv(os.path.join(figDir, figSubDir, 'HoxB8_H'+'_count.txt'), sep='\t')

# Define 2.
fitType = 'stressRegion'
fitId = '300_100'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)
df2 = taka3.getFitsInTable(df, fitType=fitType, filter_fitID=fitId)

Filters = [(df2['validatedThickness'] == True),
           (df2['surroundingThickness'] <= 900),
           (df2['ctFieldThickness'] <= 1000),
           (df2['fit_valid'] == True),
           (df2['fit_K'] < 15000),
           (df2['UI_Valid'] == True),
           (df2['cell type'] == 'HoxB8-Macro'), 
           (df2['substrate'] == 'bare glass'),
           (df2['cell subtype'].apply(lambda x : x in ['ctrl','tko'])), 
           # (data['bead type'] == 'M450'),
           (df2['date'].apply(lambda x : x in dates))]
df2_f = filterDf(df2, Filters)

# Count
CountByCond, CountByCell = makeCountDf(df2_f, condCol)
CountByCond.to_csv(os.path.join(figDir, figSubDir, 'HoxB8_K-200-400'+'_count.txt'), sep='\t')

# Group By
df_fgw2 = dataGroup_weightedAverage(df2_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'fit_K', weightCol = 'fit_ciwK', weight_method = 'ciw^2')
df_fgw2['fit_K_wAvg'] /= 1000



# Plot
fig, axes = plt.subplots(1,2, figsize=(16/gs.cm_in, 6/gs.cm_in))

# 1.
ax = axes[0]
ax.set_ylim([60, 1100])
ax.set_yscale('log')
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter='bestH0',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
ax.grid(axis='y')
ax.set_ylim([60, 2000])
ax.set_xlabel('')
ax.set_ylabel('$H_0$ (nm)')

# Prettify
renameAxes(ax,renameDict1)

# 2.
ax = axes[1]
ax.set_ylim([0.6, 11])
ax.set_yscale('log')
fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter='fit_K_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
ax.grid(axis='y')
ax.set_ylim([0.6, 20])
ax.set_xlabel('')
ax.set_ylabel('$K$ (kPa)\n' + r"$\sigma \in$" + f"[{c-hw}, {c+hw}] Pa")

# Prettify
renameAxes(ax,renameDict1)


# Show
plt.tight_layout()
plt.show()

# Save
name = 'HoxB8_H_and_E'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 200,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %% Plots -- aSFL

# MecaData_MCA = taka3.getMergedTable('MecaData_MCA')

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


figDir = 'D:/MagneticPincherData/Figures/CellTypesDataset'
figSubDir = 'iMC_project'

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

styleDict = { '3T3':{'color':gs.colorList40[20],
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
                   'none':{'color':gs.colorList40[0],
                                                  'marker':'o'},
                   'doxycyclin':{'color':gs.colorList40[30],
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

# %%% bestH0 & K-200-400

gs.set_manuscript_options_jv()

df = MecaData_MCA
dates_r1 = ['21-01-18', '21-01-21']
dates_r2 = ['21-04-27', '21-04-28', '21-09-08']
dates_r3 = ['22-07-15', '22-07-20', '22-07-27']
all_dates = dates_r1 + dates_r2 + dates_r3

# Define 1.

drugs = ['none', 'doxycyclin']
substrates = ['BSA coated glass', '20um fibronectin discs']
parameterH = 'bestH0'

Filters = [(df['validatedThickness'] == True),
           (df['bestH0'] <= 850),
           (df['ctFieldThickness'] >= 50),
           (df['UI_Valid'] == True),
           (df['cell type'] == '3T3'), 
           (df['substrate'] == '20um fibronectin discs'),
           (df['cell subtype'].apply(lambda x : x in ['aSFL-A11'])), 
           (df['drug'].apply(lambda x : x in drugs)),
           (df['date'].apply(lambda x : x in all_dates)),
           (df['FluoSelection'] == True)]

df_f = filterDf(df, Filters)

df_f, condCol = makeCompositeCol(df_f, cols=['drug'])
print(df_f[condCol].unique())

# Order
co_order = ['none', 'doxycyclin']
box_pairs=[]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameterH], aggFun = 'mean')
# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
CountByCond.to_csv(os.path.join(figDir, figSubDir, 'iMC_H_'+'_count.txt'), sep='\t')

# Define 2.
fitType = 'stressRegion'
fitId = '300_100'
c, hw = np.array(fitId.split('_')).astype(int)
fitStr = 'Fit from {:.0f} to {:.0f} Pa'.format(c-hw, c+hw)
df2 = taka3.getFitsInTable(df, fitType=fitType, filter_fitID=fitId)

Filters = [(df2['validatedThickness'] == True),
           (df2['bestH0'] <= 850),
           (df2['ctFieldThickness'] >= 50),
           (df2['UI_Valid'] == True),
           (df2['cell type'] == '3T3'), 
           (df2['substrate'] == '20um fibronectin discs'),
           (df2['cell subtype'].apply(lambda x : x in ['aSFL-A11'])), 
           (df2['drug'].apply(lambda x : x in drugs)),
           (df2['date'].apply(lambda x : x in all_dates)),
           (df2['fit_valid'] == True),
           (df2['fit_K'] < 15000),
           (df2['FluoSelection'] == True)]

df2_f = filterDf(df2, Filters)

# Count
CountByCond, CountByCell = makeCountDf(df2_f, condCol)
CountByCond.to_csv(os.path.join(figDir, figSubDir, 'iMC_K-200-400'+'_count.txt'), sep='\t')

# Group By
df_fgw2 = dataGroup_weightedAverage(df2_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'fit_K', weightCol = 'fit_ciwK', weight_method = 'ciw^2')
df_fgw2['fit_K_wAvg'] /= 1000

# %%% Plot

# Order
co_order = ['none', 'doxycyclin']
box_pairs=[]

# 
fig, axes = plt.subplots(1,2, figsize=(16/gs.cm_in, 6/gs.cm_in))

# 1.
ax = axes[0]
ax.set_ylim([90, 1100])
ax.set_yscale('log')
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameterH,
                 co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
ax.grid(axis='y')
ax.set_ylim([90, 1300])
ax.set_xlabel('')
ax.set_ylabel('$H_0$ (nm)')

# Prettify
renameAxes(ax,renameDict1)

# 2.
ax = axes[1]
ax.set_ylim([0.9, 11])
ax.set_yscale('log')
fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter='fit_K_wAvg',
                 co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
ax.grid(axis='y')
ax.set_ylim([0.9, 13])
ax.set_xlabel('')
ax.set_ylabel('$K$ (kPa)\n' + r"$\sigma \in$" + f"[{c-hw}, {c+hw}] Pa")

# Prettify
renameAxes(ax,renameDict1)


# Show
plt.tight_layout()
plt.show()

# Save
name = 'iMC_H_and_E'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 200,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %% Plots - All cell types

figDir = 'D:/MagneticPincherData/Figures/CellTypesDataset'
figSubDir = '24-07-03'

# %%% bestH0

# Define
df = MecaData_Cells
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs']
parameter = 'bestH0'

# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'].apply(lambda x : x in substrates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['bestH0'] <= 1000),
           ]
df_f = filterDf(df, Filters)
df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])
print(df_f[condCol].unique())

# Order
# co_order = ['none', 'dmso']
co_order = []

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(12, 8))
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
# rD = {'none' : 'No drug',
#       'dmso' : 'DMSO', 
#       }

# renameAxes(ax, rD, format_xticks = True)
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
ufun.archiveFig(fig, name = 'Thickness_CellTypes', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%% E_400

# Define
df = MecaData_Cells
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs']
parameter = 'E_f_<_400'
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'].apply(lambda x : x in substrates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           ]
df_f = filterDf(df, Filters)
df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])


# Order
# co_order = ['none', 'dmso']
co_order = []

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_fgw2['E_f_<_400_wAvg'] *= 1e-3

# Plot
fig, ax = plt.subplots(1,1, figsize=(12, 8))
fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
           
# Prettify
# rD = {'none' : 'No drug',
#       'dmso' : 'DMSO', 
#       }

# renameAxes(ax, rD, format_xticks = True)
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
ufun.archiveFig(fig, name = 'Stiffness_CellTypes', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%% E_400 vs bestH0 - By cells

# Define
df = MecaData_Cells
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs']
xCol = 'bestH0'
yCol = 'E_f_<_400_wAvg'
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'].apply(lambda x : x in substrates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['bestH0'] >= 80),
           (df['bestH0'] <= 1000),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           ]
df_f = filterDf(df, Filters)
df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'
df_f = df_f.dropna()

df_f, condCol = makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])


# Order
# co_order = ['none', 'dmso']
co_order = ['3T3 & Atcc-2023', '3T3 & aSFL-A11', 'DC & mouse-primary', 'Dicty & DictyBase-WT', 'HoxB8-Macro & ctrl', 'MDCK & WT']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [xCol], aggFun = 'mean').reset_index(drop=True)
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_fgw2['E_f_<_400_wAvg'] *= 1e-3

df_fgm = df_fg.merge(df_fgw2, on='cellID', how='inner', suffixes = ('', '_2'))

print(df_fgm[condCol].unique())

# Plot
fig, ax = plt.subplots(1,1, figsize=(12, 8))
ax.set_yscale('log')
ax.set_xscale('log')
# fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
#                  co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1,
#                  stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
#                  showMean = False)

sns.scatterplot(ax=ax, data=df_fgm, x=xCol, y=yCol, hue=condCol, palette='Set1', hue_order = co_order)

for cond, color in zip(co_order, gs.cL_Set12[:len(co_order)]):
    df_fgm_c = df_fgm[df_fgm[condCol] == cond]
    xfit = np.log(df_fgm_c[xCol].values)
    yfit = np.log(df_fgm_c[yCol].values)
    parms, results = ufun.fitLineHuber(xfit, yfit)
    [A, k] = parms
    pval = results.pvalues[1] # pvalue on the param 'k'
    eqnText = f"Fit ; Y = {np.exp(A):.1e} * X^{k:.1f}"
    eqnText += f"\np-val = {pval:.3f}".format()
    # xplot = np.linspace(min(xfit), max(xfit), 50)
    xplot = np.linspace(min(df_fgm_c[xCol].values), max(df_fgm_c[xCol].values), 50)
    yplot = (np.exp(A) * (xplot**(k)))
    ax.plot(xplot, yplot, color=color, ls='--', lw=1.5, label = eqnText)

ax.legend()
# Prettify
# rD = {'none' : 'No drug',
#       'dmso' : 'DMSO', 
#       }

# renameAxes(ax, rD, format_xticks = True)
# renameAxes(ax, renameDict, format_xticks = True)
# renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
# ax.set_xlabel('')

# Plot 2
N = len(co_order)
fig2, axes2 = plt.subplots(1, N, figsize=(3.5 * N, 3.5), sharey=True)

for k in range(N):
    ax = axes2[k]
    cond, color = co_order[k], gs.cL_Set12[k]
    df_fgm_c = df_fgm[df_fgm[condCol] == cond]
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    sns.scatterplot(ax=ax, data=df_fgm_c, x=xCol, y=yCol, color=color)
    
    xfit = np.log(df_fgm_c[xCol].values)
    yfit = np.log(df_fgm_c[yCol].values)
    parms, results = ufun.fitLineHuber(xfit, yfit)
    [A, k] = parms
    pval = results.pvalues[1] # pvalue on the param 'k'
    eqnText = f"Fit ; Y = {np.exp(A):.1e} * X^{k:.1f}"
    eqnText += f"\np-val = {pval:.3f}".format()
    # xplot = np.linspace(min(xfit), max(xfit), 50)
    xplot = np.linspace(min(df_fgm_c[xCol].values), max(df_fgm_c[xCol].values), 50)
    yplot = (np.exp(A) * (xplot**(k)))
    ax.plot(xplot, yplot, color=color, ls='--', lw=1.5, label = eqnText)
    
    ax.legend()
    # Prettify
    # rD = {'none' : 'No drug',
    #       'dmso' : 'DMSO', 
    #       }
    
    # renameAxes(ax, rD, format_xticks = True)
    # renameAxes(ax, renameDict, format_xticks = True)
    # renameLegend(ax, renameDict)
    ax.grid(visible=True, which='both', axis='both')
    # ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = 'StiffnessVThickness_ByComp_CellTypes_V1', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig2, name = 'StiffnessVThickness_ByComp_CellTypes_V2', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%% E_400 vs bestH0 - By comps
gs.set_manuscript_options_jv(palette = 'Set2')

# Define
df = MecaData_Cells
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs']
xCol = 'bestH0'
yCol = 'E_f_<_400'
# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'].apply(lambda x : x in substrates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['bestH0'] >= 80),
           (df['bestH0'] <= 1000),
           (df['E_f_<_400'] <= 2e4),
           (df['valid_f_<_400'] == True), 
           ]
df_f = filterDf(df, Filters)
df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])


# Order
# co_order = ['none', 'dmso']
co_order = ['3T3 & Atcc-2023', '3T3 & aSFL-A11', 'DC & mouse-primary', 'Dicty & DictyBase-WT', 'HoxB8-Macro & ctrl']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [xCol], aggFun = 'mean').reset_index(drop=True)
# df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
#                                       valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_f['E_f_<_400'] *= 1e-3

# df_fgm = df_fg.merge(df_fgw2, on='cellID', how='inner', suffixes = ('', '_2'))

print(df_fgm[condCol].unique())

# Plot
fig, ax = plt.subplots(1,1, figsize=(12, 8))
ax.set_yscale('log')
ax.set_xscale('log')
# fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
#                  co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1,
#                  stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
#                  showMean = False)

sns.scatterplot(ax=ax, data=df_f, x=xCol, y=yCol, hue=condCol, palette='Set1', hue_order = co_order)

for cond, color in zip(co_order, gs.cL_Set12[:len(co_order)]):
    df_f_c = df_f[df_f[condCol] == cond]
    xfit = np.log(df_f_c[xCol].values)
    yfit = np.log(df_f_c[yCol].values)
    parms, results = ufun.fitLineHuber(xfit, yfit)
    [A, k] = parms
    pval = results.pvalues[1] # pvalue on the param 'k'
    eqnText = f"Fit ; Y = {np.exp(A):.1e} * X^{k:.1f}"
    eqnText += f"\np-val = {pval:.3f}".format()
    # xplot = np.linspace(min(xfit), max(xfit), 50)
    xplot = np.linspace(min(df_f_c[xCol].values), max(df_f_c[xCol].values), 50)
    yplot = (np.exp(A) * (xplot**(k)))
    ax.plot(xplot, yplot, color=color, ls='--', lw=1.5, label = eqnText)

ax.legend()
# Prettify
# rD = {'none' : 'No drug',
#       'dmso' : 'DMSO', 
#       }

# renameAxes(ax, rD, format_xticks = True)
# renameAxes(ax, renameDict, format_xticks = True)
# renameLegend(ax, renameDict)
ax.grid(visible=True, which='both', axis='both')
# ax.set_xlabel('')


# Plot 2
N = len(co_order)
fig2, axes2 = plt.subplots(1, N, figsize=(3.5 * N, 3.5), sharex=True, sharey=True)

for k in range(N):
    ax = axes2[k]
    cond, color = co_order[k], gs.cL_Set12[k]
    df_f_c = df_f[df_f[condCol] == cond]
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    sns.scatterplot(ax=ax, data=df_f_c, x=xCol, y=yCol, color=color)
    
    xfit = np.log(df_f_c[xCol].values)
    yfit = np.log(df_f_c[yCol].values)
    parms, results = ufun.fitLineHuber(xfit, yfit)
    [A, k] = parms
    pval = results.pvalues[1] # pvalue on the param 'k'
    eqnText = f"Fit ; Y = {np.exp(A):.1e} * X^{k:.1f}"
    eqnText += f"\np-val = {pval:.3f}".format()
    # xplot = np.linspace(min(xfit), max(xfit), 50)
    xplot = np.linspace(min(df_f_c[xCol].values), max(df_f_c[xCol].values), 50)
    yplot = (np.exp(A) * (xplot**(k)))
    ax.plot(xplot, yplot, color=color, ls='--', lw=1.5, label = eqnText)
    
    ax.legend()
    # Prettify
    # rD = {'none' : 'No drug',
    #       'dmso' : 'DMSO', 
    #       }
    
    # renameAxes(ax, rD, format_xticks = True)
    # renameAxes(ax, renameDict, format_xticks = True)
    # renameLegend(ax, renameDict)
    ax.grid(visible=True, which='both', axis='both')
    # ax.set_xlabel('')

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Show
plt.tight_layout()
plt.show()

# Save
ufun.archiveFig(fig, name = 'StiffnessVThickness_ByComp_CellTypes_V1', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig2, name = 'StiffnessVThickness_ByComp_CellTypes_V2', ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
 

# %% Plots - All cell types - Manuscript

figDir = 'D:/MagneticPincherData/Figures/CellTypesDataset_V2'
figSubDir = 'Soutenance'

df = MecaData_Cells
print(df['cell type'].unique())
print(df['cell subtype'].unique())

co_order_2 = ['3T3 & Atcc-2023', 
            '3T3 & aSFL-A11', 
            'DC & mouse-primary', 
            'Dicty & DictyBase-WT', 
            'HoxB8-Macro & ctrl', 
            'MDCK & WT']

co_order = ['3T3 & Atcc-2023', 
            '3T3 & aSFL-A11', 
            'MDCK & WT',
            'DC & mouse-primary', 
            'HoxB8-Macro & ctrl', 
            'Dicty & DictyBase-WT',
            ]

rD = {'3T3 & Atcc-2023'      : '3T3 ATCC', 
      '3T3 & aSFL-A11'       : r'3T3 $\alpha$SFL',  
      'DC & mouse-primary'   : 'Primary DC',  
      'Dicty & DictyBase-WT' : 'Dictys Ax3',  
      'HoxB8-Macro & ctrl'   : 'HoxB8 Macro',  
      'MDCK & WT'            : 'MDCK',
      }

# %%% bestH0 & E400

gs.set_manuscript_options_jv()

# Define
df = MecaData_Cells
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs']
parameter = 'bestH0'

# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'].apply(lambda x : x in substrates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['bestH0'] <= 1000),
           ]
df_f = filterDf(df, Filters)
df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])

# Order
# co_order = ['none', 'dmso']
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[6], bp[7], bp[8]]
# bp = bp[:5] + [bp[9], bp[12], bp[-1]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(2,1, figsize=(10/gs.cm_in, 16/gs.cm_in), sharex=True)
ax = axes[0]
ax.set_yscale('log')

fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 0.8,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
           
# Prettify
# rD = {'none & 0.0' : 'No drug',
#       'Y27 & 10.0' : 'Y27\n10 µM', 
#       'Y27 & 50.0' : 'Y27\n50 µM', 
#       }

renameAxes(ax, rD, format_xticks = True, rotation=15)
# renameAxes(ax, renameDict, format_xticks = True)
# renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
ax.set_ylabel('$H_0$ (nm)')


# # Show
# plt.tight_layout()
# plt.show()


# # Count
# CountByCond, CountByCell = makeCountDf(df_f, condCol)
# # Save
# name = 'CellTypes_bestH0'
# ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
#                 figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
# ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
#                 figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
# CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')




# # %%% E400

gs.set_manuscript_options_jv('Set2')
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=gs.cL_Set21) 


# Define
df = MecaData_Cells
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs']

parameter = 'E_f_<_400'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'].apply(lambda x : x in substrates)),
           (df['E_f_<_400'] <= 1e5),
           (df['valid_f_<_400'] == True),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           ]

df_f = filterDf(df, Filters)
df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])

# print(df['cell type'].unique())
# print(df['cell subtype'].unique())

# Order
# co_order = ['none', 'dmso']
bp = makeBoxPairs(co_order)
# bp = [bp[0], bp[6], bp[7], bp[8]]
bp = bp[:5] + [bp[9], bp[12], bp[-1]]

# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_fgw2[parameter+'_wAvg'] /= 1000

# Plot
# fig, ax = plt.subplots(1,1, figsize=(17/gs.cm_in, 12/gs.cm_in))
ax = axes[1]
ax.set_yscale('log')

fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 0.8,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False, statpos = 'in',
                 showMean = False)
           
# Prettify
renameAxes(ax, rD, format_xticks = True, rotation=15)
# renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
ax.set_ylabel('$E_{400}$ (kPa)')

# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# name = 'CellTypes_E400'
name = 'CellTypes_HandE400'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%% Fluctuations

gs.set_manuscript_options_jv()

# Define
df = MecaData_Cells
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs']

XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'].apply(lambda x : x in substrates)),
           (df[XCol] <= 1000),
           (df[YCol] <= 800),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           ]

df_f = filterDf(df, Filters)
df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])

index_f = df_f[(df_f['cell type'] == 'HoxB8-Macro') & (df_f[XCol] >= 580)].index
df_f = df_f.drop(index_f, axis=0)


# Order
# co_order = ['none', 'dmso']
bp = makeBoxPairs(co_order)
# bp = [bp[0], bp[6], bp[7], bp[8]]
bp = bp[:5] + [bp[9], bp[12], bp[-1]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], 
                  numCols = [XCol, YCol], aggFun = 'mean')

# All

# # Plot
# fig, ax = plt.subplots(1,1, figsize=(8,5))
    
# fig, ax = D2Plot_wFit(df_fg, fig = fig, ax = ax, 
#                 XCol=XCol, YCol=YCol, condition=condCol, co_order = co_order,
#                 modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
#                 figSizeFactor = 1, markersizeFactor = 1)
           
# # Prettify
# rD = {'none & 0.0' : 'No drug',
#       'Y27 & 10.0' : 'Y27 (10µM)', 
#       'Y27 & 50.0' : 'Y27 (50µM)', 
#       # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
#       # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
#       }

# ax.grid(visible=True, which='major', axis='both', zorder=0)
# renameAxes(ax, rD, format_xticks = True)
# renameAxes(ax, renameDict, format_xticks = True)
# renameLegend(ax, rD)

# # Count
# CountByCond, CountByCell = makeCountDf(df_f, condCol)

# # Show
# plt.tight_layout()
# plt.show()

# # Save
# ufun.archiveFig(fig, name = figname, ext = '.pdf', dpi = 100,
#                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# 1 by 1

# Plot
fig, axes = plt.subplots(2, 3, figsize=(17/gs.cm_in, 10/gs.cm_in), sharex=True, sharey=True)
axes = axes.flatten('C')

for i in range(len(axes)):
    ax = axes[i]
    df_fc = df_fg[df_fg[condCol] == co_order[i]]
    # color = styleDict[co_order[i]]['color']
    color = gs.cL_Set2[i]
    
    # fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
    #                 XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
    #                 modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
    #                 figSizeFactor = 1, markersizeFactor = 0.5)
    
        
    sns.scatterplot(ax = ax, x=df_fc[XCol].values, y=df_fc[YCol].values, 
                    marker = 'o', s = 20, color = color, edgecolor = 'k', linewidth =0.5, alpha = 0.5,
                    zorder = 3)
    Xfit, Yfit = (df_fc[XCol].values), (df_fc[YCol].values)
    
    [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
    a_cihw = (results.conf_int(0.05)[1, 1] - results.conf_int(0.05)[1, 0])/2
    R2 = w_results.rsquared
    pval = results.pvalues[1]
    Xplot = (np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = a * Xplot + b
    
    # [b, a], results = ufun.fitLine(Xfit, Yfit)
    # R2 = results.rsquared
    # pval = results.pvalues[1]
    # Xplot = (np.linspace(min(Xfit), max(Xfit), 50))
    # Yplot = a * Xplot + b
    
    ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 2.0, zorder = 6,
            label =  r'$\bf{Fit\ y=ax+b}$' + \
                    f'\na={a:.2f}' + r'$\pm$' + f'{a_cihw:.2f}' + \
                    f' | b={b:.1f}' + \
                    f'\n$R^2$ = {R2:.2f}' + \
                    f' | p-val = {pval:.2f}')
            # label =  r'$\bf{Fit\ y=ax+b}$' + \
            #         f'\na = {a:.2f}' + r'$\pm$' + f'{a_cihw:.2f}' + \
            #         f'\nb = {b:.2f}' + \
            #         f'\n$R^2$ = {R2:.2f}' + \
            #         f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 6, handlelength=1) # , loc = 'upper right'
    ax.set_xlabel('$H_{5mT}$ (nm)')
    ax.set_ylabel('Fluctuations ampli. (nm)', fontsize=9) # [$D_9$-$D_1$]
    ax.set_xlim([0, 950])
    ax.set_title(co_order[i])
    if i%3 != 0:
        ax.set_ylabel('')
        
           
# Prettify
# rD = {'none & 0.0' : 'No drug',
#       'Y27 & 10.0' : 'Y27 (10µM)', 
#       'Y27 & 50.0' : 'Y27 (50µM)', 
#       # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
#       # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
#       }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder=0)
    renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD, loc='upper left')
    # ax.set_xlim([0, 600])
    # ax.set_ylim([0, 300])
    ax.set_xlabel('$H_{5mT}$ (nm)')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
name = 'CellTypes_Fluctuations'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')



fig, ax = plt.subplots(1, 1, figsize=(17/gs.cm_in, 9/gs.cm_in))
df_fg['Fluctu_ratio'] = df_fg[YCol]/df_fg[XCol]

Filters = [(df_fg['Fluctu_ratio'] <= 2.1), 
            ]
df_fg = filterDf(df_fg, Filters)

# ax.set_yscale('log')
# bp = makeBoxPairs(co_order)
# bp = [bp[1]]
fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter='Fluctu_ratio',
                 co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 0.9,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)

ax.set_ylabel('Fluctuations ratio') #  [$D_9$-$D_1$]
ax.set_xlabel('') 
ax.grid(visible=True, which='major', axis='both', zorder=0)
renameAxes(ax, rD, format_xticks = True, rotation=0)

# Show
plt.tight_layout()
plt.show()

# Save
name = 'CellTypes_FluctuationsRatio'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%% Stress ranges

gs.set_manuscript_options_jv(palette = 'Set1')


# Define
df = MecaData_Cells
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs']
parameter = 'bestH0'

# figname = 'bestH0' + drugSuffix

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'].apply(lambda x : x in substrates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['bestH0'] <= 1000),
           (df['valid_f_<_400'] == True), 
           (df['minStress'] > 0),
           (df['maxStress'] < 5e4), 
           ]
df_f = filterDf(df, Filters)
df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])


# fig, ax = plt.subplots(1,1, figsize=(17/gs.cm_in, 10/gs.cm_in))



# StressRange_2D_V2(df_f, fig = fig, ax = ax, condition=condCol, defaultMode = False)
# fig, axes = StressRange_2D(df_f, condition=condCol, split = True, defaultColors = True)

Nco = len(co_order)
colorList, markerList = gs.colorList30, gs.markerList10
fig, axes = plt.subplots(3, 2, figsize = (17/gs.cm_in, 12/gs.cm_in), sharex = True, sharey = True)
axes = axes.flatten('C')

for i in range(Nco):
    ax = axes[i]
    co = co_order[i]
    data_co = df_f[df_f[condCol] == co]
    color = colorList[i]
    ax.scatter(data_co['bestH0'], data_co['minStress'], marker = 'o', s = 3, color = color, edgecolor = 'k', zorder=4)
    ax.scatter(data_co['bestH0'], data_co['maxStress'], marker = 'o', s = 3, color = color, edgecolor = 'k', zorder=4)
    ax.vlines(data_co['bestH0'], data_co['minStress'], data_co['maxStress'], color = color, label = rD[co])
    ax.set_yscale('log')
    ax.set_xlim([0, 1050])
    ax.set_ylim([1e1, 1e4])
    ax.set_xlabel('$H_0$ (nm)')
    ax.set_ylabel('Stress range (Pa)')
    ax.grid(axis='both', which='major')
    ax.legend(loc='upper right')
    # print(i%2)
    # print(i%3)
    if i<4:
        ax.set_xlabel('')
    if i%2!=0:
        ax.set_ylabel('')


fig.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
name = 'CellTypes_sigmaRanges'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%% Non-linearity

# %%%% NLI

# gs.set_manuscript_options_jv(palette = 'Set1')
gs.set_defense_options_jv()

# Define
df = MecaData_Cells
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs']
parameter = 'bestH0'

# figname = 'bestH0' + drugSuffix

co_order = [
    'MDCK & WT',
    '3T3 & Atcc-2023', 
            'Dicty & DictyBase-WT',
            'DC & mouse-primary', 
            
            ]

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'].apply(lambda x : x in substrates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['bestH0'] <= 1000),
           (df['valid_f_<_400'] == True), 
           (df['minStress'] > 0),
           (df['maxStress'] < 5e4), 
           ]
df_f = filterDf(df, Filters)
df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])

df_f = computeNLMetrics_V2(df_f, th_NLI = np.log10(2), ref_strain = 0.2)

bp = makeBoxPairs(co_order)
# bp = [bp[0], bp[6], bp[7], bp[8]]
# bp = bp[:5] + [bp[9], bp[12], bp[-1]]

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)

# Plot
# fig, ax = plt.subplots(1,1, figsize=(14/gs.cm_in, 10/gs.cm_in))
fig, ax = plt.subplots(1,1, figsize=(12/gs.cm_in, 10/gs.cm_in))

# NLI_mod
parameter = 'NLI_mod'
ax = ax
# ax.set_yscale('log')
# ax.set_ylim([50, 2750])
ax.grid(visible=True, which='major', axis='y', zorder=-1)
ax.axhline(0, color='gray', lw=1, zorder=1)

fig, ax = D1Plot_violin(df_f, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                  co_order = co_order, figSizeFactor = 1, 
                  stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False)


# fig, ax = D1Plot(df_f, fig = fig, ax = ax, condition=condCol, parameter=parameter,
#                   co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 0.4,
#                   stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
#                   showMean = False, edgecolor='gray')

# bins = np.linspace(-3, 2, 11)
# for i in range(len(co_order)):
#     df_fc = df_f[df_f[condCol] == co_order[i]]
#     x = df_fc['NLI_mod'].values
#     color = styleDict[co_order[i]]['color']
#     histo, bins = np.histogram(x, bins=bins)
#     histo = histo / np.sum(histo)
#     # if i == 0:
#     #     ax.stairs(histo, edges=bins, color = color, label = co_order[i], fill = True, lw = 1.5, zorder=0)
#     # else:
#     ax.stairs(histo, edges=bins, color = color, label = co_order[i], fill = False, lw = 1.5, zorder=3)

           
# Prettify
renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True, rotation = 10)
# ax.grid(visible=True, which='major', axis='y', zorder=-1)
ax.set_xlabel('')
ax.set_ylabel('NLR')




# Show
# plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'Y27'
# drugSuffix = '_Y27'
# name = 'bestH0' + drugSuffix
name = 'CellTypes_' + 'NLR_4types'
figDir = 'C:/Users/JosephVermeil/Desktop/Manuscrit/Soutenance/python_plots'
figSubDir = ''
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 400,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%%% K-sigma

gs.set_manuscript_options_jv()

# Define
df = MecaData_Cells
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs']

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'].apply(lambda x : x in substrates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['bestH0'] <= 1000),
           ]
df_f = filterDf(df, Filters)

df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
Ncells = CountByCond.cellCount.values[0]
Ncomps = CountByCond.compCount.values[0]

# Order
# co_order = ['none', 'dmso']
bp = makeBoxPairs(co_order)
# bp = [bp[0], bp[6], bp[7], bp[8]]
bp = []

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
# fig, axes = plt.subplots(3, 1, figsize=(17/gs.cm_in, 20/gs.cm_in), sharex=True, sharey=True)

# intervals = ['200_500', '300_700', '400_900']
intervals = ['200_600']
Ni = len(intervals)
cL = plt.cm.plasma(np.linspace(0.1, 0.9, Ni))
fig, axes = plt.subplots(Ni, 1, figsize=(12/gs.cm_in, 8/gs.cm_in), sharex=True, sharey=True)

# rD = {'none & 0.0' : 'No drug',
#       'Y27 & 10.0' : 'Y27 10 µM', 
#       'Y27 & 50.0' : 'Y27 50 µM', 
#       }

for k, cond in enumerate(co_order):  
    for i, interval in enumerate(intervals):
        ax = axes
        ax.set_yscale('log')
        df_fc = df_f[df_f[condCol] == cond]        
        
        # colorDict = {cond:cL[i]}
        # labelDict = {cond:f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"}
        # colorDict = {cond:styleDict[cond]['color']}
        colorDict = {cond:gs.cL_Set2[k]}
        labelDict = {cond:rD[cond]}
        try:
            plotPopKS_V2(df_fc, fig = fig, ax = ax, condition = condCol, co_order = [], 
                         colorDict = colorDict, labelDict = labelDict, markersizefactor = 0.75,
                         fitType = 'stressGaussian', fitWidth=75, mode = interval, Sinf = 0, Ssup = np.Inf)
        except:
            pass
    
        # Prettify
        ax.grid(visible=True, which='both', axis='y')
        # legendTitle = r'$\bf{Whole\  dataset}$ | ' + f'{Ncells} cells | {Ncomps} comp'
        legendTitle = r"$\sigma \in$" + f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"
        ax.legend(fontsize = 7, title = legendTitle, title_fontsize = 8) # , '600_800'
        # ax.legend().set_visible(False)
        ax.set_xlim(0, 800)
        ax.set_ylim(0.5, 30)
        if i != len(intervals)-1:
            ax.set_xlabel('')


# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
name = 'CellTypes_K-sigma-3ranges'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%% Thickness & Stiffness

# %%%% 1. E400

gs.set_manuscript_options_jv()

# Define
df = MecaData_Cells
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs']

XCol = 'bestH0'
YCol = 'E_f_<_400'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'].apply(lambda x : x in substrates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['valid_f_<_400'] == True),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])

# Order
# co_order = ['none', 'dmso']
bp = makeBoxPairs(co_order)
# bp = [bp[0], bp[6], bp[7], bp[8]]
bp = []

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(3, 2, figsize=(17/gs.cm_in, 24/gs.cm_in), sharex=True, sharey=True)
axes = axes.flatten('C')

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

YCol += '_wAvg'
df_plot[YCol] /= 1000

for i in range(len(axes)):
    ax = axes[i]
    ax.set_xscale('log')
    ax.set_yscale('log')
    df_fc = df_plot[df_plot[condCol] == co_order[i]]
    print(co_order[i], len(df_fc))
    color = gs.colorList10[i]
    
    # fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
    #                 XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
    #                 modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
    #                 figSizeFactor = 1, markersizeFactor = 0.5)
    
        
    sns.scatterplot(ax = ax, x=df_fc[XCol].values, y=df_fc[YCol].values, 
                    marker = 'o', s = 20, color = color, edgecolor = 'k', linewidth=0.5, alpha = 0.5,
                    zorder = 3)
    Xfit, Yfit = np.log(df_fc[XCol].values), np.log(df_fc[YCol].values)
    
    [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
    A, k = np.exp(b), a
    k_cihw = (results.conf_int(0.05)[1, 1] - results.conf_int(0.05)[1, 0])/2
    R2 = w_results.rsquared
    pval = results.pvalues[1]
    Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = A * Xplot**k
    
    # [b, a], results = ufun.fitLine(Xfit, Yfit)
    # R2 = results.rsquared
    # pval = results.pvalues[1]
    # Xplot = (np.linspace(min(Xfit), max(Xfit), 50))
    # Yplot = a * Xplot + b
    
    ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 2.0, zorder = 6,
            label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + \
                        f'\nA = {A:.1e}' + \
                        f'\nk  = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
                        f'\n$R^2$ = {R2:.2f}' + \
                        f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8, loc = 'lower left')
    ax.set_xlabel('$H_0}$ (nm)')
    ax.set_ylabel('$E_{400}$ (kPa)')
    ax.set_title(co_order[i])
    if i%2 != 0:
        ax.set_ylabel('')
           
# Prettify
rD.update({'E_f_<_400_wAvg':'Elastic modulus (Pa)\nfor F < 400pN - wAvg'})

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder=0)
    renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD)
    ax.set_xlim(50, 1000)
    ax.set_ylim(0.4, 100)

# axes[0].set_xlabel('')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
name = 'CellTypes_H0-E400'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%%% 3. Eeff

gs.set_manuscript_options_jv()

# Define
df = MecaData_Cells
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs']

XCol = 'bestH0'
YCol = 'E_f_<_400'

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'].apply(lambda x : x in substrates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['valid_f_<_400'] == True),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])
df_f = computeNLMetrics_V2(df_f, th_NLI = np.log10(2), ref_strain = 0.2)

YCol = 'E_eff'

# Filter
Filters = [
           (df_f[YCol] <=  1e5),
           ]
df_f = filterDf(df_f, Filters)

# Order
co_order = ['3T3 & Atcc-2023', 
            '3T3 & aSFL-A11', 
            'MDCK & WT',
            'DC & mouse-primary', 
            'HoxB8-Macro & ctrl', 
            'Dicty & DictyBase-WT', 
            ]

colorsD = {'3T3 & Atcc-2023'      : gs.cL_Set2[0], 
      '3T3 & aSFL-A11'       : gs.cL_Set2[1],  
      'DC & mouse-primary'   : gs.cL_Set2[2],  
      'Dicty & DictyBase-WT' : gs.cL_Set2[3],  
      'HoxB8-Macro & ctrl'   : gs.cL_Set2[4],  
      'MDCK & WT'            : gs.cL_Set2[5],
      }

rD = {'3T3 & Atcc-2023'      : '3T3 ATCC', 
      '3T3 & aSFL-A11'       : r'3T3 $\alpha$SFL',  
      'DC & mouse-primary'   : 'Primary DC',  
      'Dicty & DictyBase-WT' : 'Dictys Ax3',  
      'HoxB8-Macro & ctrl'   : 'HoxB8 Macro',  
      'MDCK & WT'            : 'MDCK',
      }

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_eff', weightCol = 'ciwE_eff', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(2, 3, figsize=(17/gs.cm_in, 12/gs.cm_in), sharex=True, sharey=True)
axes = axes.flatten('C')

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

YCol += '_wAvg'
df_plot[YCol] /= 1000

for i in range(len(axes)):
    ax = axes[i]
    ax.set_xscale('log')
    ax.set_yscale('log')
    df_fc = df_plot[df_plot[condCol] == co_order[i]]
    print(co_order[i], len(df_fc))
    # color = gs.cL_Set2[i]
    color = colorsD[co_order[i]]
    
    # fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
    #                 XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
    #                 modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
    #                 figSizeFactor = 1, markersizeFactor = 0.5)
    
        
    sns.scatterplot(ax = ax, x=df_fc[XCol].values, y=df_fc[YCol].values, 
                    marker = 'o', s = 20, color = color, edgecolor = 'k', linewidth =0.5, alpha = 0.5,
                    zorder = 3)
    Xfit, Yfit = np.log(df_fc[XCol].values), np.log(df_fc[YCol].values)
    
    [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
    A, k = np.exp(b), a
    k_cihw = (results.conf_int(0.05)[1, 1] - results.conf_int(0.05)[1, 0])/2
    R2 = w_results.rsquared
    pval = results.pvalues[1]
    Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = A * Xplot**k
    
    # [b, a], results = ufun.fitLine(Xfit, Yfit)
    # R2 = results.rsquared
    # pval = results.pvalues[1]
    # Xplot = (np.linspace(min(Xfit), max(Xfit), 50))
    # Yplot = a * Xplot + b
    
    ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 2.0, zorder = 6,
            label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + \
                        f'\nA = {A:.1e}' + \
                        f'\nk  = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
                        f'\n$R^2$ = {R2:.2f}' + \
                        f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 6, loc = 'best', handlelength=1)
    ax.set_xlabel('$H_0}$ (nm)')
    ax.set_ylabel('$E_{eff}$ (kPa)')
    ax.set_title(co_order[i])
    if i%3 != 0:
        ax.set_ylabel('')
           
# Prettify
rD.update({'E_eff_wAvg':'E_{eff} (kPa)'})

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder=0)
    renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD)
    ax.set_xlim(50, 2000)
    ax.set_ylim(0.4, 300)

# axes[0].set_xlabel('')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
name = 'CellTypes_H0-Eeff'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%%% 3. Eeff - V Soutenance

gs.set_defense_options_jv()

# Define
df = MecaData_Cells2
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs']

XCol = 'bestH0'
YCol = 'E_f_<_400'

excluded_dates = ['23-03-08', '23-02-23', '23-11-26']

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'].apply(lambda x : x in substrates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['date'].apply(lambda x : x not in excluded_dates)),
           (df['valid_f_<_400'] == True),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])
df_f = computeNLMetrics_V2(df_f, th_NLI = np.log10(2), ref_strain = 0.2)

YCol = 'E_eff'

# Filter
Filters = [
           (df_f[YCol] <=  1e5),
           ]
df_f = filterDf(df_f, Filters)

# Order
co_order = ['3T3 & Atcc-2023', 
            '3T3 & aSFL-A11', 
            'MDCK & WT',
            'DC & mouse-primary', 
            'HoxB8-Macro & ctrl', 
            'Dicty & DictyBase-WT', 
            ]

colorsD = {'3T3 & Atcc-2023'      : gs.cL_Set2[0], 
      '3T3 & aSFL-A11'       : gs.cL_Set2[1],  
      'DC & mouse-primary'   : gs.cL_Set2[2],  
      'Dicty & DictyBase-WT' : gs.cL_Set2[3],  
      'HoxB8-Macro & ctrl'   : gs.cL_Set2[4],  
      'MDCK & WT'            : gs.cL_Set2[5],
      }

rD = {'3T3 & Atcc-2023'      : '3T3 ATCC', 
      '3T3 & aSFL-A11'       : r'3T3 $\alpha$SFL',  
      'DC & mouse-primary'   : 'Primary DC',  
      'Dicty & DictyBase-WT' : 'Dictys Ax3',  
      'HoxB8-Macro & ctrl'   : 'HoxB8 Macro',  
      'MDCK & WT'            : 'MDCK',
      }

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_eff', weightCol = 'ciwE_eff', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(2, 3, figsize=(17/gs.cm_in, 12/gs.cm_in), sharex=True, sharey=True)
axes = axes.flatten('C')

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

YCol += '_wAvg'
df_plot[YCol] /= 1000

for i in range(len(axes)):
    ax = axes[i]
    ax.set_xscale('log')
    ax.set_yscale('log')
    df_fc = df_plot[df_plot[condCol] == co_order[i]]
    print(co_order[i], len(df_fc))
    # color = gs.cL_Set2[i]
    color = colorsD[co_order[i]]
    
    # fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
    #                 XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
    #                 modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
    #                 figSizeFactor = 1, markersizeFactor = 0.5)
    
    medianX = np.median(df_fc[XCol].values)
    medianY = np.median(df_fc[YCol].values)
    
    if i == 0:
        alpha = 0.3
        s = 10
        medianX = 259
        medianY = 4.8
    else:
        alpha = 0.5
        s = 20
    
    sns.scatterplot(ax = ax, x=df_fc[XCol].values, y=df_fc[YCol].values, 
                    marker = 'o', s = s, color = color, edgecolor = 'k', linewidth =0.5, alpha = alpha,
                    zorder = 3, label = 'Median $H_0$ = ' + f'{medianX:.0f} nm'\
                                        f'\nMedian $E$ = ' + f'{medianY:.1f} kPa')
    Xfit, Yfit = np.log(df_fc[XCol].values), np.log(df_fc[YCol].values)
    
    
    [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
    A, k = np.exp(b), a
    k_cihw = (results.conf_int(0.05)[1, 1] - results.conf_int(0.05)[1, 0])/2
    R2 = w_results.rsquared
    pval = results.pvalues[1]
    Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = A * Xplot**k
    
    # [b, a], results = ufun.fitLine(Xfit, Yfit)
    # R2 = results.rsquared
    # pval = results.pvalues[1]
    # Xplot = (np.linspace(min(Xfit), max(Xfit), 50))
    # Yplot = a * Xplot + b
    
    # ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 2.0, zorder = 6,
    #         label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + \
    #                     f'\nA = {A:.1e}' + \
    #                     f'\nk  = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
    #                     f'\n$R^2$ = {R2:.2f}' + \
    #                     f'\np-val = {pval:.2f}')
        
    ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 2.0, zorder = 6,
            ) # label =  r'$Fit\ y\ =\ A.x^k$'

    ax.legend(fontsize = 7, loc = 'best', handlelength=1)
    ax.set_xlabel('$H_0}$ (nm)')
    ax.set_ylabel('$E_{eff}$ (kPa)')
    ax.set_title(co_order[i])
    if i%3 != 0:
        ax.set_ylabel('')
           
# Prettify
rD.update({'E_eff_wAvg':'E_{eff} (kPa)'})

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder=0)
    renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD)
    ax.set_xlim(50, 2000)
    ax.set_ylim(0.4, 300)

# axes[0].set_xlabel('')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
name = 'CellTypes_H0-Eeff_V2'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 500,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')




# %%%% 2. K
# %%%%% Dataset
# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-07-04']
drugs = ['none', 'Y27']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
XCol = 'bestH0'
YCol = 'fit_K'
fitID='300_100'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])
figname = 'H0vsE400' + drugSuffix

lb = int(fitID.split('_')[0])-int(fitID.split('_')[1])
ub = int(fitID.split('_')[0])+int(fitID.split('_')[1])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df[XCol] <= 1000),
           ]
df_f = filterDf(df, Filters)
df_ff = taka2.getFitsInTable(df_f, fitType='stressGaussian', filter_fitID=fitID)

# Filter again
Filters = [(df_ff['fit_K'] >= 0), 
           (df_ff['fit_K'] <= 1e5), 
           ]
df_ff = filterDf(df_ff, Filters)

# Group By
df_fg = dataGroup(df_ff, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_ff, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'fit_K', weightCol = 'fit_ciwK', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

YCol += '_wAvg'
df_plot[YCol] /= 1000

# %%%%% Plot

# Order
# co_order = ['none', 'dmso']
bp = makeBoxPairs(co_order)
# bp = [bp[0], bp[6], bp[7], bp[8]]
bp = []

# Plot
fig, axes = plt.subplots(1,3, figsize=(17/gs.cm_in, 10/gs.cm_in), sharex=True, sharey=True)

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

for i in range(len(axes)):
    ax = axes[i]
    ax.set_xscale('log')
    ax.set_yscale('log')
    df_fc = df_plot[df_plot[condCol] == co_order[i]]
    color = styleDict[co_order[i]]['color']
    
    # fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
    #                 XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
    #                 modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
    #                 figSizeFactor = 1, markersizeFactor = 0.5)
    
        
    sns.scatterplot(ax = ax, x=df_fc[XCol].values, y=df_fc[YCol].values, 
                    marker = 'o', s = 20, color = color, edgecolor = 'k', linewidth =0.5, alpha = 0.5,
                    zorder = 3)
    Xfit, Yfit = np.log(df_fc[XCol].values), np.log(df_fc[YCol].values)
    
    [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
    A, k = np.exp(b), a
    k_cihw = (results.conf_int(0.05)[1, 1] - results.conf_int(0.05)[1, 0])/2
    R2 = w_results.rsquared
    pval = results.pvalues[1]
    Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = A * Xplot**k
    
    # [b, a], results = ufun.fitLine(Xfit, Yfit)
    # R2 = results.rsquared
    # pval = results.pvalues[1]
    # Xplot = (np.linspace(min(Xfit), max(Xfit), 50))
    # Yplot = a * Xplot + b
    
    ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 2.0, zorder = 6,
            label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + \
                        f'\nA = {A:.1e}' + \
                        f'\nk  = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
                        f'\n$R^2$ = {R2:.2f}' + \
                        f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8)
    ax.set_xlabel('$H_0}$ (nm)')
    ylabel  = '$K$ (kPa)'
    ylabel += r" - $\sigma \in$" + f"[{lb}, {ub}] Pa"
    ax.set_ylabel(ylabel)
    ax.set_title(co_order[i])
    if i != 0:
        ax.set_ylabel('')
           
# Prettify
rD = {'E_f_<_400_wAvg':'Elastic modulus (Pa)\nfor F < 400pN - wAvg',
      'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27 (10µM)', 
      'Y27 & 50.0' : 'Y27 (50µM)', 
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder=0)
    renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD)
    ax.set_xlim(50, 1000)
    ax.set_ylim(0.4, 100)

# axes[0].set_xlabel('')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
name = f'CellTypes_H0-K-{lb}-{ub}'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')




# %%%% 2.2 E_norm


gs.set_manuscript_options_jv('Set2')
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=gs.cL_Set21) 


# Define
df = MecaData_Cells
excluded_subtypes = ['tko']
drugs = ['dmso', 'none']
substrates = ['BSA coated glass', '20um fibronectin discs']

parameter = 'E_f_<_400'
XCol = 'bestH0'
YCol = 'E_f_<_400'


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'].apply(lambda x : x in substrates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x not in excluded_subtypes)),
           (df['valid_f_<_400'] == True),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

df_f.loc[df_f['cell subtype']=='Atcc-2023-LaGFP', 'cell subtype'] = 'Atcc-2023'

df_f, condCol = makeCompositeCol(df_f, cols=['cell type', 'cell subtype'])

th_NLI = np.log10(10)
ref_strain = 0.2
df_f = computeNLMetrics_V2(df_f, th_NLI = th_NLI, ref_strain = ref_strain)
df_f = df_f.dropna(subset=['bestH0', 'E_eff'])

# Order
co_order = ['3T3 & Atcc-2023', 
            '3T3 & aSFL-A11', 
            'DC & mouse-primary', 
            'Dicty & DictyBase-WT', 
            'HoxB8-Macro & ctrl', 
            'MDCK & WT']

rD = {'3T3 & Atcc-2023'      : '3T3 ATCC', 
      '3T3 & aSFL-A11'       : r'3T3 $\alpha$SFL',  
      'DC & mouse-primary'   : 'Primary DC',  
      'Dicty & DictyBase-WT' : 'Dictys Ax3',  
      'HoxB8-Macro & ctrl'   : 'HoxB8 Macro',  
      'MDCK & WT'            : 'MDCK',
      }



# print(df['cell type'].unique())
# print(df['cell subtype'].unique())

# Order
# co_order = ['none', 'dmso']
bp = makeBoxPairs(co_order)
# bp = [bp[0], bp[6], bp[7], bp[8]]
bp = bp[:5] + [bp[9], bp[12], bp[-1]]

# Group By
df_fgw_E400 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_fgw_E400[parameter+'_wAvg'] /= 1000

#### Plot 1
fig, axes = plt.subplots(1,2, figsize=(16/gs.cm_in, 10/gs.cm_in), sharex=True)
ax = axes[0]
ax.set_yscale('log')

fig, ax = D1Plot(df_fgw_E400, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 0.7,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False, statpos = 'in',
                 showMean = False)
           
# Prettify
renameAxes(ax, rD, format_xticks = True, rotation=22)
# renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
ax.set_ylabel('$E_{400}$ (kPa)')

#### Plot 2

ax = axes[1]
ax.set_yscale('log')

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw_Eeff = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_eff', weightCol = 'ciwE_eff', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw_Eeff, on='cellID', how='inner')


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

YCol = 'E_eff_wAvg'
df_plot[YCol] /= 1000

list_expo = []

for i in range(len(co_order)):
    
    df_fc = df_plot[df_plot[condCol] == co_order[i]]
    # color = styleDict[co_order[i]]['color']
    
    # fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
    #                 XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
    #                 modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
    #                 figSizeFactor = 1, markersizeFactor = 0.5)
    # sns.scatterplot(ax = ax, x=df_fc[XCol].values, y=df_fc[YCol].values, 
    #                 marker = 'o', s = 20, color = color, edgecolor = 'k', linewidth =0.5, alpha = 0.5,
    #                 zorder = 3)
    
    Xfit, Yfit = np.log(df_fc[XCol].values), np.log(df_fc[YCol].values)
    
    [b, a], results, w_results = ufun.fitLineHuber(Xfit, Yfit, with_wlm_results = True)
    A, k = np.exp(b), a
    k_cihw = (results.conf_int(0.05)[1, 1] - results.conf_int(0.05)[1, 0])/2
    R2 = w_results.rsquared
    pval = results.pvalues[1]
    Xplot = np.exp(np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = A * Xplot**k
    
    # [b, a], results = ufun.fitLine(Xfit, Yfit)
    # R2 = results.rsquared
    # pval = results.pvalues[1]
    # Xplot = (np.linspace(min(Xfit), max(Xfit), 50))
    # Yplot = a * Xplot + b
    # ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 2.0, zorder = 6,
    #         label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + \
    #                     f'\nA = {A:.1e}' + \
    #                     f'\nk = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
    #                     f'\n$R^2$ = {R2:.2f}' + \
    #                     f'\np-val = {pval:.2f}')

    # ax.legend(fontsize = 8)
    # ax.set_xlabel('$H_0}$ (nm)')
    # ax.set_ylabel('$E_{400}$ (kPa)')
    # ax.set_title(co_order[i])
    # if i != 0:
    #     ax.set_ylabel('')
    
    H_ref = 300
    df_fc['E_norm'] = df_fc['E_eff_wAvg'] * (H_ref/df_fc['bestH0'])**k
    list_expo.append(k)
    
    df_plot.loc[df_plot[condCol] == co_order[i], 'E_norm'] = df_fc['E_norm']
    

# Plot
# fig, ax = plt.subplots(1,1, figsize=(12/gs.cm_in, 10/gs.cm_in))
ax.set_yscale('log')

fig, ax = D1Plot(df_plot, fig = fig, ax = ax, condition=condCol, parameter='E_norm',
                 co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 0.7,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
# ax.set_ylim([0.4, 200])
           
# Prettify
renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True, rotation=22)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
ax.set_ylabel('$E_{norm}$ (kPa)')


#### Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# name = 'CellTypes_E400'
name = 'CellTypes_E400_and_Enorm'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')
