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

figDir = "D:/MagneticPincherData/Figures/DrugsDataset"



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

# styleDict =  {# Drugs
#                'none':{'color': gs.colorList40[10],'marker':'o'},
#                'none & 0.0':{'color': gs.colorList40[10],'marker':'o'},
#                #
#                'dmso':{'color': gs.colorList40[19],'marker':'o'},
#                'dmso & 0.0':{'color': gs.colorList40[19],'marker':'o'},
#                #
#                'blebbistatin':{'color': gs.colorList40[22],'marker':'o'},
#                'blebbistatin & 10.0':{'color': gs.colorList40[12],'marker':'o'},
#                'blebbistatin & 50.0':{'color': gs.colorList40[26],'marker':'o'},
#                'blebbistatin & 100.0':{'color': gs.colorList40[32],'marker':'o'},
#                'blebbistatin & 250.0':{'color': gs.colorList40[36],'marker':'o'},
#                #
#                'PNB & 50.0':{'color': gs.colorList40[25],'marker':'o'},
#                'PNB & 250.0':{'color': gs.colorList40[35],'marker':'o'},
#                #
#                'latrunculinA':{'color': gs.colorList40[23],'marker':'o'},
#                'latrunculinA & 0.1':{'color': gs.colorList40[13],'marker':'o'},
#                'latrunculinA & 0.5':{'color': gs.colorList40[20],'marker':'o'},
#                'latrunculinA & 2.5':{'color': gs.colorList40[33],'marker':'o'},
#                #
#                'calyculinA':{'color': gs.colorList40[23],'marker':'o'},
#                'calyculinA & 0.25':{'color': gs.colorList40[15],'marker':'o'},
#                'calyculinA & 0.5':{'color': gs.colorList40[22],'marker':'o'},
#                'calyculinA & 1.0':{'color': gs.colorList40[30],'marker':'o'},
#                'calyculinA & 2.0':{'color': gs.colorList40[33],'marker':'o'},
#                #
#                'Y27':{'color': gs.colorList40[17],'marker':'o'},
#                'Y27 & 1.0':{'color': gs.colorList40[7],'marker':'o'},
#                'Y27 & 10.0':{'color': gs.colorList40[15],'marker':'o'},
#                'Y27 & 50.0':{'color': gs.colorList40[27],'marker':'o'},
#                'Y27 & 100.0':{'color': gs.colorList40[37],'marker':'o'},
#                #
#                'LIMKi':{'color': gs.colorList40[15],'marker':'o'},
#                'LIMKi & 10.0':{'color': gs.colorList40[15],'marker':'o'},
#                'LIMKi & 20.0':{'color': gs.colorList40[31],'marker':'o'},
#                #
#                'JLY':{'color': gs.colorList40[23],'marker':'o'},
#                'JLY & 8-5-10':{'color': gs.colorList40[23],'marker':'o'},
#                #
#                'ck666':{'color': gs.colorList40[25],'marker':'o'},
#                'ck666 & 50.0':{'color': gs.colorList40[15],'marker':'o'},
#                'ck666 & 100.0':{'color': gs.colorList40[38],'marker':'o'},
               
#                # Cell types
#                '3T3':{'color': gs.colorList40[30],'marker':'o'},
#                'HoxB8-Macro':{'color': gs.colorList40[32],'marker':'o'},
#                'DC':{'color': gs.colorList40[33],'marker':'o'},
#                # Cell subtypes
#                'aSFL':{'color': gs.colorList40[12],'marker':'o'},
#                'Atcc-2023':{'color': gs.colorList40[10],'marker':'o'},
#                'optoRhoA':{'color': gs.colorList40[13],'marker':'o'},
               
#                # Drugs + cell types
#                'aSFL-LG+++ & dmso':{'color': gs.colorList40[19],'marker':'o'},
#                'aSFL-LG+++ & blebbistatin':{'color': gs.colorList40[32],'marker':'o'},
#                'Atcc-2023 & dmso':{'color': gs.colorList40[19],'marker':'o'},
#                'Atcc-2023 & blebbistatin':{'color': gs.colorList40[22],'marker':'o'},
#                #
#                'Atcc-2023 & none':{'color': gs.colorList40[0],'marker':'o'},
#                'Atcc-2023 & none & 0.0':{'color': gs.colorList40[0],'marker':'o'},
#                'Atcc-2023 & Y27': {'color': gs.colorList40[17],'marker':'o'},
#                'Atcc-2023 & Y27 & 10.0': {'color': gs.colorList40[17],'marker':'o'},
#                #
#                'optoRhoA & none':{'color': gs.colorList40[13],'marker':'o'},
#                'optoRhoA & none & 0.0':{'color': gs.colorList40[13],'marker':'o'},
#                'optoRhoA & Y27': {'color': gs.colorList40[27],'marker':'o'},
#                'optoRhoA & Y27 & 10.0': {'color': gs.colorList40[27],'marker':'o'},
#                #
#                'Atcc-2023 & dmso':{'color': gs.colorList40[9],'marker':'o'},
#                'Atcc-2023 & dmso & 0.0':{'color': gs.colorList40[9],'marker':'o'},
#                'Atcc-2023 & blebbistatin':{'color': gs.colorList40[12],'marker':'o'},
#                'Atcc-2023 & blebbistatin & 10.0':{'color': gs.colorList40[2],'marker':'o'},
#                'Atcc-2023 & blebbistatin & 50.0':{'color': gs.colorList40[12],'marker':'o'},
#                #
#                'optoRhoA & dmso':{'color': gs.colorList40[29],'marker':'o'},
#                'optoRhoA & dmso & 0.0':{'color': gs.colorList40[29],'marker':'o'},
#                'optoRhoA & blebbistatin':{'color': gs.colorList40[32],'marker':'o'},
#                'optoRhoA & blebbistatin & 10.0':{'color': gs.colorList40[22],'marker':'o'},
#                'optoRhoA & blebbistatin & 50.0':{'color': gs.colorList40[32],'marker':'o'},
#                }


styleDict =  {# Drugs
               'none':{'color': plt.cm.Greys(0.2),'marker':'o'},
               'none & 0.0':{'color': plt.cm.Greys(0.2),'marker':'o'},
               #
               'dmso':{'color': plt.cm.Greys(0.5),'marker':'o'},
               'dmso & 0.0':{'color': plt.cm.Greys(0.5),'marker':'o'},
               #
               'blebbistatin':{'color': plt.cm.RdPu(0.5),'marker':'o'},
               'blebbistatin & 10.0':{'color':plt.cm.RdPu(0.4), 'marker':'o'},
               'blebbistatin & 50.0':{'color': plt.cm.RdPu(0.65),'marker':'o'},
               'blebbistatin & 100.0':{'color': plt.cm.RdPu(0.9),'marker':'o'},
               'blebbistatin & 250.0':{'color': plt.cm.RdPu(1.0),'marker':'o'},
               #
               'PNB & 50.0':{'color': gs.colorList40[25],'marker':'o'},
               'PNB & 250.0':{'color': gs.colorList40[35],'marker':'o'},
               #
               'latrunculinA':{'color': plt.cm.RdYlBu_r(0.75),'marker':'o'},
               'latrunculinA & 0.1':{'color': plt.cm.RdYlBu_r(0.25),'marker':'o'},
               'latrunculinA & 0.5':{'color': plt.cm.RdYlBu_r(0.75),'marker':'o'},
               'latrunculinA & 2.5':{'color': plt.cm.RdYlBu_r(0.95),'marker':'o'},
               #
               'calyculinA':{'color': gs.colorList40[23],'marker':'o'},
               'calyculinA & 0.25':{'color': gs.colorList40[15],'marker':'o'},
               'calyculinA & 0.5':{'color': gs.colorList40[22],'marker':'o'},
               'calyculinA & 1.0':{'color': gs.colorList40[30],'marker':'o'},
               'calyculinA & 2.0':{'color': gs.colorList40[33],'marker':'o'},
               #
               'Y27':{'color': plt.cm.GnBu(0.3),'marker':'o'},
               'Y27 & 10.0':{'color': plt.cm.GnBu(0.4),'marker':'o'},
               'Y27 & 50.0':{'color': plt.cm.GnBu(0.7),'marker':'o'},
               'Y27 & 100.0':{'color': plt.cm.GnBu(0.9),'marker':'o'},
               #
               'LIMKi':{'color': plt.cm.OrRd(0.5),'marker':'o'},
               'LIMKi & 10.0':{'color': plt.cm.OrRd(0.45),'marker':'o'},
               'LIMKi & 20.0':{'color': plt.cm.OrRd(0.9),'marker':'o'},
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
                axes[i].set_xticklabels(newXticksList)
                
                
def renameLegend(axes, rD, loc='best', ncols=1, fontsize=8):
    axes = ufun.toList(axes)
    N = len(axes)
    for i in range(N):
        ax = axes[i]
        L = ax.legend(loc = loc, ncols=ncols)
        Ltext = L.get_texts()
        M = len(Ltext)
        for j in range(M):
            T = Ltext[j].get_text()
            for s in rD.keys():
                if re.search(s, T):
                    Ltext[j].set_text(re.sub(s, rD[s], T))
                    Ltext[j].set_fontsize(fontsize)
                
def addStat_lib(ax, box_pairs, test = 'Mann-Whitney', verbose = False, **plotting_parameters):
    #### STATS
    listTests = ['t-test_ind', 't-test_welch', 't-test_paired', 
                 'Mann-Whitney', 'Mann-Whitney-gt', 'Mann-Whitney-ls', 
                 'Levene', 'Wilcoxon', 'Kruskal', 'Brunner-Munzel']
    if test in listTests:
        annotator = Annotator(ax, box_pairs, **plotting_parameters)
        annotator.configure(test=test, verbose=verbose, fontsize = 9,
                            line_height = 0.01, line_offset = -1, line_offset_to_group = -1).apply_and_annotate() # , loc = 'outside'
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
            boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 4},
                                boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1.5, 'alpha' : 0.75, 'zorder' : 4},
                                # boxprops={"color": color, "linewidth": 0.5},
                                whiskerprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.75, 'zorder' : 4},
                                capprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.75, 'zorder' : 4})

            
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



# def plotPopKS_V2(data, fig = None, ax = None, 
#                  condition = '', co_order = [],
#                  fitType = 'stressRegion', fitWidth=75,
#                  mode = 'wholeCurve', scale = 'lin', Sinf = 0, Ssup = np.Inf):
    
#     #### Init
#     co_values = data[condition].unique()     
#     if co_order == []:
#         co_order = np.sort(co_values)
#     if ax == None:
#         figHeight = 5
#         figWidth = 8
#         fig, ax = plt.subplots(1,1, figsize=(figWidth, figHeight))
        
#     # colors, markers = getStyleLists(co_order, styleDict)
    
#     if mode == 'wholeCurve':
#         if Ssup >= 1500:
#             xmax = 1550
#         else:
#             xmax = Ssup + 50
#         xmin = Sinf
#         ax.set_xlim([xmin, xmax])  
        
#     else:
#         bounds = mode.split('_')
#         Sinf, Ssup = int(bounds[0]), int(bounds[1])
#         extraFilters = [data['minStress'] <= Sinf, data['maxStress'] >= Ssup] # >= 800
    
#         globalExtraFilter = extraFilters[0]
#         for k in range(1, len(extraFilters)):
#             globalExtraFilter = globalExtraFilter & extraFilters[k]
#         data = data[globalExtraFilter]
            
#         ax.set_xlim([Sinf-50, Ssup+50])
        
    
#     #### Pre-treatment of fitted values
#     fitId = '_' + str(fitWidth)
#     data_fits = taka2.getFitsInTable(data, fitType=fitType, filter_fitID=fitId)
    
#     # Filter the table
#     data_fits = data_fits[(data_fits['fit_center'] >= Sinf) & (data_fits['fit_center'] <= Ssup)]    
#     data_fits = data_fits.drop(data_fits[data_fits['fit_error'] == True].index)
#     data_fits = data_fits.drop(data_fits[data_fits['fit_K'] < 0].index)
#     data_fits = data_fits.dropna(subset = ['fit_ciwK'])
    
    
#     # Compute the weights
#     data_fits['weight'] = (data_fits['fit_K']/data_fits['fit_ciwK'])**2
    
#     # In the following lines, the weighted average and weighted variance are computed
#     # using new columns as intermediates in the computation.
#     #
#     # Col 'A' = K x Weight --- Used to compute the weighted average.
#     # 'K_wAvg' = sum('A')/sum('weight') in each category (group by condCol and 'fit_center')
#     #
#     # Col 'B' = (K - K_wAvg)**2 --- Used to compute the weighted variance.
#     # Col 'C' =  B * Weight     --- Used to compute the weighted variance.
#     # 'K_wVar' = sum('C')/sum('weight') in each category (group by condCol and 'fit_center')
    
#     # Compute the weighted mean
#     data_fits['A'] = data_fits['fit_K'] * data_fits['weight']
#     grouped1 = data_fits.groupby(by=['cellID', 'fit_center'])
#     data_agg_cells = grouped1.agg({'compNum':'count',
#                                    'A':'sum', 
#                                    'weight': 'sum',
#                                    condCol:'first'})
#     data_agg_cells = data_agg_cells.reset_index()
#     data_agg_cells['K_wAvg'] = data_agg_cells['A']/data_agg_cells['weight']
#     data_agg_cells = data_agg_cells.rename(columns = {'compNum' : 'compCount'})
    
#     # 2nd selection
#     data_agg_cells = data_agg_cells.drop(data_agg_cells[data_agg_cells['compCount'] <= 1].index)
#     data_agg_cells = data_agg_cells.drop(data_agg_cells[data_agg_cells['weight'] <= 1].index)
    
#     grouped2 = data_agg_cells.groupby(by=[condCol, 'fit_center'])
#     data_agg_all = grouped2.agg({'compCount':['sum', 'count'],
#                                    'K_wAvg':['mean', 'std', 'median']})
#     data_agg_all = data_agg_all.reset_index()
#     data_agg_all.columns = ufun.flattenPandasIndex(data_agg_all.columns)
#     data_agg_all = data_agg_all.rename(columns = {'compCount_sum' : 'compCount', 'compCount_count' : 'cellCount'})
#     data_agg_all['K_wAvg_ste'] = data_agg_all['K_wAvg_std']/data_agg_all['cellCount']**0.5
    
    
#     #### Plot
#     i_color = 0
    
#     for co in co_order:
#         # try:
#         try:
#             color = styleDict[co]['color']
#         except:
#             color = gs.colorList10[i_color%10]
#             i_color = i_color + 1

            
#         df = data_agg_all[data_agg_all[condCol] == co]
        
#         centers = df['fit_center'].values
#         Kavg = df['K_wAvg_mean'].values
#         Kste = df['K_wAvg_ste'].values
#         N = df['cellCount'].values
#         total_N = np.max(N)
        
#         n = df['compCount'].values
#         total_n = np.max(n)
        
#         dof = N
#         alpha = 0.975
#         q = st.t.ppf(alpha, dof) # Student coefficient
            
#         # weighted means -- weighted ste 95% as error
#         ax.errorbar(centers, Kavg/1000, yerr = q*Kste/1000, 
#                     color = color, lw = 2, marker = 'o', markersize = 6, mec = 'k',
#                     ecolor = color, elinewidth = 1.5, capsize = 6, capthick = 1.5, 
#                     label = str(co) + ' | ' + str(total_N) + ' cells' + ' | ' + str(total_n) + ' comp')
        
#         ax.legend(loc = 'upper left', fontsize = 8)
#         ax.set_xlabel('Stress (Pa)')
#         ax.set_ylabel('K (kPa)')
#         ax.grid(visible=True, which='major', axis='y')
                    
#         # except:
#         #     pass


#     return(fig, ax)


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



# %% > Data import & export

MecaData_DrugV3 = taka2.getMergedTable('MecaData_Drugs_V3')

# %% -------

# %% Multi-drugs

# %%% 3 drugs fluctuations

gs.set_manuscript_options_jv()

# %%%% Dataset 1 --- Y27

# Define
df = MecaData_DrugV3

dates = ['23-03-08', '23-03-09', '24-03-13', '24-07-04']
drugs = ['none', 'Y27']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df[YCol] < 600),
           ]
df_f1 = filterDf(df, Filters)

index_f = df_f1[(df_f1[condCol] == 'Y27 & 10.0') & (df_f1[XCol] >= 400)].index
df_f1 = df_f1.drop(index_f, axis=0)

# Order
df_f1['co'] = df_f1['concentration'].astype(float)
co_order1 = df_f1[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()

# Group By
df_fg1 = dataGroup(df_f1, groupCol = 'cellID', idCols = [condCol], 
                  numCols = [XCol, YCol], aggFun = 'mean')

Filters = [(df_fg1['compNum'] >= 5), 
           ]
df_fg1 = filterDf(df_fg1, Filters)

# %%%% Dataset 2 --- Blebbi

# Define
df = MecaData_DrugV3

dates = ['24-07-04','23-03-16', '23-03-17', '23-02-16', '23-04-20']
drugs = ['dmso', 'blebbistatin']
cons = ['0.0', '10.0', '50.0'] #, '100.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           ]
df_f2 = filterDf(df, Filters)

index_f = df_f2[(df_f2['drug'] == 'blebbistatin') & (df_f2[YCol] >= 190)].index
df_f2 = df_f2.drop(index_f, axis=0)

# Order
df_f2['co'] = df_f2['concentration'].astype(float)
co_order2 = df_f2[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()

# Group By
df_fg2 = dataGroup(df_f2, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Refilter

Filters = [(df_fg2['compNum'] >= 5), 
           ]
df_fg2 = filterDf(df_fg2, Filters)



# %%%% Dataset 3 --- LIMKi3

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

# dates = ['24-07-04']
# dates = ['24-03-13']
dates = ['24-03-13', '24-07-04']
drugs = ['dmso', 'LIMKi']
cons = ['0.0', '10.0', '20.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           ]
df_f3 = filterDf(df, Filters)

index_f = df_f3[(df_f3[YCol] >= 400)].index
df_f3 = df_f3.drop(index_f, axis=0)
index_f = df_f3[(df_f3[XCol] >= 650)].index
df_f3 = df_f3.drop(index_f, axis=0)

# Order
df_f3['co'] = df_f3['concentration'].astype(float)
co_order3 = df_f3[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'


# Group By
df_fg3 = dataGroup(df_f3, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Refilter

Filters = [(df_fg3['compNum'] >= 5), 
           ]
df_fg3 = filterDf(df_fg3, Filters)



# %%%% Plot
fig, axesM = plt.subplots(3, 3, figsize=(17/gs.cm_in, 20/gs.cm_in), sharey=True, sharex=True)

df_list = [df_fg1, df_fg2, df_fg3]
co_list = [co_order1, co_order2, co_order3]

rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27 10 µM', 
      'Y27 & 50.0' : 'Y27 50 µM', 
      'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbistatin 10 µM', 
      'blebbistatin & 50.0' : 'Blebbistatin 50 µM', 
      'blebbistatin & 100.0' : 'Blebbistatin 100 µM',
      'blebbistatin & 250.0' : 'Blebbistatin 250 µM',
      'LIMKi & 10.0' : 'LIMKi3 10 µM', 
      'LIMKi & 20.0' : 'LIMKi3 20 µM',
      }
 
for k, df_fg in enumerate(df_list):
    axes = axesM[k,:]
    co_order = co_list[k]
    
    for i in range(len(axes)):
        ax = axes[i]
        df_fc = df_fg[df_fg[condCol] == co_order[i]]
        color = styleDict[co_order[i]]['color']
        print(co_order[i], len(df_fc))
            
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
                        f'\na = {a:.2f}' + r'$\pm$' + f'{a_cihw:.2f}' + \
                        f'\nb = {b:.2f}' + \
                        f'\n$R^2$ = {R2:.2f}' + \
                        f'\np-val = {pval:.2f}')
    
        ax.legend(fontsize = 8, loc = 'upper left')
        ax.set_xlabel('$H_{5mT}$ (nm)')
        ax.set_ylabel('$H_{5mT}$ fluctuations (nm)') #  [$D_9$-$D_1$]
        ax.set_title(co_order[i])
        if i != 0:
            ax.set_ylabel('')
        if k != 2:
            ax.set_xlabel('')
            
    for ax in axes:
        ax.grid(visible=True, which='major', axis='both', zorder=0)
        renameAxes(ax, rD, format_xticks = False)
        
           
# # Prettify
rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27 (10µM)', 
      'Y27 & 50.0' : 'Y27 (50µM)', 
      # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

# for ax in axes:
#     ax.grid(visible=True, which='major', axis='both', zorder=0)
#     renameAxes(ax, rD, format_xticks = False)
#     # renameAxes(ax, renameDict, format_xticks = False)
#     renameLegend(ax, rD, loc='upper left')
#     ax.set_xlim([0, 600])
#     ax.set_ylim([0, 300])
#     ax.set_xlabel('$H_{5mT}$ (nm)')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond1, CountByCell1 = makeCountDf(df_f1, condCol)
CountByCond2, CountByCell2 = makeCountDf(df_f2, condCol)
CountByCond3, CountByCell3 = makeCountDf(df_f3, condCol)
CountByCond = pd.concat([CountByCond1, CountByCond2, CountByCond3])
# Save
# figSubDir = 'Y27'
# drugSuffix = '_Y27'
# name = 'Fluctuations' + drugSuffix
figSubDir = ''
drugPrefix = 'Y27-BBS-LIMKi_'
name = drugPrefix + 'Fluctuations'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')







# %% Plots Y27

df = MecaData_DrugV3
print(df[df['drug']=='Y27']['date'].unique())
print(df[df['drug']=='Y27']['cell subtype'].unique())

# %%% bestH0

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13', '24-07-04']
drugs = ['none', 'Y27']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(12/gs.cm_in, 10/gs.cm_in))
ax.set_yscale('log')



fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
ax.set_ylim([50, 2750])
           
# Prettify
rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27 10 µM', 
      'Y27 & 50.0' : 'Y27 50 µM', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
# renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
ax.set_ylabel('$H_0$ (nm)')



# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'Y27'
# drugSuffix = '_Y27'
# name = 'bestH0' + drugSuffix
figSubDir = ''
drugPrefix = 'Y27_'
name = drugPrefix + 'bestH0'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')




# %%% E400

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13', '24-07-04']
drugs = ['none', 'Y27']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['E_f_<_400'] <= 5e4),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           ]

df_f = filterDf(df, Filters)

# Order
co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_fgw2[parameter+'_wAvg'] /= 1000

# Plot
fig, ax = plt.subplots(1,1, figsize=(12/gs.cm_in, 10/gs.cm_in))
ax.set_yscale('log')

fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = [], statVerbose = False,
                 showMean = False)
ax.set_ylim([0.4, 200])
           
# Prettify
renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
ax.set_ylabel('$E_{400}$ (kPa)')

# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'Y27'
# drugSuffix = '_Y27'
# name = 'E400' + drugSuffix
figSubDir = ''
drugPrefix = 'Y27_'
name = drugPrefix + 'E400'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%% Fluctuations

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13', '24-07-04']
drugs = ['none', 'Y27']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']


XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df[YCol] < 600),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

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
fig, axes = plt.subplots(3, 1, figsize=(12/gs.cm_in, 20/gs.cm_in), sharey=True)

for i in range(len(axes)):
    ax = axes[i]
    df_fc = df_fg[df_fg[condCol] == co_order[i]]
    color = styleDict[co_order[i]]['color']
    
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
                    f'\na = {a:.2f}' + r'$\pm$' + f'{a_cihw:.2f}' + \
                    f'\nb = {b:.2f}' + \
                    f'\n$R^2$ = {R2:.2f}' + \
                    f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8, loc = 'upper right')
    ax.set_xlabel('$H_{5mT}$ (nm)')
    ax.set_ylabel('$H_{5mT}$ fluctuations (nm) [$D_9$-$D_1$]')
    ax.set_title(co_order[i])
    if i != 0:
        ax.set_ylabel('')
        
           
# Prettify
rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27 (10µM)', 
      'Y27 & 50.0' : 'Y27 (50µM)', 
      # 'ctFieldThickness' : 'Elastic modulus (Pa)\nfor F < 400pN',
      # 'ctFieldFluctuAmpli' : 'Elastic modulus (Pa)\nfor F < 400pN',
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder=0)
    renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
    renameLegend(ax, rD, loc='upper left')
    ax.set_xlim([0, 600])
    ax.set_ylim([0, 300])
    ax.set_xlabel('$H_{5mT}$ (nm)')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'Y27'
# drugSuffix = '_Y27'
# name = 'Fluctuations' + drugSuffix
figSubDir = ''
drugPrefix = 'Y27_'
name = drugPrefix + 'Fluctuations'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%% Non-linearity


gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13', '24-07-04']
drugs = ['none', 'Y27']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           ]
df_f = filterDf(df, Filters)

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
Ncells = CountByCond.cellCount.values[0]
Ncomps = CountByCond.compCount.values[0]

# Order
co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12/gs.cm_in, 18/gs.cm_in), sharex=True, sharey=True)

intervals = ['200_500', '300_700', '400_900']
cL = plt.cm.plasma(np.linspace(0.1, 0.9, len(intervals)))

rD = {'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27 10 µM', 
      'Y27 & 50.0' : 'Y27 50 µM', 
      }

for k, cond in enumerate(co_order):  
    for i, interval in enumerate(intervals):
        ax = axes[i]
        ax.set_yscale('log')
        df_fc = df_f[df_f[condCol] == cond]        
        
        # colorDict = {cond:cL[i]}
        # labelDict = {cond:f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"}
        colorDict = {cond:styleDict[cond]['color']}
        labelDict = {cond:rD[cond]}
        plotPopKS_V2(df_fc, fig = fig, ax = ax, condition = condCol, co_order = [], 
                     colorDict = colorDict, labelDict = labelDict, markersizefactor = 0.75,
                     fitType = 'stressGaussian', fitWidth=75, mode = interval, Sinf = 0, Ssup = np.Inf)
    
        # Prettify
        ax.grid(visible=True, which='both', axis='y')
        # legendTitle = r'$\bf{Whole\  dataset}$ | ' + f'{Ncells} cells | {Ncomps} comp'
        legendTitle = r"$\sigma \in$" + f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"
        ax.legend(loc='lower right', fontsize = 7, title = legendTitle, title_fontsize = 8)
        # ax.legend().set_visible(False)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0.9, 20)
        if i != len(intervals)-1:
            ax.set_xlabel('')


# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'Y27'
# drugSuffix = '_Y27'
# name = 'K-sigma-3ranges' + drugSuffix
figSubDir = ''
drugPrefix = 'Y27_'
name = drugPrefix + 'K-sigma-3ranges'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%% Thickness & Stiffness

# %%%% 1. E400

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13', '24-07-04']
drugs = ['none', 'Y27']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
XCol = 'bestH0'
YCol = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(3, 1, figsize=(9.5/gs.cm_in, 22.5/gs.cm_in), sharex=True, sharey=True)

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
                        f'\nk = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
                        f'\n$R^2$ = {R2:.2f}' + \
                        f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8)
    ax.set_xlabel('$H_0}$ (nm)')
    ax.set_ylabel('$E_{400}$ (kPa)')
    ax.set_title(co_order[i])
    # if i != 0:
    #     ax.set_ylabel('')
           
# Prettify
rD = {'E_f_<_400_wAvg':'Elastic modulus (Pa)\nfor F < 400pN - wAvg',
      'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27 10 µM', 
      'Y27 & 50.0' : 'Y27 50 µM', 
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
# figSubDir = 'Y27'
# drugSuffix = '_Y27'
# name = 'H0-E400' + drugSuffix
figSubDir = ''
drugPrefix = 'Y27_'
name = drugPrefix + 'H0-E400'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%%% 1.5 E400 - All in one

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13', '24-07-04']
drugs = ['none', 'Y27']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
XCol = 'bestH0'
YCol = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

# Order
co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12/gs.cm_in, 9.5/gs.cm_in), sharex=True, sharey=True)
ax.set_xscale('log')
ax.set_yscale('log')

YCol += '_wAvg'
df_plot[YCol] /= 1000

for i in range(len(co_order)):
    ax = ax
    
    df_fc = df_plot[df_plot[condCol] == co_order[i]]
    color = styleDict[co_order[i]]['color']
    
    # fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
    #                 XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
    #                 modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
    #                 figSizeFactor = 1, markersizeFactor = 0.5)
    
        
    sns.scatterplot(ax = ax, x=df_fc[XCol].values, y=df_fc[YCol].values, 
                    marker = 'o', s = 20, color = color, edgecolor = 'k', linewidth = 0.5, alpha = 0.8,
                    zorder = 3, label=co_order[i])
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
            label =  f'k = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}')
            # label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + \
            #             f'\nA = {A:.1e}' + \
            #             f'\nk = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
            #             f'\n$R^2$ = {R2:.2f}' + \
            #             f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8) #.set_visible(False)
    ax.set_xlabel('$H_0}$ (nm)')
    ax.set_ylabel('$E_{400}$ (kPa)')
    # if i != 0:
    #     ax.set_ylabel('')
           
# Prettify
rD = {'E_f_<_400_wAvg':'Elastic modulus (Pa)\nfor F < 400pN - wAvg',
      'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27 10 µM', 
      'Y27 & 50.0' : 'Y27 50 µM', 
      }

ax.grid(visible=True, which='major', axis='both', zorder=0)
renameAxes(ax, rD, format_xticks = False)
renameLegend(ax, rD)
ax.set_xlim(50, 1000)
ax.set_ylim(0.4, 100)

# axes[0].set_xlabel('')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'Y27'
# drugSuffix = '_Y27'
# name = 'H0-E400' + drugSuffix
figSubDir = ''
drugPrefix = 'Y27_'
name = drugPrefix + 'H0-E400_AllIn1'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%%% 2. K
# %%%%% Dataset
# Define
df = MecaData_DrugV3
dates = ['23-03-08', '23-03-09', '24-03-13', '24-07-04']
drugs = ['none', 'Y27']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']
XCol = 'bestH0'
YCol = 'fit_K'
fitID='300_100'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])

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
co_order = ['none & 0.0', 'Y27 & 10.0', 'Y27 & 50.0']

# Plot
fig, axes = plt.subplots(3, 1, figsize=(9.5/gs.cm_in, 22.5/gs.cm_in), sharex=True, sharey=True)

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
                        f'\nk = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
                        f'\n$R^2$ = {R2:.2f}' + \
                        f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8)
    ax.set_xlabel('$H_0}$ (nm)')
    ylabel  = '$K$ (kPa)'
    ylabel += r" - $\sigma \in$" + f"[{lb}, {ub}] Pa"
    ax.set_ylabel(ylabel)
    ax.set_title(co_order[i])
    # if i != 0:
    #     ax.set_ylabel('')
           
# Prettify
rD = {'E_f_<_400_wAvg':'Elastic modulus (Pa)\nfor F < 400pN - wAvg',
      'none & 0.0' : 'No drug',
      'Y27 & 10.0' : 'Y27 10 µM', 
      'Y27 & 50.0' : 'Y27 50 µM', 
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
# figSubDir = 'Y27'
# drugSuffix = '_Y27'
# name = f'H0-K-{lb}-{ub}' + drugSuffix
figSubDir = ''
drugPrefix = 'Y27_'
name = drugPrefix + f'H0-K-{lb}-{ub}'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')





# %% Plots Blebbistatin

df = MecaData_DrugV3
print(df[df['drug']=='blebbistatin']['date'].unique())
print(df[df['drug']=='blebbistatin']['cell subtype'].unique())

# %%% bestH0

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3
# 
# , '22-03-28', '22-03-30', '23-02-16', '23-04-20'
dates = ['24-07-04','23-03-16', '23-03-17', '23-02-16', '23-04-20']
drugs = ['dmso', 'blebbistatin']
cons = ['0.0', '10.0', '50.0', '100.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           (df[parameter] >= 60),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'
bp = makeBoxPairs(co_order)
bp = [bp[0], bp[3], bp[2], bp[-1]]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(16/gs.cm_in, 10/gs.cm_in))
# fig, ax = plt.subplots(1,1, figsize=(12, 8))
ax.set_yscale('log')

fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
ax.set_ylim([50, 2750])
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbi 10 µM', 
      'blebbistatin & 50.0' : 'Blebbi 50 µM', 
      'blebbistatin & 100.0' : 'Blebbi 100 µM',
      'blebbistatin & 250.0' : 'Blebbi 250 µM', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
# renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
ax.set_ylabel('$H_0$ (nm)')


# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'Blebbi'
# drugSuffix = '_Blebbi'
# name = 'bestH0' + drugSuffix
figSubDir = ''
drugPrefix = 'Blebbi_'
name = drugPrefix + 'bestH0'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')




# %%% E400

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3
# 
# , '22-03-28', '22-03-30', '23-02-16', '23-04-20'
dates = ['24-07-04','23-03-16', '23-03-17', '23-02-16', '23-04-20']
drugs = ['dmso', 'blebbistatin']
cons = ['0.0', '10.0', '50.0', '100.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           (df[parameter] <= 1e5),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'
bp = makeBoxPairs(co_order)
bp = [bp[-1], bp[3], bp[0], bp[2], ]

# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_fgw2[parameter+'_wAvg'] /= 1000

# Plot
fig, ax = plt.subplots(1,1, figsize=(16/gs.cm_in, 10/gs.cm_in))
ax.set_yscale('log')

fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter+'_wAvg',
                 co_order = co_order, boxplot = 3, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
ax.set_ylim([0.4, 200])
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbi 10 µM', 
      'blebbistatin & 50.0' : 'Blebbi 50 µM', 
      'blebbistatin & 100.0' : 'Blebbi 100 µM',
      'blebbistatin & 250.0' : 'Blebbi 250 µM', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
ax.set_ylabel('$E_{400}$ (kPa)')

# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'Blebbi'
# drugSuffix = '_Blebbi'
# name = 'E400' + drugSuffix
figSubDir = ''
drugPrefix = 'Blebbi_'
name = drugPrefix + 'E400'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%% Fluctuations

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3
# 
# , 
dates = ['24-07-04','23-03-16', '23-03-17', '23-02-16', '23-04-20']
drugs = ['dmso', 'blebbistatin']
cons = ['0.0', '10.0', '50.0', '100.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           ]
df_f = filterDf(df, Filters)

index_f = df_f[(df_f['drug'] == 'blebbistatin') & (df_f[YCol] >= 190)].index
# print(df_f.loc[index_f, 'cellID'])
df_f = df_f.drop(index_f, axis=0)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'


# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Refilter

Filters = [(df_fg['compNum'] >= 5), 
           ]
df_fg2 = filterDf(df_fg, Filters)


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
fig, axes = plt.subplots(2, 2, figsize=(17/gs.cm_in, 15/gs.cm_in), sharey=True, sharex=True)
axes = axes.flatten('C')

for i in range(len(axes)):
    ax = axes[i]
    df_fc = df_fg2[df_fg2[condCol] == co_order[i]]
    color = styleDict[co_order[i]]['color']
    
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
                    f'\na = {a:.2f}' + r'$\pm$' + f'{a_cihw:.2f}' + \
                    f'\nb = {b:.2f}' + \
                    f'\n$R^2$ = {R2:.2f}' + \
                    f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8)
    ax.set_xlabel('$H_{5mT}$ (nm)')
    ax.set_ylabel('$H_{5mT}$ fluctuations (nm) [$D_9$-$D_1$]')
    ax.set_title(co_order[i])
    if i%2 != 0:
        ax.set_ylabel('')
        
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbistatin - 10 µM', 
      'blebbistatin & 50.0' : 'Blebbistatin - 50 µM', 
      'blebbistatin & 100.0' : 'Blebbistatin - 100 µM',
      'blebbistatin & 250.0' : 'Blebbistatin - 250 µM', 
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder = 0)
    renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD, loc='upper left')
    ax.set_xlim([0, 600])
    ax.set_ylim([0, 300])
    ax.set_xlabel('$H_{5mT}$ (nm)')


# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'Blebbi'
# drugSuffix = '_Blebbi'
# name = 'Fluctuations' + drugSuffix
figSubDir = ''
drugPrefix = 'Blebbi_'
name = drugPrefix + 'Fluctuations'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%% Non-linearity

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3
# 
# 
dates = ['24-07-04','23-03-16', '23-03-17', '23-02-16', '23-04-20']
drugs = ['dmso', 'blebbistatin']
cons = ['0.0', '10.0', '50.0', '100.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
Ncells = CountByCond.cellCount.values[0]
Ncomps = CountByCond.compCount.values[0]

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12/gs.cm_in, 18/gs.cm_in), sharex=True, sharey=True)

intervals = ['200_500', '300_700', '400_900']
cL = plt.cm.plasma(np.linspace(0.1, 0.9, len(intervals)))

rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbi 10 µM', 
      'blebbistatin & 50.0' : 'Blebbi 50 µM', 
      'blebbistatin & 100.0' : 'Blebbi 100 µM',
      'blebbistatin & 250.0' : 'Blebbi 250 µM', 
      }

for k, cond in enumerate(co_order):  
    for i, interval in enumerate(intervals):
        ax = axes[i]
        ax.set_yscale('log')
        df_fc = df_f[df_f[condCol] == cond]        
        
        # colorDict = {cond:cL[i]}
        # labelDict = {cond:f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"}
        colorDict = {cond:styleDict[cond]['color']}
        labelDict = {cond:rD[cond]}
        plotPopKS_V2(df_fc, fig = fig, ax = ax, condition = condCol, co_order = [], 
                     colorDict = colorDict, labelDict = labelDict, markersizefactor = 0.75,
                     fitType = 'stressGaussian', fitWidth=75, mode = interval, Sinf = 0, Ssup = np.Inf)
    
        # Prettify
        ax.grid(visible=True, which='both', axis='y')
        # legendTitle = r'$\bf{Whole\  dataset}$ | ' + f'{Ncells} cells | {Ncomps} comp'
        legendTitle = r"$\sigma \in$" + f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"
        ax.legend(loc='lower right', fontsize = 7, title = legendTitle, title_fontsize = 8)
        # ax.legend().set_visible(False)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0.9, 30)
        if i != len(intervals)-1:
            ax.set_xlabel('')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'Blebbi'
# drugSuffix = '_Blebbi'
# name = 'K-sigma-3ranges' + drugSuffix
figSubDir = ''
drugPrefix = 'Blebbi_'
name = drugPrefix + 'K-sigma-3ranges'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%% Thickness & Stiffness

# %%%% 1. E400

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3
# 
# 
dates = ['24-07-04','23-03-16', '23-03-17', '23-02-16', '23-04-20']
drugs = ['dmso', 'blebbistatin']
cons = ['0.0', '10.0', '50.0', '100.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'bestH0'
YCol = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(2, 2, figsize=(17/gs.cm_in, 15/gs.cm_in), sharex=True, sharey=True)
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
                        f'\nk = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
                        f'\n$R^2$ = {R2:.2f}' + \
                        f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8)
    ax.set_xlabel('$H_0}$ (nm)')
    ax.set_ylabel('$E_{400}$ (kPa)')
    ax.set_title(co_order[i])
    if i%2 != 0:
        ax.set_ylabel('')
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbistatin 10 µM', 
      'blebbistatin & 50.0' : 'Blebbistatin 50 µM', 
      'blebbistatin & 100.0' : 'Blebbistatin 100 µM',
      'blebbistatin & 250.0' : 'Blebbistatin 250 µM', 
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
# figSubDir = 'Blebbi'
# drugSuffix = '_Blebbi'
# name = 'H0-E400' + drugSuffix
figSubDir = ''
drugPrefix = 'Blebbi_'
name = drugPrefix + 'H0-E400'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%%% 1.5 E400 - All in one

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3
# 
# 
dates = ['24-07-04','23-03-16', '23-03-17', '23-02-16', '23-04-20']
drugs = ['dmso', 'blebbistatin']
cons = ['0.0', '10.0', '50.0', '100.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'bestH0'
YCol = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12/gs.cm_in, 9.5/gs.cm_in), sharex=True, sharey=True)
ax.set_xscale('log')
ax.set_yscale('log')

YCol += '_wAvg'
df_plot[YCol] /= 1000

for i in range(len(co_order)):
    ax = ax
    
    df_fc = df_plot[df_plot[condCol] == co_order[i]]
    color = styleDict[co_order[i]]['color']
    
    # fig, ax = D2Plot_wFit(df_fg[df_fg[condCol] == co_order[i]], fig = fig, ax = ax, 
    #                 XCol=XCol, YCol=YCol, condition=condCol, co_order = [],
    #                 modelFit=True, modelType='y=ax+b', writeEqn = True, robust = True,
    #                 figSizeFactor = 1, markersizeFactor = 0.5)
    
        
    sns.scatterplot(ax = ax, x=df_fc[XCol].values, y=df_fc[YCol].values, 
                    marker = 'o', s = 20, color = color, edgecolor = 'k', linewidth = 0.5, alpha = 0.6,
                    zorder = 3, label=co_order[i])
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
            label =  f'k = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}')
            # label =  r'$\bf{Fit\ y\ =\ A.x^k}$' + \
            #             f'\nA = {A:.1e}' + \
            #             f'\nk = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
            #             f'\n$R^2$ = {R2:.2f}' + \
            #             f'\np-val = {pval:.2f}')

     #.set_visible(False)
    ax.set_xlabel('$H_0}$ (nm)')
    ax.set_ylabel('$E_{400}$ (kPa)')
    # if i != 0:
    #     ax.set_ylabel('')
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbistatin 10 µM', 
      'blebbistatin & 50.0' : 'Blebbistatin 50 µM', 
      'blebbistatin & 100.0' : 'Blebbistatin 100 µM',
      'blebbistatin & 250.0' : 'Blebbistatin 250 µM', 
      }

ax.grid(visible=True, which='major', axis='both', zorder=0)
renameAxes(ax, rD, format_xticks = False)
renameLegend(ax, rD, ncols=2, loc='lower left')
# ax.legend(fontsize = 8)
ax.set_xlim(50, 1000)
ax.set_ylim(0.4, 100)

# axes[0].set_xlabel('')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'Y27'
# drugSuffix = '_Y27'
# name = 'H0-E400' + drugSuffix
figSubDir = ''
drugPrefix = 'Blebbi_'
name = drugPrefix + 'H0-E400_AllIn1'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%%% 2. K
# %%%%% Dataset

# Define
df = MecaData_DrugV3

# 
dates = ['24-07-04','23-03-16', '23-03-17', '23-02-16', '23-04-20']
drugs = ['dmso', 'blebbistatin']
cons = ['0.0', '10.0', '50.0', '100.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'bestH0'
YCol = 'fit_K'
fitID='300_100'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])

lb = int(fitID.split('_')[0])-int(fitID.split('_')[1])
ub = int(fitID.split('_')[0])+int(fitID.split('_')[1])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           (df[XCol] >= 50),
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

gs.set_manuscript_options_jv()

# Order
df_ff['co'] = df_ff['concentration'].astype(float)
co_order = df_ff[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()

# Plot
fig, axes = plt.subplots(2, 2, figsize=(17/gs.cm_in, 15/gs.cm_in), sharex=True, sharey=True)
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
                        f'\nk = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
                        f'\n$R^2$ = {R2:.2f}' + \
                        f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8)
    ax.set_xlabel('$H_0}$ (nm)')
    ylabel  = '$K$ (kPa)'
    ylabel += r" - $\sigma \in$" + f"[{lb}, {ub}] Pa"
    ax.set_ylabel(ylabel)
    ax.set_title(co_order[i])
    if i%2 != 0:
        ax.set_ylabel('')
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'blebbistatin & 10.0' : 'Blebbistatin 10 µM', 
      'blebbistatin & 50.0' : 'Blebbistatin 50 µM', 
      'blebbistatin & 100.0' : 'Blebbistatin 100 µM',
      'blebbistatin & 250.0' : 'Blebbistatin 250 µM', 
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
# figSubDir = 'Blebbi'
# drugSuffix = '_Blebbi'
# name = f'H0-K-{lb}-{ub}' + drugSuffix
figSubDir = ''
drugPrefix = 'Blebbi_'
name = drugPrefix + f'H0-K-{lb}-{ub}'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')





# %% Plots LIMKi3

df = MecaData_DrugV3
# print(df.drug.unique())
print(df[df['drug']=='LIMKi']['date'].unique())
print(df[df['drug']=='LIMKi']['cell subtype'].unique())

# %%% bestH0

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

# dates = ['24-07-04']
# dates = ['24-03-13']
dates = ['24-03-13', '24-07-04']
drugs = ['dmso', 'LIMKi']
cons = ['0.0', '10.0', '20.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           (df[parameter] <= 1000),
            (df[parameter] >= 110),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'
bp = makeBoxPairs(co_order)
bp = []

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(12/gs.cm_in, 10/gs.cm_in))
# fig, ax = plt.subplots(1,1, figsize=(12, 8))
ax.set_yscale('log')


fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
ax.set_ylim([50, 2750])

# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'LIMKi & 10.0' : 'LIMKi3 10 µM', 
      'LIMKi & 20.0' : 'LIMKi3 20 µM', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
# renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
ax.set_ylabel('$H_0$ (nm)')


# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'LIMKi3'
# drugSuffix = '_LIMKi3'
# name = 'bestH0' + drugSuffix
figSubDir = ''
drugPrefix = 'LIMKi3_'
name = drugPrefix + 'bestH0'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')




# %%% E400

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

# dates = ['24-07-04']
dates = ['24-03-13', '24-07-04']
# dates = ['24-03-13', '24-07-04']
drugs = ['dmso', 'LIMKi']
cons = ['0.0', '10.0', '20.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           (df[parameter] <= 2e4),
            (df['valid_f_<_400'] == True),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'
bp = makeBoxPairs(co_order)
bp = []

# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_fgw2[parameter+'_wAvg'] /= 1000

# Plot
fig, ax = plt.subplots(1,1, figsize=(12/gs.cm_in, 10/gs.cm_in))
ax.set_yscale('log')

fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter +'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
ax.set_ylim([0.4, 200])
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'LIMKi & 10.0' : 'LIMKi3 10 µM', 
      'LIMKi & 20.0' : 'LIMKi3 20 µM', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
ax.set_ylabel('$E_{400}$ (kPa)')

# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'LIMKi3'
# drugSuffix = '_LIMKi3'
# name = 'E400' + drugSuffix
figSubDir = ''
drugPrefix = 'LIMKi3_'
name = drugPrefix + 'E400'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%% Fluctuations

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

# dates = ['24-07-04']
# dates = ['24-03-13']
dates = ['24-03-13', '24-07-04']
drugs = ['dmso', 'LIMKi']
cons = ['0.0', '10.0', '20.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           ]
df_f = filterDf(df, Filters)

index_f = df_f[(df_f[YCol] >= 400)].index
# print(df_f.loc[index_f, 'cellID'])
df_f = df_f.drop(index_f, axis=0)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'


# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Refilter

Filters = [(df_fg['compNum'] >= 5), 
           ]
df_fg2 = filterDf(df_fg, Filters)


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
fig, axes = plt.subplots(1, 3, figsize=(17/gs.cm_in, 10/gs.cm_in), sharey=True, sharex=True)
axes = axes.flatten('C')

for i in range(len(axes)):
    ax = axes[i]
    df_fc = df_fg2[df_fg2[condCol] == co_order[i]]
    color = styleDict[co_order[i]]['color']
    
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
                    f'\na = {a:.2f}' + r'$\pm$' + f'{a_cihw:.2f}' + \
                    f'\nb = {b:.2f}' + \
                    f'\n$R^2$ = {R2:.2f}' + \
                    f'\np-val = {pval:.2f}')
        
    ax.legend(fontsize = 8)
    ax.set_xlabel('$H_{5mT}$ (nm)')
    ax.set_ylabel('$H_{5mT}$ fluctuations (nm) [$D_9$-$D_1$]')
    ax.set_title(co_order[i])
    if i%2 != 0:
        ax.set_ylabel('')
        
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'LIMKi & 10.0' : 'LIMKi3 10 µM', 
      'LIMKi & 20.0' : 'LIMKi3 20 µM', 
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder = 0)
    renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD, loc='upper left')
    ax.set_xlim([0, 750])
    ax.set_ylim([0, 450])
    ax.set_xlabel('$H_{5mT}$ (nm)')


# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'LIMKi3'
# drugSuffix = '_LIMKi3'
# name = 'Fluctuations' + drugSuffix
figSubDir = ''
drugPrefix = 'LIMKi3_'
name = drugPrefix + 'Fluctuations'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%% Non-linearity

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

# dates = ['24-07-04']
# dates = ['24-03-13']
dates = ['24-03-13', '24-07-04']
drugs = ['dmso', 'LIMKi']
cons = ['0.0', '10.0', '20.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
Ncells = CountByCond.cellCount.values[0]
Ncomps = CountByCond.compCount.values[0]

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12/gs.cm_in, 18/gs.cm_in), sharex=True, sharey=True)

intervals = ['200_500', '300_700', '400_900']
cL = plt.cm.plasma(np.linspace(0.1, 0.9, len(intervals)))

rD = {'dmso & 0.0' : 'DMSO',
      'LIMKi & 10.0' : 'LIMKi3 10 µM', 
      'LIMKi & 20.0' : 'LIMKi3 20 µM', 
      }

for k, cond in enumerate(co_order):  
    for i, interval in enumerate(intervals):
        ax = axes[i]
        ax.set_yscale('log')
        df_fc = df_f[df_f[condCol] == cond]        
        
        # colorDict = {cond:cL[i]}
        # labelDict = {cond:f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"}
        colorDict = {cond:styleDict[cond]['color']}
        labelDict = {cond:rD[cond]}
        plotPopKS_V2(df_fc, fig = fig, ax = ax, condition = condCol, co_order = [], 
                     colorDict = colorDict, labelDict = labelDict, markersizefactor = 0.75,
                     fitType = 'stressGaussian', fitWidth=75, mode = interval, Sinf = 0, Ssup = np.Inf)
    
        # Prettify
        ax.grid(visible=True, which='both', axis='y')
        # legendTitle = r'$\bf{Whole\  dataset}$ | ' + f'{Ncells} cells | {Ncomps} comp'
        legendTitle = r"$\sigma \in$" + f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"
        ax.legend(loc='lower right', fontsize = 7, title = legendTitle, title_fontsize = 8)
        # ax.legend().set_visible(False)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0.9, 30)
        if i != len(intervals)-1:
            ax.set_xlabel('')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'LIMKi3'
# drugSuffix = '_LIMKi3'
# name = 'K-sigma-3ranges' + drugSuffix
figSubDir = ''
drugPrefix = 'LIMKi3_'
name = drugPrefix + 'K-sigma-3ranges'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%% Thickness & Stiffness

# %%%% 1. E400

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

# dates = ['24-07-04']
# dates = ['24-03-13']
dates = ['24-03-13', '24-07-04']
drugs = ['dmso', 'LIMKi']
cons = ['0.0', '10.0', '20.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'bestH0'
YCol = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(3, 1, figsize=(9.5/gs.cm_in, 22.5/gs.cm_in), sharex=True, sharey=True)
# axes = axes.flatten('C')

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
                        f'\nk = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
                        f'\n$R^2$ = {R2:.2f}' + \
                        f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8)
    ax.set_xlabel('$H_0}$ (nm)')
    ax.set_ylabel('$E_{400}$ (kPa)')
    ax.set_title(co_order[i])
    # if i%2 != 0:
    #     ax.set_ylabel('')
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'LIMKi & 10.0' : 'LIMKi3 10 µM', 
      'LIMKi & 20.0' : 'LIMKi3 20 µM', 
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
# figSubDir = 'LIMKi3'
# drugSuffix = '_LIMKi3'
# name = 'H0-E400' + drugSuffix
figSubDir = ''
drugPrefix = 'LIMKi3_'
name = drugPrefix + 'H0-E400'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%%% 2. K
# %%%%% Dataset

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

# dates = ['24-07-04']
# dates = ['24-03-13']
dates = ['24-03-13', '24-07-04']
drugs = ['dmso', 'LIMKi']
cons = ['0.0', '10.0', '20.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'bestH0'
YCol = 'fit_K'
fitID='300_100'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])

lb = int(fitID.split('_')[0])-int(fitID.split('_')[1])
ub = int(fitID.split('_')[0])+int(fitID.split('_')[1])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           (df[XCol] >= 50),
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

gs.set_manuscript_options_jv()

# Order
df_ff['co'] = df_ff['concentration'].astype(float)
co_order = df_ff[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()

# Plot
fig, axes = plt.subplots(3, 1, figsize=(9.5/gs.cm_in, 22.5/gs.cm_in), sharex=True, sharey=True)
# axes = axes.flatten('C')

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
                        f'\nk = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
                        f'\n$R^2$ = {R2:.2f}' + \
                        f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8)
    ax.set_xlabel('$H_0}$ (nm)')
    ylabel  = '$K$ (kPa)'
    ylabel += r" - $\sigma \in$" + f"[{lb}, {ub}] Pa"
    ax.set_ylabel(ylabel)
    ax.set_title(co_order[i])
    # if i%2 != 0:
    #     ax.set_ylabel('')
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'LIMKi & 10.0' : 'LIMKi3 10 µM', 
      'LIMKi & 20.0' : 'LIMKi3 20 µM', 
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder=0)
    renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD)
    ax.set_xlim(100, 1000)
    # ax.set_ylim(0.4, 100)

# axes[0].set_xlabel('')



# Show
plt.tight_layout()
plt.show()



# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'LIMKi3'
# drugSuffix = '_LIMKi3'
# name = f'H0-K-{lb}-{ub}' + drugSuffix
figSubDir = ''
drugPrefix = 'LIMKi3_'
name = drugPrefix + f'H0-K-{lb}-{ub}'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')




# %% Plots latrunculinA

df = MecaData_DrugV3
# print(df.drug.unique())
print(df[df['drug']=='latrunculinA']['date'].unique())
print(df[df['drug']=='latrunculinA']['cell subtype'].unique())
print(df[df['drug']=='latrunculinA']['concentration'].unique())

# %%% bestH0

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

dates = ['23-11-26', '23-12-03']
drugs = ['dmso', 'latrunculinA']
cons = ['0.0', '0.1', '0.5', '2.5']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
            (df['bestH0'] <= 950),
            (df['bestH0'] >= 50),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'
bp = makeBoxPairs(co_order)
bp = [bp[-1]] + bp[0:4] 

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(16/gs.cm_in, 10/gs.cm_in))
# fig, ax = plt.subplots(1,1, figsize=(12, 8))
ax.set_yscale('log')

fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
ax.set_ylim([50, 2750])
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'latrunculinA & 0.1' : 'LatA 0.1 µM', 
      'latrunculinA & 0.5' : 'LatA 0.5 µM', 
      'latrunculinA & 2.5' : 'LatA 2.5 µM', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
# renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
ax.set_ylabel('$H_0$ (nm)')
# ax.set_ylim([100, 2500])


# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'LatA'
# drugSuffix = '_LatA'
# name = 'bestH0' + drugSuffix
figSubDir = ''
drugPrefix = 'LatA_'
name = drugPrefix + 'bestH0'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')




# %%% E400

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

dates = ['23-11-26', '23-12-03']
drugs = ['dmso', 'latrunculinA']
cons = ['0.0', '0.1', '0.5', '2.5']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           (df[parameter] <= 2e4),
            (df['valid_f_<_400'] == True),
           (df['bestH0'] <= 950),
           (df['bestH0'] >= 50),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'
bp = makeBoxPairs(co_order)
bp = [bp[-1]] + bp[0:4]

# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_fgw2[parameter+'_wAvg'] /= 1000

# Plot
fig, ax = plt.subplots(1,1, figsize=(16/gs.cm_in, 10/gs.cm_in))
ax.set_yscale('log')

fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter +'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
ax.set_ylim([0.4, 200])
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'latrunculinA & 0.1' : 'LatA 0.1 µM', 
      'latrunculinA & 0.5' : 'LatA 0.5 µM', 
      'latrunculinA & 2.5' : 'LatA 2.5 µM', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
ax.set_ylabel('$E_{400}$ (kPa)')

# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'LatA'
# drugSuffix = '_LatA'
# name = 'E400' + drugSuffix
figSubDir = ''
drugPrefix = 'LatA_'
name = drugPrefix + 'E400'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%% Fluctuations

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

dates = ['23-11-26', '23-12-03']
drugs = ['dmso', 'latrunculinA']
cons = ['0.0', '0.1', '0.5', '2.5']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           ]
df_f = filterDf(df, Filters)

index_f = df_f[(df_f[YCol] >= 400)].index
# print(df_f.loc[index_f, 'cellID'])
df_f = df_f.drop(index_f, axis=0)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'


# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Refilter

Filters = [(df_fg['compNum'] >= 5), 
           ]
df_fg2 = filterDf(df_fg, Filters)


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
fig, axes = plt.subplots(2, 2, figsize=(17/gs.cm_in, 16/gs.cm_in), sharey=True, sharex=True)
axes = axes.flatten('C')

for i in range(len(axes)):
    ax = axes[i]
    df_fc = df_fg2[df_fg2[condCol] == co_order[i]]
    color = styleDict[co_order[i]]['color']
    
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
    w_results.conf_int(0.05) 
    Xplot = (np.linspace(min(Xfit), max(Xfit), 50))
    Yplot = a * Xplot + b
    
    # [b, a], results = ufun.fitLine(Xfit, Yfit)
    # R2 = results.rsquared
    # pval = results.pvalues[1]
    # Xplot = (np.linspace(min(Xfit), max(Xfit), 50))
    # Yplot = a * Xplot + b
    
    ax.plot(Xplot, Yplot, ls = '--', c = color, lw = 2.0, zorder = 6,
            label =  r'$\bf{Fit\ y=ax+b}$' + \
                    f'\na = {a:.2f}' + r'$\pm$' + f'{a_cihw:.2f}' + \
                    f'\nb = {b:.2f}' + \
                    f'\n$R^2$ = {R2:.2f}' + \
                    f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8)
    ax.set_xlabel('$H_{5mT}$ (nm)')
    ax.set_ylabel('$H_{5mT}$ fluctuations (nm) [$D_9$-$D_1$]')
    ax.set_title(co_order[i])
    if i%2 != 0:
        ax.set_ylabel('')
        
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'latrunculinA & 0.1' : 'LatA 0.1 µM', 
      'latrunculinA & 0.5' : 'LatA 0.5 µM', 
      'latrunculinA & 2.5' : 'LatA 2.5 µM', 
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder = 0)
    renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD, loc='upper left')
    ax.set_xlim([0, 750])
    ax.set_ylim([0, 450])
    ax.set_xlabel('$H_{5mT}$ (nm)')


# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'LatA'
# drugSuffix = '_LatA'
# name = 'Fluctuations' + drugSuffix
figSubDir = ''
drugPrefix = 'LatA_'
name = drugPrefix + 'Fluctuations'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%% Non-linearity

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

dates = ['23-11-26', '23-12-03']
drugs = ['dmso', 'latrunculinA']
cons = ['0.0', '0.1', '0.5', '2.5']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
Ncells = CountByCond.cellCount.values[0]
Ncomps = CountByCond.compCount.values[0]

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12/gs.cm_in, 18/gs.cm_in), sharex=True, sharey=True)

intervals = ['300_700', '400_900', '600_1200']
cL = plt.cm.plasma(np.linspace(0.1, 0.9, len(intervals)))

rD = {'dmso & 0.0' : 'DMSO',
      'latrunculinA & 0.1' : 'LatA 0.1 µM', 
      'latrunculinA & 0.5' : 'LatA 0.5 µM', 
      'latrunculinA & 2.5' : 'LatA 2.5 µM', 
      }

for k, cond in enumerate(co_order):  
    for i, interval in enumerate(intervals):
        ax = axes[i]
        ax.set_yscale('log')
        df_fc = df_f[df_f[condCol] == cond]        
        
        # colorDict = {cond:cL[i]}
        # labelDict = {cond:f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"}
        colorDict = {cond:styleDict[cond]['color']}
        labelDict = {cond:rD[cond]}
        plotPopKS_V2(df_fc, fig = fig, ax = ax, condition = condCol, co_order = [], 
                     colorDict = colorDict, labelDict = labelDict, markersizefactor = 0.75,
                     fitType = 'stressGaussian', fitWidth=75, mode = interval, Sinf = 0, Ssup = np.Inf)
    
        # Prettify
        ax.grid(visible=True, which='both', axis='y')
        # legendTitle = r'$\bf{Whole\  dataset}$ | ' + f'{Ncells} cells | {Ncomps} comp'
        legendTitle = r"$\sigma \in$" + f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"
        ax.legend(loc='lower right', fontsize = 7, title = legendTitle, title_fontsize = 8)
        # ax.legend().set_visible(False)
        ax.set_xlim(0, 1250)
        ax.set_ylim(0.9, 30)
        if i != len(intervals)-1:
            ax.set_xlabel('')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'LatA'
# drugSuffix = '_LatA'
# name = 'K-sigma-3ranges' + drugSuffix
figSubDir = ''
drugPrefix = 'LatA_'
name = drugPrefix + 'K-sigma-3ranges'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%% Thickness & Stiffness

# %%%% 1. E400

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

dates = ['23-11-26', '23-12-03']
drugs = ['dmso', 'latrunculinA']
cons = ['0.0', '0.1', '0.5', '2.5']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'bestH0'
YCol = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(2, 2, figsize=(17/gs.cm_in, 15/gs.cm_in), sharex=True, sharey=True)
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
                        f'\nk = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
                        f'\n$R^2$ = {R2:.2f}' + \
                        f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8)
    ax.set_xlabel('$H_0}$ (nm)')
    ax.set_ylabel('$E_{400}$ (kPa)')
    ax.set_title(co_order[i])
    if i%2 != 0:
        ax.set_ylabel('')
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'latrunculinA & 0.1' : 'LatA 0.1 µM', 
      'latrunculinA & 0.5' : 'LatA 0.5 µM', 
      'latrunculinA & 2.5' : 'LatA 2.5 µM', 
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
# figSubDir = 'LatA'
# drugSuffix = '_LatA'
# name = 'H0-E400' + drugSuffix
figSubDir = ''
drugPrefix = 'LatA_'
name = drugPrefix + 'H0-E400'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%%% 2. K
# %%%%% Dataset

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

dates = ['23-11-26', '23-12-03']
drugs = ['dmso', 'latrunculinA']
cons = ['0.0', '0.1', '0.5', '2.5']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'bestH0'
YCol = 'fit_K'
fitID='600_100'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])

lb = int(fitID.split('_')[0])-int(fitID.split('_')[1])
ub = int(fitID.split('_')[0])+int(fitID.split('_')[1])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           (df[XCol] >= 50),
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

gs.set_manuscript_options_jv()

# Order
df_ff['co'] = df_ff['concentration'].astype(float)
co_order = df_ff[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()

# Plot
fig, axes = plt.subplots(2, 2, figsize=(17/gs.cm_in, 15/gs.cm_in), sharex=True, sharey=True)
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
                        f'\nk = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
                        f'\n$R^2$ = {R2:.2f}' + \
                        f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8)
    ax.set_xlabel('$H_0}$ (nm)')
    ylabel  = '$K$ (kPa)'
    ylabel += r" - $\sigma \in$" + f"[{lb}, {ub}] Pa"
    ax.set_ylabel(ylabel)
    ax.set_title(co_order[i])
    if i%2 != 0:
        ax.set_ylabel('')
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'latrunculinA & 0.1' : 'LatA 0.1 µM', 
      'latrunculinA & 0.5' : 'LatA 0.5 µM', 
      'latrunculinA & 2.5' : 'LatA 2.5 µM', 
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder=0)
    renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD)
    ax.set_xlim(50, 1000)
    # ax.set_ylim(0.4, 100)

# axes[0].set_xlabel('')



# Show
plt.tight_layout()
plt.show()



# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'LatA'
# drugSuffix = '_LatA'
# name = f'H0-K-{lb}-{ub}' + drugSuffix
figSubDir = ''
drugPrefix = 'LatA_'
name = drugPrefix + f'H0-K-{lb}-{ub}'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')





# %% Plots CK666

df = MecaData_DrugV3
# print(df.drug.unique())
print(df[df['drug']=='ck666']['date'].unique())
print(df[df['drug']=='ck666']['cell subtype'].unique())
print(df[df['drug']=='ck666']['concentration'].unique())

# %%% bestH0

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

dates = ['23-04-26', '23-04-28']
drugs = ['dmso', 'ck666']
cons = ['0.0', '50.0', '100.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

parameter = 'bestH0'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
            (df['bestH0'] <= 950),
            (df['bestH0'] >= 50),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'
bp = makeBoxPairs(co_order)
bp = bp[:]

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [parameter], aggFun = 'mean')

# Plot
fig, ax = plt.subplots(1,1, figsize=(12/gs.cm_in, 10/gs.cm_in))
# fig, ax = plt.subplots(1,1, figsize=(12, 8))
ax.set_yscale('log')

fig, ax = D1Plot(df_fg, fig = fig, ax = ax, condition=condCol, parameter=parameter,
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
ax.set_ylim([50, 2750])    

# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'ck666 & 50.0' : 'CK666 50 µM', 
      'ck666 & 100.0' : 'CK666 100 µM', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
# renameLegend(ax, renameDict)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
ax.set_ylabel('$H_0$ (nm)')
# ax.set_ylim([100, 2500])


# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'CK666'
# drugSuffix = '_CK666'
# name = 'bestH0' + drugSuffix
figSubDir = ''
drugPrefix = 'CK666_'
name = drugPrefix + 'bestH0'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')




# %%% E400

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

dates = ['23-04-26', '23-04-28']
drugs = ['dmso', 'ck666']
cons = ['0.0', '50.0', '100.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

parameter = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           (df[parameter] <= 2e4),
            (df['valid_f_<_400'] == True),
           (df['bestH0'] <= 950),
           (df['bestH0'] >= 50),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'
bp = makeBoxPairs(co_order)
bp = bp[:]

# Group By
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_fgw2[parameter+'_wAvg'] /= 1000

# Plot
fig, ax = plt.subplots(1,1, figsize=(12/gs.cm_in, 10/gs.cm_in))
ax.set_yscale('log')


fig, ax = D1Plot(df_fgw2, fig = fig, ax = ax, condition=condCol, parameter=parameter +'_wAvg',
                 co_order = co_order, boxplot = 2, figSizeFactor = 1, markersizeFactor = 1.0,
                 stats=True, statMethod='Mann-Whitney', box_pairs = bp, statVerbose = False,
                 showMean = False)
ax.set_ylim([0.4, 200])
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'ck666 & 50.0' : 'CK666 50 µM', 
      'ck666 & 100.0' : 'CK666 100 µM', 
      }

renameAxes(ax, rD, format_xticks = True)
renameAxes(ax, renameDict, format_xticks = True)
ax.grid(visible=True, which='major', axis='y')
ax.set_xlabel('')
ax.set_ylabel('$E_{400}$ (kPa)')

# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'CK666'
# drugSuffix = '_CK666'
# name = 'E400' + drugSuffix
figSubDir = ''
drugPrefix = 'CK666_'
name = drugPrefix + 'E400'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%% Fluctuations

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

dates = ['23-04-26', '23-04-28']
drugs = ['dmso', 'ck666']
cons = ['0.0', '50.0', '100.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           ]
df_f = filterDf(df, Filters)

index_f = df_f[(df_f[YCol] >= 400)].index
# print(df_f.loc[index_f, 'cellID'])
df_f = df_f.drop(index_f, axis=0)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'


# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Refilter

Filters = [(df_fg['compNum'] >= 5), 
           ]
df_fg2 = filterDf(df_fg, Filters)


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
fig, axes = plt.subplots(3, 1, figsize=(12/gs.cm_in, 20/gs.cm_in), sharey=True, sharex=True)
axes = axes.flatten('C')

for i in range(len(axes)):
    ax = axes[i]
    df_fc = df_fg2[df_fg2[condCol] == co_order[i]]
    color = styleDict[co_order[i]]['color']
    
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
                    f'\na = {a:.2f}' + r'$\pm$' + f'{a_cihw:.2f}' + \
                    f'\nb = {b:.2f}' + \
                    f'\n$R^2$ = {R2:.2f}' + \
                    f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8)
    ax.set_xlabel('$H_{5mT}$ (nm)')
    ax.set_ylabel('$H_{5mT}$ fluctuations (nm) [$D_9$-$D_1$]')
    ax.set_title(co_order[i])
    if i%2 != 0:
        ax.set_ylabel('')
        
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'ck666 & 50.0' : 'CK666 50 µM', 
      'ck666 & 100.0' : 'CK666 100 µM', 
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder = 0)
    renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD, loc='upper left')
    ax.set_xlim([0, 750])
    ax.set_ylim([0, 450])
    ax.set_xlabel('$H_{5mT}$ (nm)')


# Show
plt.tight_layout()
plt.show()


# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'CK666'
# drugSuffix = '_CK666'
# name = 'Fluctuations' + drugSuffix
figSubDir = ''
drugPrefix = 'CK666_'
name = drugPrefix + 'Fluctuations'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%% Non-linearity

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

dates = ['23-04-26', '23-04-28']
drugs = ['dmso', 'ck666']
cons = ['0.0', '50.0', '100.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'ctFieldThickness'
YCol = 'ctFieldFluctuAmpli'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()
# co_order = ['dmso & 0.0', 'blebbistatin & 10.0', 'blebbistatin & 50.0', 'blebbistatin & 100.0'] # , 'blebbistatin & 250.0'

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
Ncells = CountByCond.cellCount.values[0]
Ncomps = CountByCond.compCount.values[0]

# Group By
# df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = [XCol, YCol], aggFun = 'mean')

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12/gs.cm_in, 18/gs.cm_in), sharex=True, sharey=True)

intervals = ['200_500', '300_700', '400_900']
# intervals = ['250_450', '350_650', '450_850']
cL = plt.cm.plasma(np.linspace(0.1, 0.9, len(intervals)))

rD = {'dmso & 0.0' : 'DMSO',
      'ck666 & 50.0' : 'CK666 50 µM', 
      'ck666 & 100.0' : 'CK666 100 µM', 
      }

for k, cond in enumerate(co_order):  
    for i, interval in enumerate(intervals):
        ax = axes[i]
        ax.set_yscale('log')
        df_fc = df_f[df_f[condCol] == cond]        
        
        # colorDict = {cond:cL[i]}
        # labelDict = {cond:f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"}
        colorDict = {cond:styleDict[cond]['color']}
        labelDict = {cond:rD[cond]}
        plotPopKS_V2(df_fc, fig = fig, ax = ax, condition = condCol, co_order = [], 
                     colorDict = colorDict, labelDict = labelDict, markersizefactor = 0.75,
                     fitType = 'stressGaussian', fitWidth=75, mode = interval, Sinf = 0, Ssup = np.Inf)
    
        # Prettify
        ax.grid(visible=True, which='both', axis='y')
        # legendTitle = r'$\bf{Whole\  dataset}$ | ' + f'{Ncells} cells | {Ncomps} comp'
        legendTitle = r"$\sigma \in$" + f"[{interval.split('_')[0]}, {interval.split('_')[1]}] Pa"
        ax.legend(loc='lower right', fontsize = 7, title = legendTitle, title_fontsize = 8)
        # ax.legend().set_visible(False)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0.9, 30)
        if i != len(intervals)-1:
            ax.set_xlabel('')


# Show
plt.tight_layout()
plt.show()

# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'CK666'
# drugSuffix = '_CK666'
# name = 'K-sigma-3ranges' + drugSuffix
figSubDir = ''
drugPrefix = 'CK666_'
name = drugPrefix + 'K-sigma-3ranges'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')


# %%% Thickness & Stiffness

# %%%% 1. E400

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

dates = ['23-04-26', '23-04-28']
drugs = ['dmso', 'ck666']
cons = ['0.0', '50.0', '100.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'bestH0'
YCol = 'E_f_<_400'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])


# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           (df[XCol] >= 50),
           (df[XCol] <= 1000),
           (df[YCol] <=  1e5),
           ]
df_f = filterDf(df, Filters)

# Order
df_f['co'] = df_f['concentration'].astype(float)
co_order = df_f[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()

# Group By
df_fg = dataGroup(df_f, groupCol = 'cellID', idCols = [condCol], numCols = ['bestH0'], aggFun = 'mean') #.drop(columns=['cellID']).reset_index()
df_fg = df_fg[['bestH0']]
df_fgw2 = dataGroup_weightedAverage(df_f, groupCol = 'cellID', idCols = [condCol], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
df_plot = pd.merge(left=df_fg, right=df_fgw2, on='cellID', how='inner')

# Plot
fig, axes = plt.subplots(3, 1, figsize=(9.5/gs.cm_in, 22.5/gs.cm_in), sharex=True, sharey=True)
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
            label =  r'$\bf{Fit\ y=ax+b}$' + \
                    f'\na = {a:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
                    f'\nb = {b:.2f}' + \
                    f'\n$R^2$ = {R2:.2f}' + \
                    f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8)
    ax.set_xlabel('$H_0}$ (nm)')
    ax.set_ylabel('$E_{400}$ (kPa)')
    ax.set_title(co_order[i])
    # if i%2 != 0:
    #     ax.set_ylabel('')
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'ck666 & 50.0' : 'CK666 50 µM', 
      'ck666 & 100.0' : 'CK666 100 µM', 
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
# figSubDir = 'CK666'
# drugSuffix = '_CK666'
# name = 'H0-E400' + drugSuffix
figSubDir = ''
drugPrefix = 'CK666_'
name = drugPrefix + 'H0-E400'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')

# %%%% 2. K
# %%%%% Dataset

gs.set_manuscript_options_jv()

# Define
df = MecaData_DrugV3

dates = ['23-04-26', '23-04-28']
drugs = ['dmso', 'ck666']
cons = ['0.0', '50.0', '100.0']
substrate = '20um fibronectin discs'
subtypes = ['Atcc-2023', 'Atcc-2023-LaGFP']

XCol = 'bestH0'
YCol = 'fit_K'
fitID='300_100'
df, condCol = makeCompositeCol(df, cols=['drug', 'concentration'])

lb = int(fitID.split('_')[0])-int(fitID.split('_')[1])
ub = int(fitID.split('_')[0])+int(fitID.split('_')[1])

# Filter
Filters = [(df['validatedThickness'] == True), 
           (df['substrate'] == substrate),
           (df['date'].apply(lambda x : x in dates)),
           (df['drug'].apply(lambda x : x in drugs)),
           (df['cell subtype'].apply(lambda x : x in subtypes)),
           (df['concentration'].apply(lambda x : x in cons)),
           (df[XCol] >= 50),
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

gs.set_manuscript_options_jv()

# Order
df_ff['co'] = df_ff['concentration'].astype(float)
co_order = df_ff[[condCol, 'co']].sort_values(by = 'co')[condCol].unique()

# Plot
fig, axes = plt.subplots(3, 1, figsize=(9.5/gs.cm_in, 22.5/gs.cm_in), sharex=True, sharey=True)
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
                        f'\nk = {k:.2f}' + r'$\pm$' + f'{k_cihw:.2f}' + \
                        f'\n$R^2$ = {R2:.2f}' + \
                        f'\np-val = {pval:.2f}')

    ax.legend(fontsize = 8)
    ax.set_xlabel('$H_0}$ (nm)')
    ylabel  = '$K$ (kPa)'
    ylabel += r" - $\sigma \in$" + f"[{lb}, {ub}] Pa"
    ax.set_ylabel(ylabel)
    ax.set_title(co_order[i])
    # if i%2 != 0:
    #     ax.set_ylabel('')
           
# Prettify
rD = {'dmso & 0.0' : 'DMSO',
      'ck666 & 50.0' : 'CK666 50 µM', 
      'ck666 & 100.0' : 'CK666 100 µM', 
      }

for ax in axes:
    ax.grid(visible=True, which='major', axis='both', zorder=0)
    renameAxes(ax, rD, format_xticks = False)
    # renameAxes(ax, renameDict, format_xticks = False)
    # renameLegend(ax, rD)
    ax.set_xlim(50, 1000)
    # ax.set_ylim(0.4, 100)

# axes[0].set_xlabel('')



# Show
plt.tight_layout()
plt.show()



# Count
CountByCond, CountByCell = makeCountDf(df_f, condCol)
# Save
# figSubDir = 'CK666'
# drugSuffix = '_CK666'
# name = f'H0-K-{lb}-{ub}' + drugSuffix
figSubDir = ''
drugPrefix = 'CK666_'
name = drugPrefix + f'H0-K-{lb}-{ub}'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
CountByCond.to_csv(os.path.join(figDir, figSubDir, name+'_count.txt'), sep='\t')








# %% Next !

