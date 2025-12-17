# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 12:36:00 2025

@author: Utilisateur
"""

# %% > Imports and constants

#### Main imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os
import re
import sys
import itertools
import matplotlib

from statannotations.Annotator import Annotator
from scipy.stats import mannwhitneyu

#### Local Imports

import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)

import GraphicStyles as gs
import UtilityFunctions as ufun
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

def plot_loghist(ax, x, bins=10, color = 'gray', normalized = False):
    # hist, bins = np.histogram(x, bins=bins)
    # logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    # ax.hist(x, bins=logbins, color = color)
    logbins = np.logspace(np.log10(min(x)), np.log10(max(x)), bins, endpoint=False)
    # hist, bins = np.histogram(x, bins=logbins)
    if normalized:
        x = x / len(x)
    histo, bins, patches = ax.hist(x, bins=logbins, color = color)
    ax.set_xscale('log')
    return(ax, histo, logbins)

def plot_logstairs(ax, x, bins=10, logbins = [], normalized = False, 
                   fill = False, color = 'gray', label = ''):
    if len(logbins) == 0:
        logbins = np.logspace(np.log10(min(x)), np.log10(max(x)), bins, endpoint=False)
    histo, bins = np.histogram(x, bins=logbins)
    if normalized:
        histo = histo / np.sum(histo)
    ax.stairs(histo, edges=logbins, color = color, label = label, fill = fill, lw = 1.5)
    ax.set_xscale('log')
    return(ax, histo, logbins)

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



# def plotPopKS_V2(data, fig = None, ax = None, 
#                  condition = '', co_order = [], colorDict = {}, labelDict = {},
#                  fitType = 'stressRegion', fitWidth=75,
#                  mode = 'wholeCurve', scale = 'lin', Sinf = 0, Ssup = np.inf):
    
#     #### Init
#     co_values = data[condition].unique()     
#     if len(co_order) == 0:
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
#     data_fits = data_fits.drop(data_fits[data_fits['fit_K'] > 1e5].index)
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
#                                    condition:'first'})
#     data_agg_cells = data_agg_cells.reset_index()
#     data_agg_cells['K_wAvg'] = data_agg_cells['A']/data_agg_cells['weight']
#     data_agg_cells = data_agg_cells.rename(columns = {'compNum' : 'compCount'})
    
#     # 2nd selection
#     data_agg_cells = data_agg_cells.drop(data_agg_cells[data_agg_cells['compCount'] <= 1].index)
#     data_agg_cells = data_agg_cells.drop(data_agg_cells[data_agg_cells['weight'] <= 1].index)
    
#     grouped2 = data_agg_cells.groupby(by=[condition, 'fit_center'])
#     data_agg_all = grouped2.agg({'compCount':['sum', 'count'],
#                                    'K_wAvg':['mean', 'std', 'median']})
#     data_agg_all = data_agg_all.reset_index()
#     data_agg_all.columns = ufun.flattenPandasIndex(data_agg_all.columns)
#     data_agg_all = data_agg_all.rename(columns = {'compCount_sum' : 'compCount', 'compCount_count' : 'cellCount'})
#     data_agg_all['K_wAvg_ste'] = data_agg_all['K_wAvg_std']/data_agg_all['cellCount']**0.5
    
    
#     #### Plot
#     i_color = 0
#     for co in co_order:
#         if co in colorDict.keys():
#             pass
#         else:
#             try:
#                 colorDict[co] = styleDict[co]['color']
#             except:
#                 colorDict[co] = gs.colorList10[i_color%10]
#                 i_color = i_color + 1
        
#         if co in labelDict.keys():
#             pass
#         else:
#             labelDict[co] = str(co)
            
    
#     for co in co_order:
#         try:
#             color = colorDict[co]
#             label_co = labelDict[co]
                
#             df = data_agg_all[data_agg_all[condition] == co]
            
#             centers = df['fit_center'].values
#             Kavg = df['K_wAvg_mean'].values
#             Kste = df['K_wAvg_ste'].values
#             N = df['cellCount'].values
#             total_N = np.max(N)
            
#             n = df['compCount'].values
#             total_n = np.max(n)
            
#             dof = N
#             alpha = 0.975
#             q = st.t.ppf(alpha, dof) # Student coefficient
                
#             # weighted means -- weighted ste 95% as error
            
#             ax.errorbar(centers, Kavg/1000, yerr = q*Kste/1000, 
#                         color = color, lw = 2, marker = 'o', markersize = 6, mec = 'k',
#                         ecolor = color, elinewidth = 1.5, capsize = 6, capthick = 1.5, 
#                         label = labelDict[co] + ' | ' + str(total_N) + ' cells' + ' | ' + str(total_n) + ' comp')
            
#             ax.legend(loc = 'upper left', fontsize = 8)
#             ax.set_xlabel('Stress (Pa)')
#             ax.set_ylabel('K (kPa)')
#             ax.grid(visible=True, which='major', axis='y')
                    
#         except:
#             pass


#     return(fig, ax)







# def plotPopKS_V3(data, fig = None, ax = None, 
#                  condition = '', co_order = [], colorList = gs.colorList30, 
#                  fitType = 'stressRegion', fitWidth=75,
#                  mode = 'wholeCurve', scale = 'lin', Sinf = 0, Ssup = np.inf,
#                  legend_cells = False, legend_comp = False):
    
#     #### Init
#     if ax == None:
#         figHeight = 10/gs.cm_in
#         figWidth = 17/gs.cm_in
#         fig, ax = plt.subplots(1,1, figsize=(figWidth, figHeight))
        
#     cl = colorList
#     ml = ['o', 'D', 'P', 'X', '^']
    
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
#     data_fits = taka3.getFitsInTable(data, fitType=fitType, filter_fitID=fitId)
    
#     # Filter the table
#     data_fits = data_fits[(data_fits['fit_center'] >= Sinf) & (data_fits['fit_center'] <= Ssup)]   
#     data_fits = data_fits.drop(data_fits[data_fits['fit_error'] == True].index)
#     data_fits = data_fits.dropna(subset = ['fit_ciwK'])
    
#     data_fits = data_fits.drop(data_fits[data_fits['fit_K'] < 0].index)
#     data_fits = data_fits.drop(data_fits[data_fits['fit_K'] < data_fits['fit_ciwK']/2].index)
    
    
    
#     # Compute the weights
#     # data_fits['weight'] = (data_fits['fit_K']/data_fits['fit_ciwK'])**2
    
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
#     # data_fits['A'] = data_fits['fit_K'] * data_fits['weight']
#     # grouped1 = data_fits.groupby(by=['cellID', 'fit_center'])
#     # data_agg_cells = grouped1.agg({'compNum':'count',
#     #                                'A':'sum', 
#     #                                'weight': 'sum',
#     #                                condition:'first'})
#     # data_agg_cells = data_agg_cells.reset_index()
#     # data_agg_cells['K_wAvg'] = data_agg_cells['A']/data_agg_cells['weight']
#     # data_agg_cells = data_agg_cells.rename(columns = {'compNum' : 'compCount'})
    
#     # # 2nd selection
#     # data_agg_cells = data_agg_cells.drop(data_agg_cells[data_agg_cells['compCount'] <= 1].index)
#     # data_agg_cells = data_agg_cells.drop(data_agg_cells[data_agg_cells['weight'] <= 1].index)
    
#     # grouped2 = data_agg_cells.groupby(by=[condition, 'fit_center'])
#     # data_agg_all = grouped2.agg({'compCount':['sum', 'count'],
#     #                                'K_wAvg':['mean', 'std', 'median']})
#     # data_agg_all = data_agg_all.reset_index()
#     # data_agg_all.columns = ufun.flattenPandasIndex(data_agg_all.columns)
#     # data_agg_all = data_agg_all.rename(columns = {'compCount_sum' : 'compCount', 'compCount_count' : 'cellCount'})
#     # data_agg_all['K_wAvg_ste'] = data_agg_all['K_wAvg_std']/data_agg_all['cellCount']**0.5
    
    
#     data_fits['compID'] = data_fits['cellID'] + '_' + data_fits['compNum'].apply(lambda x : str(x))
#     list_cellID = data_fits['cellID'].unique()
#     Ncells = len(list_cellID)
    
#     #### Plot
#     i_color = 0
    
#     for i, cid in enumerate(list_cellID):
#         color = cl[i%len(cl)]
#         dfcell = data_fits.loc[data_fits['cellID'] == cid]
#         plot_parms = {'color':color, 'marker':'o', 'markersize':5, 'mec':'None',}
#         if legend_cells:
#             ax.plot([], [], label = cid.split('_')[-1], color = plot_parms['color'], ls='-')
        
#         for j in range(5):
#             if (j+1) in dfcell['compNum'].values:
#                 marker = ml[j]
#                 marker = 'o'
#                 plot_parms.update({'marker':marker, 'mec':'k', 'mew':0.5})
#                 errplot_parms = {**plot_parms, 'ecolor':color, 'elinewidth':1.5, 
#                                  'capsize':6, 'capthick':1.5, 'alpha':0.9, 'zorder':3}
#                 dfcomp = dfcell.loc[dfcell['compNum'] == j+1]
    
#                 centers = dfcomp['fit_center'].values
#                 K = dfcomp['fit_K'].values
#                 Kciw = dfcomp['fit_ciwK'].values/2
    
#                 # ax.plot(centers, K/1000, **plot_parms)
#                 ax.errorbar(centers, K/1000, yerr = Kciw/1000, **errplot_parms)
                
#     # imax = i
#     # nskip = 5 - (imax%5 + 1)
#     # for k in range(nskip):
#     #     ax.plot([], [], label=' ', c = 'None')
    
#     if legend_comp:
#         for j in range(5):
#             marker = ml[j]
#             plot_parms = {'color':'gray', 'marker':marker, 'markersize':6, 'mec':'k', 'mew':0.5}
#             ax.plot([], [], label = f'Comp n°{j+1:.0f}', **plot_parms)
    
#     if legend_cells or legend_comp:
#         ax.legend(loc = 'upper left', fontsize = 8, ncols=3)
#     else:
#         ax.legend().set_visible(False)
#     ax.set_xlabel('Stress (Pa)')
#     ax.set_ylabel('K (kPa)')
#     ax.grid(visible=True, which='major', axis='y')

#     return(fig, ax)








# def StressRange_2D(data, condition='', split = False, defaultColors = False):
        
#     co_values = list(data[condition].unique())
#     Nco = len(co_values)
    
#     if not defaultColors:
#         try:
#             colorList, markerList = getStyleLists(co_values, styleDict)
#         except:
#             colorList, markerList = gs.colorList30, gs.markerList10
#     else:
#         colorList, markerList = gs.colorList30, gs.markerList10

#     if split:
#         fig, axes = plt.subplots(Nco, 1, figsize = (8, 3*Nco), sharex = True)
#     else:
#         fig, axes = plt.subplots(1, 1, figsize = (8, 6))
    
#     for i in range(Nco):
#         if split:
#             ax = axes[i]
#         else:
#             ax = axes
        
#         co = co_values[i]
#         data_co = data[data[condition] == co]
#         color = colorList[i]
#         ax.scatter(data_co['bestH0'], data_co['minStress'], marker = 'o', s = 3, color = color, edgecolor = 'k', zorder=4)
#         ax.scatter(data_co['bestH0'], data_co['maxStress'], marker = 'o', s = 3, color = color, edgecolor = 'k', zorder=4)
#         ax.vlines(data_co['bestH0'], data_co['minStress'], data_co['maxStress'], color = colorList[i], label = co)
#         ax.set_xlim([0, 800])
#         ax.set_ylim([0, 7500])

#     out = fig, axes    
    
#     return(out)


# def StressRange_2D_V2(data, fig=None, ax=None, colorList = gs.colorList30, 
#                       condition='', defaultMode = False):
        
#     co_values = list(data[condition].unique())
#     Nco = len(co_values)
#     # cl = matplotlib.colormaps['Set1'].colors[1:] + matplotlib.colormaps['Set2'].colors[:-1]
#     cl = colorList
    
#     # if not defaultColors:
#     #     try:
#     #         colorList, markerList = getStyleLists(co_values, styleDict)
#     #     except:
#     #         colorList, markerList = gs.colorList30, gs.markerList10
#     # else:
#     #     colorList, markerList = gs.colorList30, gs.markerList10

#     if ax == None:
#         figHeight = 10/gs.cm_in
#         figWidth = 17/gs.cm_in
#         fig, ax = plt.subplots(1,1, figsize=(figWidth, figHeight))
    
#     for i in range(Nco):      
#         co = co_values[i]
#         data_co = data[data[condition] == co]
#         if defaultMode:
#             color = gs.colorList40[19]
#             alpha = 0.4
#             zo = 4
#             s = 4
#             ec = 'None'
#             labels = ['Minimum stress', 'Maximum stress', 'Compression']
#         else:
#             color = cl[i]
#             alpha = 1
#             zo = 6
#             s = 15
#             ec = color
#             labels = ['', '', '']
            
            
#         ax.scatter(data_co['bestH0'], data_co['minStress'], marker = 'o', s = s, color = 'deepskyblue', edgecolor = ec, zorder=zo, 
#                    label = labels[0])
#         ax.scatter(data_co['bestH0'], data_co['maxStress'], marker = 'o', s = s, color = 'darkred', edgecolor = ec, zorder=zo,
#                    label = labels[1])
#         ax.vlines(data_co['bestH0'], data_co['minStress'], data_co['maxStress'], color = color, alpha = alpha, zorder=zo-1,
#                   label = labels[2])
        
#     # fig.legend()
    
#     out = fig, axes    
    
#     return(out)


# def StressRange_2D_V3(data, fig=None, ax=None):
    
#     Npp = 50
#     Ntot = len(data['bestH0'].values)
#     Nb = Ntot//Npp
#     step = 100/Nb
#     Lp = step * np.arange(Nb)
#     bins = np.percentile(data['bestH0'].values, Lp)
#     data['H0_Bin'] = np.digitize(data['bestH0'], bins, right = True)
    
#     agg_dict = {'compNum':'count',
#                 'bestH0':['mean', 'std'],
#                 'minStress':['mean', 'std'],
#                 'maxStress':['mean', 'std'],
#                 }
#     all_cols = ['H0_Bin'] + list(agg_dict.keys())
#     group = data[all_cols].groupby('H0_Bin')
#     data_g = group.agg(agg_dict)
#     return(data_g)
    
#     if ax == None:
#         figHeight = 10/gs.cm_in
#         figWidth = 17/gs.cm_in
#         fig, ax = plt.subplots(1,1, figsize=(figWidth, figHeight))
    
#     for i in range(Nco):      
#         co = co_values[i]
#         data_co = data[data[condition] == co]
#         color = gs.colorList40[19]
#         alpha = 0.4
#         zo = 4
#         s = 4
#         ec = 'None'
#         labels = ['Minimum stress', 'Maximum stress', 'Compression']
            
#         ax.scatter(data_co['bestH0'], data_co['minStress'], marker = 'o', s = s, color = 'deepskyblue', edgecolor = ec, zorder=zo, 
#                    label = labels[0])
#         ax.scatter(data_co['bestH0'], data_co['maxStress'], marker = 'o', s = s, color = 'darkred', edgecolor = ec, zorder=zo,
#                    label = labels[1])
#         ax.vlines(data_co['bestH0'], data_co['minStress'], data_co['maxStress'], color = color, alpha = alpha, zorder=zo-1,
#                   label = labels[2])
        
#     # fig.legend()
    
#     out = fig, axes    
    
#     return(out)


# %% > Anumita's awesome plots



# def plotNLI(fig, ax, data, condCat, condCol, pairs, labels = [],  palette = ['#b96a9b', '#d6c9bc', '#92bda4'],
#             colorScheme = 'black', **plotChars):

#     if colorScheme == 'black':
#         plt.style.use('dark_background')
#         fontColor = '#ffffff'
#         lineColor = '#ffffff'
#     else:
#         plt.style.use('default')
#         fontColor = '#000000'
#         lineColor = '#000000'

#     NComps = data.groupby(condCol)['NLI_Plot'].count().reindex(condCat, axis = 0).reset_index()
#     linear = data[data.NLI_Plot=='linear'].groupby(condCol)['NLI_Plot'].count().reindex(condCat, axis = 0).reset_index()
#     nonlinear = data[data.NLI_Plot=='non-linear'].groupby(condCol)['NLI_Plot'].count().reindex(condCat, axis = 0).reset_index()
#     intermediate = data[data.NLI_Plot=='intermediate'].groupby(condCol)['NLI_Plot'].count().reindex(condCat, axis = 0).reset_index()

#     linear['NLI_Plot'] = [i / j * 100 for i,j in zip(linear['NLI_Plot'], NComps['NLI_Plot'])]
#     nonlinear['NLI_Plot'] = [i / j * 100 for i,j in zip(nonlinear['NLI_Plot'], NComps['NLI_Plot'])]
#     intermediate['NLI_Plot'] = [i / j * 100 for i,j in zip(intermediate['NLI_Plot'], NComps['NLI_Plot'])]

#     y1 = linear['NLI_Plot'].values
#     y2 = intermediate['NLI_Plot'].values
#     y3 = nonlinear['NLI_Plot'].values
#     N = NComps['NLI_Plot'].values

#     nonlinear['NLI_Plot'] = linear['NLI_Plot'] + nonlinear['NLI_Plot'] + intermediate['NLI_Plot']
#     intermediate['NLI_Plot'] = intermediate['NLI_Plot'] + linear['NLI_Plot']

#     sns.barplot(x=condCol,  y="NLI_Plot", data=nonlinear, color=palette[0],  ax = ax)
#     sns.barplot(x=condCol,  y="NLI_Plot", data=intermediate, color=palette[1], ax = ax)
#     sns.barplot(x=condCol,  y="NLI_Plot", data=linear, color=palette[2], ax = ax)

#     xticks = np.arange(len(condCat))

#     for xpos, ypos, yval in zip(xticks, y1/2, y1):
#         plt.text(xpos, ypos, "%.1f"%yval + '%', ha="center", va="center", color = '#000000', fontsize = 8)
#     for xpos, ypos, yval in zip(xticks, y1+y2/2, y2):
#         plt.text(xpos, ypos, "%.1f"%yval+ '%', ha="center", va="center", color = '#000000', fontsize = 8)
#     for xpos, ypos, yval in zip(xticks, y1+y2+y3/2, y3):
#         plt.text(xpos, ypos, "%.1f"%yval+ '%', ha="center", va="center", color = '#000000', fontsize = 8)
#     # add text annotation corresponding to the "total" value of each bar
#     for xpos, ypos, yval in zip(xticks, y1+y2+y3+0.5, N):
#         plt.text(xpos, ypos, "N=%d"%yval, ha="center", va="bottom", fontsize = 8)

#     try:
#         pvals = []
#         for pair in pairs:
#             a1 = data['NLI'][data[condCol] == pair[0]].values
#             b1 = data['NLI'][data[condCol] == pair[1]].values
#             U1, p = mannwhitneyu(a1, b1, nan_policy = 'omit')
#             pvals.append(p)

#         annotator = Annotator(ax = ax, pairs = pairs, x=condCol,  y="NLI_Plot", data=linear)
#         annotator.configure(text_format="simple", color = lineColor)
#         annotator.set_pvalues(pvals).annotate()
#     except:
#         pass

#     texts = ["Nonlinear", "Intermediate", "Linear"]
#     patches = [mpatches.Patch(color=palette[i], label="{:s}".format(texts[i]) ) for i in range(len(texts)) ]

#     if labels != []:
#         plt.xticks(xticks, labels, **plotChars)
    
#     ax.set_ylim([0, 110])
#     plt.xticks(xticks, **plotChars)
#     plt.yticks(**plotChars)
#     plt.tight_layout()
#     plt.legend(handles = patches, bbox_to_anchor=(1.01, 0.5), fontsize = 10, labelcolor='linecolor')
#     plt.show()

#     return(fig, ax, pvals)


# def computeNLMetrics(GlobalTable, th_NLI = np.log10(2), ref_strain = 0.2):

#     data_main = GlobalTable
#     data_main['dateID'] = GlobalTable['date']
#     data_main['manipId'] = GlobalTable['manipID']
#     data_main['cellId'] = GlobalTable['cellID']
#     data_main['dateCell'] = GlobalTable['date'] + '_' + GlobalTable['cellCode']

#     nBins = 10
#     bins = np.linspace(0, 1000, nBins)
#     data_main['H0_Bin'] = np.digitize(GlobalTable['bestH0'], bins, right = True)
#     data_main['Thickness_Bin'] = np.digitize(GlobalTable['surroundingThickness'], bins, right = True)
    
#     data_main['NLI_Plot'] = [np.nan]*len(data_main)
#     data_main['NLI_Ind'] = [np.nan]*len(data_main)
#     data_main['E_eff'] = [np.nan]*len(data_main)

#     K, Y = data_main['K_vwc_Full'], data_main['Y_vwc_Full']
#     E = Y + (K*(1 - ref_strain)**-4)

#     data_main['E_eff'] = E
#     data_main['NLI'] = np.log10(((1 - ref_strain)**-4 * K)/Y)
    
#     ciwK, ciwY = data_main['ciwK_vwc_Full'], data_main['ciwY_vwc_Full']
#     ciwE = ciwY + (ciwK*(1 - ref_strain)**-4)
#     data_main['ciwE_eff'] = ciwE
    
#     NLItypes = ['linear', 'intermediate', 'non-linear']
#     th_NLI = np.abs(th_NLI)
#     for i in NLItypes:
#         if i == 'linear':
#             index = data_main[data_main['NLI'] < -th_NLI].index
#             ID = 1
#         elif i =='non-linear':
#             index =  data_main[data_main['NLI'] > th_NLI].index
#             ID = 0
#         elif i =='intermediate':
#             index = data_main[(data_main['NLI'] > -th_NLI) & (data_main['NLI'] < th_NLI)].index
#             ID = 0.5
#         # for j in index:
#         data_main.loc[index, 'NLI_Plot'] = i
#         data_main.loc[index, 'NLI_Ind'] = ID
    
#     return(data_main)

# def computeNLMetrics_V2(GlobalTable, th_NLI = np.log10(2), ref_strain = 0.2):
#     data_main = GlobalTable
#     data_main['dateID'] = GlobalTable['date']
#     data_main['manipId'] = GlobalTable['manipID']
#     data_main['cellId'] = GlobalTable['cellID']
#     data_main['dateCell'] = GlobalTable['date'] + '_' + GlobalTable['cellCode']
#     data_main['wellID'] = GlobalTable['cellCode'].str.split('_').str[0]

#     nBins = 11
#     bins = np.linspace(1, 2000, nBins)
#     data_main['H0_Bin'] = np.digitize(GlobalTable['bestH0'], bins, right = True)
#     data_main['Thickness_Bin'] = np.digitize(GlobalTable['surroundingThickness'], bins, right = True)

#     data_main['NLI_Plot'] = [np.nan]*len(data_main)
#     data_main['NLI_Ind'] = [np.nan]*len(data_main)
#     data_main['E_eff'] = [np.nan]*len(data_main)

#     K, Y = data_main['K_vwc_Full'], data_main['Y_vwc_Full']
#     E = Y + K*(1 - ref_strain)**-4

#     data_main['E_eff'] = E
#     data_main['NLI'] = np.log10((1 - ref_strain)**-4 * K/Y)
    
#     ciwK, ciwY = data_main['ciwK_vwc_Full'], data_main['ciwY_vwc_Full']
#     ciwE = ciwY + (ciwK*(1 - ref_strain)**-4)
#     data_main['ciwE_eff'] = ciwE
    

#     data_main['Y_err_div10'], data_main['K_err_div10'] = data_main['ciwY_vwc_Full']/10, data_main['ciwK_vwc_Full']/10
#     data_main['Y_NLImod'] = data_main[["Y_vwc_Full", "Y_err_div10"]].max(axis=1)
#     data_main['K_NLImod'] = data_main[["K_vwc_Full", "K_err_div10"]].max(axis=1)
#     Y_nli, K_nli = data_main['Y_NLImod'].values, data_main['K_NLImod'].values

#     data_main['NLI_mod'] = np.log10((1 - ref_strain)**-4 * K_nli/Y_nli)
    
#     NLItypes = ['linear', 'intermediate', 'non-linear']
#     for i in NLItypes:
#         if i == 'linear':
#             index = data_main[data_main['NLI_mod'] < -th_NLI].index
#             ID = 1
#         elif i =='non-linear':
#             index =  data_main[data_main['NLI_mod'] > th_NLI].index
#             ID = 0
#         elif i =='intermediate':
#             index = data_main[(data_main['NLI_mod'] > -th_NLI) & (data_main['NLI_mod'] < th_NLI)].index
#             ID = 0.5
#         for j in index:
#             data_main['NLI_Plot'][j] = i
#             data_main['NLI_Ind'][j] = ID


#     return(data_main)

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

CountByCond, CountByCell = makeCountDf(MecaData_Phy, 'date')

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