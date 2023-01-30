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

import TrackAnalyser_dev3_AJ as tka3

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
        
        ax.legend(loc = 'upper left', fontsize = 11)
        ax.set_xlabel('Stress (Pa)')
        ax.set_ylabel('K (kPa)')
        ax.grid(visible=True, which='major', axis='y')
        
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


    return(output)

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
        c1 = data[data[cond] == bp[0]][param] #.values
        c2 = data[data[cond] == bp[1]][param] #.values
        
        if test == 'Mann-Whitney' or test == 'Wilcox_2s' or test == 'Wilcox_greater' or test == 'Wilcox_less' or test == 't-test':
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
            if pval == np.nan:
                text = 'nan'
            if pval < 0.05 and pval > 0.01:
                text = '*'
            elif pval < 0.01 and pval > 0.001:
                text = '**'
            elif pval < 0.001 and pval < 0.001:
                text = '***'
            elif pval < 0.0001:
                print('yes')
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

tka.computeGlobalTable_meca(task = 'updateExisting', fileName = 'Global_MecaData_AJ', 
                            save = True, PLOT = True, source = 'Python')

# %%%% Specific experiments
newFitSettings = {# H0
                       'methods_H0':['Chadwick', 'Dimitriadis'],
                       'zones_H0':[ 'pts_15',  '%f_15'],
                       'method_bestH0':'Dimitriadis',
                       'zone_bestH0':'%f_15',
                       }
# 
newfitValidationSettings = {'crit_nbPts': 6}

fitSettings = ufun.updateDefaultSettingsDict(newFitSettings, DEFAULT_fitSettings)
fitValidationSettings = ufun.updateDefaultSettingsDict(newfitValidationSettings, \
                                                        DEFAULT_fitValidationSettings)
    
# Task = '22-10-06 & 22-10-05 & 22-12-07'
Task = '22-12-07'

#'22-08-26_M7 & 22-08-26_M5 & 22-08-26_M10 & 22-08-26_M1 & 22-08-26_M3' # For instance '22-03-30 & '22-03-31'
fitsSubDir = '22-12-07_Dimi_f15_0Offset'

GlobalTable_meca = taka.computeGlobalTable_meca(task = Task, mode = 'fromScratch', \
                            fileName = 'Global_MecaData_22-12-07_Dimi_f15_0Offset', 
                            save = True, PLOT = False, source = 'Python', fitSettings = fitSettings,\
                               fitValidationSettings = fitValidationSettings, fitsSubDir = fitsSubDir) # task = 'updateExisting'

 # %%%% Precise dates (to plot)

date = '22-08-26' # For instance '22-03-30 & '22-03-31'
taka.computeGlobalTable_meca(task = date, mode = 'fromScratch', fileName = 'Global_MecaData_'+date, 
                            save = True, PLOT = True, source = 'Python') # task = 'updateExisting'

 # %%%% Plot all experiments

taka.computeGlobalTable_meca(task = 'all', mode = 'fromScratch', fileName = 'Global_MecaData_all', 
                            save = True, PLOT = True, source = 'Python', fitSettings = fitSettings) # task = 'updateExisting'

#%%
newFitSettings = {# H0
                       'methods_H0':['Chadwick', 'Dimitriadis'],
                       'zones_H0':['%pts_15', '%f_15'],
                       'method_bestH0':'Dimitriadis',
                       'zone_bestH0':'%f_15',
                       }

newfitValidationSettings = {'crit_nbPts': 6}
date = '23-01-23'
fitSubDir = '23-01-23_Dimi_f15'


fitSettings = ufun.updateDefaultSettingsDict(newFitSettings, DEFAULT_fitSettings)
fitValidationSettings = ufun.updateDefaultSettingsDict(newfitValidationSettings, \
                                                       DEFAULT_fitValidationSettings)

GlobalTable_mecam=  taka.computeGlobalTable_meca(task = date, mode = 'fromScratch', \
                            fileName = 'Global_MecaData_23-01-23_Dimi_f15', \
                            save = True, PLOT = True, source = 'Python', fitSettings = fitSettings,\
                               fitValidationSettings = fitValidationSettings, fitSubDir = fitSubDir ) # task = 'updateExisting'

# %% > Data import & export
#### GlobalTable_meca_Py

date = '22-08-26'
GlobalTable_meca = taka.getMergedTable('Global_MecaData_all')
GlobalTable_meca.head()
# %% Non-linear plots 


data = taka.getFitsInTable(data_main, fitType=fitType, filter_fitID=fitId)
data = data.drop(data[data['fit_error'] == True].index)
data = data.drop(data[data['fit_K'] < 0].index)
data = data.dropna(subset = ['fit_ciwK'])

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

plt.style.use('dark_background')
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

flatui = ["#dabcff", "#ffe257"]
sns.set_palette(flatui)

x = data_f['compNum']*20
sns.lineplot(x = x, y = 'surroundingThickness', data = data_f, hue = 'manip')

fig1.suptitle('[15mT = 500pN] Surrounding Thickness (nm) vs. Time (secs)')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('Time (secs)', fontsize = 25)
plt.ylabel('Surrounding thickness (nm)', fontsize = 25)

activated = mpatches.Patch(color="#ffe257", label='Activated')
control = mpatches.Patch(color='#dabcff', label='Non-activated')
plt.legend(handles=[activated, control], fontsize = 20)


plt.ylim(0,1500)

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

#%% Plots for 23-01-23

GlobalTable = taka.getMergedTable('Global_MecaData_23-01-23_Dimi_f15') #'_tka3_offset15pts')
data_main = GlobalTable
data_main['dateID'] = GlobalTable['date']
data_main['manipId'] = GlobalTable['manipID']
fitsSubDir = '23-01-23_Dimi_f15'

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

# selRows = data[(data['manip'] == 'M5') & (data['compNum'] > 3)].index
# data = data.drop(selRows, axis = 0)

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

out1 = plotPopKS(data, mainFig1, mainAx1, legendLabels = legendLabels,
                 fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = condCol, mode = 'wholeCurve', scale = 'lin', printText = False,
                                returnData = 1, returnCount = 1)

mainFig1, mainAx1, exportDf1, countDf1 = out1


out2 = plotPopKS(data, mainFig2, mainAx2, legendLabels = legendLabels,
                 fitType = 'stressGaussian', fitWidth=75, Filters = Filters, 
                                condCol = condCol, mode = '100_500', scale = 'lin', printText = True,
                                returnData = 1, returnCount = 1)

mainFig2, mainAx2, exportDf2, countDf2 = out2

plt.show()
