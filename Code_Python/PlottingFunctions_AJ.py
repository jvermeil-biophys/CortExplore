# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:51:36 2023

@author: anumi
"""
# %% > Imports and constants
#### Main imports

import random
import distinctipy
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm


import ptitprince as pt
from statannotations.Annotator import Annotator

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# from plotnine import (
#     ggplot,
#     aes,
#     stage,
#     geom_violin,
#     geom_point,
#     geom_line,
#     geom_boxplot,
#     geom_segment,
#     guides,
#     scale_fill_manual,
#     theme,
#     theme_classic,
#     facet_wrap,
#     xlim, ylim,
#     annotate,
#     ggtitle,
#     scale_color_gradient,
#     scale_fill_gradient, 
#     scale_color_gradientn,
#     scale_color_manual,
#     scale_fill_gradientn,
#     guide_colorbar
# )

# from plotnine.positions import position_jitter
# from plotnine.themes import element_rect, element_text, element_line
# from plotnine.scales import scale_y_log10, scale_x_log10

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
from scipy.stats import mannwhitneyu, wilcoxon, ranksums, ttest_ind, ttest_rel

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)

import GraphicStyles as gs
import UtilityFunctions as ufun
import TrackAnalyser_V3_Manuscript_AJ as taka


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
plt.style.use('seaborn-v0_8')

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

plt.rcParams.update({
    'font.family': 'Arial',   # Choose your font
})

SCALE_px_cm = 2.60

#%% Functions

def median_normalize_by_first_six(df, y):
    # Select the first 5 compressions
    ref = df[df['new_compNum'].between(-6, 0)][y].median()
    # Normalize all H0 values by this reference
    df[y + '_norm'] = df[y] / ref
    return df

def mean_normalize_by_first_six(df, y):
    # Select the first 5 compressions
    ref = df[df['new_compNum'].between(-6, 0)][y].mean()
    # Normalize all H0 values by this reference
    df[y + '_norm'] = df[y] / ref
    return df

def NLR_normalize_by_first_six(df, y):
    # Select the first 5 compressions
    ref = df[df['new_compNum'].between(-6, 0)][y].mean()
    # Normalize all H0 values by this reference
    df[y + '_norm'] = df[y] - ref
    return df


def normalize_by_first_five(df, y):
    # Select the first 5 compressions
    ref = df[df['compNum'].between(1, 5)][y].mean()
    # Normalize all H0 values by this reference
    df[y + '_norm'] = df[y] / ref
    return df

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


def NLIcorr(df):
    data_nli = df[['cellID', 'compNum', 'NLI_mod']]
    for i in np.unique(data_nli['cellID'].values):
       
        diff = []
        dataCell = data_nli[data_nli.cellID == i]
        for j in range(1, dataCell.compNum.max()):
            if j in dataCell.compNum.values and j+1 in dataCell.compNum.values:
                diffNLI = dataCell.NLI_mod[dataCell['compNum'] == j+1].values - dataCell.NLI_mod[dataCell['compNum'] == j].values
                diff.extend(diffNLI)
                
            else:
                diff.append(np.nan)
        
        diff = np.abs(diff).copy()
        df.loc[dataCell.index, 'NLI_corr'] = [np.nanmean((diff))]*len(dataCell)
    
    return df

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

def filterDf(Filters, data):
    globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    for k in range(0, len(Filters)):
        globalFilter = globalFilter & Filters[k]
    data_f = data[globalFilter]
    return data_f

def createAvgDf(data, condCol, dataFluoPath = None, dataAnglesPath = None, e_norm = False):
    
    try:
        data['Chadwick_%f_15_H0_log'] = np.log10((data['Chadwick_%f_15_H0'].values))
    except:
        pass
    
    group_by_cell = data.groupby(['cellID'])
    agg_dict = {'H0_vwc_Full':['var', 'std', 'mean', 'count', 'median'], 
                'bestH0_log':['var', 'std', 'mean', 'count', 'median'],
                'bestH0':['var', 'std', 'mean', 'count', 'median'],
                'NLI_corr':'first',
                'NLI_Ind':['var', 'std', 'mean', 'count'],
                'E_eff':['var', 'std', 'mean', 'count', 'median'],
                'E_eff_log':['var', 'std', 'mean', 'count', 'median'], 
                'surroundingDz':['median'],
                'surroundingDx':['median'],
                'surroundingThickness':['median'],
                'cellID' : 'first',
                'cellName':'first', 
                'NLI_Plot' : 'first', 
                'dateCell':'first',
                'ctFieldFluctuAmpli' : 'first', 
                'ctFieldThickness' : 'first', 
                'normFluctu' : 'first',
                'NLI_mod':['mean', 'count', 'median', 'std', 'var'], 
                'date' : 'first', 
                condCol:'first', 'cellCode':'first', 'manip':'first'}
    
    if dataFluoPath != None:
        added_cols = {'mean_intensity' : 'first',
                    'mean_subtracted' : 'first'}
        agg_dict.update(added_cols)
        
        
    if e_norm == True:
        added_cols = {'E_norm' : ['mean', 'count', 'std', 'var']}
        agg_dict.update(added_cols)
        
    if dataAnglesPath != None:
        added_cols = {'angle_beads' : ['first', 'mean'],
                      'angle_theta' : ['first', 'mean']}
        agg_dict.update(added_cols)
    
    try:
        added_cols = {'E_f_<_400' : ['mean'],
                      'E_f_<_400_log' : ['mean']}
        agg_dict.update(added_cols)
    except:
        pass
    
    try:
        added_cols = {'Chadwick_%f_15_H0' : ['mean'],
                      'Chadwick_%f_15_H0_log' : ['mean']}
        agg_dict.update(added_cols)
    except:
        pass
        
    avgDf = group_by_cell.agg(agg_dict)

    avgDf_wE = group_by_cell.agg({'compNum' : ['count'], 'wE_eff_log':[ 'sum'], 
                                 'weights_E_eff': ['sum']})
    
    avgDf_wE[('E_eff_log', 'wAvg')] = avgDf_wE['wE_eff_log', 'sum'] / avgDf_wE['weights_E_eff', 'sum']
    
    avgDf = avgDf.join(avgDf_wE)
    
    return avgDf.copy()


def dfCellPairs(avgDf):
    dfCleaned = avgDf.drop_duplicates(subset = ('cellID', 'first'))
    group_by_cell = dfCleaned.groupby([('dateCell', 'first')])
    dfGrouped = group_by_cell.agg({('dateCell', 'first') : ['first', 'count']})

    pairedCells = dfGrouped[('dateCell', 'first', 'first')][dfGrouped[('dateCell', 'first', 'count')] == 2].values
    dfPairs = avgDf[(avgDf[('dateCell', 'first')].apply(lambda x : x in pairedCells))]
    return dfPairs.copy(), pairedCells

def createDataTable(GlobalTable, fitsSubDir = None, dataFluoPath = None,
                    dataActPath = None, dataAnglesPath = None, dataBlebPath = None):

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
    data_main['E_eff_log'] = np.log10(GlobalTable['E_eff'].values)
    data_main['weights_E_eff_log'] = np.log10(data_main['E_eff']) / np.log10(data_main['ciwE_eff'])**2
    data_main['wE_eff_log'] =  data_main['E_eff_log'] * data_main['weights_E_eff_log']
    
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
    
    data_nli = data_main[['cellID', 'compNum', 'NLI_mod']]
    
    try:
        data_main['E_f_<_400_log'] = np.log10((GlobalTable['E_f_<_400'].values))
    except:
        pass
    
    if fitsSubDir != None:
        df_h0 = taka.getMatchingFits(data_main, fitsSubDir = fitsSubDir, fitType = 'H0')
        h0_method = ['Chadwick', '%f_15']
        df_h0 = df_h0[(df_h0['method'] == h0_method[0]) & (df_h0['zone'] == h0_method[1])]
        data_main = pd.merge(data_main, df_h0[['cellID', 'compNum', 'H0', 'nbPts']], on = ['cellID', 'compNum'], how = 'left')
        data_main = data_main.rename(columns={'H0':'{:}_{:}_H0'.format(h0_method[0], h0_method[1])})
    
    
    # for i in data_nli['cellID'].values:
    #     diff = []
    #     dataCell = data_nli[data_nli.cellID == i]
    #     for j in range(1, dataCell.compNum.max()):
    #         if j in dataCell.compNum.values and j+1 in dataCell.compNum.values:
    #             diffNLI = dataCell.NLI_mod[dataCell['compNum'] == j+1].values - dataCell.NLI_mod[dataCell['compNum'] == j].values
    #             diff.extend(diffNLI)
    #             print(i)
    #         else:
                
    #             diff.extend(np.nan)
        
    #     diff = np.ma.array(diff, mask=np.isnan(diff), fill_value=np.nan)
    #     # data_nli.loc[dataCell.index, 'NLI_corr'] = [np.sqrt(np.mean(diff**2) / len(dataCell))]*len(dataCell)
    #     data_main.loc[dataCell.index, 'NLI_corr'] = [np.mean(np.absolute(diff))]*len(dataCell)
    
    
    if dataFluoPath != None:
        dataFluo = pd.read_csv(dataFluoPath, sep = ';')
        data_main = pd.merge(data_main, dataFluo, on=['dateCell'], how='left')
        
    if dataActPath != None: 
        dataAct = pd.read_csv(dataActPath, sep = ';')
        for i in dataAct.cellID: 
            dataCell = data_main.loc[data_main['cellID'] == i, 'activation type']
            data_main.loc[data_main['cellID'] == i, 'activation type']  = list(dataAct.loc[dataAct['cellID'] == i, 'activation type'])*len(dataCell)
        
    data_main['activation type'] = data_main['activation type'].fillna('no light')
    data_main['activation frequency'] = data_main['activation frequency'].fillna(0)
    data_main['first activation'] = data_main['first activation'].fillna(-1)
    
    # if dataAnglesPath != None:
    #     allFilesAngles = os.listdir(dataAnglesPath)
    #     allFilesAngles = [i for i in allFilesAngles if '_ComputedAngles.csv' in i]
    #     angleFile_compiled = []
    #     for i in allFilesAngles:
    #         angleFilename = os.path.join(dataAnglesPath, i)
    #         angleFile = pd.read_csv(angleFilename, sep = ';')
    #         angleFile_compiled.append(angleFile)
        
    #     finalAnglesFile = pd.concat(angleFile_compiled, ignore_index=True)
    #     finalAnglesFile = finalAnglesFile.drop(columns=['cellID'])
    #     data_main = pd.merge(data_main, finalAnglesFile, on=['dateCell'], how='left')
        
    if dataAnglesPath != None:
        allFilesAngles = os.listdir(dataAnglesPath)
        allFilesAngles = [i for i in allFilesAngles if '_ComputedAngles.csv' in i]
        angleFile_compiled = []
        for i in allFilesAngles:
            angleFilename = os.path.join(dataAnglesPath, i)
            angleFile = pd.read_csv(angleFilename, sep = ';')
            angleFile_compiled.append(angleFile)
        
        finalAnglesFile = pd.concat(angleFile_compiled, ignore_index=True)
        finalAnglesFile = finalAnglesFile.drop(columns=['dateCell'])
        data_main = pd.merge(data_main, finalAnglesFile, on=['cellID', 'compNum'], how='left')
    
    if dataBlebPath != None:
        dataBleb = pd.read_csv(dataBlebPath, sep = ';')
        data_main['blebStatus'] = 0  # Initializing with 0
        
        # Loop through each row and update new_column based on compNum threshold
        for index, row in dataBleb.iterrows():
            cell_id = row['cellID']
            comp_num = row['compNum']
            

            data_main.loc[(data_main['cellID'] == cell_id) & (data_main['compNum'] >= comp_num), 'blebStatus'] = 1
        

        
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


def plotNLImod(fig, ax, data, palette = sns.color_palette("tab10"), plot = 'line', colorScheme = 'black'):
    
    if colorScheme == 'black':
        plt.style.use('default')
        fig.patch.set_facecolor('black')
        fontColor = '#ffffff'
    else: 
        plt.style.use('default')
        fontColor = '#000000'
        
    if plot == 'line':
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

# def NLIcorrvFluctu(fig, ax, data, palette = sns.color_palette("tab10"),
#                 colorScheme = 'black', plottingParams = {}, plotChars = {}):
    
#     if colorScheme == 'black':
#         plt.style.use('default')
#         fig.patch.set_facecolor('black')
#         fontColor = '#ffffff'
#     else: 
#         plt.style.use('default')
#         fontColor = '#000000'
        
#     data_nli = data[['cellID', 'compNum', 'NLI_mod']]

#     for i in data_nli['cellID'].values:
#         diff = []
#         dataCell = data_nli[data_nli.cellID == i]
#         for j in range(1, dataCell.compNum.max()):
#             if j in dataCell.compNum.values and j+1 in dataCell.compNum.values:
#                 diffNLI = dataCell.NLI_mod[dataCell['compNum'] == j+1].values - dataCell.NLI_mod[dataCell['compNum'] == j].values
#                 diff.extend(diffNLI)
#             else:
#                 diff.append(np.nan)
                
#         diff = np.ma.array(diff, mask=np.isnan(diff), fill_value=None)
#         # data_nli.loc[dataCell.index, 'NLI_corr'] = [np.sqrt(np.mean(diff**2) / len(dataCell))]*len(dataCell)
#         data_nli.loc[dataCell.index, 'NLI_corr'] = [np.mean(np.absolute(diff))]*len(dataCell)


#     data['NLI_corr'] = data_nli['NLI_corr']
#     ax = sns.scatterplot(data = data, x = 'normFluctu', y = 'NLI_corr', palette = palette, **plottingParams)
    

#     plt.xticks(**plotChars)
#     plt.yticks(**plotChars)
#     ax.set_ylabel('NLI_correlation', **plotChars)
#     ax.set_xlabel('Activity', **plotChars)

#     return fig, ax, data.copy()

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
     
def EvsH0_perCompression(fig, ax, data, condCat, condCol, hueType, xlim = (100, 2*10**3),
          palette = sns.color_palette("tab10"), h_ref = 600, colorScheme = 'black'):
    
    
    if colorScheme == 'black':
        plt.style.use('default')
        fig.patch.set_facecolor('black')
        fontColor = '#ffffff'
    else: 
        # plt.style.use('default')
        plt.style.use('seaborn-v0_8')
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
            # eqnText += " Y = {:.1e} * X^{:.1f}".format(k, a)


            eqnText += "\np-val = {:.3f}".format(pval)
            ax.scatter(x , y, color = palette[idx], label = m, s = 100)
            ax.plot(fit_x, fit_y, label =  eqnText, linewidth = 5,
                    color = 'k')
            ax.plot(fit_x, fit_y, label =  eqnText, linewidth = 2.5,
                    color = palette[idx])
            
            
            idx = idx + 1
            
        plt.text(0.5, 0.9, eqnText, fontsize=12, ha='center', fontweight='bold', transform=plt.gca().transAxes)
        plt.legend(fontsize = 12, ncol = len(mechanicsType))
        
    elif hueType == condCol:
        palette = palette
        for m in condCat:
            eqnText = ''
            toPlot = data[(data[condCol] == m)]
            # toPlot = toPlot.dropna(subset=['H0_vwc_Full', 'E_eff'])
            # x, y = toPlot['H0_vwc_Full'].values, toPlot['E_eff'].values
            
            toPlot = toPlot.dropna(subset=['bestH0_log', 'E_eff_log'])
            x, y = toPlot['bestH0_log'].values, toPlot['E_eff_log'].values
            
            # toPlot = toPlot.dropna(subset=['Chadwick_%f_15_H0_log', 'E_f_<_400_log'])
            # x, y = toPlot['Chadwick_%f_15_H0_log'].values, toPlot['E_f_<_400_log'].values
            
            
            params, results = ufun.fitLineHuber(x,y)

            k = (params[0])
            a = params[1]
            
            fit_x = np.linspace(np.min(x), np.max(x), 50)
            # fit_y = k * fit_x**a
            fit_y = a * fit_x + k
            
            fit_y = 10**fit_y
            fit_x = 10**fit_x
            
            pval = results.pvalues[1] # pvalue on the param 'a'
            eqnText += " Fit y = m * x + c\n".format(a, k)
            eqnText += " y = {:.1e} * x + {:.1f}".format(a, k)

            # eqnText += " Y = {:.1e} * X^{:.1f}".format(k, a)
            eqnText += "\np-val = {:.3f}".format(pval)
            
            # ax.plot(fit_x, fit_y, label =  eqnText, linewidth = 5,
            #         color = 'k')
            # ax.plot(fit_x, fit_y, label =  eqnText, linewidth = 2.5,
            #         color = palette[idx])
            
            toPlot['E_norm'] = toPlot['E_eff'] * (h_ref/toPlot['bestH0'])**a
            
            data.loc[(data[condCol] == m), 'E_norm'] = toPlot['E_norm']

            ax.scatter(10**x , 10**y, color = palette[idx], label = m, s =30, alpha = 0.5)
            ax.plot(fit_x, fit_y,  linewidth = 3,
                    color = 'k')
            ax.plot(fit_x, fit_y, label =  eqnText, linewidth = 1.5,
                    color = palette[idx])
            idx = idx + 1
            
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize = 11)
            
        # plt.legend(fontsize = 12, ncol = len(condCat))

    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.set_ylim(ylim)
    # ax.set_xlim(xlim)
    # ax.set_ylabel('E_effective (Pa)', fontsize=30, color = fontColor)
    # ax.set_xlabel('BestH0 (nm)', fontsize=30, color = fontColor)
    # plt.xticks(fontsize=25, color = fontColor)
    # plt.yticks(fontsize=25, color = fontColor)
    
    y_labels = np.asarray([100, 500, 1000, 3000, 10000, 25000])
    # y_ticks = np.log10(y_labels)
    ax.set_yticks(y_labels, labels = (y_labels)/1000)
    
    x_ticks = [100, 250, 500, 1000, 1500]
    ax.set_xticks(x_ticks, labels =x_ticks)


    # plt.show()
    return fig, ax, data.copy()

def pairedplot_wfluo(dfPairs, condCol, condCat, measure, stat, pairs, test='two-sided', y_limits = None,
               figsize=(7, 6), palette=sns.color_palette("tab10"), plotChars={}, plotTicks={}):
    
    lsize = 0.65
    fill_alpha = 0.7
    
    x, y = (condCol, 'first'), (measure, stat)
    dfPairsPlot = dfPairs[[x, y, ('dateCell', 'first'), ('mean_subtracted', 'first')]]  # Include the third column for color gradient
    dfPairsPlot.columns = [x[0], y[0], 'dateCell', 'mean_subtracted']  # Rename columns to use in ggplot

    dfPairsPlot[condCol] = pd.Categorical(dfPairsPlot[condCol], categories=condCat, ordered=True)
    
    pvals = []
    for pair in pairs:
        a1 = dfPairsPlot[y[0]][dfPairsPlot[condCol] == pair[1]].values
        b1 = dfPairsPlot[y[0]][dfPairsPlot[condCol] == pair[0]].values
        res = wilcoxon(a1, b1, alternative=test, zero_method='wilcox')
        pvals.append(res[1])
    
    shift = 0.1

    def alt_sign(x):
        return (-1) ** x

    m1 = aes(x=stage(condCol, after_scale="x+shift*alt_sign(x)"))  # shift outward
    m2 = aes(x=stage(condCol, after_scale="x-shift*alt_sign(x)"), group="dateCell")  # shift inward

    # Create the plot
    plot = (
        ggplot(dfPairsPlot, aes(x[0], y[0], fill='mean_subtracted', color='mean_subtracted'))  # Map both fill and color to third_col
        + geom_violin(m1, style="left-right", alpha=fill_alpha, size=lsize)
        + geom_point(m2, alpha=fill_alpha, size=6, show_legend=False)  # Apply fill gradient to points
        + geom_line(m2, color="gray", size=lsize)
        + geom_boxplot(width=shift, alpha=fill_alpha, size=lsize)
        + scale_fill_gradientn(
            colors=plt.cm.jet(np.linspace(0, 1, 256)),  # Jet color map
            limits=(dfPairsPlot['mean_subtracted'].min(), dfPairsPlot['mean_subtracted'].max()),  # Tighten the color range
            breaks=np.linspace(dfPairsPlot['mean_subtracted'].min(), dfPairsPlot['mean_subtracted'].max(), 5)  # Control the tick marks
        )
        + scale_color_gradientn(
            colors=plt.cm.jet(np.linspace(0, 1, 256)),  # Jet color map
            limits=(dfPairsPlot['mean_subtracted'].min(), dfPairsPlot['mean_subtracted'].max()),  # Tighten the color range
            breaks=np.linspace(dfPairsPlot['mean_subtracted'].min(), dfPairsPlot['mean_subtracted'].max(), 5)  # Control the tick marks
        )

        + theme_classic()
        + theme(figure_size=figsize)
        + theme(
            plot_background=element_rect(fill='black'),  # Set plot background to black
            axis_text=element_text(color='white'),
            legend_position=(0.2, 0.902),  # Adjust the position of the color scale
           
        )
    )
    
    if y_limits:
        plot += ylim(y_limits[0], y_limits[1])
        
    plot.draw()
    plt.style.use('seaborn-v0_8')
    plt.title('                   p = ' + str(np.round(pvals, 4)) + ' | ' + test, **plotChars)

    plt.ylabel(measure, **plotChars)
    plt.xlabel(condCol, **plotChars)

    return plot, pvals


def plotnine_jitter(avgDf, condCol, condCat, measure, stat, pairs, palette, pointSize = 3,
                       figsize=(17/SCALE_px_cm, 10/SCALE_px_cm), y_limits=None, logScale = False,
                      plotChars={}, plotTicks={}):
    
    lsize = 0.65
    fill_alpha = 0.8
    
    # Prepare the data frame for plotting
    x, y = (condCol, 'first'), (measure, stat)
    
    avgDf_plot = avgDf[[x, y]]
    avgDf_plot.columns = [x[0], y[0]]
    # print(avgDf_plot)
    avgDf_plot[condCol] = pd.Categorical(avgDf_plot[condCol], categories=condCat, ordered=True)
    # avgDf_plot = avgDf_plot.dropna(subset=[measure, condCol, condCat])
    
    # Calculate median for each group
    df_median = avgDf_plot.groupby(condCol)[measure].median().reset_index()

    # Add x positions for the median lines
    df_median['x'] = pd.factorize(df_median[condCol])[0] + 1
    df_median['xend'] = df_median['x'] + .25  # Adjusted to shorten the median line length
    df_median['x'] = df_median['x'] - .25  # Adjusted to shorten the median line length

    # Set shift for jittering
    shift = 0.3
    
    def rgb_to_hex(rgb):
        # Convert RGB to hex
        rgb = [int(c * 255) for c in rgb]
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def alt_sign(x):
        # Helper function to alternate signs
        return (-1) ** x
    
    # Convert palette to hex
    palette_hex = [rgb_to_hex(rgb) for rgb in palette]
    
    avgDf_plot['fill_color'] = avgDf_plot[condCol].map(dict(zip(condCat, palette_hex)))
    
    
    if pairs != None:
        p_values = {}
        for (group1, group2) in pairs:
            group1_data = avgDf_plot[avgDf_plot[condCol] == group1][measure]
            group2_data = avgDf_plot[avgDf_plot[condCol] == group2][measure]
            t_stat, p_value = st.mannwhitneyu(group1_data, group2_data)
            p_values[(group1, group2)] = p_value

    
    # Build the plot
    plot = (
        ggplot(avgDf_plot, aes(x=condCol, y=measure, fill=condCol, color = condCol))
        + geom_point(position=position_jitter(width=0.10, height=0.0), color='#000000', alpha=fill_alpha, size=pointSize)  # Decreased jitter
        + geom_boxplot(width=shift, alpha=0, size=lsize, color="black", fill="none")  # Hollow boxplot: color for lines and no fill
        + scale_fill_manual(values=dict(zip(condCat, palette_hex))) 
        + geom_segment(
            mapping=aes(x="x", xend="xend", y=measure, yend=measure),  # Corrected to use 'measure' as y-axis
            data=df_median, size=1, color='red')  # Directly setting color to red
        + guides(fill=False)
        + theme_classic()
        + theme(figure_size=figsize)
        + theme(
            plot_background=element_rect(fill='white'),  # Set plot background to white
            axis_text=element_text(color='black'),
            axis_ticks=element_line(color='black'),
            # panel_grid_major_y=element_line(color='lightgray', size=0.5),  # Light gray horizontal gridlines
            # panel_grid_minor_y=element_line(color='lightgray', size=0.5),  # Light gray minor gridlines
            panel_grid_major_y=element_line(color='lightgrey', size=0.5),
        )
    )


    # Add y limits if provided
    if y_limits != None:
        plot += ylim(y_limits[0], y_limits[1])
        
    if logScale and measure == 'bestH0_log':
        plot += theme(panel_grid_major_y=element_line(color='lightgrey', size=0.5))
        ticks = np.asarray([10, 100, 250, 500, 1000, 1200])
        plot += scale_y_log10(breaks=np.log10(ticks), labels = [str(i) for i in ticks])  # Set log scale breaks (e.g., 10, 100, 1000)
    elif logScale and measure == 'E_eff_log' or measure == 'E_f_<_400_log':
        plot += theme(panel_grid_major_y=element_line(color='lightgrey', size=0.5))
        ticks = np.asarray([500, 1000, 2500, 5000, 10000, 20000, 30000])
        plot += scale_y_log10(breaks=np.log10(ticks), labels = [str(i/1000) for i in ticks])  # Set log scale breaks (e.g., 10, 100, 1000)
        
    
    if pairs != None:
        spacing_offset = 0.15  # Vertical space between different p-value lines

        # Add p-value annotations and bars across the compared groups
        max_y = avgDf_plot[measure].max()  # Get the maximum y value for positioning
    
        for i, ((group1, group2), p_value) in enumerate(p_values.items()):
            condCatlist = list(condCat)
            # Get the x positions of the groups being compared
            x1 = condCatlist.index(group1) + 1
            x2 = condCatlist.index(group2) + 1
    
            # Calculate dynamic y position for each comparison, based on index
            y_position = max_y + spacing_offset * (i + 1)
    
            # Add a bar between the groups at the dynamic y position
            plot += geom_segment(
                aes(x=x1, xend=x2, y=y_position, yend=y_position),
                color="black", size=0.2
            )
    
            # Annotate the p-value above the bar at the dynamic y position
            plot += annotate('text', x=(x1 + x2) / 2, y=y_position + 0.05,
                             label=f'p = {p_value:.3f}', color='black', size=10, ha='center')


    # Title and labels
    plot.draw()
    # plt.title('p = ' + str(np.round(pvals, 4)) + ' | ' + test, **plotChars)
    plt.tight_layout()
    plt.ylabel(measure + ', ' + stat, **plotChars)
    plt.xlabel(condCol, **plotChars)

    return plot, df_median


def norm_pairedplot(dfPairs, condCol, condCat, measure, stat, pairs, test = 'two-sided',
               figsize = (12/SCALE_px_cm,10/SCALE_px_cm), y_limits = None, logScale = False,
               palette = sns.color_palette("tab10"), plotChars = {}, plotTicks = {}):
    

    lsize = 0.65
    fill_alpha = 0.7
    

    x, y = (condCol, 'first'), (measure, stat)
    dfPairsPlot = dfPairs[[x, y, ('dateCell', 'first')]]
    dfPairsPlot.columns = [x[0], y[0], 'dateCell']
    

    x, y = x[0], y[0]
    if 'NLI' in y:
        dfPairsPlot[y] = (dfPairsPlot[y].values)
    elif 'log' in y:
        dfPairsPlot[y] = 10**(dfPairsPlot[y].values)
        
    dfPairsPlot[condCol] = pd.Categorical(dfPairsPlot[condCol], categories=condCat, ordered=True)
    
    print(x)
    
    shift = 0.2
    
    def rgb_to_hex(rgb):
        # Ensure RGB values are in the range [0, 1]
        rgb = [int(c * 255) for c in rgb]
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def alt_sign(x):
        return (-1) ** x
    
    
    palette_hex = [rgb_to_hex(rgb) for rgb in palette]
    m1 = aes(x=stage(condCol, after_scale="x+shift*alt_sign(x)"))  # shift outward
    m2 = aes(x=stage(condCol, after_scale="x-shift*alt_sign(x)"), group="dateCell")  # shift inward
    
    dfPairsPlot[('normMeasure')] = np.nan

    # Extract the paired cells from the 'first' column in the MultiIndex
    pairedCells = dfPairsPlot[('dateCell')].unique().to_numpy()  # Using unique() to avoid duplicate processing

    # Loop over pairs and the cells
    
    for pair in pairs:
        for cell in pairedCells:
           if y != 'NLI_mod': 
                # Get the measurement for the first condition (pair[0]) for the given cell
                c1 = dfPairsPlot[y][(dfPairsPlot[('dateCell')] == cell) & (dfPairsPlot[condCol] == pair[0])].values
                
                # Get the measurement for the second condition (pair[1]) for the given cell
                c2 = dfPairsPlot[y][(dfPairsPlot[('dateCell')] == cell) & (dfPairsPlot[condCol] == pair[1])].values
                
                # Avoid division by zero
                if len(c1) > 0 and len(c2) > 0:
                    ratio = np.round(c2 / c1, 3)
                    # Update the 'normMeasure' column with the ratio for the second condition (pair[1])
                    dfPairsPlot.loc[(dfPairsPlot[('dateCell')] == cell) & (dfPairsPlot[condCol] == pair[0]), ('normMeasure')] = 1
                    dfPairsPlot.loc[(dfPairsPlot[('dateCell')] == cell) & (dfPairsPlot[condCol] == pair[1]), ('normMeasure')] = ratio
                    
                
           elif 'NLI' in y:
                
                # Get the measurement for the first condition (pair[0]) for the given cell
                c1 = dfPairsPlot[y][(dfPairsPlot[('dateCell')] == cell) & (dfPairsPlot[condCol] == pair[0])].values
                # print(c1)
                # Get the measurement for the second condition (pair[1]) for the given cell
                c2 = dfPairsPlot[y][(dfPairsPlot[('dateCell')] == cell) & (dfPairsPlot[condCol] == pair[1])].values
                
                # Avoid division by zero
                if len(c1) > 0 and len(c2) > 0:
                    new_c1 = c1 - c1
                    norm = c2 - c1
                    # Update the 'normMeasure' column with the ratio for the second condition (pair[1])
                    dfPairsPlot.loc[(dfPairsPlot[('dateCell')] == cell) & (dfPairsPlot[condCol] == pair[0]), ('normMeasure')] = new_c1
                    dfPairsPlot.loc[(dfPairsPlot[('dateCell')] == cell) & (dfPairsPlot[condCol] == pair[1]), ('normMeasure')] = norm
    
    plot = (
    ggplot(dfPairsPlot, aes(x, 'normMeasure', fill=condCol))
    # + geom_violin(m1, style="left-right", alpha=fill_alpha, size=lsize)
    + geom_point(m2, color="none", alpha=fill_alpha, size=4)
    + geom_line(m2, color="gray", size=lsize, alpha=0.6)
    + geom_boxplot(width=shift, alpha=fill_alpha, size=lsize)    
    + scale_fill_manual(values=palette_hex)
    + guides(fill=False)  
    + theme_classic()
    + theme(figure_size=figsize)
    + theme(
        plot_background=element_rect(fill='white'),  # Set plot background to black
        axis_text=element_text(color='black'),
        axis_ticks=element_line(color='black'), 
        panel_grid_major_y=element_line(color='lightgrey', size=0.5),
    )
    )
    
    if y_limits:
        plot += ylim(y_limits[0], y_limits[1])
    
    
    # if logScale and measure == 'bestH0_log' or  measure == 'Chadwick_%f_15_H0_log':
    #     plot += theme(panel_grid_major_y=element_line(color='lightgrey', size=0.5))
    #     ticks  =  np.asarray([100 ,250, 500, 1000, 1500])


    #     plot += scale_y_log10(breaks=np.log10(ticks), labels = [str(i) for i in ticks])  # Set log scale breaks (e.g., 10, 100, 1000)
    # elif logScale and measure == 'E_eff_log' or  measure == 'E_f_<_400_log':
    #     plot += theme(panel_grid_major_y=element_line(color='lightgrey', size=0.5))
    #     ticks  = [100, 500, 2000, 5000, 10000,25000, 50000]

    #     plot += scale_y_log10(breaks=np.log10(ticks), labels = [str(i/1000) for i in ticks])  #
        
    plot.draw()
    # plt.style.use('seaborn-v0_8')
    # plt.title('p = ' + str(np.round(pvals, 4)) + ' | ' + test , **plotChars)
    plt.tight_layout()

    plt.ylabel(measure + ', ' + stat, **plotChars)
    plt.xlabel(condCol, **plotChars)

    return plot, dfPairsPlot

def pairedplot(dfPairs, condCol, condCat, measure, stat, pairs, test = 'two-sided',
               figsize = (12/SCALE_px_cm,10/SCALE_px_cm), y_limits = None, logScale = False,
               palette = sns.color_palette("tab10"), plotChars = {}, plotTicks = {}):
    

    lsize = 0.65
    fill_alpha = 0.7
    
    # if measure == 'E_eff_log':
    #     dfPairs[('E_eff_log', stat)] = 10**(dfPairs[('E_eff_log', stat)].values)
    
    x, y = (condCol, 'first'), (measure, stat)
    dfPairsPlot = dfPairs[[x, y, ('dateCell', 'first')]]
    dfPairsPlot.columns = [x[0], y[0], 'dateCell']

    dfPairsPlot[condCol] = pd.Categorical(dfPairsPlot[condCol], categories=condCat, ordered=True)
    
    
    # pvals = []
    # for pair in pairs:
    #     a1 = dfPairsPlot[y[0]][dfPairsPlot[condCol] == pair[0]].values
    #     b1 = dfPairsPlot[y[0]][dfPairsPlot[condCol] == pair[1]].values
    #     res = wilcoxon(b1, a1, alternative=test, zero_method = 'wilcox')
    #     pvals.append(res[1])
        
    shift = 0.1
    
    def rgb_to_hex(rgb):
        # Ensure RGB values are in the range [0, 1]
        rgb = [int(c * 255) for c in rgb]
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def alt_sign(x):
        return (-1) ** x
    
    
    palette_hex = [rgb_to_hex(rgb) for rgb in palette]
    m1 = aes(x=stage(condCol, after_scale="x+shift*alt_sign(x)"))  # shift outward
    m2 = aes(x=stage(condCol, after_scale="x-shift*alt_sign(x)"), group="dateCell")  # shift inward
    
    if pairs != None:
        p_values = {}
        for (group1, group2) in pairs:
            group1_data = dfPairsPlot[dfPairsPlot[condCol] == group1][measure]
            group2_data = dfPairsPlot[dfPairsPlot[condCol] == group2][measure]
            t_stat, p_value = wilcoxon(group2_data, group1_data, alternative=test, zero_method = 'wilcox')
            p_values[(group1, group2)] = p_value

    plot = (
    ggplot(dfPairsPlot, aes(x[0], y[0], fill=condCol))
    + geom_violin(m1, style="left-right", alpha=fill_alpha, size=lsize)
    + geom_point(m2, color="none", alpha=fill_alpha, size=4)
    + geom_line(m2, color="gray", size=lsize, alpha=0.6)
    + geom_boxplot(width=shift, alpha=fill_alpha, size=lsize)
    + scale_fill_manual(values=palette_hex)
    + guides(fill=False)  
    + theme_classic()
    + theme(figure_size=figsize)
    + theme(
        plot_background=element_rect(fill='white'),  # Set plot background to black
        axis_text=element_text(color='black'),
        axis_ticks=element_line(color='black'), 
        panel_grid_major_y=element_line(color='lightgrey', size=0.5),
    )
    )
    
    if y_limits:
        plot += ylim(y_limits[0], y_limits[1])
    
    
    if pairs != None:
        spacing_offset = 0.15  # Vertical space between different p-value lines

        # Add p-value annotations and bars across the compared groups
        max_y = dfPairsPlot[measure].max()  # Get the maximum y value for positioning
    
        for i, ((group1, group2), p_value) in enumerate(p_values.items()):
            condCatlist = list(condCat)
            # Get the x positions of the groups being compared
            x1 = condCatlist.index(group1) + 1
            x2 = condCatlist.index(group2) + 1
    
            # Calculate dynamic y position for each comparison, based on index
            y_position = max_y + spacing_offset * (i + 1)
    
            # Add a bar between the groups at the dynamic y position
            plot += geom_segment(
                aes(x=x1, xend=x2, y=y_position, yend=y_position),
                color="black", size=0.2
            )
    
            # Annotate the p-value above the bar at the dynamic y position
            plot += annotate('text', x=(x1 + x2) / 2, y=y_position + 0.05,
                             label=f'p = {p_value:.3f}', color='black', size=15, ha='center')
    
    
    if logScale and measure == 'bestH0_log' or  measure == 'Chadwick_%f_15_H0_log':
        plot += theme(panel_grid_major_y=element_line(color='lightgrey', size=0.5))
        ticks  =  np.asarray([100 ,250, 500, 1000, 1500])


        plot += scale_y_log10(breaks=np.log10(ticks), labels = [str(i) for i in ticks])  # Set log scale breaks (e.g., 10, 100, 1000)
    elif logScale and measure == 'E_eff_log' or  measure == 'E_f_<_400_log':
        plot += theme(panel_grid_major_y=element_line(color='lightgrey', size=0.5))
        # plot += ylim(100, 50000)
        ticks  = [100, 500, 2000, 5000, 10000,25000, 50000]

        plot += scale_y_log10(breaks=np.log10(ticks), labels = [str(i/1000) for i in ticks])  #
        # plot += scale_y_log10()  #

        
    plot.draw()
    # plt.style.use('seaborn-v0_8')
    # plt.title('p = ' + str(np.round(pvals, 4)) + ' | ' + test , **plotChars)
    plt.tight_layout()

    plt.ylabel(measure + ', ' + stat, **plotChars)
    plt.xlabel(condCol, **plotChars)

    return plot

def pairedplot_woHisto(dfPairs, condCol, condCat, measure, stat, pairs, test = 'two-sided',
               figsize = (12/SCALE_px_cm,10/SCALE_px_cm), y_limits = None, logScale = False,
               palette = sns.color_palette("tab10"), plotChars = {}, plotTicks = {}):
    

    lsize = 0.65
    fill_alpha = 0.7
    
    # if measure == 'E_eff_log':
    #     dfPairs[('E_eff_log', stat)] = 10**(dfPairs[('E_eff_log', stat)].values)
    
    x, y = (condCol, 'first'), (measure, stat)
    dfPairsPlot = dfPairs[[x, y, ('dateCell', 'first')]]
    dfPairsPlot.columns = [x[0], y[0], 'dateCell']

    dfPairsPlot[condCol] = pd.Categorical(dfPairsPlot[condCol], categories=condCat, ordered=True)
    
    # pvals = []
    # for pair in pairs:
    #     a1 = dfPairsPlot[y[0]][dfPairsPlot[condCol] == pair[0]].values
    #     b1 = dfPairsPlot[y[0]][dfPairsPlot[condCol] == pair[1]].values
    #     res = wilcoxon(b1, a1, alternative=test, zero_method = 'wilcox')
    #     pvals.append(res[1])
        
    shift = 0.2
    
    def rgb_to_hex(rgb):
        # Ensure RGB values are in the range [0, 1]
        rgb = [int(c * 255) for c in rgb]
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def alt_sign(x):
        return (-1) ** x
    
    palette_hex = [rgb_to_hex(rgb) for rgb in palette]
    m1 = aes(x=stage(condCol, after_scale="x+shift*alt_sign(x)"))  # shift outward
    m2 = aes(x=stage(condCol, after_scale="x-shift*alt_sign(x)"), group="dateCell")  # shift inward
    
    if pairs != None:
        p_values = {}
        for (group1, group2) in pairs:
            group1_data = dfPairsPlot[dfPairsPlot[condCol] == group1][measure]
            group2_data = dfPairsPlot[dfPairsPlot[condCol] == group2][measure]
            t_stat, p_value = wilcoxon(group2_data, group1_data, alternative=test, zero_method = 'wilcox')
            p_values[(group1, group2)] = p_value

    plot = (
    ggplot(dfPairsPlot, aes(x[0], y[0], fill=condCol))
    # + geom_violin(m1, style="left-right", alpha=fill_alpha, size=lsize)
    + geom_point(m2, color="none", alpha=fill_alpha, size=4)
    + geom_line(m2, color="gray", size=lsize, alpha=0.6)
    + geom_boxplot(width=shift, alpha=fill_alpha, size=lsize)
    + scale_fill_manual(values=palette_hex)
    + guides(fill=False)  
    + theme_classic()
    + theme(figure_size=figsize)
    + theme(
        plot_background=element_rect(fill='white'),  # Set plot background to black
        axis_text=element_text(color='black'),
        axis_ticks=element_line(color='black'), 
        panel_grid_major_y=element_line(color='lightgrey', size=0.5),
    )
    )
    
    if y_limits:
        plot += ylim(y_limits[0], y_limits[1])
    
    
    if pairs != None:
        spacing_offset = 0.15  # Vertical space between different p-value lines

        # Add p-value annotations and bars across the compared groups
        max_y = dfPairsPlot[measure].max()  # Get the maximum y value for positioning
    
        for i, ((group1, group2), p_value) in enumerate(p_values.items()):
            condCatlist = list(condCat)
            # Get the x positions of the groups being compared
            x1 = condCatlist.index(group1) + 1
            x2 = condCatlist.index(group2) + 1
    
            # Calculate dynamic y position for each comparison, based on index
            y_position = max_y + spacing_offset * (i + 1)
    
            # Add a bar between the groups at the dynamic y position
            plot += geom_segment(
                aes(x=x1, xend=x2, y=y_position, yend=y_position),
                color="black", size=0.2
            )
    
            # Annotate the p-value above the bar at the dynamic y position
            plot += annotate('text', x=(x1 + x2) / 2, y=y_position + 0.05,
                             label=f'p = {p_value:.3f}', color='black', size=15, ha='center')
    
    
    if logScale and measure == 'bestH0_log' or  measure == 'Chadwick_%f_15_H0_log':
        plot += theme(panel_grid_major_y=element_line(color='lightgrey', size=0.5))
        ticks  =  np.asarray([100 ,250, 500, 1000, 1500])


        plot += scale_y_log10(breaks=np.log10(ticks), labels = [str(i) for i in ticks])  # Set log scale breaks (e.g., 10, 100, 1000)
    elif logScale and measure == 'E_eff_log' or  measure == 'E_f_<_400_log':
        plot += theme(panel_grid_major_y=element_line(color='lightgrey', size=0.5))
        # plot += ylim(100, 50000)
        ticks  = [100, 500, 2000, 5000, 10000,25000, 50000]

        plot += scale_y_log10(breaks=np.log10(ticks), labels = [str(i/1000) for i in ticks])  #
        # plot += scale_y_log10()  #

        
    plot.draw()
    # plt.style.use('seaborn-v0_8')
    # plt.title('p = ' + str(np.round(pvals, 4)) + ' | ' + test , **plotChars)
    plt.tight_layout()

    plt.ylabel(measure + ', ' + stat, **plotChars)
    plt.xlabel(condCol, **plotChars)

    return plot

def rainplot(fig, ax,  condCat, palette = sns.color_palette("tab10"), labels = [], pairs = None, 
             colorScheme = 'black', test = 'non-param', pointSize = 2,
             shiftBox = 0.1, shiftSwarm = 0.0,
              plottingParams = {}, plotTicks = {}, plotChars = {}):
    
    
    if colorScheme == 'black':
        plt.style.use('seaborn-v0_8')
        fig.patch.set_facecolor('black')
        fontColor = plotChars['color']
    else: 
        plt.style.use('seaborn-v0_8')
        fontColor = plotChars['color']
        
    pt.half_violinplot(inner = None, palette = palette, **plottingParams)
    
    condCol = plottingParams['x']
    df =  plottingParams['data']
    measure = plottingParams['y']

    for i, condition in enumerate(condCat):
        # Subset the data
        plotDf = df[df[condCol] == condition]
        
        # Jitter the values on the vertical axis
        x = i - shiftSwarm + np.random.uniform(high=0.2, size=len(plotDf))
        
        # Select the values of the horizontal axis
        y = plotDf[measure]
        
        # Add the rain using the scatter method.
        ax.scatter(x, y, color = palette[i], s=pointSize, linewidth=0.5,
                   edgecolor = 'k', alpha = 0.6)

    boxplot_data = [df[df[condCol] == condition][measure].values 
        for condition in condCat]

    # Vertical positions for the boxplots
    POSITIONS = [shiftBox + pos for pos in range(len(condCat))]
    medianprops = {"linewidth": 2, "color": "#FF0000", "solid_capstyle": "butt"}
    # The style of the box ... This is also used for the whiskers
    boxprops = {"linewidth": 1, "color": "#4a4a4a"}
    boxplot = ax.boxplot(
        boxplot_data, 
        positions=POSITIONS, 
        manage_ticks=False,
        showfliers = False, # Do not show the outliers beyond the caps.
        showcaps = False,   # Do not show the caps
        medianprops = medianprops,
        whiskerprops = boxprops,
        boxprops = boxprops
    )
    
    medians = [median.get_ydata()[0] for median in boxplot['medians']]
    
    pvals = []
    if pairs != None:
        for pair in pairs:
            a1 = df[measure][df[condCol] == pair[0]].values
            b1 = df[measure][df[condCol] == pair[1]].values
            if test == 'non-param':
                U1, p = mannwhitneyu(a1, b1, nan_policy = 'omit')
            elif test == 'param':
                U1, p = ttest_ind(a1, b1, nan_policy = 'omit')
                
            pvals.append(p)
    
        annotator = Annotator(ax = ax, pairs = pairs, x=condCol,  y=measure, data=df, order = condCat)
        annotator.configure(text_format="simple", color = '#000000', fontsize = plotChars['fontsize'])
        annotator.set_pvalues(pvals).annotate(line_offset_to_group=0.15)
        
    if labels != []:
        xticks = np.arange(len(condCat))
        plt.xticks(xticks - 0.03, labels, **plotTicks)
        
    # plt.xticks(**plotTicks)
    plt.yticks(**plotTicks)
    plt.ylabel(measure, **plotChars)
    plt.xlabel(condCol, **plotChars)
    
    return fig, ax, medians
    
    
def NLR_distplot(condCat, palette = sns.color_palette("tab10"), 
                           pairs = None, colorScheme = 'black', test = 'non-param',
                           plottingParams = {}, plotChars = {}):
    
    if colorScheme == 'black':
        plt.style.use('dark_background')
        fontColor = plotChars['color']
    else: 
        plt.style.use('default')
        fontColor = plotChars['color']
    
    measure = plottingParams['x']
    
    sns.histplot(palette = palette, kde = True, **plottingParams)

    plt.ylabel(measure, **plotChars)
    # plt.xlabel(**plotChars)
    
    return 

def boxplot_perCompressionLog(fig, ax, condCat, hueType = None, palette = sns.color_palette("tab10"), labels = [],
                           pairs = None, colorScheme = 'black', plotType = 'swarm', test = 'non-param',
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
            ax = sns.violinplot(hue = hueType, palette = palette, **plottingParams) 
            
        plt.legend(loc=2, prop={'size': 7}, ncol = int(np.round(N/2)))
    
    elif hueType == 'NLI_Plot':
        palette = ['#CB799C', '#d6c9bc', '#79CBA8']
        if plotType == 'swarm':
            ax = sns.swarmplot(hue = hueType, palette = palette, **plottingParams) 
            ax = sns.boxplot(data = df, x = condCol, y = measure, color = 'grey',  order = condCat,
                             medianprops={"color": 'darkred', "linewidth": 2},
                             boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.1})
            
        elif plotType == 'violin':
            plt.grid(color = '#bcbcbc')
            ax = sns.violinplot(hue = hueType, palette = palette,  **plottingParams) 
            
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
            if test == 'non-param':
                U1, p = mannwhitneyu(a1, b1, nan_policy = 'omit')
            elif test == 'param':
                U1, p = ttest_ind(a1, b1, nan_policy = 'omit')
                
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
                           pairs = None, colorScheme = 'black', plotType = 'swarm', test = 'non-param',
                           plottingParams = {}, plotChars = {}):
    
    if colorScheme == 'black':
        # plt.style.use('default')
        fig.patch.set_facecolor('black')
        fontColor = plotChars['color']
    else: 
        plt.style.use('seaborn-v0_8-darkgrid')
        fontColor = '#000000'
    
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
            ax = sns.violinplot(hue = hueType, palette = palette, alpha = 0.2,
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
        
        # plt.legend(loc=2, prop={'size': 16}, ncol = len(condCat))
        
    
    
    pvals = []
    if pairs != None:
        for pair in pairs:
            a1 = df[measure][df[condCol] == pair[0]].values
            b1 = df[measure][df[condCol] == pair[1]].values
            if test == 'non-param':
                U1, p = mannwhitneyu(a1, b1, nan_policy = 'omit')
            elif test == 'param':
                U1, p = ttest_ind(a1, b1, nan_policy = 'omit')
            pvals.append(p)
    
        annotator = Annotator(ax = ax, pairs = pairs, x=condCol,  y=measure, data=df, order = condCat)
        annotator.configure(text_format="simple", color = '#000000', fontsize = plotChars['fontsize'])
        annotator.set_pvalues(pvals).annotate()
        
    if labels != []:
        xticks = np.arange(len(condCat))
        plt.xticks(xticks, labels, **plotChars)
        
    # plt.xticks(**plotChars)
    # plt.yticks(**plotChars)
    plt.ylabel(measure, **plotChars)
    plt.xlabel(condCol, **plotChars)
    
    return fig, ax, pvals

def boxplot_perCell(fig, ax, condCat, hueType = None, palette = sns.color_palette("tab10"), 
                    labels = [], pairs = None,  colorScheme = 'black', test = 'non-param',
                    plottingParams = {}, plotChars = {}):
    
    if colorScheme == 'black':
        plt.style.use('seaborn-v0_8')
        fig.patch.set_facecolor('black')
        fontColor = plotChars['color']
    else: 
        plt.style.use('seaborn-v0_8-darkgrid')
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
        
    ax = sns.boxplot(data = avgDf, x = condCol, y = measure, palette = palette,
                      order = condCat, 
                     medianprops={"color": '#FF0000', "linewidth": 2},
                     boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha':0.2})
    
    
    medians = []
    pvals = []
    if pairs != None:
        for pair in pairs:
            a1 = avgDf[measure][avgDf[condCol] == pair[0]].values
            b1 = avgDf[measure][avgDf[condCol] == pair[1]].values
            medians.append(np.asarray([np.median(a1), np.median(b1)]))
            if test == 'non-param':
                U1, p = mannwhitneyu(a1, b1, nan_policy = 'omit')
            elif test == 'param':
                U1, p = ttest_ind(a1, b1, nan_policy = 'omit')
            pvals.append(p)
    
        annotator = Annotator(ax = ax, pairs = pairs, x=condCol,  y=measure, data=avgDf, order = condCat)
        annotator.configure(text_format="simple", color = '#000000', fontsize = plotChars['fontsize'])
        annotator.set_pvalues(pvals).annotate()
        
    # annotator = Annotator(ax = ax, pairs = pairs, x=condCol,  y=measure, data=avgDf, order = condCat)
    # annotator.configure(test='Mann-Whitney', text_format='simple', fontsize = plotChars['fontsize'],
    #                     loc='inside', show_test_name = False, color = '#000000')
    # annotator.apply_and_annotate()
    
    if labels != []:
        xticks = np.arange(len(condCat))
        plt.xticks(xticks, labels, **plotChars)
        
    plt.xticks(**plotChars)
    plt.legend(fontsize = 16)
    plt.yticks(**plotChars)
    plt.ylabel(measure, **plotChars)
    plt.xlabel(condCol, **plotChars)
    
    return fig, ax, medians

def EvH0_LogCellAvg(fig, ax, data, condCat, condCol, hueType = None, h_ref = 400,
                  palette = sns.color_palette("tab10"), plotChars = {},
                  errorbar = False, pairs = None, colorScheme = 'black'):
    
    if colorScheme == 'black':
        plt.style.use('default')
        fig.patch.set_facecolor('black')
        fontColor = '#ffffff'
    else: 
        plt.style.use('seaborn-v0_8')
        fontColor = '#000000'
    
    # h =( 'Chadwick_%f_15_H0_log', 'mean')
    # e = ('E_f_<_400_log', 'mean')
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
            toPlot = data[(data[('NLI_Plot', 'first')] == m)]
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
        
        # plt.legend(fontsize = 10, ncol = len(mechanicsType))
        
    elif hueType == 'cellID':
        N = len(data[('cellID', 'first')].unique())
        # palette = (sns.color_palette("Paired", np.round(N/2)) + sns.color_palette("husl",  np.round(N/2)))
        sns.scatterplot(ax = ax[0], data = data, x = h, y = e, hue = ('cellName', 'first'), 
                          s = 150, palette = palette, edgecolor = 'k')
        
        for m in condCat:
            eqnText = ''
            toPlot = data[(data[(condCol, 'first')] == m)]
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

            
        # plt.legend(fontsize = 10, ncol = np.round(N/2))
          
    else:
        palette = palette
        # sns.scatterplot(ax = ax[0], data = avgDf, x = h, y = e, hue = (condCol, 'first'), 
        #                s = 150, palette = palette, edgecolor = 'k')
        
        for m in condCat:
            eqnText = ''
            toPlot = data[(data[(condCol, 'first')] == m)]

            x, y = toPlot[h].values, toPlot[e].values
            
            params, results = ufun.fitLineHuber((x), (y))
            k = params[0]
            t = params[1]
            
            fit_x = np.linspace(np.min(x), np.max(x), 50)
            # fit_y = k * fit_x**t
            fit_y = t*fit_x + k
            
            fit_y = 10**fit_y
            fit_x = 10**fit_x

            pval = results.pvalues[1] # pvalue on the param 'a'
            # eqnText += " Y = {:.1e} * X^{:.1f}".format(k, t)
            eqnText += " Fit Y = m * X + C\n"
            eqnText += " Y = {:.1e} * X + {:.1f}".format(t, k)

            eqnText += "\np-val = {:.3f}".format(pval)
            ax.scatter(10**x , 10**y, color = palette[idx], label = m, s = 50, alpha = 0.7)
            # ax.plot(fit_x, fit_y,  lw = 6, linestyle = '--', color = 'k')

            # ax.plot(fit_x, fit_y, label = eqnText, lw = 6.1, linestyle = '--', color = palette[idx])
            ax.plot(fit_x, fit_y, linewidth = 3,
                    color = 'k')
            ax.plot(fit_x, fit_y, label =  eqnText, linewidth = 1.5,
                    color = palette[idx])

            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize = 11)
            
            toPlot[('E_norm', 'logAvg')] = toPlot[e] * (h_ref/toPlot[h])**t

            
            # Xfit, Yfit = np.log(toPlot[('bestH0_log', 'mean')].values), np.log(toPlot[('E_norm', 'logAvg')].values)
            data.loc[(data[(condCol, 'first')] == m), ('E_norm', 'logAvg')] = toPlot[('E_norm', 'logAvg')]
            
            
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
        
       
    # for i in range(len(ax)):
    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax[i].set_ylim(100, 10**5)
    # ax[i].set_xlim(100, 2*10**3)
    # ax.set_ylabel('E_effective (Pa)', fontsize=30, color = fontColor)
    # ax.set_xlabel('BestH0 (nm)', fontsize=30, color = fontColor)
    

    x_labels = np.asarray([100, 250, 500, 1000, 1500])
    x_ticks = np.log10(np.asarray(x_labels))
    ax.set_xticks(x_labels, labels = x_labels,**plotChars)

    y_labels =np.asarray([100, 500, 1000, 3000, 10000, 50000])

    y_ticks = np.log10(np.asarray(y_labels))
    ax.set_yticks(y_labels, labels = (y_labels)/1000,**plotChars)
    

    # plt.xticks(fontsize=25, color = fontColor)
    # plt.yticks(fontsize=25, color = fontColor)
    # plt.tight_layout()
    # plt.show()
    return fig, ax, data.copy()



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
            # eqnText += " Y = {:.1e} * X^{:.1f}".format(k, t)
            eqnText += " Y = {:.1e} * X + {:.1f}".format(k, t)

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
            res = wilcoxon(b1, a1, alternative=test, zero_method = 'wilcox')
            pvals.append(res[1])

        # ax = sns.lineplot(palette = palette, data = dfPairs, hue = hueType, style = styleType,
        #                   marker = 'o', **plottingParams)
        
        ax = sns.pointplot(palette = palette, data = dfPairs, hue = hueType, style = styleType,
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

            
            pvals.append(np.nan)
        
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

def NLRvAngle(fig, ax, dfPairs, condCat, condCol, pairedCells, palette = sns.color_palette("tab10"),
                pairs = None, colorScheme = 'black', plotType = False,
                plottingParams = {}, plotChars = {}):
    
    if colorScheme == 'black':

        fig.patch.set_facecolor('black')
        fontColor = '#ffffff'
        
    else: 
        plt.style.use('default')
        fontColor = '#000000'
        
    measure =  plottingParams['y']
    try:
        condCol =  plottingParams['hue']
    except:
        pass
        
    x = plottingParams['x']
    palette = plottingParams['palette']
    
    if plotType == False:
        sns.scatterplot(**plottingParams)
    
    elif plotType == 'delta':
        dfPairs['NLI_mod', 'diff'] = [np.nan]*len(dfPairs)
        for pair in pairs:
            for cell in pairedCells:
                c1 = dfPairs[(measure)][(dfPairs[('dateCell', 'first')] == cell) & (dfPairs[condCol] == pair[0])].values
                c2 = dfPairs[(measure)][(dfPairs[('dateCell', 'first')] == cell) & (dfPairs[condCol] == pair[1])].values
                diff = c2 - c1
                
                dfPairs.loc[(dfPairs[('dateCell', 'first')] == cell) & (dfPairs[condCol] == pair[0]), ('NLI_mod', 'diff')] = diff
                dfPairs.loc[(dfPairs[('dateCell', 'first')] == cell) & (dfPairs[condCol] == pair[1]), ('NLI_mod', 'diff')] = diff
        

        sns.scatterplot(data = dfPairs, x = x, y = ('NLI_mod', 'diff'))
    # elif plotType == 'paired':
    #     idx = 0
    #     for i in np.unique(dfPairs['dateCell', 'first'].values):
    #         toPlot = dfPairs[dfPairs['dateCell', 'first'] == i]
    #         x1 = toPlot['normFluctu', 'first'][toPlot[condCol, 'first'] == condCat[0]]
    #         y1 = toPlot['NLI_mod', 'mean'][toPlot[condCol, 'first'] == condCat[0]]
            
    #         x2 = toPlot['normFluctu', 'first'][toPlot[condCol, 'first'] == condCat[1]]
    #         y2 = toPlot['NLI_mod', 'mean'][toPlot[condCol, 'first'] == condCat[1]]

    #         ax.scatter(x1, y1, marker = 'o', color = palette[idx], s = 100)
    #         ax.scatter(x2, y2, marker = '*',  color = palette[idx], s = 100)
    #         ax.plot([x1, x2], [y1, y2], color = palette[idx]) 
            
    #         idx = idx + 1
            
    plt.xticks(**plotChars)
    plt.yticks(**plotChars)
    # plt.ylabel(measure, **plotChars)
    plt.xlabel(condCol, **plotChars)
    plt.show()

    return fig, ax
    
    
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


def plotPopKS(data_f, styleDict, fitsSubDir = '',  fitType = 'stressGaussian', 
              fitWidth=75,  condCol = '', 
              c_min = 0, c_max = np.Inf, legendLabels = [],
              mode = 'wholeCurve', scale = 'lin', printText = True,
              returnData = 0, returnCount = 0):
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, ax = plt.subplots(1,1, figsize = (15/SCALE_px_cm,10/SCALE_px_cm))

    # globalFilter = pd.Series(np.ones(data.shape[0], dtype = bool))
    # for k in range(0, len(Filters)):
    #     globalFilter = globalFilter & Filters[k]
    # data_f = data[globalFilter]
    
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
        
        df = df[df['compCount']  > 25]
        
        
        color = styleDict[co]['color']
        marker = styleDict[co]['marker']
        label = styleDict[co]['label']
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
        legendTxt = label + '\nNCells = {}\nNComp = {}'.format(cellCount, sum(N))
        
        
        # label = '{} | NCells = {}'.format(legendLabels[i], cellCount)

        # weighted means -- weighted ste 95% as error
        ax.errorbar(centers, Kavg/1000, yerr = q*Kste/1000, 
                    color = color, lw = 2, marker = marker, markersize = 6, mec = 'k',
                    ecolor = color, elinewidth = 1.2, capsize = 5, capthick = 1.2, 
                    label = legendTxt)
        
        # ax.set_title('K(s) - All compressions pooled')
        color = '#000000'
        ax.legend( bbox_to_anchor=(1.02, 1), loc = 'upper left', fontsize = 9)
        ax.set_xlabel('Stress (Pa)', fontsize = 15,  color = color)
        ax.set_ylabel('K (kPa)', fontsize = 15, color = color)
        ax.tick_params(axis='both', colors= color) 
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
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


def plotCellKS(data_f, fitsSubDir = '', condCol = '', fitType = 'stressGaussian', chosenCells = None, 
              fitWidth=75,  c_min = 0, c_max = np.Inf,  mode = 'wholeCurve'):
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, ax = plt.subplots(1,1, figsize = (7,6))


    if mode == 'wholeCurve':
        Sinf, Ssup = 0, np.Inf
        ax.set_xlim([0, 1050])  
        
    else:
        bounds = mode.split('_')
        Sinf, Ssup = int(bounds[0]), int(bounds[1])
        extraFilters = [data_f['minStress'] <= Sinf, data_f['maxStress'] >= Ssup] # >= 800
    
    
    fitId = '_' + str(fitWidth)
    data_ff = taka.getFitsInTable(data_f, fitsSubDir, fitType=fitType, filter_fitID=fitId)
    
    # Filter the table
    data_ff = data_ff[(data_ff['fit_center'] >= Sinf) & (data_ff['fit_center'] <= Ssup)]    
    data_ff = data_ff.drop(data_ff[data_ff['fit_error'] == True].index)
    data_ff = data_ff.drop(data_ff[data_ff['fit_K'] < 0].index)
    data_ff = data_ff.dropna(subset = ['fit_ciwK'])
    data_ff = data_ff.drop(data_ff[data_ff['fit_ciwK'] > 10**4].index)
    
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
    
    cells = np.array(data_ff['cellID'].unique())
    
    data_ff['A'] = data_ff['fit_K'] * data_ff['weight']
    grouped1 = data_ff.groupby(by=['cellID', 'fit_center'])
    data_agg = grouped1.agg({'compNum' : 'count',
                            'A': 'sum', 'weight': 'sum'}).reset_index()
    data_agg['K_wAvg'] = data_agg['A']/data_agg['weight']
    data_agg = data_agg.rename(columns = {'compNum' : 'compCount'})
    
    # Compute the weighted std
    data_ff['B'] = data_ff['fit_K']    
    for co in cells:
        centers = np.array(data_ff[data_ff['cellID'] == co]['fit_center'].unique())
        centers = np.array([ce for ce in centers if ((ce<c_max) and (ce>c_min))])
        
        for ce in centers:
            weighted_mean_val = data_agg.loc[(data_agg['cellID'] == co) & (data_agg['fit_center'] == ce), 'K_wAvg'].values[0]

            index_loc = (data_ff['cellID'] == co) & (data_ff['fit_center'] == ce)
            col_loc = 'B'
            data_ff.loc[index_loc, col_loc] = data_ff.loc[index_loc, 'fit_K'] - weighted_mean_val
            data_ff.loc[index_loc, col_loc] = data_ff.loc[index_loc, col_loc] ** 2
            
    data_ff['C'] = data_ff['B'] * data_ff['weight']
    grouped2 = data_ff.groupby(by=['cellID', 'fit_center'])
    data_agg2 = grouped2.agg({'compNum' : 'count',
                              'C': 'sum', 'weight': 'sum'}).reset_index()
    data_agg2['K_wVar'] = data_agg2['C']/data_agg2['weight']
    data_agg2['K_wStd'] = data_agg2['K_wVar']**0.5
    
    
    # Combine all in data_agg
    data_agg['K_wVar'] = data_agg2['K_wVar']
    data_agg['K_wStd'] = data_agg2['K_wStd']
    data_agg['K_wSte'] = data_agg['K_wStd'] / data_agg['compCount']**0.5

    
    available_cells = data_ff['cellID'].unique()
    cells = [c for c in cells if c in available_cells]
    colors = distinctipy.get_colors(len(cells))
    
    # Get unique cell types
    cell_types = data_ff.set_index('cellID').loc[cells][condCol]
    unique_cell_types = cell_types.unique()
    
    # Generate a color for each cell type
    type_colors = dict(zip(unique_cell_types, distinctipy.get_colors(len(unique_cell_types))))
    
    for i in range(len(cells)):
        co = cells[i]
        df = data_agg[data_agg['cellID'] == co]
        
        centers = df['fit_center'].values
        Kavg = df['K_wAvg'].values
        Kste = df['K_wSte'].values
        N = df['compCount'].values
          
        dof = N
        alpha = 0.975
        q = st.t.ppf(alpha, dof) 
        
        if condCol == '':
            color = colors[i]
        else:
            cell_type = cell_types.loc[co]
            if isinstance(cell_type, pd.Series):
                cell_type = cell_type.iloc[0]
            color = type_colors[cell_type]
            
        label = co

        ax.errorbar(centers,  Kavg/1000, yerr = q*Kste/1000, label = label,
                    color = color, lw = 1, ls = 'solid', marker = 'o', markersize = 8, mec = 'k',
                    ecolor = color, elinewidth = 0.8, capsize = 4, capthick = 0.8)
        
        # ax.set_title('K(s) - All compressions pooled')
        color = '#000000'
        ax.legend(loc = 'upper left', fontsize = 9)
        ax.set_xlabel('Stress (Pa)', fontsize = 20,  color = color)
        ax.set_ylabel('K (kPa)', fontsize = 20, color = color)
        ax.tick_params(axis='both', colors= color) 
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.grid(visible=True, which='major', axis='y') #, color = '#3a3b3b')
        
      
    return (fig, ax)