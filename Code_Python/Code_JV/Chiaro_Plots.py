# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:38:26 2024

@author: JosephVermeil
"""

# %% > Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import re
import datetime

from scipy.optimize import curve_fit
from scipy import signal
from cycler import cycler

import CellShapeSolver as css
import UtilityFunctions as ufun
import GraphicStyles as gs

gs.set_manuscript_options_jv()

figDir = "D:/MagneticPincherData/Figures/NanoIndentFigs"

# %% > Utility subfunctions

def stringSeries2catArray(S):
    unique_vals = S.unique()
    d = {unique_vals[k]:k for k in range(len(unique_vals))}
    A = np.array([d[x] for x in S.values])
    return(A)


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
                }
    for col in idCols:
        agg_dict[col] = 'first'
    for col in numCols:
        agg_dict[col] = aggFun
    
    all_cols = list(agg_dict.keys())
    if not groupCol in all_cols:
        all_cols.append(groupCol)
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

# %% Import Data


# %%% 24-02-26 & 24-04-11

dates = ['24-02-26', '24-04-11'] # '24-04-18'
Excluded = []
Excluded2 = ['24-02-26_M2_P1_C3-1']

chiaroDir = "D://MagneticPincherData//Data_Analysis"
df_chiaro = [pd.read_csv(os.path.join(chiaroDir, f'ChiaroData_{d}_syn.csv'), sep=',') for d in dates]
df_chiaro = pd.concat(df_chiaro)
df_chiaro = df_chiaro[df_chiaro['date'].apply(lambda x : x in dates).values]
df_chiaro['cellName'] = df_chiaro['cellID'].apply(lambda x : '_'.join(x.split('_')[1:])).values
df_chiaro['cellNum'] = df_chiaro['cellID'].apply(lambda x : x.split('_')[-1]).values

Filters = np.array([(df_chiaro['date'].apply(lambda x : x in dates)).values,
                    (df_chiaro['cellID'].apply(lambda x : x not in Excluded)).values,
                    (df_chiaro['valid_UI'] == True).values,
                    (df_chiaro['V_comp'] == 1).values,
                    # (df_chiaro['manipID'].apply(lambda x : x not in ['24-04-11_M2', '24-04-11_M5'])).values,
                    # (df_chiaro['manipID'].apply(lambda x : x in ['24-04-11_M1', '24-04-11_M2', '24-04-11_M3'])).values,
                    (df_chiaro['manipID'].apply(lambda x : x in ['24-02-26_M2', '24-02-26_M3',
                                                                 '24-04-11_M1', '24-04-11_M3', '24-04-11_M5',
                                                                 '24-04-18_M1', '24-04-18_M3'])).values,
                    # (df_chiaro['manipID'].apply(lambda x : x not in ['24-04-11_M2'])).values,
                    ])

totalFilter = np.all(Filters, axis = 0)
df_chiaro_f = df_chiaro[totalFilter]


pincherDir = "D://MagneticPincherData//Data_Analysis"
df_pincher = pd.read_csv(os.path.join(pincherDir, 'MecaData_NanoIndent_2023-2024_V2.csv'), sep=';')

df_pincher['cellCode'] = df_pincher['cellName'].apply(lambda x : x.split('_')[-1])
df_pincher['cellName'] = df_pincher['cellName'].apply(lambda x : x.split('-')[0])
df_pincher['cellID'] = df_pincher['date'] + '_' + df_pincher['cellName']
df_pincher.insert(4, 'cellLongCode', df_pincher['cellID'].apply(lambda x : '_'.join(x.split('_')[:-1])) + '_' + df_pincher['cellCode'])

_stiff = '_f_<_400'
df_pincher['valid' + _stiff] = (df_pincher['E' + _stiff] < 50e3) & (df_pincher['R2' + _stiff] > 0.5) & (df_pincher['Chi2' + _stiff] < 2.5)

Filters = np.array([df_pincher['date'].apply(lambda x : x in dates).values,
                    (df_pincher['cellID'].apply(lambda x : x not in Excluded)).values,
                    (df_pincher['cellLongCode'].apply(lambda x : x not in Excluded2)).values,
                    (df_pincher['validatedThickness'] == True).values,
                    (df_pincher['bestH0'] <= 1000).values,
                    (df_pincher['valid' + _stiff] == True).values,
                    ])

totalFilter = np.all(Filters, axis = 0)
df_pincher_f = df_pincher[totalFilter]

df_pincher_f = computeNLMetrics_V2(df_pincher_f, th_NLI = np.log10(2), ref_strain = 0.2)

df_pincherLIN_f = df_pincher_f[df_pincher_f['NLI_Plot'] == 'linear']

# %%% Merge

Filters = np.array([(df_chiaro_f['lastIndent'] == True).values,
                    ])

totalFilter = np.all(Filters, axis = 0)
df_chiaro_g = df_chiaro_f[totalFilter]

g = df_pincher_f.groupby('cellID')
df_pincher_g = g.agg({'bestH0':'mean', 'E' + _stiff :'mean', 'compNum': 'count'}).reset_index()

df_merged = df_chiaro_g.merge(df_pincher_g, on='cellID')

df_merged['E_eq'] = ((df_merged['Ka']*1e-3)/(df_merged['bestH0']*1e-9))
df_merged['E_eq_kPa'] = df_merged['E_eq'] * 1e-3
df_merged['T_eq'] = ((df_merged['E' + _stiff])*(df_merged['bestH0']*1e-9)) * 1e3
df_merged['E' + _stiff + '_kPa'] = df_merged['E' + _stiff] * 1e-3

# %%% Merge LIN

Filters = np.array([(df_chiaro_f['lastIndent'] == True).values,
                    ])

totalFilter = np.all(Filters, axis = 0)
df_chiaro_g = df_chiaro_f[totalFilter]

g = df_pincherLIN_f.groupby('cellID')
df_pincherLIN_g = g.agg({'bestH0':'mean', 'E' + _stiff :'mean', 'compNum': 'count'}).reset_index()

df_mergedLIN = df_chiaro_g.merge(df_pincherLIN_g, on='cellID')

df_mergedLIN['E_eq'] = ((df_mergedLIN['Ka']*1e-3)/(df_mergedLIN['bestH0']*1e-9))
df_mergedLIN['E_eq_kPa'] = df_mergedLIN['E_eq'] * 1e-3
df_mergedLIN['T_eq'] = ((df_mergedLIN['E' + _stiff])*(df_mergedLIN['bestH0']*1e-9)) * 1e3
df_mergedLIN['E' + _stiff + '_kPa'] = df_mergedLIN['E' + _stiff] * 1e-3

# %% Plots

# swarmplot_parameters = {'data':    data,
#                         'x':       condition,
#                         'y':       parameter,
#                         'order':   co_order,
#                         'palette': palette,
#                         'size'    : markersize, 
#                         'edgecolor'    : 'k', 
#                         'linewidth'    : 0.75*markersizeFactor
#                         }


# sns.swarmplot(ax=ax, **swarmplot_parameters)

# #### Stats    
# if stats:
#     if len(box_pairs) == 0:
#         box_pairs = makeBoxPairs(co_order)
#     addStat_lib(ax, box_pairs, test = statMethod, verbose = statVerbose, **swarmplot_parameters)


# #### Boxplot
# if boxplot>0:
#     boxplot_parameters = {'data':    data,
#                             'x':       condition,
#                             'y':       parameter,
#                             'order':   co_order,
#                             'width' : 0.5,
#                             'showfliers': False,
#                             }
#     if boxplot==1:
#         boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 2},
#                                 boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
#                                 # boxprops={"color": color, "linewidth": 0.5},
#                                 whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2},
#                                 capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 2})
    
#     elif boxplot==2:
#         boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
#                                 boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
#                                 # boxprops={"color": color, "linewidth": 0.5},
#                                 whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2},
#                                 capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 2})
    
    
#     elif boxplot == 3:
#         boxplot_parameters.update(medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 4},
#                             boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
#                             # boxprops={"color": color, "linewidth": 0.5},
#                             whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
#                             capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4})

# %%% Plots manuscript

# %%%% Simple Kh

gs.set_manuscript_options_jv()

data = df_merged
Filters = np.array([(data['valid_hertz_Z0'] == True).values,
                    ])

data_f = filterDf(data, Filters)

data_fg = dataGroup(data_f, groupCol = 'cellID', idCols = [], numCols = ['Kh'], aggFun = 'mean')


#### K_h

fig, ax = plt.subplots(1, 1, figsize = (5/gs.cm_in, 6/gs.cm_in))
ax = ax
ax.set_ylim([0, 400])
# ax.set_yscale('log')
plot_parms = {'data': data_fg,
              'ax':   ax,
              'y':    'Kh'
              }
swarmplot_parms = {**plot_parms, 
                   # 'hue' : 'date',
                   # 'palette':'Set2',
                   'size'    : 4, 
                   'edgecolor'    : 'k', 
                   'linewidth'    : 0.5,
                   }

sns.swarmplot(**swarmplot_parms)

boxplot_parms = {**plot_parms, 'width' : 0.5, 'showfliers' : False}
boxplot_parms.update(medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2},
                    # boxprops={"color": color, "linewidth": 0.5},
                    whiskerprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2},
                    capprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2})


sns.boxplot(**boxplot_parms)

ax.set_ylabel('$K_H$ (Pa)')

ax.grid(axis='y')

fig.tight_layout()
plt.show()

# Save
figSubDir = ''
name = 'Simple_Kh'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 200,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% Kh vs. H0, E400 and E400*H0

gs.set_manuscript_options_jv()

data = df_merged
Filters = np.array([(data['valid_hertz_Z0'] == True).values,
                    ])

data_f = filterDf(data, Filters)

fig, axes = plt.subplots(1, 2, figsize = (12/gs.cm_in, 6/gs.cm_in), sharey=True)

#### K_h vs H0
ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')

sns.scatterplot(ax = ax, data = data_f, x='bestH0', y='Kh', zorder=4)

ax.set_xlabel('$H_0$ (nm)')
ax.set_ylabel('$K_H$ (Pa)')
ax.set_ylim([0, 400])
ax.set_xlim([0, 1100])
ax.grid(zorder=0)

#### K_h vs E400
ax = axes[1]
ax.set_xscale('log')
# ax.set_yscale('log')

sns.scatterplot(ax = ax, data = data_f, x='E_f_<_400_kPa', y='Kh', zorder=4)

ax.set_xlabel('$E_{400}$ (kPa)')
# ax.set_xlim([100, 1100])
ax.grid(zorder=0)

fig.tight_layout()
plt.show()

# Save
figSubDir = ''
name = 'Kh_vs_pincher'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 200,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% T0 and KA

data = df_chiaro_f
# data = df_merged

Filters = np.array([(data['Ka'] >= -3).values,
                    ])
data_f = filterDf(data, Filters)

fig, axes = plt.subplots(1, 2, figsize = (12/gs.cm_in, 6/gs.cm_in))

####

ax = axes[0]
# ax.set_ylim([0, 400])
# ax.set_yscale('log')
plot_parms = {'data': data_f,
              'ax':   ax,
              'y':    'T0'
              }
swarmplot_parms = {**plot_parms, 
                   # 'hue' : 'date',
                   # 'palette':'Set2',
                   'size'    : 5, 
                   'edgecolor'    : 'k', 
                   'linewidth'    : 0.5,
                   }

sns.swarmplot(**swarmplot_parms)

boxplot_parms = {**plot_parms, 'width' : 0.5, 'showfliers' : False}
boxplot_parms.update(medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2},
                    # boxprops={"color": color, "linewidth": 0.5},
                    whiskerprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2},
                    capprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2})

sns.boxplot(**boxplot_parms)

ax.set_ylabel('$T_0$ (mN/m)')
ax.grid(axis = 'y')

####


ax = axes[1]
plot_parms = {'data': data_f,
              'ax':   ax,
              'y':    'Ka'
              }
swarmplot_parms = {**plot_parms, 
                   # 'hue' : 'date',
                   # 'palette':'Set2',
                   'size'    : 5, 
                   'edgecolor'    : 'k', 
                   'linewidth'    : 0.5,
                   }

sns.swarmplot(**swarmplot_parms)

boxplot_parms = {**plot_parms, 'width' : 0.5, 'showfliers' : False}
boxplot_parms.update(medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 2},
                    boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2},
                    # boxprops={"color": color, "linewidth": 0.5},
                    whiskerprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2},
                    capprops={"color": 'k', "linewidth": 1.5, 'alpha' : 0.7, 'zorder' : 2})

sns.boxplot(**boxplot_parms)

ax.set_ylabel('$K_A$ (mN/m)')
ax.grid(axis = 'y')


fig.tight_layout()
plt.show()

# Save
figSubDir = ''
name = 'T0_and_KA'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 200,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%%% T0 vs. H0, E400 and E400*H0

gs.set_manuscript_options_jv()

data = df_merged
Filters = np.array([(data['valid_hertz_Z0'] == True).values,
                    ])

data_f = filterDf(data, Filters)

fig, axes = plt.subplots(1, 3, figsize = (16/gs.cm_in, 6/gs.cm_in), sharey=True)

#### T0 vs H0
ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')

sns.scatterplot(ax = ax, data = data_f, x='bestH0', y='T0', zorder=4)

ax.set_xlabel('$H_0$ (nm)')
ax.set_ylabel('$T_0$ (mN/m)')
# ax.set_ylim([0, 400])
# ax.set_xlim([0, 1100])
ax.grid(zorder=0)

#### T0 vs E400
ax = axes[1]
ax.set_xscale('log')
# ax.set_yscale('log')

sns.scatterplot(ax = ax, data = data_f, x='E_f_<_400_kPa', y='T0', zorder=4)

ax.set_xlabel('$E_{400}$ (kPa)')
ax.set_ylabel('$T_0$ (mN/m)')
# ax.set_ylim([0, 400])
# ax.set_xlim([100, 1100])
ax.grid(zorder=0)

#### T0 vs E400*H0
ax = axes[2]
ax.set_xscale('log')

sns.scatterplot(ax = ax, data = data_f, x='T_eq', y='T0', zorder=4)

ax.set_xlabel('$T_{eq}$ (mN/m)')
ax.set_ylabel('$T_0$ (mN/m)')
# ax.set_ylim([0, 400])
# ax.set_xlim([100, 1100])
ax.grid(zorder=0)

fig.tight_layout()
plt.show()

# Save
figSubDir = ''
name = 'T0_vs_Pincher'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 200,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%%% KA vs. H0, E400 and E400*H0

gs.set_manuscript_options_jv()

data = df_merged
Filters = np.array([(data['valid_hertz_Z0'] == True).values,
                    ])

data_f = filterDf(data, Filters)

fig, axes = plt.subplots(1, 3, figsize = (16/gs.cm_in, 6/gs.cm_in), sharey=True)

#### T0 vs H0
ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')

sns.scatterplot(ax = ax, data = data_f, x='bestH0', y='Ka', zorder=4)

ax.set_xlabel('$H_0$ (nm)')
ax.set_ylabel('$K_A$ (mN/m)')
# ax.set_ylim([0, 400])
# ax.set_xlim([0, 1100])
ax.grid(zorder=0)

#### T0 vs E400
ax = axes[1]
ax.set_xscale('log')
# ax.set_yscale('log')

sns.scatterplot(ax = ax, data = data_f, x='E_f_<_400_kPa', y='Ka', zorder=4)

ax.set_xlabel('$E_{400}$ (kPa)')
ax.set_ylabel('$K_A$ (mN/m)')
# ax.set_ylim([0, 400])
# ax.set_xlim([100, 1100])
ax.grid(zorder=0)

#### T0 vs E400*H0
ax = axes[2]
ax.set_xscale('log')

sns.scatterplot(ax = ax, data = data_f, x='T_eq', y='Ka', zorder=4)

ax.set_xlabel('$T_{eq}$ (mN/m)')
ax.set_ylabel('$K_A$ (mN/m)')
# ax.set_ylim([0, 400])
# ax.set_xlim([100, 1100])
ax.grid(zorder=0)

fig.tight_layout()
plt.show()

# Save
figSubDir = ''
name = 'KA_vs_Pincher'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 200,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%% KA vs. H0, E400 and E400*H0 --- sort by NLR

gs.set_manuscript_options_jv()

data = df_mergedLIN
Filters = np.array([(data['valid_hertz_Z0'] == True).values,
                    ])

data_f = filterDf(data, Filters)

fig, axes = plt.subplots(1, 3, figsize = (16/gs.cm_in, 6/gs.cm_in), sharey=True)

#### T0 vs H0
ax = axes[0]
# ax.set_xscale('log')
# ax.set_yscale('log')

sns.scatterplot(ax = ax, data = data_f, x='bestH0', y='Ka', zorder=4)

ax.set_xlabel('$H_0$ (nm)')
ax.set_ylabel('$K_A$ (mN/m)')
# ax.set_ylim([0, 400])
# ax.set_xlim([0, 1100])
ax.grid(zorder=0)

#### T0 vs E400
ax = axes[1]
ax.set_xscale('log')
# ax.set_yscale('log')

sns.scatterplot(ax = ax, data = data_f, x='E_f_<_400_kPa', y='Ka', zorder=4)

ax.set_xlabel('$E_{400}$ (kPa)')
ax.set_ylabel('$K_A$ (mN/m)')
# ax.set_ylim([0, 400])
# ax.set_xlim([100, 1100])
ax.grid(zorder=0)

#### T0 vs E400*H0
ax = axes[2]
ax.set_xscale('log')

sns.scatterplot(ax = ax, data = data_f, x='T_eq', y='Ka', zorder=4)

ax.set_xlabel('$T_{eq}$ (mN/m)')
ax.set_ylabel('$K_A$ (mN/m)')
# ax.set_ylim([0, 400])
# ax.set_xlim([100, 1100])
ax.grid(zorder=0)

fig.tight_layout()
plt.show()

# Save
figSubDir = ''
name = 'KA_vs_Pincher'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 200,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%%% Tension

data = df_chiaro_f
fig, ax = plt.subplots(1, 1, figsize = (10/gs.cm_in, 10/gs.cm_in))
ax=ax
sns.scatterplot(ax=ax, data=data, x='T0', y='T0_relax', s= 50, zorder=5, alpha = 0.6) #, hue='manipID', style='cellNum'
ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], 'k--', lw=1)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel('$T_{0,comp}$ (mN/m)')
ax.set_ylabel('$T_{0,relax}$ (mN/m)')
ax.set_title("$T_{0}$ from compression vs from relaxation")
ax.legend().set_visible(False)
ax.grid()

fig.tight_layout()
plt.show()

# Save
figSubDir = ''
name = 'T0_relax_vs_comp'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 200,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %%%% Kh supp
# %%%%% Combine the data from the nanoIndent & the Pincher

dataDir = "D://MagneticPincherData//Data_Analysis"

dates = ['23-12-07', '23-12-10', '24-02-26', '24-02-28'] #, '24-02-28'] # ['23-11-13', '23-11-15', '23-12-07', '23-12-10']

Excluded = ['23-11-15_M1_P1_C1', '23-12-07_M1_P1_C2', '24-02-26_M3_P1_C5', '24-02-28_M3_P1_C4']
Excluded2 = ['24-02-26_M2_P1_C3-1']

# %%%%% Format the pincher table

df_pincher = pd.read_csv(os.path.join(dataDir, 'MecaData_NanoIndent_2023-2024.csv'), sep=';')
dates_2sets = ['23-11-13', '23-11-15', '23-12-07', '23-12-10']
dates_1set = ['24-02-26', '24-02-28']

_stiff = '_f_<_400'

# 1. Format

index_2sets = df_pincher[df_pincher['date'].apply(lambda x : x in dates_2sets)].index
df_2sets = df_pincher.loc[index_2sets, :]

df_2sets.insert(5, 'compressionSet', df_2sets['cellName'].apply(lambda x : int(x.split('-')[0][-1])))
df_2sets['cellRawNum'] = df_2sets['cellName'].apply(lambda x : x.split('_')[-1])
df_2sets['cellName'] = df_2sets['cellName'].apply(lambda x : x.split('-')[0][:-1])
df_2sets['cellID'] = df_2sets['date'] + '_' + df_2sets['cellName']
df_2sets['cellCode'] = df_2sets['cellRawNum'].apply(lambda x : x[:2] + x[3:])
df_2sets.insert(4, 'cellLongCode', df_2sets['cellID'].apply(lambda x : '_'.join(x.split('_')[:-1])) + '_' + df_2sets['cellCode'])
df_2sets = df_2sets.drop(columns = ['cellRawNum'])



# 2. Filter
Excluded = ['23-11-15_M1_P1_C1', '23-12-07_M1_P1_C2', '24-02-26_M3_P1_C5', '24-02-28_M3_P1_C4']
Excluded2 = ['24-02-26_M2_P1_C3-1']


Filters1 = np.array([df_2sets['date'].apply(lambda x : x in dates_2sets).values,
                    (df_2sets['cellID'].apply(lambda x : x not in Excluded)).values,
                    (df_2sets['cellLongCode'].apply(lambda x : x not in Excluded2)).values,
                    (df_2sets['validatedThickness'] == True).values,
                    (df_2sets['bestH0'] <= 1000).values,
                    ])
totalFilter1 = np.all(Filters1, axis = 0)
df_2sets_f1 = df_2sets[totalFilter1]


Filters2 = np.array([df_2sets['date'].apply(lambda x : x in dates_2sets).values,
                    (df_2sets['cellID'].apply(lambda x : x not in Excluded)).values,
                    (df_2sets['cellLongCode'].apply(lambda x : x not in Excluded2)).values,
                    (df_2sets['valid_f_<_400'] == True).values,
                    (df_2sets['E_f_<_400'] < 1e5).values,
                    ])
totalFilter2 = np.all(Filters2, axis = 0)
df_2sets_f2 = df_2sets[totalFilter2]

# %%%%% Plot Before/after

gs.set_manuscript_options_jv(palette='Set1')
fig, axes = plt.subplots(1, 2, figsize = (12/gs.cm_in, 6/gs.cm_in))

#### 1.
parm = 'bestH0'
ax = axes[0]

# Dataset
data = df_2sets_f1
data, condCol = makeCompositeCol(data, cols=['cellID', 'compressionSet'])
cellIDs = data['cellID'].unique()
cellsWithOnlyOneSet = []
for cid in cellIDs:
    if len(data.loc[data['cellID']==cid, 'compressionSet'].unique()) == 1:
        cellsWithOnlyOneSet.append(cid)
Excluded3 = cellsWithOnlyOneSet

# Filter
Filters = [(data['cellID'].apply(lambda x : x not in Excluded3)).values,
           ]
data_f = filterDf(data, Filters)
data_g = dataGroup(data_f, groupCol = condCol, idCols = ['compressionSet'], numCols = ['bestH0'], aggFun = 'mean')
  
# Plot
ax.set_yscale('log')

nCells = len(data['cellID'].unique())
dictPlot = {'data' : data_g,
            'ax' : ax,
            'x' : 'compressionSet',
            'y' : parm,
            'hue' : 'manipID',
            'size'    : 5, 
            'edgecolor'    : 'k', 
            'linewidth'    : 0.5
            }
# sns.boxplot(**dictPlot, fliersize = 0, width=0.6)
sns.swarmplot(**dictPlot, dodge = False) # edgecolor='black', linewidth=1

CID = data_g['cellID'].unique()
for cid in CID:
    vals = data_g.loc[data_g['cellID'] == cid, parm].values
    ax.plot([0, 1], vals, ls='-', c='grey', zorder=2, lw=1)

ax.set_ylim([0, ax.get_ylim()[1]])
ax.get_legend().remove()
ax.set_xticklabels(['Before\nindent', 'After\nindent'])
ax.set_xlabel('')
ax.set_ylabel('$H_0$ (nm)')
ax.grid(axis = 'y')
ax.set_ylim([90, 1100])
# ax.legend(loc = 'upper right')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 6)


#### 2.
data = df_2sets_f2
data, condCol = makeCompositeCol(data, cols=['cellID', 'compressionSet'])
cellIDs = data['cellID'].unique()
cellsWithOnlyOneSet = []
for cid in cellIDs:
    if len(data.loc[data['cellID']==cid, 'compressionSet'].unique()) == 1:
        cellsWithOnlyOneSet.append(cid)
Excluded3 = cellsWithOnlyOneSet

# Filter
Filters = [(data['cellID'].apply(lambda x : x not in Excluded3)).values,
           ]

data_f = filterDf(data, Filters)
data_fgw2 = dataGroup_weightedAverage(data_f, groupCol = condCol, idCols = [condCol, 'compressionSet'], 
                                      valCol = 'E_f_<_400', weightCol = 'ciwE_f_<_400', weight_method = 'ciw^2')
data_fgw2['E_f_<_400_wAvg'] /= 1000


parm = 'E_f_<_400_wAvg'
ax = axes[1]

ax.set_yscale('log')

nCells = len(data['cellID'].unique())
dictPlot = {'data' : data_fgw2,
            'ax' : ax,
            'x' : 'compressionSet',
            'y' : parm,
            'hue' : 'manipID',
            'size'    : 5, 
            'edgecolor'    : 'k', 
            'linewidth'    : 0.5
            }
# sns.boxplot(**dictPlot, fliersize = 0, width=0.6)
sns.swarmplot(**dictPlot, dodge = False) # edgecolor='black', linewidth=1

CID = data_fgw2['cellID'].unique()
for cid in CID:
    vals = data_fgw2.loc[data_fgw2['cellID'] == cid, parm].values
    ax.plot([0, 1], vals, ls='-', c='grey', zorder=2, lw=1)



ax.set_ylim([0, ax.get_ylim()[1]])
ax.get_legend().remove()
ax.set_xticklabels(['Before\nindent', 'After\nindent'])
ax.set_xlabel('')
ax.set_ylabel('$E_{400}$ (kPa)')
ax.grid(axis = 'y')
# ax.set_ylim([90, 1100])
# ax.legend(loc = 'upper right')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 6)

plt.tight_layout()
plt.show()


# Save
figSubDir = ''
name = 'Pincher_befroe_after_indent'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 200,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')



# %%%%% K & relative time

fig, ax = plt.subplots(1,1, figsize = (12/gs.cm_in, 10/gs.cm_in))

gs.set_manuscript_options_jv('Set2')
custom_cycler = (cycler(color=gs.cL_Set21))

parm = 'K_d'

df = df_chiaro_f
CID = df['cellID'].values
df['countIndent'] = np.array([np.sum(CID == cid) for cid in CID])

Filters = np.array([(df['countIndent'] >= 2).values,
                    (df['valid_hertz_Z0'] == True).values,
                    ])
df = filterDf(df, Filters)
df = df.reset_index()

CIDU = df['cellID'].unique()
df['relative Kh'] = df['Kh']
for cid in CIDU:
    index_cid = df[df['cellID']==cid].index
    firstKh = df.loc[index_cid, 'relative Kh'].values[0]
    df.loc[index_cid, 'relative Kh'] = (df.loc[index_cid, 'relative Kh'])/(firstKh)
    
rel_times_sec_list = []
for cid in CIDU:
    temp_df = df[df['cellID'] == cid]
    times_sec = np.array([str2sec(S) for S in temp_df.t0])
    rel_times_sec = times_sec - np.min(times_sec)
    rel_times_sec_list.append(rel_times_sec)

df['relative time'] = np.concatenate(rel_times_sec_list)
    
ax = ax
ax.set_prop_cycle(custom_cycler)
for cid in CIDU:
    temp_df = df[df['cellID'] == cid]
    if not np.max(temp_df['relative time']) > 300:
        ax.plot(temp_df['relative time'], temp_df['relative Kh'], ls='-', marker='o', label = cid, ms=5, mec='k')

ax.set_xlabel('Time from first indent (s)')
ax.set_ylabel('relative $K_H$')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.grid(axis='y')


plt.tight_layout()
plt.show()

# Save
figSubDir = ''
name = 'Indent_repetition'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')
ufun.archiveFig(fig, name = name, ext = '.png', dpi = 200,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')


# %%% Plot stuff

data = df_chiaro_f
# fig, axes = plt.subplots(2, 2, figsize = (12, 4))
# ax=axes[0,0]
# sns.scatterplot(ax=ax, data=data, x='T0', y='Ka', hue='manipID', style='cellNum', s= 75, zorder=5)
# # ax.scatter(data.T0, data.Ka, marker='o', c=stringSeries2catArray(data.cellID), cmap='tab20')
# ax.set_xlim([0, ax.get_xlim()[1]])
# ax.set_ylim([0, ax.get_ylim()[1]])
# ax.set_xlabel('$T_0$ (mN/m)')
# ax.set_ylabel('$K_A$ (mN/m)')
# ax.legend().set_visible(False)
# ax.grid()

# ax=axes[]
# sns.scatterplot(ax=ax, data=data, x='T0', y='Kh', hue='manipID', style='cellNum', s= 75, zorder=5)
# ax.set_xlim([0, ax.get_xlim()[1]])
# ax.set_ylim([0, ax.get_ylim()[1]])
# ax.set_xlabel('$T_0$ (mN/m)')
# ax.set_ylabel('$K_{Hertz}$ (Pa)')
# ax.legend().set_visible(False)
# ax.grid()

 

data = df_chiaro_f
fig, axes = plt.subplots(1, 3, figsize = (12, 4))
ax=axes[0]
sns.scatterplot(ax=ax, data=data, x='T0', y='Ka', hue='manipID', style='cellNum', s= 75, zorder=5)
# ax.scatter(data.T0, data.Ka, marker='o', c=stringSeries2catArray(data.cellID), cmap='tab20')
ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel('$T_0$ (mN/m)')
ax.set_ylabel('$K_A$ (mN/m)')
ax.legend().set_visible(False)
ax.grid()

ax=axes[1]
sns.scatterplot(ax=ax, data=data, x='T0', y='Kh', hue='manipID', style='cellNum', s= 75, zorder=5)
ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel('$T_0$ (mN/m)')
ax.set_ylabel('$K_{Hertz}$ (Pa)')
ax.legend().set_visible(False)
ax.grid()

ax=axes[2]
sns.scatterplot(ax=ax, data=data, x='Ka', y='Kh', hue='manipID', style='cellNum', s= 75, zorder=5)
ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel('$K_A$ (mN/m)')
ax.set_ylabel('$K_{Hertz}$ (Pa)')
ax.legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
ax.grid()
fig.suptitle("All indentations")
fig.tight_layout()


data = df_chiaro_g 
fig, axes = plt.subplots(1, 3, figsize = (12, 4))
ax=axes[0]
sns.scatterplot(ax=ax, data=data, x='T0', y='Ka', hue='manipID', style='cellNum', s= 75, zorder=5)
ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel('$T_0$ (mN/m)')
ax.set_ylabel('$K_A$ (mN/m)')
ax.legend().set_visible(False)
ax.grid()

ax=axes[1]
sns.scatterplot(ax=ax, data=data, x='T0', y='Kh', hue='manipID', style='cellNum', s= 75, zorder=5)
ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel('$T_0$ (mN/m)')
ax.set_ylabel('$K_{Hertz}$ (Pa)')
ax.legend().set_visible(False)
ax.grid()

ax=axes[2]
sns.scatterplot(ax=ax, data=data, x='Ka', y='Kh', hue='manipID', style='cellNum', s= 75, zorder=5)
ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel('$K_A$ (mN/m)')
ax.set_ylabel('$K_{Hertz}$ (Pa)')
ax.legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
ax.grid()
fig.suptitle("Last indentation for each cell")
fig.tight_layout()

plt.show()


data = df_chiaro_f
fig, ax = plt.subplots(1, 1, figsize = (6, 6))
ax=ax
sns.scatterplot(ax=ax, data=data, x='T0', y='T0_relax', hue='manipID', style='cellNum', s= 75, zorder=5)
ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], 'k--', lw=1)
ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel('$T_{0,comp}$ (mN/m)')
ax.set_ylabel('$T_{0,relax}$ (mN/m)')
ax.legend().set_visible(False)
ax.grid()

fig.suptitle("$T_{0}$ from compression vs from relaxation")

plt.show()


    
data=df_merged
figM, axesM = plt.subplots(3, 2, figsize = (12, 8), sharex='col', sharey='row')
ax=axesM[0, 0]
sns.scatterplot(ax=ax, data=data, x='bestH0', y='T0', hue='manipID', style='cellNum', s= 75, zorder=5)
ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel('$H_{cortex}$ (nm)')
ax.set_ylabel('$T_0$ (mN/m)')
ax.legend().set_visible(False)
ax.grid()

ax=axesM[1, 0]
sns.scatterplot(ax=ax, data=data, x='bestH0', y='Ka', hue='manipID', style='cellNum', s= 75, zorder=5)
ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel('$H_{cortex}$ (nm)')
ax.set_ylabel('$K_A$ (mN/m)')
ax.legend().set_visible(False)
ax.grid()

ax=axesM[2, 0]
sns.scatterplot(ax=ax, data=data, x='bestH0', y='E_eq_kPa', hue='manipID', style='cellNum', s= 75, zorder=5)
ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel('$H_{cortex}$ (nm)')
ax.set_ylabel('$E_{eq}$ (kPa)')
ax.legend().set_visible(False)
ax.grid()

ax=axesM[0, 1]
sns.scatterplot(ax=ax, data=data, x='E' + _stiff + '_kPa', y='T0', hue='manipID', style='cellNum', s= 75, zorder=5)
ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel('$E_{cortex}$ (kPa)')
ax.set_ylabel('$T_0$ (mN/m)')
ax.legend(bbox_to_anchor=(1.05, 1), fontsize = 8, ncol = 2,
                         loc='upper left', borderaxespad=0.)
ax.grid()

ax=axesM[1, 1]
sns.scatterplot(ax=ax, data=data, x='E' + _stiff + '_kPa', y='Ka', hue='manipID', style='cellNum', s= 75, zorder=5)
ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel('$E_{cortex}$ (kPa)')
ax.set_ylabel('$K_A$ (mN/m)')
ax.legend().set_visible(False)
ax.grid()

ax=axesM[2, 1]
sns.scatterplot(ax=ax, data=data, x='E' + _stiff + '_kPa', y='E_eq_kPa', hue='manipID', style='cellNum', s= 75, zorder=5)
ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel('$E_{cortex}$ (kPa)')
ax.set_ylabel('$E_{eq}$ (kPa)')
ax.legend().set_visible(False)
ax.grid()

figM.suptitle("$E_{eq}$ = $K_A$ / $H_{cortex}$")

plt.tight_layout()
plt.show()

         
# %%% Plot stuff






















# %% V2


# %%% 24-02-26

dates = ['24-02-26']
Excluded = []
Excluded2 = ['24-02-26_M2_P1_C3-1']

chiaroDir = "D://MagneticPincherData//Raw//24.02.26_NanoIndent_ChiaroData_25um"
df_chiaro = pd.read_csv(os.path.join(chiaroDir, '24-02-26_M2&3_syn.csv'), sep=',')
df_chiaro = df_chiaro[df_chiaro['date'].apply(lambda x : x in dates).values]
df_chiaro['cellName'] = df_chiaro['cellID'].apply(lambda x : '_'.join(x.split('_')[1:])).values
df_chiaro['cellNum'] = df_chiaro['cellID'].apply(lambda x : x.split('_')[-1]).values

Filters = np.array([(df_chiaro['date'].apply(lambda x : x in dates)).values,
                    (df_chiaro['cellID'].apply(lambda x : x not in Excluded)).values,
                    (df_chiaro['valid_UI'] == True).values,
                    ])

totalFilter = np.all(Filters, axis = 0)
df_chiaro_f = df_chiaro[totalFilter]


pincherDir = "D://MagneticPincherData//Data_Analysis"
df_pincher = pd.read_csv(os.path.join(pincherDir, 'MecaData_NanoIndent_2023-2024.csv'), sep=';')

df_pincher['cellCode'] = df_pincher['cellName'].apply(lambda x : x.split('_')[-1])
df_pincher['cellName'] = df_pincher['cellName'].apply(lambda x : x.split('-')[0])
df_pincher['cellID'] = df_pincher['date'] + '_' + df_pincher['cellName']
df_pincher.insert(4, 'cellLongCode', df_pincher['cellID'].apply(lambda x : '_'.join(x.split('_')[:-1])) + '_' + df_pincher['cellCode'])

_stiff = '_f_<_400'
df_pincher['valid' + _stiff] = (df_pincher['E' + _stiff] < 50e3) & (df_pincher['R2' + _stiff] > 0.5) & (df_pincher['Chi2' + _stiff] < 2.5)

Filters = np.array([df_pincher['date'].apply(lambda x : x in dates).values,
                    (df_pincher['cellID'].apply(lambda x : x not in Excluded)).values,
                    (df_pincher['cellLongCode'].apply(lambda x : x not in Excluded2)).values,
                    (df_pincher['validatedThickness'] == True).values,
                    (df_pincher['bestH0'] <= 1000).values,
                    (df_pincher['valid' + _stiff] == True).values,
                    ])

totalFilter = np.all(Filters, axis = 0)
df_pincher_f = df_pincher[totalFilter]


# %% Version 1

# %%% Combine the data from the nanoIndent & the Pincher

dataDir = "D://MagneticPincherData//Data_Analysis"

dates = ['23-12-07', '23-12-10', '24-02-26', '24-02-28'] #, '24-02-28'] # ['23-11-13', '23-11-15', '23-12-07', '23-12-10']

Excluded = ['23-11-15_M1_P1_C1', '23-12-07_M1_P1_C2', '24-02-26_M3_P1_C5', '24-02-28_M3_P1_C4']
Excluded2 = ['24-02-26_M2_P1_C3-1']

# %%%% Format the pincher table

df_pincher = pd.read_csv(os.path.join(dataDir, 'MecaData_NanoIndent_2023-2024.csv'), sep=';')
dates_2sets = ['23-11-13', '23-11-15', '23-12-07', '23-12-10']
dates_1set = ['24-02-26', '24-02-28']

_stiff = '_f_<_400'

# 1. Format

index_2sets = df_pincher[df_pincher['date'].apply(lambda x : x in dates_2sets)].index
df_2sets = df_pincher.loc[index_2sets, :]

df_2sets.insert(5, 'compressionSet', df_2sets['cellName'].apply(lambda x : int(x.split('-')[0][-1])))
df_2sets['cellRawNum'] = df_2sets['cellName'].apply(lambda x : x.split('_')[-1])
df_2sets['cellName'] = df_2sets['cellName'].apply(lambda x : x.split('-')[0][:-1])
df_2sets['cellID'] = df_2sets['date'] + '_' + df_2sets['cellName']
df_2sets['cellCode'] = df_2sets['cellRawNum'].apply(lambda x : x[:2] + x[3:])
df_2sets.insert(4, 'cellLongCode', df_2sets['cellID'].apply(lambda x : '_'.join(x.split('_')[:-1])) + '_' + df_2sets['cellCode'])
df_2sets = df_2sets.drop(columns = ['cellRawNum'])


index_1set = df_pincher[df_pincher['date'].apply(lambda x : x in dates_1set)].index
df_1set = df_pincher.loc[index_1set, :]

df_1set.insert(5, 'compressionSet', np.ones(len(df_1set), dtype=int))
df_1set['cellCode'] = df_1set['cellName'].apply(lambda x : x.split('_')[-1])
df_1set['cellName'] = df_1set['cellName'].apply(lambda x : x.split('-')[0])
df_1set['cellID'] = df_1set['date'] + '_' + df_1set['cellName']
df_1set.insert(4, 'cellLongCode', df_1set['cellID'].apply(lambda x : '_'.join(x.split('_')[:-1])) + '_' + df_1set['cellCode'])


df_pincher = pd.concat([df_2sets, df_1set])

df_pincher['valid' + _stiff] = (df_pincher['E' + _stiff] < 50e3) & (df_pincher['R2' + _stiff] > 0.5) & (df_pincher['Chi2' + _stiff] < 2.5)
rd = {1: 'Before', 2: 'After'}
df_pincher['compressionSet'] = df_pincher['compressionSet'].apply(lambda x : rd[x])

# 2. Filter
Excluded = ['23-11-15_M1_P1_C1', '23-12-07_M1_P1_C2', '24-02-26_M3_P1_C5', '24-02-28_M3_P1_C4']
Excluded2 = ['24-02-26_M2_P1_C3-1']

Filters1 = np.array([df_pincher['date'].apply(lambda x : x in dates).values,
                    (df_pincher['cellID'].apply(lambda x : x not in Excluded)).values,
                    (df_pincher['cellLongCode'].apply(lambda x : x not in Excluded2)).values,
                    (df_pincher['validatedThickness'] == True).values,
                    (df_pincher['bestH0'] <= 1000).values,
                    ])
totalFilter1 = np.all(Filters1, axis = 0)
df_pincher_f1 = df_pincher[totalFilter1]


Filters2 = np.array([df_pincher['date'].apply(lambda x : x in dates).values,
                    (df_pincher['cellID'].apply(lambda x : x not in Excluded)).values,
                    (df_pincher['cellLongCode'].apply(lambda x : x not in Excluded2)).values,
                    (df_pincher['valid' + _stiff] == True).values,
                    ])
totalFilter2 = np.all(Filters2, axis = 0)
df_pincher_f2 = df_pincher[totalFilter2]


# %%%% Format the indenter table

df_indenter = pd.read_csv(os.path.join(dataDir, 'ChiaroData_NanoIndent_2023-2024.csv'), sep=';')

# 1. Format
df_indenter['cellName'] = df_indenter['manip'] + '_P1_' + df_indenter['cell']
df_indenter['cellID'] = df_indenter['date'] + '_' + df_indenter['cellName']

df_indenter['indentName'] = df_indenter['cellName'] + '_' + df_indenter['indent']
df_indenter['indentID'] = df_indenter['cellID'] + '_' + df_indenter['indent']

# 2. Validity crit
delta_Zstart = df_indenter['Z02'] - df_indenter['Z_start']
delta_Zstop = df_indenter['Z_stop'] - df_indenter['Z02']
df_indenter['Valid_Z0'] = (delta_Zstart > 1) & (delta_Zstop > 2)

df_indenter['Valid_Fit'] = (df_indenter['Rsq2'] > 0.3) & (df_indenter['K_d'] < 1500)
df_indenter['Valid_agg'] = df_indenter['Valid_Z0'] & df_indenter['Valid_Fit']

# 3. Filters
Excluded = ['23-11-15_M1_P1_C1', '23-12-07_M1_P1_C2', '24-02-26_M3_P1_C5', '24-02-28_M3_P1_C4']

Filters = np.array([df_indenter['date'].apply(lambda x : x in dates).values,
                    (df_indenter['cellID'].apply(lambda x : x not in Excluded)).values,
                    df_indenter['Valid_agg'].values,
                    ])
totalFilter = np.all(Filters, axis = 0)
df_indenter_f = df_indenter[totalFilter]

# 4. Extra fields
def str2sec(S):
    sec = np.sum(np.array(S.split(':')).astype(int) * np.array([3600, 60, 1]))
    return(sec)

CID = df_indenter_f['cellID'].values
df_indenter_f['countIndent'] = np.array([np.sum(CID == cid) for cid in CID])
CIDU = df_indenter_f['cellID'].unique()

rel_times_sec_list = []
for cid in CIDU:
    temp_df = df_indenter_f[df_indenter_f['cellID'] == cid]
    times_sec = np.array([str2sec(S) for S in temp_df.time])
    rel_times_sec = times_sec - np.min(times_sec)
    rel_times_sec_list.append(rel_times_sec)

df_indenter_f['relative time'] = np.concatenate(rel_times_sec_list)

# %%%% Start plotting

# %%%%% Plot Only NanoIndent

Ndates = len(dates)

fig1, axes1 = plt.subplots(2, (Ndates+1)//2, figsize = (3*Ndates, 6))
axes1_f = axes1.T.flatten()

for k in range(Ndates):
    date_plot = dates[k]
    
    
    # parm = 'bestH0'
    # ax = axes1_f[0 + 2*k]
    # data = df_pincher_f1[df_pincher_f1['date'] == date_plot]
    # nCells = len(data['cellID'].unique())
    # dictPlot = {'data' : data,
    #             'ax' : ax,
    #             'x' : 'cellID',
    #             'y' : parm,
    #             'hue' : 'compressionSet'
    #             }
    # # sns.boxplot(**dictPlot, fliersize = 0, width=0.6)
    # sns.swarmplot(**dictPlot, dodge = True, size=5) # edgecolor='black', linewidth=1
    # ax.set_ylim([0, ax.get_ylim()[1]])
    # ax.get_legend().remove()
    # ax.set_xticklabels('')
    # # ax.set_xticklabels(ax.get_xticklabels(), rotation = 30)
    # ax.set_xlabel('')
    # ax.set_ylabel('$H_0$ (nm)')
    # ax.grid(axis = 'y')
    # ax.legend(loc = 'upper right')
    
    # for kk in range(0, nCells+1):
    #     ax.axvline(kk+0.5, c = 'gray', ls = ':', lw = 0.5)
    
    parm = 'K_d'
    ax = axes1_f[k]
    data = df_indenter_f[df_indenter_f['date'] == date_plot]
    nCells = len(data['cellID'].unique())
    dictPlot = {'data' : data,
                'ax' : ax,
                'x' : 'cellName',
                'y' : parm,
                }
    
    # sns.boxplot(**dictPlot, fliersize = 0, width=0.6)
    sns.swarmplot(**dictPlot, dodge = True, size=5, color = 'darkred')
    ax.set_ylim([0, ax.get_ylim()[1]])
    # ax.get_legend().remove()
    ax.set_xlabel(f'Cells of {date_plot}')
    ax.set_ylabel('$Y_{eff}$ (Pa)')
    ax.grid(axis = 'y')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, fontsize = 8)
    
    for kk in range(0, nCells+1):
        ax.axvline(kk+0.5, c = 'gray', ls = '--', lw = 0.5)


plt.tight_layout()
plt.show()


# %%%%% First Plot
Ndates = len(dates)

fig1, axes1 = plt.subplots(2, Ndates, figsize = (6*Ndates, 6))
axes1_f = axes1.T.flatten()

for k in range(Ndates):
    date_plot = dates[k]
    
    
    parm = 'bestH0'
    ax = axes1_f[0 + 2*k]
    data = df_pincher_f1[df_pincher_f1['date'] == date_plot]
    nCells = len(data['cellID'].unique())
    dictPlot = {'data' : data,
                'ax' : ax,
                'x' : 'cellID',
                'y' : parm,
                'hue' : 'compressionSet'
                }
    # sns.boxplot(**dictPlot, fliersize = 0, width=0.6)
    sns.swarmplot(**dictPlot, dodge = True, size=4) # edgecolor='black', linewidth=1
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.get_legend().remove()
    ax.set_xticklabels('')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, fontsize = 8)
    ax.set_xlabel('')
    ax.set_ylabel('$H_0$ (nm)')
    ax.grid(axis = 'y')
    ax.legend(loc = 'upper right')
    
    for kk in range(0, nCells+1):
        ax.axvline(kk+0.5, c = 'gray', ls = ':', lw = 0.5)
    
    parm = 'K_d'
    ax = axes1_f[1 + 2*k]
    data = df_indenter_f[df_indenter_f['date'] == date_plot]
    dictPlot = {'data' : data,
                'ax' : ax,
                'x' : 'cellName',
                'y' : parm,
                }
    
    # sns.boxplot(**dictPlot, fliersize = 0, width=0.6)
    sns.swarmplot(**dictPlot, dodge = True, size=5, color = 'darkred')
    ax.set_ylim([0, ax.get_ylim()[1]])
    # ax.get_legend().remove()
    ax.set_xlabel(f'Cells of {date_plot}')
    ax.set_ylabel('$Y_{eff}$ (Pa)')
    ax.grid(axis = 'y')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, fontsize = 8)
    
    for kk in range(0, nCells+1):
        ax.axvline(kk+0.5, c = 'gray', ls = '--', lw = 0.5)


plt.tight_layout()
plt.show()

# %%%%% Second Plot

Ndates = len(dates)

fig2, axes2 = plt.subplots(2,Ndates, figsize = (6*Ndates, 6))
axes2_f = axes2.T.flatten()

for k in range(Ndates):
    date_plot = dates[k]
    
    
    parm = 'E' + _stiff
    ax = axes2_f[0 + 2*k]
    data = df_pincher_f2[df_pincher_f2['date'] == date_plot]
    nCells = len(data['cellID'].unique())
    dictPlot = {'data' : data,
                'ax' : ax,
                'x' : 'cellName',
                'y' : parm,
                'hue' : 'compressionSet'
                }
    # sns.boxplot(**dictPlot, fliersize = 0, width=0.6)
    sns.swarmplot(**dictPlot, dodge = True, size=4) # edgecolor='black', linewidth=1
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.get_legend().remove()
    # ax.set_xticklabels('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, fontsize = 8)
    ax.set_xlabel('')
    ax.set_ylabel('E (Pa)')
    ax.grid(axis = 'y')
    ax.legend(loc = 'upper right')
    
    for kk in range(0, nCells+1):
        ax.axvline(kk+0.5, c = 'gray', ls = ':', lw = 0.5)
    
    parm = 'K_d'
    ax = axes2_f[1 + 2*k]
    data = df_indenter_f[df_indenter_f['date'] == date_plot]
    dictPlot = {'data' : data,
                'ax' : ax,
                'x' : 'cellName',
                'y' : parm,
                }
    
    # sns.boxplot(**dictPlot, fliersize = 0, width=0.6)
    sns.swarmplot(**dictPlot, dodge = True, size=5, color = 'darkred',) #, edgecolor='black', linewidth=1)
    ax.set_ylim([0, ax.get_ylim()[1]])
    # ax.get_legend().remove()
    ax.set_xlabel(f'Cells of {date_plot}')
    ax.set_ylabel('$Y_{eff}$ (Pa)')
    ax.grid(axis = 'y')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, fontsize = 8)
    
    for kk in range(0, nCells+1):
        ax.axvline(kk+0.5, c = 'gray', ls = ':', lw = 0.5)


plt.tight_layout()
plt.show()


# %%%%% K & relative time

fig3, axes3 = plt.subplots(1,2, figsize = (12, 6))

parm = 'K_d'

ax = axes3[0]
ax.set_title('Two indents')
CIDU = df_indenter_f[df_indenter_f['countIndent'] == 2]['cellID'].unique()
for cid in CIDU:
    temp_df = df_indenter_f[df_indenter_f['cellID'] == cid]
    if not np.max(temp_df['relative time']) > 150:
        ax.plot(temp_df['relative time'], temp_df[parm], ls='-', marker='o', label = cid)

ax.set_xlabel('Time from first indent (sec)')
ax.set_ylabel('$Y_{eff}$ (Pa)')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.legend()



ax = axes3[1]
ax.set_title('Three or more indents')
CIDU = df_indenter_f[df_indenter_f['countIndent'] > 2]['cellID'].unique()
for cid in CIDU:
    temp_df = df_indenter_f[df_indenter_f['cellID'] == cid]
    ax.plot(temp_df['relative time'], temp_df[parm], ls='-', marker='o', label = cid)

ax.set_xlabel('Time from first indent (sec)')
ax.set_ylabel('$Y_{eff}$ (Pa)')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.legend()


plt.tight_layout()
plt.show()

# %%%% Second style of plots : Average shit together

#### HERE !!!

dates = ['23-11-15', '23-12-07', '23-12-10', '24-02-26', '24-02-28']
# dates = ['23-11-15', '23-12-07', '23-12-10', '24-02-26', '24-02-28']
# dates = ['24-02-26']
dates = ['23-11-15', '23-12-07', '23-12-10', '24-02-26']
# dates = ['24-02-28']
_stiff = '_f_<_400'

# ['23-11-13', '23-11-15', '23-12-07', '23-12-10']

Excluded = ['23-11-15_M1_P1_C1', '23-12-07_M1_P1_C2', '24-02-26_M3_P1_C5', '24-02-28_M3_P1_C4']
Excluded2 = ['24-02-26_M2_P1_C3-1']


# %%%%% Format the pincher table

df_pincher = pd.read_csv(os.path.join(dataDir, 'MecaData_NanoIndent_2023-2024.csv'), sep=';')

index_2sets = df_pincher[df_pincher['date'].apply(lambda x : x in dates_2sets)].index
df_2sets = df_pincher.loc[index_2sets, :]

df_2sets.insert(5, 'compressionSet', df_2sets['cellName'].apply(lambda x : int(x.split('-')[0][-1])))
df_2sets['cellRawNum'] = df_2sets['cellName'].apply(lambda x : x.split('_')[-1])
df_2sets['cellName'] = df_2sets['cellName'].apply(lambda x : x.split('-')[0][:-1])
df_2sets['cellID'] = df_2sets['date'] + '_' + df_2sets['cellName']
df_2sets['cellCode'] = df_2sets['cellRawNum'].apply(lambda x : x[:2] + x[3:])
df_2sets.insert(4, 'cellLongCode', df_2sets['cellID'].apply(lambda x : '_'.join(x.split('_')[:-1])) + '_' + df_2sets['cellCode'])
df_2sets = df_2sets.drop(columns = ['cellRawNum'])


index_1set = df_pincher[df_pincher['date'].apply(lambda x : x in dates_1set)].index
df_1set = df_pincher.loc[index_1set, :]

df_1set.insert(5, 'compressionSet', np.ones(len(df_1set), dtype=int))
df_1set['cellCode'] = df_1set['cellName'].apply(lambda x : x.split('_')[-1])
df_1set['cellName'] = df_1set['cellName'].apply(lambda x : x.split('-')[0])
df_1set['cellID'] = df_1set['date'] + '_' + df_1set['cellName']
df_1set.insert(4, 'cellLongCode', df_1set['cellID'].apply(lambda x : '_'.join(x.split('_')[:-1])) + '_' + df_1set['cellCode'])


df_pincher = pd.concat([df_2sets, df_1set])

df_pincher['valid' + _stiff] = (df_pincher['E' + _stiff] < 50e3) & (df_pincher['R2' + _stiff] > 0.5) & (df_pincher['Chi2' + _stiff] < 2.5)
rd = {1: 'Before', 2: 'After'}
df_pincher['compressionSet'] = df_pincher['compressionSet'].apply(lambda x : rd[x])


Filters = np.array([df_pincher['date'].apply(lambda x : x in dates).values,
                    (df_pincher['cellID'].apply(lambda x : x not in Excluded)).values,
                    (df_pincher['cellLongCode'].apply(lambda x : x not in Excluded2)).values,
                    (df_pincher['validatedThickness'] == True).values,
                    (df_pincher['bestH0'] <= 1000).values,
                    (df_pincher['valid' + _stiff] == True).values,
                    (df_pincher['compressionSet'] == 'Before').values,
                    ])

totalFilter = np.all(Filters, axis = 0)
df_pincher_f = df_pincher[totalFilter]

g = df_pincher_f.groupby('cellID')
df_pincher_g = g.agg({'date':'first', 'bestH0':'mean', 'E' + _stiff :'mean', 'compNum': 'count'}).reset_index()

# %%%%% Format the indenter table

df_indenter = pd.read_csv(os.path.join(dataDir, 'ChiaroData_NanoIndent_2023-2024.csv'), sep=';')

df_indenter['cellName'] = df_indenter['manip'] + '_P1_' + df_indenter['cell']
df_indenter['cellID'] = df_indenter['date'] + '_' + df_indenter['cellName']

df_indenter['indentName'] = df_indenter['cellName'] + '_' + df_indenter['indent']
df_indenter['indentID'] = df_indenter['cellID'] + '_' + df_indenter['indent']

# 2. Validity crit
delta_Zstart = df_indenter['Z02'] - df_indenter['Z_start']
delta_Zstop = df_indenter['Z_stop'] - df_indenter['Z02']
df_indenter['Valid_Z0'] = (delta_Zstart > 1) & (delta_Zstop > 2)

df_indenter['Valid_Fit'] = df_indenter['Rsq2'] > 0.3
df_indenter['Valid_agg'] = df_indenter['Valid_Z0'] & df_indenter['Valid_Fit']


Filters = np.array([df_indenter['date'].apply(lambda x : x in dates).values,
                    (df_indenter['cellID'].apply(lambda x : x not in Excluded)).values,
                    (df_indenter['Valid_agg']).values,
                    ])
totalFilter = np.all(Filters, axis = 0)
df_indenter_f = df_indenter[totalFilter]

g = df_indenter_f.groupby('cellID')
df_indenter_g = g.agg({'K_d':'mean', 'indent': 'count'}).reset_index()



# %%%%% Merge

df_merge_g = df_pincher_g.merge(df_indenter_g, on='cellID')
df_merge_g['kBend'] = (2/3) * ((df_merge_g['bestH0']/1e3)**2 * df_merge_g['E' + _stiff])/10 # Value in pN/µm

# %%%% Start plotting

fig3, axes3 = plt.subplots(1,3, figsize = (15, 6))
fig = fig3

ax = axes3[0]
dictPlot = {'data' : df_merge_g,
            'ax' : ax,
            'x' : 'bestH0',
            'y' : 'K_d',
            'hue' : 'date',
            'sizes':(10,10),
            'edgecolor':'k',
            'zorder':6
            }
sns.scatterplot(**dictPlot)

ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel('$H_0$ (nm)')
ax.set_ylabel('$Y_{eff}$ (Pa)')
ax.grid()


ax = axes3[1]
dictPlot = {'data' : df_merge_g,
            'ax' : ax,
            'x' : 'E' + _stiff,
            'y' : 'K_d',
            'hue' : 'date',
            'sizes':(10,10),
            'edgecolor':'k',
            'zorder':6
            }
sns.scatterplot(**dictPlot)

ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel('$E_{cortex}$ (Pa)')
ax.set_ylabel('$Y_{eff}$ (Pa)')
ax.grid() #axis = 'y'


ax = axes3[2]
dictPlot = {'data' : df_merge_g,
            'ax' : ax,
            'x' : 'kBend',
            'y' : 'K_d',
            'hue' : 'date',
            'sizes':(10,10),
            'edgecolor':'k',
            'zorder':6
            }
sns.scatterplot(**dictPlot)

ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel('$k_{bend}$ (pN/µm)')
ax.set_ylabel('$Y_{eff}$ (Pa)')
ax.grid()

plt.tight_layout()
plt.show()
