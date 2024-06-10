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

import CellShapeSolver as css
import UtilityFunctions as ufun
import GraphicStyles as gs

gs.set_mediumText_options_jv()


# %% Version 2

# %%% Utility

def stringSeries2catArray(S):
    unique_vals = S.unique()
    d = {unique_vals[k]:k for k in range(len(unique_vals))}
    A = np.array([d[x] for x in S.values])
    return(A)

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

# %%% 24-02-26 & 24-04-11

dates = ['24-02-26', '24-04-11']
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
                    (df_chiaro['manipID'].apply(lambda x : x in ['24-04-11_M1', '24-04-11_M3', '24-04-11_M5'])).values,
                    # (df_chiaro['manipID'].apply(lambda x : x not in ['24-04-11_M2'])).values,
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
df_merged['E' + _stiff + '_kPa'] = df_merged['E' + _stiff] * 1e-3


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
