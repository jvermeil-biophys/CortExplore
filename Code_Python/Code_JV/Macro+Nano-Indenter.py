# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:22:07 2023

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

# %% Combine the data from the nanoIndent & the Pincher

dataDir = "D://MagneticPincherData//Data_Analysis"

dates = ['23-12-07', '23-12-10', '24-02-26', '24-02-28'] #, '24-02-28'] # ['23-11-13', '23-11-15', '23-12-07', '23-12-10']

Excluded = ['23-11-15_M1_P1_C1', '23-12-07_M1_P1_C2', '24-02-26_M3_P1_C5', '24-02-28_M3_P1_C4']
Excluded2 = ['24-02-26_M2_P1_C3-1']

# %%% Format the pincher table

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


# %%% Format the indenter table

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

# %%% Start plotting

# %%%% Plot Only NanoIndent

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


# %%%% First Plot
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

# %%%% Second Plot

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


# %%%% K & relative time

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

# %%% Second style of plots : Average shit together

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


# %%%% Format the pincher table

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

# %%%% Format the indenter table

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



# %%%% Merge

df_merge_g = df_pincher_g.merge(df_indenter_g, on='cellID')
df_merge_g['kBend'] = (2/3) * ((df_merge_g['bestH0']/1e3)**2 * df_merge_g['E' + _stiff])/10 # Value in pN/µm

# %%% Start plotting

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



# %% Cell indents

# %%% Analysis 1 by 1

# %%%% Get the indentation

date, manip, cell, indent = '23.12.10', 'M1', 'C3', '003'

fpath = f'D://MagneticPincherData//Raw//{date}_NanoIndent_ChiaroData//{manip}//{cell}//Indentations//{cell} Indentation_{indent}.txt'
f = fpath.split('//')[-1]
cell = f[:2]

resDir = {'date':[],
          'manip':[],
          'cell':[],
          'indent':[],
          'indentID':[],
          # 'path':[],
          'time':[],
          #
          'Z_start':[],
          'Z_stop':[],
          #
          'F_moy':[],
          'F_std':[],
          'F_max':[],
          #
          'K1':[],
          'Z01':[],
          'Rsq1':[],
          'K2':[],
          'Z02':[],
          'Rsq2':[],
          }

headerFormat = 'Time (s)	Load (uN)	Indentation (nm)	Cantilever (nm)	Piezo (nm)	Auxiliary'

def getLineWithString(lines, string):
    i = 0
    while string not in lines[i]:
        i+=1
    return(lines[i], i)

indent_number = int(f[:-4].split('_')[-1])
indent = f'I{indent_number:.0f}'
fopen = open(fpath, 'r')
flines = fopen.readlines()
s, i = getLineWithString(flines, headerFormat)
metadataText = flines[:i]
df = pd.read_csv(fpath, skiprows = i-1, sep='\t')
df['Z Tip (nm)'] = df['Piezo (nm)'] - df['Cantilever (nm)']

dI = {'indent_name':f, 'path':fpath, 'metadata':metadataText, 'df':df}

dI['date'] = date
dI['manip'] = manip
dI['cell'] = cell
dI['indent'] = indent

time = flines[0].split('\t')[3]
dI['time'] = time
R = float(getLineWithString(flines, 'Tip radius (um)')[0][16:])
dI['R'] = R

text_start_times, j = getLineWithString(flines, 'Step absolute start times')
text_end_times, j = getLineWithString(flines, 'Step absolute end times')
start_times = np.array(text_start_times[30:].split(',')).astype(float)
end_times = np.array(text_end_times[28:].split(',')).astype(float)

dI['ti'], dI['tf'] = start_times, end_times

fopen.close()

# %%%% Plot before fit

fig, axes = plt.subplots(1,2, figsize = (12,6))

ax=axes[0]
ax.plot(df['Time (s)'], df['Piezo (nm)'], 'b-', label='Displacement')
ax.plot(df['Time (s)'], df['Z Tip (nm)'], color='gold', ls='-', label='Z Tip')
ax.plot(df['Time (s)'], df['Cantilever (nm)'], 'g-', label='Deflection')
ax.plot(df['Time (s)'], df['Indentation (nm)'], 'r-', label='Indentation')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance (nm)')
ax.legend()

ax=axes[1]
colorList = gs.colorList10[:len(dI['ti'])]
i = 0
for ti, tf, c in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]):
    i += 1
    step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
    ax.plot(df.loc[step_indices, 'Z Tip (nm)'], df.loc[step_indices, 'Load (uN)'], color=c,
            marker = '.', markersize = 2, ls = '')

ax.set_xlabel('Distance (nm)')
ax.set_ylabel('Load (uN)')

fig.suptitle(dI['indent_name'])
plt.tight_layout()
plt.show()


# %%%% Analysis of the compression - non denoised
ti, tf = dI['ti'][0], dI['tf'][0]
compression_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
Z = df.loc[compression_indices, 'Z Tip (nm)'].values*1e-3
F = df.loc[compression_indices, 'Load (uN)'].values*1e6
R = dI['R']

mode = 'dZmax'
fractionF = 0.4
dZmax = 4

early_points = 1000
F_moy = np.median(F[:early_points])
F_std = np.std(F[:early_points])
F_max = np.max(F)

Z_start = np.min(Z)
Z_stop = np.max(Z)

# Function
def HertzFit(z, K, Z0):
    zeroInd = 1e-9 * np.ones_like(z)
    d = z - Z0
    d = np.where(d>0, d, zeroInd)
    f = (4/3) * K * (R**0.5) * (d**1.5) + F_moy
    return(f)

#### First Fit Zone
Z0_sup = np.max(Z)

#### First Fit
upper_threshold = F_moy + 0.75 * (F_max - F_moy)
i_stop = ufun.findFirst(True, F >= upper_threshold)
Z1, F1 = Z[:i_stop], F[:i_stop]
binf = [0, 0]
bsup = [5000, Z0_sup]
p0=[100, 1] # 0.99*Z0_sup
# Z1, F1 = Z[i_start1:i_stop], F[i_start1:i_stop]
# Z1, F1 = Z, F
[K1, Z01], covM1 = curve_fit(HertzFit, Z1, F1, p0=p0, bounds=(binf, bsup))
Z_fit1 = np.linspace(np.min(Z), np.max(Z), 200)
F_fit1 = HertzFit(Z_fit1, K1, Z01)
Rsq1 = ufun.get_R2(F1, HertzFit(Z1, K1, Z01))

#### Second Fit Zone
i_start = ufun.findFirst(True, Z >= Z01 - 1) # 1 µm margin
if mode == 'fractionF':
    upper_threshold = F_moy + fractionF * (F_max - F_moy)
    i_stop = ufun.findFirst(True, F >= upper_threshold)
elif mode == 'dZmax':
    i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z01 + dZmax) - 1


#### Second Fit (several iterations to converge to a consistant fit)
Z_inf = Z01 - 3  # 2 µm margin
Z0_approx = Z01
for i in range(3):
    i_start = ufun.findFirst(True, Z >= Z_inf)
    if mode == 'fractionF':
        upper_threshold = F_moy + fractionF * (F_max - F_moy)
        i_stop = ufun.findFirst(True, F >= upper_threshold)
    elif mode == 'dZmax':
        i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z0_approx + dZmax) - 1
        
    binf = [0, 0]
    bsup = [5000, Z0_sup]
    p0=[K1, 1]
    
    Z2, F2 = Z[i_start:i_stop], F[i_start:i_stop]
    [K2, Z02], covM2 = curve_fit(HertzFit, Z2, F2, p0=p0, bounds=(binf, bsup))
    Z_fit2 = np.linspace(Z[i_start], Z[i_stop], 100)
    F_fit2 = HertzFit(Z_fit2, K2, Z02)
    Rsq2 = ufun.get_R2(F2, HertzFit(Z2, K2, Z02))
    
    Z_inf = Z02 - 3 # 2 µm margin for the next iteration
    Z0_approx = Z02
    
i_start = ufun.findFirst(True, Z >= Z02)

# #### Second Fit Refinement
# i_start = ufun.findFirst(True, Z >= Z02 - 1.5)            
# if mode == 'fractionF':
#     upper_threshold = F_moy + fractionF * (F_max - F_moy)
#     i_stop = ufun.findFirst(True, F >= upper_threshold)
# elif mode == 'dZmax':
#     i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z02 + dZmax) - 1
    
# binf = [0, 0]
# bsup = [5000, Z0_sup]
# p0=[K2, Z02]

# Z2, F2 = Z[i_start:i_stop], F[i_start:i_stop]
# [K2, Z02], covM2 = curve_fit(HertzFit, Z2, F2, p0=p0, bounds=(binf, bsup))
# Z_fit2 = np.linspace(Z[i_start], Z[i_stop], 100)
# F_fit2 = HertzFit(Z_fit2, K2, Z02)
# Rsq2 = ufun.get_R2(F2, HertzFit(Z2, K2, Z02))


#### Error Fits
arraySteps = np.array([-0.25, +0.25])
Z0_error = arraySteps + Z02
# print(Z0_test)
resError = []
for i in range(len(arraySteps)):
    Z0 = Z0_error[i]
    # function
    def HertzFit_1parm(z, K):
        zeroInd = 1e-9 * np.ones_like(z)
        d = z - Z0
        d = np.where(d>0, d, zeroInd)
        f = (4/3) * K * R**0.5 * (d)**1.5 + F_moy
        return(f)
    # zone
    i_start_test = len(Z) - ufun.findFirst(True, Z[::-1] <= Z0)
    # print(Z0, np.min(Z[i_start_test:i_stop]))
    # fit
    binf = [0]
    bsup = [np.inf]
    p0=[K2]
    [K_error], covM_error = curve_fit(HertzFit_1parm, Z[i_start_test:i_stop], F[i_start_test:i_stop], p0=p0, bounds=(binf, bsup))
    resError.append(K_error)
    
# print(resError)
K_error = np.array(resError)

#### Results
fitResults = {'R':R,
               'mode':mode,
               'Z_start':Z_start,
               'Z_stop':Z_stop,
               'F_moy':F_moy,
               'F_std':F_std,
               'F_max':F_max,
               # 'i_start1':i_start1,
               # 'i_start2':i_start2,
               'i_start':i_start,
               'i_stop':i_stop,
               'K1':K1,
               'Z01':Z01,
               'covM1':covM1,
               'Rsq1':Rsq1,
               'Z_fit1':Z_fit1,
               'F_fit1':F_fit1,
               'K2':K2,
               'Z02':Z02,
               'covM2':covM2,
               'Rsq2':Rsq2,
               'Z_fit2':Z_fit2,
               'F_fit2':F_fit2,
                'Z0_error':Z0_error,
                'K_error':K_error,
               }

# %%%% Plot 1st fit



def Hertz(z, Z0, K, R, F0):
    zeroInd = 1e-9 * np.ones_like(z)
    d = z - Z0
    d = np.where(d>0, d, zeroInd)
    f = (4/3) * K * R**0.5 * (d)**1.5 + F0
    return(f)


fig, axes = plt.subplots(1,3, figsize = (15,6))

#### Reminder of the raw data

ax=axes[0]
ax.plot(df['Time (s)'], df['Piezo (nm)'], 'b-', label='Displacement')
ax.plot(df['Time (s)'], df['Z Tip (nm)'], color='gold', ls='-', label='Z Tip')
ax.plot(df['Time (s)'], df['Cantilever (nm)'], 'g-', label='Deflection')
ax.plot(df['Time (s)'], df['Indentation (nm)'], 'r-', label='Indentation')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance (nm)')
ax.legend()

#### Fitting the compression

ax=axes[1]
colorList = gs.colorList10[:len(dI['ti'])]
i = 0
for ti, tf, c in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]):
    i += 1
    step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
    # print(step_indices)
    ax.plot(df.loc[step_indices, 'Z Tip (nm)']*1e-3, df.loc[step_indices, 'Load (uN)']*1e3, color=c,
            marker = '.', markersize = 2, ls = '')

ax.set_xlabel('Distance (um)')
ax.set_ylabel('Load (nN)')

# ax.axvline(Z[fitResults['i_start1']], ls = '--', color = 'yellow', zorder = 4)
# ax.axvline(Z[fitResults['i_start2']], ls = '--', color = 'orange', zorder = 4)
ax.axvline(Z[fitResults['i_start']], ls = '--', color = 'orange', zorder = 4)
ax.axvline(Z[fitResults['i_stop']], ls = '--', color = 'red', zorder = 4)
ax.axhline((fitResults['F_moy'])*1e-3, ls = '-', color = 'k', zorder = 4)
ax.axhline((fitResults['F_moy']+fitResults['F_std'])*1e-3, ls = '--', color = 'k', zorder = 4)
labelFit1 = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(fitResults['Z01'], 
                                                                                               fitResults['covM1'][1,1]**0.5,
                                                                                               fitResults['K1'], 
                                                                                               fitResults['covM1'][0,0]**0.5,
                                                                                               fitResults['Rsq1'],)
ax.plot(fitResults['Z_fit1'], fitResults['F_fit1']*1e-3, lw=3,
        ls = '-', color = 'orange', zorder = 5, label = labelFit1)

labelFit2 = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(fitResults['Z02'], 
                                                                                       fitResults['covM2'][1,1]**0.5,
                                                                                       fitResults['K2'], 
                                                                                       fitResults['covM2'][0,0]**0.5,
                                                                                       fitResults['Rsq2'],)
ax.plot(fitResults['Z_fit2'], fitResults['F_fit2']*1e-3, 
        ls = '-', color = 'red', zorder = 7, label = labelFit2)

Z0_error, K_error = fitResults['Z0_error'], fitResults['K_error']
for j in range(len(Z0_error)):
    Z0, K = Z0_error[j], K_error[j]
    Zplot = np.linspace(Z0, Z[fitResults['i_stop']], 100)
    Ffit = Hertz(Zplot, Z0, K, fitResults['R'], fitResults['F_moy'])
    ax.plot(Zplot, Ffit*1e-3, ls = '--', color = 'cyan', zorder = 4)
dZ0 = np.abs(Z0_error[1] - Z0_error[0])/2
labelFitErrors = 'Effect of error on Z0\n+/-{:.2f} µm =>\nYeff = {:.0f}~{:.0f} Pa'.format(dZ0, K_error[0], K_error[1])
ax.plot([], [], ls = '--', color = 'cyan', zorder = 6, label = labelFitErrors)
ax.legend()



#### Residuals

ax=axes[2]
compression_indices = df[(df['Time (s)'] >= dI['ti'][0]) & (df['Time (s)'] <= dI['tf'][0])].index
Time = df.loc[compression_indices, 'Time (s)']
Zmeas, Fmeas = df.loc[compression_indices, 'Z Tip (nm)']*1e-3, df.loc[compression_indices, 'Load (uN)']
Ffit = Hertz(Zmeas, fitResults['Z02'], fitResults['K2'], fitResults['R'], fitResults['F_moy'])
residuals = Ffit*1e-6 - Fmeas
ax.plot(Zmeas[:fitResults['i_start']], residuals[:fitResults['i_start']]*1e3, 
        c = gs.colorList10[0], marker = '.', markersize = 2, ls = '')
ax.plot(Zmeas[fitResults['i_start']:fitResults['i_stop']], residuals[fitResults['i_start']:fitResults['i_stop']]*1e3, 
        c = gs.colorList10[1], marker = '.', markersize = 2, ls = '')
ax.plot(Zmeas[fitResults['i_stop']:], residuals[fitResults['i_stop']:]*1e3, 
        c = 'maroon', marker = '.', markersize = 2, ls = '')
ax.axvline(Zmeas[fitResults['i_start']], ls = '--', color = 'orange', zorder = 4)
ax.axvline(Zmeas[fitResults['i_stop']], ls = '--', color = 'red', zorder = 4)

# Zmeas_fit = Zmeas[fitResults['i_start']:fitResults['i_stop']]
# residuals_fit = residuals[fitResults['i_start']:fitResults['i_stop']]
# Zmeas_before = Zmeas[:fitResults['i_start']]
residuals_before = residuals[:fitResults['i_start']]
residuals_before = residuals_before - np.mean(residuals_before)


#### Fourier transform
timestep = 0.001

prominence1 = 0.2
distance1 = 15

residuals_before = residuals[:fitResults['i_start']]
residuals_before = residuals_before - np.mean(residuals_before)
FT_before = np.abs(np.fft.fft(residuals_before))
n_before = FT_before.size
freq_before = np.fft.fftfreq(n_before, d=timestep)[:n_before//2]
FT_normed_before = FT_before[:n_before//2] / np.max(FT_before[:n_before//2])
peaks_before, peaks_prop_before = signal.find_peaks(FT_normed_before, prominence = prominence1, distance=distance1)

prominence2 = 0.05
distance2 = 15

residuals_fit = residuals[fitResults['i_start']:fitResults['i_stop']]
residuals_fit = residuals_fit - np.mean(residuals_fit)
FT_fit = np.abs(np.fft.fft(residuals_fit))
n_fit = FT_fit.size
freq_fit = np.fft.fftfreq(n_fit, d=timestep)[:n_fit//2]
FT_normed_fit = FT_fit[:n_fit//2] / np.max(FT_fit[:n_fit//2])
peaks_fit, peaks_prop_fit = signal.find_peaks(FT_normed_fit, prominence = prominence2, distance=distance2)

fig_ft, axes_ft = plt.subplots(1, 2, figsize = (10,6))
fig_ft.suptitle('FFT of residuals')

ax_ft = axes_ft[0]
ax_ft.plot(freq_before, FT_normed_before, c = gs.colorList10[0]) #, ls='', marker='.', markersize = 3)
i = 1
for p in peaks_before:
    if freq_before[p]>15:
        ax_ft.axvline(freq_before[p], c = gs.colorList10[i], ls='--', lw=1, label=f'{freq_before[p]:.1f}', zorder = 1)
        i += 1
        i %= 10
ax_ft.legend(title = 'Peaks freq')
ax_ft.set_xlim(ax_ft.get_xlim()[0], 250)
ax_ft.set_title(f'Before contact & peaks with prominence > {prominence1:.2f}')


ax_ft = axes_ft[1]
ax_ft.plot(freq_fit, FT_normed_fit, c = gs.colorList10[1]) #, ls='', marker='.', markersize = 3)
i = 1
for p in peaks_fit:
    if freq_fit[p]>15:
        ax_ft.axvline(freq_fit[p], c = gs.colorList10[i], ls='--', lw=1, label=f'{freq_fit[p]:.1f}', zorder = 1)
        i += 1
        i %= 10
ax_ft.legend(title = 'Peaks freq')
ax_ft.set_xlim(ax_ft.get_xlim()[0], 250)
ax_ft.set_title(f'After contact & peaks with prominence > {prominence2:.2f}')


# freq2, Pxx = signal.welch(residuals_fit, fs = 1/timestep, nperseg = 2**9, noverlap = int(2**9*0.9), nfft = 2**12)
# ax_ft.semilogy(freq2, Pxx) #, ls='', marker='.', markersize = 3)

# peaks, peaks_prop = signal.find_peaks(FT_fit, height = 1, distance = 10)
# timestep = 0.001
# a, b = signal.iirnotch(w0=100, Q=50, fs=1/timestep)
# residuals_filter = signal.lfilter(a, b, residuals_fit)
# a, b = signal.iirnotch(w0=64.5, Q=50, fs=1/timestep)
# residuals_filter = signal.lfilter(a, b, residuals_filter)

# axes[2].plot(Zmeas_fit, residuals_filter, color = 'cyan',
#         marker = '.', markersize = 2, ls = '')

# timestep = 0.001
# FT_fit2 = np.abs(np.fft.fft(residuals_filter))
# n2 = FT_fit2.size
# freq2 = np.fft.fftfreq(n2, d=timestep)
# ax_ft.plot(freq2, FT_fit2) #, ls='', marker='.', markersize = 3)


fig.suptitle(dI['indent_name'])
plt.tight_layout()
plt.show()

# %%%% Denoising

timestep = 0.001
fs = 1/timestep

Nfreq_to_cut = 5

dfp = pd.DataFrame(peaks_prop_fit)
dfp['freq'] = [freq_fit[p] for p in peaks_fit]
dfp = dfp[dfp['freq'] > 10]
dfp = dfp.sort_values(by='prominences', ascending=False)

Nfreq_to_cut = min(Nfreq_to_cut, len(dfp['freq'].values))
noise_freqs = dfp['freq'][:Nfreq_to_cut]

denoised_deflect = df['Cantilever (nm)'].values[:]
denoised_load = df['Load (uN)'].values[:]
for nf in noise_freqs:
    a, b = signal.iirnotch(w0=nf, Q=30, fs=fs)
    denoised_deflect = signal.lfilter(a, b, denoised_deflect)
    denoised_load = signal.lfilter(a, b, denoised_load)

df['Cantilever - denoised'] = denoised_deflect
df['Z Tip - denoised'] = df['Piezo (nm)'].values - denoised_deflect
df['Load - denoised'] = denoised_load

# %%%% Plot denoising

timestep = 0.001
fs = 1/timestep

# residuals_fit = residuals[fitResults['i_start']:fitResults['i_stop']]
# residuals_fit = residuals_fit - np.mean(residuals_fit)
# FT_fit = np.abs(np.fft.fft(residuals_fit))
# n_fit = FT_fit.size
# freq_fit = np.fft.fftfreq(n_fit, d=timestep)[:n_fit//2]
# FT_normed_fit = FT_fit[:n_fit//2] / np.max(FT_fit[:n_fit//2])
# peaks_fit, peaks_prop_fit = signal.find_peaks(FT_normed_fit, prominence = prominence2, distance=distance2)

deflect_fit = df['Cantilever (nm)'].values[fitResults['i_start']:fitResults['i_stop']]
load_fit = df['Load (uN)'].values[fitResults['i_start']:fitResults['i_stop']]

deflect_fit = deflect_fit - np.mean(deflect_fit)
load_fit = load_fit - np.mean(load_fit)

#### Fourier transform

fig, axes = plt.subplots(1,3, figsize = (16,6))

ax = axes[0]
FT_deflect = np.abs(np.fft.fft(deflect_fit))
n = FT_deflect.size
freq_deflect = np.fft.fftfreq(n, d=timestep)

FT_deflect = FT_deflect[:n//2]
freq_deflect = freq_deflect[:n//2]
# n = n//2
ax.plot(freq_deflect, FT_deflect/np.max(FT_deflect), label = 'Raw')

FT_denoised = np.abs(np.fft.fft(denoised_deflect[fitResults['i_start']:fitResults['i_stop']] \
                                - np.mean(denoised_deflect[fitResults['i_start']:fitResults['i_stop']])))
FT_denoised = FT_denoised[:n//2]
n = n//2
ax.plot(freq_deflect, FT_denoised/np.max(FT_deflect) - 0.2, label = 'Filtered')
for f in noise_freqs:
    ax.axvline(f, c='r', ls='--', lw=1, zorder=1)
ax.legend()

# ax = axes_ft[1]
# FT_load = np.abs(np.fft.fft(early_load))
# n = FT_load.size
# freq_load = np.fft.fftfreq(n, d=timestep)
# ax.plot(freq_load, FT_load/np.max(FT_load))

# peaks, peaks_prop = signal.find_peaks(FT_deflect/np.max(FT_deflect), prominence = 0.12)
# for i in peaks:
#     print(freq_deflect[i])

# freq2, Pxx = signal.welch(residuals_fit, fs = 1/timestep, nperseg = 2**9, noverlap = int(2**9*0.9), nfft = 2**12)
# ax_ft.semilogy(freq2, Pxx) #, ls='', marker='.', markersize = 3)

# peaks, peaks_prop = signal.find_peaks(FT_fit, height = 1, distance = 10)
# timestep = 0.001
# a, b = signal.iirnotch(w0=100, Q=50, fs=1/timestep)
# residuals_filter = signal.lfilter(a, b, residuals_fit)
# a, b = signal.iirnotch(w0=64.5, Q=50, fs=1/timestep)
# residuals_filter = signal.lfilter(a, b, residuals_filter)

# axes[2].plot(Zmeas_fit, residuals_filter, color = 'cyan',
#         marker = '.', markersize = 2, ls = '')

# timestep = 0.001
# FT_fit2 = np.abs(np.fft.fft(residuals_filter))
# n2 = FT_fit2.size
# freq2 = np.fft.fftfreq(n2, d=timestep)
# ax_ft.plot(freq2, FT_fit2) #, ls='', marker='.', markersize = 3)


ax=axes[1]
ax.plot(df['Time (s)'], df['Load (uN)']*1e3, c=gs.colorList40[12], marker = '.', markersize = 2, ls = '', label = 'Raw')
ax.plot(df['Time (s)'], df['Load - denoised']*1e3, c=gs.colorList40[32], marker = '.', markersize = 2, ls = '', label = 'Filtered')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Load (uN)')
ax.legend()

ax=axes[2]
colorList = gs.colorList10
ti, tf  = dI['ti'][0], dI['tf'][0]
step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
ax.plot(df.loc[step_indices, 'Z Tip (nm)'], df.loc[step_indices, 'Load (uN)']*1e3, 
        c=gs.colorList40[10], marker = '.', markersize = 2, ls = '', label = 'Raw')
ax.plot(df.loc[step_indices, 'Z Tip - denoised'], df.loc[step_indices, 'Load - denoised']*1e3, 
        c=gs.colorList40[30], marker = '.', markersize = 2, ls = '', label = 'Filtered')
ax.set_xlabel('Distance (nm)')
ax.set_ylabel('Load (nN)')
ax.legend()

fig.suptitle(dI['indent_name'])
plt.tight_layout()
plt.show()


# %%%% Analysis of the compression - denoised
ti, tf = dI['ti'][0], dI['tf'][0]
compression_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
Z = df.loc[compression_indices, 'Z Tip - denoised'].values*1e-3
F = df.loc[compression_indices, 'Load - denoised'].values*1e6

R = dI['R']
mode = 'dZmax'
fractionF = 0.4
dZmax = 4

early_points = 1000
F_moy = np.median(F[:early_points])
F_std = np.std(F[:early_points])
F_max = np.max(F)

Z_start = np.min(Z)
Z_stop = np.max(Z)

# Function
def HertzFit(z, K, Z0):
    zeroInd = 1e-9 * np.ones_like(z)
    d = z - Z0
    d = np.where(d>0, d, zeroInd)
    f = (4/3) * K * (R**0.5) * (d**1.5) + F_moy
    return(f)

#### First Fit Zone
Z0_sup = np.max(Z)

#### First Fit
upper_threshold = F_moy + 0.75 * (F_max - F_moy)
i_stop = ufun.findFirst(True, F >= upper_threshold)
Z1, F1 = Z[:i_stop], F[:i_stop]
binf = [0, 0]
bsup = [5000, Z0_sup]
p0=[100, 1] # 0.99*Z0_sup
# Z1, F1 = Z[i_start1:i_stop], F[i_start1:i_stop]
# Z1, F1 = Z, F
[K1, Z01], covM1 = curve_fit(HertzFit, Z1, F1, p0=p0, bounds=(binf, bsup))
Z_fit1 = np.linspace(np.min(Z), np.max(Z), 200)
F_fit1 = HertzFit(Z_fit1, K1, Z01)
Rsq1 = ufun.get_R2(F1, HertzFit(Z1, K1, Z01))

#### Second Fit Zone
i_start = ufun.findFirst(True, Z >= Z01 - 1) # 1 µm margin
if mode == 'fractionF':
    upper_threshold = F_moy + fractionF * (F_max - F_moy)
    i_stop = ufun.findFirst(True, F >= upper_threshold)
elif mode == 'dZmax':
    i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z01 + dZmax) - 1


#### Second Fit (several iterations to converge to a consistant fit)
Z_inf = Z01 - 3  # 2 µm margin
Z0_approx = Z01
for i in range(3):
    i_start = ufun.findFirst(True, Z >= Z_inf)
    if mode == 'fractionF':
        upper_threshold = F_moy + fractionF * (F_max - F_moy)
        i_stop = ufun.findFirst(True, F >= upper_threshold)
    elif mode == 'dZmax':
        i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z0_approx + dZmax) - 1
        
    binf = [0, 0]
    bsup = [5000, Z0_sup]
    p0=[K1, 1]
    
    Z2, F2 = Z[i_start:i_stop], F[i_start:i_stop]
    [K2, Z02], covM2 = curve_fit(HertzFit, Z2, F2, p0=p0, bounds=(binf, bsup))
    Z_fit2 = np.linspace(Z[i_start], Z[i_stop], 100)
    F_fit2 = HertzFit(Z_fit2, K2, Z02)
    Rsq2 = ufun.get_R2(F2, HertzFit(Z2, K2, Z02))
    
    Z_inf = Z02 - 3 # 2 µm margin for the next iteration
    Z0_approx = Z02
    
i_start = ufun.findFirst(True, Z >= Z02)

# #### Second Fit Refinement
# i_start = ufun.findFirst(True, Z >= Z02 - 1.5)            
# if mode == 'fractionF':
#     upper_threshold = F_moy + fractionF * (F_max - F_moy)
#     i_stop = ufun.findFirst(True, F >= upper_threshold)
# elif mode == 'dZmax':
#     i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z02 + dZmax) - 1
    
# binf = [0, 0]
# bsup = [5000, Z0_sup]
# p0=[K2, Z02]

# Z2, F2 = Z[i_start:i_stop], F[i_start:i_stop]
# [K2, Z02], covM2 = curve_fit(HertzFit, Z2, F2, p0=p0, bounds=(binf, bsup))
# Z_fit2 = np.linspace(Z[i_start], Z[i_stop], 100)
# F_fit2 = HertzFit(Z_fit2, K2, Z02)
# Rsq2 = ufun.get_R2(F2, HertzFit(Z2, K2, Z02))


#### Error Fits
arraySteps = np.array([-0.25, +0.25])
Z0_error = arraySteps + Z02
# print(Z0_test)
resError = []
for i in range(len(arraySteps)):
    Z0 = Z0_error[i]
    # function
    def HertzFit_1parm(z, K):
        zeroInd = 1e-9 * np.ones_like(z)
        d = z - Z0
        d = np.where(d>0, d, zeroInd)
        f = (4/3) * K * R**0.5 * (d)**1.5 + F_moy
        return(f)
    # zone
    i_start_test = len(Z) - ufun.findFirst(True, Z[::-1] <= Z0)
    # print(Z0, np.min(Z[i_start_test:i_stop]))
    # fit
    binf = [0]
    bsup = [np.inf]
    p0=[K2]
    [K_error], covM_error = curve_fit(HertzFit_1parm, Z[i_start_test:i_stop], F[i_start_test:i_stop], p0=p0, bounds=(binf, bsup))
    resError.append(K_error)
    
# print(resError)
K_error = np.array(resError)

#### Results
fitResults_denoised = {'R':R,
               'mode':mode,
               'Z_start':Z_start,
               'Z_stop':Z_stop,
               'F_moy':F_moy,
               'F_std':F_std,
               'F_max':F_max,
               # 'i_start1':i_start1,
               # 'i_start2':i_start2,
               'i_start':i_start,
               'i_stop':i_stop,
               'K1':K1,
               'Z01':Z01,
               'covM1':covM1,
               'Rsq1':Rsq1,
               'Z_fit1':Z_fit1,
               'F_fit1':F_fit1,
               'K2':K2,
               'Z02':Z02,
               'covM2':covM2,
               'Rsq2':Rsq2,
               'Z_fit2':Z_fit2,
               'F_fit2':F_fit2,
                'Z0_error':Z0_error,
                'K_error':K_error,
               }


# %%%% Plot fit on raw vs. denoised

Z = df.loc[compression_indices, 'Z Tip (nm)'].values*1e-3
F = df.loc[compression_indices, 'Load (uN)'].values*1e6
Z_d = df.loc[compression_indices, 'Z Tip - denoised'].values*1e-3
F_d = df.loc[compression_indices, 'Load - denoised'].values*1e6

def Hertz(z, Z0, K, R, F0):
    zeroInd = 1e-9 * np.ones_like(z)
    d = z - Z0
    d = np.where(d>0, d, zeroInd)
    f = (4/3) * K * R**0.5 * (d)**1.5 + F0
    return(f)


fig, axes = plt.subplots(1,2, figsize = (13,6), sharey=True)

#### Fitting the raw compression

ax=axes[0]
colorList = gs.colorList10[:len(dI['ti'])]
i = 0
for ti, tf, c in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]):
    i += 1
    step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
    # print(step_indices)
    ax.plot(df.loc[step_indices, 'Z Tip (nm)']*1e-3, df.loc[step_indices, 'Load (uN)']*1e3, color=c,
            marker = '.', markersize = 2, ls = '')

ax.set_xlabel('Distance (µm)')
ax.set_ylabel('Load (nN)')

# ax.axvline(Z[fitResults['i_start1']], ls = '--', color = 'yellow', zorder = 4)
# ax.axvline(Z[fitResults['i_start2']], ls = '--', color = 'orange', zorder = 4)
ax.axvline(Z[fitResults['i_start']], ls = '--', color = 'orange', zorder = 4)
ax.axvline(Z[fitResults['i_stop']], ls = '--', color = 'red', zorder = 4)
ax.axhline((fitResults['F_moy'])*1e-3, ls = '-', color = 'k', zorder = 4)
ax.axhline((fitResults['F_moy']+fitResults['F_std'])*1e-3, ls = '--', color = 'k', zorder = 4)
labelFit1 = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(fitResults['Z01'], 
                                                                                               fitResults['covM1'][1,1]**0.5,
                                                                                               fitResults['K1'], 
                                                                                               fitResults['covM1'][0,0]**0.5,
                                                                                               fitResults['Rsq1'],)
ax.plot(fitResults['Z_fit1'], fitResults['F_fit1']*1e-3, lw=3,
        ls = '-', color = 'orange', zorder = 5, label = labelFit1)

labelFit2 = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(fitResults['Z02'], 
                                                                                       fitResults['covM2'][1,1]**0.5,
                                                                                       fitResults['K2'], 
                                                                                       fitResults['covM2'][0,0]**0.5,
                                                                                       fitResults['Rsq2'],)
ax.plot(fitResults['Z_fit2'], fitResults['F_fit2']*1e-3, 
        ls = '-', color = 'red', zorder = 7, label = labelFit2)

Z0_error, K_error = fitResults['Z0_error'], fitResults['K_error']
for j in range(len(Z0_error)):
    Z0, K = Z0_error[j], K_error[j]
    Zplot = np.linspace(Z0*1e3, Z[fitResults['i_stop']]*1e3, 100) * 1e-3
    Ffit = Hertz(Zplot, Z0, K, fitResults['R'], fitResults['F_moy'])
    ax.plot(Zplot, Ffit*1e-3, ls = '--', color = 'cyan', zorder = 4)
dZ0 = np.abs(Z0_error[1] - Z0_error[0])/2
labelFitErrors = 'Effect of error on Z0\n+/-{:.2f} µm =>\nYeff = {:.0f}~{:.0f} Pa'.format(dZ0, K_error[0], K_error[1])
ax.plot([], [], ls = '--', color = 'cyan', zorder = 6, label = labelFitErrors)
ax.legend()




#### Fitting the filtered compression

ax=axes[1]
colorList = gs.colorList10[:len(dI['ti'])]
i = 0
for ti, tf, c in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]):
    i += 1
    step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
    # print(step_indices)
    ax.plot(df.loc[step_indices, 'Z Tip - denoised']*1e-3, df.loc[step_indices, 'Load - denoised']*1e3, color=c,
            marker = '.', markersize = 2, ls = '')

ax.set_xlabel('Distance (µm)')
# ax.set_ylabel('Load (uN)')

# ax.axvline(Z[fitResults_denoised['i_start1']], ls = '--', color = 'yellow', zorder = 4)
# ax.axvline(Z[fitResults_denoised['i_start2']], ls = '--', color = 'orange', zorder = 4)
ax.axvline(Z_d[fitResults_denoised['i_start']], ls = '--', color = 'orange', zorder = 4)
ax.axvline(Z_d[fitResults_denoised['i_stop']], ls = '--', color = 'red', zorder = 4)
ax.axhline((fitResults_denoised['F_moy'])*1e-3, ls = '-', color = 'k', zorder = 4)
ax.axhline((fitResults_denoised['F_moy']+fitResults_denoised['F_std'])*1e-3, ls = '--', color = 'k', zorder = 4)
labelFit1 = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(fitResults_denoised['Z01'], 
                                                                                               fitResults_denoised['covM1'][1,1]**0.5,
                                                                                               fitResults_denoised['K1'], 
                                                                                               fitResults_denoised['covM1'][0,0]**0.5,
                                                                                               fitResults_denoised['Rsq1'],)
ax.plot(fitResults_denoised['Z_fit1'], fitResults_denoised['F_fit1']*1e-3, lw=3,
        ls = '-', color = 'orange', zorder = 5, label = labelFit1)
labelFit2 = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(fitResults_denoised['Z02'], 
                                                                                       fitResults_denoised['covM2'][1,1]**0.5,
                                                                                       fitResults_denoised['K2'], 
                                                                                       fitResults_denoised['covM2'][0,0]**0.5,
                                                                                       fitResults_denoised['Rsq2'],)
ax.plot(fitResults_denoised['Z_fit2'], fitResults_denoised['F_fit2']*1e-3, 
        ls = '-', color = 'red', zorder = 5, label = labelFit2)

Z0_error, K_error = fitResults_denoised['Z0_error'], fitResults_denoised['K_error']
for j in range(len(Z0_error)):
    Z0, K = Z0_error[j], K_error[j]
    Zplot = np.linspace(Z0*1e3, Z[fitResults_denoised['i_stop']]*1e3, 100) * 1e-3
    Ffit = Hertz(Zplot, Z0, K, fitResults_denoised['R'], fitResults_denoised['F_moy'])
    ax.plot(Zplot, Ffit*1e-3, ls = '--', color = 'cyan', zorder = 4)
dZ0 = np.abs(Z0_error[1] - Z0_error[0])/2
labelFitErrors = 'Effect of error on Z0\n+/-{:.2f} µm =>\nYeff = {:.0f}~{:.0f} Pa'.format(dZ0, K_error[0], K_error[1])
ax.plot([], [], ls = '--', color = 'cyan', zorder = 4, label = labelFitErrors)
ax.legend()



fig.suptitle(dI['indent_name'])
plt.tight_layout()
plt.show()
    
# %%% Multiple files analysis

# %%%% Functions V1

# fig_ft, ax_ft = plt.subplots(1, 1)

def FitHertz_V1(Z, F, R, mode = 'dZmax', fractionF = 0.4, dZmax = 3):
    ti, tf = dI['ti'][0], dI['tf'][0]
    compression_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
    Z = df.loc[compression_indices, 'Z Tip (nm)'].values*1e-3
    F = df.loc[compression_indices, 'Load (uN)'].values*1e6
    R = dI['R']

    mode = 'dZmax'
    fractionF = 0.4
    dZmax = 3

    early_points = 1000
    F_moy = np.median(F[:early_points])
    F_std = np.std(F[:early_points])
    F_max = np.max(F)

    Z_start = np.min(Z)
    Z_stop = np.max(Z)

    # Function
    def HertzFit(z, K, Z0):
        zeroInd = 1e-9 * np.ones_like(z)
        d = z - Z0
        d = np.where(d>0, d, zeroInd)
        f = (4/3) * K * (R**0.5) * (d**1.5) + F_moy
        return(f)

    #### First Fit Zone
    Z0_sup = np.max(Z)

    #### First Fit
    upper_threshold = F_moy + 0.75 * (F_max - F_moy)
    i_stop = ufun.findFirst(True, F >= upper_threshold)
    Z1, F1 = Z[:i_stop], F[:i_stop]
    binf = [0, 0]
    bsup = [5000, Z0_sup]
    p0=[100, 1] # 0.99*Z0_sup
    # Z1, F1 = Z[i_start1:i_stop], F[i_start1:i_stop]
    # Z1, F1 = Z, F
    [K1, Z01], covM1 = curve_fit(HertzFit, Z1, F1, p0=p0, bounds=(binf, bsup))
    Z_fit1 = np.linspace(np.min(Z), np.max(Z), 200)
    F_fit1 = HertzFit(Z_fit1, K1, Z01)
    Rsq1 = ufun.get_R2(F1, HertzFit(Z1, K1, Z01))

    #### Second Fit Zone
    i_start = ufun.findFirst(True, Z >= Z01 - 1) # 1 µm margin
    if mode == 'fractionF':
        upper_threshold = F_moy + fractionF * (F_max - F_moy)
        i_stop = ufun.findFirst(True, F >= upper_threshold)
    elif mode == 'dZmax':
        i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z01 + dZmax) - 1


    #### Second Fit (several iterations to converge to a consistent fit)
    Z_inf = Z01 - 3  # 2 µm margin
    Z0_approx = Z01
    for i in range(3):
        i_start = ufun.findFirst(True, Z >= Z_inf)
        if mode == 'fractionF':
            upper_threshold = F_moy + fractionF * (F_max - F_moy)
            i_stop = ufun.findFirst(True, F >= upper_threshold)
        elif mode == 'dZmax':
            i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z0_approx + dZmax) - 1
            
        binf = [0, 0]
        bsup = [5000, Z0_sup]
        p0=[K1, 1]
        
        Z2, F2 = Z[i_start:i_stop], F[i_start:i_stop]
        [K2, Z02], covM2 = curve_fit(HertzFit, Z2, F2, p0=p0, bounds=(binf, bsup))
        Z_fit2 = np.linspace(Z[i_start], Z[i_stop], 100)
        F_fit2 = HertzFit(Z_fit2, K2, Z02)
        Rsq2 = ufun.get_R2(F2, HertzFit(Z2, K2, Z02))
        
        Z_inf = Z02 - 3 # 2 µm margin for the next iteration
        Z0_approx = Z02
        
    i_start = ufun.findFirst(True, Z >= Z02)

    # #### Second Fit Refinement
    # i_start = ufun.findFirst(True, Z >= Z02 - 1.5)            
    # if mode == 'fractionF':
    #     upper_threshold = F_moy + fractionF * (F_max - F_moy)
    #     i_stop = ufun.findFirst(True, F >= upper_threshold)
    # elif mode == 'dZmax':
    #     i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z02 + dZmax) - 1
        
    # binf = [0, 0]
    # bsup = [5000, Z0_sup]
    # p0=[K2, Z02]

    # Z2, F2 = Z[i_start:i_stop], F[i_start:i_stop]
    # [K2, Z02], covM2 = curve_fit(HertzFit, Z2, F2, p0=p0, bounds=(binf, bsup))
    # Z_fit2 = np.linspace(Z[i_start], Z[i_stop], 100)
    # F_fit2 = HertzFit(Z_fit2, K2, Z02)
    # Rsq2 = ufun.get_R2(F2, HertzFit(Z2, K2, Z02))


    #### Error Fits
    arraySteps = np.array([-0.25, +0.25])
    Z0_error = arraySteps + Z02
    # print(Z0_test)
    resError = []
    for i in range(len(arraySteps)):
        Z0 = Z0_error[i]
        # function
        def HertzFit_1parm(z, K):
            zeroInd = 1e-9 * np.ones_like(z)
            d = z - Z0
            d = np.where(d>0, d, zeroInd)
            f = (4/3) * K * R**0.5 * (d)**1.5 + F_moy
            return(f)
        # zone
        i_start_test = len(Z) - ufun.findFirst(True, Z[::-1] <= Z0)
        # print(Z0, np.min(Z[i_start_test:i_stop]))
        # fit
        binf = [0]
        bsup = [np.inf]
        p0=[K2]
        [K_error], covM_error = curve_fit(HertzFit_1parm, Z[i_start_test:i_stop], F[i_start_test:i_stop], p0=p0, bounds=(binf, bsup))
        resError.append(K_error)
        
    # print(resError)
    K_error = np.array(resError)

    #### Results
    results = {'R':R,
                'mode':mode,
                'Z_start':Z_start,
                'Z_stop':Z_stop,
                'F_moy':F_moy,
                'F_std':F_std,
                'F_max':F_max,
                # 'i_start1':i_start1,
                # 'i_start2':i_start2,
                'i_start':i_start,
                'i_stop':i_stop,
                'K1':K1,
                'Z01':Z01,
                'covM1':covM1,
                'Rsq1':Rsq1,
                'Z_fit1':Z_fit1,
                'F_fit1':F_fit1,
                'K2':K2,
                'Z02':Z02,
                'covM2':covM2,
                'Rsq2':Rsq2,
                'Z_fit2':Z_fit2,
                'F_fit2':F_fit2,
                 'Z0_error':Z0_error,
                 'K_error':K_error,
                }
    
    return(results)



def plotFitHertz_V1(dictIndent, fitResults, show = True, save = False, savePath = ''):
    plt.ioff()
    
    def Hertz(z, Z0, K, R, F0):
        zeroInd = 1e-9 * np.ones_like(z)
        d = z - Z0
        d = np.where(d>0, d, zeroInd)
        f = (4/3) * K * R**0.5 * (d)**1.5 + F0
        return(f)
    
    df = dictIndent['df']
    df['Z Tip (nm)'] = df['Piezo (nm)'] - df['Cantilever (nm)']
    # R = dictIndent['R']
    fig, axes = plt.subplots(1,3, figsize = (18,6))
    ax=axes[0]
    ax.plot(df['Time (s)'], df['Piezo (nm)'], 'b-', label='Displacement')
    ax.plot(df['Time (s)'], df['Z Tip (nm)'], color='gold', ls='-', label='Z Tip')
    ax.plot(df['Time (s)'], df['Cantilever (nm)'], 'g-', label='Deflection')
    ax.plot(df['Time (s)'], df['Indentation (nm)'], 'r-', label='Indentation')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (nm)')
    ax.legend()
    
    ax=axes[1]
    colorList = gs.colorList10[:len(dI['ti'])]
    i = 0
    for ti, tf, c in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]):
        i += 1
        step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
        ax.plot(df.loc[step_indices, 'Z Tip (nm)']*1e-3, df.loc[step_indices, 'Load (uN)']*1e3, color=c,
                marker = '.', markersize = 2, ls = '')
    
    ax.set_xlabel('Distance (nm)')
    ax.set_ylabel('Load (uN)')
    
    ax.axvline(Z[fitResults['i_start']]*1e-3, ls = '--', color = 'orange', zorder = 4)
    ax.axvline(Z[fitResults['i_stop']]*1e-3, ls = '--', color = 'red', zorder = 4)
    ax.axhline((fitResults['F_moy'])*1e-3, ls = '-', color = 'k', zorder = 4)
    ax.axhline((fitResults['F_moy']+fitResults['F_std'])*1e-3, ls = '--', color = 'k', zorder = 4)
    labelFit1 = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(fitResults['Z01'], 
                                                                                                   fitResults['covM1'][1,1]**0.5,
                                                                                                   fitResults['K1'], 
                                                                                                   fitResults['covM1'][0,0]**0.5,
                                                                                                   fitResults['Rsq1'],)
    ax.plot(fitResults['Z_fit1'], fitResults['F_fit1']*1e-3, 
            ls = '-', color = 'orange', zorder = 5, label = labelFit1)
    labelFit2 = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(fitResults['Z02'], 
                                                                                           fitResults['covM2'][1,1]**0.5,
                                                                                           fitResults['K2'], 
                                                                                           fitResults['covM2'][0,0]**0.5,
                                                                                           fitResults['Rsq2'],)
    ax.plot(fitResults['Z_fit2'], fitResults['F_fit2']*1e-3, 
            ls = '-', color = 'red', zorder = 5, label = labelFit2)
    
    Z0_error, K_error = fitResults['Z0_error'], fitResults['K_error']
    for j in range(len(Z0_error)):
        Z0, K = Z0_error[j], K_error[j]
        Zplot = np.linspace(Z0*1e3, Z[fitResults['i_stop']], 100) * 1e-3
        Ffit = Hertz(Zplot, Z0, K, fitResults['R'], fitResults['F_moy'])
        ax.plot(Zplot, Ffit*1e-3, ls = '--', color = 'cyan', zorder = 4)
    dZ0 = np.abs(Z0_error[1] - Z0_error[0])/2
    labelFitErrors = 'Effect of error on Z0\n+/-{:.2f} µm =>\nYeff = {:.0f}~{:.0f} Pa'.format(dZ0, K_error[0], K_error[1])
    ax.plot([], [], ls = '--', color = 'cyan', zorder = 4, label = labelFitErrors)
    ax.legend()
    
    #### AX RESIDUALS
    
    ax=axes[2]
    compression_indices = df[(df['Time (s)'] >= dI['ti'][0]) & (df['Time (s)'] <= dI['tf'][0])].index
    Time = df.loc[compression_indices, 'Time (s)']
    Zmeas, Fmeas = df.loc[compression_indices, 'Z Tip (nm)']*1e-3, df.loc[compression_indices, 'Load (uN)']
    Ffit = Hertz(Zmeas, fitResults['Z02'], fitResults['K2'], fitResults['R'], fitResults['F_moy'])
    residuals = Ffit*1e-6 - Fmeas
    ax.plot(Zmeas[:fitResults['i_start']], residuals[:fitResults['i_start']]*1e3, 
            c = gs.colorList10[0], marker = '.', markersize = 2, ls = '')
    ax.plot(Zmeas[fitResults['i_start']:fitResults['i_stop']], residuals[fitResults['i_start']:fitResults['i_stop']]*1e3, 
            c = gs.colorList10[1], marker = '.', markersize = 2, ls = '')
    ax.plot(Zmeas[fitResults['i_stop']:], residuals[fitResults['i_stop']:]*1e3, 
            c = 'maroon', marker = '.', markersize = 2, ls = '')
    ax.axvline(Zmeas[fitResults['i_start']], ls = '--', color = 'orange', zorder = 4)
    ax.axvline(Zmeas[fitResults['i_stop']], ls = '--', color = 'red', zorder = 4)

    fig.suptitle(dI['indent_name'])
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close('all')
    if save:
        ufun.simpleSaveFig(fig, dI['indent_name'], savePath, '.png', 150)
        
    plt.ion()

# %%%% Script V1

date = '23-11-13'
manip = 'M1'

mainDir = f'D://MagneticPincherData//Raw//23.11.13_NanoIndent_ChiaroData//{manip}//'
fileList = [f for f in os.listdir(mainDir) if (os.path.isdir(mainDir + '//' + f) and len(f) == 2)]

filter_cell = ''
filter_cell = ''

resDir = {'date':[],
          'manip':[],
          'cell':[],
          'indent':[],
          'indentID':[],
          # 'path':[],
          'time':[],
          #
          'Z_start':[],
          'Z_stop':[],
          #
          'F_moy':[],
          'F_std':[],
          'F_max':[],
          #
          'K1':[],
          'Z01':[],
          'Rsq1':[],
          'K2':[],
          'Z02':[],
          'Rsq2':[],
          }

for f in fileList:
    dataDir = os.path.join(mainDir, f, 'indentations')
    cell = f
    
    
    # 1. Read all the Indentation files for that cell
    headerFormat = 'Time (s)	Load (uN)	Indentation (nm)	Cantilever (nm)	Piezo (nm)	Auxiliary'
    
    def getLineWithString(lines, string):
        i = 0
        while string not in lines[i]:
            i+=1
        return(lines[i], i)
        
    rawListFiles = os.listdir(dataDir)
    listIndent = []
    for f in rawListFiles:
        if f.endswith('.txt') and (filter_cell in f):
            indent_number = int(f[:-4].split('_')[-1])
            indent = f'I{indent_number:.0f}'
            fpath = os.path.join(dataDir, f)
            fopen = open(fpath, 'r')
            flines = fopen.readlines()
            s, i = getLineWithString(flines, headerFormat)
            metadataText = flines[:i]
            df_indent = pd.read_csv(fpath, skiprows = i-1, sep='\t')
            dict_indent = {'indent_name':f, 'path':fpath, 'metadata':metadataText, 'df':df_indent}
            
            dict_indent['date'] = date
            dict_indent['manip'] = manip
            dict_indent['cell'] = cell
            dict_indent['indent'] = indent
            
            time = flines[0].split('\t')[3]
            dict_indent['time'] = time
            R = float(getLineWithString(flines, 'Tip radius (um)')[0][16:])
            dict_indent['R'] = R
            
            text_start_times, j = getLineWithString(flines, 'Step absolute start times')
            text_end_times, j = getLineWithString(flines, 'Step absolute end times')
            start_times = np.array(text_start_times[30:].split(',')).astype(float)
            end_times = np.array(text_end_times[28:].split(',')).astype(float)
            
            dict_indent['ti'], dict_indent['tf'] = start_times, end_times
            
            listIndent.append(dict_indent)
            fopen.close()

    # 2. Analyze them
    for dI in listIndent[:]:
        df = dI['df']
        df['Z Tip (nm)'] = df['Piezo (nm)'] - df['Cantilever (nm)']
        R = dI['R']
        
        # i = 0
        # for ti, tf in zip(dI['ti'][::-1], dI['tf'][::-1]):
        #     i += 1
        #     step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
        
        try:
            ti, tf = dI['ti'][0], dI['tf'][0]
            compression_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
            Z = df.loc[compression_indices, 'Z Tip (nm)'].values
            F = df.loc[compression_indices, 'Load (uN)'].values
            
            fitResults = FitHertz_V1(Z * 1e-3, F * 1e6, R, mode = 'dZmax', dZmax = 3)  # nm to µm # µN to pN
            
            # def HertzFit(z, K, Z0):
            #     zeroInd = 1e-9 * np.ones_like(z)
            #     d = z - Z0
            #     d = np.where(d>0, d, zeroInd)
            #     f = (4/3) * K * (dict_indent['R']**0.5) * (d**1.5) + fitResults['F_moy']
            #     return(f)
            
            plotFitHertz_V1(dI, fitResults, show=True, save=True, savePath=mainDir)
            
            resDir['date'].append(date)
            resDir['manip'].append(manip)
            resDir['cell'].append(cell)
            resDir['indent'].append(dI['indent'])
            resDir['indentID'].append(f'{date}_{manip}_{cell}_{indent}')
            # resDir['path'].append(fpath)
            resDir['time'].append(dI['time'])
            resDir['Z_start'].append(fitResults['Z_start'])
            resDir['Z_stop'].append(fitResults['Z_stop'])
            resDir['F_moy'].append(fitResults['F_moy'])
            resDir['F_std'].append(fitResults['F_std'])
            resDir['F_max'].append(fitResults['F_max'])
            resDir['K1'].append(fitResults['K1'])
            resDir['Z01'].append(fitResults['Z01'])
            resDir['Rsq1'].append(fitResults['Rsq1'])
            resDir['K2'].append(fitResults['K2'])
            resDir['Z02'].append(fitResults['Z02'])
            resDir['Rsq2'].append(fitResults['Rsq2'])

        except:
            pass
        
resDf = pd.DataFrame(resDir)
resDf.to_csv(f'{mainDir}//{date}_{manip}_results.csv', sep=';')


# %%%% > Functions V2


def FitHertz_V2(dI, mode = 'dZmax', fractionF = 0.4, dZmax = 3, dZmax2 = 0.5,
                plot = 2, save_plot = False, save_path = ''):
    """
    Same as FitHertz_V1 with the filtering
    """
    if mode == 'dZmax':
        dZmax2 = dZmax
    
    
    df = dI['df']
    df['Z Tip (nm)'] = df['Piezo (nm)'] - df['Cantilever (nm)']
    R = dI['R']
    
    ti, tf = dI['ti'][0], dI['tf'][0]
    compression_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
    Z = df.loc[compression_indices, 'Z Tip (nm)'].values*1e-3 # nm to µm
    F = df.loc[compression_indices, 'Load (uN)'].values*1e6 # µN to pN
    Time = df.loc[compression_indices, 'Time (s)']
    # print('len(F)', len(F))

    early_points = len(F)//20
    F_moy = np.median(F[:early_points])
    F_std = np.std(F[:early_points])
    F_max = np.max(F)

    Z_start = np.min(Z)
    Z_stop = np.max(Z)

    # Fitting Function
    def HertzFit(z, K, Z0):
        zeroInd = 1e-9 * np.ones_like(z)
        d = z - Z0
        d = np.where(d>0, d, zeroInd)
        f = (4/3) * K * (R**0.5) * (d**1.5) + F_moy
        return(f)

    #### First Fit Zone
    Z0_sup = np.max(Z)

    #### First Fit
    upper_threshold = F_moy + 0.75 * (F_max - F_moy)
    i_stop = ufun.findFirst(True, F >= upper_threshold)
    Z1, F1 = Z[:i_stop], F[:i_stop]
    binf = [0, 0]
    bsup = [5000, Z0_sup]
    p0=[100, 1] # 0.99*Z0_sup
    [K1, Z01], covM1 = curve_fit(HertzFit, Z1, F1, p0=p0, bounds=(binf, bsup))
    Z_plotfit1 = np.linspace(np.min(Z), np.max(Z), 200)
    F_plotfit1 = HertzFit(Z_plotfit1, K1, Z01)
    Rsq1 = ufun.get_R2(F1, HertzFit(Z1, K1, Z01))



    #### Second Fit Zone
    i_start = ufun.findFirst(True, Z >= Z01 - 1) # 1 µm margin
    if mode == 'fractionF':
        upper_threshold = F_moy + fractionF * (F_max - F_moy)
        i_stop = ufun.findFirst(True, F >= upper_threshold)
    elif mode == 'dZmax':
        i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z01 + dZmax) - 1

    #### Second Fit (several iterations to converge to a consistent fit)
    N_it_fit = 3
    Z_inf = Z01 - 3  # 3 µm margin
    Z0_approx = Z01
    for i in range(N_it_fit):
        i_start = ufun.findFirst(True, Z >= Z_inf)
        if mode == 'fractionF':
            upper_threshold = F_moy + fractionF * (F_max - F_moy)
            i_stop = ufun.findFirst(True, F >= upper_threshold)
        elif mode == 'dZmax' or mode == 'dZmax_2values':
            i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z0_approx + dZmax) - 1
            
        binf = [0, 0]
        bsup = [5000, Z0_sup]
        p0=[K1, 1]
        
        Z2, F2 = Z[i_start:i_stop], F[i_start:i_stop]
        [K2, Z02], covM2 = curve_fit(HertzFit, Z2, F2, p0=p0, bounds=(binf, bsup))
        
        Rsq2 = ufun.get_R2(F2, HertzFit(Z2, K2, Z02))
        
        Z_inf = Z02 - 3 # 2 µm margin for the next iteration
        Z0_approx = Z02
    
    
    i_start = ufun.findFirst(True, Z >= Z02)
    
    
    Z_plotfit2 = np.linspace(Z[i_start], Z[i_stop], 100)
    F_plotfit2 = HertzFit(Z_plotfit2, K2, Z02)
    
    
    #### Residuals
    F_fit = HertzFit(Z, K2, Z02)
    residuals = F_fit - F
    residuals_before = residuals[:i_start]
    residuals_fit = residuals[i_start:i_stop]
    residuals_after = residuals[i_stop:]
    
    
    #### Fourier transforms
    timestep = 0.001
    fs = 1/timestep
    
    prominence1 = 0.2
    distance1 = 15
    
    # residuals_before_N = residuals_before - np.mean(residuals_before)
    # FT_before = np.abs(np.fft.fft(residuals_before_N))
    # n_before = FT_before.size
    # freq_before = np.fft.fftfreq(n_before, d=timestep)[:n_before//2]
    # FT_normed_before = FT_before[:n_before//2] / np.max(FT_before[:n_before//2])
    # peaks_before, peaks_prop_before = signal.find_peaks(FT_normed_before, prominence = prominence1, distance=distance1)
    
    # prominence2 = 0.05
    # distance2 = 15
    
    # residuals_fit_N = residuals_fit - np.mean(residuals_fit)
    # FT_fit = np.abs(np.fft.fft(residuals_fit_N))
    # n_fit = FT_fit.size
    # freq_fit = np.fft.fftfreq(n_fit, d=timestep)[:n_fit//2]
    # FT_normed_fit = FT_fit[:n_fit//2] / np.max(FT_fit[:n_fit//2])
    # peaks_fit, peaks_prop_fit = signal.find_peaks(FT_normed_fit, prominence = prominence2, distance=distance2)
    
    
    #### Denoising
    if mode == 'dZmax_2values':
        i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z02 + dZmax2) - 1
    
    Nfreq_to_cut = 3
    Deflect_fit = df['Cantilever (nm)'].values[i_start:i_stop]*1e-3 # nm to µm
    Deflect = df['Cantilever (nm)'].values[:]*1e-3
    Load = df['Load (uN)'].values[:]*1e6 # µN to pN
    
    prominence = 0.05
    distance = 15
    
    FT_D = np.abs(np.fft.fft(Deflect_fit - np.mean(Deflect_fit)))
    n_D = FT_D.size
    freq_D = np.fft.fftfreq(n_D, d=timestep)[:n_D//2]
    FT_normed_D = FT_D[:n_D//2] / np.max(FT_D[:n_D//2])
    peaks_D, peaks_prop_D = signal.find_peaks(FT_normed_D, prominence = prominence, distance=distance)

    dfp = pd.DataFrame(peaks_prop_D)
    dfp['freq'] = [freq_D[p] for p in peaks_D]
    dfp = dfp[dfp['freq'] > 10]
    dfp = dfp.sort_values(by='prominences', ascending=False)
    # print(dfp)

    Nfreq_to_cut = min(Nfreq_to_cut, len(dfp['freq'].values))
    noise_freqs = dfp['freq'][:Nfreq_to_cut]
    Deflect_denoised = np.copy(Deflect)
    Load_denoised = np.copy(Load)
    for nf in noise_freqs:
        a, b = signal.iirnotch(w0=nf, Q=10, fs=fs)
        Deflect_denoised = signal.lfilter(a, b, Deflect_denoised)
        Load_denoised = signal.lfilter(a, b, Load_denoised)

    df['Cantilever - denoised'] = Deflect_denoised*1e3 # µm to nm
    Z_denoised = df['Piezo (nm)'].values*1e-3 - Deflect_denoised # nm to µm
    df['Z Tip - denoised'] = Z_denoised*1e3 # µm to nm
    F_denoised = Load_denoised
    df['Load - denoised'] = F_denoised*1e-6 # pN to µN
    
    

    #### Denoised Fit
    
    if mode == 'dZmax_2values':
        binf = [0]
        bsup = [5000]
        p0=[K1]
        Z0_d = Z02
        def HertzFit_1parm(z, K):
            zeroInd = 1e-9 * np.ones_like(z)
            d = z - Z02
            d = np.where(d>0, d, zeroInd)
            f = (4/3) * K * R**0.5 * (d)**1.5 + F_moy
            return(f)
        
        [K_d], covM_d = curve_fit(HertzFit_1parm, Z_denoised[i_start:i_stop], F_denoised[i_start:i_stop],
                                p0=p0, bounds=(binf, bsup))
        
        print(covM_d)
        
        Var_K_d = covM_d[0,0]
        
        print(Var_K_d)
        
        covM_d = np.array([[Var_K_d, 0], [0, covM2[1,1]]])
        
        print(covM_d)
        
        
    else:
        binf = [0, 0]
        bsup = [5000, Z0_sup]
        p0=[K1, 1]
        [K_d, Z0_d], covM_d = curve_fit(HertzFit, Z_denoised[i_start:i_stop], F_denoised[i_start:i_stop]
                                     , p0=p0, bounds=(binf, bsup))
    
    # df.loc[step_indices, 'Z Tip (nm)']*1e-3, df.loc[step_indices, 'Load (uN)']*1e6*1e-3
    # Z_denoised[compression_indices], F_denoised[compression_indices]*1e-3    
    # print('a')
    # print(np.mean(Z2), np.mean(HertzFit(Z2, K2, Z02)), np.mean(F2), K2, Z02)    
    # print(np.mean(Z_denoised[i_start:i_stop]), np.mean(HertzFit(Z_denoised[i_start:i_stop], K_d, Z0_d)), np.mean(F_denoised[i_start:i_stop]), K_d, Z0_d)
   
    Rsq_d = ufun.get_R2(F_denoised[i_start:i_stop], HertzFit(Z_denoised[i_start:i_stop], K_d, Z0_d))
    
    F_fit_d = HertzFit(Z_denoised, Z0_d, K_d)
    
    Z_plotfit_d = np.linspace(Z_denoised[i_start], Z_denoised[i_stop], 100)
    F_plotfit_d = HertzFit(Z_plotfit_d, K_d, Z0_d)


    #### Error Fits
    arraySteps = np.array([-0.25, +0.25])
    Z0_error = arraySteps + Z02
    resError = []
    for i in range(len(arraySteps)):
        Z0 = Z0_error[i]
        # function
        def HertzFit_1parm(z, K):
            zeroInd = 1e-9 * np.ones_like(z)
            d = z - Z0
            d = np.where(d>0, d, zeroInd)
            f = (4/3) * K * R**0.5 * (d)**1.5 + F_moy
            return(f)
        # zone
        i_start_test = len(Z) - ufun.findFirst(True, Z[::-1] <= Z0)
        # fit
        binf = [0]
        bsup = [np.inf]
        p0=[K2]
        [K_error], covM_error = curve_fit(HertzFit_1parm, Z[i_start_test:i_stop], F[i_start_test:i_stop], p0=p0, bounds=(binf, bsup))
        resError.append(K_error)
    K_error = np.array(resError)
    
    #### Plot == 1
    if plot == 1:
        plt.ioff()
        fig, axes = plt.subplots(2,3, figsize = (12,6))
        colorList = gs.colorList10[:len(dI['ti'])]
        
        def Hertz(z, Z0, K, R, F0):
            zeroInd = 1e-9 * np.ones_like(z)
            d = z - Z0
            d = np.where(d>0, d, zeroInd)
            f = (4/3) * K * R**0.5 * (d)**1.5 + F0
            return(f)
        
        #### axes[0,0]
        ax = axes[0,0]
        ax.plot(df['Time (s)'], df['Piezo (nm)'], 'b-', label='Displacement')
        ax.plot(df['Time (s)'], df['Z Tip (nm)'], color='gold', ls='-', label='Z Tip')
        ax.plot(df['Time (s)'], df['Cantilever (nm)'], 'g-', label='Deflection')
        ax.plot(df['Time (s)'], df['Indentation (nm)'], 'r-', label='Indentation')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (nm)')
        ax.legend(loc='upper right')
        
        #### axes[0,1]
        ax = axes[0,1] # F in nN here : µN*1e6*1e-3 // pN*1e-3
        for ti, tf, c in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]):
            step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
            ax.plot(df.loc[step_indices, 'Z Tip (nm)']*1e-3, df.loc[step_indices, 'Load (uN)']*1e6*1e-3, color=c,
                    marker = '.', markersize = 2, ls = '')
        
        ax.set_xlabel('Distance (um)')
        ax.set_ylabel('Load (nN)')
        
        ax.axvline(Z[i_start], ls = '--', color = 'orange', zorder = 4)
        ax.axvline(Z[i_stop], ls = '--', color = 'red', zorder = 4)
        ax.axhline(F_moy*1e-3, ls = '-', color = 'k', zorder = 4)
        ax.axhline((F_moy+F_std)*1e-3, ls = '--', color = 'k', zorder = 4)
        labelFit1 = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\n$Y_{eff}$ = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(Z01, covM1[1,1]**0.5,
                                                                                                       K1, covM1[0,0]**0.5, Rsq1,)
        ax.plot(Z_plotfit1, F_plotfit1*1e-3, 
                ls = '-', color = 'orange', zorder = 5, label = labelFit1)
        
        labelFit2 = 'Second fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(Z02, covM2[1,1]**0.5,
                                                                                                       K2, covM2[0,0]**0.5, Rsq2,)
        ax.plot(Z_plotfit2, F_plotfit2*1e-3, 
                ls = '-', color = 'red', zorder = 5, label = labelFit2)
        
        # Z0_error, K_error = fitResults['Z0_error'], fitResults['K_error']
        # for j in range(len(Z0_error)):
        #     Z0, K = Z0_error[j], K_error[j]
        #     Zplot = np.linspace(Z0, Z[i_stop], 100)
        #     Ffit_error = Hertz(Zplot, Z0, K, R, F_moy)
        #     ax.plot(Zplot, Ffit_error*1e-3, ls = '--', color = 'cyan', zorder = 4)
        # dZ0 = np.abs(Z0_error[1] - Z0_error[0])/2
        # labelFitErrors = 'Effect of error on Z0\n+/-{:.2f} µm =>\nYeff = {:.0f}~{:.0f} Pa'.format(dZ0, K_error[0], K_error[1])
        # ax.plot([], [], ls = '--', color = 'cyan', zorder = 4, label = labelFitErrors)
        ax.legend(loc='upper left')


        #### axes[0,2]
        ax = axes[0,2] # Residuals in nN here : µN*1e6*1e-3 // pN*1e-3
        ax.set_xlabel('Distance (um)')
        ax.set_ylabel('Residuals (nN)')
        # Ffit = Hertz(Z, Z02, K2, R, F_moy)
        ax.plot(Z[:i_start], residuals_before*1e-3, 
                c = gs.colorList10[0], marker = '.', markersize = 2, ls = '')
        ax.plot(Z[i_start:i_stop], residuals_fit*1e-3, 
                c = gs.colorList10[1], marker = '.', markersize = 2, ls = '')
        ax.plot(Z[i_stop:], residuals_after*1e-3, 
                c = gs.colorList10[2], marker = '.', markersize = 2, ls = '')
        ax.axvline(Z[i_start], ls = '--', color = 'k', zorder = 4)
        ax.axvline(Z[i_stop], ls = '-.', color = 'k', zorder = 4)


        #### axes[1,0]
        ax = axes[1,0]
        
        # Nfreq_to_cut
        # noise_freqs
        ax.set_xlabel('Freq (Hz)')
        ax.set_ylabel('Amplitude')

        Deflect = df['Cantilever (nm)'].values[:]*1e-3 # µm to nm
        Deflect = Deflect - np.mean(Deflect)
        Deflect_filtered = np.copy(Deflect)
        for nf in noise_freqs:
            a, b = signal.iirnotch(w0=nf, Q=10, fs=fs)
            Deflect_filtered = signal.lfilter(a, b, Deflect_filtered)
        # FT raw
        FT_D = np.abs(np.fft.fft(Deflect))
        n_D = FT_D.size
        freq_D = np.fft.fftfreq(n_D, d=timestep)
        FT_D_max = np.max(FT_D[:n_D//2])
        # FT filtered
        FT_Df = np.abs(np.fft.fft(Deflect_filtered))
        n_Df = FT_Df.size
        freq_Df = np.fft.fftfreq(n_Df, d=timestep)
        FT_Df_max = np.max(FT_Df[:n_Df//2])
        
        ax.plot(freq_D[:n_Df//2], FT_D[:n_Df//2], label = 'Raw')
        ax.plot(freq_Df[:n_Df//2], FT_Df[:n_Df//2] - 0.2*FT_Df_max, label = 'Filtered')
        i = 2
        for freq in noise_freqs:
            color = gs.colorList10[i]
            ax.axvline(freq, c=color, ls='--', lw=1, zorder=1, label = f'{freq:.1f}') # 
            i = i+1
            
        ax.set_xlim([10, 150])
        ax.legend()


        #### axes[1,1]
        ax = axes[1,1] # F in nN here : µN*1e6*1e-3 // pN*1e-3
        ax.plot(Z_denoised[compression_indices], F_denoised[compression_indices]*1e-3, color=colorList[0],
                marker = '.', markersize = 2, ls = '')
        
        ax.set_xlabel('Distance (um)')
        ax.set_ylabel('Load (nN)')
        
        ax.axvline(Z_denoised[i_start], ls = '--', color = 'orange', zorder = 4)
        ax.axvline(Z_denoised[i_stop], ls = '--', color = 'red', zorder = 4)
        ax.axhline(F_moy*1e-3, ls = '-', color = 'k', zorder = 4)
        ax.axhline((F_moy+F_std)*1e-3, ls = '--', color = 'k', zorder = 4)
        labelFit_d = 'Denoised fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(Z0_d, covM_d[1,1]**0.5,
                                                                                                       K_d, covM_d[0,0]**0.5, Rsq_d,)
        ax.plot(Z_plotfit_d, F_plotfit_d*1e-3, 
                ls = '-', color = 'lime', zorder = 5, label = labelFit_d)

        ax.legend()
        
        
        #### axes[1,2]
        ax = axes[1,2] # F in nN here : µN*1e6*1e-3 // pN*1e-3
        ti, tf  = dI['ti'][0], dI['tf'][0]
        step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
        ax.plot(df.loc[step_indices, 'Z Tip (nm)']*1e-3, df.loc[step_indices, 'Load (uN)']*1e3, 
                c=gs.colorList40[10], marker = '.', markersize = 2, ls = '')
        ax.plot(df.loc[step_indices, 'Z Tip - denoised']*1e-3, df.loc[step_indices, 'Load - denoised']*1e3, 
                c=gs.colorList40[30], marker = '.', markersize = 2, ls = '')
        ax.plot([], [], 
                c=gs.colorList40[10], marker = 'o', markersize = 6, ls = '', label = 'Raw')
        ax.plot([], [],
                c=gs.colorList40[30], marker = 'o', markersize = 6, ls = '', label = 'Filtered')
        ax.set_xlabel('Distance (um)')
        ax.set_ylabel('Load (nN)')
        ax.legend()



        
        
    
    #### Plot == 2
    if plot == 2:
        plt.ioff()
        fig, axes = plt.subplots(2,2, figsize = (10,6))
        colorList = gs.colorList10[:len(dI['ti'])]
        
        def Hertz(z, Z0, K, R, F0):
            zeroInd = 1e-9 * np.ones_like(z)
            d = z - Z0
            d = np.where(d>0, d, zeroInd)
            f = (4/3) * K * R**0.5 * (d)**1.5 + F0
            return(f)
        
        #### axes[0,0]
        ax = axes[0,0]
        ax.plot(df['Time (s)'], df['Piezo (nm)'], 'b-', label='Displacement')
        ax.plot(df['Time (s)'], df['Z Tip (nm)'], color='gold', ls='-', label='Z Tip')
        ax.plot(df['Time (s)'], df['Cantilever (nm)'], 'g-', label='Deflection')
        ax.plot(df['Time (s)'], df['Indentation (nm)'], 'r-', label='Indentation')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (nm)')
        ax.legend(loc='upper right')
        
        #### axes[0,1]
        ax = axes[0,1] # F in nN here : µN*1e6*1e-3 // pN*1e-3
        for ti, tf, c in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]):
            step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
            ax.plot(df.loc[step_indices, 'Z Tip (nm)']*1e-3, df.loc[step_indices, 'Load (uN)']*1e6*1e-3, color=c,
                    marker = '.', markersize = 2, ls = '')
        
        ax.set_xlabel('Distance (um)')
        ax.set_ylabel('Load (nN)')
        
        ax.axvline(Z[i_start], ls = ':', color = 'k', lw=1, zorder = 4)
        ax.axvline(Z[i_stop], ls = '-.', color = 'k', lw=1, zorder = 4)
        ax.axhline(F_moy*1e-3, ls = '-', color = 'k', zorder = 4)
        ax.axhline((F_moy+F_std)*1e-3, ls = '--', color = 'k', zorder = 4)
        labelFit1 = f'First fit\nZ0 = {Z01:.2f} +/- {covM1[1,1]**0.5:.2f} µm\n'
        labelFit1 += f'Yeff = {K1:.0f} +/- {covM1[0,0]**0.5:.0f} Pa\nR² = {Rsq1:.3f}'
        # labelFit1 = f'First fit\n$Z_{0}$ = {:.2f} +/- {:.2f} µm\n$Y_{eff}$ = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(Z01, covM1[1,1]**0.5,
        #                                                                                                K1, covM1[0,0]**0.5, Rsq1,)
        ax.plot(Z_plotfit1, F_plotfit1*1e-3, 
                ls = '-', color = 'gold', zorder = 5, label = labelFit1)
        
        labelFit2 = f'Second fit\nZ0 = {Z02:.2f} +/- {covM2[1,1]**0.5:.2f} µm\n'
        labelFit2 += f'Yeff = {K2:.0f} +/- {covM2[0,0]**0.5:.0f} Pa\nR² = {Rsq2:.3f}'
        # labelFit2 = 'Second fit\n$Z_{0}$ = {:.2f} +/- {:.2f} µm\n$Y_{eff}$ = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(Z02, covM2[1,1]**0.5,
        #                                                                                                K2, covM2[0,0]**0.5, Rsq2,)
        ax.plot(Z_plotfit2, F_plotfit2*1e-3, 
                ls = '-', color = 'red', zorder = 5, label = labelFit2)
        
        # Z0_error, K_error = fitResults['Z0_error'], fitResults['K_error']
        # for j in range(len(Z0_error)):
        #     Z0, K = Z0_error[j], K_error[j]
        #     Zplot = np.linspace(Z0, Z[i_stop], 100)
        #     Ffit_error = Hertz(Zplot, Z0, K, R, F_moy)
        #     ax.plot(Zplot, Ffit_error*1e-3, ls = '--', color = 'cyan', zorder = 4)
        # dZ0 = np.abs(Z0_error[1] - Z0_error[0])/2
        # labelFitErrors = 'Effect of error on Z0\n+/-{:.2f} µm =>\n$Y_{eff}$ = {:.0f}~{:.0f} Pa'.format(dZ0, K_error[0], K_error[1])
        # ax.plot([], [], ls = '--', color = 'cyan', zorder = 4, label = labelFitErrors)
        ax.legend(loc='upper left')


        # #### axes[0,2]
        # ax = axes[0,2] # Residuals in nN here : µN*1e6*1e-3 // pN*1e-3
        # ax.set_xlabel('Distance (um)')
        # ax.set_ylabel('Residuals (nN)')
        # # Ffit = Hertz(Z, Z02, K2, R, F_moy)
        # ax.plot(Z[:i_start], residuals_before*1e-3, 
        #         c = gs.colorList10[0], marker = '.', markersize = 2, ls = '')
        # ax.plot(Z[i_start:i_stop], residuals_fit*1e-3, 
        #         c = gs.colorList10[1], marker = '.', markersize = 2, ls = '')
        # ax.plot(Z[i_stop:], residuals_after*1e-3, 
        #         c = gs.colorList10[2], marker = '.', markersize = 2, ls = '')
        # ax.axvline(Z[i_start], ls = '--', color = 'k', zorder = 4)
        # ax.axvline(Z[i_stop], ls = '-.', color = 'k', zorder = 4)


        #### axes[1,0]
        ax = axes[1,0]
        # labels = ['indent', 'hold', 'withdraw']
        # for ti, tf, c, lab in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]): # , labels
        #     step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
        #     ax.plot(df.loc[step_indices, 'Time (s)'], df.loc[step_indices, 'Load (uN)']*1e6*1e-3, 
        #             color=c, marker = '.', markersize = 2, ls = '')
        
        ax.plot(df['Time (s)'], df['Load (uN)']*1e6*1e-3, color='darkred', ls='-', label = 'Load')
        ax.axhline(F_moy*1e-3, ls = '-', color = 'k', lw=1, zorder = 4)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Load (nN)')
        ax.legend(loc='upper right')


        #### axes[1,1]
        ax = axes[1,1] # F in nN here : µN*1e6*1e-3 // pN*1e-3
        ax.plot(Z_denoised[compression_indices], F_denoised[compression_indices]*1e-3, color=colorList[0],
                marker = '.', markersize = 2, ls = '')
        
        ax.set_xlabel('Distance (um)')
        ax.set_ylabel('Load (nN)')
        
        ax.axvline(Z_denoised[i_start], ls = ':', color = 'k', lw=1, zorder = 4)
        ax.axvline(Z_denoised[i_stop], ls = '-.', color = 'k', lw=1, zorder = 4)
        ax.axhline(F_moy*1e-3, ls = '-', color = 'k', zorder = 4)
        ax.axhline((F_moy+F_std)*1e-3, ls = '--', color = 'k', zorder = 4)
        labelFit_d = f'Denoised fit\nZ0 = {Z0_d:.2f} +/- {covM_d[1,1]**0.5:.2f} µm\n'
        labelFit_d += f'Yeff = {K_d:.0f} +/- {covM_d[0,0]**0.5:.0f} Pa\nR² = {Rsq_d:.3f}'
        # labelFit_d = 'Denoised fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(Z0_d, covM_d[1,1]**0.5,
        #                                                                                                K_d, covM_d[0,0]**0.5, Rsq_d,)
        ax.plot(Z_plotfit_d, F_plotfit_d*1e-3, 
                ls = '-', color = 'lime', zorder = 5, label = labelFit_d)

        ax.legend()
        
        
        # #### axes[1,2]
        # ax = axes[1,2] # F in nN here : µN*1e6*1e-3 // pN*1e-3
        # ti, tf  = dI['ti'][0], dI['tf'][0]
        # step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
        # ax.plot(df.loc[step_indices, 'Z Tip (nm)']*1e-3, df.loc[step_indices, 'Load (uN)']*1e3, 
        #         c=gs.colorList40[10], marker = '.', markersize = 2, ls = '')
        # ax.plot(df.loc[step_indices, 'Z Tip - denoised']*1e-3, df.loc[step_indices, 'Load - denoised']*1e3, 
        #         c=gs.colorList40[30], marker = '.', markersize = 2, ls = '')
        # ax.plot([], [], 
        #         c=gs.colorList40[10], marker = 'o', markersize = 6, ls = '', label = 'Raw')
        # ax.plot([], [],
        #         c=gs.colorList40[30], marker = 'o', markersize = 6, ls = '', label = 'Filtered')
        # ax.set_xlabel('Distance (um)')
        # ax.set_ylabel('Load (nN)')
        # ax.legend()


    if plot >= 1:
        figtitle = f"{dI['date']}_{dI['manip']}_{dI['cell']}_{dI['indent']}_PlotType-0{plot}"
        fig.suptitle(figtitle)
        plt.tight_layout()
        show = True
        if show:
            plt.show()
        else:
            plt.close('all')
        if save_plot:
            ufun.simpleSaveFig(fig, figtitle, save_path, '.png', 150)
            
        plt.ion()
        
        
    
    
    

    #### Make Results
    results = {'R':R,
                'mode':mode,
                'fractionF':fractionF, 
                'dZmax':dZmax2,
                'Z_start':Z_start,
                'Z_stop':Z_stop,
                'F_moy':F_moy,
                'F_std':F_std,
                'F_max':F_max,
                # 'i_start1':i_start1,
                # 'i_start2':i_start2,
                'i_start':i_start,
                'i_stop':i_stop,
                'K1':K1,
                'Z01':Z01,
                'covM1':covM1,
                'Rsq1':Rsq1,
                'K2':K2,
                'Z02':Z02,
                'covM2':covM2,
                'Rsq2':Rsq2,
                'K_d':K_d,
                'Z0_d':Z0_d,
                'covM_d':covM_d,
                'Rsq_d':Rsq_d,
                'Z0_error':Z0_error,
                'K_error':K_error,
                }
    
    return(results)


# %%%% Script V2

date = '24-02-28'
# date = '23-12-10'
manip = 'M2'

# '23.12.10', 'M1', 'C3', '003'

def dateFormat(d):
    d2 = d[0:2]+'.'+d[3:5]+'.'+d[6:8]
    return(d2)

mainDir = f'D://MagneticPincherData//Raw//{dateFormat(date)}_NanoIndent_ChiaroData_3um//{manip}//'
fileList = [f for f in os.listdir(mainDir) if (os.path.isdir(mainDir + '//' + f) and len(f) in [2, 7, 9])]

filter_cell = ''

resDir = {'date':[],
          'manip':[],
          'cell':[],
          'indent':[],
          'indentID':[],
          # 'path':[],
          'time':[],
          #
          'Z_start':[],
          'Z_stop':[],
          #
          'F_moy':[],
          'F_std':[],
          'F_max':[],
          #
          'mode':[],
          'fractionF':[],
          'dZmax':[],
          'K1':[],
          'Z01':[],
          'Rsq1':[],
          'K2':[],
          'Z02':[],
          'Rsq2':[],
          'K_d':[],
          'Z0_d':[],
          'Rsq_d':[],
          }

for f in fileList:
    dataDir = os.path.join(mainDir, f, 'indentations')
    cell = f[:2]
    
    
    # 1. Read all the Indentation files for that cell
    headerFormat = 'Time (s)	Load (uN)	Indentation (nm)	Cantilever (nm)	Piezo (nm)	Auxiliary'
    
    def getLineWithString(lines, string):
        i = 0
        while string not in lines[i]:
            i+=1
        return(lines[i], i)
        
    rawListFiles = os.listdir(dataDir)
    listIndent = []
    for f in rawListFiles:
        if f.endswith('.txt') and (filter_cell in f):
            indent_number = int(f[:-4].split('_')[-1])
            indent = f'I{indent_number:.0f}'
            fpath = os.path.join(dataDir, f)
            fopen = open(fpath, 'r')
            flines = fopen.readlines()
            s, i = getLineWithString(flines, headerFormat)
            metadataText = flines[:i]
            df_indent = pd.read_csv(fpath, skiprows = i-1, sep='\t')
            dict_indent = {'indent_name':f, 'path':fpath, 'metadata':metadataText, 'df':df_indent}
            
            dict_indent['date'] = date
            dict_indent['manip'] = manip
            dict_indent['cell'] = cell
            dict_indent['indent'] = indent
            
            time = flines[0].split('\t')[3]
            dict_indent['time'] = time
            R = float(getLineWithString(flines, 'Tip radius (um)')[0][16:])
            dict_indent['R'] = R
            
            text_start_times, j = getLineWithString(flines, 'Step absolute start times')
            text_end_times, j = getLineWithString(flines, 'Step absolute end times')
            start_times = np.array(text_start_times[30:].split(',')).astype(float)
            end_times = np.array(text_end_times[28:].split(',')).astype(float)
            
            dict_indent['ti'], dict_indent['tf'] = start_times, end_times
            
            listIndent.append(dict_indent)
            fopen.close()

    # 2. Analyze them
    for dI in listIndent[:]:
        df = dI['df']
        df['Z Tip (nm)'] = df['Piezo (nm)'] - df['Cantilever (nm)']
        R = dI['R']
        
        # i = 0
        # for ti, tf in zip(dI['ti'][::-1], dI['tf'][::-1]):
        #     i += 1
        #     step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
        
        try:
            # fitResults = FitHertz_V2(dI, mode = 'dZmax', dZmax = 3)  # nm to µm # µN to pN
            fitResults = FitHertz_V2(dI, mode = 'dZmax_2values', dZmax = 3, dZmax2 = 1.0,
                                     plot = 2, save_plot = True, save_path = mainDir)
                        
            # plotFitHertz_V1(dI, fitResults, show=True, save=True, savePath=mainDir)
            
            resDir['date'].append(date)
            resDir['manip'].append(manip)
            resDir['cell'].append(cell)
            resDir['indent'].append(dI['indent'])
            resDir['indentID'].append(f'{date}_{manip}_{cell}_{indent}')
            # resDir['path'].append(fpath)
            resDir['time'].append(dI['time'])
            resDir['Z_start'].append(fitResults['Z_start'])
            resDir['Z_stop'].append(fitResults['Z_stop'])
            resDir['F_moy'].append(fitResults['F_moy'])
            resDir['F_std'].append(fitResults['F_std'])
            resDir['F_max'].append(fitResults['F_max'])
            resDir['mode'].append(fitResults['mode'])
            resDir['fractionF'].append(fitResults['fractionF'])
            resDir['dZmax'].append(fitResults['dZmax'])
            resDir['K1'].append(fitResults['K1'])
            resDir['Z01'].append(fitResults['Z01'])
            resDir['Rsq1'].append(fitResults['Rsq1'])
            resDir['K2'].append(fitResults['K2'])
            resDir['Z02'].append(fitResults['Z02'])
            resDir['Rsq2'].append(fitResults['Rsq2'])
            resDir['K_d'].append(fitResults['K_d'])
            resDir['Z0_d'].append(fitResults['Z0_d'])
            resDir['Rsq_d'].append(fitResults['Rsq_d'])

        except:
            pass
try:
    resDf = pd.DataFrame(resDir)
except:
    print(fitResults)
    for k in resDir.keys():
        print(k, ' ', len(resDir[k]))
resDf.to_csv(f'{mainDir}//{date}_{manip}_results.csv', sep=';')




# %% Debugging session

# %%% Analysis 1 by 1

#### Get the indentation

manip, sensitivity, speed, indent = 'P02_k0-45_R26', '10', 'fast', '001'
# 'P01_k0-021_R24-5' or 'P02_k0-45_R26' // 10 or 5 // very-fast, fast or slow // 001
d_speed = {'slow':1, 'fast':5, 'very-fast':10}
speed_umsec = d_speed[speed]

fpath = f'C://Users//JosephVermeil//Desktop//24-01-11_Debugging_session//End of session 2//{manip}//Sensitivity-{sensitivity}//test-{speed}//Indentations//test-{speed} Indentation_{indent}.txt'
f = fpath.split('//')[-1]
indentID = f'{manip}_S{sensitivity}_{speed}_I{indent}'

# resDir = {'manip':[],
#           'sensitivity':[],
#           'speed':[],
#           'indent':[],
#           'indentID':[],
#           # 'path':[],
#           'time':[],
#           #
#           'Z_start':[],
#           'Z_stop':[],
#           #
#           'F_moy':[],
#           'F_std':[],
#           'F_max':[],
#           #
#           'K1':[],
#           'Z01':[],
#           'Rsq1':[],
#           'K2':[],
#           'Z02':[],
#           'Rsq2':[],
#           }

headerFormat = 'Time (s)	Load (uN)	Indentation (nm)	Cantilever (nm)	Piezo (nm)	Auxiliary'

def getLineWithString(lines, string):
    i = 0
    while string not in lines[i]:
        i+=1
    return(lines[i], i)

indent_number = int(f[:-4].split('_')[-1])
indent = f'I{indent_number:.0f}'
fopen = open(fpath, 'r')
flines = fopen.readlines()
s, i = getLineWithString(flines, headerFormat)
metadataText = flines[:i]
df = pd.read_csv(fpath, skiprows = i-1, sep='\t')
df['Z Tip (nm)'] = df['Piezo (nm)'] - df['Cantilever (nm)']

dI = {'indentID':indentID, 'indent_name':f, 'path':fpath, 'metadata':metadataText, 'df':df}

dI['manip'] = manip
dI['sensitivity'] = sensitivity
dI['speed'] = speed
dI['indent'] = indent

time = flines[0].split('\t')[3]
dI['time'] = time
R = float(getLineWithString(flines, 'Tip radius (um)')[0][16:])
dI['R'] = R

text_start_times, j = getLineWithString(flines, 'Step absolute start times')
text_end_times, j = getLineWithString(flines, 'Step absolute end times')
start_times = np.array(text_start_times[30:].split(',')).astype(float)
end_times = np.array(text_end_times[28:].split(',')).astype(float)

dI['ti'], dI['tf'] = start_times, end_times

fopen.close()

#### Plot before fit

fig, axes = plt.subplots(1,2, figsize = (12,6))

ax=axes[0]
ax.plot(df['Time (s)'], df['Piezo (nm)'], 'b-', label='Displacement')
ax.plot(df['Time (s)'], df['Z Tip (nm)'], color='gold', ls='-', label='Z Tip')
ax.plot(df['Time (s)'], df['Cantilever (nm)'], 'g-', label='Deflection')
ax.plot(df['Time (s)'], df['Indentation (nm)'], 'r-', label='Indentation')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance (nm)')
ax.legend()

ax=axes[1]
colorList = gs.colorList10[:len(dI['ti'])]
i = 0
for ti, tf, c in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]):
    i += 1
    step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
    ax.plot(df.loc[step_indices, 'Z Tip (nm)'], df.loc[step_indices, 'Load (uN)'], color=c,
            marker = '.', markersize = 2, ls = '')

ax.set_xlabel('Distance (nm)')
ax.set_ylabel('Load (uN)')

fig.suptitle(dI['indentID'])
plt.tight_layout()
plt.show()


#### Segment the deflection in compression

ti, tf = dI['ti'][0], dI['tf'][0]
compression_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index

D = df.loc[compression_indices, 'Piezo (nm)'].values*1e-3
C = df.loc[compression_indices, 'Cantilever (nm)'].values*1e-3
Z = df.loc[compression_indices, 'Z Tip (nm)'].values*1e-3

R = dI['R']

mode = 'dZmax'
fractionF = 0.4
dZmax = 4

early_points = int(len(compression_indices)/5)
C_moy = np.median(C[:early_points])
C_std = np.std(C[:early_points])
C_max = np.max(C)

i_contact = len(C) - ufun.findFirst(True, C[::-1] <= C_moy + 2*C_std)

#### Plot the FT

D_signal = D[i_contact:]
Z_signal = Z[i_contact:] - np.mean(Z[i_contact:])

a, b = np.polyfit(D_signal, Z_signal, 1)

Z_signal_cor = Z_signal - (a*D_signal + b)

fig, axes = plt.subplots(1,3, figsize = (18,6))

timestep = 0.001
fst = 1/timestep

spacestep = timestep*speed_umsec
fss = 1/spacestep

ax=axes[0]
ax.plot(D_signal, Z_signal, 'g-', label='Z=f(D)')
ax.plot(D_signal, a*D_signal+b, 'r--', label='line fit')
ax.plot(D_signal, Z_signal_cor, c='orange', ls='-', label='Z_cor=f(D)')
ax.set_xlabel('Displacement Piezo (µm)')
ax.set_ylabel('Deflection measured (µm)')
ax.legend()

FT_deflect = np.abs(np.fft.fft(Z_signal_cor))
n = FT_deflect.size
freq_deflect = np.fft.fftfreq(n, d=spacestep)
FT_deflect = FT_deflect[:n//2]
freq_deflect = freq_deflect[:n//2]

ax=axes[1]
# n = n//2
ax.plot(freq_deflect, FT_deflect/np.max(FT_deflect), label = 'Raw')
ax.set_xlim([0, 40])


# FT_denoised = np.abs(np.fft.fft(denoised_deflect[fitResults['i_start']:fitResults['i_stop']] \
#                                 - np.mean(denoised_deflect[fitResults['i_start']:fitResults['i_stop']])))
# FT_denoised = FT_denoised[:n//2]
# n = n//2
# ax.plot(freq_deflect, FT_denoised/np.max(FT_deflect) - 0.2, label = 'Filtered')
# for f in noise_freqs:
#     ax.axvline(f, c='r', ls='--', lw=1, zorder=1)

ax.legend()

ax=axes[2]
ax.plot(freq_deflect, FT_deflect/np.max(FT_deflect), 'bo', label = 'Raw')
ax.set_xlim([0, 5])
ax.legend()

fig.suptitle(dI['indentID'])
plt.tight_layout()
plt.show()

# %%% Analysis by block

#### Get the indentation

manip, sensitivity = 'P02_k0-45_R26', '5'
# 'P01_k0-021_R24-5' or 'P02_k0-45_R26' // 10 or 5 // 001
speeds = ['slow', 'fast', 'very-fast']
indents = ['001', '001', '001']
d_speed = {'slow':1, 'fast':5, 'very-fast':10}

fig, AXES = plt.subplots(3,3, figsize = (12,12))
fig.suptitle(f'{manip}_S={sensitivity}')

for ii in range(len(speeds)):
    speed = speeds[ii]
    str_indent = indents[ii]
    # axes = AXES[i,:]
    speed_umsec = d_speed[speed]
    
    # def sec2um(t):
    #     return(t * speed_umsec)

    # def um2sec(x):
    #     return(x / speed_umsec)

    
    fpath = f'C://Users//JosephVermeil//Desktop//24-01-11_Debugging_session//End of session 2//{manip}//Sensitivity-{sensitivity}//test-{speed}//Indentations//test-{speed} Indentation_{str_indent}.txt'
    f = fpath.split('//')[-1]
    indentID = f'{manip}_S{sensitivity}_{speed}_I{str_indent}'
    
    headerFormat = 'Time (s)	Load (uN)	Indentation (nm)	Cantilever (nm)	Piezo (nm)	Auxiliary'
    
    def getLineWithString(lines, string):
        i = 0
        while string not in lines[i]:
            i+=1
        return(lines[i], i)
    
    indent_number = int(f[:-4].split('_')[-1])
    indent = f'I{indent_number:.0f}'
    fopen = open(fpath, 'r')
    flines = fopen.readlines()
    s, i = getLineWithString(flines, headerFormat)
    metadataText = flines[:i]
    df = pd.read_csv(fpath, skiprows = i-1, sep='\t')
    df['Z Tip (nm)'] = df['Piezo (nm)'] - df['Cantilever (nm)']
    
    dI = {'indentID':indentID, 'indent_name':f, 'path':fpath, 'metadata':metadataText, 'df':df}
    
    dI['manip'] = manip
    dI['sensitivity'] = sensitivity
    dI['speed'] = speed
    dI['indent'] = indent
    
    time = flines[0].split('\t')[3]
    dI['time'] = time
    R = float(getLineWithString(flines, 'Tip radius (um)')[0][16:])
    dI['R'] = R
    
    text_start_times, j = getLineWithString(flines, 'Step absolute start times')
    text_end_times, j = getLineWithString(flines, 'Step absolute end times')
    start_times = np.array(text_start_times[30:].split(',')).astype(float)
    end_times = np.array(text_end_times[28:].split(',')).astype(float)
    
    dI['ti'], dI['tf'] = start_times, end_times
    
    fopen.close()
    
    #### Plot before fit
    
    fig_1, axes_1 = plt.subplots(1,2, figsize = (12,6))
    
    ax=axes_1[0]
    ax.plot(df['Time (s)'], df['Piezo (nm)'], 'b-', label='Displacement')
    ax.plot(df['Time (s)'], df['Z Tip (nm)'], color='gold', ls='-', label='Z Tip')
    ax.plot(df['Time (s)'], df['Cantilever (nm)'], 'g-', label='Deflection')
    ax.plot(df['Time (s)'], df['Indentation (nm)'], 'r-', label='Indentation')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (nm)')
    ax.legend()
    
    ax=axes_1[1]
    colorList = gs.colorList10[:len(dI['ti'])]
    i = 0
    for ti, tf, c in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]):
        i += 1
        step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
        ax.plot(df.loc[step_indices, 'Z Tip (nm)'], df.loc[step_indices, 'Load (uN)'], color=c,
                marker = '.', markersize = 2, ls = '')
    
    ax.set_xlabel('Distance (nm)')
    ax.set_ylabel('Load (uN)')
    
    fig.suptitle(dI['indentID'])
    plt.tight_layout()
    plt.show()
    
    
    #### Segment the deflection in compression
    
    ti, tf = dI['ti'][0], dI['tf'][0]
    compression_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
    
    D = df.loc[compression_indices, 'Piezo (nm)'].values*1e-3
    C = df.loc[compression_indices, 'Cantilever (nm)'].values*1e-3
    Z = df.loc[compression_indices, 'Z Tip (nm)'].values*1e-3
    
    R = dI['R']
    
    mode = 'dZmax'
    fractionF = 0.4
    dZmax = 4
    
    early_points = int(len(compression_indices)/5)
    C_moy = np.median(C[:early_points])
    C_std = np.std(C[:early_points])
    C_max = np.max(C)
    
    i_contact = len(C) - ufun.findFirst(True, C[::-1] <= C_moy + 2*C_std)
    
    
    #### Plot the FT
    
    D_signal = D[i_contact:]
    Z_signal = Z[i_contact:] - np.mean(Z[i_contact:])
    a, b = np.polyfit(D_signal, Z_signal, 1)
    Z_signal_cor = Z_signal - (a*D_signal + b)
    
    timestep = 0.001
    fst = 1/timestep
    spacestep = timestep*speed_umsec
    fss = 1/spacestep
    
    ax=AXES[ii,0]
    ax.plot(D_signal, Z_signal, 'g-', label=f'Z=f(D) - {speed_umsec} um/s')
    # ax.plot(D_signal, a*D_signal+b, 'r--', label='line fit')
    # ax.plot(D_signal, Z_signal_cor, c='orange', ls='-', label='Z_cor=f(D)')
    ax.set_ylabel(f'{speed}\nDeflection measured (µm)')
    # secax = ax.secondary_xaxis('top', functions=(lambda x : x/speed_umsec, lambda x : x*speed_umsec))
    
    # if ii == 0:
    #     secax.set_xlabel('time (s)')
    if ii == len(speeds)-1:
        ax.set_xlabel('Displacement Piezo (µm)')
    ax.legend(loc='upper left')
    
    # Spatial FT
    FT_space = np.abs(np.fft.fft(Z_signal_cor))
    n = FT_space.size
    freq_space = np.fft.fftfreq(n, d=spacestep)
    FT_space = FT_space[:n//2]
    freq_space = freq_space[:n//2]
    
    ax=AXES[ii,1]
    ax.plot(freq_space, FT_space/np.max(FT_space), label = 'Spatial FFT Normalized')
    ax.set_xlim([0, 5])
    if ii == len(speeds)-1:
        ax.set_xlabel('Spatial freq (1/µm)')
    ax.legend()
    
    # Temporal FT
    FT_time = np.abs(np.fft.fft(Z_signal_cor))
    n = FT_time.size
    freq_time = np.fft.fftfreq(n, d=timestep)
    FT_time = FT_time[:n//2]
    freq_time = freq_time[:n//2]
    
    ax=AXES[ii,2]
    ax.plot(freq_time, FT_time/np.max(FT_time), label = 'Temporal FFT Normalized')
    ax.set_xlim([0, 120])
    ax.set_ylabel('Intensity')
    if ii == len(speeds)-1:
        ax.set_xlabel('freq (Hz)')
    ax.legend()


plt.tight_layout()
plt.show()


# %%%% Denoising

timestep = 0.001
fs = 1/timestep

Nfreq_to_cut = 5

dfp = pd.DataFrame(peaks_prop_fit)
dfp['freq'] = [freq_fit[p] for p in peaks_fit]
dfp = dfp[dfp['freq'] > 10]
dfp = dfp.sort_values(by='prominences', ascending=False)

Nfreq_to_cut = min(Nfreq_to_cut, len(dfp['freq'].values))
noise_freqs = dfp['freq'][:Nfreq_to_cut]

denoised_deflect = df['Cantilever (nm)'].values[:]
denoised_load = df['Load (uN)'].values[:]
for nf in noise_freqs:
    a, b = signal.iirnotch(w0=nf, Q=30, fs=fs)
    denoised_deflect = signal.lfilter(a, b, denoised_deflect)
    denoised_load = signal.lfilter(a, b, denoised_load)

df['Cantilever - denoised'] = denoised_deflect
df['Z Tip - denoised'] = df['Piezo (nm)'].values - denoised_deflect
df['Load - denoised'] = denoised_load

# %%%% Plot denoising

timestep = 0.001
fs = 1/timestep

# residuals_fit = residuals[fitResults['i_start']:fitResults['i_stop']]
# residuals_fit = residuals_fit - np.mean(residuals_fit)
# FT_fit = np.abs(np.fft.fft(residuals_fit))
# n_fit = FT_fit.size
# freq_fit = np.fft.fftfreq(n_fit, d=timestep)[:n_fit//2]
# FT_normed_fit = FT_fit[:n_fit//2] / np.max(FT_fit[:n_fit//2])
# peaks_fit, peaks_prop_fit = signal.find_peaks(FT_normed_fit, prominence = prominence2, distance=distance2)

deflect_fit = df['Cantilever (nm)'].values[fitResults['i_start']:fitResults['i_stop']]
load_fit = df['Load (uN)'].values[fitResults['i_start']:fitResults['i_stop']]

deflect_fit = deflect_fit - np.mean(deflect_fit)
load_fit = load_fit - np.mean(load_fit)

#### Fourier transform

fig, axes = plt.subplots(1,3, figsize = (16,6))

ax = axes[0]
FT_deflect = np.abs(np.fft.fft(deflect_fit))
n = FT_deflect.size
freq_deflect = np.fft.fftfreq(n, d=timestep)

FT_deflect = FT_deflect[:n//2]
freq_deflect = freq_deflect[:n//2]
# n = n//2
ax.plot(freq_deflect, FT_deflect/np.max(FT_deflect), label = 'Raw')

FT_denoised = np.abs(np.fft.fft(denoised_deflect[fitResults['i_start']:fitResults['i_stop']] \
                                - np.mean(denoised_deflect[fitResults['i_start']:fitResults['i_stop']])))
FT_denoised = FT_denoised[:n//2]
n = n//2
ax.plot(freq_deflect, FT_denoised/np.max(FT_deflect) - 0.2, label = 'Filtered')
for f in noise_freqs:
    ax.axvline(f, c='r', ls='--', lw=1, zorder=1)
ax.legend()

# ax = axes_ft[1]
# FT_load = np.abs(np.fft.fft(early_load))
# n = FT_load.size
# freq_load = np.fft.fftfreq(n, d=timestep)
# ax.plot(freq_load, FT_load/np.max(FT_load))

# peaks, peaks_prop = signal.find_peaks(FT_deflect/np.max(FT_deflect), prominence = 0.12)
# for i in peaks:
#     print(freq_deflect[i])

# freq2, Pxx = signal.welch(residuals_fit, fs = 1/timestep, nperseg = 2**9, noverlap = int(2**9*0.9), nfft = 2**12)
# ax_ft.semilogy(freq2, Pxx) #, ls='', marker='.', markersize = 3)

# peaks, peaks_prop = signal.find_peaks(FT_fit, height = 1, distance = 10)
# timestep = 0.001
# a, b = signal.iirnotch(w0=100, Q=50, fs=1/timestep)
# residuals_filter = signal.lfilter(a, b, residuals_fit)
# a, b = signal.iirnotch(w0=64.5, Q=50, fs=1/timestep)
# residuals_filter = signal.lfilter(a, b, residuals_filter)

# axes[2].plot(Zmeas_fit, residuals_filter, color = 'cyan',
#         marker = '.', markersize = 2, ls = '')

# timestep = 0.001
# FT_fit2 = np.abs(np.fft.fft(residuals_filter))
# n2 = FT_fit2.size
# freq2 = np.fft.fftfreq(n2, d=timestep)
# ax_ft.plot(freq2, FT_fit2) #, ls='', marker='.', markersize = 3)


ax=axes[1]
ax.plot(df['Time (s)'], df['Load (uN)']*1e3, c=gs.colorList40[12], marker = '.', markersize = 2, ls = '', label = 'Raw')
ax.plot(df['Time (s)'], df['Load - denoised']*1e3, c=gs.colorList40[32], marker = '.', markersize = 2, ls = '', label = 'Filtered')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Load (uN)')
ax.legend()

ax=axes[2]
colorList = gs.colorList10
ti, tf  = dI['ti'][0], dI['tf'][0]
step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
ax.plot(df.loc[step_indices, 'Z Tip (nm)'], df.loc[step_indices, 'Load (uN)']*1e3, 
        c=gs.colorList40[10], marker = '.', markersize = 2, ls = '', label = 'Raw')
ax.plot(df.loc[step_indices, 'Z Tip - denoised'], df.loc[step_indices, 'Load - denoised']*1e3, 
        c=gs.colorList40[30], marker = '.', markersize = 2, ls = '', label = 'Filtered')
ax.set_xlabel('Distance (nm)')
ax.set_ylabel('Load (nN)')
ax.legend()

fig.suptitle(dI['indent_name'])
plt.tight_layout()
plt.show()


# %%%% Analysis of the compression - denoised
ti, tf = dI['ti'][0], dI['tf'][0]
compression_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
Z = df.loc[compression_indices, 'Z Tip - denoised'].values*1e-3
F = df.loc[compression_indices, 'Load - denoised'].values*1e6

R = dI['R']
mode = 'dZmax'
fractionF = 0.4
dZmax = 4

early_points = 1000
F_moy = np.median(F[:early_points])
F_std = np.std(F[:early_points])
F_max = np.max(F)

Z_start = np.min(Z)
Z_stop = np.max(Z)

# Function
def HertzFit(z, K, Z0):
    zeroInd = 1e-9 * np.ones_like(z)
    d = z - Z0
    d = np.where(d>0, d, zeroInd)
    f = (4/3) * K * (R**0.5) * (d**1.5) + F_moy
    return(f)

#### First Fit Zone
Z0_sup = np.max(Z)

#### First Fit
upper_threshold = F_moy + 0.75 * (F_max - F_moy)
i_stop = ufun.findFirst(True, F >= upper_threshold)
Z1, F1 = Z[:i_stop], F[:i_stop]
binf = [0, 0]
bsup = [5000, Z0_sup]
p0=[100, 1] # 0.99*Z0_sup
# Z1, F1 = Z[i_start1:i_stop], F[i_start1:i_stop]
# Z1, F1 = Z, F
[K1, Z01], covM1 = curve_fit(HertzFit, Z1, F1, p0=p0, bounds=(binf, bsup))
Z_fit1 = np.linspace(np.min(Z), np.max(Z), 200)
F_fit1 = HertzFit(Z_fit1, K1, Z01)
Rsq1 = ufun.get_R2(F1, HertzFit(Z1, K1, Z01))

#### Second Fit Zone
i_start = ufun.findFirst(True, Z >= Z01 - 1) # 1 µm margin
if mode == 'fractionF':
    upper_threshold = F_moy + fractionF * (F_max - F_moy)
    i_stop = ufun.findFirst(True, F >= upper_threshold)
elif mode == 'dZmax':
    i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z01 + dZmax) - 1


#### Second Fit (several iterations to converge to a consistant fit)
Z_inf = Z01 - 3  # 2 µm margin
Z0_approx = Z01
for i in range(3):
    i_start = ufun.findFirst(True, Z >= Z_inf)
    if mode == 'fractionF':
        upper_threshold = F_moy + fractionF * (F_max - F_moy)
        i_stop = ufun.findFirst(True, F >= upper_threshold)
    elif mode == 'dZmax':
        i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z0_approx + dZmax) - 1
        
    binf = [0, 0]
    bsup = [5000, Z0_sup]
    p0=[K1, 1]
    
    Z2, F2 = Z[i_start:i_stop], F[i_start:i_stop]
    [K2, Z02], covM2 = curve_fit(HertzFit, Z2, F2, p0=p0, bounds=(binf, bsup))
    Z_fit2 = np.linspace(Z[i_start], Z[i_stop], 100)
    F_fit2 = HertzFit(Z_fit2, K2, Z02)
    Rsq2 = ufun.get_R2(F2, HertzFit(Z2, K2, Z02))
    
    Z_inf = Z02 - 3 # 2 µm margin for the next iteration
    Z0_approx = Z02
    
i_start = ufun.findFirst(True, Z >= Z02)

# #### Second Fit Refinement
# i_start = ufun.findFirst(True, Z >= Z02 - 1.5)            
# if mode == 'fractionF':
#     upper_threshold = F_moy + fractionF * (F_max - F_moy)
#     i_stop = ufun.findFirst(True, F >= upper_threshold)
# elif mode == 'dZmax':
#     i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z02 + dZmax) - 1
    
# binf = [0, 0]
# bsup = [5000, Z0_sup]
# p0=[K2, Z02]

# Z2, F2 = Z[i_start:i_stop], F[i_start:i_stop]
# [K2, Z02], covM2 = curve_fit(HertzFit, Z2, F2, p0=p0, bounds=(binf, bsup))
# Z_fit2 = np.linspace(Z[i_start], Z[i_stop], 100)
# F_fit2 = HertzFit(Z_fit2, K2, Z02)
# Rsq2 = ufun.get_R2(F2, HertzFit(Z2, K2, Z02))


#### Error Fits
arraySteps = np.array([-0.25, +0.25])
Z0_error = arraySteps + Z02
# print(Z0_test)
resError = []
for i in range(len(arraySteps)):
    Z0 = Z0_error[i]
    # function
    def HertzFit_1parm(z, K):
        zeroInd = 1e-9 * np.ones_like(z)
        d = z - Z0
        d = np.where(d>0, d, zeroInd)
        f = (4/3) * K * R**0.5 * (d)**1.5 + F_moy
        return(f)
    # zone
    i_start_test = len(Z) - ufun.findFirst(True, Z[::-1] <= Z0)
    # print(Z0, np.min(Z[i_start_test:i_stop]))
    # fit
    binf = [0]
    bsup = [np.inf]
    p0=[K2]
    [K_error], covM_error = curve_fit(HertzFit_1parm, Z[i_start_test:i_stop], F[i_start_test:i_stop], p0=p0, bounds=(binf, bsup))
    resError.append(K_error)
    
# print(resError)
K_error = np.array(resError)

#### Results
fitResults_denoised = {'R':R,
               'mode':mode,
               'Z_start':Z_start,
               'Z_stop':Z_stop,
               'F_moy':F_moy,
               'F_std':F_std,
               'F_max':F_max,
               # 'i_start1':i_start1,
               # 'i_start2':i_start2,
               'i_start':i_start,
               'i_stop':i_stop,
               'K1':K1,
               'Z01':Z01,
               'covM1':covM1,
               'Rsq1':Rsq1,
               'Z_fit1':Z_fit1,
               'F_fit1':F_fit1,
               'K2':K2,
               'Z02':Z02,
               'covM2':covM2,
               'Rsq2':Rsq2,
               'Z_fit2':Z_fit2,
               'F_fit2':F_fit2,
                'Z0_error':Z0_error,
                'K_error':K_error,
               }


# %%%% Plot fit on raw vs. denoised

Z = df.loc[compression_indices, 'Z Tip (nm)'].values*1e-3
F = df.loc[compression_indices, 'Load (uN)'].values*1e6
Z_d = df.loc[compression_indices, 'Z Tip - denoised'].values*1e-3
F_d = df.loc[compression_indices, 'Load - denoised'].values*1e6

def Hertz(z, Z0, K, R, F0):
    zeroInd = 1e-9 * np.ones_like(z)
    d = z - Z0
    d = np.where(d>0, d, zeroInd)
    f = (4/3) * K * R**0.5 * (d)**1.5 + F0
    return(f)


fig, axes = plt.subplots(1,2, figsize = (13,6), sharey=True)

#### Fitting the raw compression

ax=axes[0]
colorList = gs.colorList10[:len(dI['ti'])]
i = 0
for ti, tf, c in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]):
    i += 1
    step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
    # print(step_indices)
    ax.plot(df.loc[step_indices, 'Z Tip (nm)']*1e-3, df.loc[step_indices, 'Load (uN)']*1e3, color=c,
            marker = '.', markersize = 2, ls = '')

ax.set_xlabel('Distance (µm)')
ax.set_ylabel('Load (nN)')

# ax.axvline(Z[fitResults['i_start1']], ls = '--', color = 'yellow', zorder = 4)
# ax.axvline(Z[fitResults['i_start2']], ls = '--', color = 'orange', zorder = 4)
ax.axvline(Z[fitResults['i_start']], ls = '--', color = 'orange', zorder = 4)
ax.axvline(Z[fitResults['i_stop']], ls = '--', color = 'red', zorder = 4)
ax.axhline((fitResults['F_moy'])*1e-3, ls = '-', color = 'k', zorder = 4)
ax.axhline((fitResults['F_moy']+fitResults['F_std'])*1e-3, ls = '--', color = 'k', zorder = 4)
labelFit1 = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(fitResults['Z01'], 
                                                                                               fitResults['covM1'][1,1]**0.5,
                                                                                               fitResults['K1'], 
                                                                                               fitResults['covM1'][0,0]**0.5,
                                                                                               fitResults['Rsq1'],)
ax.plot(fitResults['Z_fit1'], fitResults['F_fit1']*1e-3, lw=3,
        ls = '-', color = 'orange', zorder = 5, label = labelFit1)

labelFit2 = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(fitResults['Z02'], 
                                                                                       fitResults['covM2'][1,1]**0.5,
                                                                                       fitResults['K2'], 
                                                                                       fitResults['covM2'][0,0]**0.5,
                                                                                       fitResults['Rsq2'],)
ax.plot(fitResults['Z_fit2'], fitResults['F_fit2']*1e-3, 
        ls = '-', color = 'red', zorder = 7, label = labelFit2)

Z0_error, K_error = fitResults['Z0_error'], fitResults['K_error']
for j in range(len(Z0_error)):
    Z0, K = Z0_error[j], K_error[j]
    Zplot = np.linspace(Z0*1e3, Z[fitResults['i_stop']]*1e3, 100) * 1e-3
    Ffit = Hertz(Zplot, Z0, K, fitResults['R'], fitResults['F_moy'])
    ax.plot(Zplot, Ffit*1e-3, ls = '--', color = 'cyan', zorder = 4)
dZ0 = np.abs(Z0_error[1] - Z0_error[0])/2
labelFitErrors = 'Effect of error on Z0\n+/-{:.2f} µm =>\nYeff = {:.0f}~{:.0f} Pa'.format(dZ0, K_error[0], K_error[1])
ax.plot([], [], ls = '--', color = 'cyan', zorder = 6, label = labelFitErrors)
ax.legend()




#### Fitting the filtered compression

ax=axes[1]
colorList = gs.colorList10[:len(dI['ti'])]
i = 0
for ti, tf, c in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]):
    i += 1
    step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
    # print(step_indices)
    ax.plot(df.loc[step_indices, 'Z Tip - denoised']*1e-3, df.loc[step_indices, 'Load - denoised']*1e3, color=c,
            marker = '.', markersize = 2, ls = '')

ax.set_xlabel('Distance (µm)')
# ax.set_ylabel('Load (uN)')

# ax.axvline(Z[fitResults_denoised['i_start1']], ls = '--', color = 'yellow', zorder = 4)
# ax.axvline(Z[fitResults_denoised['i_start2']], ls = '--', color = 'orange', zorder = 4)
ax.axvline(Z_d[fitResults_denoised['i_start']], ls = '--', color = 'orange', zorder = 4)
ax.axvline(Z_d[fitResults_denoised['i_stop']], ls = '--', color = 'red', zorder = 4)
ax.axhline((fitResults_denoised['F_moy'])*1e-3, ls = '-', color = 'k', zorder = 4)
ax.axhline((fitResults_denoised['F_moy']+fitResults_denoised['F_std'])*1e-3, ls = '--', color = 'k', zorder = 4)
labelFit1 = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(fitResults_denoised['Z01'], 
                                                                                               fitResults_denoised['covM1'][1,1]**0.5,
                                                                                               fitResults_denoised['K1'], 
                                                                                               fitResults_denoised['covM1'][0,0]**0.5,
                                                                                               fitResults_denoised['Rsq1'],)
ax.plot(fitResults_denoised['Z_fit1'], fitResults_denoised['F_fit1']*1e-3, lw=3,
        ls = '-', color = 'orange', zorder = 5, label = labelFit1)
labelFit2 = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa\nR² = {:.3f}'.format(fitResults_denoised['Z02'], 
                                                                                       fitResults_denoised['covM2'][1,1]**0.5,
                                                                                       fitResults_denoised['K2'], 
                                                                                       fitResults_denoised['covM2'][0,0]**0.5,
                                                                                       fitResults_denoised['Rsq2'],)
ax.plot(fitResults_denoised['Z_fit2'], fitResults_denoised['F_fit2']*1e-3, 
        ls = '-', color = 'red', zorder = 5, label = labelFit2)

Z0_error, K_error = fitResults_denoised['Z0_error'], fitResults_denoised['K_error']
for j in range(len(Z0_error)):
    Z0, K = Z0_error[j], K_error[j]
    Zplot = np.linspace(Z0*1e3, Z[fitResults_denoised['i_stop']]*1e3, 100) * 1e-3
    Ffit = Hertz(Zplot, Z0, K, fitResults_denoised['R'], fitResults_denoised['F_moy'])
    ax.plot(Zplot, Ffit*1e-3, ls = '--', color = 'cyan', zorder = 4)
dZ0 = np.abs(Z0_error[1] - Z0_error[0])/2
labelFitErrors = 'Effect of error on Z0\n+/-{:.2f} µm =>\nYeff = {:.0f}~{:.0f} Pa'.format(dZ0, K_error[0], K_error[1])
ax.plot([], [], ls = '--', color = 'cyan', zorder = 4, label = labelFitErrors)
ax.legend()



fig.suptitle(dI['indent_name'])
plt.tight_layout()
plt.show()




# %% Tests against bare glass

# %%% Probe A - k=0.021N/m, R=24.5um

date, manip, indent = '23.12.10', 'test_bare_glass_probe_0021', '002'

fpath = f'D://MagneticPincherData//Raw//{date}_NanoIndent_ChiaroData//{manip}//Indentations//{manip} Indentation_{indent}.txt'
f = fpath.split('//')[-1]

resDir = {'date':[],
          'manip':[],
          'indent':[],
          'indentID':[],
          # 'path':[],
          'time':[],
          #
          'Z_start':[],
          'Z_stop':[],
          #
          'F_moy':[],
          'F_std':[],
          'F_max':[],
          #
          'K1':[],
          'Z01':[],
          'Rsq1':[],
          'K2':[],
          'Z02':[],
          'Rsq2':[],
          }

headerFormat = 'Time (s)	Load (uN)	Indentation (nm)	Cantilever (nm)	Piezo (nm)	Auxiliary'

def getLineWithString(lines, string):
    i = 0
    while string not in lines[i]:
        i+=1
    return(lines[i], i)

indent_number = int(f[:-4].split('_')[-1])
indent = f'I{indent_number:.0f}'
fopen = open(fpath, 'r')
flines = fopen.readlines()
s, i = getLineWithString(flines, headerFormat)
metadataText = flines[:i]
df = pd.read_csv(fpath, skiprows = i-1, sep='\t')
df['Z Tip (nm)'] = df['Piezo (nm)'] - df['Cantilever (nm)']

dI = {'indent_name':f, 'path':fpath, 'metadata':metadataText, 'df':df}

dI['date'] = date
dI['manip'] = manip
dI['indent'] = indent

time = flines[0].split('\t')[3]
dI['time'] = time
R = float(getLineWithString(flines, 'Tip radius (um)')[0][16:])
dI['R'] = R

text_start_times, j = getLineWithString(flines, 'Step absolute start times')
text_end_times, j = getLineWithString(flines, 'Step absolute end times')
start_times = np.array(text_start_times[30:].split(',')).astype(float)
end_times = np.array(text_end_times[28:].split(',')).astype(float)

dI['ti'], dI['tf'] = start_times, end_times

fopen.close()

# %%%% Plot before fit

fig, axes = plt.subplots(1,2, figsize = (12,6))

ax=axes[0]
ax.plot(df['Time (s)'], df['Piezo (nm)'], 'b-', label='Displacement')
ax.plot(df['Time (s)'], df['Z Tip (nm)'], color='gold', ls='-', label='Z Tip')
ax.plot(df['Time (s)'], df['Cantilever (nm)'], 'g-', label='Deflection')
ax.plot(df['Time (s)'], df['Indentation (nm)'], 'r-', label='Indentation')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance (nm)')
ax.legend()

ax=axes[1]
colorList = gs.colorList10[:len(dI['ti'])]
i = 0
for ti, tf, c in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]):
    i += 1
    step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
    ax.plot(df.loc[step_indices, 'Z Tip (nm)'], df.loc[step_indices, 'Load (uN)'], color=c,
            marker = '.', markersize = 2, ls = '')

ax.set_xlabel('Distance (nm)')
ax.set_ylabel('Load (uN)')

fig.suptitle(dI['indent_name'])
plt.tight_layout()
plt.show()



# %%% Probe B - k=0.016N/m, R=26.5um

date, manip, indent = '23.12.10', 'test_bare_glass_probe_0016', '004'

fpath = f'D://MagneticPincherData//Raw//{date}_NanoIndent_ChiaroData//{manip}//Indentations//{manip} Indentation_{indent}.txt'
f = fpath.split('//')[-1]

resDir = {'date':[],
          'manip':[],
          'indent':[],
          'indentID':[],
          # 'path':[],
          'time':[],
          #
          'Z_start':[],
          'Z_stop':[],
          #
          'F_moy':[],
          'F_std':[],
          'F_max':[],
          #
          'K1':[],
          'Z01':[],
          'Rsq1':[],
          'K2':[],
          'Z02':[],
          'Rsq2':[],
          }

headerFormat = 'Time (s)	Load (uN)	Indentation (nm)	Cantilever (nm)	Piezo (nm)	Auxiliary'

def getLineWithString(lines, string):
    i = 0
    while string not in lines[i]:
        i+=1
    return(lines[i], i)

indent_number = int(f[:-4].split('_')[-1])
indent = f'I{indent_number:.0f}'
fopen = open(fpath, 'r')
flines = fopen.readlines()
s, i = getLineWithString(flines, headerFormat)
metadataText = flines[:i]
df = pd.read_csv(fpath, skiprows = i-1, sep='\t')
df['Z Tip (nm)'] = df['Piezo (nm)'] - df['Cantilever (nm)']

dI = {'indent_name':f, 'path':fpath, 'metadata':metadataText, 'df':df}

dI['date'] = date
dI['manip'] = manip
dI['indent'] = indent

time = flines[0].split('\t')[3]
dI['time'] = time
R = float(getLineWithString(flines, 'Tip radius (um)')[0][16:])
dI['R'] = R

text_start_times, j = getLineWithString(flines, 'Step absolute start times')
text_end_times, j = getLineWithString(flines, 'Step absolute end times')
start_times = np.array(text_start_times[30:].split(',')).astype(float)
end_times = np.array(text_end_times[28:].split(',')).astype(float)

dI['ti'], dI['tf'] = start_times, end_times

fopen.close()

# %%%% Plot before fit

fig, axes = plt.subplots(1,2, figsize = (12,6))

ax=axes[0]
ax.plot(df['Time (s)'], df['Piezo (nm)'], 'b-', label='Displacement')
ax.plot(df['Time (s)'], df['Z Tip (nm)'], color='gold', ls='-', label='Z Tip')
ax.plot(df['Time (s)'], df['Cantilever (nm)'], 'g-', label='Deflection')
ax.plot(df['Time (s)'], df['Indentation (nm)'], 'r-', label='Indentation')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance (nm)')
ax.legend()

ax=axes[1]
colorList = gs.colorList10[:len(dI['ti'])]
i = 0
for ti, tf, c in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]):
    i += 1
    step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
    ax.plot(df.loc[step_indices, 'Z Tip (nm)'], df.loc[step_indices, 'Load (uN)'], color=c,
            marker = '.', markersize = 2, ls = '')

ax.set_xlabel('Distance (nm)')
ax.set_ylabel('Load (uN)')

fig.suptitle(dI['indent_name'])
plt.tight_layout()
plt.show()



# %% VCI Macro-indents
# %%% Imports

dataDir = "D://MagneticPincherData//Raw//23.10.10_MacroIndenter_VCI"

D1, D2, D3 = 1.401, 2.73, 4.004
R1, R2, R3 = D1/2, D2/2, D3/2
f1, f2, f3 = 'Specimen_RawData_2.csv', 'Specimen_RawData_4.csv', 'Specimen_RawData_5.csv'
rD = {'Temps' : 'T', 'Deplacement' : 'Z', 'Charge' : 'F'}
df1 = pd.read_csv(dataDir + '//' + f1, engine = 'python', sep = ';', skiprows=[1]).rename(columns = rD)
df2 = pd.read_csv(dataDir + '//' + f2, engine = 'python', sep = ';', skiprows=[1]).rename(columns = rD)
df3 = pd.read_csv(dataDir + '//' + f3, engine = 'python', sep = ';', skiprows=[1]).rename(columns = rD)
list_df = [df1, df2, df3]
list_R = [R1, R2, R3]

# Reduced Young Modulus : Y = E / (1 - v**2)
# F = (4/3) * Y * R**0.5 * h**1.5


# %%% First one

dataDir = "D://MagneticPincherData//Raw//23.10.10_MacroIndenter_VCI"

D = 1.401
R = D/2
f = 'Specimen_RawData_2.csv'
rD = {'Temps' : 'T', 'Deplacement' : 'Z', 'Charge' : 'F'}
df = pd.read_csv(dataDir + '//' + f, engine = 'python', sep = ';', skiprows=[1]).rename(columns = rD)

early_points = 30

dict_curves = {}

imax = signal.find_peaks(df['Z'], distance=150)[0]
imin = signal.find_peaks((-1) * df['Z'], distance=150)[0]
imin = [0, imin[0], len(df)]

Ncycles = len(imax)
fig, axes = plt.subplots(2, Ncycles, figsize = (8, 6))
fig.suptitle('Curves for D = {:.3f} mm'.format(D))

for j in range(len(imax)):
    dict_curves[2*j] = {}
    dict_curves[2*j]['type'] = 'comp'
    dict_curves[2*j]['num'] = j+1
    dict_curves[2*j]['df'] = df.iloc[imin[j]:imax[j], :]
    
    dict_curves[2*j + 1] = {}
    dict_curves[2*j + 1]['type'] = 'relax'
    dict_curves[2*j + 1]['num'] = j+1
    dict_curves[2*j + 1]['df'] = df.iloc[imax[j]:imin[j+1], :]
    
for i in range(len(dict_curves)):
    i_ax = int(dict_curves[i]['type'] == 'relax')
    j_ax = i//2
    try:
        ax = axes[i_ax, j_ax]
    except:
        ax = axes[i_ax]
        
    ax.set_title(dict_curves[i]['type'] + ' ' + str(dict_curves[i]['num']))
    if dict_curves[i]['type'] == 'comp':
        F = dict_curves[i]['df']['F'].values
        Z = dict_curves[i]['df']['Z'].values
    elif dict_curves[i]['type'] == 'relax':
        F = dict_curves[i]['df']['F'].values[::-1]
        Z = dict_curves[i]['df']['Z'].values[::-1]
    
    F_moy = np.average(F[:early_points])
    F_std = np.std(F[:early_points])
    
    #
    F = F - F_moy
    F_moy = 0
    #
    
    ax.plot(Z, F)
    ax.set_xlabel('Z (mm)')
    ax.set_ylabel('F (N)')
    ax.grid(axis='y')
    
    #` Function
    
    def HertzFit(z, K, Z0):
        f = (4/3) * K * R**0.5 * (z - Z0)**1.5
        return(f)
    
    #### First Fit
    
    i_start = len(F) - ufun.findFirst(True, F[::-1] < F_moy + 1 * F_std)
    Z0_sup = Z[i_start]
    
    # ax.axhline(F_moy, color = 'orange', ls = '--')
    # ax.axhline(F_moy - 1 * F_std, color = 'cyan', ls = '--')
    # ax.axhline(F_moy + 1 * F_std, color = 'cyan', ls = '--', zorder = 5)
    # ax.axvline(Z[i_start], ls = '--', color = 'orange', zorder = 5)

    DZmax = 0.1
    i_stop = ufun.findFirst(True, Z >= Z0_sup + DZmax)
    
    ax.plot(Z[i_start:i_stop], F[i_start:i_stop], color = 'green', zorder = 4)
    
    binf = [0, 0]
    bsup = [np.inf, Z[i_start]]
    p0=[0, 0.99*Z[i_start]]
    [K, Z0], covM = curve_fit(HertzFit, Z[i_start:i_stop], F[i_start:i_stop], p0=p0, bounds=(binf, bsup))
    
    Z_fit = np.linspace(Z0, Z[i_stop], 100)
    F_fit = HertzFit(Z_fit, K, Z0)
    ax.plot(Z_fit, F_fit, color = 'orange', ls = '--', zorder = 6, label = '1st Hertz fit, K = {:.0f}kPa, z0 = {:.3f}mm'.format(K*1e3, Z0))
    ax.legend()
    
    #### Second Fit
    
    i_start_bis = ufun.findFirst(True, Z >= Z0)

    # ax.axvline(Z[i_start], ls = '--', color = 'red', zorder = 5)
   
    ax.plot(Z[i_start_bis:i_stop], F[i_start_bis:i_stop], color = 'turquoise', zorder = 3)
    
    binf = [0, 0]
    bsup = [np.inf, Z0_sup]
    p0=[0, Z0]

    [K2, Z02], covM = curve_fit(HertzFit, Z[i_start_bis:i_stop], F[i_start_bis:i_stop], p0=[K, Z0])
    
    Z_fit = np.linspace(Z02, Z[i_stop], 100)
    F_fit = HertzFit(Z_fit, K2, Z02)
    ax.plot(Z_fit, F_fit, color = 'red', ls = '--', zorder = 6, label = '2nd Hertz fit, K = {:.0f}kPa, z0 = {:.3f}mm'.format(K2*1e3, Z02))
    ax.legend()
    
plt.tight_layout()
plt.show()

# %%% Second one

dataDir = "D://MagneticPincherData//Raw//23.10.10_MacroIndenter_VCI"

D = 2.73
R = D/2
f = 'Specimen_RawData_4.csv'
rD = {'Temps' : 'T', 'Deplacement' : 'Z', 'Charge' : 'F'}
df = pd.read_csv(dataDir + '//' + f, engine = 'python', sep = ';', skiprows=[1]).rename(columns = rD)

early_points = 30

dict_curves = {}

imax = signal.find_peaks(df['Z'], distance=150)[0]
imin = signal.find_peaks((-1) * df['Z'], distance=150)[0]
imin = [0, imin[0], len(df)]

Ncycles = len(imax)
fig, axes = plt.subplots(2, Ncycles, figsize = (8, 6))
fig.suptitle('Curves for D = {:.3f} mm'.format(D))

for j in range(len(imax)):
    dict_curves[2*j] = {}
    dict_curves[2*j]['type'] = 'comp'
    dict_curves[2*j]['num'] = j+1
    dict_curves[2*j]['df'] = df.iloc[imin[j]:imax[j], :]
    
    dict_curves[2*j + 1] = {}
    dict_curves[2*j + 1]['type'] = 'relax'
    dict_curves[2*j + 1]['num'] = j+1
    dict_curves[2*j + 1]['df'] = df.iloc[imax[j]:imin[j+1], :]
    
for i in range(len(dict_curves)):
    i_ax = int(dict_curves[i]['type'] == 'relax')
    j_ax = i//2
    try:
        ax = axes[i_ax, j_ax]
    except:
        ax = axes[i_ax]
        
    ax.set_title(dict_curves[i]['type'] + ' ' + str(dict_curves[i]['num']))
    if dict_curves[i]['type'] == 'comp':
        F = dict_curves[i]['df']['F'].values
        Z = dict_curves[i]['df']['Z'].values
    elif dict_curves[i]['type'] == 'relax':
        F = dict_curves[i]['df']['F'].values[::-1]
        Z = dict_curves[i]['df']['Z'].values[::-1]
    
    F_moy = np.average(F[:early_points])
    F_std = np.std(F[:early_points])
    
    #
    F = F - F_moy
    F_moy = 0
    #
    
    ax.plot(Z, F)
    ax.set_xlabel('Z (mm)')
    ax.set_ylabel('F (N)')
    ax.grid(axis='y')
    
    #` Function
    
    def HertzFit(z, K, Z0):
        f = (4/3) * K * R**0.5 * (z - Z0)**1.5
        return(f)
    
    #### First Fit
    
    i_start = len(F) - ufun.findFirst(True, F[::-1] < F_moy + 1 * F_std)
    Z0_sup = Z[i_start]
    
    # ax.axhline(F_moy, color = 'orange', ls = '--')
    # ax.axhline(F_moy - 1 * F_std, color = 'cyan', ls = '--')
    # ax.axhline(F_moy + 1 * F_std, color = 'cyan', ls = '--', zorder = 5)
    # ax.axvline(Z[i_start], ls = '--', color = 'orange', zorder = 5)

    DZmax = 0.1
    i_stop = ufun.findFirst(True, Z >= Z0_sup + DZmax)
    
    ax.plot(Z[i_start:i_stop], F[i_start:i_stop], color = 'green', zorder = 4)
    
    binf = [0, 0]
    bsup = [np.inf, Z[i_start]]
    p0=[0, 0.99*Z[i_start]]
    [K, Z0], covM = curve_fit(HertzFit, Z[i_start:i_stop], F[i_start:i_stop], p0=p0, bounds=(binf, bsup))
    
    Z_fit = np.linspace(Z0, Z[i_stop], 100)
    F_fit = HertzFit(Z_fit, K, Z0)
    ax.plot(Z_fit, F_fit, color = 'orange', ls = '--', zorder = 6, label = '1st Hertz fit, K = {:.0f}kPa, z0 = {:.3f}mm'.format(K*1e3, Z0))
    ax.legend()
    
    #### Second Fit
    
    i_start_bis = ufun.findFirst(True, Z >= Z0)

    # ax.axvline(Z[i_start], ls = '--', color = 'red', zorder = 5)
   
    ax.plot(Z[i_start_bis:i_stop], F[i_start_bis:i_stop], color = 'turquoise', zorder = 3)
    
    binf = [0, 0]
    bsup = [np.inf, Z0_sup]
    p0=[0, Z0]

    [K2, Z02], covM = curve_fit(HertzFit, Z[i_start_bis:i_stop], F[i_start_bis:i_stop], p0=[K, Z0])
    
    Z_fit = np.linspace(Z02, Z[i_stop], 100)
    F_fit = HertzFit(Z_fit, K2, Z02)
    ax.plot(Z_fit, F_fit, color = 'red', ls = '--', zorder = 6, label = '2nd Hertz fit, K = {:.0f}kPa, z0 = {:.3f}mm'.format(K2*1e3, Z02))
    ax.legend()
    
plt.tight_layout()
plt.show()

        
# %%% Third one

dataDir = "D://MagneticPincherData//Raw//23.10.10_MacroIndenter_VCI"

D = 4.004
R = D/2
f = 'Specimen_RawData_5.csv'
rD = {'Temps' : 'T', 'Deplacement' : 'Z', 'Charge' : 'F'}
df = pd.read_csv(dataDir + '//' + f, engine = 'python', sep = ';', skiprows=[1]).rename(columns = rD)

early_points = 30

dict_curves = {}

imax = signal.find_peaks(df['Z'], distance=150)[0]
imin = signal.find_peaks((-1) * df['Z'], distance=150)[0]
imin = [0, imin[0], len(df)]

Ncycles = len(imax)
fig, axes = plt.subplots(2, Ncycles, figsize = (8, 6))
fig.suptitle('Curves for D = {:.3f} mm'.format(D))

for j in range(len(imax)):
    dict_curves[2*j] = {}
    dict_curves[2*j]['type'] = 'comp'
    dict_curves[2*j]['num'] = j+1
    dict_curves[2*j]['df'] = df.iloc[imin[j]:imax[j], :]
    
    dict_curves[2*j + 1] = {}
    dict_curves[2*j + 1]['type'] = 'relax'
    dict_curves[2*j + 1]['num'] = j+1
    dict_curves[2*j + 1]['df'] = df.iloc[imax[j]:imin[j+1], :]
    
for i in range(len(dict_curves)):
    i_ax = int(dict_curves[i]['type'] == 'relax')
    j_ax = i//2
    try:
        ax = axes[i_ax, j_ax]
    except:
        ax = axes[i_ax]
        
    ax.set_title(dict_curves[i]['type'] + ' ' + str(dict_curves[i]['num']))
    if dict_curves[i]['type'] == 'comp':
        F = dict_curves[i]['df']['F'].values
        Z = dict_curves[i]['df']['Z'].values
    elif dict_curves[i]['type'] == 'relax':
        F = dict_curves[i]['df']['F'].values[::-1]
        Z = dict_curves[i]['df']['Z'].values[::-1]
    
    F_moy = np.average(F[:early_points])
    F_std = np.std(F[:early_points])
    
    #
    F = F - F_moy
    F_moy = 0
    #
    
    ax.plot(Z, F)
    ax.set_xlabel('Z (mm)')
    ax.set_ylabel('F (N)')
    ax.grid(axis='y')
    
    #` Function
    
    def HertzFit(z, K, Z0):
        f = (4/3) * K * R**0.5 * (z - Z0)**1.5
        return(f)
    
    #### First Fit
    
    i_start = len(F) - ufun.findFirst(True, F[::-1] < F_moy + 1 * F_std)
    Z0_sup = Z[i_start]
    
    # ax.axhline(F_moy, color = 'orange', ls = '--')
    # ax.axhline(F_moy - 1 * F_std, color = 'cyan', ls = '--')
    # ax.axhline(F_moy + 1 * F_std, color = 'cyan', ls = '--', zorder = 5)
    # ax.axvline(Z[i_start], ls = '--', color = 'orange', zorder = 5)

    DZmax = 0.1
    i_stop = ufun.findFirst(True, Z >= Z0_sup + DZmax)
    
    ax.plot(Z[i_start:i_stop], F[i_start:i_stop], color = 'green', zorder = 4)
    
    binf = [0, 0]
    bsup = [np.inf, Z[i_start]]
    p0=[0, 0.99*Z[i_start]]
    [K, Z0], covM = curve_fit(HertzFit, Z[i_start:i_stop], F[i_start:i_stop], p0=p0, bounds=(binf, bsup))
    
    Z_fit = np.linspace(Z0, Z[i_stop], 100)
    F_fit = HertzFit(Z_fit, K, Z0)
    ax.plot(Z_fit, F_fit, color = 'orange', ls = '--', zorder = 6, label = '1st Hertz fit, K = {:.0f}kPa, z0 = {:.3f}mm'.format(K*1e3, Z0))
    ax.legend()
    
    #### Second Fit
    
    i_start_bis = ufun.findFirst(True, Z >= Z0)

    # ax.axvline(Z[i_start], ls = '--', color = 'red', zorder = 5)
   
    ax.plot(Z[i_start_bis:i_stop], F[i_start_bis:i_stop], color = 'turquoise', zorder = 3)
    
    binf = [0, 0]
    bsup = [np.inf, Z0_sup]
    p0=[0, Z0]

    [K2, Z02], covM = curve_fit(HertzFit, Z[i_start_bis:i_stop], F[i_start_bis:i_stop], p0=[K, Z0])
    
    Z_fit = np.linspace(Z02, Z[i_stop], 100)
    F_fit = HertzFit(Z_fit, K2, Z02)
    ax.plot(Z_fit, F_fit, color = 'red', ls = '--', zorder = 6, label = '2nd Hertz fit, K = {:.0f}kPa, z0 = {:.3f}mm'.format(K2*1e3, Z02))
    ax.legend()
    
plt.tight_layout()
plt.show()
        



# %% First plots

# %%% This, as a function of the date

def Chiaro_ExplorationPlots(date, _stiff = '_f_<_400'):
    #### Format the pincher table
    dataDir = "D://MagneticPincherData//Data_Analysis"    
    df_pincher = pd.read_csv(os.path.join(dataDir, 'MecaData_NanoIndent_2023.csv'), sep=';')
    
    # 1. Format
    
    df_pincher.insert(5, 'compressionSet', df_pincher['cellName'].apply(lambda x : int(x.split('-')[0][-1])))
    df_pincher['cellRawNum'] = df_pincher['cellName'].apply(lambda x : x.split('_')[-1])
    df_pincher['cellName'] = df_pincher['cellName'].apply(lambda x : x.split('-')[0][:-1])
    df_pincher['cellID'] = df_pincher['date'] + '_' + df_pincher['cellName']
    df_pincher['cellCode'] = df_pincher['cellRawNum'].apply(lambda x : x[:2] + x[3:])
    df_pincher.insert(4, 'cellLongCode', df_pincher['cellID'].apply(lambda x : '_'.join(x.split('_')[:-1])) + '_' + df_pincher['cellCode'])
    df_pincher['valid' + _stiff] = (df_pincher['E' + _stiff] < 30e3) & (df_pincher['R2' + _stiff] > 0.5) & (df_pincher['Chi2' + _stiff] < 1.5)
    # 2. Filter
    
    Filters = np.array([(df_pincher['date'] == date).values, # .apply(lambda x : x in dates).values,
                        # (df_pincher['cellID'] != '23-11-15_M1_P1_C1').values,
                        (df_pincher['valid' + _stiff] == True).values,
                        ])
    totalFilter = np.all(Filters, axis = 0)
    df_pincher_f = df_pincher[totalFilter]
    
    
    
    
    #### Format the indenter table
    
    df_indenter = pd.read_csv(os.path.join(dataDir, 'ChiaroData_NanoIndent_2023.csv'), sep=';')
    
    # 1. Format
    df_indenter['cellName'] = df_indenter['manip'] + '_P1_' + df_indenter['cell']
    df_indenter['cellID'] = df_indenter['date'] + '_' + df_indenter['cellName']
    
    df_indenter['indentName'] = df_indenter['cellName'] + '_' + df_indenter['indent']
    df_indenter['indentID'] = df_indenter['cellID'] + '_' + df_indenter['indent']
    
    # 2. Validity crit
    delta_Zstart = df_indenter['Z02'] - df_indenter['Z_start']
    delta_Zstop = df_indenter['Z_stop'] - df_indenter['Z02']
    df_indenter['Valid_Z0'] = (delta_Zstart > 1) & (delta_Zstop > 2)
    
    df_indenter['Valid_Fit'] = df_indenter['Rsq2'] > 0.6
    
    df_indenter['Valid_agg'] = df_indenter['Valid_Z0'] & df_indenter['Valid_Fit']
    
    # 3. Filters
    Filters = np.array([(df_indenter['date'] == date).values,
                        # (df_indenter['cellName'] != 'M1_P1_C1').values,
                        (df_indenter['Rsq2'] >= 0.7).values,
                        ])
    totalFilter = np.all(Filters, axis = 0)
    df_indenter_f = df_indenter[totalFilter]
    
    #### Start plotting
    
    #### First Plot
    
    fig1, axes1 = plt.subplots(2,1, figsize = (16, 6))
    fig = fig1
    
    parm = 'bestH0'
    ax = axes1[0]
    dictPlot = {'data' : df_pincher_f,
                'ax' : ax,
                'x' : 'cellLongCode',
                'y' : parm,
                'hue' : 'compressionSet'
                }
    sns.boxplot(**dictPlot, fliersize = 0, width=0.6)
    sns.swarmplot(**dictPlot, dodge = True, size=4, edgecolor='black', linewidth=1)
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.get_legend().remove()
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 30)
    
    parm = 'K2'
    ax = axes1[1]
    dictPlot = {'data' : df_indenter_f,
                'ax' : ax,
                'x' : 'cellID',
                'y' : parm,
                }
    
    # sns.boxplot(**dictPlot, fliersize = 0, width=0.6)
    sns.swarmplot(**dictPlot, dodge = True, size=6, edgecolor='black', linewidth=1)
    ax.set_ylim([0, ax.get_ylim()[1]])
    # ax.get_legend().remove()
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 30)
    
    plt.tight_layout()
    plt.show()
    
    #### Second Plot
    
    fig2, axes2 = plt.subplots(2,1, figsize = (8, 6))
    fig = fig2
    
    parm = 'E_f_<_400'
    ax = axes2[0]
    dictPlot = {'data' : df_pincher_f,
                'ax' : ax,
                'x' : 'cellCode',
                'y' : parm,
                'hue' : 'compressionSet'
                }
    sns.boxplot(**dictPlot, fliersize = 0, width=0.6)
    sns.swarmplot(**dictPlot, dodge = True, size=4, edgecolor='black', linewidth=1)
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.get_legend().remove()
    
    parm = 'K2'
    ax = axes2[1]
    dictPlot = {'data' : df_indenter_f,
                'ax' : ax,
                'x' : 'cell',
                'y' : parm,
                }
    sns.boxplot(**dictPlot, fliersize = 0, width=0.6)
    sns.swarmplot(**dictPlot, dodge = True, size=4, edgecolor='black', linewidth=1)
    ax.set_ylim([0, ax.get_ylim()[1]])
    # ax.get_legend().remove()
    
    plt.tight_layout()
    plt.show()
    
# %%% Run the function for all dates

dates = ['23-11-13', '23-11-15', '23-12-07', '23-12-10']

for d in dates:
    Chiaro_ExplorationPlots(d)


# %% Old stuff

# %%% First indent analysis

# %%%% Pick a folder and open the indentations inside

# C = 'C9'
# dataDir = "D://MagneticPincherData//Raw//23.11.15_NanoIndent_ChiaroData//M1//" + C + "//Indentations"

# R = 24.5
# headerFormat = 'Time (s)	Load (uN)	Indentation (nm)	Cantilever (nm)	Piezo (nm)	Auxiliary'

# def getLineWithString(lines, string):
#     i = 0
#     while string not in lines[i]:
#         i+=1
#     return(lines[i], i)
    

# rawListFiles = os.listdir(dataDir)
# listIndent = []
# for f in rawListFiles:
#     if f.endswith('.txt'):
#         # indent_number = int(f.split('_')[-1][:3])
#         fpath = os.path.join(dataDir, f)
#         fopen = open(fpath, 'r')
#         flines = fopen.readlines()
#         s, i = getLineWithString(flines, headerFormat)
#         metadataText = flines[:i]
#         df_indent = pd.read_csv(fpath, skiprows = i-1, sep='\t')
#         dirIndent = {'indent_name':f, 'path':fpath, 'metadata':metadataText, 'df':df_indent}
        
#         R = float(getLineWithString(flines, 'Tip radius (um)')[0][16:])
#         dirIndent['R'] = R
        
#         text_start_times, j = getLineWithString(flines, 'Step absolute start times')
#         text_end_times, j = getLineWithString(flines, 'Step absolute end times')
#         start_times = np.array(text_start_times[30:].split(',')).astype(float)
#         end_times = np.array(text_end_times[28:].split(',')).astype(float)
        
#         dirIndent['ti'], dirIndent['tf'] = start_times, end_times
        
#         listIndent.append(dirIndent)
#         fopen.close()
        


# # %%%% Define functions

# def FitHertz_VJojo(Z, F, fraction = 0.4):
#     early_points = 1000
#     F_moy = np.average(F[:early_points])
#     # F = F - F_moy
#     F_std = np.std(F[:early_points])
#     F_max = np.max(F)
    
#     # Function
#     def HertzFit(z, K, Z0):
#         zeroInd = 1e-9 * np.ones_like(z)
#         d = z - Z0
#         d = np.where(d>0, d, zeroInd)
#         f = (4/3) * K * (R**0.5) * (d**1.5) + F_moy
#         return(f)
    
#     #### First Fit Zone
#     i_start1 = len(F) - ufun.findFirst(True, F[::-1] < F_moy + 1 * F_std)
#     # i_start1 = len(F) - ufun.findFirst(True, F[::-1] < 0 + 1 * F_std)
#     Z0_sup = Z[i_start1]

#     # upper_threshold = F_moy + fraction * (F_max - F_moy)
#     upper_threshold = F_moy + fraction * (F_max - F_moy)
#     i_stop = ufun.findFirst(True, F >= upper_threshold)
    
#     #### First Fit
#     binf = [0, 0]
#     bsup = [5000, Z0_sup]
#     p0=[100, 1] # 0.99*Z0_sup
#     [K1, Z01], covM1 = curve_fit(HertzFit, Z[i_start1:i_stop], F[i_start1:i_stop], p0=p0, bounds=(binf, bsup))
#     Z_fit1 = np.linspace(Z01, Z[i_stop], 100)
#     F_fit1 = HertzFit(Z_fit1, K1, Z01)
    
#     #### Second Fit Zone
#     i_start2 = ufun.findFirst(True, Z >= Z01)
    
#     #### Second Fit
#     binf = [0, 0]
#     bsup = [5000, Z0_sup]
#     p0=[K1, 1]
    
#     [K2, Z02], covM2 = curve_fit(HertzFit, Z[i_start2:i_stop], F[i_start2:i_stop], p0=p0, bounds=(binf, bsup))
#     Z_fit2 = np.linspace(Z02, Z[i_stop], 100)
#     F_fit2 = HertzFit(Z_fit2, K2, Z02)
    
    
#     #### Third Fit
#     arraySteps = np.array([-0.25, +0.25])
#     Z0_error = arraySteps + Z02
#     # print(Z0_test)
#     resError = []
#     for i in range(len(arraySteps)):
#         Z0 = Z0_error[i]
#         # function
#         def HertzFit_1parm(z, K):
#             zeroInd = 1e-9 * np.ones_like(z)
#             d = z - Z0
#             d = np.where(d>0, d, zeroInd)
#             f = (4/3) * K * R**0.5 * (d)**1.5 + F_moy
#             return(f)
#         # zone
#         i_start_test = len(Z) - ufun.findFirst(True, Z[::-1] <= Z0)
#         # print(Z0, np.min(Z[i_start_test:i_stop]))
#         # fit
#         binf = [0]
#         bsup = [np.inf]
#         p0=[K2]
#         [K_error], covM_error = curve_fit(HertzFit_1parm, Z[i_start_test:i_stop], F[i_start_test:i_stop], p0=p0, bounds=(binf, bsup))
#         resError.append(K_error)
        
#     # print(resError)
#     K_error = np.array(resError)
#     print(Z0_error)
#     print(K_error)
    
#     #### Results
#     results = {'F_moy':F_moy,
#                 'F_std':F_std,
#                 'F_max':F_max,
#                 'i_start1':i_start1,
#                 'i_start2':i_start2,
#                 'i_stop':i_stop,
#                 'K1':K1,
#                 'Z01':Z01,
#                 'covM1':covM1,
#                 'Z_fit1':Z_fit1,
#                 'F_fit1':F_fit1,
#                 'K2':K2,
#                 'Z02':Z02,
#                 'covM2':covM2,
#                 'Z_fit2':Z_fit2,
#                 'F_fit2':F_fit2,
#                 'Z0_error':Z0_error,
#                 'K_error':K_error,
#                 }
    
#     return(results)


# # %%%% Analysis

# for I in listIndent[:]:
#     df = I['df']
#     df['Z Tip (nm)'] = df['Piezo (nm)'] - df['Cantilever (nm)']
#     R = I['R']
    
#     fig, axes = plt.subplots(1,2, figsize = (12,6))
#     ax=axes[0]
#     ax.plot(df['Time (s)'], df['Piezo (nm)'], 'b-', label='Displacement')
#     ax.plot(df['Time (s)'], df['Z Tip (nm)'], color='gold', ls='-', label='Z Tip')
#     ax.plot(df['Time (s)'], df['Cantilever (nm)'], 'g-', label='Deflection')
#     ax.plot(df['Time (s)'], df['Indentation (nm)'], 'r-', label='Indentation')
    
#     ax.set_xlabel('Time (s)')
#     ax.set_ylabel('Distance (nm)')
#     ax.legend()
    
    
#     i = 0
#     for ti, tf in zip(I['ti'][::-1], I['tf'][::-1]):
#         i += 1
#         step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
    
    
#     try:
#         ti, tf = I['ti'][0], I['tf'][0]
#         compression_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
#         Z = df.loc[step_indices, 'Z Tip (nm)'].values
#         F = df.loc[step_indices, 'Load (uN)'].values
#         # Z = df.loc[step_indices, 'Z Tip (nm)']
#         # F = df.loc[step_indices, 'Load (uN)']
#         results = FitHertz_VJojo(Z * 1e-3, F * 1e6, fraction = 0.6)  # nm to µm # µN to pN
        
#         ax=axes[1]
#         colorList = gs.colorList10[:len(I['ti'])]
#         i = 0
#         for ti, tf, c in zip(I['ti'][::-1], I['tf'][::-1], colorList[::-1]):
#             i += 1
#             step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
#             # print(step_indices)
#             ax.plot(df.loc[step_indices, 'Z Tip (nm)'], df.loc[step_indices, 'Load (uN)'], color=c, marker='.', ls='')
        
#         ax.set_xlabel('Distance (nm)')
#         ax.set_ylabel('Load (uN)')
        
#         ax.axvline(Z[results['i_start1']], ls = '--', color = 'yellow', zorder = 4)
#         ax.axvline(Z[results['i_start2']], ls = '--', color = 'orange', zorder = 4)
#         ax.axvline(Z[results['i_stop']], ls = '--', color = 'red', zorder = 4)
#         ax.axhline((results['F_moy'])*1e-6, ls = '-', color = 'k', zorder = 4)
#         ax.axhline((results['F_moy']+results['F_std'])*1e-6, ls = '--', color = 'k', zorder = 4)
#         ax.plot(results['Z_fit1']*1e3, results['F_fit1']*1e-6, ls = '-', color = 'orange', zorder = 5,
#                 label = 'First fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa'.format(results['Z01'], results['covM1'][1,1]**0.5,
#                                                                                                   results['K1'], results['covM1'][0,0]**0.5))
#         ax.plot(results['Z_fit2']*1e3, results['F_fit2']*1e-6, ls = '-', color = 'red', zorder = 5,
#                 label = 'Second fit\nZ0 = {:.2f} +/- {:.2f} µm\nYeff = {:.0f} +/- {:.0f} Pa'.format(results['Z02'], results['covM2'][1,1]**0.5,
#                                                                                                     results['K2'], results['covM2'][0,0]**0.5))
#         Z0_error, K_error = results['Z0_error'], results['K_error']
#         for j in range(len(Z0_error)):
#             Z0, K = Z0_error[j], K_error[j]
#             def Hertz(z, Z0, K):
#                 zeroInd = 1e-9 * np.ones_like(z)
#                 d = z - Z0
#                 d = np.where(d>0, d, zeroInd)
#                 f = (4/3) * K * R**0.5 * (d)**1.5 + results['F_moy']
#                 return(f)
#             Zplot = np.linspace(Z0*1e3, Z[results['i_stop']], 100) * 1e-3
#             Ffit = Hertz(Zplot, Z0, K)
#             ax.plot(Zplot*1e3, Ffit*1e-6, ls = '--', color = 'cyan', zorder = 4)
#         dZ0 = np.abs(Z0_error[1] - Z0_error[0])/2
#         ax.plot([], [], ls = '--', color = 'cyan', zorder = 4, label = 'Effect of error on Z0\n+/-{:.2f} µm =>\nYeff = {:.0f}~{:.0f} Pa'.format(dZ0, K_error[0], K_error[1]))
    
        
        
#     except:
#         pass
    
#     ax.legend()
    
#     fig.suptitle(I['indent_name'])
#     plt.tight_layout()
#     plt.show()
#     # ufun.simpleSaveFig(fig, I['indent_name'], dataDir, '.png', 150)