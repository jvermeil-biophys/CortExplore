# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:00:29 2024

@author: EE
"""
#%%import
#### Main imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statistics as stat

import os
import sys
import time
import random
import warnings
import itertools
import matplotlib
import numbers

from copy import copy
from cycler import cycler
from datetime import date
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

#### Local Imports

import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)


import GraphicStyles as gs
import UtilityFunctions as ufun
import TrackAnalyser as taka


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
# from bokeh.io import output_notebook, show
# from bokeh.plotting import figure
# from bokeh.models import ColumnDataSource, HoverTool, Range1d
# from bokeh.transform import factor_cmap
# from bokeh.palettes import Category10
# from bokeh.layouts import gridplot

#### stat annotations
from statannotations.Annotator import Annotator

#%% functions
def dataFluoMerge(data,data_fluo):
    for i in range(len(data_fluo['cellCode'])):
        for j in range(len(data['cellCode'])):
            if data_fluo.at[data_fluo.index[i],'cellCode']==data.at[data.index[j],'cellCode']:
                data.at[data.index[j],'Color']=data_fluo.at[data_fluo.index[i],'Color']
                data.at[data.index[j],'ratio G/R']=data_fluo.at[data_fluo.index[i],'ratio G/R']
                data.at[data.index[j],'EGFP']=data_fluo.at[data_fluo.index[i],'EGFP']
                data.at[data.index[j],'DsRed']=data_fluo.at[data_fluo.index[i],'DsRed']
                data.at[data.index[j],'phase']=data_fluo.at[data_fluo.index[i],'phase']
    return data


def percentageNLItype(data):
    N=len(data)
    intermediate=len(data[data['NLI_type']=='Intermediate'])/N
    linear=len(data[data['NLI_type']=='Linear'])/N
    nonlinear=len(data[data['NLI_type']=='Non-linear'])/N
    return linear,intermediate,nonlinear,N



def addEeff(data):
    data['E_effective']=data['Y_vwc_Full']+data['K_vwc_Full']*0.8**-4
    return data

def addNLI(data):
    data['NLI']=np.log10((data['K_vwc_Full']*0.8**-4)/data['Y_vwc_Full'])
    data.loc[data['NLI'] > 0.3, ['NLI_type']] = 'Non-linear'
    data.loc[data['NLI'] < -0.3, ['NLI_type']] = 'Linear'
    data.loc[(data['NLI'] > -0.3) & (data['NLI']<0.3), ['NLI_type']] = 'Intermediate'
    return data


def percentageNLIphase(data):
    phases=['G1','G1/S','S/G2','M']
    N_all=pd.DataFrame()
    for phase in phases :
        N_l,N_i,N_nl=0,0,0
        N=len(data[data['phase']==phase]['NLI_type'])
        for i in data[data['phase']==phase]['NLI_type']:
            if i=='Linear':
                N_l+=1
            elif i=='Intermediate':
                N_i+=1
            elif i=='Non-linear':
                N_nl+=1
        N_all.at[phase,'Linear_percentage']=N_l/N
        N_all.at[phase,'Intermediate_percentage']=N_i/N
        N_all.at[phase,'Non-linear_percentage']=N_nl/N
        N_all.at[phase,'N_comp']=N
    return N_all

def addmedian(data):
    for i in data['cellID'] :
        j=data.index[data['cellID']==i]
        for k in j:
            #data.at[k,'mean_h'] = data.loc[j[0]:j[-1]]['bestH0'].sum()/len(j)
            data.at[k,'median_h'] = stat.median(data.loc[j[0]:j[-1]]['bestH0'])
            #data.at[k,'mean_E'] = data.loc[j[0]:j[-1]]['E_effective'].sum()/len(j)
            data.at[k,'median_E'] = stat.median(data.loc[j[0]:j[-1]]['E_effective'])
    return data
#%% load data
data= taka.getGlobalTable_meca('24-06-12_VWC_HelaFucci')

path_fluo='D:/Eloise/MagneticPincherData/Raw/24.06.12_fluo'
data_fluo=pd.read_csv(os.path.join(path_fluo,'data_fluo_24.06.12.txt'),sep='\t')
data_fluo=data_fluo.rename({'Unnamed: 0':'cellCode'},axis='columns')
data_fluo['date']='24-06-12'

data=dataFluoMerge(data, data_fluo)

#%%
out='P2_C5'
data= data[data['cellCode'].str.contains(out) == False]

#%%
path='D:/Eloise/MagneticPincherData/Raw'
data.to_csv(path + '/data_24.06.04new.txt', sep='\t')



#%% add NLI stuff
data=addEeff(data)
data=addNLI(data)

#%%
path12='D:/Eloise/MagneticPincherData/Raw/24.06.12'
data12=pd.read_csv(path12 + '/data_24.06.12vf.txt', sep='\t')
data12=data12.rename({'Unnamed: 0':'i'},axis='columns')

#%% plot parameters
plot_parms = {'x':'phase', 
              'y':'bestH0',
              # 'hue':'phase',
              # 'hue_order':['G1','G1/S','S/G2','M'],
              'order':['G1','G1/S','S/G2','M'],
              'palette':['red','orange','green','limegreen'],
              'dodge':False}

#%% boxplot bestH0 per phase
plt.figure()
ax = sns.boxplot(**plot_parms)
ax = sns.swarmplot(**plot_parms)
pairs=[("G1","G1/S"),('G1','S/G2'),('G1/S','S/G2'),('S/G2','M'),('G1','M'),('G1/S','M')]

annotator = Annotator(ax, pairs, **plot_parms)
annotator.configure(test='Mann-Whitney',text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.show()

#%% NLI plot
N_all=percentageNLIphase(data)

fig, ax = plt.subplots()
rects1 = ax.bar(N_all.index, N_all['Linear_percentage'],label='linear')
rects2 = ax.bar(N_all.index, N_all['Intermediate_percentage'],bottom=N_all['Linear_percentage'],label='intermediate')
rects3 = ax.bar(N_all.index, N_all['Non-linear_percentage'],bottom=N_all['Linear_percentage']+N_all['Intermediate_percentage'],label='non_linear',color='brown')
N_comp=[i for i in N_all['N_comp']]
ax.bar_label(rects3,labels=[f'N={e:0.0f}' for e in N_comp])
ax.bar_label(rects1,labels=[f'{e:.2%}' for e in N_all['Linear_percentage']],label_type='center')
ax.bar_label(rects2,labels=[f'{e:.2%}' for e in N_all['Intermediate_percentage']],label_type='center')
ax.bar_label(rects3,labels=[f'{e:.2%}' for e in N_all['Non-linear_percentage']],label_type='center')
plt.title('')
ax.legend(bbox_to_anchor=(1.01, 1.01))
fig.set_size_inches(10, 8)
plt.show()

#%%

ax=N_all.plot(kind='bar',stacked=True,rot=0,)
ax.legend(['Linear','Intermediate','Non-Linear'],loc='upper right')
for c in ax.containers:
    labels = [str(round(v.get_height()*100,2)) + "%" if v.get_height() > 0 else '' for v in c]
    ax.bar_label(c,labels=labels, label_type='center')
plt.show()


#%%
data= taka.getGlobalTable_meca('24-07-03_VWC_HelaFucci_combined_simple')

#%%
path_fluo='D:/Eloise/MagneticPincherData/Raw/24.07.03_fluo'
data_fluo=pd.read_csv(os.path.join(path_fluo,'data_fluo_24.07.03.txt'),sep='\t')
data_fluo=data_fluo.rename({'Unnamed: 0':'cellCode'},axis='columns')
data_fluo['date']='24-07-03'

#%%
data=dataFluoMerge(data, data_fluo)

#%%
data=addEeff(data)
data=addNLI(data)

#%%
for i in data['cellID'] :
    j=data.index[data['cellID']==i]
    for k in j:
        data.at[k,'mean_h'] = data.loc[j[0]:j[-1]]['bestH0'].sum()/len(j)
        data.at[k,'median_h'] = stat.median(data.loc[j[0]:j[-1]]['bestH0'])

#%%
for i in data['cellID'] :
    j=data.index[data['cellID']==i]
    for k in j:
        data.at[k,'mean_Eeff'] = data.loc[j[0]:j[-1]]['E_effective'].sum()/len(j)
        data.at[k,'median_Eeff'] = stat.median(data.loc[j[0]:j[-1]]['E_effective'])
        
#%%
plt.figure()
ax = sns.boxplot(data=data,**plot_parms)
ax = sns.swarmplot(data=data,**plot_parms)
pairs=[("G1","G1/S"),('G1','S/G2'),('G1/S','S/G2'),('S/G2','M'),('G1','M'),('G1/S','M')]

annotator = Annotator(ax, pairs,data=data, **plot_parms)
annotator.configure(test='Mann-Whitney',text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.show()

#%%
excluded = [k for k in range(7,11)]
for i in excluded:
    print(i)
    data= data[data['compNum'].astype(str).contains(i) == False]
    

#%%
data= taka.getGlobalTable_meca('24-07-11_VWC_HelaFucci_X3')

#%%
path_fluo='D:/Eloise/MagneticPincherData/Raw/24.07.11_fluo'
data_fluo=pd.read_csv(os.path.join(path_fluo,'data_fluo_24.07.11.txt'),sep='\t')
data_fluo=data_fluo.rename({'Unnamed: 0':'cellCode'},axis='columns')
data_fluo['date']='24-07-11'
#%%
data=dataFluoMerge(data, data_fluo)

#%%
data=addEeff(data)
data=addNLI(data)
#%%
plot_parms = {'x':'phase', 
              'y':'bestH0',
              # 'hue':'phase',
              # 'hue_order':['G1','G1/S','S/G2','M'],
              'order':['G1','G1/S','S/G2','M'],
              'palette':['red','orange','green','limegreen'],
              'dodge':False}
#%%
out=['P2_C9','P2_C6']
for i in out:
    print(i)
    data= data[data['cellCode'].str.contains(i) == False]
#%%
plt.figure()
ax = sns.boxplot(data=data,**plot_parms)
ax = sns.swarmplot(data=data,**plot_parms)
pairs=[("G1","G1/S"),('G1','S/G2'),('G1/S','S/G2'),('S/G2','M'),('G1','M'),('G1/S','M')]

annotator = Annotator(ax, pairs,data=data, **plot_parms)
annotator.configure(test='Mann-Whitney',text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.show()

#%%
data= taka.getGlobalTable_meca('24-06-04_VWC_HelaFucci')

#%%
path_fluo='D:/Eloise/MagneticPincherData/Raw/24.06.04_fluo'
data_fluo=pd.read_csv(os.path.join(path_fluo,'data_fluo_24.06.04_wobg.txt'),sep='\t')
data_fluo=data_fluo.rename({'Unnamed: 0':'cellCode'},axis='columns')
data_fluo['date']='24-06-04'
#%%
data=dataFluoMerge(data, data_fluo)

#%%
data=addEeff(data)
data=addNLI(data)


#%%
out=['24-06-04_M1_P1_C2','24-06-04_M1_P2_C13','24-06-04_M1_P2_C12']
for i in excluded:
    data= data[data['cellID'].str.contains(i) == False]
#%%
for i in range(150,160):
    data=data.drop(i,axis=0)
for j in range(220,230):
    data=data.drop(j,axis=0)
    
#%% New data
#%%%
data= taka.getGlobalTable_meca('24-07-18_VWC_HelaFucci_simple')

#%%%
path_fluo='D:/Eloise/MagneticPincherData/Raw/24.07.18_fluo'
data_fluo=pd.read_csv(os.path.join(path_fluo,'data_fluo_24.07.18.txt'),sep='\t')
data_fluo=data_fluo.rename({'Unnamed: 0':'cellCode'},axis='columns')
data_fluo['date']='24-07-18'
#%%%
data=dataFluoMerge(data, data_fluo)

#%%%
data=addEeff(data)
data=addNLI(data)
data=addmedian(data)
#%%%
out=['P1_C7']
for i in out:
    print(i)
    data= data[data['cellCode'].str.contains(i) == False]
#%%% plot parameters
plot_parms = {'x':'phase', 
              'y':'E_effective',
              # 'hue':'phase',
              # 'hue_order':['G1','G1/S','S/G2','M'],
              'order':['G1','G1/S','S/G2','M'],
              'palette':['red','orange','green','limegreen'],
              'dodge':False}
#%%%
plt.figure()
ax = sns.boxplot(data=data,**plot_parms)
ax = sns.swarmplot(data=data,**plot_parms,size=3)
pairs=[("G1","G1/S"),('G1','S/G2'),('G1/S','S/G2'),('S/G2','M'),('G1','M'),('G1/S','M')]

annotator = Annotator(ax, pairs,data=data, **plot_parms)
annotator.configure(test='Mann-Whitney',text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.show()
