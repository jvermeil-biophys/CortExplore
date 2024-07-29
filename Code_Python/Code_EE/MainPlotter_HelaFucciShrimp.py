# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:05:32 2024

@author: BioMecaCell
"""

# %% > Imports and constants

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

#%%
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
#%% Use VWC model from TrackAnalyzer V3


#%%% Data from 04-06

data= taka.getGlobalTable_meca('24-05-23_VWC_HelaFucci')

#%% take fluo data
path_fluo='D:/Eloise/MagneticPincherData/Raw/24.05.23_fluo'
data_fluo=pd.read_csv(os.path.join(path_fluo,'data_fluo_24.05.23.txt'),sep='\t')
data_fluo=data_fluo.rename({'Unnamed: 0':'cellCode'},axis='columns')
data_fluo['date']='24-05-23'

#%% merge fluo and data from global meca
data=dataFluoMerge(data, data_fluo)

#%%
path='D:/Eloise/MagneticPincherData/Raw/24.05.23'
data.to_csv(path + '/data_24.05.23vf.txt', sep='\t')
#%% add Eeff column

data['E_effective']=data['Y_vwc_Full']+data['K_vwc_Full']*0.8**-4
for i in data['cellID'] :
    j=data.index[data['cellID']==i]
    for k in j:
        data.at[k,'mean_h'] = data.loc[j[0]:j[-1]]['bestH0'].sum()/len(j)
        data.at[k,'median_h'] = stat.median(data.loc[j[0]:j[-1]]['bestH0'])
for i in data['cellID'] :
    j=data.index[data['cellID']==i]
    for k in j:
        data.at[k,'mean_Eeff'] = data.loc[j[0]:j[-1]]['E_effective'].sum()/len(j)
        data.at[k,'median_Eeff'] = stat.median(data.loc[j[0]:j[-1]]['E_effective'])
#%% add NLI column

data['NLI']=np.log10((data['K_vwc_Full']*0.8**-4)/data['Y_vwc_Full'])
#%%
data.loc[data['NLI'] > 0.3, ['NLI_type']] = 'Non-linear'
data.loc[data['NLI'] < -0.3, ['NLI_type']] = 'Linear'
data.loc[(data['NLI'] > -0.3) & (data['NLI']<0.3), ['NLI_type']] = 'Intermediate' 

#%%
excluded = ['P2_C5','P1_C1','P2_C4','P1_C2','P2_C3']#,'24-05-23_M1_P2_C5','24-05-23_M2_P2_C5']
for i in excluded:
    data = data[data['cellCode'].str.contains(i) == False]

#%%
data=data[data['manipID'].str.contains('24-05-23_M2') == False]
#%%
for i in range(60,66):
    data=data.drop(i,axis=0)
for j in range(180,186):
    data=data.drop(j,axis=0)
#%% plot bestH0 VWC
sns.boxplot(data,x='phase',y='bestH0',order=['G1','G1/S','S/G2','M'])
sns.swarmplot(data,x='phase',y='bestH0',order=['G1','G1/S','S/G2','M'])
plt.show()

#%%
sns.boxplot(data,x='phase',y='ctFieldFluctuAmpli',order=['G1','G1/S','S/G2','M'])
sns.swarmplot(data,x='phase',y='ctFieldFluctuAmpli',order=['G1','G1/S','S/G2','M'])
plt.show()

#%% plot linearity of compressions

def percentageNLItype(data):
    N=len(data)
    intermediate=len(data[data['NLI_type']=='Intermediate'])/N
    linear=len(data[data['NLI_type']=='Linear'])/N
    nonlinear=len(data[data['NLI_type']=='Non-linear'])/N
    return linear,intermediate,nonlinear,N

#%%
L,I,NL,N=percentageNLItype(data)

plt.bar('04-06',L,label='linear')
plt.bar('04-06',I,bottom=L,label='intermediate')
plt.bar('04-06',NL,bottom=L+I,label='non_linear')
plt.legend()
plt.title('N=%i' %N)
plt.show()

#%% get linearity per phase
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
    
#%%
N_all=percentageNLIphase(data)

fig, ax = plt.subplots()
rects1 = ax.bar(N_all.index, N_all['Linear_percentage'],label='linear')
rects2 = ax.bar(N_all.index, N_all['Intermediate_percentage'],bottom=N_all['Linear_percentage'],label='intermediate')
rects3 = ax.bar(N_all.index, N_all['Non-linear_percentage'],bottom=N_all['Linear_percentage']+N_all['Intermediate_percentage'],label='non_linear')
N_comp=[i for i in N_all['N_comp']]
ax.bar_label(rects3,labels=[f'N={e}' for e in N_comp])
# plt.bar(N_all.index,N_all['Linear_percentage'],label='linear')
# plt.bar(N_all.index,N_all['Intermediate_percentage'],bottom=N_all['Linear_percentage'],label='intermediate')
# plt.bar(N_all.index,N_all['Non-linear_percentage'],bottom=N_all['Linear_percentage']+N_all['Intermediate_percentage'],label='non_linear')
# for i in range(len(N_all['N_comp'])):
#     n=N_all.at[N_all.index[i],'N_comp']
#     plt.title('N=%i' %n)
plt.legend()
plt.show()


#%%%

data = taka.getGlobalTable_meca('24-06-12_VWC_HelaFucci')

#%% take fluo data
path_fluo='D:/Eloise/MagneticPincherData/Raw/24.06.12_fluo'
data_fluo=pd.read_csv(os.path.join(path_fluo,'data_fluo_24.06.12.txt'),sep='\t')
data_fluo=data_fluo.rename({'Unnamed: 0':'cellCode'},axis='columns')
data_fluo['date']='24-06-12'

#%%
data=dataFluoMerge(data, data_fluo)

#%%
sns.boxplot(data,x='phase',y='bestH0',order=['G1','G1/S','S/G2','M'])
sns.swarmplot(data,x='phase',y='bestH0',order=['G1','G1/S','S/G2','M'])
plt.show()

#%%
sns.boxplot(data,x='phase',y='surroundingThickness',order=['G1','G1/S','S/G2','M'])
sns.swarmplot(data,x='phase',y='surroundingThickness',order=['G1','G1/S','S/G2','M'])
plt.show()

#%%
for i in data['cellID'] :
    j=data.index[data['cellID']==i]
    for k in j:
        data.at[k,'mean_h'] = data.loc[j[0]:j[-1]]['bestH0'].sum()/len(j)
        data.at[k,'median_h'] = stat.median(data.loc[j[0]:j[-1]]['bestH0'])

#%%
excluded = ['24-06-12_M1_P2_C5','24-06-12_M1_P1_C6','24-06-12_M1_P2_C4']#,'24-05-23_M1_P2_C5','24-05-23_M2_P2_C5']
for i in excluded:
    data = data[data['cellID'].str.contains(i) == False]
for i in range(202,212):
    data=data.drop(i,axis=0)
#%%
sns.boxplot(data,x='phase',y='mean_h',order=['G1','G1/S','S/G2','M'])
sns.swarmplot(data,x='phase',y='mean_h',order=['G1','G1/S','S/G2','M'])
plt.show()
#%%
data['E_effective']=data['Y_vwc_Full']+data['K_vwc_Full']*0.8**-4
for i in data['cellID'] :
    j=data.index[data['cellID']==i]
    for k in j:
        data.at[k,'mean_Eeff'] = data.loc[j[0]:j[-1]]['E_effective'].sum()/len(j)
        data.at[k,'median_Eeff'] = stat.median(data.loc[j[0]:j[-1]]['E_effective'])
#%% add NLI column

data['NLI']=np.log10((data['K_vwc_Full']*0.8**-4)/data['Y_vwc_Full'])

data.loc[data['NLI'] > 0.3, ['NLI_type']] = 'Non-linear'
data.loc[data['NLI'] < -0.3, ['NLI_type']] = 'Linear'
data.loc[(data['NLI'] > -0.3) & (data['NLI']<0.3), ['NLI_type']] = 'Intermediate' 
#%%
N_12=percentageNLIphase(data)

fig, ax = plt.subplots()
rects1 = ax.bar(N_12.index, N_12['Linear_percentage'],label='linear')
rects2 = ax.bar(N_12.index, N_12['Intermediate_percentage'],bottom=N_12['Linear_percentage'],label='intermediate')
rects3 = ax.bar(N_12.index, N_12['Non-linear_percentage'],bottom=N_12['Linear_percentage']+N_12['Intermediate_percentage'],label='non_linear',color='brown')
N_comp=[i for i in N_12['N_comp']]
ax.bar_label(rects3,labels=[f'N={e:0.0f}' for e in N_comp])
# plt.bar(N_all.index,N_all['Linear_percentage'],label='linear')
# plt.bar(N_all.index,N_all['Intermediate_percentage'],bottom=N_all['Linear_percentage'],label='intermediate')
# plt.bar(N_all.index,N_all['Non-linear_percentage'],bottom=N_all['Linear_percentage']+N_all['Intermediate_percentage'],label='non_linear')
# for i in range(len(N_all['N_comp'])):
#     n=N_all.at[N_all.index[i],'N_comp']
#     plt.title('N=%i' %n)
plt.title('24.06.12 - NLI index delimitations ]-2.5;2.5[')
ax.legend(bbox_to_anchor=(1.01, 1.01))
plt.show()

#%%
data.loc[data['NLI'] > 0.3, ['NLI_ind']] = 0
data.loc[data['NLI'] < -0.3, ['NLI_ind']] = 1
data.loc[(data['NLI'] > -0.3) & (data['NLI']<0.3), ['NLI_ind']] = 0.5

#%%
path='D:/Eloise/MagneticPincherData/Raw/24.06.12'
data.to_csv(path + '/data_24.06.12.txt', sep='\t')
#%%
p3=sns.swarmplot(x=data['phase'],y=data['NLI'],hue=data['cellCode'])
p3.axhline(3)
p3.axhline(-3)
p3.axhline(0.3,color='r')
p3.axhline(-0.3,color='r')

plt.show()



#%%data 04-06
data = taka.getGlobalTable_meca('24-06-04_VWC_HelaFucci')

#%%%take fluo data
path_fluo='D:/Eloise/MagneticPincherData/Raw/24.06.04_fluo'
data_fluo=pd.read_csv(os.path.join(path_fluo,'data_fluo_24.06.04_wobg.txt'),sep='\t')
data_fluo=data_fluo.rename({'Unnamed: 0':'cellCode'},axis='columns')
data_fluo['date']='24-06-04'

#%%
#%%
excluded = ['24-06-04_M1_P1_C2','24-06-04_M1_P2_C13']
for i in excluded:
    data= data[data['cellID'].str.contains(i) == False]
for i in range(150,160):
    data=data.drop(i,axis=0)
for j in range(220,230):
    data=data.drop(j,axis=0)
#%%%
data=dataFluoMerge(data, data_fluo)

#%%%
sns.boxplot(data,x=data['phase'],y=data['bestH0'],order=['G1','G1/S','S/G2','M'])
sns.swarmplot(data,x=data['phase'],y=data['bestH0'],order=['G1','G1/S','S/G2','M'])
plt.show()

#%%%
sns.boxplot(data,x='phase',y='surroundingThickness',order=['G1','G1/S','S/G2','M'])
sns.swarmplot(data,x='phase',y='surroundingThickness',order=['G1','G1/S','S/G2','M'])
plt.show()

#%%%
for i in data['cellID'] :
    j=data.index[data['cellID']==i]
    for k in j:
        data.at[k,'mean_h'] = data.loc[j[0]:j[-1]]['bestH0'].sum()/len(j)
        data.at[k,'median_h'] = stat.median(data.loc[j[0]:j[-1]]['bestH0'])
#%%
l=data.index[data['cellID']=='24-06-04_M1_P2_C9']
for k in l:
    data.at[k,'mean_h'] = data.loc[j[0]:j[-1]]['bestH0'].sum()/9
#%%%
# excluded = ['24-06-12_M1_P2_C5','24-06-12_M1_P1_C6','24-06-12_M1_P2_C4']#,'24-05-23_M1_P2_C5','24-05-23_M2_P2_C5']
# for i in excluded:
#     data = data[data['cellID'].str.contains(i) == False]
# for i in range(202,212):
#     data=data.drop(i,axis=0)
#%%%
sns.boxplot(data,x='phase',y='mean_h',order=['G1','G1/S','S/G2','M'])
sns.swarmplot(data,x='phase',y='mean_h',order=['G1','G1/S','S/G2','M'])
plt.show()
#%%
data['E_effective']=data['Y_vwc_Full']+data['K_vwc_Full']*0.8**-4

for i in data['cellID'] :
    j=data.index[data['cellID']==i]
    for k in j:
        data.at[k,'mean_Eeff'] = data.loc[j[0]:j[-1]]['E_effective'].sum()/len(j)
        data.at[k,'median_Eeff'] = stat.median(data.loc[j[0]:j[-1]]['E_effective'])
#%% add NLI column

data['NLI']=np.log10((data['K_vwc_Full']*0.8**-4)/data['Y_vwc_Full'])

data.loc[data['NLI'] > 0.3, ['NLI_type']] = 'Non-linear'
data.loc[data['NLI'] < -0.3, ['NLI_type']] = 'Linear'
data.loc[(data['NLI'] > -0.3) & (data['NLI']<0.3), ['NLI_type']] = 'Intermediate' 
#%%
N_12=percentageNLIphase(data)

fig, ax = plt.subplots()
rects1 = ax.bar(N_12.index, N_12['Linear_percentage'],label='linear')
rects2 = ax.bar(N_12.index, N_12['Intermediate_percentage'],bottom=N_12['Linear_percentage'],label='intermediate')
rects3 = ax.bar(N_12.index, N_12['Non-linear_percentage'],bottom=N_12['Linear_percentage']+N_12['Intermediate_percentage'],label='non_linear',color='brown')
N_comp=[i for i in N_12['N_comp']]
ax.bar_label(rects3,labels=[f'N={e:0.0f}' for e in N_comp])
# plt.bar(N_all.index,N_all['Linear_percentage'],label='linear')
# plt.bar(N_all.index,N_all['Intermediate_percentage'],bottom=N_all['Linear_percentage'],label='intermediate')
# plt.bar(N_all.index,N_all['Non-linear_percentage'],bottom=N_all['Linear_percentage']+N_all['Intermediate_percentage'],label='non_linear')
# for i in range(len(N_all['N_comp'])):
#     n=N_all.at[N_all.index[i],'N_comp']
#     plt.title('N=%i' %n)
plt.title('24.06.04 - NLI index delimitations ]-3;3[')
ax.legend(bbox_to_anchor=(1.01, 1.01))
plt.show()

#%%
data.loc[data['NLI'] > 0.3, ['NLI_ind']] = 0
data.loc[data['NLI'] < -0.3, ['NLI_ind']] = 1
data.loc[(data['NLI'] > -0.3) & (data['NLI']<0.3), ['NLI_ind']] = 0.5


#%%
path='D:/Eloise/MagneticPincherData/Raw'
data.to_csv(path + '/data_24.06.04_new.txt', sep='\t')
#%%
p3=sns.swarmplot(x=data['phase'],y=data['NLI'],hue=data['cellCode'],size=4)
p3.axhline(3)
p3.axhline(-3)
p3.axhline(0.3,color='r')
p3.axhline(-0.3,color='r')
plt.title('24-06-04')


























#%% Combine June experiments

path04='D:/Eloise/MagneticPincherData/Raw/24.06.04'
data04=pd.read_csv(path04 + '/data_24.06.04.txt', sep='\t')
data04=data04.rename({'Unnamed: 0':'i'},axis='columns')

path12='D:/Eloise/MagneticPincherData/Raw/24.06.12'
data12=pd.read_csv(path12 + '/data_24.06.12.txt', sep='\t')
data12=data12.rename({'Unnamed: 0':'i'},axis='columns')

path05='D:/Eloise/MagneticPincherData/Raw/24.05.23'
data05=pd.read_csv(path05 + '/data_24.05.23vf.txt', sep='\t')
data05=data05.rename({'Unnamed: 0':'i'},axis='columns')
data05= data05[data05['manipID'].str.contains('24-05-23_M1') ==True]
#%%
data_june=pd.concat([data04,data12])
#%%
s = pd.Series([x for x in range(len(data_june))])
data_june=data_june.set_index(s)
#%%
plt.figure()
plot_parms = {'data':data_june, 
              'x':'phase', 
              'y':'bestH0',
              # 'hue':'phase',
              # 'hue_order':['G1','G1/S','S/G2','M'],
              'order':['G1','G1/S','S/G2','M'],
              'palette':['red','orange','green','limegreen'],
              'dodge':False}

ax = sns.boxplot(**plot_parms)
ax = sns.swarmplot(**plot_parms)
pairs=[("G1","G1/S"),('G1','S/G2'),('G1/S','S/G2'),('S/G2','M'),('G1','M'),('G1/S','M')]

annotator = Annotator(ax, pairs, **plot_parms)
annotator.configure(test='Mann-Whitney',text_format='star', loc='inside')
annotator.apply_and_annotate()

#plt.tight_layout()
plt.title('June experiments - Mann-Whitney test')
plt.show()
#%%
plt.figure()
sns.scatterplot(data04,x='median_h',y='median_Eeff',hue='phase')
plt.xscale('log')
plt.yscale('log')
#%%
N_all=percentageNLIphase(data04)

fig, ax = plt.subplots()
rects1 = ax.bar(N_all.index, N_all['Linear_percentage'],label='linear')
rects2 = ax.bar(N_all.index, N_all['Intermediate_percentage'],bottom=N_all['Linear_percentage'],label='intermediate')
rects3 = ax.bar(N_all.index, N_all['Non-linear_percentage'],bottom=N_all['Linear_percentage']+N_all['Intermediate_percentage'],label='non_linear',color='brown')
N_comp=[i for i in N_all['N_comp']]
ax.bar_label(rects3,labels=[f'N={e:0.0f}' for e in N_comp])
# plt.bar(N_all.index,N_all['Linear_percentage'],label='linear')
# plt.bar(N_all.index,N_all['Intermediate_percentage'],bottom=N_all['Linear_percentage'],label='intermediate')
# plt.bar(N_all.index,N_all['Non-linear_percentage'],bottom=N_all['Linear_percentage']+N_all['Intermediate_percentage'],label='non_linear')
# for i in range(len(N_all['N_comp'])):
#     n=N_all.at[N_all.index[i],'N_comp']
#     plt.title('N=%i' %n)
plt.title('24-06-04 - NLI index delimitations ]-0.3;0.3[')
ax.legend(bbox_to_anchor=(1.01, 1.01))
fig.set_size_inches(10, 8)
plt.show()

#%%
p3=sns.swarmplot(x=data_june['phase'],y=data_june['NLI'],hue=data_june['date'],size=3,order=['G1','G1/S','S/G2','M'])
p3.axhline(3)
p3.axhline(-3)
p3.axhline(0.3,color='r')
p3.axhline(-0.3,color='r')
plt.title('24-06')

#%%
sns.boxplot(data=data_june,x='phase',y='surroundingThickness',order=['G1','G1/S','S/G2','M'],hue='phase',palette=['green','orange','red','limegreen'])
sns.swarmplot(data=data_june,x='phase',y='surroundingThickness',order=['G1','G1/S','S/G2','M'],size=3,hue='cellID',legend=False)
plt.ylim(0,1000)
plt.title('June data (06-04 + 06-12)')
plt.show()
#%%
plt.figure()
sns.boxplot(data=data_june,x='phase',y='median_h',order=['G1','G1/S','S/G2','M'],hue='phase',palette=['green','orange','red','limegreen'])
sns.swarmplot(data=data_june,x='phase',y='median_h',order=['G1','G1/S','S/G2','M'],size=5,hue='cellID',legend=False)
plt.ylim(0,1000)
plt.title('June data (06-04 + 06-12)')


#%%
plt.figure()
sns.scatterplot(data=data_june,x='median_h',y='E_effective',hue='phase',palette=['green','orange','red','limegreen'],style='date')
plt.title('E effective vs median H')
plt.yscale('log')
#plt.xscale('log')
#%%
def func(x, a, c):
    return a*x+c
#%%
data_M=data_june[data_june['phase'].str.contains('M') == True]
data_G1=data_june[data_june['phase'].str.contains('G1') == True]
#%%
popt,pcov=curve_fit(func,data_G1['median_h'],data_G1['E_effective'])
#%%
plt.figure()
sns.scatterplot(data=data_june,x='median_h',y='E_effective',hue='phase',palette=['green','orange','red','limegreen'],style='date')
xdata = np.linspace(np.min(data_june['median_h']), np.max(data_june['median_h']))
y=func(xdata,*popt)
#plt.plot(xdata,y,'b')
plt.title('E effective vs median H')
#plt.yscale('log')
#%%
plt.figure()
sns.lmplot(data=data_june,x='median_h',y='E_effective',hue='phase')


#%%
plt.figure()
sns.scatterplot(data=data_june,x='median_h',y='ctFieldFluctuAmpli',hue='phase',palette=['green','orange','red','limegreen'])
plt.title('Fluctuations vs median thickness')

#%%
plt.figure()
sns.scatterplot(data_june,x='EGFP',y='median_h',hue='phase',palette=['green','orange','red','limegreen'],style='date')
#plt.yscale('log')
plt.xscale('log')
plt.title('Median H vs Green fluo - June cells')


#%%











data_all=pd.concat([data05,data04,data12])
#%%%
s = pd.Series([x for x in range(len(data_all))])
data_all=data_all.set_index(s)


#%%
plt.figure()
plot_parms = {'data':data_all, 
              'x':'phase', 
              'y':'bestH0',
              # 'hue':'phase',
              # 'hue_order':['G1','G1/S','S/G2','M'],
              'order':['G1','G1/S','S/G2','M'],
              'palette':['red','orange','green','limegreen'],
              'dodge':False}

ax = sns.boxplot(**plot_parms)
ax = sns.swarmplot(**plot_parms)
pairs=[("G1","G1/S"),('G1','S/G2'),('G1/S','S/G2'),('S/G2','M'),('G1','M'),('G1/S','M')]

annotator = Annotator(ax, pairs, **plot_parms)
annotator.configure(test='Mann-Whitney',text_format='star', loc='inside')
annotator.apply_and_annotate()

#plt.tight_layout()
plt.title('23-05, 12-06, 04-06')

plt.show()
#%%
data_M=data_all[data_all['phase'].str.contains('M') == True]
data_G1=data_all[data_all['phase'].str.contains('G1') == True]
data_G2=data_all[data_all['phase'].str.contains('S/G2') == True]
data_S=data_all[data_all['phase'].str.contains('G1/S') == True]
#%%
poptG1,pcovG1=curve_fit(func,data_G1['surroundingThickness'],data_G1['ctFieldFluctuAmpli'])
poptM,pcovM=curve_fit(func,data_M['surroundingThickness'],data_M['ctFieldFluctuAmpli'])

#%%
plt.figure()
xdata = np.linspace(0,1200)
y=func(xdata,*poptM)
plt.plot(xdata,y,'b')
sns.scatterplot(data=data_all,x='surroundingThickness',y='ctFieldFluctuAmpli',hue='phase')

#%%
data_cell_all=pd.DataFrame()
for i in data_all['cellID']:
    j=data_all.index[data_all['cellID']==i]
    data_cell_all.at[i,'median_Eeff']=data_all.at[j[0],'median_Eeff']
    data_cell_all.at[i,'median_h']=data_all.at[j[0],'median_h']
    data_cell_all.at[i,'phase']=data_all.at[j[0],'phase']
    data_cell_all.at[i,'NLI']=data_all.at[j[0],'NLI']
    data_cell_all.at[i,'surroundingThickness']=data_all.at[j[0],'surroundingThickness']
    data_cell_all.at[i,'ctFieldFluctuAmpli']=data_all.at[j[0],'ctFieldFluctuAmpli']
#%%
plt.figure()
sns.boxplot(data=data_cell_all,x='phase',y='median_h',hue='phase',order=['G1','G1/S','S/G2','M'],palette=['green','red','orange','limegreen'])
sns.swarmplot(data=data_cell_all,x='phase',y='median_h',hue=data_cell_all.index,order=['G1','G1/S','S/G2','M'],legend=False)


#%%
plt.figure()
sns.scatterplot(data=data_cell_all,x='median_h',y='median_Eeff',hue='phase',palette=['green','red','orange','limegreen'])
plt.xscale('log')
plt.yscale('log')


#%%
dataC_M=data_cell_all[data_cell_all['phase'].str.contains('M') == True]
dataC_G1=data_cell_all[data_cell_all['phase'].str.contains('G1') == True]
dataC_G2=data_cell_all[data_cell_all['phase'].str.contains('S/G2') == True]
dataC_S=data_cell_all[data_cell_all['phase'].str.contains('G1/S') == True]

#%%
def funcLog(x,a,b):
    return a*x**b
#%%
poptG1,pcovG1=curve_fit(funcLog,np.log(dataC_G1['median_h']),np.log(dataC_G1['ctFieldFluctuAmpli']))
#%%
poptM,pcovM=curve_fit(funcLog,np.log(dataC_M['median_h']),np.log(dataC_M['ctFieldFluctuAmpli']))
#%%
poptG2,pcovG2=curve_fit(funcLog,np.log(dataC_G2['median_h']),np.log(dataC_G2['ctFieldFluctuAmpli']))
#%%
poptS,pcovS=curve_fit(funcLog,np.log(dataC_S['median_h']),np.log(dataC_S['ctFieldFluctuAmpli']))
#%%
paramsG1, resultsG1 = ufun.fitLineHuber((np.log(dataC_G1['median_h'])), np.log(dataC_G1['median_Eeff']))
k1 = np.exp(paramsG1.iloc[0])
a1 = paramsG1.iloc[1]
fit_xG1 = np.linspace(np.min(dataC_G1['median_h']), np.max(dataC_G1['median_h']), 50)
fit_yG1 = k1 * fit_xG1**a1

paramsM, resultsM = ufun.fitLineHuber((np.log(dataC_M['median_h'])), np.log(dataC_M['median_Eeff']))
kM = np.exp(paramsM.iloc[0])
aM = paramsM.iloc[1]
fit_xM = np.linspace(np.min(dataC_M['median_h']), np.max(dataC_M['median_h']), 50)
fit_yM = kM * fit_xM**aM

paramsG2, resultsG2 = ufun.fitLineHuber((np.log(dataC_G2['median_h'])), np.log(dataC_G2['median_Eeff']))
k2 = np.exp(paramsG2.iloc[0])
a2 = paramsG2.iloc[1]
fit_xG2 = np.linspace(np.min(dataC_G2['median_h']), np.max(dataC_G2['median_h']), 50)
fit_yG2 = k2 * fit_xG2**a2

paramsS, resultsS = ufun.fitLineHuber((np.log(dataC_S['median_h'])), np.log(dataC_S['median_Eeff']))
kS = np.exp(paramsS.iloc[0])
aS = paramsS.iloc[1]
fit_xS = np.linspace(np.min(dataC_S['median_h']), np.max(dataC_S['median_h']), 50)
fit_yS = kS * fit_xS**aS

plt.figure()
sns.scatterplot(data=data_cell_all,x=data_cell_all['median_h'],y=data_cell_all['median_Eeff'],hue='phase',hue_order=['G1','G1/S','S/G2','M'],palette=['red','orange','green','limegreen'])#,style=[data_cell_all.index[k][:8] for k in range(len(data_cell_all.index))])
plt.plot(fit_xG1,fit_yG1,'r',label=" Y = {:.2e} * X^{:.2f}".format(k1, a1))
plt.plot(fit_xS,fit_yS,'orange',label=" Y = {:.2e} * X^{:.2f}".format(kS, aS))
plt.plot(fit_xG2,fit_yG2,'green',label=" Y = {:.2e} * X^{:.2f}".format(k2, a2))
plt.plot(fit_xM,fit_yM,'limegreen',label=" Y = {:.2e} * X^{:.2f}".format(kM, aM))

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.title('Experiments from 23-05, 04-06 and 12-06 - robust fit')
#fig.set_size_inches(10,5)

#%%
paramsG1, resultsG1 = ufun.fitLineHuber((np.log(data_G1['bestH0'])), np.log(data_G1['E_effective']))
k1 = np.exp(paramsG1.iloc[0])
a1 = paramsG1.iloc[1]
fit_xG1 = np.linspace(np.min(data_G1['bestH0']), np.max(data_G1['bestH0']), 50)
fit_yG1 = k1 * fit_xG1**a1

paramsM, resultsM = ufun.fitLineHuber((np.log(data_M['bestH0'])), np.log(data_M['E_effective']))
kM = np.exp(paramsM.iloc[0])
aM = paramsM.iloc[1]
fit_xM = np.linspace(np.min(data_M['bestH0']), np.max(data_M['bestH0']), 50)
fit_yM = kM * fit_xM**aM

paramsG2, resultsG2 = ufun.fitLineHuber((np.log(data_G2['bestH0'])), np.log(data_G2['E_effective']))
k2 = np.exp(paramsG2.iloc[0])
a2 = paramsG2.iloc[1]
fit_xG2 = np.linspace(np.min(data_G2['bestH0']), np.max(data_G2['bestH0']), 50)
fit_yG2 = k2 * fit_xG2**a2

paramsS, resultsS = ufun.fitLineHuber((np.log(data_S['bestH0'])), np.log(data_S['E_effective']))
kS = np.exp(paramsS.iloc[0])
aS = paramsS.iloc[1]
fit_xS = np.linspace(np.min(data_S['bestH0']), np.max(data_S['bestH0']), 50)
fit_yS = kS * fit_xS**aS

plt.figure()
sns.scatterplot(data=data_all,x=data_all['bestH0'],y=data_all['E_effective'],hue='phase',hue_order=['G1','G1/S','S/G2','M'],palette=['red','orange','green','limegreen'],size=3)
plt.plot(fit_xG1,fit_yG1,'r',label=" Y = {:.2e} * X^{:.2f}".format(k1, a1))
plt.plot(fit_xG2,fit_yG2,'green',label=" Y = {:.2e} * X^{:.2f}".format(k2, a2))
plt.plot(fit_xM,fit_yM,'limegreen',label=" Y = {:.2e} * X^{:.2f}".format(kM, aM))
plt.plot(fit_xS,fit_yS,'orange',label=" Y = {:.2e} * X^{:.2f}".format(kS, aS))
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.title('Experiments from 23-05, 04-06 and 12-06 - robust fit')

#%%
data_lin=data_all[data_all['NLI_type'].str.contains('Linear') == True]
data_nlin=data_all[data_all['NLI_type'].str.contains('Non-linear') == True]
data_inter=data_all[data_all['NLI_type'].str.contains('Intermediate') == True]

#%%
paramsl, resultsl = ufun.fitLine((np.log(data_lin['bestH0'])), np.log(data_lin['E_effective']))
kl = np.exp(paramsl.iloc[0])
al = paramsl.iloc[1]
fit_xl = np.linspace(np.min(data_lin['bestH0']), np.max(data_lin['bestH0']), 50)
fit_yl = kl * fit_xl**al

paramsnl, resultsnl = ufun.fitLine((np.log(data_nlin['bestH0'])), np.log(data_nlin['E_effective']))
knl = np.exp(paramsnl.iloc[0])
anl = paramsnl.iloc[1]
fit_xnl = np.linspace(np.min(data_nlin['bestH0']), np.max(data_nlin['bestH0']), 50)
fit_ynl = knl * fit_xnl**anl

paramsI, resultsI = ufun.fitLine((np.log(data_inter['bestH0'])), np.log(data_inter['E_effective']))
kI = np.exp(paramsI.iloc[0])
aI = paramsI.iloc[1]
fit_xI = np.linspace(np.min(data_inter['bestH0']), np.max(data_inter['bestH0']), 50)
fit_yI = kI * fit_xI**aI


plt.figure()
sns.scatterplot(data=data_all,x=data_all['bestH0'],y=data_all['E_effective'],hue='NLI_type',palette=['cornflowerblue','green','red'],size=3)
plt.plot(fit_xnl,fit_ynl,'cornflowerblue',label=" Y = {:.2e} * X^{:.2f} \n R2 = {:.3f}".format(knl, anl,resultsnl.rsquared))
plt.plot(fit_xl,fit_yl,'green',label=" Y = {:.2e} * X^{:.2f} \n R2 = {:.3f}".format(kl, al,resultsl.rsquared))
#plt.plot(fit_xI,fit_yI,'r',label=" Y = {:.2e} * X^{:.2f}\n R2 = {:.3f}".format(kI, aI,resultsI.rsquared))
plt.legend()
plt.xscale('log')
plt.yscale('log')
#plt.xlim(10**2,10**3)
plt.title('Experiments from 23-05, 04-06 and 12-06')
#%%

paramsG1, resultsG1 = ufun.fitLine((np.log(data_G1['median_h'])), np.log(data_G1['ctFieldFluctuAmpli']))
k1 = np.exp(paramsG1.iloc[0])
a1 = paramsG1.iloc[1]
fit_xG1 = np.linspace(np.min(data_G1['median_h']), np.max(data_G1['median_h']), 50)
fit_yG1 = k1 * fit_xG1**a1

paramsM, resultsM = ufun.fitLine((np.log(data_M['median_h'])), np.log(data_M['ctFieldFluctuAmpli']))
kM = np.exp(paramsM.iloc[0])
aM = paramsM.iloc[1]
fit_xM = np.linspace(np.min(data_M['median_h']), np.max(data_M['median_h']), 50)
fit_yM = kM * fit_xM**aM

paramsG2, resultsG2 = ufun.fitLine((np.log(data_G2['median_h'])), np.log(data_G2['ctFieldFluctuAmpli']))
k2 = np.exp(paramsG2.iloc[0])
a2 = paramsG2.iloc[1]
fit_xG2 = np.linspace(np.min(data_G2['median_h']), np.max(data_G2['median_h']), 50)
fit_yG2 = k2 * fit_xG2**a2

paramsS, resultsS = ufun.fitLine((np.log(data_S['median_h'])), np.log(data_S['ctFieldFluctuAmpli']))
kS = np.exp(paramsS.iloc[0])
aS = paramsS.iloc[1]
fit_xS = np.linspace(np.min(data_S['median_h']), np.max(data_S['median_h']), 50)
fit_yS = kS * fit_xS**aS

plt.figure()
sns.scatterplot(data=data_cell_all,x=data_cell_all['median_h'],y=data_cell_all['ctFieldFluctuAmpli'],hue='phase',hue_order=['G1','G1/S','S/G2','M'],palette=['red','orange','green','limegreen'])#,style=[data_cell_all.index[k][:8] for k in range(len(data_cell_all.index))])
plt.plot(fit_xG1,fit_yG1,'r',label=" Y = {:.2e} * X^{:.2f}\n R2 = {:.3f}".format(k1, a1,resultsG1.rsquared))
plt.plot(fit_xS,fit_yS,'orange',label=" Y = {:.2e} * X^{:.2f}\n R2 = {:.3f}".format(kS, aS,resultsS.rsquared))
plt.plot(fit_xG2,fit_yG2,'green',label=" Y = {:.2e} * X^{:.2f}\n R2 = {:.3f}".format(k2, a2,resultsG2.rsquared))
plt.plot(fit_xM,fit_yM,'limegreen',label=" Y = {:.2e} * X^{:.2f}\n R2 = {:.3f}".format(kM, aM,resultsM.rsquared))

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.title('Experiments from 23-05, 04-06 and 12-06')
#fig.set_size_inches(10,5)
#%%
fig,ax=plt.subplots()
sns.scatterplot(data=data_cell_all,x=data_cell_all['median_h'],y=data_cell_all['ctFieldFluctuAmpli'],hue='phase',palette=['green','red','orange','limegreen'])
X=np.linspace(100,1200)
# Y=funcLog(X,*poptG1)
# Y_G2=funcLog(X,*poptG2)
# Y_S=funcLog(X,*poptS)
# Y_M=funcLog(X,*poptM)

plt.plot(fit_xG1,fit_yG1,'r',label=" Y = {:.2e} * X^{:.2f}".format(k1, a1))
plt.plot(fit_xG2,fit_yG2,'green',label=" Y = {:.2e} * X^{:.2f}".format(k2, a2))
plt.plot(fit_xM,fit_yM,'limegreen',label=" Y = {:.2e} * X^{:.2f}".format(kM, aM))
plt.plot(fit_xS,fit_yS,'orange',label=' Y = {:.2e} * X^{:.2f}'.format(kS, aS))
plt.legend()
# plt.plot(X,Y,'r')
# plt.plot(X,Y_G2,'green')
# plt.plot(X,Y_M,'limegreen')
# plt.plot(X,Y_S,'orange')
plt.xscale('log')
plt.yscale('log')
plt.title('Experiments from 23-05, 04-06 and 12-06 - robust fit')
#%%
plt.figure()
p3=sns.swarmplot(data=data_all,x='phase',y='NLI',hue='cellID',size=4,order=['G1','G1/S','S/G2','M'])
p3.axhline(3)
p3.axhline(-3)
p3.axhline(0.3,color='r')
p3.axhline(-0.3,color='r')
plt.title('3 experiments')

#%%
N_all=percentageNLIphase(data_all)

fig, ax = plt.subplots()
rects1 = ax.bar(N_all.index, N_all['Linear_percentage'],label='linear')
rects2 = ax.bar(N_all.index, N_all['Intermediate_percentage'],bottom=N_all['Linear_percentage'],label='intermediate')
rects3 = ax.bar(N_all.index, N_all['Non-linear_percentage'],bottom=N_all['Linear_percentage']+N_all['Intermediate_percentage'],label='non_linear',color='brown')
N_comp=[i for i in N_all['N_comp']]
ax.bar_label(rects3,labels=[f'N={e:0.0f}' for e in N_comp])
# plt.bar(N_all.index,N_all['Linear_percentage'],label='linear')
# plt.bar(N_all.index,N_all['Intermediate_percentage'],bottom=N_all['Linear_percentage'],label='intermediate')
# plt.bar(N_all.index,N_all['Non-linear_percentage'],bottom=N_all['Linear_percentage']+N_all['Intermediate_percentage'],label='non_linear')
# for i in range(len(N_all['N_comp'])):
#     n=N_all.at[N_all.index[i],'N_comp']
#     plt.title('N=%i' %n)
plt.title('All cells - NLI index delimitations ]-0.3;0.3[')
ax.legend(bbox_to_anchor=(1.01, 1.01))
fig.set_size_inches(10, 8)
plt.show()


#%%
plt.figure()
plot_parms = {'data':data_all, 
              'x':'phase', 
              'y':'bestH0',
              # 'hue':'phase',
              # 'hue_order':['G1','G1/S','S/G2','M'],
              'order':['G1','G1/S','S/G2','M'],
              'palette':['red','orange','green','limegreen'],
              'dodge':False}

ax = sns.boxplot(**plot_parms)
ax = sns.swarmplot(**plot_parms)
pairs=[("G1","G1/S"),('G1','S/G2'),('G1/S','S/G2'),('S/G2','M'),('G1','M'),('G1/S','M')]

annotator = Annotator(ax, pairs, **plot_parms)
annotator.configure(test='Mann-Whitney',text_format='star', loc='inside')
annotator.apply_and_annotate()

#plt.tight_layout()
plt.title('May-June experiments - Mann-Whitney test')
plt.show()
#%%