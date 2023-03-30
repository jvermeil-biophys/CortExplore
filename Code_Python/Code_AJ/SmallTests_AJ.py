# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:00:13 2022

@author: anumi
"""


"""

Tests to plot the first Pincher curves. Not so important but didn't want to trash just in case

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import re
from datetime import date
import sys
import scipy.stats as st


# Local imports
COMPUTERNAME = os.environ['COMPUTERNAME']
if COMPUTERNAME == 'ORDI-JOSEPH':
    mainDir = "C://Users//JosephVermeil//Desktop//ActinCortexAnalysis"
    rawDir = "D://MagneticPincherData"
    ownCloudDir = "C://Users//JosephVermeil//ownCloud//ActinCortexAnalysis"
elif COMPUTERNAME == 'LARISA':
    mainDir = "C://Users//Joseph//Desktop//ActinCortexAnalysis"
    rawDir = "F://JosephVermeil//MagneticPincherData"    
    ownCloudDir = "C://Users//Joseph//ownCloud//ActinCortexAnalysis"
elif COMPUTERNAME == 'DESKTOP-K9KOJR2':
    mainDir = "C://Users//anumi//OneDrive//Desktop//CortExplore"
    rawDir = "D:/Anumita/MagneticPincherData"  
elif COMPUTERNAME == '':
    mainDir = "C://Users//josep//Desktop//ActinCortexAnalysis"
    ownCloudDir = "C://Users//josep//ownCloud//ActinCortexAnalysis"

# Add the folder to path
sys.path.append(mainDir + "//Code_Python")
import utilityFunctions_JV as jvu

#%% Global constants

bead_dia = 4.503


# These regex are used to correct the stupid date conversions done by Excel
dateFormatExcel = re.compile(r'\d{2}/\d{2}/\d{4}')
dateFormatExcel2 = re.compile(r'\d{2}-\d{2}-\d{4}')
dateFormatOk = re.compile(r'\d{2}-\d{2}-\d{2}')

SCALE_100X = 15.8 # pix/µm 
NORMAL  = '\033[0m'
RED  = '\033[31m' # red
GREEN = '\033[32m' # green
ORANGE  = '\033[33m' # orange
BLUE  = '\033[36m' # blue

# %% Directories adress

mainDataDir = 'D:/Anumita/MagneticPincherData'
experimentalDataDir = os.path.join(mainDir, "Data_Experimental_AJ")
dataDir = os.path.join(mainDir, "Data_Analysis")
timeSeriesDataDir = os.path.join(mainDataDir, "Data_TimeSeries")

figDir = os.path.join(mainDataDir, "Figures")
todayFigDir = os.path.join(figDir, "Historique/" + str(date.today()))

#%% Pandas test code

import numpy as np
import pandas as pd

D = {'subject': ['physics', 'hindi', 'english', 'sport', 
                 'physics', 'hindi', 'english', 'sport', 
                 'physics', 'hindi', 'english', 'sport',
                 'physics', 'hindi', 'english', 'sport'],
     'teacher': ['Mrs B.', 'Mr A.', 'Mr A.', 'Mrs C.', 
                'Mrs B.', 'Mr A.', 'Mr A.', 'Mrs C.', 
                'Mrs B.', 'Mr A.', 'Mr A.', 'Mrs C.',
                'Mrs B.', 'Mr A.', 'Mr A.', 'Mrs C.'],
     'grades': [89, 58, 77, 82, 95, 62, 80, 99, 
                59, 85, 97, 47, 98, 65, 79, 82],
     'coefficient':[1,1,1,1,2,3,2,3,6,5,7,6,8,7,9,10]}

df = pd.DataFrame(D)

group_by_subject = df.groupby(by = 'subject')

df_average = group_by_subject.agg({'grades':['mean', 'std', 'count', 'sum'], 
                                   'teacher':'first'})

#%% shitty test plots

#%%% Plotting all three graphs (3D, 2D and Dz)

expt = '20220412_100xoil_3t3optorhoa_4.5beads_15mT_Mechanics'
folder = '22-04-12_M1_P1_C5_disc20um'
date = '22.04.12'

file = 'C:/Users/anumi/OneDrive/Desktop/ActinCortexAnalysis/Data_Analysis/TimeSeriesData/'+folder+'_PY.csv'
data = pd.read_csv(file, sep=';')
 #in um

xyz_dist = data['D3'] - bead_dia
xy_dist = data['D2'] - bead_dia
dz = data['dz']
t = (data['T']*1000)/60
#t = np.linspace(0,len(data['T']),len(data['T']))
Nan_thresh = 3

outlier = np.where(xyz_dist > Nan_thresh)[0]
xyz_dist[outlier] = np.nan
xy_dist[outlier] = np.nan
dz[outlier] = np.nan

plt.style.use('dark_background')

fig = plt.figure(figsize=(20,20))
fig.suptitle(folder, fontsize=16)
plt.rcParams.update({'font.size': 25})

ax1 = plt.subplot(311)
plt.plot(t, xyz_dist)
plt.axvline(x = 5, color = 'r', label = 'Activation begins')
plt.axvline(x = 8, color = 'r', label = 'Activation begins')

plt.title('3D Distance vs. Time')
#plt.xlim(0,25)


# share x only
ax2 = plt.subplot(312)
plt.plot(t, xy_dist)
plt.axvline(x = 5, color = 'r', label = 'Activation begins')
plt.axvline(x = 8, color = 'r', label = 'Activation begins')

plt.title('2D Distance (XY) vs. Time (mins)')
# make these tick labels invisible

# share x and y
ax3 = plt.subplot(313)
plt.axvline(x = 5, color = 'r', label = 'Activation begins')
plt.axvline(x = 8, color = 'r', label = 'Activation begins')

plt.title('Dz vs. Time (mins)')
plt.plot(t, dz)

plt.show()

plt.savefig('D:/Anumita/PincherPlots/'+folder+'_DistancevTime.jpg')

#%% Plotting just 3D graphs

# %%% Just 3D graphs

expt = '20220322_100xoil_3t3optorhoa_4.5beads_15mT'
folder = '22-03-22_M2_P1_C5_disc20um'
date = '22.03.22'

file = 'C:/Users/anumi/OneDrive/Desktop/ActinCortexAnalysis/Data_Analysis/TimeSeriesData/'+folder+'_PY.csv'
data = pd.read_csv(file, sep=';')
bead_dia = 4.503 #in um

xyz_dist = data['D3'] - bead_dia
xy_dist = data['D2'] - bead_dia
dz = data['dz']
t = (data['T']*1000)/60
#t = np.linspace(0,len(data['T']),len(data['T']))
# Nan_thresh = 3

# outlier = np.where(xyz_dist > Nan_thresh)[0]
# xyz_dist[outlier] = np.nan
# xy_dist[outlier] = np.nan
# dz[outlier] = np.nan

plt.style.use('dark_background')

fig = plt.figure(figsize=(30,10))
fig.suptitle(folder, fontsize=16)
plt.rcParams.update({'font.size': 40})
plt.axvline(x = 5, color = 'r', label = 'Activation begins')
# plt.xlim(0.5,8)

plt.ylabel('Thickness (um)')
plt.xlabel('Time (mins)')

plt.plot(t, xyz_dist)
plt.title('3D Distance vs. Time')

plt.savefig('D:/Anumita/PincherPlots/'+folder+'_3DistancevTime.jpg')


# %%
expt1 = '20220322_100xoil_3t3optorhoa_4.5beads_15mT'
folder1 = '22-03-22_M4_P1_C5_disc20um'
file = 'C:/Users/anumi/OneDrive/Desktop/ActinCortexAnalysis/Data_Analysis/TimeSeriesData/'+folder1+'_PY.csv'
data1 = pd.read_csv(file, sep=';')
#t1 = np.linspace(0,len(data1['T']),len(data1['T']))
t1 = (data1['T']*1000/60)
xy_dist1 = data1['D3'] - bead_dia

expt2 = '20220322_100xoil_3t3optorhoa_4.5beads_15mT'
folder2 = '22-03-22_M3_P1_C5_disc20um'
file = 'C:/Users/anumi/OneDrive/Desktop/ActinCortexAnalysis/Data_Analysis/TimeSeriesData/'+folder2+'_PY.csv'
data2 =  pd.read_csv(file, sep=';')
# t2 = np.linspace(0,len(data2['T']),len(data2['T']))
t2 = (data2['T']*1000/60)
xy_dist2 = data2['D3'] - bead_dia

Nan_thresh1 = 3
Nan_thresh2 = 3

outlier = np.where(xy_dist2 > Nan_thresh2)[0]
xy_dist2[outlier] = np.nan

outlier = np.where(xy_dist1 > Nan_thresh1)[0]
xy_dist1[outlier] = np.nan

# expt3 = '20211220_100xoil_3t3optorhoa_4.5beads_15mT'
# folder3 = '21-12-20_M3_P1_C2_disc20um'
# file = 'C:/Users/anumi/OneDrive/Desktop/ActinCortexAnalysis/Data_Analysis/TimeSeriesData/'+folder3+'_PY.csv'
# data3 = pd.read_csv(file, sep=';')
# t3 = np.linspace(0,len(data3['T']),len(data3['T']))
# xy_dist3 = data3['D2'] - bead_dia

plt.style.use('dark_background')


fig= plt.figure(figsize=(30,10))
fig.suptitle(folder, fontsize=16)

# right_side = fig.spines["right"]
# right_side.set_visible(False)
# top_side = fig.spines["top"]
# top_side.set_visible(False)


plt.rcParams.update({'font.size':35})
plt.title('3D Distance vs Time')
plt.ylabel('Thickness (um)')
plt.xlabel('Time (secs)')

plt.plot(t1, xy_dist1, label="Activation away from beads", color = 'orange')
plt.plot(t2, xy_dist2, label='Activation at beads', color = 'royalblue')
plt.axvline(x = 5, color = 'r')

# plt.plot(t3, xy_dist3, label='90s')
plt.legend()
plt.show()
plt.savefig('D:/Anumita/PincherPlots/C3_DistancevTime.jpg')

# %% All curves

expt1 = '20211220_100xoil_3t3optorhoa_4.5beads_15mT'
folder1 = '21-12-20_M1_P1_C1_disc20um'
file = 'C:/Users/anumi/OneDrive/Desktop/ActinCortexAnalysis/Data_Analysis/TimeSeriesData/'+folder1+'_PY.csv'
data1 = pd.read_csv(file, sep=';')
#t1 = np.linspace(0,len(data1['T']),len(data1['T']))
t1 = (data1['T']*1000)
xy_dist1 = data1['D3'] - bead_dia

expt2 = '20211220_100xoil_3t3optorhoa_4.5beads_15mT'
folder2 = '21-12-20_M1_P1_C3_disc20um'
file = 'C:/Users/anumi/OneDrive/Desktop/ActinCortexAnalysis/Data_Analysis/TimeSeriesData/'+folder2+'_PY.csv'
data2 =  pd.read_csv(file, sep=';')
# t2 = np.linspace(0,len(data2['T']),len(data2['T']))
t2 = (data2['T']*1000)
xy_dist2 = data2['D3'] - bead_dia

expt3 = '20220203_100xoil_3t3optorhoa_4.5beads_15mT'
folder3 = '22-02-03_M4_P1_C1_disc20um'
file = 'C:/Users/anumi/OneDrive/Desktop/ActinCortexAnalysis/Data_Analysis/TimeSeriesData/'+folder3+'_PY.csv'
data3 = pd.read_csv(file, sep=';')
#t1 = np.linspace(0,len(data1['T']),len(data1['T']))
t3 = (data3['T']*1000)
xy_dist3 = data3['D3'] - bead_dia

expt4 = '20220203_100xoil_3t3optorhoa_4.5beads_15mT'
folder4 = '22-02-03_M3_P1_C1_disc20um'
file = 'C:/Users/anumi/OneDrive/Desktop/ActinCortexAnalysis/Data_Analysis/TimeSeriesData/'+folder4+'_PY.csv'
data4 =  pd.read_csv(file, sep=';')
# t2 = np.linspace(0,len(data2['T']),len(data2['T']))
t4 = (data4['T']*1000)
xy_dist4 = data4['D3'] - bead_dia

expt5 = '20220203_100xoil_3t3optorhoa_4.5beads_15mT'
folder5 = '22-02-03_M5_P1_C3_disc20um'
file = 'C:/Users/anumi/OneDrive/Desktop/ActinCortexAnalysis/Data_Analysis/TimeSeriesData/'+folder5+'_PY.csv'
data5 =  pd.read_csv(file, sep=';')
# t2 = np.linspace(0,len(data2['T']),len(data2['T']))
t5 = (data5['T']*1000)
xy_dist5 = data5['D3'] - bead_dia


Nan_thresh1 = 3
Nan_thresh2 = 3

outlier = np.where(xy_dist2 > Nan_thresh2)[0]
xy_dist2[outlier] = np.nan

outlier = np.where(xy_dist1 > Nan_thresh1)[0]
xy_dist1[outlier] = np.nan

# expt3 = '20211220_100xoil_3t3optorhoa_4.5beads_15mT'
# folder3 = '21-12-20_M3_P1_C2_disc20um'
# file = 'C:/Users/anumi/OneDrive/Desktop/ActinCortexAnalysis/Data_Analysis/TimeSeriesData/'+folder3+'_PY.csv'
# data3 = pd.read_csv(file, sep=';')
# t3 = np.linspace(0,len(data3['T']),len(data3['T']))
# xy_dist3 = data3['D2'] - bead_dia

plt.style.use('dark_background')


fig= plt.figure(figsize=(20,20))
# fig.suptitle(fontsize=16)

# right_side = fig.spines["right"]
# right_side.set_visible(False)
# top_side = fig.spines["top"]
# top_side.set_visible(False)


plt.rcParams.update({'font.size':35})
plt.title('3D Distance vs Time')
plt.ylabel('Thickness (nm)')
plt.xlabel('Time (secs)')

plt.plot(t1, xy_dist1, label=folder1, color = 'red')
plt.plot(t2, xy_dist2, label=folder2, color = 'blue')
plt.plot(t3, xy_dist3, label=folder3, color = 'orange')
plt.plot(t4, xy_dist4, label=folder4, color = 'pink')
plt.plot(t5, xy_dist5, label=folder5, color='yellow')
# plt.plot(t3, xy_dist3, label='90s')
plt.legend()
plt.show()
plt.savefig('D:/Anumita/PincherPlots/All_DistancevTime.jpg')


# %% Plotting with fluorescence recruitment values

expt = '20220301_100xoil_3t3optorhoa_4.5beads_15mT'
folder = '22-03-01_M1_P1_C3_disc20um'
date = '22.03.01'

file = 'D:/Anumita/MagneticPincherData/Raw/'+date+'/'+folder+'_Values.csv'
data = pd.read_csv(file, sep=',')

radius = np.asarray(data['Radius_[pixels]'])

# for i in range(np.shape(data)[1]):
#     plt
