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
import datetime as dt
from datetime import date
import sys
from skimage import io
import CortexPaths as cp

import scipy.stats as st
import tifffile as tiff
import cv2
import GraphicStyles as gs

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
# import utilityFunctions_JV as jvu

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
folder = '23-09-21_M1_P1_C2_disc20um'
date = '22.09.21'

file = 'D:/Anumita/MagneticPincherData/Data_TimeSeries/'+folder+'_PY.csv'
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

#%%

cells = os.listdir()
cells = np.unique(cellDf1['cellID'])

for i in cells:
    axes = taka.plotCellTimeSeriesData(i, save = True, savePath = pathTSPlots)
    
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



#%% Changing the names of some mis-named files

path = 'D:/Anumita/MagneticPincherData/Raw/23.04.25/Y27'
newPath = 'D:/Anumita/MagneticPincherData/Raw/23.04.25/newName'

files = os.listdir(path)

for file in files:
    oldName = file.split('_')
    manip = oldName[1]
    if manip == 'M3':
        oldName[1] = 'M4'
    if manip == 'M4':
        oldName[1] = 'M5'
    oldName = '_'.join(oldName)
    os.rename(os.path.join(path, file), os.path.join(newPath, oldName))  
        
        
#%% Code to deleted some xtra triplets that arise when you use activations with different 
#exposure times (prob with labview)

pathSave = 'D:/Anumita/MagneticPincherData/Raw/23.07.12'
path = 'D:/Anumita/MagneticPincherData/Raw/23.07.12/Archive'
allFiles = os.listdir(path)
selectedCell = 'M2_P1_C1'
allTifs = [i for i in allFiles if selectedCell in i and '.tif' in i]
allFields = [i for i in allFiles if selectedCell in i and '_Field' in i]
nLoop = 1
  
for i in range(len(allTifs)):
    deletedFrames = []
    tif = allTifs[i]
    file = allFields[i]
    
    print(tif)
    
    # totalFrames = (np.linspace(0, len(stack), len(stack))
    
    if not os.path.exists(os.path.join(pathSave, tif)):
        field = np.loadtxt(os.path.join(path, file), delimiter = '\t')
        stack = tiff.imread(os.path.join(path, tif))
        totalLoops = (len(stack) - 437)//439
        for j in range(totalLoops):
            if j >= nLoop :
                deletedFrames.extend([(j*439)+439, (j*439)+440, (j*439)+441])
            elif j < nLoop: 
                deletedFrames.extend([(j*439)+438, (j*439)+439, (j*439)+440])
    
        deletedFrames = np.asarray(deletedFrames) - 1
        # totalFrames.remove(deletedFrames)
        field_new = np.delete(field, deletedFrames, 0)
        stack_new = np.delete(stack, deletedFrames, 0)
        
        np.savetxt(os.path.join(pathSave, file), field_new, delimiter = '\t')
        io.imsave(os.path.join(pathSave, tif), stack_new)
        
    else:
        print(gs.GREEN + tif + ' already exists' + gs.NORMAL)
    
#%% Code to deleted some xtra triplets that arise when you use activations with different 
#exposure times (prob with labview)

pathSave = 'D:/Anumita/MagneticPincherData/Raw/23.07.12'
path = 'D:/Anumita/MagneticPincherData/Raw/23.07.12/Archive'
allFiles = os.listdir(path)
selectedCell = 'M2_P1_C1'
allTifs = [i for i in allFiles if selectedCell in i and '.tif' in i]
allFields = [i for i in allFiles if selectedCell in i and '_Field' in i]

for i in range(len(allTifs)):
    deletedFrames = []
    tif = allTifs[i]
    file = allFields[i]
    
    print(tif)
    
    # totalFrames = (np.linspace(0, len(stack), len(stack))
    
    if not os.path.exists(os.path.join(pathSave, tif)):
        field = np.loadtxt(os.path.join(path, file), delimiter = '\t')
        stack = tiff.imread(os.path.join(path, tif))
        totalLoops = (len(stack) - 437)//439
        for j in range(totalLoops+1):
            deletedFrames.extend([(j*439)+19, (j*439)+20, (j*439)+21])
          
    
        deletedFrames = np.asarray(deletedFrames) - 1
        # totalFrames.remove(deletedFrames)
        field_new = np.delete(field, deletedFrames, 0)
        stack_new = np.delete(stack, deletedFrames, 0)
        
        np.savetxt(os.path.join(pathSave, file), field_new, delimiter = '\t')
        io.imsave(os.path.join(pathSave, tif), stack_new)
        
    else:
        print(gs.GREEN + tif + ' already exists' + gs.NORMAL)
    
     
    
#%% Code to deleted some xtra triplets that arise when you use activations with different 
#exposure times (prob with labview)
# This section is to use for cells with no activation

pathSave = 'D:/Anumita/MagneticPincherData/Raw/23.07.12'
path = 'D:/Anumita/MagneticPincherData/Raw/23.07.12/Archive'
allFiles = os.listdir(path)
selectedCell = 'M1_P3_C4'
allTifs = [i for i in allFiles if selectedCell in i and '.tif' in i]
allFields = [i for i in allFiles if selectedCell in i and '_Field' in i]
  
for i in range(len(allTifs)):
    deletedFrames = []
    tif = allTifs[i]
    file = allFields[i]
    
    print(tif)
    if not os.path.exists(os.path.join(pathSave, tif)):
        field = np.loadtxt(os.path.join(path, file), delimiter = '\t')
        
        stack = tiff.imread(os.path.join(path, tif))
        # totalFrames = (np.linspace(0, len(stack), len(stack))
        totalLoops = (len(stack) - 437)//439
        
        for j in range(totalLoops + 1):
            deletedFrames.extend([(j*439)+1, (j*439)+2, (j*439)+3])
        
        deletedFrames = np.asarray(deletedFrames) - 1
        
        
        # totalFrames.remove(deletedFrames)
        field_new = np.delete(field, deletedFrames, 0)
        stack_new = np.delete(stack, deletedFrames, 0)
        
        np.savetxt(os.path.join(pathSave, file), field_new, delimiter = '\t')
        io.imsave(os.path.join(pathSave, tif), stack_new)
    else:
        print(gs.GREEN + tif + ' already exists' + gs.NORMAL)
        
#%% Code to create videos with a decently constant frame rate with a timestamp

pathSave = 'D:/Anumita/MagneticPincherData/Raw/Videos'
path = 'D:/Anumita/MagneticPincherData/Raw/ToConvert/'
allFiles = os.listdir(path)
allTifs = [i for i in allFiles if '22-10-06' in i and '.tif' in i]
allFields = [i for i in allFiles if '22-10-06' in i and '_Field' in i]
allLogs = [i for i in allFiles if '22-10-06' in i and '_LogPY' in i]
activationFrames = np.asarray([])

for i in range(len(allTifs)):
    selectedFrames = []
    tif = allTifs[i]
    file = allFields[i]
    log = allLogs[i]
    
    print(tif)
    field = np.loadtxt(os.path.join(path, file), delimiter = '\t')
    status = pd.read_csv(os.path.join(path, log), delimiter = '\t')
    
    ctfield = status['Slice'][status['status_frame'] == 3.0].values
    selectedFrames.extend(ctfield-1)
    
    comp = status['Slice'][status['status_frame'] == 0.1].values
    compId = np.asarray(np.linspace(0,199,5), dtype = 'int')
    for j in range(len(comp)//200):
        selectedComp = comp[j*200 : (j+1)*200]
        idx = np.asarray(selectedComp[compId], dtype = 'int')
        selectedFrames.extend(idx)
    
    selectedFrames = np.asarray(np.sort(selectedFrames))

    stack = tiff.imread(os.path.join(path, tif))

    new_stack = stack[selectedFrames, :, :]
    times = (field[selectedFrames, 1] - field[0, 1])/1000
    
    activationSlices = np.asarray([np.where(selectedFrames == k)[0] for k in activationFrames])
    activationSlices = activationSlices.flatten()
    
    for z in range(len(new_stack)):
        text = str(dt.timedelta(seconds = times[z]))[2:9]
        cv2.putText(new_stack[z, :, :], text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,0,0),2,cv2.LINE_AA)
        
        if len(activationFrames) != 0 and z in activationSlices:
            cv2.circle(new_stack[z, :, :], (128, 157), 175, (0,0,255), 3)
    
    io.imsave(os.path.join(pathSave, tif), new_stack)
        
        
        
#%% Small code to plot forces

fig, ax = plt.subplots(1)

data = pd.read_csv('C:/Users/anumi/Downloads/22-02-09_M1_P1_C9_L40_20umdisc_PY.csv', sep = ';')

toPlot = data[(data['idxAnalysis'] == -2) | (data['idxAnalysis'] == 2)]

plt.plot(toPlot['T'], toPlot['F'], color = '#cc0000', lw = 4)
ax.yaxis.set_tick_params(labelsize=25)
ax.xaxis.set_tick_params(labelsize=25)
plt.tight_layout()

plt.plot()

#%% Write a txt file for ramps and compressions

def sigmoid(x):
  return 1 / (1 + np.exp(-6*x))

freq = 10000 #in Hz
tConst = 1 # in secs
tComp = 1.5 #in sec
factor = 1000
constfield = 15 # mT 
lowfield = 1
highfield = 50

xRelease = np.linspace(-1, 1, freq*1)
yRelease = (1 - sigmoid(xRelease))*(constfield-lowfield) + lowfield
constRelease = np.asarray(freq*1*[lowfield])

x = np.linspace(lowfield, highfield**(1/4), int(freq*tComp))
constArray = np.asarray(freq*tConst*[constfield])
compArray = x**4
relaxArray = np.flip(compArray)
wholeArray = []

wholeArray.extend(constArray*factor)
wholeArray.extend(yRelease*factor)
wholeArray.extend(constRelease*factor)
wholeArray.extend(compArray*factor)
wholeArray.extend(relaxArray*factor)
wholeArray.extend(constRelease*factor)

wholeArray.extend(constArray*factor)

np.savetxt(rawDir + "/CompressionFieldFiles/10mT_1-50_t4_1.5s.txt", wholeArray, fmt='%i')

plt.plot(wholeArray)
#%% Calculating the forces from the magnetic field and thickness

def computeMag_M450(B):
    M = 1.05*1600 * (0.001991*B**3 + 17.54*B**2 + 153.4*B) / (B**2 + 35.53*B + 158.1)
    return(M)

V = (4/3)*np.pi*((4.5*10**(-6))/2)**3
m = computeMag_M450(30 * 10**(-3))*V
d = 300 * 10**-9
F = (6*(4*np.pi*10**(-7))*m**2)/(4*np.pi*d**4)

f = F * 10**12 # in pN

print(f)
#%%%% Plotting a graph of field vs. distance between beads

V = (4/3)*np.pi*((4.5*10**(-6))/2)**3
dist = 400 * 10**-9
magFields = np.linspace(2, 1000, 60)
m = computeMag_M450(magFields * 10**(-3))*V

F = (6*(4*np.pi*10**(-7))*m**2)/(4*np.pi*dist**4)
f = F * 10**12 # in pN

plt.loglog(magFields, F)
# plt.xlim(0, 10**2)
# plt.ylim(0, 10**(-8))

# plt.plot(magFields, f)
# plt.grid()
# plt.xlabel('Mag. Field (B) [mT]')
# plt.ylabel('Force between a 400nm spacing (pN)')



#%% Create folders for each field of view during Cannonball experiment

filesPath = 'F:/20231006_Cannonball/BSA_Epoxy_2023/2.5V'

allFiles = os.listdir(filesPath)
allFiles = np.asarray([i for i in allFiles if i.endswith('.tif')])

allRegs = np.unique(np.asarray([i.split('_')[0] for i in allFiles]))

for i in allRegs:
    if not os.path.exists(os.path.join(filesPath, i)):
        os.mkdir(os.path.join(filesPath, i))
    selectedFiles = [k for k in allFiles if i in k]
    for j in selectedFiles:
        os.rename(os.path.join(filesPath, j), os.path.join(filesPath + '/' + i, j))