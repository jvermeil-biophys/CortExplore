# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:46:24 2023

@author: JosephVermeil
"""

# %% Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

import os
import re

from scipy.optimize import curve_fit
import statsmodels.api as sm

#### Constants

mu0 = np.pi*4e-7

# R = 4.490/2 # µm
# Volume = (4/3)*np.pi*(R*1e-6)**3 # V volume in m^3
scale = 15.8 * 0.63 # pix/µm # OBJO 63X !
freq = 50 # Hz 
visco = 1e-3 # Pa.s-1 # Water or PBS

#### Dir



# %% Utilities

# %%% 1. Useful function : Create folders for each field of view during Cannonball experiment

filesPath = 'E://23-10-28_Cannonballs//M2_M450-2023_BSA_inPBS//4.5V'

allFiles = os.listdir(filesPath)
allFiles = np.asarray([i for i in allFiles if i.endswith('.tif')])

allRegs = np.unique(np.asarray([i.split('_')[0] for i in allFiles]))

for i in allRegs:
    if not os.path.exists(os.path.join(filesPath, i)):
        os.mkdir(os.path.join(filesPath, i))
    selectedFiles = [k for k in allFiles if i in k]
    for j in selectedFiles:
        os.rename(os.path.join(filesPath, j), os.path.join(filesPath + '/' + i, j))

# %%% 2. Useful function : Add the folder ID to the image file name and move it out of the folder.
# Exemple : './/2.0V//Image2-2.tif' becomes './/2.0V_Image2-2.tif'

mainPath = 'E://23-10-28_Cannonballs//M1_M450-2025_BSA_inPBS//'

allFiles = os.listdir(mainPath)

for f in allFiles:
    secondPath = os.path.join(mainPath, f)
    allFilesIn_f = os.listdir(secondPath)
    for ff in allFilesIn_f:
        # print(os.path.join(mainPath, f + '_' + ff))
        os.rename(os.path.join(secondPath, ff), os.path.join(mainPath, f + '_' + ff))
        
        
# %%% 3. Useful function : Merge the tables made for each voltage

mainPath = 'D://MagneticPincherData//Raw//23.10.28_Cannonballs//M2_M450-2023_BSA_inPBS'
dirName = mainPath.split('//')[-1]

allFiles = os.listdir(mainPath)
list_df = []

commonLabel = 'SpeedAngles'

for f in allFiles:
    if commonLabel in f:
        f_path = os.path.join(mainPath, f)
        f_id = f.split('_')[0][:-1]
        df = pd.read_csv(f_path, sep = None, engine = 'python', usecols = [1, 2])
        df.insert(0, 'V', [float(f_id) for i in range(len(df))])
        list_df.append(df)
        
df_concat = pd.concat(list_df)

savePath = os.path.join(mainPath, dirName + '_' + commonLabel + '.csv')
df_concat.to_csv(savePath, sep=';', index = False)

# %%% Little test

R = 0.085/2
Z = np.linspace(0, 0.10, 100)

def fun(z, R):
    res = R**2 / ((R**2 + z**2)**(3/2))
    return(res)

def der_fun(z, R):
    res = 3*z / ((R**2 + z**2)**(5/2))
    return(res)

B = fun(Z, R)
B = B/fun(0, R)

dB = der_fun(Z, R)
dB = dB/max(dB)

fig, ax = plt.subplots(1,1)
ax.plot(Z*100, B, label = 'Coil diameter = 8.5cm')
ax.plot([], [], 'r-', label = 'abs(derivative)')

ax2 = ax.twinx()
ax2.plot(Z*100, dB, 'r-', label = 'abs(derivative)')

ax.set_xlabel('z (cm)')
ax.set_ylabel('B/B0')
ax2.set_ylabel('dB/dz (a.u.)')
ax.legend()

plt.show()

# %% Merging all october 2023 cannonball measurments

dirMerge = "D://MagneticPincherData//Raw//23.10_CannonBalls_Merging-all-results"
path2023 = os.path.join(dirMerge, "23-10_ALL_M450-2023_Results.csv")
path2025 = os.path.join(dirMerge, "23-10_ALL_M450-2025_Results.csv")
pathStrept = os.path.join(dirMerge, "23-10_ALL_M450-STREPT_Results.csv")

df_CB_2023 = pd.read_csv(path2023)
# group = df_CB_2023.groupby('B')
# df_CB_2023 = group.agg({'M':'mean','date':'first','expt':'first'}).reset_index()


df_CB_2025 = pd.read_csv(path2025)
# group = df_CB_2025.groupby('B')
# df_CB_2025 = group.agg({'M':'mean','date':'first','expt':'first'}).reset_index()

df_CB_Strept = pd.read_csv(pathStrept)
# group = df_CB_Strept.groupby('B')
# df_CB_Strept = group.agg({'M':'mean','date':'first','expt':'first'}).reset_index()


B0 = np.linspace(0, np.max(df_CB_2023.B), 100)
M0 = 1600 * (0.001991*B0**3 + 17.54*B0**2 + 153.4*B0) / (B0**2 + 35.53*B0 + 158.1) 

def B2M(x, k):
    M = 1600 * k * (0.001991*x**3 + 17.54*x**2 + 153.4*x) / (x**2 + 35.53*x + 158.1)
    return(M)

gs.set_manuscript_options_jv()
fig, axes = plt.subplots(1, 3, figsize = (17/gs.cm_in,8/gs.cm_in), sharex = True, sharey = True)
# fig.sharey()

# 2023
ax = axes[0]
ax.grid()
ax.set_xlim([0, 32])
ax.set_ylim([0, 18])
manip = "M450-2023"


[k_exp], covM = curve_fit(B2M, df_CB_2023['B'], df_CB_2023['M'], absolute_sigma=False)
ste = covM[0,0]**0.5
alpha = 0.975
dof = len(df_CB_2023['B'])-1
q = st.t.ppf(alpha, dof) # Student coefficient
# print(ste, q, q*ste)
print(q*ste)

ax.set_title(manip)
ax.plot(B0, M0/1e3, 'r-', label='Litterature formula', zorder = 4)
ax.scatter(df_CB_2023['B'], df_CB_2023['M']/1e3, c=df_CB_2023['expt'], marker = '.', s=40, alpha = 0.4,
           cmap='Set1', label='Measured - Nexp = 3')
ax.plot(B0, B2M(B0, k_exp)/1e3, 'g--', label='Fit: $k_{corrMag}$' + ' = {:.3f} $\pm$ {:.3f}'.format(k_exp, q*ste), zorder = 5)
ax.set_xlabel('B (mT)')
ax.set_ylabel('M (kA/m)')
ax.legend(fontsize=6, loc='lower right')

# 2025
ax = axes[1]
ax.grid()
manip = "M450-2025"

[k_exp], covM = curve_fit(B2M, df_CB_2025['B'], df_CB_2025['M'], absolute_sigma=False)
ste = (covM[0,0]**0.5) * len(df_CB_2025['B'].unique())/len(df_CB_2025['B'].unique())
alpha = 0.975
dof = len(df_CB_2025['B'])-1
q = st.t.ppf(alpha, dof) # Student coefficient
# print(ste, q, q*ste)
print(q*ste)

ax.set_title(manip)
ax.plot(B0, M0/1e3, 'r-', label='Litterature formula', zorder = 4)
ax.scatter(df_CB_2025['B'], df_CB_2025['M']/1e3, c=df_CB_2025['expt'], marker = '.', s=40, alpha = 0.4,
           cmap='Set2', label='Measured - Nexp = 2')
ax.plot(B0, B2M(B0, k_exp)/1e3, 'g--', label='Fit: $k_{corrMag}$' + ' = {:.3f} $\pm$ {:.3f}'.format(k_exp, q*ste), zorder = 5)
ax.set_xlabel('B (mT)')
# ax.set_ylabel('M (A/m)')
ax.legend(fontsize=6, loc='lower right')

# M450-Strept
ax = axes[2]
ax.grid()
manip = "M450-Strept"

[k_exp], covM = curve_fit(B2M, df_CB_Strept['B'], df_CB_Strept['M'], absolute_sigma=False)
ste = (covM[0,0]**0.5) * len(df_CB_Strept['B'].unique())/len(df_CB_Strept['B'].unique())
alpha = 0.975
dof = len(df_CB_2025['B'])-1
q = st.t.ppf(alpha, dof) # Student coefficient
# print(ste, q, q*ste)
print(q*ste)

ax.set_title(manip)
ax.plot(B0, M0/1e3, 'r-', label='Litterature formula', zorder = 4)
ax.scatter(df_CB_Strept['B'], df_CB_Strept['M']/1e3, c=df_CB_Strept['expt'], marker = '.', s=40, alpha = 0.4,
           cmap='Accent', label='Measured - Nexp = 1')
ax.plot(B0, B2M(B0, k_exp)/1e3, 'g--', label='Fit: $k_{corrMag}$' + ' = {:.3f} $\pm$ {:.3f}'.format(k_exp, q*ste), zorder = 5)
ax.set_xlabel('B (mT)')
# ax.set_ylabel('M (A/m)')
ax.legend(fontsize=6, loc='lower right')


plt.tight_layout()
plt.show()


#### Save
figDir = "D:/MagneticPincherData/Figures/PhysicsDataset"
figSubDir = 'Mat&Meth'
name = 'CannonBall_Results'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

# %% TEST OF DIFFERENT FITTING FUNCTIONS

dirMerge = "D://MagneticPincherData//Raw//23.10.28_Cannonballs//AttemptAtMergingResults"
path2023 = os.path.join(dirMerge, "23-10_ALL_M450-2023_Results.csv")
path2025 = os.path.join(dirMerge, "23-10_ALL_M450-2025_Results.csv")

df_CB_2023 = pd.read_csv(path2023)
df_CB_2025 = pd.read_csv(path2025)


B0 = np.linspace(0, np.max(df_CB_2023.B), 100)
M0 = 1600 * (0.001991*B0**3 + 17.54*B0**2 + 153.4*B0) / (B0**2 + 35.53*B0 + 158.1) 

def B2M(x, k):
    M = 1600 * k * (0.001991*x**3 + 17.54*x**2 + 153.4*x) / (x**2 + 35.53*x + 158.1)
    return(M)

def B2M_noFit(x):
    M = 1600 * (0.001991*x**3 + 17.54*x**2 + 153.4*x) / (x**2 + 35.53*x + 158.1)
    return(M)

def linear(x, k):
    y = k * x
    return(y)



fig, axes = plt.subplots(1, 2)

# 2023
ax = axes[0]
manip = "M450-2023"

[k_exp], covM = curve_fit(linear, B2M_noFit(df_CB_2023['B']), df_CB_2023['M'], absolute_sigma=False)
ste = covM[0,0]**0.5
alpha = 0.975
dof = len(df_CB_2023['B'].unique())-1
q = st.t.ppf(alpha, dof) # Student coefficient
# print(ste, q, q*ste)
print(q*ste)

ols_model = sm.OLS(df_CB_2023['M'], B2M_noFit(df_CB_2023['B']))
results_ols = ols_model.fit()
k_exp2 = results_ols.params[0]
ste2 = results_ols.HC3_se[0]
print(q*ste2)

print(k_exp, k_exp2)

M = B2M(B0, k_exp)
ax.set_title(manip)
ax.plot(M0, M0, 'k--', label='Litterature formula - y = x')
ax.plot(M0, M, 'r--', label='Fit:  k = {:.3f}'.format(k_exp))
ax.plot(B2M_noFit(df_CB_2023['B']), df_CB_2023['M'], 'k+', label='Measured')
ax.set_xlabel('M0 (A/m)')
ax.set_ylabel('M (A/m)')
ax.legend()

# 2025
ax = axes[1]
manip = "M450-2025"

[k_exp], covM = curve_fit(linear, B2M_noFit(df_CB_2025['B']), df_CB_2025['M'], absolute_sigma=False)
ste = covM[0,0]**0.5
alpha = 0.975
dof = len(df_CB_2025['B'].unique())-1
q = st.t.ppf(alpha, dof) # Student coefficient
# print(ste, q, q*ste)
print(q*ste)

ols_model = sm.OLS(df_CB_2025['M'], B2M_noFit(df_CB_2025['B']))
results_ols = ols_model.fit()
k_exp2 = results_ols.params[0]
ste2 = results_ols.HC3_se[0]
print(q*ste2)

print(k_exp, k_exp2)

M = B2M(B0, k_exp)
ax.set_title(manip)
ax.plot(M0, M0, 'k--', label='Litterature formula - y = x')
ax.plot(M0, M, 'r--', label='Fit:  k = {:.3f}'.format(k_exp))
ax.plot(B2M_noFit(df_CB_2025['B']), df_CB_2025['M'], 'k+', label='Measured')
ax.set_xlabel('M0 (A/m)')
ax.set_ylabel('M (A/m)')
ax.legend()


plt.tight_layout()
plt.show()





# %% Analysis 23.10.28_Cannonballs

dataDir = "D://MagneticPincherData//Raw//23.10.28_Cannonballs"
df_V2B = pd.read_csv(dataDir + '//' + 'Coil_V2B' + '.csv', sep = ';')
df_B2gradB = pd.read_csv(dataDir + '//' + 'Coil_B2gradB' + '.csv', sep = ';')

# %%% Coil tension-to-field coefficient

PLOT = False

#### V2B

a, b = np.polyfit(df_V2B.V, df_V2B.B, 1)
xx = np.linspace(0, 5, 50)
yy = a*xx + b

#### B2gradB

c, d = np.polyfit(df_B2gradB.X, df_B2gradB.B, 1)
ii = np.linspace(-4, 4, 50)
jj = c*ii + d

print(np.abs(c/d))

#### Plots

if PLOT:
    fig, ax = plt.subplots(1, 1)
    
    df_V2B.plot('V', 'B', fig=fig, ax=ax,
                marker = '.', color = 'k', ls = '',
                label = 'measured')
    
    ax.plot(xx, yy, 'r--', label = 'y = {:.3f} * x + {:.3f}'.format(a, b))
    ax.set_xlabel('V (V)')
    ax.set_ylabel('B (mT)')
    ax.legend()
    
    fig2, ax2 = plt.subplots(1, 1)
    
    df_B2gradB.plot('X', 'B', fig=fig2, ax=ax2,
                marker = '.', color = 'k', ls = '',
                label = 'measured')
    
    ax2.plot(ii, jj, 'r--', label = 'y = {:.3f} * x + {:.3f}'.format(c, d))
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('B (mT)')
    ax2.legend()
    
    plt.tight_layout()
    
    plt.show()


#### Define functions

def V2B(x):
    # Volt to mT
    return(a*x + b)

def B2gB(x):
    # mT to mT/mm or T/m
    return(np.abs(c)*x/np.abs(d))

# %%% An important option : X-Y velocity components separation

useTrajAngle = True

# %%% Reading the data

manip = 'M2_M450-2023_BSA_inPBS'
speed_file = manip + '_SpeedAngles'
traj_file = manip + '_TrajAngles'

R = 4.476/2 # µm
# 4.476 for M450-2023
# 4.492 for M450-2025
# 4.506 for M450-strept
Volume = (4/3)*np.pi*(R*1e-6)**3 # V volume in m^3

df_CB = pd.read_csv(dataDir + '//' + speed_file + '.csv', sep = ';') # '\t'
try:
    df_Traj = pd.read_csv(dataDir + '//' + traj_file + '.csv', sep = ';', usecols = [1]) # '\t'
    df_Traj = df_Traj.rename(columns = {'Angle' : 'TrajAngle'})
    df_CB = df_CB.merge(df_Traj, left_index = True, right_index = True)
except:
    pass

df_CB['B'] = V2B(df_CB.V)
df_CB['gradB'] = B2gB(df_CB.B)

df_CB['dX_pix'] = np.abs(df_CB.Length * np.cos(np.pi*df_CB.Angle/180))
df_CB['dT_pix'] = np.abs(df_CB.Length * np.sin(np.pi*df_CB.Angle/180))

if useTrajAngle:
    df_CB['dX_pix_corr'] = df_CB['dX_pix'] * np.cos(np.pi*df_CB.TrajAngle/180)
    df_CB['dX'] = df_CB['dX_pix_corr'] * (1/scale) # µm
    
else:
    df_CB['dX'] = df_CB['dX_pix'] * (1/scale) # µm
    
df_CB['dT'] = df_CB['dT_pix'] * (1/freq) # sec

df_CB['U'] = (1e-6 * df_CB['dX']) / df_CB['dT'] # m/s

df_CB['Fmes'] = 6 * np.pi * visco * (R*1e-6) * df_CB['U'] # N
df_CB['Fmes'] = df_CB['Fmes'] * 1e12 # pN

df_CB['M'] = df_CB['Fmes'] / (Volume * df_CB['gradB']*1e12)

# fig2, ax2 = plt.subplots(1, 1)

# df_CB.plot('B', 'U', fig=fig2, ax=ax2,
#             marker = '.', color = 'k', ls = '',
#             label = '')

# ax2.set_xlabel('B (mT)')
# ax2.set_ylabel('U (m/s)')
# ax2.legend()

# plt.tight_layout()

# plt.show()

savePath = os.path.join(dataDir, manip + '_Results.csv') 
df_CB.to_csv(savePath, sep=';', index=False)

# %%% Plotting the results and fitting the correction

# df_CB = df_CB[df_CB['V'] != 3.5]

B0 = np.linspace(0, np.max(df_CB.B), 100)
M0 = 1600 * (0.001991*B0**3 + 17.54*B0**2 + 153.4*B0) / (B0**2 + 35.53*B0 + 158.1) 

def B2M(x, k):
    M = 1600 * k * (0.001991*x**3 + 17.54*x**2 + 153.4*x) / (x**2 + 35.53*x + 158.1)
    return(M)
    
[k_exp], covM = curve_fit(B2M, df_CB['B'], df_CB['M'], absolute_sigma=True)
fig, ax = plt.subplots(1, 1)

fig.suptitle(manip)

ax.plot(B0, M0, 'r--', label='Litterature formula')
ax.plot(df_CB['B'], df_CB['M'], 'k+', label='Measured')
ax.plot(B0, B2M(B0, k_exp), 'g--', label='New fit with k = {:.3f}'.format(k_exp))
ax.set_xlabel('B (mT)')
ax.set_ylabel('M (A/m)')
ax.legend()

plt.tight_layout()

plt.show()










# %% Analysis 23.10.18_Cannonballs

dataDir = "D://MagneticPincherData//Raw//23.10.18_Cannonball"
df_V2B = pd.read_csv(dataDir + '//' + 'Coil_V2B' + '.csv', sep = ';')
df_B2gradB = pd.read_csv(dataDir + '//' + 'Coil_B2gradB' + '.csv', sep = ';')

# %%% Coil tension-to-field coefficient

PLOT = False

#### V2B

a, b = np.polyfit(df_V2B.V, df_V2B.B, 1)
xx = np.linspace(0, 5, 50)
yy = a*xx + b

#### B2gradB

c, d = np.polyfit(df_B2gradB.X, df_B2gradB.B, 1)
ii = np.linspace(-4, 4, 50)
jj = c*ii + d

print(np.abs(c/d))

#### Plots

if PLOT:
    fig, ax = plt.subplots(1, 1)
    
    df_V2B.plot('V', 'B', fig=fig, ax=ax,
                marker = '.', color = 'k', ls = '',
                label = 'measured')
    
    ax.plot(xx, yy, 'r--', label = 'y = {:.3f} * x + {:.3f}'.format(a, b))
    ax.set_xlabel('V (V)')
    ax.set_ylabel('B (mT)')
    ax.legend()
    
    fig2, ax2 = plt.subplots(1, 1)
    
    df_B2gradB.plot('X', 'B', fig=fig2, ax=ax2,
                marker = '.', color = 'k', ls = '',
                label = 'measured')
    
    ax2.plot(ii, jj, 'r--', label = 'y = {:.3f} * x + {:.3f}'.format(c, d))
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('B (mT)')
    ax2.legend()
    
    plt.tight_layout()
    
    plt.show()


#### Define functions

def V2B(x):
    # Volt to mT
    return(a*x + b)

def B2gB(x):
    # mT to mT/mm or T/m
    return(np.abs(c)*x/np.abs(d))

# %%% An important option : X-Y velocity components separation

useTrajAngle = True

# %%% Reading the data

manip = 'M5_M450-Strept-inwater'
speed_file = manip + '_SpeedAngles'
traj_file = manip + '_TrajAngles'

R = 4.506/2 # µm
# 4.476 for M450-2023
# 4.492 for M450-2025
# 4.506 for M450-strept
Volume = (4/3)*np.pi*(R*1e-6)**3 # V volume in m^3

df_CB = pd.read_csv(dataDir + '//' + speed_file + '.csv', sep = ';') # '\t'
try:
    df_Traj = pd.read_csv(dataDir + '//' + traj_file + '.csv', sep = ';', usecols = [1]) # '\t'
    df_Traj = df_Traj.rename(columns = {'Angle' : 'TrajAngle'})
    df_CB = df_CB.merge(df_Traj, left_index = True, right_index = True)
except:
    pass

df_CB['B'] = V2B(df_CB.V)
df_CB['gradB'] = B2gB(df_CB.B)

df_CB['dX_pix'] = np.abs(df_CB.Length * np.cos(np.pi*df_CB.Angle/180))
df_CB['dT_pix'] = np.abs(df_CB.Length * np.sin(np.pi*df_CB.Angle/180))

if useTrajAngle:
    df_CB['dX_pix_corr'] = df_CB['dX_pix'] * np.cos(np.pi*df_CB.TrajAngle/180)
    df_CB['dX'] = df_CB['dX_pix_corr'] * (1/scale) # µm
    
else:
    df_CB['dX'] = df_CB['dX_pix'] * (1/scale) # µm
    
df_CB['dT'] = df_CB['dT_pix'] * (1/freq) # sec

df_CB['U'] = (1e-6 * df_CB['dX']) / df_CB['dT'] # m/s

df_CB['Fmes'] = 6 * np.pi * visco * (R*1e-6) * df_CB['U'] # N
df_CB['Fmes'] = df_CB['Fmes'] * 1e12 # pN

df_CB['M'] = df_CB['Fmes'] / (Volume * df_CB['gradB']*1e12)

# fig2, ax2 = plt.subplots(1, 1)

# df_CB.plot('B', 'U', fig=fig2, ax=ax2,
#             marker = '.', color = 'k', ls = '',
#             label = '')

# ax2.set_xlabel('B (mT)')
# ax2.set_ylabel('U (m/s)')
# ax2.legend()

# plt.tight_layout()

# plt.show()

savePath = os.path.join(dataDir, manip + '_Results.csv') 
df_CB.to_csv(savePath, sep=';', index=False)

# %%% Plotting the results and fitting the correction

# df_CB = df_CB[df_CB['V'] != 3.5]

B0 = np.linspace(0, np.max(df_CB.B), 100)
M0 = 1600 * (0.001991*B0**3 + 17.54*B0**2 + 153.4*B0) / (B0**2 + 35.53*B0 + 158.1) 

def B2M(x, k):
    M = 1600 * k * (0.001991*x**3 + 17.54*x**2 + 153.4*x) / (x**2 + 35.53*x + 158.1)
    return(M)
    
[k_exp], covM = curve_fit(B2M, df_CB['B'], df_CB['M'], absolute_sigma=True)
fig, ax = plt.subplots(1, 1)

fig.suptitle(manip)

ax.plot(B0, M0, 'r--', label='Litterature formula')
ax.plot(df_CB['B'], df_CB['M'], 'k+', label='Measured')
ax.plot(B0, B2M(B0, k_exp), 'g--', label='New fit with k = {:.3f}'.format(k_exp))
ax.set_xlabel('B (mT)')
ax.set_ylabel('M (A/m)')
ax.legend()

plt.tight_layout()

plt.show()










# %% Analysis 23.10.06_Cannonball - AJ

dataDir = "D://MagneticPincherData//Raw//23.10.06_Cannonball_AJ"
df_V2B = pd.read_csv(dataDir + '//' + 'Coil_V2B' + '.csv', sep = ';')
df_B2gradB = pd.read_csv(dataDir + '//' + 'Coil_B2gradB' + '.csv', sep = ';')

# %%% Coil tension-to-field coefficient

PLOT = True

#### V2B

a, b = np.polyfit(df_V2B.V, df_V2B.B, 1)
xx = np.linspace(0, 5, 50)
yy = a*xx + b

#### B2gradB

c, d = np.polyfit(df_B2gradB.X, df_B2gradB.B, 1)
ii = np.linspace(-4, 4, 50)
jj = c*ii + d

print(np.abs(c/d))

#### Plots

if PLOT:
    fig, ax = plt.subplots(1, 1)
    
    df_V2B.plot('V', 'B', fig=fig, ax=ax,
                marker = '.', color = 'k', ls = '',
                label = 'measured')
    
    ax.plot(xx, yy, 'r--', label = 'y = {:.3f} * x + {:.3f}'.format(a, b))
    ax.set_xlabel('V (V)')
    ax.set_ylabel('B (mT)')
    ax.legend()
    
    fig2, ax2 = plt.subplots(1, 1)
    
    df_B2gradB.plot('X', 'B', fig=fig2, ax=ax2,
                marker = '.', color = 'k', ls = '',
                label = 'measured')
    
    ax2.plot(ii, jj, 'r--', label = 'y = {:.3f} * x + {:.3f}'.format(c, d))
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('B (mT)')
    ax2.legend()
    
    plt.tight_layout()
    
    plt.show()


#### Define functions

def V2B(x):
    # Volt to mT
    return(a*x + b)

def B2gB(x):
    # mT to mT/mm or T/m
    return(np.abs(c)*x/np.abs(d))

# %%% An important option : X-Y velocity components separation

useTrajAngle = False

# %%% Reading the data

# manip = 'mPEG_BSA_Results'
manip = 'M450-2023_BSA'

speed_file = manip + '_SpeedAngles'
traj_file = manip + '_TrajAngles'
# file = 'M450-2025_BSA_Results'

R = 4.476/2 # µm
Volume = (4/3)*np.pi*(R*1e-6)**3 # V volume in m^3

df_CB = pd.read_csv(dataDir + '//' + speed_file + '.csv', sep = ';') # '\t'
df_CB['B'] = V2B(df_CB.V)
df_CB['gradB'] = B2gB(df_CB.B)

df_CB['dX_pix'] = np.abs(df_CB.Length * np.cos(np.pi*df_CB.Angle/180))
df_CB['dT_pix'] = np.abs(df_CB.Length * np.sin(np.pi*df_CB.Angle/180))

df_CB['dX'] = df_CB['dX_pix'] * (1/scale) # µm
df_CB['dT'] = df_CB['dT_pix'] * (1/freq) # sec

df_CB['U'] = (1e-6 * df_CB['dX']) / df_CB['dT'] # m/s

df_CB['Fmes'] = 6 * np.pi * visco * (R*1e-6) * df_CB['U'] # N
df_CB['Fmes'] = df_CB['Fmes'] * 1e12 # pN

df_CB['M'] = df_CB['Fmes'] / (Volume * df_CB['gradB']*1e12)

# fig2, ax2 = plt.subplots(1, 1)

# df_CB.plot('B', 'U', fig=fig2, ax=ax2,
#             marker = '.', color = 'k', ls = '',
#             label = '')

# ax2.set_xlabel('B (mT)')
# ax2.set_ylabel('U (m/s)')
# ax2.legend()

# plt.tight_layout()

# plt.show()

savePath = os.path.join(dataDir, manip + '_Results.csv') 
df_CB.to_csv(savePath, sep=';', index=False)

# %%% Plotting the results and fitting the correction

# df_CB = df_CB[df_CB['V'] != 3.5]

B0 = np.linspace(0, np.max(df_CB.B), 100)
M0 = 1600 * (0.001991*B0**3 + 17.54*B0**2 + 153.4*B0) / (B0**2 + 35.53*B0 + 158.1) 

def B2M(x, k):
    M = 1600 * k * (0.001991*x**3 + 17.54*x**2 + 153.4*x) / (x**2 + 35.53*x + 158.1)
    return(M)
    
[k_exp], covM = curve_fit(B2M, df_CB['B'], df_CB['M'], absolute_sigma=True)
fig, ax = plt.subplots(1, 1)

fig.suptitle(manip)

ax.plot(B0, M0, 'r--', label='Litterature formula')
ax.plot(df_CB['B'], df_CB['M'], 'k+', label='Measured')
ax.plot(B0, B2M(B0, k_exp), 'g--', label='New fit with k = {:.3f}'.format(k_exp))
ax.set_xlabel('B (mT)')
ax.set_ylabel('M (SI)')
ax.legend()

plt.tight_layout()

plt.show()













# %% Analysis 23.09.12_Cannonball

dataDir = "D://MagneticPincherData//Raw//23.09.12_Cannonball"
df_V2B = pd.read_csv(dataDir + '//' + 'Coil_V2B' + '.csv', sep = ';')
df_B2gradB = pd.read_csv(dataDir + '//' + 'Coil_B2gradB' + '.csv', sep = ';')

# %%% Coil tension-to-field coefficient

PLOT = True

#### V2B

a, b = np.polyfit(df_V2B.V, df_V2B.B, 1)
xx = np.linspace(0, 5, 50)
yy = a*xx + b

#### B2gradB

c, d = np.polyfit(df_B2gradB.X, df_B2gradB.B, 1)
ii = np.linspace(-4, 4, 50)
jj = c*ii + d

print(np.abs(c/d))


#### Plots

if PLOT:
    fig, ax = plt.subplots(1, 1)
    
    df_V2B.plot('V', 'B', fig=fig, ax=ax,
                marker = '.', color = 'k', ls = '',
                label = 'measured')
    
    ax.plot(xx, yy, 'r--', label = 'y = {:.3f} * x + {:.3f}'.format(a, b))
    ax.set_xlabel('V (V)')
    ax.set_ylabel('B (mT)')
    ax.legend()
    
    fig2, ax2 = plt.subplots(1, 1)
    
    df_B2gradB.plot('X', 'B', fig=fig2, ax=ax2,
                marker = '.', color = 'k', ls = '',
                label = 'measured')
    
    ax2.plot(ii, jj, 'r--', label = 'y = {:.3f} * x + {:.3f}'.format(c, d))
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('B (mT)')
    ax2.legend()
    
    plt.tight_layout()
    
    plt.show()

#### Define functions

def V2B(x):
    # Volt to mT
    return(a*x + b)

def B2gB(x):
    # mT to mT/mm or T/m
    return(np.abs(c)*x/np.abs(d))

# %%% An important option : X-Y velocity components separation

useTrajAngle = False

# %%% Reading the data

manip = 'mPEG_BSA'
# manip = 'M450-2025_BSA'

speed_file = manip + '_SpeedAngles'
traj_file = manip + '_TrajAngles'
# file = 'M450-2025_BSA_Results'

R = 4.492/2 # µm
Volume = (4/3)*np.pi*(R*1e-6)**3 # V volume in m^3

df_CB = pd.read_csv(dataDir + '//' + speed_file + '.csv', sep = ';') # '\t'
df_CB['B'] = V2B(df_CB.V)
df_CB['gradB'] = B2gB(df_CB.B)

df_CB['dX_pix'] = np.abs(df_CB.Length * np.cos(np.pi*df_CB.Angle/180))
df_CB['dT_pix'] = np.abs(df_CB.Length * np.sin(np.pi*df_CB.Angle/180))

df_CB['dX'] = df_CB['dX_pix'] * (1/scale) # µm
df_CB['dT'] = df_CB['dT_pix'] * (1/freq) # sec

df_CB['U'] = (1e-6 * df_CB['dX']) / df_CB['dT'] # m/s

df_CB['Fmes'] = 6 * np.pi * visco * (R*1e-6) * df_CB['U'] # N
df_CB['Fmes'] = df_CB['Fmes'] * 1e12 # pN

df_CB['M'] = df_CB['Fmes'] / (Volume * df_CB['gradB']*1e12)

# fig2, ax2 = plt.subplots(1, 1)

# df_CB.plot('B', 'U', fig=fig2, ax=ax2,
#             marker = '.', color = 'k', ls = '',
#             label = '')

# ax2.set_xlabel('B (mT)')
# ax2.set_ylabel('U (m/s)')
# ax2.legend()

# plt.tight_layout()

# plt.show()

savePath = os.path.join(dataDir, manip + '_Results.csv') 
df_CB.to_csv(savePath, sep=';', index=False)

# %%% Plotting the results and fitting the correction

# df_CB = df_CB[df_CB['V'] != 3.5]

B0 = np.linspace(0, np.max(df_CB.B), 100)
M0 = 1600 * (0.001991*B0**3 + 17.54*B0**2 + 153.4*B0) / (B0**2 + 35.53*B0 + 158.1) 

def B2M(x, k):
    M = 1600 * k * (0.001991*x**3 + 17.54*x**2 + 153.4*x) / (x**2 + 35.53*x + 158.1)
    return(M)
    
[k_exp], covM = curve_fit(B2M, df_CB['B'], df_CB['M'], absolute_sigma=True)
fig, ax = plt.subplots(1, 1)

fig.suptitle(manip)

ax.plot(B0, M0, 'r--', label='Litterature formula')
ax.plot(df_CB['B'], df_CB['M'], 'k+', label='Measured')
ax.plot(B0, B2M(B0, k_exp), 'g--', label='New fit with k = {:.3f}'.format(k_exp))
ax.set_xlabel('B (mT)')
ax.set_ylabel('M (SI)')
ax.legend()

plt.tight_layout()

plt.show()












# %% Reanalyse Valentin M270

mu0 = np.pi*4e-7

R = 2.690/2 # µm
Volume = (4/3)*np.pi*(R*1e-6)**3 # V volume in m^3
scale = 15.8 * 0.63 # pix/µm
freq = 50 # Hz 
visco = 1e-3 # Pa.s-1

# %%% Reading the data

dataDir = "D://MagneticPincherData//Raw//20.08.01_CanonBall_M270_VL"

# file = 'mPEG_BSA_All-V_Results'
file = 'All_24-08_KimoMin'

df_CB = pd.read_csv(dataDir + '//' + file + '.txt', sep = '\t') # '\t'
# df_CB['B'] = V2B(df_CB.V)
df_CB['gradB (mT/mm)'] = B2gB(df_CB['B (mT)'])

# df_CB['dX_pix'] = np.abs(df_CB.Length * np.cos(np.pi*df_CB.Angle/180))
# df_CB['dT_pix'] = np.abs(df_CB.Length * np.sin(np.pi*df_CB.Angle/180))

# df_CB['dX'] = df_CB['dX_pix'] * (1/scale) # µm
# df_CB['dT'] = df_CB['dT_pix'] * (1/freq) # sec

df_CB['U'] = (1e-6 * df_CB['X (µm)']) / df_CB['T (s)'] # m/s

df_CB['Fmes'] = 6 * np.pi * visco * (R*1e-6) * df_CB['U'] # N
df_CB['Fmes'] = df_CB['Fmes'] * 1e12 # pN

df_CB['M'] = df_CB['Fmes'] / (Volume * df_CB['gradB (mT/mm)']*1e12)



fig2, ax2 = plt.subplots(1, 1)

df_CB.plot('B (mT)', 'U', fig=fig2, ax=ax2,
            marker = '.', color = 'k', ls = '',
            label = '')

ax2.set_xlabel('B (mT)')
ax2.set_ylabel('U (m/s)')
ax2.legend()

plt.tight_layout()

plt.show()

# %%%

# df_CB = df_CB[df_CB['V'] != 3.5]

B0 = np.linspace(0, np.max(df_CB['B (mT)']), 100)
M0 = 1600 * (0.001991*B0**3 + 17.54*B0**2 + 153.4*B0) / (B0**2 + 35.53*B0 + 158.1) 

def B2M(x, k):
    M = 1600 * k * (0.001991*x**3 + 17.54*x**2 + 153.4*x) / (x**2 + 35.53*x + 158.1)
    return(M)
    
[k_exp], covM = curve_fit(B2M, df_CB['B (mT)'], df_CB['M'], absolute_sigma=True)
fig, ax = plt.subplots(1, 1)

fig.suptitle(file)

ax.plot(B0, M0, 'r--', label='Litterature formula')
ax.plot(df_CB['B (mT)'], df_CB['M'], 'k+', label='Measured for M270 - Valentin')
ax.plot(B0, B2M(B0, k_exp), 'g--', label='New fit with k = {:.3f}'.format(k_exp))
ax.set_xlabel('B (mT)')
ax.set_ylabel('M (SI)')
ax.legend()

plt.tight_layout()

plt.show()

