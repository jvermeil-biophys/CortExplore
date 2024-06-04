# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:18:44 2024

@author: anumi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:59:54 2024

@author: anumi
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import UtilityFunctions as ufun
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu


#%%

# path = 'D:/Anumita/MagneticPincherData/Collabs-Internships/Hugo/TestCurve'
path = 'D:/Anumita/MagneticPincherData/Data_TimeSeries'
savePath = 'D:/Anumita/MagneticPincherData/Figures/NLI_Analysis/24-05-20_Replotted'

files = np.asarray(os.listdir(path))

filename = ['23-07-12', '23-05-23']
folder = '3T3OptoRhoA_AtBeads_60sFreq'

# filename = ['22-12-07_M4', '22-12-07_M5', '22-12-07_M8']
# folder = '3T3OptoRhoA_Y27_50uM'

filteredFiles = []
            
for j in filename:
    [filteredFiles.append((i)) for i in files if j in i]


folderPath = os.path.join(savePath, folder)

if not os.path.exists(folderPath):
    os.mkdir(os.path.join(folderPath))

if not os.path.exists(os.path.join(folderPath, 'Compressions')):
    os.mkdir(os.path.join(folderPath, 'Compressions'))
    
if not os.path.exists(os.path.join(folderPath, 'Cell-Based Plots')):
    os.mkdir(os.path.join(folderPath, 'Cell-Based Plots'))

# files = [i for i in files if 'test' in i]


#%% Functions

def getMaskForCompression(rawDf, i, task = 'compression'):
    try:
        Npts = len(rawDf['idxAnalysis'].values)
        iStart = ufun.findFirst(np.abs(rawDf['idxAnalysis']), i+1)
        iStop = iStart + np.sum((np.abs(rawDf['idxAnalysis']) == i+1))
        
        if task == 'compression':
            mask = (rawDf['idxAnalysis'] == i+1).values
        elif task == 'precompression':
            mask = (rawDf['idxAnalysis'] == -(i+1)).values
        elif task == 'compression & precompression':
            mask = (np.abs(rawDf['idxAnalysis']) == i+1).values
    
    except:
        Npts = len(rawDf['idxAnalysis'].values)
        iStart = ufun.findFirst(np.abs(rawDf['idxAnalysis']), i+1)
        iStop = iStart + np.sum((np.abs(rawDf['idxAnalysis']) == i+1))
        
        if task == 'compression':
            mask = (rawDf['idxAnalysis'] == i+1).values
        elif task == 'precompression':
            mask = (rawDf['idxAnalysis'] == -(i+1)).values
        elif task == 'compression & precompression':
            mask = (np.abs(rawDf['idxAnalysis']) == i+1).values

            
    return(mask)
            
def correctJumpForCompression(rawDf, listJumpsD3, i):
    colToCorrect = ['dx', 'dy', 'dz', 'D2', 'D3']
    print(i)
    mask = getMaskForCompression(rawDf, i, task = 'compression & precompression')
    iStart = ufun.findFirst(np.abs(rawDf['idxAnalysis']), i+1)
    for c in colToCorrect:
        #### CHANGE HERE: iStart+5 -> iStart+15
        jump = np.median(rawDf[c].values[iStart:iStart+5]) - np.median(rawDf[c].values[iStart-2:iStart])
        rawDf.loc[mask, c] -= jump
        if c == 'D3':
            D3corrected = True
            jumpD3 = jump
    listJumpsD3[i] = jumpD3
    return rawDf, listJumpsD3
    
def correctAllJumpsByLoop(rawDf, listJumpsD3):
    colToCorrect = ['dx', 'dy', 'dz', 'D2', 'D3']
    N = np.max(rawDf['idxLoop'])
    for i in range(1, N+1):
        idx_loop = rawDf[rawDf['idxLoop'] == i].index
        df_loop = rawDf.loc[idx_loop]
        
        idx_action = df_loop[df_loop['idxAnalysis'] != 0].index
        iStart = idx_action[0]
        
        # print(df_loop)
        # print(idx_loop)
        # print(idx_action)
        # print(iStart)
        
        for c in colToCorrect:
            jump = np.median(df_loop.loc[iStart:iStart+5, c].values) \
                 - np.median(df_loop.loc[iStart-2:iStart, c].values)
            # print(jump)     
            df_loop.loc[idx_action, c] -= jump
            if c == 'D3':
                D3corrected = True
                jumpD3 = jump
                
                
        rawDf.loc[idx_loop] = df_loop
        listJumpsD3[i-1] = jumpD3
    return rawDf, listJumpsD3
        
def refineStartStop(rawDf):
    """
    

    Returns
    -------
    None.

    """
    listB = rawDf['B'].values
    NbPtsRaw = len(listB)
    maxCompField = 70
    
    # Correct for bugs in the B data
    for k in range(1,len(listB)):
        B = listB[k]
    
        if B > 1.25*maxCompField:
            listB[k] = listB[k-1]

    offsetStart, offsetStop = 0, 0
    minB, maxB = min(listB), max(listB)
    thresholdB = 0.5
    # thresholdDeltaB = (maxB-minB)/400 # NEW CONDITION for the beginning of the compression : 
    # remove the first points where the steps in B are very very small
    
    k = 0
    while (listB[k] < minB+thresholdB):
        offsetStart += 1
        k += 1
    
    # k = 0
    # while (listB[k] < minB+thresholdB) or (listB[k+1]-listB[k] < thresholdDeltaB):
    #     offsetStart += int((listB[k] < minB+thresholdB) or ((listB[k+1]-listB[k]) < thresholdDeltaB))
    #     k += 1

    # k = 0
    # while (listB[-1-k] < minB+thresholdB):
    #     offsetStop += int(listB[-1-k] < minB+thresholdB)
    #     k += 1

    jMax = np.argmax(listB) # End of compression, beginning of relaxation

    hCompr_raw = (rawDf['D3'].values[:jMax+1] - 4.5)

    # Refinement of the compression delimitation.
    # Remove the 1-2 points at the begining where there is just the viscous relaxation of the cortex
    # because of the initial decrease of B and the cortex thickness increases.
    
    NptsCompr = len(hCompr_raw)
    k = offsetStart
    while (k<(NptsCompr//2)) and (hCompr_raw[k] < np.max(hCompr_raw[k+1:min(k+10, NptsCompr)])):
        k += 1
    offsetStart = k
    
    jStart = offsetStart # Beginning of compression
    jStop = rawDf.shape[0] - offsetStop # End of relaxation
    
    # Better compressions arrays
    hCompr = (rawDf.D3.values[jStart:jMax+1] - 4.5)
    hRelax = (rawDf.D3.values[jMax+1:jStop] - 4.5)
    fCompr = (rawDf.F.values[jStart:jMax+1])
    fRelax = (rawDf.F.values[jMax+1:jStop])
    BCompr = (rawDf.B.values[jStart:jMax+1])
    BRelax = (rawDf.B.values[jMax+1:jStop])
    TCompr = (rawDf['T'].values[jStart:jMax+1])
    TRelax = (rawDf['T'].values[jMax+1:jStop])
    
    jMax = jMax
    jStart = jStart
    jStop = jStop

    mask = np.array([((j >= jStart) and (j < jStop)) for j in range(NbPtsRaw)])

    # self.Df = self.rawDf.loc[mask]
    # self.isRefined = True
    return fCompr, hCompr

def fitCvw(x, K, Y, H0):
    # f = (np.pi*2250*(K/2*(H0**3*x**-2+2*x-3*H0)+Y/3*H0*(1-x/H0)**2))
    f = 3.1416*2250*(K/6*(H0**3*x**-2+2*x-3*H0)+Y/3*H0*(1-x/H0)**2) #including factor 3 (K/6)
    return f


#%%

dictToPlot = {'cellName': [], 'manipID' : [], 'cellID' : [], 'cellCode':[], 'compNo':[], 'nli' : [], 
              'nli_ind':[], 'manip':[],  'nli_plot': [], 'R2' : [], 
              'substrate' : [], 'k' : [], 'y':[], 'h0':[], 'e':[], 'maxH' : [], 'minH':[]}

for file in (filteredFiles):
    fh = pd.read_csv(os.path.join(path, file), sep = ';')
    
    nLoop = fh['idxAnalysis'].max()
    
    if nLoop > 3:
        rows = int(np.sqrt(nLoop))
        cols = int(np.ceil(nLoop / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize = (15,10))
    
        _axes = []
        for ax_array in axes:
            for ax in ax_array:
                _axes.append(ax)
    else:
        continue
    

    cellName = ufun.findInfosInFileName(file, 'cellName')
    manipID = ufun.findInfosInFileName(file, 'manipID')
    cellID = ufun.findInfosInFileName(file, 'cellID')
    cellCode = ufun.findInfosInFileName(file, 'date') +'_'+ cellName.split('_')[1] + '_' + cellName.split('_')[2]
    
    print(cellID)
    
    listJumpsD3 = np.zeros(nLoop, dtype = float)
    # fh, listJumpsD3 = correctAllJumpsByLoop(fh, listJumpsD3)

    # dictToSave = {'idxAnalysis' : [], 'f' : [], 'x' : []}
    
    for i in range(nLoop):
        if nLoop > 1:
            ax = _axes[i]
        else:
            ax = ax
        
        fh, listJumpsD3 = correctJumpForCompression(rawDf = fh, listJumpsD3 = listJumpsD3, i = i)
                
        compression = fh[fh['idxAnalysis'] == i+1]

        manip = file.split('_')[1]
        
        if manip == 'M3':
            substrate = 'Control'
        elif manip == 'M1': 
            substrate = 'Y27 1'
        elif manip == 'M2': 
            substrate = 'Y27 2'
        
        # if manip == 'M3':
        #     substrate = 'Control'
        # elif manip == 'M1': 
        #     substrate = 'Doxy'
        # elif manip == 'M2': 
        #     substrate = 'Doxy + Global Activation'
        
            
        jMax = np.argmax(compression['B'])
        
        f, x = compression['F'].values[:jMax+1], compression['D3'].values[:jMax+1]*1000 - 4500
        # f, x = refineStartStop(compression)
        # x = x*1000
        
        
        initH0 = x[0] + 10
        initK = 0.2*1e-3
        initY = 1*1e-3 
        initialParameters = [initK, initY, initH0]
        
        maxF = np.max(f)
        minF = np.min(f)
        maxH = np.max(x)
        minH = np.min(x)
        
        # dictToSave['idxAnalysis'].extend(np.asarray([i+1]*len(f)).astype(int))
        # dictToSave['f'].extend(f.T.astype(float))
        # dictToSave['x'].extend(x.T)
        
        try:
            params, covM = curve_fit(fitCvw, x, f, p0 = initialParameters, bounds=((0, 0, x[0]), (1e3, 1e6, 2*x[0])), method = 'trf')
            
            k = params[0]*1e6
            y = params[1]*1e6
            h0 = params[2]
            e_eff = y + (0.8)**-4 * k
            nli = np.log10((0.8)**-4 * k / y)
                        
            x_fit = np.linspace(np.min(x), np.max(x), len(compression))
            f_fit = fitCvw(x_fit, params[0], params[1], h0)
            k_fit = fitCvw(x_fit, params[0], 0, h0)
            y_fit = fitCvw(x_fit, 0, params[1], h0)
            
            r2 = ufun.get_R2(np.sort(f), (fitCvw(x, params[0], params[1], h0)))
            
            ax.plot(x, f, color = 'k')
            ax.plot(x_fit, k_fit, color = 'blue', linestyle = '--', label = 'Van Wyk\nK = {:.3e}Pa'.format(k))
            ax.plot(x_fit, y_fit, color = 'green', linestyle = '--', label = 'Hook\nY = {:.3e}Pa'.format(y))
            ax.plot(x_fit, f_fit, color = 'red', linestyle = '--', 
                    label = 'HVW\nE = {:.3e}Pa\nH0 = {:.1f}nm\nR2 = {:.2f}\nH_1 = {:.1f}'.format(e_eff, h0, r2, x[0]))
            ax.set_ylim(minF-100, maxF + 100)
            
            ax.legend()
            
            if nli > 0.3:
                nli_plot = 'non-linear'
                nli_ind = 0
            elif nli < -0.3:
                nli_plot = 'linear'
                nli_ind = 1
            elif nli > -0.3 and nli < 0.3:
                nli_plot = 'intermediate'
                nli_ind = 0.5
                
            dictToPlot['cellName'].append(cellName)
            dictToPlot['maxH'].append(maxH)
            dictToPlot['minH'].append(minH)
            dictToPlot['manipID'].append(manipID)
            dictToPlot['cellID'].append(cellID)
            dictToPlot['cellCode'].append(cellCode)
            dictToPlot['nli_ind'].append(nli_ind)
            dictToPlot['compNo'].append(i+1)
            dictToPlot['k'].append(k)
            dictToPlot['y'].append(y)
            dictToPlot['e'].append(e_eff)
            dictToPlot['h0'].append(h0)
            dictToPlot['nli'].append(nli)
            dictToPlot['substrate'].append(substrate)
            dictToPlot['manip'].append(manip)
            dictToPlot['nli_plot'].append(nli_plot)
            dictToPlot['R2'].append(r2)
        except:
            dictToPlot['cellName'].append(cellName)
            dictToPlot['manipID'].append(manipID)
            dictToPlot['maxH'].append(maxH)
            dictToPlot['minH'].append(minH)
            dictToPlot['cellCode'].append(cellCode)
            dictToPlot['cellID'].append(cellID)
            dictToPlot['nli_ind'].append(np.nan)
            dictToPlot['compNo'].append(i+1)
            dictToPlot['k'].append(np.nan)
            dictToPlot['y'].append(np.nan)
            dictToPlot['e'].append(np.nan)
            dictToPlot['h0'].append(np.nan)
            dictToPlot['nli'].append(np.nan)
            dictToPlot['substrate'].append(np.nan)
            dictToPlot['manip'].append(manip)
            dictToPlot['nli_plot'].append(np.nan)
            dictToPlot['R2'].append(np.nan)
            


    fig.suptitle(cellName) 
    # dfToSave = pd.DataFrame(dictToSave)
    # dfToSave.to_csv('D:/Anumita/MagneticPincherData/Collabs-Internships/Hugo/JumpCorrect_RefineStart/CSV_fh/'+file+'.csv',
    #                 sep = ';', index = False)
    plt.show()
    # plt.savefig('D:/Anumita/MagneticPincherData/Figures/NLI_Analysis/24-05-20_Replotted/'+folder+'/Compressions/'+file+'.png')
    
    plt.close()

#%% Testing Julien's dataset

path2 = 'D:/Anumita/MagneticPincherData/Collabs-Internships/Hugo/TestCurve'
files2 = os.listdir(path2)

fh2 = pd.read_csv(os.path.join(path2, files2[0]), sep = ',')

fig, axes = plt.subplots(1, 1, figsize = (15,10))

f2, x2 = fh2.iloc[:, 1], fh2.iloc[:, 0]

params2, covM = curve_fit(fitCvw, x2, f2, p0 = initialParameters, bounds=((0, 0, x[0]), (1e3, 1e6, 2*x[0])), method = 'trf')

k = params2[0]*1e6
y = params2[1]*1e6
h0 = params2[2]
e_eff = y + 3*(0.8)**(-4) * k
nli = np.log10(3*(0.8)**(-4)*k / y)

x_fit = np.linspace(np.min(x2), np.max(x2), len(f2))
f_fit = fitCvw(x_fit, params2[0], params2[1], h0)
k_fit = fitCvw(x_fit, params2[0], 0, h0)
y_fit = fitCvw(x_fit, 0, params2[1], h0)
    
plt.plot(x2, f2, color = 'k')
plt.plot(x_fit, k_fit, color = 'blue', linestyle = '--', label = 'Van Wyk\nK = {:.3e}Pa'.format(k))
plt.plot(x_fit, y_fit, color = 'green', linestyle = '--', label = 'Hook\nY = {:.3e}Pa'.format(y))
plt.plot(x_fit, f_fit, color = 'red', linestyle = '--', 
         label = 'HVW\nE = {:.3e}Pa\nH0 = {:.1f}nm\nR2 = {:.2f}'.format(e_eff, h0, r2))
plt.legend()
plt.show()

#%%%%
fig, axes = plt.subplots(1, 1, figsize = (15,10))

plt.plot(x, f, color = 'red', label = 'AJ')
plt.plot(x2, f2, color = 'green', label = 'JV')
plt.legend(fontsize = 15)
plt.show()


fig, axes = plt.subplots(1, 1, figsize = (15,10))
x3, f3 = x[len(x)-len(x2):], f[len(f)-len(f2):]
plt.plot(x3, f3, color = 'red', label = 'AJ')
plt.plot(x2, f2, color = 'green', label = 'JV')
plt.legend(fontsize = 15)
plt.show()

fig, axes = plt.subplots(1, 1, figsize = (15,10))
plt.plot(x3-x2)
plt.ylim(2.5, 4)
plt.show()

#%% Plotting
#%%%% Barplots
dfToPlot = pd.DataFrame(dictToPlot)
dfToPlot = dfToPlot[(dfToPlot['R2'] > 0.9)] # & (dfToPlot['h0'] < 1500)]

# plt.style.use('seaborn')
# substrate = ['epithelia', '20um discs', 'glass']

# manips = ['M1', 'M2', 'M3']
# substrate = ['control', 'activation']
# substrate = ['3T3 WT', '3T3 OptoRhoA']
# substrate = ['Y27 10uM', 'Control']
# substrate = ['Y27 50uM', 'Control']
# substrate = ['Blebbi 10uM', 'Control']
# substrate = ['Doxy + Global Activation', 'Doxy', 'Control']
# substrate = ['Blebbi', 'Control']
substrate = ['Y27 1', 'Y27 2', 'Control']


plt.style.use('dark_background')

fig, ax = plt.subplots(figsize = (15,10))
fig.patch.set_facecolor('black')
linear = []
nonlinear = []
intermediate = []
N = []
frac = []

for i in substrate:
# for i in manips:
    frac = dfToPlot[dfToPlot['substrate'] == i]
    # frac = dfToPlot[dfToPlot['manip'] == i]
    sLinear = np.sum(frac['nli_plot']=='linear')
    sNonlin = np.sum(frac['nli_plot']=='non-linear')
    sInter = np.sum(frac['nli_plot']=='intermediate')
    linear.append(sLinear)
    nonlinear.append(sNonlin)
    intermediate.append(sInter)
    N.append(sLinear + sNonlin + sInter)

a1 = dfToPlot['nli'][dfToPlot['substrate'] == substrate[0]].values
b1 = dfToPlot['nli'][dfToPlot['substrate'] == substrate[1]].values
U1, p = mannwhitneyu(a1, b1)

N = np.asarray(N)
linear = (np.asarray(linear)/N)*100
intermediate = (np.asarray(intermediate)/N)*100
nonlinear = (np.asarray(nonlinear)/N)*100

plt.bar(substrate, linear, label='linear', color = '#1c7cbb')
plt.bar(substrate, intermediate, bottom = linear, label='intermediate', color = '#f08e3b')
plt.bar(substrate, nonlinear, bottom = linear+intermediate, label='nonlinear',
        color = '#fbb809')

# plt.bar(manips, linear, label='linear', color = '#1c7cbb')
# plt.bar(manips, intermediate, bottom = linear, label='intermediate', color = '#f08e3b')
# plt.bar(manips, nonlinear, bottom = linear+intermediate, label='nonlinear',
#         color = '#fbb809')

y1 = linear
y2 = intermediate
y3 = nonlinear
fontColour2 = '#000000'
fontColour = '#ffffff'

manips = substrate
for xpos, ypos, yval in zip(manips, y1/2, y1):
    plt.text(xpos, ypos, "%.1f"%yval + '%', ha="center", va="center", fontsize = 25, color = fontColour2)
for xpos, ypos, yval in zip(manips, y1+y2/2, y2):
    plt.text(xpos, ypos, "%.1f"%yval+ '%', ha="center", va="center",fontsize = 25, color = fontColour2)
for xpos, ypos, yval in zip(manips, y1+y2+y3/2, y3):
    plt.text(xpos, ypos, "%.1f"%yval+ '%', ha="center", va="center", fontsize = 25, color = fontColour2)
# add text annotation corresponding to the "total" value of each bar
for xpos, ypos, yval in zip(manips, y1+y2+y3+0.5, N):
    plt.text(xpos, ypos, "N=%d"%yval, ha="center", va="bottom", fontsize = 20)

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.title('Test : Mann-Whitney | p-val = {:.4f}'.format(p), color = fontColour, fontsize = 20)
plt.xticks(fontsize=30, color = fontColour)
plt.yticks(fontsize=30, color = fontColour)
plt.legend(bbox_to_anchor=(1.01,0.5), loc='center left', fontsize = 20, labelcolor='linecolor')
plt.tight_layout()
plt.show()
plt.savefig('D:/Anumita/MagneticPincherData/Figures/NLI_Analysis/24-05-20_Replotted/'+folder+'/NLI_'+str(filename)+'.png')

#%%%% E vs BestH0


# dfToPlot = dfToPlot[(dfToPlot['manip'] == 'M3') | (dfToPlot['manip'] == 'M1')]

# dfToPlot['h0'][dfToPlot['substrate'] == 'epithelia'] = dfToPlot['h0'][dfToPlot['substrate'] == 'epithelia'].values / 2

pal = sns.color_palette("husl", 8)
fig = plt.figure(figsize = (15,10))
fig.patch.set_facecolor('black')


sns.scatterplot(dfToPlot, x = 'h0', y = 'e', hue = 'substrate', palette = pal)
plt.yscale('log')
plt.xscale('log')
plt.ylim(0, 10**6)
plt.xlim(0, 2*10**3)
plt.ylabel('E_effective (Pa)', fontsize=30, color = fontColour)
plt.xlabel('BestH0 (nm)', fontsize=30, color = fontColour)
plt.xticks(fontsize=25, color = fontColour)
plt.legend(fontsize = 20)
plt.yticks(fontsize=25, color = fontColour)
plt.tight_layout()
plt.show()
plt.savefig('D:/Anumita/MagneticPincherData/Figures/NLI_Analysis/24-05-20_Replotted/'+folder+'/EvsH0_(1)_'+str(filename)+'.png')

#%%%% E vs BestH0
fig = plt.figure(figsize = (15,10))
fig.patch.set_facecolor('black')

df = dfToPlot[dfToPlot['nli_plot'] != 'intermediate']
# sns.scatterplot(df, x = 'h0', y = 'e', hue = 'nli_plot', palette  = pal, style = 'substrate')
sns.scatterplot(df, x = 'h0', y = 'e', hue = 'substrate', palette  = pal)

idx = 0

mechanicsType = ['non-linear', 'linear']
for k in substrate:
# for k in mechanicsType:
    eqnText = ''
    # 
    toPlot = df[(df['substrate'] == k)]
    # toPlot = df[(df['nli_plot'] == k)]
    x, y = toPlot['h0'].values, toPlot['e'].values
    
    
    params, results = ufun.fitLineHuber((np.log(x)), np.log(y))
    k = np.exp(params[0])
    a = params[1]
    
    fit_x = np.linspace(np.min(x), np.max(x), 50)
    fit_y = k * fit_x**a
    
    # R2 = results.rsquared
    pval = results.pvalues[1] # pvalue on the param 'a'
    eqnText += " Y = {:.1e} * X^{:.1f}".format(k, a)
    eqnText += " p-val = {:.3f}".format(pval)
    # eqnText += " ; R2 = {:.2f}".format(R2)
    print("Y = {:.4e} * X^{:.4f}".format(k, a))
    print("p-value on the 'a' coefficient: {:.4e}".format(pval))
    # print("R2 of the fit: {:.4f}".format(R2))
    
    plt.plot(fit_x, fit_y, color = pal[idx], label = eqnText, linestyle = '--')
    
    idx = idx + 1

plt.yscale('log')
plt.xscale('log')
plt.ylim(0, 10**6)
plt.xlim(0, 2*10**3)
plt.ylabel('E_effective (Pa)', fontsize=30, color = fontColour)
plt.xlabel('BestH0 (nm)', fontsize=30, color = fontColour)
plt.xticks(fontsize=25, color = fontColour)
plt.legend(fontsize = 20)
plt.yticks(fontsize=25, color = fontColour)
plt.tight_layout()
plt.show()
plt.savefig('D:/Anumita/MagneticPincherData/Figures/NLI_Analysis/24-05-20_Replotted/'+folder+'/EvsH0_(2)_'+str(filename)+'.png')



#%%%% BestH0 vs manip

plt.figure(figsize=(12,10))
N = len(dfToPlot['cellName'].unique())

sns.boxplot(x = 'substrate', y = 'h0', data=dfToPlot, order = manips, color = 'grey',
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.9})
    
sns.swarmplot(x = 'substrate', y = 'h0', data=dfToPlot,linewidth = 1, order = manips,
              edgecolor='k', size = 7, hue = 'cellID', palette = sns.color_palette("Paired", (N)))


plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# plt.ylim(0, 2000)
plt.legend(loc=2, prop={'size': 7}, ncol = 9)
plt.show()
plt.savefig('D:/Anumita/MagneticPincherData/Figures/NLI_Analysis/24-05-20_Replotted/'+folder+'/BestH0_'+str(filename)+'.png')


#%% NLI vs Comp No.

# dfToPlot = dfToPlot[(dfToPlot['manip'] == 'M3') | (dfToPlot['manip'] == 'M1')]

# dfToPlot['h0'][dfToPlot['substrate'] == 'epithelia'] = dfToPlot['h0'][dfToPlot['substrate'] == 'epithelia'].values / 2

pal = sns.color_palette("husl", 8)
plt.figure(figsize = (15,10))
sns.lineplot(dfToPlot, x = 'compNo', y = 'nli_ind', hue = 'substrate', palette = pal)
plt.ylabel('NLI')
plt.xlabel('Compression No.')
plt.xticks(fontsize=25)
plt.legend(fontsize = 18)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.show()
plt.savefig('D:/Anumita/MagneticPincherData/Figures/NLI_Analysis/24-05-20_Replotted/'+folder+'/NLIvsComp_'+str(filename)+'.png')


#%%%% Cell-wise NLI comparison

fig = plt.figure(figsize=(12,10))
fig.patch.set_facecolor('black')

param = 'e'

x, y = ('substrate', 'first'), (param, 'mean')
statsDf = dfToPlot
statsDf['statsCellCode'] = [np.nan]*len(dfToPlot)
statsDf['statsCellCode'][statsDf['compNo'] == 1] = dfToPlot['cellCode'][statsDf['compNo'] == 1]

pairs = statsDf['statsCellCode'].value_counts()
pairs = pairs.index[pairs == 2].tolist()

statsDf = statsDf[statsDf['cellCode'].apply(lambda x : x in pairs)]

group_by_cell = statsDf.groupby(['cellID'])
avgDf = group_by_cell.agg({param:['var', 'std', 'mean', 'count'], 'cellName':'first', 'nli_plot' : 'first',
                            'cellCode':['first'], 'manip':'first', 'substrate':'first'})


#For statistics:
a = avgDf[y][avgDf['substrate', 'first'] == 'control']
b = avgDf[y][avgDf['substrate', 'first'] == 'activation']
test = 'two-sided'
res = wilcoxon(a, b, alternative=test, zero_method = 'wilcox')


sns.pointplot(x = x, y = y, data=avgDf, order = manips,  hue = ('cellCode', 'first'))

sns.boxplot(x = x, y = y, data=avgDf, order = manips, color = 'grey',
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.9})

fontColour = '#ffffff'
plt.title('Test : Wilcoxon paired "{:}" | p-val = {:.4f}'.format(test, res[1]), color = fontColour, fontsize = 20)

plt.xticks(fontsize=30, color = fontColour)
plt.yticks(fontsize=30, color = fontColour)
plt.ylabel(param, fontsize = 25, color = fontColour)
plt.tight_layout()
plt.savefig('D:/Anumita/MagneticPincherData/Figures/NLI_Analysis/24-05-20_Replotted/'+folder+'/Cell-based Plots/'+str(param)+'_'+str(filename)+'_.png')

#%%%% Cell-wise NLI comparison

param = 'nli_ind'
# df = dfToPlot[dfToPlot['substrate'] == 'activation']

sns.lineplot(data = dfToPlot, x = 'compNo', y = param, hue = 'cellID')

#%%%% Linearity / Nonlinarity vs deformation (strain)
plt.style.use('default')

plot = dfToPlot
y = (plot['h0'] - plot['minH'])/plot['h0']

sns.boxplot(x = 'e', y = y, data=plot, color = 'grey', 
                    medianprops={"color": 'darkred', "linewidth": 2},\
                    boxprops={ "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.9})
    
sns.swarmplot(x = 'e', y = y, data=plot,linewidth = 1, hue = 'cellID',
              edgecolor='k')
