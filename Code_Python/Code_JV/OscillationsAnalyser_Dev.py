# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:04:38 2022

@author: JosephVermeil
"""

# %% > Imports and constants

#### Main imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt


import os
import time
import random
import warnings
import itertools
import matplotlib

from copy import copy
from cycler import cycler
from datetime import date
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec
from scipy import fft, signal, interpolate

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)

import GraphicStyles as gs
import UtilityFunctions as ufun
import TrackAnalyser as taka
# import TrackAnalyser_dev_AJ as taka

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

#### Graphic options
gs.set_default_options_jv()


# %% Utility functions

def annotate_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)

def butterFilter_PlotFreqResponse(orders, frequencies, ftypes):
    N = len(ftypes)
    fig, ax = plt.subplots(1,1)
    for i in range(N):
        O, F, fT = orders[i], frequencies[i], ftypes[i]
        b, a = signal.butter(O, F, fT, analog=True)
        w, h = signal.freqs(b, a)
        ax.semilogx(w, 20 * np.log10(abs(h)), label = fT + ' order ' + str(O))
        ax.set_title('Butterworth filter frequency response')
        ax.set_xlabel('Frequency [radians / second]')
        ax.set_ylabel('Amplitude [dB]')
        ax.margins(0, 0.1)
        ax.grid(which='both', axis='both')
        try:
            for f in F:
                ax.axvline(f, color='red', lw = 1, ls = '--') # cutoff frequencies
        except:
            ax.axvline(F, color='red', lw = 1, ls = '--') # cutoff frequency
    ax.legend()
    plt.show()
    
frequencies = [(0.1, 10)] * 3
orders = [2, 4, 6]
ftypes = ['bandpass'] * 3
    
butterFilter_PlotFreqResponse(orders, frequencies, ftypes)


frequencies = [0.1, 10, (0.1, 10)]
orders = [4, 4, 2]
ftypes = ['high', 'low', 'bandpass']
    
butterFilter_PlotFreqResponse(orders, frequencies, ftypes)


# %% TimeSeries plots


# %%% List files
allTimeSeriesDataFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                          if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) 
                              and f.endswith(".csv")
                              and ('sin' in f))]
print(allTimeSeriesDataFiles)


# %%% Get a time series

df1hz = taka.getCellTimeSeriesData('22-03-21_M3_P1_C1_sin5-3_1Hz')
df2hz = taka.getCellTimeSeriesData('22-03-21_M3_P1_C1_sin5-3_2Hz')


# %%% Plot a time series

taka.plotCellTimeSeriesData('22-03-21_M3_P1_C1_sin5-3_1Hz')
taka.plotCellTimeSeriesData('22-03-21_M3_P1_C1_sin5-3_2Hz')


# %%% Variation on the plotCellTimeSeriesData -> Sinus

cellID = '22-03-21_M3_P1_C1_sin5-3_1Hz'
# cellID = '22-03-21_M3_P1_C1_sin5-3_2Hz'
fromPython = True

X = 'T'
Y = np.array(['B', 'F'])
units = np.array([' (mT)', ' (pN)'])
tSDF = taka.getCellTimeSeriesData(cellID, fromPython)


fig, ax = plt.subplots(1,1, figsize = (10,5))

if not tSDF.size == 0:
    tsDFplot = tSDF #[(tSDF['T']>=3.04) & (tSDF['T']<=6.05)]
    
    ax.plot(tsDFplot['T'], 1000*(tsDFplot['D3']-4.503), color = gs.colorList40[30], label = 'Thickness')
    ax.set_ylabel('Thickness (nm)', color = gs.colorList40[30])
    # ax.set_ylim([0, 350])
    ax.set_xlabel('Time (s)')

    axR = ax.twinx()
    axR.plot(tsDFplot['T'], tsDFplot['F'], color = gs.colorList40[23], label = 'Force')
    axR.set_ylabel('Force (pN)', color = gs.colorList40[23])
    # axR.set_ylim([0, 150])


    plt.gcf().show()
else:
    print('cell not found')
# plt.rcParams['axes.prop_cycle'] = defaultColorCycle



# %% Frequency detection

Traw, Braw, Fraw = tSDF['T'].values, tSDF['B'].values, tSDF['F'].values
D3raw = tSDF['D3'].values

B_out = np.mean(Braw[:20])

# iStart iStop maskSin

maskStartStop = np.abs(Braw - B_out) > 0.1
iStart = ufun.findFirst(True, maskStartStop)
iStop  = len(maskStartStop) - ufun.findFirst(True, maskStartStop[::-1])

A = np.arange(len(Traw), dtype = int)
maskSin = (A >= iStart) & (A < iStop)

T, B, F = Traw[maskSin], Braw[maskSin], Fraw[maskSin]
D3 = 1000*(D3raw[maskSin] - 4.477)


# FFT


SAMPLE_RATE = 100

yf = fft.rfft(B)
xf = fft.rfftfreq(np.sum(maskSin), 1/SAMPLE_RATE)

# plt.plot(xf, np.abs(yf))
# plt.show()

Bfreq = xf[1 + np.argmax(yf[1:])]



yf = fft.rfft(F)
xf = fft.rfftfreq(np.sum(maskSin), 1/SAMPLE_RATE)

# plt.plot(xf, np.abs(yf))
# plt.show()

Ffreq = xf[1 + np.argmax(yf[1:])]



yf = fft.rfft(D3)
xf = fft.rfftfreq(np.sum(maskSin), 1/SAMPLE_RATE)

# plt.plot(xf, np.abs(yf))
# plt.show()

D3freq = xf[10 + np.argmax(yf[10:])]

# %% Test of wavelet transform

dt = 1/100
fs = 1/dt
w = 10.
freq = np.linspace(0.8, 1.2, 100)
widths = w*fs / (2*freq*np.pi)

# widths = np.arange(1,51)
cwtmatr = signal.cwt(D3, signal.morlet2, widths=widths, w=w)

# plt.imshow(np.abs(cwtmatr), cmap='PRGn', aspect='auto',
#            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.pcolormesh(T, freq, np.abs(cwtmatr), cmap='viridis', shading='gouraud')
plt.show()

# %% Filters optimisation

Traw, Braw, Fraw = tSDF['T'].values, tSDF['B'].values, tSDF['F'].values
D3raw = tSDF['D3'].values
B_out = np.mean(Braw[:20])

# iStart iStop maskSin
maskStartStop = np.abs(Braw - B_out) > 0.1
iStart = ufun.findFirst(True, maskStartStop)
iStop  = len(maskStartStop) - ufun.findFirst(True, maskStartStop[::-1])
A = np.arange(len(Traw), dtype = int)
maskSin = (A >= iStart) & (A < iStop)
T, B, F = Traw[maskSin], Braw[maskSin], Fraw[maskSin]
D3 = 1000*(D3raw[maskSin] - 4.503)

SamplingFreq = 100

LowCut = D3freq/5
HighCut = D3freq*5

LowLowCut = D3freq * 0.35

D = 50/D3freq
W = 30/D3freq

b, a = signal.butter(4, HighCut, btype='lowpass', analog=False, output='ba', fs=SamplingFreq)
D3_f = signal.filtfilt(b, a, D3, padlen=150)

b, a = signal.butter(4, LowCut, btype='highpass', analog=False, output='ba', fs=SamplingFreq)
D3_f_alt = signal.filtfilt(b, a, D3, padlen=150)

b, a = signal.butter(4, LowCut, btype='highpass', analog=False, output='ba', fs=SamplingFreq)
D3_ff = signal.filtfilt(b, a, D3_f, padlen=150)

# b, a = signal.butter(2, LowLowCut, btype='lowpass', analog=False, output='ba', fs=SamplingFreq)
b, a = signal.butter(4, (LowCut, HighCut), btype='bandpass', analog=False, output='ba', fs=SamplingFreq)
D3_f2 = signal.filtfilt(b, a, D3, padlen=150)

Pf, props_Pf = signal.find_peaks(D3_f, height=None, threshold=None, distance=D, 
                                 prominence=None, width=W, wlen=None, rel_height=0.5, plateau_size=None)
XPf = [T[i] for i in Pf]
YPf = [D3_f[i] for i in Pf]

Pf2, props_Pf2 = signal.find_peaks(D3_f2, height=None, threshold=None, distance=D, 
                                 prominence=None, width=W, wlen=None, rel_height=0.5, plateau_size=None)
XPf2 = [T[i] for i in Pf2]
YPf2 = [D3_f2[i] for i in Pf2]

Pforce, props_Pforce = signal.find_peaks(-F, height=None, threshold=None, distance=D, 
                                 prominence=None, width=W, wlen=None, rel_height=0.5, plateau_size=None)
XPforce = [T[i] for i in Pforce]


# C = np.correlate(-F[:500], D3, mode = 'valid')
# idx_max = np.argmax(C[:25])
# N = len(T)//2
# delta = N - idx_max
# deltaT = 0.01 * idx_max

analytic_signal = signal.hilbert(D3_f2)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) /
                           (2.0*np.pi) * fs)

fig, ax = plt.subplots(1,1, figsize = (10,5))

# ax.plot(T, D3_f, color = gs.colorList40[30], ls = '-', lw = 0.5, label = 'Thickness')
# for ii in range(len(XPf)):
#     ax.plot([XPf[ii], XPf[ii]], [YPf[ii]-400, YPf[ii]+400], 
#             color = gs.colorList40[30], lw = 0.5, ls='--')
    
ax.plot(T, D3_f2, color = gs.colorList40[22], ls = '-', lw=1.25, label = 'Thickness')
# ax.plot(T, amplitude_envelope, color = gs.colorList40[32], ls = '--')
envelope_X = np.zeros(len(T) + 2)
envelope_X[1:-1] = T
envelope_X[0], envelope_X[-1] = T[0], T[-1]
envelope_Y = np.zeros(len(T) + 2)
envelope_Y[1:-1] = amplitude_envelope

ax.fill(envelope_X, envelope_Y, envelope_X, -envelope_Y, 
        color = gs.colorList40[2], zorder = 1, alpha = 0.5)  
for ii in range(len(XPf2)):
    ax.plot([XPf2[ii], XPf2[ii]], [YPf2[ii]-400, YPf2[ii]+400], 
            color = gs.colorList40[32], lw = 0.5, ls='--')
    
ylims = ax.get_ylim()

D3maxAmpli = np.max(np.abs(D3_f2))
ax.set_ylim([0 - 1.1*D3maxAmpli, 0 + 1.1*D3maxAmpli])
ax.plot(ax.get_xlim(), [0, 0], color = 'k', lw = 0.8)
    
ax.set_ylabel('Thickness (nm)', color = gs.colorList40[20])
ax.set_xlabel('Time (s)')

Favg = np.mean(F)
axR = ax.twinx()
axR.plot(T, F, color = gs.colorList40[13], lw=1.25, label = 'Force')
for x in XPforce:
    axR.plot([x, x], [0, 150], color = gs.colorList40[33], lw = 0.5, ls='--')
FmaxAmpli = np.max(np.abs(F))
axR.plot(axR.get_xlim(), [Favg, Favg], color = 'k', lw = 0.8)

axR.set_ylabel('Force (pN)', color = gs.colorList40[13])
axR.set_ylim([Favg - FmaxAmpli*0.55, Favg + FmaxAmpli*0.55])

# axR.plot(T[:len(C)], C, color = gs.colorList40[13])

Favg = np.mean(F)
fig2, ax2 = plt.subplots(1,1, figsize = (7,6))
sc = ax2.scatter(D3_f2, F, marker = 'o', c = T, s = 2, cmap = 'plasma')
plt.colorbar(sc, ax = ax2, label="Time (s)")
ax2.plot([0, 0], ax2.get_ylim(), color = 'k', lw = 0.8)
ax2.plot(ax2.get_xlim(), [Favg, Favg], color = 'k', lw = 0.8)
ax2.set_xlabel('Thickness (nm)')
ax2.set_ylabel('Force (pN)')


plt.tight_layout()
plt.show()


# %% Correlation based phase shift detection

Traw, Braw, Fraw = tSDF['T'].values, tSDF['B'].values, tSDF['F'].values
D3raw = tSDF['D3'].values
B_out = np.mean(Braw[:20])

# iStart iStop maskSin
maskStartStop = np.abs(Braw - B_out) > 0.1
iStart = ufun.findFirst(True, maskStartStop)
iStop  = len(maskStartStop) - ufun.findFirst(True, maskStartStop[::-1])
A = np.arange(len(Traw), dtype = int)
maskSin = (A >= iStart) & (A < iStop)
T, B, F = Traw[maskSin], Braw[maskSin], Fraw[maskSin]
D3 = 1000*(D3raw[maskSin] - 4.503)

# Filtering
SamplingFreq = 100
NPtsPeriod = SamplingFreq/D3freq

LowCut = D3freq/5
HighCut = D3freq*5

D = 50/D3freq
W = 30/D3freq

b, a = signal.butter(4, HighCut, btype='lowpass', analog=False, output='ba', fs=SamplingFreq)
D3_LP = signal.filtfilt(b, a, D3, padlen=150)

b, a = signal.butter(4, (LowCut, HighCut), btype='bandpass', analog=False, output='ba', fs=SamplingFreq)
D3_BP = signal.filtfilt(b, a, D3, padlen=150)

P_LP, props_P_LP = signal.find_peaks(D3_LP, height=None, threshold=None, distance=D, 
                                 prominence=None, width=W, wlen=None, rel_height=0.5, plateau_size=None)
XP_LP = [T[i] for i in P_LP]
YP_LP = [D3_LP[i] for i in P_LP]
aP_LP, props_aP_LP = signal.find_peaks(-D3_LP, height=None, threshold=None, distance=D, 
                                 prominence=None, width=W, wlen=None, rel_height=1, plateau_size=None)
XaP_LP = [T[i] for i in aP_LP]
YaP_LP = [D3_LP[i] for i in aP_LP]

P_BP, props_P_BP = signal.find_peaks(D3_BP, height=None, threshold=None, distance=D, 
                                 prominence=None, width=W, wlen=None, rel_height=0.5, plateau_size=None)
XP_BP = [T[i] for i in P_BP]
YP_BP = [D3_BP[i] for i in P_BP]

Pforce, props_Pforce = signal.find_peaks(F, height=None, threshold=None, distance=D, 
                                 prominence=None, width=W, wlen=None, rel_height=0.5, plateau_size=None)
XPforce = [T[i] for i in Pforce]
YPforce = [F[i] for i in Pforce]

mask_D3_LP = ((T > T[Pforce[8]]) & (T < T[Pforce[-8]]))
mask_F = ((T > T[Pforce[2]]) & (T < T[Pforce[-2]]))

C = signal.correlate(F[mask_F], D3_LP[mask_D3_LP], mode = 'full')
lags = signal.correlation_lags(len(D3_LP[mask_D3_LP]), len(F[mask_F]))
mask_lags = (np.abs(lags) < ((SamplingFreq/D3freq)//4)) & (lags <= 0)
idx_max = np.argmin(C[mask_lags]) - (SamplingFreq/D3freq)//4

# Tcorr = T[mask_F]
# Tcorr = Tcorr[:len(C)]

# deltaT = 0.01 * idx_max
deltaT = 0.01 * idx_max/2
deltaT = -0.05
# deltaT = 0


fig, ax = plt.subplots(1,1, figsize = (10,5))

# ax.plot(T, D3_f, color = gs.colorList40[30], ls = '-', lw = 0.5, label = 'Thickness')
# for ii in range(len(XPf)):
#     ax.plot([XPf[ii], XPf[ii]], [YPf[ii]-400, YPf[ii]+400], 
#             color = gs.colorList40[30], lw = 0.5, ls='--')
    
ax.plot(T[mask_D3_LP], D3_LP[mask_D3_LP], color = gs.colorList40[22],
        ls = '-', lw=1.25, label = 'Thickness')

for ii in range(2, len(XP_LP)):
    ax.plot([XP_LP[ii], XP_LP[ii]], [YP_LP[ii]-150, YP_LP[ii]+50], 
            color = gs.colorList40[32], lw = 0.5, ls='--')
    ax.plot([XaP_LP[ii], XaP_LP[ii]], [YaP_LP[ii]-150, YaP_LP[ii]+50], 
            color = gs.colorList40[32], lw = 0.5, ls='--')
    
ax.set_ylabel('Thickness (nm)', color = gs.colorList40[20])
ax.set_xlabel('Time (s)')

Favg = np.mean(F)
axR = ax.twinx()
axR.plot(T[mask_F]-deltaT, F[mask_F], color = gs.colorList40[13], lw=1.25, label = 'Force')
for ii in range(3, len(XPforce)-1):
    axR.plot([XPforce[ii]-deltaT, XPforce[ii]-deltaT], [YPforce[ii]-50, YPforce[ii]+50],
             color = gs.colorList40[33], lw = 0.5, ls='--')
FmaxAmpli = np.max(np.abs(F))
axR.plot(axR.get_xlim(), [Favg, Favg], color = 'k', lw = 0.8)

axR.set_ylabel('Force (pN)', color = gs.colorList40[13])
# axR.set_ylim([Favg - FmaxAmpli*0.55, Favg + FmaxAmpli*0.55])

# axR.plot(T[:len(C)], C, color = gs.colorList40[13])

fig2, ax2 = plt.subplots(1,1, figsize = (12,6))
ax2.plot(lags, C)
ax2.plot([0,0], ax2.get_ylim())


plt.tight_layout()
plt.show()

# %% A better phase shift detection

# %%% Import data

cellID = '22-03-21_M3_P1_C1_sin5-3_2Hz'
# cellID = '22-03-21_M3_P1_C1_sin5-3_2Hz'
fromPython = True

X = 'T'
Y = np.array(['B', 'F'])
units = np.array([' (mT)', ' (pN)'])
tSDF = taka.getCellTimeSeriesData(cellID, fromPython)

# %%% Cutting 

Traw, Braw, Fraw = tSDF['T'].values, tSDF['B'].values, tSDF['F'].values
D3raw = tSDF['D3'].values
B_out = np.mean(Braw[:20])

# iStart iStop maskSin
maskStartStop = np.abs(Braw - B_out) > 0.1
iStart = ufun.findFirst(True, maskStartStop)
iStop  = len(maskStartStop) - ufun.findFirst(True, maskStartStop[::-1])
A = np.arange(len(Traw), dtype = int)
maskSin = (A >= iStart) & (A < iStop)
T, B, F = Traw[maskSin], Braw[maskSin], Fraw[maskSin]
D3 = 1000*(D3raw[maskSin] - 4.503)

# %%% Filtering

D3Freq = 2
SamplingFreq = 100
NPtsPeriod = SamplingFreq/D3Freq

LowCut = D3Freq/3
HighCut = D3Freq*3

b, a = signal.butter(4, HighCut, btype='lowpass', 
                     analog=False, output='ba', fs=SamplingFreq)
D3_LP = signal.filtfilt(b, a, D3, padlen=150)

b, a = signal.butter(4, (LowCut, HighCut), btype='bandpass', 
                     analog=False, output='ba', fs=SamplingFreq)
D3_BP = signal.filtfilt(b, a, D3, padlen=150)

# analytic_signal = signal.hilbert(D3_f2)
# amplitude_envelope = np.abs(analytic_signal)
# instantaneous_phase = np.unwrap(np.angle(analytic_signal))
# instantaneous_frequency = (np.diff(instantaneous_phase) /
#                            (2.0*np.pi) * fs)

fig, axes = plt.subplots(3,1, figsize = (14,9))

# ax.plot(T, D3_f, color = gs.colorList40[30], ls = '-', lw = 0.5, label = 'Thickness')

ax = axes[0]
ax.plot(T, D3, color = gs.colorList40[20], ls = '-', label = 'Thickness_raw')
ax.plot(T, D3_LP, color = gs.colorList40[31], ls = '-', lw=1.25, 
        label = 'Thickness_LowPass')
ax.set_xlim([0, 1.05*np.max(T)])
ax.set_ylabel('Thickness (nm)', color = gs.colorList40[20])
ax.legend(loc='upper left')
ax = axes[1]
ax.plot(T, D3_BP, color = gs.colorList40[32], ls = '-', lw=1.25, 
        label = 'Thickness_BandPass')
ax.axhline(np.mean(D3_BP), color = 'gray', lw = 0.8)
ax.set_ylabel('Thickness (nm)', color = gs.colorList40[20])
ax.set_xlim([0, 1.05*np.max(T)])
ax.legend(loc='upper left')


ax = axes[2]
ax.plot(T, F, color = gs.colorList40[13], lw=1.25, label = 'Force_raw')
ax.axhline(np.mean(F), color = 'gray', lw = 0.8)
ax.set_ylabel('Force (pN)', color = gs.colorList40[13])
ax.set_xlim([0, 1.05*np.max(T)])
ax.legend(loc='lower left')
ax.set_xlabel('Time (s)')


plt.tight_layout()
plt.show()

# %%% Correlation

D3_f = D3_BP


SamplingFreq = 100
NPtsPeriod = int(SamplingFreq/D3Freq)

width = int(5*NPtsPeriod)
listCenters = [i for i in range(width, len(T) - width, 10)]
allIdx = np.arange(0, len(T))

ListT = []
ListDPhi = []

PLOT = False

for ic in listCenters[1:-1]:
    mask_D3 = ((allIdx > ic - (width//2)) & (allIdx < ic + (width//2) + 1))
    mask_F =  ((allIdx > ic - (width//2) - NPtsPeriod) 
               & (allIdx < ic + (width//2) + NPtsPeriod + 1))
    
    ListT.append(T[ic])
    
    altD3 = D3_f[mask_D3] - np.mean(D3_f[mask_D3])
    altF = F[mask_F] - np.mean(F[mask_F])
    
    
    # C = signal.correlate(D3_f[mask_D3], F[mask_F], mode = 'full')
    # lags = signal.correlation_lags(len(D3_f[mask_D3]), len(F[mask_F]))
    C = signal.correlate(altD3, altF, mode = 'full')
    lags = signal.correlation_lags(len(altD3), len(altF))
    
    lags_corr = lags + NPtsPeriod
    mask_lags_corr = (np.abs(lags_corr) < ((NPtsPeriod)//2)) # & (lags <= 0)
    x1, y1 = lags_corr[mask_lags_corr], C[mask_lags_corr]
    f = interpolate.interp1d(x1, y1, kind = 'quadratic')
    x2 = np.linspace(np.min(x1), np.max(x1), 10*len(x1))
    y2 = f(x2)
    
    idx_min = np.argmin(y2)
    delta = x2[idx_min]
    
    Xmin, Ymin = x2[idx_min], y2[idx_min]
    
    deltaT = delta / SamplingFreq
    
    
    if PLOT:
        fig, axes = plt.subplots(1,2, figsize = (12,6))
        ax = axes[0]
        ax.set_title('Getting the phase-shift between F and D3...')
        ax.plot(T[mask_D3], altD3, color = gs.colorList40[20],
                ls = '-', lw=1.25, label = 'Thickness')
        axR = ax.twinx()
        axR.plot(T[mask_F], altF, color = gs.colorList40[23], lw=1.25, 
                 label = 'Force')
        axR.plot(T[mask_F]+deltaT, -altF, 
                 color = gs.colorList40[13], lw=1.25, ls = '--',
                 label = 'Minus-Force-Shifted')
        
        ax.set_ylabel('Thickness (nm)', color = gs.colorList40[20])
        ax.set_xlabel('Time (s)')
        axR.set_ylabel('Force (pN)', color = gs.colorList40[13])
        axR.legend(loc = 'lower right')
        
        ax = axes[1]
        ax.set_title('... using their cross-correlation function')
        ax.plot(lags_corr, C, label = 'XCorr of F and D3')
        ax.plot(x2, y2, 
                color = 'g', lw = 2)
        ax.plot(Xmin, Ymin, 
                color = 'g', marker = 'P', markersize = 6, mec = 'k',
                label = 'Minimum of XCorr for F and D3')
        ax.axvline(0, color='red', lw = 1, ls = '--')
        ax.set_xlabel('Lag (u = 1/SamplingFreq = 1/{:.0f} s)'.format(SamplingFreq))
        ax.legend(loc = 'lower right')

    deltaPhi = deltaT*D3Freq*2*np.pi
    ListDPhi.append(deltaPhi)
    print(deltaPhi)
    
T0, DPhi0 = np.array(ListT), np.array(ListDPhi)
DPhi = np.mean(DPhi0)
DPhi_std = np.std(DPhi0)

# fig, ax = plt.subplots(1,1, figsize = (10,5))
# ax.plot(T, D3_f, color = gs.colorList40[20],
#         ls = '-', lw=1.25, label = 'Thickness')
# ax.set_ylabel('Thickness (nm)', color = gs.colorList40[20])
# ax.set_xlabel('Time (s)')

# ax.plot(T0, DT0*2000, 'b-')

# axR = ax.twinx()
# axR.plot(T, F, color = gs.colorList40[23], lw=1.25, label = 'Force')
# axR.set_ylabel('Force (pN)', color = gs.colorList40[23])

fig = plt.figure(figsize = (12,8))
fig.suptitle("Computing the phase-shift")

gs1 = GridSpec(3, 3)
ax1 = fig.add_subplot(gs1[:-1, :])
ax2 = fig.add_subplot(gs1[-1, :])

# annotate_axes(fig)

plt.show()

ax1.plot(T, D3_f, color = gs.colorList40[20],
        ls = '-', lw=1.25, label = 'Thickness_BandPass')
ax1.set_xlabel('Time (s)')
ax1.set_xlim([0, 1.05*np.max(T)])
ax1.set_ylabel('Thickness (nm)', color = gs.colorList40[20])
ax1.set_ylim([-100, 50])
ax1.legend(loc = 'upper left')


axR = ax1.twinx()
axR.plot(T, F, color = gs.colorList40[23], lw=1.25, label = 'Force_Raw')
axR.set_ylabel('Force (pN)', color = gs.colorList40[23])
axR.set_ylim([0,200])
axR.legend(loc = 'lower right')


ax2.plot(T0, DPhi0, 'b-', label = 'Mean dPhi = {:.2f} - Std = {:.3f}'.format(DPhi, DPhi_std))

ax2.set_xlabel('Time (s)')
ax2.set_xlim([0, 1.05*np.max(T)])
ax2.set_ylabel('Phase-shift (rad)')
ax2.set_ylim([0,np.pi/4])
ax2.legend(loc = 'upper left')

# plt.tight_layout()
plt.show()


# %%

Traw, Braw, Fraw = tSDF['T'].values, tSDF['B'].values, tSDF['F'].values
D3raw = tSDF['D3'].values
B_out = np.mean(Braw[:20])

# iStart iStop maskSin
maskStartStop = np.abs(Braw - B_out) > 0.1
iStart = ufun.findFirst(True, maskStartStop)
iStop  = len(maskStartStop) - ufun.findFirst(True, maskStartStop[::-1])
A = np.arange(len(Traw), dtype = int)
maskSin = (A >= iStart) & (A < iStop)
T, B, F = Traw[maskSin], Braw[maskSin], Fraw[maskSin]
D3 = 1000*(D3raw[maskSin] - 4.503)

# Filter

b, a = signal.butter(2, 0.1, btype='lowpass', analog=False, output='ba', fs=None)
D3_f = signal.filtfilt(b, a, D3, padlen=150)

b, a = signal.butter(2, 0.01, btype='highpass', analog=False, output='ba', fs=None)
D3_ff = signal.filtfilt(b, a, D3_f, padlen=150)

Pf, props_Pf = signal.find_peaks(D3_f, height=10, threshold=None, distance=20, 
                                 prominence=None, width=20, wlen=None, rel_height=0.5, plateau_size=None)
XPf = [T[i] for i in Pf]

Pff, props_Pff = signal.find_peaks(D3_ff, height=10, threshold=None, distance=20, 
                                 prominence=None, width=20, wlen=None, rel_height=0.5, plateau_size=None)
XPff = [T[i] for i in Pff]

Pforce, props_Pforce = signal.find_peaks(-F, height=None, threshold=None, distance=20, 
                                 prominence=None, width=20, wlen=None, rel_height=0.5, plateau_size=None)
XPforce = [T[i] for i in Pforce]


fig, ax = plt.subplots(1,1, figsize = (10,5))

ax.plot(T, D3, color = gs.colorList40[10], ls = '-', label = 'Thickness')
ax.plot(T, D3_f, color = gs.colorList40[32], ls = '--', label = 'Thickness_filtered_LP')
ax.plot(T, D3_ff, color = gs.colorList40[20], ls = '-', label = 'Thickness_filtered_LPHP')
ylims = ax.get_ylim()
for x in XPf:
    ax.plot([x, x], ylims, color = gs.colorList40[32], lw = 0.5, ls='--')
for x in XPff:
    ax.plot([x, x], ylims, color = gs.colorList40[30], lw = 0.5, ls='--')
    
ax.set_ylabel('Thickness (nm)', color = gs.colorList40[20])
# ax.set_ylim([0, 350])
ax.set_xlabel('Time (s)')

axR = ax.twinx()
axR.plot(T, F, color = gs.colorList40[13], label = 'Force')
axR.set_ylabel('Force (pN)', color = gs.colorList40[13])
axR.set_ylim([0, 150])
for x in XPforce:
    axR.plot([x, x], [0, 150], color = gs.colorList40[33], lw = 0.5, ls='--')



# ax.grid(axis='y')

plt.show()

# %%
from scipy import signal
import matplotlib.pyplot as plt
t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
widths = np.arange(1, 31)
cwtmatr = signal.cwt(sig, signal.ricker, widths)
plt.imshow(cwtmatr, extent=[-1, 1, 31, 1], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()

# %%
from scipy import signal
import matplotlib.pyplot as plt
t, dt = np.linspace(0, 1, 200, retstep=True)
fs = 1/dt
w = 10.
sig = np.cos(2*np.pi*(50 + 10*t)*t) + np.sin(40*np.pi*t)
freq = np.linspace(1, fs/2, 100)
widths = w*fs / (2*freq*np.pi)
cwtm = signal.cwt(sig, signal.morlet2, widths, w=w)
plt.pcolormesh(t, freq, np.abs(cwtm), cmap='viridis', shading='gouraud')
plt.show()

# %%

from scipy import signal
import matplotlib.pyplot as plt
t = np.linspace(0, 1.0, 2001)
xlow = np.sin(2 * np.pi * 5 * t)
xhigh = np.sin(2 * np.pi * 250 * t)
x = xlow + xhigh
b, a = signal.butter(3, (D3freq/10, D3freq*10), btype='bandpass', analog=False, output='ba', fs=None)
y = signal.filtfilt(b, a, x, padlen=150)
np.abs(y - xlow).max()
