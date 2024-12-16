# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:11:32 2024

@author: JosephVermeil
"""
# %% Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# import statsmodels.api as sm

import os
# import re
# import datetime

from scipy.optimize import curve_fit
from scipy import signal

import UtilityFunctions as ufun
import GraphicStyles as gs

gs.set_mediumText_options_jv()


# %% Functions V3

def initializeFitHertz(fpath, date, manip, cell, indent):
    f = fpath.split('//')[-1]
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
    dfI = pd.read_csv(fpath, skiprows = i-1, sep='\t')
    
    dI = {'indent_name':f, 'path':fpath, 'metadata':metadataText}
    
    dI['date'] = date
    dI['manip'] = manip
    dI['cell'] = cell
    dI['indent'] = indent
    
    time = flines[0].split('\t')[3]
    dI['time'] = time
    R = float(getLineWithString(metadataText, 'Tip radius (um)')[0][16:])
    dI['R'] = R
    
    X = float(getLineWithString(metadataText, 'X-position (um)')[0][16:])
    dI['X0'] = X
    Y = float(getLineWithString(metadataText, 'Y-position (um)')[0][16:])
    dI['Y0'] = Y
    Z = float(getLineWithString(metadataText, 'Z-position (um)')[0][16:])
    dI['Z0'] = Z
    
    s1, s2_dZ_comp, s3, s4_dT_comp = getLineWithString(metadataText, 'D[Z1]')[0][:-1].split('\t')
    V_comp = float(s2_dZ_comp)*1e-3/float(s4_dT_comp)
    dI['V_comp'] = V_comp # µm/s
    
    s1, s2, s3, s4_dT_rest = getLineWithString(metadataText, 'D[Z2]')[0][:-1].split('\t')
    dT_rest = float(s4_dT_rest)
    dI['dT_rest'] = dT_rest # s
    
    s1, s2, s3, s4_dT_relax = getLineWithString(metadataText, 'D[Z3]')[0][:-1].split('\t')
    V_relax = float(s2_dZ_comp)*1e-3/float(s4_dT_relax)
    dI['V_relax'] = V_relax # µm/s
    
    text_start_times, j = getLineWithString(metadataText, 'Step absolute start times')
    text_end_times, j = getLineWithString(metadataText, 'Step absolute end times')
    start_times = np.array(text_start_times[30:].split(',')).astype(float)
    end_times = np.array(text_end_times[28:].split(',')).astype(float)
    
    dI['ti'], dI['tf'] = start_times, end_times
    
    T = dfI['Time (s)'].values
    step = np.sum(np.array([(T >= tf) for tf in end_times]), axis=0)
    dfI['step'] = step
    dfI['Z Tip (nm)'] = dfI['Piezo (nm)'] - dfI['Cantilever (nm)']
    dI['df'] = dfI
    
    fopen.close()
    return(dI)





def FitHertz(dI, mode = 'dZmax', fractionF = 0.4, dZmax = 3,
                plot = 1, save_plot = False, save_path = ''):    
    
    #### 1. Initialize
    df = dI['df']
    R = dI['R']
    V_comp = dI['V_comp']
    
    ti, tf = dI['ti'][0], dI['tf'][0]
    compression_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
    Z = df.loc[compression_indices, 'Z Tip (nm)'].values*1e-3 # nm to µm
    F = df.loc[compression_indices, 'Load (uN)'].values*1e6 # µN to pN
    Time = df.loc[compression_indices, 'Time (s)']

    early_points = len(F)//10
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

    #### 2.1 First Fit Zone
    Z0_sup = np.max(Z)

    #### 2.2 First Fit
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


    #### 3.1 Second Fit Zone
    i_start = ufun.findFirst(True, Z >= Z01 - 1) # 1 µm margin
    if mode == 'fractionF':
        upper_threshold = F_moy + fractionF * (F_max - F_moy)
        i_stop = ufun.findFirst(True, F >= upper_threshold)
    elif mode == 'dZmax':
        i_stop = len(Z) - ufun.findFirst(True, Z[::-1] < Z01 + dZmax) - 1

    #### 3.2 Second Fit (several iterations to converge to a consistent fit)
    N_it_fit = 3
    Z_inf = Z01 - 3  # 3 µm margin
    Z0_approx = Z01
    for i in range(N_it_fit):
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
        
        # [K2tmp, Z02tmp], covM2 = curve_fit(HertzFit, Z2, F2, p0=p0, bounds=(binf, bsup))
        # if K2tmp <= 75:
        #     if i == 0:
        #         Z02, K2, Rsq2 = Z01, K1, Rsq1
        #     break
        # else:
        #     Z02, K2 = Z02tmp, K2tmp
        
        Rsq2 = ufun.get_R2(F2, HertzFit(Z2, K2, Z02))
        
        Z_inf = Z02 - 1 # 2 µm margin for the next iteration
        Z0_approx = Z02
    
    
    i_start = ufun.findFirst(True, Z >= Z02)
    
    
    Z_plotfit2 = np.linspace(Z[i_start], Z[i_stop], 100)
    F_plotfit2 = HertzFit(Z_plotfit2, K2, Z02)
    
    
    #### 3.3 Residuals
    F_fit = HertzFit(Z, K2, Z02)
    residuals = F_fit - F
    residuals_before = residuals[:i_start]
    residuals_fit = residuals[i_start:i_stop]
    residuals_after = residuals[i_stop:]
    
    print(Z02)
    
    #### 4.1 Fourier transforms
    timestep = 0.001
    fs = 1/timestep
    
    prominence1 = 0.2
    distance1 = 15

    #### 4.2 Denoising
    def fft_denoise(Deflect_fit, Deflect, Load,
                    Nfreq_to_cut = 3, prominence = 0.05, distance = 15):
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
        return(Deflect_denoised, Load_denoised)
        
    Nfreq_to_cut = 3
    Deflect_fit = df['Cantilever (nm)'].values[i_start:i_stop]*1e-3 # nm to µm
    Deflect = df['Cantilever (nm)'].values[:]*1e-3
    Load = df['Load (uN)'].values[:]*1e6 # µN to pN
    
    try:
        Deflect_denoised, Load_denoised = fft_denoise(Deflect_fit, Deflect, Load,
                                                      Nfreq_to_cut = 3)
    except:
        Deflect_denoised, Load_denoised = Deflect, Load

    df['Cantilever - denoised'] = Deflect_denoised*1e3 # µm to nm
    Z_denoised = df['Piezo (nm)'].values*1e-3 - Deflect_denoised # nm to µm
    df['Z Tip - denoised'] = Z_denoised*1e3 # µm to nm
    F_denoised = Load_denoised
    df['Load - denoised'] = F_denoised*1e-6 # pN to µN
    
    # Nfreq_to_cut = 3
    # Deflect_fit = df['Cantilever (nm)'].values[i_start:i_stop]*1e-3 # nm to µm
    # Deflect = df['Cantilever (nm)'].values[:]*1e-3
    # Load = df['Load (uN)'].values[:]*1e6 # µN to pN
    
    # prominence = 0.05
    # distance = 15
    
    # FT_D = np.abs(np.fft.fft(Deflect_fit - np.mean(Deflect_fit)))
    # n_D = FT_D.size
    # freq_D = np.fft.fftfreq(n_D, d=timestep)[:n_D//2]
    # FT_normed_D = FT_D[:n_D//2] / np.max(FT_D[:n_D//2])
    # peaks_D, peaks_prop_D = signal.find_peaks(FT_normed_D, prominence = prominence, distance=distance)

    # dfp = pd.DataFrame(peaks_prop_D)
    # dfp['freq'] = [freq_D[p] for p in peaks_D]
    # dfp = dfp[dfp['freq'] > 10]
    # dfp = dfp.sort_values(by='prominences', ascending=False)
    # # print(dfp)

    # Nfreq_to_cut = min(Nfreq_to_cut, len(dfp['freq'].values))
    # noise_freqs = dfp['freq'][:Nfreq_to_cut]
    # Deflect_denoised = np.copy(Deflect)
    # Load_denoised = np.copy(Load)
    # for nf in noise_freqs:
    #     a, b = signal.iirnotch(w0=nf, Q=10, fs=fs)
    #     Deflect_denoised = signal.lfilter(a, b, Deflect_denoised)
    #     Load_denoised = signal.lfilter(a, b, Load_denoised)

    # df['Cantilever - denoised'] = Deflect_denoised*1e3 # µm to nm
    # Z_denoised = df['Piezo (nm)'].values*1e-3 - Deflect_denoised # nm to µm
    # df['Z Tip - denoised'] = Z_denoised*1e3 # µm to nm
    # F_denoised = Load_denoised
    # df['Load - denoised'] = F_denoised*1e-6 # pN to µN
    

    #### 4.3 Denoised Fit    

    binf = [0, 0]
    bsup = [5000, Z0_sup]
    p0=[K1, 1]
    [K_d, Z0_d], covM_d = curve_fit(HertzFit, Z_denoised[i_start:i_stop], F_denoised[i_start:i_stop]
                                 , p0=p0, bounds=(binf, bsup))

    Rsq_d = ufun.get_R2(F_denoised[i_start:i_stop], HertzFit(Z_denoised[i_start:i_stop], K_d, Z0_d))
    
    F_fit_d = HertzFit(Z_denoised, Z0_d, K_d)
    
    Z_plotfit_d = np.linspace(Z_denoised[i_start], Z_denoised[i_stop], 100)
    F_plotfit_d = HertzFit(Z_plotfit_d, K_d, Z0_d)


    #### 5. Make Results
    results = {'R':R,
               'V_comp':V_comp,
                'mode':mode,
                'fractionF':fractionF, 
                'dZmax':dZmax,
                'Z_start':Z_start,
                'Z_stop':Z_stop,
                'F_moy':F_moy,
                'F_std':F_std,
                'F_max':F_max,
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
                }



    #### 6.1 Plot == 1
    if plot == 1:
        plt.ioff()
        fig, axes = plt.subplots(2, 2, figsize = (10,6))
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
        ax.plot(Z_plotfit1, F_plotfit1*1e-3, 
                ls = '-', color = 'gold', zorder = 5, label = labelFit1)
        
        labelFit2 = f'Second fit\nZ0 = {Z02:.2f} +/- {covM2[1,1]**0.5:.2f} µm\n'
        labelFit2 += f'Yeff = {K2:.0f} +/- {covM2[0,0]**0.5:.0f} Pa\nR² = {Rsq2:.3f}'
        ax.plot(Z_plotfit2, F_plotfit2*1e-3, 
                ls = '-', color = 'red', zorder = 5, label = labelFit2)
        

        ax.legend(loc='upper left')




        #### axes[1,0]
        ax = axes[1,0]

        
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
        ax.plot(Z_plotfit_d, F_plotfit_d*1e-3, 
                ls = '-', color = 'lime', zorder = 5, label = labelFit_d)

        ax.legend()
        

    #### 6.2 Plot == 2
    if plot == 2:
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
        
        
    #### 6.3 Plot == 3
    if plot == 3:
        plt.ioff()
        gs.set_manuscript_options_jv()
        fig, axes = plt.subplots(3, 1, figsize = (10/gs.cm_in, 18/gs.cm_in))
        colorList = gs.colorList10[:len(dI['ti'])]
        
        def Hertz(z, Z0, K, R, F0):
            zeroInd = 1e-9 * np.ones_like(z)
            d = z - Z0
            d = np.where(d>0, d, zeroInd)
            f = (4/3) * K * R**0.5 * (d)**1.5 + F0
            return(f)
        
        #### axes[0]
        ax = axes[0]
        ax.plot(df['Time (s)'], df['Piezo (nm)']/1e3, 'b-', label='Displac.')
        ax.plot(df['Time (s)'], df['Z Tip (nm)']/1e3, color='gold', ls='-', label='Z Tip Pos.')
        ax.plot(df['Time (s)'], df['Cantilever (nm)']/1e3, 'g-', label='Deflect.')
        ax.plot(df['Time (s)'], df['Indentation (nm)']/1e3, 'r-', label='Indent.')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (µm)')
        ax.grid(axis='both')
        ax.legend(loc='upper right', fontsize = 6)
        
        #### ----
        # ax = axes[0,1] # F in nN here : µN*1e6*1e-3 // pN*1e-3
        # for ti, tf, c in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]):
        #     step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
        #     ax.plot(df.loc[step_indices, 'Z Tip (nm)']*1e-3, df.loc[step_indices, 'Load (uN)']*1e6*1e-3, color=c,
        #             marker = '.', markersize = 2, ls = '')
        
        # ax.set_xlabel('Distance (um)')
        # ax.set_ylabel('Load (nN)')
        
        # ax.axvline(Z[i_start], ls = ':', color = 'k', lw=1, zorder = 4)
        # ax.axvline(Z[i_stop], ls = '-.', color = 'k', lw=1, zorder = 4)
        # ax.axhline(F_moy*1e-3, ls = '-', color = 'k', zorder = 4)
        # ax.axhline((F_moy+F_std)*1e-3, ls = '--', color = 'k', zorder = 4)
        # labelFit1 = f'First fit\nZ0 = {Z01:.2f} +/- {covM1[1,1]**0.5:.2f} µm\n'
        # labelFit1 += f'Yeff = {K1:.0f} +/- {covM1[0,0]**0.5:.0f} Pa\nR² = {Rsq1:.3f}'
        # ax.plot(Z_plotfit1, F_plotfit1*1e-3, 
        #         ls = '-', color = 'gold', zorder = 5, label = labelFit1)
        
        # labelFit2 = f'Second fit\nZ0 = {Z02:.2f} +/- {covM2[1,1]**0.5:.2f} µm\n'
        # labelFit2 += f'Yeff = {K2:.0f} +/- {covM2[0,0]**0.5:.0f} Pa\nR² = {Rsq2:.3f}'
        # ax.plot(Z_plotfit2, F_plotfit2*1e-3, 
        #         ls = '-', color = 'red', zorder = 5, label = labelFit2)
        

        # ax.legend(loc='upper left')




        #### axes[1]
        ax = axes[1]

        
        ax.plot(df['Time (s)'], df['Load (uN)']*1e6*1e-3 - F_moy*1e-3, color='darkred', ls='-', label = 'Load')
        # ax.axhline(F_moy*1e-3, ls = '-', color = 'k', lw=1, zorder = 3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Load (nN)')
        ax.legend(loc='upper right')
        ax.grid(axis='both')


        #### axes[2]
        ax = axes[2] # F in nN here : µN*1e6*1e-3 // pN*1e-3
        ax.plot(Z_denoised[compression_indices], F_denoised[compression_indices]*1e-3 - F_moy*1e-3,
                color=colorList[0], marker = '.', markersize = 2, ls = '')
        
        ax.set_xlabel('Distance (um)')
        ax.set_ylabel('Load (nN)')
        
        ax.axvline(Z_denoised[i_start], ls = ':', color = 'k', lw=1, zorder = 4)
        ax.axvline(Z_denoised[i_stop], ls = '-.', color = 'k', lw=1, zorder = 4)
        # ax.axhline(F_moy*1e-3, ls = '-', color = 'k', zorder = 3)
        # ax.axhline((F_moy+F_std)*1e-3, ls = '--', color = 'k', zorder = 4)
        labelFit_d = 'Denoised fit\n'
        labelFit_d += f'$Z_0$ = {Z0_d:.2f} $\pm$ {covM_d[1,1]**0.5:.2f} µm\n'
        labelFit_d += '$Y_{eff}$' + f' = {K_d:.0f} $\pm$ {covM_d[0,0]**0.5:.0f} Pa\nR² = {Rsq_d:.3f}'
        ax.plot(Z_plotfit_d, F_plotfit_d*1e-3 - F_moy*1e-3, 
                ls = '-', color = 'lime', zorder = 5, label = labelFit_d)
        ax.grid(axis='both')

        ax.legend(fontsize = 6)
        
        
        
        
    dictPlotName = {1:'ChiaroPlot', 2:'DenoisingPlot', 3:'ManuscriptPlot'}
    if plot >= 1:
        figtitle = f"{dI['date']}_{dI['manip']}_{dI['cell']}_{dI['indent']}_{dictPlotName[plot]}"
        fig.suptitle(figtitle)
        plt.tight_layout()
        show = True
        if show:
            plt.show()
        else:
            plt.close('all')
        if save_plot:
            ufun.archiveFig(fig, name = figtitle, ext = '.png', dpi = 200,
                            figDir = save_path, figSubDir = '', cloudSave = 'flexible')
            ufun.archiveFig(fig, name = figtitle, ext = '.pdf', dpi = 100,
                            figDir = save_path, figSubDir = '', cloudSave = 'flexible')
            
        plt.ion()
        
        
        
    
    return(results)



def findHardSurface(dI, plot = False, save_plot = False, save_path=''): 
    #### 1. Initialize
    df = dI['df']
    R = dI['R']
    
    ti, tf = dI['ti'][0], dI['tf'][0]
    compression_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
    Z = df.loc[compression_indices, 'Z Tip (nm)'].values*1e-3 # nm to µm
    F = df.loc[compression_indices, 'Load (uN)'].values*1e6 # µN to pN
    Time = df.loc[compression_indices, 'Time (s)']

    early_points = len(F)//20
    F_moy = np.median(F[:early_points])
    F_std = 6*np.std(F[:early_points])
    F_max = np.max(F)

    Z_start = np.min(Z)
    Z_stop = np.max(Z)
    
    #### 2.1 First Fit Zone
    Z0_sup = np.max(Z)

    #### 2.2 First Fit
    upper_threshold = F_moy + 0.75 * (F_max - F_moy)
    i_start = len(F) - ufun.findFirst(True, F[::-1] < F_moy + F_std) - 1
    Z0, F0 = Z[:i_start], F[:i_start]*1e-3
    Z1, F1 = Z[i_start:], F[i_start:]*1e-3
    cap = min(len(Z1), 500)
    Z1, F1 = Z1[:cap], F1[:cap]
    
    [b0, a0], results0 = ufun.fitLineHuber(Z0, F0)
    [d1, c1], results1 = ufun.fitLineHuber(F1, Z1)
    a1, b1 = 1/c1, -d1/c1
    
    z1, z2, z3 = np.min(Z0), np.min(Z1), np.max(Z1)*1.01
    
    # Zc = (b1-b0)/(a0-a1)
    Zc = (c1*b0 + d1)/(1-c1*a0)
    
    if plot:
        fig, axes = plt.subplots(1,3, figsize = (15,5))
        colorList = gs.colorList10[:len(dI['ti'])]
        
        #### axes[0]
        ax = axes[0]
        ax.plot(df['Time (s)'], df['Piezo (nm)'], 'b-', label='Displacement')
        ax.plot(df['Time (s)'], df['Z Tip (nm)'], color='gold', ls='-', label='Z Tip')
        ax.plot(df['Time (s)'], df['Cantilever (nm)'], 'g-', label='Deflection')
        ax.plot(df['Time (s)'], df['Indentation (nm)'], 'r-', label='Indentation')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (nm)')
        ax.legend(loc='upper right')
        
        #### axes[1]
        ax = axes[1] # F in nN here : µN*1e6*1e-3 // pN*1e-3
        for ti, tf, c in zip(dI['ti'][::-1], dI['tf'][::-1], colorList[::-1]):
            step_indices = df[(df['Time (s)'] >= ti) & (df['Time (s)'] <= tf)].index
            ax.plot(df.loc[step_indices, 'Z Tip (nm)']*1e-3, df.loc[step_indices, 'Load (uN)']*1e6*1e-3, color=c,
                    marker = '.', markersize = 2, ls = '')
        
        ax.set_xlabel('Distance (um)')
        ax.set_ylabel('Load (nN)')
        ax.axvline(Z[i_start], ls = ':', color = 'k', lw=1, zorder = 4)
        # ax.axvline(Z[i_stop], ls = '-.', color = 'k', lw=1, zorder = 4)
        ax.axhline(F_moy*1e-3, ls = '-', color = 'k', zorder = 4)
        ax.axhline((F_moy+F_std)*1e-3, ls = '--', color = 'k', zorder = 4)
        # ax.legend(loc='upper left')
        
        #### axes[2]
        ax = axes[2] # F in nN here : µN*1e6*1e-3 // pN*1e-3
        ax.plot(Z0, F0, color='b',
                marker = '.', markersize = 2, ls = '')
        ax.plot(Z1, F1, color='r',
                marker = '.', markersize = 2, ls = '')
        
        ax.set_xlabel('Distance (um)')
        ax.set_ylabel('Load (nN)')
        ax.axvline(Z[i_start], ls = ':', color = 'k', lw=1, zorder = 4)
        # ax.axvline(Z[i_stop], ls = '-.', color = 'k', lw=1, zorder = 4)
        ax.axhline(F_moy*1e-3, ls = '-', color = 'k', zorder = 4)
        ax.axhline((F_moy+F_std)*1e-3, ls = '--', color = 'k', zorder = 4)
        
        ax.plot([z1, z3], [a0*z1+b0, a0*z3+b0], 'g--', lw=1)
        ax.plot([z2, z3], [a1*z2+b1, a1*z3+b1], 'g--', lw=1)
        ax.plot([Zc], [a0*Zc+b0], 'co', mec='k', ms=5, label=f'Zc={Zc:.2f}')
        ax.legend(loc='upper left', fontsize = 12)
        
        fig.suptitle('Contact point computation')
        
        fig.tight_layout()
        plt.show()
        
        if save_plot:
            name = f"{dI['date']}_{dI['manip']}_{dI['cell']}_{dI['indent']}_GlassIndent"
            # ufun.simpleSaveFig(fig, name, save_path, '.png', 150)
            ufun.archiveFig(fig, name = name, ext = '.png', dpi = 200,
                            figDir = save_path, figSubDir = '', cloudSave = 'flexible')
            ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                            figDir = save_path, figSubDir = '', cloudSave = 'flexible')
        
    return(Zc)



# %% Test Functions in dev

# %%% findHardSurface

date, manip, cell, indent = '24-04-11', 'M3', 'C1-glass', '001'
date2 = '24.04.11'
specif1 = '_25um' # indenter size
specif2 = '' # height above substrate

mainDir = f'D://MagneticPincherData//Raw//{date2}_NanoIndent_ChiaroData{specif1}'
manipDir = os.path.join(mainDir, f'{manip}')
cellID = f'{date}_{manip}_P1_{cell}'
indentID = f'{date}_{manip}_{cell}_I{int(indent)}'
indent_path = os.path.join(manipDir, f'{cell}//Indentations//{cell}{specif2} Indentation_{indent}.txt')
figures_path = manipDir

dI = initializeFitHertz(indent_path, date, manip, cell, indent)
# Zc = findHardSurface(dI, plot = False)