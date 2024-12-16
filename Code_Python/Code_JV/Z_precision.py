# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:31:39 2023

@author: JosephVermeil
"""

# %% (0) Imports and settings

# 1. Imports
import numpy as np
import pandas as pd
import skimage as skm
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

import os
import time
import pyautogui
import matplotlib

from scipy import interpolate

# from skimage import io, transform
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import linear_sum_assignment


# 2. Local Imports
import GraphicStyles as gs
import UtilityFunctions as ufun


# %% (1) Classes

class Depthograph:
    def __init__(self, depthoPath, scale):
        
        self.scale = scale
        
        self.HDZfactor = 5
        self.maxDz_triplets = 60 # Max Dz allowed between images
        self.maxDz_singlets = 30
        self.HWScan_triplets = 1200 # Half width of the scans
        self.HWScan_singlets = 600

        deptho_raw = skm.io.imread(depthoPath)
        deptho_root = '.'.join(depthoPath.split('.')[:-1])
        deptho_fileName = (depthoPath.split('//')[-1])
        depthoMetadata = pd.read_csv('_'.join(depthoPath.split('_')[:-1]) + '_Metadata.csv', sep=';')
        depthoStep = depthoMetadata.loc[0, 'step']
        try:
            depthoZFocus = depthoMetadata.loc[0, 'bestFocus']
        except:
            depthoZFocus = depthoMetadata.loc[0, 'focus']
        
        self.path = depthoPath
        self.fileName = deptho_fileName
        self.deptho_raw = deptho_raw
        self.step_raw = depthoStep  
        
        self.cutBlackParts(kind = "raw")
        
        nX, nZ = self.deptho_raw.shape[1], self.deptho_raw.shape[0]
        XX, ZZ = np.arange(0, nX, 1), np.arange(0, nZ, 1)
        ZZ_HD = np.arange(0, nZ, 1/self.HDZfactor)
        # fd = interpolate.interp2d(XX, ZZ, self.deptho_raw, kind='cubic')
        # self.deptho_hd = fd(XX, ZZ_HD)
        deptho_hd = ufun.resize_2Dinterp(self.deptho_raw, fx=1, fy=self.HDZfactor)
        self.deptho_hd = deptho_hd
        self.step_hd = self.step_raw/self.HDZfactor
        
        self.deptho_hd_gb = skm.filters.gaussian(deptho_hd, sigma=(4,0))
        
        self.cutBlackParts(kind = "hd")

        self.ZFocus_raw = 0
        self.ZFocus_hd = 0
        self.ZFocus_hd_gb = 0
        self.computeFocus()
        
        self.XX = XX - len(XX)//2
        self.ZZ = (ZZ - ZZ[int(self.ZFocus_raw)])*self.step_raw 
        self.ZZ_HD = (ZZ_HD - ZZ_HD[int(self.ZFocus_hd)])*self.step_raw
        
        # print(self.XX)
        # print(self.ZZ)
        # print(self.ZZ_HD)
        
    def cutBlackParts(self, kind = 'raw'):
        if kind == 'raw':
            nz, nx = self.deptho_raw.shape
            validLine = np.ones(nz, dtype = bool)
            for z in range(nz):
                if np.sum(self.deptho_raw[z, :]) == 0:
                    validLine[z] = False
            self.deptho_raw = self.deptho_raw[validLine]
        elif kind == 'hd':
            nz, nx = self.deptho_hd.shape
            validLine = np.ones(nz, dtype = bool)
            for z in range(nz):
                if np.sum(self.deptho_hd[z, :]) <= 2000:
                    validLine[z] = False
            self.deptho_hd = self.deptho_hd[validLine]
    
    
    def computeFocus(self):
        # raw
        nz, nx = self.deptho_raw.shape
        STD = np.std(self.deptho_raw, axis = 1)
        STD_smooth = savgol_filter(STD, 101, 5)
        self.ZFocus_raw = np.argmax(STD_smooth)
        
        # Z = np.arange(nz)
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(Z, STD)
        # ax.plot(Z, STD_smooth)
        # ax.axvline(ZFocus_raw)
        
        # hd
        nz, nx = self.deptho_hd.shape
        STD = np.std(self.deptho_hd, axis = 1)
        STD_smooth = savgol_filter(STD, 505, 5)
        self.ZFocus_hd = np.argmax(STD_smooth)
        
        # Z = np.arange(nz)
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(Z, STD)
        # ax.plot(Z, STD_smooth)
        # ax.axvline(self.ZFocus_hd)
        # plt.show()
        
        # hd_gb
        nz, nx = self.deptho_hd_gb.shape
        STD = np.std(self.deptho_hd_gb, axis = 1)
        # STD_smooth = savgol_filter(STD, 505, 5)
        self.ZFocus_hd_gb = np.argmax(STD)
        
        # Z = np.arange(nz)
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(Z, STD)
        # # ax.plot(Z, STD_smooth)
        # ax.axvline(self.ZFocus_hd_gb)
        # plt.show()
        
        
    def getNlines(self, Z, N, dZ, blur = True):
        listZ = [Z + (k-N//2)*dZ for k in range(N)]
        listProfiles = []
        
        # fig, ax = plt.subplots(1, 1)
               
        for z in listZ:
            A = (self.ZZ_HD >= z)
            z_idx = ufun.findFirst(1, A)
            if blur == True:
                profile = self.deptho_hd_gb[z_idx, :]
            else:
                profile = self.deptho_hd[z_idx, :]
            # ax.plot(self.XX, profile)
            listProfiles.append(profile)
            
        return(np.array(listProfiles))
        
    
        
    def locate(self, Deptho2, Z1, N, dZ, PLOT = False):
        listProfiles = self.getNlines(Z1, N, dZ, blur = True)
        
        # now use listStatus_1, listProfiles, self.deptho + data about the jump between Nuplets ! (TBA)
        # to compute the correlation function
        nVoxels = int(np.round(int(dZ)/self.step_hd))
        # print(nVoxels)
        
        # fig, ax = plt.subplots(1, 1)
        # fig1, ax1 = plt.subplots(1, 1)

        Ztop = 0
        Zbot = len(Deptho2.ZZ_HD)
        scannedDepth = Zbot-Ztop
        
        # print(N, Ztop, Zbot, scannedDepth)
        
        listDistances = np.zeros((N, scannedDepth))
        listZ = np.zeros(N, dtype = int)
        
        subDeptho2 = Deptho2.deptho_hd_gb[Ztop:Zbot,:]
        # subDeptho2 = skm.filters.gaussian(subDeptho2, sigma=(4,0))
        # ax1.imshow(subDeptho2)
        
        for i in range(N):
            listDistances[i] = ufun.squareDistance(subDeptho2, listProfiles[i], normalize = True) # Utility functions
            listZ[i] = Ztop + np.argmin(listDistances[i])
            # ax.plot(Deptho2.ZZ_HD, listDistances[i])
            
        listStatus_1 =  [i+1 for i in range(N)]
        finalDists = ufun.matchDists(listDistances, listStatus_1, N, 
                                    nVoxels, direction = 'downward')

        sumFinalD = np.sum(finalDists, axis = 0)
        sumFinalD = savgol_filter(sumFinalD, 31, 3, mode='mirror')
        
        # ax.plot(Deptho2.ZZ_HD, sumFinalD)

        iZ2 = np.argmin(sumFinalD)
        Z2 = Deptho2.ZZ_HD[iZ2]
        # print(min(Deptho2.ZZ_HD), max(Deptho2.ZZ_HD))
        
        if PLOT:
            listZ = [Z1 + (k-N//2)*dZ for k in range(N)]
            listZidx = []
            for z in listZ:
                A = (self.ZZ_HD >= z)
                z_idx = ufun.findFirst(1, A)
                listZidx.append(z_idx)
            iZ1 = listZidx[N//2]
            
            gs.set_smallText_options_jv()
            fig = plt.figure(figsize=(16, 8))
            spec = fig.add_gridspec(N, 5)
            
            ax = fig.add_subplot(spec[:, 0])
            ax.imshow(self.deptho_hd, aspect = 'auto',
                      extent=[min(self.XX), max(self.XX), max(self.ZZ_HD), min(self.ZZ_HD)])               
            ax.axhline(self.ZZ_HD[self.ZFocus_hd_gb], c='r', ls='--')
            ax.axhline(Z1, c='g', ls='--', zorder=6)
            for z in listZ:
                ax.axhline(z, ls='--')
            
            for k in range(N):
                ax = fig.add_subplot(spec[k, 1])
                ax.plot(self.XX, listProfiles[k])
                
            ax = fig.add_subplot(spec[:, 2])
            ax.invert_yaxis()
            for k in range(N):
                ax.plot(listDistances[k], Deptho2.ZZ_HD)
            
            
            ax = fig.add_subplot(spec[:, 3])
            ax.invert_yaxis()
            for k in range(N):
                ax.plot(finalDists[k], Deptho2.ZZ_HD, lw=1)
            ax.plot(sumFinalD/N, Deptho2.ZZ_HD, c='gold', lw=2)
            ax.axhline(Z2, c='gold')
            
            ax = fig.add_subplot(spec[:, 4])
            ax.imshow(Deptho2.deptho_hd, aspect = 'auto',
                      extent=[min(self.XX), max(self.XX), max(self.ZZ_HD), min(self.ZZ_HD)])           
            ax.axhline(Z2, c='gold', ls='--')
            ax.axhline(Z1, c='g', ls='--')
            ax.axhline(Deptho2.ZZ_HD[Deptho2.ZFocus_hd_gb], c='r', ls='--')
            
            
            fig.tight_layout()
            fig.show()
        
        return(Z2)
    
    
    
    def absoluteErrorCurve(self, Deptho2, Zmin, Zmax, N, dZ):
        Z = np.arange(Zmin, Zmax, 10)
        listErr = []
        
        for z in Z:
            PLOT = False
            if z == +1000:
                PLOT = True
            errZ = z - self.locate(Deptho2, z, N, dZ, PLOT)
            listErr.append(errZ)
        arrayErr = np.array(listErr)    
        print(np.std(arrayErr))
        
        fig, ax = plt.subplots(1, 1)
        ax.plot(Z, arrayErr)
        
        
    def averageErrorCurve(self, listDepthos, Zmin, Zmax, N, dZ):
        gs.set_mediumText_options_jv()
        Z = np.arange(Zmin, Zmax, 50)
        nZ = len(Z)
        nD = len(listDepthos)
        matErr = np.zeros((nD, nZ))
        for iD in range(nD):
            Deptho2 = listDepthos[iD]
            for iZ in range(nZ):
                z = Z[iZ]
                PLOT = False
                # if z==-500:
                #     PLOT = True
                errZ = z - self.locate(Deptho2, z, N, dZ, PLOT)
                matErr[iD, iZ] = errZ
        
        avgErr = np.mean(np.abs(matErr), axis=0)
        Q3 = np.percentile(matErr, 75)
        Q1 = np.percentile(matErr, 25)
        print(Q1, Q3, Q3-Q1)
        
        # fig0, axes0 = plt.subplots(1, nD, figsize=(20, 4), sharey=True)
        # for iD in range(nD):
        #     ax=axes0[iD]
        #     ax.plot(Z, matErr[iD, :])
        #     plt.show()
        
        fig, ax = plt.subplots(1, 1, figsize=(8/gs.cm_in, 6/gs.cm_in))
        ax.plot(Z, avgErr, ls='-', c='cyan')
        # ax.axhline(Q1, ls='--')
        # ax.axhline(Q3, ls='--', label = f'interquatile - {Q3-Q1:.1f}')
        # ax.legend()
        ax.set_ylim([0, 60])
        ax.set_xlabel('$Z$ [position in the depthograph] (nm)')
        tickloc = matplotlib.ticker.MultipleLocator(500)
        ax.xaxis.set_major_locator(tickloc)
        ax.set_ylabel('Absolute error on $Z$ (nm)')
        ax.grid(visible=True, which='major', axis='y')
        fig.tight_layout()
        # ufun.archiveFig(fig, name = 'Z_AbsErr', ext = '.pdf', dpi = 150,
        #                figDir = 'D:/MagneticPincherData/Figures/PhysicsDataset/', 
        #                figSubDir = 'Zprecision', cloudSave = 'flexible')
        plt.show()
        
        
    
    def relativeErrorMap(self, Deptho2, Zmin, Zmax, N, dZ):
        gs.set_mediumText_options_jv()
        Z = np.arange(Zmin, Zmax, 50)
        nZ = len(Z)
        matrixErr = np.zeros((nZ, nZ))
        for i1 in range(nZ):
            for i2 in range(nZ):
                z1, z2 = Z[i1], Z[i2]
                errZ = (z1 - z2) - (self.locate(Deptho2, z1, N, dZ) - self.locate(Deptho2, z2, N, dZ))
                matrixErr[i1, i2] = errZ
            
        print(np.std(matrixErr))
        
        fig, ax = plt.subplots(1, 1, figsize=(7.5/gs.cm_in, 6/gs.cm_in))
        im = ax.pcolor(Z, Z, matrixErr, cmap = 'RdBu_r')
        cbar = fig.colorbar(im, ax=ax, label=r'Error on $\Delta Z$')
        ax.set_xlabel('$Z_1$ (nm)')
        ax.set_ylabel('$Z_2$ (nm)')
        # fig.tight_layout()
        # ufun.archiveFig(fig, name = 'Z_RelErr', ext = '.pdf', dpi = 150,
        #                figDir = 'D:/MagneticPincherData/Figures/PhysicsDataset/', 
        #                figSubDir = 'Zprecision', cloudSave = 'flexible')
        plt.show()
        

    


class ZScan:
    def __init__(self, zscanPath, scale):
        
        self.scale = scale

        ZScan_raw = skm.io.imread(zscanPath)
        ZScan_root = '.'.join(zscanPath.split('.')[:-1])
        ZScan_fileName = (zscanPath.split('//')[-1])
        resDf = pd.read_csv(ZScan_root + '_Results.txt', sep='\t', header=0)
        
        nz, ny, nx = ZScan_raw.shape[0], ZScan_raw.shape[1], ZScan_raw.shape[2]

        self.ZScan_raw = ZScan_raw
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.scale = scale
        
        
        X0 = resDf['XM'].values
        Y0 = resDf['YM'].values
        S0 = resDf['Slice'].values
        bestZ = S0[np.argmax(resDf['StdDev'].values)] - 1 # The index of the image with the highest Std
        
        self.X0 = X0
        self.Y0 = Y0
        self.S0 = S0
        self.XYm = np.zeros((self.nz, 2))
        self.XYm[S0-1, 0] = X0
        self.XYm[S0-1, 1] = Y0
        self.fileName = ZScan_fileName

        self.beadType = 'M450'
        self.D0 = 4.5
        
        self.ZScan_cleanROI = np.array([])
        self.validBead = True
        self.iValid = -1
        self.bestZ = bestZ
        self.validSlice = np.zeros(nz, dtype = bool)
        self.zFirst = 0
        self.zLast = nz
        self.validDepth = nz
                
        self.valid_v = True
        self.valid_h = True
        # self.depthosDict = {}
        # self.profileDict = {}
        # self.ZfocusDict = {}
        
        self.buildCleanROI()
        
        # io.imsave(ZScan_root + '_CleanROI.tif', self.ZScan_cleanROI)



    def buildCleanROI(self):
        plot = 0
        # Determine if the bead is to close to the edge on the max frame
        D0 = self.D0
        roughSize = np.floor(1.2*D0*self.scale)
        mx, Mx = np.min(self.X0 - 0.5*roughSize), np.max(self.X0 + 0.5*roughSize)
        my, My = np.min(self.Y0 - 0.5*roughSize), np.max(self.Y0 + 0.5*roughSize)
        testImageSize = mx > 0 and Mx < self.nx and my > 0 and My < self.ny

        # Aggregate the different validity test (for now only 1)
        validBead = testImageSize

        if validBead:
            for z in range(self.bestZ, -1, -1):
                if not z in self.S0:
                    break
            zFirst = z
            for z in range(self.bestZ, self.nz, +1):
                if not z in self.S0:
                    break
            zLast = z-1


            roughSize = int(np.floor(1.15*self.D0*self.scale))
            roughSize += 1 + roughSize%2
            roughCenter = int((roughSize+1)//2)

            cleanSize = ufun.getDepthoCleanSize(self.D0, self.scale)

            ZScan_cleanROI = np.zeros([self.nz, cleanSize, cleanSize])

            # try:
            for i in range(zFirst, zLast):
                xmi, ymi = self.XYm[i,0], self.XYm[i,1]
                x1, y1, x2, y2, validBead = ufun.getROI(roughSize, xmi, ymi, self.nx, self.ny)
                if not validBead:
                    if x1 < 0 or x2 > self.nx:
                        self.valid_h = False
                    if y1 < 0 or y2 > self.ny:
                        self.valid_v = False

    #                 fig, ax = plt.subplots(1,2)
    #                 ax[0].imshow(self.I[i])
                xm1, ym1 = xmi-x1, ymi-y1
                I_roughRoi = self.ZScan_raw[i,y1:y2,x1:x2]
    #                 ax[1].imshow(I_roughRoi)
    #                 fig.show()

                translation = (xm1-roughCenter, ym1-roughCenter)

                tform = skm.transform.EuclideanTransform(rotation=0, \
                                                      translation = (xm1-roughCenter, ym1-roughCenter))

                I_tmp = skm.transform.warp(I_roughRoi, tform, order = 1, preserve_range = True)

                ZScan_cleanROI[i] = np.copy(I_tmp[roughCenter-cleanSize//2:roughCenter+cleanSize//2+1,\
                                              roughCenter-cleanSize//2:roughCenter+cleanSize//2+1])

            if not self.valid_v and not self.valid_h:
                self.validBead = False

            else:
                self.zFirst = zFirst
                self.zLast = zLast
                self.validDepth = zLast-zFirst
                self.ZScan_cleanROI = ZScan_cleanROI.astype(np.uint16)
                
            Z = np.arange(self.ZScan_cleanROI.shape[0])
            STD = np.std(self.ZScan_cleanROI, axis = (1,2))

            fig, ax = plt.subplots(1, 1)
            ax.plot(Z, STD, 'r-')
            
            i_maxStd = np.argmax(STD)
            ni = 20
            p = np.polyfit(Z[i_maxStd-ni:i_maxStd+ni], 
                            STD[i_maxStd-ni:i_maxStd+ni], 
                            2)
            fit_squared = np.polyval(p, Z[i_maxStd-ni:i_maxStd+ni])
            
            ax.plot(Z[i_maxStd-ni:i_maxStd+ni], fit_squared, 'k--')
            print(Z[i_maxStd-ni:i_maxStd+ni][np.argmax(fit_squared)])
            ax.axvline(Z[i_maxStd-ni:i_maxStd+ni][np.argmax(fit_squared)], ls = '--', c = 'c')
            
            plt.show()


            # except:
            #     print('Error for the file: ' + self.fileName)
        
        
# %% TESTS

# %%% Deptho - 23.09.06
scale = 15.8
depthoLibrary = 'D:/MagneticPincherData/Raw/DepthoLibrary'
mainDeptho = '23.09.06_M4_M450_step20_100X'
mainDepthoPath = os.path.join(depthoLibrary, mainDeptho + '_Deptho.tif')
D_main = Depthograph(mainDepthoPath, scale)

intermediateDepthoDir = os.path.join(depthoLibrary, 'Intermediate_Py', mainDeptho + '_step20')
files = os.listdir(intermediateDepthoDir)
deptho2_files = []
deptho2_paths = []
deptho2_list = []
for f in files:
    if f.endswith('_deptho.tif'):
        path = os.path.join(intermediateDepthoDir, f)
        deptho2_files.append(f)
        deptho2_paths.append(path)
        deptho2_list.append(Depthograph(path, scale))

D_main.averageErrorCurve(deptho2_list, -1500, +1500, 3, 500)

# %%% Deptho - 23.04.20
scale = 15.8
depthoLibrary = 'D:/MagneticPincherData/Raw/DepthoLibrary'
mainDeptho = '23.04.20_M3_M450_step20_100X'
mainDepthoPath = os.path.join(depthoLibrary, mainDeptho + '_Deptho.tif')
D_main = Depthograph(mainDepthoPath, scale)

intermediateDepthoDir = os.path.join(depthoLibrary, 'Intermediate_Py', mainDeptho + '_step20')
files = os.listdir(intermediateDepthoDir)
deptho2_files = []
deptho2_paths = []
deptho2_list = []
for f in files:
    if f.endswith('_deptho.tif'):
        path = os.path.join(intermediateDepthoDir, f)
        deptho2_files.append(f)
        deptho2_paths.append(path)
        deptho2_list.append(Depthograph(path, scale))

D_main.averageErrorCurve(deptho2_list, -1500, +1500, 3, 500)

# %%% Deptho - 24.04.11 - M5
scale = 15.8
depthoLibrary = 'D:/MagneticPincherData/Raw/DepthoLibrary'
mainDeptho = '24.04.11_M5_M450_step20_100X'
mainDepthoPath = os.path.join(depthoLibrary, mainDeptho + '_Deptho.tif')
D_main = Depthograph(mainDepthoPath, scale)

intermediateDepthoDir = os.path.join(depthoLibrary, 'Intermediate_Py', mainDeptho + '_step20')
files = os.listdir(intermediateDepthoDir)
deptho2_files = []
deptho2_paths = []
deptho2_list = []
for f in files:
    if f.endswith('_deptho.tif'):
        path = os.path.join(intermediateDepthoDir, f)
        deptho2_files.append(f)
        deptho2_paths.append(path)
        deptho2_list.append(Depthograph(path, scale))

D_main.averageErrorCurve(deptho2_list, -1500, +1500, 3, 500)
        
# %%% Deptho - 24.02.28 - M1
scale = 15.8
depthoLibrary = 'D:/MagneticPincherData/Raw/DepthoLibrary'
mainDeptho = '24.02.28_M2_M450_step20_100X'
mainDepthoPath = os.path.join(depthoLibrary, mainDeptho + '_Deptho.tif')
D_main = Depthograph(mainDepthoPath, scale)

intermediateDepthoDir = os.path.join(depthoLibrary, 'Intermediate_Py', mainDeptho + '_step20')
files = os.listdir(intermediateDepthoDir)
deptho2_files = []
deptho2_paths = []
deptho2_list = []
for f in files:
    if f.endswith('_deptho.tif'):
        path = os.path.join(intermediateDepthoDir, f)
        deptho2_files.append(f)
        deptho2_paths.append(path)
        deptho2_list.append(Depthograph(path, scale))

D_main.averageErrorCurve(deptho2_list, -1500, +1500, 3, 500)

# %%% Deptho - 24.04.11 - M1 -> VERY GOOD ONE !!!
scale = 15.8
depthoLibrary = 'D:/MagneticPincherData/Raw/DepthoLibrary'
mainDeptho = '24.04.11_M1_M450_step20_100X'
mainDepthoPath = os.path.join(depthoLibrary, mainDeptho + '_Deptho.tif')
D_main = Depthograph(mainDepthoPath, scale)

intermediateDepthoDir = os.path.join(depthoLibrary, 'Intermediate_Py', mainDeptho + '_step20')
files = os.listdir(intermediateDepthoDir)
deptho2_files = []
deptho2_paths = []
deptho2_list = []
for f in files:
    if f.endswith('_deptho.tif'):
        path = os.path.join(intermediateDepthoDir, f)
        deptho2_files.append(f)
        deptho2_paths.append(path)
        deptho2_list.append(Depthograph(path, scale))

D_main.averageErrorCurve(deptho2_list, -1500, +1500, 3, 500)
D_main.relativeErrorMap(deptho2_list[0], -1000, +1050, 3, 500)

# %%% Deptho

depthoPath0 = 'D://MagneticPincherData//Raw//DepthoLibrary//Intermediate_Py//23.04.26_M1_M450_step20_100X_step20//db0_deptho.tif'
depthoPath1 = 'D://MagneticPincherData//Raw//DepthoLibrary//Intermediate_Py//23.04.26_M1_M450_step20_100X_step20//db0-1_deptho.tif'
depthoPath2 = 'D://MagneticPincherData//Raw//DepthoLibrary//Intermediate_Py//23.04.26_M1_M450_step20_100X_step20//db0-2_deptho.tif'
# depthoPath3 = 'D://MagneticPincherData//Raw//DepthoLibrary//Intermediate_Py//23.04.26_M1_M450_step20_100X_step20//db0-1_deptho.tif'

D0 = Depthograph(depthoPath0, 15.8)
D1 = Depthograph(depthoPath1, 15.8)
D2 = Depthograph(depthoPath2, 15.8)

# D0.absoluteErrorCurve(D2, -1500, +1500, 3, 500)
D0.absoluteErrorCurve(D1, -1500, +1500, 3, 500)

# D0.absoluteErrorCurve(D2, -1500, +1500, 1, 500)

# %%% Deptho

depthoPath10 = 'C://Users//JosephVermeil//Desktop//confocal calibration//spinning//ZStacks//CleanDeptho//Intermediate_Py//DepthoSpinning_Clean.tif//Experiment-286-2_deptho.tif'
depthoPath11 = 'C://Users//JosephVermeil//Desktop//confocal calibration//spinning//ZStacks//CleanDeptho//Intermediate_Py//DepthoSpinning_Clean.tif//Experiment-286-3_deptho.tif'
depthoPath12 = 'C://Users//JosephVermeil//Desktop//confocal calibration//spinning//ZStacks//CleanDeptho//Intermediate_Py//DepthoSpinning_Clean.tif//Experiment-287-1_deptho.tif'
depthoPath13 = 'C://Users//JosephVermeil//Desktop//confocal calibration//spinning//ZStacks//CleanDeptho//Intermediate_Py//DepthoSpinning_Clean.tif//Experiment-287-2_deptho.tif'
depthoPath20 = 'C://Users//JosephVermeil//Desktop//confocal calibration//spinning//ZStacks//FloatyDeptho//Intermediate_Py//DepthoSpinning_Floaty.tif//Experiment-286-1_deptho.tif'

D10 = Depthograph(depthoPath10, 7.5)
D11 = Depthograph(depthoPath11, 7.5)
D12 = Depthograph(depthoPath12, 7.5)
D13 = Depthograph(depthoPath13, 7.5)
D20 = Depthograph(depthoPath20, 7.5)

# D10.locate(D20, 0, 3, 500)
# D12.locate(D13, -800, 3, 500)
# D10.absoluteErrorCurve(D10, -1500, +1500, 3, 500)
D10.absoluteErrorCurve(D11, -1500, +1500, 3, 500)
D10.absoluteErrorCurve(D12, -1500, +1500, 3, 500)
D10.absoluteErrorCurve(D13, -1500, +1500, 3, 500)

D11.absoluteErrorCurve(D12, -1500, +1500, 3, 500)
D11.absoluteErrorCurve(D13, -1500, +1500, 3, 500)

D12.absoluteErrorCurve(D13, -1500, +1500, 3, 500)

D12.absoluteErrorCurve(D20, -1500, +1500, 3, 500)

# %%% Deptho relative error

depthoPath10 = 'C://Users//JosephVermeil//Desktop//confocal calibration//spinning//ZStacks//CleanDeptho//Intermediate_Py//DepthoSpinning_Clean.tif//Experiment-286-2_deptho.tif'
depthoPath11 = 'C://Users//JosephVermeil//Desktop//confocal calibration//spinning//ZStacks//CleanDeptho//Intermediate_Py//DepthoSpinning_Clean.tif//Experiment-286-3_deptho.tif'
depthoPath12 = 'C://Users//JosephVermeil//Desktop//confocal calibration//spinning//ZStacks//CleanDeptho//Intermediate_Py//DepthoSpinning_Clean.tif//Experiment-287-1_deptho.tif'
depthoPath13 = 'C://Users//JosephVermeil//Desktop//confocal calibration//spinning//ZStacks//CleanDeptho//Intermediate_Py//DepthoSpinning_Clean.tif//Experiment-287-2_deptho.tif'
depthoPath20 = 'C://Users//JosephVermeil//Desktop//confocal calibration//spinning//ZStacks//FloatyDeptho//Intermediate_Py//DepthoSpinning_Floaty.tif//Experiment-286-1_deptho.tif'

D10 = Depthograph(depthoPath10, 7.5)
D11 = Depthograph(depthoPath11, 7.5)
D12 = Depthograph(depthoPath12, 7.5)
D13 = Depthograph(depthoPath13, 7.5)
D20 = Depthograph(depthoPath20, 7.5)


D10.relativeErrorMap(D11, -1000, +1000, 3, 500)

# %%% Zscan

# zscanPath = 'D://MagneticPincherData//Raw//23.04.28_Deptho//M3//db3.tif'
# zscanPath = 'C://Users//JosephVermeil//Desktop//confocal calibration//spinning//ZStacks//CleanBeads//Experiment-285-1.tif'
zscanPath = 'C://Users//JosephVermeil//Desktop//confocal calibration//spinning//ZStacks//FloatyBeads//Experiment-285-2.tif'

ZS = ZScan(zscanPath, 7.5)


        
