# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:14:30 2022

@author: anumi

Radial profile definitions taken from: https://github.com/keflavich/image_tools/blob/master/image_tools/radialprofile.py
"""

import os
import glob
import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
from contextlib import suppress
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


import GraphicStyles as gs
import UtilityFunctions as ufun


#%% Required functions

def createFluoDf(dirSave, dirSaveAnalysed, currentCell, fluoChannel, timeRes = 10, endOffset = 5, \
                 integInterval = 5, innerMeanInt = 8, chosenAngle = 0, plot = True):
    
    csvFile = dirSave + '/' + currentCell + '_' + fluoChannel + '_fluoAnalysisRaw.csv'
    fluoDf = pd.read_csv(csvFile)
    df = fluoDf[fluoDf['angle'] == chosenAngle]
    
    fluoQuant = {'frame':[], 'mean_r':[], 'mean_l':[], 'maxval_r':[], 'maxval_l':[], 'intval_r':[], 'intval_l':[] }
    frames = []
    maxvals_r = []
    maxvals_l = []
    intvals_r = []
    intvals_l = []
    means_r = []
    means_l = []
    
    allFrames = np.unique(fluoDf['frame'])
    
    for frame in allFrames:
        dfFrame = df[df['frame'] == frame]
        lastIndex = dfFrame.index[-1]
        lastBin = dfFrame['bincenters_right'].iloc[-1*endOffset]
        
        dfFrame = dfFrame[(dfFrame['bincenters_right'] < lastBin)]
        
        max_ind_r =  dfFrame['values_right'].idxmax()        
        max_ind_l =  dfFrame['values_left'].idxmax()
        
        frames.append(frame)
        
        # if max_ind_r < lastIndex - innerMeanInt :
        try:
            
            ymin_ind_r = max_ind_r - integInterval
            ymax_ind_r = max_ind_r + integInterval
            mean_ind_r = dfFrame['bincenters_right'][max_ind_r + innerMeanInt]
            meanr = dfFrame['values_right'][(dfFrame['bincenters_right'] > mean_ind_r)].mean()
    
            intval_r = (np.trapz(dfFrame['values_right'].loc[ymin_ind_r:ymax_ind_r].values,\
                                dfFrame['bincenters_right'].loc[ymin_ind_r:ymax_ind_r].values)- meanr) \
                /(dfFrame['bincenters_right'].loc[ymax_ind_r]-dfFrame['bincenters_right'].loc[ymin_ind_r])
            
            maxvals_r.append(np.nanmax(dfFrame['values_right'].values) - meanr)
            intvals_r.append(intval_r)
            means_r.append(meanr)
            
        # else:
        except:
            print(gs.ORANGE + 'Error in identifying max intensity value rightside')
            print('Frame: ' +str(frame))
            print('Last index:' + str(lastIndex - innerMeanInt))
            print('Max index right:' + str(max_ind_r) + gs.NORMAL)
            maxvals_r.append(np.nan)
            intvals_r.append(np.nan)
            means_r.append(np.nan)
        
        try:
            ymin_ind_l = max_ind_l - integInterval
            ymax_ind_l = max_ind_l + integInterval
            mean_ind_l = dfFrame['bincenters_left'][max_ind_l + innerMeanInt]
            meanl = dfFrame['values_left'][(dfFrame['bincenters_left'] > mean_ind_l)].mean()
    
            intval_l = (np.trapz(dfFrame['values_left'].loc[ymin_ind_l:ymax_ind_l].values,\
                                dfFrame['bincenters_left'].loc[ymin_ind_l:ymax_ind_l].values) - meanl) \
                 /(dfFrame['bincenters_left'].loc[ymax_ind_l]-dfFrame['bincenters_left'].loc[ymin_ind_l])
        
            maxvals_l.append(np.nanmax(dfFrame['values_left'].values)- meanl)
            intvals_l.append(intval_l)
            means_l.append(meanl)
            
            
        except:
            print(gs.ORANGE + 'Error in identifying max intensity value leftside')
            print('Frame: ' +str(frame))
            print('Last index:' + str(lastIndex - innerMeanInt))
            print('Max index left:' + str(max_ind_l) + gs.NORMAL)
            maxvals_l.append(np.nan)
            intvals_l.append(np.nan)
            means_l.append(np.nan)
    
        
    fluoQuant = {'frame' : frames,
                  'mean_r' : means_r,
                  'mean_l' : means_l,
                  'maxval_r' : maxvals_r,
                  'maxval_l' : maxvals_l,
                  'intval_r' : intvals_r,
                  'intval_l' : intvals_l}
    
    fluoQuantDf = pd.DataFrame(fluoQuant)
    fluoQuantDf.to_csv(dirSaveAnalysed + '/' + currentCell + '_' + fluoChannel + '_fluoQuant_angle-'+str(chosenAngle)+'.csv')
    
    
    if plot:
        df = fluoQuantDf
        
        fig2, axes = plt.subplots(2,1,figsize = (15,10))
        
        x = df['frame']*timeRes
        
        sns.lineplot(x = x, y = 'maxval_r', data = df, ax = axes[0], label = 'MaxValue_Right')
        sns.lineplot(x = x, y = 'maxval_l', data = df, ax = axes[0], label = 'MaxValue_Left')
        axes[0].legend(prop = {'size': 12})
        axes[0].set_xlabel('Time (secs)')
        axes[0].set_title('Maximum Peak Value vs Time (secs)')
        
        sns.lineplot(x = x, y = 'intval_r', data = df, ax = axes[1], label = 'IntValue_Right')
        sns.lineplot(x = x, y = 'intval_l', data = df, ax = axes[1], label = 'IntValue_Left')
        axes[1].set_title('Integrated Peak Value vs Time (secs)')
        axes[1].set_xlabel('Time (secs)')
        axes[1].legend(prop = {'size': 12})
        
        plt.tight_layout()
    
    return fluoQuantDf
    
def processFluoImages(dirExt, dirSave, currentCell, fluoChannel, frameLim = 'none', profBinSize = 2, imageThickness = 10, \
                      angleStart = 0, angleEnd = 360, nAngles = 73, countourAreaThres = 100, scale = 7.58, \
                          displayContourDetails = False, plotProfilePreview = False, showMask = False, displayProcessed = False, \
                          removeInnerContour = False, R_ic = 5, reprocess = False, saveCSV = True):

    saveFile = dirSave + '/' + currentCell + '_' + fluoChannel + '_fluoAnalysisRaw.csv'
    
    if os.path.exists(saveFile) and not reprocess:
        print(gs.GREEN + 'File already processesed' + gs.NORMAL)
        return
    elif os.path.exists(saveFile) and reprocess:
        print(gs.GREEN + 'Anlaysis file avialble. Overwriting...' + gs.NORMAL)
    
    if plotProfilePreview:
        fig, axes = plt.subplots(2,1,figsize = (15,10))
        
    fluodict = {'frame': [], 
                'angle' : [], 
                'bincenters_left' : [],
                'values_left' : [],
                'background_left' : [], 
                'bincenters_right' : [], 
                'values_right' : [], 
                'background_right' : []}

    values_l = []
    bins_l = []
    bins_r = []
    values_r = []
    bg_r = []
    bg_l = []
    frameno = []
    max_left = []
    max_right = []
    angles = []
    
    allAngles = np.linspace(angleStart, angleEnd, nAngles)
    
    cellPath = dirExt+'/'+currentCell
    if os.path.exists(cellPath):
        allFrames = os.listdir(cellPath)
        allFrames = [i for i in allFrames if 'thumb' not in i and fluoChannel in i]
        allFrames.sort(key=lambda x: int(x.split('_t')[1][:-4]))
        
        cY0 = 0
        cX0 = 0
        
        if frameLim == 'none':
            frameNum = len(allFrames)
        else:
            frameNum = frameLim
    
        for i in range(0, frameNum):
            print(gs.YELLOW + 'Frame : ' + str(i) + gs.NORMAL)
            for j in allAngles:
                #Read all frames in the right order
                frame = allFrames[i]
                img = cv.imread(cellPath+'/'+frame, -1)
                img = cv.normalize(img, dst=None, alpha=0, beta=65535, norm_type=cv.NORM_MINMAX)
                
                (h, w) = img.shape[:2]
                
                # Blurring and treatment to remove noise, accurate thresholds
                kernel = np.ones((5,5),np.uint8)
                blur = cv.medianBlur(img, 5)
                
                #Make an original, un-butchered copy of the visualisable frame
                img0 = blur.copy()
                
                
                ret, mask = cv.threshold(blur,0,65535,cv.THRESH_BINARY+cv.THRESH_OTSU)
                mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)))
                mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
                
                opening = cv.normalize(mask, dst=None, alpha=0, beta=65535, dtype = cv.CV_8UC1, norm_type=cv.NORM_MINMAX)
                contours, hierarchy = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                
                #Keeping only contours with max area to avoid catching small threshold errors
                cnt_area = [cv.contourArea(c) for c in contours]
                max_area = np.argmax(cnt_area)
                c = contours[max_area]
    
            
                #Finding the centre of cortex contour
                M = cv.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                cntimg = img.copy()
                if removeInnerContour:
                #converting XY contour coordinates to R, theta co-ordinates
                    cntimg = cv.drawContours(cntimg, c, -1, (0,0,255), 1)
                    c = c.transpose(0,2,1).reshape(len(c),-1)
                    r_px = np.asarray([np.sqrt((p[0] - cX)**2 + (p[1] - cY)**2) for p in c])
                    r_um = np.asarray([np.sqrt((p[0] - cX)**2 + (p[1] - cY)**2) for p in c])/scale
                    theta = np.asarray([np.arctan2(p[1] - cY,p[0] - cX) for p in c])
                    theta_deg = np.asarray([np.arctan2(p[1] - cY,p[0] - cX) for p in c])*57.2958
                
                    r_in = r_px - R_ic
                    rin_c = [r_in*np.cos(theta) + cX, r_in*np.sin(theta) + cY]
                    rin_c = np.transpose(rin_c, (1,0))
                    rin_c = np.asarray(np.reshape(rin_c, (1, len(c), 1, 2)), dtype = np.int32)
                    cntimg = cv.drawContours(cntimg, tuple(rin_c), -1, (0,0,0), thickness = cv.FILLED)
                
                if displayContourDetails:
                    cntimg = cv.drawContours(cntimg, c, -1, (0,0,255), 2)
                    cv.circle(cntimg, (cX0, cY0), 5, (255, 255, 255), -1) 
                    cv.putText(cntimg, "center", (cX - 20, cY - 20),\
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv.imshow('Contour details', cntimg)
                
                if i == 0:
                    cY0 = cY
                    cX0 = cX
                
                #Saving background value
                if j == 0:
                    bgout_l = np.mean(img0[0:100, 0:100])
                    bgout_r = np.mean(img0[-100:, -100:])
                    
                M = cv.getRotationMatrix2D((cX0, cY0), j, 1.0)
                imgToProcess = cv.warpAffine(img0, M, (w, h))
                                   
                dyC = imageThickness
                y1c, y2c = cY0 - dyC, cY0 + dyC

                # dyI = int(75/2)
                # bgi = azimuthalAverage(imgToProcess[cY0-dyI:cY0+dyI, cX0-dyI:cX0+dyI], (cX0, cY0))
                
                img_right = (imgToProcess[y1c:y2c, cX0:])
                img_left = np.flip(imgToProcess[y1c:y2c, :cX0])
                
                with suppress(ZeroDivisionError):
                     bincen_r, prof_r = azimuthalAverage(img_right, (cX0, cY0), returnradii = True, binsize = profBinSize)
                     bincen_l, prof_l = azimuthalAverage(img_left, (cX0, cY0), returnradii = True, binsize = profBinSize)
                     
                radprof_r = prof_r - bgout_r
                radprof_l = prof_l - bgout_l
           
                max_left.append(np.nanmax(radprof_l))
                max_right.append(np.nanmax(radprof_r))
                values_l.extend(radprof_l)
                bins_l.extend(bincen_l)
                bins_r.extend(bincen_r)
                values_r.extend(radprof_r)
                bg_l.extend([bgout_l]*len(radprof_l))
                bg_r.extend([bgout_r]*len(radprof_r))
                frameno.extend([i]*len(radprof_r))
                angles.extend([j]*len(radprof_r))
                
                if plotProfilePreview and j == 0:
                    
                    axes[0].plot(bincen_r, radprof_r, label = i)
                    axes[1].plot(bincen_l, radprof_l, label = i)
                    axes[0].set_title('Right')
                    axes[1].set_title('Left')
                    axes[1].legend()
                    axes[0].legend()
                
            if displayProcessed:
                cv.imshow('Whole Image', imgToProcess)
                cv.imshow('Right', img_right)
                cv.imshow('Left', img_left)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                
            if showMask:
                cv.imshow('Mask', opening)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break


    fluodict = {'frame': frameno, 
                'angle': angles,
                'bincenters_left' : bins_l, 
                'values_left' : values_l,
                'background_left' : bg_l, 
                'bincenters_right' : bins_r, 
                'values_right' : values_r, 
                'background_right' : bg_r}
    
    fluoDf = pd.DataFrame(fluodict)
    
    if saveCSV:
        fluoDf.to_csv(saveFile)
    
    cv.destroyAllWindows()
    return fluoDf

def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False, 
        binsize= 2, weights=None, steps=False, interpnan=False, left=None, right=None,
        mask=None ):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    stddev - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and radial
        profile so you can plot a step-form radial profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values
    mask - can supply a mask (boolean array same size as image with True for OK and False for not)
        to average over only select data.
    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    if mask is None:
        mask = np.ones(image.shape,dtype='bool')
    # obsolete elif len(mask.shape) > 1:
    # obsolete     mask = mask.ravel()

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)  
    nbins = int(np.round(r.max() / binsize)+1)
    maxbin = nbins * binsize
    bins = np.linspace(0,maxbin,nbins+1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:]+bins[:-1])/2.0
    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    #nr = np.bincount(whichbin)[1:]
    nr = np.histogram(r, bins, weights=mask.astype('int'))[0]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or range(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape
    if stddev:
        # Find out which radial bin each point in the map belongs to
        whichbin = np.digitize(r.flat,bins)
        # This method is still very slow; is there a trick to do this with histograms? 
        radial_prof = np.array([image.flat[mask.flat*(whichbin==b)].std() for b in range(1,nbins+1)])
    else: 
        radial_prof = np.histogram(r, bins, weights=(image*weights*mask))[0] / np.histogram(r, bins, weights=(mask*weights))[0]

    if interpnan:
        radial_prof = np.interp(bin_centers,bin_centers[radial_prof==radial_prof],radial_prof[radial_prof==radial_prof],left=left,right=right)

    if steps:
        xarr = np.array(zip(bins[:-1],bins[1:])).ravel() 
        yarr = np.array(zip(radial_prof,radial_prof)).ravel() 
        return xarr,yarr
    elif returnradii: 
        return bin_centers,radial_prof
    elif return_nr:
        return nr,bin_centers,radial_prof
    else:
        return radial_prof

def radialAverage(image, center=None, stddev=False, returnAz=False, return_naz=False, 
        binsize=1.0, weights=None, steps=False, interpnan=False, left=None, right=None,
        mask=None, symmetric=None ):
    """
    Calculate the radially averaged azimuthal profile.
    (this code has not been optimized; it could be speed boosted by ~20x)
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    stddev - if specified, return the radial standard deviation instead of the average
    returnAz - if specified, return (azimuthArray,azimuthal_profile)
    return_naz   - if specified, return number of pixels per azimuth *and* azimuth
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and azimuthal
        profile so you can plot a step-form azimuthal profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values
    mask - can supply a mask (boolean array same size as image with True for OK and False for not)
        to average over only select data.
    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])
    theta = np.arctan2(x - center[0], y - center[1])
    theta[theta < 0] += 2*np.pi
    theta_deg = theta*180.0/np.pi
    maxangle = 360

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    if mask is None:
        # mask is only used in a flat context
        mask = np.ones(image.shape,dtype='bool').ravel()
    elif len(mask.shape) > 1:
        mask = mask.ravel()

    # allow for symmetries
    if symmetric == 2:
        theta_deg = theta_deg % 90
        maxangle = 90
    elif symmetric == 1:
        theta_deg = theta_deg % 180
        maxangle = 180

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)  
    nbins = int(np.round(maxangle / binsize))
    maxbin = nbins * binsize
    bins = np.linspace(0,maxbin,nbins+1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:]+bins[:-1])/2.0

    # Find out which azimuthal bin each point in the map belongs to
    whichbin = np.digitize(theta_deg.flat,bins)

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    nr = np.bincount(whichbin)[1:]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or range(1,nbins+1) )
    # azimuthal_prof.shape = bin_centers.shape
    if stddev:
        azimuthal_prof = np.array([image.flat[mask*(whichbin==b)].std() for b in range(1,nbins+1)])
    else:
        azimuthal_prof = np.array([(image*weights).flat[mask*(whichbin==b)].sum() / weights.flat[mask*(whichbin==b)].sum() for b in range(1,nbins+1)])

    #import pdb; pdb.set_trace()

    if interpnan:
        azimuthal_prof = np.interp(bin_centers,
            bin_centers[azimuthal_prof==azimuthal_prof],
            azimuthal_prof[azimuthal_prof==azimuthal_prof],
            left=left,right=right)

    if steps:
        xarr = np.array(zip(bins[:-1],bins[1:])).ravel() 
        yarr = np.array(zip(azimuthal_prof,azimuthal_prof)).ravel() 
        return xarr,yarr
    elif returnAz: 
        return bin_centers,azimuthal_prof
    elif return_naz:
        return nr,bin_centers,azimuthal_prof
    else:
        return azimuthal_prof

#%% Main body

dirFluo = 'D:/Anumita/MagneticPincherData/DataFluorescence/Raw'
dirExt = 'F:/Cortex Experiments/Fluorescence Experiments/20221015_3t3optorhoaLifeAct_40x_FluoTimeLapse'
dirSave = 'D:/Anumita/MagneticPincherData/DataFluorescence/FluoDataAnalysis'
dirSaveAnalysed = 'D:/Anumita/MagneticPincherData/DataFluorescence/FluoDataAnalysis/FluoDataProcessed/'

currentCell = '22-10-15_M2_P1_C9_disc20um'
fluoChannel = 'CSU640'

#%%

frameLim = 'none'
processFluoImages(dirExt, dirSave, currentCell, fluoChannel, frameLim, showMask = True, profBinSize = 3, displayProcessed = False, displayContourDetails=False, \
                  plotProfilePreview=True, reprocess = True)

#%%
createFluoDf(dirSave, dirSaveAnalysed, currentCell, fluoChannel, endOffset = 5, chosenAngle = 180)


#%% Plotting many fluoCells data in one plot

allCellsList = os.listdir(dirSaveAnalysed)
allCells = [i for i in allCellsList if i.endswith('.csv') and 'angle-5' in i]
timeRes = 10

lenSubplots = len(allCells)
rows= int(np.floor(np.sqrt(lenSubplots)))
cols= int(np.ceil(lenSubplots/rows))

fig1, axes1 = plt.subplots(rows, cols ,figsize = (15,10))
_axes1 = []

fig2, axes2 = plt.subplots(rows, cols ,figsize = (15,10))
_axes2 = []

for ax_array in axes1:
    for ax1 in ax_array:
        _axes1.append(ax1)

for ax_array in axes2:
    for ax2 in ax_array:
        _axes2.append(ax2)

for (cell, ax1, ax2) in zip(allCells, _axes1, _axes2):
    
    df = pd.read_csv(dirSaveAnalysed + '/' + cell)
    cellID = cell.split('_disc20um')[0]
    print(cellID)
    x = df['frame']*timeRes
    
    sns.lineplot(x = x, y = 'maxval_r', data = df, ax = ax1, label = 'MaxValue_Right')
    sns.lineplot(x = x, y = 'maxval_l', data = df, ax = ax1, label = 'MaxValue_Left')
    ax1.set_xlabel('Time (secs)')
    ax1.set_title(cellID)
    fig1.tight_layout()
    
    sns.lineplot(x = x, y = 'intval_r', data = df, ax = ax2, label = 'IntValue_Right')
    sns.lineplot(x = x, y = 'intval_l', data = df, ax = ax2, label = 'IntValue_Left')
    ax2.set_title(cellID)
    ax2.set_xlabel('Time (secs)')
    
    fig2.tight_layout()

#%% Inidividual frames or cells to check the boundaries of integrated intensities 

csvFile = dirSave + '/' + currentCell + '_' + fluoChannel + '_fluoAnalysisRaw.csv'
fluoDf = pd.read_csv(csvFile)

innerMeanInt = 8
integInterval = 5
endOffset = 5


dfprev = fluoDf[fluoDf['angle']== 5]
# dfprev = dfprev[dfprev['frame']== 0]

lastIndex = dfprev.index[-1]
lastBin = dfprev['bincenters_right'].iloc[-1*endOffset]

dfFrame = dfprev[(dfprev['bincenters_right'] < lastBin)]

max_ind_r =  dfFrame['values_right'].idxmax()        
max_ind_l =  dfFrame['values_left'].idxmax()


fig1, axes = plt.subplots(2,1,figsize = (15,10))

try:
    ymin_ind_r = max_ind_r - integInterval
    ymax_ind_r = max_ind_r + integInterval
    mean_ind_r = dfFrame['bincenters_right'][max_ind_r + innerMeanInt]
    meanr = dfFrame['values_right'][(dfFrame['bincenters_right'] > mean_ind_r)].mean()

    intval_r = (np.trapz(dfFrame['values_right'].loc[ymin_ind_r:ymax_ind_r].values,\
                        dfFrame['bincenters_right'].loc[ymin_ind_r:ymax_ind_r].values)) - meanr #\
        # /(dfFrame['bincenters_right'].loc[ymax_ind_r]-dfFrame['bincenters_right'].loc[ymin_ind_r]))

except:
    print(gs.ORANGE + 'Error in identifying max intensity value rightside')
    print('Last index:' + str(lastIndex - innerMeanInt))
    print('Max index right:' + str(max_ind_r) + gs.NORMAL)

try:
    ymin_ind_l = max_ind_l - integInterval
    ymax_ind_l = max_ind_l + integInterval
    mean_ind_l = dfFrame['bincenters_left'][max_ind_l + innerMeanInt]
    meanl = dfFrame['values_left'][(dfFrame['bincenters_left'] > mean_ind_l)].mean()

    intval_l = (np.trapz(dfFrame['values_left'].loc[ymin_ind_l:ymax_ind_l].values,\
                        dfFrame['bincenters_left'].loc[ymin_ind_l:ymax_ind_l].values)) - meanl
        # /(dfFrame['bincenters_left'].loc[ymax_ind_l]-dfFrame['bincenters_left'].loc[ymin_ind_l])) 

except:
    print(gs.ORANGE + 'Error in identifying max intensity value leftside')
    print('Last index:' + str(lastIndex - innerMeanInt))
    print('Max index left:' + str(max_ind_l) + gs.NORMAL)


sns.lineplot(x = 'bincenters_right', y = 'values_right', data = dfFrame, hue = 'frame', ax = axes[0])
sns.lineplot(x = 'bincenters_left', y = 'values_left', data = dfFrame, hue = 'frame', ax = axes[1])

# axes[0].axvline(x = dfprev['bincenters_right'][ymin_ind_r], color = 'red', linestyle = '--', label = 'Integrated region')
# axes[0].axvline(x = dfprev['bincenters_right'][ymax_ind_r], color = 'red', linestyle = '--')
# axes[0].axvline(x = mean_ind_r, color = 'black', label = 'Inner mean value range')
# axes[0].legend()

# axes[1].axvline(x = dfprev['bincenters_left'][ymin_ind_l], color = 'red', linestyle = '--', label = 'Integrated region')
# axes[1].axvline(x = dfprev['bincenters_left'][ymax_ind_l], color = 'red', linestyle = '--')
# axes[1].axvline(x = mean_ind_l, color = 'black', label = 'Inner mean value range')
# axes[1].legend()


#%% Testing out interpolation scipy

csvFile = dirSave + '/' + currentCell + '_' + fluoChannel + '_fluoAnalysisRaw.csv'
fluoDf = pd.read_csv(csvFile)

innerMeanInt = 8
integInterval = 5
endOffset = 5


dfprev = fluoDf[fluoDf['angle']== 5]
# dfprev = dfprev[dfprev['frame']== 10]

lastIndex = dfprev.index[-1]
lastBin = dfprev['bincenters_right'].iloc[-1*endOffset]

dfFrame = dfprev[(dfprev['bincenters_right'] < lastBin)]

max_ind_r =  dfFrame['values_right'].idxmax()        
max_ind_l =  dfFrame['values_left'].idxmax()


fig1, axes = plt.subplots(2,1,figsize = (15,10))

try:
    ymin_ind_r = max_ind_r - integInterval
    ymax_ind_r = max_ind_r + integInterval
    mean_ind_r = dfFrame['bincenters_right'][max_ind_r + innerMeanInt]
    meanr = dfFrame['values_right'][(dfFrame['bincenters_right'] > mean_ind_r)].mean()

    intval_r = (np.trapz(dfFrame['values_right'].loc[ymin_ind_r:ymax_ind_r].values,\
                        dfFrame['bincenters_right'].loc[ymin_ind_r:ymax_ind_r].values)) - meanr #\
        # /(dfFrame['bincenters_right'].loc[ymax_ind_r]-dfFrame['bincenters_right'].loc[ymin_ind_r]))

except:
    print(gs.ORANGE + 'Error in identifying max intensity value rightside')
    print('Last index:' + str(lastIndex - innerMeanInt))
    print('Max index right:' + str(max_ind_r) + gs.NORMAL)

try:
    ymin_ind_l = max_ind_l - integInterval
    ymax_ind_l = max_ind_l + integInterval
    mean_ind_l = dfFrame['bincenters_left'][max_ind_l + innerMeanInt]
    meanl = dfFrame['values_left'][(dfFrame['bincenters_left'] > mean_ind_l)].mean()

    intval_l = (np.trapz(dfFrame['values_left'].loc[ymin_ind_l:ymax_ind_l].values,\
                        dfFrame['bincenters_left'].loc[ymin_ind_l:ymax_ind_l].values)) - meanl
        # /(dfFrame['bincenters_left'].loc[ymax_ind_l]-dfFrame['bincenters_left'].loc[ymin_ind_l])) 

except:
    print(gs.ORANGE + 'Error in identifying max intensity value leftside')
    print('Last index:' + str(lastIndex - innerMeanInt))
    print('Max index left:' + str(max_ind_l) + gs.NORMAL)

fig2, axes = plt.subplots(2,1,figsize = (15,10))
dfprev = fluoDf[fluoDf['angle']== 5]
dfprev = dfprev[dfprev['frame']== 10]
x = dfprev['bincenters_left'][dfprev['values_right'].notna()].values
y = dfprev['values_left'].dropna().values

f2 = interp1d(x, y, kind='linear')
xnew = np.linspace(x[0], x[-1], num = 150)
ynew = f2(xnew)

axes[0].plot(x, y, 'o', label = 'data points')
axes[0].plot(xnew, ynew, '-', label = 'nearest')
axes[0].legend()



#%% Renaming badly named files for efficient handling after

dirExt = 'F:/Cortex Experiments/Fluorescence Experiments/20221015_3t3optorhoaLifeAct_40x_FluoTimeLapse'
allCells = os.listdir(dirExt)
newPrefix = 'cell'

for currentCell in allCells:
    print(currentCell)
    ufun.renamePrefix(dirExt, currentCell, newPrefix)
    
    
#%% Creating stacks from individual files

dirExt = 'F:/Cortex Experiments/Fluorescence Experiments/20221015_3t3optorhoaLifeAct_40x_FluoTimeLapse'
dirSave = 'D:/Anumita/MagneticPincherData/Fluorescence_Analysis/Raw'
prefix = 'cell'
channel = 'w3CSU640'

excludedCells = ufun.AllMMTriplets2Stack(dirExt, dirSave, prefix, channel)

