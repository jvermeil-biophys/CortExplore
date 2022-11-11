# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 15:25:35 2022

@author: anumi
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
# import utilityFunctions_JV as jvu
import UtilityFunctions as ufun

mainDataDir = 'D:/Anumita/MagneticPincherData'
experimentalDataDir = os.path.join(mainDir, "Data_Experimental_AJ")
dataDir = os.path.join(mainDir, "Data_Analysis")
timeSeriesDataDir = os.path.join(rawDir, "Data_TimeSeries")

figDir = os.path.join(rawDir, "Figures")
todayFigDir = os.path.join(figDir, "Historique/" + str(date.today()))

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



#%% Utility functions specific to Opto experiments

def findActivation(fieldDf):
    maxZidx = fieldDf['Z'].argmax() #Finding the index of the max Z
    maxZ = fieldDf['Z'][maxZidx] #To check if the value is correct
    return maxZidx, maxZ

def getCellTimeSeriesData(cellID, fromPython = True):
    if fromPython:
        allTimeSeriesDataFiles = [f for f in os.listdir(timeSeriesDataDir) 
                              if (os.path.isfile(os.path.join(timeSeriesDataDir, f)) 
                                  and f.endswith("PY.csv"))]
    else:
        allTimeSeriesDataFiles = [f for f in os.listdir(timeSeriesDataDir) 
                              if (os.path.isfile(os.path.join(timeSeriesDataDir, f)) 
                                  and f.endswith(".csv") and not f.endswith("PY.csv"))]
    fileFound = False
    nFile = len(allTimeSeriesDataFiles)
    iFile = 0
    while (not fileFound) and (iFile < nFile):
        f = allTimeSeriesDataFiles[iFile]
        if f.startswith(cellID + '_'):
            timeSeriesDataFilePath = os.path.join(timeSeriesDataDir, f)
            timeSeriesDataFrame = pd.read_csv(timeSeriesDataFilePath, sep=';')
            fileFound = True
        iFile += 1
    if not fileFound:
        timeSeriesDataFrame = pd.DataFrame([])
    else:
        for c in timeSeriesDataFrame.columns:
                if 'Unnamed' in c:
                    timeSeriesDataFrame = timeSeriesDataFrame.drop([c], axis=1)
    return(timeSeriesDataFrame)


def getOptoMeta(cellID):
    date = ufun.findInfosInFileName(cellID, 'date')
    date = date.replace('-', '.')
    optoMetaDataPath = rawDir+'//Raw//'+date
    allOptoMetaDataFiles = [f for f in os.listdir(optoMetaDataPath) 
                          if (os.path.isfile(os.path.join(optoMetaDataPath, f)) 
                              and f.endswith("OptoMetadata.txt"))]
    fileFound = False
    nFile = len(allOptoMetaDataFiles)
    iFile = 0
    while (not fileFound) and (iFile < nFile):
        f = allOptoMetaDataFiles[iFile]
        if f.startswith(cellID + '_'):
            optoMetaDataPath = os.path.join(optoMetaDataPath, f)
            optoMetaDatadf = pd.read_csv(optoMetaDataPath, sep='\t')
            fileFound = True
        iFile += 1
    if not fileFound:
        optoMetaDatadf = pd.DataFrame([])
    else:
        for c in optoMetaDatadf.columns:
                if 'Unnamed' in c:
                    optoMetaDatadf = optoMetaDatadf.drop([c], axis=1)
    return(optoMetaDatadf)

def addStat_df(ax, data, box_pairs, param, cond, test = 'Mann-Whitney', percentHeight = 95):
    refHeight = np.percentile(data[param].values, percentHeight)
    currentHeight = refHeight
    scale = ax.get_yscale()
    xTicks = ax.get_xticklabels()
    dictXTicks = {xTicks[i].get_text() : xTicks[i].get_position()[0] for i in range(len(xTicks))}
    for bp in box_pairs:
        c1 = data[data[cond] == bp[0]][param].values
        c2 = data[data[cond] == bp[1]][param].values
        if test == 'Mann-Whitney' or test == 'Wilcox_2s' or test == 'Wilcox_greater' or test == 'Wilcox_less' or test == 't-test':
            if test=='Mann-Whitney':
                statistic, pval = st.mannwhitneyu(c1,c2)
            elif test=='Wilcox_2s':
                statistic, pval = st.wilcoxon(c1,c2, alternative = 'two-sided')
            elif test=='Wilcox_greater':
                statistic, pval = st.wilcoxon(c1,c2, alternative = 'greater')
            elif test=='Wilcox_less':
                statistic, pval = st.wilcoxon(c1,c2, alternative = 'less')
            elif test=='t-test':
                statistic, pval = st.ttest_ind(c1,c2)
            text = 'ns'
            if pval < 0.05 and pval > 0.01:
                text = '*'
            elif pval < 0.01 and pval > 0.001:
                text = '**'
            elif pval < 0.001 and pval < 0.001:
                text = '***'
            elif pval < 0.0001:
                text = '****'
            ax.plot([bp[0], bp[1]], [currentHeight, currentHeight], 'k-', lw = 1)
            XposText = (dictXTicks[bp[0]]+dictXTicks[bp[1]])/2
            if scale == 'log':
                power = 0.01* (text=='ns') + 0.000 * (text!='ns')
                YposText = currentHeight*(refHeight**power)
            else:
                factor = 0.03 * (text=='ns') + 0.000 * (text!='ns')
                YposText = currentHeight + factor*refHeight
            ax.text(XposText, YposText, text, ha = 'center', color = 'k')
    #         if text=='ns':
    #             ax.text(posText, currentHeight + 0.025*refHeight, text, ha = 'center')
    #         else:
    #             ax.text(posText, currentHeight, text, ha = 'center')
            if scale == 'log':
                currentHeight = currentHeight*(refHeight**0.05)
            else:
                currentHeight =  currentHeight + 0.15*refHeight
        ax.set_ylim([ax.get_ylim()[0], currentHeight])

        if test == 'pairwise':
            ratio = (c2/c1)
            stdError = np.nanstd(ratio)/np.sqrt(np.size(c1))
            confInt = np.nanmean(ratio) - 1.96 * stdError
            print(stdError)
            print(confInt)
            return confInt


def ctFieldThicknessIndividual(experimentalDataDir, todayFigDir, date, save = False, background = 'default'):
    try:
        os.mkdir(todayFigDir)
    except:
        pass
    
    expDf = ufun.getExperimentalConditions(experimentalDataDir, save = False) 
    files = os.listdir(rawDir+'/Raw/'+date)

    if background == 'dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    for f in files:
        if f.endswith('.tif'):
            cellID = ufun.findInfosInFileName(f, 'cellID')
            timeSeriesDf = getCellTimeSeriesData(cellID)            
            optoMetaDataDf = getOptoMeta(cellID)
            manipID = ufun.findInfosInFileName(cellID, 'manipID')
            print(f)
            try:
               Tact = optoMetaDataDf['T_abs'].values[0]
            except:
                pass
            # Tact = timeSeriesDf['T'][timeSeriesDf['T_abs'] == Tact]
            fig = plt.figure(figsize=(20,20))
            plt.rcParams.update({'font.size': 25})
            plt.suptitle(cellID)
            t = timeSeriesDf['T'].values*1000/60
            
            plt.subplot(3, 1, 1)
            
            plt.plot(t, timeSeriesDf['D3'] - bead_dia)
            
            plt.subplot(3, 1, 2)
            plt.plot(t, timeSeriesDf['D2'] - bead_dia)
            
            plt.subplot(3, 1, 3)
            plt.plot(t, timeSeriesDf['dz'])
            
            plt.savefig(todayFigDir+'/'+cellID+'_ThicknessvTime')
            plt.show()
    plt.close('all')

def ctFieldThicknessAll(experimentalDataDir, todayFigDir, date, param_type = 'none', tag = 'all',  save = False, background = 'default'):
    bead_dia = 4.503
    
    try:
        os.mkdir(todayFigDir)
    except:
        pass
    
    if background == 'dark':
        plt.style.use('dark_background')
        
    else:
        plt.style.use('default')
    
    expDf = ufun.getExperimentalConditions(experimentalDataDir, save = False) 
    expDf = expDf[expDf['experimentType'] == 'optoGen']
    expDf = expDf[expDf['microscope'] == 'metamorph']
    cellConditionsDf = pd.read_csv(experimentalDataDir+'/cellConditions_Ct.csv')
    cellConditionsDf = cellConditionsDf[cellConditionsDf['excluded'] == 'no']
    allCells = cellConditionsDf['cellID'].values
    cellIDs = []
    
    
    if tag == 'all':
        cellIDs = allCells
    elif tag != 'all':
        selectedManipIDs = expDf['manipID'][expDf['activation type'] == tag].values
        for each in selectedManipIDs:
            cellID = (cellConditionsDf['cellID'][cellConditionsDf['cellID'].str.contains(each)])
            cellID = cellID.values
            if len(cellID) >= 1:
                cellIDs.extend(cellID)
    cellIDs = np.asarray(cellIDs)
    
    
    if param_type != 'none':
        cellID_copy = np.copy(cellIDs)
        cellIDs = []
        categories = []
        for each in cellID_copy:
            cellIDdf = cellConditionsDf[cellConditionsDf['cellID'].str.contains(each)]
            # if lencellIDdf[param_type].values != 'none':
            cellID = cellIDdf['cellID'].values
            categories.extend(cellIDdf[param_type].values)
            cellIDs.extend(cellID)
    
        cellIDs = np.asarray(cellIDs)
    
    plt.figure(figsize=(15,10))
    plt.rcParams.update({'font.size': 28})
    for cellID, i in zip(cellIDs, range(len(cellIDs))):
        timeSeriesDf = getCellTimeSeriesData(cellID)            
        optoMetaDataDf = getOptoMeta(cellID)
        Tact = optoMetaDataDf['T_abs'].values[0]
        time = timeSeriesDf['T'].values*1000/60
        label = cellID
        
        if param_type != 'none':
            param = categories[i]
            print(param)
            if param == 'polarised rear':
                color = 'blue'
                alpha = 1.0
            elif param == 'polarised front':
                color = 'green'
                alpha = 1.0
            elif param == 'global contraction':
                color = 'orange'
                alpha = 1.0
            elif param == 'blebby':
                color = 'red'
                alpha = 0.2
            elif param == 'none':
                color = 'black'
                alpha = 0.4
            plt.plot(time, timeSeriesDf['D3'].values - bead_dia, color = color, label = label, alpha = alpha)
            
        else:
           plt.rcParams['axes.prop_cycle']
           
           for i in range(23):
               plt.axvline(x= 5.5 + i*((30/60)), color = 'blue', ymax = 0.05, linewidth = 4)
           plt.plot(time, timeSeriesDf['D3'].values - bead_dia, label = label)
           
    
    plt.axvline(x = 5.0, color = 'r')
    plt.title('Thickness (um) vs Time : '+tag+' | On '+param_type)
    plt.legend(loc = 2, prop={'size': 10})
    plt.xlim(0,18)
    plt.ylim(0, 0.85)
    plt.show()
    plt.savefig(todayFigDir+'/All_'+tag+'_'+param_type+'_ThicknessvTime')


def ctFieldThicknessSummary(experimentalDataDir, todayFigDir, parameter, test = 'Wilcox_greater', plot = 0, kind = 'optogen'):
    
    try:
        os.mkdir(todayFigDir)
    except:
        pass
    
    #### No activation experiments
    if kind == 'none':
        plt.style.use('default')
        
        cellConditionsDf = pd.read_csv(experimentalDataDir+'/cellConditions_Ct.csv')
        cellConditionsDf = cellConditionsDf[cellConditionsDf['excluded'] == 'no']
        
        if len(parameter) == 1:
            if parameter[0] != 'all':
                cellConditionsDf = cellConditionsDf[cellConditionsDf['cellID'].str.contains(parameter[0])]
        elif len(parameter) > 1:
            cellConditionsDf = cellConditionsDf[(cellConditionsDf['cellID'].str.contains(parameter[0])) &\
                                                 (cellConditionsDf['cellID'].str.contains(parameter[1]))]
            
        listOfCells = np.asarray(cellConditionsDf['cellID'])
        magFields = cellConditionsDf['phenotype'].values
        
        fig1 = plt.figure(figsize = (20,10))
        plt.rcParams.update({'font.size': 12})
        
        summaryDict = {}
        summaryDict['cellID'] = []
        summaryDict['medianThicknessWhole'] = []
        summaryDict['medianThicknessToComp'] = []
        summaryDict['fluctuationsWhole'] = []
        summaryDict['ratioThickness'] = []
        summaryDict['ratioFluctuations'] = []
        summaryDict['fluctuationsToComp'] = []
        summaryDict['magField'] = []
        summaryDict['activationTag'] = []
        
        for i in range(len(listOfCells)):
            
            
            cellID = listOfCells[i]
            print(cellID)
            
            magField = magFields[i]
            timeSeriesDf = getCellTimeSeriesData(cellID)
            
            thickness = timeSeriesDf['D3'] - bead_dia
            time = timeSeriesDf['T'] #*1000/60
            
            #Thickness first and last 5 mins to compare to before and after activation
            
            thicknessBefore = thickness[:500]
            thicknessAfter = thickness[500:]
            timeBefore = time[:500]
            
            try:
                ratioThickness = thicknessAfter/thicknessBefore
            except:
                ratioThickness = np.nan
            
            
            fluctBefore = np.percentile(thicknessBefore, 90) - np.percentile(thicknessBefore, 10)
            try:
                fluctAfter = np.percentile(thicknessAfter, 90) - np.percentile(thicknessAfter, 10)
                ratioFluct = fluctAfter/fluctBefore
            except:
                fluctAfter = np.nan
                ratioFluct = np.nan
            
            summaryDict['cellID'].append(cellID)
            summaryDict['medianThicknessWhole'].append(thickness.median())
            summaryDict['medianThicknessToComp'].append(thicknessBefore.median())
            summaryDict['fluctuationsWhole'].append(np.percentile(thickness, 90) - np.percentile(thickness, 10))
            summaryDict['fluctuationsToComp'].append(fluctBefore)
            summaryDict['magField'].append(magField)
            summaryDict['ratioThickness'].append(ratioThickness)
            summaryDict['ratioFluctuations'].append(ratioFluct)
            summaryDict['activationTag'].append('Before')
            
            summaryDict['cellID'].append(cellID)
            summaryDict['medianThicknessWhole'].append(np.nan)
            summaryDict['medianThicknessToComp'].append(thicknessAfter.median())
            summaryDict['fluctuationsWhole'].append(np.nan)
            summaryDict['fluctuationsToComp'].append(fluctAfter)
            summaryDict['magField'].append(magField)
            summaryDict['ratioThickness'].append(np.nan)
            summaryDict['ratioFluctuations'].append(np.nan)
            summaryDict['activationTag'].append('After')
            
            summaryDf = pd.DataFrame(summaryDict)
            
            if plot == 1:
                
                #### Plots for individual thickness curves either for whole cell population of inidividually
                if len(parameter) == 1 and parameter[0] == 'all':
    
                    if magField == '3mT':
                        color = 'orange'
                    elif magField == '7mT':
                        color = 'blue'
                    elif magField == '15mT':
                        color = 'black'
                    plt.plot(time, thickness, color = color, label = cellID)
                    plt.legend(prop={'size': 6})
                    
                elif len(parameter) > 1 :
                    
                    if magField == '3mT':
                        color = 'orange'
                    elif magField == '7mT':
                        color = 'blue'
                    elif magField == '15mT':
                        color = 'black'
                    plt.plot(timeBefore, thicknessBefore, color = color, label = cellID)
                    plt.legend(prop={'size': 6})
                    
                elif len(parameter) == 1 and parameter[0] != 'all':
                    plt.plot(time, thickness, label = cellID)
                    plt.legend(prop={'size': 6})
                
                plt.savefig(todayFigDir+'/SummaryThickness_NoActivation_'+kind+"_"+str(parameter)+'.png')
                
                plt.show()
                cellConditionsDf.to_excel(todayFigDir+"/cellConditions_Ct_"+kind+"_"+str(parameter)+".xlsx")  
                
                # summaryDf = np.nan
                
        if plot == 2:
            
            
            #### medianThickness for different magnetic fields
            ax1 = plt.figure()
            palette = sns.color_palette("light:b")
            sns.set_palette(palette)
            dataSpec = summaryDf[summaryDf['activationTag'] == 'Before']
            ax1 = sns.boxplot(x = 'magField', y='medianThicknessWhole', data=summaryDf, palette = palette)
            ax1 = sns.swarmplot(x = 'magField', y='medianThicknessWhole', data=summaryDf, color ='black', size = 6)
            # ax1.set_ylim(0, 1)
            
            for patch in ax1.artists:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, 0.3))
                        
            # print(len(dataSpec[''])
            addStat_df(ax1, dataSpec, [('3mT', '7mT'), ('3mT', '15mT'), ('7mT', '15mT')], param = 'medianThicknessWhole', test = 'Mann-Whitney', cond = 'magField')
         
            plt.suptitle('Median Thickness | '+str(parameter))
            plt.savefig(todayFigDir+'/medianThickness_'+kind+'_'+str(parameter)+'.png')
            plt.show()
            
            #### medianThickness for different magnetic fields first 5 mins
            ax1b = plt.figure()
            palette = sns.color_palette("light:b")
            sns.set_palette(palette)
            dataSpec = summaryDf[summaryDf['activationTag'] == 'Before']
            ax1b = sns.boxplot(x = 'magField', y = 'medianThicknessToComp', data=dataSpec, palette = palette)
            ax1b = sns.swarmplot(x = 'magField', y = 'medianThicknessToComp', data=dataSpec, color ='black', size = 6)
            # ax1.set_ylim(0, 1)
            
            for patch in ax1b.artists:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, 0.3))
            
            dataSpec_test = dataSpec.dropna()
            addStat_df(ax1b, dataSpec_test, [('3mT', '15mT'), ('3mT', '7mT'), ('7mT', '15mT')], param = 'medianThicknessToComp', test = 'Mann-Whitney', cond = 'magField')
           
            
            plt.suptitle('Median Thickness_5mins | '+str(parameter))
            plt.savefig(todayFigDir+'/medianThickness5mins_'+kind+'_'+str(parameter)+'.png')
            plt.show()
            
            #### fluctuations for different magnetic fields
            ax2 = plt.figure()
            palette = sns.color_palette("light:b")
            sns.set_palette(palette)
            dataSpec = summaryDf[summaryDf['activationTag'] == 'Before']
            ax2 = sns.boxplot(x = 'magField', y='fluctuationsWhole', data=summaryDf, palette = palette)
            ax2 = sns.swarmplot(x = 'magField', y='fluctuationsWhole', data=summaryDf, color ='black', size = 6)
            # ax1.set_ylim(0, 1)
            
            for patch in ax2.artists:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, 0.3))
            
            addStat_df(ax2, dataSpec, [('3mT', '15mT'), ('3mT', '7mT'), ('7mT', '15mT')], param = 'fluctuationsWhole', test = 'Mann-Whitney', cond = 'magField')

            plt.suptitle('Fluctuations | '+str(parameter))
            plt.savefig(todayFigDir+'/fluctuations_'+kind+'_'+str(parameter)+'.png')
            plt.show()
            
            #### fluctuations for different magnetic fields first 5 mins
            ax2b = plt.figure()
            palette = sns.color_palette("light:b")
            sns.set_palette(palette)
            dataSpec = summaryDf[summaryDf['activationTag'] == 'Before']
            ax2b = sns.boxplot(x = 'magField', y = 'fluctuationsToComp', data=dataSpec, palette = palette)
            ax2b = sns.swarmplot(x = 'magField', y = 'fluctuationsToComp', data=dataSpec, color ='black', size = 6)
            # ax1.set_ylim(0, 1)
            
            for patch in ax2b.artists:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, 0.3))
            

            dataSpec_test = dataSpec.dropna()
            addStat_df(ax2b, dataSpec_test, [('3mT', '15mT'), ('3mT', '7mT'), ('7mT', '15mT')], param = 'fluctuationsToComp', test = 'Mann-Whitney', cond = 'magField')
            
            plt.suptitle('Fluctuations_5mins | '+str(parameter))
            plt.savefig(todayFigDir+'/fluctuations5mins_'+kind+'_'+str(parameter)+'.png')
            plt.show()
            
            #### median thickness vs fluctuations for different magnetic fields
            ax3 = plt.figure()
            palette = ["orange", "blue", "#000000"]
            sns.set_palette(palette)
            ax3 = sns.scatterplot(x = 'fluctuationsWhole', y ='medianThicknessWhole', data = summaryDf, hue = 'magField', palette = palette, s= 40)
            
            plt.suptitle('Median Thickness vs. Fluctuations | '+str(parameter))
            plt.savefig(todayFigDir+'/thicknessVfluctuations_'+kind+'_'+str(parameter)+'.png')
            plt.show()
            
            #### median thickness vs fluctuations for different magnetic fields only 5 mins
            
            ax3b = plt.figure()
            palette = ["orange", "blue", "#000000"]
            sns.set_palette(palette)
            dataSpec = summaryDf[summaryDf['activationTag'] == 'Before']
            ax3b = sns.scatterplot(x = 'fluctuationsToComp', y ='medianThicknessToComp', data = dataSpec, hue = 'magField', palette = palette, s= 40)
            
            plt.suptitle('Median Thickness vs. Fluctuations (5mins)| '+str(parameter))
            plt.savefig(todayFigDir+'/thicknessVfluctuations5mins_'+kind+'_'+str(parameter)+'.png')
            plt.show()
            

            #### medianThickness Before/After 5mins to compare with activation for different magnetic fields
            # 3mT
            ax4a = plt.figure()
            palette = sns.color_palette("light:b")
            sns.set_palette(palette)
            dataSpec = summaryDf[summaryDf['magField'] == '3mT']
            ax4a = sns.boxplot(x = 'activationTag', y='medianThicknessToComp', data=dataSpec, palette = palette) #, palette = palette)
            ax4a = sns.swarmplot(x = 'activationTag', y='medianThicknessToComp', data=dataSpec, color ='black', size = 6)
            # ax1.set_ylim(0, 1)
            
            for patch in ax4a.artists:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, 0.3))
            # addStat_df(ax4a, dataSpec, [('After', 'Before')], 'medianThicknessToComp', test = 'Wilcox_greater', cond = 'activationTag')
            plt.suptitle('MedianThicknessControl_3mT | '+str(parameter))
            plt.savefig(todayFigDir+'/MedianThicknessActivationControl3mT_'+kind+'_'+str(parameter)+'.png')
            plt.show()
            
            #7mT
            ax4b = plt.figure()
            palette = sns.color_palette("light:b")
            sns.set_palette(palette)
            dataSpec = summaryDf[summaryDf['magField'] == '7mT']
            ax4b = sns.boxplot(x = 'activationTag', y='medianThicknessToComp', data=dataSpec, palette = palette) #, palette = palette)
            ax4b = sns.swarmplot(x = 'activationTag', y='medianThicknessToComp', data=dataSpec, color ='black', size = 6)
            # ax1.set_ylim(0, 1)
            
            for patch in ax4a.artists:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, 0.3))
            # addStat_df(ax4b, dataSpec, [('After', 'Before')], 'medianThicknessToComp', test = 'Wilcox_greater', cond = 'activationTag')
            plt.suptitle('MedianThicknessControl_7mT | '+str(parameter))
            plt.savefig(todayFigDir+'/MedianThicknessActivationControl7mT_'+kind+'_'+str(parameter)+'.png')
            plt.show()
            
            #15 mT
            ax4c = plt.figure()
            palette = sns.color_palette("light:b")
            sns.set_palette(palette)
            dataSpec = summaryDf[summaryDf['magField'] == '15mT']
            ax4c = sns.boxplot(x = 'activationTag', y='medianThicknessToComp', data=dataSpec, palette = palette) #, palette = palette)
            ax4c = sns.swarmplot(x = 'activationTag', y='medianThicknessToComp', data=dataSpec, color ='black', size = 6)
            # ax1.set_ylim(0, 1)
            
            for patch in ax4c.artists:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, 0.3))
            
            # addStat_df(ax4c, dataSpec, [('After', 'Before')], 'medianThicknessToComp', test = 'Wilcox_greater', cond = 'activationTag')
            plt.suptitle('MedianThicknessControl_15mT | '+str(parameter))
            plt.savefig(todayFigDir+'/MedianThicknessActivationControl15mT_'+kind+'_'+str(parameter)+'.png')
            plt.show()
            
            ####Plot fluctuations vs. median thickness : Indidivually for diff activation types  
            palette = sns.color_palette("tab10")
            sns.set_palette(palette)
            
            lm = sns.lmplot(x ='medianThicknessToComp', y ='fluctuationsToComp', data = summaryDf, \
                            hue ='activationTag', col = 'magField')
                
            fig2 = lm.fig
            text_x = [0.08, 0.38, 0.7]
            text1_y = [0.812, 0.812, 0.812]
            text2_y = [0.78, 0.78, 0.78]
            fields = ['3mT', '7mT', '15mT']
            
            #Calculating correlation coefficient b/w fluctuation and median thickness
            for j in range(len(fields)):
                dataSpec = summaryDf[summaryDf['magField'] == fields[j]]
                fluctuationsSpecAfter = dataSpec['fluctuationsToComp'][dataSpec['activationTag'] == 'After'].values
                thicknessSpecAfter = dataSpec['medianThicknessToComp'][dataSpec['activationTag'] == 'After'].values
                corrCoefAfter = np.corrcoef(thicknessSpecAfter, fluctuationsSpecAfter)
        
                
                fluctuationsSpecBefore = dataSpec['fluctuationsToComp'][dataSpec['activationTag'] == 'Before'].values
                thicknessSpecBefore = dataSpec['medianThicknessToComp'][dataSpec['activationTag'] == 'Before'].values
                corrCoefBefore = np.corrcoef(thicknessSpecBefore, fluctuationsSpecBefore)
        
                
                fig2.text(text_x[j], text1_y[j], str(np.round(corrCoefAfter[0][1], 3)), color = 'orange')
                fig2.text(text_x[j], text2_y[j], str(np.round(corrCoefBefore[0][1], 3)), color = 'blue')
            
            fig2.suptitle('Fluctuations vs. Median thickness')
            fig2.tight_layout()
            plt.savefig(todayFigDir+'/Summary_MagFieldsControl_FluctuationsvsThickness.png')
            plt.show()
    
    #### Optogenetic experiments
    if kind == 'optogen':
        expDf = ufun.getExperimentalConditions(experimentalDataDir, save = False)
        cellConditionsDf = pd.read_csv(experimentalDataDir+'/cellConditions_Ct.csv')
        plt.style.use('default')
        listOfCells = np.asarray(cellConditionsDf['cellID'][cellConditionsDf['excluded'] == 'no'])

        summaryDict = {}
        summaryDict['cellID'] = []
        summaryDict['medianThickness'] = []
        summaryDict['medianThickness5minRange'] = []
        summaryDict['activationTag'] = []
        summaryDict['activationType'] = []
        summaryDict['fluctuations'] = []
        summaryDict['fluctuations5minRange'] = []
        summaryDict['blebCondition'] = []
        summaryDict['ratioThickness'] = []
        summaryDict['ratioFluctuations'] = []
        summaryDict['ratioFluctThick'] = []
        summaryDict['ratioFluctuations5min'] = []
        summaryDict['ratioThickness5min'] = []
        summaryDict['phenotype'] = []
        
        #### Constructing CT Field Summary Table
        for i in range(len(listOfCells)):
            
            cellID = listOfCells[i]
            print(cellID)
            timeSeriesDf = getCellTimeSeriesData(cellID)
            optoMetaDataDf = getOptoMeta(cellID)
            manipID = ufun.findInfosInFileName(cellID, 'manipID')
    
            Tact = optoMetaDataDf['T_abs'].values[0]
            # Tact = 5.0
        
            summaryDict['cellID'].append(cellID)
            thicknessBefore = (timeSeriesDf['D3']-bead_dia)[(timeSeriesDf['T']*1000/60 < Tact)]
            medianThicknessBefore = thicknessBefore.median()
            fluctBefore = np.percentile(thicknessBefore, 90) - np.percentile(thicknessBefore, 10)
            summaryDict['medianThickness'].append(medianThicknessBefore)
            summaryDict['activationTag'].append('Before')
            summaryDict['activationType'].append(expDf['activation type'][(expDf['manipID'] == manipID)].values[0])
            summaryDict['fluctuations'].append(fluctBefore)
            summaryDict['blebCondition'].append(cellConditionsDf['blebCondition'][cellConditionsDf['cellID']==cellID].values[0])
            summaryDict['ratioThickness'].append(np.nan)
            summaryDict['ratioFluctuations'].append(np.nan)
            summaryDict['ratioFluctThick'].append(np.nan)
            summaryDict['ratioFluctuations5min'].append(np.nan)
            summaryDict['ratioThickness5min'].append(np.nan)
            summaryDict['medianThickness5minRange'].append(medianThicknessBefore)
            summaryDict['fluctuations5minRange'].append(fluctBefore)
            summaryDict['phenotype'].append(cellConditionsDf['phenotype'][cellConditionsDf['cellID']==cellID].values[0])
            
            summaryDict['cellID'].append(cellID)
            thicknessAfter = (timeSeriesDf['D3']-bead_dia)[(timeSeriesDf['Tabs']*1000/60 > Tact)]
            medianThicknessAfter = thicknessAfter.median()
            fluctAfter = np.percentile(thicknessAfter, 90) - np.percentile(thicknessAfter, 10)
            summaryDict['medianThickness'].append(medianThicknessAfter)
            summaryDict['activationTag'].append('After')
            summaryDict['activationType'].append(expDf['activation type'][(expDf['manipID'] == manipID)].values[0])
            summaryDict['fluctuations'].append(fluctAfter)
            summaryDict['blebCondition'].append(cellConditionsDf['blebCondition'][cellConditionsDf['cellID']==cellID].values[0])
            summaryDict['ratioThickness'].append(medianThicknessAfter/medianThicknessBefore)
            summaryDict['ratioFluctuations'].append(fluctAfter/fluctBefore)
            summaryDict['ratioFluctThick'].append((fluctAfter/fluctBefore)/(medianThicknessAfter/medianThicknessBefore))
            summaryDict['phenotype'].append(cellConditionsDf['phenotype'][cellConditionsDf['cellID']==cellID].values[0])
            
            thicknessLast5min = thicknessAfter[timeSeriesDf['T']*1000/60 > 10.0]
            medianThicknessLast5min = thicknessLast5min.median()
            summaryDict['medianThickness5minRange'].append(medianThicknessLast5min)
            fluctuationsLast5min = np.percentile(thicknessLast5min, 90) - np.percentile(thicknessLast5min, 10)
            summaryDict['fluctuations5minRange'].append(fluctuationsLast5min)
            summaryDict['ratioFluctuations5min'].append(fluctuationsLast5min/fluctBefore)
            summaryDict['ratioThickness5min'].append(medianThicknessLast5min/medianThicknessBefore)
    
            summaryDf = pd.DataFrame(summaryDict)
    
            
        #### Subplots of each activation type, thickness
        activationType = ['global', 'at beads', 'away from beads']
        fig1, axs = plt.subplots(nrows = 1, ncols = 3)
        color = ['r', 'g', 'b']
        for j in range(len(activationType)):
            dataSpecific = summaryDf[summaryDf['activationType'] == activationType[j]]
            
            y1 = dataSpecific['medianThickness'][dataSpecific['activationTag'] == 'Before']
            y2 = dataSpecific['medianThickness'][dataSpecific['activationTag'] == 'After']
            
            axs[j].plot((np.zeros(len(y1), dtype = int), np.ones(len(y2), dtype = int)), (y1, y2), '-o', color = color[j])
    
            axs[j].set_title(activationType[j])
            labels = ['Before, After']
            axs[j].set_xticks = ([1,2], labels)
            axs[j].set_ylim(0, 1.5)
        plt.suptitle('Median Thickness')
        fig1.tight_layout()
        plt.savefig(todayFigDir+'/ActivationSpecificSubplots_Thickness.png')
        plt.show()
        
        #### Subplots of each activation type, fluctuations
        fig2, axs = plt.subplots(nrows = 1, ncols = 3)
        for j in range(len(activationType)):
            dataSpecific = summaryDf[summaryDf['activationType'] == activationType[j]]
            
            y1 = dataSpecific['fluctuations'][dataSpecific['activationTag'] == 'Before']
            y2 = dataSpecific['fluctuations'][dataSpecific['activationTag'] == 'After']
            
            axs[j].plot((np.zeros(len(y1), dtype = int), np.ones(len(y2), dtype = int)), (y1, y2), '-o', color = color[j])
            # axs[j].plot((y1, y2), '-o', color = color[j])
    
            axs[j].set_title(activationType[j])
            labels = ['Before, After']
            axs[j].set_ylim(0,1.5)
            axs[j].set_xticks = ([1,2], labels)
        
        plt.suptitle('Fluctuations')
        fig2.tight_layout()
        plt.savefig(todayFigDir+'/ActivationSpecificSubplots_Fluctuations.png')
        plt.show()
        
        #### medianThickness first/last 5mins boxplots, global
        ax1a_0 = plt.figure()
        activationType = 'global'
        data_onlyGlobal = summaryDf[summaryDf['activationType'] == activationType]
        ax1a_0 = sns.boxplot(x = 'activationTag', y='medianThickness5minRange', data=data_onlyGlobal, color = 'skyblue')
        ax1a_0 = sns.swarmplot(x = 'activationTag', y='medianThickness5minRange', data=data_onlyGlobal, hue = parameter)
        ax1a_0 = sns.lineplot(data=data_onlyGlobal, x="activationTag", y="medianThickness5minRange", units="cellID", color = '0.7',  estimator=None)
        addStat_df(ax1a_0, data_onlyGlobal, [('After', 'Before')], 'medianThickness5minRange', test = test, cond = 'activationTag')
        ax1a_0.set_ylim(0, 1)
        for patch in ax1a_0.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.suptitle(activationType)
        plt.savefig(todayFigDir+'/AllThickness_Before-AfterActivation5mins_'+parameter+'_'+activationType+'.png')
        plt.show()
        
        ####  medianThickness first/last 5mins boxplots, at beads
        ax1b_0 = plt.figure()
        activationType = 'at beads'
        data_onlyGlobal = summaryDf[summaryDf['activationType'] == activationType]
        ax1b_0 = sns.boxplot(x = 'activationTag', y='medianThickness5minRange', data=data_onlyGlobal, color = 'skyblue')
        ax1b_0 = sns.swarmplot(x = 'activationTag', y='medianThickness5minRange', data=data_onlyGlobal, hue = parameter)
        ax1b_0 = sns.lineplot(data=data_onlyGlobal, x="activationTag", y="medianThickness5minRange", units="cellID", color = '0.7',  estimator=None)
        addStat_df(ax1b_0, data_onlyGlobal, [('After', 'Before')], 'medianThickness5minRange', test = test, cond = 'activationTag')
        ax1b_0.set_ylim(0, 1)
        for patch in ax1b_0.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.suptitle(activationType)
        plt.savefig(todayFigDir+'/AllThickness_Before-AfterActivation5mins_'+parameter+'_'+activationType+'.png')
        plt.show()
        
        ####  medianThickness first/last 5mins boxplots, away from beads
        ax1c_0 = plt.figure()
        activationType = 'away from beads'
        data_onlyGlobal = summaryDf[summaryDf['activationType'] == activationType]
        ax1c_0 = sns.boxplot(x = 'activationTag', y='medianThickness5minRange', data=data_onlyGlobal, color = 'skyblue')
        ax1c_0 = sns.swarmplot(x = 'activationTag', y='medianThickness5minRange', data=data_onlyGlobal, hue = parameter)
        ax1c_0 = sns.lineplot(data=data_onlyGlobal, x="activationTag", y="medianThickness5minRange", units="cellID", color = '0.7',  estimator=None)
        addStat_df(ax1c_0, data_onlyGlobal, [('After', 'Before')], 'medianThickness5minRange', test = test, cond = 'activationTag')
        ax1c_0.set_ylim(0, 1)
        for patch in ax1c_0.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.suptitle(activationType)
        plt.savefig(todayFigDir+'/AllThickness_Before-AfterActivation5mins_'+parameter+'_'+activationType+'.png')
        plt.show()
        
        #### medianThickness boxplots, global
        ax1a = plt.figure()
        activationType = 'global'
        data_onlyGlobal = summaryDf[summaryDf['activationType'] == activationType]
        ax1a = sns.boxplot(x = 'activationTag', y='medianThickness', data=data_onlyGlobal, color = 'skyblue')
        ax1a = sns.swarmplot(x = 'activationTag', y='medianThickness', data=data_onlyGlobal, hue = parameter)
        ax1a = sns.lineplot(data=data_onlyGlobal, x="activationTag", y="medianThickness", units="cellID",  color = "0.7", estimator=None)
        
        addStat_df(ax1a, data_onlyGlobal, [('After', 'Before')], 'medianThickness', test = test, cond = 'activationTag')
        ax1a.set_ylim(0, 1)
        for patch in ax1a.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.suptitle(activationType)
        plt.savefig(todayFigDir+'/AllThickness_Before-AfterActivation_'+parameter+'_'+activationType+'.png')
        plt.show()
        
        ####  medianThickness boxplots, at beads
        ax1b = plt.figure()
        activationType = 'at beads'
        data_onlyGlobal = summaryDf[summaryDf['activationType'] == activationType]
        ax1b = sns.boxplot(x = 'activationTag', y='medianThickness', data=data_onlyGlobal, color = 'skyblue')
        ax1b = sns.swarmplot(x = 'activationTag', y='medianThickness', data=data_onlyGlobal, hue = parameter)
        ax1b = sns.lineplot(data=data_onlyGlobal, x="activationTag", y="medianThickness", units="cellID", color = '0.7',  estimator=None)
        addStat_df(ax1b, data_onlyGlobal, [('After', 'Before')], 'medianThickness', test = test, cond = 'activationTag')
        ax1b.set_ylim(0, 1)
        for patch in ax1b.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.suptitle(activationType)
        plt.savefig(todayFigDir+'/AllThickness_Before-AfterActivation_'+parameter+'_'+activationType+'.png')
        plt.show()
        
        ####  medianThickness boxplots, away from beads
        ax1c = plt.figure()
        activationType = 'away from beads'
        data_onlyGlobal = summaryDf[summaryDf['activationType'] == activationType]
        ax1c = sns.boxplot(x = 'activationTag', y='medianThickness', data=data_onlyGlobal, color = 'skyblue')
        ax1c = sns.swarmplot(x = 'activationTag', y='medianThickness', data=data_onlyGlobal, hue = parameter)
        ax1c = sns.lineplot(data=data_onlyGlobal, x="activationTag", y="medianThickness", units="cellID", color = '0.7',  estimator=None)
        addStat_df(ax1c, data_onlyGlobal, [('After', 'Before')], 'medianThickness', test = test, cond = 'activationTag')
        ax1c.set_ylim(0, 1)
        for patch in ax1c.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.suptitle(activationType)
        plt.savefig(todayFigDir+'/AllThickness_Before-AfterActivation_'+parameter+'_'+activationType+'.png')
        plt.show()
        
        ####  fluctuations first/last 5mins boxplots, away from beads
        ax2c_0 = plt.figure()
        activationType = 'away from beads'
        data_onlyGlobal = summaryDf[summaryDf['activationType'] == activationType]
        ax2c_0 = sns.boxplot(x = 'activationTag', y='fluctuations5minRange', data=data_onlyGlobal, color = 'skyblue')
        ax2c_0 = sns.swarmplot(x = 'activationTag', y='fluctuations5minRange', data=data_onlyGlobal, hue = parameter)
        ax2c_0 = sns.lineplot(data=data_onlyGlobal, x="activationTag", y="fluctuations5minRange", units="cellID", color = '0.7',  estimator=None)
        addStat_df(ax2c_0, data_onlyGlobal, [('After', 'Before')], 'fluctuations5minRange', test = test, cond = 'activationTag')
        ax2c_0.set_ylim(0, 1)
        for patch in ax2c_0.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.suptitle(activationType)
        plt.savefig(todayFigDir+'/AllFluctuations_Before-AfterActivation5mins_'+parameter+'_'+activationType+'.png')
        plt.show()
        
        #### fluctuations first/last 5mins boxplots, global
        ax2b_0 = plt.figure()
        activationType = 'global'
        data_onlyGlobal = summaryDf[summaryDf['activationType'] == activationType]
        ax2b_0 = sns.boxplot(x = 'activationTag', y='fluctuations5minRange', data=data_onlyGlobal, color = 'skyblue')
        ax2b_0 = sns.swarmplot(x = 'activationTag', y='fluctuations5minRange', data=data_onlyGlobal, hue = parameter)
        ax2b_0 = sns.lineplot(data=data_onlyGlobal, x="activationTag", y="fluctuations5minRange", units="cellID", color = '0.7',  estimator=None)
        addStat_df(ax2b_0, data_onlyGlobal, [('After', 'Before')], 'fluctuations5minRange', test = test, cond = 'activationTag')
        ax2b_0.set_ylim(0, 1)
        plt.suptitle(activationType)
        for patch in ax2b_0.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.savefig(todayFigDir+'/AllFluctuations_Before-AfterActivation5mins_'+parameter+'_'+activationType+'.png')
        plt.show()
        
        #### fluctuations first/last 5mins boxplots, at beads
        ax2c_0 = plt.figure()
        activationType = 'at beads'
        data_onlyGlobal = summaryDf[summaryDf['activationType'] == activationType]
        ax2c_0 = sns.boxplot(x = 'activationTag', y='fluctuations5minRange', data=data_onlyGlobal, color = 'skyblue')
        ax2c_0 = sns.swarmplot(x = 'activationTag', y='fluctuations5minRange', data=data_onlyGlobal, hue = parameter)
        ax2c_0 = sns.lineplot(data=data_onlyGlobal, x="activationTag", y="fluctuations5minRange", units="cellID",color = '0.7',   estimator=None)
        addStat_df(ax2c_0, data_onlyGlobal, [('After', 'Before')], 'fluctuations5minRange', test = test, cond = 'activationTag')
        ax2c_0.set_ylim(0, 1)
        for patch in ax2c_0.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.suptitle(activationType)
        plt.savefig(todayFigDir+'/AllFluctuations_Before-AfterActivation5mins_'+parameter+'_'+activationType+'.png')
        plt.show()
        
        ####  fluctuations boxplots, away from beads
        ax2c = plt.figure()
        activationType = 'away from beads'
        data_onlyGlobal = summaryDf[summaryDf['activationType'] == activationType]
        ax2c = sns.boxplot(x = 'activationTag', y='fluctuations', data=data_onlyGlobal, color = 'skyblue')
        ax2c = sns.swarmplot(x = 'activationTag', y='fluctuations', data=data_onlyGlobal, hue = parameter)
        ax2c = sns.lineplot(data=data_onlyGlobal, x="activationTag", y="fluctuations", units="cellID", color = '0.7',  estimator=None)
        addStat_df(ax2c, data_onlyGlobal, [('After', 'Before')], 'fluctuations', test = test, cond = 'activationTag')
        ax2c.set_ylim(0, 1)
        for patch in ax2c.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.suptitle(activationType)
        plt.savefig(todayFigDir+'/AllFluctuations_Before-AfterActivation_'+parameter+'_'+activationType+'.png')
        plt.show()
        
        #### fluctuations boxplots, global
        ax2b = plt.figure()
        activationType = 'global'
        data_onlyGlobal = summaryDf[summaryDf['activationType'] == activationType]
        ax2b = sns.boxplot(x = 'activationTag', y='fluctuations', data=data_onlyGlobal, color = 'skyblue')
        ax2b = sns.swarmplot(x = 'activationTag', y='fluctuations', data=data_onlyGlobal, hue = parameter)
        ax2b = sns.lineplot(data=data_onlyGlobal, x="activationTag", y="fluctuations", units="cellID", color = '0.7',  estimator=None)

        addStat_df(ax2b, data_onlyGlobal, [('After', 'Before')], 'fluctuations', test = test, cond = 'activationTag')
        ax2b.set_ylim(0, 1)
        plt.suptitle(activationType)
        for patch in ax2b.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.savefig(todayFigDir+'/AllFluctuations_Before-AfterActivation_'+parameter+'_'+activationType+'.png')
        plt.show()
        
        #### fluctuations boxplots, at beads
        ax2c = plt.figure()
        activationType = 'at beads'
        data_onlyGlobal = summaryDf[summaryDf['activationType'] == activationType]
        ax2c = sns.boxplot(x = 'activationTag', y='fluctuations', data=data_onlyGlobal, color = 'skyblue')
        ax2c = sns.swarmplot(x = 'activationTag', y='fluctuations', data=data_onlyGlobal, hue = parameter)
        ax2c = sns.lineplot(data=data_onlyGlobal, x="activationTag", y="fluctuations", units="cellID",color = '0.7',  estimator=None)

        addStat_df(ax2c, data_onlyGlobal, [('After', 'Before')], 'fluctuations', test = test, cond = 'activationTag')
        ax2c.set_ylim(0, 1)
        for patch in ax2c.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.suptitle(activationType)
        plt.savefig(todayFigDir+'/AllFluctuations_Before-AfterActivation_'+parameter+'_'+activationType+'.png')
        plt.show()
        
        
        ####All cells thickness before/after activation
        ax2 = plt.figure()
        ax2 = sns.boxplot(x = 'activationTag', y='medianThickness', data=summaryDf, color = 'skyblue')
        ax2 = sns.swarmplot(x = 'activationTag', y='medianThickness', data=summaryDf, hue = 'activationType')
        ax2 = sns.lineplot(data=data_onlyGlobal, x="activationTag", y="medianThickness", units="cellID", color = '0.7',  estimator=None)

        addStat_df(ax2, summaryDf, [('After', 'Before')], 'medianThickness', test = test, cond = 'activationTag')
        ax2.set_ylim(0, 1)
        for patch in ax2.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.savefig(todayFigDir+'/AllThickness_Before-AfterActivation.png')
        plt.show()
        
        
        ####Plot fluctuations before and after activation
        ax3 = plt.figure()
        ax3 = sns.boxplot(x = 'activationTag', y='fluctuations', data=summaryDf, color = 'skyblue')
        ax3 = sns.swarmplot(x = 'activationTag', y='fluctuations', data=summaryDf, hue = 'activationType')
        ax3 = sns.lineplot(data=data_onlyGlobal, x="activationTag", y="fluctuations", units="cellID", color = '0.7',  estimator=None)

        addStat_df(ax3, summaryDf, [('After', 'Before')], 'fluctuations', test = test, cond = 'activationTag')
        ax3.set_ylim(0, 1)
        
        for patch in ax3.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.savefig(todayFigDir+'/AllFluctuations_Before-AfterActivation.png')
        plt.show()
        
        ####Plot fluctuations vs. median thickness
        ax4 = plt.figure()
        ax4 = sns.scatterplot(x = 'medianThickness', y='fluctuations', data=summaryDf, \
                              hue = 'activationTag', style = 'activationType', s = 100)
        plt.savefig(todayFigDir+'/Summary_FluctuationsvsThickness.png')
        plt.show()
        
        
        ####Plot fluctuations vs. median thickness : Indidivually for diff activation types    
        lm = sns.lmplot(x ='medianThickness', y ='fluctuations', data = summaryDf, \
                        hue ='activationTag', col = 'activationType')
            
        fig3 = lm.fig
        
        text_x = [0.05, 0.38, 0.7]
        text1_y = [0.812, 0.812, 0.812]
        text2_y = [0.78, 0.78, 0.78]
        activationType = ['global', 'at beads', 'away from beads']
        #Calculating correlation coefficient b/w fluctuation and median thickness
        for j in range(len(activationType)):
            dataSpec = summaryDf[summaryDf['activationType'] == activationType[j]]
            print(activationType[j])
            fluctuationsSpecAfter = dataSpec['fluctuations'][dataSpec['activationTag'] == 'After'].values
            thicknessSpecAfter = dataSpec['medianThickness'][dataSpec['activationTag'] == 'After'].values
            corrCoefAfter = np.corrcoef(thicknessSpecAfter, fluctuationsSpecAfter)
    
            
            fluctuationsSpecBefore = dataSpec['fluctuations'][dataSpec['activationTag'] == 'Before'].values
            thicknessSpecBefore = dataSpec['medianThickness'][dataSpec['activationTag'] == 'Before'].values
            corrCoefBefore = np.corrcoef(thicknessSpecBefore, fluctuationsSpecBefore)
            # # print(corrCoefBefore)
            # print(corrCoefAfter)
            
            fig3.text(text_x[j], text1_y[j], str(np.round(corrCoefAfter[0][1], 3)), color = 'orange')
            fig3.text(text_x[j], text2_y[j], str(np.round(corrCoefBefore[0][1], 3)), color = 'blue')
        
        fig3.suptitle('Fluctuations vs. Median thickness')
        fig3.tight_layout()
        plt.savefig(todayFigDir+'/Summary_ActivationType_FluctuationsvsThickness.png')
        plt.show()
        
        
        #### Ratio of thicknesses first/last 5 mins after/before activation
        
        ax5_0 = plt.figure()
        ax5_0 = sns.scatterplot(x = 'cellID',  y = 'ratioThickness5min', data=summaryDf, style = parameter, hue = 'activationType', s = 100)
        ax5_0.set_xlabel('cellID')
        ax5_0.set_ylabel('After/Before Ratio Thickness (5 mins before/after)')
        # confInt = addStat_df(ax5, summaryDf, [('Before', 'After')], 'medianThickness', test = 'pairwise', cond = 'activationTag')
        # plt.suptitle("Mean - 1.96*StdError: " + str(confInt))
        ax5_0.set_xticklabels(summaryDf['cellID'].values, rotation = 90)
        ax5_0.axhline(y = 1.0, color = 'r')
        ax5_0.set_ylim(0,4)
        plt.savefig(todayFigDir+'/Summary_RatioThickness5mins'+parameter+'.png')
        plt.show()
        
        #### Ratio of thicknesses after/before activation
        
        ax5 = plt.figure()
        ax5 = sns.scatterplot(x = 'cellID',  y = 'ratioThickness', data=summaryDf, style = parameter, hue = 'activationType', s = 100)
        ax5.set_xlabel('cellID')
        ax5.set_ylabel('After/Before Ratio Thickness')
        # confInt = addStat_df(ax5, summaryDf, [('Before', 'After')], 'medianThickness', test = 'pairwise', cond = 'activationTag')
        # plt.suptitle("Mean - 1.96*StdError: " + str(confInt))
        ax5.set_xticklabels(summaryDf['cellID'].values, rotation = 90)
        ax5.axhline(y = 1.0, color = 'r')
        ax5.set_ylim(0,4)
        plt.savefig(todayFigDir+'/Summary_RatioThickness'+parameter+'.png')
        plt.show()
        
        #### Ratio of fluctuations first/last 5 mins after/before activation
        ax6_0 = plt.figure()
        ax6_0 = sns.scatterplot(x = 'cellID',  y = 'ratioFluctuations5min', data=summaryDf, style = parameter, hue = 'activationType', s = 100)
        ax6_0.set_xlabel('cellID')
        ax6_0.set_ylabel('After/Before Ratio Fluctuations (5 mins before/after)')
        # confInt = addStat_df(ax6, summaryDf, [('Before', 'After')], 'fluctuations', test = 'pairwise', cond = 'activationTag')
        # plt.suptitle("Mean - 1.96*StdError: " + str(confInt))
        ax6_0.set_xticklabels(summaryDf['cellID'].values, rotation = 90)
        ax6_0.axhline(y = 1.0, color = 'r')
        ax6_0.set_ylim(0,4)
        plt.savefig(todayFigDir+'/Summary_RatioFluctuations5mins'+parameter+'.png')
        plt.show()
        
        #### Ratio of fluctuations after/before activation
        ax6 = plt.figure()
        ax6 = sns.scatterplot(x = 'cellID',  y = 'ratioFluctuations', data=summaryDf, style = parameter, hue = 'activationType', s = 100)
        ax6.set_xlabel('cellID')
        ax6.set_ylabel('After/Before Ratio Fluctuations')
        # confInt = addStat_df(ax6, summaryDf, [('Before', 'After')], 'fluctuations', test = 'pairwise', cond = 'activationTag')
        # plt.suptitle("Mean - 1.96*StdError: " + str(confInt))
        ax6.set_xticklabels(summaryDf['cellID'].values, rotation = 90)
        ax6.axhline(y = 1.0, color = 'r')
        ax6.set_ylim(0,4)
        plt.savefig(todayFigDir+'/Summary_RatioFluctuations'+parameter+'.png')
        plt.show()
        
        #### Ratio of fluctuation/Ratio of Thickness Before/After
        ax7 = plt.figure()
        ax7 = sns.scatterplot(x = 'cellID',  y = 'ratioFluctThick', data=summaryDf, style = parameter, hue = 'activationType', s = 100)
        ax7.set_xlabel('cellID')
        ax7.set_ylabel('After/Before Ratio Fluctuations/Ratio Thickness')
        # confInt = addStat_df(ax7, summaryDf, [('Before', 'After')], 'ratioFluctThick', test = 'pairwise', cond = 'activationTag')
        # plt.suptitle("Mean - 1.96*StdError: " + str(confInt))
        ax7.set_xticklabels(summaryDf['cellID'].values, rotation = 90)
        ax7.axhline(y = 1.0, color = 'r')
        ax7.set_ylim(0,4)
        plt.savefig(todayFigDir+'/Summary_RatioFluctThickness'+parameter+'.png')
        plt.show()
        
        #### Ratio of fluctuation/Ratio of Thickness Before/After
        ax7b = plt.figure()
        ax7b = sns.scatterplot(x = 'ratioThickness',  y = 'ratioFluctuations', data=summaryDf, style = parameter, hue = 'activationType', s = 100)
        ax7b.set_xlabel('ratioThickness')
        ax7b.set_ylabel('Ratio Fluctuations/Ratio Thickness')
        # confInt = addStat_df(ax7, summaryDf, [('Before', 'After')], 'ratioFluctThick', test = 'pairwise', cond = 'activationTag')
        # plt.suptitle("Mean - 1.96*StdError: " + str(confInt))
        plt.savefig(todayFigDir+'/Summary_RatioFluctvsThickness'+parameter+'.png')
        plt.show()
        
        #### Median Thickness After vs. Median Thickness Before 
        ax8 = plt.figure()
        x = summaryDf['medianThickness'][summaryDf['activationTag'] == 'Before'].values
        y = summaryDf['medianThickness'][summaryDf['activationTag'] == 'After'].values
        hue = summaryDf['activationType'][summaryDf['activationTag'] == 'After'].values
        style = summaryDf[parameter][summaryDf['activationTag'] == 'After'].values
        ax8 = sns.scatterplot(x = x, y = y, hue = hue, style = style, s = 100)
        ax8.set_xlabel('medianThicknessBefore (um)')
        ax8.set_ylabel('medianThicknessAfter (um)')
        plt.savefig(todayFigDir+'/Summary_MedianThicknessBefore-After'+parameter+'.png')
        plt.legend()
        plt.show()
        
          #### Fluctuations Before vs. Fluctuation After
        ax9 = plt.figure()
        x = summaryDf['fluctuations'][summaryDf['activationTag'] == 'Before'].values
        y = summaryDf['fluctuations'][summaryDf['activationTag'] == 'After'].values
        hue = summaryDf['activationType'][summaryDf['activationTag'] == 'After'].values
        style = summaryDf[parameter][summaryDf['activationTag'] == 'After'].values
        ax9 = sns.scatterplot(x = x, y = y, hue = hue, style = style, s = 100)
        ax9.set_xlabel('fluctuationsBefore (um)')
        ax9.set_ylabel('fluctuationsAfter (um)')
        plt.savefig(todayFigDir+'/Summary_FluctuationsBefore-After_'+parameter+'.png')
        plt.legend()
        plt.show()
    
        cellConditionsDf.to_excel(todayFigDir+"/cellConditions_Ct"+kind+".xlsx")  
    plt.close('all')
    return(summaryDf)


# %% Constant field plots
#%%% Plotting summary of thickness plots
kind = 'none'
parameter = ['all'] # 'phenotype' # # ['all']
summaryDf = ctFieldThicknessSummary(experimentalDataDir, todayFigDir, parameter, plot = 2, kind = kind)

 #%%%%% Plotting summary of thickness plots if you want cell specific curves for different conditions

cellConditionsDf = pd.read_csv(experimentalDataDir+'/cellConditions_Ct.csv')
cellIDs = cellConditionsDf['cellID'][cellConditionsDf['excluded'] == 'no']

kind = 'none'
for i in cellIDs:
    date = ufun.findInfosInFileName(i,'date')
    cellID = ufun.findInfosInFileName(i,'cellID')[-5:]
    parameter =  [date, cellID] # ['none']
    summaryDf = ctFieldThicknessSummary(experimentalDataDir, todayFigDir, parameter, plot = 1, kind = kind)


# %% Close all open plots
plt.close('all')

# %% Plotting all three plots (3D, 2D, Dz vs Time) of an experiment
date = '22.06.09'
ctFieldThicknessIndividual(experimentalDataDir, todayFigDir, date, save = True, background = 'dark')


# %%
# plt.close('all')

#%%% Plotting 3D trajectories of all cells

tag = 'global'
param_type = 'none'
background = 'dark'
# param = 'phenotype'
ctFieldThicknessAll(experimentalDataDir, todayFigDir, date, param_type = param_type, \
                    tag = tag, background = background, save = True)

# %%
# plt.close('all')

#%%

cellID1 = '22-03-22_M3_P1_C5'
cellID2 = '22-03-22_M4_P1_C5'

timeSeriesDf1 = getCellTimeSeriesData(cellID1)
timeSeriesDf2 = getCellTimeSeriesData(cellID2)

fig = plt.figure(figsize=(25,10))

plt.plot(timeSeriesDf1['T']*1000/60, 1000*(timeSeriesDf1['D3'] - bead_dia), color = 'orange', label = 'Activation at from beads')
plt.plot(timeSeriesDf2['T']*1000/60, 1000*(timeSeriesDf2['D3'] - bead_dia), color = 'blue', label = 'Activation away from beads')
plt.legend(prop={'size':25})
plt.xlabel('Time (mins)', fontsize=35)
plt.ylabel('Cortex thickness (nm)', fontsize=35)
plt.yticks(fontsize=35)
plt.xticks(fontsize=35)
plt.title('3D Distance (nm) vs. Time (mins) before and after activation', fontsize=25)
plt.axvline(x=5, color = 'red', linewidth = 4)
for i in range(23):
    plt.axvline(x= 5.5 + i*((30/60)), color = 'blue', ymax = 0.05, linewidth = 4)
plt.rcParams['font.size'] = '30'
plt.tight_layout()
plt.savefig('C:/Users/anumi/OneDrive/Desktop/ImagesForIsrael/Thickness.jpg')
plt.show()

#%%
cellID1 = '22-03-22_M3_P1_C5'

timeSeriesDf1 = getCellTimeSeriesData(cellID1)

fig = plt.figure(figsize=(25,10))

plt.plot(timeSeriesDf1['T']*1000/60, 1000*(timeSeriesDf1['D3'] - bead_dia), color = 'green', label = 'Global activation')
plt.legend(prop={'size':25})
plt.xlabel('Time (mins)', fontsize=35)
plt.ylabel('Cortex thickness (nm)', fontsize=35)
plt.yticks(fontsize=35)
plt.xticks(fontsize=35)
plt.title('3D Distance (nm) vs. Time (mins) before and after activation', fontsize=25)
plt.axvline(x=5, color = 'red', linewidth = 4)
for i in range(23):
    plt.axvline(x= 5.5 + i*((30/60)), color = 'blue', ymax = 0.05, linewidth = 4)
plt.rcParams['font.size'] = '30'
plt.tight_layout()
plt.savefig('C:/Users/anumi/OneDrive/Desktop/ImagesForIsrael/Thickness_global.jpg')
plt.show()