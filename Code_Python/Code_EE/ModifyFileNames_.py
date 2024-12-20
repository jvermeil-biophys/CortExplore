# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:12:49 2022

@author: JosephVermeil
"""

import os
import re
import pandas as pd
dateFormat1 = re.compile('\d{2}-\d{2}-\d{2}')
#os.rename(r'file path\OLD file name.file type',r'file path\NEW file name.file type')


# %% Functions


def inverseDate(path, target='file', test = True, recursiveAction = False, exceptStrings = []):
    listAll = os.listdir(path)
    listFiles = []
    listDir = []
    listTarget = []
    for f in listAll:
        if os.path.isfile(os.path.join(path,f)):
            listFiles.append(f)
        elif os.path.isdir(os.path.join(path,f)):
            listDir.append(f)
    if target == 'file':
        listTarget = listFiles
    elif target == 'dir':
        listTarget = listDir
    elif target == 'all':
        listTarget = listAll
    renamedListTarget = []
    for f in listTarget:
        searchDate = re.search(dateFormat1, f)
        if searchDate:
            doExcept = False
            for s in exceptStrings:
                if s in f:
                    doExcept = True
                    print('Exception for ' + os.path.join(path,f))
            if not doExcept:
                foundDate = f[searchDate.start():searchDate.end()]
                newDate = foundDate[-2:] + foundDate[2:-2] + foundDate[:2]
                newFileName = f[:searchDate.start()] + newDate + f[searchDate.end():]
                if newDate[:2] not in [str(i) for i in range(18, 23)]:
                    print("newDate = " + newDate)
                    print('error detected in : ' + os.path.join(path,f))
                else:
                    renamedListTarget.append(newFileName)
                if not test:
                    os.rename(r''+os.path.join(path,f),r''+os.path.join(path,newFileName))
                    
    if recursiveAction:
        # Update ListDir after potential renaming
        listAll = os.listdir(path)
        listDir = []
        for f in listAll:
            if os.path.isdir(os.path.join(path,f)):
                listDir.append(f)
        # Start going recursive
        for d in listDir:
            print("Let's go into " + os.path.join(path,d))
            inverseDate(os.path.join(path,d), target, test = test, recursiveAction = True, exceptStrings = exceptStrings)
    print(renamedListTarget)
    
    
def inverseDateInsideFile(path, test = True):
    f = open(path, 'r')
    data = f.read()
    searchedDates = re.findall(dateFormat1, data)
    for date in searchedDates:
        foundDate = date
        newDate = foundDate[-2:] + foundDate[2:-2] + foundDate[:2]
        data = data.replace(foundDate, newDate)
        if newDate[:2] != '20' and newDate[:2] != '21':
            print('error detected with date : ' + foundDate)
    f.close()
    print(data)
    if not test:
        f = open(path, 'w')
        f.write(data)
        f.close()

def findAndRename(path, target_string, new_string, target='file', test = True, recursiveAction = False, exceptStrings = []):
    listAll = os.listdir(path)
    listFiles = []
    listDir = []
    listTarget = []
    for f in listAll:
        if os.path.isfile(os.path.join(path,f)):
            listFiles.append(f)
        elif os.path.isdir(os.path.join(path,f)):
            listDir.append(f)
    if target == 'file':
        listTarget = listFiles
    elif target == 'dir':
        listTarget = listDir
    elif target == 'all':
        listTarget = listAll
    renamedListTarget = []
    for f in listTarget:
        searchString = re.search(target_string, f)
        if searchString:
            doExcept = False
            for s in exceptStrings:
                if s in f:
                    doExcept = True
                    print('Exception for ' + os.path.join(path,f))
            if not doExcept:
                foundString = f[searchString.start():searchString.end()]
                newFileName = f[:searchString.start()] + new_string + f[searchString.end():]
                #newFileName = f[:searchString.start()] + foundString[:2] + f[searchString.end():]
                renamedListTarget.append(newFileName)
                if not test:
                    new_path = os.path.join(path, newFileName)
                    if not os.path.isfile(new_path):
                        os.rename(r''+os.path.join(path, f),r''+os.path.join(path, newFileName))
                        
                # else:
                #     print(f)
                    
    if recursiveAction:
        # Update ListDir after potential renaming
        listAll = os.listdir(path)
        listDir = []
        for f in listAll:
            if os.path.isdir(os.path.join(path,f)):
                listDir.append(f)
        # Start going recursive
        for d in listDir:
            print("Let's go into " + os.path.join(path,d))
            rlT = findAndRename(os.path.join(path,d), target_string, new_string, 
                          target=target, test = test, recursiveAction = True, exceptStrings = exceptStrings)
            renamedListTarget += rlT
    
    print(renamedListTarget)
    print(len(renamedListTarget))
    return(renamedListTarget)

# %% Script Dates
        
inverseDate('D://MagneticPincherData//Raw_DC//Raw_DC_JV//', target = 'all', 
            test = True, recursiveAction = True, exceptStrings = ['Deptho'])

path0 = 'D://MagneticPincherData//Raw//'

# %% Script Other renaming

path0 = 'E:\\2023-04-28_3T3atcc2023_CK666\\'
sub = 'M2_CK666_100uM'
path = path0 + sub

findAndRename(path, '_M1_', '_M2_', 
              target='all', test = True, recursiveAction = True, exceptStrings = [])


path0 = 'D://MagneticPincherData//Raw//'
date = '23.03.09'
path = path0 + date

findAndRename(path, '23.03.09', '23-03-09', 
              target='file', test = True, recursiveAction = False, exceptStrings = [])



date = '18.09.24'
path = path0 + date

findAndRename(path, 'M2_P1', 'M3_P1', 
              target='file', test = True, recursiveAction = False, exceptStrings = [])
findAndRename(path, 'M1_P2', 'M2_P1', 
              target='file', test = True, recursiveAction = False, exceptStrings = [])



date = '18.09.25'
path = path0 + date

findAndRename(path, 'M2_P2', 'M4_P1', 
              target='file', test = True, recursiveAction = False, exceptStrings = [])
findAndRename(path, 'M2_P1', 'M3_P1', 
              target='file', test = True, recursiveAction = False, exceptStrings = [])
findAndRename(path, 'M1_P2', 'M2_P1', 
              target='file', test = True, recursiveAction = False, exceptStrings = [])




date = '18.10.30'
path = path0 + date

findAndRename(path, 'M2_P2', 'M4_P1', 
              target='file', test = True, recursiveAction = False, exceptStrings = [])
findAndRename(path, 'M2_P1', 'M3_P1', 
              target='file', test = True, recursiveAction = False, exceptStrings = [])
findAndRename(path, 'M1_P2', 'M2_P1', 
              target='file', test = True, recursiveAction = False, exceptStrings = [])


# %% Script Other renaming

path0 = 'E://23-09-06_3T3Atcc-LaGFP_CalA//M1_0.25nM'
s1 = '-07-20_'
s2 = '-09-06_'
findAndRename(path0, s1, s2, 
              target='all', test = True, recursiveAction = True, exceptStrings = [])




# %% Script Other renaming

path0 = 'D://MagneticPincherData//Raw//23.11.13'
s1 = '2311-01'
s2 = '23-11-13'
findAndRename(path0, s1, s2, 
              target='all', test = True, recursiveAction = True, exceptStrings = [])

# %% Script Other renaming

path0 = 'D://MagneticPincherData//Raw//23.12.07'
s1 = r'C\d-\d'
s2 = 'Ctest'
findAndRename(path0, s1, s2, 
              target='all', test = True, recursiveAction = True, exceptStrings = [])


# %% Script Other renaming

path0 = 'D://MagneticPincherData//Raw//24.02.28'
s1 = r'C\d\d'
s2 = 'Ctest'
findAndRename(path0, s1, s2, 
              target='all', test = True, recursiveAction = True, exceptStrings = [])

#%%
path0 = 'D:/Eloise/MagneticPincherData/Raw/24.05.23'
s1 = 'C4_d'
s2 = 'C4_blurred_d'
findAndRename(path0, s1, s2,
              target='all', test = False, recursiveAction = False, exceptStrings = [])

