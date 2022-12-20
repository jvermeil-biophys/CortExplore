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
                renamedListTarget.append(newFileName)
                if not test:
                    new_path = os.path.join(path,newFileName)
                    if not os.path.isfile(new_path):
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

# %% Script Dates
        
inverseDate('D://MagneticPincherData//Raw_DC//Raw_DC_JV//', target = 'all', 
            test = True, recursiveAction = True, exceptStrings = ['Deptho'])

path0 = 'D://MagneticPincherData//Raw//'

# %% Script Other renaming

date = '18.09.24'
path = path0 + date

findAndRename(path, 'M2_P1', 'M3_P1', 
              target='file', test = False, recursiveAction = False, exceptStrings = [])
findAndRename(path, 'M1_P2', 'M2_P1', 
              target='file', test = False, recursiveAction = False, exceptStrings = [])



date = '18.09.25'
path = path0 + date

findAndRename(path, 'M2_P2', 'M4_P1', 
              target='file', test = False, recursiveAction = False, exceptStrings = [])
findAndRename(path, 'M2_P1', 'M3_P1', 
              target='file', test = False, recursiveAction = False, exceptStrings = [])
findAndRename(path, 'M1_P2', 'M2_P1', 
              target='file', test = False, recursiveAction = False, exceptStrings = [])




date = '18.10.30'
path = path0 + date

findAndRename(path, 'M2_P2', 'M4_P1', 
              target='file', test = False, recursiveAction = False, exceptStrings = [])
findAndRename(path, 'M2_P1', 'M3_P1', 
              target='file', test = False, recursiveAction = False, exceptStrings = [])
findAndRename(path, 'M1_P2', 'M2_P1', 
              target='file', test = False, recursiveAction = False, exceptStrings = [])









