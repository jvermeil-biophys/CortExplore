# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:32:12 2023

@author: JosephVermeil
"""

import os
import re
import sys


def makeFolderTree(startpath):
    size = 0
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        if level == 0:
            print('Folder size {:.3f} Go\n'.format(size/1e9))
            size = 0
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
        if level >= 0:
            for f in files:
                fp = os.path.join(root, f)
                size += os.path.getsize(fp)
        if level == 1:
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            # subindent = ' ' * 4 * (level + 1)
            # for f in files:
            #     print('{}{}'.format(subindent, f))
    print('Folder size {:.3f} Go\n'.format(size/1e9))
    
rootDir = "E://"
# rootDir = "E://2021-2022_AnalyzedData_JV_Backup2023-02-22//Figures//"

makeFolderTree(rootDir) 