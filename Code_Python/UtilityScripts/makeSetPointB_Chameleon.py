# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:49:06 2024

@author: JosephVermeil
"""

# %% (0) Imports and settings

# 1. Imports
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import statsmodels.api as sm
import matplotlib.pyplot as plt

rawDir = "D:/MagneticPincherData/Raw/24.06.14_Chameleon/"

#%% Write a txt file for ramps and compressions

def sigmoid(x):
  return 1 / (1 + np.exp(-6*x))

freq = 10000 #in Hz
tRest = 0.5
tConst = 2 # in secs
tComp = 1.5 #in sec
factor = 1000
constfield = 5 # mT
lowfield = 2.0
highfield = 50

xRelease = np.linspace(-1, 1, freq*1)
yRelease = (1 - sigmoid(xRelease))*(constfield-lowfield) + lowfield
constRelease = np.asarray(freq*tConst*[lowfield])

t_comp = np.linspace(0, tComp, int(freq*tComp))
t_relax = np.linspace(0, tComp, int(freq*tComp))
restArray = np.asarray(int(freq*tRest)*[constfield])
comp = lowfield + (highfield-lowfield) * (t_comp/tComp)**2
relaxArray = constfield - (constfield-highfield) * ((max(t_comp) - t_comp)/tComp)**2
wholeArray = []

wholeArray.extend(restArray*factor)
wholeArray.extend(yRelease*factor)
wholeArray.extend(constRelease*factor)
wholeArray.extend(comp*factor)
wholeArray.extend(relaxArray*factor)
wholeArray.extend(restArray*factor)


np.savetxt(rawDir + "/5mT_2.0-50_t2_1.5s.txt", wholeArray, fmt='%i')

plt.plot(wholeArray)
plt.show()