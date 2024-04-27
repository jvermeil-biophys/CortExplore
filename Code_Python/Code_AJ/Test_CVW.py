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

#%%

path = 'D:/Anumita/MagneticPincherData/Data_TimeSeries'
file = '24-02-27_M1_P1_C1_L70_CoilswMagnets24mT_PY.csv'

fh = pd.read_csv(os.path.join(path, file), sep = ';')

#%% Functions

def fitCvw(x, k, y, h0):
    f = np.pi*2250*(k/2*(h0**3*x**-2+2*x*h0)+y/3*h0*(1-x/h0)**2)
    return f

#%%

fh = fh[fh['idxAnalysis'] == 1]

f = fh['F']
x = fh['D3'].values

initH0 = max(x) + 100 
initE = 1 # E ~ 3*H0*F_max / pi*R*(H0-h_min)²

initialParameters = [initE, initH0]

# bounds on parameters - initial parameters must be within these
lowerBounds = (0, 0)
upperBounds = (np.Inf, np.Inf)
parameterBounds = [lowerBounds, upperBounds]

params, covM = curve_fit(fitCvw, x, f, p0 = None, bounds=((0, 0, x[0]), (1e3, 1e6, 2*x[0])))

# plt.plot(x, f, color = 'k')

f_fit = fitCvw(x, params[0], params[1], params[2])

plt.plot(x, f_fit)

plt.show()