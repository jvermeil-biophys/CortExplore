# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 18:24:50 2024

@author: anumi
"""

import pandas as pd

df1 = pd.read_csv('D:/Anumita/MagneticPincherData/Data_Analysis/MecaData_CellTypes.csv')
df2 = pd.read_csv('D:/Anumita/MagneticPincherData/Data_Analysis/VWCAnalysis_MDCK_24-07-03_JosephSettings.csv')

merged = df1.merge(df2, how = 'outer')

merged.to_csv('D:/Anumita/MagneticPincherData/Collabs-Internships/JosephMecaData_CellTypes_JV-AJ.csv', index=False)