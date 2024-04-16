# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:44:54 2023

@author: JosephVermeil
"""

# %%

import os
import pandas as pd
import numpy as np
import UtilityFunctions as ufun

# %%

target_dir = "D://MagneticPincherData//Data_Timeseries//Trajectories_raw"
list_files = os.listdir(target_dir)
for i in range(len(list_files)):
    f = list_files[i]
    f_path = os.path.join(target_dir, f)
    try:
        df = pd.read_csv(f_path, sep=None, engine='python')
    except:
        pass
    try:
        nL = np.max(df.iL) # idxLoop
        for iL in range(1, nL+1):
            df_i = df[df.iL == iL]
            first_action = ufun.findFirst(-iL, df_i.idxAnalysis.values)
            last_actionMain = ufun.findLast(iL, df_i.idxAnalysis.values)
            first_passivePart2 = ufun.findFirst(0, df_i.idxAnalysis.values[first_action:]) + first_action
            first_action += df_i.index.values[0]
            last_actionMain += df_i.index.values[0]
            first_passivePart2 += df_i.index.values[0]
            # print(last_actionMain, first_passivePart2)
            if not last_actionMain+1 == first_passivePart2:
                # print(df_i.idxAnalysis.values[last_actionMain+1:first_passivePart2])
                df_i.loc[last_actionMain+1:first_passivePart2, 'idxAnalysis'] *= (-1)
            df[df.iL == iL] = df_i
        df.to_csv(f_path, sep='\t', index=False)
        print(f)
    except:
        pass