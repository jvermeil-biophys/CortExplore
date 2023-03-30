# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:47:48 2023

@author: anumi
"""

import numpy as np
import pandas as pd

D = {'subject': ['physics', 'hindi', 'english', 'sport', 
                 'physics', 'hindi', 'english', 'sport', 
                 'physics', 'hindi', 'english', 'sport',
                 'physics', 'hindi', 'english', 'sport'],
     'teacher': ['Mrs B.', 'Mr A.', 'Mr A.', 'Mrs C.', 
                'Mrs B.', 'Mr A.', 'Mr A.', 'Mrs C.', 
                'Mrs B.', 'Mr A.', 'Mr A.', 'Mrs C.',
                'Mrs B.', 'Mr A.', 'Mr A.', 'Mrs C.'],
     'grades': [89, 58, 77, 82, 95, 62, 80, 99, 
                59, 85, 97, 47, 98, 65, 79, 82],
     'coefficient':[1,1,1,1,2,3,2,3,6,5,7,6,8,7,9,10]}

df = pd.DataFrame(D)

group_by_subject = df.groupby(by = 'subject')

df_average = group_by_subject.agg({'grades':['mean', 'std', 'count', 'sum'], 
                                   'teacher':'first'})