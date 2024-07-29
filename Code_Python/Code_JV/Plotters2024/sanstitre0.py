# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:57:05 2024

@author: BioMecaCell
"""

import matplotlib.pyplot as plt
import seaborn as sns

from statannotations.Annotator import Annotator

df = sns.load_dataset("tips")
x = "day"
y = "total_bill"
order = ['Sun', 'Thur', 'Fri', 'Sat']

plot_parms = {'data':df, 
              'x':x, 
              'y':y, 
              'order':order}

ax = sns.boxplot(**plot_parms)
ax = sns.swarmplot(**plot_parms)

pairs=[("Thur", "Fri"), ("Thur", "Sat"), ("Fri", "Sun")]

annotator = Annotator(ax, pairs, **plot_parms)
annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
annotator.apply_and_annotate()

plt.tight_layout()
plt.show()