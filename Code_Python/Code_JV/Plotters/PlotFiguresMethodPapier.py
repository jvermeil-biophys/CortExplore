# -*- coding: utf-8 -*-
"""
Created on Fri May 19 19:20:55 2023

@author: JosephVermeil
"""

# %% > Imports and constants

#### Main imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt

import os
import re
import sys
import time
import random
import numbers
import warnings
import itertools
import matplotlib

from cycler import cycler
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
# from matplotlib.gridspec import GridSpec

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)

import GraphicStyles as gs
import UtilityFunctions as ufun
import TrackAnalyser as taka
import TrackAnalyser_V2 as taka2


# %%
plt.rcParams['text.usetex'] = False


T = np.array([-150, 350, 400, 450, 950 ,1000, 1050, 1550, 1600, 1650, 2150])
Z = np.array([1.5, 0.5, 1.0, 1.5, 0.5, 1.0, 1.5, 0.5, 1.0, 1.5, 0.5])

fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.plot(T, Z, c = 'lightblue', lw = 1, ls='-')
ax.plot(T, Z, marker = 'P', c = 'dodgerblue', lw = 0, label = 'Image acquisition')

ax.set_xticks(np.array([400, 950 ,1000, 1050, 1600]))
ax.set_yticks(np.array([0.5, 1.0, 1.5]))

ax.set_xticklabels([r'$t_i-600$', r'$t_i-50$' , r'$t_i$', r'$t_i+50$', r'$t_i+600$'], fontsize = 9, rotation = 45)
ax.set_yticklabels([r'$z_0-0.5$' , r'$z_0$', r'$z_0+0.5$'], fontsize = 9)

ax.set_xlabel('Time (ms)', fontsize = 11)
ax.set_ylabel('Focal plane (µm)', fontsize = 11)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlim([250, 1750])
ax.set_ylim([0, 2.0])

ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(250, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

ax.legend()

plt.tight_layout()

plt.show()


# %%

path = 'C://Users//JosephVermeil//Desktop//MethodPaper\Figures//ExempleCell_21-04-23-M2_P1_C3//21-04-23_M2_P1_C3_thickness5mT_disc20um_wFluo_PY.csv'
gs.set_manuscript_options_jv()

df = pd.read_csv(path, sep=';')

df['h'] = (df['D3'] - 4.503) * 1000
median = np.median(df['h'])
d1, d9 = np.percentile(df['h'], 10), np.percentile(df['h'], 90)


fig, ax = plt.subplots(1, 1, figsize = (10/gs.cm_in, 6/gs.cm_in))

c1, c2, c3 = 'royalblue', 'red', 'darkorange'

ax.plot(df['T'], df['h'], lw = 1.5, color = c1) 
# ax.plot(df['T'], df['h'], lw = 0, marker = 'o', markersize = 1)

ax.axhline(y=median, color = c2, ls = '--', lw = 1.5, zorder = 4, label = 'Median')
ax.axhline(y=d1, color = c3, ls = '--', lw = 1, label = '1st & 9th deciles')
ax.axhline(y=d9, color = c3, ls = '--', lw = 1)

# , label = 'Median Thickness: {:.0f} nm'.format(median)

# ax.text(x = 340, y = 310, s = 'Median\nThickness: {:.0f} nm'.format(median),
#         va = 'bottom', ha='center', fontsize = 8, color=c2)

x_arrow = 140
# ax.annotate('Fluctuations\namplitude: {:.0f} nm'.format(d9-d1), xy=(x_arrow, d1*0.95), xytext=(x_arrow, d9*1.05), 
#             va = 'bottom', ha='center', fontsize = 8, color = c3,
#             arrowprops=dict(arrowstyle='<->', color = c3, lw = 2))
# ax.annotate('', xy=(x_arrow, d1*0.95), xytext=(x_arrow, d9*1.05), 
#             va = 'bottom', ha='center', fontsize = 8, color = c3,
#             arrowprops=dict(arrowstyle='<->', color = c3, lw = 2))


# ax.set_xlim([0, 900])
ax.set_ylim([0, 1000])
ax.set_xlabel('Time (s)')
ax.set_ylabel('$H_{5mT}$ (nm)')

ax.set_yticks(np.array([0, 200, 400 ,600, 800]))
ax.set_yticklabels(['0', '200', '400' ,'600', '800'])
ax.set_xticklabels(ax.get_xticklabels())

ax.legend(loc = 'lower right')
figDir = "D:/MagneticPincherData/Figures/PhysicsDataset"
figSubDir = 'ResultsTechnique'
name = 'Example_ConstantField'
ufun.archiveFig(fig, name = name, ext = '.pdf', dpi = 100,
                figDir = figDir, figSubDir = figSubDir, cloudSave = 'flexible')

plt.tight_layout()


# %%