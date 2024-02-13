# %% > Imports and constants

#### Main imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



import os
import sys
import time
import random
import warnings
import itertools
import matplotlib

from copy import copy
from cycler import cycler
from datetime import date
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)

import GraphicStyles as gs
import UtilityFunctions as ufun
import TrackAnalyser_V2 as taka

#### Potentially useful lines of code
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# cp.DirDataFigToday

#### Pandas
pd.set_option('display.max_columns', None)
# pd.reset_option('display.max_columns')
pd.set_option('display.max_rows', None)
pd.reset_option('display.max_rows')


####  Matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})

#### Graphic options
gs.set_default_options_jv()

#### Bokeh
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Range1d
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10
from bokeh.layouts import gridplot
output_notebook()

#### Markers
my_default_marker_list = ['o', 's', 'D', '>', '^', 'P', 'X', '<', 'v', 'p']
markerList10 = ['o', 's', 'D', '>', '^', 'P', 'X', '<', 'v', 'p']

todayFigDir = cp.DirDataFigToday
experimentalDataDir = cp.DirRepoExp

plotLabels = 25
plotTicks = 18 
plotTitle = 25
plotLegend = 25
fontColour = '#000000'


#%%

allCells = os.listdir(cp.DirDataRaw + '/24.01.25')
beadDia = 4.510
rawTrajs = os.listdir(cp.DirDataTimeseriesTraj)

rawTrajs = np.asarray([i for i in rawTrajs if '24-01-25' in i])

allCells = ['24-01-25_M1_P4_C6_disc20um_thickness.tif']

for i in allCells:
    if '.tif' in i:  
        # try:
        fig, ax = plt.subplots(2,1)
        i = i.split('.tif')
        i = i[0] 
        tsdf = pd.read_csv(cp.DirDataTimeseries + '/' + i + '_PY.csv', sep = ';')
        
        
        for j in rawTrajs:
            try:
                if i in j and 'In' in j:
                    rawTraj = pd.read_csv(os.path.join(cp.DirDataTimeseriesTraj, j), sep ='\t')
            except:
                pass
        
        D3 = tsdf['D3'] - beadDia 
        
        D2 = tsdf['D2'] - beadDia 
        
        ax[0].plot(tsdf['T'], D3, label = 'D3', color = '#000000')
        ax[0].plot(tsdf['T'], D2, label = 'D2', color = '#8e8e8e')
        
        ax2 = ax[0].twinx()
        
        if 'M1' in i:
            ax2.plot(tsdf['T'], rawTraj['Y'], label = 'X Position', color = '#3A75D4')
        elif 'M2' in i:
            ax2.plot(tsdf['T'], rawTraj['Y'], label = 'X Position', color = '#3A75D4')
        
        ax[1].plot(tsdf['T'], tsdf['dz'])
        
        
        fig.suptitle(i)
        ax[0].set_ylim(0, 0.8)
        ax[0].set_ylabel('Thickness (nm)')
        ax[1].set_ylim(-3, 3)
        ax[1].set_ylabel('dz')
        ax[0].legend()
        
        # 
        fig.savefig('G:/CortexMeetings/CortexMeeting_24-02-06/Plots_MigratingCells/' + i + '.png')
        plt.show()
        # except:
        #     print(i)
        #     pass